"""CUDA-isolated inference worker for the ComfyUI H3 SoL upscaler node."""

from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

# fixed_conditioning imports Torch transitively: normalize before that import,
# not inside run(), when Torch's allocator would already be initialized.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimax_h3_sol_upscaler.fixed_conditioning import (
    ARTIFACT_ENV,
    conditioning_cache_path,
    conditioning_identity,
    prepare_fixed_conditioning,
)
from minimax_h3_sol_upscaler.model_runtime import (
    _RUNTIME as MODEL_RUNTIME,
)
from minimax_h3_sol_upscaler.model_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    VIDEO_ONLY_REFINER_ENV,
    WORKER_TORCH_VERSION,
)
from minimax_h3_sol_upscaler.node import (
    ASYNC_NVENC_STAGING_ENV,
    GPU_AMINMAX_ENV,
    _experiment_flag,
    _has_audio,
    _load_pipeline,
    _nvenc_preset,
    _write_frames,
    natten_backend_name,
    pad_video_frames,
    validate_runtime,
)
from minimax_h3_sol_upscaler.protocol import (
    WorkerRequest,
    parse_worker_request,
)


def run(request: WorkerRequest) -> dict[str, object]:
    import torch
    from diffusers.utils import load_video

    validate_runtime(torch, sys.version_info[:2], expected_torch=WORKER_TORCH_VERSION)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SoL refiner worker requires a BF16-capable CUDA GPU")
    decoder_mode = request.get("decoder_mode", "diffusion")
    if decoder_mode == "diffusion" and natten_backend_name() != "libnatten":
        raise RuntimeError("Native NATTEN is required; refusing the OOM-prone Flex attention backend")

    source = Path(request["source"]).resolve()
    output = Path(request["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(request["width"]), int(request["height"])
    started = time.perf_counter()
    load_started = time.perf_counter()
    artifact_setting = os.environ.get(ARTIFACT_ENV, "").strip()
    conditioning_artifact = Path(artifact_setting).expanduser().resolve() if artifact_setting else None
    if conditioning_artifact is None and request.get("cache_conditioning", False):
        conditioning_artifact = conditioning_cache_path(conditioning_identity(
            str(request["prompt"]), MODEL_ID, MODEL_REVISION,
            _experiment_flag(VIDEO_ONLY_REFINER_ENV),
        ))
    pipe = _load_pipeline(
        MODEL_ID, expected_torch=WORKER_TORCH_VERSION, decoder_mode=decoder_mode,
        transformer_precision=request.get("transformer_precision", "nvfp4"),
        width=width, height=height,
        conditioning_artifact=conditioning_artifact,
        fixed_prompt=str(request["prompt"]) if conditioning_artifact is not None else None,
    )
    conditioning_kwargs = {}
    conditioning_started = time.perf_counter()
    if conditioning_artifact is not None:
        embeddings, mask = prepare_fixed_conditioning(
            pipe, conditioning_artifact,
            conditioning_identity(
                str(request["prompt"]), MODEL_ID, MODEL_REVISION, pipe.video_only_refiner
            ),
            str(request["prompt"]),
        )
        conditioning_kwargs = {"prompt_embeds": embeddings, "prompt_attention_mask": mask}
        conditioning_state = getattr(pipe, "fixed_conditioning", None)
        if conditioning_state is not None:
            MODEL_RUNTIME.bind_conditioning_artifact(
                conditioning_artifact, conditioning_state.sha256,
            )
    fixed_conditioning_prepare_s = time.perf_counter() - conditioning_started
    model_load_s = time.perf_counter() - load_started
    # At this resolution the complete spatial field fits safely on a 6000
    # Pro and avoids duplicated overlap work. Keep reference tiles at 1K,
    # where a 1024px tile leaves little VRAM headroom in ComfyUI.
    tile_size = int(request.get("tile_size") or (
        1024 if max(width, height) <= 832 else 768
    ))
    tile_stride = int(request.get("tile_stride") or (768 if tile_size == 1024 else 512))
    if pipe.diffusion_decoder is not None:
        pipe.diffusion_decoder.enable_tiling(
            tile_sample_min_height=tile_size,
            tile_sample_min_width=tile_size,
            tile_sample_min_num_frames=128,
            tile_sample_stride_height=tile_stride,
            tile_sample_stride_width=tile_stride,
            tile_sample_stride_num_frames=80,
        )
    frames = load_video(str(source))
    if not frames:
        raise ValueError("input video contains no decodable frames")
    input_count = len(frames)
    fps = Fraction(request["fps"])
    padded = pad_video_frames(frames)
    has_audio = _has_audio(str(source))
    execution_device = pipe._execution_device
    torch.cuda.reset_peak_memory_stats()
    infer_started = time.perf_counter()
    result = pipe(
        padded,
        None if conditioning_artifact is not None else str(request["prompt"]),
        **conditioning_kwargs,
        width=int(request["width"]),
        height=int(request["height"]),
        frame_rate=float(fps),
        generator=torch.Generator(execution_device).manual_seed(int(request["seed"])),
        decoder_generator=torch.Generator(execution_device).manual_seed(int(request["decoder_seed"])),
        output_type="latent" if request.get("decoder_mode", "diffusion") == "pruna_fast" else "pt",
    ).frames
    if request.get("decoder_mode", "diffusion") == "pruna_fast":
        from minimax_h3_sol_upscaler.fast_decoder import (
            decode_pruna_vae,
            pruna_decoder_compiled,
            pruna_drop_encoder_enabled,
            pruna_encoder_removed,
            pruna_inference_mode_enabled,
            pruna_output_contiguous,
            pruna_output_copy_optimization_enabled,
        )

        pruna_tile_size = int(request.get("pruna_tile_size", 1024))
        pruna_tile_stride = int(request.get("pruna_tile_stride", 1024))
        pruna_contiguous_output = pruna_output_contiguous()
        result = decode_pruna_vae(
            result, width, height, torch,
            tile_size=pruna_tile_size, tile_stride=pruna_tile_stride,
            frame_count=input_count,
        )
        pruna_output_copy_optimized = pruna_output_copy_optimization_enabled()
        pruna_encoder_removed_parameters, pruna_encoder_removed_bytes = pruna_encoder_removed()
        pruna_encoder_removal_enabled = pruna_drop_encoder_enabled()
        pruna_inference_mode = pruna_inference_mode_enabled()
    else:
        result = result[0].permute(0, 2, 3, 1)
        pruna_output_copy_optimized = False
        pruna_encoder_removed_parameters = 0
        pruna_encoder_removed_bytes = 0
        pruna_encoder_removal_enabled = False
        pruna_inference_mode = False
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - infer_started
    if result.shape[0] < input_count:
        raise RuntimeError(f"refiner returned {result.shape[0]} frames for {input_count} input frames")
    result = result[:input_count]
    export = _write_frames(
        result, str(source), str(output), fps, str(request["output_backend"]), has_audio
    )
    torch.cuda.synchronize()
    return {
        "status": "succeeded",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_build": torch.version.git_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "sdp_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "natten_backend": natten_backend_name() if pipe.diffusion_decoder is not None else "not-used",
        "model_residency": pipe.model_residency,
        "cpu_offload_enabled": pipe.cpu_offload_enabled,
        "cpu_text_encoder_enabled": bool(getattr(pipe, "cpu_text_encoder_enabled", False)),
        "decoder_fused_rope_modules": pipe.decoder_fused_rope_modules,
        "transformer_compiled": bool(pipe.transformer_compiled),
        "transformer_precision": str(pipe.transformer_precision),
        "transformer_mxfp8_linear_count": len(pipe.transformer_mxfp8_linear_names),
        "transformer_nvfp4_linear_count": len(pipe.transformer_nvfp4_linear_names),
        "transformer_mxfp8_min_linear_dimension": (
            1024 if pipe.transformer_precision == "mxfp8" else None
        ),
        "video_only_refiner": bool(pipe.video_only_refiner),
        "video_only_transformer_blocks": int(pipe.video_only_transformer_blocks),
        "cpu_preprocess_overlap": bool(pipe.cpu_preprocess_overlap),
        "prompt_cache_enabled": bool(pipe.prompt_cache_enabled),
        "prompt_cache_hit": bool(pipe.last_prompt_cache_hit),
        "prompt_cache_entries": int(pipe._prompt_cache.entries),
        "prompt_cache_bytes": int(pipe._prompt_cache.bytes),
        "fixed_conditioning_enabled": conditioning_artifact is not None,
        "text_encoder_loaded": pipe.text_encoder is not None,
        "fixed_conditioning_prepare_s": fixed_conditioning_prepare_s,
        "fixed_conditioning_sha256": (
            pipe.fixed_conditioning.sha256 if conditioning_artifact is not None else None
        ),
        "fixed_conditioning_released_parameter_bytes": (
            pipe.fixed_conditioning.released_parameter_bytes if conditioning_artifact is not None else 0
        ),
        "async_nvenc_staging": _experiment_flag(ASYNC_NVENC_STAGING_ENV),
        "decoder_mode": decoder_mode,
        "pruna_output_contiguous": (
            pruna_contiguous_output if decoder_mode == "pruna_fast" else None
        ),
        "pruna_decoder_compiled": (
            pruna_decoder_compiled() if decoder_mode == "pruna_fast" else None
        ),
        "pruna_output_copy_optimized": pruna_output_copy_optimized,
        "pruna_encoder_removal_enabled": pruna_encoder_removal_enabled,
        "pruna_encoder_removed_parameters": pruna_encoder_removed_parameters,
        "pruna_encoder_removed_bytes": pruna_encoder_removed_bytes,
        "pruna_inference_mode": pruna_inference_mode,
        "gpu_aminmax": _experiment_flag(GPU_AMINMAX_ENV),
        "nvenc_preset": (
            _nvenc_preset() if request["output_backend"] == "nvenc" else None
        ),
        "pruna_tile_size": pruna_tile_size if decoder_mode == "pruna_fast" else None,
        "pruna_tile_stride": pruna_tile_stride if decoder_mode == "pruna_fast" else None,
        "allocator_config": os.environ["PYTORCH_ALLOC_CONF"],
        "python": sys.version.split()[0],
        "model_load_s": model_load_s,
        "inference_s": inference_s,
        "frames": input_count,
        "processed_frames": len(padded),
        "trimmed_padding_frames": len(padded) - input_count,
        "width": int(request["width"]),
        "height": int(request["height"]),
        "decoder_tile_size": tile_size if pipe.diffusion_decoder is not None else None,
        "decoder_tile_stride": tile_stride if pipe.diffusion_decoder is not None else None,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "duration_seconds": input_count / float(fps),
        "audio_preserved": has_audio,
        "seed": int(request["seed"]),
        "decoder_seed": int(request["decoder_seed"]),
        "prompt": str(request["prompt"]),
        "output": str(output),
        "export": export,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_seconds": time.perf_counter() - started,
    }


def serve() -> None:
    """Process requests sequentially while preserving the cached GPU pipeline."""
    import traceback
    from contextlib import redirect_stdout

    responses = sys.stdout
    for line in sys.stdin:
        try:
            request = parse_worker_request(json.loads(line))
            # Keep stdout exclusively for protocol responses, including when
            # upstream inference components print progress or diagnostics.
            with redirect_stdout(sys.stderr):
                report = run(request)
            output_path = Path(request.get("output", "")).resolve()
            if output_path.name:
                (output_path.parent / "worker-report.json").write_text(
                    json.dumps(report, sort_keys=True, indent=2) + "\n"
                )
            payload = json.dumps({"status": "succeeded", "report": report}, sort_keys=True)
        except Exception:  # noqa: BLE001 - isolate request failures so later requests can run
            payload = json.dumps({"status": "failed", "error": traceback.format_exc()}, sort_keys=True)
        print(payload, file=responses, flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker.py REQUEST.json | --serve")
    if sys.argv[1] == "--serve":
        serve()
        return
    request = parse_worker_request(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(run(request), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
