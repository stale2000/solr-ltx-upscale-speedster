"""Optional distilled convolutional LTX decoder for throughput-sensitive runs."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

PRUNA_REPO = "PrunaAI/PrunaVAED"
PRUNA_REVISION = "4baacd7ef66a6131439542c1f05872afe042e128"
DIFFUSERS_REVISION = "e0abab83b5df05de9e7abd788643c1a7c1e42e28"
DIFFUSERS_SOURCE_SHA256 = "1dbcd54b6fc005fb1895fb96f43618660d2a3671faf17fcbc09231b4579dde17"
ARTIFACT_SHA256 = {
    "patch_diffusers.py": "dfac5ca02891f3d7061f54b1410db65d1dfb91e558bee93ca571ee1ed066dbf3",
    "vae/config.json": "b636e95d41d93027529fa726ff75fe6eed55751a8cbe602e614ef7a600d287e7",
    "vae/diffusion_pytorch_model.safetensors": "4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d",
}
PRUNA_CONTIGUOUS_OUTPUT_ENV = "SOL_H3_UPSCALER_PRUNA_CONTIGUOUS_OUTPUT"
PRUNA_COMPILE_DECODER_ENV = "SOL_H3_UPSCALER_COMPILE_PRUNA_DECODER"
PRUNA_OUTPUT_COPY_OPT_ENV = "SOL_H3_UPSCALER_PRUNA_OUTPUT_COPY_OPT"
PRUNA_DROP_ENCODER_ENV = "SOL_H3_UPSCALER_PRUNA_DROP_ENCODER"
PRUNA_INFERENCE_MODE_ENV = "SOL_H3_UPSCALER_PRUNA_INFERENCE_MODE"
_PRUNA_DECODER_COMPILED = False
_PRUNA_ENCODER_REMOVED_PARAMETERS = 0
_PRUNA_ENCODER_REMOVED_BYTES = 0


def _enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1'")
    return value == "1"


def pruna_output_contiguous() -> bool:
    """Preserve the default copy unless its benchmark flag is explicitly disabled."""
    value = os.environ.get(PRUNA_CONTIGUOUS_OUTPUT_ENV, "1")
    if value not in {"0", "1"}:
        raise ValueError(f"{PRUNA_CONTIGUOUS_OUTPUT_ENV} must be '0' or '1'")
    return value == "1"


def compile_pruna_decoder_enabled() -> bool:
    """Keep decoder compilation experimental until a target-GPU A/B promotes it."""
    value = os.environ.get(PRUNA_COMPILE_DECODER_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{PRUNA_COMPILE_DECODER_ENV} must be '0' or '1'")
    return value == "1"


def pruna_output_copy_optimization_enabled() -> bool:
    return _enabled(PRUNA_OUTPUT_COPY_OPT_ENV, True)


def pruna_drop_encoder_enabled() -> bool:
    return _enabled(PRUNA_DROP_ENCODER_ENV, True)


def pruna_inference_mode_enabled() -> bool:
    return _enabled(PRUNA_INFERENCE_MODE_ENV, False)

def pruna_decoder_compiled() -> bool:
    return _PRUNA_DECODER_COMPILED


def pruna_encoder_removed() -> tuple[int, int]:
    return _PRUNA_ENCODER_REMOVED_PARAMETERS, _PRUNA_ENCODER_REMOVED_BYTES


def _discard_encoder(vae) -> tuple[int, int]:
    encoder = getattr(vae, "encoder", None)
    if encoder is None:
        return 0, 0
    parameters = tuple(encoder.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    size = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    vae.encoder = None
    return count, size


def _postprocess_pruna_pixels(
    pixels,
    video_processor,
    width: int,
    height: int,
    *,
    frame_count: int | None,
    optimize_output_copies: bool,
    output_contiguous: bool,
):
    """Normalize, crop and lay out decoded frames, optionally skipping full-video copies."""
    if pixels.ndim != 5 or pixels.shape[0] != 1:
        raise ValueError(f"Pruna decode must return one NCTHW video, got {tuple(pixels.shape)}")
    if frame_count is not None and (frame_count <= 0 or frame_count > pixels.shape[2]):
        raise ValueError("Pruna frame_count must fit the decoded video")
    top = (pixels.shape[-2] - height) // 2
    left = (pixels.shape[-1] - width) // 2
    if top < 0 or left < 0:
        raise ValueError("Pruna decode is smaller than the requested output geometry")

    if optimize_output_copies:
        count = frame_count if frame_count is not None else pixels.shape[2]
        video = pixels[0, :, :count, top:top + height, left:left + width]
        frames = video.permute(1, 0, 2, 3)
        frames = video_processor.postprocess(frames, output_type="pt")
    else:
        frames = video_processor.postprocess_video(pixels, output_type="pt")[0]
        frames = frames[:, :, top:top + height, left:left + width]

    if tuple(frames.shape[1:]) != (3, height, width):
        raise ValueError(f"Fast decoder returned wrong geometry: {tuple(frames.shape)}")
    frames = frames.permute(0, 2, 3, 1)
    return frames.contiguous() if output_contiguous else frames


def _decode_pruna_latents(vae, latents, torch, *, inference_mode: bool):
    if inference_mode:
        with torch.inference_mode():
            return vae.decode(latents, return_dict=False)[0]
    return vae.decode(latents, return_dict=False)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pruna_vae(torch):
    """Download pinned public weights once, verify them, and load fully on CUDA."""
    global _PRUNA_DECODER_COMPILED, _PRUNA_ENCODER_REMOVED_PARAMETERS, _PRUNA_ENCODER_REMOVED_BYTES
    import diffusers.models.autoencoders.autoencoder_kl_ltx2 as ltx2
    from diffusers import AutoencoderKLLTX2Video
    from huggingface_hub import snapshot_download

    if _sha256(Path(inspect.getfile(ltx2))) != DIFFUSERS_SOURCE_SHA256:
        raise RuntimeError(f"Fast decoder requires pinned Diffusers source {DIFFUSERS_REVISION}")
    model_dir = Path(snapshot_download(
        repo_id=PRUNA_REPO,
        revision=PRUNA_REVISION,
        allow_patterns=[*ARTIFACT_SHA256],
    ))
    for name, expected in ARTIFACT_SHA256.items():
        path = model_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Pinned Pruna decoder artifact failed verification: {name}")
    patch_path = model_dir / "patch_diffusers.py"
    spec = importlib.util.spec_from_file_location("sol_h3_upscaler_pinned_pruna_patch", patch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pinned Pruna Diffusers compatibility patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_pruna_ltx2_decoder()

    vae = AutoencoderKLLTX2Video.from_pretrained(
        model_dir / "vae", torch_dtype=torch.bfloat16, local_files_only=True,
    ).eval().requires_grad_(False)
    _PRUNA_ENCODER_REMOVED_PARAMETERS = 0
    _PRUNA_ENCODER_REMOVED_BYTES = 0
    if pruna_drop_encoder_enabled():
        _PRUNA_ENCODER_REMOVED_PARAMETERS, _PRUNA_ENCODER_REMOVED_BYTES = _discard_encoder(vae)
    vae = vae.to("cuda")
    vae.enable_tiling(
        tile_sample_min_height=1024,
        tile_sample_min_width=1024,
        tile_sample_min_num_frames=16,
        tile_sample_stride_height=1024,
        tile_sample_stride_width=1024,
        tile_sample_stride_num_frames=8,
    )
    _PRUNA_DECODER_COMPILED = compile_pruna_decoder_enabled()
    if _PRUNA_DECODER_COMPILED:
        vae.decoder = torch.compile(vae.decoder)
    return vae


def decode_pruna_vae(
    latents,
    width: int,
    height: int,
    torch,
    *,
    tile_size: int = 1024,
    tile_stride: int = 1024,
    frame_count: int | None = None,
):
    """Decode already-denormalized LTX latents and return [frames,H,W,RGB]."""
    global _PRUNA_VAE
    if _PRUNA_VAE is None:
        _PRUNA_VAE = load_pruna_vae(torch)
    if tile_size <= 0 or tile_stride <= 0 or tile_stride > tile_size:
        raise ValueError("Fast decoder tile size and stride must be positive with stride <= size")
    _PRUNA_VAE.enable_tiling(
        tile_sample_min_height=tile_size,
        tile_sample_min_width=tile_size,
        tile_sample_min_num_frames=16,
        tile_sample_stride_height=tile_stride,
        tile_sample_stride_width=tile_stride,
        tile_sample_stride_num_frames=8,
    )
    from diffusers.video_processor import VideoProcessor

    latent_dtype = next(_PRUNA_VAE.parameters()).dtype
    pixels = _decode_pruna_latents(
        _PRUNA_VAE,
        latents.to(device="cuda", dtype=latent_dtype),
        torch,
        inference_mode=pruna_inference_mode_enabled(),
    )
    return _postprocess_pruna_pixels(
        pixels,
        VideoProcessor(vae_scale_factor=32),
        width,
        height,
        frame_count=frame_count if pruna_output_copy_optimization_enabled() else None,
        optimize_output_copies=pruna_output_copy_optimization_enabled(),
        output_contiguous=pruna_output_contiguous(),
    )


_PRUNA_VAE = None
