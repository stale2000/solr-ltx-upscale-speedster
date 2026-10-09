"""Validated JSON request and response types for the isolated H3 worker."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast


class SharedVAEOptions(TypedDict):
    vae_device: int
    upscale_devices: list[int]
    chunk_frames: int
    overlap_frames: int
    interpolate_overlaps: NotRequired[bool]
    transfer_mode: NotRequired[Literal["blocking", "peer_event"]]


class WorkerRequest(TypedDict):
    source: str
    output: str
    prompt: str
    width: int
    height: int
    fps: str
    seed: int
    decoder_seed: int
    output_backend: Literal["nvenc", "libx264", "libx264_lossless"]
    decoder_mode: NotRequired[Literal["diffusion", "pruna_fast"]]
    transformer_precision: NotRequired[Literal["bf16", "mxfp8", "nvfp4"]]
    pruna_tile_size: NotRequired[int]
    pruna_tile_stride: NotRequired[int]
    tile_size: NotRequired[int]
    tile_stride: NotRequired[int]
    cache_conditioning: NotRequired[bool]
    shared_vae: NotRequired[SharedVAEOptions]


class WorkerSuccessResponse(TypedDict):
    status: Literal["succeeded"]
    report: dict[str, object]


class WorkerFailureResponse(TypedDict):
    status: Literal["failed"]
    error: str


WorkerResponse = WorkerSuccessResponse | WorkerFailureResponse


def parse_worker_request(value: object) -> WorkerRequest:
    """Validate untrusted JSON before it reaches inference code."""
    if not isinstance(value, dict):
        raise TypeError("worker request must be a JSON object")
    required = {
        "source", "output", "prompt", "width", "height", "fps", "seed",
        "decoder_seed", "output_backend",
    }
    optional = {
        "tile_size", "tile_stride", "decoder_mode", "transformer_precision",
        "pruna_tile_size", "pruna_tile_stride",
        "cache_conditioning",
        "shared_vae",
    }
    if not required <= value.keys() or not value.keys() <= required | optional:
        missing = sorted(required - value.keys())
        unknown = sorted(value.keys() - required - optional)
        raise ValueError(f"invalid worker request fields; missing={missing}, unknown={unknown}")
    if "cache_conditioning" in value and type(value["cache_conditioning"]) is not bool:
        raise TypeError("worker request cache_conditioning must be a boolean")
    for key in ("source", "output", "prompt", "fps"):
        if not isinstance(value[key], str):
            raise TypeError(f"worker request field {key!r} must be a string")
    if not value["source"] or not value["output"]:
        raise ValueError("worker request source and output paths must be nonempty")
    try:
        frame_rate = Fraction(value["fps"])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("worker request fps must be a positive rational number") from exc
    if frame_rate <= 0:
        raise ValueError("worker request fps must be a positive rational number")
    for key in ("width", "height", "seed", "decoder_seed"):
        if type(value[key]) is not int:
            raise TypeError(f"worker request field {key!r} must be an integer")
    if value["width"] <= 0 or value["height"] <= 0:
        raise ValueError("worker request dimensions must be positive")
    if value["seed"] < 0 or value["decoder_seed"] < 0:
        raise ValueError("worker request seeds must be nonnegative")
    if value["output_backend"] not in ("nvenc", "libx264", "libx264_lossless"):
        raise ValueError("worker request output_backend must be 'nvenc', 'libx264', or 'libx264_lossless'")
    if value.get("decoder_mode", "diffusion") not in ("diffusion", "pruna_fast"):
        raise ValueError("worker request decoder_mode must be 'diffusion' or 'pruna_fast'")
    if "shared_vae" in value:
        from .shared_vae_plan import validate_chunk_settings

        options = value["shared_vae"]
        required_shared = {
            "vae_device", "upscale_devices", "chunk_frames", "overlap_frames",
        }
        if (not isinstance(options, dict) or not required_shared <= set(options)
                or set(options) - required_shared - {"interpolate_overlaps", "transfer_mode"}):
            raise ValueError("shared VAE options must specify devices, chunk frames, and overlap")
        if type(options.get("interpolate_overlaps", False)) is not bool:
            raise TypeError("shared VAE interpolate_overlaps must be a boolean")
        if options.get("transfer_mode", "blocking") not in {"blocking", "peer_event"}:
            raise ValueError("shared VAE transfer_mode must be blocking or peer_event")
        if value.get("decoder_mode") != "pruna_fast":
            raise ValueError("shared VAE mode requires pruna_fast")
        if value["output_backend"] not in {"nvenc", "libx264"}:
            raise ValueError("shared VAE mode requires a final nvenc or libx264 video")
        workers = options["upscale_devices"]
        if (type(options["vae_device"]) is not int or options["vae_device"] != 0
                or not isinstance(workers, list) or not 1 <= len(workers) <= 4
                or any(type(device) is not int for device in workers)
                or workers != list(range(1, len(workers) + 1))):
            raise ValueError("shared VAE worker-local topology must use VAE 0 and upscalers 1..N")
        validate_chunk_settings(options["chunk_frames"], options["overlap_frames"])
    if value.get("transformer_precision", "nvfp4") not in ("bf16", "mxfp8", "nvfp4"):
        raise ValueError("worker request transformer_precision must be 'bf16', 'mxfp8', or 'nvfp4'")
    for key in {"tile_size", "tile_stride", "pruna_tile_size", "pruna_tile_stride"} & value.keys():
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"worker request field {key!r} must be a positive integer")
    if value.get("decoder_mode", "diffusion") == "pruna_fast":
        tile_size = value.get("pruna_tile_size", 1024)
        tile_stride = value.get("pruna_tile_stride", 1024)
        if tile_stride > tile_size:
            raise ValueError("worker request Pruna tile stride must not exceed tile size")
    else:
        tile_size = value.get("tile_size", 1024 if max(value["width"], value["height"]) <= 832 else 768)
        tile_stride = value.get("tile_stride", 768 if tile_size == 1024 else 512)
        if tile_size <= 0 or tile_stride <= 0 or tile_stride > tile_size:
            raise ValueError(
                "worker request diffusion tile size and stride must be positive with stride <= size"
            )
    return cast(WorkerRequest, value)


def _report_string(report: dict[str, object], field: str) -> str:
    value = report.get(field)
    if not isinstance(value, str) or not value:
        raise TypeError(f"successful worker report field {field!r} must be a nonempty string")
    return value


def _report_int(
    report: dict[str, object], field: str, *, minimum: int = 0
) -> int:
    value = report.get(field)
    if type(value) is not int or value < minimum:
        raise TypeError(f"successful worker report field {field!r} must be an integer >= {minimum}")
    return value


def _report_number(report: dict[str, object], field: str) -> float:
    value = report.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise TypeError(f"successful worker report field {field!r} must be a finite nonnegative number")
    return float(value)


def _validate_export(value: object, report: dict[str, object]) -> None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("successful worker report export must be an object")
    export = cast(dict[str, object], value)
    for field in ("frames", "width", "height", "packing_chunk_frames", "max_packed_batch_bytes"):
        _report_int(export, field, minimum=1)
    for field in ("pack_seconds", "encode_seconds"):
        _report_number(export, field)
    for field in ("host_copy_seconds", "pipe_write_seconds", "ffmpeg_wait_seconds"):
        if field in export:
            _report_number(export, field)
    for field in ("fps", "backend", "pixel_rounding"):
        _report_string(export, field)
    if type(export.get("audio_stream_copied")) is not bool:
        raise TypeError("successful worker report export audio_stream_copied must be a boolean")
    if export["backend"] not in ("nvenc", "libx264", "libx264_lossless"):
        raise ValueError("successful worker report export backend is invalid")
    if any(export[field] != report[field] for field in ("frames", "width", "height", "fps")):
        raise ValueError("successful worker report export geometry does not match the report")
    if "input_pixel_format" in export:
        pixel_format = _report_string(export, "input_pixel_format")
        if pixel_format not in {"rgb24", "yuv420p"}:
            raise ValueError("successful worker report export input pixel format is invalid")
    if "packing_device" in export:
        _report_string(export, "packing_device")


def _validate_shared_vae_report(value: object, report: dict[str, object]) -> None:
    if not isinstance(value, dict):
        raise TypeError("shared VAE benchmark report must be an object")
    details = cast(dict[str, object], value)
    if details.get("source_frames") != report["frames"]:
        raise ValueError("shared VAE source frame count does not match its output")
    overlap = _report_int(details, "temporal_overlap_frames", minimum=1)
    if overlap not in {16, 32, 48, 64}:
        raise ValueError("shared VAE reported invalid overlap")
    for field, expected in (("source_decode_count", 1), ("intermediate_video_files", 0),
                            ("final_encode_count", 1)):
        if _report_int(details, field) != expected:
            raise ValueError(f"shared VAE report has unexpected {field}")
    topology = details.get("topology")
    if not isinstance(topology, dict):
        raise TypeError("shared VAE report must include topology")
    workers = topology.get("upscale_devices")
    if (type(topology.get("vae_device")) is not int or topology["vae_device"] != 0
            or not isinstance(workers, list) or not 1 <= len(workers) <= 4
            or any(type(device) is not int for device in workers)
            or workers != list(range(1, len(workers) + 1))
            or topology.get("same_host") is not True
            or topology.get("vae_copies") != 1
            or topology.get("transformer_copies") != len(workers)):
        raise ValueError("shared VAE report topology is invalid")
    if _report_int(details, "max_inflight_chunks", minimum=1) > 2 * len(workers):
        raise ValueError("shared VAE exceeded its two-job-per-transformer bound")
    stages = details.get("stages")
    if not isinstance(stages, dict):
        raise TypeError("shared VAE report must include stage timings")
    for field in ("source_decode_s", "vae_encode_s", "latent_upsample_s",
                  "vae_encode_and_upsample_s", "transformer_s", "latent_transfer_s",
                  "vae_decode_s", "blend_and_export_s", "final_encode_wait_s"):
        _report_number(stages, field)
    chunks = details.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise TypeError("shared VAE report must include chunk coverage")
    cursor = processed = context_frames = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or chunk.get("index") != index:
            raise ValueError("shared VAE report chunks must be in source order")
        core, window = chunk.get("core_frame_range"), chunk.get("input_frame_range")
        if (not isinstance(core, list) or not isinstance(window, list)
                or len(core) != 2 or len(window) != 2
                or any(type(item) is not int for item in (*core, *window))
                or core[0] != cursor or core[1] <= cursor
                or not 0 <= window[0] <= core[0] < core[1] <= window[1] <= report["frames"]):
            raise ValueError("shared VAE report chunk frame ranges are inconsistent")
        cursor = core[1]
        count = _report_int(chunk, "processed_frames", minimum=1)
        if count < window[1] - window[0]:
            raise ValueError("shared VAE processed fewer frames than its input window")
        processed += count
        context_frames += window[1] - window[0] - (core[1] - core[0])
    if (cursor != report["frames"] or processed != report["processed_frames"]
            or context_frames != report.get("trimmed_context_frames")):
        raise ValueError("shared VAE chunk coverage does not match its frame totals")
    _report_int(details, "latent_transfer_bytes", minimum=1)
    transfers = details.get("transfers")
    if not isinstance(transfers, list) or len(transfers) != 2 * len(chunks):
        raise ValueError("shared VAE must report both latent transfers for each chunk")
    transfer_bytes = 0
    seen: set[tuple[int, str]] = set()
    for transfer in transfers:
        if not isinstance(transfer, dict):
            raise TypeError("shared VAE latent transfer must be an object")
        index = _report_int(transfer, "chunk")
        direction = _report_string(transfer, "direction")
        pair = (index, direction)
        if index >= len(chunks) or pair in seen:
            raise ValueError("shared VAE latent transfer chunk is invalid or repeated")
        seen.add(pair)
        source = _report_int(transfer, "source_device")
        destination = _report_int(transfer, "destination_device")
        if ((direction == "conditioning_to_transformer" and (source != 0 or destination not in workers))
                or (direction == "refined_to_vae" and (source not in workers or destination != 0))
                or direction not in {"conditioning_to_transformer", "refined_to_vae"}):
            raise ValueError("shared VAE latent transfer direction is invalid")
        transfer_bytes += _report_int(transfer, "bytes", minimum=1)
        _report_number(transfer, "seconds")
        _report_string(transfer, "mechanism")
        if type(transfer.get("peer_access_available")) is not bool:
            raise TypeError("shared VAE transfer peer availability must be boolean")
    if transfer_bytes != details["latent_transfer_bytes"]:
        raise ValueError("shared VAE total transferred bytes is inconsistent")


def _validate_success_report(
    report: dict[str, object], expected_output: str | Path
) -> None:
    if _report_string(report, "status") != "succeeded":
        raise ValueError("successful worker response report status must be 'succeeded'")
    for field in (
        "model", "model_revision", "gpu", "torch", "natten_backend", "model_residency",
        "allocator_config", "python", "fps", "prompt",
    ):
        _report_string(report, field)
    if report.get("cuda") is not None and not isinstance(report["cuda"], str):
        raise TypeError("successful worker report field 'cuda' must be a string or null")
    for field in (
        "decoder_fused_rope_modules", "frames", "processed_frames", "trimmed_padding_frames",
        "width", "height", "seed", "decoder_seed", "peak_allocated_bytes", "peak_reserved_bytes",
    ):
        _report_int(report, field, minimum=1 if field in {"frames", "width", "height"} else 0)
    if type(report.get("cpu_offload_enabled")) is not bool:
        raise TypeError("successful worker report field 'cpu_offload_enabled' must be a boolean")
    for field in ("model_load_s", "inference_s", "duration_seconds", "total_seconds"):
        _report_number(report, field)
    if "fixed_conditioning_prepare_s" in report:
        _report_number(report, "fixed_conditioning_prepare_s")
    if "fixed_conditioning_sha256" in report:
        checksum = report["fixed_conditioning_sha256"]
        if checksum is not None and (
            not isinstance(checksum, str) or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("successful worker report fixed conditioning checksum is invalid")
    for field in (
        "transformer_compiled", "video_only_refiner", "cpu_preprocess_overlap",
        "pruna_decoder_compiled", "pruna_output_copy_optimized",
        "pruna_encoder_removal_enabled", "prompt_cache_enabled",
        "prompt_cache_hit", "pruna_inference_mode", "gpu_aminmax",
        "fixed_conditioning_enabled", "text_encoder_loaded",
    ):
        if field in report and type(report[field]) is not bool:
            raise TypeError(f"successful worker report field {field!r} must be a boolean")
    if "video_only_transformer_blocks" in report:
        _report_int(report, "video_only_transformer_blocks")
    for field in (
        "pruna_encoder_removed_parameters", "pruna_encoder_removed_bytes",
        "prompt_cache_entries", "prompt_cache_bytes", "fixed_conditioning_released_parameter_bytes",
    ):
        if field in report:
            _report_int(report, field)
    if (
        "pruna_output_contiguous" in report
        and report["pruna_output_contiguous"] is not None
        and type(report["pruna_output_contiguous"]) is not bool
    ):
        raise TypeError("successful worker report pruna_output_contiguous must be boolean or null")
    if "nvenc_preset" in report and report["nvenc_preset"] not in (None, "p1", "p2", "p4"):
        raise ValueError("successful worker report nvenc_preset is invalid")
    frames = _report_int(report, "frames", minimum=1)
    processed_frames = _report_int(report, "processed_frames")
    trimmed_frames = _report_int(report, "trimmed_padding_frames")
    context_frames = _report_int(report, "trimmed_context_frames") if "trimmed_context_frames" in report else 0
    if context_frames and "shared_vae_benchmark" not in report:
        raise ValueError("context trimming must belong to an explicit shared VAE benchmark")
    if processed_frames != frames + trimmed_frames + context_frames:
        raise ValueError("successful worker report frame counts are inconsistent")
    try:
        fps = Fraction(cast(str, report["fps"]))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("successful worker report fps must be a positive rational number") from exc
    duration = _report_number(report, "duration_seconds")
    if fps <= 0 or not math.isclose(
        duration,
        frames / float(fps),
        rel_tol=1e-6,
    ):
        raise ValueError("successful worker report duration does not match frames and fps")

    mode = report.get("decoder_mode", "diffusion")
    if mode not in ("diffusion", "pruna_fast"):
        raise ValueError("successful worker report decoder_mode is invalid")
    for field in ("decoder_tile_size", "decoder_tile_stride", "pruna_tile_size", "pruna_tile_stride"):
        if field in report and report[field] is not None:
            _report_int(report, field, minimum=1)
    if mode == "pruna_fast":
        size, stride = report.get("pruna_tile_size"), report.get("pruna_tile_stride")
        if type(size) is not int or type(stride) is not int:
            raise TypeError("Pruna worker report must include integer tile size and stride")
        if stride > size:
            raise ValueError("Pruna worker report tile stride must not exceed tile size")
    else:
        tile_size, tile_stride = report.get("decoder_tile_size"), report.get("decoder_tile_stride")
        has_tile_metadata = "decoder_tile_size" in report or "decoder_tile_stride" in report
        if has_tile_metadata and (type(tile_size) is not int or type(tile_stride) is not int):
            raise TypeError("diffusion worker report must include integer tile size and stride")
        if (tile_size is None) != (tile_stride is None):
            raise ValueError("diffusion worker report tile size and stride must be reported together")
        if type(tile_size) is int and type(tile_stride) is int and tile_stride > tile_size:
            raise ValueError("diffusion worker report tile stride must not exceed tile size")

    output = _report_string(report, "output")
    if Path(output).resolve() != Path(expected_output).resolve():
        raise ValueError("successful worker report output does not match the requested output path")
    _validate_export(report.get("export"), report)
    if "shared_vae_benchmark" in report:
        _validate_shared_vae_report(report["shared_vae_benchmark"], report)


def parse_worker_response(
    value: object, *, expected_output: str | Path
) -> WorkerResponse:
    """Validate the worker's untrusted stdout response before consuming it."""
    if not isinstance(value, dict):
        raise TypeError("worker response must be a JSON object")
    status = value.get("status")
    if status == "succeeded":
        report = value.get("report")
        if not isinstance(report, dict) or not all(isinstance(key, str) for key in report):
            raise TypeError("successful worker response must include an object report")
        typed_report = cast(dict[str, object], report)
        _validate_success_report(typed_report, expected_output)
        return {"status": "succeeded", "report": typed_report}
    if status == "failed":
        error = value.get("error")
        if not isinstance(error, str):
            raise TypeError("failed worker response must include a string error")
        return {"status": "failed", "error": error}
    raise ValueError("worker response status must be 'succeeded' or 'failed'")
