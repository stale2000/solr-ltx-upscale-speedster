"""Bounded FFmpeg encoding for CPU and CUDA H3 refiner outputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from fractions import Fraction
from typing import Any

NVENC_PRESET_ENV = "SOL_H3_UPSCALER_NVENC_PRESET"
NVENC_INPUT_FORMAT_ENV = "SOL_H3_UPSCALER_NVENC_INPUT_FORMAT"
GPU_AMINMAX_ENV = "SOL_H3_UPSCALER_GPU_AMINMAX"
ASYNC_NVENC_STAGING_ENV = "SOL_H3_UPSCALER_ASYNC_NVENC_STAGING"
NVENC_INPUT_HASH_ENV = "SOL_H3_UPSCALER_HASH_NVENC_INPUT"


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "FFmpeg is required for bounded video export; install imageio-ffmpeg "
            "in the ComfyUI environment"
        ) from exc


def _experiment_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean feature flag without accepting unrecognized values."""
    value = os.environ.get(name, "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1'")
    return value == "1"


def _nvenc_preset() -> str:
    preset = os.environ.get(NVENC_PRESET_ENV, "p4")
    if preset not in {"p1", "p2", "p4"}:
        raise ValueError(f"{NVENC_PRESET_ENV} must be one of p1, p2, p4")
    return preset


def _nvenc_input_pixel_format(backend: str) -> str:
    """Choose the raw pipe format, defaulting to GPU-packed YUV420P for NVENC."""
    if backend != "nvenc":
        return "rgb24"
    pixel_format = os.environ.get(NVENC_INPUT_FORMAT_ENV, "yuv420p")
    if pixel_format not in {"rgb24", "yuv420p"}:
        raise ValueError(f"{NVENC_INPUT_FORMAT_ENV} must be rgb24 or yuv420p")
    return pixel_format


def _encoder_args(backend: str) -> list[str]:
    if backend == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", _nvenc_preset(), "-cq", "18", "-bf", "0"]
    if backend == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    if backend == "libx264_lossless":
        return ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]
    raise ValueError("output_backend must be 'nvenc', 'libx264', or 'libx264_lossless'")


def ffmpeg_export_command(
    source: str,
    output: str,
    *,
    width: int,
    height: int,
    frame_rate: Fraction,
    frame_count: int,
    backend: str,
    has_audio: bool,
    input_pixel_format: str = "rgb24",
) -> list[str]:
    if min(width, height, frame_count) <= 0 or width % 2 or height % 2:
        raise ValueError("output geometry must be positive and even")
    if input_pixel_format not in {"rgb24", "yuv420p"}:
        raise ValueError("input pixel format must be rgb24 or yuv420p")
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", input_pixel_format,
        "-video_size", f"{width}x{height}",
        "-framerate", f"{frame_rate.numerator}/{frame_rate.denominator}",
        "-i", "pipe:0", "-i", source, "-map", "0:v:0",
    ]
    if has_audio:
        command += ["-map", "1:a:0", "-c:a", "copy"]
    # Bound only video frames: a global output-time limit starts at AAC's
    # negative priming timestamp and can silently discard source tail packets.
    command += [
        *_encoder_args(backend), "-pix_fmt", "yuv420p", "-frames:v", str(frame_count),
        "-movflags", "+faststart", output,
    ]
    return command


def _pack_rgb_as_yuv420p(frames: Any, torch_module: Any) -> Any:
    """Convert RGB float frames to limited-range BT.601 YUV420P on their device."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("YUV420P packing expects RGB frames shaped (N,H,W,3)")
    count, height, width, _channels = map(int, frames.shape)
    if min(count, height, width) <= 0 or height % 2 or width % 2:
        raise ValueError("YUV420P packing requires nonempty frames with even geometry")

    rgb = (frames.float() * 255.0).round().clamp(0, 255)
    red, green, blue = rgb.unbind(dim=-1)
    # Match libswscale's default limited-range BT.601 RGB->YUV conversion, and
    # perform 4:2:0 chroma averaging before crossing the GPU-to-host pipe.
    luma = 16.0 + (219.0 / 255.0) * (
        0.299 * red + 0.587 * green + 0.114 * blue
    )
    cb = (224.0 / 255.0) * (
        -0.168736 * red - 0.331264 * green + 0.5 * blue
    )
    cr = (224.0 / 255.0) * (
        0.5 * red - 0.418688 * green - 0.081312 * blue
    )
    cb = cb.reshape(count, height // 2, 2, width // 2, 2).mean(dim=(2, 4))
    cr = cr.reshape(count, height // 2, 2, width // 2, 2).mean(dim=(2, 4))
    y_bytes = luma.round().clamp(16, 235).to(torch_module.uint8).flatten(1)
    u_bytes = (128.0 + cb).round().clamp(16, 240).to(torch_module.uint8).flatten(1)
    v_bytes = (128.0 + cr).round().clamp(16, 240).to(torch_module.uint8).flatten(1)
    return torch_module.cat((y_bytes, u_bytes, v_bytes), dim=1).contiguous()


def _gpu_pixel_range(frames: Any, torch_module: Any) -> tuple[float, float]:
    """Preserve range validation while optionally combining its reductions."""
    if getattr(getattr(frames, "device", None), "type", None) == "cuda" and \
            _experiment_flag(GPU_AMINMAX_ENV):
        low, high = torch_module.aminmax(frames)
    else:
        low, high = frames.min(), frames.max()
    return float(low), float(high)


def _write_frames(frames: Any, source: str, output: str, fps: Fraction,
                  backend: str, has_audio: bool, chunk_frames: int = 4) -> dict[str, Any]:
    """Quantize small batches and stream them; never allocate a second full video."""
    if getattr(frames, "is_cuda", False):
        return _write_gpu_frames(frames, source, output, fps, backend, has_audio, chunk_frames)
    import numpy as np
    import torch

    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("refiner output must have shape (frames, height, width, 3)")
    if not np.issubdtype(frames.dtype, np.floating):
        raise ValueError("refiner output must use normalized floating-point RGB")
    count, height, width, _channels = map(int, frames.shape)
    if min(count, height, width) <= 0:
        raise ValueError("refiner output is empty")
    if not np.isfinite(frames).all():
        raise ValueError("refiner output contains non-finite pixels")
    low, high = _gpu_pixel_range(frames, torch)
    if low < -0.001 or high > 1.001:
        raise ValueError(f"refiner output is outside [0, 1]: min={low}, max={high}")
    command = ffmpeg_export_command(
        source, output, width=width, height=height, frame_rate=fps,
        frame_count=count, backend=backend, has_audio=has_audio,
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    pipe = process.stdin
    if pipe is None:
        process.kill()
        process.wait()
        raise RuntimeError("FFmpeg did not expose its raw-video input")
    start = time.perf_counter()
    pack_seconds = 0.0
    pipe_write_seconds = 0.0
    ffmpeg_wait_seconds = 0.0
    max_batch_bytes = 0
    try:
        for first in range(0, count, chunk_frames):
            tick = time.perf_counter()
            # Match Diffusers' float RGB -> NumPy round-to-even -> uint8 path.
            batch = np.ascontiguousarray(
                np.rint(frames[first:first + chunk_frames] * 255).clip(0, 255).astype(np.uint8)
            )
            pack_seconds += time.perf_counter() - tick
            max_batch_bytes = max(max_batch_bytes, int(batch.nbytes))
            write_started = time.perf_counter()
            pipe.write(memoryview(batch).cast("B"))
            pipe_write_seconds += time.perf_counter() - write_started
        flush_started = time.perf_counter()
        pipe.close()
        pipe_write_seconds += time.perf_counter() - flush_started
        wait_started = time.perf_counter()
        code = process.wait()
        ffmpeg_wait_seconds = time.perf_counter() - wait_started
        if code:
            raise RuntimeError(f"FFmpeg export failed with exit code {code}")
        if not os.path.isfile(output) or os.path.getsize(output) == 0:
            raise RuntimeError("FFmpeg returned success without a non-empty output file")
    except BaseException:
        process.kill()
        process.wait()
        raise
    return {
        "frames": count,
        "width": width,
        "height": height,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "audio_stream_copied": has_audio,
        "backend": backend,
        "pack_seconds": pack_seconds,
        "pipe_write_seconds": pipe_write_seconds,
        "ffmpeg_wait_seconds": ffmpeg_wait_seconds,
        "encode_seconds": time.perf_counter() - start,
        "packing_chunk_frames": chunk_frames,
        "max_packed_batch_bytes": max_batch_bytes,
        "pixel_rounding": "Diffusers float RGB, NumPy round-to-even, clipped uint8",
    }


def _write_gpu_frames(frames: Any, source: str, output: str, fps: Fraction,
                      backend: str, has_audio: bool, chunk_frames: int) -> dict[str, Any]:
    """Pack normalized float RGB and overlap two pinned export batches."""
    from concurrent.futures import Future, ThreadPoolExecutor

    import torch

    if chunk_frames < 1 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("CUDA output must be (frames,height,width,3), with a positive packing batch")
    if frames.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("CUDA output packing requires normalized floating-point RGB")
    count, height, width, _channels = map(int, frames.shape)
    if min(count, height, width) <= 0 or not bool(torch.isfinite(frames).all()):
        raise ValueError("CUDA refiner output is empty or contains non-finite pixels")
    low, high = _gpu_pixel_range(frames, torch)
    if low < -0.001 or high > 1.001:
        raise ValueError(f"CUDA refiner output is outside [0,1]: {low}, {high}")
    input_pixel_format = _nvenc_input_pixel_format(backend)
    command = ffmpeg_export_command(
        source, output, width=width, height=height, frame_rate=fps,
        frame_count=count, backend=backend, has_audio=has_audio,
        input_pixel_format=input_pixel_format,
    )
    async_staging = _experiment_flag(ASYNC_NVENC_STAGING_ENV, default=False)
    hash_nvenc_input = _experiment_flag(NVENC_INPUT_HASH_ENV, default=False)
    input_hasher = hashlib.sha256() if hash_nvenc_input else None
    batch_count = min(chunk_frames, count)
    staging_shape = (
        (batch_count, height * width * 3 // 2)
        if input_pixel_format == "yuv420p"
        else (batch_count, height, width, 3)
    )
    staging = [
        torch.empty(staging_shape, dtype=torch.uint8, pin_memory=True)
        for _ in range(2 if async_staging else 1)
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    pipe = process.stdin
    if pipe is None:
        process.kill()
        process.wait()
        raise RuntimeError("FFmpeg did not expose its raw-video input")
    started = time.perf_counter()
    pack_seconds = copy_seconds = pipe_write_seconds = ffmpeg_wait_seconds = 0.0
    writer = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="sol-h3-nvenc")
        if async_staging else None
    )
    write_futures: list[Future[dict[str, float]]] = []
    source_owners: list[Any | None] = [None, None]

    def write_ready_batch(
        slot: int,
        length: int,
        pack_start: Any,
        pack_end: Any,
        copy_end: Any,
    ) -> dict[str, float]:
        """Read only completed DMA and return its slot after pipe consumption."""
        copy_end.synchronize()
        payload = memoryview(staging[slot][:length].numpy()).cast("B")
        if input_hasher is not None:
            input_hasher.update(payload)
        write_started = time.perf_counter()
        written = pipe.write(payload)
        elapsed = time.perf_counter() - write_started
        if written != len(payload):
            raise OSError(f"short FFmpeg pipe write: {written} of {len(payload)} bytes")
        return {
            "pack_seconds": pack_start.elapsed_time(pack_end) / 1000,
            "copy_seconds": pack_end.elapsed_time(copy_end) / 1000,
            "pipe_write_seconds": elapsed,
        }

    try:
        if async_staging:
            assert writer is not None
            for batch_index, first in enumerate(range(0, count, chunk_frames)):
                slot = batch_index % 2
                prior = write_futures[batch_index - 2] if batch_index >= 2 else None
                if prior is not None:
                    # Resolve and account before reusing this staging slot. The
                    # final drain below starts after all results handled here.
                    timings = prior.result()
                    pack_seconds += timings["pack_seconds"]
                    copy_seconds += timings["copy_seconds"]
                    pipe_write_seconds += timings["pipe_write_seconds"]
                    source_owners[slot] = None
                length = min(chunk_frames, count - first)
                pack_start, pack_end, copy_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(3)
                ]
                copy_stream = torch.cuda.current_stream()
                pack_start.record()
                frame_batch = frames[first:first + length]
                if input_pixel_format == "yuv420p":
                    packed = _pack_rgb_as_yuv420p(frame_batch, torch)
                else:
                    packed = (frame_batch.float() * 255).round().clamp_(0, 255)
                    packed = packed.to(torch.uint8).contiguous()
                pack_end.record()
                staging[slot][:length].copy_(packed, non_blocking=True)
                packed.record_stream(copy_stream)
                copy_end.record()
                source_owners[slot] = packed
                write_futures.append(writer.submit(
                    write_ready_batch, slot, length, pack_start, pack_end, copy_end,
                ))
            # Every future except the final two was already accounted for when
            # its staging slot became eligible for reuse.
            first_undrained = max(0, len(write_futures) - 2)
            for batch_index in range(first_undrained, len(write_futures)):
                future = write_futures[batch_index]
                timings = future.result()
                pack_seconds += timings["pack_seconds"]
                copy_seconds += timings["copy_seconds"]
                pipe_write_seconds += timings["pipe_write_seconds"]
                source_owners[batch_index % 2] = None
            writer.shutdown(wait=True)
        else:
            for first in range(0, count, chunk_frames):
                length = min(chunk_frames, count - first)
                pack_start, pack_end, copy_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(3)
                ]
                pack_start.record()
                frame_batch = frames[first:first + length]
                if input_pixel_format == "yuv420p":
                    packed = _pack_rgb_as_yuv420p(frame_batch, torch)
                else:
                    packed = (frame_batch.float() * 255).round().clamp_(0, 255)
                    packed = packed.to(torch.uint8).contiguous()
                pack_end.record()
                staging[0][:length].copy_(packed, non_blocking=True)
                copy_end.record()
                copy_end.synchronize()
                pack_seconds += pack_start.elapsed_time(pack_end) / 1000
                copy_seconds += pack_end.elapsed_time(copy_end) / 1000
                payload = memoryview(staging[0][:length].numpy()).cast("B")
                if input_hasher is not None:
                    input_hasher.update(payload)
                write_started = time.perf_counter()
                written = pipe.write(payload)
                pipe_elapsed = time.perf_counter() - write_started
                if written != len(payload):
                    raise OSError(f"short FFmpeg pipe write: {written} of {len(payload)} bytes")
                pipe_write_seconds += pipe_elapsed
        flush_started = time.perf_counter()
        pipe.close()
        pipe_write_seconds += time.perf_counter() - flush_started
        wait_started = time.perf_counter()
        code = process.wait()
        ffmpeg_wait_seconds = time.perf_counter() - wait_started
        if code or not os.path.isfile(output) or os.path.getsize(output) == 0:
            raise RuntimeError(f"FFmpeg CUDA export did not complete: exit={code}")
    except BaseException:
        process.kill()
        process.wait()
        if writer is not None:
            writer.shutdown(wait=True, cancel_futures=True)
        raise
    return {
        "frames": count, "width": width, "height": height,
        "fps": f"{fps.numerator}/{fps.denominator}", "backend": backend,
        "input_pixel_format": input_pixel_format,
        "audio_stream_copied": has_audio, "packing_device": "cuda",
        "staging_buffers": len(staging),
        "pack_seconds": pack_seconds, "host_copy_seconds": copy_seconds,
        "pipe_write_seconds": pipe_write_seconds,
        "ffmpeg_wait_seconds": ffmpeg_wait_seconds,
        "encode_seconds": time.perf_counter() - started,
        "packing_chunk_frames": chunk_frames,
        "max_packed_batch_bytes": staging[0].numel(),
        "pixel_rounding": (
            "FP32 normalized RGB to limited-range BT.601 YUV420P, round-to-even"
            if input_pixel_format == "yuv420p"
            else "FP32 normalized RGB, round-to-even, clipped uint8"
        ),
        **({"nvenc_input_sha256": input_hasher.hexdigest()} if input_hasher else {}),
    }
