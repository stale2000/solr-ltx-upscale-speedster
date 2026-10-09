"""Experimental temporal chunk planning and assembly for local multi-GPU runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoChunk:
    index: int
    device: str
    core_start: int
    core_end: int
    input_start: int
    input_end: int

    @property
    def input_frames(self) -> int:
        return self.input_end - self.input_start


def parse_devices(value: str, *, maximum: int = 4) -> tuple[str, ...]:
    """Parse distinct visible CUDA ordinals, requiring two for split mode."""
    parts = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not 2 <= len(parts) <= maximum:
        raise ValueError(f"multi-GPU mode requires between 2 and {maximum} CUDA device IDs")
    if any(not piece.isdecimal() for piece in parts):
        raise ValueError("CUDA device IDs must be comma-separated visible numeric ordinals")
    canonical = tuple(str(int(piece)) for piece in parts)
    if len(set(canonical)) != len(canonical):
        raise ValueError("CUDA device IDs must be unique")
    return canonical


def resolve_visible_devices(devices: tuple[str, ...], visible: str | None = None) -> tuple[str, ...]:
    """Map Comfy's visible ordinals back to the CUDA_VISIBLE_DEVICES tokens."""
    tokens = tuple(token.strip() for token in visible.split(",") if token.strip()) if visible else ()
    if not tokens:
        return devices
    resolved: list[str] = []
    for device in devices:
        index = int(device)
        if index >= len(tokens):
            raise ValueError(f"CUDA device {device} is outside the Comfy-visible device list")
        resolved.append(tokens[index])

    def canonical(token: str) -> str:
        # CUDA accepts decimal ordinals with leading zeroes and GPU UUIDs
        # regardless of hex letter case or the optional GPU- prefix.
        if token.isdecimal():
            return str(int(token))
        return token.casefold().removeprefix("gpu-")

    if len({canonical(token) for token in resolved}) != len(resolved):
        raise ValueError("CUDA device IDs resolve to the same visible GPU")
    return tuple(resolved)


def plan_chunks(
    frame_count: int,
    devices: tuple[str, ...],
    overlap: int,
    *,
    temporal_factor: int = 8,
) -> tuple[VideoChunk, ...]:
    """Partition a clip into aligned cores with temporal context on each side."""
    if frame_count <= 0 or temporal_factor <= 0:
        raise ValueError("frame count and temporal factor must be positive")
    if overlap < 2 * temporal_factor or overlap % (2 * temporal_factor):
        raise ValueError(f"temporal overlap must be a positive multiple of {2 * temporal_factor}")
    if len(devices) < 2:
        raise ValueError("at least two distinct CUDA devices are required")
    context = overlap // 2
    if frame_count < len(devices) * (context + temporal_factor + 1):
        raise ValueError("video is too short for the selected worker count and overlap")

    boundaries = [0]
    for part in range(1, len(devices)):
        ideal = frame_count * part / len(devices)
        aligned = round(ideal / temporal_factor) * temporal_factor
        minimum = boundaries[-1] + context + temporal_factor
        maximum = frame_count - (len(devices) - part) * (context + temporal_factor)
        aligned = min(maximum, max(minimum, aligned))
        aligned -= aligned % temporal_factor
        if aligned <= boundaries[-1] or aligned >= frame_count:
            raise ValueError("video cannot be split into non-empty aligned chunks")
        boundaries.append(aligned)
    boundaries.append(frame_count)

    chunks: list[VideoChunk] = []
    for index, device in enumerate(devices):
        core_start, core_end = boundaries[index], boundaries[index + 1]
        input_start = 0 if index == 0 else core_start - context
        input_end = frame_count if index == len(devices) - 1 else core_end + context
        chunks.append(VideoChunk(index, device, core_start, core_end, input_start, input_end))
    if any(chunk.input_start % temporal_factor for chunk in chunks):
        raise ValueError("chunk inference starts must align to the model's temporal phase")
    return tuple(chunks)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg is required for temporal multi-GPU mode")
    return executable


def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("FFprobe is required for temporal multi-GPU mode")
    return executable


def _run(arguments: list[str]) -> None:
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=1800, check=False)
    if result.returncode:
        raise RuntimeError(f"FFmpeg command failed: {result.stderr[-3000:]}")


def probe_video(path: str | Path) -> dict[str, Any]:
    result = subprocess.run(
        [_ffprobe(), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("multi-GPU input must contain exactly one video stream")
    video = videos[0]
    frames = int(video.get("nb_read_frames", 0))
    if frames <= 0:
        raise ValueError("multi-GPU input must have a finite, readable frame count")
    fps = Fraction(video["avg_frame_rate"])
    nominal_fps = Fraction(video["r_frame_rate"])
    if fps <= 0 or nominal_fps != fps:
        raise ValueError("multi-GPU benchmark currently requires constant-frame-rate input")
    return {
        "frames": frames,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "pixel_format": video.get("pix_fmt"),
        "fps": fps,
        "audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def extract_chunk(
    source: str | Path,
    destination: str | Path,
    chunk: VideoChunk,
    fps: Fraction,
) -> float:
    """Write an audio-free, lossless YUV420P MP4 at exact source frame indices."""
    started = time.perf_counter()
    _run([
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-vf",
        f"trim=start_frame={chunk.input_start}:end_frame={chunk.input_end},setpts=PTS-STARTPTS,fps={fps.numerator}/{fps.denominator}",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
        "-frames:v", str(chunk.input_frames), "-movflags", "+faststart", str(destination),
    ])
    probe = probe_video(destination)
    if probe["frames"] != chunk.input_frames or probe["fps"] != fps:
        raise RuntimeError("lossless chunk extraction changed frame count or frame rate")
    return time.perf_counter() - started


def assemble_chunks(
    chunks: tuple[VideoChunk, ...],
    outputs: tuple[Path, ...],
    source: str | Path,
    destination: str | Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: Fraction,
    output_backend: str,
) -> dict[str, object]:
    """Cross-fade temporal context windows, then attach the original audio once."""
    if len(chunks) < 2 or len(chunks) != len(outputs):
        raise ValueError("every planned CUDA chunk must have exactly one output")
    started = time.perf_counter()
    for chunk, output in zip(chunks, outputs, strict=True):
        probe = probe_video(output)
        if (
            probe["frames"] != chunk.input_frames
            or probe["width"] != width
            or probe["height"] != height
            or probe["fps"] != fps
        ):
            raise RuntimeError(f"chunk {chunk.index} output does not match its exact frame contract")

    filter_parts: list[str] = []
    for index, chunk in enumerate(chunks):
        filter_parts.append(
            f"[{index}:v]fps={fps.numerator}/{fps.denominator},settb=AVTB,setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:flags=lanczos,format=yuv420p[v{index}]"
        )
    current = "v0"
    current_frames = chunks[0].input_frames
    for index, chunk in enumerate(chunks[1:], 1):
        overlap_frames = chunks[index - 1].input_end - chunk.input_start
        if overlap_frames <= 0:
            raise ValueError("neighboring CUDA chunks must have a non-empty shared context")
        offset_frames = current_frames - overlap_frames
        if offset_frames < 0:
            raise ValueError("chunk overlap exceeds the accumulated video")
        offset_seconds = Fraction(offset_frames * fps.denominator, fps.numerator)
        duration_seconds = Fraction(overlap_frames * fps.denominator, fps.numerator)
        output_name = f"joined{index}"
        filter_parts.append(
            f"[{current}][v{index}]xfade=transition=fade:"
            f"duration={float(duration_seconds):.12f}:offset={float(offset_seconds):.12f}[{output_name}]"
        )
        current = output_name
        current_frames += chunk.input_frames - overlap_frames

    if current_frames != frame_count:
        raise RuntimeError(f"chunk ranges assemble to {current_frames} frames, expected {frame_count}")
    arguments = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    arguments.extend(part for output in outputs for part in ("-i", str(output)))
    audio_index = len(outputs)
    arguments.extend(["-i", str(source), "-filter_complex", ";".join(filter_parts), "-map", f"[{current}]", "-map", f"{audio_index}:a:0?"])
    if output_backend == "nvenc":
        preset = os.environ.get("SOL_H3_UPSCALER_NVENC_PRESET", "p4")
        if preset not in {"p1", "p2", "p4"}:
            raise ValueError("SOL_H3_UPSCALER_NVENC_PRESET must be p1, p2, or p4")
        arguments.extend(["-c:v", "h264_nvenc", "-preset", preset, "-cq", "18", "-bf", "0"])
    elif output_backend == "libx264":
        arguments.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
    else:
        raise ValueError("Unsupported final output encoder")
    arguments.extend(["-pix_fmt", "yuv420p", "-frames:v", str(frame_count), "-c:a", "copy", "-movflags", "+faststart", str(destination)])
    _run(arguments)
    final = probe_video(destination)
    if final["frames"] != frame_count or final["fps"] != fps or (final["width"], final["height"]) != (width, height):
        raise RuntimeError("assembled video failed frame, rate, or output geometry validation")
    return {
        "frames": frame_count,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "assembly_seconds": time.perf_counter() - started,
        "audio_preserved": final["audio"],
        "blend": "linear cross-fade over shared temporal context",
        "output_backend": output_backend,
    }
