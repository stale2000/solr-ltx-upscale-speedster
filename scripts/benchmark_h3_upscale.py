"""Optional local coordinator for single, split, and shared-VAE H3 tests.

The command submits workflows to an already-running local ComfyUI server. It
never provisions or rents GPUs; the nodes manage their own inference processes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
import webbrowser
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

SINGLE_NODE = "MiniMaxH3SoLUpscaler"
MULTI_NODE = "MiniMaxH3SoLParallelBenchmark"
SHARED_VAE_NODE = "MiniMaxH3SoLSharedVAEBenchmark"
PROFILE_NODES = {"single": SINGLE_NODE, "multi": MULTI_NODE, "shared_vae": SHARED_VAE_NODE}
LOAD_VIDEO_NODE = "LoadVideo"
SAVE_VIDEO_NODE = "SaveVideo"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("benchmark coordinator must be a local HTTP(S) origin without credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("ComfyUI base URL has an invalid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("ComfyUI port must be between 1 and 65535")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return f"{parsed.scheme}://{host}" + (f":{port}" if port else "")


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if not 2 <= len(devices) <= 4:
        raise ValueError("multi-GPU mode requires two to four visible CUDA device IDs")
    if any(not item.isdecimal() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA device IDs must be distinct comma-separated numeric ordinals")
    return devices


def validate_shared_vae(
    devices: tuple[str, ...], vae_device: str, chunk_frames: int,
    overlap_frames: int, decoder_mode: str,
) -> None:
    if (not 1 <= len(devices) <= 4
            or any(not isinstance(item, str) or not item.isdecimal() for item in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("shared VAE mode requires one to four distinct CUDA device IDs")
    if not vae_device.isdecimal() or vae_device in devices:
        raise ValueError("VAE CUDA device must be a numeric ordinal distinct from upscale devices")
    if decoder_mode != "pruna_fast":
        raise ValueError("shared VAE mode requires pruna_fast decoding")
    if not 32 <= chunk_frames <= 512 or chunk_frames % 8:
        raise ValueError("shared VAE chunk frames must be 32 to 512 and divisible by 8")
    if overlap_frames not in {16, 32, 48, 64} or overlap_frames >= chunk_frames:
        raise ValueError("shared VAE overlap must be 16, 32, 48, or 64 frames and smaller than the chunk")


def build_workflow(
    *, profile: str, source_name: str, source_subfolder: str, prompt: str,
    width: int, height: int, seed: int, decoder_seed: int, decoder_mode: str,
    output_backend: str, transformer_precision: str, devices: tuple[str, ...],
    overlap_frames: int, filename_prefix: str, vae_device: str = "0",
    shared_vae_chunk_frames: int = 96, shared_vae_transfer_mode: str = "peer_event",
) -> dict[str, dict[str, Any]]:
    if profile not in PROFILE_NODES:
        raise ValueError("profile must be single, multi, or shared_vae")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("target dimensions must be positive even integers")
    node_inputs: dict[str, Any] = {
        "video": ["1", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "decoder_seed": decoder_seed,
        "decoder_mode": decoder_mode,
        "output_backend": output_backend,
        "transformer_precision": transformer_precision,
    }
    node_type = SINGLE_NODE
    if profile == "multi":
        if len(devices) < 2:
            raise ValueError("multi profile needs at least two CUDA devices")
        if overlap_frames not in {16, 32, 48, 64}:
            raise ValueError("overlap must be 16, 32, 48, or 64 frames")
        node_type = MULTI_NODE
        node_inputs.update({
            "cuda_devices": ",".join(devices),
            "temporal_overlap_frames": overlap_frames,
        })
    elif profile == "shared_vae":
        validate_shared_vae(devices, vae_device, shared_vae_chunk_frames, overlap_frames, decoder_mode)
        if shared_vae_transfer_mode not in {"blocking", "peer_event"}:
            raise ValueError("shared VAE transfer mode must be blocking or peer_event")
        node_type = SHARED_VAE_NODE
        node_inputs.update({
            "cuda_devices": ",".join(devices),
            "vae_device": vae_device,
            "chunk_frames": shared_vae_chunk_frames,
            "temporal_overlap_frames": overlap_frames,
            "transfer_mode": shared_vae_transfer_mode,
        })
    return {
        "1": {"class_type": LOAD_VIDEO_NODE, "inputs": {
            "file": f"{source_subfolder}/{source_name}",
        }},
        "2": {"class_type": node_type, "inputs": node_inputs},
        "3": {"class_type": SAVE_VIDEO_NODE, "inputs": {
            "video": ["2", 0], "filename_prefix": filename_prefix,
            # ComfyUI v0.36's V3 SaveVideo schema requires the selected
            # container/codec combo in API-format workflows.
            "format": "auto",
            "format.codec": "h264",
        }},
    }


def _walk(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def find_video_descriptors(outputs: object) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for value in _walk(outputs):
        if not isinstance(value, Mapping):
            continue
        filename = value.get("filename", value.get("name"))
        subfolder = value.get("subfolder", "")
        kind = value.get("type", "output")
        if (
            isinstance(filename, str)
            and filename.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
            and isinstance(subfolder, str)
            and isinstance(kind, str)
        ):
            descriptor = {"filename": filename, "subfolder": subfolder, "type": kind}
            if descriptor not in found:
                found.append(descriptor)
    return found


def make_preview_url(base_url: str, descriptor: Mapping[str, str]) -> str:
    return f"{normalize_base_url(base_url)}/view?{urlencode(dict(descriptor))}"


def write_receipt(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _extract_node_report(outputs: object) -> dict[str, Any] | None:
    for value in _walk(outputs):
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("status") == "succeeded":
            return parsed
    return None


class ComfyUIBenchmarkCoordinator:
    """Use a running local ComfyUI server to dispatch and collect benchmark jobs."""

    def __init__(self, base_url: str, *, timeout: float) -> None:
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a positive finite number")
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout

    def _request(
        self, path: str, *, method: str = "GET", body: bytes | None = None,
        content_type: str | None = None, timeout: float | None = None,
    ) -> bytes:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                return response.read()
        except HTTPError as error:
            details = error.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ComfyUI {method} {path} returned HTTP {error.code}: {details}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Cannot reach local ComfyUI at {self.base_url}: {error}") from error

    def _json(
        self, path: str, *, method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        raw = self._request(
            path,
            method=method,
            body=body,
            content_type="application/json" if body is not None else None,
        )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"ComfyUI {path} returned a non-object response")
        return value

    def preflight(self, profiles: tuple[str, ...]) -> None:
        node_types = [LOAD_VIDEO_NODE, SAVE_VIDEO_NODE]
        node_types.extend(PROFILE_NODES[profile] for profile in profiles)
        for node_type in node_types:
            info = self._json(f"/object_info/{node_type}")
            if node_type not in info:
                raise RuntimeError(f"Local ComfyUI is missing the {node_type} custom node")

    def upload_source(self, source: Path, *, subfolder: str) -> dict[str, str]:
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError("source video must be a nonempty file")
        if source.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("source video exceeds the 512 MiB benchmark upload limit")
        filename = f"h3-benchmark-{uuid.uuid4().hex[:12]}.mp4"
        boundary = uuid.uuid4().hex
        content = source.read_bytes()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            "Content-Type: video/mp4\r\n\r\n"
        ).encode() + content + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        result = json.loads(self._request(
            "/upload/image",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        ))
        if not isinstance(result, dict) or result.get("name") != filename or result.get("type") != "input":
            raise RuntimeError("ComfyUI did not confirm the benchmark video upload")
        remote_folder = result.get("subfolder", subfolder)
        if not isinstance(remote_folder, str):
            raise TypeError("ComfyUI returned an invalid uploaded subfolder")
        return {"filename": filename, "subfolder": remote_folder}

    def probe_preview(self, url: str) -> dict[str, Any]:
        """Verify the saved MP4 is reachable as browser-playable streamed bytes."""
        expected_prefix = self.base_url + "/view?"
        if not url.startswith(expected_prefix):
            raise ValueError("preview URL must point at this local ComfyUI /view route")
        request = Request(url, headers={"Range": "bytes=0-31", "Accept": "video/*"})
        try:
            with urlopen(request, timeout=min(self.timeout, 20.0)) as response:
                content_type = response.headers.get_content_type()
                sample = response.read(32)
                status = response.status
                content_range = response.headers.get("Content-Range")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"ComfyUI could not stream the benchmark preview: {error}") from error
        if status not in {200, 206} or not content_type.startswith("video/") or not sample:
            raise RuntimeError(
                f"ComfyUI preview is not browser-playable: HTTP {status}, {content_type}"
            )
        return {
            "status": status,
            "content_type": content_type,
            "range_supported": status == 206,
            "content_range": content_range,
            "sample_bytes": len(sample),
        }

    def run_workflow(self, workflow: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        workflow_started = time.perf_counter()
        submitted_at_ms = round(time.time() * 1000)
        client_id = uuid.uuid4().hex
        ack = self._json("/prompt", method="POST", payload={
            "prompt": workflow,
            "client_id": client_id,
        })
        acknowledged_at_ms = round(time.time() * 1000)
        prompt_id = ack.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"ComfyUI rejected the benchmark workflow: {ack}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._json(f"/history/{prompt_id}")
            entry = result.get(prompt_id)
            if isinstance(entry, Mapping):
                status = entry.get("status", {})
                if isinstance(status, Mapping) and status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI job failed: {status.get('messages', status)}")
                if isinstance(status, Mapping) and status.get("completed") is True:
                    outputs = entry.get("outputs", {})
                    videos = find_video_descriptors(outputs)
                    if not videos:
                        raise RuntimeError("ComfyUI completed the job without saving a video")
                    timestamps: dict[str, int] = {}
                    messages = status.get("messages", [])
                    if isinstance(messages, list):
                        for message in messages:
                            if not isinstance(message, list) or len(message) != 2:
                                continue
                            name, details = message
                            timestamp = details.get("timestamp") if isinstance(details, Mapping) else None
                            if (isinstance(name, str) and name in {"execution_start", "execution_success"}
                                    and type(timestamp) is int and name not in timestamps):
                                timestamps[name] = timestamp
                    server_execution = None
                    if {"execution_start", "execution_success"} <= timestamps.keys():
                        elapsed_ms = timestamps["execution_success"] - timestamps["execution_start"]
                        if elapsed_ms >= 0:
                            server_execution = elapsed_ms / 1000
                    workflow_wall = time.perf_counter() - workflow_started
                    history_received_at_ms = round(time.time() * 1000)
                    execution_started_at_ms = timestamps.get("execution_start")
                    execution_succeeded_at_ms = timestamps.get("execution_success")
                    return {
                        "prompt_id": prompt_id,
                        "videos": videos,
                        "node_report": _extract_node_report(outputs),
                        "comfyui_execution_seconds": server_execution,
                        "comfyui_execution_started_at_epoch_ms": execution_started_at_ms,
                        "comfyui_execution_succeeded_at_epoch_ms": execution_succeeded_at_ms,
                        "prompt_submitted_at_epoch_ms": submitted_at_ms,
                        "prompt_acknowledged_at_epoch_ms": acknowledged_at_ms,
                        "history_completed_at_epoch_ms": history_received_at_ms,
                        "queue_wait_estimate_seconds": (
                            (execution_started_at_ms - acknowledged_at_ms) / 1000
                            if execution_started_at_ms is not None else None
                        ),
                        "history_after_execution_estimate_seconds": (
                            (history_received_at_ms - execution_succeeded_at_ms) / 1000
                            if execution_succeeded_at_ms is not None else None
                        ),
                        "wall_outside_comfyui_execution_seconds": (
                            max(0.0, workflow_wall - server_execution)
                            if server_execution is not None else None
                        ),
                    }
            time.sleep(1.0)
        raise TimeoutError(f"ComfyUI benchmark job {prompt_id} exceeded {timeout:g} seconds")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    modes = {
        "single": ("single",), "multi": ("multi",), "both": ("single", "multi"),
        "shared-vae": ("shared_vae",), "all": ("single", "multi", "shared_vae"),
    }
    if args.mode not in modes:
        raise ValueError("unknown benchmark mode")
    profiles = modes[args.mode]
    device_value = args.devices or ("1,2" if "shared_vae" in profiles else "0,1")
    requested_devices = tuple(part.strip() for part in device_value.split(",") if part.strip())
    skipped_profiles: dict[str, str] = {}
    if args.mode == "all" and args.devices is not None and len(requested_devices) == 1:
        profiles = ("single", "shared_vae")
        skipped_profiles["multi"] = "Requires two transformer GPUs; one was selected for the shared-VAE profile"
    devices = (
        parse_devices(device_value)
        if any(profile == "multi" for profile in profiles)
        else requested_devices if "shared_vae" in profiles else ()
    )
    vae_device = getattr(args, "vae_device", "0")
    shared_chunk_frames = getattr(args, "shared_vae_chunk_frames", 96)
    if "shared_vae" in profiles:
        validate_shared_vae(devices, vae_device, shared_chunk_frames, args.overlap_frames, args.decoder_mode)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    coordinator = ComfyUIBenchmarkCoordinator(args.comfyui_base_url, timeout=args.timeout)
    run_id = getattr(args, "run_id", uuid.uuid4().hex[:12])
    record: dict[str, Any] = {
        "schema": "sol_h3_upscaler/h3-upscale-benchmark/1",
        "run_id": run_id,
        "status": "preflight",
        "source_name": source.name,
        "source_bytes": source.stat().st_size,
        "comfyui_base_url": coordinator.base_url,
        "profiles": {},
        "receipt_path": str(args.output) if getattr(args, "output", None) else None,
    }
    if skipped_profiles:
        record["skipped_profiles"] = skipped_profiles
    receipt_path = getattr(args, "output", None)
    if isinstance(receipt_path, Path):
        write_receipt(receipt_path, record)

    coordinator.preflight(profiles)
    upload_started = time.perf_counter()
    uploaded = coordinator.upload_source(
        source,
        subfolder=f"sol_h3_upscaler-benchmarks/{run_id}",
    )
    upload_seconds = time.perf_counter() - upload_started
    benchmark_started = time.perf_counter()
    record.update(status="running", source_upload_seconds=upload_seconds)
    if isinstance(receipt_path, Path):
        write_receipt(receipt_path, record)

    # Finish every baseline run before starting the split runs, so each profile
    # warms its own persistent process pool instead of repeatedly evicting it.
    for profile in profiles:
        profile_runs: list[dict[str, Any]] = []
        for iteration in range(args.repeats):
            workflow = build_workflow(
                profile=profile,
                source_name=uploaded["filename"],
                source_subfolder=uploaded["subfolder"],
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                seed=args.seed,
                decoder_seed=args.decoder_seed,
                decoder_mode=args.decoder_mode,
                output_backend=args.output_backend,
                transformer_precision=args.transformer_precision,
                devices=devices,
                overlap_frames=args.overlap_frames,
                vae_device=vae_device,
                shared_vae_chunk_frames=shared_chunk_frames,
                shared_vae_transfer_mode=getattr(args, "shared_vae_transfer_mode", "peer_event"),
                filename_prefix=f"SoL/H3Benchmark/{run_id}/{profile}-{iteration + 1:02d}",
            )
            started = time.perf_counter()
            try:
                result = coordinator.run_workflow(workflow, timeout=args.timeout)
            except Exception as error:
                record.update(status="failed", failed_profile=profile,
                              failed_iteration=iteration + 1, error=str(error))
                record["profiles"][profile] = profile_runs
                if isinstance(receipt_path, Path):
                    write_receipt(receipt_path, record)
                raise
            wall_seconds = time.perf_counter() - started
            output_videos = [
                {
                    **descriptor,
                    "preview_url": make_preview_url(coordinator.base_url, descriptor),
                }
                for descriptor in result["videos"]
            ]
            for item in output_videos:
                item["stream_probe"] = coordinator.probe_preview(item["preview_url"])
                node_report = result.get("node_report")
                distributed = (
                    node_report.get("shared_vae_benchmark", node_report.get("multi_gpu_benchmark"))
                    if isinstance(node_report, Mapping) else None
                )
                if isinstance(distributed, Mapping):
                    chunks = distributed.get("chunks")
                    if isinstance(chunks, list):
                        item["seam_review"] = [
                            {
                                "after_frame": chunk["core_frame_range"][1],
                                "inspect_from_frame": max(
                                    0,
                                    int(chunk["core_frame_range"][1])
                                    - int(distributed.get("temporal_overlap_frames", 0)),
                                ),
                                "inspect_through_frame": min(
                                    int(distributed.get("source_frames", 0)),
                                    int(chunk["core_frame_range"][1])
                                    + int(distributed.get("temporal_overlap_frames", 0)),
                                ),
                            }
                            for chunk in chunks[:-1]
                            if isinstance(chunk, Mapping)
                            and isinstance(chunk.get("core_frame_range"), list)
                            and len(chunk["core_frame_range"]) == 2
                        ]
            profile_run = {
                "iteration": iteration + 1,
                "wall_seconds": wall_seconds,
                "comfyui_execution_seconds": result.get("comfyui_execution_seconds"),
                "wall_outside_comfyui_execution_seconds": result.get(
                    "wall_outside_comfyui_execution_seconds"
                ),
                "queue_wait_estimate_seconds": result.get("queue_wait_estimate_seconds"),
                "history_after_execution_estimate_seconds": result.get(
                    "history_after_execution_estimate_seconds"
                ),
                "comfyui_execution_started_at_epoch_ms": result.get(
                    "comfyui_execution_started_at_epoch_ms"
                ),
                "comfyui_execution_succeeded_at_epoch_ms": result.get(
                    "comfyui_execution_succeeded_at_epoch_ms"
                ),
                "prompt_submitted_at_epoch_ms": result.get("prompt_submitted_at_epoch_ms"),
                "prompt_acknowledged_at_epoch_ms": result.get("prompt_acknowledged_at_epoch_ms"),
                "history_completed_at_epoch_ms": result.get("history_completed_at_epoch_ms"),
                "prompt_id": result["prompt_id"],
                "node_report": result["node_report"],
                "videos": output_videos,
            }
            profile_runs.append(profile_run)
            record["profiles"][profile] = profile_runs
            if isinstance(receipt_path, Path):
                write_receipt(receipt_path, record)
            print(json.dumps({
                "profile": profile,
                "iteration": iteration + 1,
                "wall_seconds": wall_seconds,
                "preview_urls": [item["preview_url"] for item in output_videos],
            }, sort_keys=True), flush=True)
        record["profiles"][profile] = profile_runs

    profile_metrics: dict[str, dict[str, Any]] = {}
    for profile, runs in record["profiles"].items():
        times = [item["wall_seconds"] for item in runs]
        profile_metrics[profile] = {
            "median_seconds": statistics.median(times),
            "first_call_seconds": times[0],
            "cold_start_seconds": times[0] if len(times) > 1 else None,
            "warm_median_seconds": statistics.median(times[1:]) if len(times) > 1 else None,
            "warm_repeats": len(times) - 1,
            "repeats": len(times),
        }
    record["profile_metrics"] = profile_metrics
    record["profile_metrics_scope"] = (
        "per-job submit-to-saved-video wall time; first call may include loading/compilation, "
        "but an already-warm server is not reset; warm medians exclude the first call"
    )
    record["relative_to_single"] = {}
    if "single" in profile_metrics:
        baseline = profile_metrics["single"]
        for profile, metrics in profile_metrics.items():
            if profile == "single":
                continue
            record["relative_to_single"][profile] = {
                "median_speedup": baseline["median_seconds"] / metrics["median_seconds"],
                "warm_median_speedup": (
                    baseline["warm_median_seconds"] / metrics["warm_median_seconds"]
                    if baseline["warm_median_seconds"] is not None and metrics["warm_median_seconds"] is not None
                    else None
                ),
            }

    if "single" in profiles and "multi" in profiles:
        single_times = [item["wall_seconds"] for item in record["profiles"]["single"]]
        multi_times = [item["wall_seconds"] for item in record["profiles"]["multi"]]
        # Each profile's first call can include loading and compiling its
        # persistent worker. Keep the historical all-run median, but expose
        # warm-only medians separately so short-clip timing is interpretable.
        single_warm_times = single_times[1:]
        multi_warm_times = multi_times[1:]
        single_warm_median = statistics.median(single_warm_times) if single_warm_times else None
        multi_warm_median = statistics.median(multi_warm_times) if multi_warm_times else None
        record["comparison"] = {
            "single_median_seconds": statistics.median(single_times),
            "multi_median_seconds": statistics.median(multi_times),
            "median_speedup": statistics.median(single_times) / statistics.median(multi_times),
            "median_scope": "all measured iterations; includes the first cold call per profile",
            "cold_start_seconds": {
                "single": single_times[0] if len(single_times) > 1 else None,
                "multi": multi_times[0] if len(multi_times) > 1 else None,
            },
            "single_warm_median_seconds": single_warm_median,
            "multi_warm_median_seconds": multi_warm_median,
            "warm_median_speedup": (
                single_warm_median / multi_warm_median
                if single_warm_median is not None and multi_warm_median is not None
                else None
            ),
            "warm_repeats_per_profile": min(len(single_warm_times), len(multi_warm_times)),
            "repeats_per_profile": args.repeats,
            "quality_review_required": True,
            "timing_scope": "per-job submit-to-saved-video wall time; the one-time source upload is reported separately",
        }
    previews = [
        video["preview_url"]
        for runs in record["profiles"].values()
        for run in runs
        for video in run["videos"]
    ]
    record["preview_urls"] = previews
    record["status"] = "complete"
    record["benchmark_wall_seconds"] = time.perf_counter() - benchmark_started
    if isinstance(receipt_path, Path):
        write_receipt(receipt_path, record)
    if args.open_results:
        for url in previews:
            webbrowser.open(url)
    return record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local constant-rate YUV420P H3 MP4")
    parser.add_argument("--comfyui-base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--prompt", required=True, help="identical prompt for all profiles")
    parser.add_argument("--mode", choices=("single", "multi", "both", "shared-vae", "all"), default="both")
    parser.add_argument("--devices", help="upscale CUDA ordinals; default 1,2 for shared-vae/all, else 0,1")
    parser.add_argument("--vae-device", default="0", help="dedicated VAE CUDA ordinal for shared-vae")
    parser.add_argument("--shared-vae-chunk-frames", type=int, default=96)
    parser.add_argument("--overlap-frames", type=int, choices=(16, 32, 48, 64), default=16)
    parser.add_argument("--shared-vae-transfer-mode", choices=("blocking", "peer_event"),
                        default="peer_event")
    parser.add_argument("--width", type=int, default=1184)
    parser.add_argument("--height", type=int, default=666)
    parser.add_argument("--seed", type=int, default=303000)
    parser.add_argument("--decoder-seed", type=int, default=20260826)
    parser.add_argument("--decoder-mode", choices=("pruna_fast", "diffusion"), default="pruna_fast")
    parser.add_argument("--output-backend", choices=("nvenc", "libx264"), default="nvenc")
    parser.add_argument("--transformer-precision", choices=("nvfp4", "mxfp8", "bf16"), default="nvfp4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--open-results", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output or (
        Path("benchmark-results") / "distributed-h3"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
    )
    args.output = output
    args.run_id = uuid.uuid4().hex[:12]
    try:
        run_benchmark(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        try:
            failed = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
        except (OSError, json.JSONDecodeError):
            failed = {}
        if not isinstance(failed, dict) or failed.get("run_id") != args.run_id:
            failed = {"schema": "sol_h3_upscaler/h3-upscale-benchmark/1", "run_id": args.run_id}
        failed.update(status="failed", error=str(error), receipt_path=str(output))
        write_receipt(output, failed)
        print(f"H3 upscale benchmark failed; receipt: {output}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "receipt": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
