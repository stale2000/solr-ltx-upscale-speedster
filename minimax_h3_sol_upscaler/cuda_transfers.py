"""Request-scoped asynchronous peer copies for the shared-VAE worker."""

from __future__ import annotations

import time
from typing import Any


class CudaPeerTransfers:
    """Queue peer copies without blocking the host on unrelated GPU work.

    Each destination gets a side stream per source/destination pair. The caller
    must make the consuming stream wait on the returned event before using the
    copy. Source tensors stay alive until their copy event completes; copied
    tensors are recorded on the consuming stream to protect allocator reuse.
    """

    MODES = ("blocking", "peer_event")

    def __init__(self, torch: Any, mode: str = "peer_event") -> None:
        if mode not in self.MODES:
            raise ValueError(f"transfer mode must be one of {self.MODES}")
        self.torch = torch
        self.mode = mode
        self.streams: dict[tuple[int, int], Any] = {}
        self.pending_sources: list[tuple[Any, Any]] = []
        self.transfers: list[dict[str, Any]] = []

    def _reap(self) -> None:
        pending = []
        for event, source in self.pending_sources:
            if not event.query():
                pending.append((event, source))
        self.pending_sources = pending

    def enqueue(
        self,
        tensor: Any,
        destination: int,
        *,
        chunk_index: int,
        direction: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        source = tensor.device
        if source.type != "cuda" or source.index is None:
            raise ValueError("shared-VAE peer transfer requires a CUDA source tensor")
        source_device = int(source.index)
        destination = int(destination)
        if source_device == destination:
            raise ValueError("shared-VAE peer transfer requires distinct CUDA devices")
        self._reap()

        peer_access = bool(torch_can_access_peer(self.torch, source_device, destination))
        if not peer_access and self.mode == "peer_event":
            raise RuntimeError(
                f"CUDA peer access is unavailable from GPU {source_device} to GPU {destination}"
            )

        if self.mode == "blocking":
            started = time.perf_counter()
            self.torch.cuda.synchronize(source_device)
            result = tensor.to(f"cuda:{destination}", non_blocking=False)
            self.torch.cuda.synchronize(destination)
            seconds = time.perf_counter() - started
            transfer = {
                "chunk": chunk_index,
                "direction": direction,
                "source_device": source_device,
                "destination_device": destination,
                "bytes": tensor.numel() * tensor.element_size(),
                "seconds": seconds,
                "submit_seconds": seconds,
                "peer_access_available": peer_access,
                "bandwidth_mib_s": (
                    tensor.numel() * tensor.element_size() / (1024.0 * 1024.0) / seconds
                    if seconds > 0
                    else None
                ),
                "mechanism": "synchronized torch.Tensor.to control",
            }
            self.transfers.append(transfer)
            return result, None, transfer

        key = (source_device, destination)
        copy_stream = self.streams.get(key)
        if copy_stream is None:
            copy_stream = self.torch.cuda.Stream(device=destination)
            self.streams[key] = copy_stream

        source_stream = self.torch.cuda.current_stream(source_device)
        ready = self.torch.cuda.Event(enable_timing=False)
        ready.record(source_stream)
        copy_stream.wait_event(ready)

        started = time.perf_counter()
        copy_start = self.torch.cuda.Event(enable_timing=True)
        copy_done = self.torch.cuda.Event(enable_timing=True)
        with self.torch.cuda.device(destination), self.torch.cuda.stream(copy_stream):
            copy_start.record(copy_stream)
            result = tensor.to(f"cuda:{destination}", non_blocking=True)
            copy_done.record(copy_stream)
        submit_seconds = time.perf_counter() - started

        transfer = {
            "chunk": chunk_index,
            "direction": direction,
            "source_device": source_device,
            "destination_device": destination,
            "bytes": tensor.numel() * tensor.element_size(),
            "seconds": 0.0,
            "submit_seconds": submit_seconds,
            "peer_access_available": peer_access,
            "mechanism": "torch.Tensor.to on an event-ordered CUDA peer-copy stream",
        }
        self.transfers.append(transfer)
        self.pending_sources.append((copy_done, tensor))
        transfer["_copy_start"] = copy_start
        transfer["_copy_done"] = copy_done
        return result, copy_done, transfer

    def wait_on_consumer(self, tensor: Any, event: Any) -> None:
        """Order a consumer thread's stream after the copy and track its buffer."""
        if event is None:
            return
        device = tensor.device
        if device.type != "cuda" or device.index is None:
            raise ValueError("shared-VAE peer copy result must be on CUDA")
        with self.torch.cuda.device(int(device.index)):
            stream = self.torch.cuda.current_stream(int(device.index))
            stream.wait_event(event)
            tensor.record_stream(stream)

    def finish(self) -> list[dict[str, Any]]:
        """Resolve copy timings and release source buffers after queued copies."""
        for event, _source in self.pending_sources:
            if not event.query():
                event.synchronize()
        self._reap()
        for transfer in self.transfers:
            if "_copy_start" not in transfer:
                continue
            start = transfer.pop("_copy_start")
            done = transfer.pop("_copy_done")
            seconds = float(start.elapsed_time(done)) / 1000.0
            transfer["seconds"] = seconds
            transfer["bandwidth_mib_s"] = (
                transfer["bytes"] / (1024.0 * 1024.0) / seconds if seconds > 0 else None
            )
        return self.transfers


def torch_can_access_peer(torch: Any, source: int, destination: int) -> bool:
    """Check both directions because the worker copies tensors both ways."""
    return bool(
        torch.cuda.can_device_access_peer(source, destination)
        and torch.cuda.can_device_access_peer(destination, source)
    )
