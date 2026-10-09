"""Persistent JSON worker transport with process and exchange ownership."""

from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import threading
from pathlib import Path

from .protocol import WorkerRequest, WorkerResponse, parse_worker_response

_WORKER_ENVIRONMENT_IDENTITY_KEYS = (
    "SOL_H3_UPSCALER_COMPILE_TRANSFORMER",
    "SOL_H3_UPSCALER_VIDEO_ONLY_REFINER",
    "SOL_H3_UPSCALER_CPU_TEXT_ENCODER",
    "SOL_H3_UPSCALER_CPU_PREPROCESS_OVERLAP",
    "SOL_H3_UPSCALER_CONDITIONING_ARTIFACT",
    "SOL_H3_UPSCALER_PROMPT_CACHE",
    "SOL_H3_UPSCALER_PRUNA_CONTIGUOUS_OUTPUT",
    "SOL_H3_UPSCALER_COMPILE_PRUNA_DECODER",
    "SOL_H3_UPSCALER_PRUNA_OUTPUT_COPY_OPT",
    "SOL_H3_UPSCALER_PRUNA_DROP_ENCODER",
    "SOL_H3_UPSCALER_PRUNA_INFERENCE_MODE",
    "SOL_H3_UPSCALER_NVENC_PRESET",
    "SOL_H3_UPSCALER_NVENC_INPUT_FORMAT",
    "SOL_H3_UPSCALER_GPU_AMINMAX",
    "SOL_H3_UPSCALER_ASYNC_NVENC_STAGING",
    "SOL_H3_UPSCALER_HASH_NVENC_INPUT",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF",
    "HOME",
    "USERPROFILE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_XET_CACHE",
    "HF_ASSETS_CACHE",
    "HF_ENDPOINT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TORCH_HOME",
    "CUDA_HOME",
    "CUDA_PATH",
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONNOUSERSITE",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CMAKE_PREFIX_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
)


def worker_environment_fingerprint(environment: dict[str, str]) -> str:
    """Hash behavior-bearing child settings without recording credentials."""
    identity = [(key, environment.get(key)) for key in _WORKER_ENVIRONMENT_IDENTITY_KEYS]
    payload = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


class PersistentWorkerClient:
    """Own one child process and serialize each complete request/response exchange."""

    def __init__(
        self,
        signature: tuple[object, ...],
        command: list[str],
        environment: dict[str, str],
        log_path: Path,
        *,
        label: str,
    ) -> None:
        self.signature = signature
        self.log_path = log_path
        self.label = label
        self._exchange_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._log_handle = log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                bufsize=1,
                env=environment,
            )
        except BaseException:
            self._log_handle.close()
            raise

    def request(
        self,
        request: WorkerRequest,
        *,
        timeout: float,
    ) -> tuple[WorkerResponse, int, Path]:
        """Exchange one complete protocol message, disposing the child on bad I/O."""
        if not self._exchange_lock.acquire(timeout=timeout):
            self._dispose()
            raise TimeoutError(f"{self.label} request queue exceeded {timeout:g} seconds")
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError(f"{self.label} is closed")
                process = self.process
                if process.poll() is not None or process.stdin is None or process.stdout is None:
                    self._dispose()
                    raise RuntimeError(f"{self.label} stopped before its request")
                try:
                    process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as exc:
                    self._dispose()
                    raise RuntimeError(f"{self.label} connection failed") from exc
                stdout = process.stdout

            lines: queue.Queue[tuple[str | None, Exception | None]] = queue.Queue(maxsize=1)

            def read_response() -> None:
                try:
                    lines.put((stdout.readline(), None))
                except (OSError, ValueError) as exc:
                    lines.put((None, exc))

            reader = threading.Thread(
                target=read_response,
                name=f"sol-h3-reader-{self.label.replace(' ', '-')}",
                daemon=True,
            )
            reader.start()
            try:
                line, error = lines.get(timeout=timeout)
            except queue.Empty as exc:
                log = self._log_tail()
                self._dispose()
                reader.join(timeout=1)
                suffix = f" Worker log tail:\n{log}" if log else ""
                raise TimeoutError(
                    f"{self.label} did not respond within {timeout:g} seconds.{suffix}"
                ) from exc
            if error is not None:
                self._dispose()
                raise RuntimeError(f"{self.label} response read failed: {error}") from error
            if not line:
                log = self._log_tail()
                self._dispose()
                suffix = f": {log}" if log else ""
                raise RuntimeError(f"{self.label} exited without a response{suffix}")
            try:
                response = parse_worker_response(json.loads(line), expected_output=request["output"])
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._dispose()
                raise RuntimeError(f"{self.label} returned an invalid protocol response") from exc
            return response, process.pid, self.log_path
        finally:
            self._exchange_lock.release()

    def close(self) -> None:
        """Stop the child immediately so active and queued requests cannot reuse it."""
        self._dispose()

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-6000:]
        except OSError:
            return ""

    def _dispose(self) -> None:
        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                process = self.process
                stdin = process.stdin
                stdout = process.stdout
                if stdin is not None:
                    try:
                        stdin.close()
                    except (OSError, ValueError):
                        pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
            finally:
                if stdout is not None:
                    try:
                        stdout.close()
                    except (OSError, ValueError):
                        pass
                self._log_handle.close()


class WorkerClientRegistry:
    """Own clients by logical worker slot while allowing independent slots to run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clients: dict[str, PersistentWorkerClient] = {}

    def request(
        self,
        key: str,
        signature: tuple[object, ...],
        command: list[str],
        environment: dict[str, str],
        log_path: Path,
        request: WorkerRequest,
        *,
        timeout: float,
        label: str,
    ) -> tuple[WorkerResponse, int, Path]:
        with self._lock:
            client = self._clients.get(key)
            if (
                client is None
                or client.signature != signature
                or client.process.poll() is not None
                or client._closed
            ):
                if client is not None:
                    self._clients.pop(key, None)
                    client.close()
                client = PersistentWorkerClient(
                    signature, command, environment, log_path, label=label,
                )
                self._clients[key] = client
        try:
            return client.request(request, timeout=timeout)
        except (RuntimeError, TimeoutError, ValueError, OSError, TypeError):
            self._remove_if_current(key, client)
            raise

    def close(self, key: str) -> None:
        with self._lock:
            client = self._clients.pop(key, None)
            if client is not None:
                client.close()

    def close_matching(self, prefix: str) -> None:
        with self._lock:
            keys = tuple(key for key in self._clients if key.startswith(prefix))
            for key in keys:
                client = self._clients.pop(key, None)
                if client is not None:
                    client.close()

    def keys(self, prefix: str = "") -> tuple[str, ...]:
        with self._lock:
            return tuple(key for key in self._clients if key.startswith(prefix))

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._clients

    def _remove_if_current(self, key: str, client: PersistentWorkerClient) -> None:
        with self._lock:
            if self._clients.get(key) is client:
                self._clients.pop(key, None)
                client.close()


__all__ = ["WorkerClientRegistry"]
