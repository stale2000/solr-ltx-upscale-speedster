"""Exact, opt-in fixed-prompt conditioning without resident text models."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

ARTIFACT_ENV = "SOL_H3_UPSCALER_CONDITIONING_ARTIFACT"
MAX_ARTIFACT_BYTES = 64 * 1024**2
ENCODING_VERSION = 1


def conditioning_cache_path(identity: dict[str, object]) -> Path:
    """Persist exact conditioning under a prompt/model/runtime-owned cache key."""
    root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return root / "sol_h3_upscaler" / "h3-conditioning" / f"{key}.pt"


def conditioning_identity(
    prompt: str, model_id: str, revision: str, video_only: bool
) -> dict[str, object]:
    """One checkpoint revision pins tokenizer, text encoder and connectors too."""
    packages = {}
    for name in ("diffusers", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    return {
        "schema": ENCODING_VERSION,
        "encoding_sources": {
            name: hashlib.sha256(
                (Path(__file__).parent / "sol_refiner_h3" / name).read_bytes()
            ).hexdigest()
            for name in ("pipeline.py", "video_only.py")
        },
        "prompt": prompt,
        "model_id": model_id,
        "model_revision": revision,
        "text_encoder_revision": revision,
        "tokenizer_revision": revision,
        "connectors_revision": revision,
        "video_only": video_only,
        "torch": str(torch.__version__),
        "packages": packages,
        "max_sequence_length": 1024,
        "embedding_channels": 4096,
        "embedding_dtype": "torch.bfloat16",
    }


def _validate_tensors(embeddings: torch.Tensor, mask: torch.Tensor) -> None:
    if embeddings.shape != (1, 1024, 4096) or embeddings.dtype != torch.bfloat16:
        raise ValueError("fixed conditioning requires BF16 embeddings shaped (1, 1024, 4096)")
    if mask.shape != (1, 1024) or mask.dtype != torch.int64:
        raise ValueError("fixed conditioning requires int64 mask shaped (1, 1024)")
    if not torch.isfinite(embeddings).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("fixed conditioning contains invalid embeddings or mask values")


def _digest(identity: dict[str, object], embeddings: torch.Tensor, mask: torch.Tensor) -> str:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for tensor in (embeddings, mask):
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FixedConditioning:
    """Private templates, with an exact identity and independent per-call tensors."""

    def __init__(
        self, identity: dict[str, object], embeddings: torch.Tensor, mask: torch.Tensor
    ) -> None:
        embeddings = embeddings.detach().to("cpu").clone()
        mask = mask.detach().to("cpu").clone()
        _validate_tensors(embeddings, mask)
        self.identity = json.loads(json.dumps(identity))
        self.embeddings = embeddings
        self.mask = mask
        self.sha256 = _digest(identity, embeddings, mask)
        self.released_parameter_bytes = 0

    def check(self, expected: dict[str, object]) -> None:
        if self.identity != expected:
            raise ValueError(
                "Fixed conditioning prompt/model/runtime mismatch. Use dynamic prompt mode "
                "or restart with a separate artifact for the exact new prompt."
            )

    def tensors(self, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.embeddings = self.embeddings.to(device)
        self.mask = self.mask.to(device)
        return self.embeddings.clone(), self.mask.clone()

    def save(self, path: Path) -> None:
        """Atomic single-file write; contains only tensors and primitive metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save({
                    "identity": self.identity,
                    "embeddings": self.embeddings.to("cpu"),
                    "mask": self.mask.to("cpu"),
                    "sha256": self.sha256,
                }, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, expected: dict[str, object]) -> FixedConditioning:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("fixed conditioning artifact exceeds size limit")
        # Bound uncompressed storage as well as the archive size before torch.load.
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_ARTIFACT_BYTES:
                raise ValueError("fixed conditioning artifact exceeds expanded size limit")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "identity", "embeddings", "mask", "sha256"
        }:
            raise ValueError("invalid fixed conditioning artifact fields")
        if payload["identity"] != expected:
            raise ValueError("fixed conditioning artifact prompt/model/runtime mismatch")
        if not all(isinstance(payload[name], torch.Tensor) for name in ("embeddings", "mask")):
            raise ValueError("invalid fixed conditioning artifact tensors")
        state = cls(expected, payload["embeddings"], payload["mask"])
        if state.sha256 != payload["sha256"]:
            raise ValueError("fixed conditioning artifact checksum mismatch")
        return state


def prepare_fixed_conditioning(
    pipe: Any, path: Path, identity: dict[str, object], prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode once or use a validated artifact; remove only text components."""
    if identity.get("prompt") != prompt:
        raise ValueError("fixed conditioning identity must match the exact supplied prompt")
    state = getattr(pipe, "fixed_conditioning", None)
    if state is None:
        if path.exists():
            state = FixedConditioning.load(path, identity)
        else:
            with torch.no_grad():
                embeddings, mask = pipe._encode_text(prompt, pipe._execution_device)
            state = FixedConditioning(identity, embeddings, mask)
            # Persist successfully before removing the ability to encode new text.
            state.save(path)
            del embeddings, mask
        parameters = {
            id(parameter): parameter
            for component in (pipe.text_encoder, pipe.connectors)
            for parameter in component.parameters()
        }
        state.released_parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in parameters.values()
        )
        del parameters
        pipe.register_modules(text_encoder=None, tokenizer=None, connectors=None)
        pipe._prompt_cache.clear()
        pipe.fixed_conditioning = state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    state.check(identity)
    return state.tensors(pipe._execution_device)
