"""Lazy model construction and cache ownership for the H3 refiner."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_ID = "Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3"
MODEL_REVISION = "4655337474c1c77fb52e9ac55111552e625e779c"
PYTHON_VERSION = (3, 12)
TORCH_VERSION = "2.14.0+cu130"
TORCH_CUDA_VERSION = "13.0"
WORKER_TORCH_VERSION = "2.13.0+cu130"
MODEL_RESIDENCY = "cuda"
DEFAULT_OUTPUT_WIDTH = 1184
DEFAULT_OUTPUT_HEIGHT = 666
VIDEO_ONLY_REFINER_ENV = "SOL_H3_UPSCALER_VIDEO_ONLY_REFINER"
CPU_TEXT_ENCODER_ENV = "SOL_H3_UPSCALER_CPU_TEXT_ENCODER"
CPU_PREPROCESS_OVERLAP_ENV = "SOL_H3_UPSCALER_CPU_PREPROCESS_OVERLAP"
_TILE = {
    "tile_sample_min_height": 768,
    "tile_sample_min_width": 768,
    "tile_sample_min_num_frames": 128,
    "tile_sample_stride_height": 512,
    "tile_sample_stride_width": 512,
    "tile_sample_stride_num_frames": 80,
}


def _pipeline_cache_key(
    model_id: str,
    optimize_decoder: bool,
    decoder_mode: str,
    transformer_compiled: bool,
    video_only_refiner: bool,
    cpu_text_encoder: bool,
    cpu_preprocess_overlap: bool,
    transformer_precision: str,
    conditioning_artifact: Path | None,
    conditioning_sha256: str | None,
) -> tuple[object, ...]:
    conditioning = (
        (str(conditioning_artifact), conditioning_sha256)
        if conditioning_artifact is not None else None
    )
    return (
        model_id, optimize_decoder, decoder_mode, transformer_compiled,
        video_only_refiner, cpu_text_encoder, cpu_preprocess_overlap,
        transformer_precision, conditioning,
    )


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


class ModelRuntime:
    """Own the resident pipeline and the lock that protects its cache identity."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.identity: tuple[object, ...] | None = None
        self.conditioning_source: tuple[str, str, str, tuple[int, int, int, int, int]] | None = None
        self.lock = threading.RLock()

    def load_pipeline(
        self,
        model_id: str,
        expected_torch: str = TORCH_VERSION,
        *,
        optimize_decoder: bool = True,
        decoder_mode: str = "diffusion",
        transformer_precision: str = "nvfp4",
        width: int = DEFAULT_OUTPUT_WIDTH,
        height: int = DEFAULT_OUTPUT_HEIGHT,
        conditioning_artifact: Path | None = None,
        fixed_prompt: str | None = None,
        transformer_flags: Callable[[str, int, int], bool],
        precision_mode: Callable[[str], str],
        experiment_flag: Callable[..., bool],
        validate_runtime: Callable[..., None],
        natten_backend_name: Callable[[], str],
        ensure_source_path: Callable[[], str],
        keep_pipeline_on_cuda: Callable[[Any], None],
    ) -> Any:
        """Load or reuse a pipeline while preserving the established cache tuple."""
        with self.lock:
            return self._load_pipeline(
                model_id,
                expected_torch,
                optimize_decoder=optimize_decoder,
                decoder_mode=decoder_mode,
                transformer_precision=transformer_precision,
                width=width,
                height=height,
                conditioning_artifact=conditioning_artifact,
                fixed_prompt=fixed_prompt,
                transformer_flags=transformer_flags,
                precision_mode=precision_mode,
                experiment_flag=experiment_flag,
                validate_runtime=validate_runtime,
                natten_backend_name=natten_backend_name,
                ensure_source_path=ensure_source_path,
                keep_pipeline_on_cuda=keep_pipeline_on_cuda,
            )

    def bind_conditioning_artifact(self, path: Path, sha256: str) -> bool:
        """Bind an artifact created by the first request without reloading models."""
        with self.lock:
            if self.pipeline is None or self.identity is None:
                return False
            state = getattr(self.pipeline, "fixed_conditioning", None)
            if getattr(state, "sha256", None) != sha256:
                raise RuntimeError(
                    "fixed conditioning artifact digest changed relative to resident tensors; retry",
                )
            artifact_identity = (str(path), sha256)
            identity = getattr(state, "identity", None)
            if not isinstance(identity, dict):
                raise TypeError("resident fixed conditioning has no validated identity")
            identity_json = json.dumps(identity, sort_keys=True)
            source_stat = _file_stat_identity(path)
            if (
                self.identity[-1] == artifact_identity
                and self.conditioning_source is not None
                and self.conditioning_source[0] == str(path)
                and self.conditioning_source[1] == identity_json
                and self.conditioning_source[3] == source_stat
            ):
                return True

            source_digest = _file_fingerprint(path)
            from .fixed_conditioning import FixedConditioning

            persisted = FixedConditioning.load(path, identity)
            if (
                persisted.sha256 != sha256
                or getattr(state, "sha256", None) != persisted.sha256
                or _file_fingerprint(path) != source_digest
                or _file_stat_identity(path) != source_stat
            ):
                raise RuntimeError(
                    "fixed conditioning artifact changed relative to resident tensors; retry",
                )
            if self.identity[-1] not in {None, (str(path), None), artifact_identity}:
                raise RuntimeError("fixed conditioning artifact path changed before binding; retry")
            self.conditioning_source = (
                str(path), identity_json, source_digest, source_stat,
            )
            if self.identity[-1] == artifact_identity:
                return True
            if self.identity[-1] in {None, (str(path), None)}:
                self.identity = (*self.identity[:-1], artifact_identity)
                return True
            return False

    def _remember_conditioning_source(
        self,
        path: Path | None,
        identity: dict[str, object] | None,
        expected_source_digest: str | None,
    ) -> None:
        if path is None or identity is None or not path.is_file():
            self.conditioning_source = None
            return
        source_stat = _file_stat_identity(path)
        source_digest = _file_fingerprint(path)
        if _file_stat_identity(path) != source_stat:
            raise RuntimeError("fixed conditioning artifact changed while fingerprinting; retry")
        if expected_source_digest is not None and source_digest != expected_source_digest:
            raise RuntimeError("fixed conditioning artifact changed while loading; restart the worker")
        self.conditioning_source = (
            str(path), json.dumps(identity, sort_keys=True), source_digest, source_stat,
        )

    def _load_pipeline(
        self,
        model_id: str,
        expected_torch: str,
        *,
        optimize_decoder: bool,
        decoder_mode: str,
        transformer_precision: str,
        width: int,
        height: int,
        conditioning_artifact: Path | None,
        fixed_prompt: str | None,
        transformer_flags: Callable[[str, int, int], bool],
        precision_mode: Callable[[str], str],
        experiment_flag: Callable[..., bool],
        validate_runtime: Callable[..., None],
        natten_backend_name: Callable[[], str],
        ensure_source_path: Callable[[], str],
        keep_pipeline_on_cuda: Callable[[Any], None],
    ) -> Any:
        if decoder_mode not in {"diffusion", "pruna_fast"}:
            raise ValueError("Unsupported decoder mode")
        transformer_compiled = transformer_flags(decoder_mode, width, height)
        transformer_precision = precision_mode(transformer_precision)
        video_only_refiner = experiment_flag(VIDEO_ONLY_REFINER_ENV, default=False)
        cpu_text_encoder = experiment_flag(CPU_TEXT_ENCODER_ENV, default=False)
        cpu_preprocess_overlap = experiment_flag(CPU_PREPROCESS_OVERLAP_ENV, default=False)
        if conditioning_artifact is not None and fixed_prompt is None:
            raise ValueError("conditioning artifact requires an exact prompt")
        cache_prefix = (
            model_id, optimize_decoder, decoder_mode, transformer_compiled,
            video_only_refiner, cpu_text_encoder, cpu_preprocess_overlap,
            transformer_precision,
        )
        fixed_conditioning = None
        conditioning_sha256 = None
        expected_conditioning_identity = None
        source_digest = None
        if conditioning_artifact is not None:
            prompt = fixed_prompt
            if prompt is None:
                raise ValueError("conditioning artifact requires an exact prompt")
            from .fixed_conditioning import FixedConditioning, conditioning_identity

            expected_conditioning_identity = conditioning_identity(
                prompt, model_id, MODEL_REVISION, video_only_refiner,
            )
            if conditioning_artifact.is_file():
                source_stat = _file_stat_identity(conditioning_artifact)
                source_digest = _file_fingerprint(conditioning_artifact)
                if _file_stat_identity(conditioning_artifact) != source_stat:
                    raise RuntimeError(
                        "fixed conditioning artifact changed while fingerprinting; retry",
                    )
                cached_state = getattr(self.pipeline, "fixed_conditioning", None)
                cached_sha256 = getattr(cached_state, "sha256", None)
                cached_identity = self.identity
                if (
                    self.pipeline is not None
                    and cached_identity is not None
                    and cached_identity[:-1] == cache_prefix
                    and cached_identity[-1] == (str(conditioning_artifact), cached_sha256)
                    and self.conditioning_source == (
                        str(conditioning_artifact),
                        json.dumps(expected_conditioning_identity, sort_keys=True),
                        source_digest,
                        source_stat,
                    )
                    and getattr(cached_state, "identity", None)
                    == expected_conditioning_identity
                ):
                    return self.pipeline
                fixed_conditioning = FixedConditioning.load(
                    conditioning_artifact, expected_conditioning_identity,
                )
                if _file_fingerprint(conditioning_artifact) != source_digest:
                    raise RuntimeError(
                        "fixed conditioning artifact changed while validating; restart the worker",
                    )
                conditioning_sha256 = fixed_conditioning.sha256
        cache_key = _pipeline_cache_key(
            model_id, optimize_decoder, decoder_mode, transformer_compiled,
            video_only_refiner, cpu_text_encoder, cpu_preprocess_overlap,
            transformer_precision, conditioning_artifact, conditioning_sha256,
        )
        if self.pipeline is not None and self.identity == cache_key:
            self._remember_conditioning_source(
                conditioning_artifact, expected_conditioning_identity, source_digest,
            )
            return self.pipeline
        if (
            self.pipeline is not None
            and conditioning_artifact is not None
            and self.identity == (*cache_key[:-1], None)
            and getattr(self.pipeline, "fixed_conditioning", None) is None
        ):
            if fixed_prompt is None:
                raise ValueError("conditioning artifact requires an exact prompt")
            from .fixed_conditioning import conditioning_identity, prepare_fixed_conditioning

            prepare_fixed_conditioning(
                self.pipeline,
                conditioning_artifact,
                conditioning_identity(
                    fixed_prompt, model_id, MODEL_REVISION, video_only_refiner,
                ),
                fixed_prompt,
            )
            self._remember_conditioning_source(
                conditioning_artifact, expected_conditioning_identity, source_digest,
            )
            self.identity = cache_key
            return self.pipeline
        if self.pipeline is not None:
            previous_pipeline = self.pipeline
            self.pipeline = None
            self.identity = None
            self.conditioning_source = None
            del previous_pipeline
            gc.collect()

        # Keep GPU and Diffusers imports inside construction so CPU hosts can
        # import the Comfy node and inspect its schema without model packages.
        import torch

        validate_runtime(torch, sys.version_info[:2], expected_torch=expected_torch)
        if not torch.cuda.is_available():
            raise RuntimeError("The NVIDIA H3 SoL Upscaler requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The NVIDIA H3 SoL Refiner requires BF16 support")
        try:
            ensure_source_path()
            import importlib

            importlib.import_module("sol_refiner_h3")
            from diffusers.pipelines import pipeline_loading_utils

            from .sol_refiner_h3 import SoLRefinerH3Pipeline
            from .sol_refiner_h3.attention import LocalNattenProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Could not load the vendored NVIDIA SoL H3 refiner. Install the pinned "
                "host requirements and set up the native-NATTEN worker with install_runtime.sh."
            ) from exc

        previous_encoder_library = pipeline_loading_utils.LOADABLE_CLASSES.get("sol_refiner_h3")
        pipeline_loading_utils.LOADABLE_CLASSES["sol_refiner_h3"] = {
            "ModelMixin": ["save_pretrained", "from_pretrained"]
        }
        try:
            pipeline = SoLRefinerH3Pipeline.from_pretrained(
                model_id,
                revision=MODEL_REVISION,
                torch_dtype=torch.bfloat16,
                **({"diffusion_decoder": None} if decoder_mode == "pruna_fast" else {}),
                **({"text_encoder": None, "tokenizer": None, "connectors": None}
                   if fixed_conditioning is not None else {}),
            )
        finally:
            if previous_encoder_library is None:
                pipeline_loading_utils.LOADABLE_CLASSES.pop("sol_refiner_h3", None)
            else:
                pipeline_loading_utils.LOADABLE_CLASSES["sol_refiner_h3"] = previous_encoder_library

        native_natten = pipeline.diffusion_decoder is None or natten_backend_name() == "libnatten"
        if not native_natten:
            raise RuntimeError(
                "Native NATTEN is required for the H3 SoL decoder. The PyTorch Flex "
                "fallback can allocate a 91 GiB attention mask and OOM."
            )
        pipeline.natten_backend = "not-used" if decoder_mode == "pruna_fast" else "libnatten"
        pipeline.cpu_preprocess_overlap = cpu_preprocess_overlap
        pipeline.cpu_text_encoder_enabled = cpu_text_encoder and fixed_conditioning is None
        pipeline.transformer_precision = transformer_precision
        pipeline.transformer_mxfp8_linear_names = ()
        pipeline.transformer_nvfp4_linear_names = ()
        pipeline.fixed_conditioning = fixed_conditioning
        pipeline.video_only_refiner = video_only_refiner
        pipeline.video_only_transformer_blocks = 0
        if video_only_refiner:
            if decoder_mode != "pruna_fast":
                raise RuntimeError("The video-only refiner is validated only for pruna_fast mode")
            from .sol_refiner_h3.video_only import enable_video_only_refiner

            pipeline.video_only_transformer_blocks = enable_video_only_refiner(
                pipeline.transformer,
            )
        if pipeline.diffusion_decoder is not None:
            pipeline.diffusion_decoder.set_attn_processor(LocalNattenProcessor())
            pipeline.diffusion_decoder.enable_tiling(**_TILE)
        pipeline.transformer_compiled = False
        if transformer_precision == "mxfp8":
            from .fp8_optimizations import enable_transformer_mxfp8

            pipeline.transformer_mxfp8_linear_names = enable_transformer_mxfp8(
                pipeline.transformer,
            )
        elif transformer_precision == "nvfp4":
            from .fp8_optimizations import enable_transformer_nvfp4

            pipeline.transformer_nvfp4_linear_names = enable_transformer_nvfp4(
                pipeline.transformer,
            )
        if pipeline.cpu_text_encoder_enabled:
            cpu_text_components = {
                "text_encoder": pipeline.text_encoder,
                "tokenizer": pipeline.tokenizer,
                "connectors": pipeline.connectors,
            }
            pipeline.register_modules(**{name: None for name in cpu_text_components})
            keep_pipeline_on_cuda(pipeline)
            pipeline.register_modules(**cpu_text_components)
        else:
            keep_pipeline_on_cuda(pipeline)
        if transformer_compiled:
            compile_blocks = getattr(pipeline.transformer, "compile_repeated_blocks", None)
            if not callable(compile_blocks):
                raise RuntimeError(
                    "The pinned LTX transformer does not support repeated-block compilation",
                )
            compile_blocks()
            pipeline.transformer_compiled = True
        pipeline.decoder_fused_rope_modules = 0
        if optimize_decoder and pipeline.diffusion_decoder is not None:
            from .decoder_optimizations import enable_decoder_rope_fusion

            pipeline.decoder_fused_rope_modules = enable_decoder_rope_fusion(
                pipeline.diffusion_decoder,
            )
        self._remember_conditioning_source(
            conditioning_artifact, expected_conditioning_identity, source_digest,
        )
        self.pipeline = pipeline
        self.identity = cache_key
        return pipeline


_RUNTIME = ModelRuntime()


__all__ = [
    "CPU_PREPROCESS_OVERLAP_ENV",
    "CPU_TEXT_ENCODER_ENV",
    "DEFAULT_OUTPUT_HEIGHT",
    "DEFAULT_OUTPUT_WIDTH",
    "MODEL_ID",
    "MODEL_RESIDENCY",
    "MODEL_REVISION",
    "PYTHON_VERSION",
    "TORCH_CUDA_VERSION",
    "TORCH_VERSION",
    "VIDEO_ONLY_REFINER_ENV",
    "WORKER_TORCH_VERSION",
    "_RUNTIME",
    "ModelRuntime",
]
