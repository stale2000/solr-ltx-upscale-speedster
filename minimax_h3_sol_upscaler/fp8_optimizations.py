"""Low-precision transformer paths for the H3 upscaler."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

MIN_MXFP8_LINEAR_DIMENSION = 1024


def _select_mxfp8_linear(module: Any, _name: str, linear_type: type) -> bool:
    return (
        isinstance(module, linear_type)
        and min(int(cast(Any, module).in_features), int(cast(Any, module).out_features))
        >= MIN_MXFP8_LINEAR_DIMENSION
    )


def enable_transformer_mxfp8(transformer: Any) -> tuple[str, ...]:
    """Quantize large transformer Linear layers to dynamic-activation MXFP8.

    Norms, embeddings, connectors, the conditioning VAE, and Pruna decoder stay
    BF16. The path fails closed so reports cannot claim a quantized run when a
    required kernel is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 benchmarking requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("MXFP8 benchmarking requires an NVIDIA Blackwell GPU (SM 10.0+)")

    try:
        workflow = import_module("torchao.prototype.mx_formats.inference_workflow")
        config_type = workflow.MXDynamicActivationMXWeightConfig
        quantize_ = import_module("torchao.quantization").quantize_
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MXFP8 was requested, but the pinned TorchAO runtime is missing. "
            "Reinstall the H3 worker runtime with install_runtime.sh."
        ) from exc

    linear_type = torch.nn.Linear
    selected_names = tuple(
        name
        for name, module in transformer.named_modules()
        if _select_mxfp8_linear(module, name, linear_type)
    )
    if not selected_names:
        raise RuntimeError("MXFP8 selected no transformer Linear layers to quantize")

    config = config_type(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
    )
    quantize_(
        transformer,
        config,
        filter_fn=lambda module, name: _select_mxfp8_linear(module, name, linear_type),
    )
    return selected_names


def _select_nvfp4_linear(module: Any, _name: str, linear_type: type) -> bool:
    if not isinstance(module, linear_type):
        return False
    in_features = int(cast(Any, module).in_features)
    out_features = int(cast(Any, module).out_features)
    return (
        min(in_features, out_features) >= MIN_MXFP8_LINEAR_DIMENSION
        and in_features % 16 == 0
        and out_features % 16 == 0
    )


def enable_transformer_nvfp4(transformer: Any) -> tuple[str, ...]:
    """Quantize compatible transformer Linear weights and activations to NVFP4."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 benchmarking requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("NVFP4 requires NVIDIA Blackwell (SM 10.0+) or newer")
    try:
        workflow = import_module("torchao.prototype.mx_formats.inference_workflow")
        config_type = workflow.NVFP4DynamicActivationNVFP4WeightConfig
        quantize_ = import_module("torchao.quantization").quantize_
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "NVFP4 was requested, but the pinned TorchAO runtime is missing. "
            "Reinstall the H3 worker runtime with install_runtime.sh."
        ) from exc

    linear_type = torch.nn.Linear
    selected_names = tuple(
        name for name, module in transformer.named_modules()
        if _select_nvfp4_linear(module, name, linear_type)
    )
    if not selected_names:
        raise RuntimeError("NVFP4 selected no compatible transformer Linear layers")
    # MSLK's fused Triton kernel avoids the large eager activation-quantization
    # temporaries that exhausted 32 GiB on long clips at target resolution.
    config = config_type(use_triton_kernel=True)
    quantize_(
        transformer,
        config,
        filter_fn=lambda module, name: _select_nvfp4_linear(module, name, linear_type),
    )
    return selected_names
