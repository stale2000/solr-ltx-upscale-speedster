"""Instance-local, arithmetic-preserving LTX-2.5 decoder optimizations."""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

type RopeCacheKey = tuple[tuple[int, ...], float, int, int, int, torch.device]
type RopeCache = dict[RopeCacheKey, list[torch.Tensor]]


class RopeEmbedding(Protocol):
    rope_dim_split: tuple[int, ...]
    base: float

    def _inv_freqs(self, axis_dim: int, device: torch.device) -> torch.Tensor: ...


@triton.jit
def _rope_kernel(
    X: Any, OUT: Any, CT: Any, ST: Any, CH: Any, SH: Any, CW: Any, SW: Any,
    COUNT: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HEADS: tl.constexpr, D: tl.constexpr, DT: tl.constexpr, DH: tl.constexpr,
    SB: tl.constexpr, SF: tl.constexpr, SY: tl.constexpr,
    SX: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    # One lane owns both members of a pair, allowing safe in-place writes:
    # no lane/program ever reads another lane/program's output addresses.
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)) * 2
    valid = i < COUNT
    dim = i % D
    head = i // D % HEADS
    x = i // (D * HEADS) % W
    y = i // (D * HEADS * W) % H
    frame = i // (D * HEADS * W * H) % T
    batch = i // (D * HEADS * W * H * T)
    base = batch * SB + frame * SF + y * SY + x * SX + head * SN
    pair = dim // 2 * 2
    even = tl.load(X + base + pair * SD, mask=valid, other=0).to(tl.float32)
    odd = tl.load(X + base + (pair + 1) * SD, mask=valid, other=0).to(tl.float32)
    temporal = dim < DT
    vertical = (dim >= DT) & (dim < DT + DH)
    horizontal = dim >= DT + DH
    ct = tl.load(CT + frame * (DT // 2) + dim // 2, mask=valid & temporal, other=0)
    st = tl.load(ST + frame * (DT // 2) + dim // 2, mask=valid & temporal, other=0)
    ch = tl.load(CH + y * (DH // 2) + (dim - DT) // 2, mask=valid & vertical, other=0)
    sh = tl.load(SH + y * (DH // 2) + (dim - DT) // 2, mask=valid & vertical, other=0)
    dw: tl.constexpr = D - DT - DH
    cw = tl.load(CW + x * (dw // 2) + (dim - DT - DH) // 2, mask=valid & horizontal, other=0)
    sw = tl.load(SW + x * (dw // 2) + (dim - DT - DH) // 2, mask=valid & horizontal, other=0)
    cosine = tl.where(temporal, ct, tl.where(vertical, ch, cw))
    sine = tl.where(temporal, st, tl.where(vertical, sh, sw))
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    tl.store(OUT + i, rotated_even, mask=valid)
    tl.store(OUT + i + 1, rotated_odd, mask=valid)


def fused_rope(
    rope: RopeEmbedding,
    hidden_states: torch.Tensor,
    cache: RopeCache,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Match eager FP32 rotation and final cast; support strided Q/K tensors."""
    if hidden_states.ndim != 6 or hidden_states.device.type != "cuda":
        raise ValueError("Fused decoder RoPE requires a CUDA (B,T,H,W,heads,dim) tensor")
    _batch, frames, height, width, heads, dim = hidden_states.shape
    split = tuple(rope.rope_dim_split)
    if sum(split) != dim or any(part <= 0 or part % 2 for part in split):
        raise ValueError("Unsupported decoder rotary dimension split")
    key = (split, rope.base, frames, height, width, hidden_states.device)
    tables = cache.get(key)
    if tables is None:
        tables = []
        for length, axis_dim in zip((frames, height, width), split):
            # Use the pinned reference's FP64 frequency creation and torch's
            # FP32 trigonometry. Cache metadata, never approximate the angles.
            inv = rope._inv_freqs(axis_dim, hidden_states.device)
            positions = torch.arange(length, dtype=torch.float32, device=hidden_states.device)
            angles = positions[:, None] * inv[None, :]
            tables.extend((angles.cos(), angles.sin()))
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
        cache[key] = tables
    if inplace and (not hidden_states.is_contiguous() or hidden_states.requires_grad):
        raise ValueError("In-place RoPE requires a contiguous, privately owned inference tensor")
    output = hidden_states if inplace else torch.empty(
        hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype
    )
    _rope_kernel[(triton.cdiv(hidden_states.numel() // 2, 256),)](
        hidden_states, output, *tables,
        hidden_states.numel(), frames, height, width, heads, dim, split[0], split[1],
        *hidden_states.stride(), BLOCK=256, num_warps=4,
        # Eager evaluates products and add/subtract in separate FP32 kernels.
        # FMA contraction would change rounding, so explicitly disable it.
        enable_fp_fusion=False,
    )
    return output


def enable_decoder_rope_fusion(diffusion_decoder: torch.nn.Module) -> int:
    """Fuse only the reference decoder's rotary operations, preserving NATTEN."""
    cached_count = getattr(diffusion_decoder, "_sol_h3_upscaler_rope_count", None)
    if isinstance(cached_count, int):
        return cached_count
    cache: RopeCache = {}
    count = 0

    def optimized_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        return fused_rope(self, hidden_states, cache)

    for module in diffusion_decoder.modules():
        if type(module).__name__ != "LTX2VideoVaeRotaryPosEmbed3D":
            continue
        if not hasattr(module, "_sol_h3_upscaler_original_rope_forward"):
            patch_target = cast(Any, module)
            patch_target._sol_h3_upscaler_original_rope_forward = module.forward
            patch_target.forward = MethodType(optimized_forward, module)
        count += 1
    if not count:
        raise RuntimeError("No compatible LTX-2.5 decoder rotary modules were found")
    patch_target = cast(Any, diffusion_decoder)
    patch_target._sol_h3_upscaler_rope_cache = cache
    patch_target._sol_h3_upscaler_rope_count = count
    return count


def enable_decoder_qk_inplace(diffusion_decoder: torch.nn.Module) -> int:
    """Reuse privately owned normalized Q/K buffers, as in the FastH3 decoder."""
    cache: RopeCache = {}

    def project_qkv(
        self: Any, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*hidden_states.shape[:-1], self.heads, self.head_dim)
        query = self.norm_q(self.to_q(hidden_states).view(shape))
        key = self.norm_k(self.to_k(hidden_states).view(shape))
        value = self.to_v(hidden_states).view(shape)
        # RMSNorm returns new owned buffers. Keep the reference BF16 scale
        # rounding, then rotate each pair in place without full-size copies.
        query.mul_(self.scale)
        return (fused_rope(self.rope, query, cache, inplace=True),
                fused_rope(self.rope, key, cache, inplace=True), value)

    count = 0
    for module in diffusion_decoder.modules():
        if type(module).__name__ != "LTX2VideoVaeNeighborhoodAttention":
            continue
        patch_target = cast(Any, module)
        patch_target.project_qkv = MethodType(project_qkv, module)
        count += 1
    if not count:
        raise RuntimeError("No compatible LTX-2.5 decoder attention modules were found")
    return count


def enable_decoder_mlp_tiling(diffusion_decoder: torch.nn.Module, tile_tokens: int) -> int:
    """Bound pointwise MLP work by tokens; retain the reference weights/ops."""
    if tile_tokens < 1:
        raise ValueError("Decoder MLP tile token count must be positive")

    def optimized_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, *token_dims, channels = hidden_states.shape
        count = math.prod(token_dims)
        if count <= tile_tokens:
            return self.w_down(F.silu(self.w_gate(hidden_states)) * self.w_up(hidden_states))
        flat = hidden_states.reshape(batch_size, count, channels)
        output = torch.empty_like(flat)
        for start in range(0, count, tile_tokens):
            tile = flat[:, start:start + tile_tokens]
            output[:, start:start + tile_tokens] = self.w_down(
                F.silu(self.w_gate(tile)) * self.w_up(tile)
            )
        return output.reshape(hidden_states.shape)

    count = 0
    for module in diffusion_decoder.modules():
        if type(module).__name__ != "LTX2VideoVaeSwiGLU":
            continue
        module.forward = MethodType(optimized_forward, module)
        count += 1
    if not count:
        raise RuntimeError("No compatible LTX-2.5 decoder MLP modules were found")
    return count
