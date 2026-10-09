"""Refiner-scoped removal of LTX audio work unused by this video-only call.

These adapters target the model's explicitly pinned Diffusers revision. They
leave every learned parameter in place and execute the upstream video
operations in their original order. The source soundtrack is handled by the
separate FFmpeg mux path, not by this latent model branch.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch


def _video_only_block_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    audio_encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    temb_audio: torch.Tensor,
    temb_ca_scale_shift: torch.Tensor,
    temb_ca_audio_scale_shift: torch.Tensor,
    temb_ca_gate: torch.Tensor,
    temb_ca_audio_gate: torch.Tensor,
    temb_prompt: torch.Tensor | None = None,
    temb_prompt_audio: torch.Tensor | None = None,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ca_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ca_audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    audio_encoder_attention_mask: torch.Tensor | None = None,
    self_attention_mask: torch.Tensor | None = None,
    audio_self_attention_mask: torch.Tensor | None = None,
    a2v_cross_attention_mask: torch.Tensor | None = None,
    v2a_cross_attention_mask: torch.Tensor | None = None,
    use_a2v_cross_attention: bool = True,
    use_v2a_cross_attention: bool = True,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the upstream video branch and return untouched audio state."""
    if use_a2v_cross_attention or use_v2a_cross_attention:
        raise RuntimeError("The video-only LTX adapter requires isolated modalities")
    if perturbation_mask is not None or self_attention_mask is not None:
        raise RuntimeError("The video-only LTX adapter does not support STG or video masks")

    batch_size = hidden_states.size(0)
    video_ada_params = self.get_mod_params(self.scale_shift_table, temb, batch_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada_params[:6]
    if self.video_cross_attn_adaln:
        shift_text_q, scale_text_q, gate_text_q = video_ada_params[6:9]

    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    video_self_attn_args = {
        "hidden_states": norm_hidden_states,
        "encoder_hidden_states": None,
        "query_rotary_emb": video_rotary_emb,
        "attention_mask": None,
    }
    if self.perturbed_attn:
        video_self_attn_args["perturbation_mask"] = perturbation_mask
        video_self_attn_args["all_perturbed"] = all_perturbed
    attn_hidden_states = self.attn1(**video_self_attn_args)
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    if self.cross_attn_adaln:
        if temb_prompt is not None:
            shift_text_kv, scale_text_kv = self.get_mod_params(
                self.prompt_scale_shift_table, temb_prompt, batch_size
            )
        else:
            shift_text_kv, scale_text_kv = (
                self.prompt_scale_shift_table[None, None]
                .to(device=hidden_states.device, dtype=hidden_states.dtype)
                .unbind(dim=2)
            )

    norm_hidden_states = self.norm2(hidden_states)
    if self.video_cross_attn_adaln:
        norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q
    if self.cross_attn_adaln:
        encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv
    attn_hidden_states = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    if self.video_cross_attn_adaln:
        attn_hidden_states = attn_hidden_states * gate_text_q
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    ff_output = self.ff(norm_hidden_states)
    hidden_states = hidden_states + ff_output * gate_mlp
    return hidden_states, audio_hidden_states


def _video_only_connectors(
    connectors: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    padding_side: str = "left",
    scale_factor: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only LTX2TextConnectors' video output while preserving its inputs."""
    from diffusers.pipelines.ltx2.connectors import (
        per_layer_masked_mean_norm,
        per_token_rms_norm,
    )

    if hidden_states.ndim == 3:
        hidden_states = hidden_states.unflatten(
            2, (connectors.config.caption_channels, -1)
        )

    if connectors.config.per_modality_projections:
        normalized = per_token_rms_norm(hidden_states).flatten(2, 3)
        bool_mask = attention_mask.bool().unsqueeze(-1)
        normalized = torch.where(bool_mask, normalized, torch.zeros_like(normalized))
        normalized = normalized * (
            connectors.config.video_hidden_dim / connectors.config.caption_channels
        ) ** 0.5
        video_projection = connectors.video_text_proj_in(normalized)
    else:
        sequence_lengths = attention_mask.sum(dim=-1)
        normalized = per_layer_masked_mean_norm(
            text_hidden_states=hidden_states,
            sequence_lengths=sequence_lengths,
            device=hidden_states.device,
            padding_side=padding_side,
            scale_factor=scale_factor,
        )
        video_projection = connectors.text_proj_in(normalized)

    text_dtype = video_projection.dtype
    additive_mask = (attention_mask.to(torch.int64) - 1).to(text_dtype)
    additive_mask = additive_mask.reshape(
        additive_mask.shape[0], 1, -1, additive_mask.shape[-1]
    )
    additive_mask = additive_mask * torch.finfo(text_dtype).max
    video_embeddings, video_attention_mask = connectors.video_connector(
        video_projection, additive_mask
    )
    binary_mask = (video_attention_mask < 1e-6).to(torch.int64)
    binary_mask = binary_mask.reshape(
        video_embeddings.shape[0], video_embeddings.shape[1], 1
    )
    video_embeddings = video_embeddings * binary_mask
    return video_embeddings, binary_mask.squeeze(-1)


def enable_video_only_refiner(transformer: Any) -> int:
    """Install the isolated video path on the pinned repeated LTX blocks."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        raise RuntimeError("Pinned LTX transformer has no repeated transformer blocks")
    installed = 0
    for block in blocks:
        if getattr(block, "_sol_h3_upscaler_video_only", False):
            installed += 1
            continue
        if not all(
            hasattr(block, name)
            for name in (
                "get_mod_params", "norm1", "attn1", "norm2", "attn2",
                "norm3", "ff", "scale_shift_table", "video_cross_attn_adaln",
                "cross_attn_adaln",
            )
        ):
            raise RuntimeError("Installed Diffusers LTX block is incompatible with video-only adapter")
        block.forward = MethodType(_video_only_block_forward, block)
        block._sol_h3_upscaler_video_only = True
        installed += 1
    if installed != len(blocks):
        raise RuntimeError("Video-only adapter did not cover every LTX transformer block")
    return installed


__all__ = ["_video_only_connectors", "enable_video_only_refiner"]
