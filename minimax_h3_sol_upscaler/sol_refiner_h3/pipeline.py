"""One-step H3 refinement using standard Diffusers LTX-2 components."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Literal

import torch
from diffusers import (
    AutoencoderKLLTX2Video,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    LTX2ConditionPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
    LTX2VideoDiffusionDecodePipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from minimax_h3_sol_upscaler.prompt_cache import PromptEmbeddingCache

DEFAULT_SIGMA = 0.9093750119
CPU_PREPROCESS_OVERLAP_ENV = "SOL_H3_UPSCALER_CPU_PREPROCESS_OVERLAP"
VIDEO_ONLY_REFINER_ENV = "SOL_H3_UPSCALER_VIDEO_ONLY_REFINER"
PROMPT_CACHE_ENV = "SOL_H3_UPSCALER_PROMPT_CACHE"
CPU_TEXT_ENCODER_ENV = "SOL_H3_UPSCALER_CPU_TEXT_ENCODER"


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1'")
    return value == "1"


def output_geometry(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """Return the x2 conditioning canvas and valid LTX temporal length."""
    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("width, height and frame count must be positive")
    return (
        math.ceil(width / 64) * 64,
        math.ceil(height / 64) * 64,
        1 + (frames - 1) // 8 * 8,
    )


class SoLRefinerH3Pipeline(DiffusionPipeline):
    """Refine H3 video in one denoising step using Diffusers LTX-2.5 components."""

    model_cpu_offload_seq = "text_encoder->connectors->vae->latent_upsampler->transformer->diffusion_decoder"
    _optional_components: ClassVar[list[str]] = [
        "text_encoder", "tokenizer", "connectors", "diffusion_decoder",
        "vae", "transformer", "latent_upsampler",
    ]

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        transformer: LTX2VideoTransformer3DModel,
        latent_upsampler: LTX2LatentUpsamplerModel,
        diffusion_decoder: LTX2VideoDiffusionDecoderModel | None,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        connectors: LTX2TextConnectors | None = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            latent_upsampler=latent_upsampler,
            diffusion_decoder=diffusion_decoder,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
        )
        self.cpu_preprocess_overlap = _env_enabled(
            CPU_PREPROCESS_OVERLAP_ENV, False
        )
        self.video_only_refiner = _env_enabled(VIDEO_ONLY_REFINER_ENV, False)
        self.prompt_cache_enabled = _env_enabled(PROMPT_CACHE_ENV, True)
        self.cpu_text_encoder_enabled = _env_enabled(CPU_TEXT_ENCODER_ENV, False)
        self._prompt_cache = PromptEmbeddingCache()
        self.last_prompt_cache_hit = False

    def _encode_text(
        self, prompt: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.text_encoder is None
            or self.tokenizer is None
            or self.connectors is None
        ):
            raise ValueError(
                "Provide prompt embeddings or load all text-conditioning components"
            )
        self.last_prompt_cache_hit = False
        encoding_device = torch.device("cpu") if self.cpu_text_encoder_enabled else device
        cache_key: tuple[object, ...] | None = None
        if self.prompt_cache_enabled:
            cache_key = (
                prompt,
                str(encoding_device),
                str(self.text_encoder.dtype),
                str(self.connectors.dtype),
                str(self.transformer.dtype),
                id(self.text_encoder),
                id(self.tokenizer),
                id(self.connectors),
                self.video_only_refiner,
            )
            cached = self._prompt_cache.get(cache_key)
            if cached is not None:
                self.last_prompt_cache_hit = True
                return cached
        text_pipeline = LTX2ConditionPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=None,
        )
        embeddings, mask, _, _ = text_pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=1024,
            device=encoding_device,
        )
        if self.video_only_refiner:
            from .video_only import _video_only_connectors

            embeddings, mask = _video_only_connectors(
                self.connectors,
                embeddings,
                mask,
                padding_side=self.tokenizer.padding_side,
            )
        else:
            embeddings, _, mask = self.connectors(
                embeddings, mask, padding_side=self.tokenizer.padding_side
            )
        if cache_key is not None:
            self._prompt_cache.put(cache_key, embeddings, mask)
        return embeddings, mask

    @torch.no_grad()
    def denoise_latents(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        sigma: float = DEFAULT_SIGMA,
        frame_rate: float = 24.0,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform exactly one conditional-only velocity prediction and Euler step."""
        if not 0 < sigma <= 1 or frame_rate <= 0:
            raise ValueError("sigma must be in (0, 1] and frame_rate must be positive")
        if (
            noisy_latents.ndim != 5
            or noisy_latents.shape[1] != self.transformer.config.in_channels
        ):
            raise ValueError(
                "Expected [batch, latent_channels, frames, height, width] latents"
            )
        if (
            self.scheduler.config.use_dynamic_shifting
            or self.scheduler.config.shift != 1
        ):
            raise ValueError(
                "The one-step checkpoint requires an unshifted FlowMatch Euler schedule"
            )
        device = self._execution_device
        dtype = self.transformer.dtype
        x = noisy_latents.to(device=device, dtype=dtype)
        batch, channels, frames, height, width = x.shape
        packed = LTX2ConditionPipeline._pack_latents(x, 1, 1)
        self.scheduler.set_timesteps(
            num_inference_steps=1, sigmas=[sigma], device=device
        )
        if (
            len(self.scheduler.timesteps) != 1
            or len(self.scheduler.sigmas) != 2
            or self.scheduler.sigmas[-1].item() != 0
            or not torch.isclose(
                self.scheduler.sigmas[0], torch.tensor(sigma, device=device), atol=1e-6
            )
        ):
            raise ValueError("Scheduler altered the checkpoint's one-step sigma")
        timestep = self.scheduler.timesteps[0].expand(batch)
        # Diffusers prompt AdaLN accepts the scaled timestep, not raw sigma.
        velocity, _ = self.transformer(
            hidden_states=packed,
            audio_hidden_states=torch.zeros(
                batch,
                1,
                self.transformer.config.audio_in_channels,
                device=device,
                dtype=dtype,
            ),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
            audio_encoder_hidden_states=torch.zeros(
                batch,
                1,
                self.transformer.config.audio_cross_attention_dim,
                device=device,
                dtype=dtype,
            ),
            timestep=timestep[:, None].expand(batch, packed.shape[1]),
            audio_timestep=timestep,
            sigma=timestep,
            audio_sigma=timestep,
            encoder_attention_mask=(
                prompt_attention_mask.to(device)
                if prompt_attention_mask is not None
                else None
            ),
            num_frames=frames,
            height=height,
            width=width,
            fps=frame_rate,
            audio_num_frames=1,
            isolate_modalities=True,
            return_dict=False,
        )
        denoised = self.scheduler.step(
            velocity.float(),
            self.scheduler.timesteps[0],
            packed.float(),
            return_dict=False,
        )[0]
        return (
            denoised.reshape(batch, frames, height, width, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    @torch.no_grad()
    def encode_conditioning(
        self, video: list[Any], *, width: int, height: int,
        dtype: torch.dtype | None = None,
        stage_timings: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Encode and upsample deterministic normalized conditioning on the VAE device.

        A role-specific pipeline may omit the transformer entirely. The caller
        transfers this compact tensor to a transformer pipeline afterwards.
        """
        if self.vae is None or self.latent_upsampler is None:
            raise ValueError("Conditioning requires the VAE and latent upsampler")
        canvas_width, canvas_height, frames = output_geometry(width, height, len(video))
        device = next(self.vae.parameters()).device
        upsample = LTX2LatentUpsamplePipeline(
            vae=self.vae, latent_upsampler=self.latent_upsampler
        )
        pixels = upsample.video_processor.preprocess_video(
            video[:frames], height=canvas_height // 2, width=canvas_width // 2
        ).to(device=device, dtype=self.vae.dtype)
        if stage_timings is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        encoded = self.vae.encode(pixels).latent_dist.mode()
        if stage_timings is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_timings["vae_encode_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        latents = upsample(
            latents=encoded, width=canvas_width // 2, height=canvas_height // 2,
            num_frames=frames, latents_normalized=False, output_type="latent",
        ).frames
        if stage_timings is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_timings["latent_upsample_s"] = time.perf_counter() - tick
        return LTX2ConditionPipeline._normalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(device=device, dtype=dtype or self.vae.dtype)

    @torch.no_grad()
    def refine_conditioned_latents(
        self, latents: torch.Tensor, prompt_embeds: torch.Tensor, *,
        sigma: float = DEFAULT_SIGMA, frame_rate: float = 24.0,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized refined latents without loading or invoking a VAE."""
        if self.transformer is None:
            raise ValueError("Refinement requires a transformer")
        latents = latents.to(
            device=self._execution_device, dtype=self.transformer.dtype
        )
        if noise is None:
            noise = randn_tensor(
                latents.shape, generator=generator,
                device=latents.device, dtype=latents.dtype,
            )
        elif noise.shape != latents.shape:
            raise ValueError("Provided noise must match the conditioning latent shape")
        else:
            noise = noise.to(device=latents.device, dtype=latents.dtype)
        noisy = ((1 - sigma) * latents.float() + sigma * noise.float()).to(latents.dtype)
        return self.denoise_latents(
            noisy, prompt_embeds, sigma=sigma, frame_rate=frame_rate,
            prompt_attention_mask=prompt_attention_mask,
        )

    @torch.no_grad()
    def denormalize_conditioned_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert the transformer's normalized output for the Pruna decoder."""
        if self.vae is None:
            raise ValueError("Latent denormalization requires VAE normalization statistics")
        return LTX2ConditionPipeline._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

    @torch.no_grad()
    def __call__(
        self,
        video: list[Any],
        prompt: str | None = None,
        *,
        width: int = 1920,
        height: int = 1080,
        frame_rate: float = 24.0,
        sigma: float = DEFAULT_SIGMA,
        generator: torch.Generator | None = None,
        decoder_generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        output_type: Literal["np", "pt", "latent"] = "np",
        return_dict: bool = True,
    ) -> LTXPipelineOutput | tuple[Any, ...]:
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt or prompt_embeds")
        if output_type not in {"np", "pt", "latent"}:
            raise ValueError("output_type must be np, pt, or latent")
        self.last_prompt_cache_hit = False
        canvas_width, canvas_height, frames = output_geometry(width, height, len(video))
        device = self._execution_device
        upsample = LTX2LatentUpsamplePipeline(
            vae=self.vae, latent_upsampler=self.latent_upsampler
        )
        if self.cpu_preprocess_overlap and prompt_embeds is None:
            # PIL decode/resize/stack is CPU work independent of text encoding.
            # The future retains this clip only; join before its CUDA transfer.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sol-h3-prep") as pool:
                pixels_future = pool.submit(
                    upsample.video_processor.preprocess_video,
                    video[:frames],
                    height=canvas_height // 2,
                    width=canvas_width // 2,
                )
                assert prompt is not None
                prompt_embeds, prompt_attention_mask = self._encode_text(prompt, device)
                pixels = pixels_future.result().to(
                    device=device, dtype=self.vae.dtype
                )
        else:
            if prompt_embeds is None:
                assert prompt is not None
                prompt_embeds, prompt_attention_mask = self._encode_text(prompt, device)
            pixels = upsample.video_processor.preprocess_video(
                video[:frames], height=canvas_height // 2, width=canvas_width // 2
            ).to(device=device, dtype=self.vae.dtype)
        # Use the posterior mode, matching the deterministic reference encoder.
        encoded = self.vae.encode(pixels).latent_dist.mode()
        latents = upsample(
            latents=encoded,
            width=canvas_width // 2,
            height=canvas_height // 2,
            num_frames=frames,
            latents_normalized=False,
            output_type="latent",
        ).frames
        latents = LTX2ConditionPipeline._normalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(device=device, dtype=self.transformer.dtype)
        denoised = self.refine_conditioned_latents(
            latents, prompt_embeds, sigma=sigma, frame_rate=frame_rate,
            generator=generator, noise=noise,
            prompt_attention_mask=prompt_attention_mask,
        )
        # Conditioning and noise buffers have reached their last use. Release
        # their GPU storage before the decoder's larger activation tensors.
        del pixels, encoded, latents, noise, upsample
        if output_type == "latent":
            result = LTX2ConditionPipeline._denormalize_latents(
                denoised,
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
        else:
            if self.diffusion_decoder is None:
                raise ValueError("The diffusion decoder was skipped; request latent output instead")
            decoder = LTX2VideoDiffusionDecodePipeline(
                diffusion_decoder=self.diffusion_decoder,
                scheduler=self.scheduler,
                vae=self.vae,
            )
            result = decoder(
                denoised,
                generator=decoder_generator
                if decoder_generator is not None
                else generator,
                output_type=output_type,
                denormalize=True,
            ).frames
            top, left = (canvas_height - height) // 2, (canvas_width - width) // 2
            if output_type == "np":
                result = result[:, :, top : top + height, left : left + width, :]
            else:
                result = result[:, :, :, top : top + height, left : left + width]
        self.maybe_free_model_hooks()
        return LTXPipelineOutput(frames=result) if return_dict else (result,)
