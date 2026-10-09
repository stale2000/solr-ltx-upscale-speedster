"""CPU-only validation and temporal scheduling for the shared VAE experiment."""

from __future__ import annotations

from itertools import pairwise

from .distributed import VideoChunk, resolve_visible_devices


def validate_chunk_settings(chunk_frames: int, overlap_frames: int) -> None:
    if type(chunk_frames) is not int or not 32 <= chunk_frames <= 512 or chunk_frames % 8:
        raise ValueError("shared VAE chunk frames must be 32–512 and divisible by 8")
    if type(overlap_frames) is not int or overlap_frames not in {16, 32, 48, 64}:
        raise ValueError("shared VAE overlap must be 16, 32, 48, or 64 frames")
    if chunk_frames <= overlap_frames:
        raise ValueError("shared VAE chunk frames must exceed temporal overlap")


def shared_vae_topology(
    vae_device: str, cuda_devices: str, visible: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return Comfy-visible slots and corresponding CUDA visibility tokens.

    The dedicated subprocess sees the VAE first, then the upscaling GPUs.
    Its internal device ordinals are consequently always 0, 1, 2, ... .
    """
    if not isinstance(vae_device, str) or not vae_device.strip().isdecimal():
        raise ValueError("VAE device must be a visible numeric CUDA ordinal")
    vae = str(int(vae_device.strip()))
    raw_workers = tuple(part.strip() for part in cuda_devices.split(",") if part.strip())
    if not 1 <= len(raw_workers) <= 4 or any(not item.isdecimal() for item in raw_workers):
        raise ValueError("shared VAE requires one to four numeric upscaling device IDs")
    workers = tuple(str(int(device)) for device in raw_workers)
    slots = (vae, *workers)
    if len(set(slots)) != len(slots):
        raise ValueError("shared VAE requires a distinct VAE GPU and distinct upscaling GPUs")
    return slots, resolve_visible_devices(slots, visible)


def plan_shared_vae_chunks(
    frame_count: int, devices: tuple[str, ...], chunk_frames: int, overlap_frames: int,
) -> tuple[VideoChunk, ...]:
    """Plan aligned cores, assigning successive chunks to reusable GPU slots.

    Context windows retain the existing temporal experiment's half-overlap
    convention. A small final core is merged into its predecessor, avoiding
    nearly empty model calls. Chunk-local noise and temporal context remain
    experimental; this is not identical to processing the complete clip.
    """
    validate_chunk_settings(chunk_frames, overlap_frames)
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("shared VAE frame count must be a positive integer")
    if not 1 <= len(devices) <= 4 or len(set(devices)) != len(devices):
        raise ValueError("shared VAE needs one to four distinct upscaling devices")
    if any(not isinstance(device, str) or not device.isdecimal() for device in devices):
        raise ValueError("shared VAE worker device IDs must be numeric ordinals")
    context = overlap_frames // 2
    boundaries = list(range(0, frame_count, chunk_frames)) + [frame_count]
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < context + 8 + 1:
        del boundaries[-2]
    return tuple(
        VideoChunk(
            index=index, device=devices[index % len(devices)],
            core_start=start, core_end=end,
            input_start=max(0, start - context),
            input_end=min(frame_count, end + context),
        )
        for index, (start, end) in enumerate(pairwise(boundaries))
    )


class GlobalLatentNoise:
    """Generate one clip-wide seeded noise field and serve aligned chunk slices.

    LTX temporal latents use one position for the first frame and then one
    position per eight-frame group. Chunk starts are aligned to that phase.
    Keeping the random field on CPU makes a chunk's values independent of GPU
    assignment and worker scheduling; only its bounded slice is copied to a GPU.
    """
    def __init__(self, torch: object, frame_count: int, seed: int) -> None:
        if type(frame_count) is not int or frame_count <= 0:
            raise ValueError("global latent noise requires a positive frame count")
        if type(seed) is not int or seed < 0:
            raise ValueError("global latent noise seed must be a nonnegative integer")
        self.torch = torch
        self.frame_count = frame_count
        self.seed = seed
        self.field = None

    def for_chunk(self, chunk: VideoChunk, latents: object):
        if len(latents.shape) != 5:
            raise ValueError("shared VAE latents must have batch/channel/time/height/width dimensions")
        if chunk.input_start % 8:
            raise ValueError("shared VAE chunk starts must align to the latent temporal phase")
        total_latents = 1 + (self.frame_count - 1 + 7) // 8
        local_latents = 1 + (chunk.input_frames - 1 + 7) // 8
        if latents.shape[2] != local_latents:
            raise ValueError(
                "shared VAE latent time dimension does not match the eight-frame temporal phase"
            )
        start = chunk.input_start // 8
        end = start + local_latents
        if end > total_latents:
            raise ValueError("shared VAE chunk latent range exceeds the clip-wide noise field")
        if self.field is None:
            generator = self.torch.Generator(device="cpu").manual_seed(self.seed)
            self.field = self.torch.randn(
                (*latents.shape[:2], total_latents, *latents.shape[3:]),
                generator=generator,
                device="cpu",
                dtype=self.torch.float32,
            )
        elif tuple(self.field.shape[:2] + self.field.shape[3:]) != tuple(
            latents.shape[:2] + latents.shape[3:]
        ):
            raise ValueError("shared VAE latent geometry changed between chunks")
        return self.field[:, :, start:end].to(device=latents.device, dtype=latents.dtype)
