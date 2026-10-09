#!/usr/bin/env python3
"""Prepare pinned SoL/Pruna model snapshots and exact-prompt conditioning.

The text encoder, tokenizer, and connectors are loaded without the VAE or
transformer. Both pinned inference snapshots are cached first so the shared
worker does not need to download weights during the first live upscale.
Conditioning is computed on CPU and saved under HF_HOME using the same
model/runtime identity that the shared-VAE worker validates.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

DEFAULT_UPSCALE_PROMPT = (
    "Upscale the source video while faithfully preserving its original scene, style, "
    "subject identities, colors, lighting, composition, and motion. Restore natural fine "
    "detail and clean edges with consistent texture across frames. Do not add objects, "
    "text, scene cuts, or new motion; do not restyle the video. Avoid flicker and temporal "
    "inconsistency."
)


def prepare_conditioning(
    package_root: Path,
    prompt: str,
    *,
    force: bool = False,
) -> Path:
    sys.path.insert(0, str(package_root))

    import torch
    from minimax_h3_sol_upscaler.fixed_conditioning import (
        FixedConditioning,
        conditioning_cache_path,
        conditioning_identity,
    )
    from minimax_h3_sol_upscaler.node import (
        MODEL_ID,
        MODEL_REVISION,
        VIDEO_ONLY_REFINER_ENV,
        _ensure_refiner_source_path,
        _experiment_flag,
    )
    from minimax_h3_sol_upscaler.fast_decoder import (
        ARTIFACT_SHA256 as PRUNA_ARTIFACT_SHA256,
        PRUNA_REPO,
        PRUNA_REVISION,
        _sha256,
    )
    from minimax_h3_sol_upscaler.sol_refiner_h3 import SoLRefinerH3Pipeline
    from diffusers import LTX2ConditionPipeline
    from diffusers.pipelines import pipeline_loading_utils
    from huggingface_hub import snapshot_download

    video_only = _experiment_flag(VIDEO_ONLY_REFINER_ENV)
    identity = conditioning_identity(prompt, MODEL_ID, MODEL_REVISION, video_only)
    model_snapshot = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    print(f"SoL H3 model snapshot ready at {model_snapshot}", flush=True)
    pruna_snapshot = Path(snapshot_download(
        repo_id=PRUNA_REPO,
        revision=PRUNA_REVISION,
        allow_patterns=list(PRUNA_ARTIFACT_SHA256),
    ))
    for name, expected in PRUNA_ARTIFACT_SHA256.items():
        path = pruna_snapshot / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Pinned Pruna decoder artifact failed verification: {name}")
    print(f"Pinned Pruna decoder snapshot ready at {pruna_snapshot}", flush=True)
    artifact = conditioning_cache_path(identity)
    if artifact.exists() and not force:
        FixedConditioning.load(artifact, identity)
        return artifact

    _ensure_refiner_source_path()
    previous = pipeline_loading_utils.LOADABLE_CLASSES.get("sol_refiner_h3")
    pipeline_loading_utils.LOADABLE_CLASSES["sol_refiner_h3"] = {
        "ModelMixin": ["save_pretrained", "from_pretrained"]
    }
    try:
        # The inference roles are explicitly omitted so deployment never needs
        # to fit the full SoL pipeline onto a single GPU to prepare text state.
        text_pipeline = SoLRefinerH3Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            vae=None,
            transformer=None,
            latent_upsampler=None,
            diffusion_decoder=None,
        )
    finally:
        if previous is None:
            pipeline_loading_utils.LOADABLE_CLASSES.pop("sol_refiner_h3", None)
        else:
            pipeline_loading_utils.LOADABLE_CLASSES["sol_refiner_h3"] = previous

    if (
        text_pipeline.text_encoder is None
        or text_pipeline.tokenizer is None
        or text_pipeline.connectors is None
    ):
        raise RuntimeError("SoL text-conditioning components did not load")
    condition_pipeline = LTX2ConditionPipeline(
        scheduler=text_pipeline.scheduler,
        vae=None,
        audio_vae=None,
        text_encoder=text_pipeline.text_encoder,
        tokenizer=text_pipeline.tokenizer,
        connectors=text_pipeline.connectors,
        transformer=None,
        vocoder=None,
    )
    with torch.inference_mode():
        embeddings, mask, _, _ = condition_pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=1024,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        if video_only:
            from minimax_h3_sol_upscaler.sol_refiner_h3.video_only import _video_only_connectors

            embeddings, mask = _video_only_connectors(
                text_pipeline.connectors,
                embeddings,
                mask,
                padding_side=text_pipeline.tokenizer.padding_side,
            )
        else:
            embeddings, _, mask = text_pipeline.connectors(
                embeddings,
                mask,
                padding_side=text_pipeline.tokenizer.padding_side,
            )
        FixedConditioning(identity, embeddings, mask).save(artifact)

    FixedConditioning.load(artifact, identity)
    del condition_pipeline, text_pipeline, embeddings, mask
    gc.collect()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        help="Prepare this exact prompt instead of the coordinator's default upscale prompt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the identity-matched artifact already exists",
    )
    args = parser.parse_args()
    prompt = args.prompt if args.prompt is not None else DEFAULT_UPSCALE_PROMPT
    if not prompt.strip():
        raise SystemExit("prompt must not be empty")
    artifact = prepare_conditioning(
        Path(__file__).resolve().parents[1],
        prompt,
        force=args.force,
    )
    print(f"verified conditioning artifact: {artifact} ({artifact.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
