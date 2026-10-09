# MiniMax-H3 SoL Upscaler

This ComfyUI custom node upscales an existing MiniMax-H3 video with NVIDIA's H3-specific SoL Refiner, an LTX-2.5 model. Connect the input `VIDEO`, provide its scene prompt, and choose an even output size that preserves the input aspect ratio. The default for a 512×288 input is 1184×666. Output is a file-backed Comfy `VIDEO`; the first audio track is copied when present, and silent input stays silent.

For installation, see the repository [README](../README.md).

## Models and runtime

The node downloads the model on first inference from `Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3`, pinned to revision `4655337474c1c77fb52e9ac55111552e625e779c`. The vendored pipeline code is based on NVlabs/Sana revision `670482d8a857d578ac8a2ea89b052d0fb47badba`; its Apache-2.0 license is included in `licenses/` with attribution in `NOTICE.md`. Model weights are not included and retain their publisher's license.

The supported host is Python 3.12 with PyTorch 2.14.0+cu130, CUDA 13.0, torchvision 0.29.0+cu130, and torchaudio 2.11.0+cu130. These pins are recorded in `../deploy/vast/fasth3-environment.json` and `runtime.json`. The node installer adds the remaining host dependencies but does not replace ComfyUI's PyTorch. It creates a persistent `.worker` environment with Python 3.12 and PyTorch 2.13.0+cu130 so native NATTEN and inference dependencies stay isolated from ComfyUI. FFmpeg and an NVIDIA GPU are required.

The worker keeps model weights on CUDA and disables CPU offload. It starts lazily on the first request, stays resident across requests, and restarts when its code/runtime identity changes. Cold model downloads and compilation can take much longer than warm requests. NVENC is the default output encoder; choose `libx264` if FFmpeg cannot use NVENC. In containers, enable `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` for NVENC.

## Inference modes

- **`pruna_fast`** (default) uses the H3 LTX-2.5 refiner and a pinned distilled LTX convolutional decoder for final pixels. It avoids loading the reference diffusion decoder. The first use downloads and verifies the Pruna decoder weights (about 1.33 GB).
- **`diffusion`** uses the reference diffusion decoder for the sharper comparison path. It requires native NATTEN in the isolated worker and more GPU memory.
- **Transformer precision** defaults to dynamic NVFP4 on Blackwell GPUs. MXFP8 and BF16 are selectable overrides. NVFP4 is faster and smaller in the measured profile, but it changes generated details; see [performance findings](PERFORMANCE.md).
- **Output size** must be even and preserve the source aspect ratio. The reference decoder uses overlapping 768-pixel tiles; smaller outputs may use a full spatial tile. Video is padded to the model's `1 + 8k` frame length and trimmed back to the original frame count.

## Optional repeated-prompt conditioning

For a deployment that always uses one exact prompt, set `SOL_H3_UPSCALER_CONDITIONING_ARTIFACT` to a persistent file path. Prepare the default prompt with:

```bash
bash minimax_h3_sol_upscaler/install_runtime.sh --prepare-shared-vae-conditioning
```

The preparation utility downloads the pinned snapshots and writes a validated, tensor-only artifact. The artifact is bound to the exact prompt, model/runtime identity, and encoding schema. A different prompt fails rather than reusing stale embeddings. Restart ComfyUI after changing the artifact or related environment settings. Ordinary requests support changing prompts without this option.

The worker also has a bounded exact-prompt embedding cache for repeated requests. It returns private tensor copies and can be disabled with `SOL_H3_UPSCALER_PROMPT_CACHE=0`.

## Optional multi-GPU benchmark nodes

The package exposes experimental split-video and shared-VAE benchmark nodes. They run on GPUs visible to the same ComfyUI host; they do not submit work to remote ComfyUI servers. Shared-VAE mode assigns VAE/decoder work to one GPU and transformer chunks to one to four other GPUs. It requires exact-prompt conditioning. The split-video node loads a full model per GPU, restarts temporal context across chunks, and may differ from whole-video output. Inspect overlap frames and compare with the single-GPU node before using results. Neither multi-GPU option is the default production path.

The optional local coordinator script submits jobs to an already-running ComfyUI server and saves timing receipts; it does not provision GPUs:

```bash
python3 scripts/benchmark_h3_upscale.py input.mp4 \
  --prompt "Preserve the source scene and motion" \
  --devices 0,1 --mode both --repeats 3 --open-results
```

For the measured speed and memory tradeoffs, including experiments that did not produce reliable gains, see [PERFORMANCE.md](PERFORMANCE.md).
