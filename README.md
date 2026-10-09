# SoL H3 Upscaler

A standalone ComfyUI custom-node repository for NVIDIA's MiniMax-H3 SoL video upscaler. It includes the H3-specific SoL Refiner node, its isolated inference worker, vendored refiner implementation, runtime installer, and conditioning preparation utility.

## Install

Clone this repository as a folder under `ComfyUI/custom_nodes` and restart ComfyUI. For example:

```bash
git clone https://github.com/stale2000/solr-ltx-upscale-speedster.git ComfyUI/custom_nodes/solr-ltx-upscale-speedster
```

Install the node dependencies into the same Python 3.12 / CUDA 13 ComfyUI runtime described in [`deploy/vast/fasth3-environment.json`](deploy/vast/fasth3-environment.json):

```bash
PYTHON_BIN=/path/to/comfy/python \
  bash ComfyUI/custom_nodes/solr-ltx-upscale-speedster/minimax_h3_sol_upscaler/install_runtime.sh
```

The installer keeps ComfyUI's PyTorch 2.14 environment separate from the persistent worker's pinned PyTorch 2.13 environment. It does not install or replace ComfyUI itself. FFmpeg and a supported NVIDIA GPU are required at runtime. Model weights download on first use and are cached by Hugging Face.

See the [upscaler guide](minimax_h3_sol_upscaler/README.md) for node inputs, output behavior, GPU/runtime requirements, and optional multi-GPU modes. The [performance notes](minimax_h3_sol_upscaler/PERFORMANCE.md) describe measured tradeoffs. `scripts/prepare_h3_upscale_conditioning.py` prepares the optional fixed-prompt artifact; `scripts/benchmark_h3_upscale.py` dispatches benchmark runs to a local ComfyUI server.

The standalone ComfyUI entry point is [`__init__.py`](__init__.py); implementation files are grouped in [`minimax_h3_sol_upscaler/`](minimax_h3_sol_upscaler/). The repository-level custom-node code is MIT-licensed. The vendored upstream refiner retains its Apache-2.0 license and notices.
