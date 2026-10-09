"""Independent MiniMax-H3 SoL video-refinement node."""

from .node import (
    MiniMaxH3SoLParallelBenchmark,
    MiniMaxH3SoLSharedVAEBenchmark,
    MiniMaxH3SoLUpscaler,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SoLUpscaler": MiniMaxH3SoLUpscaler,
    "MiniMaxH3SoLParallelBenchmark": MiniMaxH3SoLParallelBenchmark,
    "MiniMaxH3SoLSharedVAEBenchmark": MiniMaxH3SoLSharedVAEBenchmark,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SoLUpscaler": "MiniMax-H3 SoL Upscaler (LTX-2.5)",
    "MiniMaxH3SoLParallelBenchmark": "MiniMax-H3 SoL Upscaler (Multi-GPU Benchmark)",
    "MiniMaxH3SoLSharedVAEBenchmark": "MiniMax-H3 SoL Upscaler (Shared VAE GPU Benchmark)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
