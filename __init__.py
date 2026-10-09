"""ComfyUI custom-node entry point for the standalone H3 SoL upscaler repo."""

from .minimax_h3_sol_upscaler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
