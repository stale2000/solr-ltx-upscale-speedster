"""Use an installed NATTEN backend with the upstream decoder processor."""

from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodNattenProcessor,
)


class LocalNattenProcessor(LTX2VideoVaeNeighborhoodNattenProcessor):
    """Keep the upstream attention calculation while loading NATTEN locally."""

    def __init__(self, backend: str | None = None, compile_flex: bool = False):
        try:
            from natten.functional import na3d
        except ImportError as exc:
            raise ImportError(
                "Install the NATTEN wheel matching your PyTorch and CUDA versions"
            ) from exc
        if compile_flex:
            from functools import partial

            from natten.context import allow_flex_compile

            allow_flex_compile()
            self._na3d = partial(na3d, torch_compile=True)
        else:
            self._na3d = na3d
        self.backend = backend
