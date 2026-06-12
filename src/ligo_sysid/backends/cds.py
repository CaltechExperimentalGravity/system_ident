"""Real-hardware backend: awg excitation injection + nds2 online readback.

The CDS libraries (``awg``/``cdsutils``, ``nds2``, ``foton``) are provided by
the LIGO control-room environment and are **lazy-imported** inside the methods,
so importing this module (and the package) never requires them — the twin path
stays usable on a plain scientific-Python stack. Implemented in build step 8.
"""

from __future__ import annotations

import numpy as np

from .base import ChannelBackend


class CDSBackend(ChannelBackend):
    """awg-inject + nds2-readback against real CDS hardware (build step 8)."""

    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        raise NotImplementedError("CDSBackend lands in build step 8")

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        raise NotImplementedError("CDSBackend lands in build step 8")

    def ramp_down(self, channel: str, secs: float) -> None:
        raise NotImplementedError("CDSBackend lands in build step 8")
