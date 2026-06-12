"""The ``ChannelBackend`` interface: inject excitation, read back channels.

Real hardware (awg + nds2) and the digital twin both implement this. The loop
and the safety handoff use only these methods, so simulation and hardware are
truly interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ChannelBackend(ABC):
    """Abstract excitation-inject / data-readback channel access."""

    @abstractmethod
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        """Inject ``timeseries`` (sampled at ``fs`` Hz) onto ``channel``."""
        raise NotImplementedError

    @abstractmethod
    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        """Read ``duration`` seconds of data for ``channels``."""
        raise NotImplementedError

    @abstractmethod
    def ramp_down(self, channel: str, secs: float) -> None:
        """Smoothly ramp the excitation on ``channel`` to zero over ``secs``.

        First step of the safety handoff; never a hard cut.
        """
        raise NotImplementedError

    # -- safe-state handoff support (build step 5) --------------------------
    def snapshot_state(self, channels: list[str]) -> dict:
        """Capture pre-run filter/channel state for later restore. (step 5)"""
        raise NotImplementedError("snapshot_state lands in build step 5")

    def restore_state(self, snapshot: dict) -> None:
        """Restore channel/filter state from a snapshot. (step 5)"""
        raise NotImplementedError("restore_state lands in build step 5")
