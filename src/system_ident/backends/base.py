"""The ``ChannelBackend`` interface: inject excitation, read back channels.

Real hardware (awg + nds2) and the digital twin both implement this. The loop
and the safety handoff use only these methods, so simulation and hardware are
truly interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.signal as sig


class ChannelBackend(ABC):
    """Abstract excitation-inject / data-readback channel access."""

    # Actuator-safe injection: every backend ramps the excitation up over ``ramp_s`` s
    # and down over ``ramp_s`` s (Tukey) so the drive never slams the suspension. The
    # ramp is applied at injection (not baked into the periodic multisine, which must
    # stay a whole-period tiling for the leakage-free FRF). ``ramp_s=0`` disables it.
    # Backends override this in ``__init__``; the package default is 3 s.
    ramp_s: float = 3.0

    def _soft_start_stop(self, ts: np.ndarray, fs: float) -> np.ndarray:
        """Apply the ``ramp_s`` Tukey on/off envelope to an injected drive at rate ``fs``.

        Shared by every backend so the actuator-safe ramp is one implementation, not
        per-backend copies. **Any new backend that drives a real or simulated actuator
        MUST pass its injected excitation through this** (the leakage-free FRF then drops
        the tapered periods / they fall in a discarded warmup, see the backends).
        """
        ts = np.asarray(ts, dtype=float)
        n = len(ts)
        if self.ramp_s <= 0.0 or n < 2:
            return ts
        alpha = min(1.0, 2.0 * float(self.ramp_s) * float(fs) / n)   # ramp_s tapered at each end
        return ts * sig.windows.tukey(n, alpha)

    @abstractmethod
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        """Inject ``timeseries`` (sampled at ``fs`` Hz) onto ``channel``.

        Implementations MUST ramp the drive via :meth:`_soft_start_stop` (actuator
        safety) before it reaches the actuator.
        """
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
