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
    # and down over ``ramp_s`` s so the drive never slams the suspension. ``ramp_s=0``
    # disables it. Backends override this in ``__init__``; the package default is 3 s.
    #
    # THE RAMP CONTRACT (#9) -- two branches, never both:
    #   (a) ``_soft_start_stop`` on a one-shot lead+record+tail array (a Tukey /
    #       COSINE taper -- see ``RTSfreerunBackend.read``), or
    #   (b) an equal-duration TRANSPORT-LEVEL gain ramp applied outside the analysed
    #       record (a LINEAR ramp on real CDS hardware -- see ``CDSBackend``).
    # A backend MUST use exactly one, and its docstring MUST say which envelope shape
    # it produces: the two are NOT interchangeable. Looping a Tukey-tapered array
    # (rather than applying (a) to a one-shot array, or using (b)) turns the taper
    # into a periodic amplitude modulation at the loop period and re-excites the
    # plant's transient every cycle -- measured 2.81e-1 max relative FRF error,
    # against 8.8e-12 for (a) and 1.4e-11 for (b). Double-ramping (both branches at
    # once) squares the envelope over ``2*ramp_s``, breaking the equal-energy
    # assumption the leading-periods guard below relies on.
    ramp_s: float = 3.0

    def _soft_start_stop(self, ts: np.ndarray, fs: float) -> np.ndarray:
        """Apply the ``ramp_s`` Tukey (cosine) on/off envelope to an injected drive
        at rate ``fs`` -- ramp contract branch (a), above.

        Shared by every backend using this branch so the taper is one implementation,
        not per-backend copies. A backend using branch (b) instead (an equal-duration
        transport-level gain ramp) must NOT call this -- see the ramp contract on
        :attr:`ramp_s`.
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

        Implementations MUST apply the ``ramp_s`` on/off envelope via exactly one of
        the two ramp-contract branches documented on :attr:`ramp_s` (actuator
        safety) -- never both -- and must document in their own docstring which
        envelope shape (cosine / linear) the chosen branch produces.
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
