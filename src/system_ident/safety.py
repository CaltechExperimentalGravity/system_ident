"""Safety: physical watchdog + ramp-down + safe-state handoff.

Per the agreed design (see the plan's "Safety / abort" section):

* The automatic watchdog keys on **physical hardware-safety only** — actuator
  drive vs saturation, and per-DOF output RMS vs a safe envelope. Coherence and
  fit-health are surfaced as *status* but never auto-abort.
* Abort (operator STOP, watchdog breach, or normal teardown) runs ONE shared,
  idempotent routine: ramp excitation down, hand control back to the existing
  damping loops, restore pre-run filter/channel state from a snapshot taken at
  run start, and flag the aborted state with its reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SafetyAbort(RuntimeError):
    """Raised when a physical-limit breach forces the loop to stop."""


@dataclass
class SafetyLimits:
    """Physical safety limits (the only things wired to auto-abort)."""

    actuator_sat: float  # max |drive| before saturation
    rms_ceiling: dict  # per-DOF output-RMS ceiling, e.g. {"POS": ..., "PIT": ...}
    ramp_down_secs: float = 2.0
    # Pre-injection drive-sample limits (issue #4, amended 2026-08-06). Optional
    # and None-default so twin/rtsfreerun behaviour is bit-identical when unset --
    # only a backend that opts into ChannelBackend._check_drive_limits (CDSBackend)
    # ever looks at these.
    max_exc_peak: float | None = None
    max_exc_rms: float | None = None

    @classmethod
    def from_config(cls, config: dict) -> "SafetyLimits":
        """Build from a run config's ``safety`` section."""
        s = config["safety"]
        return cls(
            actuator_sat=float(s["actuator_sat"]),
            # coerce per-DoF ceilings to float (YAML may yield strings, e.g. the
            # unsigned-exponent "1.0e3" form PyYAML does not treat as a number)
            rms_ceiling={k: float(v) for k, v in s["rms_ceiling"].items()},
            ramp_down_secs=float(s.get("ramp_down_secs", 2.0)),
            max_exc_peak=float(s["max_exc_peak"]) if "max_exc_peak" in s else None,
            max_exc_rms=float(s["max_exc_rms"]) if "max_exc_rms" in s else None,
        )


@dataclass
class SafetyReport:
    """Per-segment physical-safety status (surfaced to the dashboard)."""

    drive_peaks: dict = field(default_factory=dict)  # channel -> max|drive|
    output_rms: dict = field(default_factory=dict)  # DoF -> output RMS
    breaches: list = field(default_factory=list)  # human-readable breach strings

    @property
    def ok(self) -> bool:
        return not self.breaches


class Watchdog:
    """Monitors physical limits and owns the abort/handoff routine.

    The watchdog interprets a read *segment* (a ``{channel: samples}`` dict)
    using channel->DoF maps for the excitation and readback channels. If those
    maps are not supplied it falls back to the ones the backend exposes (the
    twin does), so in simulation it is wired up automatically.
    """

    def __init__(
        self,
        backend,
        limits: SafetyLimits,
        exc_channels: dict[str, str] | None = None,
        readback_channels: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.limits = limits
        self.exc_channels = dict(
            exc_channels if exc_channels is not None
            else getattr(backend, "exc_channels", {})
        )
        self.readback_channels = dict(
            readback_channels if readback_channels is not None
            else getattr(backend, "readback_channels", {})
        )
        self._snapshot = None
        self._aborted = False
        self.abort_reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def snapshot(self, channels: list[str]) -> None:
        """Capture pre-run state for restore-on-abort."""
        self._snapshot = self.backend.snapshot_state(channels)

    def evaluate(self, segment: dict) -> SafetyReport:
        """Compute the physical-safety report for one read segment (no side effects)."""
        report = SafetyReport()
        for ch, samples in segment.items():
            arr = np.asarray(samples, dtype=float)
            if ch in self.exc_channels:
                peak = float(np.max(np.abs(arr))) if arr.size else 0.0
                report.drive_peaks[ch] = peak
                if peak > self.limits.actuator_sat:
                    report.breaches.append(
                        f"actuator saturation on {ch}: "
                        f"{peak:.3g} > {self.limits.actuator_sat:.3g}"
                    )
            if ch in self.readback_channels:
                dof = self.readback_channels[ch]
                rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
                report.output_rms[dof] = rms
                ceiling = self.limits.rms_ceiling.get(dof)
                if ceiling is not None and rms > ceiling:
                    report.breaches.append(
                        f"RMS ceiling on {dof}: {rms:.3g} > {ceiling:.3g}"
                    )
        return report

    def check(self, segment: dict) -> SafetyReport:
        """Inspect one read segment; run the handoff and raise on a breach."""
        report = self.evaluate(segment)
        if not report.ok:
            reason = "; ".join(report.breaches)
            self.abort(reason)
            raise SafetyAbort(reason)
        return report

    def abort(self, reason: str = "operator STOP") -> None:
        """Run the shared, idempotent ramp-down + safe-state handoff."""
        if self._aborted:
            return
        self._aborted = True
        self.abort_reason = reason
        # 1. smoothly ramp every excitation channel to zero
        for channel in self.exc_channels:
            self.backend.ramp_down(channel, self.limits.ramp_down_secs)
        # 2. restore pre-run channel/filter state (hands control back to damping)
        if self._snapshot is not None:
            self.backend.restore_state(self._snapshot)
