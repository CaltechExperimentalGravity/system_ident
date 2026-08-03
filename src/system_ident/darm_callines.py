"""DARM calibration-line hardware constants: the Pcal free-mass actuator range.

The calibration group tracks the time-dependent correction factors (TDCFs) with always-on
calibration lines — a Pcal displacement reference plus one actuator line per suspension stage.

This module retains only the pieces with genuine physical provenance:

* :func:`pcal_range_disp` — the photon-calibrator's maximum DARM-displacement amplitude, fixed by
  the real ±200 mW peak-to-peak power modulation on the 40 kg free test mass (radiation-pressure
  force on a quasi-free mass, so the displacement rolls off as 1/f²).
* :data:`O3_LINES` / :data:`O4_LINES` — the published O3/O4 calibration-line *frequencies*
  (O3: Sun et al. 2020, DCC P1900245; O4: arXiv:2508.08423).

.. note::
   A previous version of this module carried a Fisher-optimal cal-line *sizing/design* engine
   (per-stage actuator "authority", ``size_lines_for_target`` / ``size_lines_for_response``,
   ``reference_scheme``, the Pareto cost, response-budget propagation, and the O3/O4 head-to-head).
   Those absolute results rested on **fabricated** per-stage actuator ranges (an invented
   ``_STAGE_RANGE_M``): without the real suspension-stage ranges the actuation-side sizing and the
   cross-scheme comparisons cannot be computed, so that machinery — and every conclusion built on it
   — was removed rather than caveated. Only the Pcal range (real hardware) and the published line
   frequencies survive here.
"""
from __future__ import annotations

import numpy as np

from . import provenance as _prov

#: Published LIGO calibration-line positions [Hz] (O3: Sun 2020 §4.1, DCC P1900245; O4:
#: arXiv:2508.08423). Frequencies only, with a note of the TDCF each is associated with. These are
#: the real published values; no per-line drive amplitudes are included.
O3_LINES = [(7.9, "PCAL", "delta"), (17.1, "PCAL", "kappa_C"), (331.9, "PCAL", "f_cc"),
            (1083.7, "PCAL", "tau"), (15.6, "M0", "kappa_M0"), (16.4, "PUM", "kappa_PUM"),
            (35.9, "TST", "kappa_TST")]
O4_LINES = [(6.5, "PCAL", "delta"), (17.1, "PCAL", "kappa_C"), (33.4, "PCAL", "kappa_C"),
            (410.3, "PCAL", "f_cc"), (1083.1, "PCAL", "tau"), (15.1, "M0", "kappa_M0"),
            (16.9, "PUM", "kappa_PUM"), (34.7, "TST", "kappa_TST")]


# ── Pcal actuator range: the DARM displacement the photon calibrator can make per frequency ──────
#: Photon-calibrator hardware range: it modulates laser power (radiation-pressure force F = 2P/c) on
#: a quasi-free test mass, so its displacement rolls off as 1/f². The real Pcal has ~200 mW
#: peak-to-peak power range (given by the author, real hardware); with the 40 kg test mass this fixes
#: the ABSOLUTE Pcal line amplitude at every frequency — no free parameter.
PCAL_POWER_PP_W = _prov.record(
    "pcal_power_pp_w", 0.200, _prov.USER,
    "rana 2026-08 — Pcal hardware peak-to-peak power modulation range", unit="W")
TEST_MASS_KG = _prov.record(
    "test_mass_kg", 40.0, _prov.PAPER, "aLIGO quad test mass", unit="kg")
_C_LIGHT = _prov.record(
    "c_light", 299_792_458.0, _prov.CONSTANT, "speed of light", unit="m/s")


def pcal_range_disp(freq) -> np.ndarray:
    """Maximum Pcal DARM-displacement amplitude [m rms] at ``freq`` from the full ±200 mW power
    range: radiation-pressure force ``F_rms = (P_pp/c)/√2`` (``2·(P_pp/2)/c``, converted to rms) on
    the free test mass, ``x = F/(M(2πf)²)`` — the 1/f² free-mass actuator range."""
    f = np.asarray(freq, dtype=float)
    F_rms = PCAL_POWER_PP_W / _C_LIGHT / np.sqrt(2.0)
    return F_rms / (TEST_MASS_KG * (2.0 * np.pi * f) ** 2)
