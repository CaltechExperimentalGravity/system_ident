"""Faithfulness guard for the landing-page excitation arcade.

The browser game (``docs/assets/excitation-arcade.js``) reimplements the Fisher/CRB math in
JS for a smooth, instant UI. This test pins its Python twin (``docs/arcade_reference.py``) so
the game's numbers stay (a) internally reproducible (golden ETAs the JS is calibrated to) and
(b) faithful to the real ``system_ident`` pole convention. If these drift, the JS and the
on-page "optimal par" label must be updated in lock-step. See
``docs/superpowers/specs/2026-07-05-excitation-arcade-hero-design.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import arcade_reference as ar  # noqa: E402
from system_ident.model import TFModel  # noqa: E402


def test_frf_kernel_matches_tfmodel_pole_convention():
    """A single resonance in the arcade kernel equals TFModel's up to a global scale
    (identical poles) — so the game shows the *real* package's plant, not a lookalike."""
    w = 2 * np.pi * ar.FREQ
    f0, Q = 1.0, 20.0
    kernel = 1.0 / ((2 * np.pi * f0) ** 2 - w ** 2 + 1j * w * (2 * np.pi * f0) / Q)
    tf = TFModel.from_resonances([(f0, Q)], gain=1.0).eval(ar.FREQ)
    ratio = kernel / tf
    # constant ratio across the band <=> same poles/zeros, only overall gain differs
    assert np.std(ratio) / np.abs(np.mean(ratio)) < 1e-10


def test_golden_etas():
    """The calibrated ETAs the JS is built to reproduce."""
    flat_eta, _ = ar.eta_seconds(ar.flat_drive())
    opt_eta, _ = ar.eta_seconds(ar.optimal_drive())
    assert abs(opt_eta - 45.0) < 0.5           # optimal par ~ 45 s (calibration target)
    assert abs(flat_eta - 1768.6) < 5.0        # flat baseline ~ 1769 s
    assert flat_eta / opt_eta > 30.0           # the headline gap (~39x)


def test_optimal_is_the_floor_and_starving_a_mode_is_punished():
    """Strategy gradient: the optimal beats every naive drive, and dumping the whole budget
    on one mode is worse than flat (it starves the other mode's Q) — the game's core lesson."""
    def gauss(fc, wdec=0.12):
        P = np.exp(-0.5 * ((np.log10(ar.FREQ) - np.log10(fc)) / wdec) ** 2) + 1e-3
        return P / trapezoid(P, ar.FREQ) * ar.PX_TOT

    opt_eta, _ = ar.eta_seconds(ar.optimal_drive())
    flat_eta, _ = ar.eta_seconds(ar.flat_drive())
    m1_eta, _ = ar.eta_seconds(gauss(ar.MODES[0][0]))   # all power on mode 1 (the tall look)
    m2_eta, _ = ar.eta_seconds(gauss(ar.MODES[1][0]))   # all power on mode 2

    assert opt_eta < flat_eta < m1_eta          # optimal < flat < all-on-one-mode
    assert opt_eta < m2_eta
    # both single-mode strategies are worse than flat (starving the other mode's Q hurts)
    assert m1_eta > flat_eta and m2_eta > flat_eta


def test_calibration_constant_is_consistent():
    """C_TIME must be exactly the value that maps the optimal drive to 45 s (JS shares it)."""
    worst_opt = float(np.max(ar.frac_uncertainty(ar.optimal_drive())))
    assert abs(45.0 / worst_opt ** 2 - ar.C_TIME) / ar.C_TIME < 1e-4
