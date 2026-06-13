"""Task 2 — gauge-free Fisher information and optimal excitation for ResonatorModel.

Tests the new ``resonator_design`` module that provides:

* ``fisher_information(freq, model, Pxx, Pyy, T_tot)`` — gauge-free Fisher
  matrix (n_par × n_par) using the ResonatorModel Jacobian directly.
* ``dispersion(freq, model, Pxx, Pyy)`` — per-frequency information density.
* ``optimal_excitation(freq, model, Pyy, Px_tot)`` — Pintelon-Schoukens
  dispersion-function iteration, power-budget normalised.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from system_ident.resonator import ResonatorModel
from system_ident.resonator_design import (
    fisher_information,
    dispersion,
    optimal_excitation,
)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

@pytest.fixture
def single_mode():
    return ResonatorModel.from_resonances([(1.0, 20.0)], gain=100.0)


@pytest.fixture
def two_mode():
    return ResonatorModel.from_resonances([(0.6, 15.0), (1.5, 25.0)], gain=200.0)


@pytest.fixture
def flat_psd(single_mode):
    freq = np.linspace(0.1, 5.0, 400)
    Pxx = np.ones_like(freq)
    Pyy = 1e-6 * np.ones_like(freq)
    return freq, Pxx, Pyy


# ---------------------------------------------------------------------------
# Fisher matrix tests
# ---------------------------------------------------------------------------

def test_fisher_is_symmetric_and_spd(single_mode, flat_psd):
    """Fisher information matrix must be symmetric and strictly positive definite."""
    freq, Pxx, Pyy = flat_psd
    T_tot = 256.0
    gamma = fisher_information(freq, single_mode, Pxx, Pyy, T_tot)

    assert gamma.shape == (3, 3), f"expected (3,3) got {gamma.shape}"
    np.testing.assert_allclose(gamma, gamma.T, rtol=1e-10)
    eigvals = np.linalg.eigvalsh(gamma)
    assert np.all(eigvals > 0), f"Fisher not positive definite: eigenvalues={eigvals}"


def test_fisher_scales_linearly_with_T_tot(single_mode, flat_psd):
    """Fisher information is proportional to total measurement time."""
    freq, Pxx, Pyy = flat_psd
    g1 = fisher_information(freq, single_mode, Pxx, Pyy, T_tot=100.0)
    g2 = fisher_information(freq, single_mode, Pxx, Pyy, T_tot=200.0)
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-10)


def test_fisher_shape_two_mode(two_mode, flat_psd):
    """Two-mode model: Fisher is (5,5) = [f0_0, f0_1, Q_0, Q_1, gain]."""
    freq, Pxx, Pyy = flat_psd
    gamma = fisher_information(freq, two_mode, Pxx, Pyy, T_tot=100.0)
    assert gamma.shape == (5, 5)
    eigvals = np.linalg.eigvalsh(gamma)
    assert np.all(eigvals > 0)


def test_fisher_sensitive_to_pxx(single_mode, flat_psd):
    """Concentrating excitation near the resonance increases Fisher information."""
    freq, Pxx_flat, Pyy = flat_psd
    Px_tot = trapezoid(Pxx_flat, freq)

    # concentrate all power within ±0.2 Hz of resonance
    Pxx_narrow = np.zeros_like(freq)
    near = np.abs(freq - 1.0) < 0.2
    Pxx_narrow[near] = 1.0
    Pxx_narrow *= Px_tot / trapezoid(Pxx_narrow, freq)

    g_flat = fisher_information(freq, single_mode, Pxx_flat, Pyy, T_tot=100.0)
    g_narrow = fisher_information(freq, single_mode, Pxx_narrow, Pyy, T_tot=100.0)
    # trace(gamma) = total information; concentrated drive near resonance adds more
    assert np.trace(g_narrow) > np.trace(g_flat), (
        f"concentrated drive should increase Fisher: flat={np.trace(g_flat):.4g} "
        f"narrow={np.trace(g_narrow):.4g}"
    )


# ---------------------------------------------------------------------------
# Optimal excitation tests
# ---------------------------------------------------------------------------

def test_optimal_excitation_respects_power_budget(single_mode, flat_psd):
    """The optimised PSD integrates to exactly ``Px_tot``."""
    freq, _, Pyy = flat_psd
    Px_tot = 1e-3
    Pxx_opt = optimal_excitation(freq, single_mode, Pyy, Px_tot)
    power = trapezoid(Pxx_opt, freq)
    np.testing.assert_allclose(power, Px_tot, rtol=1e-8)


def test_optimal_excitation_is_nonnegative(single_mode, flat_psd):
    freq, _, Pyy = flat_psd
    Pxx_opt = optimal_excitation(freq, single_mode, Pyy, Px_tot=1.0)
    assert np.all(Pxx_opt >= 0), "optimised PSD has negative values"


def test_optimal_excitation_concentrates_near_resonances(single_mode, flat_psd):
    """After a few Pintelon-Schoukens iterations the drive peaks near resonances."""
    freq, _, Pyy = flat_psd
    Pxx_opt = optimal_excitation(freq, single_mode, Pyy, Px_tot=1.0, n_iter=3)

    # the resonance region should carry more power than a flat baseline
    near_res = (freq > 0.8) & (freq < 1.2)    # ±0.2 Hz of 1.0 Hz resonance
    power_near = trapezoid(Pxx_opt[near_res], freq[near_res])
    Px_tot = trapezoid(Pxx_opt, freq)
    width_frac = trapezoid(np.ones_like(freq[near_res]), freq[near_res]) / (freq[-1] - freq[0])
    # Should carry more than proportional flat share
    assert power_near / Px_tot > width_frac, (
        f"excitation not concentrated near resonance: "
        f"near_res_fraction={power_near/Px_tot:.3f} vs flat_fraction={width_frac:.3f}"
    )
