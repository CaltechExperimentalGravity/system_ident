"""Closed-loop identification: naive estimator is biased, P&S is consistent.

Self-contained (a simple resonator + velocity damper — no foton, no twin). The SOS-specific
instance is in ``test_sos_closed.py``; this file tests the estimator property itself.
"""
import numpy as np
import control
import pytest

from system_ident import closed_loop_id as clid


def _siso_loop(fs=256.0, f0=1.0, Q=50.0, kappa=8.0):
    """A 2-state resonator plant closed around a velocity damper. Returns per-line maps."""
    w0 = 2 * np.pi * f0
    G = control.tf([1.0], [1.0, w0 / Q, w0 * w0])              # displacement / force
    K = control.tf([kappa, 0.0], [1.0 / (2 * np.pi * 30.0), 1.0])   # velocity damper, 30 Hz pole
    Gd = control.c2d(G, 1.0 / fs, "tustin")
    Kd = control.c2d(K, 1.0 / fs, "tustin")
    lines = f0 * (1 + np.linspace(-6, 6, 13) / Q)             # cluster on the mode
    scale, Gk, Sk, SKk = clid.loop_frf_maps(Gd, Kd, lines, balance=False)
    return lines, Gk, Sk, SKk


def test_loop_is_stable_and_damps():
    """Sanity: the velocity damper actually closes a stable, damping loop (else the bias test
    is meaningless)."""
    _, Gk, Sk, SKk = _siso_loop()
    # closed-loop response peak is below the open-loop resonance peak (damping)
    assert np.max(np.abs(Gk * Sk)) < np.max(np.abs(Gk))


def test_naive_bias_matches_analytic():
    """The centerpiece: the measured naive-estimator bias equals the closed-form closed-loop
    bias -SK* sigma^2 / E|V|^2. This is what makes 'naive is biased' a real, quantitative claim
    rather than a strawman."""
    lines, Gk, Sk, SKk = _siso_loop()
    R = np.exp(2j * np.pi * np.random.default_rng(0).random(len(lines)))
    sigma = float(np.mean(np.abs(Gk))) * 0.3
    # measured bias: average the naive per-line estimate over many disturbance realizations
    acc = np.zeros(len(lines), complex)
    S = 300
    for sd in range(S):
        V, Y = clid.simulate(Sk, SKk, Gk, R, n_periods=48, sigma=sigma,
                             rng=np.random.default_rng(1000 + sd))
        acc += clid.naive_frf(V, Y)[:, 0, 0]
    bias_meas = acc / S - Gk[:, 0, 0]
    bias_analytic = clid.naive_bias_analytic(Sk[:, 0, 0], SKk[:, 0, 0], R, sigma ** 2)
    rel = np.abs(bias_meas - bias_analytic) / np.abs(bias_analytic)
    assert np.median(rel) < 0.1, f"median naive-bias mismatch {np.median(rel):.3f}"


def _weighted_err_fn(Sk, R):
    """FRF error weighted by drive-monitor signal power |S·R|² — the lines where the reference
    actually probes. (Unweighted averages are dominated by deep-resonance lines, where the loop
    gain is high, S is small, and the reference is suppressed, so they carry no information.)"""
    Vpow = np.abs(Sk[:, 0, 0] * R) ** 2

    def err_fn(Ghat, Gtrue):
        e = np.abs(Ghat[:, 0, 0] - Gtrue[:, 0, 0]) / np.abs(Gtrue[:, 0, 0])
        return float(np.sum(Vpow * e) / np.sum(Vpow))
    return err_fn


def test_naive_plateaus_ps_converges():
    """Naive error is a bias floor independent of P; P&S error falls with P and overtakes it.

    (The exact 1/sqrt(P) rate is SNR-dependent, so this asserts the robust qualitative facts;
    the quantitative rigor is in test_naive_bias_matches_analytic.)"""
    lines, Gk, Sk, SKk = _siso_loop()
    R = np.exp(2j * np.pi * np.random.default_rng(1).random(len(lines)))
    sigma = float(np.mean(np.abs(Gk))) * 0.5
    res = clid.sweep_periods(Sk, SKk, Gk, R, Gk, periods=[4, 16, 64, 256],
                             sigma=sigma, seeds=30, err_fn=_weighted_err_fn(Sk, R))
    # naive: a bias floor — essentially flat across P (well within 15%)
    assert abs(res.naive_err[-1] - res.naive_err[0]) / res.naive_err[0] < 0.15
    # P&S: falls meaningfully with P (bias being averaged away)...
    assert res.ps_err[-1] < 0.75 * res.ps_err[0]
    # ...and overtakes the biased naive estimate at the largest P
    assert res.ps_err[-1] < 0.75 * res.naive_err[-1]


def test_open_loop_has_no_bias():
    """Null control: with K=0 (open loop) there is no feedback correlation, so naive and P&S
    agree at every P — the bias is specifically the closed-loop effect."""
    fs = 256.0
    G = control.tf([1.0], [1.0, 2 * np.pi / 50, (2 * np.pi) ** 2])
    Gd = control.c2d(G, 1.0 / fs, "tustin")
    K0 = control.tf([0.0], [1.0])
    lines = 1.0 * (1 + np.linspace(-6, 6, 13) / 50)
    _, Gk, Sk, SKk = clid.loop_frf_maps(Gd, control.c2d(K0, 1.0 / fs, "tustin"), lines, balance=False)
    R = np.exp(2j * np.pi * np.random.default_rng(2).random(len(lines)))
    sigma = float(np.mean(np.abs(Gk))) * 0.5
    V, Y = clid.simulate(Sk, SKk, Gk, R, n_periods=32, sigma=sigma, rng=np.random.default_rng(7))
    err = lambda Gh: float(np.mean(np.abs(Gh[:, 0, 0] - Gk[:, 0, 0]) / np.abs(Gk[:, 0, 0])))
    en, ep = err(clid.naive_frf(V, Y)), err(clid.ps_frf(V, Y))
    assert abs(en - ep) / en < 0.05, f"open loop should have naive==P&S, got {en:.3e} vs {ep:.3e}"
