"""Guard the round-1 time-varying DARM tracking (src/system_ident/darm_tv.py).

The plant drifts far slower than one record, so we snapshot a stage strength κ at a
sequence of times (leakage-free P&S, Pcal as the ruler) and fit θ(t)=Σ c_k b_k(t) in
a time-basis.  These tests pin: the loop copy-with-override, the deterministic drift
profile, a single snapshot, the static limit (constant truth), and — the point of the
round — that a *drifting* κ is recovered within its Cramér–Rao band (honest pull), with
the injected drift genuinely resolvable (bound computed, not asserted by eye).
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from system_ident.darm import DARMLoop, drift_profile
from system_ident import darm_tv as tv


def _noisy_loop():
    loop = DARMLoop.default()
    loop.sensor_asd = 300.0
    loop.disturbance_asd = 3e-4
    return loop


# ── plumbing ────────────────────────────────────────────────────────────────────────
def test_with_params_overrides_stage_and_scalar():
    loop = DARMLoop.default()
    f = np.linspace(loop.fmin, loop.fmax, 50)
    hot = loop.with_params(kappa_TST=0.16)          # double the TST strength
    assert hot.stages["TST"][1] == 0.16
    assert loop.stages["TST"][1] == 0.08            # base untouched
    np.testing.assert_allclose(hot.frf_stage("TST", f),
                               2.0 * loop.frf_stage("TST", f), rtol=1e-12)
    # scalar sensing override flows through C
    dim = loop.with_params(g_c=5e5)
    np.testing.assert_allclose(dim.C(f), 0.5 * loop.C(f), rtol=1e-12)
    with pytest.raises(KeyError):
        loop.with_params(kappa_NOPE=1.0)


def test_drift_profile_shapes():
    assert drift_profile(0.0, 0.08, amp_frac=0.05, period_s=7200.0) == pytest.approx(0.08)
    # quarter period of the sine -> full +amp excursion
    assert drift_profile(1800.0, 0.08, amp_frac=0.05, period_s=7200.0) == pytest.approx(0.08 * 1.05)
    assert drift_profile(7200.0, 1.0, amp_frac=0.1, period_s=7200.0, kind="ramp") == pytest.approx(1.1)
    with pytest.raises(ValueError):
        drift_profile(0.0, 1.0, kind="bogus")


# ── one snapshot ──────────────────────────────────────────────────────────────────--
def test_snapshot_recovers_kappa_within_sigma():
    loop = _noisy_loop()
    k_true = 0.093
    khat, sig = tv.snapshot_kappa(loop, "TST", k_true, n_periods=16, seed=7)
    assert sig > 0
    assert abs(khat - k_true) / k_true < 0.05          # recovered to a few %
    assert abs(khat - k_true) < 4.0 * sig              # consistent with its own σ


# ── basis fit + CRB ─────────────────────────────────────────────────────────────────
def test_basis_matrix_derivative_matches_finite_difference():
    t = np.linspace(0.0, 100.0, 11)
    for kind, order in (("legendre", 4), ("fourier", 2)):
        B, dB = tv.basis_matrix(t, kind=kind, order=order, t0=0.0, t1=100.0)
        h = 1e-3
        Bp, _ = tv.basis_matrix(t + h, kind=kind, order=order, t0=0.0, t1=100.0)
        Bm, _ = tv.basis_matrix(t - h, kind=kind, order=order, t0=0.0, t1=100.0)
        np.testing.assert_allclose(dB, (Bp - Bm) / (2 * h), atol=1e-6)


def test_static_limit_reproduces_constant_recovery():
    """θ(t)=const must come back as a flat line at the true κ — the TV fit degrades to
    the ordinary static κ recovery when there is no drift."""
    loop = _noisy_loop()
    k0 = 0.08
    prof = functools.partial(drift_profile, base=k0, amp_frac=0.0, period_s=7200.0)
    times = np.linspace(0.0, 1800.0, 10)
    t, khat, sig = tv.track_kappa(loop, "TST", times, prof, n_periods=16, seed=20)
    fit = tv.fit_tv(t, khat, sig, kind="legendre", order=2)
    theta, s_theta, theta_dot, s_dot = fit.predict(times)
    # flat at the truth, with a genuine (positive) uncertainty, and no spurious slope
    assert np.all(s_theta > 0)
    assert np.max(np.abs(theta - k0)) / k0 < 0.03
    # the apparent slope is consistent with zero within its own CRB (no drift invented)
    assert np.median(np.abs(theta_dot) / s_dot) < 3.0


def test_tracks_injected_drift_within_crb():
    """The headline: a 5 % / hour-scale drift on the ESD stage is recovered within its
    CRB band (honest pull ~ O(1)), and the drift is genuinely resolvable."""
    loop = _noisy_loop()
    name, k0, amp, period = "TST", 0.08, 0.05, 7200.0
    prof = functools.partial(drift_profile, base=k0, amp_frac=amp, period_s=period, kind="sine")
    times = np.linspace(0.0, 3600.0, 25)
    t, khat, sig = tv.track_kappa(loop, name, times, prof, n_periods=16, seed=777)
    k_true = prof(t)

    fit = tv.fit_tv(t, khat, sig, kind="legendre", order=5)
    theta, s_theta, theta_dot, s_dot = fit.predict(t)

    # (a) the fitted curve follows the injected drift closely
    assert np.median(np.abs(theta - k_true) / k_true) < 2e-2
    # (b) honest CRB: normalised residual is O(1) — σ neither inflated nor floored
    z = np.abs(theta - k_true) / s_theta
    assert 0.15 < np.median(z) < 3.0
    assert np.all(s_theta > 0)

    # (c) feasibility gate: the drift is resolvable, not noise (bound with a number)
    res = tv.resolvability(fit, base=k0, amp_frac=amp, period_s=period, kind="sine",
                           record_s=16 * 4096 / loop.fs)
    assert res["resolve_ratio"] > 5.0                       # drift amp >> tracking σ
    assert res["local_stationarity_err"] < 0.05             # record << drift timescale


@pytest.mark.slow    # ~80 s: runs the whole example-13 campaign to build the panels.
                     # CI's docs job renders 13-darm-drift-tracking.qmd, which walks the
                     # same import + figure path, so deselecting it there loses no cover.
def test_docs_glue_imports_and_builds_figures():
    """The example-13 presentation module imports fresh (no circular import) and builds its
    panels — this is the render path, so it guards against the darm↔darm_adapter cycle."""
    import importlib
    import pathlib
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "docs"))
    dtd = importlib.import_module("darm_tv_demo")
    c = dtd.campaign(seed=1)
    assert dtd.drift_fig(c).data and dtd.tracking_error_fig(c).data
    assert dtd.resolvability_table(c) is not None
    assert c.res["resolve_ratio"] > 5.0


# ── stochastic drift + joint (several-parameter) tracking ───────────────────────────
def test_stochastic_drift_is_mean_reverting():
    """The OU drift sample has ~the requested mean and stationary std over a long record, and is
    deterministic given the seed."""
    t = np.linspace(0.0, 3.0e5, 6000)          # ≫ tau_s, so the sample std approaches the target
    x = tv.stochastic_drift(t, base=1.0, amp_frac=0.05, tau_s=1800.0, seed=3)
    assert abs(x.mean() - 1.0) < 0.02
    assert 0.03 < x.std() < 0.07               # stationary std ≈ amp_frac·base = 0.05
    np.testing.assert_array_equal(x, tv.stochastic_drift(t, base=1.0, amp_frac=0.05,
                                                         tau_s=1800.0, seed=3))


def test_joint_snapshot_untangles_parameters():
    """Recover three parameters drifting at once (optical gain g_c, SRC detuning δ, ESD strength
    κ_TST) from one set of records: each within a few σ of truth, with a symmetric unit-diagonal
    correlation matrix that exposes the g_c–κ_TST degeneracy (broken only by Pcal)."""
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True).with_params(delta=np.radians(5.0))
    loop.sensor_asd, loop.disturbance_asd = 300.0, 3.0e-4
    truth = {"g_c": 1.04e6, "delta": np.radians(5.3), "kappa_TST": 1.05}
    theta, sigma, corr, names = tv.joint_snapshot(loop, truth, n_periods=16, seed=11)
    assert names == ["g_c", "delta", "kappa_TST"]
    for n in names:
        assert abs(theta[n] - truth[n]) < 5 * sigma[n], f"{n}: {theta[n]:.4g} vs {truth[n]:.4g}"
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
    np.testing.assert_allclose(corr, corr.T, atol=1e-8)
    assert np.all(np.abs(corr) <= 1.0 + 1e-6)


def test_track_joint_shapes_and_recovery():
    """track_joint follows a simultaneous drift of two parameters and returns aligned arrays + a
    mean correlation matrix."""
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True).with_params(delta=np.radians(5.0))
    loop.sensor_asd, loop.disturbance_asd = 300.0, 3.0e-4
    t = np.linspace(0.0, 1800.0, 4)
    series = {"delta": tv.stochastic_drift(t, np.radians(5.0), amp_frac=0.05, tau_s=900, seed=1),
              "kappa_TST": drift_profile(t, base=1.0, amp_frac=0.05, period_s=3600.0)}
    tt, th, sg, corr, names = tv.track_joint(loop, series, t, n_periods=16, seed=5)
    assert set(names) == {"delta", "kappa_TST"} and len(tt) == 4
    for n in names:
        assert th[n].shape == (4,) and np.all(sg[n] > 0)
        assert np.median(np.abs(th[n] - series[n]) / sg[n]) < 4.0
    assert corr.shape == (2, 2)
