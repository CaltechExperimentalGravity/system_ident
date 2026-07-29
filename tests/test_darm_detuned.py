"""DARM Upgrade 2: SRC-detuning optical spring (Cahillane detuned sensing) + δ tracking.

The nominal SRC is tuned (BRSE, δ=0): sensing is a single real cavity pole. As the SRC error
point drifts (±~7°), the Cahillane optical-spring factor lifts the response into a complex
resonance — restoring on one side of tuning, anti-restoring on the other. δ is a *sensing*
parameter, recovered from the Pcal FRF shape and tracked by the existing TV machinery.
"""
import functools

import numpy as np
import pytest

from system_ident.darm import (
    DARMLoop, sensing_model, sensing_model_detuned, optical_spring_factor, drift_profile,
)
from system_ident import darm_tv as tv


def test_delta_zero_reproduces_single_pole_exactly():
    """The hard constraint: at δ=0 (BRSE tuned) the detuned sensing IS the single-pole model,
    to machine precision — so every existing tuned-loop test is preserved."""
    f = np.geomspace(0.3, 1500, 3000)
    C0 = sensing_model_detuned(f, 1e6, 360.0, 77e-6, fs2=0.0, Qs=2.0)
    np.testing.assert_allclose(C0, sensing_model(f, 1e6, 360.0, 77e-6), rtol=0, atol=0)
    # and via the loop
    loop = DARMLoop.default()
    np.testing.assert_allclose(loop.C(f), sensing_model(f, loop.g_c, loop.f_cc, loop.tau), rtol=1e-12)


def test_optical_spring_factor_is_unity_when_tuned():
    f = np.geomspace(1, 1000, 50)
    np.testing.assert_array_equal(optical_spring_factor(f, 0.0, 2.0), np.ones_like(f, dtype=complex))


def test_detuning_splits_the_pole_real_to_complex():
    """Sweep δ across ±7°: the optical-spring pole f_s² goes negative (anti-spring) → 0 (tuned)
    → positive (restoring complex resonance), continuously through the split at δ=0."""
    loop = DARMLoop.default()
    fs2 = [loop.with_params(delta=np.radians(d)).fs2() for d in (-7, -3, 0, 3, 7)]
    assert fs2[0] < fs2[1] < 0 == fs2[2] < fs2[3] < fs2[4]         # monotone through zero
    # restoring side is a genuine in-band resonance; the loop identity still holds under detuning
    ld = loop.with_params(delta=np.radians(7.0))
    f = np.geomspace(10, 1500, 2000)
    np.testing.assert_allclose(ld.G(f), ld.A(f) * ld.D(f) * ld.C(f), rtol=1e-9)
    fs, _ = ld.spring_pole()
    assert 100 < fs < 400                                          # spring resonance in band


def test_G_identity_holds_for_detuned_reduced_loop():
    """Both upgrades compose: reduced-quad actuation + detuned sensing, G == A·D·C exactly."""
    loop = DARMLoop.default_reduced(fmin=10.0).with_params(delta=np.radians(5.0))
    f = np.geomspace(10, 1500, 2000)
    np.testing.assert_allclose(loop.G(f), loop.A(f) * loop.D(f) * loop.C(f), rtol=1e-9)


def _noisy_reduced():
    loop = DARMLoop.default_reduced(fmin=10.0)
    loop.sensor_asd = 300.0
    loop.disturbance_asd = 3e-4
    return loop


@pytest.mark.parametrize("deg", [7.0, -7.0, 4.0])
def test_snapshot_delta_within_crb_both_signs(deg):
    """δ recovered from the Pcal FRF within its CRB, on both the restoring and anti-spring sides."""
    dh, sig = tv.snapshot_delta(_noisy_reduced(), np.radians(deg), n_periods=16, seed=5)
    assert sig > 0
    assert abs(dh - np.radians(deg)) < 4 * sig, f"{np.degrees(dh):.3f}° vs {deg}° ({(dh-np.radians(deg))/sig:.1f}σ)"


def test_tracks_injected_detuning_within_crb():
    """A slow drift of δ around a detuned operating point is tracked within its CRB (pull ~O(1))
    and is genuinely resolvable — the same TV fit as κ, reused verbatim for δ(t)."""
    loop = _noisy_reduced()
    d0, amp, period = np.radians(5.0), 0.05, 7200.0
    prof = functools.partial(drift_profile, base=d0, amp_frac=amp, period_s=period, kind="sine")
    times = np.linspace(0.0, 3600.0, 25)
    t, dhat, sig = tv.track_delta(loop, times, prof, n_periods=16, seed=777)
    d_true = prof(t)
    fit = tv.fit_tv(t, dhat, sig, kind="legendre", order=5)
    theta, s_theta, _, _ = fit.predict(t)
    assert np.median(np.abs(theta - d_true) / d_true) < 2e-2
    assert 0.15 < np.median(np.abs(theta - d_true) / s_theta) < 3.0
    res = tv.resolvability(fit, base=d0, amp_frac=amp, period_s=period, kind="sine",
                           record_s=16 * 4096 / loop.fs)
    assert res["resolve_ratio"] > 5.0
