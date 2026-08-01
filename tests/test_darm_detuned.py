"""DARM Upgrade 2: SRC-detuning coupled cavity (f_cc splits into a complex pair) + δ tracking.

The nominal SRC is tuned (BRSE, δ=0): sensing is a single real cavity pole. As the SRC error
point drifts (±~7°), the coupled-cavity response SPLITS that pole — the second pole descends from
∞, the pair collides near ±7°, then lifts off the real axis into a complex-conjugate resonance
(restoring, δ>0) or one pole crosses into the RHP (anti-restoring, δ<0). δ is a *sensing*
parameter, recovered from the Pcal FRF shape and tracked by the existing TV machinery.
"""
import functools

import numpy as np
import pytest

from system_ident.darm import (
    DARMLoop, sensing_model, sensing_model_detuned, coupled_cavity_factor,
    coupled_cavity_poles, drift_profile,
)
from system_ident import darm_tv as tv


def test_delta_zero_reproduces_single_pole_exactly():
    """The hard constraint: at δ=0 (α=0, BRSE tuned) the coupled sensing IS the single-pole model,
    to machine precision — so every existing tuned-loop test is preserved."""
    f = np.geomspace(0.3, 1500, 3000)
    C0 = sensing_model_detuned(f, 1e6, 360.0, 77e-6, alpha=0.0)
    np.testing.assert_allclose(C0, sensing_model(f, 1e6, 360.0, 77e-6), rtol=0, atol=0)
    # and via the loop
    loop = DARMLoop.default()
    np.testing.assert_allclose(loop.C(f), sensing_model(f, loop.g_c, loop.f_cc, loop.tau), rtol=1e-12)


def test_coupled_factor_is_single_pole_when_tuned():
    f = np.geomspace(1, 1000, 50)
    np.testing.assert_allclose(coupled_cavity_factor(f, 360.0, 0.0),
                               1.0 / (1.0 + 1j * f / 360.0), rtol=0, atol=0)


def test_detuning_splits_the_cavity_pole_real_to_complex():
    """Sweep δ from 0 → +8°: the two coupled-cavity poles are REAL below the collision (α<1/4)
    and become a genuine complex-conjugate pair above it — f_cc literally splits. Verify on the
    denominator ROOTS (not a separate spring factor), and that the collision is near ~7°."""
    loop = DARMLoop.default()
    below = loop.with_params(delta=np.radians(3.0))
    above = loop.with_params(delta=np.radians(8.0))
    assert below.alpha() < 0.25 < above.alpha()                   # straddles the collision
    p_below = below.cavity_poles()
    p_above = above.cavity_poles()
    # below: both poles real in s ⇒ purely imaginary in f (Re(f_pole) ≈ 0)
    assert np.max(np.abs(p_below.real)) < 1e-6 * loop.f_cc
    # above: a genuine complex-conjugate pair (nonzero real part, conjugate imaginary parts)
    assert np.min(np.abs(p_above.real)) > 1e-3 * loop.f_cc
    np.testing.assert_allclose(p_above[0].real, -p_above[1].real, rtol=1e-9)
    # collision (α=1/4) sits near ~7°
    dc = np.degrees(0.5 * np.arcsin(0.25 / loop.detune_coupling))
    assert 6.0 < dc < 8.0
    # the loop identity still holds under the split
    f = np.geomspace(10, 1500, 2000)
    np.testing.assert_allclose(above.G(f), above.A(f) * above.D(f) * above.C(f), rtol=1e-9)


def test_anti_spring_pushes_a_pole_to_the_rhp():
    """δ<0 (anti-restoring): the poles stay real but one crosses into the RHP (Re(s)>0) — the
    optical-spring instability, modeled honestly (the synthesized G keeps the twin usable)."""
    loop = DARMLoop.default().with_params(delta=np.radians(-6.0))
    assert loop.alpha() < 0
    poles_f = loop.cavity_poles()                                 # f-plane; s = i·2π·f
    s = 2j * np.pi * poles_f
    assert np.max(s.real) > 0                                     # a pole in the right half plane


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
