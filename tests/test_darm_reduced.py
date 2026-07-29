"""DARM Upgrade 1: the reduced-quad plant as the actuation-stage mechanical response.

`DARMLoop.default_reduced()` swaps each placeholder pendulum stage for the real reduced-order
QUAD longitudinal column (L_i.drive.L -> L3.disp.L). The loop math is unchanged (D = G/(A·C)
derived), so the drop-in is safe; the stages now carry the true quad modes.
"""
import numpy as np
import pytest

from system_ident.darm import DARMLoop, ReducedStageShape
from system_ident import darm_tv as tv

QUAD_MODES_HZ = (0.43, 0.52, 0.99, 1.98)      # reduced-quad longitudinal chain, in band


def _freq(loop, n=2000):
    return np.geomspace(max(loop.fmin, 0.05), loop.fmax, n)


def test_default_reduced_G_identity():
    """The derived servo keeps G == A·D·C exactly with the reduced-quad stages."""
    loop = DARMLoop.default_reduced()
    f = _freq(loop)
    np.testing.assert_allclose(loop.G(f), loop.A(f) * loop.D(f) * loop.C(f), rtol=1e-9)


def test_stage_frf_identity_holds_for_reduced():
    loop = DARMLoop.default_reduced()
    f = _freq(loop)
    for name in ("M0", "PUM", "TST"):
        shape, kappa = loop.stages[name]
        expect = loop.C(f) * kappa * shape.eval(f) / (1 + loop.G(f))
        np.testing.assert_allclose(loop.frf_stage(name, f), expect, rtol=1e-9)


def test_reduced_stages_carry_the_quad_modes():
    """The quad longitudinal resonances appear in the stage shapes (embedded, in band)."""
    loop = DARMLoop.default_reduced()
    assert loop.fmin <= 0.3          # band reaches the modes
    fm = np.linspace(0.3, 2.5, 4000)
    mag = np.abs(loop.stage("M0", fm))
    peaks = fm[[i for i in range(2, len(mag) - 2)
                if mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > 1.5 * np.median(mag)]]
    for m in QUAD_MODES_HZ:
        assert np.any(np.abs(peaks - m) < 0.03), f"quad mode {m} Hz not found in M0 stage; peaks={peaks}"


def test_absolute_scale_anchored_but_irrelevant():
    """Each stage's mechanical shape is anchored to unit magnitude at 100 Hz, so |stage_i(100)|
    == κ_i (κ semantics preserved); the absolute scale does not affect the loop because
    D = G/(A·C) is derived."""
    lr = DARMLoop.default_reduced()
    f100 = np.array([100.0])
    for name, (_, kappa) in lr.stages.items():
        assert abs(lr.stage(name, f100)[0]) == pytest.approx(kappa, rel=1e-6)
    # scaling every stage by 10x leaves the closed-loop FRFs unchanged (D re-derives)
    lr2 = DARMLoop.default_reduced()
    for name in lr2.stages:
        shape, k = lr2.stages[name]
        lr2.stages[name] = (ReducedStageShape(shape._sub, shape._ii, gain=shape.gain * 10.0), k)
    f = _freq(lr)
    np.testing.assert_allclose(lr.frf_pcal(f), lr2.frf_pcal(f), rtol=1e-9)
    np.testing.assert_allclose(lr.R(f), lr2.R(f), rtol=1e-9)


def test_kappa_recovery_within_crb_in_smooth_band():
    """With the κ campaign in the smooth region (10–1500 Hz), the Pcal ruler recovers each
    stage strength through the reduced plant within its CRB."""
    loop = DARMLoop.default_reduced(fmin=10.0)
    for name, k_true in [("TST", 0.075), ("PUM", 0.42), ("M0", 1.1)]:
        khat, sig = tv.snapshot_kappa(loop, name, k_true, n_periods=16, seed=7)
        assert abs(khat - k_true) < 4 * sig, f"{name}: {khat:.4f} vs {k_true} ({(khat-k_true)/sig:.1f}σ)"


def test_low_band_campaign_cannot_resolve_sharp_modes():
    """Finding: extending the *campaign* to 0.3 Hz biases κ recovery, because the quad modes are
    Q≈50 (linewidth ~0.01 Hz) — far narrower than the multisine bin spacing (fs/nperseg = 1 Hz) —
    so those bins corrupt the ruler. The modes are *embedded* regardless of fmin; κ must be
    measured in the smooth region. This guards that separation."""
    biased = DARMLoop.default_reduced(fmin=0.3)
    khat_lo, sig_lo = tv.snapshot_kappa(biased, "M0", 1.1, n_periods=16, seed=7)
    # the low-band campaign is clearly biased (many sigma off) — not within CRB
    assert abs(khat_lo - 1.1) > 8 * sig_lo, (
        f"expected low-band κ bias, got {khat_lo:.4f} ({(khat_lo-1.1)/sig_lo:.1f}σ) — "
        "if this now passes, the campaign resolves the modes and the caveat can be dropped")
