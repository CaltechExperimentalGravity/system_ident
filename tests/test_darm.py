# tests/test_darm.py
from __future__ import annotations
import numpy as np
import pytest
from system_ident.darm import DARMLoop, sensing_model

def _freq(loop):
    return np.linspace(loop.fmin, loop.fmax, 400)

def test_sensing_model_pole_and_delay():
    f = np.array([1.0, 360.0, 1000.0])
    H = sensing_model(f, g_c=1e6, f_cc=360.0, tau=77e-6)
    # at the cavity pole magnitude is 1/sqrt(2) of DC, phase rolls past -45 deg + delay
    assert abs(abs(H[1]) - 1e6/np.sqrt(2)) / (1e6/np.sqrt(2)) < 1e-6
    # delay adds linear phase: at 1 kHz the extra phase beyond the pole is -2*pi*f*tau
    extra = np.angle(H[2]) - np.angle(1e6/(1+1j*1000/360))
    assert np.isclose(np.angle(np.exp(1j*extra)), -2*np.pi*1000*77e-6, atol=1e-6)

def test_response_is_one_over_pcal_frf():
    loop = DARMLoop.default()
    f = _freq(loop)
    # R = (1+G)/C  and  FRF_pcal = C/(1+G)  =>  R == 1/FRF_pcal
    np.testing.assert_allclose(loop.R(f), 1.0/loop.frf_pcal(f), rtol=1e-10)

def test_G_equals_A_D_C_by_construction():
    loop = DARMLoop.default()
    f = _freq(loop)
    np.testing.assert_allclose(loop.G(f), loop.A(f)*loop.D(f)*loop.C(f), rtol=1e-9)

def test_stage_frf_identity():
    loop = DARMLoop.default()
    f = _freq(loop)
    for name in ("UIM","PUM","TST"):
        tf, kappa = loop.stages[name]
        expect = loop.C(f)*kappa*tf.eval(f)/(1+loop.G(f))
        np.testing.assert_allclose(loop.frf_stage(name, f), expect, rtol=1e-9)

def test_loop_is_stable_with_margin():
    loop = DARMLoop.default()
    f = np.geomspace(loop.fmin, loop.fmax, 4000)
    G = loop.G(f)
    mag = np.abs(G)
    # unity-gain frequency near the representative UGF
    k = int(np.argmin(np.abs(mag - 1.0)))
    f_ugf = f[k]
    assert 30.0 < f_ugf < 80.0
    pm = 180.0 + np.angle(G[k], deg=True)   # phase margin
    assert pm > 30.0


from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop

def _band_grid(loop, nperseg):
    fa = np.fft.rfftfreq(nperseg, 1/loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]

def test_simulate_deterministic_matches_pcal_frf():
    loop = DARMLoop.default()
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))   # flat, unit total power
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    derr = loop.simulate({"PCAL": x}, len(x), np.random.default_rng(1))
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(x, derr, loop.fs, nperseg, band, n_transient=1)
    # recovered closed-loop FRF tracks the analytic C/(1+G) on the excited bins
    good = np.isfinite(H_err)
    rel = np.abs(H[good] - loop.frf_pcal(freq)[good]) / np.abs(loop.frf_pcal(freq)[good])
    assert np.median(rel) < 1e-3

def test_disturbance_and_sensing_noise_color_differently():
    import scipy.signal as sig
    loop = DARMLoop.default()
    n = int(64 * loop.fs)
    # disturbance only
    ld = DARMLoop.default(); ld.disturbance_asd = 1e-18; ld.sensor_asd = 0.0
    yd = ld.simulate({}, n, np.random.default_rng(0))
    # sensing only
    ls = DARMLoop.default(); ls.disturbance_asd = 0.0; ls.sensor_asd = 1e-2
    ys = ls.simulate({}, n, np.random.default_rng(0))
    f, Pd = sig.welch(yd, fs=loop.fs, nperseg=int(4*loop.fs))
    _, Ps = sig.welch(ys, fs=loop.fs, nperseg=int(4*loop.fs))
    inband = (f >= loop.fmin) & (f <= loop.fmax)
    # the two noise paths have different numerators (C vs 1) -> different in-band shape
    shape_d = Pd[inband] / np.median(Pd[inband])
    shape_s = Ps[inband] / np.median(Ps[inband])
    assert np.max(np.abs(shape_d - shape_s)) > 0.3   # measurably distinct spectra


from system_ident.backends.darm_adapter import DARMBackend

def test_backend_recovers_pcal_frf():
    loop = DARMLoop.default(); loop.sensor_asd = 1e-3
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=2)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("PCAL_EXC", x, loop.fs)
    dur = (nperseg * nper) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], dur)
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(
        seg["PCAL_EXC"], seg["DARM_ERR"], loop.fs, nperseg, band, n_transient=1)
    good = np.isfinite(H_err)
    rel = np.abs(H[good] - loop.frf_pcal(freq)[good]) / np.abs(loop.frf_pcal(freq)[good])
    assert np.median(rel) < 5e-3

def test_backend_inject_ramps_drive():
    loop = DARMLoop.default()
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", ramp_s=3.0)
    drive = np.ones(int(20 * loop.fs))
    be.inject("PCAL_EXC", drive, loop.fs)
    mon = be.read(["PCAL_EXC"], 20.0)["PCAL_EXC"]
    assert abs(mon[0]) < 1e-9 and abs(mon[-1]) < 1e-9
    assert mon[len(mon)//2] == pytest.approx(1.0)


from system_ident.darm import recover_response, fit_sensing, recover_actuation

def _run_pcal(loop, seed=3):
    nperseg, nper = 4096, 16
    fa, band, freq = _band_grid(loop, nperseg)
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("PCAL_EXC", x, loop.fs)
    seg = be.read(["PCAL_EXC","DARM_ERR"], (nperseg*nper)/loop.fs)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                  loop.fs, nperseg, band, n_transient=1)
    return freq, band, H, H_err

def test_recover_response_tracks_truth():
    loop = DARMLoop.default(); loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    freq, band, H, H_err = _run_pcal(loop)
    R, R_sig = recover_response(H, H_err)
    good = np.isfinite(H_err)
    R_true = loop.R(freq)[good]
    rel = np.abs(R[good] - R_true) / np.abs(R_true)
    # R tracks truth to within the measurement noise (~1% per bin here) — not tighter,
    # because the recovery is genuinely noise-limited (the 5e-3 the old P_eff=1 regime
    # hit was an artefact of floored, fabricated error bars).
    assert np.median(rel) < 1.5e-2
    assert np.all(R_sig[good] > 0)
    # honesty check: the recovery error is consistent with its OWN CRB envelope — the
    # normalised residual |R-R_true|/sigma is order unity, so sigma neither under- nor
    # over-states the true uncertainty (a floored sigma would blow this up; an inflated
    # one would crush it).
    z = np.abs(R[good] - R_true) / R_sig[good]
    assert 0.3 < np.median(z) < 3.0

def test_fit_sensing_recovers_pole_and_delay():
    loop = DARMLoop.default(); loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    freq, band, H, H_err = _run_pcal(loop)
    C_meas = H * (1.0 + loop.G(freq))            # expose C with the known (1+G)
    p, sig_ = fit_sensing(freq, C_meas, H_err*np.abs(1+loop.G(freq)),
                          p0=(0.8e6, 300.0, 50e-6))
    assert abs(p["f_cc"] - 360.0)/360.0 < 0.05
    assert abs(p["tau"] - 77e-6) < 15e-6
    assert abs(p["g_c"] - 1e6)/1e6 < 0.05

def test_recover_actuation_kappas():
    loop = DARMLoop.default(); loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    nperseg, nper = 4096, 16
    fa, band, freq = _band_grid(loop, nperseg)
    # Pcal reference
    freqp, _, Hp, Hp_err = _run_pcal(loop)
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    for name, true_k in (("UIM",1.00),("PUM",0.40),("TST",0.08)):
        be = DARMBackend(loop, {"EXC": name}, "DARM_ERR", seed=5)
        x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(1))
        be.inject("EXC", x, loop.fs)
        seg = be.read(["EXC","DARM_ERR"], (nperseg*nper)/loop.fs)
        Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(seg["EXC"], seg["DARM_ERR"],
                                                        loop.fs, nperseg, band, n_transient=1)
        tf, _ = loop.stages[name]
        N = tf.eval(freq)
        comb_err = np.hypot(Hi_err/np.abs(Hi), Hp_err/np.abs(Hp)) * np.abs(Hi/Hp)
        k, ks = recover_actuation(freq, Hi, Hp, N, comb_err)
        assert abs(k - true_k)/true_k < 0.05
        assert ks > 0

def test_pcal_uncertainty_is_genuinely_estimated():
    """NPER=16 leaves enough full periods that the per-bin FRF variance is REAL
    (not the 1e-9 floor) — so the page's CRB envelope is measured, not fabricated."""
    loop = DARMLoop.default(); loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    freq, band, H, H_err = _run_pcal(loop)
    R, R_sig = recover_response(H, H_err)
    good = np.isfinite(H_err)
    frac = R_sig[good] / np.abs(R[good])
    assert np.median(frac) > 5e-3        # visible, not the ~1e-8 of a too-clean twin
    assert np.median(frac) < 5e-2        # representative, not absurd
    assert np.mean(H_err[good] / np.abs(H[good]) < 2e-9) < 0.05   # <5% of bins at the floor
