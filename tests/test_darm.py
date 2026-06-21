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
