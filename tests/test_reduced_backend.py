import numpy as np
import pytest
from system_ident.reduced_plant import ReducedStateSpacePlant
from system_ident.backends.reduced import ReducedPlantBackend
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop


def _hsts_LL_backend(sensor_asd=0.0, seed=0):
    p = ReducedStateSpacePlant.load("hsts").subplant(
        sensors=["m1.disp.L"], actuators=["m1.drive.L"])
    return p, ReducedPlantBackend(
        p, exc_channels={"EXC": "m1.drive.L"}, sens_channels={"RSP": "m1.disp.L"},
        fs=64.0, sensor_asd=sensor_asd, seed=seed, ramp_s=0.0)


def test_read_returns_requested_channels():
    _, be = _hsts_LL_backend()
    fs, nperseg, nper = 64.0, 4096, 4
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= 0.3) & (fa <= 5.0)
    Pxx = np.full(band.sum(), 1.0 / (fa[band][-1] - fa[band][0]))
    drive = multisine_from_psd(Pxx, fs, nperseg, nper, fa[band], seed=np.random.default_rng(0))
    be.inject("EXC", drive, fs)
    seg = be.read(["EXC", "RSP"], nperseg * nper / fs)
    assert set(seg) == {"EXC", "RSP"}
    assert len(seg["RSP"]) == nperseg * nper


def test_noiseless_recovery_matches_plant_frf():
    # a noiseless drive → the leakage-free FRF must equal the plant's own FRF on the excited lines
    p, be = _hsts_LL_backend(sensor_asd=0.0)
    fs, nperseg, nper, band_hz = 64.0, 4096, 4, (0.3, 5.0)
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= band_hz[0]) & (fa <= band_hz[1])
    freq = fa[band]
    Pxx = np.full(band.sum(), 1.0 / (freq[-1] - freq[0]))
    drive = multisine_from_psd(Pxx, fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("EXC", drive, fs)
    seg = be.read(["EXC", "RSP"], nperseg * nper / fs)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["EXC"], seg["RSP"], fs, nperseg, band, n_transient=1)
    G_true = p.eval(freq)[:, 0, 0]
    rel = np.abs(H - G_true) / np.maximum(np.abs(G_true), 1e-30)
    assert np.median(rel[np.isfinite(rel)]) < 1e-6
