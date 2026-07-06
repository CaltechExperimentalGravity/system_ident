import numpy as np
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


def test_no_out_of_band_energy():
    # Guard against regressing the active-bin mask in `_simulate` back to a bare
    # `np.abs(Xhat) > 0` test: non-excited rFFT bins carry ~1e-13-relative roundoff
    # noise, which `> 0` flags as "active" (here: 2017/8193 bins instead of the 301
    # truly-excited multisine lines), defeating the "evaluate G only where the drive
    # has power" sparsity and risking a spurious out-of-band tone near a high-Q mode.
    p, be = _hsts_LL_backend(sensor_asd=0.0)
    fs, nperseg, nper, band_hz = 64.0, 4096, 4, (0.3, 5.0)
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= band_hz[0]) & (fa <= band_hz[1])
    freq = fa[band]
    n_excited = int(band.sum())
    Pxx = np.full(n_excited, 1.0 / (freq[-1] - freq[0]))
    drive = multisine_from_psd(Pxx, fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("EXC", drive, fs)

    # spy on plant.eval to confirm the mask only ever hands it the truly-excited lines
    eval_sizes = []
    orig_eval = p.eval

    def _spy(f):
        eval_sizes.append(np.asarray(f).size)
        return orig_eval(f)

    p.eval = _spy
    try:
        seg = be.read(["EXC", "RSP"], nperseg * nper / fs)
    finally:
        p.eval = orig_eval
    assert eval_sizes == [n_excited], (
        f"plant.eval was called with {eval_sizes} frequencies; expected exactly "
        f"one call with the {n_excited} truly-excited bins (roundoff-noise bins leaking "
        "into the active mask again?)"
    )

    # the synthesized sensor output must carry negligible energy outside the excited
    # lines: rFFT of the *full* record (nperseg*nper samples) puts the excited harmonics
    # at bin index k_period*nper on the finer full-record grid.
    y = seg["RSP"]
    Yhat = np.fft.rfft(y)
    k_full = np.round(freq / (fs / nperseg)).astype(int) * nper
    mag = np.abs(Yhat)
    out_of_band = np.ones(mag.shape[0], dtype=bool)
    out_of_band[k_full] = False
    assert mag[out_of_band].max() < 1e-8 * mag[k_full].min()
