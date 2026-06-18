"""RTSfreerunBackend: adapter contract + plant recovery, against a mock twin.

The real rtsfreerun models build on a Linux/CDS box; here we verify the adapter
and the config/CLI wiring against a tiny in-process ``MockRTSModel`` that mimics
the ``mdl`` API (sample_rate / run / fetch_later / write) and filters the injected
drive through a known plant.
"""
import importlib.util

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
from system_ident.config import ConfigError, RunConfig
from system_ident.design.pintelon import PintelonSchoukensDesigner
from system_ident.estimators.gml import GMLEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.safety import SafetyLimits, Watchdog

FS = 512.0
NPERSEG, N_PERIODS = 8192, 4              # T_fft = 16 s, total 64 s
EXC, SENSOR, DRIVE_MON = "COIL_DRIVER_EXC", "READOUT_NOISE_OUT", "HSTS_DRV_TF_B_OUT"
NOISE_CH = "READOUT_NOISE_EXC"
PLANT = TFModel.from_resonances([(1.0, 10.0)], 100.0)


class _Buf:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class MockRTSModel:
    """Minimal fake of a compiled rtsfreerun model.

    drive (on EXC) → plant G → SENSOR (+ additive noise on NOISE_CH); the
    DRIVE_MON probe returns the drive itself. Persistent filter state across runs
    (warmup then record), matching the real ``mdl``.
    """

    def __init__(self, plant=PLANT, fs=FS, noise_channels=(NOISE_CH,)):
        self.sample_rate = float(fs)
        self._b, self._a = sig.bilinear(plant.num, plant.den, fs)
        self._zi = np.zeros(max(len(self._a), len(self._b)) - 1)
        self.noise_channels = set(noise_channels)
        self._pending = None
        self._captured: list = []
        self.writes: dict = {}

    def write(self, channel, value):
        self.writes[channel] = value

    def fetch_later(self, t0, t1, names):
        want = int(round((t1 - t0) * self.sample_rate))
        self._pending = (list(names), want)
        return lambda: self._captured

    def run(self, cycles, excitations=None, excitation_data=None):
        n = int(cycles)
        drive = np.zeros(n)
        noise = np.zeros(n)
        if excitations and excitation_data is not None:
            ed = np.asarray(excitation_data, dtype=float)
            for j, ch in enumerate(excitations):
                col = (ed[:, j] if ed.ndim == 2 else ed)[:n]
                if ch == EXC:
                    drive = drive + col
                elif ch in self.noise_channels:
                    noise = noise + col
        y, self._zi = sig.lfilter(self._b, self._a, drive, zi=self._zi)
        probes = {SENSOR: y + noise, DRIVE_MON: drive}
        if self._pending is not None:
            names, want = self._pending
            self._captured = [_Buf(probes.get(nm, np.zeros(n))[:want]) for nm in names]
            self._pending = None


def _backend(noise_floor=1e-10, **kw):
    return RTSfreerunBackend(
        mdl=MockRTSModel(), exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
        noise=[{"channel": NOISE_CH, "kind": "bosem", "params": {"floor": noise_floor}}],
        fs=FS, seed=0, **kw)


def _f0_q(model):
    p = np.roots(np.asarray(model.den, float))
    p = p[p.imag > 1e-9][0]
    w0 = abs(p)
    return w0 / (2 * np.pi), w0 / (2 * abs(p.real))


def _grid():
    f_all = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (f_all >= 0.3) & (f_all <= 10.0)
    return band, f_all[band]


def test_inject_read_frf_recovers_plant():
    band, freq = _grid()
    tw = _backend()
    tw.inject(EXC, multisine_from_psd(np.ones_like(freq), FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read([SENSOR, DRIVE_MON, EXC], NPERSEG * N_PERIODS / FS)
    assert seg[SENSOR].size == NPERSEG * N_PERIODS
    # X = drive-monitor probe, Y = sensor  ->  H = G
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg[DRIVE_MON], seg[SENSOR], FS, NPERSEG, band)
    fit = GMLEstimator().fit(freq, H, H_err, TFModel.from_resonances([(1.3, 7.0)], 130.0))
    f0, q = _f0_q(fit)
    assert abs(f0 - 1.0) < 0.05 and abs(q - 10.0) / 10.0 < 0.3


def test_full_campaign_via_loop_recovers_plant():
    backend = _backend()
    cfg = {
        "run": {"name": "rts", "excitation_mode": "sequential"},
        "channels": {"excitation": {"POS": EXC}, "readback": {"POS": SENSOR},
                     "drive": {"POS": DRIVE_MON}},
        "measurement": {"fs": FS, "freq_min": 0.3, "freq_max": 10.0,
                        "segment_duration": NPERSEG / FS, "n_segments": N_PERIODS,
                        "px_total": 1.0},
        "strategy": {"estimator": "gml", "input_designer": "pintelon_schoukens",
                     "n_design_iter": 3},
        "priors": {"POS": {"resonances": [[1.3, 7.0]], "gain": 130}},
        "safety": {"actuator_sat": 1e12, "rms_ceiling": {"POS": 1e12}, "ramp_down_secs": 2.0},
        "stop_criteria": {"uncertainty_target": 0.02, "max_iter": 2},
    }
    watchdog = Watchdog(backend, SafetyLimits.from_config(cfg))
    loop = SysIDLoop(backend, GMLEstimator(), PintelonSchoukensDesigner(), watchdog)
    result = loop.run(cfg, RunConfig(raw=cfg).build_priors(), seed=0)
    f0, q = _f0_q(result.models["POS"])
    assert abs(f0 - 1.0) < 0.05 and abs(q - 10.0) / 10.0 < 0.3


def test_decimated_read_recovers_plant():
    """Model runs at fs_model; sysID analyses at an integer-decimated fs."""
    fs_model, fs_sysid, nper = 512.0, 256.0, 4096
    tw = RTSfreerunBackend(
        mdl=MockRTSModel(fs=fs_model), exc_channels={EXC: "POS"},
        readback_channels={SENSOR: "POS"},
        noise=[{"channel": NOISE_CH, "kind": "bosem", "params": {"floor": 1e-10}}],
        fs=fs_sysid, seed=0)
    f_all = np.fft.rfftfreq(nper, 1 / fs_sysid)
    band = (f_all >= 0.3) & (f_all <= 10.0); freq = f_all[band]
    tw.inject(EXC, multisine_from_psd(np.ones_like(freq), fs_sysid, nper, 4, freq, seed=0), fs_sysid)
    seg = tw.read([SENSOR, DRIVE_MON], nper * 4 / fs_sysid)
    assert seg[SENSOR].size == nper * 4                     # exact length after decimation
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg[DRIVE_MON], seg[SENSOR], fs_sysid, nper, band)
    f0, q = _f0_q(GMLEstimator().fit(freq, H, H_err, TFModel.from_resonances([(1.3, 7.0)], 130.0)))
    assert abs(f0 - 1.0) < 0.08 and abs(q - 10.0) / 10.0 < 0.4


def test_realistic_noise_is_injected():
    tw = _backend(noise_floor=1e-6)
    tw.inject(EXC, np.zeros(int(NPERSEG * N_PERIODS)), FS)  # no drive
    seg = tw.read([SENSOR], NPERSEG * N_PERIODS / FS)
    assert np.std(seg[SENSOR]) > 0.0                        # noise alone drives the sensor


def test_from_config_builds_channel_maps():
    cfg = {"channels": {"excitation": {"POS": EXC}, "readback": {"POS": SENSOR},
                        "drive": {"POS": DRIVE_MON}},
           "rtsfreerun": {"model": "x1hsts", "warmup_s": 0.0,
                          "noise": [{"channel": NOISE_CH, "kind": "bosem"}]}}
    tw = RTSfreerunBackend.from_config(cfg, fs=FS, mdl=MockRTSModel())
    assert tw.exc_channels == {EXC: "POS"} and tw.readback_channels == {SENSOR: "POS"}
    assert tw.noise[0]["channel"] == NOISE_CH


def test_ramp_down_and_snapshot_restore():
    tw = _backend()
    drive = np.ones(int(2 * FS))
    tw.inject(EXC, drive, FS)
    snap = tw.snapshot_state([EXC])
    tw.ramp_down(EXC, 1.0)
    assert tw._drives[EXC][-1] == 0.0          # tapered to zero
    tw.restore_state(snap)
    np.testing.assert_array_equal(tw._drives[EXC], drive)


def test_build_rtsfreerun_backend_requires_model():
    with pytest.raises(ConfigError):
        RunConfig(raw={"channels": {"excitation": {"POS": EXC}, "readback": {"POS": SENSOR}},
                      "measurement": {"fs": FS, "freq_min": 0.1, "freq_max": 5.0,
                                      "segment_duration": 4.0, "px_total": 1.0},
                      "strategy": {"estimator": "gml", "input_designer": "pintelon_schoukens"},
                      "safety": {"actuator_sat": 1.0, "rms_ceiling": {"POS": 1.0}},
                      "stop_criteria": {"uncertainty_target": 0.05},
                      "rtsfreerun": {}}).build_rtsfreerun_backend()


@pytest.mark.skipif(importlib.util.find_spec("x1hsts") is None,
                    reason="rtsfreerun x1hsts model not installed (built on the twin box)")
def test_real_x1hsts_smoke(x1hsts_model):  # pragma: no cover - runs only where the twin is built
    # Share the one-per-process model instance (see conftest.x1hsts_model).
    tw = RTSfreerunBackend(mdl=x1hsts_model, exc_channels={EXC: "POS"},
                           readback_channels={SENSOR: "POS"})
    assert tw.fs_model > 0
