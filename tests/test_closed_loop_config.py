"""First-class closed-loop: config parsing, both injection points, the warning."""
import numpy as np
import pytest

from system_ident.backends.twin import TwinBackend
from system_ident.config import RunConfig
from system_ident.estimators.gml import GMLEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.plant import SuspensionPlant

WC = 2 * np.pi * 12.0
CONTROLLER = {"num": [2.0e-3, 0.0], "den": [1.0 / WC, 1.0]}  # velocity damping C(s)


def _f0_q(model):
    p = np.roots(np.asarray(model.den, float))
    p = p[p.imag > 1e-9][0]
    w0 = abs(p)
    return w0 / (2 * np.pi), w0 / (2 * abs(p.real))


def _cfg(injection_point="after_controller", with_drive=False):
    channels = {
        "excitation": {"POS": "C1:EXC"},
        "readback": {"POS": "C1:RSP"},
    }
    if with_drive:
        channels["drive"] = {"POS": "C1:DRV"}     # after-controller drive monitor
    return {
        "run": {"name": "cl", "excitation_mode": "sequential"},
        "channels": channels,
        "measurement": {"fs": 32, "freq_min": 0.1, "freq_max": 5.0,
                        "segment_duration": 64.0, "n_segments": 8, "px_total": 1.0},
        "strategy": {"estimator": "gml", "input_designer": "pintelon_schoukens",
                     "n_design_iter": 3},
        "twin": {"sensor_asd": 1e-4, "disturbance_asd": 1e-4,
                 "plant": {"POS": {"resonances": [[1.0, 20]], "gain": 100}},
                 "controllers": {"POS": CONTROLLER},
                 "injection_point": injection_point},
        "priors": {"POS": {"resonances": [[1.3, 14]], "gain": 130}},  # ~30% off
        "safety": {"actuator_sat": 1e6, "rms_ceiling": {"POS": 1e6}, "ramp_down_secs": 2.0},
        "stop_criteria": {"uncertainty_target": 0.02, "max_iter": 2},
    }


def test_config_parses_controllers_injection_and_drive():
    rc = RunConfig(raw=_cfg("before_controller", with_drive=True))
    tw = rc.build_twin_backend(seed=0)
    assert tw.controllers is not None and "POS" in tw.controllers
    num, den = tw.controllers["POS"]
    np.testing.assert_allclose(num, [2.0e-3, 0.0])
    assert tw.injection_point["POS"] == "before_controller"
    assert tw.drive_channels == {"C1:DRV": "POS"}


@pytest.mark.parametrize("injection_point,with_drive", [
    ("after_controller", False),    # twin: excitation read-back already gives u
    ("before_controller", True),    # needs the explicit drive monitor channel
])
def test_closed_loop_campaign_recovers_open_loop(injection_point, with_drive):
    cfg = _cfg(injection_point, with_drive)
    rc = RunConfig(raw=cfg)
    backend = rc.build_twin_backend(seed=0)
    priors = rc.build_priors()
    watchdog = rc.build_watchdog(backend)
    loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(), watchdog)
    result = loop.run(rc.raw, priors, seed=0)

    f0, q = _f0_q(result.models["POS"])
    # the loop must recover the OPEN-loop plant (1 Hz, Q=20), not closed-loop T
    assert abs(f0 - 1.0) / 1.0 < 0.05, f"{injection_point}: f0={f0:.3f}"
    assert abs(q - 20.0) / 20.0 < 0.25, f"{injection_point}: Q={q:.1f}"


def test_before_controller_injection_recovers_G_at_frf_level():
    """The reference-based FRF recovers G with the excitation injected before C."""
    fs, nperseg, n_per = 32.0, 2048, 8
    true = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    plant = SuspensionPlant({"POS": true}, fs)
    tw = TwinBackend(plant, {"E": "POS"}, {"R": "POS"}, fs=fs, sensor_asd=1e-4,
                     disturbance_asd=1e-4, seed=3,
                     controllers={"POS": (CONTROLLER["num"], CONTROLLER["den"])},
                     injection_point="before_controller",
                     drive_channels={"U": "POS"})
    f_all = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (f_all >= 0.1) & (f_all <= 5.0); freq = f_all[band]
    Pxx = np.ones_like(freq)
    tw.inject("E", multisine_from_psd(Pxx, fs, nperseg, n_per, freq, seed=0), fs)
    seg = tw.read(["U", "R"], nperseg * n_per / fs)        # U = after-controller drive
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["U"], seg["R"], fs, nperseg, band)

    ex = np.isfinite(H_err)
    ref = true.eval(freq)
    # near the resonance the FRF tracks the OPEN-loop plant
    near = ex & (np.abs(freq - 1.0) < 0.3)
    rel = np.abs(H[near] - ref[near]) / np.abs(ref[near])
    assert np.median(rel) < 0.1


def test_warns_on_before_controller_without_drive_channel():
    cfg = _cfg("before_controller", with_drive=False)
    rc = RunConfig(raw=cfg)
    backend = rc.build_twin_backend(seed=0)
    loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(),
                     rc.build_watchdog(backend))
    with pytest.warns(UserWarning, match="before-controller"):
        loop.run(rc.raw, rc.build_priors(), seed=0)
