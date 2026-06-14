"""Prior-robust first pass: design the opening excitation from the prior model
AND its (large) error bars, so a far prior still covers the true resonance.
"""

import numpy as np
from scipy.integrate import trapezoid

from system_ident.config import RunConfig
from system_ident.design.pintelon import optimal_excitation, prior_robust_excitation
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel

FS, NPERSEG = 32.0, 2048


def _grid():
    f_all = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (f_all >= 0.1) & (f_all <= 5.0)
    return band, f_all[band]


def test_prior_robust_spreads_power_off_the_prior():
    band, freq = _grid()
    Pyy = np.ones_like(freq)
    prior = TFModel.from_resonances([(1.5, 30.0)], 150.0)   # point estimate at 1.5 Hz
    Pxx_opt = optimal_excitation(freq, prior, Pyy, 1.0, n_iter=3)
    Pxx_rob = prior_robust_excitation(freq, prior, Pyy, 1.0, 0.5, n_iter=3)

    # both honour the power budget
    assert abs(trapezoid(Pxx_rob, freq) - 1.0) < 1e-6
    # the robust drive covers the plausible band: at 1.0 Hz (a true resonance the
    # +50% prior would miss) it carries real power the point-optimal does not
    i = int(np.argmin(np.abs(freq - 1.0)))
    assert Pxx_rob[i] > 10 * Pxx_opt[i]
    # u=0 collapses back to the point-optimal drive
    Pxx_zero = prior_robust_excitation(freq, prior, Pyy, 1.0, 0.0, n_iter=3)
    np.testing.assert_allclose(Pxx_zero, Pxx_opt, rtol=1e-9)


def _far_prior_cfg():
    return {
        "run": {"name": "far", "excitation_mode": "sequential"},
        "channels": {"excitation": {"POS": "E"}, "readback": {"POS": "R"}},
        "measurement": {"fs": FS, "freq_min": 0.1, "freq_max": 5.0,
                        "segment_duration": 64.0, "n_segments": 6, "px_total": 1.0},
        "strategy": {"estimator": "gml", "input_designer": "pintelon_schoukens",
                     "n_design_iter": 3, "prior_uncertainty": 0.5},
        "twin": {"sensor_asd": 1e-3,
                 "plant": {"POS": {"resonances": [[1.0, 20]], "gain": 100}}},
        "priors": {"POS": {"resonances": [[1.5, 30]], "gain": 150}},   # +50% off
        "safety": {"actuator_sat": 1e9, "rms_ceiling": {"POS": 1e9}},
        "stop_criteria": {"uncertainty_target": 1e-9, "max_iter": 6},
    }


def test_prior_robust_first_pass_cracks_a_far_prior():
    rc = RunConfig(raw=_far_prior_cfg())
    backend = rc.build_twin_backend(seed=0)
    loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(),
                     rc.build_watchdog(backend))
    result = loop.run(rc.raw, rc.build_priors(), seed=0)

    model = result.models["POS"]
    poles = np.roots(np.asarray(model.den, dtype=float))
    p = poles[poles.imag > 1e-9][0]
    f0 = abs(p) / (2 * np.pi)
    Q = abs(p) / (2 * abs(p.real))
    assert abs(f0 - 1.0) / 1.0 < 0.05, f"f0 stuck near the prior: {f0:.3f} Hz"
    assert abs(Q - 20.0) / 20.0 < 0.3, f"Q off: {Q:.1f}"
