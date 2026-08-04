"""Phase 2: the Pintelon-Schoukens maximum-likelihood estimator (GMLEstimator).

The ML fit minimises ``sum |H_meas - H(theta)|^2 / H_err^2`` to convergence.
These tests show it is (a) exact on clean data for both the physical
``ResonatorModel`` and the ``num/den`` ``TFModel`` (Sanathanan-Koerner path),
(b) unbiased and **efficient** — its scatter matches the Cramer-Rao bound — on
Monte-Carlo noisy data, and (c) drops into the loop as ``estimator="gml"``.
"""

from __future__ import annotations

import numpy as np

from system_ident.backends.twin import TwinBackend
from system_ident.design.pintelon import PintelonSchoukensDesigner
from system_ident.estimators.bayesian import ml_fit
from system_ident.estimators.gml import GMLEstimator
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel, pole_pair_f0_Q
from system_ident.plant import SuspensionPlant, double_pendulum
from system_ident.resonator import ResonatorModel
from system_ident.safety import SafetyLimits, Watchdog

FS = 32.0


def _freq():
    return np.linspace(0.1, 5.0, 400)


def _tf_qs(model):
    poles = np.roots(np.asarray(model.den, dtype=float))
    return sorted(pole_pair_f0_Q(p.real, p.imag)[1] for p in poles[poles.imag > 1e-9])


# --------------------------------------------------------------------------- #
# clean-data recovery
# --------------------------------------------------------------------------- #

def test_resonator_clean_recovery_single_and_two_mode():
    freq = _freq()
    est = GMLEstimator()
    for res in ([(0.6, 20.0)], [(0.6, 20.0), (1.5, 30.0)]):
        true = ResonatorModel.from_resonances(res, 300.0)
        H = true.eval(freq)
        H_err = 1e-6 * np.abs(H)
        prior = ResonatorModel.from_resonances([(f * 1.1, q * 0.7) for f, q in res], 250.0)
        fit = est.fit(freq, H, H_err, prior)
        np.testing.assert_allclose(np.sort(fit.f0), [r[0] for r in res], rtol=1e-3)
        np.testing.assert_allclose(np.sort(fit.Q), [r[1] for r in res], rtol=1e-3)


def test_tfmodel_clean_recovery_via_sanathanan_koerner():
    freq = _freq()
    true = double_pendulum()                       # 0.6 Hz Q20, 1.5 Hz Q30
    H = true.eval(freq)
    H_err = 1e-6 * np.abs(H)
    prior = TFModel.from_resonances([(0.55, 14.0), (1.6, 22.0)], 250.0)

    fit = GMLEstimator().fit(freq, H, H_err, prior)
    # gauge-invariant: the response matches, and both Qs are recovered
    np.testing.assert_allclose(fit.eval(freq), H, rtol=1e-4)
    qs = _tf_qs(fit)
    np.testing.assert_allclose(qs, [20.0, 30.0], rtol=1e-3)


# --------------------------------------------------------------------------- #
# Monte-Carlo: unbiased + attains the Cramer-Rao bound
# --------------------------------------------------------------------------- #

def _monte_carlo(res, rel_err=0.02, n_real=400, seed=0):
    freq = _freq()
    true = ResonatorModel.from_resonances(res, 200.0)
    H = true.eval(freq)
    H_err = rel_err * np.abs(H)
    _, cov = ml_fit(freq, H, H_err, true)          # CRLB at the true model
    rng = np.random.default_rng(seed)
    thetas = np.empty((n_real, true.params.size))
    for i in range(n_real):
        noise = (rng.standard_normal(H.size) + 1j * rng.standard_normal(H.size))
        H_meas = H + noise * H_err / np.sqrt(2.0)
        fit, _ = ml_fit(freq, H_meas, H_err, true)
        thetas[i] = fit.params
    return true.params, thetas, np.sqrt(np.diag(cov))


def test_ml_is_unbiased_and_efficient_single_mode():
    theta0, thetas, crlb = _monte_carlo([(0.8, 25.0)])
    bias = np.abs(thetas.mean(0) - theta0) / np.abs(theta0)
    assert np.all(bias < 0.02)                      # unbiased
    ratio = thetas.std(0) / crlb
    assert np.all(np.abs(ratio - 1.0) < 0.2)        # attains Cramer-Rao (~20%)


def test_ml_is_unbiased_and_efficient_two_mode():
    theta0, thetas, crlb = _monte_carlo([(0.6, 20.0), (1.5, 30.0)], n_real=500)
    bias = np.abs(thetas.mean(0) - theta0) / np.abs(theta0)
    assert np.all(bias < 0.03)
    ratio = thetas.std(0) / crlb
    assert np.all(np.abs(ratio - 1.0) < 0.25)


# --------------------------------------------------------------------------- #
# end-to-end: estimator="gml" in the loop, periodic measurement
# --------------------------------------------------------------------------- #

def _config():
    return {
        "run": {"excitation_mode": "sequential"},
        "channels": {
            "excitation": {"POS": "X1:EXC_POS"},
            "readback": {"POS": "X1:RESP_POS"},
        },
        "measurement": {
            "fs": FS, "freq_min": 0.1, "freq_max": 5.0,
            "segment_duration": 64.0, "n_segments": 8, "px_total": 1.0,
            "mode": "periodic", "n_transient": 1,
        },
        "strategy": {"n_design_iter": 3},
        "safety": {
            "actuator_sat": 1e3, "rms_ceiling": {"POS": 1e3}, "ramp_down_secs": 2.0,
        },
        "stop_criteria": {"uncertainty_target": 0.02, "max_iter": 4},
    }


def test_loop_with_gml_estimator_and_periodic_mode():
    config = _config()
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    backend = TwinBackend.from_config(config, plant, fs=FS, sensor_asd=1e-6, seed=0)
    wd = Watchdog(backend, SafetyLimits.from_config(config))
    loop = SysIDLoop(backend, GMLEstimator(), PintelonSchoukensDesigner(), wd)

    prior = TFModel.from_resonances([(0.55, 14.0), (1.6, 22.0)], gain=250.0)
    result = loop.run(config, {"POS": prior}, seed=1)

    assert not result.aborted
    freq = np.linspace(0.15, 4.5, 200)
    truth = double_pendulum().eval(freq)

    def err(m):
        return np.median(np.abs(m.eval(freq) / truth - 1.0))

    assert err(result.models["POS"]) < err(prior)
    qs = _tf_qs(result.models["POS"])
    assert len(qs) == 2
    assert abs(qs[0] - 20.0) / 20.0 < 0.2
    assert abs(qs[1] - 30.0) / 30.0 < 0.2
