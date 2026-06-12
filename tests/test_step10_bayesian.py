"""Tasks 1-2 — recursive Bayesian/MAP estimator + Bayesian loop mode.

The Bayesian mode refines a *good* prior with *weak* (energy-limited, low-SNR)
measurements: each pass takes one small, Levenberg-Marquardt-damped, backtracked
Gauss-Newton step, accumulates Fisher information (so the posterior covariance
shrinks), and crawls toward truth. By design it is conservative and robust (it
essentially never diverges); it does NOT relocate a resonance that sits far from
the prior — that is a local-minimum limitation of coefficient-space fitting, for
which a broadband sweep (``broadband_ls`` mode) should be used first. These tests
assert the robust, verified properties; they do not assert recovery from a far
prior.
"""

import numpy as np
import pytest

from system_ident.model import TFModel
from system_ident.estimators.bayesian import (
    prior_precision,
    bayesian_update,
    frac_uncertainty,
    reduced_params,
    model_from_reduced,
)


def _f0(model):
    d = np.asarray(model.den) / np.asarray(model.den)[0]
    p = np.roots(d)
    p = p[p.imag >= 0][0]
    return abs(p) / (2 * np.pi)


def _resp_err(model, true, freq):
    H = true.eval(freq)
    return float(np.max(np.abs(model.eval(freq) - H) / np.abs(H)))


def test_update_shrinks_covariance_and_refines_good_prior():
    """Repeated measurements shrink the covariance and refine a good prior.

    Asserts the robust properties: the (absolute) posterior covariance shrinks
    strictly every step as information accumulates, the fit gets closer to truth
    than the prior was, every step stays finite, and the model order is kept.
    """
    true = TFModel.from_resonances([(1.0, 20.0)], gain=100.0)
    prior = TFModel.from_resonances([(0.92, 15.0)], gain=80.0)  # good prior (peaks overlap)

    freq = np.linspace(0.3, 2.5, 300)
    rng = np.random.default_rng(0)
    peak = float(np.max(np.abs(true.eval(freq))))

    Lambda = prior_precision(prior, 0.4)
    model = prior
    err0 = _resp_err(prior, true, freq)
    max_var = []
    for _ in range(25):
        H = true.eval(freq)
        H_err = 0.03 * peak * np.ones_like(freq)  # realistic flat noise floor
        H_meas = H + (rng.standard_normal(len(freq)) + 1j * rng.standard_normal(len(freq))) * H_err / np.sqrt(2)
        model, Lambda = bayesian_update(freq, model, H_meas, H_err, Lambda)
        assert np.all(np.isfinite(model.den)) and np.all(np.isfinite(model.num))
        max_var.append(float(np.max(np.diag(np.linalg.inv(Lambda)))))

    # information accumulates -> absolute covariance shrinks monotonically
    for i in range(len(max_var) - 1):
        assert max_var[i] > max_var[i + 1], f"covariance grew at step {i}"
    # the estimate refined toward truth
    assert _resp_err(model, true, freq) < err0
    assert abs(_f0(model) - 1.0) < abs(_f0(prior) - 1.0)
    assert model.num.shape == true.num.shape and model.den.shape == true.den.shape


def test_step_is_robust_to_a_far_prior():
    """A far/bad prior must not blow up — the damped step keeps it finite/bounded.

    (It will stall rather than relocate the resonance; the point here is that the
    update is robust, not that it converges from anywhere.)
    """
    true = TFModel.from_resonances([(1.0, 20.0)], gain=100.0)
    prior = TFModel.from_resonances([(0.4, 5.0)], gain=10.0)  # very wrong
    freq = np.linspace(0.2, 5.0, 300)
    rng = np.random.default_rng(3)
    peak = float(np.max(np.abs(true.eval(freq))))
    Lambda = prior_precision(prior, 0.5)
    model = prior
    f0_start = _f0(model)
    for _ in range(30):
        H = true.eval(freq)
        H_err = 0.05 * peak * np.ones_like(freq)
        H_meas = H + (rng.standard_normal(len(freq)) + 1j * rng.standard_normal(len(freq))) * H_err / np.sqrt(2)
        model, Lambda = bayesian_update(freq, model, H_meas, H_err, Lambda)
        assert np.all(np.isfinite(model.den)), "estimate diverged"
    # bounded: the resonance did not run off to absurd frequencies
    assert 0.1 < _f0(model) < 5.0


def test_prior_strength_anchors_the_mean():
    """A stronger prior moves the mean LESS — impossible under measurement-only LS.

    Discriminates a true MAP update from a measurement-only fit: measurement-only
    LS converges to the same mean regardless of prior precision (the prior would
    affect only the covariance).
    """
    true = TFModel.from_resonances([(1.0, 20.0)], gain=1.0)
    prior = TFModel.from_resonances([(0.9, 16.0)], gain=0.8)
    freq = np.linspace(0.3, 2.5, 200)
    rng = np.random.default_rng(1)
    n = len(freq)
    peak = float(np.max(np.abs(true.eval(freq))))
    H = true.eval(freq)
    H_err = 0.05 * peak * np.ones_like(freq)
    H_meas = H + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * H_err / np.sqrt(2)
    Hp = prior.eval(freq)

    m_weak, _ = bayesian_update(freq, prior, H_meas, H_err, prior_precision(prior, 5.0))
    m_strong, _ = bayesian_update(freq, prior, H_meas, H_err, prior_precision(prior, 3e-4))
    move_weak = float(np.max(np.abs(m_weak.eval(freq) - Hp)))
    move_strong = float(np.max(np.abs(m_strong.eval(freq) - Hp)))
    print(f"\nmove_weak={move_weak:.4g}  move_strong={move_strong:.4g}")
    assert move_strong < 0.5 * move_weak, (
        f"prior strength did not anchor the mean (strong={move_strong:.4g}, weak={move_weak:.4g})"
    )


# ---------------------------------------------------------------------------
# Task 2 — Bayesian loop mode (end-to-end digital-twin campaign)
# ---------------------------------------------------------------------------

def test_bayesian_loop_refines_good_prior_with_weak_measurements():
    """End-to-end: a good prior + weak (low-SNR) measurements crawls toward truth.

    Plant 1.0 Hz Q20 gain100; prior 0.9 Hz Q15 gain80 (good, peaks overlap);
    sensor + disturbance noise; exploration floor on. Asserts: the first pass is
    still near the prior (it crawls, not jumps), the final model is much closer to
    truth than the prior, and the run is stable/finite.
    """
    from system_ident.config import RunConfig
    from system_ident.loop import SysIDLoop

    cfg = {
        "run": {"excitation_mode": "sequential"},
        "channels": {"excitation": {"POS": "C1:EXC"}, "readback": {"POS": "C1:RSP"}},
        "measurement": {"fs": 32, "freq_min": 0.1, "freq_max": 5.0,
                        "segment_duration": 64.0, "n_segments": 4, "px_total": 1.0, "t_ramp": 4.0},
        "twin": {"sensor_asd": 3e-3, "disturbance_asd": 3e-3,
                 "plant": {"POS": {"resonances": [[1.0, 20]], "gain": 100}}},
        "priors": {"POS": {"resonances": [[0.9, 15]], "gain": 80}},
        "strategy": {"estimator": "invfreqs", "input_designer": "pintelon_schoukens",
                     "n_design_iter": 3, "loop": "bayesian",
                     "prior_uncertainty": 0.4, "exploration": 0.3},
        "safety": {"actuator_sat": 1e9, "rms_ceiling": {"POS": 1e9}, "ramp_down_secs": 2.0},
        "stop_criteria": {"uncertainty_target": 1e-12, "max_iter": 15},
    }
    rc = RunConfig(raw=cfg)
    priors = rc.build_priors()
    backend = rc.build_twin_backend(seed=5)
    snaps = []
    loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(),
                     rc.build_watchdog(backend), listener=snaps.append)
    result = loop.run(rc.raw, priors, seed=5)

    true = backend.plant["POS"]
    prior = priors["POS"]
    freq = np.linspace(0.3, 2.5, 300)
    err_prior = _resp_err(prior, true, freq)

    f0_first = _f0(TFModel.from_dict({"num": snaps[0]["model_num"], "den": snaps[0]["model_den"]}))
    final = result.models["POS"]
    f0_final = _f0(final)

    # crawl: the first pass has not jumped to truth (still near the prior)
    assert abs(f0_first - 1.0) > 0.04, f"first-pass f0={f0_first:.3f} jumped to truth"
    # the estimate converged toward truth in resonance frequency, and refined
    # the overall response, and stayed stable.
    assert abs(f0_final - 1.0) < 0.05, f"final f0={f0_final:.3f} not near truth"
    assert abs(f0_final - 1.0) < abs(_f0(prior) - 1.0)
    assert _resp_err(final, true, freq) < err_prior
    assert np.all(np.isfinite(final.den))
    assert len(snaps) == 15
