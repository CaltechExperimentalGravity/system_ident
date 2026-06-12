"""Step-7: the SysIDLoop orchestrator, end-to-end against the digital twin."""

from __future__ import annotations

import numpy as np
import pytest

from ligo_sysid.backends.twin import TwinBackend
from ligo_sysid.design.pintelon import PintelonSchoukensDesigner
from ligo_sysid.estimators.invfreqs import InvfreqsEstimator
from ligo_sysid.loop import SysIDLoop
from ligo_sysid.model import TFModel
from ligo_sysid.plant import SuspensionPlant, double_pendulum
from ligo_sysid.safety import SafetyLimits, Watchdog

FS = 32.0


def _config(actuator_sat=1e3, rms_ceiling=1e3, target=0.05, max_iter=4):
    return {
        "run": {"excitation_mode": "sequential"},
        "channels": {
            "excitation": {"POS": "C1:EXC_POS"},
            "readback": {"POS": "C1:RESP_POS"},
        },
        "measurement": {
            "fs": FS, "freq_min": 0.1, "freq_max": 5.0,
            "segment_duration": 64.0, "n_segments": 8, "px_total": 1.0,
        },
        "strategy": {"n_design_iter": 3},
        "safety": {
            "actuator_sat": actuator_sat,
            "rms_ceiling": {"POS": rms_ceiling},
            "ramp_down_secs": 2.0,
        },
        "stop_criteria": {"uncertainty_target": target, "max_iter": max_iter},
    }


def _build(config, sensor_asd=1e-9, seed=0):
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    backend = TwinBackend.from_config(config, plant, fs=FS, sensor_asd=sensor_asd, seed=seed)
    limits = SafetyLimits.from_config(config)
    wd = Watchdog(backend, limits)
    loop = SysIDLoop(backend, InvfreqsEstimator(), PintelonSchoukensDesigner(), wd)
    return backend, loop


def _perturbed_prior():
    # shifted resonances + gain; the loop should pull this back toward truth
    return TFModel.from_resonances([(0.55, 18.0), (1.6, 28.0)], gain=250.0)


def _resp_error(model, freq):
    truth = double_pendulum().eval(freq)
    return np.median(np.abs(model.eval(freq) / truth - 1.0))


def test_loop_improves_fit_and_finishes_cleanly():
    config = _config()
    backend, loop = _build(config, sensor_asd=1e-9)
    prior = _perturbed_prior()

    result = loop.run(config, {"POS": prior}, seed=1)

    assert not result.aborted
    assert result.history  # recorded per-DoF iterations
    freq = np.linspace(0.15, 4.5, 200)
    assert _resp_error(result.models["POS"], freq) < _resp_error(prior, freq)
    # teardown ran the handoff -> injected drive cleared (monitor the drive
    # channel, which carries no sensor noise)
    np.testing.assert_array_equal(
        backend.read(["C1:EXC_POS"], 4.0)["C1:EXC_POS"], 0.0
    )


def test_loop_done_flag_with_loose_target():
    config = _config(target=1e12, max_iter=3)
    _, loop = _build(config, sensor_asd=1e-9)
    result = loop.run(config, {"POS": _perturbed_prior()}, seed=2)
    assert result.done and not result.aborted
    # an absurdly loose target is met in the first iteration
    assert max(r.iteration for r in result.history) == 0


def test_refinement_passes_accumulate_without_degrading_fit():
    # With an unreachable target the loop runs every pass: a broadband first
    # pass then optimal-designer refinement passes. Because measurements are
    # combined by inverse-variance weighting across passes, the concentrated
    # optimal drive must NOT erode the broadband fit established by pass 0.
    config = _config(target=0.0, max_iter=4)
    prior = _perturbed_prior()
    freq = np.linspace(0.15, 4.5, 200)

    for seed in range(3):
        _, loop = _build(config, sensor_asd=1e-2)
        result = loop.run(config, {"POS": prior}, seed=seed)

        assert not result.aborted
        assert sorted({r.iteration for r in result.history}) == [0, 1, 2, 3]
        assert np.all(np.isfinite(result.models["POS"].den))
        # after all refinement passes the fit stays improved (~0.078) rather
        # than degrading past 0.12 as it did without measurement accumulation
        err = _resp_error(result.models["POS"], freq)
        assert err < _resp_error(prior, freq)  # better than the prior
        assert err < 0.10                       # no concentrated-drive degradation

        # the convergence metric reflects accumulated information: it tightens
        # monotonically as passes are folded in (vs. a single-pass snapshot)
        uncs = [r.max_frac_uncertainty for r in result.history]
        assert all(b <= a * (1 + 1e-9) for a, b in zip(uncs, uncs[1:]))
        assert uncs[-1] < uncs[0]


def test_accumulate_inverse_variance_combination():
    n = 4
    accum = {"w": np.zeros(n), "wH": np.zeros(n, dtype=complex)}

    # pass 1: H=1, err=0.2 everywhere
    H1, e1 = SysIDLoop._accumulate(accum, np.ones(n, complex), np.full(n, 0.2))
    np.testing.assert_allclose(H1, 1.0)
    np.testing.assert_allclose(e1, 0.2)

    # pass 2: H=3, err=0.2 -> combined mean 2.0, error 0.2/sqrt(2)
    H2, e2 = SysIDLoop._accumulate(accum, 3 * np.ones(n, complex), np.full(n, 0.2))
    np.testing.assert_allclose(H2, 2.0)
    np.testing.assert_allclose(e2, 0.2 / np.sqrt(2))

    # a zero-weight (unexcited) pass leaves the estimate unchanged
    H3, e3 = SysIDLoop._accumulate(accum, 99 * np.ones(n, complex), np.full(n, np.inf))
    np.testing.assert_allclose(H3, 2.0)
    np.testing.assert_allclose(e3, 0.2 / np.sqrt(2))


def test_loop_aborts_on_actuator_saturation():
    config = _config(actuator_sat=1e-12)  # any real drive breaches immediately
    backend, loop = _build(config, sensor_asd=1e-9)

    result = loop.run(config, {"POS": _perturbed_prior()}, seed=3)

    assert result.aborted
    assert "actuator saturation" in result.abort_reason
    # safe-state handoff cleared the injected drive
    np.testing.assert_array_equal(
        backend.read(["C1:EXC_POS"], 4.0)["C1:EXC_POS"], 0.0
    )
