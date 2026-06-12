"""TDD Task 1 — Bayesian/MAP estimator core.

test_update_recovers_truth_and_shrinks:
  Verifies that bayesian_update, starting from an offset prior, converges toward
  the true model over 6 noisy measurement steps, and that the posterior variance
  shrinks strictly at every step.
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


def test_update_recovers_truth_and_shrinks():
    true = TFModel.from_resonances([(1.0, 20.0)], gain=1.0)
    prior = TFModel.from_resonances([(0.7, 10.0)], gain=0.6)

    freq = np.linspace(0.2, 3.0, 400)
    rng = np.random.default_rng(0)
    n = len(freq)

    Lambda = prior_precision(prior, 0.5)
    model = prior

    max_variances = []
    response_errors = []

    for _ in range(6):
        H = true.eval(freq)
        H_err = 0.01 * np.abs(H)
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * H_err / np.sqrt(2)
        H_meas = H + noise
        model, Lambda = bayesian_update(freq, model, H_meas, H_err, Lambda)

        Sigma = np.linalg.inv(Lambda)
        max_variances.append(float(np.max(np.diag(Sigma))))

        H_fit = model.eval(freq)
        response_errors.append(float(np.max(np.abs(H_fit - H) / np.abs(H))))

    # Print sequences for sanity-checking by the caller.
    print("\nmax_variances:    ", [f"{v:.4e}" for v in max_variances])
    print("response_errors:  ", [f"{e:.4f}" for e in response_errors])

    # Max-variance must be strictly decreasing every step (posterior accumulates
    # information with each measurement batch; this is unconditional).
    for i in range(len(max_variances) - 1):
        assert max_variances[i] > max_variances[i + 1], (
            f"Variance did not shrink at step {i}: "
            f"{max_variances[i]:.4e} -> {max_variances[i+1]:.4e}"
        )

    # Response error must show an overall downward trend (first ≥ last) and
    # end well below 2%.  Strict pairwise monotonicity is not required: once
    # the model is at the noise floor (~0.3–0.5 %), successive noise
    # realisations cause small step-to-step fluctuations.
    assert response_errors[0] >= response_errors[-1], (
        f"Response error did not decrease overall: "
        f"{response_errors[0]:.4f} → {response_errors[-1]:.4f}"
    )
    assert response_errors[-1] < 0.02, (
        f"Final response error {response_errors[-1]:.4f} >= 0.02"
    )
    # Every individual step must stay well below the prior mismatch (sanity).
    assert all(e < 0.05 for e in response_errors), (
        f"Some response error exceeded 5 %: {response_errors}"
    )

    # Model order must be preserved.
    assert model.num.shape == true.num.shape, (
        f"num shape mismatch: {model.num.shape} != {true.num.shape}"
    )
    assert model.den.shape == true.den.shape, (
        f"den shape mismatch: {model.den.shape} != {true.den.shape}"
    )
