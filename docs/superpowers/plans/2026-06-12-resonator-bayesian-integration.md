# Integrate ResonatorModel into the Bayesian path — Plan

> Move the Bayesian loop mode off polynomial-coefficient TFModel and onto the
> physical `ResonatorModel` (f0, Q, gain), so the estimator can relocate a far
> resonance and Q stays well-conditioned. Linear params (log-space deferred).
> The `broadband_ls` / `InvfreqsEstimator` path stays on TFModel, untouched.

## Why model-agnostic (gauge-free)
`ResonatorModel` params are physical and identifiable — no gauge. So the Bayesian
machinery should operate generically on a model that exposes:
`params -> theta`, `jacobian(freq) -> dH/dtheta (n_par, n_bin)`,
`eval(freq) -> H`, `with_params(theta) -> model`. `ResonatorModel` already
provides all four. (The TFModel-gauge versions in `fisher.py` stay for the
invfreqs/broadband path.)

## Tasks (TDD)

### Task 1: model-agnostic Bayesian core
Rewrite `src/system_ident/estimators/bayesian.py` to be gauge-free and operate on
the four-method protocol above (drop `reduced_params`/`model_from_reduced` and the
`n_num` gauge logic):
- `prior_precision(model0, unc, floor=1e-3)`: `theta0=model0.params`;
  `sigma0 = unc*max(|theta0|, floor*max|theta0|)`; `Lambda0 = diag(1/sigma0**2)`.
- `bayesian_update(freq, model, H_meas, H_err, Lambda, ...)`: identical LM logic
  to today (damped, `max_rel_step` cap, backtracking on the regularised MAP
  objective, `Lambda_new = Lambda + I`) but with `theta=model.params`,
  `J=model.jacobian(freq)` (no row drop), `model.with_params(theta_new)`.
- `frac_uncertainty(model, Lambda)`: `theta=model.params`; `max sqrt(diagΣ)/|theta|`.
- [ ] Rewrite `tests/test_step10_bayesian.py` to use `ResonatorModel`:
  - `test_update_shrinks_covariance_and_refines_good_prior`: true (1.0,Q20,g100),
    good prior (0.92,Q15,g80); 25 noisy passes; assert max-diag(Σ) strictly
    decreases, f0 & Q end closer to truth than the prior, finite, n_modes kept.
  - `test_far_prior_relocates`: far prior (0.6,Q6,g20); broadband-ish data; assert
    f0 crawls off the prior toward 1.0 (`f0_final > 0.75`, vs frozen-at-0.6 in
    coeff-space) and stays finite. (This is the headline win.)
  - `test_prior_strength_anchors_the_mean`: stronger prior moves the mean less.
- [ ] commit `feat(bayesian): gauge-free model-agnostic MAP (operates on ResonatorModel)`.

### Task 2: model-agnostic Fisher + optimal excitation for the design step
The Bayesian design step needs Fisher/dispersion on `ResonatorModel`. Add
gauge-free versions (new module `src/system_ident/resonator_design.py` or
functions in `fisher.py` guarded by model type):
- `fisher_information(freq, model, Pxx, Pyy, T_tot)`: `J=model.jacobian(freq)`;
  `gamma[i,j] = 2*Re trapezoid(conj(J_i) J_j * Pxx/Pyy, freq) * T_tot` (NO gauge
  row/col removal). `dispersion` and `optimal_excitation` as in `design/pintelon`
  but using this gauge-free Fisher.
- [ ] tests: Fisher SPD & invertible for a ResonatorModel; optimal excitation
  integrates to the power budget and concentrates near the resonances.
- [ ] commit.

### Task 3: wire ResonatorModel into the loop + config
- `config.py`: when `strategy.loop == "bayesian"`, `build_priors` returns
  `ResonatorModel.from_resonances(...)` per DoF (same resonance spec). `broadband_ls`
  keeps returning `TFModel`.
- `loop.py` `_measure_dof_bayesian`: design via the gauge-free optimal excitation
  on the current `ResonatorModel`; keep the exploration floor; inject/read
  unchanged (drive built from Pxx; twin simulates the true TFModel plant); fit via
  the model-agnostic `bayesian_update`; uncertainty via `frac_uncertainty`. The
  `_emit` snapshot uses `model.eval(freq)` and `model.to_tf()` num/den for the
  dashboard.
- `twin_demo_bayesian.yml`: unchanged values (good prior + weak), still valid.
- [ ] integration test: loop in bayesian mode with a good prior + weak
  measurements refines f0 & Q toward truth, uncertainty shrinks, first pass not at
  truth (crawl); a far-prior variant relocates (f0 moves substantially toward
  truth) rather than stalling.
- [ ] commit `feat(loop): Bayesian mode estimates physical (f0,Q,gain)`.

## Verification (controller)
Run a bad-prior and a good-prior campaign; confirm the ResonatorModel Bayesian
loop relocates the far prior (vs the coeff-space stall) and refines the good prior
with shrinking uncertainty. Capture the per-pass (f0, Q, unc) table for the docs.

## Out of scope (later)
- Log-space params (better conditioning / faster convergence).
- Migrating the broadband_ls/invfreqs path to ResonatorModel.
- Multi-mode ResonatorModel in the loop demo (start single-mode, then 2-mode).
