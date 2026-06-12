# Recursive Bayesian (MAP/EKF) System Identification — Spec + Plan

> Design approved 2026-06-12. Adds a prior-anchored, recursive system-ID loop so the
> estimate starts at a (possibly bad) prior and converges to truth over several
> measurements, with the posterior uncertainty shrinking each step.

## Why

The existing loop uses a flat **broadband-first** excitation + a *global* `invfreqs`
least-squares fit, which ignores the prior's values and recovers the model in pass 0.
That is good engineering but not the "watch it converge from a bad guess" behavior we
want to study/teach. This adds a second loop mode that genuinely iterates from the prior.

## Algorithm (per DoF)

Parameters `θ` in the existing **gauge**: full coefficient vector `model.params` =
`[num (n_num), den (n_den)]`, gauged so the leading denominator coeff `den[0]=1`, then
the fixed index `n_num` removed. So `θ` (reduced) = `[num[0..n_num-1], den[1..n_den-1]]`,
length `n_par = n_num + n_den - 1`. This matches `fisher.fisher_matrix` exactly.

State carried across passes: posterior precision `Λ` (n_par × n_par) and mean model.
Initialised to the prior: `Λ = Λ₀`, mean = `θ₀` (the prior model).

Per pass `k`:
1. **Design** excitation `Pxx` from the *current mean* model (`designer.design`, always
   `n_iter = n_design_iter` — no broadband-first override).
2. **Inject** (Tukey-windowed) + read + estimate `H_meas(f) ± σ(f)` via the existing
   `_estimate_tf`.
3. **MAP / Gauss-Newton update** (linearise `G` about the current `θ̂`):
   - Jacobian `J(f) = ∂H/∂θ` (reduced) from `TFModel.jacobian` with `logflag` all-False
     (absolute derivatives), dropping the fixed row `n_num` — same construction as
     `fisher_matrix`.
   - per-bin weight `wt(f) = 1/σ(f)²` (zero weight where `σ` is non-finite or 0).
   - measurement info `𝓘[i,j] = Σ_f wt · Re(conj(J_i)·J_j)`
   - gradient `b[i] = Σ_f wt · Re(conj(J_i)·r)`, residual `r(f) = H_meas(f) − G(f;θ̂)`
   - `Λ ← Λ + 𝓘`; `Δθ = solve(Λ, b)`; `θ̂ ← θ̂ + Δθ`
   - posterior covariance `Σ = Λ⁻¹`; fractional uncertainty `√diag(Σ)/|θ̂|` (gauge of
     `loop._frac_uncertainty`).
4. Rebuild the model: `num = θ̂[:n_num]`, `den = [1.0, *θ̂[n_num:]]`.

Prior precision: `Λ₀ = diag(1/σ₀²)` with `σ₀_i = prior_uncertainty · max(|θ₀_i|, ε·‖θ₀‖∞)`
(`ε = 1e-3` floor so a zero-valued prior coeff is not frozen). `prior_uncertainty` is a
config scalar (e.g. 0.5 = a weak/uncertain prior that crawls far; 0.05 = a confident one).

## Files

- **Create** `src/system_ident/estimators/bayesian.py`:
  - `reduced_params(model) -> (theta, n_num)` and `model_from_reduced(theta, n_num, n_den)`
    (gauge helpers).
  - `prior_precision(model0, prior_uncertainty, floor=1e-3) -> np.ndarray` (Λ₀).
  - `bayesian_update(freq, model, H_meas, H_err, Lambda) -> (model_new, Lambda_new)`.
  - `frac_uncertainty(model, Lambda) -> float` (max gauge-relative posterior σ).
- **Modify** `src/system_ident/loop.py`: add a `bayesian` loop mode. Read
  `loop_mode = config["strategy"].get("loop", "broadband_ls")` and
  `prior_uncertainty = float(config["strategy"].get("prior_uncertainty", 0.5))`.
  In `bayesian` mode: init `Lambda[d] = prior_precision(priors[d], prior_uncertainty)`;
  each pass design from the current model (always optimal), measure, `bayesian_update`,
  report uncertainty from `Lambda[d]`. Keep the existing path as `broadband_ls` (default).
- **Modify** `src/system_ident/config.py`: allow `strategy.loop ∈ {broadband_ls, bayesian}`
  (default `broadband_ls`); no new required keys.
- **Create** `src/system_ident/configs/twin_demo_bayesian.yml`: a bad-prior + noisy
  (`disturbance_asd`, `sensor_asd`) config with `strategy.loop: bayesian`,
  `prior_uncertainty: 0.5`, `t_ramp`, several `max_iter`.

## Tasks (TDD)

### Task 1: `bayesian.py` core + unit tests
- [ ] Write `tests/test_step10_bayesian.py::test_update_recovers_truth_and_shrinks`:
  true = 1 Hz Q20 resonance; prior offset; synthesise `H_meas = true.eval(freq)` with
  1% complex noise and `H_err=0.01|H|`; `Λ = prior_precision(prior, 0.5)`, `model=prior`;
  apply `bayesian_update` ~6× (fresh noise each step). Assert: `max|G_fit−G_true|/|G_true|`
  decreases and ends small (<2%); `max diag(Σ)` strictly decreases each step; the rebuilt
  model keeps the prior's order. Run → fails (module absent).
- [ ] Implement `bayesian.py` (gauge helpers, `prior_precision`, `bayesian_update`,
  `frac_uncertainty`) per the math above. Reuse `TFModel.jacobian`.
- [ ] Test passes; full suite green.
- [ ] Commit `feat(estimators): recursive Bayesian/MAP update`.

### Task 2: `bayesian` loop mode + config + integration test
- [ ] Write `tests/test_step10_bayesian.py::test_bayesian_loop_crawls_from_bad_prior`:
  noisy twin (`disturbance_asd`, `sensor_asd`), 1 Hz Q20 plant, **bad** prior (0.6 Hz Q6),
  `strategy.loop: bayesian`, `max_iter≥5`, capture listener snapshots. Assert: pass-0 mean
  is still near the prior (NOT instantly truth — `est f0` closer to prior than truth, or
  the per-pass move is bounded); the final mean is near truth (f0 within a few %);
  `max_frac_uncertainty` decreases monotonically across passes. Run → fails.
- [ ] Implement the `bayesian` branch in `loop.py` + the `strategy.loop` config option +
  `twin_demo_bayesian.yml`. Default mode unchanged (existing 62 tests stay green).
- [ ] Test passes; full suite green.
- [ ] Commit `feat(loop): prior-anchored Bayesian loop mode`.

## Verification (controller, before docs)

Run a bad-prior noisy campaign in both modes; confirm the Bayesian mode shows a visible
crawl (prior → truth over passes) with monotonically shrinking uncertainty, and the
broadband_ls mode still recovers in pass 0. Capture the per-pass `(f0, Q, frac_unc, coh)`
table for the docs.

## Out of scope (later)

- Full iterated-EKF inner iterations per pass (we do one GN step per pass).
- Non-Gaussian / robust priors; correlated prior covariance (we use diagonal Σ₀).
- Stall detection when a pathologically sharp/wrong prior never illuminates the true mode
  (worth a dedicated pedagogical example later).
