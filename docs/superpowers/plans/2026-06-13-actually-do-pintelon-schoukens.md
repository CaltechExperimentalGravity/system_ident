# Plan: Actually do Pintelon & Schoukens (leakage-free measurement + ML fit)

## Context

The user asked: P&S claim the *optimum* frequency-domain identification method — so why have we been flailing on Q estimation? Is their method wrong?

**Their method is correct and optimal. We have not been using it.** Investigation of the
code confirms we adopted only P&S's *optimal excitation design* (the dispersion-function
iteration in `design/pintelon.py` / `resonator_design.py`, which produces an optimal
excitation power spectrum `Pxx`). Everything downstream departs from P&S:

| Step | Pintelon & Schoukens | What we built |
|---|---|---|
| Excitation realization | periodic random-phase **multisine** | one-off **random** colored Gaussian noise (`excitation.py:60`, `rng.standard_normal`) |
| FRF measurement | leakage-free synchronous DFT over **integer periods**; noise covariance from period-to-period variance | **Hann-windowed Welch** on a random record (`loop.py` `_estimate_tf`/`_welch`, `noverlap=0`) → **spectral leakage** |
| Parametric fit | **maximum likelihood** (achieves Cramér–Rao) | weighted-LS `invfreqs`; `GMLEstimator` (true ML) is a `NotImplementedError` stub |

**Why this caused the flailing:** windowing a *non-periodic* random record leaks the sharp
resonance's energy into neighboring bins → the measured FRF is **biased** → the (correct)
ML weighting `1/σ_H²` then faithfully fits biased data → Q ran to 278. The tell from this
session: *noise-free* Hann fitting still diverged — with zero measurement noise, leakage is
the only distortion left. Every workaround since (the SNR-weighting change, the −3 dB
bandwidth estimator, its single-mode limitation) was fighting that artifact instead of
removing it. The "measure longer" result is *not* flailing — that is the Cramér–Rao bound,
which P&S achieves but cannot beat.

**Intended outcome:** implement the real P&S measurement + estimator so the FRF is unbiased
and the fit is asymptotically efficient and naturally multi-mode — ending the workarounds.
Periodic multisines are also how real LIGO CDS injections are done, so this aligns the
package with real-interferometer practice (the project goal).

**What this re-frames (not discards) from prior work:** the optimal-excitation *design*
(`prior_robust_excitation`, the dispersion iteration) is kept verbatim — only its time-domain
realization changes. The `resonator_from_tf` magnitude-gain fix is a genuine bug fix, kept.
The −3 dB `resonator_from_spectrum` and the MAP `bayesian_update` become *fallbacks* (robust
low-SNR / single-mode options), not the primary path.

## Approach (phased; Phase 1 is the load-bearing fix and gates Phase 2)

Key enabling fact: in `loop.run()` the working grid is `f_all = rfftfreq(nperseg, 1/fs); freq
= f_all[band]` — already the synchronous bin grid of one length-`nperseg` period, and
`total_dur = segment_duration * n_segments` is exactly `n_segments` periods. So a multisine
periodic over `nperseg` excites exactly `freq`, and reshaping the response into `n_segments`
periods gives a leakage-free, integer-period DFT. No grid re-plumbing.

Both new behaviors are **opt-in** (`measurement.mode`, `strategy.estimator`) with current
defaults preserved, so the existing 95-test suite stays green.

### Phase 1 — Periodic multisine excitation + leakage-free FRF (removes the bias)

1. **`src/system_ident/excitation.py` — new `multisine_from_psd(Pxx, fs, nperseg, n_periods, freq, seed=None, t_ramp=0.0)`**
   - Tone bins `k = round(freq*nperseg/fs)`; amplitude `A_k = sqrt(2*Pxx[k]*df)`, `df=fs/nperseg`
     (so total variance `Σ A_k²/2 ≈ trapezoid(Pxx,freq) = px_total` → same drive-power budget /
     watchdog behavior as `timeseries_from_asd`). Random phases `φ_k ~ U(0,2π)`.
   - One period via `irfft`: spectrum `S[k] = (A_k*nperseg/2)*exp(1j*φ_k)`, else 0; `period =
     irfft(S, n=nperseg)`; `out = tile(period, n_periods)`. Skip DC/Nyquist.
   - Ramp on only the **leading** `t_ramp` s (half-cosine, reuse the taper in
     `backends/twin.py` `ramp_down`); the start transient + ramp land in the dropped period.
2. **`src/system_ident/loop.py` — new `@staticmethod _estimate_tf_periodic(x, y, fs, nperseg, band, n_transient=1)`**, returning the same `(H, H_err, coh)` contract:
   - `xr, yr = x[:P*nperseg].reshape(P,nperseg), y[...]`; `X,Y = rfft(...,axis=1)[n_transient:]`
     (rectangular, integer periods → no leakage). `Hp = Y/X`; `Hbar = Hp.mean(0)`.
   - **Nonparametric noise (P&S):** `var = Σ|Hp-Hbar|²/(Peff-1)`; `H_err = sqrt(var/Peff)`.
   - `excited = |X|.min(0) > tol`; set `H_err = inf` off-excited bins (existing `_accumulate`
     maps that to zero weight). **Floor** `H_err = max(H_err, 1e-9*|Hbar|)` so a noise-free
     twin (var=0) is not silently dropped by `_accumulate`'s `1/H_err²` weighting.
   - `H_err` is the absolute std-error of the complex FRF — same units the existing
     `InvfreqsEstimator.fit` (`wt=1/(H_err√df)·|1/A|`) and `bayesian_update` (`rel_err=H_err/|H|`)
     already consume, so **estimators are unchanged**.
3. **Loop integration (minimal):** read `self._meas_mode = m.get("mode","welch")` and
   `self._n_transient = int(m.get("n_transient",1))` in `run()`; add two dispatchers
   `_make_drive(...)` and `_estimate(...)` that branch on `_meas_mode`. Replace the 4
   `timeseries_from_asd(...)` call sites (`_measure_dof`, `_measure_dof_bayesian`,
   `_measure_dof_spectrum`, `_inject_all` — add `nperseg` param to the last) and the 3
   `_estimate_tf(...)` call sites with the dispatchers. `Pyy` quiet PSD keeps using `_welch`
   (broadband noise has no leakage problem). In periodic mode pass `T_eff =
   (n_seg-n_transient)*T_perseg` to `fisher_matrix` for an honest reported uncertainty.
4. **Phase 1 validation (GATE):** new `tests/test_periodic_measurement.py` mirroring
   `test_map_weighting.py` / `test_step11_hybrid.py`:
   - realizer: one period repeats; `rfft(period)` support only at excited bins; synchronous
     periodogram `A_k²/(2df)` matches `Pxx`; `var(drive) ≈ px_total`.
   - leakage-free FRF noise-free: `H` matches `plant.eval(freq)` at excited bins to ~1e-6,
     where windowed-Welch on the same record shows the near-peak bias.
   - **headline:** noise-free + existing `InvfreqsEstimator` fit → **Q ≈ 20** (single mode)
     and **both modes** recovered on `plant.double_pendulum()` — where windowed Welch gave
     Q≈278 / single-mode-only. This empirically confirms the leakage diagnosis end-to-end
     *before* building Phase 2.
   - defaults unchanged: a run with no `mode` key is byte-identical to today.

### Phase 2 — P&S maximum-likelihood estimator (efficiency + clean multi-mode)

1. **Refactor** the GN information/gradient builder out of `bayesian_update`
   (`estimators/bayesian.py:198-206`) into `gn_normal_equations(J, r, w)`.
2. **New `ml_fit(freq, H_meas, H_err, model, ...)`** (in `bayesian.py` or `estimators/_ml_core.py`):
   full **Gauss-Newton + Levenberg-Marquardt to convergence** on `C(θ)=Σ|H_meas-G|²/H_err²`
   (= `bayesian_update`'s inner loop with `Lambda=0`, iterated rather than one capped step),
   using `model.jacobian/eval/with_params`. Returns `(model_hat, cov=inv(H_GN))` (the CRLB).
3. **Fill `src/system_ident/estimators/gml.py` `GMLEstimator.fit(freq,H,H_err,model)`** (drop-in,
   honoring the `Estimator` protocol in `estimators/base.py`):
   - `ResonatorModel` (f0/Q/gain): direct GN/LM via `ml_fit` — multi-mode automatic (order set
     by `model0`), well-conditioned, `dH/df0` relocates peaks.
   - `TFModel` (num/den): **Sanathanan–Koerner** iteration — re-call `invfreqs` with
     `wt = √w/|A^(m-1)(jω)|` updating `A` each iterate (not frozen at the prior, which is the
     invfreqs bias), then a few GN/LM polish steps on the exact objective; guard against
     RHP-pole drift with the `resonator_from_tf` magnitude pattern.
   - Register in `config.py`: `ESTIMATORS = {"invfreqs":..., "gml":GMLEstimator, "ml":GMLEstimator}`.
4. **CRLB reporting:** at convergence `H_GN == Fisher`, so the loop's existing `info +=
   fisher_matrix(...)`; `frac=_frac_uncertainty(model, inv(info))` path works unchanged; for a
   `ResonatorModel` refine use `resonator_design.fisher_information`.
5. **Phase 2 validation:** new `tests/test_step12_ml_estimator.py`:
   - clean-data recovery, single AND two-mode (`plant.double_pendulum()`), via both
     `ResonatorModel` and `TFModel` priors (exercises the SK path).
   - **Monte-Carlo (headline):** ~300 noisy realizations (`H_meas=H+(randn+1j randn)·H_err/√2`);
     assert (a) `mean(θ̂)≈θ_true` (unbiased) and (b) `std(θ̂_i) ≈ √diag(CRLB)_i` within ~20%
     (**achieves Cramér–Rao**), single and two-mode.
   - end-to-end loop with `estimator: gml` in `broadband_ls` on single + two-mode twins.

## Critical files
- `src/system_ident/excitation.py` — `multisine_from_psd` (new)
- `src/system_ident/loop.py` — `_estimate_tf_periodic`, `_make_drive`/`_estimate` dispatch, `run()` flags
- `src/system_ident/estimators/gml.py` — implement `GMLEstimator`
- `src/system_ident/estimators/bayesian.py` — extract `gn_normal_equations`, add `ml_fit`
- `src/system_ident/estimators/invfreqs.py` — reused (SK linear solve)
- `src/system_ident/config.py` — register `gml`/`ml`
- Reused unchanged: `design/pintelon.py`, `resonator_design.py` (Pxx design), `fisher.py`,
  `model.py`/`resonator.py` (`jacobian`/`eval`/`with_params`), `backends/twin.py`
- New tests: `tests/test_periodic_measurement.py`, `tests/test_step12_ml_estimator.py`

## Verification (end-to-end)
1. `conda run -n sysid python -m pytest -q` — full suite stays green (defaults unchanged) plus
   the two new test files pass.
2. Phase 1 gate: a quick experiment/notebook showing, on the noisy `TwinBackend` (Q=20 single
   and the 2-mode plant), periodic-mode FRF + existing invfreqs → unbiased Q (single + both
   modes), vs the windowed-Welch bias. Confirms the leakage diagnosis before Phase 2.
3. Phase 2: the Monte-Carlo CRLB test (std(θ̂) ≈ √diag(inv Fisher)) demonstrates the estimator
   is unbiased and efficient — the concrete sense in which we are now "doing P&S."
4. Follow-up (not in this plan): once validated, consider making `mode="periodic"` + `estimator
   ="gml"` the defaults, and revisit whether the −3 dB / MAP fallbacks are still needed.
