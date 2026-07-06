# Design — the excitation arcade hero (+ JupyterLite deep-dive)

**Date:** 2026-07-05  **Status:** approved design, pre-implementation.
**Goal:** an interactive "video game" on the docs landing page where a visitor **drags to
redistribute drive energy** across frequency and watches **how long it takes to identify the
plant to 5%** — giving a visual, intuitive feel for optimal excitation *before* any math, and
drawing them into the tutorial. A deeper JupyterLite notebook lets the curious run the **real**
`system_ident` package. Plus a batch of lower-priority pedagogy figures from the
2026-07-03 docs audit (`notes/pedagogy-enhancement-plan-2026-07.md`).

This is **docs-only** — no change to the `system_ident` pipeline or its public API.

---

## 1. What we're building (three phases)

| Phase | Deliverable | Priority |
|---|---|---|
| **1** | The arcade hero: self-contained JS on `index.qmd`, two competing modes, drag-to-redistribute, live "time-to-5%" timer with flat/optimal foils. Python faithfulness test. | headline |
| **2** | Lower-priority pedagogy items: leakage close-up, controller-cancellation figure, CRB pull-plot, per-example "what you'll learn" headers. | fill-in |
| **3** | JupyterLite deep-dive: an editable Pyodide notebook running the real package, linked from the hero. | stretch |

Phases are independent and ship in order; Phase 1 is the value.

---

## 2. Phase 1 — the arcade hero

### 2.1 The "level" (plant)

Two resonances, known amplitudes, so the game pins four parameters
`θ = (f₀₁, Q₁, f₀₂, Q₂)`:

- Mode 1: `f₀₁ = 1.0 Hz`, `Q₁ = 20`
- Mode 2: `f₀₂ = 2.5 Hz`, `Q₂ = 15`

FRF (identical formula in JS and in the Python validator):

```
G(f) = Σ_k  1 / ( 1 − (f/f₀ₖ)² + i (f/f₀ₖ)/Qₖ )
```

The plant curve `|G(f)|` is drawn fixed (gray) as the backdrop; the two peaks are the targets.

### 2.2 The drag mechanic — conserved power

The **drive PSD** `Pxx(f)` is represented on `N_bins ≈ 80` bins across the band
`[f_lo, f_hi] = [0.3, 6] Hz`, drawn as a filled gold area under the plant. Invariants:

- **Fixed total power** `Σ Pxx·df = Px_tot` (the real Pintelon–Schoukens budget constraint).
- A small per-bin **floor** `Pxx ≥ p_floor` (every excited line keeps a little power — matches
  "power on every line so it can iterate", and keeps the Fisher non-singular).

Interaction: pointer down + drag applies a **Gaussian brush** centred at the cursor frequency
(width ≈ 0.15 decade), *adding* power there; the whole spectrum is then renormalised back to
`Px_tot` (floor-clamped). Effect: piling energy on a peak visibly **drains it from elsewhere**
— "redistribute the energy". A **Reset** returns to flat; a **Show optimal** button morphs the
drive to the computed optimum (teaching aid).

### 2.3 The engine (client-side, exact same math as the package)

Per frame (on drag), compute over the bin grid:

1. **Jacobian** `∂G/∂θ` for `θ = (f₀₁,Q₁,f₀₂,Q₂)` by central finite difference of the FRF.
2. **Fisher** `I(θ) = 2 T_ref Σ_bins (Pxx/Pyy) Re[∂G*ᵢ ∂Gⱼ] df` (readout noise `Pyy` flat white
   for v1).
3. **Covariance** `C = I⁻¹` (4×4 real solve); per-parameter fractional uncertainty
   `frac_k = √C_kk / |θ_k|`; **worst** = `max_k frac_k` (the gate).
4. **Time to 5%.** Fisher ∝ T ⇒ `frac ∝ 1/√T`, so
   `T_5% = T_ref · (worst_frac(T_ref) / 0.05)²` — the headline number, instant to recompute.

Two **foils**, computed once per level (constant):

- **Flat** drive → `T_flat` (the slow baseline to beat).
- **Optimal "par"** → run the dispersion fixed point in JS
  (`Pxx ← Pxx·ν(f)` renormalised, `ν(f)=P_tot·tr[I⁻¹ M(f)]`, ~6 iterations) → `T_opt`.

### 2.4 The readout (canvas)

- **Top strip:** `|G(f)|` (log-y) with the two labelled peaks.
- **Bottom strip:** the draggable drive spectrum (gold fill), with faint dashed overlays of the
  flat and optimal drives for reference.
- **Four σ/θ meters** (f₀₁,Q₁,f₀₂,Q₂) as bars against the 5% line; the worst is red.
- **⏱ big timer** = current `T_5%`, plus a horizontal "you vs flat vs par" scale.
- **Win state:** when `worst_frac < 5%`, flash + freeze your time + "beat par?" prompt.

Calibrate `Px_tot`, `Pyy`, `T_ref` so `T_opt ≈ 30–60 s` and `T_flat ≈ 1000 s+` (pleasant human
range, and the ~30× gap is the whole thesis made visceral).

### 2.5 Faithfulness to the package (the honesty requirement)

The game's math must be the package's math, not a lookalike. Validation chain
(`tests/test_arcade_reference.py`, run via `conda run -n sysid`):

1. **Plant identity.** The toy FRF equals `TFModel.from_resonances([(1.0,20),(2.5,15)])` sampled
   across the band to < 1e-6 relative (proves the game shows the real plant). If the package's
   normalisation differs, the JS adopts the package's convention.
2. **Reference math.** A Python mirror of the engine (same finite-difference Fisher in
   `(f₀,Q)` coords — the exact routine already used in `fisher.qmd`'s info-ellipse) reproduces
   `system_ident.fisher`-derived σ/θ scaling for the flat and optimal drives.
3. **Golden numbers.** `T_flat`, `T_opt`, and `T_5%` for one hand-shaped drive are computed in
   Python and checked into the test; the JS is calibrated to match them within 5%, and the
   on-page "optimal par" label is those numbers. If `node` is available the test also runs the
   JS core and diffs against the Python golden JSON; otherwise the Python reference guards the
   math and the JS is kept in structural sync with it.

### 2.6 Integration

- The game **replaces** the current passive "optimal drive forming" Plotly animation on
  `index.qmd` (lines ~32–75) — it shows the same idea, interactively. The stat-row pill's `ratio`
  becomes the game's flat/par gap.
- Implementation: a raw-HTML `{=html}` block hosting a `<canvas>` + controls, plus
  `<script src="assets/excitation-arcade.js">` (vanilla ES, no deps; the docs site is a plain
  static site with no strict CSP, so inline/local JS is fine). Game styling added to
  `docs/custom.scss`. Asset auto-copied by Quarto because it is referenced.
- A one-line **"▶ now run the real thing"** link points to the Phase-3 notebook.

### 2.7 Files (Phase 1)

- `docs/assets/excitation-arcade.js` — engine (FRF, Jacobian, Fisher, dispersion) + canvas
  rendering + drag handling. Vanilla, framed as small pure functions (`frf`, `fisher`,
  `timeToTarget`, `optimalDrive`) so they are unit-testable.
- `docs/custom.scss` — `.arcade-*` styles (theme-aware, light/dark).
- `docs/index.qmd` — swap the passive animation block for the game HTML + script include; keep
  the surrounding hero and stat row.
- `tests/test_arcade_reference.py` — the §2.5 faithfulness tests + golden numbers.

---

## 3. Phase 2 — lower-priority pedagogy items

From the audit's figures/scaffolding plan. Prefer reuse of existing `docs/method_demo.py`
helpers; if the exact helper named in the audit differs, build a minimal equivalent in the same
house style (`sysid_plots`).

- **Leakage-bias close-up** on `overview.qmd` §"Why a periodic multisine?" — a Bode close-up of
  a resonance measured with vs without leakage (reuse `method_demo.leakage_bode_fig` if present).
- **Controller-cancellation figure** on `closing-the-loop.qmd` §"Identifying in closed loop"
  (currently ASCII only) — reference FRF returning the open-loop plant while the loop is closed
  (reuse `method_demo.closed_loop_fig` if present).
- **CRB pull-plot** on `closing-the-loop.qmd` (or example 01) — recovered θ over N seeds vs the
  CRB σ; the pull histogram ≈ 𝒩(0,1), showing the ML fit *attains* the CRB.
- **"What you'll learn" header** (1–2 bullets) atop each `examples/0*.qmd`.

Each figure is interactive Plotly in-page (no SVG/LFS), numbers computed, per the established
pattern. Deferred (not in scope now): the α-vs-concentration tradeoff figure and the
mode-finder stabilization diagram.

---

## 4. Phase 3 — JupyterLite deep-dive

An **editable, real-Python** companion for visitors who want to run the actual pipeline after
the arcade hook.

- **Mechanism:** the `quarto-live` (r-wasm / Pyodide) extension — inline runnable Python cells
  in a Quarto page — chosen over a full JupyterLite deployment for lighter integration. Downloads
  Pyodide + numpy/scipy on demand.
- **Package:** build a pure-Python `system_ident` wheel, host it under `docs/`, and
  `micropip.install` it in-page. **Import only the numpy/scipy core** (`system_ident.model`,
  `.fisher`, `.design.pintelon`) — not the compiled RTSfreerun/backends — so nothing needs a
  native extension under WASM.
- **Content:** rebuild the two-mode plant, set a drive, call the **real** `optimal_excitation`
  and `parameter_covariance`, compute time-to-target — mirroring the arcade in real code, editable.
- **Placement:** a new `examples/interactive.qmd` (or a section on the examples index), linked
  from the arcade hero.
- **Risk & fallback:** if the in-site Pyodide embed is unreliable (boot time, wheel install, CSP
  on the deployed host), fall back to a committed `.ipynb` plus an "open in Binder/Colab" badge —
  same notebook, hosted execution. This risk is why Phase 3 is last and independently shippable.

---

## 5. Testing & verification

- **Phase 1:** `tests/test_arcade_reference.py` (§2.5). Manual: render `index.qmd`, drag, confirm
  the timer moves the right way (concentrate on a peak → time drops; starve a mode → worst meter
  and timer stall), and that "Show optimal" reaches ~par.
- **Phase 2:** each page renders under `quarto render`; figure numbers spot-checked; cross-links
  resolve.
- **Phase 3:** the notebook boots and runs end-to-end in a clean browser profile; fallback path
  verified if the embed is dropped.
- Full-site `quarto render` clean before each phase's commit. SVG/LFS rule: the game is canvas JS
  (no plot files); all figures interactive Plotly — no committed image assets, so no LFS action.

---

## 6. Non-goals / YAGNI

- No third mode, no 6-DoF, no closed-loop in the arcade (two competing modes is the depth).
- No leaderboard/scoring persistence — just "you vs flat vs par" this session.
- No editing of the real package from the browser beyond the notebook cells.
- No new pipeline code, no API changes, no hardware anything (Phase-1 twin doctrine holds).

---

## 7. Open risks (tracked, not blocking)

1. **Optimal-drive fixed point in JS** must converge on the two-mode plant as it does in the
   package; if it oscillates, cap iterations and damp the update (mirror `optimal_excitation`).
2. **Finite-difference conditioning** of the 4×4 Fisher near the true params — use the same
   relative step (`1e-6·θ`) validated in `fisher.qmd`; regularise the solve if `cond` is high.
3. **quarto-live availability / Pyodide boot** (Phase 3) — mitigated by the Binder fallback.
4. **method_demo helper names** (Phase 2) — verify at implementation; build minimal if absent.
