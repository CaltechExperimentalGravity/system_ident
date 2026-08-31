# Project notes — system_ident (Pintelon–Schoukens migration)

Working notes for the agent-driven work on this repo. Committed, but not part of the
package or the published docs. Scratch scripts and rendered mockups also live in
`.claude/` but stay gitignored — see `.gitignore` for exactly what is and isn't carried.

## What this work did
Converted the codebase to a **single Pintelon–Schoukens (P&S) pipeline** and
removed everything else.

- **Measurement**: periodic Schroeder multisine → leakage-free, reference-based
  synchronous-DFT FRF with per-bin noise covariance
  (`SysIDLoop._estimate_tf_periodic`). The Welch FRF path is gone.
- **Estimator**: maximum likelihood only (`gml` / `ml` = `GMLEstimator` → `ml_fit`).
  The invfreqs/bayesian/hybrid/spectrum estimators + loop modes were removed
  (the bare `invfreqs()` survives only as the SK initializer inside GMLEstimator).
- **Twin**: genuine 2×2 MIMO cross-coupling (`coupling=`), plus closed-loop
  controllers, transport delay, and actuator saturation.
- **Examples 01–06**: all migrated to the P&S path (periodic + ML), each with a
  measurement time series ("coaxing, not slamming"). 06 is a new 2×2 MIMO example.
- **Docs/CI**: README architecture mermaid + codemap; lightweight Pages-only CI
  (build Quarto → deploy on push to `main`); tutorial + landing pages scrubbed
  of the removed concepts and elevated to a scholarly P&S exposition.

Suite at last run: **89 passed, 1 skipped**.

## Artifacts in this directory (local only — gitignored, not in the clone)
- `dashboard-mockup.pdf` / `make_dashboard_mockup.py` — live-dashboard design mockup.
- `infographic.pdf` / `make_infographic.py` — 4-page method explainer (moved here
  once its background agent finishes rendering).

## Docs standard plot set (2026-06-14)
Every example now shows the **same full diagnostic set** (no more 2–3 plots).
Two doc-only helper modules drive it (presentation only — NOT package API):
- `docs/sysid_plots.py` — house style (`style()`: big fonts SZ_AXIS=19/TICK=15/
  LEGEND=16, markers MK_DATA=7/MK_BIG=13) + named panel builders: excitation_design,
  timeseries, bode (mag+phase+coherence), coherence, residuals, parameter_recovery,
  saturation, convergence, pass_overlay, param_table.
- `docs/sysid_campaign.py` — `run_siso_passes` (transparent multi-pass P&S refinement
  reusing the library internals) + `param_sigmas` (CRB covariance → σ on f0/Q/gain/fc).
Per example: 01/02/03/05 = 8 figures (all 10 panels; bode bundles 3), 04 = 26 (3-DoF
tabs), 06 = 23 (2×2 matrix + per-element tabs). Tutorials model/fisher restyled;
fisher gained a dispersion-ν plot. `_quarto.yml`: `code-fold: true` (collapsed code),
`code-tools`, `execute-dir: project` (so `import sysid_plots` resolves from docs/).
Local render proxy (quarto not installed): `.claude/run_qmd.py` executes a page's
python cells with `fig.show` stubbed. Suite still 91 passed / 1 skipped.

Docs "wow"-feature research saved to `.claude/docs-wow-features.md` (animated Plotly
frames, closeread scrollytelling, listing gallery+lightbox, ojs sliders, social cards).

## Local Quarto + "fabulous" docs pass (2026-06-14)
- **Run everything via `conda run -n sysid ...`** (never another env, never
  hand-built binary paths + env vars). Quarto lives in the sysid env; it was
  missing `deno` until `conda run -n sysid conda install -c conda-forge quarto
  --force-reinstall -y` restored it. Render locally:
  `cd docs && conda run -n sysid quarto render [page.qmd]`.
- Installed (user-approved): `kaleido` (Plotly static PNG export) + the
  `quarto-ext/lightbox` extension (vendored in `docs/_extensions/`).
- `docs/make_assets.py` → `assets/og-card.png` + `examples/thumbnails/0{1..6}.png`
  (kaleido). Regenerate if example signatures change.
- Fabulous features now live: Examples **card-gallery** (`examples/index.qmd`),
  **social cards** (site-url/open-graph/twitter-card + og-card), themed **Mermaid**
  loop diagram (overview), **animated Plotly** (landing hero = drive forming;
  ex-01 = loop converging via `sysid_plots.animate_passes`/`animate_design`),
  **ojs sliders** (ex-01 capstone, drag f0/Q), code annotations, margin asides,
  reading-time, prev/next, back-to-top, lightbox. Full `quarto render` = 34/34 green.
- RULE: when a package/tool is missing → **ask the user + pause**; never substitute
  another lib or hunt across envs (memory: ask-for-missing-tools-and-pause).

## Closed-loop = first-class (2026-06-14, roadmap item)
Closed-loop (controller-aware) sysID is now **declarable in config**, not ad-hoc.
The measurement math was already done (reference-based FRF cancels C); this added
the wiring + both injection points.
- **Twin** (`backends/twin.py`): `injection_point` ("after_controller" default |
  "before_controller", scalar or per-DoF); `drive_channels`/`error_channels` (the
  after-C drive `u` = FRF input X, before-C error `e` = diagnostic). Closed-loop
  generalised to 6 shared-denominator filters (y_r/y_w/y_n, u_r/u_w/u_n);
  after-controller numerators are byte-identical to before. Before-controller
  algebra: `u = C(EXC−y)` → y_r=GnCn/D, u_r=GdCn/D (so Y_r/U_r = G).
- **Config** (`config.py`): `_parse_controllers`; `build_twin_backend` forwards
  `twin.controllers` (continuous C as {num,den}) + `twin.injection_point`.
  `TwinBackend.from_config` builds drive/error maps from `channels.drive`/`error`.
- **Loop** (`loop.py`): FRF input X read from `channels.drive` (falls back to the
  excitation channel — twin exc read already returns u). Soft `warnings.warn` if
  before-controller injection + no drive channel (bias risk on real hardware).
- `configs/closed_loop_demo.yml` (CLI-runnable); `tests/test_closed_loop_config.py`
  (5 tests, both injection points recover the OPEN-loop plant). Example 05 gained a
  config-driven section; tutorial closing-the-loop gained the digital 4-point
  topology + config schema. Suite: 102 passed / 1 skipped.

Remaining roadmap (see `.llm/roadmap.md`): CDS backend (step 8) [blocked: no
nds2/awg/cdsutils libs], snapshot/restore for CDS (step 5), MIMO joint/simultaneous
ID, versioned docs. NOTE: Foton export is **explicitly out of scope** — the user
rejected it repeatedly and reconfirmed on 2026-08-11 ("left for someone else to
do"); do not re-add it. Reading foton banks is fine. Memory: `no-foton-export`.

## Open / deferred
- ✅ **Prior-robust first pass — DONE.** The loop's first pass now designs from the
  prior + its error bars (`strategy.prior_uncertainty`, default 0.5) via
  `design.pintelon.prior_robust_excitation` (frequency-scales the model over
  `[1−u, 1+u]`, averages the optimal design); later passes are point-optimal.
  Verified end-to-end: cracks a +50% prior (`tests/test_prior_robust.py`).
- CDS hardware backend (`backends/cds.py`) remains a `NotImplementedError` stub.
- GitHub Pages deploys on push to `main` via the lightweight CI; the first deploy
  executes the example `.qmd` files (jupyter kernel registered in the workflow).

## Env setup for the RTSfreerun twin path (2026-06-18, on the twin box)
**Decision:** `system_ident` and `digital_twin`/`rtsfreerun` stay **separate
repos**. Do NOT install `system_ident` into the `twin` env. Instead, build the
compiled RTS model(s) **into this repo's `sysid` env** so `system_ident` and the
model import from one interpreter. (rtsfreerun = one model per process.)

- Created `sysid` env: `conda env create -f environment.yml` (py3.12, numpy 2.4.6,
  scipy 1.17.1, control 0.10.2; `system_ident` editable). Suite 109 passed/2 skipped.
- Built `x1hsts` into `sysid` (recipe now in committed **README.md** §"Run against
  the RTSfreerun digital twin"): one-time `conda install -c conda-forge make cmake
  spdlog rapidjson pybind11`, then from `digital_twin/rtsfreerun`:
  `PATH=$CONDA_PREFIX/bin:$PATH RCG_LIB_PATH=…/rtsfreerun-models/x1hsts model=x1hsts pip install .`
  — **`RCG_LIB_PATH` must point at the model source dir** or the build fails with
  "Couldn't find model file x1hsts.mdl".
- Result: `import system_ident, x1hsts` works in `sysid`; `x1hsts().sample_rate`=16384;
  `tests/test_rtsfreerun_backend.py` now 8 passed / 0 skipped (real-model case live).
- Note: the twin box (macOS/arm64) **does** have `twin` + `rtsfreerun-dev` conda envs —
  the old `.llm` note "no twin env on this macOS machine" was wrong. Deleted the
  transient `.llm/START-HERE.md` (per user; notes live here + in `.llm/roadmap.md`).

## HSTS demo — Track A1+A2 DONE (2026-06-18, twin box)
Identified the compiled `x1hsts` drive→sensor plant with the **P&S optimal-excitation
campaign** (`run_siso_passes`: broad prior-robust pass 1 from the perturbed prior →
point-optimal refinement → leakage-free reference FRF → ML fit) under the twin's
seismic+bosem noise. **NOT flat** — the same campaign the double-pendulum example runs.
- **Plant reality:** drive→sensor = order-10 `HSTS_DRV_TF_A` cascade (FM1–5; `_B`=id):
  5 modes @ 0.67/1.01/1.52/2.81/3.78 Hz, all Q≈50, interleaved near-cancelling zeros.
  Old skeleton prior (0.9 Hz/Q10) was wrong → fixed in `configs/rtsfreerun_hsts.yml`.
- **Oracle** (`backends/rtsfreerun_oracle.py`): analytic plant from scenario ZPK
  (plane-`f`→`s` via −2π, mirrors orchestrator `_coerce_zpk`) + `apply_scenario_init`
  (bare model has no filters!) + realized-SOS cross-check. yaml-oracle vs realized-SOS
  agree to 1.6e-6.
- **Results:** A1 noise-off P&S pass recovers oracle to ~0.2% median, exact len, ×64
  decim, oracle⇄SOS 1.6e-6. A2 noise-on: **frac-uncertainty 0.228→0.046→0.034 over 3
  passes** (genuine refinement), recovered vs oracle ~0.2% median, all 5 modes <0.2% f0.
- **Enabler — `fisher.safe_inverse`:** the order-10 near-cancelling plant makes the
  Fisher **rank-deficient**, so `dispersion()` / `inv(info)` were singular. `safe_inverse`
  falls back to pinv ONLY when singular (full-rank still `inv` exactly → sysIDlib
  bit-for-bit tests untouched). This is what lets P&S **optimal** excitation run on
  realistic high-order plants. Used in `fisher.dispersion`, `loop`, `sysid_campaign`.
- **`run_siso_passes` gained `x_ch=`** (FRF input = drive monitor `COIL_DRIVER_OUT`,
  distinct from injection `COIL_DRIVER_EXC`); double-pend call unchanged. Clear filter
  history per rung (`mdl.fm_clear_history`) — one model/process, carryover biases.
- **Artifacts:** `experiments/rtsfreerun/run_hsts.py` (+ `hsts_recovery.png` w/ optimal
  excitation ASD panel), `tests/test_rtsfreerun_real_model.py` (guarded A1/A2 via the
  P&S campaign), `tests/test_rtsfreerun_oracle.py`. Adapter gained `scenario=`;
  `from_config` reads `rtsfreerun.scenario`. Suite 121 passed.

## A3/A4 RE-SCOPE: real closed loops on x1hsts6dof (2026-06-18)
A3 was specced on single-DOF `x1hstsdamped`, but that model's damping bank
(`MC1_M1_DAMP_L`) ships unconfigured and `hsts_damped.yaml` `init:` never sets it
(loop stays open → A3 would equal A2). **Superseded.** Per user, A3+A4 run on the
already-built **`x1hsts6dof`** with the REAL H1/L1 damping loops, exactly as the
twin's canonical example drives it:
- Reference: `digital_twin/twin/examples/sus_hsts_6dof/{lib.py,run_rtsfree.py}`
  (NOT the older May-9 `experiments/sus/hsts_6dof_rtsfree_demo.py`, which uses stale
  MC1 names + only 4 active DOFs).
- **Plant is NOT baked in** — `x1hsts6dof` is a `cdsStatespace` block `HSTS_PLANT`
  whose (A,B,C,D) are set at runtime: `lib.load_plant_continuous()` (36-state bare-M1
  HSTS from `aligo-suspension-models/hsts_full.mat`) → ZOH discretise → `mdl.ss_set_abcd`.
  **The oracle is this discrete SS plant** `G[out,in](f)=[Cd(zI-Ad)^-1 Bd+Dd]`, z=e^{j2πf/fs}
  — a DIFFERENT oracle source than A1/A2's scenario-YAML ZPK (`rtsfreerun_oracle`).
- **Dampers**: real **L1 MC2** foton banks `SUS-MC2_M1_DAMP_<dof>` from
  `aligo_filter_files/l1/L1SUSMC2.txt`; engaged FM list from the L1 SDF
  (`sdf_filter_state_<GPS>.json`, GPS in `archive_GPS`); per-DOF CAL on **free slot FM1**
  (`SITE_BANK_CAL`: L .30/T .28/V 1.09/R .184/P .54/Y .735, tuned for τ≈5s ringdown).
  `apply_foton_bank(mdl, SITE_FOTON, src_bank, dest_bank)` with src=dest=`MC2_M1_DAMP_<d>`.
- **CHANNEL DRIFT** (spec §9 risk realized): built `.mdl` (May 17) renamed dampers
  `MC1→MC2`; ports are `DRIVE_EXC_<d>` (inject) / `READOUT_<d>` (sensor). Plant-input
  node `SUM_<d>` is NOT a readable channel; the damper out **`MC2_M1_DAMP_<d>_OUT`** IS.
- **Reference-based X (controller cancellation):** true plant input = injected drive +
  damper feedback, reconstruct `X = u + MC2_M1_DAMP_<d>_OUT` (no single plant-in channel).
  With loops CLOSED, `READOUT_<d>/X` recovers the OPEN-LOOP plant element.
- **Deps** (all in the sibling `digital_twin/` checkout, machine-specific → guard+skip):
  `twin/src` on path (`twin.foton_loader.apply_foton_bank`, `twin.plant_loader`),
  `aligo-suspension-models` (readFilter + hsts_full.mat), `aligo_filter_files/l1`.
  `twin` is NOT installed in sysid; add `twin/src` + the example dir to `sys.path`.
- **Proven in sysid**: `.claude/probe_6dof_damping.py` (loops close+damp, all 6 DOFs);
  `.claude/probe_a3_core.py` (closed-loop reference FRF recovers open-loop plant at the
  median; p90 tail = MIMO coupling = the A4 story).

### A3 closed-loop finding (2026-06-18, decision pending)
A4 (open-loop MIMO tensor) recovers cleanly (diag <0.1%, L<->P & R<->Y ~0.1-0.2%).
A3 (closed loop) is the hard one. The real L1-MC2 top-mass dampers are GENTLE
(tuned tau~5s, |Sens|peak 1.5-2), so the loop suppresses the NET plant input
SUM=DRIVE+DAMP_OUT at the resonances -> reference FRF Y/SUM is ill-conditioned
exactly at the damped modes. Flat broadband: median ~5% but resonance PEAKS 50-110%
off (median hides it; open loop nails peaks <0.1%). NOT just coupling (persists with
only the driven loop closed) -> it's loop suppression of X at resonance. Principled
fix = A2-style OPTIMAL excitation (power at the damped modes) + a CORRECT modal prior.
My quick SS->modal prior was buggy (np.poly of in-band eig(Ad) poles kept only 2 of
~5 modes -> under-ordered -> campaign couldn't recover missing modes). Need proper
SS-diagonal modal reduction for the prior. All-6-closed adds coupling bias (may need
Track B MIMO joint fit for clean peaks). Probes: .claude/probe_{helper,cancel,peak,
oneloop,campaign}.py. Backend `plant_inputs=` virtual-channel feature already landed.
A3 PATH OPTIONS for the user:
 (a) build the optimal-excitation closed-loop campaign + correct SS modal prior (clean
     single-DOF A3, loops closed) -- most faithful to spec A3.
 (b) ship A4 now, reframe A3 as "closed-loop SISO ill-conditioned at damped modes ->
     clean recovery is the MIMO joint fit (Track B)"; make Track B the next deliverable.
 (c) demonstrate cancellation only where clean (off-resonance / stronger drive), caveated.

### A3 RESOLVED — it was a sign bug (2026-06-18)
The "closed-loop peaks 50-110% off" wall was a SIGN ERROR in the plant-input
reconstruction, NOT a method/excitation/coupling/Track-B problem. The composite's
plant-input junction COIL_DRV_SUM_<d> is a "+-" sum: SUM = DRIVE_EXC - delay(MC2_M1_
DAMP_<d>_OUT) (gen_x1hsts6dof.py: Sum inputs="+-", UnitDelay LOOP_DELAY_<d>). I had
X = drive + damp_out. Wrong sign is negligible off-resonance (small feedback) but
dominates at the damped modes (large feedback) -> looked exactly like an
ill-conditioning wall. Fix: X = drive - damp_out. Now CLOSED-loop diagonal recovers
the open-loop plant to <0.1% for ALL six DOFs (incl pitch); parametric optimal-exc
campaign: L median 0.15%/peak 1.6%, all 5 modes, frac falls 0.045->0.012.
Backend `plant_inputs` gained `feedback_coeff` (-1 here). The plant-input node is NOT
an exposed channel (COIL_DRV_SUM_<d> unfetchable) so reconstruction is required; the
1-cycle LOOP_DELAY is negligible at 0.3-8 Hz. A4 off-diagonal closed-loop coupling is
the real MIMO/Track-B story; the open-loop per-pair tensor is clean.
LESSON: when "every method fails identically at the resonances," suspect a shared
reconstruction/sign bug before concluding the METHOD (P&S) is inadequate.

### Showcase page DONE (2026-06-18)
docs/examples/07-rtsfreerun-twin.qmd (freeze:true) + docs/rtsfreerun_demo.py glue
+ committed docs/_freeze/. Sections A1-A2 (x1hsts SISO under noise), A3 (x1hsts6dof
closed-loop diagonal, real L1-MC2 loops), A4 (6x6 MIMO tensor). Rendered clean (3
plotly figs + 2 tables, 0 errors), appears in examples listing. Commits 80d2767
(A3+A4 code/gate) + abdc044 (page). REMAINING: thumbnail 07.png blocked on kaleido
(NOT installed in sysid — ask before installing, per ask-for-missing-tools rule);
make_assets.thumb_07 is ready, just needs `pip install kaleido` then regen. Also: the
A3 parametric per-DOF priors hit an SVD non-convergence for pitch/yaw in my campaign
probe (generic seed) — the nonparametric gate doesn't depend on it; revisit if the
page ever shows the parametric closed-loop campaign per DOF.

### Parametric A3 priors fixed — pitch/yaw no longer crash (2026-06-18)
Two root causes, both fixed:
1. WRONG MODEL ORDER per DoF. The 6 HSTS diagonals carry different #modes (L=5, T=3,
   V=2, R=4, P=5, Y=3). Forcing 5 modes over-parameterised V/Y -> spurious out-of-band
   poles (V: 44Hz/1.3kHz) or a near-degenerate doublet (Y: 1.08/1.09) -> near-cancelling
   pole/zero pairs. Fix: HSTS6DOF.oracle_prior() grows the order one mode at a time and
   stops before a mode that is out-of-band, near-degenerate, or stops improving the fit.
2. EXC BIN UNDERFLOW. optimal_excitation concentrates drive; off-resonance bins underflow
   to EXACTLY 0; dispersion() divides dens*Pxx_tot/Pxx[i] -> 0/0 NaN -> poisons next
   Fisher -> safe_inverse pinv 'SVD did not converge'. The division is analytically a
   no-op (dens ∝ Pxx[i]) but breaks numerically at 0. Fix (user's call): floor every exc
   bin at _EXC_FLOOR_FRAC=1e-8 of peak in design/pintelon.optimal_excitation. Diverges
   from legacy only on negligible bins (significant bins match <1e-6); test_step3
   updated to compare significant bins + assert the floor; added a concentrated-design
   regression test. All 6 DoFs now recover parametrically (median <0.004), gated by
   test_rtsfreerun_6dof::test_a3_parametric_campaign_recovers_all_dofs. Helper:
   HSTS6DOF.parametric_recovery(dof). Suite 132 passed.

### Docs audit + fixes (2026-08-25..30) — PARKED, resume here

Full audit of the 25 hand-written docs pages, then fixes. Four commits, all on main
and pushed: c147eb1, fcbf0df, 601f715 (and 1676038 from the prior register sweep).

SHIPPED
- Broken refs: `configs/*.yml` named as runnable paths on two pages; they live under
  `src/system_ident/configs/`. Verified the bad path raises ConfigError.
- Example 13 was pip-runnable but missing from all three Binder lists while the
  gallery prose said only 07/10 are excluded. Added to binder/postBuild,
  gallery-binder.js RUNNABLE, and the index prose.
- Figure captions: 89 cells labelled -> 116 numbered figures on the site, was 6.
  Tabset pages needed LIST-form `fig-cap` (show_dof/show_elem emit 8 and 3 figures
  per call, so one caption cannot address them). Ex 04: 0 -> 26, ex 06 -> 21.
  `image-alt` on all 13 examples: gallery went 13/74 imgs without alt -> 0/74.
  NOTE: `fig-alt` is deliberately NOT set — verified it emits no a11y attribute on a
  Plotly cell, only an echo in the visible source listing. The caption is the
  accessible name (Quarto wraps it in figure/figcaption).
- Register: 240 bold spans dropped, 290 -> 68, survivors all structural. 19 of them
  wrapped ACROSS A LINE BREAK and are invisible to a single-line regex — remember
  this when measuring emphasis. Gallery descriptions 178-315 chars -> 83-162.
- Guards: tests/test_docs_integrity.py (pure stdlib, runs in the light CI test job) —
  prose paths exist, links/anchors resolve, freeze:true pages match their source, and
  the known-stale quarantine cannot outlive the problem. Each proven by deliberate
  breakage.
- Freeze policy: `_quarto-ci.yml` used to claim "any source change re-executes
  everything" — FALSE for freeze:true pages (front matter beats the profile, and a
  committed _freeze entry is what grants the immunity). Un-pinned the three that did
  not need it: 08 (55 s), 09 (67 s), why-optimal-excitation (61 s), and deleted their
  committed _freeze. Still pinned: 07/10 (need the compiled twin), 13 (569 s vs 67 s
  for the next slowest). .gitignore now denies docs/_freeze and allows only those
  three by name.
- Planted-Q caveat at the first mention on each page that uses it (07, 10,
  tutorial/fisher): the real Qs are much larger, the uniform low Q is for ease of
  calculation. 11 already said "(model-set)"; 12 claims no Q recovery.

OPEN — highest first
1. CI WAS NEVER WATCHED for any of the four pushes. `gh` is not installed here and
   the stay-in-project hook blocks looking for it, so I could not follow the standing
   "watch CI after every push" rule. This matters MORE than usual: CI now executes
   08/09/why-optimal-excitation itself instead of serving them from a committed
   freeze, so a failure there is newly possible. CHECK THIS FIRST TOMORROW.
2. Examples 07 and 10 are stale and cannot be re-executed without the compiled
   x1hsts twin. Waiting in source for a render on a machine that has it: their figure
   captions, the register pass, and the new planted-Q caveat. Both are listed in
   _KNOWN_STALE in tests/test_docs_integrity.py; that list self-expires.
3. Their two mixed table-plus-figure cells are the only uncaptioned figures left —
   deliberately, since the output shape cannot be observed without the twin.
4. Should the quad's model-set Q (Q = 628.3*f, from a uniform -0.005 1/s eigenvalue
   shift on 24 of 28 modes) get the same "real Qs are larger" wording as the HSTS 50?
   Left alone pending the user's call.
5. Audit items reported but NOT fixed (user did not ask): why-optimal-excitation
   asserts "No number on this page is hand-entered" but transcribes three results
   (1.9, 2x, 1.4x — all verified correct today, so it is fragility not error);
   python 3.9 badge while CI tests only 3.12; every figure loads Plotly from a CDN
   and the two frozen pages pin an older version than the rest (3.6.0 vs 3.7.0);
   one h2->h4 heading jump in 07.
6. Unrelated leftover: the `slide-decks` skill written earlier in skill-skeleton was
   never validated, installed, or committed — blocked by the stay-in-project hook.

NO src-drift detection on the three pinned pages; the .qmd hash is all Quarto keys
on. A source-fingerprint guard was considered and REJECTED: any src/ edit would fire
it and it would get refreshed reflexively. _quarto-ci.yml states the gap and gives the
re-run command instead.
