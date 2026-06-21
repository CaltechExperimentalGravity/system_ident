# Documentation site audit — `system_ident`

**Date:** 2026-06-20  **Auditor:** docs review pass (read-only)  **Branch:** `main`
**Scope:** `docs/` (landing, tutorial, examples, reference, presentation glue), `README.md`,
and consistency against `src/system_ident/`.

> A concurrent agent is adding a new "why optimal-excitation" methods page
> (`docs/tutorial/...` + `docs/method_demo.py`). That page is **not** treated as a gap here;
> this audit covers the docs as they currently exist.

---

## Executive summary

The site is, on craft, well above the bar for a research package: the landing page has a
genuine narrative hook ("your sweep is wasting 90% of its time"), the tutorial derives the
method honestly from Fisher information, and examples 01–08 are unusually rigorous — every
SISO example runs the same nine-panel diagnostic set, and the "show me, don't trust me"
ethos (raw time series, residual histograms, oracle overlays in 07/08) is real and
distinctive. The house style is centralized (`docs/sysid_plots.py`) with data-driven
log-y ranges, large fonts, and a consistent palette; committed graphics (thumbnails,
favicon) are SVG in Git LFS as the hard rule requires. **No house-style/LFS violations
were found.**

The problems are not craft — they are **drift and onboarding**. The fastest-moving parts of
the package (the API surface, the example count, the twin demo) have outrun the prose, so a
new engineer's first 60 seconds contain several statements that are wrong or oversell a stub.
And the "land → method → first example → **your own subsystem**" arc breaks at the last step:
there is no install/quickstart page and no "point this at my suspension" guide; the run-config
YAML — the thing a user must actually write — is documented only in scattered fragments.

The single highest-value fix is to **stop the drift**: regenerate the API reference, and
correct the four concrete falsehoods on the landing page (below). They are cheap and they are
exactly what destroys trust in a precision-instrument tool.

### Top 5 highest-impact findings

1. **The landing page's headline stat is a dead placeholder.** The "Variance reduction vs
   flat sweep" pill renders literally as `—×` in the published `_site/index.html` — nothing
   populates `#stat-ratio`. (Critical #1)
2. **The API reference is stale and inconsistent with `_quarto.yml`.** It documents the
   *legacy* `excitation.timeseries_from_asd` and a *removed* `InvfreqsEstimator` class, while
   the pipeline's central synthesiser `excitation.multisine_from_psd` has no reference page.
   (Critical #2)
3. **The landing quickstart misdescribes the twin demo** as "the full POS/PIT/YAW suspension
   campaign" — `twin_demo.yml` is 2-DoF (POS/PIT), `max_iter: 2`. (Critical #3)
4. **The landing page sells the CDS backend as a working "single config line, no code
   changes"** — `CDSBackend` raises `NotImplementedError`. Contradicts the project's own
   show-me ethos and the README's honest "stub" framing. (Critical #4)
5. **The onboarding arc has no last mile:** no install/quickstart page, no run-config
   reference, no "bring your own subsystem" guide. (Important #7)

---

## Critical — fix first (wrong, or breaks trust on first contact)

### C1. The landing-page headline metric is never filled in
`docs/index.qmd` lines 80–84 render a stat pill `<div class="stat-value" id="stat-ratio">—×</div>`
under the label "Variance reduction vs flat sweep" — the single most important number on the
page. The Python block computes `ratio` (line 66) but only injects it into the *figure title*;
nothing sets `#stat-ratio`. Confirmed in the committed render: `_site/index.html` still shows
`id="stat-ratio">—×</div>`.
**Fix:** either populate the pill (emit a tiny inline `<script>` that writes the computed
`ratio`, or hard-code the representative number you already cite in the figure title), or
delete the placeholder pill. A dash where the hero promises a number is the worst possible
first impression for a measurement tool.

### C2. The committed API reference is stale and self-inconsistent
`_quarto.yml` (lines 102–104) declares the Core reference contents as `…excitation.multisine_from_psd…`,
but the committed `docs/reference/index.qmd` (line 13) and `docs/reference/_sidebar.yml`
(line 10) instead list **`excitation.timeseries_from_asd`** — the *legacy coloured-noise
drive* that the README codemap itself flags as superseded. Consequences:
- The pipeline's actual excitation synthesiser, `excitation.multisine_from_psd`
  (`src/system_ident/excitation.py:112`) — the periodic Schroeder multisine that **every**
  tutorial and example uses — has **no reference page** in the committed output.
- `docs/reference/estimators.invfreqs.InvfreqsEstimator.qmd` documents a class that **no
  longer exists** (`estimators/invfreqs.py` exports only the function `invfreqs`; README
  codemap confirms the standalone estimator was removed).
- Because `_quarto.yml` renders `reference/*.qmd` by glob (line 11), both stale files
  (`InvfreqsEstimator`, `timeseries_from_asd`) are still published as **orphan pages** even
  after a `quartodoc build` regenerates the sidebar.
**Fix:** run `quartodoc build` from `docs/`, delete the orphaned `*.qmd` (InvfreqsEstimator,
timeseries_from_asd), and confirm the published reference documents `multisine_from_psd`.
Then decide a policy: either **stop committing generated reference** (gitignore
`docs/reference/*.qmd` + `_sidebar.yml`, regenerate in CI — the cleanest), or add a
pre-commit/CI check that the committed reference matches `_quarto.yml`. Right now the repo
ships a reference that disagrees with its own config.

### C3. "Full POS/PIT/YAW campaign" — the advertised twin demo is 2-DoF
`docs/index.qmd:150–161` ("Run a full campaign in 60 seconds") tells the reader that
`system_ident run …/twin_demo.yml --twin --yes` exercises "the full **POS/PIT/YAW**
suspension campaign." But `src/system_ident/configs/twin_demo.yml` is explicitly **two DoFs**
(its own header comment: "Two DoFs against a resonant twin"; channels = POS, PIT only) and
caps at `max_iter: 2`. The genuine 3-DoF POS/PIT/YAW campaign is **example 04**, a different
config. Also the advertised output line `DONE (target reached); per-DoF fractional
uncertainty ~1e-9` is not guaranteed by a `max_iter: 2` run and should be verified against an
actual `conda run -n sysid system_ident run …` invocation, not asserted.
**Fix:** change "POS/PIT/YAW" to "POS/PIT" (or point the quickstart at the 3-DoF config and
say so), and regenerate the output line from a real run.

### C4. The CDS backend is sold as working; it is a stub
The hero feature card "Same API — twin or hardware" (`docs/index.qmd:135–141`) states
"swapping in the CDS backend is a **single config line, no code changes**." But
`src/system_ident/backends/cds.py` raises `NotImplementedError("CDSBackend lands in build
step 8")` on `inject`/`read`/etc. The README is honest about this ("**stub** raising
`NotImplementedError`", lines 93 & 217); the landing page is not. For a tool whose whole
pitch (pages 07/08) is auditability and not-trusting-assertions, presenting an unimplemented
path in the present tense is a self-inflicted credibility wound.
**Fix:** reword to aspirational/explicit — e.g. "the loop speaks a backend abstraction so a
CDS hardware backend drops in behind the same API (in progress)" — matching the README's
candor.

---

## Important — fix soon (gaps and staleness that mislead, but not on the home page)

### I5. Examples index prose undercounts and over-generalizes
`docs/examples/index.qmd:17–21` opens "**Six** end-to-end Pintelon–Schoukens campaigns,
ordered from the simplest single resonance to a fully coupled 2×2 MIMO suspension," and
claims "Every example runs the *same* full diagnostic set." There are **eight** (`01`–`08`).
Examples **07** (compiled RTSfreerun CDS twin) and **08** (DARM calibration) are the most
impressive pages in the set and are *different in kind* — they do **not** run the 9-panel SISO
diagnostic set; they have their own oracle-overlay / closed-loop tensor / swept-sine-comparison
panels. The current sentence both undercounts them and mischaracterizes them.
**Fix:** "Eight campaigns — six teaching plants (01–06) running an identical nine-panel
diagnostic set, then two real-plant capstones: the compiled-CDS suspension twin (07) and DARM
calibration (08)." This also *sells* the two best pages instead of hiding them.

### I6. Examples 07 and 08 are nearly undiscoverable in navigation
The navbar (`_quarto.yml:41–42`) exposes Examples as a single link to the gallery; 07/08 are
reachable only by scrolling. Yet the **README leads with 07** as the headline result
(closed-loop 6-DOF on the real CDS model). No tutorial page links forward to them either —
e.g. `closing-the-loop.qmd`'s closed-loop section is the natural place to say "see this run
for real on the compiled twin in [07]," and it instead only links the teaching example 05.
**Fix:** add forward cross-links from the relevant tutorial sections to 07/08, and consider a
"Highlights" or expanded Examples menu in the navbar that names the capstones.

### I7. The onboarding arc has no last mile — install, config, "your own subsystem"
The stated goal is landing → method → first example → *their own subsystem*. The final step is
unserved:
- **No install/quickstart page.** Installation exists only as hand-rolled HTML on the landing
  page and in the README; a reader who clicks "Understand the method" lands in
  `tutorial/overview.qmd` (pure theory) with no "install and run in two minutes" page in the
  Tutorial menu.
- **No run-config reference.** `RunConfig._validate` (`src/system_ident/config.py:99`) enforces
  a real schema (`channels`, `measurement`, `strategy.estimator`, `strategy.input_designer`,
  `safety`, `twin`, `priors`, `stop_criteria`), but the YAML is documented only in scattered
  fragments — `safety-and-ops.qmd` shows the `safety:` block, `closing-the-loop.qmd` shows
  `channels:`/`twin.controllers:`. There is nowhere a reader can see the **whole** config with
  every key explained. This is the single biggest blocker to "use it on my suspension."
- **No "bring your own plant / choose a prior / pick model order" guide.** The most actionable
  operational wisdom in the whole repo — "L carries five modes; pitch and yaw carry fewer, and
  over-modelling them is what used to break the optimal-excitation design"
  (`examples/07-rtsfreerun-twin.qmd:165`) — is buried in a capstone example, not in a tutorial.
**Fix:** add (a) a short **Quickstart** page (install + the twin one-liner + "what success
looks like"), top of the Tutorial menu; and (b) a **Configuration / Bring your own subsystem**
page that walks the full `twin_demo.yml` key-by-key and gives the prior/model-order guidance.

### I8. Coherence is used everywhere but never defined
Coherence γ²(f) appears in every Bode panel and is named as a dashboard panel and as
"coherence health" (`safety-and-ops.qmd:8–11`, 77), but it is **never defined or explained**
in the tutorial — what it means, how it's estimated here (per-line from period-to-period
scatter), and how to read it alongside the CRB. For the explicit "reading CRB/coherence" goal,
this is a real conceptual gap.
**Fix:** a short subsection in `fisher.qmd` or `closing-the-loop.qmd` defining coherence as it
is computed here and how a low-coherence bin should change a reader's trust in the fit.

---

## Nice-to-have (polish)

- **N9. Repetition of "the first pass is special."** It appears three times in the tutorial —
  `overview.qmd` column-margin note (lines 41–45) *and* callout-tip (174–183), plus
  `closing-the-loop.qmd`'s pass strategy. Consolidate to one canonical statement and
  cross-reference it.
- **N10. The README is the de-facto best getting-started doc, and the site lacks its
  architecture view.** The README's loop mermaid + codemap table (lines 48–98) is excellent and
  has **no equivalent on the site**. Porting it into a docs "Architecture" page would give the
  site the high-level map it currently only has in passing on the landing page.
- **N11. Unsourced stat pills.** `index.qmd:88–96` asserts "O(n²) Fisher cost", "3 fixed-point
  iters", "~1 DoF/min" with no backing. Fine as marketing, but mildly at odds with the
  show-me ethos; consider linking each to the page that demonstrates it (e.g. the convergence
  panel in 01).
- **N12. Link-style inconsistency in `model.qmd`.** Its "See also" (lines 120–122) links to
  `../reference/model.TFModel.qmd` (source path) while the reference index uses
  `…qmd#anchor`. Quarto resolves both, but pick one convention.
- **N13. Install-extra inconsistency.** The landing quickstart says `pip install -e ".[dev]"`
  (which only adds pytest) to *run* the demo; the README uses `pip install -e .` for the core
  twin path. Align them so a reader isn't told to install test deps to run a demo.

---

## Suggested doc roadmap (in priority order)

1. **Stop the drift (½ day).** Regenerate the API reference and delete the two orphan pages
   (C2); fix the four landing-page falsehoods — dead stat pill (C1), 2-DoF vs POS/PIT/YAW (C3),
   the CDS overstatement (C4); regenerate the advertised demo output line from a real
   `conda run -n sysid` invocation. These are the trust-breakers and they are cheap.
2. **Fix the examples index + surface 07/08 (½ day).** Correct the count and reframe 07/08 as
   real-plant capstones (I5); add forward cross-links from the tutorial (I6). This costs almost
   nothing and immediately promotes the strongest content.
3. **Close the onboarding last mile (1–2 days).** Add a Quickstart page and a
   Configuration / "bring your own subsystem" page documenting the full run-config YAML and the
   prior/model-order guidance currently buried in example 07 (I7). This is what turns a reader
   who *understands* the method into a user who *runs* it on their own plant.
4. **Concept polish (½ day).** Define coherence where it's used (I8); de-duplicate the
   "first pass is special" note (N9); port the README architecture map into the site (N10).

**Decisive first move:** item 1. Everything else builds on a site whose own first screen and
API reference are internally consistent.
