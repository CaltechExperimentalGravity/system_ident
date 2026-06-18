# Design: P&S system_ident demo on RTS suspensions (A1→A4)

**Date:** 2026-06-18
**Status:** approved (design); pending implementation plan
**Topic:** Demonstrate the Pintelon–Schoukens (P&S) optimal-excitation pipeline identifying the
LIGO digital twin's *compiled-CDS* (rtsfreerun) suspension plants under the twin's own realistic
seismic + readout noise, in four escalating rungs.

This spec is the source of truth for the demo work. It supersedes the prior `.llm/roadmap.md`
"Track A" sketch (which remains the higher-level context) and the deleted transient
`.llm/START-HERE.md`.

---

## 1. Goal & guiding decisions

Run the existing P&S loop against the compiled twin models — "the same C numerics a real front-end
runs, no hardware" — and confirm we recover the known drive→sensor plant each time.

Decisions locked in during brainstorming (2026-06-18):

- **Both, sequenced.** Each rung is *validated* (recovery asserted against an analytic oracle)
  **before** its showcase content is written. Validation gates the showcase.
- **Full ladder A1→A4.** Wiring smoke → open-loop SISO → closed-loop → MIMO/coupled.
- **A4 = per-pair FRF tensor now; Track B (parametric joint MIMO fit) is the named follow-on.**
- **Showcase = one combined Quarto example page** in `docs/examples/`, with a section per rung.
- **A1 is validation-only** — no published section of its own beyond a short intro; it is the
  guard that catches channel-name / rate / oracle mistakes before A2–A4.
- **Oracle is both** a real, tested package utility *and* consumed by the docs page (docs imports
  the package utility; only page-specific glue lives in `docs/`).

### Environment (already established)
`system_ident` and `digital_twin`/`rtsfreerun` are **separate repos**. The compiled RTS model(s)
are built **into this repo's `sysid` conda env** (not `system_ident` into the twin env). Recipe is
in the committed `README.md` §"Run against the RTSfreerun digital twin"; see the
`rtsfreerun-env-strategy` memory. On the dev box `colossus`, `x1hsts` is already built into `sysid`
and `tests/test_rtsfreerun_backend.py` runs its real-model case (8 passed / 0 skipped).

---

## 2. Architecture & data flow

The integration surface is unchanged: `RTSfreerunBackend` (`backends/rtsfreerun_adapter.py`) drives
a built `mdl` (`run` / `fetch_later` / `write`) behind the standard `ChannelBackend` interface, so
`SysIDLoop` / designer / estimator / safety run unmodified. This demo adds **no new pipeline code** —
it adds models, configs, an analytic oracle, validation tests, and one showcase page.

```
prior → P&S design → RTSfreerunBackend.inject(COIL_DRIVER_EXC)
      → mdl.run (sysID multisine + twin seismic/bosem noise) → fetch_later
      → read: X = COIL_DRIVER_OUT (drive monitor), Y = READOUT_NOISE_OUT (sensor)
      → leakage-free reference FRF → ML fit → accumulate/CRB → repeat
      → compare recovered TFModel vs analytic oracle (HSTS_DRV_TF_A ∘ _B)
```

FRF excitation is **always** the P&S multisine (`excitation.multisine_from_psd`). Twin Gaussian
noise (seismic/bosem) is background/disturbance only — never the identification drive. Grid
constraint everywhere: `T_fft = nperseg/fs > 3/f_min`; drop the first `n_transient` period(s).

---

## 3. The analytic oracle (make-or-break)

**New package utility:** `src/system_ident/backends/rtsfreerun_oracle.py` (importable, unit-tested,
used by both the validation tests and the docs page).

Responsibilities:
1. Parse a twin scenario YAML (`digital_twin/twin/scenarios/hsts.yaml`, `hsts_damped.yaml`,
   the 6dof scenario) — specifically the `init:` ZPK sections for the drive→sensor filter modules
   `HSTS_DRV_TF_A` and `HSTS_DRV_TF_B`.
2. **Compose** the two filter modules in series → the full drive→sensor transfer function.
3. Convert rtsfreerun `plane='f'` to the `s`-plane convention `system_ident` speaks — the twin's
   `plane='f'` roots are **opposite-signed from foton** (negate roots), per digital_twin `CLAUDE.md`.
4. Return a `TFModel` (and/or `(f0, Q, gain)`), so recovery is scored as parameter error or per-bin
   magnitude error against it.

The path to the twin scenario YAML is a config/argument input (the twin repo location is
machine-specific). When the scenario file is absent, the oracle raises a clear error and dependent
tests skip — never silently substitute.

**Oracle self-check (gated in A1):** an independent FRF of the model with **noise off** must overlap
the analytic oracle curve. If they disagree, the oracle (sign/compose/units) is wrong and is fixed
**before** any recovery number is trusted.

---

## 4. The four rungs

Each rung, in order: **build the model into `sysid`** (README recipe; A3/A4 builds happen at their
rung, not upfront) → **confirm channel names against the built `.mdl`** (do not trust skeleton config
names) → **validation gate** (assert in the guarded test) → **then** write that rung's section of the
page.

| Rung | Model | Loop | Inject / read / FRF-input X | Recovery bar |
|---|---|---|---|---|
| **A1** smoke | `x1hsts` | open, noise off | inject `COIL_DRIVER_EXC`; read `READOUT_NOISE_OUT`; X=`COIL_DRIVER_OUT` | plumbing only: output length = `nperseg·n_periods` after decimation; injected lines land on the right bins; 16384→`fs` is clean integer decimation; **oracle ⇄ noise-off FRF overlap** |
| **A2** open SISO | `x1hsts` | open, noise **on** (`seismic ligo-india`→`ISI_RESIDUAL_EXC`, `bosem`→`READOUT_NOISE_EXC`) | same | f0/Q/gain within tolerance of the oracle under realistic noise; uncertainty target met; drive stays under `COIL_DRIVER_LIMIT` (30000) |
| **A3** closed | `x1hstsdamped` | damping bank engaged | first-class closed-loop mode: `drive:` = after-controller drive monitor, `injection_point` set | the **open-loop** plant is recovered (controller cancelled), not the suppressed closed-loop response |
| **A4** MIMO | `x1hsts6dof` | open (coupling ⊥ closed-loop) | drive each input DoF; form the full out×in FRF tensor | each per-pair FRF matches the analytic `(out,in)` element (diagonal anti-resonance notches + off-diagonal coupling) |

Channel names above come from `hsts.yaml` and the skeleton `configs/rtsfreerun_demo.yml`; they are
**re-verified against each built `.mdl`** at the rung. Any correction is recorded in
`.claude/NOTES.md` and `.llm/roadmap.md`.

---

## 5. New / changed files

- `src/system_ident/backends/rtsfreerun_oracle.py` — analytic-oracle utility (§3). **Package code.**
- `tests/test_rtsfreerun_oracle.py` — unit tests for compose + sign conversion against known ZPK.
- `tests/test_rtsfreerun_real_model.py` — A1–A4 validation, **guarded** by
  `importlib.util.find_spec("x1hsts")` etc. (skips with no model, like
  `test_rtsfreerun_backend.py`), so the suite stays green on every machine.
- `experiments/rtsfreerun/` — one runnable script per rung that produces the figures + the executed
  campaign the page embeds (so the page is reproducible from a single entry point).
- `src/system_ident/configs/rtsfreerun_hsts.yml` (rename/clean of `rtsfreerun_demo.yml`, channels
  verified), `rtsfreerun_damped.yml` (A3), `rtsfreerun_6dof.yml` (A4). The README §"Run against the
  RTSfreerun digital twin" command references `rtsfreerun_demo.yml` — update it to the renamed
  `rtsfreerun_hsts.yml` as part of this work (or keep the old name as an alias to avoid churn).
- `docs/rtsfreerun_demo.py` — **presentation-only** glue (NOT package API), sibling of
  `sysid_plots.py` / `sysid_campaign.py`: drives a rung for the page and builds its panels, importing
  the oracle from the package and reusing the existing plot helpers.
- `docs/examples/07-rtsfreerun-twin.qmd` — the single combined showcase page (sections: smoke intro,
  A2 open-loop SISO, A3 closed-loop, A4 MIMO), with `freeze: true`.
- `docs/_freeze/` (the frozen execution cache for the page) — **committed**.
- `docs/examples/index.qmd` — add the new page to the example index/nav.

---

## 6. CI / publishing

Examples 01–06 execute live in CI (jupyter kernel). The RTS page **cannot** — `x1hsts` is not
installed in GitHub Actions. Therefore:

- The RTS page sets `freeze: true` in its front matter.
- We **commit `docs/_freeze/`** for it (locally executed on a box with the model).
- CI renders the page from the frozen cache and never imports the model → GitHub Pages stays green.
- **Acceptance check:** a clean `quartodoc build && quarto render docs` succeeds on a machine /
  environment with the model **absent**, proving the freeze path.

---

## 7. Verification strategy

1. **Per-rung validation gate** (the real bar): the guarded test asserts recovery vs the oracle for
   that rung. Gates writing that rung's page section.
2. **Oracle unit tests** pass (compose + sign conversion).
3. **Full suite green** in `sysid`: `conda run -n sysid python -m pytest -q` (new guarded tests skip
   only where a model is genuinely absent).
4. **Docs build green without the model**: `quartodoc build && quarto render docs` from frozen cache.
5. Results logged: tick `.llm/roadmap.md` Track A results stubs (A1–A4) and note any channel-name
   corrections discovered against the real `.mdl`.

---

## 8. Out of scope (named follow-ons)

- **Track B** — the parametric joint MIMO fit (shared normal-mode poles across matrix elements,
  per-element residues, MIMO Fisher/CRB). A4's per-pair FRF upgrades to it when Track B lands.
- Breadth across other suspensions (`x1quad`, etc.).
- Foton export — permanently out of scope (rejected repeatedly).

---

## 9. Risks

- **Oracle correctness** (sign/compose/units) — mitigated by the A1 noise-off self-check before any
  recovery claim.
- **Channel-name drift** between skeleton config and built `.mdl` — mitigated by re-verifying at each
  rung against the model.
- **A3/A4 model builds** may surface build issues on `colossus` — if a model won't build, **stop and
  ask the user** (standing instruction); do not improvise a substitute.
- **Freeze cache staleness** — the committed `_freeze/` must be regenerated whenever the page's code
  changes; the acceptance check (build with model absent) catches a missing/stale cache.
