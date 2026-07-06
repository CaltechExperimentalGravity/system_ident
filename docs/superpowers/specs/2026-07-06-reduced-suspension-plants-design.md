# Design — reduced-order suspension plants in `system_ident`

**Date:** 2026-07-06  **Status:** approved design, pre-implementation.
**Goal:** bring the digital twin's **modal-truncation reduced** aLIGO suspension state-space
models into `system_ident` as first-class, self-contained sysID plants — realistic multi-stage
dynamics that are *rich enough but not unwieldy* (QUAD 75-state, HSTS ~36-state), and that run
**everywhere** (CI, Binder, no compiled twin). They serve three roles: a portable plant/backend
the P&S pipeline identifies, the **ground-truth oracle** for the MIMO modal fit, and the
realistic plants behind the **multi-DOF** worked examples.

Publish decision (owner, 2026-07-06): the reduced models are **OK to publish** — committed into
the now-public repo.

---

## 1. What comes in

Two reduced continuous state-space models, copied as committed data with a metadata sidecar:

| Model | Source (twin) | Size | Notes |
|---|---|---|---|
| **QUAD** | `twin/outputs/quad_reduced.npz` | 75 states, 36 in, 42 out, ~108 KB | modal truncation of the 1483-state full QUAD; the flagship reduction |
| **HSTS** (triple) | modal-truncation of `aligo-suspension-models/hsts_full.mat` via the twin's `sus_modal` | ~36 states | triples are already compact (a 50 Hz cut is a near-no-op); regenerated in the QUAD format for consistency |

Each `.npz` carries `A, B, C, D, eigvals_red, f_mode_cut`. The modes preserve each real-plant
mode's **frequency and Q exactly** (modal truncation is eigenvalue-preserving), so the
eigenvalues *are* the oracle.

**Provenance & self-containment.** The FRF is `G(f) = C(2πif·I − A)⁻¹B + D` — **pure numpy**,
no `control`/`slycot`/twin dependency at runtime (verified: the QUAD FRF tensor and its 36 modes,
f₀ 0.43–427 Hz with realistic Q, compute from the `.npz` alone). Two things the raw `.npz` lacks,
added on import:
- **Channel labels** — the input/output index → physical DOF·stage map (e.g. input col → `M0.L`
  drive, output row → `L3.L` disp), extracted from the twin's `SUS_indices_connections` (`sic`)
  and written to the sidecar so the plant is physically named and self-describing.
- **Provenance manifest** — source `.mat`, reduction method (modal truncation), cutoff `f_c`,
  twin commit, and a license/attribution note (the aLIGO suspension models' provenance).

---

## 2. Architecture

### 2.1 Data — `src/system_ident/models/`
- `quad_reduced.npz`, `hsts_reduced.npz` (committed).
- `quad_reduced.json`, `hsts_reduced.json` — sidecars: `input_labels`, `output_labels`,
  `modes` (f0/Q per 2×2 block), `provenance` (source, method, f_c, twin commit, license note).

### 2.2 `reduced_plant.py::ReducedStateSpacePlant`
The core, dependency-light (numpy/scipy only):
- `ReducedStateSpacePlant.load(name)` — load `.npz` + sidecar by model name.
- `.eval(freq) -> (F, n_out, n_in) complex` — the FRF tensor (numpy solve; the sysID target).
- `.modes() -> [(f0, Q), ...]` — exact modes from `eigvals_red` (the **oracle**).
- `.channels()` — the labelled input/output DOFs.
- `.subplant(sensors=[...], actuators=[...]) -> ReducedStateSpacePlant` — select a physical
  sub-block (e.g. the 6 top-mass damping DOFs of the HSTS, or one SISO L→L element) so examples
  use a tractable slice of the full I/O.

### 2.3 `ReducedPlantBackend` (the `ChannelBackend` API)
Drives the P&S pipeline exactly like `TwinBackend`, so `SysIDLoop`/the MIMO campaign are
unchanged. `inject`/`read` synthesize the periodic measured response from the reduced plant +
a sensor-noise floor (reuse the existing noise conventions), so the existing synchronous-DFT
leakage-free estimator consumes it. **Implementation note (defer to plan):** the response is
synthesized in the frequency domain — `Y(f) = G(f)·X(f) + noise` per period — to honor the
FRF-based core; a `scipy.signal.lsim` time-domain path (pure scipy, real transients) is an option
if transient realism is wanted. Either way: no slycot.

---

## 3. The three roles (phased)

**Phase 1 — Foundation (this spec's detailed scope).** §2.1–2.3: the committed models +
sidecars, `ReducedStateSpacePlant`, `ReducedPlantBackend`, and tests (§5). Nothing user-facing
changes yet; this is the reusable substrate the other phases stand on.

**Phase 2 — Portable demo (Role 1).** A new worked example: the full P&S pipeline (prior-robust
→ optimal multisine → leakage-free FRF → rank-1 modal fit → CRB) identifying the **reduced QUAD**
(or a damping-DOF sub-block), scored against its exact modes. Runs in CI/Binder — the realistic
middle between the lumped teaching plants and the local-only compiled twin.

**Phase 3 — Oracle for the MIMO fit (Role 2).** Use `hsts_reduced.modes()` as the ground truth
the HSTS 6-DOF rank-1 modal fit is scored against, driving the `ReducedPlantBackend` — a
**portable, self-contained** cousin of the compiled-twin SRM demo (which stays the local-only
"real production loops" flagship). Makes the blind-fit validation reproducible everywhere.

**Phase 4 — Realistic teaching plants (Role 3).** Adopt the reduced plants in the **multi-DOF**
examples (04 multi-DoF suspension, 06 2×2 MIMO, 09 rank-1 modal). **Do NOT touch 01–03** — a
75-state QUAD would wreck "01 — single resonance," the pedagogical on-ramp; simplicity is the
point there. Update prose/numbers/figures for the migrated examples.

**Phase 5 — Time-domain fidelity check (final, local-only).** Cross-check the FRF-based reduced
plant against the **compiled rtsfreerun twin** in the time domain: the reduced-plant FRF (and its
modes) vs the compiled-twin response over the band. Gated on the twin being present
(`$DIGITAL_TWIN_DIR`), skipped in CI — like the existing `test_rtsfreerun_real_model.py`. Adds a
row to the twin-fidelity ledger: "reduced SS ≈ compiled twin to X% over [0.1, f_c]."

---

## 4. Provenance, licensing, self-containment
- The reduced `.npz` are derived from aLIGO suspension models; the owner has confirmed they are
  OK to publish. Each sidecar records source, method, cutoff, twin commit, and an attribution/
  license note so the derivation is traceable in the public repo.
- Runtime is numpy/scipy only — no twin, `control`, or `slycot` import — so the models work in
  CI, Binder, and the Pyodide page. The twin is needed only to *regenerate* the models (a
  documented `scripts/`-style step) and for the Phase-5 fidelity check.

## 5. Testing
- **Foundation:** `ReducedStateSpacePlant.eval` FRF shape/finiteness; `.modes()` equals the
  `.npz` eigenvalues (f0/Q); `.subplant` selects the right rows/cols; the sidecar labels match
  the SS dimensions; `ReducedPlantBackend` inject/read round-trips and the leakage-free estimator
  recovers the plant FRF to tolerance on a noiseless drive.
- **Phase 2/3:** the pipeline recovers the reduced plant's modes to within CRB on the demo; the
  blind HSTS fit scores against `hsts_reduced.modes()`.
- **Phase 5:** reduced-vs-compiled-twin FRF agreement (local-only, twin-gated).
- Full-site `quarto render` clean; existing SISO/MIMO tests still pass.

## 6. Non-goals / YAGNI
- No time-domain `lsim`/`control`/`slycot` at runtime (frequency-domain core; Phase-5 uses the
  compiled twin, not a reduced-model lsim).
- No re-implementation of the twin's reduction algorithm in `system_ident` — we **copy the
  reduced artifacts** (+ a documented regen step that shells to the twin's `sus_modal`), per the
  "don't build the twin into system_ident" rule.
- No migration of examples 01–03 (keep the simple lumped on-ramp).
- HLTS and other suspension types are out of scope (QUAD + HSTS only).

## 7. Open items to resolve in implementation
1. **HSTS reduced generation.** The twin has an ASC-specific `hsts_asc_reduced_order32.npz`; we
   want the general damping-DOF HSTS. Regenerate via the twin's `sus_modal` modal truncation on
   `hsts_full.mat` (keeps all ~36 states) into the QUAD `.npz` format, and pick the DOF sub-block
   the SRM/MIMO work uses.
2. **Channel-label extraction.** Pull the real `sic` input/output labels from the twin and write
   them to the sidecars (the raw `.npz` `input_labels` is a placeholder string).
3. **Backend noise model.** Reuse the existing sensor-noise convention (the SISO/MIMO campaigns'
   `sensor_asd`/`Pyy`) so drive/SNR semantics match the rest of the pipeline.
4. **QUAD cutoff.** The committed `quad_reduced.npz` used `f_c = 100 Hz` (36 modes to 427 Hz);
   confirm this is the intended canonical cut vs the doc's 50 Hz, or re-reduce.
