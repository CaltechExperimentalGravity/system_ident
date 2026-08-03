# system_ident — agent working rules

`system_ident` is a single Pintelon–Schoukens optimal-excitation system-identification
pipeline for LIGO suspensions (periodic multisine → leakage-free reference FRF → ML fit →
Cramér–Rao-driven excitation). These rules apply to every agent (and subagent) working here.

## Feasibility gate (HARD RULE — the most important one)

Before you (a) call a result a **fundamental / physical / identifiability limit**, (b) pivot
to **"document the limitation,"** or (c) **ask the user how to proceed** on a stuck or
underwhelming result — **STOP, compute the relevant bound with real numbers, and show it.**
If the information / headroom is there, the failure is *your implementation* (init, model
order, parameterization, conditioning, under-driving, a sign/index error) — fix that.

- **Estimation/recovery "limit"?** Compute the **CRB / Fisher information**. CRB ≪ the
  observed error ⇒ implementation bug, not a limit. (History here: common-denominator pole
  "non-identifiability" = parameterization+conditioning; 6-DoF "Q-limit" = under-modeling;
  "unresolvable 0.67 Hz doublet" = fitting 13 modes and collapsing the pair.)
- **"Can't resolve X"?** Parametric ML **super-resolves**. Two modes Δf apart, linewidth
  Γ=f0/Q, are resolvable once **SNR·N ≳ (Γ/Δf)⁴** — finite, beatable by SNR/record length.
  NEVER apply the non-parametric limit (Rayleigh / peak-pick / one DFT bin / one linewidth)
  to a parametric fit.
- **"Noise-limited"?** Compute the SNR **and the actuator/dynamic-range headroom**. Fisher ∝
  SNR ∝ drive². You bury a fixed noise floor under drive; under-driving (e.g. 0.05 counts
  against a 30000-count coil limit) is not a noise limit.
- **"Resolution-limited"?** `df = 1/T` and excited-line placement are **knobs**: cluster
  lines where the parameters are informative (Fisher-optimal), and `T ≥ ~Q/f0` to resolve a Q.

The measurement-design knobs — **drive amplitude (to the actuator limit), df / record length,
excited-line placement, model order, # periods** — are things you CONTROL and set from the
math, not fixed constraints. Only present "it's a limit / give up" AFTER the bound proves it,
**with the number and the cost to beat it.** Extended checklist (local):
`.llm/engineering-practices.md`.

## Standing rules

- **Stay on the single P&S pipeline.** Check the scraped book `.llm/ps-book/` and
  `.llm/pintelon-schoukens-mimo-fit.md` before asserting what the method can or cannot do.
  When stuck, suspect a bug/conditioning/parameterization error before diverging to a non-P&S
  escape.
- **Run everything via `conda run -n sysid`** (not bare binaries). **Trunk-based:** commit and
  push to `main` (no PRs/branches). **Plots:** SVG, in Git LFS, data-driven limits.
  > *Note (2026-08-03):* it is unclear whether the trunk-based rule is still the right one for
  > **parallel development** — several people, each driving their own agents. The CDS hardware
  > campaign is running on a long-lived branch (`feat/cds-hardware-backend`) as an explicit,
  > acknowledged exception. The rule above is unchanged pending a decision; see the open question in
  > `notes/cds-hardware-bringup-2026-08.md`.
- **Use python-control** (`import control`) for anything the controls lib covers — state-space,
  `c2d`/`sample_system`, frequency response, `feedback`/interconnection, `tf`/`zpk`/`ss`. Never
  hand-roll these in pure numpy/scipy; drop down only when nothing in python-control fits, and
  say why. (Details: `.llm/engineering-practices.md` §Tooling.)
- **Don't guess** LIGO CDS / operational specifics — ask short, direct questions. **Phase 1
  (RTSfreerun digital twin) only** — no real-hardware (pyepics/pyawg/cdsutils) work until
  explicitly told.
  > *Note (2026-08-03):* this gate has been **lifted for hardware *transport* work** — writing and
  > twin-validating `CDSBackend` (awg/cdsutils/gpstime) is now in scope, per
  > `docs/superpowers/specs/2026-08-03-cds-hardware-backend-design.md`. **Driving real hardware is
  > not**: see the Hardware safety rules below. The line above is left in place rather than deleted
  > so the change is visible, not silent.

## Hardware safety (HARD RULE — no exceptions, no self-authorization)

The `CDSBackend` path drives **real** LIGO interferometer hardware. The twin, `rtsfreerun` and
`reduced` backends do not. These rules are non-negotiable and apply to every agent and subagent.

- **Only humans authorize the hardware.** **ONLY HUMANS** may configure real hardware or approve
  running a test on it. Never configure real hardware, and never self-authorize or assume approval for
  a hardware test. Close coordination with a human operator is required whenever real hardware is
  involved.
- **Every injection needs its own approval.** **Each individual instance** where a signal is injected
  into real hardware must be **separately approved** by a human operator. **One approval never carries
  over** to a subsequent injection — ask again, every time. A campaign-wide "yes" (e.g. `--yes`) can
  never satisfy this, and is rejected on the hardware path.
- **In practice:** simulation, twin runs, `--help`, imports and smoke tests need no approval. Anything
  that could actuate the real interferometer is blocked until an operator approves *that specific
  action*.
- **Never** claim a twin-green test suite validates hardware behaviour. The gaps are enumerated in
  `src/system_ident/backends/cds.py` and `docs/superpowers/specs/2026-08-03-cds-hardware-backend-design.md` §6.

(Adopted 2026-08-03 from the sibling project `automatic-frf-measurement`, which learned them on the
Caltech 40m. Operator-facing detail: `docs/tutorial/safety-and-ops.qmd`.)
