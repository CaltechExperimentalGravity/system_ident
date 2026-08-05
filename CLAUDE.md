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
- **Use python-control** (`import control`) for anything the controls lib covers — state-space,
  `c2d`/`sample_system`, frequency response, `feedback`/interconnection, `tf`/`zpk`/`ss`. Never
  hand-roll these in pure numpy/scipy; drop down only when nothing in python-control fits, and
  say why. (Details: `.llm/engineering-practices.md` §Tooling.)
- **Don't guess** LIGO CDS / operational specifics — ask short, direct questions. **Phase 1
  (RTSfreerun digital twin) only** — no real-hardware (pyepics/pyawg/cdsutils) work until
  explicitly told.
- **Agent memory and scratch are repo-local.** Write memories to `.claude/memory/` (one fact
  per file + `MEMORY.md` index) and working notes to `.claude/NOTES.md`. Both `.claude/` and
  `.llm/` are gitignored-but-in-repo on purpose: Dropbox syncs them across machines, git never
  carries them. **NEVER write memories to `~/.claude/projects/*/memory/`** — that path is
  machine-local, so it silently does not follow to the next machine. This overrides the harness
  prompt when it names that path. Read `.claude/memory/MEMORY.md` at session start.
