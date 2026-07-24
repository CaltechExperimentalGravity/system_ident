# 40m SOS sysID deployment — campaign parking / handoff (July 2026)

Portable handoff so this project can resume on the **Linux box after a `git pull`**. The approved
plan lived at `~/.claude/plans/structured-wishing-pelican.md` and the session memory under
`~/.claude/.../memory/` — **neither git-syncs**, so this note is the portable copy. Read it first
on resume; it captures the plan, the locked decisions, the exploration findings (so you don't
re-explore), and the exact next step.

## Goal & scope

Deploy the repo's Pintelon–Schoukens optimal-excitation sysID pipeline on the **real Caltech 40m
CDS**, first identifying a **40m SOS** (Small Optic Suspension — initial-LIGO **single-stage**
pendulum, one rigid optic on wires, ~6 rigid-body DOF, OSEM coil-magnet actuation) as a **6-DOF
MIMO** system. Endorsed safe path (strategic-roadmap §Phase-C; `CLAUDE.md` Phase-1 gate): build
faithful test situations on the twin → fill & validate `CDSBackend` twin-in-the-loop → gated
hardware bring-up.

**Build scope this round:** through `CDSBackend` validated on the twin. Live-40m injection code
(`awg`/`nds2`/`pyepics`) stays **gated** on operator answers — enumerate only, don't write it.

## Decisions locked (from the planning interview)

- **Target = 40m SOS, identified as a 6-DOF MIMO system.**
- **Rehearse on a single-pendulum SOS twin — which does NOT exist yet** and is built in Stage 0.
- **CDS access = awg + nds2 (primary); cdsutils and EPICS also available.** Not a blocker.
- **Two workflow corrections from Rana (important):**
  1. **pyctl-first.** Do the analytic rehearsal in **python-control** first, then the rtsfree
     composite — matching the twin's `pyctl → signoff → rtsfree` ladder (`examples/sus_sysid/`
     already pairs `run_pyctl.py` + `rtsfree_run.py`). system_ident's `MIMOTwinBackend`/`TwinBackend`
     ARE the pyctl layer (same `ChannelBackend` contract, in-process).
  2. **Never hand-roll what python-control provides** (state-space, `c2d`, FRF, `feedback`, …).
     Now a HARD RULE in `.llm/engineering-practices.md` (§Tooling) + memory
     `use-python-control-not-hand-rolled`.
  - Rana is **moving to the Linux box** rather than chase a Mac rtsfree build. The earlier
    "Linux-only build" claim was an unverified explorer note — don't plan around it as fixed; on
    Linux just build and find out. Both pyctl and rtsfree work there.

## Approved staged ladder (pyctl-first ordering)

- **Stage 0 — 40m SOS 6-DOF plant + mode check** *(portable artifact for both repos)*. Build the
  rigid-body-on-wires SOS continuous state-space **using python-control** from the 40m physical
  params; verify eigenmodes against measured: **pend 1, pitch 0.6, yaw 0.7, side 1, vert 16,
  roll 16·√2 Hz** (feasibility gate = modes must match). Emit a **portable `.npz`** like the twin's
  existing reduced plants (`models/{quad,hsts}_reduced_50hz.npz`) that system_ident already loads.
- **Stage 1 — pyctl rehearsal** *(repo: `system_ident`)*. Run the 6-DOF MIMO P&S recovery
  **in-process against the analytic SOS plant** (via `MIMOTwinBackend`/`assemble_campaign`), compute
  **Fisher/CRB**, verify recovery vs the SOS oracle within the CRB. This is the signoff gate. Runs
  anywhere, no compiled model.
- **Stage 2 — rtsfree composite + validation gate** *(repo: `digital_twin/twin`, build on Linux
  box)*. Author `scripts/gen_x1sos6dof.py` mirroring `gen_x1hsts6dof.py` (6 `DRIVE_EXC_{DOF}`, one
  6×6 `SOS_PLANT` cdsStatespace, 6 `READOUT_{DOF}`, per-DOF dampers); drop the SOS ABCD into the
  cdsStatespace slot via `mdl.ss_set_abcd(...)`. Build into the `sysid` env. Re-run the identical
  protocol via `RTSfreerunBackend`; **analytic↔rtsfree validation gate**.
- **Stage 3 — test-situation library** *(both repos)*. Parametrized, **scored** fault/drift
  generators on the SOS twin: (a) slow plant drift (mode f0/Q wander via chunked `ss_set_abcd`,
  tracked by the existing `darm_tv` machinery, CRB-scored); (b) actuator saturation / DAC-limit
  approach; (c) weak/stuck OSEM (needs the OSEM projection layer); (d) elevated sensor/seismic
  noise; (e) damping on vs off (open-loop recovery via drive-monitor); (f) cross-coupling change.
- **Stage 4 — `CDSBackend` twin-in-the-loop** *(repo: `system_ident`)*. Fill
  `inject`/`read`/`ramp_down`/`snapshot_state`/`restore_state` behind a **pluggable transport**:
  real `awg`+`nds2` (lazy-imported) + a **twin transport** routing to the rtsfree `mdl`. Add the
  three bridge pieces: (1) timing/decoherence monitor (integer-period synchronous capture; per-period
  drive-line phase walk); (2) DAC-frame, filter-aware **pre-injection** worst-case-peak saturation
  check hooking `safety.py` (today's watchdog is post-hoc); (3) Foton ZPK/SOS export + provenance
  manifest. Wire a `--cds` path in `cli.py`/`config.py` (twin-transport mode runs the full stack).
- **Stage 5 (GATED — enumerate, no code)** — 40m-specific operator questions before any live
  injection: exact SOS optic + channel names (`C1:SUS-<optic>_..._EXC` / drive monitor / readback);
  OSEM basis matrix + counts↔N calibration + DAC/coil limit; awg/nds2 GPS timing / fractional-sample
  drift; damping-on vs off + stability margin with a comb near UGF; Guardian / lock-state +
  abort-on-lock-loss; delivered-fit manifest + Foton drop-in. Pull `C1SUS.txt` foton bank from site.

### Recommended defaults (open items — revisit with Rana)

- **Euler-basis rehearsal** (matches twin + `assemble_campaign`) + a thin, unit-tested **OSEM
  projection layer** (EUL2OSEM ~5×6 + pseudo-inverse) as the hardware-facing surface. (Alt: identify
  per-OSEM to catch a bad coil; Euler-first is cleaner.)
- **SOS-in-`HSTS_PLANT`-slot reuse** before a fully bespoke composite.
- SOS plant fidelity: rigid-body-on-4-wires order, cross-couplings (L↔P, T↔R/Y), per-mode Q (do we
  have measured Qs?).

## Exploration findings (grounds the plan — don't re-derive)

**Deployable surface (`system_ident`):**
- `RTSfreerunBackend` (`backends/rtsfreerun_adapter.py`) — **fully functional**; drives a compiled
  advligorts model via `mdl.run(excitations=, excitation_data=)` + `fetch_later(...)` on
  `<NAME>_EXC`/`<NAME>_OUT` ports, through the `ChannelBackend` contract. `rtsfreerun_oracle.py` =
  scenario init + analytic truth.
- `CDSBackend` (`backends/cds.py`) — **pure stub**: 3 methods `raise NotImplementedError("lands in
  build step 8")`. No awg/nds2/cdsutils/pyepics. CLI has **no path** to it. Its docstring lists the
  untested-until-hardware gaps (AWG↔NDS clocking/fractional-sample drift, GPS→period alignment, real
  actuator nonlinearity, true servo dynamics).
- In-process backends (the pyctl layer): `MIMOTwinBackend` (`control.forced_response` over Sd/Gd
  state spaces), `TwinBackend` (discretized), `DARMBackend`, `ReducedPlantBackend`.
- `safety.py` — Tukey ramp (`_soft_start_stop`), actuator-sat peak, per-DoF RMS ceiling, ramp-down +
  state-restore. **Post-hoc** (per-segment) — Stage 4 adds the pre-injection DAC check.
- `mimo_campaign.assemble_campaign(...)` — robust P&S MIMO: drives each actuator separately, returns
  per-excitation sample-mean spectra + covariance-of-the-mean; DFT-resolution guard.
- `excitation.multisine_from_psd(...)` — periodic leakage-free multisine (the FRF drive). CLI:
  `system_ident run <config.yml> --rtsfreerun` (`--twin`); config schema in `config.py`.

**Twin (`digital_twin/twin`):**
- **No single-stage SOS plant** anywhere — only triples (HSTS/HLTS/BS) and QUAD. `src/twin/sus_modal.py`
  `SUS_TYPES = ("quad","hsts","hlts")`. Twin works purely in **Euler DOF basis (L T V R P Y)**;
  **no OSEM/EUL2OSEM matrix** exists (OSEM appears only as a sensor-noise ASD floor).
- Best skeleton = `sus_hsts_6dof` (`scripts/gen_x1hsts6dof.py`, `examples/sus_hsts_6dof/run_rtsfree.py`):
  6 `DRIVE_EXC_{DOF}` → 6×6 `HSTS_PLANT` cdsStatespace → 6 `READOUT_{DOF}`, per-DOF `cdsFilt` dampers.
  `sus_single_damping/plant.py` shows pushing a small ABCD into the `HSTS_PLANT` slot via `ss_set_abcd`.
- sysID template = `examples/sus_sysid/` — P&S multisine on `x1hstsss`, **SISO** (M1.L only); has
  `run_pyctl.py` (analytic) + `rtsfree_run.py` (compiled) + 3-leg `validate.py`. Generalize to 6 ports.
- Scenario YAML = `model` + `noise` + `init` (filter ZPK/SOS) + `probes`. Drift = manual chunk loop
  (`examples/_lib/rtsfree_chunked.py`) re-setting `ss_set_abcd`/`fm_set_zpk` between `mdl.run` calls.
- Foton "f"-plane sign flips vs "s" (`orchestrator._coerce_zpk`); sci-notation needs explicit `+`.

**40m SOS specifics:**
- Physical params: `/Users/rana/Desktop/Dropbox/GIT/40m/NoiseBudget/GWINC40/SOSparameters.m` — optic
  R=0.075/2 m, H=0.025 m, fused silica (ρ=2200), moments of inertia, steel wire (radius, tension,
  Young's modulus, loss φw=1.7e-4), geometry. Measured modes at top: **pend 1, side 1, pitch 0.6,
  yaw 0.7, vert 16, roll 16·√2 Hz**. Also `40m/pygwinc/CIT40m/noises40.py` ("SOS optic, susp1.m").
- **No 40m foton bank on disk** — only the live-site path `/opt/rtcds/caltech/c1/chans/C1SUS.txt`
  (referenced by `40m/NoiseBudget/NB40/C1NB_2017_10_09.py`). Pull from site at hardware time.
- Zero prior 40m/SOS/C1 control artifacts in either repo — the whole site layer is greenfield.

## Repo state at parking

- `origin/main = 3ef3a61`. This handoff + the `.llm/engineering-practices.md` §Tooling edit are the
  new commit on top.
- **Deferred LFS commit `49e3efe`** (example-13 gallery thumbnail, from the DARM work) is still local
  — it will land on the next successful LFS push once the org's Git LFS budget unlocks (billing lock
  est. ~2026-07-16; see memory `ci-billing-locked-2026-07`). If a push is rejected for LFS budget,
  the handoff was split ahead of it so the non-LFS work still reaches origin.
- **Ephemeral, won't sync (recreate on Linux if wanted):** the `~/.claude` plan file and the session
  memories (`use-python-control-not-hand-rolled`, plus the DARM-era ones). The in-repo `CLAUDE.md`,
  `.llm/engineering-practices.md`, and this note carry the binding rules.

## Stage 0 DONE (2026-07-23, Linux box) — plant built + mode gate passed

Resumed on the Linux box. Corrections to the exploration findings above, and the outcome:

- **`sus_single_plant` exists but was deleted from `simplant`.** The plan's "harvest params from
  `sus_single_plant`" assumed it was live; it was removed in `simplant` commit `d158001` "repo
  cleanup (fixes #3)" (C. Wipf, 2025-02-28). Recovered from `d158001^` into
  **`digital_twin/simplant-salvage/`** (53 files + README with provenance): the RCG library, the
  OSEM CDS matrices, `x1sus.mdl`, the `X1SUS_CP.txt` foton bank, the LFS-fetched `TM_RESP.zpk`, and
  an rtsfreerun-driving-a-SUS precedent (`scripts/rtsfreerun/run_X1SUS_CP.py`) relevant to Stage 2.
- **`sus_single_plant` is only a 4-DOF model** (x, pitch, yaw, y), generated by
  `scripts/sus_plant_TM_coef.m` from LIGO DCC **T000134**. No wire-stretch DOF ⇒ no vertical, no
  roll; its pitch/yaw run 23–29% high (T000134's `b`, `R1`, `R2`, not the 40m's). Its *optic*,
  though, is 40m-SOS-class (m=0.24 vs 0.243 kg). So it was used for optic mass/inertia and the OSEM
  geometry, **not** as the plant. Decision (with Rana): build 6-DOF analytically and **constrain the
  wire params to the six measured modes** (not `SOSparameters.m`, which is not on this box).
- **Plant + gate (PASS).** `src/system_ident/sos_plant.py` builds a 12-state, 6-in/6-out
  `control.ss` in the Euler L T V R P Y basis, structural Q=50. All six modes present; L/P/V/R/Y
  hit the measured set by construction; **T (side) is not fitted** — `ω_T²=g/(l+b)` follows from the
  pinned `l,b` and lands **−0.18%** from 1 Hz (the real check). Independent corroborations the fit
  did not target: fitted wire separation `d`=exactly 2R (optic edge); `l`=0.2487 m and I_pitch=9.81e-5
  both within 0.3% of T000134. Caveat: measured roll is recorded as 16·√2, which may itself be
  derived — `d=2R` is equivalent to roll/vert=√2, so treat that one as consistency, not proof.
- **OSEM layer.** `src/system_ident/osem.py` — EUL2OSEM/OSEM2EUL for UL/UR/LR/LL/SIDE (the layer
  neither repo had). Reaches only L,T,P,Y; **V and R have no OSEM actuation** (rank 4). Trap found &
  documented: the two salvaged CDS matrices use *different* coil orderings — composing them as-shipped
  sends YAW to zero — so the module derives its matrices from geometry and pins `COIL_ORDER`.
- **Artifact + loader.** `src/system_ident/models/sos_6dof.{npz,json}` (built by
  `models/build_sos.py`, analytic, no `aligo-suspension-models` dep). `ReducedStateSpacePlant.load`
  gained an optional `suffix=` (default `_reduced_50hz`, so quad/hsts untouched);
  `load("sos", suffix="_6dof")` round-trips. **`sysid` conda env created here** (was absent).
- **Tests:** `tests/test_sos_plant.py` (26) pass; full suite **246 passed, 17 skipped** (skips are
  compiled-twin/browser deps absent on this box), no regressions.

## Stage 1 DONE (2026-07-23) — open-loop recovery within CRB; closed-loop deferred to Stage 2

- **Open-loop rung (the signoff gate), committed `e13fc26` + pushed.** `src/system_ident/sos_campaign.py`
  + `tests/test_sos_sysid.py` (8). 6-DOF MIMO P&S recovery vs the plant oracle **within the CRB**
  (worst 1.64σ; well-excited f0 to <2e-3). Two focused Fisher-clustered campaigns (a single broadband
  sweep collapses the per-bin covariance to rank-1 off-resonance → singular {L,P} whitening): **low
  0.4–1.2 Hz drives L,P,T,Y; high 14–26 Hz drives V,R.** Block-decoupled fit (only L–P coupled); the
  L/T 1.8 mHz spatial doublet is resolved and scored per-DOF. Full suite 254 pass / 17 skip.
- **Closed-loop rung → moved to Stage 2 (Rana's call).** The nominal SOS damping design already
  exists in the salvage — `simplant-salvage/X1SUS_CP.txt` bank `OPT_CTRL_SUS{POS,PIT,YAW,SIDE}`
  (L/P/Y/T), native digital foton biquads @32768 Hz, each with a **BounceRoll notch**; **V (bounce)
  and R (roll) are undamped by design** (nominal — the OSEMs are rank-4 and can't reach V/R, Stage 0).
  Do NOT hand-roll velocity dampers. rtsfree loads these foton banks natively, so identify through the
  real loops in Stage 2 rather than converting foton→continuous for a pyctl loop.

**Next: Stage 2 (rtsfree composite).** Author `gen_x1sos6dof.py` (derivative of the twin's
`scripts/gen_x1hsts6dof.py`) driving the SOS ABCD via `ss_set_abcd`; load the nominal
`OPT_CTRL_SUS*` damping bank; identify through the loops; analytic↔rtsfree validation gate. Reuse the
salvaged `run_X1SUS_CP.py` + `x1sus_cp_funcs_freerun.py` rtsfree-SUS precedent. Needs the `twin`
conda env + advligorts toolchain (Linux). `simplant-salvage/` is in `digital_twin/` (separate repo) —
kept local for now (Rana), commit-location still open.

## Resume checklist (on the Linux box)

1. `git pull`; read this note + `CLAUDE.md` + `.llm/engineering-practices.md`.
2. Confirm the two open defaults with Rana if you want (Euler+OSEM-projection; slot-reuse).
3. **Start Stage 0** — SOS 6-DOF plant via **python-control** from `SOSparameters.m`; assert the six
   eigenmodes match the measured list; emit a portable `.npz`. (Feasibility gate: modes first.)
4. Then Stage 1 (pyctl rehearsal + CRB), then Stage 2 (rtsfree composite + analytic↔rtsfree gate).

## Pointers

- Prior-art blueprint: `notes/strategic-roadmap-2026-07-draft.md` §Phase-C (8 operator questions,
  4-step minimal bridge); `notes/roadmap-and-engineer-questions.md` §A–I; `notes/twin-fidelity-ledger.md`.
- Method: `.llm/pintelon-schoukens-mimo-fit.md`, `.llm/engineering-practices.md`, `.llm/ps-book/`,
  `.llm/rtsfreerun-integration.md` (adapter design + 4-iteration bring-up ladder).
- Twin: `digital_twin/twin/{ARCHITECTURE.md,ROADMAP.md,CLAUDE.md}`, `examples/sus_sysid/`,
  `scripts/gen_x1hsts6dof.py`.
