# CDS hardware backend — portable handoff (August 2026)

Portable copy so this campaign can resume on any machine after a `git pull`. The session plan lived at
`~/.claude/plans/` and the session memories under `~/.claude/.../memory/` — **neither git-syncs**, so
this note plus the spec and plan are the durable record. Same reason
`notes/40m-sos-campaign-handoff-2026-07.md` exists.

- **Spec (why):** `docs/superpowers/specs/2026-08-03-cds-hardware-backend-design.md`
- **Plan (how, staged):** `docs/superpowers/plans/2026-08-03-cds-hardware-backend.md`
- **Branch:** `feat/cds-hardware-backend`, tagged `cds-backend/00-branch-point` (unchanged from
  `main` @ `f13c293`) and `cds-backend/01-plan` (this documentation commit).
- **Tracking:** GitHub issues #5–#29 on this repo (plus #4), `[deferred]`-prefixed for Component 2.

## What this campaign is

Graft the **real-RTCDS transport** from the sibling project `automatic-frf-measurement` (branch
`40m-sys-test`, GitLab) into this repo's already-stubbed `CDSBackend`, and validate it
twin-in-the-loop. Both projects implement the same Pintelon–Schoukens methodology, but only that one
has driven real 40m hardware — its July 2026 run is the only ground truth either project has about the
CDS transport, recorded in its `CHANGES-40m-sys-test.md`.

This is Stage 4 of the ladder in `notes/40m-sos-campaign-handoff-2026-07.md:59-65` (Stages 0 and 1
DONE), and `notes/strategic-roadmap-2026-07-draft.md:282-345` §Phase-C.

**Component 1 (now):** transport + backend + safety enforcement + `--cds` wiring, exit-gated on a
full-stack run through a twin transport against the **existing compiled `x1hsts` model**. No live
injection.
**Component 2 (deferred):** the site-profile layer and gated live bring-up — kept separate so the
40m-specific items get reviewed and generalised, and non-40m hardware needs only a new profile.

## Decisions locked

- **Scope split** as above. Component 1 must not hard-code a single site channel name or the site IFO
  value.
- **Exit gate uses `x1hsts`**, not an SOS composite. The transport is plant-agnostic, so this decouples
  the work from Stage 2 (`gen_x1sos6dof.py` — unwritten, lives in the `digital_twin` repo).
- **One long-lived branch**, merged when the exit gate passes. `CLAUDE.md:42`'s trunk-based rule is not
  rewritten; a dated note flags that it may need revisiting for parallel multi-developer AI work
  (issue #25).
- **`CLAUDE.md:47-49`'s Phase-1 hardware gate is lifted for transport work** as of 2026-08-03. Live
  injection remains human-gated, per the new hardware-safety rules. Recorded rather than deleted so a
  future reader does not revert the port on sight.
- ~~**A newer environment is not available for the hardware path.**~~ **REVERSED 2026-08-03** (later
  the same day). The claim below was wrong: it described the environment *installed* on the deployment
  machine (a 2022 `cds-crtools 3.1.2` clone) and generalised that to "the channel". CDS publishes a
  **python 3.11** environment (`cds-py311.yaml`) with CDS **4.1.4**, and all those packages are on
  conda-forge with `py311`/`py312`/`py313` builds. **`sysid_deploy` (py3.11 + CDS 4.1.4) is built and
  read-validated.** Struck through rather than deleted so the mistake stays visible. Original text:
  *The CDS 3.1.2 control packages (`foton`, `python-foton`, `python-awg`, `libawg`, `dtt-*`,
  `python-nds2-client`) are py3.9-only builds in the deployment machine's channel — which is exactly why
  the sibling repo had to pin `python=3.9` and drop the `anaconda` metapackage. This was the option we
  were asked to investigate; the answer is that the reverse is cheap.* See §Environment.
- **This repo is public.** Deployment-machine hostnames, accounts, paths and SSH mechanics stay in an
  untracked local note; only redacted workflow-level facts are committed
  (`notes/deployment.md`). The sibling repo tracks the concrete values only because that GitLab project
  is private.

## Findings that shaped the design (don't re-derive)

All measured in a py3.9 CDS-baseline environment before the spec was written. Full evidence in the
spec; the headlines:

1. **Period-boundary alignment is a non-problem; synthesising X is catastrophic.** `H = mean(Y_p)/mean(X_p)`
   makes a common time shift cancel identically — measured FRF error **7.4e-12** at sample offsets of
   1/37/511 with a real readback X, but **2.00e+0 (200%)** if X is the stashed drive array. So
   `read()` must never synthesise X, `channels.drive` becomes mandatory, and there is **no** GPS
   alignment code.
2. **`_soft_start_stop` is wrong for a looping transport.** `ArbitraryLoop` repeats the staged array, so
   a Tukey taper becomes periodic AM and the seam re-excites the plant every cycle — measured
   **2.81e-1** error, vs **1.4e-11** using the AWG's own `ramptime` (which is what the sibling repo
   does). `base.py`'s "MUST pass through `_soft_start_stop`" contract mandates the broken construction
   and is rewritten.
   > **AMENDED 2026-08-04 (issue #31, spec §2.3).** Note the emphasis: `_soft_start_stop` is wrong **for
   > a *looping* transport** — not wrong in itself. The conclusion originally drawn from this, *"so use
   > the AWG's own `ramptime`"*, was a mistake, because **the AWG ramp is linear** (`awgSetGain`,
   > measured to five decimals in §8b) while `_soft_start_stop` is a **cosine** Tukey taper, which is
   > **gentler on the actuator and preferred**. The linearity is the reason the transport ramp is *not*
   > an equivalent substitute.
   >
   > The fix is to stop looping: build the whole excitation as one array with a cosine envelope —
   > segmented ramp-on / settle / main / ramp-off — and inject it once via **`awg.ArbitraryStream`**.
   > Note this table's own third row already scored the **one-shot lead+record+tail construction at
   > 8.8e-12**, so the preferred construction was measured from the start, and it is the
   > `RTSfreerunBackend` one — twin and hardware can share a single envelope construction.
   > `ArbitraryLoop` is retained as a **supported peer mode** (`cds.exc_mode: stream | loop`), for
   > development, on request, and for very long records where it is structurally immune to the stream's
   > underrun risk.
3. **A bare `CDSBackend` silently disables all automatic safety.** `config.py:121-122` builds the
   watchdog with no channel maps and `safety.py:77-84` falls back to `getattr(backend, "exc_channels",
   {})`; with neither attribute, `check()` never raises **and** `abort()` never calls `ramp_down` — for
   a breach, an operator STOP, or the normal teardown.
4. **`P_eff == 1` gives one pass weight 4.6e+19, forever.** `loop.py:444` zeroes `var_H` and
   `loop.py:460`'s `1e-9` floor converts that to a measured 4.56e+19 in `_accumulate`, so one bad pass
   permanently swamps the campaign. Verbatim the sibling repo's §2.4 lesson (*exclude with `inf`, never
   `0`*), and **reachable from shipped config** via `design/resolution.py:58`'s own recommended
   `n_transient` on `configs/rtsfreerun_hsts.yml`.
5. **`H_err` and coherence cannot see any of this** — they are period-to-period scatter, and a constant
   offset is common-mode, so a 200%-wrong FRF reports coherence **1.00000**. An independent check is
   required.
6. **`resample_poly` destroys drive periodicity** — 53% deviation on the last period — and this is
   **live in `RTSfreerunBackend` today** at the shipped 256/16384 rate ratio.
7. **Nothing stops an excitation on Ctrl-C.** Zero matches for
   `KeyboardInterrupt|atexit|signal\.|SIGINT|finally` across `src/system_ident/`; `loop.py:181` catches
   only `SafetyAbort`.
8. **Simultaneous mode is currently *slower* than sequential** — each DoF triggers its own `read()`.
   At physics-sized resolution a 3-DoF × 3-iteration campaign is ≈**7.0 h**; a read cache brings it to
   ≈2.5 h.

And two of the sibling repo's scariest findings are **NOT** latent here — recorded so nobody "fixes"
working code: the `normalise_rms` ≈22.6× overshoot (measured realised `var/px_total` = **1.033** here)
and the unbanded first excitation with a −17.35-count DC offset (DC is explicitly dropped;
measured mean −1.4e-17).

## Hardware safety — non-negotiable

Copied from the sibling repo's `ai/memories/hardware_safety.md`; now a hard-rule section in
`CLAUDE.md` and prose in `docs/tutorial/safety-and-ops.qmd`.

- **Only humans authorize the hardware.** AI agents must never configure real hardware and never
  self-authorize or assume approval.
- **Every individual injection needs separate human approval. One approval never carries over.**
- Simulation / twin / smoke tests need no approval.

The sibling repo enforces neither rule in code — they are documentation-only there. This repo will
enforce them: a single-use approval token inside `CDSBackend.inject()`, `--yes` rejected on the
hardware path, and pre-injection amplitude ceilings (spec §5).

## Environment

**Superseded 2026-08-03 — read this box first.** The hardware path is now **`sysid_deploy`: python
3.11 with CDS 4.1.4**, built and read-validated on the deployment machine. Declarative spec
`environment_deploy.yml`, exact export `environment_deploy_lock.yml`, full measured detail in **spec
§8a**. Headlines:

- python 3.11.15 / numpy 1.26.4 / scipy 1.13.1 / control 0.10.2 / slycot 0.6.1; CDS `4.1.4` on
  `py311` builds; `cdsutils 1.7.0`, `gpstime 0.10.0`, `nds2 0.16.8`/`0.16.12`.
- Deployment-gate subset **61 passed / 1 skipped** — *identical* to the py3.9 baseline. Full suite
  **290 / 8 / 17**, and all 8 failures are `np.trapezoid` (numpy < 2.0) in the arcade/playground half.
- Live **read-only** `cdsutils.getdata` returned data at the expected rate. **No injection.**
- ~~**Unverified and human-gated:** awg here is 4.1.4 …~~ **RESOLVED 2026-08-04, operator-run:
  awg 4.1.4 does drive the branch-3.4 front ends.** A human operator hand-injected a 1 Hz sine
  (20 pp, 3 s ramp) and a 2 Hz square (2√3 pp, √5 s ramp) on one suspension ASC-pitch excitation
  channel under live supervision. Both recover their commanded frequency, peak-to-peak, shape and
  **ramp time** to float-level residuals — full numbers in **spec §8b**. The fallback ladder in §8a is
  therefore *not* needed, but is retained. Note this covers two **built-in waveforms**, not
  `awg.ArbitraryLoop`; and it is **not** standing authorization — every future injection still needs its
  own approval. Issue #27.
- Also open: the deploy key needs **repo-admin rights**, so the checkout is `rsync`-delivered for now
  and cannot `git fetch`.

The paragraphs below describe the *previous* target and are retained because fallback rungs 2–4 still
reference that stack.

The hardware path is pinned to the site CDS baseline: **python 3.9.13 / numpy 1.22.4 / scipy 1.8.1 /
control 0.9.2 / slycot 0.4.0.0**. Not because those versions are required by the hardware, but because
the working environment is a clone of the site CDS environment and the compiled `foton`/`awg`/`nds2`
extensions are built against that ABI. Nothing may bump numpy/scipy under them.

**The decisive measurement: the CDS-relevant half of this repo already passes on that baseline.**
`test_step5_safety`, `test_step7_loop`, `test_step8_cli`, `test_periodic_measurement`,
`test_rtsfreerun_backend`, `test_excitation`, `test_step4_twin`, `test_step6_estimator`,
`test_step12_ml_estimator`, `test_resolution` → **61 passed / 1 skipped**. The full suite does not
(250 passed / 35 failed / 12 skipped / 4 errors), but every failure is in the DARM / MIMO / SOS /
arcade / playground half, which this port does not touch. So the deployment gate is that **named
subset**, not "the suite passes", and compat work is not on the critical path.

Compat delta is four root causes, two of which need no shim (spec §8): the `frdata`/`fresp` helper;
`control.tf2ss(x)` → `control.ss(x)`; `np.trapezoid` → `scipy.integrate.trapezoid`; and the
order-dependent circular import at `backends/darm_adapter.py:14` ← `darm.py:506` (present on 3.12 too).

Two environment traps carried over from the sibling repo's first hardware run, neither a code bug:
- **The site IFO variable is not set by conda**, and `cdsutils/nds.py` reads it at *import* time. Set it
  (a site fact, so it belongs in the Component 2 site profile, never hardcoded).
- **Do not source the site workstation rc script.** It prepends a legacy site CDS python stack to
  `PYTHONPATH`, and `import cdsutils` then dies with `ModuleNotFoundError: No module named 'matrix'`.
  Unsetting `PYTHONPATH` recovers.

## Resume checklist

1. `git pull`; check out `feat/cds-hardware-backend`.
2. Read `CLAUDE.md`, then the spec, then the plan. Note that `CLAUDE.md`'s `.llm/` pointers
   (`engineering-practices.md`, `ps-book/`, `pintelon-schoukens-mimo-fit.md`) are **gitignored and
   absent** — `CLAUDE.md` + `notes/` are the binding rules.
3. Record the baseline: `conda run -n sysid python -m pytest tests/ -q` → expect 254 passed / 17
   skipped.
4. Start at plan **Stage A** (the fake transport harness). It unblocks everything, and both hard physics
   problems can then be settled with numbers before any hardware time is requested.
5. Do **not** start with the hardware code, and do not skip Stage B — the port's correctness rests on
   those five loop fixes.

## Open questions

1. Does `CLAUDE.md:42`'s trunk-based rule still hold for parallel development — several people, each
   with their own agents? (issue #25)
2. Simultaneous-mode start semantics cannot be settled off-hardware: `inj.start(ramptime>0,
   wait=False)` blocks until `start_gps`, so three loops started in a Python `for` loop with a common
   `start_gps` leave loops 2 and 3 starting late. Each individual FRF is still unbiased (X is a
   readback) but the drives are not synchronous and the loop cannot tell.
3. The other hardware-only unknowns, i.e. the human-gated set: whether awg 3.1.2 accepts the untapered
   integer-period array with `ramptime` start/stop; whether a channel's AWG slot is released so a
   second `ArbitraryLoop` succeeds; whether `_EXC` is NDS-readable at the site; `getdata` live
   short/gap behaviour; actual DAC counts vs the design budget.
   > **Partially answered 2026-08-04** by the operator-run injections in spec §8b — and on **awg 4.1.4**,
   > not 3.1.2:
   > - **`_EXC` IS NDS-readable at the site**, at the model rate (16384 Hz): the records *are* `_EXC`
   >   captures. **Closed.** This matters because §2.1 forbids synthesising X.
   > - **`ramptime` is a clean linear gain ramp**, recovered to five decimals, symmetric up and down. So
   >   the start/stop ramp mechanism works. What is still untested is whether it does so for an
   >   **`ArbitraryLoop`-staged untapered integer-period array** — a different awg API from the built-in
   >   sine/square used here. **Half-closed.**
   > - **Still open:** AWG slot release for a second `ArbitraryLoop`; live `getdata` short/gap behaviour;
   >   actual DAC counts vs the design budget.
   > - **New, from issue #31** — the default excitation mode is now `awg.ArbitraryStream`, which has
   >   **never been driven at the site**: can `append` sustain a multi-hour real-time feed without an
   >   underrun; what does an underrun look like at the front end (gap, hold, or silence); does
   >   `set_gain(gain, ramptime=…)` ramp a *stream* as cleanly as it ramps a loop; and does `abort()`
   >   stop a stream promptly. All human-gated, per-injection.
