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

## Implementation status (2026-08-10)

Stages A–F are implemented and tested — against the Stage A fakes and
`TwinTransport` only, exactly as scoped; **no real hardware has been touched,
no `awg`/`cdsutils` import has ever succeeded on this machine.** Full suite
**372 passed / 17 skipped** (up from the 254/17 baseline cited above, which
was itself stale from ordinary `main` drift — see the branch-drift note in
the plan's history, not from anything this campaign changed).

New/changed files, by stage:
- **A** (#5): `tests/_fake_cds.py` (`FakeArbitraryLoop`/`Stream`, `FakeGetdata`,
  `FakeGpstime`, `sys.modules` `install()`), `tests/test_fake_cds_harness.py`.
- **B** (#6–#10): `loop.py` (`_longest_contiguous_run`, `P_eff<2 → inf`),
  `config.py` (`n_segments ≥ n_transient+3`), `backends/base.py` (ramp
  contract rewrite), `backends/rtsfreerun_adapter.py` (`sig.resample`, not
  `resample_poly`, on the tiled drive), regression tests in
  `test_periodic_measurement.py`.
- **C** (#11/#12, #31/#32): `backends/cds_transport.py` — `CDSTransportError`
  hierarchy, `CDSTransport` protocol, `AWGNDSTransport`, `TwinTransport`;
  `test_cds_lazy_import.py`, `test_cds_transport.py`.
- **D** (#13–#16): `backends/cds.py` filled in — pre-flight probe, dual-mode
  `inject`/`read`, the generation-keyed read cache, chunked fetch + record
  verdict, lifecycle (`atexit`/`SIGINT`/`SIGTERM`, snapshot/restore);
  `loop.py` now catches `CDSTransportError`/`KeyboardInterrupt` alongside
  `SafetyAbort` in its abort path. `test_cds_backend.py`.
- **E** (#4, #17–#19): `safety.py` (`max_exc_peak`/`max_exc_rms`),
  `backends/base.py` (`_check_drive_limits`), `backends/cds.py` (the
  single-use per-injection approval token). `test_cds_safety.py`.
- **F** (#20): `config.py` (`BACKENDS` registry, `build_cds_backend`),
  `cli.py` (`--cds`, `--yes` rejected for real hardware), the site-agnostic
  `configs/cds_twin_transport.yml`. `test_cds_wiring.py`.

**Bugs found and fixed along the way** (recorded because each would have
been a real problem on the eventual hardware/twin run, not just a test
failure):
- The Stage A fake's test-point retrievability required an explicit
  `open_stream()` bracket around even a single bare `getdata`-style call —
  stricter than the real API (`backend_rtcds.py`'s own usage is bare calls,
  no separate open step). Fixed to only gate re-fetching a channel that was
  already streamed and closed.
- Fixing #6 (`P_eff<2 → inf`, never a near-infinite weight) correctly
  converted two previously-silent failures into hard exclusions: `test_darm.py`'s
  PCal test and the shipped `twin_demo.yml` were *both* structurally down to
  `P_eff=1` on every pass, always — the old bug was the only reason they
  "worked." Both fixed with real headroom, not by loosening the new check.
- `TwinTransport.stream()` reset every handle's read phase to array-index-0
  on every chunk — no persistent cursor. The throwaway settle read and the
  analysed read that follows it started from different phases, injecting a
  discontinuity exactly at the settle/analysed boundary. Would have silently
  corrupted the eventual `x1hsts` exit-gate run. Fixed with a per-handle
  cursor.
- `CDSBackend`'s `exc_channels`/`readback_channels` were first built
  `{dof: channel}`; `Watchdog` (and every other backend) needs
  `{channel: dof}` or auto-abort is silently dead. Caught immediately by the
  first test run.
- The per-injection approval gate initially prompted identically regardless
  of transport. Spec S1 is explicit that "simulation / twin / smoke tests
  need no such approval" — `config.py::build_cds_backend` now auto-approves
  for `cds.transport: twin` and uses the real interactive deny-by-default
  prompt only for `awg_nds`. `CDSBackend` itself stays transport-agnostic
  (whatever `authorizer` it's given); direct construction (bypassing
  `config.py`) still defaults to the real prompt, the safe direction to fail.

**Deferred, with reasons** (not oversights — each was investigated first):
- **#8, the independent misalignment check.** Built and numerically verified
  the literal spec mechanism ("compare `Xbar` phase to the known injected
  phase") — it does NOT catch the stashed-X failure it's meant to: a stashed
  X is bit-identical to the design at zero delay, indistinguishable from a
  legitimate real readback with zero delay, by construction. Catching it
  needs either Y/prior-model coupling (architecturally heavier, and this
  static method has no business holding a model) or a period-variance floor
  on X (false-positives on every twin/rtsfreerun test, whose X is
  legitimately noise-free by construction). The real fix already ships:
  Stage D's "never synthesise X" structural invariant. Documented in
  `loop.py` next to `_estimate_tf_periodic`.
- **#4 idea (b), `power_mult`.** Designed in spec S5 but not implemented —
  touches `config.py` validation (`px_total`/`power_mult` mutual exclusion)
  and `loop.py`'s design call site, not just `cds.py`.
- **#4 idea (a), the pre-flight test excitation.** Spec itself flags this
  "for collaborator review rather than settled" — not implemented.
- **Spec S9.3's loop-mode simultaneous-start skew minimisation** (start all
  channels unramped, then gain-ramp separately). Used the simpler, proven
  per-channel `ramptime` from `backend_rtcds.py` instead; the nuanced
  version is explicitly still an open, partly hardware-only question.
- **Per-pass reject-and-continue.** A reject-tier fault
  (`DataIntegrityError`/`TimingFault`) on an excited record currently aborts
  the whole campaign through the same path as `SafetyAbort`, not the
  narrower per-pass zero-weight skip spec S4.3.6 describes. Safe, but more
  conservative than the ideal; needs `_measure_dof`-level plumbing.

**Update (same day, later): Stage G's fault coverage and a first pass of Stage I are also done.**
`tests/test_cds_faults.py` — one case per #32 taxonomy item plus the three structural checks
(chunking invisible, excited gap rejects at zero weight, passive read retries bounded/reported) — 385
passed / 18 skipped overall now (13 new tests, 1 new skip for the real-transport smoke test, which
only runs where `awg` is installed).

Writing that suite surfaced three more real bugs, fixed the same way as everything above — caught by
building the consumer, not by inspection:
- **`_preflight()` misclassified two fault types.** A non-injectable channel (#32 item 6) and a
  not-found channel (item 7) were both raising the generic `TransportUnavailable` instead of
  `ChannelNotInjectable`/`ChannelNotFound`. Cosmetic for the abort path (both are still
  `CDSTransportError`), but wrong for anything downstream that branches on the specific type.
- **`read_chunk_s` was stored but never used.** `CDSBackend._fetch_verdict` called
  `transport.fetch(channels, duration)` with no chunk size, which both transports default to "one
  block spanning the whole duration" — meaning chunked reading, and the S4.3.2 fault-detection-during-
  the-record it exists for, was never actually exercised by a real `read()` call. Fixed by threading
  `self.read_chunk_s` through; `CDSTransport.fetch()`'s signature gained an optional `chunk_s` param.
- **A fault during the post-start settle read could leave a live, actuating channel with no teardown.**
  `_fetch_verdict`'s reject-tier handling deliberately does NOT hard-stop (a fault mid-campaign
  rejects one record, not everything) — correct in general, but wrong for the settle read specifically,
  which runs immediately after channels go live and before the campaign has analysed anything. Fixed:
  `_start_staged` now hard-stops on ANY exception from the settle read, not just what
  `_fetch_verdict` already handles.

Also found, while writing the tests rather than being a code bug: several one-shot fake fault
injections (`insert_gap`, `change_rate`) are only observable WITHIN one continuous multi-chunk
`stream()` call — a fresh `fetch()`/`stream()` call always starts with no prior block to compare its
first chunk against, so the same fault armed immediately before a fresh call is silently absorbed.
Real, if narrow: an inter-record GPS-backwards-step or rate-change cannot currently be detected by
this backend, only an intra-record one. Not fixed this round; noted here rather than left implicit.

Docs (Stage I, partial): `docs/index.qmd`'s "currently a stub" line corrected; `docs/_quarto.yml`
registers `CDSBackend`, `CDSTransport` and the previously-missing `RTSfreerunBackend`;
`docs/tutorial/safety-and-ops.qmd` gained the concrete `inject()` prompt contents, an operator-facing
fault table (one row per #32 item: what you'll see, reject-vs-abort, what to check), the
`max_exc_peak`/`max_exc_rms` config keys, and a `--cds` CLI example. Not done: `docs/tutorial/`'s other
pages, and no `quarto render` was actually run (only a YAML-validity + symbol-import sanity check) --
quarto/quartodoc weren't invoked.

**Update (same day, later still): Stage H's remaining compat work (#23) is also done.** Per spec S8a,
control 0.10.2 (this env's version) already accepts `frdata`/`tf2ss` directly, so those two compat
items were already moot — confirmed by grep, no code needed touching. What was left, both fixed and
verified in THIS environment (not just claimed for py3.9):
- **5 `np.trapezoid` sites** (numpy >= 2 only) → `scipy.integrate.trapezoid`, matching the existing
  `src/` convention (`design/pintelon.py`) — works on both old and new numpy.
- **The order-dependent circular import**, `backends/darm_adapter.py` ← `darm.py`. Reproduced first:
  `import system_ident.backends.darm_adapter` before `system_ident.darm` finishes loading raised
  `ImportError: cannot import name 'DARMBackend' from partially initialized module` — on python 3.12,
  confirming spec's "present on 3.12 too" note is not a py3.9-only concern. Fixed by moving the
  `DARMBackend` import from `darm.py`'s bottom (module scope) to a local import inside the two
  functions that actually construct one. Verified both import orders succeed, and that
  `multisine_response_sigma` still runs correctly.
- **A new CI leg**, `.github/workflows/py311-compat` in `.github/workflows/ci.yml`: python 3.11
  (matching `sysid_deploy`, not 3.9 — S8a's own correction), running the CDS-relevant deployment-gate
  subset (the original named 10 files plus the CDS backend's own test files, which didn't exist when
  that subset was fixed). Only triggers on push to `main`/`workflow_dispatch`, so it does not fire on
  this branch. Validated locally: the exact file list passes (148 passed / 2 skipped, matching the
  x1hsts/awg skips expected off the deployment machine) and the YAML parses; the 3.11 interpreter
  itself was not available to test against directly on this machine.

That closes every stage the plan scopes as reachable without the twin box or real hardware (A–I, H).
Full suite: **385 passed / 18 skipped.**

**Not started, and not reachable from this machine:** the real full-stack run against the compiled
`x1hsts` model (needs the twin box) and anything involving actual `awg`/`cdsutils` hardware access.

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

9. **Many front-end channels are not recorded to disk** — they must be captured **live** from the
   framebuilder or they are unretrievable, and that includes the `<IFO>:<MDL>-<optic>_..._EXC`
   excitation readback that finding 1 makes mandatory. *Operator-supplied 2026-08-04, not measured
   here.* Four consequences, all in spec **§4.3.1**: the NDS request must stay **open** for the whole
   record (N back-to-back `getdata` calls lose samples at every boundary); a lost or rejected
   test-point record **cannot be re-fetched**, so re-taking an excited one is a new injection needing
   fresh approval; pre-flight channel validation must happen **before** an injection is burned; and
   the read cache is a correctness affordance, not just a speed-up.
   - **`_DQ` is a useful prior:** fast non-EPICS channels normally carry it, so it normally identifies
     a readback with look-back — confirmed by probing, never assumed. But `_DQ` channels are usually
     served at a **lower rate** (properly decimated), so rate checks are **per channel**, and a
     full-rate X against a `_DQ` Y breaks §7's exact `Ybar/Xbar` filter cancellation. Default is warn
     and proceed. The filters' type is unknown — plausibly IIR; they run continuously on the realtime
     machines, so transients only matter around a restart, and a restart already hard-faults.
10. **Framebuilders failing to deliver long data stretches HAS been observed** — the one #32 item that
    is not hypothetical. Cause **never investigated** (candidates: test-point timeout, framebuilder
    resource limits, a client-side limitation); workarounds are commonly used instead. **Not
    authoritative** — no numbers, no logs, no root cause — so it is recorded as a *weak empirical
    driver* sitting alongside the architectural reasons for chunked reads (spec §4.3.2), never as
    their basis. The binding constraint is the **resources available to the framebuilder**: hardware
    with less memory, **and/or** a machine also running other tasks. Issue #32.

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

**Superseded 2026-08-10 — Stages A–F are done; see "Implementation status" above.** Steps 3–5 below
are the ORIGINAL start-of-campaign checklist, kept for history. Resuming now means: Stage G (the
dedicated exit-gate suite, plus the real `x1hsts` run — needs the twin box) or Stage I (docs).

1. `git pull`; check out `feat/cds-hardware-backend`.
2. Read `CLAUDE.md`, then the spec, then the plan. Note that `CLAUDE.md`'s `.llm/` pointers
   (`engineering-practices.md`, `ps-book/`, `pintelon-schoukens-mimo-fit.md`) are **gitignored and
   absent** — `CLAUDE.md` + `notes/` are the binding rules.
3. ~~Record the baseline: `conda run -n sysid python -m pytest tests/ -q` → expect 254 passed / 17
   skipped.~~ That baseline is stale (ordinary `main` drift); current branch baseline is **372 passed
   / 17 skipped** as of 2026-08-10 (see above).
4. ~~Start at plan **Stage A**~~ Done. Both hard physics problems were settled with numbers before any
   hardware time was requested, per the ordering principle.
5. ~~Do **not** start with the hardware code, and do not skip Stage B~~ Done — Stage B landed before C/D.

## Open questions

1. Does `CLAUDE.md:42`'s trunk-based rule still hold for parallel development — several people, each
   with their own agents? (issue #25)
   > **Amended 2026-08-06.** `git branch -a` shows a second long-lived non-`main` branch on the remote,
   > `feat/pintelon-schoukens-closed-loop`, alongside this campaign's `feat/cds-hardware-backend` — so
   > the "one acknowledged exception" framing above is incomplete. Still a process decision, not made
   > here. Candidate rule text recorded in spec §9 item 1: permit a long-lived branch for (a)
   > hardware-safety-relevant work needing collaborator review before merge, or (b) avoiding
   > destructive conflicts between concurrent agents — with an explicit merge criterion and deletion
   > after merge.
2. Simultaneous-mode start semantics cannot be settled off-hardware: `inj.start(ramptime>0,
   wait=False)` blocks until `start_gps`, so three loops started in a Python `for` loop with a common
   `start_gps` leave loops 2 and 3 starting late. Each individual FRF is still unbiased (X is a
   readback) but the drives are not synchronous and the loop cannot tell.
   > **Amended 2026-08-06 — narrowed, not settled.** `loop.py` and all three existing backends already
   > stash the drive on `inject()` and assemble/start everything together on the first `read()` of a
   > generation, so a stream-mode design needs no interface change: open every staged stream and
   > `append()`-start them in one tight loop (skew bounded by Python overhead, not a blocking
   > `start_gps` wait) — `awg.ArbitraryStream.open()` isn't documented as blocking, unlike
   > `ArbitraryLoop.start`. Loop mode keeps the `ramptime=0`-then-`set_gain` candidate fix. What's left
   > genuinely hardware-only: confirming `open()` doesn't block and measuring the actual skew. Full
   > design in spec §9.3.
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
4. **New, from issue #32** (spec §4.3, §6) — the read side, and none of it settleable off-hardware:
   - Can the live NDS path sustain a continuous multi-hour **test-point** stream without dropping
     blocks, and can `cdsutils.getdata` be used that way at all, or is the NDS2 iterate/stride API
     required? This is load-bearing: a test point cannot be re-fetched.
   - What does a **test-point release by another user** look like to a reader mid-record — an error, a
     gap, silence, or held values?
   - What framebuilder resource limit sets a workable **chunk size**? The shipped default is
     provisional until this is measured.
     > **Amended 2026-08-06.** The knob didn't exist at all before this (`read_chunk_s`,
     > `passive_read_retries` — grep of `src/`/`configs/*.yml` found neither). Now fixed to a
     > reasoned-but-still-provisional starting value: `read_chunk_s = 1.0 s`, `passive_read_retries =
     > 3` @ 0.5 s/1 s/2 s backoff (spec §9.4). This unblocks implementation; it does not supply the
     > framebuilder measurement.
   - What **type** are the front-end `_DQ` decimation filters (plausibly IIR, unconfirmed), and how
     big is the `D_Y/D_X` residual under an X/Y rate mismatch?
   - **Root cause of the observed long-stretch failures** (finding 10) — never investigated.
