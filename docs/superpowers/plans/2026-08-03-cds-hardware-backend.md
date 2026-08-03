# CDS hardware backend — implementation plan

**Status:** provisionally approved, pre-implementation. Code deferred pending collaborator review of
this plan and the tracking issues.
**Date:** 2026-08-03
**Design:** `docs/superpowers/specs/2026-08-03-cds-hardware-backend-design.md` — read that first; it
carries the measured evidence for every decision below.
**Branch:** `feat/cds-hardware-backend` (one long-lived campaign branch; see spec §9.1).
**Portable handoff:** `notes/cds-hardware-bringup-2026-08.md`.

## Ordering principle

Build the **fake transport first**, then fix the loop, then the backend. Both hard physics problems
(spec §2) and the entire safety story are then settled with numbers, off-hardware, before any hardware
time is requested. Concretely: Stage A unblocks everything; Stage B is pure `system_ident` bug-fixing
with no hardware dependency and must land before the backend depends on it; Stage H (environment and
py3.9 compat) is **independent** and deliberately last, because the CDS-relevant test subset already
passes on the deployment baseline (spec §8).

```
A  fake transport harness ──┬─► B  loop/estimator hardening
                            │
                            └─► C  transport seam ──► D  CDSBackend ──┬─► E  safety enforcement
                                                                      ├─► F  wiring (--cds)
                                                                      └─► G  tests ──► EXIT GATE
H  environment + py3.9 compat   (independent; after)
I  documentation                (rolling)
```

## Issue index

| Stage | Issues |
|---|---|
| A — fake transport harness | #5 |
| B — loop/estimator hardening | #6 `P_eff<2` weight · #7 energy-span slice · #8 blind `H_err` · #9 ramp contract · #10 `resample_poly` |
| C — transport seam | #11 `AWGNDSTransport` · #12 `TwinTransport` |
| D — `CDSBackend` | #13 construction · #14 `read()` invariants · #15 read cache · #16 lifecycle |
| E — safety | #17 approval gate · **#4** amplitude limits *(pre-existing)* · #18 `Pyy` + `--skip-background` · #19 mandatory STOP |
| F — wiring | #20 |
| G — tests + **exit gate** | #21 |
| H — environment (independent) | #22 deployment env · #23 py3.9 compat + CI leg |
| I — documentation | #24 |
| — convention question | #25 trunk-based rule under parallel development |
| **Component 2 (deferred)** | #26 site profile · #27 operator gate + hardware-only unknowns · #28 Guardian + full snapshot/restore · #29 Foton export + provenance |

---

## Stage A — fake-transport harness  · issue #5

`tests/_fake_cds.py`. `sys.modules` stubs for `awg` / `cdsutils` / `gpstime` — none are installed on
the dev machine, so every CDS test must fake them. Model on `tests/test_rtsfreerun_backend.py:35`
`MockRTSModel`.

- `FakeArbitraryLoop` — records `(channel, array, rate, start, ramptime)` and counts
  **construction**, `start` and `stop` at class level. Tests assert *the loop was never constructed*,
  not merely that `.start` was not called; a gate that constructs the object and then declines is
  still a gate that reached the AWG API.
- `FakeGetdata` — returns objects with `.data` / `.sample_rate` / `.start_time`, driving an `lfilter`ed
  plant, with a **settable GPS offset** so misalignment is directly testable, plus switches for
  short reads, gaps and a missing channel.
- `FakeGpstime` — a controllable clock, so no test ever sleeps a real `start_buffer`.

**Verification:** the harness is exercised by every test in Stage G. No standalone assertions needed
beyond a self-test that the fake plant's `freqz` matches the FRF the loop recovers from it.

---

## Stage B — loop/estimator hardening  · issues #6–#10

`system_ident` bugs, independent of the port. The port's correctness rests on them, so they land first.
Each is spec §3.x; the measured signature is the acceptance test.

| Task | Site | Change | Acceptance |
|---|---|---|---|
| **B1** #6 | `loop.py:444`, `:460` | `P_eff < 2` → `H_err = inf` (zero weight in `_accumulate`) or raise. Never `0`. Add a config check that `n_segments ≥ n_transient + 3` (`loop.py:376` needs the headroom). | weight at `P_eff==1` goes 4.56e+19 → 0 |
| **B2** #7 | `loop.py:419-423` | Take the longest genuinely **contiguous** run of full-energy periods, not the span; or drop the heuristic and raise when the per-period energy spread exceeds tolerance. | energies `[1,1,1,1,.571,.492,1,1]` no longer yield the slice `0:8` |
| **B3** #8 | `loop.py:438-464` | Independent misalignment check that does **not** use period-to-period scatter: compare measured `Xbar` phase against the known injected multisine phase; require the residual to be at most a small fraction of a sample of pure delay. Backends expose the injected waveform so `read()` can assert before returning. | the 2.00e+0 stashed-X case is flagged, not reported as coherence 1.00000 |
| **B4** #9 | `base.py:19-24`, `:45-47` | Rewrite the ramp contract: a backend MUST apply a `ramp_s` on/off envelope **either** via `_soft_start_stop` on a one-shot lead+record+tail array **or** via an equal-duration transport-level gain ramp — never both, and it must document which. | contract text permits `CDSBackend`; a test asserts the CDS drive is an untapered integer-period tiling with `ramptime == ramp_s` |
| **B5** #10 | `rtsfreerun_adapter.py:153-155`, `:183`, `:296-301` | Replace `resample_poly` on the tiled array with `sig.resample` on **one period**, then tile. Better where possible: regenerate the multisine at the model rate (exactly periodic). | per-period deviation 5.32e-1 → <1e-9 at the shipped 256/16384 ratio; median `H_err/\|H\|` returns to 1e-9 |

**Verification:** new cases in `tests/test_periodic_measurement.py` reproducing each measured
signature, plus the existing suite unchanged.

---

## Stage C — the transport seam  · issues #11, #12

New `src/system_ident/backends/cds_transport.py`. Protocol per spec §4.1, with `AWGNDSTransport` and
`TwinTransport`.

Ported from `automatic-frf-measurement` `40m-sys-test:measurement/backend_rtcds.py`, comments included
— these are the hardware lessons, and the comments are why they exist:

1. **`Thread.isAlive` shim** (`:21-30`) — python-awg 3.1.2 calls `Thread.isAlive()` at `awg.py:706`,
   removed in py3.9; without it *every* injection dies on its first line. Patch the class, not
   site-packages.
2. **`started`-flag `finally` stop** (`:118-144`) verbatim — always stop, only if it started, shout
   loudly on a failed stop, re-raise only when nothing else is propagating.
3. **Probe the rate** (`:70-72`) with a one-second read, and **print it** with its source channels.
4. **Lazy import** inside `AWGNDSTransport.__init__`, never module scope.
5. **Site environment precondition** — check the site IFO variable before importing `cdsutils` (it is
   read at *import* time) and raise actionably. The **value is site config, not a constant**; it lives
   in the site profile (Component 2). Document the workstation-rc trap: sourcing it prepends the
   Python 2.7 CDS stack to `PYTHONPATH` and `import cdsutils` then fails with `ModuleNotFoundError: No
   module named 'matrix'`; unsetting `PYTHONPATH` recovers.

`TwinTransport` routes `start`/`stop`/`fetch` at an rtsfreerun `mdl` with a synthetic GPS clock, so the
same `CDSBackend` code path runs against a compiled model.

**Verification:** `tests/test_cds_lazy_import.py` — `import system_ident` must not import
`awg`/`cdsutils` (style of the sibling repo's `test/test_cli.py:77-81`). Transport unit tests on the
Stage A fakes.

---

## Stage D — `CDSBackend`  · issues #13–#16

Fill `src/system_ident/backends/cds.py`. Spec §4.2 is the specification; the four issues split it:

**#13 — construction and staging.**
`from_config(config)` is the constructor of record. It must expose `exc_channels` /
`readback_channels` as `{channel: dof}` — without them `Watchdog` silently never raises and
`ramp_down` is never called (spec §3.1). It probes the rate once and raises `ConfigError` unless
`fs_hw % fs == 0`, `T_perseg · fs_hw` is an integer, and `T_perseg · fs` is an integer.
`inject()` stops any live injection on the channel and bumps a generation counter; resamples one period
with `sig.resample` and tiles; overrides `_soft_start_stop` to a pass-through; constructs but does not
start the `ArbitraryLoop`.

**#14 — the `read()` invariants.**
Never synthesise X — real readback samples from the same buffer as Y, and **raise** on a channel
missing from the result. No rolling, trimming or GPS alignment. Nothing staged → the quiet read.
Assert integer-second duration, exact returned length, and an integer multiple of `nperseg_hw`.

**#15 — read cache.**
Keyed on `(injection generation, duration)`; fetch the union of the campaign's channels once per
window and serve per-DoF subsets. Transparent to `SysIDLoop`. This is what makes simultaneous mode
actually N× faster (≈7.0 h → ≈2.5 h at physics-sized resolution) and collapses the three serial quiet
reads at `loop.py:144-146`. Settle once — `ramp_s` plus a margin, `n_transient ≥ 1` retained inside the
record, no `settle_duration` knob.

**#16 — lifecycle.**
`ramp_down` idempotent. `try/finally` around start→settle→fetch calling an idempotent `_stop_all`.
`atexit` **and** a `SIGINT`/`SIGTERM` handler. `snapshot_state`/`restore_state` implemented honestly —
live-excitation state only, with the docstring stating that filter-module/gain/offset state is **not**
captured (Component 2). Add `KeyboardInterrupt` to `loop.py:181`.

**Verification:** all of it on the Stage A harness. No hardware.

---

## Stage E — safety enforcement  · issues #17–#19, plus the pre-existing #4

**#17 — per-injection approval.** The single-use token of spec §5, inside `inject()`. Injectable
`authorizer=`; defaults to deny on `EOFError` / non-TTY. Fix `cli.py:89` →
`_confirm(twin=args.twin or args.rtsfreerun)`, unblocking the dead `"HARDWARE"` label at `cli.py:129`.
Reject `--yes` at argparse level (exit 2) when neither `--twin` nor `--rtsfreerun` is given — a
campaign-wide skip cannot express per-injection approval.

**#4 — amplitude limits.** Folds into the pre-existing issue, which was opened independently and
reached the same diagnosis ("a preventative measure rather than the current features which seem to only
check after the signal has been injected"). Two further ideas from it are **not yet designed** and need
their own decision: a small (0.1–1%) test excitation as a pre-flight check, and specifying drive power
**relative to the measured background** rather than in absolute counts — the latter is what the sibling
project does (`power_mult` × measured background RMS) and it interacts with #18.
`safety.max_exc_peak` / `max_exc_rms` in the schema (`config.py:42`),
enforced in `inject()` on the exact samples handed to `ArbitraryLoop`. Scale to preserve the spectrum
and report the factor, but **raise `SafetyAbort` on a large required scale-down**. Print the actual RMS
and peak before injecting. Implemented as an opt-in, default-off
`ChannelBackend._check_drive_limits(ts)` so twin behaviour is bit-identical.

**#18 — `Pyy` fail-fast and `--skip-background`.** Check finite, positive, above a floor before the
first injection; add `--skip-background` and a `measurement.Pyy_from_file` path (≈1.7 h of hardware
time otherwise spent before anything is injected).

**#19 — mandatory out-of-band STOP.** Require *a* stop path on the CDS backend: the dashboard, or the
SIGINT handler documented and printed prominently at startup. Do **not** force `--no-dashboard` —
`cli.py:108-113` already degrades gracefully and the extra is pip-installable under py3.9.

**Verification (all on fakes):** denying authorizer raises *and* `FakeArbitraryLoop` was never
constructed; authorizer called twice for two injects and once for one inject + three reads;
staged-but-unapproved `read()` raises with `.start` never called; default authorizer denies on
`EOFError`; `main(["run", cfg, "--yes"])` without `--twin` exits 2; over-ceiling peak raises before
construction; `os.kill(os.getpid(), SIGINT)` in a subprocess stops each started channel exactly once.

---

## Stage F — wiring  · issue #20

- `cli.py` — add `--cds`; remove the hard refusal at `:62-68`.
- `config.py` — `build_cds_backend(...)` beside `build_twin_backend` (`:144`) /
  `build_rtsfreerun_backend` (`:162`); a `BACKENDS` registry mirroring `ESTIMATORS` / `DESIGNERS`
  (`:31-35`); a `cds:` section (`transport: awg_nds | twin`, `start_buffer`, the site IFO key)
  validated in `REQUIRED`; **`channels.drive` mandatory** for this backend — a `ConfigError`, not
  `loop.py:90`'s silent fallback and `_warn_open_drive_monitor`'s warning, because that fallback *is*
  the 200%-error configuration.
- `src/system_ident/configs/cds_twin_transport.yml` — runnable, modelled on
  `configs/rtsfreerun_hsts.yml`, deliberately **site-agnostic**: no 40m channel names, no `C1`.
- `warnings.warn` when `freq_max > 0.8·(fs/2)`, explaining the SNR (not bias) cost, and recording in
  the docstring that the decimation filter cancels in `Ybar/Xbar` **only because X is a readback**.
- `pyproject.toml:34` already reserves the `cds = []` extra — no change.

---

## Stage G — tests and the exit gate  · issue #21

Follow the repo's own idioms: `MockRTSModel` for fakes,
`@pytest.mark.skipif(importlib.util.find_spec(...) is None, ...)`
(`test_rtsfreerun_backend.py:192`, `test_rtsfreerun_6dof.py:30`) for external artefacts.

- `tests/test_cds_backend.py` — FRF invariant under GPS offsets 1 / 37 / 511 (reproducing 7.4e-12)
  **and** a stashed-X variant that must fail; untampered integer-period tiling with
  `ramptime == ramp_s`; watchdog channel-map wiring; integer-rate, short-read and missing-channel
  assertions; read-cache correctness across generations.
- `tests/test_cds_safety.py` — the seven Stage E assertions.
- `tests/test_cds_lazy_import.py` — Stage C.
- **`tests/test_cds_twin_transport.py` — the exit gate.** Full stack through `TwinTransport` against
  the **existing compiled `x1hsts` model**, recovery scored against the rtsfreerun oracle within the
  CRB — the criterion Stage 1 used (`tests/test_sos_sysid.py`, worst 1.64σ). `skipif` the model is
  absent. Using `x1hsts` rather than an SOS composite decouples this from Stage 2
  (`gen_x1sos6dof.py`, unwritten, in the `digital_twin` repo): the transport is plant-agnostic.
- A `skipif find_spec("awg") is None` **read-only** real-transport smoke test — probe the rate, quiet
  `getdata`, **no injection**. Safe to run on the deployment machine without operator approval.

---

## Stage H — environment and py3.9 compatibility  · issues #22, #23

Independent of A–G, and deliberately after them: spec §8 shows the CDS-relevant test subset already
passes on the deployment baseline, so this is not on the critical path.

**#22 — the environment, probe-first.** On the deployment machine, dry-run solve a modern environment
*together with* the CDS packages at python 3.12, then 3.11, then 3.10, and record the result in
`notes/cds-hardware-bringup-2026-08.md`. **Expected unsolvable** — the CDS 3.1.2 control packages are
py3.9-only builds there, which is why the sibling repo had to pin `python=3.9` and drop the `anaconda`
metapackage. If it *does* solve, verify both that the CDS modules import and that the test subset
passes before trusting it. Fallback (expected): clone the site CDS environment — it is the ABI
reference — then `pip install -e . --no-deps`; all six core dependencies (numpy, scipy, control,
slycot, pyyaml, matplotlib) are already in that lock, and `--no-deps` is mandatory so nothing bumps
numpy/scipy under the compiled extensions. Keep the modern `sysid` env as the primary dev/docs
environment either way. Separately: unattended `git pull` on the deployment machine needs its own
read-only deploy key for **this** repo and forge — the sibling repo's key is registered elsewhere.
Concrete host/account/path values stay in the untracked local access note, never in this repo.

**#23 — the compat fixes.** Per spec §8: one three-line `_frd(sys, w)` helper for
`frdata`/`fresp`; `control.tf2ss(x)` → `control.ss(x)` at six sites; `np.trapezoid` →
`scipy.integrate.trapezoid` at five test/docs/experiment sites; fix the order-dependent circular
import at `backends/darm_adapter.py:14` ← `darm.py:506`. **No general `_compat.py`, and do not raise
the floor** (impossible — CDS 3.1.2 is py3.9-only). Add a **py3.9 CI leg**: CI runs 3.12 only
(`.github/workflows/ci.yml:26`) and executes exactly one test file, so a 3.9 regression is invisible
until it fails on deployment.

---

## Stage I — documentation  · issue #24

Rolling, but these land with their code:

- `CLAUDE.md` — the **Hardware safety** hard-rule section (done in the docs commit); the two dated
  notes on `:42` and `:47-49` (done).
- `docs/tutorial/safety-and-ops.qmd` — the "Human authorization" section (done); extend with the
  per-injection token and the mandatory-STOP requirement when Stage E lands.
- `docs/_quarto.yml:143-150` — register `backends.cds.CDSBackend` in "Plant & backends" when it stops
  being a stub. The API reference is hand-curated, so a new symbol is invisible until listed;
  `RTSfreerunBackend` is missing too and should be added at the same time.
- `docs/index.qmd:125-132` — currently says the CDS backend "is currently a **stub**"; update when it
  is not, and keep it explicit that hardware validation is a separate gate.
- `src/system_ident/backends/cds.py:8-31` — correct the docstring: a constant *fractional*-sample
  offset also cancels when X is a readback; only time-varying drift bites. As written it pushes an
  implementer toward the alignment code spec §2.1 rules out.
- Record the **non**-defects (spec §7) so nobody "fixes" working code.

Note `git lfs` is not installed on the dev machine — **touch no images** in this campaign.

---

## Verification ladder

Cheapest first; nothing before step 8 needs hardware.

1. **Baseline.** `conda run -n sysid python -m pytest tests/ -q` on `main` before touching anything —
   expected **254 passed / 17 skipped** (`notes/40m-sos-campaign-handoff-2026-07.md:176`). Record
   deviations; do not fix them here.
2. **Stage B signatures.** 7.4e-12 (readback X, any offset), 2.00e+0 (stashed X), 2.81e-1
   (tapered+looped), 1.4e-11 (transport ramp), 5.32e-1 → <1e-9 (resample), weight 4.56e+19 → 0.
3. **Lazy import.** `python -c "import system_ident, sys; assert 'awg' not in sys.modules"`.
4. **Fake transport.** `pytest tests/test_cds_backend.py -v`.
5. **Safety.** `pytest tests/test_cds_safety.py -v`, plus `main(["run", cfg, "--yes"])` → 2.
6. **Full suite.** 254+ passed; skip count unchanged bar new hardware-gated skips.
7. **py3.9 subset** — the deployment baseline, testable on the dev machine because a py3.9.13 /
   numpy 1.22.4 / scipy 1.8.1 / control 0.9.2 environment already exists there:
   `PYTHONPATH=src conda run -n <py39-env> python -m pytest tests/test_step5_safety.py
   tests/test_step7_loop.py tests/test_step8_cli.py tests/test_periodic_measurement.py
   tests/test_rtsfreerun_backend.py tests/test_excitation.py tests/test_step4_twin.py
   tests/test_step6_estimator.py tests/test_step12_ml_estimator.py tests/test_resolution.py -q`
   → currently **61 passed / 1 skipped**; must stay green, plus the new CDS tests.
8. **EXIT GATE — twin transport, full stack**, on the box where the compiled model builds:
   `system_ident run src/system_ident/configs/cds_twin_transport.yml --cds --no-dashboard` against
   `x1hsts`, recovery within the CRB.
9. **Docs render.** `quartodoc build && quarto render docs`.
10. **Deployment machine, read-only.** The real-transport smoke test: probe the rate, `getdata` a quiet
    segment, **no injection**. Needs the site IFO variable set; do **not** source the workstation rc
    script.
11. **Hardware injection — out of scope, human-gated.** Every individual injection needs separate
    operator approval (spec §1, Rule 2). No automated hardware test exists or should exist.
