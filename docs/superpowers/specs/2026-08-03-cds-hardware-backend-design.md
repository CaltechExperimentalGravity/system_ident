# CDS hardware backend — real-RTCDS transport behind `ChannelBackend`

**Status:** approved design, pre-implementation (provisional — pending collaborator review)
**Date:** 2026-08-03
**Arc:** Stage 4 of the 40m SOS deployment ladder (`notes/40m-sos-campaign-handoff-2026-07.md:59-65`);
`notes/strategic-roadmap-2026-07-draft.md:282-345` §Phase-C.
- **Stage 0 (done):** 40m SOS 6-DOF plant + mode gate (`sos_plant.py`, `osem.py`).
- **Stage 1 (done, `e13fc26`):** 6-DOF MIMO P&S recovery within the CRB (worst 1.64σ).
- **This spec (Component 1):** fill `CDSBackend` behind a pluggable transport, validated
  twin-in-the-loop. **No live injection.**
- **Component 2 (deferred):** site-profile layer + gated live bring-up.

## 0. Why now, and where the code is coming from

The sibling project `automatic-frf-measurement` (branch `40m-sys-test`) implements the same
Pintelon–Schoukens methodology and **has driven real 40m hardware**. That first hardware run is the
only source of ground truth either project has about the CDS transport, and it produced a specific,
short list of lessons recorded in that repo's `CHANGES-40m-sys-test.md`.

Its entire hardware surface is one 150-line module, `measurement/backend_rtcds.py`, whose only CDS
calls are:

```python
from awg import ArbitraryLoop          # inject
from cdsutils import getdata           # read back
from gpstime import gpstime            # schedule

rb   = getdata([*capture_channels, exc_channel], 1)   # rate is PROBED, not configured
inj  = ArbitraryLoop(exc_channel, excitation, rate=exc_rate, start=start_gps)
inj.start(ramptime=ramp_time, wait=False)
inj.stop(ramptime=ramp_time)                          # always, in a finally
```

It depends on nothing else from that repo but a four-field dataclass. Meanwhile `system_ident`
already has the socket: `ChannelBackend` (`src/system_ident/backends/base.py:16`) and a stubbed
`CDSBackend` (`src/system_ident/backends/cds.py:41`) whose three methods raise `NotImplementedError`.

**So this is a graft, not a merge.** Nothing from the other repo's `measurement/core.py`,
`optimisation/`, `signals/` or `interact/` is ported — `system_ident` has its own, better-validated
equivalents (its `_estimate_tf_periodic` is leakage-free where the other repo's `estimate_frf` is
Welch/CSD). What is ported is the transport and the lessons.

## 1. Hardware safety — the binding rules

Copied from `automatic-frf-measurement`'s `ai/memories/hardware_safety.md`. That repo's convention is
a per-topic memory file; `system_ident` has no such directory, so these rules live in `CLAUDE.md` as a
hard-rule section, in `docs/tutorial/safety-and-ops.qmd` as prose, and here as design constraints.
**Reproduced verbatim** because they are policy, not paraphrasable guidance:

> ## Rule 1 — only humans authorize the hardware
>
> **ONLY HUMANS** are authorised to configure the real hardware and to approve running tests on it.
> Claude / AI agents must **never** configure the real hardware, and must never self-authorize or
> assume approval for a hardware test. Close coordination with human operators is required whenever
> the real hardware is involved.
>
> ## Rule 2 — every signal injection needs separate approval
>
> **Each individual instance** where a signal is injected into the real hardware must be
> **separately approved** by a human operator. One approval never carries over to subsequent
> injections — ask again, every time.
>
> ## In practice
>
> - Simulation / smoke tests need no such approval.
> - Anything that could actuate the real interferometer is blocked until a human operator approves
>   that specific action.

Notably `automatic-frf-measurement` enforces **neither rule in code** — there is no `input()`, no
`--confirm`, no dry-run on its hardware path; the rules are documentation-only there. `system_ident`
should enforce them, because it already has most of the machinery (§5).

## 2. The two hard problems, settled with measurements

Both were stress-tested numerically in the py3.9 CDS-baseline environment before this spec was
written. The measured numbers are the justification; do not re-litigate them from first principles.

### 2.1 Period-boundary alignment is a **non-problem** — but synthesising X is catastrophic

`_estimate_tf_periodic` (`loop.py:383`) reshapes on the assumption that sample 0 is a period boundary
(`P = len(x) // nperseg`, `:408-411`). The obvious conclusion — that `read()` must GPS-align the
buffer — is **wrong**. The estimator is a ratio of averages, `H = mean_p(Y_p) / mean_p(X_p)`
(`loop.py:432-435`), so a common time shift multiplies X and Y by the same phasor and cancels
identically.

Measured (Q=20 @ 1 Hz plant, fs 256, nperseg 1024, P=8, settled; truth = the discrete plant's own
`freqz`), max in-band relative FRF error:

| case | X = real readback | X = stashed drive array |
|---|---|---|
| aligned | 7.4e-12 | — |
| rolled 1 sample | 7.3e-12 | **1.96e-1** |
| rolled 37 samples | 1.7e-12 | **2.00e+0** |
| rolled 511 samples | 5.4e-12 | **2.00e+0** |
| rolled 4096 (= 4·nperseg) | 9.9e-13 | 9.9e-13 |

A **200% error** — the FRF is destroyed — for any offset that is not an exact multiple of `nperseg`.
A constant *fractional*-sample offset behaves the same way and cancels equally well with a readback X;
only time-*varying* drift bites.

**Design consequence — an invariant, not a calculation:**

1. `CDSBackend.read()` **never synthesises X.** For the excitation / drive-monitor channel it returns
   real `getdata` samples from the same buffer as Y, and **raises** if a requested channel is absent
   from the result. (`_EXC` is a testpoint rather than a recorded channel at many sites; the other
   repo read it successfully at the 40m, but that is a site fact, not a guarantee.)
2. **No rolling, trimming or GPS alignment in `read()`.** Return the raw buffer. Computing the offset
   from `((start_time − start_gps)·fs) mod nperseg` is not even reliably possible —
   `gpstime.now().gps()` is truncated to `int` at the injection site, `getdata`'s `start_time` is the
   NDS buffer's own boundary (integer-second aligned in practice, but that is an implementation detail,
   not a contract), the AWG's start latency relative to `start_gps` is unspecified, and resampling
   introduces the rate ratio. It would be a correction whose failure mode is a silent 200% error.
3. `channels.drive` becomes **mandatory** for this backend. `loop.py:90` currently falls back to
   `exc[d]` silently and `_warn_open_drive_monitor` (`:274-297`) only warns. On hardware, "X falls back
   to the excitation channel" plus "the excitation channel returns the stashed drive" is precisely the
   200%-error configuration, so it must be a `ConfigError`.
4. `RTSfreerunBackend` *does* synthesise X for its excitation channels
   (`rtsfreerun_adapter.py:228-230`). That is safe there because the adapter owns the sample clock
   exactly. **Do not copy that pattern here.**

The `cds.py:8-31` docstring needs correcting: "a constant integer-sample offset *cancels* … but a
fractional-sample clock drift would re-introduce leakage" is only half right, and as written it pushes
an implementer toward exactly the alignment code this section rules out.

### 2.2 `_soft_start_stop` is wrong for a **looping** transport

> **CORRECTION (2026-08-04).** Everything below about the *looping* transport stands — the measurements
> are real. Two things in it are wrong, and **§2.3 supersedes its design consequence**:
>
> 1. **The two ramps are not interchangeable.** `_soft_start_stop` is `scipy.signal.windows.tukey`, a
>    **cosine** taper (C¹ — slope starts at zero). `ArbitraryLoop.start(ramptime)` is `set_gain(0)` then
>    `set_gain(g, ramptime=…)` → `awgSetGain`, a **linear** gain ramp with a slope discontinuity at both
>    ends. The 2026-08-04 operator run (§8b) measured that linearity to five decimals. A cosine taper is
>    **gentler on the actuator and is preferred**, so "use the transport's own ramp" is a *downgrade* in
>    envelope quality, not a neutral swap. It was accepted below only because the looped construction
>    left no alternative.
> 2. **"`ramp_down` would have to rewrite an already-queued array, which is impossible" is false** for
>    `awg.ArbitraryStream`. `ArbitraryStream` — which `ArbitraryLoop` *subclasses* — carries
>    `set_gain(gain, ramptime=…)` → `awgSetGain` and `abort()` → `SIStrAbort`, both **independent of the
>    queued data**. So `ramp_down(channel, secs) → set_gain(0, ramptime=secs)` maps exactly on a one-shot
>    stream, and §5's mandatory STOP has `abort()`. This removes one of the two arguments that pushed the
>    design to `ArbitraryLoop`.
>
> Note the table below already scores **one-shot lead+record+tail at 8.8e-12**, as good as the
> `ramptime` path — so the preferred construction was measured all along, and it is the
> `RTSfreerunBackend` one, which means twin and hardware can share a single envelope construction.
> `ArbitraryLoop` remains a **supported peer mode**, not a deprecated path: see §2.3.

`base.py:26-39` applies a Tukey envelope to the whole injected array, and `base.py:30-32` says every
actuating backend MUST use it. But `ArbitraryLoop` **repeats** the staged array, so the taper becomes
a periodic amplitude modulation at the *loop* period rather than a one-shot envelope: the drive is
periodic at the loop period, not at `nperseg`, so the reshape is no longer synchronous and the DFT
leaks — and the seam (ramp to zero, ramp from zero) re-excites the plant's transient **every cycle**,
so no steady state is ever reached. `_choose_transient` cannot help: the transient recurs, it does not
decay.

Measured (`ramp_s = 3.0` s over a 32 s / 8-period array, looped, reading 8 periods from the steady
stream):

| construction | max rel FRF error | median `H_err/\|H\|` | median coherence | periods kept by `loop.py:421` |
|---|---|---|---|---|
| `_soft_start_stop` + AWG loop, read at the loop seam | **2.81e-1** | 6.0e-2 | 0.99638 | `[1,2,3,4,5,6]` |
| same, read rolled 3000 samples | 6.68e-2 | 2.8e-1 | 0.92973 | `[0,1,2,3,6,7]` ← **non-contiguous** |
| one-shot lead+record+tail (the `RTSfreerunBackend` construction) | 8.8e-12 | — | — | — |
| **AWG's own `ramptime`, array untapered** | **1.4e-11** | — | — | — |

This is the live configured path: `configs/*.yml` set `t_ramp: 3.0`, which
`rtsfreerun_adapter.py:143` maps to `ramp_s`.

**Design consequence:** `CDSBackend` overrides `_soft_start_stop` to return its input unchanged and
uses the transport's own gain ramp — `inj.start(ramptime=...)` / `inj.stop(ramptime=...)` — which is
exactly what `automatic-frf-measurement` does (`backend_rtcds.py:123,134`). It also makes
`ramp_down(channel, secs) → inj.stop(ramptime=secs)` map exactly; under a one-shot construction
`ramp_down` would have to rewrite an already-queued array, which is impossible.

**The two ramps cannot coexist.** Double-ramping squares the envelope over `2·ramp_s`, lowering the
leading periods' energy further and breaking the equal-energy assumption `loop.py:419-423` relies on.

So `base.py`'s contract must be rewritten (not weakened): *a backend MUST apply a `ramp_s` on/off
envelope, **either** via `_soft_start_stop` on a one-shot lead+record+tail array (see
`RTSfreerunBackend.read`) **or** via an equal-duration transport-level gain ramp (see `CDSBackend`) —
never both, and it must document which.* A test asserting the staged CDS drive is an untapered
integer-period tiling with `ramptime == ramp_s` is what keeps that from becoming a loophole.

**Amended by §2.3:** the contract stays two-branch, but each branch must additionally document **which
envelope shape it produces** — the omission that let a cosine taper and a linear gain ramp be treated as
equivalent in the first place.

### 2.3 The segmented `ArbitraryStream` excitation, and two peer modes

Since a cosine envelope cannot be had from a *looped* array (§2.2: tapering the staged array gives
**2.81e-1**), get it from a different awg API. Build the whole excitation as **one array carrying a
cosine on/off envelope** and inject it once with **`awg.ArbitraryStream`**, in four temporal segments:

| # | segment | duration | envelope | analysed? |
|---|---|---|---|---|
| 1 | **ramp-on** | `t_ramp` | cosine (rising half-taper) | no |
| 2 | **settle** — the system under test's transient decays | `warmup_s` | flat, full amplitude | **no — discarded** |
| 3 | **main excitation** — read back, FRF measured | `segment_duration × n_segments` | flat, full amplitude | **yes** |
| 4 | **ramp-off** | `t_ramp` | cosine (falling half-taper) | no |

**No new configuration.** The four durations derive from keys that already exist and already mean this
in the twin and `rtsfreerun` backends — `t_ramp`, `warmup_s`, `segment_duration`, `n_segments` — so there
is one meaning per knob across every backend. Shipped `configs/rtsfreerun_hsts.yml`: `3 + 24 + 96 + 3 =
126 s`.

Segments 1, 2 and 4 lie **outside** the analysed window, which is the whole point: the taper never
touches the periods that enter the FRF, so §2.1's requirements are untouched and the periodicity of
segment 3 is exact. `read()` must therefore window **segment 3 only** (§4.2), and a config check must
enforce `t_ramp ≤` the lead it is tapering into.

#### Two supported peer modes — `cds.exc_mode: stream | loop`

`ArbitraryLoop` is **not** demoted to a fallback. Both modes are supported, both tested, both in the
fake harness. `exc_mode` defaults to **`stream`**; selecting `loop` is a deliberate choice and **warns**,
naming the consequence — a linear envelope, harsher on the actuator than the cosine taper.

`loop` has four legitimate roles:

1. **Fallback** if stream injection misbehaves on hardware.
2. **Development pipeline** — simpler to drive.
3. **Valid on request**, or wherever a linear ramp carries no disadvantage for the use case.
4. **Very long excitations.** ← see the correction below.

#### Role 4, computed rather than asserted

Per the feasibility gate: do not claim a limit without the number.

- Shipped config, one experiment = 126 s. At `fs_hw = 16384` in float64: `126 × 16384 × 8` ≈ **16.5 MB**.
  Memory is a non-issue.
- A **Q-limited** physics-sized record is different. `T ≥ ~Q/f0` gives `T_perseg ≈ 1493 s` for Q = 1000
  at 0.67 Hz, so 6 periods ≈ 8955 s ≈ 2.5 h → **≈1.17 GB per channel**, ≈3.5 GB for 3 DoF simultaneous.
  Large but tractable, and §4.2's campaign estimates already run to hours.
- **But `append(data, scale)` can be fed in chunks.** The segmented envelope needs only the total
  *length* up front — not the samples — so the array need never be resident, and it can be generated
  period by period. **The memory rationale for `loop` therefore dissolves.**
- **What does not dissolve is stream underrun.** `append` pushes data in real time from Python; a stall
  (GC, NFS, network) starves the stream and leaves a **gap in the excitation**. `ArbitraryLoop` is
  structurally immune, because the front end repeats a staged buffer with no further Python involvement.

So role 4 is justified — **on underrun-immunity grounds, not memory.** That is the stronger argument, and
it makes the case less remote than a pure edge case: it scales with record duration, and the durations
this campaign targets are hours.

#### Underrun is a §3.3-class fault, and must be detected explicitly

A gap in the excitation is exactly what `H_err` and coherence **cannot see** (§3.3: they are
period-to-period scatter, so a corrupted FRF can still report coherence ≈ 1). Two detections already
exist in the design and should be reused rather than reinvented:

1. **§3.3's independent check** — measured `Xbar` phase against the known injected multisine phase.
2. **Compare the `_EXC` readback against the commanded array.** Newly known to be feasible: §8b measured
   `_EXC` as NDS-readable at the model rate, round-tripping amplitude and shape exactly. A gap shows up
   directly.

#### Deferred — planned, not implemented

- **Shrinking segment 2, via issue #1** (Local Polynomial Method / GraFIT). LPM estimates the
  transient/leakage term rather than waiting it out, so the discarded settle could shrink and with it the
  dominant fixed cost per experiment. **Defer planning.**
- **MIMO.** Segments 1–4 constitute one **'experiment'**. A MIMO measurement contains one experiment
  **per input**, so `n_experiments = n_inputs`. **Defer planning.**

**§4.1 and §4.2 below still specify `ArbitraryLoop` as the only transport** and must be reworked to
express both modes. Not done here — tracked on the owning issue.

## 3. Defects in `system_ident` that a hardware run would hit

Found while stress-testing the above. All are independent of the port; the port's correctness depends
on them, so they are fixed first.

### 3.1 A bare `CDSBackend` silently disables all automatic safety

`config.py:121-122` builds the watchdog with no channel maps, and `safety.py:77-84` falls back to
`getattr(backend, "exc_channels", {})` / `readback_channels`. `CDSBackend` has neither attribute, so:

- `Watchdog.evaluate` (`safety.py:100-118`) never matches a channel → `breaches` is always `[]` →
  **`check()` never raises.** Actuator-saturation and RMS-ceiling auto-abort are both dead.
- `Watchdog.abort` (`safety.py:137-138`) iterates `self.exc_channels` → **`ramp_down` is never
  called**, for a breach, an operator STOP, *or* the `loop.py:187` normal teardown.
- `loop.py:232,237` report `nan` and nothing complains.

Fix: `CDSBackend` must expose `exc_channels` / `readback_channels` as `{channel: dof}`, the same shape
`TwinBackend` and `RTSfreerunBackend` use, built by a `from_config` classmethod. Regression guard:
`Watchdog(CDSBackend.from_config(cfg), limits).exc_channels != {}`.

### 3.2 `P_eff == 1` gives one pass weight 4.6e19, forever

`loop.py:444` sets `var_H = 0` when only one period survives; `loop.py:460`'s `H_err = maximum(H_err,
1e-9·|Hb|)` floor turns that into a measured weight of **4.56e+19** in `_accumulate`
(`loop.py:317-322`) — identical to a healthy 8-period pass. One bad pass then permanently swamps every
subsequent pass for the rest of the campaign.

This is verbatim the other repo's §2.4 lesson: *a bin is excluded by making its uncertainty
**infinite**, not zero — zeroing it produces an infinite weight, the opposite of the intent.* And it
is **reachable from shipped config**, not hypothetical: `_choose_transient` (`loop.py:359-360`) returns
`min(n_min, max(P-1, 0))` when `P <= n_min + 2`; on `configs/rtsfreerun_hsts.yml` (`n_segments: 6`,
`segment_duration: 16`, Q≈50 modes at 0.67 Hz → τ = 23.8 s), `design/resolution.py:58`'s own
recommendation gives `n_transient = ceil(3·23.8/16) = 5` → `P=6, n_drop=5` → **`P_eff = 1`**. The
config only escapes by omitting `n_transient` and defaulting to 1.

Fix: `P_eff < 2` → `H_err = inf` (zero weight) or raise, consistent with `loop.py:410`'s "needs at
least 2 whole periods". Never 0. Plus a config check that `n_segments ≥ n_transient + 3`
(`_choose_transient` needs that headroom to adapt at all, `loop.py:376`).

The rest of §2.4 **is** already correct here: `loop.py:458` uses `inf` for unexcited bins and
`loop.py:317` weights on `isfinite & > 0`.

### 3.3 `H_err` and coherence are structurally blind to §2

Measured: a 200%-wrong FRF (stashed X, rolled 37) reports median `H_err/|H|` = **1.0e-9** and median
coherence **1.00000** — indistinguishable from the 7.4e-12-accurate case.

Because `var_H` (`loop.py:438-443`) is period-to-period residual scatter, and a constant time offset
is common-mode across every period, `Y_p − H·X_p ≡ 0` for all `p`. Neither existing guard helps:
`_choose_transient` (`:370-379`) also keys on scatter, and the energy test (`:419-423`) only sums
`xr**2`, which a pure time shift leaves unchanged.

Fix: an **independent** check that does not use scatter — compare measured `Xbar` phase against the
*known injected* multisine phase and require the residual to be at most a small fraction of a sample
of pure delay. The backend has the injected waveform, so it can assert this before returning.

### 3.4 `loop.py:419-423` takes the span, not the contiguous run

```python
full = np.flatnonzero(e >= 0.999 * e.max())
if full.size >= 2:
    xr, yr = xr[full[0]: full[-1] + 1], yr[full[0]: full[-1] + 1]
```

Measured in §2.2's rolled case, per-period energy/max = `1.000 1.000 1.000 1.000 0.571 0.492 1.000
1.000`, so `full = [0,1,2,3,6,7]` and the slice `full[0]:full[-1]+1` is `0:8` — it keeps the 57% and
49% periods. The comment claims "the contiguous block of full-energy periods"; it is the *span* of
them, which is the whole array whenever the low-energy periods are interior — exactly the looped-taper
geometry. Fix: longest genuine run, or drop the heuristic once the ramp lives outside the record and
**raise** when the per-period energy spread exceeds tolerance.

### 3.5 `resample_poly` destroys the drive's periodicity — and is live today

Measured on an 8-period tiled multisine, per-period max deviation from the interior period:
`6.90e-02  0  0  0  0  0  0  5.32e-01`. The last period is off by **53%**, the first by 6.9%, because
`resample_poly`'s FIR sees the zero-padded array ends. Resampling *one* period then tiling is worse
(5.33e-1 seam discontinuity). The AWG loops that, so the corrupt period recurs forever.

Alternatives measured against the `resample_poly` interior: `scipy.signal.resample` (FFT, exactly
periodic) on one period → **8.6e-4**; regenerating the multisine directly at the hardware rate →
**exactly 0**, with line frequencies on exact bins of both grids.

**This defect is live in `RTSfreerunBackend` now**: `inject()` uses `resample_poly`
(`rtsfreerun_adapter.py:153-155`) and `read()` tiles the resampled array via
`_fit_periodic`/`np.resize` (`:183`, `:296-301`), exercised by the shipped `configs/rtsfreerun_hsts.yml`
fs 256 / model 16384 ratio. It self-flags there (median `H_err/|H|` rises 1.0e-9 → 1.8e-4), unlike
§3.3, but it should be fixed at both sites.

### 3.6 Nothing stops an excitation on Ctrl-C

`grep -rn 'KeyboardInterrupt|atexit|signal\.|SIGINT|finally' src/system_ident/*.py
src/system_ident/backends/*.py` → **zero matches.** `loop.py:181` catches only `SafetyAbort`, so an
interrupt during a multi-hour `read()` propagates past `watchdog.abort()` at `loop.py:187` and the AWG
keeps driving. The other repo solved this at `backend_rtcds.py:127-144` with a `finally` + `started`
flag + loud warning + conditional re-raise; that logic is correct and hard-won and is ported verbatim.

Fix belongs in the backend, not the loop (the loop must stay backend-agnostic, `base.py:3-5`):
`try/finally` around start→settle→fetch calling an idempotent `_stop_all`, plus `atexit` **and** a
`SIGINT`/`SIGTERM` handler (`atexit` does not fire on `SIGTERM`). Add `KeyboardInterrupt` to
`loop.py:181` as belt-and-braces.

## 4. Architecture

### 4.1 The transport seam

`src/system_ident/backends/cds_transport.py` — what
`notes/40m-sos-campaign-handoff-2026-07.md:60-61` asks for ("behind a **pluggable transport**"), and
what makes everything in §5 verifiable without hardware.

```
CDSTransport (Protocol)
    now_gps() -> float
    probe_rate(channels) -> float                       # getdata(chans, 1)
    start(channel, array, rate, start_gps, ramptime) -> handle
    stop(handle, ramptime) -> None
    fetch(channels, duration) -> dict[str, Capture]      # Capture: data, start_gps, rate

AWGNDSTransport   # awg + cdsutils + gpstime, lazy-imported in __init__
TwinTransport     # routes to an rtsfreerun mdl, synthetic GPS clock
```

Ported from `backend_rtcds.py`, comments included:

- **The `Thread.isAlive` shim** (`:21-30`). python-awg 3.1.2 calls `Thread.isAlive()` at `awg.py:706`,
  removed in Python 3.9, so `ArbitraryLoop.start()` dies on its first line and *every* injection fails
  before it begins. The site CDS stack runs awg under an older interpreter where `isAlive` still
  exists, so this only appears in a pinned modern conda env. Keep the comment explaining why patching
  site-packages is wrong (an env rebuild discards it).
- **The `started`-flag `finally` stop** (`:118-144`), verbatim.
- **Rate is probed, not configured** (`:70-72`), and printed — a Foton-derived seed model assumes a
  front-end rate and a mismatch silently invalidates the run.
- **Lazy import** inside `AWGNDSTransport.__init__`, never at module scope, so `import system_ident`
  stays clean on a plain scientific stack (`cds.py:3-6` already states this contract). Guarded by a
  test in the style of the other repo's `test/test_cli.py:77-81`.
- **Site environment precondition.** `cdsutils/nds.py` reads the site IFO variable at *import* time and
  raises `NDSError: IFO environment variable not specified` otherwise. Check it before importing, and
  raise with an actionable message. The variable's **value is site configuration, not a constant** —
  it is `C1` at the 40m; that value belongs in the site profile (Component 2), never hardcoded.
  Related trap to document: do **not** source the site workstation rc script — it prepends a legacy
  site CDS python stack to `PYTHONPATH` and `import cdsutils` then dies with `ModuleNotFoundError: No
  module named 'matrix'`; unsetting `PYTHONPATH` recovers.

### 4.2 `CDSBackend` — stage on `inject`, execute on `read`

The other repo's shape is one blocking `measure()`; this repo's is `inject()` then `read()`. The
resolution already exists in-repo: `RTSfreerunBackend.inject` (`rtsfreerun_adapter.py:149-158`) merely
*stages* the drive and `read()` (`:160-212`) executes it.

`from_config(config)` is the constructor of record, because several invariants can only be established
there: the channel maps (§3.1), and — probing once — that the hardware rate is an integer multiple of
`measurement.fs`, that `T_perseg · fs_hw` is an integer, and that `T_perseg · fs` is an integer.
Otherwise the drive is not periodic at `nperseg` after decimation and §2.1's cancellation guarantee
evaporates. Raise `ConfigError`, print the probed rate and its source channels. Zero-risk,
pre-injection, and worth more than any downstream cleverness.

`inject(channel, ts, fs)`:
1. Stop any live injection on that channel first and bump a generation counter — otherwise iteration 2
   of *simultaneous* mode leaks a running excitation with no handle to stop it (`loop.py:164`'s
   `ramp_down` is inside the `if sequential` branch only).
2. Resample **one period** with `sig.resample` and tile; never `resample_poly`, never resample the
   tiled array (§3.5). Assert the length is an exact multiple of the period.
3. Do **not** call `_soft_start_stop` (§2.2).
4. Pre-injection drive check, then the approval gate (§5). Construct but do **not** start the
   `ArbitraryLoop`.

`read(channels, duration)`:
- Nothing staged → the quiet/background read, matching `backend_rtcds.py:109-115`'s `excitation is
  None` early return; this is what `loop.py:145` does for `Pyy`.
- The §2.1 invariants: real readback for X, no alignment, raise on a missing channel.
- Assert `duration` rounds to an integer second (`getdata` takes ints — `total_dur = T_perseg · n_seg`
  at `loop.py:97` is a float), that the returned length is `round(duration · fs_hw)` per channel (NDS
  live reads can come back short or gapped and nothing at `loop.py:214-217` checks), and that it is an
  exact multiple of `nperseg_hw`. Silence here becomes a §2.1/§3.4-class error downstream.
- **Read cache** keyed on `(injection generation, duration)`: fetch the union of every channel the
  campaign needs once per window, serve per-DoF subsets. Transparent to `SysIDLoop`. Today each DoF
  triggers its own `read()` (`loop.py:214`), so *simultaneous* mode — whose entire purpose is to
  measure N DoFs in the time of one — is **slower** than sequential. At physics-sized resolution
  (Q≈50 at 0.67 Hz → `df ≈ 3 mHz`, `T ≈ 256 s`, per `design/resolution.py:15-16`) a 3-DoF × 3-iteration
  campaign is ≈**7.0 h**; the cache brings it to ≈2.5 h. It also collapses the three serial quiet reads
  at `loop.py:144-146`.
- **Settle once.** The other repo sleeps a settle *and* this repo drops `n_transient` periods inside
  the record while scaling `_fisher_time_factor` (`loop.py:110`) accordingly — doing both pays twice
  and makes the Fisher time wrong. Sleep only `ramp_s` plus a margin, keep `n_transient ≥ 1` inside the
  record (it is the adaptive guard against a wrong prior Q), and do **not** add a `settle_duration`
  knob that duplicates it.

`ramp_down(channel, secs)` → `transport.stop(handle, ramptime=secs)`, **idempotent**: `loop.py:164`
calls it per DoF and `Watchdog.abort` (`safety.py:137-138`) calls it again for every channel at
`loop.py:187`, so the last DoF is stopped twice — and the other repo found that calling `stop()` in the
wrong state raises `cannot join thread before it is started` and masks the real error.

`snapshot_state` / `restore_state` — **mandatory**: `loop.py:140` calls them unconditionally and the
base raises (`base.py:66`), so a campaign dies at line 140 without them. `TwinBackend`
(`twin.py:223-230`) and `RTSfreerunBackend` (`rtsfreerun_adapter.py:246-250`) just save/restore their
in-memory drives. Capture only what can actually be restored (which channels have a live excitation,
and their ramptime) and **say so in the docstring**: filter-module switch/gain/offset state is *not*
captured. On hardware `restore_state` is the "hand control back to the damping loops" step
(`safety.py:9-11`), and doing that properly needs `ezca`/`pyepics` — absent from
`pyproject.toml:21-28` — plus operator sign-off on what may be written. That is Component 2. Do not
fake it.

## 5. Safety enforcement

**The gate lives inside `CDSBackend.inject()`** — the only code path from any caller to the actuator,
and the only place holding the actual samples.

Rejected alternatives: a `Watchdog` pre-injection hook or a `loop.py` change is a *convention*, not an
invariant — any script, notebook or test that constructs the backend and calls `inject()` directly
bypasses it, and it makes `SysIDLoop` backend-aware against `base.py:3-5` ("the loop and the safety
handoff use only these methods, so simulation and hardware are truly interchangeable"). A
decorator/mixin has the same bypass unless applied at class level, at which point it *is* the
inject-internal gate with extra indirection.

**Approval is a single-use token**, because staging and actuation are now separate:

- `inject()` prompts — printing channel, hardware rate, duration, **RMS, peak and crest**, and the
  `start_gps` window — then mints `_approval[channel]`.
- `read()` refuses to start any staged injection whose token is missing or consumed; consumes it on
  start.
- Re-reads of an already-running injection do **not** re-prompt: nothing new is actuated. A new
  `inject()` mints a new token. That is Rule 2 exactly.
- The prompt is injectable (`authorizer=`) for testability and **defaults to deny** on `EOFError` /
  non-TTY. Never proceed in batch.

**Absolute amplitude limits, enforced pre-injection.** `actuator_sat` exists but is only read at
`safety.py:105` from `evaluate()`, which runs at `loop.py:215` — *after* inject and read. The budget is
pure power (`design/pintelon.py:91,103`; `excitation.py:170-186`) with **no count ceiling and no peak
bound anywhere**; `configs/rtsfreerun_hsts.yml:39`'s "keep the peak < COIL_DRIVER_LIMIT" is a comment
enforced by nothing.

Measured on a twin-demo-shaped multisine (fs 256, nperseg 1024, band 0.3–8 Hz):

| quantity | value |
|---|---|
| realised `var(drive)` / `px_total` | 1.033 (trapezoid-vs-rectangle endpoints) |
| crest `peak/RMS`, Schroeder phases | **1.90** |
| crest after `_soft_start_stop` | 2.02 |
| `var(tapered)/var(untapered)` | 0.883 |

1.90 is better than the other repo's ~3× broadband figure, but 1.9× of an *unbounded* budget is still
unbounded, and `excitation.py:98-104` already says the peak that binds is at the DAC, not at design
time. Add `safety.max_exc_peak` / `max_exc_rms` to the schema (`config.py:42`) and enforce them in
`inject()` on the exact samples handed to `ArbitraryLoop`. Port the other repo's `limit_peak`
semantics — scale the whole series to preserve the spectrum, report the factor — but **raise
`SafetyAbort` when the required scale-down is large**: a 22× shrink means the design is wrong, not
that it should be quietly scaled. Print the actual RMS and peak before injecting; that is the other
repo's cheapest, highest-value lesson. Implement as an opt-in, default-off
`ChannelBackend._check_drive_limits(ts)` so twin behaviour stays bit-identical.

Note the 12% energy the ramp discards feeds `fisher_matrix` (`loop.py:225-227`) via the *designed*
`Pxx`, not the realised one, so reported uncertainty is optimistic by ~6% in amplitude. Record it;
don't chase it.

**Fail fast on a bad `Pyy`.** `loop.py:143-146` feeds it to `designer.design` → `fisher.dispersion`,
which divides by it. A dead readback or a wrong channel name yields zeros → inf/NaN → "SVD did not
converge" minutes later. `design/pintelon.py:24-30` documents that failure mode for `Pxx`; nothing
guards `Pyy`. Check finite, positive and above a floor **before the first injection**. Add
`--skip-background` and a `measurement.Pyy_from_file` path (the other repo's §3.1): at physics-sized
resolution the unconditional quiet measurement is ≈**1.7 h before anything is injected**.

**Fix the existing CLI gate.** `cli.py:89` calls `_confirm(twin=True)` unconditionally, so the
`"HARDWARE"` label at `cli.py:129` is dead code and a hardware run announces itself as a twin run →
`_confirm(twin=args.twin or args.rtsfreerun)`. And `--yes` (`cli.py:45`) is a campaign-wide carry-over
approval, which Rule 2 forbids → **argparse-level rejection (exit 2)** when neither `--twin` nor
`--rtsfreerun` is given. The CLI prompt is then a pre-flight; the *gate* is the token.

**An out-of-band STOP is mandatory on hardware.** `cli.py:118-123` is the only wiring of
`watchdog.abort("operator STOP")` and it lives in the dashboard — so "the dashboard is optional" is
true for the twin and **false for hardware**, especially given §3.6. Do *not* force `--no-dashboard`:
`cli.py:108-113` already degrades gracefully, and `fastapi`/`uvicorn`/`websockets` are pure-Python,
pip-installable under py3.9 and already declared at `pyproject.toml:31`. Instead require *a* stop path
— dashboard, or the SIGINT handler documented and printed prominently at startup. Never headless with
neither.

## 6. Scope

**In scope (Component 1).** The transport seam and both implementations; `CDSBackend` including the
§2 invariants; the §3 loop fixes; the §5 safety enforcement; `--cds` wiring with a **site-agnostic**
config; mock-transport and twin-transport tests. Exit gate: a full-stack run through `TwinTransport`
against the **existing compiled `x1hsts` model**, scored against the rtsfreerun oracle within the CRB —
the same criterion Stage 1 used (`tests/test_sos_sysid.py`, worst 1.64σ). This deliberately decouples
the work from Stage 2 (`gen_x1sos6dof.py`, not yet written, in the `digital_twin` repo): the transport
is plant-agnostic, so an HSTS validates it as well as an SOS would.

**Out of scope (Component 2).** Any live injection. The site-profile layer (channel naming, the IFO
key, counts↔newtons, DAC/coil limits, OSEM basis, front-end rate — data, not code). Guardian /
lock-state abort. Full filter-module/SDF snapshot and restore. Foton ZPK/SOS export and the provenance
manifest. The operator-answer gate (`notes/40m-sos-campaign-handoff-2026-07.md:66-70`,
`notes/strategic-roadmap-2026-07-draft.md:309-334`).

**Deliberately deferred so it can be generalised.** Component 1 must not hard-code a single 40m
channel name or the `C1` IFO value, so that non-40m hardware needs only a new profile.

**The hardware-only unknowns** — the human-gated set, which no fake transport can settle: whether awg
3.1.2 accepts the untapered integer-period array with `ramptime` start/stop semantics; whether a
channel's AWG slot is released so a second `ArbitraryLoop` on it succeeds; multi-channel common
`start_gps` when `start()` blocks until it (simultaneous mode); whether `_EXC` is NDS-readable at the
site; `getdata` live short/gap behaviour; actual DAC counts against the design budget.

## 7. What is NOT a defect here

Recorded so nobody "fixes" working code. Two of the other repo's scariest findings do **not** apply:

- **§4.2, `normalise_rms` overshooting the requested RMS by `sqrt(Δf_full/Δf_dec)` ≈ 22.6× on
  iterations ≥ 1.** Not latent: this repo normalises the design PSD to a power budget and
  `multisine_from_psd` (`excitation.py:173,185`) realises it consistently — measured realised
  `var/px_total` = **1.033**.
- **§4.3, first-iteration excitation unbanded, with a −17.35-count residual DC offset.** Not latent:
  `excitation.py:170-173` keeps only `k ≥ 1` (DC explicitly dropped) at in-band `freq` bins
  (`loop.py:101-103`); measured `mean(drive)` = −1.4e-17, and 1.9e-4 after the array taper
  (7.4e-5 of RMS) — which disappears entirely once §2.2's transport ramp replaces it.

And one that is much less binding here:

- **§2.4, `--f-max = 0.8·f_nyquist` against the decimation anti-alias rolloff.** The other repo's
  `scipy.signal.decimate` uses `cheby1(8, 0.05, 0.8/q)`, measured **−9.04 dB at 0.9·Nyquist** and
  −22.37 dB at Nyquist. `resample_poly`'s Kaiser FIR is +0.02 dB at 0.8 and −0.67 dB at 0.9. And being
  LTI and applied to both X and Y, it **cancels exactly in `Ybar/Xbar`** — so it costs SNR, not bias.
  A `warnings.warn` above `0.8·(fs/2)` suffices; no `--f-max` flag. **That cancellation is a second
  reason X must never be a synthesised array** (§2.1): synthesise X and the filter no longer cancels,
  giving a systematic −2.4 dB at 0.95·Nyquist.

Also not ported: the other repo's `effective_coherence`, which algebraically ignores its `u_signal`
argument (its §4.0) so denominator read-back noise never enters any reported uncertainty. This repo
derives coherence from `var_b` (`loop.py:462-464`) and does not have that bug.

## 8. Environment

> **CORRECTION (2026-08-03, later the same day) — the paragraph immediately below is WRONG, and the
> environment it rules out has since been built and validated.** It is left in place, not deleted, so
> the mistake is visible and nobody re-derives it. The error was one of scope: it described the
> environment *installed* on the deployment machine (a 2022 `cds-crtools 3.1.2` clone), and generalised
> that to "the channel". The channel says otherwise — CDS publishes a **python 3.11** environment
> (`cds-py311.yaml`, 2026-03-12) carrying `cds-crtools`/`foton`/`python-foton`/`libawg`/`python-awg`/
> `dtt-*` **4.1.4**, and every one of those packages is on conda-forge with linux-64
> `py311`/`py312`/`py313` builds. See **§8a** below for what was actually measured, and issue #22.

The CDS control packages (`foton`, `python-foton`, `python-awg`, `libawg`, `dtt-*` 3.1.2, plus
`python-nds2-client`) are **py3.9-only builds** in the deployment machine's channel — which is why the
sibling repo had to pin `python=3.9` (its commit `86a4f7a`) and drop the `anaconda` metapackage. So a
newer environment is **not available for the hardware path** until the site rolls out a newer CDS. This
was the option to investigate; the design therefore targets the pinned baseline, and Step 0 of the
plan probes it empirically on the box rather than inferring.

The decisive measurement: **the CDS-relevant half of this repo already passes on that baseline.** In a
py3.9.13 / numpy 1.22.4 / scipy 1.8.1 / control 0.9.2 environment, `test_step5_safety`,
`test_step7_loop`, `test_step8_cli`, `test_periodic_measurement`, `test_rtsfreerun_backend`,
`test_excitation`, `test_step4_twin`, `test_step6_estimator`, `test_step12_ml_estimator` and
`test_resolution` give **61 passed / 1 skipped**. The full suite does not (250 passed / 35 failed / 12
skipped / 4 errors), but every failure is in the DARM / MIMO / SOS / arcade / playground half, which
this port does not touch. **Compat work is therefore not on the critical path**, and the deployment
gate is that named subset, not "the suite passes".

Four root causes, and two need no shim because a third spelling works on both stacks:

| break | resolution | sites |
|---|---|---|
| `FrequencyResponseData.frdata` (absent before control 0.10.2; it was `.fresp`) | **the one genuine shim** — a `_frd(sys, w)` helper trying `frdata` then `fresp` (prefer `frdata`; `.fresp` warns on 0.10) | `darm_actuation.py:144,208`, `closed_loop_id.py:37` + 4 test sites → 7 calls collapse to 1 |
| `control.tf2ss(StateSpace)` rejected on 0.9.2, accepted on 0.10 | `control.ss(x)` — works on both | `mimo_loop.py:31`, `mimo_plant.py:40,72`, `sos_closed.py:70`, `darm_actuation.py:124`, `tests/test_mimo.py:236` — 11 failures |
| `np.trapezoid` (numpy ≥ 2) | `from scipy.integrate import trapezoid` (scipy ≥ 1.6) — already this repo's `src/` convention (`design/pintelon.py:18`) | zero `src/` sites; 5 test/docs/experiment sites |
| `plotly` absent | environment, not version | 3 collection errors |

Plus the order-dependent circular import at `backends/darm_adapter.py:14` ← `darm.py:506` (move the
bottom import into the function or a `TYPE_CHECKING` block). It reproduces identically on py3.12 —
pre-existing and version-independent.

**Do not raise the floor** (impossible — CDS 3.1.2 is py3.9-only) and **do not build a general
`_compat.py`**: one three-line helper plus two mechanical renames. Add a py3.9 CI leg — CI runs 3.12
only (`.github/workflows/ci.yml:26`) and executes exactly one test file, so a 3.9 regression is
invisible until it fails on deployment.

### 8a. Environment — what was actually built and measured (2026-08-03)

The deployment environment is **`sysid_deploy`: python 3.11 with CDS 4.1.4.** It is a *lean* env — the
CDS control packages this project uses plus its own dependencies — with every version pinned to the
CDS-published `cds-py311.yaml` (2026-03-12), which is what the help desk confirms as supported.
Declarative spec: `environment_deploy.yml`; exact export: `environment_deploy_lock.yml`.

Solved and installed versions:

| | |
|---|---|
| python | 3.11.15 |
| numpy / scipy | 1.26.4 / 1.13.1 |
| control / slycot | 0.10.2 / 0.6.1 |
| matplotlib-base / pyyaml / plotly | 3.10.8 / 6.0.3 / 6.6.0 |
| cds-crtools / foton / python-foton / libawg / python-awg / dtt-awgstream | 4.1.4 (`py311` builds) |
| cdsutils / gpstime | 1.7.0 / 0.10.0 |
| nds2-client / python-nds2-client | 0.16.8 / 0.16.12 |

268 packages, ~740 MB. Nothing dragged numpy to 2.x — the specific hazard, since the compiled
extensions are built against 1.26.

**Measured, read-only, on the deployment machine. No injection was performed.**

1. `awg`, `foton`, `gpstime`, `nds2` and `cdsutils` all import.
2. **Deployment-gate subset: 61 passed / 1 skipped** — *identical* to the py3.9 baseline above. The
   gate is unchanged by the move.
3. Full suite: **290 passed / 8 failed / 17 skipped**, against py3.9's 250 passed / 35 failed / 12
   skipped / 4 errors. All 8 failures are `np.trapezoid` in `test_arcade` / `test_playground`.
4. `import system_ident` pulls in neither `awg` nor `cdsutils` — the lazy-import contract holds.
5. `cdsutils.getdata(<readback>, 2)` returned live data: exactly `2 × 256` samples, all finite, with
   the buffer exposing `.data` / `.sample_rate` / `.start_time` — which is what Stage A's `FakeGetdata`
   is specified to mimic, so the fake matches the real object.

**The compat table above is now mostly moot**, because control 0.10.2 ships in this environment:

| break | status at py3.11 / control 0.10.2 |
|---|---|
| `frdata` / `fresp` | **no shim needed** — `.frdata` exists. Do not write `_frd()`. |
| `control.tf2ss(StateSpace)` | **accepted** — the 6 renames and 11 failures are gone |
| `plotly` absent | **present** (6.6.0) — the 3 collection errors are gone |
| `np.trapezoid` | **still broken** (numpy 1.26 < 2.0) — the 8 remaining failures, 5 sites |
| `darm_adapter` circular import | unchanged; version-independent |

So issue #23 reduces to the 5 `np.trapezoid` sites plus the circular import, and its extra CI leg
should be **py3.11** (matching deployment), not py3.9.

`awg` 4.1.4 no longer calls `Thread.isAlive`, so §4.1's shim is unnecessary here — keep it
`hasattr`-guarded rather than deleted, because fallback rungs 2 and 4 below still need it.

### 8b. awg 4.1.4 **does** drive the front ends — operator-run, 2026-08-04

The open question of §8a — whether an `awg` **4.1.4** client can inject into front ends running
**advLigoRTS branch-3.4** (2017), where the only previously proven client was 3.1.2 — has been
**answered yes**, by a **human operator** running two excitations by hand under live supervision on a
single suspension ASC-pitch excitation channel. Concrete channel and file paths are in the untracked
local access note; this repo stays site-agnostic.

Both records were captured from the `_EXC` channel itself at **16384 Hz** and verified numerically
against what was commanded:

| commanded | measured |
|---|---|
| sine, 1 Hz, 20 pp, 3 s ramp | pp **20.000000** (whole-period windows, zero spread); LSQ fit 1.00000000 Hz, amplitude 10.000000, **residual rms 1.96e-07** (0.0000 % of amplitude), no odd harmonics; ramp up/down **3.017 / 3.002 s** |
| square, 2 Hz, 2√3 pp, √5 s ramp | pp **3.464102**; levels exactly **±√3**, symmetric to 0.00e+00; h3/h5/h7 = 0.33333/0.20000/0.14286 (ideal square); period 0.500000 s over 49 transitions; ramp up/down **2.23607 / 2.23607 s** vs √5 = 2.2360680 |

Both had leading and trailing zeros (GUI-captured windows) and ran well under 25 s. The sine's +0.56 %
ramp figure is Hilbert-envelope ripple at 2f in the *estimator*, not a deviation in the data — the
square wave, whose envelope is exactly `|x|`, recovers √5 to five decimals in both directions.

**Three things this settles, beyond the client-version question:**

1. **The AWG's own `ramptime` is a clean *linear* gain ramp**, measured to five decimals and symmetric
   on the way down.
   > **CORRECTED 2026-08-04.** This bullet originally continued: *"…so `_soft_start_stop`'s Tukey taper
   > is not merely unnecessary — the correct alternative is confirmed to exist."* **That inverted the
   > conclusion.** What the measurement establishes is that the transport ramp is **linear**, and
   > `_soft_start_stop` is a **cosine** taper — which is *gentler on the actuator and preferred*. So the
   > linearity is precisely the reason the transport ramp is **not** an equivalent substitute. The
   > measurement is sound; the inference was not. See **§2.3** for the resolution: a cosine envelope
   > baked into a one-shot `ArbitraryStream` array, with `ArbitraryLoop` retained as a supported peer
   > mode. What this bullet *does* establish is narrower and still useful: the transport gain-ramp
   > mechanism works on real hardware and is accurately characterised, which is what makes `loop` a
   > usable peer mode and `set_gain`-based `ramp_down` trustworthy.
2. **`_EXC` is NDS-readable at the site, at the model rate.** That was an explicit hardware-only
   unknown (see the bring-up note's open questions). It matters because §2.1 requires `read()` to use a
   real readback for X rather than synthesising it, and this is the channel that supplies it.
3. **Amplitude and shape survive the round trip exactly** — pp, levels and harmonic structure come back
   at their commanded values with float-level residuals, so there is no hidden gain, offset or filtering
   between the commanded waveform and the `_EXC` record on this path.

**What this does NOT establish — do not over-read it:**

- These were **two simple built-in waveforms on one channel**. The backend's actual path is
  `awg.ArbitraryLoop` with a tiled, integer-period array (§4.2), which is a *different* awg API. Unless
  the operator used `ArbitraryLoop`, the remaining §9 unknowns stand: whether awg accepts the untapered
  integer-period array with `ramptime` start/stop, whether a channel's AWG slot is released so a second
  `ArbitraryLoop` succeeds, live `getdata` short/gap behaviour, and actual DAC counts vs the design
  budget.
- **Nothing here is standing authorization.** Per §1 Rule 2, every future injection still needs its own
  separate operator approval; two successful injections authorize a third exactly as much as zero would.
- Simultaneous-mode start semantics (§9.3) are untouched by this.

Recorded so the client-version risk is not re-litigated, and so the ramp evidence is not lost.

Fallback ladder if awg 4.1.4 will not drive the front ends. Each rung is a fresh
operator-supervised bring-up:

```
1. python 3.11 + CDS 4.1.4   <- built and read-validated (this section)
2. python 3.10 + CDS 3.1.2   <- isolates the variable: proven awg, newer python.
                                3.10 is the HIGHEST python with a python-awg 3.1.2 build.
3. python 3.10 + CDS 4.1.4   <- site cds-py310.yaml; note it is ALSO 4.1.4, so it only helps
                                if the fault is python-3.11-specific rather than awg-generation
4. python 3.9  + CDS 3.1.2   <- the pre-existing installed env; the original fallback of record
```

Practical notes for anyone repeating this: the deployment machine's conda is 22.11.1 with **no mamba**,
and its classic solver could not solve this in 30 minutes (it falls through `current_repodata.json`
because the pins predate latest-only metadata, then grinds on full repodata); a static `micromamba` in
a user-owned directory does it in seconds, and base conda need not be touched. `cds-crtools` pulls
**ROOT** (~255 MB, a third of the download) — droppable for a slimmer env.

## 9. Open questions

1. **Does the trunk-based rule (`CLAUDE.md:42`) still hold for parallel development** — several
   people, each with their own agents? This work uses a long-lived branch, which the rule as written
   forbids; the rule is likely aimed at a single-agent workflow. Flagged, not changed.
2. **The Phase-1 gate (`CLAUDE.md:47-49`)** has been lifted for hardware *transport* work as of
   2026-08-03, with live injection still human-gated. Recorded rather than deleted, so a future reader
   does not revert the port on sight.
3. **Simultaneous-mode start semantics** cannot be resolved off-hardware: `inj.start(ramptime>0,
   wait=False)` blocks until `start_gps`, so starting three loops in a Python `for` loop with a common
   `start_gps` leaves loops 2 and 3 starting late, with their `start_gps` in the past. §2.1 means each
   individual FRF is still unbiased (each X is a readback), but the three drives are then not the
   synchronous simultaneous excitation the mode assumes, and the loop cannot tell. Candidate fix —
   start all with `ramptime=0` at a common `start_gps` and ramp separately — is unverifiable without
   the real `awg`.
