# Driving system_ident against the digital twin (RTSfreerun) — design + roadmap

How `system_ident`'s Pintelon–Schoukens pipeline runs on the LIGO digital twin
(`GIT/digital_twin/`) with realistic noise. Companion to [digital-twin.md](digital-twin.md).
Working notes — committed, but not part of the package or the published docs.

## The idea
`system_ident` already abstracts the instrument behind `backends/base.ChannelBackend`
(`inject(channel, ts, fs)` / `read(channels, duration)` / `ramp_down` / `snapshot_state` /
`restore_state`). The twin's `mdl` exposes `run(cycles, excitations, excitation_data)` +
`fetch_later`. So we add **one adapter backend** — `RTSfreerunBackend` — that drives a built
`mdl`, and the existing `SysIDLoop` / estimator / designer / safety stack runs unchanged. We
identify the twin's drive→sensor plant *under the twin's own realistic seismic + readout noise*,
which is exactly the "CDS-aware" demonstration.

## Adapter design (`src/system_ident/backends/rtsfreerun_adapter.py`)
`RTSfreerunBackend(ChannelBackend)` — mirrors `TwinBackend`/`CDSBackend`:
- **Lazy-imports** the model module by name (`importlib.import_module(model)`) so importing the
  package never needs the twin (like `CDSBackend`'s lazy CDS imports). Construct with a model
  name (or an injected `mdl`/factory for tests), channel maps, a noise spec, warmup seconds.
- `fs_model = mdl.sample_rate`.
- `inject(channel, ts, fs)` — resample `ts` to `fs_model`; stash per `_EXC` channel.
- `read(channels, duration)`:
  - assemble `excitation_data` = stashed sysID drives **+** fresh `twin.noise` realizations on
    their channels (seismic/readout — realistic background);
  - `mdl.run(n_warm)` then `fetch_later(0, duration, probes)` + `mdl.run(n_record)`; collect
    `buf.data`;
  - return `{ch: resample(probe_or_stashed_drive, fs)}`. For a requested **excitation** channel,
    return the stashed injected drive (it is not a probe) — satisfies the watchdog's drive-peak
    check; the FRF input X comes from a **drive-monitor probe** (`channels.drive`, the closed-loop
    work we just shipped).
- `ramp_down`, `snapshot_state`/`restore_state` — stashed drives (+ any `mdl.write` gains).
- `from_config(config, **kwargs)` — build the maps + model/noise spec.

**Channel mapping** (config `channels`): sysID `excitation` → model `_EXC` port; `readback` →
sensor probe `_OUT`; `drive` (FRF input X) → after-controller drive probe `_OUT`. Noise channels
(e.g. `ISI_RESIDUAL_EXC`, `READOUT_NOISE_EXC`) live in the backend's noise spec, not the sysID maps.

**Rates**: keep the sysID `measurement.fs` = `mdl.sample_rate` (16384) in the demo to avoid
resampling, or let the adapter resample. The multisine `nperseg` must still satisfy `T_fft > 3/f_min`.

## Local testing without the twin (`tests/test_rtsfreerun_backend.py`)
`MockRTSModel` — a ~40-line fake implementing `sample_rate`, `run`, `fetch_later`, `write`,
filtering injected excitations through a known SISO (or closed-loop) plant via `scipy.signal`.
Lets the adapter + config + a **plant-recovery** test run in the `sysid` env on macOS. A real-twin
smoke test is guarded by `importlib.util.find_spec("x1hsts")` → `pytest.skip`.

## Config + CLI (`config.py`, `cli.py`, `configs/rtsfreerun_hsts.yml`)
- `RunConfig.build_rtsfreerun_backend(seed)` parses a `rtsfreerun:` section (model, `model_rate`,
  channel maps already in `channels`, a `noise:` list reusing twin presets, `warmup_s`).
- `cli.py`: add `--rtsfreerun`; `_run` selects `build_rtsfreerun_backend`; downstream
  `SysIDLoop.run(config.raw, priors)` is unchanged.
- Demo runs in an env with **both** `system_ident` and a built twin model
  (`pip install -e <system_ident>` into the `twin` env, or build the model into `sysid`).

## Iteration roadmap (the "many iterations")
> Now deepened into runnable steps in [roadmap.md](roadmap.md) **Track A** (the active thread). The
> four iterations below are the detailed source; the master roadmap adds prereqs, verify commands,
> and a per-iteration results log.

Track status here across sessions (like digital_twin's `.llm/roadmap.md`).
1. **Wiring smoke** — `x1sysexample`: inject a multisine, `read` the response; confirm round-trip
   and rate handling. (mock-tested locally; real on the twin box.)
2. **Open-loop SISO suspension** — `x1hsts` one DoF: inject at the actuator (`COIL_DRIVER_EXC`),
   read `READOUT_NOISE_OUT`, with `seismic` + `bosem` noise on; recover the drive→sensor plant;
   compare to the analytic `HSTS_DRV_TF`.
3. **Closed-loop** — `x1hstsdamped` / `hsts_damped.yaml`: engage the damping bank; use the
   first-class closed-loop mode (drive-monitor channel) to recover the **open-loop** plant.
4. **MIMO / coupled** — `x1hsts6dof`: multi-DoF + cross-coupling campaign; compare to the analytic
   suspension matrix; exercise the coupled-plant machinery.

## Status (skeleton landed 2026-06-15)
- [x] Phase 0 wiki · [x] adapter (`backends/rtsfreerun_adapter.py`) · [x] mock + 8 tests
  (`tests/test_rtsfreerun_backend.py`, recovery via FRF + full SysIDLoop + decimation) ·
  [x] config/CLI (`config.build_rtsfreerun_backend`, `cli --rtsfreerun`) ·
  [x] demo config (`configs/rtsfreerun_hsts.yml`, x1hsts — the skeleton
  `rtsfreerun_demo.yml` was superseded by it and no longer exists).
- Verified here against the mock (109 passed). Real-twin demo run is **next, on the twin box**.
- Iteration 1 (wiring smoke on x1hsts) is the immediate next step once run on the twin.
- Each roadmap demo above is a separate refinement pass; keep this list current.
