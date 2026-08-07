# The LIGO digital twin (`GIT/digital_twin/`) — summary for system_ident

Working notes on the companion digital-twin repo, so `system_ident` can drive its sysID
pipeline against realistic CDS twin models. Source: a sibling checkout at
`<workspace>/digital_twin/` (the `twin/` module). Read-only reference — nothing here is
built from this repo.

## What it is
A **teaching co-simulation** of a LIGO subsystem: a CDS front-end `.mdl` (Simulink-style)
is compiled by **advligorts RCG** into an importable Python module via **`rtsfreerun`**, then
driven from Python with synthetic noise — "same channel names, same filter modules, same foton
conventions, running on a laptop with no hardware." The *same C numerics a real front-end runs*,
not a Python reimplementation.

## Repo layout (top level)
- `rtsfreerun/` — build harness: `.mdl` → importable Python module (clones advligorts, runs RCG,
  compiles a pybind11 extension). Build is Linux-oriented (`model=… pip install ./rtsfreerun`).
- `rtsfreerun-models/` — model sources: `x1arm`, `x1imc`, `x1drmi` (cavities); `x1hsts`,
  `x1hsts6dof`, `x1hstsdamped`, `x1quad`, `x1quad6dof`, `x1quaddamped` (suspensions);
  `x1hamisi` (ISI); `x1sysexample`, `x1sim` (examples).
- `aligo-suspension-models/` — aLIGO suspension state-space `.mat` plants (BS/HLTS/HSTS/QUAD).
- `twin/` — **the orchestrator + scenarios + noise + analysis** (the end-user layer).

## The `mdl` API (what `rtsfreerun` exposes) — the integration surface
`pkg = importlib.import_module("x1hsts"); mdl = pkg.x1hsts()` gives an object with:
- `mdl.sample_rate` → float (e.g. **16384 Hz** for LSC/suspension composites; trust it over YAML).
- `mdl.run(cycles=N, excitations=[ch,…] | None, excitation_data=(N,k) ndarray | None)` — advance
  N cycles (1 cycle = 1 sample); `excitation_data[:, j]` feeds `excitations[j]` each cycle.
- `fetch = mdl.fetch_later(t0, t1, names)`; call `mdl.run(...)`; then `buffers = fetch()`;
  each `buf.data` is the full-rate probe array. (`mdl.run(testpoints=…)` returns DAQ-downsampled
  data — use `fetch_later` for full rate.)
- `mdl.write(channel, value)` — set a gain/limit/scalar (e.g. `COIL_DRIVER_LIMIT`).
- `mdl.fm_set_zpk(fm, section, z, p, k, plane)`, `mdl.fm_set_sos(...)`, `mdl.fm_set_switches(fm, **sw)`
  — configure a `cdsFilt` filter-module section live.

**Channel naming** — every `cdsFilt` block has ports: `<NAME>_EXC` (inject), `<NAME>_OUT`
(probe/testpoint), `<NAME>_IN`/`_IN1`, `<NAME>_GAIN`, `<NAME>_LIMIT`. Sum blocks have **no**
testpoint. The `.mdl` is the source of truth for routing; the orchestrator does not route signals.

## Canonical drive pattern (from `twin/src/twin/orchestrator.py::run_scenario`)
```
mdl = importlib.import_module(model).<model>()
_apply_init(mdl, scenario.init)            # fm_set_zpk/sos/switches + write from YAML
fs = mdl.sample_rate
n_warm, n_rec = WARMUP_S*fs, duration*fs   # WARMUP_S = 5.0 (lets ~0.01 Hz LSC pole settle)
chans, exc_data = build_excitations(...)   # noise columns, one realization per channel
mdl.run(n_warm, excitations=chans, excitation_data=exc_data[:n_warm])     # warmup (discard)
fetch = mdl.fetch_later(0, duration, probe_names)
mdl.run(n_rec, excitations=chans, excitation_data=exc_data[n_warm:n_warm+n_rec])
buffers = fetch()                          # buf.data per probe; optional ADC quantize
```
Gotchas (from digital_twin `CLAUDE.md`): rtsfreerun `plane='f'` is **opposite-signed** from foton
(orchestrator sends `plane='s'` with negated roots); PyYAML sci-notation needs an explicit
exponent sign; independent excitation realization per injection channel.

## Realistic noise (`twin/src/twin/noise.py`) — IFFT-of-colored-Gaussian → ASD-matched series
- **Seismic** (`kind: seismic`): Lorentzian microseism peak + 1/f² rolloff + NLNM floor;
  presets `ligo-india` (3e-6 m/√Hz @ 0.15 Hz), `lho-like`, `llo-like`. `generate_ground_motion(...)`
  for 6 DoF. Injected at plant `_EXC` ports (e.g. `ISI_RESIDUAL_EXC`).
- **Readout** (flat + 1/f knee): `bosem` (~1e-10 m/√Hz), `osem` (~3e-10). Injected at a readout
  `_EXC` (e.g. `READOUT_NOISE_EXC`).
- **Shot noise**: `wfs_shot` (rad/√Hz angular), `pd_shot` (W/√Hz). `generate_from_asd(asd_fn, dur, fs, rng)`.
- **ADC quantization**: per-probe `quantize: {bits, range, dither}` applied post-fetch.

## A real suspension scenario (`twin/scenarios/hsts.yaml`, model `x1hsts`, 16384 Hz)
- noise: `seismic ligo-india` → `ISI_RESIDUAL_EXC`; `bosem` → `READOUT_NOISE_EXC`.
- plant chain (init'd via foton ZPK): `HSTS_GND_TF_A/B` (ground→sus), `HSTS_DRV_TF_A/B`
  (**drive→sus, the TF we'd identify**), `COIL_DRIVER` (LIMIT 30000), `READOUT_NOISE`.
- probes: `READOUT_NOISE_OUT` (sensor, 16-bit quantized), `HSTS_DRV_TF_B_OUT`, `HSTS_GND_TF_B_OUT`.
- For sysID: inject a multisine at an actuator `_EXC` (e.g. `COIL_DRIVER_EXC`), read `READOUT_NOISE_OUT`;
  recover the drive→sensor plant. Closed-loop variant: `hsts_damped.yaml`.

## Other useful modules in `twin/src/twin/`
- `orchestrator.py` — `run_scenario`, `_build_excitations`, `_apply_init` (the drive loop above).
- `noise.py` — the noise generators + presets.
- `validate.py` — `design_multisine` (Schroeder, bin-aligned) + `measure_frf(scenario, exc, probe, …)`:
  **already a Pintelon–Schoukens DFT-per-period FRF** with per-line σ. (system_ident does the richer
  optimal-excitation / ML / CRB / closed-loop version — we drive the model, not this.)
- `plant_loader.py` / `foton_loader.py` — aLIGO `.mat` → ZPK/SOS; load real H1/L1 foton banks.
- `config.py` — scenario schema (`Scenario`, `NoiseSpec`, `ProbeSpec`).

## Install / run (the twin's own env; Linux build)
`conda env create -f environment.yml && conda activate twin` (numpy/scipy/control + RCG toolchain:
cmake/make/compilers/pybind11/perl-json). `bash scripts/setup_env_perl.sh`. Build a model:
`PATH=$CONDA_PREFIX/bin:$PATH RCG_LIB_PATH=$PWD/rtsfreerun-models/x1hsts model=x1hsts pip install ./rtsfreerun`.
On the twin box (the lab dev machine, macOS/arm64) there **are** `twin` + `rtsfreerun-dev`
envs. But for `system_ident` we **don't** use them — we build the RTS model into this repo's `sysid`
env instead (separate repos; see README §"Run against the RTSfreerun digital twin" and the
`[[rtsfreerun-env-strategy]]` memory).

## Agent-doc convention (mirror it)
digital_twin uses an `.llm/` directory (overview / roadmap / decisions / conventions /
session-log) + a per-module `CLAUDE.md` + `ARCHITECTURE.md`/`ROADMAP.md`. We mirror the
`.llm/` part here, except that ours is committed. See [rtsfreerun-integration.md](rtsfreerun-integration.md) for the plan.
