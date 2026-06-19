[![CI](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml/badge.svg)](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-2D9D6B?logo=quarto&logoColor=white)](https://caltechexperimentalgravity.github.io/system_ident/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-1F8AC0)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-1F8AC0?logo=python&logoColor=white)](https://www.python.org/)

# system_ident

**📖 Documentation: [caltechexperimentalgravity.github.io/system_ident](https://caltechexperimentalgravity.github.io/system_ident/)**

Real-time, optimal-excitation system identification for LIGO suspensions
(3-DoF: POS / Pitch / Yaw), built as a single Pintelon–Schoukens (P&S) pipeline,
with a digital-twin backend and a live operational dashboard.

`system_ident` runs iterative, optimal-excitation identification of a resonant
suspension plant as one clean P&S loop: an optimally designed **periodic
multisine** is injected, the plant is measured by a **leakage-free,
reference-based synchronous-DFT** frequency-response estimate carrying a per-bin
noise covariance, and a Gaussian **maximum-likelihood** fit refines the model
toward the Cramér–Rao bound. A digital-twin backend and a (stubbed) CDS-hardware
backend share one channel API, so simulation and hardware are interchangeable; a
physical-safety watchdog performs a safe-state handoff on any stop; and an
optional websocket dashboard renders the measurement as it converges.

## Documentation

The full documentation site is rendered from this repo and published to GitHub Pages:
**<https://caltechexperimentalgravity.github.io/system_ident/>**.

- **[Worked examples](https://caltechexperimentalgravity.github.io/system_ident/examples/)** —
  end-to-end P&S campaigns from a single resonance to a coupled 2×2 MIMO suspension, plus
  the [compiled-LIGO-twin closed-loop demo](https://caltechexperimentalgravity.github.io/system_ident/examples/07-rtsfreerun-twin.html)
  (open- and closed-loop 6-DOF identification on the real CDS model).
- **[Tutorial](https://caltechexperimentalgravity.github.io/system_ident/tutorial/overview.html)** —
  the method end to end: the [model](https://caltechexperimentalgravity.github.io/system_ident/tutorial/model.html),
  [Fisher information & optimal excitation](https://caltechexperimentalgravity.github.io/system_ident/tutorial/fisher.html),
  [closing the loop](https://caltechexperimentalgravity.github.io/system_ident/tutorial/closing-the-loop.html),
  and [safety & ops](https://caltechexperimentalgravity.github.io/system_ident/tutorial/safety-and-ops.html).
- **[API reference](https://caltechexperimentalgravity.github.io/system_ident/reference/)** —
  the package surface, generated from docstrings.

## Architecture

The loop is closed: each pass measures the plant, refits the model, accumulates
Fisher information, and — until the fractional parameter uncertainty drops below
target — feeds the refined model back into the optimal designer so the next
excitation concentrates power where it buys the most information.

```mermaid
flowchart TD
    P["Prior model<br/>(TFModel / ResonatorModel)"]
    D["Optimal periodic-multisine design<br/>PintelonSchoukensDesigner"]
    B["Backend inject + read<br/>TwinBackend now / CDSBackend stub"]
    F["Leakage-free reference-based FRF<br/>+ per-bin noise covariance<br/>SysIDLoop._estimate_tf_periodic"]
    M["Maximum-likelihood fit<br/>GMLEstimator / ml_fit"]
    C["Accumulated-Fisher CRB<br/>fisher_matrix"]
    K{"Fractional uncertainty<br/>below target?"}
    DONE(["DONE: safe-state handoff"])

    P --> D --> B --> F --> M --> C --> K
    K -->|no| D
    K -->|yes| DONE

    W["Watchdog<br/>ramp-down + safe-state handoff"]
    DASH["Live dashboard<br/>Plotly + websocket STOP"]
    W -.->|monitors, aborts| B
    DASH -.->|snapshots, STOP| M
```

The first pass designs a *prior-robust* excitation from the prior model and its
(large) error bars — power spread over the plausible resonance band `f0*(1±u)`
so a far prior still covers the true resonance; later passes use the now-trusted
model for point-optimal excitation. Because every pass is an independent
measurement of the same LTI system on a fixed frequency grid, passes are combined
by inverse-variance weighting per bin (`SysIDLoop._accumulate`) and the model is
refit on the accumulated estimate — keeping the first pass's band coverage while
folding in each optimal pass's resonance-sharpening information.

## Codemap

| Module / subpackage | Responsibility |
| --- | --- |
| `model.TFModel` | Canonical SISO `H(s) = num/den` model — the shared parameterisation (`eval`, `jacobian`, `params`, `with_params`) every estimator, designer, and Fisher routine speaks. |
| `resonator.ResonatorModel` | Physical product-of-resonances model in `(f0, Q, gain)`; its `dH/df0` gradient relocates a peak away from the prior. `to_tf()` converts to `TFModel`. |
| `fisher` | Fisher information matrix, parameter covariance, and the P&S **dispersion** function — the engine behind both optimal design and the convergence/DONE criterion. |
| `excitation.multisine_from_psd` | Builds the injectable **periodic Schroeder-phase multisine** realising a designed PSD (leakage-free, low crest factor). `timeseries_from_asd` is the legacy coloured-noise drive. |
| `design/pintelon.PintelonSchoukensDesigner` | The optimal input designer: dispersion-function fixed-point iteration that reallocates drive power to the most informative bins under a fixed budget. |
| `estimators/gml.GMLEstimator` | The **only** estimator (`gml` / `ml`): a Sanathanan–Koerner `invfreqs` linearisation for a start, then exact-objective ML polish. |
| `estimators/bayesian` | Model-agnostic ML primitives: `ml_fit` (Gauss-Newton + Levenberg–Marquardt to the CRB) and `gn_normal_equations`. |
| `estimators/invfreqs` | Weighted inverse-frequency LS fit — survives only as the SK linearisation step inside `GMLEstimator` (not a standalone estimator). |
| `loop.SysIDLoop` | The orchestration loop: design → inject/read → safety-check → leakage-free FRF (`_estimate_tf_periodic`) → ML fit → inverse-variance accumulate → accumulated-Fisher convergence → repeat; emits dashboard snapshots. |
| `backends/base.ChannelBackend` | The inject / read / ramp-down + snapshot/restore interface that makes twin and hardware interchangeable. |
| `backends/twin.TwinBackend` | Digital twin: filters the drive through the discretised plant plus sensor/disturbance noise. Supports **MIMO cross-coupling**, **closed-loop controllers**, transport delay, and actuator saturation. |
| `backends/cds.CDSBackend` | Real-hardware backend (awg inject + nds2 readback) — **stub** raising `NotImplementedError`; lazy-imports the CDS libraries. |
| `plant.SuspensionPlant` | A named set of per-DoF `TFModel`s + sample rate, built from a `(f0, Q)` resonance spec; what the twin drives. |
| `safety.Watchdog` | Physical-limit watchdog (actuator saturation, output-RMS ceiling) owning the one shared, idempotent ramp-down + safe-state handoff. |
| `config.RunConfig` | Loads/validates the run YAML and builds the concrete plant, backend, priors, estimator, designer, and watchdog the loop needs. |
| `cli` | The `system_ident` CLI — YAML base + flag overrides + a confirm-before-inject guard; the primary way to drive a run. |
| `dashboard` | Dependency-free pub/sub `SnapshotHub` plus a lazily-built FastAPI/websocket server serving the live Plotly UI with the STOP button. |

## Documentation

Full docs (pedagogy, API reference, and executed worked examples) are built with
Quarto and published by CI to GitHub Pages. Build them locally with:

```bash
pip install -e ".[docs]"
(cd docs && quartodoc build) && quarto render docs   # output in docs/_site
```

## Install

```bash
pip install -e .          # core (twin path)
pip install -e .[dev]     # + pytest
pip install -e .[dashboard]   # + fastapi/uvicorn/websockets for the live UI
```

Core deps: numpy, scipy, control, pyyaml, matplotlib.

## Test

```bash
pytest
```

A handful of tests validate the ported math **bit-for-bit** against the original
`sysIDlib` engine, used as an oracle. The single reference file
`sys_id_dev/sysIDlib.py` is included so they run out of the box; if you remove
it, those tests `skip` (the rest of the suite is fully standalone).

## Run the digital-twin demo

```bash
system_ident run src/system_ident/configs/twin_demo.yml --twin --yes
# -> DONE (target reached), per-DoF fractional uncertainty below target
```

With the dashboard extra installed, drop `--yes` and a live Plotly view
(transfer function, coherence, designed excitation) is served locally with a
STOP button that triggers the safe-state handoff.

## Run against the RTSfreerun digital twin (compiled CDS models)

The `--rtsfreerun` path drives a **compiled** advligorts front-end model (built
from a `.mdl` by [`rtsfreerun`](https://controlsystems.docs.ligo.org/rtsfreerun))
instead of the pure-Python `TwinBackend`, so the identification runs against the
*same C numerics a real front-end runs*. This needs the same C numerics
**importable in the same interpreter as `system_ident`** — i.e. the RTSfreerun
model package installed into this repo's `sysid` env.

> `system_ident` and `rtsfreerun`/`digital_twin` are **separate repos**. We do
> **not** install `system_ident` into the twin env; instead we install the
> compiled RTS model(s) into the `sysid` env. (`rtsfreerun` only supports one
> model per Python process — install whichever model a demo needs.)

**Required, one-time per model.** From the `digital_twin` checkout (the repo that
contains `rtsfreerun/` + `rtsfreerun-models/`), with the `sysid` env active:

```bash
conda activate sysid

# 1. RTS build toolchain into sysid (one-time; needs a C/C++ compiler too)
conda install -c conda-forge make cmake spdlog rapidjson pybind11

# 2. Build + install one model (e.g. x1hsts) into sysid.
#    RCG_LIB_PATH must point at that model's source dir, and conda's bin must be
#    first on PATH so the conda cmake/make are used.
cd <digital_twin>/rtsfreerun
PATH=$CONDA_PREFIX/bin:$PATH \
  RCG_LIB_PATH=<digital_twin>/rtsfreerun-models/x1hsts \
  model=x1hsts pip install .

# 3. Verify it imports in the same interpreter as system_ident
python -c "import system_ident, x1hsts; print('sample_rate', x1hsts.x1hsts().sample_rate)"
# -> sample_rate 16384
```

Repeat step 2 with a different `model=`/`RCG_LIB_PATH` for other models
(`x1hstsdamped`, `x1hsts6dof`, …). Then run the HSTS demo — the proven P&S path
(periodic multisine → leakage-free FRF → ML fit) identifying the compiled
`x1hsts` drive→sensor suspension plant under the twin's own seismic + readout
noise, scored against an analytic oracle (`backends/rtsfreerun_oracle`):

```bash
python experiments/rtsfreerun/run_hsts.py     # prints A1+A2 recovery, writes a Bode overlay
```

The run config is `src/system_ident/configs/rtsfreerun_hsts.yml` (channels,
band, noise, the scenario that loads the plant, and the corrected 5-mode prior).
Recovery is gated by `tests/test_rtsfreerun_real_model.py` (A1 wiring/oracle, A2
SISO recovery), which `skip`s when no model is installed — so the rest of the
suite is unaffected on machines without the twin.

**Closed-loop, full 6-DOF (A3+A4).** With the `x1hsts6dof` composite built and the
twin's `sus_hsts_6dof` example archives present (production **L1-MC2** foton banks +
the bare-M1 HSTS state-space plant), the same P&S reference-FRF identifies the 6×6
MIMO suspension under the **real damping loops closed around all six DOFs**:

```bash
python experiments/rtsfreerun/run_hsts6dof.py   # A4 open-loop tensor + A3 closed-loop diagonal
```

A4 recovers the open-loop FRF tensor (diagonal anti-resonances + L↔P / R↔Y coupling);
A3 recovers the **open-loop** plant diagonal *through* the closed loops (controller
cancelled) using the true plant input `DRIVE_EXC − damper_feedback`. Gated by
`tests/test_rtsfreerun_6dof.py` (skips without the twin archives).

## Status

Implemented and tested: model / Fisher / excitation, the resonant plant, the
digital-twin backend (now with MIMO coupling + closed-loop controllers), the
maximum-likelihood (`gml`) estimator, the P&S periodic-multisine measurement and
optimal-excitation design, the orchestration loop (with cross-pass measurement
accumulation and accumulated-Fisher convergence), the safety watchdog + handoff,
config + CLI, and the dashboard core.

Not yet done: the CDS hardware backend (`backends/cds.py` — needs the LIGO CDS
`nds2`/awg libraries and hardware) is a stub that raises `NotImplementedError`.
