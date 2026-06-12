[![CI](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml/badge.svg)](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml)

# system_ident

Real-time, optimal-excitation system identification for LIGO suspensions
(3-DoF: POS / Pitch / Yaw), with a digital-twin backend and a live operational
dashboard.

This is a self-contained, installable package. It consolidates the proven
research code (the `sysIDlib` engine, the suspension plant, the inverse-frequency
estimator, the Pintelon–Schoukens optimal-excitation design) behind one tool:
a CLI, one channel API shared by a digital twin and (later) CDS hardware,
pluggable estimator/input-design strategies, a physical-safety watchdog with a
safe-state handoff, and a websocket dashboard.

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
# -> DONE (target reached), per-DoF fractional uncertainty ~1e-9
```

With the dashboard extra installed, drop `--yes`/add nothing and a live Plotly
view (transfer function, coherence, designed excitation) is served locally with
a STOP button that triggers the safe-state handoff.

## Status

Implemented and tested: model / Fisher / excitation, the resonant plant, the
digital-twin backend, the invfreqs estimator, the orchestration loop (with
cross-pass measurement accumulation and accumulated-Fisher convergence), the
safety watchdog + handoff, config + CLI, and the dashboard core.

Not yet done: the CDS hardware backend (`backends/cds.py` — needs the LIGO CDS
`nds2`/awg libraries and hardware) and Foton export (`export/foton.py`) are
stubs that raise `NotImplementedError`.
