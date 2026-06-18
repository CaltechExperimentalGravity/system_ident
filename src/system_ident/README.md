# system_ident

Real-time, optimal-excitation **system identification for LIGO suspensions**.

`system_ident` consolidates the proven research code in this repository — the
`sys_id_dev/sysIDlib.py` engine, the `Plant_Model` 3-DoF suspension, and the
`Optimal_Controls` estimators — into one
installable tool that runs iterative system identification of a 3-DoF
suspension (POS / Pitch / Yaw) against either:

- **real CDS hardware** — `awg` excitation injection + `nds2` online readback, or
- **a digital twin** — the same channel API pointed at a simulation,

through one identical channel interface, while an operator watches a **local
web dashboard** (FastAPI + websockets + Plotly.js) and can **STOP** at any
time, triggering a safe ramp-down + state handoff.

## Status

Under construction. The package skeleton and interfaces are in place; behaviour
is being filled in per the build order:

1. Package skeleton + `pyproject.toml`  ← **done**
2. `model` / `plant` / `fisher` / `excitation` (wrap `sysIDlib` + `Plant_Model`)
3. Default `GMLEstimator` + `PintelonSchoukensDesigner`
4. `TwinBackend`
5. `safety` + `loop` (full closed loop on the twin, headless)
6. Live dashboard + STOP
7. `config` + `cli` + example config
8. `CDSBackend` (real `awg`/`nds2`)
9. Remaining estimator/designer strategies

## Pintelon-Schoukens measurement & ML fit

The measurement is a periodic random-phase **multisine** (Schroeder phases, low
crest factor), and the FRF is read off a leakage-free, integer-period
synchronous DFT. The estimate is the *reference-based* ratio-of-averages
`mean(Y)/mean(X)`, which recovers the open-loop plant even with a damping loop
closed (the naive `S_yx/S_xx` is loop-biased). Pair it with
`strategy.estimator: gml` for the maximum-likelihood (Gauss-Newton/LM) fit,
which is unbiased, attains the Cramér-Rao bound, and is naturally multi-mode.

**The measurement is validated only on the digital twin** — see the
`LIMITATIONS` note in `backends/cds.py` for the real-CDS effects (AWG↔NDS
clocking, timestamp alignment, actuator nonlinearity, true servo dynamics) that
remain untested until `CDSBackend` is implemented.

## Install (development)

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[dev,dashboard]"  # also the live dashboard stack
```

The CDS libraries (`nds2`, `awg`/`cdsutils`) come from the LIGO CDS
environment and are lazy-imported, so the twin path works on a plain
scientific-Python stack.

## Usage (target interface)

```bash
system_ident run configs/three_dof_twin.yml --twin
```
