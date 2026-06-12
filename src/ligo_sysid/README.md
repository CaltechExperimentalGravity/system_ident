# ligo_sysid

Real-time, optimal-excitation **system identification for LIGO suspensions**.

`ligo_sysid` consolidates the proven research code in this repository — the
`sys_id_dev/sysIDlib.py` engine, the `Plant_Model` 3-DoF suspension, the
`Optimal_Controls` estimators, and the `CDS_Interface` Foton export — into one
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
3. Default `InvfreqsEstimator` + `PintelonSchoukensDesigner`
4. `TwinBackend`
5. `safety` + `loop` (full closed loop on the twin, headless)
6. Live dashboard + STOP
7. `config` + `cli` + example config
8. `CDSBackend` (real `awg`/`nds2`)
9. Foton export
10. Remaining estimator/designer strategies

## Install (development)

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[dev,dashboard]"  # also the live dashboard stack
```

The CDS libraries (`nds2`, `awg`/`cdsutils`, `foton`) come from the LIGO CDS
environment and are lazy-imported, so the twin path works on a plain
scientific-Python stack.

## Usage (target interface)

```bash
ligo-sysid run configs/three_dof_twin.yml --twin
```
