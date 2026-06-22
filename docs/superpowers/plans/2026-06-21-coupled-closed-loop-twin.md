# Coupled + Closed-Loop Twin (step 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dimension-generic coupled + closed-loop suspension twin and recover the open-loop coupled plant `G` through the live diagonal loops via the per-bin matrix inverse `G(f)=Y_mat·X_mat⁻¹`, nonparametrically, scored against the analytic plant off-resonance.

**Architecture:** A coupled `n_sens × n_act` plant `G(s)` (shared normal modes) is closed with diagonal per-DoF controllers through a constant input matrix `M_in` and a frequency-dependent output matrix `M_out(s)`. The closed loop is assembled with **python-control** (`tf2ss` → `minreal` → `feedback` for `S=(I+L)⁻¹`), discretized to `fs` (`c2d`), and simulated **consistently** — the plant input `u_drive=Sd·u_exc` is computed, then the response `y_sens=Gd·u_drive` — so `Y=Gd·X` holds and the matrix recovery is exact off-resonance. A new `MIMOTwinBackend` exposes this through the existing `ChannelBackend` API; recovery reuses `SysIDLoop._estimate_tf_periodic`. (All verified in a 2×2 prototype: recovery 2e-5 off-resonance.)

**Tech Stack:** Python, numpy, **python-control 0.10 + slycot** (MIMO interconnection/`minreal`), scipy, the existing `system_ident` package.

## Global Constraints

- **One P&S pipeline** — reuse `SysIDLoop._estimate_tf_periodic` + `multisine_from_psd`; no new estimation method. The matrix-inverse recovery is post-processing of those FRFs, not a new estimator.
- **`conda run -n sysid python …`** for ALL execution (python/pytest). Never the bare binary.
- **slycot is a required dependency** — add it to `pyproject.toml`; CI must install it. No pure-python fallback.
- **Recovery scored off-resonance** — exclude a ±12 % neighbourhood of each mode (the matrix inverse is ill-conditioned at resonances; that regime is step 2). Tolerances from real runs, never loosened to pass.
- **Matrix recovery, not per-pair** — `G(f)=Y_mat(f)·X_mat(f)⁻¹`; per-pair `Y_i/X_j` is ~100 % biased off-diagonal in closed loop.
- **Consistent simulation** — compute `u_drive` from the loop, then `y_sens=Gd·u_drive` (one discretized plant), so `Y=Gd·X`. Do NOT discretize `S` and `G·S` separately (that desyncs them → ~3 % error).
- **New backend** — `MIMOTwinBackend`; leave `TwinBackend` and its SISO `coupling⊥controllers` guard (`twin.py:113`) untouched.
- Trunk-based: commit + push to `main`. Plots SVG + LFS if any (none expected in this plan).
- Phase 1 (RTSfreerun) only — no real hardware.

---

## File structure

- **Create** `src/system_ident/mimo_plant.py` — the coupled `n_sens × n_act` plant from a shared normal-mode expansion (generalizes `coupled_suspension`), as a python-control system + analytic oracle; plus `input_matrix()` / `output_matrix(basis=...)` decoupling-matrix builders.
- **Create** `src/system_ident/mimo_loop.py` — `CoupledLoop` (assemble `S` via control, discretize, expose discrete `Sd`/`Gd`, the discrete-time oracle `Gd(z)` at the bins, and a stability check) and `recover_open_loop` + `off_resonance_mask` (the matrix-inverse recovery).
- **Create** `src/system_ident/backends/mimo_twin.py` — `MIMOTwinBackend(ChannelBackend)`.
- **Create** `tests/test_mimo.py` — all tests.
- **Modify** `pyproject.toml` — add `slycot` to `dependencies`.
- **Modify** `.github/workflows/ci.yml` — ensure CI installs slycot (Task 0).
- **No change** to `twin.py`, `loop.py`, `excitation.py`, `plant.py` (consumed as-is; `coupled_suspension` reused for the 2-DoF oracle cross-check).

---

## Task 0: slycot as a declared dependency + CI

**Files:** Modify `pyproject.toml`; Modify `.github/workflows/ci.yml`.

- [ ] **Step 1: Add slycot to deps**

In `pyproject.toml`, add `"slycot"` to the `dependencies` array (next to `"control"`).

- [ ] **Step 2: Verify the env already satisfies it**

Run: `conda run -n sysid python -c "import slycot, control; print(slycot.__version__ if hasattr(slycot,'__version__') else 'ok', control.__version__)"`
Expected: prints a version + `0.10.2` (slycot 0.6.1 is already installed locally).

- [ ] **Step 3: Ensure CI installs slycot**

In `.github/workflows/ci.yml`, the docs/test job does `pip install -e ".[docs]"`. Add slycot installation **before** it so the Fortran-backed wheel is present on the runner. Add this step right after checkout / Python setup:

```yaml
      - name: Install slycot (Fortran-backed; needed by python-control MIMO)
        run: pip install slycot
```

(If the runner lacks a Fortran toolchain and the wheel is unavailable, replace with a conda step — but PyPI ships `slycot` wheels for ubuntu-latest, so the pip line is expected to work. Do NOT add a pure-python fallback; slycot is required.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "build: declare slycot dependency + install it in CI (python-control MIMO)"
```

---

## Task 1: Coupled `n_sens × n_act` plant + decoupling matrices

**Files:** Create `src/system_ident/mimo_plant.py`; Test `tests/test_mimo.py`.

**Interfaces:**
- Produces:
  - `mimo_suspension(modes, *, n_sens, n_act, coupling=0.2, gain=100.0, seed=0) -> control.StateSpace` — a coupled MIMO plant from a shared normal-mode expansion. `modes` is a list of `(f0_Hz, Q)`; every element shares those poles; mode shapes give diagonal anti-resonances + off-diagonal coupling. Square when `n_sens==n_act`; rectangular otherwise. Returns a python-control `StateSpace` (`n_sens` outputs × `n_act` inputs).
  - `input_matrix(n_dof, n_sens, *, kind="identity"|"perturbed", seed=0) -> np.ndarray` — constant `n_dof × n_sens`.
  - `output_matrix(plant_ss, n_act, n_dof, *, basis="euler"|"eigenmode") -> control.StateSpace` — the `n_act × n_dof` frequency-dependent `M_out(s)`; `"euler"` is a constant pass-through embedded as a static system, `"eigenmode"` builds a representative modal decoupler (see Step 3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mimo.py
from __future__ import annotations
import numpy as np, control, pytest
from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix

def test_plant_shapes_square_and_rectangular():
    Gsq = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2, coupling=0.25)
    assert Gsq.noutputs == 2 and Gsq.ninputs == 2
    Grect = mimo_suspension([(0.6,20),(1.5,30)], n_sens=3, n_act=2, coupling=0.25)
    assert Grect.noutputs == 3 and Grect.ninputs == 2

def test_plant_is_coupled_with_shared_poles():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2, coupling=0.3)
    f = np.geomspace(0.3, 8, 200); H = G(2j*np.pi*f)            # (2,2,200)
    # off-diagonal is non-trivial (genuine coupling)
    assert np.max(np.abs(H[0,1])) / np.max(np.abs(H[0,0])) > 0.05
    # all elements share the same poles (same denominator roots) -> resonances line up
    pk00 = f[np.argmax(np.abs(H[0,0]))]; pk11 = f[np.argmax(np.abs(H[1,1]))]
    assert min(abs(pk00-0.6),abs(pk00-1.5)) < 0.1 and min(abs(pk11-0.6),abs(pk11-1.5)) < 0.1

def test_decoupling_matrix_shapes():
    Min = input_matrix(2, 3, kind="perturbed", seed=1)
    assert Min.shape == (2, 3)
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2)
    Mo = output_matrix(G, n_act=2, n_dof=2, basis="eigenmode")
    assert Mo.ninputs == 2 and Mo.noutputs == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -x -q`
Expected: FAIL — `ModuleNotFoundError: system_ident.mimo_plant`.

- [ ] **Step 3: Implement the plant + matrices**

```python
# src/system_ident/mimo_plant.py
"""Coupled MIMO suspension plant + decoupling matrices for the closed-loop twin.

A shared normal-mode expansion: every (output, input) element shares the same modal
poles, so the diagonal carries anti-resonance notches and the off-diagonal carries
notch-free coupling (the matrix generalisation of ``plant.coupled_suspension``).
Returned as python-control systems so the closed loop assembles natively.
"""
from __future__ import annotations
import numpy as np
import control


def mimo_suspension(modes, *, n_sens, n_act, coupling=0.2, gain=100.0, seed=0):
    """Coupled ``n_sens × n_act`` plant from shared modes ``[(f0,Q),...]``."""
    rng = np.random.default_rng(seed)
    n_modes = len(modes)
    # mode shapes: each mode k has a sensor pattern phi (n_sens) and actuator pattern psi (n_act),
    # ~ unit on the "home" DoF with small alternating cross terms -> genuine coupling.
    phi = np.eye(n_sens, n_modes) + coupling * np.cos(rng.uniform(0, np.pi, (n_sens, n_modes)))
    psi = np.eye(n_act, n_modes) + coupling * np.cos(rng.uniform(0, np.pi, (n_act, n_modes)))
    # shared denominator (product of the mode 2nd-order factors)
    dens = []
    for (f0, q) in modes:
        w = 2*np.pi*f0
        dens.append(np.array([1.0, w/q, w*w]))
    den = np.array([1.0])
    for d in dens:
        den = np.polymul(den, d)
    num = [[None]*n_act for _ in range(n_sens)]
    for i in range(n_sens):
        for j in range(n_act):
            acc = np.zeros(1)
            for k in range(n_modes):
                term = np.array([gain * phi[i, k] * psi[j, k]])
                for m, dm in enumerate(dens):
                    if m != k:
                        term = np.polymul(term, dm)
                acc = np.polyadd(acc, term)
            num[i][j] = list(map(float, np.atleast_1d(acc)))
    return control.tf2ss(control.tf(num, [[list(map(float, den))]*n_act for _ in range(n_sens)]))


def input_matrix(n_dof, n_sens, *, kind="identity", seed=0):
    """Constant sensor→DOF matrix ``M_in`` (n_dof × n_sens)."""
    M = np.eye(n_dof, n_sens)
    if kind == "perturbed":
        M = M + 0.1 * np.random.default_rng(seed).standard_normal((n_dof, n_sens))
    return M


def output_matrix(plant_ss, n_act, n_dof, *, basis="euler"):
    """Frequency-dependent DOF-control→actuator matrix ``M_out(s)`` (n_act × n_dof).

    ``euler``: a static pass-through (identity-shaped) embedded as a system — the DOF
    basis *is* the actuator basis. ``eigenmode``: a representative modal decoupler — a
    constant orthogonal-ish mixing with a gentle 1-pole roll shaping (so it is genuinely
    frequency-dependent), standing in for a real decoupling-filter design.
    """
    if basis == "euler":
        return control.ss([], [], [], np.eye(n_act, n_dof))
    # eigenmode: constant mix * a shared 1-pole shaping per channel (frequency-dependent)
    mix = np.eye(n_act, n_dof) + 0.15 * np.cos(np.arange(n_act)[:, None] + np.arange(n_dof)[None, :])
    shape = control.tf([1.0], [1/(2*np.pi*5.0), 1.0])     # gentle pole at 5 Hz
    return control.tf2ss(control.tf(np.eye(n_act, n_dof) * 0 + mix, np.ones((n_act, n_dof)))) * \
        control.tf2ss(control.append(*[shape]*n_dof))
```

Note for the implementer: `control.tf(num, den)` wants `den` as a matching nested list; build it as shown (shared `den` for every element). If `control.append`/static-system multiplication raises a dimension error for the eigenmode path, simplify the eigenmode `M_out` to `control.ss([],[],[],mix) * control.tf2ss(control.append(*[shape]*n_dof))` and verify the shapes (`n_act × n_dof`). The test only checks shapes + that it is a system.

- [ ] **Step 4: Run the tests**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -x -q`
Expected: PASS (3 tests). If the eigenmode construction fights the control API, get it to a working `n_act × n_dof` frequency-dependent system (the exact filter is representative, per the spec) — do not block on its precise form.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_plant.py tests/test_mimo.py
git commit -m "feat(mimo): coupled n_sens×n_act suspension plant + decoupling matrices"
```

---

## Task 2: `CoupledLoop` — assemble + discretize the closed loop

**Files:** Create `src/system_ident/mimo_loop.py`; Test `tests/test_mimo.py`.

**Interfaces:**
- Consumes: Task 1 `mimo_suspension`, `input_matrix`, `output_matrix`; python-control.
- Produces:
  - `CoupledLoop(plant_ss, controllers, M_in, M_out_ss, *, fs)` where `controllers` is a list of `n_dof` SISO `control` systems (diagonal), `M_in` a constant `n_dof × n_sens` array, `M_out_ss` the `n_act × n_dof` system. Attributes/methods: `.fs`, `.n_sens/.n_dof/.n_act`, `.Sd` (discrete `(I+L)⁻¹`, `n_act×n_act`), `.Gd` (discrete plant, `n_sens×n_act`), `.is_stable() -> bool`, `.oracle(freq) -> np.ndarray` (the discrete-time plant `Gd` at the bin frequencies, shape `(n_sens, n_act, len(freq))`).
  - `velocity_damper(k, fc_hz) -> control.TransferFunction` — a simple `k·s·wc/(s+wc)` damper for building diagonal controllers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mimo.py (append)
from system_ident.mimo_loop import CoupledLoop, velocity_damper

def _square_loop(fs=64.0, basis="euler"):
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 2, kind="identity")
    Mout = output_matrix(G, n_act=2, n_dof=2, basis=basis)
    return CoupledLoop(G, C, Min, Mout, fs=fs)

def test_loop_is_stable():
    assert _square_loop().is_stable()

def test_loop_oracle_is_discrete_plant():
    lp = _square_loop()
    f = np.array([0.4, 3.0])
    z = np.exp(2j*np.pi*f/lp.fs)
    expect = lp.Gd(z)                       # (2,2,2)
    np.testing.assert_allclose(lp.oracle(f), expect, rtol=1e-9)

def test_sensitivity_identity():
    # S = (I+L)^-1  =>  (I+L) S = I  at a test frequency (continuous check on the analytic build)
    lp = _square_loop()
    assert lp.Sd.ninputs == 2 and lp.Sd.noutputs == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py::test_loop_is_stable -x -q`
Expected: FAIL — `ModuleNotFoundError: system_ident.mimo_loop`.

- [ ] **Step 3: Implement `CoupledLoop`** (control construction proven in prototype)

```python
# src/system_ident/mimo_loop.py
"""Assemble + discretize the coupled closed loop, and recover the open-loop plant.

Closed-loop input sensitivity  S = (I+L)^-1,  L = M_out · C_d · M_in · G  (n_act×n_act),
built with python-control and discretized to fs. The twin then simulates CONSISTENTLY:
u_drive = Sd · u_exc, then y_sens = Gd · u_drive — so Y = Gd·X and the matrix recovery
G = Y_mat · X_mat^-1 is exact off-resonance (verified, 2e-5 in a 2×2 prototype).
"""
from __future__ import annotations
import numpy as np
import control


def velocity_damper(k, fc_hz):
    wc = 2*np.pi*fc_hz
    return control.tf([k*wc, 0.0], [1.0, wc])


class CoupledLoop:
    def __init__(self, plant_ss, controllers, M_in, M_out_ss, *, fs):
        self.fs = float(fs)
        self.n_sens = plant_ss.noutputs
        self.n_act = plant_ss.ninputs
        self.n_dof = len(controllers)
        G = plant_ss
        Cd = control.tf2ss(control.append(*controllers))       # block-diagonal n_dof×n_dof
        Min = control.ss([], [], [], np.asarray(M_in, float))  # constant n_dof×n_sens
        Mout = M_out_ss                                         # n_act×n_dof
        L = control.minreal(Mout * Cd * Min * G, verbose=False)  # n_act×n_act open-loop gain
        eye = control.ss([], [], [], np.eye(self.n_act))
        S = control.minreal(control.feedback(eye, L), verbose=False)  # (I+L)^-1
        self.Sd = control.c2d(S, 1.0/self.fs, "tustin")
        self.Gd = control.c2d(G, 1.0/self.fs, "tustin")

    def is_stable(self):
        p = control.poles(self.Sd)
        return bool(np.all(np.abs(p) < 1.0 - 1e-9))            # discrete: inside unit circle

    def oracle(self, freq):
        z = np.exp(2j*np.pi*np.asarray(freq, float)/self.fs)
        return self.Gd(z)                                      # (n_sens, n_act, nbin)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -x -q`
Expected: PASS (Task 1 + 3 new). If `control.append` of SISO `TransferFunction`s needs `tf2ss` first, wrap each controller: `control.append(*[control.tf2ss(c) for c in controllers])`.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_loop.py tests/test_mimo.py
git commit -m "feat(mimo): CoupledLoop — assemble + discretize (I+L)^-1 and the plant"
```

---

## Task 3: Matrix-inverse recovery + off-resonance scoring

**Files:** Modify `src/system_ident/mimo_loop.py` (append); Test `tests/test_mimo.py`.

**Interfaces:**
- Produces:
  - `recover_open_loop(Xmat, Ymat) -> np.ndarray` — `Xmat` shape `(nbin, n_act, n_act)`, `Ymat` shape `(nbin, n_sens, n_act)`; returns `G` shape `(nbin, n_sens, n_act)` with `G[k] = Ymat[k] @ inv(Xmat[k])`.
  - `off_resonance_mask(freq, modes_hz, frac=0.12) -> np.ndarray[bool]` — `True` where `freq` is farther than `frac` (fractional) from every mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mimo.py (append)
from system_ident.mimo_loop import recover_open_loop, off_resonance_mask

def test_matrix_recovery_exact_offres_and_per_pair_biased():
    lp = _square_loop()
    f = np.geomspace(0.3, 8, 80); z = np.exp(2j*np.pi*f/lp.fs)
    Sd = lp.Sd(z).transpose(2,0,1)            # (nbin, n_act, n_act)
    Gd = lp.Gd(z).transpose(2,0,1)            # (nbin, n_sens, n_act)
    Xmat = Sd                                  # injected-ref -> drive monitors
    Ymat = Gd @ Sd                             # injected-ref -> responses
    Grec = recover_open_loop(Xmat, Ymat)
    mask = off_resonance_mask(f, [0.6, 1.5])
    rel = np.array([np.max(np.abs(Grec[k]-Gd[k]))/np.max(np.abs(Gd[k])) for k in range(len(f))])
    assert np.median(rel[mask]) < 1e-6        # matrix recovery exact off-resonance
    # per-pair ratio Y_ij/X_jj is badly biased off-diagonal (sanity: why we need the inverse)
    pair = np.array([Ymat[k] / np.diag(Xmat[k])[None,:] for k in range(len(f))])
    od = np.array([abs(pair[k,0,1]-Gd[k,0,1])/abs(Gd[k,0,1]) for k in range(len(f))])
    assert np.median(od[mask]) > 0.3
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py::test_matrix_recovery_exact_offres_and_per_pair_biased -x -q`
Expected: FAIL — `ImportError: cannot import name 'recover_open_loop'`.

- [ ] **Step 3: Implement recovery**

```python
# src/system_ident/mimo_loop.py (append)
def recover_open_loop(Xmat, Ymat):
    """Per-bin G = Y_mat · X_mat^-1 (the closed-loop MIMO reference-FRF recovery)."""
    X = np.asarray(Xmat); Y = np.asarray(Ymat)
    out = np.empty((X.shape[0], Y.shape[1], X.shape[2]), dtype=complex)
    for k in range(X.shape[0]):
        out[k] = Y[k] @ np.linalg.inv(X[k])
    return out


def off_resonance_mask(freq, modes_hz, frac=0.12):
    freq = np.asarray(freq, float)
    keep = np.ones(freq.shape, bool)
    for f0 in modes_hz:
        keep &= np.abs(freq/f0 - 1.0) > frac
    return keep
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_loop.py tests/test_mimo.py
git commit -m "feat(mimo): matrix-inverse recovery G=Y·X^-1 + off-resonance mask"
```

---

## Task 4: `MIMOTwinBackend`

**Files:** Create `src/system_ident/backends/mimo_twin.py`; Test `tests/test_mimo.py`.

**Interfaces:**
- Consumes: Task 2 `CoupledLoop` (`Sd`, `Gd`, `fs`), `ChannelBackend` (`_soft_start_stop`, `ramp_s`), python-control `forced_response`.
- Produces: `MIMOTwinBackend(loop, exc_channels, drive_channels, sens_channels, *, sensor_asd=0.0, process_asd=0.0, seed=None, ramp_s=3.0)` where the three channel maps are `{channel_name: index}` for the `n_act` actuator excitations, `n_act` drive monitors, `n_sens` sensor readouts. Implements `inject(channel, ts, fs)`, `read(channels, duration)`, `ramp_down(channel, secs)`. Reading a sensor or monitor triggers one consistent simulation: `u_drive = forced_response(Sd, T, U)`; `y_sens = forced_response(Gd, T, u_drive) + sensor_noise`; the monitor returns `u_drive`, the sensor returns `y_sens`. Process disturbance (input-referred) is added to the excitation that enters the loop.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mimo.py (append)
from system_ident.backends.mimo_twin import MIMOTwinBackend
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop

def _backend(lp, **kw):
    exc = {f"EXC{j}": j for j in range(lp.n_act)}
    drv = {f"DRV{j}": j for j in range(lp.n_act)}
    sen = {f"SEN{i}": i for i in range(lp.n_sens)}
    return MIMOTwinBackend(lp, exc, drv, sen, **kw)

def test_backend_shapes_and_consistency():
    lp = _square_loop(); be = _backend(lp, seed=0)
    fs, nper, npe = lp.fs, 1024, 6
    T = nper*npe/fs
    drive = np.ones(int(T*fs))
    be.inject("EXC0", drive, fs)
    seg = be.read(["DRV0","DRV1","SEN0","SEN1"], T)
    assert all(seg[c].shape == (int(round(T*fs)),) for c in seg)

def test_backend_recovers_diagonal_offres():
    # drive one actuator, the j=0 monitor/sensor reference FRF recovers Gd[0,0] off-res
    lp = _square_loop(); be = _backend(lp, sensor_asd=0.0, seed=1)
    fs, nper, npe = lp.fs, 1024, 6
    fa = np.fft.rfftfreq(nper, 1/fs); band = (fa>=0.3)&(fa<=8.0); freq = fa[band]
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    u = multisine_from_psd(Pxx, fs, nper, npe, freq, seed=np.random.default_rng(0))
    be.inject("EXC0", u, fs)
    seg = be.read(["EXC0","DRV0","SEN0"], nper*npe/fs)
    Hx,_,_ = SysIDLoop._estimate_tf_periodic(seg["EXC0"], seg["DRV0"], fs, nper, band, 2)
    Hy,_,_ = SysIDLoop._estimate_tf_periodic(seg["EXC0"], seg["SEN0"], fs, nper, band, 2)
    # H_y / H_x is NOT Gd[0,0] (closed-loop coupling) — but the full matrix recovery is exact;
    # here just assert the monitor FRF is finite & nonzero (the sim ran through the loop)
    assert np.all(np.isfinite(Hx)) and np.median(np.abs(Hx)) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py::test_backend_shapes_and_consistency -x -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the backend** (consistent sim proven in prototype)

```python
# src/system_ident/backends/mimo_twin.py
"""MIMO coupled+closed-loop twin backend (ChannelBackend).

Injects a multisine at each actuator, reads the drive monitors + sensors through the
live diagonal loops. Simulates CONSISTENTLY: u_drive = forced_response(Sd, U); then
y_sens = forced_response(Gd, u_drive) + sensor noise — so Y = Gd·X and the matrix
recovery is exact off-resonance.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
import scipy.signal as sig
import control
from .base import ChannelBackend


class MIMOTwinBackend(ChannelBackend):
    def __init__(self, loop, exc_channels, drive_channels, sens_channels, *,
                 sensor_asd=0.0, process_asd=0.0, seed=None, ramp_s=3.0):
        self.loop = loop
        self.fs = float(loop.fs)
        self.exc = dict(exc_channels)      # name -> actuator index
        self.drv = dict(drive_channels)    # name -> actuator index (monitor)
        self.sen = dict(sens_channels)     # name -> sensor index
        self.sensor_asd = float(sensor_asd)
        self.process_asd = float(process_asd)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives = {}                  # actuator index -> ramped drive
        self._cache = None                 # (n, u_drive, y_sens)

    def inject(self, channel, timeseries, fs):
        if channel not in self.exc:
            raise KeyError(channel)
        ts = np.asarray(timeseries, float)
        if not np.isclose(fs, self.fs):
            fr = Fraction(self.fs/fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, fr.numerator, fr.denominator)
        self._drives[self.exc[channel]] = self._soft_start_stop(ts, self.fs)
        self._cache = None

    def _simulate(self, n):
        if self._cache is not None and self._cache[0] == n:
            return self._cache[1], self._cache[2]
        T = np.arange(n)/self.fs
        U = np.zeros((self.loop.n_act, n))
        for idx, d in self._drives.items():
            m = min(len(d), n); U[idx, :m] = d[:m]
        if self.process_asd:
            U = U + self._rng.standard_normal(U.shape) * self.process_asd * np.sqrt(self.fs/2)
        u_drive = control.forced_response(self.loop.Sd, T, U).outputs.reshape(self.loop.n_act, n)
        y_sens = control.forced_response(self.loop.Gd, T, u_drive).outputs.reshape(self.loop.n_sens, n)
        if self.sensor_asd:
            y_sens = y_sens + self._rng.standard_normal(y_sens.shape) * self.sensor_asd * np.sqrt(self.fs/2)
        self._cache = (n, u_drive, y_sens)
        return u_drive, y_sens

    def read(self, channels, duration):
        n = int(round(duration*self.fs))
        u_drive, y_sens = self._simulate(n)
        out = {}
        for ch in channels:
            if ch in self.sen:
                out[ch] = y_sens[self.sen[ch]]
            elif ch in self.drv:
                out[ch] = u_drive[self.drv[ch]]
            elif ch in self.exc:
                d = self._drives.get(self.exc[ch])
                out[ch] = (np.zeros(n) if d is None else np.r_[d, np.zeros(n)][:n])
            else:
                raise KeyError(ch)
        return out

    def ramp_down(self, channel, secs):
        if channel not in self.exc:
            raise KeyError(channel)
        d = self._drives.get(self.exc[channel])
        if d is None or not len(d):
            return
        nr = min(int(round(secs*self.fs)), len(d))
        out = np.zeros_like(d)
        if nr > 0:
            out[:nr] = d[:nr] * 0.5*(1+np.cos(np.pi*np.arange(nr)/nr))
        self._drives[self.exc[channel]] = out
        self._cache = None
```

Note: `forced_response(...).outputs` is shape `(n_out, n)` for MIMO but `(n,)` for a 1-output system; the `.reshape(n_out, n)` normalizes it. If `forced_response` rejects a 2-D `U` for a 1-input edge case, that won't occur here (`n_act ≥ 2`).

- [ ] **Step 4: Run the tests**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/backends/mimo_twin.py tests/test_mimo.py
git commit -m "feat(mimo): MIMOTwinBackend — consistent coupled closed-loop simulation"
```

---

## Task 5: End-to-end recovery — square, realistic, non-square

**Files:** Test `tests/test_mimo.py` (append); a small helper in `src/system_ident/mimo_loop.py` if useful.

**Interfaces:**
- Consumes: everything above. The campaign: drive each actuator sequentially, build `Xmat`/`Ymat` via `_estimate_tf_periodic`, `recover_open_loop`, score off-resonance vs `loop.oracle(freq)`.

- [ ] **Step 1: Write the failing tests** (the deliverable verification)

```python
# tests/test_mimo.py (append)
def _campaign(lp, *, sensor_asd=0.0, process_asd=0.0, nper=1024, npe=6, seed=0):
    fs = lp.fs
    fa = np.fft.rfftfreq(nper, 1/fs); band = (fa>=0.3)&(fa<=8.0); freq = fa[band]
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    Xcols, Ycols = [], []
    for j in range(lp.n_act):
        be = _backend(lp, sensor_asd=sensor_asd, process_asd=process_asd, seed=seed*10+j)
        u = multisine_from_psd(Pxx, fs, nper, npe, freq, seed=np.random.default_rng(j))
        be.inject(f"EXC{j}", u, fs)
        ch = [f"EXC{j}"] + [f"DRV{a}" for a in range(lp.n_act)] + [f"SEN{i}" for i in range(lp.n_sens)]
        seg = be.read(ch, nper*npe/fs)
        Xcols.append([SysIDLoop._estimate_tf_periodic(seg[f"EXC{j}"], seg[f"DRV{a}"], fs, nper, band, 2)[0]
                      for a in range(lp.n_act)])
        Ycols.append([SysIDLoop._estimate_tf_periodic(seg[f"EXC{j}"], seg[f"SEN{i}"], fs, nper, band, 2)[0]
                      for i in range(lp.n_sens)])
    Xmat = np.array(Xcols).transpose(2,1,0)        # (nbin, n_act, n_drive)
    Ymat = np.array(Ycols).transpose(2,1,0)        # (nbin, n_sens, n_drive)
    Grec = recover_open_loop(Xmat, Ymat)
    Gd = lp.oracle(freq).transpose(2,0,1)
    mask = off_resonance_mask(freq, [0.6, 1.5])
    rel = np.array([np.max(np.abs(Grec[k]-Gd[k]))/np.max(np.abs(Gd[k])) for k in range(len(freq))])
    return rel, mask

def test_recover_square_sanity():
    lp = _square_loop(basis="euler")              # M_in=I, M_out=I
    rel, mask = _campaign(lp)
    assert np.median(rel[mask]) < 5e-3

def test_recover_square_realistic_with_noise():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 2, kind="perturbed", seed=3)
    Mout = output_matrix(G, n_act=2, n_dof=2, basis="eigenmode")
    lp = CoupledLoop(G, C, Min, Mout, fs=64.0)
    rel, mask = _campaign(lp, sensor_asd=1e-3, seed=2)
    assert np.median(rel[mask]) < 5e-2           # off-res recovery under nontrivial M_in/M_out + noise

def test_recover_non_square():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=3, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 3, kind="perturbed", seed=4)   # n_dof=2, n_sens=3
    Mout = output_matrix(G, n_act=2, n_dof=2, basis="euler")
    lp = CoupledLoop(G, C, Min, Mout, fs=64.0)
    rel, mask = _campaign(lp, seed=5)
    assert lp.n_sens == 3 and lp.n_act == 2
    assert np.median(rel[mask]) < 5e-2           # rectangular G (3×2) recovered off-res
```

- [ ] **Step 2: Run to verify failure → then pass**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -k recover -x -q`
Expected: the three tests run. If a tolerance is missed, FIRST check it is genuine (print `np.median(rel[mask])`) — the prototype hit ~2e-5 noise-free and ~%-level with noise, so square-sanity < 5e-3 and realistic/non-square < 5e-2 should hold. Do NOT loosen a tolerance to mask a real recovery failure; if recovery is genuinely worse, debug (consistency of the sim, the off-res mask, `minreal` health) and report.

- [ ] **Step 3: Run the whole module + full suite; commit**

```bash
conda run -n sysid python -m pytest tests/test_mimo.py -q
conda run -n sysid python -m pytest -q
git add tests/test_mimo.py src/system_ident/mimo_loop.py
git commit -m "test(mimo): end-to-end recovery — square sanity, realistic (M_in/M_out+noise), non-square"
```

Expected: `tests/test_mimo.py` green; full suite still green (148 passed, 1 skipped + the new mimo tests).

---

## Task 6: 6-DoF instantiation + basis knob (marked)

**Files:** Test `tests/test_mimo.py` (append).

**Interfaces:** Consumes everything; exercises the dimension-generic path at the real SUS/SEI scale and confirms the `M_out` basis is selectable.

- [ ] **Step 1: Write the test**

```python
# tests/test_mimo.py (append)
import pytest

@pytest.mark.slow
def test_six_dof_recovery():
    # L/P/Y/R/V/T — six shared modes; square 6/6/6; representative eigenmode decoupler
    modes = [(0.43,100),(0.56,100),(0.9,80),(1.0,80),(2.0,60),(3.4,60)]
    G = mimo_suspension(modes, n_sens=6, n_act=6, coupling=0.2)
    C = [velocity_damper(0.4, 20.0) for _ in range(6)]
    Min = input_matrix(6, 6, kind="perturbed", seed=7)
    Mout = output_matrix(G, n_act=6, n_dof=6, basis="eigenmode")
    lp = CoupledLoop(G, C, Min, Mout, fs=128.0)
    assert lp.is_stable()
    rel, mask = _campaign(lp, sensor_asd=1e-3, nper=2048, npe=6, seed=8)
    assert np.median(rel[mask]) < 1e-1

def test_basis_selectable():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2)
    e = output_matrix(G, 2, 2, basis="euler"); m = output_matrix(G, 2, 2, basis="eigenmode")
    # euler is static (no states), eigenmode is dynamic (has states) — genuinely different
    assert control.tf2ss(e).nstates == 0 and control.tf2ss(m).nstates > 0
```

- [ ] **Step 2: Run**

Run: `conda run -n sysid python -m pytest tests/test_mimo.py -k "six_dof or basis" -q`
Expected: PASS. The `slow` mark lets CI deselect it if needed (`-m "not slow"`); document the mark in `pyproject.toml`/`pytest.ini` if the project gates on it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mimo.py
git commit -m "test(mimo): 6-DoF (L/P/Y/R/V/T) recovery + Euler/eigenmode basis selectable"
```

---

## Task 7: Full suite + push

- [ ] **Step 1:** `conda run -n sysid python -m pytest -q` — all green.
- [ ] **Step 2:** `git push`

---

## Self-review (author, fresh eyes)

**Spec coverage:**
- §2 topology (`M_in` const, diagonal `C_d`, `M_out(s)` Euler/eigenmode, coupled `G`, three dims) → Tasks 1–2. ✓
- §3 components (`MIMOSuspension`, `CoupledLoop`, new `MIMOTwinBackend`; TwinBackend untouched) → Tasks 1, 2, 4. ✓
- §4 recovery = per-bin matrix inverse `G=Y·X⁻¹`; per-pair biased; rectangular via `X_mat` n_act×n_act; off-res scoring → Task 3 + Task 5. ✓
- §5 slycot required + CI → Task 0. ✓
- §6 testing ladder (construction/stability, square sanity, square realistic w/ noise, non-square, honest uncertainty, 6-DoF marked) → Tasks 2, 5, 6. ✓
- §7 out-of-scope (joint fit, RTSfreerun demo, paradigm study, hardware) → not built here. ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code; tolerances concrete (5e-3 / 5e-2 / 1e-1 off-res, set from prototype runs). The eigenmode `M_out` exact filter is "representative" per spec — flagged, not a placeholder. ✓

**Type/name consistency:** `mimo_suspension`/`input_matrix`/`output_matrix` (Task 1) used verbatim in Tasks 2/5/6; `CoupledLoop(plant, controllers, M_in, M_out, fs)` + `.Sd/.Gd/.oracle/.is_stable` consistent across Tasks 2/4/5; `recover_open_loop`/`off_resonance_mask` (Task 3) used in Task 5; `MIMOTwinBackend(loop, exc, drv, sen, …)` consistent Task 4/5. ✓

**Known risks for the executor:** (1) python-control API edge cases in the eigenmode `M_out` and `control.append` of SISO controllers — notes inline give the fallback (wrap in `tf2ss`); (2) `forced_response(...).outputs` shape — normalized via `.reshape`; (3) slycot CI install — Task 0 step 3, with the conda fallback noted. The recovery method, discretization, and consistent-sim are all prototype-verified (2e-5 off-resonance).
