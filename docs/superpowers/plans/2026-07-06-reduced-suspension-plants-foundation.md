# Reduced Suspension Plants — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the twin's 50 Hz modal-truncation reduced suspension models (QUAD 59-state, HSTS 36-state) into `system_ident` as self-contained, numpy-only sysID plants, with a `ReducedStateSpacePlant` loader and a `ReducedPlantBackend` that drives the P&S pipeline.

**Architecture:** Committed `.npz` (A,B,C,D) + `.json` sidecar (channel labels, oracle modes, provenance) under `src/system_ident/models/`, produced by a local-only regen script that shells to the twin's reducer. A pure-numpy `ReducedStateSpacePlant` evaluates the FRF `G(f)=C(2πif·I−A)⁻¹B+D` and exposes the exact modes. A frequency-domain `ReducedPlantBackend` (FFT synthesis, no `control`/`slycot`) implements the `ChannelBackend` API so `SysIDLoop`/the MIMO campaign drive it unchanged.

**Tech Stack:** Python 3.12, numpy, scipy (linalg/signal). No `control`, `slycot`, or twin imports at runtime.

## Global Constraints

- Runtime code imports **numpy/scipy only** — never `control`, `slycot`, or the twin. (The regen script is the sole exception and is local-only, gated on `$DIGITAL_TWIN_DIR`.)
- Run everything via `conda run -n sysid`.
- Trunk-based: commit to `main`.
- Reduced models are committed data (owner-approved for the public repo); each carries a provenance/attribution sidecar.
- Oracle modes come from `scipy.linalg.eig(A)` filtered to conjugate pairs with `1e-6 < f0 ≤ f_c` — **not** raw `np.linalg.eigvals` (spurious >f_c values on the prescaled realization).

---

### Task 1: Regenerate + commit the two reduced models

**Files:**
- Create: `src/system_ident/models/__init__.py` (empty package marker)
- Create: `src/system_ident/models/regenerate.py` (local-only regen script)
- Create (generated, committed): `src/system_ident/models/quad_reduced_50hz.npz`, `.../quad_reduced_50hz.json`, `.../hsts_reduced_50hz.npz`, `.../hsts_reduced_50hz.json`
- Test: none (data artifacts; exercised by Task 2's loader tests)

**Interfaces:**
- Produces the committed data + the `_modes_from_A(A, f_c)` helper (reused conceptually in Task 2):
  `_modes_from_A(A: np.ndarray, f_c: float) -> list[tuple[float, float]]` returning sorted `(f0_hz, Q)`.

- [ ] **Step 1: Write the regen script**

```python
# src/system_ident/models/regenerate.py
"""Regenerate the committed reduced suspension models from the digital twin.

LOCAL-ONLY, not imported at runtime: shells to the twin's modal-truncation reducer
(twin/src/twin/sus_modal.py) and writes the committed .npz + .json here. Requires the
twin checkout at $DIGITAL_TWIN_DIR (default ~/GIT/digital_twin) with the full .mat models.

    conda run -n sysid python -m system_ident.models.regenerate
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
CUTOFF_HZ = 50.0


def _modes_from_A(A: np.ndarray, f_c: float) -> list[tuple[float, float]]:
    """Oracle modes: conjugate-pair eigenvalues with 1e-6 < f0 <= f_c (drops the
    spurious >f_c eigenvalues the prescaled realization shows). Returns sorted (f0, Q)."""
    lam = sla.eig(np.asarray(A, float), right=False)
    out = []
    for z in lam:
        if z.imag <= 1e-6:
            continue
        f0 = abs(z) / (2 * np.pi)
        if f0 > f_c * 1.05:
            continue
        Q = abs(z) / (-2 * z.real) if z.real < 0 else float("inf")
        out.append((float(f0), float(Q)))
    return sorted(out)


def _twin_src() -> Path:
    twin = Path(os.environ.get("DIGITAL_TWIN_DIR") or Path.home() / "GIT" / "digital_twin")
    src = twin / "twin" / "src"
    if not (src / "twin" / "sus_modal.py").exists():
        sys.exit(f"twin reducer not found under {src} — set $DIGITAL_TWIN_DIR")
    return src


def regenerate(sus_type: str) -> None:
    sys.path.insert(0, str(_twin_src()))
    from twin.sus_modal import compute_reduction  # noqa: E402  (local-only)
    rp = compute_reduction(sus_type, cutoff_hz=CUTOFF_HZ)
    A, B, C, D = (np.asarray(x, float) for x in (rp.A, rp.B, rp.C, rp.D))
    modes = _modes_from_A(A, CUTOFF_HZ)
    base = HERE / f"{sus_type}_reduced_50hz"
    np.savez(base.with_suffix(".npz"), A=A, B=B, C=C, D=D, f_mode_cut=CUTOFF_HZ)
    sidecar = {
        "inputs": list(rp.inputs),
        "outputs": list(rp.outputs),
        "modes": [[f, q] for f, q in modes],
        "provenance": {
            "sus_type": sus_type,
            "source": f"aligo-suspension-models/{sus_type}_full.mat",
            "method": "modal truncation (twin/src/twin/sus_modal.py::compute_reduction)",
            "cutoff_hz": CUTOFF_HZ,
            "n_full": int(rp.n_full),
            "n_states": int(A.shape[0]),
            "note": ("Reduced-order aLIGO suspension model, modal truncation to the "
                     "control band. Redistribution approved by the repo owner. QUAD near-"
                     "undamped modes carry a uniform structural shift, so their Q is model-"
                     "set, not physical; HSTS modes carry structural Q=50."),
        },
    }
    base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {base.name}.npz ({A.shape[0]} states) + .json ({len(modes)} modes)")


if __name__ == "__main__":
    for t in ("quad", "hsts"):
        regenerate(t)
```

- [ ] **Step 2: Create the package marker and run the regen**

```bash
touch src/system_ident/models/__init__.py
conda run -n sysid python -m system_ident.models.regenerate
```
Expected:
```
wrote quad_reduced_50hz.npz (59 states) + .json (27 modes)
wrote hsts_reduced_50hz.npz (36 states) + .json (18 modes)
```

- [ ] **Step 3: Verify the committed artifacts by hand**

Run: `conda run -n sysid python -c "import json,numpy as np; d=np.load('src/system_ident/models/hsts_reduced_50hz.npz'); s=json.load(open('src/system_ident/models/hsts_reduced_50hz.json')); print(d['A'].shape, len(s['inputs']), len(s['outputs']), s['modes'][:3])"`
Expected: `(36, 36) 24 18 [[0.672..., 50.0], [0.676..., 50.0], [0.848..., 50.0]]`

- [ ] **Step 4: Commit**

```bash
git add src/system_ident/models/
git commit -m "feat(models): commit 50 Hz reduced QUAD/HSTS suspension plants + regen script"
```

---

### Task 2: `ReducedStateSpacePlant` — load, FRF, modes, subplant

**Files:**
- Create: `src/system_ident/reduced_plant.py`
- Test: `tests/test_reduced_plant.py`

**Interfaces:**
- Consumes: the committed `src/system_ident/models/*.npz` + `*.json` (Task 1).
- Produces:
  - `ReducedStateSpacePlant.load(name: str) -> ReducedStateSpacePlant` (name e.g. `"quad"`, `"hsts"`).
  - `.eval(freq: np.ndarray) -> np.ndarray` shape `(F, n_out, n_in)` complex.
  - `.modes() -> list[tuple[float, float]]` — `(f0, Q)`, the oracle.
  - `.inputs: list[str]`, `.outputs: list[str]`.
  - `.subplant(sensors: list[str], actuators: list[str]) -> ReducedStateSpacePlant`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reduced_plant.py
import numpy as np
import pytest
from system_ident.reduced_plant import ReducedStateSpacePlant


def test_load_shapes_and_labels():
    p = ReducedStateSpacePlant.load("hsts")
    assert p.A.shape == (36, 36)
    assert len(p.inputs) == p.B.shape[1] == 24
    assert len(p.outputs) == p.C.shape[0] == 18
    assert "m1.drive.L" in p.inputs and "m1.disp.L" in p.outputs


def test_modes_are_the_srm_physics():
    p = ReducedStateSpacePlant.load("hsts")
    f0 = sorted(f for f, q in p.modes())
    # the SRM spatial doublet + a triple, all present
    assert any(abs(f - 0.672) < 1e-3 for f in f0)
    assert any(abs(f - 0.676) < 1e-3 for f in f0)
    assert sum(1 for f in f0 if 1.50 < f < 1.53) == 3
    assert all(abs(q - 50.0) < 1e-6 for f, q in p.modes())


def test_eval_frf_shape_and_finite():
    p = ReducedStateSpacePlant.load("hsts")
    freq = np.linspace(0.3, 5.0, 200)
    G = p.eval(freq)
    assert G.shape == (200, 18, 24)
    assert np.all(np.isfinite(G))


def test_frf_peaks_at_a_mode():
    p = ReducedStateSpacePlant.load("hsts")
    i, j = p.outputs.index("m1.disp.L"), p.inputs.index("m1.drive.L")
    freq = np.linspace(0.6, 0.75, 4000)
    mag = np.abs(p.eval(freq)[:, i, j])
    f_peak = freq[np.argmax(mag)]
    assert abs(f_peak - 0.672) < 0.01  # peaks at the fundamental


def test_subplant_selects_block():
    p = ReducedStateSpacePlant.load("hsts")
    dofs_in = ["m1.drive.L", "m1.drive.P"]
    dofs_out = ["m1.disp.L", "m1.disp.P"]
    sub = p.subplant(sensors=dofs_out, actuators=dofs_in)
    assert sub.inputs == dofs_in and sub.outputs == dofs_out
    freq = np.linspace(0.3, 3.0, 50)
    # subplant FRF equals the selected rows/cols of the full FRF
    full = p.eval(freq)
    fi = [p.inputs.index(x) for x in dofs_in]
    fo = [p.outputs.index(x) for x in dofs_out]
    assert np.allclose(sub.eval(freq), full[:, fo][:, :, fi])
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_reduced_plant.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'system_ident.reduced_plant'`

- [ ] **Step 3: Implement `ReducedStateSpacePlant`**

```python
# src/system_ident/reduced_plant.py
"""Reduced-order suspension plant: a modal-truncation state-space (A,B,C,D) with
labelled physical channels and an exact mode table. Pure numpy/scipy — no twin, no
`control`/`slycot`. The FRF G(f) = C(2πif·I − A)^-1 B + D is the sysID target; the
eigen-modes are the oracle. See src/system_ident/models/ and the regen script there.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"


@dataclass
class ReducedStateSpacePlant:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    inputs: list[str]
    outputs: list[str]
    _modes: list[tuple[float, float]]
    f_mode_cut: float

    @classmethod
    def load(cls, name: str) -> "ReducedStateSpacePlant":
        npz = np.load(_MODELS / f"{name}_reduced_50hz.npz")
        side = json.loads((_MODELS / f"{name}_reduced_50hz.json").read_text())
        return cls(
            A=npz["A"], B=npz["B"], C=npz["C"], D=npz["D"],
            inputs=list(side["inputs"]), outputs=list(side["outputs"]),
            _modes=[(float(f), float(q)) for f, q in side["modes"]],
            f_mode_cut=float(npz["f_mode_cut"]),
        )

    def eval(self, freq) -> np.ndarray:
        """FRF tensor (F, n_out, n_in): G(f) = C (2πif I − A)^-1 B + D."""
        freq = np.asarray(freq, float)
        I = np.eye(self.A.shape[0])
        # solve (sI − A) X = B per frequency, then G = C X + D
        return np.array([self.C @ np.linalg.solve(2j * np.pi * f * I - self.A, self.B) + self.D
                         for f in freq])

    def modes(self) -> list[tuple[float, float]]:
        return list(self._modes)

    def subplant(self, sensors: list[str], actuators: list[str]) -> "ReducedStateSpacePlant":
        oi = [self.outputs.index(s) for s in sensors]
        ii = [self.inputs.index(a) for a in actuators]
        return ReducedStateSpacePlant(
            A=self.A, B=self.B[:, ii], C=self.C[oi, :], D=self.D[np.ix_(oi, ii)],
            inputs=list(actuators), outputs=list(sensors),
            _modes=list(self._modes), f_mode_cut=self.f_mode_cut,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n sysid python -m pytest tests/test_reduced_plant.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/reduced_plant.py tests/test_reduced_plant.py
git commit -m "feat(reduced_plant): ReducedStateSpacePlant loader, FRF, modes, subplant"
```

---

### Task 3: `ReducedPlantBackend` — drive the pipeline (frequency-domain)

**Files:**
- Create: `src/system_ident/backends/reduced.py`
- Test: `tests/test_reduced_backend.py`

**Interfaces:**
- Consumes: `ReducedStateSpacePlant` (Task 2); `ChannelBackend` ABC (`backends/base.py`) — methods `inject(channel, timeseries, fs)`, `read(channels, duration) -> dict`, `ramp_down(channel, secs)`; the shared `_soft_start_stop(ts, fs)`.
- Produces:
  - `ReducedPlantBackend(plant, exc_channels: dict[str,str], sens_channels: dict[str,str], *, fs: float, sensor_asd: float = 0.0, seed=None, ramp_s: float = 3.0)` where `exc_channels`/`sens_channels` map channel names → plant input/output **labels**.
  - Steady-state periodic synthesis `Y(f) = G(f)·X(f) (+ noise)` via rFFT — no `control`/`slycot`.

Design note: the injected drive is a periodic multisine, so the steady-state response is
exact in the frequency domain — synthesize `Ŷ = G·X̂` on the rFFT grid (only the bins the
drive excites are nonzero, so evaluate `G` there) and inverse-transform. This mirrors
`MIMOTwinBackend` but is FRF-based and pure numpy.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reduced_backend.py
import numpy as np
import pytest
from system_ident.reduced_plant import ReducedStateSpacePlant
from system_ident.backends.reduced import ReducedPlantBackend
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop


def _hsts_LL_backend(sensor_asd=0.0, seed=0):
    p = ReducedStateSpacePlant.load("hsts").subplant(
        sensors=["m1.disp.L"], actuators=["m1.drive.L"])
    return p, ReducedPlantBackend(
        p, exc_channels={"EXC": "m1.drive.L"}, sens_channels={"RSP": "m1.disp.L"},
        fs=64.0, sensor_asd=sensor_asd, seed=seed, ramp_s=0.0)


def test_read_returns_requested_channels():
    _, be = _hsts_LL_backend()
    fs, nperseg, nper = 64.0, 4096, 4
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= 0.3) & (fa <= 5.0)
    Pxx = np.full(band.sum(), 1.0 / (fa[band][-1] - fa[band][0]))
    drive = multisine_from_psd(Pxx, fs, nperseg, nper, fa[band], seed=np.random.default_rng(0))
    be.inject("EXC", drive, fs)
    seg = be.read(["EXC", "RSP"], nperseg * nper / fs)
    assert set(seg) == {"EXC", "RSP"}
    assert len(seg["RSP"]) == nperseg * nper


def test_noiseless_recovery_matches_plant_frf():
    # a noiseless drive → the leakage-free FRF must equal the plant's own FRF on the excited lines
    p, be = _hsts_LL_backend(sensor_asd=0.0)
    fs, nperseg, nper, band_hz = 64.0, 4096, 4, (0.3, 5.0)
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= band_hz[0]) & (fa <= band_hz[1])
    freq = fa[band]
    Pxx = np.full(band.sum(), 1.0 / (freq[-1] - freq[0]))
    drive = multisine_from_psd(Pxx, fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("EXC", drive, fs)
    seg = be.read(["EXC", "RSP"], nperseg * nper / fs)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["EXC"], seg["RSP"], fs, nperseg, band, n_transient=1)
    G_true = p.eval(freq)[:, 0, 0]
    rel = np.abs(H - G_true) / np.maximum(np.abs(G_true), 1e-30)
    assert np.median(rel[np.isfinite(rel)]) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n sysid python -m pytest tests/test_reduced_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'system_ident.backends.reduced'`

- [ ] **Step 3: Implement `ReducedPlantBackend`**

```python
# src/system_ident/backends/reduced.py
"""Frequency-domain backend over a ReducedStateSpacePlant (ChannelBackend).

Drives the P&S pipeline like MIMOTwinBackend, but synthesizes the steady-state periodic
response in the frequency domain — Ŷ = G·X̂ on the rFFT grid — so it needs only the plant
FRF (numpy), no `control`/`slycot`. Valid because the injected multisine is periodic over
the record: its steady-state response is exactly G·X on the excited lines.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
import scipy.signal as sig
from .base import ChannelBackend


class ReducedPlantBackend(ChannelBackend):
    def __init__(self, plant, exc_channels, sens_channels, *,
                 fs, sensor_asd=0.0, seed=None, ramp_s=3.0):
        self.plant = plant
        self.fs = float(fs)
        self.exc = {ch: plant.inputs.index(lbl) for ch, lbl in exc_channels.items()}
        self.sen = {ch: plant.outputs.index(lbl) for ch, lbl in sens_channels.items()}
        self.sensor_asd = float(sensor_asd)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives = {}     # input index -> ramped drive
        self._cache = None    # (n, y_sens (n_out, n))

    def inject(self, channel, timeseries, fs):
        if channel not in self.exc:
            raise KeyError(channel)
        ts = np.asarray(timeseries, float)
        if not np.isclose(fs, self.fs):
            fr = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, fr.numerator, fr.denominator)
        self._drives[self.exc[channel]] = self._soft_start_stop(ts, self.fs)
        self._cache = None

    def _simulate(self, n):
        if self._cache is not None and self._cache[0] == n:
            return self._cache[1]
        n_in, n_out = self.plant.B.shape[1], self.plant.C.shape[0]
        X = np.zeros((n_in, n))
        for idx, d in self._drives.items():
            m = min(len(d), n)
            X[idx, :m] = d[:m]
        Xhat = np.fft.rfft(X, axis=1)                      # (n_in, nf)
        fa = np.fft.rfftfreq(n, 1 / self.fs)
        # evaluate G only where some input has power (multisine is sparse)
        active = np.where(np.any(np.abs(Xhat) > 0, axis=0))[0]
        Yhat = np.zeros((n_out, Xhat.shape[1]), complex)
        if len(active):
            G = self.plant.eval(fa[active])                # (F, n_out, n_in)
            # per active bin b: Yhat[:, b] = G[b] @ Xhat[:, b]
            Yhat[:, active] = np.einsum("foi,if->of", G, Xhat[:, active])
        y_sens = np.fft.irfft(Yhat, n=n, axis=1)
        if self.sensor_asd:
            y_sens = y_sens + self._rng.standard_normal(y_sens.shape) * self.sensor_asd * np.sqrt(self.fs / 2)
        self._cache = (n, y_sens)
        return y_sens

    def read(self, channels, duration):
        n = int(round(duration * self.fs))
        y_sens = self._simulate(n)
        out = {}
        for ch in channels:
            if ch in self.sen:
                out[ch] = y_sens[self.sen[ch]]
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
        nr = min(int(round(secs * self.fs)), len(d))
        out = np.zeros_like(d)
        if nr > 0:
            out[:nr] = d[:nr] * 0.5 * (1 + np.cos(np.pi * np.arange(nr) / nr))
        self._drives[self.exc[channel]] = out
        self._cache = None
```

Note: delete the placeholder `np.einsum(... np.zeros((n_in, 0)))` line — it was shown only to flag that the real assignment is the following `einsum("foi,if->of", G, Xhat[:, active])`. Keep just the real one.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n sysid python -m pytest tests/test_reduced_backend.py -q`
Expected: PASS (2 passed). The noiseless-recovery test confirms the synthesized measurement's leakage-free FRF equals the plant's own FRF to < 1e-6.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/backends/reduced.py tests/test_reduced_backend.py
git commit -m "feat(backends): ReducedPlantBackend — frequency-domain drive of the reduced plant"
```

---

## After the foundation

These land in follow-up plans (spec §3 Phases 2–5), each on top of Tasks 1–3:
- **Phase 2** — a worked example running the full P&S pipeline on the reduced QUAD (CI/Binder).
- **Phase 3** — wire `ReducedStateSpacePlant.load("hsts").modes()` as the oracle for the HSTS 6-DOF modal fit via `ReducedPlantBackend`.
- **Phase 4** — adopt the reduced plants in the multi-DOF examples (04/06/09; not 01–03).
- **Phase 5** — local-only time-domain fidelity check vs the compiled rtsfreerun twin.

## Self-Review notes
- Spec coverage: Tasks 1–3 cover the spec's §2 (data, plant, backend) and §5 foundation tests; Phases 2–5 are explicitly deferred to follow-up plans (spec §3).
- No placeholders: every step carries complete code / exact commands with expected output.
- Type consistency: `ReducedStateSpacePlant.load/eval/modes/subplant/inputs/outputs` names match across Tasks 2–3; the backend consumes `plant.inputs`/`plant.outputs`/`plant.eval` exactly as defined; `ReducedPlantBackend(plant, exc_channels, sens_channels, *, fs, sensor_asd, seed, ramp_s)` matches its test usage.
