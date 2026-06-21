# DARM calibration via P&S — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable closed-loop DARM twin and a docs page (Example 08) that runs the existing Pintelon–Schoukens multisine pipeline against it — recovering the response `R(f)`, the sensing function `C`, and the three actuation strengths `κ_U/κ_PU/κ_T` under real process-disturbance + sensing noise, and comparing the result head-to-head with a swept sine on the same twin.

**Architecture:** A new `DARMLoop` package model exposes the sensing `C(f)` (cavity pole + delay), the 3-stage actuation `A(f)`, a derived representative servo `D(f)`, and the closed-loop FRFs per injection point. Because the sensing delay makes a rational time-domain loop intractable, the twin synthesizes the closed-loop response in the **frequency domain** (exact for the periodic multisine; the suspension resonances sit below the measurement band so in-band dynamics are smooth). A `DARMBackend` wraps it in the existing `ChannelBackend` API so `SysIDLoop._estimate_tf_periodic` and the glue run unchanged. Recovery and a swept-sine comparison live in `darm.py`; presentation glue + the page mirror the Example 07 (`rtsfreerun`) pattern.

**Tech Stack:** Python, numpy, scipy (`signal`, `optimize`), plotly (docs only), Quarto (`freeze: true`), Git LFS for SVGs.

## Global Constraints

- **One P&S pipeline only.** Reuse `SysIDLoop._estimate_tf_periodic`, `multisine_from_psd`, and the Fisher/estimator machinery. No new estimation method. The sensing/actuation *parameterization* is domain physics, not a new measurement method.
- **Every plot is SVG and tracked in Git LFS.** No graphics outside LFS.
- **Y-limits are data-driven and verified against real data before render** — never a fixed guess; traces must never be clipped. Use `sysid_plots._logy_range` / explicit ranges computed from the arrays.
- **Representative numbers, labeled.** `f_cc≈360 Hz`, `τ≈77 µs`, UGF≈50 Hz — labeled "representative, not a specific IFO state". No real Foton DARM filter import.
- **No Schroeder/crest-factor framing.** The multisine win for DARM is simultaneity + leakage-free estimation + CRB allocation.
- **Trunk-based:** commit and push straight to `main`. No PRs/topic branches.
- **Backend ramp default is 3 s Tukey** (`ChannelBackend.ramp_s`), applied at injection via `_soft_start_stop`; the leakage-free FRF drops the tapered periods.
- Test tolerances are set from real recovery runs, never loosened to pass; no skipif-hidden regressions.

---

## File structure

- **Create** `src/system_ident/darm.py` — `DARMLoop` (model + frequency-domain `simulate`), `sensing_model`, `fit_sensing`, `recover_response`, `recover_actuation`, `swept_sine_response_sigma`, `multisine_response_sigma`. One focused module; the recovery helpers are small pure functions beside the model they serve.
- **Create** `src/system_ident/backends/darm_adapter.py` — `DARMBackend(ChannelBackend)`.
- **Create** `tests/test_darm.py` — loop self-consistency, synthesis, backend FRF, recovery, comparison.
- **Create** `docs/darm_demo.py` — presentation glue + figure wrappers (mirrors `docs/rtsfreerun_demo.py`).
- **Create** `docs/examples/08-darm-calibration.qmd` — the page (`freeze: true`).
- **Create** `docs/examples/thumbnails/08.svg` — page thumbnail (LFS).
- **Modify** `docs/_quarto.yml` — only if the `examples/*.qmd` glob (line 10) needs an explicit ordering entry; verify it picks up `08` automatically (it globs, so likely no change).
- **No change** to `loop.py`, `excitation.py`, `model.py`, `fisher.py` — consumed as-is.

---

## Task 1: `DARMLoop` model core (C, A, D, G, R + FRF identities)

**Files:**
- Create: `src/system_ident/darm.py`
- Test: `tests/test_darm.py`

**Interfaces:**
- Produces:
  - `sensing_model(freq, g_c, f_cc, tau) -> np.ndarray` (complex) — `g_c/(1+i f/f_cc)·e^{-i2πfτ}`.
  - `DARMLoop` dataclass-like class with attributes `fs, fmin, fmax, g_c, f_cc, tau, stages` (dict `name->(TFModel, kappa)`), `f_ugf, f_hi`; methods `C(freq)`, `A(freq)`, `stage(name, freq)`, `G(freq)`, `D(freq)`, `R(freq)`, `frf_pcal(freq)`, `frf_stage(name, freq)`, all returning complex arrays. `ports` property -> `["PCAL","UIM","PUM","TST"]`.
  - `DARMLoop.default() -> DARMLoop` classmethod with the representative numbers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_darm.py
from __future__ import annotations
import numpy as np
import pytest
from system_ident.darm import DARMLoop, sensing_model

def _freq(loop):
    return np.linspace(loop.fmin, loop.fmax, 400)

def test_sensing_model_pole_and_delay():
    f = np.array([1.0, 360.0, 1000.0])
    H = sensing_model(f, g_c=1e6, f_cc=360.0, tau=77e-6)
    # at the cavity pole magnitude is 1/sqrt(2) of DC, phase rolls past -45 deg + delay
    assert abs(abs(H[1]) - 1e6/np.sqrt(2)) / (1e6/np.sqrt(2)) < 1e-6
    # delay adds linear phase: at 1 kHz the extra phase beyond the pole is -2*pi*f*tau
    extra = np.angle(H[2]) - np.angle(1e6/(1+1j*1000/360))
    assert np.isclose(np.angle(np.exp(1j*extra)), -2*np.pi*1000*77e-6, atol=1e-6)

def test_response_is_one_over_pcal_frf():
    loop = DARMLoop.default()
    f = _freq(loop)
    # R = (1+G)/C  and  FRF_pcal = C/(1+G)  =>  R == 1/FRF_pcal
    np.testing.assert_allclose(loop.R(f), 1.0/loop.frf_pcal(f), rtol=1e-10)

def test_G_equals_A_D_C_by_construction():
    loop = DARMLoop.default()
    f = _freq(loop)
    np.testing.assert_allclose(loop.G(f), loop.A(f)*loop.D(f)*loop.C(f), rtol=1e-9)

def test_stage_frf_identity():
    loop = DARMLoop.default()
    f = _freq(loop)
    for name in ("UIM","PUM","TST"):
        tf, kappa = loop.stages[name]
        expect = loop.C(f)*kappa*tf.eval(f)/(1+loop.G(f))
        np.testing.assert_allclose(loop.frf_stage(name, f), expect, rtol=1e-9)

def test_loop_is_stable_with_margin():
    loop = DARMLoop.default()
    f = np.geomspace(loop.fmin, loop.fmax, 4000)
    G = loop.G(f)
    mag = np.abs(G)
    # unity-gain frequency near the representative UGF
    k = int(np.argmin(np.abs(mag - 1.0)))
    f_ugf = f[k]
    assert 30.0 < f_ugf < 80.0
    pm = 180.0 + np.angle(G[k], deg=True)   # phase margin
    assert pm > 30.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_darm.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'system_ident.darm'`.

- [ ] **Step 3: Implement the model core**

```python
# src/system_ident/darm.py
"""A representative closed-loop DARM twin and the P&S recovery on it.

The DARM loop is  d_err = C·(x_free + x_pc + Σ κ_i N_i c_i)/(1+G),  G = C·A·D,
with sensing C (cavity pole + delay), three-stage actuation A, and a derived
servo D.  Because the sensing delay makes a rational time-domain loop
intractable, the twin synthesises the closed-loop response in the frequency
domain (exact for the periodic P&S multisine; the suspension resonances sit
below the measurement band, so the in-band dynamics are smooth).

All numbers are *representative of an Advanced-LIGO DARM loop, not a specific
interferometer state* — a single coupled-cavity pole + delay for C, three
pendulum-stage actuators, and a UGF≈50 Hz open-loop gain shaped for stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import TFModel


def sensing_model(freq, g_c: float, f_cc: float, tau: float) -> np.ndarray:
    """Optical sensing response C(f) = g_c/(1+i f/f_cc)·exp(-i 2π f τ) [ct/m]."""
    f = np.asarray(freq, dtype=float)
    return g_c / (1.0 + 1j * f / f_cc) * np.exp(-2j * np.pi * f * tau)


def _pendulum_stage(f_pend: float, q: float, gain: float) -> TFModel:
    """One quad actuation stage: a pendulum force→displacement TF [m/ct].

    In the DARM band (well above f_pend) this is the ~1/f² actuator rolloff; the
    resonance itself sits below the measurement band.
    """
    return TFModel.from_resonances([(f_pend, q)], gain)


@dataclass
class DARMLoop:
    """Representative closed-loop DARM twin (single loop, three actuation stages)."""

    fs: float = 4096.0
    fmin: float = 10.0
    fmax: float = 1500.0
    # sensing
    g_c: float = 1.0e6          # optical gain [ct/m]
    f_cc: float = 360.0         # coupled-cavity pole [Hz]
    tau: float = 77.0e-6        # light-travel / processing delay [s]
    # actuation: name -> (stage TFModel, kappa strength)
    stages: dict = field(default_factory=dict)
    # open-loop-gain shape (used to derive the servo D = G/(A·C))
    f_ugf: float = 50.0         # unity-gain frequency [Hz]
    f_hi: float = 400.0         # high-frequency control rolloff pole [Hz]
    # disturbance / sensing noise ASDs (set on the twin used for simulation)
    disturbance_asd: float = 0.0   # process (length) disturbance, [m/√Hz] referred to x_free
    sensor_asd: float = 0.0        # readout noise on d_err, [ct/√Hz]

    @classmethod
    def default(cls) -> "DARMLoop":
        stages = {
            "UIM": (_pendulum_stage(0.43, 300.0, 4.0e-7), 1.00),
            "PUM": (_pendulum_stage(1.00, 200.0, 8.0e-8), 0.40),
            "TST": (_pendulum_stage(3.40, 100.0, 1.2e-8), 0.08),
        }
        return cls(stages=stages)

    @property
    def ports(self) -> list[str]:
        return ["PCAL", "UIM", "PUM", "TST"]

    # -- elements ----------------------------------------------------------
    def C(self, freq) -> np.ndarray:
        return sensing_model(freq, self.g_c, self.f_cc, self.tau)

    def stage(self, name: str, freq) -> np.ndarray:
        tf, kappa = self.stages[name]
        return kappa * tf.eval(freq)

    def A(self, freq) -> np.ndarray:
        return sum(self.stage(n, freq) for n in self.stages)

    def _ol_shape(self, freq) -> np.ndarray:
        """The *designed* open-loop gain G(f): integrator to UGF, a control
        rolloff pole, and the sensing transport delay — shaped for a stable loop
        with healthy phase margin.  D is then derived so G = A·D·C exactly."""
        f = np.asarray(freq, dtype=float)
        return (self.f_ugf / (1j * f)) / (1.0 + 1j * f / self.f_hi) \
            * np.exp(-2j * np.pi * f * self.tau)

    def G(self, freq) -> np.ndarray:
        return self._ol_shape(freq)

    def D(self, freq) -> np.ndarray:
        """Representative digital servo, derived from the designed G: D = G/(A·C)."""
        return self.G(freq) / (self.A(freq) * self.C(freq))

    def R(self, freq) -> np.ndarray:
        """The calibration deliverable: counts→displacement response (1+G)/C."""
        return (1.0 + self.G(freq)) / self.C(freq)

    # -- closed-loop FRFs per injection point ------------------------------
    def frf_pcal(self, freq) -> np.ndarray:
        """d_err/x_pc = C/(1+G)  (Pcal displacement → DARM error)."""
        return self.C(freq) / (1.0 + self.G(freq))

    def frf_stage(self, name: str, freq) -> np.ndarray:
        """d_err/c_i = C·κ_i·N_i/(1+G)  (stage drive counts → DARM error)."""
        return self.C(freq) * self.stage(name, freq) / (1.0 + self.G(freq))

    def disturbance_to_derr(self, freq) -> np.ndarray:
        """x_free enters at the test mass like x_pc: C/(1+G)."""
        return self.frf_pcal(freq)

    def sensing_to_derr(self, freq) -> np.ndarray:
        """Readout noise n adds at d_err and is loop-suppressed: 1/(1+G)."""
        return 1.0 / (1.0 + self.G(freq))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_darm.py -x -q`
Expected: PASS (5 tests). If `test_loop_is_stable_with_margin` fails, adjust `f_hi` upward (e.g. 500) — do NOT loosen the assertion; the loop must genuinely have PM>30°.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/darm.py tests/test_darm.py
git commit -m "feat(darm): representative closed-loop DARM twin (C, A, D, G, R) + FRF identities"
```

---

## Task 2: Frequency-domain closed-loop synthesis (`DARMLoop.simulate`)

**Files:**
- Modify: `src/system_ident/darm.py` (add `simulate`)
- Test: `tests/test_darm.py`

**Interfaces:**
- Consumes: Task 1 `DARMLoop`, its `frf_pcal/frf_stage/disturbance_to_derr/sensing_to_derr`.
- Produces: `DARMLoop.simulate(drives: dict[str, np.ndarray], n: int, rng) -> np.ndarray` returning `d_err[n]`. `drives` maps a port name (`"PCAL"/"UIM"/"PUM"/"TST"`) to an injected time series (already length-`n` or shorter, zero-padded).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_darm.py (append)
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop

def _band_grid(loop, nperseg):
    fa = np.fft.rfftfreq(nperseg, 1/loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]

def test_simulate_deterministic_matches_pcal_frf():
    loop = DARMLoop.default()
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))   # flat, unit total power
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    derr = loop.simulate({"PCAL": x}, len(x), np.random.default_rng(1))
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(x, derr, loop.fs, nperseg, band, n_transient=1)
    # recovered closed-loop FRF tracks the analytic C/(1+G) on the excited bins
    good = np.isfinite(H_err)
    rel = np.abs(H[good] - loop.frf_pcal(freq)[good]) / np.abs(loop.frf_pcal(freq)[good])
    assert np.median(rel) < 1e-3

def test_disturbance_and_sensing_noise_color_differently():
    import scipy.signal as sig
    loop = DARMLoop.default()
    n = int(64 * loop.fs)
    # disturbance only
    ld = DARMLoop.default(); ld.disturbance_asd = 1e-18; ld.sensor_asd = 0.0
    yd = ld.simulate({}, n, np.random.default_rng(0))
    # sensing only
    ls = DARMLoop.default(); ls.disturbance_asd = 0.0; ls.sensor_asd = 1e-2
    ys = ls.simulate({}, n, np.random.default_rng(0))
    f, Pd = sig.welch(yd, fs=loop.fs, nperseg=int(4*loop.fs))
    _, Ps = sig.welch(ys, fs=loop.fs, nperseg=int(4*loop.fs))
    inband = (f >= loop.fmin) & (f <= loop.fmax)
    # the two noise paths have different numerators (C vs 1) -> different in-band shape
    shape_d = Pd[inband] / np.median(Pd[inband])
    shape_s = Ps[inband] / np.median(Ps[inband])
    assert np.max(np.abs(shape_d - shape_s)) > 0.3   # measurably distinct spectra
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_darm.py::test_simulate_deterministic_matches_pcal_frf -x -q`
Expected: FAIL with `AttributeError: 'DARMLoop' object has no attribute 'simulate'`.

- [ ] **Step 3: Implement `simulate`**

```python
# src/system_ident/darm.py (add to DARMLoop)
    def _white(self, asd: float, n: int, rng) -> np.ndarray:
        if asd == 0.0:
            return np.zeros(n)
        # one-sided ASD A -> discrete white-noise std A·sqrt(fs/2)
        return rng.standard_normal(n) * asd * np.sqrt(self.fs / 2.0)

    def simulate(self, drives: dict, n: int, rng) -> np.ndarray:
        """Synthesise d_err[n] for injected ``drives`` under process disturbance +
        sensing noise, by frequency-domain closed-loop filtering.

        Deterministic drives are periodic (P&S multisine), so rfft·H·irfft is the
        exact periodic steady-state response; the stochastic disturbance/sensing
        noise are coloured by their closed-loop transfer functions.
        """
        n = int(n)
        f = np.fft.rfftfreq(n, d=1.0 / self.fs)
        Y = np.zeros(len(f), dtype=complex)
        for port, x in drives.items():
            x = np.asarray(x, dtype=float)
            xf = np.zeros(n)
            xf[: min(len(x), n)] = x[: n]
            H = self.frf_pcal(f) if port == "PCAL" else self.frf_stage(port, f)
            Y += np.fft.rfft(xf) * H
        # process disturbance x_free -> d_err  (C/(1+G))
        if self.disturbance_asd:
            w = self._white(self.disturbance_asd, n, rng)
            Y += np.fft.rfft(w) * self.disturbance_to_derr(f)
        # readout/sensing noise n -> d_err  (1/(1+G))
        if self.sensor_asd:
            v = self._white(self.sensor_asd, n, rng)
            Y += np.fft.rfft(v) * self.sensing_to_derr(f)
        return np.fft.irfft(Y, n)
```

Note: `frf_pcal`/`frf_stage` evaluate at `f` including DC (`f[0]=0`); `1j*f` is 0 at DC so `G` is `inf` there → `frf_pcal(0)=0`. Guard by setting the DC bin of each `H` to 0 before multiply if any NaN appears; add `H = np.where(np.isfinite(H), H, 0.0)` inside the loop and for the noise paths.

- [ ] **Step 4: Apply the DC-bin guard, then run the tests**

Edit the three `H`/transfer evaluations in `simulate` to wrap with `np.where(np.isfinite(...), ..., 0.0)`.

Run: `pytest tests/test_darm.py -x -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/darm.py tests/test_darm.py
git commit -m "feat(darm): frequency-domain closed-loop synthesis under disturbance + sensing noise"
```

---

## Task 3: `DARMBackend` on the ChannelBackend API

**Files:**
- Create: `src/system_ident/backends/darm_adapter.py`
- Test: `tests/test_darm.py`

**Interfaces:**
- Consumes: `DARMLoop.simulate`, `ChannelBackend._soft_start_stop`.
- Produces: `DARMBackend(loop, exc_channels: dict[str,str], derr_channel: str, fs=None, seed=None, ramp_s=3.0)`. `exc_channels` maps channel name → port (`"PCAL"/"UIM"/"PUM"/"TST"`). Implements `inject(channel, ts, fs)`, `read(channels, duration)`, `ramp_down(channel, secs)`. Reading the `derr_channel` returns `loop.simulate(current_drives, n, rng)`; reading an exc channel returns its ramped drive monitor (the FRF input X). `from_config(config, loop, **kw)` classmethod.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_darm.py (append)
from system_ident.backends.darm_adapter import DARMBackend

def test_backend_recovers_pcal_frf():
    loop = DARMLoop.default(); loop.sensor_asd = 1e-3
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=2)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("PCAL_EXC", x, loop.fs)
    dur = (nperseg * nper) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], dur)
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(
        seg["PCAL_EXC"], seg["DARM_ERR"], loop.fs, nperseg, band, n_transient=1)
    good = np.isfinite(H_err)
    rel = np.abs(H[good] - loop.frf_pcal(freq)[good]) / np.abs(loop.frf_pcal(freq)[good])
    assert np.median(rel) < 5e-3

def test_backend_inject_ramps_drive():
    loop = DARMLoop.default()
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", ramp_s=3.0)
    drive = np.ones(int(20 * loop.fs))
    be.inject("PCAL_EXC", drive, loop.fs)
    mon = be.read(["PCAL_EXC"], 20.0)["PCAL_EXC"]
    assert abs(mon[0]) < 1e-9 and abs(mon[-1]) < 1e-9
    assert mon[len(mon)//2] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_darm.py::test_backend_inject_ramps_drive -x -q`
Expected: FAIL — `ModuleNotFoundError: ...darm_adapter`.

- [ ] **Step 3: Implement the backend**

```python
# src/system_ident/backends/darm_adapter.py
"""Digital-twin backend for the DARM loop: same channel API, backed by DARMLoop.

Injecting a multisine on Pcal or any actuation stage and reading the DARM error
synthesises the closed-loop response (frequency domain) under the loop's process
disturbance + sensing noise, so the existing P&S loop / FRF run unchanged.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal as sig

from ..darm import DARMLoop
from .base import ChannelBackend


class DARMBackend(ChannelBackend):
    def __init__(self, loop: DARMLoop, exc_channels: dict, derr_channel: str,
                 fs: float | None = None, seed=None, ramp_s: float = 3.0) -> None:
        self.loop = loop
        self.exc_channels = dict(exc_channels)           # channel -> port
        self.derr_channel = derr_channel
        self.fs = float(fs if fs is not None else loop.fs)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives: dict[str, np.ndarray] = {}          # port -> ramped drive

    @classmethod
    def from_config(cls, config: dict, loop: DARMLoop, **kwargs) -> "DARMBackend":
        ch = config["channels"]
        exc = {chan: port for port, chan in ch["excitation"].items()}
        derr = ch["readback"]["DARM"]
        kwargs.setdefault("ramp_s", float(config.get("measurement", {}).get("t_ramp", 3.0)))
        return cls(loop, exc, derr, fs=float(config["measurement"]["fs"]), **kwargs)

    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        ts = np.asarray(timeseries, dtype=float)
        if not np.isclose(fs, self.fs):
            frac = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, frac.numerator, frac.denominator)
        self._drives[self.exc_channels[channel]] = self._soft_start_stop(ts, self.fs)

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        n = int(round(duration * self.fs))
        out: dict[str, np.ndarray] = {}
        derr = None
        for ch in channels:
            if ch == self.derr_channel:
                if derr is None:
                    derr = self.loop.simulate(self._drives, n, self._rng)
                out[ch] = derr
            elif ch in self.exc_channels:
                out[ch] = self._fit_length(self._drives.get(self.exc_channels[ch]), n)
            else:
                raise KeyError(f"unknown channel {ch!r}")
        return out

    def ramp_down(self, channel: str, secs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        port = self.exc_channels[channel]
        drive = self._drives.get(port)
        if drive is None or len(drive) == 0:
            return
        n_ramp = min(int(round(secs * self.fs)), len(drive))
        ramped = np.zeros_like(drive)
        if n_ramp > 0:
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))
            ramped[:n_ramp] = drive[:n_ramp] * taper
        self._drives[port] = ramped

    @staticmethod
    def _fit_length(x, n):
        if x is None:
            return np.zeros(n)
        if len(x) >= n:
            return np.asarray(x[:n], dtype=float)
        out = np.zeros(n)
        out[: len(x)] = x
        return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_darm.py -x -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/backends/darm_adapter.py tests/test_darm.py
git commit -m "feat(darm): DARMBackend on the ChannelBackend API (3 s ramp, freq-domain synthesis)"
```

---

## Task 4: Recovery — response `R(f)`, sensing `C`, actuation `κ`

**Files:**
- Modify: `src/system_ident/darm.py` (add `recover_response`, `fit_sensing`, `recover_actuation`)
- Test: `tests/test_darm.py`

**Interfaces:**
- Consumes: a measured FRF `(H, H_err)` from `SysIDLoop._estimate_tf_periodic`; the known model `(1+G)` and stage shapes `N_i` from `DARMLoop`.
- Produces:
  - `recover_response(H_pcal, H_err) -> (R, R_sigma)` where `R = 1/H_pcal`, `R_sigma = H_err/|H_pcal|²` (model-free deliverable + CRB envelope).
  - `fit_sensing(freq, C_meas, C_err, p0) -> (params dict {g_c,f_cc,tau}, sigma dict)` via weighted complex least squares.
  - `recover_actuation(freq, H_stage, H_pcal, N_stage, comb_err) -> (kappa, kappa_sigma)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_darm.py (append)
from system_ident.darm import recover_response, fit_sensing, recover_actuation

def _run_pcal(loop, seed=3):
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(0))
    be.inject("PCAL_EXC", x, loop.fs)
    seg = be.read(["PCAL_EXC","DARM_ERR"], (nperseg*nper)/loop.fs)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                  loop.fs, nperseg, band, n_transient=1)
    return freq, band, H, H_err

def test_recover_response_tracks_truth():
    loop = DARMLoop.default(); loop.sensor_asd = 1e-3
    freq, band, H, H_err = _run_pcal(loop)
    R, R_sig = recover_response(H, H_err)
    good = np.isfinite(H_err)
    rel = np.abs(R[good] - loop.R(freq)[good]) / np.abs(loop.R(freq)[good])
    assert np.median(rel) < 5e-3
    assert np.all(R_sig[good] > 0)

def test_fit_sensing_recovers_pole_and_delay():
    loop = DARMLoop.default(); loop.sensor_asd = 1e-3
    freq, band, H, H_err = _run_pcal(loop)
    C_meas = H * (1.0 + loop.G(freq))            # expose C with the known (1+G)
    p, sig_ = fit_sensing(freq, C_meas, H_err*np.abs(1+loop.G(freq)),
                          p0=(0.8e6, 300.0, 50e-6))
    assert abs(p["f_cc"] - 360.0)/360.0 < 0.05
    assert abs(p["tau"] - 77e-6) < 15e-6
    assert abs(p["g_c"] - 1e6)/1e6 < 0.05

def test_recover_actuation_kappas():
    loop = DARMLoop.default(); loop.sensor_asd = 1e-3
    nperseg, nper = 4096, 8
    fa, band, freq = _band_grid(loop, nperseg)
    # Pcal reference
    freqp, _, Hp, Hp_err = _run_pcal(loop)
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    for name, true_k in (("UIM",1.00),("PUM",0.40),("TST",0.08)):
        be = DARMBackend(loop, {"EXC": name}, "DARM_ERR", seed=5)
        x = multisine_from_psd(Pxx, loop.fs, nperseg, nper, freq, seed=np.random.default_rng(1))
        be.inject("EXC", x, loop.fs)
        seg = be.read(["EXC","DARM_ERR"], (nperseg*nper)/loop.fs)
        Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(seg["EXC"], seg["DARM_ERR"],
                                                        loop.fs, nperseg, band, n_transient=1)
        tf, _ = loop.stages[name]
        N = tf.eval(freq)
        comb_err = np.hypot(Hi_err/np.abs(Hi), Hp_err/np.abs(Hp)) * np.abs(Hi/Hp)
        k, ks = recover_actuation(freq, Hi, Hp, N, comb_err)
        assert abs(k - true_k)/true_k < 0.05
        assert ks > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_darm.py::test_recover_response_tracks_truth -x -q`
Expected: FAIL — `ImportError: cannot import name 'recover_response'`.

- [ ] **Step 3: Implement the recovery functions**

```python
# src/system_ident/darm.py (append, module level)
from scipy.optimize import least_squares   # add to imports at top


def recover_response(H_pcal: np.ndarray, H_err: np.ndarray) -> tuple:
    """Model-free DARM response R = 1/(d_err/x_pc) with its CRB envelope.

    R = 1/H_pcal;  σ_R = σ_H/|H_pcal|²  (first-order propagation of the FRF error).
    Unexcited bins (non-finite H_err) get σ_R = inf.
    """
    H = np.asarray(H_pcal)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.abs(H) > 0, 1.0 / H, 0.0)
        R_sigma = np.where(np.isfinite(H_err) & (np.abs(H) > 0),
                           np.asarray(H_err) / np.abs(H) ** 2, np.inf)
    return R, R_sigma


def fit_sensing(freq, C_meas, C_err, p0) -> tuple:
    """Weighted complex least-squares fit of C(f)=g_c/(1+i f/f_cc)·e^{-i2πfτ}.

    Returns (params, sigma) dicts over {g_c, f_cc, tau}; sigma from the
    Gauss–Newton covariance (JᵀJ)⁻¹ at the solution (the CRB for white,
    correctly-weighted residuals).
    """
    f = np.asarray(freq, dtype=float)
    Cm = np.asarray(C_meas)
    good = np.isfinite(C_err) & (np.asarray(C_err) > 0) & np.isfinite(Cm)
    f, Cm, w = f[good], Cm[good], 1.0 / np.asarray(C_err)[good]

    def resid(p):
        g_c, f_cc, tau = p
        r = (sensing_model(f, g_c, f_cc, tau) - Cm) * w
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, np.asarray(p0, dtype=float), method="lm")
    params = {"g_c": sol.x[0], "f_cc": sol.x[1], "tau": sol.x[2]}
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        s = np.full(3, np.nan)
    sigma = {"g_c": s[0], "f_cc": s[1], "tau": s[2]}
    return params, sigma


def recover_actuation(freq, H_stage, H_pcal, N_stage, comb_err) -> tuple:
    """Stage strength κ_i = mean of (H_stage/H_pcal)/N_i, Pcal as the ruler.

    H_stage/H_pcal = κ_i N_i, so dividing by the known stage shape N_i yields a
    per-bin κ estimate; combine by inverse-variance over the excited bins.
    """
    ratio = np.asarray(H_stage) / np.asarray(H_pcal) / np.asarray(N_stage)
    good = np.isfinite(comb_err) & (np.asarray(comb_err) > 0) & np.isfinite(ratio)
    # error on κ per bin ≈ comb_err / |N_i|
    sig_k = np.asarray(comb_err)[good] / np.abs(np.asarray(N_stage)[good])
    w = 1.0 / sig_k ** 2
    kappa = float(np.sum(w * np.real(ratio[good])) / np.sum(w))
    kappa_sigma = float(1.0 / np.sqrt(np.sum(w)))
    return kappa, kappa_sigma
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_darm.py -x -q`
Expected: PASS (12 tests). If `fit_sensing` tolerance is tight, widen the band used or raise `n_periods`; do not loosen the assertion past the stated 5%.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/darm.py tests/test_darm.py
git commit -m "feat(darm): recover response R(f), sensing C(g,f_cc,tau), and actuation kappas"
```

---

## Task 5: Swept-sine vs multisine comparison harness

**Files:**
- Modify: `src/system_ident/darm.py` (add `multisine_response_sigma`, `swept_sine_response_sigma`)
- Test: `tests/test_darm.py`

**Interfaces:**
- Consumes: `DARMLoop`, `DARMBackend`, `SysIDLoop._estimate_tf_periodic`, `multisine_from_psd`.
- Produces:
  - `multisine_response_sigma(loop, *, nperseg, n_periods, px_total, seed) -> (freq, R, R_sigma, T_total)` — runs ONE Pcal multisine over the whole band; `T_total = nperseg*n_periods/fs`.
  - `swept_sine_response_sigma(loop, freq_points, *, nperseg, dwell_periods, px_total, seed) -> (freq_points, R_sigma, T_used)` — each frequency a single-line, **full-power**, **ramp-free** dwell of `dwell_periods` periods (≥2, so a per-bin variance can be formed; ramp-free so the swept baseline is *not* handicapped by the actuator ramp). Returns the absolute σ(R) at each point and the honest wall-clock `T_used = len(points)·dwell_periods·nperseg/fs` the sweep spends.
  - `sweep_time_to_match_coverage(loop, *, nperseg, dwell_periods) -> float` — wall-clock for a sweep to visit *every* band bin for `dwell_periods` each: the coverage the single multisine window buys in `n_periods·nperseg/fs` s.

**Why this shape (verified during planning):** at 1 Hz bins a dense (e.g. 40-point) sweep cannot fit in equal wall-clock. The honest framing is: in the SAME `T_total`, the multisine measures all ~1490 band bins at once while a sweep at `dwell_periods=2` resolves only `T_total·fs/(dwell_periods·nperseg)` ≈ 8 frequencies — and matching the multisine's full-band coverage costs the sweep ~`sweep_time_to_match_coverage/T_total` ≈ 185× longer. The twin's representative noise (`sensor_asd≈300`, `disturbance_asd≈3e-4`) is set so per-bin σ(R)/R is a visible ~1% (NOT the effectively-noise-free 1e-8 the original plan's `sensor_asd=1e-3` against `g_C=1e6` produced). Also: `n_periods=16` (not 8) so the 3 s ramp leaves ~10 full periods → `P_eff≈9` → a *genuine* per-bin variance; at `n_periods=8` only 2 full periods survive → `P_eff=1` → `H_err` collapses to the 1e-9 floor and the CRB envelope would be fabricated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_darm.py (append)
from system_ident.darm import (multisine_response_sigma, swept_sine_response_sigma,
                               sweep_time_to_match_coverage)

def test_comparison_harness_produces_both_envelopes():
    loop = DARMLoop.default(); loop.disturbance_asd = 3e-4; loop.sensor_asd = 300.0
    freq, R, R_sig, T = multisine_response_sigma(loop, nperseg=4096, n_periods=16,
                                                 px_total=1.0, seed=0)
    good = np.isfinite(R_sig)
    assert np.all(R_sig[good] > 0) and T == pytest.approx(16.0, rel=1e-6)
    # honest, visible, representative uncertainty — NOT floored to ~1e-9, not absurd
    frac = R_sig[good] / np.abs(R[good])
    assert 1e-3 < np.median(frac) < 5e-2
    # equal wall-clock: 8 points × 2 periods × 1 s = the same 16 s
    pts = np.geomspace(loop.fmin, loop.fmax, 8)
    fp, s, T_used = swept_sine_response_sigma(loop, pts, nperseg=4096, dwell_periods=2,
                                              px_total=1.0, seed=0)
    assert s.shape == pts.shape and np.all(np.isfinite(s) & (s > 0))
    assert T_used == pytest.approx(16.0, rel=1e-6)
    # covering the whole band by sweep costs far more than the one multisine window
    assert sweep_time_to_match_coverage(loop, nperseg=4096, dwell_periods=2) > 20 * T
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_darm.py::test_comparison_harness_produces_both_envelopes -x -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the harness**

```python
# src/system_ident/darm.py (append)
from .backends.darm_adapter import DARMBackend   # local import to avoid cycle; see note
from .excitation import multisine_from_psd
from .loop import SysIDLoop


def _band_grid(loop, nperseg):
    fa = np.fft.rfftfreq(int(nperseg), d=1.0 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def multisine_response_sigma(loop, *, nperseg=4096, n_periods=16, px_total=1.0, seed=0):
    """One Pcal multisine over the whole band → R(f) and its per-bin σ (CRB envelope).

    n_periods=16: with the 3 s actuator ramp this leaves ~10 full-energy periods
    (P_eff≈9), so the per-bin variance is genuinely estimated — not the floored,
    fabricated uncertainty that n_periods=8 (only 2 full periods → P_eff=1) produces.
    """
    fa, band, freq = _band_grid(loop, nperseg)
    Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    x = multisine_from_psd(Pxx, loop.fs, nperseg, n_periods, freq, seed=np.random.default_rng(seed))
    be.inject("PCAL_EXC", x, loop.fs)
    T_total = (nperseg * n_periods) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], T_total)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                  loop.fs, nperseg, band, n_transient=1)
    R, R_sigma = recover_response(H, H_err)
    return freq, R, R_sigma, T_total


def swept_sine_response_sigma(loop, freq_points, *, nperseg=4096, dwell_periods=2,
                              px_total=1.0, seed=0):
    """Idealised swept sine on the same twin: each frequency a single-line, full-power,
    **ramp-free** dwell of ``dwell_periods`` periods (≥2, so a per-bin variance can be
    formed; ramp-free so the baseline is not handicapped by the 3 s actuator ramp).

    Returns ``(freq_points, R_sigma, T_used)`` — absolute σ(R) per point and the honest
    wall-clock ``T_used = len·dwell_periods·nperseg/fs`` the sweep spends.
    """
    freq_points = np.asarray(freq_points, dtype=float)
    nperseg = int(nperseg)
    n_per = max(2, int(dwell_periods))
    fa = np.fft.rfftfreq(nperseg, d=1.0 / loop.fs)
    out = np.full(len(freq_points), np.inf)
    rng = np.random.default_rng(seed)
    for i, fpt in enumerate(freq_points):
        k = int(np.argmin(np.abs(fa - fpt)))
        Pxx = np.array([px_total / (fa[1] - fa[0])])   # all power on the one line
        band = (fa >= fa[k] - 1e-9) & (fa <= fa[k] + 1e-9)
        be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=rng, ramp_s=0.0)
        x = multisine_from_psd(Pxx, loop.fs, nperseg, n_per, np.array([fa[k]]), seed=rng)
        be.inject("PCAL_EXC", x, loop.fs)
        seg = be.read(["PCAL_EXC", "DARM_ERR"], (nperseg * n_per) / loop.fs)
        # ramp-free single tone → no transient → keep all periods (n_transient=0)
        H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                      loop.fs, nperseg, band, n_transient=0)
        sel = np.isfinite(H_err) & (np.abs(H) > 0)
        if np.any(sel):
            R, R_sigma = recover_response(H, H_err)
            out[i] = float(np.min(R_sigma[sel]))      # absolute σ(R) at the driven line
    T_used = len(freq_points) * n_per * nperseg / loop.fs
    return freq_points, out, T_used


def sweep_time_to_match_coverage(loop, *, nperseg=4096, dwell_periods=2):
    """Wall-clock for a swept sine to visit EVERY band bin for ``dwell_periods`` each —
    the full-band coverage the single multisine window gets in n_periods·nperseg/fs s."""
    _, _, freq = _band_grid(loop, nperseg)
    return len(freq) * max(2, int(dwell_periods)) * nperseg / loop.fs
```

Note on the import cycle: `darm.py` importing `backends.darm_adapter` which imports `darm` — Python handles this because the adapter only needs `DARMLoop` (defined before the bottom-of-file import runs). If an `ImportError` appears, move these three helper functions into a new `src/system_ident/darm_compare.py` that imports both. Prefer keeping them in `darm.py` with the import at the bottom (after the class) as written.

- [ ] **Step 4: Run the full test module**

Run: `pytest tests/test_darm.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Run the whole suite to confirm no regressions; commit**

```bash
pytest -q
git add src/system_ident/darm.py tests/test_darm.py
git commit -m "feat(darm): swept-sine vs multisine sigma(R) comparison on the same twin"
```

Expected: full suite green (prior 133 passed, 1 skipped, plus the new darm tests).

---

## Task 6: Presentation glue `docs/darm_demo.py`

**Files:**
- Create: `docs/darm_demo.py`
- Test: a smoke check appended to `tests/test_darm.py` (import + one campaign builds a figure).

**Interfaces:**
- Consumes: everything in `darm.py`, `sysid_plots as sp`.
- Produces campaign functions returning `SimpleNamespace`, and figure wrappers returning plotly figures:
  - `truth_bodes()` → fig (C, A stages, D, G, R magnitudes).
  - `pcal_audit(seed)` → ns with `t, drive, derr, freq, H, H_err, coh, excited, R, R_sigma, C_meas, fit, sigma`.
  - `actuation_campaign(seed)` → ns with per-stage `kappa, sigma, true`.
  - `comparison(seed)` → ns with `freq, R_sigma_ms, pts, sigma_sweep, T_total`.
  - figure wrappers: `truth_fig`, `pcal_timeseries_fig`, `pcal_bode_fig`, `response_envelope_fig`, `sensing_table`, `actuation_table`, `comparison_fig`, `fom_table`.

- [ ] **Step 1: Write the glue** (presentation only; mirrors `rtsfreerun_demo.py` structure and `sysid_plots` house style)

```python
# docs/darm_demo.py
"""Presentation-only glue for the DARM calibration example page (08).

NOT package API — the docs sibling of ``sysid_plots``. It runs the DARM twin
campaigns (Pcal response, sensing fit, actuation kappas, swept-sine comparison)
and builds the page's plotly panels in the shared house style.  Every figure is
exported to SVG (Git LFS) by the page's render.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import sysid_plots as sp  # noqa: E402
from system_ident.darm import (  # noqa: E402
    DARMLoop, recover_response, fit_sensing, recover_actuation,
    multisine_response_sigma, swept_sine_response_sigma, sweep_time_to_match_coverage,
)
from system_ident.backends.darm_adapter import DARMBackend  # noqa: E402
from system_ident.excitation import multisine_from_psd  # noqa: E402
from system_ident.loop import SysIDLoop  # noqa: E402

# NPER=16 so the 3 s ramp leaves ~10 full periods → a genuine per-bin variance
# (P_eff≈9). NPER=8 would leave only 2 full periods → P_eff=1 → fabricated CRB bars.
NPERSEG, NPER = 4096, 16


def _grid(loop):
    fa = np.fft.rfftfreq(NPERSEG, 1 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def _twin(seed=1):
    # Representative noise tuned so the per-bin σ(R)/R is a visible ~1% (the O4-era
    # cal target scale), NOT the effectively-noise-free ~1e-8 that a tiny sensor_asd
    # against g_C=1e6 gives. disturbance (length noise, via C/(1+G)) is comparable at
    # the low band; sensor (readout, via 1/(1+G)) dominates higher — a two-component floor.
    loop = DARMLoop.default()
    loop.disturbance_asd = 3.0e-4     # representative length-noise floor [m/√Hz]
    loop.sensor_asd = 300.0           # representative DARM readout noise [ct/√Hz]
    return loop


# ── campaigns ─────────────────────────────────────────────────────────────────
def pcal_audit(seed=1):
    loop = _twin()
    fa, band, freq = _grid(loop)
    px_total = 1.0
    Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed))
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    be.inject("PCAL_EXC", x, loop.fs)
    dur = (NPERSEG * NPER) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], dur)
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                    loop.fs, NPERSEG, band, n_transient=1)
    excited = np.isfinite(H_err)
    R, R_sig = recover_response(H, H_err)
    one_plus_G = 1.0 + loop.G(freq)
    C_meas = H * one_plus_G
    p, s = fit_sensing(freq, C_meas, H_err * np.abs(one_plus_G), p0=(0.8e6, 300.0, 50e-6))
    t = np.arange(len(seg["PCAL_EXC"])) / loop.fs
    ff = np.geomspace(loop.fmin, loop.fmax, 600)
    return SimpleNamespace(loop=loop, t=t, drive=seg["PCAL_EXC"], derr=seg["DARM_ERR"],
                           freq=freq, band=band, H=H, H_err=H_err, coh=coh, excited=excited,
                           R=R, R_sigma=R_sig, C_meas=C_meas, fit=p, sigma=s, ff=ff)


def actuation_campaign(seed=2):
    loop = _twin()
    fa, band, freq = _grid(loop)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))
    # Pcal reference
    bp = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    xp = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed))
    bp.inject("PCAL_EXC", xp, loop.fs)
    sp_seg = bp.read(["PCAL_EXC", "DARM_ERR"], (NPERSEG * NPER) / loop.fs)
    Hp, Hp_err, _ = SysIDLoop._estimate_tf_periodic(sp_seg["PCAL_EXC"], sp_seg["DARM_ERR"],
                                                    loop.fs, NPERSEG, band, n_transient=1)
    rows = []
    for name in ("UIM", "PUM", "TST"):
        be = DARMBackend(loop, {"EXC": name}, "DARM_ERR", seed=seed + 1)
        xi = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed + 2))
        be.inject("EXC", xi, loop.fs)
        si = be.read(["EXC", "DARM_ERR"], (NPERSEG * NPER) / loop.fs)
        Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(si["EXC"], si["DARM_ERR"],
                                                        loop.fs, NPERSEG, band, n_transient=1)
        tf, true_k = loop.stages[name]
        N = tf.eval(freq)
        comb = np.hypot(Hi_err / np.abs(Hi), Hp_err / np.abs(Hp)) * np.abs(Hi / Hp)
        k, ks = recover_actuation(freq, Hi, Hp, N, comb)
        rows.append((name, true_k, k, ks))
    return SimpleNamespace(loop=loop, rows=rows)


def comparison(seed=0):
    loop = _twin()
    freq, R, R_sig, T = multisine_response_sigma(loop, nperseg=NPERSEG, n_periods=NPER,
                                                 px_total=1.0, seed=seed)
    # equal wall-clock sweep: in the SAME T, a 2-period dwell resolves only ~T*fs/(2*nperseg)
    # frequencies (= NPER//2 = 8 points here), vs the multisine's whole-band coverage.
    n_pts = NPER // 2
    pts = np.geomspace(loop.fmin, loop.fmax, n_pts)
    fp, ssweep, T_used = swept_sine_response_sigma(loop, pts, nperseg=NPERSEG,
                                                   dwell_periods=2, px_total=1.0, seed=seed)
    t_cover = sweep_time_to_match_coverage(loop, nperseg=NPERSEG, dwell_periods=2)
    return SimpleNamespace(loop=loop, freq=freq, frac_ms=R_sig / np.abs(loop.R(freq)),
                           excited=np.isfinite(R_sig), pts=fp,
                           frac_sweep=ssweep / np.abs(loop.R(fp)),
                           T=T, T_used=T_used, t_cover=t_cover,
                           n_bins=int(np.isfinite(R_sig).sum()), n_pts=n_pts)


# ── figures (house style; data-driven y-ranges) ───────────────────────────────
def truth_fig(height=560):
    loop = _twin()
    ff = np.geomspace(loop.fmin, loop.fmax, 700)
    fig = make_subplots(rows=1, cols=1)
    series = [("|C| sensing", np.abs(loop.C(ff)), sp.SKY),
              ("|A| actuation", np.abs(loop.A(ff)), sp.GOLD),
              ("|G| open-loop", np.abs(loop.G(ff)), sp.GREEN),
              ("|R| response", np.abs(loop.R(ff)), sp.ROSE)]
    for name, y, c in series:
        fig.add_scatter(x=ff, y=y, mode="lines", name=name, line=dict(color=c, width=2.4))
    yr = sp._logy_range([y for _, y, _ in series], decades=8)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="magnitude")
    fig.update_layout(title="DARM twin — sensing C, actuation A, open-loop G, response R")
    return sp.style(fig, height=height)


def pcal_timeseries_fig(a, *, height=520):
    drive_tr = [("Pcal multisine (3 s Tukey on/off)", a.drive, sp.GOLD)]
    motion_tr = [("DARM error d_err", a.derr, sp.SKY)]
    return sp.timeseries(a.t, drive_tr, motion_tr, height=height,
                         drive_unit="x_pc drive [a.u.]", motion_unit="d_err [ct]",
                         titles=["<b>Pcal excitation</b> — the injected multisine",
                                 "<b>DARM error</b> — response under disturbance + readout noise"])


def pcal_bode_fig(a, *, height=760):
    traces = [dict(name="analytic C/(1+G)", H=a.loop.frf_pcal(a.freq), color=sp.INK, width=2.4),
              dict(name="measured FRF", H=a.H, color=sp.ROSE, mode="markers",
                   err=a.H_err, mask=a.excited)]
    return sp.bode(a.freq, traces, coh=a.coh, coh_mask=a.excited, height=height,
                   ylabel="|d_err/x_pc|")


def response_envelope_fig(a, *, height=520):
    """R(f) with its ±σ CRB envelope vs the analytic truth."""
    m = a.excited
    f, R, s = a.freq[m], np.abs(a.R[m]), a.R_sigma[m]
    fig = go.Figure()
    fig.add_scatter(x=a.ff, y=np.abs(a.loop.R(a.ff)), mode="lines",
                    name="analytic R", line=dict(color=sp.INK, width=2.4))
    fig.add_scatter(x=f, y=R, mode="markers", name="recovered R",
                    marker=dict(color=sp.GOLD, size=sp.MK_DATA),
                    error_y=dict(type="data", array=s, visible=True,
                                 color=sp._fade(sp.GOLD, 0.4), width=0, thickness=1.1))
    yr = sp._logy_range([np.abs(a.loop.R(a.ff)), R], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|R(f)|  [m/ct]")
    fig.update_layout(title="DARM response R(f) = 1/(d_err/x_pc) — recovered ± CRB vs truth")
    return sp.style(fig, height=height)


def sensing_table(a):
    p, s = a.fit, a.sigma
    rows = [["optical gain g_C [ct/m]", f"{a.loop.g_c:.4g}", f"{p['g_c']:.4g}", f"{s['g_c']:.2g}"],
            ["cavity pole f_cc [Hz]", f"{a.loop.f_cc:.2f}", f"{p['f_cc']:.2f}", f"{s['f_cc']:.2f}"],
            ["delay τ [µs]", f"{a.loop.tau*1e6:.1f}", f"{p['tau']*1e6:.1f}", f"{s['tau']*1e6:.2f}"]]
    return sp.param_table(["sensing parameter", "true", "recovered", "σ (CRB)"], rows,
                          caption="Sensing function C — representative truth vs P&S recovery")


def actuation_table(d):
    rows = [[n, f"{tk:.3f}", f"{k:.3f}", f"{ks:.2g}", f"{abs(k-tk)/tk*100:.2f}%"]
            for (n, tk, k, ks) in d.rows]
    return sp.param_table(["stage", "true κ", "recovered κ", "σ (CRB)", "|Δ|"], rows,
                          caption="Actuation strengths κ — Pcal as the absolute ruler")


def comparison_fig(c, *, height=520):
    """Equal wall-clock: the multisine's dense whole-band σ(R)/R envelope vs the few
    frequencies a swept sine resolves in the same time. The sweep points sit on (or
    below) the envelope because each spends full power on one line — but it only reaches
    `n_pts` frequencies; matching the multisine's coverage costs it `t_cover` (annotated)."""
    m = c.excited
    fig = go.Figure()
    fig.add_scatter(x=c.freq[m], y=c.frac_ms[m], mode="lines",
                    name=f"P&S multisine — all {c.n_bins} bins in one {c.T:.0f} s window",
                    line=dict(color=sp.GOLD, width=2.6))
    fig.add_scatter(x=c.pts, y=c.frac_sweep, mode="markers",
                    name=f"swept sine — {c.n_pts} points in the same {c.T_used:.0f} s",
                    marker=dict(color=sp.GRAY, size=sp.MK_BIG, symbol="x",
                                line=dict(width=1.5)))
    yr = sp._logy_range([c.frac_ms[m], c.frac_sweep], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="σ(R)/|R|")
    fig.add_annotation(x=0.5, y=1.0, xref="paper", yref="paper", yanchor="bottom",
                       showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                       text=f"same band coverage by sweep ≈ {c.t_cover/60:.0f} min "
                            f"({c.t_cover/c.T:.0f}× the one multisine window)")
    fig.update_layout(title="Fractional response uncertainty — same twin, same noise, "
                            "equal wall-clock")
    return sp.style(fig, height=height)


def fom_table(c=None):
    rows = [
        ["Frequencies per measurement", "1 (dwell)", "all band bins at once"],
        ["Leakage", "windowed / settle each point", "leakage-free (periodic)"],
        ["Noise model", "assumed / averaged", "per-bin, from period-to-period variance"],
        ["Budget allocation", "uniform / manual", "CRB-optimal (Fisher-matched)"],
        ["Loop handling", "model out the servo", "FRF cancels it (reference-based)"],
    ]
    if c is not None:
        rows.append([f"Time for full-band coverage", f"≈{c.t_cover/60:.0f} min",
                     f"{c.T:.0f} s (one window)"])
    return sp.param_table(["figure of merit", "swept sine", "P&S multisine"], rows,
                          caption="Where the multisine method differs for DARM "
                                  "(representative; the efficiency is shown above, not asserted)")
```

- [ ] **Step 2: Append the glue smoke test and run it**

```python
# tests/test_darm.py (append)
def test_glue_imports_and_builds_a_figure():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "docs"))
    dd = importlib.import_module("darm_demo")
    fig = dd.truth_fig()
    assert fig.data            # at least one trace
    a = dd.pcal_audit(seed=1)
    assert np.isfinite(a.fit["f_cc"])
```

Run: `pytest tests/test_darm.py::test_glue_imports_and_builds_a_figure -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/darm_demo.py tests/test_darm.py
git commit -m "feat(darm): presentation glue + figures for the DARM calibration page"
```

---

## Task 7: The page `08-darm-calibration.qmd` + thumbnail + render

**Files:**
- Create: `docs/examples/08-darm-calibration.qmd`
- Create: `docs/examples/thumbnails/08.svg` (generated, LFS)
- Modify (verify only): `docs/_quarto.yml`

**Interfaces:**
- Consumes: `docs/darm_demo.py` figure/campaign functions.

- [ ] **Step 1: Write the page**

````markdown
---
title: "08 — DARM calibration via Pintelon–Schoukens"
description: "The same multisine pipeline applied to the gravitational-wave readout: one Pcal multisine recovers the DARM response R(f) and the sensing function C leakage-free, per-stage drives recover the actuation strengths, and a swept sine is run head-to-head on the same twin."
image: thumbnails/08.svg
categories: [calibration, closed-loop, DARM]
freeze: true
---

The earlier examples identify suspensions. This one identifies the **gravitational-wave
readout itself** — the DARM loop a LIGO calibration group characterises every run. The
method does not change: a periodic multisine, a leakage-free reference FRF, a measured
per-bin noise model, and a Cramér–Rao envelope. Only the plant is new: a closed loop with
a sensing function `C`, three-stage actuation `A`, and a digital servo `D`, under **both**
process disturbance and readout noise.

::: {.callout-note appearance="simple"}
## Representative, not a specific interferometer
The twin uses representative Advanced-LIGO numbers — a coupled-cavity pole near 360 Hz, a
~77 µs delay, three pendulum-stage actuators, and a UGF≈50 Hz loop. It is a *demonstration*
of the measurement, not a calibration of any real instrument. The page is rendered with
`freeze: true`.
:::

```{python}
#| code-summary: "Setup — import the presentation glue"
import sys, pathlib
_p = pathlib.Path.cwd()
for _c in [_p, *_p.parents]:
    if (_c / "docs" / "darm_demo.py").exists() and str(_c / "docs") not in sys.path:
        sys.path.insert(0, str(_c / "docs"))
    if (_c / "darm_demo.py").exists() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))
import numpy as np
import darm_demo as dd
```

## The loop and what calibration delivers

DARM strain is `h = R·d_err/L` with response `R = (1+G)/C` and open-loop gain `G = A·D·C`.
The calibration job is to know `R(f)` — and, underneath it, the sensing `C` and the three
actuation strengths — to ~1 % in magnitude and ~1° in phase across the band.

```{python}
#| code-summary: "The twin's truth — C, A, G, R"
dd.truth_fig()
```

## One Pcal multisine — what you'd watch

A single periodic multisine injected at Pcal, ramped on and off over 3 s, with the DARM
error read out under the loop's real disturbance and readout noise.

```{python}
#| code-summary: "Raw Pcal measurement — drive in, d_err out"
a = dd.pcal_audit(seed=1)
dd.pcal_timeseries_fig(a)
```

The leakage-free FRF `d_err/x_pc = C/(1+G)` with its per-line coherence, on the analytic truth:

```{python}
#| code-summary: "Leakage-free FRF + coherence"
dd.pcal_bode_fig(a)
```

## The deliverable: R(f), model-free

Because `d_err/x_pc = C/(1+G)`, the response is simply its reciprocal — `R(f) = 1/FRF`, with
no model in the loop. Its CRB envelope falls straight out of the FRF's per-bin error.

```{python}
#| code-summary: "Recovered R(f) ± CRB vs truth"
dd.response_envelope_fig(a)
```

## Underneath: the sensing function C

Multiplying the FRF by the (known, designed) loop factor `(1+G)` exposes the optical sensing
`C`; a weighted complex fit returns the optical gain, the coupled-cavity pole, and the delay.

```{python}
#| code-summary: "Sensing fit — g_C, f_cc, τ"
from IPython.display import display
display(dd.sensing_table(a))
```

## The actuation strengths, Pcal as the ruler

A Pcal-only measurement cannot separate the three stages (they live inside `G` as a product).
Driving each stage in turn and dividing by the Pcal FRF removes `C/(1+G)`, leaving `κ_i N_i`
referenced to the absolute Pcal meter — so each `κ` comes back on its own.

```{python}
#| code-summary: "Recover κ_U / κ_PU / κ_T"
d = dd.actuation_campaign(seed=2)
display(dd.actuation_table(d))
```

## Head-to-head with the swept sine

Same twin, same disturbance + readout noise, same total wall-clock. The multisine measures
every bin at once; the swept sine spends the clock one frequency at a time, so in equal time
it reaches only a handful of frequencies (each tight, because it spends full power there) —
and covering the whole band to the same uncertainty costs it ~100× longer. Shown on the same
twin, not asserted.

```{python}
#| code-summary: "σ(R)/|R| — multisine (whole band) vs swept sine (few points), equal wall-clock"
c = dd.comparison(seed=0)
dd.comparison_fig(c)
```

```{python}
#| code-summary: "Where the methods differ"
display(dd.fom_table(c))
```

## Honest gaps

This twin is a single DARM loop with a one-pole sensing function and static strengths. It
omits the optical spring / SRC detuning, time-dependent coefficients (TDCF tracking lines),
and the cross-couplings of a real interferometer. The efficiency claim above is a
*demonstration on a twin*; the real test is to run both injections on the same instrument
state and compare the `σ(R(f))`-per-hour envelopes — see `notes/darm-calibration-via-pns.md`.
````

- [ ] **Step 2: Generate the thumbnail** (reuse `truth_fig`, export SVG)

```bash
cd docs && python -c "import darm_demo as dd; dd.truth_fig().write_image('examples/thumbnails/08.svg')" && cd ..
```
Expected: `docs/examples/thumbnails/08.svg` created. (Requires `kaleido`; if missing, STOP and ask before installing.)

- [ ] **Step 3: Confirm thumbnail and any new SVGs are LFS-tracked**

```bash
git check-attr filter -- docs/examples/thumbnails/08.svg
```
Expected: `filter: lfs`. If not, verify `.gitattributes` covers `docs/**/*.svg` (it should from the SVG+LFS rule); add the pattern if missing.

- [ ] **Step 4: Render the page locally (freeze) and eyeball every figure**

```bash
cd docs && quarto render examples/08-darm-calibration.qmd && cd ..
```
Expected: renders without error; `docs/_freeze/examples/08-darm-calibration/` created. **Open the rendered HTML and verify every trace is visible and on-scale** (the data-driven y-ranges must not clip the response, the FRF, or the comparison envelopes). If any trace is clipped, fix the `_logy_range` decades/inputs in `darm_demo.py` against the actual arrays — never ship a clipped plot.

- [ ] **Step 5: Verify the examples nav picks up page 08**

```bash
grep -n "examples/\*\.qmd\|08-darm" docs/_quarto.yml
```
Expected: the `examples/*.qmd` glob (line ~10) covers it; no manual nav edit needed. If examples are listed explicitly elsewhere, add `08-darm-calibration.qmd` in order.

- [ ] **Step 6: Commit the page, freeze cache, thumbnail, and figures**

```bash
git add docs/examples/08-darm-calibration.qmd docs/examples/thumbnails/08.svg docs/_freeze/examples/08-darm-calibration
git add docs/_quarto.yml 2>/dev/null || true
git commit -m "docs(darm): Example 08 — DARM calibration via P&S (executable twin)"
```

---

## Task 8: Final suite + push

- [ ] **Step 1: Full test suite**

Run: `pytest -q`
Expected: all green (prior 133 passed, 1 skipped + the new `tests/test_darm.py`). If anything fails, fix before pushing.

- [ ] **Step 2: Push to main**

```bash
git push
```

- [ ] **Step 3: Update memory index pointer (optional)** — if a DARM-page memory is warranted, add one line to `MEMORY.md`; otherwise skip (the spec/plan in `docs/superpowers/` is the durable record).

---

## Self-review (author, fresh eyes)

**Spec coverage:**
- §2 twin (C cavity-pole+delay, 3-stage A, derived servo D, two noise sources) → Tasks 1–2. ✓
- §3 Pcal campaign (C, R) + per-stage campaign (κ, Pcal ruler) → Task 4 (`recover_response`, `fit_sensing`, `recover_actuation`). ✓
- §4 swept-sine head-to-head, same twin/noise/wall-clock → Task 5 + `comparison_fig`/`fom_table`. ✓
- §5 code layout (`darm.py`, `darm_adapter.py`, `darm_demo.py`, `test_darm.py`, `08-*.qmd`) → Tasks 1–7. ✓
- §6 page sections 1–8 → Task 7 page (loop, truth, raw, FRF, R envelope, sensing, κ, comparison, gaps). ✓
- §7 tests (self-consistency, noise coloring, recovery, comparison) → Tasks 1–5 tests. ✓
- §9 hard rules (SVG+LFS, data-driven y-limits, no Schroeder, trunk-based) → Global Constraints + Task 7 steps 3–4. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; tolerances are concrete numbers. ✓

**Type/name consistency:** `frf_pcal`/`frf_stage`/`disturbance_to_derr`/`sensing_to_derr`/`simulate` defined in Tasks 1–2 and used verbatim in Tasks 3–6; `recover_response`/`fit_sensing`/`recover_actuation`/`multisine_response_sigma`/`swept_sine_response_sigma` defined in Tasks 4–5 and imported by name in Task 6; `DARMBackend(loop, exc_channels, derr_channel, ...)` signature consistent across Tasks 3, 5, 6. ✓

**Known risk to watch during execution:** the import-cycle note in Task 5 (resolve by `darm_compare.py` if needed) and the `kaleido`/SVG-export dependency in Task 7 (ask before installing if absent).
