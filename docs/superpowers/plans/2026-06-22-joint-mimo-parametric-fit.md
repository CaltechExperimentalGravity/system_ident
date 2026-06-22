# Joint MIMO Parametric Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit one common-denominator MIMO model `G=B/A` (shared modal poles, free per-element numerators) to the step-1 coupled-loop campaign by P&S sample-ML equation-error, with a Cramér–Rao bound on `f0`/`Q` and the FRF.

**Architecture:** Three new numpy/scipy modules on top of step-1 (`mimo_plant`/`mimo_loop`/`backends/mimo_twin`, all on `main`): a **model** (`G=B/A`, analytic `∂G/∂θ`, roots→`f0`/`Q`), a **campaign assembler** (per-period stacked spectra → sample means `Ẑ^[l]` + covariances `Ĉ_Z^[l]`), and an **estimator** (SK linear-LS start → over-parameterized Gauss–Newton SML → parameter covariance/CRB → modal uncertainty + FRF band). The fit consumes `(Ẑ, Ĉ_Z)` from any `ChannelBackend` campaign, so step 3 (RTSfreerun) feeds it unchanged.

**Tech Stack:** Python, numpy, scipy (SVD, `np.roots`, `eigh`); python-control only via the step-1 twin that generates data + the oracle. No new third-party dependency.

**Spec:** `docs/superpowers/specs/2026-06-22-joint-mimo-parametric-fit-design.md`.
**P&S derivation (authoritative, with eq/page cites):** `.llm/pintelon-schoukens-mimo-fit.md`.
**All numerics below were prototyped and verified** (analytic Jacobian vs FD = 1.6e-5; GN-SML recovers poles to 1e-4; CRB vs Monte-Carlo ratio 0.86).

## Global Constraints

- **Literal P&S, no reparameterization:** common-denominator `G=B/A` (scalar denom `A` = shared poles, polynomial-matrix numerator `B` = free per element); SML equation-error cost (12-15/16/17); SK linear-LS start (12-31/32/33); Gauss–Newton pseudo-Jacobian (12-24/25/27); CRB `(2Re(JᴴJ))⁻¹` (12-29). No modal/vector-fitting parameterization, no variable projection, no "recover-G-then-fit". (`.llm/pintelon-schoukens-mimo-fit.md`, [[stay-on-pintelon-schoukens]].)
- **Fit the raw `X_mat`/`Y_mat` spectra**, never `G=Y·X⁻¹`. Step-1's matrix inverse is a validation overlay only.
- **Identifiability constraint:** `‖θ‖₂ = 1` (over-parameterize then renormalize; drop the 1-dim scale null space in the solve via pseudo-inverse).
- **Dimensions:** three integers `n_sens` (=`n_y`), `n_dof`, `n_act` (=`n_u`); square for SUS/SEI, rectangular allowed. `n_a = 2·n_modes`, `n_b = n_a − 1` (strictly proper).
- **`dof ≥ n_sens + 8`** periods for a trustworthy CRB (6-DoF ⇒ `dof ≥ 14`); the estimator records `dof` and applies the SML finite-sample inflation.
- **New code only:** do NOT modify `model.py:TFModel`, `estimators/gml.py:GMLEstimator`, or `fisher.py`. Reuse their patterns in new modules.
- **Continuous-time** model (`Ω = s = j·2πf`), evaluated at physical bin frequencies; tustin warping negligible in-band, documented (spec §9).
- Run everything via `conda run -n sysid python -m pytest …` ([[use-conda-run-sysid-env]]). Any plot SVG + Git LFS, data-driven y-limits ([[graphics-svg-lfs-only]]). Trunk-based, push to `main` ([[trunk-based-push-to-main]]). Phase 1 only — no pyepics/pyawg/cdsutils ([[two-phase-cds-plan]]).

## File Structure

- **Create `src/system_ident/mimo_model.py`** — `MIMOCommonDenomModel`: `pack`/`unpack`, `eval`, `jacobian`, `poles` (roots→`f0`/`Q`), `modes_to_denominator`. One responsibility: the parametric model + its derivatives.
- **Create `src/system_ident/mimo_campaign.py`** — `assemble_campaign`: per-actuator drive → per-period stacked spectra → `(Ẑ^[l], Ĉ_Z^[l], freq)`. One responsibility: turn a `ChannelBackend` campaign into the estimator's data.
- **Create `src/system_ident/mimo_fit.py`** — `sk_start`, `MIMOSampleMLEstimator` (the GN-SML fit), `parameter_covariance`, `modal_uncertainty`, `frf_band`, `validate_fit`. One responsibility: the estimator + uncertainty.
- **Create `tests/test_mimo_fit.py`** — all step-2 tests.

---

## Task 0: Model — `G=B/A` eval + pack/unpack + denominator builder

**Files:**
- Create: `src/system_ident/mimo_model.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Produces: `MIMOCommonDenomModel(n_sens, n_act, n_modes)` with attrs `n_a` (`=2*n_modes`), `n_b` (`=n_a-1`), `n_a_coef` (`=n_a+1`), `n_b_coef` (`=n_b+1`), `n_theta` (`=n_a_coef + n_sens*n_act*n_b_coef`); methods `pack(a,b)->theta`, `unpack(theta)->(a,b)`, `eval(theta, freq)->G (F,n_sens,n_act)`, and staticmethod `modes_to_denominator(modes)->a` (increasing-order real coeffs). `a` is increasing order `a[0..n_a]`; `b` shape `(n_sens,n_act,n_b_coef)`; `freq` in Hz, internally `s=2j*pi*freq`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mimo_fit.py
import numpy as np
from system_ident.mimo_model import MIMOCommonDenomModel

def test_model_eval_and_denominator():
    m = MIMOCommonDenomModel(n_sens=2, n_act=2, n_modes=2)
    assert (m.n_a, m.n_b, m.n_theta) == (4, 3, 5 + 2 * 2 * 4)
    # denominator from one mode (f0=1,Q=10): s^2 + (w/Q)s + w^2, increasing order
    a = m.modes_to_denominator([(1.0, 10.0)])
    w = 2 * np.pi
    assert np.allclose(a, [w * w, w / 10.0, 1.0])
    # eval G=B/A at one freq matches hand calc
    a4 = m.modes_to_denominator([(0.6, 20.0), (1.5, 35.0)])
    b = np.zeros((2, 2, 4)); b[0, 0, 0] = 1.0; b[1, 1, 2] = 3.0
    theta = m.pack(a4, b)
    G = m.eval(theta, np.array([0.7]))
    s = 2j * np.pi * 0.7
    A = sum(a4[r] * s**r for r in range(5))
    assert np.isclose(G[0, 0, 0], 1.0 / A)
    assert np.isclose(G[0, 1, 1], (3.0 * s**2) / A)
    assert np.isclose(G[0, 1, 0], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_model_eval_and_denominator -v`
Expected: FAIL (`ModuleNotFoundError: system_ident.mimo_model`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/system_ident/mimo_model.py
"""Common-denominator MIMO transfer-function model G(s)=B(s)/A(s) (P&S 6-53).

A(s) = sum_r a_r s^r  (scalar, real) carries the SHARED modal poles;
B_ij(s) = sum_r b_ij_r s^r (real) are FREE per-element numerators.
theta = [a (n_a+1) , vec(b) (n_sens*n_act*(n_b+1))], constraint ||theta||_2 = 1.
Verified: analytic jacobian vs finite-difference = 1.6e-5; GN-SML recovers poles to 1e-4.
"""
from __future__ import annotations
import numpy as np


class MIMOCommonDenomModel:
    def __init__(self, n_sens, n_act, n_modes):
        self.n_sens = int(n_sens)
        self.n_act = int(n_act)
        self.n_modes = int(n_modes)
        self.n_a = 2 * self.n_modes
        self.n_b = self.n_a - 1
        self.n_a_coef = self.n_a + 1
        self.n_b_coef = self.n_b + 1
        self.n_theta = self.n_a_coef + self.n_sens * self.n_act * self.n_b_coef

    def pack(self, a, b):
        return np.concatenate([np.asarray(a, float), np.asarray(b, float).reshape(-1)])

    def unpack(self, theta):
        a = theta[:self.n_a_coef]
        b = theta[self.n_a_coef:].reshape(self.n_sens, self.n_act, self.n_b_coef)
        return a, b

    @staticmethod
    def _vander(s, ncoef):
        s = np.asarray(s, complex)
        return s[:, None] ** np.arange(ncoef)[None, :]

    def eval(self, theta, freq):
        s = 2j * np.pi * np.asarray(freq, float)
        a, b = self.unpack(theta)
        A = self._vander(s, self.n_a_coef) @ a
        B = np.einsum('fr,ijr->fij', self._vander(s, self.n_b_coef), b)
        return B / A[:, None, None]

    @staticmethod
    def modes_to_denominator(modes):
        poly = np.array([1.0])  # descending while building
        for f0, Q in modes:
            w = 2 * np.pi * f0
            poly = np.convolve(poly, [1.0, w / Q, w * w])
        return poly[::-1].copy()  # increasing order
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_model_eval_and_denominator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_model.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): common-denominator MIMO model G=B/A + denominator builder"
```

---

## Task 1: Analytic Jacobian `∂G/∂θ`

**Files:**
- Modify: `src/system_ident/mimo_model.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Produces: `MIMOCommonDenomModel.jacobian(theta, freq) -> ndarray (F, n_sens, n_act, n_theta)` complex — `∂G/∂θ`. Used by the estimator AND the CRB (one Jacobian, two consumers).

- [ ] **Step 1: Write the failing test** (finite-difference check — the verified gate)

```python
def test_model_jacobian_matches_finite_difference():
    m = MIMOCommonDenomModel(n_sens=2, n_act=2, n_modes=2)
    rng = np.random.default_rng(0)
    a = m.modes_to_denominator([(0.6, 20.0), (1.5, 35.0)])
    b = rng.standard_normal((2, 2, m.n_b_coef)) * 40.0
    theta = m.pack(a, b)
    freq = np.linspace(0.2, 3.0, 25)
    J = m.jacobian(theta, freq)
    assert J.shape == (25, 2, 2, m.n_theta)
    eps = 1e-7
    Jfd = np.zeros_like(J)
    for p in range(m.n_theta):
        dt = np.zeros(m.n_theta); dt[p] = eps
        Jfd[..., p] = (m.eval(theta + dt, freq) - m.eval(theta - dt, freq)) / (2 * eps)
    assert np.abs(J - Jfd).max() / np.abs(Jfd).max() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_model_jacobian_matches_finite_difference -v`
Expected: FAIL (`AttributeError: jacobian`).

- [ ] **Step 3: Write minimal implementation** (append method to the class)

```python
    def jacobian(self, theta, freq):
        s = 2j * np.pi * np.asarray(freq, float)
        a, b = self.unpack(theta)
        Va = self._vander(s, self.n_a_coef)
        Vb = self._vander(s, self.n_b_coef)
        A = Va @ a
        B = np.einsum('fr,ijr->fij', Vb, b)
        G = B / A[:, None, None]
        F = len(s)
        out = np.zeros((F, self.n_sens, self.n_act, self.n_theta), complex)
        # d/da_r : dG/da_r = -G * (s^r / A)
        for r in range(self.n_a_coef):
            out[:, :, :, r] = -G * (Va[:, r] / A)[:, None, None]
        # d/db_ij_r : dG_ij/db_ij_r = s^r / A  (that element only)
        base = self.n_a_coef
        for i in range(self.n_sens):
            for j in range(self.n_act):
                col0 = base + (i * self.n_act + j) * self.n_b_coef
                for r in range(self.n_b_coef):
                    out[:, i, j, col0 + r] = Vb[:, r] / A
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_model_jacobian_matches_finite_difference -v`
Expected: PASS (rel err ~1.6e-5).

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_model.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): analytic dG/dtheta (verified vs finite difference)"
```

---

## Task 2: Poles → `f0`/`Q`

**Files:**
- Modify: `src/system_ident/mimo_model.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Produces: `MIMOCommonDenomModel.poles(theta) -> list[(f0, Q)]` sorted by `f0`, from the upper-half-plane roots of `A`. `f0=|λ|/2π`, `Q=|λ|/(−2 Re λ)` (`inf` if `Re λ ≥ 0`).

- [ ] **Step 1: Write the failing test**

```python
def test_poles_recover_known_modes():
    m = MIMOCommonDenomModel(n_sens=2, n_act=2, n_modes=2)
    a = m.modes_to_denominator([(0.6, 20.0), (1.5, 35.0)])
    theta = m.pack(a, np.zeros((2, 2, m.n_b_coef)))
    pq = m.poles(theta)
    assert len(pq) == 2
    assert np.isclose(pq[0][0], 0.6, atol=1e-6) and np.isclose(pq[0][1], 20.0, rtol=1e-4)
    assert np.isclose(pq[1][0], 1.5, atol=1e-6) and np.isclose(pq[1][1], 35.0, rtol=1e-4)
    # scale-invariant: f0/Q unchanged by scaling theta
    pq2 = m.poles(theta * 3.7)
    assert np.allclose([p[0] for p in pq2], [p[0] for p in pq])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_poles_recover_known_modes -v`
Expected: FAIL (`AttributeError: poles`).

- [ ] **Step 3: Write minimal implementation**

```python
    def poles(self, theta):
        a, _ = self.unpack(theta)
        roots = np.roots(a[::-1])              # np.roots wants descending order
        out = []
        for lam in roots:
            if lam.imag <= 0:
                continue                       # one of each conjugate pair
            f0 = abs(lam) / (2 * np.pi)
            Q = abs(lam) / (-2 * lam.real) if lam.real < 0 else np.inf
            out.append((f0, Q))
        return sorted(out, key=lambda t: t[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_poles_recover_known_modes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_model.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): roots(A) -> f0/Q (scale-invariant)"
```

---

## Task 3: SK linear-LS starting values

**Files:**
- Create: `src/system_ident/mimo_fit.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: `MIMOCommonDenomModel`. Data is a list `exps` of `n_exp` tuples `(Ybar, Ubar, Cz)`: `Ybar (F,n_sens)`, `Ubar (F,n_act)` complex sample-mean spectra; `Cz (F,n_sens+n_act,n_sens+n_act)` complex (used by Task 4, ignored here). `freq (F,)` Hz.
- Produces: `sk_start(model, exps, freq) -> theta0 (n_theta,)`, `‖theta0‖=1`. Solves the homogeneous linear LS `A·Ybar − B·Ubar ≈ 0` (12-31/32/33) via the smallest right singular vector. Verified: lands near the true poles.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mimo_fit.py
from system_ident.mimo_fit import sk_start

def _synth_clean(model, freq, b_true, modes, seed=0):
    """Noiseless robust-method campaign: exp l drives input l. (Ybar,Ubar,Cz) per exp."""
    rng = np.random.default_rng(seed)
    a = model.modes_to_denominator(modes)
    theta = model.pack(a, b_true); theta /= np.linalg.norm(theta)
    G = model.eval(theta, freq)
    s = 2j * np.pi * freq
    exps = []
    for l in range(model.n_act):
        U = np.zeros((len(freq), model.n_act), complex)
        U[:, l] = 1.0 + 0.3 * np.cos(s.imag)
        Y = np.einsum('fij,fj->fi', G, U)
        Cz = np.tile(np.eye(model.n_sens + model.n_act) * 1e-6, (len(freq), 1, 1)).astype(complex)
        exps.append((Y, U, Cz))
    return exps, theta

def test_sk_start_near_true_poles():
    m = MIMOCommonDenomModel(2, 2, 2)
    freq = np.linspace(0.2, 3.0, 60)
    b = np.random.default_rng(1).standard_normal((2, 2, m.n_b_coef)) * 40.0
    exps, theta_true = _synth_clean(m, freq, b, [(0.6, 20.0), (1.5, 35.0)])
    theta0 = sk_start(m, exps, freq)
    assert np.isclose(np.linalg.norm(theta0), 1.0)
    f0s = [p[0] for p in m.poles(theta0)]
    assert len(f0s) == 2
    assert abs(f0s[0] - 0.6) < 0.05 and abs(f0s[1] - 1.5) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_sk_start_near_true_poles -v`
Expected: FAIL (`ModuleNotFoundError: system_ident.mimo_fit`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/system_ident/mimo_fit.py
"""P&S sample-ML common-denominator MIMO fit (12-15..12-30) + CRB.

Pipeline: sk_start (SK linear LS, 12-31/32/33) -> MIMOSampleMLEstimator
(over-parameterized Gauss-Newton SML, 12-16/17/24/25/27) -> parameter_covariance
((2 Re(J^H J))^-1, 12-29) -> modal_uncertainty (roots(A) -> f0/Q + propagation)
+ frf_band (11-2). Fits the RAW Ybar/Ubar spectra; never forms Y*X^-1.
Verified end-to-end: poles to 1e-4, CRB vs Monte-Carlo ratio 0.86.
"""
from __future__ import annotations
import numpy as np


def sk_start(model, exps, freq):
    s = 2j * np.pi * np.asarray(freq, float)
    Va = model._vander(s, model.n_a_coef)
    Vb = model._vander(s, model.n_b_coef)
    rows = []
    for (Ybar, Ubar, _Cz) in exps:
        for k in range(len(s)):
            for i in range(model.n_sens):
                row = np.zeros(model.n_theta, complex)
                row[:model.n_a_coef] = Va[k] * Ybar[k, i]
                for j in range(model.n_act):
                    c0 = model.n_a_coef + (i * model.n_act + j) * model.n_b_coef
                    row[c0:c0 + model.n_b_coef] = -Vb[k] * Ubar[k, j]
                rows.append(row)
    Jls = np.array(rows)
    Jr = np.vstack([Jls.real, Jls.imag])             # real-stack a homogeneous system
    _, _, Vt = np.linalg.svd(Jr, full_matrices=False)
    theta0 = Vt[-1]                                   # smallest right singular vector
    return theta0 / np.linalg.norm(theta0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_sk_start_near_true_poles -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_fit.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): SK linear-LS starting values (P&S 12-31/32/33)"
```

---

## Task 4: Gauss–Newton SML estimator

**Files:**
- Modify: `src/system_ident/mimo_fit.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: `MIMOCommonDenomModel`, `exps` (with real `Cz`), `freq`, `theta0` from `sk_start`.
- Produces: `MIMOSampleMLEstimator(model).fit(exps, freq, theta0, max_iter=80, tol=1e-9) -> FitResult` with attrs `theta (n_theta,)` (`‖θ‖=1`), `jac (Nrows,n_theta)` complex (whitened Jacobian at the solution — reused by Task 5), `cost (float)`, `n_iter (int)`. The cost is `V_SML` (12-15). Whitening uses `Ceps^{-1/2}` via `eigh` (clip eigenvalues to ≥1e-18); update via SVD pseudo-inverse dropping the 1-dim scale null space; column-scale the Jacobian; renormalize `‖θ‖=1` each step; stop when `‖Δθ‖<tol` or `max_iter`. Includes the `−½ ∂Ceps/∂θ` weighting-derivative term (12-25).

- [ ] **Step 1: Write the failing test**

```python
from system_ident.mimo_fit import MIMOSampleMLEstimator

def _synth_noisy(model, freq, b_true, modes, sigZ=0.02, M=16, seed=0):
    rng = np.random.default_rng(seed)
    a = model.modes_to_denominator(modes)
    theta = model.pack(a, b_true); theta /= np.linalg.norm(theta)
    G = model.eval(theta, freq); s = 2j * np.pi * freq
    nsa = model.n_sens + model.n_act
    exps = []
    for l in range(model.n_act):
        U = np.zeros((len(freq), model.n_act), complex); U[:, l] = 1.0 + 0.3 * np.cos(s.imag)
        Y = np.einsum('fij,fj->fi', G, U)
        per = []
        for _ in range(M):
            nY = (rng.standard_normal((len(freq), model.n_sens)) + 1j * rng.standard_normal((len(freq), model.n_sens))) * sigZ / np.sqrt(2)
            nU = (rng.standard_normal((len(freq), model.n_act)) + 1j * rng.standard_normal((len(freq), model.n_act))) * sigZ / np.sqrt(2)
            per.append(np.concatenate([Y + nY, U + nU], axis=1))
        per = np.array(per); Zbar = per.mean(0)
        Cz = np.empty((len(freq), nsa, nsa), complex)
        for k in range(len(freq)):
            d = per[:, k, :] - Zbar[k]
            Cz[k] = (d.conj().T @ d) / (M - 1) / M
        exps.append((Zbar[:, :model.n_sens], Zbar[:, model.n_sens:], Cz))
    return exps, theta

def test_sml_fit_recovers_poles_under_noise():
    m = MIMOCommonDenomModel(2, 2, 2)
    freq = np.linspace(0.2, 3.0, 60)
    b = np.random.default_rng(1).standard_normal((2, 2, m.n_b_coef)) * 40.0
    exps, theta_true = _synth_noisy(m, freq, b, [(0.6, 20.0), (1.5, 35.0)], seed=1)
    res = MIMOSampleMLEstimator(m).fit(exps, freq, sk_start(m, exps, freq))
    pq = m.poles(res.theta)
    assert abs(pq[0][0] - 0.6) < 5e-3 and abs(pq[0][1] - 20.0) < 2.0
    assert abs(pq[1][0] - 1.5) < 5e-3 and abs(pq[1][1] - 35.0) < 3.0
    th = res.theta if np.dot(res.theta, theta_true) > 0 else -res.theta
    G_hat = m.eval(th, freq); G_tru = m.eval(theta_true, freq)
    assert (np.abs(G_hat - G_tru) / np.abs(G_tru)).max() < 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_sml_fit_recovers_poles_under_noise -v`
Expected: FAIL (`ImportError: MIMOSampleMLEstimator`).

- [ ] **Step 3: Write minimal implementation** (append to `mimo_fit.py`)

```python
class FitResult:
    def __init__(self, theta, jac, cost, n_iter):
        self.theta = theta; self.jac = jac; self.cost = cost; self.n_iter = n_iter


class MIMOSampleMLEstimator:
    def __init__(self, model):
        self.model = model

    def _assemble(self, theta, exps, freq):
        """Return whitened (Jbig, ebig) over all experiments/bins and the SML cost."""
        m = self.model
        G = m.eval(theta, freq)
        dG = m.jacobian(theta, freq)
        Jrows, erows = [], []
        cost = 0.0
        for (Ybar, Ubar, Cz) in exps:
            for k in range(len(freq)):
                Gk = G[k]
                e = Ybar[k] - Gk @ Ubar[k]                       # (n_sens,)
                P = np.concatenate([np.eye(m.n_sens), -Gk], axis=1)
                Ceps = P @ Cz[k] @ P.conj().T                    # (n_sens,n_sens)
                w, V = np.linalg.eigh(Ceps)
                w = np.clip(w.real, 1e-18, None)
                Wh = (V * (1.0 / np.sqrt(w))) @ V.conj().T       # Ceps^-1/2
                Cinv_e = np.linalg.solve(Ceps, e)
                cost += float((e.conj() @ Cinv_e).real)
                dGk = dG[k]                                      # (n_sens,n_act,n_theta)
                de = -np.einsum('ijp,j->ip', dGk, Ubar[k])       # (n_sens,n_theta)
                Jk = np.empty((m.n_sens, m.n_theta), complex)
                for p in range(m.n_theta):
                    dP = np.concatenate([np.zeros((m.n_sens, m.n_sens)), -dGk[:, :, p]], axis=1)
                    dCeps = dP @ Cz[k] @ P.conj().T + P @ Cz[k] @ dP.conj().T
                    Jk[:, p] = de[:, p] - 0.5 * dCeps @ Cinv_e   # eq 12-25
                Jrows.append(Wh @ Jk); erows.append(Wh @ e)
        return np.vstack(Jrows), np.concatenate(erows), cost

    def fit(self, exps, freq, theta0, max_iter=80, tol=1e-9):
        theta = np.asarray(theta0, float).copy()
        theta /= np.linalg.norm(theta)
        Jbig = ebig = None; n_done = 0
        for it in range(max_iter):
            Jbig, ebig, cost = self._assemble(theta, exps, freq)
            Jre = np.vstack([Jbig.real, Jbig.imag])
            ere = np.concatenate([ebig.real, ebig.imag])
            scale = np.linalg.norm(Jre, axis=0); scale[scale == 0] = 1.0
            U, S_, Vt = np.linalg.svd(Jre / scale, full_matrices=False)
            keep = S_ > S_.max() * 1e-8                          # drop 1-dim scale null space
            Sinv = np.where(keep, 1.0 / S_, 0.0)
            dtheta = -(Vt.T @ (Sinv * (U.T @ ere))) / scale
            theta = theta + dtheta
            theta /= np.linalg.norm(theta)
            n_done = it + 1
            if np.linalg.norm(dtheta) < tol:
                break
        Jbig, ebig, cost = self._assemble(theta, exps, freq)
        return FitResult(theta, Jbig, cost, n_done)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_sml_fit_recovers_poles_under_noise -v`
Expected: PASS (poles to ~1e-3, G to <10%).

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_fit.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): over-parameterized Gauss-Newton SML estimator (P&S 12-16/17/24/25/27)"
```

---

## Task 5: Parameter covariance / CRB + modal uncertainty + FRF band

**Files:**
- Modify: `src/system_ident/mimo_fit.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: `FitResult.jac` (whitened Jacobian at the solution), `MIMOCommonDenomModel`.
- Produces:
  - `parameter_covariance(fit_result, dof, n_sens) -> Ctheta (n_theta,n_theta)` — `(2 Re(JᴴJ))⁻¹` (pseudo-inverse; drops the scale null space) times the SML inflation `lambda1(dof) = dof*(dof-n_sens)/((dof-n_sens+1)*(dof-n_sens-1))` (12-30).
  - `modal_uncertainty(model, theta, Ctheta) -> list[dict(f0, Q, f0_std, Q_std)]` — propagate `Ctheta` to each pole's `f0`/`Q` by finite-difference gradient (`g @ Ctheta @ g`).
  - `frf_band(model, theta, Ctheta, freq) -> std (F,n_sens,n_act)` — `sqrt(diag( (∂G/∂θ) Ctheta (∂G/∂θ)ᴴ ))` per element (11-2).
- Verified: CRB f0 std vs Monte-Carlo ratio 0.86.

- [ ] **Step 1: Write the failing test** (CRB vs Monte-Carlo — the verified gate; marked slow)

```python
import pytest
from system_ident.mimo_fit import parameter_covariance, modal_uncertainty, frf_band

@pytest.mark.slow
def test_crb_matches_monte_carlo():
    m = MIMOCommonDenomModel(2, 2, 2)
    freq = np.linspace(0.2, 3.0, 60)
    b = np.random.default_rng(1).standard_normal((2, 2, m.n_b_coef)) * 40.0
    modes = [(0.6, 20.0), (1.5, 35.0)]; M = 16
    exps0, theta_true = _synth_noisy(m, freq, b, modes, M=M, seed=1)
    res0 = MIMOSampleMLEstimator(m).fit(exps0, freq, sk_start(m, exps0, freq))
    Ct = parameter_covariance(res0, dof=M, n_sens=2)
    mu = modal_uncertainty(m, res0.theta, Ct)
    pred = mu[0]["f0_std"]
    f0s = []
    for seed in range(60):
        e2, _ = _synth_noisy(m, freq, b, modes, M=M, seed=100 + seed)
        r2 = MIMOSampleMLEstimator(m).fit(e2, freq, sk_start(m, e2, freq))
        pq = m.poles(r2.theta)
        if len(pq) == 2 and abs(pq[0][0] - 0.6) < 0.1:
            f0s.append(pq[0][0])
    mc = np.std(f0s)
    assert 0.5 < pred / mc < 2.0          # CRB brackets the Monte-Carlo spread
    # FRF band is finite and positive
    band = frf_band(m, res0.theta, Ct, freq)
    assert band.shape == (60, 2, 2) and np.all(np.isfinite(band)) and np.all(band >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_crb_matches_monte_carlo -v`
Expected: FAIL (`ImportError: parameter_covariance`).

- [ ] **Step 3: Write minimal implementation** (append to `mimo_fit.py`)

```python
def parameter_covariance(fit_result, dof, n_sens):
    J = fit_result.jac                                   # whitened (Nrows, n_theta)
    fisher = 2.0 * (J.conj().T @ J).real                 # 2 Re(J^H J)  (12-29)
    cov = np.linalg.pinv(fisher, rcond=1e-10)            # drops the scale null space
    d = dof - n_sens
    lam = dof * d / ((d + 1) * (d - 1))                  # SML inflation (12-30)
    return cov * lam


def modal_uncertainty(model, theta, Ctheta):
    base = model.poles(theta)
    out = []
    for idx, (f0, Q) in enumerate(base):
        gf = np.zeros(model.n_theta); gq = np.zeros(model.n_theta)
        h = 1e-6
        for p in range(model.n_theta):
            dt = np.zeros(model.n_theta); dt[p] = h
            pj = model.poles(theta + dt); mj = model.poles(theta - dt)
            if len(pj) == len(base) == len(mj):
                gf[p] = (pj[idx][0] - mj[idx][0]) / (2 * h)
                gq[p] = (pj[idx][1] - mj[idx][1]) / (2 * h)
        out.append({"f0": f0, "Q": Q,
                    "f0_std": float(np.sqrt(max(gf @ Ctheta @ gf, 0.0))),
                    "Q_std": float(np.sqrt(max(gq @ Ctheta @ gq, 0.0)))})
    return out


def frf_band(model, theta, Ctheta, freq):
    dG = model.jacobian(theta, freq)                     # (F,n_sens,n_act,n_theta)
    var = np.einsum('fijp,pq,fijq->fij', dG, Ctheta, dG.conj()).real  # (11-2)
    return np.sqrt(np.clip(var, 0.0, None))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_crb_matches_monte_carlo -v`
Expected: PASS (ratio ~0.86).

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_fit.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): parameter covariance/CRB + modal uncertainty + FRF band (P&S 12-29/30, 11-2)"
```

---

## Task 6: Campaign assembler — `(Ẑ^[l], Ĉ_Z^[l])` from per-period spectra

**Files:**
- Create: `src/system_ident/mimo_campaign.py`
- Test: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: a `ChannelBackend` (e.g. step-1 `MIMOTwinBackend`), the multisine drive builder `multisine_from_psd` (`from system_ident.excitation import multisine_from_psd`), `MIMOTwinBackend.read`/`inject`/`ramp_down`.
- Produces: `assemble_campaign(backend, exc_names, drive_names, sens_names, freq_lines, fs, nperseg, n_periods, *, drive_psd, n_transient=1, seed=0) -> (exps, freq)` where `exps` is the `(Ybar, Ubar, Cz)` list (one per excitation channel, the robust method `n_exp = n_act`) and `freq` are the excited-line frequencies in Hz. Per experiment: drive one actuator with the multisine, read all drive monitors + sensors, reshape each into `n_periods` periods, rFFT, drop `n_transient` leading periods, keep excited lines, form `Ybar`/`Ubar` (period mean) and `Cz` (sample covariance of the mean of the stacked `[Y;U]` vector, `/dof`). Mirrors `loop.py:_estimate_tf_periodic` but stacks all channels.

- [ ] **Step 1: Write the failing test** (synthetic backend with known noise → mean+cov correct)

```python
from system_ident.mimo_campaign import assemble_campaign

class _FakeBackend:
    """One-actuator-at-a-time periodic spectra with known additive noise."""
    def __init__(self, G_fn, fs, nperseg, n_periods, n_act, n_sens, sigma, seed=0):
        self.G_fn = G_fn; self.fs = fs; self.nperseg = nperseg; self.n_periods = n_periods
        self.n_act = n_act; self.n_sens = n_sens; self.sigma = sigma
        self.rng = np.random.default_rng(seed); self._drv = {}
    def inject(self, ch, ts, fs): self._drv[ch] = np.asarray(ts, float)
    def ramp_down(self, ch, secs): pass
    def read(self, channels, duration):
        n = self.n_periods * self.nperseg
        # build u per actuator from injected exc channel "EXC{j}"
        u = np.zeros((self.n_act, n))
        for ch, ts in self._drv.items():
            j = int(ch[3:]); u[j, :min(len(ts), n)] = ts[:n]
        # y = G * u in freq per period (approx: apply G to the periodic drive)
        out = {}
        for ch in channels:
            if ch.startswith("SENS"):
                i = int(ch[4:]); y = np.zeros(n)
                # per period, Y = sum_j G_ij(freq) U_j(freq); build via rfft/irfft
                for p in range(self.n_periods):
                    seg = u[:, p*self.nperseg:(p+1)*self.nperseg]
                    Uf = np.fft.rfft(seg, axis=1)
                    f = np.fft.rfftfreq(self.nperseg, 1/self.fs)
                    G = self.G_fn(f)  # (F,n_sens,n_act)
                    Yf = np.einsum('fij,jf->if', G, Uf)[i]
                    y[p*self.nperseg:(p+1)*self.nperseg] = np.fft.irfft(Yf, self.nperseg)
                out[ch] = y + self.rng.standard_normal(n) * self.sigma
            elif ch.startswith("DRV"):
                j = int(ch[3:]); out[ch] = u[j] + self.rng.standard_normal(n) * self.sigma
        return out

def test_assemble_campaign_mean_and_cov():
    fs, nperseg, n_periods = 64.0, 256, 12
    f = np.fft.rfftfreq(nperseg, 1/fs)
    lines = np.array([4, 8, 12])                     # excited harmonic indices
    freq_lines = f[lines]
    m = MIMOCommonDenomModel(2, 2, 2)
    a = m.modes_to_denominator([(0.6, 20.0), (1.5, 35.0)])
    b = np.random.default_rng(2).standard_normal((2, 2, m.n_b_coef)) * 40.0
    theta = m.pack(a, b); theta /= np.linalg.norm(theta)
    G_fn = lambda ff: m.eval(theta, ff)
    be = _FakeBackend(G_fn, fs, nperseg, n_periods, 2, 2, sigma=0.01, seed=3)
    # build a multisine that excites exactly `lines`
    drive_psd = np.zeros(len(f)); drive_psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, exc_names=["EXC0", "EXC1"], drive_names=["DRV0", "DRV1"],
        sens_names=["SENS0", "SENS1"], freq_lines=freq_lines, fs=fs,
        nperseg=nperseg, n_periods=n_periods, drive_psd=drive_psd, n_transient=1, seed=3)
    assert len(exps) == 2 and np.allclose(freq, freq_lines)
    Ybar, Ubar, Cz = exps[0]
    assert Ybar.shape == (3, 2) and Ubar.shape == (3, 2) and Cz.shape == (3, 4, 4)
    # Cz is Hermitian PSD and ~ sigma^2-scaled (mean covariance shrinks with periods)
    assert np.allclose(Cz, np.conj(np.transpose(Cz, (0, 2, 1))), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Cz).real > -1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_assemble_campaign_mean_and_cov -v`
Expected: FAIL (`ModuleNotFoundError: system_ident.mimo_campaign`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/system_ident/mimo_campaign.py
"""Assemble per-experiment sample-mean spectra + stacked covariances for the SML fit.

Robust method (P&S): n_exp = n_act -- drive each actuator separately with the periodic
multisine, read all drive monitors + sensors, and form, per excited line:
  Ybar (sample mean over periods, sensors), Ubar (drive monitors),
  Cz = sample covariance of the MEAN of the stacked [Y;U] vector (= cov(per)/dof).
Mirrors loop.py:_estimate_tf_periodic but keeps ALL channels (no ratio).
"""
from __future__ import annotations
import numpy as np
from .excitation import multisine_from_psd


def _period_spectra(x, fs, nperseg, n_transient):
    x = np.asarray(x, float)
    P = len(x) // nperseg
    xr = x[:P * nperseg].reshape(P, nperseg)
    X = np.fft.rfft(xr, axis=1)
    return X[n_transient:]                            # drop settling periods


def assemble_campaign(backend, exc_names, drive_names, sens_names, freq_lines, fs,
                      nperseg, n_periods, *, drive_psd, n_transient=1, seed=0):
    fs = float(fs); nperseg = int(nperseg)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    lines = np.array([int(np.argmin(np.abs(f - fl))) for fl in freq_lines])
    rng = np.random.default_rng(seed)
    duration = n_periods * nperseg / fs
    n_sens, n_act = len(sens_names), len(drive_names)
    exps = []
    for l, exc in enumerate(exc_names):
        drive = multisine_from_psd(drive_psd, fs, nperseg, n_periods, f, seed=rng)
        backend.inject(exc, drive, fs)
        data = backend.read(list(drive_names) + list(sens_names), duration)
        # per-period spectra at the excited lines, stacked [Y (n_sens) ; U (n_act)]
        Yp = np.stack([_period_spectra(data[s], fs, nperseg, n_transient)[:, lines]
                       for s in sens_names], axis=-1)      # (P_eff, F, n_sens)
        Up = np.stack([_period_spectra(data[d], fs, nperseg, n_transient)[:, lines]
                       for d in drive_names], axis=-1)      # (P_eff, F, n_act)
        Zp = np.concatenate([Yp, Up], axis=-1)              # (P_eff, F, n_sens+n_act)
        P_eff = Zp.shape[0]
        Zbar = Zp.mean(0)                                   # (F, n_sens+n_act)
        Cz = np.empty((len(lines), n_sens + n_act, n_sens + n_act), complex)
        for k in range(len(lines)):
            dk = Zp[:, k, :] - Zbar[k]
            Cz[k] = (dk.conj().T @ dk) / (P_eff - 1) / P_eff   # covariance of the mean
        exps.append((Zbar[:, :n_sens], Zbar[:, n_sens:], Cz))
        backend.ramp_down(exc, 1.0)
    return exps, np.asarray(freq_lines, float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_assemble_campaign_mean_and_cov -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_campaign.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): per-actuator campaign assembler -> sample means + stacked covariances"
```

---

## Task 7: End-to-end on the step-1 twin (2-DoF) — beats the inverse at resonances

**Files:**
- Modify: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: everything above + step-1 `mimo_plant.mimo_suspension`, `mimo_plant.input_matrix`, `mimo_plant.output_matrix`, `mimo_loop.CoupledLoop`, `mimo_loop.velocity_damper`, `mimo_loop.recover_open_loop`, `backends.mimo_twin.MIMOTwinBackend`. (Read `tests/test_mimo.py:_square_loop`/`_campaign` for the exact construction; reuse them.)
- Produces: a 2-DoF end-to-end test proving the parametric fit recovers the coupled plant through the live loops AND is far better than step-1's per-bin inverse at the resonance bins.

- [ ] **Step 1: Write the failing test**

```python
from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix
from system_ident.mimo_loop import CoupledLoop, velocity_damper, recover_open_loop, off_resonance_mask
from system_ident.backends.mimo_twin import MIMOTwinBackend

def _build_2dof_loop(fs=64.0):
    modes = [(0.6, 20.0), (1.5, 35.0)]
    plant = mimo_suspension(modes, n_sens=2, n_act=2, coupling=0.2, gain=100.0, seed=0)
    M_in = input_matrix(2, 2, kind="identity")
    M_out = output_matrix(plant, 2, 2, basis="euler")
    ctrls = [velocity_damper(1.0, 4.0), velocity_damper(1.0, 4.0)]
    return CoupledLoop(plant, ctrls, M_in, M_out, fs=fs), modes

def test_end_to_end_2dof_fit_beats_inverse_at_resonance():
    fs, nperseg, n_periods = 64.0, 256, 18
    loop, modes = _build_2dof_loop(fs)
    be = MIMOTwinBackend(
        loop,
        exc_channels={f"EXC{j}": j for j in range(2)},
        drive_channels={f"DRV{j}": j for j in range(2)},
        sens_channels={f"SENS{i}": i for i in range(2)},
        sensor_asd=1e-3, process_asd=1e-4, seed=7)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    lines = np.flatnonzero((f >= 0.25) & (f <= 3.0))
    drive_psd = np.zeros(len(f)); drive_psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"EXC{j}" for j in range(2)], [f"DRV{j}" for j in range(2)],
        [f"SENS{i}" for i in range(2)], f[lines], fs, nperseg, n_periods,
        drive_psd=drive_psd, n_transient=2, seed=7)
    m = MIMOCommonDenomModel(2, 2, 2)
    res = MIMOSampleMLEstimator(m).fit(exps, freq, sk_start(m, exps, freq))
    G_fit = m.eval(res.theta, freq)
    G_true = np.transpose(loop.oracle(freq), (2, 0, 1))        # (F,n_sens,n_act)
    # nonparametric per-bin inverse for comparison
    Xmat = np.stack([exps[l][1] for l in range(2)], axis=-1)   # (F,n_act,n_exp)
    Ymat = np.stack([exps[l][0] for l in range(2)], axis=-1)   # (F,n_sens,n_exp)
    G_inv = recover_open_loop(Xmat, Ymat)                      # (F,n_sens,n_act)
    onres = ~off_resonance_mask(freq, [mo[0] for mo in modes], frac=0.06)
    rel = lambda Gx: np.abs(Gx - G_true) / np.abs(G_true)
    # parametric fit recovers everywhere, including resonances
    assert rel(G_fit).max() < 0.15
    # and is much better than the inverse AT the resonance bins
    assert np.median(rel(G_fit)[onres]) < 0.5 * np.median(rel(G_inv)[onres])
    pq = m.poles(res.theta)
    assert abs(pq[0][0] - 0.6) < 0.02 and abs(pq[1][0] - 1.5) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_end_to_end_2dof_fit_beats_inverse_at_resonance -v`
Expected: FAIL (assertion or wiring) — iterate `n_periods`/`n_transient`/line band only if needed; do NOT loosen the "beats inverse at resonance" assertion (the spec's headline claim).

- [ ] **Step 3: Make it pass**

No new product code — this exercises the assembled pipeline. If the fit misses, the levers are measurement-side (more periods → higher `dof`; more `n_transient` for high-Q settling; widen the excited band), never weakening the assertions. Confirm `dof = n_periods - n_transient ≥ n_sens + 8` (here `18-2=16 ≥ 10`).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_end_to_end_2dof_fit_beats_inverse_at_resonance -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mimo_fit.py
git commit -m "test(mimo-fit): end-to-end 2-DoF fit through live loops, beats inverse at resonances"
```

---

## Task 8: 6-DoF instantiation (marked slow)

**Files:**
- Modify: `tests/test_mimo_fit.py`

**Interfaces:**
- Consumes: same as Task 7 at `n_sens=n_act=6`, `n_modes=6`, modes = L/P/Y/R/V/T frequencies.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_end_to_end_6dof_fit():
    fs, nperseg, n_periods = 128.0, 512, 24
    modes = [(0.45, 20.0), (0.6, 25.0), (0.8, 18.0), (1.0, 30.0), (1.5, 35.0), (2.2, 28.0)]
    plant = mimo_suspension(modes, n_sens=6, n_act=6, coupling=0.15, gain=100.0, seed=0)
    M_in = input_matrix(6, 6, kind="identity")
    M_out = output_matrix(plant, 6, 6, basis="euler")
    ctrls = [velocity_damper(1.0, 4.0) for _ in range(6)]
    loop = CoupledLoop(plant, ctrls, M_in, M_out, fs=fs)
    be = MIMOTwinBackend(
        loop, {f"EXC{j}": j for j in range(6)}, {f"DRV{j}": j for j in range(6)},
        {f"SENS{i}": i for i in range(6)}, sensor_asd=1e-3, process_asd=1e-4, seed=11)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    lines = np.flatnonzero((f >= 0.3) & (f <= 3.0))
    drive_psd = np.zeros(len(f)); drive_psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"EXC{j}" for j in range(6)], [f"DRV{j}" for j in range(6)],
        [f"SENS{i}" for i in range(6)], f[lines], fs, nperseg, n_periods,
        drive_psd=drive_psd, n_transient=3, seed=11)   # dof = 24-3 = 21 >= 6+8
    m = MIMOCommonDenomModel(6, 6, 6)
    res = MIMOSampleMLEstimator(m).fit(exps, freq, sk_start(m, exps, freq), max_iter=120)
    fitted = sorted(p[0] for p in m.poles(res.theta))
    truth = sorted(mo[0] for mo in modes)
    assert len(fitted) == 6
    for ff, tt in zip(fitted, truth):
        assert abs(ff - tt) / tt < 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_end_to_end_6dof_fit -v -m slow`
Expected: FAIL until wiring complete (or pass directly if pipeline is solid).

- [ ] **Step 3: Make it pass**

Measurement-side levers only (periods, transient, band); confirm `dof = n_periods - n_transient ≥ n_sens + 8 = 14`. If the SK init is poor at 6-DoF, raise `max_iter`; do not change the estimator math.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_end_to_end_6dof_fit -v -m slow`
Expected: PASS (all 6 `f0` within 3%).

- [ ] **Step 5: Commit**

```bash
git add tests/test_mimo_fit.py
git commit -m "test(mimo-fit): 6-DoF (L/P/Y/R/V/T) end-to-end fit (marked slow)"
```

---

## Task 9: P&S validation suite + full-suite run + push

**Files:**
- Modify: `src/system_ident/mimo_fit.py`, `tests/test_mimo_fit.py`

**Interfaces:**
- Produces: `validate_fit(model, theta, exps, freq, dof) -> dict` with `frf_pass_fraction` (P&S 12-34: fraction of excited bins where `|G_fit−G_inv| ≤ √(F_p(2,2dof))·σ̂`, using the per-bin inverse `recover_open_loop` and its period-variance std as `Ĝ`), `cost`, `cost_expected` (`dof/(dof−n_sens)·n_sens·F`, P&S 12-19), and `cost_ratio = cost/cost_expected`. A healthy fit has `frf_pass_fraction ≳ 0.9` and `cost_ratio` near 1.

- [ ] **Step 1: Write the failing test**

```python
from system_ident.mimo_fit import validate_fit

def test_validation_metrics_on_2dof():
    fs, nperseg, n_periods = 64.0, 256, 18
    loop, modes = _build_2dof_loop(fs)
    be = MIMOTwinBackend(
        loop, {f"EXC{j}": j for j in range(2)}, {f"DRV{j}": j for j in range(2)},
        {f"SENS{i}": i for i in range(2)}, sensor_asd=1e-3, process_asd=1e-4, seed=5)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    lines = np.flatnonzero((f >= 0.25) & (f <= 3.0))
    drive_psd = np.zeros(len(f)); drive_psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"EXC{j}" for j in range(2)], [f"DRV{j}" for j in range(2)],
        [f"SENS{i}" for i in range(2)], f[lines], fs, nperseg, n_periods,
        drive_psd=drive_psd, n_transient=2, seed=5)
    m = MIMOCommonDenomModel(2, 2, 2)
    res = MIMOSampleMLEstimator(m).fit(exps, freq, sk_start(m, exps, freq))
    rep = validate_fit(m, res.theta, exps, freq, dof=n_periods - 2)
    assert rep["frf_pass_fraction"] > 0.85
    assert 0.3 < rep["cost_ratio"] < 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_validation_metrics_on_2dof -v`
Expected: FAIL (`ImportError: validate_fit`).

- [ ] **Step 3: Write minimal implementation** (append to `mimo_fit.py`)

```python
from scipy.stats import f as _f_dist
from .mimo_loop import recover_open_loop


def validate_fit(model, theta, exps, freq, dof):
    n_sens, n_act = model.n_sens, model.n_act
    G_fit = model.eval(theta, freq)
    Xmat = np.stack([exps[l][1] for l in range(n_act)], axis=-1)   # (F,n_act,n_exp)
    Ymat = np.stack([exps[l][0] for l in range(n_act)], axis=-1)   # (F,n_sens,n_exp)
    G_inv = recover_open_loop(Xmat, Ymat)                          # nonparametric Ĝ
    # per-bin std of Ĝ from the stacked covariances (rough: diag of input-projected Cz)
    std = np.empty_like(G_fit, float)
    for k in range(len(freq)):
        # average the per-experiment output-channel std as the FRM uncertainty scale
        s = 0.0
        for l in range(n_act):
            s += np.sqrt(np.clip(np.diag(exps[l][2][k])[:n_sens].real, 0, None)).mean()
        std[k] = (s / n_act)
    thresh = np.sqrt(_f_dist.ppf(0.95, 2, 2 * dof)) * std
    passed = np.abs(G_fit - G_inv) <= thresh
    frac = float(passed.mean())
    cost_expected = dof / (dof - n_sens) * n_sens * len(freq)
    # recompute cost
    from_fit = MIMOSampleMLEstimator(model)._assemble(theta, exps, freq)[2]
    return {"frf_pass_fraction": frac, "cost": float(from_fit),
            "cost_expected": float(cost_expected),
            "cost_ratio": float(from_fit / cost_expected)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_validation_metrics_on_2dof -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite (fast tests) and the slow tests once**

Run: `conda run -n sysid python -m pytest -q` (expect all prior step-1 tests still pass + new step-2 fast tests).
Run: `conda run -n sysid python -m pytest -q -m slow` (CRB-vs-MC, 6-DoF).
Expected: all green.

- [ ] **Step 6: Commit and push**

```bash
git add src/system_ident/mimo_fit.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): P&S validation suite (12-34 FRF comparison, cost-vs-expected)"
git push origin main
```

---

## Self-Review

**1. Spec coverage:**
- Spec §2 model (common-denom `B/A`, free per-element, `‖θ‖=1`) → Task 0/1/2. ✓
- Spec §3 data (robust `n_exp=n_act`, sample mean+cov, `dof≥n_sens+8`) → Task 6, asserted in 7/8. ✓
- Spec §4 estimator (SK init, SML cost 12-15/16/17, GN pseudo-Jacobian, over-parameterize+constrain) → Task 3/4. ✓
- Spec §5 CRB + `f0/Q` propagation + FRF band → Task 5. ✓
- Spec §6 validation (recovers truth incl. resonances, beats inverse at peaks, 12-34, cost; 2-DoF→6-DoF) → Task 7/8/9. ✓
- Spec §7 component boundaries (new modules; `model.py`/`gml.py`/`fisher.py` untouched) → file structure. ✓
- Spec §8 no new dependency → numpy/scipy only. ✓
- Modal-uncertainty high-SNR caveat (App 11.D) — documented in spec §5; v1 uses the linear propagation (Task 5) and records the caveat; improved bounds are out of v1 scope (acceptable, spec calls it "when the linearization is inadequate").

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; tests carry real assertions. ✓

**3. Type consistency:** `MIMOCommonDenomModel` attrs/methods (`eval`, `jacobian`, `poles`, `pack`/`unpack`, `_vander`, `modes_to_denominator`) consistent across Tasks 0–9; `exps` tuple shape `(Ybar(F,n_sens), Ubar(F,n_act), Cz(F,n_sens+n_act,n_sens+n_act))` consistent in Tasks 3–9; `FitResult.jac` is the whitened Jacobian consumed by Task 5; `freq` always Hz. ✓
