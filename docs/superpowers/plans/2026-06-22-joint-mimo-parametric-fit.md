# Joint MIMO Modal Identification — Completion & Test-Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and fully test the **rank-1 modal** step-2 fit — back the already-landed, verified modules with comprehensive TDD coverage, add the one missing component (`validate_fit`), and pass a whole-branch review.

**Architecture:** The core is **already on `main`** (commit `a7ec7cd`), verified end-to-end (all 6 modal frequencies of a coupled 6-DoF suspension recovered to 0.15% through the closed loop, CRB-consistent): `mimo_modal.py` (`Rank1ModalModel`), `mimo_fit.py` (peak-pick + IQML-LM SML + CRB), `mimo_campaign.py` (campaign assembler). This plan **does not re-implement** that code — each task writes the comprehensive test coverage for a component (fixing any gaps the tests reveal) and adds `validate_fit`. Because the code exists, the TDD loop is "write the test → run it (it should pass against the landed code, or RED if it exposes a real gap) → fix the gap if any → commit."

**Tech Stack:** Python, numpy, scipy. python-control only via the step-1 twin (data + oracle). No new third-party dependency.

**Spec:** `docs/superpowers/specs/2026-06-22-joint-mimo-parametric-fit-design.md` (v2, rank-1 modal).
**P&S derivation:** `.llm/pintelon-schoukens-mimo-fit.md` + `.llm/ps-book/full-text-pymupdf4llm.md`.

## Global Constraints

- **Rank-1 modal model (verified):** `G_ij(s) = Σ_k φ_k,i ψ_k,j / (sₙ² + b_k sₙ + a_k)`, `sₙ = s/s_ref`. Shared poles `(a_k,b_k)`, per-mode rank-1 residues `R_k = φ_k ψ_kᵀ`. **Do not** revert to the common-denominator `B/A` form — it is unidentifiable at MIMO scale (spec §0). [[rank1-modal-mimo-fit]], [[stay-on-pintelon-schoukens]].
- **Data-driven peak-pick init** on the step-1 **recovered open-loop `G=Y·X⁻¹`** (resonances are full peaks there; the closed-loop sensor spectrum has them suppressed). Prior is a fallback only.
- **IQML / iteratively-reweighted Levenberg–Marquardt** SML (frozen weighting per iteration, P&S §9.12.2); cost-based step acceptance.
- **Frequency normalization required:** `set_reference(freq)` → `s_ref = 2π·median(freq)`.
- **`dof ≥ n_sens + 8`** periods for a trustworthy CRB; the SML inflation is `λ₂(dof)=dof²/((dof−n_sens+1)(dof−n_sens−1))` (P&S 12-30).
- **Dimension-generic:** `n_sens`, `n_act` independent — **square AND rectangular** must be tested.
- **Campaigns need adequate frequency resolution** (`df ≲ half the closest mode spacing`) — verified: coarse bins (0.25 Hz) miss modes 0.15 Hz apart; fine bins (0.03 Hz) recover exactly.
- **Do not modify** the landed `src/` modules' *interfaces*; fix internals only if a test exposes a real bug. Do not touch `model.py`/`gml.py`/`fisher.py`.
- Run via `conda run -n sysid python -m pytest …` ([[use-conda-run-sysid-env]]). The fit needs no torch; ignore `KMP_DUPLICATE_LIB_OK` for these tests. Any plot SVG + Git LFS ([[graphics-svg-lfs-only]]). Trunk-based, push to `main` ([[trunk-based-push-to-main]]). Phase 1 only ([[two-phase-cds-plan]]).

## File Structure

- **Extend `tests/test_mimo_modal.py`** — model coverage (Task 1).
- **Create `tests/test_mimo_fit.py`** — init, estimator, CRB, validation (Tasks 2–4).
- **Create `tests/test_mimo_campaign.py`** — campaign assembler (Task 5).
- **Modify `src/system_ident/mimo_fit.py`** — add `validate_fit` (Task 4).

The existing `tests/test_mimo_modal.py` already covers: Jacobian-vs-FD, poles roundtrip, peak-pick, and a marked-slow 6-DoF closed-loop recovery. Do **not** duplicate those; add the gaps named below.

---

## Task 1: `Rank1ModalModel` — complete coverage

**Files:**
- Modify: `tests/test_mimo_modal.py`

**Interfaces (already implemented in `src/system_ident/mimo_modal.py`):**
- `Rank1ModalModel(n_sens, n_act, n_modes)` → attrs `per` (`=2+n_sens+n_act`), `n_theta` (`=n_modes*per`), `s_ref`; methods `set_reference(freq)→self`, `unpack(theta)→[(a,b,phi,psi),...]`, `pack(ab,phi,psi)→theta`, `eval(theta,freq)→(F,n_sens,n_act)`, `jacobian(theta,freq)→(F,n_sens,n_act,n_theta)`, `poles(theta)→[(f0,Q),...]`, `ab_from_modes(modes)→[(a,b),...]`.

- [ ] **Step 1: Write the tests** (eval value, pack/unpack roundtrip, set_reference default no-op, rectangular shapes)

```python
# append to tests/test_mimo_modal.py
def test_eval_value_single_mode():
    m = Rank1ModalModel(2, 2, 1).set_reference(np.linspace(0.5, 2.0, 10))
    ab = m.ab_from_modes([(1.0, 10.0)])
    phi = np.array([[2.0, 0.5]]); psi = np.array([[1.0, 0.3]])
    theta = m.pack(ab, phi, psi)
    G = m.eval(theta, np.array([0.8]))
    sn = 2j * np.pi * 0.8 / m.s_ref
    D = sn * sn + ab[0][1] * sn + ab[0][0]
    assert np.isclose(G[0, 0, 0], 2.0 * 1.0 / D)
    assert np.isclose(G[0, 1, 1], 0.5 * 0.3 / D)

def test_pack_unpack_roundtrip():
    m = Rank1ModalModel(3, 2, 2)
    rng = np.random.default_rng(0)
    ab = [(rng.random()+0.5, rng.random()+0.1) for _ in range(2)]
    phi = rng.standard_normal((2, 3)); psi = rng.standard_normal((2, 2))
    theta = m.pack(ab, phi, psi)
    assert theta.shape == (m.n_theta,)
    got = m.unpack(theta)
    for k in range(2):
        assert np.isclose(got[k][0], ab[k][0]) and np.isclose(got[k][1], ab[k][1])
        assert np.allclose(got[k][2], phi[k]) and np.allclose(got[k][3], psi[k])

def test_set_reference_default_is_noop():
    m = Rank1ModalModel(2, 2, 1)
    assert m.s_ref == 1.0                      # default before set_reference

def test_rectangular_shapes():
    m = Rank1ModalModel(n_sens=3, n_act=2, n_modes=2).set_reference(np.linspace(0.3, 3.0, 20))
    assert m.n_theta == 2 * (2 + 3 + 2)
    theta = np.ones(m.n_theta)
    G = m.eval(theta, np.linspace(0.3, 3.0, 20))
    assert G.shape == (20, 3, 2)
    J = m.jacobian(theta, np.linspace(0.3, 3.0, 20))
    assert J.shape == (20, 3, 2, m.n_theta)
```

- [ ] **Step 2: Run** `conda run -n sysid python -m pytest tests/test_mimo_modal.py -v -m "not slow"` — expect all PASS (code exists). If any FAIL, it has exposed a real gap → fix `mimo_modal.py` internals minimally, re-run.
- [ ] **Step 3: Commit**

```bash
git add tests/test_mimo_modal.py
git commit -m "test(mimo-modal): eval value, pack/unpack, rectangular shapes"
```

---

## Task 2: `mimo_fit` initialization — peak-pick + residue init

**Files:**
- Create: `tests/test_mimo_fit.py`

**Interfaces (implemented in `src/system_ident/mimo_fit.py`):**
- `peak_pick_modes(G, freq, n_modes, *, default_Q=20.0) -> [(f0,Q),...]` — from `|G|²` peaks.
- `init_residues(model, ab, exps, freq) -> (phi(M,n_sens), psi(M,n_act))` — linear residue LS + rank-1 SVD. `exps` = list of `(Ybar(F,n_sens), Ubar(F,n_act), Cz(F,n_s+n_a,n_s+n_a))`.
- `initial_theta(model, exps, freq, G_nonparametric, *, prior_modes=None) -> theta`.

- [ ] **Step 1: Write the tests** — a small synthetic open-loop helper, then peak-pick + residue recovery.

```python
# tests/test_mimo_fit.py
import numpy as np
import pytest
from system_ident.mimo_modal import Rank1ModalModel
from system_ident.mimo_fit import (peak_pick_modes, init_residues, initial_theta,
                                    MIMOModalEstimator, parameter_covariance,
                                    modal_uncertainty, frf_band)

def synth_openloop(model, modes, phi, psi, *, sigZ=0.0, M=20, seed=0, freq=None):
    """Robust-method campaign on G = model truth (open loop). Returns (exps, freq, theta_true, G)."""
    if freq is None:
        freq = np.linspace(0.3, 3.0, 120)
    model.set_reference(freq)
    ab = model.ab_from_modes(modes)
    theta = model.pack(ab, phi, psi)
    G = model.eval(theta, freq); s = 2j * np.pi * freq
    rng = np.random.default_rng(seed); nsa = model.n_sens + model.n_act; exps = []
    for l in range(model.n_act):
        U = np.zeros((len(freq), model.n_act), complex); U[:, l] = 1.0 + 0.3 * np.cos(s.imag)
        Y = np.einsum('fij,fj->fi', G, U); per = []
        for _ in range(max(M, 2)):
            nY = (rng.standard_normal((len(freq), model.n_sens)) + 1j*rng.standard_normal((len(freq), model.n_sens))) * sigZ/np.sqrt(2)
            nU = (rng.standard_normal((len(freq), model.n_act)) + 1j*rng.standard_normal((len(freq), model.n_act))) * sigZ/np.sqrt(2)
            per.append(np.concatenate([Y + nY, U + nU], axis=1))
        per = np.array(per); Zb = per.mean(0)
        Cz = np.empty((len(freq), nsa, nsa), complex)
        for k in range(len(freq)):
            d = per[:, k, :] - Zb[k]; Cz[k] = (d.conj().T @ d) / (len(per) - 1) / len(per)
        exps.append((Zb[:, :model.n_sens], Zb[:, model.n_sens:], Cz))
    return exps, freq, theta, G

def test_peak_pick_finds_two_modes():
    m = Rank1ModalModel(2, 2, 2)
    freq = np.linspace(0.3, 3.0, 200); m.set_reference(freq)
    phi = np.array([[1.0, 0.2], [0.3, 1.0]]); psi = np.array([[1.0, 0.1], [0.2, 1.0]])
    _, _, theta, G = synth_openloop(m, [(0.7, 40), (1.8, 50)], phi, psi, freq=freq)
    found = sorted(f for f, _ in peak_pick_modes(G, freq, 2))
    assert abs(found[0] - 0.7) < 0.05 and abs(found[1] - 1.8) < 0.05

def test_init_residues_recovers_rank1_shapes():
    m = Rank1ModalModel(3, 2, 2)
    phi = np.array([[1.0, 0.4, 0.2], [0.1, 1.0, 0.5]]); psi = np.array([[1.0, 0.3], [0.2, 1.0]])
    exps, freq, theta, G = synth_openloop(m, [(0.6, 30), (1.6, 40)], phi, psi, sigZ=1e-4, M=20, seed=1)
    ab = m.ab_from_modes([(0.6, 30), (1.6, 40)])
    phi_hat, psi_hat = init_residues(m, ab, exps, freq)
    # residue matrices match up to per-mode sign/scale gauge -> compare R_k = phi psi^T
    for k in range(2):
        R = np.outer(phi[k], psi[k]); Rh = np.outer(phi_hat[k], psi_hat[k])
        assert np.allclose(R, Rh, atol=1e-2)
```

- [ ] **Step 2: Run** `conda run -n sysid python -m pytest tests/test_mimo_fit.py -v` — expect PASS.
- [ ] **Step 3: Commit**

```bash
git add tests/test_mimo_fit.py
git commit -m "test(mimo-fit): peak-pick + rank-1 residue init"
```

---

## Task 3: Estimator + CRB

**Files:**
- Modify: `tests/test_mimo_fit.py`

**Interfaces (implemented):**
- `MIMOModalEstimator(model).fit(exps, freq, theta0, *, max_iter=200, tol=1e-9) -> FitResult(theta, jac, cost, n_iter)`.
- `parameter_covariance(fit_result, dof, n_sens) -> Ctheta`; `modal_uncertainty(model, theta, Ctheta) -> [{f0,Q,f0_std,Q_std},...]`; `frf_band(model, theta, Ctheta, freq) -> (F,n_sens,n_act)`.

- [ ] **Step 1: Write the tests** — convergence from a perturbed init, and CRB-vs-Monte-Carlo (marked slow).

```python
# append to tests/test_mimo_fit.py
def test_estimator_recovers_poles_from_perturbed_init():
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.0,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=2)
    th0 = initial_theta(m, exps, freq, G)              # data-driven (peak-pick)
    res = MIMOModalEstimator(m).fit(exps, freq, th0)
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 5e-3 and abs(fit[1]-1.6) < 5e-3

@pytest.mark.slow
def test_crb_matches_monte_carlo():
    m = Rank1ModalModel(2, 2, 2)
    phi = np.array([[1.,.3],[.2,1.]]); psi = np.array([[1.,.2],[.1,1.]])
    modes = [(0.6,30),(1.6,40)]; M = 24
    e0, freq, th_t, G = synth_openloop(m, modes, phi, psi, sigZ=0.01, M=M, seed=1)
    r0 = MIMOModalEstimator(m).fit(e0, freq, initial_theta(m, e0, freq, G))
    Ct = parameter_covariance(r0, dof=M, n_sens=2)
    pred = modal_uncertainty(m, r0.theta, Ct)[0]["f0_std"]
    f0s = []
    for sd in range(40):
        e2, _, _, G2 = synth_openloop(m, modes, phi, psi, sigZ=0.01, M=M, seed=100+sd)
        r2 = MIMOModalEstimator(m).fit(e2, freq, initial_theta(m, e2, freq, G2))
        pq = m.poles(r2.theta)
        if len(pq) == 2 and abs(pq[0][0]-0.6) < 0.1: f0s.append(pq[0][0])
    mc = np.std(f0s)
    assert 0.4 < pred/mc < 2.5                         # CRB brackets the Monte-Carlo spread
    band = frf_band(m, r0.theta, Ct, freq)
    assert band.shape == (len(freq), 2, 2) and np.all(np.isfinite(band)) and np.all(band >= 0)
```

- [ ] **Step 2: Run** `conda run -n sysid python -m pytest tests/test_mimo_fit.py -v` (and `-m slow` for the CRB test). Expect PASS.
- [ ] **Step 3: Commit**

```bash
git add tests/test_mimo_fit.py
git commit -m "test(mimo-fit): estimator convergence + CRB vs Monte-Carlo"
```

---

## Task 4: Implement + test `validate_fit`

**Files:**
- Modify: `src/system_ident/mimo_fit.py`
- Modify: `tests/test_mimo_fit.py`

**Interfaces (NEW — to implement):**
- `validate_fit(model, theta, exps, freq, dof, modes_hz=None) -> dict` with `frf_rel_median_offres` (median `|G_fit−G_inv|/|G_inv|` at off-resonance bins, `G_inv = recover_open_loop`), `cost`, `cost_expected` (`dof/(dof−n_sens)·n_sens·len(freq)`, P&S 12-19), `cost_ratio`. No fabricated `σ̂_Ĝ`; the rigorous per-bin 12-34 F-test is a documented follow-on.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mimo_fit.py
from system_ident.mimo_fit import validate_fit
def test_validate_fit_metrics():
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    modes = [(0.6,30),(1.6,40)]
    exps, freq, theta_true, G = synth_openloop(m, modes, phi, psi, sigZ=5e-3, M=20, seed=3)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    rep = validate_fit(m, res.theta, exps, freq, dof=20, modes_hz=[mm[0] for mm in modes])
    assert rep["frf_rel_median_offres"] < 0.1
    assert 0.3 < rep["cost_ratio"] < 3.0
```

- [ ] **Step 2: Run** `conda run -n sysid python -m pytest tests/test_mimo_fit.py::test_validate_fit_metrics -v` — expect FAIL (`ImportError: validate_fit`).
- [ ] **Step 3: Implement** (append to `src/system_ident/mimo_fit.py`)

```python
from .mimo_loop import recover_open_loop, off_resonance_mask


def validate_fit(model, theta, exps, freq, dof, modes_hz=None):
    n_sens, n_act = model.n_sens, model.n_act
    G_fit = model.eval(theta, freq)
    Xmat = np.stack([exps[l][1] for l in range(n_act)], axis=-1)   # (F,n_act,n_exp)
    Ymat = np.stack([exps[l][0] for l in range(n_act)], axis=-1)   # (F,n_sens,n_exp)
    G_inv = recover_open_loop(Xmat, Ymat)                          # nonparametric overlay
    keep = (np.ones(len(freq), bool) if modes_hz is None
            else off_resonance_mask(freq, modes_hz, frac=0.08))
    rel = np.abs(G_fit - G_inv) / np.maximum(np.abs(G_inv), 1e-30)
    frf_rel_median_offres = float(np.median(rel[keep]))
    cost = MIMOModalEstimator(model)._assemble(theta, exps, freq)[2]
    cost_expected = dof / (dof - n_sens) * n_sens * len(freq)
    return {"frf_rel_median_offres": frf_rel_median_offres,
            "cost": float(cost), "cost_expected": float(cost_expected),
            "cost_ratio": float(cost / cost_expected)}
```

- [ ] **Step 4: Run** the test — expect PASS.
- [ ] **Step 5: Commit**

```bash
git add src/system_ident/mimo_fit.py tests/test_mimo_fit.py
git commit -m "feat(mimo-fit): validate_fit (off-res agreement + cost-vs-expected, P&S 12-19)"
```

---

## Task 5: Campaign assembler + end-to-end (square, non-square, prior-robustness) + full suite

**Files:**
- Create: `tests/test_mimo_campaign.py`
- Modify: `tests/test_mimo_fit.py`

**Interfaces (implemented):**
- `assemble_campaign(backend, exc_names, drive_names, sens_names, freq_lines, *, fs, nperseg, n_periods, drive_psd, n_transient=1, seed=0) -> (exps, freq)`.

- [ ] **Step 1: Campaign assembler test** (synthetic backend → sample mean/cov shape + Hermitian PSD)

```python
# tests/test_mimo_campaign.py
import numpy as np
from system_ident.mimo_campaign import assemble_campaign

class _FakeBackend:
    def __init__(self, fs, nperseg, n_periods, n_act, n_sens, sigma, seed=0):
        self.fs=fs; self.nperseg=nperseg; self.n_periods=n_periods
        self.n_act=n_act; self.n_sens=n_sens; self.sigma=sigma
        self.rng=np.random.default_rng(seed); self._d={}
    def inject(self, ch, ts, fs): self._d[ch]=np.asarray(ts,float)
    def ramp_down(self, ch, secs): pass
    def read(self, channels, duration):
        n=self.n_periods*self.nperseg; u=np.zeros((self.n_act,n))
        for ch,ts in self._d.items():
            j=int(ch[1:]); u[j,:min(len(ts),n)]=ts[:n]
        out={}
        for ch in channels:
            if ch.startswith("S"):
                i=int(ch[1:]); out[ch]=u[i % self.n_act]+self.rng.standard_normal(n)*self.sigma
            else:
                j=int(ch[1:]); out[ch]=u[j]+self.rng.standard_normal(n)*self.sigma
        return out

def test_assemble_campaign_shapes_and_psd():
    fs, nperseg, nper = 64.0, 256, 12
    f = np.fft.rfftfreq(nperseg, 1/fs); lines = np.array([4, 8, 12]); psd = np.zeros(len(f)); psd[lines]=1.0
    be = _FakeBackend(fs, nperseg, nper, 2, 2, 0.01, seed=3)
    exps, freq = assemble_campaign(be, ["E0","E1"], ["D0","D1"], ["S0","S1"], f[lines],
                                   fs=fs, nperseg=nperseg, n_periods=nper, drive_psd=psd, n_transient=1, seed=3)
    assert len(exps) == 2 and np.allclose(freq, f[lines])
    Yb, Ub, Cz = exps[0]
    assert Yb.shape == (3, 2) and Ub.shape == (3, 2) and Cz.shape == (3, 4, 4)
    assert np.allclose(Cz, np.conj(np.transpose(Cz, (0, 2, 1))), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Cz).real > -1e-9)
```

- [ ] **Step 2: Non-square + prior-robustness end-to-end** (open-loop, fast — the marked-slow closed-loop 6-DoF already exists in `test_mimo_modal.py`)

```python
# append to tests/test_mimo_fit.py
def test_nonsquare_recovery():
    m = Rank1ModalModel(n_sens=4, n_act=2, n_modes=2)   # rectangular
    phi = np.array([[1.,.3,.2,.1],[.2,1.,.4,.3]]); psi = np.array([[1.,.2],[.1,1.]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=4)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 1e-2 and abs(fit[1]-1.6) < 1e-2

def test_prior_independent_recovery():
    # peak-pick is data-driven: recovery does not depend on the prior at all
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=5)
    # wildly wrong prior (50% off) is ignored because peak-pick reads modes from G
    th0 = initial_theta(m, exps, freq, G, prior_modes=[(0.3, 20), (2.4, 20)])
    res = MIMOModalEstimator(m).fit(exps, freq, th0)
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 1e-2 and abs(fit[1]-1.6) < 1e-2
```

- [ ] **Step 3: Run the full suite**

Run: `conda run -n sysid python -m pytest tests/test_mimo_modal.py tests/test_mimo_fit.py tests/test_mimo_campaign.py -v`
Then the slow ones: `conda run -n sysid python -m pytest tests/test_mimo_modal.py tests/test_mimo_fit.py -m slow -v`
Expected: all PASS. Also run the whole repo suite to confirm no regression: `conda run -n sysid python -m pytest -q`.

- [ ] **Step 4: Commit and push**

```bash
git add tests/test_mimo_campaign.py tests/test_mimo_fit.py
git commit -m "test(mimo-fit): campaign assembler, non-square, prior-independent recovery"
git push origin main
```

---

## Self-Review

**Spec coverage:** model (§2) → Task 1; peak-pick init (§4) → Task 2; estimator (§5) → Task 3; CRB (§6) → Task 3; validation (§7) → Task 4 + Task 5 end-to-end; campaign (§3) → Task 5; rank-1/dimension-generic → Tasks 1+5 (rectangular). The marked-slow closed-loop 6-DoF recovery already lives in `test_mimo_modal.py` and is the headline end-to-end test. ✓

**Placeholder scan:** every step has concrete test code + exact commands. ✓

**Type consistency:** `Rank1ModalModel`/`MIMOModalEstimator`/`FitResult.jac`, the `exps` tuple `(Ybar(F,n_sens), Ubar(F,n_act), Cz(F,n_s+n_a,n_s+n_a))`, and `freq` in Hz are consistent across Tasks 1–5 and match the landed `src/` code. ✓
