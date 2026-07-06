"""Reference implementation of the landing-page excitation-arcade engine.

Presentation-only (not package API). This is the **Python twin** of
``docs/assets/excitation-arcade.js``: the two share one set of constants and one set of
formulas, so the browser game computes the *same* Fisher / Cramér–Rao numbers the
``system_ident`` pipeline does. ``tests/test_arcade.py`` asserts (a) this reference matches
committed golden numbers the JS is calibrated to, and (b) the resonance kernel matches the
package's ``TFModel`` pole convention. Keep this file and the JS numerically in lock-step:
edit both, or neither.

The plant is two competing resonances; the game pins ``theta = (f0_1, Q1, f0_2, Q2)``. The
per-mode amplitudes are chosen so the two peaks have equal height, so *both* modal Q's are
comparably hard — starving either mode punishes its Q. The measurement "time to identify to
5%" for a drive ``Pxx`` is ``ETA = C * worst_frac(Pxx)**2`` where ``worst_frac`` is the worst
parameter's CRB fractional uncertainty at the reference budget and ``C`` calibrates the
optimal drive to a pleasant ~45 s (Fisher is linear in time, so ``frac ~ 1/sqrt(T)``).
"""
from __future__ import annotations

import numpy as np

# ── locked constants (MUST equal the JS ENGINE constants) ───────────────────────────
N_BINS = 120
F_LO, F_HI = 0.3, 6.0
MODES = ((1.0, 20.0), (2.5, 20.0))                 # (f0, Q) per mode
T_REF = 100.0                                       # reference record length [s]
PX_TOT = 1.0                                        # fixed drive-power budget
TARGET = 0.05                                       # 5% fractional-uncertainty goal
C_TIME = 1835.19108                                 # ETA calibration (optimal -> 45 s)

FREQ = np.linspace(F_LO, F_HI, N_BINS)
DF = FREQ[1] - FREQ[0]
# equal-peak-height amplitudes: |G_k(f0)| = A_k*Q_k/w0_k^2 ~ 1  ->  A_k = w0_k^2 / Q_k
AMPS = tuple((2 * np.pi * f0) ** 2 / Q for f0, Q in MODES)
THETA = np.array([MODES[0][0], MODES[0][1], MODES[1][0], MODES[1][1]], float)


def plant_frf(theta=THETA, freq=FREQ):
    """Two-resonance FRF, same s-domain pole form as ``TFModel.from_resonances``.

    ``G(f) = sum_k A_k / ((w0_k^2 - w^2) + i w w0_k / Q_k)``,  ``w = 2*pi*f``.
    """
    w = 2 * np.pi * np.asarray(freq, float)
    G = np.zeros(len(w), complex)
    for k, (f0, Q) in enumerate(((theta[0], theta[1]), (theta[2], theta[3]))):
        w0 = 2 * np.pi * f0
        G += AMPS[k] / ((w0 ** 2 - w ** 2) + 1j * w * w0 / Q)
    return G


def _jacobian():
    """dG/dtheta by central finite difference — independent of the drive, so precomputed."""
    d = []
    for i in range(4):
        h = 1e-6 * max(abs(THETA[i]), 1e-3)
        hi, lo = THETA.copy(), THETA.copy()
        hi[i] += h; lo[i] -= h
        d.append((plant_frf(hi) - plant_frf(lo)) / (2 * h))
    return np.array(d)                              # (4, N_BINS)


_D = _jacobian()
# per-bin real information kernels R[b] = Re[dG_i* dG_j], so Fisher = 2*T*sum_b Pxx[b] R[b] df
_R = np.einsum("ib,jb->bij", _D.conj(), _D).real    # (N_BINS, 4, 4)


def fisher(Pxx):
    """Fisher matrix in (f0_1,Q1,f0_2,Q2) at the reference budget for drive ``Pxx``."""
    return 2 * T_REF * np.einsum("b,bij->ij", np.asarray(Pxx, float), _R) * DF


def frac_uncertainty(Pxx):
    """Per-parameter CRB fractional uncertainty sqrt(diag(I^-1))/|theta| at T_REF."""
    C = np.linalg.inv(fisher(Pxx))
    return np.sqrt(np.clip(np.diag(C), 0, None)) / np.abs(THETA)


def eta_seconds(Pxx):
    """(overall ETA to 5%, per-parameter ETA) for drive ``Pxx``."""
    fr = frac_uncertainty(Pxx)
    per = C_TIME * fr ** 2
    return float(np.max(per)), per


def flat_drive():
    return np.full(N_BINS, PX_TOT / (F_HI - F_LO))


def optimal_drive(n_iter=16):
    """Dispersion fixed point in (f0,Q) coords — the 'par' drive (computed once)."""
    P = flat_drive()
    for _ in range(n_iter):
        Iinv = np.linalg.inv(fisher(P))
        nu = PX_TOT * np.einsum("ij,bji->b", Iinv, 2 * T_REF * _R)
        P = P * np.clip(nu, 0, None)
        P = P / (np.trapezoid(P, FREQ)) * PX_TOT
    return P
