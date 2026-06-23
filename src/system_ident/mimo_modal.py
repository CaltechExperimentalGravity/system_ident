"""Rank-1 modal MIMO transfer-function model (step-2 joint fit).

    G_ij(s) = sum_k  phi[k,i] * psi[k,j] / (sn^2 + b_k*sn + a_k),   sn = s / s_ref

Shared modal poles (a_k, b_k) with per-mode RANK-1 real residues R_k = phi_k psi_k^T
(sensor mode shape phi_k, actuator mode shape psi_k). This is P&S's Remark (iii) form for
rank-1 residues (System Identification: A Frequency Domain Approach, 2nd ed., section 6.6):
the free-numerator common-denominator model does not identify the poles at MIMO scale
(the numerators absorb pole errors), whereas the rank-1 modal form does. Coupling lives in
the mode shapes, not the residue rank. Frequency normalization (section 12.3.3) keeps the
high-order evaluation conditioned. See `.llm/pintelon-schoukens-mimo-fit.md` and
`docs/superpowers/specs/2026-06-22-joint-mimo-parametric-fit-design.md`.
"""
from __future__ import annotations
import numpy as np


class Rank1ModalModel:
    """Rank-1 modal model with shared poles; analytic Jacobian; roots -> f0/Q."""

    def __init__(self, n_sens: int, n_act: int, n_modes: int):
        self.n_sens = int(n_sens)
        self.n_act = int(n_act)
        self.n_modes = int(n_modes)
        self.per = 2 + self.n_sens + self.n_act      # a, b, phi(n_sens), psi(n_act)
        self.n_theta = self.n_modes * self.per
        self.s_ref = 1.0

    def set_reference(self, freq) -> "Rank1ModalModel":
        """Set s_ref = 2*pi*median(freq); the model then evaluates in sn = s/s_ref."""
        self.s_ref = float(2 * np.pi * np.median(np.asarray(freq, float)))
        return self

    def unpack(self, theta):
        """Return [(a_k, b_k, phi_k, psi_k), ...] for k = 0..n_modes-1."""
        out = []
        for k in range(self.n_modes):
            o = k * self.per
            a = theta[o]
            b = theta[o + 1]
            phi = theta[o + 2:o + 2 + self.n_sens]
            psi = theta[o + 2 + self.n_sens:o + self.per]
            out.append((a, b, phi, psi))
        return out

    def pack(self, ab, phi, psi) -> np.ndarray:
        """Assemble theta from poles ab=[(a,b),...] and shapes phi (M,n_sens), psi (M,n_act)."""
        v = []
        for k in range(self.n_modes):
            v += [float(ab[k][0]), float(ab[k][1])]
            v += list(np.asarray(phi[k], float))
            v += list(np.asarray(psi[k], float))
        return np.asarray(v, float)

    def _sn(self, freq):
        return 2j * np.pi * np.asarray(freq, float) / self.s_ref

    def eval(self, theta, freq) -> np.ndarray:
        """G of shape (len(freq), n_sens, n_act)."""
        sn = self._sn(freq)
        G = np.zeros((len(sn), self.n_sens, self.n_act), complex)
        for (a, b, phi, psi) in self.unpack(theta):
            D = sn * sn + b * sn + a
            G += np.outer(phi, psi)[None] / D[:, None, None]
        return G

    def jacobian(self, theta, freq) -> np.ndarray:
        """Analytic dG/dtheta of shape (len(freq), n_sens, n_act, n_theta)."""
        sn = self._sn(freq)
        F = len(sn)
        J = np.zeros((F, self.n_sens, self.n_act, self.n_theta), complex)
        for k, (a, b, phi, psi) in enumerate(self.unpack(theta)):
            o = k * self.per
            D = sn * sn + b * sn + a
            R = np.outer(phi, psi)
            J[:, :, :, o] = -R[None] / (D * D)[:, None, None]                 # d/da_k
            J[:, :, :, o + 1] = -(R[None]) * (sn / (D * D))[:, None, None]    # d/db_k
            for ii in range(self.n_sens):                                    # d/dphi_k,ii
                J[:, ii, :, o + 2 + ii] = psi[None, :] / D[:, None]
            for jj in range(self.n_act):                                     # d/dpsi_k,jj
                J[:, :, jj, o + 2 + self.n_sens + jj] = phi[None, :] / D[:, None]
        return J

    def poles(self, theta):
        """Physical poles as a sorted list of (f0_Hz, Q)."""
        out = []
        for (a, b, phi, psi) in self.unpack(theta):
            for lam in np.roots([1.0, b, a]) * self.s_ref:   # un-normalize sn -> s
                if lam.imag <= 0:
                    continue
                f0 = abs(lam) / (2 * np.pi)
                Q = abs(lam) / (-2 * lam.real) if lam.real < 0 else np.inf
                out.append((f0, Q))
        return sorted(out, key=lambda t: t[0])

    def ab_from_modes(self, modes):
        """Normalized (a_k, b_k) for modes=[(f0,Q),...]."""
        ab = []
        for f0, q in modes:
            w = 2 * np.pi * f0
            ab.append(((w * w) / self.s_ref ** 2, (w / q) / self.s_ref))
        return ab
