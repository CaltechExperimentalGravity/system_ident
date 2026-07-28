"""Closed-loop system identification under a genuine in-loop disturbance.

The core Pintelon–Schoukens property this module demonstrates and tests: when a control
loop is **closed** and a real disturbance enters **inside** the loop, the naive open-loop
FRF estimator is **biased**, while period-averaging a periodic excitation (the P&S remedy)
is **consistent**.

The signal model, per excited line, with reference ``r``, plant ``G``, controller ``K``
(here ``K`` folds in any per-DOF calibration ``kappa``), and disturbance ``e`` at the
plant output::

    y = G v + e                          (measurement, corrupted by e)
    v = r - K y   =>   v = S r - S K e   (drive monitor; S = (I + K G)^-1)

Because the drive monitor ``v`` carries ``-S K e``, it is correlated with ``e``
(``E[e v*] != 0``), so the naive estimator ``Ĝ = <Y V*>/<V V*>`` is biased by
``-SK* σ_e² / E|V|²`` — pulled toward the inverse controller. Since ``r`` is periodic and
known while ``e`` is not, averaging over periods drives ``V̄ -> S r`` and ``Ȳ -> G S r``, so
``Ȳ / V̄ -> G``: the bias vanishes as ``1/sqrt(P)``.

Everything here is pure numpy + python-control (no twin, no rtsfree), and works uniformly
for SISO (``n = 1``) and MIMO (``n × n`` per line). The P&S estimator reuses
:func:`system_ident.mimo_loop.recover_open_loop`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import control

from .mimo_loop import recover_open_loop


def _eval(sysd, freq_hz):
    w = 2 * np.pi * np.asarray(freq_hz, float)
    return np.atleast_3d(np.moveaxis(control.frequency_response(sysd, w).frdata, -1, 0))


def loop_frf_maps(Gd, Kd, freq_hz, *, balance=True):
    """Per-line plant / sensitivity / (S·K) maps for a discrete plant ``Gd`` and controller
    ``Kd`` (both ``control`` LTI at the same sample rate).

    Returns ``(scale, Gk, Sk, SKk)`` — each map shape ``(len(freq), n, n)`` — with
    ``Gk = G(z)``, ``Sk = (I + K G)^-1``, ``SKk = Sk @ K``.

    ``balance=True`` first applies the diagonal similarity ``s_i = sqrt(max_k |G_ii(k)|)``,
    ``Ĝ = D^-1 G D^-1`` and — crucially — the **reciprocal** scaling on the controller,
    ``K̂ = D K D``, so that the loop is preserved (``K̂ Ĝ = D (K G) D^-1``) and the sensitivity
    comes out correctly as the similarity ``Ŝ = D S D^-1``. ``Sk``/``SKk`` are then formed from
    the balanced pair, never by scaling ``S`` directly (that would use the wrong transform).
    Balancing makes every DOF O(1); a plant whose DOF gains span orders of magnitude (SI
    suspension: translational vs. angular ~1e4x) otherwise lets the large DOF dominate the MIMO
    recovery and the disturbance on them corrupt the small ones. Mode frequencies are invariant.
    With ``balance=False`` the returned ``scale`` is all ones.
    """
    Gk = _eval(Gd, freq_hz)
    Kk = _eval(Kd, freq_hz)
    n = Gk.shape[1]
    if balance:
        s = np.sqrt(np.array([np.max(np.abs(Gk[:, i, i])) for i in range(n)]))
        Dinv = np.diag(1.0 / s); D = np.diag(s)
        Gk = Dinv[None] @ Gk @ Dinv[None]
        Kk = D[None] @ Kk @ D[None]
    else:
        s = np.ones(n)
    Sk = np.linalg.inv(np.eye(n)[None] + Kk @ Gk)
    SKk = Sk @ Kk
    return s, Gk, Sk, SKk


def simulate(Sk, SKk, Gk, R, *, n_periods, sigma, rng):
    """Steady-state periodic closed-loop experiment with an in-loop output disturbance.

    Drives each actuator ``j`` in turn with the periodic reference ``R`` (per-line phasors),
    injecting a fresh disturbance realization each period. Frequency-domain and exact for the
    periodic steady state — no time-stepping. Returns ``(V, Y)`` each shape
    ``(n_exp, n_periods, n_lines, n)``: drive monitors and readouts.

    ``sigma`` is the per-DOF disturbance std (scalar or length-``n``). Work in balanced units
    (see :func:`balance`) so a scalar ``sigma`` is uniform across DOF.
    """
    K = Sk.shape[0]
    n = Sk.shape[1]
    sigma = np.broadcast_to(np.asarray(sigma, float), (n,))
    # reference contribution to the drive monitor: Vref[k, :, j] = R[k] * Sk[k][:, j]
    Vref = R[:, None, None] * Sk                       # (K, out, j)
    E = (rng.standard_normal((n, n_periods, K, n))
         + 1j * rng.standard_normal((n, n_periods, K, n))) / np.sqrt(2)
    E = E * sigma[None, None, None, :]
    V = Vref.transpose(2, 0, 1)[:, None] - np.einsum('kab,jpkb->jpka', SKk, E)
    Y = np.einsum('kab,jpkb->jpka', Gk, V) + E
    return V, Y


def ps_frf(V, Y):
    """Consistent P&S estimate: period-average, then per-line ``Ȳ · X̄^-1``
    (via :func:`~system_ident.mimo_loop.recover_open_loop`). Shape ``(n_lines, n, n)``."""
    Xk = np.moveaxis(V.mean(1), 0, -1)                 # (K, n, n_exp)
    Yk = np.moveaxis(Y.mean(1), 0, -1)
    return recover_open_loop(Xk, Yk)


def naive_frf(V, Y):
    """Biased open-loop-method estimate: MIMO least squares over all periods, no averaging,
    ``Ĝ = (Σ_p Y_p X_p^H)(Σ_p X_p X_p^H)^-1`` per line. Shape ``(n_lines, n, n)``."""
    Xp = np.moveaxis(V, 0, -1)                         # (P, K, n, n_exp)
    Yp = np.moveaxis(Y, 0, -1)
    SYX = np.einsum('pkaj,pkbj->kab', Yp, Xp.conj())
    SXX = np.einsum('pkaj,pkbj->kab', Xp, Xp.conj())
    return SYX @ np.linalg.inv(SXX)


def naive_bias_analytic(Sk, SKk, R, sigma2):
    """Leading-order analytic naive-estimator bias per line, ``-SK* σ² / E|V|²`` with
    ``E|V|² = |S R|² + |SK|² σ²`` — SISO-exact (matches simulation to <4% across decades of
    disturbance). Inputs are the SISO per-line scalars (``(K,)`` after squeezing).
    """
    Sk = np.asarray(Sk).ravel(); SKk = np.asarray(SKk).ravel(); R = np.asarray(R).ravel()
    EV2 = np.abs(Sk * R) ** 2 + np.abs(SKk) ** 2 * sigma2
    return -np.conj(SKk) * sigma2 / EV2


@dataclass
class ConvergenceResult:
    n_periods: np.ndarray          # the P values swept
    naive_err: np.ndarray          # mean recovery error vs true FRF, naive estimator
    ps_err: np.ndarray             # ... P&S estimator


def sweep_periods(Sk, SKk, Gk, R, Gtrue, *, periods, sigma, seeds, err_fn):
    """Run :func:`simulate` + both estimators over a list of period counts and seeds.

    ``err_fn(Ghat, Gtrue) -> float`` scores a recovered FRF (e.g. per-line diagonal relative
    error). Returns a :class:`ConvergenceResult` with seed-averaged errors, for the gate:
    ``naive_err`` plateaus (bias floor independent of P), ``ps_err`` falls ~``1/sqrt(P)``.
    """
    periods = list(periods)
    ne = np.zeros(len(periods)); pe = np.zeros(len(periods))
    for i, P in enumerate(periods):
        for sd in range(seeds):
            V, Y = simulate(Sk, SKk, Gk, R, n_periods=P, sigma=sigma,
                            rng=np.random.default_rng(sd))
            ne[i] += err_fn(naive_frf(V, Y), Gtrue)
            pe[i] += err_fn(ps_frf(V, Y), Gtrue)
    return ConvergenceResult(np.array(periods), ne / seeds, pe / seeds)
