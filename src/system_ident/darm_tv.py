"""Track a slowly-drifting DARM parameter and fit its time variation.

Round-1 system-ID capability — *not* physically-accurate drift yet.  We inject a
known slow variation into one scalar loop parameter (a stage actuation strength
κ) and show the Pintelon–Schoukens machinery recovers θ(t).

Because the plant drifts far slower than one measurement record, each record is
locally stationary (P&S §14.3.4.1: a linear model around a slowly-moving operating
point).  So the estimator is two-step:

1. **Snapshot** — at a sequence of times take a leakage-free P&S measurement and
   recover the instantaneous κ with Pcal as the ruler (`recover_actuation`).  The
   sensing C cancels in H_stage/H_pcal, so the κ snapshot is immune to sensing
   drift.  Each snapshot carries an honest per-estimate σ.
2. **Basis fit** — fit θ(t)=Σ c_k b_k(t) in a time-basis to the snapshot series by
   weighted least squares (a Lataire–Pintelon basis expansion, in two-step form).
   The coefficient covariance (BᵀWB)⁻¹ IS the Cramér–Rao bound on the drift curve
   θ(t) and its rate θ̇(t), given honest per-snapshot σ.

The local-stationarity approximation costs O(record / drift-timescale); with a
~16 s record and an hour-scale drift that is <1 %, far below the per-snapshot σ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Import .darm FIRST: it defines DARMLoop and then does a cycle-safe *bottom* import of
# darm_adapter, so darm must be fully initialised before we pull DARMBackend — otherwise a
# fresh `import darm_tv` (e.g. the docs render, where nothing has loaded darm yet) hits the
# darm ↔ darm_adapter cycle mid-initialisation.
from .darm import DARMLoop, recover_actuation  # noqa: F401  (DARMLoop re-exported for callers)
from .backends.darm_adapter import DARMBackend
from .excitation import multisine_from_psd
from .loop import SysIDLoop


# ── measurement front end (reuses the existing leakage-free P&S estimator) ──────────
def _band_grid(loop, nperseg):
    fa = np.fft.rfftfreq(int(nperseg), d=1.0 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def _frf(loop, port, freq, band, nperseg, n_periods, px_total, seed):
    """One leakage-free closed-loop FRF for an injection ``port`` (Pcal or a stage)."""
    channel = "PCAL_EXC" if port == "PCAL" else "EXC"
    Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    be = DARMBackend(loop, {channel: port}, "DARM_ERR", seed=seed)
    x = multisine_from_psd(Pxx, loop.fs, nperseg, n_periods, freq,
                           seed=np.random.default_rng(seed))
    be.inject(channel, x, loop.fs)
    seg = be.read([channel, "DARM_ERR"], (nperseg * n_periods) / loop.fs)
    return SysIDLoop._estimate_tf_periodic(seg[channel], seg["DARM_ERR"],
                                           loop.fs, nperseg, band, n_transient=1)


def snapshot_kappa(base_loop, name, kappa_value, *, nperseg=4096, n_periods=16,
                   px_total=1.0, seed=0):
    """One leakage-free snapshot of stage strength κ_<name> at an operating point.

    Sets κ_<name>=``kappa_value`` on a copy of ``base_loop``, injects a Pcal
    reference and a stage multisine, and recovers κ with Pcal as the ruler.
    Returns ``(kappa_hat, sigma_kappa)``.
    """
    loop = base_loop.with_params(**{f"kappa_{name}": kappa_value})
    fa, band, freq = _band_grid(loop, nperseg)
    Hp, Hp_err, _ = _frf(loop, "PCAL", freq, band, nperseg, n_periods, px_total, seed)
    Hi, Hi_err, _ = _frf(loop, name, freq, band, nperseg, n_periods, px_total, seed + 1)
    tf, _ = loop.stages[name]
    N = tf.eval(freq)
    comb_err = np.hypot(Hi_err / np.abs(Hi), Hp_err / np.abs(Hp)) * np.abs(Hi / Hp)
    return recover_actuation(freq, Hi, Hp, N, comb_err)


def track_kappa(base_loop, name, times, profile, *, nperseg=4096, n_periods=16,
                px_total=1.0, seed=0):
    """Snapshot κ_<name> at every t in ``times`` with true value ``profile(t)``.

    ``profile`` is a callable t→κ_true (e.g. a ``functools.partial`` of
    ``darm.drift_profile``).  Returns ``(times, kappa_hat, sigma)`` arrays — the
    measured drift time-series to be handed to :func:`fit_tv`.
    """
    times = np.asarray(times, dtype=float)
    khat = np.empty(len(times))
    sig = np.empty(len(times))
    for j, t in enumerate(times):
        khat[j], sig[j] = snapshot_kappa(base_loop, name, float(profile(t)),
                                         nperseg=nperseg, n_periods=n_periods,
                                         px_total=px_total, seed=seed + j)
    return times, khat, sig


# ── time-basis expansion + CRB (the Lataire–Pintelon TV fit) ────────────────────────
def basis_matrix(t, *, kind="legendre", order=4, t0=None, t1=None):
    """Design matrix ``B[j,k]=b_k(t_j)`` and its time-derivative ``dB[j,k]=ḃ_k(t_j)``.

    ``kind="legendre"``: Legendre polynomials in s=2(t−t0)/(t1−t0)−1 ∈ [−1,1] up to
    degree ``order`` (order+1 columns) — well-conditioned, assumes no drift period.
    ``kind="fourier"``: [1, cos(mωt), sin(mωt)]_{m=1..order}, ω=2π/(t1−t0).
    """
    t = np.asarray(t, dtype=float)
    t0 = float(np.min(t) if t0 is None else t0)
    t1 = float(np.max(t) if t1 is None else t1)
    span = (t1 - t0) or 1.0
    if kind == "legendre":
        s = 2.0 * (t - t0) / span - 1.0
        dsdt = 2.0 / span
        B = np.polynomial.legendre.legvander(s, order)
        dB = np.zeros_like(B)
        for k in range(order + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            dc = np.polynomial.legendre.legder(c) if k >= 1 else np.zeros(1)
            dB[:, k] = np.polynomial.legendre.legval(s, dc) * dsdt
        return B, dB
    if kind == "fourier":
        w = 2.0 * np.pi / span
        cols = [np.ones_like(t)]
        dcols = [np.zeros_like(t)]
        for m in range(1, order + 1):
            cols += [np.cos(m * w * t), np.sin(m * w * t)]
            dcols += [-m * w * np.sin(m * w * t), m * w * np.cos(m * w * t)]
        return np.column_stack(cols), np.column_stack(dcols)
    raise ValueError(f"unknown basis {kind!r}")


@dataclass
class TVFit:
    """Result of a time-varying basis fit; ``.predict`` gives θ(t) and θ̇(t) with CRB."""

    coeffs: np.ndarray
    cov: np.ndarray
    kind: str
    order: int
    t0: float
    t1: float

    def predict(self, t_new):
        """Return ``(theta, sigma_theta, theta_dot, sigma_theta_dot)`` at ``t_new``."""
        t_new = np.atleast_1d(np.asarray(t_new, dtype=float))
        B, dB = basis_matrix(t_new, kind=self.kind, order=self.order,
                             t0=self.t0, t1=self.t1)
        theta = B @ self.coeffs
        theta_dot = dB @ self.coeffs
        var = np.einsum("jk,kl,jl->j", B, self.cov, B)
        var_dot = np.einsum("jk,kl,jl->j", dB, self.cov, dB)
        return (theta, np.sqrt(np.clip(var, 0.0, None)),
                theta_dot, np.sqrt(np.clip(var_dot, 0.0, None)))


def fit_tv(t, y, sigma, *, kind="legendre", order=4) -> TVFit:
    """Weighted-LS basis fit of θ(t)=Σ c_k b_k(t) to snapshots ``(t, y ± sigma)``.

    The coefficient covariance (BᵀWB)⁻¹ with W=diag(1/σ²) is the CRB on the drift
    curve for honest per-snapshot σ.  Non-finite / non-positive-σ points are dropped.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    good = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if np.count_nonzero(good) <= order:
        raise ValueError("not enough good snapshots for the requested basis order")
    t0, t1 = float(np.min(t[good])), float(np.max(t[good]))
    B, _ = basis_matrix(t[good], kind=kind, order=order, t0=t0, t1=t1)
    w = 1.0 / sigma[good] ** 2
    BtW = B.T * w
    cov = np.linalg.inv(BtW @ B)
    coeffs = cov @ (BtW @ y[good])
    return TVFit(coeffs, cov, kind, order, t0, t1)


# ── feasibility gate: is the injected drift resolvable? ─────────────────────────────
def resolvability(fit: TVFit, *, base, amp_frac, period_s, kind="sine",
                  record_s=None, n_grid=400):
    """Numbers for the feasibility gate — no verdict without the bound.

    Compares the *tracking* uncertainty σ_θ(t) (from the fit CRB) to the injected
    drift amplitude, and σ_θ̇ to the true peak drift rate.  ``resolve_ratio ≫ 1``
    means the drift is resolved, not noise; a small ratio means add SNR / snapshots,
    not "it's a limit".
    """
    ts = np.linspace(fit.t0, fit.t1, int(n_grid))
    _, s_theta, _, s_dot = fit.predict(ts)
    drift_amp = base * amp_frac
    true_peak_rate = (base * amp_frac * 2.0 * np.pi / period_s if kind == "sine"
                      else base * amp_frac / period_s)
    out = {
        "drift_amp": float(drift_amp),
        "sigma_theta_med": float(np.median(s_theta)),
        "resolve_ratio": float(drift_amp / np.median(s_theta)),
        "true_peak_rate": float(true_peak_rate),
        "sigma_theta_dot_med": float(np.median(s_dot)),
        "rate_resolve_ratio": float(true_peak_rate / np.median(s_dot)) if np.median(s_dot) > 0 else np.inf,
    }
    if record_s is not None:
        out["local_stationarity_err"] = float(record_s / period_s)
    return out
