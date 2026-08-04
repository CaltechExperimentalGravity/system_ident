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
    # Full actuation shape D_i·N_i (excludes only κ), so the ruler recovers κ even when a
    # hierarchical distribution filter D_i is set. Without distribution this is just N_mech.
    N = loop.stage(name, freq) / kappa_value
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


def cal_line_response(base_loop, freqs, *, nperseg=4096, n_periods=16, px_total=1.0, seed=0):
    """Inject calibration lines on every actuation stage; return the ruler-calibrated per-stage
    actuation ``A_i = κ_i·D_i·N_i`` at the line frequencies.

    The hierarchical DARM actuation drives several quad masses (M0/PUM/TST); to see the crossover
    of each stage with the one below, put cal lines near where the stages hand off and compare
    ``|A_i|`` across stages. Each line is a fixed tone on one stage's drive; measured against a
    Pcal reference, ``H_stage/H_pcal = A_i`` cancels the sensing ``C`` and the loop ``1/(1+G)``,
    leaving the stage actuation itself — exactly the calibration quantity. (The crossover only
    appears once the ``distribution`` filters are populated; without them the raw mechanical
    columns don't cross.)

    Parameters
    ----------
    freqs : cal-line frequencies [Hz], shared across stages (snapped to the rfft grid so the
        lines are leakage-free). Place them near the expected crossovers.

    Returns ``{stage: (lines_hz, A_complex, A_sigma)}`` — the per-stage actuation and its 1σ at
    each cal line. ``A_sigma`` propagates the stage- and Pcal-FRF errors (the honest CRB on the
    line measurement).
    """
    loop = base_loop
    fa = np.fft.rfftfreq(int(nperseg), d=1.0 / loop.fs)
    bins = np.unique([int(round(f * nperseg / loop.fs)) for f in np.atleast_1d(freqs)])
    bins = bins[(fa[bins] >= loop.fmin) & (fa[bins] <= loop.fmax)]
    if len(bins) == 0:
        raise ValueError("no cal-line frequencies fall in the loop band")
    lines = fa[bins]
    band = np.zeros(len(fa), bool); band[bins] = True
    Hp, Hp_err, _ = _frf(loop, "PCAL", lines, band, nperseg, n_periods, px_total, seed)
    out = {}
    for k, name in enumerate(loop.stages):
        Hi, Hi_err, _ = _frf(loop, name, lines, band, nperseg, n_periods, px_total, seed + 1 + k)
        A = Hi / Hp                                      # = κ_i·D_i·N_i (ruler-calibrated)
        A_sigma = np.abs(A) * np.hypot(Hi_err / np.abs(Hi), Hp_err / np.abs(Hp))
        out[name] = (lines, A, A_sigma)
    return out


def snapshot_delta(base_loop, delta_value, *, nperseg=4096, n_periods=16,
                   px_total=1.0, seed=0, delta_init=None):
    """One leakage-free snapshot of the SRC detuning δ at an operating point.

    δ is a **sensing** parameter (it does not cancel in the κ ruler), so it is recovered from
    the **Pcal FRF shape**: ``C_meas = H_pcal·(1+G)`` (G is the designed, known open-loop gain),
    then a weighted complex least-squares fit of the coupled detuned-cavity sensing model for δ,
    with the other sensing params held at their loop values. Returns ``(delta_hat, sigma_delta)``.

    The coupled term ``α·u² = detune_coupling·sin(2δ)·(f/f_cc)²`` is linear in δ through the tuned
    point (``sin2δ ≈ 2δ``) and grows as ``f²``, so — perhaps counter-intuitively — δ is
    well-identified even near BRSE (small detuning leaves a measurable high-frequency curvature in
    ``C``); ``sigma_delta`` does not blow up at δ→0.
    """
    from scipy.optimize import least_squares
    from .darm import sensing_model_detuned

    loop = base_loop.with_params(delta=delta_value)
    fa, band, freq = _band_grid(loop, nperseg)
    Hp, Hp_err, _ = _frf(loop, "PCAL", freq, band, nperseg, n_periods, px_total, seed)
    one_plus_G = 1.0 + loop.G(freq)
    C_meas = Hp * one_plus_G                        # = C exactly, plus measurement noise
    C_err = np.maximum(Hp_err * np.abs(one_plus_G), 1e-30)

    def model(delta):
        alpha = loop.detune_coupling * np.sin(2.0 * delta)
        return sensing_model_detuned(freq, loop.g_c, loop.f_cc, loop.tau, alpha)

    def resid(p):
        r = (C_meas - model(p[0])) / C_err
        return np.concatenate([r.real, r.imag])

    p0 = 0.5 * delta_value if delta_init is None else delta_init   # init off the truth
    sol = least_squares(resid, [p0 if p0 != 0 else 0.02], method="lm")
    delta_hat = float(sol.x[0])
    # Gauss–Newton CRB: cov = (JᵀJ)⁻¹ on the whitened residual
    JtJ = sol.jac.T @ sol.jac
    sigma = float(np.sqrt(np.linalg.inv(JtJ)[0, 0])) if JtJ[0, 0] > 0 else np.inf
    return delta_hat, sigma


def track_delta(base_loop, times, profile, *, nperseg=4096, n_periods=16,
                px_total=1.0, seed=0):
    """Snapshot δ at every t (true value ``profile(t)``). Returns ``(times, delta_hat, sigma)``
    for :func:`fit_tv` — the same TV machinery as κ, since it is parameter-agnostic."""
    times = np.asarray(times, dtype=float)
    dhat = np.empty(len(times)); sig = np.empty(len(times))
    for j, t in enumerate(times):
        dhat[j], sig[j] = snapshot_delta(base_loop, float(profile(t)), nperseg=nperseg,
                                         n_periods=n_periods, px_total=px_total, seed=seed + j)
    return times, dhat, sig


# ── realistic random drift + joint (several-parameter-at-once) tracking ──────────────
def stochastic_drift(times, base, *, amp_frac=0.05, tau_s=1800.0, seed=0):
    """A **random** drift sample θ(t): a mean-reverting (Ornstein–Uhlenbeck) wander about ``base``
    with correlation time ``tau_s`` [s] and stationary std ``amp_frac·base`` — a meandering,
    physically-realistic drift rather than the smooth deterministic curve of
    :func:`~system_ident.darm.drift_profile`. Deterministic given ``seed`` so the injected truth is
    exactly known when scoring the recovery. Returns θ at each ``times`` entry."""
    times = np.asarray(times, dtype=float)
    rng = np.random.default_rng(seed)
    sd = amp_frac * base
    x = np.empty(len(times))
    x[0] = sd * rng.standard_normal()
    for i in range(1, len(times)):
        a = np.exp(-(times[i] - times[i - 1]) / tau_s)          # OU decay over the gap
        x[i] = a * x[i - 1] + np.sqrt(max(1.0 - a * a, 0.0)) * sd * rng.standard_normal()
    return base + x


def joint_snapshot(base_loop, truth: dict, *, nperseg=4096, n_periods=16, px_total=1.0, seed=0):
    """Recover SEVERAL drifting parameters at once from one set of leakage-free records, with their
    joint covariance — so a wobble in one parameter is *untangled* from the others.

    ``truth`` maps knob → true value; keys are ``DARMLoop`` scalar fields (``g_c, f_cc, delta,
    tau``) and/or ``kappa_<STAGE>``. A Pcal record constrains the sensing knobs; each drifting
    stage adds its own record for that κ. All records are fit JOINTLY by weighted complex least
    squares to the model FRFs (``C/(1+G)`` for Pcal, ``C·κ_i·D_iN_i/(1+G)`` for a stage), started
    from the base-loop nominal (not the truth). Returns ``(theta_hat, sigma, corr, names)``: dicts
    of recovered values and 1σ, the parameter correlation matrix, and the parameter order."""
    from scipy.optimize import least_squares
    from .darm import sensing_model_detuned

    names = list(truth)
    loop = base_loop.with_params(**truth)
    fa, band, freq = _band_grid(loop, nperseg)
    inv1pG = 1.0 / (1.0 + loop.G(freq))                          # G is designed (θ-independent)
    stages = [n[len("kappa_"):] for n in names if n.startswith("kappa_")]
    Hp, Hp_err, _ = _frf(loop, "PCAL", freq, band, nperseg, n_periods, px_total, seed)
    meas = {"PCAL": (Hp, Hp_err)}
    DN = {}                                                      # κ=1 stage shape (D_i·N_i), once
    for k, st in enumerate(stages):
        meas[st] = _frf(loop, st, freq, band, nperseg, n_periods, px_total, seed + 1 + k)[:2]
        DN[st] = base_loop.stage(st, freq) / base_loop.stages[st][1]
    base = {"g_c": base_loop.g_c, "f_cc": base_loop.f_cc,
            "delta": base_loop.delta, "tau": base_loop.tau}

    def C_of(d):
        alpha = base_loop.detune_coupling * np.sin(2.0 * d.get("delta", base["delta"]))
        return sensing_model_detuned(freq, d.get("g_c", base["g_c"]),
                                     d.get("f_cc", base["f_cc"]), d.get("tau", base["tau"]), alpha)

    def resid(p):
        d = dict(zip(names, p))
        C = C_of(d)
        Hpm, Hpe = meas["PCAL"]
        g = np.isfinite(Hpe) & (Hpe > 0)
        r = [((Hpm[g] - C[g] * inv1pG[g]) / Hpe[g]).real,
             ((Hpm[g] - C[g] * inv1pG[g]) / Hpe[g]).imag]
        for st in stages:
            Him, Hie = meas[st]
            kap = d.get("kappa_" + st, base_loop.stages[st][1])
            model = C * kap * DN[st] * inv1pG
            gi = np.isfinite(Hie) & (Hie > 0)
            r += [((Him[gi] - model[gi]) / Hie[gi]).real, ((Him[gi] - model[gi]) / Hie[gi]).imag]
        return np.concatenate(r)

    p0 = np.array([base["g_c"] if n == "g_c" else base["f_cc"] if n == "f_cc"
                   else base["delta"] if n == "delta" else base["tau"] if n == "tau"
                   else base_loop.stages[n[len("kappa_"):]][1] for n in names], dtype=float)
    sol = least_squares(resid, p0, method="trf", x_scale="jac")   # auto-scale disparate magnitudes
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    s = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(np.outer(s, s) > 0, cov / np.outer(s, s), 0.0)
    return dict(zip(names, sol.x)), dict(zip(names, s)), corr, names


def track_joint(base_loop, truth_series: dict, times, *, nperseg=4096, n_periods=16,
                px_total=1.0, seed=0):
    """Track several parameters drifting **simultaneously**. ``truth_series`` maps knob → an array
    of true values aligned with ``times`` (from :func:`~system_ident.darm.drift_profile` or
    :func:`stochastic_drift`). Each time is a :func:`joint_snapshot`. Returns ``(times, theta_hat,
    sigma, corr_mean, names)`` — per-parameter recovered arrays, 1σ arrays, and the mean snapshot
    correlation matrix (off-diagonals show which parameters are hard to separate)."""
    times = np.asarray(times, dtype=float)
    names = list(truth_series)
    th = {n: np.empty(len(times)) for n in names}
    sg = {n: np.empty(len(times)) for n in names}
    corr_sum = None
    for j in range(len(times)):
        truth = {n: float(truth_series[n][j]) for n in names}
        theta, sigma, corr, _ = joint_snapshot(base_loop, truth, nperseg=nperseg,
                                               n_periods=n_periods, px_total=px_total, seed=seed + j)
        for n in names:
            th[n][j] = theta[n]; sg[n][j] = sigma[n]
        corr_sum = corr if corr_sum is None else corr_sum + corr
    return times, th, sg, corr_sum / len(times), names


# ── line-based readout: inject the DESIGNED cal lines (not a broadband multisine) ────────────────
def _line_bins(loop, nperseg, freqs):
    """Snap ``freqs`` to the synchronous DFT grid ``k·fs/nperseg`` and return ``(snapped_freqs,
    band_mask)`` over the full rfft grid. Line injection + integer-period DFT ⇒ leakage-free readout
    at exactly these bins (the same synchronous-grid trick the broadband path uses, restricted to a
    few lines)."""
    fa = np.fft.rfftfreq(int(nperseg), d=1.0 / loop.fs)
    bins = np.unique([int(round(f * nperseg / loop.fs)) for f in np.atleast_1d(freqs)])
    band = np.zeros(len(fa), bool)
    band[bins] = True
    return fa[bins], band


def _frf_lines(loop, port, lines_hz, rms_amps, nperseg, n_periods, seed):
    """Leakage-free FRF measured ONLY at the designed line frequencies for one port. ``rms_amps`` is
    the injected line amplitude per line in the port's native rms units (Pcal displacement [m], or
    stage drive [ct]). Returns ``(freqs, H, H_err)`` at the grid-snapped lines."""
    channel = "PCAL_EXC" if port == "PCAL" else "EXC"
    freqs, band = _line_bins(loop, nperseg, lines_hz)
    df = loop.fs / nperseg
    Pxx = (np.asarray(rms_amps, dtype=float) ** 2) / df           # one line per freq bin
    be = DARMBackend(loop, {channel: port}, "DARM_ERR", seed=seed)
    x = multisine_from_psd(Pxx, loop.fs, nperseg, n_periods, freqs, seed=np.random.default_rng(seed))
    be.inject(channel, x, loop.fs)
    seg = be.read([channel, "DARM_ERR"], (nperseg * n_periods) / loop.fs)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg[channel], seg["DARM_ERR"], loop.fs, nperseg,
                                                  band, n_transient=1)
    return freqs, H, H_err


def joint_snapshot_lines(base_loop, truth: dict, roster, *, nperseg=4096, n_periods=16, seed=0):
    """Joint TDCF snapshot from the **designed** cal lines — the optimal few-line excitation instead
    of a broadband multisine.

    ``roster`` = ``[(freq_hz, port, disp_amp_m)]`` from
    :func:`system_ident.darm_callines.design_lines`: each entry is one line, its port
    (``'PCAL'``/stage), and the DARM displacement it makes on the base loop. The injected DRIVE is
    held fixed (computed once from ``base_loop``) as θ drifts — a real always-on cal line has fixed
    drive. Each port is injected at its own line(s), read leakage-free, and all ports are fit JOINTLY
    for ``θ = truth.keys()`` (``C/(1+G)`` for Pcal, ``C·κ_i·D_iN_i/(1+G)`` for a stage — a stage line
    depends on the sensing params too, which is the joint coupling). Returns
    ``(theta_hat, sigma, corr, names)`` like :func:`joint_snapshot`."""
    from scipy.optimize import least_squares
    from .darm import sensing_model_detuned

    names = list(truth)
    loop = base_loop.with_params(**truth)
    stages = [n[len("kappa_"):] for n in names if n.startswith("kappa_")]
    base = {"g_c": base_loop.g_c, "f_cc": base_loop.f_cc,
            "delta": base_loop.delta, "tau": base_loop.tau}
    # group by port; convert disp_amp → fixed injected rms amp in the port's native units, using the
    # grid-SNAPPED frequency so the drive conversion matches the (snapped) injection + fit (a stage
    # TF is steep, so an unsnapped conversion would bias κ).
    def _snap(f):
        return round(f * nperseg / base_loop.fs) * base_loop.fs / nperseg
    by_port = {}
    for f, port, disp in roster:
        fs_ = _snap(float(f))
        amp = disp if port == "PCAL" else disp / abs(base_loop.stage(port, [fs_])[0])
        fl, al = by_port.setdefault(port, ([], []))
        fl.append(fs_); al.append(float(amp))
    meas = {}
    for k, (port, (fl, al)) in enumerate(by_port.items()):
        meas[port] = _frf_lines(loop, port, fl, al, nperseg, n_periods, seed + k)
    DN = {st: base_loop.stage(st, meas[st][0]) / base_loop.stages[st][1]
          for st in stages if st in meas}                          # κ=1 stage shape at its line(s)

    def C_of(d, freq):
        alpha = base_loop.detune_coupling * np.sin(2.0 * d.get("delta", base["delta"]))
        return sensing_model_detuned(freq, d.get("g_c", base["g_c"]),
                                     d.get("f_cc", base["f_cc"]), d.get("tau", base["tau"]), alpha)

    def resid(p):
        d = dict(zip(names, p))
        r = []
        for port, (fr, H, He) in meas.items():
            inv1pG = 1.0 / (1.0 + loop.G(fr))
            model = C_of(d, fr) * inv1pG
            if port != "PCAL":
                model = model * d.get("kappa_" + port, base_loop.stages[port][1]) * DN[port]
            g = np.isfinite(He) & (He > 0)
            r += [((H[g] - model[g]) / He[g]).real, ((H[g] - model[g]) / He[g]).imag]
        return np.concatenate(r)

    p0 = np.array([base[n] if n in base else base_loop.stages[n[len("kappa_"):]][1]
                   for n in names], dtype=float)
    sol = least_squares(resid, p0, method="trf", x_scale="jac")
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    s = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(np.outer(s, s) > 0, cov / np.outer(s, s), 0.0)
    return dict(zip(names, sol.x)), dict(zip(names, s)), corr, names


def track_joint_lines(base_loop, truth_series: dict, times, roster, *, nperseg=4096,
                      n_periods=16, seed=0):
    """Track several parameters drifting simultaneously, read out by the **designed** cal-line roster
    (see :func:`joint_snapshot_lines`). ``truth_series`` maps knob → true-value array aligned with
    ``times``. Returns ``(times, theta_hat, sigma, corr_mean, names)`` — the same shape as
    :func:`track_joint`, so it drops straight into :func:`fit_tv`/:func:`resolvability`."""
    times = np.asarray(times, dtype=float)
    names = list(truth_series)
    th = {n: np.empty(len(times)) for n in names}
    sg = {n: np.empty(len(times)) for n in names}
    corr_sum = None
    for j in range(len(times)):
        truth = {n: float(truth_series[n][j]) for n in names}
        theta, sigma, corr, _ = joint_snapshot_lines(base_loop, truth, roster, nperseg=nperseg,
                                                      n_periods=n_periods, seed=seed + j)
        for n in names:
            th[n][j] = theta[n]; sg[n][j] = sigma[n]
        corr_sum = corr if corr_sum is None else corr_sum + corr
    return times, th, sg, corr_sum / len(times), names


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
