"""Fisher-optimal calibration-line design for the DARM TDCFs.

The calibration group tracks a handful of *time-dependent correction factors* (TDCFs) with
always-on calibration lines — a Pcal displacement reference plus one actuator line per suspension
stage. This module answers, the Pintelon–Schoukens way: **where to put the lines and how big to
make them so every required parameter is measured to a target precision (0.1%) in a target time
(< 5 min)** — and how that compares to the LIGO O3/O4 line schemes.

Parameter vector θ (the coupled-cavity twin, :class:`~system_ident.darm.DARMLoop`):

    sensing   κ_C (=g_c) · f_cc · δ (SRC detuning; splits the cavity pole) · τ
    actuation κ_M0 · κ_PUM · κ_TST   (hierarchical M0 / PUM / ESD drive)

Each calibration line is a monochromatic injection whose closed-loop FRF is measured against the
DARM noise floor. For a line of DARM-displacement amplitude ``a`` at ``f`` over time ``T`` against
floor ``S_x(f) = disturbance² + (readout/|C|)²`` [m²/Hz], the FRF is measured with SNR
``= a·√(T)/√S_x`` and fractional error ``1/SNR``. The Fisher information on θ (in *fractional* /
log parameters) is the standard complex-FRF sum

    Γ_ab = Σ_lines 2·SNR² · Re[ conj(∂lnH/∂lnθ_a) · ∂lnH/∂lnθ_b ].

Every observable is ``∝ C(f;θ)`` — Pcal lines observe ``H = C/(1+G)``; stage lines observe
``H = C·κ_i·(D_iN_i)/(1+G)``. So ``∂lnH/∂lnθ = ∂lnC/∂lnθ`` for the sensing params (cheap, analytic
in C) and ``∂lnH/∂lnκ_i = 1`` (0 for the other stages / Pcal). The θ-independent pieces (``G`` and
the stage shapes ``D_iN_i``) are precomputed once, so the plant is never re-solved in the sizing
loop. The CRB is ``Γ⁻¹`` via :func:`_crb_cov` — an eigenvalue-floored inverse so an
under-constrained parameter comes back with a large σ rather than a spurious zero variance;
``σ_θ = √diag`` is the fractional 1σ. Same math as :mod:`fisher` (the ``2·Re[∂H*·∂H]·weight``
structure), specialised to the DARM parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .darm import DARMLoop, darm_design_asd, sensing_model_detuned
from .fisher import safe_inverse

# ── the parameter vector θ ──────────────────────────────────────────────────────
#: Sensing params in θ order and their DARMLoop knobs; then the three stage strengths.
_SENSING = [("kappa_C", "g_c"), ("f_cc", "f_cc"), ("delta", "delta"), ("tau", "tau")]
STAGE_KINDS = ("M0", "PUM", "TST")
TDCF_PARAMS: list[str] = [name for name, _ in _SENSING] + [f"kappa_{s}" for s in STAGE_KINDS]

# Rough LIGO reference line positions [Hz] (O3: Sun 2020 §4.1; O4: arXiv:2508.08423). Frequencies
# only — for the head-to-head every scheme is sized to the same total drive (exact per-line
# amplitude match is a later phase). Each entry (freq, kind, target-param note).
O3_LINES = [(7.9, "PCAL", "delta"), (17.1, "PCAL", "kappa_C"), (331.9, "PCAL", "f_cc"),
            (1083.7, "PCAL", "tau"), (15.6, "M0", "kappa_M0"), (16.4, "PUM", "kappa_PUM"),
            (35.9, "TST", "kappa_TST")]
O4_LINES = [(6.5, "PCAL", "delta"), (17.1, "PCAL", "kappa_C"), (33.4, "PCAL", "kappa_C"),
            (410.3, "PCAL", "f_cc"), (1083.1, "PCAL", "tau"), (15.1, "M0", "kappa_M0"),
            (16.9, "PUM", "kappa_PUM"), (34.7, "TST", "kappa_TST")]


@dataclass
class Line:
    """One calibration line: frequency [Hz], kind (``"PCAL"`` or a stage in :data:`STAGE_KINDS`),
    DARM-displacement amplitude [m rms], and an optional ``target`` TDCF note."""
    freq: float
    kind: str
    amp: float = 1.0
    target: str = ""


def default_cal_loop(delta_deg: float = 5.0) -> DARMLoop:
    """The base twin for cal-line design: the M0-damped hierarchical reduced-quad loop at a
    representative slight SRC detuning (so every TDCF, including δ, is nonzero and fractional
    precision is well defined), with the **real** Advanced-LIGO DARM displacement-noise floor
    (:func:`~system_ident.darm.darm_design_asd`)."""
    loop = DARMLoop.default_reduced(fmin=0.3, hierarchical=True).with_params(
        delta=np.radians(delta_deg))
    loop.noise_asd = darm_design_asd
    return loop


def floor_asd(loop: DARMLoop, freq) -> np.ndarray:
    """DARM-displacement noise floor [m/√Hz] — the real ``noise_asd`` curve if set, else the legacy
    two-scalar model (see :meth:`DARMLoop.displacement_noise_asd`)."""
    return loop.displacement_noise_asd(np.asarray(freq, dtype=float))


def _sensing_C(loop: DARMLoop, freqs: np.ndarray, over: dict) -> np.ndarray:
    """Sensing ``C(f)`` with the sensing params optionally overridden (for the log-Jacobian)."""
    g_c = over.get("g_c", loop.g_c)
    f_cc = over.get("f_cc", loop.f_cc)
    delta = over.get("delta", loop.delta)
    tau = over.get("tau", loop.tau)
    alpha = loop.detune_coupling * np.sin(2.0 * delta)
    return sensing_model_detuned(freqs, g_c, f_cc, tau, alpha)


# ── a precomputed line set: θ-independent pieces + the log-Jacobian ──────────────────────
@dataclass
class LineSet:
    """Frozen geometry of a roster: frequencies, kinds, the ``floor``, and the log-Jacobian
    ``J[j,a] = ∂lnH_j/∂lnθ_a`` — all θ-independent (evaluated once at the loop's operating point).
    Only per-line amplitudes and the integration time vary afterwards, so :func:`fisher` never
    re-solves the plant."""
    freqs: np.ndarray
    kinds: list[str]
    floor: np.ndarray
    J: np.ndarray          # (n_lines, n_params) complex


def build_lineset(loop: DARMLoop, lines: list[Line], step: float = 1e-6) -> LineSet:
    freqs = np.array([ln.freq for ln in lines], dtype=float)
    kinds = [ln.kind for ln in lines]
    # dlnC/dlnθ for the four sensing params (central difference on C only — cheap, analytic)
    C0 = _sensing_C(loop, freqs, {})
    J = np.zeros((len(lines), len(TDCF_PARAMS)), dtype=complex)
    for a, (_, knob) in enumerate(_SENSING):
        v = float(getattr(loop, knob))
        h = step * (abs(v) if v != 0 else 1.0)
        Cp = _sensing_C(loop, freqs, {knob: v + h})
        Cm = _sensing_C(loop, freqs, {knob: v - h})
        J[:, a] = v * (Cp - Cm) / (2.0 * h) / C0     # ∂lnC/∂lnθ (= ∂lnH/∂lnθ, since H ∝ C)
    # dlnH/dlnκ_i = 1 for a line on stage i, else 0
    for j, k in enumerate(kinds):
        if k in STAGE_KINDS:
            J[j, len(_SENSING) + STAGE_KINDS.index(k)] = 1.0
    return LineSet(freqs, kinds, floor_asd(loop, freqs), J)


def fisher(ls: LineSet, amps: np.ndarray, T: float) -> np.ndarray:
    """Fisher information Γ on the log-TDCF vector for line amplitudes ``amps`` (DARM-displacement,
    m rms) over time ``T``: ``Γ = Σ_j 2·SNR_j²·Re[conj(J_j)⊗J_j]``, ``SNR_j = amp_j·√T/floor_j``.

    Actuator lines enter through the ruler ``H_stage/H_pcal = κ_i·D_i·N_i`` (``∂lnH/∂lnκ_i = 1`` —
    a frequency-independent shape), but the SNR carries ``floor(f)``, so with the **real
    frequency-dependent** DARM floor an actuator line's κ_i precision DOES depend on its frequency
    (a line on the low-frequency wall is penalised). Every line frequency is therefore optimised."""
    snr2 = (np.asarray(amps, float) ** 2) * T / ls.floor ** 2
    Gamma = np.zeros((len(TDCF_PARAMS), len(TDCF_PARAMS)))
    for j in range(len(ls.kinds)):
        Gamma += 2.0 * snr2[j] * np.real(np.outer(np.conj(ls.J[j]), ls.J[j]))
    return Gamma


def _crb_cov(Gamma: np.ndarray, rel_floor: float = 1e-12) -> np.ndarray:
    """CRB covariance ``Γ⁻¹`` via a symmetric eigen-regularised inverse: eigenvalues are floored at
    ``rel_floor·λ_max`` so a poorly-constrained (near-singular) direction gets a LARGE variance —
    not the zero that a plain pseudo-inverse would wrongly return. This makes an unmeasured
    parameter show up as a huge σ (and a huge time-to-target), which is the honest CRB."""
    G = np.nan_to_num(np.asarray(Gamma, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    w, V = np.linalg.eigh(0.5 * (G + G.T))
    w = np.maximum(w, rel_floor * max(float(w.max()), 0.0) + 1e-300)
    return (V / w) @ V.T


def sigma(ls: LineSet, amps: np.ndarray, T: float) -> dict[str, float]:
    """Fractional 1σ on each TDCF: ``√diag(Γ⁻¹)`` (CRB), keyed by parameter label."""
    s = np.sqrt(np.clip(np.diag(_crb_cov(fisher(ls, amps, T))), 0.0, np.inf))
    return {name: float(s[i]) for i, name in enumerate(TDCF_PARAMS)}


def tdcf_sigma(loop: DARMLoop, lines: list[Line], T: float) -> dict[str, float]:
    """Convenience: fractional CRB σ per TDCF for an explicit (amplitude-carrying) roster."""
    ls = build_lineset(loop, lines)
    return sigma(ls, np.array([ln.amp for ln in lines], float), T)


# ── frequency seeding (dispersion-lite: put each line where it is most informative) ──────
def seed_lines(loop: DARMLoop, *, fmin: float = 0.3, fmax: float = 1500.0,
               n_grid: int = 300) -> list[Line]:
    """A dispersion-seeded roster: one actuator line per stage placed in that stage's **authority
    band** — where it dominates the hierarchical drive (``|A_i|/Σ|A|``: M0 low, PUM mid, TST high,
    naturally distinct for demodulation) — and one Pcal line per sensing parameter where that
    parameter is most informative (``|∂lnC/∂lnθ|/floor``). Seven lines for seven TDCFs; the Pcal
    frequencies are refined and all amplitudes set by :func:`size_lines_for_target` (the actuator
    frequencies are fixed by the hierarchy — the ruler makes κ_i frequency-agnostic, see
    :func:`fisher`)."""
    grid = np.geomspace(fmin, fmax, n_grid)
    fl = floor_asd(loop, grid)
    mags = np.array([np.abs(loop.stage(st, grid)) for st in STAGE_KINDS])
    dominant = np.argmax(mags, axis=0)                       # which stage dominates at each bin
    lines: list[Line] = []
    for si, st in enumerate(STAGE_KINDS):                    # centre of this stage's authority band
        band = grid[(dominant == si) & (grid <= 100.0)]      # cap so the top stage isn't at the edge
        f_st = float(np.exp(np.mean(np.log(band)))) if band.size else float(grid[np.argmax(mags[si])])
        lines.append(Line(f_st, st, target=f"kappa_{st}"))
    C0 = _sensing_C(loop, grid, {})
    for name, knob in _SENSING:                              # Pcal line per sensing param
        v = float(getattr(loop, knob)); h = 1e-6 * (abs(v) if v != 0 else 1.0)
        dlnC = v * (_sensing_C(loop, grid, {knob: v + h}) - _sensing_C(loop, grid, {knob: v - h})) \
            / (2.0 * h) / C0
        info = np.abs(dlnC) / fl
        lines.append(Line(float(grid[int(np.argmax(info))]), "PCAL", target=name))
    return lines


# ── size the drive so every TDCF hits the target precision in the target time ────────────
def _allocate(ls: LineSet, logw: np.ndarray, A_tot: float) -> np.ndarray:
    """Relative weights ``exp(logw)`` → amplitudes with ‖amp‖₂ = A_tot."""
    w = np.exp(logw - logw.max())
    return A_tot * w / np.sqrt(np.sum(w ** 2))


def size_lines_for_target(loop: DARMLoop, *, A_tot: float = 1.0e-3, target: float = 1e-3,
                          T_max: float = 300.0, T_ref: float = 60.0,
                          lines: list[Line] | None = None, optimize_freq: bool = True,
                          fmin: float = 0.3, fmax: float = 1500.0, seed: int = 0) -> dict:
    """Place + size cal lines so EVERY TDCF reaches ``target`` (0.1%) fractional 1σ, and report
    whether that is met within ``T_max`` (5 min) at total drive ``A_tot``.

    Minimises the **worst-parameter** fractional σ (c-optimal: all params must hit the target, so
    the binding one governs) at fixed total drive ``‖amp‖₂ = A_tot`` over the per-line amplitudes
    and — when ``optimize_freq`` — the **Pcal** (sensing) line frequencies, via differential
    evolution. The three **actuator** frequencies stay at their hierarchy bands from
    :func:`seed_lines` (the ruler makes κ_i frequency-agnostic — see :func:`fisher`). The Fisher
    needs only the analytic ``C``/floor, so this is cheap. Because the CRB scales as ``1/T``, per-
    parameter time to
    reach ``target`` is ``T_req,θ = T_ref·(σ_θ(T_ref)/target)²``; the binding requirement is
    ``max_θ T_req``.

    Returns a dict: ``lines`` (sized), ``sigma`` (per-TDCF at ``T_ref``), ``t_req`` (per-TDCF, s),
    ``t_req_max``, ``feasible`` (``≤ T_max``), ``binding`` (worst TDCF), and ``lineset``.
    """
    if lines is None:
        lines = seed_lines(loop)
    kinds = [ln.kind for ln in lines]
    targets = [ln.target for ln in lines]
    f_seed = np.array([ln.freq for ln in lines], dtype=float)
    n = len(lines)
    pcal = list(range(n))   # optimise EVERY line frequency (unconstrained; the real
    npc = len(pcal)             # frequency-dependent floor makes actuator placement matter)
    lf_lo, lf_hi = np.log10(fmin), np.log10(fmax)
    min_sep = np.log10(1.15)                                # lines ≥15% apart (demodulable)

    def freqs_of(x):
        f = f_seed.copy()                                  # all line frequencies optimised
        if optimize_freq:
            f[pcal] = 10.0 ** np.clip(x[n:], lf_lo, lf_hi)
        return f

    def worst(x):
        freqs = freqs_of(x)
        try:
            ls = build_lineset(loop, [Line(float(f), k) for f, k in zip(freqs, kinds)])
            w = max(sigma(ls, _allocate(ls, x[:n], A_tot), T_ref).values())
        except Exception:
            return 1e30                                    # pathological config → penalise
        if not np.isfinite(w):
            return 1e30
        if optimize_freq:                                  # keep lines apart (demodulation)
            lf = np.log10(freqs); d = np.abs(lf[:, None] - lf[None, :]); d[np.diag_indices(n)] = np.inf
            gap = d.min()
            if gap < min_sep:
                w *= 1.0 + 50.0 * (min_sep - gap) / min_sep
        return w

    if optimize_freq:
        x0 = np.concatenate([np.zeros(n), np.log10(f_seed[pcal])])
        bounds = [(-6, 6)] * n + [(lf_lo, lf_hi)] * npc
        x = differential_evolution(worst, bounds, seed=seed, maxiter=200, tol=1e-8,
                                    polish=True, init="sobol", x0=x0).x
    else:
        x = minimize(worst, np.zeros(n), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-6, maxiter=6000)).x
    freqs = freqs_of(x)
    ls0 = build_lineset(loop, [Line(float(f), k) for f, k in zip(freqs, kinds)])
    amps0 = _allocate(ls0, x[:n], A_tot)
    order = sorted(range(n), key=lambda i: freqs[i])
    sized = [Line(float(freqs[i]), kinds[i], float(amps0[i]), targets[i]) for i in order]
    # rebuild the lineset in the SAME (frequency-sorted) order as ``sized`` so the returned
    # ``lineset`` and ``lines`` amplitudes line up for downstream use (e.g. response_budget)
    ls = build_lineset(loop, sized)
    sig = sigma(ls, np.array([ln.amp for ln in sized], float), T_ref)
    t_req = {k: float(T_ref * (s / target) ** 2) for k, s in sig.items()}
    binding = max(t_req, key=t_req.get)
    return dict(lines=sized, sigma=sig, t_req=t_req, t_req_max=t_req[binding],
                feasible=bool(t_req[binding] <= T_max), binding=binding, T_ref=T_ref,
                target=target, A_tot=A_tot, lineset=ls)


def sigma_vs_time(loop: DARMLoop, lines: list[Line], times: np.ndarray) -> dict[str, np.ndarray]:
    """Per-TDCF fractional σ(t) over ``times`` (CRB ∝ 1/√T from one evaluation)."""
    times = np.asarray(times, dtype=float)
    ls = build_lineset(loop, lines)
    amps = np.array([ln.amp for ln in lines], float)
    s0 = sigma(ls, amps, float(times[0]))
    return {k: v * np.sqrt(times[0] / times) for k, v in s0.items()}


# ── propagate the TDCF CRB into the response-function systematic budget δR/R(f) ──────────
def response_log_jacobian(loop: DARMLoop, freqs: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """``∂lnR/∂lnθ_a(f)`` for the seven TDCFs — how a fractional error in each parameter propagates
    into the detector response ``R = (1+G)/C`` (shape ``(F, 7)`` complex).

    Following Sun 2020: the digital servo ``D`` is FIXED (it is a known filter), so
    ``R(θ) = (1 + A(θ)·D·C(θ))/C(θ)`` with ``D = D_nominal`` — a κ_i error moves ``A`` (hence ``G``,
    hence ``R``), and a sensing error moves ``C``. (This differs from the twin's *derived*-``D``
    invariant, under which ``R`` is insensitive to κ; here ``D`` is held fixed to match how the real
    calibration pipeline propagates errors.)"""
    freqs = np.asarray(freqs, dtype=float)
    D_fixed = loop.D(freqs)                                 # nominal digital servo, held fixed

    def lnR(lp):
        A, C = lp.A(freqs), lp.C(freqs)
        return np.log((1.0 + A * D_fixed * C) / C)

    J = np.zeros((len(freqs), len(TDCF_PARAMS)), dtype=complex)
    knobs = [knob for _, knob in _SENSING] + [f"kappa_{s}" for s in STAGE_KINDS]
    for a, knob in enumerate(knobs):
        v = (float(loop.stages[knob[6:]][1]) if knob.startswith("kappa_")
             else float(getattr(loop, knob)))
        h = step * (abs(v) if v != 0 else 1.0)
        J[:, a] = (lnR(loop.with_params(**{knob: v + h}))
                   - lnR(loop.with_params(**{knob: v - h}))) / (2.0 * h / v)   # ∂lnR/∂lnθ
    return J


def response_budget(loop: DARMLoop, ls: LineSet, amps: np.ndarray, T: float,
                    freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Statistical 1σ on the detector response from the cal-line CRB: propagate the full TDCF
    covariance ``Σ = Γ⁻¹`` through ``∂lnR/∂lnθ``. Returns ``(sigma_mag_pct, sigma_phase_deg)`` over
    ``freqs`` — the magnitude [%] and phase [deg] response uncertainty the lines deliver at time
    ``T`` (to lay against the O3/O4 budgets)."""
    cov = _crb_cov(fisher(ls, amps, T))                 # TDCF covariance (fractional/log params)
    J = response_log_jacobian(loop, freqs)                  # (F, 7) complex
    reJ, imJ = J.real, J.imag
    var_mag = np.einsum("fa,ab,fb->f", reJ, cov, reJ)       # Var(Re δlnR) = |δR/R| magnitude
    var_phase = np.einsum("fa,ab,fb->f", imJ, cov, imJ)     # Var(Im δlnR) = phase [rad]
    return (100.0 * np.sqrt(np.clip(var_mag, 0, np.inf)),
            np.degrees(np.sqrt(np.clip(var_phase, 0, np.inf))))


#: Published Advanced-LIGO O3A response-error budget (Sun 2020, 68% CI, 20–2000 Hz): the total
#: systematic-error+uncertainty upper limit and the systematic-error-alone level. O4's final budget
#: is deferred to a forthcoming publication (arXiv:2508.08423), so we ground everything in O3.
O3_BUDGET = {"total_mag_pct": 7.0, "total_phase_deg": 4.0,
             "syst_mag_pct": 2.0, "syst_phase_deg": 2.0}

#: RANDOM (statistical) response-uncertainty target levels for the measurement-design study, keyed
#: by name → (magnitude %, phase °). The O3 random level is the uncertainty part of the O3 total,
#: in quadrature with the systematic (√(7²−2²) ≈ 6.7%, √(4²−2²) ≈ 3.46°). "O4-class" uses the O3
#: systematic floor (2%/2°) as a provisional tighter aspiration (final O4 budget forthcoming).
#: "0.1%" is the stretch target. These are the iso-precision contours of the Pareto design plane.
TARGET_LEVELS = {
    "O3 random": (float(np.sqrt(7.0**2 - 2.0**2)), float(np.sqrt(4.0**2 - 2.0**2))),
    "O4-class (prov.)": (2.0, 2.0),
    "0.1% stretch": (0.1, 0.1),
}


def rho_of_target(mag_pct: float, phase_deg: float) -> float:
    """Combined response-error level ``ρ = √((mag fraction)² + (phase rad)²)`` — one scalar that
    folds a (magnitude %, phase °) budget into a single random-error target."""
    return float(np.hypot(mag_pct / 100.0, np.radians(phase_deg)))


def band_response_rho(loop: DARMLoop, ls: LineSet, amps: np.ndarray, T: float,
                      fband=(20.0, 2000.0), n: int = 200, J_R: np.ndarray | None = None) -> float:
    """Band-max combined response error ``ρ(f) = √(Var(Re δlnR) + Var(Im δlnR))`` at time ``T``
    (magnitude-fraction and phase-radian uncertainty in quadrature) — the single number the
    measurement-design study drives down. ``J_R`` (the response log-Jacobian on the band grid) may
    be supplied to avoid recomputing it in an optimiser loop."""
    f = np.geomspace(fband[0], fband[1], n)
    if J_R is None:
        J_R = response_log_jacobian(loop, f)
    cov = _crb_cov(fisher(ls, amps, T))
    var = (np.einsum("fa,ab,fb->f", J_R.real, cov, J_R.real)
           + np.einsum("fa,ab,fb->f", J_R.imag, cov, J_R.imag))
    return float(np.sqrt(np.max(var)))


def pareto_cost(loop: DARMLoop, ls: LineSet, amps: np.ndarray, rho_target: float,
                T_ref: float = 60.0) -> float:
    """Measurement cost ``K = amplitude²·time`` needed to bring the band-max response error to
    ``rho_target``. Response σ scales as ``1/√(A²·T)``, so ``K`` is scheme-characteristic and the
    iso-precision contour is ``A²·T = K`` (``A(T) = √(K/T)``). Smaller ``K`` = gentler+faster."""
    A2 = float(np.sum(np.asarray(amps, float) ** 2))            # ‖amp‖² = A_tot²
    return A2 * T_ref * (band_response_rho(loop, ls, amps, T_ref) / rho_target) ** 2


def size_lines_for_response(loop: DARMLoop, *, A_tot: float = 1.0, T_ref: float = 60.0,
                            fmin: float = 0.3, fmax: float = 1500.0, seed: int = 0,
                            lines: list[Line] | None = None) -> dict:
    """**Response-optimal** design: place the Pcal line frequencies + split the drive to MINIMISE the
    band-max response error ρ (the quantity the O3/O4 budget is quoted in), rather than the worst
    single parameter. This is the scheme for the gentle/fast study — it reaches a target ``δR/R``
    with the least injected energy ``A²·T``. Actuator frequencies stay in their hierarchy bands
    (:func:`seed_lines`). Returns ``dict(lines, lineset, rho, cost(rho_target))``."""
    if lines is None:
        lines = seed_lines(loop)
    kinds = [ln.kind for ln in lines]
    targets = [ln.target for ln in lines]
    f_seed = np.array([ln.freq for ln in lines], float)
    n = len(lines)
    pcal = list(range(n))   # optimise EVERY line frequency (unconstrained)
    lf_lo, lf_hi = np.log10(fmin), np.log10(fmax)
    min_sep = np.log10(1.15)
    J_R = response_log_jacobian(loop, np.geomspace(20.0, 2000.0, 200))   # precompute once

    def freqs_of(x):
        f = f_seed.copy(); f[pcal] = 10.0 ** np.clip(x[n:], lf_lo, lf_hi); return f

    def obj(x):
        freqs = freqs_of(x)
        try:
            ls = build_lineset(loop, [Line(float(f), k) for f, k in zip(freqs, kinds)])
            rho = band_response_rho(loop, ls, _allocate(ls, x[:n], A_tot), T_ref, J_R=J_R)
        except Exception:
            return 1e30                                    # pathological config → penalise
        if not np.isfinite(rho):
            return 1e30
        lf = np.log10(freqs); d = np.abs(lf[:, None] - lf[None, :]); d[np.diag_indices(n)] = np.inf
        return rho * (1.0 + 50.0 * max(0.0, min_sep - d.min()) / min_sep)

    x0 = np.concatenate([np.zeros(n), np.log10(f_seed[pcal])])
    bounds = [(-6, 6)] * n + [(lf_lo, lf_hi)] * len(pcal)
    x = differential_evolution(obj, bounds, seed=seed, maxiter=200, tol=1e-9, polish=True,
                               init="sobol", x0=x0).x
    freqs = freqs_of(x)
    order = sorted(range(n), key=lambda i: freqs[i])
    sized = [Line(float(freqs[i]), kinds[i], 0.0, targets[i]) for i in order]
    ls = build_lineset(loop, sized)
    amps = _allocate(ls, x[:n][order], A_tot)
    for ln, a in zip(sized, amps):
        ln.amp = float(a)
    rho = band_response_rho(loop, ls, amps, T_ref, J_R=J_R)
    return dict(lines=sized, lineset=ls, amps=amps, rho=rho, A_tot=A_tot, T_ref=T_ref)


def naive_broadband(loop: DARMLoop, *, n_pcal: int = 12, A_tot: float = 1.0,
                    fmin: float = 10.0, fmax: float = 2000.0) -> tuple[LineSet, np.ndarray]:
    """A 'do-nothing-smart' baseline: ``n_pcal`` Pcal lines spread log-uniformly across the band
    plus one actuator line per stage (hierarchy bands), ALL at equal amplitude (no Fisher
    placement, no allocation). Returns ``(lineset, amps)`` with ‖amp‖₂ = A_tot."""
    pcal = list(np.geomspace(fmin, fmax, n_pcal))
    act = [ln for ln in seed_lines(loop) if ln.kind in STAGE_KINDS]
    lines = [Line(f, "PCAL") for f in pcal] + [Line(ln.freq, ln.kind) for ln in act]
    ls = build_lineset(loop, lines)
    amps = np.full(len(lines), A_tot / np.sqrt(len(lines)))
    return ls, amps


def reference_scheme(loop: DARMLoop, positions, *, A_tot: float = 1.0e-3, T_ref: float = 60.0,
                     target: float = 1e-3) -> dict:
    """Size a fixed set of line ``positions`` (e.g. :data:`O3_LINES` / :data:`O4_LINES`) at the
    same total drive and optimised allocation, for the rough head-to-head. Same return as
    :func:`size_lines_for_target`."""
    lines = [Line(f, kind, target=tgt) for f, kind, tgt in positions
             if kind in ("PCAL",) + STAGE_KINDS]
    # fixed frequencies (the real LIGO positions) — only the amplitude allocation is optimised
    return size_lines_for_target(loop, A_tot=A_tot, target=target, T_ref=T_ref, lines=lines,
                                 optimize_freq=False)
