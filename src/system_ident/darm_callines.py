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
loop. The CRB is ``Γ⁻¹`` (:func:`fisher.safe_inverse`); ``σ_θ = √diag`` is the fractional 1σ. Same
math as :mod:`fisher` (the ``2·Re[∂H*·∂H]·weight`` structure), specialised to the DARM parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .darm import DARMLoop, sensing_model_detuned
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


def default_cal_loop(delta_deg: float = 5.0, *, disturbance_asd: float = 3.0e-4,
                     sensor_asd: float = 300.0) -> DARMLoop:
    """The base twin for cal-line design: the M0-damped hierarchical reduced-quad loop at a
    representative slight SRC detuning (so every TDCF, including δ, is nonzero and fractional
    precision is well defined), with the representative DARM noise floor set."""
    loop = DARMLoop.default_reduced(fmin=0.3, hierarchical=True).with_params(
        delta=np.radians(delta_deg))
    loop.disturbance_asd, loop.sensor_asd = disturbance_asd, sensor_asd
    return loop


def floor_asd(loop: DARMLoop, freq) -> np.ndarray:
    """DARM-displacement noise floor √(disturbance² + (readout/|C|)²) [m/√Hz]."""
    f = np.asarray(freq, dtype=float)
    return np.sqrt(loop.disturbance_asd ** 2 + (loop.sensor_asd / np.abs(loop.C(f))) ** 2)


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
    for si, st in enumerate(STAGE_KINDS):
        col = len(_SENSING) + si
        for j, k in enumerate(kinds):
            if k == st:
                J[j, col] = 1.0
    return LineSet(freqs, kinds, floor_asd(loop, freqs), J)


def fisher(ls: LineSet, amps: np.ndarray, T: float) -> np.ndarray:
    """Fisher information Γ on the log-TDCF vector for line amplitudes ``amps`` over time ``T``.
    ``Γ = Σ_j 2·SNR_j²·Re[conj(J_j)⊗J_j]``, ``SNR_j = amp_j·√T/floor_j``."""
    snr2 = (np.asarray(amps, float) ** 2) * T / ls.floor ** 2
    Gamma = np.zeros((len(TDCF_PARAMS), len(TDCF_PARAMS)))
    for j in range(len(ls.kinds)):
        Gamma += 2.0 * snr2[j] * np.real(np.outer(np.conj(ls.J[j]), ls.J[j]))
    return Gamma


def sigma(ls: LineSet, amps: np.ndarray, T: float) -> dict[str, float]:
    """Fractional 1σ on each TDCF: ``√diag(Γ⁻¹)`` (CRB), keyed by parameter label."""
    crb = safe_inverse(fisher(ls, amps, T))
    s = np.sqrt(np.clip(np.diag(crb), 0.0, np.inf))
    return {name: float(s[i]) for i, name in enumerate(TDCF_PARAMS)}


def tdcf_sigma(loop: DARMLoop, lines: list[Line], T: float) -> dict[str, float]:
    """Convenience: fractional CRB σ per TDCF for an explicit (amplitude-carrying) roster."""
    ls = build_lineset(loop, lines)
    return sigma(ls, np.array([ln.amp for ln in lines], float), T)


# ── frequency seeding (dispersion-lite: put each line where it is most informative) ──────
def seed_lines(loop: DARMLoop, *, fmin: float = 0.3, fmax: float = 1500.0,
               n_grid: int = 300) -> list[Line]:
    """A dispersion-seeded roster: one actuator line per stage placed where that stage *dominates*
    the actuation (``|stage_i|/Σ|stage|`` — M0 low, PUM mid, TST high, so the lines are spread and
    each cleanly measures its own κ_i), and one Pcal line per sensing parameter at the frequency
    where that parameter is most informative (``|∂lnC/∂lnθ|/floor``). Seven lines for seven TDCFs;
    amplitudes are set by :func:`size_lines_for_target`. (κ_i information is frequency-independent
    — ``∂lnH/∂lnκ_i = 1`` — so a stage line's frequency is chosen for realism/diversity; the Pcal
    line placement is where it matters.)"""
    grid = np.geomspace(fmin, fmax, n_grid)
    fl = floor_asd(loop, grid)
    mags = {st: np.abs(loop.stage(st, grid)) for st in STAGE_KINDS}
    total = sum(mags.values())
    lines: list[Line] = []
    for st in STAGE_KINDS:                                   # where this stage dominates the drive
        lines.append(Line(float(grid[int(np.argmax(mags[st] / total))]), st, target=f"kappa_{st}"))
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
    the binding one governs) at fixed total drive ``‖amp‖₂ = A_tot`` over both the per-line
    amplitudes and — when ``optimize_freq`` — the line frequencies (the line *kinds* are fixed:
    three actuator lines + four Pcal lines from :func:`seed_lines`, seeded then refined by
    differential evolution). The Fisher needs only the analytic ``C``/floor, so this is cheap
    despite re-placing every line. Because the CRB scales as ``1/T``, the per-parameter time to
    reach ``target`` is ``T_req,θ = T_ref·(σ_θ(T_ref)/target)²``; the binding requirement is
    ``max_θ T_req``.

    Returns a dict: ``lines`` (sized), ``sigma`` (per-TDCF at ``T_ref``), ``t_req`` (per-TDCF, s),
    ``t_req_max``, ``feasible`` (``≤ T_max``), ``binding`` (worst TDCF), and ``lineset``.
    """
    if lines is None:
        lines = seed_lines(loop)
    kinds = [ln.kind for ln in lines]
    n = len(lines)
    lf_lo, lf_hi = np.log10(fmin), np.log10(fmax)

    def unpack(x):
        logw = x[:n]
        logf = x[n:] if optimize_freq else np.log10([ln.freq for ln in lines])
        freqs = 10.0 ** np.clip(logf, lf_lo, lf_hi)
        return freqs, logw

    def worst(x):
        freqs, logw = unpack(x)
        ls = build_lineset(loop, [Line(float(f), k) for f, k in zip(freqs, kinds)])
        return max(sigma(ls, _allocate(ls, logw, A_tot), T_ref).values())

    x0 = np.concatenate([np.zeros(n), np.log10([ln.freq for ln in lines])])
    if optimize_freq:
        bounds = [(-6, 6)] * n + [(lf_lo, lf_hi)] * n
        sol = differential_evolution(worst, bounds, seed=seed, maxiter=120, tol=1e-7,
                                     polish=True, init="sobol", x0=x0)
        x = sol.x
    else:
        x = minimize(lambda w: worst(np.concatenate([w, x0[n:]])), np.zeros(n),
                     method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=6000)).x
        x = np.concatenate([x, x0[n:]])
    freqs, logw = unpack(x)
    ls = build_lineset(loop, [Line(float(f), k) for f, k in zip(freqs, kinds)])
    amps = _allocate(ls, logw, A_tot)
    order = sorted(range(n), key=lambda i: freqs[i])
    sized = [Line(float(freqs[i]), kinds[i], float(amps[i]),
                  lines[i].target if not optimize_freq else _nearest_target(kinds[i], i, lines))
             for i in order]
    sig = sigma(ls, amps, T_ref)
    t_req = {k: float(T_ref * (s / target) ** 2) for k, s in sig.items()}
    binding = max(t_req, key=t_req.get)
    return dict(lines=sized, sigma=sig, t_req=t_req, t_req_max=t_req[binding],
                feasible=bool(t_req[binding] <= T_max), binding=binding, T_ref=T_ref,
                target=target, A_tot=A_tot, lineset=ls)


def _nearest_target(kind: str, idx: int, seed: list[Line]) -> str:
    """Carry the seed's target label onto a re-placed line of the same kind (best-effort)."""
    same = [ln for ln in seed if ln.kind == kind]
    return same[0].target if same else ""


def sigma_vs_time(loop: DARMLoop, lines: list[Line], times: np.ndarray) -> dict[str, np.ndarray]:
    """Per-TDCF fractional σ(t) over ``times`` (CRB ∝ 1/√T from one evaluation)."""
    times = np.asarray(times, dtype=float)
    ls = build_lineset(loop, lines)
    amps = np.array([ln.amp for ln in lines], float)
    s0 = sigma(ls, amps, float(times[0]))
    return {k: v * np.sqrt(times[0] / times) for k, v in s0.items()}


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
