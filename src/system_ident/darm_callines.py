"""DARM calibration-line hardware constants: the Pcal free-mass actuator range.

The calibration group tracks the time-dependent correction factors (TDCFs) with always-on
calibration lines — a Pcal displacement reference plus one actuator line per suspension stage.

This module retains only the pieces with genuine physical provenance:

* :func:`pcal_range_disp` — the photon-calibrator's maximum DARM-displacement amplitude, fixed by
  the real ±200 mW peak-to-peak power modulation on the 40 kg free test mass (radiation-pressure
  force on a quasi-free mass, so the displacement rolls off as 1/f²).
* :data:`O3_LINES` / :data:`O4_LINES` — the published O3/O4 calibration-line *frequencies*
  (O3: Sun et al. 2020, DCC P1900245; O4: arXiv:2508.08423).

.. note::
   A previous version of this module carried a Fisher-optimal cal-line *sizing/design* engine
   (per-stage actuator "authority", ``size_lines_for_target`` / ``size_lines_for_response``,
   ``reference_scheme``, the Pareto cost, response-budget propagation, and the O3/O4 head-to-head).
   Those absolute results rested on **fabricated** per-stage actuator ranges (an invented
   ``_STAGE_RANGE_M``): without the real suspension-stage ranges the actuation-side sizing and the
   cross-scheme comparisons cannot be computed, so that machinery — and every conclusion built on it
   — was removed rather than caveated. Only the Pcal range (real hardware) and the published line
   frequencies survive here.
"""
from __future__ import annotations

import numpy as np

from . import provenance as _prov

#: The eight O4 Pcal monitoring-line frequencies [Hz], read verbatim from the legend of Fig. 4 of
#: Wade et al. 2025 (arXiv:2508.08423) — the frequencies at which the LIGO calibration monitoring
#: (CalMonitor) reports the systematic error η_R. These are real published line positions.
O4_PCAL_LINES_HZ = (17.1, 33.43, 53.67, 77.73, 102.13, 284.01, 410.3, 1083.7)

#: Illustrative reference rosters used only for the head-to-head comparison — a Pcal line tagged
#: with the TDCF it most informs, plus one actuator line per hierarchical stage. The Pcal
#: frequencies are the published ones (O4 from :data:`O4_PCAL_LINES_HZ`; O3 the ~332 Hz sensing /
#: ~7.9 Hz detuning / ~1083 Hz high-f lines from Sun 2020 §4.1). The per-stage actuator-line
#: frequencies are REPRESENTATIVE placeholders (the papers do not tabulate per-stage line positions
#: or amplitudes); the optimal positions are what :func:`design_lines` computes.
O3_LINES = [(7.9, "PCAL", "delta"), (17.1, "PCAL", "kappa_C"), (331.9, "PCAL", "f_cc"),
            (1083.7, "PCAL", "tau"), (15.6, "M0", "kappa_M0"), (16.4, "PUM", "kappa_PUM"),
            (35.9, "TST", "kappa_TST")]
O4_LINES = [(17.1, "PCAL", "kappa_C"), (33.43, "PCAL", "kappa_C"), (102.13, "PCAL", "delta"),
            (410.3, "PCAL", "f_cc"), (1083.7, "PCAL", "tau"), (15.6, "M0", "kappa_M0"),
            (16.4, "PUM", "kappa_PUM"), (35.9, "TST", "kappa_TST")]


# ── Pcal actuator range: the DARM displacement the photon calibrator can make per frequency ──────
#: Photon-calibrator hardware range: it modulates laser power (radiation-pressure force F = 2P/c) on
#: a quasi-free test mass, so its displacement rolls off as 1/f². The real Pcal has ~200 mW
#: peak-to-peak power range (given by the author, real hardware); with the 40 kg test mass this fixes
#: the ABSOLUTE Pcal line amplitude at every frequency — no free parameter.
PCAL_POWER_PP_W = _prov.record(
    "pcal_power_pp_w", 0.200, _prov.USER,
    "rana 2026-08 — Pcal hardware peak-to-peak power modulation range", unit="W")
TEST_MASS_KG = _prov.record(
    "test_mass_kg", 40.0, _prov.PAPER, "aLIGO quad test mass", unit="kg")
_C_LIGHT = _prov.record(
    "c_light", 299_792_458.0, _prov.CONSTANT, "speed of light", unit="m/s")


def pcal_range_disp(freq) -> np.ndarray:
    """Maximum Pcal DARM-displacement amplitude [m rms] at ``freq`` from the full ±200 mW power
    range: radiation-pressure force ``F_rms = (P_pp/c)/√2`` (``2·(P_pp/2)/c``, converted to rms) on
    the free test mass, ``x = F/(M(2πf)²)`` — the 1/f² free-mass actuator range."""
    f = np.asarray(freq, dtype=float)
    F_rms = PCAL_POWER_PP_W / _C_LIGHT / np.sqrt(2.0)
    return F_rms / (TEST_MASS_KG * (2.0 * np.pi * f) ** 2)


# ── Grounded drift priors + cross-check anchors from the O3/O4 calibration papers ────────────────
#: Representative fractional 1σ drift of the sensing/optical gain κ_C over an observing run. Sun et
#: al. 2020 (arXiv:2005.02531, §4.2): "the measured fractional variation of κ_C is typically at the
#: level of 1%–2%, and can be as large as ∼10%." 0.02 is the representative (upper-typical) 1σ.
KAPPA_C_DRIFT_FRAC = _prov.record(
    "kappa_C_drift_frac", 0.02, _prov.PAPER,
    "Sun et al. 2020 (arXiv:2005.02531) §4.2: κ_C fractional variation 'typically 1%–2%'",
    note="representative 1σ = upper end of the stated 1–2% typical range")

#: Cross-check anchor (not a design input): the 17.1 Hz Pcal monitoring line stands at ~4e-19
#: strain/√Hz in the O4 Hanford ASD (Wade et al. 2025 Fig. 2, read off the plot). × the 4 km arm ⇒
#: ~1.6e-15 m/√Hz DARM displacement — a sanity check on the Pcal displacement predicted from the
#: ±200 mW budget, see :func:`pcal_budget_crosscheck`.
PCAL_LINE_17HZ_STRAIN = _prov.record(
    "pcal_line_17hz_strain", 4.0e-19, _prov.PAPER,
    "Wade et al. 2025 (arXiv:2508.08423) Fig. 2: 17.1 Hz Pcal line ASD peak ≈4e-19 strain/√Hz",
    unit="1/√Hz", note="read from figure; order-of-magnitude cross-check only")


# ── Joint TDCF parameter vector ──────────────────────────────────────────────────────────────────
#: The joint DARM+suspension parameter vector θ tracked by the cal lines. Sensing: κ_C (via the
#: loop's ``g_c``), the coupled-cavity pole ``f_cc``, the SRC detuning ``delta`` (δ; the coupled
#: model's spring/pole-split knob, replacing the factorised f_s/Q), and the residual delay ``tau``.
#: Actuation: the hierarchical per-stage strengths ``kappa_M0/PUM/TST``. Every one is a field
#: ``DARMLoop.with_params`` can perturb, so the Jacobian is a finite difference on the real twin.
PARAM_NAMES = ("g_c", "f_cc", "delta", "tau", "kappa_M0", "kappa_PUM", "kappa_TST")

# Finite-difference steps per parameter: (step, is_fractional). Fractional for scale knobs (g_c,
# kappa_i), absolute for the shape knobs (f_cc [Hz], delta [rad], tau [s]) — the latter can be zero
# at nominal so an absolute step is required.
_FD_STEP = {"g_c": (1e-4, True), "f_cc": (1e-3, True), "delta": (1e-4, False),
            "tau": (1e-3, True), "kappa_M0": (1e-4, True), "kappa_PUM": (1e-4, True),
            "kappa_TST": (1e-4, True)}


def _observable(loop, actuator: str, freq):
    """Complex closed-loop observable at a cal line: the Pcal FRF ``C/(1+G)`` (``actuator='PCAL'``)
    or a stage FRF ``C·κ_i·N_i/(1+G)`` (``actuator`` a stage name). These are exactly the model
    FRFs :func:`system_ident.darm_tv.joint_snapshot` fits — so a stage line depends on the sensing
    params (through ``C`` and ``1+G``) as well as its own κ, which is the joint coupling."""
    f = np.atleast_1d(np.asarray(freq, dtype=float))
    return loop.frf_pcal(f) if actuator == "PCAL" else loop.frf_stage(actuator, f)


def _nominal(loop, names) -> np.ndarray:
    return np.array([loop.stages[n[len("kappa_"):]][1] if n.startswith("kappa_")
                     else getattr(loop, n) for n in names], dtype=float)


def _log_jacobian(loop, actuator: str, freq, names) -> np.ndarray:
    """``∂ ln H / ∂θ_k`` at ``freq`` for one line, via central finite difference on ``with_params``.
    Returns a complex array shape ``(n_par, n_freq)``. The log-derivative makes the Fisher weight a
    dimensionless line SNR² and cancels the (unit-bearing) overall FRF scale."""
    f = np.atleast_1d(np.asarray(freq, dtype=float))
    H0 = _observable(loop, actuator, f)
    nom = _nominal(loop, names)
    J = np.empty((len(names), len(f)), dtype=complex)
    for k, name in enumerate(names):
        step, frac = _FD_STEP[name]
        d = step * abs(nom[k]) if frac else step
        if d == 0.0:
            d = step or 1e-6
        Hp = _observable(loop.with_params(**{name: nom[k] + d}), actuator, f)
        Hm = _observable(loop.with_params(**{name: nom[k] - d}), actuator, f)
        J[k] = (Hp - Hm) / (2.0 * d) / H0
    return J


def joint_fisher(loop, lines, T: float, *, names=PARAM_NAMES, floor_fn=None):
    """Coupled Fisher information / CRB for the joint TDCF vector from a set of cal lines.

    Parameters
    ----------
    loop : DARMLoop
        The twin, at the operating point where the CRB is evaluated. Its ``noise_asd`` (the real
        DARM displacement floor) sets the readout noise unless ``floor_fn`` overrides it.
    lines : list of ``(freq_hz, actuator, disp_amp_m)``
        Each cal line: its frequency, the injection port (``'PCAL'`` or a stage name), and the DARM
        **displacement amplitude** [m rms] the line produces at that frequency (see
        :func:`line_displacement`). This amplitude is where the drive budget / force cap enters.
    T : float
        Integration time [s]. Fisher ∝ T, so every σ ∝ 1/√T.

    Returns
    -------
    (gamma, cov, corr, names) : the ``(n,n)`` Fisher matrix, its CRB ``cov = safe_inverse(gamma)``
    (absolute-θ units), the parameter correlation matrix, and the parameter order.

    The line's contribution is ``2·Re[∂lnH*_a ∂lnH_b] · ρ²`` with the line SNR² ``ρ² =
    disp_amp²·T/n(f)²`` — the standard Fisher-from-coherent-line form (mirrors
    :func:`system_ident.fisher.fisher_matrix`, with a physical-parameter log-Jacobian). Because a
    stage line's ``∂lnH`` has non-zero sensing entries, the off-diagonal sensing↔actuation blocks
    are populated: this is the joint coupling, not two separate estimates.
    """
    floor = floor_fn if floor_fn is not None else loop.displacement_noise_asd
    n_par = len(names)
    gamma = np.zeros((n_par, n_par))
    for f_hz, actuator, disp in lines:
        J = _log_jacobian(loop, actuator, [f_hz], names)[:, 0]   # (n_par,) complex
        n_f = float(np.atleast_1d(floor([f_hz]))[0])
        rho2 = (float(disp) ** 2) * T / (n_f ** 2)               # line SNR²
        contrib = 2.0 * np.real(np.outer(np.conj(J), J)) * rho2
        if np.all(np.isfinite(contrib)):                          # skip a pathological line, don't poison Γ
            gamma += contrib
    from .fisher import safe_inverse
    cov = safe_inverse(gamma)
    s = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(np.outer(s, s) > 0, cov / np.outer(s, s), 0.0)
    return gamma, cov, corr, list(names)


# ── Drive budgets: the Pcal ±200 mW cap and the (derived) per-stage force caps ────────────────────
def stage_force_caps(loop, *, ref_freq_hz: float = 35.0, names=PARAM_NAMES) -> dict:
    """Per-stage drive-force caps [twin drive units], DERIVED, not from the papers.

    The O3/O4 papers do not tabulate the suspension-stage calibration-line amplitudes, so there is
    no paper number for the per-stage drive. We instead cap each stage so that, at ``ref_freq_hz``,
    its line makes the **same DARM displacement the grounded ±200 mW Pcal budget makes there**
    (a ruler-matched budget): ``F_cap_i = pcal_range_disp(f_ref) / |stage_i(f_ref)|``. Recorded
    ``DERIVED`` from the real Pcal range + the twin's own actuator TF — an explicit modelling choice,
    clearly not a measured per-stage drive.
    """
    stages = [n[len("kappa_"):] for n in names if n.startswith("kappa_")]
    x_ref = float(pcal_range_disp(ref_freq_hz))
    caps = {}
    for st in stages:
        tf = float(np.abs(loop.stage(st, [ref_freq_hz])[0]))
        caps[st] = _prov.record(
            f"stage_force_cap_{st}", x_ref / tf if tf > 0 else np.inf, _prov.DERIVED,
            f"ruler-matched to pcal_range_disp({ref_freq_hz:g} Hz)/|stage {st} TF|; papers give no "
            f"per-stage line amplitude", note="modelling choice, not a paper value")
    return caps


def line_displacement(loop, actuator: str, freq_hz: float, caps: dict | None = None,
                      pcal_weight: float = 1.0) -> float:
    """DARM displacement amplitude [m rms] a line produces — the ``disp_amp`` for :func:`joint_fisher`.

    Pcal line: ``pcal_weight · pcal_range_disp(f)`` (``pcal_weight`` is that line's share of the
    ±200 mW force budget, ≤1). Stage line: ``min(F_cap_i·|stage_i(f)|, pcal_range_disp(f))`` — the
    force-capped drive rolled through the actuator TF, but never driven past the Pcal ruler it is
    compared against. The force cap binds where the stage is weak (mid/high f); the ruler ceiling
    binds where the stage response explodes (low f) — without it a low-f top-mass line would carry
    ~1e10× the SNR of the Pcal lines and swamp the sensing information (a g_c↔κ degeneracy). Both
    the cap and the ceiling are grounded quantities. ``caps`` from :func:`stage_force_caps`."""
    if actuator == "PCAL":
        return float(pcal_weight * pcal_range_disp(freq_hz))
    if caps is None:
        caps = stage_force_caps(loop)
    force_disp = caps[actuator] * np.abs(loop.stage(actuator, [freq_hz])[0])
    return float(min(force_disp, pcal_range_disp(freq_hz)))       # ruler ceiling


def pcal_budget_crosscheck(loop) -> dict:
    """Order-of-magnitude check that the ±200 mW Pcal displacement is consistent with the 17.1 Hz
    line height read off Wade 2025 Fig. 2 (``PCAL_LINE_17HZ_STRAIN`` × 4 km arm). Returns both
    displacements and their ratio; they should agree to within ~1–2 orders of magnitude (the figure
    line is one line's share of the budget at a chosen SNR, not the full-budget maximum)."""
    from .darm import _ALIGO_ARM_LENGTH_M
    predicted = float(pcal_range_disp(17.1))                       # full-budget max at 17.1 Hz
    from_fig = PCAL_LINE_17HZ_STRAIN * _ALIGO_ARM_LENGTH_M         # displacement from the figure
    return {"pcal_range_disp_17hz_m": predicted, "o4_fig2_disp_m": from_fig,
            "ratio": predicted / from_fig}


# ── A-optimal line design under the force caps ───────────────────────────────────────────────────
def _prior_cov(prior_std, names) -> np.ndarray:
    """Diagonal prior covariance Σ_prior [absolute-θ² units] from a per-parameter 1σ dict. For the
    scale knobs (g_c, kappa_i) the dict value is a FRACTIONAL std (× nominal handled by the caller
    via ``nominal``); pass absolute std for f_cc/delta/tau."""
    return np.diag(np.array([prior_std[n] ** 2 for n in names], dtype=float))


def _scaled_fisher(gamma: np.ndarray, prior_cov: np.ndarray):
    """Return ``(Γ', std)`` with ``Γ'_ab = Γ_ab·σ_a·σ_b`` — the data Fisher in prior-σ units. In
    these units the Gaussian prior is the identity, so posterior/cost inverses are dimensionless and
    well-conditioned even though the raw θ span many orders of magnitude (g_c∼1e6 … τ∼1e-4)."""
    std = np.sqrt(np.diag(prior_cov))
    return gamma * np.outer(std, std), std


def _psd_eig(m: np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, clipped to ≥0. The scaled data Fisher is PSD by the Schur
    product theorem, but finite-difference/round-off can leave tiny negative eigenvalues; clipping
    keeps every downstream inverse/trace honest (no spurious negative variance)."""
    return np.clip(np.linalg.eigvalsh(0.5 * (m + m.T)), 0.0, None)


def posterior_cov(gamma: np.ndarray, prior_cov: np.ndarray) -> np.ndarray:
    """Bayesian posterior covariance ``Σ_post = (Γ_data + Σ_prior⁻¹)⁻¹`` [absolute-θ units],
    computed in prior-scaled units for conditioning. Always well-conditioned (the prior term is
    full-rank PD), so it is valid even where the data Fisher alone is rank-deficient."""
    Gs, std = _scaled_fisher(gamma, prior_cov)
    w, V = np.linalg.eigh(0.5 * (Gs + Gs.T))
    post_scaled = (V * (1.0 / (np.clip(w, 0.0, None) + 1.0))) @ V.T
    return post_scaled * np.outer(std, std)                       # back to absolute units


def a_optimal_cost(gamma: np.ndarray, prior_cov: np.ndarray) -> float:
    """Bayesian A-optimality objective ``tr(Σ_prior⁻¹·Σ_post) = tr((Γ'+I)⁻¹) = Σ_i 1/(λ_i+1)`` in
    prior-σ units (λ_i = eigenvalues of the scaled data Fisher Γ'). Bounded in ``(0, n_par]``: it
    equals ``n_par`` when the lines add nothing and → 0 as the lines fully determine θ. Smaller ⇒
    the design resolves the drifts (relative to their priors) better. Robustly PSD."""
    Gs, _ = _scaled_fisher(gamma, prior_cov)
    return float(np.sum(1.0 / (_psd_eig(Gs) + 1.0)))


def _single_line_info(loop, actuator, freq_hz, param_idx, caps=None, pcal_weight=1.0) -> float:
    """The (unnormalised) Fisher information a single line at ``freq_hz`` on ``actuator`` carries
    about parameter ``param_idx`` — ``|∂lnH/∂θ|²·disp²/floor²``. Used to seed line placement at each
    parameter's information peak (the physical-parameter analogue of the P&S dispersion function)."""
    J = _log_jacobian(loop, actuator, [freq_hz], PARAM_NAMES)[param_idx, 0]
    disp = line_displacement(loop, actuator, freq_hz, caps, pcal_weight)
    floor = float(loop.displacement_noise_asd([freq_hz])[0])
    return float((abs(J) ** 2) * (disp ** 2) / (floor ** 2))


def _seed_frequencies(loop, names, caps, band, n_pcal, n_scan=160):
    """Physics-aware seed: each stage line at the frequency where that stage best informs its own κ
    (its SNR peak in band), and ``n_pcal`` Pcal lines at the information peaks of the sensing
    parameters (g_c, f_cc, delta, tau — cycled if n_pcal differs). Beats a blind log-spread, which
    strands the steep top-mass (M0) line above its actuation band."""
    fs = np.geomspace(band[0], band[1], n_scan)
    stages = [n[len("kappa_"):] for n in names if n.startswith("kappa_")]
    sensing = [n for n in names if not n.startswith("kappa_")]
    stage_seed = [fs[int(np.argmax([_single_line_info(loop, st, f, names.index("kappa_" + st), caps)
                                    for f in fs]))] for st in stages]
    pcal_seed = []
    for j in range(n_pcal):
        p = sensing[j % len(sensing)]
        k = names.index(p)
        pcal_seed.append(fs[int(np.argmax([_single_line_info(loop, "PCAL", f, k, pcal_weight=1.0)
                                           for f in fs]))])
    return np.log10(np.array(pcal_seed + stage_seed, dtype=float))


def design_lines(loop, prior_std: dict, *, T: float = 60.0, caps: dict | None = None,
                 n_pcal: int = 4, band=None, names=PARAM_NAMES, seed_freqs=None):
    """Place the cal lines to minimise the A-optimal cost ``tr(Σ_prior⁻¹·Σ_snapshot)`` under the
    force caps — a **few** optimally-placed lines, not a broadband drive.

    Roster: ``n_pcal`` Pcal lines (share the ±200 mW budget equally) plus one actuator line on each
    hierarchical stage (M0/PUM/TST) at its force cap. The free variables are the line **frequencies**
    (log-spaced within ``band``); ``scipy.optimize.minimize`` (Nelder–Mead) refines them from a
    dispersion/log-spread seed. Returns a dict with the sized ``roster`` (list of
    ``(freq, actuator, disp_amp)``), the resulting ``cov``/``sigma`` (per-param absolute 1σ),
    ``sigma_frac`` (fractional for the scale knobs), the ``cost``, and the ``margins`` (prior σ /
    snapshot σ per parameter; >1 ⇒ the drift is resolved).
    """
    from scipy.optimize import minimize
    # Default to the real calibration band: ≥10 Hz (where the O4 floor is measured data, not the
    # clamped low-f endpoint, and the quad stages are past their longitudinal resonances so a
    # force-capped line stays at a physical amplitude — matching where real O3/O4 lines sit).
    band = band or (max(loop.fmin, 10.0), min(loop.fmax, 1200.0))
    lo, hi = np.log10(band[0]), np.log10(band[1])
    caps = caps if caps is not None else stage_force_caps(loop, names=names)
    stages = [n[len("kappa_"):] for n in names if n.startswith("kappa_")]
    nom = _nominal(loop, names)
    # prior std in ABSOLUTE θ units (fractional priors × nominal for the scale knobs)
    frac_knobs = {"g_c", "kappa_M0", "kappa_PUM", "kappa_TST"}
    abs_std = {n: (prior_std[n] * abs(nom[i]) if n in frac_knobs else prior_std[n])
               for i, n in enumerate(names)}
    P = _prior_cov(abs_std, names)

    def roster_from(x):
        pcal_f = 10.0 ** np.clip(x[:n_pcal], lo, hi)
        stage_f = 10.0 ** np.clip(x[n_pcal:], lo, hi)
        roster = [(float(f), "PCAL", line_displacement(loop, "PCAL", float(f),
                                                        pcal_weight=1.0 / n_pcal)) for f in pcal_f]
        roster += [(float(f), st, line_displacement(loop, st, float(f), caps))
                   for f, st in zip(stage_f, stages)]
        return roster

    def cost(x):
        gamma, _, _, _ = joint_fisher(loop, roster_from(x), T, names=names)
        return a_optimal_cost(gamma, P)

    if seed_freqs is None:
        seed = _seed_frequencies(loop, names, caps, band, n_pcal)
    else:
        seed = np.log10(np.asarray(seed_freqs, dtype=float))
    res = minimize(cost, seed, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 4000})
    # Nelder-Mead can wander uphill in this multimodal 7-D landscape; keep the better of seed/result.
    best_x = res.x if cost(res.x) < cost(seed) else seed
    roster = roster_from(best_x)
    gamma, _, corr, _ = joint_fisher(loop, roster, T, names=names)
    # Per-snapshot precision is the FREQUENTIST data CRB (each drift snapshot is an independent
    # measurement — the prior sets the drift we compare against, not part of one snapshot). Compute
    # it in prior-σ units for conditioning, then map back. The Bayesian posterior is used only as the
    # (well-posed) design objective above.
    Gs, std = _scaled_fisher(gamma, P)
    # Honest data CRB via eigen-inverse: a near-zero eigenvalue (a data-degenerate direction — e.g.
    # a stage with no in-band authority, or f_cc/τ with no high-f line) gives a HUGE variance and a
    # margin → 0, instead of the min-norm pinv silently reporting a small σ the prior is really
    # carrying. tol relative to the largest eigenvalue flags the rank.
    w, V = np.linalg.eigh(0.5 * (Gs + Gs.T))
    w = np.clip(w, 0.0, None)
    tol = max(w.max(), 1.0) * len(w) * 1e-12
    rank = int(np.sum(w > tol))
    inv_w = np.where(w > tol, 1.0 / w, 1.0 / tol)                 # degenerate dir → ~1/tol (huge var)
    scaled_cov = (V * inv_w) @ V.T
    scaled_var = np.clip(np.diag(scaled_cov), 0.0, np.inf)        # (σ_data/σ_prior)² per param
    sigma = np.sqrt(scaled_var) * std                             # absolute-θ 1σ
    cov = scaled_cov * np.outer(std, std)
    sigma_frac = {n: (sigma[i] / abs(nom[i]) if n in frac_knobs else sigma[i])
                  for i, n in enumerate(names)}
    # margin = drift amplitude (prior 1σ) / snapshot 1σ = 1/scaled σ; >1 ⇒ a snapshot resolves the drift
    margins = {n: (1.0 / np.sqrt(scaled_var[i]) if scaled_var[i] > 0 else np.inf)
               for i, n in enumerate(names)}
    return {"roster": roster, "cov": cov, "corr": corr, "sigma": dict(zip(names, sigma)),
            "sigma_frac": sigma_frac, "cost": a_optimal_cost(gamma, P), "margins": margins,
            "names": list(names), "T": T, "prior_std_abs": abs_std,
            "rank": rank, "full_rank": rank == len(names)}
