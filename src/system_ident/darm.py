"""A representative closed-loop DARM twin and the P&S recovery on it.

The DARM loop is  d_err = C·(x_free + x_pc + Σ κ_i N_i c_i)/(1+G),  G = C·A·D,
with sensing C (cavity pole + delay), three-stage actuation A, and a derived
servo D.  Because the sensing delay makes a rational time-domain loop
intractable, the twin synthesises the closed-loop response in the frequency
domain (exact for the periodic P&S multisine; the suspension resonances sit
below the measurement band, so the in-band dynamics are smooth).

All numbers are *representative of an Advanced-LIGO DARM loop, not a specific
interferometer state* — a single coupled-cavity pole + delay for C, three
pendulum-stage actuators, and a UGF≈50 Hz open-loop gain shaped for stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

from .model import TFModel
from .reduced_plant import ReducedStateSpacePlant


def sensing_model(freq, g_c: float, f_cc: float, tau: float) -> np.ndarray:
    """Optical sensing response C(f) = g_c/(1+i f/f_cc)·exp(-i 2π f τ) [ct/m].

    The BRSE (tuned SRC) response — a single real coupled-cavity pole. The detuned case is
    :func:`sensing_model_detuned`, which splits this pole into a complex pair (coupled-cavity
    response) and reduces to exactly this expression at zero detuning.
    """
    f = np.asarray(freq, dtype=float)
    return g_c / (1.0 + 1j * f / f_cc) * np.exp(-2j * np.pi * f * tau)


def coupled_cavity_factor(freq, f_cc: float, alpha: float) -> np.ndarray:
    """Coupled detuned-cavity optical response  1 / (1 + i·u − α·u²),  u = f/f_cc.

    ``alpha`` is the (signed, dimensionless) SRC-detuning coupling, ``α = A·sin(2δ)``. This is the
    coupled arm+SRC cavity response (Cahillane 2017): at the design tuning the two coupled-cavity
    resonances combine to a single in-band pole, and drifting off tuning SPLITS them.

    * ``α = 0`` (tuned): the factor is the single real cavity pole ``1/(1+i f/f_cc)`` exactly, so
      the detuned sensing reduces to :func:`sensing_model` to machine precision.
    * ``0 < α < 1/4``: second-order, TWO real poles — the second descends from ∞ as α grows.
    * ``α = 1/4``: the poles COLLIDE (double pole).
    * ``α > 1/4`` (restoring optical spring, δ>0): the pair lifts off the real axis into a stable
      complex-conjugate resonance — f_cc has *split into a complex pair*.
    * ``α < 0`` (anti-restoring spring, δ<0): the poles stay real but one crosses into the RHP —
      the optical-spring instability. Both signs are represented honestly.

    The collision detuning is set by ``A``: ``A·sin(2δ_c) = 1/4`` (A≈1.03 → δ_c≈7°).
    """
    f = np.asarray(freq, dtype=float)
    if alpha == 0.0:                       # tuned: byte-identical to the single-pole sensing_model
        return 1.0 / (1.0 + 1j * f / f_cc)
    u = f / f_cc
    return 1.0 / (1.0 + 1j * f / f_cc - alpha * u * u)


def coupled_cavity_poles(f_cc: float, alpha: float) -> np.ndarray:
    """The pole frequencies [Hz] of :func:`coupled_cavity_factor` (roots of ``1 + i u − α u² = 0``,
    ``u = f/f_cc``), returned as complex ``f``. Real (imaginary-``u``) below the α=1/4 collision,
    a complex-conjugate pair above it. ``α = 0`` → the single pole at ``f = i·f_cc`` (+∞ second)."""
    if alpha == 0.0:
        return np.array([1j * f_cc, np.inf + 0j])
    disc = np.sqrt(complex(-1.0 + 4.0 * alpha))
    u = np.array([(1j + disc) / (2.0 * alpha), (1j - disc) / (2.0 * alpha)])
    return f_cc * u


def sensing_model_detuned(freq, g_c: float, f_cc: float, tau: float, alpha: float) -> np.ndarray:
    """Detuned DARM sensing: ``g_c · coupled_cavity_factor(f, f_cc, α) · e^{-i2πfτ}``.

    ``alpha = 0`` reproduces :func:`sensing_model` exactly (byte-match); ``alpha ≠ 0`` splits the
    coupled-cavity pole per :func:`coupled_cavity_factor`.
    """
    if alpha == 0.0:                       # tuned: byte-identical to the single-pole model
        return sensing_model(freq, g_c, f_cc, tau)
    f = np.asarray(freq, dtype=float)
    return g_c * coupled_cavity_factor(freq, f_cc, alpha) * np.exp(-2j * np.pi * f * tau)


def drift_profile(t, base: float, *, amp_frac: float = 0.05,
                  period_s: float = 7200.0, kind: str = "sine",
                  phase: float = 0.0) -> np.ndarray:
    """A simple, *known* slow time-variation θ(t) for one drifting scalar parameter.

    ``kind="sine"`` → base·(1 + amp_frac·sin(2π t/period_s + phase));
    ``kind="ramp"`` → base·(1 + amp_frac·t/period_s).

    Deliberately deterministic so the injected truth is exactly known when scoring
    the recovered drift.  This is the round-1 "basic time variation" placeholder; a
    seeded stochastic / GP wander and physically-accurate drift are a later swap.
    """
    t = np.asarray(t, dtype=float)
    if kind == "sine":
        return base * (1.0 + amp_frac * np.sin(2 * np.pi * t / period_s + phase))
    if kind == "ramp":
        return base * (1.0 + amp_frac * (t / period_s))
    raise ValueError(f"unknown drift kind {kind!r}")


def _pendulum_stage(f_pend: float, q: float, gain: float) -> TFModel:
    """One quad actuation stage: a pendulum force→displacement TF [m/ct].

    In the DARM band (well above f_pend) this is the ~1/f² actuator rolloff; the
    resonance itself sits below the measurement band.
    """
    return TFModel.from_resonances([(f_pend, q)], gain)


class ReducedStageShape:
    """A DARM actuation-stage shape backed by one column of the reduced QUAD plant.

    Duck-types the ``TFModel`` interface the loop uses — just ``.eval(freq)`` — so it drops into
    ``DARMLoop.stage``/``A``/``with_params(kappa_*)`` and ``snapshot_kappa``/``recover_actuation``
    with no signature changes: the shape is ``gain * [L3.disp.L / L_i.drive.L](f)``, the real
    reduced-order quad longitudinal chain (test mass = L3, DARM DOF = ``.L``) instead of a lumped
    pendulum. ``gain`` folds the per-stage counts→force calibration; its absolute value is
    irrelevant to the closed-loop FRFs because ``D = G/(A·C)`` is derived (only stage *shapes*
    matter). ``.eval`` is memoised per ``freq``-array identity so repeated snapshots on the same
    grid don't re-solve the 59-state system per bin.
    """

    def __init__(self, subplant, in_idx: int, gain: float = 1.0, out_idx: int = 0):
        self._sub = subplant
        self._ii = int(in_idx)
        self._oi = int(out_idx)
        self.gain = float(gain)
        self._cache: dict = {}

    def eval(self, freq) -> np.ndarray:
        freq = np.asarray(freq, dtype=float)
        key = (id(freq), freq.shape, float(freq[0]), float(freq[-1]))
        col = self._cache.get(key)
        if col is None:
            col = self._sub.eval(freq)[:, self._oi, self._ii]
            self._cache[key] = col
        return self.gain * col


@dataclass
class DARMLoop:
    """Representative closed-loop DARM twin (single loop, three actuation stages)."""

    fs: float = 4096.0
    fmin: float = 10.0
    fmax: float = 1500.0
    # sensing
    g_c: float = 1.0e6          # optical gain [ct/m]
    f_cc: float = 360.0         # coupled-cavity pole [Hz]
    tau: float = 77.0e-6        # light-travel / processing delay [s]
    # SRC-detuning coupled cavity (Cahillane). delta = 0 is BRSE (tuned): C() is the single real
    # cavity pole exactly. Detuning splits it — α = detune_coupling·sin(2·delta); the coupled-cavity
    # pole becomes a complex-conjugate pair once |α|>1/4 (collision). delta is the physical knob the
    # TV tracker drifts (±~7° in practice); detune_coupling sets the collision detuning
    # (A·sin(2δ_c)=1/4 → A≈1.03 puts δ_c≈7°). See :func:`coupled_cavity_factor`.
    delta: float = 0.0            # SRC detuning phase [rad]
    detune_coupling: float = 1.033  # A in α=A·sin(2δ); collision at δ_c≈7° (A·sin2δ_c=1/4)
    # actuation: name -> (stage TFModel, kappa strength)
    stages: dict = field(default_factory=dict)
    # hierarchical control allocation: name -> distribution filter (any object with .eval(freq)),
    # so the per-stage actuation is A_i = κ_i · D_i(f) · N_i(f). Absent → unity (stages sum raw,
    # no crossover). The crossover between adjacent stages lives entirely in these filters — the
    # mechanical columns N_i don't cross (TST always dominates). Populate from the twin's DARM
    # actuation design; cal lines then measure where |A_i| = |A_{i+1}|.
    distribution: dict = field(default_factory=dict)
    # open-loop-gain shape (used to derive the servo D = G/(A·C))
    f_ugf: float = 50.0         # unity-gain frequency [Hz]
    f_hi: float = 400.0         # high-frequency control rolloff pole [Hz]
    # disturbance / sensing noise ASDs (set on the twin used for simulation)
    disturbance_asd: float = 0.0   # process (length) disturbance, [m/√Hz] referred to x_free
    sensor_asd: float = 0.0        # readout noise on d_err, [ct/√Hz]

    @classmethod
    def default(cls) -> "DARMLoop":
        stages = {
            "UIM": (_pendulum_stage(0.43, 300.0, 4.0e5), 1.00),
            "PUM": (_pendulum_stage(1.00, 200.0, 8.0e4), 0.40),
            "TST": (_pendulum_stage(3.40, 100.0, 1.2e4), 0.08),
        }
        return cls(stages=stages)

    #: Reduced-quad actuator column per DARM stage: (drive channel, κ). Test mass = L3,
    #: DARM DOF = longitudinal (.L). The hierarchical DARM actuation drives M0 (top), PUM (L2),
    #: TST (L3); all read out at L3.disp.L. (Channel↔stage map — adjust if the 40m convention
    #: differs.) κ here are placeholders for the non-hierarchical loop only; the hierarchical loop
    #: (``default_reduced(hierarchical=True)``) uses the M0-damped compliances with κ=1 and the
    #: nested-offload distribution filters — see ``darm_actuation.hierarchical_stage_shapes``.
    _REDUCED_MAP = {"M0":  ("M0.drive.L", 1.00),
                    "PUM": ("L2.drive.L", 0.40),
                    "TST": ("L3.drive.L", 0.08)}

    @classmethod
    def default_reduced(cls, *, fmin: float = 0.3, hierarchical: bool = False) -> "DARMLoop":
        """DARM loop whose actuation stages are the real reduced-order QUAD longitudinal chain.

        Each stage's shape is a column of ``quad_reduced_50hz`` (``L_i.drive.L → L3.disp.L``)
        instead of a lumped pendulum, so the twin carries the true multi-resonance structure and
        cross-stage shapes. ``fmin`` defaults to 0.3 Hz so the quad longitudinal modes (~0.43–2 Hz)
        sit in band. The absolute scale is irrelevant (the loop depends only on stage shapes via
        ``D = G/(A·C)``); the per-stage gains just keep magnitudes O(1) and preserve κ semantics.

        ``hierarchical=True`` builds the twin's nested-offload DARM actuation: the stage shapes are
        the **M0-damped** QUAD compliances (``darm_actuation.hierarchical_stage_shapes`` — the twin
        damps the quad before designing the hierarchy) and the distribution filters ``D_M0=O_A·O_B,
        D_PUM=O_A, D_TST=1`` push the drive up the chain. The offload runs in FORCE units, so the
        compliances alone carry the relative stage strengths (κ_i = 1) — no separate drive-referred
        authority weighting, which would double-count. With this, the DARM-referred contributions
        cross at the design targets ``F_PT≈0.5 Hz`` (M0/PUM) and ``F_EP≈10 Hz`` (PUM/TST), which the
        cal lines measure. ``hierarchical=False`` keeps the lumped placeholder κ on the undamped
        columns (no distribution, no crossover — TST always dominates).
        """
        if hierarchical:
            from .darm_actuation import hierarchical_stage_shapes, hierarchical_distribution
            shapes = hierarchical_stage_shapes()
            stages = {name: (shapes[name], 1.0) for name in shapes}   # κ=1; force-unit offload
            return cls(stages=stages, fmin=fmin, distribution=hierarchical_distribution())
        plant = ReducedStateSpacePlant.load("quad")
        acts = [ch for ch, _ in cls._REDUCED_MAP.values()]
        sub = plant.subplant(sensors=["L3.disp.L"], actuators=acts)
        f_anchor = np.array([100.0])
        stages = {}
        for j, (name, (chan, kappa)) in enumerate(cls._REDUCED_MAP.items()):
            gain = 1.0 / abs(sub.eval(f_anchor)[0, 0, j])         # unit at 100 Hz
            stages[name] = (ReducedStageShape(sub, in_idx=j, gain=gain), kappa)
        return cls(stages=stages, fmin=fmin)

    @property
    def ports(self) -> list[str]:
        return ["PCAL"] + list(self.stages)

    def with_params(self, **overrides) -> "DARMLoop":
        """Copy of the loop with scalar parameters or stage strengths overridden.

        Recognised keys: the sensing/shape fields ``g_c, f_cc, tau, f_ugf, f_hi`` and
        ``kappa_<STAGE>`` (e.g. ``kappa_TST=0.09``) to set one actuation strength.
        Everything else is inherited.  Lets a caller evaluate the loop at a drifted
        operating point without mutating the base twin — the locally-stationary
        snapshot at one instant of a slowly time-varying plant.
        """
        stages = dict(self.stages)
        scalar = {}
        for key, val in overrides.items():
            if key.startswith("kappa_"):
                name = key[len("kappa_"):]
                if name not in stages:
                    raise KeyError(f"unknown stage {name!r}")
                tf, _ = stages[name]
                stages[name] = (tf, float(val))
            else:
                scalar[key] = val
        return replace(self, stages=stages, **scalar)

    # -- elements ----------------------------------------------------------
    def alpha(self) -> float:
        """Signed SRC-detuning coupling α = detune_coupling·sin(2·δ) (dimensionless). 0 at the
        tuned BRSE point; |α|>1/4 is past the pole-collision (complex pair, δ>0 / RHP, δ<0)."""
        return float(self.detune_coupling * np.sin(2.0 * self.delta))

    def cavity_poles(self) -> np.ndarray:
        """The two coupled-cavity pole frequencies [Hz] (complex) at the current δ — real below the
        α=1/4 collision, a complex-conjugate pair above it (see :func:`coupled_cavity_poles`)."""
        return coupled_cavity_poles(self.f_cc, self.alpha())

    def C(self, freq) -> np.ndarray:
        # Coupled detuned-cavity sensing; δ=0 (default) reduces to the single-pole BRSE model
        # exactly, and detuning splits the cavity pole into a complex pair (coupled_cavity_factor).
        return sensing_model_detuned(freq, self.g_c, self.f_cc, self.tau, self.alpha())

    def stage(self, name: str, freq) -> np.ndarray:
        """Per-stage actuation A_i(f) = κ_i · D_i(f) · N_i(f) [counts→DARM displacement], where
        D_i is the hierarchical distribution filter (unity if none set for this stage)."""
        tf, kappa = self.stages[name]
        out = kappa * tf.eval(freq)
        dist = self.distribution.get(name)
        if dist is not None:
            out = out * dist.eval(freq)
        return out

    def A(self, freq) -> np.ndarray:
        return sum(self.stage(n, freq) for n in self.stages)

    def _ol_shape(self, freq) -> np.ndarray:
        """The *designed* open-loop gain G(f): integrator to UGF, a control
        rolloff pole, and the sensing transport delay — shaped for a stable loop
        with healthy phase margin.  D is then derived so G = A·D·C exactly."""
        f = np.asarray(freq, dtype=float)
        # The 1/f integrator diverges at DC (f=0, the rfft grid's first bin, always
        # out of band). Compute with the expected divide/invalid warnings silenced,
        # then set the DC bin to 0 so nothing downstream (1+G, C/(1+G), …) trips an
        # invalid-value warning on the propagated infinity.
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (self.f_ugf / (1j * f)) / (1.0 + 1j * f / self.f_hi) \
                * np.exp(-2j * np.pi * f * self.tau)
        return np.where(f == 0.0, 0.0, g)

    def G(self, freq) -> np.ndarray:
        return self._ol_shape(freq)

    def D(self, freq) -> np.ndarray:
        """Representative digital servo, derived from the designed G: D = G/(A·C)."""
        return self.G(freq) / (self.A(freq) * self.C(freq))

    def R(self, freq) -> np.ndarray:
        """The calibration deliverable: counts→displacement response (1+G)/C."""
        return (1.0 + self.G(freq)) / self.C(freq)

    # -- closed-loop FRFs per injection point ------------------------------
    def frf_pcal(self, freq) -> np.ndarray:
        """d_err/x_pc = C/(1+G)  (Pcal displacement → DARM error)."""
        return self.C(freq) / (1.0 + self.G(freq))

    def frf_stage(self, name: str, freq) -> np.ndarray:
        """d_err/c_i = C·κ_i·N_i/(1+G)  (stage drive counts → DARM error)."""
        return self.C(freq) * self.stage(name, freq) / (1.0 + self.G(freq))

    def disturbance_to_derr(self, freq) -> np.ndarray:
        """x_free enters at the test mass like x_pc: C/(1+G)."""
        return self.frf_pcal(freq)

    def sensing_to_derr(self, freq) -> np.ndarray:
        """Readout noise n adds at d_err and is loop-suppressed: 1/(1+G)."""
        return 1.0 / (1.0 + self.G(freq))

    # -- simulation -----------------------------------------------------------
    def _white(self, asd: float, n: int, rng) -> np.ndarray:
        if asd == 0.0:
            return np.zeros(n)
        # one-sided ASD A -> discrete white-noise std A·sqrt(fs/2)
        return rng.standard_normal(n) * asd * np.sqrt(self.fs / 2.0)

    def simulate(self, drives: dict, n: int, rng) -> np.ndarray:
        """Synthesise d_err[n] for injected ``drives`` under process disturbance +
        sensing noise, by frequency-domain closed-loop filtering.

        Deterministic drives are periodic (P&S multisine), so rfft·H·irfft is the
        exact periodic steady-state response; the stochastic disturbance/sensing
        noise are coloured by their closed-loop transfer functions.
        """
        n = int(n)
        f = np.fft.rfftfreq(n, d=1.0 / self.fs)
        Y = np.zeros(len(f), dtype=complex)
        for port, x in drives.items():
            x = np.asarray(x, dtype=float)
            xf = np.zeros(n)
            xf[: min(len(x), n)] = x[: n]
            H = self.frf_pcal(f) if port == "PCAL" else self.frf_stage(port, f)
            H = np.where(np.isfinite(H), H, 0.0)
            Y += np.fft.rfft(xf) * H
        # process disturbance x_free -> d_err  (C/(1+G))
        if self.disturbance_asd:
            w = self._white(self.disturbance_asd, n, rng)
            Hd = np.where(np.isfinite(self.disturbance_to_derr(f)),
                          self.disturbance_to_derr(f), 0.0)
            Y += np.fft.rfft(w) * Hd
        # readout/sensing noise n -> d_err  (1/(1+G))
        if self.sensor_asd:
            v = self._white(self.sensor_asd, n, rng)
            Hs = np.where(np.isfinite(self.sensing_to_derr(f)),
                          self.sensing_to_derr(f), 0.0)
            Y += np.fft.rfft(v) * Hs
        return np.fft.irfft(Y, n)


def recover_response(H_pcal: np.ndarray, H_err: np.ndarray) -> tuple:
    """Model-free DARM response R = 1/(d_err/x_pc) with its CRB envelope.

    R = 1/H_pcal;  σ_R = σ_H/|H_pcal|²  (first-order propagation of the FRF error).
    Unexcited bins (non-finite H_err) get σ_R = inf.
    """
    H = np.asarray(H_pcal)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.abs(H) > 0, 1.0 / H, 0.0)
        R_sigma = np.where(np.isfinite(H_err) & (np.abs(H) > 0),
                           np.asarray(H_err) / np.abs(H) ** 2, np.inf)
    return R, R_sigma


def fit_sensing(freq, C_meas, C_err, p0) -> tuple:
    """Weighted complex least-squares fit of C(f)=g_c/(1+i f/f_cc)·e^{-i2πfτ}.

    Returns (params, sigma) dicts over {g_c, f_cc, tau}; sigma from the
    Gauss–Newton covariance (JᵀJ)⁻¹ at the solution (the CRB for white,
    correctly-weighted residuals).
    """
    f = np.asarray(freq, dtype=float)
    Cm = np.asarray(C_meas)
    good = np.isfinite(C_err) & (np.asarray(C_err) > 0) & np.isfinite(Cm)
    f, Cm, w = f[good], Cm[good], 1.0 / np.asarray(C_err)[good]

    def resid(p):
        g_c, f_cc, tau = p
        r = (sensing_model(f, g_c, f_cc, tau) - Cm) * w
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, np.asarray(p0, dtype=float), method="lm")
    params = {"g_c": sol.x[0], "f_cc": sol.x[1], "tau": sol.x[2]}
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        s = np.full(3, np.nan)
    sigma = {"g_c": s[0], "f_cc": s[1], "tau": s[2]}
    return params, sigma


def recover_actuation(freq, H_stage, H_pcal, N_stage, comb_err) -> tuple:
    """Stage strength κ_i = mean of (H_stage/H_pcal)/N_i, Pcal as the ruler.

    H_stage/H_pcal = κ_i N_i, so dividing by the known stage shape N_i yields a
    per-bin κ estimate; combine by inverse-variance over the excited bins.
    """
    ratio = np.asarray(H_stage) / np.asarray(H_pcal) / np.asarray(N_stage)
    good = np.isfinite(comb_err) & (np.asarray(comb_err) > 0) & np.isfinite(ratio)
    # error on κ per bin ≈ comb_err / |N_i|
    sig_k = np.asarray(comb_err)[good] / np.abs(np.asarray(N_stage)[good])
    w = 1.0 / sig_k ** 2
    kappa = float(np.sum(w * np.real(ratio[good])) / np.sum(w))
    kappa_sigma = float(1.0 / np.sqrt(np.sum(w)))
    return kappa, kappa_sigma


# ---------------------------------------------------------------------------
# Swept-sine vs multisine comparison harness
# Imports at the bottom: darm_adapter imports DARMLoop (defined above), so the
# cycle is safe — Python's module system resolves DARMLoop before these lines run.
# ---------------------------------------------------------------------------
from .backends.darm_adapter import DARMBackend   # noqa: E402 — cycle-safe bottom import
from .excitation import multisine_from_psd
from .loop import SysIDLoop


def _band_grid(loop, nperseg):
    fa = np.fft.rfftfreq(int(nperseg), d=1.0 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def multisine_response_sigma(loop, *, nperseg=4096, n_periods=16, px_total=1.0, seed=0):
    """One Pcal multisine over the whole band → R(f) and its per-bin σ (CRB envelope).

    n_periods=16: with the 3 s actuator ramp this leaves ~10 full-energy periods
    (P_eff≈9), so the per-bin variance is genuinely estimated — not the floored,
    fabricated uncertainty that n_periods=8 (only 2 full periods → P_eff=1) produces.
    """
    fa, band, freq = _band_grid(loop, nperseg)
    Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    x = multisine_from_psd(Pxx, loop.fs, nperseg, n_periods, freq, seed=np.random.default_rng(seed))
    be.inject("PCAL_EXC", x, loop.fs)
    T_total = (nperseg * n_periods) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], T_total)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                  loop.fs, nperseg, band, n_transient=1)
    R, R_sigma = recover_response(H, H_err)
    return freq, R, R_sigma, T_total


def swept_sine_response_sigma(loop, freq_points, *, nperseg=4096, dwell_periods=4,
                              px_total=1.0, seed=0):
    """Idealised swept sine on the same twin: each frequency a single-line, full-power,
    **ramp-free** dwell of ``dwell_periods`` periods (≥4 required so P_eff≥3 and the
    per-bin variance is genuinely estimated — 2 periods give P_eff=2 which underflows
    to the 1e-9 estimator floor, producing a fabricated σ; ramp-free so the baseline
    is not handicapped by the 3 s actuator ramp).

    Returns ``(freq_points, R_sigma, T_used)`` — absolute σ(R) per point and the honest
    wall-clock ``T_used = len·dwell_periods·nperseg/fs`` the sweep spends.
    """
    freq_points = np.asarray(freq_points, dtype=float)
    nperseg = int(nperseg)
    n_per = max(2, int(dwell_periods))
    fa = np.fft.rfftfreq(nperseg, d=1.0 / loop.fs)
    out = np.full(len(freq_points), np.inf)
    rng = np.random.default_rng(seed)
    for i, fpt in enumerate(freq_points):
        k = int(np.argmin(np.abs(fa - fpt)))
        Pxx = np.array([px_total / (fa[1] - fa[0])])   # all power on the one line
        band = (fa >= fa[k] - 1e-9) & (fa <= fa[k] + 1e-9)
        be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=rng, ramp_s=0.0)
        x = multisine_from_psd(Pxx, loop.fs, nperseg, n_per, np.array([fa[k]]), seed=rng)
        be.inject("PCAL_EXC", x, loop.fs)
        seg = be.read(["PCAL_EXC", "DARM_ERR"], (nperseg * n_per) / loop.fs)
        # ramp-free single tone → no transient → keep all periods (n_transient=0)
        H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                      loop.fs, nperseg, band, n_transient=0)
        sel = np.isfinite(H_err) & (np.abs(H) > 0)
        if np.any(sel):
            R, R_sigma = recover_response(H, H_err)
            out[i] = float(np.min(R_sigma[sel]))      # absolute σ(R) at the driven line
    T_used = len(freq_points) * n_per * nperseg / loop.fs
    return freq_points, out, T_used


def sweep_time_to_match_coverage(loop, *, nperseg=4096, dwell_periods=2):
    """Wall-clock for a swept sine to visit EVERY band bin for ``dwell_periods`` each —
    the full-band coverage the single multisine window gets in n_periods·nperseg/fs s."""
    _, _, freq = _band_grid(loop, nperseg)
    return len(freq) * max(2, int(dwell_periods)) * nperseg / loop.fs
