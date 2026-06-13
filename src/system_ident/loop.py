"""The sysID orchestration loop.

``SysIDLoop`` wires a :class:`~system_ident.backends.base.ChannelBackend`, an
:class:`~system_ident.estimators.base.Estimator`, an
:class:`~system_ident.design.base.InputDesigner`, and a
:class:`~system_ident.safety.Watchdog`, then runs, per degree of freedom:

    measure quiet noise -> design excitation -> inject -> read -> safety-check
    -> estimate TF -> fit -> assess uncertainty -> repeat

in the config-selected excitation mode (sequential POS->Pitch->Yaw by default,
or simultaneous uncorrelated drives). The loop stops when every DoF reaches the
fractional-uncertainty target, when the iteration budget is exhausted, or when
the watchdog aborts; in every case it finishes through the shared safe-state
handoff.

The first measurement pass uses a flat, broadband excitation so the global
``invfreqs`` refit is well conditioned; subsequent passes use the optimal
designer to refine the (now-trusted) parameters.

Each pass is an independent measurement of the same LTI system on the fixed
Welch grid, so passes are combined by inverse-variance weighting per frequency
bin and the model is refit on the accumulated estimate (see ``_accumulate``).
This keeps the broadband coverage of the first pass while folding in the
resonance-sharpening information of the concentrated optimal passes — so the
global refit does not degrade as refinement proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig

from .estimators.bayesian import bayesian_update, frac_uncertainty, prior_precision
from .excitation import multisine_from_psd, timeseries_from_asd
from .fisher import fisher_matrix
from .model import TFModel
from .resonator import ResonatorModel, resonator_from_spectrum, resonator_from_tf
from .resonator_design import optimal_excitation as _res_optimal_excitation
from .resonator_design import prior_robust_excitation as _res_prior_robust_excitation
from .safety import SafetyAbort


def _resonances_of(model):
    """(f0, Q) pairs for a prior model (ResonatorModel or TFModel)."""
    if hasattr(model, "f0"):                       # ResonatorModel
        return list(zip(np.atleast_1d(model.f0), np.atleast_1d(model.Q)))
    poles = np.roots(np.asarray(model.den, dtype=float))   # TFModel
    out = []
    for p in poles[poles.imag > 1e-9]:
        out.append((abs(p) / (2 * np.pi), abs(p) / (2 * abs(p.real))))
    return out


def _nperseg_for_resolution(priors, fs, resolve_bins=6.0, q_safety=2.0):
    """Smallest nperseg whose bin width resolves every prior resonance.

    The half-power bandwidth f0/Q must span >= ``resolve_bins`` Welch bins for the
    -3 dB Q estimate to be unbiased, i.e. df = fs/nperseg <= (f0/Q)/resolve_bins,
    so nperseg >= resolve_bins * fs * Q / f0. We size from the PRIOR (f0, Q), which
    can underestimate the true Q (within the +/-50% envelope the true Q can be ~2x
    the prior), so a ``q_safety`` factor pessimistically narrows the assumed
    bandwidth. Returns the max over all prior modes.
    """
    need = 0.0
    for m in priors.values():
        for f0, Q in _resonances_of(m):
            if f0 > 0:
                need = max(need, resolve_bins * fs * (q_safety * Q) / f0)
    return int(np.ceil(need))


@dataclass
class IterationRecord:
    """One per-DoF measurement step, for the dashboard convergence view."""

    iteration: int
    dof: str
    max_frac_uncertainty: float
    output_rms: float


@dataclass
class LoopResult:
    """Outcome of a campaign."""

    models: dict  # TFModel (broadband_ls) or ResonatorModel (bayesian)
    history: list[IterationRecord] = field(default_factory=list)
    done: bool = False
    aborted: bool = False
    abort_reason: str | None = None


@dataclass
class SysIDLoop:
    """Closed-loop sysID orchestrator."""

    backend: object
    estimator: object
    designer: object
    watchdog: object
    listener: object = None  # optional callable(snapshot_dict) for the dashboard

    def run(
        self,
        config: dict,
        priors: dict[str, TFModel],
        seed: int | None = None,
    ) -> LoopResult:
        rng = np.random.default_rng(seed)
        ch = config["channels"]
        exc = ch["excitation"]      # {dof: channel}
        rb = ch["readback"]         # {dof: channel}
        m = config["measurement"]
        fs = float(m["fs"])
        T_perseg = float(m["segment_duration"])
        n_seg = int(m.get("n_segments", 8))
        px_total = float(m["px_total"])
        total_dur = T_perseg * n_seg
        nperseg = int(round(T_perseg * fs))

        # working frequency grid: Welch bins inside [freq_min, freq_max]
        f_all = np.fft.rfftfreq(nperseg, d=1 / fs)
        band = (f_all >= float(m["freq_min"])) & (f_all <= float(m["freq_max"]))
        freq = f_all[band]

        t_ramp = float(m.get("t_ramp", 0.0))
        # Measurement mode: "welch" (default, windowed Welch on a random record,
        # preserves existing behaviour byte-for-byte) or "periodic" (Pintelon-
        # Schoukens periodic multisine + leakage-free synchronous-DFT FRF).
        self._meas_mode = m.get("mode", "welch")
        self._n_transient = int(m.get("n_transient", 1))
        # Honest reported uncertainty in periodic mode: the dropped transient
        # period(s) carry no usable information, so scale the Fisher time down.
        self._fisher_time_factor = (
            max(n_seg - self._n_transient, 1) / n_seg
            if self._meas_mode == "periodic" else 1.0
        )
        n_iter = int(config["strategy"].get("n_design_iter", 3))
        target = float(config["stop_criteria"]["uncertainty_target"])
        max_iter = int(config["stop_criteria"].get("max_iter", 5))
        sequential = config["run"].get("excitation_mode", "sequential") == "sequential"
        loop_mode = config["strategy"].get("loop", "broadband_ls")
        prior_uncertainty = float(config["strategy"].get("prior_uncertainty", 0.5))
        # Hybrid mode: broadband_ls locks the model for the first n_locate passes,
        # then we hand off to the Bayesian MAP refinement (treating the locked
        # model as a fairly confident prior of strength lock_uncertainty).
        n_locate = int(config["strategy"].get("n_locate", 3))
        lock_uncertainty = float(config["strategy"].get("lock_uncertainty", 0.15))
        # Bayesian-refine excitation band: spread the (concentrated) drive over
        # the prior's plausible resonance band f0*(1 +/- exc_band) so a still-
        # uncertain prior covers where the resonance actually is (prior-robust
        # excitation). Tied to the prior strength so it is ONE knob: a confident
        # lock -> tight/efficient drive, an uncertain prior -> wider/robust drive.
        # Floored so even a tight lock keeps enough spread to cover the resonance.
        _base_band = lock_uncertainty if loop_mode == "hybrid" else prior_uncertainty
        exc_band = max(_base_band, 0.2)

        # Refine estimator: "bayesian" (MAP, default) or "spectrum" (robust -3 dB
        # half-power bandwidth). The spectrum refine needs the resonance peak
        # resolved, so auto-size the Welch segment from the prior's (f0, Q) -- this
        # only lengthens the segment, never shortens it, and re-derives the grid.
        refine = config["strategy"].get("refine", "bayesian")
        spectrum_refine = loop_mode == "hybrid" and refine == "spectrum"
        if spectrum_refine:
            # The -3 dB-bandwidth estimator is single-resonance: a smaller mode
            # sitting on a larger neighbour's shoulder never drops to its own
            # half-power, so multi-mode needs per-mode windowing + single-sided
            # width (not yet implemented). Fail loudly rather than silently
            # no-op; multi-mode plants should use refine='bayesian' / broadband_ls.
            for d in priors:
                n_modes = len(_resonances_of(priors[d]))
                if n_modes != 1:
                    raise ValueError(
                        f"refine='spectrum' supports single-resonance DoFs only; "
                        f"DoF {d!r} has {n_modes} modes. Use refine='bayesian' "
                        f"(or loop='broadband_ls') for multi-mode plants."
                    )
            nperseg = max(nperseg, _nperseg_for_resolution(priors, fs))
            T_perseg = nperseg / fs
            total_dur = T_perseg * n_seg
            f_all = np.fft.rfftfreq(nperseg, d=1 / fs)
            band = (f_all >= float(m["freq_min"])) & (f_all <= float(m["freq_max"]))
            freq = f_all[band]

        dofs = list(priors)
        models = {d: priors[d] for d in dofs}
        result = LoopResult(models=models)
        # Per-DoF history of independent spectrum-refine estimates (f0, Q, gain),
        # combined by running median; the across-pass scatter gives the uncertainty.
        spectrum_hist = {d: {"f0": [], "Q": [], "gain": []} for d in dofs}

        # Per-DoF accumulator for inverse-variance combination of every pass's
        # measured response on the (fixed) Welch grid. Each pass is an
        # independent measurement of the same LTI system, so combining them this
        # way keeps the broadband coverage of the first pass while folding in the
        # resonance-sharpening information of the optimal passes.
        accum = {
            d: {"w": np.zeros(len(freq)), "wH": np.zeros(len(freq), dtype=complex)}
            for d in dofs
        }

        # Per-DoF accumulated Fisher information (broadband_ls only).
        # The Bayesian mode tracks information via the posterior precision Lambda.
        if loop_mode != "bayesian":
            n_gauge = {d: priors[d].num.size + priors[d].den.size - 1 for d in dofs}
            info = {d: np.zeros((n_gauge[d], n_gauge[d])) for d in dofs}

        # Bayesian mode: per-DoF posterior precision matrix, initialised from
        # the prior.  Not used in broadband_ls mode.
        if loop_mode == "bayesian":
            Lambda = {d: prior_precision(priors[d], prior_uncertainty) for d in dofs}
        elif loop_mode == "hybrid":
            Lambda = {}   # filled at the locate -> refine transition (pass n_locate)

        # capture pre-run state for the safe handoff
        self.watchdog.snapshot([exc[d] for d in dofs] + [rb[d] for d in dofs])

        # quiet-time readback PSD per DoF (no excitation) -> Pyy for design/Fisher
        Pyy = {}
        for d in dofs:
            quiet = self.backend.read([rb[d]], total_dur)[rb[d]]
            Pyy[d] = self._welch(quiet, fs, nperseg, band)

        try:
            for it in range(max_iter):
                uncertainties = {}
                # Hybrid handoff: after n_locate broadband_ls passes, convert each
                # located TFModel to a ResonatorModel and seed the posterior so the
                # remaining passes run the Bayesian MAP refinement.
                if loop_mode == "hybrid" and it == n_locate:
                    for d in dofs:
                        models[d] = resonator_from_tf(models[d])
                        Lambda[d] = prior_precision(models[d], lock_uncertainty)
                bayesian_phase = loop_mode == "bayesian" or (
                    loop_mode == "hybrid" and it >= n_locate)
                if bayesian_phase:
                    # Concentrated prior-robust excitation per DoF; refine the model
                    # either by the robust -3 dB-bandwidth estimator (spectrum) or
                    # the Bayesian MAP step (default).
                    for d in dofs:
                        if spectrum_refine:
                            uncertainties[d] = self._measure_dof_spectrum(
                                it, d, exc[d], rb[d], models, Pyy, freq, fs,
                                nperseg, band, total_dur, px_total, n_iter, t_ramp,
                                rng, result, exc_band, spectrum_hist[d],
                            )
                        else:
                            uncertainties[d] = self._measure_dof_bayesian(
                                it, d, exc[d], rb[d], models, Pyy, freq, fs,
                                nperseg, band, total_dur, px_total, n_iter, t_ramp,
                                rng, result, Lambda, exc_band,
                            )
                        # stop this DoF's drive before moving to the next
                        self.backend.ramp_down(exc[d], self.watchdog.limits.ramp_down_secs)
                else:
                    # The first pass uses a flat, broadband excitation so the global
                    # invfreqs refit is well conditioned; later passes use the
                    # optimal designer to refine the now-trusted parameters.
                    design_iter = 0 if it == 0 else n_iter
                    if sequential:
                        for d in dofs:
                            uncertainties[d] = self._measure_dof(
                                it, d, exc[d], rb[d], models, Pyy, freq, fs,
                                nperseg, band, total_dur, px_total, design_iter, rng,
                                result, accum[d], info[d], t_ramp=t_ramp,
                            )
                            # stop this DoF's drive before moving to the next
                            self.backend.ramp_down(exc[d], self.watchdog.limits.ramp_down_secs)
                    else:
                        self._inject_all(dofs, exc, models, Pyy, freq, fs, nperseg,
                                         total_dur, px_total, design_iter, rng,
                                         t_ramp=t_ramp)
                        for d in dofs:
                            uncertainties[d] = self._measure_dof(
                                it, d, exc[d], rb[d], models, Pyy, freq, fs,
                                nperseg, band, total_dur, px_total, design_iter, rng,
                                result, accum[d], info[d], reuse_injection=True,
                                t_ramp=t_ramp,
                            )

                # Judge convergence on the refine phase only: the hybrid locate
                # phase (broadband_ls) reports an over-confident Fisher uncertainty
                # that would otherwise stop the loop before any refinement.
                in_refine = loop_mode != "hybrid" or it >= n_locate
                if in_refine and uncertainties and all(
                    u <= target for u in uncertainties.values()
                ):
                    result.done = True
                    break
        except SafetyAbort as exc_err:
            result.aborted = True
            result.abort_reason = str(exc_err)
            return result

        # normal teardown also runs the shared handoff
        self.watchdog.abort("normal teardown")
        return result

    # -- per-DoF measurement -------------------------------------------------
    def _measure_dof(
        self, it, dof, exc_ch, rb_ch, models, Pyy, freq, fs, nperseg, band,
        total_dur, px_total, n_iter, rng, result, accum, info,
        reuse_injection=False, t_ramp=0.0,
    ) -> float:
        # honour an operator STOP issued between segments (e.g. from the
        # dashboard, which calls watchdog.abort() out of band)
        if self.watchdog.aborted:
            raise SafetyAbort(self.watchdog.abort_reason or "operator STOP")

        Pxx = self.designer.design(freq, models[dof], Pyy[dof], px_total, n_iter=n_iter)

        if not reuse_injection:
            drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng, t_ramp)
            self.backend.inject(exc_ch, drive, fs)

        seg = self.backend.read([rb_ch, exc_ch], total_dur)
        report = self.watchdog.check(seg)  # raises SafetyAbort on a breach

        H_meas, H_err, coh = self._estimate(seg[exc_ch], seg[rb_ch], fs, nperseg, band)
        # fold this pass into the inverse-variance accumulator, then refit on the
        # combined estimate (retains broadband coverage across passes)
        H_acc, err_acc = self._accumulate(accum, H_meas, H_err)
        models[dof] = self.estimator.fit(freq, H_acc, err_acc, models[dof])

        # add the information this pass delivered (at the updated model) and
        # report the uncertainty from the accumulated covariance
        info += fisher_matrix(
            freq, models[dof], Pxx, Pyy[dof], total_dur * self._fisher_time_factor
        )
        frac = self._frac_uncertainty(models[dof], np.linalg.inv(info))
        result.history.append(
            IterationRecord(
                iteration=it, dof=dof, max_frac_uncertainty=frac,
                output_rms=report.output_rms.get(dof, float("nan")),
            )
        )
        self._emit(
            it, dof, freq, models[dof], Pxx, H_acc, coh, frac,
            report.drive_peaks.get(exc_ch, float("nan")),
            report.output_rms.get(dof, float("nan")),
        )
        return frac

    # -- Bayesian per-DoF measurement ----------------------------------------
    def _measure_dof_bayesian(
        self, it, dof, exc_ch, rb_ch, models, Pyy, freq, fs, nperseg, band,
        total_dur, px_total, n_iter, t_ramp, rng, result, Lambda, exc_band=0.2,
    ) -> float:
        """One measurement pass in Bayesian mode.

        Designs *prior-robust* excitation from the current model on every pass —
        the optimal (Fisher-information-maximising) drive averaged over the
        prior's plausible resonance band ``f0*(1 +/- exc_band)`` — so the
        concentrated drive covers where the resonance actually is even when the
        prior is off, then injects, reads, and performs one small damped MAP
        step via :func:`bayesian_update`.  The posterior precision ``Lambda[dof]``
        accumulates information across passes.
        """
        # honour an operator STOP issued between segments
        if self.watchdog.aborted:
            raise SafetyAbort(self.watchdog.abort_reason or "operator STOP")

        # Prior-robust excitation on the current ResonatorModel: the optimal
        # drive spread over the prior's plausible band so it covers the true
        # resonance even from an offset prior (efficient AND robust).
        Pxx = _res_prior_robust_excitation(
            freq, models[dof], Pyy[dof], px_total, exc_band, n_iter=n_iter
        )

        drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng, t_ramp)
        self.backend.inject(exc_ch, drive, fs)

        seg = self.backend.read([rb_ch, exc_ch], total_dur)
        report = self.watchdog.check(seg)  # raises SafetyAbort on a breach

        H_meas, H_err, coh = self._estimate(seg[exc_ch], seg[rb_ch], fs, nperseg, band)

        # MAP update: fold this measurement into the posterior
        models[dof], Lambda[dof] = bayesian_update(
            freq, models[dof], H_meas, H_err, Lambda[dof]
        )
        frac = frac_uncertainty(models[dof], Lambda[dof])

        result.history.append(
            IterationRecord(
                iteration=it, dof=dof, max_frac_uncertainty=frac,
                output_rms=report.output_rms.get(dof, float("nan")),
            )
        )
        self._emit(
            it, dof, freq, models[dof], Pxx, H_meas, coh, frac,
            report.drive_peaks.get(exc_ch, float("nan")),
            report.output_rms.get(dof, float("nan")),
        )
        return frac

    def _measure_dof_spectrum(
        self, it, dof, exc_ch, rb_ch, models, Pyy, freq, fs, nperseg, band,
        total_dur, px_total, n_iter, t_ramp, rng, result, exc_band, hist,
    ) -> float:
        """One spectrum-refine pass: estimate (f0, Q, gain) from the half-power
        bandwidth of this (independent) measurement, then combine with previous
        passes by running median; the across-pass scatter gives the uncertainty.

        Robust where the MAP/LS fit is fragile: Q comes from the bins around the
        peak (``resonator_from_spectrum``), not a fit to the window-distorted shape.
        Each pass is an independent measurement, so the running estimate tightens
        as ~1/sqrt(n_passes) (Q is variance-limited).
        """
        if self.watchdog.aborted:
            raise SafetyAbort(self.watchdog.abort_reason or "operator STOP")

        Pxx = _res_prior_robust_excitation(
            freq, models[dof], Pyy[dof], px_total, exc_band, n_iter=n_iter
        )
        drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng, t_ramp)
        self.backend.inject(exc_ch, drive, fs)
        seg = self.backend.read([rb_ch, exc_ch], total_dur)
        report = self.watchdog.check(seg)

        H_meas, _, coh = self._estimate(seg[exc_ch], seg[rb_ch], fs, nperseg, band)
        f0_guess = float(np.atleast_1d(models[dof].f0)[0])
        try:
            est = resonator_from_spectrum(freq, np.abs(H_meas), f0_guess=f0_guess)
            hist["f0"].append(float(est.f0[0]))
            hist["Q"].append(float(est.Q[0]))
            hist["gain"].append(float(est.gain))
        except ValueError:
            pass   # under-resolved / no clean peak this pass -> skip, keep prior passes

        # running median estimate + robust across-pass uncertainty (SEM ~ sigma/sqrt(n))
        n = len(hist["Q"])
        if n >= 1:
            f0m, Qm, gm = (float(np.median(hist[k])) for k in ("f0", "Q", "gain"))
            models[dof] = ResonatorModel.from_resonances([(f0m, Qm)], gm)
        if n >= 3:
            def _sem_frac(vals, med):
                sigma = 1.4826 * float(np.median(np.abs(np.asarray(vals) - med)))
                return sigma / np.sqrt(n) / max(abs(med), 1e-30)
            frac = max(_sem_frac(hist["f0"], f0m), _sem_frac(hist["Q"], Qm),
                       _sem_frac(hist["gain"], gm))
        else:
            frac = 1.0    # not enough passes yet to assess precision

        result.history.append(
            IterationRecord(iteration=it, dof=dof, max_frac_uncertainty=frac,
                            output_rms=report.output_rms.get(dof, float("nan")))
        )
        self._emit(
            it, dof, freq, models[dof], Pxx, H_meas, coh, frac,
            report.drive_peaks.get(exc_ch, float("nan")),
            report.output_rms.get(dof, float("nan")),
        )
        return frac

    def _emit(self, it, dof, freq, model, Pxx, H_acc, coh, frac, drive, rms):
        """Push a per-iteration snapshot to the dashboard listener, if any.

        The snapshot is plain JSON-able data (keys match
        ``dashboard.ws.SNAPSHOT_FIELDS``); the loop stays independent of the
        dashboard stack.

        Works for both ``TFModel`` (broadband_ls) and ``ResonatorModel``
        (bayesian): uses ``model.to_tf()`` for the num/den dashboard fields
        and ``model.eval(freq)`` for the magnitude.  Physical parameters
        (f0, etc.) are included when the model exposes them.
        """
        if self.listener is None:
            return
        tf = model.to_tf()           # TFModel for both model types (no-op for TFModel)
        snap = {
            "iteration": it,
            "dof": dof,
            "freq": np.asarray(freq).tolist(),
            "model_num": tf.num.tolist(),
            "model_den": tf.den.tolist(),
            "model_mag": np.abs(model.eval(freq)).tolist(),
            "meas_mag": np.abs(H_acc).tolist(),
            "coherence": np.asarray(coh).tolist(),
            "excitation_asd": np.sqrt(Pxx).tolist(),
            "max_frac_uncertainty": float(frac),
            "drive_level": float(drive),
            "output_rms": float(rms),
        }
        # Physical parameters (ResonatorModel only)
        if hasattr(model, "f0"):
            snap["model_f0"] = float(model.f0[0])
            snap["model_Q"] = float(model.Q[0])
            snap["model_gain"] = float(model.gain)
        self.listener(snap)

    def _inject_all(self, dofs, exc, models, Pyy, freq, fs, nperseg, total_dur,
                    px_total, n_iter, rng, t_ramp=0.0):
        for d in dofs:
            Pxx = self.designer.design(freq, models[d], Pyy[d], px_total, n_iter=n_iter)
            drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng, t_ramp)
            self.backend.inject(exc[d], drive, fs)

    @staticmethod
    def _accumulate(accum, H, H_err):
        """Inverse-variance-combine this pass into the running estimate.

        Bins with non-finite or zero error (e.g. unexcited bins in a
        concentrated optimal pass) contribute zero weight and are simply carried
        by the other passes.
        """
        w = np.where(np.isfinite(H_err) & (H_err > 0), 1.0 / H_err**2, 0.0)
        accum["w"] += w
        accum["wH"] += w * H
        good = accum["w"] > 0
        H_acc = np.where(good, accum["wH"] / np.where(good, accum["w"], 1.0), 0.0)
        err_acc = np.where(good, 1.0 / np.sqrt(np.where(good, accum["w"], 1.0)), np.inf)
        return H_acc, err_acc

    # -- excitation / estimation dispatch ------------------------------------
    def _make_drive(self, Pxx, total_dur, fs, freq, nperseg, rng, t_ramp):
        """Build the injectable drive for the configured measurement mode."""
        if self._meas_mode == "periodic":
            n_periods = int(round(total_dur * fs / nperseg))
            return multisine_from_psd(
                Pxx, fs, nperseg, n_periods, freq, seed=rng, t_ramp=t_ramp
            )
        return timeseries_from_asd(
            total_dur, fs, freq, np.sqrt(Pxx), seed=rng, t_ramp=t_ramp
        )

    def _estimate(self, x, y, fs, nperseg, band):
        """Estimate the FRF for the configured measurement mode."""
        if self._meas_mode == "periodic":
            return self._estimate_tf_periodic(x, y, fs, nperseg, band, self._n_transient)
        return self._estimate_tf(x, y, fs, nperseg, band)

    # -- spectral helpers ----------------------------------------------------
    @staticmethod
    def _welch(x, fs, nperseg, band):
        _, P = sig.welch(x, fs=fs, nperseg=nperseg, noverlap=0)
        return P[band]

    @staticmethod
    def _estimate_tf(x, y, fs, nperseg, band):
        f, Pxx = sig.welch(x, fs=fs, nperseg=nperseg, noverlap=0)
        _, Pyy = sig.welch(y, fs=fs, nperseg=nperseg, noverlap=0)
        _, Pyx = sig.csd(y, x, fs=fs, nperseg=nperseg, noverlap=0)
        H = Pyx / Pxx
        coh = np.clip(np.abs(Pyx) ** 2 / (Pxx * Pyy), 1e-6, 1 - 1e-9)
        n_avg = max(len(x) // nperseg, 1)
        rel_err = np.sqrt((1 - coh) / (2 * n_avg * coh))
        H_err = np.abs(H) * rel_err
        return H[band], H_err[band], coh[band]

    @staticmethod
    def _choose_transient(X, Y, n_min, P):
        """Number of leading periods to drop before the response is periodic.

        Closed-loop suspension settling is set by the *damped* mode time-constant
        ``tau = Q/(pi f0)``; one period is enough for the damped Q ~ 5-30 case but
        not for a high-Q (lightly damped) mode whose ringdown exceeds a period. We
        therefore start at ``n_min`` and drop further leading periods while their
        per-period FRF still drifts relative to the trailing-period scatter — an
        adaptive guard so the diagnosis does not depend on the prior's Q being right.
        """
        n_min = max(1, int(n_min))
        if P <= n_min + 2:
            return min(n_min, max(P - 1, 0))
        mag = np.abs(X).mean(0)
        mmax = float(np.max(mag))
        if mmax <= 0:
            return n_min
        strong = mag > 0.5 * mmax            # most-excited lines pin the FRF
        Hp = Y[:, strong] / X[:, strong]     # per-period FRF, shape (P, n_strong)
        n_tail = max(2, (P - n_min) // 2)
        tail = Hp[-n_tail:]
        ref = tail.mean(0)
        rms_ref = float(np.sqrt(np.mean(np.abs(ref) ** 2)))
        scat = float(np.sqrt(np.mean(np.abs(tail - ref[None, :]) ** 2)))
        # "settled enough": never chase a transient below ~0.1% of the FRF
        # magnitude, so a (near-)noise-free record is not over-dropped — the real
        # noise floor dominates this on a noisy measurement.
        scat = max(scat, 1e-3 * rms_ref)
        for p in range(n_min, P - 2):
            dev = float(np.sqrt(np.mean(np.abs(Hp[p] - ref) ** 2)))
            if dev <= 3.0 * scat:
                return p
        return n_min

    @staticmethod
    def _estimate_tf_periodic(x, y, fs, nperseg, band, n_transient=1):
        """Leakage-free FRF from a periodic (multisine) excitation — the P&S path.

        Reshapes the response into integer periods and takes a rectangular
        (un-windowed) DFT of each: because the drive is periodic over ``nperseg``,
        this is leakage-free, so a sharp resonance is measured without the
        near-peak bias that windowed Welch on a random record introduces.

        The estimate is the *reference-based* (ratio-of-averages) FRF
        ``H = mean_p(Y_p) / mean_p(X_p)``, NOT the average-of-ratios
        ``mean_p(Y_p/X_p)``.  Averaging the complex spectra over periods projects
        onto the known, deterministic multisine, which is uncorrelated with the
        loop/sensor noise; in **closed loop** this cancels the controller and
        returns the open-loop plant, where the naive ``S_yx/S_xx`` is biased by the
        fed-back noise.  ``X``/``Y`` are read from the drive monitor / response, so
        no separate reference channel is needed.

        Returns the same ``(H, H_err, coh)`` contract as :meth:`_estimate_tf`, with
        ``H_err`` the absolute standard error of the complex FRF (from the
        period-to-period residual scatter) so the downstream estimators are
        unchanged.  Unexcited bins (tiny drive) get ``H_err = inf`` -> zero weight.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nperseg = int(nperseg)
        P = len(x) // nperseg
        if P < 2:
            raise ValueError("periodic FRF needs at least 2 whole periods")
        xr = x[: P * nperseg].reshape(P, nperseg)
        yr = y[: P * nperseg].reshape(P, nperseg)
        X = np.fft.rfft(xr, axis=1)
        Y = np.fft.rfft(yr, axis=1)

        n_drop = SysIDLoop._choose_transient(X, Y, n_transient, P)
        Xk, Yk = X[n_drop:], Y[n_drop:]
        P_eff = Xk.shape[0]

        Xbar = Xk.mean(0)
        Ybar = Yk.mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = Ybar / Xbar
            # Nonparametric noise: scatter of the per-period residual about the
            # averaged FRF (Pintelon & Schoukens, the period-variance estimator).
            resid = Yk - H[None, :] * Xk
            if P_eff > 1:
                var_H = np.sum(np.abs(resid) ** 2, axis=0) / (
                    (P_eff - 1) * P_eff * np.abs(Xbar) ** 2
                )
            else:
                var_H = np.zeros_like(np.abs(Xbar))

        Hb = H[band]
        var_b = var_H[band]
        magb = np.abs(Xbar[band])
        H_err = np.sqrt(np.clip(var_b, 0.0, None))

        # Excited lines only: a concentrated optimal drive leaves many in-band
        # bins essentially undriven; mark those non-finite so _accumulate drops them.
        excited = magb > 1e-3 * float(np.max(magb)) if magb.size else magb.astype(bool)
        # Carry undriven bins as a clean zero (Xbar=0 -> Hb is inf/nan on a
        # noise-free twin); _accumulate gives them zero weight, but a non-finite
        # H would still poison ``w*H``, so zero them explicitly.
        Hb = np.where(excited & np.isfinite(Hb), Hb, 0.0)
        H_err = np.where(excited, H_err, np.inf)
        # Floor so a noise-free twin (var=0) keeps a finite, tiny weight.
        H_err = np.maximum(H_err, 1e-9 * np.abs(Hb))

        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.clip(1.0 / (1.0 + var_b / np.abs(Hb) ** 2), 1e-6, 1 - 1e-9)
        coh = np.where(excited & np.isfinite(coh), coh, 1e-6)
        return Hb, H_err, coh

    @staticmethod
    def _frac_uncertainty(model, cov) -> float:
        """Max fractional parameter uncertainty from a covariance matrix.

        Uses the same gauge as ``fisher_matrix`` (leading denominator
        coefficient fixed at 1, so its row/column is absent from ``cov``).
        """
        par = model.params.astype(float)
        par = par / par[model.n_num]
        keep = [k for k in range(len(par)) if k != model.n_num]
        gauge = np.abs(par[keep])
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(gauge > 0, sigma / gauge, sigma)
        return float(np.max(frac))
