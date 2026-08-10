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

The first measurement pass designs a prior-robust excitation from the prior
model and its (large) error bars — power spread over the plausible resonance
band ``f0*(1±u)`` so a far prior still covers the true resonance; subsequent
passes use the now-trusted model for point-optimal excitation.

Each pass is an independent measurement of the same LTI system on the fixed
frequency grid, so passes are combined by inverse-variance weighting per
frequency bin and the model is refit on the accumulated estimate (see
``_accumulate``).
This keeps the first pass's band coverage while folding in the
resonance-sharpening information of the concentrated optimal passes — so the
global refit does not degrade as refinement proceeds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig

from .excitation import multisine_from_psd
from .fisher import fisher_matrix, safe_inverse
from .model import TFModel
from .safety import SafetyAbort


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

    models: dict  # per-DoF fitted TFModel
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
        exc = ch["excitation"]      # {dof: channel}  (where the multisine is injected)
        rb = ch["readback"]         # {dof: channel}  (the response y -> FRF output Y)
        # Closed-loop: the FRF input X is the after-controller drive u. Read it from
        # an optional per-DoF ``channels.drive`` monitor; with no drive channel it
        # falls back to the excitation channel (on the twin the excitation read-back
        # already returns u; on hardware that channel must be the drive monitor).
        drive_mon = ch.get("drive", {})
        self._warn_open_drive_monitor(config, drive_mon)
        m = config["measurement"]
        fs = float(m["fs"])
        T_perseg = float(m["segment_duration"])
        n_seg = int(m.get("n_segments", 8))
        px_total = float(m["px_total"])
        total_dur = T_perseg * n_seg
        nperseg = int(round(T_perseg * fs))

        # working frequency grid: rfft bins inside [freq_min, freq_max]
        f_all = np.fft.rfftfreq(nperseg, d=1 / fs)
        band = (f_all >= float(m["freq_min"])) & (f_all <= float(m["freq_max"]))
        freq = f_all[band]

        # Pintelon-Schoukens periodic-multisine measurement is the only path: a
        # periodic drive measured by a leakage-free synchronous DFT. The first
        # ``n_transient`` periods are dropped (settling) and carry no information,
        # so the reported Fisher time is scaled down accordingly.
        self._n_transient = int(m.get("n_transient", 1))
        self._fisher_time_factor = max(n_seg - self._n_transient, 1) / n_seg
        n_iter = int(config["strategy"].get("n_design_iter", 3))
        target = float(config["stop_criteria"]["uncertainty_target"])
        max_iter = int(config["stop_criteria"].get("max_iter", 5))
        sequential = config["run"].get("excitation_mode", "sequential") == "sequential"
        # First pass designs from the prior + its (large) error bars: a prior-
        # robust drive spread over the plausible resonance band f0*(1±u) so a far
        # prior still covers the true resonance. Later passes use the now-trusted
        # model for point-optimal excitation (prior_uncertainty=0).
        prior_uncertainty = float(config["strategy"].get("prior_uncertainty", 0.5))

        dofs = list(priors)
        models = {d: priors[d] for d in dofs}
        result = LoopResult(models=models)

        # Per-DoF accumulator for inverse-variance combination of every pass's
        # measured response on the (fixed) frequency grid. Each pass is an
        # independent measurement of the same LTI system, so combining them this
        # way keeps the broadband coverage of the first pass while folding in the
        # resonance-sharpening information of the optimal passes.
        accum = {
            d: {"w": np.zeros(len(freq)), "wH": np.zeros(len(freq), dtype=complex)}
            for d in dofs
        }

        # Per-DoF accumulated Fisher information.
        n_gauge = {d: priors[d].num.size + priors[d].den.size - 1 for d in dofs}
        info = {d: np.zeros((n_gauge[d], n_gauge[d])) for d in dofs}

        # capture pre-run state for the safe handoff
        self.watchdog.snapshot([exc[d] for d in dofs] + [rb[d] for d in dofs])

        # quiet-time readback PSD per DoF (no excitation) -> Pyy for design/Fisher
        Pyy = {}
        for d in dofs:
            quiet = self.backend.read([rb[d]], total_dur)[rb[d]]
            Pyy[d] = self._noise_psd(quiet, fs, nperseg, band)

        try:
            for it in range(max_iter):
                uncertainties = {}
                # First pass: prior-robust excitation from the prior + its error
                # bars (covers the plausible resonance band). Later passes use the
                # now-trusted model for point-optimal excitation.
                prior_u = prior_uncertainty if it == 0 else 0.0
                if sequential:
                    for d in dofs:
                        uncertainties[d] = self._measure_dof(
                            it, d, exc[d], rb[d], drive_mon.get(d, exc[d]),
                            models, Pyy, freq, fs,
                            nperseg, band, total_dur, px_total, n_iter, rng,
                            result, accum[d], info[d], prior_u=prior_u,
                        )
                        # stop this DoF's drive before moving to the next
                        self.backend.ramp_down(exc[d], self.watchdog.limits.ramp_down_secs)
                else:
                    self._inject_all(dofs, exc, models, Pyy, freq, fs, nperseg,
                                     total_dur, px_total, n_iter, rng,
                                     prior_u=prior_u)
                    for d in dofs:
                        uncertainties[d] = self._measure_dof(
                            it, d, exc[d], rb[d], drive_mon.get(d, exc[d]),
                            models, Pyy, freq, fs,
                            nperseg, band, total_dur, px_total, n_iter, rng,
                            result, accum[d], info[d], reuse_injection=True,
                            prior_u=prior_u,
                        )

                if uncertainties and all(u <= target for u in uncertainties.values()):
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
        self, it, dof, exc_ch, rb_ch, x_ch, models, Pyy, freq, fs, nperseg, band,
        total_dur, px_total, n_iter, rng, result, accum, info,
        reuse_injection=False, prior_u=0.0,
    ) -> float:
        # honour an operator STOP issued between segments (e.g. from the
        # dashboard, which calls watchdog.abort() out of band)
        if self.watchdog.aborted:
            raise SafetyAbort(self.watchdog.abort_reason or "operator STOP")

        Pxx = self.designer.design(
            freq, models[dof], Pyy[dof], px_total, n_iter=n_iter,
            prior_uncertainty=prior_u,
        )

        if not reuse_injection:
            drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng)
            self.backend.inject(exc_ch, drive, fs)

        # Read the response (Y), the FRF input X (the after-controller drive
        # monitor ``x_ch``; == exc_ch in open loop / when no drive channel is set),
        # and the excitation channel for the watchdog's drive check (deduped).
        read_chans = list(dict.fromkeys([rb_ch, x_ch, exc_ch]))
        seg = self.backend.read(read_chans, total_dur)
        report = self.watchdog.check(seg)  # raises SafetyAbort on a breach

        H_meas, H_err, coh = self._estimate(seg[x_ch], seg[rb_ch], fs, nperseg, band)
        # fold this pass into the inverse-variance accumulator, then refit on the
        # combined estimate (retains broadband coverage across passes)
        H_acc, err_acc = self._accumulate(accum, H_meas, H_err)
        models[dof] = self.estimator.fit(freq, H_acc, err_acc, models[dof])

        # add the information this pass delivered (at the updated model) and
        # report the uncertainty from the accumulated covariance
        info += fisher_matrix(
            freq, models[dof], Pxx, Pyy[dof], total_dur * self._fisher_time_factor
        )
        frac = self._frac_uncertainty(models[dof], safe_inverse(info))
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

    def _emit(self, it, dof, freq, model, Pxx, H_acc, coh, frac, drive, rms):
        """Push a per-iteration snapshot to the dashboard listener, if any.

        The snapshot is plain JSON-able data (keys match
        ``dashboard.ws.SNAPSHOT_FIELDS``); the loop stays independent of the
        dashboard stack.  Uses ``model.to_tf()`` for the num/den dashboard
        fields and ``model.eval(freq)`` for the magnitude.
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

    @staticmethod
    def _warn_open_drive_monitor(config, drive_mon) -> None:
        """Warn when a closed loop uses before-controller injection without a
        ``channels.drive`` monitor: on hardware the FRF input would then be the bare
        reference (X = r), biasing the estimate toward the closed-loop response T."""
        twin = config.get("twin", {})
        controllers = twin.get("controllers", {})
        if not controllers:
            return
        ip = twin.get("injection_point", "after_controller")

        def _ip(d):
            return ip if isinstance(ip, str) else ip.get(d, "after_controller")

        missing = [d for d in controllers
                   if _ip(d) == "before_controller" and d not in drive_mon]
        if missing:
            warnings.warn(
                f"closed-loop before-controller injection without channels.drive "
                f"for {missing}: the FRF input falls back to the excitation channel "
                "(the drive monitor on the twin, but the bare reference on real "
                "hardware — set channels.drive to the after-controller drive there).",
                stacklevel=2,
            )

    def _inject_all(self, dofs, exc, models, Pyy, freq, fs, nperseg, total_dur,
                    px_total, n_iter, rng, prior_u=0.0):
        for d in dofs:
            Pxx = self.designer.design(
                freq, models[d], Pyy[d], px_total, n_iter=n_iter,
                prior_uncertainty=prior_u,
            )
            drive = self._make_drive(Pxx, total_dur, fs, freq, nperseg, rng)
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
    def _make_drive(self, Pxx, total_dur, fs, freq, nperseg, rng):
        """Build the injectable periodic-multisine drive (Pintelon-Schoukens).

        The multisine stays a clean integer-period tiling — the leakage-free FRF needs
        whole periods. The actuator-safe on/off ramp is applied by the **backend** at
        injection (``ChannelBackend.ramp_s``, default 3 s), not baked into the drive.
        """
        n_periods = int(round(total_dur * fs / nperseg))
        return multisine_from_psd(Pxx, fs, nperseg, n_periods, freq, seed=rng)

    def _estimate(self, x, y, fs, nperseg, band):
        """Leakage-free periodic-multisine FRF (Pintelon-Schoukens)."""
        return self._estimate_tf_periodic(x, y, fs, nperseg, band, self._n_transient)

    # -- spectral helpers ----------------------------------------------------
    @staticmethod
    def _noise_psd(x, fs, nperseg, band):
        # quiet-time readout noise floor (Welch's PSD method; no FRF involved)
        _, P = sig.welch(x, fs=fs, nperseg=nperseg, noverlap=0)
        return P[band]

    @staticmethod
    def _longest_contiguous_run(idx: np.ndarray) -> tuple[int, int]:
        """The longest run of consecutive integers in a sorted, unique array
        ``idx``, as ``(start, stop)`` (``stop`` exclusive). Ties are broken in
        favour of the LATER run: later periods are more likely to be past any
        settling transient (#7)."""
        best_start, best_stop = int(idx[0]), int(idx[0]) + 1
        run_start = idx[0]
        for i in range(1, idx.size):
            if idx[i] != idx[i - 1] + 1:
                run_start = idx[i]
            run_stop = idx[i] + 1
            if run_stop - run_start >= best_stop - best_start:
                best_start, best_stop = int(run_start), int(run_stop)
        return best_start, best_stop

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
        near-peak bias that a windowed transform of a random record introduces.

        The estimate is the *reference-based* (ratio-of-averages) FRF
        ``H = mean_p(Y_p) / mean_p(X_p)``, NOT the average-of-ratios
        ``mean_p(Y_p/X_p)``.  Averaging the complex spectra over periods projects
        onto the known, deterministic multisine, which is uncorrelated with the
        loop/sensor noise; in **closed loop** this cancels the controller and
        returns the open-loop plant, where the naive ``S_yx/S_xx`` is biased by the
        fed-back noise.  ``X``/``Y`` are read from the drive monitor / response, so
        no separate reference channel is needed.

        Returns ``(H, H_err, coh)`` with ``H_err`` the absolute standard error of
        the complex FRF (from the period-to-period residual scatter) so the
        downstream estimators are unchanged.  Unexcited bins (tiny drive) get
        ``H_err = inf`` -> zero weight.

        NOT implemented here: #8's independent misalignment check ("a stashed
        drive array read back in place of X reports coherence 1.00000 on a
        200%-wrong FRF, because ``var_H`` is period-to-period scatter and a
        constant time offset is common-mode"). Investigated and deliberately
        deferred, not overlooked -- comparing ``Xbar``'s phase against the
        known injected design's phase does NOT catch it: a stashed X is
        bit-identical to the design (zero delay, zero residual) while a
        legitimate real readback can differ by hundreds of samples and is
        fine (S2.1), so that comparison alone cannot tell them apart --
        verified numerically, not assumed. Distinguishing them needs either
        (a) cross-referencing against Y/H against a trusted prior model,
        which this backend-agnostic static method has no business holding,
        or (b) a period-to-period variance floor on X itself, which
        false-positives on every twin/rtsfreerun test (their X is
        legitimately noise-free by construction). Both routes are
        hardware/noise-context-dependent, so the real fix is the structural
        invariant Stage D's ``CDSBackend`` already commits to (spec S2.1
        design consequence #1): never synthesise X, always a genuine
        ``getdata`` readback. Revisit here only if that invariant turns out
        to need a second, independent line of defense.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nperseg = int(nperseg)
        P = len(x) // nperseg
        if P < 2:
            raise ValueError("periodic FRF needs at least 2 whole periods")
        xr = x[: P * nperseg].reshape(P, nperseg)
        yr = y[: P * nperseg].reshape(P, nperseg)
        # Keep only the full-amplitude periods. Drives are injected through a soft
        # Tukey on/off ramp (the backends' default actuator-safe start/stop), so the
        # first/last period(s) are tapered and carry less energy; averaging them would
        # bias the leakage-free FRF. Restrict to the LONGEST genuinely CONTIGUOUS run
        # of full-energy periods -- not the span between the first and last full-energy
        # index (#7): a looped-taper geometry can leave low-energy periods *interior* to
        # the array (e.g. energies [1,1,1,1,.571,.492,1,1]), and the span
        # ``full[0]:full[-1]+1`` then keeps them too, which is the whole array. A clean
        # (un-ramped) drive has equal-energy periods, so this is a no-op there.
        e = (xr ** 2).sum(axis=1)
        if e.size and e.max() > 0:
            full = np.flatnonzero(e >= 0.999 * e.max())
            if full.size >= 2:
                start, stop = SysIDLoop._longest_contiguous_run(full)
                xr, yr = xr[start:stop], yr[start:stop]
                P = xr.shape[0]
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
                # A single surviving period has no residual scatter to measure
                # from -- exclude it with INFINITE uncertainty (zero weight in
                # _accumulate), never zero: zeroing var_H here previously fed
                # the ``1e-9 * |Hb|`` floor below, turning "exclude this pass"
                # into a measured weight of 4.56e+19 that then permanently
                # swamped every other pass for the rest of the campaign (#6).
                var_H = np.full_like(np.abs(Xbar), np.inf)

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
