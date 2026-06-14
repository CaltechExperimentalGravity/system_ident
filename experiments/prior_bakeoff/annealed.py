"""Annealed convergence strategy for the prior bake-off.

Why the baseline fails (0/7): at low SNR with a mis-placed resonance, the
cheapest way to cut the misfit is to FLATTEN the model (gain/Q -> small) rather
than SLIDE the peak onto the data.  Far priors freeze; near priors collapse Q.

Annealing recipe (escape the basin, then sharpen):

  1. log-parameters (log f0, log Q, log|gain|): gain/Q cannot collapse through
     zero, and a "fractional" prior is literally a log-sigma.

  2. start BROAD: model Q is set LOW (Q_start) so the peak is wide, overlaps the
     true peak, and the f0 gradient is informative -> the peak can slide.  The
     start gain is scaled to keep the broad peak's height sensible
     (peak |H| ~ gain*Q/w0^2).

  3. SNR-weighted band early: bins are weighted by the MEASURED magnitude
     |H_meas| (high-SNR bins), so the few on-peak bins at the true frequency
     dominate the fit and pull f0 toward the data peak fast.  The weighting is
     annealed off (full band restored) as f0 locks.

  4. WIDE prior -> TIGHT prior: Lambda (the per-step trust region) starts small
     (large, mobile steps) and tightens (settle / sharpen) over the schedule.

  5. ADAPTIVE rising Q-floor: Q is held broad until f0 has actually SETTLED
     (measured by the per-pass change in f0), and only THEN is a rising Q-floor
     imposed to ratchet Q up out of the broad basin.  This is the crucial fix
     for the far (f0-50%) case: sharpening Q before f0 arrives freezes the slide
     and collapses the gain.  A fixed Q-ceiling kills late over-sharpen runaway.

Schedule state is carried in a closure (the adaptive "phase" advances only while
f0 is settled), with the pass index k available too.

Run:
    conda run --no-capture-output -n sysid python experiments/prior_bakeoff/annealed.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments/prior_bakeoff")
import harness as H  # noqa: E402

from system_ident.resonator import ResonatorModel  # noqa: E402
from system_ident.estimators.bayesian import bayesian_update  # noqa: E402


def _ramp(x, x0, x1):
    """Linear 0->1 ramp: 0 for x<=x0, 1 for x>=x1, linear between."""
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    return float(np.clip((x - x0) / (x1 - x0), 0.0, 1.0))


def make_strategy(
    *,
    log_mode=True,
    Q_start=2.0,
    gain_mode="preserve_peak",        # 'prior' | 'preserve_peak' | 'preserve_dc'
    # per-parameter prior (log-)sigmas, START (wide) -> END (tight): f0, Q, gain
    unc_f0=(0.7, 0.02),
    unc_Q=(2.0, 0.20),
    unc_gain=(1.0, 0.10),
    lam_floor=1e-3,
    # SNR band weighting: down-weight low-|H_meas| (noise-dominated) bins.  OFF
    # by default -- with a tall broad start it can let f0 drift the wrong way;
    # the measured-peak gate below already supplies the f0-locking signal.
    band_weight=False,
    band_frac=0.25,                   # bins with |H_meas| < band_frac*max get inflated
    band_infl=8.0,                    # err inflation for those (noise-only) bins
    # ADAPTIVE Q-floor gated on the MEASURED peak: only sharpen Q once the model
    # peak actually sits on the measured resonance (robust at this SNR via a
    # running-mean magnitude).  This prevents sharpening before f0 has arrived
    # (which would freeze the far-case slide and collapse the gain).
    gate_eps=0.06,                    # |log(f0_model) - log(f_peak)| to count "on peak"
    phase_need=5,                     # on-peak passes before Q-floor fully ramped
    Qfloor_lo=2.0,
    Qfloor_hi=14.0,
    Qceil=35.0,                       # hard upper clamp (kills over-sharpen runaway)
    # step size
    max_rel_step=0.8,
    n_pass=H.N_PASS,
):
    """Return (init_fn, step_fn) capturing the schedule hyperparameters."""

    state = {"phase": 0.0, "magbar": None}

    def _sigma_vec(model, prog):
        """Scheduled per-parameter prior sigma at anneal progress prog in [0,1]."""
        theta = np.asarray(model.params, dtype=float)
        max_abs = float(np.max(np.abs(theta)))

        def interp(pair):
            lo, hi = pair
            return lo * (hi / lo) ** prog

        uf, uq, ug = interp(unc_f0), interp(unc_Q), interp(unc_gain)
        if log_mode:
            return np.array([max(uf, lam_floor), max(uq, lam_floor), max(ug, lam_floor)])
        base = np.maximum(np.abs(theta), lam_floor * max_abs)
        return np.array([uf, uq, ug]) * base

    def init_fn(f0, Q, gain):
        state["phase"] = 0.0
        state["magbar"] = None
        if gain_mode == "preserve_peak":
            g0 = gain * (Q / Q_start)     # keep gain*Q (peak height) ~ prior
        elif gain_mode == "preserve_dc":
            g0 = gain
        else:
            g0 = gain
        m = ResonatorModel.from_resonances([(f0, Q_start)], g0, log=log_mode)
        sig = _sigma_vec(m, 0.0)
        return m, np.diag(1.0 / sig ** 2)

    def step_fn(freq, model, Hm, He, Lam_in, k):
        prog = float(np.clip(state["phase"] / max(phase_need, 1), 0.0, 1.0))

        # ---- running-mean magnitude -> robust measured-peak frequency ----
        mag = np.abs(Hm)
        if state["magbar"] is None:
            state["magbar"] = mag.copy()
        else:
            state["magbar"] = 0.6 * state["magbar"] + 0.4 * mag
        f_peak = float(freq[int(np.argmax(state["magbar"]))])

        # ---- SNR band weighting: down-weight noise-only bins ----
        He_use = He
        if band_weight and band_infl > 1.0:
            off = state["magbar"] < band_frac * float(np.max(state["magbar"]))
            He_use = He.copy()
            He_use[off] = He_use[off] * band_infl

        # ---- prior precision (wide -> tight with anneal phase) ----
        sig = _sigma_vec(model, prog)
        Lam = np.diag(1.0 / sig ** 2)

        model, Lam = bayesian_update(
            freq, model, Hm, He_use, Lam, max_rel_step=max_rel_step
        )

        # ---- gate on MEASURED peak: advance phase only when model is on-peak ----
        on_peak = abs(np.log(float(model.f0[0])) - np.log(f_peak)) < gate_eps
        if on_peak:
            state["phase"] += 1.0
        else:
            state["phase"] = 0.0          # keep Q broad until f0 arrives
        prog_after = float(np.clip(state["phase"] / max(phase_need, 1), 0.0, 1.0))

        # ---- adaptive rising Q-floor + fixed ceiling ----
        qf = Qfloor_lo + (Qfloor_hi - Qfloor_lo) * prog_after
        Qnow = float(model.Q[0])
        Qclamp = min(max(Qnow, qf), Qceil)
        if Qclamp != Qnow:
            theta = np.asarray(model.params, dtype=float).copy()
            theta[1] = np.log(Qclamp) if log_mode else Qclamp
            model = model.with_params(theta)

        return model, Lam

    return init_fn, step_fn


def run(name, **hp):
    init_fn, step_fn = make_strategy(**hp)
    rows = H.evaluate(init_fn, step_fn)
    nconv = H.report(name, rows)
    return nconv, rows


if __name__ == "__main__":
    run("annealed (defaults)")
