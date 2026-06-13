"""Hybrid loop mode: broadband_ls locate/lock -> Bayesian MAP refine.

The hybrid mode gets the best of both: broadband_ls (broadband excitation +
global invfreqs + accumulation) robustly locks the model from a far prior
(±50%, prior-independent), then hands off to the conservative Bayesian MAP
refinement (concentrated excitation, small damped steps) for low-SNR tracking.
This is the production pipeline for the real ±20-50%-prior + low-SNR envelope.
"""

import numpy as np
import pytest

from system_ident.config import RunConfig
from system_ident.loop import SysIDLoop
from system_ident.resonator import ResonatorModel
from system_ident.model import TFModel


def _cfg(prior_f0, prior_Q, prior_gain, n_locate=3, max_iter=6):
    return {
        "run": {"excitation_mode": "sequential"},
        "channels": {"excitation": {"POS": "C1:EXC"}, "readback": {"POS": "C1:RSP"}},
        "measurement": {"fs": 32, "freq_min": 0.1, "freq_max": 5.0,
                        "segment_duration": 64.0, "n_segments": 4, "px_total": 1.0, "t_ramp": 4.0},
        "twin": {"sensor_asd": 3e-3, "disturbance_asd": 3e-3,
                 "plant": {"POS": {"resonances": [[1.0, 20]], "gain": 100}}},
        "priors": {"POS": {"resonances": [[prior_f0, prior_Q]], "gain": prior_gain}},
        "strategy": {"estimator": "invfreqs", "input_designer": "pintelon_schoukens",
                     "n_design_iter": 3, "loop": "hybrid",
                     "n_locate": n_locate, "lock_uncertainty": 0.15, "exploration": 0.2},
        "safety": {"actuator_sat": 1e9, "rms_ceiling": {"POS": 1e9}, "ramp_down_secs": 2.0},
        "stop_criteria": {"uncertainty_target": 1e-9, "max_iter": max_iter},
    }


def _run(cfg, seed=3):
    rc = RunConfig(raw=cfg)
    be = rc.build_twin_backend(seed=seed)
    loop = SysIDLoop(be, rc.build_estimator(), rc.build_designer(), rc.build_watchdog(be))
    return rc, be, loop.run(rc.raw, rc.build_priors(), seed=seed)


@pytest.mark.parametrize("pf,pq,pg", [(1.5, 30, 150), (0.5, 10, 50), (1.3, 12, 70)])
def test_hybrid_cold_starts_from_far_prior(pf, pq, pg):
    """A far prior (±50% / mixed) is locked by broadband_ls then refined."""
    rc, be, result = _run(_cfg(pf, pq, pg))
    model = result.models["POS"]
    # hybrid hands off to the physical model for the refine phase
    assert isinstance(model, ResonatorModel)
    f0, Q = float(model.f0[0]), float(model.Q[0])
    assert abs(f0 - 1.0) < 0.05, f"f0={f0:.3f} (prior {pf}, true 1.0)"
    assert abs(Q - 20.0) / 20.0 < 0.25, f"Q={Q:.1f} (prior {pq}, true 20)"


def _cfg_spectrum(prior_f0, prior_Q, prior_gain, n_locate=3, max_iter=16):
    """Hybrid config with the robust -3 dB-bandwidth (spectrum) refine."""
    cfg = _cfg(prior_f0, prior_Q, prior_gain, n_locate=n_locate, max_iter=max_iter)
    cfg["strategy"]["refine"] = "spectrum"
    cfg["measurement"]["n_segments"] = 8          # enough averaging for Q variance
    cfg["stop_criteria"]["uncertainty_target"] = 0.08
    return cfg


@pytest.mark.parametrize("pf,pq,pg", [(1.5, 30, 150), (0.5, 10, 50), (1.3, 12, 70)])
def test_hybrid_spectrum_refine_recovers_f0_Q_gain(pf, pq, pg):
    """Spectrum refine: broadband_ls locates, then the -3 dB bandwidth nails Q.

    Recovers all three parameters prior-independently and unbiased, where the
    Bayesian/LS refine left Q biased. The Welch segment is auto-sized from the
    prior so the peak is resolved.
    """
    rc, be, result = _run(_cfg_spectrum(pf, pq, pg))
    model = result.models["POS"]
    assert isinstance(model, ResonatorModel)
    f0, Q, gain = float(model.f0[0]), float(model.Q[0]), float(model.gain)
    assert abs(f0 - 1.0) < 0.03, f"f0={f0:.3f}"
    assert abs(Q - 20.0) / 20.0 < 0.25, f"Q={Q:.1f}"
    assert abs(gain - 100.0) / 100.0 < 0.25, f"gain={gain:.0f}"


def test_hybrid_spectrum_rejects_multimode_prior():
    """Spectrum refine is single-resonance; a multi-mode prior fails loudly."""
    cfg = _cfg_spectrum(1.0, 20, 100)
    cfg["priors"]["POS"]["resonances"] = [[0.6, 15], [1.8, 25]]   # two modes
    cfg["twin"]["plant"]["POS"]["resonances"] = [[0.6, 15], [1.8, 25]]
    with pytest.raises(ValueError, match="single-resonance"):
        _run(cfg)


def test_hybrid_spectrum_segment_autosized_for_resolution():
    """The spectrum refine lengthens the Welch segment to resolve the peak."""
    from system_ident.loop import _nperseg_for_resolution
    from system_ident.resonator import ResonatorModel
    fs = 32.0
    priors = {"POS": ResonatorModel.from_resonances([(1.0, 20.0)], 100.0)}
    nperseg = _nperseg_for_resolution(priors, fs)      # 6 bins * 2x Q safety
    df = fs / nperseg
    bandwidth = 1.0 / 20.0                              # f0/Q = 0.05 Hz
    assert bandwidth / df >= 6.0, f"peak spans only {bandwidth/df:.1f} bins"


def test_hybrid_builds_tfmodel_priors():
    """Hybrid starts in broadband_ls, so config priors are TFModels (phase 1)."""
    rc = RunConfig(raw=_cfg(1.2, 24, 120))
    priors = rc.build_priors()
    assert isinstance(priors["POS"], TFModel)


def test_dashboard_snapshot_accepts_resonator_fields():
    """The refine phase emits model_f0/Q/gain; the dashboard schema allows them."""
    from system_ident.dashboard.ws import validate_snapshot, SNAPSHOT_FIELDS
    snap = {k: 0 for k in SNAPSHOT_FIELDS}
    snap.update(model_f0=1.0, model_Q=20.0, model_gain=100.0)
    assert validate_snapshot(snap) is snap          # optional fields allowed
    with pytest.raises(ValueError):
        validate_snapshot({**snap, "bogus_field": 1})  # genuinely unexpected still rejected


def test_hybrid_with_no_refine_passes_is_broadband_ls():
    """If n_locate >= max_iter the hybrid never hands off -> stays broadband_ls."""
    rc, be, result = _run(_cfg(1.5, 30, 150, n_locate=5, max_iter=4))
    # never transitioned -> still a TFModel, but broadband_ls already converges it
    model = result.models["POS"]
    assert isinstance(model, TFModel)
    p = np.roots(model.den / model.den[0])
    p = p[p.imag > 0][0]
    f0 = abs(p) / (2 * np.pi)
    assert abs(f0 - 1.0) < 0.05, f"broadband_ls f0={f0:.3f}"
