"""Track A validation gate against the *real* compiled twin model (``x1hsts``).

Skipped unless the rtsfreerun ``x1hsts`` model is built into this env (see README
§"Run against the RTSfreerun digital twin"). These are the honest recovery checks
the demo rests on — the Pintelon-Schoukens **optimal-excitation** campaign (broad
prior-robust multisine from the perturbed prior, then point-optimal refinement;
each pass a leakage-free reference FRF folded in by inverse-variance accumulation +
ML refit — the same `run_siso_passes` the double-pendulum example runs) driving the
compiled CDS suspension under the twin's own seismic + readout noise, scored against
the analytic oracle.

A1 (wiring + oracle): the scenario init realises the plant, the analytic oracle
agrees with the model's loaded SOS (exact rate / integer decimation), and a
noise-off P&S pass recovers it.

A2 (open-loop SISO recovery): with the twin's noise on, the optimal-excitation
campaign recovers the HSTS drive→sensor plant (the order-10 ``HSTS_DRV_TF`` cascade,
5 modes ~0.67–3.78 Hz each Q≈50), the fractional uncertainty falls pass over pass,
and the recovered plant matches the oracle in FRF and modal frequencies.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from system_ident.backends import rtsfreerun_oracle as orc
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
from sysid_campaign import run_siso_passes  # noqa: E402  (the P&S campaign the examples use)

# The twin scenario the model was built from. Path is machine-specific; skip if absent.
SCENARIO = Path("/Users/rana/Desktop/Dropbox/GIT/digital_twin/twin/scenarios/hsts.yaml")

EXC, DRIVE_MON, SENSOR = "COIL_DRIVER_EXC", "COIL_DRIVER_OUT", "READOUT_NOISE_OUT"
FS, NPER, NPERIODS = 256.0, 4096, 6
PX_TOTAL = 1.0e7
NOISE = [{"channel": "ISI_RESIDUAL_EXC", "kind": "seismic", "params": {"preset": "ligo-india"}},
         {"channel": "READOUT_NOISE_EXC", "kind": "bosem", "params": {"floor": 1e-10}}]


def _band():
    fa = np.fft.rfftfreq(NPER, 1 / FS)
    band = (fa >= 0.3) & (fa <= 8.0)
    return band, fa[band]


@pytest.fixture(scope="module")
def twin(x1hsts_model):
    """The shared model (one per process) with the HSTS scenario plant loaded."""
    if not SCENARIO.exists():
        pytest.skip(f"twin scenario not found at {SCENARIO}")
    scen = orc.load_scenario(SCENARIO)
    orc.apply_scenario_init(x1hsts_model, scen)
    modules = sorted({op["fm"] for op in scen.get("init", []) if "fm" in op})
    return x1hsts_model, scen, orc.analytic_plant(scen), modules


def _campaign(twin, noise, *, seed, n_passes):
    """Run the P&S optimal-excitation campaign from a perturbed prior, clean state."""
    mdl, scen, _, modules = twin
    band, freq = _band()
    mdl.fm_clear_history(*modules)              # clean filter state (one model/process)
    be = RTSfreerunBackend(mdl=mdl, exc_channels={EXC: "POS"},
                           readback_channels={SENSOR: "POS"},
                           noise=noise, fs=FS, warmup_s=24.0, seed=seed)
    prior = orc.prior_from_scenario(scen, perturb=0.08, rng=np.random.default_rng(7))
    return run_siso_passes(be, EXC, SENSOR, prior, x_ch=DRIVE_MON, fs=FS, nperseg=NPER,
                           n_periods=NPERIODS, band=band, freq=freq, Pyy=np.ones_like(freq),
                           px_total=PX_TOTAL, n_passes=n_passes, prior_uncertainty=0.6, seed=seed)


# ── A1 — wiring + oracle ──────────────────────────────────────────────────────
def test_a1_oracle_matches_realized_sos(twin):
    """Analytic oracle (yaml ZPK) == the model's actually-loaded SOS."""
    mdl, scen, oracle, _ = twin
    _, freq = _band()
    Hy, Hs = oracle.eval(freq), orc.realized_plant_response(mdl, freq, scen)
    assert (np.abs(Hs - Hy) / np.maximum(np.abs(Hy), 1e-30)).max() < 1e-3


def test_a1_rate_is_integer_decimation(twin):
    mdl = twin[0]
    assert mdl.sample_rate == 16384 and np.isclose(mdl.sample_rate / FS, 64.0)


def test_a1_noise_off_recovers_oracle(twin):
    """Plumbing + routing: a noise-off P&S pass recovers the oracle exactly."""
    _, _, oracle, _ = twin
    h = _campaign(twin, [], seed=0, n_passes=1)[-1]
    assert h["response"].size == NPER * NPERIODS          # exact length after decimation
    ff = np.geomspace(0.3, 8.0, 300)
    rel = np.abs(h["model"].eval(ff) - oracle.eval(ff)) / np.abs(oracle.eval(ff))
    assert np.median(rel) < 0.02 and np.percentile(rel, 90) < 0.06


# ── A2 — open-loop SISO recovery under realistic noise ────────────────────────
@pytest.mark.parametrize("seed", [0, 1])
def test_a2_recovers_plant_under_noise(twin, seed):
    """Optimal-excitation recovery of the HSTS plant with seismic + readout noise."""
    _, _, oracle, _ = twin
    hist = _campaign(twin, NOISE, seed=seed, n_passes=3)

    assert np.max(np.abs(hist[-1]["drive"])) < 30000.0        # under COIL_DRIVER_LIMIT
    assert hist[-1]["frac"] < hist[0]["frac"]                 # P&S refinement reduces uncertainty

    fit = hist[-1]["model"]
    ff = np.geomspace(0.3, 8.0, 300)
    rel = np.abs(fit.eval(ff) - oracle.eval(ff)) / np.abs(oracle.eval(ff))
    assert np.median(rel) < 0.03 and np.percentile(rel, 90) < 0.10

    rec = [f for f, _ in orc.plant_modes(fit)]
    for f0, _ in orc.plant_modes(oracle):
        assert any(abs(f - f0) / f0 < 0.03 for f in rec), f"missed mode {f0:.3f} Hz"
