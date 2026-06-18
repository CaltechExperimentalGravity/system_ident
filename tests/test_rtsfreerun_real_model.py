"""Track A validation gate against the *real* compiled twin model (``x1hsts``).

Skipped unless the rtsfreerun ``x1hsts`` model is built into this env (see README
§"Run against the RTSfreerun digital twin"). These are the honest recovery checks
the demo rests on — the basic Pintelon–Schoukens path (periodic multisine →
leakage-free reference FRF → ML fit) driving the compiled CDS suspension under the
twin's own seismic + readout noise, scored against the analytic oracle.

A1 (wiring + oracle): the scenario init realises the plant, the analytic oracle
agrees with the model's loaded SOS, and a noise-off FRF through the adapter matches
the oracle (exact lengths, integer decimation).

A2 (open-loop SISO recovery): with the twin's noise on, recover the HSTS
drive→sensor plant (the order-10 ``HSTS_DRV_TF`` cascade, 5 modes ~0.67–3.78 Hz
each Q≈50) and match the oracle in FRF and modal frequencies.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from system_ident.backends import rtsfreerun_oracle as orc
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
from system_ident.estimators.gml import GMLEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("x1hsts") is None,
    reason="rtsfreerun x1hsts model not installed (see README: build it into the sysid env)")

# The twin scenario the model was built from. Path is machine-specific; skip if absent.
SCENARIO = Path("/Users/rana/Desktop/Dropbox/GIT/digital_twin/twin/scenarios/hsts.yaml")

EXC, DRIVE_MON, SENSOR = "COIL_DRIVER_EXC", "COIL_DRIVER_OUT", "READOUT_NOISE_OUT"
FS, NPER, NPERIODS = 256.0, 4096, 4
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
    return x1hsts_model, scen, orc.analytic_plant(scen)


def _measure(mdl, noise, *, amp, seed, warmup_s):
    band, freq = _band()
    be = RTSfreerunBackend(mdl=mdl, exc_channels={EXC: "POS"},
                           readback_channels={SENSOR: "POS"},
                           noise=noise, fs=FS, warmup_s=warmup_s, seed=seed)
    drive = multisine_from_psd(np.ones_like(freq), FS, NPER, NPERIODS, freq, seed=seed)
    drive = drive / np.max(np.abs(drive)) * amp
    be.inject(EXC, drive, FS)
    seg = be.read([SENSOR, DRIVE_MON], NPER * NPERIODS / FS)
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(seg[DRIVE_MON], seg[SENSOR], FS, NPER, band)
    return seg, freq, H, H_err, coh


# ── A1 — wiring + oracle ──────────────────────────────────────────────────────
def test_a1_oracle_matches_realized_sos(twin):
    """Analytic oracle (yaml ZPK) == the model's actually-loaded SOS."""
    mdl, scen, oracle = twin
    _, freq = _band()
    Hy = oracle.eval(freq)
    Hs = orc.realized_plant_response(mdl, freq, scen)
    rel = np.abs(Hs - Hy) / np.maximum(np.abs(Hy), 1e-30)
    assert rel.max() < 1e-3                       # two independent oracles agree


def test_a1_rate_is_integer_decimation(twin):
    mdl, _, _ = twin
    assert mdl.sample_rate == 16384
    assert np.isclose(mdl.sample_rate / FS, 64.0)  # clean integer decimation


def test_a1_noise_off_frf_matches_oracle(twin):
    """Plumbing + routing: a noise-off FRF through the adapter == the oracle."""
    mdl, scen, oracle = twin
    seg, freq, H, H_err, coh = _measure(mdl, [], amp=2000.0, seed=0, warmup_s=24.0)
    assert seg[SENSOR].size == NPER * NPERIODS    # exact length after decimation
    Ho = oracle.eval(freq)
    rel = np.abs(np.abs(H) - np.abs(Ho)) / np.abs(Ho)
    assert np.median(coh) > 0.99
    assert np.median(rel) < 0.03 and np.percentile(rel, 90) < 0.06


# ── A2 — open-loop SISO recovery under realistic noise ────────────────────────
@pytest.mark.parametrize("seed", [0, 1])
def test_a2_recovers_plant_under_noise(twin, seed):
    """Recover the HSTS drive→sensor plant with seismic + readout noise on."""
    mdl, scen, oracle = twin
    seg, freq, H, H_err, coh = _measure(mdl, NOISE, amp=6000.0, seed=seed, warmup_s=24.0)
    assert np.max(np.abs(seg[DRIVE_MON])) < 30000.0           # under COIL_DRIVER_LIMIT
    assert np.median(coh) > 0.95

    prior = orc.prior_from_scenario(scen, perturb=0.12, rng=np.random.default_rng(100 + seed))
    fit = GMLEstimator().fit(freq, H, H_err, prior)

    ff = np.geomspace(0.3, 8.0, 300)
    rel = np.abs(fit.eval(ff) - oracle.eval(ff)) / np.abs(oracle.eval(ff))
    assert np.median(rel) < 0.03 and np.percentile(rel, 90) < 0.10

    # every oracle mode is matched by a recovered pole within 3 %
    rec = [f for f, _ in orc.plant_modes(fit)]
    for f0, _ in orc.plant_modes(oracle):
        assert any(abs(f - f0) / f0 < 0.03 for f in rec), f"missed mode {f0:.3f} Hz"
