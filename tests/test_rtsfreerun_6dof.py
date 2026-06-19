"""Track A3+A4 validation gate — real closed-loop 6-DOF HSTS on ``x1hsts6dof``.

Drives the compiled ``x1hsts6dof`` composite the way the digital-twin's canonical
example does: the bare-M1 6×6 MIMO HSTS plant with the **real production L1-MC2
top-mass dampers** closed around all six DOFs. Skipped unless the model + twin
example + production filter/plant archives are all present (see
``experiments/rtsfreerun/hsts6dof_loop.py``), so the suite stays green elsewhere.

**A4 (open-loop MIMO tensor).** With the loops open, the leakage-free reference-FRF
tensor ``READOUT_i / DRIVE_EXC_j`` matches the analytic state-space oracle
(``orc.state_space_frf`` of the discretised plant): diagonal anti-resonances and the
dominant L↔P / R↔Y cross-couplings come back per-pair.

**A3 (closed-loop diagonal recovery).** With all six real dampers engaged, the
reference-based FRF ``READOUT_d / PLANT_IN_d`` — where ``PLANT_IN`` is the true
plant input ``DRIVE_EXC − damper_feedback`` (the ``"+-"`` ``COIL_DRV_SUM`` junction) —
recovers the **open-loop** plant diagonal (controller cancelled), not the suppressed
closed-loop response. The feedback sign is load-bearing: it is negligible
off-resonance but dominates at the damped modes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "rtsfreerun"))
import hsts6dof_loop as h6  # noqa: E402

pytestmark = pytest.mark.skipif(
    not h6.deps_available(),
    reason="x1hsts6dof model / twin example / production archives not present",
)

FS, NPERSEG, NPERIODS, NPASSES = 256.0, 4096, 4, 2


def _grid():
    fa = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (fa >= 0.3) & (fa <= 8.0)
    return band, fa[band]


@pytest.fixture(scope="module")
def model():
    return h6.HSTS6DOF()


@pytest.fixture(scope="module")
def tensors(model):
    """Open- and closed-loop FRF tensors + the analytic oracle, measured once."""
    band, freq = _grid()
    kw = dict(fs=FS, nperseg=NPERSEG, n_periods=NPERIODS, band=band, freq=freq,
              n_passes=NPASSES, warmup_s=16.0, seed=0)
    H_open = model.measure_tensor(closed=False, **kw)
    H_closed = model.measure_tensor(closed=True, **kw)
    return model, freq, H_open, H_closed


def _idx(model, d):
    return model.dofs.index(d)


# ── A4 — open-loop MIMO tensor ────────────────────────────────────────────────
def test_a4_open_loop_diagonal_recovers(tensors):
    """Every diagonal drive→sense element matches the SS oracle (anti-resonances)."""
    model, freq, H_open, _ = tensors
    M = model.rel_err_tensor(H_open, freq)
    diag = np.diag(M)
    assert np.all(diag < 0.01), dict(zip(model.dofs, np.round(diag, 4)))


@pytest.mark.parametrize("out_dof,in_dof", [("L", "P"), ("P", "L"), ("R", "Y"), ("Y", "R")])
def test_a4_open_loop_dominant_couplings(tensors, out_dof, in_dof):
    """The physically dominant L↔P and R↔Y off-diagonal couplings come back per-pair."""
    model, freq, H_open, _ = tensors
    i, j = _idx(model, out_dof), _idx(model, in_dof)
    G = model.oracle_tensor(freq)[:, i, j]
    rel = np.median(np.abs(H_open[i, j] - G) / np.abs(G))
    assert rel < 0.02, f"{out_dof}<-{in_dof} rel-err {rel:.4f}"


# ── A3 — closed-loop diagonal recovery (controller cancelled) ─────────────────
def test_a3_closed_loop_recovers_open_loop_plant(tensors):
    """All six real loops closed: the reference FRF returns the *open-loop* diagonal."""
    model, freq, _, H_closed = tensors
    M = model.rel_err_tensor(H_closed, freq)
    diag = np.diag(M)
    assert np.all(diag < 0.01), dict(zip(model.dofs, np.round(diag, 4)))


def test_a3_parametric_campaign_recovers_all_dofs(model):
    """The A2-style optimal-excitation parametric campaign recovers every DoF's
    closed-loop diagonal — including pitch/yaw, which over-parameterise (V has 2
    modes, Y has 3) and used to crash the optimal-excitation design with an SVD
    failure until the per-DoF prior order + the excitation floor landed."""
    band, freq = _grid()
    for dof in model.dofs:
        hist = model.parametric_recovery(dof, fs=FS, nperseg=NPERSEG, n_periods=NPERIODS,
                                         band=band, freq=freq, n_passes=3, warmup_s=16.0)
        fit = hist[-1]["model"]
        G = model.oracle_tensor(freq)[:, _idx(model, dof), _idx(model, dof)]
        rel = float(np.median(np.abs(fit.eval(freq) - G) / np.abs(G)))
        assert rel < 0.02, f"{dof}: parametric campaign median rel-err {rel:.4f}"
        assert hist[-1]["frac"] <= hist[0]["frac"]            # CRB uncertainty does not grow


def test_a3_feedback_sign_matters(tensors, model):
    """Sanity that A3 is a real closed-loop test: with the wrong feedback sign
    (drive **+** feedback), the damped-resonance recovery is badly biased — so the
    clean recovery above is genuinely cancelling an active loop, not a no-op."""
    band, freq = _grid()
    j = _idx(model, "L")
    G = model.oracle_tensor(freq)[:, j, j]
    pk = np.argsort(np.abs(G))[-8:]                       # the resonance peak bins

    from system_ident.excitation import multisine_from_psd
    from system_ident.loop import SysIDLoop
    from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend

    def peak_relerr(coeff):
        model.set_loops(True); model.reset()
        be = RTSfreerunBackend(
            mdl=model.mdl, exc_channels={model.exc(d): d for d in model.dofs},
            readback_channels={model.readout("L"): "L"},
            plant_inputs={model.plant_in("L"): {"exc": model.exc("L"),
                          "feedback": [model.damp_out("L")], "feedback_coeff": coeff}},
            fs=FS, warmup_s=16.0, seed=0)
        acc = dict(w=np.zeros(len(freq)), wH=np.zeros(len(freq), dtype=complex))
        Pxx = np.full(len(freq), 1.0e7 / (freq[-1] - freq[0]))
        H = None
        for p in range(NPASSES):
            dr = multisine_from_psd(Pxx, FS, NPERSEG, NPERIODS, freq,
                                    seed=np.random.default_rng(p))
            be.inject(model.exc("L"), dr, FS)
            seg = be.read([model.plant_in("L"), model.readout("L")], NPERSEG * NPERIODS / FS)
            be.inject(model.exc("L"), np.zeros_like(dr), FS)
            hh, ee, _ = SysIDLoop._estimate_tf_periodic(seg[model.plant_in("L")],
                                                        seg[model.readout("L")], FS, NPERSEG, band)
            H, _ = SysIDLoop._accumulate(acc, hh, ee)
        return float(np.median(np.abs(H[pk] - G[pk]) / np.abs(G[pk])))

    right = peak_relerr(-1.0)     # the "+-" junction sign
    wrong = peak_relerr(+1.0)
    assert right < 0.05, f"correct-sign peak recovery {right:.4f}"
    assert wrong > 5 * right, f"wrong sign {wrong:.4f} not clearly worse than {right:.4f}"
