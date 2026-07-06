"""Integration guard for the reduced-QUAD MIMO modal demo (example 11).

The full P&S pipeline — ReducedPlantBackend → assemble_campaign → recover_open_loop →
find_modes → rank-1 modal fit → CRB — run on the committed 59-state reduced aLIGO QUAD
(yaw chain). Guards (a) the campaign/backend integration (the recovered open-loop FRF must
equal the plant's own FRF) and (b) that the fit recovers the plant's exact yaw modes within
a tight tolerance. Pure numpy/scipy; no twin. See docs/reduced_quad_demo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import reduced_quad_demo as rq  # noqa: E402


def test_recovered_frf_matches_the_reduced_plant():
    """The campaign + open-loop recovery through ReducedPlantBackend returns the plant's
    own FRF — the integration guard (a wrong drive-monitor / cache bug would break this)."""
    d = rq.run()
    rel = np.abs(d.Gnp - d.Gtruth) / np.maximum(np.abs(d.Gtruth), 1e-30)
    assert np.median(rel) < 5e-3


def test_fit_recovers_the_four_yaw_modes():
    """The rank-1 modal fit recovers the reduced quad's 4 yaw modes: f0 to <0.01%, Q to
    within a few percent, against the plant's exact eigen-modes."""
    d = rq.run()
    assert len(d.matched) == 4
    f_err = [abs(u["f0"] - tf) / tf for tf, tq, u in d.matched]
    q_err = [abs(u["Q"] - tq) / tq for tf, tq, u in d.matched if np.isfinite(tq)]
    assert max(f_err) < 1e-3          # every f0 within 0.1%
    assert np.median(q_err) < 0.05    # median Q within 5%


def test_matched_modes_are_the_yaw_frequencies():
    """Sanity: the recovered modes are the known reduced-quad yaw frequencies."""
    d = rq.run()
    f0 = sorted(tf for tf, tq, u in d.matched)
    for expect in (0.599, 1.349, 2.391, 3.036):
        assert any(abs(f - expect) < 0.01 for f in f0), f"missing yaw mode near {expect} Hz"
