"""Integration guard for the CLOSED-LOOP reduced-QUAD demo (example 12).

The reduced aLIGO quad yaw chain closed around velocity dampers (python-control time-domain
simulation), identified through the live loops: the reference-based FRF cancels the controller
and the rank-1 fit recovers the open-loop modes. Guards that the loop is stable and the modes
come back to a part in ~100 *through the dampers*. See docs/reduced_quad_closed_demo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import reduced_quad_closed_demo as rc  # noqa: E402


def test_loop_is_stable():
    """The reduced quad closed around velocity dampers is a stable loop (all Sd poles inside
    the unit circle) — otherwise the identification is meaningless."""
    assert rc.run().stable is True


def test_modes_recovered_through_the_loops():
    """The reference-based FRF cancels the dampers: the four yaw modes come back near their
    open-loop truth (f0 to <1.5%) through the closed loops."""
    d = rc.run()
    assert len(d.matched) == 4
    f_err = [abs(u["f0"] - tf) / tf for tf, tq, u in d.matched]
    assert max(f_err) < 1.5e-2, f"max f0 error through loops = {max(f_err):.2e}"


def test_recovered_are_the_yaw_frequencies():
    d = rc.run()
    f0 = sorted(tf for tf, tq, u in d.matched)
    for expect in (0.599, 1.349, 2.391, 3.036):
        assert any(abs(f - expect) < 0.02 for f in f0), f"missing yaw mode near {expect} Hz"
