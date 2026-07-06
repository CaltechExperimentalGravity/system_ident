"""Fidelity check: the reduced HSTS vs the compiled rtsfreerun twin (reduced-plants phase 5).

LOCAL-ONLY, twin-gated (skipped in CI, like ``test_rtsfreerun_real_model.py``). The committed
reduced HSTS is a modal truncation of ``hsts_full.mat``; the compiled ``x1hsts6dof`` CDS twin is
built from the same ``.mat``. This test confirms the reduced model reproduces the compiled twin's
in-band modal physics — the final validation that the portable numpy plant used in examples 11 /
the phase-3 HSTS test is faithful to the real front-end numerics. See notes/twin-fidelity-ledger.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "rtsfreerun"))
sys.path.insert(0, str(ROOT / "docs"))

_HAVE_TWIN = importlib.util.find_spec("x1hsts6dof") is not None

pytestmark = pytest.mark.skipif(
    not _HAVE_TWIN, reason="compiled x1hsts6dof twin not built into this env (local-only check)")


def _compiled_twin_modes(band=(0.3, 4.0)):
    import srm6dof_loop as s6                       # imports the twin's lib (needs the twin)
    m6 = s6.SRM6DOF()
    s = np.log(np.linalg.eigvals(m6.Ad)) * m6.fs_model   # discrete -> continuous eigenvalues
    return sorted(abs(l) / (2 * np.pi) for l in s
                  if l.imag > 1e-6 and band[0] < abs(l) / (2 * np.pi) < band[1])


def test_reduced_hsts_modes_match_the_compiled_twin():
    """Every in-band reduced-HSTS mode coincides with a compiled-twin (x1hsts6dof) mode —
    the reduced numpy plant is faithful to the real CDS front-end's modal physics."""
    from system_ident.reduced_plant import ReducedStateSpacePlant
    twin = _compiled_twin_modes()
    reduced = sorted(f for f, q in ReducedStateSpacePlant.load("hsts").modes()
                     if 0.3 < f < 4.0)
    assert len(reduced) == len(twin), (len(reduced), len(twin))
    max_rel = max(min(abs(f - t) for t in twin) / f for f in reduced)
    assert max_rel < 1e-6, f"reduced vs compiled-twin mode mismatch = {max_rel:.2e}"
