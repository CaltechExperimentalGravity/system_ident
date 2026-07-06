"""Portable SRM validation on the reduced HSTS (reduced-plants phase 3).

The compiled-twin SRM demo (docs/examples/10) identifies the L1-SRM HSTS through its real
production loops and resolves the 0.672/0.676 Hz spatial doublet by fitting the orthogonal
{L,P,V} / {T,R,Y} planes (``fit_block_decoupled``). This test reproduces that key result on the
committed **reduced HSTS** — numpy-only, no compiled twin — establishing ``hsts_reduced`` as a
portable oracle for the HSTS 6-DOF modal fit: its modes ARE the SRM physics, and the spatial
doublet resolves the same way. See docs/reduced_quad_demo.py for the sibling QUAD demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from system_ident.reduced_plant import ReducedStateSpacePlant  # noqa: E402
from system_ident.backends.reduced import ReducedPlantBackend  # noqa: E402
from system_ident.mimo_campaign import assemble_campaign  # noqa: E402
from system_ident.mimo_fit import fit_block_decoupled  # noqa: E402

DOFS = ("L", "T", "V", "R", "P", "Y")


def test_reduced_hsts_modes_are_the_srm_physics():
    """The reduced HSTS oracle carries the exact SRM modes: the 0.672/0.676 spatial doublet,
    the 1.512/1.516/1.527 triplet, and structural Q=50 throughout."""
    modes = ReducedStateSpacePlant.load("hsts").modes()
    f0 = sorted(f for f, q in modes)
    assert any(abs(f - 0.6725) < 1e-3 for f in f0)
    assert any(abs(f - 0.6758) < 1e-3 for f in f0)
    assert sum(1 for f in f0 if 1.50 < f < 1.53) == 3
    assert all(abs(q - 50.0) < 1e-6 for f, q in modes)


def test_spatial_doublet_resolves_by_plane_decoupling():
    """Drive the reduced-HSTS top mass and resolve the 0.672/0.676 doublet the way example 10
    does — one rank-1 fit per orthogonal DOF plane. Each plane sees ONE mode near 0.674, so
    both members come back on the oracle (no frequency super-resolution)."""
    p = ReducedStateSpacePlant.load("hsts")
    acts = [f"m1.drive.{d}" for d in DOFS]
    sens = [f"m1.disp.{d}" for d in DOFS]
    sub = p.subplant(sensors=sens, actuators=acts)
    N = 6
    gscale = float(np.median(np.abs(sub.eval(np.linspace(0.55, 0.75, 50)))))
    be = ReducedPlantBackend(
        sub, {f"E{j}": acts[j] for j in range(N)}, {f"S{i}": sens[i] for i in range(N)},
        fs=32.0, sensor_asd=gscale * 1e-3, seed=7)
    f = np.fft.rfftfreq(4096, 1 / 32.0)
    band = np.flatnonzero((f >= 0.55) & (f <= 0.80))       # focused on the fundamental doublet
    lines = band[:: max(1, len(band) // 60)]
    psd = np.zeros(len(f)); psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"E{j}" for j in range(N)], [f"E{j}" for j in range(N)],
        [f"S{i}" for i in range(N)], f[lines], fs=32.0, nperseg=4096, n_periods=10,
        drive_psd=psd, n_transient=1, seed=7)

    iA = [DOFS.index(d) for d in ("L", "P", "V")]           # the two orthogonal HSTS planes
    iB = [DOFS.index(d) for d in ("T", "R", "Y")]
    blocks = [{"sensors": iA, "actuators": iA, "modes": [(0.6725, 50.0)]},
              {"sensors": iB, "actuators": iB, "modes": [(0.6758, 50.0)]}]
    res = fit_block_decoupled(exps, freq, blocks, dof=10 - 1)
    fA, fB = res[0]["mu"][0]["f0"], res[1]["mu"][0]["f0"]
    qA, qB = res[0]["mu"][0]["Q"], res[1]["mu"][0]["Q"]
    # each plane recovers its own fundamental near the oracle (well within 1%)
    assert abs(fA - 0.6725) < 4e-3, f"{{L,P,V}} plane f0 = {fA}"
    assert abs(fB - 0.6758) < 4e-3, f"{{T,R,Y}} plane f0 = {fB}"
    # the doublet is RESOLVED: the two planes give two distinct modes (not one collapsed
    # pole), and {T,R,Y} sits above {L,P,V} — the real spatial split, no super-resolution.
    # (Q is only loosely pinned by this deliberately fast/narrow campaign — the exact Q=50
    # SRM physics is asserted from the oracle in the modes test above; here both are genuine
    # underdamped resonances.)
    assert fB - fA > 1.5e-3, f"doublet not resolved: fA={fA}, fB={fB}"
    assert qA > 15.0 and qB > 15.0, f"planes not resonant: qA={qA}, qB={qB}"
