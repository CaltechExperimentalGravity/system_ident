"""Build the committed 40m SOS 6-DOF plant artifact (``sos_6dof.{npz,json}``).

Unlike :mod:`system_ident.models.regenerate` — which shells to the twin's modal-truncation
reducer and needs the ``aligo-suspension-models`` ``.mat`` files — this builder is fully
analytic and self-contained: it depends only on :mod:`system_ident.sos_plant`. Nothing
outside this repo is required to reproduce the artifact.

    conda run -n sysid python -m system_ident.models.build_sos

Channel labels follow the twin's single-mass convention (``m1.drive.<DOF>`` /
``m1.disp.<DOF>``), so consumers address channels by name rather than by position. That
matters here: the twin uses **two** different DOF orderings — ``L T V R P Y`` in
``scripts/gen_x1hsts6dof.py`` and ``L T V Y P R`` in the ``.mat``-derived sidecar labels —
so positional assumptions are unsafe. This plant is built in ``L T V R P Y`` order and every
channel carries its DOF in the label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from system_ident.sos_plant import (
    DOFS, MEASURED_MODES_HZ, DEFAULT_STRUCTURAL_Q,
    SOSOptic, build_plant, fit_suspension, modes_hz,
)

HERE = Path(__file__).resolve().parent
STEM = "sos_6dof"

#: Above the highest SOS mode (roll, ~22.6 Hz). This plant is NOT modally truncated — the
#: field exists for the ReducedStateSpacePlant dataclass; it is not a reduction cutoff.
F_MODE_CUT_HZ = 50.0


def build() -> tuple[dict, dict]:
    optic = SOSOptic()
    susp = fit_suspension(optic)
    sys = build_plant(optic, susp, structural_q=DEFAULT_STRUCTURAL_Q)

    A, B, C, D = (np.asarray(x, float) for x in (sys.A, sys.B, sys.C, sys.D))
    arrays = dict(A=A, B=B, C=C, D=D, f_mode_cut=F_MODE_CUT_HZ)

    sidecar = {
        "inputs": [f"m1.drive.{d}" for d in DOFS],
        "outputs": [f"m1.disp.{d}" for d in DOFS],
        "modes": [[f, q] for f, q in modes_hz(sys)],
        "provenance": {
            "sus_type": "sos",
            "source": "analytic 6-DOF rigid-body-on-wires (system_ident.sos_plant)",
            "method": ("optic inertia from 40m SOS geometry "
                       f"(R={optic.radius_m} m, H={optic.thickness_m} m, "
                       f"rho={optic.density_kg_m3} kg/m^3 -> m={optic.mass_kg:.4f} kg); "
                       "suspension parameters constrained to the measured 40m eigenmodes"),
            "dof_order": list(DOFS),
            "n_states": int(A.shape[0]),
            "structural_q": DEFAULT_STRUCTURAL_Q,
            "f_mode_cut_hz": F_MODE_CUT_HZ,
            "measured_modes_hz": dict(MEASURED_MODES_HZ),
            "fitted_suspension": {
                "wire_length_m": susp.wire_length_m,
                "breakoff_m": susp.breakoff_m,
                "wire_sep_m": susp.wire_sep_m,
                "r1r2_m2": susp.r1r2_m2,
                "k_vert_N_per_m": susp.k_vert_N_per_m,
            },
            "fitted_dofs": ["L", "P", "V", "R", "Y"],
            "predicted_dofs": ["T"],
            "note": (
                "L/P/V/R/Y are constrained to the measured 40m modes, so their agreement is "
                "by construction and is not evidence. T (side) is NOT fitted: omega_T^2 = "
                "g/(l+b) follows from the pinned l and b, and lands within ~0.2% of the "
                "measured 1 Hz. Structural Q=50 is the repo-wide convention "
                "(sus_modal.DEFAULT_STRUCTURAL_Q), not a 40m measurement. This plant is "
                "analytic and NOT modally truncated; f_mode_cut is a dataclass field, not a "
                "reduction cutoff. The measured roll value is recorded as 16*sqrt(2) Hz, "
                "which may itself be derived rather than independently measured — the "
                "fitted wire separation coming out at exactly 2R is equivalent to that "
                "ratio, so treat it as a consistency check, not independent confirmation. "
                "V and R have no OSEM actuation path (see system_ident.osem)."
            ),
        },
    }
    return arrays, sidecar


def main() -> None:
    arrays, sidecar = build()
    np.savez(HERE / f"{STEM}.npz", **arrays)
    (HERE / f"{STEM}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    n_modes = len(sidecar["modes"])
    print(f"wrote {STEM}.npz ({arrays['A'].shape[0]} states) + .json ({n_modes} modes)")
    for f0, q in sidecar["modes"]:
        print(f"    {f0:9.4f} Hz   Q={q:.1f}")


if __name__ == "__main__":
    main()
