"""Regenerate the committed reduced suspension models from the digital twin.

LOCAL-ONLY, not imported at runtime: shells to the twin's modal-truncation reducer
(twin/src/twin/sus_modal.py) and writes the committed .npz + .json here. Requires the
twin checkout at $DIGITAL_TWIN_DIR (default ~/GIT/digital_twin) with the full .mat models.

    conda run -n sysid python -m system_ident.models.regenerate
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
CUTOFF_HZ = 50.0


def _modes_from_A(A: np.ndarray, f_c: float) -> list[tuple[float, float]]:
    """Oracle modes: conjugate-pair eigenvalues with 1e-6 < f0 <= 1.05*f_c (a 5% margin
    keeps the boundary mode just above the cutoff — e.g. the QUAD's 50.37 Hz mode — while
    still dropping the spurious >>f_c eigenvalues the prescaled realization shows). Returns
    sorted (f0, Q)."""
    lam = sla.eig(np.asarray(A, float), right=False)
    out = []
    for z in lam:
        if z.imag <= 1e-6:
            continue
        f0 = abs(z) / (2 * np.pi)
        if f0 > f_c * 1.05:
            continue
        Q = abs(z) / (-2 * z.real) if z.real < 0 else float("inf")
        out.append((float(f0), float(Q)))
    return sorted(out)


def _twin_src() -> Path:
    twin = Path(os.environ.get("DIGITAL_TWIN_DIR") or Path.home() / "GIT" / "digital_twin")
    src = twin / "twin" / "src"
    if not (src / "twin" / "sus_modal.py").exists():
        sys.exit(f"twin reducer not found under {src} — set $DIGITAL_TWIN_DIR")
    return src


def regenerate(sus_type: str) -> None:
    sys.path.insert(0, str(_twin_src()))
    from twin.sus_modal import compute_reduction  # noqa: E402  (local-only)
    rp = compute_reduction(sus_type, cutoff_hz=CUTOFF_HZ)
    A, B, C, D = (np.asarray(x, float) for x in (rp.A, rp.B, rp.C, rp.D))
    modes = _modes_from_A(A, CUTOFF_HZ)
    base = HERE / f"{sus_type}_reduced_50hz"
    np.savez(base.with_suffix(".npz"), A=A, B=B, C=C, D=D, f_mode_cut=CUTOFF_HZ)
    sidecar = {
        "inputs": list(rp.inputs),
        "outputs": list(rp.outputs),
        "modes": [[f, q] for f, q in modes],
        "provenance": {
            "sus_type": sus_type,
            "source": f"aligo-suspension-models/{sus_type}_full.mat",
            "method": "modal truncation (twin/src/twin/sus_modal.py::compute_reduction)",
            "cutoff_hz": CUTOFF_HZ,
            "n_full": int(rp.n_full),
            "n_states": int(A.shape[0]),
            "note": ("Reduced-order aLIGO suspension model, modal truncation to the "
                     "control band. Redistribution approved by the repo owner. QUAD near-"
                     "undamped modes carry a uniform structural shift, so their Q is model-"
                     "set, not physical; HSTS modes carry structural Q=50."),
        },
    }
    base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {base.name}.npz ({A.shape[0]} states) + .json ({len(modes)} modes)")


if __name__ == "__main__":
    for t in ("quad", "hsts"):
        regenerate(t)
