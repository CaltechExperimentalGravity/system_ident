"""Analytic oracle + scenario init for rtsfreerun twin models.

A digital-twin *scenario* (``digital_twin/twin/scenarios/*.yaml``) specifies a
suspension plant as foton ZPK loaded into ``cdsFilt`` modules. The bare compiled
model (``x1hsts.x1hsts()``) ships with **no** filters — the twin orchestrator
applies the scenario ``init:`` block at run time. This module gives
``system_ident`` two things it needs to identify that plant:

1. :func:`apply_scenario_init` — replicate the orchestrator's ``_apply_init`` so a
   bare ``mdl`` realises the scenario plant before we drive it. Foton ZPK in the
   ``'f'`` plane is converted exactly as the orchestrator does (roots → ``-2π·root``,
   sent as plane ``'s'``); see ``digital_twin/twin/src/twin/orchestrator.py``.
2. The **analytic oracle**: :func:`analytic_plant` composes the drive→sensor filter
   modules' ZPK into a continuous :class:`~system_ident.model.TFModel` — the *truth*
   a recovered model is scored against. :func:`realized_plant_response` reads the
   model's *loaded* discrete SOS back (``fm_get_sos``) as an independent cross-check
   that the init was applied correctly (the two agree to ~1e-6 on ``x1hsts``).

Nothing here imports the twin package or any compiled model at import time — the
``mdl``-taking functions accept an already-built model, so importing
``system_ident`` never requires the twin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.signal as sig

from ..model import TFModel

# The drive→sensor filter modules for the shipped HSTS scenarios. The suspension
# transfer function the sysID recovers is the series product of these modules.
HSTS_DRIVE_MODULES = ("HSTS_DRV_TF_A", "HSTS_DRV_TF_B")


def load_scenario(path: str | Path) -> dict:
    """Load a twin scenario YAML into a dict."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


# ── foton ZPK → s-plane, exactly as the twin orchestrator does ────────────────
def _to_complex(v) -> complex:
    return complex(v["re"], v["im"]) if isinstance(v, dict) else complex(v)


def _coerce_zpk(zpk: dict) -> tuple[list[complex], list[complex], float]:
    """``{z, p, k, plane}`` (foton) → s-plane ``(zeros, poles, gain)``.

    Mirrors ``orchestrator._coerce_zpk``: ``plane='f'`` roots are mapped to
    ``s = -2π·root`` (the negation puts them in the stable LHP); ``plane='s'``
    roots pass through. Gain is unchanged.
    """
    plane = zpk.get("plane", "s")
    z = [_to_complex(v) for v in zpk.get("z", [])]
    p = [_to_complex(v) for v in zpk.get("p", [])]
    if plane == "f":
        z = [-2 * np.pi * v for v in z]
        p = [-2 * np.pi * v for v in p]
    elif plane != "s":
        raise ValueError(f"unsupported zpk plane: {plane!r}")
    return z, p, float(zpk["k"])


def apply_scenario_init(mdl, scenario: dict) -> None:
    """Apply a scenario's ``init:`` block to a built model.

    Replicates ``twin.orchestrator._apply_init`` using only the model's own
    ``fm_set_zpk`` / ``fm_set_sos`` / ``fm_set_switches`` / ``write`` API, so the
    bare ``mdl`` realises the scenario plant (and its ground/readout paths).
    """
    for op in scenario.get("init", []):
        if "fm" in op and "zpk" in op:
            z, p, k = _coerce_zpk(op["zpk"])
            mdl.fm_set_zpk(op["fm"], op["section"], z=z, p=p, k=k, plane="s")
        elif "fm" in op and "sos" in op:
            mdl.fm_set_sos(op["fm"], op["section"], op["sos"])
        elif "fm" in op and "switches" in op:
            mdl.fm_set_switches(op["fm"], **op["switches"])
        elif "gain" in op:
            mdl.write(op["gain"], op["value"])
        elif "write" in op:
            mdl.write(op["write"], op["value"])
        else:
            raise ValueError(f"unrecognized init op: {op!r}")


def _real_tf_from_zpk(zeros, poles, gain) -> TFModel:
    """``zpk2tf`` then drop the (numerically zero) imaginary part of the coeffs."""
    num, den = sig.zpk2tf(np.asarray(zeros), np.asarray(poles), gain)
    num, den = np.atleast_1d(num), np.atleast_1d(den)
    for name, c in (("num", num), ("den", den)):
        if np.max(np.abs(c.imag)) > 1e-6 * max(np.max(np.abs(c.real)), 1.0):
            raise ValueError(f"{name} has non-negligible imaginary part; roots not conjugate-symmetric")
    return TFModel(num=num.real, den=den.real)


def analytic_plant(scenario: dict, modules: Sequence[str] = HSTS_DRIVE_MODULES) -> TFModel:
    """Compose the drive→sensor plant from the scenario ZPK as a :class:`TFModel`.

    Series product of every ``zpk`` section of ``modules`` (the truth oracle). For
    the shipped HSTS scenario this is the order-10 / 8-zero ``HSTS_DRV_TF`` cascade.
    """
    modset = set(modules)
    zeros: list[complex] = []
    poles: list[complex] = []
    gain = 1.0
    for op in scenario.get("init", []):
        if op.get("fm", "") in modset and "zpk" in op:
            z, p, k = _coerce_zpk(op["zpk"])
            zeros += z
            poles += p
            gain *= k
    return _real_tf_from_zpk(zeros, poles, gain)


def realized_plant_response(mdl, freq, scenario: dict,
                            modules: Sequence[str] = HSTS_DRIVE_MODULES) -> np.ndarray:
    """Frequency response of the plant the model *actually loaded*, from ``fm_get_sos``.

    Independent of :func:`analytic_plant` — used as a cross-check that
    :func:`apply_scenario_init` realised the intended plant (they agree to ~1e-6
    on ``x1hsts``).
    """
    modset = set(modules)
    fs = float(mdl.sample_rate)
    w = 2 * np.pi * np.asarray(freq, float) / fs
    H = np.ones(w.shape, dtype=complex)
    for op in scenario.get("init", []):
        if op.get("fm", "") in modset and "zpk" in op:
            sos = mdl.fm_get_sos(op["fm"], op["section"], "py")
            _, h = sig.sosfreqz(sos, worN=w)
            H = H * h
    return H


def plant_modes(model: TFModel) -> list[tuple[float, float]]:
    """``(f0_Hz, Q)`` for each conjugate pole pair of ``model``, low → high f0."""
    roots = np.roots(np.asarray(model.den, float))
    modes = [(abs(r) / (2 * np.pi), abs(r) / (2 * abs(r.real)))
             for r in roots if r.imag > 1e-9]
    return sorted(modes)


def _perturb_conjugate(roots: np.ndarray, perturb: float, rng) -> np.ndarray:
    """Scale each root's real & imag parts by ``1 + perturb·N(0,1)``, *preserving
    conjugate symmetry* (so the resulting polynomial coefficients stay real).

    Each conjugate pair is perturbed once and mirrored; real roots are scaled in
    place. Perturbing real and imag independently moves both f0 and Q.
    """
    roots = np.asarray(roots, dtype=complex)
    out: list[complex] = []
    used = np.zeros(len(roots), dtype=bool)
    for i, r in enumerate(roots):
        if used[i]:
            continue
        used[i] = True
        if abs(r.imag) < 1e-9:
            out.append(complex(r.real * (1 + perturb * rng.standard_normal()), 0.0))
            continue
        new = complex(r.real * (1 + perturb * rng.standard_normal()),
                      r.imag * (1 + perturb * rng.standard_normal()))
        out.append(new)
        out.append(np.conj(new))
        for j in range(i + 1, len(roots)):          # consume the conjugate partner
            if not used[j] and abs(roots[j] - np.conj(r)) < 1e-6 * abs(r):
                used[j] = True
                break
    return np.array(out)


def prior_from_scenario(scenario: dict, modules: Sequence[str] = HSTS_DRIVE_MODULES,
                        *, perturb: float = 0.0, gain_scale: float = 1.0,
                        rng=None) -> TFModel:
    """A prior :class:`TFModel` of the *correct order*, derived from the scenario.

    Returns :func:`analytic_plant` with each pole/zero perturbed by ``perturb``
    (conjugate-symmetrically, so coefficients stay real) and the gain scaled by
    ``gain_scale``. ``perturb=0`` returns the truth (its *order* is what the
    estimator needs; the values are refined by measurement). Models "we know the
    suspension design; the measurement gives us the as-built plant."
    """
    truth = analytic_plant(scenario, modules)
    zeros = np.roots(np.asarray(truth.num, float)) if len(truth.num) > 1 else np.array([])
    poles = np.roots(np.asarray(truth.den, float))
    gain = truth.num[0] / truth.den[0]
    if perturb:
        rng = rng if rng is not None else np.random.default_rng()
        zeros = _perturb_conjugate(zeros, perturb, rng)
        poles = _perturb_conjugate(poles, perturb, rng)
    return _real_tf_from_zpk(zeros, poles, gain * gain_scale)
