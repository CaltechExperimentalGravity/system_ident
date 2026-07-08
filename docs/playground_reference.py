"""Reference implementation of the excitation-playground engine.

Presentation-only (not package API). This is the **Python twin** of
``docs/assets/excitation-playground.js``: the two share one set of constants and one set of
formulas, so the browser sandbox computes the *same* Fisher / Cramer-Rao numbers the
``system_ident`` pipeline does, and the *same* crest factors on the synthesized drive.

The plant and the Fisher/CRB machinery are reused verbatim from ``arcade_reference`` (same
locked constants), which ``tests/test_arcade.py`` already pins to the package's ``TFModel``
pole convention. This module adds the two things the playground needs beyond the arcade:

1. **Excitation power spectra** ``Pxx(f)`` for each drive family (optimal / flat / shaped
   multisine, linear & log swept sine, white & shaped noise). ``ETA = C_TIME * worst_frac**2``
   is a function of ``Pxx`` alone — estimation speed is set by the *power spectrum*.
2. **Crest factors** of the actual time-domain drive. Crest is set by the *phases*, not the
   power: a Schroeder multisine and a random-phase multisine can share one ``Pxx`` yet have
   wildly different crest. Crest is measured on the synthesized waveform the DAC would emit
   (no whitening-filter conflation) — ``max|x| / rms(x)``.

``tests/test_playground.py`` pins the golden ETAs and crest factors the JS is calibrated to.
Keep this file and the JS numerically in lock-step: edit both, or neither.
"""
from __future__ import annotations

import numpy as np

import arcade_reference as ar  # same locked constants + Fisher/CRB kernel

# ── shared frequency grid / budget (from arcade_reference) ──────────────────────────
FREQ = ar.FREQ
DF = ar.DF
F_LO, F_HI = ar.F_LO, ar.F_HI
PX_TOT = ar.PX_TOT
C_TIME = ar.C_TIME

# ── time-domain synthesis grid (MUST equal the JS SYNTH constants) ──────────────────
FS = 40.0            # synthesis sample rate [Hz] (Nyquist 20 Hz > 6 Hz band)
NT = 2048            # samples in the synthesized drive window
SEED = 0x51D          # PRNG seed for random-phase / noise realizations (shared with JS)
TVEC = np.arange(NT) / FS


# ── deterministic PRNG shared with the JS (mulberry32) ──────────────────────────────
def mulberry32(seed):
    """Yield the same float sequence as the JS mulberry32 (uint32 arithmetic).

    Faithful to ``t = Math.imul(a^a>>>15, 1|a); t = (t + Math.imul(t^t>>>7, 61|t)) ^ t;``
    (``+`` binds tighter than ``^``, so the trailing ``^ t`` xors in the pre-add value).
    """
    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    a = seed & 0xFFFFFFFF
    while True:
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = (((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        yield ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def _rand_phases(n, seed=SEED):
    g = mulberry32(seed)
    return np.array([2 * np.pi * next(g) for _ in range(n)])


# ── excitation power spectra Pxx(f) on the FREQ grid (each normalized to PX_TOT) ────
def _normalize(P):
    P = np.clip(np.asarray(P, float), 0, None)
    area = np.trapezoid(P, FREQ)
    return P / area * PX_TOT if area > 0 else P


def power_flat():
    return _normalize(np.ones_like(FREQ))


def power_optimal():
    return ar.optimal_drive()


def power_shaped(alpha):
    """1/f**alpha shaped density (alpha=0 flat, 1 pink, 2 red)."""
    return _normalize(FREQ ** (-float(alpha)))


POWER = {
    "optimal": power_optimal,
    "flat": power_flat,
    "pink": lambda: power_shaped(1.0),
}


def power_of(kind):
    """Power spectrum for a named drive family (used for ETA)."""
    if kind in ("opt_schroeder", "opt_random", "shaped_fisher"):
        return power_optimal()
    if kind in ("flat_schroeder", "flat_random", "cophased", "chirp_lin", "white"):
        return power_flat()
    if kind == "pink":
        return power_shaped(1.0)
    raise KeyError(kind)


def eta_of(kind):
    """Time-to-5% for a named drive family (Fisher/CRB, depends only on Pxx)."""
    return ar.eta_seconds(power_of(kind))[0]


# ── time-domain drive synthesis + crest factor ──────────────────────────────────────
def _crest(x):
    x = np.asarray(x, float)
    rms = np.sqrt(np.mean(x ** 2))
    return float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0


def schroeder_phases(P):
    """Schroeder (1970) low-crest phases for line powers P: phi_k = -2*pi*sum_{l<k}(k-l)q_l."""
    q = np.asarray(P, float) / np.sum(P)
    phi = np.zeros(len(q))
    acc = 0.0
    run = 0.0  # running sum of q_l
    for k in range(1, len(q)):
        run += q[k - 1]
        acc += run
        phi[k] = -2 * np.pi * acc
    return phi


def multisine(P, phase="schroeder", seed=SEED):
    """Synthesize a multisine drive from line-power spectrum P; return (waveform, crest)."""
    P = np.asarray(P, float)
    amp = np.sqrt(2 * np.clip(P, 0, None) * DF)  # amplitude per line
    if phase == "schroeder":
        phi = schroeder_phases(np.clip(P, 1e-30, None))
    elif phase == "random":
        phi = _rand_phases(len(P), seed)
    else:  # cophased / zero
        phi = np.zeros(len(P))
    x = (amp[:, None] * np.cos(2 * np.pi * FREQ[:, None] * TVEC[None, :] + phi[:, None])).sum(0)
    return x, _crest(x)


def swept_sine():
    """Constant-amplitude linear swept sine over the band; return (waveform, crest)."""
    T = NT / FS
    mu = (F_HI - F_LO) / T
    x = np.sin(2 * np.pi * (F_LO * TVEC + 0.5 * mu * TVEC ** 2))
    return x, _crest(x)


def noise(alpha=0.0, seed=SEED):
    """Colored Gaussian-like noise as a random-phase shaped multisine; return (waveform, crest)."""
    return multisine(power_shaped(alpha) if alpha else power_flat(), "random", seed)


# ── the catalog: everything the scoreboard races ────────────────────────────────────
def waveform_of(kind):
    """Synthesized drive (waveform, crest) for a named family."""
    if kind == "opt_schroeder":
        return multisine(power_optimal(), "schroeder")
    if kind == "opt_random":
        return multisine(power_optimal(), "random")
    if kind == "flat_schroeder":
        return multisine(power_flat(), "schroeder")
    if kind == "flat_random":
        return multisine(power_flat(), "random")
    if kind == "cophased":
        return multisine(power_flat(), "zero")
    if kind == "chirp_lin":
        return swept_sine()
    if kind == "white":
        return noise(0.0)
    if kind == "pink":
        return noise(1.0)
    if kind == "shaped_fisher":
        return multisine(power_optimal(), "random")
    raise KeyError(kind)


CATALOG = [
    "opt_schroeder", "opt_random", "flat_schroeder", "flat_random", "cophased",
    "chirp_lin", "white", "pink", "shaped_fisher",
]


def scoreboard():
    """(kind -> (eta_seconds, crest_factor)) for every catalog entry."""
    return {k: (eta_of(k), waveform_of(k)[1]) for k in CATALOG}


if __name__ == "__main__":  # pragma: no cover - manual calibration aid
    print(f"{'drive':<18}{'ETA (s)':>12}{'crest':>10}")
    for k, (eta, cr) in scoreboard().items():
        print(f"{k:<18}{eta:>12.2f}{cr:>10.3f}")
