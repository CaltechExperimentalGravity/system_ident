"""Size a periodic-multisine campaign from the prior model's ringdown.

Pintelon-Schoukens require the segment (period) length to exceed the plant time
constant for a low-leakage FRF (the Antoni-Schoukens leakage bound), and the frequency
resolution is df = fs/nperseg = 1/T. So the prior model's slowest-ringing (highest-Q,
lowest-f) mode sets the period — and, to resolve a mode's Q, df must put several bins
across its resonance:

    mode (f0, Q):  ringdown  tau = Q / (pi f0),   FWHM bandwidth  Delta_f = f0 / Q.
    resolve Q  ->  df <= Delta_f / bins_per_fwhm   <=>   T = 1/df >= bins_per_fwhm * Q / f0.
    steady state (drop transient) -> T_record covers settle_factor * tau_max.

`recommend_resolution` turns the prior modes into the concrete campaign knobs so the
measurement is sized by the physics, not guessed. (Verified on the RTSfreerun HSTS: a
Q~50 mode at 0.67 Hz has Delta_f ~ 13 mHz, so df ~ 3 mHz / T ~ 256 s -- a long record,
exactly because high-Q suspension modes ring for a long time.)
"""
from __future__ import annotations
import numpy as np


def recommend_resolution(modes, fs, *, bins_per_fwhm=4, settle_factor=3.0, pow2=True):
    """Recommend (nperseg, df, n_transient) for a campaign that resolves the prior modes.

    Parameters
    ----------
    modes : sequence of (f0_Hz, Q)
        The prior model's resonances (frequency in Hz, quality factor).
    fs : float
        Sample rate of the campaign [Hz].
    bins_per_fwhm : int
        Frequency bins required across the narrowest resonance FWHM (>=3 for a usable Q).
    settle_factor : float
        Multiples of the slowest ringdown to allow as the transient (dropped periods).
    pow2 : bool
        Round nperseg up to a power of two (fast rFFT, integer-period multisine).

    Returns
    -------
    nperseg : int   FFT length / samples per period.
    df : float      achieved frequency resolution fs/nperseg [Hz].
    n_transient : int   leading periods to drop for steady state.
    """
    modes = [(float(f0), float(q)) for f0, q in modes]
    if not modes:
        raise ValueError("need at least one prior mode")
    f0 = np.array([m[0] for m in modes])
    Q = np.array([m[1] for m in modes])
    if np.any(f0 <= 0) or np.any(Q <= 0):
        raise ValueError("modes need positive f0 and Q")
    fwhm = f0 / Q                                   # resonance bandwidth per mode
    df_target = fwhm.min() / float(bins_per_fwhm)   # the narrowest peak is binding
    n_raw = fs / df_target
    nperseg = int(2 ** np.ceil(np.log2(n_raw))) if pow2 else int(np.ceil(n_raw))
    df = fs / nperseg
    T = nperseg / fs                                # period length [s]
    tau_max = float((Q / (np.pi * f0)).max())       # slowest ringdown
    n_transient = max(1, int(np.ceil(settle_factor * tau_max / T)))
    return nperseg, df, n_transient
