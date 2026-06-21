"""Turn a designed excitation spectrum into an injectable time series.

Ported from ``sys_id_dev/sysIDlib.py::time_series_from_asd_vect``: colours
Gaussian noise to a tabulated ASD and trims the boundary-tainted ends, producing
a finite-duration drive ready for ``ChannelBackend.inject``.

Validated against the legacy engine (same seed -> same samples) in
``tests/test_step2_validation.py``.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig


def _nextpow2(n: int) -> int:
    return 2 ** int(np.ceil(np.log2(n)))


def timeseries_from_asd(
    duration: float,
    fs: float,
    freq: np.ndarray,
    asd: np.ndarray,
    seed: int | np.random.Generator | None = None,
    t_ramp: float = 0.0,
) -> np.ndarray:
    """Generate a coloured Gaussian-noise time series matching the tabulated ASD.

    .. warning::
       This is **not** an FRF excitation. System identification always drives with
       the deterministic Pintelon–Schoukens **multisine** (:func:`multisine_from_psd`)
       — a random-noise drive leaks and gives a biased, high-variance FRF. This
       generator is kept only for coloured *disturbance/background* time series and
       as a bit-for-bit port of the legacy ``sysIDlib.time_series_from_asd_vect``
       (validated in ``tests/test_step2_validation.py``).

    Parameters
    ----------
    duration:
        Length of the returned series [s].
    fs:
        Sample rate [Hz].
    freq, asd:
        Tabulated one-sided amplitude spectral density (``asd`` at ``freq`` [Hz]);
        the ASD is log-interpolated and frequencies outside the table get zero
        amplitude.
    seed:
        Seed or ``numpy`` ``Generator`` for reproducibility.
    t_ramp:
        Duration [s] of the cosine-taper transition at each end of the waveform
        (Tukey window).  ``0`` (default) leaves the waveform untapered, preserving
        existing behaviour exactly.

    Returns
    -------
    data : ndarray, shape ``(int(duration*fs),)``
        Time series whose realised ASD follows ``asd``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Synthesise at 2x the requested length and keep the middle to drop edges.
    N = int(duration * fs)
    Nfft = _nextpow2(int(2 * duration * fs))

    noise = rng.standard_normal(Nfft)
    spec = np.fft.rfft(noise * sig.windows.tukey(Nfft, 0.2))
    f_grid = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    pos = freq > 0
    log_asd = interp.interp1d(
        freq[pos], np.log(asd[pos]), bounds_error=False, fill_value=-np.inf
    )
    desired_asd = np.exp(log_asd(f_grid))

    # Standard-normal noise carries ASD ~ sqrt(2/fs); rescale to the target.
    spec *= desired_asd * np.sqrt(fs / 2.0)

    data = np.fft.irfft(spec)
    result = data[N // 2 : N // 2 + N]

    if t_ramp > 0.0:
        # alpha so that t_ramp seconds are tapered on each end
        alpha = min(1.0, 2.0 * t_ramp * fs / N)
        result = result * sig.windows.tukey(N, alpha)

    return result


def _schroeder_phases(amp: np.ndarray) -> np.ndarray:
    """Generalised Schroeder phases for a multisine with line amplitudes ``amp``.

    Pintelon & Schoukens, *System Identification: A Frequency Domain Approach*,
    Sec. 4: ``phi_k = -2*pi * sum_{j<k} (k-j) p_j`` with ``p_j`` the relative
    power of line ``j``.  These deterministic phases minimise the crest factor
    (peak drive / RMS) of the multisine *as defined here*, i.e. in plant/force-
    referred units.  Note the crest factor is **not** invariant under filtering:
    if there is a whitening/actuation chain between the digital request and the
    actuator's hard limit (the DAC in LIGO), the optimised phases are reshaped and
    the peak that actually binds is at the DAC, not here.  The phase choice does not
    affect the FRF or the Cramér–Rao bound (those depend on the line *amplitudes*),
    only the time-domain crest.
    """
    power = amp ** 2
    total = float(np.sum(power))
    if total <= 0:
        return np.zeros_like(amp)
    p = power / total
    idx = np.arange(amp.size)
    cum_p = np.concatenate([[0.0], np.cumsum(p)[:-1]])        # sum_{j<k} p_j
    cum_jp = np.concatenate([[0.0], np.cumsum(idx * p)[:-1]])  # sum_{j<k} j p_j
    return -2.0 * np.pi * (idx * cum_p - cum_jp)


def multisine_from_psd(
    Pxx: np.ndarray,
    fs: float,
    nperseg: int,
    n_periods: int,
    freq: np.ndarray,
    phase: str = "schroeder",
    seed: int | np.random.Generator | None = None,
    t_ramp: float = 0.0,
) -> np.ndarray:
    """Periodic random-phase multisine realising the excitation PSD ``Pxx``.

    This is the Pintelon-Schoukens excitation: a sum of harmonically related
    sinusoids whose period is exactly ``nperseg`` samples, tiled ``n_periods``
    times.  Because the waveform is *periodic* over the analysis window, a
    synchronous (integer-period) DFT of the response is leakage-free — unlike a
    one-off random record, which must be windowed and then leaks a sharp
    resonance into neighbouring bins.

    Parameters
    ----------
    Pxx, freq:
        Target one-sided excitation PSD ``Pxx`` sampled at ``freq`` [Hz]; ``freq``
        must be (a subset of) the synchronous bin grid ``k*fs/nperseg`` so each
        entry lands exactly on a DFT bin.  A line is placed at every ``freq`` bin
        with amplitude ``A_k = sqrt(2*Pxx_k*df)`` (``df = fs/nperseg``), so the
        total power ``sum A_k**2/2 ~ trapezoid(Pxx, freq)`` matches the budget the
        designer allocated (same watchdog behaviour as ``timeseries_from_asd``).
    fs, nperseg, n_periods:
        Sample rate [Hz], samples per period, number of tiled periods.  The
        returned series has ``nperseg*n_periods`` samples.
    phase:
        ``"schroeder"`` (default, low crest factor) or ``"random"``.
    seed:
        Seed / ``Generator`` (used only for ``phase="random"``).
    t_ramp:
        Half-cosine ramp-up applied to the **leading** ``t_ramp`` s only, so the
        start transient lands in the first (dropped) period.  ``0`` leaves the
        waveform a clean integer-period tiling.

    Returns
    -------
    data : ndarray, shape ``(nperseg*n_periods,)``
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    nperseg = int(nperseg)
    fs = float(fs)
    df = fs / nperseg
    freq = np.asarray(freq, dtype=float)
    Pxx = np.asarray(Pxx, dtype=float)

    # Line bin indices on the synchronous grid; drop DC and (for even nperseg)
    # the Nyquist line, which cannot carry an independent phase.
    k = np.round(freq * nperseg / fs).astype(int)
    keep = (k >= 1) & (k <= (nperseg - 1) // 2)
    k = k[keep]
    amp = np.sqrt(np.maximum(2.0 * Pxx[keep] * df, 0.0))

    if phase == "schroeder":
        phi = _schroeder_phases(amp)
    elif phase == "random":
        phi = rng.uniform(0.0, 2.0 * np.pi, size=amp.size)
    else:
        raise ValueError(f"unknown phase scheme {phase!r}")

    # One period via the rfft synthesis convention: a real tone A*cos(2*pi*k*t/N +
    # phi) has rfft coefficient (N/2)*A*exp(i*phi) at bin k.
    spec = np.zeros(nperseg // 2 + 1, dtype=complex)
    spec[k] = (nperseg / 2.0) * amp * np.exp(1j * phi)
    period = np.fft.irfft(spec, n=nperseg)

    out = np.tile(period, int(n_periods))

    if t_ramp > 0.0:
        n_ramp = min(int(round(t_ramp * fs)), out.size)
        if n_ramp > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
            out[:n_ramp] *= ramp

    return out
