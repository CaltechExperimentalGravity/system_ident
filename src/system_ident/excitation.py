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
    """Generate a coloured-noise drive matching the tabulated ASD.

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
