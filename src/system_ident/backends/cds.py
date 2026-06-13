"""Real-hardware backend: awg excitation injection + nds2 online readback.

The CDS libraries (``awg``/``cdsutils``, ``nds2``, ``foton``) are provided by
the LIGO control-room environment and are **lazy-imported** inside the methods,
so importing this module (and the package) never requires them — the twin path
stays usable on a plain scientific-Python stack. Implemented in build step 8.

LIMITATIONS — what the periodic / ML path has and has NOT been validated against
--------------------------------------------------------------------------------
The Pintelon-Schoukens periodic-multisine measurement (``measurement.mode:
periodic``), the reference-based closed-loop FRF, and the ML estimator
(``estimator: gml``) are validated **only on the digital twin**, which can now
model a closed control loop, a response-path transport delay, and actuator
saturation (see :class:`~system_ident.backends.twin.TwinBackend`).  The twin
still idealises several real-CDS effects, so the following remain **UNTESTED**
until this backend is implemented and run in the control room:

* **AWG↔NDS sample alignment / clocking.**  A constant integer-sample offset
  between drive and response *cancels* in the ratio-of-averages FRF, but a
  fractional-sample clock drift between generation and acquisition would
  re-introduce leakage; the twin assumes perfect co-sampling.
* **NDS GPS-timestamp alignment** of the first analysed sample to the injected
  period boundary (the reshape grid).
* **Real actuator nonlinearity** beyond a hard clip (DAC slew, analog
  saturation shape), which the Schroeder crest-factor choice mitigates but does
  not eliminate.
* **True controller dynamics** (the twin's feedback ``C(s)`` is a rational
  stand-in for the real digital servo + filters).

Do not read a green twin suite as "validated on hardware"; these gaps close
only when ``CDSBackend`` lands and is exercised on a real suspension.
"""

from __future__ import annotations

import numpy as np

from .base import ChannelBackend


class CDSBackend(ChannelBackend):
    """awg-inject + nds2-readback against real CDS hardware (build step 8)."""

    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        raise NotImplementedError("CDSBackend lands in build step 8")

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        raise NotImplementedError("CDSBackend lands in build step 8")

    def ramp_down(self, channel: str, secs: float) -> None:
        raise NotImplementedError("CDSBackend lands in build step 8")
