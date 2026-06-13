"""Achievable parameter uncertainty for a suspension sysID in the REAL regime.

Physical model (approved):
  * Plant G: POS resonance f0=1 Hz, Q=20 (force->displacement; ~1/f^2 above).
  * Disturbance (test-mass motion, no drive): seismic 1e-7*(1/f)^2 m/rtHz through
    the pendulum transmissibility (DC~1, peak ~Q at f0, 1/f^2 above).
  * OSEM readout noise: flat 1e-10 m/rtHz. Crosses the disturbance near a few Hz.
  * Actuator range (response ceiling R_max): flat = R0 below 1 Hz, R0/f^2 from
    1-10 Hz, flat = R0/100 above 10 Hz, with R0 = 10 * seismic 0-1 Hz RMS
    (we can overdrive the low band 10x). The 0-1 Hz worth of response power,
    B = R0^2, is the (allocatable) drive budget; R_max(f) is the per-bin ceiling.
  * Time: 180 s.

We compute the Cramer-Rao bound (Fisher) on (f0, Q, gain) for three ways of
spending the gentle drive: flat, Fisher-optimal, and prior-band (concentrate over
a +/-20% band around a +20%-off prior). Lower uncertainty = better.

Run: conda run -n sysid python experiments/realistic/suspension_crlb.py
"""

from __future__ import annotations

import numpy as np

from system_ident.resonator import ResonatorModel
from system_ident.resonator_design import fisher_information, optimal_excitation

F0, Q, GAIN = 1.0, 20.0, 1.0
T_TOT = 180.0
SEIS = 1e-7          # ground motion 1e-7*(1/f)^2 m/rtHz
SENS = 1e-10         # OSEM flat m/rtHz

model = ResonatorModel.from_resonances([(F0, Q)], GAIN)
freq = np.logspace(np.log10(0.05), np.log10(30.0), 1500)
w0 = 2 * np.pi * F0

Gmag = np.abs(model.eval(freq))
T_pend = (w0 ** 2) * Gmag                       # transmissibility, DC ~ 1
S_dist = SEIS * (1.0 / freq) ** 2 * T_pend      # output seismic motion ASD
S_sens = SENS * np.ones_like(freq)
Pyy = S_dist ** 2 + S_sens ** 2                 # quiet readout PSD

# crossover (disturbance == sensor)
cross = freq[np.argmin(np.abs(S_dist - S_sens))]

# actuator response ceiling R_max(f)
lo01 = (freq >= 0.1) & (freq <= 1.0)
sigma_seis_01 = np.sqrt(np.trapezoid((S_dist[lo01]) ** 2, freq[lo01]))
R0 = 10.0 * sigma_seis_01
R_max = np.where(freq <= 1.0, R0, np.where(freq < 10.0, R0 / freq ** 2, R0 / 100.0))
B = R0 ** 2                                     # response-power budget (0-1 Hz worth)
usable = (freq >= 0.1) & (freq <= 10.0)


def crlb(R_drive):
    """Given a response-ASD allocation, return (sigma_f0, sigma_Q, sigma_gain)."""
    R2 = np.clip(R_drive, 0, R_max) ** 2
    Pxx = np.where(Gmag > 0, R2 / Gmag ** 2, 0.0)   # drive PSD producing that response
    gamma = fisher_information(freq, model, Pxx, Pyy, T_TOT)
    cov = np.linalg.inv(gamma)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    return sig  # [f0, Q, gain]


def scale_to_budget(R2_shape):
    """Scale a (non-negative) response-power shape to the budget B over the band,
    clipping to R_max^2 with one redistribution pass."""
    R2 = np.where(usable, R2_shape, 0.0)
    for _ in range(40):
        free = R2 < R_max ** 2 - 1e-30
        used = np.trapezoid(np.minimum(R2, R_max ** 2)[~free & usable], freq[~free & usable]) if (~free & usable).any() else 0.0
        rem = B - used
        if rem <= 0 or not (free & usable).any():
            break
        cur = np.trapezoid(R2[free & usable], freq[free & usable])
        if cur <= 0:
            break
        R2 = np.where(free & usable, R2 * (rem / cur), np.minimum(R2, R_max ** 2))
        R2 = np.minimum(R2, R_max ** 2)
    return np.sqrt(R2)


# --- the binding per-frequency drive cap ------------------------------------
# Core constraint: response can exceed the DISTURBANCE by at most SNR_CAP (lock /
# sensor saturation), AND cannot exceed the actuator authority R_max(f).
SNR_CAP = 10.0
R_drive = np.minimum(SNR_CAP * S_dist, R_max) * usable
SNR = np.zeros_like(freq)
SNR[usable] = (R_drive / np.sqrt(Pyy))[usable]

print(f"regime: disturbance==sensor crossover ~ {cross:.1f} Hz;  "
      f"seismic 0-1Hz RMS = {sigma_seis_01:.2e} m;  R0(=10x) = {R0:.2e}")
print(f"achievable SNR: peak {np.max(SNR):.1f} at {freq[np.argmax(SNR)]:.2f} Hz; "
      f"median over usable {np.median(SNR[usable]):.1f}")
print(f"  SNR @ 0.3/1.0/3.0/10 Hz = "
      + ", ".join(f"{SNR[np.argmin(np.abs(freq-f))]:.1f}" for f in (0.3, 1.0, 3.0, 10.0)))

sf, sq, sg = crlb(R_drive)
print(f"\nCRLB in {T_TOT:.0f} s, driving every bin to its cap (the realistic best):")
print(f"  sigma_f0 = {sf:.5f} Hz  ({100*sf/F0:.2f} %)")
print(f"  sigma_Q  = {100*sq/Q:.2f} %")
print(f"  sigma_gain = {100*sg/GAIN:.2f} %")

# Does CONCENTRATING the drive help? Under a per-bin SNR cap you can't exceed the
# cap anywhere, so "drive everything to the cap" is already optimal; an excitation
# that drives ONLY the resonance band can only do WORSE (it gives up the other
# informative bins). Show that.
band = (freq >= 0.85) & (freq <= 1.15)
R_resonly = R_drive * band
sf2, sq2, sg2 = crlb(R_resonly)
print(f"\nresonance-band-only drive: sigma_f0={100*sf2/F0:.2f} %, sigma_Q={100*sq2/Q:.2f} %  "
      f"(worse -> no power-allocation tradeoff exists under a per-bin SNR cap)")
