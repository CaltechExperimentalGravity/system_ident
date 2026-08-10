"""Prior-uncertainty-aware excitation, with an OFFSET prior.

Truth: f0=1.0 Hz, Q=20. Prior: f0=1.2 (+20% off), Q=24, gain=1.2. Realistic weak
actuator (DAC budget + analog filter 2p@1Hz/2z@10Hz), seismic+OSEM noise, 180 s.
The drive is DESIGNED from the (wrong) prior; the measurement is of the TRUE plant.

Compares, at the same DAC budget:
  flat          -- broadband (useless, weak actuator)
  point@prior   -- optimal excitation at the point prior (concentrates at 1.2 Hz,
                   MISSES the true 1.0 Hz resonance)
  robust@prior  -- prior_robust_excitation spreading over the prior's +/-u band
                   (covers the true resonance)
  ideal@truth   -- optimal excitation at the true f0 (reference; unknowable)

Run: conda run -n sysid python experiments/realistic/prior_robust_demo.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from system_ident.resonator import ResonatorModel
from system_ident.resonator_design import (
    fisher_information, optimal_excitation, prior_robust_excitation,
)

T_TOT = 180.0
SEIS, SENS = 1e-7, 1e-10

true = ResonatorModel.from_resonances([(1.0, 20.0)], 1.0)
prior = ResonatorModel.from_resonances([(1.2, 24.0)], 1.2)        # +20% off

freq = np.logspace(np.log10(0.05), np.log10(30.0), 1500)
w = 2 * np.pi * freq
wp, wz = 2 * np.pi * 1.0, 2 * np.pi * 10.0
Gmag = np.abs(true.eval(freq))
T_pend = (2 * np.pi * 1.0) ** 2 * Gmag
S_dist = SEIS * (1.0 / freq) ** 2 * T_pend
Pyy = S_dist ** 2 + SENS ** 2
Hmag = np.abs(((1j * w + wz) / (1j * w + wp)) ** 2)
Pyy_eff = Pyy / Hmag ** 2
usable = (freq >= 0.1) & (freq <= 20.0)


def to_budget(u, X2):
    u = np.where(usable, np.clip(u, 1e-300, None), 0.0)
    return u * (X2 / trapezoid(u, freq))


def crlb_at_truth(u):
    g = fisher_information(freq, true, u * Hmag ** 2, Pyy, T_TOT)
    return np.sqrt(np.clip(np.diag(np.linalg.inv(g)), 0, None))


def snr(u):
    return np.sqrt(u * Hmag ** 2 * Gmag ** 2) / np.sqrt(Pyy)


# calibrate DAC budget so the ideal (truth) drive peaks at SNR ~ 10
u_id1 = to_budget(optimal_excitation(freq, true, Pyy_eff, 1.0, n_iter=3), 1.0)
X2 = (10.0 / np.max(snr(u_id1)[usable])) ** 2

strategies = {
    "flat": to_budget(np.ones_like(freq), X2),
    "point@prior": to_budget(optimal_excitation(freq, prior, Pyy_eff, X2, n_iter=3), X2),
    "robust@prior u=.2": to_budget(prior_robust_excitation(freq, prior, Pyy_eff, X2, 0.2), X2),
    "robust@prior u=.5": to_budget(prior_robust_excitation(freq, prior, Pyy_eff, X2, 0.5), X2),
    "ideal@truth": to_budget(optimal_excitation(freq, true, Pyy_eff, X2, n_iter=3), X2),
}

i_true = np.argmin(np.abs(freq - 1.0))
print("Drive designed from a +20%-off prior (1.2 Hz); measuring the true 1.0 Hz plant.\n")
print(f"{'strategy':18} {'SNR@true f0':>12} {'sig_f0 %':>9} {'sig_Q %':>9} {'sig_gain %':>11}")
for name, u in strategies.items():
    sf, sq, sg = crlb_at_truth(u)
    print(f"{name:18} {snr(u)[i_true]:12.2f} {100*sf/1.0:9.3f} {100*sq/20:9.2f} {100*sg/1.0:11.3f}")

print("\n=> point@prior misses the true resonance (low SNR there -> poor); robust@prior")
print("   covers it and approaches ideal; flat is useless. The prior + its uncertainty")
print("   shape the gentle drive onto where the resonance actually is.")
