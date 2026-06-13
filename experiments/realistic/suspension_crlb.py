"""Why the drive MUST be shaped: a weak actuator can't give SNR everywhere.

Actuator = DAC (total command-RMS budget X_dac) + analog filter H_act with 2
poles @ 1 Hz, 2 zeros @ 10 Hz (flat x100 below 1 Hz, ~1/f^2 from 1->10 Hz, flat
x1 above). The saturating resource is the TOTAL DAC command RMS, allocated across
frequency. There is NO separate per-bin SNR cap -- the achievable SNR is simply
whatever the (weak) actuator + plant deliver. A flat broadband command wastes the
budget where H_act*G is small (it gets no SNR there); a shaped/optimal command
puts the limited DAC where it actually buys Fisher information.

Chain: X_cmd --DAC(budget)--> *H_act =force=> *G =resp=> + (seismic + OSEM noise)
We identify the suspension G (f0=1 Hz, Q=20). Time = 180 s.

Run: conda run -n sysid python experiments/realistic/suspension_crlb.py
"""

from __future__ import annotations

import numpy as np

from system_ident.resonator import ResonatorModel
from system_ident.resonator_design import fisher_information, optimal_excitation

F0, Q, GAIN = 1.0, 20.0, 1.0
T_TOT = 180.0
SEIS, SENS = 1e-7, 1e-10

model = ResonatorModel.from_resonances([(F0, Q)], GAIN)
freq = np.logspace(np.log10(0.05), np.log10(30.0), 1500)
w = 2 * np.pi * freq
wp, wz = 2 * np.pi * 1.0, 2 * np.pi * 10.0

Gmag = np.abs(model.eval(freq))
T_pend = (2 * np.pi * F0) ** 2 * Gmag
S_dist = SEIS * (1.0 / freq) ** 2 * T_pend
Pyy = (S_dist ** 2) + (SENS ** 2)
H_act = ((1j * w + wz) / (1j * w + wp)) ** 2
Hmag = np.abs(H_act)

usable = (freq >= 0.1) & (freq <= 20.0)
Pyy_eff = Pyy / Hmag ** 2     # effective noise for DAC-command optimal design


def crlb(u_cmd):
    """u_cmd = |X_cmd|^2 (DAC command PSD). Returns (sig_f0, sig_Q, sig_gain)."""
    gamma = fisher_information(freq, model, u_cmd * Hmag ** 2, Pyy, T_TOT)
    return np.sqrt(np.clip(np.diag(np.linalg.inv(gamma)), 0, None))


def to_budget(shape, X2):
    s = np.where(usable, np.clip(shape, 1e-300, None), 0.0)
    return s * (X2 / np.trapezoid(s, freq))


def resp_snr(u):
    return (np.sqrt(u * Hmag ** 2 * Gmag ** 2) / np.sqrt(Pyy)) * usable


# reference DAC budget where the OPTIMAL drive peaks at SNR ~ 10 (response ~ sqrt(budget))
u1 = to_budget(optimal_excitation(freq, model, Pyy_eff, 1.0, n_iter=3), 1.0)
X_ref = (10.0 / np.max(resp_snr(u1))) ** 2

print("Actuator-range sweep (DAC budget down by 10x). No imposed SNR cap;")
print("achievable SNR is whatever the actuator delivers. Optimal vs flat command:\n")
print(f"{'DAC (rel)':>9} {'flat peakSNR':>12} {'opt peakSNR':>12} {'sigQ flat %':>12} {'sigQ opt %':>11} {'opt/flat':>9}")
for mult in (1.0, 0.1, 0.01, 0.001):
    X2 = X_ref * mult
    uf = to_budget(np.ones_like(freq), X2)
    uo = to_budget(optimal_excitation(freq, model, Pyy_eff, X2, n_iter=3), X2)
    qf, qo = crlb(uf)[1], crlb(uo)[1]
    print(f"{mult:9.3f} {np.max(resp_snr(uf)):12.2f} {np.max(resp_snr(uo)):12.2f} "
          f"{100*qf/Q:12.2f} {100*qo/Q:11.2f} {qf/qo:9.1f}")

# show that a FLAT broadband command gets SNR only near the resonance, not "everywhere"
uf = to_budget(np.ones_like(freq), X_ref)
snr_f = resp_snr(uf)
print("\nflat command, SNR vs frequency (at the realistic actuator range):")
for f in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
    print(f"   {f:5.1f} Hz : SNR = {snr_f[np.argmin(np.abs(freq-f))]:6.2f}")
print("\n=> a flat drive squanders the limited DAC where H_act*G is small; you MUST")
print("   shape it onto the resonance to get usable SNR -> optimal excitation (and")
print("   the prior that tells you where to put it) is necessary, not optional.")
