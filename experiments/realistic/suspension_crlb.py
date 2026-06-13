"""Achievable suspension-sysID uncertainty with the REAL actuator chain.

Actuator (approved): a DAC with a fixed range (total command-RMS budget) followed
by an analog filter H_act with 2 poles @ 1 Hz and 2 zeros @ 10 Hz:
    |H_act|: flat (x100) below 1 Hz, ~1/f^2 from 1->10 Hz, flat (x1) above 10 Hz.
So the actuator is strong at low f and weak above 1 Hz, and the SATURATING
resource is the total DAC command RMS -- a budget you ALLOCATE across frequency.
This restores the optimal-excitation tradeoff (unlike a per-bin response cap).

Chain:  X_cmd(f)  --DAC(budget)-->  *H_act(f)  =force=>  *G(f)  =resp=>  + noise
We identify G (suspension resonance, f0=1 Hz, Q=20). Disturbance = seismic through
the pendulum; sensor = flat OSEM floor. Time = 180 s.

DAC-optimal allocation trick: with Pforce = |X_cmd|^2 |H_act|^2 and the budget on
|X_cmd|^2, the Fisher = 2T int |dG/dth|^2 Pforce/Pyy = 2T int |dG/dth|^2 |X_cmd|^2 /
(Pyy/|H_act|^2). So the DAC-optimal command is the standard optimal excitation
computed against an EFFECTIVE noise Pyy_eff = Pyy/|H_act|^2.

Run: conda run -n sysid python experiments/realistic/suspension_crlb.py
"""

from __future__ import annotations

import numpy as np

from system_ident.resonator import ResonatorModel
from system_ident.resonator_design import fisher_information, optimal_excitation

F0, Q, GAIN = 1.0, 20.0, 1.0
T_TOT = 180.0
SEIS, SENS = 1e-7, 1e-10
SNR_OVERDRIVE = 10.0      # we can lift the 0-1 Hz response RMS to 10x the seismic RMS

model = ResonatorModel.from_resonances([(F0, Q)], GAIN)
freq = np.logspace(np.log10(0.05), np.log10(30.0), 1500)
w = 2 * np.pi * freq
wp, wz = 2 * np.pi * 1.0, 2 * np.pi * 10.0

Gmag = np.abs(model.eval(freq))
T_pend = (2 * np.pi * F0) ** 2 * Gmag
S_dist = SEIS * (1.0 / freq) ** 2 * T_pend
Pyy = (S_dist ** 2) + (SENS ** 2)

# actuator analog filter: 2 poles @1 Hz, 2 zeros @10 Hz
H_act = ((1j * w + wz) / (1j * w + wp)) ** 2
Hmag = np.abs(H_act)

usable = (freq >= 0.1) & (freq <= 20.0)
lo01 = (freq >= 0.1) & (freq <= 1.0)

# TWO constraints together:
# (1) per-bin SNR cap: driven response can exceed the disturbance by <=10x
#     (sensor/lock saturation) -> u_cmd(f) <= u_cap(f).
# (2) total DAC command budget X_dac^2 (the actuator/DAC range), allocatable.
SNR_CAP = 10.0
u_cap = (SNR_CAP ** 2) * Pyy / (Hmag ** 2 * Gmag ** 2)        # u giving response SNR = 10
u_cap = np.where(usable, u_cap, 0.0)
# DAC range = just enough to drive the 0-1 Hz band to the SNR cap (overdrive the
# seismic-dominated low band 10x). Above 1 Hz H_act is weak -> little left.
X_dac2 = np.trapezoid(u_cap[lo01], freq[lo01])


def crlb(u_cmd):
    """u_cmd = |X_cmd(f)|^2 (DAC command PSD). Returns (sig_f0, sig_Q, sig_gain)."""
    Pforce = np.minimum(u_cmd, u_cap) * Hmag ** 2
    gamma = fisher_information(freq, model, Pforce, Pyy, T_TOT)
    sig = np.sqrt(np.clip(np.diag(np.linalg.inv(gamma)), 0, None))
    return sig


def fill(shape):
    """Water-fill a (relative) command shape to the DAC budget, capped per-bin."""
    u = np.where(usable, np.clip(shape, 1e-300, None), 0.0)
    for _ in range(80):
        u = np.minimum(u, u_cap)
        capped = u >= u_cap - 1e-300
        used = np.trapezoid(np.where(capped, u, 0.0), freq)
        rem = X_dac2 - used
        free = usable & ~capped
        cur = np.trapezoid(np.where(free, u, 0.0), freq)
        if rem <= 1e-30 or cur <= 0 or not free.any():
            break
        u = np.where(free, u * (rem / cur), u)
    return np.minimum(u, u_cap)


# --- strategies (all spend the SAME DAC budget, all obey the per-bin cap) ----
u_flat = fill(np.ones_like(freq))
Pyy_eff = Pyy / Hmag ** 2
u_opt = fill(optimal_excitation(freq, model, Pyy_eff, X_dac2, n_iter=3))
band = (freq >= 0.96) & (freq <= 1.44)        # +20%-off prior, +/-20% band
u_prior = fill(np.where(band, 1.0, 0.0))


def snr_of(u):
    R = np.sqrt(u * Hmag ** 2 * Gmag ** 2)   # driven response ASD
    return np.max((R / np.sqrt(Pyy))[usable])


print(f"actuator |H_act|: DC={Hmag[0]:.0f}x, @10Hz={Hmag[np.argmin(np.abs(freq-10))]:.2f}x  "
      f"(2p@1Hz, 2z@10Hz)")
print(f"constraints: per-bin response SNR <= {SNR_CAP:.0f}, total DAC budget = "
      f"saturate the 0-1 Hz band to the cap\n")
print(f"{'strategy':12} {'peak SNR':>9} {'sig_f0 %':>9} {'sig_Q %':>9} {'sig_gain %':>11}")
for name, u in [("flat", u_flat), ("DAC-optimal", u_opt), ("prior-band", u_prior)]:
    sf, sq, sg = crlb(u)
    print(f"{name:12} {snr_of(u):9.1f} {100*sf/F0:9.3f} {100*sq/Q:9.2f} {100*sg/GAIN:11.3f}")

sff = crlb(u_flat); sfo = crlb(u_opt)
print(f"\noptimal vs flat: sigma_Q {sff[1]/sfo[1]:.1f}x tighter, sigma_f0 {sff[0]/sfo[0]:.1f}x "
      f"(=> {(sff[1]/sfo[1])**2:.1f}x less time for the same Q error)")

# Does the excitation-allocation benefit appear? It depends on the DAC range: if
# the DAC can REACH the SNR cap across the informative band you just saturate it
# (no tradeoff); if the DAC is the binding constraint (can't reach the cap) you
# must ALLOCATE -> optimal concentrates near the resonance and beats flat.
print("\nDAC-budget sweep (relative to 'saturate the 0-1 band'): optimal vs flat")
print(f"{'DAC x':>7} {'peak SNR(flat)':>15} {'sig_Q flat %':>13} {'sig_Q opt %':>12} {'opt/flat':>9}")
base = X_dac2
for mult in (0.03, 0.1, 0.3, 1.0, 3.0):
    X_dac2 = base * mult
    uf = fill(np.ones_like(freq))
    uo = fill(optimal_excitation(freq, model, Pyy / Hmag ** 2, X_dac2, n_iter=3))
    qf, qo = crlb(uf)[1], crlb(uo)[1]
    psnr = np.max((np.sqrt(np.minimum(uf, u_cap) * Hmag ** 2 * Gmag ** 2) / np.sqrt(Pyy))[usable])
    print(f"{mult:7.2f} {psnr:15.1f} {100*qf/Q:13.2f} {100*qo/Q:12.2f} {qf/qo:9.1f}")
print("\n=> optimal excitation helps only when the DAC can't reach the SNR cap "
      "(scarce actuator); once it can, you just saturate the band.")
