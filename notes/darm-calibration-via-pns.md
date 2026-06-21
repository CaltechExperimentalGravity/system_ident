# A Pintelon–Schoukens DARM calibration: architecture and comparison to the swept‑sine sweep

**Status:** design note / discussion draft — for the LIGO calibration group.
**Scope:** measuring the DARM response function `R(f)` (sensing `C`, actuation `A`) with an
*optimal‑excitation periodic‑multisine* system‑identification (P&S) measurement, and a fair
head‑to‑head against the present photon‑calibrator swept‑sine sweep. No change to the Pcal
absolute standard is proposed.

This note is feedstock for discussion, not a results paper. Numbers attributed to published
work are cited; a few citations are flagged *unverified* and must be checked against the source
PDFs before they are quoted in anything circulated.

---

## 1. What we are measuring (notation, so we agree on symbols)

Strain reconstruction inverts the DARM loop:

```
h(t) = R · d_err / L ,     R(f) = (1 + G)/C ,     G = A · D · C
```

- **C(f,t)** — sensing: optical gain `H_C`, coupled‑cavity pole `f_cc` (~360 Hz H1 / ~370 Hz L1),
  time delay `τ_C ≈ 77.6 µs`, optical‑spring `(f_s, Q_s)`, and the scalar time‑dependent
  correction `κ_C(t)`. (Cahillane 2017 reference values: H1 `H_C = 3.834 ± 0.003 mA/pm`,
  `f_cc = 360 ± 2 Hz`.) [1]
- **A(f,t)** — actuation as the sum of three quadruple‑suspension stages UIM/PUM/TST(ESD), each a
  scale `H_i` × shape × delay, tracked by `κ_T(t)` and `κ_PU(t)`. [1,5]
- **D(f)** — digital filters, known to negligible uncertainty.
- **Absolute reference** — the photon calibrator (Pcal), a 1047 nm radiation‑pressure actuator,
  `~0.75%` displacement uncertainty in O1/O2 → `~0.4%` (O3) → `0.3%/0.15%` (O4). [8,9]

The deliverable the calibration group actually ships is a **68% envelope on `R(f)`** over
20 Hz–2 kHz — magnitude (%) and phase (°) — that separates *systematic error* (bias) from
*statistical uncertainty*, in a form downstream PE can marginalise over [2,11,12].

**This is a frequency‑domain system‑identification problem.** `C` and the per‑stage `A_i` are
LTI transfer functions measured by driving a known excitation and estimating an FRF. That is
exactly the problem the Pintelon–Schoukens framework is built for — and, as far as the literature
search found, **no one has applied P&S optimal‑excitation ID to GW calibration** (§6).

---

## 2. The present‑day sweep (baseline)

**Absolute reference.** Pcal swept‑sine gives `C_meas(f) = H_Pcal · (1+G) · [d_err / x_T^PC]`;
per‑stage actuation sweeps give `[H_i A_i]_meas = (1/H_Pcal)(x_T^PC/d_err)(d_err/x_i)`,
referenced to Pcal. [8,1]

**The sweep itself.** A full sensing transfer function is **~60 single‑sine points over
20 Hz–1.2 kHz, ~1 hour**, reaching **~1% magnitude / ~1° phase** (statistical, from coherence)
over ~30 Hz–1.2 kHz. Above 1 kHz, where Pcal SNR is weak, **single frequencies are driven for
many hours each**. [8] Full sweeps run roughly **weekly** during a run. [2]

**Continuous tracking (TDCFs).** Four calibration lines (H1: ~35.9 Hz on ESD, ~36.7 & ~331.9 Hz
on Pcal, ~37.3 Hz on DARM ctrl; plus a ~1083.7 Hz Pcal line in the pipeline era) are demodulated
continuously to track `κ_T, κ_PU, κ_C` and `f_cc(t)`, e.g. `κ_C = |S|²/Re[S]`, `f_cc` from the
real/imag ratio of the demodulated sensing line. [5,4]

**Uncertainty estimation.** MCMC fit of the reference sweeps + Gaussian‑process regression on the
residuals + TDCF and Pcal distributions → **10,000 `R(f)` realisations**; the GP‑regression
residual term dominates. [1] Achieved: O1 `±4.6%/±3.0%` mag, `±2.7°/±2.0°` phase; O3a `<7%/<4°`;
O3b `11.29%/9.18°` (worse — *frequency‑dependent modelling* error, not the absolute reference);
**O4 target `<1%/1°`** with network absolute reference `<0.1%/0.1°`. [1,2,3,6]

**Two facts that matter for what follows.**
1. A *broadband* Pcal injection already exists — colored random noise to measure the whole TF at
   once — but it is used **only as an offline cross‑check when not observing**, is **not**
   optimised (no crest‑factor/Schroeder design, no periodic‑multisine analysis), and is not the
   primary estimator. [8,2]
2. The O3b regression was limited by **imperfect frequency‑dependent modelling of `R(f)`** — i.e.
   the *shape* measurement, exactly what a leakage‑free broadband estimate targets. [3]

---

## 3. The P&S measurement, mapped onto DARM

Replace "one sine at a time, then GP‑regress the gaps" with **one designed periodic multisine that
measures every frequency at once, leakage‑free, with a per‑bin noise estimate and a Cramér–Rao
error bar** — injected through the *same Pcal*, so the absolute standard is untouched.

The four ingredients (this is the pipeline `system_ident` already implements for suspensions; the
quad actuation stages *are* suspensions):

1. **Periodic multisine.** A sum of harmonically related lines whose period is exactly the
   analysis window, tiled `N` periods. The win over a swept sine is **simultaneity** — every
   frequency is measured at once in one window — not crest‑factor; Pcal force is not the binding
   constraint here, so the phases need not be Schroeder‑optimised, a plain multisine is enough. [15]
2. **Leakage‑free synchronous‑DFT FRF.** Because the drive is periodic over the window, an
   integer‑period DFT is leakage‑free: the sharp `f_cc` roll‑off and any resonance are measured
   **without the windowing bias** that a one‑off record incurs. This directly attacks the O3b
   frequency‑dependent‑modelling error. [15,3]
3. **Per‑bin nonparametric noise covariance** from the period‑to‑period scatter (the Local
   Polynomial Method). The error bar on each FRF bin is *measured*, not assumed — which is what the
   calibration envelope needs, and what GP‑regression is currently doing the job of. [15,1]
4. **ML parametric fit + Fisher/CRB**, then **optimal experiment design**: allocate the Pcal power
   budget to the bins that most constrain the *parameters you care about* — `f_cc`, `H_C`, the
   delay, the stage gains — instead of spreading it uniformly. The Cramér–Rao bound says where a
   marginal Hz of drive buys the most reduction in `σ(R(f))`. [15]

**Proposed protocol (sensing `C`, one measurement window):**

1. From the current best `C, A` model, design the optimal Pcal multisine PSD (concentrate power at
   `f_cc` and where `∂R/∂θ` is largest), Schroeder‑phase it, scale to the Pcal force limit.
2. Inject `N+1` periods through Pcal; drop the first (transient); keep `N` steady periods.
3. Form the leakage‑free reference FRF `d_err / x_Pcal` → `C_meas(f)` with its per‑bin σ.
4. ML‑fit `(H_C, f_cc, τ_C, optical spring, κ_C)`; accumulate the Fisher matrix → CRB envelope.
5. Repeat per actuation stage with the drive at `x_U / x_P / x_T`, referenced to Pcal.
6. Roll the per‑bin FRF + the parametric fit into the same `R(f)` realisation machinery [1].

Steps 1–4 are the `system_ident` loop unchanged; only the channels and the plant model are DARM's.

---

## 4. Head‑to‑head, on the calibration group's figures of merit

| FOM | Swept‑sine baseline | P&S multisine | Who wins |
|---|---|---|---|
| **σ(R(f)), 20 Hz–2 kHz** | ~few‑%/few‑° (O3); O4 target <1%/1° [1,2,6] | Same envelope machinery; leakage‑free shape removes a known bias term; CRB gives a principled statistical floor | P&S on *shape* bias; **must be demonstrated**, not assumed |
| **Sweep time for that σ** | ~1 hr / ~60 pts (20 Hz–1.2 kHz); **hours/point >1 kHz** [8] | All bins measured simultaneously in one window; CRB concentrates the budget where it matters | P&S — biggest expected gain is **>1 kHz** and total wall‑clock |
| **Line/comb contamination of the strain band** | 4–5 narrow lines, subtractable [4,5,6] | A *comb* sits in band — more lines. Mitigate: run in non‑observing time (as the broadband cross‑check already does [8]), or known‑phase ⇒ subtractable, or use only for the periodic full measurement | **Baseline** — this is P&S's real cost; quantify the footprint honestly |
| **Non‑stationarity** (`f_cc`, optical gain, ESD charge drift within a sweep) | Mitigated by continuous TDCF lines; but a 1‑hr sweep spans real drift [5] | Whole TF in **one short stationary window**; periodic averaging + LPM give a nonparametric noise estimate and tolerate smooth drift | **P&S** — structurally the right tool |
| **Systematic vs statistical separation** | MCMC + GP residual (GP term dominates) [1] | Leakage‑free ⇒ smaller method bias near `f_cc`; CRB ⇒ explicit statistical envelope | P&S, *if* the model is right (same model‑error exposure as today) |
| **Pcal absolute traceability** | Radiation‑pressure standard, 0.3–0.15% (O4) [9] | **Unchanged** — injected through the same Pcal | Tie |

**Honest accounting of where P&S does *not* help.** The absolute scale is Pcal either way — P&S
buys nothing there. If the *parametric model* of `C`/`A` is wrong, P&S inherits exactly the same
systematic as the swept sweep (it fits the same model). And the comb's in‑band footprint is a
genuine operational cost the swept sweep largely avoids during observation. The pitch is **shape
fidelity + measurement‑time efficiency + non‑stationarity robustness**, not absolute accuracy.

---

## 5. Why this is worth a real test, in one paragraph

The two things that limited O3b — *frequency‑dependent modelling of `R(f)`* [3] and the
time‑for‑accuracy cost above 1 kHz [8] — are precisely the two things a leakage‑free,
optimally‑excited, periodically‑averaged measurement is designed to improve, at no cost to the
Pcal standard. The method is not exotic: broadband Pcal injection already exists [8]; P&S simply
makes it *periodic* (leakage‑free), *optimally shaped* (CRB), and *self‑error‑barred* (LPM),
instead of unoptimised colored noise used only offline.

---

## 6. Gaps and what must be validated before claiming anything

- **No prior art.** The literature search found **zero** applications of Pintelon–Schoukens
  optimal multisine / LPM to GW‑detector calibration. The existing broadband Pcal injection is
  *not* this. State the novelty precisely: not "broadband injection" (old), but "optimal‑excitation
  frequency‑domain ID with quantified efficiency and a measured noise model" (absent). [8,15]
- **The efficiency claim must be quantified, not asserted.** Spreading power across a comb lowers
  per‑bin SNR; the win comes from simultaneity + CRB allocation + leakage‑free estimation. Whether
  the *net* `σ(R(f))`‑per‑hour beats the swept sweep is an empirical question — run both on the
  same interferometer state and compare envelopes.
- **Comb footprint** (number/amplitude/duty cycle of lines, subtractability during observation)
  must be specified and shown acceptable, or the method is restricted to non‑observing time.
- **Citations to verify before circulation** (flagged by the source search): Wade O4 (arXiv:
  2508.08423) author list/venue; Bhattacharjee 2022 (2207.00621) title/authors; Vitale 2021
  (2009.10192) and Payne 2020 (2009.10193); the Cramér–Rao‑requirements paper (1712.09719) first
  author; and the **exact O3a calibration‑line table and Eq. numbers** in Tuyenbayev [5] / Sun [2]
  against the published PDFs. No finalised peer‑reviewed **O4** systematic‑error paper was found —
  O4 numbers above are *targets* from the methods preprint [6], not achieved results.

---

## References

1. Cahillane et al., "Calibration uncertainty for Advanced LIGO's first and second observing
   runs," *Phys. Rev. D* **96**, 102001 (2017). arXiv:1708.03023.
2. Sun et al., "Characterization of systematic error in Advanced LIGO calibration,"
   *Class. Quantum Grav.* **37**, 225008 (2020). arXiv:2005.02531.
3. Sun et al., "…in the second half of O3," (2021). arXiv:2107.00129.
4. Viets et al., "Reconstructing the calibrated strain signal in the Advanced LIGO detectors,"
   *Class. Quantum Grav.* **35**, 095015 (2018). arXiv:1710.09973.
5. Tuyenbayev et al., "Improving LIGO calibration accuracy by tracking and compensating for slow
   temporal variations," *Class. Quantum Grav.* **34**, 015002 (2017). arXiv:1608.05134.
6. Wade et al., "Toward low‑latency, high‑fidelity calibration of the LIGO detectors…," (2025).
   arXiv:2508.08423. *(author list/venue unverified)*
7. Bhattacharjee et al., "…time‑dependent filters," (2022). arXiv:2207.00621. *(unverified)*
8. Karki et al., "The Advanced LIGO photon calibrators," *Rev. Sci. Instrum.* **87**, 114503
   (2016). arXiv:1608.05055.
9. Bhattacharjee et al., "Fiducial displacements with improved accuracy…," *Class. Quantum Grav.*
   **38**, 015009 (2021).
10. Goetz, Savage et al., "Accurate calibration of test mass displacement…," *Class. Quantum Grav.*
    **27**, 084024 (2010). arXiv:0911.0853.
11. Essick, "Calibration uncertainty's impact on gravitational‑wave observations," *Phys. Rev. D*
    **105**, 082002 (2022).
12. Vitale et al., "Physical approach to the marginalization of LIGO calibration uncertainties,"
    *Phys. Rev. D* **103**, 063016 (2021). arXiv:2009.10192. *(unverified)*
13. Payne et al., "Gravitational‑wave astronomy with a physical calibration model," *Phys. Rev. D*
    **102**, 122004 (2020). arXiv:2009.10193.
14. Hall et al., "Systematic calibration error requirements… via the Cramér–Rao bound," (2017).
    arXiv:1712.09719. *(first author unverified)*
15. Pintelon & Schoukens, *System Identification: A Frequency Domain Approach*, 2nd ed.,
    Wiley/IEEE Press (2012).
