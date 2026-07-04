# Twin-fidelity ledger — SRM 6-DoF RTSfreerun

Every noise / disturbance / actuator / sensor component the SRM twin campaign applies, with
its source and honest status: **Physical** (traceable to the real twin/plant), **Referred
approximation** (physical level, but injected at a proxy port the compiled model exposes),
or **Idealization** (a simplification that makes the problem easier than reality).

This ledger exists because a *fabricated* noise source (a 16-bit ±1 mm "ADC" quantizer bolted
onto the recorded readout — not part of the twin) once slipped in and corrupted a campaign
(removed, commit `e8d8358`). Rule: **no noise/disturbance/distortion may be applied that is
not listed here and traceable to the twin.** The guard test `tests/test_twin_fidelity.py`
enforces the mechanical part (no ADC/DAC/quantizer distortion constants; the listed knobs
stay documented).

## Ledger

| Component | Where (`srm6dof_loop.py` unless noted) | Status |
|---|---|---|
| Seismic ASD (`ligo-india`; microseism ~3e-6 @0.15 Hz, NLNM floor) | `_seismic_at_m1_asd` (`SEISMIC_PRESET`) | **Physical** — matches the digital-twin `noise.py`; *but* referred to the coil node `DRIVE_EXC`, not the suspension point (referred approximation, labeled) |
| gnd→M1 via `HSTS_GND_TF` × ISI transmissibility | `_seismic_at_m1_asd` | **Physical** — real `hsts_full.mat` residues + real `ham_isi_transmissibility`; P has zero seismic (no gnd→M1 pitch path — legitimate) |
| OSEM/BOSEM readout noise, `1e-10 m/√Hz`, `1 Hz` knee | `bosem_noise_spec` (`BOSEM_FLOOR`, `BOSEM_KNEE_HZ`) | **Referred approximation** — plausible/physical level, injected at the damper *sensor node* `DAMP_EXC`, not through an in-loop quantized sensor (documented compromise; a `READOUT_NOISE` `cdsFilt` rebuild à la `x1hstsdamped` would make it truly in-loop) |
| Actuator: 30000-count coil limit | asserted in the report; `saturate` clip in the adapter (default off) | **Not enforced in the SRM campaign** — peak drive ~0.07 counts, ~4e5× headroom, so no saturation modeled |
| Uniform **Q = 50 on every mode** | inherited from the `hsts_full.mat` state space | **Idealization** — the real HSTS has per-mode Q; "all targets identical & known" makes Q recovery easier than reality (Phase-B item: per-mode-Q twin variant) |
| 16-bit ±1 mm ADC quantizer | **REMOVED** (`e8d8358`) | **Gone — verified** (guard test asserts no `quantize_readout`/`ADC_*` symbol returns) |
| `cds.py` real-hardware backend | `backends/cds.py` | **Pure stub** — methods `raise NotImplementedError`; honest LIMITATIONS docstring (Phase C) |
| Adapter seismic ASD is a *copy* of the twin's | `backends/rtsfreerun_adapter._seismic_asd` | Values match now; **drift risk** — reconcile periodically |

## Tracked noise/disturbance knobs (must stay documented above)
- `SEISMIC_PRESET` — the ground-motion ASD preset.
- `BOSEM_FLOOR`, `BOSEM_KNEE_HZ` — the OSEM readout-noise floor and knee.

## Forbidden (the guard test fails if any reappears)
- Any `quantize`/`ADC_*`/`DAC_*` distortion constant or function in the twin/experiment.
- Any post-hoc mangling of the recorded readout (round/clip/quantize) that is not a
  traceable, in-loop physical effect.
