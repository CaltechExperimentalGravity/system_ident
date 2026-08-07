---
name: crest-factor-lives-at-the-dac
description: Schroeder/crest-factor only matters at the DAC after whitening; plant-referred crest optimization is meaningless
metadata: 
  node_type: memory
  type: project
---

The actuator's hard saturation limit in LIGO is at the **DAC** (DAC counts/volts), and there
are **whitening + actuation filters between the digital excitation request and the DAC**. Crest
factor is **not invariant under filtering**, so minimizing the multisine's crest factor in
**plant/force-referred units** (what Schroeder phasing does on the excitation as defined) does
**not** reduce the crest factor of the **DAC-referred** waveform — the whitening filter scrambles
the carefully arranged phases, and the DAC crest is generally higher and unrelated.

**Why:** crest factor = peak/RMS is reference-frame dependent. The constraint binds where the
signal saturates (the DAC), so crest must be optimized in the DAC frame, through the whitening
chain — a filter-aware phase/time optimization the `system_ident` package does **not** currently
do. The twin/sim also **omits the whitening filters entirely**, so it cannot even represent the
DAC frame.

**How to apply:** make **no crest-factor / peak-drive advantage claim** from Schroeder phasing or
from plant-referred peak numbers. The multisine's robust, frame-independent wins are: leakage-free
estimation, whole-band simultaneity (the *time* win), a measured per-bin noise model, and
CRB-optimal *allocation* (where in frequency to put power). The *time* win is fully frame-robust;
RMS / in-band-power is plant-referred but defensible; **peak/crest does not transfer to the DAC**.
This generalizes the earlier DARM-specific point in [[stay-on-pintelon-schoukens]]: it's not just
that Pcal isn't force-limited — plant-referred Schroeder helps nowhere there's a whitening chain.
A genuine crest-factor optimization would be future work: minimize the DAC-referred waveform's
crest through the modeled whitening/actuation filters.
