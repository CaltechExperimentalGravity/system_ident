---
name: schroeder-has-no-benefit-here
description: HARD RULE — Schroeder (low-crest) phase gives NO benefit in this pipeline; stop claiming it does
metadata: 
  node_type: memory
  type: feedback
---

**Schroeder phase provides no usable benefit in the system_ident pipeline. Do not present it as
helping crest, headroom, or anything else — in the docs, the playground, or analysis.**

Three independent reasons it's a dead end here:
1. The Fisher-optimal drive is essentially **two tones** (N_eff ≈ 2, all power on the modes) — crest
   ≈ 2 for *any* phase, so there is nothing to minimize.
2. We **never drive flat/broadband** (standing rule [[no-flat-noise-drives]]), so the only regime
   where Schroeder's low-crest math does anything never occurs.
3. Crest is a **DAC-referred** limit; the actuation/whitening chain between synthesis and the DAC
   reassigns per-frequency phases, so a crest minimized at synthesis does not hold at the DAC where
   the range actually binds. See [[crest-factor-lives-at-the-dac]].

**Why:** this has been stated as a project rule multiple times — Schroeder keeps getting
hallucinated back in, and it has no benefit here. I repeatedly re-introduced a Schroeder crest
advantage (first "wins both axes," then
"wins broadband") and had to remove it each time. This memory replaces the earlier, WRONG
`schroeder-only-helps-broadband` note.

**How to apply:** Treat Schroeder as generic textbook trivia that is irrelevant to this system. If
crest/headroom comes up, the honest levers are the **spectrum** (a concentrated optimal drive is
naturally low-crest) and the actuator range at the DAC — never a phase trick.
