---
name: verify-model-components-real
description: "Don't fabricate physics — verify a noise source / model component is real (in the actual model/code) before treating it as physical"
metadata: 
  node_type: memory
  type: feedback
---

Do NOT invent or assume physical components. A prior session bolted a fictitious "16-bit
±1 mm ADC" quantizer onto the recorded readout (meter-domain, out-of-loop) — not part of the
RTSfreerun twin at all. It corrupted the high-SNR FRF bins and triggered a long wrong-headed
chase. The user: "why does this happen?" and "stop sabotaging my research."

**Why:** fabricated physics produces results that are wrong AND embarrassing to present;
worse, it sends debugging down ghost trails (I made TWO confident-but-wrong causal claims
about that ADC before checking).

**How to apply:**
- Before treating any noise source / quantizer / disturbance as physical, confirm it exists
  in the actual model or twin (grep the code, read the loader, eigendecompose the plant) —
  don't trust report prose or a prior session's comment.
- When the user questions a premise, STOP and re-examine the assumption in-context and answer
  — do not run off and spawn more compute to defend the theory. A pointed question is a
  brake, not a work order.
- State causal claims only after verifying by computation. See
  [[compute-the-bound-before-claiming-a-limit]] and [[dont-guess-ask]].
