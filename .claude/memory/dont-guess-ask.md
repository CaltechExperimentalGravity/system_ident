---
name: dont-guess-ask
description: "HARD RULE — do not guess/assume, especially LIGO CDS/operational specifics; ask short questions instead"
metadata: 
  node_type: memory
  type: feedback
---

Do **not** guess or assume your way into long speculative answers. The user (a LIGO
expert) finds it actively depressing and a waste of his paid time when I fabricate
domain detail — especially about **LIGO CDS / control-room operations** (awg, nds2,
Foton, GPS timing, channels, sites, hardware) — and present it confidently as if I know it.

**Why:** I do not know the CDS/operational reality; he does, far better. Inventing it is
worse than useless — it's noise he has to wade through and correct.

**How to apply:**
- When I don't actually know something, **ask a short, direct question** — do not produce a
  wall of plausible-sounding assumptions or lecture him on his own domain.
- Default to brevity. One or two real questions beat a long speculative menu of options I made up.
- Distinguish what I can verify (this repo's Python: package code, docs, tests) from what I
  cannot (how CDS/hardware actually behaves). State the former, ask about the latter.
- This extends [[never-silently-reverse-user-commands]]: don't substitute my guess for his intent.
- **Once he decides, commit — don't re-litigate.** When he's made a call (e.g. "need slycot"), stop
  offering fallbacks/escape hatches or re-pitching alternatives. That reads as trying to change his
  mind and irritates. Record the decision and move on.
