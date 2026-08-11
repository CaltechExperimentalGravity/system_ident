---
name: no-foton-export
description: HARD RULE — never build a Foton ZPK/SOS exporter; reading foton banks is fine
metadata:
  type: feedback
---

**Foton EXPORT is out of scope for `system_ident`. Never build it, never re-propose it.**
Confirmed by the user on 2026-08-11: "the FOTON stuff is left for someone else to do." This
had been rejected repeatedly before that, and the note recording it lived only in a
machine-local memory that did not survive — hence this file, in the committed repo memory.

**Why:** delivering an identified plant into a site foton bank is a different job, owned by
someone else. This repo's deliverable is the identified model plus its provenance/manifest
(roadmap Track E), not a drop-in filter file.

**How to apply:**
- Do not write a Foton/`.txt` filter-bank writer, a ZPK→foton serialiser, or a "drop-in
  filter" step — in `CDSBackend`, in a run artifact, in a report, or anywhere else.
- If a plan or handoff note lists Foton export as a step (the 40m Stage-4 list did), treat
  that line as struck and say so rather than silently implementing it.
- **Reading is fine and is load-bearing** — `twin.foton_loader.apply_foton_bank` /
  `readFilter` is how the real L1-MC2 and `OPT_CTRL_SUS*` damping banks get loaded for the
  closed-loop work. The prohibition is on export only.

Related: [[two-phase-cds-plan]], [[dont-guess-ask]], [[never-silently-reverse-user-commands]].
