Agent memory index for `system_ident` — one line per memory, loaded at session start.
Each entry is one durable fact in its own file; write new ones here, never to `~/.claude/projects/*/memory/`.

- [RTSfreerun env strategy](rtsfreerun-env-strategy.md) — separate repos; build RTS models into the sysid env, not system_ident into twin
- [Stay on Pintelon–Schoukens](stay-on-pintelon-schoukens.md) — single P&S pipeline; when stuck, suspect a bug before diverging (flat exc / Track B)
- [Never silently reverse user commands](never-silently-reverse-user-commands.md) — HARD RULE: don't revert/undo a user-directed change without asking approval
- [Graphics: SVG + LFS only](graphics-svg-lfs-only.md) — HARD RULE: every plot is SVG, all graphics in Git LFS
- [Trunk-based, push to main](trunk-based-push-to-main.md) — no PRs/topic branches; commit and push straight to main
- [Use conda run for the sysid env](use-conda-run-sysid-env.md) — run python/quarto via `conda run -n sysid`, not the bare binary
- [Crest factor lives at the DAC](crest-factor-lives-at-the-dac.md) — no Schroeder/peak-drive claims; plant-referred crest ≠ DAC crest (whitening filters)
- [Don't guess — ask](dont-guess-ask.md) — HARD RULE: no speculative walls, esp. LIGO CDS/ops; ask short questions instead of inventing an answer
- [Two-phase CDS plan](two-phase-cds-plan.md) — Phase 1 RTSfreerun (now); HARD RULE: don't discuss Phase 2 real hardware until told
- [Rank-1 modal MIMO fit](rank1-modal-mimo-fit.md) — common-denom B/A fails at 6-DoF (numerators absorb pole error); use rank-1 modal + data-driven peak-pick init (exact from 50% priors)
- [Compute the bound before claiming a limit](compute-the-bound-before-claiming-a-limit.md) — HARD RULE: CRB/SNR/headroom/resolution with numbers before calling anything a limit or asking how to proceed; see `.llm/engineering-practices.md`
- [SRM doublet is spatial](srm-doublet-is-spatial.md) — 0.672/0.676 Hz pair is REAL: two orthogonal HSTS modes (L,P,V plane vs T,R,Y plane); resolve via `fit_block_decoupled`, not frequency super-resolution
- [No flat/noise drives — robust multisine](no-flat-noise-drives.md) — HARD RULE: forbidden to drive flat/broadband or with an "off-res SNR" floor; use P&S optimal/prior_robust multisine
- [Verify model components are real](verify-model-components-real.md) — don't fabricate physics (e.g. the hallucinated ADC); check a "noise source"/component exists in the actual model/code before treating it as physical
- [Schroeder has no benefit here](schroeder-has-no-benefit-here.md) — HARD RULE: Schroeder phase gives NO benefit in this pipeline (few-tone drive + never-broadband + DAC/whitening); stop claiming it does
- [Always watch CI after push](always-watch-ci-after-push.md) — HARD RULE: after EVERY push, immediately launch a background `gh run watch` on the triggered run
- [Use python-control, not hand-rolled](use-python-control-not-hand-rolled.md) — HARD RULE: use python-control for anything the controls lib covers; never reinvent state-space/c2d/FRF/feedback in numpy/scipy
- [No heavy compute in slide generation](no-heavy-compute-in-slide-generation.md) — HARD RULE: build decks from `docs/_freeze` / existing caches; cache-first generators, never recompute what already ran
- [CI stays light](ci-stays-light.md) — heavy tests and renders run locally; measure step timings before proposing any CI speedup
- [Never open the user's browser](never-open-the-users-browser.md) — HARD RULE: never `open`/preview a rendered page or URL; report the path and stop
