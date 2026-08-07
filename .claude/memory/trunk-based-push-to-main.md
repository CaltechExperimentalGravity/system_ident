---
name: trunk-based-push-to-main
description: "This project is trunk-based — commit and push directly to main, no PRs, no topic branches"
metadata: 
  node_type: memory
  type: feedback
---

This project does **not** use pull requests. Work directly on `main` and push to
`main` (trunk-based). Do not create feature/topic branches or open PRs.

**Why:** the user's stated workflow (2026-06-18) — "I don't want to use PRs in this
project. we should just keep pushing to main."

**How to apply:**
- Commit on `main`; `git push origin main` directly. This **overrides** the default
  harness guidance to branch before committing on the default branch.
- Don't open PRs or suggest them as the path to land work.
- If work is sitting on a topic branch, fast-forward `main` to it and push, then drop
  the topic branch — don't leave parallel branches around.
- Still only push/commit when the user asks (force-pushes / history rewrites especially
  — confirm those). Related: [[graphics-svg-lfs-only]].
