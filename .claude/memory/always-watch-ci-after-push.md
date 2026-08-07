---
name: always-watch-ci-after-push
description: "HARD RULE — after EVERY push, immediately launch a background `gh run watch` on the triggered CI run"
metadata: 
  node_type: memory
  type: feedback
---

Every push to `main` triggers CI (build + deploy to GitHub Pages). **Every single time** CI runs,
immediately launch a **background** job to watch it.

**Match the run to the pushed SHA — do NOT `gh run list --limit 1` right after a push.** GitHub
often hasn't registered the new run yet, so `--limit 1` returns the *previous* commit's (already
green) run and the watch exits in seconds against the wrong run. Poll for the run whose `headSha`
equals HEAD, then watch that id:

```
SHA=$(git rev-parse HEAD)
for i in $(seq 1 20); do
  RID=$(gh run list --limit 10 --json databaseId,headSha -q "[.[]|select(.headSha==\"$SHA\")][0].databaseId")
  [ -n "$RID" ] && break; sleep 3
done
gh run watch "$RID" --exit-status
```

Run the watch with `run_in_background: true` so it doesn't block, and report pass/fail when the
notification fires. Do this without being asked — after every push, no exceptions. (Watch exiting
in a few seconds = you grabbed a stale run; re-resolve by SHA.)

**Why:** standing project rule, stated emphatically and more than once: every time CI runs, a
background `gh` watch goes with it — every time. A silent red CI that deploys a broken site should
surface on its own, not wait for the next manual check.

**How to apply:** `gh` is installed and authenticated. After each `git
push`, launch the background watch in the same turn. If a run fails, fetch `gh run view --log-failed`
and fix before moving on. (`glab` is also present but this repo is on GitHub — use `gh`.)
