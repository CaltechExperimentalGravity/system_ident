---
name: never-search-outside-repo
description: Hard boundary — never read, list, or grep anything outside /home/rana/GIT/system_ident
metadata:
  type: feedback
---

**Never** search, list, read, or grep any path outside `/home/rana/GIT/system_ident`.
No `find /home/rana`, no `grep -r ~`, no `ls ~/Dropbox`, no wandering into `~/docs/`,
`~/public_html/`, or any other directory — not even to locate a file the user says exists.

**Why:** on 2026-08-12 I ran `find /home/rana -iname "*slide*"` and a home-wide grep while
hunting for a slide deck. The user's reaction: "NO!!!! never look outside of this dir!!! that
was non consensual disk scraping!!!! horrible." Scanning a personal home directory is a
privacy violation regardless of intent, and no task justifies it.

**How to apply:** if something can't be found inside the repo, say so and ask the user for
the path. Do not widen the search radius. The scratchpad dir named in the harness prompt is
fine to write to; everything else outside the repo is off limits.
