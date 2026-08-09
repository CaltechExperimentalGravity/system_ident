---
name: never-open-the-users-browser
description: "HARD RULE — never open a rendered page, deck, or URL in the user's browser; report the path and stop"
metadata:
  node_type: memory
  type: feedback
---

**Never open anything in the user's browser.** No `open`, `xdg-open`, `webbrowser.open`,
`quarto preview`, `--browser`, or any command whose side effect is launching or focusing a
browser window. This covers rendered decks, docs sites, HTML reports, and plain URLs.

When something has been rendered or built, **report the file path and stop.** The user opens
it themselves, always.

**Why:** stated directly (2026-08-09) after a slide-deck build ended by announcing "the deck
is open in your browser." Opening a window seizes control of the user's screen and their
window/tab arrangement for a step they have already chosen to do by hand. It is not a
convenience; it is an interruption.

**How to apply:** finish a render by printing the path (`talks/_site/ligo-sysid.html`) and the
re-render command. Prefer `quarto render` over `quarto preview` — preview both serves and opens.
Pass this rule to any dispatched subagent that renders, builds, or publishes anything, since
the same instinct shows up there. Related: [[no-heavy-compute-in-slide-generation]].
