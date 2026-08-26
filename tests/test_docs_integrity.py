"""Guards on the hand-written docs, so the published site cannot rot silently.

These are pure-stdlib and read the ``.qmd`` sources only — no Quarto, no docs
extras — so they run in the light ``test`` CI job on every push, not just in the
docs build.

Three failure modes have actually bitten this repo, one test each:

1. A path named in prose that does not exist. ``configs/closed_loop_demo.yml``
   was printed as a runnable command on two pages for months; the configs live
   under ``src/system_ident/configs/``, so anyone who copy-pasted got a
   ``ConfigError``.
2. A cross-page link or anchor that does not resolve, usually after a heading is
   retitled and the referring page is not updated.
3. A ``freeze: true`` page whose source moved on without being re-executed.
   Quarto's freeze caches the executed markdown *including the prose*, so such a
   page keeps serving its old text and old figures forever — the site and the
   repo disagree and nothing in the build says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: The generated API reference (``quartodoc build``) is gitignored and may be
#: absent; its prose lives in ``src/`` docstrings, so it is not ours to check.
PAGES = sorted(p for p in DOCS.rglob("*.qmd") if "reference" not in p.parts)

pytestmark = pytest.mark.skipif(not PAGES, reason="docs/ not present in this checkout")


# ── shared parsing helpers ──────────────────────────────────────────────────

_FENCE = re.compile(r"^\s*(```|~~~)")


def _outside_fences(text: str):
    """Yield ``(lineno, line)`` for lines that are not inside a code fence."""
    fenced = False
    for n, line in enumerate(text.splitlines(), 1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield n, line


def _slug(heading: str) -> str:
    """Pandoc's auto-identifier, close enough for the ASCII+math headings here."""
    t = re.sub(r"\{[^}]*\}", "", heading)          # attribute block
    t = re.sub(r"`([^`]*)`", r"\1", t)             # code spans
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # links keep their text
    t = re.sub(r"\$[^$]*\$", "", t)                # pandoc drops inline math
    t = re.sub(r"[*_]", "", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s-]", "", t.lower())
    t = re.sub(r"\s+", "-", t.strip())
    return re.sub(r"^[^a-z]+", "", re.sub(r"-+", "-", t).strip("-"))


def _anchors(path: Path) -> set[str]:
    """Every id a link can target: explicit ``{#id}``, heading slugs, cell labels."""
    out: set[str] = set()
    text = path.read_text()
    for _, line in _outside_fences(text):
        head = re.match(r"^#{1,6}\s+(.*)$", line)
        if head:
            explicit = re.search(r"\{#([\w:.-]+)", head.group(1))
            out.add(explicit.group(1) if explicit else _slug(head.group(1)))
        out.update(m.group(1) for m in re.finditer(r"\{#([\w:.-]+)", line))
    # `#| label: fig-x` makes a cross-referenceable float, and lives in a fence
    out.update(m.group(1) for m in re.finditer(r"^#\|\s*label:\s*([\w:.-]+)", text, re.M))
    return out


# ── 1. paths named in prose ─────────────────────────────────────────────────

#: Repo-relative path shapes the docs quote as things a reader can run or open.
_PATH_RE = re.compile(
    r"(?<![\w/.-])("
    r"(?:src|tests|configs|binder|docs|talks|notes|experiments)/[\w./-]*"
    r"[\w-]\.(?:ya?ml|py|toml|json|npz|qmd|txt|sh|md)"
    r")"
)


def test_repo_paths_named_in_prose_exist():
    """A path printed in the docs must resolve, or the command around it is a lie."""
    missing = []
    for page in PAGES:
        for lineno, line in _outside_fences(page.read_text()):
            for m in _PATH_RE.finditer(line):
                rel = m.group(1)
                if not (ROOT / rel).exists():
                    missing.append(f"{page.relative_to(ROOT)}:{lineno}  {rel}")
    assert not missing, "paths named in the docs that do not exist:\n  " + "\n  ".join(missing)


# ── 2. internal links ───────────────────────────────────────────────────────

_LINK_RE = re.compile(r"\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
_EXTERNAL = ("http://", "https://", "mailto:", "{{")


def test_internal_links_and_anchors_resolve():
    """Every in-repo link target exists, and every ``#fragment`` is a real anchor."""
    anchors = {p: _anchors(p) for p in PAGES}
    broken = []
    for page in PAGES:
        for lineno, line in _outside_fences(page.read_text()):
            for m in _LINK_RE.finditer(line):
                href = m.group(1)
                where = f"{page.relative_to(ROOT)}:{lineno}  {href}"
                if href.startswith(_EXTERNAL):
                    continue
                if href.startswith("#"):
                    if href[1:] not in anchors[page]:
                        broken.append(f"{where}  -- no such anchor on this page")
                    continue
                file_part, _, frag = href.partition("#")
                target = (page.parent / file_part).resolve()
                if not target.exists():
                    broken.append(f"{where}  -- target does not exist")
                    continue
                if frag and target.suffix == ".qmd" and "reference" not in target.parts:
                    if frag not in anchors.get(target, set()):
                        broken.append(f"{where}  -- no such anchor in the target")
    assert not broken, "broken internal links:\n  " + "\n  ".join(broken)


# ── 3. frozen pages ─────────────────────────────────────────────────────────

def _frozen_pages() -> list[Path]:
    """Pages pinning ``freeze: true`` in their own front matter."""
    out = []
    for page in PAGES:
        head = page.read_text().split("---")
        if len(head) > 1 and re.search(r"^freeze:\s*true\s*$", head[1], re.M):
            out.append(page)
    return out


#: Pages known to be stale that this machine cannot fix, with the reason. These
#: two need the compiled RTSfreerun twin (``x1hsts``) to execute, which is not
#: installable from PyPI and is absent from CI — so their prose and figures sit
#: in the repo waiting for a render on a machine that has it. This is a to-do
#: list, not an exemption: the test below fails if one of them becomes current,
#: which is the signal to delete it from here.
_KNOWN_STALE = {
    "docs/examples/07-rtsfreerun-twin.qmd": "needs the compiled RTSfreerun twin",
    "docs/examples/10-srm-hsts.qmd": "needs the compiled RTSfreerun twin",
}


def _staleness() -> dict[str, str]:
    """``{page: diagnosis}`` for every ``freeze: true`` page that is out of date."""
    stale = {}
    for page in _frozen_pages():
        rel = page.relative_to(DOCS).with_suffix("")
        cache = DOCS / "_freeze" / rel / "execute-results" / "html.json"
        key = str(page.relative_to(ROOT))
        if not cache.exists():
            stale[key] = "freeze:true but no cached result"
            continue
        stored = json.loads(cache.read_text()).get("hash")
        current = hashlib.md5(page.read_bytes()).hexdigest()
        if stored != current:
            stale[key] = f"stale (cached {stored}, source {current})"
    return stale


def test_frozen_pages_match_their_source():
    """A ``freeze: true`` page must have been re-executed since its last edit.

    Quarto keys the freeze cache on the md5 of the ``.qmd``, and the cache holds
    the executed markdown *including the prose*. When the two disagree, the
    published page keeps serving its old text and old figures — editing it has no
    visible effect, and no other part of the build says so. Refresh with

        conda run -n sysid quarto render docs/<page>.qmd --to html

    run from ``docs/``.
    """
    unexpected = {k: v for k, v in _staleness().items() if k not in _KNOWN_STALE}
    assert not unexpected, (
        "frozen pages whose source has moved on since they were last executed:\n  "
        + "\n  ".join(f"{k}  -- {v}" for k, v in unexpected.items())
    )


def test_known_stale_list_has_not_gone_out_of_date():
    """Anything in ``_KNOWN_STALE`` that is now current must be removed from it.

    Otherwise the quarantine outlives the problem and silently exempts a page that
    is once again being checked.
    """
    stale = _staleness()
    fixed = [f"{k}  ({why})" for k, why in _KNOWN_STALE.items() if k not in stale]
    assert not fixed, (
        "these pages are current again — delete them from _KNOWN_STALE:\n  "
        + "\n  ".join(fixed)
    )


def test_every_frozen_page_says_why_it_is_frozen():
    """``freeze: true`` opts a page out of all drift detection, so it needs a reason.

    Without one the pin spreads by copy-paste onto pages that could perfectly well
    re-execute in CI, and each one is a place where the code can change without
    anyone noticing the published numbers no longer follow from it.
    """
    unexplained = [
        p.relative_to(ROOT)
        for p in _frozen_pages()
        if "freeze: true" not in p.read_text().split("---", 2)[-1]
    ]
    assert not unexplained, (
        "pages pinning freeze: true without explaining it in the body:\n  "
        + "\n  ".join(str(p) for p in unexplained)
    )
