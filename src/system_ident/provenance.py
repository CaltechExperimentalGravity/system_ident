"""Provenance for physical numbers, so an invented value cannot ship silently.

Every physical *input* used to produce a result must declare where it came from. Values are
registered through :func:`record`, which forces a provenance ``kind``; :func:`require_grounded`
(run in CI, see ``tests/test_provenance.py``) fails the build if any registered value is
``ASSUMED`` — a placeholder standing in for a real number we do not have — unless it is on an
explicit, reviewed allow-list.

Honest scope (do not overclaim this): the guarantee covers only values routed through
:func:`record`. It CANNOT detect a bare literal hard-coded inline and never registered. Its job is
to (a) force provenance on the physical inputs that matter, (b) make every placeholder visible in
one place (and in the rendered docs' "Assumptions" box), and (c) make shipping an unreviewed
placeholder a hard CI failure. The discipline it supports: route physical INPUT parameters — masses,
actuator ranges, powers, sensitivity anchors, representative drive amplitudes — through
``record()``. Ordinary math constants, array sizes, tolerances, and filter coefficients do not need
it. When you do not have a real number, register it ``ASSUMED`` (it will fail CI until allow-listed
or replaced) — never dress a guess up as ``MEASURED``/``PAPER``/``USER``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Provenance kinds, most-trusted first.
MEASURED = "measured"    # from real instrument data / a data file
PAPER = "paper"          # from a cited publication (put the citation in `source`)
USER = "user"            # given by a domain expert (put who + when in `source`)
CONSTANT = "constant"    # a physical or mathematical constant
DERIVED = "derived"      # computed from other recorded values (name them in `source`)
ASSUMED = "assumed"      # PLACEHOLDER: a real value we do NOT have. CI flags/fails on these.

_KINDS = {MEASURED, PAPER, USER, CONSTANT, DERIVED, ASSUMED}


@dataclass(frozen=True)
class Entry:
    name: str
    value: float
    kind: str
    source: str
    unit: str = ""
    note: str = ""


_REGISTRY: dict[str, "Entry"] = {}


def record(name: str, value: float, kind: str, source: str = "", *,
           unit: str = "", note: str = "") -> float:
    """Register a physical input with its provenance and return ``value`` unchanged.

    Use inline at the point of definition, e.g.
    ``TEST_MASS_KG = record("test_mass_kg", 40.0, PAPER, "aLIGO quad final mass", unit="kg")``.
    A non-``CONSTANT`` value must name a real ``source``; an ``ASSUMED`` value is allowed to be
    registered (so placeholders are explicit and discoverable) but will fail
    :func:`require_grounded` until it is replaced with a real number or deliberately allow-listed.
    """
    if kind not in _KINDS:
        raise ValueError(f"{name!r}: unknown provenance kind {kind!r}; use one of {sorted(_KINDS)}")
    if kind != CONSTANT and not source:
        raise ValueError(f"{name!r}: a {kind} value must name a real source (who/what/where)")
    _REGISTRY[name] = Entry(name, float(value), kind, source, unit, note)
    return value


def registry() -> dict[str, "Entry"]:
    """A copy of all registered entries."""
    return dict(_REGISTRY)


def assumed() -> dict[str, "Entry"]:
    """The entries currently marked ``ASSUMED`` (placeholders for numbers we do not have)."""
    return {n: e for n, e in _REGISTRY.items() if e.kind == ASSUMED}


def require_grounded(allow: frozenset[str] = frozenset()) -> None:
    """Raise ``AssertionError`` if any registered value is ``ASSUMED`` and not in ``allow``.

    ``allow`` is a deliberate, reviewed acknowledgement that a named placeholder is knowingly in
    use; allow-listed placeholders still surface in :func:`assumptions_table`. Call this in CI.
    """
    bad = {n: e for n, e in assumed().items() if n not in allow}
    if bad:
        lines = "\n".join(
            f"  - {e.name} = {e.value:g} {e.unit}  [{e.source or 'no source'}]  {e.note}".rstrip()
            for e in bad.values())
        raise AssertionError(
            "ASSUMED (placeholder / not-a-real-measurement) values may not ship. "
            "Replace with a real number, or explicitly allow-list after review:\n" + lines)


def assumptions_table() -> str:
    """A markdown table of every registered value and its provenance — drop into a rendered page so
    readers can tell real inputs from placeholders at a glance."""
    if not _REGISTRY:
        return "_(no physical inputs registered)_"
    rows = ["| input | value | provenance | source |", "| --- | --- | --- | --- |"]
    for e in _REGISTRY.values():
        flag = " ⚠️ **PLACEHOLDER**" if e.kind == ASSUMED else ""
        rows.append(f"| `{e.name}` | {e.value:g} {e.unit} | {e.kind}{flag} | {e.source} |")
    return "\n".join(rows)
