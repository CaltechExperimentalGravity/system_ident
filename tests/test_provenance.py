"""CI gate: no fabricated ('ASSUMED') physical number may ship.

Importing the physics modules runs their `provenance.record(...)` calls; `require_grounded()` then
fails if any registered value is a placeholder standing in for a real number we do not have. This is
the teeth behind "never make up numbers": a guess can be committed to the code only as an explicit
ASSUMED entry, and that fails the build until it is replaced or deliberately allow-listed.
"""
import importlib

import pytest

from system_ident import provenance as prov

# Modules whose physical inputs must be grounded. Importing them populates the registry.
_PHYSICS_MODULES = ["system_ident.darm", "system_ident.darm_callines"]

# Reviewed, deliberately-accepted placeholders (empty on purpose: nothing should be assumed).
_ALLOWED_ASSUMED = frozenset()


@pytest.fixture(scope="module", autouse=True)
def _load_physics():
    for m in _PHYSICS_MODULES:
        importlib.import_module(m)


def test_no_ungrounded_assumed_values():
    prov.require_grounded(allow=_ALLOWED_ASSUMED)


def test_every_registered_value_has_a_recognised_kind():
    for e in prov.registry().values():
        assert e.kind in prov._KINDS
        if e.kind != prov.CONSTANT:
            assert e.source, f"{e.name}: non-constant value with no source"
