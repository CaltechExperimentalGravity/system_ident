"""``import system_ident`` (and its CDS transport module) must never require
``awg``/``cdsutils``/``gpstime`` -- none is installed on the dev machine, and
the twin path must stay usable on a plain scientific-Python stack.

Style of the sibling repo's ``test/test_cli.py:77-81``. Runs in a subprocess
so a previous test's ``sys.modules`` pollution (e.g. Stage A's fake stubs)
can never hide a real regression here.
"""
from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> None:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_system_ident_does_not_import_cds_libs():
    _run(
        "import system_ident, sys\n"
        "assert 'awg' not in sys.modules\n"
        "assert 'cdsutils' not in sys.modules\n"
        "assert 'gpstime' not in sys.modules\n"
    )


def test_import_cds_transport_module_does_not_import_cds_libs():
    _run(
        "import system_ident.backends.cds_transport, sys\n"
        "assert 'awg' not in sys.modules\n"
        "assert 'cdsutils' not in sys.modules\n"
        "assert 'gpstime' not in sys.modules\n"
    )


def test_awgndstransport_construction_is_the_only_thing_that_imports_them():
    """Importing the class must not import the libs; constructing an
    instance is what does -- verified via the Stage A fakes (tests/_fake_cds.py)
    rather than raw stub modules, so this also confirms AWGNDSTransport() runs
    cleanly against them, not just that some module ends up in sys.modules."""
    _run(
        "import os; os.environ['IFO'] = 'X1'\n"
        "import sys; sys.path.insert(0, 'tests')\n"
        "from system_ident.backends.cds_transport import AWGNDSTransport\n"
        "assert 'awg' not in sys.modules\n"
        "import _fake_cds\n"
        "with _fake_cds.install():\n"
        "    AWGNDSTransport()\n"
        "    assert 'awg' in sys.modules and 'cdsutils' in sys.modules\n"
        "assert 'awg' not in sys.modules  # install() restores sys.modules on exit\n"
    )
