"""Step-1 smoke tests: the package and its skeleton import cleanly.

These assert structure only — concrete numeric behaviour is added (and tested)
in later build steps.
"""

import numpy as np
import pytest

import system_ident
from system_ident import ChannelBackend, Estimator, InputDesigner, TFModel


def test_version():
    assert isinstance(system_ident.__version__, str)


def test_public_interfaces_are_abstract():
    # The three locked strategy/backend interfaces must be ABCs (not directly
    # instantiable) so concrete implementations are required.
    for abc in (Estimator, InputDesigner, ChannelBackend):
        with pytest.raises(TypeError):
            abc()  # type: ignore[abstract]


def test_tfmodel_dict_roundtrip():
    m = TFModel(num=[1.0], den=[1.0, 0.1, 1.0])
    again = TFModel.from_dict(m.to_dict())
    assert np.allclose(again.num, m.num)
    assert np.allclose(again.den, m.den)


def test_submodules_import():
    # Every skeleton module should import without pulling optional/CDS deps.
    import importlib

    for name in [
        "system_ident.model",
        "system_ident.fisher",
        "system_ident.excitation",
        "system_ident.safety",
        "system_ident.loop",
        "system_ident.config",
        "system_ident.cli",
        "system_ident.estimators.invfreqs",
        "system_ident.estimators.weighted_ls",
        "system_ident.estimators.gml",
        "system_ident.estimators.vectfit",
        "system_ident.design.pintelon",
        "system_ident.design.sho",
        "system_ident.backends.twin",
        "system_ident.backends.cds",
        "system_ident.export.foton",
        "system_ident.dashboard.server",
        "system_ident.dashboard.ws",
    ]:
        importlib.import_module(name)


def test_cli_version(capsys):
    # `system_ident --version` exits 0 and prints the version.
    from system_ident.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert system_ident.__version__ in capsys.readouterr().out
