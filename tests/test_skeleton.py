"""Step-1 smoke tests: the package and its skeleton import cleanly.

These assert structure only — concrete numeric behaviour is added (and tested)
in later build steps.
"""

import numpy as np
import pytest

import ligo_sysid
from ligo_sysid import ChannelBackend, Estimator, InputDesigner, TFModel


def test_version():
    assert isinstance(ligo_sysid.__version__, str)


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
        "ligo_sysid.model",
        "ligo_sysid.fisher",
        "ligo_sysid.excitation",
        "ligo_sysid.safety",
        "ligo_sysid.loop",
        "ligo_sysid.config",
        "ligo_sysid.cli",
        "ligo_sysid.estimators.invfreqs",
        "ligo_sysid.estimators.weighted_ls",
        "ligo_sysid.estimators.gml",
        "ligo_sysid.estimators.vectfit",
        "ligo_sysid.design.pintelon",
        "ligo_sysid.design.sho",
        "ligo_sysid.backends.twin",
        "ligo_sysid.backends.cds",
        "ligo_sysid.export.foton",
        "ligo_sysid.dashboard.server",
        "ligo_sysid.dashboard.ws",
    ]:
        importlib.import_module(name)


def test_cli_version(capsys):
    # `ligo-sysid --version` exits 0 and prints the version.
    from ligo_sysid.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert ligo_sysid.__version__ in capsys.readouterr().out
