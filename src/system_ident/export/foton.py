"""Foton export of the fitted plant.

Reuses the ``ss2tf -> tf2zpk -> Foton zpk-string`` conversion from
``CDS_Interface/SS2Foton.py`` and finishes the ``foton.FilterFile`` write that
``CDS_Interface/coeff2Foton.py`` only sketched. ``foton`` is lazy-imported (a
CDS-only dependency). Export is a feature, not an acceptance gate. Implemented
in build step 9.
"""

from __future__ import annotations

from ..model import TFModel


def model_to_foton_zpk(model: TFModel) -> str:
    """Render a TFModel as a Foton ``zpk(...)`` design string. (build step 9)"""
    raise NotImplementedError("model_to_foton_zpk lands in build step 9")


def write_foton_filter(model: TFModel, filter_file: str, module: str) -> None:
    """Write the model into a Foton filter file/module. (build step 9)"""
    raise NotImplementedError("write_foton_filter lands in build step 9")
