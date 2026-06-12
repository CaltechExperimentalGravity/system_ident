"""Input-design strategies — pluggable ways to compute the next excitation.

The second genuinely-swappable axis (alongside :mod:`system_ident.estimators`).
Concrete strategies wrap existing repo code behind
:class:`~system_ident.design.base.InputDesigner`.
"""

from .base import InputDesigner

__all__ = ["InputDesigner"]
