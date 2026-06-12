"""Input-design strategies — pluggable ways to compute the next excitation.

The second genuinely-swappable axis (alongside :mod:`ligo_sysid.estimators`).
Concrete strategies wrap existing repo code behind
:class:`~ligo_sysid.design.base.InputDesigner`.
"""

from .base import InputDesigner

__all__ = ["InputDesigner"]
