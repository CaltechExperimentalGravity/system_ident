"""Channel backends — the identical-API swap between hardware and the twin.

Deliberately a *thin* interface with exactly two concrete implementations
(:mod:`~ligo_sysid.backends.cds` and :mod:`~ligo_sysid.backends.twin`), not an
extensibility surface. The twin shares the same API so the whole loop, safety
handoff, and dashboard run unchanged in simulation.
"""

from .base import ChannelBackend

__all__ = ["ChannelBackend"]
