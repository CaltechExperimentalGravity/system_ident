"""Websocket protocol for the dashboard.

The server PUSHES per-iteration snapshots (current model, measured response,
coherence, designed excitation ASD, convergence uncertainty, drive level,
output RMS) to the browser, and receives a STOP control message back. No
polling.

This module is dependency-free (no fastapi/websockets) so it can be imported
and tested anywhere; the transport lives in :mod:`system_ident.dashboard.server`.
The snapshot is the contract between :meth:`system_ident.loop.SysIDLoop._emit`
and the browser.
"""

from __future__ import annotations

import json

# Keys in every per-iteration snapshot pushed to the browser. Must match the
# dict produced by SysIDLoop._emit.
SNAPSHOT_FIELDS = (
    "iteration",
    "dof",
    "freq",
    "model_num",
    "model_den",
    "model_mag",
    "meas_mag",
    "coherence",
    "excitation_asd",
    "max_frac_uncertainty",
    "drive_level",
    "output_rms",
)

# Control messages the browser may send back.
CONTROL_STOP = "stop"


def validate_snapshot(snapshot: dict) -> dict:
    """Return ``snapshot`` if it carries exactly the expected fields, else raise."""
    missing = set(SNAPSHOT_FIELDS) - snapshot.keys()
    extra = snapshot.keys() - set(SNAPSHOT_FIELDS)
    if missing or extra:
        raise ValueError(
            f"bad snapshot (missing={sorted(missing)}, unexpected={sorted(extra)})"
        )
    return snapshot


def to_json(snapshot: dict) -> str:
    """Serialise a (validated) snapshot to a websocket text frame."""
    return json.dumps(validate_snapshot(snapshot))


def parse_control(message: str) -> str | None:
    """Parse a control frame from the browser; returns ``CONTROL_STOP`` or None."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return None
    action = data.get("action") if isinstance(data, dict) else None
    return CONTROL_STOP if action == CONTROL_STOP else None
