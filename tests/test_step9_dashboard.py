"""Step-9: dashboard protocol, snapshot hub, page, and loop wiring.

These exercise the dependency-free core. The FastAPI transport is only
imported (and asserted to fail cleanly) according to whether the dashboard
extra is installed in the running environment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import system_ident
from system_ident.config import RunConfig
from system_ident.dashboard import server, ws
from system_ident.loop import SysIDLoop
from system_ident.safety import SafetyAbort

DEMO = Path(system_ident.__file__).parent / "configs" / "twin_demo.yml"
HAVE_FASTAPI = importlib.util.find_spec("fastapi") is not None


def _good_snapshot():
    return {f: 0 for f in ws.SNAPSHOT_FIELDS}


# --- protocol -----------------------------------------------------------------
def test_validate_snapshot_roundtrip():
    snap = _good_snapshot()
    assert ws.validate_snapshot(snap) is snap
    assert json.loads(ws.to_json(snap)).keys() == snap.keys()


def test_validate_snapshot_rejects_bad_fields():
    with pytest.raises(ValueError, match="missing"):
        ws.validate_snapshot({"iteration": 0})
    extra = _good_snapshot() | {"surprise": 1}
    with pytest.raises(ValueError, match="unexpected"):
        ws.validate_snapshot(extra)


def test_parse_control():
    assert ws.parse_control(json.dumps({"action": "stop"})) == ws.CONTROL_STOP
    assert ws.parse_control(json.dumps({"action": "nope"})) is None
    assert ws.parse_control("not json") is None


# --- hub ----------------------------------------------------------------------
def test_hub_fanout_and_unsubscribe():
    hub = server.SnapshotHub()
    seen = []
    unsub = hub.subscribe(seen.append)
    assert hub.subscriber_count == 1

    hub.publish(_good_snapshot())
    assert hub.latest == _good_snapshot()
    assert len(seen) == 1

    unsub()
    assert hub.subscriber_count == 0
    hub.publish(_good_snapshot())
    assert len(seen) == 1  # no longer notified


def test_hub_rejects_bad_snapshot():
    with pytest.raises(ValueError):
        server.SnapshotHub().publish({"iteration": 0})


# --- HTML page ----------------------------------------------------------------
def test_render_html_has_expected_markers():
    html = server.render_html()
    for marker in ("Plotly", "WebSocket", "STOP", "/ws", "bode", "coh", "exc"):
        assert marker in html


# --- loop wiring --------------------------------------------------------------
def _loop_with_listener(listener, seed=0):
    cfg = RunConfig.load(DEMO)
    backend = cfg.build_twin_backend(seed=seed)
    wd = cfg.build_watchdog(backend)
    loop = SysIDLoop(
        backend, cfg.build_estimator(), cfg.build_designer(), wd, listener=listener
    )
    return loop, cfg, wd


def test_loop_emits_valid_snapshots_to_hub():
    hub = server.SnapshotHub()
    captured = []
    hub.subscribe(captured.append)
    loop, cfg, _ = _loop_with_listener(hub.publish)

    result = loop.run(cfg.raw, cfg.build_priors(), seed=0)

    assert not result.aborted
    assert captured  # snapshots were pushed
    for snap in captured:
        ws.validate_snapshot(snap)        # exact field set
        json.loads(ws.to_json(snap))      # JSON-serialisable
    assert {s["dof"] for s in captured} == {"POS", "PIT"}
    assert hub.latest == captured[-1]


def test_operator_stop_via_listener_aborts_with_handoff():
    # a listener that issues STOP on the first snapshot; the loop must notice
    # between segments and shut down through the safe-state handoff
    state = {"wd": None}

    def stopper(_snap):
        state["wd"].abort("operator STOP")

    loop, cfg, wd = _loop_with_listener(stopper)
    state["wd"] = wd

    result = loop.run(cfg.raw, cfg.build_priors(), seed=0)
    assert result.aborted
    assert result.abort_reason == "operator STOP"


# --- transport (env-dependent) ------------------------------------------------
@pytest.mark.skipif(HAVE_FASTAPI, reason="dashboard extra installed")
def test_create_app_without_extra_raises_install_hint():
    with pytest.raises(ModuleNotFoundError, match="dashboard extra"):
        server.create_app(server.SnapshotHub())


@pytest.mark.skipif(not HAVE_FASTAPI, reason="dashboard extra not installed")
def test_create_app_builds_when_extra_present():
    app = server.create_app(server.SnapshotHub())
    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/" in routes and "/ws" in routes
