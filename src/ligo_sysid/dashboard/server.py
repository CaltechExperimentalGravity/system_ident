"""Dashboard server: a pub/sub hub, the HTML page, and a lazily-built FastAPI app.

The hub and HTML are dependency-free and testable anywhere. ``fastapi`` /
``uvicorn`` are imported lazily inside :func:`create_app` / :func:`serve`, so
importing this module (and the headless / ``--no-dashboard`` paths) never
requires the dashboard extra. Install it with ``pip install ligo-sysid[dashboard]``.

Wiring: pass ``SnapshotHub.publish`` as the :class:`~ligo_sysid.loop.SysIDLoop`
``listener``; every iteration is fanned out to all connected browsers. The
browser's STOP button triggers ``on_stop`` (typically ``watchdog.abort``), which
the loop notices between segments and shuts down through the safe-state handoff.
"""

from __future__ import annotations

from typing import Callable

from .ws import SNAPSHOT_FIELDS, to_json, validate_snapshot

_INSTALL_HINT = (
    "the dashboard extra is not installed; run `pip install ligo-sysid[dashboard]` "
    "(fastapi + uvicorn + websockets)"
)


class SnapshotHub:
    """In-process fan-out of loop snapshots to dashboard subscribers.

    Keeps the most recent snapshot (so a newly-connected browser can render
    immediately) and notifies every subscriber on each publish.
    """

    def __init__(self) -> None:
        self.latest: dict | None = None
        self._subscribers: list[Callable[[dict], None]] = []

    def publish(self, snapshot: dict) -> None:
        """Loop listener entry point: validate, store, and fan out a snapshot."""
        validate_snapshot(snapshot)
        self.latest = snapshot
        for cb in list(self._subscribers):
            cb(snapshot)

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        """Register ``callback``; returns an unsubscribe function."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def render_html() -> str:
    """Static dashboard page: Plotly.js plots fed by the /ws websocket + STOP."""
    fields = ", ".join(repr(f) for f in SNAPSHOT_FIELDS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ligo-sysid live</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; }}
    #status {{ font-weight: 600; }}
    #stop {{ background: #b00; color: #fff; border: 0; padding: .6rem 1rem;
             font-size: 1rem; cursor: pointer; border-radius: 4px; }}
    .plot {{ width: 100%; height: 320px; }}
  </style>
</head>
<body>
  <h1>ligo-sysid &mdash; live system identification</h1>
  <p><span id="status">connecting&hellip;</span>
     &nbsp; <button id="stop">STOP</button></p>
  <div id="bode" class="plot"></div>
  <div id="coh" class="plot"></div>
  <div id="exc" class="plot"></div>
  <script>
    // snapshot fields: {fields}
    const ws = new WebSocket("ws://" + location.host + "/ws");
    const stop = document.getElementById("stop");
    const status = document.getElementById("status");
    stop.onclick = () => ws.send(JSON.stringify({{action: "stop"}}));
    ws.onopen = () => status.textContent = "connected";
    ws.onclose = () => status.textContent = "disconnected";
    ws.onmessage = (ev) => {{
      const s = JSON.parse(ev.data);
      status.textContent = `${{s.dof}} iter ${{s.iteration}} ` +
        `— frac unc ${{s.max_frac_uncertainty.toExponential(2)}}`;
      Plotly.react("bode", [
        {{x: s.freq, y: s.model_mag, name: "model", mode: "lines"}},
        {{x: s.freq, y: s.meas_mag, name: "measured", mode: "markers"}},
      ], {{title: "Transfer function |H|", xaxis: {{type: "log"}},
           yaxis: {{type: "log"}}}});
      Plotly.react("coh", [{{x: s.freq, y: s.coherence, mode: "lines"}}],
        {{title: "Coherence", xaxis: {{type: "log"}}, yaxis: {{range: [0, 1]}}}});
      Plotly.react("exc", [{{x: s.freq, y: s.excitation_asd, mode: "lines"}}],
        {{title: "Designed excitation ASD", xaxis: {{type: "log"}},
          yaxis: {{type: "log"}}}});
    }};
  </script>
</body>
</html>"""


def create_app(hub: SnapshotHub, on_stop: Callable[[], None] | None = None):
    """Build the FastAPI app (lazy import). Raises if the extra is missing."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ModuleNotFoundError as err:  # pragma: no cover - env without extra
        raise ModuleNotFoundError(_INSTALL_HINT) from err

    import asyncio

    from .ws import parse_control

    app = FastAPI(title="ligo-sysid dashboard")
    html = render_html()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return html

    @app.websocket("/ws")
    async def ws_endpoint(socket: WebSocket) -> None:
        await socket.accept()
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # bridge the synchronous hub into this socket's async queue
        unsubscribe = hub.subscribe(
            lambda snap: loop.call_soon_threadsafe(queue.put_nowait, snap)
        )
        try:
            if hub.latest is not None:  # render immediately on connect
                await socket.send_text(to_json(hub.latest))

            async def pump() -> None:
                while True:
                    await socket.send_text(to_json(await queue.get()))

            async def listen() -> None:
                while True:
                    if parse_control(await socket.receive_text()) and on_stop:
                        on_stop()

            await asyncio.gather(pump(), listen())
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()

    return app


def serve(
    hub: SnapshotHub,
    on_stop: Callable[[], None] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Run the dashboard server (lazy import of uvicorn). Blocks."""
    try:
        import uvicorn
    except ModuleNotFoundError as err:  # pragma: no cover - env without extra
        raise ModuleNotFoundError(_INSTALL_HINT) from err
    uvicorn.run(create_app(hub, on_stop), host=host, port=port)
