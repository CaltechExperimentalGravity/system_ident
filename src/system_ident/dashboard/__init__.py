"""Local live dashboard: FastAPI + websockets + Plotly.js.

An *operational monitor* (not an analysis workbench): the sysID loop pushes a
per-iteration snapshot over a websocket, the browser renders it with Plotly.js,
and a single STOP button triggers the safe-state handoff. The dashboard stack
(fastapi/uvicorn/websockets) is an optional extra and is imported lazily by
:mod:`~system_ident.dashboard.server`. Implemented in build step 6.
"""
