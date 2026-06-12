"""Command-line interface — the primary way to drive a run.

The CLI is a first-class product (not a thin wrapper over the library API): a
YAML config is the base, flags override it, and hardware runs require a guided
confirm-before-inject step. Today the digital-twin path runs end-to-end; the
CDS-hardware path and the live dashboard are wired in later build steps.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading

from . import __version__
from .config import ConfigError, RunConfig
from .loop import SysIDLoop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ligo-sysid",
        description="Real-time optimal-excitation system ID for LIGO suspensions.",
    )
    parser.add_argument("--version", action="version", version=f"ligo-sysid {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a system-ID campaign from a config file.")
    run.add_argument("config", help="Path to the run YAML config.")
    run.add_argument("--twin", action="store_true",
                     help="Use the digital-twin backend instead of CDS hardware.")
    run.add_argument("--estimator", help="Override the estimator strategy.")
    run.add_argument("--duration", type=float, help="Override per-segment duration [s].")
    run.add_argument("--px-total", type=float, dest="px_total",
                     help="Override the total input-power budget.")
    run.add_argument("--no-dashboard", action="store_true",
                     help="Run headless (no live dashboard).")
    run.add_argument("--port", type=int, default=8000,
                     help="Port for the live dashboard.")
    run.add_argument("--seed", type=int, default=0,
                     help="Seed for the twin's sensor noise / excitation.")
    run.add_argument("--yes", action="store_true",
                     help="Skip the confirm-before-inject prompt.")
    return parser


def _run(args) -> int:
    try:
        config = RunConfig.load(args.config)
        config.apply_overrides(
            estimator=args.estimator,
            segment_duration=args.duration,
            px_total=args.px_total,
        )
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2

    if not args.twin:
        print(
            "the CDS-hardware backend is not available yet; run with --twin to use "
            "the digital twin.",
            file=sys.stderr,
        )
        return 2

    try:
        backend = config.build_twin_backend(seed=args.seed)
        priors = config.build_priors()
        watchdog = config.build_watchdog(backend)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2

    dofs = list(priors)
    print(f"running twin campaign: {len(dofs)} DoF(s) {dofs}")
    if not args.yes and not _confirm(twin=True):
        print("aborted before injection.")
        return 1

    listener = _maybe_start_dashboard(args, watchdog)

    loop = SysIDLoop(
        backend, config.build_estimator(), config.build_designer(),
        watchdog, listener=listener,
    )
    result = loop.run(config.raw, priors, seed=args.seed)
    return _report(result)


def _maybe_start_dashboard(args, watchdog):
    """Start the live dashboard server if requested + available; else headless.

    Returns the loop ``listener`` (the hub's publish) or ``None``.
    """
    if args.no_dashboard:
        return None
    if not (importlib.util.find_spec("fastapi") and importlib.util.find_spec("uvicorn")):
        print("note: dashboard extra not installed "
              "(pip install ligo-sysid[dashboard]); running headless.")
        return None

    from .dashboard.server import SnapshotHub, serve

    hub = SnapshotHub()
    threading.Thread(
        target=serve,
        kwargs={"hub": hub, "on_stop": lambda: watchdog.abort("operator STOP"),
                "port": args.port},
        daemon=True,
    ).start()
    print(f"dashboard: http://127.0.0.1:{args.port}")
    return hub.publish


def _confirm(twin: bool) -> bool:
    label = "twin" if twin else "HARDWARE"
    try:
        reply = input(f"about to inject excitation on the {label}. type 'yes' to proceed: ")
    except EOFError:
        return False
    return reply.strip().lower() == "yes"


def _report(result) -> int:
    if result.aborted:
        print(f"ABORTED: {result.abort_reason}")
        print("safe-state handoff completed.")
        return 1
    status = "DONE (target reached)" if result.done else "stopped (iteration budget)"
    print(status)
    last = {}
    for rec in result.history:
        last[rec.dof] = rec.max_frac_uncertainty
    for dof, frac in last.items():
        print(f"  {dof}: max fractional uncertainty = {frac:.3g}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        return _run(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
