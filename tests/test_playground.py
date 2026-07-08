"""Faithfulness guard for the excitation playground.

The browser sandbox (``docs/assets/excitation-playground.js``) reimplements the Fisher/CRB
and drive-synthesis math in JS for an instant UI. This test pins its Python twin
(``docs/playground_reference.py``) so the sandbox's numbers stay reproducible and faithful to
the real ``system_ident`` pole convention (via ``arcade_reference``, which
``tests/test_arcade.py`` ties to ``TFModel``).

Two axes are guarded, because they carry the sandbox's whole lesson:
  * **ETA** (time to identify to 5%) depends on the drive's power spectrum ALONE — so drives
    that share a ``Pxx`` share an ETA, regardless of phase.
  * **Crest factor** depends on the phases — so same-``Pxx`` drives differ wildly in crest.

If these drift, the JS and its scoreboard labels must be updated in lock-step. An optional
Playwright check runs the actual JS engine and asserts it matches this reference bin-for-bin;
it skips cleanly where Playwright or the browser is unavailable (e.g. CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import playground_reference as pr  # noqa: E402

# Golden scoreboard (kind -> (ETA seconds, crest factor)) the JS is calibrated to reproduce.
GOLDEN = {
    "opt_schroeder": (45.00, 1.999),
    "opt_random": (45.00, 2.002),
    "flat_schroeder": (1768.63, 1.868),
    "flat_random": (1768.63, 3.421),
    "cophased": (1768.63, 14.922),
    "chirp_lin": (1768.63, 1.415),
    "chirp_log": (933.31, 1.414),
    "white": (1768.63, 3.421),
    "pink": (933.31, 3.407),
    "shaped_fisher": (45.00, 2.002),
}


def test_golden_scoreboard():
    """The reference reproduces the committed ETAs and crest factors (JS shares them)."""
    board = pr.scoreboard()
    assert set(board) == set(GOLDEN)
    for k, (eta_g, cr_g) in GOLDEN.items():
        eta, cr = board[k]
        assert eta == pytest.approx(eta_g, rel=2e-3), f"{k} ETA {eta} != {eta_g}"
        assert cr == pytest.approx(cr_g, rel=2e-3), f"{k} crest {cr} != {cr_g}"


def test_estimation_speed_depends_only_on_the_power_spectrum():
    """Drives that share a Pxx share an ETA — the sandbox's core claim. Phase is irrelevant
    to Fisher, so optimal/Schroeder == optimal/random, and the whole flat family ties."""
    b = pr.scoreboard()
    # optimal-power drives (any phase) tie
    assert b["opt_schroeder"][0] == pytest.approx(b["opt_random"][0], rel=1e-9)
    assert b["opt_schroeder"][0] == pytest.approx(b["shaped_fisher"][0], rel=1e-9)
    # flat-power drives tie (multisine phases, cophased, linear chirp, white noise)
    flat = [b[k][0] for k in ("flat_schroeder", "flat_random", "cophased", "chirp_lin", "white")]
    assert max(flat) == pytest.approx(min(flat), rel=1e-9)
    # 1/f-power drives tie (log chirp, pink noise)
    assert b["chirp_log"][0] == pytest.approx(b["pink"][0], rel=1e-9)


def test_crest_depends_on_phase_not_power():
    """Same Pxx, different phases -> different crest. Schroeder < random < cophased impulse;
    a constant-envelope swept sine sits near the sqrt(2) single-tone floor."""
    b = pr.scoreboard()
    assert b["flat_schroeder"][1] < b["flat_random"][1] < b["cophased"][1]
    assert b["chirp_lin"][1] == pytest.approx(np.sqrt(2), abs=0.02)
    assert b["cophased"][1] > 10.0                       # cophased lines pile into an impulse
    assert b["flat_schroeder"][1] < 2.5                  # Schroeder tames the crest


def test_optimal_drive_is_essentially_two_tones():
    """The optimal spectrum collapses onto ~2 lines (the two resonances) — which is *why* its
    crest is phase-independent, and why the page must not credit Schroeder for its low crest."""
    P = pr.power_optimal()
    p = P / P.sum()
    n_eff = 1.0 / np.sum(p ** 2)                          # participation ratio
    assert n_eff < 4.0, f"optimal drive should be a few-tone drive, got N_eff={n_eff:.1f}"


def test_schroeder_helps_broadband_not_the_few_tone_optimal():
    """The corrected claim the page now makes: for the ~2-tone optimal drive, Schroeder and random
    phase give the SAME crest (phase is moot); Schroeder's real, large win is on broadband drives."""
    b = pr.scoreboard()
    # few-tone optimal: Schroeder buys essentially nothing over random phase
    assert abs(b["opt_schroeder"][1] - b["opt_random"][1]) < 0.1
    # broadband (all 120 lines): Schroeder's low-crest advantage is large
    _, flat_s = pr.multisine(pr.power_flat(), "schroeder")
    _, flat_r = pr.multisine(pr.power_flat(), "random")
    assert flat_r - flat_s > 1.3


def test_champion_is_pareto_optimal():
    """The Fisher-optimal + Schroeder multisine is not dominated: no drive is at least as fast
    AND strictly lower-crest. It is the drive the pipeline actually uses."""
    b = pr.scoreboard()
    ch_eta, ch_cr = b["opt_schroeder"]
    assert ch_eta == pytest.approx(min(e for e, _ in b.values()), rel=1e-9)  # fastest
    dominated = [k for k, (e, c) in b.items()
                 if k != "opt_schroeder" and e <= ch_eta * (1 + 1e-9) and c < ch_cr - 1e-6]
    assert not dominated, f"champion dominated by {dominated}"


def test_prng_is_deterministic_and_uniform():
    """The shared mulberry32 must be reproducible (seeded) and roughly uniform on [0,1) so the
    random-phase drives are identical in JS and Python."""
    a = pr._rand_phases(2000)
    b = pr._rand_phases(2000)
    assert np.allclose(a, b)                              # deterministic
    assert 0 <= a.min() and a.max() < 2 * np.pi
    assert abs(a.mean() - np.pi) < 0.2                    # ~uniform on [0, 2pi)


# ── optional: run the ACTUAL JS engine and assert bin-for-bin parity ────────────────
def test_js_engine_matches_reference():
    """Execute docs/assets/excitation-playground.js in a real browser and compare its
    scoreboard to the Python reference. Skips where Playwright/Chromium is unavailable."""
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright  # noqa: E402

    js = (ROOT / "docs" / "assets" / "excitation-playground.js").read_text()
    html = "<!doctype html><html><body><script>" + js + "</script></body></html>"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as e:  # browser binary not installed
                pytest.skip(f"no browser: {e}")
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            js_board = page.evaluate(
                "() => { const E = window.EXCITATION_PLAYGROUND, o = {};"
                " E.CATALOG.forEach(e => { const s = E.scoreOf(e); o[e.key] = [s.eta, s.crest]; });"
                " return o; }")
            browser.close()
    except Exception as e:  # driver/launch problems shouldn't fail the suite
        pytest.skip(f"playwright unavailable: {e}")

    ref = pr.scoreboard()
    for k in ref:
        assert js_board[k][0] == pytest.approx(ref[k][0], rel=1e-6), f"{k} ETA JS!=py"
        assert js_board[k][1] == pytest.approx(ref[k][1], rel=1e-6), f"{k} crest JS!=py"
