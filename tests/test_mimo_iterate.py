"""Unit tests for the MIMO iterative loop policy (callback-based, no campaigns)."""
import numpy as np

from system_ident.mimo_iterate import iterate_mimo


def _run(frac_seq, *, target, max_passes=5, prior_u=0.5):
    """Drive iterate_mimo with a scripted frac_unc sequence; record the design calls."""
    seq = iter(frac_seq)
    seen = []                                   # (u, modes) passed to design each pass

    def design(modes, u):
        seen.append((u, list(modes)))
        return ("psd", u)

    def measure(pxx, k):
        return ("exps", k)

    def fit(exps):
        fu = next(seq)
        return {"modes": [(1.0 + fu, 50.0)], "frac_unc": fu, "recovery": 0.1 * fu}

    final, hist = iterate_mimo(design, measure, fit, prior_modes=[(1.0, 50.0)],
                               prior_u=prior_u, target_frac_unc=target, max_passes=max_passes)
    return final, hist, seen


def test_iterate_stops_when_target_met():
    # 0.1 (>t) -> 0.02 (>t) -> 0.005 (<=0.01) stop: 3 passes, converged.
    final, hist, seen = _run([0.1, 0.02, 0.005], target=0.01)
    assert len(hist) == 3
    assert final["converged"] is True
    assert final["frac_unc"] == 0.005
    # robust first pass, then point-optimal (u=0) thereafter
    assert [u for u, _ in seen] == [0.5, 0.0, 0.0]
    # each redesign trusts the PREVIOUS pass's fitted modes
    assert abs(seen[1][1][0][0] - 1.1) < 1e-12      # pass-1 design saw pass-0 modes (1+0.1)
    assert abs(seen[2][1][0][0] - 1.02) < 1e-12     # pass-2 design saw pass-1 modes (1+0.02)


def test_iterate_stops_at_max_passes_when_not_converged():
    final, hist, seen = _run([0.5, 0.5, 0.5], target=0.01, max_passes=3)
    assert len(hist) == 3
    assert final["converged"] is False
    assert [u for u, _ in seen] == [0.5, 0.0, 0.0]


def test_iterate_converges_first_pass():
    # already below target on pass 0 -> single pass, no point-optimal redesign
    final, hist, seen = _run([0.001], target=0.01)
    assert len(hist) == 1 and final["converged"] is True
    assert [u for u, _ in seen] == [0.5]


def test_u_decay_keeps_a_shrinking_nonzero_spread():
    # u_decay=0.3: pass 0 uses prior_u; later passes shrink but stay > 0 (the anti-over-
    # concentration fix), vs the default u_decay=0 which drops straight to 0.
    seq = iter([0.1, 0.1, 0.1])
    us = []

    def design(modes, u):
        us.append(u); return "psd"

    from system_ident.mimo_iterate import iterate_mimo
    iterate_mimo(design, lambda p, k: "e", lambda e: {"modes": [(1., 50.)], "frac_unc": next(seq)},
                 prior_modes=[(1., 50.)], prior_u=0.5, target_frac_unc=0.0,
                 max_passes=3, u_decay=0.3)
    assert us[0] == 0.5
    assert abs(us[1] - 0.15) < 1e-12 and abs(us[2] - 0.045) < 1e-12   # 0.5*0.3, 0.5*0.3^2


def test_combine_folds_all_passes_into_the_fit():
    # with combine, the fit sees the ACCUMULATED list of every pass's exps
    from system_ident.mimo_iterate import iterate_mimo
    seq = iter([0.1, 0.1])
    seen_lens = []

    def fit(exps):
        seen_lens.append(len(exps))          # exps is the combined list here
        return {"modes": [(1., 50.)], "frac_unc": next(seq)}

    iterate_mimo(lambda m, u: "psd", lambda p, k: ("exp", k), fit,
                 prior_modes=[(1., 50.)], prior_u=0.5, target_frac_unc=0.0,
                 max_passes=2, combine=lambda all_exps: list(all_exps))
    assert seen_lens == [1, 2]               # pass 0 sees 1 pass; pass 1 sees both
