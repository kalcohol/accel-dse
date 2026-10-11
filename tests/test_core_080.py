"""0.80: PD SLO-rate speedups are exact — the stability-only path (no TTFT laws, no bulk-chain solves when every
stable cap gives the same KV verdict) and the TPOT-miss shortcut return what the full mode returns."""
from dataclasses import replace

from accel_dse.core import pdqueue as q
from accel_dse.core.pdsim import capture_ctx
from accel_dse.core.validation import v4_scenario


def _ctx(**pd):
    s = v4_scenario(0.6)
    if pd:
        s = replace(s, pd=replace(s.pd, **pd))
    rep, ctx = capture_ctx(s)
    return rep, ctx


def test_stab_only_matches_full_mode_over_lambda():
    for pd in ({}, {"kv_policy": "recompute", "kv_capacity_GB": 8.0}):
        rep, ctx = _ctx(**pd)
        lam0 = rep["queue"]["lambda_rps"]
        cu = {**ctx, "slo": None}
        for f in (0.2, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6, 2.5, 4.0):
            lam = lam0 * f
            full = q._pd_mode(cu, lam)["stable"]
            fast = q._pd_mode({**cu, "_stab_only": True}, lam)["stable"]
            assert full == fast, (pd, f, full, fast)


def test_prefill_stable_caps_iff_server_stable():
    rep, ctx = _ctx()
    lam0 = rep["queue"]["lambda_rps"] / ctx["r_p"]
    pp, pts = ctx["ppool"], ctx["pts"]
    for f in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 20.0):
        for cap in (None, 1, 4, 16):
            caps = q._prefill_stable_caps(pp, lam0 * f, pts, 1.0, cap)
            pre = q._prefill_server(pp, lam0 * f, pts, 1.0, cap, law=False)
            assert bool(caps) == (pre is not None), (f, cap)
            if pre is not None:
                assert pre["b"] in [b for b, _ in caps]


def test_slo_rate_unchanged_by_stab_only():
    rep, ctx = _ctx()
    sv = v4_scenario(0.6).serving
    a = q._slo_rate(q._pd_mode, {**ctx}, rep["queue"]["lambda_rps"], sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3)
    orig = q._pd_mode_base

    def no_shortcut(c, lam):                                   # the 0.79 path: never stability-only
        return orig({k: v for k, v in c.items() if k != "_stab_only"}, lam)
    q._pd_mode_base = no_shortcut
    try:
        b = q._slo_rate(q._pd_mode, {**ctx}, rep["queue"]["lambda_rps"], sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3)
    finally:
        q._pd_mode_base = orig
    assert a == b and a > 0
