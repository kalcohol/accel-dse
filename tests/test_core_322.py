"""0.322: design report / compare (0.320 / 0.321) on the static Pages build — own channels, slow lane, progress label,
stale / superseded replies discarded; the static smoke covers them.  Drama 3 c256 second-GPU wait re-examined:
measured-only implied host time per frame (diagnostic, never a model term)."""
import ast
import os

from accel_dse.core import extspeech as XS
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_report_channels_lane_and_label():
    js = W.read()
    calls = [c for c in W.req_calls(js) if c["path"] == "/api/report"]
    by_fn = {c["fn"]: c["chan"] for c in calls}
    assert by_fn == {"runDesignReport": "report", "runDesignCompare": "report_cmp"}, by_fn
    assert "/api/report" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")
    # progress pill names both stages
    assert W.any_string_contains(js, "设计结论报告") and W.any_string_contains(js, "设计对比")
    assert W.has(js, "report_cmp:")
    # the pill keys compare by its channel (it shares /api/report with the single report)
    assert W.has(js, "const CHAN_STAGE = new Set(['report_cmp'])") and W.has(js, "CHAN_STAGE.has(chan) ? chan : ep")


def test_report_stale_and_superseded_discard():
    js = W.read()
    fns = W.functions(js)
    for fn in ("runDesignReport", "runDesignCompare"):
        f = fns[fn]
        assert W.has(f, "const res0 = S.dseRes") and W.has(f, "if (reportStale(st, res0)) return"), fn
        assert W.has(f, "if (!rr) return"), fn                      # superseded: never clears the newer status
        assert not W.has(f, "if (!rr) { st.textContent = ''"), fn
        assert W.has(f, "staleTag(S.dseVer)") and W.has(f, "st.dataset.done = '1'"), fn
    assert W.has(fns["reportStale"], "S.dseRes === res0")
    assert W.has(fns["dropReports"], "cancel(chan)")
    assert W.has(fns["invalidate"], "dropReports()") and W.has(fns["runInsightDse"], "dropReports()")
    assert {"ins-dse-report-status", "ins-dse-compare-status"} <= set(W.stale_tracked(js))


def test_static_smoke_covers_report():
    src = open(os.path.join(ROOT, "scripts/ui_static_smoke.py"), encoding="utf-8").read()
    strs = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"#ins-dse-report", "#ins-dse-compare", "#ins-dse-compare-status"} <= strs


def test_host_implied_diagnostic():
    t2 = {(r["c"], r["vocoder_gpu"]): r for r in XS.table("default", t2=True)}
    for r in t2.values():
        want = (1 / r["meas"] - 1 / r["pred"]) * 1e6
        assert abs(r["host_implied_us"] - want) < 0.2 + 1e-3 * abs(want)
    # no constant per-request host term fits: Table 1 falls with c, Table 2 (second GPU) rises — so it stays a diagnostic
    t1 = XS.table("default")
    assert t1[0]["host_implied_us"] > t1[-1]["host_implied_us"]
    assert t2[(256, "second")]["host_implied_us"] > t2[(128, "second")]["host_implied_us"]
    # predictions unchanged by 0.322 (the diagnostic is measured-only)
    assert t2[(256, "second")]["ratio"] == 1.882 and t2[(128, "second")]["ratio"] == 1.397
