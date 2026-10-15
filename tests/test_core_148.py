"""0.148: greedy multi-step upgrade path (core.insight.upgrade_path, /api/insight_path) — never worse than the start."""
import pathlib

from accel_dse import api as A
from accel_dse.core.budget import Budget
from accel_dse.core.insight import REL_COST, upgrade_path
from accel_dse.core.scenario import Scenario

_B = Scenario().replace("layout.tp", 4).replace("serving.batch", 64).replace("chip.formats.dequant_elems_per_cycle", 256.0)
_M = ["gpt-oss-120b", "mimo-v2.6-flash"]


def _check_never_worse(r):
    assert all(m["cum"] >= 1.0 for p in r["path"] for m in p["models"])
    fwd = r["metric"] in ("ttft", "tpot")
    for mid, v0 in r["start"].items():
        assert (r["final"][mid] <= v0) if fwd else (r["final"][mid] >= v0)
    cg = [1.0] + [p["cum_gain"] for p in r["path"]]
    cc = [0.0] + [p["cum_cost"] for p in r["path"]]
    assert all(b > a for a, b in zip(cg, cg[1:])) and all(b >= a for a, b in zip(cc, cc[1:]))


def test_path_never_worse_all_metrics():
    for metric in ("tpot", "ttft", "thr"):
        _check_never_worse(upgrade_path(_B, _M, metric, steps=3))


def test_path_includes_dequant_or_mixed_when_dequant_bound():
    r = upgrade_path(_B, _M, "tpot", steps=3)
    assert r["path"][0]["key"] in {"dequant_x4", "mixed_w4a8", "mixed_w4a16"} and r["cost_mode"] == "proxy"
    assert {"dequant_x4", "mixed_w4a8", "mixed_w4a16", "mixed_w8a16"} <= set(REL_COST)


def test_path_reevaluates_with_interactions():
    """Step gains are measured on the updated config: cum gain = product of step gains (per-model geomean chain)."""
    r = upgrade_path(_B, _M, "tpot", steps=4)
    prod = 1.0
    for p in r["path"]:
        prod *= p["gain"]
    assert abs(prod / r["path"][-1]["cum_gain"] - 1) < 1e-9


def test_budget_caps():
    r = upgrade_path(_B, _M, "tpot", steps=6, max_cost=1.0)
    assert r["path"] and r["path"][-1]["cum_cost"] <= 1.0 + 1e-9
    bud = Budget(mm2_per_mib_sram=1.0, mm2_per_kmac=0.5, mm2_fixed=100.0, die_mm2=200.0)
    r = upgrade_path(_B, _M, "ttft", bud, steps=6)
    assert r["cost_mode"] == "area" and r["cap_mm2"] == 200.0
    _check_never_worse(r)


def test_no_gain_returns_empty_path_not_worse():
    r = upgrade_path(_B, ["qwen3-8b"], "tpot", steps=2, max_cost=0.01)
    assert r["path"] == [] and r["final"] == r["start"]


def test_api_and_bad_inputs():
    r = A.api_insight_path({"models": ["qwen3-8b"], "metric": "ttft", "steps": 1})
    assert "hash" in r and r["metric"] == "ttft"
    for bad in ({"metric": "x"}, {"steps": 0}, {"steps": True}, {"steps": 11}, {"max_cost": -1},
                {"max_cost": "1"}, {"models": ["nope"]}, {"foo": 1}):
        try:
            A.api_insight_path(bad)
        except A.ApiError:
            continue
        raise AssertionError(bad)


def test_ui_wiring():
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert 'id="ins-path-tbl"' in html and "function runInsightPath" in js and "/api/insight_path" in js
    srv = (pathlib.Path(A.__file__).parent / "serve.py").read_text()
    assert '"/api/insight_path": api.api_insight_path' in srv
