"""0.157: datapath sizing headroom (insight sizing_report)."""
from accel_dse import api as A
from accel_dse.core.budget import Budget
from accel_dse.core.insight import SIZING, TIE, _point, _gain, model_scenario, sizing_report
from accel_dse.core.energy import EnergyTable
from accel_dse.core.scenario import Scenario

MODELS = ["qwen3-8b", "gpt-oss-20b"]


def _holds(base, models, key, v, domain_metrics):
    setter = SIZING[key][3]
    for mid in models:
        s = model_scenario(base, mid)
        r, p = _point(s, EnergyTable(), "llm"), _point(s.replace("chip", setter(s.chip, v)), EnergyTable(), "llm")
        if r["fits"] and not p["fits"]:
            return False
        if any((g := _gain(m, r, p)) is not None and g < 1 - TIE for m in domain_metrics):
            return False
    return True


def test_sizing_min_ok_holds_and_fail_point_fails():
    base = Scenario()
    r = sizing_report(base, MODELS, Budget(mm2_per_mib_sram=0.5))
    mets = list(r["metrics"])
    assert {x["key"] for x in r["resources"]} == set(SIZING)
    for x in r["resources"]:
        assert 0 < x["min_ok"] <= x["current"] and x["frac"] <= 1
        assert _holds(base, MODELS, x["key"], x["min_ok"], mets)          # reported point is a real pass
        if not x["floor_hit"]:
            assert x["fail_at"] < x["min_ok"] and x["limited_by"]["model"] in MODELS
            assert not _holds(base, MODELS, x["key"], x["fail_at"], mets)
    sram = next(x for x in r["resources"] if x["key"] == "sram")
    assert abs(sram["mm2_saved"] - (sram["current"] - sram["min_ok"]) * 0.5) < 1e-9


def test_sizing_is_design_relevant_and_set_monotone():
    # adding a model to the set can only raise (never lower) each minimum
    one = {x["key"]: x["min_ok"] for x in sizing_report(Scenario(), ["qwen3-8b"])["resources"]}
    two = {x["key"]: x["min_ok"] for x in sizing_report(Scenario(), MODELS)["resources"]}
    for k in SIZING:
        assert two[k] >= one[k] * 0.7   # bisection resolution
    # default chip has real headroom on dequant for the default LLM set (the guidance is non-trivial)
    assert two["dequant"] < Scenario().chip.lanes * 0.5


def test_sizing_api_and_ui():
    r = A.api_insight_sizing({"models": ["wan2.1-1.3b"], "resources": ["lanes"]})
    assert r["domain"] == "gen" and [x["key"] for x in r["resources"]] == ["lanes"]
    try:
        A.api_insight_sizing({"resources": ["bogus"]})
        raise AssertionError("bogus resource accepted")
    except A.ApiError:
        pass
    from pathlib import Path
    web = Path(A.__file__).parent / "web"
    assert "ins-size-run" in (web / "index.html").read_text() and "/api/insight_sizing" in (web / "app.js").read_text()
