"""0.161: area-aware joint sizing (per-mm² rank, beam ≥ greedy, verified only) and the binding-resource summary."""
import math
from pathlib import Path

from accel_dse import api as A
from accel_dse.core.budget import Budget, DENSITY_PRESETS
from accel_dse.core.insight import SIZING, TIE, _JOINT_ORDER, _gain, _point, model_scenario, sizing_report
from accel_dse.core.energy import EnergyTable
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

D = DENSITY_PRESETS["n7_tpuv4i"]["fields"]
B = Budget(mm2_per_kmac=D["mm2_per_kmac"][0], mm2_per_mib_sram=D["mm2_per_mib_sram"][0])


def _holds(models, vals, domain, mets):
    for mid in models:
        s = model_scenario(Scenario(), mid)
        r, t = _point(s, EnergyTable(), domain), s
        for k in _JOINT_ORDER:
            if k in vals:
                t = SIZING[k][3](t, vals[k])
        p = _point(t, EnergyTable(), domain)
        if (r["fits"] and not p["fits"]) or any((g := _gain(m, r, p)) is not None and g < 1 - TIE for m in mets):
            return False
    return True


def test_joint_verified_and_beam_not_worse_than_greedy():
    for budget in (None, B):
        for models in (["esm2-650m"], ["wan2.1-1.3b"]):
            r = sizing_report(Scenario(), models, budget, joint=True)
            j = r["joint"]
            vals = {e["key"]: e["joint"] for e in j["resources"]}
            assert j["holds"] and _holds(models, vals, r["domain"], list(r["metrics"]))
            if budget is None:
                assert j["shrink"] >= j["greedy_shrink"] - 1e-12
            else:
                assert j["mm2_saved_total"] >= j["greedy_mm2_saved"] - 1e-9


def test_area_aware_saves_more_priced_area_than_unpriced_rank():
    # the 0.160 rank (worst gain only) vs the 0.161 per-mm² rank, both scored in mm² with the same densities
    for models in (["esm2-650m"], ["wan2.1-1.3b"], ["qwen3-8b", "gpt-oss-20b"]):
        un = sizing_report(Scenario(), models, None, joint=True)["joint"]
        pr = sizing_report(Scenario(), models, B, joint=True)["joint"]
        per = {"sram": D["mm2_per_mib_sram"][0], "cores": D["mm2_per_kmac"][0] * Scenario().chip.rows * Scenario().chip.cols / 1024}
        cur = {e["key"]: e["joint"] / e["frac"] for e in un["resources"]}
        un_mm2 = sum((cur[e["key"]] - e["joint"]) * per[e["key"]] for e in un["resources"] if e["key"] in per)
        assert pr["mm2_saved_total"] >= un_mm2 - 1e-9


def test_binding_summary():
    r = sizing_report(Scenario(), ["qwen3-8b", "gpt-oss-20b"], joint=True)
    bd = r["binding"]
    keys = {x["key"] for x in bd["binding"]}
    assert keys == {x["key"] for x in r["resources"] if x["frac"] >= 1 - 1e-12}
    fr = [x["frac"] for x in bd["most_slack"]]
    assert fr == sorted(fr) and all(f < 1 for f in fr)
    assert set(bd["joint_only"]).isdisjoint(keys)
    assert "binding" in A.api_insight_sizing({"models": ["esm2-650m"], "resources": ["lanes", "sram"]})
    web = Path(A.__file__).parent / "web"
    assert "ins-size-binding" in W.html_ids(W.read(W.INDEX_HTML)) and W.mentions(W.read(), "greedy_mm2_saved")   # 0.286: parsed
