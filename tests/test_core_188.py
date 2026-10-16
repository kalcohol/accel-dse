"""0.188: chip-side DRAM PHY line, static power via idle_frac_tdp × tdp_W, prefill energy in insight / sizing."""
from accel_dse.core.energy import EnergyTable, energy_report, apply_energy_preset
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario
from accel_dse.core.insight import sizing_report, _point

HBM, LP = "hbm3e_4s_12h24g_9600", "lpddr5x_4x64_8533_16g"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_presets_sourced():
    d = apply_energy_preset({"energy_preset": "tpuv4i_7nm"})
    assert d["pJ_bit_phy_hbm"] == 0.29 and abs(d["pJ_bit_phy_lpddr"] - 1.3) < 1e-12
    assert abs(d["idle_frac_tdp"] - 71.8 / 700) < 1e-12 and "tdp_W" not in d


def test_phy_line_by_kind():
    for m, f, v in ((HBM, "pJ_bit_phy_hbm", 0.29), (LP, "pJ_bit_phy_lpddr", 1.3)):
        e = energy_report(evaluate(Scenario().replace("mem_id", m)), _t(tdp_W=150.0))
        assert e["phy_pJ_field"] == f and not e["missing"]
        assert abs(e["J_by_action"]["dram_phy"] - e["counts_per_unit"]["dram"] * v * 8e-12) < 1e-15
    e = energy_report(evaluate(Scenario().replace("mem_id", HBM)), EnergyTable(pJ_mac=1.0))
    assert "dram_phy" not in e.get("J_by_action", {})


def test_idle_from_fraction_and_partial():
    r = evaluate(Scenario())
    e0 = energy_report(r, _t())
    assert "tdp_W" in e0["missing"] and "idle_W" not in e0["missing"]
    e = energy_report(r, _t(tdp_W=200.0))
    assert not e["missing"] and abs(e["idle_W_used"] - 200 * 71.8 / 700) < 1e-9
    assert abs(e["J_by_action"]["idle"] - e["idle_W_used"] * e["counts_per_unit"]["idle_card_s"]) < 1e-12
    e2 = energy_report(r, _t(tdp_W=200.0, idle_W=5.0))           # explicit idle_W wins
    assert e2["idle_W_used"] == 5.0 and e2["J_per_unit"] < e["J_per_unit"]
    e3 = energy_report(r, _t(tdp_W=400.0))                       # static term monotone in TDP
    assert e3["J_per_unit"] > e["J_per_unit"]


def test_api_tdp_from_budget():
    from accel_dse import api
    t = api._energy_table({"energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"tdp_W_card": 300}})
    assert t.tdp_W == 300.0
    t = api._energy_table({"energy": {"energy_preset": "tpuv4i_7nm", "tdp_W": 100}, "budget": {"tdp_W_card": 300}})
    assert t.tdp_W == 100.0


def test_prefill_energy_in_point_and_sizing():
    t = _t(tdp_W=150.0)
    p = _point(Scenario(), t, "llm")
    assert p["J_prefill"] and p["J_prefill"] > 0 and p["J_prefill"] != p["J_tok"] and not p["J_partial"]
    a = sizing_report(Scenario(), resources=["sram"], table=t)
    b = sizing_report(Scenario(), resources=["sram"], table=t, energy_tie=True)
    assert "J_prefill" in b["metrics"]
    assert b["resources"][0]["min_ok"] >= a["resources"][0]["min_ok"]
    tr = b["resources"][0]["traffic"]
    assert all("J_prefill" in x for x in tr.values())
    for x in tr.values():
        assert x["J"] <= 1.01 + 1e-9 and x["J_prefill"] <= 1.01 + 1e-9


def test_ui_wiring():
    h = open("accel_dse/web/index.html").read(); js = open("accel_dse/web/app.js").read()
    for k in ("pJ_bit_phy_hbm", "pJ_bit_phy_lpddr", "idle_frac_tdp", "tdp_W"):
        assert f'id="e-{k}"' in h and f"'{k}'" in js
    assert "dram_phy" in js and "J_prefill" in js
