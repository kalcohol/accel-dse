"""0.216: HBM standby/refresh static power; availability sources."""
from accel_dse import api, mem_catalog
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, hbm_static_W


def _t(**kw):
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    return EnergyTable(mW_hbm_die_standby=f["mW_hbm_die_standby"][0], mW_hbm_Gb_refresh=f["mW_hbm_Gb_refresh"][0], **kw)


def test_static_scales_with_stacks_and_dies():
    t = _t()
    b = hbm_static_W("hbm3e_4s_8h24g_9200", t)["W"]
    assert abs(b - 3.0336) < 1e-6
    assert abs(hbm_static_W("hbm3e_4s_12h24g_9200", t)["W"] / b - 1.5) < 1e-9
    assert abs(hbm_static_W("hbm3e_8s_8h24g_9200", t)["W"] / b - 2.0) < 1e-9
    assert hbm_static_W("lpddr5x_8c_8533", t) is None and hbm_static_W(None, t) is None


def test_no_double_count_with_idle():
    from accel_dse.core.energy import energy_report
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.scenario import Scenario
    r = evaluate(Scenario(chip=CHIPS["1P"], model="qwen3-8b"))
    assert "hbm_static" in energy_report(r, _t(pJ_mac=0.4))["J_by_action"]
    assert "hbm_static" not in energy_report(r, _t(pJ_mac=0.4, idle_W=50.0))["J_by_action"]


def test_whatif_power_column():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "formats_override": [["kv", "fp8"]]}})
    rows = {x["key"]: x for x in w["sensitivity"]["balance_whatif"]["rows"]}
    assert abs(rows["hbm_12h24g"]["d_hbm_W"] - 1.5168) < 1e-6
    assert abs(rows["hbm_16h24g"]["d_hbm_W"] - 3.0336) < 1e-6
    assert abs(rows["dram_bw_x2"]["d_hbm_W"] - 3.0336) < 1e-6
    assert rows["ctx_half"]["d_hbm_W"] == 0.0


def test_cap_sources_cover_vendor_tags():
    for (g, h, d) in mem_catalog.HBM_CAP_SOURCES:
        assert (h, d) in mem_catalog.HBM_CATALOG[g]["cap_tags"]
    assert "qualification" in mem_catalog.HBM_CAP_SOURCES[("HBM3E", 12, 24)]
