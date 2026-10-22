"""0.216: HBM standby/refresh static power; availability sources."""
from accel_dse import api, mem_catalog
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, hbm_static_W


def _t(**kw):
    # 0.333: the preset fills per-channel standby (mW_hbm_ch_standby) instead; the per-die field stays a user input —
    # these tests keep exercising it at its ≤ 0.332 preset value 48 mW (DRAMsim3 HBM2 IDD2N × 1.2 V)
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    return EnergyTable(mW_hbm_die_standby=48.0, mW_hbm_Gb_refresh=f["mW_hbm_Gb_refresh"][0], **kw)


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
    # 0.333: standby is per 64-bit channel (376.8 mW), so a taller stack adds refresh only (4 stacks × 4 dies × 24 Gb ×
    # 1.95 mW/Gb = 0.7488 W; was 1.5168 with 48 mW per die) and doubling the stacks adds 64 channels + their refresh
    # (25.611 W; was 3.0336)
    assert abs(rows["hbm_12h24g"]["d_hbm_W"] - 0.7488) < 1e-6
    assert abs(rows["hbm_16h24g"]["d_hbm_W"] - 1.4976) < 1e-6
    assert abs(rows["dram_bw_x2"]["d_hbm_W"] - 25.611111111111107) < 1e-6
    assert rows["ctx_half"]["d_hbm_W"] == 0.0


def test_cap_sources_cover_vendor_tags():
    for (g, h, d) in mem_catalog.HBM_CAP_SOURCES:
        assert (h, d) in mem_catalog.HBM_CATALOG[g]["cap_tags"]
    assert "qualification" in mem_catalog.HBM_CAP_SOURCES[("HBM3E", 12, 24)]
