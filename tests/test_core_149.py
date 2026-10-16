"""0.149: cited density presets per node, interface PHY area, d2d knob fix, beam-search upgrade path."""
import pathlib

from accel_dse import api as A
from accel_dse.core.budget import DENSITY_PRESETS, Budget, apply_preset, area_estimate, full_area, phy_area
from accel_dse.core.evaluate import evaluate
from accel_dse.core.insight import knobs, upgrade_path
from accel_dse.core.scenario import Scenario

_STATUS = {"来源", "推导", "二手", "二手/假设", "假设", "推导/二手"}   # 0.162: derived from a secondary-sourced table


def test_presets_labelled_and_values():
    for pid, p in DENSITY_PRESETS.items():
        for f, (v, st, src) in p["fields"].items():
            assert v > 0 and st in _STATUS and len(src) > 10, (pid, f)
    n7 = DENSITY_PRESETS["n7_tpuv4i"]["fields"]
    assert abs(n7["mm2_per_mib_sram"][0] - 0.875) < 1e-12 and abs(n7["mm2_per_kmac"][0] - 0.6875) < 1e-12
    assert abs(n7["mm2_per_d2d_unit"][0] - 0.3888 * 1.043) < 1e-12 and n7["mm2_per_dq_lpddr"][1] == "推导/二手" and abs(n7["mm2_per_dq_lpddr"][0] - 0.695 / 24) < 1e-12
    assert DENSITY_PRESETS["n5_derived"]["fields"]["mm2_per_mib_sram"][1] == "推导"


def test_preset_fills_only_unset_and_user_wins():
    d = apply_preset({"density_preset": "n7_tpuv4i", "mm2_per_kmac": 9.0})
    assert d["mm2_per_kmac"] == 9.0 and abs(d["mm2_per_mib_sram"] - 0.875) < 1e-12 and "density_preset" not in d
    for bad in ({"budget": {"density_preset": "zz"}}, {"budget": {"mm2_per_dq_hbm": -1}}):
        try:
            A.api_eval(bad)
        except A.ApiError:
            continue
        raise AssertionError(bad)


def test_default_budget_unchanged_without_phy():
    s = Scenario()
    b = Budget(mm2_per_mib_sram=1.0, mm2_per_kmac=0.5, mm2_fixed=10.0)
    assert full_area(s, b)["mm2"] == area_estimate(s.chip, b)["mm2"] and phy_area(s, b)["terms"] == {}
    assert Scenario().hash() == "723894c17df5d7bb"


def test_phy_terms_mechanism():
    b = Budget(mm2_per_dq_lpddr=0.03, mm2_per_dq_hbm=0.01, mm2_per_d2d_unit=0.4, mm2_per_GBps_c2c=0.02)
    s = Scenario()
    from accel_dse import mem_catalog
    bits = mem_catalog.parse_mem_id(s.mem_id).bus_bits
    assert abs(phy_area(s, b)["terms"]["dram_phy"] - bits * 0.03) < 1e-9
    s2 = s.replace("layout.tp", 4).replace("package_cards", 2).replace("d2d_enabled", True)
    t = phy_area(s2, b)["terms"]
    assert abs(t["d2d_phy"] - s2.d2d_units * 0.4) < 1e-9 and abs(t["c2c_phy"] - s2.link.GBps * 0.02) < 1e-9
    r = A.api_eval({"scenario": {"model": "qwen3-8b"}, "budget": {"density_preset": "n7_tpuv4i", "mm2_fixed": 50}})
    assert "dram_phy" in r["budget"]["area"]["terms"]


def test_d2d_knob_actually_changes_bandwidth():
    s = Scenario().replace("layout.tp", 4).replace("package_cards", 2).replace("d2d_enabled", True)
    f = next(t for k, _, t, _ in knobs(s) if k == "d2d_x2")
    assert abs(f(s).d2d_link.GBps / s.d2d_link.GBps - 2) < 1e-12


def test_beam_never_worse_than_greedy_and_can_win():
    s = Scenario().replace("layout.tp", 4).replace("serving.batch", 64)
    ms = ["gpt-oss-120b", "qwen3-8b"]
    g = upgrade_path(s, ms, "tpot", steps=4, max_cost=6.0)
    b = upgrade_path(s, ms, "tpot", steps=4, max_cost=6.0, beam=3)
    gg, bg = g["path"][-1]["cum_gain"], b["path"][-1]["cum_gain"]
    assert bg > gg * 1.01 and b["search"] == "beam3" and abs(b["greedy_cum_gain"] - gg) < 1e-12
    assert all(m["cum"] >= 1.0 for p in b["path"] for m in p["models"]) and b["path"][-1]["cum_cost"] <= 6.0 + 1e-9
    s2 = s.replace("chip.formats.dequant_elems_per_cycle", 256.0)
    for met in ("ttft", "thr"):
        g = upgrade_path(s2, ms, met, steps=3, max_cost=4.0)
        b = upgrade_path(s2, ms, met, steps=3, max_cost=4.0, beam=3)
        assert (b["path"][-1]["cum_gain"] if b["path"] else 1) >= (g["path"][-1]["cum_gain"] if g["path"] else 1) * (1 - 1e-9)


def test_area_mode_costs_phy_for_dram_knob():
    s = Scenario().replace("layout.tp", 4).replace("serving.batch", 64)
    bud = Budget(**apply_preset({"density_preset": "n7_tpuv4i", "mm2_fixed": 100.0}))
    r = upgrade_path(s, ["gpt-oss-120b"], "tpot", bud, steps=2)
    st = {p["key"]: p for p in r["path"]}
    assert r["cost_mode"] == "area" and "dram_count" in st and st["dram_count"]["cost_basis"] == "mm²"
    rep = A.api_insight({"models": ["qwen3-8b"], "budget": {"density_preset": "n7_tpuv4i", "mm2_fixed": 100}})
    k = {x["key"]: x for x in rep["knobs"]}
    assert k["dram_count"]["d_mm2"] and k["dram_count"]["d_mm2"] > 0          # PHY open item resolved


def test_api_beam_and_ui():
    r = A.api_insight_path({"models": ["qwen3-8b"], "steps": 1, "beam": 3})
    assert r["beam"] == 3
    for bad in ({"beam": 0}, {"beam": 6}, {"beam": True}):
        try:
            A.api_insight_path(bad)
        except A.ApiError:
            continue
        raise AssertionError(bad)
    assert set(A.api_catalog()["density_presets"]) == set(DENSITY_PRESETS)
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert 'id="b-preset"' in html and 'id="ins-beam"' in html and "function initDensityPresets" in js
    assert 'id="b-mm2_per_dq_hbm"' in html and "'mm2_per_GBps_c2c'" in js
