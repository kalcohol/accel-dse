"""0.300: presets carry the on-chip wire + NoC router (NoC in preset energy by default); energy-aware design search
(Pareto front throughput vs J/unit under area / power budget, /api/insight_dse)."""
import json
from dataclasses import replace

from accel_dse import api, serve
from accel_dse.core.budget import Budget, apply_preset
from accel_dse.core.energy import (ENERGY_PRESETS, NOC_HOP_PJ_BIT_45NM, NOC_HOP_PJ_BIT_7NM, PRESET_ONCHIP, EnergyTable,
                                   apply_energy_preset, energy_report)
from accel_dse.core.energy_dse import design_grid, dominates, energy_design_search, pareto
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _b(**kw):
    return Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", **kw}))


# ------------------------------------------------------------------------------------------------ (1) preset default
def test_presets_carry_wire_and_noc_sourced():
    f45, f7 = ENERGY_PRESETS["horowitz_45nm"]["fields"], ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    assert abs(f45["pJ_bit_mm_onchip"][0] - 310 / 256 / 10) < 1e-12 and abs(f7["pJ_bit_mm_onchip"][0] - 200 / 256 / 10) < 1e-12
    assert f45["pJ_bit_hop_noc"][0] == NOC_HOP_PJ_BIT_45NM and f7["pJ_bit_hop_noc"][0] == NOC_HOP_PJ_BIT_7NM
    for f in (f45, f7):
        for k in ("pJ_bit_mm_onchip", "pJ_bit_hop_noc"):
            assert f[k][1] in ("来源", "推导", "来源/假设") and len(f[k][2]) > 40
    assert set(PRESET_ONCHIP) == set(ENERGY_PRESETS)
    t = _t()
    assert t.pJ_bit_hop_noc == NOC_HOP_PJ_BIT_7NM and t.pJ_bit_mm_onchip == 200 / 256 / 10


def test_preset_user_values_win_and_sqrtc_kept():
    assert _t(pJ_bit_hop_noc=0).pJ_bit_hop_noc == 0 and _t(pJ_bit_mm_onchip=0).pJ_bit_mm_onchip == 0
    t = _t(slc_access_ref_mib=1.0)          # √C opt-in: the distance model is not filled (they exclude each other)
    assert t.pJ_bit_mm_onchip is None and t.slc_access_ref_mib == 1.0 and t.pJ_bit_hop_noc is not None
    js = W.read()
    assert "slc_access_ref_mib" in W.function_source(js, "initEnergyPresets")


def test_preset_energy_now_includes_noc_and_wire():
    """old → new on 1P qwen3-8b (tpuv4i_7nm): the preset J/token grows by the on-chip movement terms only."""
    for ph, lo, hi in (("decode", 0.003, 0.04), ("prefill", 0.03, 0.12)):
        s = Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase=ph, batch=8, prompt=2048, ctx=4096))
        r = evaluate(s)
        new = energy_report(r, _t())
        old = energy_report(r, replace(_t(), pJ_bit_mm_onchip=None, pJ_bit_hop_noc=None))
        assert lo < new["J_per_unit"] / old["J_per_unit"] - 1 < hi
        assert new["breakdown"]["J"]["noc"] > 0 and new["breakdown"]["J"]["wire"] > 0
        assert old["breakdown"]["J"]["noc"] == 0 and old["breakdown"]["J"]["wire"] == 0
        assert abs(new["J_by_action"]["mac"] - old["J_by_action"]["mac"]) < 1e-12 * max(1.0, old["J_by_action"]["mac"])


# ------------------------------------------------------------------------------------------------ (2) design search
def test_pareto_helpers():
    k = (("thr", 1), ("J", -1))
    a, b, c = {"thr": 10, "J": 1.0}, {"thr": 9, "J": 1.2}, {"thr": 12, "J": 2.0}
    assert dominates(a, b, k) and not dominates(b, a, k) and not dominates(a, c, k)
    assert pareto([a, b, c], k) == [0, 2]
    d = {"thr": 10.05, "J": 1.5}                       # +0.5 % throughput is a tie at 1 %: a dominates d
    assert pareto([a, d], k, 0.01) == [0] and pareto([a, d], k, 0.0) == [0, 1]
    e = {"thr": 10.05, "J": 1.005}                     # within tie on both: neither dominates
    assert pareto([a, e], k, 0.01) == [0, 1]


def test_grid_keeps_whole_engines():
    ch = CHIPS["1P"]
    g = design_grid(ch)
    assert len(g) == 27 and sum(p["base"] for p in g) == 1
    for p in g:
        assert p["engines"] * p["side"] ** 2 == p["macs"]
    g = design_grid(CHIPS["100T"], mac_scales=(1.0,), sram_scales=(1.0,), topologies=("mesh", "ring"))
    assert {p["side"] for p in g} == {28, 56, 112} and len(g) == 6


def _search(chip="1P", ph="decode", batch=64, **kw):
    s = Scenario(model=M, chip=CHIPS[chip], serving=Serving(phase="decode", batch=batch, prompt=2048, ctx=4096))
    return energy_design_search(s, _t(), kw.pop("budget", _b(die_mm2=800, power_W_card=250)), phase=ph, **kw)


def test_search_area_and_front():
    o = _search()
    rows = o["rows"]
    assert o["n_feasible"] == sum(1 for x in rows if x["feasible"]) and o["front"]
    for x in rows:
        if x.get("mm2") and x["mm2"] > 800:
            assert not x["feasible"] and "面积" in x["why"]
        if x["feasible"]:
            assert abs(x["perf_W"] * x["J_per_unit"] - 1) < 1e-12
    f = [rows[i] for i in o["front"]]
    assert all(x["feasible"] and x["front"] for x in f)
    ok = [x for x in rows if x["feasible"]]
    for x in ok:                                      # nothing feasible beats a front point beyond the tie
        for y in f:
            assert not dominates(x, y, (("thr", 1), ("J_per_unit", -1)), o["tie"])
    # decode b64 is DRAM-bound on 1P: half the MACs keeps throughput and cuts J/token (clock + idle MACs) —
    # the front holds only MAC ×0.5 designs, and the base (839 mm²) is over the 800 mm² budget
    assert {x["mac_x"] for x in f} == {0.5} and not rows[o["base"]]["feasible"]
    assert o["best"]["perf_W"]["J_per_unit"] == min(x["J_per_unit"] for x in ok)


def test_search_power_dvfs_vs_reject():
    d = _search(ph="prefill", batch=1)
    capped = [x for x in d["rows"] if x.get("capped")]
    assert capped and all(x["W_card"] <= 250 * (1 + 1e-6) and x["f_scale"] < 1 and x["thr"] < x["thr_nominal"] for x in capped)
    r = _search(ph="prefill", batch=1, power_mode="reject")
    assert not any(x.get("capped") for x in r["rows"])
    assert all(not x["feasible"] for x in r["rows"] if x.get("W_nominal", 0) > 250 * (1 + 1e-9))
    # under a 250 W card budget the full-MAC chip only matches the half-MAC throughput (clock-capped)
    best = d["rows"][d["best"]["thr"]["index"]]
    half = max((x for x in d["rows"] if x["feasible"] and x["mac_x"] == 0.5), key=lambda x: x["thr"])
    assert best["thr"] < half["thr"] * 1.05


def test_search_no_budget_and_errors():
    o = _search(budget=None)
    assert o["area"] is False and o["power_cap_W"] is None and all(x["feasible"] for x in o["rows"] if x.get("fits"))
    assert any("面积密度" in n for n in o["notes"]) and any("功耗" in n for n in o["notes"])
    s = Scenario(model=M, chip=CHIPS["1P"])
    for kw, msg in (({"power_mode": "x"}, "power_mode"), ({"curve": "x"}, "curve"), ({"topologies": ("torus",)}, "topolog"),
                    ({"mac_scales": (0.3,)}, "网格")):
        try:
            energy_design_search(s, _t(), None, **kw)
        except ValueError as e:
            assert msg in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    try:
        energy_design_search(s, EnergyTable(), None)
    except ValueError:
        pass
    else:
        raise AssertionError("needs an energy table")


def test_api_and_ui():
    assert "/api/insight_dse" in serve.POST_ROUTES
    assert "/api/insight_dse" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")
    body = {"scenario": {"model": M, "chip": {"name": "100T"}}, "energy": {"energy_preset": "tpuv4i_7nm"},
            "budget": {"density_preset": "n7_tpuv4i", "die_mm2": 150}, "mac_scales": [0.5, 1], "sram_scales": [1]}
    st, p = serve.respond("POST", "/api/insight_dse", json.dumps(body).encode())
    out = json.loads(p)
    assert st == 200 and out["die_mm2"] == 150 and len(out["rows"]) == 6 and out["front"]
    for bad in ({"mac_scales": "1"}, {"sides": [3.5]}, {"power_mode": "x"}, {"tie": True}, {"bogus": 1}):
        assert serve.respond("POST", "/api/insight_dse", json.dumps({**body, **bad}).encode())[0] == 400, bad
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"ins-dse-run", "ins-dse-status", "ins-dse-summary", "ins-dse-tbl", "ins-dse-chart", "ins-dse-phase",
            "ins-dse-power", "ins-dse-macs", "ins-dse-sram", "ins-dse-topo"} <= ids
    assert [c["fn"] for c in W.req_calls(js) if c["path"] == "/api/insight_dse"] == ["runInsightDse"]
    assert W.object_dict(W.literal(js, "STAGE"))["insight_dse"] == "能耗设计搜索"
    src = W.function_source(js, "runInsightDse")
    assert "paretoChart" in src and "perf_W" in src and "front" in src
    assert "ins-dse-status" in W.function_source(js, "invalidate")
    assert hasattr(api, "api_insight_dse")


def test_default_shifts_design_conclusions():
    """Recorded old → new: with the preset's wire term a bigger SLC costs more per access, so the 25 °C J-optimal SLC
    for qwen3-8b on 100T + LPDDR5X (b8) drops 32 → 16 GiB (= the 85 °C optimum: 'hot cuts the J-opt' no longer holds);
    and the energy-tied SRAM sizing of wan2.1-1.3b is no longer limited by J/token."""
    from accel_dse.core.insight import insight_report
    s = Scenario(model=M, chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g", serving=Serving(batch=8))
    for t, j25, cuts in ((_t(), 16.0, False), (_t(pJ_bit_mm_onchip=0, pJ_bit_hop_noc=0), 32.0, True)):
        rec = insight_report(s, [M], table=t)["slc_temp"]["recommend"][0]
        assert rec["opt_J_gib_25C"] == j25 and rec["opt_J_gib_85C"] == 16.0 and rec["hot_cuts_J_opt"] is cuts
    w = api.api_insight_sizing({"models": ["wan2.1-1.3b"], "energy": {"energy_preset": "tpuv4i_7nm"},
                                "resources": ["sram"], "energy_tie": True})["resources"][0]
    assert w["limited_by"]["metric"] != "J_tok"
