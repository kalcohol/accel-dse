"""0.324: custom HBM4 / 4E base die (Chip.hbm_base, core/hbm_base.py; MODEL §269) — opt-in, std bit-identical to
0.323; host PHY area / controller / PHY pJ / memory $ change, shoreline need and bandwidth do not; eval ≡ DSE row;
report / UI / CLI wiring.  Plus CAP_MEMO: exact reuse of evaluate / the cap bisection inside one search — results
byte-identical to the unmemoised path."""
import copy
import json
import math
import os
import subprocess
import sys

from accel_dse import api
from accel_dse.core import budget as bmod, energy_dse, hbm_base
from accel_dse.core.tco import TCO_PRESETS, mem_price
from tests import webstruct as W  # noqa: E402

def _approx(v, rel=1e-9, abs=0.0):
    class A:
        def __eq__(self, x):
            return x is not None and math.isclose(x, v, rel_tol=rel, abs_tol=abs)

        def __repr__(self):
            return f"≈{v}"
    return A()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HBM4 = "hbm4_8s_12h24g_10000"


def _body(hb=None, mem=HBM4, cap=400, phase="decode", model="qwen3-8b"):
    sc = {"model": model, "mem_id": mem, "serving": {"phase": phase, "batch": 64 if phase == "decode" else 8,
                                                     "prompt": 2048, "ctx": 4096}}
    if hb is not None:
        sc["chip"] = {"hbm_base": hb}
    b = {"chip_preset": "1P", "scenario": sc, "energy": {"energy_preset": "tpuv4i_7nm"},
         "budget": {"density_preset": "n5_derived", **({"power_W_card": cap} if cap else {})}, "cooling": "air_oam",
         "tco": {"preset": "accel_2026", "cost": "n5_cowos_s", "d0_per_cm2": 0.1}}
    if cap:
        b["power_cap"] = True
    return b


def _dse(b):
    d = {**copy.deepcopy(b), "mac_scales": [0.5, 1], "sram_scales": [1], "f_scales": [1, 0.7], "cost": "n5_cowos_s",
         "tco": "accel_2026", "d0_per_cm2": 0.1}
    d.pop("power_cap", None)
    return api.api_insight_dse(d)


def test_std_unchanged_from_0323():
    """Pinned on the 0.323 tree (626fdb3) with no hbm_base key: every number identical; std adds no DSE key."""
    for hb in (None, "std"):
        e = api.api_eval(_body(hb))
        assert e["energy"]["J_per_unit"] == 0.037492266659349203
        assert e["tco"]["usd_per_Munit"] == 0.011879644207241864
        assert "hbm_base" not in e
        r = _dse(_body(hb))
        assert r["rows"][r["base"]]["J_per_unit"] == 0.03774018739951754
        assert "hbm_base" not in r and not any("hbm_custom" in x for x in r["rows"])
    assert api.api_eval(_body(None))["energy"]["phy_pJ_per_bit"] == _approx(0.29)   # new output (0.324)


def test_custom_mechanisms():
    s, c = api.api_eval(_body("std", cap=None)), api.api_eval(_body("custom", cap=None))
    hb = c["hbm_base"]
    assert hb["applies"] and hb["mode"] == "custom" and hb["stacks"] == 8 and hb["edge_per_stack_mm"] == 11
    assert hb["custom_phy_mm2_per_stack"] == _approx(8.5 * 1.5)
    # host die: 8 × (2048 DQ PHY 32 mm² − 12.75 mm² D2D) smaller, grossed up by the uncore share of the final die
    # (budget.full_area, the existing rule for every term); shoreline need per stack unchanged
    from accel_dse.api import _budget
    u = _budget({"budget": {"density_preset": "n5_derived"}}).uncore_frac
    assert hb["std_phy_mm2_per_stack"] == _approx(32.0)        # 2048 × 0.015625 — Samsung HC 2026 "> 8 × 4 mm"
    d_die = s["shoreline"]["die_mm2"] - c["shoreline"]["die_mm2"]
    assert d_die == _approx(8 * (32.0 - 12.75) / (1 - u), rel=1e-9)
    assert s["shoreline"]["mem"]["edges"] == c["shoreline"]["mem"]["edges"]
    # PHY energy × 0.55; throughput unchanged without a cap (stack bandwidth unchanged)
    assert c["energy"]["phy_pJ_per_bit"] == _approx(0.55 * s["energy"]["phy_pJ_per_bit"])
    assert c["energy"]["J_per_unit"] < s["energy"]["J_per_unit"]
    assert c["tco"]["thr"] == s["tco"]["thr"]
    # memory $ × 1.10 on the sourced price; package $ down (smaller die)
    assert c["tco"]["mem_usd"] == _approx(1.10 * s["tco"]["mem_usd"])
    assert c["tco"]["pkg_usd"] < s["tco"]["pkg_usd"]
    assert "定制基础晶粒" in c["tco"]["mem_price_src"]
    assert any(n.startswith("近存计算") for n in hb["notes"])        # listed, not modelled


def test_area_terms_and_controller():
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.scenario import Scenario
    from dataclasses import replace
    b = bmod.auto_area_budget()
    b = replace(b, mm2_per_unit_mc=5.0)
    st = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HBM4)
    cu = Scenario(model="qwen3-8b", chip=replace(CHIPS["1P"], hbm_base="custom"), mem_id=HBM4)
    assert bmod.phy_area(cu, b)["terms"]["dram_phy"] == _approx(8 * 12.75)
    assert bmod.phy_area(st, b)["terms"]["dram_phy"] == _approx(2048 * 8 * b.mm2_per_dq_hbm)
    ts, tc = bmod.full_area(st, b)["terms"], bmod.full_area(cu, b)["terms"]
    assert ts.get("mem_ctrl") and not tc.get("mem_ctrl")              # the controller moves to the base die


def test_price_rules():
    p = TCO_PRESETS["accel_2026"]
    v0, _ = mem_price(p, "HBM4")
    v1, tag = mem_price(p, "HBM4", custom=True)
    assert v1 == _approx(1.10 * v0) and "1.1" in tag
    assert mem_price(p, "HBM4", {"HBM4": 12.0}, custom=True)[0] == 12.0      # a user price is taken as given
    assert hbm_base.USD_FACTOR_RANGE == (1.05, 1.10) and hbm_base.PJ_FACTOR_RANGE == (0.30, 0.55)


def test_hbm3e_ignored_with_note():
    mem = "hbm3e_8s_8h24g_9200"
    s, c = api.api_eval(_body("std", mem=mem)), api.api_eval(_body("custom", mem=mem))
    assert c["energy"]["J_per_unit"] == s["energy"]["J_per_unit"] and c["tco"]["usd_per_Munit"] == s["tco"]["usd_per_Munit"]
    assert not c["hbm_base"]["applies"]
    assert any(w.startswith("hbm_base=custom") for w in c["summary"]["warnings"])
    r = _dse(_body("custom", mem=mem))
    assert not any(x.get("hbm_custom") for x in r["rows"])


def test_eval_equals_dse_row_custom():
    for mem, cap in ((HBM4, 400), (HBM4, None)):
        e = api.api_eval(_body("custom", mem=mem, cap=cap))
        d = {**_body("custom", mem=mem, cap=cap), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s",
             "tco": "accel_2026", "d0_per_cm2": 0.1}
        d.pop("power_cap", None)
        r = api.api_insight_dse(d)
        row = r["rows"][r["base"]]
        assert r["hbm_base"] == "custom" and row["hbm_custom"]
        assert row["tco"]["usd_per_Munit"] == _approx(e["tco"]["usd_per_Munit"], rel=1e-9)
        assert row["thr"] == _approx(e["tco"]["thr"], rel=1e-9)
        assert row["usd_good_pkg"] == _approx(e["tco"]["pkg_usd"], rel=1e-9)


def test_report_item_only_when_custom():
    for hb, want in (("std", False), ("custom", True)):
        b = {**_body(hb, cap=None), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"}
        b.pop("power_cap", None)
        rep = api.api_report(b)
        txt = json.dumps(rep, ensure_ascii=False)
        assert ('"hbm_base"' in txt and "HBM 基础晶粒" in txt) == want


def test_chip_validation_ui_cli():
    from accel_dse.core.hardware import CHIPS
    from dataclasses import replace
    try:
        replace(CHIPS["1P"], hbm_base="logic")
        raise AssertionError("hbm_base='logic' accepted")
    except ValueError:
        pass
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "c-hbm_base" in W.html_ids(html) and W.select_options(html, "c-hbm_base") == ["", "custom"]
    assert W.has(js, "S.chipOver.hbm_base = e.target.value") and W.has(js, "$('c-hbm_base').value = S.chipOver.hbm_base || ''")
    assert W.count(js, "for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject'])") == 2   # 0.306 pin
    out = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--chip", "1P", "--mem", HBM4,
                          "--hbm-base", "custom", "--json"], capture_output=True, text=True, cwd=ROOT, check=True)
    d = json.loads(out.stdout)
    assert d["hbm_base"]["applies"] and d["shoreline"]["die_mm2"] == _approx(620.436, abs=1e-2)


def test_hbm_base_study_thermal_flip():
    """The smaller custom host die raises W/mm²: prefill MAC×1 mono turns thermally unbuildable on air (MODEL §269.3)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import chiplet_example as X
    finally:
        sys.path.pop(0)
    o = {hb: X.run("qwen3-8b", "prefill", out=lambda *_: None, cost="n5_cowos_l", mem=X.HBM4_MEM, cooling="air_oam",
                   tco="accel_2026", include_unbuildable=True, hbm_base=hb, macs=(1.0,)) for hb in ("std", "custom")}
    a, b = o["std"]["rows"][0], o["custom"]["rows"][0]
    assert a["dies"] == 1 and a["buildable"] is True and b["buildable"] is False
    assert a["shoreline"] == b["shoreline"] == "tight" and a["thr"] == b["thr"]
    assert b["thermal"]["flux_W_mm2"] > a["thermal"]["flux_W_mm2"] and b["mm2"] < a["mm2"]


def _bodies():
    base = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "prefill", "batch": 8,
                                                                               "prompt": 4096, "ctx": 4096}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived", "power_W_card": 150},
            "mac_scales": [0.5, 1, 2], "sram_scales": [1], "topologies": ["mesh"], "mem_types": ["HBM3E", "LPDDR5X"],
            "f_scales": [1, 0.7], "dies": [1, 2], "d2d_stds": ["ucie-a-64"], "cost": "n5_cowos_s",
            "tco": "accel_2026", "cooling": "air_hgx", "d0_per_cm2": 0.1}
    mix = copy.deepcopy(base)
    mix["scenario"]["serving"] = {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}
    mix["mix_out_tokens"] = 256
    cu = copy.deepcopy(base)
    cu.update(mem_types=["HBM4", "HBM3E"], include_unbuildable=True, power_scope="card")
    cu["scenario"]["chip"] = {"hbm_base": "custom"}
    return {"prefill150": base, "mix150": mix, "custom_unbuild": cu}


def test_cap_memo_identical_and_scoped():
    old = energy_dse.CAP_MEMO
    calls = {"n": 0}
    from accel_dse.core import evaluate as evmod
    orig_e = energy_dse.evaluate

    def spy(*a, **k):
        calls["n"] += 1
        return orig_e(*a, **k)
    try:
        energy_dse.evaluate = evmod.evaluate = spy        # powercap imports evaluate at call time
        for name, b in _bodies().items():
            out, n = {}, {}
            for flag in (False, True):
                energy_dse.CAP_MEMO = flag
                calls["n"] = 0
                out[flag] = json.dumps(api.clean(api.api_insight_dse(copy.deepcopy(b))), sort_keys=True)
                n[flag] = calls["n"]
            assert out[False] == out[True], name
            assert n[True] < n[False], (name, n)
            assert energy_dse._CAP_MEMO_D is None          # the store lives only for one search
    finally:
        energy_dse.CAP_MEMO = old
        energy_dse.evaluate = evmod.evaluate = orig_e
