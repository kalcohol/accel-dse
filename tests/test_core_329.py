"""0.329: per-generation HBM host interface (MODEL §274).  HBM3 / HBM3E get their own PHY density
mm2_per_dq_hbm3 = Samsung Hot Chips 2026 slide 11 HBM3 PHY 8 × 3 mm / 1024 DQ (bump-map limited; HBM3E taken as the
HBM3 bump map 「假设」) — old → new 16 → 24 mm² per stack — and the n5_derived per-HBM4-stack controller scaled by
JEDEC channel count (HBM3 / 3E 16 of 32: 0 → 3.575 mm² per stack).  HBM4 / 4E unchanged.  Host IO mm2_host_io kept
(1.8 %) and reconciled with a 5 nm PCIe / Ethernet SerDes (VLSI 2022)."""
import json
from dataclasses import replace

from accel_dse import api
from accel_dse.core import budget as bm
from accel_dse.core.energy_dse import PHY_AREA_PROXY
from accel_dse.core import hbm_base
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import _density
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

H3E, H4, LP = "hbm3e_8s_8h24g_9200", "hbm4_8s_12h24g_10000", "lpddr5x_4x64_8533_16g"


def _ap(x, rel=1e-9):
    class A:
        def __eq__(self, y):
            return abs(y - x) <= rel * max(abs(x), abs(y), 1e-300)
    return A()


def _body(mem=H3E, bud=None, dens="n5_derived"):
    return {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": mem,
            "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": dens, **(bud or {})},
            "sensitivity": False, "tco": {"preset": "accel_2026", "cost": "n5_cowos_s", "d0_per_cm2": 0.1}}


def test_phy_field_value_tag_source():
    for k in ("n5_derived", "n7_tpuv4i", "n28_tpuv1"):        # bump-map limited: the same in every node preset
        v, tag, src = bm.DENSITY_PRESETS[k]["fields"]["mm2_per_dq_hbm3"]
        assert v == _ap(24 / 1024) and tag == "来源/推导"
        for s in ("Hot Chips 2026 slide 11", "8 × 3 mm", "bump map", "「假设」", "16 mm² per stack"):
            assert s in src, s
        assert bm.DENSITY_PRESETS[k]["fields"]["mm2_per_dq_hbm"][0] == _ap(32 / 2048)       # HBM4 unchanged
    assert bm.Budget().mm2_per_dq_hbm3 is None and "mm2_per_dq_hbm3" in bm._PHY_DENS


def test_dq_field_and_channel_scaling():
    b = bm.Budget(mm2_per_dq_hbm=0.015, mm2_per_dq_hbm3=0.02)
    assert bm.hbm_dq_field(b, "HBM3") == bm.hbm_dq_field(b, "hbm3e") == "mm2_per_dq_hbm3"
    assert bm.hbm_dq_field(b, "HBM4") == bm.hbm_dq_field(b, "HBM4E") == "mm2_per_dq_hbm"
    assert bm.hbm_dq_field(bm.Budget(mm2_per_dq_hbm=0.015), "HBM3E") == "mm2_per_dq_hbm"     # unset → old field
    assert bm.MC_CHANNELS == {"HBM3": 16, "HBM3E": 16, "HBM4": 32, "HBM4E": 32} and bm.MC_REF_CHANNELS == 32
    assert bm.mc_scale("HBM3E") == 0.5 and bm.mc_scale("HBM4") == 1.0 and bm.mc_scale("LPDDR5X") is None
    c = bm.Budget(mm2_per_hbm_mc=7.15)
    assert bm.mc_mm2_per_unit(c, "HBM3E") == _ap(3.575) and bm.mc_mm2_per_unit(c, "HBM4E") == 7.15
    assert bm.mc_mm2_per_unit(replace(c, mm2_per_unit_mc=2.0), "HBM3E") == 2.0               # generic still wins


def test_hbm3e_old_to_new_area():
    new = api.api_eval(_body())
    old = api.api_eval(_body(bud={"mm2_per_dq_hbm3": 1 / 64, "mm2_per_hbm_mc": 0}))             # ≤ 0.328 semantics
    tn = new["budget"]["area"]["terms"]
    assert tn["dram_phy"] == _ap(8 * 24.0) and tn["mem_ctrl"] == _ap(8 * 3.575)
    assert old["shoreline"]["die_mm2"] == _ap(654.2, rel=1e-4)                                # = 0.328 default
    assert new["shoreline"]["die_mm2"] == _ap(old["shoreline"]["die_mm2"] + (64 + 28.6) / (1 - 0.23), rel=1e-9)
    assert new["shoreline"]["die_mm2"] == _ap(774.462, rel=1e-5)
    assert new["energy"]["J_per_unit"] == old["energy"]["J_per_unit"]                       # area only, no energy term
    assert new["tco"]["usd_per_Munit"] > old["tco"]["usd_per_Munit"]


def test_hbm4_and_lpddr_unchanged():
    assert api.api_eval(_body(mem=H4))["shoreline"]["die_mm2"] == _ap(894.72, rel=1e-5)     # = 0.328
    a = api.api_eval(_body(mem=H4, bud={"mm2_per_dq_hbm3": 0.5}))                           # HBM3 field ignored on HBM4
    assert a["shoreline"]["die_mm2"] == _ap(894.72, rel=1e-5)
    assert api.api_eval(_body(mem=LP))["shoreline"]["die_mm2"] == _ap(497.596, rel=1e-5)


def test_eval_equals_dse_all_presets():
    for dp in ("n5_derived", "n7_tpuv4i", "n28_tpuv1"):
        bod = _body(dens=dp)
        e = api.api_eval(bod)
        d = {**bod, "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.1}
        d.pop("sensitivity")
        r = api.api_insight_dse(d)
        assert r["rows"][r["base"]]["mm2"] == _ap(e["shoreline"]["die_mm2"])
        assert e["budget"]["area"]["terms"]["dram_phy"] == _ap(192.0)                         # node-independent PHY
        assert ("mem_ctrl" in e["budget"]["area"]["terms"]) == (dp == "n5_derived")


def test_report_items():
    def rep(mem):
        b = {**_body(mem=mem), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"}
        b.pop("sensitivity")
        return api.api_report(b)

    def items(r):
        out = {}

        def walk(o):
            if isinstance(o, dict):
                if o.get("key") in ("hbm_phy", "hbm_mc"):
                    out[o["key"]] = o
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(r)
        return out
    i3, i4, il = items(rep(H3E)), items(rep(H4)), items(rep(LP))
    assert i3["hbm_phy"]["value"] == _ap(24.0) and "1024 DQ × mm2_per_dq_hbm3" in i3["hbm_phy"]["src"]
    assert i3["hbm_mc"]["value"] == _ap(3.575) and "× 16 / 32 通道" in i3["hbm_mc"]["src"]
    assert i4["hbm_phy"]["value"] == _ap(32.0) and "2048 DQ × mm2_per_dq_hbm）" in i4["hbm_phy"]["src"]
    assert i4["hbm_mc"]["value"] == _ap(7.15) and "通道" not in i4["hbm_mc"]["src"].split("）")[-1]
    assert not il
    assert "Samsung" in json.dumps(i3["hbm_phy"], ensure_ascii=False)


def test_hbm_base_insight_and_notes():
    b = bm.auto_area_budget()
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=H3E, serving=Serving(phase="decode", batch=8, ctx=4096))
    assert hbm_base.report(s, b)["std_phy_mm2_per_stack"] == _ap(24.0)
    assert hbm_base.report(replace(s, mem_id=H4), b)["std_phy_mm2_per_stack"] == _ap(32.0)
    assert _density(b, "mm2_per_dq", s) == _ap(24 / 1024) and _density(b, "mm2_per_dq", replace(s, mem_id=H4)) == _ap(1 / 64)
    assert "HBM3 / 3E" in PHY_AREA_PROXY["HBM"] and "窄总线同密度" not in PHY_AREA_PROXY["HBM"]


def test_host_io_reconciled_value_kept():
    v, tag, src = bm.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_host_io"]
    assert v == _ap(0.018 * 331) and tag == "二手"                                           # value unchanged
    assert "VLSITechnologyandCir46769.2022.9830304" in src and "5.96" in src
    assert all(bm.DENSITY_PRESETS[k]["fields"]["mm2_host_io"] == bm.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_host_io"]
               for k in ("n7_tpuv4i", "n28_tpuv1"))


def test_ui():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "b-mm2_per_dq_hbm3" in W.html_ids(html)
    assert "mm² / HBM3 DQ" in html and "Samsung Hot Chips 2026 第 11 页" in html and "按 16 / 32 通道折算" in html
    assert W.has(js, "'mm2_per_dq_hbm3'") and W.has(js, "'（每 HBM4 栈；HBM3 / 3E × 16/32 通道；悬停看出处）'")
