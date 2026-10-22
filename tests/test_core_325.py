"""0.325: adversarial review of the 0.324 custom HBM base die (MODEL §270) — host D2D area from the model's UCIe-A
geometry; the D2D link replaces both HBM PHY ends (device pJ/bit loses its memory-side PHY, gains the base-die half
of the link), also in the PD queue; the host die held at the shoreline floor of abutting stacks; the card-level
thermal precedent judged on its own area definition; near-memory compute listed, not modelled."""
import math
import os
import sys

from accel_dse import api, mem_catalog
from accel_dse.core import hbm_base, pdqueue, thermal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HBM4 = "hbm4_8s_12h24g_10000"


def _close(a, b, rel=1e-9):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


def _body(hb, chip="1P", mem=HBM4, phase="decode", cap=None, cool="air_oam"):
    b = {"chip_preset": chip, "scenario": {"model": "qwen3-8b", "mem_id": mem, "chip": {"hbm_base": hb},
                                           "serving": {"phase": phase, "batch": 64 if phase == "decode" else 8,
                                                       "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"},
         "budget": {"density_preset": "n5_derived", **({"power_W_card": cap} if cap else {})}, "cooling": cool,
         "sensitivity": False, "tco": {"preset": "accel_2026", "cost": "n5_cowos_s", "d0_per_cm2": 0.1}}
    if cap:
        b["power_cap"] = True
    return b


def test_host_d2d_from_ucie_geometry():
    for mem, units in ((HBM4, 7), ("hbm4_8s_12h24g_8000", 6), ("hbm4e_8s_12h24g_12000", 8)):
        sp = mem_catalog.parse_mem_id(mem)
        assert hbm_base.host_d2d_units(sp) == units == math.ceil(hbm_base.stack_GBps(sp) / 384.0)
        assert _close(hbm_base.host_d2d_mm2(sp), units * 0.3888 * 1.043)
        assert hbm_base.host_d2d_edge_mm(sp) < 11.0          # the link is narrower than the stack: footprint decides
    assert hbm_base.CUSTOM["base_die_d2d_mm2"][0] == 8.5 * 1.5     # memory side, reference only


def test_interface_energy_both_ends():
    s, c = api.api_eval(_body("std")), api.api_eval(_body("custom"))
    js, jc = s["energy"]["J_by_action"], c["energy"]["J_by_action"]
    ratio = jc["dram_phy"] / js["dram_phy"]
    assert _close(ratio, 0.125 / 0.29)
    bits = js["dram_phy"] / (0.29e-12)                      # DRAM bits per unit (same traffic in both)
    assert _close(jc["dram"] - js["dram"], bits * (0.125 - 0.29) * 1e-12, rel=1e-6)
    ip = c["hbm_base"]["iface_pJ"]
    assert _close(ip["link"], 0.25) and _close(ip["std_pair"], 0.58)
    lo, hi = hbm_base.VENDOR_IFACE_RATIO
    assert lo <= ip["ratio"] <= hi                           # inside Marvell's −45 … −75 % (cross-check)
    # the heat of the base-die half sits in the stacks: DRAM-device W falls less than card W
    assert c["energy"]["J_per_unit"] < s["energy"]["J_per_unit"]


def test_pdqueue_device_delta():
    t = api._energy_table({"energy": {"energy_preset": "tpuv4i_7nm"}})
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.scenario import Scenario
    from dataclasses import replace
    st = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HBM4)
    cu = Scenario(model="qwen3-8b", chip=replace(CHIPS["1P"], hbm_base="custom"), mem_id=HBM4)
    e3 = Scenario(model="qwen3-8b", chip=replace(CHIPS["1P"], hbm_base="custom"), mem_id="hbm3e_8s_8h24g_9200")
    assert pdqueue._dev_delta(st, t) == 0.0 and pdqueue._dev_delta(e3, t) == 0.0
    assert _close(pdqueue._dev_delta(cu, t), 0.125 - 0.29) and _close(pdqueue._phy_pJ(cu, t), 0.125)


def test_shoreline_floor_padding():
    for chip, mem in (("1P", HBM4), ("H100-like", "hbm4_6s_12h24g_10000")):
        s, c = api.api_eval(_body("std", chip, mem)), api.api_eval(_body("custom", chip, mem))
        hb = c["hbm_base"]
        assert hb["shore_pad_mm2"] and hb["shore_pad_mm2"] > 0
        assert c["shoreline"]["status"] != "over" and c["shoreline"]["die_mm2"] < s["shoreline"]["die_mm2"]
        # the floor is tight: 1 % less die would be over
        from accel_dse.core.budget import shoreline_report
        scn = api.scenario_from_body(_body("custom", chip, mem))
        assert shoreline_report(scn, c["shoreline"]["die_mm2"] * 0.99)["status"] == "over"
        assert c["tco"]["pkg_usd"] > 0 and "填充硅" in hb["notes"][0]


def test_thermal_card_area_is_precedent_definition():
    s, c = api.api_eval(_body("std", phase="prefill")), api.api_eval(_body("custom", phase="prefill"))
    ts, tc = s["thermal"], c["thermal"]
    assert _close(tc["card_area_mm2"], s["shoreline"]["die_mm2"])          # = the std-interface die
    assert _close(tc["flux_W_mm2"], tc["pkg_W"] / tc["card_area_mm2"])
    assert _close(tc["die_flux_W_mm2"], tc["die_W"] / c["shoreline"]["die_mm2"])   # physical, real area
    # 0.324: custom was "over" on its shrunk die; with the precedent area both were "ok".  0.333: HBM energy
    # (5.80 pJ/bit + per-channel standby) pushes the std package to 784.7 W / 0.877 W/mm² → over air_oam; custom
    # stays ok (751.7 W / 0.840 on the same card area) — the precedent definition still keeps the custom row buildable
    assert ts["status"] == "over" and tc["status"] == "ok"
    r = thermal.thermal_report(600.0, 500.0, 1, 50.0, "air_oam", 0.0, None, card_area_mm2=700.0)
    assert _close(r["flux_W_mm2"], r["pkg_W"] / 700.0) and _close(r["die_flux_W_mm2"], r["die_W"] / 500.0)
    assert thermal.thermal_report(600.0, 500.0, 1, 50.0, "air_oam")["flux_W_mm2"] > r["flux_W_mm2"]


def test_reticle_and_eval_equals_dse_row():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import chiplet_example as X
    finally:
        sys.path.pop(0)
    o = {hb: X.run("qwen3-8b", "decode", out=lambda *_: None, cost="n5_cowos_l", mem=X.HBM4_MEM, cooling="air_oam",
                   tco="accel_2026", include_unbuildable=True, hbm_base=hb, macs=(2.0,)) for hb in ("std", "custom")}
    a, b = o["std"]["rows"][0], o["custom"]["rows"][0]
    assert a["dies"] == b["dies"] == 1 and a["mm2"] > 858 and not a["feasible"]
    assert b["mm2"] < 858 and b["feasible"] and b["buildable"]          # the custom die fits the reticle
    for cap, cool in ((None, "air_oam"), (400, "liquid")):
        e = api.api_eval(_body("custom", "H100-like", "hbm4_6s_12h24g_10000", cap=cap, cool=cool))
        d = {**_body("custom", "H100-like", "hbm4_6s_12h24g_10000", cap=cap, cool=cool), "mac_scales": [1],
             "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.1}
        d.pop("power_cap", None)
        d.pop("sensitivity", None)
        r = api.api_insight_dse(d)
        row = r["rows"][r["base"]]
        assert _close(row["mm2"], e["shoreline"]["die_mm2"]) and row["shoreline"] == e["shoreline"]["status"]
        assert row["thermal"]["status"] == e["thermal"]["status"]
        assert _close(row["thermal"]["flux_W_mm2"], e["thermal"]["flux_W_mm2"], rel=1e-6)
        assert _close(row["tco"]["usd_per_Munit"], e["tco"]["usd_per_Munit"], rel=1e-9)


def test_near_memory_compute_documented_not_modelled():
    c = api.api_eval(_body("custom"))
    notes = c["hbm_base"]["notes"]
    assert any(n.startswith("近存计算") and "未建模" in n for n in notes)
    assert any(n.startswith("内存控制器功耗") for n in notes)
    assert not any(k.startswith(("nmc", "near_mem")) for k in c["hbm_base"])
