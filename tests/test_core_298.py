"""0.298: MAC energy = 16b FP multiply + FP32 accumulate (presets); distance-based local SRAM access energy."""
import math
from dataclasses import replace
from pathlib import Path

from accel_dse.core import pdqueue
from accel_dse.core.energy import (ENERGY_PRESETS, PJ_MAC_MULT_ONLY, SLC_MM2_PER_MIB_DEFAULT, EnergyTable,
                                   apply_energy_preset, energy_report, sram_pJ_bit, sram_slice_mib, sram_wire_mm)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import CORE_ACTIONS, _point
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_presets_mult_plus_fp32_accumulate():
    h, s = ENERGY_PRESETS["horowitz_45nm"]["fields"], ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    assert abs(h["pJ_mac"][0] - 2.0) < 1e-12 and abs(s["pJ_mac"][0] - 2.0 / 2.6) < 1e-12
    assert PJ_MAC_MULT_ONLY == {"horowitz_45nm": 1.1, "tpuv4i_7nm": 1.1 / 2.6}
    for f in (h, s):
        assert f["pJ_mac"][1] == "推导" and "FP32" in f["pJ_mac"][2] and "accumulation" in f["pJ_mac"][2]


def test_mac_energy_delta_is_accumulate_only():
    r = evaluate(Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="prefill", prompt=4096, batch=1)))
    eo, en = energy_report(r, _t(pJ_mac=PJ_MAC_MULT_ONLY["tpuv4i_7nm"])), energy_report(r, _t())
    d = en["J_per_unit"] - eo["J_per_unit"]
    assert abs(d - en["counts_per_unit"]["mac"] * 0.9 / 2.6 * 1e-12) < 1e-12 * max(1.0, en["J_per_unit"])
    assert 0.5 < d / eo["J_per_unit"] < 0.7              # prefill is MAC-dominated: ≈ +61 % on 1P


def test_sram_distance_formula_and_default_flat():
    ch = CHIPS["1P"]
    assert sram_pJ_bit(_t(), ch) == _t().pJ_bit_sram          # no wire coefficient → flat (≤ 0.297)
    t = _t(pJ_bit_mm_onchip=0.078)
    assert sram_slice_mib(ch) == ch.sram_mib / ch.engines
    d = sram_wire_mm(t, ch)
    assert abs(d - 0.5 * math.sqrt(ch.sram_mib / ch.engines * SLC_MM2_PER_MIB_DEFAULT)) < 1e-12
    assert abs(sram_pJ_bit(t, ch) - (t.pJ_bit_sram + 0.078 * d)) < 1e-15
    assert abs(sram_wire_mm(t, replace(ch, sram_mib=4 * ch.sram_mib)) - 2 * d) < 1e-9
    assert abs(sram_wire_mm(t, replace(ch, engines=4 * ch.engines)) - d / 2) < 1e-9


def test_report_field_and_pd_parity():
    scn = Scenario(model=M, chip=CHIPS["100T"], serving=Serving(batch=8))
    r = evaluate(scn)
    t = _t(pJ_bit_mm_onchip=0.078)
    ef, ew = energy_report(r, _t()), energy_report(r, t)
    assert "sram_access" not in ef
    sa = ew["sram_access"]
    assert sa["scale"] > 1 and sa["engines"] == 16 and abs(sa["slice_mib"] - 4.0) < 1e-12
    assert abs(ew["J_by_action"]["sram"] - ef["J_by_action"]["sram"] * sa["scale"]) < 1e-15
    pd = pdqueue._energy({"sram": 1e6}, 0.0, t, pools=((scn, 0.0),), pool_names=("decode",))
    assert abs(pd["J_by_action"]["sram"] - 1e6 * sa["pJ_bit"] * 1e-12 * 8) < 1e-15


def test_larger_sram_saves_less_with_wire():
    def j(mib, t):
        ch = replace(CHIPS["1P"], sram_mib=float(mib))
        return energy_report(evaluate(Scenario(model=M, chip=ch, serving=Serving(batch=8))), t)["J_per_unit"]
    f, w = _t(), _t(pJ_bit_mm_onchip=0.078)
    assert 0 < j(64, w) - j(512, w) < j(64, f) - j(512, f)


def test_dvfs_scales_wire_term():
    assert CORE_ACTIONS["wire"] == "pJ_bit_mm_onchip"
    r = evaluate(Scenario(model=M, chip=CHIPS["1P"], serving=Serving(batch=8)))
    t = _t(pJ_bit_mm_onchip=0.078)
    _, e = _point(r, t, "rx5700xt_n7", 0.5)
    assert e["sram_access"]["pJ_bit_mm"] < 0.078


def test_ui():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert W.mentions(js, "sram_access") and W.mentions(js, "slice_mib")
    assert "FP32 累加" in html and "引擎数" in html
