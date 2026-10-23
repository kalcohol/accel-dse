"""0.342: adversarial ming-omni-tts-0.5b; energy/thermal/TCO consistency 0.333–0.341; Orca/pocket still pending."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.energy import (
    ENERGY_PRESETS,
    EnergyTable,
    apply_energy_preset,
    dram_pJ_field,
    energy_report,
)
from accel_dse.core.energy_dse import MEM_PROXY
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import hbm_standby_W
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS
from accel_dse.core.thermal import package_power
from accel_dse.core.tts import tts_head_ops

CHIP, MEM = CHIPS["100T"], "lpddr5x_4x64_8533_16g"


def test_ming_omni_adversarial_ditar():
    m = get_model("ming-omni-tts-0.5b")
    assert m.param_check()["rel_err"] == 0.0
    ts = m.tts
    assert ts.flow_depth == 8 and ts.flow_steps == 10 and ts.cfg_scale == 2.0
    assert ts.rate == 44100 and ts.patch == 4 and ts.history == 32
    assert abs(ts.s_per_step - 0.32) < 1e-9
    m05 = get_model("ming-3.0-tts-0.5b")
    assert abs(ts.s_per_step / m05.tts.s_per_step - 4.0) < 1e-9
    ops = tts_head_ops(ts, 1, 0)
    assert any("flow0." in o.name for o in ops) and any("flow9." in o.name for o in ops)
    sc = Scenario(model="ming-omni-tts-0.5b", chip=CHIP, mem_id=MEM,
                  serving=Serving(batch=1, prompt=64, out_len=32))
    assert evaluate(sc).fits
    k = speech_kpis(sc)
    assert k["kind"] == "tts" and k["rtf"] < 1.0


def test_energy_thermal_tco_consistency_333_341():
    """Pins from the 0.342 consistency audit — no NPU public numbers this pass."""
    F = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    pj3, pj4 = F["pJ_bit_dram_hbm"][0], F["pJ_bit_dram_hbm4"][0]
    assert abs(pj4 - pj3 / 1.4) < 1e-12
    assert "4.14" in MEM_PROXY["HBM4"] and "5.80" in MEM_PROXY["HBM4"]
    assert "2.67" in MEM_PROXY["LPDDR6"]
    table = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))
    assert abs(table.pJ_bit_dram_hbm - pj3) < 1e-12
    assert abs(table.pJ_bit_dram_hbm4 - pj4) < 1e-12
    pp0 = package_power(300.0, 700.0, 1, 50.0, None, 0.0)
    pp1 = package_power(300.0, 700.0, 1, 50.0, None, 40.0)
    assert abs(pp0["static_floor_W"] - pp1["static_floor_W"]) < 1e-9
    assert pp1["standby_W"] == 40.0 and pp1["pkg_W"] > pp0["pkg_W"]
    assert hbm_standby_W({"hbm_static": {"kind": "LPDDR", "standby_W": 2.5}}) == 2.5
    chip = CHIPS["100T"]
    r3 = evaluate(Scenario(model="qwen3-8b", chip=chip, mem_id="hbm3e_8s_8h24g_9200",
                           serving=Serving(batch=4, prompt=256, out_len=64)))
    r4 = evaluate(Scenario(model="qwen3-8b", chip=chip, mem_id="hbm4_8s_12h24g_10000",
                           serving=Serving(batch=4, prompt=256, out_len=64)))
    rl = evaluate(Scenario(model="qwen3-8b", chip=chip, mem_id="lpddr5x_4x64_8533_16g",
                           serving=Serving(batch=4, prompt=256, out_len=64)))
    assert dram_pJ_field(r3, table) == "pJ_bit_dram_hbm"
    assert dram_pJ_field(r4, table) == "pJ_bit_dram_hbm4"
    assert dram_pJ_field(rl, table) == "pJ_bit_dram_lpddr"
    e_hi = energy_report(r3, EnergyTable(pJ_bit_dram_hbm=pj3, pJ_mac=0.0, pJ_vec=0.0,
                                         pJ_bit_sram=0.0, pJ_bit_phy_hbm=0.0))
    e_lo = energy_report(r3, EnergyTable(pJ_bit_dram_hbm=pj4, pJ_mac=0.0, pJ_vec=0.0,
                                         pJ_bit_sram=0.0, pJ_bit_phy_hbm=0.0))
    assert abs(e_hi["J_by_action"]["dram"] / e_lo["J_by_action"]["dram"] - pj3 / pj4) < 1e-9


def test_speech_count_342():
    assert len(SPEECH_MODELS) == 40
    assert "ming-omni-tts-0.5b" in SPEECH_MODELS


def test_changelog_no_ui_patch_342():
    cl = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert "## [0.342.0]" in cl
    sec = cl.split("## [0.342.0]")[1].split("## [")[0]
    assert "UI patch not applied" in sec
    assert "consistency" in sec.lower() or "Consistency" in sec
