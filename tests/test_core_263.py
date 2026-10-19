"""0.263: idle-aware incremental SLC leakage; 25/85 C one-click SLC energy compare; load-aware pin."""
from dataclasses import replace
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import slc_energy_report
from accel_dse.core.scenario import Scenario, Serving

LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-30b-a3b"
ROOT = Path(__file__).resolve().parents[1]


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _moe(gib, idle=False, **sv):
    ch = replace(CHIPS["100T"], slc_mib=gib * 1024, slc_policy="dip") if gib else CHIPS["100T"]
    t = _t(idle_W=20.0) if idle else _t()
    r = evaluate(Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8, **sv)))
    return energy_report(r, t), r


def test_idle_charges_slc_not_shared_sram():
    e8, _ = _moe(8, idle=True)
    assert abs(e8["sram_leak_W_card"] - 8 * 1024 * _t().uW_per_mib_sram_leak * 1e-6) < 1e-9
    assert "增量" in e8["sram_leak_scope"]
    e8b, _ = _moe(8, idle=False)
    assert e8b["sram_leak_W_card"] > e8["sram_leak_W_card"]


def test_temps_compare_25_vs_85():
    base = Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8))
    r = slc_energy_report(base, _t(), [M], gibs=(0, 4, 8, 16, 32), temps=(25, 85))
    assert r["compare"] and r["temps"] == [25.0, 85.0] and len(r["summary"]) == 2
    cold, hot = r["summary"][0], r["summary"][1]
    assert cold["opt_J_gib"] == 32.0 and cold["crossover_gib"] is None
    assert hot["opt_J_gib"] == 0.0 and hot["crossover_gib"] == 4.0


def test_api_temps_and_ui():
    assert "/api/insight_slc" in serve.POST_ROUTES
    out = api.api_insight_slc({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"},
                                            "serving": {"batch": 8}},
                               "energy": {"energy_preset": "tpuv4i_7nm"},
                               "gibs": [0, 4, 32], "temps": [25, 85]})
    assert out["compare"] and out["summary"][1]["crossover_gib"] == 4.0
    html = (ROOT / "accel_dse/web/index.html").read_text()
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert 'id="ins-slc-cmp"' in html and "temps = [25, 85]" in js


def test_load_aware_pin():
    load = tuple(1.0 / (i + 1) for i in range(128))
    ch = replace(CHIPS["100T"], slc_mib=16 * 1024, slc_policy="pin")
    u = evaluate(Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8)))
    z = evaluate(Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8, moe_expert_load=load)))
    mu, mz = u.stages[0].mem, z.stages[0].mem
    assert abs(mu.slc_frac_exp - mu.slc_frac_exp_bytes) < 1e-12
    assert mz.slc_frac_exp > mz.slc_frac_exp_bytes and z.tpot < u.tpot
    d0 = sum(st.dram["total"] for st in evaluate(Scenario(
        model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8, moe_expert_load=load))).stages)
    assert abs(sum(s.dram["total"] + s.dram.get("slc", 0.0) for s in z.stages) - d0) <= 1e-6 * d0
