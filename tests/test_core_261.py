"""0.261: SRAM / SLC leakage vs die temperature (energy.sram_temp_C; 3 nm macro measured at 25 and 85 °C); MiniMax-H3
token refiner without RoPE now sourced; dip SLC expert hits under a measured expert load (A0 popularity order)."""
import math
from dataclasses import replace
from pathlib import Path

from accel_dse import api
from accel_dse.core import catalog as C
from accel_dse.core.energy import (SRAM_LEAK_T, EnergyTable, apply_energy_preset, energy_report, leak_temp_factor,
                                   sram_leak_uW_per_mib)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard, full_io_ops
from accel_dse.core.memplan import dip_exp_hit
from accel_dse.core.powercap import _point, volt
from accel_dse.core.scenario import Scenario, Serving

LP = "lpddr5x_4x64_8533_16g"
ROOT = Path(__file__).resolve().parents[1]


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_temperature_factor_source_points():
    assert SRAM_LEAK_T["ratio_hot"] == 73.5 / 15.5
    assert leak_temp_factor(None) == 1.0 and abs(leak_temp_factor(25) - 1.0) < 1e-15
    assert abs(leak_temp_factor(85) - 73.5 / 15.5) < 1e-12
    assert abs(leak_temp_factor(55) - math.sqrt(73.5 / 15.5)) < 1e-12           # exponential in T between the points
    xs = [leak_temp_factor(t) for t in range(0, 126, 5)]
    assert all(b > a for a, b in zip(xs, xs[1:]))
    for bad in (-1.0, 126.0):
        try:
            _t(sram_temp_C=bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert sram_leak_uW_per_mib(_t()) == _t().uW_per_mib_sram_leak
    assert abs(sram_leak_uW_per_mib(_t(sram_temp_C=85)) - _t().uW_per_mib_sram_leak * 73.5 / 15.5) < 1e-9


def _moe(gib, **sv):
    ch = replace(CHIPS["100T"], slc_mib=gib * 1024, slc_policy="dip") if gib else CHIPS["100T"]
    return evaluate(Scenario(model="qwen3-30b-a3b", chip=ch, mem_id=LP, serving=Serving(batch=8, **sv)))


def test_hot_die_leakage_in_report_and_band():
    r = _moe(32)
    e25, e85 = energy_report(r, _t()), energy_report(r, _t(sram_temp_C=85))
    assert e25["sram_leak_temp_C"] == 25.0 and e85["sram_leak_temp_C"] == 85
    assert abs(e85["sram_leak_W_card"] - e25["sram_leak_W_card"] * 73.5 / 15.5) < 1e-9
    assert abs(e25["sram_leak_W_card_85C"] - e85["sram_leak_W_card"]) < 1e-9
    assert e85["J_per_unit"] > e25["J_per_unit"]
    # hot die: a 32 GiB SRAM SLC costs more J/token than no SLC (its leakage outgrows the DRAM energy it saves)
    assert e85["J_per_unit"] > energy_report(_moe(0), _t(sram_temp_C=85))["J_per_unit"]
    assert e25["J_per_unit"] < energy_report(_moe(0), _t())["J_per_unit"]
    _, ec = _point(r, _t(sram_temp_C=85), "rx5700xt_n7", 0.5)                    # leakage ∝ V on top of T
    assert abs(ec["sram_leak_W_card"] - e85["sram_leak_W_card"] * volt("rx5700xt_n7", 0.5)) < 1e-9
    assert "sram_temp_C" not in e25["missing"]


def test_ui_and_api_field():
    html = (ROOT / "accel_dse" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "accel_dse" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="e-sram_temp_C"' in html and "'sram_temp_C'" in js
    assert EnergyTable.from_dict({"sram_temp_C": 70}).sram_temp_C == 70


def test_h3_refiner_no_rope_sourced():
    m = C.get_model("minimax-h3")
    assert all(a[5] == "qknorm" for a in m.io_attn)
    assert any("No AdaLN and no rotary embedding" in n for n in m.notes)
    ph = Phase("full", 1, 4096, 0, frames=1)
    v = [o for o in full_io_ops(m, ph, Shard(), "pre") if o.name.endswith("qk_norm_rope")]
    q = next(o for o in full_io_ops(m, ph, Shard(), "pre") if o.name.endswith(".qk"))
    assert v and v[0].vec == q.m * 56 * 128 * 6                                   # QK-norm only, no RoPE term


def test_dip_exp_hit_rules():
    assert dip_exp_hit((0.1,) * 16, 0.5, 0.25) == 0.25                           # uniform: read share = byte share
    assert dip_exp_hit((0.3, 0.01), 0.5, 0.0) == 0.0 and dip_exp_hit((0.3, 0.01), 0.5, 1.0) == 1.0
    p = tuple(8 / 128 / (i + 1) for i in range(128))
    xs = [dip_exp_hit(p, 0.3, f / 20) for f in range(21)]
    assert all(b >= a for a, b in zip(xs, xs[1:])) and all(x >= f / 20 for x, f in zip(xs, range(21)))
    assert dip_exp_hit(p, 0.3, 0.25) > 0.4


def test_measured_load_raises_expert_hits_and_conserves():
    E = 128
    load = tuple(1.0 / (i + 1) for i in range(E))
    d0 = sum(st.dram["total"] for st in _moe(0, moe_expert_load=load).stages)
    for gib in (8, 16, 32):
        u, z = _moe(gib), _moe(gib, moe_expert_load=load)
        mu, mz = u.stages[0].mem, z.stages[0].mem
        assert mu.slc_frac_exp == mu.slc_frac_exp_bytes                          # uniform: unchanged from 0.260
        assert mz.slc_frac_exp_bytes == mu.slc_frac_exp_bytes and mz.slc_frac_exp > mz.slc_frac_exp_bytes
        assert z.tpot < u.tpot
        assert abs(sum(s.dram["total"] + s.dram.get("slc", 0.0) for s in z.stages) - d0) <= 1e-6 * d0
    flat = _moe(16, moe_expert_load=(1.0,) * E)
    assert abs(flat.tpot - _moe(16).tpot) < 1e-12                                 # a flat measured load = uniform


def test_api_reports_bytes_share():
    load = tuple(1.0 / (i + 1) for i in range(128))
    d = api._stage_dict(_moe(16, moe_expert_load=load).stages[0])["mem"]
    assert d["slc_frac_exp_bytes"] < d["slc_frac_exp"] <= 1.0
