"""0.260: token-refiner attention cores for HunyuanVideo / MiniMax-H3 (io_attn norm field); dip per-class reuse
(every-step classes retained before routed experts, memplan.dip_split); UI control for workload.cfg_branches;
SLC leakage charged as on-die SRAM, and core-rail leakage ∝ V under DVFS (AccelWattch MICRO 2021 eq. 2)."""
from dataclasses import replace
from pathlib import Path

from accel_dse.core import catalog as C
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard, full_io_ops
from accel_dse.core.memplan import DIP_EPS, dip_hit, dip_split
from accel_dse.core.powercap import _point, volt
from accel_dse.core.scenario import Scenario, Serving

LP = "lpddr5x_4x64_8533_16g"
WEB = Path(__file__).resolve().parent.parent / "accel_dse" / "web"


def test_refiner_io_attn_hunyuan_h3():
    hv, h3 = C.get_model("hunyuanvideo"), C.get_model("minimax-h3")
    assert hv.io_attn and all(a[2:] == (24, 128, 1, "") for a in hv.io_attn)
    assert h3.io_attn and all(a[2] == 56 and a[3] == 128 and a[5] == "qknorm" for a in h3.io_attn)
    for m in (hv, h3):
        assert abs(m.param_check()["rel_err"]) < 1e-5                          # io_attn adds no weights
    ph = Phase("full", 1, 4096, 0, frames=1)
    ops = {o.name: o for o in full_io_ops(hv, ph, Shard(), "pre")}
    names = [n for n in ops if n.startswith("io.refiner")]
    assert names and not any(n.endswith("qk_norm_rope") for n in names)          # "" → no per-token vector op
    qk = next(o for n, o in ops.items() if n.endswith(".qk"))
    assert qk.m == qk.n and qk.k == 128 and qk.count == 24
    ops3 = {o.name: o for o in full_io_ops(h3, ph, Shard(), "pre")}
    v = [o for n, o in ops3.items() if n.endswith("qk_norm_rope")]
    q3 = next(o for n, o in ops3.items() if n.endswith(".qk"))
    assert v and q3.count == 56 and v[0].vec == q3.m * 56 * 128 * 6               # QK-norm only (no RoPE term)


def test_dip_split_rules():
    K, C_, E = 10.0, 6.0, 30.0
    assert dip_split(20.0, 12.0, 0.0, 0.0) == (1.0, 1.0)
    assert abs(dip_split(8.0, 20.0, 0.0, 0.0)[0] - dip_hit(8.0, 20.0)) < 1e-15    # E = 0 → 0.258 formula
    assert dip_split(40.0, 6.0, 30.0, 3.0) == (1.0, 1.0)
    fc, fe = dip_split(K, C_, E, 3.0)
    Kp = K - DIP_EPS * max(0.0, C_ + 3.0 - K)
    assert fc == 1.0 and abs(fe - (Kp - C_) / E) < 1e-15 and fe <= fc            # every-step classes first
    fc2, fe2 = dip_split(4.0, C_, E, 3.0)
    assert fe2 == 0.0 and 0 < fc2 < 1
    prev = (0.0, 0.0)
    for k in range(1, 40):
        f = dip_split(float(k), C_, E, 3.0)
        assert f[0] >= prev[0] and f[1] >= prev[1]
        prev = f


def _moe(pol, gib, mid="qwen3-30b-a3b"):
    ch = replace(CHIPS["100T"], slc_mib=gib * 1024, slc_policy=pol) if gib else CHIPS["100T"]
    return evaluate(Scenario(model=mid, chip=ch, mem_id=LP, serving=Serving(batch=8)))


def test_moe_dip_conserves_and_orders():
    d0 = sum(st.dram["total"] for st in _moe("pin", 0).stages)
    prev = None
    for gib in (4, 16, 32):
        r = _moe("dip", gib)
        m = r.stages[0].mem
        assert abs(sum(s.dram["total"] + s.dram.get("slc", 0.0) for s in r.stages) - d0) <= 1e-6 * d0
        assert m.slc_frac_exp <= m.slc_frac
        assert prev is None or r.tpot <= prev
        prev = r.tpot
    assert _moe("dip", 16).tpot < _moe("pin", 16).tpot < _moe("lru", 16).tpot


def test_cfg_branches_ui_control():
    w = C.model_info("moss-image-base")["workload"] if hasattr(C, "model_info") else C._workload_dict(C.get_model("moss-image-base"))
    assert w["uncond"] == {"text_tokens": 10, "skip_layers": [22]}
    assert C._workload_dict(C.get_model("hunyuanvideo"))["uncond"] is None
    html, js = (WEB / "index.html").read_text(), (WEB / "app.js").read_text()
    assert 'id="w-cfg_branches"' in html and "uncond-only" in html
    assert "w-cfg_branches" in js and ".uncond-only" in js


T = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))


def test_slc_leakage_charged():
    e0 = energy_report(_moe("dip", 0), T)
    for gib in (8, 32):
        e = energy_report(_moe("dip", gib), T)
        assert abs(e["sram_leak_W_card"] - e0["sram_leak_W_card"] - gib * 1024 * T.uW_per_mib_sram_leak * 1e-6) < 1e-9
        assert e["sram_leak_scope"].endswith("+ slc")
    ti = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", "idle_W": 20.0}))
    # 0.263: shared SRAM suppressed by idle; SLC still charged as an increment
    ei = energy_report(_moe("dip", 8), ti)
    assert abs(ei["sram_leak_W_card"] - 8 * 1024 * ti.uW_per_mib_sram_leak * 1e-6) < 1e-9
    assert "增量" in ei["sram_leak_scope"] or "slc" in ei["sram_leak_scope"]


def test_leakage_scales_with_v_under_dvfs():
    sc = Scenario(chip=replace(CHIPS["1P"], state_sram_mib=1024), model="qwen3-8b").replace("serving.batch", 64)
    r = evaluate(sc)
    e1 = energy_report(r, T)
    for curve, s in (("rx5700xt_n7", 0.5), ("dfs", 0.5)):
        _, e = _point(r, T, curve, s)
        assert abs(e["sram_leak_W_card"] - e1["sram_leak_W_card"] * volt(curve, s)) < 1e-9
        assert e.get("idle_W_used") == e1.get("idle_W_used") and e.get("hbm_static_W_card") == e1.get("hbm_static_W_card")
