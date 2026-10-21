"""0.292 (MODEL §237): Jais-3-MoVA — mixture-of-value-experts attention (routed value projections, sigmoid v_router
with selection bias, silu experts, softplus attention output gate; modeling_k2_horizon K2HorizonMoVAAttention);
removal of the 0.291 per-run test report cache."""
import math
from pathlib import Path

from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, moe_routing
from accel_dse.core.model import load_release

MOVA = {"jais-3-mova-36b-a4b": "inception42/Jais-3-MoVA-36B-A4B", "jais-3-mova-36b-a4b-fp8": "IFM/Jais-3-MoVA-36B-A4B-FP8"}
H, KV, HD, EV, KV_K = 2560, 8, 128, 64, 4


def test_mova_entries_exact_listed_as_released_dtypes():
    idx = {e["id"]: e for e in entries()}
    for mid, hf in MOVA.items():
        e = idx[mid]
        assert e["hf_id"] == hf and e["provider"] == "inception" and e["family"] == "Jais 3" and e["listed"], mid
        pc = get_model(mid).param_check()
        assert pc["ours"] == pc["release"] == 37_444_792_020 and pc["rel_err"] == 0.0, (mid, pc)
    bf = dict(get_model("jais-3-mova-36b-a4b").formats)
    assert {r.fmt for r in bf.values()} == {"fp32"}                       # inception42 safetensors are fp32
    f8 = dict(get_model("jais-3-mova-36b-a4b-fp8").formats)
    assert f8["expert"].fmt == "fp8" and all(f8[r].fmt == "bf16" for r in ("attn", "router", "shared_expert", "mlp"))


def test_mova_layer_structure_from_config():
    m = get_model("jais-3-mova-36b-a4b")
    c = load_release(MOVA["jais-3-mova-36b-a4b"])["config"]
    assert (c["mova_num_experts"], c["mova_num_experts_per_tok"], c["attention_gate_func"]) == (EV, KV_K, "softplus")
    mova = [L for L in m.layers if L.core.v_experts]
    assert len(mova) == 45 and all(L.ffn.kind == "moe" for L in mova)          # every sparse layer, not mlp_only 0–2
    for L in m.layers[:3]:
        assert not L.core.v_experts and [l.name for l in L.attn_linears] == ["q", "k", "v", "o"]
    L = mova[0]
    by = {l.name: l for l in L.attn_linears}
    assert "v" not in by and by["v_router"].n == EV and by["v_experts"].params == EV * H * KV * HD
    assert by["q"].n == 2 * 32 * HD                                            # q_proj + gate_proj [H·hd, h]
    assert all(L.core.out_gate_fn == "softplus" for L in m.layers) and L.core.v_topk == KV_K
    assert L.misc_params == 2 * H + 100 + EV                                   # norms + MoE router bias + v_router bias
    inactive = 45 * (EV - KV_K) * H * KV * HD
    assert m.params() - m.active_params() - inactive == sum(
        (Lx.ffn.n_experts - Lx.ffn.top_k) * 3 * H * Lx.ffn.d_expert for Lx in mova)


def test_mova_ops_touched_value_experts_only():
    m = get_model("jais-3-mova-36b-a4b-fp8")
    li = next(i for i, L in enumerate(m.layers) if L.core.v_experts)
    for b in (1, 8, 64):
        ops = {o.name: o for o in build_rank_ops(m, li, Phase("decode", b, 1, ctx=4096))}
        r = moe_routing(EV, KV_K, b, 1)
        v = ops["attn.v_experts"]
        assert (v.count, v.m, v.k, v.n) == (r.hit, r.m_e, H, KV * HD) and v.w_params == r.hit * H * KV * HD
        assert r.hit == min(EV, math.ceil(EV * (1 - (1 - KV_K / EV) ** b) - 1e-9))
        assert abs(v.share - r.pairs_local / (r.hit * r.m_e)) < 1e-12 and "attn.v" not in ops
        assert ops["v_expert_combine"].vec == b * KV_K * KV * HD * 6 + b * EV * 6
        assert ops["attn_out_gate"].vec == b * 32 * HD * 5
    pre = {o.name: o for o in build_rank_ops(m, li, Phase("prefill", 1, 4096))}["attn.v_experts"]
    assert pre.count == EV and pre.m == 4096 * KV_K // EV                         # prefill touches every value expert
    tp2 = {o.name: o for o in build_rank_ops(m, li, Phase("decode", 8, 1, ctx=4096), Shard(tp=2, ep=2))}
    assert tp2["attn.v_experts"].n == KV * HD // 2                              # attention TP: half the KV heads


def test_other_jais_unchanged():
    for mid in ("jais-3-7b", "jais-3-375b-a23b-fp8"):
        m = get_model(mid)
        assert all(not L.core.v_experts and not L.core.out_gate_fn for L in m.layers), mid
        assert m.param_check()["rel_err"] == 0.0


def test_runcache_removed():
    t = Path(__file__).resolve().parent
    assert not (t / "runcache.py").exists()
    hits = [p.name for p in t.glob("*.py") if p.name != Path(__file__).name and "runcache" in p.read_text()]
    assert not hits, hits


def test_runner_orders_longest_first():
    import importlib.util
    t = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("rt292", t / "run_tests.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    mods = [t / n for n in ("test_a.py", "test_core_057.py", "test_b.py", "test_c.py")]
    got = [m.name for m in rt.order_longest_first(mods, {"test_b.py": 300.0, "test_c.py": 5.0, "x": "?"})]
    assert got == ["test_core_057.py", "test_b.py", "test_c.py", "test_a.py"]   # unknown SLOW_FIRST, then by time
    real = rt.order_longest_first(sorted(t.glob("test_*.py")), rt._durations())
    assert sorted(real) == sorted(t.glob("test_*.py"))                         # same set: only the order changes
