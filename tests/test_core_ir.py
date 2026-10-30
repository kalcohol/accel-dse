"""L1/L2 — per-rank IR conservation and stage-plan tests."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, moe_routing, head_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import _PRODUCT_TAG_ORDER
from accel_dse.core import vector_rules as vr
from accel_dse.core.parallel import Layout, enumerate_layouts, plan_stages, stage_layer_counts


def _sum(ops, attr, pred=lambda o: True):
    return sum(getattr(o, attr) for o in ops if pred(o))


def test_stage_plan_contiguous_remainder_first():
    assert stage_layer_counts(61, 8) == [8, 8, 8, 8, 8, 7, 7, 7]
    st = plan_stages(61, 8)
    assert st[0].first == 0 and st[-1].last == 61
    assert all(a.last == b.first for a, b in zip(st, st[1:]))
    assert st[0].has_embed and not st[0].has_head and st[-1].has_head
    for pp in range(1, 62):
        assert sum(stage_layer_counts(61, pp)) == 61
    try:
        stage_layer_counts(4, 5)
        raise AssertionError("pp > layers accepted")
    except ValueError:
        pass


def test_tp_conservation_dense_gemm_flops_and_weights():
    s = get_model("qwen3-32b")          # 64 q heads, 8 kv heads: divisible by tp ≤ 8
    ph = Phase("decode", 8, 1, 1024)
    g = build_rank_ops(s, 3, ph)
    for tp in (2, 4, 8):
        r = build_rank_ops(s, 3, ph, Shard(tp=tp))
        isg = lambda o: o.kind == "gemm"
        assert abs(_sum(r, "flops", isg) * tp - _sum(g, "flops", isg)) < 1e-6 * _sum(g, "flops", isg)
        assert abs(_sum(r, "w_params", isg) * tp - _sum(g, "w_params", isg)) == 0
        isa = lambda o: o.kind == "attn"
        assert abs(_sum(r, "flops", isa) * tp - _sum(g, "flops", isa)) < 1e-6 * _sum(g, "flops", isa)
        # KV traffic split across ranks (GQA, n_kv divisible by tp)
        assert abs(_sum(r, "kv_read") * tp - _sum(g, "kv_read")) < 1e-6 * _sum(g, "kv_read")
        assert any(o.kind == "comm" and o.comm_kind == "allreduce" for o in r)


def test_tp_kv_heads_replicated_when_tp_exceeds_kv():
    s = get_model("qwen3-8b")           # 8 kv heads
    ph = Phase("decode", 4, 1, 1024)
    g = build_rank_ops(s, 0, ph)
    r = build_rank_ops(s, 0, ph, Shard(tp=16))
    # each rank keeps ceil(8/16)=1 kv head → 16 ranks hold 2× the global KV
    assert abs(_sum(r, "kv_read") * 16 - 2 * _sum(g, "kv_read")) < 1e-6 * _sum(g, "kv_read")


def test_mla_latent_replicated_on_tp_ranks():
    s = get_model("deepseek-v3")
    ph = Phase("decode", 4, 1, 2048)
    g = build_rank_ops(s, 0, ph)
    r = build_rank_ops(s, 0, ph, Shard(tp=8, ep=8))
    assert abs(_sum(r, "kv_read") - _sum(g, "kv_read")) < 1e-6 * _sum(g, "kv_read")


def test_moe_routing_conservation_and_bounds():
    for E, k in ((256, 8), (128, 8), (8, 2), (512, 10)):
        for T in (1, 3, 16, 64, 1024):
            for ep in (1, 2, 8):
                r = moe_routing(E, k, T, ep)
                assert abs(r.pairs_local * ep - T * k) < 1e-9
                assert 1 <= r.hit <= r.n_local
                assert r.m_e * r.hit >= r.pairs_local - 1e-9
                g = moe_routing(E, k, T, 1)
                assert r.hit * ep >= g.hit or r.hit == r.n_local
    r = moe_routing(256, 8, 1, 1)
    assert r.hit == 8 and r.m_e == 1


def test_ep_expert_storage_conservation():
    s = get_model("deepseek-v3")
    full = stage_storage(s, 3, 4, False, False, Shard(), 1024)
    for ep in (2, 4, 8, 16, 32):
        part = stage_storage(s, 3, 4, False, False, Shard(tp=ep, ep=ep), 1024)
        assert abs(part.expert_w * ep - full.expert_w) < 1e-6 * full.expert_w


def test_dense_and_shared_replicated_across_ep():
    s = get_model("deepseek-v3")
    ph = Phase("decode", 64, 1, 1024)
    a = build_rank_ops(s, 5, ph, Shard(tp=1, dp=8, ep=8))
    shared = [o for o in a if o.name.startswith("moe.shared")]
    g = [o for o in build_rank_ops(s, 5, ph) if o.name.startswith("moe.shared")]
    # attention-DP: every rank runs the full shared expert on its own 64/8 tokens
    assert sum(o.w_params for o in shared) == sum(o.w_params for o in g)
    assert shared[0].m == 8 and g[0].m == 64


def test_pp_conservation_of_stored_weights():
    for mid in ("deepseek-v3", "qwen3-32b", "qwen3-next-80b-a3b"):
        s = get_model(mid)
        one = stage_storage(s, 0, s.n_layers, True, True, Shard(), 4096)
        for pp in (2, 4, 8):
            tot = sum(stage_storage(s, st.first, st.last, st.has_embed, st.has_head, Shard(), 4096).weights
                      for st in plan_stages(s.n_layers, pp))
            assert abs(tot - one.weights) < 1e-6 * one.weights, (mid, pp)


def test_layout_enumeration_valid_and_complete():
    for cards in (1, 2, 4, 8, 16):
        lays = enumerate_layouts(cards, 61, True)
        assert all(l.cards == cards and l.valid_for(True) for l in lays)
        assert len(set(lays)) == len(lays)
        d = enumerate_layouts(cards, 64, False)
        assert all(l.dp == 1 and l.ep == 1 for l in d)
    assert Layout().cards == 1


def test_lm_head_vocab_parallel():
    s = get_model("qwen3-8b")
    ph = Phase("decode", 4, 1, 100)
    g = [o for o in head_ops(s, ph) if o.kind == "gemm"][0]
    r = [o for o in head_ops(s, ph, Shard(tp=4)) if o.kind == "gemm"][0]
    assert r.n == math.ceil(s.vocab / 4) and g.n == s.vocab


def _gemm_sig(ops):
    return [
        (o.name, o.k, o.n, o.m, o.flops, o.w_params, o.count)
        for o in ops if o.kind == "gemm"
    ]


def _ffn_names(ops):
    keep = {"act", "mlp_mul", "expert_act", "expert_b.act"}
    out = []
    for o in ops:
        if o.kind == "comm":
            continue
        if o.name in keep or o.name.startswith(("mlp.", "moe.", "expert")):
            out.append(o.name)
    return tuple(out)


def _op_sig(ops):
    return [
        (o.name, o.kind, o.vec, o.sfu, o.a_fmt, o.k, o.n, o.m, o.flops)
        for o in ops
    ]


def _bare(model):
    return replace(
        model,
        layers=tuple(replace(L, vec_kind="", product_rule="") for L in model.layers),
        vec_kind="")


def _retarget(model, li: int, kind: str, rule: str):
    layer = replace(model.layers[li], vec_kind=kind, product_rule=rule)
    layers = model.layers[:li] + (layer,) + model.layers[li + 1:]
    return replace(model, layers=layers)


def _mlp_mul(model, li: int, ph: Phase, shard: Shard | None = None):
    ops = build_rank_ops(model, li, ph, shard or Shard())
    return [o for o in ops if o.name == "mlp_mul"]


def test_silu_product_conserves_gemm_and_order():
    """Shared silu(gate)*up billing keeps GEMM shape and inserts mlp_mul in place."""
    ph = Phase("prefill", 1, 4096, 0)
    expect = {
        ("seed-oss-36b", 0): (
            "mlp.gate_up", "act", "mlp_mul", "mlp.down"),
        ("qwen3-8b", 0): (
            "mlp.gate_up", "act", "mlp_mul", "mlp.down"),
        ("exaone-4.0-32b", 0): (
            "mlp.gate_up", "act", "mlp_mul", "mlp.down"),
        ("glm-5", 0): (
            "mlp.gate_up", "act", "mlp_mul", "mlp.down"),
        ("glm-5", 3): (
            "moe.router", "moe.shared_gate_up", "mlp_mul", "moe.shared_down",
            "expert.gate_up", "expert_act", "mlp_mul", "expert.down"),
        ("mixtral-8x22b", 0): (
            "moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down"),
        ("muse-spark-open", 0): (
            "moe.router", "moe.latent_up", "moe.latent_down",
            "expert.gate_up", "expert_act", "mlp_mul", "expert.down"),
        ("qwen3.5-397b-a17b", 0): (
            "moe.router", "moe.shared_gate_up", "mlp_mul", "moe.shared_down",
            "moe.shared_expert_gate",
            "expert.gate_up", "expert_act", "mlp_mul", "expert.down"),
        ("ernie-4.5-300b-a47b-pt", 0): (
            "mlp.gate_up", "act", "mlp_mul", "mlp.down"),
        ("ernie-4.5-300b-a47b-pt", 3): (
            "moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down"),
    }
    for (mid, li), names in expect.items():
        model = get_model(mid)
        tagged = build_rank_ops(model, li, ph, Shard())
        bare = build_rank_ops(_bare(model), li, ph, Shard())
        assert _gemm_sig(tagged) == _gemm_sig(bare), (mid, li)
        assert _ffn_names(tagged) == names, (mid, li, _ffn_names(tagged))
        assert "mlp_mul" not in _ffn_names(bare)


def test_dense_silu_product_scales_with_tokens_and_tp():
    model = get_model("seed-oss-36b")
    assert model.layers[0].ffn.d_ff % 2 == 0
    narrow = _mlp_mul(model, 0, Phase("prefill", 1, 128, 0))
    wide = _mlp_mul(model, 0, Phase("prefill", 1, 256, 0))
    split = _mlp_mul(model, 0, Phase("prefill", 1, 128, 0), Shard(tp=2))
    assert len(narrow) == 1 and wide[0].vec == narrow[0].vec * 2
    assert split[0].vec * 2 == narrow[0].vec
    assert narrow[0].a_fmt == model.act_fmt and narrow[0].sfu == 0.0


def test_product_rule_id_matches_frozen_kind():
    """An unbound layer selects the same dense product by rule id, not a new vec_kind."""
    ph = Phase("prefill", 1, 256, 0)
    seed = get_model("seed-oss-36b")
    tagged = build_rank_ops(seed, 0, ph, Shard())
    by_rule = build_rank_ops(
        _retarget(seed, 0, "", vr.RULE_DENSE), 0, ph, Shard())
    assert _op_sig(by_rule) == _op_sig(tagged)
    mix = get_model("mixtral-8x22b")
    tagged_m = build_rank_ops(mix, 0, ph, Shard())
    by_plain = build_rank_ops(
        _retarget(mix, 0, "", vr.RULE_ROUTED_PLAIN), 0, ph, Shard())
    assert _op_sig(by_plain) == _op_sig(tagged_m)
    assert _mlp_mul(mix, 0, ph)[0].a_fmt == mix.act_fmt
    muse = get_model("muse-spark-open")
    role = muse.layers[0].ffn.expert_role or "expert"
    assert _mlp_mul(muse, 0, ph)[-1].a_fmt == muse.act_fmt_for(role)


def test_product_rule_rejects_bad_geometry_and_unknown_id():
    seed = get_model("seed-oss-36b")
    ungated = replace(seed.layers[0].ffn, gated=False)
    bad = replace(seed, layers=(replace(seed.layers[0], ffn=ungated),) + seed.layers[1:])
    try:
        build_rank_ops(bad, 0, Phase("prefill", 1, 32, 0), Shard())
        raise AssertionError("ungated seed accepted")
    except ValueError as exc:
        assert str(exc) == "seed swiglu mlp is not gate_up then down"
    unknown = _retarget(seed, 0, "", "not_a_product_rule")
    try:
        build_rank_ops(unknown, 0, Phase("prefill", 1, 32, 0), Shard())
        raise AssertionError("unknown product rule accepted")
    except ValueError as exc:
        assert "unknown product rule" in str(exc)


def test_product_registry_order_and_frozen_kinds():
    assert _PRODUCT_TAG_ORDER == vr.TAG_ORDER
    assert set(vr.LEGACY_TAGGERS).isdisjoint(vr.ORDER_SLOTS)
    assert vr.rule_ids() == tuple(vr.RULES)
    assert "magistral" not in " ".join(vr.FROZEN_PRODUCT_KINDS)
    src = (Path(__file__).resolve().parents[1] / "accel_dse" / "core" / "ir.py").read_text()
    for kind in (
        "seed_swiglu", "olmo3_swiglu", "jais3_swiglu", "ernievl_swiglu",
        "glm5_swiglu", "qwen35_moe", "muse_swiglu", "mixtral_swiglu",
        "minimax_swiglu",
    ):
        assert f'vec_kind == "{kind}"' not in src
    assert 'vec_kind == "kimi_situ"' in src
    assert "_emit_dense_silu" in src

