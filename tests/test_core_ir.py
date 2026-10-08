"""L1/L2 — per-rank IR conservation and stage-plan tests."""
from __future__ import annotations

import math

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, moe_routing, head_ops
from accel_dse.core.memplan import stage_storage
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
