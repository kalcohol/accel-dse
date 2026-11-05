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
        "minimax_swiglu", "ernie300_swiglu", "exaone4",
    ):
        assert f'vec_kind == "{kind}"' not in src
    assert 'vec_kind == "kimi_situ"' in src
    assert 'vec_kind == "whisper"' in src
    assert "_emit_dense_silu" in src


def test_phase_b_legacy_taggers_are_allow_row_predicates():
    """0.481: the ten hand-written swiglu taggers are AllowRow predicates.

    Baseline at 0.480 was 29 ``def _tag_*_vectors`` and 12 of them ran inside
    the product cluster. The cluster now calls only the two order slots.
    """
    import inspect
    import re

    from accel_dse.core import model as M

    model_src = (Path(__file__).resolve().parents[1] / "accel_dse" / "core" / "model.py").read_text()
    defs = re.findall(r"^def (_tag_\w+_vectors)\b", model_src, re.M)
    assert len(defs) == 19
    assert "_tag_kimi_situ_vectors" in defs and "_tag_llama3_vectors" in defs
    assert not any(name.endswith("_swiglu_vectors") or name == "_tag_qwen35_moe_vectors" for name in defs)
    covered = {
        "muse_swiglu", "mixtral_swiglu", "seed_swiglu", "olmo3_swiglu",
        "jais3_swiglu", "ernievl_swiglu", "minimax_swiglu", "ernie300_swiglu",
        "glm5_swiglu", "qwen35_moe",
    }
    assert {row.name for row in vr.ALLOW_ROWS} == covered
    for row in vr.ALLOW_ROWS:
        assert row.predicate == "_pred_" + row.name
        assert row.legacy_tagger in vr.LEGACY_TAGGERS
        fn = getattr(M, row.predicate)
        alias = getattr(M, row.legacy_tagger)
        assert callable(fn) and alias.__name__ == row.legacy_tagger
    cluster = inspect.getsource(M._tag_product_cluster)
    assert not re.search(r"_tag_\w+_vectors\s*\(", cluster)
    assert "_ORDER_SLOT_FN" in cluster
    assert set(M._ORDER_SLOT_FN) == set(vr.ORDER_SLOTS)


def test_phase_b_hot_zone_kind_compares_dropped():
    """0.480 ``_rank_ops`` had 7 ``L.vec_kind`` string compares (2 ernie300, 5 kimi).

    Predicates own the norm checks. SituAndMul stays one compare. That is an
    85% drop, above the 40% gate. Whisper erf-GELU still uses its own guard.
    """
    import re

    src = (Path(__file__).resolve().parents[1] / "accel_dse" / "core" / "ir.py").read_text()
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def _rank_ops"))
    end = next(i for i, line in enumerate(lines) if line.startswith("def layer_skew"))
    body = "\n".join(lines[start:end])
    compares = re.findall(r"vec_kind\s*(?:==|!=)\s*\"[^\"]+\"", body)
    assert compares == ['vec_kind == "kimi_situ"']
    assert 'vec_kind == "ernie300_swiglu"' not in body
    assert vr.pre_norm_fp32("whisper") and vr.pre_norm_fp32("ernie300_swiglu")
    assert vr.pre_norm_fp32("carbon_llama") and not vr.pre_norm_fp32("exaone4")
    assert vr.post_norm_fp32("exaone4") and not vr.post_norm_fp32("kimi_situ")
    assert vr.fp32_final_norm("qwen2_text") and vr.final_norm_copies("qwen2_text", 16, 8) == 2
    assert vr.final_norm_copies("whisper", 16, 8) == 1
    assert not vr.fp32_final_norm("kimi_situ")
    assert not vr.fp32_final_norm("evo2_gelu")


def test_magistral_small_bills_dense_product_by_existing_rule():
    """0.482: Magistral-Small-2506 silu(gate)*up uses dense_silu_product.

    vec_kind stays empty, so the kind has no product binding. The row is
    already in the front 60 and is not moved. Block norms stay on the
    historical dtype. No new vec_kind.
    """
    from accel_dse.core.model import _apply_unbound_product_rule, _tc, from_release, load_release

    m = get_model("magistral-small")
    assert m.vec_kind == ""
    assert all(L.vec_kind == "" and L.product_rule == vr.RULE_DENSE for L in m.layers)
    assert m.hf_id == "mistralai/Magistral-Small-2506"
    assert m.hidden == 5120 and m.n_layers == 40
    assert m.layers[0].ffn.d_ff == 32768
    assert m.layers[0].attn_linears[0].n == 32 * 128
    assert m.layers[0].attn_linears[0].n != m.hidden
    assert "magistral" not in " ".join(vr.FROZEN_PRODUCT_KINDS)
    assert vr.UNBOUND_PRODUCT_RULES["magistral-small"] == vr.RULE_DENSE
    ph = Phase("prefill", 1, 4096, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    names = [o.name for o in ops if o.name in ("mlp.gate_up", "act", "mlp_mul", "mlp.down")]
    assert names == ["mlp.gate_up", "act", "mlp_mul", "mlp.down"]
    by = {o.name: o for o in ops}
    assert by["mlp_mul"].vec == float(4096 * 32768) and by["mlp_mul"].sfu == 0.0
    assert by["mlp_mul"].a_fmt == "bf16" and by["act"].a_fmt == "bf16"
    assert by["act"].vec == 4096 * 32768 * 4 and by["act"].sfu == float(4096 * 32768 * 2)
    assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
    assert by["mlp.gate_up"].k == 5120 and by["mlp.gate_up"].n == 2 * 32768
    assert by["mlp.down"].k == 32768 and by["mlp.down"].n == 5120
    bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
    assert _gemm_sig(ops) == _gemm_sig(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in _ffn_names(build_rank_ops(bare, 0, ph, Shard()))
    split = build_rank_ops(m, 0, ph, Shard(tp=2))
    mul = next(o for o in split if o.name == "mlp_mul")
    assert mul.vec * 2 == by["mlp_mul"].vec
    head = {o.name: o for o in head_ops(m, ph, Shard())}
    assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    width = sum(next(o for o in build_rank_ops(m, li, Phase("prefill", 1, 1, 0), Shard())
                     if o.name == "mlp_mul").vec for li in range(m.n_layers))
    assert width == 40 * 32768
    fp8 = replace(m, act_fmt="fp8")
    got = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got["act"].a_fmt == "fp8" and got["mlp_mul"].a_fmt == "fp8"
    assert got["attn_norm"].a_fmt == "bf16" and got["mlp.gate_up"].w_fmt == "bf16"
    small = get_model("mistral-small-3.2-24b-instruct-2506")
    assert small.vec_kind == "mistral_small" and small.layers[0].product_rule == ""
    large = get_model("mistral-large")
    assert large.vec_kind == "mistral_small" and large.layers[0].product_rule == ""
    plain = get_model("mistral-7b")
    assert plain.vec_kind == "" and plain.layers[0].product_rule == ""
    assert "mlp_mul" not in [o.name for o in build_rank_ops(plain, 0, Phase("prefill", 1, 16, 0), Shard())]
    other = _apply_unbound_product_rule(plain, _tc(load_release(plain.hf_id)["config"]))
    assert other is plain
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "magistral small hidden_act is not silu"
    try:
        build_rank_ops(replace(m, layers=(replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False)),)
                               + m.layers[1:]), 0, Phase("prefill", 1, 16, 0), Shard())
        raise AssertionError("ungated accepted")
    except ValueError as exc:
        assert str(exc) == "dense silu product is not gate_up then down"


def test_qwen3_coder_bills_routed_product_by_existing_rule():
    """0.483: Qwen3-Coder-480B silu(gate)*up uses routed_plain_silu_product.

    vec_kind stays empty. Both rows stay outside the front 60. The
    product is the routed expert width. Block norms and q/k RMSNorm stay
    on the historical path. No new vec_kind.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import _apply_unbound_product_rule, _tc, from_release, load_release

    width = 62 * 8 * 2560
    assert width == 1_269_760
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-coder-480b-a35b"] == vr.RULE_ROUTED_PLAIN
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-coder-480b-a35b-fp8"] == vr.RULE_ROUTED_PLAIN
    front = {e["id"] for e in entries()[:60]}
    assert "qwen3-coder-480b-a35b" not in front
    assert "qwen3-coder-480b-a35b-fp8" not in front
    assert "qwen3_coder" not in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    for mid, af, ew in (
            ("qwen3-coder-480b-a35b", "bf16", "bf16"),
            ("qwen3-coder-480b-a35b-fp8", "fp8", "fp8")):
        m = get_model(mid)
        assert m.vec_kind == ""
        assert all(L.vec_kind == "" and L.product_rule == vr.RULE_ROUTED_PLAIN for L in m.layers)
        assert m.hidden == 6144 and m.n_layers == 62 and not m.mtp_layers
        assert m.layers[0].ffn.d_expert == 2560 and m.layers[0].ffn.top_k == 8
        assert m.layers[0].ffn.n_experts == 160 and m.layers[0].ffn.n_shared == 0
        assert m.layers[0].attn_linears[0].n == 96 * 128
        assert m.layers[0].attn_linears[0].n != m.hidden
        assert m.param_check()["rel_err"] == 0
        ops = build_rank_ops(m, 0, ph, Shard())
        names = _ffn_names(ops)
        assert names == ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
        by = {o.name: o for o in ops}
        assert by["mlp_mul"].vec == float(4096 * 8 * 2560) and by["mlp_mul"].sfu == 0.0
        assert by["mlp_mul"].a_fmt == af and by["mlp_mul"].flops == 0
        assert by["expert_act"].a_fmt == "bf16"
        assert by["expert_act"].vec == 4096 * 8 * 2560 * 4
        assert by["expert_act"].sfu == float(4096 * 8 * 2560 * 2)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert "q_rms" not in by and "k_rms" not in by
        assert by["expert.gate_up"].k == 6144 and by["expert.gate_up"].n == 2 * 2560
        assert by["expert.down"].k == 2560 and by["expert.down"].n == 6144
        assert by["expert.gate_up"].w_fmt == ew and by["moe.router"].w_fmt == "bf16"
        bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert _gemm_sig(ops) == _gemm_sig(bare_ops)
        assert _ffn_names(bare_ops) == ("moe.router", "expert.gate_up", "expert.down", "expert_act")
        got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
                  for li in range(m.n_layers))
        assert got == width
        head = {o.name: o for o in head_ops(m, ph, Shard())}
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    m = get_model("qwen3-coder-480b-a35b")
    split = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    mul = next(o for o in split if o.name == "mlp_mul")
    assert mul.vec * 2 == 4096 * 8 * 2560
    assert next(o for o in split if o.name == "expert.down").k == 2560
    narrow = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 2 == 4096 * 8 * 2560
    assert next(o for o in narrow if o.name == "expert.down").k == 1280
    try:
        build_rank_ops(m, 0, ph, Shard(tp=2))
        raise AssertionError("tp without ep accepted")
    except ValueError as exc:
        assert "ep" in str(exc) and "tp" in str(exc)
    fp8 = replace(m, act_fmt="fp8")
    got = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got["mlp_mul"].a_fmt == "fp8" and got["expert_act"].a_fmt == "bf16"
    assert got["expert.gate_up"].a_fmt == "fp8" and got["expert.gate_up"].w_fmt == "bf16"
    assert got["attn_norm"].a_fmt == "bf16" and got["ffn_norm"].a_fmt == "bf16"
    for mid in ("glm-4.5", "glm-4.6", "qwen3-coder-next-fp8", "qwen3-30b-a3b"):
        other = get_model(mid)
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("magistral-small").layers[0].product_rule == vr.RULE_DENSE
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "qwen3 coder hidden_act is not silu"
    shared = replace(m.layers[0], ffn=replace(m.layers[0].ffn, n_shared=1, d_shared=2560))
    try:
        build_rank_ops(replace(m, layers=(shared,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("shared expert accepted")
    except ValueError as exc:
        assert str(exc) == "routed silu product is not swiglu"
    plain = get_model("qwen3-30b-a3b")
    skipped = _apply_unbound_product_rule(plain, _tc(load_release(plain.hf_id)["config"]))
    assert skipped is plain


def test_qwen3_235b_bills_routed_product_by_existing_rule():
    """0.484: Qwen3-235B-A22B silu(gate)*up uses routed_plain_silu_product.

    vec_kind stays empty. Both rows stay in the front 60 and are not moved.
    The product is the routed expert width. Block norms and q/k RMSNorm stay
    on the historical path. No new vec_kind. glm-4.5 / glm-4.6 stay unbound.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import _apply_unbound_product_rule, _tc, from_release, load_release

    width = 94 * 8 * 1536
    assert width == 1_155_072
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-235b-a22b"] == vr.RULE_ROUTED_PLAIN
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-235b-a22b-fp8"] == vr.RULE_ROUTED_PLAIN
    ids = [e["id"] for e in entries()]
    assert ids.index("qwen3-235b-a22b") == 21
    assert ids.index("qwen3-235b-a22b-fp8") == 42
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "qwen3_235b" not in vr.FROZEN_PRODUCT_KINDS
    assert "glm5_swiglu" in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    for mid, af, ew, rw in (
            ("qwen3-235b-a22b", "bf16", "bf16", "bf16"),
            ("qwen3-235b-a22b-fp8", "fp8", "fp8", "fp32")):
        m = get_model(mid)
        assert m.vec_kind == ""
        assert all(L.vec_kind == "" and L.product_rule == vr.RULE_ROUTED_PLAIN for L in m.layers)
        assert m.hidden == 4096 and m.n_layers == 94 and not m.mtp_layers
        assert m.layers[0].ffn.d_expert == 1536 and m.layers[0].ffn.top_k == 8
        assert m.layers[0].ffn.n_experts == 128 and m.layers[0].ffn.n_shared == 0
        assert m.layers[0].attn_linears[0].n == 64 * 128
        assert m.layers[0].attn_linears[0].n != m.hidden
        assert m.param_check()["rel_err"] == 0.0
        ops = build_rank_ops(m, 0, ph, Shard())
        names = _ffn_names(ops)
        assert names == ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
        by = {o.name: o for o in ops}
        assert by["mlp_mul"].vec == float(4096 * 8 * 1536) and by["mlp_mul"].sfu == 0.0
        assert by["mlp_mul"].a_fmt == af and by["mlp_mul"].flops == 0.0
        assert by["expert_act"].a_fmt == "bf16"
        assert by["expert_act"].vec == 4096 * 8 * 1536 * 4
        assert by["expert_act"].sfu == float(4096 * 8 * 1536 * 2)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert "q_rms" not in by and "k_rms" not in by
        assert by["expert.gate_up"].k == 4096 and by["expert.gate_up"].n == 2 * 1536
        assert by["expert.down"].k == 1536 and by["expert.down"].n == 4096
        assert by["expert.gate_up"].w_fmt == ew and by["moe.router"].w_fmt == rw
        bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert _gemm_sig(ops) == _gemm_sig(bare_ops)
        assert _ffn_names(bare_ops) == ("moe.router", "expert.gate_up", "expert.down", "expert_act")
        got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
                  for li in range(m.n_layers))
        assert got == width
        head = {o.name: o for o in head_ops(m, ph, Shard())}
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    m = get_model("qwen3-235b-a22b")
    split = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    mul = next(o for o in split if o.name == "mlp_mul")
    assert mul.vec * 2 == 4096 * 8 * 1536
    assert next(o for o in split if o.name == "expert.down").k == 1536
    narrow = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 2 == 4096 * 8 * 1536
    assert next(o for o in narrow if o.name == "expert.down").k == 768
    try:
        build_rank_ops(m, 0, ph, Shard(tp=2))
        raise AssertionError("tp without ep accepted")
    except ValueError as exc:
        assert "ep" in str(exc) and "tp" in str(exc)
    fp8 = replace(m, act_fmt="fp8")
    got = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got["mlp_mul"].a_fmt == "fp8" and got["expert_act"].a_fmt == "bf16"
    assert got["expert.gate_up"].a_fmt == "fp8" and got["expert.gate_up"].w_fmt == "bf16"
    assert got["attn_norm"].a_fmt == "bf16" and got["ffn_norm"].a_fmt == "bf16"
    for mid in ("glm-4.5", "glm-4.6", "qwen3-30b-a3b", "qwen3-vl-30b-a3b",
                "qwen3-coder-next-fp8"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("qwen3-coder-480b-a35b").layers[0].product_rule == vr.RULE_ROUTED_PLAIN
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "qwen3 235b hidden_act is not silu"
    shared = replace(m.layers[0], ffn=replace(m.layers[0].ffn, n_shared=1, d_shared=1536))
    try:
        build_rank_ops(replace(m, layers=(shared,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("shared expert accepted")
    except ValueError as exc:
        assert str(exc) == "routed silu product is not swiglu"
    smaller = get_model("qwen3-30b-a3b")
    skipped = _apply_unbound_product_rule(smaller, _tc(load_release(smaller.hf_id)["config"]))
    assert skipped is smaller


def test_qwen3_vl_235b_bills_routed_product_by_existing_rule():
    """0.485: Qwen3-VL-235B-A22B silu(gate)*up uses routed_plain_silu_product.

    vec_kind stays empty. Both rows stay outside the front 60 and are not
    moved. The product is the text-expert width. Block norms, q/k RMSNorm,
    mRoPE, and the vision tower stay on the historical path. No new vec_kind.
    glm-4.5 / glm-4.6 stay unbound. The smaller VL-30B rows stay unbound.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import _apply_unbound_product_rule, _qwen3_vl_235b_ignore, _tc, from_release, load_release

    width = 94 * 8 * 1536
    assert width == 1_155_072
    assert len(_qwen3_vl_235b_ignore()) == 339
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-vl-235b-a22b"] == vr.RULE_ROUTED_PLAIN
    assert vr.UNBOUND_PRODUCT_RULES["qwen3-vl-235b-a22b-fp8"] == vr.RULE_ROUTED_PLAIN
    ids = [e["id"] for e in entries()]
    assert ids.index("qwen3-vl-235b-a22b") == 331
    assert ids.index("qwen3-vl-235b-a22b-fp8") == 332
    assert "qwen3-vl-235b-a22b" not in ids[:60]
    assert "qwen3-vl-235b-a22b-fp8" not in ids[:60]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "qwen3_vl_235b" not in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    for mid, af, ew in (
            ("qwen3-vl-235b-a22b", "bf16", "bf16"),
            ("qwen3-vl-235b-a22b-fp8", "fp8", "fp8")):
        m = get_model(mid)
        assert m.vec_kind == ""
        assert all(L.vec_kind == "" and L.product_rule == vr.RULE_ROUTED_PLAIN for L in m.layers)
        assert m.hidden == 4096 and m.n_layers == 94 and not m.mtp_layers
        assert m.layers[0].ffn.d_expert == 1536 and m.layers[0].ffn.top_k == 8
        assert m.layers[0].ffn.n_experts == 128 and m.layers[0].ffn.n_shared == 0
        assert m.layers[0].attn_linears[0].n == 64 * 128
        assert m.layers[0].attn_linears[0].n != m.hidden
        assert m.param_check()["rel_err"] == 0.0
        ops = build_rank_ops(m, 0, ph, Shard())
        names = _ffn_names(ops)
        assert names == ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
        by = {o.name: o for o in ops}
        assert by["mlp_mul"].vec == float(4096 * 8 * 1536) and by["mlp_mul"].sfu == 0.0
        assert by["mlp_mul"].a_fmt == af and by["mlp_mul"].flops == 0.0
        assert by["expert_act"].a_fmt == "bf16"
        assert by["expert_act"].vec == 4096 * 8 * 1536 * 4
        assert by["expert_act"].sfu == float(4096 * 8 * 1536 * 2)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert "q_rms" not in by and "k_rms" not in by
        assert by["expert.gate_up"].k == 4096 and by["expert.gate_up"].n == 2 * 1536
        assert by["expert.down"].k == 1536 and by["expert.down"].n == 4096
        assert by["expert.gate_up"].w_fmt == ew and by["moe.router"].w_fmt == "bf16"
        bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert _gemm_sig(ops) == _gemm_sig(bare_ops)
        assert _ffn_names(bare_ops) == ("moe.router", "expert.gate_up", "expert.down", "expert_act")
        assert by["rope"].vec == {o.name: o for o in bare_ops}["rope"].vec
        got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
                  for li in range(m.n_layers))
        assert got == width
        head = {o.name: o for o in head_ops(m, ph, Shard())}
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    m = get_model("qwen3-vl-235b-a22b")
    split = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    mul = next(o for o in split if o.name == "mlp_mul")
    assert mul.vec * 2 == 4096 * 8 * 1536
    assert next(o for o in split if o.name == "expert.down").k == 1536
    narrow = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 2 == 4096 * 8 * 1536
    assert next(o for o in narrow if o.name == "expert.down").k == 768
    try:
        build_rank_ops(m, 0, ph, Shard(tp=2))
        raise AssertionError("tp without ep accepted")
    except ValueError as exc:
        assert "ep" in str(exc) and "tp" in str(exc)
    fp8 = replace(m, act_fmt="fp8")
    got = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got["mlp_mul"].a_fmt == "fp8" and got["expert_act"].a_fmt == "bf16"
    assert got["expert.gate_up"].a_fmt == "fp8" and got["expert.gate_up"].w_fmt == "bf16"
    assert got["attn_norm"].a_fmt == "bf16" and got["ffn_norm"].a_fmt == "bf16"
    for mid in ("glm-4.5", "glm-4.6", "qwen3-30b-a3b", "qwen3-vl-30b-a3b",
                "qwen3-vl-30b-a3b-fp8", "qwen3-coder-next-fp8"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("qwen3-235b-a22b").layers[0].product_rule == vr.RULE_ROUTED_PLAIN
    assert get_model("qwen3-coder-480b-a35b").layers[0].product_rule == vr.RULE_ROUTED_PLAIN
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "qwen3 vl 235b hidden_act is not silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "rope_scaling": None})
        raise AssertionError("null rope accepted")
    except ValueError as exc:
        assert str(exc) == "qwen3 vl 235b rope_scaling is not mrope"
    vis = dict(cfg["vision_config"])
    vis["hidden_act"] = "silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "vision_config": vis})
        raise AssertionError("vision silu accepted")
    except ValueError as exc:
        # vision_spec runs before the product checker and already refuses
        # a tower whose hidden_act is not gelu_pytorch_tanh.
        assert str(exc) == "qwen3vl hidden_act is not gelu_pytorch_tanh"
    shared = replace(m.layers[0], ffn=replace(m.layers[0].ffn, n_shared=1, d_shared=1536))
    try:
        build_rank_ops(replace(m, layers=(shared,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("shared expert accepted")
    except ValueError as exc:
        assert str(exc) == "routed silu product is not swiglu"
    smaller = get_model("qwen3-vl-30b-a3b")
    skipped = _apply_unbound_product_rule(smaller, _tc(load_release(smaller.hf_id)["config"]))
    assert skipped is smaller
    fp8_m = get_model("qwen3-vl-235b-a22b-fp8")
    fp8_rel = load_release(fp8_m.hf_id)
    fp8_cfg = _tc(fp8_rel["config"])
    bad_q = dict(fp8_cfg["quantization_config"])
    bad_q["ignored_layers"] = [n for n in bad_q["ignored_layers"] if n != "lm_head"]
    try:
        from_release(fp8_m.id, fp8_m.hf_id, fp8_rel, {**fp8_cfg, "quantization_config": bad_q})
        raise AssertionError("trimmed ignore list accepted")
    except ValueError as exc:
        assert str(exc) == "qwen3 vl 235b quantization_config is not fp8"


def test_qwen38_27b_bills_dense_product_by_existing_rule():
    """0.486: Qwen3.8-27B silu(gate)*up uses dense_silu_product.

    vec_kind stays empty. The bf16 row stays in the front 60 and is not
    moved. The FP8 row stays outside. The product is the dense MLP width.
    MTP, block norms, q/k RMSNorm, mRoPE, and the vision tower stay on the
    historical path. No new vec_kind. glm-4.5 / glm-4.6 stay unbound.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import (
        _apply_unbound_product_rule, _qwen38_27b_ignore, _tc, from_release, load_release)

    width = 64 * 17408
    assert width == 1_114_112
    assert len(_qwen38_27b_ignore()) == 882
    assert vr.UNBOUND_PRODUCT_RULES["qwen3.8-27b"] == vr.RULE_DENSE
    assert vr.UNBOUND_PRODUCT_RULES["qwen3.8-27b-fp8"] == vr.RULE_DENSE
    ids = [e["id"] for e in entries()]
    assert ids.index("qwen3.8-27b") == 7
    assert ids.index("qwen3.8-27b-fp8") == 325
    assert "qwen3.8-27b" in ids[:60]
    assert "qwen3.8-27b-fp8" not in ids[:60]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "qwen38_27b" not in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    for mid, af, mw in (
            ("qwen3.8-27b", "bf16", "bf16"),
            ("qwen3.8-27b-fp8", "fp8", "fp8")):
        m = get_model(mid)
        assert m.vec_kind == ""
        assert all(L.vec_kind == "" and L.product_rule == vr.RULE_DENSE for L in m.layers)
        assert m.mtp_layers[0].product_rule == "" and m.mtp_layers[0].vec_kind == ""
        assert m.hidden == 5120 and m.n_layers == 64 and len(m.mtp_layers) == 1
        assert m.layers[0].ffn.d_ff == 17408 and m.layers[0].ffn.kind == "dense"
        assert m.layers[0].core.kind == "linear" and m.layers[3].core.kind == "gqa"
        assert m.layers[3].attn_linears[0].n == 24 * 256 * 2
        assert m.layers[3].attn_linears[0].n != m.hidden
        assert m.param_check()["rel_err"] == 0.0
        ops = build_rank_ops(m, 0, ph, Shard())
        names = _ffn_names(ops)
        assert names == ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
        by = {o.name: o for o in ops}
        assert by["mlp_mul"].vec == float(4096 * 17408) and by["mlp_mul"].sfu == 0.0
        assert by["mlp_mul"].a_fmt == af and by["mlp_mul"].flops == 0.0
        assert by["act"].a_fmt == af
        assert by["act"].vec == 4096 * 17408 * 4
        assert by["act"].sfu == float(4096 * 17408 * 2)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert "q_rms" not in by and "k_rms" not in by and "gelu" not in by
        assert by["mlp.gate_up"].k == 5120 and by["mlp.gate_up"].n == 2 * 17408
        assert by["mlp.down"].k == 17408 and by["mlp.down"].n == 5120
        assert by["mlp.gate_up"].w_fmt == mw and by["mlp.down"].w_fmt == mw
        full = build_rank_ops(m, 3, ph, Shard())
        assert _ffn_names(full) == ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
        assert {o.name: o for o in full}["mlp_mul"].vec == float(4096 * 17408)
        bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert _gemm_sig(ops) == _gemm_sig(bare_ops)
        assert _ffn_names(bare_ops) == ("mlp.gate_up", "mlp.down", "act")
        full_by = {o.name: o for o in full}
        bare_full = {o.name: o for o in build_rank_ops(bare, 3, ph, Shard())}
        assert full_by["rope"].vec == bare_full["rope"].vec
        got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
                  for li in range(m.n_layers))
        assert got == width
        mtp_names = _ffn_names(build_rank_ops(m, 0, one, Shard(), layer=m.mtp_layers[0]))
        assert "mlp_mul" not in mtp_names
        head = {o.name: o for o in head_ops(m, ph, Shard())}
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    m = get_model("qwen3.8-27b")
    split = build_rank_ops(m, 0, ph, Shard(tp=2))
    assert next(o for o in split if o.name == "mlp_mul").vec * 2 == 4096 * 17408
    assert next(o for o in split if o.name == "mlp.down").k == 17408 // 2
    narrow = build_rank_ops(m, 3, ph, Shard(tp=4))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 4 == 4096 * 17408
    assert next(o for o in narrow if o.name == "mlp.down").k == 17408 // 4
    fp8 = replace(m, act_fmt="fp8")
    got = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got["act"].a_fmt == "fp8" and got["mlp_mul"].a_fmt == "fp8"
    assert got["mlp.gate_up"].a_fmt == "fp8" and got["mlp.gate_up"].w_fmt == "bf16"
    assert got["attn_norm"].a_fmt == "bf16" and got["ffn_norm"].a_fmt == "bf16"
    published = get_model("qwen3.8-27b-fp8")
    pw = next(o for o in build_rank_ops(published, 0, ph, Shard()) if o.name == "mlp.gate_up")
    assert pw.w_fmt == "fp8" and round(pw.w_bits, 6) == 8.000977
    for mid in ("glm-4.5", "glm-4.6", "qwen3-30b-a3b", "qwen3-vl-30b-a3b",
                "qwen3.8-27b-ready", "qwen3.5-9b", "qwen3-coder-next-fp8"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("magistral-small").layers[0].product_rule == vr.RULE_DENSE
    assert get_model("qwen3-vl-235b-a22b").layers[0].product_rule == vr.RULE_ROUTED_PLAIN
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "qwen38 27b hidden_act is not silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "rope_parameters": None})
        raise AssertionError("null rope accepted")
    except ValueError as exc:
        assert str(exc) == "qwen38 27b rope_parameters is not mrope"
    vis = dict(cfg["vision_config"])
    vis["hidden_act"] = "silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "vision_config": vis})
        raise AssertionError("vision silu accepted")
    except ValueError as exc:
        assert str(exc) == "qwen35 vision hidden_act is not gelu_pytorch_tanh"
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    try:
        build_rank_ops(replace(m, layers=(ungated,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("ungated accepted")
    except ValueError as exc:
        assert str(exc) == "dense silu product is not gate_up then down"
    ready = get_model("qwen3.8-27b-ready")
    skipped = _apply_unbound_product_rule(ready, _tc(load_release(ready.hf_id)["config"]))
    assert skipped is ready
    fp8_m = get_model("qwen3.8-27b-fp8")
    fp8_rel = load_release(fp8_m.hf_id)
    fp8_cfg = _tc(fp8_rel["config"])
    bad_q = dict(fp8_cfg["quantization_config"])
    bad_q["modules_to_not_convert"] = [n for n in bad_q["modules_to_not_convert"] if n != "lm_head"]
    try:
        from_release(fp8_m.id, fp8_m.hf_id, fp8_rel, {**fp8_cfg, "quantization_config": bad_q})
        raise AssertionError("trimmed ignore list accepted")
    except ValueError as exc:
        assert str(exc) == "qwen38 27b quantization_config is not fp8"


def test_internlm2_20b_bills_dense_product_by_existing_rule():
    """0.487: InternLM2.5-20B silu(gate)*up uses dense_silu_product.

    vec_kind stays empty. The row is already in the front 60 and is not
    moved. The product is the dense MLP width. Block RMSNorm and dynamic
    NTK RoPE stay on the historical path. No new vec_kind. glm-4.5 /
    glm-4.6 and the Qwen3.8 same-width clones stay unbound.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import (
        _apply_unbound_product_rule, _tc, from_release, load_release)

    width = 48 * 16384
    assert width == 786_432
    assert vr.UNBOUND_PRODUCT_RULES["internlm2-5-20b"] == vr.RULE_DENSE
    ids = [e["id"] for e in entries()]
    assert ids.index("internlm2-5-20b") == 35
    assert "internlm2-5-20b" in ids[:60]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "internlm2_20b" not in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    m = get_model("internlm2-5-20b")
    assert m.vec_kind == ""
    assert all(L.vec_kind == "" and L.product_rule == vr.RULE_DENSE for L in m.layers)
    assert m.mtp_layers == ()
    assert m.hidden == 6144 and m.n_layers == 48
    assert m.layers[0].ffn.d_ff == 16384 and m.layers[0].ffn.kind == "dense"
    assert m.layers[0].core.kind == "gqa"
    assert m.layers[0].attn_linears[0].n == m.hidden
    assert m.param_check()["rel_err"] == 0.0
    ops = build_rank_ops(m, 0, ph, Shard())
    names = _ffn_names(ops)
    assert names == ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
    by = {o.name: o for o in ops}
    assert by["mlp_mul"].vec == float(4096 * 16384) and by["mlp_mul"].sfu == 0.0
    assert by["mlp_mul"].a_fmt == "bf16" and by["mlp_mul"].flops == 0.0
    assert by["act"].a_fmt == "bf16"
    assert by["act"].vec == 4096 * 16384 * 4
    assert by["act"].sfu == float(4096 * 16384 * 2)
    assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
    assert by["attn_norm"].sfu == 4096 and by["ffn_norm"].sfu == 4096
    assert "q_rms" not in by and "k_rms" not in by and "gelu" not in by
    assert by["mlp.gate_up"].k == 6144 and by["mlp.gate_up"].n == 2 * 16384
    assert by["mlp.down"].k == 16384 and by["mlp.down"].n == 6144
    assert by["mlp.gate_up"].w_fmt == "bf16" and by["mlp.down"].w_fmt == "bf16"
    bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    assert _gemm_sig(ops) == _gemm_sig(bare_ops)
    assert _ffn_names(bare_ops) == ("mlp.gate_up", "mlp.down", "act")
    assert by["rope"].vec == {o.name: o for o in bare_ops}["rope"].vec
    got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
              for li in range(m.n_layers))
    assert got == width
    flops = 0.0
    bare_flops = 0.0
    for li in range(m.n_layers):
        flops += sum(o.flops for o in build_rank_ops(m, li, ph, Shard())
                     if o.kind == "gemm" and o.name.startswith("mlp."))
        bare_flops += sum(o.flops for o in build_rank_ops(bare, li, ph, Shard())
                          if o.kind == "gemm" and o.name.startswith("mlp."))
    assert flops == bare_flops == 118.747255799808 * 1e12
    head = {o.name: o for o in head_ops(m, ph, Shard())}
    bare_head = {o.name: o for o in head_ops(bare, ph, Shard())}
    assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    assert head["final_norm"].vec == bare_head["final_norm"].vec == 4096 * 6144 * 4
    split = build_rank_ops(m, 0, ph, Shard(tp=2))
    assert next(o for o in split if o.name == "mlp_mul").vec * 2 == 4096 * 16384
    assert next(o for o in split if o.name == "mlp.down").k == 16384 // 2
    narrow = build_rank_ops(m, 0, ph, Shard(tp=4))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 4 == 4096 * 16384
    assert next(o for o in narrow if o.name == "mlp.down").k == 16384 // 4
    eighth = build_rank_ops(m, 0, ph, Shard(tp=8))
    assert next(o for o in eighth if o.name == "mlp_mul").vec * 8 == 4096 * 16384
    assert next(o for o in eighth if o.name == "mlp.down").k == 16384 // 8
    fp8 = replace(m, act_fmt="fp8")
    got_fp8 = {o.name: o for o in build_rank_ops(fp8, 0, ph, Shard())}
    assert got_fp8["act"].a_fmt == "fp8" and got_fp8["mlp_mul"].a_fmt == "fp8"
    assert got_fp8["mlp.gate_up"].a_fmt == "fp8" and got_fp8["mlp.gate_up"].w_fmt == "bf16"
    assert got_fp8["attn_norm"].a_fmt == "bf16" and got_fp8["ffn_norm"].a_fmt == "bf16"
    for mid in ("glm-4.5", "glm-4.6", "qwen3-30b-a3b", "qwen3-vl-30b-a3b",
                "qwen3.8-27b-ready", "startlux-decision-v1-27b", "pplx-decider-v1-27b",
                "phi-4", "internlm3-8b", "qwen3.5-9b"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("qwen3.8-27b").layers[0].product_rule == vr.RULE_DENSE
    assert get_model("magistral-small").layers[0].product_rule == vr.RULE_DENSE
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "internlm2 20b hidden_act is not silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "rope_scaling": None})
        raise AssertionError("null rope accepted")
    except ValueError as exc:
        assert str(exc) == "internlm2 20b rope_scaling is not dynamic"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "bias": True})
        raise AssertionError("bias accepted")
    except ValueError as exc:
        assert str(exc) == "internlm2 20b bias is not false"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "pretraining_tp": 2})
        raise AssertionError("pretraining_tp accepted")
    except ValueError as exc:
        assert str(exc) == "internlm2 20b pretraining_tp is not 1"
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    try:
        build_rank_ops(replace(m, layers=(ungated,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("ungated accepted")
    except ValueError as exc:
        assert str(exc) == "dense silu product is not gate_up then down"
    phi = get_model("phi-4")
    skipped = _apply_unbound_product_rule(phi, _tc(load_release(phi.hf_id)["config"]))
    assert skipped is phi
    ready = get_model("qwen3.8-27b-ready")
    skipped_ready = _apply_unbound_product_rule(ready, _tc(load_release(ready.hf_id)["config"]))
    assert skipped_ready is ready


def test_minimax_m2_bills_routed_product_by_existing_rule():
    """0.488: MiniMax-M2 silu(gate)*up uses routed_plain_silu_product.

    vec_kind stays empty. The four rows stay outside the front 60 and are
    not moved. The product is the routed expert width. Block RMSNorm,
    full-width q/k RMSNorm, and partial RoPE stay on the historical path.
    No new vec_kind. glm-4.5 / glm-4.6 stay unbound. phi-4 is smaller.
    """
    from accel_dse.core.catalog import entries
    from accel_dse.core.model import (
        _MINIMAX_M2_IGNORE, _apply_unbound_product_rule, _bill_minimax_m2_product,
        _tc, from_release, load_release)

    width = 62 * 8 * 1536
    assert width == 761_856
    assert _MINIMAX_M2_IGNORE == ["gate", "e_score_correction_bias", "lm_head"]
    for mid in ("minimax-m2", "minimax-m2.1", "minimax-m2.5", "minimax-m2.7"):
        assert vr.UNBOUND_PRODUCT_RULES[mid] == vr.RULE_ROUTED_PLAIN
    ids = [e["id"] for e in entries()]
    assert ids.index("minimax-m2.7") == 93
    assert ids.index("minimax-m2") == 141
    assert ids.index("minimax-m2.1") == 142
    assert ids.index("minimax-m2.5") == 143
    assert "minimax-m2" not in ids[:60]
    assert "minimax-m2.7" not in ids[:60]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("internlm2-5-20b") == 35
    assert "minimax_m2" not in vr.FROZEN_PRODUCT_KINDS
    ph = Phase("prefill", 1, 4096, 0)
    one = Phase("prefill", 1, 1, 0)
    ctx = {"minimax-m2": 196608, "minimax-m2.1": 196608,
           "minimax-m2.5": 196608, "minimax-m2.7": 204800}
    for mid in ctx:
        m = get_model(mid)
        assert m.vec_kind == ""
        assert all(L.vec_kind == "" and L.product_rule == vr.RULE_ROUTED_PLAIN for L in m.layers)
        assert m.mtp_layers == ()
        assert m.hidden == 3072 and m.n_layers == 62 and m.max_ctx == ctx[mid]
        assert m.layers[0].ffn.kind == "moe" and m.layers[0].ffn.d_expert == 1536
        assert m.layers[0].ffn.top_k == 8 and m.layers[0].ffn.n_experts == 256
        assert m.layers[0].ffn.n_shared == 0 and m.layers[0].ffn.d_shared == 0
        assert m.layers[0].core.kind == "gqa"
        assert m.layers[0].attn_linears[0].n == 48 * 128
        assert m.layers[0].attn_linears[0].n != m.hidden
        assert m.layers[0].misc_params == 2 * 3072 + (48 + 8) * 128 + 256
        assert m.act_fmt == "fp8" and m.quantized_release
        assert m.fmt("router").fmt == "fp32" and m.fmt("expert").fmt == "fp8"
        assert m.param_check()["rel_err"] == 0.0
        assert m.params() == 228_689_764_864
        assert any(n.startswith("0.488:") for n in m.notes)
        ops = build_rank_ops(m, 0, ph, Shard())
        names = _ffn_names(ops)
        assert names == ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
        by = {o.name: o for o in ops}
        assert by["mlp_mul"].vec == float(4096 * 8 * 1536) and by["mlp_mul"].sfu == 0.0
        assert by["mlp_mul"].a_fmt == "fp8" and by["mlp_mul"].flops == 0.0
        assert by["expert_act"].a_fmt == "bf16"
        assert by["expert_act"].vec == 4096 * 8 * 1536 * 4
        assert by["expert_act"].sfu == float(4096 * 8 * 1536 * 2)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert by["attn_norm"].sfu == 4096 and by["ffn_norm"].sfu == 4096
        assert "q_rms" not in by and "k_rms" not in by and "gelu" not in by
        assert by["expert.gate_up"].k == 3072 and by["expert.gate_up"].n == 2 * 1536
        assert by["expert.down"].k == 1536 and by["expert.down"].n == 3072
        assert by["expert.gate_up"].w_fmt == "fp8" and by["moe.router"].w_fmt == "fp32"
        assert round(by["expert.gate_up"].w_bits, 6) == 8.001953
        bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert _gemm_sig(ops) == _gemm_sig(bare_ops)
        assert _ffn_names(bare_ops) == ("moe.router", "expert.gate_up", "expert.down", "expert_act")
        assert by["rope"].vec == {o.name: o for o in bare_ops}["rope"].vec
        got = sum(next(o for o in build_rank_ops(m, li, one, Shard()) if o.name == "mlp_mul").vec
                  for li in range(m.n_layers))
        assert got == width
        head = {o.name: o for o in head_ops(m, ph, Shard())}
        bare_head = {o.name: o for o in head_ops(bare, ph, Shard())}
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
        assert head["final_norm"].vec == bare_head["final_norm"].vec == 4096 * 3072 * 4
    m = get_model("minimax-m2")
    flops = 0.0
    bare = replace(m, layers=tuple(replace(L, product_rule="") for L in m.layers))
    bare_flops = 0.0
    for li in range(m.n_layers):
        flops += sum(o.flops for o in build_rank_ops(m, li, ph, Shard())
                     if o.kind == "gemm" and o.name.startswith("expert."))
        bare_flops += sum(o.flops for o in build_rank_ops(bare, li, ph, Shard())
                          if o.kind == "gemm" and o.name.startswith("expert."))
    assert flops == bare_flops == 57.518202028032 * 1e12
    try:
        build_rank_ops(m, 0, ph, Shard(tp=2))
        raise AssertionError("tp without ep accepted")
    except ValueError as exc:
        assert "ep" in str(exc) and "tp" in str(exc)
    split = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    assert next(o for o in split if o.name == "mlp_mul").vec * 2 == 4096 * 8 * 1536
    assert next(o for o in split if o.name == "expert.down").k == 1536
    narrow = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    assert next(o for o in narrow if o.name == "mlp_mul").vec * 2 == 4096 * 8 * 1536
    assert next(o for o in narrow if o.name == "expert.down").k == 768
    eighth = build_rank_ops(m, 0, ph, Shard(tp=8, ep=8))
    assert next(o for o in eighth if o.name == "mlp_mul").vec * 8 == 4096 * 8 * 1536
    assert next(o for o in eighth if o.name == "expert.down").k == 1536
    bf16 = replace(m, act_fmt="bf16")
    got_bf16 = {o.name: o for o in build_rank_ops(bf16, 0, ph, Shard())}
    assert got_bf16["mlp_mul"].a_fmt == "bf16" and got_bf16["expert_act"].a_fmt == "bf16"
    assert got_bf16["expert.gate_up"].a_fmt == "bf16" and got_bf16["expert.gate_up"].w_fmt == "fp8"
    assert got_bf16["attn_norm"].a_fmt == "bf16" and got_bf16["ffn_norm"].a_fmt == "bf16"
    for mid in ("glm-4.5", "glm-4.6", "qwen3-30b-a3b", "qwen3-vl-30b-a3b",
                "qwen3.8-27b-ready", "startlux-decision-v1-27b", "pplx-decider-v1-27b",
                "phi-4", "internlm3-8b", "qwen3.5-9b", "minimax-m3", "mimo-v2.6-pro"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert other.vec_kind != "minimax_swiglu"
        assert other.layers[0].product_rule == ""
        assert "mlp_mul" not in _ffn_names(build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard()))
    assert get_model("internlm2-5-20b").layers[0].product_rule == vr.RULE_DENSE
    assert get_model("qwen3-235b-a22b").layers[0].product_rule == vr.RULE_ROUTED_PLAIN
    assert get_model("minimax-text-01").vec_kind == "minimax_swiglu"
    assert get_model("minimax-text-01").layers[0].product_rule == ""
    rel = load_release(m.hf_id)
    cfg = _tc(rel["config"])
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "hidden_act": "gelu"})
        raise AssertionError("gelu accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 hidden_act is not silu"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "shared_intermediate_size": 9216})
        raise AssertionError("shared expert accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 shared expert is set"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "mlp_intermediate_size": 4096})
        raise AssertionError("mlp width accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 mlp_intermediate_size is not the release"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "max_position_embeddings": 204800})
        raise AssertionError("m2.7 context accepted on m2")
    except ValueError as exc:
        assert str(exc) == "minimax m2 max_position_embeddings is not the row"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "attn_type_list": [0] + [1] * 61})
        raise AssertionError("lightning layer accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 attn_type_list is not full attention"
    try:
        from_release(m.id, m.hf_id, rel, {**cfg, "swiglu_limit": 7.0})
        raise AssertionError("swiglu limit accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 swiglu_limit is set"
    shared = replace(m.layers[0], ffn=replace(m.layers[0].ffn, n_shared=1, d_shared=1536))
    try:
        build_rank_ops(replace(m, layers=(shared,) + m.layers[1:]), 0, ph, Shard())
        raise AssertionError("shared expert accepted")
    except ValueError as exc:
        assert str(exc) == "routed silu product is not swiglu"
    tagged = replace(m, vec_kind="minimax_swiglu")
    try:
        _bill_minimax_m2_product(tagged, cfg)
        raise AssertionError("tagged kind accepted")
    except ValueError as exc:
        assert str(exc) == "minimax m2 vec_kind is already set"
    phi = get_model("phi-4")
    skipped = _apply_unbound_product_rule(phi, _tc(load_release(phi.hf_id)["config"]))
    assert skipped is phi
    ready = get_model("qwen3.8-27b-ready")
    skipped_ready = _apply_unbound_product_rule(ready, _tc(load_release(ready.hf_id)["config"]))
    assert skipped_ready is ready


def test_phase_b_predicate_leaves_foreign_id_untagged():
    from accel_dse.core.model import _apply_one_predicate, _tc, load_release

    spec = get_model("qwen3-8b")
    cfg = _tc(load_release(spec.hf_id)["config"])
    for name in vr.LEGACY_TAGGERS:
        out = _apply_one_predicate(spec, cfg, name)
        assert out is spec
    one = next(iter(vr.allow_row("seed_swiglu").model_ids))
    tagged = get_model(one)
    assert tagged.vec_kind == "seed_swiglu"
    assert tagged.layers[0].vec_kind == "seed_swiglu"

