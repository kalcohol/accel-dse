"""0.479: GLM-5, GLM-5.2, GLM-5.3, and Atria-Dawn bill the published SwiGLU product.

GlmMoeDsaMLP and GlmMoeDsaExperts are act_fn(gate) * up, then down. hidden_act
is silu. Dense SiLU stays act. Routed SiLU stays expert_act. The product is
mlp_mul on the dense width, on the useful token-expert pairs, and on the one
shared expert. a_fmt follows act_fmt. Shared SiLU stays unbilled. The MTP
block stays untagged. vec_kind is glm5_swiglu. GLM-5.3 and Atria-Dawn-FP8
keep the released dynamic fp8 activation. GLM-5.3-Flash stays untagged.
Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _ERNIE300_SWIGLU, _ERNIEVL_SWIGLU, _GLM5_SWIGLU, _GLM5_SWIGLU_BF16,
    _GLM5_SWIGLU_FP8, _JAIS3_SWIGLU, _KIMI_SITU, _MINIMAX_SWIGLU, _MIXTRAL_SWIGLU,
    _MUSE_SWIGLU, _OLMO3_SWIGLU, _QWEN35_MOE, _SEED_SWIGLU,
    _tag_ernie300_swiglu_vectors, _tag_ernievl_swiglu_vectors, _tag_glm5_swiglu_vectors,
    _tag_jais3_swiglu_vectors, _tag_kimi_situ_vectors, _tag_minimax_swiglu_vectors,
    _tag_mixtral_swiglu_vectors, _tag_muse_swiglu_vectors, _tag_olmo3_swiglu_vectors,
    _tag_qwen35_moe_vectors, _tag_seed_swiglu_vectors, _tc, from_release, load_release,
)


_ROWS = {
    "glm-5": "zai-org/GLM-5",
    "glm-5.2": "zai-org/GLM-5.2",
    "glm-5.3": "zai-org/GLM-5.3",
    "atria-dawn": "internlm/Atria-Dawn",
    "atria-dawn-fp8": "internlm/Atria-Dawn-FP8",
}
_BF16 = ("glm-5", "glm-5.2", "atria-dawn")
_FP8 = ("glm-5.3", "atria-dawn-fp8")
_PARAMS = {
    "glm-5": 743_911_218_432,
    "glm-5.2": 743_377_019_904,
    "glm-5.3": 743_377_019_904,
    "atria-dawn": 743_377_019_904,
    "atria-dawn-fp8": 743_377_019_904,
}
_H = 6144
_INNER = 12288
_DE = 2048
_DEPTH = 78
_DENSE = 3
_MOE = 75
_EXP = 256
_TOPK = 8
_HEADS = 64
_QK = 256
_PRODUCT = 1_419_264
_DENSE_NAMES = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_MOE_NAMES = ("moe.router", "moe.shared_gate_up", "mlp_mul", "moe.shared_down",
              "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ")
_GEMM_TFLOP = "214.301688201216"
_GEMM = ("mlp.gate_up", "mlp.down", "moe.shared_gate_up", "moe.shared_down",
         "expert.gate_up", "expert.down")


def _names(ops):
    return [o.name for o in ops]


def _mul(op, n: float, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "mlp_mul"
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(hf, edit):
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
    return lambda: from_release("glm-5", hf, rel)


def _bare(model):
    return replace(
        model,
        layers=tuple(replace(L, vec_kind="") for L in model.layers),
        vec_kind="")


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.count,
                            o.w_fmt, o.a_fmt, o.share, o.replicated))
    return out


def _ffn(ops):
    return [o for o in ops if o.name in _GEMM or o.name in ("moe.router",)]


def test_version_is_479():
    assert __version__ == "0.480.0"


def test_five_rows_bill_the_shared_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("glm-5.3") == 0
    assert ids.index("glm-5") == 14
    assert ids.index("glm-5.2") == 15
    assert ids.index("atria-dawn") == 170
    assert ids.index("atria-dawn-fp8") == 171
    assert _GLM5_SWIGLU == frozenset(_ROWS)
    assert _GLM5_SWIGLU_BF16 == frozenset(_BF16)
    assert _GLM5_SWIGLU_FP8 == frozenset(_FP8)
    for prior in (_QWEN35_MOE, _KIMI_SITU, _MUSE_SWIGLU, _MIXTRAL_SWIGLU,
                  _SEED_SWIGLU, _OLMO3_SWIGLU, _JAIS3_SWIGLU, _ERNIEVL_SWIGLU,
                  _MINIMAX_SWIGLU, _ERNIE300_SWIGLU):
        assert _GLM5_SWIGLU.isdisjoint(prior)
    assert _DENSE * _INNER + _MOE * (_TOPK + 1) * _DE == _PRODUCT
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.kv_fmt == "bf16" and m.vec_kind == "glm5_swiglu"
        assert m.params() == m.release_params == _PARAMS[mid]
        assert m.param_check()["rel_err"] == 0.0
        assert len(m.mtp_layers) == 1 and m.mtp_layers[0].vec_kind == ""
        assert m.tie_embeddings is False
        assert all(L.vec_kind == "glm5_swiglu" for L in m.layers)
        assert all(L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER
                   for L in m.layers[:_DENSE])
        assert all(L.ffn.kind == "moe" and L.ffn.gated and L.ffn.n_shared == 1
                   and L.ffn.d_shared == _DE and L.ffn.n_experts == _EXP
                   and L.ffn.top_k == _TOPK and L.ffn.d_expert == _DE
                   and L.ffn.latent == 0 and not L.ffn.groups
                   for L in m.layers[_DENSE:])
        assert m.fmt("router").fmt == "bf16"
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "glm_moe_dsa"
        assert cfg["architectures"] == ["GlmMoeDsaForCausalLM"]
        assert cfg["hidden_act"] == "silu" and cfg["n_shared_experts"] == 1
        assert cfg["n_routed_experts"] == _EXP and cfg["num_experts_per_tok"] == _TOPK
        assert cfg["moe_intermediate_size"] == _DE and cfg["intermediate_size"] == _INNER
        assert cfg["first_k_dense_replace"] == _DENSE and cfg["rms_norm_eps"] == 1e-5
        assert cfg["routed_scaling_factor"] == 2.5 and cfg["scoring_func"] == "sigmoid"
        assert cfg["topk_method"] == "noaux_tc" and cfg["norm_topk_prob"] is True
        assert cfg.get("rope_scaling") is None and cfg["tie_word_embeddings"] is False
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        assert any(n.startswith("0.479") for n in m.notes)
        assert _tag_glm5_swiglu_vectors(_bare(m), cfg).vec_kind == "glm5_swiglu"
        if mid in _BF16:
            assert m.act_fmt == "bf16" and not m.quantized_release
            assert m.fmt("mlp").fmt == "bf16" and m.fmt("expert").fmt == "bf16"
            assert m.fmt("shared_expert").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
            assert cfg.get("quantization_config") is None
        else:
            assert m.act_fmt == "fp8" and m.quantized_release
            assert m.fmt("mlp").fmt == "fp8" and m.fmt("expert").fmt == "fp8"
            assert m.fmt("shared_expert").fmt == "fp8" and m.fmt("attn").fmt == "fp8"
            assert m.fmt("embed").fmt == "bf16" and m.fmt("lm_head").fmt == "bf16"
            qcfg = cfg["quantization_config"]
            assert qcfg["quant_method"] == "fp8" and qcfg["activation_scheme"] == "dynamic"
    assert sum(1 for L in get_model("glm-5").layers if L.core.idx_heads) == _DEPTH
    assert sum(1 for L in get_model("glm-5.2").layers if L.core.idx_heads) == 21
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("minimax-text-01").vec_kind == "minimax_swiglu"
    assert get_model("ernie-4.5-300b-a47b-pt").vec_kind == "ernie300_swiglu"
    assert get_model("glm-5.3-flash").vec_kind != "glm5_swiglu"
    assert get_model("glm-4.5").vec_kind != "glm5_swiglu"
    assert get_model("glm-4.6").vec_kind != "glm5_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model("glm-5")
    twin = get_model("glm-5.2")
    atria = get_model("atria-dawn")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    dense = build_rank_ops(m, 0, ph, Shard())
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    act = next(o for o in dense if o.name == "act")
    assert act.vec == t * _INNER * 4 and act.sfu == float(t * _INNER * 2) and act.a_fmt == "bf16"
    assert [n for n in _names(dense) if n in _DENSE_NAMES or n in _BANNED] == list(_DENSE_NAMES)
    up = next(o for o in dense if o.name == "mlp.gate_up")
    down = next(o for o in dense if o.name == "mlp.down")
    assert (up.k, up.n) == (_H, 2 * _INNER) and (down.k, down.n) == (_INNER, _H)
    assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    ops = build_rank_ops(m, _DENSE, ph, Shard())
    bare_ops = build_rank_ops(bare, _DENSE, ph, Shard())
    muls = [o for o in ops if o.name == "mlp_mul"]
    assert len(muls) == 2
    _mul(muls[0], t * _DE, _DENSE, "bf16")
    _mul(muls[1], t * _TOPK * _DE, _DENSE, "bf16")
    act = next(o for o in ops if o.name == "expert_act")
    assert act.vec == t * _TOPK * _DE * 4 and act.sfu == float(t * _TOPK * _DE * 2)
    assert act.a_fmt == "bf16" and act.flops == 0
    names = _names(ops)
    assert [n for n in names if n in _MOE_NAMES or n in _BANNED] == list(_MOE_NAMES)
    assert names.index("moe.shared_gate_up") < names.index("mlp_mul") < names.index("moe.shared_down")
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("expert.down")
    assert names.index("expert_act") < names.index("expert.down")
    sup = next(o for o in ops if o.name == "moe.shared_gate_up")
    sdn = next(o for o in ops if o.name == "moe.shared_down")
    eup = next(o for o in ops if o.name == "expert.gate_up")
    edn = next(o for o in ops if o.name == "expert.down")
    assert (sup.k, sup.n) == (_H, 2 * _DE) and (sdn.k, sdn.n) == (_DE, _H)
    assert (eup.k, eup.n) == (_H, 2 * _DE) and (edn.k, edn.n) == (_DE, _H)
    assert eup.w_fmt == "bf16" and eup.w_bits == 16.0 and sup.w_fmt == "bf16"
    assert next(o for o in ops if o.name == "moe.router").w_fmt == "bf16"
    assert next(o for o in ops if o.name == "rope").vec == t * _HEADS * (_QK // 2) * 3
    assert next(o for o in ops if o.name == "rope").vec == next(
        o for o in bare_ops if o.name == "rope").vec
    for name in ("attn_norm", "ffn_norm"):
        got = next(o for o in ops if o.name == name)
        old = next(o for o in bare_ops if o.name == name)
        assert (got.vec, got.sfu, got.a_fmt) == (t * _H * 4, float(t), "bf16")
        assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    flops = 0.0
    bare_flops = 0.0
    width = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        if li < _DENSE:
            _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
            width += t * _INNER
        else:
            got = [o for o in layer if o.name == "mlp_mul"]
            _mul(got[0], t * _DE, li, "bf16")
            _mul(got[1], t * _TOPK * _DE, li, "bf16")
            width += t * (_TOPK + 1) * _DE
        for o in layer:
            if o.name in _GEMM:
                flops += o.flops
        for o in build_rank_ops(bare, li, ph, Shard()):
            if o.name in _GEMM:
                bare_flops += o.flops
    assert width == t * _PRODUCT
    assert flops == bare_flops
    assert f"{flops / 1e12:.12f}" == _GEMM_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    old_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (old_head.vec, old_head.sfu, old_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0 and head.vec == t * _H * 4
    assert [(o.name, o.k, o.n, o.w_fmt, o.flops) for o in _ffn(
        build_rank_ops(twin, _DENSE, ph, Shard()))] == [(o.name, o.k, o.n, o.w_fmt, o.flops)
        for o in _ffn(ops)]
    assert _sched(twin, ph, Shard()) == _sched(atria, ph, Shard())
    fp = get_model("glm-5.3")
    fp_ops = build_rank_ops(fp, _DENSE, ph, Shard())
    fp_muls = [o for o in fp_ops if o.name == "mlp_mul"]
    _mul(fp_muls[0], t * _DE, _DENSE, "fp8")
    _mul(fp_muls[1], t * _TOPK * _DE, _DENSE, "fp8")
    assert next(o for o in fp_ops if o.name == "expert_act").a_fmt == "bf16"
    assert next(o for o in fp_ops if o.name == "expert.gate_up").w_fmt == "fp8"
    assert next(o for o in fp_ops if o.name == "moe.shared_gate_up").w_fmt == "fp8"
    assert next(o for o in build_rank_ops(fp, 0, ph, Shard()) if o.name == "act").a_fmt == "fp8"
    assert _sched(fp, ph, Shard()) == _sched(get_model("atria-dawn-fp8"), ph, Shard())
    side = build_rank_ops(m, 0, ph, Shard(), layer=m.mtp_layers[0])
    assert "mlp_mul" not in _names(side)
    assert _names(side).index("expert.down") < _names(side).index("expert_act")


def test_tp_request_fp8_and_cleared_kind():
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.parallel import Layout
    from accel_dse.core.scenario import Scenario, Serving

    m = get_model("glm-5")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    _raises(lambda: build_rank_ops(m, _DENSE, ph, Shard(tp=2)), "ep·etp")
    _raises(lambda: evaluate(Scenario(
        model=m.id, layout=Layout(tp=2),
        serving=Serving(phase="prefill", batch=1, prompt=t))), "PP1·TP2 invalid for MoE")
    ep = build_rank_ops(m, _DENSE, ph, Shard(tp=2, ep=2))
    ep_muls = [o for o in ep if o.name == "mlp_mul"]
    _mul(ep_muls[0], t * (_DE // 2), _DENSE, "bf16")
    _mul(ep_muls[1], (t * _TOPK / 2) * _DE, _DENSE, "bf16")
    assert next(o for o in ep if o.name == "expert.down").k == _DE
    etp = build_rank_ops(m, _DENSE, ph, Shard(tp=2, etp=2))
    etp_muls = [o for o in etp if o.name == "mlp_mul"]
    _mul(etp_muls[0], t * (_DE // 2), _DENSE, "bf16")
    _mul(etp_muls[1], t * _TOPK * (_DE // 2), _DENSE, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == _DE // 2
    dense = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * (_INNER // 2), 0, "bf16")
    for tp in (4, 8):
        got = build_rank_ops(m, _DENSE, ph, Shard(tp=tp, ep=tp))
        got_muls = [o for o in got if o.name == "mlp_mul"]
        _mul(got_muls[0], t * (_DE // tp), _DENSE, "bf16")
        _mul(got_muls[1], (t * _TOPK / tp) * _DE, _DENSE, "bf16")
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, _DENSE, ph, Shard())
    got_muls = [o for o in got if o.name == "mlp_mul"]
    _mul(got_muls[0], t * _DE, _DENSE, "fp8")
    _mul(got_muls[1], t * _TOPK * _DE, _DENSE, "fp8")
    gem = next(o for o in got if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    assert next(o for o in got if o.name == "moe.shared_gate_up").w_fmt == "bf16"
    assert next(o for o in got if o.name == "expert_act").a_fmt == "bf16"
    dense = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    assert next(o for o in dense if o.name == "act").a_fmt == "fp8"
    assert next(o for o in dense if o.name == "attn_norm").a_fmt == "bf16"
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, _DENSE, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("moe.shared_gate_up") < bnames.index("moe.shared_down")
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("glm-5.3-flash", "glm-4.5", "glm-4.6", "drama-3", "qwen3-235b-a22b",
                "ernie-4.5-21b-a3b-pt", "ernie-4.5-vl-28b-a3b-pt", "jais-3-7b",
                "deepseek-v3", "kimi-k2"):
        other = get_model(mid)
        assert other.vec_kind != "glm5_swiglu"
        assert all(L.vec_kind != "glm5_swiglu" for L in other.layers)
    flash = get_model("glm-5.3-flash")
    names = _names(build_rank_ops(flash, 3, ph, Shard()))
    assert "mlp_mul" not in names
    assert names.index("expert.gate_up") < names.index("expert.down") < names.index("expert_act")
    wide = get_model("glm-5")
    c_ok = _tc(load_release(_ROWS["glm-5"])["config"])
    for mid in ("jais-3-32b", "ernie-4.5-300b-a47b-pt", "minimax-text-01",
                "glm-5.3-flash", "glm-4.5"):
        spoof = replace(_bare(get_model(mid)), id="glm-5")
        _raises(lambda spoof=spoof: _tag_glm5_swiglu_vectors(spoof, c_ok), "glm5 swiglu")
    kept = replace(_bare(wide), id="glm-5.3-flash")
    assert _tag_glm5_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_minimax_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_ernie300_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_ernievl_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_jais3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_olmo3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["glm-5"]

    def _kind(edit):
        cfg = json.loads(json.dumps(c_ok))
        edit(cfg)
        return lambda: _tag_glm5_swiglu_vectors(_bare(wide), cfg)

    _raises(_kind(lambda c: c.__setitem__("hidden_act", "gelu")),
            "glm5 swiglu hidden_act is not silu")
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "glm5 swiglu hidden_act is not silu")
    _raises(_kind(lambda c: c.__setitem__("model_type", "glm4_moe")),
            "glm5 swiglu model_type is not glm_moe_dsa")
    _raises(_kind(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "glm5 swiglu swiglu_limit is set")
    _raises(_kind(lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "glm5 swiglu architecture is not glm_moe_dsa")
    assert _kind(lambda c: c.__setitem__("n_shared_experts", 0))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("moe_intermediate_size", 1536))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("num_experts_per_tok", 4))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("n_routed_experts", 128))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("head_dim", 192))().vec_kind == "glm5_swiglu"
    _raises(_kind(lambda c: c.__setitem__("head_dim", 128)),
            "glm5 swiglu head_dim is not the release")
    _raises(_kind(lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "glm5 swiglu rope_scaling is set")
    _raises(_kind(lambda c: c.__setitem__("tie_word_embeddings", True)),
            "glm5 swiglu tie_word_embeddings is not false")
    _raises(_kind(lambda c: c.__setitem__("rms_norm_eps", 1e-6)),
            "glm5 swiglu rms_norm_eps is not 1e-5")
    _raises(_kind(lambda c: c.__setitem__("routed_scaling_factor", 1.0)),
            "glm5 swiglu routed_scaling_factor is not 2.5")
    _raises(_kind(lambda c: c.__setitem__("quantization_config", {"quant_method": "fp8"})),
            "glm5 swiglu quantization_config is set")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_glm5_swiglu_vectors(short, c_ok),
            "glm5 swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[_DENSE], ffn=replace(wide.layers[_DENSE].ffn, n_shared=0))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=wide.layers[:_DENSE] + (ungated,) + wide.layers[_DENSE + 1:]),
        _DENSE, ph, Shard()), "glm5 swiglu shared expert is not swiglu")
    dense_bad = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(dense_bad,) + wide.layers[1:]), 0, ph, Shard()),
        "glm5 swiglu mlp is not gate_up then down")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi

    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    test_fingerprints_change_only_v4()
    from accel_dse.core import evaluate as E
    pins = {
        "qwen3-32b": "b975b7b3760e",
        "qwen3-8b": "11f6f3a5d21f",
        "llama-3.3-70b": "8e50c6cabdc8",
        "mixtral-8x22b": "0d8711691e1e",
        "seed-oss-36b": "f547c49c5769",
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
