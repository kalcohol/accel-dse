"""0.476: ERNIE-4.5-VL 424B bills the published text SwiGLU product.

Ernie4_5_MLP and Ernie4_5_MoeMLP are both down(F.silu(gate) * up).
Dense SiLU stays act. Expert SiLU stays expert_act or expert_b.act.
The product is mlp_mul. a_fmt follows act_fmt. The class does not
upcast. The vision pool runs only for multimodal tokens. RMSNorm and
3-D RoPE stay on the historical ops. vec_kind is ernievl_swiglu. No
earlier allowlist is extended. The two rows sit outside the first 60.
The 28B rows stay untagged. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _ERNIEVL_SWIGLU, _ERNIE_MOE, _JAIS3_SWIGLU, _KIMI_SITU, _LLAMA3,
    _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _OLMO3_SWIGLU, _QWEN35_MOE, _SEED_SWIGLU,
    _tag_ernie_moe_vectors, _tag_ernievl_swiglu_vectors, _tag_jais3_swiglu_vectors,
    _tag_kimi_situ_vectors, _tag_llama3_vectors, _tag_mixtral_swiglu_vectors,
    _tag_muse_swiglu_vectors, _tag_olmo3_swiglu_vectors, _tag_qwen35_moe_vectors,
    _tag_seed_swiglu_vectors, _tc, from_release, load_release,
)


_ROWS = {
    "ernie-4.5-vl-424b-a47b-pt": "baidu/ERNIE-4.5-VL-424B-A47B-PT",
    "ernie-4.5-vl-424b-a47b": "baidu/ERNIE-4.5-VL-424B-A47B",
}
_H = 8192
_INNER = 28672
_DEPTH = 54
_DENSE = 3
_HEADS = 64
_KV = 8
_HD = 128
_EXP = 64
_TOPK = 8
_DE = 3584
_DEB = 1536
_PARAMS = 422_722_779_520
_NORM = 892_928
_MISC_D = 2 * _H
_MISC_M = 2 * _H + 2 * _EXP
_DENSE_TEXT = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_MOE_TEXT = ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "q_rms", "k_rms", "shared_gate_up")
_FFN_TFLOP = "311.711546474496"
_ROPE = {"mrope_section": [22, 22, 20], "type": "default"}


def _names(ops):
    return [o.name for o in ops]


def _act(op, n: float, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "act"
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _eact(op, n: float, layer: int, name: str = "expert_act") -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == name
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


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


def _bad(hf, edit, mid="ernie-4.5-vl-424b-a47b-pt"):
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
    return lambda: from_release(mid, hf, rel)


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


def test_version_is_476():
    assert __version__ == "0.479.0"


def test_two_rows_bill_the_text_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(mid) for mid in _ROWS] == [163, 167]
    assert _ERNIEVL_SWIGLU == frozenset(_ROWS)
    assert _ERNIEVL_SWIGLU.isdisjoint(_JAIS3_SWIGLU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_OLMO3_SWIGLU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_SEED_SWIGLU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _ERNIEVL_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_LLAMA3)
    assert _ERNIEVL_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _ERNIEVL_SWIGLU.isdisjoint(_ERNIE_MOE)
    assert _DENSE * _INNER + (_DEPTH - _DENSE) * _TOPK * _DE == 1_548_288
    assert _HEADS * _HD == _H
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.vec_kind == "ernievl_swiglu"
        assert m.params() == _PARAMS and m.release_params == _PARAMS
        assert m.param_check()["rel_err"] == 0.0
        assert m.mtp_layers == () and m.tie_embeddings is False
        assert m.kv_fmt == "bf16" and m.act_fmt == "bf16" and not m.quantized_release
        assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.fmt("expert").fmt == "bf16" and m.fmt("router").fmt == "fp32"
        assert all(L.vec_kind == "ernievl_swiglu" for L in m.layers)
        for i, L in enumerate(m.layers):
            assert L.core.window is None and not L.core.no_rope and L.core.rope_dim == 0
            assert not L.core.out_gate_fn
            if i < _DENSE:
                assert L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER
                assert L.misc_params == _MISC_D
            else:
                assert L.ffn.kind == "moe" and L.ffn.gated and L.ffn.top_k == _TOPK
                assert L.ffn.n_experts == _EXP and L.ffn.d_expert == _DE
                assert L.ffn.n_experts_b == _EXP and L.ffn.d_expert_b == _DEB
                assert L.ffn.n_shared == 0 and L.misc_params == _MISC_M
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "ernie4_5_moe_vl"
        assert cfg["architectures"] == ["Ernie4_5_VLMoeForConditionalGeneration"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["use_bias"] is False and cfg["use_rmsnorm"] is True
        assert cfg["moe_gate"] == "topk" and cfg["moe_use_aux_free"] is True
        assert "moe_num_shared_experts" not in cfg
        for key in ("attention_probs_dropout_prob", "hidden_dropout_prob",
                    "moe_dropout_prob", "moe_all_to_all_dropout"):
            assert key not in cfg
        assert "num_nextn_predict_layers" not in cfg and "head_dim" not in cfg
        assert cfg["moe_layer_interval"] == 1
        assert cfg["moe_layer_start_index"] == [_DENSE, _DENSE]
        assert cfg["moe_layer_end_index"] == [_DEPTH - 1, _DEPTH - 1]
        assert cfg["moe_num_experts"] == [_EXP, _EXP]
        assert cfg["moe_intermediate_size"] == [_DE, _DEB]
        assert cfg["moe_k"] == _TOPK
        assert cfg["moe_capacity"] == [64, 64, 64]
        assert cfg["moe_multimodal_dispatch_use_allgather"] == "v2-alltoall-unpad-text"
        assert cfg["hidden_size"] == _H and cfg["intermediate_size"] == _INNER
        assert cfg["num_hidden_layers"] == _DEPTH
        assert cfg["num_attention_heads"] == _HEADS and cfg["num_key_value_heads"] == _KV
        assert cfg["rope_scaling"] == _ROPE
        assert cfg["rope_3d"] is True and cfg["freq_allocation"] == 20
        assert cfg["torch_dtype"] == "bfloat16"
        assert cfg["tie_word_embeddings"] is False and cfg["rms_norm_eps"] == 1e-5
        assert cfg.get("quantization_config") is None
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        roles = load_release(hf)["roles"]["norm"]["bf16"]
        assert roles["params"] == _NORM
        assert any(n.startswith("0.476") for n in m.notes)
        assert _tag_ernievl_swiglu_vectors(_bare(m), cfg).vec_kind == "ernievl_swiglu"
    for mid in ("ernie-4.5-vl-28b-a3b-pt", "ernie-4.5-vl-28b-a3b",
                "ernie-4.5-vl-28b-a3b-base", "ernie-4.5-vl-28b-a3b-thinking",
                "jais-3-7b", "jais-3-3.7b", "jais-3-0.9b", "drama-3"):
        assert get_model(mid).vec_kind == ""
    for mid in ("glm-5", "glm-5.2", "glm-5.3", "atria-dawn", "atria-dawn-fp8"):
        assert get_model(mid).vec_kind == "glm5_swiglu"
    assert get_model("jais-3-32b").vec_kind == "jais3_swiglu"
    assert get_model("olmo-3-32b-think").vec_kind == "olmo3_swiglu"
    assert get_model("seed-oss-36b").vec_kind == "seed_swiglu"
    assert get_model("mixtral-8x22b").vec_kind == "mixtral_swiglu"
    assert get_model("ernie-4.5-300b-a47b-pt").vec_kind == "ernie300_swiglu"
    assert get_model("minimax-text-01").vec_kind == "minimax_swiglu"
    assert get_model("minimax-m1-80k").vec_kind == "minimax_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model("ernie-4.5-vl-424b-a47b-pt")
    other = get_model("ernie-4.5-vl-424b-a47b")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    ops = build_rank_ops(m, 0, ph, Shard())
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    _mul(next(o for o in ops if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    _act(next(o for o in ops if o.name == "act"), t * _INNER, 0, "bf16")
    names = _names(ops)
    assert [n for n in names if n in _DENSE_TEXT or n in _BANNED] == list(_DENSE_TEXT)
    assert names.index("mlp.gate_up") < names.index("act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("mlp.down")
    assert "expert_act" not in names and "quick_gelu" not in names
    up = next(o for o in ops if o.name == "mlp.gate_up")
    down = next(o for o in ops if o.name == "mlp.down")
    assert (up.k, up.n) == (_H, 2 * _INNER)
    assert (down.k, down.n) == (_INNER, _H)
    assert up.w_fmt == "bf16" and down.w_fmt == "bf16" and up.w_bits == 16.0
    assert up.a_fmt == "bf16" and down.a_fmt == "bf16"
    q = next(o for o in ops if o.name == "attn.qkv")
    assert q.n == _HEADS * _HD + 2 * _KV * _HD
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    for name in ("attn_norm", "ffn_norm", "rope", "residual"):
        got = next(o for o in ops if o.name == name)
        old = next(o for o in bare_ops if o.name == name)
        assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
    assert next(o for o in ops if o.name == "attn_norm").a_fmt == "bf16"
    assert next(o for o in ops if o.name == "attn_norm").sfu == float(t)
    assert next(o for o in ops if o.name == "rope").vec == t * _HEADS * (_HD // 2) * 3
    moe = build_rank_ops(m, _DENSE, ph, Shard())
    moe_names = _names(moe)
    assert [n for n in moe_names if n in _MOE_TEXT or n in _BANNED] == list(_MOE_TEXT)
    assert "router_b" not in "".join(moe_names) and "expert_b" not in "".join(moe_names)
    _eact(next(o for o in moe if o.name == "expert_act"), t * _TOPK * _DE, _DENSE)
    _mul(next(o for o in moe if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "bf16")
    assert moe_names.index("expert.gate_up") < moe_names.index("expert_act")
    assert moe_names.index("expert_act") < moe_names.index("mlp_mul") < moe_names.index("expert.down")
    eup = next(o for o in moe if o.name == "expert.gate_up")
    edown = next(o for o in moe if o.name == "expert.down")
    assert (eup.k, eup.n) == (_H, 2 * _DE)
    assert (edown.k, edown.n) == (_DE, _H)
    assert eup.w_fmt == "bf16" and edown.w_fmt == "bf16"
    assert next(o for o in moe if o.name == "moe.router").w_fmt == "fp32"
    assert all(o.flops == 0 for o in moe if o.kind == "vector")
    flops = 0.0
    bare_flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        if li < _DENSE:
            _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
            want = ("mlp.gate_up", "mlp.down")
        else:
            _mul(next(o for o in layer if o.name == "mlp_mul"), t * _TOPK * _DE, li, "bf16")
            want = ("expert.gate_up", "expert.down")
        for o in layer:
            if o.name in want:
                flops += o.flops
        for o in build_rank_ops(bare, li, ph, Shard()):
            if o.name in want:
                bare_flops += o.flops
    assert flops == bare_flops
    assert f"{flops / 1e12:.12f}" == _FFN_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    o0 = build_rank_ops(other, 0, ph, Shard())
    o3 = build_rank_ops(other, _DENSE, ph, Shard())
    _mul(next(o for o in o0 if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    _mul(next(o for o in o3 if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "bf16")
    mm = Phase("prefill", 1, t, mm=256)
    mixed = build_rank_ops(m, _DENSE, mm, Shard())
    muls = [o for o in mixed if o.name == "mlp_mul"]
    assert len(muls) == 2
    _mul(muls[0], (t - 256) * _TOPK * _DE, _DENSE, "bf16")
    _mul(muls[1], 256 * _TOPK * _DEB, _DENSE, "bf16")
    _eact(next(o for o in mixed if o.name == "expert_b.act"), 256 * _TOPK * _DEB,
          _DENSE, "expert_b.act")
    mixed_names = _names(mixed)
    assert mixed_names.index("moe.router") < mixed_names.index("moe.router_b")
    assert mixed_names.index("expert.down") < mixed_names.index("expert_b.gate_up")
    assert mixed_names.index("expert_b.act") < mixed_names.index("expert_b.down")


def test_tp_request_fp8_and_cleared_kind():
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.parallel import Layout
    from accel_dse.core.scenario import Scenario, Serving

    m = get_model("ernie-4.5-vl-424b-a47b-pt")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    dense = build_rank_ops(m, 0, ph, Shard(tp=2))
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * (_INNER // 2), 0, "bf16")
    assert next(o for o in dense if o.name == "mlp.down").k == _INNER // 2
    _raises(lambda: build_rank_ops(m, _DENSE, ph, Shard(tp=2)), "ep·etp")
    _raises(lambda: evaluate(Scenario(
        model=m.id, layout=Layout(tp=2),
        serving=Serving(phase="prefill", batch=1, prompt=t))), "PP1·TP2 invalid for MoE")
    ep = build_rank_ops(m, _DENSE, ph, Shard(tp=2, ep=2))
    _mul(next(o for o in ep if o.name == "mlp_mul"), (t * _TOPK / 2) * _DE, _DENSE, "bf16")
    assert next(o for o in ep if o.name == "expert.down").k == _DE
    etp = build_rank_ops(m, _DENSE, ph, Shard(tp=2, etp=2))
    _mul(next(o for o in etp if o.name == "mlp_mul"), t * _TOPK * (_DE // 2), _DENSE, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == _DE // 2
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    _act(next(o for o in got if o.name == "act"), t * _INNER, 0, "fp8")
    gem = next(o for o in got if o.name == "mlp.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    moe = build_rank_ops(fp8, _DENSE, ph, Shard())
    _mul(next(o for o in moe if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "fp8")
    _eact(next(o for o in moe if o.name == "expert_act"), t * _TOPK * _DE, _DENSE)
    egem = next(o for o in moe if o.name == "expert.gate_up")
    assert egem.a_fmt == "fp8" and egem.w_fmt == "bf16"
    bare = _bare(m)
    b0 = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in b0
    assert b0.index("mlp.gate_up") < b0.index("mlp.down") < b0.index("act")
    b3 = _names(build_rank_ops(bare, _DENSE, ph, Shard()))
    assert "mlp_mul" not in b3
    assert b3.index("expert.gate_up") < b3.index("expert.down") < b3.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("jais-3-7b", "jais-3-375b-a23b", "olmo-3-7b-think", "seed-oss-36b",
                "minimax-text-01", "minimax-m1-80k", "glm-5.3", "atria-dawn",
                "qwen3-235b-a22b", "drama-3", "ernie-4.5-vl-28b-a3b-pt"):
        other = get_model(mid)
        assert other.vec_kind != "ernievl_swiglu"
        assert all(L.vec_kind != "ernievl_swiglu" for L in other.layers)
        if other.layers[0].ffn.kind == "dense" and other.vec_kind == "":
            names = _names(build_rank_ops(other, 0, ph, Shard()))
            assert "mlp_mul" not in names
    ernie = get_model("ernie-4.5-300b-a47b-pt")
    li = next(i for i, L in enumerate(ernie.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(ernie, li, ph, Shard()))
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("expert.down")
    assert ernie.layers[li].vec_kind == "ernie300_swiglu"
    wide = get_model("ernie-4.5-vl-424b-a47b-pt")
    c_ok = _tc(load_release(_ROWS["ernie-4.5-vl-424b-a47b-pt"])["config"])
    for mid in ("jais-3-32b", "ernie-4.5-300b-a47b-pt", "minimax-text-01",
                "ernie-4.5-vl-28b-a3b-pt"):
        spoof = replace(_bare(get_model(mid)), id="ernie-4.5-vl-424b-a47b-pt")
        _raises(lambda spoof=spoof: _tag_ernievl_swiglu_vectors(spoof, c_ok),
                "ernievl swiglu")
    kept = replace(_bare(wide), id="ernie-4.5-vl-28b-a3b-pt")
    assert _tag_ernievl_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_jais3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_olmo3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_ernie_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["ernie-4.5-vl-424b-a47b-pt"]
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "ernievl swiglu hidden_act is not silu")
    wrong_type = dict(c_ok)
    wrong_type["model_type"] = "ernie4_5_moe"
    _raises(lambda: _tag_ernievl_swiglu_vectors(_bare(wide), wrong_type),
            "ernievl swiglu model_type is not ernie4_5_moe_vl")
    _raises(_bad(hf, lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "ernievl swiglu swiglu_limit is set")
    _raises(_bad(hf, lambda c: c.__setitem__("architectures", ["Ernie4_5_MoeForCausalLM"])),
            "ernievl swiglu architecture is not ernie4_5_vl")
    biased = dict(c_ok)
    biased["use_bias"] = True
    _raises(lambda: _tag_ernievl_swiglu_vectors(_bare(wide), biased),
            "ernievl swiglu use_bias is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("moe_num_shared_experts", 0)),
            "ernievl swiglu shared expert is set")
    _raises(_bad(hf, lambda c: c.__setitem__("moe_dropout_prob", 0.0)),
            "ernievl swiglu dropout is set")
    _raises(_bad(hf, lambda c: c.__setitem__("rope_scaling", {"type": "default"})),
            "ernievl swiglu rope_scaling is not the published mrope")
    _raises(_bad(hf, lambda c: c.__setitem__("rope_3d", False)),
            "ernievl swiglu rope_3d is not the published split")
    _raises(_bad(hf, lambda c: c.__setitem__("torch_dtype", "float16")),
            "ernievl swiglu dtype is not bfloat16")
    _raises(_bad(hf, lambda c: c.__setitem__("tie_word_embeddings", True)),
            "ernievl swiglu tie_word_embeddings is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("rms_norm_eps", 1e-6)),
            "ernievl swiglu rms_norm_eps is not 1e-5")
    _raises(_bad(hf, lambda c: c.__setitem__("moe_k", 6)),
            "ernievl swiglu moe_k is not 8")
    _raises(_bad(hf, lambda c: c.__setitem__("moe_intermediate_size", [1536, 1536])),
            "ernievl swiglu expert width is not the 424b release")
    _raises(_bad(hf, lambda c: c.__setitem__("intermediate_size", 16384)),
            "ernievl swiglu width is not the 424b release")
    _raises(_bad(hf, lambda c: c.__setitem__("head_dim", 128)),
            "ernievl swiglu head_dim is set")
    _raises(_bad(hf, lambda c: c.__setitem__("quantization_config", {"quant_method": "fp8"})),
            "ernievl swiglu quantization_config is set")
    mtp = dict(c_ok)
    mtp["num_nextn_predict_layers"] = 1
    _raises(lambda: _tag_ernievl_swiglu_vectors(_bare(wide), mtp),
            "ernievl swiglu mtp is set")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_ernievl_swiglu_vectors(short, c_ok),
            "ernievl swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_ernievl_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "ernievl swiglu mlp is not dense swiglu")
    dropped = replace(wide.layers[0], ffn_linears=wide.layers[0].ffn_linears[:1])
    _raises(lambda: _tag_ernievl_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "ernievl swiglu mlp is not gate_up then down")
    runtime = replace(
        wide.layers[0],
        vec_kind="ernievl_swiglu",
        ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
        "ernievl swiglu mlp is not gate_up then down")
    moe_rt = replace(
        wide.layers[_DENSE],
        vec_kind="ernievl_swiglu",
        ffn=replace(wide.layers[_DENSE].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=wide.layers[:_DENSE] + (moe_rt,) + wide.layers[_DENSE + 1:]),
        _DENSE, ph, Shard()),
        "ernievl swiglu expert is not swiglu")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    # These two rows are outside the first 60. Stage and pipe stay.
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
