"""0.477: MiniMax-Text-01 and MiniMax-M1 bill the published expert SwiGLU product.

MiniMaxText01BlockSparseTop2MLP and MiniMaxM1BlockSparseTop2MLP are
w2(silu(w1(x)) * w3(x)). SiLU stays expert_act. The product is mlp_mul
on the useful token-expert pairs. a_fmt follows act_fmt. There is no
shared expert. vec_kind is minimax_swiglu. No earlier allowlist is
extended. The rows are catalog indexes 27 and 28. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _ERNIEVL_SWIGLU, _JAIS3_SWIGLU, _KIMI_SITU, _LLAMA3, _MINIMAX_SWIGLU,
    _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _OLMO3_SWIGLU, _QWEN35_MOE, _SEED_SWIGLU,
    _tag_ernievl_swiglu_vectors, _tag_jais3_swiglu_vectors, _tag_kimi_situ_vectors,
    _tag_llama3_vectors, _tag_minimax_swiglu_vectors, _tag_mixtral_swiglu_vectors,
    _tag_muse_swiglu_vectors, _tag_olmo3_swiglu_vectors, _tag_qwen35_moe_vectors,
    _tag_seed_swiglu_vectors, _tc, from_release, load_release,
)


_ROWS = {
    "minimax-text-01": "MiniMaxAI/MiniMax-Text-01",
    "minimax-m1-80k": "MiniMaxAI/MiniMax-M1-80k",
}
_EMBED = {"minimax-text-01": "fp32", "minimax-m1-80k": "bf16"}
_ARCH = {
    "minimax-text-01": ("minimax_text_01", ["MiniMaxText01ForCausalLM"]),
    "minimax-m1-80k": ("minimax_m1", ["MiniMaxM1ForCausalLM"]),
}
_H = 6144
_DE = 9216
_DEPTH = 80
_EXP = 32
_TOPK = 2
_HEADS = 64
_KV = 8
_HD = 128
_FULL = tuple(i for i in range(_DEPTH) if i % 8 == 7)
_PARAMS = 456_089_657_856
_RELEASE = 456_089_655_296
_TEXT = ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "shared_gate_up", "act")
_FFN_TFLOP = "222.651104624640"


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
    return lambda: from_release("minimax-text-01", hf, rel)


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


def test_version_is_477():
    assert __version__ == "0.482.0"


def test_two_rows_bill_the_expert_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("minimax-text-01") == 27
    assert ids.index("minimax-m1-80k") == 28
    assert _MINIMAX_SWIGLU == frozenset(_ROWS)
    assert _MINIMAX_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _MINIMAX_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _MINIMAX_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_SEED_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_OLMO3_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_JAIS3_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_ERNIEVL_SWIGLU)
    assert _MINIMAX_SWIGLU.isdisjoint(_LLAMA3)
    assert _DEPTH * _TOPK * _DE == 1_474_560
    assert len(_FULL) == 10
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.act_fmt == "bf16" and m.kv_fmt == "bf16" and m.state_fmt == "fp32"
        assert m.vec_kind == "minimax_swiglu"
        assert m.params() == _PARAMS and m.release_params == _RELEASE
        assert m.param_check()["rel_err"] == (_PARAMS - _RELEASE) / _RELEASE
        assert m.mtp_layers == () and m.tie_embeddings is False
        assert not m.quantized_release
        assert all(L.vec_kind == "minimax_swiglu" for L in m.layers)
        assert all(L.ffn.kind == "moe" and L.ffn.gated and L.ffn.n_shared == 0
                   and L.ffn.d_shared == 0 and L.ffn.latent == 0 and L.ffn.d_ff == 0
                   for L in m.layers)
        assert all(L.ffn.n_experts == _EXP and L.ffn.top_k == _TOPK
                   and L.ffn.d_expert == _DE for L in m.layers)
        assert sum(1 for L in m.layers if L.core.kind == "linear") == 70
        assert [i for i, L in enumerate(m.layers) if L.core.kind == "gqa"] == list(_FULL)
        assert m.fmt("expert").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.fmt("router").fmt == "fp32" and m.fmt("embed").fmt == _EMBED[mid]
        assert m.fmt("lm_head").fmt == "bf16"
        cfg = _tc(load_release(hf)["config"])
        mt, arch = _ARCH[mid]
        assert cfg["model_type"] == mt and cfg["architectures"] == arch
        assert cfg["hidden_act"] == "silu"
        assert cfg["shared_intermediate_size"] == 0 and cfg["shared_moe_mode"] == "sigmoid"
        assert cfg["postnorm"] is True and cfg["sliding_window"] is None
        assert cfg["attention_dropout"] == 0.0 and cfg["router_jitter_noise"] == 0.0
        assert cfg["output_router_logits"] is False
        assert cfg["rotary_dim"] == _HD // 2 and cfg["head_dim"] == _HD
        assert cfg["rms_norm_eps"] == 1e-5 and cfg["tie_word_embeddings"] is False
        assert cfg.get("quantization_config") is None
        assert cfg["layernorm_mlp_alpha"] == 3.5565588200778455
        assert cfg["layernorm_mlp_beta"] == 1.0
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        assert any(n.startswith("0.477") for n in m.notes)
        assert _tag_minimax_swiglu_vectors(_bare(m), cfg).vec_kind == "minimax_swiglu"
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("mixtral-8x22b").vec_kind == "mixtral_swiglu"
    assert get_model("ernie-4.5-vl-424b-a47b").vec_kind == "ernievl_swiglu"
    assert get_model("minimax-m2").vec_kind != "minimax_swiglu"
    assert get_model("minimax-m3").vec_kind != "minimax_swiglu"
    assert get_model("glm-5.3").vec_kind != "minimax_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model("minimax-text-01")
    other = get_model("minimax-m1-80k")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    ops = build_rank_ops(m, 0, ph, Shard())
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    _mul(next(o for o in ops if o.name == "mlp_mul"), t * _TOPK * _DE, 0, "bf16")
    assert next(o for o in ops if o.name == "mlp_mul").vec == 4096 * 2 * 9216
    act = next(o for o in ops if o.name == "expert_act")
    assert act.vec == t * _TOPK * _DE * 4 and act.sfu == float(t * _TOPK * _DE * 2)
    assert act.a_fmt == "bf16" and act.flops == 0
    names = _names(ops)
    assert [n for n in names if n in _TEXT or n in _BANNED] == list(_TEXT)
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("expert.down")
    assert "rope" not in names
    up = next(o for o in ops if o.name == "expert.gate_up")
    down = next(o for o in ops if o.name == "expert.down")
    assert (up.k, up.n) == (_H, 2 * _DE)
    assert (down.k, down.n) == (_DE, _H)
    assert up.w_fmt == "bf16" and down.w_fmt == "bf16" and up.w_bits == 16.0
    assert up.a_fmt == "bf16" and down.a_fmt == "bf16"
    assert next(o for o in ops if o.name == "moe.router").w_fmt == "fp32"
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    for name in ("attn_norm", "ffn_norm", "residual"):
        got = next(o for o in ops if o.name == name)
        old = next(o for o in bare_ops if o.name == name)
        assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
    assert next(o for o in ops if o.name == "attn_norm").a_fmt == "bf16"
    assert next(o for o in ops if o.name == "attn_norm").sfu == float(t)
    full = build_rank_ops(m, _FULL[0], ph, Shard())
    full_bare = build_rank_ops(bare, _FULL[0], ph, Shard())
    _mul(next(o for o in full if o.name == "mlp_mul"), t * _TOPK * _DE, _FULL[0], "bf16")
    assert next(o for o in full if o.name == "rope").vec == t * _HEADS * (_HD // 2) * 3
    assert next(o for o in full if o.name == "rope").vec == next(
        o for o in full_bare if o.name == "rope").vec
    assert next(o for o in full if o.name == "attn.qkv").n == _HEADS * _HD + 2 * _KV * _HD
    flops = 0.0
    bare_flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        _mul(next(o for o in layer if o.name == "mlp_mul"), t * _TOPK * _DE, li, "bf16")
        for o in layer:
            if o.name in ("expert.gate_up", "expert.down"):
                flops += o.flops
        for o in build_rank_ops(bare, li, ph, Shard()):
            if o.name in ("expert.gate_up", "expert.down"):
                bare_flops += o.flops
    assert flops == bare_flops
    assert f"{flops / 1e12:.12f}" == _FFN_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    o0 = build_rank_ops(other, 0, ph, Shard())
    o7 = build_rank_ops(other, _FULL[0], ph, Shard())
    _mul(next(o for o in o0 if o.name == "mlp_mul"), t * _TOPK * _DE, 0, "bf16")
    _mul(next(o for o in o7 if o.name == "mlp_mul"), t * _TOPK * _DE, _FULL[0], "bf16")
    assert _sched(m, ph, Shard()) == _sched(other, ph, Shard())


def test_tp_request_fp8_and_cleared_kind():
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.parallel import Layout
    from accel_dse.core.scenario import Scenario, Serving

    m = get_model("minimax-text-01")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    _raises(lambda: build_rank_ops(m, 0, ph, Shard(tp=2)), "ep·etp")
    _raises(lambda: evaluate(Scenario(
        model=m.id, layout=Layout(tp=2),
        serving=Serving(phase="prefill", batch=1, prompt=t))), "PP1·TP2 invalid for MoE")
    ep = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    _mul(next(o for o in ep if o.name == "mlp_mul"), (t * _TOPK / 2) * _DE, 0, "bf16")
    assert next(o for o in ep if o.name == "expert.down").k == _DE
    etp = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    _mul(next(o for o in etp if o.name == "mlp_mul"), t * _TOPK * (_DE // 2), 0, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == _DE // 2
    for tp in (4, 8):
        got = build_rank_ops(m, 0, ph, Shard(tp=tp, ep=tp))
        _mul(next(o for o in got if o.name == "mlp_mul"), (t * _TOPK / tp) * _DE, 0, "bf16")
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _TOPK * _DE, 0, "fp8")
    gem = next(o for o in got if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    assert next(o for o in got if o.name == "expert_act").a_fmt == "bf16"
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("minimax-m2", "minimax-m2.1", "minimax-m3", "minimax-m3.1-flash",
                "glm-5", "glm-5.2", "glm-5.3", "atria-dawn", "drama-3",
                "qwen3-235b-a22b", "ernie-4.5-vl-28b-a3b-pt", "jais-3-7b"):
        other = get_model(mid)
        assert other.vec_kind != "minimax_swiglu"
        assert all(L.vec_kind != "minimax_swiglu" for L in other.layers)
    ernie = get_model("ernie-4.5-300b-a47b-pt")
    li = next(i for i, L in enumerate(ernie.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(ernie, li, ph, Shard()))
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("expert.down")
    assert ernie.layers[li].vec_kind == "ernie300_swiglu"
    wide = get_model("minimax-text-01")
    c_ok = _tc(load_release(_ROWS["minimax-text-01"])["config"])
    for mid in ("jais-3-32b", "ernie-4.5-300b-a47b-pt", "glm-5.3", "minimax-m2",
                "ernie-4.5-vl-424b-a47b-pt"):
        spoof = replace(_bare(get_model(mid)), id="minimax-text-01")
        _raises(lambda spoof=spoof: _tag_minimax_swiglu_vectors(spoof, c_ok),
                "minimax swiglu")
    kept = replace(_bare(wide), id="minimax-m2")
    assert _tag_minimax_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_jais3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_olmo3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_ernievl_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["minimax-text-01"]
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "minimax swiglu hidden_act is not silu")
    _raises(_bad(hf, lambda c: c.__setitem__("model_type", "minimax_m2")),
            "minimax swiglu model_type is not the published pair")
    _raises(_bad(hf, lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "minimax swiglu swiglu_limit is set")
    _raises(_bad(hf, lambda c: c.__setitem__("architectures", ["MiniMaxM1ForCausalLM"])),
            "minimax swiglu architecture is not the published pair")
    _raises(_bad(hf, lambda c: c.__setitem__("shared_intermediate_size", 9216)),
            "minimax swiglu shared expert is set")
    _raises(_bad(hf, lambda c: c.__setitem__("postnorm", False)),
            "minimax swiglu postnorm is not true")
    _raises(_bad(hf, lambda c: c.__setitem__("attn_type_list", [1] * 80)),
            "minimax swiglu attn_type_list is not the published pattern")
    _raises(_bad(hf, lambda c: c.__setitem__("layernorm_mlp_alpha", 1.0)),
            "minimax swiglu layernorm alpha is not the release")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "minimax swiglu attention_dropout is not 0")
    _raises(_bad(hf, lambda c: c.__setitem__("router_jitter_noise", 0.1)),
            "minimax swiglu router_jitter_noise is not 0")
    _raises(_bad(hf, lambda c: c.__setitem__("output_router_logits", True)),
            "minimax swiglu output_router_logits is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("sliding_window", 4096)),
            "minimax swiglu sliding_window is set")
    _raises(_bad(hf, lambda c: c.__setitem__("tie_word_embeddings", True)),
            "minimax swiglu tie_word_embeddings is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("rms_norm_eps", 1e-6)),
            "minimax swiglu rms_norm_eps is not 1e-5")
    _raises(_bad(hf, lambda c: c.__setitem__("num_experts_per_tok", 4)),
            "minimax swiglu experts are not 32 top-2")
    _raises(_bad(hf, lambda c: c.__setitem__("intermediate_size", 8192)),
            "minimax swiglu width is not the text release")
    _raises(_bad(hf, lambda c: c.__setitem__("head_dim", 64)),
            "minimax swiglu heads are not the text release")
    _raises(_bad(hf, lambda c: c.__setitem__("quantization_config", {"quant_method": "fp8"})),
            "minimax swiglu quantization_config is set")
    _raises(_bad(hf, lambda c: c.__setitem__("num_shared_experts", 1)),
            "minimax swiglu shared or alternate expert key is set")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_minimax_swiglu_vectors(short, c_ok),
            "minimax swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_minimax_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "minimax swiglu expert is not swiglu")
    dropped = replace(wide.layers[0], ffn_linears=())
    _raises(lambda: _tag_minimax_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "minimax swiglu linears are not router")
    runtime = replace(
        wide.layers[0],
        vec_kind="minimax_swiglu",
        ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
        "minimax swiglu expert is not swiglu")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    # These two rows are inside the first 60. Default decode ticks match
    # the untagged schedule at the hashed summary, and TP2 still refuses.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "6381be58dbab8f57"
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
