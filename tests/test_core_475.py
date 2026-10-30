"""0.475: Jais-3-32B bills the published dense SwiGLU product.

K2HorizonMLP is down(silu(gate) * up). SiLU stays act. The product is
mlp_mul on the intermediate width. a_fmt follows act_fmt. The class does
not upcast. Group RMSNorm stays on the historical pre-norm ops.
query_key_norm is false. rope_head_dim equals head_dim, so the existing
rope op is unchanged. vec_kind is jais3_swiglu. No earlier allowlist is
extended. The two rows sit outside the first 60. Smaller Jais rows stay
untagged. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _JAIS3_SWIGLU, _KIMI_SITU, _LLAMA3, _MIXTRAL_SWIGLU, _MUSE_SWIGLU,
    _OLMO3_SWIGLU, _QWEN35_MOE, _SEED_SWIGLU, _tag_jais3_swiglu_vectors,
    _tag_kimi_situ_vectors, _tag_llama3_vectors, _tag_mixtral_swiglu_vectors,
    _tag_muse_swiglu_vectors, _tag_olmo3_swiglu_vectors, _tag_qwen35_moe_vectors,
    _tag_seed_swiglu_vectors, _tc, from_release, load_release,
)


_ROWS = {
    "jais-3-32b": "inception42/Jais-3-32B",
    "jais-3-32b-fp8": "IFM/Jais-3-32B-FP8",
}
_H = 5120
_INNER = 26624
_DEPTH = 64
_HEADS = 64
_KV = 8
_HD = 128
_PARAMS = 34_779_304_960
_NORM = 660_480
_MISC = 2 * _H
_TEXT = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "q_rms", "k_rms", "expert_act",
           "attn_out_gate")
_FFN_TFLOP = 214.40476741632


def _names(ops):
    return [o.name for o in ops]


def _act(op, n: float, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "act"
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
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


def _bad(hf, edit, mid="jais-3-32b"):
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


def test_version_is_475():
    assert __version__ == "0.481.0"


def test_two_rows_bill_the_dense_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(mid) for mid in _ROWS] == [303, 308]
    assert _JAIS3_SWIGLU == frozenset(_ROWS)
    assert _JAIS3_SWIGLU.isdisjoint(_OLMO3_SWIGLU)
    assert _JAIS3_SWIGLU.isdisjoint(_SEED_SWIGLU)
    assert _JAIS3_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _JAIS3_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _JAIS3_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _JAIS3_SWIGLU.isdisjoint(_LLAMA3)
    assert _JAIS3_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _DEPTH * _INNER == 1_703_936
    assert _HEADS * _HD != _H
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.vec_kind == "jais3_swiglu"
        assert m.params() == _PARAMS and m.release_params == _PARAMS
        assert m.param_check()["rel_err"] == 0.0
        assert m.mtp_layers == () and m.tie_embeddings is False
        assert m.kv_fmt == "bf16"
        assert all(L.vec_kind == "jais3_swiglu" for L in m.layers)
        assert all(L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER
                   for L in m.layers)
        assert all(L.misc_params == _MISC for L in m.layers)
        assert all(L.core.window is None and not L.core.no_rope and L.core.rope_dim == 0
                   and not L.core.out_gate_fn for L in m.layers)
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "k2_horizon"
        assert cfg["architectures"] == ["K2HorizonForCausalLM"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["attention_bias"] is False
        assert cfg["attention_gate_func"] is None
        assert cfg["query_key_norm"] is False
        assert "mlp_bias" not in cfg and "residual_dropout" not in cfg
        assert "attention_out_bias" not in cfg
        assert cfg["attention_dropout"] == 0.0
        assert cfg["layernorm_num_groups"] == 4
        assert cfg["rope_parameters"]["rope_type"] == "default"
        assert cfg["rope_head_dim"] == _HD and cfg["head_dim"] == _HD
        assert cfg["sliding_window"] is None and cfg["use_sliding_window"] is False
        assert cfg.get("dtype") == "bfloat16"
        assert cfg["tie_word_embeddings"] is False
        assert cfg["rms_norm_eps"] == 1e-6
        assert cfg["mlp_only_layers"] == list(range(_DEPTH))
        assert cfg["num_experts"] == 0 and cfg["moe_intermediate_size"] == 0
        assert cfg["intermediate_size"] == _INNER
        assert cfg["num_attention_heads"] == _HEADS and cfg["num_key_value_heads"] == _KV
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        roles = load_release(hf)["roles"]["norm"]["bf16"]
        assert roles["params"] == _NORM
        assert any(n.startswith("0.475") for n in m.notes)
        assert _tag_jais3_swiglu_vectors(_bare(m), cfg).vec_kind == "jais3_swiglu"
    bf = get_model("jais-3-32b")
    fp = get_model("jais-3-32b-fp8")
    assert bf.act_fmt == "bf16" and bf.fmt("mlp").fmt == "bf16" and not bf.quantized_release
    assert fp.act_fmt == "fp8" and fp.fmt("mlp").fmt == "fp8" and fp.fmt("attn").fmt == "fp8"
    assert fp.fmt("embed").fmt == "bf16" and fp.fmt("lm_head").fmt == "bf16"
    assert fp.quantized_release
    for mid in ("jais-3-7b", "jais-3-7b-fp8", "jais-3-3.7b", "jais-3-3.7b-fp8",
                "jais-3-0.9b", "jais-3-375b-a23b", "jais-3-375b-a23b-fp8",
                "jais-3-mova-36b-a4b", "jais-3-mova-36b-a4b-fp8"):
        assert get_model(mid).vec_kind == ""
    assert get_model("olmo-3-32b-think").vec_kind == "olmo3_swiglu"
    assert get_model("seed-oss-36b").vec_kind == "seed_swiglu"
    assert get_model("mixtral-8x22b").vec_kind == "mixtral_swiglu"
    assert get_model("minimax-text-01").vec_kind == "minimax_swiglu"
    assert get_model("glm-5.3").vec_kind == "glm5_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model("jais-3-32b")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    ops = build_rank_ops(m, 0, ph, Shard())
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    _mul(next(o for o in ops if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    assert next(o for o in ops if o.name == "mlp_mul").vec == 4096 * 26624
    _act(next(o for o in ops if o.name == "act"), t * _INNER, 0, "bf16")
    names = _names(ops)
    assert [n for n in names if n in _TEXT or n in _BANNED] == list(_TEXT)
    assert names.index("mlp.gate_up") < names.index("act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("mlp.down")
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
    flops = 0.0
    bare_flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
        for o in layer:
            if o.name in ("mlp.gate_up", "mlp.down"):
                flops += o.flops
        for o in build_rank_ops(bare, li, ph, Shard()):
            if o.name in ("mlp.gate_up", "mlp.down"):
                bare_flops += o.flops
    assert flops == bare_flops
    assert flops / 1e12 == _FFN_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    fp = get_model("jais-3-32b-fp8")
    fp_ops = build_rank_ops(fp, 0, ph, Shard())
    _mul(next(o for o in fp_ops if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    _act(next(o for o in fp_ops if o.name == "act"), t * _INNER, 0, "fp8")
    fp_up = next(o for o in fp_ops if o.name == "mlp.gate_up")
    assert fp_up.a_fmt == "fp8" and fp_up.w_fmt == "fp8"
    assert fp_up.w_bits == fp.fmt("mlp").bits
    fp_bare = build_rank_ops(_bare(fp), 0, ph, Shard())
    for name in ("attn_norm", "ffn_norm", "rope", "residual"):
        got = next(o for o in fp_ops if o.name == name)
        old = next(o for o in fp_bare if o.name == name)
        assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
    assert next(o for o in fp_bare if o.name == "act").a_fmt == "bf16"


def test_tp_request_fp8_and_cleared_kind():
    m = get_model("jais-3-32b")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for tp in (2, 4, 8):
        got = build_rank_ops(m, 0, ph, Shard(tp=tp))
        _mul(next(o for o in got if o.name == "mlp_mul"), t * (_INNER // tp), 0, "bf16")
        assert next(o for o in got if o.name == "mlp.down").k == _INNER // tp
        assert _sched(m, ph, Shard(tp=tp)) == _sched(_bare(m), ph, Shard(tp=tp))
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    _act(next(o for o in got if o.name == "act"), t * _INNER, 0, "fp8")
    gem = next(o for o in got if o.name == "mlp.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("mlp.gate_up") < bnames.index("mlp.down") < bnames.index("act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    published = get_model("jais-3-32b-fp8")
    assert _sched(published, ph, Shard()) == _sched(_bare(published), ph, Shard())


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("jais-3-7b", "jais-3-375b-a23b", "jais-3-mova-36b-a4b",
                "olmo-3-7b-think", "seed-oss-36b", "minimax-text-01", "glm-5.3",
                "qwen3-235b-a22b", "drama-3", "ernie-4.5-vl-424b-a47b"):
        other = get_model(mid)
        assert other.vec_kind != "jais3_swiglu"
        assert all(L.vec_kind != "jais3_swiglu" for L in other.layers)
        names = _names(build_rank_ops(other, 0, ph, Shard()))
        if other.layers[0].ffn.kind == "dense" and other.vec_kind == "":
            assert "mlp_mul" not in names
    wide = get_model("jais-3-32b")
    c_ok = _tc(load_release(_ROWS["jais-3-32b"])["config"])
    c_fp = _tc(load_release(_ROWS["jais-3-32b-fp8"])["config"])
    for mid in ("jais-3-7b", "olmo-3-32b-think", "seed-oss-36b", "jais-3-375b-a23b"):
        spoof = replace(_bare(get_model(mid)), id="jais-3-32b")
        _raises(lambda spoof=spoof: _tag_jais3_swiglu_vectors(spoof, c_ok),
                "jais3 swiglu")
    kept = replace(_bare(wide), id="jais-3-7b")
    assert _tag_jais3_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_olmo3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["jais-3-32b"]
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "jais3 swiglu hidden_act is not silu")
    _raises(_bad(hf, lambda c: c.__setitem__("model_type", "llama")),
            "jais3 swiglu model_type is not k2_horizon")
    _raises(_bad(hf, lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "jais3 swiglu swiglu_limit is set")
    _raises(_bad(hf, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "jais3 swiglu architecture is not k2_horizon")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_bias", True)),
            "jais3 swiglu attention_bias is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("mlp_bias", False)),
            "jais3 swiglu bias key is set")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_gate_func", "silu")),
            "jais3 swiglu attention_gate_func is set")
    _raises(_bad(hf, lambda c: c.__setitem__("query_key_norm", True)),
            "jais3 swiglu query_key_norm is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "jais3 swiglu attention_dropout is not 0")
    _raises(_bad(hf, lambda c: c.__setitem__("layernorm_num_groups", 1)),
            "jais3 swiglu layernorm_num_groups is not 4")
    _raises(_bad(hf, lambda c: c.__setitem__("rope_parameters", {"rope_type": "yarn"})),
            "jais3 swiglu rope_type is not default")
    _raises(_bad(hf, lambda c: c.__setitem__("rope_head_dim", 64)),
            "jais3 swiglu rope_head_dim is not head_dim")
    _raises(_bad(hf, lambda c: c.__setitem__("sliding_window", 4096)),
            "jais3 swiglu sliding_window is set")
    _raises(_bad(hf, lambda c: c.__setitem__("dtype", "float16")),
            "jais3 swiglu dtype is not bfloat16")
    _raises(_bad(hf, lambda c: c.__setitem__("tie_word_embeddings", True)),
            "jais3 swiglu tie_word_embeddings is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("rms_norm_eps", 1e-5)),
            "jais3 swiglu rms_norm_eps is not 1e-6")
    _raises(_bad(hf, lambda c: c.__setitem__("num_experts", 8)),
            "jais3 swiglu moe key is set")
    _raises(_bad(hf, lambda c: c.__setitem__("mlp_only_layers", [0, 1, 2])),
            "jais3 swiglu mlp_only_layers is not every layer")
    _raises(_bad(hf, lambda c: c.__setitem__("intermediate_size", 16384)),
            "jais3 swiglu width is not the 32b release")
    _raises(_bad(hf, lambda c: c.__setitem__("quantization_config", {"quant_method": "fp8"})),
            "jais3 swiglu quantization_config is set")
    _raises(_bad(_ROWS["jais-3-32b-fp8"], lambda c: c.__setitem__("quantization_config", None),
                 "jais-3-32b-fp8"),
            "jais3 swiglu quant method is not compressed-tensors")
    assert _tag_jais3_swiglu_vectors(_bare(get_model("jais-3-32b-fp8")), c_fp).vec_kind == "jais3_swiglu"
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_jais3_swiglu_vectors(short, c_ok),
            "jais3 swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_jais3_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "jais3 swiglu mlp is not dense swiglu")
    dropped = replace(wide.layers[0], ffn_linears=wide.layers[0].ffn_linears[:1])
    _raises(lambda: _tag_jais3_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "jais3 swiglu mlp is not gate_up then down")
    runtime = replace(
        wide.layers[0],
        vec_kind="jais3_swiglu",
        ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
        "jais3 swiglu mlp is not gate_up then down")


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
