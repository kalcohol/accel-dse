"""0.470: Llama-3.3-70B bills published RMSNorm and the SwiGLU product.

LlamaForCausalLM uses LlamaRMSNorm and LlamaMLP. The reduction is the T5
recipe in fp32. SiLU stays the existing act op. The gate times up product
is mlp_mul. attention_bias, mlp_bias, and pretraining_tp are published
False, False, and 1. There is no q/k norm. q width equals hidden.
rope_scaling llama3 only rescales inv_freq at init. attention_scaling is
1, so the existing rope op is unchanged. vec_kind reuses carbon_llama.
The Llama-2 and Carbon allowlists are not extended. Only llama-3.3-70b
is on this allowlist. It is inside the first 60 catalog rows. Vector
flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    RoleFormat, _CARBON_LLAMA, _KIMI_SITU, _LLAMA2, _LLAMA3, _QWEN2_TEXT,
    _QWEN35_MOE, _tag_carbon_vectors, _tag_kimi_situ_vectors, _tag_llama2_vectors,
    _tag_llama3_vectors, _tag_qwen25_72b_vectors, _tag_qwen35_moe_vectors,
    _tag_qwen3_vectors, _tc, from_release, load_release,
)


_MID = "llama-3.3-70b"
_HF = "unsloth/Llama-3.3-70B-Instruct"
_H = 8192
_INNER = 28672
_DEPTH = 80
_HEADS = 64
_KV = 8
_HD = 128
_PARAMS = 70_553_706_496
_NORM = 1_318_912
_VOCAB = 128256
_ORDER = (
    "attn_norm", "attn.qkv", "attn.o", "qk", "pv", "softmax", "rope",
    "ffn_norm", "mlp.gate_up", "act", "mlp_mul", "mlp.down", "residual",
)
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "q_rms", "k_rms", "situ")
_SMALLER = ("llama-3.1-8b", "llama-3.2-3b", "llama-3.2-1b")
_ROPE = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def _by(ops):
    found = {}
    for op in ops:
        if op.name in found and op.kind == "vector" and not op.name.endswith(".ew"):
            raise AssertionError(op.name)
        found[op.name] = op
    return found


def _act(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "act"
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _mul(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _rms(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release(_HF)))
    edit(rel["config"])
    return lambda: from_release(_MID, _HF, rel)


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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops,
                            o.w_fmt, o.a_fmt, o.replicated))
    return out


def _ffn(depth: int, h: int, inner: int, t: int = 4096) -> int:
    return depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)


def test_version_is_470():
    assert __version__ == "0.479.0"


def test_row_is_front60_and_reuses_carbon_llama():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index(_MID) == 52
    assert _LLAMA3 == frozenset({_MID})
    assert _LLAMA3.isdisjoint(_LLAMA2)
    assert _LLAMA3.isdisjoint(_CARBON_LLAMA)
    assert _LLAMA3.isdisjoint(_QWEN2_TEXT)
    assert _LLAMA3.isdisjoint(_QWEN35_MOE)
    assert _LLAMA3.isdisjoint(_KIMI_SITU)
    m = get_model(_MID)
    assert m.hf_id == _HF
    assert m.params() == _PARAMS and m.release_params == _PARAMS
    assert m.param_check()["rel_err"] == 0.0
    assert m.n_layers == _DEPTH and m.hidden == _H and m.act_fmt == "bf16"
    assert m.tie_embeddings is False and m.vocab == _VOCAB
    assert m.vec_kind == "carbon_llama" and m.final_norm_params is None
    assert all(L.vec_kind == "carbon_llama" for L in m.layers)
    assert all(L.misc_params == 2 * _H for L in m.layers)
    assert all(L.ffn.gated and L.ffn.d_ff == _INNER and L.ffn.kind == "dense" for L in m.layers)
    assert all(L.core.kind == "gqa" and L.core.qk_dim == _HD for L in m.layers)
    assert all(L.core.n_q == _HEADS and L.core.n_kv == _KV for L in m.layers)
    assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
    assert _HEADS * _HD == _H
    assert _DEPTH * _INNER == 2_293_760
    by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
    assert (by_lin["q"].k, by_lin["q"].n) == (_H, _H)
    assert (by_lin["k"].k, by_lin["k"].n) == (_H, _KV * _HD)
    assert (by_lin["v"].k, by_lin["v"].n) == (_H, _KV * _HD)
    assert (by_lin["o"].k, by_lin["o"].n) == (_H, _H)
    assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
    assert m.fmt("mlp").bits == 16.0 and m.fmt("attn").bits == 16.0
    cfg = load_release(_HF)["config"]
    assert cfg["model_type"] == "llama"
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert cfg["hidden_act"] == "silu"
    assert cfg["hidden_size"] == cfg["num_attention_heads"] * _HD
    assert cfg["attention_bias"] is False
    assert cfg["mlp_bias"] is False
    assert cfg["pretraining_tp"] == 1
    assert cfg["rope_scaling"] == _ROPE
    assert cfg["attention_dropout"] == 0.0
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["transformers_version"] == "4.57.1"
    assert "sliding_window" not in cfg and "layer_types" not in cfg
    roles = load_release(_HF)["roles"]["norm"]["bf16"]
    assert roles["params"] == _NORM
    assert any("0.470" in n for n in m.notes)
    assert not any("0.464" in n or "0.445" in n for n in m.notes)
    bare = _bare(m)
    c = _tc(cfg)
    assert _tag_carbon_vectors(bare, c).vec_kind == ""
    assert _tag_llama2_vectors(bare, c).vec_kind == ""
    assert _tag_qwen25_72b_vectors(bare, c).vec_kind == ""
    assert _tag_qwen3_vectors(bare, c).vec_kind == ""
    assert _tag_qwen35_moe_vectors(bare, c).vec_kind == ""
    assert _tag_kimi_situ_vectors(bare, c).vec_kind == ""
    assert _tag_llama3_vectors(bare, c).vec_kind == "carbon_llama"
    # Integer factors are the same published numbers.
    rel = json.loads(json.dumps(load_release(_HF)))
    rel["config"]["rope_scaling"]["factor"] = 8
    rel["config"]["rope_scaling"]["high_freq_factor"] = 4
    rel["config"]["rope_scaling"]["low_freq_factor"] = 1
    rel["config"]["attention_dropout"] = 0
    tagged = from_release(_MID, _HF, rel)
    assert tagged.vec_kind == "carbon_llama"


def test_default_window_pre_norm_order():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    m = get_model(_MID)
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _act(by["act"], t * _INNER, 0, "bf16")
    _mul(by["mlp_mul"], t * _INNER, 0, "bf16")
    _rms(by["attn_norm"], t, _H, 0)
    _rms(by["ffn_norm"], t, _H, 0)
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, _H, 2 * _INNER)
    assert (down.m, down.k, down.n) == (t, _INNER, _H)
    assert up.flops == 2 * t * _H * (2 * _INNER)
    assert down.flops == 2 * t * _INNER * _H
    assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    assert down.w_fmt == "bf16" and down.w_bits == 16.0
    names = [o.name for o in ops]
    assert names == list(_ORDER)
    assert names.index("mlp.gate_up") < names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
    for banned in _BANNED:
        assert banned not in names
    rope = by["rope"]
    assert rope.vec == t * _HEADS * (_HD // 2) * 3
    assert rope.sfu == 0.0 and rope.a_fmt == "bf16" and rope.flops == 0
    last = _by(build_rank_ops(m, _DEPTH - 1, ph, Shard()))
    _act(last["act"], t * _INNER, _DEPTH - 1, "bf16")
    _mul(last["mlp_mul"], t * _INNER, _DEPTH - 1, "bf16")
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, _H, _DEPTH)
    two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
    _rms(two["final_norm"], 2 * t, _H, _DEPTH)
    bare = _bare(m)
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    bare_by = _by(bare_ops)
    bare_names = [o.name for o in bare_ops]
    assert "mlp_mul" not in bare_names
    assert bare_names.index("mlp.down") < bare_names.index("act")
    assert by["rope"].vec == bare_by["rope"].vec
    assert by["rope"].sfu == bare_by["rope"].sfu
    assert by["residual"].vec == bare_by["residual"].vec
    assert bare_by["attn_norm"].a_fmt == "bf16" and bare_by["attn_norm"].sfu == float(t)
    assert bare_by["ffn_norm"].a_fmt == "bf16"
    assert bare_by["act"].a_fmt == "bf16"
    bare_head = _by(head_ops(bare, ph, Shard()))
    assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
    assert head["final_norm"].vec == bare_head["final_norm"].vec
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    assert all(o.flops == b.flops for o, b in zip(
        (x for x in ops if x.kind == "gemm"),
        (x for x in bare_ops if x.kind == "gemm")))


def test_decode_batch_tp_and_fp8():
    m = get_model(_MID)
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], _INNER, 0, "bf16")
    _mul(dec["mlp_mul"], _INNER, 0, "bf16")
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    assert dec["attn_norm"].sfu == 1.0
    assert dec["rope"].vec == _HEADS * (_HD // 2) * 3
    assert "q_rms" not in dec and "k_rms" not in dec
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * _INNER, 0, "bf16")
    _mul(two["mlp_mul"], 2 * _INNER, 0, "bf16")
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * _INNER, 0, "bf16")
    _mul(pre["mlp_mul"], 8192 * _INNER, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    half = _INNER // 2
    assert half == 14336
    _act(loc["act"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    assert loc["mlp.gate_up"].n == _INNER and loc["mlp.down"].k == half
    assert loc["ffn_norm"].vec == 4096 * _H * 4 and loc["mlp_mul"].replicated == 1
    assert loc["attn_norm"].vec == 4096 * _H * 4
    assert loc["rope"].vec == 4096 * (_HEADS // 2) * (_HD // 2) * 3
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn_norm") < names.index("attn.qkv")
    assert names.index("attn.o") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert names.index("ffn_norm") < names.index("mlp.gate_up")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
    assert "q_rms" not in names and "k_rms" not in names
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * _INNER, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * _INNER, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    # rope does not follow the request activation dtype. The op stays the
    # historical default, same element count as the untagged model.
    assert op["rope"].a_fmt == "bf16" and op["rope"].vec == 4096 * _HEADS * (_HD // 2) * 3
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_smaller_llama_stay_untagged():
    ph = Phase("prefill", 1, 256, 0)
    m = get_model(_MID)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4), Shard(tp=8)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    c = _tc(load_release(_HF)["config"])
    retagged = _tag_llama3_vectors(bare, c)
    assert retagged.vec_kind == "carbon_llama"
    assert all(L.vec_kind == "carbon_llama" for L in retagged.layers)
    for mid in _SMALLER:
        other = get_model(mid)
        assert other.vec_kind == "", mid
        assert all(L.vec_kind == "" for L in other.layers), mid
        pops = build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())
        assert "mlp_mul" not in [o.name for o in pops], mid
        assert load_release(other.hf_id)["config"].get("rope_scaling") is not None
    swapped = from_release("llama-3.1-8b", _HF, load_release(_HF))
    assert swapped.vec_kind == ""
    assert _tag_llama3_vectors(swapped, c).vec_kind == ""
    llama2 = get_model("llama-2-70b")
    assert llama2.vec_kind == "carbon_llama"
    assert any("0.464" in n for n in llama2.notes)
    assert not any("0.470" in n for n in llama2.notes)
    assert _tag_llama3_vectors(llama2, _tc(load_release(llama2.hf_id)["config"])) is llama2
    carbon = get_model("carbon-8b")
    assert carbon.vec_kind == "carbon_llama"
    assert any("0.445" in n for n in carbon.notes)
    assert not any("0.470" in n for n in carbon.notes)
    qwen = get_model("qwen2.5-72b")
    assert qwen.vec_kind == "qwen2_text"
    assert not any("0.470" in n for n in qwen.notes)
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"


def test_published_flags_and_linear_shape_are_refused():
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu")),
            "llama3 hidden_act is not silu")
    _raises(_bad(lambda c: c.__setitem__("model_type", "qwen2")),
            "llama3 model_type is not llama")
    _raises(_bad(lambda c: c.__setitem__("architectures", ["Qwen2ForCausalLM"])),
            "llama3 architecture is not llama")
    _raises(_bad(lambda c: c.__setitem__("attention_bias", True)),
            "llama3 attention_bias is not false")
    _raises(_bad(lambda c: c.__delitem__("attention_bias")),
            "llama3 attention_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("mlp_bias", True)),
            "llama3 mlp_bias is not false")
    _raises(_bad(lambda c: c.__delitem__("mlp_bias")),
            "llama3 mlp_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("pretraining_tp", 2)),
            "llama3 pretraining_tp is not 1")
    _raises(_bad(lambda c: c.__delitem__("pretraining_tp")),
            "llama3 pretraining_tp is not 1")
    _raises(_bad(lambda c: c.__setitem__("rope_scaling", {"rope_type": "linear", "factor": 8.0})),
            "llama3 rope_scaling is not llama3")
    _raises(_bad(lambda c: c.__setitem__("rope_scaling", None)),
            "llama3 rope_scaling is not the published llama3 factors")
    _raises(_bad(lambda c: c["rope_scaling"].__setitem__("factor", 32.0)),
            "llama3 rope_scaling is not the published llama3 factors")
    _raises(_bad(lambda c: c["rope_scaling"].__setitem__("extra", 1)),
            "llama3 rope_scaling is not the published llama3 factors")
    _raises(_bad(lambda c: c.__setitem__("attention_dropout", 0.1)),
            "llama3 attention_dropout is not 0")
    _raises(_bad(lambda c: c.__delitem__("attention_dropout")),
            "llama3 attention_dropout is not 0")
    _raises(_bad(lambda c: c.__setitem__("use_sliding_window", True)),
            "llama3 use_sliding_window is not false")
    _raises(_bad(lambda c: c.__setitem__("sliding_window", 1024)),
            "llama3 sliding_window is not null")
    _raises(_bad(lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "llama3 layer_types is not full_attention")
    _raises(_bad(lambda c: c.__setitem__("head_dim", 64)),
            "llama3 head_dim is not hidden over heads")
    _raises(_bad(lambda c: c.__setitem__("num_attention_heads", 48)),
            "llama3 hidden is not heads times head_dim")
    m = get_model(_MID)
    c_ok = _tc(load_release(_HF)["config"])
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()),
            "carbon mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_llama3_vectors(broken_g, c_ok),
            "llama3 mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_llama3_vectors(broken_m, c_ok),
            "llama3 has no qk norm")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_llama3_vectors(broken_h, c_ok),
            "llama3 attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 128)
    broken_q = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(broken_q,) + m.layers[1:])
    _raises(lambda: _tag_llama3_vectors(broken_w, c_ok),
            "llama3 qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_llama3_vectors(broken_win, c_ok),
            "llama3 attention is not gqa head_dim")
    two = replace(m, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_llama3_vectors(two, c_ok),
            "llama3 final norm is not one")
    fp32 = replace(m, act_fmt="fp32")
    _raises(lambda: _tag_llama3_vectors(fp32, c_ok),
            "llama3 activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("fp16", 16.0) if k == "mlp" else v) for k, v in m.formats)
    wide_w = replace(_bare(m), formats=fmts)
    _raises(lambda: _tag_llama3_vectors(wide_w, c_ok),
            "llama3 weight dtype is not bf16")
    rel2 = json.loads(json.dumps(load_release("NousResearch/Llama-2-70b-hf")))
    rel2["config"]["attention_bias"] = False
    rel2["config"]["mlp_bias"] = False
    rel2["config"]["pretraining_tp"] = 1
    rel2["config"]["attention_dropout"] = 0.0
    _raises(lambda: from_release(_MID, "NousResearch/Llama-2-70b-hf", rel2),
            "llama3 rope_scaling is not the published llama3 factors")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    expect = _ffn(_DEPTH, _H, _INNER, t)
    m = get_model(_MID)
    got = 0.0
    for li in range(m.n_layers):
        for o in build_rank_ops(m, li, ph, Shard()):
            if o.name in ("mlp.gate_up", "mlp.down"):
                got += o.flops
            if o.kind == "vector":
                assert o.flops == 0
    assert got == expect
    assert got == 461_794_883_665_920


def test_fingerprints_and_kpi():
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651, _kpi
    # Decode summaries on the locked window match the untagged model, so
    # the pipe hashes stay at the 0.468 / 0.469 values.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    from tests.test_core_125 import test_fingerprints_change_only_v4
    test_fingerprints_change_only_v4()
    from accel_dse.core import evaluate as E
    with E.model_flags(FUSE_QKV=False):
        assert _kpi(_MID) == "8e50c6cabdc8"
    assert _FULL_0651[_MID] == "8e50c6cabdc8"
    assert _FULL_0651["qwen3-32b"] == "b975b7b3760e"
    assert _FULL_0651["qwen3-8b"] == "11f6f3a5d21f"
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
