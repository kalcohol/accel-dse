"""0.464: Llama-2 and Yi-1.5 bill published RMSNorm and the SwiGLU product.

LlamaForCausalLM uses LlamaRMSNorm and LlamaMLP. The reduction is the T5
recipe in fp32. SiLU stays the existing act op. The gate times up product
is mlp_mul. Transformers 4.31 hardcodes projection bias to False and
defaults pretraining_tp to 1 (the unsplit product). An omitted key is that
published default. Transformers 4.37 still hardcodes MLP bias False.
Yi-1.5 publishes attention_bias false and pretraining_tp 1, and omits
mlp_bias. There is no q/k norm. q width equals hidden. vec_kind reuses
carbon_llama. The Carbon allowlist is not extended. These three ids are
inside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    RoleFormat, _CARBON_LLAMA, _LLAMA2, _LLAMA2_BF16, _LLAMA2_FP16, _QWEN2_TEXT,
    _tag_carbon_vectors, _tag_llama2_vectors, _tag_qwen25_72b_vectors,
    _tag_qwen3_vectors, _tc, from_release, load_release,
)


# id, hf, hidden, inner, depth, heads, kv heads, head_dim, params, release,
# persistent inv_freq gap, weight dtype, vocab, norm params.
_ROWS = (
    ("llama-2-70b", "NousResearch/Llama-2-70b-hf", 8192, 28672, 80, 64, 8, 128,
     68_976_648_192, 68_976_653_312, 5120, "fp16", 32000, 1_318_912),
    ("llama-2-7b", "NousResearch/Llama-2-7b-hf", 4096, 11008, 32, 32, 32, 128,
     6_738_415_616, 6_738_417_664, 2048, "fp16", 32000, 266_240),
    ("yi-1.5-34b", "01-ai/Yi-1.5-34B", 7168, 20480, 60, 56, 8, 128,
     34_388_917_248, 34_388_917_248, 0, "bf16", 64000, 867_328),
)
_ORDER = (
    "attn_norm", "attn.qkv", "attn.o", "qk", "pv", "softmax", "rope",
    "ffn_norm", "mlp.gate_up", "act", "mlp_mul", "mlp.down", "residual",
)
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "q_rms", "k_rms")
# Front-60 rows that stay on the shared tail.
_UNTOUCHED = (
    # 0.466 bills qwen2.5-3b and qwen2.5-1.5b with qwen2_text.
    "llama-3.1-8b",
    # 0.465 bills qwen3-8b, its fp8 and AWQ twins, and qwen3-1.7b with qwen3.
    "qwen3-32b", "qwen3-4b", "qwen3-0.6b",
    "mistral-7b", "mixtral-8x22b", "magistral-small",
)


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


def _bad(mid: str, hf: str, edit):
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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops,
                            o.w_fmt, o.a_fmt, o.replicated))
    return out


def _ffn(depth: int, h: int, inner: int, t: int = 4096) -> int:
    return depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)


def test_version_is_464():
    assert __version__ == "0.466.0"


def test_three_rows_are_front60_and_reuse_carbon_llama():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert _LLAMA2 == _LLAMA2_FP16 | _LLAMA2_BF16
    assert _LLAMA2.isdisjoint(_CARBON_LLAMA)
    assert _LLAMA2.isdisjoint(_QWEN2_TEXT)
    for mid, hf, h, inner, depth, heads, hkv, hd, params, release, gap, w, vocab, norm in _ROWS:
        assert ids.index(mid) < 60
        assert mid in _LLAMA2 and mid not in _CARBON_LLAMA
        m = get_model(mid)
        assert m.hf_id == hf
        assert m.params() == params and m.release_params == release
        assert release - m.params() == gap
        if gap == 0:
            assert m.param_check()["rel_err"] == 0.0
        else:
            assert m.param_check()["rel_err"] != 0.0
            assert gap == depth * (hd // 2)
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is False and m.vocab == vocab
        assert m.vec_kind == "carbon_llama" and m.final_norm_params is None
        assert all(L.vec_kind == "carbon_llama" for L in m.layers)
        assert all(L.misc_params == 2 * h for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == heads and L.core.n_kv == hkv for L in m.layers)
        assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
        assert heads * hd == h
        assert depth * inner == {80: 2_293_760, 32: 352_256, 60: 1_228_800}[depth]
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (h, h)
        assert (by_lin["k"].k, by_lin["k"].n) == (h, hkv * hd)
        assert (by_lin["v"].k, by_lin["v"].n) == (h, hkv * hd)
        assert (by_lin["o"].k, by_lin["o"].n) == (h, h)
        assert m.fmt("mlp").fmt == w and m.fmt("attn").fmt == w
        assert m.fmt("mlp").bits == 16.0
        if w == "fp16":
            assert m.fmt("attn").bits > 16.0
        else:
            assert m.fmt("attn").bits == 16.0
        cfg = load_release(hf)["config"]
        assert cfg["model_type"] == "llama"
        assert cfg["architectures"] == ["LlamaForCausalLM"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["hidden_size"] == cfg["num_attention_heads"] * hd
        assert "mlp_bias" not in cfg
        roles = load_release(hf)["roles"]["norm"][w]
        assert roles["params"] == norm
        assert any("0.464" in n for n in m.notes)
        assert not any("0.445" in n for n in m.notes)
        bare = _bare(m)
        c = _tc(cfg)
        assert _tag_carbon_vectors(bare, c).vec_kind == ""
        assert _tag_qwen25_72b_vectors(bare, c).vec_kind == ""
        assert _tag_qwen3_vectors(bare, c).vec_kind == ""
        assert _tag_llama2_vectors(bare, c).vec_kind == "carbon_llama"
    cfg70 = load_release("NousResearch/Llama-2-70b-hf")["config"]
    for key in ("attention_bias", "mlp_bias", "pretraining_tp", "rope_scaling"):
        assert key not in cfg70
    assert cfg70["transformers_version"] == "4.31.0.dev0"
    assert cfg70["torch_dtype"] == "float16"
    cfg7 = load_release("NousResearch/Llama-2-7b-hf")["config"]
    assert "attention_bias" not in cfg7 and "mlp_bias" not in cfg7
    assert cfg7["pretraining_tp"] == 1 and cfg7["rope_scaling"] is None
    assert cfg7["transformers_version"] == "4.31.0.dev0"
    cfgy = load_release("01-ai/Yi-1.5-34B")["config"]
    assert cfgy["attention_bias"] is False and "mlp_bias" not in cfgy
    assert cfgy["pretraining_tp"] == 1 and cfgy["rope_scaling"] is None
    assert cfgy["attention_dropout"] == 0.0
    assert cfgy["rope_theta"] == 5_000_000.0 and cfgy["rms_norm_eps"] == 1e-6
    assert cfgy["transformers_version"] == "4.37.2"
    assert cfgy["torch_dtype"] == "bfloat16"


def test_explicit_published_defaults_still_tag():
    rel = json.loads(json.dumps(load_release("NousResearch/Llama-2-70b-hf")))
    rel["config"]["attention_bias"] = False
    rel["config"]["mlp_bias"] = False
    rel["config"]["pretraining_tp"] = 1
    rel["config"]["rope_scaling"] = None
    m = from_release("llama-2-70b", "NousResearch/Llama-2-70b-hf", rel)
    assert m.vec_kind == "carbon_llama"
    assert all(L.vec_kind == "carbon_llama" for L in m.layers)
    rel7 = json.loads(json.dumps(load_release("NousResearch/Llama-2-7b-hf")))
    rel7["config"]["attention_bias"] = False
    rel7["config"]["mlp_bias"] = False
    tagged = from_release("llama-2-7b", "NousResearch/Llama-2-7b-hf", rel7)
    assert tagged.vec_kind == "carbon_llama"
    rely = json.loads(json.dumps(load_release("01-ai/Yi-1.5-34B")))
    rely["config"]["mlp_bias"] = False
    yi = from_release("yi-1.5-34b", "01-ai/Yi-1.5-34B", rely)
    assert yi.vec_kind == "carbon_llama"


def test_default_window_pre_norm_order():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, _hf, h, inner, depth, _heads, _hkv, _hd, *_rest in _ROWS:
        w = _rest[3]
        m = get_model(mid)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
        _rms(by["attn_norm"], t, h, 0)
        _rms(by["ffn_norm"], t, h, 0)
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, h, 2 * inner)
        assert (down.m, down.k, down.n) == (t, inner, h)
        assert up.flops == 2 * t * h * (2 * inner)
        assert down.flops == 2 * t * inner * h
        assert up.w_fmt == w and up.w_bits == 16.0 and up.a_fmt == "bf16"
        assert down.w_fmt == w and down.w_bits == 16.0
        assert [o.name for o in ops] == list(_ORDER)
        for banned in _BANNED:
            assert banned not in [o.name for o in ops]
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, depth)
        two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
        _rms(two["final_norm"], 2 * t, h, depth)
        bare = _bare(m)
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert "mlp_mul" not in [o.name for o in bare_ops]
        assert by["residual"].vec == _by(bare_ops)["residual"].vec
        assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        assert all(o.flops == 0 for o in ops if o.kind == "vector")


def test_decode_batch_tp_and_fp8():
    for mid, _hf, h, inner, _depth, _heads, _hkv, _hd, *_rest in _ROWS:
        w = _rest[3]
        m = get_model(mid)
        dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
        _act(dec["act"], inner, 0, "bf16")
        _mul(dec["mlp_mul"], inner, 0, "bf16")
        assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
        assert dec["attn_norm"].sfu == 1.0
        assert "q_rms" not in dec and "k_rms" not in dec
        two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
        _act(two["act"], 2 * inner, 0, "bf16")
        _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
        pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
        _act(pre["act"], 8192 * inner, 0, "bf16")
        assert pre["mlp.gate_up"].m == 8192
        sh = Shard(tp=2)
        loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
        half = inner // 2
        _act(loc["act"], 4096 * half, 0, "bf16")
        _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
        assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == half
        assert loc["ffn_norm"].vec == 4096 * h * 4 and loc["mlp_mul"].replicated == 1
        assert loc["attn_norm"].vec == 4096 * h * 4
        names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
        assert names.index("attn_norm") < names.index("attn.qkv")
        assert names.index("attn.o") < names.index("attn_allreduce")
        assert names.index("mlp.down") < names.index("mlp_allreduce")
        assert names.index("ffn_norm") < names.index("mlp.gate_up")
        assert "q_rms" not in names and "k_rms" not in names
        fp8 = replace(m, act_fmt="fp8")
        op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
        _act(op["act"], 4096 * inner, 0, "fp8")
        _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
        assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
        assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == w
        assert op["mlp.down"].w_bits == 16.0
        head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
        assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_other_rows_stay_untagged():
    ph = Phase("prefill", 1, 256, 0)
    for mid, hf, *_rest in _ROWS:
        m = get_model(mid)
        bare = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(bare, ph, shard)
        c = _tc(load_release(hf)["config"])
        retagged = _tag_llama2_vectors(bare, c)
        assert retagged.vec_kind == "carbon_llama"
        assert all(L.vec_kind == "carbon_llama" for L in retagged.layers)
    for mid in _UNTOUCHED:
        other = get_model(mid)
        assert other.vec_kind == "", mid
        assert all(L.vec_kind == "" for L in other.layers), mid
        pops = build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())
        assert "mlp_mul" not in [o.name for o in pops], mid
    # 0.466 bills qwen2.5-3b. The Llama-2 gate still does not claim it.
    assert _tag_llama2_vectors(_bare(get_model("qwen2.5-3b")), _tc({})).vec_kind == ""
    assert get_model("qwen2.5-3b").vec_kind == "qwen2_text"
    qwen = get_model("qwen2.5-72b")
    assert qwen.vec_kind == "qwen2_text"
    assert any("0.463" in n for n in qwen.notes)
    assert not any("0.464" in n for n in qwen.notes)
    assert get_model("mistral-large").vec_kind == "mistral_small"
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    carbon = get_model("carbon-8b")
    assert carbon.vec_kind == "carbon_llama"
    assert any("0.445" in n for n in carbon.notes)
    assert not any("0.464" in n for n in carbon.notes)
    assert _tag_llama2_vectors(carbon, _tc(load_release(carbon.hf_id)["config"])) is carbon
    llama3 = get_model("llama-3.1-8b")
    assert llama3.vec_kind == ""
    assert load_release(llama3.hf_id)["config"].get("rope_scaling") is not None


def test_published_flags_and_linear_shape_are_refused():
    mid, hf = "llama-2-70b", "NousResearch/Llama-2-70b-hf"
    _raises(_bad(mid, hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "llama2 hidden_act is not silu")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("model_type", "qwen2")),
            "llama2 model_type is not llama")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("architectures", ["Qwen2ForCausalLM"])),
            "llama2 architecture is not llama")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("attention_bias", True)),
            "llama2 attention_bias is not false")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("mlp_bias", True)),
            "llama2 mlp_bias is not false")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("pretraining_tp", 2)),
            "llama2 pretraining_tp is not 1")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("rope_scaling", {"rope_type": "llama3"})),
            "llama2 rope_scaling is not null")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "llama2 attention_dropout is not 0")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("use_sliding_window", True)),
            "llama2 use_sliding_window is not false")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("sliding_window", 1024)),
            "llama2 sliding_window is not null")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "llama2 layer_types is not full_attention")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("head_dim", 64)),
            "llama2 head_dim is not hidden over heads")
    _raises(_bad(mid, hf, lambda c: c.__setitem__("num_attention_heads", 48)),
            "llama2 hidden is not heads times head_dim")
    _raises(_bad("yi-1.5-34b", "01-ai/Yi-1.5-34B", lambda c: c.__setitem__("mlp_bias", True)),
            "llama2 mlp_bias is not false")
    _raises(_bad("llama-2-7b", "NousResearch/Llama-2-7b-hf",
                 lambda c: c.__setitem__("attention_bias", True)),
            "llama2 attention_bias is not false")
    m = get_model(mid)
    c_ok = _tc(load_release(hf)["config"])
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()),
            "carbon mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_llama2_vectors(broken_g, c_ok),
            "llama2 mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_llama2_vectors(broken_m, c_ok),
            "llama2 has no qk norm")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_llama2_vectors(broken_h, c_ok),
            "llama2 attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 128)
    broken_q = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(broken_q,) + m.layers[1:])
    _raises(lambda: _tag_llama2_vectors(broken_w, c_ok),
            "llama2 qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_llama2_vectors(broken_win, c_ok),
            "llama2 attention is not gqa head_dim")
    two = replace(m, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_llama2_vectors(two, c_ok),
            "llama2 final norm is not one")
    fp32 = replace(m, act_fmt="fp32")
    _raises(lambda: _tag_llama2_vectors(fp32, c_ok),
            "llama2 activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("bf16", 16.0) if k == "mlp" else v) for k, v in m.formats)
    wide_w = replace(_bare(m), formats=fmts)
    _raises(lambda: _tag_llama2_vectors(wide_w, c_ok),
            "llama2 weight dtype is not fp16")
    yi = get_model("yi-1.5-34b")
    yi_c = _tc(load_release(yi.hf_id)["config"])
    yi_fmts = tuple((k, RoleFormat("fp16", 16.0) if k == "attn" else v) for k, v in yi.formats)
    _raises(lambda: _tag_llama2_vectors(replace(_bare(yi), formats=yi_fmts), yi_c),
            "llama2 weight dtype is not bf16")
    carbon = get_model("carbon-8b")
    crel = json.loads(json.dumps(load_release(carbon.hf_id)))
    del crel["config"]["attention_bias"]
    _raises(lambda: from_release("carbon-8b", carbon.hf_id, crel),
            "carbon attention_bias is not false")
    crel2 = json.loads(json.dumps(load_release(carbon.hf_id)))
    del crel2["config"]["pretraining_tp"]
    _raises(lambda: from_release("carbon-8b", carbon.hf_id, crel2),
            "carbon pretraining_tp is not 1")
    crel3 = json.loads(json.dumps(load_release(carbon.hf_id)))
    del crel3["config"]["mlp_bias"]
    _raises(lambda: from_release("carbon-8b", carbon.hf_id, crel3),
            "carbon mlp_bias is not false")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, _hf, h, inner, depth, *_rest in _ROWS:
        expect = _ffn(depth, h, inner, t)
        m = get_model(mid)
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == expect


def _fpx(top):
    h = hashlib.sha256()
    for e in entries()[:60]:
        if e["id"].startswith("deepseek-v4"):
            continue
        for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}},
                      {"layout": {"tp": 2}}):
            try:
                h.update(json.dumps(evaluate(Scenario.from_dict({"model": e["id"], **top, **extra})).summary(),
                                    sort_keys=True, default=str).encode())
            except Exception as ex:  # noqa: BLE001
                h.update(type(ex).__name__.encode())
    return h.hexdigest()[:16]


def test_fingerprints_and_kpi():
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651, _kpi
    # LPDDR decode ticks stay bit-identical. The pipe hash covers the full
    # summary, so the default pipe and the no-DeepSeek-V4 pipe move.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "555d25ec2cbe1bfe"
    assert _fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e"
    assert _fpx({}) == "666d0c5f91fbb6a4"
    from accel_dse.core import evaluate as E
    with E.model_flags(FUSE_QKV=False):
        assert _kpi("yi-1.5-34b") == "4a9848721f34"
        assert _kpi("qwen2.5-72b") == "51b90d8f5e68"
        assert _kpi("mistral-large") == "4ce7bca4c910"
    assert _FULL_0651["yi-1.5-34b"] == "4a9848721f34"
    assert _FULL_0651["qwen2.5-72b"] == "51b90d8f5e68"
    assert _FULL_0651["mistral-large"] == "4ce7bca4c910"
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
    assert "llama-2-70b" not in _FULL_0651 and "llama-2-7b" not in _FULL_0651
    assert _kpi("esmfold") == "3bbede544308"
    assert _kpi("esm2-3b") == "34f7544287cd"
    assert _kpi("wan2.1-1.3b") == "309919c00604"
