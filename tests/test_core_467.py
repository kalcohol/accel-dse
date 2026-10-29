"""0.467: front-60 Qwen3 rows whose q width is not hidden bill qwen3.

Qwen3-32B, its fp8 and AWQ releases, Qwen3-4B, and Qwen3-0.6B reuse the
0.449 recipe. Qwen3RMSNorm is the T5 recipe in fp32. SiLU stays the
existing act op. The gate times up product is mlp_mul. q_norm and k_norm
are that RMSNorm on head_dim, one rsqrt per head. q width is
num_attention_heads * head_dim and is not hidden. k and v width is
num_key_value_heads * head_dim. The residual width stays hidden_size.
No new act kind and no new vec_kind. The 0.449, 0.453, and 0.465
allowlists are not extended. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _QWEN3_449, _QWEN3_453, _QWEN3_465, _QWEN3_467,
    _tag_qwen3_467_vectors, _tag_qwen3_vectors, _tc, from_release, load_release,
)


# id, hf, hidden, inner, depth, q heads, kv heads, head_dim, params, norm,
# tie, act, w_fmt, ffn gemm sum at prompt 4096
_ROWS = (
    ("qwen3-32b", "Qwen/Qwen3-32B", 5120, 25600, 64, 64, 8, 128,
     32_762_123_264, 676_864, False, "bf16", "bf16", 206_158_430_208_000),
    ("qwen3-32b-fp8", "Qwen/Qwen3-32B-FP8", 5120, 25600, 64, 64, 8, 128,
     32_762_123_264, 676_864, False, "fp8", "fp8", 206_158_430_208_000),
    ("qwen3-32b-awq", "Qwen/Qwen3-32B-AWQ", 5120, 25600, 64, 64, 8, 128,
     32_762_123_264, 676_864, False, "bf16", "int4", 206_158_430_208_000),
    ("qwen3-4b", "Qwen/Qwen3-4B", 2560, 9728, 36, 32, 8, 128,
     4_022_468_096, 196_096, True, "bf16", "bf16", 22_033_182_228_480),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", 1024, 3072, 28, 16, 8, 128,
     596_049_920, 65_536, True, "bf16", "bf16", 2_164_663_517_184),
)
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul")
_ORDER = (
    "attn_norm", "attn.qkv", "attn.o", "q_rms", "k_rms", "qk", "pv", "softmax",
    "rope", "ffn_norm", "mlp.gate_up", "act", "mlp_mul", "mlp.down", "residual",
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


def _rms(op, rows: int, width: int, heads: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows * heads)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(hf: str, mid: str, edit):
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


def test_version_is_467():
    assert __version__ == "0.479.0"


def test_five_rows_are_front60_and_reuse_qwen3():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(row[0]) for row in _ROWS] == [23, 39, 43, 45, 47]
    assert _QWEN3_467 == frozenset(row[0] for row in _ROWS)
    assert _QWEN3_467.isdisjoint(_QWEN3_449)
    assert _QWEN3_467.isdisjoint(_QWEN3_453)
    assert _QWEN3_467.isdisjoint(_QWEN3_465)
    for mid, hf, h, inner, depth, hq, hkv, hd, params, norm, tie, af, wf, _ffn in _ROWS:
        assert ids.index(mid) < 60
        m = from_release(mid, hf)
        assert m.hf_id == hf
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == af
        assert m.tie_embeddings is tie and m.vocab == 151936
        assert m.final_norm_params is None
        assert m.vec_kind == "qwen3"
        assert all(L.vec_kind == "qwen3" for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        assert norm == depth * (2 * h + 2 * hd) + h
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert all(L.core.window is None for L in m.layers)
        qw, kw = hq * hd, hkv * hd
        assert qw != h
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (h, qw)
        assert (by_lin["k"].k, by_lin["k"].n) == (h, kw)
        assert (by_lin["v"].k, by_lin["v"].n) == (h, kw)
        assert (by_lin["o"].k, by_lin["o"].n) == (qw, h)
        assert m.fmt("mlp").fmt == wf and m.fmt("attn").fmt == wf
        cfg = load_release(hf)["config"]
        assert cfg["model_type"] == "qwen3"
        assert cfg["architectures"] == ["Qwen3ForCausalLM"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["attention_bias"] is False
        assert cfg["attention_dropout"] == 0.0
        assert cfg["rope_scaling"] is None and cfg["sliding_window"] is None
        assert cfg["use_sliding_window"] is False
        assert cfg["head_dim"] == hd
        assert cfg["hidden_size"] != cfg["num_attention_heads"] * cfg["head_dim"]
        assert "mlp_bias" not in cfg and "layer_types" not in cfg
        roles = load_release(hf)["roles"]["norm"]["bf16"]
        assert roles["params"] == norm
        assert any(n.startswith("0.467") for n in m.notes)
        assert not any(n.startswith("0.449") or n.startswith("0.453") or n.startswith("0.465")
                       for n in m.notes)
        # Qwen3-4B ties the embedding and does not store a second lm_head.
        # Qwen3-0.6B stores the copy, so the existing de-dup note stays.
        if mid == "qwen3-0.6b":
            assert any("tied lm_head" in n for n in m.notes)
        else:
            assert not any("tied lm_head" in n for n in m.notes)
        bare = _bare(m)
        c = _tc(cfg)
        assert _tag_qwen3_vectors(bare, c).vec_kind == ""
        assert _tag_qwen3_467_vectors(bare, c).vec_kind == "qwen3"
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, af)
        _mul(by["mlp_mul"], t * inner, 0, af)
        _rms(by["q_rms"], t, qw, hq, 0)
        _rms(by["k_rms"], t, kw, hkv, 0)
        _rms(by["attn_norm"], t, h, 1, 0)
        _rms(by["ffn_norm"], t, h, 1, 0)
        assert by["q_rms"].vec != by["attn_norm"].vec
        assert by["q_rms"].vec == t * qw * 4
        names = [o.name for o in ops]
        for banned in _BANNED:
            assert banned not in names
        assert [n for n in _ORDER if n in names] == list(_ORDER)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        up = by["mlp.gate_up"]
        assert up.w_fmt == wf and up.a_fmt == af
        if mid.endswith("-fp8"):
            assert round(up.w_bits, 6) == 8.000977 and up.w_bits != 8.0
            qcfg = cfg["quantization_config"]
            assert qcfg["quant_method"] == "fp8" and qcfg["activation_scheme"] == "dynamic"
            assert qcfg["fmt"] == "e4m3" and qcfg["weight_block_size"] == [128, 128]
            assert cfg["torch_dtype"] == "bfloat16"
        elif mid.endswith("-awq"):
            assert up.w_bits == 4.15625
            qcfg = cfg["quantization_config"]
            assert qcfg["quant_method"] == "awq" and qcfg["bits"] == 4 and qcfg["group_size"] == 128
            assert cfg["torch_dtype"] == "float16"
        else:
            assert up.w_bits == 16.0
            assert cfg["torch_dtype"] == "bfloat16"
    equal = get_model("qwen3-8b")
    assert equal.vec_kind == "qwen3"
    assert any(n.startswith("0.465") for n in equal.notes)
    assert not any(n.startswith("0.467") for n in equal.notes)
    assert equal.layers[0].attn_linears[0].n == equal.hidden
    assert _tag_qwen3_467_vectors(
        _bare(equal), _tc(load_release(equal.hf_id)["config"])).vec_kind == ""


def test_default_window_q_width_is_not_hidden():
    m = get_model("qwen3-32b")
    h, inner, depth, hq, hkv, hd = 5120, 25600, 64, 64, 8, 128
    qw = hq * hd
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _act(by["act"], t * inner, 0, "bf16")
    _mul(by["mlp_mul"], t * inner, 0, "bf16")
    _rms(by["q_rms"], t, qw, hq, 0)
    _rms(by["k_rms"], t, hkv * hd, hkv, 0)
    _rms(by["attn_norm"], t, h, 1, 0)
    _rms(by["ffn_norm"], t, h, 1, 0)
    assert by["q_rms"].vec == 4096 * 8192 * 4
    assert by["k_rms"].vec == 4096 * 1024 * 4
    assert by["attn_norm"].vec == 4096 * 5120 * 4
    assert by["q_rms"].sfu == 4096 * 64 and by["attn_norm"].sfu == 4096
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, h, 2 * inner)
    assert (down.m, down.k, down.n) == (t, inner, h)
    qkv, o = by["attn.qkv"], by["attn.o"]
    assert (qkv.k, qkv.n) == (h, qw + 2 * hkv * hd)
    assert (o.k, o.n) == (qw, h)
    names = [o.name for o in ops]
    assert names.index("attn.o") < names.index("q_rms") < names.index("k_rms") < names.index("qk")
    assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
    assert "gelu" not in names
    last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
    _rms(last["q_rms"], t, qw, hq, depth - 1)
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, h, 1, depth)
    small = get_model("qwen3-0.6b")
    top = _by(build_rank_ops(small, 0, Phase("prefill", 1, 16, 0), Shard()))
    _rms(top["q_rms"], 16, 2048, 16, 0)
    _rms(top["k_rms"], 16, 1024, 8, 0)
    _rms(top["attn_norm"], 16, 1024, 1, 0)
    assert top["q_rms"].vec != top["attn_norm"].vec
    assert top["k_rms"].vec == top["attn_norm"].vec
    four = _by(build_rank_ops(get_model("qwen3-4b"), 0, Phase("prefill", 1, 16, 0), Shard()))
    _rms(four["q_rms"], 16, 4096, 32, 0)
    _rms(four["attn_norm"], 16, 2560, 1, 0)
    assert four["q_rms"].vec != four["attn_norm"].vec
    bare = _bare(m)
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    assert "mlp_mul" not in [o.name for o in bare_ops]
    assert "q_rms" not in [o.name for o in bare_ops]
    assert by["residual"].vec == _by(bare_ops)["residual"].vec
    assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
    assert all(o.flops == 0 for o in ops if o.kind == "vector")


def test_decode_batch_tp_and_request_fp8():
    m = get_model("qwen3-32b")
    h, inner, hq, hkv, hd = 5120, 25600, 64, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["q_rms"].vec != dec["attn_norm"].vec
    assert dec["attn_norm"].sfu == 1.0
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    half = inner // 2
    _act(loc["act"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    _rms(loc["q_rms"], 4096, (hq // 2) * hd, hq // 2, 0)
    _rms(loc["k_rms"], 4096, (hkv // 2) * hd, hkv // 2, 0)
    assert loc["q_rms"].vec == 4096 * 4096 * 4
    assert loc["attn_norm"].vec == 4096 * h * 4
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == half
    assert loc["attn.qkv"].k == h
    assert loc["attn.qkv"].n == (hq // 2) * hd + 2 * (hkv // 2) * hd
    assert loc["attn.o"].k == (hq // 2) * hd and loc["attn.o"].n == h
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("q_rms") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    for tp, mid, heads, kv, width in (
            (4, "qwen3-4b", 32, 8, 9728),
            (8, "qwen3-0.6b", 16, 8, 3072)):
        shard = _by(build_rank_ops(get_model(mid), 0, Phase("prefill", 1, 64, 0), Shard(tp=tp)))
        _rms(shard["q_rms"], 64, (heads // tp) * hd, heads // tp, 0)
        _rms(shard["k_rms"], 64, (kv // tp) * hd, kv // tp, 0)
        assert shard["mlp.down"].k == width // tp
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["q_rms"].a_fmt == "fp32" and op["attn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    pub = get_model("qwen3-32b-fp8")
    published = _by(build_rank_ops(pub, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(published["act"], 16 * inner, 0, "fp8")
    _rms(published["q_rms"], 16, hq * hd, hq, 0)
    assert published["q_rms"].vec != published["attn_norm"].vec
    assert published["mlp.gate_up"].w_fmt == "fp8" and published["mlp.gate_up"].a_fmt == "fp8"
    awq = get_model("qwen3-32b-awq")
    got = _by(build_rank_ops(awq, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(got["act"], 16 * inner, 0, "bf16")
    assert got["mlp.gate_up"].w_fmt == "int4" and got["mlp.gate_up"].w_bits == 4.15625
    assert got["q_rms"].a_fmt == "fp32"


def test_non_vector_schedule_and_other_rows_stay_untagged():
    ph = Phase("prefill", 1, 256, 0)
    for mid, hf, *_rest in _ROWS:
        m = get_model(mid)
        bare = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4), Shard(tp=8)):
            assert _sched(m, ph, shard) == _sched(bare, ph, shard)
        c = _tc(load_release(hf)["config"])
        assert _tag_qwen3_467_vectors(bare, c).vec_kind == "qwen3"
    llama = get_model("llama-3.1-8b")
    assert llama.vec_kind == ""
    assert "mlp_mul" not in [o.name for o in build_rank_ops(llama, 0, Phase("prefill", 1, 16, 0), Shard())]
    spoof = replace(_bare(llama), id="qwen3-32b")
    _raises(lambda: _tag_qwen3_467_vectors(spoof, _tc(load_release(llama.hf_id)["config"])),
            "qwen3 head width model_type is not qwen3")
    equal = get_model("qwen3-8b")
    spoof_eq = replace(_bare(equal), id="qwen3-32b")
    _raises(lambda: _tag_qwen3_467_vectors(spoof_eq, _tc(load_release(equal.hf_id)["config"])),
            "qwen3 head width hidden equals heads times head_dim")
    wide = get_model("qwen3-32b")
    spoof_old = replace(_bare(wide), id="qwen3-8b")
    _raises(lambda: _tag_qwen3_vectors(spoof_old, _tc(load_release(wide.hf_id)["config"])),
            "qwen3 hidden is not heads times head_dim")
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert not any("0.467" in n for n in get_model("qwen3-14b").notes)


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "Qwen/Qwen3-32B", "qwen3-32b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen3 head width hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")),
            "qwen3 head width model_type is not qwen3")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "qwen3 head width architecture is not qwen3")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)),
            "qwen3 head width attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "qwen3 head width attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)),
            "qwen3 head width mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 80)),
            "qwen3 head width hidden equals heads times head_dim")
    _raises(_bad(hf, mid, lambda c: c.pop("head_dim")),
            "qwen3 head width head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen3 head width rope_scaling is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen3 head width use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "qwen3 head width sliding_window is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3 head width layer_types is not full_attention")
    _raises(_bad("Qwen/Qwen3-4B", "qwen3-4b", lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen3 head width hidden_act is not silu")
    m = get_model(mid)
    c_ok = _tc(load_release(hf)["config"])
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_467_vectors(broken_g, c_ok),
            "qwen3 head width mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_467_vectors(broken_m, c_ok),
            "qwen3 head width qk norm is not head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=m.hidden)
    broken_w = replace(m, layers=(replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:]),)
                       + m.layers[1:])
    _raises(lambda: _tag_qwen3_467_vectors(broken_w, c_ok),
            "qwen3 head width qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_467_vectors(broken_win, c_ok),
            "qwen3 head width attention is not gqa head_dim")
    two = replace(m, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_qwen3_467_vectors(two, c_ok),
            "qwen3 head width final norm is not one")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "qwen3"
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()),
            "qwen3 mlp is not gate_up then down")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, _hf, h, inner, depth, *_rest, hand in _ROWS:
        expect = depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        assert expect == hand
        m = get_model(mid)
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == expect


def test_fingerprints_and_kpi_stay():
    from accel_dse.core import evaluate as E
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    test_fingerprints_change_only_v4()
    pins = {
        "qwen3-32b": "b975b7b3760e",
        "qwen3-32b-fp8": "cb91adbc72c9",
        "qwen3-32b-awq": "3e44c63965e3",
        "qwen3-4b": "659c4bad45d8",
        "qwen3-0.6b": "e61d067c924b",
        "qwen3-8b": "11f6f3a5d21f",
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
