"""0.449: Qwen3-14B dense LMs bill published RMSNorm and the SwiGLU product.

Qwen3RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. q_norm and k_norm are that RMSNorm
on head_dim (one rsqrt per head). q width equals hidden, so q_rms vec
matches attn_norm and sfu stays the head count. Hidden norms stay
pre-norm. These two ids are outside the first 60 catalog rows. Front-60
Qwen3 rows stay on the shared tail. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_qwen3_vectors, _tc, from_release, load_release


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, tie, act
_ROWS = {
    "qwen3-14b": ("Qwen/Qwen3-14B", 5120, 17408, 40, 40, 8, 128, 14_768_307_200, False, "bf16"),
    "qwen3-14b-fp8": ("Qwen/Qwen3-14B-FP8", 5120, 17408, 40, 40, 8, 128, 14_768_307_200, False, "fp8"),
}
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")


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
    return replace(model, layers=tuple(replace(L, vec_kind="") for L in model.layers), vec_kind="")


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_449():
    assert __version__ == "0.468.0"


def test_qwen3_14b_bills_fp32_rmsnorm_qk_and_the_product():
    for mid, (hf, h, inner, depth, hq, hkv, hd, params, tie, af) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == af
        assert m.tie_embeddings is tie
        assert m.vec_kind == "qwen3"
        assert all(L.vec_kind == "qwen3" for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert hq * hd == h
        cfg = load_release(hf)["config"]
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "qwen3"
        assert cfg["rope_scaling"] is None and cfg["sliding_window"] is None
        assert cfg["use_sliding_window"] is False
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, af)
        _mul(by["mlp_mul"], t * inner, 0, af)
        _rms(by["q_rms"], t, hq * hd, hq, 0)
        _rms(by["k_rms"], t, hkv * hd, hkv, 0)
        _rms(by["attn_norm"], t, h, 1, 0)
        _rms(by["ffn_norm"], t, h, 1, 0)
        assert by["q_rms"].vec == by["attn_norm"].vec
        assert by["q_rms"].sfu == t * hq and by["attn_norm"].sfu == float(t)
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("attn.o")
        assert names.index("attn.o") < names.index("q_rms") < names.index("k_rms") < names.index("qk")
        assert names.index("qk") < names.index("rope")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert names.index("mlp.down") < names.index("residual")
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = _bare(m)
        assert "mlp_mul" not in [o.name for o in build_rank_ops(bare, 0, ph, Shard())]
        assert "q_rms" not in [o.name for o in build_rank_ops(bare, 0, ph, Shard())]
        assert by["residual"].vec == _by(build_rank_ops(bare, 0, ph, Shard()))["residual"].vec
        assert _by(build_rank_ops(bare, 0, ph, Shard()))["attn_norm"].a_fmt == "bf16"
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, 1, depth)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        up = by["mlp.gate_up"]
        if mid.endswith("-fp8"):
            assert up.w_fmt == "fp8" and up.a_fmt == "fp8"
            assert round(up.w_bits, 6) == 8.000977 and up.w_bits != 8.0
        else:
            assert up.w_fmt == "bf16" and up.a_fmt == "bf16" and up.w_bits == 16.0
        assert any("0.449" in n for n in m.notes)


def test_default_window_pre_norm_order():
    m = get_model("qwen3-14b")
    h, inner, depth, hq, hkv, hd = 5120, 17408, 40, 40, 8, 128
    assert hq * hd == h
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _act(by["act"], t * inner, 0, "bf16")
    _mul(by["mlp_mul"], t * inner, 0, "bf16")
    _rms(by["q_rms"], t, hq * hd, hq, 0)
    _rms(by["k_rms"], t, hkv * hd, hkv, 0)
    _rms(by["attn_norm"], t, h, 1, 0)
    _rms(by["ffn_norm"], t, h, 1, 0)
    assert by["q_rms"].vec == 4096 * 5120 * 4 and by["q_rms"].sfu == 4096 * 40
    assert by["k_rms"].vec == 4096 * 1024 * 4 and by["k_rms"].sfu == 4096 * 8
    assert by["mlp_mul"].vec == 4096 * 17408
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, h, 2 * inner)
    assert (down.m, down.k, down.n) == (t, inner, h)
    assert up.flops == 2 * t * h * (2 * inner)
    assert down.flops == 2 * t * inner * h
    assert up.w_fmt == "bf16" and up.w_bits == 16.0
    names = [o.name for o in ops]
    assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
    assert names.index("q_rms") < names.index("k_rms") < names.index("qk") < names.index("rope")
    assert names.index("rope") < names.index("ffn_norm")
    assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
    last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
    _act(last["act"], t * inner, depth - 1, "bf16")
    _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
    _rms(last["q_rms"], t, hq * hd, hq, depth - 1)
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, h, 1, depth)
    two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
    _rms(two["final_norm"], 2 * t, h, 1, depth)


def test_decode_batch_tp_and_fp8():
    m = get_model("qwen3-14b")
    h, inner, hq, hkv, hd = 5120, 17408, 40, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["attn_norm"].sfu == 1.0
    assert dec["mlp.gate_up"].m == 1
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    _rms(two["q_rms"], 2, hq * hd, hq, 0)
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * inner, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    _rms(loc["q_rms"], 4096, 20 * hd, 20, 0)
    _rms(loc["k_rms"], 4096, 4 * hd, 4, 0)
    _mul(loc["mlp_mul"], 4096 * (inner // 2), 0, "bf16")
    assert loc["q_rms"].replicated == 1 and loc["k_rms"].replicated == 1
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == inner // 2
    assert loc["attn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("q_rms") < names.index("attn_allreduce")
    assert names.index("attn_allreduce") < names.index("ffn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["q_rms"].a_fmt == "fp32" and op["k_rms"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0
    pub = get_model("qwen3-14b-fp8")
    got = _by(build_rank_ops(pub, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(got["act"], 16 * inner, 0, "fp8")
    _mul(got["mlp_mul"], 16 * inner, 0, "fp8")
    _rms(got["q_rms"], 16, hq * hd, hq, 0)
    assert got["attn_norm"].a_fmt == "fp32" and got["mlp.gate_up"].w_fmt == "fp8"
    assert got["mlp.gate_up"].a_fmt == "fp8"
    assert round(got["mlp.gate_up"].w_bits, 6) == 8.000977
    assert round(got["attn.qkv"].w_bits, 6) == 8.000977


def test_non_vector_schedule_and_other_models():
    m = get_model("qwen3-14b")
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("llama-3.1-8b",
                "olmo-3-32b-think", "qwen3-vl-235b-a22b", "qwen3-vl-30b-a3b",
                "qwen3-coder-480b-a35b", "qwen3-tts-1.7b-base",
                "mistral-small-3.2-24b-instruct-2506"):
        other = get_model(mid)
        assert other.vec_kind != "qwen3"
        assert all(L.vec_kind != "qwen3" for L in other.layers)
    # 0.467 bills the head-width mismatch rows on a separate gate.
    assert get_model("qwen3-32b").vec_kind == "qwen3"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    # 0.467 bills qwen3-32b. The shared-tail exemplar is llama-3.1-8b.
    plain = get_model("llama-3.1-8b")
    pops = build_rank_ops(plain, 0, Phase("prefill", 1, 16, 0), Shard())
    pby = _by(pops)
    assert "mlp_mul" not in [o.name for o in pops] and "q_rms" not in [o.name for o in pops]
    assert pby["attn_norm"].a_fmt == "bf16" and pby["ffn_norm"].a_fmt == "bf16"
    head = _by(head_ops(plain, Phase("prefill", 1, 16, 0), Shard()))
    assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    for mid in ("evo2-7b", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert other.vec_kind != "qwen3"
        assert all(L.vec_kind != "qwen3" for L in other.layers)
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in _ROWS:
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "19cac027be730bc4"
    assert "qwen3-14b" not in _FULL_0651
    assert "qwen3-14b-fp8" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "Qwen/Qwen3-14B", "qwen3-14b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3 hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "qwen3 model_type is not qwen3")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)), "qwen3 attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "qwen3 attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen3 mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__delitem__("head_dim")), "qwen3 head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 0)), "qwen3 head_dim is not positive")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_heads", 39)),
            "qwen3 hidden is not heads times head_dim")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen3 rope_scaling is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen3 use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "qwen3 sliding_window is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3 layer_types is not full_attention")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "qwen3"
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen3 mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "qwen3 mlp is not gate_up then down")
    rel = load_release(hf)
    c = _tc(rel["config"])
    _raises(lambda: _tag_qwen3_vectors(broken_g, c), "qwen3 mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_m, c), "qwen3 qk norm is not head_dim")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_h, c), "qwen3 attention is not gqa head_dim")
    lins = tuple(l for l in m.layers[0].attn_linears if l.name != "q")
    broken_q = replace(m, layers=(replace(m.layers[0], attn_linears=lins),) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "qwen3 attention is not q then k")
    zero = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=0))
    broken_z = replace(m, layers=(zero,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_z, 0, ph, Shard()), "qwen3 head_dim is not positive")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_w, 0, ph, Shard()),
            "qwen3 qk norm width is not a whole number of heads")
    _raises(lambda: _tag_qwen3_vectors(broken_w, c), "qwen3 qk width is not heads times head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    hand = 87_617_332_838_400
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid in ("qwen3-14b", "qwen3-14b-fp8"):
        m = get_model(mid)
        h = m.hidden
        inner = m.layers[0].ffn.d_ff
        expect = m.n_layers * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        assert expect == hand
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == hand
