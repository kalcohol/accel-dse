"""0.447: EXAONE-4 bills published post-RMSNorm, head RMSNorm, and the SwiGLU product.

Exaone4RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. q_norm and k_norm are that RMSNorm
on head_dim (one rsqrt per head). The two hidden norms sit after attention
and after the FFN, not before the GEMMs. Hybrid full layers stay NoPE.
Vector flops stay 0. These three ids are outside the first 60 catalog rows.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_exaone_vectors, _tc, from_release, load_release


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, tie, nope layer or None
_ROWS = {
    "exaone-4.0-32b": ("LGAI-EXAONE/EXAONE-4.0-32B", 5120, 27392, 64, 40, 8, 128, 32_003_216_384, False, 3),
    "exaone-4.0.1-32b": ("LGAI-EXAONE/EXAONE-4.0.1-32B", 5120, 27392, 64, 40, 8, 128, 32_003_216_384, False, 3),
    "exaone-4.0-1.2b": ("LGAI-EXAONE/EXAONE-4.0-1.2B", 2048, 4096, 30, 32, 8, 64, 1_279_391_488, True, None),
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


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_447():
    assert __version__ == "0.473.0"


def test_post_rmsnorm_qk_and_the_product():
    for mid, (hf, h, inner, depth, hq, hkv, hd, params, tie, nope) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is tie and m.vec_kind == "exaone4"
        assert m.final_norm_params is None
        assert all(L.vec_kind == "exaone4" for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        qn = hq * hd
        kn = hkv * hd
        assert qn == h
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _rms(by["q_rms"], t, qn, hq, 0)
        _rms(by["k_rms"], t, kn, hkv, 0)
        assert by["q_rms"].replicated == 1 and by["k_rms"].replicated == 1
        assert by["q_rms"].vec == by["attn_norm"].vec
        assert by["q_rms"].sfu == float(t * hq) and by["q_rms"].sfu != by["attn_norm"].sfu
        _act(by["act"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
        _rms(by["attn_norm"], t, h, 1, 0)
        _rms(by["ffn_norm"], t, h, 1, 0)
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, h, 2 * inner)
        assert (down.m, down.k, down.n) == (t, inner, h)
        assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
        assert up.flops == 2 * t * h * (2 * inner)
        assert down.flops == 2 * t * inner * h
        names = [o.name for o in ops]
        assert names.index("attn.qkv") < names.index("attn.o") < names.index("q_rms")
        assert names.index("q_rms") < names.index("k_rms") < names.index("qk")
        assert names.index("qk") < names.index("rope") < names.index("attn_norm")
        assert names.index("attn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert names.index("mlp.down") < names.index("ffn_norm") < names.index("residual")
        assert "attn_allreduce" not in names and "mlp_allreduce" not in names
        assert "attn.q" not in names and "attn.k" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
        assert _by(build_rank_ops(m, 0, ph, Shard()))["residual"].vec == \
            _by(build_rank_ops(bare, 0, ph, Shard()))["residual"].vec
        bare_names = [o.name for o in build_rank_ops(bare, 0, ph, Shard())]
        assert bare_names.index("attn_norm") < bare_names.index("attn.qkv")
        assert bare_names.index("ffn_norm") < bare_names.index("mlp.gate_up")
        assert "mlp_mul" not in bare_names and "q_rms" not in bare_names
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _rms(last["q_rms"], t, qn, hq, depth - 1)
        _rms(last["k_rms"], t, kn, hkv, depth - 1)
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, 1, depth)
        note = " ".join(m.notes)
        assert "mlp_mul" in note and "llama3" in note
        if nope is None:
            assert all(not L.core.no_rope for L in m.layers)
            assert "rope" in [o.name for o in build_rank_ops(m, depth - 1, ph, Shard())]
        else:
            assert m.layers[0].core.no_rope is False and m.layers[nope].core.no_rope
            nope_ops = build_rank_ops(m, nope, ph, Shard())
            nope_names = [o.name for o in nope_ops]
            assert "rope" not in nope_names
            nb = _by(nope_ops)
            _rms(nb["q_rms"], t, qn, hq, nope)
            _rms(nb["k_rms"], t, kn, hkv, nope)
            _rms(nb["attn_norm"], t, h, 1, nope)
            _rms(nb["ffn_norm"], t, h, 1, nope)
            assert nope_names.index("qk") < nope_names.index("attn_norm") < nope_names.index("mlp.gate_up")
            assert nope_names.index("mlp.down") < nope_names.index("ffn_norm")
    m32 = get_model("exaone-4.0-32b")
    by = _by(build_rank_ops(m32, 0, Phase("prefill", 1, 4096, 0), Shard()))
    assert by["q_rms"].vec == 4096 * 5120 * 4 and by["q_rms"].sfu == 4096 * 40
    assert by["k_rms"].vec == 4096 * 1024 * 4 and by["k_rms"].sfu == 4096 * 8
    assert by["mlp_mul"].vec == 4096 * 27392
    assert by["act"].vec == 4096 * 27392 * 4
    tiny = get_model("exaone-4.0-1.2b")
    tb = _by(build_rank_ops(tiny, 0, Phase("prefill", 1, 4096, 0), Shard()))
    assert tb["q_rms"].vec == 4096 * 2048 * 4 and tb["q_rms"].sfu == 4096 * 32
    assert tb["k_rms"].vec == 4096 * 512 * 4 and tb["k_rms"].sfu == 4096 * 8
    assert tb["mlp_mul"].vec == 4096 * 4096
    assert tb["attn_norm"].sfu == 4096.0


def test_decode_batch_tp_and_requested_fp8():
    m = get_model("exaone-4.0-32b")
    h, inner, hq, hkv, hd = 5120, 27392, 40, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    assert dec["q_rms"].sfu == float(hq) and dec["attn_norm"].sfu == 1.0
    nope = build_rank_ops(m, 3, Phase("decode", 1, 1, 4096), Shard())
    assert all(o.name != "rope" for o in nope)
    assert _by(nope)["q_rms"].sfu == float(hq)
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
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == inner // 2
    assert loc["q_rms"].replicated == 1 and loc["k_rms"].replicated == 1
    assert loc["ffn_norm"].vec == 4096 * h * 4 and loc["attn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("attn_allreduce") < names.index("attn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce") < names.index("ffn_norm")
    nope_tp = [o.name for o in build_rank_ops(m, 3, Phase("prefill", 1, 4096, 0), sh)]
    assert "rope" not in nope_tp
    assert nope_tp.index("attn_allreduce") < nope_tp.index("attn_norm")
    tiny = get_model("exaone-4.0-1.2b")
    loc_t = _by(build_rank_ops(tiny, 0, Phase("prefill", 1, 64, 0), sh))
    _rms(loc_t["q_rms"], 64, 16 * 64, 16, 0)
    _rms(loc_t["k_rms"], 64, 4 * 64, 4, 0)
    _mul(loc_t["mlp_mul"], 64 * (4096 // 2), 0, "bf16")
    assert loc_t["q_rms"].sfu == 64 * 16
    assert loc_t["attn_norm"].vec == 64 * 2048 * 4
    assert "rope" in [o.name for o in build_rank_ops(tiny, 29, Phase("prefill", 1, 64, 0), Shard())]
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


def test_non_vector_schedule_and_other_models():
    m = get_model("exaone-4.0-32b")
    ph = Phase("prefill", 1, 256, 0)
    bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("llama-3.1-8b",
                "qwen3-vl-235b-a22b", "qwen3-vl-30b-a3b",
                "olmo-3-32b-think", "qwen3-coder-480b-a35b"):
        other = get_model(mid)
        assert other.vec_kind == ""
        assert all(L.vec_kind != "exaone4" for L in other.layers)
    # 0.449 bills qwen3-14b. It stays off the EXAONE post-norm path.
    billed = get_model("qwen3-14b")
    assert billed.vec_kind == "qwen3"
    assert all(L.vec_kind != "exaone4" for L in billed.layers)
    # 0.448 bills these two as granite. They stay off the EXAONE post-norm path.
    for mid in ("granite-4.1-30b", "granite-4.2-8b"):
        other = get_model(mid)
        assert other.vec_kind == "granite"
        assert all(L.vec_kind != "exaone4" for L in other.layers)
    carbon = get_model("carbon-x")
    assert carbon.vec_kind == "carbon_qwen"
    qwen = get_model("qwen3-vl-32b")
    assert qwen.vec_kind == "qwen3vl_text"
    qnames = [o.name for o in build_rank_ops(qwen, 0, Phase("prefill", 1, 16, 0), Shard())]
    assert qnames.index("ffn_norm") < qnames.index("mlp.gate_up")
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
        assert other.vec_kind != "exaone4"
        assert all(L.vec_kind != "exaone4" for L in other.layers)
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in _ROWS:
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    assert "exaone-4.0-32b" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "LGAI-EXAONE/EXAONE-4.0-32B", "exaone-4.0-32b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "exaone4 hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "exaone4 model_type is not exaone4")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)), "exaone4 attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "exaone4 attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "exaone4 mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__delitem__("head_dim")), "exaone4 head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 0)), "exaone4 head_dim is not positive")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 64)), "exaone4 hidden is not heads times head_dim")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["softmax"])),
            "exaone4 layer_types is not sliding_attention or full_attention")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "exaone4"
    assert load_release(hf)["config"]["rope_scaling"]["rope_type"] == "llama3"
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "exaone4 mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "exaone4 mlp is not gate_up then down")
    lins = tuple(l for l in m.layers[0].attn_linears if l.name != "q")
    broken_q = replace(m, layers=(replace(m.layers[0], attn_linears=lins),) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "exaone4 attention is not q then k")
    zero = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=0))
    broken_h = replace(m, layers=(zero,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_h, 0, ph, Shard()), "exaone4 head_dim is not positive")
    q = m.layers[0].attn_linears[0]
    assert q.name == "q"
    wide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_w, 0, ph, Shard()),
            "exaone4 qk norm width is not a whole number of heads")
    rel = load_release(hf)
    c = _tc(rel["config"])
    _raises(lambda: _tag_exaone_vectors(broken_w, c), "exaone4 qk width is not heads times head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    m = get_model("exaone-4.0-32b")
    t, h, inner = 4096, 5120, 27392
    ph = Phase("prefill", 1, t, 0)
    hand = m.n_layers * (2 * t * h * (2 * inner) + 2 * t * inner * h)
    got = 0.0
    for li in range(m.n_layers):
        for o in build_rank_ops(m, li, ph, Shard()):
            if o.name in ("mlp.gate_up", "mlp.down"):
                got += o.flops
            if o.kind == "vector":
                assert o.flops == 0
    assert got == hand
    assert hand == 220_589_520_322_560
