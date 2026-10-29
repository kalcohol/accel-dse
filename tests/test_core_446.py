"""0.446: dense Qwen3-VL text bills published RMSNorm and the SwiGLU product.

Qwen3VLTextRMSNorm is the T5 recipe in fp32. SiLU stays the existing act
op. The gate times up product is mlp_mul. q_norm and k_norm are that
RMSNorm on head_dim (one rsqrt per head). MoE text rows stay on the shared
tail. The vision tower is not retagged. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_qwen3vl_vectors, _tc, from_release, load_release
from accel_dse.core.vision import image_grid, vision_ops


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, language params, tie
_ROWS = {
    "qwen3-vl-32b": ("Qwen/Qwen3-VL-32B-Instruct", 5120, 25600, 64, 64, 8, 128, 32_762_123_264, False),
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", 4096, 12288, 36, 32, 8, 128, 8_190_735_360, False),
    "qwen3-vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", 2560, 9728, 36, 32, 8, 128, 4_022_468_096, True),
    "qwen3-vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", 2048, 6144, 28, 16, 8, 128, 1_720_574_976, True),
}
_FP8 = {
    "qwen3-vl-32b-fp8": "qwen3-vl-32b",
    "qwen3-vl-8b-fp8": "qwen3-vl-8b",
    "qwen3-vl-4b-fp8": "qwen3-vl-4b",
    "qwen3-vl-2b-fp8": "qwen3-vl-2b",
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
    edit(rel["config"]["text_config"])
    return lambda: from_release(mid, hf, rel)


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_446():
    assert __version__ == "0.470.0"


def test_dense_text_bills_fp32_rmsnorm_qk_and_the_product():
    for mid, (hf, h, inner, depth, hq, hkv, hd, params, tie) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is tie and m.vec_kind == "qwen3vl_text"
        assert m.final_norm_params is None
        assert all(L.vec_kind == "qwen3vl_text" for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        qn = hq * hd
        kn = hkv * hd
        assert qn != h or mid in ("qwen3-vl-8b", "qwen3-vl-2b")
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _rms(by["q_rms"], t, qn, hq, 0)
        _rms(by["k_rms"], t, kn, hkv, 0)
        assert by["q_rms"].replicated == 1 and by["k_rms"].replicated == 1
        if qn != h:
            assert by["q_rms"].vec != t * h * 4
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
        assert names.index("q_rms") < names.index("k_rms") < names.index("qk") < names.index("rope")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert "attn.q" not in names and "attn.k" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
        assert _by(build_rank_ops(m, 0, ph, Shard()))["residual"].vec == \
            _by(build_rank_ops(bare, 0, ph, Shard()))["residual"].vec
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _rms(last["q_rms"], t, qn, hq, depth - 1)
        _rms(last["k_rms"], t, kn, hkv, depth - 1)
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, 1, depth)
    m32 = get_model("qwen3-vl-32b")
    by = _by(build_rank_ops(m32, 0, Phase("prefill", 1, 4096, 0), Shard()))
    assert by["q_rms"].vec == 4096 * 8192 * 4 and by["q_rms"].sfu == 4096 * 64
    assert by["k_rms"].vec == 4096 * 1024 * 4 and by["k_rms"].sfu == 4096 * 8
    assert by["mlp_mul"].vec == 4096 * 25600
    assert by["act"].vec == 4096 * 25600 * 4


def test_decode_batch_tp_and_released_fp8():
    m = get_model("qwen3-vl-32b")
    h, inner, hq, hkv, hd = 5120, 25600, 64, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    _rms(two["q_rms"], 2, hq * hd, hq, 0)
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * inner, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    _rms(loc["q_rms"], 4096, 32 * hd, 32, 0)
    _rms(loc["k_rms"], 4096, 4 * hd, 4, 0)
    _mul(loc["mlp_mul"], 4096 * (inner // 2), 0, "bf16")
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == inner // 2
    assert loc["q_rms"].replicated == 1 and loc["k_rms"].replicated == 1
    assert loc["ffn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    m8 = get_model("qwen3-vl-8b")
    loc8 = _by(build_rank_ops(m8, 0, Phase("prefill", 1, 64, 0), sh))
    _rms(loc8["q_rms"], 64, 16 * 128, 16, 0)
    _rms(loc8["k_rms"], 64, 4 * 128, 4, 0)
    _mul(loc8["mlp_mul"], 64 * (12288 // 2), 0, "bf16")
    assert loc8["mlp.gate_up"].n == 12288
    m4 = get_model("qwen3-vl-4b")
    loc4 = _by(build_rank_ops(m4, 0, Phase("prefill", 1, 64, 0), sh))
    _rms(loc4["q_rms"], 64, 16 * 128, 16, 0)
    assert loc4["q_rms"].vec != 64 * 2560 * 4
    _mul(loc4["mlp_mul"], 64 * (9728 // 2), 0, "bf16")
    m2 = get_model("qwen3-vl-2b")
    loc2 = _by(build_rank_ops(m2, 0, Phase("prefill", 1, 64, 0), sh))
    _rms(loc2["q_rms"], 64, 8 * 128, 8, 0)
    _rms(loc2["k_rms"], 64, 4 * 128, 4, 0)
    _mul(loc2["mlp_mul"], 64 * (6144 // 2), 0, "bf16")
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["q_rms"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    for fid, base in _FP8.items():
        released = get_model(fid)
        _h, inn, depth, nh, nkv, hdim, params, tie = _ROWS[base][1:]
        assert released.act_fmt == "fp8" and released.vec_kind == "qwen3vl_text"
        assert released.params() == params and released.param_check()["rel_err"] == 0.0
        assert released.tie_embeddings is tie and released.n_layers == depth
        got = _by(build_rank_ops(released, 0, Phase("prefill", 1, 16, 0), Shard()))
        _act(got["act"], 16 * inn, 0, "fp8")
        _mul(got["mlp_mul"], 16 * inn, 0, "fp8")
        _rms(got["q_rms"], 16, nh * hdim, nh, 0)
        _rms(got["k_rms"], 16, nkv * hdim, nkv, 0)
        assert got["attn_norm"].a_fmt == "fp32" and got["ffn_norm"].a_fmt == "fp32"
        assert got["mlp.gate_up"].w_fmt == "fp8" and got["mlp.gate_up"].a_fmt == "fp8"
        assert round(got["mlp.gate_up"].w_bits, 6) == 8.001953
        assert got["mlp.down"].w_fmt == "fp8"
        head = _by(head_ops(released, Phase("prefill", 1, 16, 0), Shard()))
        assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 16.0


def test_non_vector_schedule_vision_and_other_models():
    m = get_model("qwen3-vl-32b")
    ph = Phase("prefill", 1, 256, 0)
    bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
    for shard in (Shard(), Shard(tp=2)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("qwen3-vl-235b-a22b", "qwen3-vl-235b-a22b-fp8",
                "qwen3-vl-30b-a3b", "qwen3-vl-30b-a3b-fp8",
                "llama-3.1-8b"):
        other = get_model(mid)
        assert other.vec_kind == ""
        assert all(L.vec_kind == "" for L in other.layers)
        ops = build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())
        names = [o.name for o in ops]
        assert "mlp_mul" not in names and "q_rms" not in names and "k_rms" not in names
        by = _by(ops)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        if "act" in by:
            assert by["act"].a_fmt == "bf16"
        else:
            assert "expert_act" in names
        head = _by(head_ops(other, Phase("prefill", 1, 16, 0), Shard()))
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    carbon = get_model("carbon-x")
    assert carbon.vec_kind == "carbon_qwen"
    assert all(L.vec_kind == "carbon_qwen" for L in carbon.layers)
    for mid in ("evo2-7b", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert other.vec_kind != "qwen3vl_text"
        assert all(L.vec_kind != "qwen3vl_text" for L in other.layers)
    m2 = get_model("qwen3-vl-2b")
    assert m2.vision is not None and m2.vision.ln_mode == "qwen3vl"
    vnames = [o.name for o in vision_ops(m2.vision, image_grid(m2.vision, 224, 224), 1)]
    assert "mlp_mul" not in vnames and "q_rms" not in vnames and "k_rms" not in vnames
    lang = [o.name for o in build_rank_ops(m2, 0, Phase("prefill", 1, 16, 0), Shard())]
    assert "mlp_mul" in lang and "q_rms" in lang
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in list(_ROWS) + list(_FP8):
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "19cac027be730bc4"
    assert "qwen3-vl-32b" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "Qwen/Qwen3-VL-32B-Instruct", "qwen3-vl-32b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3-vl hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "qwen3")), "qwen3-vl model_type is not qwen3_vl_text")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)), "qwen3-vl attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "qwen3-vl attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen3-vl mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__delitem__("head_dim")), "qwen3-vl head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 0)), "qwen3-vl head_dim is not positive")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3-vl layer_types is not full_attention")
    m = get_model("qwen3-vl-32b")
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen3-vl mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "qwen3-vl mlp is not gate_up then down")
    lins = tuple(l for l in m.layers[0].attn_linears if l.name != "q")
    broken_q = replace(m, layers=(replace(m.layers[0], attn_linears=lins),) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "qwen3-vl attention is not q then k")
    zero = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=0))
    broken_h = replace(m, layers=(zero,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_h, 0, ph, Shard()), "qwen3-vl head_dim is not positive")
    q = m.layers[0].attn_linears[0]
    wide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_w, 0, ph, Shard()),
            "qwen3-vl qk norm width is not a whole number of heads")
    rel = load_release(hf)
    c = _tc(rel["config"])
    _raises(lambda: _tag_qwen3vl_vectors(broken_w, c), "qwen3-vl qk width is not heads times head_dim")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "qwen3vl_text"
    assert "mrope" not in " ".join(loaded.notes)


def test_ffn_gemm_flops_match_the_closed_form():
    m = get_model("qwen3-vl-32b")
    t, h, inner = 4096, 5120, 25600
    ph = Phase("prefill", 1, t, 0)
    hand = m.n_layers * (2 * t * h * (2 * inner) + 2 * t * inner * h)
    got = 0.0
    for li in range(m.n_layers):
        for o in build_rank_ops(m, li, ph, Shard()):
            if o.name in ("mlp.gate_up", "mlp.down"):
                got += o.flops
            if o.kind == "vector":
                assert o.flops == 0
    assert got == hand == 206_158_430_208_000
