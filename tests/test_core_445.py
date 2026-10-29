"""0.445: Carbon causal siblings bill published RMSNorm and the SwiGLU product.

LlamaRMSNorm and Qwen3RMSNorm are the T5 recipe in fp32. SiLU stays the
existing act op. The gate times up product is mlp_mul. Carbon-X also applies
that RMSNorm on head_dim (one rsqrt per head). Other Llama and Qwen3 rows
stay on the shared tail. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import from_release, load_release


_LLAMA = {
    "carbon-8b": ("HuggingFaceBio/Carbon-8B", 4096, 14336, 32, 32, 8, 128, 8_255_705_088),
    "carbon-3b": ("HuggingFaceBio/Carbon-3B", 3072, 8448, 30, 32, 4, 96, 3_451_444_224),
    "carbon-500m": ("HuggingFaceBio/Carbon-500M", 1024, 3072, 28, 16, 8, 64, 511_894_528),
}
_X = ("HuggingFaceBio/Carbon-X", 5120, 17408, 40, 40, 8, 128, 14_822_046_720)
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


def test_version_is_445():
    assert __version__ == "0.469.0"


def test_llama_carbons_bill_fp32_rmsnorm_and_the_product():
    for mid, (hf, h, inner, depth, _hq, _hkv, hd, params) in _LLAMA.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.vec_kind == "carbon_llama"
        assert all(L.vec_kind == "carbon_llama" for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
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
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert "q_rms" not in names and "k_rms" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, 1, depth)


def test_carbon_x_bills_qk_rmsnorm_on_head_dim():
    hf, h, inner, depth, hq, hkv, hd, params = _X
    m = from_release("carbon-x", hf)
    assert m.params() == m.release_params == params
    assert m.param_check()["rel_err"] == 0.0
    assert m.vec_kind == "carbon_qwen" and m.n_layers == depth and m.hidden == h
    assert all(L.vec_kind == "carbon_qwen" and L.core.qk_dim == hd for L in m.layers)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _rms(by["q_rms"], t, hq * hd, hq, 0)
    _rms(by["k_rms"], t, hkv * hd, hkv, 0)
    assert by["q_rms"].replicated == 1 and by["k_rms"].replicated == 1
    _act(by["act"], t * inner, 0, "bf16")
    _mul(by["mlp_mul"], t * inner, 0, "bf16")
    _rms(by["attn_norm"], t, h, 1, 0)
    _rms(by["ffn_norm"], t, h, 1, 0)
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, h, 2 * inner)
    assert (down.m, down.k, down.n) == (t, inner, h)
    assert up.flops == 2 * t * h * (2 * inner) and down.flops == 2 * t * inner * h
    assert up.w_fmt == "bf16" and up.w_bits == 16.0
    names = [o.name for o in ops]
    assert names.index("attn.qkv") < names.index("q_rms") < names.index("k_rms") < names.index("qk")
    assert names.index("qk") < names.index("rope")
    assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
    assert "attn.q" not in names and "attn.k" not in names
    assert all(o.name not in _BANNED for o in ops)
    last = build_rank_ops(m, depth - 1, ph, Shard())
    lby = _by(last)
    _rms(lby["q_rms"], t, hq * hd, hq, depth - 1)
    _rms(lby["k_rms"], t, hkv * hd, hkv, depth - 1)
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, h, 1, depth)


def test_decode_batch_tp_and_fp8():
    m8 = get_model("carbon-8b")
    h, inner = 4096, 14336
    dec = _by(build_rank_ops(m8, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    assert "q_rms" not in dec
    two = _by(build_rank_ops(m8, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    pre = _by(build_rank_ops(m8, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * inner, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m8, 0, Phase("prefill", 1, 4096, 0), sh))
    half = inner // 2
    _act(loc["act"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == half
    assert loc["ffn_norm"].vec == 4096 * h * 4 and loc["mlp_mul"].replicated == 1
    names = [o.name for o in build_rank_ops(m8, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    m3 = get_model("carbon-3b")
    loc3 = _by(build_rank_ops(m3, 0, Phase("prefill", 1, 64, 0), Shard(tp=2)))
    _mul(loc3["mlp_mul"], 64 * (8448 // 2), 0, "bf16")
    m5 = get_model("carbon-500m")
    loc5 = _by(build_rank_ops(m5, 0, Phase("prefill", 1, 64, 0), Shard(tp=2)))
    _mul(loc5["mlp_mul"], 64 * (3072 // 2), 0, "bf16")
    fp8 = replace(m8, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    mx = get_model("carbon-x")
    xt = _by(build_rank_ops(mx, 0, Phase("prefill", 1, 4096, 0), Shard(tp=2)))
    _rms(xt["q_rms"], 4096, 20 * 128, 20, 0)
    _rms(xt["k_rms"], 4096, 4 * 128, 4, 0)
    _mul(xt["mlp_mul"], 4096 * (17408 // 2), 0, "bf16")
    assert xt["q_rms"].replicated == 1 and xt["k_rms"].replicated == 1
    xfp = replace(mx, act_fmt="fp8")
    xop = _by(build_rank_ops(xfp, 0, Phase("decode", 1, 1, 128), Shard()))
    _act(xop["act"], 17408, 0, "fp8")
    _mul(xop["mlp_mul"], 17408, 0, "fp8")
    assert xop["q_rms"].a_fmt == "fp32" and xop["k_rms"].a_fmt == "fp32"
    assert xop["attn_norm"].a_fmt == "fp32" and xop["mlp.down"].w_fmt == "bf16"


def test_non_vector_schedule_and_other_models():
    m = get_model("carbon-x")
    ph = Phase("prefill", 1, 256, 0)
    bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
    for shard in (Shard(), Shard(tp=2)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("llama-3.1-8b",):
        other = get_model(mid)
        assert other.vec_kind == ""
        assert all(L.vec_kind == "" for L in other.layers)
        ops = build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())
        names = [o.name for o in ops]
        assert "mlp_mul" not in names and "q_rms" not in names and "k_rms" not in names
        by = _by(ops)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        assert by["act"].a_fmt == "bf16"
        head = _by(head_ops(other, Phase("prefill", 1, 16, 0), Shard()))
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    for mid in ("evo2-7b", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert other.vec_kind not in ("carbon_llama", "carbon_qwen")
        assert all(L.vec_kind not in ("carbon_llama", "carbon_qwen") for L in other.layers)
    a = get_model("carbon-a-1.2b")
    assert a.vec_kind == "generanno"
    anames = [o.name for o in build_rank_ops(a, 0, Phase("full", 1, 128, 0), Shard())]
    assert "silu" in anames and "act" not in anames
    assert "q_rms" not in anames and "k_rms" not in anames
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in ("carbon-8b", "carbon-3b", "carbon-500m", "carbon-x"):
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "19cac027be730bc4"
    assert "carbon-8b" not in _FULL_0651 and "carbon-x" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf8, mid8 = "HuggingFaceBio/Carbon-8B", "carbon-8b"
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("hidden_act", "gelu")), "carbon hidden_act is not silu")
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("model_type", "qwen2")), "carbon llama model_type is not llama")
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("attention_bias", True)), "carbon attention_bias is not false")
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("mlp_bias", True)), "carbon mlp_bias is not false")
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("pretraining_tp", 2)), "carbon pretraining_tp is not 1")
    _raises(_bad(hf8, mid8, lambda c: c.__setitem__("rope_scaling", {})), "carbon rope_scaling is not null")
    hfx = "HuggingFaceBio/Carbon-X"
    _raises(_bad(hfx, "carbon-x", lambda c: c.__setitem__("model_type", "llama")), "carbon-x model_type is not qwen3")
    _raises(_bad(hfx, "carbon-x", lambda c: c.__setitem__("hidden_act", "gelu_new")), "carbon hidden_act is not silu")
    _raises(_bad(hfx, "carbon-x", lambda c: c.__setitem__("mlp_bias", True)), "carbon-x mlp_bias is not false")
    _raises(_bad(hfx, "carbon-x", lambda c: c["rope_parameters"].__setitem__("rope_type", "yarn")),
            "carbon-x rope_type is not default")
    _raises(_bad(hfx, "carbon-x", lambda c: c["layer_types"].__setitem__(0, "sliding_attention")),
            "carbon-x layer_types is not full_attention")
    _raises(_bad(hfx, "carbon-x", lambda c: c.__delitem__("head_dim")), "carbon-x head_dim is missing")
    m = get_model("carbon-8b")
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    x = get_model("carbon-x")
    lins = tuple(l for l in x.layers[0].attn_linears if l.name != "q")
    broken_q = replace(x, layers=(replace(x.layers[0], attn_linears=lins),) + x.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "carbon-x attention is not q then k")


def test_ffn_gemm_flops_match_the_closed_form():
    m = get_model("carbon-x")
    t, h, inner = 4096, 5120, 17408
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
