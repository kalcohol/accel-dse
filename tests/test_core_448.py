"""0.448: Granite-4.1 / 4.2 dense LMs bill published RMSNorm and the SwiGLU product.

GraniteRMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. There is no q/k norm. The hidden
norms stay pre-norm. Residual, attention, embedding, and logits scalars
are not 0.5 and stay on the existing residual add. Speech rows stay
untagged. These eighteen ids are outside the first 60 catalog rows.
Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_granite_vectors, _tc, from_release, load_release


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, tie, act
_ROWS = {
    "granite-4.1-30b": ("ibm-granite/granite-4.1-30b", 4096, 32768, 64, 32, 8, 128, 28_865_728_512, True, "bf16"),
    "granite-4.1-30b-fp8": ("ibm-granite/granite-4.1-30b-fp8", 4096, 32768, 64, 32, 8, 128, 28_865_728_512, True, "fp8"),
    "granite-4.1-8b": ("ibm-granite/granite-4.1-8b", 4096, 12800, 40, 32, 8, 128, 8_380_551_168, True, "bf16"),
    "granite-4.1-8b-fp8": ("ibm-granite/granite-4.1-8b-fp8", 4096, 12800, 40, 32, 8, 128, 8_380_551_168, True, "fp8"),
    "granite-4.1-3b": ("ibm-granite/granite-4.1-3b", 2560, 8192, 40, 40, 8, 64, 3_402_836_480, True, "bf16"),
    "granite-4.1-3b-fp8": ("ibm-granite/granite-4.1-3b-fp8", 2560, 8192, 40, 40, 8, 64, 3_402_836_480, True, "fp8"),
    "granite-4.2-30b": ("ibm-granite/granite-4.2-30b", 4096, 32768, 64, 32, 8, 128, 29_276_770_304, False, "bf16"),
    "granite-4.2-30b-fp8": ("ibm-granite/granite-4.2-30b-fp8", 4096, 32768, 64, 32, 8, 128, 29_276_770_304, False, "fp8"),
    "granite-4.2-8b": ("ibm-granite/granite-4.2-8b", 4096, 12800, 40, 32, 8, 128, 8_791_592_960, False, "bf16"),
    "granite-4.2-8b-fp8": ("ibm-granite/granite-4.2-8b-fp8", 4096, 12800, 40, 32, 8, 128, 8_791_592_960, False, "fp8"),
    "granite-4.2-3b": ("ibm-granite/granite-4.2-3b", 2560, 8192, 40, 40, 8, 64, 3_659_737_600, False, "bf16"),
    "granite-4.2-3b-fp8": ("ibm-granite/granite-4.2-3b-fp8", 2560, 8192, 40, 40, 8, 64, 3_659_737_600, False, "fp8"),
    "granite-4.2-30b-mxfp4": ("ibm-granite/granite-4.2-30b-mxfp4", 4096, 32768, 64, 32, 8, 128, 29_276_770_304, False, "bf16"),
    "granite-4.2-30b-nvfp4": ("ibm-granite/granite-4.2-30b-nvfp4", 4096, 32768, 64, 32, 8, 128, 29_276_770_304, False, "bf16"),
    "granite-4.2-8b-mxfp4": ("ibm-granite/granite-4.2-8b-mxfp4", 4096, 12800, 40, 32, 8, 128, 8_791_592_960, False, "bf16"),
    "granite-4.2-8b-nvfp4": ("ibm-granite/granite-4.2-8b-nvfp4", 4096, 12800, 40, 32, 8, 128, 8_791_592_960, False, "bf16"),
    "granite-4.2-3b-mxfp4": ("ibm-granite/granite-4.2-3b-mxfp4", 2560, 8192, 40, 40, 8, 64, 3_659_737_600, False, "bf16"),
    "granite-4.2-3b-nvfp4": ("ibm-granite/granite-4.2-3b-nvfp4", 2560, 8192, 40, 40, 8, 64, 3_659_737_600, False, "bf16"),
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


def test_version_is_448():
    assert __version__ == "0.457.0"


def test_granite_bills_fp32_rmsnorm_and_the_product():
    for mid, (hf, h, inner, depth, hq, hkv, hd, params, tie, af) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == af
        assert m.tie_embeddings is tie
        assert m.vec_kind == "granite"
        assert all(L.vec_kind == "granite" for L in m.layers)
        assert all(L.misc_params == 2 * h for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert hq * hd == h
        cfg = load_release(hf)["config"]
        assert cfg["residual_multiplier"] != 0.5
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "granite"
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, af)
        _mul(by["mlp_mul"], t * inner, 0, af)
        _rms(by["attn_norm"], t, h, 0)
        _rms(by["ffn_norm"], t, h, 0)
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv")
        assert names.index("qk") < names.index("rope")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert names.index("mlp.down") < names.index("residual")
        assert "q_rms" not in names and "k_rms" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = _bare(m)
        assert "mlp_mul" not in [o.name for o in build_rank_ops(bare, 0, ph, Shard())]
        assert by["residual"].vec == _by(build_rank_ops(bare, 0, ph, Shard()))["residual"].vec
        assert _by(build_rank_ops(bare, 0, ph, Shard()))["attn_norm"].a_fmt == "bf16"
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, depth)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        up = by["mlp.gate_up"]
        if mid.endswith("-mxfp4"):
            assert up.w_fmt == "mxfp4" and up.a_fmt == "mxfp4" and up.w_bits == 4.25
        elif mid.endswith("-nvfp4"):
            assert up.w_fmt == "nvfp4" and up.a_fmt == "nvfp4"
            assert up.w_bits != 4.5 and 4.5 < up.w_bits < 4.51
        elif mid.endswith("-fp8"):
            assert up.w_fmt == "fp8" and up.a_fmt == "fp8" and up.w_bits != 8.0
        else:
            assert up.w_fmt == "bf16" and up.a_fmt == "bf16" and up.w_bits == 16.0
        assert any("0.448" in n for n in m.notes)


def test_default_window_pre_norm_order():
    specs = {
        "granite-4.1-30b": (4096, 32768, 64, 32, 8, 128),
        "granite-4.2-30b": (4096, 32768, 64, 32, 8, 128),
        "granite-4.1-8b": (4096, 12800, 40, 32, 8, 128),
        "granite-4.1-3b": (2560, 8192, 40, 40, 8, 64),
    }
    for mid, (h, inner, depth, hq, _hkv, hd) in specs.items():
        m = get_model(mid)
        assert hq * hd == h
        t = 4096
        ph = Phase("prefill", 1, t, 0)
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
        assert up.w_fmt == "bf16" and up.w_bits == 16.0
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("qk") < names.index("rope") < names.index("ffn_norm")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        assert "q_rms" not in names and "k_rms" not in names
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, depth)
        two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
        _rms(two["final_norm"], 2 * t, h, depth)


def test_decode_batch_tp_fp8_and_four_bit():
    m = get_model("granite-4.1-30b")
    h, inner = 4096, 32768
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
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn_norm") < names.index("attn.qkv")
    assert names.index("attn.o") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert names.index("ffn_norm") < names.index("mlp.gate_up")
    m8 = get_model("granite-4.1-8b")
    loc8 = _by(build_rank_ops(m8, 0, Phase("prefill", 1, 64, 0), sh))
    _mul(loc8["mlp_mul"], 64 * (12800 // 2), 0, "bf16")
    assert loc8["mlp.gate_up"].n == 12800 and loc8["mlp.down"].k == 6400
    m3 = get_model("granite-4.1-3b")
    loc3 = _by(build_rank_ops(m3, 0, Phase("prefill", 1, 64, 0), sh))
    _mul(loc3["mlp_mul"], 64 * (8192 // 2), 0, "bf16")
    assert loc3["mlp.gate_up"].n == 8192 and loc3["mlp.down"].k == 4096
    assert loc3["ffn_norm"].vec == 64 * 2560 * 4
    assert "q_rms" not in loc3
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0
    pub = get_model("granite-4.1-30b-fp8")
    got = _by(build_rank_ops(pub, 0, Phase("prefill", 1, 8, 0), Shard()))
    _act(got["act"], 8 * inner, 0, "fp8")
    _mul(got["mlp_mul"], 8 * inner, 0, "fp8")
    assert got["mlp.gate_up"].w_fmt == "fp8" and got["mlp.gate_up"].a_fmt == "fp8"
    assert got["mlp.gate_up"].w_bits == 8.002766927083334
    assert got["attn.qkv"].w_bits == 8.00390625
    assert got["attn_norm"].a_fmt == "fp32"
    mx = get_model("granite-4.2-8b-mxfp4")
    mxo = _by(build_rank_ops(mx, 0, Phase("prefill", 1, 8, 0), Shard()))
    _act(mxo["act"], 8 * 12800, 0, "bf16")
    _mul(mxo["mlp_mul"], 8 * 12800, 0, "bf16")
    assert mxo["mlp.gate_up"].w_fmt == "mxfp4" and mxo["mlp.gate_up"].a_fmt == "mxfp4"
    assert mxo["mlp.gate_up"].w_bits == 4.25
    nv = get_model("granite-4.2-8b-nvfp4")
    nvo = _by(build_rank_ops(nv, 0, Phase("prefill", 1, 8, 0), Shard()))
    _act(nvo["act"], 8 * 12800, 0, "bf16")
    assert nvo["mlp.gate_up"].w_fmt == "nvfp4" and nvo["mlp.gate_up"].a_fmt == "nvfp4"
    assert nvo["mlp.gate_up"].w_bits == 4.500001220703125
    nv30 = get_model("granite-4.2-30b-nvfp4")
    assert _by(build_rank_ops(nv30, 0, Phase("prefill", 1, 8, 0), Shard()))["mlp.down"].w_bits == 4.500000476837158
    nv3 = get_model("granite-4.2-3b-nvfp4")
    assert _by(build_rank_ops(nv3, 0, Phase("prefill", 1, 8, 0), Shard()))["mlp.down"].w_bits == 4.500003051757813


def test_non_vector_schedule_and_other_models():
    m = get_model("granite-4.1-30b")
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("qwen3-32b", "llama-3.1-8b", "olmo-3-32b-think",
                "qwen3-vl-235b-a22b", "qwen3-vl-30b-a3b"):
        other = get_model(mid)
        assert other.vec_kind == ""
        assert all(L.vec_kind != "granite" for L in other.layers)
    # 0.449 bills qwen3-14b. It is not the Granite pre-norm-only tag.
    billed = get_model("qwen3-14b")
    assert billed.vec_kind == "qwen3"
    assert all(L.vec_kind != "granite" for L in billed.layers)
    for mid in ("granite-speech-5.0-2b", "granite-speech-4.1-2b-plus",
                "granite-speech-5.0-470m-turboctc"):
        other = get_model(mid)
        assert other.vec_kind != "granite"
        assert all(L.vec_kind != "granite" for L in other.layers)
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    plain = get_model("qwen3-32b")
    pops = build_rank_ops(plain, 0, Phase("prefill", 1, 16, 0), Shard())
    pby = _by(pops)
    assert "mlp_mul" not in [o.name for o in pops] and "q_rms" not in [o.name for o in pops]
    assert pby["attn_norm"].a_fmt == "bf16" and pby["ffn_norm"].a_fmt == "bf16"
    head = _by(head_ops(plain, Phase("prefill", 1, 16, 0), Shard()))
    assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    for mid in ("evo2-7b", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert other.vec_kind != "granite"
        assert all(L.vec_kind != "granite" for L in other.layers)
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in _ROWS:
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "1e15ae85430a04f3"
    assert "granite-4.1-30b" not in _FULL_0651
    assert "granite-4.2-8b" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "ibm-granite/granite-4.1-30b", "granite-4.1-30b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "granite hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "granite model_type is not granite")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)), "granite attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "granite attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "granite mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_heads", 33)),
            "granite hidden is not heads times head_dim")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "granite"
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "granite mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "granite mlp is not gate_up then down")
    rel = load_release(hf)
    c = _tc(rel["config"])
    _raises(lambda: _tag_granite_vectors(broken_g, c), "granite mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_granite_vectors(broken_m, c), "granite has no qk norm")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_granite_vectors(broken_h, c), "granite attention is not gqa head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    cases = {
        "granite-4.1-30b": 211_106_232_532_992,
        "granite-4.2-30b": 211_106_232_532_992,
        "granite-4.1-8b": 51_539_607_552_000,
        "granite-4.1-3b": 20_615_843_020_800,
    }
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, hand in cases.items():
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
