"""0.465: front-60 Qwen3 rows whose hidden equals heads * head_dim bill qwen3.

Qwen3-8B, its fp8 and AWQ releases, and Qwen3-1.7B reuse the 0.449 recipe.
Qwen3RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. q_norm and k_norm are that RMSNorm
on head_dim (one rsqrt per head). q width equals hidden. No new act kind
and no new vec_kind. qwen3-32b, qwen3-4b, and qwen3-0.6b fail this product
gate. 0.467 bills them on a separate gate. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_qwen3_vectors, _tc, from_release, load_release


# id, hf, hidden, inner, depth, q heads, kv heads, head_dim, params, tie, act, w_fmt
_ROWS = (
    ("qwen3-8b", "Qwen/Qwen3-8B", 4096, 12288, 36, 32, 8, 128, 8_190_735_360, False, "bf16", "bf16"),
    ("qwen3-8b-fp8", "Qwen/Qwen3-8B-FP8", 4096, 12288, 36, 32, 8, 128, 8_190_735_360, False, "fp8", "fp8"),
    ("qwen3-8b-awq", "Qwen/Qwen3-8B-AWQ", 4096, 12288, 36, 32, 8, 128, 8_190_735_360, False, "bf16", "int4"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B", 2048, 6144, 28, 16, 8, 128, 1_720_574_976, True, "bf16", "bf16"),
)
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")
_HAND_8B = 44_530_220_924_928
_HAND_17 = 8_658_654_068_736


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


def test_version_is_465():
    assert __version__ == "0.467.0"


def test_four_rows_are_front60_and_reuse_qwen3():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(mid) for mid, *_rest in _ROWS] == [24, 40, 44, 46]
    assert ids[23] == "qwen3-32b" and ids[45] == "qwen3-4b" and ids[47] == "qwen3-0.6b"
    assert ids[49] == "qwen2.5-3b" and ids[50] == "qwen2.5-1.5b"
    for mid, hf, h, inner, depth, hq, hkv, hd, params, tie, af, wf in _ROWS:
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == af
        assert m.tie_embeddings is tie and m.final_norm_params is None
        assert m.vec_kind == "qwen3"
        assert all(L.vec_kind == "qwen3" for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert hq * hd == h
        cfg = load_release(hf)["config"]
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "qwen3"
        assert cfg["architectures"] == ["Qwen3ForCausalLM"]
        assert cfg["rope_scaling"] is None and cfg["sliding_window"] is None
        assert cfg["use_sliding_window"] is False
        assert any("0.465" in n for n in m.notes)
        assert not any("0.449" in n for n in m.notes)
        assert not any("0.453" in n for n in m.notes)
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
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("attn.o")
        assert names.index("attn.o") < names.index("q_rms") < names.index("k_rms") < names.index("qk")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        up = by["mlp.gate_up"]
        assert up.w_fmt == wf and up.a_fmt == af
        if mid.endswith("-fp8"):
            assert round(up.w_bits, 6) == 8.000977 and up.w_bits != 8.0
            qcfg = cfg["quantization_config"]
            assert qcfg["quant_method"] == "fp8" and qcfg["activation_scheme"] == "dynamic"
            assert qcfg["fmt"] == "e4m3" and qcfg["weight_block_size"] == [128, 128]
        elif mid.endswith("-awq"):
            assert up.w_bits == 4.15625
            qcfg = cfg["quantization_config"]
            assert qcfg["quant_method"] == "awq" and qcfg["bits"] == 4 and qcfg["group_size"] == 128
            assert cfg["torch_dtype"] == "float16"
        else:
            assert up.w_bits == 16.0
    small = get_model("qwen3-1.7b")
    assert any("tied lm_head" in n for n in small.notes)
    assert small.notes[-1].startswith("0.465")


def test_default_window_pre_norm_order():
    m = get_model("qwen3-8b")
    h, inner, depth, hq, hkv, hd = 4096, 12288, 36, 32, 8, 128
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
    assert by["q_rms"].vec == 4096 * 4096 * 4 and by["q_rms"].sfu == 4096 * 32
    assert by["k_rms"].vec == 4096 * 1024 * 4 and by["k_rms"].sfu == 4096 * 8
    assert by["mlp_mul"].vec == 4096 * 12288
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, h, 2 * inner)
    assert (down.m, down.k, down.n) == (t, inner, h)
    assert up.flops == 2 * t * h * (2 * inner) and down.flops == 2 * t * inner * h
    assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    names = [o.name for o in ops]
    assert names.index("q_rms") < names.index("k_rms") < names.index("qk") < names.index("rope")
    assert names.index("rope") < names.index("ffn_norm")
    last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
    _act(last["act"], t * inner, depth - 1, "bf16")
    _rms(last["q_rms"], t, hq * hd, hq, depth - 1)
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, h, 1, depth)
    tiny = get_model("qwen3-1.7b")
    top = _by(head_ops(tiny, Phase("prefill", 1, 16, 0), Shard()))
    _rms(top["final_norm"], 16, 2048, 1, 28)
    assert top["final_norm"].vec == 16 * 2048 * 4


def test_decode_batch_tp_and_request_fp8():
    m = get_model("qwen3-8b")
    h, inner, hq, hkv, hd = 4096, 12288, 32, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["attn_norm"].sfu == 1.0 and dec["mlp.gate_up"].m == 1
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * inner, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    _rms(loc["q_rms"], 4096, 16 * hd, 16, 0)
    _rms(loc["k_rms"], 4096, 4 * hd, 4, 0)
    _mul(loc["mlp_mul"], 4096 * (inner // 2), 0, "bf16")
    assert loc["q_rms"].replicated == 1 and loc["k_rms"].replicated == 1
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == inner // 2
    assert loc["attn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("q_rms") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    small = _by(build_rank_ops(get_model("qwen3-1.7b"), 0, Phase("prefill", 1, 4096, 0), sh))
    _rms(small["q_rms"], 4096, 8 * hd, 8, 0)
    _rms(small["k_rms"], 4096, 4 * hd, 4, 0)
    _mul(small["mlp_mul"], 4096 * 3072, 0, "bf16")
    assert small["mlp.down"].k == 3072
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["q_rms"].a_fmt == "fp32" and op["k_rms"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0
    awq = replace(get_model("qwen3-8b-awq"), act_fmt="fp8")
    got = _by(build_rank_ops(awq, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(got["act"], 16 * inner, 0, "fp8")
    _mul(got["mlp_mul"], 16 * inner, 0, "fp8")
    assert got["mlp.gate_up"].w_fmt == "int4" and got["mlp.gate_up"].w_bits == 4.15625
    assert got["mlp.gate_up"].a_fmt == "fp8" and got["attn_norm"].a_fmt == "fp32"
    pub = get_model("qwen3-8b-fp8")
    published = _by(build_rank_ops(pub, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(published["act"], 16 * inner, 0, "fp8")
    assert published["mlp.gate_up"].w_fmt == "fp8" and published["mlp.gate_up"].a_fmt == "fp8"


def test_non_vector_schedule_and_other_rows_stay_untagged():
    ph = Phase("prefill", 1, 256, 0)
    for mid, hf, *_rest in _ROWS:
        m = get_model(mid)
        bare = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(bare, ph, shard)
        c = _tc(load_release(hf)["config"])
        retagged = _tag_qwen3_vectors(bare, c)
        assert retagged.vec_kind == "qwen3"
        assert all(L.vec_kind == "qwen3" for L in retagged.layers)
    # 0.467 bills the head-width mismatch rows. This gate still refuses them.
    for mid, hidden, heads in (("qwen3-32b", 5120, 64), ("qwen3-4b", 2560, 32), ("qwen3-0.6b", 1024, 16)):
        other = get_model(mid)
        assert other.vec_kind == "qwen3"
        assert other.hidden == hidden and other.layers[0].core.n_q == heads
        assert other.hidden != heads * other.layers[0].core.qk_dim
        cfg = _tc(load_release(other.hf_id)["config"])
        assert _tag_qwen3_vectors(_bare(other), cfg).vec_kind == ""
        spoof = replace(_bare(other), id="qwen3-8b")
        _raises(lambda c=cfg, s=spoof: _tag_qwen3_vectors(s, c),
                "qwen3 hidden is not heads times head_dim")
    for mid in ("llama-3.1-8b",):
        other = get_model(mid)
        assert other.vec_kind == ""
        assert "mlp_mul" not in [o.name for o in build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())]
    # 0.466 bills the two smaller Qwen2.5 rows with qwen2_text. This gate does not.
    for mid in ("qwen2.5-3b", "qwen2.5-1.5b"):
        other = get_model(mid)
        assert other.vec_kind == "qwen2_text"
        assert any("0.466" in n for n in other.notes)
        assert not any("0.465" in n for n in other.notes)
    old = get_model("qwen3-14b")
    assert old.vec_kind == "qwen3"
    assert any("0.449" in n for n in old.notes) and not any("0.465" in n for n in old.notes)
    spine = get_model("qwen3-8b-dspine")
    assert spine.vec_kind == "qwen3"
    assert any("0.453" in n for n in spine.notes) and not any("0.465" in n for n in spine.notes)
    assert get_model("qwen2.5-72b").vec_kind == "qwen2_text"
    assert get_model("mistral-large").vec_kind == "mistral_small"
    assert get_model("llama-2-70b").vec_kind == "carbon_llama"
    assert get_model("yi-1.5-34b").vec_kind == "carbon_llama"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "Qwen/Qwen3-8B", "qwen3-8b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3 hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "qwen3 model_type is not qwen3")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)), "qwen3 attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "qwen3 attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen3 mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__delitem__("head_dim")), "qwen3 head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 0)), "qwen3 head_dim is not positive")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_heads", 31)),
            "qwen3 hidden is not heads times head_dim")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen3 rope_scaling is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen3 use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "qwen3 sliding_window is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3 layer_types is not full_attention")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    c = _tc(load_release(hf)["config"])
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen3 mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_g, c), "qwen3 mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_m, c), "qwen3 qk norm is not head_dim")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_h, c), "qwen3 attention is not gqa head_dim")
    lins = tuple(layer for layer in m.layers[0].attn_linears if layer.name != "q")
    broken_q = replace(m, layers=(replace(m.layers[0], attn_linears=lins),) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "qwen3 attention is not q then k")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken_w, c), "qwen3 qk width is not heads times head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, _hf, h, inner, depth, *_rest in _ROWS:
        expect = depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        if h == 4096:
            assert expect == _HAND_8B
        else:
            assert expect == _HAND_17
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
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    from accel_dse.core import evaluate as E
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "7d44cdadbf38cc9e"
    # The no-DeepSeek-V4 pair is locked in 0.125. Re-run that assert.
    test_fingerprints_change_only_v4()
    with E.model_flags(FUSE_QKV=False):
        assert _kpi("qwen3-8b") == "11f6f3a5d21f"
        assert _kpi("qwen3-8b-fp8") == "25b06545a766"
        assert _kpi("qwen3-8b-awq") == "d0c28036e214"
        assert _kpi("qwen3-1.7b") == "c062686f3bf2"
        assert _kpi("mistral-large") == "4ce7bca4c910"
        assert _kpi("qwen2.5-72b") == "51b90d8f5e68"
        assert _kpi("yi-1.5-34b") == "4a9848721f34"
    assert _FULL_0651["qwen3-8b"] == "11f6f3a5d21f"
    assert _FULL_0651["qwen3-8b-fp8"] == "25b06545a766"
    assert _FULL_0651["qwen3-8b-awq"] == "d0c28036e214"
    assert _FULL_0651["qwen3-1.7b"] == "c062686f3bf2"
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
    assert _kpi("esmfold") == "3bbede544308"
    assert _kpi("esm2-3b") == "34f7544287cd"
    assert _kpi("wan2.1-1.3b") == "309919c00604"
