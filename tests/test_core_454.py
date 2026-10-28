"""0.454: WeVisOne-VL text bills the published Qwen3-VL RMSNorm and SwiGLU product.

Qwen3VLTextRMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. q_norm and k_norm are that RMSNorm on
head_dim (one rsqrt per head). Weights stay fp32 and activations stay bf16.
4B q width is wider than hidden. rope_parameters mRoPE does not add an op.
The vision tower's gelu_pytorch_tanh is not retagged. These two ids are
outside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import RoleFormat, _tag_qwen3vl_vectors, _tc, from_release, load_release
from accel_dse.core.vision import image_grid, vision_ops


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, vocab
_ROWS = {
    "wevisone-vl-4b": (
        "tencent/WeVisOne-VL-4B", 2560, 9728, 36, 32, 8, 128, 4_415_233_536, 152680),
    "wevisone-vl-2b": (
        "tencent/WeVisOne-VL-2B", 2048, 6144, 28, 16, 8, 128, 2_034_787_328, 152680),
}
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")
_HAND = {
    "wevisone-vl-4b": 22_033_182_228_480,
    "wevisone-vl-2b": 8_658_654_068_736,
}
_EXTRA = {"wevisone-vl-4b": 1_087_488, "wevisone-vl-2b": 516_096}
_FP32 = "safetensors 是 fp32（config dtype 为 bfloat16）。语言 GEMM 与视觉塔权重按 fp32 计，激活仍是 bf16。"


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


def _bad(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"]["text_config"])
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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_454():
    assert __version__ == "0.459.0"


def test_wevisone_text_bills_fp32_rmsnorm_qk_and_the_product():
    for mid, row in _ROWS.items():
        hf, h, inner, depth, hq, hkv, hd, params, vocab = row
        m = from_release(mid, hf)
        rel = load_release(hf)
        cfg = rel["config"]["text_config"]
        qn, kn = hq * hd, hkv * hd
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.vocab == vocab and m.tie_embeddings is False and m.domain == "llm"
        assert m.final_norm_params is None and m.vec_kind == "qwen3vl_text"
        assert all(L.vec_kind == "qwen3vl_text" for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        assert all(L.ffn.gated and L.ffn.kind == "dense" and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert rel["roles"]["norm"]["fp32"]["params"] == depth * (2 * h + 2 * hd) + h
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "qwen3_vl_text"
        assert cfg["attention_bias"] is False and cfg["attention_dropout"] == 0.0
        assert "mlp_bias" not in cfg and cfg.get("rope_scaling") is None
        assert cfg.get("sliding_window") is None and cfg.get("use_sliding_window") is None
        rp = cfg["rope_parameters"]
        assert rp["rope_type"] == "default" and rp["mrope_interleaved"] is True
        assert rp["mrope_section"] == [24, 20, 20]
        assert depth * (inner + qn * 4 + kn * 4) == _EXTRA[mid]
        for role in ("attn", "mlp", "embed", "lm_head"):
            assert m.fmt(role).fmt == "fp32" and m.fmt(role).bits == 32.0
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
        _rms(by["q_rms"], t, qn, hq, 0)
        _rms(by["k_rms"], t, kn, hkv, 0)
        _rms(by["attn_norm"], t, h, 1, 0)
        _rms(by["ffn_norm"], t, h, 1, 0)
        assert by["q_rms"].replicated == 1 and by["k_rms"].replicated == 1
        if mid == "wevisone-vl-4b":
            assert qn == 4096 and qn > h
            assert by["q_rms"].vec == t * 4096 * 4
            assert by["q_rms"].vec != by["attn_norm"].vec
        else:
            assert qn == h
            assert by["q_rms"].vec == by["attn_norm"].vec
            assert by["q_rms"].sfu == float(t * hq) and by["attn_norm"].sfu == float(t)
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, h, 2 * inner)
        assert (down.m, down.k, down.n) == (t, inner, h)
        assert up.flops == 2 * t * h * (2 * inner) and down.flops == 2 * t * inner * h
        assert up.w_fmt == "fp32" and up.w_bits == 32.0 and up.a_fmt == "bf16"
        assert down.w_fmt == "fp32" and down.w_bits == 32.0
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("attn.o")
        assert names.index("attn.o") < names.index("q_rms") < names.index("k_rms") < names.index("qk")
        assert names.index("qk") < names.index("rope")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert names.index("mlp.down") < names.index("residual")
        assert "attn.q" not in names and "attn.k" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        cleared = _bare(m)
        bare_ops = build_rank_ops(cleared, 0, ph, Shard())
        assert "mlp_mul" not in [o.name for o in bare_ops]
        assert "q_rms" not in [o.name for o in bare_ops]
        assert by["residual"].vec == _by(bare_ops)["residual"].vec
        assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _act(last["act"], t * inner, depth - 1, "bf16")
        _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
        _rms(last["q_rms"], t, qn, hq, depth - 1)
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, 1, depth)
        bare_head = _by(head_ops(cleared, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        assert _FP32 in m.notes
        assert any("0.446" in n for n in m.notes)
        assert any("0.454" in n for n in m.notes)
        assert m.vision is not None and m.vision.ln_mode == "qwen3vl"
        vnames = [o.name for o in vision_ops(m.vision, image_grid(m.vision, 224, 224), 1)]
        assert "vis.mgelu" in vnames and "vis.0.attn_ln" in vnames
        assert "mlp_mul" not in vnames and "q_rms" not in vnames and "k_rms" not in vnames
        assert "mlp_mul" in names and "q_rms" in names


def test_decode_batch_tp_and_request_fp8():
    m = get_model("wevisone-vl-4b")
    h, inner, hq, hkv, hd = 2560, 9728, 32, 8, 128
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    _rms(dec["q_rms"], 1, hq * hd, hq, 0)
    _rms(dec["k_rms"], 1, hkv * hd, hkv, 0)
    assert dec["q_rms"].vec == 4096 * 4 and dec["attn_norm"].vec == h * 4
    assert dec["attn_norm"].sfu == 1.0 and dec["mlp.gate_up"].m == 1
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    _rms(two["q_rms"], 2, hq * hd, hq, 0)
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
    assert loc["mlp.gate_up"].w_fmt == "fp32" and loc["mlp.gate_up"].w_bits == 32.0
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("q_rms") < names.index("attn_allreduce")
    assert names.index("attn_allreduce") < names.index("ffn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    two_b = get_model("wevisone-vl-2b")
    loc2 = _by(build_rank_ops(two_b, 0, Phase("prefill", 1, 4096, 0), sh))
    _rms(loc2["q_rms"], 4096, 8 * hd, 8, 0)
    _rms(loc2["k_rms"], 4096, 4 * hd, 4, 0)
    _mul(loc2["mlp_mul"], 4096 * 3072, 0, "bf16")
    assert loc2["mlp.gate_up"].n == 6144 and loc2["mlp.down"].k == 3072
    assert loc2["q_rms"].vec == 4096 * 1024 * 4
    assert loc2["attn_norm"].vec == 4096 * 2048 * 4
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["q_rms"].a_fmt == "fp32"
    assert op["ffn_norm"].a_fmt == "fp32" and op["k_rms"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "fp32"
    assert op["mlp.down"].w_bits == 32.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_other_models():
    m = get_model("wevisone-vl-4b")
    ph = Phase("prefill", 1, 256, 0)
    cleared = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-tts-0.6b-base", "qwen3-asr-1.7b", "qwen3-asr-0.6b",
                "fun-asr-nano-2512", "cosyvoice2-0.5b", "fun-cosyvoice3-0.5b-2512",
                "ming-3.0-tts-0.5b", "ming-omni-tts-0.5b"):
        other = get_model(mid)
        assert other.vec_kind != "qwen3vl_text"
        assert all(L.vec_kind != "qwen3vl_text" for L in other.layers)
        assert not any("0.454" in n for n in other.notes)
    assert get_model("qwen3-8b").vec_kind == ""
    assert get_model("qwen3-32b").vec_kind == ""
    dense = get_model("qwen3-vl-4b")
    assert dense.vec_kind == "qwen3vl_text"
    assert any("0.446" in n for n in dense.notes)
    assert not any("0.454" in n for n in dense.notes)
    assert "mrope" not in " ".join(dense.notes)
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("qwen3-8b-dspine").vec_kind == "qwen3"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    assert get_model("mistral-small-3.2-24b-instruct-2506").vec_kind == "mistral_small"
    assert get_model("kimi-audio-7b").vec_kind == "qwen2_text"
    assert get_model("fireredtts3-base").vec_kind == "qwen3"
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    by_id = {e["id"]: e for e in entries()}
    for mid in _ROWS:
        assert mid not in ids[:60]
        assert by_id[mid]["domain"] == "vlm"
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "1e15ae85430a04f3"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "wevisone-vl-4b"
    _raises(_bad(mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3-vl hidden_act is not silu")
    _raises(_bad(mid, lambda c: c.__setitem__("model_type", "qwen3")), "qwen3-vl model_type is not qwen3_vl_text")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_bias", True)), "qwen3-vl attention_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "qwen3-vl attention_dropout is not 0")
    _raises(_bad(mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen3-vl mlp_bias is not false")
    _raises(_bad(mid, lambda c: c.__delitem__("head_dim")), "qwen3-vl head_dim is missing")
    _raises(_bad(mid, lambda c: c.__setitem__("head_dim", 0)), "qwen3-vl head_dim is not positive")
    _raises(_bad(mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3-vl layer_types is not full_attention")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen3-vl rope_scaling is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen3-vl use_sliding_window is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "qwen3-vl sliding_window is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_parameters", {"rope_type": "default"})),
            "qwen3-vl rope_parameters is not mrope default")
    _raises(_bad(mid, lambda c: c["rope_parameters"].__setitem__("mrope_section", [16, 24, 24])),
            "qwen3-vl rope_parameters is not mrope default")
    _raises(_bad("wevisone-vl-2b", lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen3-vl hidden_act is not silu")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    rel = load_release(_ROWS[mid][0])
    c = _tc(rel["config"])
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1, vec_kind="")
    broken_m = replace(m, layers=(extra,) + tuple(replace(L, vec_kind="") for L in m.layers[1:]), vec_kind="")
    _raises(lambda: _tag_qwen3vl_vectors(broken_m, c), "qwen3-vl qk norm is not head_dim")
    fp16 = replace(broken_m, act_fmt="fp16")
    _raises(lambda: _tag_qwen3vl_vectors(fp16, c), "qwen3-vl activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("bf16", 16.0) if k == "mlp" else v) for k, v in m.formats)
    bf = replace(m, formats=fmts, layers=tuple(replace(L, vec_kind="") for L in m.layers), vec_kind="")
    _raises(lambda: _tag_qwen3vl_vectors(bf, c), "qwen3-vl weight dtype is not fp32")
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen3-vl mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "qwen3-vl mlp is not gate_up then down")
    lins = tuple(lin for lin in m.layers[0].attn_linears if lin.name != "q")
    broken_q = replace(m, layers=(replace(m.layers[0], attn_linears=lins),) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_q, 0, ph, Shard()), "qwen3-vl attention is not q then k")
    q = m.layers[0].attn_linears[0]
    wide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_w, 0, ph, Shard()),
            "qwen3-vl qk norm width is not a whole number of heads")
    _raises(lambda: _tag_qwen3vl_vectors(
        replace(broken_w, layers=tuple(replace(L, vec_kind="") for L in broken_w.layers), vec_kind=""), c),
        "qwen3-vl qk width is not heads times head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, row in _ROWS.items():
        m = get_model(mid)
        h = m.hidden
        inner = m.layers[0].ffn.d_ff
        expect = m.n_layers * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        assert expect == _HAND[mid]
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == _HAND[mid]
