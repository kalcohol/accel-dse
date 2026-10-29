"""0.455: CosyVoice and Ming 0.5B backbones bill the published Qwen2 RMSNorm and product.

Qwen2RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op. The
gate times up product is mlp_mul. There is no q/k RMSNorm. CosyVoice stores
the LM in fp32 and norms every prompt token. Ming stores the LM in bf16. Its
DiTAR head still reads one row per sequence, so final_norm is not scaled by
the prompt length. flow, HiFT, and DiTAR stay on their existing graphs. These
four ids are outside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.cosyvoice import flow_ops, hift_ops
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import RoleFormat, _tag_qwen2_text_vectors, _tc, from_release, load_release
from accel_dse.core.tts import tts_head_ops


# id -> hf, params(), vocab, weight format, final_norm rows at prompt 4096
_ROWS = {
    "cosyvoice2-0.5b": (
        "FunAudioLLM/CosyVoice2-0.5B", 641_931_904, 6564, "fp32", 4096),
    "fun-cosyvoice3-0.5b-2512": (
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", 642_283_136, 6761, "fp32", 4096),
    "ming-3.0-tts-0.5b": (
        "inclusionAI/Ming-3.0-tts-0.5B", 494_032_768, 151936, "bf16", 1),
    "ming-omni-tts-0.5b": (
        "inclusionAI/Ming-omni-tts-0.5B", 494_032_768, 151936, "bf16", 1),
}
_H, _INNER, _DEPTH, _HQ, _HKV, _HD = 896, 4864, 24, 14, 2, 64
_MISC = 2 * _H + (_HQ + 2 * _HKV) * _HD
_HAND = 2_570_537_926_656
_EXTRA = 116_736
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")
_BITS = {"fp32": 32.0, "bf16": 16.0}


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
    edit(rel["config"]["llm_config"])
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


def test_version_is_455():
    assert __version__ == "0.465.0"


def test_qwen2_05b_backbones_bill_fp32_rmsnorm_and_the_product():
    for mid, row in _ROWS.items():
        hf, params, vocab, weights, final_rows = row
        m = from_release(mid, hf)
        rel = load_release(hf)
        cfg = rel["config"]["llm_config"]
        bits = _BITS[weights]
        assert m.n_layers == _DEPTH and m.hidden == _H and m.act_fmt == "bf16"
        assert m.vocab == vocab and m.domain == "llm" and m.vec_kind == "qwen2_text"
        assert all(L.vec_kind == "qwen2_text" for L in m.layers)
        assert m.final_norm_params is None
        assert all(L.misc_params == _MISC for L in m.layers)
        assert all(L.ffn.gated and L.ffn.kind == "dense" and L.ffn.d_ff == _INNER for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == _HD for L in m.layers)
        assert all(L.core.n_q == _HQ and L.core.n_kv == _HKV and L.core.window is None for L in m.layers)
        assert rel["roles"]["norm"][weights]["params"] == _DEPTH * (2 * _H) + _H
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "qwen2"
        assert cfg["architectures"] == ["Qwen2ForCausalLM"]
        assert cfg["use_sliding_window"] is False and cfg["sliding_window"] == 32768
        assert "attention_bias" not in cfg and "mlp_bias" not in cfg
        assert cfg.get("rope_scaling") is None and cfg.get("layer_types") is None
        assert cfg.get("head_dim") is None and cfg["attention_dropout"] == 0.0
        assert _DEPTH * _INNER == _EXTRA
        for role in ("attn", "mlp"):
            assert m.fmt(role).fmt == weights and m.fmt(role).bits == bits
        if mid == "cosyvoice2-0.5b":
            assert m.release_params - m.params() == 6564
            assert m.params() == params
            assert m.param_check()["rel_err"] != 0.0
            assert m.tie_embeddings is False and m.tts is None and m.cosy is not None
        elif mid == "fun-cosyvoice3-0.5b-2512":
            assert m.params() == m.release_params == params
            assert m.param_check()["rel_err"] == 0.0
            assert m.tie_embeddings is False and m.tts is None and m.cosy is not None
        elif mid == "ming-omni-tts-0.5b":
            assert m.params() == m.release_params == params
            assert m.param_check()["rel_err"] == 0.0
            assert rel["params_llm"] - params == 151936 * _H
            assert m.tie_embeddings is True and m.tts is not None and m.cosy is None
        else:
            assert m.params() == m.release_params == params
            assert m.param_check()["rel_err"] == 0.0
            assert "lm_head" not in rel["roles"]
            assert m.tie_embeddings is True and m.tts is not None and m.cosy is None
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * _INNER, 0, "bf16")
        _mul(by["mlp_mul"], t * _INNER, 0, "bf16")
        _rms(by["attn_norm"], t, _H, 1, 0)
        _rms(by["ffn_norm"], t, _H, 1, 0)
        assert "q_rms" not in by and "k_rms" not in by
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, _H, 2 * _INNER)
        assert (down.m, down.k, down.n) == (t, _INNER, _H)
        assert up.flops == 2 * t * _H * (2 * _INNER) and down.flops == 2 * t * _INNER * _H
        assert up.w_fmt == weights and up.w_bits == bits and up.a_fmt == "bf16"
        assert down.w_fmt == weights and down.w_bits == bits
        q = next(lin for lin in m.layers[0].attn_linears if lin.name == "q")
        assert q.n == _H
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("attn.o")
        assert names.index("attn.o") < names.index("qk") < names.index("rope")
        assert names.index("rope") < names.index("ffn_norm") < names.index("mlp.gate_up")
        assert names.index("mlp.gate_up") < names.index("act") < names.index("mlp_mul")
        assert names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        assert "attn.q" not in names and "attn.k" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        cleared = _bare(m)
        bare_ops = build_rank_ops(cleared, 0, ph, Shard())
        assert "mlp_mul" not in [o.name for o in bare_ops]
        assert by["residual"].vec == _by(bare_ops)["residual"].vec
        assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["attn_norm"].sfu == float(t)
        last = _by(build_rank_ops(m, _DEPTH - 1, ph, Shard()))
        _act(last["act"], t * _INNER, _DEPTH - 1, "bf16")
        _mul(last["mlp_mul"], t * _INNER, _DEPTH - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], final_rows, _H, 1, _DEPTH)
        bare_head = _by(head_ops(cleared, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        assert bare_head["final_norm"].vec == head["final_norm"].vec
        assert any("0.452" in n for n in m.notes)
        assert any("0.455" in n for n in m.notes)
        assert not any("0.454" in n for n in m.notes)
        if m.cosy is not None:
            chunk, ev = flow_ops(m.cosy, 8)
            voc = hift_ops(m.cosy, 16)
            assert "mlp_mul" not in [o.name for o in chunk + ev + voc]
            assert "mlp_mul" not in [o.name for o in head_ops(m, ph, Shard())]
        else:
            tops = tts_head_ops(m.tts, 1, m.n_layers)
            assert tops[0].name == "tts.stop_head"
            assert all(o.name != "mlp_mul" for o in tops)
            assert head["final_norm"].vec == _H * 4
            assert head["final_norm"].vec != t * _H * 4


def test_decode_batch_tp_and_request_fp8():
    cosy = get_model("cosyvoice2-0.5b")
    ming = get_model("ming-3.0-tts-0.5b")
    dec = _by(build_rank_ops(cosy, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], _INNER, 0, "bf16")
    _mul(dec["mlp_mul"], _INNER, 0, "bf16")
    _rms(dec["attn_norm"], 1, _H, 1, 0)
    assert dec["attn_norm"].sfu == 1.0 and dec["mlp.gate_up"].m == 1
    assert dec["mlp.gate_up"].w_fmt == "fp32" and dec["mlp.gate_up"].w_bits == 32.0
    one = _by(head_ops(cosy, Phase("decode", 1, 1, 4096), Shard()))
    _rms(one["final_norm"], 1, _H, 1, _DEPTH)
    two = _by(build_rank_ops(cosy, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * _INNER, 0, "bf16")
    _mul(two["mlp_mul"], 2 * _INNER, 0, "bf16")
    pre = _by(build_rank_ops(cosy, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * _INNER, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    pair = _by(head_ops(cosy, Phase("prefill", 2, 4096, 0), Shard()))
    _rms(pair["final_norm"], 8192, _H, 1, _DEPTH)
    ming_pre = _by(head_ops(ming, Phase("prefill", 2, 4096, 0), Shard()))
    _rms(ming_pre["final_norm"], 2, _H, 1, _DEPTH)
    assert ming_pre["final_norm"].vec == 2 * _H * 4
    ming_dec = _by(build_rank_ops(ming, 0, Phase("decode", 1, 1, 4096), Shard()))
    assert ming_dec["mlp.gate_up"].w_fmt == "bf16" and ming_dec["mlp.gate_up"].w_bits == 16.0
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(cosy, 0, Phase("prefill", 1, 4096, 0), sh))
    _mul(loc["mlp_mul"], 4096 * (_INNER // 2), 0, "bf16")
    assert loc["mlp_mul"].replicated == 1
    assert loc["mlp.gate_up"].n == _INNER and loc["mlp.down"].k == _INNER // 2
    assert loc["attn_norm"].vec == 4096 * _H * 4
    assert loc["mlp.gate_up"].w_fmt == "fp32" and loc["mlp.gate_up"].w_bits == 32.0
    names = [o.name for o in build_rank_ops(cosy, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("attn_allreduce") < names.index("ffn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert "q_rms" not in names and "k_rms" not in names
    loc_m = _by(build_rank_ops(ming, 0, Phase("prefill", 1, 4096, 0), sh))
    _mul(loc_m["mlp_mul"], 4096 * (_INNER // 2), 0, "bf16")
    assert loc_m["mlp.gate_up"].n == _INNER and loc_m["mlp.down"].k == _INNER // 2
    assert loc_m["mlp.gate_up"].w_fmt == "bf16" and loc_m["mlp.down"].w_bits == 16.0
    for src, weights, bits in ((cosy, "fp32", 32.0), (ming, "bf16", 16.0)):
        fp8 = replace(src, act_fmt="fp8")
        op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
        _act(op["act"], 4096 * _INNER, 0, "fp8")
        _mul(op["mlp_mul"], 4096 * _INNER, 0, "fp8")
        assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
        assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == weights
        assert op["mlp.down"].w_bits == bits
    cosy_head = _by(head_ops(replace(cosy, act_fmt="fp8"), Phase("prefill", 1, 4096, 0), Shard()))
    assert cosy_head["final_norm"].a_fmt == "fp32" and cosy_head["final_norm"].sfu == 4096.0
    ming_head = _by(head_ops(replace(ming, act_fmt="fp8"), Phase("prefill", 1, 4096, 0), Shard()))
    assert ming_head["final_norm"].a_fmt == "fp32" and ming_head["final_norm"].sfu == 1.0
    assert ming_head["final_norm"].vec == _H * 4


def test_published_sliding_window_key_does_not_open_a_window():
    for mid in _ROWS:
        rel = json.loads(json.dumps(load_release(_ROWS[mid][0])))
        rel["config"]["llm_config"]["sliding_window"] = 1024
        rel["config"]["llm_config"]["attention_bias"] = True
        m = from_release(mid, _ROWS[mid][0], rel)
        assert m.vec_kind == "qwen2_text"
        assert all(L.core.window is None for L in m.layers)


def test_non_vector_schedule_and_other_models():
    ph = Phase("prefill", 1, 256, 0)
    for mid in _ROWS:
        m = get_model(mid)
        cleared = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "qwen2.5-3b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-tts-0.6b-base", "qwen3-asr-1.7b", "qwen3-asr-0.6b",
                "fun-asr-nano-2512"):
        other = get_model(mid)
        assert other.vec_kind != "qwen2_text"
        assert all(L.vec_kind != "qwen2_text" for L in other.layers)
        assert not any("0.455" in n for n in other.notes)
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert get_model("qwen3-32b").vec_kind == ""
    assert get_model("qwen2.5-72b").vec_kind == "qwen2_text"
    assert any("0.463" in n for n in get_model("qwen2.5-72b").notes)
    assert not any("0.455" in n for n in get_model("qwen2.5-72b").notes)
    kimi = get_model("kimi-audio-7b")
    assert kimi.vec_kind == "qwen2_text"
    assert any("0.452" in n for n in kimi.notes)
    assert not any("0.455" in n for n in kimi.notes)
    assert get_model("qwen2.5-omni-7b").vec_kind == "qwen2_text"
    assert get_model("qwen2-audio-7b").vec_kind == "qwen2_text"
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("qwen3-8b-dspine").vec_kind == "qwen3"
    assert get_model("fireredtts3-base").vec_kind == "qwen3"
    assert get_model("wevisone-vl-4b").vec_kind == "qwen3vl_text"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("mistral-small-3.2-24b-instruct-2506").vec_kind == "mistral_small"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    by_id = {e["id"]: e for e in entries()}
    for mid in _ROWS:
        assert mid not in ids[:60]
        assert by_id[mid]["domain"] == "tts"
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "25e9cd50ab57455e"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "cosyvoice2-0.5b"
    _raises(_bad(mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen2 text hidden_act is not silu")
    _raises(_bad(mid, lambda c: c.__setitem__("model_type", "llama")), "qwen2 text model_type is not qwen2")
    _raises(_bad(mid, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "qwen2 text architecture is not qwen2")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_bias", False)),
            "qwen2 text attention_bias is not true")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen2 text rope_scaling is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "qwen2 text attention_dropout is not 0")
    _raises(_bad(mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen2 text mlp_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen2 text use_sliding_window is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("use_mrope", True)), "qwen2 text use_mrope is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen2 text layer_types is not full_attention")
    _raises(_bad(mid, lambda c: c.__setitem__("head_dim", 128)),
            "qwen2 text head_dim is not hidden over heads")
    _raises(_bad("ming-3.0-tts-0.5b", lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2 text hidden_act is not silu")
    _raises(_bad("fun-cosyvoice3-0.5b-2512", lambda c: c.__setitem__("model_type", "llama")),
            "qwen2 text model_type is not qwen2")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    rel = load_release(_ROWS[mid][0])
    c = _tc(rel["config"])
    ming = get_model("ming-3.0-tts-0.5b")
    cm = _tc(load_release(_ROWS["ming-3.0-tts-0.5b"][0])["config"])
    cleared = _bare(m)
    fp16 = replace(cleared, act_fmt="fp16")
    _raises(lambda: _tag_qwen2_text_vectors(fp16, c), "qwen2 text activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("bf16", 16.0) if k == "mlp" else v) for k, v in m.formats)
    bf = replace(cleared, formats=fmts)
    _raises(lambda: _tag_qwen2_text_vectors(bf, c), "qwen2 text weight dtype is not fp32")
    ming_cleared = _bare(ming)
    fp32 = tuple((k, RoleFormat("fp32", 32.0) if k == "mlp" else v) for k, v in ming.formats)
    wide_w = replace(ming_cleared, formats=fp32)
    _raises(lambda: _tag_qwen2_text_vectors(wide_w, cm), "qwen2 text weight dtype is not bf16")
    _raises(lambda: _tag_qwen2_text_vectors(replace(ming_cleared, act_fmt="fp16"), cm),
            "qwen2 text activation dtype is not bf16")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1, vec_kind="")
    broken_m = replace(m, layers=(extra,) + tuple(replace(L, vec_kind="") for L in m.layers[1:]), vec_kind="")
    _raises(lambda: _tag_qwen2_text_vectors(broken_m, c), "qwen2 text bias is not qkv")
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen2 text mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "qwen2 text mlp is not gate_up then down")
    _raises(lambda: _tag_qwen2_text_vectors(broken_g, c), "qwen2 text mlp is not dense swiglu")
    q = m.layers[0].attn_linears[0]
    wide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(
        replace(broken_w, layers=tuple(replace(L, vec_kind="") for L in broken_w.layers), vec_kind=""), c),
        "qwen2 text qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(broken_win, c), "qwen2 text attention is not gqa head_dim")
    two = replace(cleared, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_qwen2_text_vectors(two, c), "qwen2 text final norm is not one")
    bad_heads = dict(c)
    bad_heads["num_attention_heads"] = 15
    _raises(lambda: _tag_qwen2_text_vectors(cleared, bad_heads),
            "qwen2 text hidden is not heads times head_dim")
    other = get_model("qwen2.5-72b")
    cleared = replace(
        other, layers=tuple(replace(L, vec_kind="") for L in other.layers), vec_kind="")
    assert _tag_qwen2_text_vectors(cleared, _tc(load_release(other.hf_id)["config"])).vec_kind == ""


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    expect = _DEPTH * (2 * t * _H * (2 * _INNER) + 2 * t * _INNER * _H)
    assert expect == _HAND
    for mid in _ROWS:
        m = get_model(mid)
        got = 0.0
        extra = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
                if o.name == "mlp_mul":
                    extra += o.vec
        assert got == _HAND
        assert extra == t * _EXTRA
