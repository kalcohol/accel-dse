"""0.453: Qwen3 speech and DSpine rows bill published RMSNorm and the SwiGLU product.

Qwen3RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. q_norm and k_norm are that RMSNorm
on head_dim (one rsqrt per head). q width equals hidden. Hidden norms stay
pre-norm. qwen3-8b-dspine tags the target and the five draft layers.
moss-tts tags the language tower only. fireredtts3-base tags the backbone
only, and its TTS final_norm stays one row per sequence. These three ids
are outside the first 60 catalog rows. Front-60 qwen3-8b and qwen3-1.7b
stay on the shared tail. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.fireredtts3 import fire_head_ops
from accel_dse.core.ir import Phase, Shard, build_rank_ops, dspark_ops, head_ops
from accel_dse.core.model import _tag_qwen3_vectors, _tc, from_release, load_release
from accel_dse.core.moss_tts import moss_audio_head_ops


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, tie, act, w_fmt, w_bits, vocab
_ROWS = {
    "qwen3-8b-dspine": (
        "Qwen/Qwen3-8B+tencent/Qwen3-8B-DSpine-b16",
        4096, 12288, 36, 32, 8, 128, 8_190_735_360, False, "bf16", "bf16", 16.0, 151936),
    "moss-tts": (
        "OpenMOSS-Team/MOSS-TTS",
        4096, 12288, 36, 32, 8, 128, 8_355_492_864, False, "bf16", "bf16", 16.0, 155648),
    "fireredtts3-base": (
        "FireRedTeam/FireRedTTS3",
        2048, 6144, 28, 16, 8, 128, 1_720_574_976, True, "bf16", "fp32", 32.0, 151936),
}
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")
_HAND = {
    "qwen3-8b-dspine": 44_530_220_924_928,
    "moss-tts": 44_530_220_924_928,
    "fireredtts3-base": 8_658_654_068_736,
}


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


def _text(rel, mid):
    if mid == "moss-tts":
        return rel["config"]["language_config"]
    if mid == "fireredtts3-base":
        return rel["config"]["backbone"]
    return rel["config"]


def _bad(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(_text(rel, mid))
    return lambda: from_release(mid, hf, rel)


def _bare(model):
    return replace(
        model,
        layers=tuple(replace(L, vec_kind="") for L in model.layers),
        mtp_layers=tuple(replace(L, vec_kind="") for L in model.mtp_layers),
        vec_kind="")


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_453():
    assert __version__ == "0.463.0"


def test_qwen3_speech_and_dspine_bill_fp32_rmsnorm_qk_and_the_product():
    for mid, row in _ROWS.items():
        hf, h, inner, depth, hq, hkv, hd, params, tie, af, wf, bits, vocab = row
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == af and m.vocab == vocab
        assert m.tie_embeddings is tie and m.domain == "llm"
        assert m.vec_kind == "qwen3"
        assert all(L.vec_kind == "qwen3" for L in m.layers)
        assert all(L.misc_params == 2 * h + 2 * hd for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert hq * hd == h
        cfg = _text(load_release(hf), mid)
        assert cfg["hidden_act"] == "silu" and cfg["model_type"] == "qwen3"
        assert cfg["attention_bias"] is False
        assert cfg.get("rope_scaling") is None and cfg.get("sliding_window") is None
        assert cfg.get("use_sliding_window") is False
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
        up = by["mlp.gate_up"]
        assert up.w_fmt == wf and up.a_fmt == af and up.w_bits == bits
        assert any("0.449" in n for n in m.notes)
        assert any("0.453" in n for n in m.notes)
        head = _by(head_ops(m, ph, Shard()))
        if mid == "fireredtts3-base":
            _rms(head["final_norm"], 1, h, 1, depth)
            assert head["final_norm"].vec == h * 4
        else:
            _rms(head["final_norm"], t, h, 1, depth)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0


def test_default_window_pre_norm_order():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, row in _ROWS.items():
        _hf, h, inner, depth, hq, hkv, hd, _params, _tie, af, wf, bits, _vocab = row
        m = get_model(mid)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, af)
        _mul(by["mlp_mul"], t * inner, 0, af)
        _rms(by["q_rms"], t, hq * hd, hq, 0)
        _rms(by["k_rms"], t, hkv * hd, hkv, 0)
        _rms(by["attn_norm"], t, h, 1, 0)
        _rms(by["ffn_norm"], t, h, 1, 0)
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, h, 2 * inner)
        assert (down.m, down.k, down.n) == (t, inner, h)
        assert up.flops == 2 * t * h * (2 * inner)
        assert down.flops == 2 * t * inner * h
        assert up.w_fmt == wf and up.w_bits == bits
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("q_rms") < names.index("k_rms") < names.index("qk") < names.index("rope")
        assert names.index("rope") < names.index("ffn_norm")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
        _act(last["act"], t * inner, depth - 1, af)
        _mul(last["mlp_mul"], t * inner, depth - 1, af)
        _rms(last["q_rms"], t, hq * hd, hq, depth - 1)
    eight = get_model("qwen3-8b-dspine")
    head = _by(head_ops(eight, ph, Shard()))
    _rms(head["final_norm"], t, 4096, 1, 36)
    two = _by(head_ops(eight, Phase("prefill", 2, t, 0), Shard()))
    _rms(two["final_norm"], 2 * t, 4096, 1, 36)
    fire = get_model("fireredtts3-base")
    one = _by(head_ops(fire, ph, Shard()))
    _rms(one["final_norm"], 1, 2048, 1, 28)
    pair = _by(head_ops(fire, Phase("prefill", 2, t, 0), Shard()))
    _rms(pair["final_norm"], 2, 2048, 1, 28)
    assert pair["final_norm"].vec == 2 * 2048 * 4


def test_decode_batch_tp_fp8_draft_and_side_heads():
    m = get_model("qwen3-8b-dspine")
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
    assert names.index("attn_allreduce") < names.index("ffn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["q_rms"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0
    assert len(m.mtp_layers) == 5
    assert all(L.vec_kind == "qwen3" and L.ffn.d_ff == inner for L in m.mtp_layers)
    draft = dspark_ops(m, Phase("decode", 1, 1, 4096))
    dnames = [o.name for o in draft]
    assert dnames.count("dspine.read") == 5 and dnames.count("dspine.write") == 5
    assert dnames.count("mlp_mul") == 5 and dnames.count("q_rms") == 5 and dnames.count("k_rms") == 5
    first = m.n_layers + 1
    dmul = next(o for o in draft if o.name == "mlp_mul")
    dq = next(o for o in draft if o.name == "q_rms")
    dk = next(o for o in draft if o.name == "k_rms")
    _mul(dmul, 16 * inner, first, "bf16")
    _rms(dq, 16, hq * hd, hq, first)
    _rms(dk, 16, hkv * hd, hkv, first)
    assert "mlp_mul" not in [o.name for o in dspark_ops(m, Phase("prefill", 1, 64, 0))]
    bare = _bare(m)
    assert "mlp_mul" not in [o.name for o in dspark_ops(bare, Phase("decode", 1, 1, 4096))]
    assert "dspark_main_norm" in dnames
    moss = get_model("moss-tts")
    audio = moss_audio_head_ops(moss.moss_tts, 1)
    assert "mlp_mul" not in [o.name for o in audio] and "q_rms" not in [o.name for o in audio]
    fire = get_model("fireredtts3-base")
    flow = fire_head_ops(fire.tts, 1, fire.n_layers)
    assert "mlp_mul" not in [o.name for o in flow] and "q_rms" not in [o.name for o in flow]
    floc = _by(build_rank_ops(fire, 0, Phase("prefill", 1, 4096, 0), Shard(tp=2)))
    _rms(floc["q_rms"], 4096, 8 * 128, 8, 0)
    _rms(floc["k_rms"], 4096, 4 * 128, 4, 0)
    _mul(floc["mlp_mul"], 4096 * 3072, 0, "bf16")
    assert floc["mlp.gate_up"].n == 6144 and floc["mlp.down"].k == 3072
    assert floc["mlp.gate_up"].w_fmt == "fp32" and floc["mlp.gate_up"].w_bits == 32.0
    f8 = replace(fire, act_fmt="fp8")
    fop = _by(build_rank_ops(f8, 0, Phase("prefill", 1, 16, 0), Shard()))
    _act(fop["act"], 16 * 6144, 0, "fp8")
    _mul(fop["mlp_mul"], 16 * 6144, 0, "fp8")
    assert fop["q_rms"].a_fmt == "fp32" and fop["mlp.gate_up"].w_fmt == "fp32"
    assert fop["mlp.gate_up"].w_bits == 32.0


def test_non_vector_schedule_and_other_models():
    m = get_model("qwen3-8b-dspine")
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b", "wevisone-vl-4b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-tts-0.6b-base", "fun-asr-nano-2512",
                "cosyvoice2-0.5b", "fun-cosyvoice3-0.5b-2512", "ming-3.0-tts-0.5b"):
        other = get_model(mid)
        assert other.vec_kind != "qwen3"
        assert all(L.vec_kind != "qwen3" for L in other.layers)
        assert all(L.vec_kind != "qwen3" for L in other.mtp_layers)
    assert get_model("qwen3-8b").vec_kind == ""
    assert get_model("qwen3-1.7b").vec_kind == ""
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert not any("0.453" in n for n in get_model("qwen3-14b").notes)
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    assert get_model("mistral-small-3.2-24b-instruct-2506").vec_kind == "mistral_small"
    assert get_model("kimi-audio-7b").vec_kind == "qwen2_text"
    plain = get_model("qwen3-8b")
    pops = build_rank_ops(plain, 0, Phase("prefill", 1, 16, 0), Shard())
    pby = _by(pops)
    assert "mlp_mul" not in [o.name for o in pops] and "q_rms" not in [o.name for o in pops]
    assert pby["attn_norm"].a_fmt == "bf16" and pby["ffn_norm"].a_fmt == "bf16"
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    by_id = {e["id"]: e for e in entries()}
    for mid in _ROWS:
        assert mid not in ids[:60]
    assert by_id["qwen3-8b-dspine"]["domain"] == "llm"
    assert by_id["moss-tts"]["domain"] == "tts"
    assert by_id["fireredtts3-base"]["domain"] == "tts"
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "a9cc980add47b869"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "qwen3-8b-dspine"
    _raises(_bad(mid, lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3 hidden_act is not silu")
    _raises(_bad(mid, lambda c: c.__setitem__("model_type", "llama")), "qwen3 model_type is not qwen3")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_bias", True)), "qwen3 attention_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_dropout", 0.1)), "qwen3 attention_dropout is not 0")
    _raises(_bad(mid, lambda c: c.__setitem__("mlp_bias", True)), "qwen3 mlp_bias is not false")
    _raises(_bad(mid, lambda c: c.__delitem__("head_dim")), "qwen3 head_dim is missing")
    _raises(_bad(mid, lambda c: c.__setitem__("head_dim", 0)), "qwen3 head_dim is not positive")
    _raises(_bad(mid, lambda c: c.__setitem__("num_attention_heads", 31)),
            "qwen3 hidden is not heads times head_dim")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen3 rope_scaling is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen3 use_sliding_window is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "qwen3 sliding_window is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen3 layer_types is not full_attention")
    _raises(_bad("moss-tts", lambda c: c.__setitem__("hidden_act", "gelu")), "qwen3 hidden_act is not silu")
    _raises(_bad("moss-tts", lambda c: c.__setitem__("rope_scaling", {"rope_type": "default"})),
            "qwen3 rope_scaling is not null")
    _raises(_bad("fireredtts3-base", lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen3 hidden_act is not silu")
    _raises(_bad("fireredtts3-base", lambda c: c.__setitem__("num_attention_heads", 15)),
            "qwen3 hidden is not heads times head_dim")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    rel = load_release(_ROWS[mid][0])
    c = _tc(rel["config"])
    extra = replace(m.mtp_layers[0], misc_params=m.mtp_layers[0].misc_params + 1)
    broken = replace(m, mtp_layers=(extra,) + m.mtp_layers[1:])
    _raises(lambda: _tag_qwen3_vectors(broken, c), "qwen3 qk norm is not head_dim")
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken_l = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_l, 0, ph, Shard()), "qwen3 mlp is not gate_up then down")


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
