"""0.452: Qwen2 text towers bill published RMSNorm and the SwiGLU product.

Qwen2RMSNorm and Qwen2_5OmniRMSNorm are the T5 recipe in fp32. SiLU stays
the existing act op. The gate times up product is mlp_mul. There is no q/k
norm. Hidden norms stay pre-norm. Kimi-Audio counts norm and mimo_norm, so
the final rsqrt runs twice. These six ids are outside the first 60 catalog
rows. Audio towers and the Omni talker keep their existing ops. Vector
flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_qwen2_text_vectors, _tc, from_release, load_release
from accel_dse.core.omni import talker_ops
from accel_dse.core.vision import audio_ops


# id -> hf, params, family, domain, H, inner, depth, Hq, Hkv, hd, misc, norms, vocab, ffn
_ROWS = {
    "qwen2.5-omni-7b": (
        "Qwen/Qwen2.5-Omni-7B", 7_615_616_512, "Qwen2.5-Omni", "omni",
        3584, 18944, 28, 28, 4, 128, 11776, 1, 152064, 46_720_654_245_888),
    "qwen2.5-omni-3b": (
        "Qwen/Qwen2.5-Omni-3B", 3_397_103_616, "Qwen2.5-Omni", "omni",
        2048, 11008, 36, 16, 2, 128, 6656, 1, 151936, 19_945_828_122_624),
    "qwen2-audio-7b-instruct": (
        "Qwen/Qwen2-Audio-7B-Instruct", 7_754_878_976, "Qwen2-Audio", "asr",
        4096, 11008, 32, 32, 32, 128, 20480, 1, 156032, 35_459_249_995_776),
    "qwen2-audio-7b": (
        "Qwen/Qwen2-Audio-7B", 7_754_878_976, "Qwen2-Audio", "asr",
        4096, 11008, 32, 32, 32, 128, 20480, 1, 156032, 35_459_249_995_776),
    "kimi-audio-7b-instruct": (
        "moonshotai/Kimi-Audio-7B-Instruct", 9_735_124_992, "Kimi-Audio", "omni",
        3584, 18944, 34, 28, 4, 128, 11776, 2, 168448, 56_732_223_012_864),
    "kimi-audio-7b": (
        "moonshotai/Kimi-Audio-7B", 9_735_124_992, "Kimi-Audio", "omni",
        3584, 18944, 34, 28, 4, 128, 11776, 2, 168448, 56_732_223_012_864),
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


def _rms(op, rows: int, width: int, layer: int, nrm: int = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 * nrm and op.sfu == float(rows * nrm)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _edit_where(hf: str) -> str:
    if hf.startswith("Qwen/Qwen2.5-Omni"):
        return "thinker"
    if hf.startswith("Qwen/Qwen2-Audio"):
        return "text"
    return "config"


def _bad(hf: str, mid: str, edit, where: str | None = None):
    rel = json.loads(json.dumps(load_release(hf)))
    slot = where or _edit_where(hf)
    if slot == "config":
        target = rel["config"]
    elif slot == "thinker":
        target = rel["config"]["thinker_config"]["text_config"]
    else:
        target = rel["config"]["text_config"]
    edit(target)
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


def _geom(row):
    _hf, _params, _family, _domain, h, inner, depth, hq, hkv, hd, misc, norms, vocab, ffn = row
    return h, inner, depth, hq, hkv, hd, misc, norms, vocab, ffn


def test_version_is_452():
    assert __version__ == "0.463.0"


def test_qwen2_text_bills_fp32_rmsnorm_and_the_product():
    for mid, row in _ROWS.items():
        hf, params, family, domain, h, inner, depth, hq, hkv, hd, misc, norms, vocab, _ffn = row
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is False and m.vocab == vocab
        assert m.vec_kind == "qwen2_text"
        assert all(L.vec_kind == "qwen2_text" for L in m.layers)
        assert all(L.misc_params == misc for L in m.layers)
        assert misc == 2 * h + (hq + 2 * hkv) * hd
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
        assert hq * hd == h
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (h, h)
        assert (by_lin["k"].k, by_lin["k"].n) == (h, hkv * hd)
        assert (by_lin["v"].k, by_lin["v"].n) == (h, hkv * hd)
        assert (by_lin["o"].k, by_lin["o"].n) == (h, h)
        if norms == 2:
            assert m.final_norm_params == 2 * h
            assert [ln.name for ln in m.io_post] == ["mimo_output"]
        else:
            assert m.final_norm_params is None and m.io_post == ()
        cat = next(e for e in entries() if e["id"] == mid)
        assert cat["domain"] == domain and cat["family"] == family
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
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
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert "mlp_mul" not in [o.name for o in bare_ops]
        assert by["residual"].vec == _by(bare_ops)["residual"].vec
        assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, h, depth, norms)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        up = by["mlp.gate_up"]
        assert up.w_fmt == "bf16" and up.a_fmt == "bf16" and up.w_bits == 16.0
        assert any("0.452" in n for n in m.notes)


def test_default_window_pre_norm_order():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, row in _ROWS.items():
        h, inner, depth, _hq, _hkv, _hd, _misc, norms, _vocab, ffn = _geom(row)
        m = get_model(mid)
        assert m.vec_kind == "qwen2_text" and m.hidden == h
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
        _rms(head["final_norm"], t, h, depth, norms)
        two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
        _rms(two["final_norm"], 2 * t, h, depth, norms)
        assert depth * (up.flops + down.flops) == ffn


def test_decode_batch_tp_and_fp8():
    for mid, row in _ROWS.items():
        h, inner, _depth, _hq, _hkv, _hd, _misc, norms, _vocab, _ffn = _geom(row)
        m = get_model(mid)
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
        assert loc["attn_norm"].vec == 4096 * h * 4
        names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
        assert names.index("attn_norm") < names.index("attn.qkv")
        assert names.index("attn.o") < names.index("attn_allreduce")
        assert names.index("mlp.down") < names.index("mlp_allreduce")
        assert names.index("ffn_norm") < names.index("mlp.gate_up")
        assert "q_rms" not in names and "k_rms" not in names
        fp8 = replace(m, act_fmt="fp8")
        op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
        _act(op["act"], 4096 * inner, 0, "fp8")
        _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
        assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
        assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
        assert op["mlp.down"].w_bits == 16.0
        head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
        assert head["final_norm"].a_fmt == "fp32"
        assert head["final_norm"].sfu == float(4096 * norms)


def test_audio_towers_and_talker_stay_on_their_recipes():
    audio = get_model("qwen2-audio-7b-instruct")
    ops = audio_ops(audio.vision.audio, 10.0, 1)
    names = [o.name for o in ops]
    assert "aud.conv1.gelu" in names and "aud.0.gelu" in names and "aud.0.attn_ln" in names
    assert "mlp_mul" not in names and "q_rms" not in names and "act" not in names
    base = audio_ops(get_model("qwen2-audio-7b").vision.audio, 10.0, 1)
    assert [(o.name, o.vec, o.sfu, o.a_fmt) for o in base] == [
        (o.name, o.vec, o.sfu, o.a_fmt) for o in ops]
    gelu = next(o for o in ops if o.name == "aud.0.gelu")
    ln = next(o for o in ops if o.name == "aud.0.attn_ln")
    assert gelu.a_fmt == "bf16" and ln.a_fmt == "fp32"
    om = audio_ops(get_model("qwen2.5-omni-7b").vision.audio, 10.0, 1)
    om_names = [o.name for o in om]
    assert "aud.conv1.gelu" in om_names and "aud.0.gelu" in om_names and "aud.ln" in om_names
    assert "mlp_mul" not in om_names and "q_rms" not in om_names
    om3 = audio_ops(get_model("qwen2.5-omni-3b").vision.audio, 10.0, 1)
    assert "aud.0.gelu" in [o.name for o in om3] and "mlp_mul" not in [o.name for o in om3]
    kimi = audio_ops(get_model("kimi-audio-7b-instruct").vision.audio, 1.0, 1)
    kn = [o.name for o in kimi]
    assert "aud.proj.silu" in kn and "aud.proj.ln" in kn and "aud.0.gelu" in kn
    assert "mlp_mul" not in kn and "q_rms" not in kn
    kbase = audio_ops(get_model("kimi-audio-7b").vision.audio, 1.0, 1)
    assert [(o.name, o.vec, o.sfu, o.a_fmt) for o in kbase] == [
        (o.name, o.vec, o.sfu, o.a_fmt) for o in kimi]
    for mid in ("qwen2.5-omni-7b", "qwen2.5-omni-3b"):
        spec = get_model(mid)
        talk = talker_ops(spec.omni, 1, 64, 0)
        tnames = [o.name for o in talk]
        assert "mlp_mul" not in tnames and "act" not in tnames and "q_rms" not in tnames
        norm = next(o for o in talk if o.name == "talker.0.norm")
        assert norm.vec == 64 * spec.omni.t_h * 8
        final = next(o for o in talk if o.name == "talker.final_norm")
        assert final.vec == spec.omni.t_h * 8


def test_non_vector_schedule_and_other_models():
    for mid in ("kimi-audio-7b-instruct", "qwen2.5-omni-7b", "qwen2-audio-7b"):
        m = get_model(mid)
        ph = Phase("prefill", 1, 256, 0)
        bare = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    # 0.455 bills the Ming-0.5B Qwen2 backbone as qwen2_text. Qwen3-TTS stays off it.
    # 0.463 bills front-60 qwen2.5-72b with this same vec_kind, through a
    # separate gate. The 0.452 allowlist is unchanged. The other rows stay off it.
    for mid in ("qwen3-32b", "qwen2.5-3b", "llama-3.1-8b", "mistral-large", "mistral-7b",
                "mixtral-8x22b", "magistral-small", "olmo-3-32b-think", "qwen3-vl-235b-a22b",
                "qwen3-coder-480b-a35b", "qwen3-tts-1.7b-base"):
        other = get_model(mid)
        assert other.vec_kind != "qwen2_text"
        assert all(L.vec_kind != "qwen2_text" for L in other.layers)
    assert get_model("qwen2.5-72b").vec_kind == "qwen2_text"
    assert get_model("step-audio-2-r1.1").vec_kind == "step_audio"
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    assert get_model("mistral-small-3.2-24b-instruct-2506").vec_kind == "mistral_small"
    plain = get_model("qwen2.5-3b")
    pops = build_rank_ops(plain, 0, Phase("prefill", 1, 16, 0), Shard())
    pby = _by(pops)
    assert "mlp_mul" not in [o.name for o in pops] and "q_rms" not in [o.name for o in pops]
    assert pby["attn_norm"].a_fmt == "bf16" and pby["ffn_norm"].a_fmt == "bf16"
    head = _by(head_ops(plain, Phase("prefill", 1, 16, 0), Shard()))
    assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].sfu == 0.0
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in _ROWS:
        assert mid not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "a9cc980add47b869"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_sliding_window_key_does_not_open_a_window():
    hf, mid = "Qwen/Qwen2.5-Omni-7B", "qwen2.5-omni-7b"
    rel = json.loads(json.dumps(load_release(hf)))
    rel["config"]["thinker_config"]["text_config"]["sliding_window"] = 1024
    m = from_release(mid, hf, rel)
    assert m.vec_kind == "qwen2_text"
    assert all(L.core.window is None for L in m.layers)


def test_published_flags_and_linear_shape_are_refused():
    omni_hf, omni = "Qwen/Qwen2.5-Omni-7B", "qwen2.5-omni-7b"
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2 text hidden_act is not silu")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("model_type", "qwen2")),
            "qwen2 text model_type is not qwen2_5_omni_text")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "qwen2 text attention_dropout is not 0")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("mlp_bias", True)),
            "qwen2 text mlp_bias is not false")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen2 text use_sliding_window is not false")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen2 text layer_types is not full_attention")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen2 text rope_scaling is not mrope default")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("head_dim", 64)),
            "qwen2 text head_dim is not hidden over heads")
    _raises(_bad(omni_hf, omni, lambda c: c.__setitem__("architectures", ["Qwen2ForCausalLM"]), "config"),
            "qwen2 text architecture is not omni")
    audio_hf, audio = "Qwen/Qwen2-Audio-7B-Instruct", "qwen2-audio-7b-instruct"
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2 text hidden_act is not silu")
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__("model_type", "llama")),
            "qwen2 text model_type is not qwen2")
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__("use_mrope", True)),
            "qwen2 text use_mrope is not false")
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen2 text rope_scaling is not null")
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__("attention_bias", False)),
            "qwen2 text attention_bias is not true")
    _raises(_bad(audio_hf, audio, lambda c: c.__setitem__(
        "architectures", ["Qwen2ForCausalLM"]), "config"),
            "qwen2 text architecture is not qwen2 audio")
    kimi_hf, kimi = "moonshotai/Kimi-Audio-7B-Instruct", "kimi-audio-7b-instruct"
    _raises(_bad(kimi_hf, kimi, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2 text hidden_act is not silu")
    _raises(_bad(kimi_hf, kimi, lambda c: c.__setitem__("model_type", "qwen2")),
            "qwen2 text model_type is not empty")
    _raises(_bad(kimi_hf, kimi, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen2 text rope_scaling is not null")
    _raises(_bad(kimi_hf, kimi, lambda c: c.__setitem__("kimia_mimo_layers", 5)),
            "qwen2 text mimo layers is not 6")
    _raises(_bad(kimi_hf, kimi, lambda c: c.__setitem__("architectures", ["Qwen2ForCausalLM"])),
            "qwen2 text architecture is not kimi")
    loaded = from_release(omni, omni_hf)
    assert loaded.vec_kind == "qwen2_text"
    c_ok = _tc(load_release(omni_hf)["config"])
    # _tc injects attention_bias True for the Omni thinker, so a JSON false is overwritten.
    no_bias = dict(c_ok)
    no_bias["attention_bias"] = False
    _raises(lambda: _tag_qwen2_text_vectors(loaded, no_bias),
            "qwen2 text attention_bias is not true")
    bad_heads = dict(c_ok)
    bad_heads["num_attention_heads"] = 6
    _raises(lambda: _tag_qwen2_text_vectors(loaded, bad_heads),
            "qwen2 text hidden is not heads times head_dim")
    m = get_model(omni)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "qwen2 text mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "qwen2 text mlp is not gate_up then down")
    _raises(lambda: _tag_qwen2_text_vectors(broken_g, c_ok), "qwen2 text mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(broken_m, c_ok), "qwen2 text bias is not qkv")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(broken_h, c_ok), "qwen2 text attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 128)
    broken_q = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(broken_q,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(broken_w, c_ok),
            "qwen2 text qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_qwen2_text_vectors(broken_win, c_ok),
            "qwen2 text attention is not gqa head_dim")
    one = replace(m, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_qwen2_text_vectors(one, c_ok), "qwen2 text final norm is not one")
    km = get_model(kimi)
    kc = _tc(load_release(kimi_hf)["config"])
    single = replace(km, final_norm_params=km.hidden)
    _raises(lambda: _tag_qwen2_text_vectors(single, kc), "qwen2 text final norm is not two")
    other = get_model("qwen2.5-72b")
    cleared = replace(
        other, layers=tuple(replace(L, vec_kind="") for L in other.layers), vec_kind="")
    assert _tag_qwen2_text_vectors(cleared, _tc(load_release(other.hf_id)["config"])).vec_kind == ""


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, row in _ROWS.items():
        h, inner, depth, _hq, _hkv, _hd, _misc, _norms, _vocab, ffn = _geom(row)
        expect = depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        assert expect == ffn
        m = get_model(mid)
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == ffn
