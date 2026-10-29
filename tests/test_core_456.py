"""0.456: OmniASR LLM text towers bill the published Llama RMSNorm and product.

LlamaRMSNorm is the T5 recipe in fp32. SiLU stays the existing act op. The
gate times up product is mlp_mul. There is no q/k RMSNorm. The four catalog
rows share one decoder. Weights stay fp32. Activations stay bf16. final_norm
covers every prompt token. The audio tower stays on its existing graph. These
ids are outside the first 60 catalog rows. Vector flops stay 0. vec_kind
reuses carbon_llama.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import RoleFormat, _tag_omniasr_vectors, _tc, from_release, load_release
from accel_dse.core.vision import audio_ops


# id -> hf, vision params, tensor count, bytes_total, audio hidden, audio layers
_ROWS = {
    "omniasr-llm-300m-v2": (
        "bezzam/omniasr-llm-300m-v2", 319_635_968, 533, 6_553_815_040, 1024, 24),
    "omniasr-llm-1b-v2": (
        "bezzam/omniasr-llm-1b-v2", 967_742_976, 917, 9_146_243_072, 1280, 48),
    "omniasr-llm-3b-v2": (
        "bezzam/omniasr-llm-3b-v2", 3_068_711_424, 1109, 17_550_116_864, 2048, 60),
    "omniasr-llm-7b-v2": (
        "bezzam/omniasr-llm-7b-v2", 6_493_073_920, 2197, 31_247_566_848, 2048, 128),
}
_H, _INNER, _DEPTH, _HQ, _HKV, _HD = 4096, 2816, 12, 8, 8, 512
_VOCAB = 11984
_PARAMS = 1_318_817_792
_MISC = 2 * _H
_EXTRA = 33_792
_HAND = 3_401_614_098_432
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


def _bad(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"]["text_config"])
    return lambda: from_release(mid, hf, rel)


def _bad_root(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
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


def test_version_is_456():
    assert __version__ == "0.478.0"


def test_omniasr_text_bills_fp32_rmsnorm_and_the_product():
    for mid, row in _ROWS.items():
        hf, vision, tensors, total, ah, al = row
        m = from_release(mid, hf)
        rel = load_release(hf)
        cfg = rel["config"]
        tc = cfg["text_config"]
        assert m.n_layers == _DEPTH and m.hidden == _H and m.act_fmt == "bf16"
        assert m.vocab == _VOCAB and m.domain == "llm" and m.vec_kind == "carbon_llama"
        assert all(L.vec_kind == "carbon_llama" for L in m.layers)
        assert m.final_norm_params is None and m.tie_embeddings is False
        assert m.tts is None and m.cosy is None
        assert all(L.misc_params == _MISC for L in m.layers)
        assert all(L.ffn.gated and L.ffn.kind == "dense" and L.ffn.d_ff == _INNER for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == _HD for L in m.layers)
        assert all(L.core.n_q == _HQ and L.core.n_kv == _HKV and L.core.window is None for L in m.layers)
        assert rel["roles"]["norm"]["fp32"]["params"] == _DEPTH * (2 * _H) + _H
        assert rel["n_tensors"] == tensors and rel["bytes_total"] == total
        assert rel["bytes_llm"] == 5_275_271_168 and rel["params_llm"] == _PARAMS
        assert rel["fetched"] == "2026-10-22" and cfg["dtype"] == "float32"
        assert cfg["model_type"] == "omniasr"
        assert cfg["architectures"] == ["OmniASRForConditionalGeneration"]
        assert cfg["transformers_version"] == "5.21.0.dev0"
        assert tc["hidden_act"] == "silu" and tc["model_type"] == "llama"
        assert tc["attention_bias"] is False and tc["mlp_bias"] is False
        assert tc["pretraining_tp"] == 1 and tc["attention_dropout"] == 0.0
        assert tc["head_dim"] == _HD and tc.get("rope_scaling") is None
        assert tc["rope_parameters"] == {"rope_theta": 10000.0, "rope_type": "default"}
        assert tc.get("layer_types") is None and tc["tie_word_embeddings"] is False
        assert _DEPTH * _INNER == _EXTRA
        assert m.params() == m.release_params == _PARAMS
        assert m.param_check()["rel_err"] == 0.0
        assert m.vision_params == vision
        for role in ("attn", "mlp", "embed", "lm_head", "norm"):
            assert m.fmt(role).fmt == "fp32" and m.fmt(role).bits == 32.0
        aud = m.vision.audio
        assert aud is not None and aud.ctc is False
        assert aud.dims[0] == ah and aud.depths[0] == al and aud.w_fmt == "fp32"
        aops = audio_ops(aud, 1.0, 1)
        assert any(o.name.endswith(".gelu") for o in aops)
        assert any(o.name.endswith(".attn_ln") for o in aops)
        assert all(o.name != "mlp_mul" for o in aops)
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
        assert up.w_fmt == "fp32" and up.w_bits == 32.0 and up.a_fmt == "bf16"
        assert down.w_fmt == "fp32" and down.w_bits == 32.0
        q = next(lin for lin in m.layers[0].attn_linears if lin.name == "q")
        k = next(lin for lin in m.layers[0].attn_linears if lin.name == "k")
        assert q.n == _H and k.n == _H
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
        _rms(head["final_norm"], t, _H, 1, _DEPTH)
        assert "mlp_mul" not in [o.name for o in head_ops(m, ph, Shard())]
        bare_head = _by(head_ops(cleared, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        assert bare_head["final_norm"].vec == head["final_norm"].vec
        assert any("0.456" in n for n in m.notes)
        assert not any("0.445" in n for n in m.notes)
        assert not any("0.455" in n for n in m.notes)


def test_decode_batch_tp_and_request_fp8():
    m = get_model("omniasr-llm-7b-v2")
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], _INNER, 0, "bf16")
    _mul(dec["mlp_mul"], _INNER, 0, "bf16")
    _rms(dec["attn_norm"], 1, _H, 1, 0)
    assert dec["attn_norm"].sfu == 1.0 and dec["mlp.gate_up"].m == 1
    assert dec["mlp.gate_up"].w_fmt == "fp32" and dec["mlp.gate_up"].w_bits == 32.0
    one = _by(head_ops(m, Phase("decode", 1, 1, 4096), Shard()))
    _rms(one["final_norm"], 1, _H, 1, _DEPTH)
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * _INNER, 0, "bf16")
    _mul(two["mlp_mul"], 2 * _INNER, 0, "bf16")
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * _INNER, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    pair = _by(head_ops(m, Phase("prefill", 2, 4096, 0), Shard()))
    _rms(pair["final_norm"], 8192, _H, 1, _DEPTH)
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    _mul(loc["mlp_mul"], 4096 * (_INNER // 2), 0, "bf16")
    assert loc["mlp_mul"].replicated == 1
    assert loc["mlp.gate_up"].n == _INNER and loc["mlp.down"].k == _INNER // 2
    assert loc["attn_norm"].vec == 4096 * _H * 4
    assert loc["mlp.gate_up"].w_fmt == "fp32" and loc["mlp.gate_up"].w_bits == 32.0
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn.o") < names.index("attn_allreduce") < names.index("ffn_norm")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert "q_rms" not in names and "k_rms" not in names
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * _INNER, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * _INNER, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "fp32"
    assert op["mlp.down"].w_bits == 32.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_published_sliding_window_key_does_not_open_a_window():
    rel = json.loads(json.dumps(load_release(_ROWS["omniasr-llm-7b-v2"][0])))
    rel["config"]["text_config"]["sliding_window"] = 1024
    m = from_release("omniasr-llm-7b-v2", _ROWS["omniasr-llm-7b-v2"][0], rel)
    assert m.vec_kind == "carbon_llama"
    assert all(L.core.window is None for L in m.layers)


def test_non_vector_schedule_and_other_models():
    ph = Phase("prefill", 1, 256, 0)
    for mid in _ROWS:
        m = get_model(mid)
        cleared = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "qwen2.5-72b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-tts-0.6b-base", "qwen3-asr-1.7b", "qwen3-asr-0.6b",
                "fun-asr-nano-2512", "ernie-4.5-0.3b-pt", "omniasr-ctc-7b-v2"):
        other = get_model(mid)
        assert other.vec_kind != "carbon_llama"
        assert all(L.vec_kind != "carbon_llama" for L in other.layers)
        assert not any("0.456" in n for n in other.notes)
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert get_model("qwen3-32b").vec_kind == "qwen3"
    assert get_model("llama-3.1-8b").vec_kind == ""
    assert get_model("omniasr-ctc-300m-v2").vec_kind == ""
    carbon = get_model("carbon-8b")
    assert carbon.vec_kind == "carbon_llama"
    assert any("0.445" in n for n in carbon.notes)
    assert not any("0.456" in n for n in carbon.notes)
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    assert get_model("qwen2-audio-7b").vec_kind == "qwen2_text"
    assert get_model("cosyvoice2-0.5b").vec_kind == "qwen2_text"
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("wevisone-vl-4b").vec_kind == "qwen3vl_text"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    by_id = {e["id"]: e for e in entries()}
    visions = set()
    for mid in _ROWS:
        assert mid not in ids[:60]
        assert by_id[mid]["domain"] == "asr"
        visions.add(get_model(mid).vision_params)
    assert len(visions) == 4
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "omniasr-llm-7b-v2"
    _raises(_bad(mid, lambda c: c.__setitem__("hidden_act", "gelu")), "omniasr text hidden_act is not silu")
    _raises(_bad(mid, lambda c: c.__setitem__("model_type", "qwen2")), "omniasr text model_type is not llama")
    _raises(_bad_root(mid, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "omniasr text architecture is not omniasr")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_bias", True)),
            "omniasr text attention_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("mlp_bias", True)), "omniasr text mlp_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "omniasr text attention_dropout is not 0")
    _raises(_bad(mid, lambda c: c.__setitem__("pretraining_tp", 2)),
            "omniasr text pretraining_tp is not 1")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "omniasr text rope_scaling is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_parameters", {"rope_type": "yarn"})),
            "omniasr text rope_type is not default")
    _raises(_bad(mid, lambda c: c.__setitem__(
        "rope_parameters", {"rope_type": "default", "mrope_section": [24, 20, 20]})),
        "omniasr text rope_type is not default")
    _raises(_bad(mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "omniasr text layer_types is not full_attention")
    _raises(_bad(mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "omniasr text use_sliding_window is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("head_dim", 128)),
            "omniasr text hidden is not heads times head_dim")
    _raises(_bad("omniasr-llm-300m-v2", lambda c: c.__setitem__("hidden_act", "gelu")),
            "omniasr text hidden_act is not silu")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    rel = load_release(_ROWS[mid][0])
    c = _tc(rel["config"])
    cleared = _bare(m)
    fp16 = replace(cleared, act_fmt="fp16")
    _raises(lambda: _tag_omniasr_vectors(fp16, c), "omniasr text activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("bf16", 16.0) if k == "mlp" else v) for k, v in m.formats)
    bf = replace(cleared, formats=fmts)
    _raises(lambda: _tag_omniasr_vectors(bf, c), "omniasr text weight dtype is not fp32")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1, vec_kind="")
    broken_m = replace(m, layers=(extra,) + tuple(replace(L, vec_kind="") for L in m.layers[1:]), vec_kind="")
    _raises(lambda: _tag_omniasr_vectors(broken_m, c), "omniasr text norm is not two hidden")
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    cleared_g = replace(broken_g, layers=tuple(replace(L, vec_kind="") for L in broken_g.layers), vec_kind="")
    _raises(lambda: _tag_omniasr_vectors(cleared_g, c), "omniasr text mlp is not dense swiglu")
    q = m.layers[0].attn_linears[0]
    wide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(wide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: _tag_omniasr_vectors(
        replace(broken_w, layers=tuple(replace(L, vec_kind="") for L in broken_w.layers), vec_kind=""), c),
        "omniasr text qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_omniasr_vectors(
        replace(broken_win, layers=tuple(replace(L, vec_kind="") for L in broken_win.layers), vec_kind=""), c),
        "omniasr text attention is not gqa head_dim")
    two = replace(cleared, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_omniasr_vectors(two, c), "omniasr text final norm is not one")
    bad_heads = dict(c)
    bad_heads["num_attention_heads"] = 7
    _raises(lambda: _tag_omniasr_vectors(cleared, bad_heads),
            "omniasr text hidden is not heads times head_dim")
    other = get_model("llama-3.1-8b")
    assert _tag_omniasr_vectors(other, _tc(load_release(other.hf_id)["config"])).vec_kind == ""
    assert _tag_omniasr_vectors(get_model("carbon-8b"), c).vec_kind == "carbon_llama"


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
