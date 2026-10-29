"""0.450: Step-Audio-2 text towers bill published Qwen2 RMSNorm and the SwiGLU product.

Qwen2RMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. There is no q/k norm. q/k/v bias
is billed in misc. The hidden norms stay pre-norm. The audio encoder
keeps erf-GELU and ReLU. These four ids are outside the first 60 catalog
rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_step_audio_vectors, _tc, from_release, load_release
from accel_dse.core.vision import audio_ops


# id -> hf, hidden, inner, depth, q heads, kv heads, head_dim, params, vision, domain, misc
_ROWS = {
    "step-audio-2-r1.1": (
        "stepfun-ai/Step-Audio-R1.1", 5120, 27648, 64, 40, 8, 128,
        32_832_033_792, 654_999_808, "asr", 17408),
    "step-audio-2-mini": (
        "stepfun-ai/Step-Audio-2-mini", 3584, 18944, 28, 28, 4, 128,
        7_663_326_720, 651_852_544, "omni", 11776),
    "step-audio-2-mini-base": (
        "stepfun-ai/Step-Audio-2-mini-Base", 3584, 18944, 28, 28, 4, 128,
        7_663_326_720, 651_852_544, "omni", 11776),
    "step-audio-2-mini-think": (
        "stepfun-ai/Step-Audio-2-mini-Think", 3584, 18944, 28, 28, 4, 128,
        7_663_326_720, 651_852_544, "omni", 11776),
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


def _bad(hf: str, mid: str, edit, where: str = "config"):
    rel = json.loads(json.dumps(load_release(hf)))
    target = rel["config"] if where == "config" else rel["config"]["text_config"]
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


def test_version_is_450():
    assert __version__ == "0.462.0"


def test_step_audio_bills_fp32_rmsnorm_and_the_product():
    for mid, (hf, h, inner, depth, hq, hkv, hd, params, vision, domain, misc) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is False
        assert m.vec_kind == "step_audio"
        assert all(L.vec_kind == "step_audio" for L in m.layers)
        assert all(L.misc_params == misc for L in m.layers)
        assert misc == 2 * h + (hq + 2 * hkv) * hd
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
        assert hq * hd == h
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (h, hq * hd)
        assert (by_lin["k"].k, by_lin["k"].n) == (h, hkv * hd)
        assert (by_lin["v"].k, by_lin["v"].n) == (h, hkv * hd)
        assert (by_lin["o"].k, by_lin["o"].n) == (hq * hd, h)
        cfg = load_release(hf)["config"]
        assert cfg["model_type"] == "step_audio_2"
        assert "hidden_act" not in cfg["text_config"]
        assert cfg["text_config"]["rope_scaling"] is None
        assert cfg["text_config"]["num_attention_groups"] == hkv
        assert m.vision.audio.params == m.vision_params == vision
        row = next(e for e in entries() if e["id"] == mid)
        assert row["domain"] == domain
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
        _rms(head["final_norm"], t, h, depth)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        up = by["mlp.gate_up"]
        assert up.w_fmt == "bf16" and up.a_fmt == "bf16" and up.w_bits == 16.0
        assert any("0.450" in n for n in m.notes)


def test_default_window_pre_norm_order():
    specs = {
        "step-audio-2-r1.1": (5120, 27648, 64, 40, 8, 128),
        "step-audio-2-mini": (3584, 18944, 28, 28, 4, 128),
        "step-audio-2-mini-base": (3584, 18944, 28, 28, 4, 128),
        "step-audio-2-mini-think": (3584, 18944, 28, 28, 4, 128),
    }
    for mid, (h, inner, depth, hq, _hkv, hd) in specs.items():
        m = get_model(mid)
        assert hq * hd == h and m.vec_kind == "step_audio"
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


def test_decode_batch_tp_and_fp8():
    m = get_model("step-audio-2-r1.1")
    h, inner = 5120, 27648
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
    mini = get_model("step-audio-2-mini")
    loc_m = _by(build_rank_ops(mini, 0, Phase("prefill", 1, 64, 0), sh))
    _mul(loc_m["mlp_mul"], 64 * (18944 // 2), 0, "bf16")
    assert loc_m["mlp.gate_up"].n == 18944 and loc_m["mlp.down"].k == 9472
    assert loc_m["ffn_norm"].vec == 64 * 3584 * 4
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_audio_tower_stays_on_erf_gelu_and_relu():
    for mid in _ROWS:
        audio = get_model(mid).vision.audio
        ops = audio_ops(audio, 10.0, 1)
        names = [o.name for o in ops]
        assert "aud.proj.gelu" in names and "aud.proj.relu" in names
        assert "mlp_mul" not in names and "q_rms" not in names and "k_rms" not in names
        assert all(o.name not in ("silu", "mlp_mul") for o in ops)
        gelu = next(o for o in ops if o.name == "aud.proj.gelu")
        relu = next(o for o in ops if o.name == "aud.proj.relu")
        assert gelu.a_fmt == "bf16" and relu.a_fmt == "bf16"
        assert gelu.vec == gelu.sfu * 5


def test_non_vector_schedule_and_other_models():
    m = get_model("step-audio-2-r1.1")
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "llama-3.1-8b", "qwen2.5-72b",
                "qwen2.5-omni-7b", "qwen2-audio-7b-instruct",
                "mistral-small-3.2-24b-instruct-2506", "olmo-3-32b-think",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b"):
        other = get_model(mid)
        assert other.vec_kind != "step_audio"
        assert all(L.vec_kind != "step_audio" for L in other.layers)
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    plain = get_model("qwen2.5-72b")
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


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "stepfun-ai/Step-Audio-R1.1", "step-audio-2-r1.1"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "step audio hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "step audio model_type is not step_audio_2")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "step audio attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "step audio mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "step audio sliding_window is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "step audio use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "step audio layer_types is not full_attention")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"}), "text"),
            "step audio rope_scaling is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_heads", 39), "text"),
            "step audio hidden is not heads times head_dim")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_groups", 4), "text"),
            "step audio num_attention_groups is not num_key_value_heads")
    _raises(_bad(hf, mid, lambda c: c["text_config"].__setitem__("hidden_act", "gelu")),
            "step audio hidden_act is not silu")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "step_audio"
    rel = load_release(hf)
    c = dict(_tc(rel["config"]))
    c["attention_bias"] = False
    _raises(lambda: _tag_step_audio_vectors(loaded, c), "step audio attention_bias is not true")
    c_ok = _tc(rel["config"])
    assert c_ok.get("hidden_act") is None
    silu = dict(c_ok)
    silu["hidden_act"] = "silu"
    assert _tag_step_audio_vectors(loaded, silu).vec_kind == "step_audio"
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "step audio mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "step audio mlp is not gate_up then down")
    _raises(lambda: _tag_step_audio_vectors(broken_g, c_ok), "step audio mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_step_audio_vectors(broken_m, c_ok), "step audio bias is not qkv")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_step_audio_vectors(broken_h, c_ok), "step audio attention is not gqa head_dim")


def test_ffn_gemm_flops_match_the_closed_form():
    cases = {
        "step-audio-2-r1.1": 222_651_104_624_640,
        "step-audio-2-mini": 46_720_654_245_888,
        "step-audio-2-mini-base": 46_720_654_245_888,
        "step-audio-2-mini-think": 46_720_654_245_888,
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
