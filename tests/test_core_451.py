"""0.451: Mistral-Small text towers bill published RMSNorm and the SwiGLU product.

MistralRMSNorm is the T5 recipe in fp32. SiLU stays the existing act op.
The gate times up product is mlp_mul. There is no q/k norm. Linear bias
is hardcoded false, so misc stays the two hidden norms. Hidden norms stay
pre-norm. sliding_window is null. These three ids are outside the first
60 catalog rows. The Pixtral vision tower keeps its own RMSNorm, SiLU,
and projector erf-GELU. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_mistral_small_vectors, _tc, from_release, load_release
from accel_dse.core.vision import image_grid, vision_ops


# id -> hf, params, family
_ROWS = {
    "mistral-small-3.2-24b-instruct-2506": (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506", 22_901_314_560, "Mistral-Small"),
    "mistral-small-3.1-24b-instruct-2503": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503", 22_901_314_560, "Mistral-Small"),
    "magistral-small-2509": (
        "mistralai/Magistral-Small-2509", 22_901_314_560, "Magistral"),
}
_H, _INNER, _DEPTH, _HQ, _HKV, _HD = 5120, 32768, 40, 32, 8, 128
_VISION = 438_958_080
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")
_FFN = 164_926_744_166_400


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


def _bad(hf: str, mid: str, edit, where: str = "text"):
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


def test_version_is_451():
    assert __version__ == "0.461.0"


def test_mistral_small_bills_fp32_rmsnorm_and_the_product():
    for mid, (hf, params, family) in _ROWS.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == _DEPTH and m.hidden == _H and m.act_fmt == "bf16"
        assert m.tie_embeddings is True and m.vocab == 131072
        assert m.vec_kind == "mistral_small"
        assert all(L.vec_kind == "mistral_small" for L in m.layers)
        assert all(L.misc_params == 2 * _H for L in m.layers)
        assert all(L.ffn.gated and L.ffn.d_ff == _INNER for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == _HD for L in m.layers)
        assert all(L.core.n_q == _HQ and L.core.n_kv == _HKV for L in m.layers)
        assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
        assert _HQ * _HD == 4096 and _HQ * _HD != _H
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (_H, _HQ * _HD)
        assert (by_lin["k"].k, by_lin["k"].n) == (_H, _HKV * _HD)
        assert (by_lin["v"].k, by_lin["v"].n) == (_H, _HKV * _HD)
        assert (by_lin["o"].k, by_lin["o"].n) == (_HQ * _HD, _H)
        cfg = load_release(hf)["config"]
        assert cfg["model_type"] == "mistral3"
        assert cfg["text_config"]["model_type"] == "mistral"
        assert cfg["text_config"]["hidden_act"] == "silu"
        assert cfg["text_config"]["sliding_window"] is None
        assert cfg["text_config"]["head_dim"] == _HD
        assert cfg["projector_hidden_act"] == "gelu"
        assert cfg["vision_config"]["hidden_act"] == "silu"
        assert "attention_bias" not in cfg and "attention_bias" not in cfg["text_config"]
        assert "mlp_bias" not in cfg and "mlp_bias" not in cfg["text_config"]
        assert m.vision.family == "pixtral" and m.vision.params == m.vision_params == _VISION
        row = next(e for e in entries() if e["id"] == mid)
        assert row["domain"] == "vlm" and row["family"] == family
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * _INNER, 0, "bf16")
        _mul(by["mlp_mul"], t * _INNER, 0, "bf16")
        _rms(by["attn_norm"], t, _H, 0)
        _rms(by["ffn_norm"], t, _H, 0)
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv")
        assert names.index("qk") < names.index("rope")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
        assert names.index("mlp.down") < names.index("residual")
        assert "q_rms" not in names and "k_rms" not in names
        assert all(o.name not in _BANNED for o in ops)
        assert all(not o.name.startswith("vis.") for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = _bare(m)
        bare_ops = build_rank_ops(bare, 0, ph, Shard())
        assert "mlp_mul" not in [o.name for o in bare_ops]
        assert by["residual"].vec == _by(bare_ops)["residual"].vec
        assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
        assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, _H, _DEPTH)
        bare_head = _by(head_ops(bare, ph, Shard()))
        assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
        up = by["mlp.gate_up"]
        assert up.w_fmt == "bf16" and up.a_fmt == "bf16" and up.w_bits == 16.0
        assert any("0.451" in n for n in m.notes)


def test_default_window_pre_norm_order():
    for mid in _ROWS:
        m = get_model(mid)
        assert m.vec_kind == "mistral_small" and m.hidden == _H
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * _INNER, 0, "bf16")
        _mul(by["mlp_mul"], t * _INNER, 0, "bf16")
        _rms(by["attn_norm"], t, _H, 0)
        _rms(by["ffn_norm"], t, _H, 0)
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, _H, 2 * _INNER)
        assert (down.m, down.k, down.n) == (t, _INNER, _H)
        assert up.flops == 2 * t * _H * (2 * _INNER)
        assert down.flops == 2 * t * _INNER * _H
        assert up.w_fmt == "bf16" and up.w_bits == 16.0
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("qk") < names.index("rope") < names.index("ffn_norm")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        assert "q_rms" not in names and "k_rms" not in names
        last = _by(build_rank_ops(m, _DEPTH - 1, ph, Shard()))
        _act(last["act"], t * _INNER, _DEPTH - 1, "bf16")
        _mul(last["mlp_mul"], t * _INNER, _DEPTH - 1, "bf16")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], t, _H, _DEPTH)
        two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
        _rms(two["final_norm"], 2 * t, _H, _DEPTH)


def test_decode_batch_tp_and_fp8():
    m = get_model("mistral-small-3.2-24b-instruct-2506")
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], _INNER, 0, "bf16")
    _mul(dec["mlp_mul"], _INNER, 0, "bf16")
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    assert dec["attn_norm"].sfu == 1.0
    assert "q_rms" not in dec and "k_rms" not in dec
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * _INNER, 0, "bf16")
    _mul(two["mlp_mul"], 2 * _INNER, 0, "bf16")
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * _INNER, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    half = _INNER // 2
    _act(loc["act"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    assert loc["mlp.gate_up"].n == _INNER and loc["mlp.down"].k == half
    assert loc["ffn_norm"].vec == 4096 * _H * 4 and loc["mlp_mul"].replicated == 1
    assert loc["attn_norm"].vec == 4096 * _H * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn_norm") < names.index("attn.qkv")
    assert names.index("attn.o") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert names.index("ffn_norm") < names.index("mlp.gate_up")
    assert "q_rms" not in names and "k_rms" not in names
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * _INNER, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * _INNER, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_pixtral_vision_stays_on_its_own_recipe():
    m = get_model("mistral-small-3.2-24b-instruct-2506")
    g = image_grid(m.vision, 1024, 1024)
    assert (g.px, g.ph, g.pw, g.merged, g.llm_tokens) == ((1036, 1036), 74, 74, 1369, 1406)
    ops = vision_ops(m.vision, g, 1)
    names = [o.name for o in ops]
    assert "vis.ln_pre" in names and "vis.0.attn_rms" in names
    assert "vis.0.silu" in names and "vis.0.mlp_mul" in names
    assert "vis.proj_rms" in names and "vis.proj_gelu" in names
    rows = 74 * 74
    silu = next(o for o in ops if o.name == "vis.0.silu")
    mul = next(o for o in ops if o.name == "vis.0.mlp_mul")
    gelu = next(o for o in ops if o.name == "vis.proj_gelu")
    ln = next(o for o in ops if o.name == "vis.ln_pre")
    assert silu.vec == rows * 4096 * 4 and silu.sfu == float(rows * 4096 * 2) and silu.a_fmt == "bf16"
    assert mul.vec == float(rows * 4096) and mul.sfu == 0.0 and mul.a_fmt == "bf16"
    assert gelu.vec == 1369 * _H * 5 and gelu.sfu == float(1369 * _H) and gelu.a_fmt == "bf16"
    assert ln.vec == rows * 1024 * 4 and ln.sfu == float(rows) and ln.a_fmt == "fp32"
    assert "act" not in names and "q_rms" not in names and "k_rms" not in names
    for mid in _ROWS:
        other = vision_ops(get_model(mid).vision, g, 1)
        assert [(o.name, o.vec, o.sfu, o.a_fmt) for o in other] == [
            (o.name, o.vec, o.sfu, o.a_fmt) for o in ops]


def test_non_vector_schedule_and_other_models():
    m = get_model("mistral-small-3.2-24b-instruct-2506")
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("qwen3-32b", "qwen2.5-72b", "llama-3.1-8b", "mistral-large", "mistral-7b",
                "mixtral-8x22b", "magistral-small", "qwen2.5-omni-7b", "qwen2-audio-7b-instruct",
                "olmo-3-32b-think", "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b"):
        other = get_model(mid)
        assert other.vec_kind != "mistral_small"
        assert all(L.vec_kind != "mistral_small" for L in other.layers)
    assert get_model("step-audio-2-r1.1").vec_kind == "step_audio"
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("granite-4.1-30b").vec_kind == "granite"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"
    assert get_model("qwen3-vl-32b").vec_kind == "qwen3vl_text"
    assert get_model("carbon-x").vec_kind == "carbon_qwen"
    plain = get_model("mistral-large")
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
    assert _fp({}) == "1e15ae85430a04f3"
    for mid in _ROWS:
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral-small-3.2-24b-instruct-2506"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")), "mistral small hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")), "mistral small model_type is not mistral")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "mistral small attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", True)),
            "mistral small attention_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)), "mistral small mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("sliding_window", 4096)),
            "mistral small sliding_window is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "mistral small use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "mistral small layer_types is not full_attention")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "mistral small rope_scaling is not null")
    loaded_early = from_release(mid, hf)
    bad_heads = dict(_tc(load_release(hf)["config"]))
    bad_heads["num_key_value_heads"] = 0
    _raises(lambda: _tag_mistral_small_vectors(loaded_early, bad_heads),
            "mistral small attention heads are not positive")
    _raises(_bad(hf, mid, lambda c: c.pop("head_dim")), "mistral small head_dim is missing")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("architectures", ["MistralForCausalLM"]), "config"),
            "mistral small architecture is not mistral3")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "mistral_small"
    rel = load_release(hf)
    c_ok = _tc(rel["config"])
    assert c_ok.get("hidden_act") == "silu" and c_ok.get("attention_bias") is None
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "mistral small mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "mistral small mlp is not gate_up then down")
    _raises(lambda: _tag_mistral_small_vectors(broken_g, c_ok), "mistral small mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_mistral_small_vectors(broken_m, c_ok), "mistral small has no qk norm")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_mistral_small_vectors(broken_h, c_ok), "mistral small attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + _HD)
    broken_q = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(broken_q,) + m.layers[1:])
    _raises(lambda: _tag_mistral_small_vectors(broken_w, c_ok),
            "mistral small qk width is not heads times head_dim")
    other = get_model("mistral-7b")
    assert _tag_mistral_small_vectors(other, _tc(load_release(other.hf_id)["config"])).vec_kind == ""


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    expect = _DEPTH * (2 * t * _H * (2 * _INNER) + 2 * t * _INNER * _H)
    assert expect == _FFN
    for mid in _ROWS:
        m = get_model(mid)
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == _FFN
