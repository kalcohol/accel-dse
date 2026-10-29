"""0.474: OLMo-3 32B bills the published dense SwiGLU product.

Olmo3MLP is down(silu(gate) * up). SiLU stays act. The product is
mlp_mul on the intermediate width. a_fmt follows act_fmt. The class does
not upcast. Post-attention and post-feedforward RMSNorms stay on the
historical ops. Full-projection q/k RMSNorm, YaRN attention_scaling, and
the sliding window are not this product. vec_kind is olmo3_swiglu. No
earlier allowlist is extended. The three rows sit outside the first 60.
The 7B rows stay untagged. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _KIMI_SITU, _LLAMA3, _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _OLMO3_SWIGLU,
    _OLMO3_YARN_FACTOR, _QWEN35_MOE, _SEED_SWIGLU, _tag_kimi_situ_vectors,
    _tag_llama3_vectors, _tag_mixtral_swiglu_vectors, _tag_muse_swiglu_vectors,
    _tag_olmo3_swiglu_vectors, _tag_qwen35_moe_vectors, _tag_seed_swiglu_vectors,
    _tc, from_release, load_release,
)


_ROWS = {
    "olmo-3-32b-think": "allenai/Olmo-3-32B-Think",
    "olmo-3.1-32b-instruct": "allenai/Olmo-3.1-32B-Instruct",
    "olmo-3-1125-32b": "allenai/Olmo-3-1125-32B",
}
_H = 5120
_INNER = 27648
_DEPTH = 64
_HEADS = 40
_KV = 8
_HD = 128
_PARAMS = 32_233_522_176
_NORM = 1_053_696
_MISC = 2 * _H + _HEADS * _HD + _KV * _HD
_TEXT = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "q_rms", "k_rms", "expert_act")
_FFN_TFLOP = 222.65110462464
_PATTERN = (["sliding_attention"] * 3 + ["full_attention"]) * (_DEPTH // 4)


def _names(ops):
    return [o.name for o in ops]


def _act(op, n: float, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "act"
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _mul(op, n: float, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "mlp_mul"
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(hf, edit):
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
    return lambda: from_release("olmo-3-32b-think", hf, rel)


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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.count,
                            o.w_fmt, o.a_fmt, o.share, o.replicated))
    return out


def test_version_is_474():
    assert __version__ == "0.479.0"


def test_three_rows_bill_the_dense_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(mid) for mid in _ROWS] == [359, 360, 361]
    assert _OLMO3_SWIGLU == frozenset(_ROWS)
    assert _OLMO3_SWIGLU.isdisjoint(_SEED_SWIGLU)
    assert _OLMO3_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _OLMO3_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _OLMO3_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _OLMO3_SWIGLU.isdisjoint(_LLAMA3)
    assert _OLMO3_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _DEPTH * _INNER == 1_769_472
    assert _HEADS * _HD == _H
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.act_fmt == "bf16" and m.vec_kind == "olmo3_swiglu"
        assert m.params() == _PARAMS and m.release_params == _PARAMS
        assert m.param_check()["rel_err"] == 0.0
        assert m.mtp_layers == () and m.tie_embeddings is False
        assert all(L.vec_kind == "olmo3_swiglu" for L in m.layers)
        assert all(L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER
                   for L in m.layers)
        assert all(L.misc_params == _MISC for L in m.layers)
        assert sum(1 for L in m.layers if L.core.window == 4096) == 48
        assert sum(1 for L in m.layers if L.core.window is None) == 16
        assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "olmo3"
        assert cfg["architectures"] == ["Olmo3ForCausalLM"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["attention_bias"] is False
        assert "attention_out_bias" not in cfg and cfg.get("mlp_bias") is not True
        assert "residual_dropout" not in cfg
        assert cfg["attention_dropout"] == 0.0
        assert cfg["rope_scaling"]["rope_type"] == "yarn"
        assert cfg["rope_scaling"]["attention_factor"] == _OLMO3_YARN_FACTOR
        assert cfg["sliding_window"] == 4096
        assert cfg["layer_types"] == _PATTERN
        assert cfg.get("dtype") == "bfloat16" or cfg.get("torch_dtype") == "bfloat16"
        assert cfg["tie_word_embeddings"] is False
        assert cfg["rms_norm_eps"] == 1e-6
        assert "head_dim" not in cfg
        assert cfg["intermediate_size"] == _INNER
        assert cfg["num_attention_heads"] == _HEADS and cfg["num_key_value_heads"] == _KV
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        roles = load_release(hf)["roles"]["norm"]["bf16"]
        assert roles["params"] == _NORM
        assert any(n.startswith("0.474") for n in m.notes)
        assert _tag_olmo3_swiglu_vectors(_bare(m), cfg).vec_kind == "olmo3_swiglu"
    assert get_model("olmo-3-7b-think").vec_kind == ""
    assert get_model("olmo-3-7b-instruct").vec_kind == ""
    assert get_model("olmo-3-1025-7b").vec_kind == ""
    assert get_model("seed-oss-36b").vec_kind == "seed_swiglu"
    assert get_model("mixtral-8x22b").vec_kind == "mixtral_swiglu"
    assert get_model("jais-3-32b").vec_kind != "olmo3_swiglu"
    assert get_model("exaone-4.0-32b").vec_kind == "exaone4"


def test_default_window_product_and_gemm_flops():
    m = get_model("olmo-3-32b-think")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    for li, window in ((0, 4096), (3, None)):
        ops = build_rank_ops(m, li, ph, Shard())
        bare_ops = build_rank_ops(bare, li, ph, Shard())
        _mul(next(o for o in ops if o.name == "mlp_mul"), t * _INNER, li, "bf16")
        assert next(o for o in ops if o.name == "mlp_mul").vec == 4096 * 27648
        _act(next(o for o in ops if o.name == "act"), t * _INNER, li, "bf16")
        names = _names(ops)
        assert [n for n in names if n in _TEXT or n in _BANNED] == list(_TEXT)
        assert names.index("mlp.gate_up") < names.index("act") < names.index("mlp_mul")
        assert names.index("mlp_mul") < names.index("mlp.down")
        up = next(o for o in ops if o.name == "mlp.gate_up")
        down = next(o for o in ops if o.name == "mlp.down")
        assert (up.k, up.n) == (_H, 2 * _INNER)
        assert (down.k, down.n) == (_INNER, _H)
        assert up.w_fmt == "bf16" and down.w_fmt == "bf16" and up.w_bits == 16.0
        assert up.a_fmt == "bf16" and down.a_fmt == "bf16"
        q = next(o for o in ops if o.name == "attn.qkv")
        assert q.n == _HEADS * _HD + 2 * _KV * _HD
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        for name in ("attn_norm", "ffn_norm", "rope", "residual"):
            got = next(o for o in ops if o.name == name)
            old = next(o for o in bare_ops if o.name == name)
            assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
        assert next(o for o in ops if o.name == "attn_norm").a_fmt == "bf16"
        assert next(o for o in ops if o.name == "attn_norm").sfu == float(t)
        assert next(o for o in ops if o.name == "rope").vec == t * _HEADS * (_HD // 2) * 3
        assert m.layers[li].core.window == window
        assert not m.layers[li].core.no_rope
    flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
        for o in layer:
            if o.name in ("mlp.gate_up", "mlp.down"):
                flops += o.flops
    assert flops / 1e12 == _FFN_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    other = get_model("olmo-3-1125-32b")
    twin = build_rank_ops(other, 0, ph, Shard())
    _mul(next(o for o in twin if o.name == "mlp_mul"), t * _INNER, 0, "bf16")


def test_tp_request_fp8_and_cleared_kind():
    m = get_model("olmo-3-32b-think")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for tp in (2, 4, 8):
        got = build_rank_ops(m, 0, ph, Shard(tp=tp))
        _mul(next(o for o in got if o.name == "mlp_mul"), t * (_INNER // tp), 0, "bf16")
        assert next(o for o in got if o.name == "mlp.down").k == _INNER // tp
        assert _sched(m, ph, Shard(tp=tp)) == _sched(_bare(m), ph, Shard(tp=tp))
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    _act(next(o for o in got if o.name == "act"), t * _INNER, 0, "fp8")
    gem = next(o for o in got if o.name == "mlp.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("mlp.gate_up") < bnames.index("mlp.down") < bnames.index("act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("olmo-3-7b-think", "jais-3-32b", "seed-oss-36b",
                "minimax-text-01", "glm-5.3", "qwen3-235b-a22b", "drama-3",
                "llama-3.1-8b", "ernie-4.5-vl-424b-a47b"):
        other = get_model(mid)
        assert other.vec_kind != "olmo3_swiglu"
        assert all(L.vec_kind != "olmo3_swiglu" for L in other.layers)
        names = _names(build_rank_ops(other, 0, ph, Shard()))
        if other.layers[0].ffn.kind == "dense" and other.vec_kind == "":
            assert "mlp_mul" not in names
    wide = get_model("olmo-3-32b-think")
    c_ok = _tc(load_release(_ROWS["olmo-3-32b-think"])["config"])
    for mid in ("jais-3-32b", "seed-oss-36b", "olmo-3-7b-think", "exaone-4.0-32b"):
        spoof = replace(_bare(get_model(mid)), id="olmo-3-32b-think")
        _raises(lambda spoof=spoof: _tag_olmo3_swiglu_vectors(spoof, c_ok),
                "olmo3 swiglu")
    kept = replace(_bare(wide), id="jais-3-32b")
    assert _tag_olmo3_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["olmo-3-32b-think"]
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "olmo3 swiglu hidden_act is not silu")
    _raises(_bad(hf, lambda c: c.__setitem__("model_type", "llama")),
            "olmo3 swiglu model_type is not olmo3")
    _raises(_bad(hf, lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "olmo3 swiglu swiglu_limit is set")
    _raises(_bad(hf, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "olmo3 swiglu architecture is not olmo3")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_bias", True)),
            "olmo3 swiglu attention_bias is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_out_bias", False)),
            "olmo3 swiglu attention_out_bias is set")
    _raises(_bad(hf, lambda c: c.__setitem__("mlp_bias", True)),
            "olmo3 swiglu mlp_bias is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("residual_dropout", 0.0)),
            "olmo3 swiglu residual_dropout is set")
    _raises(_bad(hf, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "olmo3 swiglu attention_dropout is not 0")
    _raises(_bad(hf, lambda c: c.__setitem__("rope_scaling", {"rope_type": "default"})),
            "olmo3 swiglu rope_scaling is not yarn")
    _raises(_bad(hf, lambda c: c["rope_scaling"].__setitem__("attention_factor", 1.0)),
            "olmo3 swiglu attention_factor is not published")
    _raises(_bad(hf, lambda c: c.__setitem__("sliding_window", 2048)),
            "olmo3 swiglu sliding_window is not 4096")
    _raises(_bad(hf, lambda c: c.__setitem__("dtype", "float16")),
            "olmo3 swiglu dtype is not bfloat16")
    _raises(_bad(hf, lambda c: c.__setitem__("tie_word_embeddings", True)),
            "olmo3 swiglu tie_word_embeddings is not false")
    _raises(_bad(hf, lambda c: c.__setitem__("rms_norm_eps", 1e-5)),
            "olmo3 swiglu rms_norm_eps is not 1e-6")
    _raises(_bad(hf, lambda c: c.__setitem__("num_experts", 8)),
            "olmo3 swiglu moe key is set")
    _raises(_bad(hf, lambda c: c.__setitem__("layer_types", ["full_attention"] * _DEPTH)),
            "olmo3 swiglu layer_types is not 3 sliding then full")
    _raises(_bad(hf, lambda c: c.__setitem__("head_dim", 64)),
            "olmo3 swiglu head_dim is not hidden over heads")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_olmo3_swiglu_vectors(short, c_ok),
            "olmo3 swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_olmo3_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "olmo3 swiglu mlp is not dense swiglu")
    dropped = replace(wide.layers[0], ffn_linears=wide.layers[0].ffn_linears[:1])
    _raises(lambda: _tag_olmo3_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "olmo3 swiglu mlp is not gate_up then down")
    runtime = replace(
        wide.layers[0],
        vec_kind="olmo3_swiglu",
        ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
        "olmo3 swiglu mlp is not gate_up then down")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    # These three rows are outside the first 60. Stage and pipe stay.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    test_fingerprints_change_only_v4()
    from accel_dse.core import evaluate as E
    pins = {
        "qwen3-32b": "b975b7b3760e",
        "qwen3-8b": "11f6f3a5d21f",
        "llama-3.3-70b": "8e50c6cabdc8",
        "mixtral-8x22b": "0d8711691e1e",
        "seed-oss-36b": "f547c49c5769",
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
