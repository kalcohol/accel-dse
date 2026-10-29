"""0.473: Seed-OSS-36B bills the published dense SwiGLU product.

SeedOssMLP is down(silu(gate) * up). SiLU stays act. The product is
mlp_mul on the intermediate width. a_fmt follows act_fmt. The class does
not upcast. Residual dropout and attention dropout run only while
training. Block norms stay on the historical dtype. rope_type default
keeps attention_scaling at 1. vec_kind is seed_swiglu. No earlier
allowlist is extended. The row is catalog index 36. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _KIMI_SITU, _LLAMA3, _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _QWEN35_MOE,
    _SEED_SWIGLU, _tag_kimi_situ_vectors, _tag_llama3_vectors,
    _tag_mixtral_swiglu_vectors, _tag_muse_swiglu_vectors,
    _tag_qwen35_moe_vectors, _tag_seed_swiglu_vectors, _tc, from_release,
    load_release,
)


_MID = "seed-oss-36b"
_HF = "ByteDance-Seed/Seed-OSS-36B-Instruct"
_H = 5120
_INNER = 27648
_DEPTH = 64
_HEADS = 80
_KV = 8
_HD = 128
_PARAMS = 36_151_104_512
_NORM = 660_480
_TEXT = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "q_rms", "k_rms", "expert_act")
_FFN_TFLOP = 222.65110462464
_ROPE = {"rope_type": "default"}


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


def _bad(edit):
    rel = json.loads(json.dumps(load_release(_HF)))
    edit(rel["config"])
    return lambda: from_release(_MID, _HF, rel)


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


def test_version_is_473():
    assert __version__ == "0.474.0"


def test_one_row_bills_the_dense_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index(_MID) == 36
    assert _SEED_SWIGLU == frozenset({_MID})
    assert _SEED_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _SEED_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _SEED_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _SEED_SWIGLU.isdisjoint(_LLAMA3)
    assert _SEED_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _DEPTH * _INNER == 1_769_472
    assert _HEADS * _HD != _H
    m = get_model(_MID)
    assert m.hf_id == _HF and m.hidden == _H and m.n_layers == _DEPTH
    assert m.act_fmt == "bf16" and m.vec_kind == "seed_swiglu"
    assert m.params() == _PARAMS and m.release_params == _PARAMS
    assert m.param_check()["rel_err"] == 0.0
    assert m.mtp_layers == () and m.tie_embeddings is False
    assert all(L.vec_kind == "seed_swiglu" for L in m.layers)
    assert all(L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER for L in m.layers)
    assert all(L.misc_params == 2 * _H + _HEADS * _HD + 2 * _KV * _HD for L in m.layers)
    assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
    cfg = _tc(load_release(_HF)["config"])
    assert cfg["model_type"] == "seed_oss"
    assert cfg["architectures"] == ["SeedOssForCausalLM"]
    assert cfg["hidden_act"] == "silu"
    assert cfg["attention_bias"] is True and cfg["attention_out_bias"] is False
    assert cfg["mlp_bias"] is False
    assert cfg["attention_dropout"] == 0.1 and cfg["residual_dropout"] == 0.1
    assert cfg["rope_scaling"] == _ROPE
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["tie_word_embeddings"] is False
    assert cfg["rms_norm_eps"] == 1e-6
    assert cfg["head_dim"] == _HD and cfg["intermediate_size"] == _INNER
    assert cfg["num_attention_heads"] == _HEADS and cfg["num_key_value_heads"] == _KV
    assert not any("swiglu" in k or "clamp" in k for k in cfg)
    assert "sliding_window" not in cfg and "layer_types" not in cfg
    roles = load_release(_HF)["roles"]["norm"]["bf16"]
    assert roles["params"] == _NORM
    assert any(n.startswith("0.473") for n in m.notes)
    assert _tag_seed_swiglu_vectors(_bare(m), cfg).vec_kind == "seed_swiglu"
    assert get_model("mixtral-8x22b").vec_kind == "mixtral_swiglu"
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("muse-spark-open").vec_kind == "muse_swiglu"
    assert get_model("olmo-3-32b-think").vec_kind != "seed_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model(_MID)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    bare_ops = build_rank_ops(_bare(m), 0, ph, Shard())
    _mul(next(o for o in ops if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    assert next(o for o in ops if o.name == "mlp_mul").vec == 4096 * 27648
    act = next(o for o in ops if o.name == "act")
    _act(act, t * _INNER, 0, "bf16")
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
    flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
        for o in layer:
            if o.name in ("mlp.gate_up", "mlp.down"):
                flops += o.flops
    assert flops / 1e12 == _FFN_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(_bare(m), ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    assert m.layers[0].core.kind == "gqa"
    assert m.layers[0].core.n_q == _HEADS and m.layers[0].core.n_kv == _KV
    assert m.layers[0].core.qk_dim == _HD and m.layers[0].core.window is None


def test_tp_request_fp8_and_cleared_kind():
    m = get_model(_MID)
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
    for mid in ("olmo-3-32b-think", "olmo-3.1-32b-instruct", "jais-3-32b",
                "minimax-text-01", "glm-5.3", "qwen3-235b-a22b", "drama-3",
                "llama-3.1-8b", "qwen3-8b", "ernie-4.5-vl-424b-a47b"):
        other = get_model(mid)
        assert other.vec_kind != "seed_swiglu"
        assert all(L.vec_kind != "seed_swiglu" for L in other.layers)
        names = _names(build_rank_ops(other, 0, ph, Shard()))
        if other.layers[0].ffn.kind == "dense" and other.vec_kind == "":
            assert "mlp_mul" not in names
    wide = get_model(_MID)
    c_ok = _tc(load_release(_HF)["config"])
    for mid in ("olmo-3-32b-think", "mixtral-8x22b", "llama-3.3-70b", "jais-3-32b"):
        spoof = replace(_bare(get_model(mid)), id=_MID)
        _raises(lambda spoof=spoof: _tag_seed_swiglu_vectors(spoof, c_ok),
                "seed swiglu")
    kept = replace(_bare(wide), id="olmo-3-32b-think")
    assert _tag_seed_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu")),
            "seed swiglu hidden_act is not silu")
    _raises(_bad(lambda c: c.__setitem__("model_type", "llama")),
            "seed swiglu model_type is not seed_oss")
    _raises(_bad(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "seed swiglu swiglu_limit is set")
    _raises(_bad(lambda c: c.__setitem__("sliding_window", 4096)),
            "seed swiglu moe or sliding key is set")
    _raises(_bad(lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "seed swiglu architecture is not seed_oss")
    _raises(_bad(lambda c: c.__setitem__("attention_bias", False)),
            "seed swiglu attention_bias is not true")
    _raises(_bad(lambda c: c.__setitem__("attention_out_bias", True)),
            "seed swiglu attention_out_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("mlp_bias", True)),
            "seed swiglu mlp_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("attention_dropout", 0.0)),
            "seed swiglu attention_dropout is not 0.1")
    _raises(_bad(lambda c: c.__setitem__("residual_dropout", 0.0)),
            "seed swiglu residual_dropout is not 0.1")
    _raises(_bad(lambda c: c.__setitem__("rope_scaling", {"rope_type": "llama3"})),
            "seed swiglu rope_scaling is not default")
    _raises(_bad(lambda c: c.__setitem__("torch_dtype", "float16")),
            "seed swiglu dtype is not bfloat16")
    _raises(_bad(lambda c: c.__setitem__("tie_word_embeddings", True)),
            "seed swiglu tie_word_embeddings is not false")
    _raises(_bad(lambda c: c.__setitem__("rms_norm_eps", 1e-5)),
            "seed swiglu rms_norm_eps is not 1e-6")
    _raises(_bad(lambda c: c.__setitem__("num_experts", 8)),
            "seed swiglu moe or sliding key is set")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_seed_swiglu_vectors(short, c_ok),
            "seed swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_seed_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "seed swiglu mlp is not dense swiglu")
    dropped = replace(wide.layers[0], ffn_linears=wide.layers[0].ffn_linears[:1])
    _raises(lambda: _tag_seed_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "seed swiglu mlp is not gate_up then down")
    runtime = replace(
        wide.layers[0],
        vec_kind="seed_swiglu",
        ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: build_rank_ops(replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
            "seed swiglu mlp is not gate_up then down")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    # Stage summaries of the front 60 stay. Pipe tpot last bits move.
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
