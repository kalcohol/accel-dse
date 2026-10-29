"""0.472: Mixtral-8x22B bills the published expert SwiGLU product.

MixtralBlockSparseTop2MLP is w2(silu(w1(x)) * w3(x)). SiLU stays
expert_act. The product is mlp_mul on the useful token-expert pairs.
a_fmt follows act_fmt. There is no shared expert and no latent pool.
vec_kind is mixtral_swiglu. No earlier allowlist is extended. The row
is catalog index 30. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _KIMI_SITU, _LLAMA3, _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _QWEN35_MOE,
    _tag_kimi_situ_vectors, _tag_llama3_vectors, _tag_mixtral_swiglu_vectors,
    _tag_muse_swiglu_vectors, _tag_qwen35_moe_vectors, _tc, from_release,
    load_release,
)


_MID = "mixtral-8x22b"
_HF = "mistralai/Mixtral-8x22B-v0.1"
_H = 6144
_DE = 16384
_DEPTH = 56
_EXP = 8
_TOPK = 2
_HEADS = 48
_KV = 8
_HD = 128
_PARAMS = 140_620_634_112
_TEXT = ("expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "shared_gate_up")
_FFN_TFLOP = 277.076930199552


def _names(ops):
    return [o.name for o in ops]


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


def test_version_is_472():
    assert __version__ == "0.473.0"


def test_one_row_bills_the_expert_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index(_MID) == 30
    assert _MIXTRAL_SWIGLU == frozenset({_MID})
    assert _MIXTRAL_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _MIXTRAL_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _MIXTRAL_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _MIXTRAL_SWIGLU.isdisjoint(_LLAMA3)
    assert _DEPTH * _TOPK * _DE == 1_835_008
    m = get_model(_MID)
    assert m.hf_id == _HF and m.hidden == _H and m.n_layers == _DEPTH
    assert m.act_fmt == "bf16" and m.vec_kind == "mixtral_swiglu"
    assert m.params() == _PARAMS and m.release_params == _PARAMS
    assert m.param_check()["rel_err"] == 0.0
    assert m.mtp_layers == ()
    assert all(L.vec_kind == "mixtral_swiglu" for L in m.layers)
    assert all(L.ffn.kind == "moe" and L.ffn.gated and L.ffn.n_shared == 0 for L in m.layers)
    assert all(L.ffn.n_experts == _EXP and L.ffn.top_k == _TOPK and L.ffn.d_expert == _DE
               and L.ffn.latent == 0 for L in m.layers)
    assert all(L.misc_params == 2 * _H for L in m.layers)
    assert m.fmt("expert").fmt == "bf16" and m.act_fmt_for("expert") == "bf16"
    cfg = _tc(load_release(_HF)["config"])
    assert cfg["model_type"] == "mixtral"
    assert cfg["architectures"] == ["MixtralForCausalLM"]
    assert cfg["hidden_act"] == "silu"
    assert cfg["sliding_window"] is None
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["output_router_logits"] is False
    assert cfg["num_local_experts"] == _EXP and cfg["num_experts_per_tok"] == _TOPK
    assert cfg["intermediate_size"] == _DE and cfg["hidden_size"] == _H
    assert not any("swiglu" in k or "clamp" in k for k in cfg)
    assert any(n.startswith("0.472") for n in m.notes)
    assert _tag_mixtral_swiglu_vectors(_bare(m), cfg).vec_kind == "mixtral_swiglu"
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("muse-spark-open").vec_kind == "muse_swiglu"
    assert get_model("seed-oss-36b").vec_kind != "mixtral_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model(_MID)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    mul = next(o for o in ops if o.name == "mlp_mul")
    _mul(mul, t * _TOPK * _DE, 0, "bf16")
    assert mul.vec == 4096 * 2 * 16384
    act = next(o for o in ops if o.name == "expert_act")
    assert act.vec == t * _TOPK * _DE * 4 and act.sfu == float(t * _TOPK * _DE * 2)
    assert act.a_fmt == "bf16" and act.flops == 0
    names = _names(ops)
    assert [n for n in names if n in _TEXT or n in _BANNED] == list(_TEXT)
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("expert.down")
    up = next(o for o in ops if o.name == "expert.gate_up")
    down = next(o for o in ops if o.name == "expert.down")
    assert (up.k, up.n) == (_H, 2 * _DE)
    assert (down.k, down.n) == (_DE, _H)
    assert up.w_fmt == "bf16" and down.w_fmt == "bf16" and up.w_bits == 16.0
    assert up.a_fmt == "bf16" and down.a_fmt == "bf16"
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    flops = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        _mul(next(o for o in layer if o.name == "mlp_mul"), t * _TOPK * _DE, li, "bf16")
        for o in layer:
            if o.name in ("expert.gate_up", "expert.down"):
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
    _raises(lambda: build_rank_ops(m, 0, ph, Shard(tp=2)), "ep·etp")
    ep = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    _mul(next(o for o in ep if o.name == "mlp_mul"), (t * _TOPK / 2) * _DE, 0, "bf16")
    assert next(o for o in ep if o.name == "expert.down").k == _DE
    etp = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    _mul(next(o for o in etp if o.name == "mlp_mul"), t * _TOPK * (_DE // 2), 0, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == _DE // 2
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _TOPK * _DE, 0, "fp8")
    gem = next(o for o in got if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    assert next(o for o in got if o.name == "expert_act").a_fmt == "bf16"
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("seed-oss-36b", "minimax-text-01", "glm-5.3", "qwen3-235b-a22b",
                "drama-3", "deepseek-v3", "kimi-k2", "qwen3-8b", "llama-3.1-8b"):
        other = get_model(mid)
        assert other.vec_kind != "mixtral_swiglu"
        assert all(L.vec_kind != "mixtral_swiglu" for L in other.layers)
    ernie = get_model("ernie-4.5-300b-a47b-pt")
    li = next(i for i, L in enumerate(ernie.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(ernie, li, ph, Shard()))
    assert "mlp_mul" not in names
    assert names.index("expert.gate_up") < names.index("expert.down") < names.index("expert_act")
    wide = get_model(_MID)
    c_ok = _tc(load_release(_HF)["config"])
    for mid in ("qwen3.8-2.4t", "seed-oss-36b", "muse-spark-open"):
        spoof = replace(_bare(get_model(mid)), id=_MID)
        _raises(lambda spoof=spoof: _tag_mixtral_swiglu_vectors(spoof, c_ok),
                "mixtral swiglu")
    kept = replace(_bare(wide), id="muse-spark-open")
    assert _tag_mixtral_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu")),
            "mixtral swiglu hidden_act is not silu")
    _raises(_bad(lambda c: c.__setitem__("model_type", "qwen3_moe")),
            "mixtral swiglu model_type is not mixtral")
    _raises(_bad(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "mixtral swiglu swiglu_limit is set")
    _raises(_bad(lambda c: c.__setitem__("sliding_window", 4096)),
            "mixtral swiglu sliding_window is set")
    _raises(_bad(lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "mixtral swiglu architecture is not mixtral")
    _raises(_bad(lambda c: c.__setitem__("num_shared_experts", 1)),
            "mixtral swiglu shared or alternate expert key is set")
    _raises(_bad(lambda c: c.__setitem__("attention_dropout", 0.1)),
            "mixtral swiglu attention_dropout is not 0")
    _raises(_bad(lambda c: c.__setitem__("torch_dtype", "float16")),
            "mixtral swiglu dtype is not bfloat16")
    _raises(_bad(lambda c: c.__setitem__("output_router_logits", True)),
            "mixtral swiglu output_router_logits is not false")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_mixtral_swiglu_vectors(short, c_ok),
            "mixtral swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    _raises(lambda: _tag_mixtral_swiglu_vectors(
        replace(_bare(wide), layers=(ungated,) + wide.layers[1:]), c_ok),
        "mixtral swiglu expert is not swiglu")
    dropped = replace(wide.layers[0], ffn_linears=())
    _raises(lambda: _tag_mixtral_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "mixtral swiglu linears are not router")
    shared = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, n_shared=1, d_shared=_DE))
    _raises(lambda: _tag_mixtral_swiglu_vectors(
        replace(_bare(wide), layers=(shared,) + wide.layers[1:]), c_ok),
        "mixtral swiglu expert is not swiglu")
    runtime = replace(
        wide.layers[0],
        vec_kind="mixtral_swiglu",
        ffn=replace(wide.layers[0].ffn, n_shared=1))
    _raises(lambda: build_rank_ops(replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
            "mixtral swiglu expert is not swiglu")


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
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
