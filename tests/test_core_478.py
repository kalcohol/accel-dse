"""0.478: ERNIE-4.5-300B bills the published routed expert SwiGLU product.

Ernie4_5_MoeExperts is act_fn(gate) * up, then down. hidden_act is silu.
SiLU stays expert_act. The product is mlp_mul on the useful token-expert
pairs. a_fmt follows act_fmt. There is no shared expert. Dense layers stay
the 0.458 carbon_llama bill. vec_kind on the MoE layers and the model is
ernie300_swiglu. The 21B rows stay carbon_llama. These three ids are
outside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _ERNIE300_SWIGLU, _ERNIEVL_SWIGLU, _ERNIE_MOE, _JAIS3_SWIGLU, _KIMI_SITU,
    _MINIMAX_SWIGLU, _MIXTRAL_SWIGLU, _MUSE_SWIGLU, _OLMO3_SWIGLU, _QWEN35_MOE,
    _SEED_SWIGLU, _tag_ernie300_swiglu_vectors, _tag_ernie_moe_vectors,
    _tag_ernievl_swiglu_vectors, _tag_jais3_swiglu_vectors, _tag_kimi_situ_vectors,
    _tag_minimax_swiglu_vectors, _tag_mixtral_swiglu_vectors, _tag_muse_swiglu_vectors,
    _tag_olmo3_swiglu_vectors, _tag_qwen35_moe_vectors, _tag_seed_swiglu_vectors,
    _tc, from_release, load_release,
)


_ROWS = {
    "ernie-4.5-300b-a47b-pt": "baidu/ERNIE-4.5-300B-A47B-PT",
    "ernie-4.5-300b-a47b-base-pt": "baidu/ERNIE-4.5-300B-A47B-Base-PT",
    "ernie-4.5-300b-a47b-base": "baidu/ERNIE-4.5-300B-A47B-Base",
}
_MTP = {"ernie-4.5-300b-a47b-pt": 1, "ernie-4.5-300b-a47b-base-pt": 0,
        "ernie-4.5-300b-a47b-base": 0}
_H = 8192
_INNER = 28672
_DE = 3584
_DEPTH = 54
_DENSE = 3
_EXP = 64
_TOPK = 8
_HEADS = 64
_KV = 8
_HD = 128
_PARAMS = 299_484_163_264
_TEXT = ("moe.router", "expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_DENSE_NAMES = ("mlp.gate_up", "act", "mlp_mul", "mlp.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ", "shared_gate_up")
_GEMM_TFLOP = "311.711546474496"


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


def _bad(hf, edit):
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
    return lambda: from_release("ernie-4.5-300b-a47b-pt", hf, rel)


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


def test_version_is_478():
    assert __version__ == "0.480.0"


def test_three_rows_bill_the_expert_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("ernie-4.5-300b-a47b-pt") == 120
    assert ids.index("ernie-4.5-300b-a47b-base-pt") == 121
    assert ids.index("ernie-4.5-300b-a47b-base") == 122
    assert _ERNIE300_SWIGLU == frozenset(_ROWS)
    assert _ERNIE300_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _ERNIE300_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _ERNIE300_SWIGLU.isdisjoint(_MUSE_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_MIXTRAL_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_SEED_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_OLMO3_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_JAIS3_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_ERNIEVL_SWIGLU)
    assert _ERNIE300_SWIGLU.isdisjoint(_MINIMAX_SWIGLU)
    assert _ERNIE300_SWIGLU < _ERNIE_MOE
    assert (_DEPTH - _DENSE) * _TOPK * _DE == 1_462_272
    for mid, hf in _ROWS.items():
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
        assert m.vec_kind == "ernie300_swiglu"
        assert m.params() == m.release_params == _PARAMS
        assert m.param_check()["rel_err"] == 0.0
        assert len(m.mtp_layers) == _MTP[mid]
        assert all(L.vec_kind == "" for L in m.mtp_layers)
        assert m.tie_embeddings is False and not m.quantized_release
        assert all(L.vec_kind == "carbon_llama" for L in m.layers[:_DENSE])
        assert all(L.vec_kind == "ernie300_swiglu" for L in m.layers[_DENSE:])
        assert all(L.ffn.kind == "dense" and L.ffn.gated and L.ffn.d_ff == _INNER
                   for L in m.layers[:_DENSE])
        assert all(L.ffn.kind == "moe" and L.ffn.gated and L.ffn.n_shared == 0
                   and L.ffn.d_shared == 0 and L.ffn.latent == 0 and L.ffn.d_ff == 0
                   and L.ffn.n_experts == _EXP and L.ffn.top_k == _TOPK
                   and L.ffn.d_expert == _DE for L in m.layers[_DENSE:])
        assert m.fmt("expert").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.fmt("mlp").fmt == "bf16" and m.fmt("router").fmt == "fp32"
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "ernie4_5_moe"
        assert cfg["architectures"] == ["Ernie4_5_MoeForCausalLM"]
        assert cfg["hidden_act"] == "silu" and cfg["use_bias"] is False
        assert cfg.get("rope_scaling") is None
        assert cfg.get("moe_num_shared_experts") in (None, 0)
        assert cfg["moe_k"] == _TOPK and cfg["moe_num_experts"] == _EXP
        assert cfg["moe_intermediate_size"] == _DE
        assert cfg["rms_norm_eps"] == 1e-5 and cfg["tie_word_embeddings"] is False
        assert cfg.get("quantization_config") is None
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        assert any(n.startswith("0.458") for n in m.notes)
        assert any(n.startswith("0.478") for n in m.notes)
        assert _tag_ernie300_swiglu_vectors(_bare(m), cfg).vec_kind == "ernie300_swiglu"
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("minimax-text-01").vec_kind == "minimax_swiglu"
    assert get_model("ernie-4.5-vl-424b-a47b").vec_kind == "ernievl_swiglu"
    assert get_model("ernie-4.5-21b-a3b-pt").vec_kind == "carbon_llama"
    assert get_model("glm-5.3").vec_kind != "ernie300_swiglu"


def test_default_window_product_and_gemm_flops():
    m = get_model("ernie-4.5-300b-a47b-pt")
    other = get_model("ernie-4.5-300b-a47b-base")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    bare = _bare(m)
    carbon_layers = replace(
        m, vec_kind="carbon_llama",
        layers=tuple(replace(L, vec_kind="carbon_llama") for L in m.layers))
    dense = build_rank_ops(m, 0, ph, Shard())
    dense_old = build_rank_ops(carbon_layers, 0, ph, Shard())
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * _INNER, 0, "bf16")
    assert [n for n in _names(dense) if n in _DENSE_NAMES or n in _BANNED] == list(_DENSE_NAMES)
    for name in ("attn_norm", "ffn_norm"):
        got = next(o for o in dense if o.name == name)
        old = next(o for o in dense_old if o.name == name)
        assert (got.vec, got.sfu, got.a_fmt) == (t * _H * 4, float(t), "fp32")
        assert (got.vec, got.sfu, got.a_fmt) == (old.vec, old.sfu, old.a_fmt)
    ops = build_rank_ops(m, _DENSE, ph, Shard())
    bare_ops = build_rank_ops(bare, _DENSE, ph, Shard())
    _mul(next(o for o in ops if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "bf16")
    assert next(o for o in ops if o.name == "mlp_mul").vec == 4096 * 8 * 3584
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
    assert next(o for o in ops if o.name == "moe.router").w_fmt == "fp32"
    assert next(o for o in ops if o.name == "rope").vec == t * _HEADS * (_HD // 2) * 3
    assert next(o for o in ops if o.name == "rope").vec == next(
        o for o in bare_ops if o.name == "rope").vec
    assert next(o for o in ops if o.name == "attn.qkv").n == _HEADS * _HD + 2 * _KV * _HD
    for name in ("attn_norm", "ffn_norm"):
        got = next(o for o in ops if o.name == name)
        old = next(o for o in bare_ops if o.name == name)
        assert got.a_fmt == "fp32" and got.sfu == float(t)
        assert (got.vec, got.sfu) == (old.vec, old.sfu)
        assert old.a_fmt == "bf16"
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    flops = 0.0
    bare_flops = 0.0
    width = 0.0
    for li in range(_DEPTH):
        layer = build_rank_ops(m, li, ph, Shard())
        if li < _DENSE:
            _mul(next(o for o in layer if o.name == "mlp_mul"), t * _INNER, li, "bf16")
            gemm_names = ("mlp.gate_up", "mlp.down")
        else:
            _mul(next(o for o in layer if o.name == "mlp_mul"), t * _TOPK * _DE, li, "bf16")
            width += t * _TOPK * _DE
            gemm_names = ("expert.gate_up", "expert.down")
        for o in layer:
            if o.name in ("mlp.gate_up", "mlp.down", "expert.gate_up", "expert.down"):
                flops += o.flops
        for o in build_rank_ops(bare, li, ph, Shard()):
            if o.name in gemm_names or o.name in ("mlp.gate_up", "mlp.down",
                                                  "expert.gate_up", "expert.down"):
                bare_flops += o.flops
    assert width == t * 1_462_272
    assert flops == bare_flops
    assert f"{flops / 1e12:.12f}" == _GEMM_TFLOP
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    carbon = replace(m, vec_kind="carbon_llama")
    carbon_head = next(o for o in head_ops(carbon, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (carbon_head.vec, carbon_head.sfu, carbon_head.a_fmt)
    assert head.a_fmt == "fp32" and head.sfu == float(t) and head.vec == t * _H * 4
    o3 = build_rank_ops(other, _DENSE, ph, Shard())
    _mul(next(o for o in o3 if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "bf16")
    assert _sched(m, ph, Shard()) == _sched(other, ph, Shard())
    side = build_rank_ops(m, 0, ph, Shard(), layer=m.mtp_layers[0])
    assert "mlp_mul" not in _names(side)


def test_tp_request_fp8_and_cleared_kind():
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.parallel import Layout
    from accel_dse.core.scenario import Scenario, Serving

    m = get_model("ernie-4.5-300b-a47b-pt")
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    _raises(lambda: build_rank_ops(m, _DENSE, ph, Shard(tp=2)), "ep·etp")
    _raises(lambda: evaluate(Scenario(
        model=m.id, layout=Layout(tp=2),
        serving=Serving(phase="prefill", batch=1, prompt=t))), "PP1·TP2 invalid for MoE")
    ep = build_rank_ops(m, _DENSE, ph, Shard(tp=2, ep=2))
    _mul(next(o for o in ep if o.name == "mlp_mul"), (t * _TOPK / 2) * _DE, _DENSE, "bf16")
    assert next(o for o in ep if o.name == "expert.down").k == _DE
    etp = build_rank_ops(m, _DENSE, ph, Shard(tp=2, etp=2))
    _mul(next(o for o in etp if o.name == "mlp_mul"), t * _TOPK * (_DE // 2), _DENSE, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == _DE // 2
    for tp in (4, 8):
        got = build_rank_ops(m, _DENSE, ph, Shard(tp=tp, ep=tp))
        _mul(next(o for o in got if o.name == "mlp_mul"), (t * _TOPK / tp) * _DE, _DENSE, "bf16")
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, _DENSE, ph, Shard())
    _mul(next(o for o in got if o.name == "mlp_mul"), t * _TOPK * _DE, _DENSE, "fp8")
    gem = next(o for o in got if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16" and gem.w_bits == 16.0
    assert next(o for o in got if o.name == "expert_act").a_fmt == "bf16"
    assert next(o for o in got if o.name == "attn_norm").a_fmt == "fp32"
    dense = build_rank_ops(fp8, 0, ph, Shard())
    _mul(next(o for o in dense if o.name == "mlp_mul"), t * _INNER, 0, "fp8")
    assert next(o for o in dense if o.name == "act").a_fmt == "fp8"
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, _DENSE, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("ernie-4.5-21b-a3b-pt", "ernie-4.5-21b-a3b-base-pt",
                "ernie-4.5-21b-a3b-thinking", "glm-5", "glm-5.2", "glm-5.3",
                "atria-dawn", "drama-3", "qwen3-235b-a22b",
                "ernie-4.5-vl-28b-a3b-pt", "jais-3-7b", "minimax-m2"):
        other = get_model(mid)
        assert other.vec_kind != "ernie300_swiglu"
        assert all(L.vec_kind != "ernie300_swiglu" for L in other.layers)
    narrow = get_model("ernie-4.5-21b-a3b-pt")
    li = next(i for i, L in enumerate(narrow.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(narrow, li, ph, Shard()))
    assert "mlp_mul" not in names
    assert names.index("expert.gate_up") < names.index("expert.down") < names.index("expert_act")
    wide = get_model("ernie-4.5-300b-a47b-pt")
    c_ok = _tc(load_release(_ROWS["ernie-4.5-300b-a47b-pt"])["config"])
    for mid in ("jais-3-32b", "glm-5.3", "minimax-text-01", "ernie-4.5-21b-a3b-pt",
                "ernie-4.5-vl-424b-a47b-pt"):
        spoof = replace(_bare(get_model(mid)), id="ernie-4.5-300b-a47b-pt")
        _raises(lambda spoof=spoof: _tag_ernie300_swiglu_vectors(spoof, c_ok),
                "ernie300 swiglu")
    kept = replace(_bare(wide), id="ernie-4.5-21b-a3b-pt")
    assert _tag_ernie300_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_minimax_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_ernievl_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_jais3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_seed_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_mixtral_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_muse_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_olmo3_swiglu_vectors(_bare(wide), c_ok).vec_kind == ""
    hf = _ROWS["ernie-4.5-300b-a47b-pt"]

    def _kind(edit):
        cfg = json.loads(json.dumps(c_ok))
        edit(cfg)
        return lambda: _tag_ernie300_swiglu_vectors(_bare(wide), cfg)

    _raises(_kind(lambda c: c.__setitem__("hidden_act", "gelu")),
            "ernie300 swiglu hidden_act is not silu")
    _raises(_bad(hf, lambda c: c.__setitem__("hidden_act", "gelu")),
            "ernie moe hidden_act is not silu")
    _raises(_kind(lambda c: c.__setitem__("model_type", "ernie4_5_moe_vl")),
            "ernie300 swiglu model_type is not ernie4_5_moe")
    _raises(_kind(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "ernie300 swiglu swiglu_limit is set")
    _raises(_kind(lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "ernie300 swiglu architecture is not ernie4_5_moe")
    _raises(_kind(lambda c: c.__setitem__("moe_num_shared_experts", 2)),
            "ernie300 swiglu shared expert is set")
    assert _kind(lambda c: c.__setitem__("moe_k", 6))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("moe_intermediate_size", 1536))().vec_kind == ""
    assert _kind(lambda c: c.__setitem__("moe_capacity", [32, 32, 32]))().vec_kind == "ernie300_swiglu"
    assert _kind(lambda c: c.__setitem__("moe_gate", "top2_fused"))().vec_kind == "ernie300_swiglu"
    _raises(_kind(lambda c: c.__setitem__("moe_gate", "gshard")),
            "ernie300 swiglu moe_gate is not topk")
    _raises(_kind(lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "ernie300 swiglu rope_scaling is not null")
    _raises(_bad(hf, lambda c: c.__setitem__("use_bias", True)),
            "use_bias is not modeled")
    _raises(_kind(lambda c: c.__setitem__("tie_word_embeddings", True)),
            "ernie300 swiglu tie_word_embeddings is not false")
    _raises(_kind(lambda c: c.__setitem__("rms_norm_eps", 1e-6)),
            "ernie300 swiglu rms_norm_eps is not 1e-5")
    _raises(_kind(lambda c: c.__setitem__("quantization_config", {"quant_method": "fp8"})),
            "ernie300 swiglu quantization_config is set")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_ernie300_swiglu_vectors(short, c_ok),
            "ernie300 swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[_DENSE], ffn=replace(wide.layers[_DENSE].ffn, gated=False))
    _raises(lambda: _tag_ernie300_swiglu_vectors(
        replace(_bare(wide), layers=wide.layers[:_DENSE] + (ungated,) + wide.layers[_DENSE + 1:]),
        c_ok), "ernie300 swiglu expert is not swiglu")
    runtime = replace(
        wide.layers[_DENSE],
        vec_kind="ernie300_swiglu",
        ffn=replace(wide.layers[_DENSE].ffn, gated=False))
    _raises(lambda: build_rank_ops(
        replace(wide, layers=wide.layers[:_DENSE] + (runtime,) + wide.layers[_DENSE + 1:]),
        _DENSE, ph, Shard()), "ernie300 swiglu expert is not swiglu")
    dense_as_moe = replace(wide.layers[0], vec_kind="ernie300_swiglu")
    _raises(lambda: build_rank_ops(
        replace(wide, layers=(dense_as_moe,) + wide.layers[1:]), 0, ph, Shard()),
        "ernie300 swiglu layer is not moe")
    assert _tag_ernie_moe_vectors(_bare(wide), c_ok).vec_kind == "carbon_llama"


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi

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
