"""0.469: Kimi-K3 SituAndMul is one situ op, not silu(gate) * up.

modeling_kimi_linear.py casts both halves to fp32. The released betas are
4 and 25. Dense KimiMLP, the shared KimiMLP, and routed KimiBlockSparseMLP
all use that formula. vec_kind is kimi_situ. The historical act and
expert_act are not emitted. Vector flops stay 0. GEMM names, shapes, and
FLOPs stay put. No earlier allowlist is extended.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, _situ_op, build_rank_ops, head_ops
from accel_dse.core.model import (
    _KIMI_SITU, _QWEN3_449, _QWEN3_453, _QWEN3_465, _QWEN3_467, _QWEN35_MOE,
    _tag_kimi_situ_vectors, _tc, from_release, load_release,
)
from accel_dse.core.scenario import Scenario


_HF = "moonshotai/Kimi-K3"
_MID = "kimi-k3"
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "mlp_mul", "act", "expert_act")
_WIDTH = 1 * 33792 + 92 * (16 * 3072 + 2 * 3072)


def _names(ops):
    return [o.name for o in ops]


def _situs(ops):
    return [o for o in ops if o.name == "situ"]


def _situ(op, n: float, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "situ"
    assert op.vec == float(n) * 11 and op.sfu == float(n) * 4
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _text(rel: dict) -> dict:
    cfg = rel["config"]
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc.get("hidden_size") else cfg


def _bad(edit):
    rel = json.loads(json.dumps(load_release(_HF)))
    edit(_text(rel))
    return lambda: from_release(_MID, _HF, rel)


def _bare(model):
    return replace(
        model,
        layers=tuple(replace(L, vec_kind="") for L in model.layers),
        vec_kind="")


def _sched(model, ph, shard, li: int):
    out = []
    for o in build_rank_ops(model, li, ph, shard):
        if o.kind != "vector":
            out.append((o.name, o.kind, o.m, o.k, o.n, o.flops, o.count,
                        o.w_fmt, o.a_fmt, o.share, o.replicated))
    return out


def test_version_is_469():
    assert __version__ == "0.472.0"


def test_kimi_k3_bills_published_situ():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index(_MID) == 4
    assert _KIMI_SITU == frozenset({_MID})
    assert _KIMI_SITU.isdisjoint(_QWEN35_MOE)
    assert _KIMI_SITU.isdisjoint(_QWEN3_449 | _QWEN3_453 | _QWEN3_465 | _QWEN3_467)
    assert _WIDTH == 5_121_024
    m = from_release(_MID, _HF)
    assert m.hf_id == _HF and m.hidden == 7168 and m.n_layers == 93
    assert m.act_fmt == "bf16" and m.vec_kind == "kimi_situ"
    assert m.params() == 2_779_481_465_440
    assert m.release_params == 2_779_484_478_208
    rel = m.param_check()["rel_err"]
    assert rel != 0.0 and abs(rel) < 2e-6
    assert not m.mtp_layers
    assert all(L.vec_kind == "kimi_situ" for L in m.layers)
    assert m.layers[0].ffn.kind == "dense" and m.layers[0].ffn.gated
    assert m.layers[0].ffn.d_ff == 33792
    dense = m.layers[0].ffn_linears
    assert [ln.name for ln in dense] == ["gate_up", "down"]
    assert dense[0].n == 2 * 33792 and dense[1].k == 33792
    moe = m.layers[1].ffn
    assert moe.kind == "moe" and moe.gated
    assert (moe.n_experts, moe.top_k, moe.d_expert) == (896, 16, 3072)
    assert (moe.n_shared, moe.d_shared, moe.latent) == (2, 3072, 3584)
    assert not moe.d_ff and not moe.groups and not moe.zero_experts and not moe.n_experts_b
    by = {ln.name: ln for ln in m.layers[1].ffn_linears}
    assert by["shared_gate_up"].n == 2 * 6144 and by["shared_down"].k == 6144
    assert by["latent_up"].n == 3584 and by["latent_down"].k == 3584
    assert m.fmt("expert").fmt == "mxfp4"
    assert round(m.fmt("expert").bits, 6) == 4.270364
    cfg = _tc(load_release(_HF)["config"])
    assert cfg["model_type"] == "kimi_linear"
    assert cfg["hidden_act"] == "situ"
    assert cfg["activation_situ_beta"] == 4.0
    assert cfg["activation_situ_linear_beta"] == 25.0
    assert cfg["first_k_dense_replace"] == 1
    assert cfg["latent_moe_use_norm"] is True
    assert not cfg.get("num_nextn_predict_layers")
    assert any(n.startswith("0.469") for n in m.notes)
    bare = _bare(m)
    assert _tag_kimi_situ_vectors(bare, cfg).vec_kind == "kimi_situ"
    t = 64
    ph = Phase("prefill", 1, t, 0)
    ops0 = build_rank_ops(m, 0, ph, Shard())
    s0 = _situs(ops0)
    assert len(s0) == 1
    _situ(s0[0], t * 33792, 0)
    names0 = _names(ops0)
    assert names0[names0.index("mlp.gate_up"):names0.index("mlp.down") + 1] == [
        "mlp.gate_up", "situ", "mlp.down"]
    ops1 = build_rank_ops(m, 1, ph, Shard())
    shared, routed = _situs(ops1)
    _situ(shared, t * 6144, 1)
    _situ(routed, t * 16 * 3072, 1)
    names1 = _names(ops1)
    assert names1[names1.index("moe.shared_gate_up"):names1.index("moe.shared_down") + 1] == [
        "moe.shared_gate_up", "situ", "moe.shared_down"]
    i_up = names1.index("expert.gate_up")
    assert names1[i_up:names1.index("expert.down") + 1] == ["expert.gate_up", "situ", "expert.down"]
    for banned in _BANNED:
        assert banned not in names0 and banned not in names1
    assert all(o.flops == 0 for o in ops0 + ops1 if o.kind == "vector")
    up = next(o for o in ops1 if o.name == "expert.gate_up")
    assert up.k == 3584 and up.n == 2 * 3072
    assert up.w_fmt == "mxfp4" and up.a_fmt == "bf16"
    assert round(up.w_bits, 6) == 4.270364
    kinds = {L.core.kind for L in m.layers}
    assert "linear" in kinds
    full_i = next(i for i, L in enumerate(m.layers) if L.core.kind != "linear" and i)
    full = _situs(build_rank_ops(m, full_i, ph, Shard()))
    assert len(full) == 2
    _situ(full[1], t * 16 * 3072, full_i)
    last = _situs(build_rank_ops(m, 92, ph, Shard()))
    _situ(last[1], t * 16 * 3072, 92)
    one = Phase("prefill", 1, 1, 0)
    total = 0.0
    sfu = 0.0
    for li in range(m.n_layers):
        for op in build_rank_ops(m, li, one, Shard()):
            if op.name == "situ":
                total += op.vec
                sfu += op.sfu
    assert total == float(_WIDTH) * 11
    assert sfu == float(_WIDTH) * 4
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(bare, ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert Scenario().model == "qwen3-8b"


def test_tp_request_fp8_and_cleared_kind():
    m = get_model(_MID)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    _raises(lambda: build_rank_ops(m, 1, ph, Shard(tp=2)), "must equal")
    ep = build_rank_ops(m, 1, ph, Shard(tp=2, ep=2))
    shared, routed = _situs(ep)
    _situ(shared, t * (6144 // 2), 1)
    _situ(routed, (t * 16 / 2) * 3072, 1)
    assert next(o for o in ep if o.name == "expert.down").k == 3072
    assert next(o for o in ep if o.name == "moe.shared_down").k == 6144 // 2
    etp = build_rank_ops(m, 1, ph, Shard(tp=2, etp=2))
    shared, routed = _situs(etp)
    _situ(shared, t * (6144 // 2), 1)
    _situ(routed, t * 16 * (3072 // 2), 1)
    assert next(o for o in etp if o.name == "expert.down").k == 3072 // 2
    fp8 = replace(m, act_fmt="fp8")
    got = build_rank_ops(fp8, 1, ph, Shard())
    for op in _situs(got):
        assert op.a_fmt == "fp32"
    _situ(_situs(got)[0], t * 6144, 1)
    _situ(_situs(got)[1], t * 16 * 3072, 1)
    gem = next(o for o in got if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "mxfp4"
    shared_up = next(o for o in got if o.name == "moe.shared_gate_up")
    assert shared_up.a_fmt == "fp8" and shared_up.w_fmt == "bf16"
    dense = build_rank_ops(fp8, 0, ph, Shard())
    assert _situs(dense)[0].a_fmt == "fp32"
    assert next(o for o in dense if o.name == "mlp.gate_up").a_fmt == "fp8"
    assert next(o for o in dense if o.name == "mlp.down").w_fmt == "bf16"
    bare = _bare(m)
    b0 = _names(build_rank_ops(bare, 0, ph, Shard()))
    b1 = _names(build_rank_ops(bare, 1, ph, Shard()))
    assert "situ" not in b0 and "situ" not in b1
    assert b0.index("mlp.gate_up") < b0.index("mlp.down") < b0.index("act")
    assert b1.index("expert.gate_up") < b1.index("expert.down") < b1.index("expert_act")
    assert "mlp_mul" not in b1
    act = next(o for o in build_rank_ops(bare, 0, ph, Shard()) if o.name == "act")
    assert act.vec == t * 33792 * 4 and act.sfu == t * 33792 * 2
    expert = next(o for o in build_rank_ops(bare, 1, ph, Shard()) if o.name == "expert_act")
    assert expert.vec == t * 16 * 3072 * 4 and expert.sfu == t * 16 * 3072 * 2
    assert _sched(m, ph, Shard(), 0) == _sched(bare, ph, Shard(), 0)
    assert _sched(m, ph, Shard(), 1) == _sched(bare, ph, Shard(), 1)
    assert _sched(m, ph, Shard(tp=2, ep=2), 1) == _sched(bare, ph, Shard(tp=2, ep=2), 1)
    assert _sched(m, ph, Shard(tp=2, etp=2), 1) == _sched(bare, ph, Shard(tp=2, etp=2), 1)


def test_other_rows_stay_untagged():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("qwen3-235b-a22b", "mixtral-8x22b", "gpt-oss-120b", "deepseek-v4-pro",
                "raon-1-mini-preview", "drama-3", "kimi-k2", "qwen3-8b"):
        m = get_model(mid)
        assert m.vec_kind != "kimi_situ"
        assert all(L.vec_kind != "kimi_situ" for L in m.layers)
        li = next((i for i, L in enumerate(m.layers) if L.ffn.kind == "moe"), 0)
        assert "situ" not in _names(build_rank_ops(m, li, ph, Shard()))
    qwen = get_model("qwen3.8-2.4t")
    qn = _names(build_rank_ops(qwen, 0, ph, Shard()))
    assert "situ" not in qn and "mlp_mul" in qn and "expert_act" in qn
    wide = get_model(_MID)
    other = get_model("qwen3-235b-a22b")
    cfg = _tc(load_release(_HF)["config"])
    spoof = replace(_bare(other), id=_MID)
    _raises(lambda: _tag_kimi_situ_vectors(spoof, cfg),
            "kimi situ depth is not num_hidden_layers")
    kept = replace(_bare(wide), id="qwen3-8b")
    assert _tag_kimi_situ_vectors(kept, cfg).vec_kind == ""


def test_published_flags_are_refused():
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "silu")),
            "kimi situ hidden_act is not situ")
    _raises(_bad(lambda c: c.__setitem__("model_type", "qwen3_moe")),
            "kimi situ model_type is not kimi_linear")
    _raises(_bad(lambda c: c.__setitem__("activation_situ_beta", 5.0)),
            "kimi situ beta is not 4 and 25")
    _raises(_bad(lambda c: c.__setitem__("activation_situ_linear_beta", 1.0)),
            "kimi situ beta is not 4 and 25")
    _raises(_bad(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "kimi situ swiglu_limit is set")
    _raises(_bad(lambda c: c.__setitem__("clamp", 1)),
            "kimi situ swiglu_limit is set")
    _raises(_bad(lambda c: c.__setitem__("first_k_dense_replace", 0)),
            "kimi situ first dense layer is not 1")
    _raises(_bad(lambda c: c.__setitem__("moe_intermediate_size", 0)),
            "kimi situ expert width is not positive")
    _raises(_bad(lambda c: c.__setitem__("latent_moe_use_norm", False)),
            "kimi situ latent norm flag is not true")
    m = get_model(_MID)
    cfg = _tc(load_release(_HF)["config"])
    _raises(lambda: _tag_kimi_situ_vectors(replace(m, mtp_layers=(m.layers[-1],)), cfg),
            "kimi situ mtp layers are present")
    _raises(lambda: _tag_kimi_situ_vectors(replace(_bare(m), layers=m.layers[:-1]), cfg),
            "kimi situ depth is not num_hidden_layers")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    _raises(lambda: _tag_kimi_situ_vectors(
        replace(_bare(m), layers=(ungated,) + m.layers[1:]), cfg),
        "kimi situ dense mlp is not swiglu")
    ph = Phase("prefill", 1, 8, 0)
    runtime = replace(m, layers=(replace(ungated, vec_kind="kimi_situ"),) + m.layers[1:])
    _raises(lambda: build_rank_ops(runtime, 0, ph, Shard()),
            "kimi situ mlp is not gate_up then down")
    no_shared = replace(
        m.layers[1],
        vec_kind="kimi_situ",
        ffn=replace(m.layers[1].ffn, n_shared=0))
    _raises(lambda: build_rank_ops(replace(m, layers=(m.layers[0], no_shared) + m.layers[2:]), 1, ph, Shard()),
            "kimi situ shared expert is not situ")
    dropped = replace(
        m.layers[1],
        vec_kind="kimi_situ",
        ffn=replace(m.layers[1].ffn, gated=False),
        ffn_linears=tuple(ln for ln in m.layers[1].ffn_linears if ln.name != "shared_gate_up"))
    _raises(lambda: build_rank_ops(replace(m, layers=(m.layers[0], dropped) + m.layers[2:]), 1, ph, Shard()),
            "kimi situ expert is not situ")
    _raises(lambda: _situ_op(0, -1.0), "kimi situ width is negative")
    loaded = from_release(_MID, _HF)
    assert loaded.vec_kind == "kimi_situ"
    assert _sched(loaded, ph, Shard(), 1) == _sched(_bare(loaded), ph, Shard(), 1)


def test_fingerprints_and_kpi_stay():
    from accel_dse.core import evaluate as E
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "afa0f7989c12811a"
    test_fingerprints_change_only_v4()
    pins = {
        "qwen3-32b": "b975b7b3760e",
        "qwen3-32b-fp8": "cb91adbc72c9",
        "qwen3-32b-awq": "3e44c63965e3",
        "qwen3-4b": "659c4bad45d8",
        "qwen3-0.6b": "e61d067c924b",
        "qwen3-8b": "11f6f3a5d21f",
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
