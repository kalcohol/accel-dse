"""0.468: Qwen3.5 MoE expert product is mlp_mul.

Qwen3_5MoeExperts and the shared Qwen3_5MoeMLP are down(silu(gate) * up).
SiLU on routed experts stays expert_act. The product is one multiply.
Shared SiLU, the sigmoid shared_expert_gate, and the routing-weight
multiply are not billed. vec_kind is qwen35_moe. No new act kind. The
0.449, 0.453, 0.465, and 0.467 allowlists are not extended. Vector flops
stay 0. GEMM names, shapes, and FLOPs stay put.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _QWEN3_449, _QWEN3_453, _QWEN3_465, _QWEN3_467, _QWEN35_MOE,
    _tag_qwen35_moe_vectors, _tc, from_release, load_release,
)


# id, hf, hidden, depth, experts, topk, d_expert, d_shared, act, expert w_fmt
_ROWS = (
    ("qwen3.8-2.4t", "Qwen/Qwen3.8-2.4T-A95B", 8192, 92, 512, 10, 2048, 2048, "bf16", "bf16"),
    ("qwen3.5-397b-a17b", "Qwen/Qwen3.5-397B-A17B", 4096, 60, 512, 10, 1024, 1024, "bf16", "bf16"),
    ("intern-s2-397b", "internlm/Intern-S2-397B", 4096, 60, 512, 10, 1024, 1024, "bf16", "bf16"),
    ("minicpm-v-4.7-35b-a3b", "openbmb/MiniCPM-V-4.7-35B-A3B", 2048, 40, 256, 8, 512, 512, "bf16", "bf16"),
    ("startlux-decision-v1-35b-a3b", "startlux-models/StartLux-Decision-v1-35B-A3B",
     2048, 40, 256, 8, 512, 512, "bf16", "bf16"),
    ("ornith-1.5-122b", "ornith-ai/Ornith-1.5-122B", 3072, 48, 256, 8, 1024, 1024, "bf16", "bf16"),
    ("ornith-1.5-122b-fp8", "ornith-ai/Ornith-1.5-122B-FP8", 3072, 48, 256, 8, 1024, 1024, "fp8", "fp8"),
    ("ornith-1.5-397b", "ornith-ai/Ornith-1.5-397B", 4096, 60, 512, 10, 1024, 1024, "bf16", "bf16"),
    ("ornith-1.5-397b-fp8", "ornith-ai/Ornith-1.5-397B-FP8", 4096, 60, 512, 10, 1024, 1024, "fp8", "fp8"),
    ("ornith-1.5-35b-a3b", "ornith-ai/Ornith-1.5-35B-A3B", 2048, 40, 256, 8, 512, 512, "bf16", "bf16"),
    ("ornith-1.5-35b-a3b-fp8", "ornith-ai/Ornith-1.5-35B-A3B-FP8", 2048, 40, 256, 8, 512, 512, "fp8", "fp8"),
    ("ornith-1.0-397b", "ornith-ai/Ornith-1.0-397B", 4096, 60, 512, 10, 1024, 1024, "bf16", "bf16"),
    ("ornith-1.0-397b-fp8", "ornith-ai/Ornith-1.0-397B-FP8", 4096, 60, 512, 10, 1024, 1024, "fp8", "fp8"),
    ("ornith-1.0-35b", "ornith-ai/Ornith-1.0-35B", 2048, 40, 256, 8, 512, 512, "bf16", "bf16"),
    ("ornith-1.0-35b-fp8", "ornith-ai/Ornith-1.0-35B-FP8", 2048, 40, 256, 8, 512, 512, "fp8", "fp8"),
)
_FRONT = ("qwen3.8-2.4t", "qwen3.5-397b-a17b")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul")
_TEXT = (
    "moe.shared_gate_up", "mlp_mul", "moe.shared_down", "moe.shared_expert_gate",
    "expert.gate_up", "expert_act", "mlp_mul", "expert.down",
)


def _names(ops):
    return [o.name for o in ops]


def _muls(ops):
    return [o for o in ops if o.name == "mlp_mul"]


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


def _text(rel: dict) -> dict:
    cfg = rel["config"]
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc.get("hidden_size") else cfg


def _bad(hf: str, mid: str, edit):
    rel = json.loads(json.dumps(load_release(hf)))
    edit(_text(rel))
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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.count,
                            o.w_fmt, o.a_fmt, o.share, o.replicated))
    return out


def test_version_is_468():
    assert __version__ == "0.471.0"


def test_fifteen_rows_bill_the_published_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert [ids.index(mid) for mid in _FRONT] == [6, 25]
    assert _QWEN35_MOE == frozenset(row[0] for row in _ROWS)
    assert _QWEN35_MOE.isdisjoint(_QWEN3_449)
    assert _QWEN35_MOE.isdisjoint(_QWEN3_453)
    assert _QWEN35_MOE.isdisjoint(_QWEN3_465)
    assert _QWEN35_MOE.isdisjoint(_QWEN3_467)
    for mid, hf, h, depth, n_exp, topk, de, dsh, af, wf in _ROWS:
        m = from_release(mid, hf)
        assert m.hf_id == hf and m.hidden == h and m.n_layers == depth
        assert m.act_fmt == af and m.vec_kind == "qwen35_moe"
        assert m.param_check()["rel_err"] == 0.0
        assert m.params() == m.release_params
        assert all(L.vec_kind == "qwen35_moe" for L in m.layers)
        assert all(L.vec_kind == "" for L in m.mtp_layers)
        assert all(L.ffn.kind == "moe" and L.ffn.gated for L in m.layers)
        assert all(L.ffn.n_experts == n_exp and L.ffn.top_k == topk for L in m.layers)
        assert all(L.ffn.d_expert == de and L.ffn.n_shared == 1 and L.ffn.d_shared == dsh
                   for L in m.layers)
        assert m.fmt("expert").fmt == wf
        cfg = _text(load_release(hf))
        assert cfg["model_type"] == "qwen3_5_moe_text"
        assert cfg["hidden_act"] == "silu"
        assert cfg["moe_intermediate_size"] == de
        assert cfg["shared_expert_intermediate_size"] == dsh
        assert not cfg.get("mlp_only_layers") and not cfg.get("first_k_dense_replace")
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        assert any(n.startswith("0.468") for n in m.notes)
        bare = _bare(m)
        assert _tag_qwen35_moe_vectors(bare, cfg).vec_kind == "qwen35_moe"
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        muls = _muls(ops)
        assert len(muls) == 2
        _mul(muls[0], t * dsh, 0, af)
        _mul(muls[1], t * topk * de, 0, af)
        names = _names(ops)
        assert [n for n in names if n in _TEXT or n == "mlp_mul"][-len(_TEXT):] == list(_TEXT)
        for banned in _BANNED:
            assert banned not in names
        act = next(o for o in ops if o.name == "expert_act")
        assert act.vec == t * topk * de * 4 and act.sfu == float(t * topk * de * 2)
        assert act.a_fmt == "bf16" and act.flops == 0
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        up = next(o for o in ops if o.name == "expert.gate_up")
        assert up.w_fmt == wf and up.a_fmt == af
        if mid == "qwen3.8-2.4t":
            assert ids.index(mid) < 60
        if mid.endswith("-fp8"):
            assert up.w_bits != 8.0 and round(up.w_bits, 6) > 8.0
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert get_model("drama-3").vec_kind == ""


def test_default_window_qwen38_product():
    m = get_model("qwen3.8-2.4t")
    t, depth, topk, de, dsh = 4096, 92, 10, 2048, 2048
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    shared, routed = _muls(ops)
    _mul(shared, t * dsh, 0, "bf16")
    _mul(routed, t * topk * de, 0, "bf16")
    assert shared.vec == 4096 * 2048
    assert routed.vec == 4096 * 10 * 2048
    names = _names(ops)
    assert names.index("moe.shared_gate_up") < names.index("mlp_mul") < names.index("moe.shared_down")
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("expert.down")
    assert names.index("expert_act") < names[names.index("expert_act") + 1:].index("mlp_mul") + names.index("expert_act") + 1
    assert "gelu" not in names
    full = build_rank_ops(m, 3, ph, Shard())
    assert m.layers[3].core.kind == "gqa"
    fm = _muls(full)
    _mul(fm[0], t * dsh, 3, "bf16")
    _mul(fm[1], t * topk * de, 3, "bf16")
    last = _muls(build_rank_ops(m, depth - 1, ph, Shard()))
    _mul(last[1], t * topk * de, depth - 1, "bf16")
    side = build_rank_ops(m, 0, ph, Shard(), layer=m.mtp_layers[0])
    assert "mlp_mul" not in _names(side)
    assert _names(side).index("expert.down") < _names(side).index("expert_act")
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(_bare(m), ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    small = get_model("qwen3.5-397b-a17b")
    sm = _muls(build_rank_ops(small, 0, ph, Shard()))
    _mul(sm[0], 4096 * 1024, 0, "bf16")
    _mul(sm[1], 4096 * 10 * 1024, 0, "bf16")
    per_tok = depth * (topk * de + dsh)
    assert per_tok == 2_072_576
    assert 60 * (10 * 1024 + 1024) == 675_840


def test_tp_request_fp8_and_cleared_kind():
    m = get_model("qwen3.8-2.4t")
    t, topk, de, dsh = 4096, 10, 2048, 2048
    ph = Phase("prefill", 1, t, 0)
    ep = build_rank_ops(m, 0, ph, Shard(tp=2, ep=2))
    em = _muls(ep)
    _mul(em[0], t * (dsh // 2), 0, "bf16")
    _mul(em[1], (t * topk / 2) * de, 0, "bf16")
    down = next(o for o in ep if o.name == "expert.down")
    shared_down = next(o for o in ep if o.name == "moe.shared_down")
    assert down.k == de and shared_down.k == dsh // 2
    etp = build_rank_ops(m, 0, ph, Shard(tp=2, etp=2))
    tm = _muls(etp)
    _mul(tm[0], t * (dsh // 2), 0, "bf16")
    _mul(tm[1], t * topk * (de // 2), 0, "bf16")
    assert next(o for o in etp if o.name == "expert.down").k == de // 2
    fp8 = replace(m, act_fmt="fp8")
    got = _muls(build_rank_ops(fp8, 0, ph, Shard()))
    _mul(got[0], t * dsh, 0, "fp8")
    _mul(got[1], t * topk * de, 0, "fp8")
    gem = next(o for o in build_rank_ops(fp8, 0, ph, Shard()) if o.name == "expert.gate_up")
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16"
    pub = get_model("ornith-1.5-397b-fp8")
    published = build_rank_ops(pub, 0, Phase("prefill", 1, 16, 0), Shard())
    pm = _muls(published)
    _mul(pm[0], 16 * 1024, 0, "fp8")
    _mul(pm[1], 16 * 10 * 1024, 0, "fp8")
    eg = next(o for o in published if o.name == "expert.gate_up")
    assert eg.w_fmt == "fp8" and eg.a_fmt == "fp8"
    assert round(eg.w_bits, 6) == 8.007812
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_moe_rows_stay_untagged():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("qwen3-235b-a22b", "mixtral-8x22b", "gpt-oss-120b", "deepseek-v4-pro",
                "kimi-k3", "drama-3", "qwen3.8-27b", "qwen3-8b"):
        m = get_model(mid)
        assert m.vec_kind != "qwen35_moe"
        li = next(i for i, L in enumerate(m.layers) if L.ffn.kind == "moe") if any(
            L.ffn.kind == "moe" for L in m.layers) else 0
        assert "mlp_mul" not in _names(build_rank_ops(m, li, ph, Shard())) or m.vec_kind in (
            "qwen3", "carbon_llama", "qwen2_text")
    ernie = get_model("ernie-4.5-300b-a47b-pt")
    li = next(i for i, L in enumerate(ernie.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(ernie, li, ph, Shard()))
    assert "mlp_mul" not in names
    assert names.index("expert.gate_up") < names.index("expert.down") < names.index("expert_act")
    wide = get_model("qwen3.8-2.4t")
    other = get_model("qwen3-235b-a22b")
    spoof = replace(_bare(other), id="qwen3.8-2.4t")
    _raises(lambda: _tag_qwen35_moe_vectors(spoof, _tc(load_release(wide.hf_id)["config"])),
            "qwen35 moe expert is not swiglu")
    _raises(lambda: _tag_qwen35_moe_vectors(spoof, _tc(load_release(other.hf_id)["config"])),
            "qwen35 moe model_type is not qwen3_5_moe_text")
    kept = replace(_bare(wide), id="qwen3-235b-a22b")
    assert _tag_qwen35_moe_vectors(kept, _tc(load_release(wide.hf_id)["config"])).vec_kind == ""


def test_published_flags_are_refused():
    hf, mid = "Qwen/Qwen3.8-2.4T-A95B", "qwen3.8-2.4t"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen35 moe hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "qwen3_moe")),
            "qwen35 moe model_type is not qwen3_5_moe_text")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "qwen35 moe swiglu_limit is set")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_only_layers", [0])),
            "qwen35 moe mlp_only_layers is not empty")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("moe_intermediate_size", 0)),
            "qwen35 moe expert width is not positive")
    m = get_model(mid)
    c_ok = _tc(load_release(hf)["config"])
    c_dense = dict(c_ok)
    c_dense["first_k_dense_replace"] = 1
    _raises(lambda: _tag_qwen35_moe_vectors(_bare(m), c_dense),
            "qwen35 moe first_k_dense_replace is set")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken = replace(_bare(m), layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_qwen35_moe_vectors(broken, c_ok),
            "qwen35 moe expert is not swiglu")
    ph = Phase("prefill", 1, 8, 0)
    tagged = replace(ungated, vec_kind="qwen35_moe")
    runtime = replace(m, layers=(tagged,) + m.layers[1:])
    _raises(lambda: build_rank_ops(runtime, 0, ph, Shard()),
            "qwen35 moe shared expert is not swiglu")
    no_shared = replace(
        m.layers[0],
        vec_kind="qwen35_moe",
        ffn=replace(m.layers[0].ffn, gated=False),
        ffn_linears=tuple(l for l in m.layers[0].ffn_linears if l.name != "shared_gate_up"))
    _raises(lambda: build_rank_ops(replace(m, layers=(no_shared,) + m.layers[1:]), 0, ph, Shard()),
            "qwen35 moe expert is not swiglu")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "qwen35_moe"
    assert _sched(loaded, ph, Shard()) == _sched(_bare(loaded), ph, Shard())


def test_fingerprints_and_kpi_stay():
    from accel_dse.core import evaluate as E
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "19cac027be730bc4"
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
