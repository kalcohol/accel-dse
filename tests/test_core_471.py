"""0.471: Muse Spark latent experts bill the published SwiGLU product.

The routed expert is down(silu(gate) * up) at moe_intermediate_size, after
the pooler. SiLU stays expert_act. The product is mlp_mul. a_fmt is the
published expert activation. There is no shared expert. Dense MTP drafts
stay untagged. vec_kind is muse_swiglu. No earlier allowlist is extended.
The six rows sit outside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _KIMI_SITU, _LLAMA3, _MUSE_SWIGLU, _QWEN35_MOE, _tag_kimi_situ_vectors,
    _tag_llama3_vectors, _tag_muse_swiglu_vectors, _tag_qwen35_moe_vectors,
    _tc, from_release, load_release,
)


# id, hf, expert act, expert weight, mtp dense intermediate (0 = no draft)
_ROWS = (
    ("muse-spark-open", "meta-models/Muse-Spark-Open-816B-A42B", "bf16", "bf16", 0),
    ("muse-spark-open-fp8", "amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8", "fp8", "fp8", 0),
    ("muse-spark-open-mxfp4", "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4", "mxfp4", "mxfp4", 0),
    ("muse-spark-open-nvfp4", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4", "nvfp4", "nvfp4", 0),
    ("muse-spark-open-nvfp4-dspark",
     "meta-models/Muse-Spark-Open-816B-A42B-NVFP4+Inferact/Muse-Spark-Open-DSpark",
     "nvfp4", "nvfp4", 16384),
    ("muse-spark-open-nvfp4-dflash2",
     "meta-models/Muse-Spark-Open-816B-A42B-NVFP4+nvidia/Muse-Spark-Open-816B-A42B-DFlash2",
     "nvfp4", "nvfp4", 24576),
)
_INDEX = (176, 177, 178, 179, 180, 218)
_HF = _ROWS[0][1]
_MID = _ROWS[0][0]
_H = 8192
_LAT = 4096
_DE = 4096
_DEPTH = 62
_EXP = 256
_TOPK = 8
_PARAMS = 819_989_536_256
_RELEASE = 819_990_552_064
_TEXT = ("expert.gate_up", "expert_act", "mlp_mul", "expert.down")
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "situ")
_BITS = {
    "muse-spark-open-fp8": 8.007812,
    "muse-spark-open-mxfp4": 4.629032,
    "muse-spark-open-nvfp4": 4.870971,
}


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


def _bad(edit, where: str = "text"):
    rel = json.loads(json.dumps(load_release(_HF)))
    if where == "text":
        edit(rel["config"]["text_config"])
    else:
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


def test_version_is_471():
    assert __version__ == "0.475.0"


def test_six_rows_bill_the_latent_product():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert tuple(ids.index(row[0]) for row in _ROWS) == _INDEX
    assert all(i >= 60 for i in _INDEX)
    assert _MUSE_SWIGLU == frozenset(row[0] for row in _ROWS)
    assert _MUSE_SWIGLU.isdisjoint(_QWEN35_MOE)
    assert _MUSE_SWIGLU.isdisjoint(_KIMI_SITU)
    assert _MUSE_SWIGLU.isdisjoint(_LLAMA3)
    assert _DEPTH * _TOPK * _DE == 2_031_616
    for mid, hf, af, wf, mtp_inner in _ROWS:
        m = get_model(mid)
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.act_fmt == "bf16" and m.vec_kind == "muse_swiglu"
        assert m.params() == _PARAMS and m.release_params == _RELEASE
        assert abs(m.param_check()["rel_err"] - ( _PARAMS - _RELEASE) / _RELEASE) < 1e-18
        assert m.param_check()["rel_err"] != 0.0
        assert all(L.vec_kind == "muse_swiglu" for L in m.layers)
        assert all(L.ffn.kind == "moe" and L.ffn.gated for L in m.layers)
        assert all(L.ffn.n_experts == _EXP and L.ffn.top_k == _TOPK for L in m.layers)
        assert all(L.ffn.d_expert == _DE and L.ffn.latent == _LAT and L.ffn.n_shared == 0
                   for L in m.layers)
        assert m.fmt("expert").fmt == wf
        assert m.act_fmt_for("expert") == af
        cfg = _tc(load_release(hf)["config"])
        assert cfg["model_type"] == "muse_spark_text"
        assert cfg["architectures"] == ["MuseSparkForConditionalGeneration"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["num_shared_experts"] == 0
        assert cfg["moe_hidden_size"] == _LAT and cfg["moe_intermediate_size"] == _DE
        assert cfg["mlp_layer_types"] == ["sparse"] * _DEPTH
        assert not any("swiglu" in k or "clamp" in k for k in cfg)
        assert any(n.startswith("0.471") for n in m.notes)
        if mtp_inner:
            assert len(m.mtp_layers) == 5
            assert all(L.vec_kind == "" and L.ffn.kind == "dense" and L.ffn.d_ff == mtp_inner
                       for L in m.mtp_layers)
        else:
            assert m.mtp_layers == ()
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        muls = [o for o in ops if o.name == "mlp_mul"]
        assert len(muls) == 1
        _mul(muls[0], t * _TOPK * _DE, 0, af)
        names = _names(ops)
        assert [n for n in names if n in _TEXT] == list(_TEXT)
        for banned in _BANNED:
            assert banned not in names
        act = next(o for o in ops if o.name == "expert_act")
        assert act.vec == t * _TOPK * _DE * 4 and act.sfu == float(t * _TOPK * _DE * 2)
        assert act.a_fmt == "bf16" and act.flops == 0
        up = next(o for o in ops if o.name == "expert.gate_up")
        down = next(o for o in ops if o.name == "expert.down")
        assert (up.k, up.n) == (_LAT, 2 * _DE)
        assert (down.k, down.n) == (_DE, _LAT)
        assert up.w_fmt == wf and up.a_fmt == af and down.a_fmt == af
        if mid in _BITS:
            assert round(up.w_bits, 6) == _BITS[mid]
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        bare = _bare(m)
        assert _tag_muse_swiglu_vectors(bare, cfg).vec_kind == "muse_swiglu"
        assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert get_model("qwen3.8-2.4t").vec_kind == "qwen35_moe"
    assert get_model("kimi-k3").vec_kind == "kimi_situ"
    assert get_model("llama-3.3-70b").vec_kind == "carbon_llama"
    assert get_model("mixtral-8x22b").vec_kind != "muse_swiglu"
    assert get_model("drama-3").vec_kind == ""


def test_default_window_product_and_drafts():
    m = get_model(_MID)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    mul = next(o for o in ops if o.name == "mlp_mul")
    _mul(mul, t * _TOPK * _DE, 0, "bf16")
    assert mul.vec == 4096 * 8 * 4096
    act = next(o for o in ops if o.name == "expert_act")
    assert act.vec == 4096 * 8 * 4096 * 4 and act.sfu == float(4096 * 8 * 4096 * 2)
    names = _names(ops)
    assert names.index("expert.gate_up") < names.index("expert_act") < names.index("mlp_mul")
    assert names.index("mlp_mul") < names.index("expert.down")
    assert "moe.shared_gate_up" not in names and "gelu" not in names
    full = build_rank_ops(m, 1, ph, Shard())
    assert m.layers[1].core.window is None
    _mul(next(o for o in full if o.name == "mlp_mul"), t * _TOPK * _DE, 1, "bf16")
    last = build_rank_ops(m, _DEPTH - 1, ph, Shard())
    _mul(next(o for o in last if o.name == "mlp_mul"), t * _TOPK * _DE, _DEPTH - 1, "bf16")
    head = next(o for o in head_ops(m, ph, Shard()) if o.name == "final_norm")
    bare_head = next(o for o in head_ops(_bare(m), ph, Shard()) if o.name == "final_norm")
    assert (head.vec, head.sfu, head.a_fmt) == (bare_head.vec, bare_head.sfu, bare_head.a_fmt)
    assert head.a_fmt == "bf16" and head.sfu == 0.0
    for mid, inner in (("muse-spark-open-nvfp4-dspark", 16384),
                       ("muse-spark-open-nvfp4-dflash2", 24576)):
        draft = get_model(mid)
        side = build_rank_ops(draft, 0, ph, Shard(), layer=draft.mtp_layers[0])
        assert "mlp_mul" not in _names(side)
        assert draft.mtp_layers[0].ffn.d_ff == inner
        assert draft.mtp_layers[0].vec_kind == ""


def test_tp_request_fp8_and_cleared_kind():
    m = get_model(_MID)
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    _raises(lambda: build_rank_ops(m, 0, ph, Shard(tp=2)),
            "ep·etp")
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
    assert gem.a_fmt == "fp8" and gem.w_fmt == "bf16"
    assert next(o for o in got if o.name == "expert_act").a_fmt == "bf16"
    for mid, af in (("muse-spark-open-fp8", "fp8"),
                    ("muse-spark-open-mxfp4", "mxfp4"),
                    ("muse-spark-open-nvfp4", "nvfp4")):
        pub = replace(get_model(mid), act_fmt="fp8")
        ops = build_rank_ops(pub, 0, Phase("prefill", 1, 16, 0), Shard())
        _mul(next(o for o in ops if o.name == "mlp_mul"), 16 * _TOPK * _DE, 0, af)
        up = next(o for o in ops if o.name == "expert.gate_up")
        assert up.a_fmt == af and up.w_fmt == af
        assert next(o for o in ops if o.name == "expert_act").a_fmt == "bf16"
    bare = _bare(m)
    bnames = _names(build_rank_ops(bare, 0, ph, Shard()))
    assert "mlp_mul" not in bnames
    assert bnames.index("expert.gate_up") < bnames.index("expert.down") < bnames.index("expert_act")
    assert _sched(m, ph, Shard()) == _sched(bare, ph, Shard())
    assert _sched(m, ph, Shard(tp=2, ep=2)) == _sched(bare, ph, Shard(tp=2, ep=2))
    assert _sched(m, ph, Shard(tp=2, etp=2)) == _sched(bare, ph, Shard(tp=2, etp=2))


def test_other_rows_and_refuses():
    ph = Phase("prefill", 1, 32, 0)
    for mid in ("mixtral-8x22b", "seed-oss-36b", "minimax-text-01", "glm-5.3",
                "qwen3-235b-a22b", "deepseek-v4-pro", "gpt-oss-120b", "drama-3",
                "qwen3-8b", "llama-3.1-8b"):
        m = get_model(mid)
        assert m.vec_kind != "muse_swiglu"
        assert all(L.vec_kind != "muse_swiglu" for L in m.layers)
    ernie = get_model("ernie-4.5-300b-a47b-pt")
    li = next(i for i, L in enumerate(ernie.layers) if L.ffn.kind == "moe")
    names = _names(build_rank_ops(ernie, li, ph, Shard()))
    assert "mlp_mul" not in names
    assert names.index("expert.gate_up") < names.index("expert.down") < names.index("expert_act")
    wide = get_model(_MID)
    other = get_model("mixtral-8x22b")
    c_ok = _tc(load_release(_HF)["config"])
    spoof = replace(_bare(other), id=_MID)
    _raises(lambda: _tag_muse_swiglu_vectors(spoof, c_ok),
            "muse swiglu depth is not num_hidden_layers")
    kept = replace(_bare(wide), id="mixtral-8x22b")
    assert _tag_muse_swiglu_vectors(kept, c_ok).vec_kind == ""
    assert _tag_qwen35_moe_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_kimi_situ_vectors(_bare(wide), c_ok).vec_kind == ""
    assert _tag_llama3_vectors(_bare(wide), c_ok).vec_kind == ""
    c_type = dict(c_ok)
    c_type["model_type"] = "qwen3_moe"
    _raises(lambda: _tag_muse_swiglu_vectors(_bare(wide), c_type),
            "muse swiglu model_type is not muse_spark_text")
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu")),
            "muse swiglu hidden_act is not silu")
    _raises(_bad(lambda c: c.__setitem__("swiglu_limit", 7.0)),
            "muse swiglu swiglu_limit is set")
    _raises(_bad(lambda c: c.__setitem__("num_shared_experts", 1)),
            "muse swiglu shared experts are not zero")
    _raises(_bad(lambda c: c.__setitem__("mlp_layer_types", ["dense"] * _DEPTH)),
            "muse swiglu mlp layers are not all sparse")
    _raises(_bad(lambda c: c.__setitem__("moe_intermediate_size", 0)),
            "muse swiglu expert width is not positive")
    _raises(_bad(lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"]), "outer"),
            "muse swiglu architecture is not muse")
    short = replace(_bare(wide), layers=wide.layers[:-1])
    _raises(lambda: _tag_muse_swiglu_vectors(short, c_ok),
            "muse swiglu depth is not num_hidden_layers")
    ungated = replace(wide.layers[0], ffn=replace(wide.layers[0].ffn, gated=False))
    broken = replace(_bare(wide), layers=(ungated,) + wide.layers[1:])
    _raises(lambda: _tag_muse_swiglu_vectors(broken, c_ok),
            "muse swiglu expert is not latent swiglu")
    dropped = replace(
        wide.layers[0],
        ffn_linears=tuple(ln for ln in wide.layers[0].ffn_linears if ln.name != "latent_up"))
    _raises(lambda: _tag_muse_swiglu_vectors(
        replace(_bare(wide), layers=(dropped,) + wide.layers[1:]), c_ok),
        "muse swiglu linears are not router and latent")
    wide_lat = replace(wide.layers[0].ffn_linears[1], n=_LAT + 1)
    # latent_up is the second linear in the published order.
    assert wide.layers[0].ffn_linears[1].name == "latent_up"
    bad_lat = replace(
        wide.layers[0],
        ffn_linears=(wide.layers[0].ffn_linears[0], wide_lat, wide.layers[0].ffn_linears[2]))
    _raises(lambda: _tag_muse_swiglu_vectors(
        replace(_bare(wide), layers=(bad_lat,) + wide.layers[1:]), c_ok),
        "muse swiglu latent width is not moe_hidden_size")
    draft = get_model("muse-spark-open-nvfp4-dspark")
    tagged_mtp = replace(draft.mtp_layers[0], vec_kind="qwen3")
    _raises(lambda: _tag_muse_swiglu_vectors(
        replace(_bare(draft), mtp_layers=(tagged_mtp,) + draft.mtp_layers[1:]),
        _tc(load_release(draft.hf_id)["config"])),
        "muse swiglu mtp is already tagged")
    moe_mtp = replace(draft.mtp_layers[0], ffn=replace(draft.mtp_layers[0].ffn, kind="moe"))
    _raises(lambda: _tag_muse_swiglu_vectors(
        replace(_bare(draft), mtp_layers=(moe_mtp,) + draft.mtp_layers[1:]),
        _tc(load_release(draft.hf_id)["config"])),
        "muse swiglu mtp is not dense")
    runtime = replace(
        wide.layers[0],
        vec_kind="muse_swiglu",
        ffn=replace(wide.layers[0].ffn, n_shared=1))
    _raises(lambda: build_rank_ops(replace(wide, layers=(runtime,) + wide.layers[1:]), 0, ph, Shard()),
            "muse swiglu expert is not latent swiglu")


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    # Muse is outside entries()[:60], so the locked decode hashes stay.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
    test_fingerprints_change_only_v4()
    from accel_dse.core import evaluate as E
    pins = {
        "qwen3-32b": "b975b7b3760e",
        "qwen3-8b": "11f6f3a5d21f",
        "llama-3.3-70b": "8e50c6cabdc8",
        "esmfold": "3bbede544308",
        "esm2-3b": "34f7544287cd",
        "wan2.1-1.3b": "309919c00604",
    }
    with E.model_flags(FUSE_QKV=False):
        for mid, pin in pins.items():
            assert _FULL_0651[mid] == pin
            assert _kpi(mid) == pin
