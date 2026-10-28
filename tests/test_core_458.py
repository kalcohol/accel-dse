"""0.458: ERNIE-4.5 MoE dense layers bill published Llama RMSNorm and the product.

Ernie4_5_MoeRMSNorm subclasses LlamaRMSNorm. SiLU stays the existing act
op. The gate times up product is mlp_mul on dense gate_up/down layers only.
MoE layers keep expert_act. There is no q/k RMSNorm. vec_kind reuses
carbon_llama. The 0.3B row and the VL rows stay untagged. These six ids are
outside the first 60 catalog rows. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import RoleFormat, _tag_ernie_moe_vectors, from_release, load_release


_WIDE = ("ernie-4.5-300b-a47b-pt", "ernie-4.5-300b-a47b-base-pt", "ernie-4.5-300b-a47b-base")
_NARROW = ("ernie-4.5-21b-a3b-pt", "ernie-4.5-21b-a3b-base-pt", "ernie-4.5-21b-a3b-thinking")
_SHAPE = {
    "ernie-4.5-300b-a47b-pt": (8192, 28672, 54, 3, 64, 8, 128, 299_484_163_264, 1),
    "ernie-4.5-300b-a47b-base-pt": (8192, 28672, 54, 3, 64, 8, 128, 299_484_163_264, 0),
    "ernie-4.5-300b-a47b-base": (8192, 28672, 54, 3, 64, 8, 128, 299_484_163_264, 0),
    "ernie-4.5-21b-a3b-pt": (2560, 12288, 28, 1, 20, 4, 128, 21_825_437_888, 1),
    "ernie-4.5-21b-a3b-base-pt": (2560, 12288, 28, 1, 20, 4, 128, 21_825_437_888, 0),
    "ernie-4.5-21b-a3b-thinking": (2560, 12288, 28, 1, 20, 4, 128, 21_825_437_888, 0),
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


def test_version_is_458():
    assert __version__ == "0.458.0"


def test_ernie_moe_dense_bills_fp32_rmsnorm_and_the_product():
    for mid, shape in _SHAPE.items():
        h, inner, depth, n_dense, heads, kv, hd, params, n_mtp = shape
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.hidden == h and m.n_layers == depth and m.vec_kind == "carbon_llama"
        assert m.params() == m.release_params == params and m.param_check()["rel_err"] == 0.0
        assert m.act_fmt == "bf16" and m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.fmt("expert").fmt == "bf16"
        assert rel["config"]["hidden_act"] == "silu" and rel["config"]["use_bias"] is False
        assert rel["config"].get("rope_scaling") is None
        assert rel["config"]["model_type"] == "ernie4_5_moe"
        assert len(m.mtp_layers) == n_mtp
        assert all(L.vec_kind == "" for L in m.mtp_layers)
        assert all(L.vec_kind == "carbon_llama" for L in m.layers)
        assert any("0.458" in n for n in m.notes)
        assert sum(L.ffn.kind == "dense" for L in m.layers) == n_dense
        assert m.layers[0].ffn.d_ff == inner and m.layers[0].misc_params == 2 * h
        assert m.layers[0].core.qk_dim == hd and m.layers[0].core.n_q == heads
        assert m.layers[0].core.n_kv == kv and m.layers[n_dense].ffn.kind == "moe"
        ph = Phase("prefill", 1, 4096, 0)
        op = _by(build_rank_ops(m, 0, ph, Shard()))
        _act(op["act"], 4096 * inner, 0, "bf16")
        _mul(op["mlp_mul"], 4096 * inner, 0, "bf16")
        _rms(op["attn_norm"], 4096, h, 1, 0)
        _rms(op["ffn_norm"], 4096, h, 1, 0)
        assert "q_rms" not in op and "k_rms" not in op
        assert op["mlp.gate_up"].m == 4096 and op["mlp.gate_up"].k == h
        assert op["mlp.gate_up"].n == 2 * inner and op["mlp.gate_up"].w_fmt == "bf16"
        assert op["mlp.gate_up"].w_bits == 16.0 and op["mlp.gate_up"].a_fmt == "bf16"
        assert op["mlp.down"].m == 4096 and op["mlp.down"].k == inner and op["mlp.down"].n == h
        names = [o.name for o in build_rank_ops(m, 0, ph, Shard())]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("qk") < names.index("rope") < names.index("ffn_norm")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        for banned in _BANNED:
            assert banned not in names
        moe = _by(build_rank_ops(m, n_dense, ph, Shard()))
        assert "mlp_mul" not in moe and "expert_act" in moe
        _rms(moe["attn_norm"], 4096, h, 1, n_dense)
        _rms(moe["ffn_norm"], 4096, h, 1, n_dense)
        moe_names = [o.name for o in build_rank_ops(m, n_dense, ph, Shard())]
        if mid in _NARROW:
            i = moe_names.index("moe.shared_gate_up")
            assert moe_names[i + 1] == "moe.shared_down"
        else:
            assert not any(n.startswith("moe.shared") for n in moe_names)
        assert moe_names.index("expert.gate_up") < moe_names.index("expert.down") < moe_names.index("expert_act")
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], 4096, h, 1, depth)
        if n_mtp:
            side = [o.name for o in build_rank_ops(m, 0, ph, Shard(), layer=m.mtp_layers[0])]
            assert "mlp_mul" not in side
        tp = _by(build_rank_ops(m, 0, ph, Shard(tp=2)))
        _mul(tp["mlp_mul"], 4096 * (inner // 2), 0, "bf16")
        _act(tp["act"], 4096 * (inner // 2), 0, "bf16")
        assert tp["mlp.gate_up"].n == inner and tp["mlp.down"].k == inner // 2
        assert tp["attn_norm"].vec == 4096 * h * 4 and tp["attn_norm"].replicated == 1
        fp8 = replace(m, act_fmt="fp8")
        fop = _by(build_rank_ops(fp8, 0, ph, Shard()))
        _act(fop["act"], 4096 * inner, 0, "fp8")
        _mul(fop["mlp_mul"], 4096 * inner, 0, "fp8")
        assert fop["attn_norm"].a_fmt == "fp32" and fop["ffn_norm"].a_fmt == "fp32"
        assert fop["mlp.gate_up"].a_fmt == "fp8" and fop["mlp.gate_up"].w_fmt == "bf16"
        assert fop["mlp.down"].w_bits == 16.0
        fhead = _by(head_ops(fp8, ph, Shard()))
        assert fhead["final_norm"].a_fmt == "fp32" and fhead["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_other_models():
    ph = Phase("prefill", 1, 256, 0)
    for mid in _SHAPE:
        m = get_model(mid)
        cleared = _bare(m)
        # MoE layers require ep * etp == tp * dp. Dense FFN still splits on tp.
        for shard in (Shard(), Shard(tp=2, ep=2)):
            assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "qwen2.5-72b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-asr-1.7b", "fun-asr-nano-2512",
                "ernie-4.5-0.3b-pt", "ernie-4.5-vl-424b-a47b-pt", "moshiko-7b"):
        other = get_model(mid)
        assert not any("0.458" in n for n in other.notes)
    assert get_model("ernie-4.5-0.3b-pt").vec_kind == ""
    assert get_model("ernie-4.5-vl-28b-a3b-pt").vec_kind == ""
    assert get_model("qwen3-32b").vec_kind == ""
    assert get_model("qwen3-8b").vec_kind == ""
    assert get_model("moshiko-7b").vec_kind == "carbon_llama"
    assert not any("0.458" in n for n in get_model("moshiko-7b").notes)
    carbon = get_model("carbon-8b")
    assert carbon.vec_kind == "carbon_llama"
    assert any("0.445" in n for n in carbon.notes)
    assert not any("0.458" in n for n in carbon.notes)
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("ernie-4.5-21b-a3b-pt") == 117
    assert ids.index("ernie-4.5-300b-a47b-base") == 122
    assert "ernie-4.5-300b-a47b-pt" not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "1e15ae85430a04f3"
    import hashlib
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario

    def fpx(top):
        h = hashlib.sha256()
        for e in entries()[:60]:
            if e["id"].startswith("deepseek-v4"):
                continue
            for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}},
                          {"layout": {"tp": 2}}):
                try:
                    h.update(json.dumps(evaluate(Scenario.from_dict({"model": e["id"], **top, **extra})).summary(),
                                        sort_keys=True, default=str).encode())
                except Exception as ex:  # noqa: BLE001
                    h.update(type(ex).__name__.encode())
        return h.hexdigest()[:16]

    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "8a5fa1d0881da387"
    for mid in _SHAPE:
        assert mid not in ids[:60]
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "ernie-4.5-300b-a47b-base-pt"
    m = get_model(mid)
    rel = load_release(m.hf_id)
    c = rel["config"]
    cleared = _bare(m)

    def bad(edit):
        cfg = json.loads(json.dumps(c))
        edit(cfg)
        return lambda: from_release(mid, m.hf_id, rel, cfg)

    _raises(bad(lambda cfg: cfg.__setitem__("hidden_act", "gelu")), "ernie moe hidden_act is not silu")
    _raises(bad(lambda cfg: cfg.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "ernie moe rope_scaling is not null")
    _raises(bad(lambda cfg: cfg.__setitem__("architectures", ["LlamaForCausalLM"])),
            "ernie moe architecture is not ernie4_5_moe")
    _raises(bad(lambda cfg: cfg.__setitem__("head_dim", 64)), "ernie moe head_dim is not hidden // heads")
    _raises(bad(lambda cfg: cfg.__setitem__("use_bias", True)), "use_bias is not modeled")
    fp16 = replace(cleared, act_fmt="fp16")
    _raises(lambda: _tag_ernie_moe_vectors(fp16, c), "ernie moe activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("fp32", 32.0) if k == "mlp" else v) for k, v in m.formats)
    wide = replace(cleared, formats=fmts)
    _raises(lambda: _tag_ernie_moe_vectors(wide, c), "ernie moe weight dtype is not bf16")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1, vec_kind="")
    broken_m = replace(m, layers=(extra,) + tuple(replace(L, vec_kind="") for L in m.layers[1:]), vec_kind="")
    _raises(lambda: _tag_ernie_moe_vectors(broken_m, c), "ernie moe norm is not two hidden")
    q = m.layers[0].attn_linears[0]
    qwide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(qwide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: _tag_ernie_moe_vectors(
        replace(broken_w, layers=tuple(replace(L, vec_kind="") for L in broken_w.layers), vec_kind=""), c),
        "ernie moe qk width is not heads times head_dim")
    bad_c = dict(c)
    bad_c["model_type"] = "llama"
    _raises(lambda: _tag_ernie_moe_vectors(cleared, bad_c), "ernie moe model_type is not ernie4_5_moe")
    two = replace(cleared, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_ernie_moe_vectors(two, c), "ernie moe final norm is not one")
    other = get_model("llama-3.1-8b")
    assert _tag_ernie_moe_vectors(other, c).vec_kind == ""
    assert _tag_ernie_moe_vectors(get_model("ernie-4.5-0.3b-pt"), c).vec_kind == ""
    assert not any("0.458" in n for n in _tag_ernie_moe_vectors(get_model("carbon-8b"), c).notes)


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, shape in _SHAPE.items():
        h, inner, _depth, n_dense, *_rest = shape
        expect = n_dense * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        m = get_model(mid)
        got = 0.0
        extra = 0.0
        for li, L in enumerate(m.layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.kind == "vector":
                    assert o.flops == 0
                if L.ffn.kind == "dense" and o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.name == "mlp_mul":
                    extra += o.vec
                    assert L.ffn.kind == "dense"
        assert got == expect
        assert extra == t * n_dense * inner
    assert 3 * 28672 == 86_016
    assert 1 * 12288 == 12_288
