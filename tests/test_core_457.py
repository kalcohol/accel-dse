"""0.457: Moshi temporal towers bill published RMSNorm and the SwiGLU product.

rms_norm_f32 is the T5 recipe in fp32. SiLU stays the existing act op. The
gate times up product is mlp_mul. There is no q/k RMSNorm. layer_scale is
null, so the residual add stays the existing one. The two catalog rows share
one temporal decoder. Weights stay bf16. Activations stay bf16. final_norm
covers every prompt token. The depformer and the Mimi codec stay on their
existing graphs. These ids are outside the first 60 catalog rows. Vector
flops stay 0. vec_kind reuses carbon_llama.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import RoleFormat, _tag_moshi_vectors, _tc, from_release, load_release
from accel_dse.core.moshi import depformer_ops, mimi_decode_ops, mimi_encode_ops


_ROWS = {
    "moshiko-7b": ("kyutai/moshiko-pytorch-bf16", "2026-10-20"),
    "moshika-7b": ("kyutai/moshika-pytorch-bf16", "2026-10-23"),
}
_H, _INNER, _DEPTH, _HEADS, _HD = 4096, 11264, 32, 32, 128
_VOCAB = 32000
_PARAMS = 6_973_366_272
_MISC = 2 * _H
_EXTRA = 360_448
_HAND = 36_283_883_716_608
_TENSORS = 673
_BYTES = 15_760_063_876
_BYTES_LLM = 13_946_732_544
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


def _bad(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"]["lm"])
    return lambda: from_release(mid, hf, rel)


def _bad_root(mid: str, edit):
    hf = _ROWS[mid][0]
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel["config"])
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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_457():
    assert __version__ == "0.479.0"


def test_moshi_temporal_bills_fp32_rmsnorm_and_the_product():
    for mid, row in _ROWS.items():
        hf, fetched = row
        m = from_release(mid, hf)
        rel = load_release(hf)
        lm = rel["config"]["lm"]
        assert rel["fetched"] == fetched and rel["n_tensors"] == _TENSORS
        assert rel["bytes_total"] == _BYTES and rel["bytes_llm"] == _BYTES_LLM
        assert lm["norm"] == "rms_norm_f32" and lm["gating"] == "silu"
        assert lm["hidden_scale"] == 4.125 and lm["context"] == 3000
        assert lm["positional_embedding"] == "rope" and lm["causal"] is True
        assert "layer_scale" not in lm and lm["num_heads"] == _HEADS
        assert m.hf_id == hf and m.hidden == _H and m.n_layers == _DEPTH
        assert m.vocab == _VOCAB and m.domain == "llm" and m.vec_kind == "carbon_llama"
        assert all(L.vec_kind == "carbon_llama" for L in m.layers)
        assert m.tie_embeddings is False and m.act_fmt == "bf16"
        assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.params() == m.release_params == _PARAMS and m.param_check()["rel_err"] == 0.0
        assert m.final_norm_params is None
        assert len(m.layers) == _DEPTH and m.layers[0].misc_params == _MISC
        assert m.layers[0].ffn.d_ff == _INNER and m.layers[0].core.qk_dim == _HD
        assert m.layers[0].core.window == 3000 and m.layers[0].core.n_q == _HEADS
        assert any("0.457" in n for n in m.notes)
        ph = Phase("prefill", 1, 4096, 0)
        op = _by(build_rank_ops(m, 0, ph, Shard()))
        _act(op["act"], 4096 * _INNER, 0, "bf16")
        _mul(op["mlp_mul"], 4096 * _INNER, 0, "bf16")
        _rms(op["attn_norm"], 4096, _H, 1, 0)
        _rms(op["ffn_norm"], 4096, _H, 1, 0)
        assert "q_rms" not in op and "k_rms" not in op
        assert op["mlp.gate_up"].m == 4096 and op["mlp.gate_up"].k == _H
        assert op["mlp.gate_up"].n == 2 * _INNER and op["mlp.gate_up"].w_fmt == "bf16"
        assert op["mlp.gate_up"].w_bits == 16.0 and op["mlp.gate_up"].a_fmt == "bf16"
        assert op["mlp.down"].m == 4096 and op["mlp.down"].k == _INNER and op["mlp.down"].n == _H
        names = [o.name for o in build_rank_ops(m, 0, ph, Shard())]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("qk") < names.index("rope") < names.index("ffn_norm")
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
        assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
        for banned in _BANNED:
            assert banned not in names
        head = _by(head_ops(m, ph, Shard()))
        _rms(head["final_norm"], 4096, _H, 1, _DEPTH)
        side = [o.name for o in depformer_ops(m.moshi, 1)]
        side += [o.name for o in mimi_encode_ops(m.moshi, 1)]
        side += [o.name for o in mimi_decode_ops(m.moshi, 1)]
        assert "mlp_mul" not in side
        tp = _by(build_rank_ops(m, 0, ph, Shard(tp=2)))
        _mul(tp["mlp_mul"], 4096 * (_INNER // 2), 0, "bf16")
        _act(tp["act"], 4096 * (_INNER // 2), 0, "bf16")
        assert tp["mlp.gate_up"].n == _INNER and tp["mlp.down"].k == _INNER // 2
        assert tp["attn_norm"].vec == 4096 * _H * 4 and tp["attn_norm"].replicated == 1
        tnames = [o.name for o in build_rank_ops(m, 0, ph, Shard(tp=2))]
        assert tnames.index("attn.o") < tnames.index("attn_allreduce") < tnames.index("ffn_norm")
        assert tnames.index("mlp.down") < tnames.index("mlp_allreduce")
        assert "q_rms" not in tnames and "k_rms" not in tnames
        fp8 = replace(m, act_fmt="fp8")
        fop = _by(build_rank_ops(fp8, 0, ph, Shard()))
        _act(fop["act"], 4096 * _INNER, 0, "fp8")
        _mul(fop["mlp_mul"], 4096 * _INNER, 0, "fp8")
        assert fop["attn_norm"].a_fmt == "fp32" and fop["ffn_norm"].a_fmt == "fp32"
        assert fop["mlp.gate_up"].a_fmt == "fp8" and fop["mlp.gate_up"].w_fmt == "bf16"
        assert fop["mlp.down"].w_bits == 16.0
        fhead = _by(head_ops(fp8, ph, Shard()))
        assert fhead["final_norm"].a_fmt == "fp32" and fhead["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_other_models():
    ph = Phase("prefill", 1, 256, 0)
    for mid in _ROWS:
        m = get_model(mid)
        cleared = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
    for mid in ("qwen3-32b", "qwen3-8b", "qwen3-1.7b", "qwen2.5-72b", "llama-3.1-8b",
                "olmo-3-32b-think", "jais-3-32b",
                "qwen3-vl-235b-a22b", "qwen3-coder-480b-a35b",
                "qwen3-tts-1.7b-base", "qwen3-asr-1.7b", "fun-asr-nano-2512",
                "ernie-4.5-0.3b-pt", "omniasr-llm-7b-v2"):
        other = get_model(mid)
        assert not any("0.457" in n for n in other.notes)
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert get_model("qwen3-32b").vec_kind == "qwen3"
    assert get_model("llama-3.1-8b").vec_kind == ""
    assert get_model("omniasr-llm-7b-v2").vec_kind == "carbon_llama"
    assert not any("0.457" in n for n in get_model("omniasr-llm-7b-v2").notes)
    carbon = get_model("carbon-8b")
    assert carbon.vec_kind == "carbon_llama"
    assert any("0.445" in n for n in carbon.notes)
    assert not any("0.457" in n for n in carbon.notes)
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("moshiko-7b") == 267 and ids.index("moshika-7b") == 268
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
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

    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "f8346734e98af4a2"
    for mid in _ROWS:
        assert mid not in ids[:60]
        assert mid not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    mid = "moshiko-7b"
    _raises(_bad(mid, lambda c: c.__setitem__("gating", "gelu")), "moshi gating is not silu")
    _raises(_bad(mid, lambda c: c.__setitem__("norm", "layer_norm")), "moshi norm is not rms_norm_f32")
    _raises(_bad(mid, lambda c: c.__setitem__("layer_scale", 0.01)), "moshi layer_scale is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("positional_embedding", "sin")),
            "moshi positional_embedding is not rope")
    _raises(_bad(mid, lambda c: c.__setitem__("causal", False)), "moshi causal is not true")
    _raises(_bad(mid, lambda c: c.__setitem__("attention_bias", True)), "moshi attention_bias is not false")
    _raises(_bad(mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "moshi rope_scaling is not null")
    _raises(_bad(mid, lambda c: c.__setitem__("hidden_scale", 4.0)),
            "moshi hidden_scale is not 4.125")
    _raises(_bad(mid, lambda c: c.__setitem__("context", 128)), "moshi attention window is not 3000")
    _raises(_bad(mid, lambda c: c.__setitem__("num_heads", 7)),
            "moshi hidden is not heads times head_dim")
    _raises(_bad_root(mid, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "moshi architecture is not moshi")
    _raises(_bad("moshika-7b", lambda c: c.__setitem__("gating", "relu")), "moshi gating is not silu")
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    rel = load_release(_ROWS[mid][0])
    c = _tc(rel["config"])
    cleared = _bare(m)
    fp16 = replace(cleared, act_fmt="fp16")
    _raises(lambda: _tag_moshi_vectors(fp16, c), "moshi activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("fp32", 32.0) if k == "mlp" else v) for k, v in m.formats)
    wide = replace(cleared, formats=fmts)
    _raises(lambda: _tag_moshi_vectors(wide, c), "moshi weight dtype is not bf16")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1, vec_kind="")
    broken_m = replace(m, layers=(extra,) + tuple(replace(L, vec_kind="") for L in m.layers[1:]), vec_kind="")
    _raises(lambda: _tag_moshi_vectors(broken_m, c), "moshi norm is not two hidden")
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "carbon mlp is not gate_up then down")
    cleared_g = replace(broken_g, layers=tuple(replace(L, vec_kind="") for L in broken_g.layers), vec_kind="")
    _raises(lambda: _tag_moshi_vectors(cleared_g, c), "moshi mlp is not dense swiglu")
    q = m.layers[0].attn_linears[0]
    qwide = replace(q, n=q.n + 1)
    bad_w = replace(m.layers[0], attn_linears=(qwide,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(bad_w,) + m.layers[1:])
    _raises(lambda: _tag_moshi_vectors(
        replace(broken_w, layers=tuple(replace(L, vec_kind="") for L in broken_w.layers), vec_kind=""), c),
        "moshi qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_moshi_vectors(
        replace(broken_win, layers=tuple(replace(L, vec_kind="") for L in broken_win.layers), vec_kind=""), c),
        "moshi attention is not gqa head_dim")
    two = replace(cleared, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_moshi_vectors(two, c), "moshi final norm is not one")
    bad_c = dict(c)
    bad_c["model_type"] = "llama"
    _raises(lambda: _tag_moshi_vectors(cleared, bad_c), "moshi model_type is not moshi")
    bad_types = dict(c)
    bad_types["layer_types"] = ["full_attention"] * len(c["layer_types"])
    _raises(lambda: _tag_moshi_vectors(cleared, bad_types), "moshi layer_types is not sliding_attention")
    bad_w = dict(c)
    bad_w["intermediate_size"] = c["intermediate_size"] + 1
    _raises(lambda: _tag_moshi_vectors(cleared, bad_w), "moshi swiglu width is not the published gating")
    other = get_model("llama-3.1-8b")
    assert _tag_moshi_vectors(other, _tc(load_release(other.hf_id)["config"])).vec_kind == ""
    assert _tag_moshi_vectors(get_model("carbon-8b"), c).vec_kind == "carbon_llama"


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    expect = _DEPTH * (2 * t * _H * (2 * _INNER) + 2 * t * _INNER * _H)
    assert expect == _HAND
    for mid in _ROWS:
        m = get_model(mid)
        got = 0.0
        extra = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
                if o.name == "mlp_mul":
                    extra += o.vec
        assert got == _HAND
        assert extra == t * _EXTRA
