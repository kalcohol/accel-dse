"""0.462: Mistral-Large bills published RMSNorm and the SwiGLU product.

MistralForCausalLM uses MistralRMSNorm and MistralMLP, the same classes as
the Mistral-Small text tower. The reduction is the T5 recipe in fp32. SiLU
stays the existing act op. The gate times up product is mlp_mul. There is
no q/k norm. q width equals hidden. vec_kind reuses mistral_small. This id
is inside the first 60 catalog rows. carbon_llama still requires model_type
llama and does not accept the checkpoint. Vector flops stay 0.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    _CARBON_LLAMA, _MISTRAL_LARGE, _MISTRAL_SMALL, _tag_carbon_vectors,
    _tag_mistral_large_vectors, _tag_mistral_small_vectors, _tc, from_release,
    load_release,
)


_MID = "mistral-large"
_HF = "mistralai/Mistral-Large-Instruct-2411"
_H, _INNER, _DEPTH, _HQ, _HKV, _HD = 12288, 28672, 88, 96, 8, 128
_PARAMS = 122_610_069_504
_NORM = _DEPTH * (2 * _H) + _H
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul")
_FFN = 761_961_558_048_768
# Front-60 rows that stay on the shared tail. Smaller than this row, or the
# published config fails an existing gate.
_UNTOUCHED = (
    # 0.463 bills qwen2.5-72b with vec_kind qwen2_text.
    # 0.464 bills llama-2-70b, llama-2-7b, and yi-1.5-34b with carbon_llama.
    # 0.466 bills qwen2.5-3b and qwen2.5-1.5b with qwen2_text.
    "llama-3.1-8b",
    # 0.465 bills qwen3-8b, its fp8 and AWQ twins, and qwen3-1.7b with qwen3.
    # 0.467 bills qwen3-32b, its fp8 and AWQ twins, qwen3-4b, and qwen3-0.6b.
    "mistral-7b", "mixtral-8x22b", "magistral-small",
)


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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops,
                            o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_462():
    assert __version__ == "0.471.0"


def test_mistral_large_is_the_front60_row_and_reuses_mistral_small():
    ids = [e["id"] for e in entries()]
    assert ids.index(_MID) < 60
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert _MID in _MISTRAL_LARGE and _MID not in _MISTRAL_SMALL
    assert _MID not in _CARBON_LLAMA
    m = get_model(_MID)
    assert m.hf_id == _HF
    assert m.params() == m.release_params == _PARAMS
    assert m.param_check()["rel_err"] == 0.0
    assert m.n_layers == _DEPTH and m.hidden == _H and m.act_fmt == "bf16"
    assert m.tie_embeddings is False and m.vocab == 32768
    assert m.vec_kind == "mistral_small"
    assert all(L.vec_kind == "mistral_small" for L in m.layers)
    assert all(L.misc_params == 2 * _H for L in m.layers)
    assert all(L.ffn.gated and L.ffn.d_ff == _INNER for L in m.layers)
    assert all(L.core.kind == "gqa" and L.core.qk_dim == _HD for L in m.layers)
    assert all(L.core.n_q == _HQ and L.core.n_kv == _HKV for L in m.layers)
    assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
    assert _HQ * _HD == _H
    assert _DEPTH * _INNER == 2_523_136
    by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
    assert (by_lin["q"].k, by_lin["q"].n) == (_H, _H)
    assert (by_lin["k"].k, by_lin["k"].n) == (_H, _HKV * _HD)
    assert (by_lin["v"].k, by_lin["v"].n) == (_H, _HKV * _HD)
    assert (by_lin["o"].k, by_lin["o"].n) == (_H, _H)
    cfg = load_release(_HF)["config"]
    assert cfg["model_type"] == "mistral"
    assert cfg["architectures"] == ["MistralForCausalLM"]
    assert cfg["hidden_act"] == "silu"
    assert cfg["sliding_window"] is None
    assert cfg["head_dim"] == _HD
    assert cfg["hidden_size"] == cfg["num_attention_heads"] * cfg["head_dim"]
    assert "rope_scaling" not in cfg
    assert "attention_bias" not in cfg and "mlp_bias" not in cfg
    assert "layer_types" not in cfg and "use_sliding_window" not in cfg
    # carbon_llama's published-field gate does not accept this config.
    assert cfg.get("model_type") != "llama"
    assert cfg.get("attention_bias") is not False
    assert cfg.get("mlp_bias") is not False
    assert cfg.get("pretraining_tp") != 1
    roles = load_release(_HF)["roles"]["norm"]["bf16"]
    assert roles["params"] == _NORM
    assert any("0.462" in n for n in m.notes)
    assert "按 SiLU" not in " ".join(m.notes)


def test_default_window_pre_norm_order():
    m = get_model(_MID)
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
    assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    assert down.w_fmt == "bf16" and down.w_bits == 16.0
    names = [o.name for o in ops]
    assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
    assert names.index("qk") < names.index("rope") < names.index("ffn_norm")
    assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down")
    assert names.index("mlp.down") < names.index("residual")
    assert "q_rms" not in names and "k_rms" not in names
    assert "gelu" not in names
    for banned in _BANNED:
        assert banned not in names
    last = _by(build_rank_ops(m, _DEPTH - 1, ph, Shard()))
    _act(last["act"], t * _INNER, _DEPTH - 1, "bf16")
    _mul(last["mlp_mul"], t * _INNER, _DEPTH - 1, "bf16")
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, _H, _DEPTH)
    two = _by(head_ops(m, Phase("prefill", 2, t, 0), Shard()))
    _rms(two["final_norm"], 2 * t, _H, _DEPTH)
    bare = _bare(m)
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    assert "mlp_mul" not in [o.name for o in bare_ops]
    assert by["residual"].vec == _by(bare_ops)["residual"].vec
    assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
    assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
    bare_head = _by(head_ops(bare, ph, Shard()))
    assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
    assert all(o.flops == 0 for o in ops if o.kind == "vector")


def test_decode_batch_tp_and_fp8():
    m = get_model(_MID)
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


def test_non_vector_schedule_and_other_rows_stay_untagged():
    m = get_model(_MID)
    ph = Phase("prefill", 1, 256, 0)
    bare = _bare(m)
    for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    c = _tc(load_release(_HF)["config"])
    assert _tag_mistral_small_vectors(bare, c).vec_kind == ""
    assert _tag_carbon_vectors(bare, c).vec_kind == ""
    retagged = _tag_mistral_large_vectors(bare, c)
    assert retagged.vec_kind == "mistral_small"
    assert all(L.vec_kind == "mistral_small" for L in retagged.layers)
    for mid in _UNTOUCHED:
        other = get_model(mid)
        assert other.vec_kind == "", mid
        assert all(L.vec_kind == "" for L in other.layers), mid
        pops = build_rank_ops(other, 0, Phase("prefill", 1, 16, 0), Shard())
        assert "mlp_mul" not in [o.name for o in pops], mid
    small = get_model("mistral-small-3.2-24b-instruct-2506")
    assert small.vec_kind == "mistral_small"
    assert small.layers[0].attn_linears[0].n != small.hidden
    assert get_model("qwen3-14b").vec_kind == "qwen3"
    assert get_model("carbon-8b").vec_kind == "carbon_llama"


def test_published_flags_and_linear_shape_are_refused():
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu")),
            "mistral large hidden_act is not silu")
    _raises(_bad(lambda c: c.__setitem__("model_type", "llama")),
            "mistral large model_type is not mistral")
    _raises(_bad(lambda c: c.__setitem__("architectures", ["Mistral3ForConditionalGeneration"])),
            "mistral large architecture is not mistral")
    _raises(_bad(lambda c: c.__setitem__("attention_dropout", 0.1)),
            "mistral large attention_dropout is not 0")
    _raises(_bad(lambda c: c.__setitem__("attention_bias", True)),
            "mistral large attention_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("mlp_bias", True)),
            "mistral large mlp_bias is not false")
    _raises(_bad(lambda c: c.__setitem__("sliding_window", 4096)),
            "mistral large sliding_window is not null")
    _raises(_bad(lambda c: c.__setitem__("use_sliding_window", True)),
            "mistral large use_sliding_window is not false")
    _raises(_bad(lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "mistral large layer_types is not full_attention")
    _raises(_bad(lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "mistral large rope_scaling is not null")
    _raises(_bad(lambda c: c.__setitem__("head_dim", 64)),
            "mistral large hidden is not heads times head_dim")
    _raises(_bad(lambda c: c.pop("head_dim")), "mistral large head_dim is missing")
    loaded = from_release(_MID, _HF)
    assert loaded.vec_kind == "mistral_small"
    c_ok = _tc(load_release(_HF)["config"])
    m = get_model(_MID)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()),
            "mistral small mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()),
            "mistral small mlp is not gate_up then down")
    _raises(lambda: _tag_mistral_large_vectors(broken_g, c_ok),
            "mistral large mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_mistral_large_vectors(broken_m, c_ok),
            "mistral large has no qk norm")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_mistral_large_vectors(broken_h, c_ok),
            "mistral large attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + _HD)
    broken_q = replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:])
    broken_w = replace(m, layers=(broken_q,) + m.layers[1:])
    _raises(lambda: _tag_mistral_large_vectors(broken_w, c_ok),
            "mistral large qk width is not heads times head_dim")
    other = get_model("mistral-7b")
    assert _tag_mistral_large_vectors(other, _tc(load_release(other.hf_id)["config"])).vec_kind == ""
    fp32 = replace(m, act_fmt="fp32")
    _raises(lambda: _tag_mistral_large_vectors(fp32, c_ok),
            "mistral large activation dtype is not bf16")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    expect = _DEPTH * (2 * t * _H * (2 * _INNER) + 2 * t * _INNER * _H)
    assert expect == _FFN
    m = get_model(_MID)
    got = 0.0
    for li in range(m.n_layers):
        for o in build_rank_ops(m, li, ph, Shard()):
            if o.name in ("mlp.gate_up", "mlp.down"):
                got += o.flops
            if o.kind == "vector":
                assert o.flops == 0
    assert got == _FFN


def _fpx(top):
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


def test_fingerprints_and_kpi():
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651, _kpi
    # Stage stays f54f33dfa13cdd41. Default pipe moves.
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "19cac027be730bc4"
    assert _fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e"
    assert _fpx({}) == "0f2c94a6e5524bc5"
    # The locked KPI is the unfused wide stage path (pin_module_unfused). Those
    # 10-digit fields do not move. The default fused pipe fingerprint does.
    from accel_dse.core import evaluate as E
    with E.model_flags(FUSE_QKV=False):
        assert _kpi("mistral-large") == "4ce7bca4c910"
    assert _FULL_0651["mistral-large"] == "4ce7bca4c910"
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
    assert _kpi("esmfold") == "3bbede544308"
    assert _kpi("esm2-3b") == "34f7544287cd"
    assert _kpi("wan2.1-1.3b") == "309919c00604"
