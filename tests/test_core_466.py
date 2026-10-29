"""0.466: front-60 Qwen2.5-3B and Qwen2.5-1.5B bill qwen2_text.

Qwen2ForCausalLM uses Qwen2RMSNorm and Qwen2MLP, the same classes as
Qwen2.5-72B. The reduction is the T5 recipe in fp32. SiLU stays the
existing act op. The gate times up product is mlp_mul. Qwen2Attention
hardcodes q/k/v bias and has no q/k norm. q width equals hidden.
use_sliding_window is false, so the JSON sliding_window key does not open a
layer window. vec_kind reuses qwen2_text. These ids are inside the first 60
catalog rows. The 0.452 / 0.455 allowlist and the 0.463 allowlist are not
extended. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import (
    RoleFormat, _QWEN25_466, _QWEN25_72B, _QWEN2_TEXT,
    _tag_qwen25_466_vectors, _tag_qwen25_72b_vectors, _tag_qwen2_text_vectors,
    _tc, from_release, load_release,
)


# id, hf, hidden, inner, depth, q heads, kv heads, head_dim, params, norm,
# misc, max_window_layers, ffn gemm sum at prompt 4096
_ROWS = (
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", 2048, 11008, 36, 16, 2, 128,
     3_085_938_688, 149_504, 6656, 70, 19_945_828_122_624),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1536, 8960, 28, 12, 2, 128,
     1_543_714_304, 87_552, 5120, 21, 9_470_402_887_680),
)
_BANNED = ("gelu", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu",
           "gelu_new", "mask_mul", "q_rms", "k_rms")
_ORDER = (
    "attn_norm", "attn.qkv", "attn.o", "qk", "pv", "softmax", "rope",
    "ffn_norm", "mlp.gate_up", "act", "mlp_mul", "mlp.down", "residual",
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


def _bad(hf: str, mid: str, edit):
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
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops,
                            o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_466():
    assert __version__ == "0.467.0"


def test_two_rows_are_front60_and_reuse_qwen2_text():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids[49] == "qwen2.5-3b" and ids[50] == "qwen2.5-1.5b"
    assert _QWEN25_466 == frozenset({"qwen2.5-3b", "qwen2.5-1.5b"})
    assert _QWEN25_466.isdisjoint(_QWEN2_TEXT)
    assert _QWEN25_466.isdisjoint(_QWEN25_72B)
    assert _QWEN25_72B == frozenset({"qwen2.5-72b"})
    for mid, hf, h, inner, depth, hq, hkv, hd, params, norm, misc, mwl, _ffn in _ROWS:
        assert ids.index(mid) < 60
        assert mid in _QWEN25_466 and mid not in _QWEN2_TEXT and mid not in _QWEN25_72B
        m = from_release(mid, hf)
        assert m.hf_id == hf
        assert m.params() == m.release_params == params
        assert m.param_check()["rel_err"] == 0.0
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.tie_embeddings is True and m.vocab == 151936
        assert m.final_norm_params is None
        assert m.vec_kind == "qwen2_text"
        assert all(L.vec_kind == "qwen2_text" for L in m.layers)
        assert all(L.misc_params == misc for L in m.layers)
        assert misc == 2 * h + (hq + 2 * hkv) * hd
        assert norm == depth * (2 * h) + h
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        assert all(L.core.kind == "gqa" and L.core.qk_dim == hd for L in m.layers)
        assert all(L.core.n_q == hq and L.core.n_kv == hkv for L in m.layers)
        assert all(L.core.window is None and L.core.no_rope is False for L in m.layers)
        assert hq * hd == h
        assert depth * inner == {36: 396_288, 28: 250_880}[depth]
        by_lin = {ln.name: ln for ln in m.layers[0].attn_linears}
        assert (by_lin["q"].k, by_lin["q"].n) == (h, h)
        assert (by_lin["k"].k, by_lin["k"].n) == (h, hkv * hd)
        assert (by_lin["v"].k, by_lin["v"].n) == (h, hkv * hd)
        assert (by_lin["o"].k, by_lin["o"].n) == (h, h)
        assert m.fmt("mlp").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
        assert m.fmt("mlp").bits == 16.0 and m.fmt("attn").bits == 16.0
        cfg = load_release(hf)["config"]
        assert cfg["model_type"] == "qwen2"
        assert cfg["architectures"] == ["Qwen2ForCausalLM"]
        assert cfg["hidden_act"] == "silu"
        assert cfg["use_sliding_window"] is False
        assert cfg["sliding_window"] == 32768
        assert cfg["max_window_layers"] == mwl
        assert cfg["attention_dropout"] == 0.0
        assert cfg["tie_word_embeddings"] is True
        assert cfg["torch_dtype"] == "bfloat16"
        assert cfg["transformers_version"] == "4.43.1"
        assert "attention_bias" not in cfg and "mlp_bias" not in cfg
        assert "rope_scaling" not in cfg and "head_dim" not in cfg
        assert "layer_types" not in cfg and "use_mrope" not in cfg
        assert cfg["hidden_size"] == cfg["num_attention_heads"] * hd
        roles = load_release(hf)["roles"]["norm"]["bf16"]
        assert roles["params"] == norm
        assert any(n.startswith("0.466") for n in m.notes)
        assert not any(n.startswith("0.452") or n.startswith("0.455") or n.startswith("0.463")
                       for n in m.notes)
        assert not any("tied lm_head" in n for n in m.notes)
        bare = _bare(m)
        c = _tc(cfg)
        assert _tag_qwen2_text_vectors(bare, c).vec_kind == ""
        assert _tag_qwen25_72b_vectors(bare, c).vec_kind == ""
        assert _tag_qwen25_466_vectors(bare, c).vec_kind == "qwen2_text"
        t = 64
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _act(by["act"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
        _rms(by["attn_norm"], t, h, 0)
        _rms(by["ffn_norm"], t, h, 0)
        names = [o.name for o in ops]
        for banned in _BANNED:
            assert banned not in names
        assert [n for n in _ORDER if n in names] == list(_ORDER)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        up = by["mlp.gate_up"]
        assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    big = get_model("qwen2.5-72b")
    assert big.vec_kind == "qwen2_text"
    assert any("0.463" in n for n in big.notes)
    assert not any("0.466" in n for n in big.notes)
    assert _tag_qwen25_466_vectors(_bare(big), _tc(load_release(big.hf_id)["config"])).vec_kind == ""


def test_published_sliding_window_key_does_not_open_a_window():
    for mid, hf, *_rest in _ROWS:
        rel = json.loads(json.dumps(load_release(hf)))
        rel["config"]["sliding_window"] = 1024
        rel["config"]["attention_bias"] = True
        m = from_release(mid, hf, rel)
        assert m.vec_kind == "qwen2_text"
        assert all(L.core.window is None for L in m.layers)
        assert all(L.vec_kind == "qwen2_text" for L in m.layers)


def test_default_window_pre_norm_order():
    m = get_model("qwen2.5-3b")
    h, inner, depth = 2048, 11008, 36
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _act(by["act"], t * inner, 0, "bf16")
    _mul(by["mlp_mul"], t * inner, 0, "bf16")
    _rms(by["attn_norm"], t, h, 0)
    _rms(by["ffn_norm"], t, h, 0)
    assert by["mlp_mul"].vec == 4096 * 11008
    assert by["attn_norm"].vec == 4096 * 2048 * 4 and by["attn_norm"].sfu == 4096
    up, down = by["mlp.gate_up"], by["mlp.down"]
    assert (up.m, up.k, up.n) == (t, h, 2 * inner)
    assert (down.m, down.k, down.n) == (t, inner, h)
    assert up.flops == 2 * t * h * (2 * inner) and down.flops == 2 * t * inner * h
    assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
    names = [o.name for o in ops]
    assert names.index("attn_norm") < names.index("attn.qkv") < names.index("attn.o")
    assert names.index("attn.o") < names.index("qk") < names.index("rope") < names.index("ffn_norm")
    assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("act")
    assert names.index("act") < names.index("mlp_mul") < names.index("mlp.down") < names.index("residual")
    assert "q_rms" not in names and "k_rms" not in names and "gelu" not in names
    last = _by(build_rank_ops(m, depth - 1, ph, Shard()))
    _act(last["act"], t * inner, depth - 1, "bf16")
    _mul(last["mlp_mul"], t * inner, depth - 1, "bf16")
    head = _by(head_ops(m, ph, Shard()))
    _rms(head["final_norm"], t, h, depth)
    small = get_model("qwen2.5-1.5b")
    top = _by(head_ops(small, Phase("prefill", 1, 16, 0), Shard()))
    _rms(top["final_norm"], 16, 1536, 28)
    assert top["final_norm"].vec == 16 * 1536 * 4
    bare = _bare(m)
    bare_ops = build_rank_ops(bare, 0, ph, Shard())
    assert "mlp_mul" not in [o.name for o in bare_ops]
    assert by["residual"].vec == _by(bare_ops)["residual"].vec
    assert _by(bare_ops)["attn_norm"].a_fmt == "bf16"
    assert _by(bare_ops)["ffn_norm"].a_fmt == "bf16"
    bare_head = _by(head_ops(bare, ph, Shard()))
    assert bare_head["final_norm"].a_fmt == "bf16" and bare_head["final_norm"].sfu == 0.0
    assert all(o.flops == 0 for o in ops if o.kind == "vector")


def test_decode_batch_tp_and_request_fp8():
    m = get_model("qwen2.5-3b")
    h, inner = 2048, 11008
    dec = _by(build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard()))
    _act(dec["act"], inner, 0, "bf16")
    _mul(dec["mlp_mul"], inner, 0, "bf16")
    assert dec["mlp.gate_up"].m == 1 and dec["ffn_norm"].sfu == 1.0
    assert dec["attn_norm"].sfu == 1.0
    assert "q_rms" not in dec and "k_rms" not in dec
    two = _by(build_rank_ops(m, 0, Phase("decode", 2, 1, 8192), Shard()))
    _act(two["act"], 2 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * inner, 0, "bf16")
    pre = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _act(pre["act"], 8192 * inner, 0, "bf16")
    assert pre["mlp.gate_up"].m == 8192
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    half = inner // 2
    _act(loc["act"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == half
    assert loc["ffn_norm"].vec == 4096 * h * 4 and loc["mlp_mul"].replicated == 1
    assert loc["attn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("attn_norm") < names.index("attn.qkv")
    assert names.index("attn.o") < names.index("attn_allreduce")
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    assert names.index("ffn_norm") < names.index("mlp.gate_up")
    assert "q_rms" not in names and "k_rms" not in names
    small = _by(build_rank_ops(get_model("qwen2.5-1.5b"), 0, Phase("prefill", 1, 4096, 0), sh))
    _mul(small["mlp_mul"], 4096 * 4480, 0, "bf16")
    assert small["mlp.gate_up"].n == 8960 and small["mlp.down"].k == 4480
    assert small["attn_norm"].vec == 4096 * 1536 * 4
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _act(op["act"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "fp32" and op["ffn_norm"].a_fmt == "fp32"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    head = _by(head_ops(fp8, Phase("prefill", 1, 4096, 0), Shard()))
    assert head["final_norm"].a_fmt == "fp32" and head["final_norm"].sfu == 4096.0


def test_non_vector_schedule_and_other_rows_stay_untagged():
    ph = Phase("prefill", 1, 256, 0)
    for mid, hf, *_rest in _ROWS:
        m = get_model(mid)
        bare = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(bare, ph, shard)
        c = _tc(load_release(hf)["config"])
        retagged = _tag_qwen25_466_vectors(bare, c)
        assert retagged.vec_kind == "qwen2_text"
        assert all(L.vec_kind == "qwen2_text" for L in retagged.layers)
    # 0.467 bills the head-width mismatch rows. This gate still does not claim them.
    for mid, hidden, heads in (("qwen3-32b", 5120, 64), ("qwen3-4b", 2560, 32), ("qwen3-0.6b", 1024, 16)):
        other = get_model(mid)
        assert other.vec_kind == "qwen3"
        assert other.hidden == hidden and other.layers[0].core.n_q == heads
        assert other.hidden != heads * other.layers[0].core.qk_dim
        assert _tag_qwen25_466_vectors(
            _bare(other), _tc(load_release(other.hf_id)["config"])).vec_kind == ""
    llama = get_model("llama-3.1-8b")
    assert llama.vec_kind == ""
    assert load_release(llama.hf_id)["config"].get("rope_scaling") is not None
    spoof = replace(llama, id="qwen2.5-3b")
    _raises(lambda: _tag_qwen25_466_vectors(spoof, _tc(load_release(llama.hf_id)["config"])),
            "qwen2.5 small model_type is not qwen2")
    assert get_model("qwen2.5-omni-7b").vec_kind == "qwen2_text"
    assert any("0.452" in n for n in get_model("qwen2.5-omni-7b").notes)
    assert not any("0.466" in n for n in get_model("qwen2.5-omni-7b").notes)
    assert get_model("mistral-large").vec_kind == "mistral_small"
    assert get_model("llama-2-70b").vec_kind == "carbon_llama"
    assert get_model("qwen3-8b").vec_kind == "qwen3"
    assert not any("0.466" in n for n in get_model("qwen3-8b").notes)


def test_published_flags_and_linear_shape_are_refused():
    hf, mid = "Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3b"
    _raises(_bad(hf, mid, lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2.5 small hidden_act is not silu")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("model_type", "llama")),
            "qwen2.5 small model_type is not qwen2")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("architectures", ["LlamaForCausalLM"])),
            "qwen2.5 small architecture is not qwen2")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_dropout", 0.1)),
            "qwen2.5 small attention_dropout is not 0")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("attention_bias", False)),
            "qwen2.5 small attention_bias is not true")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("mlp_bias", True)),
            "qwen2.5 small mlp_bias is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_sliding_window", True)),
            "qwen2.5 small use_sliding_window is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("use_mrope", True)),
            "qwen2.5 small use_mrope is not false")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("layer_types", ["sliding_attention"])),
            "qwen2.5 small layer_types is not full_attention")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("rope_scaling", {"rope_type": "yarn"})),
            "qwen2.5 small rope_scaling is not null")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("head_dim", 64)),
            "qwen2.5 small head_dim is not hidden over heads")
    _raises(_bad(hf, mid, lambda c: c.__setitem__("num_attention_heads", 12)),
            "qwen2.5 small hidden is not heads times head_dim")
    _raises(_bad("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b",
                 lambda c: c.__setitem__("hidden_act", "gelu")),
            "qwen2.5 small hidden_act is not silu")
    loaded = from_release(mid, hf)
    assert loaded.vec_kind == "qwen2_text"
    c_ok = _tc(load_release(hf)["config"])
    m = get_model(mid)
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()),
            "qwen2 text mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: _tag_qwen25_466_vectors(broken_g, c_ok),
            "qwen2.5 small mlp is not dense swiglu")
    extra = replace(m.layers[0], misc_params=m.layers[0].misc_params + 1)
    broken_m = replace(m, layers=(extra,) + m.layers[1:])
    _raises(lambda: _tag_qwen25_466_vectors(broken_m, c_ok),
            "qwen2.5 small bias is not qkv")
    wide = replace(m.layers[0], core=replace(m.layers[0].core, qk_dim=m.layers[0].core.qk_dim + 1))
    broken_h = replace(m, layers=(wide,) + m.layers[1:])
    _raises(lambda: _tag_qwen25_466_vectors(broken_h, c_ok),
            "qwen2.5 small attention is not gqa head_dim")
    q = m.layers[0].attn_linears[0]
    wide_q = replace(q, n=q.n + 128)
    broken_w = replace(m, layers=(replace(m.layers[0], attn_linears=(wide_q,) + m.layers[0].attn_linears[1:]),)
                       + m.layers[1:])
    _raises(lambda: _tag_qwen25_466_vectors(broken_w, c_ok),
            "qwen2.5 small qk width is not heads times head_dim")
    window = replace(m.layers[0], core=replace(m.layers[0].core, window=128))
    broken_win = replace(m, layers=(window,) + m.layers[1:])
    _raises(lambda: _tag_qwen25_466_vectors(broken_win, c_ok),
            "qwen2.5 small attention is not gqa head_dim")
    two = replace(m, final_norm_params=2 * m.hidden)
    _raises(lambda: _tag_qwen25_466_vectors(two, c_ok),
            "qwen2.5 small final norm is not one")
    fp32 = replace(m, act_fmt="fp32")
    _raises(lambda: _tag_qwen25_466_vectors(fp32, c_ok),
            "qwen2.5 small activation dtype is not bf16")
    fmts = tuple((k, RoleFormat("fp32", 32.0) if k == "mlp" else v) for k, v in m.formats)
    wide_w = replace(_bare(m), formats=fmts)
    _raises(lambda: _tag_qwen25_466_vectors(wide_w, c_ok),
            "qwen2.5 small weight dtype is not bf16")


def test_ffn_gemm_flops_match_the_closed_form():
    t = 4096
    ph = Phase("prefill", 1, t, 0)
    for mid, _hf, h, inner, depth, *_rest, hand in _ROWS:
        expect = depth * (2 * t * h * (2 * inner) + 2 * t * inner * h)
        assert expect == hand
        m = get_model(mid)
        got = 0.0
        for li in range(m.n_layers):
            for o in build_rank_ops(m, li, ph, Shard()):
                if o.name in ("mlp.gate_up", "mlp.down"):
                    got += o.flops
                if o.kind == "vector":
                    assert o.flops == 0
        assert got == expect


def test_fingerprints_and_kpi_stay():
    from tests.test_core_0100 import _fp
    from tests.test_core_125 import test_fingerprints_change_only_v4
    from tests.test_core_coverage import _FULL_0651, _kpi
    from accel_dse.core import evaluate as E
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "7d44cdadbf38cc9e"
    test_fingerprints_change_only_v4()
    with E.model_flags(FUSE_QKV=False):
        assert _kpi("qwen2.5-3b") == "8b8d7db5d3eb"
        assert _kpi("qwen2.5-1.5b") == "45cf41450b64"
        assert _kpi("qwen2.5-72b") == "51b90d8f5e68"
        assert _kpi("mistral-large") == "4ce7bca4c910"
        assert _kpi("yi-1.5-34b") == "4a9848721f34"
        assert _kpi("qwen3-8b") == "11f6f3a5d21f"
        assert _kpi("qwen3-8b-fp8") == "25b06545a766"
        assert _kpi("qwen3-8b-awq") == "d0c28036e214"
        assert _kpi("qwen3-1.7b") == "c062686f3bf2"
    assert _FULL_0651["qwen2.5-3b"] == "8b8d7db5d3eb"
    assert _FULL_0651["qwen2.5-1.5b"] == "45cf41450b64"
    assert _FULL_0651["qwen2.5-72b"] == "51b90d8f5e68"
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
    assert _kpi("esmfold") == "3bbede544308"
    assert _kpi("esm2-3b") == "34f7544287cd"
    assert _kpi("wan2.1-1.3b") == "309919c00604"
