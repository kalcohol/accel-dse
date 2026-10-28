"""0.437: ESM-2 encoder LayerNorm and erf-GELU.

transformers modeling_esm.py: pre-LN nn.LayerNorm, gelu = x * 0.5 * (1 + erf(x / sqrt(2))),
query scaled by head_dim**-0.5, rotary, no QK-norm. MLM head is dense, the same gelu,
LayerNorm, then the tied decoder plus bias. Vector flops stay 0. GEMM shapes stay put.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import build_esm
from accel_dse.core.ir import Phase, Shard, build_rank_ops, full_io_ops
from accel_dse.core.model import NativeWorkload, load_release


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _ln(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _gelu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release("facebook/esm2_t33_650M_UR50D")))
    edit(rel["config"])
    wl = NativeWorkload("protein", seq_len=8)
    return lambda: build_esm("x", "facebook/esm2_t33_650M_UR50D", rel, wl)


def test_version_is_437():
    assert __version__ == "0.448.0"


def test_esm2_650m_bills_layernorm_and_erf_gelu():
    m = get_model("esm2-650m")
    assert m.vec_kind == "esm2" and m.params() == m.release_params
    assert m.layers[0].vec_kind == "esm2" and m.layers[0].core.qk_norm is False
    assert m.layers[0].core.q_scale is True and m.layers[0].ffn.gated is False
    t, h, inter, heads, depth, vocab = 514, 1280, 5120, 20, 33, 33
    ops = build_rank_ops(m, 0, Phase("full", 1, t, 0), Shard())
    by = _by(ops)
    _ln(by["attn_norm"], t, h, 0)
    _ln(by["ffn_norm"], t, h, 0)
    _gelu(by["gelu"], t * inter, 0, "bf16")
    assert by["q_scale"].vec == t * h and by["q_scale"].sfu == 0.0 and by["q_scale"].a_fmt == "bf16"
    assert by["rope"].vec == t * heads * (h // heads) * 2 * 3 and by["rope"].sfu == 0.0
    assert by["residual"].vec == t * h * 2 and by["residual"].sfu == 0.0
    assert "qk_norm_rope" not in by and "act" not in by
    assert "quick_gelu" not in by and "gelu_new" not in by
    names = [o.name for o in ops if o.kind == "gemm"]
    assert names == ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.up", "mlp.down"]
    assert by["attn.q"].w_fmt == "fp32" and by["attn.q"].w_bits == 32.0 and by["attn.q"].a_fmt == "bf16"
    assert by["attn.q"].m == t and by["mlp.up"].n == inter
    last = _by(build_rank_ops(m, depth - 1, Phase("full", 1, t, 0), Shard()))
    _ln(last["attn_norm"], t, h, depth - 1)
    _gelu(last["gelu"], t * inter, depth - 1, "bf16")
    post = full_io_ops(m, Phase("full", 1, t, 0), Shard(), "post")
    pb = _by(post)
    _ln(pb["final_norm"], t, h, depth)
    _gelu(pb["lm_gelu"], t * h, depth, "bf16")
    _ln(pb["lm_ln"], t, h, depth)
    assert pb["lm_bias"].vec == t * vocab and pb["lm_bias"].sfu == 0.0 and pb["lm_bias"].a_fmt == "bf16"
    order = [o.name for o in post]
    assert order.index("io.lm_dense") < order.index("lm_gelu") < order.index("lm_ln")
    assert order.index("lm_ln") < order.index("mlm_decoder") < order.index("lm_bias")
    assert pb["mlm_decoder"].k == h and pb["mlm_decoder"].n == vocab and pb["mlm_decoder"].flops == 2 * t * h * vocab
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    short = _by(build_rank_ops(m, 0, Phase("full", 1, 130, 0), Shard()))
    _gelu(short["gelu"], 130 * inter, 0, "bf16")
    _ln(short["ffn_norm"], 130, h, 0)
    two = _by(build_rank_ops(m, 0, Phase("full", 2, t, 0), Shard()))
    _gelu(two["gelu"], 2 * t * inter, 0, "bf16")
    _ln(two["attn_norm"], 2 * t, h, 0)


def test_esm2_tp_sp_and_fp8_keep_released_weight_dtype():
    m = get_model("esm2-650m")
    t, h, inter = 514, 1280, 5120
    sh = Shard(tp=2, sp=2)
    loc = _by(build_rank_ops(m, 0, Phase("full", 1, t, 0), sh))
    rows = -(-t // 2)
    _gelu(loc["gelu"], rows * -(-inter // 2), 0, "bf16")
    _ln(loc["attn_norm"], rows, h, 0)
    assert loc["q_scale"].vec == rows * -(-20 // 2) * (h // 20)
    assert loc["mlp.up"].n == inter // 2 and loc["attn.q"].n == h // 2
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("full", 1, t, 0), Shard()))
    _gelu(op["gelu"], t * inter, 0, "fp8")
    _ln(op["attn_norm"], t, h, 0)
    assert op["q_scale"].a_fmt == "fp8" and op["attn.q"].a_fmt == "fp8"
    assert op["attn.q"].w_fmt == "fp32" and op["attn.q"].w_bits == 32.0
    post = _by(full_io_ops(fp8, Phase("full", 1, t, 0), Shard(), "post"))
    assert post["lm_gelu"].a_fmt == "fp8" and post["lm_ln"].a_fmt == "fp32"
    assert post["final_norm"].a_fmt == "fp32" and post["mlm_decoder"].w_fmt == "fp32"


def test_esm2_3b_same_recipe():
    m = get_model("esm2-3b")
    assert m.vec_kind == "esm2" and m.params() == m.release_params
    t, h, inter, depth = 514, 2560, 10240, 36
    by = _by(build_rank_ops(m, 0, Phase("full", 1, t, 0), Shard()))
    _ln(by["attn_norm"], t, h, 0)
    _gelu(by["gelu"], t * inter, 0, "bf16")
    assert by["rope"].vec == t * h * 6
    post = _by(full_io_ops(m, Phase("full", 1, t, 0), Shard(), "post"))
    _ln(post["final_norm"], t, h, depth)
    assert post["lm_bias"].vec == t * 33


def test_other_full_forwards_keep_the_assumed_tail():
    wan = get_model("wan2.1-1.3b")
    assert wan.vec_kind == "" and wan.layers[0].vec_kind == ""
    w = _by(build_rank_ops(wan, 0, Phase("full", 1, 8, 0), Shard()))
    assert w["act"].vec == 8 * wan.layers[0].ffn.d_ff * 8 and w["act"].sfu == 0.0
    assert w["attn_norm"].sfu == 0.0 and w["attn_norm"].a_fmt == "bf16"
    assert "gelu" not in w and "qk_norm_rope" in w
    fold = get_model("esmfold")
    assert fold.vec_kind == "" and fold.hidden == 1024
    assert fold.layers[0].vec_kind == "esm2" and fold.layers[0].vec_width == 2560
    f = _by(build_rank_ops(fold, 0, Phase("full", 1, 8, 0), Shard()))
    assert "act" not in f and "qk_norm_rope" not in f and f["gelu"].vec == 8 * 10240 * 5


def test_umt5_bills_rmsnorm_and_not_tanh_gelu():
    from accel_dse.core.pipeline import pipeline_for, text_ops
    for mid in ("wan2.1-1.3b", "wan2.1-14b", "wan2.2-a14b"):
        assert pipeline_for(mid).text[0].key == "umt5-xxl"
    te = pipeline_for("wan2.1-1.3b").text[0]
    ops = text_ops(te, 1, "fp8")[0]
    by = _by(ops)
    rows, h, depth = 512, 4096, 24
    _ln(by["te.0.attn_rms"], rows, h, 0)
    _ln(by["te.0.mlp_rms"], rows, h, 0)
    _ln(by["te.23.attn_rms"], rows, h, 23)
    _ln(by["te.final_rms"], rows, h, depth)
    assert sum(1 for n in by if n.endswith(".attn_rms") or n.endswith(".mlp_rms") or n == "te.final_rms") == 49
    assert not any("gelu" in n or n.endswith(".silu") or n.endswith(".mlp_mul") for n in by)
    names = [o.name for o in ops]
    gemms = [o for o in ops if o.kind == "gemm"]
    assert all(names.index(g.name) < names.index("te.0.attn_rms") for g in gemms)
    assert names.index("te.0.attn_rms") < names.index("te.0.qk") < names.index("te.0.mlp_rms")
    assert names[-1] == "te.final_rms"
    g = gemms[0]
    assert g.w_fmt == "bf16" and g.w_bits == 16.0 and g.a_fmt == "fp8" and g.vec == g.m * g.n * 3.0
    assert all(o.causal == 1.0 for o in ops if o.kind == "attn")
    assert all(o.flops == 0 for o in ops if o.a_fmt == "fp32")
    two = _by(text_ops(te, 2, "bf16")[0])
    _ln(two["te.0.attn_rms"], 1024, h, 0)
    _ln(two["te.final_rms"], 1024, h, depth)


def test_esm_config_mismatch_is_refused():
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "silu")), "hidden_act is not gelu")
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "gelu_new")), "hidden_act is not gelu")
    _raises(_bad(lambda c: c.__setitem__("hidden_act", "quick_gelu")), "hidden_act is not gelu")
    _raises(_bad(lambda c: c.__setitem__("position_embedding_type", "absolute")),
            "position_embedding_type is not rotary")
    _raises(_bad(lambda c: c.__setitem__("emb_layer_norm_before", True)),
            "emb_layer_norm_before is not false")
    _raises(_bad(lambda c: c.__setitem__("hidden_dropout_prob", 0.1)), "dropout is not 0")
    _raises(_bad(lambda c: c.__setitem__("is_folding_model", True)), "folding model is not the encoder")
