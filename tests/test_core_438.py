"""0.438: Carbon-A GenerannoRMSNorm and SiLU SwiGLU.

modeling_generanno.py: RMSNorm is T5LayerNorm (fp32 pow(2).mean, one rsqrt, times weight).
The MLP is down(silu(gate) * up) when pretraining_tp is 1. RoPE is applied before
repeat_kv. There is no QK-norm. The two score heads are bias-free linears. Vector
flops stay 0. GEMM shapes stay put.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import build_generanno
from accel_dse.core.ir import Phase, Shard, build_rank_ops, full_io_ops
from accel_dse.core.model import NativeWorkload, load_release


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _rms(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _silu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _mul(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release("HuggingFaceBio/Carbon-A-1.2B")))
    edit(rel)
    wl = NativeWorkload("genomics", seq_len=8)
    return lambda: build_generanno("x", "HuggingFaceBio/Carbon-A-1.2B", rel, wl)


def test_version_is_438():
    assert __version__ == "0.451.0"


def test_carbon_a_bills_rmsnorm_and_silu():
    m = get_model("carbon-a-1.2b")
    assert m.vec_kind == "generanno" and m.params() == m.release_params == 1153656832
    assert m.param_check()["rel_err"] == 0.0
    assert m.layers[0].vec_kind == "generanno" and m.layers[0].core.qk_norm is False
    assert m.layers[0].core.causal is False and m.layers[0].core.q_scale is False
    assert m.layers[0].ffn.gated is True and m.act_fmt == "bf16" and m.vocab_head is False
    t, h, inter, nq, nkv, hd, depth = 16384, 2048, 5632, 32, 4, 64, 26
    ops = build_rank_ops(m, 0, Phase("full", 1, t, 0), Shard())
    by = _by(ops)
    _rms(by["attn_rms"], t, h, 0)
    _rms(by["mlp_rms"], t, h, 0)
    _silu(by["silu"], t * inter, 0, "bf16")
    _mul(by["mlp_mul"], t * inter, 0, "bf16")
    assert by["rope"].vec == t * (nq + nkv) * hd * 3 and by["rope"].sfu == 0.0
    assert by["rope"].a_fmt == "bf16"
    assert by["residual"].vec == t * h * 2 and by["residual"].sfu == 0.0
    for name in ("qk_norm_rope", "q_scale", "gelu", "act", "attn_norm", "ffn_norm", "final_norm"):
        assert name not in by
    names = [o.name for o in ops if o.kind == "gemm"]
    assert names == ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate_up", "mlp.down"]
    order = [o.name for o in ops]
    assert order.index("mlp.gate_up") < order.index("mlp.down") < order.index("silu")
    assert order.index("silu") < order.index("mlp_mul") < order.index("residual")
    q, k = by["attn.q"], by["attn.k"]
    assert (q.m, q.k, q.n) == (t, h, h) and q.w_fmt == "bf16" and q.w_bits == 16.0
    assert q.a_fmt == "bf16" and q.vec == 0.0 and q.flops == 2 * t * h * h
    assert k.n == nkv * hd and by["mlp.gate_up"].n == 2 * inter and by["mlp.down"].k == inter
    assert by["mlp.down"].n == h and by["mlp.gate_up"].vec == 0.0
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    last = _by(build_rank_ops(m, depth - 1, Phase("full", 1, t, 0), Shard()))
    _rms(last["attn_rms"], t, h, depth - 1)
    _silu(last["silu"], t * inter, depth - 1, "bf16")
    post = full_io_ops(m, Phase("full", 1, t, 0), Shard(), "post")
    pb = _by(post)
    _rms(pb["final_rms"], t, h, depth)
    orderp = [o.name for o in post]
    assert orderp.index("final_rms") < orderp.index("io.score_0") < orderp.index("io.score_1")
    assert "final_norm" not in pb and "mlm_decoder" not in pb and "act" not in pb and "gelu" not in pb
    assert pb["io.score_0"].m == t and pb["io.score_0"].k == h and pb["io.score_0"].n == 12
    assert pb["io.score_0"].w_fmt == "bf16" and pb["io.score_1"].n == 12
    nrms = 1
    for li in range(depth):
        layer_ops = build_rank_ops(m, li, Phase("full", 1, t, 0), Shard())
        nrms += sum(1 for o in layer_ops if o.name in ("attn_rms", "mlp_rms"))
    assert nrms == depth * 2 + 1
    short = _by(build_rank_ops(m, 0, Phase("full", 1, 130, 0), Shard()))
    _silu(short["silu"], 130 * inter, 0, "bf16")
    assert short["attn_rms"].sfu == 130.0
    two = _by(build_rank_ops(m, 0, Phase("full", 2, t, 0), Shard()))
    _silu(two["silu"], 2 * t * inter, 0, "bf16")
    _rms(two["attn_rms"], 2 * t, h, 0)


def test_carbon_a_tp_sp_and_fp8_follow_billed_activation():
    m = get_model("carbon-a-1.2b")
    t, h, inter = 16384, 2048, 5632
    sh = Shard(tp=2, sp=2)
    loc = _by(build_rank_ops(m, 0, Phase("full", 1, t, 0), sh))
    rows, inter_loc = t // 2, inter // 2
    _silu(loc["silu"], rows * inter_loc, 0, "bf16")
    _mul(loc["mlp_mul"], rows * inter_loc, 0, "bf16")
    _rms(loc["attn_rms"], rows, h, 0)
    _rms(loc["mlp_rms"], rows, h, 0)
    assert loc["rope"].vec == rows * (16 + 2) * 64 * 3 and loc["rope"].a_fmt == "bf16"
    assert loc["attn.q"].n == 1024 and loc["attn.k"].n == 128
    assert loc["mlp.gate_up"].n == inter and loc["mlp.down"].k == inter_loc
    assert loc["mlp.down"].n == h and loc["attn.q"].m == rows
    post = _by([o for o in full_io_ops(m, Phase("full", 1, t, 0), sh, "post") if o.name != "io_allgather"])
    _rms(post["final_rms"], rows, h, 26)
    assert post["io.score_0"].n == 12 and post["io.score_0"].m == rows
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("full", 1, t, 0), Shard()))
    _silu(op["silu"], t * inter, 0, "fp8")
    _mul(op["mlp_mul"], t * inter, 0, "fp8")
    _rms(op["attn_rms"], t, h, 0)
    _rms(op["mlp_rms"], t, h, 0)
    assert op["rope"].a_fmt == "bf16"
    assert op["attn.q"].a_fmt == "fp8" and op["attn.q"].w_fmt == "bf16" and op["attn.q"].w_bits == 16.0
    assert op["mlp.down"].a_fmt == "fp8" and op["mlp.down"].w_fmt == "bf16"
    head = _by(full_io_ops(fp8, Phase("full", 1, t, 0), Shard(), "post"))
    assert head["final_rms"].a_fmt == "fp32" and head["io.score_0"].a_fmt == "fp8"
    assert head["io.score_0"].w_fmt == "bf16"


def test_esm2_rope_stays_the_mha_count():
    m = get_model("esm2-650m")
    by = _by(build_rank_ops(m, 0, Phase("full", 1, 514, 0), Shard()))
    assert by["rope"].a_fmt == "bf16" and by["rope"].vec == 514 * 1280 * 6
    assert "qk_norm_rope" not in by


def test_generanno_ir_refuses_other_blocks():
    m = get_model("carbon-a-1.2b")
    ph = Phase("full", 1, 8, 0)
    adaln = replace(m, adaln=True)
    _raises(lambda: build_rank_ops(adaln, 0, ph, Shard()), "generanno vector billing is RMSNorm and SwiGLU")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    dense = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(dense, 0, ph, Shard()), "generanno vector billing is RMSNorm and SwiGLU")
    moe = replace(m.layers[0], ffn=replace(m.layers[0].ffn, kind="moe"))
    routed = replace(m, layers=(moe,) + m.layers[1:])
    _raises(lambda: build_rank_ops(routed, 0, ph, Shard()), "generanno vector billing is RMSNorm and SwiGLU")


def test_generanno_config_mismatch_is_refused():
    _raises(_bad(lambda r: r["config"].__setitem__("model_type", "llama")), "model_type is not generanno")
    _raises(_bad(lambda r: r["config"].__setitem__("hidden_act", "gelu_new")), "hidden_act is not silu")
    _raises(_bad(lambda r: r["config"].__setitem__("hidden_act", "quick_gelu")), "hidden_act is not silu")
    _raises(_bad(lambda r: r["config"].__setitem__("attention_bias", True)), "attention_bias is not false")
    _raises(_bad(lambda r: r["config"].__setitem__("mlp_bias", True)), "mlp_bias is not false")
    _raises(_bad(lambda r: r["config"].__setitem__("attention_dropout", 0.1)), "attention_dropout is not 0")
    _raises(_bad(lambda r: r["config"].__setitem__("rope_scaling", {"rope_type": "linear"})),
            "rope_scaling is not null")
    _raises(_bad(lambda r: r["config"].__setitem__("pretraining_tp", 2)), "pretraining_tp is not 1")
    _raises(_bad(lambda r: r["config"].__setitem__("use_mlp_classifier", True)),
            "use_mlp_classifier is not false")
    _raises(_bad(lambda r: r["config"].__setitem__("num_prediction_heads", 1)),
            "num_prediction_heads is not 2")
    _raises(_bad(lambda r: r["config"].__setitem__("k", 5)), "score width is not k*2")
    _raises(_bad(lambda r: r["config"].__setitem__("head_dim", 32)), "head_dim is not hidden/heads")
    _raises(_bad(lambda r: r["config"].__setitem__("num_attention_heads", 30)),
            "head counts do not divide")
    _raises(_bad(lambda r: r["shapes"].__setitem__("model.layers.0.input_layernorm.weight", [1024])),
            "model.layers.0.input_layernorm.weight shape is not [2048]")
    _raises(_bad(lambda r: r["shapes"].__setitem__("model.layers.0.self_attn.q_norm.weight", [64])),
            "generanno has q_norm")
    _raises(_bad(lambda r: r["shapes"].__setitem__("score.0.bias", [12])), "score bias is present")
