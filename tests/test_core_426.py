"""0.426: DeepSeek-V4.1 vision RMSNorm, gate SiLU, product, and aligner erf-GELU.

RMSNorm reduces in fp32, one rsqrt per patch token. F.silu is x * sigmoid(x)
on the gate only. The product with up is one multiply. F.gelu after aligner w1
is the erf form, on merged rows at the text hidden size. RoPE, pad, unfold, and
the full residual adds are not ops. The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.model import from_release, load_release
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

MID = "deepseek-v4.1-flash"
HF = "deepseek-ai/DeepSeek-V4.1-Flash"
H = 1024
INTER = 2816
DEPTH = 32
HEADS = 16
TEXT = 5120
ROWS = 74 * 74
MERGED = 625
GFLOP = 8530.47492608


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


def _silu(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows * width * 2)
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0


def _mul(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width and op.sfu == 0.0
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0


def _gelu(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_deepseek_v41_bills_rmsnorm_silu_and_aligner_gelu():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "ds" and vs.vec_fmt == "bf16"
    assert vs.gated is True and vs.depth == DEPTH and vs.heads == HEADS
    assert vs.hidden == H and vs.inter == INTER and vs.qkv == H and vs.merge == 3
    assert vs.post == ((9 * H, TEXT), (TEXT, TEXT)) and vs.params == 485268480
    rel = load_release(m.hf_id)
    vc = rel["config"]["vision_config"]
    assert vc["model_type"] == "deepseek_v41_vision" and "hidden_act" not in vc
    assert rel["config"]["text_config"]["hidden_act"] == "silu"
    assert rel["config"]["text_config"]["swiglu_limit"] == 10.0
    assert set(rel["roles"]["vision"]) == {"bf16"}
    assert rel["roles"]["vision"]["bf16"]["params"] == 485268480
    assert "DeepSeek-V4.1" in " ".join(m.notes)
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.ph * g.pw, g.merged, g.llm_tokens) == (74, 74, ROWS, MERGED, 652)
    assert g.ph % 3 == 2 and ROWS // 9 != MERGED
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _rms(by["vis.0.norm1"], ROWS, H, 0)
    _rms(by["vis.0.norm2"], ROWS, H, 0)
    _rms(by[f"vis.{DEPTH - 1}.norm1"], ROWS, H, DEPTH - 1)
    _rms(by[f"vis.{DEPTH - 1}.norm2"], ROWS, H, DEPTH - 1)
    _silu(by["vis.0.silu"], ROWS, INTER, 0)
    _mul(by["vis.0.mlp_mul"], ROWS, INTER, 0)
    _silu(by[f"vis.{DEPTH - 1}.silu"], ROWS, INTER, DEPTH - 1)
    _rms(by["vis.out_rms"], ROWS, H, DEPTH)
    _gelu(by["vis.align_gelu"], MERGED, TEXT, DEPTH)
    assert by["vis.0.qkv"].n == 3 * H and by["vis.0.qkv"].vec == ROWS * (3 * H) * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.qkv"].a_fmt == "bf16"
    assert by["vis.0.mlp_in"].n == 2 * INTER and by["vis.0.mlp_in"].vec == ROWS * (2 * INTER) * 3
    assert by["vis.0.mlp_out"].k == INTER and by["vis.0.mlp_out"].n == H
    assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].k == 9 * H
    assert by["vis.merger.0"].n == TEXT and by["vis.merger.0"].vec == MERGED * TEXT * 3
    assert by["vis.merger.1"].k == TEXT and by["vis.merger.1"].n == TEXT
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * ROWS * ROWS * 5
    names = [op.name for op in ops]
    assert names.index("vis.0.norm1") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.norm2") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.silu") < names.index("vis.0.mlp_mul")
    assert names.index("vis.0.mlp_mul") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.out_rms")
    assert names.index("vis.out_rms") < names.index("vis.merger.0") < names.index("vis.align_gelu")
    assert names.index("vis.align_gelu") < names.index("vis.merger.1")
    norms = [op for op in ops if op.name.endswith(".norm1") or op.name.endswith(".norm2")
             or op.name == "vis.out_rms"]
    silus = [op for op in ops if op.name.endswith(".silu")]
    muls = [op for op in ops if op.name.endswith(".mlp_mul")]
    gels = [op for op in ops if op.name == "vis.align_gelu"]
    assert len(norms) == 2 * DEPTH + 1 and len(silus) == DEPTH and len(muls) == DEPTH
    assert len(gels) == 1
    for absent in ("vis.0.attn_ln", "vis.0.gelu", "vis.0.q_rms", "vis.0.k_rms", "vis.0.relu",
                   "vis.post_rms", "vis.proj_gelu", "vis.proj_silu", "vis.proj_mul", "vis.ln_q",
                   "vis.mgelu", "vis.0.relu2"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - GFLOP) < 1e-6

    square = image_grid(vs, 448, 448)
    assert (square.ph, square.pw, square.merged, square.llm_tokens) == (39, 39, 169, 184)
    assert square.ph % 3 == 0 and square.merged == square.ph * square.pw // 9
    sb = _by(vision_ops(vs, square, 1, "bf16"))
    _rms(sb["vis.0.norm1"], 1521, H, 0)
    _silu(sb["vis.0.silu"], 1521, INTER, 0)
    _mul(sb["vis.0.mlp_mul"], 1521, INTER, 0)
    _rms(sb["vis.out_rms"], 1521, H, DEPTH)
    _gelu(sb["vis.align_gelu"], 169, TEXT, DEPTH)
    assert abs(vision_flops(vs, square, 1) / 1e9 - 1580.25740288) < 1e-6
    small = image_grid(vs, 256, 256)
    assert (small.ph, small.pw, small.merged, small.llm_tokens) == (39, 39, 169, 184)

    wide = image_grid(vs, 640, 480)
    assert (wide.ph, wide.pw, wide.merged, wide.llm_tokens) == (35, 46, 192, 206)
    assert wide.ph % 3 == 2 and wide.pw % 3 == 1 and wide.ph * wide.pw // 9 != wide.merged
    wb = _by(vision_ops(vs, wide, 1, "bf16"))
    _rms(wb["vis.out_rms"], 1610, H, DEPTH)
    _gelu(wb["vis.align_gelu"], 192, TEXT, DEPTH)
    _silu(wb["vis.0.silu"], 1610, INTER, 0)
    assert abs(vision_flops(vs, wide, 1) / 1e9 - 1693.43082496) < 1e-6

    tall = image_grid(vs, 1024, 768)
    assert (tall.ph, tall.pw, tall.merged, tall.llm_tokens) == (55, 74, 475, 496)
    assert tall.ph * tall.pw // 9 == 452 and tall.merged == 475
    tb = _by(vision_ops(vs, tall, 1, "bf16"))
    _rms(tb["vis.0.norm1"], 4070, H, 0)
    _gelu(tb["vis.align_gelu"], 475, TEXT, DEPTH)
    assert tb["vis.merger.0"].m == 475 and tb["vis.merger.1"].n == TEXT
    assert abs(vision_flops(vs, tall, 1) / 1e9 - 5591.70625536) < 1e-6

    two = _by(vision_ops(vs, g, 2, "bf16"))
    _rms(two["vis.0.norm1"], 2 * ROWS, H, 0)
    _silu(two["vis.0.silu"], 2 * ROWS, INTER, 0)
    _mul(two["vis.0.mlp_mul"], 2 * ROWS, INTER, 0)
    _rms(two["vis.out_rms"], 2 * ROWS, H, DEPTH)
    _gelu(two["vis.align_gelu"], 2 * MERGED, TEXT, DEPTH)
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * MERGED

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.mlp_in"].a_fmt == "fp8"
    assert forced["vis.merger.0"].a_fmt == "fp8" and forced["vis.merger.0"].w_fmt == "bf16"
    assert forced["vis.0.qkv"].w_bits == 16.0 and forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
    assert forced["vis.0.silu"].a_fmt == "bf16" and forced["vis.0.mlp_mul"].a_fmt == "bf16"
    assert forced["vis.align_gelu"].a_fmt == "bf16"
    assert forced["vis.0.norm1"].a_fmt == "fp32" and forced["vis.out_rms"].a_fmt == "fp32"


def test_deepseek_v41_refuses_a_missing_or_mixed_dtype():
    rel = load_release(HF)
    bare = vision_spec(HF, rel["config"])
    assert bare is not None and bare.vec_fmt == "" and bare.family == "ds"
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "deepseek vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "deepseek vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", post=()), g, 1),
            "deepseek vision block is not fused qkv plus a two-linear aligner")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", gated=False), g, 1),
            "deepseek vision block is not fused qkv plus a two-linear aligner")
    spec = from_release("synth", HF, rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16"
    assert spec.param_check()["rel_err"] == 0.0
    assert "F.silu" in " ".join(spec.notes)
    mixed = copy.deepcopy(rel)
    mixed["roles"]["vision"]["fp32"] = {"bytes": 4, "params": 1}
    _raises(lambda: from_release("synth", HF, rel=mixed, cfg=mixed["config"]),
            "deepseek tower vision role is not one dtype")
