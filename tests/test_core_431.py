"""0.431: Ling-3.0-flash-VL LayerNorm and projector erf-GELU.

Block norm1/norm2 and the norm-only merger are nn.LayerNorm on patch rows.
nn.GELU() sits between linear_proj.0 and linear_proj.2, on merged rows at the
text hidden size. The block MLP stays gelu_pytorch_tanh. Vector flops stay 0.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

MID = "ling-3.0-flash-vl"
HF = "inclusionAI/Ling-3.0-flash-VL"
H, DEPTH, HEADS, INTER, TEXT = 1152, 27, 16, 4304, 2560
ROWS = 64 * 64
MERGED = 1024
PARAMS = 434_248_944
FLOP_1024 = 5_506_919_825_408
FLOP_768 = 3_738_810_974_208
FLOP_480 = 1_180_984_934_400
FLOP_1080 = 15_096_730_091_520
FLOP_1 = 13_389_529_088


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _norm(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _gelu(op, rows: int, width: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _lns(ops) -> list:
    found = []
    for op in ops:
        if op.name == "vis.pre_ln" or op.name.endswith(".attn_ln") or op.name.endswith(".mlp_ln"):
            found.append(op)
    return found


def test_ling_bills_layernorm_and_projector_gelu():
    m = get_model(MID)
    vs = m.vision
    assert vs.family == "qwen" and vs.ln_mode == "ling" and vs.vec_fmt == "bf16"
    assert vs.gated is False and vs.deepstack == 0 and vs.kv_heads == 0
    assert vs.hidden == H and vs.depth == DEPTH and vs.heads == HEADS and vs.inter == INTER
    assert vs.qkv == H and vs.params == PARAMS and vs.post == ((4 * H, TEXT), (TEXT, TEXT))
    assert m.params() == m.release_params == 124_414_211_552
    assert m.param_check()["rel_err"] == 0.0 and m.coverage == "full"
    rel = load_release(HF)
    vc = rel["config"]["vision_config"]
    assert vc["hidden_act"] == "gelu_pytorch_tanh" and vc["disable_merger_proj"] is True
    assert vc["model_type"] == "qwen3_moe_vit" and vc["out_hidden_size"] == 4096
    assert "mlp_depth" not in rel["config"] and rel["config"]["norm_query_embeds"] is False
    role = rel["roles"]["vision"]
    assert list(role) == ["bf16"] and role["bf16"]["params"] == PARAMS
    note = " ".join(m.notes)
    assert "nn.LayerNorm" in note and "nn.GELU()" in note and "gelu_pytorch_tanh" in note
    assert "mlp_depth" in note and "norm_query_embeds" in note and "FFN_GELU" in note
    assert "fp32" in note and vs.vec_fmt in note
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens) == (64, 64, MERGED, MERGED)
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _norm(by["vis.0.attn_ln"], ROWS, H, 0)
    _norm(by["vis.0.mlp_ln"], ROWS, H, 0)
    _norm(by[f"vis.{DEPTH - 1}.attn_ln"], ROWS, H, DEPTH - 1)
    _norm(by[f"vis.{DEPTH - 1}.mlp_ln"], ROWS, H, DEPTH - 1)
    _norm(by["vis.pre_ln"], ROWS, H, DEPTH)
    _gelu(by["vis.proj_gelu"], MERGED, TEXT, DEPTH, "bf16")
    assert by["vis.0.qkv"].m == ROWS and by["vis.0.qkv"].n == 3 * H
    assert by["vis.0.qkv"].vec == ROWS * 3 * H * 3
    assert by["vis.0.qkv"].a_fmt == "bf16" and by["vis.0.qkv"].w_fmt == "bf16"
    assert by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.mlp_in"].n == INTER and by["vis.0.mlp_in"].vec == ROWS * INTER * 3
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * ROWS * ROWS * 5
    assert (by["vis.merger.0"].m, by["vis.merger.0"].k, by["vis.merger.0"].n) == (MERGED, 4 * H, TEXT)
    assert by["vis.merger.0"].vec == MERGED * TEXT * 3
    assert (by["vis.merger.1"].m, by["vis.merger.1"].k, by["vis.merger.1"].n) == (MERGED, TEXT, TEXT)
    assert by["vis.merger.1"].vec == MERGED * TEXT * 3
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") < names.index("vis.0.attn_ln") < names.index("vis.0.qkv")
    assert names.index("vis.0.qkv") < names.index("vis.0.o") < names.index("vis.0.mlp_ln")
    assert names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out")
    assert names.index("vis.0.mlp_out") < names.index("vis.0.qk") < names.index("vis.0.softmax")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.pre_ln")
    assert names.index("vis.pre_ln") < names.index("vis.merger.0") < names.index("vis.proj_gelu")
    assert names.index("vis.proj_gelu") < names.index("vis.merger.1")
    assert len(_lns(ops)) == 1 + 2 * DEPTH
    assert sum(op.name == "vis.proj_gelu" for op in ops) == 1
    for absent in ("vis.0.gelu", "vis.0.silu", "vis.0.relu", "vis.mgelu", "vis.mgelu2",
                   "vis.0.mlp_mul", "vis.ln_post", "vis.0.quick_gelu"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) - FLOP_1024) < 1.0
    wide = image_grid(vs, 1024, 768)
    assert (wide.ph, wide.pw, wide.merged, wide.llm_tokens) == (48, 64, 768, 768)
    wby = _by(vision_ops(vs, wide, 1))
    _norm(wby["vis.pre_ln"], 48 * 64, H, DEPTH)
    _gelu(wby["vis.proj_gelu"], 768, TEXT, DEPTH, "bf16")
    assert wby["vis.merger.0"].m == 768 and abs(vision_flops(vs, wide, 1) - FLOP_768) < 1.0
    small = image_grid(vs, 640, 480)
    assert (small.ph, small.pw, small.merged, small.llm_tokens) == (30, 40, 300, 300)
    sby = _by(vision_ops(vs, small, 1))
    _norm(sby["vis.0.attn_ln"], 30 * 40, H, 0)
    _gelu(sby["vis.proj_gelu"], 300, TEXT, DEPTH, "bf16")
    assert abs(vision_flops(vs, small, 1) - FLOP_480) < 1.0
    tall = image_grid(vs, 1920, 1080)
    assert (tall.ph, tall.pw, tall.merged, tall.llm_tokens) == (68, 120, 2040, 2040)
    tby = _by(vision_ops(vs, tall, 1))
    _norm(tby["vis.pre_ln"], 68 * 120, H, DEPTH)
    _gelu(tby["vis.proj_gelu"], 2040, TEXT, DEPTH, "bf16")
    assert tby["vis.0.qkv"].m == 68 * 120 and tby["vis.merger.0"].m == 2040
    assert abs(vision_flops(vs, tall, 1) - FLOP_1080) < 1.0
    tiny = image_grid(vs, 1, 1)
    assert (tiny.ph, tiny.pw, tiny.merged, tiny.llm_tokens) == (4, 4, 4, 4)
    yby = _by(vision_ops(vs, tiny, 1))
    _norm(yby["vis.pre_ln"], 16, H, DEPTH)
    _gelu(yby["vis.proj_gelu"], 4, TEXT, DEPTH, "bf16")
    assert abs(vision_flops(vs, tiny, 1) - FLOP_1) < 1.0


def test_ling_two_images_dtype_and_other_towers():
    vs = get_model(MID).vision
    g = image_grid(vs, 1024, 1024)
    two = _by(vision_ops(vs, g, 2, "bf16"))
    _norm(two["vis.0.attn_ln"], 2 * ROWS, H, 0)
    _norm(two["vis.pre_ln"], 2 * ROWS, H, DEPTH)
    _gelu(two["vis.proj_gelu"], 2 * MERGED, TEXT, DEPTH, "bf16")
    assert two["vis.0.qkv"].m == 2 * ROWS and two["vis.0.qkv"].vec == 2 * ROWS * 3 * H * 3
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * MERGED and two["vis.merger.1"].m == 2 * MERGED
    pair = _by(vision_ops(vs, g, 2, "fp8"))
    _norm(pair["vis.pre_ln"], 2 * ROWS, H, DEPTH)
    _gelu(pair["vis.proj_gelu"], 2 * MERGED, TEXT, DEPTH, "bf16")
    assert pair["vis.0.attn_ln"].a_fmt == "fp32" and pair["vis.proj_gelu"].a_fmt == "bf16"
    assert pair["vis.0.qkv"].a_fmt == "fp8" and pair["vis.0.qkv"].vec == 2 * ROWS * 3 * H * 3
    assert pair["vis.0.qkv"].w_fmt == "bf16" and pair["vis.0.qkv"].w_bits == 16.0
    assert pair["vis.0.mlp_in"].a_fmt == "fp8" and pair["vis.merger.0"].a_fmt == "fp8"
    assert pair["vis.merger.1"].a_fmt == "fp8" and pair["vis.merger.1"].w_fmt == "bf16"
    assert pair["vis.0.softmax"].sfu == 0.0
    ev = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", prompt=32, images=1)))
    assert ev.vision["image_tokens"] == MERGED
    for mid, mode in (("qwen3.5-9b", "qwen35"), ("qwen3-vl-2b", "qwen3vl"),
                      ("qwen3.8-flash-next-fp4-local", "")):
        other = get_model(mid)
        assert other.vision is not None and other.vision.ln_mode == mode
        names = [op.name for op in vision_ops(other.vision, image_grid(other.vision, 1024, 1024), 1)]
        assert "vis.proj_gelu" not in names, mid
        assert "Ling-3.0-flash-VL" not in " ".join(other.notes)


def test_ling_refuses_other_shapes():
    rel = load_release(HF)
    base = rel["config"]
    bare = vision_spec(HF, base)
    assert bare is not None and bare.ln_mode == "ling" and bare.vec_fmt == ""
    assert bare.params == PARAMS
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "ling vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "ling vision activation dtype is not a vision float")
    billed = replace(bare, vec_fmt="bf16")
    _raises(lambda: vision_ops(replace(billed, gated=True), g, 1),
            "ling vision block is not fused qkv plus a two-linear projector")
    _raises(lambda: vision_ops(replace(billed, post=((4 * H, TEXT), (TEXT, 4096))), g, 1),
            "ling vision block is not fused qkv plus a two-linear projector")
    _raises(lambda: vision_ops(replace(billed, post=((4 * H, 4 * H), (4 * H, TEXT))), g, 1),
            "ling vision block is not fused qkv plus a two-linear projector")
    _raises(lambda: vision_ops(replace(billed, qkv=H * 2), g, 1),
            "ling vision block is not fused qkv plus a two-linear projector")
    silu = copy.deepcopy(base)
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF, silu), "ling vision hidden_act is not gelu_pytorch_tanh")
    one = copy.deepcopy(base)
    one["mlp_depth"] = 1
    _raises(lambda: vision_spec(HF, one), "ling mlp_depth is not 2")
    off = copy.deepcopy(base)
    off["vision_config"]["disable_merger_proj"] = False
    assert vision_spec(HF, off) is None
