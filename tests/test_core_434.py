"""0.434: MiniMax-M3 / M3.1-Flash vision LayerNorm and erf-GELU.

pre_layrnorm and each block's layer_norm1 / layer_norm2 are nn.LayerNorm on
patch rows. ACT2FN["gelu"] (the erf form) sits on the MLP intermediate and
after the first projector linear and the first merge linear. The conv weight
is fp32. patch_merge_mlp parameters stay in the language role. Vector flops
stay 0. Text swigluoai is not billed.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

MIDS = ("minimax-m3", "minimax-m3.1-flash")
HF = {
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "minimax-m3.1-flash": "MiniMaxAI/MiniMax-M3.1-Flash",
}
H, DEPTH, HEADS, INTER, TEXT = 1280, 32, 16, 5120, 6144
ROWS, MERGED = 48 * 48, 576
PARAMS = 676_811_264
VISION_BYTES = 1_356_633_088
LANG = {
    "minimax-m3": (426_174_558_336, 426_363_328_896),
    "minimax-m3.1-flash": (426_186_354_816, 426_375_126_144),
}
FLOP_1024 = 4_203_387_813_888
FLOP_768 = 3_889_715_281_920
FLOP_480 = 2_663_719_518_208
FLOP_TINY = 23_192_338_432


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


def test_minimax_m3_bills_layernorm_and_erf_gelu():
    specs = []
    for mid in MIDS:
        m = get_model(mid)
        vs = m.vision
        assert vs.family == "minimax" and vs.ln_mode == "minimax" and vs.vec_fmt == "bf16"
        assert vs.gated is False and vs.kv_heads == 0 and vs.deepstack == 0
        assert vs.hidden == H and vs.depth == DEPTH and vs.heads == HEADS and vs.inter == INTER
        assert vs.qkv == H and vs.patch == 14 and vs.tpatch == 2 and vs.merge == 2
        assert vs.params == PARAMS and vs.extra == 0
        assert abs(vs.params * vs.w_Bpp - VISION_BYTES) < 1
        assert vs.post == ((H, TEXT), (TEXT, TEXT), (4 * TEXT, TEXT), (TEXT, TEXT))
        ours, release = LANG[mid]
        assert m.params() == ours and m.release_params == release
        assert m.param_check()["ours"] == ours and m.param_check()["release"] == release
        assert m.coverage == "full"
        rel = load_release(HF[mid])
        assert rel["config"]["vision_config"]["hidden_act"] == "gelu"
        assert rel["config"]["projector_hidden_act"] == "gelu"
        assert rel["config"]["text_config"]["hidden_act"] == "swigluoai"
        role = rel["roles"]["vision"]
        assert role["fp32"]["params"] == 1_505_280 and role["fp32"]["bytes"] == 6_021_120
        assert role["bf16"]["params"] == 675_305_984
        assert role["bf16"]["params"] + role["fp32"]["params"] == PARAMS
        assert rel["roles"]["other"]["bf16"]["params"] == 188_755_968
        note = " ".join(m.notes)
        assert "pre_layrnorm" in note and "ACT2FN['gelu']" in note and "swigluoai" in note
        assert "fp32" in note and "patch_merge_mlp" in note
        specs.append(vs)
    assert specs[0] == specs[1]
    vs = specs[0]
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.px) == (48, 48, MERGED, MERGED + 2, (672, 672))
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _norm(by["vis.pre_ln"], ROWS, H, 0)
    _norm(by["vis.0.attn_ln"], ROWS, H, 0)
    _norm(by["vis.0.mlp_ln"], ROWS, H, 0)
    _norm(by[f"vis.{DEPTH - 1}.attn_ln"], ROWS, H, DEPTH - 1)
    _norm(by[f"vis.{DEPTH - 1}.mlp_ln"], ROWS, H, DEPTH - 1)
    _gelu(by["vis.0.gelu"], ROWS, INTER, 0, "bf16")
    _gelu(by[f"vis.{DEPTH - 1}.gelu"], ROWS, INTER, DEPTH - 1, "bf16")
    _gelu(by["vis.proj_gelu"], ROWS, TEXT, DEPTH, "bf16")
    _gelu(by["vis.mgelu"], MERGED, TEXT, DEPTH, "bf16")
    patch = by["vis.patch_embed"]
    assert (patch.m, patch.k, patch.n) == (ROWS, 3 * 2 * 14 * 14, H)
    assert patch.w_fmt == "fp32" and patch.w_bits == 32.0 and patch.a_fmt == "bf16"
    assert patch.vec == ROWS * H * 3 and patch.w_params == patch.k * patch.n
    assert by["vis.0.qkv"].m == ROWS and by["vis.0.qkv"].n == 3 * H
    assert by["vis.0.qkv"].vec == ROWS * 3 * H * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.mlp_in"].n == INTER and by["vis.0.mlp_in"].k == H
    assert by["vis.0.mlp_out"].k == INTER and by["vis.0.mlp_out"].n == H
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * ROWS * ROWS * 5
    assert (by["vis.merger.0"].m, by["vis.merger.0"].k, by["vis.merger.0"].n) == (ROWS, H, TEXT)
    assert (by["vis.merger.1"].m, by["vis.merger.1"].k, by["vis.merger.1"].n) == (ROWS, TEXT, TEXT)
    assert (by["vis.merger.2"].m, by["vis.merger.2"].k, by["vis.merger.2"].n) == (MERGED, 4 * TEXT, TEXT)
    assert (by["vis.merger.3"].m, by["vis.merger.3"].k, by["vis.merger.3"].n) == (MERGED, TEXT, TEXT)
    for name in ("vis.merger.0", "vis.merger.1", "vis.merger.2", "vis.merger.3"):
        assert by[name].w_fmt == "bf16" and by[name].w_bits == 16.0
        assert by[name].vec == by[name].m * by[name].n * 3
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") < names.index("vis.pre_ln") < names.index("vis.0.attn_ln")
    assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.gelu") < names.index("vis.0.mlp_out")
    assert names.index("vis.0.mlp_out") < names.index("vis.0.qk") < names.index("vis.0.softmax")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.merger.0")
    assert names.index("vis.merger.0") < names.index("vis.proj_gelu") < names.index("vis.merger.1")
    assert names.index("vis.merger.1") < names.index("vis.merger.2") < names.index("vis.mgelu")
    assert names.index("vis.mgelu") < names.index("vis.merger.3")
    assert len(_lns(ops)) == 1 + 2 * DEPTH
    assert sum(op.name.endswith(".gelu") for op in ops) == DEPTH
    assert sum(op.name in ("vis.proj_gelu", "vis.mgelu") for op in ops) == 2
    for absent in ("vis.0.silu", "vis.0.quick_gelu", "vis.0.relu", "vis.out_ln", "vis.post_ln",
                   "vis.0.mlp_mul", "vis.ln_pre", "vis.after_rms"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) - FLOP_1024) < 1.0


def test_minimax_lengths_two_images_and_dtype():
    vs = get_model("minimax-m3").vision
    wide = image_grid(vs, 1024, 768)
    assert (wide.ph, wide.pw, wide.merged, wide.llm_tokens, wide.px) == (40, 54, 540, 542, (756, 560))
    wby = _by(vision_ops(vs, wide, 1))
    _norm(wby["vis.pre_ln"], 40 * 54, H, 0)
    _gelu(wby["vis.0.gelu"], 40 * 54, INTER, 0, "bf16")
    _gelu(wby["vis.proj_gelu"], 40 * 54, TEXT, DEPTH, "bf16")
    _gelu(wby["vis.mgelu"], 540, TEXT, DEPTH, "bf16")
    assert wby["vis.merger.0"].m == 40 * 54 and wby["vis.merger.2"].m == 540
    assert abs(vision_flops(vs, wide, 1) - FLOP_768) < 1.0
    small = image_grid(vs, 640, 480)
    assert (small.ph, small.pw, small.merged, small.llm_tokens, small.px) == (34, 46, 391, 393, (644, 476))
    sby = _by(vision_ops(vs, small, 1))
    _norm(sby["vis.0.attn_ln"], 34 * 46, H, 0)
    _gelu(sby["vis.mgelu"], 391, TEXT, DEPTH, "bf16")
    assert abs(vision_flops(vs, small, 1) - FLOP_480) < 1.0
    tall = image_grid(vs, 1920, 1080)
    assert (tall.ph, tall.pw, tall.merged, tall.llm_tokens, tall.px) == (36, 64, MERGED, MERGED + 2, (896, 504))
    tby = _by(vision_ops(vs, tall, 1))
    assert tby["vis.0.qkv"].m == 36 * 64 and tby["vis.0.qk"].m == 36 * 64
    assert tby["vis.merger.2"].m == MERGED and tby["vis.merger.2"].k == 4 * TEXT
    assert abs(vision_flops(vs, tall, 1) - FLOP_1024) < 1.0
    tiny = image_grid(vs, 8, 8)
    assert tiny == image_grid(vs, 1, 1)
    assert (tiny.ph, tiny.pw, tiny.merged, tiny.llm_tokens, tiny.px) == (4, 4, 4, 6, (56, 56))
    yby = _by(vision_ops(vs, tiny, 1))
    _norm(yby["vis.pre_ln"], 16, H, 0)
    _gelu(yby["vis.0.gelu"], 16, INTER, 0, "bf16")
    _gelu(yby["vis.mgelu"], 4, TEXT, DEPTH, "bf16")
    assert abs(vision_flops(vs, tiny, 1) - FLOP_TINY) < 1.0
    _raises(lambda: image_grid(vs, 1, 400), "minimax image aspect ratio exceeds 200")
    g = image_grid(vs, 1024, 1024)
    two = _by(vision_ops(vs, g, 2, "bf16"))
    _norm(two["vis.pre_ln"], 2 * ROWS, H, 0)
    _gelu(two["vis.0.gelu"], 2 * ROWS, INTER, 0, "bf16")
    _gelu(two["vis.proj_gelu"], 2 * ROWS, TEXT, DEPTH, "bf16")
    _gelu(two["vis.mgelu"], 2 * MERGED, TEXT, DEPTH, "bf16")
    assert two["vis.0.qkv"].m == 2 * ROWS and two["vis.0.qkv"].vec == 2 * ROWS * 3 * H * 3
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * ROWS and two["vis.merger.2"].m == 2 * MERGED
    pair = _by(vision_ops(vs, g, 2, "fp8"))
    assert pair["vis.patch_embed"].a_fmt == "fp8" and pair["vis.patch_embed"].w_fmt == "fp32"
    assert pair["vis.patch_embed"].w_bits == 32.0
    assert pair["vis.0.qkv"].a_fmt == "fp8" and pair["vis.0.qkv"].w_fmt == "bf16"
    assert pair["vis.0.mlp_in"].a_fmt == "fp8" and pair["vis.merger.2"].a_fmt == "fp8"
    assert pair["vis.merger.2"].w_fmt == "bf16" and pair["vis.merger.2"].w_bits == 16.0
    assert pair["vis.0.gelu"].a_fmt == "bf16" and pair["vis.proj_gelu"].a_fmt == "bf16"
    assert pair["vis.mgelu"].a_fmt == "bf16" and pair["vis.pre_ln"].a_fmt == "fp32"
    assert pair["vis.0.softmax"].sfu == 0.0
    flash = get_model("minimax-m3.1-flash")
    assert flash.fmt("expert").bits < 5.0
    assert 8.0 <= flash.fmt("attn").bits <= 8.5
    ev = evaluate(Scenario(model="minimax-m3", serving=Serving(phase="prefill", prompt=32, images=1)))
    assert ev.vision["image_tokens"] == MERGED + 2
    for mid in ("qwen3-vl-2b", "ling-3.0-flash-vl", "ernie-4.5-vl-28b-a3b", "qwen3.5-9b"):
        other = get_model(mid)
        assert other.vision is not None and other.vision.family != "minimax"
        assert other.vision.ln_mode != "minimax"
        assert "pre_layrnorm" not in " ".join(other.notes)


def test_minimax_refuses_other_shapes():
    rel = load_release(HF["minimax-m3"])
    base = rel["config"]
    bare = vision_spec(HF["minimax-m3"], base)
    assert bare is not None and bare.ln_mode == "minimax" and bare.vec_fmt == ""
    assert bare.params == PARAMS and bare.w_Bpp == 2.0
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "minimax vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "minimax vision activation dtype is not a vision float")
    billed = replace(bare, vec_fmt="bf16")
    _raises(lambda: vision_ops(replace(billed, gated=True), g, 1),
            "minimax vision block is not fused qkv plus the projector")
    _raises(lambda: vision_ops(replace(billed, post=((H, TEXT), (TEXT, TEXT), (4 * H, TEXT), (TEXT, TEXT))), g, 1),
            "minimax vision block is not fused qkv plus the projector")
    _raises(lambda: vision_ops(replace(billed, qkv=H * 2), g, 1),
            "minimax vision block is not fused qkv plus the projector")
    silu = copy.deepcopy(base)
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF["minimax-m3"], silu), "minimax vision hidden_act is not gelu")
    quick = copy.deepcopy(base)
    quick["projector_hidden_act"] = "quick_gelu"
    _raises(lambda: vision_spec(HF["minimax-m3"], quick), "minimax projector activation is not gelu")
    nobias = copy.deepcopy(base)
    nobias["multimodal_projector_bias"] = False
    _raises(lambda: vision_spec(HF["minimax-m3"], nobias), "minimax projector bias is not true")
