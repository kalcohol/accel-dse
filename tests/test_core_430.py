"""0.430: Step-5 Perception Encoder LayerNorm.

ln_pre and each block's ln_1 / ln_2 are nn.LayerNorm. The releases omit
use_ln_pre (class default True) and set use_ln_post false. quick_gelu and
LayerScale stay in the GEMM tail. Vector flops stay 0.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

IDS = ("step-5-preview", "step-5-preview-nvfp4")
HF = "stepfun-ai/Step-5-Preview"
H, DEPTH, HEADS, INTER, TEXT = 1536, 47, 16, 8960, 4096
OVER = 52 * 52
CROP = 36 * 36
PARAMS = 1_981_014_784
FLOP_OVER = 11_634_513_346_560
FLOP_CROP4 = 31_831_995_777_024
FLOP_CROP8 = 52_029_478_207_488


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


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _lns(ops, prefix: str) -> list:
    found = []
    for op in ops:
        if op.name == prefix + "ln_pre":
            found.append(op)
            continue
        if not op.name.startswith(prefix):
            continue
        rest = op.name[len(prefix):]
        if rest[:1].isdigit() and (rest.endswith(".attn_ln") or rest.endswith(".mlp_ln")):
            found.append(op)
    return found


def test_step5_bills_layernorm():
    models = [get_model(mid) for mid in IDS]
    assert models[0].vision == models[1].vision
    for m in models:
        vs = m.vision
        assert vs.family == "fixed" and vs.ln_mode == "step5" and vs.ln_pre is True
        assert vs.post_ln is False and vs.gated is False and vs.vec_fmt == ""
        assert vs.hidden == H and vs.depth == DEPTH and vs.heads == HEADS and vs.inter == INTER
        assert vs.qkv == H and vs.params == PARAMS
        assert m.param_check()["rel_err"] == 0.0
        rel = load_release(m.hf_id)
        vc = rel["config"]["vision_config"]
        assert vc["hidden_act"] == "quick_gelu" and vc["use_ln_post"] is False
        assert "use_ln_pre" not in vc and vc["use_cls_token"] is False
        assert vc["pool_type"] == "none" and vc["ls_init_value"] == 0.1
        assert rel["config"]["projector_bias"] is False
        role = rel["roles"]["vision"]
        assert list(role) == ["bf16"] and role["bf16"]["params"] == PARAMS
        note = " ".join(m.notes)
        assert "nn.LayerNorm" in note and "quick_gelu" in note and "LayerScale" in note
        assert "use_ln_pre" in note and "use_ln_post" in note and "fp32" in note
    vs = models[0].vision
    g = image_grid(vs, 1, 1)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.views) == (52, 52, 169, 171, ((52, 52, 1),))
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _norm(by["vis.ln_pre"], OVER, H, 0)
    _norm(by["vis.0.attn_ln"], OVER, H, 0)
    _norm(by["vis.0.mlp_ln"], OVER, H, 0)
    _norm(by[f"vis.{DEPTH - 1}.attn_ln"], OVER, H, DEPTH - 1)
    _norm(by[f"vis.{DEPTH - 1}.mlp_ln"], OVER, H, DEPTH - 1)
    assert by["vis.0.qkv"].m == OVER and by["vis.0.qkv"].n == 3 * H
    assert by["vis.0.qkv"].vec == OVER * 3 * H * 3
    assert by["vis.0.qkv"].a_fmt == "bf16" and by["vis.0.qkv"].w_fmt == "bf16"
    assert by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.mlp_in"].n == INTER and by["vis.0.mlp_in"].vec == OVER * INTER * 3
    assert by["vis.0.qk"].m == OVER and by["vis.0.qk"].count == HEADS
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * OVER * OVER * 5
    assert (by["vis.merger.0"].m, by["vis.merger.0"].k, by["vis.merger.0"].n) == (676, 9 * H, 2 * H)
    assert by["vis.merger.0"].vec == 676 * (2 * H) * 3
    assert (by["vis.merger.1"].m, by["vis.merger.1"].k, by["vis.merger.1"].n) == (169, 18 * H, 4 * H)
    assert by["vis.merger.1"].vec == 169 * (4 * H) * 3
    assert (by["vis.merger.2"].m, by["vis.merger.2"].k, by["vis.merger.2"].n) == (169, 4 * H, TEXT)
    assert by["vis.merger.2"].vec == 169 * TEXT * 3
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") < names.index("vis.ln_pre") < names.index("vis.0.attn_ln")
    assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index("vis.0.qk") < names.index("vis.0.pv") < names.index("vis.0.softmax")
    assert names.index("vis.0.softmax") < names.index("vis.merger.0")
    assert len(_lns(ops, "vis.")) == 1 + 2 * DEPTH
    for absent in ("vis.ln_post", "vis.0.gelu", "vis.0.silu", "vis.0.relu", "vis.0.quick_gelu",
                   "vis.0.ls1", "vis.0.ls2", "vis.0.mlp_mul"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) - FLOP_OVER) < 1.0
    small = image_grid(vs, 640, 480)
    assert small.llm_tokens == 171 and small.views == ((52, 52, 1),)
    assert abs(vision_flops(vs, small, 1) - FLOP_OVER) < 1.0


def test_step5_crops_two_images_and_dtype():
    vs = get_model("step-5-preview").vision
    square = image_grid(vs, 1024, 1024)
    wide = image_grid(vs, 1024, 768)
    assert square.views == wide.views == ((52, 52, 1), (36, 36, 4))
    assert square.llm_tokens == wide.llm_tokens == 504 and square.merged == 493
    ops = vision_ops(vs, square, 1, "bf16")
    by = _by(ops)
    _norm(by["vis.ln_pre"], OVER, H, 0)
    _norm(by["vis.v1.ln_pre"], 4 * CROP, H, 0)
    _norm(by["vis.v1.0.attn_ln"], 4 * CROP, H, 0)
    _norm(by["vis.v1.0.mlp_ln"], 4 * CROP, H, 0)
    assert by["vis.v1.0.qkv"].m == 4 * CROP and by["vis.v1.0.qk"].m == CROP
    assert by["vis.v1.0.qk"].count == 4 * HEADS
    assert by["vis.v1.merger.0"].m == 4 * CROP // 4
    assert "vis.ln_post" not in by and "vis.v1.ln_post" not in by
    assert len(_lns(ops, "vis.")) == 1 + 2 * DEPTH
    assert len(_lns(ops, "vis.v1.")) == 1 + 2 * DEPTH
    assert abs(vision_flops(vs, square, 1) - FLOP_CROP4) < 1.0
    assert abs(vision_flops(vs, wide, 1) - FLOP_CROP4) < 1.0
    tall = image_grid(vs, 1920, 1080)
    assert tall.views == ((52, 52, 1), (36, 36, 8)) and tall.llm_tokens == 836 and tall.merged == 817
    tby = _by(vision_ops(vs, tall, 1))
    _norm(tby["vis.v1.ln_pre"], 8 * CROP, H, 0)
    assert tby["vis.v1.0.qkv"].m == 8 * CROP and tby["vis.v1.0.qk"].count == 8 * HEADS
    assert tby["vis.v1.merger.0"].m == 8 * CROP // 4
    assert abs(vision_flops(vs, tall, 1) - FLOP_CROP8) < 1.0
    two = _by(vision_ops(vs, image_grid(vs, 1, 1), 2, "bf16"))
    _norm(two["vis.ln_pre"], 2 * OVER, H, 0)
    assert two["vis.0.qkv"].m == 2 * OVER and two["vis.0.qkv"].vec == 2 * OVER * 3 * H * 3
    assert two["vis.0.qk"].m == OVER and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * 676
    pair = _by(vision_ops(vs, square, 2, "fp8"))
    _norm(pair["vis.v1.ln_pre"], 2 * 4 * CROP, H, 0)
    assert pair["vis.v1.0.attn_ln"].a_fmt == "fp32" and pair["vis.ln_pre"].a_fmt == "fp32"
    assert pair["vis.0.qkv"].a_fmt == "fp8" and pair["vis.0.qkv"].vec == 2 * OVER * 3 * H * 3
    assert pair["vis.0.qkv"].w_fmt == "bf16" and pair["vis.0.qkv"].w_bits == 16.0
    assert pair["vis.v1.0.mlp_in"].a_fmt == "fp8" and pair["vis.v1.merger.2"].a_fmt == "fp8"
    assert pair["vis.v1.0.qk"].count == 2 * 4 * HEADS
    ev = evaluate(Scenario(model="step-5-preview", serving=Serving(phase="prefill", prompt=32, images=1)))
    assert ev.vision["image_tokens"] == 504


def test_step5_refuses_other_shapes_and_optional_post_norm():
    rel = load_release(HF)
    base = rel["config"]
    assert "use_ln_pre" not in base["vision_config"]
    bare = vision_spec(HF, base)
    assert bare is not None and bare.ln_mode == "step5" and bare.ln_pre is True and bare.post_ln is False
    assert bare.params == PARAMS and bare.vec_fmt == ""
    g = image_grid(bare, 1, 1)
    billed = _by(vision_ops(bare, g, 1))
    assert "vis.ln_pre" in billed and "vis.ln_post" not in billed

    off = copy.deepcopy(base)
    off["vision_config"]["use_ln_pre"] = False
    dropped = vision_spec(HF, off)
    assert dropped is not None and dropped.ln_pre is False and dropped.params == PARAMS - 2 * H
    gone = _by(vision_ops(dropped, image_grid(dropped, 1, 1), 1))
    assert "vis.ln_pre" not in gone and "vis.0.attn_ln" in gone and "vis.0.mlp_ln" in gone

    post = copy.deepcopy(base)
    post["vision_config"]["use_ln_post"] = True
    added = vision_spec(HF, post)
    assert added is not None and added.post_ln is True and added.params == PARAMS + 2 * H
    extra = _by(vision_ops(added, image_grid(added, 1, 1), 1))
    _norm(extra["vis.ln_post"], OVER, H, DEPTH)
    names = [op.name for op in vision_ops(added, image_grid(added, 1, 1), 1)]
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.ln_post") < names.index("vis.merger.0")

    omitted = copy.deepcopy(base)
    omitted["vision_config"].pop("use_ln_post")
    omitted["vision_config"].pop("hidden_act")
    omitted["vision_config"].pop("ls_init_value")
    omitted["vision_config"].pop("pool_type")
    kept = vision_spec(HF, omitted)
    assert kept is not None and kept.ln_pre is True and kept.post_ln is False and kept.params == PARAMS

    tanh = copy.deepcopy(base)
    tanh["vision_config"]["hidden_act"] = "gelu_pytorch_tanh"
    _raises(lambda: vision_spec(HF, tanh), "step5 vision hidden_act is not quick_gelu")
    cls = copy.deepcopy(base)
    cls["vision_config"]["use_cls_token"] = True
    _raises(lambda: vision_spec(HF, cls), "step5 vision use_cls_token is not false")
    pool = copy.deepcopy(base)
    pool["vision_config"]["pool_type"] = "mean"
    _raises(lambda: vision_spec(HF, pool), "step5 vision pool_type is not none")
    scale = copy.deepcopy(base)
    scale["vision_config"]["ls_init_value"] = None
    _raises(lambda: vision_spec(HF, scale), "step5 vision ls_init_value is missing")
    flag = copy.deepcopy(base)
    flag["vision_config"]["use_ln_pre"] = 1
    _raises(lambda: vision_spec(HF, flag), "step5 vision layernorm flag is not a bool")

    _raises(lambda: vision_ops(replace(bare, gated=True), g, 1),
            "step5 vision block is not fused qkv plus two stride-2 convolutions")
    _raises(lambda: vision_ops(replace(bare, post=bare.post[:2]), g, 1),
            "step5 vision block is not fused qkv plus two stride-2 convolutions")
    _raises(lambda: vision_ops(replace(bare, kv_heads=1), g, 1),
            "step5 vision block is not fused qkv plus two stride-2 convolutions")
