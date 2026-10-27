"""0.433: ERNIE-4.5-VL LayerNorm, resampler erf-GELU, and RMSNorm.

DFNRopeVisionBlock runs nn.LayerNorm before attention, before the MLP, and
once more after the blocks. VariableResolutionResamplerModel hardcodes
nn.GELU() (erf) and LayerNorm on the spatial and temporal linears, then
RMSNorm after the projector. The tower MLP stays quick_gelu. Vector flops
stay 0.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops, vision_spec

IDS28 = (
    "ernie-4.5-vl-28b-a3b-pt",
    "ernie-4.5-vl-28b-a3b",
    "ernie-4.5-vl-28b-a3b-base",
    "ernie-4.5-vl-28b-a3b-thinking",
)
IDS424 = (
    "ernie-4.5-vl-424b-a47b-pt",
    "ernie-4.5-vl-424b-a47b",
)
HF = "baidu/ERNIE-4.5-VL-28B-A3B-PT"
HF_THINK = "baidu/ERNIE-4.5-VL-28B-A3B-Thinking"
H = 1280
HEADS = 16
DEPTH = 32
SD = 5120
TD = 10240
TEXT28 = 2560
TEXT424 = 8192
FP16_PARAMS = 630_433_280
FP16_BYTES = 1_260_866_560
RES28 = 144_225_280
RES424 = 173_072_384
VIS28 = 774_658_560
VIS424 = 803_505_664
GEMM_1024 = 7_293_408_174_080
GEMM_1024_768 = 5_322_216_775_680
GEMM_640 = 2_083_069_829_120
GEMM_TINY = 21_310_177_280
GEMM_HD = 14_336_421_765_120
GEMM_VID2 = 12_105_155_543_040
GEMM_424 = 7_372_360_704_000


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


def _gelu(op, rows: int, width: int, af: str = "bf16") -> None:
    assert op.kind == "vector" and op.layer == DEPTH
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _rms(op, rows: int, width: int) -> None:
    assert op.kind == "vector" and op.layer == DEPTH
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _gemm(ops) -> float:
    return sum(op.flops for op in ops if op.kind == "gemm")


def test_ernie_vl_bills_ln_resampler_gelu_and_rms():
    models = [get_model(mid) for mid in IDS28]
    wide = [get_model(mid) for mid in IDS424]
    base = models[0]
    vs = base.vision
    assert vs is not None and vs.family == "ernievl" and vs.vec_fmt == "bf16"
    for m in models[1:]:
        assert m.vision == vs, m.id
        assert m.param_check()["rel_err"] == 0.0
    assert wide[0].vision == wide[1].vision
    assert wide[0].vision.vec_fmt == "bf16"
    assert wide[0].param_check()["rel_err"] == 0.0
    assert base.param_check()["rel_err"] == 0.0
    assert vs.params == VIS28 and wide[0].vision.params == VIS424
    assert vs.post[-1] == (SD, TEXT28) and wide[0].vision.post[-1] == (SD, TEXT424)
    assert vs.post[:4] == wide[0].vision.post[:4]
    assert vs.depth == DEPTH and vs.hidden == H and vs.heads == HEADS
    rel = load_release(base.hf_id)
    role = rel["roles"]["vision"]
    assert set(role) == {"fp16", "bf16"}
    assert role["fp16"]["params"] == FP16_PARAMS and role["fp16"]["bytes"] == FP16_BYTES
    assert role["bf16"]["params"] == RES28 and role["bf16"]["bytes"] == 2 * RES28
    assert rel["config"]["hidden_act"] == "silu"
    assert rel["config"]["vision_config"]["hidden_act"] == "quick_gelu"
    think = load_release(HF_THINK)
    assert set(think["roles"]["vision"]) == {"bf16"}
    assert think["roles"]["vision"]["bf16"]["params"] == VIS28
    assert think["roles"]["vision"]["bf16"]["bytes"] == 2 * VIS28
    rel424 = load_release(wide[0].hf_id)
    role424 = rel424["roles"]["vision"]
    assert role424["bf16"]["params"] == RES424 and role424["fp16"]["params"] == FP16_PARAMS
    note = " ".join(base.notes)
    assert "ERNIE-4.5-VL" in note and "nn.LayerNorm" in note and "quick_gelu" in note
    assert "nn.GELU()" in note and "RMSNorm" in note and "index_select" in note
    assert "Thinking" in note and vs.vec_fmt in note and "fp32" in note

    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.tframes) == (74, 74, 1369, 1371, 1)
    ops = vision_ops(vs, g, 1)
    by = _by(ops)
    rows = 74 * 74
    _ln(by["vis.0.attn_ln"], rows, H, 0)
    _ln(by["vis.0.mlp_ln"], rows, H, 0)
    _ln(by["vis.31.attn_ln"], rows, H, 31)
    _ln(by["vis.31.mlp_ln"], rows, H, 31)
    _ln(by["vis.out_ln"], rows, H, DEPTH)
    _gelu(by["vis.sp_gelu"], 1369, SD)
    _ln(by["vis.sp_ln"], 1369, SD, DEPTH)
    _gelu(by["vis.tm_gelu"], 1369, SD)
    _ln(by["vis.tm_ln"], 1369, SD, DEPTH)
    _rms(by["vis.after_rms"], 1369, TEXT28)
    assert by["vis.0.qkv"].m == rows and by["vis.0.qkv"].n == 3 * H
    assert by["vis.0.qkv"].vec == rows * 3 * H * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.qkv"].a_fmt == "bf16"
    assert by["vis.0.mlp_in"].n == SD and by["vis.0.mlp_in"].vec == rows * SD * 3
    assert [(by[f"vis.merger.{i}"].m, by[f"vis.merger.{i}"].k, by[f"vis.merger.{i}"].n)
            for i in range(5)] == [
                (1369, SD, SD), (1369, SD, SD), (1369, TD, SD), (1369, SD, SD), (1369, SD, TEXT28)]
    assert by["vis.merger.0"].vec == 1369 * SD * 3 and by["vis.merger.0"].w_fmt == "bf16"
    assert by["vis.merger.4"].vec == 1369 * TEXT28 * 3
    assert by["vis.0.qk"].m == rows and by["vis.0.qk"].count == HEADS
    assert by["vis.0.softmax"].sfu == 0 and by["vis.0.softmax"].vec == HEADS * rows * rows * 5
    names = [op.name for op in ops]
    assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index("vis.31.softmax") < names.index("vis.out_ln") < names.index("vis.merger.0")
    assert names.index("vis.merger.0") < names.index("vis.sp_gelu") < names.index("vis.merger.1")
    assert names.index("vis.merger.1") < names.index("vis.sp_ln") < names.index("vis.merger.2")
    assert names.index("vis.merger.2") < names.index("vis.tm_gelu") < names.index("vis.merger.3")
    assert names.index("vis.merger.3") < names.index("vis.tm_ln") < names.index("vis.merger.4")
    assert names.index("vis.merger.4") < names.index("vis.after_rms")
    lns = [op for op in ops if op.name.endswith("ln")]
    gels = [op for op in ops if "gelu" in op.name]
    rms = [op for op in ops if "rms" in op.name]
    assert len(lns) == 67 and len(gels) == 2 and len(rms) == 1
    assert all(op.flops == 0 and op.w_params == 0 for op in lns + gels + rms)
    for absent in ("vis.0.gelu", "vis.0.silu", "vis.0.quick_gelu", "vis.0.relu", "vis.mgelu"):
        assert absent not in by, absent
    assert _gemm(ops) == GEMM_1024
    assert sum(op.flops for op in ops if op.kind == "vector") == 0

    wide_ops = _by(vision_ops(wide[0].vision, g, 1))
    _ln(wide_ops["vis.0.attn_ln"], rows, H, 0)
    _gelu(wide_ops["vis.sp_gelu"], 1369, SD)
    _rms(wide_ops["vis.after_rms"], 1369, TEXT424)
    assert wide_ops["vis.merger.4"].n == TEXT424
    assert wide_ops["vis.merger.4"].vec == 1369 * TEXT424 * 3
    assert _gemm(vision_ops(wide[0].vision, g, 1)) == GEMM_424

    g768 = image_grid(vs, 1024, 768)
    assert (g768.ph, g768.pw, g768.merged, g768.llm_tokens) == (54, 74, 999, 1001)
    b768 = _by(vision_ops(vs, g768, 1))
    _ln(b768["vis.out_ln"], 54 * 74, H, DEPTH)
    _gelu(b768["vis.tm_gelu"], 999, SD)
    _rms(b768["vis.after_rms"], 999, TEXT28)
    assert b768["vis.0.qkv"].m == 54 * 74 and _gemm(vision_ops(vs, g768, 1)) == GEMM_1024_768

    g640 = image_grid(vs, 640, 480)
    assert (g640.ph, g640.pw, g640.merged) == (34, 46, 391)
    b640 = _by(vision_ops(vs, g640, 1))
    _ln(b640["vis.0.attn_ln"], 34 * 46, H, 0)
    _gelu(b640["vis.sp_gelu"], 391, SD)
    assert _gemm(vision_ops(vs, g640, 1)) == GEMM_640

    tiny = image_grid(vs, 8, 8)
    assert (tiny.ph, tiny.pw, tiny.merged, tiny.llm_tokens) == (4, 4, 4, 6)
    bt = _by(vision_ops(vs, tiny, 1))
    _ln(bt["vis.out_ln"], 16, H, DEPTH)
    _gelu(bt["vis.sp_gelu"], 4, SD)
    _rms(bt["vis.after_rms"], 4, TEXT28)
    assert _gemm(vision_ops(vs, tiny, 1)) == GEMM_TINY

    ghd = image_grid(vs, 1920, 1080)
    assert (ghd.ph, ghd.pw, ghd.merged, ghd.llm_tokens) == (78, 138, 2691, 2693)
    bhd = _by(vision_ops(vs, ghd, 1))
    _ln(bhd["vis.out_ln"], 78 * 138, H, DEPTH)
    _gelu(bhd["vis.tm_gelu"], 2691, SD)
    assert bhd["vis.0.qkv"].m == 78 * 138 and bhd["vis.merger.2"].m == 2691
    assert _gemm(vision_ops(vs, ghd, 1)) == GEMM_HD

    two = _by(vision_ops(vs, g, 2))
    _ln(two["vis.0.attn_ln"], 2 * rows, H, 0)
    _ln(two["vis.out_ln"], 2 * rows, H, DEPTH)
    _gelu(two["vis.sp_gelu"], 2 * 1369, SD)
    _ln(two["vis.tm_ln"], 2 * 1369, SD, DEPTH)
    _rms(two["vis.after_rms"], 2 * 1369, TEXT28)
    assert two["vis.0.qkv"].m == 2 * rows and two["vis.0.qkv"].vec == 2 * by["vis.0.qkv"].vec
    assert two["vis.0.qk"].m == rows and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * 1369 and two["vis.merger.4"].m == 2 * 1369
    assert two["vis.merger.0"].vec == 2 * by["vis.merger.0"].vec

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.merger.0"].a_fmt == "fp8"
    assert forced["vis.merger.4"].a_fmt == "fp8"
    assert forced["vis.0.qkv"].w_fmt == "bf16" and forced["vis.merger.2"].w_bits == 16.0
    assert forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
    assert forced["vis.merger.0"].vec == by["vis.merger.0"].vec
    assert forced["vis.sp_gelu"].a_fmt == "bf16" and forced["vis.tm_gelu"].a_fmt == "bf16"
    assert forced["vis.sp_gelu"].vec == by["vis.sp_gelu"].vec
    assert forced["vis.0.attn_ln"].a_fmt == "fp32" and forced["vis.out_ln"].a_fmt == "fp32"
    assert forced["vis.sp_ln"].a_fmt == "fp32" and forced["vis.after_rms"].a_fmt == "fp32"

    pre = video_pre(base)
    two_f = video_grid(vs, pre, 2, 1024, 1024)
    assert (two_f.slice.ph, two_f.slice.pw, two_f.slice.merged, two_f.slice.tframes) == (68, 68, 1156, 2)
    vid = _by(vision_ops(vs, two_f.slice, 1))
    _ln(vid["vis.0.attn_ln"], 2 * 68 * 68, H, 0)
    _ln(vid["vis.out_ln"], 2 * 68 * 68, H, DEPTH)
    _gelu(vid["vis.sp_gelu"], 2 * 1156, SD)
    _ln(vid["vis.sp_ln"], 2 * 1156, SD, DEPTH)
    _gelu(vid["vis.tm_gelu"], 1156, SD)
    _ln(vid["vis.tm_ln"], 1156, SD, DEPTH)
    _rms(vid["vis.after_rms"], 1156, TEXT28)
    assert [(vid[f"vis.merger.{i}"].m, vid[f"vis.merger.{i}"].k, vid[f"vis.merger.{i}"].n)
            for i in range(5)] == [
                (2 * 1156, SD, SD), (2 * 1156, SD, SD), (1156, TD, SD),
                (1156, SD, SD), (1156, SD, TEXT28)]
    assert vid["vis.0.qk"].count == 2 * HEADS and vid["vis.0.qk"].m == 68 * 68
    assert vid["vis.0.softmax"].sfu == 0
    assert "vis.tpool" not in vid
    assert _gemm(vision_ops(vs, two_f.slice, 1)) == GEMM_VID2
    pair = _by(vision_ops(vs, two_f.slice, 2))
    _ln(pair["vis.0.attn_ln"], 4 * 68 * 68, H, 0)
    _gelu(pair["vis.sp_gelu"], 4 * 1156, SD)
    _rms(pair["vis.after_rms"], 2 * 1156, TEXT28)
    assert pair["vis.0.qk"].m == 68 * 68 and pair["vis.0.qk"].count == 4 * HEADS

    hd = video_grid(vs, pre, 32, 1280, 720)
    assert (hd.slice.ph, hd.slice.pw, hd.slice.merged, hd.slice.tframes) == (52, 92, 1196, 32)
    hb = _by(vision_ops(vs, hd.slice, 1))
    _ln(hb["vis.out_ln"], 32 * 52 * 92, H, DEPTH)
    _gelu(hb["vis.sp_gelu"], 32 * 1196, SD)
    _ln(hb["vis.tm_ln"], 16 * 1196, SD, DEPTH)
    _rms(hb["vis.after_rms"], 16 * 1196, TEXT28)
    assert hb["vis.merger.2"].m == 16 * 1196 and hb["vis.merger.2"].k == TD
    assert hb["vis.0.qk"].count == 32 * HEADS and hb["vis.0.qk"].m == 52 * 92

    clip = evaluate(Scenario(model="ernie-4.5-vl-28b-a3b-pt", serving=Serving(
        phase="prefill", batch=1, prompt=32, images=1)))
    assert clip.vision["tokens_per_image"] == 1371


def test_ernie_vl_refuses_other_activations_and_a_bad_role():
    rel = load_release(HF)
    silu = copy.deepcopy(rel["config"])
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF, silu), "ernie vision hidden_act is not quick_gelu")
    omitted = copy.deepcopy(rel["config"])
    omitted["vision_config"].pop("hidden_act")
    _raises(lambda: vision_spec(HF, omitted), "ernie vision hidden_act is not quick_gelu")

    vs = get_model("ernie-4.5-vl-28b-a3b-pt").vision
    g = image_grid(vs, 1024, 1024)
    _raises(lambda: vision_ops(replace(vs, vec_fmt="fp8"), g, 1),
            "ernie vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(vs, vec_fmt=""), g, 1),
            "ernie vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(vs, post=vs.post[:4]), g, 1),
            "ernie vision block is not fused qkv plus the resampler")
    _raises(lambda: vision_ops(replace(vs, gated=True), g, 1),
            "ernie vision block is not fused qkv plus the resampler")

    mixed = copy.deepcopy(rel)
    mixed["roles"]["vision"]["fp32"] = {"params": 8, "bytes": 32}
    _raises(lambda: from_release("synth-mix", HF, rel=mixed, cfg=mixed["config"]),
            "ernie vision role is not fp16 tower plus bf16 resampler")
    wrong = copy.deepcopy(rel)
    wrong["roles"]["vision"]["bf16"] = {"params": RES28 + 1, "bytes": 2 * (RES28 + 1)}
    _raises(lambda: from_release("synth-res", HF, rel=wrong, cfg=wrong["config"]),
            "ernie bf16 vision role is not the resampler")
    think = copy.deepcopy(load_release(HF_THINK))
    think["roles"]["vision"]["bf16"] = {"params": VIS28 - 1, "bytes": 2 * (VIS28 - 1)}
    _raises(lambda: from_release("synth-think", HF_THINK, rel=think, cfg=think["config"]),
            "ernie bf16 vision role is not the tower")

    step = get_model("step-5-preview").vision
    assert step is not None and step.family == "fixed"
    quiet = vision_ops(step, image_grid(step, 8, 8), 1)
    assert any(op.name == "vis.ln_pre" for op in quiet)
    assert not any(op.name in ("vis.sp_gelu", "vis.after_rms", "vis.out_ln") for op in quiet)
