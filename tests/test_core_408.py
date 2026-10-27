"""0.408: LongCat rope2d bills the inv_freq outer and the cos/sin repeat.

VisionRotaryEmbedding2D.forward_ runs torch.outer(seq, inv_freq) once per
encoder forward. seqlen is max(t, h, w); a still image has t = 1, so the side
is max(ph, pw). The table is shared across layers and across images of that
grid. apply_rotary_pos_emb_vision then repeat(1, 1, 2) on cos and on sin, for
q and again for k. Neither op is a GEMM. emb_mult stays 1 and dropout stays 0.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops, image_grid, vision_ops

HIDDEN = 1280
DEPTH = 32
HEADS = 16
HD = HIDDEN // HEADS
DIM = HD // 2
INV = (DIM + 1) // 2


def test_longcat_rope_bills_outer_once_and_repeat_per_layer():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "longcatvit" and vs.gated
    assert vs.head_dim == HD and INV == 20 and INV != HD // 2
    g = image_grid(vs, 1024, 1024)
    tokens = g.ph * g.pw
    assert (g.ph, g.pw, tokens) == (72, 72, 72 * 72)
    side = max(g.ph, g.pw)
    ops = vision_ops(vs, g, 1, "fp8")
    assert ops and all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") + 1 == names.index("vis.rope_outer")
    assert names.index("vis.rope_outer") < names.index("vis.0.norm1")
    assert names.count("vis.rope_outer") == 1
    outer = next(op for op in ops if op.name == "vis.rope_outer")
    assert outer.kind == "vector" and outer.layer == 0
    assert outer.vec == side * INV
    assert outer.vec == 72 * 20
    assert outer.vec != tokens * INV and outer.vec != DEPTH * side * INV
    assert outer.sfu == 0 and outer.a_fmt == "fp32"
    assert outer.w_params == 0 and outer.flops == 0
    for li in range(DEPTH):
        assert names.index(f"vis.{li}.rope") + 1 == names.index(f"vis.{li}.rope_repeat")
        assert names.index(f"vis.{li}.rope_repeat") + 1 == names.index(f"vis.{li}.q_norm")
        rep = next(op for op in ops if op.name == f"vis.{li}.rope_repeat")
        rope = next(op for op in ops if op.name == f"vis.{li}.rope")
        assert rep.kind == "vector" and rep.layer == li
        assert rep.vec == 4 * tokens * HD
        assert rep.vec != 2 * tokens * HD and rep.vec != tokens * (HD // 2)
        assert rep.sfu == 0 and rep.a_fmt == "fp32"
        assert rep.w_params == 0 and rep.flops == 0
        assert rope.sfu == 2 * tokens * (HD // 2) * 2
        assert rope.vec != rep.vec
    assert not any("gather" in op.name or "pos_ids" in op.name or "emb_mult" in op.name
                   or "drop" in op.name for op in ops)
    wide = vision_ops(vs, g, 2)
    assert next(op for op in wide if op.name == "vis.rope_outer").vec == outer.vec
    assert next(op for op in wide if op.name == "vis.0.rope_repeat").vec == 2 * 4 * tokens * HD
    assert vision_ops(vs, g, 0) == []
    rect = image_grid(vs, 896, 448)
    assert rect.ph != rect.pw and max(rect.ph, rect.pw) == 64
    ro = next(op for op in vision_ops(vs, rect, 3) if op.name == "vis.rope_outer")
    rr = next(op for op in vision_ops(vs, rect, 3) if op.name == "vis.0.rope_repeat")
    assert ro.vec == max(rect.ph, rect.pw) * INV
    assert ro.vec != rect.ph * rect.pw * INV
    assert rr.vec == 4 * 3 * rect.ph * rect.pw * HD
    text = " ".join(m.notes)
    fp_text = " ".join(fp.notes)
    for needle in ("torch.outer", "arange", "repeat(1, 1, 2)", "不记成 GEMM", "emb_mult", "head_dim/2"):
        assert needle in text and needle in fp_text
    other = get_model("step-5-preview").vision
    assert other is not None and other.family != "longcatvit"
    og = image_grid(other, 1024, 1024)
    assert og is not None
    onames = [op.name for op in vision_ops(other, og, 1)]
    assert not any(name == "vis.rope_outer" or name.endswith(".rope_repeat") for name in onames)


def test_longcat_audio_still_skips_unit_scale_and_zero_dropout():
    vs = get_model("longcat-flash-omni").vision
    assert vs is not None and vs.audio is not None
    ops = audio_ops(vs.audio, 1.0, 1, "fp8")
    assert ops
    assert not any("rope" in op.name or "emb_mult" in op.name or "drop" in op.name for op in ops)
    assert any(op.name == "aud.proj_gelu" for op in ops)
    assert any(op.name.endswith(".relu") for op in ops)
