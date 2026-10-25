"""0.405: LongCat vision SwiGLU bills the inner RMSNorm the class actually runs.

SwiGLU.forward is fc2(RMSNorm(silu(fc1(x)) * fc3(x))). The norm upcasts to
fp32 and applies one rsqrt per patch token. The weight is already in the
tower parameter total. Projector rows (ph * pw / 2) are the wrong M.
Other ViT families do not grow this op. q/k and residual RMSNorm stay in
the shared GEMM epilogue.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import image_grid, vision_ops

INNER = 3456
DEPTH = 32


def test_longcat_vision_bills_the_swiglu_inner_rmsnorm_on_patch_tokens():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "longcatvit" and vs.gated and vs.inter == INNER
    assert vs.depth == DEPTH
    g = image_grid(vs, 1024, 1024)
    tokens = g.ph * g.pw
    assert tokens == 72 * 72 == 2 * g.merged
    ops = vision_ops(vs, g, 1, "fp8")
    assert ops and all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    norms = [op for op in ops if op.name.endswith(".mlp_norm")]
    assert len(norms) == DEPTH
    assert sum(op.sfu for op in ops) == DEPTH * tokens
    for li, op in enumerate(norms):
        assert op.name == f"vis.{li}.mlp_norm"
        assert op.kind == "vector" and op.layer == li
        assert op.vec == tokens * INNER * 4 and op.sfu == tokens
        assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
        assert names.index(f"vis.{li}.mlp_in") + 1 == names.index(op.name)
        assert names.index(op.name) + 1 == names.index(f"vis.{li}.mlp_out")
        assert op.vec != g.merged * INNER * 4
    wide = vision_ops(vs, g, 2)
    wide_norms = [op for op in wide if op.name.endswith(".mlp_norm")]
    assert len(wide_norms) == DEPTH
    assert all(op.vec == 2 * tokens * INNER * 4 and op.sfu == 2 * tokens for op in wide_norms)
    assert vision_ops(vs, g, 0) == []
    text = " ".join(m.notes)
    fp_text = " ".join(fp.notes)
    assert "内层 RMSNorm" in text and "内层 RMSNorm" in fp_text
    other = get_model("step-5-preview").vision
    assert other is not None and other.family != "longcatvit"
    og = image_grid(other, 1024, 1024)
    assert og is not None
    assert not any(op.name.endswith(".mlp_norm") for op in vision_ops(other, og, 1))
