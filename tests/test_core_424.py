"""0.424: Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp vision LayerNorm and merger erf-GELU.

nn.LayerNorm reduces in fp32, one rsqrt per patch row. nn.GELU() after merger
fc1 is the erf form at 4 * hidden. The block MLP stays gelu_pytorch_tanh.
The shared GEMM tail is unchanged. A vision role with more than one dtype
is not billed.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import entries, get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid_of, vision_flops, vision_ops, vision_spec

# id, hidden, depth, heads, intermediate, merger N, vision model_type, 1024^2 GEMM GFLOP
CASES = (
    ("qwen3.5-9b", 1152, 27, 16, 4304, 4096, "qwen3_5", 5551.480111104),
    ("qwen3.5-397b-a17b", 1152, 27, 16, 4304, 4096, "qwen3_5_moe", 5551.480111104),
    ("ornith-1.5-9b", 1152, 27, 16, 4304, 4096, "qwen3_5_vision", 5551.480111104),
    ("ornith-1.5-35b-a3b", 1152, 27, 16, 4304, 2048, "qwen3_5_moe_vision", 5532.152758272),
    ("qwen3.8-27b", 1152, 27, 16, 4304, 5120, "qwen3_5", 5561.143787520),
    ("qwen3.8-27b-fp8", 1152, 27, 16, 4304, 5120, "qwen3_5", 5561.143787520),
    ("qwen3.8-flash-next", 1152, 27, 16, 4304, 2560, "qwen4_exp", 5536.984596480),
    ("intern-decision-0.8b", 768, 12, 12, 3072, 1024, "qwen3_5", 1349.693472768),
)
TYPES = ("qwen3_5", "qwen3_5_moe", "qwen4_exp", "qwen3_5_vision", "qwen3_5_moe_vision")


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


def _gelu(op, rows: int, width: int, layer: int, fmt: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == fmt and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_qwen35_vision_bills_layernorm_and_merger_erf_gelu():
    for mid, h, depth, heads, inter, out_n, mt, gflop in CASES:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "qwen" and vs.ln_mode == "qwen35", mid
        assert vs.vec_fmt == "bf16" and vs.deepstack == 0 and vs.gated is False, mid
        assert vs.hidden == h and vs.depth == depth and vs.heads == heads and vs.inter == inter, mid
        hm = h * 4
        assert vs.post == ((hm, hm), (hm, out_n)), mid
        rel = load_release(m.hf_id)
        vc = rel["config"]["vision_config"]
        assert vc["model_type"] == mt and vc["hidden_act"] == "gelu_pytorch_tanh", mid
        assert vc.get("deepstack_visual_indexes") in (None, [], ()), mid
        assert set(rel["roles"]["vision"]) == {"bf16"}, mid
        assert "Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp" in " ".join(m.notes), mid
        g = image_grid(vs, 1024, 1024)
        assert (g.ph, g.pw, g.merged) == (64, 64, 1024), mid
        ops = vision_ops(vs, g, 1, "bf16")
        by = _by(ops)
        rows = 64 * 64
        _ln(by["vis.0.attn_ln"], rows, h, 0)
        _ln(by["vis.0.mlp_ln"], rows, h, 0)
        _ln(by[f"vis.{depth - 1}.attn_ln"], rows, h, depth - 1)
        _ln(by["vis.pre_ln"], rows, h, depth)
        _gelu(by["vis.mgelu"], 1024, hm, depth, "bf16")
        assert by["vis.0.qkv"].vec == rows * (3 * h) * 3 and by["vis.0.qkv"].n == 3 * h, mid
        assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0, mid
        assert by["vis.0.qkv"].a_fmt == "bf16", mid
        assert by["vis.0.mlp_in"].vec == rows * inter * 3 and by["vis.0.mlp_in"].n == inter, mid
        assert by["vis.merger.0"].m == 1024 and by["vis.merger.0"].k == hm and by["vis.merger.0"].n == hm, mid
        assert by["vis.merger.0"].vec == 1024 * hm * 3, mid
        assert by["vis.merger.1"].k == hm and by["vis.merger.1"].n == out_n, mid
        assert by["vis.0.qk"].m == rows and by["vis.0.qk"].count == heads, mid
        assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == heads * rows * rows * 5, mid
        names = [op.name for op in ops]
        assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
        assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
        assert names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
        assert names.index("vis.pre_ln") < names.index("vis.merger.0") < names.index("vis.mgelu")
        assert names.index("vis.mgelu") < names.index("vis.merger.1")
        lns = [op for op in ops if op.name.endswith("attn_ln") or op.name.endswith("mlp_ln")
               or op.name == "vis.pre_ln"]
        gels = [op for op in ops if op.name.endswith("gelu")]
        assert len(lns) == 2 * depth + 1 and len(gels) == 1, mid
        for absent in ("vis.0.gelu", "vis.0.silu", "vis.0.relu", "vis.0.norm1", "vis.ds0.ln",
                       "vis.ds0.gelu", "vis.pgelu", "vis.post_ln", "vis.ln_q", "vis.proj_gelu"):
            assert absent not in by, (mid, absent)
        assert abs(vision_flops(vs, g, 1) / 1e9 - gflop) < 1e-6, mid

        wide = image_grid(vs, 1920, 1080)
        assert (wide.ph, wide.pw, wide.merged) == (68, 120, 2040), mid
        wb = _by(vision_ops(vs, wide, 1, "bf16"))
        _ln(wb["vis.pre_ln"], 68 * 120, h, depth)
        _gelu(wb["vis.mgelu"], 2040, hm, depth, "bf16")
        assert wb["vis.0.qk"].m == 68 * 120 and wb["vis.merger.1"].n == out_n, mid

        small = image_grid(vs, 640, 480)
        assert (small.ph, small.pw, small.merged) == (30, 40, 300), mid
        sb = _by(vision_ops(vs, small, 1, "bf16"))
        _ln(sb["vis.0.attn_ln"], 1200, h, 0)
        _gelu(sb["vis.mgelu"], 300, hm, depth, "bf16")

        two = _by(vision_ops(vs, g, 2, "bf16"))
        assert two["vis.0.attn_ln"].vec == 2 * by["vis.0.attn_ln"].vec
        assert two["vis.0.attn_ln"].sfu == 2 * rows
        assert two["vis.pre_ln"].sfu == 2 * rows
        assert two["vis.mgelu"].vec == 2 * by["vis.mgelu"].vec
        assert two["vis.0.qk"].m == rows and two["vis.0.qk"].count == 2 * heads, mid
        assert two["vis.merger.0"].m == 2 * 1024, mid

        forced = _by(vision_ops(vs, g, 1, "fp8"))
        assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.qkv"].w_fmt == "bf16", mid
        assert forced["vis.0.qkv"].w_bits == 16.0 and forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
        assert forced["vis.0.mlp_in"].a_fmt == "fp8" and forced["vis.merger.0"].a_fmt == "fp8"
        assert forced["vis.merger.0"].w_fmt == "bf16" and forced["vis.merger.0"].w_bits == 16.0
        assert forced["vis.mgelu"].a_fmt == "bf16" and forced["vis.mgelu"].vec == by["vis.mgelu"].vec
        assert forced["vis.0.attn_ln"].a_fmt == "fp32" and forced["vis.pre_ln"].vec == by["vis.pre_ln"].vec


def test_qwen35_video_slice_uses_the_slice_rows():
    for mid, h, depth, heads, out_n in (
            ("intern-decision-0.8b", 768, 12, 12, 1024),
            ("qwen3.8-27b", 1152, 27, 16, 5120),
            ("qwen3.8-flash-next", 1152, 27, 16, 2560)):
        m = get_model(mid)
        clip = video_grid_of(m, Scenario(model=mid).replace(
            "serving.videos", 1).replace("serving.video_frames", 4).serving)
        assert clip.t == 2 and clip.slice.tframes == 1, mid
        assert (clip.slice.ph, clip.slice.pw, clip.slice.merged) == (44, 80, 880), mid
        by = _by(vision_ops(m.vision, clip.slice, 1, "bf16"))
        hm = h * 4
        _ln(by["vis.pre_ln"], 44 * 80, h, depth)
        _ln(by["vis.0.mlp_ln"], 44 * 80, h, 0)
        _gelu(by["vis.mgelu"], 880, hm, depth, "bf16")
        assert by["vis.merger.1"].n == out_n and by["vis.merger.1"].m == 880, mid
        assert by["vis.0.qk"].m == 44 * 80 and by["vis.0.qk"].count == heads, mid
        assert "vis.0.gelu" not in by and "vis.tpool" not in by, mid


def test_qwen35_refuses_other_acts_and_a_missing_dtype():
    rel = load_release("Qwen/Qwen3.5-9B")
    silu = copy.deepcopy(rel["config"])
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(rel["repo"], silu), "qwen35 vision hidden_act is not gelu_pytorch_tanh")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("hidden_act")
    _raises(lambda: vision_spec(rel["repo"], missing), "qwen35 vision hidden_act is not gelu_pytorch_tanh")
    bare = vision_spec(rel["repo"], rel["config"])
    assert bare is not None and bare.ln_mode == "qwen35" and bare.vec_fmt == "" and bare.deepstack == 0
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "qwen35 vision activation dtype is not a vision float")
    from dataclasses import replace
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "qwen35 vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", post=()), g, 1),
            "qwen35 vision block is not fused qkv plus a two-linear merger")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", gated=True), g, 1),
            "qwen35 vision block is not fused qkv plus a two-linear merger")
    spec = from_release("synth", "Qwen/Qwen3.5-9B", rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16" and spec.vision.ln_mode == "qwen35"
    assert spec.param_check()["rel_err"] == 0.0
    flash = from_release("synth-flash", "Qwen/Qwen3.8-Flash-Next")
    assert flash.vision is not None and flash.vision.ln_mode == "qwen35"
    assert flash.vision.post[1][1] == 2560 and flash.vision.vec_fmt == "bf16"


def test_mixed_vision_role_stays_on_the_gemm_tail():
    m = get_model("qwen3.8-flash-next-fp4-local")
    assert m.vision is not None and m.vision.ln_mode == "" and m.vision.family == "qwen"
    assert m.param_check()["rel_err"] == 0.0
    rel = load_release(m.hf_id)
    assert set(rel["roles"]["vision"]) == {"bf16", "fp8", "nvfp4"}
    names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
    for needle in ("vis.0.attn_ln", "vis.pre_ln", "vis.mgelu", "vis.0.gelu"):
        assert needle not in names, needle
    assert "Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp" not in " ".join(m.notes)


def test_other_towers_do_not_gain_qwen35():
    billed = []
    skipped = []
    for e in entries():
        m = get_model(e["id"])
        vs = getattr(m, "vision", None)
        if vs is None:
            continue
        if vs.family != "qwen":
            assert vs.ln_mode != "qwen35", e["id"]
            continue
        rel = load_release(m.hf_id)
        vc = (rel["config"].get("vision_config") or {})
        mt = vc.get("model_type")
        if mt not in TYPES:
            assert vs.ln_mode != "qwen35", e["id"]
            continue
        keys = [k for k, v in (rel["roles"].get("vision") or {}).items()
                if isinstance(v, dict) and v.get("params")]
        if len(keys) != 1:
            assert vs.ln_mode == "", e["id"]
            skipped.append(e["id"])
            continue
        assert vs.ln_mode == "qwen35" and vs.vec_fmt == keys[0], e["id"]
        assert vs.deepstack == 0, e["id"]
        billed.append(e["id"])
    assert skipped == ["qwen3.8-flash-next-fp4-local"]
    for mid in ("qwen3.5-9b", "qwen3.8-27b", "qwen3.8-flash-next", "intern-decision-0.8b",
                "ornith-1.5-9b", "ornith-1.5-35b-a3b", "qwen3.5-397b-a17b"):
        assert mid in billed, mid
    assert "qwen3-vl-2b" not in billed and "mimo-v2.6-flash" not in billed
    vl = get_model("qwen3-vl-2b")
    assert vl.vision.ln_mode == "qwen3vl"
    ernie = get_model("ernie-4.5-vl-28b-a3b")
    enames = [op.name for op in vision_ops(ernie.vision, image_grid(ernie.vision, 1024, 1024), 1)]
    # 0.433 bills ERNIE's own LayerNorm as attn_ln. The Qwen3.5 merger names stay off this tower.
    assert "vis.mgelu" not in enames and "vis.pre_ln" not in enames
    assert "vis.0.attn_ln" in enames and "vis.out_ln" in enames and "vis.after_rms" in enames
    pre = evaluate(Scenario(model="qwen3.5-9b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1024 and pre.step > 0
