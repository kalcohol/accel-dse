"""0.425: C-RADIO LayerNorm, erf-GELU, and projector RMSNorm.

nn.LayerNorm reduces in fp32, one rsqrt per billed row. ACT2FN["gelu"] is
nn.functional.gelu, the erf form, on the MLP intermediate. The projector
RMSNorm is on the shuffled rows at 4 * hidden. Super-VL also runs
vision_final_layernorm. Bolt does not. relu2 and the identity LayerScale
are not ops. The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.model import _radio_vision_fmt, load_release
from accel_dse.core.scenario import Scenario
from accel_dse.core.vision import image_grid, video_grid_of, vision_flops, vision_ops, vision_spec

ROOT = Path(__file__).resolve().parents[1]
H = 1280
DEPTH = 32
HEADS = 16
INTER = 5120
SHUF = 5120
PROJ = 20480
GFLOP = 8297.339944960
CASES = (
    ("nemotron-3.5-super-vl-120b-a12b-bf16", True, 4096),
    ("nemotron-3.5-super-vl-120b-a12b-nvfp4", True, 4096),
    ("nemotron-3.5-bolt-74b-a9b-bf16", False, 4096),
    ("nemotron-3.5-bolt-74b-a9b-24gb", False, 4096),
)


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


def _gelu(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0


def _rms(op, rows: int, width: int, layer: int) -> None:
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


def test_radio_bills_layernorm_erf_gelu_and_projector_rmsnorm():
    for mid, final, text_n in CASES:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "radio" and vs.post_ln is final, mid
        assert vs.vec_fmt == "bf16" and vs.gated is False and vs.depth == DEPTH, mid
        assert vs.hidden == H and vs.inter == INTER and vs.heads == HEADS, mid
        assert vs.post[0] == (SHUF, PROJ) and vs.post[1][0] == PROJ and vs.post[1][1] == text_n, mid
        rel = load_release(m.hf_id)
        vc = rel["config"]["vision_config"]
        assert vc["hidden_act"] == "gelu" and vc["use_swiglu_ffn"] is False, mid
        assert vc["layerscale_value"] == 1.0 and vc["drop_path_rate"] == 0.0, mid
        assert set(rel["roles"]["vision"]) == {"bf16", "fp32"}, mid
        assert rel["roles"]["vision"]["fp32"]["params"] == 6, mid
        assert "C-RADIO" in " ".join(m.notes), mid
        g = image_grid(vs, 1024, 1024)
        assert (g.ph, g.pw, g.merged) == (64, 64, 1024), mid
        ops = vision_ops(vs, g, 1, "bf16")
        by = _by(ops)
        rows = 64 * 64
        _ln(by["vis.0.attn_ln"], rows, H, 0)
        _ln(by["vis.0.mlp_ln"], rows, H, 0)
        _ln(by[f"vis.{DEPTH - 1}.attn_ln"], rows, H, DEPTH - 1)
        _gelu(by["vis.0.gelu"], rows, INTER, 0)
        _gelu(by[f"vis.{DEPTH - 1}.gelu"], rows, INTER, DEPTH - 1)
        _rms(by["vis.proj_rms"], 1024, SHUF, DEPTH)
        if final:
            _ln(by["vis.final_ln"], rows, H, DEPTH)
        else:
            assert "vis.final_ln" not in by, mid
        assert by["vis.0.qkv"].n == 3 * H and by["vis.0.qkv"].vec == rows * (3 * H) * 3, mid
        assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0, mid
        assert by["vis.0.qkv"].a_fmt == "bf16", mid
        assert by["vis.0.mlp_in"].n == INTER and by["vis.0.mlp_in"].vec == rows * INTER * 3, mid
        assert by["vis.0.mlp_out"].k == INTER and by["vis.0.mlp_out"].n == H, mid
        assert by["vis.merger.0"].m == 1024 and by["vis.merger.0"].k == SHUF, mid
        assert by["vis.merger.0"].n == PROJ and by["vis.merger.0"].vec == 1024 * PROJ * 3, mid
        assert by["vis.merger.1"].k == PROJ and by["vis.merger.1"].n == text_n, mid
        assert by["vis.0.qk"].m == rows and by["vis.0.qk"].count == HEADS, mid
        assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * rows * rows * 5, mid
        names = [op.name for op in ops]
        assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
        assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
        assert names.index("vis.0.mlp_in") < names.index("vis.0.gelu") < names.index("vis.0.mlp_out")
        assert names.index("vis.0.mlp_out") < names.index("vis.0.qk")
        if final:
            assert names.index("vis.final_ln") < names.index("vis.proj_rms")
        assert names.index("vis.proj_rms") < names.index("vis.merger.0") < names.index("vis.merger.1")
        lns = [op for op in ops if op.name.endswith("attn_ln") or op.name.endswith("mlp_ln")
               or op.name == "vis.final_ln"]
        gels = [op for op in ops if op.name.endswith(".gelu")]
        assert len(lns) == 2 * DEPTH + (1 if final else 0) and len(gels) == DEPTH, mid
        for absent in ("vis.0.silu", "vis.0.relu", "vis.0.relu2", "vis.proj_relu", "vis.proj_gelu",
                       "vis.0.ls1", "vis.0.norm1", "vis.0.q_rms", "vis.mgelu"):
            assert absent not in by, (mid, absent)
        assert abs(vision_flops(vs, g, 1) / 1e9 - GFLOP) < 1e-6, mid

        wide = image_grid(vs, 1024, 768)
        assert (wide.ph, wide.pw, wide.merged) == (48, 64, 768), mid
        wb = _by(vision_ops(vs, wide, 1, "bf16"))
        _ln(wb["vis.0.attn_ln"], 48 * 64, H, 0)
        _gelu(wb["vis.0.gelu"], 48 * 64, INTER, 0)
        _rms(wb["vis.proj_rms"], 768, SHUF, DEPTH)
        if final:
            _ln(wb["vis.final_ln"], 48 * 64, H, DEPTH)
        assert abs(vision_flops(vs, wide, 1) / 1e9 - 5707.608883200) < 1e-6, mid

        two = _by(vision_ops(vs, g, 2, "bf16"))
        _ln(two["vis.0.attn_ln"], 2 * rows, H, 0)
        _gelu(two["vis.0.gelu"], 2 * rows, INTER, 0)
        _rms(two["vis.proj_rms"], 2048, SHUF, DEPTH)
        assert two["vis.0.qk"].m == rows and two["vis.0.qk"].count == 2 * HEADS, mid
        assert two["vis.merger.0"].m == 2048, mid

        forced = _by(vision_ops(vs, g, 1, "fp8"))
        assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.mlp_in"].a_fmt == "fp8", mid
        assert forced["vis.merger.0"].a_fmt == "fp8" and forced["vis.merger.0"].w_fmt == "bf16", mid
        assert forced["vis.0.qkv"].w_bits == 16.0 and forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec, mid
        assert forced["vis.0.gelu"].a_fmt == "bf16" and forced["vis.proj_rms"].a_fmt == "fp32", mid
        assert forced["vis.0.attn_ln"].a_fmt == "fp32", mid


def test_radio_video_slice_uses_the_same_counts():
    m = get_model("nemotron-3.5-super-vl-120b-a12b-bf16")
    sv = Scenario(model=m.id).replace("serving.videos", 1).serving
    vg = video_grid_of(m, sv)
    assert (vg.slice.ph, vg.slice.pw, vg.slice.merged, vg.t) == (24, 42, 252, 16)
    rows = 24 * 42
    by = _by(vision_ops(m.vision, vg.slice, 1, "bf16"))
    _ln(by["vis.final_ln"], rows, H, DEPTH)
    _gelu(by["vis.0.gelu"], rows, INTER, 0)
    _rms(by["vis.proj_rms"], 252, SHUF, DEPTH)
    assert by["vis.merger.1"].n == 4096 and by["vis.0.qk"].count == HEADS
    bolt = get_model("nemotron-3.5-bolt-74b-a9b-bf16")
    bb = _by(vision_ops(bolt.vision, vg.slice, 1, "bf16"))
    assert "vis.final_ln" not in bb
    _rms(bb["vis.proj_rms"], 252, SHUF, DEPTH)
    _gelu(bb["vis.0.gelu"], rows, INTER, 0)


def test_radio_refuses_a_different_activation_or_role():
    rel = json.loads((ROOT / "accel_dse/data/releases/nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16.json").read_text())
    cfg = json.loads(json.dumps(rel["config"]))
    cfg["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(rel["repo"], cfg), "radio hidden_act is not gelu")
    cfg = json.loads(json.dumps(rel["config"]))
    cfg["vision_config"]["use_swiglu_ffn"] = True
    _raises(lambda: vision_spec(rel["repo"], cfg), "radio use_swiglu_ffn is not false")
    cfg = json.loads(json.dumps(rel["config"]))
    cfg["vision_config"]["layerscale_value"] = 0.1
    _raises(lambda: vision_spec(rel["repo"], cfg), "radio layerscale_value is not 1")
    bad = json.loads(json.dumps(rel))
    bad["roles"]["vision"]["fp32"]["params"] = 7
    _raises(lambda: _radio_vision_fmt(bad), "radio conditioner is not 6 fp32 values")
    bad = json.loads(json.dumps(rel))
    bad["roles"]["vision"]["fp8"] = {"bytes": 8, "params": 8}
    _raises(lambda: _radio_vision_fmt(bad), "radio tower vision role is not bf16 plus the conditioner")
    vs = get_model("nemotron-3.5-super-vl-120b-a12b-bf16").vision
    bare = vs.__class__(**{**vs.__dict__, "vec_fmt": ""})
    _raises(lambda: vision_ops(bare, image_grid(vs, 1024, 1024), 1), "radio vision activation dtype")
