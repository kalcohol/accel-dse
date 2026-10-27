"""0.428: Muse Spark NaViT LayerNorm and ACT2FN["gelu"] (erf).

ln_pre runs on patch rows before the window gather. Each block has a
LayerNorm before attention and before the MLP, and an erf-GELU on the MLP
intermediate. ln_post runs on patch rows after the blocks. The adapter is
act(fc2(act(fc1(x)))), so both linears are followed by an erf-GELU at
projector_hidden_size. The fused qkv GEMM is not split. The shared GEMM
tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import (
    audio_ops, image_grid, tile_keys, video_grid, video_pre, vision_flops, vision_ops, vision_spec,
)

IDS = (
    "muse-spark-open",
    "muse-spark-open-fp8",
    "muse-spark-open-mxfp4",
    "muse-spark-open-nvfp4",
    "muse-spark-open-nvfp4-dspark",
    "muse-spark-open-nvfp4-dflash2",
)
HF = "meta-models/Muse-Spark-Open-816B-A42B"
H = 1536
INTER = 8960
DEPTH = 50
HEADS = 16
PROJ = 4096
HM = 4 * H
ROWS = 72 * 72
MERGED = 1296
WIN_KEYS = 4460544


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


def _gelu(op, rows: int, width: int, layer: int, af: str = "bf16") -> None:
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


def test_muse_navit_bills_layernorm_and_erf_gelu():
    models = [get_model(mid) for mid in IDS]
    base = models[0]
    vs = base.vision
    assert vs is not None
    for m in models[1:]:
        assert m.vision == vs, m.id
    assert vs.family == "qwen" and vs.ln_mode == "muse" and vs.vec_fmt == "bf16"
    assert vs.gated is False and vs.depth == DEPTH and vs.heads == HEADS
    assert vs.hidden == H and vs.inter == INTER and vs.qkv == H and vs.merge == 2
    assert vs.post == ((HM, PROJ), (PROJ, PROJ))
    assert vs.tile == 32 and len(vs.full_idx) == 13 and vs.full_idx[-1] == DEPTH - 1
    assert vs.perm == 2 and vs.params == 2045977600 and vs.extra == 151394816
    assert abs((vs.params - vs.extra) / 1e9 - 1.9) < 0.02
    pc = base.param_check()
    assert pc["ours"] == 819989536256 and pc["release"] == 819990552064
    assert pc["rel_err"] == (pc["ours"] - pc["release"]) / pc["release"]
    assert pc["rel_err"] < 0.0
    rel = load_release(base.hf_id)
    vc = rel["config"]["vision_config"]
    assert vc["hidden_act"] == "gelu" and vc["intermediate_size"] == INTER
    assert rel["config"]["projector_hidden_act"] == "gelu"
    assert rel["config"]["projector_hidden_size"] == PROJ
    assert rel["config"]["out_hidden_size"] == HM
    assert set(rel["roles"]["vision"]) == {"bf16"}
    assert rel["roles"]["vision"]["bf16"]["params"] == 2045977600
    note = " ".join(base.notes)
    assert "Muse Spark NaViT" in note and "ACT2FN" in note and "nn.LayerNorm" in note
    assert "vision_projection" in note and "perception RMSNorm" in note
    assert "act(fc2(act(fc1(x))))" in note
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.ph * g.pw, g.merged, g.llm_tokens) == (72, 72, ROWS, MERGED, MERGED + 2)
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _ln(by["vis.ln_pre"], ROWS, H, 0)
    _ln(by["vis.0.attn_ln"], ROWS, H, 0)
    _ln(by["vis.0.mlp_ln"], ROWS, H, 0)
    _ln(by[f"vis.{DEPTH - 1}.attn_ln"], ROWS, H, DEPTH - 1)
    _ln(by[f"vis.{DEPTH - 1}.mlp_ln"], ROWS, H, DEPTH - 1)
    _ln(by["vis.post_ln"], ROWS, H, DEPTH)
    _gelu(by["vis.0.gelu"], ROWS, INTER, 0)
    _gelu(by[f"vis.{DEPTH - 1}.gelu"], ROWS, INTER, DEPTH - 1)
    _gelu(by["vis.mgelu"], MERGED, PROJ, DEPTH)
    _gelu(by["vis.mgelu2"], MERGED, PROJ, DEPTH)
    assert by["vis.mgelu"].vec != MERGED * HM * 5
    assert by["vis.mgelu"].vec != MERGED * 8192 * 5
    assert by["vis.0.qkv"].n == 3 * H and by["vis.0.qkv"].m == ROWS and by["vis.0.qkv"].k == H
    assert by["vis.0.qkv"].vec == ROWS * (3 * H) * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.qkv"].a_fmt == "bf16"
    assert by["vis.0.mlp_in"].n == INTER and by["vis.0.mlp_in"].vec == ROWS * INTER * 3
    assert by["vis.0.mlp_out"].k == INTER and by["vis.0.mlp_out"].n == H
    assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].k == HM and by["vis.merger.0"].n == PROJ
    assert by["vis.merger.0"].vec == MERGED * PROJ * 3
    assert by["vis.merger.1"].k == PROJ and by["vis.merger.1"].n == PROJ
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS
    assert by["vis.0.qk"].causal == WIN_KEYS / (ROWS * ROWS) and by["vis.0.qk"].causal < 0.2
    assert by["vis.3.qk"].causal == 1.0 and by["vis.3.qk"].m == ROWS
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS * WIN_KEYS * 5
    assert by["vis.3.softmax"].sfu == 0.0 and by["vis.3.softmax"].vec == HEADS * ROWS * ROWS * 5
    assert tile_keys(72, 72, 32) == WIN_KEYS
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") < names.index("vis.ln_pre") < names.index("vis.win_perm")
    assert names.index("vis.win_perm") < names.index("vis.0.attn_ln") < names.index("vis.0.qkv")
    assert names.index("vis.0.qkv") < names.index("vis.0.o") < names.index("vis.0.mlp_ln")
    assert names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in") < names.index("vis.0.gelu")
    assert names.index("vis.0.gelu") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index("vis.0.qk") < names.index("vis.0.pv") < names.index("vis.0.softmax")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.post_ln")
    assert names.index("vis.post_ln") < names.index("vis.merger.0") < names.index("vis.mgelu")
    assert names.index("vis.mgelu") < names.index("vis.merger.1") < names.index("vis.mgelu2")
    lns = [op for op in ops if op.name == "vis.ln_pre" or op.name == "vis.post_ln"
           or op.name.endswith(".attn_ln") or op.name.endswith(".mlp_ln")]
    gels = [op for op in ops if "gelu" in op.name]
    assert len(lns) == 2 * DEPTH + 2 and len(gels) == DEPTH + 2
    for absent in ("vis.0.silu", "vis.0.relu", "vis.proj_ln", "vis.out_rms", "vis.proj_rms", "vis.tpool"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - 22449.178017792) < 1e-6

    wide = image_grid(vs, 1920, 1080)
    assert (wide.ph, wide.pw, wide.ph * wide.pw, wide.merged, wide.llm_tokens) == (78, 138, 10764, 2691, 2693)
    wb = _by(vision_ops(vs, wide, 1, "bf16"))
    _ln(wb["vis.ln_pre"], 10764, H, 0)
    _ln(wb["vis.post_ln"], 10764, H, DEPTH)
    _gelu(wb["vis.0.gelu"], 10764, INTER, 0)
    _gelu(wb["vis.mgelu"], 2691, PROJ, DEPTH)
    _gelu(wb["vis.mgelu2"], 2691, PROJ, DEPTH)
    assert wb["vis.merger.0"].m == 2691 and wb["vis.0.qk"].causal < 0.2 and wb["vis.3.qk"].causal == 1.0
    assert abs(vision_flops(vs, wide, 1) / 1e9 - 51445.593833472) < 1e-6

    two = _by(vision_ops(vs, g, 2, "bf16"))
    _ln(two["vis.ln_pre"], 2 * ROWS, H, 0)
    _ln(two["vis.0.attn_ln"], 2 * ROWS, H, 0)
    _ln(two["vis.0.mlp_ln"], 2 * ROWS, H, 0)
    _ln(two["vis.post_ln"], 2 * ROWS, H, DEPTH)
    _gelu(two["vis.0.gelu"], 2 * ROWS, INTER, 0)
    _gelu(two["vis.mgelu"], 2 * MERGED, PROJ, DEPTH)
    _gelu(two["vis.mgelu2"], 2 * MERGED, PROJ, DEPTH)
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * MERGED

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.mlp_in"].a_fmt == "fp8"
    assert forced["vis.merger.0"].a_fmt == "fp8" and forced["vis.merger.1"].a_fmt == "fp8"
    assert forced["vis.merger.0"].w_fmt == "bf16" and forced["vis.0.qkv"].w_bits == 16.0
    assert forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
    assert forced["vis.0.gelu"].a_fmt == "bf16" and forced["vis.mgelu"].a_fmt == "bf16"
    assert forced["vis.mgelu2"].a_fmt == "bf16" and forced["vis.0.gelu"].vec == by["vis.0.gelu"].vec
    assert forced["vis.ln_pre"].a_fmt == "fp32" and forced["vis.post_ln"].a_fmt == "fp32"
    assert forced["vis.0.attn_ln"].a_fmt == "fp32"

    pre = evaluate(Scenario(model="muse-spark-open", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == MERGED + 2

    aud = audio_ops(vs.audio, 60, 1)
    assert vs.audio.params == 117840384 and vs.audio.tokens(60) == 750
    assert vs.audio.vec_fmt == "bf16"
    assert any(op.name == "aud.0.0.gelu" for op in aud)
    assert any(op.name == "aud.dual_rms" for op in aud)


def test_muse_video_slice_has_no_temporal_pool():
    m = get_model("muse-spark-open")
    vg = video_grid(m.vision, video_pre(m), 4, 1280, 720)
    g = vg.slice
    assert vg.frames == 4 and vg.t == 2 and vg.llm_tokens == 507
    assert (g.ph, g.pw, g.ph * g.pw, g.merged, g.tframes) == (24, 42, 1008, 252, 1)
    ops = vision_ops(m.vision, g, 1, "bf16")
    by = _by(ops)
    _ln(by["vis.post_ln"], 1008, H, DEPTH)
    _gelu(by["vis.0.gelu"], 1008, INTER, 0)
    _gelu(by["vis.mgelu"], 252, PROJ, DEPTH)
    _gelu(by["vis.mgelu2"], 252, PROJ, DEPTH)
    assert by["vis.merger.0"].m == 252 and "vis.tpool" not in by
    assert abs(vision_flops(m.vision, g, 1) / 1e9 - 3978.913775616) < 1e-6
    assert video_pre(get_model("muse-spark-open-nvfp4-dspark")) is None


def test_muse_refuses_other_activations_and_a_missing_dtype():
    rel = load_release(HF)
    tanh = copy.deepcopy(rel["config"])
    tanh["vision_config"]["hidden_act"] = "gelu_pytorch_tanh"
    _raises(lambda: vision_spec(HF, tanh), "muse vision hidden_act is not gelu")
    silu = copy.deepcopy(rel["config"])
    silu["projector_hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF, silu), "muse projector_hidden_act is not gelu")
    missing = copy.deepcopy(rel["config"])
    missing.pop("projector_hidden_act")
    _raises(lambda: vision_spec(HF, missing), "muse projector_hidden_act is not gelu")
    wide = copy.deepcopy(rel["config"])
    wide["out_hidden_size"] = H
    _raises(lambda: vision_spec(HF, wide), "muse out_hidden_size is not hidden * merge * merge")

    bare = vision_spec(HF, rel["config"])
    assert bare is not None and bare.ln_mode == "muse" and bare.vec_fmt == ""
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "muse vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "muse vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", gated=True), g, 1),
            "muse vision block is not fused qkv plus a two-linear adapter")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", post=()), g, 1),
            "muse vision block is not fused qkv plus a two-linear adapter")

    spec = from_release("synth", HF, rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16" and spec.vision.ln_mode == "muse"
    empty = copy.deepcopy(rel)
    empty["roles"]["vision"] = {}
    _raises(lambda: from_release("synth-empty", HF, rel=empty, cfg=empty["config"]),
            "muse tower vision role is not one dtype")
    mixed = copy.deepcopy(rel)
    mixed["roles"]["vision"]["fp32"] = {"params": 8, "bytes": 32}
    cleared = from_release("synth-mix", HF, rel=mixed, cfg=mixed["config"])
    assert cleared.vision is not None and cleared.vision.ln_mode == "" and cleared.vision.vec_fmt == ""
    names = [op.name for op in vision_ops(cleared.vision, g, 1)]
    assert "vis.ln_pre" not in names and "vis.0.gelu" not in names and "vis.mgelu2" not in names


def test_other_towers_do_not_gain_muse_navit_gelu():
    for mid in ("qwen3.5-9b", "kimi-k2.5", "deepseek-v4.1-flash"):
        m = get_model(mid)
        assert m.vision is not None and m.vision.ln_mode != "muse", mid
        names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
        assert "vis.mgelu2" not in names, mid
        assert "vis.ln_pre" not in names, mid
        assert "Muse Spark NaViT" not in " ".join(m.notes)
