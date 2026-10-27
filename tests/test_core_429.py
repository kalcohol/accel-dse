"""0.429: Gemma-4 vision RMSNorm and unified embedder LayerNorm.

The ViT bills four Gemma4RMSNorm(hidden) calls per block and q/k RMSNorm on
head_dim. v_norm has with_scale=False and is not an op. The MLP stays
gelu_pytorch_tanh. The 12B embedder bills three nn.LayerNorm calls. Its
pre-projection RMSNorm has no scale. The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_flops, vision_ops, vision_spec

VIT = (
    "gemma-4-31b-it",
    "gemma-4-31b",
    "gemma-4-26b-a4b-it",
    "gemma-4-26b-a4b",
    "gemma-4-e4b-it",
    "gemma-4-e4b",
    "gemma-4-e2b-it",
    "gemma-4-e2b",
    "nemotron-3.5-bolt-e4b-bf16",
)
UNI = ("gemma-4-12b-it", "gemma-4-12b")
PAD = 280 * 9
H31, HEADS31, HD31, DEPTH31, INTER31 = 1152, 16, 72, 27, 4304
H16, HEADS16, HD16, DEPTH16, INTER16 = 768, 12, 64, 16, 3072


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


def test_gemma4_vit_bills_rmsnorm():
    models = [get_model(mid) for mid in VIT]
    by_id = {m.id: m for m in models}
    assert by_id["gemma-4-31b"].vision == by_id["gemma-4-31b-it"].vision
    assert by_id["gemma-4-26b-a4b"].vision == by_id["gemma-4-26b-a4b-it"].vision
    assert by_id["gemma-4-e2b"].vision == by_id["gemma-4-e2b-it"].vision
    assert by_id["gemma-4-e4b-it"].vision == by_id["nemotron-3.5-bolt-e4b-bf16"].vision
    assert by_id["gemma-4-31b-it"].vision != by_id["gemma-4-26b-a4b-it"].vision
    for m in models:
        vs = m.vision
        assert vs.family == "gemma4" and vs.ln_mode == "" and vs.gated is True
        assert vs.rz["act"] == "gelu_pytorch_tanh"
        assert m.param_check()["rel_err"] == 0.0
        rel = load_release(m.hf_id)
        assert rel["config"]["vision_config"]["hidden_activation"] == "gelu_pytorch_tanh"
        note = " ".join(m.notes)
        assert "Gemma4RMSNorm" in note and "head_dim" in note and "with_scale" in note
        assert "gelu_pytorch_tanh" in note and "GeGLU" in note
    wide = by_id["gemma-4-31b-it"]
    small = by_id["gemma-4-e2b-it"]
    vs = wide.vision
    assert vs.hidden == H31 and vs.depth == DEPTH31 and vs.heads == HEADS31 and vs.inter == INTER31
    assert vs.qkv == H31 and vs.head_dim == HD31 and vs.rz["kvd"] == H31 and vs.rz["std"] == 1
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.llm_tokens) == (48, 48, 256)
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _norm(by["vis.0.in_rms"], PAD, H31, 0)
    _norm(by["vis.0.post_attn_rms"], PAD, H31, 0)
    _norm(by["vis.0.pre_ffn_rms"], PAD, H31, 0)
    _norm(by["vis.0.post_ffn_rms"], PAD, H31, 0)
    _norm(by[f"vis.{DEPTH31 - 1}.in_rms"], PAD, H31, DEPTH31 - 1)
    _norm(by["vis.0.q_rms"], PAD * HEADS31, HD31, 0)
    _norm(by["vis.0.k_rms"], PAD * HEADS31, HD31, 0)
    assert by["vis.0.q_rms"].sfu == float(PAD * HEADS31)
    assert by["vis.0.in_rms"].sfu == float(PAD)
    assert by["vis.0.q_rms"].vec == by["vis.0.in_rms"].vec
    assert by["vis.0.q"].m == PAD and by["vis.0.q"].k == H31 and by["vis.0.q"].n == H31
    assert by["vis.0.q"].vec == PAD * H31 * 3
    assert by["vis.0.q"].w_fmt == "bf16" and by["vis.0.q"].w_bits == 16.0 and by["vis.0.q"].a_fmt == "bf16"
    assert by["vis.0.k"].n == H31 and by["vis.0.v"].n == H31 and by["vis.0.o"].k == H31
    assert by["vis.0.mlp_in"].n == 2 * INTER31 and by["vis.0.mlp_in"].vec == PAD * (2 * INTER31) * 3
    assert by["vis.0.mlp_out"].k == INTER31 and by["vis.0.mlp_out"].n == H31
    assert by["vis.0.qk"].m == PAD and by["vis.0.qk"].count == HEADS31
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS31 * PAD * PAD * 5
    assert by["vis.proj"].m == 256 and (by["vis.proj"].k, by["vis.proj"].n) == (H31, 5376)
    assert by["vis.pool"].vec == PAD * H31 and by["vis.pool"].layer == DEPTH31
    assert by["vis.std"].vec == 256 * H31 * 2 and by["vis.std"].layer == DEPTH31
    names = [op.name for op in ops]
    assert names.index("vis.patch_embed") < names.index("vis.0.in_rms") < names.index("vis.0.q")
    assert names.index("vis.0.q") < names.index("vis.0.k") < names.index("vis.0.v")
    assert names.index("vis.0.v") < names.index("vis.0.q_rms") < names.index("vis.0.k_rms")
    assert names.index("vis.0.k_rms") < names.index("vis.0.o") < names.index("vis.0.post_attn_rms")
    assert names.index("vis.0.post_attn_rms") < names.index("vis.0.pre_ffn_rms") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out") < names.index("vis.0.post_ffn_rms")
    assert names.index("vis.0.post_ffn_rms") < names.index("vis.0.qk") < names.index("vis.0.pv")
    assert names.index("vis.0.pv") < names.index("vis.0.softmax")
    rms = [op for op in ops if op.name.endswith("_rms")]
    assert len(rms) == 6 * DEPTH31
    for absent in ("vis.0.v_rms", "vis.0.silu", "vis.0.gelu", "vis.0.relu", "vis.0.mlp_mul",
                   "vis.ln1", "vis.0.clamp"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - 3544.229412864) < 1e-6
    assert vs.params == 575743536
    ev = small.vision
    assert ev.hidden == H16 and ev.depth == DEPTH16 and ev.heads == HEADS16 and ev.head_dim == HD16
    assert ev.rz["std"] == 0 and ev.inter == INTER16
    eops = _by(vision_ops(ev, image_grid(ev, 1024, 1024), 1))
    _norm(eops["vis.0.in_rms"], PAD, H16, 0)
    _norm(eops["vis.0.q_rms"], PAD * HEADS16, HD16, 0)
    _norm(eops["vis.0.k_rms"], PAD * HEADS16, HD16, 0)
    assert eops["vis.0.q_rms"].sfu == float(PAD * HEADS16)
    assert eops["vis.0.in_rms"].sfu == float(PAD)
    assert "vis.std" not in eops and "vis.0.clamp" not in eops
    assert eops["vis.proj"].n == 1536
    assert len([op for op in eops.values() if op.name.endswith("_rms")]) == 6 * DEPTH16
    assert abs(vision_flops(ev, image_grid(ev, 1024, 1024), 1) / 1e9 - 1076.726071296) < 1e-6
    assert ev.params == 475729088


def test_gemma4_vit_lengths_video_and_dtype():
    wide = get_model("gemma-4-31b-it")
    vs = wide.vision
    mid = image_grid(vs, 640, 480)
    assert mid.llm_tokens == 266
    by = _by(vision_ops(vs, mid, 1))
    assert by["vis.0.in_rms"].sfu == float(PAD)
    assert by["vis.proj"].m == 266 and by["vis.std"].vec == 266 * H31 * 2
    tall = image_grid(vs, 1920, 1080)
    assert tall.llm_tokens == 264
    tby = _by(vision_ops(vs, tall, 1))
    assert tby["vis.0.q_rms"].sfu == float(PAD * HEADS31) and tby["vis.proj"].m == 264
    two = _by(vision_ops(vs, image_grid(vs, 1024, 1024), 2, "bf16"))
    assert two["vis.0.in_rms"].sfu == float(2 * PAD)
    assert two["vis.0.q_rms"].sfu == float(2 * PAD * HEADS31)
    assert two["vis.0.k_rms"].vec == 2 * PAD * H31 * 4
    assert two["vis.0.qk"].m == PAD and two["vis.0.qk"].count == 2 * HEADS31
    assert two["vis.proj"].m == 512 and two["vis.0.q"].m == 2 * PAD
    fp8 = _by(vision_ops(vs, image_grid(vs, 1024, 1024), 1, "fp8"))
    assert fp8["vis.0.q"].a_fmt == "fp8" and fp8["vis.0.q"].vec == PAD * H31 * 3
    assert fp8["vis.0.q"].w_fmt == "bf16" and fp8["vis.0.q"].w_bits == 16.0
    assert fp8["vis.0.mlp_in"].a_fmt == "fp8" and fp8["vis.0.in_rms"].a_fmt == "fp32"
    assert fp8["vis.0.q_rms"].a_fmt == "fp32"
    pre = video_pre(wide)
    vid = video_grid(vs, pre, 32, 1280, 720)
    assert vid.frames == 32 and vid.slice.llm_tokens == 66 and vid.slice.pad_soft == 70
    vops = _by(vision_ops(vs, vid.slice, vid.frames))
    rows = 32 * 70 * 9
    _norm(vops["vis.0.in_rms"], rows, H31, 0)
    _norm(vops["vis.0.q_rms"], rows * HEADS31, HD31, 0)
    assert vops["vis.0.q"].m == rows and vops["vis.proj"].m == 32 * 66
    assert vops["vis.0.qk"].m == 70 * 9 and vops["vis.0.qk"].count == 32 * HEADS31
    assert vops["vis.std"].vec == 32 * 66 * H31 * 2
    assert vops["vis.0.in_rms"].sfu != float(32 * 66)
    edge = video_grid(vs, pre, 32, 4000, 48)
    assert edge.slice.llm_tokens == 70
    eops = _by(vision_ops(vs, edge.slice, 1))
    assert eops["vis.0.in_rms"].sfu == float(70 * 9) and eops["vis.proj"].m == 70
    moe = get_model("gemma-4-26b-a4b-it").vision
    mops = _by(vision_ops(moe, image_grid(moe, 1024, 1024), 1))
    _norm(mops["vis.0.in_rms"], PAD, H31, 0)
    _norm(mops["vis.0.k_rms"], PAD * HEADS31, HD31, 0)
    assert mops["vis.proj"].n == 2816
    assert abs(vision_flops(moe, image_grid(moe, 1024, 1024), 1) / 1e9 - 3542.719463424) < 1e-6
    assert moe.params == 572794416
    e4 = get_model("gemma-4-e4b-it").vision
    e4ops = _by(vision_ops(e4, image_grid(e4, 1024, 1024), 1))
    _norm(e4ops[f"vis.{DEPTH16 - 1}.post_ffn_rms"], PAD, H16, DEPTH16 - 1)
    assert e4ops["vis.proj"].n == 2560 and "vis.std" not in e4ops
    assert abs(vision_flops(e4, image_grid(e4, 1024, 1024), 1) / 1e9 - 1077.128724480) < 1e-6
    assert e4.params == 478088384
    r = evaluate(Scenario(model="gemma-4-31b-it", serving=Serving(phase="prefill", prompt=32, images=1)))
    assert r.vision["image_tokens"] == 256


def test_gemma4_unified_bills_layernorm():
    models = [get_model(mid) for mid in UNI]
    assert models[0].vision == models[1].vision
    vs = models[0].vision
    assert vs.family == "gemma4u" and vs.depth == 0 and vs.rz["mm"] == 3840
    assert vs.rz["patch_dim"] == 6912 and vs.params == 52379904
    for m in models:
        assert m.param_check()["rel_err"] == 0.0
        note = " ".join(m.notes)
        assert "nn.LayerNorm" in note and "patch_dim" in note and "with_scale=False" in note
        assert "Gemma4UnifiedRMSNorm" in note
    g = image_grid(vs, 1024, 1024)
    assert g.llm_tokens == 256
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    assert [op.name for op in ops] == ["vis.ln1", "vis.patch_dense", "vis.ln2", "vis.pos_ln", "vis.proj"]
    _norm(by["vis.ln1"], 280, 6912, 0)
    _norm(by["vis.ln2"], 280, 3840, 0)
    _norm(by["vis.pos_ln"], 280, 3840, 0)
    assert by["vis.ln1"].vec != 280 * 3840 * 4
    assert by["vis.ln1"].sfu == 280.0 and by["vis.ln1"].sfu != 256.0
    assert by["vis.patch_dense"].m == 280 and (by["vis.patch_dense"].k, by["vis.patch_dense"].n) == (6912, 3840)
    assert by["vis.patch_dense"].vec == 280 * 3840 * 3
    assert by["vis.patch_dense"].w_fmt == "bf16" and by["vis.patch_dense"].w_bits == 16.0
    assert by["vis.proj"].m == 280 and (by["vis.proj"].k, by["vis.proj"].n) == (3840, 3840)
    for absent in ("vis.pre_rms", "vis.0.in_rms", "vis.0.gelu", "vis.0.silu", "vis.std"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - 23.121100800) < 1e-6
    wide = image_grid(vs, 1920, 1080)
    assert wide.llm_tokens == 264
    wops = _by(vision_ops(vs, wide, 1))
    assert wops["vis.ln1"].sfu == 280.0 and wops["vis.proj"].m == 280
    two = _by(vision_ops(vs, g, 2, "fp8"))
    assert two["vis.ln2"].sfu == 560.0 and two["vis.ln2"].a_fmt == "fp32"
    assert two["vis.patch_dense"].m == 560 and two["vis.patch_dense"].a_fmt == "fp8"
    assert two["vis.patch_dense"].vec == 560 * 3840 * 3
    assert two["vis.patch_dense"].w_fmt == "bf16" and two["vis.proj"].a_fmt == "fp8"
    vid = video_grid(vs, video_pre(models[0]), 32, 1280, 720)
    assert vid.slice.pad_soft == 70 and vid.slice.llm_tokens == 66
    vops = _by(vision_ops(vs, vid.slice, vid.frames))
    _norm(vops["vis.pos_ln"], 32 * 70, 3840, 0)
    assert vops["vis.patch_dense"].m == vops["vis.proj"].m == 32 * 70
    assert vops["vis.ln1"].sfu != float(32 * 66)
    edge = video_grid(vs, video_pre(models[0]), 32, 4000, 48)
    assert _by(vision_ops(vs, edge.slice, 1))["vis.ln1"].sfu == 70.0
    r = evaluate(Scenario(model="gemma-4-12b-it", serving=Serving(phase="prefill", prompt=32, images=1)))
    assert r.vision["image_tokens"] == 256


def test_gemma4_refuses_other_activation_and_bad_width():
    m = get_model("gemma-4-31b-it")
    rel = load_release(m.hf_id)
    silu = copy.deepcopy(rel["config"])
    silu["vision_config"]["hidden_activation"] = "silu"
    _raises(lambda: vision_spec(m.hf_id, silu), "gelu_pytorch_tanh")
    missing = copy.deepcopy(rel["config"])
    del missing["vision_config"]["hidden_activation"]
    _raises(lambda: vision_spec(m.hf_id, missing), "gelu_pytorch_tanh")
    g = image_grid(m.vision, 1024, 1024)
    swapped = dict(m.vision.rz)
    swapped["act"] = "gelu"
    bad = replace(m.vision, pre=tuple(sorted(swapped.items())))
    _raises(lambda: vision_ops(bad, g, 1), "gelu_pytorch_tanh")
    odd = dict(m.vision.rz)
    odd["kvd"] = HD31 + 1
    broken = replace(m.vision, pre=tuple(sorted(odd.items())))
    _raises(lambda: vision_ops(broken, g, 1), "whole number of heads")
    one = dict(m.vision.rz)
    one["kvd"] = HD31
    narrow = replace(m.vision, pre=tuple(sorted(one.items())))
    ops = _by(vision_ops(narrow, g, 1))
    _norm(ops["vis.0.k_rms"], PAD, HD31, 0)
    _norm(ops["vis.0.q_rms"], PAD * HEADS31, HD31, 0)
    assert ops["vis.0.k"].n == HD31 and ops["vis.0.q"].n == H31
    assert ops["vis.0.k_rms"].vec == PAD * HD31 * 4
    assert ops["vis.0.k_rms"].vec != ops["vis.0.q_rms"].vec
