"""0.406: LongCat vision and FSMN bill the norms the public classes already run.

LongCatVisionEncoderLayer runs norm1, q_norm, k_norm, norm2, the SwiGLU inner
norm, and two LayerScales. q_norm and k_norm flatten heads, so each is one
RMSNorm over embed_dim. VisionProjector starts with LayerNorm on the
width-paired rows. Each LongCatAudioEncoderLayer runs two LayerNorm(proj)
modules. SiLU, projector GELU, RoPE, ReLU6, and the mask multiply are billed in 0.407.
This file still locks the norm and LayerScale shapes.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import AudioSpec, audio_ops, image_grid, vision_ops

HIDDEN = 1280
INNER = 3456
DEPTH = 32
HEADS = 16
PROJ = 1536
AUDIO_H = 6144
AUDIO_LAYERS = 22


def _block(li: int) -> tuple[str, ...]:
    return (
        f"vis.{li}.norm1", f"vis.{li}.qkv", f"vis.{li}.rope", f"vis.{li}.q_norm",
        f"vis.{li}.k_norm", f"vis.{li}.o", f"vis.{li}.ls1", f"vis.{li}.norm2",
        f"vis.{li}.mlp_in", f"vis.{li}.silu", f"vis.{li}.mlp_mul", f"vis.{li}.mlp_norm",
        f"vis.{li}.mlp_out", f"vis.{li}.ls2",
        f"vis.{li}.qk", f"vis.{li}.pv", f"vis.{li}.softmax",
    )


def test_longcat_vision_bills_block_norms_and_layerscale_on_patch_tokens():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "longcatvit" and vs.gated
    assert (vs.hidden, vs.inter, vs.depth, vs.heads) == (HIDDEN, INNER, DEPTH, HEADS)
    assert vs.head_dim == HIDDEN // HEADS and vs.head_dim != HIDDEN
    g = image_grid(vs, 1024, 1024)
    tokens = g.ph * g.pw
    assert tokens == 72 * 72 and g.merged == tokens // 2
    ops = vision_ops(vs, g, 1, "fp8")
    assert ops and all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    block = _block(0)
    idx = [names.index(name) for name in block]
    assert idx == list(range(idx[0], idx[0] + len(block)))
    assert names.index("vis.proj_ln") + 1 == names.index("vis.merger.0")
    qkv = next(op for op in ops if op.name == "vis.0.qkv")
    assert qkv.vec == tokens * (3 * HIDDEN) * 3
    for li in range(DEPTH):
        for name, width in ((f"vis.{li}.norm1", HIDDEN), (f"vis.{li}.q_norm", HIDDEN),
                            (f"vis.{li}.k_norm", HIDDEN), (f"vis.{li}.norm2", HIDDEN),
                            (f"vis.{li}.mlp_norm", INNER)):
            op = next(item for item in ops if item.name == name)
            assert op.kind == "vector" and op.layer == li
            assert op.vec == tokens * width * 4 and op.sfu == tokens
            assert op.sfu != tokens * HEADS
            assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
        for name in (f"vis.{li}.ls1", f"vis.{li}.ls2"):
            op = next(item for item in ops if item.name == name)
            assert op.kind == "vector" and op.layer == li
            assert op.vec == tokens * HIDDEN * 2 and op.sfu == 0
            assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0
    ln = next(op for op in ops if op.name == "vis.proj_ln")
    assert ln.kind == "vector" and ln.layer == DEPTH
    assert ln.vec == g.merged * (2 * HIDDEN) * 4 and ln.sfu == g.merged
    assert ln.sfu != tokens and ln.vec != tokens * (2 * HIDDEN) * 4
    assert ln.a_fmt == "fp32" and ln.w_params == 0 and ln.flops == 0
    sm = next(op for op in ops if op.name == "vis.0.softmax")
    assert sm.vec == HEADS * tokens * tokens * 5 and sm.sfu == 0
    norm_sfu = sum(op.sfu for op in ops if op.name == "vis.proj_ln" or op.name.endswith(
        (".norm1", ".q_norm", ".k_norm", ".norm2", ".mlp_norm")))
    assert norm_sfu == DEPTH * tokens * 5 + g.merged
    wide = vision_ops(vs, g, 2)
    assert next(op for op in wide if op.name == "vis.0.q_norm").sfu == 2 * tokens
    assert next(op for op in wide if op.name == "vis.0.ls1").vec == 2 * tokens * HIDDEN * 2
    assert next(op for op in wide if op.name == "vis.proj_ln").sfu == 2 * g.merged
    assert vision_ops(vs, g, 0) == []
    text = " ".join(m.notes)
    fp_text = " ".join(fp.notes)
    for needle in ("内层 RMSNorm", "embed_dim", "LayerScale", "memory_ln", "不升 fp32"):
        assert needle in text and needle in fp_text
    other = get_model("step-5-preview").vision
    assert other is not None and other.family != "longcatvit"
    og = image_grid(other, 1024, 1024)
    assert og is not None
    onames = [op.name for op in vision_ops(other, og, 1)]
    assert not any(name.endswith((".q_norm", ".k_norm", ".norm1", ".norm2", ".ls1", ".ls2", ".proj_ln"))
                   for name in onames)


def test_longcat_audio_bills_both_fsmn_layernorms_on_proj_width():
    vs = get_model("longcat-flash-omni").vision
    assert vs is not None and vs.audio is not None and vs.audio.kind == "lcafsmn"
    au = vs.audio
    frames = au.tokens(1.0)
    proj, hidden = au.dims[0], au.dims[1]
    nlayer = au.depths[0]
    assert (frames, proj, hidden, nlayer) == (13, PROJ, AUDIO_H, AUDIO_LAYERS)
    ops = audio_ops(au, 1.0, 1, "fp8")
    assert all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    for i in range(nlayer):
        mem = next(op for op in ops if op.name == f"aud.{i}.mem_ln")
        ffn = next(op for op in ops if op.name == f"aud.{i}.ffn_ln")
        assert mem.kind == ffn.kind == "vector" and mem.layer == ffn.layer == i
        assert mem.vec == ffn.vec == frames * proj * 4
        assert mem.sfu == ffn.sfu == frames
        assert mem.vec != frames * hidden * 4
        assert mem.a_fmt == ffn.a_fmt == "fp32"
        assert mem.w_params == ffn.w_params == 0 and mem.flops == ffn.flops == 0
        assert names.index(f"aud.{i}.mem_ln") < names.index(f"aud.{i}.dw")
        assert names.index(f"aud.{i}.dw") < names.index(f"aud.{i}.ffn_ln")
        assert names.index(f"aud.{i}.ffn_ln") + 1 == names.index(f"aud.{i}.ffn_up")
    assert next(op for op in ops if op.name == "aud.0.dw").vec == frames * proj * 11
    assert next(op for op in ops if op.name == "aud.16.dw").vec == frames * proj * 12
    assert next(op for op in ops if op.name == "aud.16.mem_ln").vec == frames * proj * 4
    ln_sfu = sum(op.sfu for op in ops if op.name.endswith((".mem_ln", ".ffn_ln")))
    assert ln_sfu == nlayer * 2 * frames
    wide = audio_ops(au, 1.0, 2)
    assert next(op for op in wide if op.name == "aud.0.ffn_ln").sfu == 2 * frames
    assert audio_ops(au, 0.01, 1) == []
    tiny = AudioSpec(16000, (160,), (32,), (1,), 3, 1.0, 32, 1, kind="g4lin")
    assert not any(op.name.endswith((".mem_ln", ".ffn_ln")) for op in audio_ops(tiny, 0.1, 1))
