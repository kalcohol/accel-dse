"""0.407: LongCat vision and FSMN bill the elementwise ops the public classes run.

SwiGLU applies ACT2FN["silu"] to fc1 only, then multiplies by fc3, then the
inner RMSNorm. VisionProjector.p0 runs two nn.GELU() (erf, not tanh) on the
width-paired rows. apply_rotary_pos_emb_vision upcasts q and k and computes
cos/sin on freqs of length head_dim/2, once per call. The audio encoder's
released activation is relu6, on the FFN hidden width. It multiplies by an
int32 mask before each layer and before the DNN. The audio projector GELU is
the same erf form on the text hidden width. Softmax stays the bidirectional
score vector. Other towers do not grow these ops.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import AudioSpec, audio_ops, image_grid, vision_ops

HIDDEN = 1280
INNER = 3456
DEPTH = 32
HEADS = 16
HD = HIDDEN // HEADS
ADAPTOR = 7168
PROJ = 1536
AUDIO_H = 6144
TEXT = 6144
AUDIO_LAYERS = 22
NDNN = 2


def test_longcat_vision_bills_silu_gelu_and_rope_on_the_published_shapes():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "longcatvit" and vs.gated
    assert vs.head_dim == HD and HD % 2 == 0
    g = image_grid(vs, 1024, 1024)
    tokens = g.ph * g.pw
    assert tokens == 72 * 72 and g.merged == tokens // 2
    ops = vision_ops(vs, g, 1, "fp8")
    assert ops and all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    half = HD // 2
    for li in range(DEPTH):
        assert names.index(f"vis.{li}.qkv") + 1 == names.index(f"vis.{li}.rope")
        assert names.index(f"vis.{li}.rope") + 1 == names.index(f"vis.{li}.rope_repeat")
        assert names.index(f"vis.{li}.rope_repeat") + 1 == names.index(f"vis.{li}.q_norm")
        assert names.index(f"vis.{li}.mlp_in") + 1 == names.index(f"vis.{li}.silu")
        assert names.index(f"vis.{li}.silu") + 1 == names.index(f"vis.{li}.mlp_mul")
        assert names.index(f"vis.{li}.mlp_mul") + 1 == names.index(f"vis.{li}.mlp_norm")
        silu = next(op for op in ops if op.name == f"vis.{li}.silu")
        mul = next(op for op in ops if op.name == f"vis.{li}.mlp_mul")
        rope = next(op for op in ops if op.name == f"vis.{li}.rope")
        assert silu.kind == mul.kind == rope.kind == "vector"
        assert silu.layer == mul.layer == rope.layer == li
        assert silu.vec == tokens * INNER * 4 and silu.sfu == tokens * INNER * 2
        assert silu.vec != tokens * (2 * INNER) * 4
        assert silu.a_fmt == "bf16" and silu.w_params == 0 and silu.flops == 0
        assert mul.vec == tokens * INNER and mul.sfu == 0 and mul.a_fmt == "bf16"
        assert mul.w_params == 0 and mul.flops == 0
        assert rope.vec == 2 * tokens * (HEADS * HD * 3 + half * 2)
        assert rope.sfu == 2 * tokens * half * 2
        assert rope.sfu != tokens * HEADS and rope.sfu != tokens * HEADS * HD
        assert rope.a_fmt == "fp32" and rope.w_params == 0 and rope.flops == 0
    assert names.index("vis.merger.0") + 1 == names.index("vis.proj_gelu0")
    assert names.index("vis.proj_gelu0") + 1 == names.index("vis.merger.1")
    assert names.index("vis.merger.1") + 1 == names.index("vis.proj_gelu1")
    assert names.index("vis.proj_gelu1") + 1 == names.index("vis.merger.2")
    assert "vis.proj_gelu2" not in names
    for name in ("vis.proj_gelu0", "vis.proj_gelu1"):
        gelu = next(op for op in ops if op.name == name)
        assert gelu.kind == "vector" and gelu.layer == DEPTH
        assert gelu.vec == g.merged * ADAPTOR * 5 and gelu.sfu == g.merged * ADAPTOR
        assert gelu.vec != tokens * ADAPTOR * 5 and gelu.vec != g.merged * (2 * HIDDEN) * 5
        assert gelu.a_fmt == "bf16" and gelu.sfu != 0
        assert gelu.w_params == 0 and gelu.flops == 0
    sm = next(op for op in ops if op.name == "vis.0.softmax")
    assert sm.vec == HEADS * tokens * tokens * 5 and sm.sfu == 0
    norm_sfu = sum(op.sfu for op in ops if op.name == "vis.proj_ln" or op.name.endswith(
        (".norm1", ".q_norm", ".k_norm", ".norm2", ".mlp_norm")))
    silu_sfu = sum(op.sfu for op in ops if op.name.endswith(".silu"))
    rope_sfu = sum(op.sfu for op in ops if op.name.endswith(".rope"))
    gelu_sfu = sum(op.sfu for op in ops if op.name.startswith("vis.proj_gelu"))
    assert silu_sfu == DEPTH * tokens * INNER * 2
    assert rope_sfu == DEPTH * 2 * tokens * half * 2
    assert gelu_sfu == 2 * g.merged * ADAPTOR
    assert sum(op.sfu for op in ops) == norm_sfu + silu_sfu + rope_sfu + gelu_sfu
    wide = vision_ops(vs, g, 2)
    assert next(op for op in wide if op.name == "vis.0.silu").sfu == 2 * tokens * INNER * 2
    assert next(op for op in wide if op.name == "vis.0.rope").sfu == 2 * (2 * tokens) * half * 2
    assert next(op for op in wide if op.name == "vis.proj_gelu0").sfu == 2 * g.merged * ADAPTOR
    assert vision_ops(vs, g, 0) == []
    text = " ".join(m.notes)
    fp_text = " ".join(fp.notes)
    for needle in ("SiLU", "erf", "head_dim/2", "relu6", "int32", "emb_mult"):
        assert needle in text and needle in fp_text
    other = get_model("step-5-preview").vision
    assert other is not None and other.family != "longcatvit"
    og = image_grid(other, 1024, 1024)
    assert og is not None
    onames = [op.name for op in vision_ops(other, og, 1)]
    assert not any(name.endswith((".silu", ".mlp_mul", ".rope")) or name.startswith("vis.proj_gelu")
                   for name in onames)


def test_longcat_audio_bills_relu6_mask_and_projector_gelu():
    vs = get_model("longcat-flash-omni").vision
    assert vs is not None and vs.audio is not None and vs.audio.kind == "lcafsmn"
    au = vs.audio
    frames = au.tokens(1.0)
    proj, hidden = au.dims[0], au.dims[1]
    nlayer, _right, ndnn = au.depths
    assert (frames, proj, hidden, nlayer, ndnn) == (13, PROJ, AUDIO_H, AUDIO_LAYERS, NDNN)
    assert au.lin is not None and au.lin[0] == (AUDIO_H, TEXT) and au.lin[1] == (TEXT, TEXT)
    ops = audio_ops(au, 1.0, 1, "fp8")
    assert all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in ops if op.kind == "gemm")
    names = [op.name for op in ops]
    assert names.index("aud.pre") + 1 == names.index("aud.pre_relu")
    assert names.index("aud.pre_relu") + 1 == names.index("aud.pre_down")
    relus = [op for op in ops if op.name.endswith(".relu") or op.name.endswith("_relu")]
    assert len(relus) == 1 + nlayer + ndnn
    for op in relus:
        assert op.kind == "vector" and op.vec == frames * hidden * 2 and op.sfu == 0
        assert op.vec != frames * proj * 2
        assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0
    masks = [op for op in ops if op.name.endswith(".mask") or op.name == "aud.dnn_mask"]
    assert len(masks) == nlayer + 1
    for op in masks:
        assert op.kind == "vector" and op.vec == frames * proj and op.sfu == 0
        assert op.vec != frames * hidden
        assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
    for i in range(nlayer):
        assert names.index(f"aud.{i}.mask") + 1 == names.index(f"aud.{i}.mem_ln")
        assert names.index(f"aud.{i}.ffn_up") + 1 == names.index(f"aud.{i}.relu")
        assert names.index(f"aud.{i}.relu") + 1 == names.index(f"aud.{i}.ffn_down")
        assert next(op for op in ops if op.name == f"aud.{i}.relu").layer == i
        assert next(op for op in ops if op.name == f"aud.{i}.mask").layer == i
    assert names.index("aud.dnn_mask") + 1 == names.index("aud.dnn.0")
    assert names.index("aud.dnn.0") + 1 == names.index("aud.dnn.0.relu")
    assert names.index("aud.dnn.1") + 1 == names.index("aud.dnn.1.relu")
    assert names.index("aud.dnn.1.relu") + 1 == names.index("aud.out")
    assert names.index("aud.proj.0") + 1 == names.index("aud.proj_gelu")
    assert names.index("aud.proj_gelu") + 1 == names.index("aud.proj.1")
    assert "aud.proj_gelu1" not in names
    gelu = next(op for op in ops if op.name == "aud.proj_gelu")
    assert gelu.kind == "vector" and gelu.layer == 0
    assert gelu.vec == frames * TEXT * 5 and gelu.sfu == frames * TEXT
    assert gelu.vec != frames * proj * 5
    assert gelu.a_fmt == "bf16" and gelu.w_params == 0 and gelu.flops == 0
    assert sum(op.sfu for op in ops) == nlayer * 2 * frames + frames * TEXT
    assert not any("drop" in op.name or "emb_mult" in op.name for op in ops)
    wide = audio_ops(au, 1.0, 2)
    assert next(op for op in wide if op.name == "aud.0.relu").vec == 2 * frames * hidden * 2
    assert next(op for op in wide if op.name == "aud.dnn_mask").vec == 2 * frames * proj
    assert next(op for op in wide if op.name == "aud.proj_gelu").sfu == 2 * frames * TEXT
    assert audio_ops(au, 0.01, 1) == []
    tiny = AudioSpec(16000, (160,), (32,), (1,), 3, 1.0, 32, 1, kind="g4lin")
    assert not any(op.name.endswith((".relu", ".mask", "_relu")) or "proj_gelu" in op.name
                   or op.name == "aud.dnn_mask" for op in audio_ops(tiny, 0.1, 1))
