"""0.409: Granite Speech bills the published SiLU, GLU, half-residual and LayerNorms.

GraniteSpeechConformerBlock and GraniteSpeech5EncoderBlock run those ops on
countable widths. The Q-Former query path runs erf-GELU. Eval dropout is not
an op. BatchNorm rsqrt is once per channel. The GEMM tail is unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def test_granite_speech_bills_silu_glu_and_half_residual():
    base = get_model("granite-speech-5.0-2b")
    plus = get_model("granite-speech-4.1-2b-plus")
    assert base.param_check()["rel_err"] == 0.0 and plus.param_check()["rel_err"] == 0.0
    ops = audio_ops(base.vision.audio, 1.0, 1)
    by = _by(ops)
    names = [op.name for op in ops]
    t = by["aud.in"].m
    h = by["aud.in"].n
    ff = by["aud.0.ff1_up"].n
    inner = by["aud.0.conv_down"].k
    assert (t, h, ff, inner) == (50, 1024, 4096, 2048)
    assert names.index("aud.0.ff1_ln") < names.index("aud.0.ff1_up")
    assert names.index("aud.0.ff1_up") + 1 == names.index("aud.0.ff1_silu")
    assert names.index("aud.0.ff1_silu") + 1 == names.index("aud.0.ff1_down")
    assert names.index("aud.0.ff1_down") + 1 == names.index("aud.0.ff1_half")
    assert names.index("aud.0.ff1_half") + 1 == names.index("aud.0.attn_ln")
    assert names.index("aud.0.attn_ln") < names.index("aud.0.q")
    assert names.index("aud.0.conv_up") + 1 == names.index("aud.0.glu")
    assert names.index("aud.0.glu") + 1 == names.index("aud.0.dconv")
    assert names.index("aud.0.dconv") + 1 == names.index("aud.0.bn")
    assert names.index("aud.0.bn") + 1 == names.index("aud.0.conv_silu")
    assert names.index("aud.0.ff2_up") + 1 == names.index("aud.0.ff2_silu")
    assert names.index("aud.0.ff2_half") + 1 == names.index("aud.0.out_ln")
    silu = by["aud.0.ff1_silu"]
    assert silu.kind == "vector" and silu.vec == t * ff * 4 and silu.sfu == t * ff * 2
    assert silu.a_fmt == "bf16" and silu.w_params == 0 and silu.flops == 0
    assert silu.vec != t * h * 4
    assert by["aud.0.ff2_silu"].vec == silu.vec and by["aud.0.ff2_silu"].sfu == silu.sfu
    glu = by["aud.0.glu"]
    assert glu.vec == t * inner * 4 and glu.sfu == t * inner * 2 and glu.a_fmt == "bf16"
    assert glu.vec != t * (2 * inner) * 4
    # This checkpoint's FFN width is 2 * conv inner. SiLU still uses the FFN width.
    assert ff == 2 * inner and silu.vec == 2 * glu.vec
    bn = by["aud.0.bn"]
    assert bn.vec == t * inner * 4 and bn.sfu == inner and bn.a_fmt == "bf16"
    assert bn.sfu != t * inner and bn.sfu != float(t)
    half = by["aud.0.ff1_half"]
    assert half.vec == t * h * 2 and half.sfu == 0 and half.a_fmt == "bf16"
    assert by["aud.0.ff2_half"].vec == half.vec
    ln = by["aud.0.attn_ln"]
    assert ln.vec == t * h * 4 and ln.sfu == t and ln.a_fmt == "fp32"
    assert by["aud.0.q"].m == 200 and ln.sfu != by["aud.0.q"].m
    for name in ("aud.0.ff1_ln", "aud.0.conv_ln", "aud.0.ff2_ln", "aud.0.out_ln"):
        assert by[name].sfu == t and by[name].a_fmt == "fp32"
    assert by["aud.0.conv_silu"].vec == t * inner * 4 and by["aud.0.conv_silu"].a_fmt == "bf16"
    assert by["aud.0.ff1_up"].vec == t * ff * 3
    assert by["aud.0.softmax"].sfu == 0 and by["aud.0.softmax"].vec == 8 * 200 * 200 * 5
    assert sum(1 for name in by if name.endswith(".ff1_silu")) == 16
    assert not any("pool" in name or "drop" in name or "emb_mult" in name for name in by)
    nq = by["aud.qf.0.fc1"].m
    qff = by["aud.qf.0.fc1"].n
    qh = by["aud.qf.0.fc1"].k
    assert (nq, qh, qff) == (12, 1024, 4096)
    gelu = by["aud.qf.0.gelu"]
    assert names.index("aud.qf.0.fc1") + 1 == names.index("aud.qf.0.gelu")
    assert gelu.vec == nq * qff * 5 and gelu.sfu == nq * qff and gelu.a_fmt == "bf16"
    assert gelu.vec != nq * qh * 5
    assert by["aud.qf.ln"].sfu == nq and by["aud.qf.ln"].a_fmt == "fp32"
    assert by["aud.qf.0.attn_ln"].sfu == nq and by["aud.qf.0.cross_ln"].sfu == nq
    assert by["aud.qf.1.out_ln"].vec == nq * qh * 4
    assert by["aud.qf.0.softmax"].sfu == 0 and by["aud.qf.0.csoftmax"].sfu == 0
    wide = _by(audio_ops(base.vision.audio, 1.0, 2))
    assert wide["aud.0.ff1_silu"].vec == 2 * silu.vec
    assert wide["aud.0.bn"].sfu == bn.sfu and wide["aud.0.bn"].vec == 2 * bn.vec
    assert wide["aud.qf.0.gelu"].vec == 2 * gelu.vec
    assert wide["aud.qf.ln"].sfu == 2 * nq
    same = _by(audio_ops(plus.vision.audio, 1.0, 1))
    assert same["aud.0.ff1_silu"].vec == silu.vec and same["aud.0.glu"].vec == glu.vec
    assert same["aud.qf.0.ck"].k == 2048 and by["aud.qf.0.ck"].k == 1024
    assert same["aud.qf.0.gelu"].vec == gelu.vec
    forced = _by(audio_ops(base.vision.audio, 1.0, 1, "fp8"))
    assert forced["aud.0.ff1_up"].a_fmt == "fp8"
    assert forced["aud.0.ff1_silu"].a_fmt == "bf16" and forced["aud.0.glu"].a_fmt == "bf16"
    assert forced["aud.0.bn"].a_fmt == "bf16" and forced["aud.qf.0.gelu"].a_fmt == "bf16"
    text = " ".join(base.notes)
    plus_text = " ".join(plus.notes)
    for needle in ("SiLU", "GLU", "erf", "Dropout", "不另计 GEMM", "Shaw"):
        assert needle in text and needle in plus_text
    assert "float32" in plus_text
    assert audio_ops(base.vision.audio, 0.0, 1) == []


def test_granite_speech5_ctc_bills_pool_and_fp32_batchnorm():
    m = get_model("granite-speech-5.0-470m-turboctc")
    assert m.param_check()["rel_err"] == 0.0
    ops = audio_ops(m.vision.audio, 1.0, 1)
    by = _by(ops)
    names = [op.name for op in ops]
    h, ff, inner = 1024, 4096, 2048
    assert by["aud.0.ff1_up"].m == 50 and by["aud.0.conv_down"].m == 25 and by["aud.0.ff2_up"].m == 25
    assert by["aud.1.ff1_up"].m == 25 and by["aud.1.conv_down"].m == 13 and by["aud.1.ff2_up"].m == 12
    assert by["aud.2.ff1_up"].m == 12 and by["aud.2.conv_down"].m == 12 and by["aud.2.ff2_up"].m == 12
    assert by["aud.0.ff1_silu"].vec == 50 * ff * 4 and by["aud.0.ff1_silu"].a_fmt == "bf16"
    assert by["aud.0.ff2_silu"].vec == 25 * ff * 4
    assert by["aud.1.ff1_silu"].vec == 25 * ff * 4 and by["aud.1.ff2_silu"].vec == 12 * ff * 4
    assert by["aud.0.glu"].vec == 50 * inner * 4 and by["aud.1.glu"].vec == 25 * inner * 4
    assert by["aud.0.glu"].a_fmt == "bf16" and by["aud.0.glu"].vec != 50 * (2 * inner) * 4
    assert by["aud.0.bn"].vec == 25 * inner * 4 and by["aud.0.bn"].sfu == inner
    assert by["aud.0.bn"].a_fmt == "fp32" and by["aud.0.conv_silu"].a_fmt == "fp32"
    assert by["aud.0.conv_silu"].vec == 25 * inner * 4 and by["aud.0.conv_silu"].sfu == 25 * inner * 2
    assert by["aud.1.bn"].vec == 13 * inner * 4 and by["aud.1.bn"].sfu == inner
    assert by["aud.1.conv_silu"].vec == 13 * inner * 4 and by["aud.1.conv_silu"].a_fmt == "fp32"
    assert by["aud.2.bn"].a_fmt == "fp32" and by["aud.2.conv_silu"].vec == 12 * inner * 4
    assert by["aud.0.pool"].vec == 25 * h * 2 and by["aud.0.pool"].sfu == 0 and by["aud.0.pool"].a_fmt == "bf16"
    assert by["aud.1.pool"].vec == 12 * h * 2
    assert "aud.2.pool" not in by and "aud.15.pool" not in by
    assert names.index("aud.0.conv_down") < names.index("aud.0.pool") < names.index("aud.0.ff2_ln")
    assert by["aud.0.ff1_ln"].sfu == 50 and by["aud.0.out_ln"].sfu == 25
    assert by["aud.1.attn_ln"].sfu == 25 and by["aud.1.ff2_ln"].sfu == 12 and by["aud.1.out_ln"].sfu == 12
    assert by["aud.0.ff1_half"].vec == 50 * h * 2 and by["aud.1.ff2_half"].vec == 12 * h * 2
    assert by["aud.0.ff1_ln"].a_fmt == "fp32" and by["aud.0.softmax"].sfu == 0
    assert by["aud.0.ff1_up"].vec == 50 * ff * 3
    assert not any("qf" in name or "drop" in name or "gelu" in name for name in by)
    wide = _by(audio_ops(m.vision.audio, 1.0, 2))
    assert wide["aud.0.ff1_silu"].vec == 2 * by["aud.0.ff1_silu"].vec
    assert wide["aud.0.bn"].sfu == inner and wide["aud.1.bn"].sfu == inner
    assert wide["aud.0.bn"].vec == 2 * by["aud.0.bn"].vec
    assert wide["aud.0.pool"].vec == 2 * by["aud.0.pool"].vec
    forced = _by(audio_ops(m.vision.audio, 1.0, 1, "fp8"))
    assert forced["aud.0.ff1_up"].a_fmt == "fp8"
    assert forced["aud.0.ff1_silu"].a_fmt == "bf16" and forced["aud.0.glu"].a_fmt == "bf16"
    assert forced["aud.0.bn"].a_fmt == "fp32" and forced["aud.0.conv_silu"].a_fmt == "fp32"
    text = " ".join(m.notes)
    for needle in ("没有 granite", "SiLU", "GLU", "fp32", "activation_dropout", "不是 GEMM"):
        assert needle in text
    other = get_model("omniasr-ctc-300m-v2").vision.audio
    onames = [op.name for op in audio_ops(other, 1.0, 1, "fp32")]
    assert not any(name.endswith(".glu") or name.endswith(".ff1_silu") or name.endswith(".bn")
                   or name.endswith(".pool") for name in onames)
    assert audio_ops(m.vision.audio, 0.0, 1) == []
