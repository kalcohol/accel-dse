"""0.411: d1 FastConformer bills the published SiLU, GLU, half-residual, LayerNorm and BatchNorm.

ConformerLayer runs those ops on countable widths. F.glu keeps d_model. The 0.5 residual
is x + ff * 0.5 on d_model. BatchNorm rsqrt is once per channel. The adapter and the
residual run F.gelu (erf). The class does not cast. Subsampling ReLU is not an op.
The GEMM tail is unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops

MID = "d1-omni-600m"
LAYERS = 17
D = 512
FF = 2048
OUT = 1024
RES = 512


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def test_d1_fastconformer_bills_silu_glu_half_and_batchnorm():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    ops = audio_ops(m.vision.audio, 1.0, 1, "fp32")
    by = _by(ops)
    names = [op.name for op in ops]
    t = by["aud.0.ff1_up"].m
    assert t == 13
    assert by["aud.0.ff1_up"].n == FF and by["aud.0.ff1_up"].k == D
    assert by["aud.0.pw1"].n == 2 * D and by["aud.adapter.1"].n == OUT
    assert by["aud.residual.down"].n == RES and by["aud.residual.up"].k == RES
    assert names.index("aud.sub.out") + 1 == names.index("aud.0.ff1_ln")
    assert names.index("aud.0.ff1_ln") + 1 == names.index("aud.0.ff1_up")
    assert names.index("aud.0.ff1_up") + 1 == names.index("aud.0.ff1_silu")
    assert names.index("aud.0.ff1_silu") + 1 == names.index("aud.0.ff1_down")
    assert names.index("aud.0.ff1_down") + 1 == names.index("aud.0.ff1_half")
    assert names.index("aud.0.ff1_half") + 1 == names.index("aud.0.attn_ln")
    assert names.index("aud.0.attn_ln") + 1 == names.index("aud.0.q")
    assert names.index("aud.0.softmax") + 1 == names.index("aud.0.conv_ln")
    assert names.index("aud.0.conv_ln") + 1 == names.index("aud.0.pw1")
    assert names.index("aud.0.pw1") + 1 == names.index("aud.0.glu")
    assert names.index("aud.0.glu") + 1 == names.index("aud.0.dw")
    assert names.index("aud.0.dw") + 1 == names.index("aud.0.bn")
    assert names.index("aud.0.bn") + 1 == names.index("aud.0.conv_silu")
    assert names.index("aud.0.conv_silu") + 1 == names.index("aud.0.pw2")
    assert names.index("aud.0.pw2") + 1 == names.index("aud.0.ff2_ln")
    assert names.index("aud.0.ff2_up") + 1 == names.index("aud.0.ff2_silu")
    assert names.index("aud.0.ff2_down") + 1 == names.index("aud.0.ff2_half")
    assert names.index("aud.0.ff2_half") + 1 == names.index("aud.0.out_ln")
    assert names.index("aud.16.out_ln") + 1 == names.index("aud.adapter.ln")
    assert names.index("aud.adapter.ln") + 1 == names.index("aud.adapter.1")
    assert names.index("aud.adapter.1") + 1 == names.index("aud.adapter.gelu")
    assert names.index("aud.adapter.gelu") + 1 == names.index("aud.adapter.2")
    assert names.index("aud.adapter.2") + 1 == names.index("aud.residual.ln")
    assert names.index("aud.residual.ln") + 1 == names.index("aud.residual.down")
    assert names.index("aud.residual.down") + 1 == names.index("aud.residual.gelu")
    assert names.index("aud.residual.gelu") + 1 == names.index("aud.residual.up")
    silu = by["aud.0.ff1_silu"]
    assert silu.kind == "vector" and silu.vec == t * FF * 4 and silu.sfu == t * FF * 2
    assert silu.a_fmt == "fp32" and silu.w_params == 0 and silu.flops == 0 and silu.layer == 0
    assert silu.vec != t * D * 4
    assert by["aud.0.ff2_silu"].vec == silu.vec and by["aud.0.ff2_silu"].sfu == silu.sfu
    glu = by["aud.0.glu"]
    assert glu.vec == t * D * 4 and glu.sfu == t * D * 2 and glu.a_fmt == "fp32"
    assert glu.vec != t * (2 * D) * 4 and glu.w_params == 0
    bn = by["aud.0.bn"]
    assert bn.vec == t * D * 4 and bn.sfu == D and bn.a_fmt == "fp32"
    assert bn.sfu != t * D and bn.sfu != float(t) and bn.w_params == 0
    half = by["aud.0.ff1_half"]
    assert half.vec == t * D * 2 and half.sfu == 0 and half.a_fmt == "fp32"
    assert by["aud.0.ff2_half"].vec == half.vec
    assert by["aud.0.conv_silu"].vec == t * D * 4 and by["aud.0.conv_silu"].sfu == t * D * 2
    assert by["aud.0.conv_silu"].a_fmt == "fp32"
    for name in ("aud.0.ff1_ln", "aud.0.attn_ln", "aud.0.conv_ln", "aud.0.ff2_ln", "aud.0.out_ln"):
        assert by[name].vec == t * D * 4 and by[name].sfu == t and by[name].a_fmt == "fp32"
        assert by[name].w_params == 0
    assert by["aud.adapter.ln"].vec == t * D * 4 and by["aud.adapter.ln"].sfu == t
    assert by["aud.residual.ln"].vec == t * OUT * 4 and by["aud.residual.ln"].sfu == t
    assert by["aud.residual.ln"].a_fmt == "fp32" and by["aud.adapter.ln"].a_fmt == "fp32"
    gelu = by["aud.adapter.gelu"]
    assert gelu.kind == "vector" and gelu.vec == t * OUT * 5 and gelu.sfu == float(t * OUT)
    assert gelu.a_fmt == "fp32" and gelu.w_params == 0 and gelu.flops == 0
    assert gelu.vec != t * D * 5
    assert by["aud.residual.gelu"].vec == t * RES * 5 and by["aud.residual.gelu"].sfu == float(t * RES)
    assert by["aud.residual.gelu"].vec != t * OUT * 5
    assert by["aud.0.ff1_up"].vec == t * FF * 3 and by["aud.0.ff1_up"].a_fmt == "fp32"
    assert by["aud.0.ff1_up"].w_fmt == "fp32"
    assert by["aud.0.softmax"].sfu == 0 and by["aud.0.softmax"].vec == 8 * t * t * 5
    assert by["aud.0.pos"].m == 2 * t - 1
    assert by["aud.16.glu"].layer == LAYERS - 1 and by["aud.16.out_ln"].sfu == t
    assert "aud.17.glu" not in by
    for suffix in (".ff1_silu", ".ff2_silu", ".conv_silu", ".glu", ".bn", ".ff1_half", ".ff2_half",
                   ".ff1_ln", ".attn_ln", ".conv_ln", ".ff2_ln", ".out_ln"):
        assert sum(1 for name in by if name.endswith(suffix)) == LAYERS
    assert sum(1 for name in by if name.endswith(".gelu")) == 2
    assert not any("drop" in name or "emb_mult" in name for name in by)
    # 0.412 bills subsampling ReLU and the time-mask multiply. Conformer layers still have neither.
    assert not any(("relu" in name or "mask" in name) and not name.startswith("aud.sub.") for name in by)
    core = [name for name in names if name.startswith("aud.sub.") and "relu" not in name and "mask" not in name]
    assert core == ["aud.sub.0", "aud.sub.dw0", "aud.sub.pw0", "aud.sub.dw1", "aud.sub.pw1", "aud.sub.out"]
    wide = _by(audio_ops(m.vision.audio, 1.0, 2, "fp32"))
    assert wide["aud.0.ff1_silu"].vec == 2 * silu.vec and wide["aud.0.ff1_silu"].sfu == 2 * silu.sfu
    assert wide["aud.0.glu"].vec == 2 * glu.vec and wide["aud.0.ff1_half"].vec == 2 * half.vec
    assert wide["aud.0.bn"].sfu == bn.sfu and wide["aud.0.bn"].vec == 2 * bn.vec
    assert wide["aud.0.ff1_ln"].sfu == 2 * t and wide["aud.adapter.gelu"].vec == 2 * gelu.vec
    assert wide["aud.0.pos"].m == by["aud.0.pos"].m
    long = _by(audio_ops(m.vision.audio, 10.0, 1, "fp32"))
    assert long["aud.0.ff1_up"].m == 125
    assert long["aud.0.ff1_silu"].vec == 125 * FF * 4 and long["aud.0.glu"].vec == 125 * D * 4
    assert long["aud.0.bn"].sfu == D and long["aud.0.bn"].vec == 125 * D * 4
    assert long["aud.0.ff1_half"].vec == 125 * D * 2
    assert long["aud.adapter.gelu"].sfu == 125 * OUT and long["aud.residual.gelu"].vec == 125 * RES * 5
    assert long["aud.0.out_ln"].sfu == 125
    short = _by(audio_ops(m.vision.audio, 0.5, 1, "fp32"))
    assert short["aud.0.ff1_up"].m == 7 and short["aud.0.glu"].vec == 7 * D * 4
    assert short["aud.0.bn"].sfu == D and short["aud.adapter.gelu"].vec == 7 * OUT * 5
    forced = _by(audio_ops(m.vision.audio, 1.0, 1, "fp8"))
    assert forced["aud.0.ff1_up"].a_fmt == "fp32" and forced["aud.0.pw1"].a_fmt == "fp32"
    assert forced["aud.0.ff1_silu"].a_fmt == "fp32" and forced["aud.0.glu"].a_fmt == "fp32"
    assert forced["aud.0.bn"].a_fmt == "fp32" and forced["aud.0.conv_silu"].a_fmt == "fp32"
    assert forced["aud.0.ff1_half"].a_fmt == "fp32" and forced["aud.adapter.gelu"].a_fmt == "fp32"
    assert forced["aud.residual.gelu"].a_fmt == "fp32" and forced["aud.0.ff1_ln"].a_fmt == "fp32"
    assert forced["aud.0.ff1_up"].vec == by["aud.0.ff1_up"].vec
    text = " ".join(m.notes)
    for needle in ("SiLU", "GLU", "erf", "BatchNorm", "fp32", "不是 GEMM", "gelu_pytorch_tanh",
                   "双向", "lm_head", "F.gelu"):
        assert needle in text, needle
    assert audio_ops(m.vision.audio, 0.0, 1) == []
    other = [op.name for op in audio_ops(get_model("omniasr-ctc-300m-v2").vision.audio, 1.0, 1, "fp32")]
    assert not any(name.endswith(".glu") or name.endswith(".bn") or name.endswith(".ff1_silu")
                   or name == "aud.adapter.gelu" for name in other)
    granite = [op.name for op in audio_ops(get_model("granite-speech-5.0-2b").vision.audio, 1.0, 1)]
    assert "aud.adapter.gelu" not in granite and "aud.residual.gelu" not in granite
