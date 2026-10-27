"""0.413: Gemma-4 audio conformer bills the folded tails that match existing recipes.

Gemma4AudioSubSampleConvProjectionLayer multiplies by the feature mask, then
Conv2d, then ReLU(LayerNorm). The feed-forward is SiLU and a 0.5 residual.
LightConv is GLU on the kept half and SiLU at hidden width. With-scale
RMSNorm matches the LongCat count. Softcap, the scale-free embed norm, and
the full residual adds stay out. GEMM shapes are unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops, image_grid, vision_ops

MODELS = (
    "gemma-4-e2b-it",
    "gemma-4-e2b",
    "gemma-4-e4b-it",
    "gemma-4-e4b",
    "nemotron-3.5-bolt-e4b-bf16",
)
H = 1024
FF = 4096
C0 = 128
C1 = 32
MELS = 128
HEADS = 8
LAYERS = 12
RMS = (
    "ff1_rms", "ff1_post", "attn_rms", "attn_out", "lconv_rms",
    "lconv_norm", "ff2_rms", "ff2_post", "out_rms",
)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _layer_order(li: int) -> list[str]:
    p = f"aud.{li}."
    return [
        p + "ff1_rms", p + "ff1_up", p + "ff1_silu", p + "ff1_down", p + "ff1_post", p + "ff1_half",
        p + "attn_rms", p + "q", p + "k", p + "v", p + "rel", p + "qk", p + "relqk", p + "pv",
        p + "softmax", p + "post", p + "attn_out",
        p + "lconv_rms", p + "lconv_start", p + "glu", p + "lconv", p + "lconv_norm",
        p + "lconv_silu", p + "lconv_end",
        p + "ff2_rms", p + "ff2_up", p + "ff2_silu", p + "ff2_down", p + "ff2_post", p + "ff2_half",
        p + "out_rms",
    ]


def test_gemma4_audio_catalog_is_the_conformer():
    for mid in MODELS:
        m = get_model(mid)
        au = m.vision.audio
        assert m.param_check()["rel_err"] == 0.0, mid
        assert au is not None and au.kind == "g4conf" and au.dims == (H, C0, C1), mid
        assert au.depths == (LAYERS,) and au.heads == HEADS and au.w_fmt == "bf16", mid
        text = " ".join(m.notes)
        for needle in (
            "silu", "residual_weight", "0.5", "LayerNorm", "nn.ReLU", "RMSNorm",
            "softcap", "gelu_pytorch_tanh", "4*hidden", "不是 2*hidden", "通用 GEMM 尾项不减",
        ):
            assert needle in text, (mid, needle)


def test_gemma4_audio_bills_subsampling_and_block_tails():
    m = get_model("gemma-4-e2b-it")
    ops = audio_ops(m.vision.audio, 1.0, 1, "bf16")
    by = _by(ops)
    names = [op.name for op in ops]
    sub = [
        "aud.sub.mask0", "aud.conv0", "aud.sub.ln0", "aud.sub.relu0", "aud.sub.mask1",
        "aud.conv1", "aud.sub.ln1", "aud.sub.relu1", "aud.sub_proj",
    ]
    order = sub + _layer_order(0)
    assert names[:len(order)] == order
    assert names[names.index("aud.11.ff1_rms"):names.index("aud.out")] == _layer_order(11)
    assert names[-2:] == ["aud.out", "aud.embed"]

    mask0 = by["aud.sub.mask0"]
    assert mask0.kind == "vector" and mask0.vec == 99 * MELS and mask0.sfu == 0
    assert mask0.a_fmt == "fp32" and mask0.w_params == 0 and mask0.flops == 0
    assert mask0.vec != C0 * 99 * MELS
    conv0 = by["aud.conv0"]
    assert (conv0.m, conv0.k, conv0.n) == (3200, 9, C0)
    assert conv0.vec == 3200 * C0 * 3 and conv0.a_fmt == "bf16"
    ln0 = by["aud.sub.ln0"]
    assert ln0.vec == 50 * 64 * C0 * 4 and ln0.sfu == 50 * 64 and ln0.a_fmt == "fp32"
    assert ln0.w_params == 0
    relu0 = by["aud.sub.relu0"]
    assert relu0.vec == 50 * 64 * C0 and relu0.vec != 50 * 64 * C0 * 2
    assert relu0.sfu == 0 and relu0.a_fmt == "bf16" and relu0.flops == 0
    assert by["aud.sub.mask1"].vec == relu0.vec and by["aud.sub.mask1"].a_fmt == "bf16"
    assert by["aud.sub.mask1"].a_fmt != "fp32"
    conv1 = by["aud.conv1"]
    assert (conv1.m, conv1.k, conv1.n) == (800, 9 * C0, C1) and conv1.vec == 800 * C1 * 3
    ln1 = by["aud.sub.ln1"]
    assert ln1.vec == 25 * 32 * C1 * 4 and ln1.sfu == 25 * 32 and ln1.a_fmt == "fp32"
    assert by["aud.sub.relu1"].vec == 25 * 32 * C1 and by["aud.sub.relu1"].a_fmt == "bf16"
    assert by["aud.sub_proj"].m == 25

    silu = by["aud.0.ff1_silu"]
    assert silu.vec == 25 * FF * 4 and silu.sfu == 25 * FF * 2 and silu.a_fmt == "bf16"
    assert silu.layer == 0 and silu.vec != 25 * H * 4
    assert by["aud.0.ff2_silu"].vec == silu.vec and by["aud.0.ff2_silu"].sfu == silu.sfu
    assert by["aud.0.ff1_up"].vec == 25 * FF * 3 and by["aud.0.ff1_up"].a_fmt == "bf16"
    glu = by["aud.0.glu"]
    assert glu.vec == 25 * H * 4 and glu.sfu == 25 * H * 2 and glu.a_fmt == "bf16"
    assert glu.vec != 25 * (2 * H) * 4
    lsilu = by["aud.0.lconv_silu"]
    assert lsilu.vec == 25 * H * 4 and lsilu.sfu == 25 * H * 2 and lsilu.a_fmt == "bf16"
    half = by["aud.0.ff1_half"]
    assert half.vec == 25 * H * 2 and half.sfu == 0 and half.a_fmt == "bf16"
    assert by["aud.0.ff2_half"].vec == half.vec
    rms = by["aud.0.ff1_rms"]
    assert rms.vec == 25 * H * 4 and rms.sfu == 25 and rms.a_fmt == "fp32" and rms.w_params == 0
    for suffix in RMS:
        op = by[f"aud.0.{suffix}"]
        assert op.vec == rms.vec and op.sfu == rms.sfu and op.a_fmt == "fp32", suffix
        assert by[f"aud.11.{suffix}"].vec == op.vec
    assert by["aud.11.ff1_silu"].vec == silu.vec and by["aud.11.glu"].vec == glu.vec
    sm = by["aud.0.softmax"]
    assert sm.sfu == 0 and sm.vec == HEADS * 36 * 24 * 5
    assert by["aud.0.rel"].m == 13
    assert by["aud.0.lconv"].vec == 25 * H * 5 and by["aud.0.lconv"].a_fmt == "bf16"
    assert by["aud.out"].n == 1536 and by["aud.embed"].n == 1536 and by["aud.embed"].m == 25

    assert sum(1 for name in by if name.endswith(tuple("." + s for s in RMS))) == LAYERS * len(RMS)
    assert sum(1 for name in by if name.endswith(".ff1_silu") or name.endswith(".ff2_silu")) == LAYERS * 2
    assert sum(1 for name in by if name.endswith(".lconv_silu")) == LAYERS
    assert not any(
        "embed" in name and "rms" in name or "softcap" in name or "lconv_half" in name or "gelu" in name
        for name in by
    )

    wide = _by(audio_ops(m.vision.audio, 1.0, 2, "bf16"))
    for name in (
        "aud.sub.mask0", "aud.sub.mask1", "aud.sub.relu0", "aud.sub.relu1", "aud.sub.ln0", "aud.sub.ln1",
        "aud.0.ff1_silu", "aud.0.ff2_silu", "aud.0.lconv_silu", "aud.0.glu", "aud.0.ff1_half", "aud.0.ff2_half",
        "aud.0.ff1_rms",
    ):
        assert wide[name].vec == 2 * by[name].vec, name
    assert wide["aud.0.ff1_rms"].sfu == 2 * rms.sfu
    assert wide["aud.0.out_rms"].sfu == 2 * by["aud.0.out_rms"].sfu
    assert wide["aud.0.rel"].m == 13 and wide["aud.0.ff1_up"].m == 50

    long = _by(audio_ops(m.vision.audio, 10.0, 1, "bf16"))
    assert long["aud.sub.mask0"].vec == 999 * MELS
    assert long["aud.0.ff1_silu"].vec == 250 * FF * 4
    assert long["aud.sub.ln0"].sfu == 500 * 64
    assert long["aud.embed"].m == 250 and long["aud.0.ff1_up"].m == 250

    short = _by(audio_ops(m.vision.audio, 0.5, 1, "bf16"))
    assert short["aud.sub.mask0"].vec == 50 * MELS
    assert short["aud.sub.relu0"].vec == 25 * 64 * C0
    assert short["aud.0.ff1_up"].m == 13 and short["aud.embed"].m == 13

    forced = _by(audio_ops(m.vision.audio, 1.0, 1, "fp8"))
    assert forced["aud.0.ff1_up"].a_fmt == "fp8" and forced["aud.conv0"].a_fmt == "fp8"
    assert forced["aud.0.ff1_up"].vec == by["aud.0.ff1_up"].vec
    for name in ("aud.0.ff1_silu", "aud.0.glu", "aud.0.ff1_half", "aud.sub.relu0", "aud.sub.mask1", "aud.0.lconv_silu"):
        assert forced[name].a_fmt == "bf16", name
    assert forced["aud.sub.mask0"].a_fmt == "fp32"
    assert forced["aud.sub.ln0"].a_fmt == "fp32" and forced["aud.0.ff1_rms"].a_fmt == "fp32"
    assert audio_ops(m.vision.audio, 0.0, 1) == []

    e4 = _by(audio_ops(get_model("gemma-4-e4b-it").vision.audio, 1.0, 1))
    nem = _by(audio_ops(get_model("nemotron-3.5-bolt-e4b-bf16").vision.audio, 1.0, 1))
    assert e4["aud.embed"].n == 2560 and nem["aud.embed"].n == 2560
    assert e4["aud.out"].n == 1536 and nem["aud.out"].n == 1536
    assert e4["aud.0.ff1_silu"].vec == silu.vec

    au = get_model("gemma-4-e4b-it").vision.audio
    hit = None
    for samples in range(160, 8000):
        sec = samples / 16000.0
        tower, prompt = au._g4_audio_lens(sec)
        if tower > prompt > 0:
            hit = (sec, tower, prompt)
            break
    assert hit is not None
    sec, tower, prompt = hit
    grown = _by(audio_ops(au, sec, 2))
    assert grown["aud.0.ff1_up"].m == tower * 2
    assert grown["aud.0.ff1_up"].m != prompt * 2

    vis = vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)
    assert not any(op.name.endswith(".silu") for op in vis)
    lin = audio_ops(get_model("gemma-4-12b-it").vision.audio, 1.0, 1)
    assert get_model("gemma-4-12b-it").vision.audio.kind == "g4lin"
    assert [op.name for op in lin] == ["aud.proj"]
    for mid in ("d1-omni-600m", "omniasr-ctc-300m-v2", "granite-speech-5.0-2b", "longcat-flash-omni"):
        other = [op.name for op in audio_ops(get_model(mid).vision.audio, 1.0, 1)]
        assert "aud.sub.ln0" not in other, mid
        assert not any(name.endswith(".ff1_rms") or name.endswith(".lconv_silu") for name in other), mid
