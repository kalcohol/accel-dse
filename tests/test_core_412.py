"""0.412: d1 subsampling bills nn.ReLU and the time-mask multiply.

ConvSubsampling (audio.py) multiplies by _time_mask before every Sequential
module and once more before the linear. nn.ReLU is max(x, 0): one clip.
ReLU6 is two clips, so these widths are not doubled. The mask is one multiply
in fp32. The depthwise masked_fill is a where and stays out. The sinusoid
table stays out. GEMM shapes are unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops

MID = "d1-omni-600m"
CH = 256
FEAT = 128


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _conv_len(n: int) -> int:
    return (n - 1) // 2 + 1


def _mel(seconds: float) -> int:
    n = min(30 * 16000, max(0, round(seconds * 16000)))
    if n < 8000:
        n = 8000
    return n // 160


def _shapes(mel: int) -> tuple[tuple[int, int, int], ...]:
    """(channels, time, freq) seen by each mask, then by the following module."""
    c, t, f = 1, mel, FEAT
    out = []
    for kind in ("conv", "relu", "dw", "pw", "relu", "dw", "pw", "relu"):
        out.append((c, t, f))
        if kind == "conv":
            t, f, c = _conv_len(t), _conv_len(f), CH
        elif kind == "dw":
            t, f = _conv_len(t), _conv_len(f)
    out.append((CH, t, f))
    return tuple(out)


def test_d1_subsampling_bills_relu_and_time_mask():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    ops = audio_ops(m.vision.audio, 1.0, 1, "fp32")
    by = _by(ops)
    names = [op.name for op in ops]
    mel = _mel(1.0)
    assert mel == 100
    shapes = _shapes(mel)
    assert len(shapes) == 9
    order = [
        "aud.sub.mask0", "aud.sub.0", "aud.sub.mask1", "aud.sub.relu0",
        "aud.sub.mask2", "aud.sub.dw0", "aud.sub.mask3", "aud.sub.pw0",
        "aud.sub.mask4", "aud.sub.relu1", "aud.sub.mask5", "aud.sub.dw1",
        "aud.sub.mask6", "aud.sub.pw1", "aud.sub.mask7", "aud.sub.relu2",
        "aud.sub.mask8", "aud.sub.out",
    ]
    assert [name for name in names if name.startswith("aud.sub.")] == order
    assert names.index("aud.sub.out") + 1 == names.index("aud.0.ff1_ln")
    for i, (c, t, f) in enumerate(shapes):
        mask = by[f"aud.sub.mask{i}"]
        assert mask.kind == "vector" and mask.vec == c * t * f and mask.sfu == 0
        assert mask.a_fmt == "fp32" and mask.w_params == 0 and mask.flops == 0
        assert mask.layer == 0 and mask.vec != c * t * f * 2
    assert by["aud.sub.mask0"].vec == 100 * FEAT
    assert by["aud.sub.mask0"].vec != CH * 100 * FEAT
    relus = [by[f"aud.sub.relu{i}"] for i in range(3)]
    expect = (CH * 50 * 64, CH * 25 * 32, CH * 13 * 16)
    for op, elems in zip(relus, expect):
        assert op.kind == "vector" and op.vec == elems and op.sfu == 0
        assert op.vec != elems * 2 and op.a_fmt == "fp32"
        assert op.w_params == 0 and op.flops == 0 and op.layer == 0
    assert by["aud.sub.relu0"].vec == by["aud.sub.mask1"].vec == by["aud.sub.mask2"].vec
    assert by["aud.sub.relu2"].vec == by["aud.sub.mask8"].vec
    assert by["aud.sub.0"].m == 50 * 64 and by["aud.sub.0"].k == 9 and by["aud.sub.0"].n == CH
    assert by["aud.sub.0"].vec == 50 * 64 * CH * 3 and by["aud.sub.0"].a_fmt == "fp32"
    assert by["aud.sub.dw0"].vec == 25 * 32 * CH * 9
    assert by["aud.sub.pw0"].m == 25 * 32 and by["aud.sub.pw1"].m == 13 * 16
    assert by["aud.sub.out"].m == 13 and by["aud.sub.out"].k == CH * 16 and by["aud.sub.out"].n == 512
    assert by["aud.0.ff1_silu"].vec == 13 * 2048 * 4
    assert by["aud.0.pos"].m == 25
    assert not any("fill" in name or "pad" in name or "sin" in name or name.endswith(".pos_emb") for name in by)
    assert sum(1 for name in by if name.startswith("aud.sub.relu")) == 3
    assert sum(1 for name in by if name.startswith("aud.sub.mask")) == 9
    wide = _by(audio_ops(m.vision.audio, 1.0, 2, "fp32"))
    assert wide["aud.sub.relu0"].vec == 2 * relus[0].vec
    assert wide["aud.sub.mask0"].vec == 2 * by["aud.sub.mask0"].vec
    assert wide["aud.sub.mask8"].vec == 2 * by["aud.sub.mask8"].vec
    assert wide["aud.0.pos"].m == by["aud.0.pos"].m
    assert wide["aud.sub.0"].m == 2 * by["aud.sub.0"].m
    long = _by(audio_ops(m.vision.audio, 10.0, 1, "fp32"))
    assert _mel(10.0) == 1000
    assert long["aud.sub.mask0"].vec == 1000 * FEAT
    assert long["aud.sub.relu0"].vec == CH * 500 * 64
    assert long["aud.sub.relu1"].vec == CH * 250 * 32
    assert long["aud.sub.relu2"].vec == CH * 125 * 16
    assert long["aud.sub.out"].m == 125 and long["aud.0.ff1_up"].m == 125
    short = _by(audio_ops(m.vision.audio, 0.5, 1, "fp32"))
    assert _mel(0.5) == 50
    assert short["aud.sub.mask0"].vec == 50 * FEAT
    assert short["aud.sub.relu0"].vec == CH * 25 * 64
    assert short["aud.sub.relu1"].vec == CH * 13 * 32
    assert short["aud.sub.relu2"].vec == CH * 7 * 16
    assert short["aud.sub.out"].m == 7
    forced = _by(audio_ops(m.vision.audio, 1.0, 1, "fp8"))
    assert forced["aud.sub.relu0"].a_fmt == "fp32" and forced["aud.sub.mask0"].a_fmt == "fp32"
    assert forced["aud.sub.relu2"].vec == relus[2].vec
    assert forced["aud.sub.0"].a_fmt == "fp32" and forced["aud.sub.0"].vec == by["aud.sub.0"].vec
    text = " ".join(m.notes)
    for needle in ("nn.ReLU", "一次截断", "relu6", "时间 mask", "一次乘", "比较不另计",
                   "masked_fill", "where", "正弦", "不是 GEMM", "gelu_pytorch_tanh"):
        assert needle in text, needle
    assert audio_ops(m.vision.audio, 0.0, 1) == []
    for mid in ("omniasr-ctc-300m-v2", "granite-speech-5.0-2b", "longcat-flash-omni"):
        other = [op.name for op in audio_ops(get_model(mid).vision.audio, 1.0, 1)]
        assert not any(name.startswith("aud.sub.relu") or name.startswith("aud.sub.mask") for name in other)
