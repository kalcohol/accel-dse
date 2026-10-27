"""0.410: OmniASR bills the published erf-GELU.

ACT2FN["gelu"] is nn.functional.gelu. The feature extractor runs it on every
conv output, the positional conv runs it after the even-kernel pad slice, and
each FFN runs it on the intermediate width. The class does not cast. Eval
dropout is not an op. The GEMM tail is unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops

# (L - k) // s + 1 on 16 kHz, one second. The last length is the encoder frames.
FE_1S = (3199, 1599, 799, 399, 199, 99, 49)
CONV = 512
CTC = (
    ("omniasr-ctc-300m-v2", 1024, 24, 4096),
    ("omniasr-ctc-1b-v2", 1280, 48, 5120),
    ("omniasr-ctc-3b-v2", 2048, 60, 8192),
    ("omniasr-ctc-7b-v2", 2048, 128, 8192),
)
LLM = (
    ("omniasr-llm-300m-v2", 1024, 24, 4096),
    ("omniasr-llm-1b-v2", 1280, 48, 5120),
    ("omniasr-llm-3b-v2", 2048, 60, 8192),
    ("omniasr-llm-7b-v2", 2048, 128, 8192),
)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _gelu_sites(by, frames, hidden, layers, ff, clips=1):
    rows = frames * clips
    for i, length in enumerate(FE_1S):
        op = by[f"aud.fe.{i}.gelu"]
        n = length * clips * CONV
        assert op.kind == "vector" and op.vec == n * 5 and op.sfu == float(n)
        assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
        assert op.vec != length * clips * hidden * 5
    pos = by["aud.pos.gelu"]
    assert pos.vec == rows * hidden * 5 and pos.sfu == float(rows * hidden)
    assert pos.a_fmt == "fp32" and pos.vec != rows * CONV * 5
    gelu = by["aud.0.gelu"]
    assert gelu.vec == rows * ff * 5 and gelu.sfu == float(rows * ff) and gelu.layer == 0
    assert gelu.a_fmt == "fp32" and gelu.vec != rows * hidden * 5
    assert by[f"aud.{layers - 1}.gelu"].vec == gelu.vec
    assert by[f"aud.{layers - 1}.gelu"].layer == layers - 1
    assert sum(1 for name in by if name.endswith(".gelu") and name.startswith("aud.fe.")) == 7
    assert sum(1 for name in by if name.endswith(".gelu")) == 7 + 1 + layers


def test_omniasr_bills_erf_gelu_on_conv_pos_and_ffn():
    for mid, hidden, layers, ff in CTC + LLM:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        ops = audio_ops(m.vision.audio, 1.0, 1, "fp32")
        by = _by(ops)
        names = [op.name for op in ops]
        assert by["aud.fe.6"].m == 49 and by["aud.0.fc1"].m == 49
        assert by["aud.0.fc1"].n == ff and by["aud.pos"].n == hidden
        for i in range(7):
            assert names.index(f"aud.fe.{i}") + 1 == names.index(f"aud.fe.{i}.gelu")
        assert names.index("aud.fe.6.gelu") + 1 == names.index("aud.fe.proj")
        assert names.index("aud.pos") + 1 == names.index("aud.pos.gelu")
        assert names.index("aud.0.fc1") + 1 == names.index("aud.0.gelu")
        assert names.index("aud.0.gelu") + 1 == names.index("aud.0.fc2")
        assert "aud.fe.proj.gelu" not in by and "aud.proj.gelu" not in by
        head = "aud.ctc" if m.vision.audio.ctc else "aud.proj.linear0"
        assert head in by and f"{head}.gelu" not in by
        assert not any(name.endswith(".gelu") and "proj" in name for name in by)
        _gelu_sites(by, 49, hidden, layers, ff)
        assert by["aud.0.fc1"].vec == 49 * ff * 3
        assert by["aud.0.softmax"].sfu == 0 and by["aud.0.softmax"].vec == 16 * 49 * 49 * 5
        wide = _by(audio_ops(m.vision.audio, 1.0, 2, "fp32"))
        assert wide["aud.fe.0.gelu"].vec == 2 * by["aud.fe.0.gelu"].vec
        assert wide["aud.fe.0.gelu"].sfu == 2 * by["aud.fe.0.gelu"].sfu
        assert wide["aud.pos.gelu"].sfu == 2 * by["aud.pos.gelu"].sfu
        assert wide["aud.0.gelu"].vec == 2 * by["aud.0.gelu"].vec
        long = _by(audio_ops(m.vision.audio, 10.0, 1, "fp32"))
        assert long["aud.fe.6.gelu"].sfu == 499 * CONV
        assert long["aud.0.gelu"].vec == 499 * ff * 5
        assert long["aud.pos.gelu"].vec == 499 * hidden * 5
        forced = _by(audio_ops(m.vision.audio, 1.0, 1, "fp8"))
        assert forced["aud.0.fc1"].a_fmt == "fp8" and forced["aud.fe.0"].a_fmt == "fp8"
        assert forced["aud.fe.0.gelu"].a_fmt == "fp32"
        assert forced["aud.pos.gelu"].a_fmt == "fp32"
        assert forced["aud.0.gelu"].a_fmt == "fp32"
        assert forced["aud.0.fc1"].vec == by["aud.0.fc1"].vec
        text = " ".join(m.notes)
        for needle in ("erf", "Dropout", "LayerDrop", "LayerNorm"):
            assert needle in text, (mid, needle)
        if m.vision.audio.ctc:
            assert "没有 Llama" in text and "Shaw" in text and "不是 GEMM" in text
            assert "ctc_head" in text
        else:
            assert "multi_modal_projector" in text and "hidden" in text
    quiet = audio_ops(get_model("omniasr-ctc-300m-v2").vision.audio, 0.0, 1, "fp32")
    assert [op.name for op in quiet if op.name.endswith(".gelu")] == [f"aud.fe.{i}.gelu" for i in range(7)]
    assert all(op.vec == 0 and op.sfu == 0 for op in quiet if op.name.endswith(".gelu"))
    other = get_model("granite-speech-5.0-2b").vision.audio
    onames = [op.name for op in audio_ops(other, 1.0, 1)]
    assert not any(name.startswith("aud.fe.") and name.endswith(".gelu") for name in onames)
    d1 = [op.name for op in audio_ops(get_model("d1-omni-600m").vision.audio, 1.0, 1)]
    assert not any(name.endswith(".gelu") for name in d1)
