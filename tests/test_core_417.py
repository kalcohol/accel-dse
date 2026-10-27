"""0.417: OmniASR bills the LayerNorms the published forward runs.

Each conv output is normalized on the conv channels before erf-GELU. The
projection is normalized on those channels again. Each block has a pre-attention
norm and a pre-FFN norm. The encoder norm follows the blocks. The positional
conv has none. LayerNorm reduces in fp32. The GEMM tail is unchanged.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops

FE_1S = (3199, 1599, 799, 399, 199, 99, 49)
FE_HALF = (1599, 799, 399, 199, 99, 49, 24)
CONV = 512
HEADS = 16
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


def _is_ln(name: str) -> bool:
    return name.endswith(".ln") or name.endswith("_ln")


def _expect(layers: int, head: str) -> list[str]:
    names = []
    for i in range(7):
        names += [f"aud.fe.{i}", f"aud.fe.{i}.ln", f"aud.fe.{i}.gelu"]
    names += ["aud.fe.ln", "aud.fe.proj", "aud.pos", "aud.pos.gelu"]
    for li in range(layers):
        p = f"aud.{li}."
        names += [p + "attn_ln", p + "qkv", p + "qk", p + "pv", p + "softmax", p + "o",
                  p + "mlp_ln", p + "fc1", p + "gelu", p + "fc2"]
    names += ["aud.ln", head]
    return names


def _ln(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
    assert op.vec != rows * width * 5


def test_omniasr_bills_layernorm_on_conv_proj_and_blocks():
    for mid, hidden, layers, ff in CTC + LLM:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        au = m.vision.audio
        assert au.kind == "w2v" and au.w_fmt == "fp32" and au.heads == HEADS, mid
        ops = audio_ops(au, 1.0, 1, "fp32")
        by = _by(ops)
        head = "aud.ctc" if au.ctc else "aud.proj.linear0"
        assert [op.name for op in ops] == _expect(layers, head), mid
        assert au.tokens(1.0) == 49
        for i, length in enumerate(FE_1S):
            _ln(by[f"aud.fe.{i}.ln"], length, CONV, 0)
            assert by[f"aud.fe.{i}.ln"].vec != length * hidden * 4
            assert by[f"aud.fe.{i}.gelu"].vec == length * CONV * 5
        _ln(by["aud.fe.ln"], 49, CONV, 0)
        assert by["aud.fe.ln"].vec != 49 * hidden * 4
        assert by["aud.fe.proj"].k == CONV and by["aud.fe.proj"].n == hidden
        assert "aud.pos.ln" not in by
        attn = by["aud.0.attn_ln"]
        mlp = by["aud.0.mlp_ln"]
        _ln(attn, 49, hidden, 0)
        _ln(mlp, 49, hidden, 0)
        assert attn.vec != 49 * CONV * 4 and mlp.vec == attn.vec
        _ln(by[f"aud.{layers - 1}.attn_ln"], 49, hidden, layers - 1)
        _ln(by[f"aud.{layers - 1}.mlp_ln"], 49, hidden, layers - 1)
        final = by["aud.ln"]
        _ln(final, 49, hidden, layers)
        assert final.vec == attn.vec
        assert by["aud.0.fc1"].vec == 49 * ff * 3 and by["aud.0.fc1"].w_fmt == "fp32"
        assert by["aud.0.fc1"].w_bits == 32.0
        assert by["aud.0.softmax"].sfu == 0 and by["aud.0.softmax"].vec == HEADS * 49 * 49 * 5
        assert by["aud.0.qk"].m == 49 and by["aud.0.qk"].count == HEADS
        assert sum(1 for name in by if _is_ln(name)) == 9 + 2 * layers
        assert sum(op.w_params for op in ops if _is_ln(op.name)) == 0
        assert f"{head}.gelu" not in by and f"{head}.ln" not in by
        assert not any(name.endswith(".half") or "mask" in name for name in by)

        wide = _by(audio_ops(au, 1.0, 2, "fp32"))
        assert wide["aud.fe.0.ln"].vec == 2 * by["aud.fe.0.ln"].vec
        assert wide["aud.fe.0.ln"].sfu == 2 * by["aud.fe.0.ln"].sfu
        assert wide["aud.fe.ln"].sfu == 2 * 49
        assert wide["aud.0.attn_ln"].sfu == 2 * 49 and wide["aud.0.mlp_ln"].vec == 2 * mlp.vec
        assert wide["aud.ln"].sfu == 98
        assert wide["aud.0.qk"].m == 49 and wide["aud.0.qk"].count == 2 * HEADS
        assert wide["aud.0.fc1"].vec == 2 * by["aud.0.fc1"].vec

        assert au.tokens(10.0) == 499
        long = _by(audio_ops(au, 10.0, 1, "fp32"))
        assert long["aud.fe.6.ln"].sfu == 499 and long["aud.fe.ln"].vec == 499 * CONV * 4
        assert long["aud.0.attn_ln"].vec == 499 * hidden * 4
        assert long["aud.ln"].sfu == 499 and long["aud.0.gelu"].vec == 499 * ff * 5

        assert au.tokens(0.5) == 24
        short = _by(audio_ops(au, 0.5, 1, "fp32"))
        for i, length in enumerate(FE_HALF):
            assert short[f"aud.fe.{i}.ln"].sfu == length
            assert short[f"aud.fe.{i}.ln"].vec == length * CONV * 4
        assert short["aud.fe.ln"].vec == 24 * CONV * 4
        assert short["aud.0.mlp_ln"].vec == 24 * hidden * 4
        assert short["aud.ln"].sfu == 24

        forced = _by(audio_ops(au, 1.0, 1, "fp8"))
        assert forced["aud.0.fc1"].a_fmt == "fp8" and forced["aud.fe.0"].a_fmt == "fp8"
        assert forced["aud.0.fc1"].w_fmt == "fp32" and forced["aud.fe.proj"].w_fmt == "fp32"
        assert forced["aud.0.fc1"].w_bits == 32.0
        assert forced["aud.0.fc1"].vec == by["aud.0.fc1"].vec
        for name in ("aud.fe.0.ln", "aud.fe.ln", "aud.0.attn_ln", "aud.0.mlp_ln", "aud.ln"):
            assert forced[name].a_fmt == "fp32" and forced[name].vec == by[name].vec, name
        assert forced["aud.0.gelu"].a_fmt == "fp32" and forced["aud.fe.0.gelu"].a_fmt == "fp32"

        text = " ".join(m.notes)
        for needle in ("LayerNorm", "fp32", "不是 GEMM", "Dropout", "LayerDrop", "位置卷积没有 LayerNorm"):
            assert needle in text, (mid, needle)
        if au.ctc:
            for needle in ("ctc_head", "没有 Llama", "Shaw", "masked_fill"):
                assert needle in text, (mid, needle)
        else:
            assert "multi_modal_projector" in text and "hidden" in text

    quiet = audio_ops(get_model("omniasr-ctc-300m-v2").vision.audio, 0.0, 1, "fp32")
    assert [op.name for op in quiet if _is_ln(op.name)] == [f"aud.fe.{i}.ln" for i in range(7)]
    assert all(op.vec == 0 and op.sfu == 0 for op in quiet if _is_ln(op.name))
    for mid in ("whisper-large-v3", "qwen3-asr-1.7b", "fun-asr-nano-2512",
                "d1-omni-600m", "gemma-4-e2b-it", "granite-speech-5.0-2b"):
        names = [op.name for op in audio_ops(get_model(mid).vision.audio, 1.0, 1)]
        assert "aud.fe.0.ln" not in names and "aud.fe.ln" not in names, mid
