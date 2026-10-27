"""0.416: the AuT encoder bills erf-GELU and the LayerNorms it runs.

Conv2d outputs, each FFN, and the projector use the erf form. Attention and the
MLP each have a pre-norm. The encoder LayerNorm sits after the blocks. Index-Echo
2B also bills F.gelu on connector w1. Vector dtypes follow the vision role. GEMM
weights stay bf16. The connector scalars are not ops.
"""
from __future__ import annotations

import json
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import audio_ops, audio_spec

ROOT = Path(__file__).resolve().parents[1]
C = 480
ASR = (
    ("qwen3-asr-1.7b", 1024, 24, 4096, 16, 2048),
    ("qwen3-asr-0.6b", 896, 18, 3584, 14, 1024),
)
OMNI = (
    "qwen3-omni-30b-a3b",
    "qwen3-omni-30b-a3b-thinking",
    "qwen3-omni-30b-a3b-captioner",
)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _conv_elems(nch: int) -> tuple[int, int, int]:
    f, t = 128, 100
    out = []
    for _ in range(3):
        f, t = (f + 1) // 2, (t + 1) // 2
        out.append(nch * f * t * C)
    return tuple(out)


def _raises(cfg: dict, needle: str) -> None:
    try:
        audio_spec(cfg)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _layer(li: int, windows: int) -> list[str]:
    p = f"aud.{li}."
    names = [p + "attn_ln", p + "qkv"]
    for wi in range(windows):
        sfx = f".w{wi}" if wi else ""
        names += [p + "qk" + sfx, p + "pv" + sfx, p + "softmax" + sfx]
    names += [p + "o", p + "mlp_ln", p + "fc1", p + "gelu", p + "fc2"]
    return names


def test_aut_bills_erf_gelu_and_layernorm():
    for mid, d, layers, ff, heads, out in ASR:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        au = m.vision.audio
        assert au.kind == "aut" and au.vec_fmt == "bf16" and au.gelu_lin == (0,), mid
        ops = audio_ops(au, 1.0, 1, "fp32")
        by = _by(ops)
        names = [op.name for op in ops]
        assert au.enc_positions(1.0) == 13
        conv = _conv_elems(1)
        for i, n in enumerate(conv, start=1):
            gelu = by[f"aud.conv2d{i}.gelu"]
            assert gelu.kind == "vector" and gelu.vec == n * 5 and gelu.sfu == float(n)
            assert gelu.a_fmt == "bf16" and gelu.w_params == 0 and gelu.flops == 0
            assert names.index(f"aud.conv2d{i}") + 1 == names.index(f"aud.conv2d{i}.gelu")
        assert names.index("aud.conv2d3.gelu") + 1 == names.index("aud.conv_out")
        assert "aud.conv_out.gelu" not in by
        assert by["aud.conv2d1"].m == 64 * 50 and by["aud.conv2d1"].n == C
        assert by["aud.conv_out"].m == 13 and by["aud.conv_out"].k == C * 16
        block = _layer(0, 1)
        assert names[names.index("aud.0.attn_ln"):names.index("aud.0.fc2") + 1] == block
        assert names[names.index(f"aud.{layers - 1}.fc2"):] == [
            f"aud.{layers - 1}.fc2", "aud.ln", "aud.proj.linear0", "aud.proj.gelu",
            "aud.proj.linear1",
        ]
        ln = by["aud.0.attn_ln"]
        assert ln.vec == 13 * d * 4 and ln.sfu == 13 and ln.a_fmt == "fp32" and ln.layer == 0
        assert by["aud.0.mlp_ln"].vec == ln.vec and by["aud.ln"].vec == ln.vec
        assert by["aud.ln"].sfu == 13 and by["aud.ln"].layer == layers
        gelu = by["aud.0.gelu"]
        assert gelu.vec == 13 * ff * 5 and gelu.sfu == float(13 * ff) and gelu.a_fmt == "bf16"
        assert gelu.layer == 0 and gelu.vec != 13 * d * 5
        assert by[f"aud.{layers - 1}.gelu"].layer == layers - 1
        proj = by["aud.proj.gelu"]
        assert proj.vec == 13 * d * 5 and proj.sfu == float(13 * d) and proj.a_fmt == "bf16"
        assert proj.vec != 13 * out * 5
        assert (by["aud.proj.linear0"].k, by["aud.proj.linear0"].n) == (d, d)
        assert (by["aud.proj.linear1"].k, by["aud.proj.linear1"].n) == (d, out)
        assert by["aud.0.fc1"].vec == 13 * ff * 3 and by["aud.0.fc1"].w_fmt == "bf16"
        assert by["aud.0.qkv"].m == 13 and by["aud.0.qk"].m == 13 and by["aud.0.qk"].count == heads
        assert by["aud.0.softmax"].sfu == 0 and by["aud.0.softmax"].vec == heads * 13 * 13 * 5
        assert "aud.conn.gelu" not in by and "aud.proj.relu" not in by and "aud.proj.silu" not in by
        text = " ".join(m.notes)
        for needle in ("erf", "LayerNorm", "F.gelu", "bf16", "beta"):
            assert needle in text, (mid, needle)
        half = _by(audio_ops(au, 0.5, 1))
        assert au.enc_positions(0.5) == 7
        assert half["aud.conv2d1.gelu"].vec == by["aud.conv2d1.gelu"].vec
        assert half["aud.0.gelu"].vec == 7 * ff * 5 and half["aud.ln"].sfu == 7
        assert half["aud.proj.gelu"].vec == 7 * d * 5
        wide = _by(audio_ops(au, 1.5, 1))
        assert au.enc_positions(1.5) == 20 and len(au.chunks(1.5)) == 2
        assert wide["aud.conv2d3.gelu"].vec == 2 * by["aud.conv2d3.gelu"].vec
        assert wide["aud.0.gelu"].vec == 20 * ff * 5 and wide["aud.ln"].sfu == 20
        two = _by(audio_ops(au, 1.0, 2))
        assert two["aud.conv2d1.gelu"].vec == 2 * by["aud.conv2d1.gelu"].vec
        assert two["aud.0.gelu"].sfu == 2 * gelu.sfu and two["aud.ln"].sfu == 26
        assert two["aud.0.qk"].m == 13 and two["aud.0.qk"].count == 2 * heads
        long = _by(audio_ops(au, 10.0, 1))
        assert au.enc_positions(10.0) == 130 and len(au.chunks(10.0)) == 10
        assert long["aud.conv2d2.gelu"].vec == 10 * by["aud.conv2d2.gelu"].vec
        assert long["aud.0.gelu"].vec == 130 * ff * 5 and long["aud.ln"].sfu == 130
        assert long["aud.0.qk"].m == 104 and long["aud.0.qk.w1"].m == 26
        assert long["aud.proj.gelu"].vec == 130 * d * 5
        forced = _by(audio_ops(au, 1.0, 1, "fp8"))
        assert forced["aud.0.fc1"].a_fmt == "fp8" and forced["aud.conv2d1"].a_fmt == "fp8"
        assert forced["aud.0.fc1"].w_fmt == "bf16" and forced["aud.0.fc1"].vec == by["aud.0.fc1"].vec
        assert forced["aud.conv2d1.gelu"].a_fmt == "bf16" and forced["aud.0.gelu"].a_fmt == "bf16"
        assert forced["aud.proj.gelu"].a_fmt == "bf16" and forced["aud.0.attn_ln"].a_fmt == "fp32"
        assert forced["aud.ln"].a_fmt == "fp32"


def test_qwen3_omni_uses_the_same_aut_counts():
    base = None
    for mid in OMNI:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        au = m.vision.audio
        assert au.kind == "aut" and au.vec_fmt == "bf16" and au.dims[0] == 1280, mid
        assert au.depths[0] == 32 and au.gelu_lin == (0,), mid
        by = _by(audio_ops(au, 1.0, 1))
        assert by["aud.0.gelu"].vec == 13 * 5120 * 5 and by["aud.0.gelu"].a_fmt == "bf16"
        assert by["aud.ln"].vec == 13 * 1280 * 4 and by["aud.ln"].sfu == 13
        assert by["aud.proj.gelu"].vec == 13 * 1280 * 5
        assert by["aud.proj.linear1"].n == 2048 and "aud.conn.gelu" not in by
        assert by["aud.31.gelu"].layer == 31
        if base is None:
            base = by["aud.0.gelu"].vec
        assert by["aud.0.gelu"].vec == base


def test_index_echo_connector_gelu_is_only_on_2b():
    two = get_model("index-echo-s2st-2b")
    nine = get_model("index-echo-s2st-9b")
    assert two.param_check()["rel_err"] == 0.0 and nine.param_check()["rel_err"] == 0.0
    a2, a9 = two.vision.audio, nine.vision.audio
    assert a2.kind == "aut" and a2.vec_fmt == "bf16" and a2.gelu_lin == (0, 2)
    assert a9.gelu_lin == (0,) and a9.out == 4096 and a2.out == 2048
    b2 = _by(audio_ops(a2, 1.0, 1))
    names = [op.name for op in audio_ops(a2, 1.0, 1)]
    assert names[names.index("aud.ln"):] == [
        "aud.ln", "aud.proj.linear0", "aud.proj.gelu", "aud.proj.linear1",
        "aud.proj.linear2", "aud.conn.gelu", "aud.proj.linear3",
    ]
    assert b2["aud.proj.gelu"].vec == 13 * 1280 * 5 and b2["aud.proj.gelu"].a_fmt == "bf16"
    conn = b2["aud.conn.gelu"]
    assert conn.vec == 13 * 2048 * 5 and conn.sfu == float(13 * 2048) and conn.a_fmt == "bf16"
    assert conn.w_params == 0 and conn.flops == 0
    assert (b2["aud.proj.linear2"].k, b2["aud.proj.linear2"].n) == (2048, 2048)
    assert (b2["aud.proj.linear3"].k, b2["aud.proj.linear3"].n) == (2048, 2048)
    assert "aud.proj.linear1.gelu" not in b2 and "aud.proj.linear3.gelu" not in b2
    wide = _by(audio_ops(a2, 1.0, 2))
    assert wide["aud.conn.gelu"].vec == 2 * conn.vec and wide["aud.ln"].sfu == 26
    b9 = _by(audio_ops(a9, 1.0, 1))
    assert "aud.conn.gelu" not in b9
    assert b9["aud.proj.gelu"].vec == 13 * 1280 * 5
    assert (b9["aud.proj.linear2"].k, b9["aud.proj.linear2"].n) == (2048, 4096)
    assert b9["aud.proj.linear1"].n == 2048
    text = " ".join(two.notes)
    assert "w1" in text and "9B" in text and "beta" in text


def test_other_towers_do_not_gain_aut_conv_gelu():
    for mid in ("whisper-large-v3", "omniasr-ctc-300m-v2", "fun-asr-nano-2512",
                "d1-omni-600m", "gemma-4-e2b-it"):
        names = [op.name for op in audio_ops(get_model(mid).vision.audio, 1.0, 1)]
        assert "aud.conv2d1.gelu" not in names and "aud.conn.gelu" not in names, mid


def test_aut_refuses_a_non_erf_activation_and_a_strange_connector():
    q = json.loads((ROOT / "accel_dse/data/releases/Qwen__Qwen3-ASR-1.7B.json").read_text())["config"]
    bad = json.loads(json.dumps(q))
    bad["thinker_config"]["audio_config"]["activation_function"] = "gelu_pytorch_tanh"
    _raises(bad, "aut activation_function is not gelu")
    missing = json.loads(json.dumps(q))
    missing["thinker_config"]["audio_config"].pop("activation_function")
    _raises(missing, "aut activation_function is not gelu")
    om = json.loads((ROOT / "accel_dse/data/releases/Qwen__Qwen3-Omni-30B-A3B-Instruct.json").read_text())["config"]
    obad = json.loads(json.dumps(om))
    obad["thinker_config"]["audio_config"]["activation_function"] = "quick_gelu"
    _raises(obad, "aut activation_function is not gelu")
    echo = json.loads((ROOT / "accel_dse/data/releases/IndexTeam__Index-Echo-S2ST-2B.json").read_text())["config"]
    strange = json.loads(json.dumps(echo))
    strange["_echo_connector"] = {"other.weight": [8, 8]}
    _raises(strange, "index echo connector is not w1/w2 or proj")
