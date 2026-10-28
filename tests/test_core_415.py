"""0.415: the shared Whisper encoder bills erf-GELU and the LayerNorms it runs.

conv1 / conv2 and each block MLP use the erf form. Attention and MLP each have a
pre-norm. The encoder LayerNorm sits after the blocks, after the pool on
Qwen2-Audio, Qwen2.5-Omni and Step-Audio, before the stack on Kimi, and before
the projector on Ming. Step also bills adaptor GELU and ReLU. Kimi bills adaptor
SiLU and LayerNorm. Vector dtypes follow the vision role. GEMM weights stay bf16.
"""
from __future__ import annotations

import json
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.model import load_release
from accel_dse.core.vision import audio_ops, audio_spec

W3 = "whisper-large-v3"
D = 1280
FF = 5120
HEADS = 20
LAYERS = 32
MEL = 3000
T = 1500
ROOT = Path(__file__).resolve().parents[1]


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _layer(li: int) -> list[str]:
    p = f"aud.{li}."
    return [
        p + "attn_ln", p + "qkv", p + "qk", p + "pv", p + "softmax", p + "o",
        p + "mlp_ln", p + "fc1", p + "gelu", p + "fc2",
    ]


def _raises(cfg: dict, needle: str) -> None:
    try:
        audio_spec(cfg)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_whisper_family_keeps_the_tower_and_the_vision_dtype():
    ents = {e["id"]: e for e in list_models()}
    want = {
        "whisper-large-v3": "fp16",
        "whisper-large-v3-turbo": "fp16",
        "whisper-tiny": "fp32",
        "whisper-large": "fp32",
        "distil-whisper-large-v3.5": "fp32",
        "distil-whisper-large-v2": "fp16",
        "distil-whisper-large-v3": "fp16",
        "qwen2-audio-7b-instruct": "bf16",
        "qwen2.5-omni-7b": "bf16",
        "step-audio-2-mini": "bf16",
        "kimi-audio-7b-instruct": "bf16",
        "ming-3.0-asr-tiny": "bf16",
    }
    for mid, fmt in want.items():
        m = get_model(mid)
        au = m.vision.audio
        assert au is not None and au.kind == "whisper", mid
        assert m.param_check()["rel_err"] == 0.0, mid
        assert au.w_fmt == "bf16" and au.w_bits == 16.0, mid
        assert au.vec_fmt == fmt, (mid, au.vec_fmt)
        assert ents[mid]["id"] == mid
    rel = load_release("distil-whisper/distil-large-v2")
    assert rel["torch_dtype"] == "float32"
    assert rel["config"]["torch_dtype"] == "float32"
    assert get_model("distil-whisper-large-v2").vision.audio.vec_fmt == "fp16"
    text = " ".join(get_model(W3).notes)
    for needle in (
        "erf", "gelu_pytorch_tanh", "quick_gelu", "0.461", "torch_dtype",
        "vision 角色的 fp16", "历史的 bf16", "fp32", "Dropout", "F.normalize",
        "nn.ReLU", "nn.SiLU", "通用 GEMM 尾项不减", "没有 mlp_mul",
    ):
        assert needle in text, needle
    assert "GELU 激活按 SiLU 的逐元素开销计「近似」" not in text
    assert "解码器仍把 GELU 按 SiLU" not in text


def test_whisper_large_v3_bills_erf_gelu_and_layernorm():
    au = get_model(W3).vision.audio
    ops = audio_ops(au, 1.0, 1, "bf16")
    by = _by(ops)
    names = [op.name for op in ops]
    head = ["aud.conv1", "aud.conv1.gelu", "aud.conv2", "aud.conv2.gelu"] + _layer(0)
    assert names[:len(head)] == head
    assert names[names.index("aud.31.attn_ln"):] == _layer(31) + ["aud.ln"]
    for missing in ("aud.proj.relu", "aud.proj.silu", "aud.proj.gelu", "aud.0.in_ln", "aud.pool", "aud.dropout"):
        assert missing not in by, missing
    assert sum(1 for name in by if name.endswith(".gelu")) == 2 + LAYERS
    assert sum(1 for name in by if name.endswith("_ln") or name == "aud.ln") == 2 * LAYERS + 1

    c1 = by["aud.conv1.gelu"]
    assert c1.vec == MEL * D * 5 and c1.sfu == MEL * D and c1.a_fmt == "fp16"
    assert c1.w_params == 0 and c1.flops == 0
    assert by["aud.conv2.gelu"].vec == T * D * 5 and by["aud.conv2.gelu"].sfu == T * D
    assert by["aud.conv2.gelu"].a_fmt == "fp16"
    ln = by["aud.0.attn_ln"]
    assert ln.vec == T * D * 4 and ln.sfu == T and ln.a_fmt == "fp32"
    assert by["aud.0.mlp_ln"].vec == T * D * 4 and by["aud.0.mlp_ln"].sfu == T
    assert by["aud.ln"].vec == T * D * 4 and by["aud.ln"].sfu == T and by["aud.ln"].a_fmt == "fp32"
    gelu = by["aud.0.gelu"]
    assert gelu.vec == T * FF * 5 and gelu.sfu == T * FF and gelu.a_fmt == "fp16"
    assert by["aud.0.fc1"].vec == T * FF * 3 and by["aud.0.fc1"].m == T and by["aud.0.fc1"].n == FF
    assert by["aud.0.fc1"].a_fmt == "bf16" and by["aud.0.fc1"].w_fmt == "bf16" and by["aud.0.fc1"].w_bits == 16.0
    assert by["aud.conv1"].m == MEL and by["aud.conv1"].a_fmt == "bf16" and by["aud.conv1"].w_fmt == "bf16"
    sm = by["aud.0.softmax"]
    assert sm.sfu == 0 and sm.vec == HEADS * T * T * 5
    assert by["aud.0.qk"].m == T and by["aud.0.qk"].count == HEADS

    wide = _by(audio_ops(au, 1.0, 2, "bf16"))
    long = _by(audio_ops(au, 45.0, 1, "bf16"))
    for name in ("aud.conv1.gelu", "aud.0.gelu", "aud.0.attn_ln", "aud.ln"):
        assert wide[name].vec == 2 * by[name].vec, name
        assert long[name].vec == 2 * by[name].vec, name
    assert wide["aud.0.attn_ln"].sfu == 2 * T and long["aud.ln"].sfu == 2 * T
    assert wide["aud.0.qk"].m == T and wide["aud.0.qk"].count == 2 * HEADS
    assert long["aud.0.qk"].m == T and long["aud.0.qk"].count == 2 * HEADS
    assert wide["aud.0.fc1"].vec == 2 * T * FF * 3 and wide["aud.0.softmax"].sfu == 0

    forced = _by(audio_ops(au, 1.0, 1, "fp8"))
    assert forced["aud.0.fc1"].a_fmt == "fp8" and forced["aud.conv1"].a_fmt == "fp8"
    assert forced["aud.0.fc1"].w_fmt == "bf16" and forced["aud.0.qkv"].w_fmt == "bf16"
    assert forced["aud.0.fc1"].vec == by["aud.0.fc1"].vec
    for name in ("aud.conv1.gelu", "aud.conv2.gelu", "aud.0.gelu", "aud.31.gelu"):
        assert forced[name].a_fmt == "fp16", name
    for name in ("aud.0.attn_ln", "aud.0.mlp_ln", "aud.31.mlp_ln", "aud.ln"):
        assert forced[name].a_fmt == "fp32", name

    tiny = get_model("whisper-tiny").vision.audio
    tiny_ops = _by(audio_ops(tiny, 1.0, 1, "bf16"))
    assert tiny.vec_fmt == "fp32"
    assert tiny_ops["aud.0.gelu"].a_fmt == "fp32" and tiny_ops["aud.conv1.gelu"].a_fmt == "fp32"
    assert tiny_ops["aud.0.fc1"].w_fmt == "bf16" and tiny_ops["aud.0.attn_ln"].a_fmt == "fp32"


def test_qwen2_audio_and_omni_layernorm_follows_the_pool():
    q = get_model("qwen2-audio-7b-instruct").vision.audio
    ops = audio_ops(q, 10.0, 1)
    by = _by(ops)
    names = [op.name for op in ops]
    assert names[names.index("aud.31.fc2"):] == ["aud.31.fc2", "aud.pool", "aud.ln", "aud.proj.linear0"]
    assert by["aud.0.qkv"].m == T and by["aud.0.gelu"].vec == T * FF * 5
    assert by["aud.0.gelu"].a_fmt == "bf16" and by["aud.conv1.gelu"].vec == MEL * D * 5
    assert by["aud.ln"].vec == 750 * D * 4 and by["aud.ln"].sfu == 750
    assert by["aud.ln"].sfu != T and by["aud.ln"].a_fmt == "fp32"
    assert by["aud.proj.linear0"].m == 750 and by["aud.proj.linear0"].n == 4096
    assert q.tokens(10.0) == 250
    for missing in ("aud.proj.gelu", "aud.proj.relu", "aud.proj.silu"):
        assert missing not in by

    om = get_model("qwen2.5-omni-7b").vision.audio
    ob = _by(audio_ops(om, 10.0, 1))
    assert ob["aud.0.qkv"].m == 500 and ob["aud.0.qk"].m == 100 and ob["aud.0.qk"].count == 5 * HEADS
    assert ob["aud.conv1.gelu"].vec == 5 * 200 * D * 5 and ob["aud.conv1.gelu"].a_fmt == "bf16"
    assert ob["aud.0.gelu"].vec == 500 * FF * 5 and ob["aud.0.gelu"].sfu == 500 * FF
    assert ob["aud.ln"].vec == 250 * D * 4 and ob["aud.ln"].sfu == 250 and ob["aud.ln"].a_fmt == "fp32"
    assert ob["aud.proj.linear0"].m == 250 and ob["aud.proj.linear0"].n == 3584
    assert not any("mask" in name for name in ob)
    short = _by(audio_ops(om, 0.5, 1))
    assert om.mel_frames(0.5) == 50 and om.enc_positions(0.5) == 25 and om.tokens(0.5) == 12
    assert short["aud.conv1.gelu"].vec == 50 * D * 5
    assert short["aud.0.gelu"].vec == 25 * FF * 5
    assert short["aud.ln"].sfu == 12 and short["aud.proj.linear0"].m == 12


def test_step_kimi_and_ming_tails():
    step = get_model("step-audio-2-mini").vision.audio
    ops = audio_ops(step, 10.0, 1)
    by = _by(ops)
    names = [op.name for op in ops]
    assert names[names.index("aud.pool"):] == [
        "aud.pool", "aud.ln", "aud.proj.conv0", "aud.proj.gelu",
        "aud.proj.linear0", "aud.proj.relu", "aud.proj.linear1",
    ]
    assert step.enc_positions(10.0) == 501 and step.tokens(10.0) == 125
    assert by["aud.ln"].vec == 250 * D * 4 and by["aud.ln"].sfu == 250 and by["aud.ln"].a_fmt == "fp32"
    assert by["aud.proj.gelu"].vec == 125 * D * 5 and by["aud.proj.gelu"].a_fmt == "bf16"
    relu = by["aud.proj.relu"]
    assert relu.vec == 125 * 2048 and relu.vec != 125 * 2048 * 2
    assert relu.sfu == 0 and relu.a_fmt == "bf16"
    assert [by["aud.proj.linear0"].n, by["aud.proj.linear1"].n] == [2048, 3584]
    assert by["aud.proj.conv0"].m == 125 and by["aud.0.fc1"].vec == 501 * FF * 3
    short = _by(audio_ops(step, 0.5, 1))
    assert step.enc_positions(0.5) == 26 and step.tokens(0.5) == 7
    assert short["aud.ln"].sfu == 13 and short["aud.proj.relu"].vec == 7 * 2048
    assert short["aud.proj.gelu"].vec == 7 * D * 5

    kimi = get_model("kimi-audio-7b-instruct").vision.audio
    kb = _by(audio_ops(kimi, 1.0, 1))
    kn = [op.name for op in audio_ops(kimi, 1.0, 1)]
    assert kn[kn.index("aud.31.fc2"):] == [
        "aud.31.fc2", "aud.ln", "aud.proj.linear0", "aud.proj.silu",
        "aud.proj.linear1", "aud.proj.ln",
    ]
    assert kimi.tokens(1.0) == 13 and kimi.enc_positions(1.0) == T
    assert kb["aud.ln"].vec == T * D * 4 and kb["aud.ln"].sfu == T and kb["aud.ln"].a_fmt == "fp32"
    assert (kb["aud.proj.linear0"].m, kb["aud.proj.linear0"].k, kb["aud.proj.linear0"].n) == (375, 5120, 3584)
    silu = kb["aud.proj.silu"]
    assert silu.vec == 375 * 3584 * 4 and silu.sfu == 375 * 3584 * 2 and silu.a_fmt == "bf16"
    assert kb["aud.proj.ln"].vec == 375 * 3584 * 4 and kb["aud.proj.ln"].sfu == 375
    assert kb["aud.proj.ln"].a_fmt == "fp32"
    assert "aud.proj.relu" not in kb and not any("dropout" in name for name in kb)
    assert kb["aud.0.gelu"].a_fmt == "bf16" and kb["aud.0.fc1"].w_fmt == "bf16"

    ming = get_model("ming-3.0-asr-tiny").vision.audio
    mb = _by(audio_ops(ming, 1.0, 1))
    mn = [op.name for op in audio_ops(ming, 1.0, 1)]
    assert mn[mn.index("aud.31.fc2"):] == [
        "aud.31.fc2", "aud.ln", "aud.proj.conv0", "aud.proj.gelu", "aud.proj.linear",
    ]
    assert ming.enc_positions(1.0) == 50 and ming.tokens(1.0) == 25
    assert mb["aud.ln"].vec == 50 * D * 4 and mb["aud.ln"].sfu == 50 and mb["aud.ln"].a_fmt == "fp32"
    assert mb["aud.proj.gelu"].vec == 25 * 1536 * 5 and mb["aud.proj.gelu"].a_fmt == "bf16"
    assert (mb["aud.proj.linear"].m, mb["aud.proj.linear"].n) == (25, 1536)
    assert "aud.proj.relu" not in mb and "aud.proj.silu" not in mb
    two = _by(audio_ops(ming, 1.0, 2))
    assert two["aud.proj.gelu"].vec == 2 * mb["aud.proj.gelu"].vec
    assert two["aud.ln"].sfu == 2 * 50 and two["aud.0.qk"].m == 50 and two["aud.0.qk"].count == 2 * HEADS


def test_other_towers_do_not_gain_the_whisper_names():
    # d1 already bills its own attn_ln. The whisper names are the conv GELUs.
    for mid in ("fun-asr-nano-2512", "gemma-4-e2b-it", "d1-omni-600m", "omniasr-ctc-300m-v2", "qwen3-asr-1.7b"):
        other = get_model(mid).vision.audio
        assert other is not None and other.kind != "whisper", mid
        names = [op.name for op in audio_ops(other, 1.0, 1)]
        assert "aud.conv1.gelu" not in names and "aud.conv2.gelu" not in names, mid
        # 0.416 AuT and 0.417 OmniASR bill their own pre-attention LayerNorm under the same name.
        if mid not in ("d1-omni-600m", "qwen3-asr-1.7b", "omniasr-ctc-300m-v2"):
            assert "aud.0.attn_ln" not in names, mid


def test_whisper_family_refuses_a_non_erf_activation():
    w = json.loads((ROOT / "accel_dse/data/releases/openai__whisper-large-v3.json").read_text())["config"]
    bad = dict(w)
    bad["activation_function"] = "gelu_pytorch_tanh"
    _raises(bad, "whisper activation_function is not gelu")
    missing = dict(w)
    missing.pop("activation_function")
    _raises(missing, "whisper activation_function is not gelu")

    q = json.loads((ROOT / "accel_dse/data/releases/Qwen__Qwen2-Audio-7B-Instruct.json").read_text())["config"]
    qbad = json.loads(json.dumps(q))
    qbad["audio_config"]["activation_function"] = "quick_gelu"
    _raises(qbad, "qwen2_audio activation_function is not gelu")

    om = json.loads((ROOT / "accel_dse/data/releases/Qwen__Qwen2.5-Omni-7B.json").read_text())["config"]
    obad = json.loads(json.dumps(om))
    obad["thinker_config"]["audio_config"]["activation_function"] = "gelu_pytorch_tanh"
    _raises(obad, "qwen2_5_omni activation_function is not gelu")

    step = json.loads((ROOT / "accel_dse/data/releases/stepfun-ai__Step-Audio-2-mini.json").read_text())["config"]
    assert audio_spec(step).kind == "whisper"
    sbad = json.loads(json.dumps(step))
    sbad["audio_encoder_config"]["activation_function"] = "gelu_pytorch_tanh"
    _raises(sbad, "step_audio activation_function is not gelu")

    ming = json.loads((ROOT / "accel_dse/data/releases/inclusionAI__Ming-3.0-asr-tiny.json").read_text())["config"]
    assert audio_spec(ming).kind == "whisper"
    mbad = json.loads(json.dumps(ming))
    mbad["audio_config"]["activation_function"] = "gelu_pytorch_tanh"
    _raises(mbad, "ming activation_function is not gelu")

    kimi = json.loads((ROOT / "accel_dse/data/releases/moonshotai__Kimi-Audio-7B-Instruct.json").read_text())
    kw = dict(kimi["kimi_audio"]["whisper"])
    cfg = {"_kimi_whisper": kw, "hidden_size": 3584, "kimia_adaptor_input_dim": 5120}
    assert audio_spec(cfg).kind == "whisper" and audio_spec(cfg).stack == 4
    kbad = {"_kimi_whisper": {**kw, "activation_function": "quick_gelu"}, "hidden_size": 3584}
    _raises(kbad, "kimi activation_function is not gelu")
