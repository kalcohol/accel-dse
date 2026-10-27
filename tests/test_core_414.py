"""0.414: Fun-ASR-Nano bills the folded ReLU, LayerNorm, and FSMN mask multiply.

FunAsrNanoEncoderLayer runs input LayerNorm, attention, the FSMN mask multiply,
post LayerNorm, and ACT2FN["relu"]. The output LayerNorm exists only on the two
published layers. The projector ReLU sits between its two linears. Adaptor
layers keep the LayerNorms and the ReLU and have no FSMN. GEMM shapes stay.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.vision import _sanm_spec, audio_ops

FN = "fun-asr-nano-2512"
D = 512
FF = 2048
MEL = 560
AD = 1024
ADF = 256
HEADS = 4
KERNEL = 11
FINAL = (49, 69)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _enc(li: int, final: bool) -> list[str]:
    p = f"aud.{li}."
    names = [
        p + "in_ln", p + "qkv", p + "qk", p + "pv", p + "softmax", p + "o",
        p + "mask", p + "fsmn", p + "post_ln", p + "fc1", p + "relu", p + "fc2",
    ]
    if final:
        names.append(p + "out_ln")
    return names


def _ad(j: int) -> list[str]:
    p = f"aud.ad{j}."
    return [
        p + "in_ln", p + "qkv", p + "qk", p + "pv", p + "softmax", p + "o",
        p + "post_ln", p + "fc1", p + "relu", p + "fc2",
    ]


def _release_configs() -> tuple[dict, dict]:
    path = Path(__file__).resolve().parents[1] / "accel_dse/data/releases/FunAudioLLM__Fun-ASR-Nano-2512-hf.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
    return dict(cfg["audio_config"]), dict(cfg["adaptor_config"])


def _raises(ac: dict, ad: dict, needle: str) -> None:
    try:
        _sanm_spec(ac, ad)
    except ValueError as exc:
        assert needle in str(exc)
        return
    raise AssertionError(needle)


def test_fun_asr_nano_catalog_keeps_the_tower():
    m = get_model(FN)
    au = m.vision.audio
    assert m.param_check()["rel_err"] == 0.0
    assert au is not None and au.kind == "sanm" and au.sanm_final == FINAL
    assert au.w_fmt == "bf16" and au.heads == HEADS and au.depths == (70,)
    text = " ".join(m.notes)
    for needle in (
        "Fun-ASR-Nano", "mask", "ACT2FN['relu']", "Identity", "第 49 层", "第 69 层",
        "第 0 层", "max(x, 0)", "relu6", "projector_hidden_size", "adaptor",
        "intermediate", "fp32", "bf16", "activation_dropout", "sqrt(d)", "通用 GEMM 尾项不减",
    ):
        assert needle in text, needle


def test_fun_asr_nano_bills_relu_layernorm_and_mask():
    m = get_model(FN)
    au = m.vision.audio
    ops = audio_ops(au, 1.0, 1, "bf16")
    by = _by(ops)
    names = [op.name for op in ops]
    assert names[:len(_enc(0, False))] == _enc(0, False)
    assert names[names.index("aud.49.in_ln"):names.index("aud.50.in_ln")] == _enc(49, True)
    assert names[names.index("aud.50.in_ln"):names.index("aud.51.in_ln")] == _enc(50, False)
    assert names[names.index("aud.69.in_ln"):] == _enc(69, True) + [
        "aud.proj.linear0", "aud.proj.relu", "aud.proj.linear1",
    ] + _ad(0) + _ad(1)
    assert [name for name in names if name.endswith(".out_ln")] == ["aud.49.out_ln", "aud.69.out_ln"]
    for missing in ("aud.0.out_ln", "aud.48.out_ln", "aud.50.out_ln", "aud.ad0.mask", "aud.ad0.fsmn", "aud.ad0.out_ln",
                    "aud.ad1.mask", "aud.ad1.out_ln"):
        assert missing not in by, missing

    t = 17
    ln0 = by["aud.0.in_ln"]
    assert ln0.vec == t * MEL * 4 and ln0.sfu == t and ln0.a_fmt == "fp32"
    assert ln0.w_params == 0 and ln0.flops == 0
    assert by["aud.1.in_ln"].vec == t * D * 4 and by["aud.1.in_ln"].sfu == t
    assert by["aud.1.in_ln"].a_fmt == "fp32"
    mask = by["aud.0.mask"]
    assert mask.kind == "vector" and mask.vec == t * D and mask.vec != t * MEL
    assert mask.sfu == 0 and mask.a_fmt == "bf16" and mask.w_params == 0 and mask.flops == 0
    post = by["aud.0.post_ln"]
    assert post.vec == t * D * 4 and post.sfu == t and post.a_fmt == "fp32" and post.w_params == 0
    relu = by["aud.0.relu"]
    assert relu.vec == t * FF and relu.vec != t * FF * 2
    assert relu.sfu == 0 and relu.a_fmt == "bf16" and relu.w_params == 0 and relu.flops == 0
    assert by["aud.0.fc1"].vec == t * FF * 3 and by["aud.0.fc1"].a_fmt == "bf16"
    assert by["aud.0.fc1"].m == t and by["aud.0.fc1"].n == FF
    assert (by["aud.0.qkv"].k, by["aud.0.qkv"].n) == (MEL, 3 * D)
    assert (by["aud.1.qkv"].k, by["aud.1.qkv"].n) == (D, 3 * D)
    fsmn = by["aud.0.fsmn"]
    assert fsmn.vec == t * D * 2 * KERNEL and fsmn.w_params == D * KERNEL
    for li in FINAL:
        out = by[f"aud.{li}.out_ln"]
        assert out.vec == t * D * 4 and out.sfu == t and out.a_fmt == "fp32" and out.layer == li
    sm = by["aud.0.softmax"]
    assert sm.sfu == 0 and sm.vec == HEADS * t * t * 5
    prelu = by["aud.proj.relu"]
    assert names.index("aud.proj.linear0") + 1 == names.index("aud.proj.relu")
    assert names.index("aud.proj.relu") + 1 == names.index("aud.proj.linear1")
    assert prelu.vec == t * FF and prelu.sfu == 0 and prelu.a_fmt == "bf16" and prelu.layer == 70
    assert by["aud.proj.linear0"].n == FF and by["aud.proj.linear0"].vec == t * FF * 3
    assert (by["aud.proj.linear1"].k, by["aud.proj.linear1"].n) == (FF, AD)
    for j in (0, 1):
        assert by[f"aud.ad{j}.in_ln"].vec == t * AD * 4 and by[f"aud.ad{j}.in_ln"].sfu == t
        assert by[f"aud.ad{j}.in_ln"].a_fmt == "fp32"
        assert by[f"aud.ad{j}.post_ln"].vec == t * AD * 4
        arelu = by[f"aud.ad{j}.relu"]
        assert arelu.vec == t * ADF and arelu.vec != t * ADF * 2
        assert arelu.sfu == 0 and arelu.a_fmt == "bf16" and arelu.layer == 71 + j
        assert by[f"aud.ad{j}.fc1"].vec == t * ADF * 3 and by[f"aud.ad{j}.fc1"].n == ADF
    assert sum(1 for name in by if name.endswith(".relu")) == 70 + 1 + 2
    assert sum(1 for name in by if name.endswith(".mask")) == 70
    assert sum(1 for name in by if name.endswith(".fsmn")) == 70
    assert sum(1 for name in by if name.endswith(".in_ln")) == 72
    assert not any("gelu" in name or "silu" in name or "softcap" in name for name in by)
    w = sum(op.w_params for op in ops)
    ln = 2 * MEL + 69 * 2 * D + 70 * 2 * D + 4 * D + 2 * 2 * 2 * AD
    assert w == au.params - ln

    wide = _by(audio_ops(au, 1.0, 2, "bf16"))
    for name in ("aud.0.relu", "aud.0.mask", "aud.0.in_ln", "aud.0.post_ln", "aud.49.out_ln", "aud.proj.relu", "aud.ad0.relu"):
        assert wide[name].vec == 2 * by[name].vec, name
    assert wide["aud.0.in_ln"].sfu == 2 * ln0.sfu
    assert wide["aud.49.out_ln"].sfu == 2 * t
    assert wide["aud.ad0.in_ln"].sfu == 2 * t
    assert wide["aud.0.qk"].m == t and wide["aud.0.qk"].count == 2 * HEADS
    assert wide["aud.0.fc1"].m == 2 * t and wide["aud.0.fc1"].vec == 2 * t * FF * 3
    assert wide["aud.0.softmax"].vec == 2 * HEADS * t * t * 5 and wide["aud.0.softmax"].sfu == 0

    long = _by(audio_ops(au, 10.0, 1, "bf16"))
    assert au.tokens(10.0) == 167
    assert long["aud.0.relu"].vec == 167 * FF and long["aud.0.mask"].vec == 167 * D
    assert long["aud.0.in_ln"].sfu == 167 and long["aud.0.fc1"].vec == 167 * FF * 3

    short = _by(audio_ops(au, 0.5, 1, "bf16"))
    assert au.tokens(0.5) == 8
    n = round(0.5 * 16000)
    fb = 1 + (n - 400) // 160
    assert -(-fb // 6) == 8
    assert short["aud.0.relu"].vec == 8 * FF and short["aud.0.in_ln"].vec == 8 * MEL * 4
    assert short["aud.proj.relu"].vec == 8 * FF and short["aud.ad1.relu"].vec == 8 * ADF

    forced = _by(audio_ops(au, 1.0, 1, "fp8"))
    assert forced["aud.0.fc1"].a_fmt == "fp8" and forced["aud.0.qkv"].a_fmt == "fp8"
    assert forced["aud.0.fc1"].vec == by["aud.0.fc1"].vec
    assert forced["aud.proj.linear0"].a_fmt == "fp8" and forced["aud.ad0.fc1"].a_fmt == "fp8"
    for name in ("aud.0.relu", "aud.0.mask", "aud.69.relu", "aud.proj.relu", "aud.ad0.relu", "aud.ad1.relu"):
        assert forced[name].a_fmt == "bf16", name
    for name in ("aud.0.in_ln", "aud.0.post_ln", "aud.49.out_ln", "aud.69.out_ln", "aud.ad0.in_ln", "aud.ad1.post_ln"):
        assert forced[name].a_fmt == "fp32", name

    for mid in ("whisper-large-v3", "omniasr-ctc-300m-v2", "d1-omni-600m", "gemma-4-e2b-it"):
        other = get_model(mid).vision.audio
        assert other is not None and other.kind != "sanm" and other.sanm_final == (), mid
        onames = [op.name for op in audio_ops(other, 1.0, 1)]
        assert "aud.0.in_ln" not in onames, mid
        assert "aud.proj.relu" not in onames, mid


def test_fun_asr_nano_refuses_a_different_act_or_final_norm():
    ac, ad = _release_configs()
    bad_ac = dict(ac)
    bad_ac["hidden_act"] = "silu"
    _raises(bad_ac, ad, "fun_asr_nano hidden_act is not relu")
    bad_ad = dict(ad)
    bad_ad["hidden_act"] = "gelu"
    _raises(ac, bad_ad, "fun_asr_nano hidden_act is not relu")
    bad_proj = dict(ad)
    bad_proj["projector_hidden_act"] = "gelu_pytorch_tanh"
    _raises(ac, bad_proj, "fun_asr_nano hidden_act is not relu")
    missing = dict(ac)
    del missing["num_timestamp_prediction_layers"]
    _raises(missing, ad, "fun_asr_nano timestamp layers are missing")
    flagged = dict(ac)
    flagged["num_timestamp_prediction_layers"] = True
    _raises(flagged, ad, "fun_asr_nano timestamp layers are missing")
    for n_tp in (0, 70):
        shifted = dict(ac)
        shifted["num_timestamp_prediction_layers"] = n_tp
        _raises(shifted, ad, "fun_asr_nano final LayerNorm layers are not two encoder layers")
    au = get_model(FN).vision.audio
    try:
        audio_ops(replace(au, sanm_final=()), 1.0, 1)
    except ValueError as exc:
        assert "not two encoder layers" in str(exc)
    else:
        raise AssertionError("empty sanm_final")
    try:
        audio_ops(replace(au, lin=((D, FF),)), 1.0, 1)
    except ValueError as exc:
        assert "not two linears" in str(exc)
    else:
        raise AssertionError("one projector linear")
