"""0.432: Muse Spark audio AutoEncoder RMSNorm and nn.GELU() (erf).

Each patchifier Linear and each depthwise conv is followed by RMSNorm.
nn.GELU() between the block linears is the erf form. One RMSNorm sits before
o_proj and dual_norm sits after it. Vector flops stay 0.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.model import from_release, load_release
from accel_dse.core.vision import audio_ops, vision_spec

IDS = (
    "muse-spark-open",
    "muse-spark-open-fp8",
    "muse-spark-open-mxfp4",
    "muse-spark-open-nvfp4",
    "muse-spark-open-nvfp4-dspark",
    "muse-spark-open-nvfp4-dflash2",
)
HF = "meta-models/Muse-Spark-Open-816B-A42B"
PATCHES = (16, 6, 5, 4)
DIMS = (256, 512, 1024, 2048)
DEPTHS = (2, 2, 2, 2)
OUT = 8192
KERNEL = 7
MLP = 4.0
PARAMS = 117_840_384
GEMM_60 = 593_707_008_000
ROLE_PARAMS = 2_045_977_600
ROLE_BYTES = 4_091_955_200


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _rms(op, rows: int, width: int) -> None:
    assert op.kind == "vector" and op.layer == 0
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _gelu(op, rows: int, width: int, af: str = "bf16") -> None:
    assert op.kind == "vector" and op.layer == 0
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _rows(seconds: float, clips: int = 1) -> tuple[int, list[tuple[int, int, int]]]:
    tokens = math.ceil(round(seconds * 24000) / 1920)
    t = tokens * 1920 * clips
    stages = []
    for pt, d in zip(PATCHES, DIMS):
        t //= pt
        stages.append((t, d, int(d * MLP)))
    return tokens, stages


def test_muse_audio_bills_rms_and_erf_gelu():
    models = [get_model(mid) for mid in IDS]
    base = models[0]
    vs = base.vision
    assert vs is not None and vs.audio is not None
    for m in models[1:]:
        assert m.vision == vs, m.id
    a = vs.audio
    assert a.kind == "muse" and a.vec_fmt == "bf16" and vs.vec_fmt == "bf16"
    assert a.patches == PATCHES and a.dims == DIMS and a.depths == DEPTHS
    assert a.kernel == KERNEL and a.mlp == MLP and a.out == OUT
    assert a.params == PARAMS and a.samples_per_token == 1920
    assert a.tokens(60) == 750 and a.tokens(1) == 13 and a.tokens(3) == 38
    rel = load_release(base.hf_id)
    ac = rel["config"]["audio_config"]
    assert ac["hidden_act"] == "gelu" and ac["mlp_ratio"] == 4.0
    assert ac["rms_norm_eps"] == 1e-6 and ac["out_hidden_size"] == OUT
    role = rel["roles"]["vision"]
    assert list(role) == ["bf16"]
    assert role["bf16"]["params"] == ROLE_PARAMS and role["bf16"]["bytes"] == ROLE_BYTES
    pc = base.param_check()
    assert pc["ours"] == 819989536256 and pc["release"] == 819990552064
    assert pc["rel_err"] != 0.0
    note = " ".join(base.notes)
    assert "Muse Spark NaViT" in note and "ACT2FN" in note and "nn.LayerNorm" in note
    assert "vision_projection" in note and "perception RMSNorm" in note
    assert "act(fc2(act(fc1(x))))" in note
    assert "nn.GELU()" in note and "dual_norm" in note and "tok_embeddings_norm" in note
    assert "patchify" in note and "fp32" in note and a.vec_fmt in note

    tokens, stages = _rows(60)
    assert tokens == 750 and stages[0] == (90000, 256, 1024) and stages[-1] == (750, 2048, 8192)
    ops = audio_ops(a, 60, 1)
    by = _by(ops)
    _rms(by["aud.0.patch_rms"], 90000, 256)
    _rms(by["aud.0.0.rms"], 90000, 256)
    _rms(by["aud.0.1.rms"], 90000, 256)
    _gelu(by["aud.0.0.gelu"], 90000, 1024)
    _gelu(by["aud.0.1.gelu"], 90000, 1024)
    _rms(by["aud.1.patch_rms"], 15000, 512)
    _gelu(by["aud.1.0.gelu"], 15000, 2048)
    _rms(by["aud.2.patch_rms"], 3000, 1024)
    _gelu(by["aud.2.1.gelu"], 3000, 4096)
    _rms(by["aud.3.patch_rms"], 750, 2048)
    _rms(by["aud.3.1.rms"], 750, 2048)
    _gelu(by["aud.3.0.gelu"], 750, 8192)
    _rms(by["aud.norm"], 750, 2048)
    _rms(by["aud.dual_rms"], 750, OUT)
    assert by["aud.0.patch"].m == 90000 and by["aud.0.patch"].k == 16 and by["aud.0.patch"].n == 256
    assert by["aud.0.patch"].vec == 90000 * 256 * 3
    assert by["aud.0.patch"].w_fmt == "bf16" and by["aud.0.patch"].w_bits == 16.0
    assert by["aud.0.patch"].a_fmt == "bf16"
    assert by["aud.0.0.conv"].vec == 90000 * 256 * 2 * KERNEL
    assert by["aud.0.0.fc1"].n == 1024 and by["aud.0.0.fc1"].vec == 90000 * 1024 * 3
    assert by["aud.0.0.fc1"].w_fmt == "bf16" and by["aud.0.0.fc1"].w_bits == 16.0
    assert by["aud.3.1.fc2"].m == 750 and by["aud.3.1.fc2"].k == 8192 and by["aud.3.1.fc2"].n == 2048
    assert by["aud.3.1.fc2"].vec == 750 * 2048 * 3
    assert (by["aud.o_proj"].m, by["aud.o_proj"].k, by["aud.o_proj"].n) == (750, 2048, OUT)
    assert by["aud.o_proj"].vec == 750 * OUT * 3 and by["aud.o_proj"].w_fmt == "bf16"
    names = [op.name for op in ops]
    assert names.index("aud.0.patch") < names.index("aud.0.patch_rms") < names.index("aud.0.0.conv")
    assert names.index("aud.0.0.conv") < names.index("aud.0.0.rms") < names.index("aud.0.0.fc1")
    assert names.index("aud.0.0.fc1") < names.index("aud.0.0.gelu") < names.index("aud.0.0.fc2")
    assert names.index("aud.0.0.fc2") < names.index("aud.0.1.conv") < names.index("aud.3.1.fc2")
    assert names.index("aud.3.1.fc2") < names.index("aud.norm") < names.index("aud.o_proj")
    assert names.index("aud.o_proj") < names.index("aud.dual_rms")
    rms = [op for op in ops if op.name.endswith("rms") or op.name == "aud.norm"]
    gels = [op for op in ops if op.name.endswith(".gelu")]
    assert len(rms) == 14 and len(gels) == 8
    assert all(op.flops == 0 and op.w_params == 0 for op in rms + gels)
    for absent in ("aud.0.0.silu", "aud.0.0.quick_gelu", "aud.0.0.relu", "aud.tok_embeddings_norm",
                   "vis.0.gelu", "aud.0.0.ls1"):
        assert absent not in by, absent
    gemm = sum(op.flops for op in ops if op.kind == "gemm")
    assert gemm == GEMM_60
    assert sum(op.flops for op in ops if op.kind == "vector") == 0

    one_tok, one = _rows(1)
    assert one_tok == 13 and one[0][0] == 1560 and one[-1][0] == 13
    ob = _by(audio_ops(a, 1, 1))
    _rms(ob["aud.0.patch_rms"], 1560, 256)
    _gelu(ob["aud.0.0.gelu"], 1560, 1024)
    _rms(ob["aud.3.1.rms"], 13, 2048)
    _gelu(ob["aud.3.0.gelu"], 13, 8192)
    _rms(ob["aud.norm"], 13, 2048)
    _rms(ob["aud.dual_rms"], 13, OUT)
    assert ob["aud.o_proj"].m == 13 and ob["aud.0.patch"].vec == 1560 * 256 * 3

    three_tok, three = _rows(3)
    assert three_tok == 38 and three[0][0] == 4560 and three[-1][0] == 38
    tb = _by(audio_ops(a, 3, 1))
    _rms(tb["aud.0.patch_rms"], 4560, 256)
    _gelu(tb["aud.2.1.gelu"], 152, 4096)
    _rms(tb["aud.norm"], 38, 2048)
    _rms(tb["aud.dual_rms"], 38, OUT)

    two = _by(audio_ops(a, 60, 2))
    _rms(two["aud.0.patch_rms"], 180000, 256)
    _gelu(two["aud.0.0.gelu"], 180000, 1024)
    _rms(two["aud.1.0.rms"], 30000, 512)
    _gelu(two["aud.3.1.gelu"], 1500, 8192)
    _rms(two["aud.norm"], 1500, 2048)
    _rms(two["aud.dual_rms"], 1500, OUT)
    assert two["aud.0.patch"].m == 180000 and two["aud.o_proj"].m == 1500
    assert two["aud.0.patch"].vec == 2 * by["aud.0.patch"].vec
    assert two["aud.0.0.fc1"].vec == 2 * by["aud.0.0.fc1"].vec

    forced = _by(audio_ops(a, 60, 1, "fp8"))
    assert forced["aud.0.patch"].a_fmt == "fp8" and forced["aud.0.0.fc1"].a_fmt == "fp8"
    assert forced["aud.o_proj"].a_fmt == "fp8"
    assert forced["aud.0.patch"].w_fmt == "bf16" and forced["aud.0.0.fc2"].w_bits == 16.0
    assert forced["aud.0.patch"].vec == by["aud.0.patch"].vec
    assert forced["aud.0.0.fc1"].vec == by["aud.0.0.fc1"].vec
    assert forced["aud.o_proj"].vec == by["aud.o_proj"].vec
    assert forced["aud.0.0.gelu"].a_fmt == "bf16" and forced["aud.3.0.gelu"].a_fmt == "bf16"
    assert forced["aud.0.0.gelu"].vec == by["aud.0.0.gelu"].vec
    assert forced["aud.0.patch_rms"].a_fmt == "fp32" and forced["aud.norm"].a_fmt == "fp32"
    assert forced["aud.dual_rms"].a_fmt == "fp32" and forced["aud.3.1.rms"].a_fmt == "fp32"


def test_muse_audio_refuses_other_activations_and_a_missing_dtype():
    rel = load_release(HF)
    silu = copy.deepcopy(rel["config"])
    silu["audio_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF, silu), "muse audio hidden_act is not gelu")
    omitted = copy.deepcopy(rel["config"])
    omitted["audio_config"].pop("hidden_act")
    built = vision_spec(HF, omitted)
    assert built is not None and built.audio is not None and built.audio.kind == "muse"
    assert built.audio.vec_fmt == ""
    quiet = audio_ops(built.audio, 60, 1)
    assert any(op.name == "aud.o_proj" for op in quiet)
    assert not any("gelu" in op.name or "rms" in op.name for op in quiet)

    a = get_model("muse-spark-open").vision.audio
    _raises(lambda: audio_ops(replace(a, vec_fmt="fp8"), 1, 1),
            "muse audio activation dtype is not a vision float")
    _raises(lambda: audio_ops(replace(a, mlp=4.1), 1, 1),
            "muse audio mlp width is not an integer")
    empty = audio_ops(replace(a, vec_fmt=""), 60, 1)
    assert any(op.name == "aud.0.patch" for op in empty)
    assert not any("gelu" in op.name or "rms" in op.name for op in empty)

    mixed = copy.deepcopy(rel)
    mixed["roles"]["vision"]["fp32"] = {"params": 8, "bytes": 32}
    cleared = from_release("synth-mix", HF, rel=mixed, cfg=mixed["config"])
    assert cleared.vision is not None and cleared.vision.ln_mode == ""
    assert cleared.vision.audio is not None and cleared.vision.audio.vec_fmt == ""
    mixed_ops = audio_ops(cleared.vision.audio, 1, 1)
    assert not any("gelu" in op.name or "rms" in op.name for op in mixed_ops)

    whisper = get_model("whisper-large-v3").vision.audio
    assert whisper is not None and whisper.kind == "whisper"
    wnames = [op.name for op in audio_ops(whisper, 30, 1)]
    assert "aud.dual_rms" not in wnames and "aud.0.0.gelu" not in wnames
    assert "aud.0.patch_rms" not in wnames
