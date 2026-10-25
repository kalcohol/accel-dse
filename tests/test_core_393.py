"""0.393: Granite Speech 5 CTC encoder, with no granite LM.

The Shaw conformer, stride-2 subsampling, tied CTC head, and fp32 BatchNorm affine
are billed from GraniteSpeech5ForCTC. 0.392 Q-Former / Shaw / scalar properties stay put.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, audio_spec

MID = "granite-speech-5.0-470m-turboctc"
PARAMS = 472_928_256
FP32_AFFINE = 16 * 2 * 2048


def test_turboctc_catalog_matches_the_header():
    m = get_model(MID)
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == PARAMS == rel["params_vision"] == m.release_params == m.vision_params
    assert rel["params_llm"] == 0 and rel["params_mtp"] == 0
    assert m.ctc_only and m.domain == "asr" and not m.kv_cache and m.layers == ()
    assert m.n_layers == 0 and m.tie_embeddings and m.coverage == "full" and not m.coverage_reasons
    assert m.hidden == 1024 and m.vocab == 16384
    assert m.params() != m.vocab * m.hidden
    assert rel["n_tensors"] == 550 and rel["bytes_total"] == 946_249_856
    assert rel["roles"]["buffer"]["int64"]["bytes"] == 262_272
    vis = rel["roles"]["vision"]
    assert set(vis) == {"bf16", "fp32"}
    assert vis["fp32"]["params"] == FP32_AFFINE and vis["fp32"]["bytes"] == 4 * FP32_AFFINE
    assert vis["bf16"]["bytes"] == 2 * vis["bf16"]["params"]
    assert vis["bf16"]["params"] + vis["fp32"]["params"] == PARAMS
    assert "lm_head" not in rel["roles"] and "embed" not in rel["roles"]
    assert m.fmt("attn").fmt == "bf16" and m.act_fmt == "bf16"
    assert m.vision.wfmt == () and m.vision.audio.w_fmt == "bf16"
    assert abs(m.vision.params * m.vision.w_Bpp - sum(x["bytes"] for x in vis.values())) < 1
    au = m.vision.audio
    assert au.kind == "gsctc" and au.gs_sub == (0, 1) and au.chunk == 128 and au.gs_mid == 16384
    assert au._gsctc_frames(1.0) == (50, 25) + (12,) * 15
    assert au.tokens(1.0) == au.enc_positions(1.0) == 12 and au.prompt_tokens(1.0) == 0
    assert au.tokens(0.0) == 0 and au._gsctc_frames(0.0) == ()
    listed = next(e for e in list_models() if e["id"] == MID)
    assert listed["domain"] == "asr" and listed["family"] == "Granite-Speech" and listed["coverage"] == "full"
    assert listed["dtype"] == "W bf16 · A bf16"
    assert "没有 granite" in " ".join(m.notes)


def test_turboctc_ops_follow_the_class():
    au = get_model(MID).vision.audio
    ops = audio_ops(au, 1.0, 1)
    by = {}
    for op in ops:
        assert op.name not in by, op.name
        by[op.name] = op
    assert (by["aud.in"].m, by["aud.in"].k, by["aud.in"].n) == (50, 320, 1024)
    assert (by["aud.0.q"].m, by["aud.0.k"].m, by["aud.0.v"].m) == (128, 128, 128)
    assert by["aud.0.o"].m == 50 and by["aud.0.ff1_up"].m == 50 and by["aud.0.ff1_up"].n == 4096
    assert by["aud.0.ff2_up"].m == 25 and by["aud.0.conv_down"].m == 25
    assert by["aud.0.dconv"].kind == "vector" and by["aud.0.dconv"].vec == 25 * 2048 * 7
    assert (by["aud.1.conv_down"].m, by["aud.1.ff2_up"].m) == (13, 12)
    assert by["aud.1.dconv"].vec == 13 * 2048 * 7
    assert by["aud.2.q"].m == 128 and by["aud.2.o"].m == 12 and by["aud.2.conv_down"].m == 12
    assert by["aud.2.ff2_down"].m == 12 and by["aud.15.ff1_up"].m == 12
    shaw = by["aud.0.shaw"]
    assert (shaw.m, shaw.k, shaw.n, shaw.count) == (128, 128, 128, 8)
    assert shaw.w_params == 0 and shaw.flops == 2 * 128 * 128 * 128 * 8
    assert (by["aud.mid.out"].m, by["aud.mid.out"].k, by["aud.mid.out"].n) == (12, 1024, 16384)
    assert (by["aud.mid.back"].k, by["aud.mid.back"].n) == (16384, 1024)
    assert (by["aud.ctc"].m, by["aud.ctc"].k, by["aud.ctc"].n) == (12, 1024, 16384)
    assert by["aud.ctc"].w_params == by["aud.mid.out"].w_params == 1024 * 16384
    assert sum(op.w_params for op in ops if op.name in ("aud.mid.out", "aud.ctc")) == 2 * 1024 * 16384
    assert au.params == PARAMS
    assert all(op.w_fmt == "bf16" and op.a_fmt == "bf16" for op in ops if op.kind == "gemm")
    assert not any("qf" in op.name or "proj" in op.name for op in ops)
    pre = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=4, audio_s=1.0)))
    assert pre.asr["enc_positions"] == 12 and pre.asr["prompt_audio_tokens"] == 0
    assert pre.vision["audio"]["tokens"] == 12 and pre.step > 0 and pre.fits
    dec = evaluate(Scenario(model=MID, serving=Serving(phase="decode", batch=1, audio_s=1.0)))
    assert any("没有自回归" in w for w in dec.warnings)
    assert dec.asr["enc_positions"] == 12 and dec.step > 0


def test_turboctc_refuses_a_different_tower():
    rel = load_release(get_model(MID).hf_id)
    kept = audio_spec(rel["config"])
    assert kept is not None and kept.params == PARAMS

    def gone(mut):
        cfg = copy.deepcopy(rel["config"])
        mut(cfg)
        assert audio_spec(cfg) is None

    gone(lambda cfg: cfg.__setitem__("tie_word_embeddings", False))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("attention_bias", False))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("hidden_act", "gelu"))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("num_key_value_heads", 4))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("context_size", 600))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("head_dim", 64))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("conv_kernel_size", 8))
    gone(lambda cfg: cfg.__setitem__("vocab_size", 1024))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("subsample_layers", [0, 0]))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("subsample_layers", [16]))
    bare = copy.deepcopy(rel["config"])
    bare["encoder_config"]["subsample_layers"] = []
    spec = audio_spec(bare)
    assert spec is not None and spec.params == PARAMS and spec._gsctc_frames(1.0) == (50,) * 17


def test_granite_speech_qformer_properties_hold():
    base = get_model("granite-speech-5.0-2b")
    plus = get_model("granite-speech-4.1-2b-plus")
    for m in (base, plus):
        ops = audio_ops(m.vision.audio, 1.0, 1)
        names = [op.name for op in ops]
        assert not any("embed" in name or "text" in name for name in names)
        assert all(op.kind != "lookup" for op in ops)
        shaw = next(op for op in ops if op.name == "aud.0.shaw")
        assert shaw.flops == 2.0 * shaw.m * shaw.k * shaw.n * shaw.count
        ups = [op for op in ops if op.name.endswith(".conv_up")]
        downs = [op for op in ops if op.name.endswith(".conv_down")]
        assert len(ups) == len(downs) == 16
        assert not any(op.name.endswith(".conv_glu") for op in ops)
    assert next(op for op in audio_ops(plus.vision.audio, 1.0, 1) if op.name == "aud.qf.0.ck").k == 2048
    assert next(op for op in audio_ops(plus.vision.audio, 1.0, 1) if op.name == "aud.proj").k == 1024
    assert next(op for op in audio_ops(base.vision.audio, 1.0, 1) if op.name == "aud.qf.0.ck").k == 1024
    rel = load_release(base.hf_id)
    cfg = copy.deepcopy(rel["config"])
    enc = cfg["encoder_config"]
    for key in ("dropout", "attention_dropout"):
        if key in enc:
            enc[key] = 0.5
    again = audio_spec(cfg)
    assert again is not None and again.params == base.vision.audio.params
    assert [op.name for op in audio_ops(again, 1.0, 1)] == [op.name for op in audio_ops(base.vision.audio, 1.0, 1)]
