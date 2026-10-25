"""0.395: OmniASR CTC is the Wav2Vec2 tower plus a biased linear head.

OmniASRForCTC.ctc_head is Linear(hidden, vocab) with bias on every encoder frame.
There is no Llama decoder and no Shaw conformer. Weights and activations are fp32.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, audio_spec

VOCAB = 10288
# (params, bytes, tensors, hidden, layers, ffn) from the released safetensors headers
WANT = {
    "omniasr-ctc-300m-v2": (325_982_768, 1_303_931_072, 422, 1024, 24, 4096),
    "omniasr-ctc-1b-v2": (975_674_928, 3_902_699_712, 806, 1280, 48, 5120),
    "omniasr-ctc-3b-v2": (3_081_398_832, 12_325_595_328, 998, 2048, 60, 8192),
    "omniasr-ctc-7b-v2": (6_505_761_328, 26_023_045_312, 2086, 2048, 128, 8192),
}


def test_omniasr_ctc_catalog_matches_the_header():
    listed = {e["id"]: e for e in list_models()}
    for mid, (params, nbytes, nt, hidden, layers, ff) in WANT.items():
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0, mid
        assert m.params() == params == rel["params_vision"] == m.release_params == m.vision_params, mid
        assert rel["params_llm"] == 0 and rel["params_mtp"] == 0, mid
        assert rel["n_tensors"] == nt and rel["bytes_total"] == nbytes == 4 * params, mid
        assert m.ctc_only and m.domain == "asr" and not m.kv_cache and m.layers == ()
        assert m.n_layers == 0 and m.tie_embeddings and m.coverage == "full" and not m.coverage_reasons
        assert m.arch == "OmniASRForCTC" and m.hidden == hidden and m.vocab == VOCAB
        assert m.params() != m.vocab * m.hidden
        vis = rel["roles"]["vision"]
        assert set(rel["roles"]) == {"vision"} and set(vis) == {"fp32"}
        assert vis["fp32"]["params"] == params and vis["fp32"]["bytes"] == 4 * params
        assert "lm_head" not in rel["roles"] and "embed" not in rel["roles"]
        assert m.fmt("attn").fmt == "fp32" and m.act_fmt == "fp32" and m.kv_fmt == "fp32"
        assert m.vision.w_Bpp == 4.0 and m.final_norm_params == 0
        au = m.vision.audio
        assert au.kind == "w2v" and au.ctc and au.w_fmt == "fp32" and au.w_bits == 32.0
        assert au.dims[0] == hidden and au.depths[0] == layers and int(au.mlp * hidden) == ff
        assert au.heads == 16 and au.out == VOCAB and au.lin == ((hidden, VOCAB),)
        assert au.n_mels == 512 and au.ch == 16 and au.kernel == 128
        head = hidden * VOCAB + VOCAB
        assert head == {1024: 10_545_200, 1280: 13_178_928, 2048: 21_080_112}[hidden]
        assert au.tokens(0) == 0 and au.tokens(1.0) == 49 and au.tokens(10.0) == 499
        assert au.enc_positions(1.0) == 49 and au.prompt_tokens(1.0) == au.prompt_tokens(10.0) == 0
        row = listed[mid]
        assert row["domain"] == "asr" and row["family"] == "OmniASR" and row["coverage"] == "full"
        assert row["dtype"] == "W fp32 · A fp32"
        assert "没有 Llama" in " ".join(m.notes) and "Shaw" in " ".join(m.notes)


def test_omniasr_ctc_ops_follow_the_class():
    for mid, (params, _nbytes, _nt, hidden, layers, _ff) in WANT.items():
        au = get_model(mid).vision.audio
        ops = audio_ops(au, 1.0, 1, "fp32")
        by = {}
        for op in ops:
            assert op.name not in by, op.name
            by[op.name] = op
        assert not any("shaw" in name or name.startswith("aud.proj") for name in by)
        ctc = by["aud.ctc"]
        assert (ctc.m, ctc.k, ctc.n, ctc.count) == (49, hidden, VOCAB, 1)
        assert ctc.w_params == hidden * VOCAB + VOCAB
        assert ctc.w_params != hidden * VOCAB
        qk, pv = by["aud.0.qk"], by["aud.0.pv"]
        assert (qk.m, qk.n, qk.causal) == (49, 49, 1.0)
        assert (pv.m, pv.k, pv.causal) == (49, 49, 1.0)
        assert qk.count == pv.count == 16
        assert by[f"aud.{layers - 1}.o"].m == 49
        assert by["aud.fe.0"].k == 10 and by["aud.fe.proj"].k == 512
        assert all(op.w_fmt == "fp32" and op.a_fmt == "fp32" and op.w_bits == 32.0
                   for op in ops if op.kind == "gemm")
        assert sum(op.w_params for op in ops if op.name == "aud.ctc") == hidden * VOCAB + VOCAB
        two = audio_ops(au, 1.0, 2, "fp32")
        ctc2 = next(op for op in two if op.name == "aud.ctc")
        assert ctc2.m == 98 and ctc2.w_params == ctc.w_params
        assert au.params == params
    pre = evaluate(Scenario(model="omniasr-ctc-300m-v2",
                            serving=Serving(phase="prefill", batch=1, prompt=4, audio_s=1.0)))
    assert pre.asr["enc_positions"] == 49 and pre.asr["prompt_audio_tokens"] == 0
    assert pre.vision["audio"]["tokens"] == 49 and pre.step > 0 and pre.fits
    dec = evaluate(Scenario(model="omniasr-ctc-300m-v2",
                            serving=Serving(phase="decode", batch=1, audio_s=1.0)))
    assert any("没有自回归" in w for w in dec.warnings)
    assert dec.asr["enc_positions"] == 49 and dec.step > 0
    try:
        evaluate(Scenario(model="omniasr-ctc-300m-v2", layout=Layout(tp=2),
                          serving=Serving(phase="prefill", batch=1, audio_s=1.0)))
        raise AssertionError("tp 2 accepted")
    except ValueError as exc:
        assert "tp / sp" in str(exc)
    try:
        evaluate(Scenario(model="omniasr-ctc-300m-v2", layout=Layout(pp=2),
                          serving=Serving(phase="prefill", batch=1, audio_s=1.0)))
        raise AssertionError("pp 2 accepted")
    except ValueError as exc:
        assert "pp exceeds layer count" in str(exc)


def test_omniasr_ctc_refuses_a_different_tower():
    rel = load_release(get_model("omniasr-ctc-300m-v2").hf_id)
    kept = audio_spec(rel["config"])
    assert kept is not None and kept.ctc and kept.params == WANT["omniasr-ctc-300m-v2"][0]

    def gone(mut):
        cfg = copy.deepcopy(rel["config"])
        mut(cfg)
        assert audio_spec(cfg) is None

    gone(lambda cfg: cfg.pop("vocab_size"))
    gone(lambda cfg: cfg.__setitem__("vocab_size", 0))
    gone(lambda cfg: cfg["audio_config"].__setitem__("hidden_act", "relu"))
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_bias", False))
    gone(lambda cfg: cfg["audio_config"].__setitem__("num_attention_heads", 7))
    gone(lambda cfg: cfg["audio_config"].__setitem__("num_conv_pos_embedding_groups", 3))
    uneven = list(rel["config"]["audio_config"]["conv_dim"])
    uneven[-1] = 256
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_dim", uneven))
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_dim", []))
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_kernel", [10, 3]))
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_stride", rel["config"]["audio_config"]["conv_stride"] + [2]))
    for key in ("activation_dropout", "attention_dropout", "hidden_dropout", "layerdrop"):
        cfg = copy.deepcopy(rel["config"])
        cfg["audio_config"][key] = 0.9
        again = audio_spec(cfg)
        assert again is not None and again.params == kept.params and again.lin == kept.lin


def test_omniasr_llm_projector_is_unchanged():
    m = get_model("omniasr-llm-300m-v2")
    au = m.vision.audio
    assert not au.ctc and au.lin == ((1024, 4096),) and au.prompt_tokens(10.0) == 499
    assert au.tokens(10.0) == 499 and m.vision.w_Bpp == 2.0
    names = [op.name for op in audio_ops(au, 10.0, 1)]
    assert "aud.proj.linear0" in names and "aud.ctc" not in names
    assert m.param_check()["rel_err"] == 0.0 and not m.ctc_only
