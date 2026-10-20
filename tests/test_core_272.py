"""0.272 speech round 3: Fun-ASR-Nano-2512 (transformers fun_asr_nano: SAN-M encoder with FSMN, LFR 7/6, adaptor,
Qwen3-0.6B); Qwen2.5-Omni speech card with an SLC pin (DiT re-read weights first, then the Talker)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.omni import omni_pin, omni_slc_plan, talker_weight_bytes
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.vision import audio_ops

LP = "lpddr5x_4x64_8533_16g"
FN, OM = "fun-asr-nano-2512", "qwen2.5-omni-7b"


def test_fun_asr_nano_exact_params():
    """Header (FunAudioLLM/Fun-ASR-Nano-2512-hf, bf16): Qwen3-0.6B (tied) 596,049,920; audio tower + projector +
    adaptor 233,741,920 = 70 SAN-M layers (layer 0: 560 → 512) + 2 final LayerNorms + Linear 512 → 2048 → 1024 +
    2 adaptor blocks (d1024, ReLU MLP 256)."""
    m = get_model(FN)
    a = m.vision.audio
    assert m.params() == 596_049_920 and m.param_check()["rel_err"] == 0.0
    assert a.params == m.vision_params == 233_741_920 and a.kind == "sanm"
    d, k = 512, 11
    blk = lambda i, dd, f, fs: 2 * i + 3 * (i * dd + dd) + dd * dd + dd + (dd * k if fs else 0) + 2 * dd + 2 * dd * f + f + dd
    hand = blk(560, d, 2048, True) + 69 * blk(d, d, 2048, True) + 4 * d + 512 * 2048 + 2048 + 2048 * 1024 + 1024 \
        + 2 * blk(1024, 1024, 256, False)
    assert hand == a.params
    e = {x["id"]: x for x in list_models()}[FN]
    assert e["domain"] == "asr" and e["family"] == "Fun-ASR-Nano" and e["coverage"] == "full"


def test_lfr_tokens_and_cap():
    """FunAsrNanoFeatureExtractor: fbank frames 1 + (N − 400) // 160, LFR ⌈frames / 6⌉ = audio tokens (mask sum);
    the sinusoid table has 2049 rows used from position 1 → at most 2048 positions (≈ 122.9 s)."""
    a = get_model(FN).vision.audio
    assert a.tokens(30.0) == 500 and a.tokens(10.0) == 167 and a.tokens(1.0) == 17
    assert a.enc_positions(30.0) == a.prompt_tokens(30.0) == 500
    assert a.maxf == 2048 and a.tokens(200.0) == 2048
    r = evaluate(Scenario(model=FN, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=8, ctx=58, audio_s=130.0)))
    assert r.asr["prompt_audio_tokens"] == 2048 and abs(r.asr["truncated_s"] - (130.0 - 2048 * 0.06)) < 1e-9


def test_sanm_flops_hand_count():
    """30 s (T = 500): 70 blocks (q/k/v/o + ReLU MLP 2048, layer 0 from 560) + attention 2·2·T²·d per layer,
    projector, 2 adaptor blocks — ≈ 0.27 TFLOP vs Whisper-v3's 2.28 per 30 s window."""
    a = get_model(FN).vision.audio
    T, d = 500, 512
    ops = audio_ops(a, 30.0, 1)
    fl = sum(2 * o.m * o.k * o.n * o.count for o in ops if o.kind in ("gemm", "attn"))
    hand = (2 * T * (560 * 3 * d + d * d + 2 * d * 2048) + 69 * 2 * T * (4 * d * d + 2 * d * 2048) + 70 * 4 * T * T * d
            + 2 * T * (512 * 2048 + 2048 * 1024) + 2 * (2 * T * (4 * 1024 ** 2 + 2 * 1024 * 256) + 4 * T * T * 1024))
    assert abs(fl / hand - 1) < 1e-12 and 0.26e12 < fl < 0.28e12
    fsmn = [o for o in ops if o.name.endswith(".fsmn")]
    assert len(fsmn) == 70 and all(o.vec == T * d * 2 * 11 for o in fsmn)
    # weight reads = the tower's params minus LayerNorms (not streamed): 70·2 + 2 final + adaptor 2·2 norms
    w = sum(o.w_params for o in ops)
    ln = 2 * 560 + 69 * 2 * d + 70 * 2 * d + 4 * d + 2 * 2 * 2 * 1024
    assert w == a.params - ln


def test_fun_asr_nano_speech_kpis_and_api():
    k = speech_kpis(Scenario(model=FN, chip=CHIPS["100T"], mem_id=LP,
                             serving=Serving(batch=1, prompt=32, out_len=100, audio_s=30.0)))
    assert k["kind"] == "asr" and k["prompt_audio_tokens"] == 500 and 0 < k["encoder_s"] < k["first_token_s"]
    q3 = speech_kpis(Scenario(model="qwen3-asr-1.7b", chip=CHIPS["100T"], mem_id=LP,
                              serving=Serving(batch=1, prompt=32, out_len=100, audio_s=30.0)))
    assert k["tpot_s"] < q3["tpot_s"] and k["rtf"] < q3["rtf"]          # 0.6B vs 1.7B decoder dominates at B1
    o = api.api_eval({"scenario": {"model": FN, "serving": {"audio_s": 12, "prompt": 8, "out_len": 40, "ctx": 28}},
                      "sensitivity": False, "goodput": False})
    assert o["speech"]["kind"] == "asr"


def _omni(gib: float, b: int, policy: str = "pin") -> dict:
    ch = replace(CHIPS["100T"], slc_mib=gib * 1024.0, slc_policy=policy)
    return speech_kpis(Scenario(model=OM, chip=ch, mem_id=LP,
                                serving=Serving(batch=b, prompt=64, out_len=60, audio_s=10.0)), 10.0)


def test_omni_slc_plan_order():
    """DiT SRAM-miss weights first (50·b reads / s per byte vs the Talker's 50), then the Talker (2.64 GB bf16)."""
    o = get_model(OM).omni
    miss = omni_pin(o, 64 * 2 ** 20)
    wt, wd = talker_weight_bytes(o), o.dit_run_params * 4.0 * miss
    assert abs(wt - 2.642e9) < 1e6
    assert omni_slc_plan(o, miss, 0.0, 1) == (0.0, 0.0)
    h = omni_slc_plan(o, miss, 2 ** 30, 1)
    assert h[1] == 0.0 and abs(h[0] - 2 ** 30 / wd) < 1e-12
    h = omni_slc_plan(o, miss, wd + 0.5 * wt, 4)
    assert h[0] == 1.0 and abs(h[1] - 0.5) < 1e-12
    assert omni_slc_plan(o, miss, 8 * 2 ** 30, 8) == (1.0, 1.0)


def test_omni_slc_and_batching():
    """No SLC (or a non-pin policy) → the 0.271 numbers; batching alone cannot reach real time on 100T + LPDDR5X
    (token2wav batch 1: DiT time grows with b, the Talker step barely does); a 4 GiB pinned SLC holds DiT + Talker."""
    base, lru = _omni(0, 1), _omni(4, 1, "lru")
    assert base["slc"]["talker_hit"] == 0.0 and base["rtf"] == lru["rtf"] and base["talker_step_s"] == lru["talker_step_s"]
    assert base["streams_realtime"] == 0 and base["rtf"] > 1.0
    b8 = _omni(0, 8)
    assert b8["talker_step_s"] < 1.1 * base["talker_step_s"] and b8["rtf"] > 3 * base["rtf"]
    one, four = _omni(1, 1), _omni(4, 1)
    assert one["slc"]["talker_hit"] == 0.0 and one["slc"]["dit_hit"] > 0.8 and one["dit_chunk_s"] < base["dit_chunk_s"]
    assert one["talker_step_s"] == base["talker_step_s"]
    assert four["slc"]["talker_hit"] == 1.0 and four["talker_step_s"] < 0.2 * base["talker_step_s"]
    assert four["rtf"] < 0.25 and four["streams_realtime"] >= 8 and four["first_audio_s"] < base["first_audio_s"]


def test_step_audio_2_mini():
    """Header (stepfun-ai/Step-Audio-2-mini, bf16): Qwen2Model 28 × h3584 (q / k / v bias) + untied lm_head (vocab
    158720) = 7,663,326,720; encoder (Whisper blocks, learned positions 1500 × 1280) + Adaptor (conv k3 s2 + 2 Linears)
    = 651,852,544.  Tokens per the forward: ((L + 1)//2//2 − 1)//2 + 1 over L = (samples + 479) // 160 mel frames."""
    m = get_model("step-audio-2-mini")
    a = m.vision.audio
    assert m.params() == 7_663_326_720 and m.param_check()["rel_err"] == 0.0 and not m.tie_embeddings
    assert a.params == m.vision_params == 651_852_544
    for s in (1.0, 10.0, 17.3):
        L = (round(s * 16000) + 479) // 160
        assert a.mel_frames(s) == L and a.tokens(s) == ((L + 1) // 2 // 2 - 1) // 2 + 1
    assert a.tokens(10.0) == 125 and a.enc_positions(10.0) == 501 and a.mel_frames(40.0) == 2999
    ops = audio_ops(a, 10.0, 1)
    conv = [o for o in ops if o.name == "aud.proj.conv0"][0]
    assert (conv.m, conv.k, conv.n) == (125, 3 * 1280, 1280)
    assert [o.n for o in ops if o.name.startswith("aud.proj.linear")] == [2048, 3584]
    e = {x["id"]: x for x in list_models()}["step-audio-2-mini"]
    assert e["domain"] == "omni" and e["family"] == "Step-Audio-2"      # 0.273: + token2wav → omni
    r = evaluate(Scenario(model="step-audio-2-mini", chip=CHIPS["100T"], mem_id=LP,
                          serving=Serving(batch=1, prompt=8, ctx=58, audio_s=40.0)))
    assert r.asr["prompt_audio_tokens"] == a.tokens(40.0) == 375 and abs(r.asr["truncated_s"] - 10.01) < 1e-9
