"""0.270 speech round: Whisper-large-v3 / -turbo (encoder-decoder, cross-attention decode path, 30 s padded
windows), Qwen2-Audio-7B-Instruct (Whisper-v3 encoder + AvgPool + Qwen-7B), Qwen3-ASR-1.7B (AuT encoder +
Qwen3-1.7B); audio-seconds serving convention (core/speech.py)."""
from pathlib import Path

from accel_dse import api
from accel_dse.core import ir
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.vision import audio_ops
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
W3, WT, Q2, Q3 = "whisper-large-v3", "whisper-large-v3-turbo", "qwen2-audio-7b-instruct", "qwen3-asr-1.7b"


def test_exact_params_vs_header():
    """Released safetensors headers (HF): decoder / LLM = params_llm, audio encoder (+ projector) = params_vision.
    0.358: Whisper encoder-decoder params() includes encoder + decoder; others params() = LLM only."""
    want = {W3: (906_521_600, 636_968_960), WT: (171_909_120, 636_968_960),
            Q2: (7_754_878_976, 642_215_936), Q3: (1_720_574_976, 317_477_504)}
    fams = {W3: "Whisper", WT: "Whisper", Q2: "Qwen2-Audio", Q3: "Qwen3-ASR"}
    ents = {e["id"]: e for e in list_models()}
    for mid, (llm, enc) in want.items():
        m = get_model(mid)
        # 0.358: Whisper encoder-decoder params() = encoder + decoder; others = LLM only
        if m.enc_ctx > 0:  # Whisper encoder-decoder
            assert m.params() == llm + enc and m.param_check()["rel_err"] == 0.0, mid
        else:  # Qwen2-Audio / Qwen3-ASR: LLM + separate audio tower
            assert m.params() == llm and m.param_check()["rel_err"] == 0.0, mid
        assert m.vision.audio.params == enc == m.vision_params, mid
        # 0.358: Whisper is full coverage after params fix
        assert ents[mid]["domain"] == "asr" and ents[mid]["family"] == fams[mid], mid
        if m.enc_ctx > 0:  # Whisper
            assert ents[mid]["coverage"] == "full", mid
        else:
            assert ents[mid]["coverage"] == "full", mid
    assert get_model(W3).params() == 1_543_490_560     # 0.358: params() includes encoder
    w = get_model(W3)
    assert w.enc_ctx == 1500 and w.max_ctx == 448 and w.tie_embeddings and sum(L.cross is not None for L in w.layers) == 32
    assert sum(L.cross is not None for L in get_model(WT).layers) == 4
    assert get_model(Q2).enc_ctx == 0 and not get_model(Q2).vision.audio.cross


def test_audio_positions_and_prompt_tokens():
    w, q2, q3 = (get_model(x).vision.audio for x in (W3, Q2, Q3))
    # Whisper: every pass padded to 30 s → 1500 positions; long-form = sequential windows; nothing in the prompt
    assert w.chunks(10) == (3000,) and w.enc_positions(10) == 1500 and w.enc_positions(45) == 3000
    assert w.prompt_tokens(30) == 0 and w.windows(60) == 2 and w.windows(60.01) == 3
    # Qwen2-Audio processor: (((L − 1)//2 + 1) − 2)//2 + 1 over the unpadded frames; truncated at 30 s
    assert (q2.tokens(10), q2.tokens(30), q2.tokens(45)) == (250, 750, 750) and q2.enc_positions(45) == 1500
    # Qwen3-ASR AuT: 13 positions per 100-frame chunk, tail chunk by three (k3, s2, p1) convs; windows of 104
    assert q3.tokens(10) == 130 and q3.tokens(1.5) == 13 + 7 and q3.win == 104 and q3.enc_positions(10) == q3.prompt_tokens(10)
    # Muse Spark (raw-sample encoder, no chunks): enc_positions = tokens (the 0.175 path; Result.asr calls it)
    mu = get_model("muse-spark-open").vision.audio
    assert mu.enc_positions(3.0) == mu.tokens(3.0) == mu.prompt_tokens(3.0)


def test_encoder_flops_hand_count():
    """Whisper-v3 encoder, one 30 s window: 2 · 1500 · (32 blocks × 12·1280² + conv2 3·1280²) + 2 · 3000 · 3·128·1280
    + attention 32 × 2 × 2 · 1500² · 1280 ≈ 2.28 TFLOP; AuT 30 s ≈ 2 · 390 · 302 M + convs ≪ Whisper (8× fewer positions)."""
    w = get_model(W3).vision.audio
    ops = audio_ops(w, 30.0, 1)
    fl = sum(2 * o.m * o.k * o.n * o.count for o in ops if o.kind in ("gemm", "attn"))
    hand = 2 * 1500 * (32 * 12 * 1280 ** 2 + 3 * 1280 ** 2) + 2 * 3000 * 3 * 128 * 1280 + 32 * 2 * 2 * 1500 ** 2 * 1280
    assert abs(fl / hand - 1) < 1e-9
    q3 = get_model(Q3).vision.audio
    fl3 = sum(2 * o.m * o.k * o.n * o.count for o in audio_ops(q3, 30.0, 1) if o.kind in ("gemm", "attn"))
    assert fl3 < fl / 5


def test_cross_attention_ops_decode_vs_prefill():
    m = get_model(W3)
    dec = ir.build_rank_ops(m, 0, Phase("decode", 4, 1, 60), Shard())
    n = {o.name: o for o in dec}
    assert "cross.q" in n and "cross.o" in n and "cross.k" not in n and "cross.v" not in n
    assert n["cross_qk"].n == 1500 and n["cross_qk"].count == 4 * 20 and n["cross_qk"].m == 1
    assert n["cross_softmax"].kv_read == 4 * 1500 * 20 * 128 * 2 and n["cross_softmax"].kv_write == 0
    pre = {o.name: o for o in ir.build_rank_ops(m, 0, Phase("prefill", 4, 8, 0), Shard())}
    assert pre["cross.k"].m == 4 * 1500 and pre["cross_softmax"].kv_write == 4 * 1500 * 20 * 128 * 2
    # chunked-prefill continuation reads the cached cross K/V instead of re-projecting
    cont = {o.name: o for o in ir.build_rank_ops(m, 0, Phase("prefill", 4, 8, 8), Shard())}
    assert "cross.k" not in cont and cont["cross_softmax"].kv_read > 0
    # TP 4: heads split, o row-split + all-reduce
    tp = {o.name: o for o in ir.build_rank_ops(m, 0, Phase("decode", 4, 1, 60), Shard(tp=4))}
    assert tp["cross_qk"].count == 4 * 5 and "cross_allreduce" in tp


def test_cross_kv_state_and_batch_mechanism():
    """Cross K/V per sequence = layers · 1500 · 2 · 1280 · 2 B (245.76 MB v3, 30.72 MB turbo) — fixed per request and
    read every decode step: at batch 32 it is 7.9 GB / step for v3, the decode cost at batch (not the weights)."""
    def ev(mid, b):
        return evaluate(Scenario(model=mid, chip=CHIPS["100T"], mem_id=LP,
                                 serving=Serving(batch=b, prompt=8, ctx=58, audio_s=30.0)))
    r1, r32, t32 = ev(W3, 1), ev(W3, 32), ev(WT, 32)
    assert abs(r1.asr["cross_kv_MB_per_seq"] - 245.76) < 1e-9 and abs(t32.asr["cross_kv_MB_per_seq"] - 30.72) < 1e-9
    assert abs(r32.stages[0].mem.state_total - 32 * 245.76e6) / (32 * 245.76e6) < 1e-9
    assert r32.tpot > 4 * r1.tpot                    # B1: weights (1.8 GB) dominate; B32: 7.9 GB of cross K/V
    assert r32.tpot / t32.tpot > 5                    # 8× fewer decoder layers ⇒ 8× less cross K/V and weights


def test_speech_convention():
    s = Scenario(model=W3, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=8, out_len=100, audio_s=30.0))
    k = speech_kpis(s)
    assert k["kind"] == "asr" and abs(k["total_s"] - (k["first_token_s"] + 100 * k["tpot_s"])) < 1e-12
    assert abs(k["rtf"] - k["total_s"] / 30.0) < 1e-12 and 0 < k["encoder_s"] < k["first_token_s"]
    assert k["prompt_audio_tokens"] == 0 and k["enc_positions"] == 1500
    pre = evaluate(s.replace("serving.phase", "prefill"))
    a = pre.asr
    assert abs(a["audio_s_per_s"] - 30.0 / pre.step) < 1e-9 and a["first_token_s"] == pre.step
    # Qwen2-Audio: audio positions join the prompt (750 for 30 s), truncation reported past 30 s
    q = evaluate(Scenario(model=Q2, chip=CHIPS["100T"], mem_id=LP,
                          serving=Serving(batch=1, prompt=8, ctx=58, audio_s=40.0)))
    assert q.asr["prompt_audio_tokens"] == 750 and abs(q.asr["truncated_s"] - 10.0) < 1e-9
    # TTS through the same convention: first audio after one decode step
    t = speech_kpis(Scenario(model="ming-3.0-tts-tiny", chip=CHIPS["100T"], mem_id=LP, serving=Serving(prompt=32)), 2.0)
    assert t["kind"] == "tts" and t["steps"] == 25 and abs(t["first_audio_s"] - t["ttft_s"] - t["tpot_s"]) < 1e-12


def test_api_and_ui():
    o = api.api_eval({"scenario": {"model": WT, "serving": {"audio_s": 12, "prompt": 8, "out_len": 40, "ctx": 28}},
                      "sensitivity": False, "goodput": False})
    assert o["speech"]["kind"] == "asr" and o["summary"]["asr"]["windows"] == 1
    try:
        api.api_eval({"scenario": {"model": WT}, "audio_out_s": 0, "sensitivity": False})
        assert False
    except api.ApiError:
        pass
    js = W.read()                                      # 0.286: token match
    assert W.has(js, "s.asr") and W.has(js, "r.speech")
