"""0.273 speech round 4: UI surfaces the speech API blocks (Omni split / SLC / audio-in, CosyVoice split, ASR cap);
Step-Audio-2-mini speech output (CosyVoice2-style token2wav with K/V-cached chunked flow + HiFT)."""
from pathlib import Path

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.cosyvoice import flow_chunk_ops, hift_ops
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
SA = "step-audio-2-mini"


def test_ui_surfaces_speech_blocks():
    js = (ROOT / "accel_dse/web/app.js").read_text()
    for key in ("p.slc", "c.talker_hit", "c.dit_hit", "Omni 首音频拆分", "p.audio_in", "split_first", "p.flow_s",
                "a.max_audio_s", "p.stream === 'cache'", "p.label"):
        assert W.mentions(js, key), key
    hl = (ROOT / "scripts/ui_headless_smoke.py").read_text()
    assert "qwen2.5-omni-7b" in hl and "cosyvoice2-0.5b" in hl


def test_asr_cap_seconds():
    caps = {"qwen2-audio-7b-instruct": 30.0, "fun-asr-nano-2512": 122.88, "qwen2.5-omni-7b": 30.0, SA: 29.99,
            "qwen3-asr-1.7b": None}
    for mid, cap in caps.items():
        r = evaluate(Scenario(model=mid, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=8, ctx=58, audio_s=200.0)))
        got = r.asr["max_audio_s"]
        assert (got is None and cap is None) or abs(got - cap) < 1e-9, (mid, got)
        if cap:
            assert abs(r.asr["truncated_s"] - (200.0 - cap)) < 1e-9


def test_step_audio_token2wav_params():
    """token2wav/flow.pt 155,802,352 + hift.pt 20,821,295 (fp32, state-dict metadata via HTTP Range) — outside
    params_llm / params_vision; the release json carries them as params_flow / params_vocoder."""
    m = get_model(SA)
    c = m.cosy
    assert c is not None and c.stream == "cache" and c.version == 2 and c.label == "Step-Audio-2 token2wav"
    assert c.flow_params == 155_802_352 and c.hift_params == 20_821_295
    assert m.params() == 7_663_326_720 and m.vision_params == 651_852_544
    est = c._p("flow.decoder.estimator.")
    # 16 DiT blocks: q/k/v/proj 4·(512² + 512), q/k LayerNorm(64) ×2, causal conv block 2·(512·512·3 + 512) + LN 1024,
    # MLP 512 → 2048 → 512, adaLN 512 → 9·512; + t_embedder (256 → 512 → 512), in_proj 320 → 512, final layer
    blk = 4 * (512 * 512 + 512) + 2 * 128 + 2 * (512 * 512 * 3 + 512) + 1024 + (512 * 2048 + 2048) + (2048 * 512 + 512) \
        + (512 * 9 * 512 + 9 * 512)
    hand = 16 * blk + (256 * 512 + 512) + (512 * 512 + 512) + (320 * 512 + 512) + (512 * 1024 + 1024) + (512 * 80 + 80)
    assert est == hand
    e = {x["id"]: x for x in list_models()}[SA]
    assert e["domain"] == "omni" and e["family"] == "Step-Audio-2"


def test_step_audio_chunk_geometry():
    """examples-vllm-stream.py: CHUNK_SIZE 25 (+3 look-ahead), fixed; inference_chunk: conformer keys = all tokens since
    setup (never trimmed), estimator keys = prompt mel + ≤ 100 cached frames + the new 50."""
    c = get_model(SA).cosy
    assert c.hops(250) == [(25 * i, 25, False) for i in range(9)] + [(225, 25, True)]
    assert c.hops(30) == [(0, 25, False), (25, 5, True)]
    ch, ev = flow_chunk_ops(c, 25, 75 + 50, 150 + 100)
    o = {x.name: x for x in ch + ev}
    a = o["flow.encoder.encoders.0.self_attn.qk"]
    assert (a.m, a.n, a.count) == (25, 150, 8)
    u = o["flow.encoder.up_encoders.0.self_attn.qk"]
    assert (u.m, u.n) == (50, 300)
    assert o["flow.encoder.encoders.0.self_attn.linear_pos.weight"].m == 2 * 150 - 1
    e = o["flow.decoder.estimator.blocks.0.attn.qk"]
    assert (e.m, e.n, e.count) == (50, 300, 2 * 8)
    assert o["flow.decoder.estimator.blocks.0.attn.to_q.weight"].m == 2 * 50
    assert o["flow.decoder.estimator.blocks.0.adaLN_modulation.1.weight"].m == 2
    # one chunk reads every flow weight once (+ the estimator again per extra Euler step)
    assert sum(x.w_params for x in ch + ev) == c.flow_params
    assert sum(x.w_params for x in hift_ops(c, 58)) == c.hift_params


def test_step_audio_speech_kpis():
    s = Scenario(model=SA, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=120, out_len=250, audio_s=10.0))
    k = speech_kpis(s, 10.0)
    f = k["split_first"]
    assert k["kind"] == "cosyvoice" and k["stream"] == "cache" and k["hops"] == [25] * 10
    assert abs(k["first_audio_s"] - (f["lm_ttft_s"] + 28 * k["tpot_s"] + f["flow_chunk0_s"] + f["hift_chunk0_s"])) < 1e-12
    assert max(k["flow_s"]) - min(k["flow_s"]) < 0.05 * max(k["flow_s"])     # cached: chunk cost ~ flat in position
    assert k["audio_in"]["prompt_audio_tokens"] == 125
    # the 7B LLM must emit 25 speech tokens / s: on 100T + LPDDR5X it cannot (TPOT ≈ 74 ms → LM RTF ≈ 1.85)
    assert k["lm_rtf"] > 1.5 and k["rtf"] < 0.05
    cv = speech_kpis(Scenario(model="cosyvoice2-0.5b", chip=CHIPS["100T"], mem_id=LP,
                              serving=Serving(batch=1, prompt=120, out_len=250)), 10.0)
    assert cv["stream"] == "recompute" and cv["label"] == "CosyVoice2"
    o = api.api_eval({"scenario": {"model": SA, "serving": {"audio_s": 10, "prompt": 64, "out_len": 60}},
                      "sensitivity": False, "goodput": False})
    assert o["speech"]["label"] == "Step-Audio-2 token2wav"


KA = "kimi-audio-7b-instruct"


def test_kimi_audio_params_exact():
    """HF index 9,766,336,640 = LLM 9,735,124,992 (+ 2,176 bf16 rotary inv_freq buffers, not params) + vq_adaptor
    31,209,472; whisper-large-v3/ encoder 636,968,960; audio_detokenizer 1,584,023,120 (fp32); vocoder 241,118,777."""
    m = get_model(KA)
    h, f, V, kvd = 3584, 18944, 168448, 4 * 128
    layer = (h * h + h) + 2 * (h * kvd + kvd) + h * h + 3 * h * f + 2 * h
    assert m.n_layers == 28 + 6 and m.params() == 3 * V * h + 34 * layer + 2 * h == m.release_params == 9_735_124_992
    assert [l.name for l in m.io_post] == ["mimo_output"] and m.io_post[0].rows == "seq"
    adaptor = (5120 * h + h) + (h * h + h) + 2 * h
    assert adaptor == 31_209_472 and m.vision_params == 636_968_960 + adaptor == m.vision.params
    k = m.kimi
    assert (k.dit_params, k.voc_params) == (1_584_023_120, 241_118_777)
    blk = (3 * 2304 * 2304 + 3 * 2304) + (2304 * 2304 + 2304) + 2 * 2304 * 9216 + 9216 + 2304 + (2304 * 6 * 2304 + 6 * 2304)
    assert k.dit_params == 16 * blk + 16385 * 2304 + (256 * 2304 + 2304) + (2304 * 2304 + 2304) + (80 * 2304 + 2304) \
        + (2304 * 80 + 80) + (2304 * 2 * 2304 + 2 * 2304)
    e = {x["id"]: x for x in list_models()}[KA]
    assert e["domain"] == "omni" and e["family"] == "Kimi-Audio" and e["coverage"] == "partial"
    assert "GLM-4-Voice" in m.coverage_reasons[0]


def test_kimi_audio_input_tokens():
    """whisper.py: per 30 s segment token_len = (L − 1) // 1280 + 1; every segment padded to 3000 mel frames."""
    a = get_model(KA).vision.audio
    assert [a.tokens(s) for s in (1, 10, 30, 45, 61)] == [13, 125, 375, 563, 763]
    assert a.windows(61) == 3 and a.chunks(61) == (3000,) * 3


def test_kimi_detokenizer_plan_and_ops():
    from accel_dse.core.kimi_audio import dit_eval_ops, voc_ops
    k = get_model(KA).kimi
    assert k.evals == 29 and k.delay == 6 and k.mel_rate == 50
    p = k.plan(125)
    assert [c["T"] for c in p] == [120, 168, 168, 168, 68] and [c["kv"] for c in p] == [0, 72, 192, 312, 432]
    assert [c["ret"] for c in p] == [36, 72, 168, 72, 152] and sum(c["ret"] for c in p) == 4 * 125
    assert [c["voc"] for c in p] == [72, 156, 204, 156, 152]
    long = k.plan(500)
    assert max(c["kv"] for c in long) == 900 and sum(c["ret"] for c in long) == 2000
    ops = dit_eval_ops(k, 168, 900)
    g = [o for o in ops if o.kind == "gemm"]
    assert sum(o.w_params for o in g) == k.dit_params - 16385 * 2304       # the semantic table is looked up
    ada = [o for o in g if o.name == "detok.blocks.0.adaLN_modulation.1.weight"][0]
    assert (ada.m, ada.k, ada.n) == (168, 2304, 6 * 2304) and ada.w_bits == 16.0     # c is per frame; bf16
    qk = [o for o in ops if o.name == "detok.blocks.0.attn.qk"][0]
    assert (qk.m, qk.n, qk.count) == (168, 900 + 168, 18)
    v = voc_ops(k, 72)
    assert sum(o.w_params for o in v if o.kind == "gemm") == k.voc_params
    up0 = [o for o in v if o.name.startswith("voc.ups.0.")][0]
    assert (up0.m, up0.k, up0.n) == (72, 2048, 1024 * 9)
    post = [o for o in v if o.name.startswith("voc.conv_post.")][0]
    assert post.m == 72 * 480


def test_kimi_speech_kpis():
    s = Scenario(model=KA, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=120, out_len=250, audio_s=10.0))
    k = speech_kpis(s, 10.0)
    f = k["split_first"]
    assert k["kind"] == "kimi" and k["audio_tokens"] == 125 and k["chunks"] == [30, 30, 30, 30, 5]
    assert abs(k["first_audio_s"] - (f["lm_ttft_s"] + 35 * k["tpot_s"] + f["dit_chunk0_s"] + f["voc_chunk0_s"])) < 1e-12
    assert abs(k["first_audio_released_s"] - k["first_audio_s"] - 95 * k["tpot_s"]) < 1e-9
    assert abs(k["lm_rtf"] - 12.5 * k["tpot_s"]) < 1e-12 and k["first_chunk_audio_s"] == 0.72
    # DiT weight-bound at batch 1: 30 reads of ~3.1 GB bf16 per chunk dominate; chunk 0 (120 rows) ≈ later (168)
    assert abs(k["dit_s"][0] - k["dit_s"][1]) < 0.05 * k["dit_s"][1] and max(k["voc_s"]) < 0.1 * min(k["dit_s"])
    assert k["audio_in"]["prompt_audio_tokens"] == 125 and k["fits"]
    o = api.api_eval({"scenario": {"model": KA, "serving": {"audio_s": 10, "prompt": 64, "out_len": 60}},
                      "sensitivity": False, "goodput": False})
    assert o["speech"]["kind"] == "kimi"
    js = W.read()                                      # 0.286: token match
    assert W.has(js, "p.kind === 'kimi'") and W.mentions(js, "first_audio_released_s")
