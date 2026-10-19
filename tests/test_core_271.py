"""0.271 speech round 2 (docs/MODEL.md §216): CosyVoice2 / Fun-CosyVoice3 (LM → 25 Hz tokens → chunked flow → HiFT) and
Qwen2.5-Omni-7B — Thinker (Qwen2.5-7B) + Whisper-v3 audio tower (200-frame
chunks, valid frames only) + Talker + token2wav (DiT RK4 × CFG, 72-frame streaming chunks) + BigVGAN; exact params
against the released safetensors header, op-count hand checks, the 0.269 re-read pin on the DiT, speech KPIs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.model import load_release
from accel_dse.core.cosyvoice import cosy_pin, flow_ops, hift_ops
from accel_dse.core.omni import bigvgan_ops, dit_eval_ops, omni_pin, omni_spec, talker_ops
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.vision import audio_ops

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
OM = "qwen2.5-omni-7b"
REL = load_release("Qwen/Qwen2.5-Omni-7B")


def _header_total() -> int:
    return sum(v["params"] for r in REL["roles"].values() for v in r.values())


def test_exact_params_vs_header():
    m = get_model(OM)
    o = m.omni
    assert m.params() == REL["params_llm"] == 7_615_616_512          # Thinker = Qwen2.5-7B (q / k / v bias, untied)
    assert m.vision.audio.params == 639_647_232                       # audio tower (no stored positions) + bos/eos rows
    assert m.vision.extra == 676_550_144 and m.vision.params == REL["params_vision"]   # ViT stored only
    assert o.talker_params == REL["params_talker"] == 1_351_360_256
    assert o.dit_params == 333_607_792 and o.bigvgan_params == 115_443_504 and o.ecapa_params == 1_729_280
    assert o.t2w_params == REL["params_token2wav"]
    assert REL["params_llm"] + REL["params_vision"] + REL["params_talker"] + REL["params_token2wav"] == _header_total()
    assert REL["roles"]["token2wav"].keys() == {"fp32"} and REL["roles"]["talker"].keys() == {"bf16"}
    assert m.coverage == "partial" and any("视觉塔" in r for r in m.coverage_reasons)


def test_audio_tower_tokens_and_ops():
    a = get_model(OM).vision.audio
    assert a.chunks(10.0) == (200,) * 5 and a.tokens(10.0) == 250 and a.tokens(1.0) == 25
    assert a.mel_frames(40.0) == 3000 and a.tokens(40.0) == 750      # processor truncation at 30 s
    ops = {x.name: x for x in audio_ops(a, 10.0, 1)}
    assert ops["aud.0.qkv"].m == 500 and ops["aud.0.qk"].m == 100 and ops["aud.0.qk"].count == 5 * 20
    assert ops["aud.proj.linear0"].m == 250 and ops["aud.proj.linear0"].n == 3584


def test_streaming_geometry_and_rates():
    o = get_model(OM).omni
    assert o.hop == 240 and o.mel_per_s == 100 and o.codec_per_s == 50
    assert o.chunk == 72 and o.past == 48 and o.future == 24 and o.evals == 36 and o.cfg_mult == 2
    assert [o.dit_window(i) for i in range(3)] == [96, 144, 144] and o.first_codec == 48
    assert o.voc_frames(0) == 72 and o.voc_frames(1) == 112 and abs(o.first_audio_s - 0.52) < 1e-12


def test_op_hand_counts():
    o = get_model(OM).omni
    T, d, ff = 144, o.d, o.ff
    ops = dit_eval_ops(o, T)
    blk = sum(x.flops for x in ops if x.kind == "gemm" and x.name.split(".")[-1] in ("qkv", "o", "ff1", "ff2"))
    assert blk == 2 * (2 * T) * o.depth * (4 * d * d + 2 * d * ff)
    w = sum(x.w_params for x in ops if x.kind == "gemm")
    assert w == o.dit_run_params - (d // o.heads) // 2                # every run weight + bias once per evaluation
    assert abs(sum(x.w_params for x in dit_eval_ops(o, T, 0.25) if x.kind == "gemm") - 0.25 * w) < 200
    M = 112
    v = bigvgan_ops(o, M)
    stages = o._voc_stages()
    L, rb = M, 0
    for ci, co, k, r in stages:
        L *= r
        rb += 2 * 3 * (3 + 7 + 11) * co * co * L                      # 2 convs × 3 dilations × Σ kernels per AMP stage
    assert sum(x.flops for x in v if ".conv" in x.name and ".k" in x.name) == 2 * rb
    assert sum(x.w_params for x in v if x.kind == "gemm") == o.bigvgan_params - sum(
        3 * 6 * 2 * co for _, co, _, _ in stages) - 2 * 24                # all but the SnakeBeta α / β vectors
    t = talker_ops(o, 4, 1, 100)
    assert sum(x.w_params for x in t if x.kind == "gemm") == o.talker_run_params - o.t_layers * 2 * 896 - 896   # − norms
    assert sum(x.kv_read for x in t if x.kind == "attn") == 4 * 100 * 24 * 2 * 512 * 2


def test_dit_reread_pin():
    o = get_model(OM).omni
    W = o.dit_run_params * 4.0
    assert abs(omni_pin(o, 64 * 2 ** 20) - (1 - (64 - 2) * 2 ** 20 / W)) < 1e-12
    assert omni_pin(o, 4 * W) == 0.0 and omni_pin(o, 0) == 1.0


def test_speech_kpis_omni():
    s = Scenario(model=OM, chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=4, prompt=64, out_len=60, audio_s=10.0))
    k = speech_kpis(s, 10.0)
    assert k["kind"] == "omni" and k["codec_tokens"] == 500 and k["audio_in"]["prompt_audio_tokens"] == 250
    assert abs(k["first_audio_s"] - sum(k["split_first"].values())) < 1e-12
    assert abs(k["split_first"]["talker_steps_s"] - 48 * k["talker_step_s"]) < 1e-12
    busy = 50 * k["talker_step_s"] + 4 * (100 / 72) * (k["dit_chunk_s"] + k["voc_chunk_s"])
    assert abs(k["rtf"] - busy) < 1e-12 and k["first_audio_last_s"] > k["first_audio_s"]
    rt = k["streams_realtime"]
    assert rt >= 1
    a = speech_kpis(s.replace("serving.batch", rt), 10.0)
    b = speech_kpis(s.replace("serving.batch", rt + 1), 10.0)
    assert a["rtf"] <= 1.0 < b["rtf"]
    # more SRAM pins more of the DiT → a cheaper chunk (same chip otherwise)
    big = speech_kpis(s.replace("chip", CHIPS["1P"].__class__(**{**CHIPS["1P"].__dict__, "sram_mib": 4096.0})), 10.0)
    assert big["dit_pin_miss"] == 0.0 and big["dit_chunk_s"] < k["dit_chunk_s"]


def test_catalog_api_ui():
    e = next(x for x in list_models() if x["id"] == OM)
    assert e["domain"] == "omni" and e["family"] == "Qwen2.5-Omni" and e["coverage"] == "partial"
    o = api.api_eval({"scenario": {"model": OM, "serving": {"prompt": 32, "out_len": 20, "ctx": 42}},
                      "audio_out_s": 2, "sensitivity": False, "goodput": False})
    assert o["speech"]["kind"] == "omni" and o["speech"]["codec_tokens"] == 100
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert "p.kind === 'omni'" in js and "omni: '全模态'" in js
    repos = (ROOT / "scripts/release_repos.txt").read_text().split()
    assert "Qwen/Qwen2.5-Omni-7B" in repos


CV2, CV3 = "cosyvoice2-0.5b", "fun-cosyvoice3-0.5b-2512"


def test_cosyvoice_params_exact_vs_pt_state_dicts():
    for mid, rel_id, flow, hift, voc in ((CV2, "FunAudioLLM/CosyVoice2-0.5B", 112_549_360, 20_821_295, 6564),
                                         (CV3, "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", 332_257_120, 20_779_887, 6761)):
        rel = load_release(rel_id)
        m = get_model(mid)
        c = m.cosy
        assert m.vocab == voc and not m.tie_embeddings
        assert c.flow_params == rel["params_flow"] == flow and c.hift_params == rel["params_vocoder"] == hift
        # LM: Qwen2.5-0.5B layers + speech table + llm_decoder + text embedding / text lm_head as lookup tables
        assert m.lookup_params == 2 * 151936 * 896 + (1792 if mid == CV2 else 0)
        assert abs(rel["params_llm"] - m.params()) == (6564 if mid == CV2 else 0)     # llm_decoder bias (CosyVoice2)
        assert m.coverage == "full"
        ch, ev = flow_ops(c, 103)
        assert sum(o.w_params for o in ch + ev if o.kind == "gemm") == flow            # every flow weight read once
        assert sum(o.w_params for o in hift_ops(c, 50) if o.kind == "gemm") == hift


def test_cosyvoice_streaming_hops_and_recompute():
    c = get_model(CV2).cosy
    assert c.hops(250) == [(0, 25, False), (25, 50, False), (75, 100, False), (175, 75, True)]
    assert c.hops(250, warm=True) == [(0, 100, False), (100, 100, False), (200, 50, True)]
    assert replace_prompt(c, 60).hops(100)[0] == (0, 40, False)            # prompt padded to a multiple of 25
    # flow recomputes over every token so far: estimator rows grow with the chunk's total tokens
    a = sum(o.flops for o in flow_ops(c, 103)[1] if o.kind == "gemm")
    b = sum(o.flops for o in flow_ops(c, 206)[1] if o.kind == "gemm")
    assert 1.95 < b / a < 2.05
    c3 = get_model(CV3).cosy
    est = sum(math.prod(s) for n, s in c3.tensors if n.startswith("flow.decoder.estimator.") and len(s) == 2
              and ".transformer_blocks." in n and "attn_norm" not in n)
    blk = sum(o.flops for o in flow_ops(c3, 100)[1] if o.kind == "gemm" and ".transformer_blocks." in o.name
              and "attn_norm" not in o.name)
    assert blk == 2 * (2 * 200) * est                                            # CFG 2 × 200 mel frames
    assert cosy_pin(c3, 4 * c3.flow_params * 4.0) == 0.0


def test_cosyvoice_speech_kpis():
    s = Scenario(model=CV3, chip=CHIPS["1P"], mem_id=LP, serving=Serving(batch=2, prompt=120, out_len=250))
    k = speech_kpis(s, 10.0)
    assert k["kind"] == "cosyvoice" and k["speech_tokens"] == 250 and k["hops"] == [25, 50, 100, 75]
    assert abs(k["first_audio_s"] - sum(k["split_first"].values())) < 1e-12
    assert abs(k["split_first"]["lm_tokens_s"] - 28 * k["tpot_s"]) < 1e-12
    assert k["first_audio_warm_s"] > k["first_audio_s"]                            # warm instance: first hop 100
    assert abs(k["rtf"] - 2 * k["speech_card_s_per_request"] / 10.0) < 1e-12
    assert abs(k["lm_rtf"] - 25 * k["tpot_s"]) < 1e-12
    h = k["hift_s"]
    assert h[-1] > h[0] and h == sorted(h)                                        # CosyVoice3 re-vocodes all mel
    o = api.api_eval({"scenario": {"model": CV2, "serving": {"prompt": 32, "out_len": 50}}, "audio_out_s": 2,
                      "sensitivity": False, "goodput": False})
    assert o["speech"]["kind"] == "cosyvoice" and o["speech"]["version"] == 2
    e = {x["id"]: x for x in list_models()}
    assert e[CV2]["domain"] == e[CV3]["domain"] == "tts" and e[CV3]["family"] == "Fun-CosyVoice3"
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert "p.kind === 'cosyvoice'" in js


def replace_prompt(c, n):
    from dataclasses import replace
    return replace(c, prompt_tokens=n)
