"""0.270 speech serving convention (docs/MODEL.md §215): speech workloads are costed per audio second, end to end.

ASR (audio in → text out; models with an audio encoder, ``serving.audio_s`` > 0):
    one request = ``audio_s`` seconds of audio + ``prompt`` text tokens → ``out_len`` text tokens
    prefill pass (encoder windows + prompt; ``ctx`` 0)          → first-token latency T_ft
    decode at the mid-generation context (prompt + audio positions + out_len / 2) → TPOT
    time per request  T = T_ft + out_len · TPOT                (static batch, phases serial 「近似」)
    RTF = T / audio_s ;  audio s / s per card = batch · audio_s / T / cards
TTS (text in → audio out; models with a TTS head): ``audio_out_s`` seconds out = ⌈audio_out_s / s_per_step⌉ decode
    steps; first-audio latency = T_ft + one step (the streaming decoder emits each step's chunk); RTF = steps · TPOT /
    audio_out_s ; audio-out s / s per card = batch · audio_out_s / (T_ft + steps · TPOT) / cards.

Omni (Qwen2.5-Omni, 0.271, MODEL §216; core/omni.py): Thinker (the catalog LLM, its audio tower when audio_s > 0) →
Talker (50 codec tokens / audio s) → token2wav DiT chunks (36 RK4 evaluations × CFG per 72-frame chunk, batch 1) →
BigVGAN.  Placement 「假设」: Talker + token2wav on one extra card of the same chip per data-parallel replica (vLLM-Omni
runs them as separate pipeline stages), serving that replica's streams; the Thinker scenario is unchanged.
    first audio = Thinker T_ft + one Thinker TPOT (the Talker needs the first reply token 「假设」) + Talker prefill over the
                  prompt + 48 Talker steps (codec tokens of DiT window 0) + DiT chunk 0 (96 frames) + BigVGAN (72 frames)
    speech-card seconds per output-audio second for b streams = 50 · t_talker_step(b) + b · (100 / 72) · (t_DiT + t_voc)
    (= RTF of every stream; real time when ≤ 1; DiT weights pinned by the 0.269 re-read rule when SRAM allows)

Nothing here is fitted: both phases are ordinary ``evaluate`` calls on the scenario; the out_len and audio_out_s
defaults are the caller's workload assumption 「假设」.
"""
from __future__ import annotations

import math

from .evaluate import evaluate
from .scenario import Scenario


def _omni_time(ops: list, sys, org: str, af: str) -> float:
    """evaluate._component_time + the K/V and lookup bytes the component path leaves out (Talker K/V cache)."""
    from .evaluate import _component_time
    c = _component_time([(ops, 1)], sys, org, af)
    kv = sum(o.kv_read + o.kv_write for o in ops)
    return c["s"] + kv / (sys.dram_GBps * 1e9)


def _omni_kpis(scn: Scenario, pre, audio_out_s: float) -> dict:
    from .hardware import System
    from .ir import _cdiv
    from .omni import bigvgan_ops, dit_eval_ops, omni_pin, talker_ops
    sv, lay, m = scn.serving, scn.layout, pre.model
    o = m.omni
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + sv.out_len // 2))
    b = _cdiv(sv.batch, lay.dp)
    P = pre.scenario.serving.prompt         # text + audio positions (expand_images); the Talker prefills over all of them
    n_codec = max(1, math.ceil(audio_out_s * o.codec_per_s - 1e-9))
    t_pre = _omni_time(talker_ops(o, b, P, 0), sys, org, "bf16")

    def t_step(bb: int) -> float:
        return _omni_time(talker_ops(o, bb, 1, P + n_codec // 2), sys, org, "bf16")

    miss = omni_pin(o, sys.chip.sram_bytes)

    def t_dit(T: int) -> float:
        return (_omni_time(dit_eval_ops(o, T), sys, org, "fp32")
                + (o.evals - 1) * _omni_time(dit_eval_ops(o, T, miss), sys, org, "fp32"))
    ts = t_step(b)
    d0, d1 = t_dit(o.dit_window(0)), t_dit(o.dit_window(1))
    v0, v1 = (_omni_time(bigvgan_ops(o, o.voc_frames(i)), sys, org, "fp32") for i in (0, 1))
    first = pre.ttft + dec.tpot + t_pre + o.first_codec * ts + d0 + v0
    per_chunk = d1 + v1
    chunks_per_s = o.mel_per_s / o.chunk
    busy = o.codec_per_s * ts + b * chunks_per_s * per_chunk
    rt = 0
    lo, hi = 0, 1
    f = lambda bb: o.codec_per_s * t_step(bb) + bb * chunks_per_s * per_chunk <= 1.0
    while hi <= 4096 and f(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1 and lo < 4096:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    rt = lo
    tot = first + max(audio_out_s * busy, sv.out_len * dec.tpot)
    need = o.stored_bytes + b * o.talker_kv_bytes(P + n_codec)
    out = {"kind": "omni", "audio_out_s": audio_out_s, "out_len": sv.out_len, "codec_tokens": n_codec,
           "codec_per_s": o.codec_per_s, "chunk_s": o.chunk_s, "dit_evals": o.evals, "cfg": o.cfg_mult,
           "first_audio_s": first, "first_audio_last_s": first + (b - 1) * (d0 + v0),
           "first_chunk_audio_s": o.first_audio_s,
           "split_first": {"thinker_ttft_s": pre.ttft, "thinker_tpot_s": dec.tpot, "talker_prefill_s": t_pre,
                           "talker_steps_s": o.first_codec * ts, "dit_chunk0_s": d0, "voc_chunk0_s": v0},
           "ttft_s": pre.ttft, "tpot_s": dec.tpot, "talker_step_s": ts, "dit_chunk_s": d1, "voc_chunk_s": v1,
           "dit_pin_miss": miss, "speech_card_s_per_audio_s": busy, "rtf": busy, "streams_per_card": b,
           "streams_realtime": rt, "total_s": tot, "speech_cards": lay.dp,
           "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / (pre.scenario.layout.cards + lay.dp),
           "speech_card_GB": need / 1e9, "speech_card_fits": need <= sys.dram_bytes,
           "fits": pre.fits and dec.fits and need <= sys.dram_bytes}
    au = getattr(getattr(m, "vision", None), "audio", None)
    if au is not None and sv.audio_s:
        out["audio_in"] = {"audio_s": sv.audio_s, "prompt_audio_tokens": au.prompt_tokens(sv.audio_s),
                           "encoder_s": ((pre.vision or {}).get("audio") or {}).get("s", 0.0)}
    return out


def _cosy_kpis(scn: Scenario, pre, audio_out_s: float) -> dict:
    """0.271 CosyVoice2 / 3 (core/cosyvoice.py): the LM (catalog model) decodes 25 speech tokens / s on the scenario's
    cards; flow + HiFT run per streamed chunk on a speech card per replica 「假设」, batch 1 (one stream after another).
        first audio = T_ft + (hop₀ + 3 look-ahead) · TPOT + flow(prompt + hop₀ + 3 tokens) + HiFT(chunk 0)
        speech-card s per request = Σ_chunks flow(prompt + offset + hop (+3)) + HiFT(new mel | all mel so far)
        RTF (speech card, b streams) = b · that / audio_out_s ;  LM RTF = 25 · TPOT"""
    from .cosyvoice import cosy_pin, flow_ops, hift_ops, scaled
    from .hardware import System
    from .ir import _cdiv
    sv, lay, m = scn.serving, scn.layout, pre.model
    c = m.cosy
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    n_tok = max(1, math.ceil(audio_out_s * c.token_rate - 1e-9))
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + n_tok // 2))
    miss = cosy_pin(c, sys.chip.sram_bytes)
    from .evaluate import _component_time

    def t_flow(N: int) -> float:
        ch, ev = flow_ops(c, N)
        g = [(ch, 1), (ev, 1)] + ([(scaled(ev, miss), c.steps - 1)] if c.steps > 1 else [])
        return _component_time(g, sys, org, "fp32")["s"] + sum(o.kv_read for o in ch) / (sys.dram_GBps * 1e9)

    def t_hift(M: int) -> float:
        return _component_time([(hift_ops(c, M), 1)], sys, org, "fp32")["s"]

    def chunks(warm: bool) -> list[tuple[float, float, int]]:
        out = []
        for off, hop, fin in c.hops(n_tok, warm):
            N = c.prompt_tokens + off + hop + (0 if fin else c.lookahead)
            new_mel = hop * c.mel_ratio
            M = (off + hop) * c.mel_ratio if c.version == 3 else new_mel + (c.mel_cache if off else 0)
            out.append((t_flow(N), t_hift(M), hop))
        return out
    cold = chunks(False)
    busy = sum(f + h for f, h, _ in cold)
    h0 = cold[0][2]
    first = pre.ttft + (h0 + c.lookahead) * dec.tpot + cold[0][0] + cold[0][1]
    warm = chunks(True)
    first_w = pre.ttft + (warm[0][2] + c.lookahead) * dec.tpot + warm[0][0] + warm[0][1]
    b = _cdiv(sv.batch, lay.dp)
    lm_s = pre.ttft + n_tok * dec.tpot
    tot = max(lm_s + cold[-1][0] + cold[-1][1], pre.ttft + (h0 + c.lookahead) * dec.tpot + b * busy)
    return {"kind": "cosyvoice", "version": c.version, "audio_out_s": audio_out_s, "speech_tokens": n_tok,
            "token_rate": c.token_rate, "prompt_speech_tokens": c.prompt_tokens, "flow_steps": c.steps, "cfg": c.cfg_mult,
            "hops": [hop for _, _, hop in cold], "first_audio_s": first, "first_audio_warm_s": first_w,
            "split_first": {"lm_ttft_s": pre.ttft, "lm_tokens_s": (h0 + c.lookahead) * dec.tpot,
                            "flow_chunk0_s": cold[0][0], "hift_chunk0_s": cold[0][1]},
            "ttft_s": pre.ttft, "tpot_s": dec.tpot, "lm_rtf": c.token_rate * dec.tpot,
            "flow_s": [f for f, _, _ in cold], "hift_s": [h for _, h, _ in cold], "flow_pin_miss": miss,
            "speech_card_s_per_request": busy, "rtf": b * busy / audio_out_s, "streams_per_card": b,
            "streams_realtime": int(audio_out_s // busy) if busy > 0 else 0, "total_s": tot, "speech_cards": lay.dp,
            "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / (pre.scenario.layout.cards + lay.dp),
            "speech_card_GB": c.stored_bytes / 1e9, "speech_card_fits": c.stored_bytes <= sys.dram_bytes,
            "fits": pre.fits and dec.fits and c.stored_bytes <= sys.dram_bytes}


def speech_kpis(scn: Scenario, audio_out_s: float = 10.0) -> dict:
    sv = scn.serving
    pre = evaluate(scn.replace("serving.phase", "prefill"))
    m = pre.model
    if getattr(m, "omni", None) is not None:
        return _omni_kpis(scn, pre, audio_out_s)
    if getattr(m, "cosy", None) is not None:
        return _cosy_kpis(scn, pre, audio_out_s)
    ts = getattr(m, "tts", None)
    au = getattr(getattr(m, "vision", None), "audio", None)
    if ts is not None:
        steps = max(1, math.ceil(audio_out_s / ts.s_per_step - 1e-9))
        dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + steps // 2))
        tot = pre.ttft + steps * dec.tpot
        return {"kind": "tts", "audio_out_s": audio_out_s, "steps": steps, "first_audio_s": pre.ttft + dec.tpot,
                "ttft_s": pre.ttft, "tpot_s": dec.tpot, "total_s": tot, "rtf": steps * dec.tpot / audio_out_s,
                "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / pre.scenario.layout.cards,
                "fits": pre.fits and dec.fits}
    if au is None or not sv.audio_s:
        raise ValueError("speech_kpis：模型没有音频编码器 / TTS 头，或 serving.audio_s = 0")
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + sv.out_len // 2))
    tot = pre.ttft + sv.out_len * dec.tpot
    return {"kind": "asr", "audio_s": sv.audio_s, "out_len": sv.out_len, "first_token_s": pre.ttft,
            "encoder_s": ((pre.vision or {}).get("audio") or {}).get("s", 0.0), "tpot_s": dec.tpot, "total_s": tot,
            "rtf": tot / sv.audio_s, "audio_s_per_s_card": sv.batch * sv.audio_s / tot / pre.scenario.layout.cards,
            "enc_positions": au.enc_positions(sv.audio_s), "prompt_audio_tokens": au.prompt_tokens(sv.audio_s),
            "fits": pre.fits and dec.fits}
