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

Nothing here is fitted: both phases are ordinary ``evaluate`` calls on the scenario; the out_len and audio_out_s
defaults are the caller's workload assumption 「假设」.
"""
from __future__ import annotations

import math

from .evaluate import evaluate
from .scenario import Scenario


def speech_kpis(scn: Scenario, audio_out_s: float = 10.0) -> dict:
    sv = scn.serving
    pre = evaluate(scn.replace("serving.phase", "prefill"))
    m = pre.model
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
