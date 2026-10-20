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
from functools import lru_cache

from .evaluate import evaluate
from .scenario import Scenario


def _omni_time(ops: list, sys, org: str, af: str) -> float:
    """evaluate._component_time + the K/V and lookup bytes the component path leaves out (Talker K/V cache)."""
    from .evaluate import _component_time
    c = _component_time([(ops, 1)], sys, org, af)
    kv = sum(o.kv_read + o.kv_write for o in ops)
    return c["s"] + kv / (sys.dram_GBps * 1e9)


def _omni_kpis(scn: Scenario, pre, audio_out_s: float, lm_only: bool = False) -> dict:
    from .hardware import System
    from .ir import _cdiv
    from .omni import bigvgan_ops, dit_eval_ops, omni_pin, omni_slc_plan, scale_weights, talker_ops, talker_weight_bytes
    sv, lay, m = scn.serving, scn.layout, pre.model
    o = m.omni
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + sv.out_len // 2))
    b = _cdiv(sv.batch, lay.dp)
    P = pre.scenario.serving.prompt         # text + audio positions (expand_images); the Talker prefills over all of them
    n_codec = max(1, math.ceil(audio_out_s * o.codec_per_s - 1e-9))
    if lm_only:              # 0.277: thinker load + fit verdict only (same expressions as below)
        need = o.stored_bytes + b * o.talker_kv_bytes(P + n_codec)
        return {"kind": "omni", "lm_stages": {"thinker": sv.out_len * dec.tpot / audio_out_s},
                "fits": pre.fits and dec.fits and need <= sys.dram_bytes}
    t_pre = _omni_time(talker_ops(o, b, P, 0), sys, org, "bf16")

    miss = omni_pin(o, sys.chip.sram_bytes)
    # 0.272: an SLC with slc_policy "pin" holds the DiT SRAM-miss weights, then the Talker weights (omni_slc_plan);
    # hits are served at slc_GBps on a port parallel to DRAM (schedule.t_slc): t = max(component time with the DRAM
    # share of the weights, SLC bytes / slc_GBps) 「近似」.  No SLC (or another policy) → identical to 0.271.
    slc = sys.chip.slc_bytes if sys.chip.slc_policy == "pin" else 0.0
    slc_bw = sys.chip.slc_GBps * 1e9
    wt, wd = talker_weight_bytes(o), o.dit_run_params * 4.0

    def t_step(bb: int) -> float:
        h = omni_slc_plan(o, miss, slc, bb)[1]
        c = _omni_time(scale_weights(talker_ops(o, bb, 1, P + n_codec // 2), 1.0 - h), sys, org, "bf16")
        return max(c, h * wt / slc_bw) if h else c

    def t_dit(T: int, bb: int = b) -> float:
        h = omni_slc_plan(o, miss, slc, bb)[0]
        first = _omni_time(dit_eval_ops(o, T, 1.0 - miss * h), sys, org, "fp32")
        rr = _omni_time(dit_eval_ops(o, T, miss * (1.0 - h)), sys, org, "fp32")
        if h:
            first, rr = max(first, miss * h * wd / slc_bw), max(rr, miss * h * wd / slc_bw)
        return first + (o.evals - 1) * rr
    ts = t_step(b)
    d0, d1 = t_dit(o.dit_window(0)), t_dit(o.dit_window(1))
    v0, v1 = (_omni_time(bigvgan_ops(o, o.voc_frames(i)), sys, org, "fp32") for i in (0, 1))
    first = pre.ttft + dec.tpot + t_pre + o.first_codec * ts + d0 + v0
    per_chunk = d1 + v1
    chunks_per_s = o.mel_per_s / o.chunk
    busy = o.codec_per_s * ts + b * chunks_per_s * per_chunk
    rt = 0
    lo, hi = 0, 1
    f = lambda bb: o.codec_per_s * t_step(bb) + bb * chunks_per_s * ((t_dit(o.dit_window(1), bb) if slc else d1) + v1) <= 1.0
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
           "dit_pin_miss": miss,
           "slc": {"mib": slc / 2 ** 20, "dit_hit": omni_slc_plan(o, miss, slc, b)[0],
                   "talker_hit": omni_slc_plan(o, miss, slc, b)[1], "talker_weight_GB": wt / 1e9,
                   "talker_s_per_audio_s": o.codec_per_s * ts, "dit_s_per_audio_s": b * chunks_per_s * d1},
           "speech_card_s_per_audio_s": busy, "rtf": busy, "streams_per_card": b,
           "streams_realtime": rt, "total_s": tot, "speech_cards": lay.dp,
           "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / (pre.scenario.layout.cards + lay.dp),
           "speech_card_GB": need / 1e9, "speech_card_fits": need <= sys.dram_bytes,
           "fits": pre.fits and dec.fits and need <= sys.dram_bytes}
    au = getattr(getattr(m, "vision", None), "audio", None)
    if au is not None and sv.audio_s:
        out["audio_in"] = {"audio_s": sv.audio_s, "prompt_audio_tokens": au.prompt_tokens(sv.audio_s),
                           "encoder_s": ((pre.vision or {}).get("audio") or {}).get("s", 0.0)}
    return out


def _cosy_kpis(scn: Scenario, pre, audio_out_s: float, lm_only: bool = False) -> dict:
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
    if lm_only:              # 0.277: LM load + fit verdict only
        return {"kind": "cosyvoice", "lm_stages": {"lm": c.token_rate * dec.tpot},
                "fits": pre.fits and dec.fits and c.stored_bytes <= sys.dram_bytes}
    miss = cosy_pin(c, sys.chip.sram_bytes)
    from .evaluate import _component_time

    def t_flow(N: int) -> float:
        ch, ev = flow_ops(c, N)
        g = [(ch, 1), (ev, 1)] + ([(scaled(ev, miss), c.steps - 1)] if c.steps > 1 else [])
        return _component_time(g, sys, org, "fp32")["s"] + sum(o.kv_read for o in ch) / (sys.dram_GBps * 1e9)

    def t_hift(M: int) -> float:
        return _component_time([(hift_ops(c, M), 1)], sys, org, "fp32")["s"]

    def t_flow_cached(N: int, off: int, cache: int) -> float:
        from .cosyvoice import flow_chunk_ops
        ch, ev = flow_chunk_ops(c, N, off, cache)
        g = [(ch, 1), (ev, 1)] + ([(scaled(ev, miss), c.steps - 1)] if c.steps > 1 else [])
        return _component_time(g, sys, org, "fp32")["s"] + sum(o.kv_read for o in ch) / (sys.dram_GBps * 1e9)

    def chunks(warm: bool) -> list[tuple[float, float, int]]:
        out = []
        if c.stream == "cache":
            # 0.273 Step-Audio-2: setup_cache on the prompt (+3) once per voice (not per request 「假设」); chunk i runs
            # its own tokens over the conformer cache (prompt + earlier chunks; ``warm``: one earlier 10 s response
            # already in it — the cache is never trimmed) and the estimator cache (prompt mel + ≤ est_keep frames)
            base = c.prompt_tokens + (n_tok if warm else 0)
            pm = c.prompt_tokens * c.mel_ratio
            for i, (off, hop, fin) in enumerate(c.hops(n_tok)):
                cache = pm + min(c.est_keep, (off + (n_tok if warm else 0)) * c.mel_ratio)
                M = hop * c.mel_ratio + c.mel_cache
                out.append((t_flow_cached(hop, base + off, cache), t_hift(M), hop))
            return out
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
    au = getattr(getattr(m, "vision", None), "audio", None)
    extra = {"label": c.label or f"CosyVoice{c.version}", "stream": c.stream}
    if au is not None and sv.audio_s:          # 0.273 Step-Audio-2: speech in + speech out
        extra["audio_in"] = {"audio_s": sv.audio_s, "prompt_audio_tokens": au.prompt_tokens(sv.audio_s),
                             "encoder_s": ((pre.vision or {}).get("audio") or {}).get("s", 0.0)}
    return {**extra, "kind": "cosyvoice", "version": c.version, "audio_out_s": audio_out_s, "speech_tokens": n_tok,
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


def _kimi_kpis(scn: Scenario, pre, audio_out_s: float, lm_only: bool = False) -> dict:
    """0.273 Kimi-Audio (core/kimi_audio.py): the LM (catalog model, 28 + 6 layers, two heads) emits one audio token per
    step at 12.5 / s after ``delay`` blank steps; the detokenizer (DiT + BigVGAN) runs per chunk on a speech card per
    replica 「假设」, batch 1.
        first audio (streaming the chunk API 「假设」) = T_ft + (delay + 30 − 1) · TPOT + DiT chunk 0 (29 evals on 120
                      frames + 1 commit eval on 72) + BigVGAN(72 frames) — returns 36 frames = 0.72 s of audio
        first audio (released kimia.py: generate everything, then detokenize) = T_ft + (delay + n − 1) · TPOT + chunk 0
        speech-card s per request = Σ_chunks DiT (evals + commit) + BigVGAN ;  RTF = b · that / audio_out_s
        LM RTF = 12.5 · TPOT"""
    from .evaluate import _component_time
    from .hardware import System
    from .ir import _cdiv
    from .kimi_audio import dit_eval_ops, kimi_pin, scaled, voc_ops
    sv, lay, m = scn.serving, scn.layout, pre.model
    k = m.kimi
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    n_tok = max(1, math.ceil(audio_out_s * k.token_rate - 1e-9))
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", sv.prompt + k.delay + n_tok // 2))
    if lm_only:              # 0.277: LM load + fit verdict only
        need = k.stored_bytes + _cdiv(sv.batch, lay.dp) * k.kv_bytes(k.max_kv)
        return {"kind": "kimi", "lm_stages": {"lm": k.token_rate * dec.tpot},
                "fits": pre.fits and dec.fits and need <= sys.dram_bytes}
    miss = kimi_pin(k, sys.chip.sram_bytes)
    bw = sys.dram_GBps * 1e9

    def t_eval(T: int, kv: int, first: bool) -> float:
        ops = dit_eval_ops(k, T, kv)
        # K/V of the cached frames re-read by every evaluation (bf16, 16 blocks) 「推导」
        return _component_time([(ops if first else scaled(ops, miss), 1)], sys, org, "bf16")["s"] + k.kv_bytes(kv) / bw

    def t_voc(M: int) -> float:
        return _component_time([(voc_ops(k, M), 1)], sys, org, "bf16")["s"]
    rows = []
    for c in k.plan(n_tok):
        dit = t_eval(c["T"], c["kv"], True) + (k.evals - 1) * t_eval(c["T"], c["kv"], False)
        if c["commit"]:
            dit += t_eval(c["keep"], c["kv"], False)
        rows.append((dit, t_voc(c["voc"]), c))
    busy = sum(d + v for d, v, _ in rows)
    lead = k.delay + rows[0][2]["tokens"] - 1
    first = pre.ttft + lead * dec.tpot + rows[0][0] + rows[0][1]
    first_rel = pre.ttft + (k.delay + n_tok - 1) * dec.tpot + rows[0][0] + rows[0][1]
    b = _cdiv(sv.batch, lay.dp)
    lm_s = pre.ttft + (k.delay + n_tok - 1) * dec.tpot
    tot = max(lm_s + rows[-1][0] + rows[-1][1], pre.ttft + lead * dec.tpot + b * busy)
    need = k.stored_bytes + b * k.kv_bytes(k.max_kv)
    out = {"kind": "kimi", "label": k.label, "audio_out_s": audio_out_s, "audio_tokens": n_tok,
           "token_rate": k.token_rate, "delay_steps": k.delay, "dit_evals": k.evals, "chunk_tokens": k.chunk,
           "look_ahead_tokens": k.look_ahead, "chunks": [c["tokens"] for _, _, c in rows],
           "dit_rows": [c["T"] for _, _, c in rows], "dit_kv_frames": [c["kv"] for _, _, c in rows],
           "voc_frames": [c["voc"] for _, _, c in rows], "ret_frames": [c["ret"] for _, _, c in rows],
           "first_audio_s": first, "first_audio_released_s": first_rel,
           "first_chunk_audio_s": rows[0][2]["ret"] / k.mel_rate,
           "split_first": {"lm_ttft_s": pre.ttft, "lm_tokens_s": lead * dec.tpot, "dit_chunk0_s": rows[0][0],
                           "voc_chunk0_s": rows[0][1]},
           "ttft_s": pre.ttft, "tpot_s": dec.tpot, "lm_rtf": k.token_rate * dec.tpot,
           "dit_s": [d for d, _, _ in rows], "voc_s": [v for _, v, _ in rows], "dit_pin_miss": miss,
           "speech_card_s_per_request": busy, "rtf": b * busy / audio_out_s, "streams_per_card": b,
           "streams_realtime": int(audio_out_s // busy) if busy > 0 else 0, "total_s": tot, "speech_cards": lay.dp,
           "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / (pre.scenario.layout.cards + lay.dp),
           "speech_card_GB": need / 1e9, "speech_card_fits": need <= sys.dram_bytes,
           "fits": pre.fits and dec.fits and need <= sys.dram_bytes}
    au = getattr(getattr(m, "vision", None), "audio", None)
    if au is not None and sv.audio_s:
        out["audio_in"] = {"audio_s": sv.audio_s, "prompt_audio_tokens": au.prompt_tokens(sv.audio_s),
                           "encoder_s": ((pre.vision or {}).get("audio") or {}).get("s", 0.0)}
    return out


def _moshi_kpis(scn: Scenario, pre, audio_out_s: float, lm_only: bool = False) -> dict:
    """0.274 Moshi (core/moshi.py): per 80 ms frame, serially on the replica's card 「假设」 (server.py runs Mimi and
    the LM on one device): Mimi encode (user frame) → temporal step (the catalog model's decode at batch b, context =
    the KV ring buffer: min(serving.ctx or 3000, 3000) frames) → 8 depformer steps → Mimi decode.
        frame time = t_enc + TPOT + t_depformer + t_dec ;  real time ⇔ frame time ≤ 80 ms ;  RTF = frame / 80 ms
        response latency = (1 + delay) frames + one frame's compute (Kyutai: 160 ms theoretical)"""
    from .evaluate import _component_time
    from .hardware import System
    from .ir import _cdiv
    from .moshi import (FRAME_S, depformer_ops, depth_weight_bytes, layer_passes, mimi_decode_ops, mimi_encode_ops,
                        mimi_weight_bytes)
    sv, lay, m = scn.serving, scn.layout, pre.model
    k = m.moshi
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    ctx = min(k.context, sv.ctx or k.context)
    # 0.275: the existing per-layer fixed overhead (Scenario.layer_overhead_us, 0.69; off by default → bit-identical)
    # also on the depformer's 48 serial one-row layer passes and the Mimi transformer layers; the temporal step gets it
    # from evaluate as before
    lp = layer_passes(k)
    oh = scn.layer_overhead_us * 1e-6

    def frame(bb: int) -> dict:
        dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", ctx).replace("serving.batch", bb * lay.dp))
        t = {"temporal_s": dec.tpot,
             "depformer_s": _component_time([(depformer_ops(k, bb), 1)], sys, org, "bf16")["s"],
             "mimi_encode_s": _component_time([(mimi_encode_ops(k, bb), 1)], sys, org, "fp32")["s"],
             "mimi_decode_s": _component_time([(mimi_decode_ops(k, bb), 1)], sys, org, "fp32")["s"]}
        if oh:
            for key, n in lp.items():
                t[key] += n * oh
        t["frame_s"] = sum(t.values())
        t["fits"] = dec.fits
        return t
    b = _cdiv(sv.batch, lay.dp)
    f = frame(b)
    if lm_only:              # 0.277: frame load + fit verdict only (no streams search)
        return {"kind": "moshi", "lm_stages": {"frame": f["frame_s"] / FRAME_S}, "fits": f["fits"] and pre.fits}
    lo, hi = 0, 1
    while hi <= 1024 and frame(hi)["frame_s"] <= FRAME_S:
        lo, hi = hi, hi * 2
    while hi - lo > 1 and lo < 1024:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if frame(mid)["frame_s"] <= FRAME_S else (lo, mid)
    wb = {"temporal_GB": (m.params() - m.lookup_params - m.embed_params) * m.fmt("attn").bits / 8 / 1e9,
          "kv_GB": m.n_layers * 2 * ctx * m.hidden * 2 / 1e9,
          "depformer_GB": depth_weight_bytes(k) / 1e9, "mimi_GB": mimi_weight_bytes(k) / 1e9}
    tot = sum(wb.values())
    lat = (1 + k.delay_frames) * FRAME_S + f["frame_s"]
    return {"kind": "moshi", "label": "Moshi 全双工", "frame_ms": FRAME_S * 1e3, "streams_per_card": b, "ctx_frames": ctx,
            "split_frame": {x: f[x] for x in ("mimi_encode_s", "temporal_s", "depformer_s", "mimi_decode_s")},
            "frame_s": f["frame_s"], "rtf": f["frame_s"] / FRAME_S, "realtime": f["frame_s"] <= FRAME_S,
            "first_audio_s": lat, "latency_floor_s": (1 + k.delay_frames) * FRAME_S,
            "streams_realtime": lo, "bytes_per_frame": wb, "bytes_per_frame_GB": tot,
            "dram_GBps_for_realtime": tot / FRAME_S, "dram_GBps": sys.dram_GBps,
            "depformer_steps": k.dep_q, "depformer_layer_passes": k.dep_q * k.dep_layers,
            "layer_passes": lp, "layer_overhead_us": scn.layer_overhead_us,
            "layer_overhead_s": oh * sum(lp.values()),
            "audio_out_s_per_s_card": (b if f["frame_s"] <= FRAME_S else b * FRAME_S / f["frame_s"]) / pre.scenario.layout.cards,
            "fits": f["fits"] and pre.fits}


FIRERED_PROMPT_S = 5.0      # voice-prompt seconds when serving.audio_s = 0 「假设」 (zero-shot cloning needs a prompt)


def _cv(x) -> int:
    return int(x() if callable(x) else x)


@lru_cache(maxsize=256)
def _firered_fixed(t, pa: float, steps: int, sys, org: str) -> tuple:
    """0.277: per-request prompt encode seconds and the RedAE decode cost — batch-independent component times, memoised
    (a pure function of the frozen spec / system), so a batch search does not rebuild them every rerun."""
    from .evaluate import _component_time
    from .fireredtts3 import prompt_ops, redae_decode_ops
    return (_component_time([(prompt_ops(t, pa), 1)], sys, org, "fp32")["s"],
            _component_time([(redae_decode_ops(t, steps * t.patch), 1)], sys, org, "fp32"))


def _firered_kpis(scn: Scenario, t, audio_out_s: float) -> dict:
    """0.275 FireRedTTS3-Base (core/fireredtts3.py), one text segment, non-streaming 「来源」 (generate → redae.decode):
        prompt   = RedAE encoder (prompt audio) + patch encoder (prompt latents) + spk projections
        prefill  = backbone over 1 (speaker) + text tokens + prompt patches (+ the first head step)
        steps    = ⌈audio_out_s / 0.16⌉ AR steps, each = backbone step + DiT head (10 × CFG 2) + patch encoder (TPOT)
        decode   = RedAE decoder over all 4 · steps latents at once
        first audio = prompt + TTFT + steps · TPOT + decode (= the whole segment) ;  RTF = (steps · TPOT + decode) / audio
    ``serving.audio_s`` is the voice-prompt length (default FIRERED_PROMPT_S 「假设」)."""
    from .evaluate import _component_time
    from .fireredtts3 import prompt_ops, prompt_patches, redae_decode_ops
    from .hardware import System
    sv = scn.serving
    pa = sv.audio_s or FIRERED_PROMPT_S
    pp = prompt_patches(t, pa)
    p_tok = sv.prompt + 1 + pp
    pre = evaluate(scn.replace("serving.phase", "prefill").replace("serving.prompt", p_tok).replace("serving.audio_s", 0))
    steps = max(1, math.ceil(audio_out_s / t.s_per_step - 1e-9))
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.prompt", p_tok)
                   .replace("serving.ctx", p_tok + steps // 2).replace("serving.audio_s", 0))
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    b = max(1, sv.batch // max(1, scn.layout.dp))
    t1, dr = _firered_fixed(t, pa, steps, sys, scn.mapping)
    t_prompt = t1 * b
    t_dec = dr["s"] * b                   # one decode per request (RedAE.decode takes b = 1 「来源」)
    gen = steps * dec.tpot
    tot = t_prompt + pre.ttft + gen + t_dec
    # 0.276 DiT pin: the fp32 flow head (flow_params · 4 B) is read flow_steps times per AR step 「来源」 — the 0.269
    # re-read class (memplan.StageStorage.rr_w / rr_reads).  memplan pins it SRAM-first, then SLC (policy pin), ahead of
    # the once-read backbone; this block only reports that plan.  slc_full_GiB = the SLC that would hold all of it 「推导」.
    mp = dec.stages[-1].mem
    rr = t.flow_params * getattr(t, "w_bytes", 2.0)
    miss = max(0.0, 1.0 - (mp.pinned_rr + mp.slc_rr) / rr) if rr else 0.0
    pin = {"dit_GB": rr / 1e9, "reads_per_step": t.flow_steps, "sram_pinned_GB": mp.pinned_rr / 1e9,
           "slc_GB": mp.slc_rr / 1e9, "miss": miss, "dram_GB_per_step": miss * rr * t.flow_steps / 1e9,
           "slc_full_GiB": max(0.0, rr - mp.pinned_rr) / 2 ** 30,
           "slc_policy": scn.chip.slc_policy, "slc_mib": scn.chip.slc_bytes / 2 ** 20}
    # 0.276 bf16 re-cast (option, not in tpot_s): _backbone_one_step runs under torch.autocast(bf16) over fp32 weights
    # 「来源」, and the autocast cast cache is dropped when the context exits each step, so every backbone Linear weight
    # is read fp32 (4 B, counted) + written bf16 (2 B) + read bf16 (2 B) per step 「推导」 → +4 B/param of DRAM traffic on
    # an eager GPU path.  Embedding lookups are not autocast ops and the DiT head runs outside the autocast region
    # 「来源」.  Added serially at DRAM bandwidth (upper bound 「近似」); an engine that casts on load pays 0.
    n_cast = max(0, _cv(dec.model.params) - _cv(dec.model.embed_params))
    x_s = 4.0 * n_cast / (sys.dram_GBps * 1e9)
    recast = {"params": n_cast, "extra_GB_per_step": 4.0 * n_cast / 1e9, "extra_s_per_step": x_s,
              "tpot_s": dec.tpot + x_s, "rtf": (steps * (dec.tpot + x_s) + t_dec) / audio_out_s, "counted": False}
    return {"kind": "tts", "label": "FireRedTTS3", "dit_pin": pin, "bf16_recast": recast, "stream": "none", "audio_out_s": audio_out_s, "steps": steps,
            "prompt_audio_s": pa, "prompt_patches": pp, "first_audio_s": tot, "ttft_s": pre.ttft, "tpot_s": dec.tpot,
            "total_s": tot, "rtf": (gen + t_dec) / audio_out_s, "lm_rtf": dec.tpot / t.s_per_step,
            "split_first": {"prompt_s": t_prompt, "prefill_s": pre.ttft, "steps_s": gen, "redae_decode_s": t_dec},
            "redae_decode_bound": dr["bound"], "flow_steps": t.flow_steps, "cfg": t.cfg_mult,
            "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / pre.scenario.layout.cards,
            "fits": pre.fits and dec.fits}


@lru_cache(maxsize=1024)
def _dac_window(fs, new: int, ctx: int, hist: int, batch: int, sys, org: str) -> float:
    """0.277: seconds of one DAC window decode (``batch`` streams' windows in one forward, fp32) — memoised pure function."""
    from .drama3 import dac_ops
    from .evaluate import _component_time
    return _component_time([(dac_ops(fs, new, ctx, hist, batch), 1)], sys, org, "fp32")["s"]


@lru_cache(maxsize=1024)
def _dac_window_dram(fs, new: int, ctx: int, hist: int, batch: int, sys, org: str) -> tuple[float, float]:
    """0.278: DRAM bytes of one DAC window decode split (weights, activations) in GB — the same op sums as
    ``_dac_window`` (evaluate._component_time: weights = hot + exp + w_extra, activations = act)."""
    from types import SimpleNamespace
    from .drama3 import dac_ops
    from .evaluate import _sum_ops
    d = _sum_ops(dac_ops(fs, new, ctx, hist, batch), sys, org, SimpleNamespace(act_fmt="fp32", kv_fmt="fp32"))
    return (d["hot"] + d["exp"] + d["w_extra"]) / 1e9, d["act"] / 1e9


def _fish_kpis(scn: Scenario, fs, audio_out_s: float, lm_only: bool = False) -> dict:
    """0.276 Fish Audio Drama 3 (core/drama3.py), streaming 「来源」 (vllm-omni fish_omni → dac_decoder async chunks):
        prompt  = text tokens + one row per voice-reference frame (serving.audio_s, default drama3.REF_S 「假设」)
        step    = one 10-codebook frame (2048 / 44.1 kHz = 46.4 ms) = backbone token + 13-layer codebook tower + heads
        LM RTF  = TPOT / frame (Stage 0)
    0.277 Stage 1 (DAC decode, drama3.dac_ops) on the same card(s) 「来源」 (recipe: both stages on one GPU; deploy memory
    0.78 stage 0 / 0.1 codec), time-shared with the LM 「假设」 (no overlap: the two loads add).  Windows =
    drama3.dac_windows (ramp 6 … 32, then 64); the b streams of a replica decode each window in one batched forward
    (per_step_batch 「来源」; windows aligned 「近似」).  Stateful 「假设」 (fish_speech_dac_stateful: only new frames); the
    module decoder's re-decode of the ≤ 32 left-context frames is reported as ``dac.redecode_s_per_request``.
        DAC load  = Σ_windows DAC(window, b) / audio_out_s ;  RTF = LM RTF + DAC load
        first audio = prefill + 6 decode steps + DAC(window 0, batch 1)
        memory: the 0.74 GB fp32 decode weights on the LM card (last PP stage)
    ``lm_only`` (speech_insight.speech_streams): only {kind, lm_stages, fits} — same expressions."""
    from .drama3 import CHUNK_RAMP, REF_S, dac_gflop_per_frame, dac_params, dac_windows
    from .hardware import System
    sv = scn.serving
    ref = sv.audio_s or REF_S
    rows = fs.ref_rows(ref)
    p_tok = sv.prompt + rows
    pre = evaluate(scn.replace("serving.phase", "prefill").replace("serving.prompt", p_tok).replace("serving.audio_s", 0))
    steps = max(1, math.ceil(audio_out_s / fs.frame_s - 1e-9))
    dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.prompt", p_tok)
                   .replace("serving.ctx", p_tok + steps // 2).replace("serving.audio_s", 0))
    lm_rtf = dec.tpot / fs.frame_s
    fits = pre.fits and dec.fits
    x_dac, gb = 0.0, 0.0
    if fs.codec:
        sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                     scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
        b = max(1, sv.batch // max(1, scn.layout.dp))
        win = dac_windows(steps)
        tb = [_dac_window(fs, n, 0, done, b, sys, scn.mapping) for n, _, done in win]
        x_dac = sum(tb) / audio_out_s
        gb = dac_params(fs) * 4.0 / 1e9
        mp = dec.stages[-1].mem
        fits = fits and mp.dram_need + gb * 1e9 <= mp.dram_cap
    if lm_only:
        return {"kind": "tts", "lm_stages": {"lm": lm_rtf + x_dac}, "fits": fits}
    c0 = min(CHUNK_RAMP[0], steps)
    first = pre.ttft + c0 * dec.tpot
    tot = pre.ttft + steps * dec.tpot
    split = {"prefill_s": pre.ttft, "first_chunk_steps_s": c0 * dec.tpot}
    out = {"kind": "tts", "label": "Drama 3", "stream": "chunk", "audio_out_s": audio_out_s, "steps": steps,
           "frame_ms": fs.frame_s * 1e3, "ref_audio_s": ref, "ref_rows": rows, "prompt_rows": p_tok,
           "first_chunk_frames": c0, "ttft_s": pre.ttft, "tpot_s": dec.tpot, "lm_rtf": lm_rtf, "rtf": lm_rtf + x_dac,
           "dac_costed": bool(fs.codec)}
    if fs.codec:
        t1 = [_dac_window(fs, n, 0, done, 1, sys, scn.mapping) for n, _, done in win]
        tr = [_dac_window(fs, n, ctx, 0, 1, sys, scn.mapping) for n, ctx, _ in win]
        first += t1[0]
        tot += tb[-1]
        split["dac_window0_s"] = t1[0]
        out["dac"] = {"windows": [n for n, _, _ in win], "batch": b, "s": tb, "s_batch1": t1, "load": x_dac,
                      "s_per_request_batch1": sum(t1), "redecode_s_per_request": sum(tr), "params": dac_params(fs),
                      "GB": gb, "stateful": True, "colocated": True, "gflop_per_frame": dac_gflop_per_frame(fs)}
        # 0.278 DRAM traffic per replica step-window sweep (all windows of one request set, batch b): weights vs
        # activations — the SLC-pinning argument (weights are read once per window; activations dominate)
        dd = [_dac_window_dram(fs, n, 0, done, b, sys, scn.mapping) for n, _, done in win]
        out["dac"]["dram_weights_GB"] = sum(w for w, _ in dd)
        out["dac"]["dram_act_GB"] = sum(a for _, a in dd)
    out.update({"first_audio_s": first, "total_s": tot, "split_first": split,
                "audio_out_s_per_s_card": sv.batch * audio_out_s / tot / pre.scenario.layout.cards, "fits": fits})
    return out


def _echo_kpis(scn: Scenario, pre, audio_out_s: float, lm_only: bool = False) -> dict:
    """0.279 Index-Echo S2ST (core/echo.py): one source clip → translated speech, every stage serial on the scenario's
    card(s) (README: one CUDA GPU; ``dub`` is batch 1 and non-streaming 「来源」):
        ST    = AuT encoder + ST prefill (ST_PROMPT text + audio positions) + greedy decode of n_src + 1 + m + 1 tokens
                + one more full forward over prompt + those tokens (``extract``, hidden states; the audio embeddings are
                reused, so the encoder is not run again)
        map   = Hidden2CVMapper over the m translation tokens (fp32)
        CV3   = the fun-cosyvoice3 catalog LM on one card (fp32 as released): prefill sos + CV3_INSTRUCT + n_src + m + 1
                + 25 · audio_s prompt speech tokens, then n_speech = clip(25 · audio_out_s, 2m, min(20m, 1500)) decode steps
        audio = flow over prompt + n_speech tokens (CFM steps × CFG) + HiFT over the generated mel, once (stream=False)
        first audio = total (non-streaming); RTF = total / audio_out_s
    0.280 concurrent clips: b = ⌈serving.batch / dp⌉ clips per replica share the card(s) 「假设」 (``dub`` itself is batch 1):
    the ST prefill / decode / hidden pass and the CV3 prefill / decode run batched (b sequences per step), the mapper,
    flow and HiFT run per clip back to back (b ×) — x(b) = total(b) / audio_out_s is the replica's real-time load
    (every clip's output audio is produced per round).  b = 1 is the 0.279 result.  ``lm_only``
    (speech_insight.speech_streams): only {kind, lm_stages {pipeline: x}, fits} — same expressions.
    m = n_tgt ≈ TEXT_TOK_PER_S · audio_out_s and n_src ≈ TEXT_TOK_PER_S · audio_s 「假设」 (the ST and CV3 tokenizers are
    both Qwen BPE — the same count for the translation 「近似」)."""
    from .catalog import get_model
    from .cosyvoice import cosy_pin, flow_ops, hift_ops, scaled
    from .echo import CV3_INSTRUCT, ST_PROMPT, TEXT_TOK_PER_S, mapper_ops
    from .evaluate import _component_time
    from .hardware import System
    m_ = pre.model
    e = m_.echo
    sv = scn.serving
    a_s = sv.audio_s or 10.0
    n_src = max(1, math.ceil(TEXT_TOK_PER_S * a_s - 1e-9))
    m = max(1, min(e.max_len, math.ceil(TEXT_TOK_PER_S * audio_out_s - 1e-9)))
    n_out = min(e.max_new_tokens, n_src + 1 + m + 1)
    b = max(1, -(-sv.batch // max(1, scn.layout.dp)))
    base = scn.replace("serving.batch", b * max(1, scn.layout.dp)).replace("serving.audio_s", a_s)
    st_pre = evaluate(base.replace("serving.phase", "prefill").replace("serving.prompt", ST_PROMPT))
    enc = ((st_pre.vision or {}).get("audio") or {}).get("s", 0.0)
    st_dec = evaluate(base.replace("serving.phase", "decode").replace("serving.prompt", ST_PROMPT)
                      .replace("serving.ctx", ST_PROMPT + n_out // 2))
    hid = evaluate(base.replace("serving.phase", "prefill").replace("serving.prompt", ST_PROMPT + n_out))
    t_hid = hid.ttft - ((hid.vision or {}).get("audio") or {}).get("s", 0.0)
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    org = scn.mapping
    t_map = _component_time([(mapper_ops(e, m), 1)], sys, org, "fp32")["s"]
    cv = get_model(e.cosy_id)
    c = cv.cosy
    psp = math.ceil(c.token_rate * min(a_s, 30.0) - 1e-9)        # prompt speech tokens of the source clip (≤ 30 s)
    # 0.280 CosyVoice3 frontend from the released ONNX graphs (core/onnxfront.py): speech_tokenizer_v3 on the card
    # (CUDAExecutionProvider), one session run per clip over ≤ 30 s of source audio 「来源」 frontend.py; CampPlus on the
    # host CPU (CPUExecutionProvider) — its FLOPs are reported, not charged to the card
    from .model import load_release
    from .onnxfront import front_flops, front_ops
    fe = (load_release(cv.hf_id) or {}).get("onnx_frontend") or {}
    stk, cpp = fe.get("speech_tokenizer"), fe.get("campplus")
    t_tok = _component_time([(front_ops(stk, min(a_s, stk["max_s"] or a_s), "s3tok"), 1)], sys, org, "fp32")["s"] \
        if stk else 0.0
    n_sp = e.speech_tokens(m, math.ceil(c.token_rate * audio_out_s - 1e-9))
    cv_p = 1 + CV3_INSTRUCT + n_src + m + 1 + psp
    cbase = (Scenario(model=e.cosy_id, chip=scn.chip, mem_id=scn.mem_id)
             .replace("mem_eff", scn.mem_eff).replace("mapping", scn.mapping).replace("serving.batch", b))
    cv_pre = evaluate(cbase.replace("serving.phase", "prefill").replace("serving.prompt", cv_p))
    cv_dec = evaluate(cbase.replace("serving.phase", "decode").replace("serving.prompt", cv_p)
                      .replace("serving.ctx", cv_p + n_sp // 2))
    miss = cosy_pin(c, sys.chip.sram_bytes)
    ch, ev = flow_ops(c, psp + n_sp)
    g = [(ch, 1), (ev, 1)] + ([(scaled(ev, miss), c.steps - 1)] if c.steps > 1 else [])
    t_flow = _component_time(g, sys, org, "fp32")["s"] + sum(o.kv_read for o in ch) / (sys.dram_GBps * 1e9)
    t_hift = _component_time([(hift_ops(c, n_sp * c.mel_ratio), 1)], sys, org, "fp32")["s"]
    split = {"encoder_s": enc, "st_prefill_s": st_pre.ttft - enc, "st_decode_s": n_out * st_dec.tpot,
             "st_hidden_pass_s": t_hid, "mapper_s": b * t_map, "cv3_prefill_s": cv_pre.ttft,
             "cv3_decode_s": n_sp * cv_dec.tpot, "flow_s": b * t_flow, "hift_s": b * t_hift,
             "speech_tokenizer_s": b * t_tok}
    # 0.280: the ST and CV3 evaluations each pin their own hot weights in SRAM / SLC — on one card the two phases
    # time-share it, so every round re-loads each phase's pinned set from DRAM once 「推导」 (phase-batched rounds; clips
    # staggered across phases are not modelled 「假设」)
    pin = lambda r: r.stages[-1].mem.pinned_hot + r.stages[-1].mem.slc_hot
    split["repin_s"] = (pin(st_dec) + pin(cv_dec)) / (sys.dram_GBps * 1e9)
    tot = sum(split.values())
    mem = st_pre.stages[-1].mem
    extra_B = cv_dec.stages[-1].mem.dram_need + c.stored_bytes + e.mapper_params * 4.0 + (stk["params"] * 4.0 if stk else 0.0)
    fits = st_pre.fits and st_dec.fits and hid.fits and cv_pre.fits and cv_dec.fits and mem.dram_need + extra_B <= mem.dram_cap
    if lm_only:
        return {"kind": "s2st", "lm_stages": {"pipeline": tot / audio_out_s}, "fits": fits}
    return {"kind": "s2st", "label": e.label, "audio_s": a_s, "audio_out_s": audio_out_s, "src_tokens": n_src,
            "tgt_tokens": m, "st_out_tokens": n_out, "speech_tokens": n_sp, "prompt_speech_tokens": psp,
            "cv3_prompt": cv_p, "flow_steps": c.steps, "cfg": c.cfg_mult, "split": split, "clips": b,
            "frontend": {"speech_tokenizer_gflop": front_flops(stk, min(a_s, stk["max_s"] or a_s)) / 1e9 if stk else None,
                         "speech_tokenizer_params": stk["params"] if stk else 0,
                         "campplus_gflop_host": front_flops(cpp, a_s) / 1e9 if cpp else None,
                         "campplus_params": cpp["params"] if cpp else 0, "campplus_on": "host CPU"},
            "dominant": max(split, key=split.get), "first_audio_s": tot, "total_s": tot, "rtf": tot / audio_out_s,
            "ttft_s": st_pre.ttft, "st_tpot_s": st_dec.tpot, "cv3_tpot_s": cv_dec.tpot,
            "audio_out_s_per_s_card": b * max(1, scn.layout.dp) * audio_out_s / tot / pre.scenario.layout.cards,
            "card_GB": (mem.dram_need + extra_B) / 1e9, "fits": fits}


def speech_kpis(scn: Scenario, audio_out_s: float = 10.0, lm_only: bool = False) -> dict:
    """``lm_only`` (0.277, speech_insight.speech_streams): omni / cosyvoice / kimi / moshi return only
    {kind, lm_stages, fits} — the LM-card stage loads and the fit verdict by the same expressions as the full KPIs, without
    the speech-card timing (or Moshi's own streams search) that the LM-card search never reads.  Other kinds ignore it."""
    sv = scn.serving
    pre = evaluate(scn.replace("serving.phase", "prefill"))
    m = pre.model
    if getattr(m, "omni", None) is not None:
        return _omni_kpis(scn, pre, audio_out_s, lm_only)
    if getattr(m, "echo", None) is not None:
        return _echo_kpis(scn, pre, audio_out_s, lm_only)
    if getattr(m, "cosy", None) is not None:
        return _cosy_kpis(scn, pre, audio_out_s, lm_only)
    if getattr(m, "kimi", None) is not None:
        return _kimi_kpis(scn, pre, audio_out_s, lm_only)
    if getattr(m, "moshi", None) is not None:
        return _moshi_kpis(scn, pre, audio_out_s, lm_only)
    if getattr(m, "fish", None) is not None:
        return _fish_kpis(scn, m.fish, audio_out_s, lm_only)
    ts = getattr(m, "tts", None)
    au = getattr(getattr(m, "vision", None), "audio", None)
    if ts is not None and getattr(ts, "kind", "") == "firered":
        return _firered_kpis(scn, ts, audio_out_s)
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
