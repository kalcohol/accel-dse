"""Cross-model speech insight (0.274): which stage bounds real time / first audio, and what unlocks real time.

For every catalog speech model the existing speech KPIs (core/speech.py) are evaluated at batch 1 on the scenario's
chip + memory, and every stage that must keep up with the audio clock is expressed as busy seconds per audio second
x (x ≤ 1 ⇔ the stage keeps up; an ASR stage per *input* audio second, a speech-output stage per *output* second):

  asr        encoder, LM (prefill rest + decode)
  tts        LM steps (Ming TTS head)
  cosyvoice  LM tokens (token_rate / s), speech card (flow + HiFT)
  kimi       LM tokens (12.5 / s), detokenizer card (DiT + vocoder)
  omni       Thinker reply (out_len tokens), speech card (Talker 50 / s + DiT + BigVGAN, serial on one card)
  moshi      frame (Mimi encode + temporal step + 8 depformer steps + Mimi decode, serial, 12.5 frames / s)

The binding stage is the one with the largest x.  When it misses real time (x > 1) and is bound by weight re-reads
from DRAM (the decode-type stages here: every token / frame re-reads the stage's weights — evaluate reports the
bound), the remedy follows from bytes, not from fitting 「推导」:

    excess DRAM time per audio second  Δ = x − 1
    → effective DRAM bandwidth needed  BW_need = BW_eff · x                       (all bytes stay in DRAM)
    → or pin S bytes of re-read weights in an SLC; each audio second re-reads them r times
         (r = tokens / frames per audio second), saving S · r bytes:  S = Δ · BW_eff / r
         at an SLC read bandwidth ≥ S · r                                       「近似」 (stage stays memory bound)

``verify`` re-runs the same KPIs with the chip's SLC = S (policy pin, next 64 MiB) so the claim is checked by the
engine's own SLC path, not by the formula.  Nothing here is tuned to a measurement."""
from __future__ import annotations

import math

from .catalog import get_model
from .evaluate import evaluate
from .hardware import System
from .scenario import Scenario

SPEECH_MODELS = ("moshiko-7b", "kimi-audio-7b-instruct", "step-audio-2-mini", "qwen2.5-omni-7b", "cosyvoice2-0.5b",
                 "fun-cosyvoice3-0.5b-2512", "ming-3.0-tts-tiny", "qwen2-audio-7b-instruct", "qwen3-asr-1.7b",
                 "fun-asr-nano-2512", "whisper-large-v3", "whisper-large-v3-turbo", "ming-3.0-asr-tiny")
OUT_LEN = 64            # reply / transcript tokens 「假设」 (same default as the UI speech panel)
PROMPT = 64             # text prompt tokens 「假设」


def _stages(k: dict, audio_s: float) -> tuple[dict, float, str, float]:
    """(stages {name: x}, r re-reads / audio s of the decode-type stage, first-audio dominant part, first-audio s)."""
    kind = k["kind"]
    if kind == "asr":
        enc = k["encoder_s"]
        return ({"encoder": enc / audio_s, "lm": (k["total_s"] - enc) / audio_s}, k["out_len"] / audio_s,
                "lm_first_token" if k["first_token_s"] > 2 * enc else "encoder", k["first_token_s"])
    if kind == "tts":
        return ({"lm": k["rtf"]}, k["steps"] / k["audio_out_s"], "lm_ttft", k["first_audio_s"])
    sf = k.get("split_first") or k.get("split_frame") or {}
    dom = max(sf, key=sf.get) if sf else ""
    if kind in ("cosyvoice", "kimi"):
        return ({"lm": k["lm_rtf"], "speech_card": k["rtf"]}, k["token_rate"], dom, k["first_audio_s"])
    if kind == "omni":
        return ({"thinker": k["out_len"] * k["tpot_s"] / k["audio_out_s"], "speech_card": k["rtf"]}, k["codec_per_s"],
                dom, k["first_audio_s"])
    if kind == "moshi":
        return ({"frame": k["rtf"]}, 1.0 / (k["frame_ms"] / 1e3), dom, k["first_audio_s"])
    raise ValueError(f"speech insight: unknown speech kind {kind}")


def _scn(base: Scenario, mid: str, audio_s: float) -> Scenario:
    m = get_model(mid)
    au = getattr(getattr(m, "vision", None), "audio", None)
    s = base.replace("model", mid).replace("serving.batch", 1).replace("serving.prompt", PROMPT)
    s = s.replace("serving.out_len", OUT_LEN).replace("serving.audio_s", audio_s if au is not None else 0)
    return s.replace("layout", type(base.layout)())


def speech_bounds(base: Scenario, models=SPEECH_MODELS, audio_s: float = 10.0, verify: bool = True) -> dict:
    from .speech import speech_kpis
    sys = System(base.chip, base.mem_id, base.mem_eff)
    bw = sys.dram_GBps
    rows = []
    for mid in models:
        scn = _scn(base, mid, audio_s)
        k = speech_kpis(scn, audio_s)
        st, r, dom, first = _stages(k, audio_s)
        bind = max(st, key=st.get)
        x = st[bind]
        dec = evaluate(scn.replace("serving.phase", "decode").replace("serving.ctx", PROMPT + OUT_LEN))
        row = {"model": mid, "kind": k["kind"], "stages": st, "binding": bind, "x": x, "realtime": x <= 1.0,
               "first_audio_s": first, "first_audio_dominant": dom, "rereads_per_audio_s": r,
               "lm_bound": dec.bound, "fits": k["fits"]}
        if k["kind"] == "moshi":
            row["latency_floor_s"] = k["latency_floor_s"]
        if x > 1.0:
            row["dram_GBps_eff_needed"] = bw * x
            if bind == "speech_card" and k["kind"] in ("cosyvoice", "kimi"):
                row["unlock_note"] = "语音卡（DiT / flow）受限：按 speech_cards 扩卡或 SLC 固定 DiT 权重（见 eval speech.slc）"
            else:
                s_gb = (x - 1.0) * bw / r
                row["slc_GB_needed"] = s_gb
                row["slc_GBps_needed"] = s_gb * r
                if verify:
                    mib = math.ceil(s_gb * 1e9 / 2 ** 20 / 64) * 64
                    chip = base.chip.__class__(**{**base.chip.__dict__, "slc_mib": float(mib), "slc_policy": "pin"})
                    kv = speech_kpis(_scn(base.replace("chip", chip), mid, audio_s), audio_s)
                    xv = _stages(kv, audio_s)[0][bind]
                    row["verify"] = {"slc_mib": mib, "x": xv, "realtime": xv <= 1.0}
        rows.append(row)
    return {"chip": base.chip.name, "mem_id": base.mem_id, "dram_GBps_eff": bw, "audio_s": audio_s,
            "batch": 1, "out_len": OUT_LEN, "rows": rows,
            "method": "x = 忙碌秒 / 音频秒；x>1 时：所需有效 DRAM 带宽 = BW·x；或 SLC 固定 S = (x−1)·BW / r 字节的复读权重"
                      "（r = 每音频秒的 token / 帧数）「推导」；verify = 用引擎自身 SLC 路径复算「近似」"}
