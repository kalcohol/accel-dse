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

``verify`` re-runs the same KPIs with the chip's SLC = existing + S (policy pin, S rounded up to 64 MiB) so the claim is checked by the
engine's own SLC path, not by the formula.  Nothing here is tuned to a measurement.

0.275: each row is built from the scenario's hardware / system fields only — serving (batch 1, prompt 64, reply 64,
audio_s), layout, format overrides, workload, PD and other model-specific fields are reset to their defaults, so the
UI's current model settings (e.g. a draft model) never leak into a speech row."""
from __future__ import annotations

import math

from .catalog import get_model
from .evaluate import evaluate
from .hardware import System
from .scenario import Scenario

SPEECH_MODELS = ("moshiko-7b", "kimi-audio-7b-instruct", "step-audio-2-mini", "qwen2.5-omni-7b", "cosyvoice2-0.5b",
                 "fun-cosyvoice3-0.5b-2512", "fireredtts3-base", "drama-3", "ming-3.0-tts-tiny", "qwen2-audio-7b-instruct", "qwen3-asr-1.7b",
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
        sf = k.get("split_first")          # 0.275 FireRedTTS3: non-streaming → first audio = whole segment
        return ({"lm": k["rtf"]}, k["steps"] / k["audio_out_s"], max(sf, key=sf.get) if sf else "lm_ttft",
                k["first_audio_s"])
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


# model-specific scenario fields reset to their defaults (a speech row must not inherit the UI model's spec decoding,
# format overrides, PD, workload or layout); hardware / system / overhead fields are kept
_MODEL_FIELDS = ("layout", "formats_override", "workload", "pd", "pp_split", "indexer", "attn_mode", "hyena_prefill")


def _scn(base: Scenario, mid: str, audio_s: float) -> Scenario:
    import dataclasses
    from .scenario import Serving
    m = get_model(mid)
    au = getattr(getattr(m, "vision", None), "audio", None)
    dflt = {}
    for f in dataclasses.fields(Scenario):
        if f.name in _MODEL_FIELDS:
            dflt[f.name] = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
    sv = Serving(batch=1, prompt=PROMPT, out_len=OUT_LEN, audio_s=audio_s if au is not None else 0)
    return dataclasses.replace(base, model=mid, serving=sv, **dflt)


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
                    # on top of any SLC the chip already has (x was measured with it)
                    mib = base.chip.slc_mib + math.ceil(s_gb * 1e9 / 2 ** 20 / 64) * 64
                    chip = base.chip.__class__(**{**base.chip.__dict__, "slc_mib": float(mib), "slc_policy": "pin"})
                    kv = speech_kpis(_scn(base.replace("chip", chip), mid, audio_s), audio_s)
                    xv = _stages(kv, audio_s)[0][bind]
                    row["verify"] = {"slc_mib": mib, "x": xv, "realtime": xv <= 1.0}
        rows.append(row)
    return {"chip": base.chip.name, "mem_id": base.mem_id, "dram_GBps_eff": bw, "audio_s": audio_s,
            "batch": 1, "out_len": OUT_LEN, "rows": rows,
            "method": "x = 忙碌秒 / 音频秒；x>1 时：所需有效 DRAM 带宽 = BW·x；或 SLC 固定 S = (x−1)·BW / r 字节的复读权重"
                      "（r = 每音频秒的 token / 帧数）「推导」；verify = 用引擎自身 SLC 路径复算「近似」"}


# ---- 0.276: real-time streams per card vs SLC size ---------------------------------------------------------------
SLC_GIB = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SPEECH_CARD = ("speech_card",)
MAX_STREAMS = 1024


def _with_slc(base: Scenario, gib: float) -> Scenario:
    """The chip with an SLC of ``gib`` GiB (policy pin; 0 → the chip's own SLC setting)."""
    if not gib:
        return base
    chip = base.chip.__class__(**{**base.chip.__dict__, "slc_mib": float(gib * 1024), "slc_policy": "pin"})
    return base.replace("chip", chip)


def _lm_ok(base: Scenario, mid: str, audio_s: float, b: int) -> tuple[bool, dict, dict]:
    """b concurrent streams on the LM card(s): every stage except the separate speech card keeps up, and it fits."""
    from .speech import speech_kpis
    k = speech_kpis(_scn(base, mid, audio_s).replace("serving.batch", b), audio_s)
    st = _stages(k, audio_s)[0]
    lm = {s: v for s, v in st.items() if s not in SPEECH_CARD}
    return (bool(k["fits"]) and max(lm.values()) <= 1.0), lm, k


def _max_streams(ok, guess: int = 1) -> int:
    """Largest b in [0, MAX_STREAMS] with ok(b) — engine reruns, ok assumed monotone in b.  Galloping search from
    ``guess`` (the linear prediction): doubling up / halving down to a bracket, then bisection."""
    memo: dict[int, bool] = {}
    f = lambda b: memo[b] if b in memo else memo.setdefault(b, bool(ok(b)))
    if not f(1):
        return 0
    g = max(1, min(int(guess), MAX_STREAMS))
    if f(g):
        lo, hi = g, min(2 * g, MAX_STREAMS + 1)
        while hi <= MAX_STREAMS and f(hi):
            lo, hi = hi, min(2 * hi, MAX_STREAMS + 1)
        if g == MAX_STREAMS:
            return MAX_STREAMS
    else:
        lo, hi = 1, g
        while hi // 2 > 1:
            c = hi // 2
            if f(c):
                lo = c
                break
            hi = c
    hi = min(hi, MAX_STREAMS + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return lo


def speech_streams(base: Scenario, models=SPEECH_MODELS, audio_s: float = 10.0, slc_gib=SLC_GIB) -> dict:
    """Real-time streams per card vs SLC size, per speech model (one replica = the LM card(s) + the speech cards it
    needs).  Every number is an engine rerun:

        N_lm   = largest b with every LM-card stage x(b) ≤ 1 and the replica fitting (doubling + bisection, ≤ 1024)
        N_sc   = real-time streams of one speech card (the speech KPIs' own streams_realtime at batch 1; cosyvoice / kimi /
                 omni — batch-1 detokenizers serve streams back to back, so x_sc(b) = b · x_sc(1))
        speech cards = ⌈N_lm / N_sc⌉ ;  streams / card = N_lm / (LM cards + speech cards)

    Mechanism read-out 「推导」: x_lm(b) ≈ x0 + d · b (weights read once per step for the whole batch, KV / per-stream work
    per stream), fitted from b = 1, 2 → predicted N_lm = ⌊(1 − x0) / d⌋ — shown next to the searched value; a pinned SLC
    lowers x0 (the shared weight read), so streams grow by ≈ Δx0 / d."""
    from .speech import speech_kpis
    out_rows = []
    for mid in models:
        row = {"model": mid, "curve": []}
        for gib in slc_gib:
            sb = _with_slc(base, gib)
            ok1, lm1, k1 = _lm_ok(sb, mid, audio_s, 1)
            lm_cards = _scn(sb, mid, audio_s).layout.cards
            n_sc = None
            if "speech_card" in _stages(k1, audio_s)[0]:
                n_sc = int(k1.get("streams_realtime") or 0)
            _, lm2, _ = _lm_ok(sb, mid, audio_s, 2)
            x1, x2 = max(lm1.values()), max(lm2.values())
            d = x2 - x1
            pred = (max(0, math.floor((1.0 - (x1 - d)) / d + 1e-9)) if d > 1e-12 else (MAX_STREAMS if x1 <= 1 else 0)) \
                if x1 <= 1 else 0
            n_lm = _max_streams(lambda b: _lm_ok(sb, mid, audio_s, b)[0], max(1, pred)) if ok1 else 0
            if n_lm >= MAX_STREAMS:
                limit = "cap"
            else:                            # what stops one more stream: real time or memory
                _, lmn, kn = _lm_ok(sb, mid, audio_s, n_lm + 1)
                limit = "realtime" if max(lmn.values()) > 1.0 else "memory" if not kn["fits"] else "realtime"
            if n_sc is None:
                sc_cards, n = 0, n_lm
            elif n_sc == 0:
                sc_cards, n = None, 0        # one speech card cannot carry even one stream in real time
            else:
                sc_cards, n = (math.ceil(n_lm / n_sc) if n_lm else 0), n_lm
            cards = lm_cards + (sc_cards or 0)
            row["kind"] = k1["kind"]
            row["curve"].append({"slc_gib": gib, "x1": x1, "streams_lm": n_lm, "streams_lm_pred": min(pred, MAX_STREAMS),
                                 "streams_speech_card": n_sc, "speech_cards": sc_cards, "cards": cards,
                                 "streams": n, "streams_per_card": n / cards if cards and n else 0.0,
                                 "lm_binding": max(lm1, key=lm1.get), "lm_limit": limit, "capped": n_lm >= MAX_STREAMS})
        c0 = row["curve"][0]
        row["first_realtime_slc_gib"] = next((c["slc_gib"] for c in row["curve"] if c["streams"] > 0), None)
        row["gain_per_gib"] = [(c["streams_per_card"] - c0["streams_per_card"]) / c["slc_gib"] if c["slc_gib"] else 0.0
                               for c in row["curve"]]
        out_rows.append(row)
    return {"chip": base.chip.name, "mem_id": base.mem_id, "audio_s": audio_s, "slc_gib": list(slc_gib),
            "max_streams": MAX_STREAMS, "rows": out_rows,
            "method": "N_lm = 最大 b（LM 卡上各阶段 x(b) ≤ 1 且放得下，引擎逐点重算：倍增 + 二分）；语音卡路数 = 语音 KPI 自身 batch-1 "
                      "实时路数；语音卡数 = ⌈N_lm / N_sc⌉；每卡路数 = N_lm / (LM 卡 + 语音卡)；预测 = 由 b = 1, 2 线性外推 "
                      "x(b) ≈ x0 + d·b 「推导」；SLC 以 pin 策略加在每张卡上「假设」"}
