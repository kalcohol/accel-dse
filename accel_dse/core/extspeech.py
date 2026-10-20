"""0.282: external speech measurements — Fish Audio Drama 3 on one H200 (SGLang-Omni cookbook, Tables 1–3).

Diagnostic only (like extval variants): the engine is evaluated on the H200 reference chip under an ext variant and the
published serving loop is reproduced from its sourced settings; nothing here is fitted, and nothing feeds the cost model.

Closed loop at concurrency c (every stream restarts as soon as it ends, steady state) 「推导」:
    one cycle of c requests = c prefills (batch 1, serial 「近似」: SGLang batches up to 16 first frames, and
    prefill_decode_interval 4 keeps them out of decode rounds) + N decode steps at batch c + the vocoder's work
    (same GPU, own process → time-shared, loads add 「假设」)
    frames/s = c·N / (c·ttft₁ + N·tpot(c) + c·vocoder_per_request)
    TTFA     = ttft₁ + stream_stride · tpot(c) + vocoder(first window)        (no queueing)
Vocoder modes:
    "sglang"  — SGLang-Omni (the measured stack) 「来源」 sglang_omni/models/fishaudio_drama3 config.py + vocoder.py @
                0029194: strides 10, 10, 12, 32, then 64; every window re-decodes min(32, emitted) earlier frames
                (stream_overlap_tokens 32); each stream's window decodes alone (CUDA graphs captured at batch 1).
    "vllm"    — the catalog's vllm-omni model (speech._fish_kpis: ramp 6 … 32, 64; stateful, the c streams batched).
Workload 「推导」/「假设」: N = Table 3 latency mean / RTF mean = 1.235 / 0.281 s = 4.40 s of audio → 95 frames
(Seed-TTS English, non-streaming c = 8, same prompts); reference = the cookbook's 3.9 s Seed-TTS clip 「假设」 (each
prompt clones its own reference; lengths not published); 64 text rows (template + text + transcript) 「假设」.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "ext_measurements_speech.json"
STRIDE, RAMP, FOLLOW, OVERLAP = 10, (10, 12, 32), 64, 32      # sglang config.py DEFAULT_STREAM_* / stream_overlap_tokens
OUT_S = 1.235 / 0.281          # Table 3: mean latency / mean RTF = mean audio seconds per request 「推导」
REF_S = 3.9                    # cookbook reference clip common_voice_en_10119832.wav 「假设」 (as every prompt's)
TEXT_ROWS = 64                 # 「假设」


def rows() -> list[dict]:
    return json.loads(DATA.read_text())["rows"]


def sglang_windows(frames: int) -> list[tuple[int, int, int]]:
    """[(new, left context, emitted)] of one SGLang-Omni stream 「来源」 stream_windows.py: window i covers stride_i new
    frames plus min(overlap, emitted) earlier ones; a final partial window decodes what is left."""
    out, done, i = [], 0, 0
    strides = (STRIDE,) + RAMP
    while done < frames:
        n = min(strides[i] if i < len(strides) else FOLLOW, frames - done)
        out.append((n, min(OVERLAP, done), done))
        done += n
        i += 1
    return out


def _scenario(variant: str, c: int):
    from . import extval as X
    r = dict(hw="H200-SXM", model="drama-3", tp=1, dtype="release")
    return X.scenario(r, variant, phase="decode", batch=c, ctx=512, prompt=TEXT_ROWS, audio_s=REF_S)


def predict(variant: str, c: int, vocoder: str = "sglang", out_s: float = OUT_S) -> dict:
    from .catalog import get_model
    from .drama3 import CHUNK_RAMP
    from .hardware import System
    from .speech import _dac_window, speech_kpis
    scn = _scenario(variant, c)
    k1 = speech_kpis(scn.replace("serving.batch", 1), out_s)
    kc = speech_kpis(scn, out_s)
    fs = get_model("drama-3").fish
    N = kc["steps"]
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    if vocoder == "none":           # vocoder on a second GPU (Table 2): the engine's GPU runs prefill + decode only
        voc_req, voc_first, first_frames = 0.0, 0.0, STRIDE
    elif vocoder == "sglang":
        win = sglang_windows(N)
        voc_req = sum(_dac_window(fs, n, ctx, done, 1, sys, scn.mapping) for n, ctx, done in win)
        voc_first = _dac_window(fs, win[0][0], 0, 0, 1, sys, scn.mapping)
        first_frames = STRIDE
    else:
        voc_req = sum(kc["dac"]["s"]) / c          # the c streams' windows batched
        voc_first = kc["dac"]["s_batch1"][0]
        first_frames = CHUNK_RAMP[0]
    ttft1, tpot = k1["ttft_s"], kc["tpot_s"]
    cycle = c * ttft1 + N * tpot + c * voc_req
    return {"c": c, "frames": N, "ttft1_s": ttft1, "tpot_s": tpot, "vocoder_s_per_request": voc_req,
            "frames_per_s": c * N / cycle, "ttfa_s": ttft1 + first_frames * tpot + voc_first,
            "shares": {"prefill": c * ttft1 / cycle, "decode": N * tpot / cycle, "vocoder": c * voc_req / cycle},
            "fits": kc["fits"]}


def table(variant: str, vocoder: str = "sglang", t2: bool = False) -> list[dict]:
    """Table 1 rows (vocoder on the engine's GPU), or with ``t2`` the Table 2 rows (same / second GPU; the second-GPU
    rows drop the vocoder term).  Table 2 also counts each stream's end-of-speech frame (N + 1)."""
    out = []
    for r in rows():
        if r["metric"] != ("frames_per_s_t2" if t2 else "frames_per_s"):
            continue
        voc = "none" if r.get("vocoder_gpu") == "second" else vocoder
        p = predict(variant, r["concurrency"], voc)
        if t2:
            p["frames_per_s"] *= (p["frames"] + 1) / p["frames"]
        out.append({"id": r["id"], "c": r["concurrency"], "vocoder_gpu": r.get("vocoder_gpu"), "meas": r["value"], "pred": round(p["frames_per_s"], 1),
                    "ratio": round(p["frames_per_s"] / r["value"], 3),
                    "ttfa_meas_ms": r.get("ttfa_p50_ms"), "ttfa_pred_ms": round(p["ttfa_s"] * 1e3, 1),
                    "shares": {k: round(v, 3) for k, v in p["shares"].items()}})
    return out


def summary(t: list[dict]) -> dict:
    lr = [math.log(x["ratio"]) for x in t]
    return {"n": len(t), "gmean_ratio": round(math.exp(sum(lr) / len(lr)), 3),
            "ratio_c1": t[0]["ratio"], "ratio_cmax": t[-1]["ratio"]}
