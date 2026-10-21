"""0.282: external speech measurements — Fish Audio Drama 3 on one H200 (SGLang-Omni cookbook, Tables 1–3).

Diagnostic only (like extval variants): the engine is evaluated on the H200 reference chip under an ext variant and the
published serving loop is reproduced from its sourced settings; nothing here is fitted, and nothing feeds the cost model.

Closed loop at concurrency c (every stream restarts as soon as it ends, steady state) 「推导」:
    one cycle of c requests = c prefills (batch 1, serial 「近似」: SGLang batches up to 16 first frames, and
    prefill_decode_interval 4 keeps them out of decode rounds) + N decode steps at batch c + the vocoder's work
    (same GPU, own process → time-shared, loads add 「假设」)
    frames/s = c·N / (c·ttft₁ + N·tpot(c) + c·vocoder_per_request)
    TTFA (0.294, closed-loop queueing; the 0.282 no-queueing form is kept as ``ttfa_noq_s``) 「推导」:
        a        = (c − 1) / N            other streams' arrivals per decode round (each stream restarts every N rounds)
        T_round  = tpot(c) + a · (ttft₁ + vocoder_per_request)   a decode round stretched by the others' prefills
                   (and, vocoder on the same GPU, their windows — time-shared, loads add), i.e. the mean-field share
        p_win    = I·a / (I·a + 1)        chance an arrival finds the prefill gate shut: SGLang prefill_decode_interval
                   I = 4 「来源」 (engine_builder.py DECODE_STEPS_BETWEEN_PREFILLS; sglang scheduler: after a prefill batch
                   the next I rounds defer prefill, then prefill runs first if any request waits); a gate-open period ends
                   with exactly one arrival, which is prefilled at once
        W_gate   = p_win · (I − 1)/2 · T_round       wait for the gate (arrival uniform over the I deferred rounds)
        k        = 1 + p_win · I · a       requests prefilled in the same batch (all waiting ones; first frame for all at
                   the end; serial cost k·ttft₁ as in the throughput cycle 「近似」; the 16-request first-frame graph cap
                   is never reached here: I·a ≤ 10.6 at c = 256)
        W_voc    = λ·E[s²] / (2(1 − ρ))    the first window queued behind the others' windows at the vocoder (one
                   server, windows decode one at a time; M/G/1 Pollaczek–Khinchine 「近似」, s = DAC window GPU seconds)
        TTFA     = W_gate + k·ttft₁ + stream_stride · T_round + W_voc + vocoder(first window)
    The vocoder on a second GPU still decodes the first window there (0.282 dropped it from TTFA).
    Not modelled in TTFA: host / IPC hops (the c = 1 residual).
0.295 start burst (``burst``, synchronized arrivals — the benchmark starts all c streams at t = 0) 「推导」:
    SGLang prefills every waiting request in one forward up to max_prefill_tokens (default 16384 「来源」 sglang
    server_args; real bound max(16384, context_length 16384); Drama 3 disables chunked prefill) → batches of
    B = ⌊16384 / prompt rows⌋ requests; between batches the gate runs 4 decode rounds at the running batch R.  Batch j
    gets its first frame at the end of its (batched) prefill P(n_j) and its first audio after `stride` more decode
    rounds (later batches' prefills interleave) plus its first window: the n_j first windows reach the vocoder
    together and decode one at a time (same GPU: on the shared timeline; second GPU: on its own).  Deterministic
    schedule → wave-1 mean / max TTFA.  Only the first wave (c of the run's 1,088 requests) starts synchronized.
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

MAX_PREFILL_TOKENS = 16384     # sglang server_args max_prefill_tokens default (Drama 3 context_length 16384)
RUN_REQUESTS = 1088            # cookbook Tables 1–2: 1,088 requests per concurrency level
PREFILL_INTERVAL = 4          # sglang_omni fishaudio_drama3/engine_builder.py DECODE_STEPS_BETWEEN_PREFILLS @ 0029194
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
    if vocoder in ("none", "sglang"):
        # "none": vocoder on a second GPU (Table 2) — the engine's GPU runs prefill + decode only; the SGLang windows
        # still decode there (one at a time), so the first window and its queue stay in TTFA
        win = sglang_windows(N)
        ws = [_dac_window(fs, n, ctx, done, 1, sys, scn.mapping) for n, ctx, done in win]
        voc_req = 0.0 if vocoder == "none" else sum(ws)
        voc_first, first_frames = ws[0], STRIDE
    else:
        voc_req = sum(kc["dac"]["s"]) / c          # the c streams' windows batched
        voc_first = kc["dac"]["s_batch1"][0]
        first_frames = CHUNK_RAMP[0]
        ws = [voc_req]                             # one batched window per stream-window 「近似」 for the queue term
    ttft1, tpot = k1["ttft_s"], kc["tpot_s"]
    cycle = c * ttft1 + N * tpot + c * voc_req
    q = _ttfa_queue(c, N, ttft1, tpot, voc_req, ws, cycle, first_frames)     # c / cycle requests per second
    return {"c": c, "frames": N, "ttft1_s": ttft1, "tpot_s": tpot, "vocoder_s_per_request": voc_req,
            "frames_per_s": c * N / cycle, "ttfa_s": q["total"] + voc_first,
            "ttfa_noq_s": ttft1 + first_frames * tpot + voc_first,
            "ttfa_parts": {**{k: v for k, v in q.items() if k != "total"}, "vocoder_first": voc_first},
            "shares": {"prefill": c * ttft1 / cycle, "decode": N * tpot / cycle, "vocoder": c * voc_req / cycle},
            "fits": kc["fits"]}


def _ttfa_queue(c: int, N: int, ttft1: float, tpot: float, voc_req: float, ws: list[float], cycle: float,
                first_frames: int, interval: int = PREFILL_INTERVAL) -> dict:
    """0.294 closed-loop TTFA terms (module docstring) without the first window's own decode, in seconds."""
    a = (c - 1) / N
    t_round = tpot + a * (ttft1 + voc_req)
    p_win = interval * a / (interval * a + 1)
    gate = p_win * (interval - 1) / 2 * t_round
    k = 1 + p_win * interval * a
    lam = c * len(ws) / cycle                       # windows per second reaching the vocoder
    es, es2 = sum(ws) / len(ws), sum(x * x for x in ws) / len(ws)
    rho = lam * es
    wv = lam * es2 / (2 * (1 - rho)) if rho < 1 else math.inf
    return {"gate": gate, "prefill": k * ttft1, "prefill_batch": k, "decode": first_frames * t_round,
            "t_round": t_round, "vocoder_queue": wv, "vocoder_util": rho,
            "total": gate + k * ttft1 + first_frames * t_round + wv}


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
                    "ttfa_noq_ms": round(p["ttfa_noq_s"] * 1e3, 1),
                    # measured-only: the first `stride` frames at the measured per-stream frame rate (c / frames/s) 「推导」
                    "ttfa_rate_ms": round(STRIDE * r["concurrency"] / r["value"] * 1e3, 1),
                    "ttfa_burst_wave1_ms": round(burst(variant, r["concurrency"], voc if voc != "vllm" else "sglang")["wave1_mean_s"] * 1e3, 1),
                    "ttfa_ratio": round(p["ttfa_s"] * 1e3 / r["ttfa_p50_ms"], 3) if r.get("ttfa_p50_ms") else None,
                    "shares": {k: round(v, 3) for k, v in p["shares"].items()}})
    return out


def summary(t: list[dict]) -> dict:
    lr = [math.log(x["ratio"]) for x in t]
    return {"n": len(t), "gmean_ratio": round(math.exp(sum(lr) / len(lr)), 3),
            "ratio_c1": t[0]["ratio"], "ratio_cmax": t[-1]["ratio"],
            "ttfa_gmean_ratio": round(math.exp(sum(math.log(x["ttfa_ratio"]) for x in t) / len(t)), 3)}


def burst(variant: str, c: int, vocoder: str = "sglang") -> dict:
    """0.295 wave-1 TTFA of c synchronized arrivals (module docstring), seconds; ``vocoder`` "sglang" (same GPU) or
    "none" (second GPU).  Uses the batched prefill P(n) = speech_kpis(batch n).ttft_s, decode rounds tpot(R)."""
    from .catalog import get_model
    from .hardware import System
    from .speech import _dac_window, speech_kpis
    scn = _scenario(variant, 1)
    kp = {}

    def k(b):
        if b not in kp:
            kp[b] = speech_kpis(scn.replace("serving.batch", b), OUT_S)
        return kp[b]
    rows_p = k(1)["prompt_rows"]
    B = max(1, MAX_PREFILL_TOKENS // rows_p)
    sizes = [min(B, c - j * B) for j in range(-(-c // B))]
    fs = get_model("drama-3").fish
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, 1, scn.mem_cap_GB, scn.kv_mem_frac, scn.kv_mem_eff)
    s0 = _dac_window(fs, STRIDE, 0, 0, 1, sys, scn.mapping)
    same = vocoder == "sglang"
    st = {"t": 0.0, "R": 0, "voc_free": 0.0}
    pending, first = [], []

    def rounds(m):
        for _ in range(m):
            if st["R"] == 0:
                return
            st["t"] += k(st["R"])["tpot_s"]
            for p in pending:
                p[1] -= 1
            for p in [p for p in pending if p[1] <= 0]:
                pending.remove(p)
                if same:                                   # windows on the shared timeline, one at a time
                    first.extend(st["t"] + (i + 1) * s0 for i in range(p[0]))
                    st["t"] += p[0] * s0
                else:                                      # second GPU: its own queue
                    t0 = max(st["t"], st["voc_free"])
                    first.extend(t0 + (i + 1) * s0 for i in range(p[0]))
                    st["voc_free"] = t0 + p[0] * s0
    for j, n in enumerate(sizes):
        if j:
            rounds(PREFILL_INTERVAL)
        st["t"] += k(n)["ttft_s"]
        st["R"] += n
        pending.append([n, STRIDE])
    while pending:
        rounds(1)
    return {"c": c, "batch_cap": B, "batches": sizes, "wave1_mean_s": sum(first) / len(first), "wave1_max_s": max(first),
            "wave1_share": min(1.0, c / RUN_REQUESTS)}
