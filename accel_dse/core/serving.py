"""L6 — serving metrics on top of single-step evaluation.

Goodput with prefill amortisation (aggregated serving, time-shared replica 「假设」):
  each request = S prompt tokens (prefill) + out_len generated tokens (decode).
  R_d = decode tokens/s at the chosen batch (TPOT ≤ SLO), R_p = prefill
  prompt-tokens/s at the largest prefill batch meeting the TTFT SLO.
  goodput = 1 / (1/R_d + (S / out_len) / R_p)        [output tokens / s / replica]
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate import Result, evaluate
from .scenario import Scenario


@dataclass
class Goodput:
    decode_tok_s: float
    prefill_tok_s: float
    prefill_batch: int
    ttft_ms: float
    goodput_tok_s: float
    goodput_per_card: float
    decode_share: float      # fraction of replica time spent decoding


def best_prefill(scn: Scenario, b_cap: int = 64) -> tuple[int, Result | None]:
    base = scn.replace("serving.phase", "prefill")
    best_b, best_r = 0, None
    b = 1
    while b <= b_cap:
        r = evaluate(base.replace("serving.batch", b))
        if not (r.fits and r.ttft * 1e3 <= scn.serving.ttft_slo_ms):
            break
        if best_r is None or r.throughput > best_r.throughput:
            best_b, best_r = b, r
        b *= 2
    return best_b, best_r


def goodput(decode: Result) -> Goodput:
    scn = decode.scenario
    pb, pr = best_prefill(scn)
    rd = decode.throughput
    if pr is None or rd <= 0:
        return Goodput(rd, 0.0, 0, float("inf"), 0.0, 0.0, 0.0)
    rp = pr.throughput
    S, out = scn.serving.prompt, scn.serving.out_len
    g = 1.0 / (1.0 / rd + (S / out) / rp)
    share = (1.0 / rd) / (1.0 / rd + (S / out) / rp)
    return Goodput(rd, rp, pb, pr.ttft * 1e3, g, g / scn.layout.cards, share)
