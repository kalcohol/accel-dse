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
    ttft_ok: bool = True     # False → even a single-request prefill misses the TTFT SLO (goodput still shown)


def best_prefill(scn: Scenario, b_cap: int = 64) -> tuple[int, Result | None]:
    """Largest-throughput prefill batch (powers of two) meeting the TTFT SLO; (0, None) if none."""
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


_PREFILL_MEMO: dict = {}


def _prefill(scn: Scenario) -> tuple[int, Result, bool]:
    """(prefill batch, result, ttft_ok) used by goodput; memoised per scenario (batch-independent)."""
    key = scn.replace("serving.batch", 1).replace("serving.phase", "decode")
    hit = _PREFILL_MEMO.get(key)
    if hit is None:
        pb, pr = best_prefill(scn)
        ok = pr is not None
        if pr is None:
            pb, pr = 1, evaluate(scn.replace("serving.phase", "prefill").replace("serving.batch", 1))
        if len(_PREFILL_MEMO) > 4096:
            _PREFILL_MEMO.clear()
        hit = _PREFILL_MEMO[key] = (pb, pr, ok)
    return hit


def prefill_rate(scn: Scenario) -> tuple[float, bool]:
    """Prompt tokens/s per replica entering the goodput formula (0 if prefill does not fit)."""
    pb, pr, ok = _prefill(scn)
    return (pr.throughput if pr.fits else 0.0), ok


def goodput(decode: Result) -> Goodput:
    scn = decode.scenario
    pb, pr, ok = _prefill(scn)
    rd = decode.throughput
    if rd <= 0 or not pr.fits:
        return Goodput(rd, 0.0, 0, float("inf"), 0.0, 0.0, 0.0, False)
    rp = pr.throughput
    S, out = scn.serving.prompt, scn.serving.out_len
    g = 1.0 / (1.0 / rd + (S / out) / rp)
    share = (1.0 / rd) / (1.0 / rd + (S / out) / rp)
    return Goodput(rd, rp, pb, pr.ttft * 1e3, g, g / scn.layout.cards, share, ok)
