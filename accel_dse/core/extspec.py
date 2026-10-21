"""0.295: external speculative-decoding measurements — LFM2.5-24B-A2B (FP8) with the LiquidAI DSpark draft on one B300
(model repo benchmarks/b300-fp8.csv + protocol.json, 8 datasets × concurrency 1 … 256).

Diagnostic only (like extspeech): nothing here is fitted, nothing feeds the cost model.  The question (AGENTS §244):
the catalog chips predict a b1 DSpark speed-up of 1.11–1.17×, the release reports 2.38× — which mechanism explains it?

    (a) expert union of the verified block: the target verifies n = k + 1 = 10 tokens per sequence; uniform routing
        touches E·(1 − (1 − k_e/E)^(c·n)) experts (ir.moe_routing 「假设」) instead of E·(1 − (1 − k_e/E)^c)
    (b) expert locality across consecutive tokens: the n tokens of one sequence re-use experts.  Upper bound
        (``locality_bound``): the block routes like ONE token per sequence — the union expert bytes of the step drop to
        the baseline step's, DRAM time scaled by bytes 「近似」 (no public LFM2 routing statistic → bound only)
    (c) hardware baseline: the B300 run is far from bandwidth-bound.  Variant "default" = extval's per-SM GPU path with
        the trace-sourced per-kernel floor (MODEL §39 / §119; not fitted); "catalog" = the bandwidth roofline

Closed loop at concurrency c (every stream restarts as soon as it ends; mean field, as extspeech) 「推导」:
    tokens/s = c · OUT / (c · ttft₁(PROMPT) + OUT / τ · step(c))         τ = 1 for the target alone
step(c) = evaluate(decode, batch c, ctx = PROMPT + OUT/2) (with spec_k 9, spec_accept from the row's measured τ for
DSpark — spec_accept_from_tau; the measured τ is an input, like the speech diagnostic's measured stride schedule).
Hardware: the extval B200 reference chip as a B300 proxy 「近似」 — HGX B300 / B200 list the same dense FP8 (72 PF / 8
GPUs sparse → 4.5 PF dense) and NVLink 5 (1.8 TB/s); both 8 TB/s HBM3e; B300 has 160 SMs vs 148 and 288 vs 180 GB
(not binding for a 24B FP8 model).  PROMPT = 256 tokens 「假设」 (prompt lengths not published).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "ext_measurements_spec.json"
TARGET, PAIR, HW = "lfm2.5-24b-a2b-fp8", "lfm2.5-24b-a2b-fp8-dspark", "B200"
K = 9                      # README / protocol: --speculative-dspark-block-size 9, 10 draft tokens (τ ceiling 10)
PROMPT = 256               # 「假设」
CTX_Q = 64                 # ctx rounded to a multiple (memoisation; < 3 % of the shortest mean context)


def rows() -> list[dict]:
    return json.loads(DATA.read_text())["rows"]


def _ctx(out: float) -> int:
    return max(CTX_Q, int(round((PROMPT + out / 2) / CTX_Q)) * CTX_Q)


@lru_cache(maxsize=4096)
def _step(variant: str, c: int, ctx: int, accept: float) -> dict:
    """One decode step of batch c (accept < 0: the target alone)."""
    from . import extval as X
    from .evaluate import evaluate
    model = TARGET if accept < 0 else PAIR
    sv = dict(phase="decode", batch=c, ctx=ctx)
    if accept >= 0:
        sv.update(spec_k=K, spec_accept=accept)
    res = evaluate(X.scenario(dict(hw=HW, model=model, tp=1, dtype="release"), variant, **sv))
    st = res.stages[res.heaviest_stage]
    d = st.dram if isinstance(st.dram, dict) else vars(st.dram)
    bw = d["total"] / st.time.t_dram if st.time.t_dram > 0 else math.inf     # bytes / s actually achieved in the step
    return {"step": res.step, "tps": res.tokens_per_step, "exp_bytes": d.get("exp_dram", 0.0), "bytes": d["total"],
            "bw": bw, "t_kern": st.time.t_kern, "t_dram": st.time.t_dram, "fits": res.fits}


@lru_cache(maxsize=64)
def _ttft(variant: str) -> float:
    from . import extval as X
    from .evaluate import evaluate
    return evaluate(X.scenario(dict(hw=HW, model=TARGET, tp=1, dtype="release"), variant, phase="prefill", batch=1,
                               prompt=PROMPT, ctx=PROMPT)).ttft


def predict(r: dict, variant: str = "default") -> dict:
    """Predicted target-only / DSpark tokens/s of one measurement row, and the locality upper bound."""
    from .model import spec_accept_from_tau
    c, tau = r["concurrency"], r["tau"]
    a = round(spec_accept_from_tau(min(tau, K + 1 - 1e-9), K), 4)
    b = _step(variant, c, _ctx(r["out_baseline"]), -1.0)
    s = _step(variant, c, _ctx(r["out_dspark"]), a)
    t1 = _ttft(variant)

    def tok_s(step, tps, out):
        return c * out / (c * t1 + out / tps * step)
    base = tok_s(b["step"], 1.0, r["out_baseline"])
    spec = tok_s(s["step"], s["tps"], r["out_dspark"])
    # (b) bound: the verified block's union expert bytes → the target-alone step's expert bytes (same c sequences)
    saved = max(0.0, s["exp_bytes"] - b["exp_bytes"]) / s["bw"] if s["bw"] < math.inf else 0.0
    loc = tok_s(max(s["step"] - saved, 1e-12), s["tps"], r["out_dspark"])
    return {"c": c, "accept": a, "tau_model": s["tps"], "base_tok_s": base, "spec_tok_s": spec, "ratio": spec / base,
            "ratio_locality_bound": loc / base, "step_base_ms": b["step"] * 1e3, "step_spec_ms": s["step"] * 1e3,
            "exp_union": s["exp_bytes"] / b["exp_bytes"] if b["exp_bytes"] else 1.0,
            "kern_share_base": b["t_kern"] / b["step"] if b["step"] else 0.0}


def table(variant: str = "default") -> list[dict]:
    out = []
    for r in rows():
        p = predict(r, variant)
        out.append({"id": r["id"], "c": r["concurrency"], "dataset": r["dataset"],
                    "base_meas": r["baseline_tok_s"], "base_pred": round(p["base_tok_s"], 1),
                    "base_ratio": round(p["base_tok_s"] / r["baseline_tok_s"], 3),
                    "speedup_meas": r["ratio"], "speedup_pred": round(p["ratio"], 3),
                    "speedup_locality_bound": round(p["ratio_locality_bound"], 3),
                    "exp_union": round(p["exp_union"], 2), "kern_share_base": round(p["kern_share_base"], 3)})
    return out


def by_concurrency(t: list[dict]) -> list[dict]:
    """Arithmetic means over the 8 datasets per concurrency (the README's 'Arithmetic mean' row convention)."""
    out = []
    for c in sorted({x["c"] for x in t}):
        g = [x for x in t if x["c"] == c]
        m = lambda k: round(sum(x[k] for x in g) / len(g), 3)
        out.append({"c": c, "n": len(g), "speedup_meas": m("speedup_meas"), "speedup_pred": m("speedup_pred"),
                    "speedup_locality_bound": m("speedup_locality_bound"), "base_ratio": m("base_ratio"),
                    "exp_union": m("exp_union"), "kern_share_base": m("kern_share_base")})
    return out


def summary(t: list[dict]) -> dict:
    lr = [math.log(x["speedup_pred"] / x["speedup_meas"]) for x in t]
    lb = [math.log(x["base_ratio"]) for x in t]
    return {"n": len(t), "speedup_gmean_ratio": round(math.exp(sum(lr) / len(lr)), 3),
            "base_gmean_ratio": round(math.exp(sum(lb) / len(lb)), 3)}
