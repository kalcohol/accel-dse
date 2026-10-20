"""0.284: per-op dispatch sensitivity of LLM decode (the 0.283 speech read-out, for any catalog LLM).

The NPU dispatch design variable (Chip.dispatch / dispatch_us, 0.203; "trace" 1.5 µs per op 「二手」 tt-metal #20681,
"host" 20 µs per op 「二手」 vllm-ascend #13021, "static" 10 µs per step 「二手」 JAX jit) defaults to ideal (0).  At
small batch a decode step is many small ops, so a per-op dispatch / sync latency u can matter as much as bandwidth.

Dispatch is a serial term of every stage (t_stage = max(compute, DRAM, SLC, link) + t_sync), so the decode step of a
single-stage layout is exactly linear in u 「推导」:

    TPOT(u) = TPOT(0) + n · u        n = dispatched ops per decode step, measured by the engine:
                                          n = (TPOT(1 µs) − TPOT(0)) / 1 µs   (each sequence's token pays all n)
    u_10 = 0.1 · TPOT(0) / n          per-op latency at which TPOT degrades by 10 %
    u_eq = TPOT(0) / n                per-op latency at which dispatch equals all other step time

TPOT at the sourced trace / host / static defaults are separate engine runs (not the line), so a multi-stage or
pipelined layout, where the slope need not be exact, still reports its true values.  Nothing is fitted."""
from __future__ import annotations

import dataclasses

from .evaluate import evaluate
from .hardware import DISPATCH_US
from .scenario import Scenario

LLM_MODELS = ("qwen3-0.6b", "llama-3.2-1b", "qwen3-1.7b", "qwen3-8b", "llama-3.1-8b", "qwen3-30b-a3b", "gpt-oss-20b")
BATCHES = (1, 8, 32)
CTX = 1024                 # decode context 「假设」
MODES = ("trace", "static", "host")


def _with(base: Scenario, mode: str, us: float | None) -> Scenario:
    return base.replace("chip", dataclasses.replace(base.chip, dispatch=mode, dispatch_us=us))


def _decode(base: Scenario, mid: str, b: int, ctx: int) -> Scenario:
    import dataclasses as dc
    from .scenario import Serving
    dflt = {f.name: (f.default_factory() if f.default_factory is not dc.MISSING else f.default)
            for f in dc.fields(Scenario) if f.name in ("layout", "formats_override", "workload", "pd", "pp_split")}
    sv = Serving(batch=b, phase="decode", ctx=ctx)
    return dc.replace(base, model=mid, serving=sv, **dflt)


def step_sensitivity(scn: Scenario, r0=None) -> dict:
    """Linear per-op sensitivity of one scenario's step (decode: TPOT; prefill: the prefill step) — two engine runs
    (ideal, 1 µs/op).  ``r0``: an existing ideal-dispatch result to reuse.  ``exact``: single stage, one microbatch
    (the step is then exactly linear in u; a pipelined step is a max over stages and the line is a 「近似」)."""
    if r0 is None or scn.chip.dispatch:
        r0 = evaluate(_with(scn, "", None))
    t0 = r0.step
    n = max(0.0, (evaluate(_with(scn, "trace", 1.0)).step - t0) / 1e-6)
    us = {m: DISPATCH_US[m] for m in ("trace", "host")}
    return {"ops_per_step": n, "step_ms": t0 * 1e3,
            "us_10": 0.1 * t0 / (n * 1e-6) if n > 0 else None, "us_eq": t0 / (n * 1e-6) if n > 0 else None,
            "slowdown_at": {m: (t0 + n * u * 1e-6) / t0 if t0 > 0 else None for m, u in us.items()}, "us_at": us,
            "exact": len(r0.stages) == 1 and r0.microbatches == 1, "chip_dispatch": scn.chip.dispatch or "ideal"}


def dispatch_row(scn: Scenario) -> dict:
    """Sensitivity of one decode scenario (its own chip / memory / layout)."""
    r0 = evaluate(_with(scn, "", None))
    t0 = r0.tpot
    t1 = evaluate(_with(scn, "trace", 1.0)).tpot
    n = max(0.0, (t1 - t0) / 1e-6)
    at = {m: evaluate(_with(scn, m, None)).tpot for m in MODES}
    b = scn.serving.batch
    return {"model": scn.model, "batch": b, "tpot_ms": t0 * 1e3, "bound": r0.bound, "fits": r0.fits,
            "ops_per_step": n, "ops_per_token": n / b if b else n,
            "us_10": 0.1 * t0 / (n * 1e-6) if n > 0 else None,
            "us_eq": t0 / (n * 1e-6) if n > 0 else None,
            "tpot_ms_at": {m: v * 1e3 for m, v in at.items()},
            "slowdown_at": {m: v / t0 for m, v in at.items()}}


def dispatch_sensitivity(base: Scenario, models=LLM_MODELS, batches=BATCHES, ctx: int = CTX) -> dict:
    rows = []
    for mid in models:
        for b in batches:
            rows.append(dispatch_row(_decode(base, mid, b, ctx)))
    return {"chip": base.chip.name, "mem_id": base.mem_id, "ctx": ctx, "chip_dispatch": base.chip.dispatch or "ideal",
            "us_at": {m: DISPATCH_US[m] for m in MODES}, "rows": rows,
            "method": "TPOT(u) = TPOT(0) + n·u（n = 每 decode 步下发 op 数，引擎 0 / 1 µs·op⁻¹ 两次实算）；"
                      "u_10 = 0.1·TPOT(0)/n（TPOT 劣化 10 % 的每 op 延迟），u_eq = TPOT(0)/n「推导」；"
                      "trace / static / host 列为引擎按来源默认值实算「二手」"}
