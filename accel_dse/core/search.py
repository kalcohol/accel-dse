"""L7 — exact search over batch × layout × mapping, Pareto front.

Batch search (per layout): maximise throughput B·E/step(B) subject to
TPOT ≤ SLO and capacity.  Exactness relies on two properties that the test
suite certifies by brute force on every catalog family:
  (P1) step(B) is non-decreasing in B;
  (P2) feasibility (fits ∧ TPOT ≤ SLO) is monotone (true ⇒ true for smaller B).
Given P1, thr(B) ≤ hi·E/step(lo) for B ∈ [lo, hi]; branch-and-bound on that
bound returns the exact maximiser with far fewer evaluations than a scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import get_model
from .evaluate import Result, evaluate
from .parallel import Layout, enumerate_layouts
from .scenario import Scenario

B_CAP = 4096


@dataclass
class BatchBest:
    batch: int
    result: Result | None
    evals: int
    b_max: int


def _feasible(r: Result) -> bool:
    return r.fits and r.tpot * 1e3 <= r.scenario.serving.tpot_slo_ms


def best_batch(base: Scenario, b_cap: int = B_CAP) -> BatchBest:
    memo: dict[int, Result] = {}

    def ev(b: int) -> Result:
        if b not in memo:
            memo[b] = evaluate(base.replace("serving.batch", b))
        return memo[b]

    if not _feasible(ev(1)):
        return BatchBest(0, ev(1), len(memo), 0)
    # exponential + binary search for the largest feasible batch (P2)
    lo, hi = 1, 2
    while hi <= b_cap and _feasible(ev(hi)):
        lo, hi = hi, hi * 2
    hi = min(hi, b_cap + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _feasible(ev(mid)):
            lo = mid
        else:
            hi = mid
    b_max = lo
    thr = lambda b: ev(b).throughput
    best_b = b_max
    best = thr(b_max)
    stack = [(1, b_max)]
    while stack:
        a, b = stack.pop()
        if a > b:
            continue
        ub = b * ev(a).tokens_per_step / ev(a).step     # (P1) upper bound on [a, b]
        if ub <= best * (1 + 1e-12):
            continue
        if b - a <= 2:
            for x in range(a, b + 1):
                if thr(x) > best * (1 + 1e-12) or (abs(thr(x) - best) <= 1e-12 * best and x < best_b):
                    best, best_b = thr(x), x
            continue
        mid = (a + b) // 2
        if thr(mid) > best * (1 + 1e-12):
            best, best_b = thr(mid), mid
        stack += [(a, mid), (mid + 1, b)]
    return BatchBest(best_b, ev(best_b), len(memo), b_max)


def brute_force_batch(base: Scenario, b_cap: int) -> tuple[int, float]:
    best_b, best = 0, -1.0
    for b in range(1, b_cap + 1):
        r = evaluate(base.replace("serving.batch", b))
        if not _feasible(r):
            continue
        if r.throughput > best * (1 + 1e-12):
            best, best_b = r.throughput, b
    return best_b, best


@dataclass
class LayoutRow:
    layout: Layout
    mapping: str
    batch: int
    result: Result | None

    @property
    def per_card(self) -> float:
        return self.result.per_card if (self.result is not None and self.batch) else 0.0


def search_layouts(base: Scenario, cards: int, mappings: tuple[str, ...] | None = None,
                   max_tp: int | None = None) -> list[LayoutRow]:
    m = get_model(base.model)
    rows = []
    for org in (mappings or (base.mapping,)):
        for lay in enumerate_layouts(cards, m.n_layers, m.is_moe, max_tp=max_tp):
            scn = base.replace("layout", lay).replace("mapping", org)
            bb = best_batch(scn)
            rows.append(LayoutRow(lay, org, bb.batch, bb.result if bb.batch else None))
    rows.sort(key=lambda r: -r.per_card)
    return rows


def pareto(points: list[tuple[float, float, object]]) -> list[tuple[float, float, object]]:
    """Non-dominated set for (minimise x, maximise y)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out, best_y = [], -float("inf")
    for p in pts:
        if p[1] > best_y:
            out.append(p)
            best_y = p[1]
    return out


def tpot_throughput_front(base: Scenario, batches: list[int] | None = None) -> list[dict]:
    batches = batches or [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    pts = []
    for b in batches:
        r = evaluate(base.replace("serving.batch", b))
        if r.fits:
            pts.append((r.tpot * 1e3, r.per_card, b))
    return [{"tpot_ms": x, "tok_s_card": y, "batch": b} for x, y, b in pareto(pts)]
