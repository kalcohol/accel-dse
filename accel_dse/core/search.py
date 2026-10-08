"""L7 — exact search over batch × layout × mapping, Pareto front.

Batch search (per layout): maximise throughput B·E/step(B) subject to
the latency SLO (LLM: TPOT; video: clip latency; protein: batch latency) and capacity.  Exactness relies on two properties that the test
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
    return r.fits and r.slo_ok


class _BatchSearch:
    """Batch search state for one scenario (evaluations memoised across the two phases)."""

    def __init__(self, base: Scenario, b_cap: int = B_CAP):
        self.base, self.b_cap, self.memo = base, b_cap, {}
        self.b_max = None
        self._lo = self._hi = None          # b_max ∈ [_lo, _hi) while the bracket is open

    def ev(self, b: int) -> Result:
        r = self.memo.get(b)
        if r is None:
            r = self.memo[b] = evaluate(self.base.replace("serving.batch", b))
        return r

    def thr(self, b: int) -> float:
        return self.ev(b).throughput

    def _piecewise(self, hi: int) -> float:
        """Bound on max thr(B), B ≤ hi, from every evaluated feasible point (P1):
        thr(B) ≤ (p_{i+1} − 1)·E/step(p_i) for B ∈ [p_i, p_{i+1})."""
        pts = sorted(b for b, r in self.memo.items() if b <= hi and _feasible(r))
        ub = 0.0
        for i, p in enumerate(pts):
            nxt = pts[i + 1] - 1 if i + 1 < len(pts) else hi
            r = self.memo[p]
            ub = max(ub, nxt * r.tokens_per_step / r.step)
        return ub

    def _segment(self, a: int, b: int) -> float:
        """Piecewise P1 bound on thr over [a, b] ⊆ [1, b_max] from the evaluated points in it."""
        self.ev(a)
        pts = sorted(x for x in self.memo if a <= x <= b)
        ub = 0.0
        for i, p in enumerate(pts):
            nxt = pts[i + 1] - 1 if i + 1 < len(pts) else b
            r = self.memo[p]
            ub = max(ub, nxt * r.tokens_per_step / r.step)
        return ub

    def bound(self, floor: float | None = None) -> float:
        """Valid upper bound on the best feasible throughput (tokens/s per replica).

        The b_max bracket (P2: exponential, then binary search) is only refined while
        the bound could still beat ``floor``; ``floor=None`` refines to the exact b_max,
        ``floor=inf`` stops after the exponential phase."""
        if self.b_max is not None:
            return self._piecewise(self.b_max) if self.b_max else 0.0
        if self._lo is None:
            if not _feasible(self.ev(1)):
                self.b_max = 0
                return 0.0
            lo, hi = 1, 2
            while hi <= self.b_cap and _feasible(self.ev(hi)):
                lo, hi = hi, hi * 2
            self._lo, self._hi = lo, min(hi, self.b_cap + 1)
        while True:
            ub = self._piecewise(self._hi - 1)
            if self._hi - self._lo <= 1:
                self.b_max = self._lo
                return ub
            if floor is not None and ub <= floor * (1 + 1e-12):
                return ub
            mid = (self._lo + self._hi) // 2
            if _feasible(self.ev(mid)):
                self._lo = mid
            else:
                self._hi = mid

    def find_bmax(self) -> int:
        """Largest feasible batch; 0 if batch 1 is infeasible."""
        self.bound()
        return self.b_max

    def upper_bound(self) -> float:
        return self.bound()

    def solve(self, floor: float = 0.0) -> int:
        """Exact throughput-maximising batch, or 0 if it cannot beat ``floor`` (tokens/s per replica)."""
        if self.bound(floor) <= floor * (1 + 1e-12):
            return 0
        b_max = self.find_bmax()
        if not b_max:
            return 0
        best_b, best = 0, 0.0
        inc = floor
        if floor <= 0.0:
            best_b, best = b_max, self.thr(b_max)
            inc = best
        stack = [(1, b_max)]
        while stack:
            a, b = stack.pop()
            if a > b:
                continue
            ub = self._segment(a, b)
            if ub <= inc * (1 + 1e-12):
                continue
            if b - a <= 2:
                for x in range(a, b + 1):
                    t = self.thr(x)
                    if t > inc * (1 + 1e-12) or (best_b and abs(t - best) <= 1e-12 * best and x < best_b):
                        best, best_b = t, x
                        inc = max(inc, t)
                continue
            mid = (a + b) // 2
            t = self.thr(mid)
            if t > inc * (1 + 1e-12):
                best, best_b, inc = t, mid, t
            stack += [(a, mid), (mid + 1, b)]
        return best_b


def best_batch(base: Scenario, b_cap: int = B_CAP) -> BatchBest:
    s = _BatchSearch(base, b_cap)
    b_max = s.find_bmax()
    if not b_max:
        return BatchBest(0, s.ev(1), len(s.memo), 0)
    b = s.solve()
    return BatchBest(b, s.ev(b), len(s.memo), b_max)


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
    goodput: object = None      # serving.Goodput when objective == "goodput"

    @property
    def per_card(self) -> float:
        return self.result.per_card if (self.result is not None and self.batch) else 0.0

    @property
    def goodput_card(self) -> float:
        return self.goodput.goodput_per_card if self.goodput is not None else 0.0

    def score(self, objective: str) -> float:
        return self.goodput_card if objective == "goodput" else self.per_card


def search_layouts(base: Scenario, cards: int, mappings: tuple[str, ...] | None = None,
                   max_tp: int | None = None, objective: str = "decode", top: int | None = None,
                   stats: dict | None = None, floor: float = 0.0) -> list[LayoutRow]:
    """Rank layouts on ``cards`` cards.  objective:
    decode  – decode tokens/s/card at the best batch meeting the TPOT SLO
    goodput – output tokens/s/card incl. prefill amortisation (serving.goodput); the
              decode-best batch also maximises goodput for a layout (goodput is monotone in R_d)

    ``top=k`` returns only the exact top-k rows: a cheap upper bound per layout
    (exponential b_max bracket + piecewise P1 bound) orders the layouts; each is then
    solved with the k-th best exact score as floor, refining its bound only while it
    could still win (branch and bound across layouts; certified against the full
    ranking in the tests).  ``floor`` (score per card, with ``top``) seeds the bound:
    only layouts that strictly beat it are returned (used by the stability check with
    the base top-1 layout's exact score as the incumbent).
    """
    if objective not in ("decode", "goodput"):
        raise ValueError("objective must be decode|goodput")
    from .serving import goodput as _goodput, prefill_rate as _prefill_rate
    m = get_model(base.model)
    if objective == "goodput" and not m.kv_cache:
        raise ValueError("goodput 目标仅适用于 LLM / VLM（视频 / 蛋白质模型用 decode 目标：单位/s/卡）")
    cands = []
    for org in (mappings or (base.mapping,)):
        for lay in enumerate_layouts(cards, m.n_layers, m.is_moe, max_tp=max_tp, full=not m.kv_cache,
                                     pair=m.is_pair):
            cands.append((lay, org, _BatchSearch(base.replace("layout", lay).replace("mapping", org))))
    rows: list[LayoutRow] = []

    def kth() -> float:
        if top is None:
            return 0.0
        if len(rows) < top:
            return floor
        return max(floor, sorted((r.score(objective) for r in rows), reverse=True)[top - 1])

    if top is not None:
        cands.sort(key=lambda c: -c[2].bound(float("inf")))
    pruned = 0
    for lay, org, bs in cands:
        need = kth() * cards                                # per-replica tokens/s to beat
        if top is not None and bs.bound(need) <= need * (1 + 1e-12):
            if _feasible(bs.ev(1)):
                pruned += 1
            else:                       # infeasible at batch 1 (capacity / SLO): reported as a batch-0 row
                rows.append(LayoutRow(lay, org, 0, None))
            continue
        if objective == "goodput" and need > 0:
            rp, ok = _prefill_rate(bs.base)
            c = bs.base.serving.prompt / bs.base.serving.out_len
            denom = 1.0 / need - (c / rp if rp > 0 else float("inf"))
            if rp <= 0 or denom <= 0:
                pruned += 1
                continue
            need = 1.0 / denom                               # decode rate needed to beat the k-th goodput
        b = bs.solve(need if top is not None else 0.0)
        if not b:
            if _feasible(bs.ev(1)):     # feasible but cannot beat the k-th best → pruned
                pruned += 1
            else:                       # infeasible at batch 1 (capacity / SLO)
                rows.append(LayoutRow(lay, org, 0, None))
            continue
        row = LayoutRow(lay, org, b, bs.ev(b))
        if objective == "goodput" and row.result is not None:
            row.goodput = _goodput(row.result)
        if top is not None and row.score(objective) <= floor * (1 + 1e-12):
            pruned += 1                 # goodput floor conversion is exact only up to rounding
            continue
        rows.append(row)
    rows.sort(key=lambda r: -r.score(objective))
    if stats is not None:
        stats.update(layouts=len(cands), solved=len(rows), pruned=pruned,
                     evals=sum(len(c[2].memo) for c in cands))
    return rows[:top] if top is not None else rows


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
    """Latency–throughput Pareto front over batch (LLM: TPOT vs tok/s/card; video / protein: request
    latency vs units/s/card, keys latency_ms / units_s_card)."""
    batches = batches or [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    full = not get_model(base.model).kv_cache
    pts = []
    for b in batches:
        r = evaluate(base.replace("serving.batch", b))
        if r.fits:
            pts.append(((r.latency if full else r.tpot) * 1e3, r.per_card, b))
    if full:
        return [{"latency_ms": x, "units_s_card": y, "batch": b} for x, y, b in pareto(pts)]
    return [{"tpot_ms": x, "tok_s_card": y, "batch": b} for x, y, b in pareto(pts)]
