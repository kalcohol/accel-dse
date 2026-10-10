"""L2 — parallel layout and pipeline-stage plan.

A ``Layout`` places one model replica on ``pp·tp·dp`` cards:
  pp  pipeline stages (integer, contiguous layer ranges)
  tp  attention tensor parallel; dp attention data parallel inside a stage
  ep·etp = tp·dp  expert parallel × expert tensor parallel for MoE layers
  sp  sequence parallel (Ulysses) — non-autoregressive models only (video DiT, protein encoders; structure
      models: DAP, the pair / MSA grids split along a residue axis);
      their dp splits the forward batch (incl. the CFG cond/uncond pair) inside the replica
Cards = pp·tp·dp·sp.  Single card = Layout() (all 1).  There is no second code path.

Stage plan: layers split contiguously, remainder to the *first* stages
(61 / 8 → [8,8,8,8,8,7,7,7]); embedding on stage 0; final norm + LM head +
MTP modules on the last stage.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

from .ir import Shard


@dataclass(frozen=True)
class Layout:
    pp: int = 1
    tp: int = 1
    dp: int = 1
    ep: int = 1
    etp: int = 1
    sp: int = 1

    def __post_init__(self):
        for k in ("pp", "tp", "dp", "ep", "etp", "sp"):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"layout.{k} must be an integer ≥ 1 (got {v!r})")

    @property
    def cards(self) -> int:
        return self.pp * self.tp * self.dp * self.sp

    @property
    def shard(self) -> Shard:
        return Shard(self.tp, self.dp, self.ep, self.etp, self.sp)

    def valid_for(self, moe: bool, full: bool = False, pair: bool = False) -> bool:
        """MoE LLM: ep·etp = tp·dp; dense LLM: dp = ep = etp = 1; both sp = 1.
        Non-autoregressive (``full``, dense): ep = etp = 1, any dp / sp.
        Structure models (``pair``): tp = 1; sp = DAP degree (0.45: pair / MSA grids split over sp ranks), any dp / pp."""
        if full:
            return self.ep == 1 and self.etp == 1 and (not pair or self.tp == 1)
        if self.sp != 1:
            return False
        if moe:
            return self.ep * self.etp == self.tp * self.dp
        return self.ep == 1 and self.etp == 1 and self.dp == 1

    @property
    def label(self) -> str:
        s = f"PP{self.pp}·TP{self.tp}"
        if self.dp > 1:
            s += f"·DP{self.dp}"
        if self.ep > 1:
            s += f"·EP{self.ep}"
        if self.etp > 1:
            s += f"·ETP{self.etp}"
        if self.sp > 1:
            s += f"·SP{self.sp}"
        return s


@dataclass(frozen=True)
class Stage:
    index: int
    first: int          # first layer (inclusive)
    last: int           # last layer (exclusive)
    has_embed: bool
    has_head: bool

    @property
    def n_layers(self) -> int:
        return self.last - self.first


def stage_layer_counts(n_layers: int, pp: int) -> list[int]:
    if pp < 1 or pp > n_layers:
        raise ValueError(f"pp must be in [1, {n_layers}] (got {pp})")
    base, rem = divmod(n_layers, pp)
    return [base + (1 if i < rem else 0) for i in range(pp)]


def plan_stages(n_layers: int, pp: int) -> list[Stage]:
    out, start = [], 0
    counts = stage_layer_counts(n_layers, pp)
    for i, c in enumerate(counts):
        out.append(Stage(i, start, start + c, i == 0, i == pp - 1))
        start += c
    return out


PP_SPLITS = ("cost", "layers")
EXEC_OVERLAPS = ("stage", "class", "kernel", "serial", "tbo")   # 0.66 (+ tbo 0.67): see schedule.StageTime.total


def plan_stages_balanced(costs: list, pp: int, first: tuple, last: tuple, p2p: tuple,
                         mem: list | None = None, mem_first: float = 0.0, mem_last: float = 0.0,
                         mem_cap: float = math.inf, alternatives: bool = False):
    """0.62: contiguous PP split minimising the slowest stage (the pipeline tick) instead of equal layer counts.

    ``costs[i]`` = (t_array, t_vector, t_dram, t_link, t_sync) of layer i; ``first`` / ``last`` = the same for the
    embedding / io-pre ops (stage 0) and the head / io-post / MTP ops (last stage); ``p2p`` = the activation send
    every non-last stage adds.  A stage's time follows the stage rule max(Σarray, Σvector, Σdram, Σlink) + Σsync
    (DRAM time = touched bytes at full streaming, i.e. ignoring SRAM / SLC residency -- the proxy).  ``mem`` (per-layer
    stored bytes, ``mem_first`` / ``mem_last`` for the io stages) caps every stage at ``mem_cap`` so balancing never
    pushes a stage over the capacity the equal-count split already needs.  Exact min-max over contiguous splits
    (binary search on the tick + greedy packing), then boundaries evened out inside the feasible window; ties keep the
    equal-layer-count split.  ``alternatives``: return a tuple of candidate splits (evened, front-greedy) -- both
    proxy-optimal -- for the caller to evaluate (the proxy ignores residency / activations); the equal-count split
    is returned as a plain list when nothing beats it."""
    L = len(costs)
    if pp < 1 or pp > L:
        raise ValueError(f"pp must be in [1, {L}] (got {pp})")
    lc = plan_stages(L, pp)
    if pp == 1:
        return lc
    P = [(0.0,) * 5]
    for c in costs:
        P.append(tuple(a + b for a, b in zip(P[-1], c)))
    M = [0.0]
    for x in (mem or [0.0] * L):
        M.append(M[-1] + x)

    def cost(a: int, b: int) -> float:
        v = [P[b][k] - P[a][k] for k in range(5)]
        for extra, on in ((first, a == 0), (last, b == L), (p2p, b < L)):
            if on:
                for k in range(5):
                    v[k] += extra[k]
        return max(v[0], v[1], v[2], v[3]) + v[4]

    def mem_of(a: int, b: int) -> float:
        return M[b] - M[a] + (mem_first if a == 0 else 0.0) + (mem_last if b == L else 0.0)

    def split(T: float) -> list[int] | None:
        """Greedy: each stage takes the most layers keeping cost ≤ T (and ≥ 1 layer left per later stage)."""
        bounds, a = [0], 0
        for s in range(pp - 1):
            best = None
            for b in range(a + 1, L - (pp - 1 - s) + 1):
                if cost(a, b) > T or mem_of(a, b) > mem_cap:
                    break
                best = b
            if best is None:
                return None
            bounds.append(best)
            a = best
        if cost(a, L) > T or mem_of(a, L) > mem_cap:
            return None
        return bounds + [L]

    lc_b = [st.first for st in lc] + [L]
    t_lc = max(cost(lc_b[i], lc_b[i + 1]) for i in range(pp))
    lo, hi = 0.0, t_lc
    best = None
    for _ in range(60):
        mid = (lo + hi) / 2
        b = split(mid)
        if b is None:
            lo = mid
        else:
            hi, best = mid, b
        if hi - lo <= 1e-12 * max(hi, 1e-30):
            break
    if best is None:
        return lc
    t_best = max(cost(best[i], best[i + 1]) for i in range(pp))
    if t_best >= t_lc * (1 - 1e-9):
        return lc                       # equal counts already balanced (homogeneous stack): keep them
    even = _even_out(best, hi, L, pp, costs, first, last, cost, mem_of, mem_cap)
    mk = lambda bd: [Stage(i, bd[i], bd[i + 1], i == 0, i == pp - 1) for i in range(pp)]
    return (mk(even), mk(best)) if alternatives and even != best else mk(even) if not alternatives else (mk(best),)


def _even_out(greedy: list[int], T: float, L: int, pp: int, costs, first, last, cost, mem_of, mem_cap) -> list[int]:
    """Among the splits whose slowest stage is ≤ T, pick boundaries near an even share of the remaining work instead of
    the front-heavy greedy ones (which leave the last stage nearly idle).  back[k] = the earliest layer stages k … pp−1
    can start from and still fit under T (backward greedy); each forward boundary stays in the feasible window."""
    one = lambda c: max(c[0], c[1], c[2], c[3]) + c[4]
    W = [0.0]
    for c in costs:
        W.append(W[-1] + one(c))
    w_first, w_last = one(first), one(last)
    back = [0] * (pp + 1)
    back[pp] = L
    for k in range(pp - 1, 0, -1):
        e, x = back[k + 1], None
        for a in range(e - 1, k - 1, -1):
            if cost(a, e) > T or mem_of(a, e) > mem_cap:
                break
            x = a
        if x is None:
            return greedy
        back[k] = x
    out, a = [0], 0
    for s in range(pp - 1):
        hi = None
        for b in range(a + 1, L - (pp - 1 - s) + 1):
            if cost(a, b) > T or mem_of(a, b) > mem_cap:
                break
            hi = b
        lo = max(a + 1, back[s + 1])
        if hi is None or lo > hi:
            return greedy
        share = (W[L] - W[a] + w_last + (w_first if a == 0 else 0.0)) / (pp - s)
        base = W[a] - (w_first if a == 0 else 0.0)
        b = min(range(lo, hi + 1), key=lambda x: (abs(W[x] - base - share), x))
        out.append(b)
        a = b
    if cost(a, L) > T or mem_of(a, L) > mem_cap:
        return greedy
    return out + [L]


MAX_REPLICA_CARDS = 8192   # 0.60: cards per replica accepted by API / UI / CLI (was 64); search stays interactive (see MODEL §19.1)
MAX_POOL_CARDS = 65536     # 0.60: PD pool cards (many replicas)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_layouts(cards: int, n_layers: int, moe: bool, *, max_tp: int | None = None,
                      full: bool = False, pair: bool = False) -> list[Layout]:
    """All layouts that use exactly ``cards`` cards for one replica (``full``: PP·TP·DP·SP of a dense
    non-autoregressive model; ``pair``: structure models, PP·DP·SP with SP = DAP, no TP)."""
    out = []
    if full:
        for pp in _divisors(cards):
            if pp > n_layers:
                continue
            if pair:
                for dp in _divisors(cards // pp):
                    out.append(Layout(pp, 1, dp, 1, 1, cards // pp // dp))
                continue
            for tp in _divisors(cards // pp):
                if max_tp and tp > max_tp:
                    continue
                for dp in _divisors(cards // pp // tp):
                    out.append(Layout(pp, tp, dp, 1, 1, cards // pp // tp // dp))
        return out
    for pp in _divisors(cards):
        if pp > n_layers:
            continue
        rest = cards // pp
        for tp in _divisors(rest):
            if max_tp and tp > max_tp:
                continue
            dp = rest // tp
            if not moe:
                if dp == 1:
                    out.append(Layout(pp, tp, 1, 1, 1))
                continue
            for ep in _divisors(rest):
                out.append(Layout(pp, tp, dp, ep, rest // ep))
    return out
