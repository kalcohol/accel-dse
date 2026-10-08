"""Prefix-cache capacity + LRU eviction (0.53, opt-in ``pd.prefix_len`` > 0; all 「假设」).

Working set: N distinct shared prefixes (system prompts / few-shot templates / multi-turn histories) of ``prefix_len``
tokens, requested with Zipf(α) popularity p_i ∝ i^−α (α = 0 uniform), independently per request (IRM).  Every request
carries one of them; a hit skips the prefix's prefill (and, if the decode pool holds it too, its KV hand-off).

Cache: LRU over whole prefixes, K = how many prefixes fit in the capacity.  Hit rate by the Che approximation
(Che, Tung & Wang 2002; Fricker, Robert & Roberts 2012): the characteristic time T solves Σ_i (1 − e^{−p_i·T}) = K,
item i is resident with probability h_i = 1 − e^{−p_i·T}, and the request hit rate is H = Σ_i p_i·h_i.  K ≥ N → H = 1
(steady state; cold misses ignored); K = 0 → H = 0.  N ≤ 4096 is exact; larger N keeps ranks ≤ 1024 exact and
merges the tail into geometric rank bins (ratio 1.01, midpoint-integral Zipf mass) — error ≪ 1 %.

Capacity per replica (prefixes): ``pd.prefix_cache_GB`` / footprint, or (None) the DRAM left on each card after
weights + the active batch's KV + runtime reserve (memplan), divided by one prefix's per-card bytes, min over pipeline
stages, × attention-DP groups.  Footprint of one prefix = its KV + indexer keys + recurrent-state snapshot at
``prefix_len`` tokens, as stored on the cards (TP shards / replicas counted as memplan does).
Routing: random → each replica's cache sees the whole population (H at per-replica K); ``pd.prefix_affinity`` →
prefix-aware routing partitions the prefixes (H at the aggregate K · replicas).
Not modelled: partial-prefix (radix-tree) matches, block granularity, cache warm-up, eviction cost, the colocated
replica's transient prefill KV, the queueing model's larger batch caps shrinking the free memory.
"""

from __future__ import annotations

import math

from .memplan import stage_storage
from .parallel import plan_stages

EXACT_N = 4096
HEAD = 1024
RATIO = 1.01


def zipf_groups(n: int, alpha: float) -> list[tuple[float, float]]:
    """[(count, p_each)] over ranks 1..n with Σ count·p = 1."""
    if n <= EXACT_N:
        raw = [(1.0, i ** -alpha) for i in range(1, n + 1)]
    else:
        raw = [(1.0, i ** -alpha) for i in range(1, HEAD + 1)]
        a = HEAD + 1
        while a <= n:
            b = min(n + 1, max(a + 1, int(a * RATIO)))
            c = b - a
            lo, hi = a - 0.5, b - 0.5
            mass = math.log(hi / lo) if alpha == 1 else (hi ** (1 - alpha) - lo ** (1 - alpha)) / (1 - alpha)
            raw.append((float(c), mass / c))
            a = b
    tot = sum(c * q for c, q in raw)
    return [(c, q / tot) for c, q in raw]


def che_T(groups, K: float) -> float:
    """Characteristic time T with Σ count·(1 − e^{−p·T}) = K (requires 0 < K < N)."""
    def occ(T):
        return sum(c * -math.expm1(-q * T) for c, q in groups)
    lo, hi = 0.0, 1.0
    while occ(hi) < K:
        hi *= 2.0
        if hi > 1e300:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if occ(mid) < K:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * hi:
            break
    return 0.5 * (lo + hi)


def resident(groups, K: float) -> list[float]:
    """Per-group residency probability h_j under LRU with K slots."""
    n = sum(c for c, _ in groups)
    if K <= 0:
        return [0.0] * len(groups)
    if K >= n:
        return [1.0] * len(groups)
    T = che_T(groups, K)
    return [-math.expm1(-q * T) for _, q in groups]


def lru_hit(n: int, alpha: float, K: float) -> float:
    g = zipf_groups(n, alpha)
    return sum(c * q * h for (c, q), h in zip(g, resident(g, K)))


def per_card_bytes(model, layout, tokens: int) -> list[float]:
    """Bytes of one cached prefix of ``tokens`` tokens on one card of each pipeline stage (KV + idx + state)."""
    out = []
    for st in plan_stages(model.n_layers, layout.pp):
        s = stage_storage(model, st.first, st.last, st.has_embed, st.has_head, layout.shard, tokens, 0)
        out.append(s.kv_per_seq + s.idx_per_seq + s.state_per_seq)
    return out


def capacity(model, result, tokens: int, cache_GB: float | None) -> dict:
    """Prefixes one replica can hold (``result`` = that pool's evaluation at its operating batch)."""
    lay = result.scenario.layout
    per = per_card_bytes(model, lay, tokens)
    cps = lay.cards // lay.pp                     # cards per stage
    foot = sum(b * cps / lay.dp for b in per)     # one prefix, all cards of one DP group
    free = [max(0.0, s.mem.dram_cap - s.mem.dram_need) for s in result.stages]
    free_tot = sum(f * cps for f in free)
    if cache_GB is not None:
        K = math.floor(cache_GB * 1e9 / foot) if foot > 0 else 0
        src, cap = "pd.prefix_cache_GB", cache_GB * 1e9
    else:
        k_st = [math.floor(f / b) if b > 0 else math.inf for f, b in zip(free, per)]
        K = min(k_st) * lay.dp if k_st else 0
        if not math.isfinite(K):
            K = 0
        src, cap = "derived_free_dram", free_tot
    return {"K": int(K), "capacity_GB": cap / 1e9, "footprint_MB": foot / 1e6, "free_GB": free_tot / 1e9,
            "source": src, "fits": result.fits}


def pool_hits(groups, K: int, replicas: int, affinity: bool) -> list[float]:
    return resident(groups, K * replicas if affinity else K)


def hit_of(groups, h) -> float:
    return sum(c * q * x for (c, q), x in zip(groups, h))


def both_hit(groups, h1, h2) -> float:
    return sum(c * q * x * y for (c, q), x, y in zip(groups, h1, h2))
