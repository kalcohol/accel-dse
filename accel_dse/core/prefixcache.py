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
Not modelled here: block granularity (partial-prefix radix matching: the 0.58 tree model below), cache warm-up, eviction cost, the colocated
replica's transient prefill KV, the queueing model's larger batch caps shrinking the free memory.
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace

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


def per_card_bytes(model, layout, tokens: int, ranges: list | None = None) -> list[float]:
    """Bytes of one cached prefix of ``tokens`` tokens on one card of each pipeline stage (KV + idx + state).
    ``ranges`` = the evaluation's stage layer ranges (0.62: the PP split is cost-balanced, not equal counts)."""
    out = []
    sts = plan_stages(model.n_layers, layout.pp) if ranges is None else \
        [SimpleNamespace(first=a, last=b, has_embed=i == 0, has_head=i == len(ranges) - 1) for i, (a, b) in enumerate(ranges)]
    for st in sts:
        s = stage_storage(model, st.first, st.last, st.has_embed, st.has_head, layout.shard, tokens, 0)
        out.append(s.kv_per_seq + s.idx_per_seq + s.state_per_seq)
    return out


def capacity(model, result, tokens: int, cache_GB: float | None) -> dict:
    """Prefixes one replica can hold (``result`` = that pool's evaluation at its operating batch)."""
    lay = result.scenario.layout
    per = per_card_bytes(model, lay, tokens, [s.layers for s in result.stages])
    cps = lay.cards // lay.pp                     # cards per stage
    foot = sum(b * cps / lay.dp for b in per)     # one prefix, all cards of one DP group
    free = [max(0.0, s.mem.dram_cap - s.mem.dram_need) for s in result.stages]
    free_tot = sum(f * cps for f in free)
    if cache_GB is not None:
        K = math.floor(cache_GB * 1e9 / foot) if foot > 0 else 0
        Kf = cache_GB * 1e9 / foot if foot > 0 else 0.0
        src, cap = "pd.prefix_cache_GB", cache_GB * 1e9
    else:
        k_st = [math.floor(f / b) if b > 0 else math.inf for f, b in zip(free, per)]
        kf_st = [f / b if b > 0 else math.inf for f, b in zip(free, per)]
        K = min(k_st) * lay.dp if k_st else 0
        Kf = min(kf_st) * lay.dp if kf_st else 0.0
        if not math.isfinite(K):
            K, Kf = 0, 0.0
        src, cap = "derived_free_dram", free_tot
    return {"K": int(K), "K_float": Kf, "capacity_GB": cap / 1e9, "footprint_MB": foot / 1e6,
            "free_GB": free_tot / 1e9, "source": src, "fits": result.fits}


def pool_hits(groups, K: int, replicas: int, affinity: bool) -> list[float]:
    return resident(groups, K * replicas if affinity else K)


def hit_of(groups, h) -> float:
    return sum(c * q * x for (c, q), x in zip(groups, h))


def both_hit(groups, h1, h2) -> float:
    return sum(c * q * x * y for (c, q), x, y in zip(groups, h1, h2))


# --------------------------------------------------------------------------- 0.58 radix / partial prefix matching
#
# ``pd.prefix_tree`` = levels root → leaf, each (tokens L_k, branching n_k, Zipf α_k) 「假设」: e.g. system prompt →
# document → conversation history.  A request picks one child per level independently (Zipf over the n_k children of
# its parent, IRM), so its prompt starts with a path of Σ L_k shared tokens.  The cache is one LRU over tree nodes
# measured in tokens (vLLM / SGLang radix cache: blocks of a shared prefix are stored once).  A lookup matches the
# deepest resident node on the path and saves the tokens of levels 1 … k (a partial hit).  Touching a path refreshes
# every node on it, so a node is never less recent than its child: LRU keeps the tree property (child resident ⇒
# parent resident).  Che per node: node popularity p = Π of the per-level Zipf masses on its path, residency
# h = 1 − e^{−p·T}, and T solves Σ_nodes L_k·h = C (tokens; the capacity is shared by all levels).  Then
# P(match depth ≥ k) = H_k = Σ_{level-k nodes} p·h, exactly (by the tree property) given Che.

TREE_RATIO = 1.01
TREE_SLACK = "mean"        # capacity slack of the boundary node (0.82: "size"-biased tested, worse — kept 0.81)


def _merge_groups(groups) -> list[tuple[float, float]]:
    """Merge (count, p) groups into geometric p-bins (ratio TREE_RATIO; mass preserved)."""
    if len(groups) <= EXACT_N:
        return list(groups)
    lr = math.log(TREE_RATIO)
    bins: dict[int, list[float]] = {}
    for c, q in groups:
        if q <= 0 or c <= 0:
            continue
        b = bins.setdefault(int(math.floor(math.log(q) / lr)), [0.0, 0.0])
        b[0] += c
        b[1] += c * q
    return [(c, m / c) for c, m in bins.values()]


def tree_groups(levels) -> list[list[tuple[float, float]]]:
    """Per level k: [(node count, p_each)] with Σ count·p = 1 (path popularity = product of the level Zipfs)."""
    out, prev = [], [(1.0, 1.0)]
    for _, n, a in levels:
        z = zipf_groups(int(n), float(a))
        prev = _merge_groups([(c1 * c2, q1 * q2) for c1, q1 in prev for c2, q2 in z])
        out.append(prev)
    return out


def tree_resident(levels, C_tokens: float, groups=None) -> list[list[float]]:
    """Per level, per group: Che residency under one token-capacity LRU shared by all levels."""
    gs = groups or tree_groups(levels)
    if C_tokens <= 0:
        return [[0.0] * len(g) for g in gs]
    # 0.61.2: a node longer than the whole cache can never be resident, and neither can anything below it (a radix
    # child needs its parent).  Che over all levels gave such nodes h > 0 and charged their tokens against the
    # capacity — e.g. levels [100×10, 5000×10] in 3000 tokens: H_1 = 0.10 instead of 1.
    # 0.63 (external review): the test is the cumulative path Σ_{j≤k} L_j, not the node alone — a level-k node is only
    # reachable with all its ancestors resident (levels [60, 60] in 100 tokens: H_2 was 83 %, LRU gives 0)
    m, acc = len(levels), 0.0
    for k, (L, _, _) in enumerate(levels):
        acc += L
        if acc > C_tokens:
            m = k
            break
    lv, gs_fit = levels[:m], gs[:m]
    zero = [[0.0] * len(g) for g in gs[m:]]
    if m and EXACT_TREE and _n_paths(lv) <= EXACT_PATHS and all(len(g) == n for g, n in zip(gs_fit, _level_nodes(lv))):
        ex = _tree_exact_memo(tuple((int(L), int(n), float(a)) for L, n, a in lv), float(C_tokens))
        if ex is not None:                         # None: over the enumeration budget → Che below
            return [list(x) for x in ex] + zero
    tot = sum(L * c for (L, _, _), g in zip(lv, gs_fit) for c, _ in g)
    if C_tokens >= tot:
        return [[1.0] * len(g) for g in gs_fit] + zero
    # 0.112: a leading chain of single-node levels (n = 1, e.g. one shared system prompt) is touched by every request,
    # so it is always the most recent and (it fits: cumulative check above) always resident under LRU.  Che gave it
    # h = 1 − e^{−T} < 1, charged only part of its tokens and broke monotonicity in C (H_1 1.0 → 0.9999 as C grew,
    # when the exact path switched to Che).  Pin it and run Che on the rest of the capacity.
    j = 0
    while j < m and len(gs_fit[j]) == 1 and gs_fit[j][0][0] == 1.0 and gs_fit[j][0][1] == 1.0:
        j += 1
    if 0 < j < m:
        pin = sum(L for L, _, _ in lv[:j])
        rest = _tree_che(lv[j:], gs_fit[j:], C_tokens - pin)
        return [[1.0] for _ in range(j)] + rest + zero
    return _tree_che(lv, gs_fit, C_tokens) + zero


def _tree_che(lv, gs_fit, C_tokens: float) -> list[list[float]]:
    if C_tokens <= 0:
        return [[0.0] * len(g) for g in gs_fit]

    def occ(T):
        return sum(L * c * -math.expm1(-q * T) for (L, _, _), g in zip(lv, gs_fit) for c, q in g)

    def solve(C):
        lo, hi = 0.0, 1.0
        while occ(hi) < C and hi < 1e300:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if occ(mid) < C:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-12 * hi:
                break
        return 0.5 * (lo + hi)
    # 0.81: the LRU holds whole nodes (evicts until used ≤ C), so part of C is always slack.  One node length: exactly
    # ⌊C/L⌋ nodes.  Mixed lengths: slack ≈ ½ the mean evicted node length (eviction mix ∝ node miss rate p·(1 − h),
    # one Che pass at C).  Che on the raw C overstated hits when the cache holds a few nodes (128-token nodes in 308
    # tokens: H 0.24 vs the RadixLRU's 0.20)
    lens = {L for L, _, _ in lv}
    if len(lens) == 1:
        L0 = lens.pop()
        C_eff = math.floor(C_tokens / L0) * L0
    else:
        T0 = solve(C_tokens)
        w0 = [(L, c * q * math.exp(-q * T0)) for (L, _, _), g in zip(lv, gs_fit) for c, q in g]
        if TREE_SLACK == "size":
            # 0.82: the node straddling the capacity boundary is size-biased (inspection paradox) — its type ∝ L ×
            # boundary density p·e^{−pT}; slack ~ U(0, L) of it → E = ½·Σ w L² / Σ w L
            num = sum(w * L * L for L, w in w0)
            den = sum(w * L for L, w in w0)
        else:
            num = sum(w * L for L, w in w0)
            den = sum(w for L, w in w0)
        C_eff = C_tokens - 0.5 * num / den if den > 0 else C_tokens
    if C_eff <= 0:
        return [[0.0] * len(g) for g in gs_fit]
    T = solve(C_eff)
    return [[-math.expm1(-q * T) for _, q in g] for g in gs_fit]


EXACT_TREE = True
EXACT_PATHS = 4096           # stored paths considered for exact enumeration …
EXACT_BUDGET = 100_000      # … if the fitting-subset DP stays within this many states (0.83; 0.82: 50k order steps)


def _n_paths(levels) -> int:
    n = 1
    for _, k, _ in levels:
        n *= int(k)
    return n


def _level_nodes(levels) -> list[int]:
    out, n = [], 1
    for _, k, _ in levels:
        n *= int(k)
        out.append(n)
    return out


@functools.lru_cache(maxsize=4096)
def _tree_exact_memo(levels: tuple, C: float):
    r = _tree_exact(levels, C)
    return None if r is None else tuple(tuple(x) for x in r)


def _tree_exact(levels, C: float):
    """0.82 / 0.83: exact per-node residency for a small radix tree (``levels`` already cut to the levels that fit).
    IRM + LRU: the stationary recency order of the paths is sampling without replacement ∝ path probability
    (move-to-front); a path touch refreshes root → leaf, so in node recency order each path contributes its
    not-yet-listed nodes root first, and the token LRU keeps the longest recency prefix that fits C.  The resident set
    depends only on the set A of the most recent paths that fit completely and on the next path b (its new nodes
    root-first until one does not fit), so (0.83) a DP over fitting subsets replaces 0.82's enumeration of orders:
    f(A) = P(A is the set of the |A| most recent paths) = Σ_{a∈A} f(A − a)·p_a / (1 − P(A − a)).  None when the
    number of fitting subsets exceeds EXACT_BUDGET (→ Che)."""
    m = len(levels)
    zs = [zipf_groups(int(n), float(a)) for _, n, a in levels]
    ns = [int(n) for _, n, _ in levels]
    Ls = [int(L) for L, _, _ in levels]
    paths = []

    def build(k, idx, pr):
        if k == m:
            paths.append((tuple(idx), pr))
            return
        for j_, (_, q) in enumerate(zs[k]):
            build(k + 1, idx + [j_], pr * q)
    build(0, [], 1.0)
    if len(paths) > EXACT_BUDGET:
        return None
    nodes = [[path[:k + 1] for k in range(m)] for path, _ in paths]
    P = [pr for _, pr in paths]
    res: dict = {}
    # layer by subset size: state bitmask → (f, P(A), used, frozenset nodes)
    layer = {0: (1.0, 0.0, 0, frozenset())}
    n_states = 0
    while layer:
        nxt: dict = {}
        for mask, (f, pa, used, nd) in layer.items():
            rem = 1.0 - pa
            if f <= 0:
                continue
            for b in range(len(paths)):
                if mask >> b & 1:
                    continue
                pb = f * P[b] / rem if rem > 0 else 0.0
                u, add, full = used, [], False
                for k, x in enumerate(nodes[b]):
                    if x in nd:
                        continue
                    if u + Ls[k] > C:
                        full = True
                        break
                    u += Ls[k]
                    add.append(x)
                if full:                           # b straddles: final resident set
                    for x in nd:
                        res[x] = res.get(x, 0.0) + pb
                    for x in add:
                        res[x] = res.get(x, 0.0) + pb
                    continue
                m2 = mask | (1 << b)
                if m2 in nxt:
                    g = nxt[m2]
                    nxt[m2] = (g[0] + pb, g[1], g[2], g[3])
                else:
                    nxt[m2] = (pb, pa + P[b], u, nd | frozenset(add))
                    n_states += 1
                    if n_states > EXACT_BUDGET:
                        return None
        # every path fitting (A = all) is handled by the caller (C ≥ total → all resident)
        for mask, (f, pa, used, nd) in nxt.items():
            if mask == (1 << len(paths)) - 1:
                for x in nd:
                    res[x] = res.get(x, 0.0) + f
        layer = {k_: v for k_, v in nxt.items() if k_ != (1 << len(paths)) - 1}
    out = []
    for k in range(m):
        cnt = 1
        for n_ in ns[:k + 1]:
            cnt *= n_
        row = [0.0] * cnt
        for x, v in res.items():
            if len(x) == k + 1:
                ii = 0
                for j_, n_ in zip(x, ns[:k + 1]):
                    ii = ii * n_ + j_
                row[ii] = min(1.0, v)
        out.append(row)
    return out


def tree_depth_tail(groups, h) -> list[float]:
    """[H_1 … H_m]: P(match depth ≥ k)."""
    out, prev = [], 1.0
    for g, hk in zip(groups, h):
        prev = min(prev, sum(c * q * x for (c, q), x in zip(g, hk)))    # a match at depth k implies depth k − 1
        out.append(prev)
    return out


def tree_both_tail(groups, h1, h2) -> list[float]:
    """P(both caches hold the level-k node of the request's path) per level (independent caches given the node)."""
    return [sum(c * q * x * y for (c, q), x, y in zip(g, a, b)) for g, a, b in zip(groups, h1, h2)]


def tree_depth_law(tail) -> list[float]:
    """[P(D = 0), …, P(D = m)] from the tails H_k (H_0 = 1)."""
    t = [1.0] + list(tail) + [0.0]
    return [max(0.0, t[k] - t[k + 1]) for k in range(len(t) - 1)]


def tree_cum_tokens(levels) -> list[int]:
    """Matched tokens at depth 0 … m."""
    out, acc = [0], 0
    for L, _, _ in levels:
        acc += int(L)
        out.append(acc)
    return out
