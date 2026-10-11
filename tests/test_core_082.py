"""0.82: exact radix residency for small trees (MTF recency order); restore-wait law (default unchanged)."""
import math
import random

import accel_dse.core.prefixcache as PC
import accel_dse.core.pdqueue as Q
from accel_dse.core.pdsim import RadixLRU, _tree_paths


def _tail(levels, C):
    g = PC.tree_groups(levels)
    return PC.tree_depth_tail(g, PC.tree_resident(levels, C, g))


def test_exact_two_children_room_for_one():
    # 64-token root + two 4096-token children, room for one: the resident child is the last one used → Σ p²
    z = PC.zipf_groups(2, 0.8)
    want = sum(q * q for _, q in z)
    t = _tail([(64, 1, 0.0), (4096, 2, 0.8)], 4239)
    assert t[0] == 1.0 and abs(t[1] - want) < 1e-12
    # uniform 10 children of 64 after a 128 root, room for one child: 1/10
    assert abs(_tail([(128, 1, 0.0), (64, 10, 0.0)], 210)[1] - 0.1) < 1e-12


def test_exact_never_worse_than_che_vs_radix_lru():
    rng = random.Random(82)
    ex, ch = [], []
    for i in range(14):
        levels = [(rng.choice([16, 64, 512, 4096]), rng.choice([1, 2, 3, 10]), rng.choice([0.0, 0.8, 1.2])) for _ in range(2)]
        if levels[0][0] == levels[1][0]:
            continue
        tot = levels[0][0] * levels[0][1] + levels[1][0] * levels[0][1] * levels[1][1]
        C = rng.uniform(0.05, 0.9) * tot
        c = RadixLRU(C, [L for L, _, _ in levels])
        P = _tree_paths(random.Random(i), levels, 30000)
        for p in P[:8000]:
            c.access(p)
        c.reset_stats()
        for p in P[8000:]:
            c.access(p)
        T = sum(c.depth_n)
        sim = [sum(c.depth_n[k:]) / T for k in (1, 2)]
        for flag, out in ((True, ex), (False, ch)):
            PC.EXACT_TREE = flag
            try:
                a = _tail(levels, C)
            finally:
                PC.EXACT_TREE = True
            out.append(max(abs(x - y) for x, y in zip(a, sim)))
    assert all(e <= c_ + 0.01 for e, c_ in zip(ex, ch)), list(zip(ex, ch))
    assert sum(ex) < sum(ch)


def test_exact_budget_falls_back_to_che():
    lv = [(128, 2, 0.0), (4096, 1, 1.2), (16, 30, 0.8)]     # many small leaves fit → over the budget → Che
    assert PC._tree_exact(lv, 5205) is None
    t = _tail(lv, 5205)
    assert all(0 <= x <= 1 for x in t)


def test_victim_wait_default_and_resid_limits():
    assert Q.VICTIM_WAIT == "exp"
    from types import SimpleNamespace
    # fixed lengths S = 1000, o = 200; K = 3 footprints at the crossing → remaining Ψ = 3·1200 − K split over 3
    ctx = {"len": SimpleNamespace(points=[(1.0, 1000, 200)]), "_victim_wait": "resid"}
    K = 3300.0
    wl = Q._victim_wait_law(ctx, {3: 1.0}, 0.0, K, {"ES": 1000, "Eg": 100}, 0.01, 1.0)
    psi_k = (3 * 1200 - K) / 3                                   # E[R | Σ F = K] = Ψ/k tokens
    # min of the 2 others, each ~ U(0, 2·Ψ/k) → mean 2Ψ/k / 3 tokens × 10 ms
    assert abs(wl["mean"] - 2 * psi_k / 3 * 0.01) < 0.02 * wl["mean"]
    assert wl["q"](1.0) == 0.0 and wl["q"](100.0) > wl["q"](10.0) > 0
    e = Q._victim_wait_law({**ctx, "_victim_wait": "exp"}, {3: 1.0}, 0.0, K, {}, 0.01, 2.0)
    assert e["mean"] == 0.5 and e["q"](math.e) == 0.5
