"""0.83: saturation-preemption growth law (Gamma, lifetime CV²), restore-wait blend, subset-DP exact radix."""
import itertools
import math
from types import SimpleNamespace

import accel_dse.core.pdqueue as Q
import accel_dse.core.prefixcache as PC


def test_gamma_tail_closed_forms():
    assert abs(Q._gamma_q(1, 2.0) - math.exp(-2)) < 1e-14
    assert abs(Q._gamma_q(3, 1.0) - math.exp(-1) * 2.5) < 1e-13
    assert abs(Q._gamma_q(2, 5.0) - math.exp(-5) * 6) < 1e-13
    assert abs(Q._gamma_q(1000, 1000) - 0.5) < 0.01                  # Wilson–Hilferty branch, median ≈ mean


def test_sat_evict_limits_and_monotone():
    pts = [(1.0, 4096, 512)]
    old = (512 / 4096) * (1 - math.exp(-8)) * math.exp(-1)              # ≤ 0.82 Exp term (fixed lengths)
    assert abs(Q._sat_evict_gamma(pts, 512, 1.0, n=400) - old) < 1e-3 * old
    assert Q._sat_evict_gamma(pts, 512, 0.0) == 0.0                     # fixed lifetimes: the walk only oscillates
    v = [Q._sat_evict_gamma(pts, 512, c) for c in (0.01, 0.1, 0.5, 1.0)]
    assert all(a < b for a, b in zip(v, v[1:]))
    x = {"tpot_mean": 1.0, "tpot_p90": 1.0}
    assert Q._life_cv2([(1.0, 100, 50)], x) == 0.0
    assert Q._life_cv2([(0.9, 100, 10), (0.1, 100, 1000)], x) == 1.0    # capped at Poisson


def _brute(levels, C):
    zs = [PC.zipf_groups(int(n), float(a)) for _, n, a in levels]
    paths = [(idx, math.prod(zs[k][j][1] for k, j in enumerate(idx))) for idx in itertools.product(*[range(int(n)) for _, n, _ in levels])]
    Ls = [L for L, _, _ in levels]
    res = {}
    for perm in itertools.permutations(range(len(paths))):
        pr, rem = 1.0, 1.0
        for i in perm:
            pr *= paths[i][1] / rem
            rem -= paths[i][1]
        nd, u = [], 0
        stop = False
        for i in perm:
            for k in range(len(levels)):
                x = paths[i][0][:k + 1]
                if x in nd:
                    continue
                if u + Ls[k] > C:
                    stop = True
                    break
                u += Ls[k]
                nd.append(x)
            if stop:
                break
        for x in nd:
            res[x] = res.get(x, 0.0) + pr
    return res


def test_subset_dp_equals_order_enumeration():
    for levels, C in (([(64, 1, 0.0), (4096, 2, 0.8)], 4239), ([(100, 2, 0.8), (60, 3, 0.0)], 330),
                      ([(50, 3, 1.2), (20, 2, 0.5)], 170), ([(30, 5, 0.9)], 100)):
        ex = PC._tree_exact(levels, C)
        br = _brute(levels, C)
        for k, row in enumerate(ex):
            ns = [int(n) for _, n, _ in levels[:k + 1]]
            for i, v in enumerate(row):
                idx, r = [], i
                for n in reversed(ns):
                    idx.append(r % n)
                    r //= n
                assert abs(v - br.get(tuple(reversed(idx)), 0.0)) < 1e-12, (levels, C, k, i)


def test_blend_limits():
    ctx = {"len": SimpleNamespace(points=[(1.0, 1000, 200)])}
    K, kvi = 3300.0, {"ES": 1000, "Eg": 100}
    e = Q._victim_wait_law({**ctx, "_victim_wait": "exp"}, {3: 1.0}, 0.0, K, kvi, 0.01, 2.0)
    b1 = Q._victim_wait_law({**ctx, "_victim_wait": "blend"}, {3: 1.0}, 0.0, K, kvi, 0.01, 2.0, 1.0)
    assert b1["model"] == "exp" and b1["q"](math.e) == e["q"](math.e)
    r = Q._victim_wait_law({**ctx, "_victim_wait": "resid"}, {3: 1.0}, 0.0, K, kvi, 0.01, 2.0)
    b0 = Q._victim_wait_law({**ctx, "_victim_wait": "blend"}, {3: 1.0}, 0.0, K, kvi, 0.01, 2.0, 0.0)
    assert abs(b0["mean"] - r["mean"]) < 1e-12 and abs(b0["q"](20.0) - r["q"](20.0)) < 1e-6
    bh = Q._victim_wait_law({**ctx, "_victim_wait": "blend"}, {3: 1.0}, 0.0, K, kvi, 0.01, 2.0, 0.5)
    assert abs(bh["mean"] - 0.5 * (r["mean"] + e["mean"])) < 1e-12
