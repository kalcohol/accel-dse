"""0.132 EP-skew routing: exact occupancy from a measured expert load (max over ranks of Σ 1−(1−p_e)^T) matches a
seeded Monte-Carlo routing sim; the scalar popular rule is unchanged; opt-in spread="uniform"; validation."""
import math
import random

from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import SkewSpec, hot_rank_probs, moe_routing
from accel_dse.core.scenario import Scenario


def _mc_hot(load, ep, k, T, trials, rng):
    # independent-inclusion routing per expert (p_e) — checks the occupancy algebra, max over ranks of E[distinct]
    p = hot_rank_probs(load, ep, k)
    best = 0.0
    for r in p:
        tot = 0
        for _ in range(trials):
            tot += sum(1 for pe in r if any(rng.random() < pe for _ in range(T)))
        best = max(best, tot / trials)
    return best


def test_hot_rank_probs_sum_to_k():
    load = tuple(float(i % 7 + 1) for i in range(64))
    p = hot_rank_probs(load, 8, 4)
    assert len(p) == 8 and all(len(r) == 8 for r in p)
    assert abs(sum(map(sum, p)) - 4) < 1e-9


def test_exact_occupancy_vs_monte_carlo():
    rng = random.Random(132)
    load = tuple(1.0 / (i + 1) for i in range(64))
    for T in (2, 8, 32):
        mc = _mc_hot(load, 8, 4, T, 400, rng)
        r = moe_routing(64, 4, T, 8, SkewSpec(2.0, hot_rank_probs(load, 8, 4)))
        assert abs(r.hit - math.ceil(mc - 1e-9)) <= 1, (T, r.hit, mc)


def test_scalar_popular_unchanged_and_uniform_opt_in():
    for T in (1, 4, 16, 64):
        a = moe_routing(256, 8, T, 8, 2.0)
        b = moe_routing(256, 8, T, 8, SkewSpec(2.0))
        assert (a.hit, a.pairs_local) == (b.hit, b.pairs_local)
        u = moe_routing(256, 8, T, 8, SkewSpec(2.0, (), "uniform"))
        assert u.hit >= a.hit and u.pairs_local == a.pairs_local
    assert moe_routing(256, 8, 4, 1, SkewSpec(2.0, (), "uniform")).hit == moe_routing(256, 8, 4, 1).hit


def test_hot_rank_uniform_spread_equals_exact():
    # load concentrated evenly on rank 0 (skew s): uniform spread and exact occupancy agree
    E, ep, k, s = 64, 8, 4, 3.0
    load = tuple(s if e < 8 else (8 - s) / 7 for e in range(E))
    for T in (2, 8, 32):
        x = moe_routing(E, k, T, ep, SkewSpec(s, hot_rank_probs(load, ep, k)))
        u = moe_routing(E, k, T, ep, SkewSpec(s, (), "uniform"))
        assert x.hit == u.hit, T


def test_spread_validation_and_default_omitted():
    s = Scenario()
    assert "moe_skew_spread" not in str(s.to_dict()) if hasattr(s, "to_dict") else True
    try:
        s.replace("serving.moe_skew_spread", "hot")
        raise AssertionError
    except ValueError:
        pass
    from accel_dse.core.parallel import Layout
    base = s.replace("model", "deepseek-v4-flash").replace("layout", Layout(tp=8, ep=8)).replace("serving.batch", 16).replace("serving.moe_skew", 2.0)
    pop = evaluate(base)
    uni = evaluate(base.replace("serving.moe_skew_spread", "uniform"))
    assert uni.throughput < pop.throughput
