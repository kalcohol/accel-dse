"""0.133 opt-in serving.moe_rank_max = "expected": per-step expected max over EP ranks of distinct experts and
token-expert pairs (normal order statistic) vs a seeded Monte-Carlo; default unchanged; validation; CLI."""
import random
import statistics as st

from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import SkewSpec, _emax_normal, _rank_max, moe_routing
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario


def _mc(E, k, ep, T, trials, rng):
    nl = E // ep
    md, mp = [], []
    for _ in range(trials):
        ch = [set() for _ in range(ep)]
        pr = [0] * ep
        for _t in range(T):
            for e in rng.sample(range(E), k):
                ch[e // nl].add(e)
                pr[e // nl] += 1
        md.append(max(map(len, ch)))
        mp.append(max(pr))
    return st.mean(md), st.mean(mp)


def test_rank_max_vs_monte_carlo_uniform():
    rng = random.Random(133)
    for E, k, ep, T in ((256, 8, 8, 16), (128, 4, 4, 64), (64, 2, 8, 8), (256, 6, 8, 4)):
        d, p = _rank_max(E, k, T, ep, (), 1.0)
        md, mp = _mc(E, k, ep, T, 300, rng)
        assert abs(d - md) / md < 0.12 and abs(p - mp) / mp < 0.12, (E, k, ep, T, d, md, p, mp)
        assert d > E // ep * (1 - (1 - k / E) ** T) and p > T * k / ep


def test_emax_normal_iid_matches_integration():
    a = _emax_normal(((10.0, 2.0),) * 8)
    b = _emax_normal(((10.0, 2.0),) * 7 + ((10.0, 2.0000001),))
    assert abs(a - b) < 0.02 * a and abs(_emax_normal(((3.0, 0.0), (5.0, 0.0))) - 5.0) < 1e-12


def test_default_off_unchanged_and_on_never_faster():
    for T in (1, 8, 64):
        a = moe_routing(256, 8, T, 8, 1.0)
        b = moe_routing(256, 8, T, 8, SkewSpec(1.0))
        c = moe_routing(256, 8, T, 8, SkewSpec(1.0, (), "popular", "expected"))
        assert (a.hit, a.pairs_local) == (b.hit, b.pairs_local)
        assert c.hit >= a.hit and c.pairs_local >= a.pairs_local
    assert moe_routing(256, 8, 8, 1, SkewSpec(1.0, (), "popular", "expected")) == moe_routing(256, 8, 8, 1)
    s = Scenario().replace("model", "deepseek-v4-flash").replace("layout", Layout(tp=8, ep=8)).replace("serving.batch", 16)
    # 0.141: "expected" is the default
    off, on = evaluate(s.replace("serving.moe_rank_max", "off")), evaluate(s)
    assert s.serving.moe_rank_max == "expected"
    assert on.throughput <= off.throughput * (1 + 1e-12) and on.throughput < off.throughput


def test_validation_and_cli():
    try:
        Scenario().replace("serving.moe_rank_max", "p99")
        raise AssertionError
    except ValueError:
        pass
    from pathlib import Path
    assert "--moe-rank-max" in (Path(__file__).resolve().parents[1] / "accel_dse/cli.py").read_text()
