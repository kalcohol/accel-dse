"""0.79: closed-form TTFT tail exact for M/D/1 (the +36 % p99 vs DES at load 0.9 is DES finite-horizon bias);
search / optimizer layer vs brute force on random small spaces; video / protein extremes; MSA-cap warning."""
import math
import random
from decimal import Decimal, getcontext

from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.queueing import md1
from accel_dse.core.scenario import Scenario, Serving, Workload
from accel_dse.core.search import _feasible, best_batch, brute_force_batch, pareto, search_layouts


def _md1_exact_cdf(lam, D, x):
    """Erlang's exact M/D/1 waiting-time law, 80-digit arithmetic (alternating series)."""
    getcontext().prec = 80
    lam, D, x = Decimal(lam), Decimal(D), Decimal(x)
    s, k = Decimal(0), 0
    while k * D <= x:
        t = lam * (k * D - x)
        s += (t ** k if k else Decimal(1)) / Decimal(math.factorial(k)) * (-t).exp()
        k += 1
    return float((1 - lam * D) * s)


def test_md1_quantiles_exact_at_high_load():
    for lam, D in ((1.1625715334139748, 0.7741459), (0.5, 1.0), (9.5, 0.1)):
        q = md1(lam, D)
        for p in (0.5, 0.9, 0.99):
            assert abs(_md1_exact_cdf(lam, D, q[f"p{round(p * 100)}"]) - p) < 2e-4, (lam, D, p)


def _rs(rng):
    return Scenario(model=rng.choice(["qwen3-8b", "qwen3-0.6b", "llama-3.1-8b", "qwen3-30b-a3b", "gpt-oss-20b"]),
                    chip=CHIPS[rng.choice(list(CHIPS))],
                    mem_id=rng.choice(["lpddr5x_4x64_8533_16g", "hbm3e_8s_12h24g_9200", "lpddr5x_2x64_8533_8g"]),
                    serving=Serving(ctx=rng.choice([512, 2048, 8192, 32768]), tpot_slo_ms=rng.choice([5, 20, 50, 200])))


def test_batch_search_equals_brute_force():
    rng, n = random.Random(11), 0
    while n < 25:
        s = _rs(rng)
        bb = best_batch(s, 32)
        b2, t2 = brute_force_batch(s, 32)
        t1 = bb.result.throughput if bb.batch else -1.0
        assert bb.batch == b2 or abs(t1 - t2) <= 1e-9 * max(t2, 1), (s.model, bb.batch, b2)
        n += 1


def test_layout_topk_equals_full_and_never_below_feasible():
    rng = random.Random(12)
    for _ in range(4):
        s, cards = _rs(rng), rng.choice([2, 4])
        full = search_layouts(s, cards)
        top = search_layouts(s, cards, top=2)
        assert [round(r.per_card, 9) for r in full if r.batch][:2] == [round(r.per_card, 9) for r in top if r.batch]
        assert [(str(r.layout), r.batch) for r in top] == [(str(r.layout), r.batch) for r in search_layouts(s, cards, top=2)]
        for r in full:
            if r.batch:
                r1 = evaluate(s.colocated().replace("layout", r.layout).replace("serving.batch", 1))
                assert not _feasible(r1) or full[0].per_card >= r1.per_card * (1 - 1e-12)


def test_pareto_is_exact_non_dominated_set():
    rng = random.Random(13)
    for _ in range(1500):
        pts = [(rng.choice([1.0, 2.0, rng.random() * 5, float("nan")]), rng.choice([1.0, 2.0, rng.random() * 5]), j)
               for j in range(rng.randint(0, 10))]
        ok = [p for p in pts if p[0] == p[0]]
        nd = {(p[0], p[1]) for p in ok
              if not any(q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in ok)}
        fr = pareto(pts)
        assert {(p[0], p[1]) for p in fr} == nd and len(fr) == len(nd)


def test_video_protein_extremes_and_msa_cap_warning():
    ch, mem = CHIPS["1P"], "hbm3e_8s_12h24g_9200"
    for mid, wl in (("wan2.1-1.3b", Workload(frames=1)), ("wan2.1-1.3b", Workload(height=64, width=64, steps=1)),
                    ("esm2-650m", Workload(seq_len=1)), ("alphafold2", Workload(seq_len=1)),
                    ("boltz-1", Workload(samples=25, recycles=20))):
        r = evaluate(Scenario(model=mid, chip=ch, mem_id=mem, workload=wl))
        assert math.isfinite(r.latency) and r.latency > 0 and r.per_card > 0
    prev = 0.0
    for f in (1, 9, 49, 129):                           # latency monotone in frames
        lat = evaluate(Scenario(model="wan2.1-1.3b", chip=ch, mem_id=mem, workload=Workload(frames=f))).latency
        assert lat >= prev
        prev = lat
    big = evaluate(Scenario(model="wan2.1-1.3b", chip=ch, mem_id=mem, serving=Serving(batch=65536)))
    assert not big.fits and math.isfinite(big.latency)
    for mid, cap in (("alphafold2", 508), ("boltz-1", 4096)):
        w = evaluate(Scenario(model=mid, chip=ch, mem_id=mem, workload=Workload(msa=16384))).warnings
        assert any(f"上限 {cap} 行" in x for x in w)
        assert not any("上限" in x for x in evaluate(Scenario(model=mid, chip=ch, mem_id=mem)).warnings)
