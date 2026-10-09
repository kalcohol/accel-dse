"""0.65: greedy bulk-service prefill queue (M/G^[b]/1, core/bulkq) instead of the fluid τ_b = T(b)/b M/G/1; DES
stability by drift / utilisation instead of "all requests finished"; idle EP ranks' SRAM / link / dequant counts;
every integer output-stationary row tile.  Invariant / property tests."""
from __future__ import annotations

import math
import random

from accel_dse import api
from accel_dse.core import bulkq, pdsim, pdqueue as pq
from accel_dse.core.energy import energy_report, scaleup_bytes
from accel_dse.core.evaluate import evaluate
from accel_dse.core.memplan import ACC_BYTES, gemm_blocking
from accel_dse.core.parallel import Layout
from accel_dse.core.queueing import md1
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
_QN = {"model": "qwen3-next-80b-a3b", "mem_id": HBM, "layout": {"tp": 2, "ep": 2},
       "serving": {"batch": 16, "prompt": 1024, "out_len": 512, "ttft_slo_ms": 80, "tpot_slo_ms": 100},
       "pd": {"enabled": True, "prefill_layout": {"tp": 1}, "prefix_hit": 0.0, "load": 0.6}}
T = lambda k: 0.0713 + 0.0081 * (k - 1)          # Qwen3-Next-like batch wall (s)


def _ctx(body):
    s = api.scenario_from_body({"scenario": body, "chip_preset": "1P"})
    return pdsim.capture_ctx(s)


def _det(b):
    return tuple(((T(k), 1.0),) for k in range(1, b + 1))


def _mc(lam, b, n=60000, seed=5):
    """Greedy bulk server Monte Carlo (pdsim.PrefillReplica logic) → sorted sojourns (wait + own wall)."""
    rng = random.Random(seed)
    t = free = 0.0
    q, out = [], []
    arr = []
    for _ in range(n):
        t += rng.expovariate(lam)
        arr.append(t)
    i = 0
    while i < n or q:
        if not q:
            q.append(arr[i]); free = max(free, arr[i]); i += 1
        while i < n and arr[i] <= free:
            q.append(arr[i]); i += 1
        k = min(len(q), b)
        batch, q = q[:k], q[k:]
        free += T(k)
        out += [free - a for a in batch]
    out = out[n // 10:]
    out.sort()
    return out


# ------------------------------------------------------------------ 1. bulk queue
def test_bulk_b1_is_md1():
    for rho in (0.2, 0.6, 0.9):
        lam = rho / T(1)
        L = bulkq.bulk_law(lam, 1, _det(1))
        w = md1(lam, T(1), (0.5, 0.9, 0.99))
        assert abs(L["mean"] - (w["mean"] + T(1))) < 1e-6 * L["mean"], rho
        for q in (0.5, 0.9, 0.99):
            assert abs(bulkq.law_quantile(L, q) - (w[f"p{q * 100:g}"] + T(1))) < 0.01 * L["mean"], (rho, q)


def test_bulk_matches_monte_carlo():
    for b, rho in ((2, 0.7), (4, 0.5), (4, 0.9), (8, 0.8)):
        lam = rho * b / T(b)
        L = bulkq.bulk_law(lam, b, _det(b))
        xs = _mc(lam, b)
        assert abs(L["mean"] - sum(xs) / len(xs)) < 0.03 * L["mean"], (b, rho)
        for q in (0.5, 0.9):
            sim = xs[int(q * len(xs))]
            assert abs(bulkq.law_quantile(L, q) - sim) < 0.05 * sim, (b, rho, q)


def test_bulk_mean_little_consistent_and_monotone_property():
    """Chain mean (Little) = mean of the tagged law; mean, p90 and busy share non-decreasing in λ; busy share ≥ the
    fluid ρ = λ·T(b)/b (partial batches occupy the server longer per request); stability iff λ·T(b) < b."""
    for b in (2, 4, 16):
        prev = (0.0, 0.0, 0.0)
        for i in range(1, 10):
            lam = (i / 10) * b / T(b)
            m = bulkq.bulk_mean(lam, b, _det(b))
            L = bulkq.bulk_law(lam, b, _det(b))
            assert abs(m["mean"] - L["mean"]) < 2e-3 * m["mean"], (b, i)
            p90 = bulkq.law_quantile(L, 0.9)
            busy = bulkq.bulk_busy(lam, b, _det(b))
            assert busy >= lam * T(b) / b - 1e-9 and busy < 1
            cur = (m["mean"], p90, busy)
            assert all(c >= p * (1 - 1e-6) for c, p in zip(cur, prev)), (b, i, cur, prev)
            prev = cur
        assert bulkq.bulk_mean(1.0001 * b / T(b), b, _det(b)) is None


def test_bulk_mixture_law_of_batch_mean():
    """D_k = law of the mean of k draws: mean preserved, variance / k (before atom merging)."""
    per = [(0.05, 0.3), (0.1, 0.4), (0.3, 0.3)]
    mu = sum(t * w for t, w in per)
    var = sum(w * (t - mu) ** 2 for t, w in per)
    for k in (1, 2, 3, 8):
        D = bulkq.batch_wall_law(k, per)
        m = sum(t * w for t, w in D)
        v = sum(w * (t - m) ** 2 for t, w in D)
        assert abs(sum(w for _, w in D) - 1) < 1e-9 and abs(m - mu) < 1e-9
        assert v <= var / k * (1 + 1e-6)


def test_prefill_server_picks_min_exact_mean():
    rep, ctx = _ctx(_QN)
    pp = ctx["ppool"]
    lam = rep["queue"]["lambda_rps"] / ctx["r_p"]
    pre = pq._prefill_server(pp, lam, ctx["pts"])
    means = {}
    for b in pq.B_CAPS:
        x = pq._prefill_server(pp, lam, ctx["pts"], cap=b)
        if x is not None and x["b"] == b:
            means[b] = x["mean"]
    assert pre["mean"] <= min(means.values()) * (1 + 1e-12) and pre["b"] in means
    assert pre["b"] > 1 and "law" in pre and abs(pre["law"]["mean"] - pre["mean"]) < 2e-3 * pre["mean"]


def test_pd_ttft_close_to_des_cap2():
    """The 0.64 report: cap 2 p90 79.4 ms closed vs ≈ 86 DES at load 0.6.  0.65 within 6 %."""
    _, ctx = _ctx(_QN)
    c = {**ctx, "_cap": 2, "slo": None}
    lam = 1.7353
    x = pq._pd_mode(c, lam)
    s = pdsim.simulate(c, lam, "pd", n_req=6000, warmup=800, seed=3)
    assert s["batch_cap"] == 2 and x["prefill"]["batch_cap"] == 2
    for q in ("p50", "p90"):
        assert abs(x["ttft"][q] - s["ttft"][q]) < 0.06 * s["ttft"][q], (q, x["ttft"][q], s["ttft"][q])


# ------------------------------------------------------------------ 2. DES stability
def test_des_stability_flags_overload_even_when_complete():
    """Qwen3-Next PD is decode-limited (stable 2.89 rps, prefill alone ≈ 14): past the limit every measured request
    still finishes (0.64's criterion) but the post-TTFT time grows across the run → unstable; below it, stable."""
    rep, ctx = _ctx(_QN)
    lam_s = rep["queue"]["modes"]["pd"]["stable_rate_rps"]
    hi = pdsim.simulate(ctx, 1.3 * lam_s, "pd", n_req=1500, warmup=300)
    lo = pdsim.simulate(ctx, 0.5 * lam_s, "pd", n_req=1500, warmup=300)
    assert hi["complete"] and not hi["stable"] and hi["drift_post"] > pdsim.DRIFT_TOL
    assert lo["stable"] and max(lo["drift_ttft"], lo["drift_post"]) < pdsim.DRIFT_TOL and lo["util"] < pdsim.UTIL_MAX
    r = pdsim.slo_rate(ctx, "pd", 1e3, 1e3, lam_s, n_req=800, warmup=200)        # no SLO → the DES stability limit
    assert 0.8 * lam_s < r < 1.15 * lam_s, (r, lam_s)


def test_drift_statistic():
    assert pdsim._drift([1.0] * 100) == 0.0
    assert abs(pdsim._drift([1.0 + i / 99 for i in range(100)]) - 0.8 / 1.5) < 0.02   # windows 1.1 … 1.9, mean 1.5
    assert pdsim._drift([1.0] * 10) == 0.0                                       # too short: no verdict


# ------------------------------------------------------------------ 3. idle EP ranks / integer tiles
def test_idle_ep_ranks_sram_link_dequant_are_expert_parts():
    for b in (1, 3):
        r = evaluate(Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(dp=4, ep=4), serving=Serving(batch=b)))
        a = energy_report(r)["counts_per_unit"]
        st = r.stages[0]
        ea = st.exp_acts
        assert 0 < ea["sram"] < st.sram_bytes and 0 <= ea["conv_w"] <= st.conv_w + 1e-18 and 0 < ea["link_frac"] <= 1
        busy, idle = b, 4 - b
        units = r.throughput * r.step
        sram = sum(s.sram_bytes * busy + s.exp_acts["sram"] * idle for s in r.stages) * r.microbatches / units
        link = sum(scaleup_bytes(s.time) * (busy + s.exp_acts["link_frac"] * idle) for s in r.stages) \
            * r.microbatches / units
        assert abs(a["sram"] - sram) < 1e-6 * sram, (b, a["sram"], sram)
        assert abs(a.get("link", 0.0) - link) <= 1e-6 * max(link, 1.0)
    # no idle ranks (batch ≥ dp) → unchanged accounting: every rank charged its own step
    r = evaluate(Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(dp=4, ep=4), serving=Serving(batch=8)))
    a = energy_report(r)["counts_per_unit"]
    units = r.throughput * r.step
    assert abs(a["sram"] - sum(s.sram_bytes for s in r.stages) * 4 * r.microbatches / units) < 1e-6 * a["sram"]


def test_gemm_blocking_integer_tile_optimal_property():
    rng = random.Random(65)
    for _ in range(400):
        m, k, n = rng.randint(1, 3000), rng.randint(1, 4096), rng.randint(1, 8192)
        bud = rng.choice((1e5, 2 ** 20, 4 * 2 ** 20, 24 * 2 ** 20))
        A, W = m * k * 2.0, k * n * rng.choice((0.5, 1.0, 2.0))
        ra, rw = gemm_blocking(m, k, n, A, W, bud)
        brute = [A * math.ceil(W / bud - 1e-12) + W, A + W * max(1, math.ceil(A / bud - 1e-12))]
        brute += [A * math.ceil(n / min(n, int(bud // (bm * ACC_BYTES)))) + W * math.ceil(m / bm)
                  for bm in range(1, m + 1) if int(bud // (bm * ACC_BYTES)) >= 1]
        assert abs(A * ra + W * rw - min(brute)) <= 1e-9 * min(brute), (m, k, n, bud)
