"""0.238: wait-correlated TTFT (0.237 topping π(w) and follower iff W > θ) in the fused chunked paths (vLLM,
sglang_mixed) where the running decodes leave C(k) = budget − k — per batch bin the follower phase law at the
(rounded) batch, quasi-static mixture over bins (queueing.mg1_corr_mix_sum_quantile); DES p99 parity at
n4000×10; runtime guard."""
import math
import random

from accel_dse.core.disagg import disagg_report
from accel_dse.core.queueing import mg1_corr_mix_sum_quantile, mg1_corr_sum_quantile, mg1_mix_sum_quantile
from test_core_234 import _pd
from tests.workmeter import guard


def test_mix_reduces_to_single_and_to_independent():
    lam, svc = 12.0, [(0.85, 0.021), (0.15, 0.024)]
    rows = [(0.7, 0.020, 0.034, 0.004, 0, False), (0.15, 0.021, 0.035, 0.005, 0, False),
            (0.15, 0.022, 0.026, 0.003, 0, True)]
    for q in (0.5, 0.9, 0.99):
        a = mg1_corr_sum_quantile(lam, svc, rows, q, theta=0.006)
        b = mg1_corr_mix_sum_quantile(lam, [(1.0, svc, rows, 0.006, 0.0)], q)
        assert abs(a - b) < 1e-9
    # no sharing anywhere (S′ = ∞) → the independent drifting-batch mixture of 0.55, vacations included
    bins = [(0.5, [(1.0, 0.020)], [(1.0, 0.020, 0.020, math.inf, 0, False)], math.inf, 0.002),
            (0.5, [(1.0, 0.024)], [(1.0, 0.024, 0.024, math.inf, 0, False)], math.inf, 0.003)]
    envs = [(0.5, [(1.0, 0.020, 0.020)], 0.002), (0.5, [(1.0, 0.024, 0.024)], 0.003)]
    for q in (0.5, 0.9, 0.99):
        a = mg1_corr_mix_sum_quantile(lam, bins, q)
        b = mg1_mix_sum_quantile(lam, envs, q)
        assert abs(a - b) / b < 0.01, (q, a, b)


def test_vacation_counts_for_topping_monte_carlo():
    # Lindley M/D/1 waits ⊕ U(0, v); topped (latency c) with prob 1 − e^{−λ(W + u + S′)}, else a
    lam, tau, a, c, sp, v = 14.0, 0.021, 0.020, 0.036, 0.004, 0.004
    rows = [(1.0, a, c, sp, 0, False)]
    rng = random.Random(7)
    w, xs = 0.0, []
    for _ in range(400000):
        u = rng.random() * v
        top = rng.random() < 1 - math.exp(-lam * (w + u + sp))
        xs.append(w + u + (c if top else a))
        w = max(0.0, w + tau - rng.expovariate(lam))
    xs.sort()
    for q in (0.9, 0.99):
        mc = xs[int(q * len(xs))]
        an = mg1_corr_mix_sum_quantile(lam, [(1.0, [(1.0, tau)], rows, math.inf, v)], q)
        assert abs(an - mc) / mc < 0.02, (q, an, mc)
    # the vacation residual raises the topping probability → quantile above the u-independent form shifted by v/2
    an0 = mg1_corr_mix_sum_quantile(lam, [(1.0, [(1.0, tau)], rows, math.inf, 0.0)], 0.99)
    assert an > an0


# DES TTFT (ms) at n_req=4000, warmup 800, seeds 1..10 (0.238, /workspace/sd238/seeds_*.txt); 0.237 analytic p99
# was 105.33 (vl4096, −0.052), 112.83 (mx4096, −0.082), 144.70 (vl6000, −0.032)
DES = {"vl4096": (dict(prompt=4096, cap=48.0, load=0.85, chunk_profile="vllm"), 71.48, 111.15),
       "mx4096": (dict(prompt=4096, cap=48.0, load=0.85, chunk_profile="sglang_mixed"), 76.54, 122.85),
       "vl6000": (dict(prompt=6000, cap=48.0, load=0.85, chunk_profile="vllm"), 97.44, 149.51)}


def test_fused_off_multiple_p99_parity_and_runtime():
    for name, (kw, p90, p99) in DES.items():
        # 0.312: the runtime guard was the report's own CPU time (same 60 s bound); 0.332: deterministic work (calls in
        # accel_dse code, tests/workmeter.py) at the work 60 quiet-box CPU seconds buy — CPU seconds inflate under -j 8
        with guard(60, f"238 {name}"):
            c = disagg_report(_pd(**kw))["queue"]["modes"]["coloc_chunked"]["ttft_ms"]
        assert abs(c["p90"] - p90) / p90 < 0.03, (name, c["p90"], p90)
        assert abs(c["p99"] - p99) / p99 < 0.03, (name, c["p99"], p99)


def test_vllm_3000_unchanged_residual():
    # the −0.044 p99 residual at vllm 3000 is not this mechanism (topping and follower correlations cancel)
    c = disagg_report(_pd(prompt=3000, cap=48.0, load=0.8, chunk_profile="vllm"))["queue"]["modes"]["coloc_chunked"]
    # 0.246: prompt 3000 at cap 48 is budget-chunked (C(B) = 1984 < 2 prompts) → the slot account replaces the 0.243
    # block (phase chain at B − 1, k̄): p99 72.39 → 71.61 ms, mean 34.02 → 34.03 (the account's backlog branch at small
    # p_sat; DES p99 ≈ 72.39 / (1 − 0.044) = 75.7 per 0.238, residual −0.044 → −0.054, same sign, not this mechanism)
    assert abs(c["ttft_ms"]["p99"] - 71.61) < 0.5 and abs(c["ttft_ms"]["mean"] - 34.03) < 0.1


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py);
# not in the runtime guard, which times the full report
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals(), skip=("test_fused_off_multiple_p99_parity_and_runtime",))
