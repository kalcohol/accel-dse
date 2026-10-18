"""0.237: SGLang short-prompt TTFT p99 — the topped-up latency and the follower status are correlated with the
prompt's own wait (wait-correlated W ⊕ latency, queueing.mg1_corr_sum_quantile); DES p99 parity over many seeds;
runtime guard for the phase-law paths (test_core_052 CLI / KPI-type cases)."""
import math
import random
import time

from accel_dse.core import pdsim
from accel_dse.core.disagg import disagg_report
from accel_dse.core.pdsim import simulate
from accel_dse.core.queueing import mg1_corr_sum_quantile, mg1_sum_quantile
from test_core_234 import _pd


def test_corr_reduces_to_independent_sum():
    lam, svc = 10.0, [(0.9, 0.022), (0.1, 0.024)]
    rows = [(0.9, 0.022, 0.022, math.inf), (0.1, 0.024, 0.024, math.inf)]
    for q in (0.5, 0.9, 0.99):
        a = mg1_corr_sum_quantile(lam, svc, rows, q)
        b = mg1_sum_quantile(lam, [0.022, 0.024], [0.9, 0.1], [0.022, 0.024], q)
        assert abs(a - b) <= 2e-3 * b, (q, a, b)


def test_corr_marginal_and_monte_carlo():
    # M/G/1 FCFS with a two-point latency: topped iff an arrival came during the own wait + S′.  Lindley simulation
    # of exactly that model vs the closed form (only the correlation differs from the independent convolution)
    lam, sa, sc, sp = 12.0, 0.020, 0.034, 0.004
    rng = random.Random(237)
    w, ttft, arr = 0.0, [], []
    t = 0.0
    times = []
    for _ in range(400000):
        t += rng.expovariate(lam)
        times.append(t)
    W = []
    for j in range(len(times)):
        if j:
            w = max(0.0, w + 0.021 - (times[j] - times[j - 1]))
        W.append(w)
    for j in range(len(times) - 1):
        topped = times[j + 1] - times[j] < W[j] + sp
        ttft.append(W[j] + (sc if topped else sa))
    ttft.sort()
    n = len(ttft)
    rows = [(1.0, sa, sc, sp)]
    for q in (0.9, 0.99):
        a = mg1_corr_sum_quantile(lam, [(1.0, 0.021)], rows, q)
        assert abs(a - ttft[int(q * n)]) <= 0.02 * ttft[int(q * n)], (q, a, ttft[int(q * n)])
    # the 0.236 independent convolution with the same marginal π (MC: 0.083) misses the tail by ≈ 9 %
    pi = 0.0834
    ind = mg1_sum_quantile(lam, [0.021, 0.0210001], [1 - pi, pi], [sa, sc], 0.99)
    assert ind < 0.93 * ttft[int(0.99 * n)]


def test_sglang_short_prompt_p99_parity_and_mean_unchanged():
    # DES p99 over 6 seeds (n 3000): 0.236 analytic was −0.15 (2500) / −0.12 (3000); now within ±0.08
    for prompt, mean in ((2500, 26.86), (3000, 31.85)):
        rep, ctx = pdsim.capture_ctx(_pd(prompt=prompt, cap=48.0, load=0.85, chunk_profile="sglang"))
        c = rep["queue"]["modes"]["coloc_chunked"]
        assert abs(c["ttft_ms"]["mean"] - mean) < 0.02           # the mean is unchanged (same marginal π)
        p99 = sorted(simulate(ctx, rep["queue"]["lambda_rps"], mode="coloc_chunked", n_req=3000, warmup=600,
                              seed=sd)["ttft"]["p99"] * 1e3 for sd in range(1, 7))
        des = sum(p99[1:-1]) / 4                                   # trimmed mean over seeds (DES excursions)
        err = (c["ttft_ms"]["p99"] - des) / des
        assert abs(err) <= 0.08, (prompt, c["ttft_ms"]["p99"], des, err)


def test_runtime_guard_phase_law_paths():
    # 0.236 perf limits keep these well under a minute each (they reached > 22 min / > 30 min without them).
    # 0.238: CPU time, not wall (the suite runs files in parallel: 66.95 s wall under load vs 57.6 s alone), and the
    # phase-law paths (the colocated modes) are guarded on their own at 45 s (24.4 s alone, 30.0 under suite load) — the rest of the vllm 4096 cv 0.5
    # report is the disaggregated bulk law (≈ 60 % of its 52.6 s already in 0.237), kept under a 90 s gross bound.
    from accel_dse.core import pdqueue
    spent = [0.0]

    def timed(fn):
        def run(*a, **k):
            c0 = time.process_time()
            try:
                return fn(*a, **k)
            finally:
                spent[0] += time.process_time() - c0
        return run
    saved = (pdqueue._coloc_chunked, pdqueue._coloc_prefill_first)
    pdqueue._coloc_chunked, pdqueue._coloc_prefill_first = timed(saved[0]), timed(saved[1])
    try:
        _guard_cases(spent)
    finally:
        pdqueue._coloc_chunked, pdqueue._coloc_prefill_first = saved


def _guard_cases(spent):
    for kw in (dict(prompt=4096, cap=48.0, chunk_profile="sglang_mixed"), dict(prompt=6000, cap=48.0, chunk_profile="sglang"),
               dict(prompt=6000, cap=48.0, chunk_profile="vllm_v0"), dict(prompt=4096, chunk_profile="vllm", prompt_cv=0.5)):
        spent[0] = 0.0
        c0 = time.process_time()
        q = disagg_report(_pd(**kw))["queue"]
        total = time.process_time() - c0
        assert q["modes"]["coloc_chunked"].get("stable") is not None
        assert spent[0] > 0, kw
        assert spent[0] < 45, (kw, spent[0])
        assert total < 90, (kw, total)
