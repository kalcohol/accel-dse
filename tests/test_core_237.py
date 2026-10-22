"""0.237: SGLang short-prompt TTFT p99 — the topped-up latency and the follower status are correlated with the
prompt's own wait (wait-correlated W ⊕ latency, queueing.mg1_corr_sum_quantile); DES p99 parity over many seeds;
runtime guard for the phase-law paths (test_core_052 CLI / KPI-type cases)."""
import math
import random

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
    # 0.277: CPU seconds still inflate under `run_tests.py -j 8` (62 s seen vs 29.3 s alone at 0.276 for the vllm cv 0.5
    # case), so the primary guard is now deterministic work: engine reruns (_Pool.run) and chunk-fused step costs
    # (_chunk_fused) per case, at ≤ 1.5 × their 0.277 counts — load-independent and tighter than either time bound for the
    # regression class these limits exist for (exploding loops).  The CPU-time bounds stay: 45 s / 90 s run alone, × 2
    # under the parallel runner (ACCEL_TEST_JOBS > 1), still far below the 22 min / 30 min regressions.
    # 0.279: the phase-law sub-guard also counts deterministic work — calls of the colocated modes' inner solvers
    # (_PHASE_FNS) and the summed size of what they return (law atoms, birth-death π, queue weights) at ≤ 1.5 × their
    # 0.279 values — and that is now its primary guard.  CPU seconds measured under `-j 8` inflated 9.5 × for the same
    # work (rt278b: vllm_v0 phase law 136 s CPU, 14.3 s alone, identical work counts), so under the parallel runner the
    # CPU-time bounds are gross catastrophe bounds (× 6: 270 s / 540 s, still ≥ 4 × below the 22 min / 30 min
    # regressions); run alone they stay 45 s / 90 s.
    # 0.332: the two time bounds are deterministic work too (tests/workmeter.py: calls in accel_dse code), at the work
    # 45 s / 90 s of quiet-box CPU buy — the alone thresholds, now also under -j 8 (was × 6 there), load-independent.
    from accel_dse.core import pdqueue
    spent = [0.0]
    work = {"run": 0, "chunk": 0, "meter": None}

    def timed(fn):
        def run(*a, **k):
            c0 = work["meter"].now()
            try:
                return fn(*a, **k)
            finally:
                spent[0] += work["meter"].now() - c0
        return run

    def counted(key, fn):
        def run(*a, **k):
            work[key] += 1
            return fn(*a, **k)
        return run
    def sized(i, fn):
        def run(*a, **k):
            r = fn(*a, **k)
            work["calls"][i] += 1
            work["size"][i] += _size(r)
            return r
        return run
    saved = (pdqueue._coloc_chunked, pdqueue._coloc_prefill_first, pdqueue._Pool.run, pdqueue._chunk_fused)
    saved_fns = tuple(getattr(pdqueue, n) for n in _PHASE_FNS)
    pdqueue._coloc_chunked, pdqueue._coloc_prefill_first = timed(saved[0]), timed(saved[1])
    pdqueue._Pool.run, pdqueue._chunk_fused = counted("run", saved[2]), counted("chunk", saved[3])
    for i, n in enumerate(_PHASE_FNS):
        setattr(pdqueue, n, sized(i, saved_fns[i]))
    try:
        _guard_cases(spent, work)
    finally:
        pdqueue._coloc_chunked, pdqueue._coloc_prefill_first, pdqueue._Pool.run, pdqueue._chunk_fused = saved
        for n, f in zip(_PHASE_FNS, saved_fns):
            setattr(pdqueue, n, f)


def _size(r) -> int:
    """Deterministic size of a solver result: elements of every (nested) list / tuple / dict."""
    if isinstance(r, dict):
        return sum(_size(v) for v in r.values())
    if isinstance(r, (list, tuple)):
        return len(r) + sum(_size(v) for v in r if isinstance(v, (list, tuple, dict)))
    return 0


# 0.277 work counts per case (fresh process, box): (_Pool.run calls, _chunk_fused calls)
_WORK = {"sglang_mixed": (13710, 3360), "sglang": (9010, 106), "vllm_v0": (17105, 6341), "vllm": (37869, 24090)}
# 0.279 phase-law inner solvers: per case (calls, summed result sizes) in _PHASE_FNS order (fresh process, box)
_PHASE_FNS = ("_phase_law", "_phase_iters", "_queue_weights", "_queue_k_law", "_decode_birth_death", "_occ_tau", "_mg1_busy",
              "_slot_backlog")
_PHASE_WORK = {
    "sglang_mixed": ((1642, 286, 168, 24, 250, 159, 164, 24), (48757, 0, 11005, 96177, 350862, 0, 20284, 5529)),
    "sglang": ((37, 69, 0, 0, 179, 103, 140, 19), (6117, 0, 0, 0, 185003, 0, 27831, 5529)),
    "vllm_v0": ((1784, 352, 182, 26, 261, 202, 176, 26), (56422, 0, 7971, 114534, 396383, 0, 41897, 6111)),
    "vllm": ((1628, 1950, 168, 24, 250, 151, 164, 24), (488325, 0, 16985, 93174, 356540, 0, 101211, 5820)),
}


def _guard_cases(spent, work):
    from tests.workmeter import guard, within
    for kw in (dict(prompt=4096, cap=48.0, chunk_profile="sglang_mixed"), dict(prompt=6000, cap=48.0, chunk_profile="sglang"),
               dict(prompt=6000, cap=48.0, chunk_profile="vllm_v0"), dict(prompt=4096, chunk_profile="vllm", prompt_cv=0.5)):
        spent[0] = 0.0
        work.update(run=0, chunk=0, calls=[0] * len(_PHASE_FNS), size=[0] * len(_PHASE_FNS))
        prof = kw["chunk_profile"]
        with guard(90, f"237 report {prof}") as m:              # whole report: ≤ 90 s-equivalent work
            work["meter"] = m
            q = disagg_report(_pd(**kw))["queue"]
        within(m, spent[0], 45, f"237 phase-law {prof}")        # colocated phase-law paths: ≤ 45 s-equivalent
        assert q["modes"]["coloc_chunked"].get("stable") is not None
        assert spent[0] > 0, kw
        r0, c0_ = _WORK[kw["chunk_profile"]]
        assert 0 < work["run"] <= 1.5 * r0 and work["chunk"] <= 1.5 * c0_ + 10, (kw, work["run"], work["chunk"])
        pc, ps = _PHASE_WORK[kw["chunk_profile"]]
        for n, c, c_ref, z, z_ref in zip(_PHASE_FNS, work["calls"], pc, work["size"], ps):
            assert c <= 1.5 * c_ref + 10 and z <= 1.5 * z_ref + 10, (kw, n, c, c_ref, z, z_ref)
        assert work["calls"][0] > 0 and work["calls"][4] > 0, (kw, work["calls"])


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py);
# not in the runtime guard, which times the full report
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals(), skip=("test_runtime_guard_phase_law_paths",))
