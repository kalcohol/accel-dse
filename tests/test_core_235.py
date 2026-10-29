"""0.235: the §185 gaps — analytic SGLang non-mixed tail sharing (PrefillAdder truncation, cap 1) and the vLLM
bulk-regime (b+1)-th prompt head; adversarial non-multiple prompt lengths (analytic vs DES in the V4 bands) and a
fuzzer over prompt length / chunk / budget / max_num_seqs with the DES packing invariants."""
import random

from accel_dse.core import pdsim
from accel_dse.core.disagg import disagg_report
from accel_dse.core.pdqueue import _Pool, _tail_pi, _tail_rows, chunk_at
from accel_dse.core.pdsim import compare, simulate
from accel_dse.core.validation import V4_BANDS
from test_core_234 import _pd, _record

TTFT_TPOT = ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_mean", "tpot_p90")


def test_tail_rows_and_pi():
    pool = _Pool(_pd(prompt=6000, cap=48.0))
    rows = _tail_rows(pool, 0, [(1.0, 6000, 0), (1.0, 8192, 0), (1.0, 3000, 0)], 4096)
    a, b, c = rows
    # 6000 = 4096 + 1904: shared last iteration is longer than the tail alone, the share is its token part
    assert a["n"] == 2 and a["tail"] == 1904 and a["latpk"] > a["solo"] and a["pre"] < a["solo"]
    assert abs(a["packd"] - (a["pre"] + 1904 / 4096 * a["tsh"])) < 1e-12
    # a multiple of C: nothing to share
    assert b["tail"] == 4096 and b["packd"] == b["solo"] == b["latpk"] and b["pre"] == float("inf")
    # one-iteration prompt: S′ = 0 → π ≈ ρ² ≪ π of the two-chunk prompt at the same load
    pis, rho = _tail_pi(5.0, [1.0, 1.0, 1.0], rows)
    assert 0 < rho < 1 and pis[1] == 0.0 and 0 < pis[2] < pis[0] < 1


def test_sglang_tail_share_reported_only_off_multiples():
    q = disagg_report(_pd(prompt=6000, cap=48.0, load=0.85, chunk_profile="sglang"))["queue"]
    ts = q["modes"]["coloc_chunked"]["prefill"].get("tail_share")
    assert ts and ts["C"] == 4096 and 0.05 < ts["pi"][0] < 0.4
    # case D (4096 = C) and E (16384 = 4·C): no tail → the 0.234 numbers
    # 0.465 re-pin: qwen3-8b vector time. 40.488 → 40.648, 190.310 → 190.952
    for kw, mean in ((dict(cap=48.0), 40.648), (dict(prompt=16384, out=256, batch=32, cap=48.0), 190.952)):
        c = disagg_report(_pd(chunk_profile="sglang", **kw))["queue"]["modes"]["coloc_chunked"]
        assert "tail_share" not in c["prefill"] and abs(c["ttft_ms"]["mean"] - mean) < 0.02


def test_bulk_head_probability():
    q = disagg_report(_pd(prompt=700, cap=48.0, load=0.9))["queue"]
    c = q["modes"]["coloc_chunked"]
    assert c["stable"] and max(c["prefill"]["pack"]["b_env"]) >= 2


def test_adversarial_parity_non_multiples():
    cases = (dict(prompt=6000, cap=48.0, load=0.85, chunk_profile="sglang"),       # 4096 + 1904
             dict(prompt=2500, cap=48.0, load=0.85, chunk_profile="sglang"),       # C/2 < new < C
             dict(prompt=3000, cap=48.0, load=0.8, chunk_profile="vllm"),          # 2048 budget, tail sharing
             dict(prompt=700, cap=48.0, load=0.9, chunk_profile="vllm"),           # b = 2 + head
             dict(prompt=5000, load=0.85, chunk_profile="vllm"))                   # b = 3 + head
    for kw in cases:
        r = compare(_pd(**kw), n_req=800, warmup=200, seeds=(11, 12), modes=("coloc_chunked",))
        e = r["modes"]["coloc_chunked"]["err"]
        # itl_max is a lattice atom (one prompt length: walls n·T, MODEL §74) — not banded here
        for k in TTFT_TPOT:
            assert abs(e[k]) <= V4_BANDS[k], (kw, k, e)


def test_fuzz_packing_invariants():
    rng = random.Random(235)
    for _ in range(6):
        prof = rng.choice(("vllm", "sglang", "sglang_mixed"))
        budget = rng.choice((1024, 2048, 3000, 4096, 6000))
        batch = rng.choice((8, 16, 32))                                          # max_num_seqs (slots)
        prompt = rng.randint(300, 9000)
        s = _pd(prompt=prompt, out=96, batch=batch, cap=48.0, load=rng.uniform(0.7, 0.95), chunk_profile=prof,
                chunk_budget=budget)
        rep, ctx = pdsim.capture_ctx(s)
        if not rep["queue"]["modes"]["coloc_chunked"].get("stable"):
            continue
        log, undo = _record()
        try:
            simulate(ctx, rep["queue"]["lambda_rps"], mode="coloc_chunked", n_req=200, warmup=20,
                     seed=rng.randint(1, 999))
        finally:
            undo()
        for spec, k, members, running, pieces in log:
            toks = sum(e - q for _, _, q, e in pieces)
            if spec.get("nonfused"):
                assert toks <= spec["C"] and members == []
            else:
                assert toks + k <= spec["budget"] and toks <= chunk_at(spec, k) and members == running
            # vLLM V1 counts the prefilling requests in RUNNING: decodes + prompts in flight ≤ max_num_seqs;
            # max_num_partial_prefills does not exist in V1 → FCFS leaves at most one partial prompt
            assert k + len(pieces) <= batch, (prof, k, len(pieces), batch)
            assert sum(1 for S, p, q, e in pieces if e < S) <= 1


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
