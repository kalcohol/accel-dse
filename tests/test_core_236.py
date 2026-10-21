"""0.236: the 0.235 deferred packing mechanisms — the follower's head start (phase chain of the FCFS token stream:
a prompt that starts inside its predecessor's topped-up last iteration has its own pieces, tail and next head set
by that phase) and the vLLM bulk head moving work (the head batch carries it, the next batch starts with the
remainder); no head when the batch is slot-bound."""
from accel_dse.core import pdsim
from accel_dse.core.disagg import disagg_report
from accel_dse.core.pdqueue import _phase_iters, _phase_law, _Pool, _tail_pi, _tail_rows
from accel_dse.core.pdsim import simulate
from test_core_234 import _pd, _record


def test_phase_iters_leader_equals_tail_rows():
    pool = _Pool(_pd(prompt=6000, cap=48.0))
    r = _tail_rows(pool, 0, [(1.0, 6000, 0)], 4096)[0]
    a, b = _phase_iters(pool, 0, 6000, 0, 4096, 0, False), _phase_iters(pool, 0, 6000, 0, 4096, 0, True)
    assert abs(a["share"] - r["solo"]) < 1e-12 and abs(b["share"] - r["packd"]) < 1e-12
    assert abs(b["lat"] - r["latpk"]) < 1e-12 and abs(a["pre"] - r["pre"]) < 1e-12 and a["used"] == 1904
    # follower at φ = 1904: 2192 + 3808 tokens → its last iteration leaves only 288 for the next head
    f, fh = _phase_iters(pool, 0, 6000, 0, 4096, 1904, False), _phase_iters(pool, 0, 6000, 0, 4096, 1904, True)
    assert f["used"] == 3808 and fh["lat"] - f["lat"] < b["lat"] - a["lat"]      # the head start credit
    assert f["pre"] < a["pre"]                                                    # its last iteration starts sooner


def test_phase_law_multiples_and_leader_pi():
    pool = _Pool(_pd(prompt=6000, cap=48.0))
    m = _phase_law(pool, 0, [(1.0, 8192, 0)], 4096, 5.0)
    assert m["phases"] == {0: 1.0} and m["pi"] == 0.0 and len(m["atoms"]) == 1     # multiple of C: no sharing
    L = _phase_law(pool, 0, [(1.0, 6000, 0)], 4096, 5.0)
    pis, rho = _tail_pi(5.0, [1.0], _tail_rows(pool, 0, [(1.0, 6000, 0)], 4096))
    assert abs(L["rho"] - rho) < 0.01 and abs(L["pi"] - pis[0]) < 0.02
    assert set(L["phases"]) >= {0, 1904, 3808} and abs(sum(L["phases"].values()) - 1) < 1e-9
    assert abs(sum(w for w, *_ in L["atoms"]) - 1) < 1e-9


def _des_follower(kw, n_req=1500):
    """DES share of prompt starts that begin inside another prompt's iteration (the piece is not first in pack order)."""
    rep, ctx = pdsim.capture_ctx(_pd(**kw))
    c = rep["queue"]["modes"]["coloc_chunked"]
    log, undo = _record()
    try:
        simulate(ctx, rep["queue"]["lambda_rps"], mode="coloc_chunked", n_req=n_req, warmup=200, seed=7)
    finally:
        undo()
    starts = fol = 0
    for *_, pieces in log:
        for j, (S, p, q, e) in enumerate(pieces):
            if q == p:
                starts += 1
                fol += j > 0
    return c, fol / starts


def test_follower_share_matches_des():
    # the stationary phase law vs the DES token stream (SGLang prefill-only path)
    # (fused profiles keep followers at φ = 0 in the k ≥ 1 bins — cost, deferred; their follower_env is 0 there)
    for kw in (dict(prompt=6000, cap=48.0, load=0.85, chunk_profile="sglang"),
               dict(prompt=6000, cap=48.0, load=0.6, chunk_profile="sglang")):
        c, des = _des_follower(kw)
        ana = c["prefill"]["tail_share"]["follower"]
        assert abs(ana - des) < 0.03, (kw, ana, des)


def test_bulk_head_slot_bound_and_report():
    # 700-token prompts, budget 2048: b = 2 + head when a slot is free
    c = disagg_report(_pd(prompt=700, cap=48.0, load=0.9))["queue"]["modes"]["coloc_chunked"]
    hp = [x for x in c["prefill"]["pack"]["head_p_env"] if x is not None]
    assert hp and all(0 <= x < 1 for x in hp)
    # max_num_seqs 4 with ≥ 2 decodes running: b = B − k is slot-bound → no head (pdsim admits none)
    c2 = disagg_report(_pd(prompt=300, cap=48.0, load=0.9, batch=4))["queue"]["modes"]["coloc_chunked"]
    assert c2.get("stable")
    pk = c2["prefill"]["pack"]
    seen = False
    for b_, h_ in zip(pk["b_env"], pk["head_p_env"]):
        if b_ < (2048 - 4) // 300:          # b = B − k < ⌊C/new⌋: slot-bound
            assert h_ is None, pk
            seen = True
    assert seen, pk


def test_fuzz_phase_law_invariants():
    import random
    rng = random.Random(236)
    pool = _Pool(_pd(prompt=6000, cap=48.0))
    for _ in range(25):
        C = rng.choice((1024, 2048, 3000, 4096))
        pts = [(rng.uniform(0.2, 1.0), rng.randint(C // 2 + 1, 3 * C), 0) for _ in range(rng.choice((1, 1, 2)))]
        lam = rng.uniform(0.5, 12.0)
        L = _phase_law(pool, rng.choice((0, 1, 8)), pts, C, lam)
        if L["rho"] >= 1:
            continue
        tot = sum(w for w, _, _ in pts)
        assert abs(sum(L["phases"].values()) - 1) < 1e-9 and all(0 <= f < C for f in L["phases"])
        assert abs(sum(m for m, *_ in L["atoms"]) - tot) < 1e-6 * tot
        assert all(s > 0 and l_ >= s * 0.999 for _, s, l_, _ in L["atoms"])        # own latency ≥ its share
        assert 0 <= L["follower"] <= L["pi"] + 1e-4      # = π unless a mixture phase rounds to 0


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
