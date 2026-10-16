"""0.112: randomized audit standing tests (radix pin, spec_k / MoE skew / dtype / PP invariants) and the perf memos
(bit-identical to the unmemoised functions)."""
import math
import random

from accel_dse import api as A
from accel_dse.core import overlap as O
from accel_dse.core import pdqueue as PQ
from accel_dse.core.dtypes import FORMATS, FormatSupport, fmt, gemm_exec
from accel_dse.core.prefixcache import tree_depth_tail, tree_groups, tree_resident


def _H(lv, C):
    g = tree_groups(lv)
    return tree_depth_tail(g, tree_resident(lv, C, g))


def test_radix_single_root_always_resident_and_monotone():
    # was: H_1 = 1.0 at C = 1697 (exact path) but 0.9999 at C = 17672 (Che: h = 1 − e^{−T} for the shared root)
    lv = [(64, 1, 0.7057356849014481), (2000, 12, 0.10627380808825074), (500, 5, 0.26187346769779296)]
    assert _H(lv, 1697.1294339496917)[0] == 1.0 and _H(lv, 17671.67450092482)[0] == 1.0
    R = random.Random(112)
    for _ in range(25):
        lv = [(R.choice([16, 64, 500, 2000]), R.choice([1, 1, 3, 12]), R.uniform(0, 1.6)) for _ in range(R.randint(1, 3))]
        tot = sum(L * math.prod(n for _, n, _ in lv[:k + 1]) for k, (L, _, _) in enumerate(lv))
        prev, pm = None, None
        for C in sorted(R.uniform(0, 1.1 * tot) for _ in range(6)):
            H = _H(lv, C)
            # levels whose cumulative path fits: across that boundary LRU itself is not monotone (a deeper level that
            # starts to fit evicts siblings — RadixLRU shows the same cliff), so monotonicity is checked within it
            m = sum(1 for k in range(len(lv)) if sum(L for L, _, _ in lv[:k + 1]) <= C)
            if m != pm:
                prev, pm = None, m
            assert all(-1e-12 <= x <= 1 + 1e-12 for x in H) and all(b <= a + 1e-12 for a, b in zip(H, H[1:])), (lv, C, H)
            if prev is not None:
                assert all(b >= a - 1e-9 for a, b in zip(prev, H)), (lv, C, prev, H)
            if lv[0][1] == 1 and C >= lv[0][0]:
                assert H[0] >= 1 - 1e-12, (lv, C, H)
            prev = H


def _s(sc):
    return A.api_eval({"scenario": sc, "sensitivity": False})["summary"]


def test_spec_moe_pp_invariants():
    base = {"model": "deepseek-v3", "layout": {"tp": 8, "ep": 8}, "serving": {"phase": "decode", "batch": 16, "ctx": 4096}}
    sv = base["serving"]
    s0 = _s(base)
    lo, hi, z = (_s({**base, "serving": {**sv, "spec_k": 2, "spec_accept": a}}) for a in (0.3, 0.8, 0.0))
    assert hi["tok_s"] >= lo["tok_s"] and lo["step_ms"] == hi["step_ms"] and z["tok_s"] <= s0["tok_s"]
    assert _s({**base, "serving": {**sv, "spec_accept": 0.9}}) == s0           # accept is inert without spec_k
    sk = [_s({**base, "serving": {**sv, "moe_skew": x}})["step_ms"] for x in (1.0, 2.0, 7.9, 64.0)]
    assert all(b >= a * (1 - 1e-12) for a, b in zip(sk, sk[1:]))      # 0.141: flat under the rank-max default
    big = _s({**base, "serving": {**sv, "moe_skew": 64.0}})
    assert any("上限" in w for w in big["warnings"])
    hot = [0.0] * 256; hot[0] = 1.0
    assert _s({**base, "serving": {**sv, "moe_expert_load": hot}})["step_ms"] <= big["step_ms"] * (1 + 1e-12)
    pp = {"model": "qwen3-8b", "layout": {"pp": 4}, "serving": {"phase": "decode", "batch": 2, "ctx": 2048}}
    assert _s(pp)["microbatches"] == 2


def test_dtype_exec_matrix_flags():
    B = lambda n: "fp8" if n.startswith("fp8") else n
    R = random.Random(3)
    F = list(FORMATS)
    for w in F:
        for a in F:
            for _ in range(4):
                nat = {B(f): float(R.choice([1, 2, 4])) for f in F if R.random() < 0.5}
                nat[R.choice(["bf16", "fp16", "fp32"])] = 1.0
                sup = FormatSupport(rates=tuple(nat.items()))
                ex, rate, conv, flag = gemm_exec(w, a, sup)
                assert sup.native(ex) and rate == sup.rate(ex)
                assert (conv == "none") == (flag == 0)
                assert bool(flag & 1) == (B(w) != B(ex)) and bool(flag & 2) == (B(a) != B(ex))
                assert fmt(ex).compute_bits >= fmt(a).compute_bits or not sup.native(a)


def test_perf_memos_bit_identical():
    R = random.Random(9)
    for _ in range(40):
        items = [(R.random() * 1e-5, {"hot": R.random() * 1e7, "kv": R.random() * 1e6}, R.choice([0.0, 0.0, 1e-6]))
                 for _ in range(R.randint(1, 6))]
        segs = [(items, R.randint(1, 60))]
        f = {"hot": R.random(), "kv": 1.0, "act": 1.0}
        args = (segs, f, 1e-12 * R.uniform(1, 3), R.choice([2e6, 3e7]), R.random() < 0.5, 1.0, 1e-6, 0.0)
        O._PIPE_MEMO.clear()
        a = O.stage_pipe_time(*args)
        assert a == O.stage_pipe_time(*args) == O._stage_pipe_time(*args)
    raw = PQ._gamma_q.__wrapped__
    for _ in range(200):
        a, z = R.uniform(0.2, 600), R.uniform(0, 900)
        assert PQ._gamma_q(a, z) == raw(a, z)


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "8410808fd4f1e578"   # 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "b8ca44d92e7589cd"   # 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)
