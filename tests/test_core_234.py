"""0.234: multi-prompt packing within one iteration budget — vLLM V1 (running decodes first, then the waiting queue
FCFS while token_budget > 0 and len(running) < max_num_seqs, the last admitted request truncated) and SGLang
PrefillAdder (chunked request first, whole requests, the first that does not fit truncated).  Multi-piece fused step,
budget / starvation invariants in the DES, parity, energy identity, fuzz."""
import random

from accel_dse.core import pdsim
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import (_chunk_fused, _fused_multi, _fused_step, _packed_fused, _Pool, chunk_at,
                                    chunk_spec, pack_fill)
from accel_dse.core.pdsim import compare, simulate
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.core.validation import V4_BANDS

HB = "hbm3e_8s_12h24g_9200"


def _pd(prompt=4096, out=512, batch=64, cap=0.0, **pd):
    kw = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6)
    kw.update(pd)
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2), mem_cap_GB=cap,
                    serving=Serving(batch=batch, prompt=prompt, out_len=out), pd=PDConfig(**kw))


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_pack_fill_fcfs_budget():
    rng = random.Random(2340)
    for _ in range(300):
        q = [rng.randint(1, 5000) for _ in range(rng.randint(0, 6))]
        bud = rng.randint(1, 9000)
        f = pack_fill(bud, q)
        assert sum(t for _, t in f) <= bud
        assert [i for i, _ in f] == list(range(len(f)))           # FCFS, no skipping
        for j, (i, t) in enumerate(f):
            if j < len(f) - 1:
                assert t == q[i]                                   # only the last piece may be truncated
        if f and sum(q) >= bud:
            assert sum(t for _, t in f) == bud                     # the budget is used up when work remains


def test_multi_piece_reduces_to_one_piece_and_batched_prefill():
    pool = _Pool(_pd())
    S = 4096
    for k in (1, 8, 32):
        dec = pool.run("decode", k)
        plain = pool.run("prefill", 1, S, 0)
        assert _packed_fused(pool, k, [(S, 0, 0, S)]) == _chunk_fused(pool, k, S, 0, 0, S)
        assert _fused_multi(dec, [plain]) == _fused_step(dec, plain, 1.0, 0.0)
        t1, t2, t3 = (_packed_fused(pool, k, [(S, 0, 0, S)] * j) for j in (1, 2, 3))
        assert t1 < t2 < t3 < 3 * t1                               # weights read once: below three iterations
        # a whole + a partial piece lies between one and two whole prompts
        tp = _packed_fused(pool, k, [(S, 0, 0, S), (S, 0, 0, 1024)])
        assert t1 < tp < t2
    # no decode: j whole prompts = the batched prefill of j prompts
    for j in (2, 3):
        assert abs(_packed_fused(pool, 0, [(S, 0, 0, S)] * j) - pool.run("prefill", j, S, 0).step) < 1e-12


def test_spec_pack_flags():
    class P:
        chunk_budget = None
        chunk_tokens = 512
    for prof, pk in (("vllm", True), ("vllm_v0", True), ("sglang", True), ("sglang_mixed", True),
                     ("legacy_512", False), ("fixed", False)):
        P.chunk_profile = prof
        assert bool(chunk_spec(P, 288.0).get("pack")) is pk, prof


def _record():
    """Patch ColocChunked to log every chunk iteration: (spec, k, credited members, running, pieces); → (log, undo)."""
    log = []
    C_ = pdsim.ColocChunked
    o_start, o_packed, o_pf = C_._start, C_._packed, pdsim._packed_fused
    cur = {}

    def rec(self, dur, kind, payload):
        if kind == "rep_chunk_done":
            log.append((self.spec, len(self.running), list(payload[0]), list(self.running), cur.get("pieces", [])))
        return o_start(self, dur, kind, payload)

    def pf(pool, k, pieces):
        cur["pieces"] = list(pieces)
        return o_pf(pool, k, pieces)

    def packed(self):
        cur["pieces"] = []
        return o_packed(self)
    C_._start, C_._packed, pdsim._packed_fused = rec, packed, pf

    def undo():
        C_._start, C_._packed, pdsim._packed_fused = o_start, o_packed, o_pf
    return log, undo


def test_des_budget_never_exceeded_decodes_never_starved():
    rng = random.Random(234)
    for prof, cap, prompt in (("vllm", 48.0, 4096), ("vllm", 0.0, 4096), ("sglang_mixed", 48.0, 3000),
                              ("sglang", 48.0, 6000), ("vllm", 48.0, rng.choice((1500, 2500, 5000)))):
        s = _pd(prompt=prompt, out=128, batch=32, cap=cap, load=0.85, chunk_profile=prof)
        rep, ctx = pdsim.capture_ctx(s)
        lam = rep["queue"]["lambda_rps"]
        log, undo = _record()
        try:
            simulate(ctx, lam, mode="coloc_chunked", n_req=300, warmup=50, seed=rng.randint(1, 99))
        finally:
            undo()
        assert log, prof
        multi = 0
        for spec, k, members, running, pieces in log:
            assert pieces, prof
            toks = sum(e - q for _, _, q, e in pieces)
            if spec.get("nonfused"):
                assert toks <= spec["C"] and members == []         # SGLang: prefill-only, decodes paused
            else:
                assert toks + k <= spec["budget"], (prof, toks, k)  # budget includes the running decodes
                assert members == running                          # every running decode advances (decode-first)
            partial = [pc for pc in pieces if pc[3] < pc[0]]
            assert len(partial) <= 1                               # at most one prompt is partial
            multi += len(pieces) > 1
        assert multi > 0, prof                                     # packing actually happens at this load


def test_parity_packed_in_v4_bands():
    rng = random.Random(234)
    fz = rng.choice(((dict(prompt=3000, chunk_budget=6000, cap=48.0), "sglang_mixed"),
                     (dict(prompt=2500, cap=48.0, load=0.8), "vllm"),
                     (dict(prompt=1000, out=256, batch=32, cap=48.0, load=0.8), "vllm")))
    for kw, prof, keys in ((dict(cap=48.0), "vllm", None), (dict(cap=48.0), "sglang_mixed", None),
                           (dict(load=0.85), "vllm", None), (fz[0], fz[1], None),
                           (dict(cap=48.0), "sglang", ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_mean", "tpot_p90"))):
        s = _pd(chunk_profile=prof, **kw)
        r = compare(s, n_req=800, warmup=200, seeds=(11, 12), modes=("coloc_chunked",))
        e = r["modes"]["coloc_chunked"]["err"]
        for k in keys or ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_mean", "tpot_p90", "itl_max"):
            assert abs(e[k]) <= V4_BANDS[k], (prof, kw, k, e)


def test_report_pack_fields_and_old_to_new():
    q = disagg_report(_pd(), energy=_t(idle_W=20.0))["queue"]
    c = q["modes"]["coloc_chunked"]
    pk = c["prefill"]["pack"]
    assert set(pk["b_env"]) <= {3, 4} and 3 in pk["b_env"]       # ⌊(16384 − k) / 4096⌋ whole prompts (k = 0: 4)
    # A: 0.233 TTFT mean 42.06 / p99 82.93 → packing two/three compute-bound prompts delays the first one
    assert 42.0 < c["ttft_ms"]["mean"] < 43.5 and c["ttft_ms"]["p99"] > 82.93
    d = disagg_report(_pd(cap=48.0))["queue"]["modes"]["coloc_chunked"]["prefill"]["pack"]
    assert all(b == 1 for b in d["b_env"]) and all(0 < p_ < 1 for p_ in d["pi_env"])


def test_fuzz_packing_energy_identity():
    rng = random.Random(234)
    for i in range(5):
        prof = ("vllm", "sglang", "sglang_mixed", "vllm_v0", "vllm")[i]
        kw = {"chunk_profile": prof}
        if prof == "sglang_mixed":
            kw["chunk_budget"] = rng.choice((2048, 3000, 6000))
        prompt = rng.choice((1000, 2048, 3000, 4096, 6000))
        s = _pd(prompt=prompt, out=256, batch=rng.choice((16, 32)), load=rng.uniform(0.3, 0.9),
                cap=rng.choice((0.0, 48.0)), **kw)
        q = disagg_report(s, energy=_t(idle_W=rng.uniform(0, 30)))["queue"]
        x = q["modes"]["coloc_chunked"]
        if x.get("stable"):
            t = x["ttft_ms"]
            assert 0 < t["p50"] <= t["p90"] <= t["p99"] + 1e-9
            if q["chunk_tokens"] and prof != "sglang":
                assert x["prefill"]["pack"]["b_env"]
        lam, o = q["lambda_rps"], s.serving.out_len
        for m, en in q["energy"].items():
            W = en["J_per_token"] / en["card_s_per_token"]
            assert 0 < W < 2000 and abs(en["J_per_token"] * lam * o / 8 / W - 1) < 1e-9, m
