"""0.233: the fused chunked-prefill iteration evaluates each chunk as it runs (a prefill of the chunk's tokens with the
earlier tokens cached: its own activation footprint / spill, the prefix-KV re-read) fused with the decodes — was an
f-share of the whole prompt (which carried the whole prompt's activation spill).  Invariants, DES parity, energy,
fuzz."""
import random

from dataclasses import replace
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _chunk_bounds, _chunk_fused, _chunk_plan, _chunk_res, _fused_step, _Pool
from accel_dse.core.pdsim import compare
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.core.validation import V4_BANDS

HB = "hbm3e_8s_12h24g_9200"


def _pd(prompt=4096, out=512, batch=64, cap=0.0, **pd):
    kw = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6)
    kw.update(pd)
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2), mem_cap_GB=cap,
                    serving=Serving(batch=batch, prompt=prompt, out_len=out), pd=PDConfig(**kw))


def _t(**kw):
    t = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))
    # 0.300: the presets now fill the on-chip wire + NoC router; this file pins the ≤ 0.299 preset (no on-chip
    # movement terms) unless a test passes them
    return replace(t, **{k: None for k in ("pJ_bit_mm_onchip", "pJ_bit_hop_noc") if k not in kw})


def _act(r):
    return sum(st.dram.get("act", 0.0) for st in r.stages)


def _dram(r):
    return sum(st.dram["total"] for st in r.stages)


def test_single_chunk_equals_plain_prefill():
    pool = _Pool(_pd())
    for S, p in ((4096, 0), (4096, 2048), (1000, 0)):
        plain = pool.run("prefill", 1, S, p)
        assert _chunk_res(pool, S, p, p, S) is plain and _chunk_bounds(S, p, 8192) == [(p, S)]
        for k in (1, 16, 64):
            dec = pool.run("decode", k)
            n, f, rr = _chunk_plan(S, p, 8192)
            assert n == 1 and f == 1.0 and rr == 0.0
            # the ≤ 0.232 f-share formula and the per-chunk evaluation coincide for a one-chunk prompt
            assert _chunk_fused(pool, k, S, p, p, S) == _fused_step(dec, plain, 1.0, 0.0) == _fused_step(dec, plain, f, rr)


def test_chunking_vs_one_shot_and_spill():
    pool = _Pool(_pd(prompt=16384, out=256, batch=32))
    seen = set()
    for S, C in ((4096, 1024), (4096, 2048), (8192, 2048), (16384, 4096), (16384, 2048)):
        one = pool.run("prefill", 1, S, 0)
        ch = [pool.run("prefill", 1, e, q) for q, e in _chunk_bounds(S, 0, C)]
        t_sum, d_sum = sum(c.ttft for c in ch), sum(_dram(c) for c in ch)
        if _act(one) == 0:
            # no spill: chunking only adds work (weights re-read per chunk, prefix KV re-read)
            assert t_sum >= one.ttft and d_sum >= _dram(one), (S, C)
        else:
            seen.add("spill")
            assert all(_act(c) < _act(one) for c in ch)
            if d_sum < _dram(one):
                seen.add("cheaper")
    assert seen == {"spill", "cheaper"}          # 16384 tokens spill activations; 4096-token chunks keep them on chip


def test_report_single_chunk_cases_unchanged_and_multi_chunk_costlier():
    # A (288 GiB → budget 16384 ≥ 4096 + 64): one chunk per prompt → the 0.231/0.232 numbers
    q = disagg_report(_pd(), energy=_t(idle_W=20.0))["queue"]
    c = q["modes"]["coloc_chunked"]
    # 0.234: budget packing (up to 3–4 whole prompts per iteration) moved the TTFT mean 42.0556 → 42.4804; energy unchanged
    # 0.235: the (b+1)-th prompt's head in the leftover budget (b = 3, P(head) tiny) → 42.4832; p50/p90/p99 unchanged
    # 0.236: the head's work moves into the head batch and the next batch shrinks (D_b two-point) → 42.4807
    # 0.242: TTFT bins follow the queue-weighted batch law (the fused share grows with k even with one chunk per
    # prompt — the decodes ride in the iteration) → 42.5230; energy unchanged
    # 0.298: preset pJ_mac += 32b FP accumulate 0.9 / 2.6 pJ → J/token 0.20190 → 0.22584 (= old + MAC count × 0.9/2.6 pJ)
    eq = q["energy"]["coloc_chunked"]
    # 0.333: preset HBM 3.44 → 5.80 pJ/bit (idle_W set here, so no HBM standby) → 0.22584 → 0.28948 = old + DRAM bytes ×
    # 8 × (5.80173 − 3.44) pJ
    from accel_dse.core.energy import ENERGY_PRESETS
    dpj = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["pJ_bit_dram_hbm"][0] - 3.44
    assert abs(c["ttft_ms"]["mean"] - 42.5230) < 1e-3 and abs(eq["J_per_token"] - 0.28948) < 2e-5
    assert abs(eq["J_per_token"] - eq["counts_per_token"]["dram"] * 8 * dpj * 1e-12 - 0.22584) < 2e-5
    assert abs(eq["J_per_token"] - eq["counts_per_token"]["dram"] * 8 * dpj * 1e-12
               - eq["counts_per_token"]["mac"] * 0.9 / 2.6 * 1e-12 - 0.20190) < 2e-5
    # D (48 GB → budget 2048): 2–3 chunks; a chunk with its prefix costs more than its token share of the prompt
    d = disagg_report(_pd(cap=48.0))["queue"]["modes"]["coloc_chunked"]
    assert d["stable"] and d["ttft_ms"]["mean"] > c["ttft_ms"]["mean"]


def test_des_parity_multi_chunk():
    s = _pd(cap=48.0)            # vllm budget 2048 − k → 2–3 fused chunks per 4096-token prompt
    r = compare(s, n_req=800, warmup=200, seeds=(11, 12), modes=("coloc_chunked",))
    e = r["modes"]["coloc_chunked"]["err"]
    for k in ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_mean", "tpot_p90", "itl_max"):
        assert abs(e[k]) <= V4_BANDS[k], (k, e)


def test_fuzz_fused_chunks():
    rng = random.Random(233)
    for i in range(4):
        prof = ("vllm", "legacy_512", "sglang_mixed", "fixed")[i]
        kw = {"chunk_profile": prof}
        if prof == "fixed":
            kw["chunk_tokens"] = rng.choice((700, 1500, 3000))
        if prof == "sglang_mixed":
            kw["chunk_budget"] = rng.choice((1024, 2048, 3000))
        prompt = rng.choice((2048, 4096, 6000))
        s = _pd(prompt=prompt, out=256, batch=rng.choice((16, 32)), rate_rps=rng.uniform(2, 5), cap=48.0, **kw)
        q = disagg_report(s, energy=_t(idle_W=rng.uniform(0, 30)))["queue"]
        x = q["modes"]["coloc_chunked"]
        assert q["chunk_tokens"] >= 0
        if not x["stable"]:
            continue
        # the longest gap is at least one fused chunk iteration and the chunk count matches the plan
        C = x["chunk_tokens"]
        assert x["itl_max_ms"] >= x["decode"]["iter_ms_no_chunk"] - 1e-9
        assert abs(x["prefill"]["chunks"] - _chunk_plan(prompt, 0, C)[0]) <= 1
        lam, o = q["lambda_rps"], s.serving.out_len
        for m, en in q["energy"].items():
            W = en["J_per_token"] / en["card_s_per_token"]
            assert 0 < W < 2000 and abs(en["J_per_token"] * lam * o / 8 / W - 1) < 1e-9, m


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
