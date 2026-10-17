"""0.232: SGLang chunked-prefill profiles — ``sglang`` (default, no mixed chunk: prefill-first, a long prompt runs as
consecutive prefill-only chunks of chunked_prefill_size while the decodes pause; non-fused) and ``sglang_mixed``
(--enable-mixed-chunk = the vLLM fused rule with budget = chunked_prefill_size); tiers, chunk-pool view, DES parity,
decode-pause accounting, energy; fuzz."""
import random

from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import (CHUNK_PROFILES, _ChunkPool, _Pool, _per_unit, chunk_spec, sglang_chunk)
from accel_dse.core.pdsim import compare
from accel_dse.core.scenario import CHUNK_PROFILE_IDS, PDConfig, Scenario, Serving
from accel_dse.core.validation import V4_BANDS

HB = "hbm3e_8s_12h24g_9200"


def _pd(prompt=4096, out=512, batch=64, cap=0.0, **pd):
    kw = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6)
    kw.update(pd)
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2), mem_cap_GB=cap,
                    serving=Serving(batch=batch, prompt=prompt, out_len=out), pd=PDConfig(**kw))


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


_C: dict = {}


def _q(**kw):
    k = tuple(sorted(kw.items()))
    if k not in _C:
        _C[k] = disagg_report(_pd(**kw), energy=_t(idle_W=20.0))["queue"]
    return _C[k]


def test_tiers_and_spec():
    assert set(CHUNK_PROFILES) == set(CHUNK_PROFILE_IDS)
    # SGLang arg_groups/memory_hook.py (gpu_mem MiB): < 20 GiB 2048, < 35 4096, < 60 4096, < 90 8192, < 160 8192, else 16384
    g = (8, 19.9, 20, 34.9, 35, 59.9, 60, 89.9, 90, 159.9, 160, 288)
    assert [sglang_chunk(x) for x in g] == [2048, 2048, 4096, 4096, 4096, 4096, 8192, 8192, 8192, 8192, 16384, 16384]
    s = chunk_spec(PDConfig(chunk_profile="sglang"), 80.0)
    assert s["C"] == 8192 and s["nonfused"] and s["tag"] == "来源"
    m = chunk_spec(PDConfig(chunk_profile="sglang_mixed", chunk_budget=3000), 80.0)
    assert m["budget"] == 3000 and not m.get("nonfused") and m["tag"] == "用户"


def test_chunk_pool_view():
    pool = _Pool(_pd())
    v = _ChunkPool(pool, 1024)
    one = v.run("prefill", 1, 4096, 0)
    parts = [pool.run("prefill", 1, e, q) for q, e in ((0, 1024), (1024, 2048), (2048, 3072), (3072, 4096))]
    assert abs(one.ttft - sum(r.ttft for r in parts)) < 1e-15 and one.fits and len(one._chunks) == 4
    plain = pool.run("prefill", 1, 4096, 0)
    assert one.ttft > plain.ttft                     # every chunk re-reads the weights and the earlier KV
    assert v.run("prefill", 2, 4096, 0).fits is False                # a batch cannot pack more than C tokens
    assert v.run("prefill", 2, 512, 0) is pool.run("prefill", 2, 512, 0)     # 2 × 512 ≤ 1024: one packed iteration
    assert v.run("decode", 8) is pool.run("decode", 8)
    pu, pp = _per_unit(one), _per_unit(plain)
    assert pu["dram"] > pp["dram"] and abs(pu["mac"] / pp["mac"] - 1) < 0.05


def test_report_sglang_vs_prefill_first_and_mixed_vs_vllm():
    # 288 GiB card → chunked_prefill_size 16384 ≥ prompt: SGLang = prefill-first exactly
    q = _q(chunk_profile="sglang")
    x, pf = q["modes"]["coloc_chunked"], q["modes"]["coloc_prefill_first"]
    assert q["chunk_tokens"] == 16384 and x["ttft_ms"] == pf["ttft_ms"] and x["tpot_mean_ms"] == pf["tpot_mean_ms"]
    assert abs(q["energy"]["coloc_chunked"]["J_per_token"] - q["energy"]["coloc_prefill_first"]["J_per_token"]) < 1e-12
    # C = 1024 < prompt: 4 prefill-only chunks per prompt, slower prefill, longer decode pauses, more energy
    q1 = _q(chunk_profile="sglang", chunk_budget=1024)
    x1, pf1 = q1["modes"]["coloc_chunked"], q1["modes"]["coloc_prefill_first"]
    assert x1["stable"] and x1["prefill"]["chunks"] == 4 and x1["prefill"]["nonfused"]
    assert x1["ttft_ms"]["mean"] > pf1["ttft_ms"]["mean"] and x1["stall_ms"] > pf1["stall_ms"] / pf1["prefill"]["batch_cap"]
    assert x1["decode"]["share"] < pf1["decode"]["share"]
    assert q1["energy"]["coloc_chunked"]["J_per_token"] > q1["energy"]["coloc_prefill_first"]["J_per_token"]
    # --enable-mixed-chunk is the vLLM fused rule with the same budget
    a = _q(chunk_profile="sglang_mixed", chunk_budget=3000)["modes"]["coloc_chunked"]
    b = _q(chunk_profile="vllm", chunk_budget=3000)["modes"]["coloc_chunked"]
    assert a["ttft_ms"] == b["ttft_ms"] and a["tpot_mean_ms"] == b["tpot_mean_ms"]


def test_decode_pause_accounting():
    # the decodes pause for the whole chunked prefill: stall ≥ Σ chunk iterations of one prompt; decode share = 1 − ρ_prefill
    s = _pd(chunk_profile="sglang", chunk_budget=1024)
    q = _q(chunk_profile="sglang", chunk_budget=1024)
    x = q["modes"]["coloc_chunked"]
    pool = _Pool(s)
    tot = sum(pool.run("prefill", 1, e, q_).ttft for q_, e in ((0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)))
    assert x["prefill"]["batch_cap"] == 1 and abs(x["stall_ms"] - tot * 1e3) < 1e-6
    assert x["itl_max_ms"] >= x["stall_ms"]
    assert 0 < x["decode"]["share"] < 1


def test_des_parity():
    s = _pd(chunk_profile="sglang", chunk_budget=1024)
    r = compare(s, n_req=1000, warmup=250, seeds=(11, 12), modes=("coloc_chunked",))
    e = r["modes"]["coloc_chunked"]["err"]
    # V4 bands (itl_max left out: one prompt length puts the busy-period p99 on a lattice atom, MODEL §74)
    for k in ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_mean", "tpot_p90"):
        assert abs(e[k]) <= V4_BANDS[k], (k, e)


def test_fuzz_sglang():
    rng = random.Random(232)
    for i in range(4):
        prof = ("sglang", "sglang_mixed", "sglang", "sglang")[i]
        prompt = rng.choice((2048, 4096, 6144))
        batch = rng.choice((16, 32, 64))
        rate = rng.uniform(2.0, 6.0)
        prev = None
        for C in sorted(rng.sample((512, 1024, 2048, 4096, 8192), 3)):
            s = _pd(prompt=prompt, out=256, batch=batch, rate_rps=rate, chunk_profile=prof, chunk_budget=C)
            q = disagg_report(s, energy=_t(idle_W=rng.uniform(0, 30)))["queue"]
            x = q["modes"]["coloc_chunked"]
            assert q["chunk_tokens"] >= 0
            if not x["stable"]:
                continue
            if prof == "sglang":
                # TTFT never rises with a larger chunk (fewer prefill-only iterations, wider packing)
                if prev is not None:
                    assert x["ttft_ms"]["mean"] <= prev * (1 + 1e-9), (prompt, C, batch)
                prev = x["ttft_ms"]["mean"]
                assert x["itl_max_ms"] >= x["stall_ms"] - 1e-9
            lam, o = q["lambda_rps"], s.serving.out_len
            for m, en in q["energy"].items():
                W = en["J_per_token"] / en["card_s_per_token"]
                assert 0 < W < 2000 and abs(en["J_per_token"] * lam * o / 8 / W - 1) < 1e-9, m
