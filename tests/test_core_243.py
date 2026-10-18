"""0.243: (1) the fused decode fixed point oscillated near the decode-capacity limit (A vllm λ56.8: seen batch
57.2 → 63.2 → 52.3) and the TTFT law fell from 31 s to 0.14 s — a false feasible window that set the SLO rate to the
stability limit; the oscillating case is now bisected.  (2) The slot block clusters on the full-batch bins, spread
over the m = B − J* survivor deaths of the cycle.  DES SLO rates (TTFT p90 2 s crossing, 8 seeds n4000):
A vllm 55.95, G vllm 49.42, E sglang_mixed 18.43."""
from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"


def _ctx(prompt, prof, cap=None, batch=64, out=512):
    kw = {"mem_cap_GB": cap} if cap else {}
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id=HB, **kw,
                 serving=Serving(batch=batch, prompt=prompt, out_len=out),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile=prof))
    return pdsim.capture_ctx(s)


def test_fused_ttft_monotone_near_capacity_and_slo_rate_vs_des():
    rep, ctx = _ctx(4096, "vllm")
    prev = 0.0
    for lam in (54.0, 55.0, 55.5, 56.0, 56.5, 56.8, 57.0):
        r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
        if not r["stable"]:
            break
        assert r["ttft"]["p90"] >= prev, (lam, r["ttft"]["p90"], prev)     # 0.242: 56.5 → 31 s, 56.8 → 0.14 s
        prev = r["ttft"]["p90"]
    rate = rep["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    # 0.245: long-run DES (6 seeds × n64000) 55.28 — the n4000 reference 55.95 was transient near saturation
    assert abs(rate - 55.28) / 55.28 < 0.015, rate                          # 0.242: 57.00, 0.244: 56.20


def test_g_vllm_slo_rate_vs_des():
    rep, _ = _ctx(5000, "vllm")
    rate = rep["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    assert abs(rate - 49.02) / 49.02 < 0.015, rate                          # long-run DES (n4000: 49.42); 0.244: 49.99


def test_clustered_block_e_mixed():
    rep, ctx = _ctx(16384, "sglang_mixed", cap=48.0, batch=32, out=256)
    r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 18.6)
    # 0.246: E mixed is budget-chunked → the slot account (was the 0.243 clustered block; phi 0 < φ ≤ 1)
    assert r.get("slot_block") is None
    sb = r["slot_backlog"]
    assert 0 < sb["p_sat"] < 1 and sb["beta"] > 0
    # 0.245: long-run DES (6 seeds × n64000) TTFT p90 at 18.6 ≈ 2015 ms (18.483 → 1902, 18.668 → 2081; n4000: 2163)
    assert 0.85 * 2015 < r["ttft"]["p90"] * 1e3 < 1.15 * 2015                # 0.242 1856, 0.243 1935
    rate = rep["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    assert abs(rate - 18.59) / 18.59 < 0.015, rate    # long-run DES 18.59 (n4000 18.43); 0.242 18.75, 0.245 18.68, 0.246 18.73
