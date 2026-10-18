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
    assert abs(rate - 55.95) / 55.95 < 0.015, rate                          # 0.242: 57.00 (= stability limit)


def test_g_vllm_slo_rate_vs_des():
    rep, _ = _ctx(5000, "vllm")
    rate = rep["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    assert abs(rate - 49.42) / 49.42 < 0.015, rate                          # 0.242: 50.72


def test_clustered_block_e_mixed():
    rep, ctx = _ctx(16384, "sglang_mixed", cap=48.0, batch=32, out=256)
    r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 18.6)
    sb = r["slot_block"]
    assert sb is not None and 0 < sb["phi"] <= 1 and sb["p"] <= sb["phi"]
    assert 1856 < r["ttft"]["p90"] * 1e3 < 2163                              # 0.242 1856, DES 2163
    rate = rep["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    assert 18.43 <= rate < 18.75, rate                                       # DES 18.43, 0.242 18.75
