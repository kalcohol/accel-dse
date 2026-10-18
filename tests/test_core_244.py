"""0.244: SGLang non-mixed counts the prefilling requests in max_running_requests (scheduler._get_new_batch_prefill_raw:
the admission loop stops at len(can_run_list) >= max_running_requests − len(running_batch)), so in a backlog every
admission waits for a death: the admission server costs τ + β (β = out·T_d(B)/B), and the blocking is the backlog
itself (pdqueue._slot_backlog: diffusion of the slot account U = (τ + β)·q − β·f).  Long-run DES SLO rates
(TTFT p90 2 s crossing, 6 seeds × n64000 — the 0.243 n4000 runs are transient near saturation): A sglang 55.05,
D sglang 54.95, F sglang 42.66; sglang ranks last among the chunked engines in A/D/F (0.243: second)."""
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


def _rate(rep, mode="coloc_chunked"):
    return rep["queue"]["modes"][mode]["slo_rate_rps"]


def test_slot_backlog_a_monotone_and_mechanism():
    rep, ctx = _ctx(4096, "sglang")
    prev, prev_p = 0.0, 0.0
    for lam in (50.0, 52.0, 53.5, 54.3, 55.06, 55.7, 56.3):
        r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
        assert r["stable"], lam
        sb = r["prefill"]["slot_backlog"]
        assert 0 < sb["p_sat"] < 1 and sb["p_sat"] >= prev_p
        # admission server τ + β: β = out·T_d(B)/B, ρ′ = λ_c(τ + β) < 1
        assert abs(sb["rho_adm"] - lam / ctx["r_c"] * (sb["tau"] + sb["beta"])) < 1e-9
        assert r["ttft"]["p90"] >= prev, (lam, r["ttft"]["p90"], prev)
        prev, prev_p = r["ttft"]["p90"], sb["p_sat"]
    # long-run DES TTFT p90 (ms) 55.06 → 2019, 55.7 → 3344 (0.243 analytic: 97 ms at 55.7)
    r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 55.06)
    assert 1700 < r["ttft"]["p90"] * 1e3 < 2400
    # far from saturation the backlog branch vanishes
    r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 30.0)
    assert r["prefill"]["slot_backlog"]["p_sat"] < 1e-3


def test_sglang_slo_rates_vs_long_des_and_ranking():
    for prompt, cap, des, others in ((4096, None, 55.05, ("vllm", "vllm_v0")),
                                     (4096, 48.0, 54.95, ("vllm", "sglang_mixed")),
                                     (6000, 48.0, 42.66, ("vllm", "sglang_mixed"))):
        rep, _ = _ctx(prompt, "sglang", cap=cap)
        rate = _rate(rep)
        assert abs(rate - des) / des < 0.015, (prompt, cap, rate)      # 0.243: 56.73 / 56.69 / 44.36
        for prof in others:                                            # DES: sglang last among the chunked engines
            assert rate < _rate(_ctx(prompt, prof, cap=cap)[0]), (prompt, cap, prof)


def test_prefill_first_and_fused_untouched():
    _, ctx = _ctx(4096, "vllm")
    r = pdqueue._coloc_prefill_first({**ctx, "slo": (2.0, 0.05)}, 50.0)
    assert "slot_backlog" not in r.get("prefill", {})
    r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 50.0)
    assert "slot_backlog" not in (r.get("prefill") or {})
