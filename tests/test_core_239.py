"""0.239: the wait-correlated survival's single-service (M/D/1) wait used Erlang's alternating series out to 400 τ;
past ≈ 6 τ it loses all precision at high ρ (P(W > 40τ) = 1.0 at ρ = 0.95), which broke the fused TTFT quantiles
near saturation (case E: p50 > mean·20, erratic SLO rates — the 0.238 KPI shift).  Exact only within MD1_EXACT_SPAN
τ, Cramér–Lundberg tail beyond, as mg1_mix_sum_quantile."""
import math

from accel_dse.core.queueing import _cl_params, mg1_corr_sum_quantile, mg1_mix_sum_quantile, md1_cdf


def test_md1_series_breaks_beyond_span_and_corr_tail_is_sane():
    lam, tau = 9.5, 0.1                                  # ρ = 0.95
    rho, th, c = _cl_params(lam, [(1.0, tau)])
    cl40 = c * math.exp(-th * 40 * tau)                  # the asymptotically exact tail at 40 τ
    assert 1e-3 < cl40 < 0.5
    assert abs((1 - md1_cdf(lam, tau, 40 * tau)) - cl40) > 0.5 * cl40     # the series has no correct digit left
    rows = [(1.0, 0.05, 0.05, math.inf, 0, False)]
    qs = [mg1_corr_sum_quantile(lam, [(1.0, tau)], rows, q) for q in (0.5, 0.9, 0.99)]
    ref = [mg1_mix_sum_quantile(lam, [(1.0, [(1.0, tau, 0.05)])], q) for q in (0.5, 0.9, 0.99)]
    assert qs[0] < qs[1] < qs[2]
    for a, b in zip(qs, ref):
        assert abs(a - b) / b < 0.02, (qs, ref)


def test_case_e_quantiles_monotone_near_saturation():
    from accel_dse.core import pdqueue, pdsim
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.parallel import Layout
    from accel_dse.core.scenario import PDConfig, Scenario, Serving
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", mem_cap_GB=48.0,
                 serving=Serving(batch=32, prompt=16384, out_len=256),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile="sglang_mixed"))
    rep, ctx = pdsim.capture_ctx(s)
    prev = 0.0
    for lam in (15.0, 17.0, 17.37, 18.5, 19.32):
        t = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)["ttft"]
        assert t["p50"] < 3 * t["mean"] and t["p50"] <= t["p90"] <= t["p99"], (lam, t)
        assert t["p90"] > prev, (lam, t)
        prev = t["p90"]
    # 0.238 had 19.76 → 17.37 here (numerical artefact); 0.239 fixed value 19.76 (0.240: 19.79, TTFT-bound);
    # 0.241 slot-cap blocking: 19.23; 0.242 queue-weighted batch law + block: 18.75 (DES ≈ 18.43)
    c = rep["queue"]["modes"]["coloc_chunked"]
    assert abs(c["slo_rate_rps"] - 18.75) < 0.05, c["slo_rate_rps"]
