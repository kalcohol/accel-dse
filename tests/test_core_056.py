"""0.56 — strict scenario typing (400 not 500), KV admission order (vLLM: admit before prefill, wait in TTFT / SLO
goodput), preemption mixture into the TPOT / worst-gap tails, swap preemption over the host link, 15 length bins,
per-pool static power in the PD energy."""

from __future__ import annotations

import dataclasses

from accel_dse.api import ApiError, api_eval, api_layouts
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.lengths import BINS
from accel_dse.core.pdsim import capture_ctx, simulate
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import PDConfig
from accel_dse.core.validation import v4_scenario


def _kv(scn, pol, gb, **kw):
    return dataclasses.replace(scn, pd=dataclasses.replace(scn.pd, kv_policy=pol, kv_capacity_GB=gb, **kw))


def _bad(body, frag):
    try:
        api_eval(body)
    except ApiError as e:
        assert frag in str(e), (frag, str(e))
        return
    raise AssertionError(f"expected ApiError for {body}")


def test_scenario_strict_types():
    base = {"model": "qwen3-8b", "serving": {"batch": 8, "prompt": 512, "out_len": 64}}
    a = api_eval({"scenario": {**base, "chip": "1P"}})                       # §25: chip as a preset name → 200
    b = api_eval({"scenario": base, "chip_preset": "1P"})
    assert a["summary"] == b["summary"]
    c = api_eval({"scenario": {**base, "serving": {**base["serving"], "batch": 8.0}}, "chip_preset": "1P"})
    assert c["summary"] == b["summary"]                                       # integral float accepted for an int
    _bad({"scenario": {**base, "chip": "nope"}}, "chip")
    _bad({"scenario": {**base, "serving": {**base["serving"], "batch": 2.5}}}, "Serving.batch")
    _bad({"scenario": {**base, "serving": {**base["serving"], "prompt": 1e300}}}, "prompt")
    _bad({"scenario": {**base, "serving": {**base["serving"], "batch": "8"}}}, "Serving.batch")
    _bad({"scenario": {**base, "model": 5}}, "model")
    _bad({"scenario": {**base, "serving": [1, 2]}}, "serving")
    _bad({"scenario": {**base, "pd": {"enabled": True, "load": "x"}}}, "load")
    _bad({"scenario": base, "chip_preset": ["1P"]}, "chip_preset")


def test_pd_scenario_through_layout_search():
    sc = {"model": "qwen3-8b", "serving": {"batch": 8, "prompt": 512, "out_len": 64},
          "pd": {"enabled": True, "prefill_cards": 1, "decode_cards": 2}}
    r = api_layouts({"scenario": sc, "chip_preset": "1P", "cards": 2})        # 0.55: ValueError from pd.decode_cards
    assert r["rows"]


def test_bins_avoid_reported_quantiles():
    assert BINS == 15 and all(abs(q * BINS - round(q * BINS)) > 1e-9 for q in (0.5, 0.9, 0.99))


def test_kv_admit_order_and_fold():
    base = v4_scenario(0.85, 0.0, False, "dense8b")
    off = disagg_report(base)["queue"]["modes"]["pd"]
    off2 = disagg_report(_kv(base, "off", None, kv_admit="after_prefill"))["queue"]["modes"]["pd"]
    assert off["ttft_ms"] == off2["ttft_ms"] and "kv_cap" not in off2           # kv_admit is inert when off
    bef = disagg_report(_kv(base, "wait", 12.0))["queue"]["modes"]["pd"]
    aft = disagg_report(_kv(base, "wait", 12.0, kv_admit="after_prefill"))["queue"]["modes"]["pd"]
    cb, ca = bef["kv_cap"], aft["kv_cap"]
    assert cb["admit"] == "before_prefill" and cb["binds"] and cb["in_ttft"]
    assert ca["admit"] == "after_prefill" and not ca["in_ttft"]
    assert bef["ttft_ms"]["p90"] > aft["ttft_ms"]["p90"] and bef["ttft_ms"]["mean"] > aft["ttft_ms"]["mean"]
    assert 0 < cb["p_wait"] <= 1 and cb["slot_wait_mean_ms"] > 0
    assert bef["slo_rate_rps"] <= aft["slo_rate_rps"] * 1.001                  # wait folded → goodput can only drop
    try:
        PDConfig(kv_admit="sometimes")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_preemption_mixture_tails():
    base = v4_scenario(0.85, 0.0, False, "dense8b")
    off = disagg_report(base)["queue"]["modes"]["pd"]
    rc = disagg_report(_kv(base, "recompute", 12.0))["queue"]["modes"]["pd"]
    c = rc["kv_cap"]
    assert c["preempt_sat"] > 0 and c["preempt_per_req"] > c["preempt_rice"]
    assert c["victim_gap_mean_ms"] > c["recompute_ms"] > 0 and 0 < c["restore_share"] < 1
    assert rc["itl_max_ms"] > off["itl_max_ms"] and rc["tpot_p99_ms"] >= rc["tpot_p90_ms"]


def test_swap_closed_form_and_des():
    base = v4_scenario(0.85, 0.0, False, "dense8b")
    d = disagg_report(_kv(base, "swap", 12.0))["queue"]["modes"]["pd"]["kv_cap"]
    assert d["policy"] == "swap" and d["swap_source"] == "workload.host_GBps" and d["swap_GBps_card"] > 0
    s25 = disagg_report(_kv(base, "swap", 12.0, swap_GBps=d["swap_GBps_card"] / 2))["queue"]["modes"]["pd"]["kv_cap"]
    assert s25["swap_source"] == "pd.swap_GBps" and abs(s25["swap_ms"] / d["swap_ms"] - 2) < 1e-6
    for bad in (0.0, -1.0, 1e6, True):
        try:
            PDConfig(kv_policy="swap", swap_GBps=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    rep, ctx = capture_ctx(_kv(base, "swap", 12.0))
    s = simulate(ctx, rep["queue"]["lambda_rps"], "pd", n_req=500, warmup=120, seed=3)
    assert s["complete"] and s["preempt_per_req"] >= 0 and ctx["swap"]["Bps_replica"] > 0


def test_des_admission_orders():
    base = v4_scenario(0.85, 0.0, False, "dense8b")
    rep, ctx = capture_ctx(_kv(base, "wait", 12.0))
    lam = rep["queue"]["lambda_rps"]
    sb = simulate(ctx, lam, "pd", n_req=500, warmup=120, seed=5)
    sa = simulate(ctx | {"kv_admit": "after_prefill"}, lam, "pd", n_req=500, warmup=120, seed=5)
    assert sb["complete"] and sa["complete"] and sb["slot_wait"]["mean"] > 0 and sa["slot_wait"]["mean"] > 0
    assert sb["ttft_ms"]["p90"] > sa["ttft_ms"]["p90"]                          # vLLM order: the wait sits in TTFT
    o1 = simulate(ctx | {"kv_policy": "off"}, lam, "coloc_chunked", n_req=300, warmup=80, seed=5)
    o2 = simulate(ctx | {"kv_policy": "off", "kv_admit": "after_prefill"}, lam, "coloc_chunked", n_req=300, warmup=80, seed=5)
    assert o1["ttft_ms"] == o2["ttft_ms"] and "slot_wait" not in o1


def test_static_power_per_pool():
    s = v4_scenario(0.6, 0.0, False, "dense8b")
    t1 = EnergyTable(pJ_mac=0.5, idle_W=100.0)
    t2 = EnergyTable(pJ_mac=0.5, idle_W=100.0, idle_W_prefill=300.0)
    e1 = disagg_report(s, energy=t1)["queue"]["energy"]
    e2 = disagg_report(s, energy=t2)["queue"]["energy"]
    rep, ctx = capture_ctx(s)
    lam, o = rep["queue"]["lambda_rps"], ctx["out"]
    p_cs = ctx["n_p"] / lam / o
    assert set(e1["pd"]["J_by_action"]) == {"mac", "idle"}                     # no split without idle_W_prefill (0.51)
    assert abs(e2["pd"]["J_by_action"]["idle_prefill"] - 300.0 * p_cs) < 1e-9 * 300 * p_cs
    assert abs(e2["pd"]["J_per_token"] - e1["pd"]["J_per_token"] - 200.0 * p_cs) < 1e-9 * e1["pd"]["J_per_token"] + 1e-12
    assert e2["coloc_chunked"] == e1["coloc_chunked"]                           # colocated cards all run the decode chip
    assert abs(e1["pd"]["tok_per_J"] * e1["pd"]["J_per_token"] - 1) < 1e-12 and 0 < e1["pd"]["static_share"] < 1
    e3 = disagg_report(s, energy=EnergyTable(idle_W_prefill=300.0))["queue"]["energy"]
    assert "static_note" in e3["pd"] and "static_note" in e3["coloc_chunked"]
    assert "idle_W_prefill" not in energy_report(evaluate(s), EnergyTable(pJ_mac=0.5))["missing"]


def test_exact_mg1_wait_law():
    """P-K wait law: mean = λE[S²]/(2(1 − ρ)) (to grid accuracy), M/D/1 reproduced, P(W > 0+) = ρ (0.56)."""
    from accel_dse.core.queueing import _pk_surv, md1_cdf, mg1
    pts = [(0.4, 0.25), (0.4, 1.0), (0.2, 2.5)]
    for rho in (0.3, 0.5, 0.85):
        sv, x_end, th = _pk_surv(rho, pts)                 # E[S] = 1 → λ = ρ
        m2 = sum(w * t * t for w, t in pts)
        n, hx = 40000, 40 * x_end / 40000
        mean = sum(sv((i + 0.5) * hx) for i in range(n)) * hx
        assert abs(mean / (rho * m2 / (2 * (1 - rho))) - 1) < 1e-3, (rho, mean)
        assert abs(sv(1e-9) - rho) < 1e-6 and sv(-1) == 1.0
        q = mg1(rho, [t for _, t in pts], [w for w, _ in pts])
        assert abs(sv(q["p99"]) - 0.01) < 1e-6
    sv, _, _ = _pk_surv(0.8, [(1.0, 1.0)])
    assert max(abs(1 - sv(x) - md1_cdf(0.8, 1.0, x)) for x in (0.3, 1.0, 2.5, 4.0, 5.9)) < 1e-3
