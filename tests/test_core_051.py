"""0.51 — PD queueing / tail latency, continuous batching, chunked prefill, KV contention, PD energy (all 「假设」)."""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import random

from accel_dse import api, cli
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, action_counts
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _coloc_chunked, _fused_step, _pd_mode, _Pool
from accel_dse.core.queueing import erlang_c, md1, md1_theta, mdc_wait, poisson_quantile
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import PDConfig, Scenario, Serving

BASE = dict(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=2),
            serving=Serving(batch=64, prompt=4096, out_len=512))
PD = PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _scn(**pd):
    return Scenario(**BASE, pd=PDConfig(**{**PD.__dict__, **pd}))


_CACHE: dict = {}


def _rep(**pd):
    k = tuple(sorted(pd.items()))
    if k not in _CACHE:
        _CACHE[k] = disagg_report(_scn(**pd))
    return _CACHE[k]


# ------------------------------------------------------------------ closed-form queueing helpers
def _lindley(rho, n=200_000, seed=7):
    rnd, w, ws = random.Random(seed), 0.0, []
    for _ in range(n):
        ws.append(w)
        w = max(0.0, w + 1.0 - rnd.expovariate(rho))      # τ = 1, interarrival ~ Exp(λ = ρ)
    ws.sort()
    return sum(ws) / n, ws[int(0.9 * n)], ws[int(0.99 * n)]


def test_md1_matches_lindley_simulation():
    for rho in (0.5, 0.8, 0.9):
        x = md1_theta(rho)
        assert abs(rho * math.expm1(x) - x) < 1e-9 and x > 0
        q = md1(rho, 1.0)
        assert q["stable"] and _close(q["mean"], rho / (2 * (1 - rho))) and q["p_wait"] == rho
        m, p90, p99 = _lindley(rho)
        assert abs(q["mean"] - m) / m < 0.08
        assert abs(q["p90"] - p90) / p90 < 0.08 and abs(q["p99"] - p99) / p99 < 0.08
        assert q["p50"] <= q["p90"] <= q["p99"]
    assert md1(0.05, 1.0)["p90"] == 0.0                    # ρ ≤ 1 − q → quantile 0
    assert not md1(1.0, 1.0)["stable"] and md1(1.0, 1.0)["p99"] == math.inf
    assert md1(0.0, 1.0)["mean"] == 0.0


def test_erlang_c_mdc_and_poisson():
    assert _close(erlang_c(1, 0.5), 0.5) and _close(erlang_c(2, 1.0), 1 / 3)
    assert erlang_c(4, 0) == 0.0 and erlang_c(4, 4.0) == 1.0
    w = mdc_wait(1.0, 1.5, 2)                             # a = 1.5, c = 2
    assert _close(w["p_wait"], erlang_c(2, 1.5)) and _close(w["mean"], w["p_wait"] * 0.5 / (2 / 1.5 - 1.0))
    assert not mdc_wait(10.0, 1.0, 4)["stable"]
    assert poisson_quantile(10, 0.9) == 14 and poisson_quantile(10, 0.5) == 10 and poisson_quantile(0, 0.9) == 0
    assert poisson_quantile(1000, 0.5) == 1000 and 1035 <= poisson_quantile(1000, 0.9) <= 1045


# ------------------------------------------------------------------ report
def test_queue_present_only_with_pd_and_default_untouched():
    assert Scenario().pd.load == 0.8 and Scenario().pd.rate_rps is None and Scenario().pd.chunk_tokens == 512
    assert "pd" not in api.api_eval({"scenario": {"model": "qwen3-8b"}})
    q = _rep()["queue"]
    assert set(q["modes"]) == {"pd", "coloc_prefill_first", "coloc_chunked"} and "假设" in q["basis"]
    assert _close(q["lambda_rps"], 0.8 * _rep()["req_s"]) and q["load"] == 0.8


def test_pd_mode_consistency():
    r = _rep()
    x = r["queue"]["modes"]["pd"]
    assert x["stable"]
    t = x["ttft_ms"]
    assert t["p50"] <= t["p90"] <= t["p99"] and t["mean"] >= x["prefill"]["ttft_b_ms"]
    d = x["decode"]
    lam_d = x["lambda_rps"] / r["decode"]["replicas"]
    assert _close(d["occupancy"], lam_d * 512 * x["tpot_mean_ms"] / 1e3, 1e-4)         # Little's law (birth–death)
    assert d["occupancy"] <= d["running_batch"] <= d["slots"] and d["running_p90"] >= d["running_batch"]
    assert x["tpot_mean_ms"] <= x["tpot_p90_ms"] <= x["tpot_p99_ms"]
    # at partial load the running batch is below the configured batch → TPOT below the fluid (full-batch) TPOT
    assert x["tpot_mean_ms"] < r["tpot_ms"]
    assert 0 <= x["kv"]["u_coll"] < 1 and x["kv"]["GBps_req_avail"] > 0


def test_load_and_rate_override_monotone():
    lo, hi = _rep(load=0.5)["queue"], _rep(load=0.9)["queue"]
    for m in ("pd", "coloc_prefill_first"):
        assert lo["modes"][m]["ttft_ms"]["p90"] < hi["modes"][m]["ttft_ms"]["p90"]
        assert lo["modes"][m]["tpot_mean_ms"] <= hi["modes"][m]["tpot_mean_ms"]
    r = _rep(rate_rps=2.0)["queue"]
    assert r["lambda_rps"] == 2.0 and r["load"] is None
    big = _rep(rate_rps=1e4)["queue"]["modes"]
    assert all(not v["stable"] and v["why"] for v in big.values())


def test_kv_contention_and_override():
    base = _rep()["queue"]["modes"]["pd"]
    k = base["kv"]
    assert k["shared_tier"] and 0 < k["u_coll"] < 1 and k["u_kv_decode"] > 0
    # the KV transfer gets what the pools' own collectives leave on the shared tier (TP2 pair → 2 × 400 GB/s)
    assert _close(k["GBps_req_avail"], 2 * 400.0 * (1 - k["u_coll"]), 1e-9)
    # a dedicated KV path (pd.kv_GBps) does not compete with the collectives
    ded = _rep(kv_GBps=400.0)["queue"]["modes"]["pd"]
    assert not ded["kv"]["shared_tier"] and ded["kv"]["u_coll"] == 0 and _close(ded["kv"]["GBps_req_avail"], 800.0)
    assert ded["kv"]["exposed_ms"] < k["exposed_ms"] and ded["ttft_ms"]["p90"] <= base["ttft_ms"]["p90"]
    slow = _rep(kv_GBps=20.0)["queue"]["modes"]["pd"]
    assert slow["stable"] and slow["kv"]["exposed_ms"] > k["exposed_ms"] and slow["ttft_ms"]["p90"] > base["ttft_ms"]["p90"]
    # contention: the collective-heavy prefill on a slow shared tier is slower than on a dedicated path
    sh = disagg_report(_scn().replace("link.GBps", 50.0))["queue"]["modes"]["pd"]
    if sh["stable"]:
        assert sh["kv"]["u_kv_prefill"] > 0 and sh["prefill"]["kv_slowdown"] >= 1.0


def test_chunked_prefill_fused_iteration():
    s = _scn()
    pool = _Pool(s)
    dec, pre1 = pool.run("decode", 8), pool.run("prefill", 1)
    assert _fused_step(dec, pre1, 0.0, 0.0) == dec.step
    a, b = _fused_step(dec, pre1, 0.125, 0.0), _fused_step(dec, pre1, 0.25, 0.0)
    assert dec.step < a <= b
    assert _fused_step(dec, pre1, 0.125, 0.4) >= a                   # prefix KV re-read only adds traffic
    c = _rep()["queue"]["modes"]["coloc_chunked"]
    assert c["stable"] and c["prefill"]["chunks"] == 8 and c["prefill"]["iter_ms"] >= c["decode"]["iter_ms_no_chunk"]
    assert c["ttft_ms"]["p50"] >= c["prefill"]["chunks"] * c["prefill"]["iter_ms"] - 1e-9
    # chunking bounds the worst inter-token gap far below the prefill-first stall
    pf = _rep()["queue"]["modes"]["coloc_prefill_first"]
    assert c["itl_max_ms"] < pf["itl_max_ms"] and pf["itl_max_ms"] >= pf["stall_ms"]
    big = _rep(chunk_tokens=4096)["queue"]["modes"]["coloc_chunked"]
    assert big["prefill"]["chunks"] == 1 and big["itl_max_ms"] > c["itl_max_ms"]


def test_slo_rate_and_split_search():
    s = Scenario(**{**BASE, "serving": Serving(batch=64, prompt=4096, out_len=512, ttft_slo_ms=400, tpot_slo_ms=10)},
                 pd=PD)
    r = disagg_report(s)
    q = r["queue"]
    ctx_note = q["modes"]["pd"]
    rate = ctx_note["slo_rate_rps"]
    assert rate > 0 and _close(ctx_note["slo_goodput_per_card"], rate * 512 / 8)
    # re-evaluate the PD mode just below / above the reported rate
    sp = Scenario(**{**BASE, "serving": s.serving}, pd=PDConfig(**{**PD.__dict__, "rate_rps": rate * 0.999}))
    xs = disagg_report(sp)["queue"]["modes"]["pd"]
    assert xs["stable"] and xs["ttft_ms"]["p90"] <= 400 + 1e-6 and xs["tpot_p90_ms"] <= 10 + 1e-6
    sp = Scenario(**{**BASE, "serving": s.serving}, pd=PDConfig(**{**PD.__dict__, "rate_rps": rate * 1.05}))
    xs = disagg_report(sp)["queue"]["modes"]["pd"]
    assert not xs["stable"] or xs["ttft_ms"]["p90"] > 400 or xs["tpot_p90_ms"] > 10
    b = q["pd_slo_best_split"]
    assert b["prefill_cards"] + b["decode_cards"] == 8 and b["slo_rate_rps"] >= rate - 1e-9
    assert {(x["prefill_cards"], x["decode_cards"]) for x in q["pd_slo_splits"]} >= {(2, 6), (4, 4), (6, 2)}
    # no SLO → the stable limit ≈ the fluid capacity (queueing only adds latency, never capacity)
    q0 = _rep()["queue"]["modes"]["pd"]
    assert q0["slo_rate_rps"] <= _rep()["req_s"] * 1.0001 and q0["slo_rate_rps"] > 0.9 * _rep()["req_s"]


def test_pd_energy_from_action_counts():
    r = _rep()
    q = r["queue"]
    x = q["energy"]["pd"]
    s = _scn()
    pool_p, pool_d = _Pool(s), _Pool(s)
    mp = q["modes"]["pd"]
    pre = pool_p.run("prefill", mp["prefill"]["batch_cap"])
    dec = pool_d.run("decode", mp["decode"]["running_batch"])
    ap, ad = action_counts(pre), action_counts(dec)
    mac = (ap["counts"]["mac"] / ap["units"] * 4096 + ad["counts"]["mac"] / ad["units"] * 512) / 512
    assert _close(x["counts_per_token"]["mac"], mac, 1e-9)
    assert _close(x["card_s_per_token"], 8 / q["lambda_rps"] / 512)
    assert "J_per_token" not in x
    # chunked prefill shares the decode iteration's weight read → no more DRAM per token than prefill-first + re-read
    e = q["energy"]
    assert e["coloc_chunked"]["counts_per_token"]["dram"] < e["coloc_prefill_first"]["counts_per_token"]["dram"]
    t = EnergyTable(pJ_mac=0.5, pJ_bit_dram=4.0, idle_W=50.0)
    rr = disagg_report(s, energy=t)["queue"]["energy"]["pd"]
    want = rr["counts_per_token"]["mac"] * 0.5e-12 + rr["counts_per_token"]["dram"] * 8 * 4e-12 \
        + 50.0 * rr["card_s_per_token"]
    assert _close(rr["J_per_token"], want, 1e-9) and set(rr["J_by_action"]) == {"mac", "dram", "idle"}


def test_validation_api_cli():
    for bad in (dict(load=0.0), dict(load=1.0), dict(load=True), dict(rate_rps=0.0), dict(rate_rps=-1.0),
                dict(chunk_tokens=8), dict(chunk_tokens=64.5)):
        try:
            PDConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    out = api.api_eval({"scenario": {"model": "qwen3-8b", "serving": {"batch": 16},
                                     "pd": {"enabled": True, "decode_cards": 3, "load": 0.5, "chunk_tokens": 256}},
                        "energy": {"pJ_mac": 0.3}})
    q = out["pd"]["queue"]
    assert q["load"] == 0.5 and q["chunk_tokens"] == 256 and "J_per_token" in q["energy"]["pd"]
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-load", "0.6", "--pd-rate", "3", "--pd-chunk", "1024",
                      "--out-len", "128", "--ttft-slo", "500"])
    b = cli._body(a)["scenario"]
    assert b["pd"]["load"] == 0.6 and b["pd"]["rate_rps"] == 3 and b["pd"]["chunk_tokens"] == 1024
    assert b["serving"]["out_len"] == 128 and b["serving"]["ttft_slo_ms"] == 500
    a = p.parse_args(["--model", "qwen3-8b", "--pd"])
    assert set(cli._body(a)["scenario"]["pd"]) == {"enabled", "prefill_layout", "prefill_cards", "decode_cards",
                                                    "kv_GBps", "kv_layerwise"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["eval", "--model", "qwen3-8b", "--batch", "16", "--pd", "--pd-decode-cards", "3"]) == 0
    assert "queueing" in buf.getvalue() and "prefill-first" in buf.getvalue()
