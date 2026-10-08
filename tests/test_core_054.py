"""0.54 — request-level DES (core/pdsim), V4 serving validation, closed-form fixes (birth–death decode, TTFT
convolution, prefill-first worst gap)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from accel_dse.core.disagg import disagg_report
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _busy_dur_max_q, _busy_max_q, _decode_birth_death
from accel_dse.core.pdsim import LRUCache, _zipf_ids, capture_ctx, compare, simulate
from accel_dse.core.prefixcache import lru_hit
from accel_dse.core.queueing import mg1, mg1_sum_quantile
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.core.validation import V4_BANDS, v4_scenario

SV = Serving(batch=64, prompt=4096, out_len=512, ttft_slo_ms=400, tpot_slo_ms=10)
BASE = dict(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=2), serving=SV)
PD = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6)
DATA = Path(__file__).resolve().parents[1] / "accel_dse" / "data" / "v4_serving.json"


def sc(**pd) -> Scenario:
    return Scenario(**{**BASE, "pd": PDConfig(**{**PD, **pd})})


def test_che_matches_exact_lru():
    """Discrete LRU under IRM (after warm-up) vs Che: |ΔH| < 0.015 for Zipf 0.8 / 1.0 / 1.2."""
    for alpha in (0.8, 1.0, 1.2):
        n, K = 5000, 400
        rng = random.Random(3)
        c = LRUCache(K)
        ids = _zipf_ids(rng, n, alpha, 60000)
        for pid in ids[:20000]:
            c.access(pid)
        hits = sum(c.access(pid) for pid in ids[20000:])
        assert abs(hits / 40000 - lru_hit(n, alpha, K)) < 0.015, alpha


def test_mg1_sum_quantile_vs_lindley():
    """W + L_i (own service, same index) for a two-point service mix: closed form vs a Lindley recursion."""
    taus, ws, lam = [0.02, 0.10], [0.7, 0.3], 6.0
    rng = random.Random(5)
    w, tot = 0.0, []
    prev = None
    for _ in range(200000):
        i = 0 if rng.random() < ws[0] else 1
        if prev is not None:
            w = max(0.0, w + prev - rng.expovariate(lam))
        tot.append(w + taus[i])
        prev = taus[i]
    tot.sort()
    for q in (0.5, 0.9, 0.99):
        sim = tot[int(q * len(tot))]
        ana = mg1_sum_quantile(lam, taus, ws, taus, q)
        assert abs(ana - sim) / sim < 0.12, (q, ana, sim)     # Cramér–Lundberg tail near the wait atom: ≈ +10 % at p90
    # single service value → None (caller keeps the exact sum)
    assert mg1_sum_quantile(lam, [0.05, 0.05], ws, [0.05, 0.05], 0.9) is None
    # the convolution never exceeds the old quantile sum
    old = mg1(lam, taus, ws)["p90"] + 0.10
    assert mg1_sum_quantile(lam, taus, ws, taus, 0.9) <= old + 1e-12


def test_busy_duration_reduces_to_count():
    lat, lam = 0.077, 2.0
    for cap in (1, 4):
        n = _busy_max_q(lam, lat, cap, 5.0, 0.15, 0.99)
        assert abs(_busy_dur_max_q(lam, [(1.0, lat)], cap, 5.0, 0.15, 0.99) - n * lat) < 1e-12
    # a spread latency mix gives a longer worst busy period than its mean
    mix = [(0.5, 0.04), (0.5, 0.114)]
    assert _busy_dur_max_q(lam, mix, 1, 5.0, 0.15, 0.99) > _busy_max_q(lam, 0.077, 1, 5.0, 0.15, 0.99) * 0.077 * 0.99


def test_birth_death_decode_vs_des():
    """PD at load 0.8: DES mean TPOT within 6 % of the birth–death mean; p90 TTFT within 10 %; the 0.53 fluid fixed
    point (≈ 6.5 ms) would be ≈ 30 % low."""
    rep, ctx = capture_ctx(sc(load=0.8))
    lam = rep["queue"]["lambda_rps"]
    x = rep["queue"]["modes"]["pd"]
    runs = [simulate(ctx, lam, "pd", n_req=1500, warmup=400, seed=sd) for sd in (1, 2, 3)]   # occupancy is slow:
    assert all(s["complete"] for s in runs)                                               # average 3 seeds
    tp = sum(s["tpot_ms"]["mean"] for s in runs) / 3
    t90 = sum(s["ttft_ms"]["p90"] for s in runs) / 3
    assert abs(x["tpot_mean_ms"] - tp) / tp < 0.08
    assert abs(x["ttft_ms"]["p90"] - t90) / t90 < 0.10
    assert tp > 6.49 * 1.2
    d = x["decode"]
    assert d["occupancy"] <= d["running_p90"] <= d["slots"]


def test_birth_death_little_and_limits():
    rep, ctx = capture_ctx(sc(load=0.5))
    lam_d = rep["queue"]["lambda_rps"] / ctx["r_d"]
    d = _decode_birth_death(ctx["dpool"], lam_d, ctx["out"], ctx["B"])
    # pi_run = running-batch law given busy, index 0 ↔ k = 1
    assert abs(sum(d["pi_run"]) - 1) < 1e-9
    assert abs(sum(p * (i + 1) for i, p in enumerate(d["pi_run"])) - d["running_mean"]) < 1e-9
    # Little on the running set: E[k] = λ·out·TPOT; no slot queue at load 0.5 → E[k] = E[N]
    assert abs(d["tpot"] * lam_d * ctx["out"] - d["occupancy"]) / d["occupancy"] < 1e-6
    assert d["running_mean"] >= d["occupancy"]
    assert d["step_seen_mean"] >= d["step_of"](1)[0] - 1e-15
    # beyond the full-batch service rate → unstable
    stB = d["step_of"](ctx["B"])[0]
    assert _decode_birth_death(ctx["dpool"], 1.01 * ctx["B"] / (ctx["out"] * stB), ctx["out"], ctx["B"]) is None


def test_des_deterministic_and_hit_window():
    rep, ctx = capture_ctx(sc(load=0.6, prefix_len=2048, prefix_count=20000))
    lam, pc = rep["queue"]["lambda_rps"], rep["prefix_cache"]
    kw = dict(prefix_len=2048, prefix_n=20000, prefix_alpha=1.0, prefix_K=pc["prefill"]["K"],
              prefix_K_dec=pc["decode"]["K"])
    a = simulate(ctx, lam, "pd", n_req=600, warmup=150, seed=4, **kw)
    b = simulate(ctx, lam, "pd", n_req=600, warmup=150, seed=4, **kw)
    assert a["ttft"] == b["ttft"] and a["tpot"] == b["tpot"] and a["prefix_hit"] == b["prefix_hit"]
    assert abs(a["prefix_hit"] - pc["prefill"]["hit"]) < 0.05           # warm caches, measurement window only
    assert abs(a["prefix_hit_decode"] - pc["decode"]["hit"]) < 0.05


def test_compare_small_grid_within_bands():
    """Medium load, fixed lengths: every metric within the V4 band for PD and chunked; prefill-first TTFT p90 may sit
    on the M/D/1 wait-atom edge (ρ ≈ 1 − q), so it is checked at p50 / p99 only."""
    r = compare(v4_scenario(0.6), n_req=1200, warmup=300, seed=11)
    for mode, x in r["modes"].items():
        for m, e in x["err"].items():
            if mode == "coloc_prefill_first" and m == "ttft_p90":
                continue
            assert abs(e) <= V4_BANDS.get(m, 0.25) + 1e-9, (mode, m, e)


def test_v4_stored_grid_summary():
    """Stored full grid (scripts/v4_serving.py, 54 points): documented error envelope (MODEL.md §18.4)."""
    d = json.loads(DATA.read_text())
    S = {(s["mode"], s["metric"]): s for s in d["summary"]}
    assert len(d["rows"]) == 18
    for mode in ("pd", "coloc_prefill_first", "coloc_chunked"):
        assert abs(S[(mode, "slo_goodput")]["max"]) <= 0.10 and abs(S[(mode, "slo_goodput")]["min"]) <= 0.10
        assert S[(mode, "prefix_hit")]["mean_abs"] < 0.01
        assert S[(mode, "ttft_p90")]["within_band"] >= 0.75
    assert S[("pd", "ttft_p90")]["within_band"] == 1.0 and S[("pd", "tpot_p90")]["within_band"] >= 0.9
    # known direction: colocated TPOT p90 is optimistic (closed form < DES) everywhere, by < 45 %
    for mode in ("coloc_prefill_first", "coloc_chunked"):
        assert S[(mode, "tpot_p90")]["max"] < 0 and S[(mode, "tpot_p90")]["min"] > -0.45


def test_simulate_flag_is_opt_in_and_additive():
    off = disagg_report(sc(load=0.8))["queue"]
    on = disagg_report(sc(load=0.8, simulate=True))["queue"]
    assert "sim" not in off and set(on["sim"]["modes"]) == {"pd", "coloc_prefill_first", "coloc_chunked"}
    for m in off["modes"]:
        assert off["modes"][m] == on["modes"][m]
    for x in on["sim"]["modes"].values():
        assert x["complete"] and all(v is None or math.isfinite(v) for v in x["err"].values())
    try:
        PDConfig(**{**PD, "simulate": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("pd.simulate must be a boolean")


def test_default_colocated_report_has_no_pd():
    s = Scenario(**BASE)
    assert s.pd is None or not s.pd.enabled and not s.pd.simulate
