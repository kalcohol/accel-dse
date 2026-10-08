"""0.52 — PD request-length spread, prefix cache, pool layout search, chunked-iteration mean fix (all 「假设」)."""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import random

from accel_dse import api, cli
from accel_dse.core.disagg import disagg_report, kv_bytes_per_request
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.lengths import BINS, lengths_of, lognormal_bins
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _Pool, _chunk_plan, prefix_tokens
from accel_dse.core.queueing import dquantile, md1, mdc_wait, mg1, erlang_c
from accel_dse.core.scenario import PDConfig, Scenario, Serving

SV = Serving(batch=64, prompt=4096, out_len=512, ttft_slo_ms=400, tpot_slo_ms=10)
BASE = dict(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=2), serving=SV)
PD = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6)
_CACHE: dict = {}


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _scn(**kw):
    return Scenario(**BASE, pd=PDConfig(**{**PD, **kw}))


def _rep(**kw):
    k = tuple(sorted(kw.items()))
    if k not in _CACHE:
        _CACHE[k] = disagg_report(_scn(**kw))
    return _CACHE[k]


# ------------------------------------------------------------------ lengths
def test_lognormal_bins_and_lengths():
    assert lognormal_bins(4096, 0) == [(1.0, 4096)]
    b = lognormal_bins(4096, 1.0)
    assert len(b) == BINS and all(_close(w, 1 / BINS) for w, _ in b)
    vals = [v for _, v in b]
    assert vals == sorted(vals) and abs(sum(w * v for w, v in b) - 4096) < BINS     # mean kept (up to rounding)
    sv = Serving(prompt=2048, out_len=256)
    fx = lengths_of(PDConfig(), sv)
    assert fx.trivial and fx.source == "fixed" and fx.ctx_ratio == 1.0 and fx.out_cs2 == 0.0
    cv = lengths_of(PDConfig(prompt_cv=0.5, out_cv=1.0), sv)
    assert len(cv.points) == BINS * BINS and _close(sum(w for w, _, _ in cv.points), 1.0)
    assert len(cv.prompts()) == BINS and 0.3 < math.sqrt(cv.prompt_cs2) < 0.5 and 0.7 < math.sqrt(cv.out_cs2) < 1.0
    # length-biased decode context: E[out·(S + out/2)] / E[out] over (E[S] + E[out]/2); independent S, out → only out
    m, m2 = cv.mean_out, sum(w * o * o for w, _, o in cv.points)
    assert _close(cv.ctx_ratio, (cv.mean_S + m2 / (2 * m)) / (cv.mean_S + m / 2)) and cv.ctx_ratio > 1
    mx = lengths_of(PDConfig(length_mix=((3.0, 1000, 100), (1.0, 5000, 900))), sv)
    assert mx.source == "mix" and _close(mx.mean_S, 2000) and _close(mx.mean_out, 300)
    assert _close(mx.ctx_ratio, (0.75 * 100 * 1050 + 0.25 * 900 * 5450) / 300 / (2000 + 150))


def test_mg1_matches_simulation_and_reduces_to_md1():
    assert mg1(0.6, [2.0], [1.0]) == md1(0.6, 2.0)
    for rho in (0.5, 0.8):
        q = mg1(rho, [0.25, 1.0, 2.5], [0.4, 0.4, 0.2])          # mean service 1.0
        assert _close(q["rho"], rho) and _close(q["mean"], rho * (0.4 * 0.0625 + 0.4 + 0.2 * 6.25) / (2 * (1 - rho)))
        rnd, w, ws = random.Random(5), 0.0, []
        for _ in range(300_000):
            ws.append(w)
            u = rnd.random()
            s = 0.25 if u < 0.4 else 1.0 if u < 0.8 else 2.5
            w = max(0.0, w + s - rnd.expovariate(rho))
        ws.sort()
        n = len(ws)
        assert abs(q["p90"] - ws[int(0.9 * n)]) / ws[int(0.9 * n)] < 0.08
        assert abs(q["p99"] - ws[int(0.99 * n)]) / ws[int(0.99 * n)] < 0.08
    assert not mg1(1.5, [0.5, 1.0], [0.5, 0.5])["stable"] and mg1(1.2, [0.5, 1.0], [0.5, 0.5])["stable"]
    assert dquantile([(0.5, 1.0), (0.4, 2.0), (0.1, 9.0)], 0.9) == 2.0 and dquantile([(1, 3.0)], 0.99) == 3.0
    a, b = mdc_wait(1.0, 1.5, 2), mdc_wait(1.0, 1.5, 2, cs2=1.0)
    assert _close(b["mean"], 2 * a["mean"]) and _close(a["p_wait"], erlang_c(2, 1.5))


# ------------------------------------------------------------------ prefix cache in the evaluator
def test_serving_prefix_cached():
    for bad in (-1, 4096, 5000, 1.5, True):
        try:
            Serving(prompt=4096, prefix_cached=bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    s = Scenario(**{**BASE, "serving": Serving(phase="prefill", batch=4, prompt=4096)})
    r0 = evaluate(s)
    assert evaluate(s.replace("serving.prefix_cached", 0)).ttft == r0.ttft
    prev = r0
    for p in (1024, 2048, 3072):
        r = evaluate(s.replace("serving.prefix_cached", p))
        assert r.ttft < prev.ttft
        assert _close(r.throughput, 4 * 4096 / r.step)        # throughput still counts the whole prompt
        prev = r
    d = Scenario(**BASE)
    assert evaluate(d.replace("serving.prefix_cached", 2048)).step == evaluate(d).step     # decode unaffected


# ------------------------------------------------------------------ PD with lengths / prefix
def test_trivial_mix_takes_the_general_path_and_matches_default():
    a = _rep()
    assert lengths_of(PDConfig(length_mix=((1.0, 4096, 512),)), SV).trivial       # one row = fixed lengths
    b = _rep(length_mix=((0.5, 4096, 512), (0.5, 4096, 512)))                     # same lengths, general code path
    assert a["lengths"]["plain"] and not b["lengths"]["plain"] and b["lengths"]["source"] == "mix"
    for k in ("req_s", "goodput_per_card", "ttft_ms", "tpot_ms"):
        assert _close(a[k], b[k], 1e-9), k
    assert _close(a["coloc"]["goodput_per_card"], b["coloc"]["goodput_per_card"], 1e-9)
    for m in a["queue"]["modes"]:
        x, y = a["queue"]["modes"][m], b["queue"]["modes"][m]
        assert _close(x["ttft_ms"]["p90"], y["ttft_ms"]["p90"], 1e-6) and _close(x["tpot_p90_ms"], y["tpot_p90_ms"], 1e-9)


def test_length_spread_effects():
    fx, cv = _rep(rate_rps=5.0), _rep(rate_rps=5.0, prompt_cv=1.0)
    assert cv["queue"]["modes"]["pd"]["ttft_ms"]["p90"] > fx["queue"]["modes"]["pd"]["ttft_ms"]["p90"]
    assert cv["lengths"]["decode_ctx"] == fx["lengths"]["decode_ctx"] == SV.ctx       # prompt spread alone: no bias
    oc = _rep(rate_rps=5.0, out_cv=1.0)
    assert oc["lengths"]["decode_ctx"] > SV.ctx and oc["tpot_ms"] > fx["tpot_ms"]
    assert oc["queue"]["modes"]["pd"]["decode"]["slot_wait_ms"]["mean"] >= fx["queue"]["modes"]["pd"]["decode"]["slot_wait_ms"]["mean"]
    # fluid prefill capacity over the mix = r_p · b / Σ w_i TTFT(b, S_i)
    mix = ((0.7, 1024, 256), (0.3, 11264, 1109))
    r = _rep(length_mix=mix)
    s = _scn(length_mix=mix)
    pb = r["prefill"]["batch"]
    pool = _Pool(s.replace("serving.prompt", 4096))
    t = 0.7 * pool.run("prefill", pb, 1024).ttft + 0.3 * pool.run("prefill", pb, 11264).ttft
    assert _close(r["cap_req_s"]["prefill"], pb / t, 1e-9)
    assert _close(r["cap_req_s"]["decode"], 3 * r["decode"]["tok_s_replica"] / r["lengths"]["mean_out"], 1e-9)
    assert _close(r["kv"]["bytes_per_req"], 0.7 * kv_bytes_per_request(get_model("qwen3-8b"), 1024)
                  + 0.3 * kv_bytes_per_request(get_model("qwen3-8b"), 11264), 1e-9)
    # stable rate ≈ fluid capacity for PD (queueing adds latency, not capacity)
    x = r["queue"]["modes"]["pd"]
    assert x["stable_rate_rps"] <= r["req_s"] * 1.0001 and x["stable_rate_rps"] > 0.9 * r["req_s"]


def test_prefix_cache_effects():
    base, h = _rep(), _rep(prefix_hit=0.5)
    m = get_model("qwen3-8b")
    assert h["lengths"]["prefix_tokens_rep"] == 2048 == prefix_tokens(4096, 0.5)
    assert _close(h["kv"]["bytes_per_req"], kv_bytes_per_request(m, 4096) / 2)
    nd = _rep(prefix_hit=0.5, prefix_on_decode=False)
    assert _close(nd["kv"]["bytes_per_req"], kv_bytes_per_request(m, 4096))
    assert h["prefill"]["ttft_ms"] < base["prefill"]["ttft_ms"] or h["prefill"]["batch"] > base["prefill"]["batch"]
    assert h["cap_req_s"]["prefill"] > base["cap_req_s"]["prefill"]
    assert h["tpot_ms"] == base["tpot_ms"]                                       # decode unchanged
    assert h["coloc"]["goodput_per_card"] > base["coloc"]["goodput_per_card"]
    qb, qh = base["queue"]["modes"], h["queue"]["modes"]
    for k in qb:
        assert qh[k]["ttft_ms"]["p90"] < qb[k]["ttft_ms"]["p90"]
    assert qh["coloc_chunked"]["prefill"]["chunks"] == 4 == _chunk_plan(4096, 2048, 512)[0]
    n, f, rr = _chunk_plan(4096, 0, 512)
    assert n == 8 and _close(f, 1 / 8) and _close(rr, (4096 - 512) / (2 * 4096))     # 0.51 re-read when no prefix
    assert prefix_tokens(10, 0.99) == 9 and prefix_tokens(100, 0) == 0


def test_chunked_iteration_mean_fix():
    r = _rep()
    c = r["queue"]["modes"]["coloc_chunked"]
    lam_c = c["lambda_rps"] / r["coloc"]["replicas"]
    nu = lam_c * c["prefill"]["chunks"]
    t0, rho = c["decode"]["iter_ms_no_chunk"] / 1e3, c["prefill"]["rho"]
    t1 = c["prefill"]["iter_ms"] / 1e3
    # 0.54: tbar = T₀/(1−ν(T₁−T₀)), x = ν·tbar; ρ_prefill = ν·T₁.  Mean TPOT is birth–death E[k]/(λ out), not tbar.
    tbar = t0 / (1 - nu * (t1 - t0))
    assert _close(c["prefill"]["chunk_share"], nu * tbar, 1e-9) and c["prefill"]["chunk_share"] < rho
    assert _close(rho, nu * t1, 1e-9)
    assert tbar < rho * t1 + (1 - rho) * t0                                  # below the 0.51 time-average
    assert c["tpot_mean_ms"] / 1e3 >= t0                                      # occupancy ≥ 1 step at least


def test_layout_search():
    assert _rep()["layout_search"] is None
    r = _rep(search_layouts=True)
    ls = r["layout_search"]
    rows = ls["rows"]
    assert ls["cards"] == 8 and ls["pairs"] >= len(rows) > 0
    assert all(rows[i]["req_s"] >= rows[i + 1]["req_s"] - 1e-12 for i in range(len(rows) - 1))
    assert ls["best_fluid"]["goodput_per_card"] >= r["goodput_per_card"] - 1e-9
    cur = ls["current_layouts"]
    assert cur["prefill_layout"] == "PP1·TP2" and cur["decode_layout"] == "PP1·TP2" and "slo_goodput_per_card" in cur
    assert ls["best_slo"]["slo_goodput_per_card"] >= cur["slo_goodput_per_card"] - 1e-9
    assert cur["slo_goodput_per_card"] >= r["queue"]["pd_slo_best_split"]["slo_goodput_per_card"] - 1e-6 * 600
    assert all(x["prefill_cards"] + x["decode_cards"] == 8 for x in rows)


def test_validation_api_cli():
    for bad in (dict(prompt_cv=-0.1), dict(out_cv=5.0), dict(prefix_hit=1.0), dict(prefix_hit=-0.1),
                dict(length_mix=((1.0, 0, 5),)), dict(length_mix=((0.0, 10, 5),)), dict(length_mix=((1.0, 10),)),
                dict(length_mix=((1.0, 10, 5),), prompt_cv=0.5), dict(length_mix=tuple((1.0, 10, 5) for _ in range(17))),
                dict(prefix_on_decode=1), dict(search_layouts="yes")):
        try:
            PDConfig(**bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(bad)
    s = Scenario.from_dict({"pd": {"length_mix": [[1, 100, 20], [3, 300, 40]]}})
    assert s.pd.length_mix == ((1.0, 100, 20), (3.0, 300, 40))
    assert Scenario.from_dict(s.to_dict()) == s
    out = api.api_eval({"scenario": {"model": "qwen3-8b", "serving": {"batch": 16},
                                     "pd": {"enabled": True, "decode_cards": 3, "prompt_cv": 0.5, "prefix_hit": 0.25}}})
    assert out["pd"]["lengths"]["source"] == "cv" and out["pd"]["lengths"]["prefix_hit"] == 0.25
    try:
        api.api_eval({"scenario": {"model": "qwen3-8b", "pd": {"enabled": True, "length_mix": [[1, 10]]}}})
        raise AssertionError
    except api.ApiError:
        pass
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-mix", "0.7:1024:256,0.3:8192:900", "--pd-prefix-hit", "0.4",
                      "--pd-prefix-not-on-decode", "--pd-search-layouts", "--prefix-cached", "128"])
    b = cli._body(a)["scenario"]
    assert b["pd"]["length_mix"] == [[0.7, 1024, 256], [0.3, 8192, 900]] and b["pd"]["prefix_hit"] == 0.4
    assert b["pd"]["prefix_on_decode"] is False and b["pd"]["search_layouts"] and b["serving"]["prefix_cached"] == 128
    a = p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-prompt-cv", "0.5", "--pd-out-cv", "1"])
    b = cli._body(a)["scenario"]["pd"]
    assert b["prompt_cv"] == 0.5 and b["out_cv"] == 1.0 and "length_mix" not in b
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["eval", "--model", "qwen3-8b", "--batch", "16", "--pd", "--pd-decode-cards", "3",
                         "--pd-prompt-cv", "0.5", "--pd-prefix-hit", "0.3", "--pd-search-layouts"]) == 0
    o = buf.getvalue()
    assert "lengths" in o and "M/G/1" in o and "layout search" in o and "prefix hit 30%" in o
