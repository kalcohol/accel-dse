"""0.295: speculative-decoding MoE gap (core/extspec.py: LFM2.5-24B-A2B FP8 + DSpark on one B300) — expert union,
locality bound, GPU per-kernel baseline; Drama 3 synchronized start burst (core/extspeech.burst)."""
import csv
import json
import math
from pathlib import Path

from accel_dse.core import extspec as XP
from accel_dse.core import extspeech as XS

ROOT = Path(__file__).resolve().parents[1]


def test_spec_rows_sourced():
    rs = XP.rows()
    assert len(rs) == 72 and {r["concurrency"] for r in rs} == {1, 2, 4, 8, 16, 32, 64, 128, 256}
    c1 = [r for r in rs if r["concurrency"] == 1]
    assert abs(sum(r["ratio"] for r in c1) / 8 - 2.38) < 0.01          # README "Arithmetic mean" 2.38×
    assert abs(sum(r["tau"] for r in c1) / 8 - 5.16) < 0.01             # README mean tokens / step 5.16
    m = next(r for r in rs if r["id"] == "lfm-dspark-b300-math500-c1")
    assert m["baseline_tok_s"] == 324.08 and m["out_cap"] == 2048


def test_union_counts_match_uniform_routing():
    r = next(x for x in XP.rows() if x["id"] == "lfm-dspark-b300-math500-c1")
    p = XP.predict(r, "catalog")
    # 64 experts top-4: 10 verified tokens touch ⌈64(1 − (60/64)^10)⌉ = 31 experts vs 4 for one token
    assert math.ceil(64 * (1 - (60 / 64) ** 10)) == 31 and abs(p["exp_union"] - 31 / 4) < 0.05
    assert abs(p["tau_model"] - r["tau"]) < 1e-3                       # the measured τ is the input


def test_mechanisms_c1():
    t = {v: XP.by_concurrency(XP.table(v)) for v in ("catalog", "default")}
    roof, gpu = t["catalog"][0], t["default"][0]
    assert roof["c"] == gpu["c"] == 1
    # (c) the bandwidth roofline reproduces the 0.294 catalog-chip finding (~1.1×); the per-kernel floor baseline
    # (trace-sourced, not fitted) lands near the published 2.38×
    assert roof["speedup_pred"] < 1.3 and 2.0 < gpu["speedup_pred"] < 3.0 and gpu["kern_share_base"] > 0.6
    # (b) perfect locality would overshoot far beyond the measurement → strong locality is not supported
    assert gpu["speedup_locality_bound"] > 1.5 * gpu["speedup_meas"]
    # (a) the union shrinks with concurrency (all 64 experts already touched at c ≥ 128)
    assert t["default"][-1]["exp_union"] == 1.0 and t["default"][0]["exp_union"] > 7


def test_ext_validation_spec_fresh():
    d = json.loads((ROOT / "accel_dse/data/ext_validation.json").read_text())
    cur = {x["id"]: x["speedup_pred"] for x in XP.table("default")}
    assert {x["id"]: x["speedup_pred"] for x in d["spec"]["default"]["rows"]} == cur, "re-run scripts/ext_validate.py"


def test_burst_schedule():
    b = XS.burst("default", 256)
    from accel_dse.core.speech import speech_kpis
    rows = speech_kpis(XS._scenario("default", 1).replace("serving.batch", 1), XS.OUT_S)["prompt_rows"]
    assert b["batch_cap"] == 16384 // rows and sum(b["batches"]) == 256 and len(b["batches"]) == math.ceil(256 / b["batch_cap"])
    assert b["wave1_mean_s"] <= b["wave1_max_s"] and abs(b["wave1_share"] - 256 / 1088) < 1e-12
    one = XS.burst("default", 1)
    p1 = XS.predict("default", 1)
    assert abs(one["wave1_mean_s"] - p1["ttfa_s"]) < 2e-3                  # c = 1: the same path as the steady state


def test_burst_does_not_explain_c256():
    t2 = {(x["c"], x["vocoder_gpu"]): x for x in XS.table("default", t2=True)}
    for g in ("same", "second"):
        x = t2[(256, g)]
        # wave 1 (23.5 % of the run) is below even the steady state; the measured p50 exceeds the stride frames at the
        # measured per-stream rate by > 1.4 s — a wait the synchronized start cannot produce
        assert x["ttfa_burst_wave1_ms"] < x["ttfa_pred_ms"] and x["ttfa_meas_ms"] - x["ttfa_rate_ms"] > 1400
