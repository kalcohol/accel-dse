"""0.296: Drama 3 batched prefill in the closed-loop steady state (core/extspeech.py) and the LFM2.5 DSpark c8–64 gap
located by mechanism (core/extspec.token_curve)."""
import json
import math
from pathlib import Path

from accel_dse.core import extspec as XP
from accel_dse.core import extspeech as XS

ROOT = Path(__file__).resolve().parents[1]


def test_batch_distribution():
    d = XS._batch_dist(1.0, 110)
    assert d == [(1, 1.0)]
    d = XS._batch_dist(3.5, 110)
    assert math.isclose(sum(w for _, w in d), 1.0) and math.isclose(sum(n * w for n, w in d), 3.5, rel_tol=1e-6)
    d = XS._batch_dist(30.0, 8)                     # the cap absorbs the tail
    assert d[-1][0] == 8 and max(n for n, _ in d) == 8 and d[-1][1] > 0.99


def test_gate_matches_0294():
    p, k = XS._gate(9, 100)
    assert math.isclose(p, 0.32 / 1.32) and math.isclose(k, 1 + p * 0.32)
    assert XS._gate(1, 95) == (0.0, 1.0)


def test_single_stream_unchanged():
    b, s = XS.predict("default", 1), XS.predict("default", 1, prefill="serial")
    assert b["frames_per_s"] == s["frames_per_s"] and b["ttfa_s"] == s["ttfa_s"]
    assert b["prefill_s_per_request"] == b["ttft1_s"]


def test_batched_prefill_cheaper_per_request():
    for c in (8, 64, 256):
        b, s = XS.predict("default", c), XS.predict("default", c, prefill="serial")
        assert b["prefill_s_per_request"] < s["prefill_s_per_request"] == s["ttft1_s"]
        # the batch a request waits for costs less than k serial prefills but more than one
        assert b["ttft1_s"] < b["ttfa_parts"]["prefill"] < s["ttfa_parts"]["prefill"]
        assert b["frames_per_s"] > s["frames_per_s"] and b["ttfa_s"] < s["ttfa_s"]


def test_rows_carry_serial_record():
    for x in XS.table("default"):
        assert x["pred"] >= x["pred_serial"] and x["ttfa_pred_ms"] <= x["ttfa_serial_ms"]


def test_ttfa_ratio_flat_but_low():
    # diagnostic shape (never fitted): batched TTFA under-predicts ~ uniformly over c = 1 … 64 (0.64–0.73)
    r = [x["ttfa_ratio"] for x in XS.table("default") if x["c"] <= 64]
    assert max(r) / min(r) < 1.2 and max(r) < 0.8


def test_token_curve_locates_lfm_gap():
    tc = XP.token_curve(XP.table("default"))
    # a DSpark step costs what a target-alone step of the same token count costs (measured and predicted alike) …
    same = tc["spec_vs_base_at_tokens"]
    assert [x["c"] for x in same] == [1, 2, 4, 8, 16]
    assert all(0.85 < x["meas"] < 1.2 and 0.95 < x["pred"] < 1.2 for x in same)
    # … so the c8–64 speed-up over-prediction is the target's token slope: ≥ 4× under-priced from 64 tokens on
    sl = {x["c"]: x for x in tc["base_slope_us_per_token"]}
    assert all(sl[k]["meas"] > 4 * sl[k]["pred"] for k in ("32-64", "64-128", "128-256"))


def test_ext_validation_fresh_0296():
    d = json.loads((ROOT / "accel_dse/data/ext_validation.json").read_text())
    assert d["spec"]["default"]["token_curve"] == XP.token_curve(XP.table("default")), "re-run scripts/ext_validate.py"
    cur = {x["id"]: (x["pred_serial"], x["ttfa_serial_ms"]) for x in XS.table("default")}
    assert {x["id"]: (x["pred_serial"], x["ttfa_serial_ms"]) for x in d["speech"]["default"]["rows"]} == cur
