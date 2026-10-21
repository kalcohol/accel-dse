"""0.294: Drama 3 closed-loop TTFA (core/extspeech.py) — the SGLang prefill gate (prefill_decode_interval 4), the
others' prefills / colocated vocoder windows stretching each decode round, co-batched prefill and the vocoder queue."""
import json
import math
from pathlib import Path

from accel_dse.core import extspeech as XS

ROOT = Path(__file__).resolve().parents[1]


def test_interval_sourced():
    assert XS.PREFILL_INTERVAL == 4


def test_queue_terms_closed_form():
    # hand-derived: c = 9, N = 100 → a = 0.08, I·a = 0.32, p_win = 0.32 / 1.32
    # 0.296 signature: per-request prefill cost 0.010, the prefill batch waited for 0.015, p_win
    a, p = 0.08, 0.32 / 1.32
    q = XS._ttfa_queue(9, 100, 0.010, 0.015, p, 0.005, 0.002, [0.001, 0.003], 1.0, 10)
    tr = 0.005 + a * 0.012
    assert q["prefill"] == 0.015
    assert math.isclose(q["t_round"], tr)
    assert math.isclose(q["gate"], p * 1.5 * tr)
    assert math.isclose(q["prefill_batch"], 1 + p * 0.32)
    lam, es, es2 = 18.0, 0.002, (1e-6 + 9e-6) / 2
    assert math.isclose(q["vocoder_util"], lam * es)
    assert math.isclose(q["vocoder_queue"], lam * es2 / (2 * (1 - lam * es)))
    assert math.isclose(q["total"], q["gate"] + q["prefill"] + 10 * tr + q["vocoder_queue"])


def test_single_stream_reduces_to_no_queue():
    p = XS.predict("default", 1)
    assert p["ttfa_parts"]["gate"] == 0 and p["ttfa_parts"]["prefill_batch"] == 1
    assert abs(p["ttfa_s"] - p["ttfa_noq_s"]) < 1e-4            # only the (tiny) vocoder queue differs


def test_ttfa_grows_with_load_and_flattens_ratio():
    t = XS.table("default")
    pred = [x["ttfa_pred_ms"] for x in t]
    assert pred == sorted(pred) and all(x["ttfa_pred_ms"] >= x["ttfa_noq_ms"] for x in t)
    r = [x["ttfa_ratio"] for x in t]
    noq = [x["ttfa_noq_ms"] / x["ttfa_meas_ms"] for x in t]
    # diagnostic shape (never fitted): the no-queue form drifts 0.69 → 0.38 over c = 1 … 64; queueing keeps it flat
    assert max(noq) / min(noq) > 1.7 and max(r) / min(r) < 1.3
    p, ps = XS.predict("default", 64), XS.predict("default", 64, prefill="serial")
    # the others' prefills / windows stretch a round: ×1.87 priced serially, ×1.57 batched (0.296)
    assert ps["ttfa_parts"]["decode"] > 1.8 * 10 * ps["tpot_s"] and 1.4 < p["ttfa_parts"]["t_round"] / p["tpot_s"] < 1.8


def test_second_gpu_keeps_first_window():
    p = XS.predict("default", 128, "none")
    assert p["vocoder_s_per_request"] == 0 and p["ttfa_parts"]["vocoder_first"] > 0


def test_ext_validation_ttfa_fresh():
    d = json.loads((ROOT / "accel_dse/data/ext_validation.json").read_text())
    cur = {x["id"]: x["ttfa_pred_ms"] for x in XS.table("default")}
    assert {x["id"]: x["ttfa_pred_ms"] for x in d["speech"]["default"]["rows"]} == cur, "re-run scripts/ext_validate.py"


# --- freshness: LiquidAI LFM2.5-24B-A2B-DSpark (2026-10-20 JST) on the FP8 target, as served by the README ---------
DS = "lfm2.5-24b-a2b-fp8-dspark"


def test_lfm_dspark_structure_and_params():
    from accel_dse.core.catalog import get_model
    m, t = get_model(DS), get_model("lfm2.5-24b-a2b-fp8")
    assert m.draft == "dspark" and m.draft_block == 9 and m.draft_targets == 5 and m.draft_markov == 256
    assert not m.draft_reader and m.spec_published == (5.16, 9) and len(m.mtp_layers) == 5
    assert all(L.core.window is None for L in m.mtp_layers)
    pc = m.param_check()
    assert pc["rel_err"] == 0 and abs(pc["mtp_ours"] - 327707521) < 5   # safetensors header (floor of extra / layers)
    assert m.layers == t.layers and m.vocab == t.vocab == 128000          # the FP8 target as released


def test_lfm_dspark_spec_eval():
    from accel_dse.core.model import spec_accept_from_tau
    from accel_dse.core.schedule import spec_expected_tokens
    from accel_dse import api
    a = spec_accept_from_tau(5.16, 9)
    assert abs(spec_expected_tokens(9, a) - 5.16) < 1e-6
    sv = {"batch": 1, "ctx": 4096}
    ar = api.api_eval({"chip_preset": "H100-like", "scenario": {"model": DS, "serving": sv}})
    sp = api.api_eval({"chip_preset": "H100-like", "scenario": {"model": DS, "serving": {**sv, "spec_k": 9, "spec_accept": a}}})
    assert sp["model"]["spec_published"]["k"] == 9
    # bandwidth-bound roofline: 10 verified tokens touch ~7× the experts of one (64 experts top-4) → modest speed-up,
    # far below the published 2.38× (B300 baseline 324 tok/s is not bandwidth-bound) — diagnostic, never fitted
    assert 1.0 < sp["summary"]["tok_s"] / ar["summary"]["tok_s"] < 2.0


def test_sweep_orgs_0294():
    import importlib.util
    spec = importlib.util.spec_from_file_location("freshness_sweep", ROOT / "scripts" / "freshness_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    orgs = m.sweep_orgs()
    assert {"Zyphra", "fla-hub", "recursal", "neuphonic", "hume-ai"} <= set(orgs) and len(orgs) == len(set(orgs))
