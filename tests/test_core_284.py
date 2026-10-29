"""0.284: per-op dispatch sensitivity beyond speech — eval read-out (Result.dispatch_sens), LLM decode insight
(core/dispatch_insight.py, /api/insight_dispatch) and its UI panel; catalog defaults stay ideal."""
import json
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.dispatch_insight import _with, dispatch_sensitivity, step_sensitivity
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, DISPATCH_US
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
S = Scenario(model="qwen3-0.6b", chip=CHIPS["100T"], mem_id=LP).replace("serving.phase", "decode")


def test_catalog_defaults_ideal():
    assert all(c.dispatch == "" and c.dispatch_us is None for c in CHIPS.values())
    assert DISPATCH_US == {"static": 10.0, "trace": 1.5, "host": 20.0}


def test_step_sensitivity_exact():
    d = step_sensitivity(S)
    assert d["exact"] and d["ops_per_step"] > 100 and abs(d["ops_per_step"] - round(d["ops_per_step"])) < 1e-6
    t0 = evaluate(S).step
    host = evaluate(_with(S, "host", None)).step
    assert abs(d["slowdown_at"]["host"] - host / t0) < 1e-9
    at10 = evaluate(_with(S, "trace", d["us_10"])).step
    assert abs(at10 / t0 - 1.1) < 1e-9
    assert abs(d["us_eq"] - 10 * d["us_10"]) < 1e-9


def test_eval_readout_not_in_summary():
    r = evaluate(S)
    assert "dispatch_sens" not in r.summary()
    out = api.result_dict(r)
    assert out["dispatch_sens"]["ops_per_step"] == step_sensitivity(S)["ops_per_step"]
    rh = evaluate(_with(S, "host", None))          # the chip's own dispatch is stripped for the reference
    assert abs(rh.dispatch_sens()["step_ms"] - r.step * 1e3) < 1e-9


def test_llm_insight_rows():
    out = dispatch_sensitivity(S, ("qwen3-0.6b", "qwen3-30b-a3b"), (1, 8))
    rows = {(x["model"], x["batch"]): x for x in out["rows"]}
    assert len(rows) == 4 and out["chip_dispatch"] == "ideal"
    small, moe = rows[("qwen3-0.6b", 1)], rows[("qwen3-30b-a3b", 1)]
    assert moe["ops_per_step"] > small["ops_per_step"]               # more layers / expert ops per step
    # 0.467 bills q/k RMSNorm and mlp_mul on qwen3-0.6b. Both batches still
    # dispatch 367 ops. The 1 us slope subtraction lands 1 ulp apart.
    assert abs(rows[("qwen3-0.6b", 8)]["ops_per_token"] - small["ops_per_step"] / 8) < 1e-9
    for x in out["rows"]:
        assert x["tpot_ms_at"]["host"] > x["tpot_ms_at"]["trace"] > x["tpot_ms"]
        assert abs(x["tpot_ms_at"]["host"] / x["tpot_ms"] - (1 + 20 * x["ops_per_step"] * 1e-6 / (x["tpot_ms"] / 1e3))) < 1e-9
    # small-batch small models are the most dispatch-sensitive: u_10 grows with batch (TPOT grows, ops do not)
    assert small["us_10"] < rows[("qwen3-0.6b", 8)]["us_10"]


def test_api_route_and_ui():
    body = {"scenario": {"model": "qwen3-8b", "mem_id": LP, "chip": {"name": "100T"}}, "models": ["qwen3-0.6b"],
            "batches": [1]}
    assert "/api/insight_dispatch" in serve.POST_ROUTES
    st, payload = serve.respond("POST", "/api/insight_dispatch", json.dumps(body).encode())
    assert st == 200 and json.loads(payload)["rows"][0]["us_10"] > 0
    for bad in ({"batches": [0]}, {"batches": "1"}, {"models": []}, {"ctx": 0}, {"nope": 1}):
        try:
            api.api_insight_dispatch({**body, **bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    for i in ("ins-dp-models", "ins-dp-batches", "ins-dp-run", "ins-dp-status", "ins-dp-summary", "ins-dp-tbl"):
        assert i in W.html_ids(html), i
    assert [c["fn"] for c in W.req_calls(js) if c["path"] == "/api/insight_dispatch"] == ["runInsightDispatch"]
    assert W.has(W.functions(js)["renderWalls"], "dispSensNote(r.dispatch_sens)") or W.has(js, "dispSensNote(r.dispatch_sens)")
    assert "/api/insight_dispatch" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")
