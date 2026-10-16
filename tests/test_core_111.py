"""0.111: opt-in pd.ttft_resid = busy (prefill-first TTFT decode residual only while an iteration is in flight)."""
from accel_dse import api as A
from accel_dse.core.scenario import Scenario

SC = {"model": "carbon-500m", "serving": {"phase": "decode", "batch": 32, "prompt": 512, "out_len": 512},
      "pd": {"enabled": True, "load": 0.3}}


def _q(extra=None):
    sc = {**SC, "pd": {**SC["pd"], **(extra or {})}}
    return A.api_eval({"scenario": sc, "sensitivity": False})["pd"]["queue"]["modes"]


def test_default_full_and_validation():
    s = Scenario.from_dict(SC)
    assert s.pd.ttft_resid == "full" and "ttft_resid" not in s.to_dict()["pd"]
    assert Scenario.from_dict({"model": "qwen3-8b"}).hash() == "723894c17df5d7bb"
    a, b = _q(), _q({"ttft_resid": "full"})
    assert a == b
    try:
        Scenario.from_dict({**SC, "pd": {**SC["pd"], "ttft_resid": "maybe"}}); raise AssertionError
    except ValueError as e:
        assert "ttft_resid" in str(e)


def test_busy_lowers_small_model_prefill_first_ttft_toward_des():
    full, busy = _q()["coloc_prefill_first"]["ttft_ms"], _q({"ttft_resid": "busy"})["coloc_prefill_first"]["ttft_ms"]
    # DES (6000 requests, val5/v0111_resid): mean 8.25, p50 6.67, p90 11.89 ms; full: 10.12 / 10.11 / 13.72
    assert busy["mean"] < full["mean"] and busy["p50"] < full["p50"] and busy["p90"] < full["p90"]
    for k, des in (("mean", 8.25), ("p50", 6.67), ("p90", 11.89)):
        assert abs(busy[k] / des - 1) < 0.1, (k, busy[k])
    # PD and chunked modes are untouched
    assert _q()["pd"] == _q({"ttft_resid": "busy"})["pd"]


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "84df3be517a53ef4"   # 0.173 re-baseline (was 0a954915c5fb003f)
    assert _fp({}) == "1b755ac357ec6d2e"   # 0.173 re-baseline (was 53ee9e08165795c6)
