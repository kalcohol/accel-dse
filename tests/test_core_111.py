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
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"   # 0.437 re-baseline (umT5 RMSNorm; was cd25d689b7492f42) ← 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "7d44cdadbf38cc9e"   # 0.467 re-baseline (Qwen3 head width; was 555d25ec2cbe1bfe) ← 0.466 re-baseline (Qwen2.5-3B and 1.5B RMSNorm; was 25e9cd50ab57455e) ← 0.464 re-baseline (Llama-2 and Yi-1.5 RMSNorm; was a9cc980add47b869) ← 0.462 re-baseline (Mistral-Large RMSNorm; was 1e15ae85430a04f3) ← 0.437 re-baseline (umT5 RMSNorm; was 99a72aef99a2e674) ← 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


# 0.193: the busy-vs-DES operating point is pinned to the ≤ 0.192 reference GPUs (pin_legacy)
from tests.pin_legacy import pin_module_0192  # noqa: E402
pin_module_0192(globals(), only={"test_busy_lowers_small_model_prefill_first_ttft_toward_des"})


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
