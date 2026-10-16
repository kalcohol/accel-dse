"""0.207: ROCm paged-attention KV-stream efficiency (FlyDSL PR #825) on the MI300X reference chip only; small-M GEMM
source (aiter tuned table) checked, not applied; NVIDIA unchanged."""
import math

from accel_dse.core import extval as X


def test_pa_curve_points():
    # bytes / µs ÷ 5.3 TB/s at the published shapes
    assert abs(X.pa_rocm_eff(32 * 4096 * 256) - 0.181) < 0.002
    assert abs(X.pa_rocm_eff(128 * 4096 * 256) - 0.303) < 0.002
    assert abs(X.pa_rocm_eff(128 * 16384 * 256) - 0.449) < 0.002     # mean of the two 537 MB points
    assert X.pa_rocm_eff(1e3) == X.pa_rocm_eff(32 * 4096 * 256) and X.pa_rocm_eff(1e12) == X.pa_rocm_eff(537e6)
    a = [X.pa_rocm_eff(b) for b in (4e7, 8e7, 2e8, 4e8)]
    assert all(x < y for x, y in zip(a, a[1:]))


def test_pa_only_mi300x_decode():
    assert X.VARIANTS["default"].get("pa_rocm") and not X.VARIANTS["default_0206"].get("pa_rocm")
    assert X.PA_ROCM_HW == frozenset({"MI300X"})
    for r in X.rows()[:20] + X.rows(X.DATA_R3):
        if r["hw"] != "MI300X" and r["metric"] in ("static_tok_s_gpu", "ttft_ms", "max_tok_s_total"):
            assert X.predict(r)["pred"] == X.predict(r, "default_0206")["pred"], r["id"]
    lat = [r for r in X.rows(X.DATA_R3) if r["metric"] == "static_latency_s"]
    p_new = X.predict(lat[0]); p_old = X.predict(lat[0], "default_0206")
    assert p_new["ttft_s"] == p_old["ttft_s"]                 # prefill (FA, not PA) untouched
    assert p_new["tpot_s"] >= p_old["tpot_s"]


def _score(rs, v):
    l = []
    for r in rs:
        p = (X.predict_cb(r, v) if r["metric"] == "max_tok_s_total" else X.predict(r, v))["pred"]
        x = p / r["value"]
        l.append(-math.log(x) if r["metric"] in ("static_latency_s", "ttft_ms") else math.log(x))
    return math.exp(sum(l) / len(l)), 100 * (math.exp(sum(abs(a) for a in l) / len(l)) - 1)


def test_mi300x_before_after():
    lat = [r for r in X.rows(X.DATA_R3) if r["metric"] == "static_latency_s"]
    ml = [r for r in X.rows(X.DATA_R3) if r["hw"] == "MI300X" and r["metric"] == "max_tok_s_total"]
    (g0, e0), (g1, e1) = _score(ml, "default_0206"), _score(ml, "default")
    assert 1.6 < g0 < 1.7 and 1.45 < g1 < 1.53 and e1 < e0
    (h0, f0), (h1, f1) = _score(lat, "default_0206"), _score(lat, "default")
    assert f1 < f0 and 0.94 < h1 < 1.0
