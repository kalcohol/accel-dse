"""0.105: per-source batching estimator for max-load rows (opt-in extval.est_variant)."""
import accel_dse.core.extval as X


def test_every_max_source_has_a_documented_estimator():
    for path in (X.DATA, X.DATA_MOE, X.DATA_R3, X.DATA_R4):
        for r in X.rows(path):
            if r["metric"].startswith("max_tok_s"):
                e = X.estimator(r)
                assert e["kind"] == "cb", r["src"]
    assert X.estimator({"src": "unknown"})["kind"] == "static"
    assert X.SOURCE_ESTIMATOR["rocm-vllm-0.11.2"] == dict(kind="cb", chunk=131072, max_seqs=1024)   # 0.179: README --max-num-batched-tokens


def test_est_variant_uses_cb_and_cap():
    v = X.est_variant("catalog_kernel")
    r = next(r for r in X.rows(X.DATA_R4) if r["hw"].startswith("H200") and "deepseek" in r["model"])
    a = X.predict(r, v)
    b = X.predict_cb(r, "catalog_kernel")
    assert a["pred"] == b["pred"]
    capped = X.predict_cb(r, "catalog_kernel", max_seqs=4)
    assert capped["batch"] <= 4 and capped["pred"] <= b["pred"]
    assert X.predict(r, "catalog_kernel")["pred"] != a["pred"]       # default unchanged (static search)


def _pd_fixed_rate():
    from dataclasses import replace
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.scenario import Scenario, Serving, PDConfig
    from accel_dse.core.parallel import Layout
    from accel_dse.core.disagg import disagg_report
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=2), exec_overlap="stage",
                 serving=Serving(phase="decode", batch=8, ctx=4096, prompt=1024, out_len=128),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(), prefill_cards=1, decode_cards=2, load=0.8))
    lam = disagg_report(s)["queue"]["lambda_rps"]
    return s.replace("pd", replace(s.pd, rate_rps=lam)), replace, disagg_report


def test_pd_ttft_never_rises_with_prefix_or_link_at_fixed_rate():
    """0.105 PD audit (MODEL §51): at a fixed absolute arrival rate, a prefix-cache hit or a faster link never raises
    the queueing TTFT (mean / p90 / p99), and the fluid TTFT never rises at an unchanged prefill batch.  The headline
    fluid ttft_ms is the TTFT of best_prefill's batch, which may legitimately grow when prefill gets faster (a larger
    batch fits the SLO) -- that is a batch choice, not a latency regression."""
    s, replace, rep = _pd_fixed_rate()
    r0 = rep(s)
    for s2 in (s.replace("pd", replace(s.pd, prefix_hit=0.5)), s.replace("link", replace(s.link, GBps=2 * s.link.GBps)),
               s.replace("pd", replace(s.pd, kv_GBps=4 * s.link.GBps))):
        r1 = rep(s2)
        assert r1["queue"]["lambda_rps"] == r0["queue"]["lambda_rps"]
        q0, q1 = r0["queue"]["modes"]["pd"]["ttft_ms"], r1["queue"]["modes"]["pd"]["ttft_ms"]
        for k in ("mean", "p90", "p99"):
            assert q1[k] <= q0[k] * (1 + 1e-6), (k, q0[k], q1[k])
        if r1["prefill"]["batch"] <= r0["prefill"]["batch"]:
            assert r1["ttft_ms"] <= r0["ttft_ms"] * (1 + 1e-6)


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
