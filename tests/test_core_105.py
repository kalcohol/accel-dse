"""0.105: per-source batching estimator for max-load rows (opt-in extval.est_variant)."""
import accel_dse.core.extval as X


def test_every_max_source_has_a_documented_estimator():
    for path in (X.DATA, X.DATA_MOE, X.DATA_R3, X.DATA_R4):
        for r in X.rows(path):
            if r["metric"].startswith("max_tok_s"):
                e = X.estimator(r)
                assert e["kind"] == "cb", r["src"]
    assert X.estimator({"src": "unknown"})["kind"] == "static"
    assert X.SOURCE_ESTIMATOR["rocm-vllm-0.11.2"] == dict(kind="cb", chunk=16384, max_seqs=1024)


def test_est_variant_uses_cb_and_cap():
    v = X.est_variant("catalog_kernel")
    r = next(r for r in X.rows(X.DATA_R4) if r["hw"].startswith("H200") and "deepseek" in r["model"])
    a = X.predict(r, v)
    b = X.predict_cb(r, "catalog_kernel")
    assert a["pred"] == b["pred"]
    capped = X.predict_cb(r, "catalog_kernel", max_seqs=4)
    assert capped["batch"] <= 4 and capped["pred"] <= b["pred"]
    assert X.predict(r, "catalog_kernel")["pred"] != a["pred"]       # default unchanged (static search)
