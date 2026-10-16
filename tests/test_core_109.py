"""0.109: prefill two-batch overlap for the sources that document it (extval candidate); API audit fixes."""
import json

import accel_dse.core.extval as X
from accel_dse import api as A
from accel_dse import serve


def test_tbo_prefill_only_for_documented_sources():
    c = X.candidate_variant("catalog_kernel")
    assert X.VARIANTS[c]["tbo_pf"] is True
    assert X.TBO_PREFILL_SRC == {"sglang-ep-2025", "deepseek-profile"}
    for r in X.rows(X.DATA_MOE):
        s_pf = X.scenario(r, c, phase="prefill", batch=1, prompt=1024)
        s_dec = X.scenario(r, c, phase="decode", batch=8, ctx=1024)
        assert s_pf.exec_overlap == ("tbo" if r["src"] in X.TBO_PREFILL_SRC else "kernel"), r["id"]
        assert s_dec.exec_overlap == "kernel", r["id"]


def test_tbo_prefill_moves_pf_rows_toward_measured():
    c = X.candidate_variant("catalog_kernel")
    base = X.VARIANTS[c]
    X.VARIANTS["_no_tbopf"] = {**base, "tbo_pf": False}
    for r in X.rows(X.DATA_MOE):
        if r["metric"] != "prefill_tok_s_node":
            continue
        a = X.predict(r, "_no_tbopf")["pred"] / r["value"]
        b = X.predict(r, c)["pred"] / r["value"]
        assert a < b and 0.8 < b < 1.15, (r["id"], a, b)
    X.VARIANTS.pop("_no_tbopf")


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "8410808fd4f1e578"   # 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "b8ca44d92e7589cd"   # 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


def _ev(sc):
    return A.api_eval({"scenario": sc, "sensitivity": False})["summary"]


def test_eval_summary_nulls_inapplicable_latencies_like_sweep():   # 0.109 API audit: eval said "0 ms"
    cases = (({"model": "qwen3-8b"}, "mem_eff", 0.6), ({"model": "qwen3-8b", "serving": {"phase": "prefill", "batch": 1, "prompt": 512}}, "mem_eff", 0.6),
             ({"model": "esm2-650m"}, "chip.sram_mib", 64))
    for sc, path, v in cases:
        row = A.api_sweep({"scenario": sc, "path": path, "values": [v]})["rows"][0]
        s2 = json.loads(json.dumps(sc)); x = s2
        *h, last = path.split(".")
        for k in h:
            x = x.setdefault(k, {})
        x[last] = v
        e = _ev(s2)
        for k in ("tpot_ms", "ttft_ms", "latency_ms", "tok_s_card", "bound", "fits"):
            assert row[k] == e[k], (sc, k, row[k], e[k])
    d = _ev({"model": "qwen3-8b"})
    assert d["tpot_ms"] > 0 and d["ttft_ms"] is None and d["latency_ms"] is None


def test_sensitivity_endpoint_equals_eval_block():
    sc = {"model": "qwen3-8b"}
    a = A.api_eval({"scenario": sc, "sensitivity": True})["sensitivity"]
    b = A.api_sensitivity({"scenario": sc})["sensitivity"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_adversarial_bodies_are_400_never_500():
    for raw in (b"", b"[]", b"null", b"{", b'{"scenario": NaN}', b"\xff\xfe", b'"x"', b'{"scenario": {"model": 1e999}}',
                b'{"scenario": {"model": "qwen3-8b", "kv_mem_eff": 0}}', b'{"scenario": {"model": "qwen3-8b", "layout": {"tp": "2"}}}',
                b'{"scenario": {"model": "qwen3-8b", "serving": {"batch": 1e308}}}', b'{"scenario": {"model": "qwen3-8b"}, "objetive": "x"}'):
        for route in ("/api/eval", "/api/sweep", "/api/pareto", "/api/sensitivity", "/api/fit"):
            st, out = serve.handle("POST", route, raw)
            assert st in (200, 400), (route, raw, st)
            json.dumps(out, allow_nan=False)
