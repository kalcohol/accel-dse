"""API / server / CLI over core v2: strict input, thin-layer equivalence, live HTTP round trip."""
from __future__ import annotations

import contextlib
import io
import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

from accel_dse import __version__, api, cli
from accel_dse.core.catalog import UNLISTED, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIP_1P
from accel_dse.core.mapping import ORGS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.search import best_batch, search_layouts
from accel_dse.serve import MAX_BODY, Handler, handle

ROOT = Path(__file__).resolve().parents[1]

# video / protein catalog ids on core v2 (0.41: Wan2.1, CogVideoX, ESM-2; 0.42: the remaining video DiTs)
LIVE_DOMAIN = {"wan2.1-14b", "wan2.1-1.3b", "cogvideox-5b", "cogvideox-2b", "esm2-3b", "esm2-650m",
               "wan2.2-a14b", "hunyuanvideo", "ltx-video", "mochi-1", "opensora-stdit3", "minimax-h3",
               "esmfold", "alphafold2", "openfold", "boltz-1", "protenix"}


def _post(path, body):
    return handle("POST", path, json.dumps(body).encode())


def test_strict_json_and_unknown_keys_rejected():
    for raw in (b'{"scenario":{"mem_eff":NaN}}', b'{"scenario":{"mem_eff":Infinity}}', b'{"cards":-Infinity}', b"{bad"):
        st, out = handle("POST", "/api/eval", raw)
        assert st == 400 and "JSON" in out["error"], raw
    assert _post("/api/eval", {"scenario": {"bogus": 1}})[0] == 400
    assert _post("/api/eval", {"scenario": {"serving": {"batchh": 2}}})[0] == 400
    assert _post("/api/eval", {"scenario": {"model": "no-such-model"}})[0] == 400
    assert _post("/api/eval", {"scenario": {"layout": {"tp": 0}}})[0] == 400
    assert _post("/api/eval", {"scenario": {"model": "qwen3-30b-a3b", "layout": {"tp": 2}}})[0] == 400  # EP·ETP ≠ TP·DP
    assert handle("POST", "/api/eval", b"[1]")[0] == 400
    assert handle("POST", "/api/eval", b" " * (MAX_BODY + 1))[0] == 413
    assert _post("/api/layouts", {"cards": 0})[0] == 400 and _post("/api/layouts", {"objective": "x"})[0] == 400
    assert handle("GET", "/api/eval")[0] == 405 and handle("GET", "/api/nope")[0] == 404


def test_api_is_a_thin_layer_over_core():
    st, out = _post("/api/eval", {"chip_preset": "1P", "scenario": {
        "model": "qwen3-32b", "mem_id": "hbm3e_8s_12h24g_9200", "mapping": "ws_broad", "chip": {"sram_mib": 256},
        "layout": {"tp": 2}, "serving": {"batch": 8, "ctx": 8192}}})
    assert st == 200
    ref = evaluate(Scenario(model="qwen3-32b", chip=replace(CHIP_1P, sram_mib=256.0), mem_id="hbm3e_8s_12h24g_9200",
                            mapping="ws_broad", layout=Layout(tp=2), serving=Serving(batch=8, ctx=8192)))
    assert out["summary"]["tpot_ms"] == ref.tpot * 1e3 and out["hash"] == ref.scenario.hash()
    assert sum(s["t_ms"]["total"] for s in out["stages"]) > 0 and out["model"]["coverage"] == "full"


def test_best_batch_meets_slo_and_is_maximal():
    body = {"scenario": {"model": "qwen3-8b", "serving": {"tpot_slo_ms": 120.0}}, "best_batch": True}
    out = api.api_eval(body)
    b = out["summary"]["batch"]
    assert out["summary"]["tpot_ms"] <= 120.0
    base = api.scenario_from_body(body)
    assert b == best_batch(base).batch
    assert evaluate(base.replace("serving.batch", b + 1)).tpot * 1e3 > 120.0


def test_compare_rows_equal_per_mapping_search():
    body = {"scenario": {"model": "qwen3-30b-a3b", "mem_id": "hbm3e_8s_12h24g_9200", "serving": {"ctx": 2048}},
            "cards": 2}
    out = api.api_compare(body)
    assert [r["mapping"] for r in out["rows"]] == list(ORGS)
    base = api.scenario_from_body(body)
    for r in out["rows"]:
        top = search_layouts(base.replace("mapping", r["mapping"]), 2)[0]
        assert r["layout"] == top.layout.label and r["batch"] == top.batch and r["tok_s_card"] == top.per_card
        assert r["bound"] in ("MAC", "FEED", "VECTOR", "DRAM", "LINK", "SYNC") and 0 < r["array_util"] <= 1
        assert isinstance(r["ttft_ok"], bool)
    assert out["best_mapping"] == max(out["rows"], key=lambda r: r["score"])["mapping"]
    # reconf picks the per-op best organisation → never below any fixed organisation at the same layout/batch
    rec = next(r for r in out["rows"] if r["mapping"] == "reconf")
    assert all(rec["score"] >= r["score"] * (1 - 1e-9) for r in out["rows"])


def test_sweep_is_whitelisted_controlled_replacement():
    body = {"scenario": {"model": "qwen3-8b"}, "path": "chip.sram_port_Bpc", "values": [2048, 7616, 32768]}
    out = api.api_sweep(body)
    base = api.scenario_from_body(body)
    for row in out["rows"]:
        assert row["tpot_ms"] == evaluate(base.replace("chip.sram_port_Bpc", float(row["value"]))).tpot * 1e3
    for bad in ({"path": "chip.name", "values": [1]}, {"path": "serving.batch", "values": [1.5]},
                {"path": "serving.batch", "values": []}, {"path": "serving.batch", "values": [True]}):
        assert _post("/api/sweep", bad)[0] == 400, bad


def test_memory_endpoint_resolves_ids_and_fields():
    a = api.api_memory({"id": "hbm3e_8s_12h24g_9200"})
    b = api.api_memory({k: v for k, v in a["fields"].items() if k != "cap"})
    assert a["id"] == b["id"] and abs(a["raw_GBps"] - 9420.8) < 1e-9 and a["capacity_GiB"] == 288
    assert _post("/api/memory", {"type": "HBM3E", "bogus": 1})[0] == 400


def test_models_three_axis_labels_llm_only():
    """(0.41: name kept) every evaluable model carries the three labels; video / protein entries are evaluable only
    for the release-backed builders (LIVE_DOMAIN), every other one is a catalog-only row."""
    models = list_models()
    raw = json.loads((ROOT / "accel_dse" / "data" / "series_catalog.json").read_text())
    non_llm = {e["id"] for e in raw["entries"] if e.get("domain") != "llm"}
    live = LIVE_DOMAIN
    ids = {m["id"] for m in models}
    assert live <= non_llm and non_llm & ids == live
    assert {m["domain"] for m in models if m["id"] in live} == {"gen", "protein"}
    out = api.api_models()                                      # … the rest listed as catalog-only rows
    assert non_llm - set(UNLISTED) - live == {o["id"] for o in out["offline"]}
    assert sorted(out["catalog"]) == sorted([m["id"] for m in models] + [o["id"] for o in out["offline"]])
    for m in models:
        assert m["provenance"] in ("official", "mirror") and m["coverage"] in ("full", "partial", "proxy")
        assert m["proxy_badge"] == (m["coverage"] == "proxy") and m["dtype"].startswith("W ")
        assert abs(m["param_err"]) < 0.005, m["id"]


def test_live_server_round_trip():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        h = json.loads(urllib.request.urlopen(base + "/api/health").read())
        assert h["version"] == __version__
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "app.js" in html and urllib.request.urlopen(base + "/app.js").status == 200
        for bad in ("/../pyproject.toml", "/%2e%2e/accel_dse/api.py", "/nope.html"):
            try:
                urllib.request.urlopen(base + bad)
                raise AssertionError(bad)
            except urllib.error.HTTPError as e:
                assert e.code == 404
        rq = urllib.request.Request(base + "/api/eval", data=b'{"scenario":{"model":"qwen3-0.6b"}}',
                                    headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(rq).read())
        assert out["summary"]["model"] == "qwen3-0.6b"
        # infeasible prefill (671B on one 64 GiB card): TTFT is infinite in core → null on the wire, not invalid JSON
        rq = urllib.request.Request(base + "/api/eval", data=b'{"scenario":{"model":"deepseek-v3"}}')
        out = json.loads(urllib.request.urlopen(rq).read(), parse_constant=lambda c: 1 / 0)
        assert out["summary"]["fits"] is False and out["goodput"]["ttft_ms"] is None
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/api/eval", data=b'{"cards":NaN}'))
            raise AssertionError("NaN accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "non-finite" in json.loads(e.read())["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_commands():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["eval", "--model", "qwen3-0.6b", "--batch", "4", "--json"]) == 0
    out = json.loads(buf.getvalue())
    ref = evaluate(Scenario(model="qwen3-0.6b", serving=Serving(batch=4)))
    assert out["summary"]["tpot_ms"] == ref.tpot * 1e3
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert cli.main(["eval", "--model", "nope"]) == 2
        assert cli.main(["search", "--model", "qwen3-0.6b", "--cards", "2"]) == 0


def test_version_and_doc_links():
    py = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert py == __version__
    assert f"## [{__version__}]" in (ROOT / "CHANGELOG.md").read_text()
    for doc in ("README.md", "README.en.md", "docs/MODEL.md"):
        p = ROOT / doc
        for link in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", p.read_text()):
            if not link.startswith(("http://", "https://", "mailto:")):
                assert (p.parent / link).exists(), f"{doc}: broken link {link}"


def test_fit_hint_gives_feasible_one_click_fixes():
    """/api/fit: an OOM scenario gets a min-card layout and a larger memory that both really fit."""
    hbm = "hbm3e_8s_12h24g_9200"
    st, f = _post("/api/fit", {"scenario": {"model": "deepseek-v3", "mem_id": hbm}})
    assert st == 200 and not f["fits"] and f["need_GiB"] > f["cap_GiB"] and f["cards"] == 1
    mc = f["min_cards"]
    lay = Layout(**mc["layout_obj"])
    assert lay.cards == mc["cards"] > 1
    r = evaluate(Scenario(model="deepseek-v3", mem_id=hbm, layout=lay, serving=Serving(batch=mc["batch"])))
    assert r.fits and abs(max(s.mem.dram_need for s in r.stages) / 2**30 - mc["need_GiB"]) < 1e-6
    for n in (2, 4, 8, 16, 32, 64):           # minimal: no smaller power-of-two card count holds it at batch 1
        if n >= mc["cards"]:
            break
        assert not search_layouts(Scenario(model="deepseek-v3", mem_id=hbm), n, top=1)[0].batch  # batch-0 row = infeasible
    mm = f["min_mem"]
    assert mm["capacity_GiB"] > f["cap_GiB"]
    assert evaluate(Scenario(model="deepseek-v3", mem_id=mm["id"])).fits
    # batch overflow only: the largest batch that fits, and it is maximal
    st, f = _post("/api/fit", {"scenario": {"model": "qwen3-32b", "mem_id": hbm, "serving": {"batch": 4096, "ctx": 32768}}})
    assert st == 200 and not f["fits"] and f["fits_batch1"] and "min_cards" not in f
    b = f["max_batch"]
    base = Scenario(model="qwen3-32b", mem_id=hbm, serving=Serving(ctx=32768))
    assert evaluate(base.replace("serving.batch", b)).fits and not evaluate(base.replace("serving.batch", b + 1)).fits
    # a feasible scenario returns no fixes
    st, f = _post("/api/fit", {"scenario": {"model": "qwen3-8b"}})
    assert st == 200 and f["fits"] and not ({"min_cards", "min_mem", "max_batch"} & f.keys())
