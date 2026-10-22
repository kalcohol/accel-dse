"""0.320: design verdict report (core/report.py, POST /api/report, scripts/design_report.py, UI button) and the
DeepGEMM masked-kernel 256+ rows ratio (extkern prices the DeepGEMM kernel at DeepGEMM's own per-shape ceilings)."""
import importlib.util
import json
import math
import os
import tempfile

from accel_dse import api, serve
from accel_dse.core import extkern as XK
from accel_dse.core.energy_dse import D0_NOTE
from accel_dse.core.report import TAGS, tags_in, to_markdown
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
                                                                           "prompt": 2048, "ctx": 4096}},
        "energy": {"energy_preset": "tpuv4i_7nm"}, "mac_scales": [1], "sram_scales": [1]}
FULL = {**BASE, "budget": {"density_preset": "n5_derived"}, "dies": [1, 2], "cost": "n5_cowos_s", "tco": "accel_2026",
        "d0_per_cm2": 0.1, "slo_tpot_ms": 50, "cooling": "air_hgx"}
_C = {}


def _rep(body=None, **kw):
    b = {**(body or FULL), **kw}
    key = json.dumps(b, sort_keys=True)
    if key not in _C:
        _C[key] = api.api_report(b)
    return _C[key]


def _items(rep):
    return [it for s in rep["sections"] for it in s["items"]]


def test_every_number_labelled_or_missing():
    for body in (FULL, BASE):
        rep = _rep(body)["report"]
        for it in _items(rep):
            if it["value"] is None:
                assert it["missing"] and it["tag"] is None, it
            else:
                assert it["tag"] and all(t in TAGS for t in it["tag"].split("/")), it
                assert "missing" not in it
        assert {m["key"] for m in rep["missing"]} == {it["key"] for it in _items(rep) if it["value"] is None}
    assert tags_in("x「来源」y「推导」z「来源」") == "来源/推导" and tags_in("none") is None


def test_full_report_matches_dse_row():
    o = _rep()
    rep = o["report"]
    r = api.api_insight_dse(FULL)
    x = r["rows"][r["tco"]["best"]]
    assert o["row"] == r["tco"]["best"] == rep["row"] and rep["verdict"]["best_tco"]
    v = {it["key"]: it["value"] for it in _items(rep)}
    assert v["usd_per_Munit"] == x["tco"]["usd_per_Munit"] and v["usd_good_pkg"] == x["usd_good_pkg"]
    assert v["thr"] == x["thr"] and v["J_per_unit"] == x["J_per_unit"] and v["asm_Y"] == x["asm_Y"]
    assert math.isclose(v["slo_margin"], 50 / x["tpot_ms"] - 1)
    assert [s["id"] for s in rep["sections"]] == ["design", "perf", "energy", "shoreline", "thermal", "cost", "tco"]
    tco = next(s for s in rep["sections"] if s["id"] == "tco")
    assert tco["sens"]["corners"]["adverse"]["d"] > 0 > tco["sens"]["corners"]["favourable"]["d"]
    assert math.isclose(tco["sens"]["corners"]["adverse"]["d"], r["tco"]["sens"]["corners"]["adverse"]["d"], rel_tol=1e-9)
    assert rep["verdict"]["slo"] == "ok" and rep["verdict"]["thermal"] == "ok"
    # mono / split verdict of this design flips on D0 inside its sourced range (0.318 / 0.319 study)
    assert any(w["kind"] == "mono_split" and "D0" in w["text"] for w in rep["near_ties"])
    # the cheapest other design (another engine side) is within the 1 % tie but robust at the corners
    c = next(c for c in rep["pair_checks"] if c["kind"] == "cheapest_other")
    assert c["within_tie"] and c["corners"] == {"adverse": "this", "favourable": "this"} and not c["flips"]


def test_memory_alternative_pair_check():
    b = {**FULL, "cost": "n5_cowos_l", "dies": None, "mem_types": ["HBM3E"], "mem_count_scales": [1, 2],
         "slo_tpot_ms": None, "cooling": None}
    b = {k: v for k, v in b.items() if v is not None}
    rep = _rep(b)["report"]
    assert "hbm3e_8s" in rep["design"]
    c = next(c for c in rep["pair_checks"] if c["kind"] == "other_memory")
    assert "hbm3e_4s" in c["rival"] and 0.2 < c["d"] < 0.26 and set(c["corners"].values()) == {"this"}
    v = {it["key"]: it for it in _items(rep)}
    assert v["slo_tpot_ms"]["value"] is None and "未设 SLO" in v["slo_tpot_ms"]["missing"]
    assert v["cooling"]["value"] is None and v["flux_ratio"]["value"] is None


def test_missing_inputs_never_zero():
    rep = _rep(BASE)["report"]
    keys = {m["key"]: m["reason"] for m in rep["missing"]}
    assert "cost" in next(s["items"][0]["missing"] for s in rep["sections"] if s["id"] == "cost") or "成本" in keys["usd_good_pkg"]
    assert "TCO" in keys["usd_per_Munit"] and "密度" in keys["mm2"]
    lp6 = _rep({**BASE, "budget": {"density_preset": "n5_derived"}, "cost": "n5_cowos_s", "tco": "accel_2026",
                "mem_types": ["LPDDR6"], "mem_count_scales": [1]})["report"]
    m = {x["key"]: x["reason"] for x in lp6["missing"]}
    assert "LPDDR6" in m["usd_per_Munit"] and "LPDDR6" in m["mem_usd_per_GB"] and "LPDDR6" in m["capex_usd"]
    nod0 = _rep({**FULL, "d0_per_cm2": None})["report"]
    v = {it["key"]: it for it in _items(nod0)}
    assert v["d0"]["value"] is None and v["die_yield"]["value"] is None and v["usd_good_pkg"]["tag"] == "假设"


def test_user_prices_and_cost_dict_base():
    rep = _rep({**FULL, "cost": {"preset": "n5_cowos_s", "asm_model": "flat"}, "mem_usd_per_GB": {"HBM3E": 12}})["report"]
    v = {it["key"]: it for it in _items(rep)}
    assert v["mem_usd_per_GB"]["value"] == 12 and v["mem_usd_per_GB"]["tag"] == "用户输入"
    assert v["asm_Y"]["value"] == 1.0 and "0.318" in v["asm_Y"]["src"]
    assert next(s for s in rep["sections"] if s["id"] == "tco").get("sens")       # base rebuilt and checked exact


def test_markdown_and_formats():
    o = _rep()
    md = o["markdown"]
    for t in ("# 设计结论报告", "## SLO 下的性能", "## 能耗", "## 岸线可建性", "## 热 / 功耗", "## 良率与每好封装成本",
              "## TCO", "联合角点", "## 近似持平 / 翻转警告", "## 缺项", "| 项 | 值 | 单位 | 标签 | 出处 |"):
        assert t in md, t
    assert md == to_markdown(o["report"]) and "缺：未填功耗预算" in md
    j = api.api_report({**FULL, "report_format": "json"})
    assert "markdown" not in j and j["report"]["row"] == o["row"]
    m = api.api_report({**FULL, "report_format": "markdown", "report_row": 0})
    assert "report" not in m and m["row"] == 0 and "第 0 行" in m["markdown"]
    json.dumps(o["report"], allow_nan=False)
    for bad in ({**FULL, "report_row": 99}, {**FULL, "report_row": True}, {**FULL, "report_format": "pdf"},
                {**FULL, "bogus": 1}):
        try:
            api.api_report(bad)
            raise AssertionError(bad)
        except api.ApiError:
            pass


def test_cli_and_route():
    assert serve.POST_ROUTES["/api/report"] is api.api_report
    spec = importlib.util.spec_from_file_location("design_report", os.path.join(ROOT, "scripts", "design_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        assert m.main(["--example", "--md", d + "/r.md", "--json", d + "/r.json"]) == 0
        rep = json.load(open(d + "/r.json", encoding="utf-8"))
        assert rep["report"] == "design_verdict" and open(d + "/r.md", encoding="utf-8").read().startswith("# 设计结论报告")
        assert m.main(["--example", "--row", "999"]) == 2


def test_ui_and_static():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-dse-report", "ins-dse-report-status"} <= W.html_ids(html) and "runDesignReport" in W.functions(js)
    assert any(c["path"] == "/api/report" for c in W.req_calls(js))
    assert W.has(js, "S.dseBody = dseBody") and W.any_string_contains(js, "text/markdown")
    assert "/api/report" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")
    assert "0.319 按部件" in D0_NOTE and "键合良率未计" not in D0_NOTE


def test_deepgemm_masked_large_m():
    s = XK.summary()
    # README masked table (kernel author) and the DeepSeek_Simulator CSV agree within 2 %
    assert all(0.98 <= r["ratio"] <= 1.02 for r in s["readme_masked_check"]) and len(s["readme_masked_check"]) == 6
    bm = {x["m"]: x["ratio"] for x in s["grouped_by_m"]}
    old = {x["m"]: x["ratio"] for x in XK.summary(deepgemm=False)["grouped_by_m"]}
    assert all(bm[m] == old[m] for m in bm if m <= 128)              # DRAM-bound small m: unchanged
    assert 0.95 < bm[2048] < 1.0 and all(0.82 < bm[m] < 0.9 for m in (256, 512, 1024))
    assert old[512] < 0.66 and s["grouped_large_m_ratio_floor"] > 0.9


def test_raon_release():
    """0.320 freshness: KRAFTON Raon-1-mini-preview (public 2026-10-22 JST) — params = release header exactly, 1-based
    rda_config.mla_layers, Raon_mHC selector rows 2N + (N−1)² = 17, MXFP8 experts."""
    from accel_dse.core.catalog import list_models, get_model
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.hardware import CHIPS
    from accel_dse.core.scenario import Scenario, Serving
    from accel_dse.core import ir
    f = {x["id"]: x for x in list_models()}["raon-1-mini-preview"]
    assert f["provider"] == "krafton" and f["param_err"] == 0.0 and abs(f["params_B"] - 28.7832) < 1e-3
    assert f["is_moe"] and f["n_layers"] == 25 and 4.2 < f["active_B"] < 4.4
    m = get_model("raon-1-mini-preview")
    full = [i for i, L in enumerate(m.layers) if L.core.kind != "linear"]
    assert full == [3, 7, 11, 15, 19, 23, 24]
    assert all(L.hc == 4 and L.hc_rows == 17 for L in m.layers)
    assert any(l.name == "rda_w" for l in m.layers[0].attn_linears)
    fm = dict(m.formats)
    assert fm["expert"].fmt == "fp8" and abs(fm["expert"].bits - 8.25) < 1e-9 and fm["attn"].fmt == "bf16"
    ops = ir._hc_ops(m, 0, m.layers[0], 1, 1, "attn")
    assert ops[0].n == 17 and ops[0].k == 4 * m.hidden
    for ph, kw in (("decode", dict(batch=8, ctx=4096)), ("prefill", dict(batch=1, prompt=4096, ctx=4096))):
        r = evaluate(Scenario(model="raon-1-mini-preview", chip=CHIPS["H100-like"], mem_id="hbm3_5s_8h16g_5234",
                              serving=Serving(phase=ph, **kw)))
        assert r.fits and r.per_card > 0
