"""0.321: side-by-side design comparison (core/report.compare_report, POST /api/report report_rows / report_compare,
scripts/design_report.py --compare, UI button); colocated vocoder time-slicing (extspeech); Dory logit_scale."""
import importlib.util
import json
import os
import tempfile

from accel_dse import api
from accel_dse.core import extspeech as XS
from accel_dse.core import ir
from accel_dse.core.catalog import get_model
from accel_dse.core.report import TAGS, compare_markdown
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
                                                                           "prompt": 2048, "ctx": 4096}},
        "energy": {"energy_preset": "tpuv4i_7nm"}, "mac_scales": [1], "sram_scales": [1]}
FULL = {**BASE, "budget": {"density_preset": "n5_derived"}, "dies": [1, 2], "cost": "n5_cowos_s", "tco": "accel_2026",
        "d0_per_cm2": 0.1, "slo_tpot_ms": 50, "cooling": "air_hgx"}
MIXED = {**BASE, "budget": {"density_preset": "n5_derived"}, "cost": "n5_cowos_s", "tco": "accel_2026",
         "mem_types": ["HBM3E", "LPDDR6"], "mem_count_scales": [1]}
_C = {}


def _cmp(body, **kw):
    b = {**body, **kw}
    k = json.dumps(b, sort_keys=True)
    if k not in _C:
        _C[k] = api.api_report(b)
    return _C[k]


def test_cells_are_the_single_reports():
    """Each column is exactly that design's own report item (value, tag, src, missing) — nothing recomputed."""
    o = _cmp(FULL, report_rows=[3, 0, 1])
    c = o["compare"]
    assert o["rows"] == c["rows"] == [3, 0, 1] and [d["letter"] for d in c["designs"]] == ["A", "B", "C"]
    for n, row in enumerate(c["rows"]):
        single = api.api_report({**FULL, "report_row": row, "report_format": "json"})["report"]
        for s in single["sections"]:
            cs = next(x for x in c["sections"] if x["id"] == s["id"])
            for it in s["items"]:
                e = next(x for x in cs["items"] if x["key"] == it["key"])
                cell = e["cells"][n]
                assert cell["value"] == it["value"] and cell["tag"] == it["tag"] and cell["src"] == it["src"], (row, it["key"])
    for s in c["sections"]:
        for e in s["items"]:
            for cell in e["cells"]:
                if cell["value"] is None:
                    assert cell["missing"] and cell["tag"] is None
                else:
                    assert all(t in TAGS for t in cell["tag"].split("/"))
    json.dumps(c, allow_nan=False)


def test_differences_winners_and_ties():
    c = _cmp(FULL, report_rows=[3, 0, 1])["compare"]
    items = {e["key"]: e for s in c["sections"] for e in s["items"]}
    u = items["usd_per_Munit"]
    v = [x["value"] for x in u["cells"]]
    assert u["best"] == v.index(min(v)) and u["direction"] == "lower" and u["differs"]
    assert abs(u["delta"][1] - (v[1] / v[0] - 1)) < 1e-15 and u["delta"][0] == 0
    p = items["perf_W"]
    assert p["best"] == [x["value"] for x in p["cells"]].index(max(x["value"] for x in p["cells"]))
    assert not items["model"]["differs"] and items["array"]["differs"] and "best" not in items["array"]
    # near ties are mechanism checks: the $/M pair check (ranges + corners) for every design vs A
    assert len(c["pair_checks"]) == 2 and all(pc["within_tie"] for pc in c["pair_checks"])
    assert all(pc["corners"] == {"adverse": "this", "favourable": "this"} for pc in c["pair_checks"])
    assert sum(w["kind"] == "tco_pair" for w in c["near_ties"]) == 2
    assert not any(w["kind"] == "metric" and w["key"] == "usd_per_Munit" for w in c["near_ties"])
    # energy shares: every design shows the union of top shares (the row's own computed value, not "missing")
    sh = [e for e in items.values() if e["key"].startswith("share_")]
    assert sh and all(cell["value"] is not None for e in sh for cell in e["cells"])
    assert c["n_differ"] == sum(1 for e in items.values() if e.get("differs"))


def test_unpriced_design_is_missing_not_zero():
    o = _cmp(MIXED, report_rows=[2, 3])
    c = o["compare"]
    items = {e["key"]: e for s in c["sections"] for e in s["items"]}
    u = items["usd_per_Munit"]
    assert u["cells"][0]["value"] > 0 and u["cells"][1]["value"] is None and "LPDDR6" in u["cells"][1]["missing"]
    assert u["missing_in"] == [1] and u["delta"][1] is None and not c["pair_checks"]
    assert any("未定价" in w["text"] for w in c["near_ties"])
    assert "缺：" in o["markdown"] and "LPDDR6" in o["markdown"]


def test_default_pair_and_bad_inputs():
    o = _cmp(FULL, report_compare=True)
    r = api.api_insight_dse(FULL)
    best = r["tco"]["best"]
    others = [j for j, y in enumerate(r["rows"]) if j != best and (y.get("tco") or {}).get("usd_per_Munit") is not None]
    assert o["rows"] == [best, min(others, key=lambda j: r["rows"][j]["tco"]["usd_per_Munit"])]
    for bad in ({"report_rows": [1]}, {"report_rows": [0, 0]}, {"report_rows": [0, 1, 2, 3]}, {"report_rows": [0, 99]},
                {"report_rows": [0, True]}, {"report_rows": "0,1"}, {"report_compare": 1},
                {"report_rows": [0, 1], "report_row": 0}):
        try:
            api.api_report({**FULL, **bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    j = api.api_report({**FULL, "report_rows": [0, 1], "report_format": "json"})
    assert "markdown" not in j and "compare" in j


def test_markdown():
    o = _cmp(FULL, report_rows=[3, 0, 1])
    md = o["markdown"]
    assert md == compare_markdown(o["compare"])
    for t in ("# 设计对比报告", "## 关键指标", "## 近似持平 / 翻转警告", "| A vs B |", "| A vs C |", "◆", "**",
              "## 各设计自身的警告与缺项", "| 标签 | 出处 |"):
        assert t in md, t


def test_cli_compare():
    spec = importlib.util.spec_from_file_location("design_report", os.path.join(ROOT, "scripts", "design_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        assert m.main(["--example", "--compare", "3,0", "--md", d + "/c.md", "--json", d + "/c.json"]) == 0
        c = json.load(open(d + "/c.json", encoding="utf-8"))
        assert c["report"] == "design_compare" and c["rows"] == [3, 0]
        assert open(d + "/c.md", encoding="utf-8").read().startswith("# 设计对比报告")
        assert m.main(["--example", "--compare", "auto", "--md", d + "/a.md"]) == 0
    assert m.main(["--example", "--compare", "3"]) == 2 and m.main(["--example", "--compare", "x,y"]) == 2
    assert m.main(["--example", "--compare", "3,0", "--row", "1"]) == 2


def test_ui_compare_button():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-dse-compare", "ins-dse-cmp-rows", "ins-dse-compare-status"} <= W.html_ids(html)
    assert "runDesignCompare" in W.functions(js)
    fn = W.function_source(js, "runDesignCompare")
    assert W.any_string_contains(fn, "/api/report") and W.has(fn, "report_rows") and W.has(fn, "report_compare")


def test_colocated_vocoder_timeslice():
    """Time-sliced contexts (sglang-omni @ 0029194: vocoder in its own process on gpu 0, MPS off) — switch cost is a
    published measurement (145 µs), the full-slice reading an upper bound; never fitted to the 1.70× measurement."""
    assert XS.SWITCH_S == 145e-6 and XS.SLICE_S == 2e-3
    assert XS._window_s(1e-3, "none") == 1e-3 and XS._window_s(1e-3, "switch") == 1e-3 + 2.9e-4
    assert XS._window_s(1e-3, "slice") == 2e-3 + 2.9e-4 and XS._window_s(3e-3, "slice") == 3e-3 + 2.9e-4
    co = {x["c"]: x for x in XS.colocated()}
    for c in (128, 256):
        x = co[c]
        assert x["none"] < x["switch"] < x["slice"]
    assert co[256]["switch"] < co[256]["meas"] < co[256]["slice"]           # inside the published band at c 256
    assert co[128]["slice"] < co[128]["meas"]                               # c 128: residual beyond the upper bound
    a = XS.predict("default", 128, "none")
    assert a["vocoder_s_per_request"] == 0.0                                # second GPU: no time-slicing term


def test_dory_logit_scale_counted():
    m = get_model("dory1.0-4b")
    assert m.logit_scale and m.io_misc_params == 131072 and m.param_check()["rel_err"] == 0.0
    assert m.release_params == 3975229440
    from dataclasses import replace
    ph = ir.Phase("decode", 4, 1, 1024)
    s = next(o for o in ir.head_ops(m, ph) if o.name == "sample")
    s0 = next(o for o in ir.head_ops(replace(m, logit_scale=False), ph) if o.name == "sample")
    assert s.vec == s0.vec * 4 / 3 and s.vec == 4 * m.vocab * 4                # + 1 multiply per logit
    assert not get_model("qwen3-8b").logit_scale
