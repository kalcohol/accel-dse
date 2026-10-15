"""0.151: insight/path multi-domain native metrics + CSV/JSON export of ranking and upgrade path."""
import json
import pathlib
import re

from accel_dse import api as A
from accel_dse.core.insight import (
    DEFAULT_MODELS_BY_DOMAIN, DOMAIN_METRICS, insight_domain, insight_report, upgrade_path,
)
from accel_dse.core.scenario import Scenario

_B = Scenario().replace("layout.tp", 2)


def test_domains_native_metrics():
    assert insight_domain(["wan2.1-1.3b", "qwen-image"]) == "gen"
    assert insight_domain(["esm2-650m", "boltz-1"]) == "protein"
    assert insight_domain(["carbon-a-1.2b"]) == "genomics"
    for dom, models in DEFAULT_MODELS_BY_DOMAIN.items():
        r = insight_report(_B, list(models)[:2] if len(models) > 1 else list(models))
        assert r["domain"] == dom and set(r["metrics"]) == set(DOMAIN_METRICS[dom])
        assert r["default_metric"] in r["metrics"] and all(m in next(k["score"] for k in r["knobs"]) for m in r["metrics"])


def test_mixed_domain_rejected():
    try:
        A.api_insight({"models": ["wan2.1-1.3b", "qwen3-8b"]})
    except A.ApiError as e:
        assert "domain" in str(e)
    else:
        raise AssertionError
    try:
        A.api_insight_path({"models": ["esm2-650m"], "metric": "tpot"})
    except A.ApiError as e:
        assert "lat" in str(e) or "thr" in str(e)
    else:
        raise AssertionError


def test_non_llm_path_never_worse():
    for models, met in ((["wan2.1-1.3b", "flux.2-klein-4b"], "thr"),
                        (["esm2-650m"], "lat"),
                        (["carbon-a-1.2b"], "thr")):
        r = upgrade_path(_B, models, met, steps=2)
        assert r["domain"] in DOMAIN_METRICS and all(m["cum"] >= 1.0 for p in r["path"] for m in p["models"])
        fwd = met == "lat"
        for mid, v0 in r["start"].items():
            assert (r["final"][mid] <= v0) if fwd else (r["final"][mid] >= v0)


def test_llm_unchanged_defaults():
    r = A.api_insight({})
    assert r["domain"] == "llm" and r["default_metric"] == "tpot" and "ttft" in r["metrics"]
    assert Scenario().hash() == "723894c17df5d7bb"


def test_export_wiring_and_schema():
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert re.search(r"const EXP_SCHEMA = 1;", js)          # envelope unchanged; new kinds under same schema
    assert "kind: 'insight'" in js and "kind: 'insight_path'" in js
    assert "function setInsightExport" in js and "function setInsightPathExport" in js
    assert 'id="ins-exp-csv"' in html and 'id="ins-path-exp-json"' in html
    assert "function paintInsMetric" in js
    # README blurbs (no version bullets)
    root = pathlib.Path(A.__file__).resolve().parents[1]
    zh, en = (root / "README.md").read_text(), (root / "README.en.md").read_text()
    assert "## 设计洞察" in zh and "/api/insight_path" in zh and "0.151" not in zh.split("## 设计洞察")[1].split("##")[0]
    assert "## Design insight" in en and "frames/s" in en and "0.151" not in en.split("## Design insight")[1].split("##")[0]


def test_export_json_shape_via_columns():
    """The export envelope (tool/schema/kind/columns/rows) matches compare/sweep; column keys are stable."""
    r = A.api_insight({"models": ["flux.2-klein-4b", "qwen-image"]})
    assert r["domain"] == "gen"
    cols = ["rank", "key", "label", "score", "d_mm2", "gain_per_mm2", "cost", "inverted"]
    rows = sorted(r["knobs"], key=lambda k: -(k["score"]["thr"] or 0))
    payload = {"tool": "accel-dse", "schema": 1, "kind": "insight",
               "columns": [{"key": c} for c in cols],
               "rows": [{"rank": i + 1, "key": k["key"], "score": k["score"]["thr"]} for i, k in enumerate(rows)]}
    assert json.loads(json.dumps(payload))["schema"] == 1 and payload["rows"][0]["key"] == r["rank"]["thr"][0]
