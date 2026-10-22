"""0.343: functional UI from ui-audit (row flex, wl-kind clear, empty scroll); visual restyle still pending."""
from __future__ import annotations

from pathlib import Path

from tests import webstruct as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def test_wl_kind_cleared_without_workload():
    js = W.APP_JS.read_text()
    assert W.has(js, "$('wl-kind').textContent = ''")
    body = W.functions(js)["paintDomain"]
    assert W.has(body, "$('wl-kind').textContent = ''")


def test_row_toolbar_flex_not_stacked():
    assert ".toolbar, .row" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "display: flex" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "flex-wrap: wrap" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "visual harmonisation" not in (ROOT / "accel_dse/web/style.css").read_text()
    assert "--fs-kpi" not in (ROOT / "accel_dse/web/style.css").read_text()
    assert "--ctl-h" not in (ROOT / "accel_dse/web/style.css").read_text()


def test_empty_scroll_table_no_padding():
    assert ".scroll:has(> table:empty)" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "padding-bottom: 0" in (ROOT / "accel_dse/web/style.css").read_text().split(
        ".scroll:has(> table:empty)")[1].split("}")[0]


def test_changelog_functional_ui_visual_pending():
    assert "## [0.343.0]" in CL
    sec = CL.split("## [0.343.0]")[1].split("## [")[0]
    assert "`.row`" in sec or "row" in sec.lower()
    assert "visual" in sec.lower()
    assert "waiting" in sec.lower() or "not applied" in sec.lower() or "Pending" in sec
    assert "wl-kind" in sec
