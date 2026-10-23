"""0.353: adversarial front_cap near-ties on power-budget axis; no visual UI."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from accel_dse.core.report import design_report, to_markdown
from accel_dse.core.thermal import COOLING

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cooling_limits_not_retuned():
    assert COOLING["air_oam"]["pkg_W"] == 1000.0 and COOLING["air_oam"]["flux_W_mm2"] == 0.86
    assert COOLING["liquid"]["pkg_W"] == 1400.0 and COOLING["liquid"]["flux_W_mm2"] == 0.96
    assert COOLING["liquid"]["die_flux_W_mm2"] == 1.70


def test_front_cap_near_ties_package_scope():
    """liquid+package: binding is 功耗预算; front points sit on cap — must warn front_cap (0.352 only had margin)."""
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
               cooling="liquid", power_scope="package")
    assert o.get("power_cap_W") and o.get("power_scope") == "package"
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i["value"] for i in th["items"]}
    assert it["thermal_binding"] == "功耗预算"
    ft = th["front_thermal"]
    assert ft.get("limit_cap_W") == o["power_cap_W"]
    assert ft.get("cap_scope") == "package"
    tie = o.get("tie") or 0.01
    assert ft["n_cap_within_tie"] >= 1
    for p in ft["points"]:
        assert p["cap_W"] == p["pkg_W"]
        assert abs(p["cap_ratio"] - p["pkg_W"] / o["power_cap_W"]) < 1e-12
        assert p["cap_within_tie"] is ((1 - p["cap_ratio"]) <= tie)
        assert p["any_within_tie"] is (p["within_tie"] or p["pkg_within_tie"] or p["cap_within_tie"])
    assert sum(1 for w in rep["near_ties"] if w["kind"] == "front_cap") == ft["n_cap_within_tie"]
    md = to_markdown(rep)
    assert "预算占用" in md and "功耗预算" in md


def test_front_cap_card_scope_uses_card_W():
    """power_scope=card: cap axis is card_W (not pkg); liquid still binds on 热流, front_cap may be absent."""
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
               cooling="liquid", power_scope="card")
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i["value"] for i in th["items"]}
    assert it["thermal_binding"] == "热流"
    ft = th["front_thermal"]
    assert ft.get("cap_scope") == "card"
    for p in ft["points"]:
        row_th = o["rows"][p["row"]]["thermal"]
        assert p["cap_W"] == row_th["card_W"]
        assert abs(p["cap_ratio"] - row_th["card_W"] / o["power_cap_W"]) < 1e-12


def test_no_cap_omits_front_cap_fields():
    """No power_cap → front still reports flux/pkg; no cap_* / front_cap warnings."""
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling="liquid")
    # strip cap if example always injects one
    o = dict(o)
    o["power_cap_W"] = None
    o["power_cap_src"] = None
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    ft = th.get("front_thermal")
    assert ft is not None
    assert ft.get("limit_cap_W") is None
    assert ft.get("n_cap_within_tie") == 0
    assert all(p.get("cap_ratio") is None for p in ft["points"])
    assert not any(w["kind"] == "front_cap" for w in rep["near_ties"])


def test_changelog_353():
    assert "## [0.353.0]" in CL
    sec = CL.split("## [0.353.0]")[1].split("## [")[0]
    assert "front_cap" in sec or "前沿" in sec
    assert "budget" in sec.lower() or "功耗预算" in sec
    assert "visual" in sec.lower() or "UI" in sec or "视觉" in sec
