"""0.154: optional TCO (card_usd / pue / amort_years) → insight CapEx+OpEx ranking; PUE presets labelled; no silent defaults."""
import pathlib

from accel_dse import api as A
from accel_dse.core.budget import TCO_PRESETS, Budget, apply_tco_preset
from accel_dse.core.insight import _YEAR_S


def test_tco_presets_and_validation():
    assert set(TCO_PRESETS) == {"uptime_2024_avg", "uptime_2023_capw"}
    for p in TCO_PRESETS.values():
        assert set(p["fields"]) == {"pue"}            # no published card price / amortisation adopted
        for v, st, src in p["fields"].values():
            assert v >= 1 and st == "来源" and "Uptime" in src
    assert apply_tco_preset({"tco_preset": "uptime_2024_avg", "pue": 1.2})["pue"] == 1.2
    assert apply_tco_preset({"tco_preset": "uptime_2024_avg"})["pue"] == 1.56
    b = Budget()
    assert b.card_usd is None and b.pue is None and b.amort_years is None
    for bad in ({"pue": 0.9}, {"amort_years": 0}):
        try:
            Budget(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_insight_tco_math_and_ranking():
    bud = {"card_usd": 10000, "amort_years": 4, "usd_per_kWh": 0.1, "pue": 1.5}
    r = A.api_insight({"models": ["qwen3-8b"], "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": bud})
    m = r["models"][0]
    t = r["tco"]["qwen3-8b"]
    assert abs(t["capex"] - 10000 / (4 * _YEAR_S * m["thr"])) < 1e-15
    assert abs(t["opex"] - m["J_tok"] * 1.5 * 0.1 / 3.6e6) < 1e-18 and abs(t["tco"] - t["capex"] - t["opex"]) < 1e-18
    assert r["rank_tco"] and all(next(k for k in r["knobs"] if k["key"] == key)["tco_gain"] > 1.01 for key in r["rank_tco"])
    # CapEx only without $/kWh, and noted
    r2 = A.api_insight({"models": ["qwen3-8b"], "budget": {"card_usd": 10000, "amort_years": 4}})
    assert r2["tco"]["qwen3-8b"]["opex"] is None and any("只含 CapEx" in n for n in r2["notes"])
    # nothing filled → nothing
    r0 = A.api_insight({"models": ["qwen3-8b"]})
    assert r0["tco"] == {} and r0["rank_tco"] == []
    assert set(A.api_catalog()["tco_presets"]) == set(TCO_PRESETS)


def test_ui_tco_wiring():
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    for i in ("b-card_usd", "b-amort_years", "b-pue", "b-tco-preset"):
        assert f'id="{i}"' in html
    assert "tco_presets" in js and "rank_tco" in js and "TCO" in js
