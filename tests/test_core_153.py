"""0.153: absolute $/kWh + TDP → insight rank_per_usd; electricity presets labelled; Step-5 vision stays partial."""
import pathlib

from accel_dse import api as A
from accel_dse.core.budget import ELECTRICITY_PRESETS, Budget, apply_electricity_preset
from accel_dse.core.catalog import get_model
from accel_dse.core.insight import insight_report
from accel_dse.core.scenario import Scenario

_STATUS = {"来源", "推导", "二手", "二手/假设", "假设"}
_B = Scenario()


def test_electricity_presets_labelled_no_silent_default():
    assert set(ELECTRICITY_PRESETS) == {
        "eia_us_industrial_2024", "eia_us_commercial_2024", "cn_36city_industrial_2024_09"}
    for pid, p in ELECTRICITY_PRESETS.items():
        for f, (v, st, src) in p["fields"].items():
            assert f == "usd_per_kWh" and v > 0 and st in _STATUS and len(src) > 20, (pid, f, st)
    assert abs(ELECTRICITY_PRESETS["eia_us_industrial_2024"]["fields"]["usd_per_kWh"][0] - 0.0813) < 1e-12
    assert ELECTRICITY_PRESETS["eia_us_industrial_2024"]["fields"]["usd_per_kWh"][1] == "来源"
    assert ELECTRICITY_PRESETS["cn_36city_industrial_2024_09"]["fields"]["usd_per_kWh"][1] == "二手/假设"
    # precedence: user value wins; cleared preset key
    d = apply_electricity_preset({"electricity_preset": "eia_us_industrial_2024", "usd_per_kWh": 0.05})
    assert d["usd_per_kWh"] == 0.05 and "electricity_preset" not in d
    d0 = apply_electricity_preset({"electricity_preset": "eia_us_commercial_2024"})
    assert abs(d0["usd_per_kWh"] - 0.1275) < 1e-12
    # Budget defaults stay None — no silent rate
    assert Budget().usd_per_kWh is None and Budget().tdp_W_card is None
    r0 = A.api_insight({"models": ["qwen3-8b"]})
    assert r0["usd_per_kWh"] is None and r0["j_source"] is None and r0["rank_per_usd"]["tpot"] == []


def test_insight_rank_per_usd_energy_and_tdp():
    # energy table + electricity preset → Δ$ from ΔJ
    r = A.api_insight({"models": ["qwen3-8b"], "energy": {"energy_preset": "tpuv4i_7nm"},
                       "budget": {"electricity_preset": "eia_us_industrial_2024"}})
    assert r["j_source"] == "能耗表" and abs(r["usd_per_kWh"] - 0.0813) < 1e-9
    assert any(k["d_usd"] is not None and k["gain_per_usd"]["tpot"] is not None for k in r["knobs"])
    # TDP-only path (no energy table)
    r2 = A.api_insight({"models": ["qwen3-8b"], "budget": {"usd_per_kWh": 0.10, "tdp_W_card": 400}})
    assert r2["j_source"] == "TDP×时间" and any(k["gain_per_usd"]["tpot"] is not None for k in r2["knobs"])
    # note when TDP used without table
    assert any("tdp_W_card" in n for n in r2["notes"])
    # energy wins over TDP when both present
    r3 = A.api_insight({"models": ["qwen3-8b"], "energy": {"energy_preset": "horowitz_45nm"},
                        "budget": {"usd_per_kWh": 0.1, "tdp_W_card": 999}})
    assert r3["j_source"] == "能耗表"
    assert set(A.api_catalog()["electricity_presets"]) == set(ELECTRICITY_PRESETS)


def test_step5_vision_still_partial_documented():
    m = get_model("step-5-preview")
    assert m.coverage == "partial"
    reason = " ".join(m.coverage_reasons)
    assert "728" in reason and "504" in reason and ("外部" in reason or "external" in reason.lower())
    assert "未" in reason  # unpublished / not counted


def test_ui_cost_and_elec_preset_wiring():
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert 'id="b-elec-preset"' in html and 'id="b-usd_per_kWh"' in html and 'id="b-tdp_W_card"' in html
    assert "function initElectricityPresets" in js and "rank_per_usd" in js and "每美元" in js
    assert "不静默填电价" in html or "不静默" in html
