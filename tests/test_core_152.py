"""0.152: cited energy presets + insight rank_per_J; multi-domain insight/path standing guards."""
import pathlib

from accel_dse import api as A
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, apply_energy_preset
from accel_dse.core.insight import DOMAIN_METRICS, upgrade_path, insight_report
from accel_dse.core.scenario import Scenario

_STATUS = {"来源", "推导", "二手", "二手/假设", "假设"}
_B = Scenario().replace("layout.tp", 2)


def test_energy_presets_labelled():
    assert set(ENERGY_PRESETS) == {"horowitz_45nm", "tpuv4i_7nm"}
    for pid, p in ENERGY_PRESETS.items():
        assert "idle_W" not in p["fields"] and "idle_W_prefill" not in p["fields"]
        for f, (v, st, src) in p["fields"].items():
            assert v > 0 and st in _STATUS and len(src) > 20, (pid, f, st)
    h = ENERGY_PRESETS["horowitz_45nm"]["fields"]
    assert h["pJ_mac"][0] == 1.1 and abs(h["pJ_bit_sram"][0] - 5 / 32) < 1e-12 and h["pJ_bit_dram"][0] == 20.0
    assert h["pJ_mac"][1] == "来源" and ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["pJ_mac"][1] == "推导"


def test_energy_preset_precedence_and_api():
    d = apply_energy_preset({"energy_preset": "tpuv4i_7nm", "pJ_mac": 0.0, "idle_W": 55.0})
    assert d["pJ_mac"] == 0.0 and d["idle_W"] == 55.0 and "idle_W" not in ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    r = A.api_eval({"scenario": {"model": "qwen3-8b"}, "energy": {"energy_preset": "horowitz_45nm"}})
    assert "pJ_mac" in r["energy"]["provided"] and r["energy"]["J_per_unit"] > 0
    r = A.api_eval({"energy": {"energy_preset": "tpuv4i_7nm", "pJ_bit_dram": 1.0}})
    # override sticks: J from dram uses 1.0 not 3.17
    assert abs(EnergyTable(**{k: v for k, v in apply_energy_preset(
        {"energy_preset": "tpuv4i_7nm", "pJ_bit_dram": 1.0}).items()
        if k in EnergyTable.__dataclass_fields__}).pJ_bit_dram - 1.0) < 1e-12
    try:
        A.api_eval({"energy": {"energy_preset": "zz"}})
    except A.ApiError:
        pass
    else:
        raise AssertionError
    assert set(A.api_catalog()["energy_presets"]) == set(ENERGY_PRESETS)


def test_insight_rank_per_J_with_preset():
    r = A.api_insight({"models": ["qwen3-8b"], "energy": {"energy_preset": "tpuv4i_7nm"}})
    assert r["energy_provided"] and isinstance(r["rank_per_J"], dict) and "tpot" in r["rank_per_J"]
    # at least one knob has a finite or win-win gain_per_J once energy is on
    assert any(k["gain_per_J"]["tpot"] is not None for k in r["knobs"])
    r0 = A.api_insight({"models": ["qwen3-8b"]})
    assert r0["energy_provided"] == [] and all(k["gain_per_J"]["tpot"] is None for k in r0["knobs"])


def test_multidomain_path_never_worse():
    cases = [
        (["wan2.1-1.3b", "flux.2-klein-4b"], "thr"),
        (["qwen-image"], "lat"),
        (["esm2-650m", "boltz-1"], "lat"),
        (["carbon-a-1.2b"], "thr"),
        (["qwen3-8b"], "tpot"),
    ]
    for ms, met in cases:
        r = upgrade_path(_B, ms, met, steps=2, beam=3)
        assert r["domain"] in DOMAIN_METRICS
        fwd = met in ("ttft", "tpot", "lat")
        for mid, v0 in r["start"].items():
            v1 = r["final"][mid]
            assert (v1 <= v0 * (1 + 1e-12)) if fwd else (v1 >= v0 * (1 - 1e-12))
        assert all(m["cum"] >= 1.0 - 1e-12 for p in r["path"] for m in p["models"])
        assert (r["path"][-1]["cum_gain"] if r["path"] else 1.0) >= r["greedy_cum_gain"] * (1 - 1e-9)


def test_ui_energy_preset_wiring():
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert 'id="e-preset"' in html and "function initEnergyPresets" in js and "energy_presets" in js
    assert "rank_per_J" in js and "每焦耳" in js
