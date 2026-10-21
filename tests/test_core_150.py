"""0.150 hostile review of insight / upgrade path / density presets (0.146–0.149): fixed bugs + invariants."""
import pathlib

from accel_dse import api as A
from accel_dse.core.budget import DENSITY_PRESETS, Budget, apply_preset, area_estimate, full_area, phy_area
from accel_dse.core.insight import knobs, upgrade_path
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

_PHY = Budget(mm2_per_dq_lpddr=0.03, mm2_per_d2d_unit=0.4, mm2_per_GBps_c2c=0.02, mm2_fixed=10.0,
              mm2_per_mib_sram=1.0, mm2_per_kmac=0.5)


def test_max_cost_applies_in_area_mode():
    s = Scenario().replace("layout.tp", 4).replace("serving.batch", 64)
    bud = Budget(**apply_preset({"density_preset": "n7_tpuv4i", "mm2_fixed": 100.0}))
    r = upgrade_path(s, ["gpt-oss-120b"], "tpot", bud, steps=3, max_cost=3.0)
    assert r["cost_mode"] == "area" and r["max_cost"] == 3.0
    assert not r["path"] or r["path"][-1]["cum_cost"] <= 3.0 + 1e-9
    r = upgrade_path(s, ["gpt-oss-120b"], "tpot", bud, steps=3, max_cost=3.0, beam=3)
    assert not r["path"] or r["path"][-1]["cum_cost"] <= 3.0 + 1e-9


def test_no_c2c_phy_or_knob_inside_one_package():
    s = Scenario().replace("layout.tp", 2).replace("package_cards", 2).replace("d2d_enabled", True)
    t = phy_area(s, _PHY)
    assert "c2c_phy" not in t["terms"] and "d2d_phy" in t["terms"] and any("D2D" in n for n in t["notes"])
    assert "c2c_x2" not in {k for k, *_ in knobs(s)}
    s4 = s.replace("layout.tp", 4)                       # 2 packages → the link is used
    assert "c2c_phy" in phy_area(s4, _PHY)["terms"] and "c2c_x2" in {k for k, *_ in knobs(s4)}
    s1 = Scenario().replace("layout.tp", 2)              # monolithic dies: 2 cards → link
    assert "c2c_phy" in phy_area(s1, _PHY)["terms"]


def test_phy_additive_and_fixed_note():
    s = Scenario().replace("layout.tp", 4)
    a, c, p = full_area(s, _PHY), area_estimate(s.chip, _PHY), phy_area(s, _PHY)
    assert abs(a["mm2"] - c["mm2"] - p["mm2"]) < 1e-9 and not set(c["terms"]) & set(p["terms"])
    assert any("mm2_fixed" in n for n in p["notes"])


def test_preset_precedence_and_units():
    d = apply_preset({"density_preset": "n28_tpuv1", "mm2_per_mib_sram": 0.0, "mm2_per_kmac": None})
    assert d["mm2_per_mib_sram"] == 0.0 and abs(d["mm2_per_kmac"] - 330 / 4 / 64) < 1e-12   # 0 is a value; None = unset
    assert "int8" in DENSITY_PRESETS["n28_tpuv1"]["fields"]["mm2_per_kmac"][2]
    # 31.8 Mib/mm² array density ↔ 0.25 mm²/MiB (MiB = 8 Mib)
    assert abs(8 / 31.8 - 0.2516) < 1e-3 and "0.25 mm²/MiB" in DENSITY_PRESETS["n5_derived"]["fields"]["mm2_per_mib_sram"][2]


def test_beam_deterministic_and_pd_note():
    s = Scenario().replace("layout.tp", 4).replace("serving.batch", 64)
    k = lambda r: [(p["key"], p["cum_gain"]) for p in r["path"]]
    a = upgrade_path(s, ["gpt-oss-120b", "qwen3-8b"], "tpot", steps=3, max_cost=6.0, beam=3)
    b = upgrade_path(s, ["gpt-oss-120b", "qwen3-8b"], "tpot", steps=3, max_cost=6.0, beam=3)
    assert k(a) == k(b)
    r = A.api_insight_path({"scenario": {"pd": {"enabled": True}}, "models": ["qwen3-8b"], "steps": 1})
    assert any("PD" in n for n in r["notes"])
    r = A.api_insight({"scenario": {"pd": {"enabled": True}}, "models": ["qwen3-8b"]})
    assert any("PD" in n for n in r["notes"])


def test_every_domain_area_and_budget():
    for m in ("wan2.1-1.3b", "qwen-image", "esmfold", "boltz-1", "carbon-a-1.2b", "qwen3-8b"):
        r = A.api_eval({"scenario": {"model": m}, "budget": {"density_preset": "n5_derived", "mm2_fixed": 20}})
        ar = r["budget"]["area"]
        assert ar["mm2"] > 0 and "dram_phy" in ar["terms"], m


def test_ui_preset_keeps_user_values():
    js = W.read()                                      # 0.286: parsed
    assert W.has(js, "S.presetFilled[f] !== cur") and W.any_string_contains(js, "保留你的输入")


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
