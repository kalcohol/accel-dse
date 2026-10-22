"""0.187: primary-sourced DRAM pJ/bit; sizing energy exposure and energy-constrained sizing."""
from accel_dse import api as A
from accel_dse.core.energy import ENERGY_PRESETS
from accel_dse.core.insight import sizing_report
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

E = {"energy_preset": "tpuv4i_7nm", "pJ_bit_mm_onchip": 0, "pJ_bit_hop_noc": 0}   # 0.300: ≤ 0.299 preset (no on-chip movement)


def test_dram_presets_primary():
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    # 0.333: 3.44 (SK hynix IEDM 2023 HBM2E × 0.8) → 5.80 (H200 measured active; FGDRAM kept as cross-check)
    assert abs(f["pJ_bit_dram_hbm"][0] - 5.8017342581848474) < 1e-9 and "2610.14244" in f["pJ_bit_dram_hbm"][2]
    # 0.335: 4.0 (Grace all-in) → 3.23 (Micron Y52Q IDD4R−IDD2N); Grace kept as cross-check in the source text
    assert abs(f["pJ_bit_dram_lpddr"][0] - 3.2298828125000005) < 1e-9 and "IDD4R" in f["pJ_bit_dram_lpddr"][2]
    assert "Grace" in f["pJ_bit_dram_lpddr"][2] and "4.0" in f["pJ_bit_dram_lpddr"][2]
    h = ENERGY_PRESETS["horowitz_45nm"]["fields"]
    assert "pJ_bit_dram_hbm" not in h and "pJ_bit_dram_lpddr" not in h      # no 45 nm-era per-kind source


def test_traffic_exposed_without_table():
    r = sizing_report(Scenario(), ["wan2.1-1.3b"], resources=["sram"])["resources"][0]
    t = r["traffic"]["wan2.1-1.3b"]
    assert t["dram_B"] >= 1.0 and "J" not in t                 # shrinking SRAM never lowers DRAM bytes


def test_energy_tie_only_tightens():
    for m in ("wan2.1-1.3b", "esm2-650m"):
        a = A.api_insight_sizing({"models": [m], "energy": E, "resources": ["sram", "dequant"]})
        b = A.api_insight_sizing({"models": [m], "energy": E, "resources": ["sram", "dequant"], "energy_tie": True})
        for x, y in zip(a["resources"], b["resources"]):
            assert y["min_ok"] >= x["min_ok"]                    # an extra constraint can never allow more shrink
            assert y["traffic"][m]["J"] <= 1.01 + 1e-9           # the contract holds at the reported point
    w = A.api_insight_sizing({"models": ["wan2.1-1.3b"], "energy": E, "resources": ["sram"], "energy_tie": True})
    assert w["resources"][0]["limited_by"]["metric"] == "J_tok" and w["energy_tie"] is True


def test_energy_tie_requires_table():
    try:
        sizing_report(Scenario(), ["qwen3-8b"], resources=["sram"], energy_tie=True)
    except ValueError:
        return
    raise AssertionError("energy_tie without a table must be rejected")


def test_api_rejects_non_bool_and_ui_wiring():
    from pathlib import Path
    r = A.api_insight_sizing({"models": ["qwen3-8b"], "resources": ["sram"], "energy_tie": "yes"})
    assert r["energy_tie"] is False                              # only literal true enables it
    web = Path(A.__file__).parent / "web"
    assert "ins-size-energy" in W.html_ids(W.read(W.INDEX_HTML)) and "trafficCell" in W.functions(W.read())   # 0.286: parsed
