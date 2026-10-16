"""0.211: area / power cost of the dedicated state SRAM vs 2× DRAM bandwidth (what-if lever cost)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import DENSITY_PRESETS, Budget, area_estimate
from accel_dse.core.hardware import CHIPS

RWKV = "rwkv7-g1k-7.2b"


def test_state_sram_in_area_and_budget():
    d = DENSITY_PRESETS["n5_derived"]["fields"]["mm2_per_mib_sram"][0]
    b = Budget(mm2_per_mib_sram=d, mm2_per_mib_slc=d, mm2_per_kmac=1.0)
    a0 = area_estimate(CHIPS["1P"], b)
    a1 = area_estimate(replace(CHIPS["1P"], state_sram_mib=1024.0), b)
    assert abs(a1["terms"]["state_sram"] - 1024 * d) < 1e-9 and abs(a1["mm2"] - a0["mm2"] - 1024 * d) < 1e-6
    assert "state_sram" not in a0["terms"]
    nb = area_estimate(replace(CHIPS["1P"], state_sram_mib=1.0), Budget(mm2_per_kmac=1.0))
    assert "mm2_per_mib_sram" in nb["missing"]


def test_whatif_cost_columns_rwkv_1p():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]["balance_whatif"]
    assert w["cost_basis"]["density_preset"] == "n5_derived" and w["cost_basis"]["reticle_mm2"] == 858
    rows = {x["key"]: x for x in w["rows"]}
    st, bw = rows["state_sram"], rows["dram_bw_x2"]
    # ~10.1 GiB × 0.68 mm²/MiB ≈ 7 000 mm²: beyond one reticle; 2× HBM PHY pins: 4096 × 1/64 mm² = 64 mm²
    assert 6500 < st["d_area_mm2"] < 7600 and st["over_reticle"]
    assert abs(bw["d_area_mm2"] - 64.0) < 1.0 and not bw["over_reticle"]
    # state on chip cuts DRAM energy per token at its balance batch; 2× bandwidth moves no extra bits per token
    assert st["d_J_per_tok"] < 0 and abs(bw["d_J_per_tok"]) < 1e-12
    assert st["cost_batch"] == st["critical_batch"] and bw["cost_batch"] == bw["critical_batch"]
    assert w["base"]["area_mm2"] > 0 and all(x.get("W_card", 1) > 0 for x in w["rows"])


def test_ui_reads_cost_keys():
    from pathlib import Path
    js = (Path(api.__file__).parent / "web" / "app.js").read_text()
    for k in ("d_area_mm2", "over_reticle", "J_per_tok", "cost_batch", "base_J_per_tok_at_cost_batch", "W_card", "cost_basis"):
        assert k in js, k
