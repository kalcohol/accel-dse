"""0.211: area / power cost of the dedicated state SRAM vs 2× DRAM bandwidth (what-if lever cost)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import DENSITY_PRESETS, Budget, area_estimate
from accel_dse.core.hardware import CHIPS
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

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
    # 0.213: Δ area now includes the uncore share (÷ 0.77) and host IO: 7052 → 9158 mm²
    assert 8800 < st["d_area_mm2"] < 9500 and st["over_reticle"]
    assert abs(bw["d_area_mm2"] - 64.0 / 0.77) < 1.0 and not bw["over_reticle"]      # 0.213: 64 → 83.1 mm² with uncore
    # state on chip cuts DRAM energy per token at its balance batch; 2× bandwidth moves no extra bits per token
    # 0.212: J/token now includes SRAM leakage over the time window, so 2× bandwidth moves it slightly (< 0.1 %)
    # 0.216: + HBM standby/refresh of the 4 extra stacks (+3.03 W static) → +0.7 % J/token, < 1 %
    # 0.228: + per-cycle clock power (tpuv4i_7nm pJ_clk_mac_cycle): 2× bandwidth shortens the step at the same batch,
    # so clock energy per token falls — d_J was +0.7 %, now ≈ −6 % (negative, under 10 %)
    assert st["d_J_per_tok"] < 0 and -0.1 * bw["J_per_tok"] < bw["d_J_per_tok"] < 0 and abs(bw["d_hbm_W"] - 3.0336) < 1e-6
    assert st["cost_batch"] == st["critical_batch"] and bw["cost_batch"] == bw["critical_batch"]
    assert w["base"]["area_mm2"] > 0 and all(x.get("W_card", 1) > 0 for x in w["rows"])


def test_ui_reads_cost_keys():
    from pathlib import Path
    js = W.read()                                      # 0.286: exact identifiers / keys, not substrings
    for k in ("d_area_mm2", "over_reticle", "J_per_tok", "cost_batch", "base_J_per_tok_at_cost_batch", "W_card", "cost_basis"):
        assert W.mentions(js, k), k
