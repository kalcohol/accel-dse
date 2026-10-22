"""0.186: DRAM energy by memory kind (HBM vs LPDDR); preset values; fallback and invariants."""
import math

from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, apply_energy_preset, dram_pJ_field, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario

LP, HBM = "lpddr5x_4x64_8533_16g", "hbm3e_4s_12h24g_9600"


def _r(m):
    return evaluate(Scenario().replace("mem_id", m))


def test_field_selection_and_fallback():
    t0 = EnergyTable(pJ_bit_dram=5.0)
    assert dram_pJ_field(_r(LP), t0) == dram_pJ_field(_r(HBM), t0) == "pJ_bit_dram"
    t = EnergyTable(pJ_bit_dram=5.0, pJ_bit_dram_hbm=2.0)
    assert dram_pJ_field(_r(HBM), t) == "pJ_bit_dram_hbm" and dram_pJ_field(_r(LP), t) == "pJ_bit_dram"


def test_dram_energy_is_bytes_times_kind_value():
    for m, f, v in ((LP, "pJ_bit_dram_lpddr", 7.0), (HBM, "pJ_bit_dram_hbm", 3.0)):
        r = _r(m)
        e = energy_report(r, EnergyTable(pJ_bit_dram=100.0, **{f: v}))
        assert math.isclose(e["J_by_action"]["dram"], e["counts_per_unit"]["dram"] * v * 8e-12)
        assert e["dram_pJ_field"] == f


def test_old_tables_unchanged_without_kind_fields():
    r = _r(LP)
    a = energy_report(r, EnergyTable(pJ_bit_dram=3.0, pJ_mac=0.5))
    assert math.isclose(a["J_by_action"]["dram"], a["counts_per_unit"]["dram"] * 3.0 * 8e-12)


def test_only_kind_field_is_not_missing_generic():
    e = energy_report(_r(HBM), EnergyTable(pJ_bit_dram_hbm=4.0))
    assert "pJ_bit_dram" not in e["missing"] and "dram" in e["J_by_action"]
    e = energy_report(_r(LP), EnergyTable(pJ_bit_dram_hbm=4.0))     # HBM value never prices LPDDR
    assert "pJ_bit_dram" in e["missing"] and "dram" not in e.get("J_by_action", {})


def test_preset_values_and_labels():
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    # 0.333: HBM 3.44 → 5.80 pJ/bit (H200 measured active, Olgun et al. 2026; MODEL §278)
    assert abs(f["pJ_bit_dram_hbm"][0] - 5.8017342581848474) < 1e-9 and f["pJ_bit_dram_hbm"][1] == "推导"
    assert abs(f["pJ_bit_dram_lpddr"][0] - 4.0) < 1e-9 and f["pJ_bit_dram_lpddr"][1] == "推导"
    assert "pJ_bit_dram_hbm" not in ENERGY_PRESETS["horowitz_45nm"]["fields"]
    t = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))
    jl, jh = energy_report(_r(LP), t)["J_by_action"]["dram"], energy_report(_r(HBM), t)["J_by_action"]["dram"]
    # same model, same DRAM bytes: the DRAM line is bytes × the kind's pJ/bit.  ≤ 0.332 LPDDR cost more per bit (4.0 vs
    # 3.44); 0.333 HBM is the measured device-side active 5.80 → HBM's line is 5.80 / 4.0 × LPDDR's.  Caveat (MODEL §278,
    # pending): LPDDR 4.0 is an all-in subsystem figure (Grace), not a device-side dynamic one
    assert abs(jh / jl - f["pJ_bit_dram_hbm"][0] / f["pJ_bit_dram_lpddr"][0]) < 1e-9
    # whole J/token still favours HBM here (shorter step → less static / clock energy per token)
    assert energy_report(_r(HBM), t)["J_per_unit"] < energy_report(_r(LP), t)["J_per_unit"]


def test_validation():
    for bad in (-1.0, float("nan"), True):
        try:
            EnergyTable(pJ_bit_dram_lpddr=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_api_and_cli_accept_fields():
    import subprocess
    import sys
    from accel_dse import api as A
    r = A.api_eval({"scenario": {"mem_id": HBM}, "energy": {"pJ_bit_dram_hbm": 4.0}})
    assert r["energy"]["dram_pJ_field"] == "pJ_bit_dram_hbm"
    out = subprocess.run([sys.executable, "-m", "accel_dse", "eval", "--help"], capture_output=True, text=True)
    assert "--pJ-bit-dram-lpddr" in out.stdout and "--pJ-bit-dram-hbm" in out.stdout
