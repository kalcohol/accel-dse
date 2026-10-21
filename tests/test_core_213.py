"""0.213: uncore area (control / NoC / routing share + host IO + optional memory controllers) in the die proxy."""
from accel_dse import api
from accel_dse.core.budget import DENSITY_PRESETS, Budget, full_area
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario

RWKV = "rwkv7-g1k-7.2b"


def _b(**kw):
    f = {k: v for k, (v, _, _) in DENSITY_PRESETS["n5_derived"]["fields"].items() if k in Budget.__dataclass_fields__}
    return Budget(**{**f, **kw})


def test_presets_carry_uncore():
    for p in DENSITY_PRESETS.values():
        assert p["fields"]["uncore_frac"][0] == 0.23 and abs(p["fields"]["mm2_host_io"][0] - 5.958) < 1e-9
        assert "mm2_per_unit_mc" not in p["fields"]


def test_uncore_share_of_final_die():
    s = Scenario(chip=CHIPS["1P"])
    a0 = full_area(s, _b(uncore_frac=None, mm2_host_io=None))
    a1 = full_area(s, _b())
    core = a0["mm2"] + a1["terms"]["host_io"]
    assert abs(a1["terms"]["uncore"] / a1["mm2"] - 0.23) < 1e-12 and abs(a1["mm2"] - core / 0.77) < 1e-9
    a2 = full_area(s, _b(mm2_per_unit_mc=2.0))           # 4 HBM stacks × 2 mm²
    assert a2["terms"]["mem_ctrl"] == 8.0


def test_mm2_fixed_suppresses_uncore():
    a = full_area(Scenario(chip=CHIPS["1P"]), _b(mm2_fixed=100.0))
    assert "uncore" not in a["terms"] and "host_io" not in a["terms"]
    assert any("mm2_fixed" in n for n in a["phy_notes"])


def test_uncore_frac_range():
    try:
        Budget(uncore_frac=1.0)
    except ValueError:
        return
    raise AssertionError("uncore_frac 1.0 accepted")


def test_rwkv_1p_die_and_shoreline_with_uncore():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]["balance_whatif"]
    b, rows = w["base"], {x["key"]: x for x in w["rows"]}
    assert 560 < b["area_mm2"] < 580 and b["shoreline"]["status"] == "ok"          # 434 → 571 mm²: 4 stacks fit 2 sides
    bw = rows["dram_bw_x2"]
    # 0.311: one die edge kept for host I/O → 8 × 11 mm > 3 × 25.6 mm = "over" (was "tight" on four edges)
    assert 640 < bw["area_mm2"] < 670 and bw["shoreline"]["status"] == "over" and not bw["over_reticle"]
    st = rows["state_sram"]
    assert st["over_reticle"] and st["area_mm2"] > 9000
    # consistency: one die area drives both the reticle flag and the shoreline
    for x in [b, *rows.values()]:
        if x.get("shoreline"):
            assert abs(x["shoreline"]["die_mm2"] - x["area_mm2"]) < 1e-9
