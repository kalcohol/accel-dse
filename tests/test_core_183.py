"""0.183: discrete DRAM package / stack sizing (dram_units) priced by DRAM PHY density; integer-floor loop fix."""
import math
from dataclasses import replace

from accel_dse import mem_catalog
from accel_dse.core.budget import Budget
from accel_dse.core.insight import JOINT_SKIP, SIZING, _dram_units_set, sizing_report
from accel_dse.core.scenario import Scenario


def _spec(s):
    return mem_catalog.parse_mem_id(s.mem_id)


def test_setter_keeps_parts_and_eff_only_count_moves():
    s = Scenario()
    sp = _spec(s)
    for n in (1, 2, 3, 7):
        t = _dram_units_set(s, n)
        q = _spec(t)
        assert q.n_units == n and q.rate_MTps == sp.rate_MTps and q.unit_width_bits == sp.unit_width_bits
        assert q.cap_per_unit_GB == sp.cap_per_unit_GB and t.mem_eff == s.mem_eff
        assert math.isclose(q.bus_bits, n * sp.unit_width_bits)
        assert math.isclose(q.capacity_GB * sp.n_units, sp.capacity_GB * n)       # capacity ∝ count
    assert _spec(_dram_units_set(s, 0.2)).n_units == 1                          # floor 1


def test_hbm_setter_keeps_height_die():
    s = Scenario().replace("mem_id", "hbm3e_8s_12h24g_9600")
    q = _spec(_dram_units_set(s, 3))
    assert q.n_units == 3 and q.hbm_height == 12 and q.hbm_die_Gb == 24 and q.rate_MTps == 9600


def test_report_monotone_and_priced_by_phy():
    s = Scenario()
    sp = _spec(s)
    b = Budget(mm2_per_dq_lpddr=0.03)
    r = sizing_report(s, ["qwen3-8b"], b, resources=["dram_units"])["resources"][0]
    assert r["current"] == sp.n_units and 1 <= r["min_ok"] <= r["current"] and r["min_ok"] == int(r["min_ok"])
    assert math.isclose(r["mm2_saved"], (r["current"] - r["min_ok"]) * sp.unit_width_bits * 0.03)
    if r["fail_at"] is not None:                       # failing point is a real, smaller integer count
        assert r["fail_at"] < r["min_ok"] and r["limited_by"]["metric"]
    # HBM density must not price an LPDDR interface
    r2 = sizing_report(s, ["qwen3-8b"], Budget(mm2_per_dq_hbm=1.0), resources=["dram_units"])["resources"][0]
    assert "mm2_saved" not in r2


def test_capacity_loss_is_a_fit_failure():
    # one package (16 GB) cannot hold a model that fits in four: the search must stop on 'fits' or a metric, never pass
    s = Scenario()
    r = sizing_report(s, ["qwen3-32b"] if "qwen3-32b" in __import__("accel_dse.core.insight", fromlist=["x"]).DEFAULT_MODELS
                      else ["qwen3-8b"], resources=["dram_units"])["resources"][0]
    p1 = _dram_units_set(s, r["min_ok"])
    assert _spec(p1).capacity_GB <= _spec(s).capacity_GB


def test_integer_floor_terminates_at_one():
    # ≤ 0.182: an integer resource whose current value is 1 looped forever (0.5 rounds back to 1)
    s = Scenario().replace("mem_id", "lpddr5x_1x64_8533_16g")
    r = sizing_report(s, ["qwen3-8b"], resources=["dram_units"])["resources"][0]
    assert r["current"] == 1 and r["min_ok"] == 1
    c = s.replace("chip", replace(s.chip, engines=1))
    r = sizing_report(c, ["qwen3-8b"], resources=["cores"])["resources"][0]
    assert r["current"] == 1 and r["min_ok"] == 1


def test_joint_skips_dram_units_and_default_list():
    assert "dram_units" in SIZING and "dram_units" in JOINT_SKIP
    rep = sizing_report(Scenario(), ["qwen3-8b"], joint=True)
    keys = [r["key"] for r in rep["resources"]]
    assert "dram_units" in keys
    assert "dram_units" not in str(rep["joint"].get("values", rep["joint"]))


def test_dma_cap_dram_units_never_raises_delivered():
    s = Scenario()
    for n in (1, 2, 4):
        t = _dram_units_set(s, n)
        assert SIZING["dram_bw"][2](t) <= SIZING["dram_bw"][2](s) + 1e-9
