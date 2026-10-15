"""0.162: sourced/derived vector-lane and dequant densities priced in sizing; unpriced-resource modes."""
from pathlib import Path

from accel_dse import api as A
from accel_dse.core.budget import Budget, DENSITY_PRESETS, apply_preset, area_estimate
from accel_dse.core.insight import sizing_report
from accel_dse.core.scenario import Scenario


def test_presets_carry_labelled_lane_and_dequant_densities():
    for pid, p in DENSITY_PRESETS.items():
        f = p["fields"]
        kmac = f["mm2_per_kmac"][0]
        for k, r in (("mm2_per_klane", 11884 / 5824), ("mm2_per_kdq", 1676 / 5824)):
            v, st, src = f[k]
            assert abs(v - kmac * r) < 1e-12 and "推导" in st and "二手" in st and "Horowitz" in src and "lower bound" in src
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i"}))
    assert b.mm2_per_klane and b.mm2_per_kdq


def test_area_proxy_unchanged_by_new_fields():
    ch = Scenario().chip
    base = dict(mm2_per_kmac=0.7, mm2_per_mib_sram=0.9)
    a0 = area_estimate(ch, Budget(**base))
    a1 = area_estimate(ch, Budget(**base, mm2_per_klane=1.4, mm2_per_kdq=0.2))
    assert a0 == a1                                   # sizing-only densities: die-area proxy / fingerprints untouched


def test_lanes_and_dequant_priced_and_units():
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i"}))
    r = sizing_report(Scenario(), ["esm2-650m"], b, resources=["lanes", "dequant", "cores"])
    ch = Scenario().chip
    for x in r["resources"]:
        per = {"lanes": b.mm2_per_klane / 1024, "dequant": b.mm2_per_kdq / 1024,
               "cores": ch.rows * ch.cols / 1024 * b.mm2_per_kmac}[x["key"]]
        assert abs(x["mm2_saved"] - (x["current"] - x["min_ok"]) * per) < 1e-9


def test_unpriced_modes():
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i"}))
    last = sizing_report(Scenario(), ["wan2.1-1.3b"], b, joint=True)["joint"]
    free = sizing_report(Scenario(), ["wan2.1-1.3b"], b, joint=True, unpriced="free")["joint"]
    assert last["unpriced_mode"] == "last" and free["unpriced_mode"] == "free"
    assert last["unpriced"] == ["dram_bw"] and last["holds"] and free["holds"]
    bw = lambda j: next(e["joint"] for e in j["resources"] if e["key"] == "dram_bw")
    assert bw(free) >= bw(last) - 1e-9           # 'free' gives the unpriced bandwidth back first; 'last' defers it
    try:
        sizing_report(Scenario(), ["wan2.1-1.3b"], b, joint=True, unpriced="bogus")
        raise AssertionError("bad mode accepted")
    except ValueError:
        pass
    r = A.api_insight_sizing({"models": ["esm2-650m"], "joint": True, "unpriced": "free",
                              "budget": {"density_preset": "n5_derived"}})
    assert r["joint"]["unpriced_mode"] == "free"
    assert "ins-size-unpriced" in (Path(A.__file__).parent / "web" / "index.html").read_text()
