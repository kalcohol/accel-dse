"""0.359: DVFS must not scale HBM/LPDDR device static with the card-W ratio.

The old scale (dram_dev_W * W_capped / W_nominal) dropped stack static below the
clock-held floor and assigned that heat to the die. Card flux and the power-cap
axis (thermal_binding, front_cap) stay on pkg_W / card_W.
"""
from __future__ import annotations

from accel_dse import api
from accel_dse.core.budget import DENSITY_PRESETS, Budget, shoreline_report
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, energy_report
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import F_MIN, _point, dram_device_W, dvfs_cap, hbm_device_static_W
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.thermal import cap_power_fn, dram_dev_W, thermal_report

MEM = "hbm3e_8s_8h24g_9200"


def _table():
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    return EnergyTable.from_dict({k: v for k, (v, _, _) in f.items()
                                  if k in EnergyTable.__dataclass_fields__
                                  and k not in ("idle_W", "idle_frac_tdp", "tdp_W")})


def _scn():
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=MEM,
                    serving=Serving(phase="prefill", batch=1, prompt=2048, ctx=4096))


def _budget(cap: float) -> Budget:
    f = DENSITY_PRESETS["n5_derived"]["fields"]
    return Budget(**{k: v for k, (v, _, _) in f.items() if k in Budget.__dataclass_fields__},
                  power_W_card=cap)


def test_capped_dram_device_holds_hbm_static():
    t = _table()
    s = _scn()
    r = evaluate(s)
    er = energy_report(r, t)
    a = shoreline_report(s)["die_mm2"]
    pk = cap_power_fn(a, 1, "package")
    d = dvfs_cap(r, t, 450.0, "dfs", pk=pk)
    assert d.get("binds") and d.get("feasible", True) and d["duty"] == 1.0
    _q, e = _point(r, t, "dfs", d["s"])
    dev = hbm_device_static_W(e)
    assert dev > 40.0
    assert abs(d["dram_dev_W"] - dram_dev_W(e)) < 1e-9
    assert d["dram_dev_W"] >= dev - 1e-9
    scaled = dram_dev_W(er) * d["W_card"] / er["avg_W_per_card"]
    assert scaled < dev
    assert d["dram_dev_W"] - scaled > 10.0
    kw = dict(standby_W=e["hbm_static"]["standby_W"], device_static_W=dev)
    th_old = thermal_report(d["W_card"], a, 1, scaled, "air_oam", d["static_W"], None, **kw)
    th = thermal_report(d["W_card"], a, 1, d["dram_dev_W"], "air_oam", d["static_W"], None, **kw)
    assert th["die_W"] < th_old["die_W"] - 10.0
    assert abs(th["flux_W_mm2"] - th_old["flux_W_mm2"]) < 1e-9
    assert abs(th["power"]["delivered_W"] - d["dram_dev_W"] - th["die_W"]) < 1e-6


def test_duty_cycle_holds_device_static():
    t = _table()
    r = evaluate(_scn())
    er = energy_report(r, t)
    d = dvfs_cap(r, t, 80.0, "dfs", pk=None)
    assert d.get("binds") and d.get("feasible", True) and d["duty"] < 1.0
    _q, lo = _point(r, t, "dfs", F_MIN)
    slow = 1.0 / d["duty"]
    assert abs(d["dram_dev_W"] - dram_device_W(lo, slow)) < 1e-9
    assert d["dram_dev_W"] + 1e-6 >= hbm_device_static_W(lo)
    assert d["dram_dev_W"] > dram_dev_W(er) * d["W_card"] / er["avg_W_per_card"]


def test_dse_row_and_eval_use_capped_device_watts():
    t = _table()
    s = _scn()
    ch = s.chip
    cap = 450.0
    o = energy_design_search(s, t, _budget(cap), mac_scales=(1.0,), sram_scales=(1.0,),
                             sides=(ch.rows,), topologies=(ch.noc_topology,), mems=(MEM,),
                             dies=(1,), mono_clusters=False, cooling="air_oam", power_scope="package")
    capped = [x for x in o["rows"] if x.get("capped")]
    assert len(capped) == 1
    x = capped[0]
    assert x["dram_dev_W"] >= x["device_static_W"] - 1e-6
    delivered = x["thermal"]["card_W"] + x["thermal"]["static_add_W"]
    assert abs(delivered - x["dram_dev_W"] - x["thermal"]["die_W"]) < 1e-4
    r = evaluate(s)
    _q, e = _point(r, t, "dfs", x["f_scale"])
    assert abs(x["dram_dev_W"] - dram_dev_W(e)) < 1e-4
    body = {"chip_preset": "1P", "power_cap": True, "power_scope": "package", "cooling": "air_oam",
            "scenario": {"model": "qwen3-8b", "mem_id": MEM,
                         "serving": {"phase": "prefill", "batch": 1, "prompt": 2048, "ctx": 4096}},
            "energy": {"energy_preset": "tpuv4i_7nm"},
            "budget": {"power_W_card": cap, "density_preset": "n5_derived"},
            "sensitivity": False}
    ev = api.api_eval(body)
    dfs = ev["energy"]["power_cap"]["dvfs"]["dfs"]
    th = ev["thermal"]
    assert th.get("operating_point") == "capped"
    assert abs(th["dram_dev_W"] - dfs["dram_dev_W"]) < 1e-6
    assert th["dram_dev_W"] >= hbm_device_static_W(ev["energy"]) - 1e-6
    assert abs(th["power"]["delivered_W"] - th["dram_dev_W"] - th["die_W"]) < 1e-4
    nominal = dram_dev_W(ev["energy"])
    scaled = nominal * dfs["W_card"] / ev["energy"]["avg_W_per_card"]
    assert th["dram_dev_W"] > scaled + 10.0
