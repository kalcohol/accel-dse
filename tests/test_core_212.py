"""0.212: HBM die-shoreline check; SRAM leakage as a capacity-dependent static term (no double counting)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import HBM_STACK_EDGE_MM, Budget, budget_report, shoreline_check
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario

RWKV = "rwkv7-g1k-7.2b"


def test_shoreline_levels():
    s = Scenario(chip=CHIPS["1P"])                         # hbm3e 4 stacks → 44 mm
    assert shoreline_check(s, 900.0)["status"] == "ok"     # 2 × 30 mm
    assert shoreline_check(s, 300.0)["status"] == "tight"  # 2 × 17.3 < 44 ≤ 4 × 17.3
    assert shoreline_check(s, 100.0)["status"] == "over" and "多晶粒" in shoreline_check(s, 100.0)["warning"]
    assert shoreline_check(s, None) is None
    assert shoreline_check(Scenario(chip=CHIPS["100T"]), 400.0) is None          # LPDDR: no HBM shoreline
    assert shoreline_check(s, 900.0)["need_mm"] == 4 * HBM_STACK_EDGE_MM


def test_budget_report_carries_shoreline():
    r = evaluate(Scenario(chip=CHIPS["1P"]))
    rep = budget_report(r, Budget(die_mm2=300.0))
    assert rep["shoreline"]["status"] == "tight" and rep["shoreline"]["die_mm2"] == 300.0


def _leak_table(**kw):
    v = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["uW_per_mib_sram_leak"][0]
    return EnergyTable(pJ_mac=0.4, uW_per_mib_sram_leak=v, **kw), v


def test_leak_state_buffer_always_shared_only_without_idle():
    t, v = _leak_table()
    assert abs(v - 230.6867) < 1e-3
    ch = replace(CHIPS["1P"], state_sram_mib=1024.0)
    r = evaluate(Scenario(chip=ch, model=RWKV))
    e = energy_report(r, t)
    assert abs(e["sram_leak_W_card"] - (1024 + ch.sram_mib) * v * 1e-6) < 1e-12
    t2, _ = _leak_table(idle_W=50.0)
    e2 = energy_report(r, t2)                     # measured idle already holds the shared SRAM leakage
    assert abs(e2["sram_leak_W_card"] - 1024 * v * 1e-6) < 1e-12 and "idle" in e2["J_by_action"]
    assert e2["static_share"] > 0


def test_whatif_shows_shoreline_and_leak():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]["balance_whatif"]
    rows = {x["key"]: x for x in w["rows"]}
    bw, st = rows["dram_bw_x2"], rows["state_sram"]
    assert bw["shoreline"]["stacks"] == 8 and bw["shoreline"]["status"] != "ok"
    assert bw["shoreline"]["need_mm"] > bw["shoreline"]["edge_typical_mm"]
    assert 2.0 < st["d_leak_W"] < 2.8                     # ≈ 10.1 GiB × 230.7 µW/MiB ≈ 2.39 W (25 °C lower bound)
    assert abs(bw["d_leak_W"]) < 1e-12


def test_ui_reads_new_keys():
    from pathlib import Path
    js = (Path(api.__file__).parent / "web" / "app.js").read_text()
    for k in ("shoreline", "edge_typical_mm", "need_mm", "d_leak_W", "leak_W"):
        assert k in js, k
