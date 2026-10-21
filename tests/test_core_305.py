"""0.305: NoC default = XY per-link routing + DRAM ingress through the NoC (memory columns); LPDDR self-refresh /
refresh power vs temperature."""
import math
from dataclasses import replace

from accel_dse import api
from accel_dse.core.energy import (LPDDR_SR_MW_T, EnergyTable, apply_energy_preset, energy_report, hbm_static_W,
                                   lpddr_refresh_factor, lpddr_sr_temp_factor)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NOC_MEM_ATTACH, System
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
LP = "lpddr5x_4x64_8533_16g"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _chip(e, side, topo="mesh", link=64.0, **kw):
    return replace(CHIPS["1P"], engines=e, rows=side, cols=side, noc_topology=topo, noc_link_Bpc=link, **kw)


def _thr(c, phase="decode", batch=64):
    return evaluate(Scenario(model=M, chip=c, serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096))).throughput


# ---------------------------------------------------------------- NoC defaults
def test_defaults_and_unset_link_unchanged():
    c = CHIPS["1P"]
    assert c.noc_routing == "xy" and c.noc_mem_attach == "edge" and NOC_MEM_ATTACH == ("edge", "none")
    assert math.isinf(c.noc_mem_GBps())                    # link unset: no NoC limit at all
    s = Scenario(model=M, chip=c)
    assert abs(System(chip=c, mem_id=s.mem_id).dram_GBps - System(chip=c, mem_id=s.mem_id).mem.effective_GBps) < 1e-9
    # presets serialise without the new fields (omit_default) → hashes unchanged
    assert "noc_routing" not in api.api_eval({"chip_preset": "1P", "scenario": {"model": M}, "sensitivity": False}).get(
        "scenario", {}).get("chip", {})
    try:
        replace(c, noc_mem_attach="top")
        raise AssertionError
    except ValueError:
        pass


def test_mem_ingress_links_and_cap():
    m = _chip(32, 128)
    assert m.noc_mem_links() == 2 * 5 and abs(m.noc_mem_GBps() - 10 * 64 * m.freq_ghz) < 1e-9
    assert replace(m, noc_topology="ring").noc_mem_links() == 4
    assert replace(m, noc_topology="crossbar").noc_mem_links() == 10
    assert math.isinf(replace(m, noc_mem_attach="none").noc_mem_GBps())
    sysm = System(chip=m, mem_id="hbm3e_4s_8h24g_9200")
    assert abs(sysm.dram_GBps - min(sysm.mem.effective_GBps, 640.0)) < 1e-9


def test_ingress_scales_with_sqrt_engines_in_decode():
    free = _thr(_chip(512, 32, link=0.0))
    t32, t512, t2048 = _thr(_chip(32, 128)), _thr(_chip(512, 32)), _thr(_chip(2048, 16))
    # memory-bound decode: delivered DRAM ∝ 2·⌊√E⌋·L until the memory's own bandwidth
    assert t32 < t512 < free and abs(t2048 / _thr(_chip(2048, 16, link=0.0)) - 1) < 1e-6
    assert abs(t512 / t32 - 22 / 5) < 0.15 * 22 / 5
    # "none" restores the 0.304 behaviour (decode unaffected by the link)
    assert abs(_thr(_chip(32, 128, noc_mem_attach="none")) / _thr(_chip(32, 128, link=0.0)) - 1) < 1e-6
    # ring: 4 ingress links regardless of E
    assert abs(_thr(_chip(32, 128, "ring")) / _thr(_chip(512, 32, "ring")) - 1) < 0.02


def test_bisection_still_selectable():
    sv = "prefill"
    b = _thr(_chip(32, 128, noc_routing="bisection", noc_mem_attach="none"), sv, 1)
    x = _thr(_chip(32, 128, noc_mem_attach="none"), sv, 1)
    assert x < b                                            # XY busiest link > bisection average
    o = api.api_insight_noc({"chip_preset": "1P", "scenario": {"model": M, "serving": {"phase": "prefill", "batch": 1,
                             "prompt": 1024}}, "energy": {"energy_preset": "tpuv4i_7nm"}, "phases": ["prefill"],
                             "topologies": ["mesh"], "noc_link_Bpc": 64})
    assert o["noc_routing"] == "xy" and any("DRAM" in n for n in o["notes"])


# ---------------------------------------------------------------- LPDDR temperature
def test_sr_temperature_curve():
    p = dict(LPDDR_SR_MW_T)
    assert abs(p[25.0] - 0.771) < 1e-9 and abs(p[85.0] - 9.99) < 1e-9
    assert abs(lpddr_sr_temp_factor(85) - 1) < 1e-12 and lpddr_sr_temp_factor(None) == 1.0
    for T, v in p.items():
        assert abs(lpddr_sr_temp_factor(T) * p[85.0] - v) < 1e-9, T
    assert lpddr_sr_temp_factor(0) == lpddr_sr_temp_factor(25)
    fs = [lpddr_sr_temp_factor(T) for T in range(0, 96, 5)]
    assert all(a <= b + 1e-15 for a, b in zip(fs, fs[1:]))
    # log-linear: geometric mean at the midpoint
    assert abs(lpddr_sr_temp_factor(35) - math.sqrt(lpddr_sr_temp_factor(25) * lpddr_sr_temp_factor(45))) < 1e-12
    assert lpddr_refresh_factor(85) == 1.0 and lpddr_refresh_factor(86) == 2.0 and lpddr_refresh_factor(None) == 1.0


def test_temperature_in_background_power():
    base = hbm_static_W(LP, _t(), 0.0, 1.0)
    assert abs(hbm_static_W(LP, _t(dram_temp_C=85), 0.0, 1.0)["W"] - base["W"]) < 1e-12    # None = 85 °C max
    cold = hbm_static_W(LP, _t(dram_temp_C=25), 0.0, 1.0)
    assert abs(cold["selfrefresh_W"] / base["selfrefresh_W"] - 0.771 / 9.99) < 1e-6
    hot = hbm_static_W(LP, _t(dram_temp_C=95), 1.0, 0.0)
    ref = hbm_static_W(LP, _t(), 1.0, 0.0)
    assert abs(hot["refresh_W"] / ref["refresh_W"] - 2) < 1e-12
    # the sram_temp_C input is used when dram_temp_C is unset; dram_temp_C wins
    assert hbm_static_W(LP, _t(sram_temp_C=25), 0.0, 1.0)["W"] == cold["W"]
    assert hbm_static_W(LP, _t(sram_temp_C=25, dram_temp_C=85), 0.0, 1.0)["W"] == base["W"]
    # HBM unaffected
    hb = "hbm3e_4s_8h24g_9200"
    assert hbm_static_W(hb, _t(dram_temp_C=25))["W"] == hbm_static_W(hb, _t())["W"]
    for bad in (-1, 96):
        try:
            _t(dram_temp_C=bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    # report: prefill (self-refresh heavy) cheaper cold; memory-bound decode (no self-refresh) unchanged ≤ 85 °C
    r = evaluate(Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(phase="prefill", batch=1, prompt=2048)))
    assert energy_report(r, _t(dram_temp_C=25))["J_per_unit"] < energy_report(r, _t())["J_per_unit"]
    r = evaluate(Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(phase="decode", batch=8)))
    assert energy_report(r, _t(dram_temp_C=25))["J_per_unit"] == energy_report(r, _t())["J_per_unit"]


def test_ui_wiring():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "e-dram_temp_C" in W.html_ids(html) and W.has(js, "'sram_temp_C', 'dram_temp_C'")
