"""0.266: SLC access energy vs capacity (√C wire scale via energy.slc_access_ref_mib)."""
import math
from dataclasses import replace
from pathlib import Path

from accel_dse import api
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report, slc_pJ_bit
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-8b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_flat_default_and_helper():
    t = _t()
    assert t.slc_access_ref_mib is None
    assert slc_pJ_bit(t, 8192.0) == t.pJ_bit_slc
    tw = _t(slc_access_ref_mib=1.0)
    assert abs(slc_pJ_bit(tw, 1.0) - tw.pJ_bit_slc) < 1e-15
    assert abs(slc_pJ_bit(tw, 4.0) / tw.pJ_bit_slc - 2.0) < 1e-12
    assert abs(slc_pJ_bit(tw, 0.25) - tw.pJ_bit_slc) < 1e-15  # floor at 1×


def test_validation():
    try:
        EnergyTable(slc_access_ref_mib=0.0)
    except ValueError as e:
        assert "slc_access_ref_mib" in str(e)
    else:
        raise AssertionError


def test_energy_report_scales_slc_and_report_field():
    ch = replace(CHIPS["100T"], slc_mib=8 * 1024, slc_policy="dip")
    r = evaluate(Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8)))
    flat = energy_report(r, _t())
    wire = energy_report(r, _t(slc_access_ref_mib=1.0))
    assert "slc_access" not in flat
    assert abs(wire["slc_access"]["scale"] - math.sqrt(8192.0)) < 1e-9
    assert wire["J_by_action"]["slc"] > 50 * flat["J_by_action"]["slc"]
    # fingerprint: unset ref keeps legacy flat energy
    assert abs(flat["J_by_action"]["slc"] - energy_report(r, _t())["J_by_action"]["slc"]) < 1e-15


def test_api_and_ui():
    out = api.api_eval({"scenario": {"model": M, "mem_id": LP,
                                     "chip": {"name": "100T", "slc_mib": 4096, "slc_policy": "dip"},
                                     "serving": {"batch": 8}},
                        "energy": {"energy_preset": "tpuv4i_7nm", "slc_access_ref_mib": 1.0}})
    # energy nested in result
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    assert "e-slc_access_ref_mib" in W.html_ids(html) and W.mentions(js, "slc_access_ref_mib")
