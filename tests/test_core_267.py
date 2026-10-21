"""0.267: distance-based SLC wire energy (energy.pJ_bit_mm_onchip, slc_mm2_per_mib)."""
import math
from dataclasses import replace
from pathlib import Path

from accel_dse.core import pdqueue
from accel_dse.core.energy import (SLC_MM2_PER_MIB_DEFAULT, EnergyTable, apply_energy_preset, energy_report,
                                   slc_pJ_bit, slc_wire_mm)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import slc_energy_report
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-30b-a3b"


def _t(**kw):
    t = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))
    # 0.300: the presets now fill the on-chip wire + NoC router; this file pins the ≤ 0.299 preset (no on-chip
    # movement terms) unless a test passes them
    return replace(t, **{k: None for k in ("pJ_bit_mm_onchip", "pJ_bit_hop_noc") if k not in kw})


def test_distance_formula():
    t = _t(pJ_bit_mm_onchip=0.078)
    assert abs(SLC_MM2_PER_MIB_DEFAULT - 8.388608 / 35.5) < 1e-15
    d = slc_wire_mm(t, 8192)
    assert abs(d - 0.5 * math.sqrt(8192 * SLC_MM2_PER_MIB_DEFAULT)) < 1e-12
    assert abs(slc_pJ_bit(t, 8192) - (t.pJ_bit_slc + 0.078 * d)) < 1e-12
    assert slc_pJ_bit(t, 0) == t.pJ_bit_slc
    t2 = _t(pJ_bit_mm_onchip=0.078, slc_mm2_per_mib=4 * SLC_MM2_PER_MIB_DEFAULT)
    assert abs(slc_wire_mm(t2, 8192) - 2 * d) < 1e-9          # 4× area → 2× distance


def test_validation_exclusive_and_density():
    for kw in ({"pJ_bit_mm_onchip": 0.1, "slc_access_ref_mib": 1.0}, {"slc_mm2_per_mib": 0.0}):
        try:
            EnergyTable(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)


def test_report_field_and_pd_parity():
    ch = replace(CHIPS["100T"], slc_mib=32 * 1024, slc_policy="dip")
    scn = Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8))
    t = _t(pJ_bit_mm_onchip=0.078)
    e = energy_report(evaluate(scn), t)
    sa = e["slc_access"]
    assert abs(sa["wire_mm"] - 44.0) < 0.1 and sa["over_reticle"] is True
    pd = pdqueue._energy({"slc": 1e6}, 0.0, t, pools=((scn, 0.0),), pool_names=("decode",))
    assert abs(pd["J_by_action"]["slc"] - 1e6 * sa["pJ_bit"] * 1e-12 * 8) < 1e-15


def test_insight_curve_distance_vs_sqrtc():
    base = Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8))
    g = (0, 8, 32)
    flat = slc_energy_report(base, _t(), [M], gibs=g)["models"][0]
    wire = slc_energy_report(base, _t(pJ_bit_mm_onchip=0.078), [M], gibs=g)["models"][0]
    sq = slc_energy_report(base, _t(slc_access_ref_mib=1.0), [M], gibs=g)["models"][0]
    assert flat["opt_J_gib"] == wire["opt_J_gib"] == 32.0 and sq["opt_J_gib"] == 0.0
    j = lambda m, x: next(c["J_tok"] for c in m["curve"] if c["gib"] == x)
    assert j(flat, 32) < j(wire, 32) < j(wire, 0)
    assert wire["curve"][-1]["slc_wire_mm"] > 40


def test_ui():
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    assert {"e-pJ_bit_mm_onchip", "e-slc_mm2_per_mib"} <= W.html_ids(html)
    assert "pJ_bit_mm_onchip" in W.strings(js) and W.mentions(js, "wire_mm")
