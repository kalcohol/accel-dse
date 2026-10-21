"""0.264: 25/85 °C SLC energy packaged into default insight_report; UI apply hooks."""
from dataclasses import replace
from pathlib import Path

from accel_dse import api
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import INSIGHT_SLC_GIB, insight_report
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-8b"
ROOT = Path(__file__).resolve().parents[1]


def _t(**kw):
    t = EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))
    # 0.300: the presets now fill the on-chip wire + NoC router; this file pins the ≤ 0.299 preset (no on-chip
    # movement terms) unless a test passes them
    return replace(t, **{k: None for k in ("pJ_bit_mm_onchip", "pJ_bit_hop_noc") if k not in kw})


def _base():
    return Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8))


def test_insight_embeds_slc_temp_when_leak_set():
    r = insight_report(_base(), [M], table=_t())
    st = r["slc_temp"]
    assert st and st["temps"] == [25.0, 85.0] and st["gibs"] == list(INSIGHT_SLC_GIB)
    assert st["leak_charged"] and st["recommend"]
    rec = st["recommend"][0]
    assert rec["model"] == M and rec["hot_cuts_J_opt"] is True
    assert rec["opt_J_gib_25C"] == 32.0 and rec["opt_J_gib_85C"] == 16.0
    assert rec["crossover_gib_85C"] == 1.0
    assert any("SLC" in n for n in r["notes"])


def test_insight_skips_slc_temp_without_leak():
    r = insight_report(_base(), [M], table=EnergyTable())
    assert r["slc_temp"] is None


def test_api_insight_slc_temp_and_ui():
    out = api.api_insight({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"},
                                        "serving": {"batch": 8}},
                           "energy": {"energy_preset": "tpuv4i_7nm", "pJ_bit_mm_onchip": 0, "pJ_bit_hop_noc": 0},
                           "models": [M]})
    assert out["slc_temp"] and out["slc_temp"]["recommend"][0]["hot_cuts_J_opt"]
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    assert "ins-slc-temp" in W.html_ids(html) and "paintInsightSlcTemp" in W.functions(js)
    assert W.any_string_contains(js, "应用") and W.mentions(js, "opt_J_gib_25C")
