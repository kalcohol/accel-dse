"""0.262: J/token-optimal SLC size under energy.sram_temp_C (insight.slc_energy_report); static route for insight_slc."""
from dataclasses import replace

from accel_dse import api, serve
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import SLC_GIB_GRID, slc_energy_report
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-30b-a3b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _base():
    return Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8))


def test_cold_die_prefers_large_slc_for_energy():
    r = slc_energy_report(_base(), _t(), [M], gibs=(0, 8, 32, 64))
    m = r["models"][0]
    assert r["leak_charged"] and r["sram_temp_C"] == 25.0
    assert m["opt_J_gib"] == 64.0 and m["crossover_gib"] is None
    assert m["opt_J"] < m["J0"]
    js = [c["J_tok"] for c in m["curve"] if c.get("J_tok") is not None]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(js, js[1:]))          # monotone down at 25 °C on this grid


def test_hot_die_crossover_and_time_trade():
    r = slc_energy_report(_base(), _t(sram_temp_C=85), [M], gibs=(0, 4, 8, 16, 32))
    m = r["models"][0]
    assert m["opt_J_gib"] == 0.0 and m["crossover_gib"] == 4.0
    assert m["opt_time_gib"] == 32.0 and m["opt_time"] < next(c["tpot_ms"] for c in m["curve"] if c["gib"] == 0)
    # at the crossover the leak alone already exceeds the DRAM energy saved vs none
    c4 = next(c for c in m["curve"] if c["gib"] == 4)
    assert c4["J_tok"] > m["J0"] and c4["leak_W"] > 1.0


def test_idle_still_charges_slc_leak():
    r = slc_energy_report(_base(), _t(idle_W=20.0, sram_temp_C=85), [M], gibs=(0, 1, 32))
    assert r["leak_charged"] and any("idle" in n or "增量" in n for n in r["notes"])
    m = r["models"][0]
    assert m["crossover_gib"] == 1.0
    c0 = next(c for c in m["curve"] if c["gib"] == 0)
    c1 = next(c for c in m["curve"] if c["gib"] == 1)
    assert c1["J_tok"] > c0["J_tok"] and (c1.get("leak_W") or 0) > 0


def test_api_and_route():
    assert "/api/insight_slc" in serve.POST_ROUTES
    out = api.api_insight_slc({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"},
                                            "serving": {"batch": 8}},
                               "energy": {"energy_preset": "tpuv4i_7nm", "sram_temp_C": 85},
                               "gibs": [0, 4, 32]})
    assert out["models"][0]["opt_J_gib"] == 0.0 and out["models"][0]["crossover_gib"] == 4.0
    assert out["hash"]


def test_grid_default_and_policy():
    assert 0 in SLC_GIB_GRID and 64 in SLC_GIB_GRID
    try:
        slc_energy_report(_base(), _t(), [M], policy="nope")
        raise AssertionError
    except ValueError:
        pass


def test_ui_hooks():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed
    assert "ins-slc-run" in W.html_ids(html)
    assert [c["fn"] for c in W.req_calls(js) if c["path"] == "/api/insight_slc"] == ["runInsightSlc"]
