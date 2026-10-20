"""0.229: clock gating of idle MAC arrays (Chip.clk_ungated_share): P_clk = P_full · (g + (1 − g) · u); default g = 1
(no change from 0.228); invariant J/unit × units/s/card = W/card; fuzz."""
import random
from dataclasses import replace

from accel_dse import api as A
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Chip
from accel_dse.core.powercap import _point, dvfs_cap
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", "idle_W": 20.0, **kw}))


def _sc(g=None, model="qwen3-8b", phase="decode", b=64):
    ch = CHIPS["1P"] if g is None else replace(CHIPS["1P"], clk_ungated_share=g)
    return Scenario(chip=ch, model=model).replace("serving.phase", phase).replace("serving.batch", b)


def _units(e, cards):
    return e["units_per_window"] / e["window_s"] / cards


def test_default_is_ungated_and_hash_stable():
    assert Chip().clk_ungated_share == 1.0 and CHIPS["1P"].clk_ungated_share == 1.0
    assert _sc().hash() == _sc(1.0).hash() != _sc(0.5).hash()           # omit_default: 0.228 hashes unchanged
    assert "clk_ungated_share" not in _sc().to_dict()["chip"]
    e = energy_report(evaluate(_sc()), _t())
    ch = CHIPS["1P"]
    assert abs(e["clock_W_card"] - ch.macs * ch.freq_ghz * 1e9 * _t().pJ_clk_mac_cycle * 1e-12) < 1e-9   # = 0.228
    assert e["clock_gating"]["factor"] == 1.0


def test_validation():
    for bad in (-0.1, 1.5, True, float("nan")):
        try:
            replace(CHIPS["1P"], clk_ungated_share=bad)
        except ValueError as ex:
            assert "clk_ungated_share" in str(ex)
        else:
            raise AssertionError(bad)


def test_gating_formula_and_util():
    for g in (0.0, 0.25, 0.96):
        r = evaluate(_sc(g))
        e = energy_report(r, _t())
        u = e["clock_gating"]["mac_util"]
        assert abs(u - e["counts_per_unit"]["mac"] / e["counts_per_unit"]["clock"]) < 1e-12 and 0 < u < 0.2   # decode
        full = r.scenario.chip.macs * r.scenario.chip.freq_ghz * 1e9 * _t().pJ_clk_mac_cycle * 1e-12
        assert abs(e["clock_W_card"] - full * (g + (1 - g) * u)) < 1e-9
        assert abs(e["J_per_unit"] * _units(e, 1) / e["avg_W_per_card"] - 1) < 1e-9


def test_gating_saves_in_decode_not_prefill():
    d1, d0 = (energy_report(evaluate(_sc(g)), _t()) for g in (1.0, 0.25))
    assert d0["avg_W_per_card"] < 0.85 * d1["avg_W_per_card"]          # memory-bound: arrays mostly stalled
    p1, p0 = (energy_report(evaluate(_sc(g, phase="prefill", b=8)), _t()) for g in (1.0, 0.25))
    assert p0["clock_gating"]["mac_util"] > 0.9 and p0["avg_W_per_card"] > 0.99 * p1["avg_W_per_card"]


def test_cap_dfs_with_gating():
    # with most clock power gated, less of it falls with f → clock-only DFS sheds less at a memory-bound point
    r1, r0 = evaluate(_sc(1.0)), evaluate(_sc(0.25))
    P1, P0 = energy_report(r1, _t())["avg_W_per_card"], energy_report(r0, _t())["avg_W_per_card"]
    d1, d0 = dvfs_cap(r1, _t(), 0.9 * P1, "dfs"), dvfs_cap(r0, _t(), 0.9 * P0, "dfs")
    assert d1["throughput_factor"] > 0.99 and d0["throughput_factor"] < d1["throughput_factor"]
    q, e = _point(r0, _t(), "dfs", d0["s"])
    assert q.scenario.chip.clk_ungated_share == 0.25                   # the re-evaluated chip keeps the lever
    assert abs(e["J_per_unit"] * _units(e, 1) / e["avg_W_per_card"] - 1) < 1e-9


def test_api_and_ui():
    out = A.api_eval({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "chip": {"clk_ungated_share": 0.5}},
                      "energy": {"energy_preset": "tpuv4i_7nm"}})
    assert out["energy"]["clock_gating"]["ungated_share"] == 0.5
    assert A.SWEEP_PATHS["chip.clk_ungated_share"] is float
    from pathlib import Path
    web = Path(A.__file__).parent / "web"
    assert "c-clk_ungated_share" in W.html_ids(W.read(W.INDEX_HTML))      # 0.286: parsed
    js = W.read()
    assert "clk_ungated_share" in W.strings(js) and W.mentions(js, "clock_gating") and W.mentions(js, "mac_util")


def test_fuzz_gating():
    rng = random.Random(229)
    for _ in range(14):
        m = rng.choice(("qwen3-8b", "qwen3-32b", "rwkv7-g1k-7.2b"))
        ph = rng.choice(("decode", "prefill"))
        b = rng.choice((1, 8, 64)) if ph == "decode" else rng.choice((1, 4))
        g1, g2 = sorted(rng.uniform(0, 1) for _ in range(2))
        sc = _sc(g1, m, ph, b)
        sc = sc.replace("chip", replace(sc.chip, freq_ghz=rng.uniform(0.6, 2.0)))
        r1 = evaluate(sc)
        r2 = evaluate(sc.replace("chip", replace(sc.chip, clk_ungated_share=g2)))
        t = _t(pJ_clk_mac_cycle=rng.uniform(0.01, 0.4))
        e1, e2 = energy_report(r1, t), energy_report(r2, t)
        for e in (e1, e2):
            assert abs(e["J_per_unit"] * _units(e, r1.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-9
            assert 0 <= e["clock_gating"]["mac_util"] <= 1 and e["clock_gating"]["factor"] <= 1 + 1e-12
        assert e1["J_by_action"]["clock"] <= e2["J_by_action"]["clock"] * (1 + 1e-12)   # more gating, less clock J
        P = e1["avg_W_per_card"]
        pc = energy_report(r1, replace(t, tdp_W=rng.uniform(0.5, 0.95) * P), power_cap=True)["power_cap"]
        if pc.get("capped") and not pc["dvfs"]["dfs"].get("error"):
            assert pc["dvfs"]["dfs"]["throughput_factor"] >= pc["throughput_factor"] - 2e-3
