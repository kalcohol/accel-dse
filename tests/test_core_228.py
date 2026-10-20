"""0.228: per-cycle dynamic energy (clock tree / always-clocked registers, pJ_clk_mac_cycle): paid every core cycle,
busy or stalled; falls with f under clock-only DFS; scales with V² on the core rail; J/unit × units/s/card = W/card."""
import random
from dataclasses import replace

from accel_dse import api as A
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import CORE_ACTIONS, _point, dvfs_cap, volt
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", "idle_W": 20.0, **kw}))


def _sc(model="qwen3-8b", phase="decode", b=64):
    return Scenario(chip=CHIPS["1P"], model=model).replace("serving.phase", phase).replace("serving.batch", b)


def _units(e, cards):
    return e["units_per_window"] / e["window_s"] / cards


def test_preset_value_and_label():
    v, tag, src = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["pJ_clk_mac_cycle"]
    # (121.7 − 71.8) W ÷ (989.4e12 / 2 MAC/s) — H100 SXM with / without a CUDA context
    assert abs(v - 49.9 / (989.4e12 / 2) * 1e12) < 1e-12 and 0.100 < v < 0.102
    assert tag == "推导/二手" and "121.7" in src and "71.8" in src
    assert "pJ_clk_mac_cycle" not in ENERGY_PRESETS["horowitz_45nm"]["fields"]     # no 45 nm source: not filled
    assert "pJ_clk_mac_cycle" in CORE_ACTIONS.values()


def test_clock_term_is_per_cycle_and_dynamic():
    r = evaluate(_sc())
    ch = r.scenario.chip
    e = energy_report(r, _t())
    W = ch.macs * ch.freq_ghz * 1e9 * _t().pJ_clk_mac_cycle * 1e-12
    assert abs(e["clock_W_card"] - W) < 1e-9
    assert abs(e["J_by_action"]["clock"] - W * e["window_s"] * r.scenario.layout.cards / e["units_per_window"]) < 1e-12
    assert abs(e["counts_per_unit"]["clock"] * _t().pJ_clk_mac_cycle * 1e-12 - e["J_by_action"]["clock"]) < 1e-15
    # dynamic, not static: static_share excludes it
    stat = sum(e["J_by_action"].get(k, 0.0) for k in ("idle", "sram_leak", "hbm_static"))
    assert abs(e["static_share"] - stat / e["J_per_unit"]) < 1e-12
    e0 = energy_report(r, replace(_t(), pJ_clk_mac_cycle=None))
    assert "clock" not in e0["J_by_action"] and "clock_W_card" not in e0
    assert abs(e["avg_W_per_card"] - e0["avg_W_per_card"] - W) < 1e-6
    assert "pJ_clk_mac_cycle" not in e0["missing"]


def test_clock_follows_f_and_v2():
    r = evaluate(_sc())
    for curve in ("dfs", "rx5700xt_n7"):
        for s in (0.3, 0.6, 1.0):
            q, e = _point(r, _t(), curve, s)
            ch = q.scenario.chip
            assert abs(ch.freq_ghz - r.scenario.chip.freq_ghz * s) < 1e-12
            want = ch.macs * ch.freq_ghz * 1e9 * _t().pJ_clk_mac_cycle * volt(curve, s) ** 2 * 1e-12
            assert abs(e["clock_W_card"] - want) < 1e-9


def test_memory_bound_dfs_now_sheds_power():
    # DRAM-bound decode: the op rate is set by memory; only the per-cycle term falls with f → clock-only DFS sheds
    # power without losing throughput; without the term it needed s ≈ 0.36 and lost 12 % (0.227)
    r = evaluate(_sc())
    assert r.bound == "DRAM"
    t, t0 = _t(), replace(_t(), pJ_clk_mac_cycle=None)
    P, P0 = energy_report(r, t)["avg_W_per_card"], energy_report(r, t0)["avg_W_per_card"]
    d, d0 = dvfs_cap(r, t, 0.9 * P, "dfs"), dvfs_cap(r, t0, 0.9 * P0, "dfs")
    assert d["throughput_factor"] > 0.99 and d["s"] > 0.5
    assert d0["throughput_factor"] < 0.9 and d0["s"] < 0.4
    pc = energy_report(r, replace(t, tdp_W=0.7 * P), power_cap=True)["power_cap"]
    assert pc["dvfs"]["dfs"]["throughput_factor"] > pc["throughput_factor"] + 0.1     # dfs beats duty cycling


def test_compute_bound_dfs_close_to_duty():
    # MAC-bound prefill: cycles per unit are fixed, so the clock energy per unit is too — dfs ≈ duty cycling
    r = evaluate(_sc(phase="prefill", b=8))
    assert r.bound == "MAC"
    P = energy_report(r, _t())["avg_W_per_card"]
    for frac in (0.9, 0.6):
        pc = energy_report(r, _t(tdp_W=frac * P), power_cap=True)["power_cap"]
        assert abs(pc["dvfs"]["dfs"]["throughput_factor"] - pc["throughput_factor"]) < 0.01


def test_whatif_dfs_reevaluated():
    s = _sc("qwen3-8b", b=64)
    r = evaluate(s)
    w = A.balance_whatif(s, r, 120.0)
    pc = w["base"]["power_cap"]
    d = dvfs_cap(r, A._whatif_table(), 120.0, "dfs")
    assert pc["binds"] and abs(pc["tf_dfs"] - d["throughput_factor"]) < 1e-9 and "s_dfs" in pc
    assert A._whatif_table().pJ_clk_mac_cycle is not None


def test_ui_keys():
    from pathlib import Path
    web = Path(A.__file__).parent / "web"
    js, html = W.read(), W.read(W.INDEX_HTML)          # 0.286: parsed
    assert W.mentions(js, "clock_W_card") and "pJ_clk_mac_cycle" in W.strings(js) and "e-pJ_clk_mac_cycle" in W.html_ids(html)
    out = A.api_eval({"chip_preset": "1P", "scenario": {"model": "qwen3-8b"},
                      "energy": {"energy_preset": "tpuv4i_7nm"}})
    e = out["energy"]
    assert "clock" in e["J_by_action"] and e["clock_W_card"] > 0


def test_fuzz_invariant_and_monotone():
    rng = random.Random(228)
    models = ("qwen3-8b", "qwen3-32b", "rwkv7-g1k-7.2b")
    for _ in range(14):
        m = rng.choice(models)
        ph = rng.choice(("decode", "prefill"))
        b = rng.choice((1, 4, 16, 64)) if ph == "decode" else rng.choice((1, 2, 8))
        f = rng.uniform(0.6, 2.0)
        sc = _sc(m, ph, b)
        sc = sc.replace("chip", replace(sc.chip, freq_ghz=f))
        r = evaluate(sc)
        cards = r.scenario.layout.cards
        pj = rng.uniform(0.0, 0.5)
        t = _t(pJ_clk_mac_cycle=pj)
        e = energy_report(r, t)
        assert abs(e["J_per_unit"] * _units(e, cards) / e["avg_W_per_card"] - 1) < 1e-9
        e2 = energy_report(r, _t(pJ_clk_mac_cycle=2 * pj))
        assert abs(e2["J_by_action"]["clock"] - 2 * e["J_by_action"]["clock"]) <= 1e-12 + 1e-9 * e2["J_by_action"]["clock"]
        assert e2["avg_W_per_card"] >= e["avg_W_per_card"] - 1e-9
        frac = rng.uniform(0.5, 0.95)
        P = e["avg_W_per_card"]
        pc = energy_report(r, replace(t, tdp_W=frac * P), power_cap=True)["power_cap"]
        if pc.get("error") or not pc.get("capped"):
            continue
        d = pc["dvfs"]["dfs"]
        if d.get("error"):
            continue
        assert d["throughput_factor"] >= pc["throughput_factor"] - 2e-3, (m, ph, b, f, pj, frac)
        assert d["W_card"] <= frac * P * (1 + 1e-9)
        q, eq = _point(r, t, "dfs", d["s"])
        assert abs(eq["J_per_unit"] * _units(eq, cards) / eq["avg_W_per_card"] - 1) < 1e-9
