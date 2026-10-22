"""0.227: power-capped operation by clock / voltage scaling (core/powercap.py); static-power fix of the 0.189 cap;
per-lever cap binding in the balance what-if; fuzz incl. J/unit × units/s/card = W/card at the capped point."""
import random

from accel_dse import api as A
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import CAP_CURVES, F_MIN, VF_CURVES, dvfs_cap, static_W, volt
from accel_dse.core.scenario import Scenario


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", "idle_W": 20.0, **kw}))


def _sc(model="qwen3-8b", phase="decode", b=64):
    return Scenario(chip=CHIPS["1P"], model=model).replace("serving.phase", phase).replace("serving.batch", b)


def _units(e, cards):
    return e["units_per_window"] / e["window_s"] / cards


# ---- curves
def test_vf_curves():
    assert volt("dfs", 0.2) == volt("dfs", 1.0) == 1.0
    pts = VF_CURVES["rx5700xt_n7"]["points"]
    assert abs(pts[0][0] - 800 / 2019) < 1e-12 and abs(pts[0][1] - 704 / 1185) < 1e-12
    assert volt("rx5700xt_n7", 0.1) == pts[0][1] and volt("rx5700xt_n7", 1.0) == 1.0
    vs = [volt("rx5700xt_n7", s / 100) for s in range(10, 101)]
    assert all(b >= a for a, b in zip(vs, vs[1:]))
    assert VF_CURVES["rx5700xt_n7"]["tag"] == "二手" and F_MIN == 0.15


# ---- 0.189 fix: static = idle + SRAM leak + HBM static
def test_static_fix():
    from dataclasses import replace
    t0 = replace(_t(), idle_W=None, idle_frac_tdp=None)      # no idle figure → SRAM leakage + HBM standby are the static
    r = evaluate(_sc())
    e = energy_report(r, t0)
    P = e["avg_W_per_card"]
    ps = static_W(e)
    assert ps > 0 and e.get("sram_leak_W_card") and e.get("hbm_static_W_card") and not e.get("idle_W_used")
    pc = energy_report(r, replace(t0, tdp_W=0.7 * P), power_cap=True)["power_cap"]
    assert abs(pc["P_static_W"] - ps) < 1e-9
    assert abs(pc["slowdown"] - (P - ps) / (0.7 * P - ps)) < 1e-9


# ---- dvfs
def test_dfs_equals_duty_and_vf_better():
    # 0.228: dfs ≡ duty cycling only without the per-cycle term (pJ_clk_mac_cycle); with it dfs ≥ duty (test_core_228)
    from dataclasses import replace
    nc = lambda **kw: replace(_t(**kw), pJ_clk_mac_cycle=None)      # noqa: E731
    for sc in (_sc(), _sc(phase="prefill", b=8), _sc("qwen3-32b", b=128)):
        r = evaluate(sc)
        P = energy_report(r, nc())["avg_W_per_card"]
        for frac in (0.95, 0.7, 0.5):
            pc = energy_report(r, nc(tdp_W=frac * P), power_cap=True)["power_cap"]
            d, v = pc["dvfs"]["dfs"], pc["dvfs"]["rx5700xt_n7"]
            assert abs(d["throughput_factor"] - pc["throughput_factor"]) < 2e-4, (sc.model, frac)
            assert v["throughput_factor"] >= d["throughput_factor"] - 1e-6
            for x in (d, v):
                assert x["binds"] and x["W_card"] <= frac * P * (1 + 1e-9) and x["W_card"] > frac * P * 0.995


def test_not_binding_and_floor_and_infeasible():
    r = evaluate(_sc())
    P = energy_report(r, _t())["avg_W_per_card"]
    x = dvfs_cap(r, _t(), 2 * P, "rx5700xt_n7")
    assert not x["binds"] and x["throughput_factor"] == 1.0 and x["s"] == 1.0
    ps = static_W(energy_report(r, _t()))
    lo = dvfs_cap(r, _t(), ps + 1.0, "dfs")                          # below the F_MIN power → duty cycling on top
    assert lo["s"] == F_MIN and lo["duty"] < 1 and abs(lo["W_card"] - (ps + 1.0)) < 1e-9
    bad = dvfs_cap(r, _t(), ps * 0.5, "dfs")
    assert bad["feasible"] is False and "error" in bad


def test_memory_bound_keeps_throughput_with_voltage():
    # DRAM-bound decode: the V–f curve sheds core power without losing throughput at a mild cap
    r = evaluate(_sc("qwen3-32b", b=128))
    assert r.bound == "DRAM"
    P = energy_report(r, _t())["avg_W_per_card"]
    v = dvfs_cap(r, _t(), 0.9 * P, "rx5700xt_n7")
    d = dvfs_cap(r, _t(), 0.9 * P, "dfs")
    # 0.228: with the per-cycle term, clock-only DFS sheds 10 % too (was tf < 0.9 without it — see test_core_228)
    # 0.333: HBM 5.80 pJ/bit raises the unsheddable DRAM share of card energy 34 → 47 %, so clock-only DFS needs s 0.513 →
    # 0.471 for the same 10 % cut and starts to leave the DRAM-bound regime: tf 0.9961 → 0.9813 (V–f still 1.0)
    assert v["throughput_factor"] > 0.99 and v["bound"] == "DRAM" and d["throughput_factor"] > 0.98


def test_invariant_at_capped_point():
    from accel_dse.core.powercap import _point
    r = evaluate(_sc(phase="prefill", b=8))
    P = energy_report(r, _t())["avg_W_per_card"]
    for c in CAP_CURVES:
        x = dvfs_cap(r, _t(), 0.6 * P, c)
        q, e = _point(r, _t(), c, x["s"])
        assert abs(e["J_per_unit"] * _units(e, q.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-9
        assert abs(x["J_per_unit"] * _units(energy_report(r, _t()), 1) * x["throughput_factor"] / x["W_card"] - 1) < 1e-6


# ---- what-if
def test_whatif_cap_binding():
    s = A.scenario_from_body({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"batch": 64, "ctx": 4096}}})
    r = evaluate(s)
    w0 = A.balance_whatif(s, r)
    assert "power_cap_W" not in w0 and all("power_cap" not in x for x in w0["rows"])     # opt-in
    w = A.balance_whatif(s, r, 120.0)
    assert w["power_cap_W"] == 120.0 and "power_cap" in w["base"]
    bw = [x for x in w["rows"] if x["key"] == "dram_bw_x2"][0]
    assert bw["works"] and bw["power_cap"]["binds"] and bw["power_cap"]["gain_dfs"] < 1 + A.WHATIF_GAIN
    assert "dram_bw_x2" in w["binds"] and "dram_bw_x2" not in w["works_capped"]
    for x in w["rows"]:
        pc = x.get("power_cap")
        if pc and pc.get("feasible"):
            assert pc["tf_vf"] >= pc["tf_dfs"] - 1e-6 and 0 < pc["tf_dfs"] <= 1


def test_sensitivity_route_cap_validation():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"batch": 64}}}
    for bad in (True, "120", -1, 0, 1e6):
        try:
            A.api_sensitivity({**b, "power_cap_W": bad})
            raise AssertionError(bad)
        except A.ApiError as e:
            assert "power_cap_W" in str(e)


# ---- fuzz
def test_fuzz_cap_monotone_and_invariant():
    rnd = random.Random(227)
    for _ in range(6):
        sc = _sc(rnd.choice(["qwen3-8b", "qwen3-32b", "glm-5.3-flash"]), rnd.choice(["decode", "prefill"]),
                 rnd.choice([1, 8, 64]))
        try:
            r = evaluate(sc)
        except ValueError:
            continue
        e = energy_report(r, _t())
        P, ps = e["avg_W_per_card"], static_W(e)
        caps = sorted(ps + (P - ps) * f for f in (rnd.uniform(0.02, 0.3), rnd.uniform(0.3, 0.7), rnd.uniform(0.7, 1.2)))
        for c in CAP_CURVES:
            tfs = [dvfs_cap(r, _t(), cap, c)["throughput_factor"] for cap in caps]
            assert all(0 < t <= 1 for t in tfs) and all(b >= a - 1e-6 for a, b in zip(tfs, tfs[1:])), (sc.model, c, tfs)
