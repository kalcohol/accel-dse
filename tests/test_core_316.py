"""0.316: the power cap compares the same complete package power as the thermal check (one shared function,
thermal.cap_power_fn → package_power), power_scope "card" reproduces 0.315; designs over the selected cooling limit
leave the Pareto front (front_thermal_out) only when a cooling preset is chosen; freeze-aware timing guards."""
import importlib.util
import os
import signal
import subprocess
import sys
import time

from accel_dse import api
from accel_dse.core.thermal import POWER_SCOPES, cap_power_fn, card_W_for_cap, package_power
from accel_dse.core.energy_dse import energy_design_search
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 0.315 (commit 517b865, power cap on card W) — chiplet_example case 1 (qwen3-8b prefill tp, L0), 700 W budget:
OLD_CAPPED_THR = {9: 93791.672, 10: 96713.337, 11: 95072.811, 12: 96217.883, 13: 93824.535, 15: 94746.84,
                  16: 90821.435}
_C = {}


def _ex():
    if "m" not in _C:
        spec = importlib.util.spec_from_file_location("chiplet_example",
                                                      os.path.join(ROOT, "scripts", "chiplet_example.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _C["m"] = m
    return _C["m"]


def _case1(cooling, scope):
    k = (cooling, scope)
    if k not in _C:
        _C[k] = _ex().run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling=cooling,
                          power_scope=scope)
    return _C[k]


def test_shared_function_and_inverse():
    assert POWER_SCOPES == ("package", "card") and cap_power_fn(600.0, 1, "card") is None
    pk = cap_power_fn(600.0, 1, "package")
    for c, st, iw in ((300.0, 10.0, None), (500.0, 60.0, None), (400.0, 0.0, 80.0)):
        assert pk(c, st, iw) == package_power(c, 600.0, 1, st, iw)["pkg_W"]
    c = card_W_for_cap(pk, 700.0, 10.0)
    assert abs(pk(c, 10.0) - 700.0) < 1e-6 and c < 700.0
    assert card_W_for_cap(None, 700.0) == 700.0
    assert card_W_for_cap(pk, 10.0) is None                     # floor + VR loss alone exceed the cap
    try:
        cap_power_fn(600.0, 1, "board")
        raise AssertionError
    except ValueError:
        pass


def test_dse_cap_equals_thermal_package_power():
    o = _case1("air_oam", "package")
    cap = o["power_cap_W"]
    assert o["power_scope"] == "package"
    capped = [x for x in o["rows"] if x.get("capped")]
    assert len(capped) == 9
    for x in capped:                                            # the very same number on both checks
        assert abs(x["thermal"]["pkg_W"] - x["cap_pkg_W"]) <= 1e-9 * x["cap_pkg_W"]
        assert x["cap_pkg_W"] <= cap * (1 + 1e-9)


def test_card_scope_reproduces_0315():
    # 0.329 re-pin: the air_oam front [4, 17] → [1, 0, 15] (HBM3E PHY 16 → 24 mm² + 3.575 mm² controller per stack
    # change the die areas / flux of every row); the capped set and capped clocks are unchanged
    for cool, front in ((None, [12]), ("air_oam", [1, 0, 15])):
        o = _case1(cool, "card")
        assert o["power_scope"] == "card" and o["front"] == front
        got = {i: x["thr"] for i, x in enumerate(o["rows"]) if x.get("capped")}
        assert set(got) == set(OLD_CAPPED_THR)
        # 0.323: CAP_ITERS 12 → 16 continues the same bisection (the 12-step clock is a prefix), so the capped clock
        # only rises, by ≤ (1 − F_MIN)/2^12 — the 0.315 values (12 steps) are lower bounds within 3e-4
        for i, t in OLD_CAPPED_THR.items():
            assert t - 1e-3 <= got[i] <= t * (1 + 3e-4), (i, got[i], t)
    # package scope caps more (static floor + VR loss count against the budget) and never runs faster
    new = _case1(None, "package")
    old = _case1(None, "card")
    assert new["front"] == [12]
    for a, b in zip(new["rows"], old["rows"]):
        if b.get("capped"):
            assert a.get("capped")
        if a.get("feasible") and b.get("feasible"):
            assert a["thr"] <= b["thr"] * (1 + 1e-9)


def test_front_thermal_out_only_with_cooling():
    o = _case1("air_oam", "card")                               # 0.315 scope: row 12 is over air and was on the front
    assert o["n_front_thermal_out"] == 1 and o["rows"][12]["front_thermal_out"] is True
    assert not o["rows"][12].get("front") and "热流密度" in o["rows"][12]["why"]
    assert all("_ok_wo_thermal" not in x for x in o["rows"])
    assert o["n_thermal_over"] >= 1
    n = _case1(None, "card")
    assert n["n_front_thermal_out"] == 0 and n["rows"][12].get("front")
    assert not any(x.get("front_thermal_out") for x in n["rows"])
    try:
        energy_design_search(None, None, None, power_scope="board")
        raise AssertionError
    except ValueError as e:
        assert "power_scope" in str(e)


def test_api_scope():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_4s_8h24g_9200",
                                           "serving": {"phase": "prefill", "batch": 1, "prompt": 4096, "ctx": 8192}},
         "energy": {"energy_preset": "tpuv4i_7nm", "tdp_W": 400}, "sensitivity": False, "power_cap": True,
         "cooling": "air_oam"}
    p = api.api_eval(b)
    c = api.api_eval({**b, "power_scope": "card"})
    pp, pc = p["energy"]["power_cap"], c["energy"]["power_cap"]
    assert pp["scope"] == "package" and pc["scope"] == "card"
    # 0.323: the thermal check runs at the capped dfs point (as the energy DSE) — ≤ 0.322 at the uncapped package
    # power pkg_needed_W (630.63 → 399.99 W here); pkg_needed_W stays the nominal requirement the cap is set against
    th = p["thermal"]
    assert th["operating_point"] == "capped" and abs(th["pkg_W"] - pp["dvfs"]["dfs"]["pkg_W"]) < 1e-9
    assert th["pkg_W"] <= 400 * (1 + 1e-9) < pp["pkg_needed_W"]
    assert pc["cap_card_W"] == 400 and pp["cap_card_W"] < 400
    assert pp["throughput_factor"] < pc["throughput_factor"]
    try:
        api.api_eval({**b, "power_scope": "x"})
        raise AssertionError
    except api.ApiError:
        pass


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert 'id="ins-dse-cardscope"' in html and "ex.power_scope = 'card'" in js
    assert "x.front_thermal_out ? '✗热'" in js and "r.n_front_thermal_out" in js and "与热判定同口径" in js
    assert "'热 / 封装 W'" in js and "未选冷却方案" in js


def _fc(mode, s, stop):
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "freeze_clock.py"), mode, str(s)],
                         stdout=subprocess.PIPE, text=True)
    if stop:
        time.sleep(1.0)
        p.send_signal(signal.SIGSTOP)
        time.sleep(2.0)
        p.send_signal(signal.SIGCONT)
    d = dict(kv.split("=") for kv in p.communicate(timeout=120)[0].split())
    return {k: float(v) for k, v in d.items()}


def _fake(script):
    """0.331: drive FreezeClock.observe with a scripted (wall, cpu) sequence — no threads, no real time.
    ``script`` = list of (wall gap, cpu gained) per sampler wake-up; returns (clock, wall, excused list)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from freeze_clock import FreezeClock
    t = {"w": 100.0, "c": 7.0}
    fc = FreezeClock(wall=lambda: t["w"], cpu=lambda: t["c"], start=False)
    a0, ex = fc.now(), []
    for dw, dc in script:
        t["w"] += dw
        t["c"] += dc
        ex.append(fc.observe(t["w"], t["c"]))
    return fc, t["w"] - 100.0, fc.now() - a0, ex


P = 0.05


def test_freeze_clock_excuses_only_non_execution():
    # 0.331: deterministic (fake clock).  ≤ 0.330 ran three real subprocesses and asserted on wall time; under -j 8 on
    # msa1 the GIL case failed once (frozen > 0): a starved process (or a host freeze) is a legitimate stall for the
    # sampler, so wall-time equalities cannot be asserted on a loaded host.  Same three cases, now exact:
    # (a) CPU-bound run stopped 2 s (SIGSTOP / host freeze stand-in): the 2 s gap is excused minus one period
    fc, wall, act, ex = _fake([(P, P)] * 20 + [(2.0, 0.0)] + [(P, P)] * 20)
    assert abs(fc.frozen() - (2.0 - P)) < 1e-9 and abs(act - (wall - (2.0 - P))) < 1e-9 and act > 2.0
    assert len(fc.stalls()) == 1 and fc.stalls()[0][1:] == (2.0, 0.0)
    # (b) sleeping / blocking hang: sampler wakes every period, nothing excused, charged in full
    fc, wall, act, ex = _fake([(P, 0.0)] * 40)
    assert fc.frozen() == 0 and abs(act - wall) < 1e-9 and abs(wall - 2.0) < 1e-9
    # (c) one C call holding the GIL for 2 s: one 2 s gap with CPU ≈ gap — not a stall, charged in full
    fc, wall, act, ex = _fake([(P, P)] * 4 + [(2.0, 2.0)] + [(P, P)] * 4)
    assert fc.frozen() == 0 and abs(act - wall) < 1e-9


def test_freeze_clock_thresholds_and_cpu_never_excused():
    # thresholds 「假设」: gap ≥ 0.5 s and CPU < 20 % of the gap (both strict as coded)
    assert _fake([(0.49, 0.0)])[3] == [0.0]
    assert abs(_fake([(0.5, 0.0)])[3][0] - (0.5 - P)) < 1e-12
    assert _fake([(1.0, 0.2)])[3] == [0.0]                       # exactly 20 % CPU → running, not frozen
    # 0.331 fix: a starved-but-running process (CPU 15 % of a 2 s gap — e.g. a GIL call under heavy -j load) is a
    # stall, but its executed CPU stays charged: excused 2.0 − 0.3 = 1.7 (≤ 0.330: 2.0 − 0.05 = 1.95, active 0.05 s
    # < the 0.3 s it executed)
    fc, wall, act, ex = _fake([(2.0, 0.3)])
    assert abs(ex[0] - 1.7) < 1e-12 and abs(act - 0.3) < 1e-12
    # invariant: active ≥ CPU executed, for any mix of gaps
    import random
    rnd = random.Random(316)
    for _ in range(200):
        sc = [(g, g * rnd.choice((0.0, 0.05, 0.19, 0.2, 0.5, 1.0))) for g in
              (rnd.choice((P, 0.3, 0.5, 0.8, 2.0, 7.5)) for _ in range(30))]
        fc, wall, act, ex = _fake(sc)
        assert act >= sum(c for _, c in sc) - 1e-9 and act >= len(sc) * P - 1e-9 and act <= wall + 1e-9


def test_freeze_clock_real_process_invariants():
    # real sampler thread + real SIGSTOP / GIL call; only load-independent assertions (a busy host can add stalls
    # but can never make a stopped process run or un-charge executed CPU)
    # (active vs CPU over the clock's whole life: excused ≤ gap − CPU-in-gap for every stall ⇒ act_all ≥ cpu_all)
    f = _fc("spin", 3, True)            # CPU-bound 3 s, stopped ≥ 2 s
    assert f["frozen"] >= 1.5 and f["cpu"] >= 2.9 and f["active"] <= f["wall"]
    assert f["act_all"] >= f["cpu_all"] - 0.02
    g = _fc("gil", 2, False)            # one C call holding the GIL: its CPU is always charged
    assert g["act_all"] >= g["cpu_all"] - 0.02 and g["active"] <= g["wall"] + 1e-3 and g["cpu"] > 0.5
