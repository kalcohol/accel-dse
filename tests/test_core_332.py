"""0.332: load-robust runtime guards — deterministic work meter (tests/workmeter.py) replaces wall / CPU-time bounds in
099 / 209 / 210 / 237 / 238 / 289; per-guard quiet-box rates keep the pre-0.332 thresholds (calls ≤ bound × rate ⇔
quiet CPU ≤ bound); decision timers in scripts / validation are monotonic."""
import ast
import os
import re

from accel_dse.core.scenario import Scenario
from tests import workmeter as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _burn():
    for _ in range(20):
        Scenario().hash()


def test_meter_is_deterministic_and_counts_only_package_code():
    _burn()  # warm caches
    with W.meter() as a:
        _burn()
    with W.meter() as b:
        _burn()
    assert a.calls == b.calls > 1000
    with W.meter() as c:
        sorted(range(10000), key=lambda x: -x)  # test-module code: switched off, not counted
    assert c.calls == 0


def test_meter_frees_its_tool_id_and_nests():
    with W.meter() as outer:
        with W.meter() as inner:
            _burn()
        _burn()
    assert inner.calls > 0 and outer.calls > inner.calls
    with W.meter() as again:  # ids released
        _burn()
    assert again.calls > 0


def test_rates_are_rounded_down_measurements_and_floor_is_minimum():
    assert W.RATES and W.RATE_FLOOR == min(W.RATES.values())
    for lab, r in W.RATES.items():
        assert r >= W.RATE_FLOOR and len(str(r).rstrip("0")) <= 2, (lab, r)
    src = open(os.path.join(ROOT, "tests", "workmeter.py"), encoding="utf-8").read()
    for lab, r in W.RATES.items():  # each rate ≤ the logged measurement calls / cpu
        m = re.search(re.escape(repr(lab)) + r": [\d_]+,  # ([\d.]+) s, ([\d_]+)", src)
        assert m and float(m.group(1)) >= 0.05, lab
        assert r <= int(m.group(2).replace("_", "")) / float(m.group(1)), lab


def test_budget_per_label_and_floor():
    lab = "237 report vllm"
    assert W.budget(90, lab) == int(90 * W.RATES[lab])
    assert W.budget(2, "no such label") == int(2 * W.RATE_FLOOR)


def test_guard_trips_on_overrun_and_passes_within():
    _burn()
    try:
        with W.guard(1e-6, "332 overrun"):
            _burn()
        raise AssertionError("passed")
    except AssertionError as e:
        assert e.args[0][0] == "work" and e.args[0][1] == "332 overrun"
    with W.guard(5, "332 ok") as m:
        _burn()
    assert m.calls > 0


def test_guarded_tests_use_workmeter_not_clocks():
    for n in ("099", "209", "210", "237", "238", "289"):
        src = open(os.path.join(ROOT, "tests", f"test_core_{n}.py"), encoding="utf-8").read()
        assert "from tests.workmeter import" in src, n
        assert "process_time" not in src and "perf_counter" not in src, n


def test_decision_timers_are_monotonic():
    for rel in ("scripts/static_parity_check.py", "scripts/ui_headless_smoke.py", "scripts/ui_static_smoke.py",
                "accel_dse/core/validation.py"):
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
        bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "time"
               and isinstance(n.value, ast.Name) and n.value.id == "time"]
        assert not bad, (rel, bad)


def test_calibration_script_present():
    src = open(os.path.join(ROOT, "scripts", "workmeter_calibrate.py"), encoding="utf-8").read()
    assert "WM_CALIBRATE" in src and "RATE_FLOOR" in src
