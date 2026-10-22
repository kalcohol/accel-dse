"""0.332: deterministic work meter for the suite's runtime guards (stdlib, ``sys.monitoring``, Python ≥ 3.12).

Why: wall time and even process CPU time are not load-independent on the shared hosts — rt278b measured 9.5 × the CPU
seconds for identical work under ``run_tests.py -j 8`` (AGENTS §229), and 0.303 / 0.331 guards failed only under load.
The quantity a runtime guard exists to bound — how much work a request does (exploding loops, 22-min regressions) —
is a property of the code path, so this meter counts it:

  work = Python function starts in accel_dse code + calls issued from accel_dse code (Python or C, e.g. sorted / sum),

the same number for the same request whatever else the box is doing.  Code outside the package (stdlib, tests) is
switched off per code location after its first event (``sys.monitoring.DISABLE``), so it costs nothing afterwards."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "accel_dse") + os.sep
_M = sys.monitoring
_E = _M.events


def _tool() -> int:
    for t in (3, 4):
        if _M.get_tool(t) is None:
            return t
    raise RuntimeError("no free sys.monitoring tool id")


class Meter:
    def __init__(self):
        self.calls = 0

    def now(self) -> float:
        """Work so far in this meter's unit (calls; process CPU seconds in WM_CALIBRATE=cpu mode)."""
        return self.calls


@contextmanager
def meter(events: str = "start+call"):
    """Count deterministic work inside the block: ``with meter() as m: …; m.calls``."""
    m, tid = Meter(), _tool()
    pkg, dis = _PKG, _M.DISABLE

    def start(code, off):
        if code.co_filename.startswith(pkg):
            m.calls += 1
            return None
        return dis

    def call(code, off, fn, arg0):
        if code.co_filename.startswith(pkg):
            m.calls += 1
            return None
        return dis

    def jump(code, off, dest):
        if code.co_filename.startswith(pkg):
            if dest < off:
                m.calls += 1
            return None
        return dis

    _M.use_tool_id(tid, "accel-workmeter")
    try:
        ev = _E.PY_START
        _M.register_callback(tid, _E.PY_START, start)
        if "call" in events:
            _M.register_callback(tid, _E.CALL, call)
            ev |= _E.CALL
        if "jump" in events:
            _M.register_callback(tid, _E.JUMP, jump)
            ev |= _E.JUMP
        _M.set_events(tid, ev)
        _M.restart_events()
        yield m
    finally:
        _M.set_events(tid, 0)
        _M.register_callback(tid, _E.PY_START, None)
        _M.register_callback(tid, _E.CALL, None)
        _M.register_callback(tid, _E.JUMP, None)
        _M.free_tool_id(tid)


# ---------------------------------------------------------------- guards
# Calls per CPU-second differ ~8× between workloads (2.3 M/s protenix … 18.8 M/s qwen3-8b prefill), so one global rate
# would either loosen the light guards or fail the heavy ones (237 report vllm: 460 M calls in 57.7 s of a 90 s bound).
# Each guard therefore converts its CPU bound with ITS OWN quiet-box rate — calls ≤ bound × rate ⇔ quiet CPU ≤ bound,
# the pre-0.332 threshold exactly — measured 2026-10-22 on the box, single process, no load, by
# scripts/workmeter_calibrate.py (MODEL §277; 近似: rates rounded down, so every budget is ≤ the old CPU bound).
# Blocks under 0.05 s CPU (rate unresolvable) and labels not listed use RATE_FLOOR = the lowest measured rate.
RATE_FLOOR = 2_300_000  # protenix 2,310,566/s rounded down
RATES = {  # label: calls per quiet-box CPU-second (measured rate rounded DOWN to 2 significant figures; cpu, calls)
    '099 wan2.1-14b pipe eval': 11_000_000,  # 0.094 s, 1_092_650
    '210 fuzz 0 eval': 3_800_000,  # 0.069 s, 262_852
    '210 fuzz 12 eval': 4_000_000,  # 0.070 s, 285_013
    '210 fuzz 13 sensitivity': 4_800_000,  # 7.131 s, 34_457_727
    '210 fuzz 14 fit': 4_500_000,  # 0.075 s, 341_135
    '210 fuzz 2 fit': 5_400_000,  # 0.378 s, 2_057_157
    '210 fuzz 3 sweep': 4_400_000,  # 0.069 s, 303_685
    '210 fuzz 4 eval': 5_400_000,  # 0.409 s, 2_248_864
    '210 fuzz 6 fit': 6_800_000,  # 5.864 s, 40_270_781
    '210 huge qwen3-8b prefill': 18_000_000,  # 0.099 s, 1_856_193
    '210 huge qwen3-next-80b-a3b prefill': 10_000_000,  # 0.117 s, 1_265_558
    '210 huge rwkv7-g1k-7.2b decode': 5_000_000,  # 1.297 s, 6_555_653
    '210 huge startlux-decision-v1-35b-a3b prefill': 15_000_000,  # 0.289 s, 4_472_730
    '237 phase-law sglang': 4_500_000,  # 3.142 s, 14_235_351
    '237 phase-law sglang_mixed': 9_200_000,  # 12.021 s, 110_834_468
    '237 phase-law vllm': 8_900_000,  # 29.393 s, 262_172_553
    '237 phase-law vllm_v0': 8_000_000,  # 14.684 s, 117_490_410
    '237 report sglang': 6_200_000,  # 5.256 s, 32_944_175
    '237 report sglang_mixed': 9_400_000,  # 16.623 s, 156_455_520
    '237 report vllm': 7_900_000,  # 57.698 s, 459_972_549
    '237 report vllm_v0': 7_900_000,  # 15.186 s, 120_484_942
    '238 mx4096': 8_700_000,  # 10.613 s, 93_072_295
    '238 vl4096': 10_000_000,  # 16.062 s, 162_796_028
    '238 vl6000': 8_500_000,  # 16.151 s, 137_407_011
    '289 boltz-1': 3_600_000,  # 0.912 s, 3_318_716
    '289 protenix': 2_300_000,  # 2.253 s, 5_205_676
}


def budget(bound_s: float, label: str = "") -> int:
    """Work budget equivalent to ``bound_s`` CPU seconds of ``label``'s workload on the quiet reference box."""
    return int(bound_s * RATES.get(label, RATE_FLOOR))


class _CpuMeter(Meter):
    def __init__(self):
        super().__init__()
        self._c0 = time.process_time()

    def now(self) -> float:
        return time.process_time() - self._c0


def _log(label: str, unit: str, v: float) -> None:
    f = os.environ.get("WM_LOG")
    if f:
        with open(f, "a") as fh:
            fh.write(f"{label}\t{unit}\t{v!r}\n")


@contextmanager
def guard(bound_s: float, label: str):
    """Runtime guard on deterministic work: the block may do at most ``budget(bound_s)`` calls (the work ``bound_s``
    quiet-box CPU seconds buy).  ``WM_CALIBRATE=cpu|calls`` (scripts/workmeter_calibrate.py) logs the block's CPU
    seconds (unmetered) or calls to ``WM_LOG`` instead of asserting."""
    cal = os.environ.get("WM_CALIBRATE")
    if cal == "cpu":
        m = _CpuMeter()
        yield m
        _log(label, "cpu", m.now())
        return
    with meter() as m:
        yield m
    if cal == "calls":
        _log(label, "calls", m.calls)
        return
    b = budget(bound_s, label)
    assert m.calls <= b, ("work", label, m.calls, b, bound_s)


def within(m: Meter, used: float, bound_s: float, label: str) -> None:
    """Sub-guard on part of a guarded block (``used`` = m.now() deltas summed by the caller)."""
    cal = os.environ.get("WM_CALIBRATE")
    if cal:
        _log(label, cal, used)
        return
    b = budget(bound_s, label)
    assert used <= b, ("work", label, used, b, bound_s)
