"""0.71 schedule audit: invariants of Chip.instance_sched over a deterministic random sweep of geometries / shapes.

- auto is never slower than wide or split (they are among its candidates);
- auto never beats the ideal MAC bound (work conservation: cnt·m·k·n MACs over all engines);
- auto never streams faster than the chip-total SRAM port allows (port shared over the active groups);
- doubling cores (per-core array and per-core accumulator fixed, total SRAM port either geometry-derived or held
  fixed) never makes a batched GEMM slower (idle groups give their port share to the active ones).
"""
import random
from dataclasses import replace

from accel_dse.core.hardware import Chip
from accel_dse.core.mapping import gemm_cost, gemm_exec, _fmt


def _cases(n=600, seed=11):
    r = random.Random(seed)
    for _ in range(n):
        yield (r.choice([8, 16, 32, 64, 128, 256]), r.choice([8, 16, 32, 64, 128, 256]),
               r.choice([1, 2, 3, 4, 6, 8, 12, 16, 32, 64]), r.choice([None, 256.0, 1024.0, 8192.0]),
               r.choice([64.0, 256.0, 1024.0]), r.choice([1, 2, 3, 8, 17, 64, 128, 1000]),
               r.choice([16, 64, 128, 576, 4097]), r.choice([1, 64, 128, 512, 4096]),
               r.choice([2, 7, 32, 128, 1000]), r.choice(["bf16", "fp8"]))


def test_schedule_invariants():
    for R, C, E, port, acc, m, k, n, cnt, fm in _cases():
        base = Chip("a", 1.0, R, C, E, sram_port_Bpc=port, acc_kib=acc, instance_sched="wide")
        cost = lambda ch: gemm_cost(ch, "reconf", m, k, n, count=cnt, w_fmt=fm, a_fmt=fm)
        w, s = cost(base), cost(replace(base, instance_sched="split"))
        a = cost(replace(base, instance_sched="auto"))
        case = (R, C, E, port, acc, m, k, n, cnt, fm)
        assert a.cycles <= min(w.cycles, s.cycles) * (1 + 1e-12), case
        ideal = m * k * n * cnt / (base.macs * gemm_exec(fm, fm, base.formats)[1])
        assert a.cycles >= ideal * (1 - 1e-9), case
        b = _fmt(fm).bits / 8
        assert a.cycles >= cnt * (k * n * b + m * k * b + m * n * 2) / base.port_Bpc * (1 - 1e-9), case
        a2 = cost(replace(base, engines=2 * E, acc_kib=2 * acc, instance_sched="auto"))
        assert a2.cycles <= a.cycles * (1 + 1e-9), case
