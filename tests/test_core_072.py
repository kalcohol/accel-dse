"""0.72: auto for single GEMMs (M / N split over core groups), optional NoC hop latency, optional wave barriers."""
from __future__ import annotations

from dataclasses import replace

from accel_dse.core.hardware import CHIPS, Chip
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.scenario import Scenario


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")


def test_single_gemm_n_split_beats_wide_fill():
    # 32 cores of 32×32: wide = 32 × 1024 array, fill 32 + 1024; a narrow N = 64 GEMM uses 1/16 of the columns
    w = Chip("c", 1.0, 32, 32, 32, instance_sched="wide")
    a = replace(w, instance_sched="auto")
    gw = gemm_cost(w, "reconf", 256, 4096, 64)
    ga = gemm_cost(a, "reconf", 256, 4096, 64)
    assert ga.cycles < 0.5 * gw.cycles, (ga.cycles, gw.cycles)
    # split mode keeps count = 1 on the wide array (0.70 semantics: one instance per engine)
    s = replace(w, instance_sched="split")
    assert gemm_cost(s, "reconf", 256, 4096, 64).cycles == gw.cycles


def test_single_gemm_wide_when_it_fits():
    # a GEMM that fills the wide array: auto never worse, and equal to wide when wide is already best
    w = Chip("c", 1.0, 64, 64, 4, instance_sched="wide")
    a = replace(w, instance_sched="auto")
    gw, ga = gemm_cost(w, "reconf", 4096, 4096, 4096), gemm_cost(a, "reconf", 4096, 4096, 4096)
    assert ga.cycles <= gw.cycles and ga.cycles >= 0.99 * gw.cycles


def test_noc_and_overhead_default_off_and_hash_stable():
    c = Chip()
    assert c.noc_hop_cycles == 0.0 and c.wave_overhead_cycles == 0.0 and c.noc_cycles() == 0.0
    s = Scenario(model="qwen3.8-27b")
    assert s.hash() == Scenario(model="qwen3.8-27b", chip=replace(s.chip, noc_topology="mesh")).hash()
    assert s.hash() != Scenario(model="qwen3.8-27b", chip=replace(s.chip, noc_hop_cycles=1.0)).hash()
    _raises(ValueError, lambda: Chip(noc_topology="torus"))
    _raises(ValueError, lambda: Chip(noc_hop_cycles=-1.0))
    _raises(ValueError, lambda: Chip(wave_overhead_cycles=float("inf")))


def test_noc_diameters():
    for topo, e, d in (("crossbar", 16, 1), ("ring", 16, 8), ("mesh", 16, 6), ("mesh", 17, 8), ("mesh", 1, 0),
                       ("ring", 2, 1)):
        c = Chip("c", 1.0, 8, 8, e, noc_topology=topo, noc_hop_cycles=3.0)
        assert c.noc_cycles() == 3.0 * d, (topo, e, c.noc_cycles())


def test_noc_and_overhead_add_latency_to_wide():
    w = Chip("c", 1.0, 32, 32, 16, instance_sched="wide")
    g0 = gemm_cost(w, "reconf", 64, 1024, 1024, count=8)
    g1 = gemm_cost(replace(w, noc_hop_cycles=10.0), "reconf", 64, 1024, 1024, count=8)
    g2 = gemm_cost(replace(w, wave_overhead_cycles=100.0), "reconf", 64, 1024, 1024, count=8)
    assert g1.cycles == g0.cycles + 10.0 * 6 and g2.cycles == g0.cycles + 100.0


def test_wave_barriers_penalise_many_waves():
    # many tiny instances: lock-step waves on 16 groups pay a barrier per wave, so with a large barrier auto
    # falls back towards fewer, longer streams — never slower than wide
    a = Chip("c", 1.0, 16, 16, 16)
    kw = dict(count=4096)
    free = gemm_cost(a, "reconf", 1, 64, 64, **kw).cycles
    slow = gemm_cost(replace(a, wave_overhead_cycles=1000.0), "reconf", 1, 64, 64, **kw).cycles
    wide = gemm_cost(replace(a, instance_sched="wide", wave_overhead_cycles=1000.0), "reconf", 1, 64, 64, **kw).cycles
    assert free < slow <= wide


def test_catalog_chips_noc_off():
    assert all(c.noc_hop_cycles == 0.0 and c.wave_overhead_cycles == 0.0 for c in CHIPS.values())
