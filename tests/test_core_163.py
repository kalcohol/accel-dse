"""0.163: sizing setter isolation (each sized resource moves alone) — guards the ≤ 0.162 lanes→dequant coupling."""
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import SIZING
from accel_dse.core.scenario import Scenario


def test_each_setter_moves_only_its_resource():
    for name, ch in CHIPS.items():
        s = Scenario().replace("chip", ch)
        for k, w in SIZING.items():
            for f in (0.5, 0.13):
                s2 = w[3](s, w[2](s) * f)
                # 0.183: dropping DRAM packages narrows the bus, so delivered bandwidth must follow (never rise)
                moved = [j for j, x in SIZING.items() if j != k and abs(x[2](s2) - x[2](s)) > 1e-9
                         and not (k == "dram_units" and j == "dram_bw" and x[2](s2) < x[2](s))]
                assert not moved, (name, k, f, moved)
                assert abs(w[2](s2) - (round(w[2](s) * f) if w[4] else w[2](s) * f)) <= 1e-6 * w[2](s) + (1 if w[4] else 0)


def test_lanes_keeps_default_dequant_rate():
    s = Scenario()
    assert s.chip.formats.dequant_elems_per_cycle is None          # default: dequant rate = vector width
    t = SIZING["lanes"][3](s, s.chip.lanes / 4)
    assert t.chip.lanes == s.chip.lanes // 4 and t.chip.formats.dequant_elems_per_cycle == float(s.chip.lanes)
