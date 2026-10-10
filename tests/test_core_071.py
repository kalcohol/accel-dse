"""0.71: Chip.instance_sched — independent GEMM instances over core groups (wide / split / auto)."""
from dataclasses import replace

import pytest

from accel_dse.core.hardware import CHIPS, Chip
from accel_dse.core.mapping import gemm_cost, _groups

SHAPES = [(128, 4097, 512, 64), (8, 128, 4096, 512), (8, 4096, 128, 512), (4096, 128, 4096, 64), (2, 7168, 4096, 128),
          (512, 576, 4097, 16), (3, 5, 7, 1000)]


def test_default_wide_and_validation():
    assert Chip().instance_sched == "wide" and all(c.instance_sched == "wide" for c in CHIPS.values())
    with pytest.raises(ValueError):
        Chip(instance_sched="lockstep")
    assert _groups(32) == (1, 2, 4, 8, 16) and _groups(1) == ()


@pytest.mark.parametrize("geom", [(128, 128, 32), (64, 64, 128), (256, 256, 8), (32, 32, 512), (56, 56, 16)])
def test_auto_le_split_le_wide_and_ge_ideal(geom):
    r, c, e = geom
    ch = Chip("t", 1.0, r, c, e)
    for m, k, n, cnt in SHAPES:
        w, s, a = (gemm_cost(replace(ch, instance_sched=x), "reconf", m, k, n, count=cnt).cycles
                   for x in ("wide", "split", "auto"))
        ideal = m * k * n * cnt / ch.macs
        assert a <= s * (1 + 1e-12) and a <= w * (1 + 1e-12) and s <= w * (1 + 1e-12)
        assert a >= ideal * (1 - 1e-9)


def test_single_instance_and_single_engine_unchanged():
    ch = Chip("t", 1.0, 64, 64, 32)
    for x in ("split", "auto"):
        assert gemm_cost(replace(ch, instance_sched=x), "reconf", 512, 1024, 768).cycles == \
            gemm_cost(ch, "reconf", 512, 1024, 768).cycles
    one = Chip("t1", 1.0, 64, 64, 1)
    assert gemm_cost(replace(one, instance_sched="auto"), "reconf", 8, 128, 4096, count=64).cycles == \
        gemm_cost(one, "reconf", 8, 128, 4096, count=64).cycles


def test_split_instances_alias():
    ch = Chip("t", 1.0, 128, 128, 32)
    assert gemm_cost(replace(ch, split_instances=True), "reconf", 128, 4097, 512, count=64).cycles == \
        gemm_cost(replace(ch, instance_sched="split"), "reconf", 128, 4097, 512, count=64).cycles


def test_feed_shared_port():
    """A group gets g/E of the SRAM port: a feed-bound batched op is not sped up beyond the shared port."""
    ch = Chip("t", 1.0, 32, 32, 512, sram_port_Bpc=4096.0)
    w = gemm_cost(ch, "reconf", 1, 4096, 128, count=4096)
    a = gemm_cost(replace(ch, instance_sched="auto"), "reconf", 1, 4096, 128, count=4096)
    assert a.cycles >= a.feed_cycles and a.feed_cycles >= 0.99 * (4096 * 128 * 2 * 4096) / 4096
