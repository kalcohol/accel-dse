"""0.135 opt-in CTA tile I/O (Chip.tile_io = "dram"): each output tile pays one K-stage prologue and its C
writeback at the SM's DRAM share; with a format ceiling the main loop runs at the ceiling.  Checked against the
DeepSeek prefill trace's DeepGEMM shapes (H800, 108 / 132 SMs); default unchanged; validation."""
import math
from dataclasses import replace

from accel_dse.core import extval as X
from accel_dse.core.hardware import System
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario

# (m, k, n, count, trace ms per call, SMs) — deepseek-ai/profile-data prefill.json, one micro-batch of 8192 tokens
SHAPES = ((8192, 7168, 4096, 8, 3.567, 108), (8192, 2048, 7168, 8, 2.402, 108), (8192, 7168, 4096, 1, 0.448, 108),
          (8192, 2048, 7168, 1, 0.313, 108), (8192, 7168, 2112, 1, 0.264, 108), (8192, 1536, 24576, 1, 0.854, 108),
          (8192, 16384, 7168, 1, 1.702, 108), (8192, 7168, 36864, 1, 3.438, 132), (8192, 18432, 7168, 1, 1.679, 132))
CEIL = (("fp8", 0.7832238504295098),)


def _ms(ch, m, k, n, c):
    return gemm_cost(ch, "reconf", m, k, n, count=c, w_fmt="fp8", a_fmt="fp8").cycles / (ch.freq_ghz * 1e9) * 1e3


def test_tile_io_matches_deepseek_trace():
    h = REF_HW["H800-SXM"]
    bpc = System(h.chip, h.mem_id, X.EFF["catalog"]).dram_GBps / h.chip.freq_ghz
    errs = []
    for m, k, n, c, tr, E in SHAPES:
        ch = replace(h.chip, engines=E, instance_sched="cta", tile_io="dram", tile_io_Bpc=bpc, fmt_ceiling=CEIL)
        r = _ms(ch, m, k, n, c) / tr
        assert 0.85 < r < 1.05, (m, k, n, r)
        errs.append(abs(math.log(r)))
    assert math.exp(sum(errs) / len(errs)) - 1 < 0.10
    # small-K shapes (down K = 2048, q_b K = 1536) were the optimistic ones: tile I/O raises them most
    base = replace(h.chip, engines=108, instance_sched="cta", fmt_ceiling=CEIL)
    io = replace(base, tile_io="dram", tile_io_Bpc=bpc)
    up = _ms(io, 8192, 7168, 4096, 8) / _ms(base, 8192, 7168, 4096, 8)
    down = _ms(io, 8192, 2048, 7168, 8) / _ms(base, 8192, 2048, 7168, 8)
    assert down > up > 1.0


def test_default_unchanged_and_validation():
    h = REF_HW["H800-SXM"]
    a = gemm_cost(replace(h.chip, instance_sched="cta"), "reconf", 512, 2048, 7168, count=8, w_fmt="fp8", a_fmt="fp8")
    b = gemm_cost(replace(h.chip, instance_sched="cta", tile_io="dram"), "reconf", 512, 2048, 7168, count=8,
                  w_fmt="fp8", a_fmt="fp8")                      # flag without bandwidth = evaluate not run: unchanged
    assert a.cycles == b.cycles
    assert Scenario().chip.tile_io == ""
    try:
        replace(h.chip, tile_io="l2")
        raise AssertionError
    except ValueError:
        pass


def test_evaluate_fills_bandwidth():
    from accel_dse.core.evaluate import evaluate
    s = Scenario().replace("model", "llama-3.1-8b").replace("serving.phase", "prefill").replace("serving.batch", 4)
    s0 = s.replace("chip", replace(s.chip, instance_sched="cta"))
    s1 = s.replace("chip", replace(s.chip, instance_sched="cta", tile_io="dram"))
    assert evaluate(s1).ttft > evaluate(s0).ttft
