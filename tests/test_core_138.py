"""0.138: opt-in GEMM_CEILING (published library-best dense-GEMM fractions per chip / format, per-tile main-loop rate
under tile I/O) and tile_io = "persistent" (I/O hidden under the neighbouring tile's main loop)."""
from dataclasses import replace

from accel_dse.core import extval as X
from accel_dse.core.hardware import System
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.refhw import REF_HW


def _cyc(ch, m=8192, k=2048, n=7168, c=8, f="fp8"):
    return gemm_cost(ch, "reconf", m, k, n, count=c, w_fmt=f, a_fmt=f).cycles


def test_persistent_between_cta_and_dram():
    h = REF_HW["H800-SXM"]
    bpc = System(h.chip, h.mem_id, X.EFF["catalog"]).dram_GBps / h.chip.freq_ghz
    base = replace(h.chip, instance_sched="cta")
    d = _cyc(replace(base, tile_io="dram", tile_io_Bpc=bpc))
    p = _cyc(replace(base, tile_io="persistent", tile_io_Bpc=bpc))
    assert p < d and abs(p - _cyc(base)) < 0.02 * _cyc(base)   # main-loop bound: I/O (and epilogue) hidden


def test_gemm_ceiling_values_and_scope():
    assert abs(dict(X.GEMM_CEILING["A100"])["gemm:fp16"] - 0.844) < 1e-3
    assert abs(dict(X.GEMM_CEILING["B200"])["gemm:nvfp4"] - 0.691) < 1e-3
    assert "fp8" not in str(X.GEMM_CEILING["B200"])          # no public B200 fp8 figure: stays at peak
    C = X.candidate_variant()
    r = next(r for r in X.rows() if r["hw"].startswith("A100"))
    ch = X.scenario(r, X.ceil_variant(X.tileio_variant(C))).chip
    assert ("gemm:fp16", 263.4 / 312) in ch.fmt_ceiling and ch.tile_io == "dram"
    assert X.scenario(r, C).chip.fmt_ceiling == ()            # candidate itself unchanged


def test_ceiling_slows_bf16_tiles_only_with_tile_io():
    h = REF_HW["A100-SXM-80GB"]
    bpc = System(h.chip, h.mem_id, X.EFF["catalog"]).dram_GBps / h.chip.freq_ghz
    io = replace(h.chip, instance_sched="cta", tile_io="dram", tile_io_Bpc=bpc)
    a = _cyc(io, f="fp16")
    b = _cyc(replace(io, fmt_ceiling=X.GEMM_CEILING["A100"]), f="fp16")
    assert b > a
