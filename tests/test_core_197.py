"""0.197: L2-resident activations in the GEMM DRAM floor; Op w_bits 0 = unset (format bytes)."""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E, mapping as M
from accel_dse.core.refhw import REF_HW


def _ch(n):
    h = REF_HW[n]
    from accel_dse.core.hardware import System
    s = System(h.chip, h.mem_id)
    return replace(h.chip, tile_io_Bpc=s.dram_GBps / h.chip.freq_ghz, fmt_ceiling=()), s


def test_l2_activation_discount():
    ch, s = _ch("H100-SXM")
    # 256 × 8192 × 4096 bf16 decode GEMM: 6.3 MB of activations sit in L2 (50 MB) → floor = weights only
    c = M.gemm_cost(ch, "reconf", 256, 8192, 4096, w_bits=16.0)
    w = 8192 * 4096 * 2 / ch.tile_io_Bpc
    assert w <= c.cycles < 1.08 * w
    with E.model_flags(CTA_L2_ACT=False):
        c0 = M.gemm_cost(ch, "reconf", 256, 8192, 4096, w_bits=16.0)
    assert c0.cycles > 1.08 * w
    # activations beyond L2 still stream: 65536 tokens × 8192 bf16 = 1 GiB
    big = M.gemm_cost(ch, "reconf", 65536, 8192, 128, w_bits=16.0)
    assert big.cycles >= (65536 * 8192 * 2 + 65536 * 128 * 2 - 50 * 2 ** 20) / ch.tile_io_Bpc


def test_w_bits_zero_is_unset():
    ch, _ = _ch("B200")
    a = M.gemm_cost(ch, "reconf", 64, 8192, 8192, w_bits=0.0)
    b = M.gemm_cost(ch, "reconf", 64, 8192, 8192, w_bits=16.0)
    assert a.cycles == b.cycles
    with E.model_flags(W_BITS_ZERO_FIX=False):
        z = M.gemm_cost(ch, "reconf", 64, 8192, 8192, w_bits=0.0)
    assert z.cycles < 0.25 * b.cycles     # ≤ 0.196: the 128 MB weight operand was free
