"""L3 — hand-checked mapping bounds for each organisation."""
from __future__ import annotations
from dataclasses import replace

import math

from accel_dse.core.dtypes import FormatSupport, gemm_exec
from accel_dse.core.hardware import CHIP_100T, Chip
from accel_dse.core.mapping import gemm_cost

C = replace(CHIP_100T, instance_sched="wide")  # 0.71: hand checks of the wide array (auto default would spread count>1 over cores)
#          # 56×56×16 @1GHz, Ce = 896, port = 4·(56+896)·2 = 7616 B/cycle


def test_100t_geometry():
    assert C.c_eff == 896 and C.macs == 50176
    assert abs(C.peak_tflops_bf16 - 100.352) < 1e-9
    assert C.port_Bpc == 7616


def test_os_hand_check_m1():
    g = gemm_cost(C, "os", 1, 5120, 25600)
    assert g.mac_cycles == 1 * math.ceil(25600 / 896) * 5120 == 148480
    feed = (1 * 5120 * 25600 * 2 + 29 * 1 * 5120 * 2 + 1 * 25600 * 2) / 7616
    assert abs(g.feed_cycles - feed) < 1e-6
    assert g.bound == "mac" and g.dataflow == "os"


def test_os_hand_check_m64_fp8_rate():
    g = gemm_cost(C, "os", 64, 1024, 1792, w_fmt="fp8", a_fmt="fp8", w_bits=8)
    assert g.mac_cycles == math.ceil(64 / 56) * 2 * 1024 / 2.0     # 2048
    assert g.conversion == "none" and g.exec_fmt == "fp8"


def test_ws_edge_equals_os_at_m1_plus_fill():
    g = gemm_cost(C, "ws_edge", 1, 5120, 25600)
    tiles = math.ceil(5120 / 56) * math.ceil(25600 / 896)          # 92·29
    assert g.mac_cycles == tiles * 56 + (56 + 896)
    g10 = gemm_cost(C, "ws_edge", 1, 5120, 25600, count=10)
    assert g10.mac_cycles == 10 * tiles * 56 + (56 + 896)          # fill once per op
    o = gemm_cost(C, "os", 1, 5120, 25600)
    assert abs(g.mac_cycles - o.mac_cycles) / o.mac_cycles < 0.02


def test_ws_broadside_hand_check():
    g = gemm_cost(C, "ws_broad", 1, 5120, 25600)
    tiles = 92 * 29
    assert g.mac_cycles == tiles * 1 + 952
    feed = (5120 * 25600 * 2 + 29 * 5120 * 2 + 25600 * 2) / 7616
    assert abs(g.feed_cycles - feed) < 1e-6 and g.bound == "feed"


def test_ws_psum_spill_beyond_accumulator():
    rows = C.acc_rows                                  # 1 MiB / (4·896) = 292
    assert rows == 292
    a = gemm_cost(C, "ws_broad", rows, 1024, 896)
    b = gemm_cost(C, "ws_broad", rows + 100, 1024, 896)
    spill = 2 * (math.ceil(1024 / 56) - 1) * 100 * 896 * 4 / 7616
    base_b = (1024 * 896 * 2 + 1 * (rows + 100) * 1024 * 2 + (rows + 100) * 896 * 2) / 7616
    assert abs(b.feed_cycles - (base_b + spill)) < 1e-6


def test_gemv_unit_hand_check():
    g = gemm_cost(C, "os_vec", 1, 4096, 4096)
    assert g.dataflow == "gemv"
    assert abs(g.mac_cycles - 4096 * 4096 / (50176 // 8)) < 1e-6
    big = gemm_cost(C, "os_vec", 512, 4096, 4096)
    assert big.dataflow == "os"            # large M → array wins


def test_reconf_is_min_of_candidates():
    for m, k, n in ((1, 7168, 2048), (8, 7168, 4096), (64, 4096, 4096), (300, 2048, 7168), (2048, 4096, 1024)):
        r = gemm_cost(C, "reconf", m, k, n)
        cands = [gemm_cost(C, o, m, k, n).cycles for o in ("os", "ws_edge", "ws_broad")]
        gemv = gemm_cost(C, "os_vec", m, k, n)
        cands.append(gemv.cycles)
        assert abs(r.cycles - min(cands)) < 1e-9


def test_count_scales_linearly():
    a = gemm_cost(C, "os", 3, 7168, 4096)
    b = gemm_cost(C, "os", 3, 7168, 4096, count=10)
    assert abs(b.cycles - 10 * a.cycles) < 1e-9


def test_dtype_exec_rules():
    hw = FormatSupport()
    assert gemm_exec("fp8", "fp8", hw)[:3] == ("fp8", 2.0, "none")
    assert gemm_exec("mxfp4", "bf16", hw)[2] == "dequant_w"
    assert gemm_exec("int4", "bf16", hw)[2] == "dequant_w"
    assert gemm_exec("fp8", "bf16", hw)[2] == "dequant_w"
    no8 = FormatSupport(rates=(("bf16", 1.0),))
    assert gemm_exec("fp8", "fp8", no8)[:3] == ("bf16", 1.0, "upcast_both")
    g = gemm_cost(C, "os", 4, 2048, 2048, w_fmt="mxfp4", a_fmt="bf16", w_bits=4.25)
    assert g.conversion == "dequant_w" and g.convert_elems == 2048 * 2048


def test_port_bandwidth_is_design_variable():
    wide = Chip("w", 1.0, 56, 56, 16, sram_port_Bpc=4 * 7616)
    a = gemm_cost(C, "ws_broad", 1, 8192, 8192)
    b = gemm_cost(wide, "ws_broad", 1, 8192, 8192)
    assert abs(a.feed_cycles / b.feed_cycles - 4) < 1e-9
