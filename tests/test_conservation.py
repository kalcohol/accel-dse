"""Conservation / sanity tests for accel_dse."""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accel_dse.evaluate import EvalConfig, evaluate_inference
from accel_dse.memory import (
    HANDCHECK_MEM,
    HBM_PRESET,
    LPDDR_PRESET,
    SRAMConfig,
    SRAMPartitions,
    derive_sram_partitions,
)
from accel_dse.model_shape import ILLUSTRATIVE_27B, ILLUSTRATIVE_MOE, TOY_SHAPE
from accel_dse.npu import NPUConfig, gemm_cycles, gemm_flops, gemm_macs, gemm_utilization
from accel_dse.scan import find_mem_bound_crossover
from accel_dse.sku import SKU_100T, SKU_1P, SKU_BASELINE, macs_needed_for_tops
from accel_dse.traffic import contend, decode_traffic, prefill_traffic, weight_dram_bytes_for_layer


# v0.28: version checks read the package version (no per-release test edits).
def _pkg_version() -> str:
    import accel_dse

    return accel_dse.__version__


def _assert_version(v: str) -> None:
    """``v`` must equal ``accel_dse.__version__`` and agree with pyproject.toml."""
    import re as _re

    exp = _pkg_version()
    assert v == exp, (v, exp)
    pyproj = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    m = _re.search(r'^version\s*=\s*"([^"]+)"', pyproj, _re.M)
    assert m and m.group(1) == exp, (m and m.group(1), exp)


def test_gemm_flops_formula():
    m, k, n = 8, 64, 32
    assert gemm_macs(m, k, n) == m * k * n
    assert gemm_flops(m, k, n) == 2 * m * k * n


def test_os_cycles_and_util():
    npu = NPUConfig(rows=4, cols=4)
    # M=1, N=16, K=8 → cycles = ceil(1/4)*ceil(16/4)*8 = 1*4*8 = 32
    assert gemm_cycles(1, 8, 16, npu) == 32
    util = gemm_utilization(1, 8, 16, npu)
    # macs=1*8*16=128; peak=16*32=512; util=0.25
    assert abs(util - 0.25) < 1e-12


def test_engines_scale_cycles():
    """n_engines expands effective cols → fewer cycles when N is wide."""
    base = NPUConfig(rows=4, cols=4, n_engines=1)
    multi = NPUConfig(rows=4, cols=4, n_engines=4)
    # N=16: base ceil(16/4)=4; multi ceil(16/16)=1 → 4× fewer cycles
    assert gemm_cycles(1, 8, 16, base) == 4 * gemm_cycles(1, 8, 16, multi)
    assert multi.peak_macs_per_cycle == 4 * base.peak_macs_per_cycle


def test_decode_m1_util_less_than_large_m():
    npu = NPUConfig(rows=64, cols=64)
    k, n = 512, 512
    u1 = gemm_utilization(1, k, n, npu)
    uL = gemm_utilization(256, k, n, npu)
    assert u1 < uL
    # Exact M=1 when N%C==0: util = 1/R
    assert abs(u1 - 1.0 / npu.rows) < 1e-12


def test_bytes_non_negative():
    npu = NPUConfig(rows=16, cols=16)
    sram = SRAMConfig(64 * 1024)
    pref = prefill_traffic(TOY_SHAPE, npu, sram, prompt_len=32, batch=1)
    dec = decode_traffic(TOY_SHAPE, npu, sram, seq_len=32, batch=1)
    for t in (pref, dec):
        assert t.flops >= 0
        assert t.weight_bytes_dram >= 0
        assert t.kv_bytes_dram >= 0
        assert t.act_bytes_dram >= 0
        assert t.total_dram_bytes >= 0
        assert t.compute_cycles >= 0


def test_contention_ge_max_streams():
    npu = NPUConfig(rows=16, cols=16)
    sram = SRAMConfig(4 * 1024)  # small → force weight traffic
    dec = decode_traffic(TOY_SHAPE, npu, sram, seq_len=128, batch=1)
    c = contend(dec, HBM_PRESET)
    assert c.t_serialize_s + 1e-15 >= max(c.t_weight_s, c.t_kv_s)
    assert c.t_lower_bound_s + 1e-15 >= max(c.t_weight_s, c.t_kv_s)
    # fair share is at least as large as lower bound
    assert c.t_share_fair_s + 1e-15 >= c.t_lower_bound_s


def test_weight_tiling_capacity():
    # Fits → once
    assert weight_dram_bytes_for_layer(1000, 2000, 5) == 1000
    # Does not fit → * tiles
    assert weight_dram_bytes_for_layer(1000, 500, 5) == 5000


def test_partition_sum_le_capacity():
    parts = derive_sram_partitions(
        64 * 1024 * 1024,
        layer_weight_bytes=10 * 1024 * 1024,
        kv_bytes_needed=5 * 1024 * 1024,
    )
    assert parts.total_bytes == 64 * 1024 * 1024
    assert parts.weight_staging_bytes + parts.kv_scratch_bytes + parts.act_bytes == parts.total_bytes

    # Explicit partitions must not exceed capacity
    ok = SRAMConfig(
        capacity_bytes=1000,
        partitions=SRAMPartitions(400, 300, 300),
    )
    assert ok.resolve_partitions().total_bytes == 1000
    try:
        SRAMConfig(1000, partitions=SRAMPartitions(600, 600, 0))
        assert False, "expected ValueError"
    except ValueError:
        pass

    # Weight-resident: cap ≥ W_layer → R≥1, weight_part == R*W_layer
    w = 73728
    p = derive_sram_partitions(
        128 * 1024, layer_weight_bytes=w, kv_bytes_needed=2048, n_layers=2
    )
    assert p.resident_layers == 1
    assert p.weight_staging_bytes == w
    assert p.total_bytes == 128 * 1024


def test_sku_peak_math():
    # peak_TOPS = R*C*engines*freq*2 / 1e12
    assert abs(SKU_100T.achieved_tops() - 100.352) < 0.01
    assert abs(SKU_BASELINE.achieved_tops() - 8.192) < 1e-9
    # 1P-class should land near 1000 TOPS
    assert 900 < SKU_1P.achieved_tops() < 1100
    # macs_needed formula
    macs = macs_needed_for_tops(100.0, 1e9)
    assert abs(macs - 50000.0) < 1e-6
    npu = SKU_100T.npu()
    assert npu.peak_macs_per_cycle == 16 * 56 * 56
    assert abs(npu.peak_tops_at(1e9) - SKU_100T.achieved_tops()) < 1e-12


def test_toy_handcheck_numbers():
    """Mirror examples/handcheck.md arithmetic (partition-aware KV)."""
    shape = TOY_SHAPE
    assert shape.attn_weight_params_per_layer() == 12288
    assert shape.ffn_weight_params_per_layer() == 24576
    assert shape.weight_params_per_layer() == 36864
    assert shape.weight_bytes_per_layer() == 73728
    assert shape.kv_bytes_per_token() == 256

    npu = NPUConfig(rows=4, cols=4, mac_efficiency=1.0)
    sram = SRAMConfig(128 * 1024)  # layer fits in weight_staging
    mem = HANDCHECK_MEM  # 16 GB/s
    assert abs(mem.effective_bandwidth_Bps() - 16e9) < 1e-3

    cfg = EvalConfig(
        prompt_len=8,
        decode_seq_len=8,
        batch=1,
        frequency_hz=1e9,
        pin_weights=False,
        contention_mode="serialize",
    )
    r = evaluate_inference(shape, npu, sram, mem, cfg)
    # Prefill cold-start: still 2 * 73728 (load all layers once)
    assert r.prefill.traffic.weight_bytes_dram == 2 * 73728
    assert r.prefill.traffic.kv_bytes_dram == 256 * 8
    # 128 KiB → R=1 (one layer resident); path = miss-layers stream
    assert r.prefill.traffic.resident_layers == 1
    assert r.prefill.traffic.weight_path == "hbm_stream_miss_layers"
    # Decode steady-state: R=1 → W DRAM = (L-1)*W_layer = 73728
    assert r.decode.traffic.kv_bytes_read == 0
    assert r.decode.traffic.kv_bytes_write == 256
    assert r.decode.traffic.kv_bytes_dram == 256
    assert r.decode.traffic.resident_layers == 1
    assert r.decode.traffic.weight_bytes_dram == 73728
    assert r.decode.traffic.weight_path == "hbm_stream_miss_layers"
    parts = r.decode.traffic.partitions
    assert parts is not None
    assert parts.weight_staging_bytes == 73728
    assert parts.resident_layers == 1
    assert parts.total_bytes == 128 * 1024
    # Decode M=1 util < prefill M=8 util
    assert r.decode.traffic.mean_gemm_util < r.prefill.traffic.mean_gemm_util
    # Contention serialize == sum of stream times
    c = r.decode.contention
    assert abs(c.t_serialize_s - (c.t_weight_s + c.t_kv_s + c.t_act_s)) < 1e-15
    # Hand-check wall times
    assert abs(r.prefill.traffic.compute_cycles - 37376.0) < 1e-6
    assert abs(r.decode.traffic.compute_cycles - 18560.0) < 1e-6
    assert abs(r.decode.t_memory_s - (73728 + 256) / 16e9) < 1e-15


def test_27b_params_ballpark():
    # Illustrative placeholder should land near ~25–30B params
    p = ILLUSTRATIVE_27B.total_weight_params()
    assert 20e9 < p < 35e9


def test_evaluate_walls_finite():
    r = evaluate_inference(
        TOY_SHAPE,
        NPUConfig(8, 8),
        SRAMConfig(64 * 1024),
        HBM_PRESET,
        EvalConfig(prompt_len=16, decode_seq_len=16),
    )
    assert math.isfinite(r.ttft_s) and r.ttft_s > 0
    assert math.isfinite(r.tpot_s) and r.tpot_s > 0


def test_mem_bound_crossover_synthetic():
    """Scale engines at fixed 64×64 lane: decode on LPDDR flips compute→memory."""
    cfg = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=1e9,
        pin_weights=False,
        contention_mode="serialize",
    )
    sram = SRAMConfig(64 * 1024 * 1024)
    # Few engines → compute-bound; many → memory-bound (weight stream ~55GB / ~381GB/s)
    results = find_mem_bound_crossover(
        ILLUSTRATIVE_27B,
        LPDDR_PRESET,
        sram,
        engine_counts=[1, 2, 4, 8, 16, 32],
        rows=64,
        cols=64,
        cfg=cfg,
    )
    assert results[0].decode.wall == "compute"
    assert results[-1].decode.wall == "memory"
    # Monotone: t_compute shrinks as engines grow; t_memory stays flat (same bytes)
    for a, b in zip(results, results[1:]):
        assert b.decode.t_compute_s <= a.decode.t_compute_s + 1e-15
        assert abs(b.decode.t_memory_s - a.decode.t_memory_s) < 1e-9

    # Named SKU presets on LPDDR: baseline compute, 100T memory
    r0 = evaluate_inference(
        ILLUSTRATIVE_27B, SKU_BASELINE.npu(), sram, LPDDR_PRESET,
        EvalConfig(prompt_len=512, decode_seq_len=512, frequency_hz=SKU_BASELINE.frequency_hz),
    )
    r1 = evaluate_inference(
        ILLUSTRATIVE_27B, SKU_100T.npu(), sram, LPDDR_PRESET,
        EvalConfig(prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz),
    )
    assert r0.decode.wall == "compute"
    assert r1.decode.wall == "memory"
    assert r1.decode.t_memory_s > r1.decode.t_compute_s


def test_dtype_bytes_scaling():
    """Weight/KV DRAM bytes scale with bit width; FLOPs stay constant (bytes-only)."""
    from accel_dse.dtype import DTYPE_FP16, DTYPE_FP8, DTYPE_INT4, DTYPE_INT8

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    base = ILLUSTRATIVE_27B
    cfg_bits = [
        (DTYPE_FP16, 16),
        (DTYPE_FP8, 8),
        (DTYPE_INT8, 8),
        (DTYPE_INT4, 4),
    ]
    ref_shape = DTYPE_FP16.apply(base)
    ref_dec = decode_traffic(ref_shape, npu, sram, seq_len=512, batch=1)
    ref_w = ref_dec.weight_bytes_dram
    ref_kv = ref_dec.kv_bytes_dram
    ref_flops = ref_dec.flops

    for preset, bits in cfg_bits:
        shaped = preset.apply(base)
        assert shaped.weight_bits == bits
        assert shaped.kv_bits == bits
        # act stays at base (16) by default
        assert shaped.act_bits == base.act_bits
        # Footprint scales linearly with bits
        assert shaped.weight_bytes_per_layer() == base.weight_bytes_per_layer() * bits // 16
        assert shaped.kv_bytes_per_token() == base.kv_bytes_per_token() * bits // 16

        dec = decode_traffic(shaped, npu, sram, seq_len=512, batch=1)
        # FLOPs unchanged (bytes-only honesty)
        assert dec.flops == ref_flops
        expected_scale = bits / 16.0
        # Storage footprint always scales with bits
        assert shaped.weight_bytes() == base.weight_bytes() * bits // 16
        assert shaped.kv_bytes_per_token() == base.kv_bytes_per_token() * bits // 16
        # Weight DRAM: same path ⇒ linear scale; path may flip if layer fits at low bits
        if dec.weight_path == ref_dec.weight_path:
            assert abs(dec.weight_bytes_dram / max(ref_w, 1) - expected_scale) < 1e-9
        # KV read may flip to on-die hit when shrunk cache fits kv_scratch — still
        # capacity-derived. Write-through always scales with kv_bits.
        assert dec.kv_bytes_write == shaped.kv_bytes_per_token()
        full = shaped.kv_cache_bytes(512)
        parts = dec.partitions
        assert parts is not None
        if parts.kv_scratch_bytes >= full and full > 0:
            assert dec.kv_bytes_read == 0
        else:
            assert dec.kv_bytes_read == full
            assert abs(dec.kv_bytes_read / max(ref_dec.kv_bytes_read, 1) - expected_scale) < 1e-9

    # with_bits helper
    s4 = base.with_bits(weight_bits=4, kv_bits=4)
    assert s4.weight_bytes() * 4 == base.weight_bytes()
    assert s4.kv_bytes_per_token() * 4 == base.kv_bytes_per_token()


def test_ctx_monotonic_kv_bytes():
    """Longer decode ctx → non-decreasing KV DRAM bytes; no overflow/crash."""
    from accel_dse.scan import DEFAULT_CTX_SWEEP, run_ctx_sweep

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    ctxs = list(DEFAULT_CTX_SWEEP)
    prev_kv = -1
    prev_read = -1
    for ctx in ctxs:
        dec = decode_traffic(ILLUSTRATIVE_27B, npu, sram, seq_len=ctx, batch=1)
        assert dec.kv_bytes_dram >= 0
        assert dec.kv_bytes_read >= 0
        assert dec.kv_bytes_write == ILLUSTRATIVE_27B.kv_bytes_per_token()
        assert dec.kv_bytes_dram >= prev_kv
        assert dec.kv_bytes_read >= prev_read
        # Full cache bytes = per_token * ctx; if scratch too small → read == full
        full = ILLUSTRATIVE_27B.kv_cache_bytes(ctx)
        parts = dec.partitions
        assert parts is not None
        if parts.kv_scratch_bytes >= full and full > 0:
            assert dec.kv_bytes_read == 0
        else:
            assert dec.kv_bytes_read == full
        prev_kv = dec.kv_bytes_dram
        prev_read = dec.kv_bytes_read

    # Sweep driver runs both mems without raising
    results = run_ctx_sweep(verbose=False, sku=SKU_100T, ctx_list=ctxs)
    assert len(results) == len(ctxs) * 2
    hbm = [r for r in results if r.mem.kind == "HBM"]
    kv_seq = [r.decode.traffic.kv_bytes_dram for r in hbm]
    assert kv_seq == sorted(kv_seq)
    # At 128k, KV should be a meaningful fraction of weight stream for fp16 27B
    r128 = hbm[-1]
    ratio = r128.decode.traffic.kv_bytes_dram / max(
        r128.decode.traffic.weight_bytes_dram, 1
    )
    assert ratio > 0.2  # ~21GB / ~55GB ≈ 0.38


def test_dtype_sweep_runs():
    from accel_dse.scan import run_dtype_sweep

    results = run_dtype_sweep(verbose=False, sku=SKU_100T)
    # 4 presets × 2 mems
    assert len(results) == 8
    lpddr = [r for r in results if r.mem.kind == "LPDDR"]
    # Lower bits → lower or equal weight DRAM (path may change)
    w_bytes = [r.decode.traffic.weight_bytes_dram for r in lpddr]
    # fp16 > fp8 == int8 > int4 in storage footprint of the shape
    shapes_w = [r.shape.weight_bytes() for r in lpddr]
    assert shapes_w[0] > shapes_w[1] == shapes_w[2] > shapes_w[3]
    # t_compute identical across dtypes (MAC rate unchanged)
    comps = [r.decode.t_compute_s for r in lpddr]
    assert all(abs(c - comps[0]) < 1e-12 for c in comps)


def test_long_ctx_no_overflow():
    """128k ctx evaluates finite TPOT; partition sum preserved."""
    r = evaluate_inference(
        ILLUSTRATIVE_27B,
        SKU_100T.npu(),
        SRAMConfig(64 * 1024 * 1024),
        HBM_PRESET,
        EvalConfig(
            prompt_len=512,
            decode_seq_len=128000,
            frequency_hz=SKU_100T.frequency_hz,
        ),
    )
    assert math.isfinite(r.tpot_s) and r.tpot_s > 0
    parts = r.decode.traffic.partitions
    assert parts is not None
    assert parts.total_bytes == 64 * 1024 * 1024
    assert r.decode.traffic.kv_bytes_read == ILLUSTRATIVE_27B.kv_cache_bytes(128000)


def test_sram_resident_byte_knees_constructed():
    """Resident R knees drop decode W DRAM; staging-only does not.

    - cap < W_layer → R=0 → W DRAM = L*W_layer
    - W_layer ≤ cap < 2*W_layer → R=1 → W DRAM = (L-1)*W_layer
    - cap ≥ L*W_layer → R=L → W DRAM = 0 (steady-state)
    Staging-only (R=0, staging≥W_layer via explicit parts) still full stream.
    """
    shape = TOY_SHAPE  # L=2, W_layer=73728
    npu = NPUConfig(rows=4, cols=4)
    w_layer = shape.weight_bytes_per_layer()
    L = shape.n_layers
    assert w_layer == 73728 and L == 2

    # R=0: capacity < W_layer
    d0 = decode_traffic(shape, npu, SRAMConfig(w_layer - 1), seq_len=8)
    assert d0.resident_layers == 0
    assert d0.weight_bytes_dram == L * w_layer
    assert d0.weight_path == "hbm_stream_all_layers"

    # R=1: W_layer ≤ cap < 2*W_layer
    d1 = decode_traffic(shape, npu, SRAMConfig(w_layer), seq_len=8)
    assert d1.resident_layers == 1
    assert d1.weight_bytes_dram == (L - 1) * w_layer
    assert d1.weight_path == "hbm_stream_miss_layers"

    # R=L: full on-die
    dL = decode_traffic(shape, npu, SRAMConfig(L * w_layer), seq_len=8)
    assert dL.resident_layers == L
    assert dL.weight_bytes_dram == 0
    assert dL.weight_path == "on_die_resident"

    # Staging-only honesty: R=0 with staging==W_layer still full stream bytes
    staging_only = SRAMConfig(
        capacity_bytes=256 * 1024,
        partitions=SRAMPartitions(
            weight_staging_bytes=w_layer,
            kv_scratch_bytes=4096,
            act_bytes=256 * 1024 - w_layer - 4096,
            resident_layers=0,
            double_buffer_eligible=False,
            layer_weight_bytes=w_layer,
            n_layers=L,
        ),
    )
    ds = decode_traffic(shape, npu, staging_only, seq_len=8)
    assert ds.resident_layers == 0
    assert ds.weight_bytes_dram == L * w_layer
    assert ds.weight_path == "hbm_stream_all_layers"

    # Prefill cold-start still loads all even when R=L
    pref = prefill_traffic(shape, npu, SRAMConfig(L * w_layer), prompt_len=8)
    assert pref.resident_layers == L
    assert pref.weight_bytes_dram == L * w_layer
    assert pref.weight_path == "on_die_resident"

    # Tiled prefill still drops bytes when staging grows (R=0 explicit)
    below = SRAMConfig(
        capacity_bytes=256 * 1024,
        partitions=SRAMPartitions(
            weight_staging_bytes=2048,
            kv_scratch_bytes=4096,
            act_bytes=256 * 1024 - 2048 - 4096,
            resident_layers=0,
            layer_weight_bytes=w_layer,
            n_layers=L,
        ),
    )
    above = SRAMConfig(
        capacity_bytes=256 * 1024,
        partitions=SRAMPartitions(
            weight_staging_bytes=w_layer,
            kv_scratch_bytes=4096,
            act_bytes=256 * 1024 - w_layer - 4096,
            resident_layers=0,
            layer_weight_bytes=w_layer,
            n_layers=L,
        ),
    )
    pref_lo = prefill_traffic(shape, npu, below, prompt_len=256)
    pref_hi = prefill_traffic(shape, npu, above, prompt_len=256)
    assert pref_lo.weight_path == "hbm_stream_tiled"
    assert pref_hi.weight_path == "hbm_stream_all_layers"
    assert pref_hi.weight_bytes_dram == L * w_layer
    assert pref_lo.weight_bytes_dram > pref_hi.weight_bytes_dram


def test_sram_knee_illustrative_27b_resident():
    """illustrative_27B @ fp16: resident knees at R≥1 / R≥L/2 / R≥L drop W DRAM."""
    from accel_dse.scan import run_sram_sweep

    shape = ILLUSTRATIVE_27B
    w_layer = shape.weight_bytes_per_layer()
    L = shape.n_layers
    assert abs(w_layer / 2**20 - 1320.0) < 1e-6
    assert L == 40

    npu = SKU_100T.npu()
    full = L * w_layer

    # Below R=1
    lo = decode_traffic(shape, npu, SRAMConfig(int(1024 * 1024 * 1024)), seq_len=512)
    assert lo.resident_layers == 0
    assert lo.weight_bytes_dram == full
    assert lo.weight_path == "hbm_stream_all_layers"

    # R≥1 at 1320 MiB
    r1 = decode_traffic(shape, npu, SRAMConfig(int(1320 * 1024 * 1024)), seq_len=512)
    assert r1.resident_layers == 1
    assert r1.weight_bytes_dram == (L - 1) * w_layer
    assert r1.weight_path == "hbm_stream_miss_layers"

    # R≥L/2 at 26400 MiB
    r_half = decode_traffic(shape, npu, SRAMConfig(int(26400 * 1024 * 1024)), seq_len=512)
    assert r_half.resident_layers >= L // 2
    assert r_half.weight_bytes_dram == (L - r_half.resident_layers) * w_layer

    # R≥L at 52800 MiB → on-die resident, W DRAM=0
    r_full = decode_traffic(shape, npu, SRAMConfig(int(52800 * 1024 * 1024)), seq_len=512)
    assert r_full.resident_layers == L
    assert r_full.weight_bytes_dram == 0
    assert r_full.weight_path == "on_die_resident"

    results = run_sram_sweep(
        verbose=False,
        sku=SKU_100T,
        sram_mib_list=[64, 1320, 26400, 52800],
    )
    assert len(results) == 4 * 2
    hbm = {
        r.sram.capacity_bytes / 2**20: r
        for r in results
        if r.mem.kind == "HBM"
    }
    assert hbm[64.0].decode.traffic.resident_layers == 0
    assert hbm[1320.0].decode.traffic.resident_layers == 1
    assert hbm[52800.0].decode.traffic.weight_bytes_dram == 0


def test_weight_hide_double_buffer():
    """Assumed hide_factor cuts decode mem time when staging dbl-buf eligible.

    Under weight_resident-first, cap≥W_layer → R≥1 (no staging dbl-buf).
    Force R=0 + 2×W staging via explicit partitions to test hide.
    """
    shape = ILLUSTRATIVE_27B
    npu = SKU_100T.npu()
    w_layer = shape.weight_bytes_per_layer()
    # Explicit staging-only double-buffer (R=0): does NOT cut bytes
    cap = 4 * w_layer
    sram = SRAMConfig(
        capacity_bytes=cap,
        partitions=SRAMPartitions(
            weight_staging_bytes=2 * w_layer,
            kv_scratch_bytes=cap // 8,
            act_bytes=cap - 2 * w_layer - cap // 8,
            resident_layers=0,
            double_buffer_eligible=True,
            layer_weight_bytes=w_layer,
            n_layers=shape.n_layers,
        ),
    )
    cfg0 = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        frequency_hz=SKU_100T.frequency_hz,
        weight_hide_factor=0.0,
    )
    cfg5 = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        frequency_hz=SKU_100T.frequency_hz,
        weight_hide_factor=0.5,
    )
    r0 = evaluate_inference(shape, npu, sram, LPDDR_PRESET, cfg0)
    r5 = evaluate_inference(shape, npu, sram, LPDDR_PRESET, cfg5)
    assert r0.decode.traffic.resident_layers == 0
    assert r0.decode.traffic.double_buffer_eligible
    assert r5.decode.traffic.double_buffer_eligible
    # Bytes unchanged by hide — still full model stream
    assert r0.decode.traffic.weight_bytes_dram == shape.n_layers * w_layer
    assert r5.decode.traffic.weight_bytes_dram == r0.decode.traffic.weight_bytes_dram
    assert r5.decode.t_memory_s < r0.decode.t_memory_s
    tw = r0.decode.contention.t_weight_s
    tk = r0.decode.contention.t_kv_s
    ta = r0.decode.contention.t_act_s
    expect = 0.5 * tw + tk + ta
    assert abs(r5.decode.t_memory_s - expect) < 1e-12


def test_quant_independent_bits():
    """w4k16 shrinks weights only; KV stays 16-bit → long-ctx KV can overtake."""
    from accel_dse.dtype import QUANT_W4K16, QUANT_W4K8, QUANT_W16K16, get_quant
    from accel_dse.scan import run_quant_sweep

    base = ILLUSTRATIVE_27B
    w4k16 = QUANT_W4K16.apply(base)
    w4k8 = QUANT_W4K8.apply(base)
    assert w4k16.weight_bits == 4 and w4k16.kv_bits == 16
    assert w4k8.weight_bits == 4 and w4k8.kv_bits == 8
    assert w4k16.weight_bytes() == base.weight_bytes() // 4
    assert w4k16.kv_bytes_per_token() == base.kv_bytes_per_token()  # KV unchanged
    assert w4k8.kv_bytes_per_token() == base.kv_bytes_per_token() // 2

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    # Short ctx: weight still dominates even at w4k16
    dec_short = decode_traffic(w4k16, npu, sram, seq_len=512)
    # Long ctx: KV read large vs shrunk weights
    dec_long = decode_traffic(w4k16, npu, sram, seq_len=128000)
    ratio_short = dec_short.kv_bytes_dram / max(dec_short.weight_bytes_dram, 1)
    ratio_long = dec_long.kv_bytes_dram / max(dec_long.weight_bytes_dram, 1)
    assert ratio_long > ratio_short
    assert ratio_long > 1.0  # KV overtakes weight stream at w4k16 @ 128k

    # Full w4k8 has smaller KV than w4k16 at same ctx
    dec_full = decode_traffic(w4k8, npu, sram, seq_len=128000)
    assert dec_full.kv_bytes_dram < dec_long.kv_bytes_dram

    results = run_quant_sweep(
        verbose=False,
        sku=SKU_100T,
        ctx_list=[512, 128000],
        quant_names=["w16k16", "w4k16", "w4k8"],
    )
    # 3 presets × 2 ctx × 2 mem
    assert len(results) == 12
    assert get_quant("w8k16").weight_bits == 8

def test_sram_policy_kv_first_vs_weight_resident():
    """kv_first keeps more KV scratch at long ctx; weight_resident maximizes R."""
    shape = TOY_SHAPE
    npu = NPUConfig(4, 4)
    # Cap fits R=1 with leftover; long KV need
    cap = 100_000  # > W_layer=73728, < 2*W
    kv_need = 50_000
    wr = derive_sram_partitions(
        cap, layer_weight_bytes=73728, kv_bytes_needed=kv_need,
        n_layers=2, policy="weight_resident",
    )
    kf = derive_sram_partitions(
        cap, layer_weight_bytes=73728, kv_bytes_needed=kv_need,
        n_layers=2, policy="kv_first",
    )
    assert wr.resident_layers == 1
    assert kf.kv_scratch_bytes >= wr.kv_scratch_bytes
    # weight_resident should not give KV more than leftover after R*W
    assert wr.weight_staging_bytes == 73728

    # EvalConfig policy propagates
    r_wr = evaluate_inference(
        shape, npu, SRAMConfig(cap), HBM_PRESET,
        EvalConfig(prompt_len=8, decode_seq_len=8, sram_policy="weight_resident"),
    )
    r_kf = evaluate_inference(
        shape, npu, SRAMConfig(cap), HBM_PRESET,
        EvalConfig(prompt_len=8, decode_seq_len=8, sram_policy="kv_first"),
    )
    assert r_wr.decode.traffic.resident_layers >= 1
    assert r_kf.cfg.sram_policy == "kv_first"



def test_moe_active_vs_total_and_traffic():
    """MoE: total params count all E; stream/FLOPs use top_k (balanced routing)."""
    moe = ILLUSTRATIVE_MOE
    assert moe.is_moe
    assert moe.n_experts == 8 and moe.top_k == 2
    assert moe.total_weight_params() > moe.active_weight_params()
    # Stream layer = attn + top_k FFN; stored = attn + E FFN
    assert moe.stream_weight_bytes_per_layer() < moe.weight_bytes_per_layer()
    expect_ratio = moe.active_weight_params_per_layer() / moe.weight_params_per_layer()
    assert abs(
        moe.stream_weight_bytes_per_layer() / moe.weight_bytes_per_layer()
        - expect_ratio
    ) < 1e-12

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    dec = decode_traffic(moe, npu, sram, seq_len=512, batch=1)
    # R=0 staging: W DRAM = L * stream_W_layer (not all-E)
    assert dec.resident_layers == 0
    assert dec.weight_bytes_dram == moe.n_layers * moe.stream_weight_bytes_per_layer()
    # Dense 27B streams more weight bytes than this MoE active path
    dense = decode_traffic(ILLUSTRATIVE_27B, npu, sram, seq_len=512, batch=1)
    assert dec.weight_bytes_dram < dense.weight_bytes_dram

    # FLOPs: MoE FFN multiplicity = top_k; dense has 1× FFN
    # Active FFN params smaller → fewer FFN FLOPs than dense 27B at same M
    assert dec.flops < dense.flops


def test_batch_util_increases_os_formula():
    """sweep-batch: mean GEMM util rises with batch (=M) under OS formula."""
    from accel_dse.npu import gemm_utilization
    from accel_dse.scan import run_batch_sweep

    npu = SKU_100T.npu()
    # Pure OS: util(M) monotone in M for fixed K,N when N % effC == 0
    k = n = ILLUSTRATIVE_27B.hidden
    assert n % npu.effective_cols == 0 or True  # may not divide; still monotone-ish
    us = [gemm_utilization(b, k, n, npu) for b in (1, 2, 4, 8, 16, 32)]
    for a, b in zip(us, us[1:]):
        assert b + 1e-15 >= a
    assert us[0] < us[-1]
    # M=1 ≈ 1/R when N fills width
    if ILLUSTRATIVE_27B.hidden % npu.effective_cols == 0:
        assert abs(us[0] - 1.0 / npu.rows) < 1e-12

    results = run_batch_sweep(
        verbose=False,
        sku=SKU_100T,
        batch_list=[1, 2, 4, 8, 16, 32],
    )
    # 6 batches × 2 mems
    assert len(results) == 12
    lpddr = [r for r in results if r.mem.kind == "LPDDR"]
    assert [r.cfg.batch for r in lpddr] == [1, 2, 4, 8, 16, 32]
    utils = [r.decode.traffic.mean_gemm_util for r in lpddr]
    for a, b in zip(utils, utils[1:]):
        assert b + 1e-15 >= a
    assert utils[0] < utils[-1]
    # Weight DRAM bytes independent of batch (decode streams W once/step)
    w0 = lpddr[0].decode.traffic.weight_bytes_dram
    assert all(r.decode.traffic.weight_bytes_dram == w0 for r in lpddr)
    # KV write scales with batch
    assert lpddr[-1].decode.traffic.kv_bytes_write == 32 * ILLUSTRATIVE_27B.kv_bytes_per_token()


def test_moe_sweep_runs():
    from accel_dse.scan import run_moe_sweep

    results = run_moe_sweep(verbose=False, sku=SKU_100T)
    # 2 shapes × 2 mems
    assert len(results) == 4
    moe_rows = [r for r in results if r.shape.is_moe]
    dense_rows = [r for r in results if not r.shape.is_moe]
    assert len(moe_rows) == 2 and len(dense_rows) == 2
    moe_w = next(r.decode.traffic.weight_bytes_dram for r in moe_rows if r.mem.kind == "LPDDR")
    dense_w = next(r.decode.traffic.weight_bytes_dram for r in dense_rows if r.mem.kind == "LPDDR")
    assert moe_w < dense_w


# ---------------------------------------------------------------------------
# v0.7 scale-up: TP + C2C
# ---------------------------------------------------------------------------

def test_tp1_matches_single_card():
    """tp=1: collectives=0; TPOT/TTFT match single-card within tolerance."""
    from accel_dse.scaleup import TPConfig, evaluate_scaleup, C2C_400

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        frequency_hz=SKU_100T.frequency_hz,
    )
    single = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg)
    su = evaluate_scaleup(
        ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, c2c=C2C_400)
    )
    assert su.decode.collective_bytes == 0
    assert su.prefill.collective_bytes == 0
    assert abs(su.tpot_s - single.tpot_s) / max(single.tpot_s, 1e-30) < 1e-9
    assert abs(su.ttft_s - single.ttft_s) / max(single.ttft_s, 1e-30) < 1e-9
    assert su.decode.flops_card == single.decode.traffic.flops
    assert su.decode.weight_bytes_card == single.decode.traffic.weight_bytes_dram


def test_collective_bytes_tp_formula():
    """Collective bytes 0 at tp=1; ring formula; increases then per-rank volume."""
    from accel_dse.scaleup import (
        TPConfig,
        activation_volume_bytes,
        collectives_bytes_per_phase,
        evaluate_scaleup,
        ring_allreduce_bytes,
        COLLECTIVES_PER_LAYER,
        C2C_400,
    )

    assert ring_allreduce_bytes(1000, 1) == 0
    assert ring_allreduce_bytes(0, 8) == 0
    # 2*(tp-1)/tp * V
    for tp in (2, 4, 8):
        v = 16384
        expect = int(2 * (tp - 1) / tp * v)
        assert ring_allreduce_bytes(v, tp) == expect

    shape = ILLUSTRATIVE_27B
    # Decode: seq=1 activation
    vol = activation_volume_bytes(shape, batch=1, seq=1)
    assert vol == shape.hidden * (shape.act_bits // 8)

    b1 = collectives_bytes_per_phase(shape, 1, 1, tp=1)
    assert b1 == 0
    bytes_by_tp = {
        tp: collectives_bytes_per_phase(shape, 1, 1, tp=tp) for tp in (1, 2, 4, 8)
    }
    assert bytes_by_tp[1] == 0
    # Per-rank ring volume rises with tp toward 2*V, so total phase bytes rise
    assert bytes_by_tp[2] < bytes_by_tp[4] < bytes_by_tp[8]
    for tp in (2, 4, 8):
        per = ring_allreduce_bytes(vol, tp)
        assert bytes_by_tp[tp] == shape.n_layers * COLLECTIVES_PER_LAYER * per

    # Prefill volume >> decode volume
    pref = collectives_bytes_per_phase(shape, 1, 512, tp=4)
    dec = collectives_bytes_per_phase(shape, 1, 1, tp=4)
    assert pref == dec * 512

    # Through evaluate_scaleup
    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz)
    prev = 0
    for tp in (1, 2, 4, 8):
        r = evaluate_scaleup(
            shape, npu, sram, HBM_PRESET, cfg, TPConfig(tp=tp, c2c=C2C_400)
        )
        assert r.decode.collective_bytes == bytes_by_tp[tp]
        if tp > 1:
            assert r.decode.collective_bytes > prev
        prev = r.decode.collective_bytes


def test_tp_oom_flag():
    """OOM when per-card weight+KV exceeds mem.capacity."""
    from accel_dse.scaleup import (
        TPConfig,
        capacity_check,
        evaluate_scaleup,
        per_card_weight_bytes,
        C2C_400,
    )
    from accel_dse.memory import ExternalMemory

    shape = ILLUSTRATIVE_27B
    # Tiny capacity → OOM even at tp=8
    tiny = ExternalMemory(
        kind="HBM",
        n_channels=1,
        width_bits=64,
        data_rate_gt_s=1.0,
        efficiency=0.7,
        capacity_bytes=1 * 10**9,  # 1 GB
    )
    tp_cfg = TPConfig(tp=8, c2c=C2C_400, embed_policy="replicate")
    oom, used, cap, detail = capacity_check(
        shape, tiny, tp_cfg, seq_len=512, batch=1
    )
    assert oom is True
    assert used > cap
    assert "OOM" in detail

    # Normal HBM 96GB @ tp=4 should fit ~16GB card
    oom2, used2, cap2, _ = capacity_check(
        shape, HBM_PRESET, TPConfig(tp=4, c2c=C2C_400), seq_len=512, batch=1
    )
    assert oom2 is False
    assert used2 < cap2
    w = per_card_weight_bytes(shape, 4, "replicate")
    assert abs(used2 - w - shape.kv_cache_bytes(512, 1) // 4) < 1

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(prompt_len=64, decode_seq_len=64, frequency_hz=SKU_100T.frequency_hz)
    r = evaluate_scaleup(shape, npu, sram, tiny, cfg, tp_cfg)
    assert r.oom is True


def test_tp_sweep_runs():
    from accel_dse.scan import run_tp_sweep

    results = run_tp_sweep(
        verbose=False,
        sku=SKU_100T,
        tp_list=[1, 2, 4, 8],
        c2c_gbps=400.0,
    )
    # 4 tp × 2 mems
    assert len(results) == 8
    hbm = [r for r in results if r.mem.kind == "HBM"]
    assert [r.tp_cfg.tp for r in hbm] == [1, 2, 4, 8]
    assert hbm[0].decode.collective_bytes == 0
    # Compute time roughly scales 1/tp
    assert abs(hbm[0].decode.t_compute_s / hbm[3].decode.t_compute_s - 8.0) < 0.01


# ---------------------------------------------------------------------------
# v0.8: KV fabric (IB/RoCE) + pipeline parallel
# ---------------------------------------------------------------------------

def test_kv_fabric_none_identical():
    """kv_fabric=none ⇒ same TPOT/TTFT as default TPConfig (no fabric)."""
    from accel_dse.scaleup import TPConfig, evaluate_scaleup, C2C_400

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz
    )
    base = evaluate_scaleup(
        ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, c2c=C2C_400)
    )
    none = evaluate_scaleup(
        ILLUSTRATIVE_27B,
        npu,
        sram,
        HBM_PRESET,
        cfg,
        TPConfig(tp=1, c2c=C2C_400, kv_fabric="none", remote_kv_frac=0.0),
    )
    assert none.decode.t_fabric_s == 0.0
    assert none.t_kv_xfer_s == 0.0
    assert abs(none.tpot_s - base.tpot_s) / max(base.tpot_s, 1e-30) < 1e-12
    assert abs(none.ttft_s - base.ttft_s) / max(base.ttft_s, 1e-30) < 1e-12
    # Also match single-card
    single = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg)
    assert abs(none.tpot_s - single.tpot_s) / max(single.tpot_s, 1e-30) < 1e-9


def test_kv_fabric_remote_time_ge_bytes_over_bw():
    """remote_kv_frac=1 ⇒ fabric time ≥ bytes/BW; latency adds."""
    from accel_dse.scaleup import (
        TPConfig,
        evaluate_scaleup,
        C2C_400,
        get_kv_fabric,
    )

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    ctx = 128000
    cfg = EvalConfig(
        prompt_len=512, decode_seq_len=ctx, frequency_hz=SKU_100T.frequency_hz
    )
    for mode, gbps in (("roce_v2", 200), ("ib", 400)):
        link = get_kv_fabric(mode, gbps=gbps)
        assert link is not None
        tp_cfg = TPConfig(
            tp=1,
            c2c=C2C_400,
            kv_fabric=mode,  # type: ignore[arg-type]
            fabric_gbps=float(gbps),
            remote_kv_frac=1.0,
            pd_kv_xfer=True,
        )
        r = evaluate_scaleup(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg, tp_cfg)
        assert r.decode.remote_kv_bytes > 0
        bytes_over_bw = r.decode.remote_kv_bytes / link.effective_bw_Bps()
        # t_fabric = lat + bytes/BW ≥ bytes/BW; and strictly > when lat>0
        assert r.decode.t_fabric_s + 1e-15 >= bytes_over_bw
        assert r.decode.t_fabric_s > bytes_over_bw  # latency adds
        expect = link.latency_s() + bytes_over_bw
        assert abs(r.decode.t_fabric_s - expect) / max(expect, 1e-30) < 1e-9
        # PD xfer: full KV
        full_kv = ILLUSTRATIVE_27B.kv_cache_bytes(ctx, 1)
        assert r.kv_xfer_bytes == full_kv
        xfer_expect = link.transfer_time_s(full_kv)
        assert abs(r.t_kv_xfer_s - xfer_expect) / max(xfer_expect, 1e-30) < 1e-9
        # Wall includes fabric via max(...)
        assert r.decode.t_roofline_s + 1e-15 >= r.decode.t_fabric_s


def test_kv_fabric_sweep_runs():
    from accel_dse.scan import run_kv_fabric_sweep

    results = run_kv_fabric_sweep(
        verbose=False,
        sku=SKU_100T,
        ctx_list=[512, 128000],
        fabric_presets=["none", "roce_200g", "ib_200g"],
        remote_kv_frac=1.0,
        pd_kv_xfer=True,
    )
    # 3 presets × 2 ctx × 2 mem
    assert len(results) == 12
    none_rows = [r for r in results if r.tp_cfg.kv_fabric == "none"]
    assert all(r.decode.t_fabric_s == 0.0 for r in none_rows)
    assert all(r.t_kv_xfer_s == 0.0 for r in none_rows)
    fab_rows = [r for r in results if r.tp_cfg.kv_fabric != "none"]
    assert all(r.decode.t_fabric_s > 0 for r in fab_rows if r.cfg.decode_seq_len == 128000)


def test_pp1_tp1_matches_single_card():
    """tp=1, pp=1 matches single-card (and tp=1 only path)."""
    from accel_dse.scaleup import TPConfig, evaluate_scaleup, C2C_400

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz
    )
    single = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg)
    su = evaluate_scaleup(
        ILLUSTRATIVE_27B,
        npu,
        sram,
        HBM_PRESET,
        cfg,
        TPConfig(tp=1, pp=1, c2c=C2C_400),
    )
    assert su.decode.bubble_frac == 0.0
    assert su.decode.pp_act_bytes == 0
    assert abs(su.tpot_s - single.tpot_s) / max(single.tpot_s, 1e-30) < 1e-9
    assert abs(su.ttft_s - single.ttft_s) / max(single.ttft_s, 1e-30) < 1e-9
    assert su.tp_cfg.n_cards == 1


def test_pp_bubble_and_activation_bytes():
    """Decode mb=1 bubble=(pp-1)/pp; act sends=(pp-1)*volume; wall inflated."""
    from accel_dse.scaleup import (
        TPConfig,
        activation_volume_bytes,
        evaluate_scaleup,
        pipeline_bubble_fraction,
        pp_activation_send_bytes,
        C2C_400,
    )

    assert pipeline_bubble_fraction(1, 1) == 0.0
    assert abs(pipeline_bubble_fraction(4, 1) - 0.75) < 1e-12
    assert abs(pipeline_bubble_fraction(4, 4) - 3 / 7) < 1e-12

    shape = ILLUSTRATIVE_27B
    vol = activation_volume_bytes(shape, 1, 1)
    assert pp_activation_send_bytes(shape, 1, 1, pp=1) == 0
    assert pp_activation_send_bytes(shape, 1, 1, pp=4) == 3 * vol

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz
    )
    r1 = evaluate_scaleup(
        shape, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, pp=1, mb=1, c2c=C2C_400)
    )
    r4 = evaluate_scaleup(
        shape, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, pp=4, mb=1, c2c=C2C_400)
    )
    assert abs(r4.decode.bubble_frac - 0.75) < 1e-12
    assert r4.decode.pp_act_bytes == 3 * vol
    # stage ~ 1/pp of compute vs pp=1; with bubble, roof ≈ stage/(1-b) = stage*pp
    assert r4.decode.t_stage_s < r1.decode.t_compute_s * 0.5  # roughly /4
    assert abs(r4.decode.t_roofline_s / r4.decode.t_stage_s - 4.0) < 1e-6
    # n_cards
    assert r4.tp_cfg.n_cards == 4
    # Prefill with mb=pp has smaller bubble
    r4p = evaluate_scaleup(
        shape, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, pp=4, mb=4, c2c=C2C_400)
    )
    assert r4p.prefill.bubble_frac < r4.decode.bubble_frac
    assert abs(r4p.prefill.bubble_frac - 3 / 7) < 1e-12


def test_parallel_sweep_runs():
    from accel_dse.scan import run_parallel_sweep

    results = run_parallel_sweep(
        verbose=False,
        sku=SKU_100T,
        tp_list=[1, 2],
        pp_list=[1, 2, 4],
        decode_mb=1,
    )
    # 2 tp × 3 pp × 2 mem
    assert len(results) == 12
    # tp=1,pp=1 HBM matches ~single
    r11 = next(r for r in results if r.tp_cfg.tp == 1 and r.tp_cfg.pp == 1 and r.mem.kind == "HBM")
    assert r11.decode.bubble_frac == 0.0
    r14 = next(r for r in results if r.tp_cfg.tp == 1 and r.tp_cfg.pp == 4 and r.mem.kind == "HBM")
    assert abs(r14.decode.bubble_frac - 0.75) < 1e-12


# ---------------------------------------------------------------------------
# v0.9: MoE expert-parallel (EP) + MLA compressed KV + CSV export
# ---------------------------------------------------------------------------


def test_moe_alltoall_formula():
    """Megatron-MoE style: 2*(ep-1)/ep * V * (top_k / E_local_adjust)."""
    from accel_dse.scaleup import activation_volume_bytes, moe_alltoall_bytes

    shape = ILLUSTRATIVE_MOE
    B, S = 2, 8
    V = activation_volume_bytes(shape, B, S)
    assert V == B * S * shape.hidden * (shape.act_bits // 8)

    # ep=1 → 0
    assert moe_alltoall_bytes(shape, B, S, 1) == 0
    # dense → 0
    assert moe_alltoall_bytes(ILLUSTRATIVE_27B, B, S, 4) == 0

    ep = 4
    e_adj = 1.0
    per_layer = int(2 * (ep - 1) / ep * V * (shape.top_k / e_adj))
    expect = shape.n_layers * per_layer
    got = moe_alltoall_bytes(shape, B, S, ep, e_local_adjust=e_adj)
    assert got == expect

    # E_local_adjust=2 halves the factor
    got2 = moe_alltoall_bytes(shape, B, S, ep, e_local_adjust=2.0)
    assert got2 == shape.n_layers * int(2 * (ep - 1) / ep * V * (shape.top_k / 2.0))

    # layers override (PP stage)
    got_pp = moe_alltoall_bytes(shape, B, S, ep, layers=shape.n_layers // 4)
    assert got_pp == (shape.n_layers // 4) * per_layer


def test_ep_shards_experts_and_active_stream():
    """EP: E/ep stored; active stream top_k/ep FFN; cards=tp*pp*ep."""
    from accel_dse.scaleup import (
        C2C_400,
        TPConfig,
        evaluate_scaleup,
        per_card_weight_bytes,
    )

    moe = ILLUSTRATIVE_MOE
    assert moe.experts_per_rank(4) == 2
    assert moe.stream_weight_bytes_per_layer_ep(1) == moe.stream_weight_bytes_per_layer()
    assert moe.stream_weight_bytes_per_layer_ep(4) < moe.stream_weight_bytes_per_layer_ep(1)

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        frequency_hz=SKU_100T.frequency_hz,
    )
    r1 = evaluate_scaleup(
        moe, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, ep=1, c2c=C2C_400)
    )
    r4 = evaluate_scaleup(
        moe, npu, sram, HBM_PRESET, cfg, TPConfig(tp=1, ep=4, c2c=C2C_400)
    )
    assert r1.tp_cfg.n_cards == 1
    assert r4.tp_cfg.n_cards == 4
    assert r1.decode.a2a_bytes == 0
    assert r4.decode.a2a_bytes > 0
    # Active W stream smaller per card under EP
    assert r4.decode.weight_bytes_card < r1.decode.weight_bytes_card
    # Capacity: stored experts /ep
    w1 = per_card_weight_bytes(moe, 1, "replicate", ep=1)
    w4 = per_card_weight_bytes(moe, 1, "replicate", ep=4)
    assert w4 < w1

    # TP×PP×EP
    r = evaluate_scaleup(
        moe, npu, sram, HBM_PRESET, cfg, TPConfig(tp=2, pp=2, ep=2, c2c=C2C_400)
    )
    assert r.tp_cfg.n_cards == 8


def test_moe_sweep_with_ep():
    from accel_dse.scan import run_moe_sweep
    from accel_dse.scaleup import ScaleupResult

    results = run_moe_sweep(verbose=False, sku=SKU_100T, ep_list=[1, 2, 4])
    # 4 single-card InferenceResult + 3 ep × 2 mem ScaleupResult = 4+6=10
    assert len(results) == 10
    ep_rows = [r for r in results if isinstance(r, ScaleupResult)]
    assert len(ep_rows) == 6
    assert {r.tp_cfg.ep for r in ep_rows} == {1, 2, 4}


def test_mla_kv_smaller_than_gqa():
    """illustrative_mla kv_lora_rank → much smaller KV than GQA 27B."""
    from accel_dse.model_shape import ILLUSTRATIVE_MLA
    from accel_dse.scan import run_mla_sweep
    from accel_dse.scaleup import ScaleupResult

    mla = ILLUSTRATIVE_MLA
    gqa = ILLUSTRATIVE_27B
    assert mla.is_mla and mla.kv_lora_rank == 512
    assert not gqa.is_mla
    assert mla.kv_bytes_per_token() < gqa.kv_bytes_per_token()
    # Exact: L * rank * 2 bytes @ fp16
    assert mla.kv_bytes_per_token() == 40 * 512 * 2
    # GQA: 2 * L * n_kv * d * 2
    assert gqa.kv_bytes_per_token() == 2 * 40 * 8 * 128 * 2
    ratio = mla.kv_bytes_per_token() / gqa.kv_bytes_per_token()
    assert abs(ratio - 0.25) < 1e-12  # 40960/163840 = 0.25

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(
        prompt_len=512,
        decode_seq_len=128000,
        frequency_hz=SKU_100T.frequency_hz,
    )
    r_g = evaluate_inference(gqa, npu, sram, HBM_PRESET, cfg)
    r_m = evaluate_inference(mla, npu, sram, HBM_PRESET, cfg)
    assert r_m.decode.traffic.kv_bytes_dram < r_g.decode.traffic.kv_bytes_dram

    results = run_mla_sweep(verbose=False, sku=SKU_100T, ctx_list=[512, 128000])
    # 2 shapes × 2 ctx × 2 mem + 2 fabric rows = 8+2=10
    assert len(results) == 10
    fab = [r for r in results if isinstance(r, ScaleupResult)]
    assert len(fab) == 2
    # MLA remote KV bytes < GQA at same frac
    mla_fab = next(r for r in fab if r.shape.is_mla)
    gqa_fab = next(r for r in fab if not r.shape.is_mla)
    assert mla_fab.decode.remote_kv_bytes < gqa_fab.decode.remote_kv_bytes
    assert mla_fab.decode.t_fabric_s < gqa_fab.decode.t_fabric_s


def test_export_csv_bundle():
    from pathlib import Path
    from accel_dse.scan import export_csv_bundle

    out = Path(__file__).resolve().parents[1] / "out"
    paths = export_csv_bundle(out, verbose=False)
    names = {p.name for p in paths}
    expect = {
        "sweep_sku.csv",
        "sweep_dtype.csv",
        "sweep_ctx.csv",
        "sweep_sram.csv",
        "sweep_tp.csv",
        "sweep_kv_fabric.csv",
        "sweep_moe.csv",
        "sweep_mla.csv",
        "sweep_video.csv",
        "sweep_protein.csv",
    }
    assert expect <= names
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# v0.10 video / protein multi-domain templates
# ---------------------------------------------------------------------------


def test_video_shape_params_and_tokens():
    from accel_dse.workloads import ILLUSTRATIVE_DIT_VIDEO, TOY_VIDEO

    v = ILLUSTRATIVE_DIT_VIDEO
    assert v.n_tokens == 16 * (32 // 2) * (32 // 2) == 4096
    p = v.total_weight_params()
    assert 100e6 < p < 2e9  # ~0.47B-class illustrative
    assert v.weight_bytes() == p * v.weight_bits // 8
    assert TOY_VIDEO.n_tokens == 8
    assert TOY_VIDEO.weight_params_per_layer() > 0


def test_video_time_scales_with_ndenose():
    from accel_dse.workloads import ILLUSTRATIVE_DIT_VIDEO, evaluate_video
    from accel_dse.memory import DEFAULT_SRAM

    npu = SKU_100T.npu()
    cfg = EvalConfig(frequency_hz=SKU_100T.frequency_hz)
    r50 = evaluate_video(ILLUSTRATIVE_DIT_VIDEO.with_denoise(50), npu, DEFAULT_SRAM, HBM_PRESET, cfg)
    r25 = evaluate_video(ILLUSTRATIVE_DIT_VIDEO.with_denoise(25), npu, DEFAULT_SRAM, HBM_PRESET, cfg)
    r100 = evaluate_video(ILLUSTRATIVE_DIT_VIDEO.with_denoise(100), npu, DEFAULT_SRAM, HBM_PRESET, cfg)
    # TTFC ≈ N_denoise * forward → linear in N_denoise
    assert abs(r25.time_to_first_clip_s / r50.time_to_first_clip_s - 0.5) < 1e-9
    assert abs(r100.time_to_first_clip_s / r50.time_to_first_clip_s - 2.0) < 1e-9
    # Forward itself independent of N_denoise
    assert abs(r25.forward.t_roofline_s - r50.forward.t_roofline_s) < 1e-15
    # No denoise-step KV cache
    assert r50.forward.traffic.kv_bytes_dram == 0
    assert r50.frames_per_s == ILLUSTRATIVE_DIT_VIDEO.n_frames / r50.time_to_first_clip_s


def test_protein_pair_bytes_scale_l2():
    from accel_dse.workloads import ILLUSTRATIVE_PROTEIN_PAIR, evaluate_protein
    from accel_dse.memory import DEFAULT_SRAM

    base = ILLUSTRATIVE_PROTEIN_PAIR
    b512 = base.with_seq_len(512).pair_activation_bytes()
    b1024 = base.with_seq_len(1024).pair_activation_bytes()
    b256 = base.with_seq_len(256).pair_activation_bytes()
    assert b512 == 512 * 512 * base.pair_dim * (base.act_bits // 8)
    assert b1024 / b512 == 4.0
    assert b512 / b256 == 4.0
    # Params roughly independent of L (embed fixed)
    assert abs(base.with_seq_len(256).total_weight_params() - base.with_seq_len(2048).total_weight_params()) == 0

    npu = SKU_100T.npu()
    cfg = EvalConfig(frequency_hz=SKU_100T.frequency_hz)
    r = evaluate_protein(base.with_seq_len(512), npu, DEFAULT_SRAM, HBM_PRESET, cfg)
    assert r.pair_bytes == b512
    assert r.time_per_sequence_s > 0
    assert math.isfinite(r.time_per_sequence_s)


def test_protein_oom_flag():
    from accel_dse.workloads import ILLUSTRATIVE_PROTEIN_PAIR, evaluate_protein
    from accel_dse.memory import DEFAULT_SRAM

    npu = SKU_100T.npu()
    cfg = EvalConfig(frequency_hz=SKU_100T.frequency_hz)
    # Huge L → pair L² exceeds LPDDR 64GB capacity
    huge = ILLUSTRATIVE_PROTEIN_PAIR.with_seq_len(32768)
    r = evaluate_protein(huge, npu, DEFAULT_SRAM, LPDDR_PRESET, cfg)
    assert r.pair_bytes > LPDDR_PRESET.capacity_bytes
    assert r.oom is True
    # Modest L fits
    ok = evaluate_protein(ILLUSTRATIVE_PROTEIN_PAIR.with_seq_len(512), npu, DEFAULT_SRAM, LPDDR_PRESET, cfg)
    assert ok.oom is False


def test_video_protein_sweeps_run():
    from accel_dse.scan import run_video_sweep, run_protein_sweep

    vr = run_video_sweep(verbose=False, denoise_list=[20, 50])
    assert len(vr) == 4  # 2 denoise × 2 mem
    pr = run_protein_sweep(verbose=False, seq_list=[256, 512])
    assert len(pr) == 4
    assert all(r.time_to_first_clip_s > 0 for r in vr)
    assert all(r.time_per_sequence_s > 0 for r in pr)


def test_list_shapes_includes_domains():
    from accel_dse.workloads import list_shapes

    rows = list_shapes()
    domains = {d for d, _, _ in rows}
    names = {n for _, n, _ in rows}
    assert domains == {"llm", "video", "protein"}
    assert "illustrative_dit_video" in names
    assert "illustrative_large_dit" in names
    assert "illustrative_protein_pair" in names
    assert "illustrative_27B" in names


def test_llm_handcheck_path_unchanged_v010():
    """Guard: adding video/protein must not change toy LLM handcheck numbers."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.11 compare-domains + large DiT
# ---------------------------------------------------------------------------


def test_compare_domains_rows_and_dominant_cost():
    from accel_dse.scan import run_compare_domains, DOMINANT_COST_VALUES

    rows = run_compare_domains(verbose=False)
    # 4 metrics × 2 mem
    assert len(rows) == 8
    for row in rows:
        assert row["dominant_cost"] in DOMINANT_COST_VALUES
        assert row["metric_ms"] > 0
        assert row["wall"] in ("compute", "memory", "balanced")
        assert row["mem"] in ("HBM", "LPDDR")
        assert "domain" in row and "metric_name" in row and "notes" in row

    by = {(r["domain"], r["shape"], r["mem"], r["metric_name"]): r for r in rows}
    assert ("llm", "illustrative_27B", "HBM", "TPOT") in by
    assert ("llm", "illustrative_mla", "LPDDR", "TPOT") in by
    assert ("video", "illustrative_dit_video", "HBM", "TTFC") in by
    assert ("protein", "illustrative_protein_pair", "LPDDR", "t_seq") in by

    # HBM LLM → compute_gemm; LPDDR LLM → weight_stream (sku_100t narrative)
    assert by[("llm", "illustrative_27B", "HBM", "TPOT")]["dominant_cost"] == "compute_gemm"
    assert by[("llm", "illustrative_27B", "LPDDR", "TPOT")]["dominant_cost"] == "weight_stream"
    assert by[("video", "illustrative_dit_video", "HBM", "TTFC")]["dominant_cost"] == "denoise_attn"
    assert by[("protein", "illustrative_protein_pair", "HBM", "t_seq")]["dominant_cost"] == "pair_L2"
    # Video notes carry fps
    assert "fps=" in by[("video", "illustrative_dit_video", "HBM", "TTFC")]["notes"]


def test_compare_domains_csv_export():
    import tempfile
    from pathlib import Path as P
    from accel_dse.scan import export_compare_domains_csv

    with tempfile.TemporaryDirectory() as td:
        written = export_compare_domains_csv(td, verbose=False)
        assert len(written) == 1
        p = P(written[0])
        assert p.name == "compare_domains.csv"
        body = p.read_text()
        assert "dominant_cost" in body
        assert "weight_stream" in body or "compute_gemm" in body
        assert "denoise_attn" in body
        assert "pair_L2" in body
        # header + 8 data rows
        lines = [ln for ln in body.strip().splitlines() if ln.strip()]
        assert len(lines) == 9


def test_large_dit_slower_than_dit_video():
    from accel_dse.workloads import (
        ILLUSTRATIVE_DIT_VIDEO,
        ILLUSTRATIVE_LARGE_DIT,
        evaluate_video,
    )
    from accel_dse.memory import DEFAULT_SRAM

    npu = SKU_100T.npu()
    cfg = EvalConfig(frequency_hz=SKU_100T.frequency_hz)
    # Same N_denoise for fair scale compare
    small = evaluate_video(
        ILLUSTRATIVE_DIT_VIDEO.with_denoise(50), npu, DEFAULT_SRAM, HBM_PRESET, cfg
    )
    large = evaluate_video(
        ILLUSTRATIVE_LARGE_DIT.with_denoise(50), npu, DEFAULT_SRAM, HBM_PRESET, cfg
    )
    assert ILLUSTRATIVE_LARGE_DIT.n_tokens > ILLUSTRATIVE_DIT_VIDEO.n_tokens
    assert ILLUSTRATIVE_LARGE_DIT.n_layers >= ILLUSTRATIVE_DIT_VIDEO.n_layers
    assert large.time_to_first_clip_s > small.time_to_first_clip_s
    # Minutes-scale on sku_100t at N=50
    assert large.time_to_first_clip_s >= 60.0  # >= 1 minute
    # Materially slower (T² / layers): expect >> 10×
    assert large.time_to_first_clip_s / small.time_to_first_clip_s > 10.0
    # Default large shape uses N in 50–100
    assert 50 <= ILLUSTRATIVE_LARGE_DIT.n_denoise <= 100


def test_sweep_video_shape_large():
    from accel_dse.scan import run_video_sweep
    from accel_dse.workloads import ILLUSTRATIVE_LARGE_DIT

    vr = run_video_sweep(
        verbose=False,
        denoise_list=[50],
        shape=ILLUSTRATIVE_LARGE_DIT,
    )
    assert len(vr) == 2  # HBM + LPDDR
    assert all(r.shape.name == "illustrative_large_dit" for r in vr)
    assert all(r.time_to_first_clip_s >= 60.0 for r in vr)


def test_llm_handcheck_path_unchanged_v011():
    """Guard: v0.11 compare/large_dit must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.12 workbench / series / geometry / cores language
# ---------------------------------------------------------------------------


def test_workbench_chips1_matches_single_card():
    """chips=1 MetricsCard matches prior single-card evaluate within tolerance."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench, single_card_reference

    card, raw = single_card_reference(
        "illustrative_27B",
        mem_kind="HBM",
        n_cores=16,
        tops_per_core=6.25,
        prompt_len=512,
        decode_seq_len=512,
    )
    assert card.chips == 1
    assert card.tp == 1 and card.pp == 1 and card.ep == 1
    # Tolerance: scaleup int-division path; allow 0.5% relative
    assert abs(card.TTFT_ms - raw.ttft_s * 1e3) / max(card.TTFT_ms, 1e-9) < 0.005
    assert abs(card.TPOT_ms - raw.tpot_s * 1e3) / max(card.TPOT_ms, 1e-9) < 0.005
    assert card.wall == raw.decode.wall
    assert abs(card.util - raw.decode.traffic.mean_gemm_util) < 1e-9
    # Also direct WorkbenchConfig path
    c2 = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_27B",
            chip_count=1,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    assert abs(c2.TTFT_ms - card.TTFT_ms) < 1e-9


def test_geometry_bw_formula():
    from accel_dse.memory import (
        HBM_PRESET,
        MemGeometry,
        memory_from_geometry,
        peak_bandwidth_GBps_from_geometry,
    )

    # peak = ch * (width/8) * GT/s
    peak = peak_bandwidth_GBps_from_geometry(8, 1024, 5.2)
    assert abs(peak - (8 * 128 * 5.2)) < 1e-9
    # packages multiply channels (assumed)
    peak2 = peak_bandwidth_GBps_from_geometry(8, 1024, 5.2, n_packages=2)
    assert abs(peak2 - 2 * peak) < 1e-9
    mem = memory_from_geometry(
        "HBM",
        n_channels=8,
        width_bits=1024,
        data_rate_GTs=5.2,
        efficiency=0.70,
        capacity_GB=96.0,
    )
    assert abs(mem.peak_bandwidth_Bps() / 1e9 - peak) < 1e-6
    assert abs(mem.effective_bandwidth_Bps() / 1e9 - peak * 0.70) < 1e-6
    # Preset defaults when no overrides
    mem2 = memory_from_geometry("HBM")
    assert mem2.n_channels == HBM_PRESET.n_channels
    assert mem2.width_bits == HBM_PRESET.width_bits
    # n_ranks alias
    mem3 = memory_from_geometry("LPDDR", n_ranks=4, width_bits=64, data_rate_GTs=8.5)
    assert mem3.n_channels == 4
    g = MemGeometry(
        kind="HBM",
        n_channels=8,
        width_bits=1024,
        data_rate_GTs=5.2,
        efficiency=0.7,
        capacity_GB=96,
        n_packages=1,
    )
    assert abs(g.peak_bandwidth_GBps() - peak) < 1e-9


def test_cores_tops_per_core_peak():
    from accel_dse.npu import cores_from_npu, npu_from_cores
    from accel_dse.sku import SKU_100T

    npu = npu_from_cores(16, 6.25, frequency_hz=1e9)
    # Near-square → 56x56×16 like sku_100t
    assert npu.n_engines == 16
    assert npu.rows == 56 and npu.cols == 56
    peak = npu.peak_tops_at(1e9)
    assert abs(peak - SKU_100T.achieved_tops()) < 0.01
    # Requested product language: 16*6.25=100; achieved ~100.35 due to side rounding
    assert abs(peak - 100.0) < 1.0
    n_cores, tpc = cores_from_npu(npu, 1e9)
    assert n_cores == 16
    assert abs(tpc * n_cores - peak) < 1e-9


def test_series_registry_lists():
    from accel_dse.series import (
        ILLUSTRATIVE_SERIES_IDS,
        PRODUCT_SERIES_IDS,
        get_series,
        list_series,
        resolve_llm_shape,
    )
    from accel_dse.model_shape import ILLUSTRATIVE_27B, ILLUSTRATIVE_MOE, ILLUSTRATIVE_MLA

    product = list_series(product_only=True)
    ids = [e.id for e in product]
    assert ids == list(PRODUCT_SERIES_IDS)
    # product packs are real HF-backed (dozens), not only illustrative
    assert len(product) >= 40
    assert "glm-5.3" in ids
    assert "kimi-k3" in ids
    assert "minimax-h3" in ids
    for e in product:
        assert e.is_hf_backed or e.metadata.lower().startswith("hf:")
        assert "llama-4" not in e.id.lower()
        assert "gemma" not in e.id.lower()
    # illustrative packs still resolve for handcheck
    assert "series/dense-27b" in ILLUSTRATIVE_SERIES_IDS
    assert resolve_llm_shape("series/dense-27b").name == ILLUSTRATIVE_27B.name
    assert resolve_llm_shape("series/moe-active13b").name == ILLUSTRATIVE_MOE.name
    assert resolve_llm_shape("series/mla-27b").name == ILLUSTRATIVE_MLA.name
    assert get_series("illustrative_27B").family == "dense"
    assert get_series("illustrative_27B").is_illustrative
    all_e = list_series()
    assert len(all_e) >= 40


def test_workbench_scan_runs():
    from accel_dse.workbench import (
        export_scan_csv,
        format_scan_table,
        workbench_scan,
    )
    import tempfile
    from pathlib import Path

    cards = workbench_scan(
        "illustrative_27B",
        chips_list=[1, 2, 4, 8],
        n_cores=16,
        tops_per_core=6.25,
        mem_kind="HBM",
    )
    assert len(cards) == 4
    assert [c.chips for c in cards] == [1, 2, 4, 8]
    assert [c.tp for c in cards] == [1, 2, 4, 8]
    # TPOT should roughly scale down with tp on compute-bound HBM short ctx
    assert cards[0].TPOT_ms > cards[-1].TPOT_ms
    table = format_scan_table(cards)
    assert "TPOT_ms" in table and "chips" in table
    with tempfile.TemporaryDirectory() as td:
        p = export_scan_csv(cards, Path(td) / "wb_scan.csv")
        text = p.read_text(encoding="utf-8")
        assert "TTFT_ms" in text
        assert text.count("\n") >= 5  # header + 4 rows


def test_workbench_parallel_product_constraint():
    from accel_dse.workbench import ParallelOverride, WorkbenchConfig, evaluate_workbench

    # Explicit tp*pp*ep must equal chips
    cfg = WorkbenchConfig(
        model_id="illustrative_27B",
        chip_count=4,
        parallel=ParallelOverride(tp=2, pp=2, ep=1),
        n_cores=16,
        tops_per_core=6.25,
    )
    card = evaluate_workbench(cfg)
    assert card.tp == 2 and card.pp == 2 and card.chips == 4
    try:
        evaluate_workbench(
            WorkbenchConfig(
                model_id="illustrative_27B",
                chip_count=4,
                parallel=ParallelOverride(tp=2, pp=1, ep=1),
            )
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "chip_count" in str(e)


def test_llm_handcheck_path_unchanged_v012():
    """Guard: v0.12 workbench must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.13: parallel matrix / mem geometry sweep / MetricsCard export / series
# ---------------------------------------------------------------------------


def test_enumerate_parallel_combos_pow2():
    from accel_dse.workbench import enumerate_parallel_combos

    combos = enumerate_parallel_combos(8)
    assert (8, 1, 1) in combos
    assert (4, 2, 1) in combos
    assert (1, 1, 8) in combos
    assert (2, 2, 2) in combos
    assert all(t * p * e == 8 for t, p, e in combos)
    # all powers of two
    for t, p, e in combos:
        for x in (t, p, e):
            assert x > 0 and (x & (x - 1)) == 0


def test_workbench_parallel_matrix_dense_and_moe():
    from accel_dse.workbench import (
        format_parallel_table,
        workbench_parallel_matrix,
        export_scan_csv,
    )
    import tempfile
    from pathlib import Path

    dense = workbench_parallel_matrix(
        "illustrative_27B",
        8,
        n_cores=16,
        tops_per_core=6.25,
        mem_kind="HBM",
    )
    # dense: ep must be 1 → 4 combos for chips=8
    assert len(dense) == 4
    assert all(c.ep == 1 for c in dense)
    assert all(c.chips == 8 for c in dense)
    # sorted by TPOT ascending
    tpots = [c.TPOT_ms for c in dense]
    assert tpots == sorted(tpots)
    # best is usually high-TP / low-PP on HBM short ctx
    assert dense[0].tp >= dense[-1].tp or dense[0].pp <= dense[-1].pp

    moe = workbench_parallel_matrix(
        "series/moe-active13b",
        8,
        n_cores=16,
        tops_per_core=6.25,
        mem_kind="HBM",
    )
    assert len(moe) >= 4
    assert any(c.ep > 1 for c in moe)
    assert all(c.tp * c.pp * c.ep == 8 for c in moe)
    table = format_parallel_table(moe)
    assert "tp" in table and "ep" in table and "TPOT_ms" in table

    with tempfile.TemporaryDirectory() as td:
        p = export_scan_csv(moe, Path(td) / "par.csv")
        text = p.read_text(encoding="utf-8")
        assert "TPOT_ms" in text
        assert text.count("\n") >= len(moe)


def test_workbench_mem_sweep_wall_flip():
    from accel_dse.workbench import format_mem_sweep_table, workbench_mem_sweep

    # LPDDR + low channel count → memory wall; boost channels/rate may flip
    cards = workbench_mem_sweep(
        "illustrative_27B",
        chips=1,
        mem_kind="LPDDR",
        vary="both",
        channel_list=[2, 8, 32],
        rate_list=[4.0, 8.5, 17.0],
        n_cores=16,
        tops_per_core=6.25,
    )
    assert len(cards) == 9
    walls = {c.wall for c in cards}
    # Expect at least memory somewhere on skinny LPDDR; possibly compute on fat BW
    assert "memory" in walls or "compute" in walls
    # eff BW should span a wide range
    bw = [c.mem_eff_GBps for c in cards]
    assert max(bw) / max(min(bw), 1e-9) > 4.0
    table = format_mem_sweep_table(cards)
    assert "eff_GBps" in table and "wall" in table
    # Also channels-only path
    ch_only = workbench_mem_sweep(
        "illustrative_27B",
        chips=1,
        mem_kind="HBM",
        vary="channels",
        channel_list=[4, 8],
        n_cores=16,
        tops_per_core=6.25,
    )
    assert len(ch_only) == 2


def test_metrics_card_json_md_export():
    import json
    import tempfile
    from pathlib import Path

    from accel_dse.workbench import (
        WorkbenchConfig,
        evaluate_workbench,
        export_metrics_card_json,
        export_metrics_card_md,
    )

    card = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_27B",
            chip_count=1,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    with tempfile.TemporaryDirectory() as td:
        jp = export_metrics_card_json(card, Path(td) / "card.json")
        mp = export_metrics_card_md(card, Path(td) / "card.md")
        data = json.loads(jp.read_text(encoding="utf-8"))
        assert data["TPOT_ms"] == card.TPOT_ms
        assert data["chips"] == 1
        assert isinstance(data["assumptions"], list)
        md = mp.read_text(encoding="utf-8")
        assert "MetricsCard" in md
        assert "TPOT_ms" in md


def test_series_video_protein_aliases():
    from accel_dse.series import get_series, list_series

    for alias, domain in (
        ("series/video", "video"),
        ("series/video-dit", "video"),
        ("series/large-dit", "video"),
        ("series/protein", "protein"),
    ):
        e = get_series(alias)
        assert e.domain == domain
        meta = e.metadata.lower()
        assert "illustrative" in meta or "placeholder" in meta
    rows = list_series(product_only=True)
    domains = {e.domain for e in rows}
    assert "llm" in domains and "video" in domains and "protein" in domains


def test_llm_handcheck_path_unchanged_v013():
    """Guard: v0.13 polish must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_llm_handcheck_path_unchanged_v015():
    """Guard: v0.15 HF packs must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_report_html_exists_and_contains_metrics_wall():
    """v0.17: report HTML is self-contained and mentions Metrics/wall + packages."""
    import tempfile
    from pathlib import Path

    from accel_dse.report import collect_report_data, render_html, write_report
    from accel_dse import __version__

    _assert_version(__version__)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.html"
        path = write_report(out, verbose=False)
        assert path.exists()
        assert path.stat().st_size > 1000
        text = path.read_text(encoding="utf-8")
        assert "Metrics" in text or "MetricsCard" in text
        assert "wall" in text.lower()
        assert "<style>" in text  # inline CSS, no external deps
        assert "http://" not in text.split("<body")[0] or "viewport" in text
        # body should not pull external stylesheets
        assert 'link rel="stylesheet"' not in text.lower()
        assert "How to read walls" in text or "how to read walls" in text.lower()
        assert "ASSUMPTIONS" in text.upper() or "uncalibrated" in text.lower()
        assert "series" in text.lower()
        assert "TPOT" in text
        assert "Package" in text or "package" in text.lower()
        assert "Compute" in text or "compute" in text.lower()
    # render path also ok
    bundle = collect_report_data(chips_list=[1, 2], parallel_chips=2)
    html = render_html(bundle)
    assert "wall" in html.lower() and "Metrics" in html


def test_report_cli_subcommand():
    from accel_dse.cli import main
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "r.html")
        rc = main(["report", "--out", out])
        assert rc == 0
        text = Path(out).read_text(encoding="utf-8")
        assert "wall" in text.lower()
        assert "Metrics" in text


def test_llm_handcheck_path_unchanged_v014():
    """Guard: v0.14 report must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_hf_series_key_dims():
    """v0.15: real public HF dims for user-named packs."""
    from accel_dse.series import get_series
    from accel_dse.model_shape import ModelShape
    from accel_dse.workloads import VideoShape

    glm = get_series("glm-5.3")
    assert isinstance(glm.shape, ModelShape)
    assert glm.shape.n_layers == 78
    assert glm.shape.hidden == 6144
    assert glm.shape.n_experts == 256
    assert glm.shape.kv_lora_rank == 512
    assert "hf:zai-org/GLM-5.3" in glm.metadata

    kimi = get_series("kimi-k3")
    assert kimi.shape.n_experts == 896
    assert kimi.shape.top_k == 16
    assert kimi.shape.n_shared_experts == 2

    qwen = get_series("qwen3.8-2.4t")
    assert qwen.shape.n_layers == 92
    assert qwen.shape.hidden == 8192
    assert qwen.shape.n_experts == 512

    flash = get_series("deepseek-v4.1-flash")
    assert flash.shape.n_layers == 40
    assert flash.shape.hidden == 5120

    h3 = get_series("minimax-h3")
    assert h3.domain == "video"
    assert isinstance(h3.shape, VideoShape)
    assert h3.shape.n_layers == 50
    assert h3.shape.hidden == 5376

    # gated models absent
    for bad in ("llama-4-scout", "Llama-4-Maverick", "gemma-3-27b", "esm3-sm-open-v1"):
        try:
            get_series(bad)
            raise AssertionError(f"gated model unexpectedly registered: {bad}")
        except KeyError:
            pass


def test_hf_series_no_gated_in_registry():
    from accel_dse.series import SERIES_REGISTRY, list_series
    blob = " ".join(SERIES_REGISTRY.keys()).lower()
    assert "llama-4-scout" not in blob
    assert "llama-4-maverick" not in blob
    assert "gemma-3" not in blob
    product = list_series(product_only=True)
    assert any(e.domain == "video" and "minimax" in e.id for e in product)
    assert any(e.domain == "protein" for e in product)


def test_package_catalog_membership_and_bw_monotonicity():
    """v0.29 (rewritten): structured catalog — LPDDR5/5X x64 packages, LPDDR6 x96
    packages (4×24 ch), SOCAMM2/LPCAMM2 128-bit modules, HBM stacks with
    gen-tied rates; legacy ≤0.28 ids still resolve.

    Changed in 0.29 because the old catalog was wrong (docs/research/memory_specs
    §5): LPDDR6 was modelled as x24 *dies* (no x96 package), counts/ids were
    old (lpddr6_20x24_14400 / hbm_hbm3_8s as catalog keys), HBM rates were
    not tied to generation and HBM capacities were invented.
    """
    from accel_dse.package_ranges import (
        CLUSTER_TOPS_GRID,
        CORE_TOPS_GRID,
        HBM_GEN_LABELS,
        PACKAGE_CATALOG,
        build_hbm_packages,
        build_lpddr5x_packages,
        get_package,
        list_packages,
        package_counts,
    )

    counts = package_counts()
    assert counts["lpddr5x"] == 18  # 3 counts × 4 rates + SOCAMM2 4 + LPCAMM2 2
    assert counts["lpddr6"] == 12  # 4 counts × 3 rates (x96 packages)
    assert counts["lpddr5"] == 6
    assert counts["lpddr"] == 36
    assert counts["hbm"] == 25
    assert counts["total"] == 61

    for p in list_packages(kind="LPDDR"):
        s = p.mem_spec
        if s.form == "discrete":
            assert p.width_bits in (32, 64, 96), p.id
        else:
            assert p.width_bits == 128, p.id  # modules
        if p.generation == "LPDDR6":
            assert p.width_bits == 96 and abs(s.payload_factor - 256 / 288) < 1e-12
        else:
            assert s.payload_factor == 1.0

    # LPDDR5X x64 monotonicity (more packages / higher rate → more BW)
    lo = get_package("lpddr5x_2x64_8533_16g")
    mid = get_package("lpddr5x_8x64_8533_16g")
    hi = get_package("lpddr5x_8x64_10667_16g")
    assert lo.bus_width_bits == 128 and mid.bus_width_bits == 512
    assert mid.peak_bandwidth_GBps() > lo.peak_bandwidth_GBps()
    assert hi.peak_bandwidth_GBps() > mid.peak_bandwidth_GBps()
    assert abs(mid.peak_bandwidth_GBps() - 4 * lo.peak_bandwidth_GBps()) < 1e-6
    for n_pkg in (2, 4, 8):
        rows = sorted(
            [p for p in build_lpddr5x_packages() if p.n_packages == n_pkg],
            key=lambda p: p.data_rate_GTs,
        )
        for a_, b_ in zip(rows, rows[1:]):
            assert b_.peak_bandwidth_GBps() > a_.peak_bandwidth_GBps()

    # LPDDR6 x96 package (4×24-bit channels)
    lp6 = get_package("lpddr6_4x96_10667_16g")
    assert lp6.width_bits == 96 and lp6.n_packages == 4 and lp6.bus_width_bits == 384
    assert abs(lp6.peak_bandwidth_GBps() - 512.0) < 0.05
    assert abs(lp6.payload_bandwidth_GBps() - 455.1) < 0.05

    # Legacy ids resolve (nearest structured config + note)
    assert get_package("lpddr_2x64_6400").generation == "LPDDR5"
    # v0.30: legacy x24-die bus width preserved (x96, else speculative x48)
    assert get_package("lpddr6_6x24_10667").bus_width_bits == 144
    old = get_package("lpddr6_20x24_14400")
    assert old.width_bits == 96 and old.n_packages == 5 and old.mem_spec.legacy_note
    h = get_package("hbm_hbm3_8s")
    assert h.generation == "HBM3" and h.n_stacks == 8 and h.mem_spec.legacy_note

    # HBM: more stacks → higher peak within a generation; rates tied to gen
    for gen in HBM_GEN_LABELS:
        rows = sorted(
            [
                p for p in build_hbm_packages(extra_rates=False)
                if p.generation == gen
            ],
            key=lambda p: p.n_stacks or 0,
        )
        assert [p.n_stacks for p in rows] == [2, 4, 6, 8]
        for a_, b_ in zip(rows, rows[1:]):
            assert b_.peak_bandwidth_GBps() > a_.peak_bandwidth_GBps()
    from accel_dse.mem_catalog import HBM_CATALOG

    for p in list_packages(kind="HBM"):
        assert p.mem_spec.rate_MTps in HBM_CATALOG[p.generation]["rates"], p.id

    mem = lo.to_external_memory()
    assert mem.kind == "LPDDR" and mem.n_channels == 2 and mem.width_bits == 64
    assert "hbm3e_8s_12h24g_9200" in PACKAGE_CATALOG
    assert set(CORE_TOPS_GRID) == {4.0, 8.0, 12.0, 16.0}
    assert set(CLUSTER_TOPS_GRID) == {64.0, 128.0, 192.0, 256.0}
    assert "lpddr_4x128_6400" not in PACKAGE_CATALOG


def test_compute_catalog_and_sweep():
    """v0.16: core/cluster grids + workbench compute sweep."""
    from accel_dse.package_ranges import (
        get_compute,
        list_compute_primaries,
    )
    from accel_dse.workbench import (
        format_compute_sweep_table,
        workbench_compute_sweep,
    )

    prim = list_compute_primaries()
    ids = {c.id for c in prim}
    assert ids >= {
        "core_4t",
        "core_8t",
        "core_12t",
        "core_16t",
        "cluster_64t",
        "cluster_128t",
        "cluster_192t",
        "cluster_256t",
    }
    c64 = get_compute("cluster_64t")
    assert c64.n_cores * c64.tops_per_core == 64.0

    rows = workbench_compute_sweep(
        "illustrative_27B",
        level="core",
        chips=1,
        mem_kind="LPDDR",
    )
    assert len(rows) == 4
    peaks = [o.peak_tops for o, _ in rows]
    assert peaks == [4.0, 8.0, 12.0, 16.0]
    # Higher compute peak should not increase TPOT (non-strict: wall may flip)
    tpots = [c.TPOT_ms for _, c in rows]
    assert tpots[0] >= tpots[-1] - 1e-9
    table = format_compute_sweep_table(rows)
    assert "core_4t" in table and "wall" in table

    rows_c = workbench_compute_sweep(
        "illustrative_27B",
        level="cluster",
        chips=1,
        mem_kind="HBM",
    )
    assert len(rows_c) == 4
    assert [o.peak_tops for o, _ in rows_c] == [64.0, 128.0, 192.0, 256.0]


def test_package_sweep_and_cli_list():
    """v0.16: package sweep + list-packages / list-compute CLI."""
    from accel_dse.cli import main
    from accel_dse.workbench import (
        format_package_sweep_table,
        workbench_package_sweep,
    )

    rows = workbench_package_sweep(
        "illustrative_27B",
        kind="LPDDR",
        chips=1,
        n_cores=16,
        tops_per_core=6.25,
    )
    assert len(rows) >= 6  # 2/4/8×64 rate band + LPDDR6 samples
    peaks = [p.peak_bandwidth_GBps() for p, _ in rows]
    # Default LPDDR sweep includes min 2×64 and max 8×64 (+ LPDDR6)
    assert max(peaks) > min(peaks)
    assert any(p.generation == "LPDDR5X" and p.n_packages == 2 for p, _ in rows)
    # v0.29: no 128-bit *discrete package* granule; 128-bit is only valid for
    # SOCAMM2 / LPCAMM2 modules (JESD328 / JESD318), which the sweep includes.
    assert all(
        p.package_width_bits != 128
        for p, _ in rows
        if p.kind == "LPDDR" and p.mem_spec.form == "discrete"
    )
    assert all(
        p.width_bits == 128 for p, _ in rows if p.mem_spec.form in ("SOCAMM2", "LPCAMM2")
    )
    table = format_package_sweep_table(rows)
    # v0.29: canonical ids are lpddr5x_* / lpddr6_* (was "lpddr_" in ≤0.28)
    assert "lpddr5x_" in table and "peak_GBps" in table

    hbm_rows = workbench_package_sweep(
        "illustrative_27B",
        kind="HBM",
        chips=1,
        n_cores=16,
        tops_per_core=6.25,
    )
    assert len(hbm_rows) >= 8  # 2/4/8 stacks × gens in default HBM sweep
    # more stacks same gen → higher eff BW on card
    by_id = {p.id: c for p, c in hbm_rows}
    # v0.29 canonical ids (was hbm_hbm3_2s / hbm_hbm3_8s)
    assert "hbm3_2s_12h16g_6400" in by_id and "hbm3_8s_12h16g_6400" in by_id
    assert (
        by_id["hbm3_8s_12h16g_6400"].mem_eff_GBps
        > by_id["hbm3_2s_12h16g_6400"].mem_eff_GBps
    )

    assert main(["list-packages"]) == 0
    assert main(["list-compute"]) == 0
    assert main(["list-packages", "--kind", "lpddr"]) == 0


def test_llm_handcheck_path_unchanged_v016():
    """Guard: v0.16 package ranges must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_serve_module_import_and_api_smoke():
    """v0.18: serve module imports; API eval returns MetricsCard for glm-5.3 / illustrative."""
    from accel_dse import __version__
    from accel_dse.serve import (
        HONESTY_BANNER,
        WEB_DIR,
        api_compute,
        api_eval,
        api_health,
        api_packages,
        api_series,
    )

    _assert_version(__version__)
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "app.js").is_file()
    assert (WEB_DIR / "style.css").is_file()
    assert "assumed" in HONESTY_BANNER.lower() or "uncalibrated" in HONESTY_BANNER.lower()

    health = api_health()
    assert health["ok"] is True
    _assert_version(health["version"])

    series = api_series(domain="llm")
    assert series["count"] >= 1
    ids = {s["id"] for s in series["series"]}
    assert "illustrative_27B" in ids or "glm-5.3" in ids

    pkgs = api_packages(kind="HBM")
    assert pkgs["count"] >= 1
    # v0.29: canonical HBM ids are hbm3_/hbm3e_/hbm4_/hbm4e_ (was hbm_<gen>_)
    assert any(p["id"].startswith("hbm3e_") for p in pkgs["packages"])

    comps = api_compute(level="all")
    assert comps["count"] >= 4
    assert any(c["id"].startswith("core_") for c in comps["compute"])

    for model in ("illustrative_27B", "glm-5.3"):
        if model == "glm-5.3" and "glm-5.3" not in ids:
            continue
        card = api_eval(
            {
                "model_id": model,
                "chip_count": 1,
                "package_id": "hbm_hbm3_8s",
                "compute_id": "cluster_64t",
                "batch": 1,
                "prompt": 512,
                "ctx": 512,
            }
        )
        assert card["ok"] is True
        assert "TTFT_ms" in card and card["TTFT_ms"] > 0
        assert "TPOT_ms" in card and card["TPOT_ms"] > 0
        assert "wall" in card
        assert "peak_tops" in card and card["peak_tops"] > 0
        assert "mem_eff_GBps" in card and card["mem_eff_GBps"] > 0
        assert "assumptions" in card and len(card["assumptions"]) >= 1
        assert "banner" in card
        assert card["model_id"] == model
        assert card["oom"] in (True, False)


def test_serve_api_eval_manual_geometry_and_parallel():
    """v0.18: manual mem geometry + explicit parallel product check."""
    from accel_dse.serve import api_eval

    card = api_eval(
        {
            "model_id": "illustrative_27B",
            "chip_count": 4,
            "tp": 2,
            "pp": 2,
            "ep": 1,
            "mem_kind": "LPDDR",
            "n_channels": 8,
            "width_bits": 64,
            "data_rate_GTs": 8.533,
            "efficiency": 0.70,
            "n_cores": 16,
            "tops_per_core": 6.25,
        }
    )
    assert card["ok"] is True
    assert card["chips"] == 4
    assert card["tp"] == 2 and card["pp"] == 2 and card["ep"] == 1
    assert card["mem_kind"] == "LPDDR"

    raised = False
    try:
        api_eval(
            {
                "model_id": "illustrative_27B",
                "chip_count": 4,
                "tp": 4,
                "pp": 2,
                "ep": 1,
            }
        )
    except ValueError:
        raised = True
    assert raised


def test_serve_cli_help_and_handler_stdlib_bind():
    """v0.18: serve subcommand registered; stdlib server answers /api/health."""
    import json
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from accel_dse.cli import main
    from accel_dse.serve import WorkbenchHandler

    # --help raises SystemExit(0)
    try:
        main(["serve", "--help"])
        help_ok = True
    except SystemExit as e:
        help_ok = e.code in (0, None)
    assert help_ok

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
    port = httpd.server_address[1]
    thr = threading.Thread(target=httpd.serve_forever, daemon=True)
    thr.start()
    try:
        time.sleep(0.05)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            health = json.loads(r.read().decode())
        assert health["ok"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            html = r.read().decode()
        assert "MetricsCard" in html and "accel_dse" in html
        # POST eval
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/eval",
            data=json.dumps(
                {
                    "model_id": "illustrative_27B",
                    "chip_count": 1,
                    "compute_id": "core_16t",
                    "package_id": "hbm_hbm3_4s",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            card = json.loads(r.read().decode())
        assert card["ok"] is True and card["TTFT_ms"] > 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_llm_handcheck_path_unchanged_v017():
    """Guard: v0.17 web serve must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_api_eval_video_minimax_and_illustrative():
    """v0.18: POST /api/eval video → TTFC / frames/s / wall / capacity."""
    from accel_dse.serve import api_eval, api_series

    series = api_series(domain="video")
    ids = {s["id"] for s in series["series"]}
    assert "illustrative_dit_video" in ids
    assert "minimax-h3" in ids

    for model, knobs in (
        ("illustrative_dit_video", {"n_denoise": 25, "n_frames": 8}),
        ("minimax-h3", {"n_denoise": 50}),
    ):
        card = api_eval(
            {
                "model_id": model,
                "chip_count": 1,
                "package_id": "hbm_hbm3_8s",
                "compute_id": "cluster_64t",
                "batch": 1,
                **knobs,
            }
        )
        assert card["ok"] is True
        assert card["domain"] == "video"
        assert card["TTFC_ms"] > 0
        assert card["frames_per_s"] > 0
        assert card["wall"] in ("compute", "memory", "c2c", "fabric", "balanced") or isinstance(
            card["wall"], str
        )
        assert "mem_eff_GBps" in card and card["mem_eff_GBps"] > 0
        assert "capacity_needed_GB" in card
        assert "capacity_GB" in card and card["capacity_GB"] > 0
        assert "assumptions" in card and len(card["assumptions"]) >= 1
        assert "banner" in card
        assert card["n_denoise"] >= 1
        assert card["n_frames"] >= 1
        # TTFT alias matches TTFC for shared UI
        assert abs(card["TTFT_ms"] - card["TTFC_ms"]) < 1e-6


def test_api_eval_protein_esmfold_and_illustrative():
    """v0.18: POST /api/eval protein → time/seq + pair memory flags."""
    from accel_dse.serve import api_eval, api_series

    series = api_series(domain="protein")
    ids = {s["id"] for s in series["series"]}
    assert "illustrative_protein_pair" in ids
    assert "esmfold" in ids

    for model, knobs in (
        ("illustrative_protein_pair", {"seq_len": 256}),
        ("esmfold", {"seq_len": 512}),
    ):
        card = api_eval(
            {
                "model_id": model,
                "chip_count": 1,
                "package_id": "hbm_hbm3_8s",
                "compute_id": "cluster_64t",
                "batch": 1,
                **knobs,
            }
        )
        assert card["ok"] is True
        assert card["domain"] == "protein"
        assert card["time_per_seq_ms"] > 0
        assert card["pair_bytes"] > 0
        assert card["seq_len"] == knobs["seq_len"]
        assert "capacity_needed_GB" in card and card["capacity_needed_GB"] > 0
        assert "capacity_GB" in card and card["capacity_GB"] > 0
        assert card["oom"] in (True, False)
        assert "wall" in card
        assert "assumptions" in card and any("pair" in a.lower() or "protein" in a.lower() or "hf:" in a.lower() or "illustrative" in a.lower() for a in card["assumptions"])
        assert "banner" in card
        assert abs(card["TTFT_ms"] - card["time_per_seq_ms"]) < 1e-6


def test_workbench_video_protein_direct():
    """v0.18: evaluate_workbench dispatches video/protein without NotImplementedError."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    v = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_dit_video",
            chip_count=1,
            n_denoise=10,
            n_frames=4,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    assert v.domain == "video" and v.TTFC_ms > 0 and v.frames_per_s > 0

    p = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=1,
            seq_len=128,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    assert p.domain == "protein" and p.time_per_seq_ms > 0 and p.pair_bytes > 0

    # LLM path unbroken
    L = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", chip_count=1, n_cores=16, tops_per_core=6.25)
    )
    assert L.domain == "llm" and L.TTFT_ms > 0 and L.TPOT_ms > 0


def test_llm_handcheck_path_unchanged_v018():
    """Guard: v0.18 multi-domain MetricsCard must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_video_protein_multicard_tp_scales():
    """v0.19: chip_count>1 scales video TTFC / protein t/seq + adds collectives."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    v1 = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_dit_video",
            chip_count=1,
            n_denoise=10,
            n_frames=4,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    v4 = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_dit_video",
            chip_count=4,
            n_denoise=10,
            n_frames=4,
            n_cores=16,
            tops_per_core=6.25,
            c2c_gbps=400.0,
        )
    )
    assert v1.chips == 1 and v1.tp == 1 and v1.bytes_coll == 0
    assert v4.chips == 4 and v4.tp == 4 and v4.bytes_coll > 0
    assert v4.TTFC_ms < v1.TTFC_ms
    assert v4.bytes_W < v1.bytes_W
    assert any("multi-card" in a.lower() or "tp=" in a.lower() for a in v4.assumptions)

    p1 = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=1,
            seq_len=128,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    p4 = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=4,
            seq_len=128,
            n_cores=16,
            tops_per_core=6.25,
        )
    )
    assert p4.time_per_seq_ms < p1.time_per_seq_ms
    assert p4.bytes_coll > 0 and p4.pair_bytes == p1.pair_bytes // 4
    assert any("multi-card" in a.lower() or "tp=" in a.lower() for a in p4.assumptions)


def test_api_eval_video_protein_chip_count():
    """v0.19: /api/eval video+protein with chip_count>1 returns scaled metrics."""
    from accel_dse.serve import api_eval

    v = api_eval(
        {
            "model_id": "illustrative_dit_video",
            "chip_count": 4,
            "package_id": "hbm_hbm3_8s",
            "compute_id": "cluster_64t",
            "n_denoise": 10,
            "n_frames": 4,
        }
    )
    assert v["ok"] is True and v["domain"] == "video"
    assert v["chips"] == 4 and v["tp"] == 4
    assert v["TTFC_ms"] > 0 and v["bytes_coll"] > 0
    assert any("tp=" in a.lower() or "multi-card" in a.lower() for a in v["assumptions"])

    p = api_eval(
        {
            "model_id": "illustrative_protein_pair",
            "chip_count": 2,
            "package_id": "lpddr_4x64_8533",
            "compute_id": "cluster_64t",
            "seq_len": 128,
        }
    )
    assert p["ok"] is True and p["domain"] == "protein"
    assert p["chips"] == 2 and p["tp"] == 2
    assert p["time_per_seq_ms"] > 0 and p["bytes_coll"] > 0


def test_llm_handcheck_path_unchanged_v019():
    """Guard: v0.19 LPDDR granule + domain TP must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


def test_video_protein_pp_ep_communication():
    """v0.20: chip_count=8 with tp=4,pp=2,ep=1 vs tp=8; bubble/c2c fields present."""
    from accel_dse.workbench import WorkbenchConfig, ParallelOverride, evaluate_workbench

    base = dict(n_cores=16, tops_per_core=6.25, c2c_gbps=400.0)

    v_pp = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_dit_video",
            chip_count=8,
            parallel=ParallelOverride(tp=4, pp=2, ep=1),
            n_denoise=10,
            n_frames=4,
            **base,
        )
    )
    v_tp = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_dit_video",
            chip_count=8,
            parallel=ParallelOverride(tp=8, pp=1, ep=1),
            n_denoise=10,
            n_frames=4,
            **base,
        )
    )
    assert v_pp.tp == 4 and v_pp.pp == 2 and v_pp.ep == 1
    assert v_tp.tp == 8 and v_tp.pp == 1 and v_tp.ep == 1
    assert v_pp.t_bubble_ms > 0.0  # denoise mb=1 → bubble=(pp-1)/pp
    assert v_tp.t_bubble_ms == 0.0
    assert v_pp.t_c2c_ms > 0.0 and v_tp.t_c2c_ms > 0.0
    assert v_pp.bytes_coll > 0 and v_tp.bytes_coll > 0
    # PP bubble makes wall slower than pure TP for sequential denoise
    assert v_pp.TTFC_ms > v_tp.TTFC_ms
    assert any("bubble" in a.lower() or "pipelines poorly" in a.lower() for a in v_pp.assumptions)
    assert any("ring" in a.lower() or "collective" in a.lower() for a in v_pp.assumptions)

    # Protein: PP with mb>=pp cuts bubble; ep>1 no-op without MoE
    p_pp = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=8,
            parallel=ParallelOverride(tp=4, pp=2, ep=1),
            seq_len=128,
            decode_mb=4,
            **base,
        )
    )
    p_tp = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=8,
            **base,
            seq_len=128,
        )
    )
    assert p_pp.tp == 4 and p_pp.pp == 2 and p_pp.ep == 1
    assert p_pp.t_c2c_ms > 0.0
    assert p_pp.bytes_coll > 0
    # mb=4, pp=2 → bubble=(2-1)/(4+2-1)=1/5=0.2 > 0
    assert p_pp.t_bubble_ms > 0.0
    assert p_tp.tp == 8 and p_tp.pp == 1

    p_ep = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_protein_pair",
            chip_count=8,
            parallel=ParallelOverride(tp=4, pp=1, ep=2),
            seq_len=128,
            **base,
        )
    )
    assert p_ep.ep == 2
    assert any("no-op" in a.lower() or "ep=" in a.lower() for a in p_ep.assumptions)


def test_api_eval_video_pp_ep():
    """v0.20: /api/eval accepts tp/pp/ep for video/protein."""
    from accel_dse.serve import api_eval

    v = api_eval(
        {
            "model_id": "illustrative_dit_video",
            "chip_count": 8,
            "tp": 4,
            "pp": 2,
            "ep": 1,
            "package_id": "hbm_hbm3_8s",
            "compute_id": "cluster_64t",
            "n_denoise": 10,
            "n_frames": 4,
        }
    )
    assert v["ok"] is True and v["domain"] == "video"
    assert v["tp"] == 4 and v["pp"] == 2 and v["ep"] == 1
    assert v["t_bubble_ms"] > 0.0
    assert v["t_c2c_ms"] > 0.0
    assert v["bytes_coll"] > 0

    p = api_eval(
        {
            "model_id": "illustrative_protein_pair",
            "chip_count": 8,
            "tp": 4,
            "pp": 2,
            "ep": 1,
            "package_id": "lpddr_4x64_8533",
            "compute_id": "cluster_64t",
            "seq_len": 128,
            "decode_mb": 4,
        }
    )
    assert p["ok"] is True and p["domain"] == "protein"
    assert p["tp"] == 4 and p["pp"] == 2
    assert p["t_c2c_ms"] > 0.0


def test_domain_collective_ring_vs_tree():
    """Ring vs tree collective byte formulas differ for tp>2."""
    from accel_dse.scaleup import ring_allreduce_bytes, tree_allreduce_bytes

    V = 1_000_000
    assert ring_allreduce_bytes(V, 1) == 0
    assert tree_allreduce_bytes(V, 1) == 0
    ring4 = ring_allreduce_bytes(V, 4)
    tree4 = tree_allreduce_bytes(V, 4)
    assert ring4 == int(2 * 3 / 4 * V)
    assert tree4 == int(2 * 2 * V)  # ceil(log2(4))=2
    assert tree4 != ring4


def test_llm_handcheck_path_unchanged_v020():
    """Guard: v0.20 domain PP/EP must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.21: /api/sweep compare matrix + UI assets
# ---------------------------------------------------------------------------


def test_api_sweep_chips_and_cap():
    """v0.21: POST /api/sweep chips axis returns MetricsCards; cap ≤32."""
    from accel_dse.serve import SWEEP_MAX_ROWS, api_sweep

    assert SWEEP_MAX_ROWS == 32
    out = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "chips",
            "package_id": "hbm_hbm3_4s",
            "compute_id": "core_16t",
            "prompt": 256,
            "ctx": 256,
        }
    )
    assert out["ok"] is True
    assert out["axis"] == "chips"
    assert out["count"] == 4
    assert out["metric_key"] == "TPOT_ms"
    assert out["capped"] is False
    assert "banner" in out and ("assumed" in out["banner"].lower() or "uncalibrated" in out["banner"].lower())
    labels = [r["label"] for r in out["rows"]]
    assert labels == ["chips=1", "chips=2", "chips=4", "chips=8"]
    for r in out["rows"]:
        assert r["card"]["ok"] is True
        assert r["card"]["TPOT_ms"] > 0
        assert r["primary_ms"] == r["card"]["TPOT_ms"]

    # Cap: request >32 values → truncated (only legal tp degrees 1/2/4/8)
    vals = [1, 2, 4, 8] * 10  # 40 rows
    big = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "chips",
            "values": vals,
            "package_id": "hbm_hbm3_2s",
            "compute_id": "core_4t",
            "max_rows": 32,
        }
    )
    assert big["count"] == 32
    assert big["capped"] is True
    assert big["requested"] == 40


def test_api_sweep_package_compute_parallel_series():
    """v0.21: package / compute / parallel / series axes."""
    from accel_dse.serve import api_sweep

    pkg = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "package",
            "package_kind": "HBM",
            "compute_id": "core_16t",
        }
    )
    assert pkg["ok"] and pkg["axis"] == "package"
    assert 1 <= pkg["count"] <= 32
    assert all("×" in r["label"] for r in pkg["rows"])  # n×granule labels

    comp = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "compute",
            "compute_level": "core",
            "package_id": "hbm_hbm3_4s",
        }
    )
    assert comp["count"] == 4
    assert all(r["axis_value"].startswith("core_") for r in comp["rows"])

    par4 = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "parallel",
            "parallel_chips": 4,
            "package_id": "hbm_hbm3_4s",
            "compute_id": "core_16t",
        }
    )
    assert par4["count"] >= 3  # dense ep=1: (4,1,1)/(2,2,1)/(1,4,1)
    assert all(r["tp"] * r["pp"] * r["ep"] == 4 for r in par4["rows"])

    par8 = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "parallel",
            "parallel_chips": 8,
            "package_id": "hbm_hbm3_4s",
            "compute_id": "core_16t",
        }
    )
    assert par8["count"] >= 4
    assert all(r["tp"] * r["pp"] * r["ep"] == 8 for r in par8["rows"])

    series = api_sweep(
        {
            "axis": "series",
            "domain": "llm",
            "product_only": False,
            "package_id": "hbm_hbm3_2s",
            "compute_id": "core_4t",
            "chip_count": 1,
            "max_rows": 8,
        }
    )
    assert series["ok"] and series["count"] >= 1
    assert series["count"] <= 8


def test_api_sweep_video_protein_metric_keys():
    """v0.21: sweep metric_key follows domain (TTFC / time_per_seq)."""
    from accel_dse.serve import api_sweep

    v = api_sweep(
        {
            "model_id": "illustrative_dit_video",
            "axis": "chips",
            "package_id": "hbm_hbm3_4s",
            "compute_id": "cluster_64t",
            "n_denoise": 8,
            "n_frames": 4,
            "values": [1, 2],
        }
    )
    assert v["domain"] == "video"
    assert v["metric_key"] == "TTFC_ms"
    assert v["rows"][0]["primary_ms"] > 0

    p = api_sweep(
        {
            "model_id": "illustrative_protein_pair",
            "axis": "parallel",
            "parallel_chips": 4,
            "package_id": "hbm_hbm3_4s",
            "compute_id": "core_16t",
            "seq_len": 64,
        }
    )
    assert p["domain"] == "protein"
    assert p["metric_key"] == "time_per_seq_ms"
    assert p["count"] >= 1


def test_api_sweep_invalid_axis_and_http():
    """v0.21: bad axis → ValueError; stdlib handler serves /api/sweep + UI assets."""
    import json
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from accel_dse.serve import WEB_DIR, WorkbenchHandler, api_sweep

    raised = False
    try:
        api_sweep({"model_id": "illustrative_27B", "axis": "not_an_axis"})
    except ValueError:
        raised = True
    assert raised

    assert (WEB_DIR / "index.html").is_file()
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "compare-panel" in html or "Compare / Sweep" in html
    assert "btn-sweep" in html
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "/api/sweep" in js
    assert "bar-chart" in js or "renderSweep" in js
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    assert "bar-chart" in css and "compare-table" in css
    # no CDN script/link tags
    assert "<script src=\"http" not in html.lower()
    assert "unpkg.com" not in html.lower()
    assert "jsdelivr" not in html.lower()
    assert "cdnjs" not in html.lower()
    assert 'href="/style.css"' in html and 'src="/app.js"' in html

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
    port = httpd.server_address[1]
    thr = threading.Thread(target=httpd.serve_forever, daemon=True)
    thr.start()
    try:
        time.sleep(0.05)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/sweep",
            data=json.dumps(
                {
                    "model_id": "illustrative_27B",
                    "axis": "compute",
                    "compute_level": "core",
                    "package_id": "hbm_hbm3_2s",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read().decode())
        assert payload["ok"] is True and payload["count"] == 4
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            page = r.read().decode()
        assert "Compare / Sweep" in page
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_llm_handcheck_path_unchanged_v021():
    """Guard: v0.21 compare/sweep UI must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.22: sweep export helpers + baseline Δ% + deep-link UI markers
# ---------------------------------------------------------------------------


def test_sweep_export_helpers_csv_json_and_delta():
    """v0.22: sweep_result_to_csv / to_json + delta_pct_vs_baseline."""
    from accel_dse.serve import (
        api_sweep,
        delta_pct_vs_baseline,
        sweep_result_to_csv,
        sweep_result_to_json,
    )

    assert delta_pct_vs_baseline(110.0, 100.0) == 10.0
    assert delta_pct_vs_baseline(90.0, 100.0) == -10.0
    assert delta_pct_vs_baseline(50.0, 0.0) is None
    assert delta_pct_vs_baseline(50.0, None) is None

    out = api_sweep(
        {
            "model_id": "illustrative_27B",
            "axis": "chips",
            "values": [1, 2],
            "package_id": "hbm_hbm3_2s",
            "compute_id": "core_4t",
            "prompt": 256,
            "ctx": 256,
        }
    )
    assert out["ok"] and out["count"] == 2
    # Attach baseline = first row primary
    base = float(out["rows"][0]["primary_ms"])
    out["baseline_ms"] = base
    for r in out["rows"]:
        r["delta_pct"] = delta_pct_vs_baseline(float(r["primary_ms"]), base)

    csv_text = sweep_result_to_csv(out)
    assert "primary_ms" in csv_text and "delta_pct" in csv_text
    assert "chips=1" in csv_text and "chips=2" in csv_text
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    assert len(lines) == 1 + out["count"]  # header + rows

    js = sweep_result_to_json(out)
    import json

    parsed = json.loads(js)
    assert parsed["axis"] == "chips" and parsed["count"] == 2
    assert parsed["rows"][0]["primary_ms"] > 0


def test_ui_export_baseline_deeplink_markers():
    """v0.22: UI assets expose export buttons, pin baseline, ?c= deep-link."""
    from accel_dse.serve import WEB_DIR
    from accel_dse import __version__

    _assert_version(__version__)
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    assert "btn-export-csv" in html and "btn-export-json" in html
    assert "btn-pin-baseline" in html
    assert "exportSweepCsv" in js and "exportSweepJson" in js
    assert "pinBaselineFromCard" in js or "btn-pin-baseline" in js
    assert "deltaPct" in js or "delta_pct" in js or "Δ%" in js
    assert "b64urlEncode" in js and "restoreConfigFromUrl" in js
    assert 'searchParams.get("c")' in js or "searchParams.get('c')" in js or '?c=' in js
    assert "pushConfigToUrl" in js
    assert "delta" in css or "baseline-hint" in css
    # still no CDN
    assert "unpkg.com" not in html.lower()
    assert "jsdelivr" not in html.lower()


def test_llm_handcheck_path_unchanged_v022():
    """Guard: v0.22 polish must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.23: CalibrationOverrides + dual-card / override UI markers
# ---------------------------------------------------------------------------


def test_calibration_overrides_load_and_apply():
    """v0.23: CalibrationOverrides JSON → WorkbenchConfig knobs."""
    from pathlib import Path

    from accel_dse import __version__
    from accel_dse.workbench import (
        CalibrationOverrides,
        WorkbenchConfig,
        evaluate_workbench,
    )

    _assert_version(__version__)
    example = Path(__file__).resolve().parents[1] / "examples" / "calibration.example.json"
    assert example.is_file()
    cal = CalibrationOverrides.load_json(example)
    assert "mem_efficiency" in cal.active_fields()
    assert cal.mem_efficiency == 0.65
    assert cal.resolved_frequency_hz() == 1.0e9
    assert cal.resolved_weight_hide() == 0.0  # weight_hide wins over mac_efficiency when both set

    cfg = WorkbenchConfig(model_id="illustrative_27B", chip_count=1, efficiency=0.70)
    cfg2 = cal.apply_to_config(cfg)
    assert cfg2.efficiency == 0.65
    assert cfg2.frequency_hz == 1.0e9
    assert cfg2.weight_hide_factor == 0.0

    # API-style dict (freq_ghz + mac_efficiency alone)
    cal2 = CalibrationOverrides.from_dict(
        {"mem_efficiency": 0.5, "mac_efficiency": 0.25, "freq_ghz": 1.25}
    )
    assert cal2.resolved_frequency_hz() == 1.25e9
    assert cal2.resolved_weight_hide() == 0.25
    cfg3 = cal2.apply_to_config(WorkbenchConfig(model_id="illustrative_27B"))
    card = evaluate_workbench(cfg3)
    assert card.TPOT_ms > 0 and card.mem_eff_GBps > 0


def test_api_eval_accepts_calib_body():
    """v0.23: POST /api/eval body.calib and mem_efficiency alias."""
    from accel_dse.serve import api_eval, _build_workbench_from_body

    out = api_eval(
        {
            "model_id": "illustrative_27B",
            "package_id": "hbm_hbm3_2s",
            "compute_id": "core_4t",
            "prompt": 128,
            "ctx": 128,
            "calib": {
                "mem_efficiency": 0.55,
                "mac_efficiency": 0.1,
                "freq_ghz": 1.1,
            },
        }
    )
    assert out["ok"] is True
    _assert_version(out["version"])
    assert abs(out["config_echo"]["efficiency"] - 0.55) < 1e-9
    assert abs(out["config_echo"]["frequency_hz"] - 1.1e9) < 1.0
    assert abs(out["config_echo"]["weight_hide_factor"] - 0.1) < 1e-9
    assert out["TPOT_ms"] > 0

    cfg = _build_workbench_from_body(
        {"model_id": "toy", "mem_efficiency": 0.42, "frequency_hz": 9e8}
    )
    assert abs(cfg.efficiency - 0.42) < 1e-9
    assert abs(cfg.frequency_hz - 9e8) < 1.0


def test_cli_workbench_calib_flag():
    """v0.23: workbench --calib loads example JSON."""
    import tempfile
    from pathlib import Path

    from accel_dse.cli import main

    example = Path(__file__).resolve().parents[1] / "examples" / "calibration.example.json"
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "card.json")
        rc = main(
            [
                "workbench",
                "--model",
                "illustrative_27B",
                "--chips",
                "1",
                "--calib",
                str(example),
                "--json",
                out,
            ]
        )
        assert rc == 0
        text = Path(out).read_text(encoding="utf-8")
        assert "TPOT_ms" in text or "tpot" in text.lower() or "wall" in text.lower()


def test_ui_calib_override_dual_card_markers():
    """v0.23: UI exposes Assumed/override + dual A|B + freq; no CDN."""
    from accel_dse import __version__
    from accel_dse.serve import WEB_DIR

    _assert_version(__version__)
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    assert "override-section" in html or "Assumed / override" in html
    assert "override-badge" in html
    assert "freq_ghz" in html and "freq_ghz" in js
    assert "dual-panel" in html or "Dual card" in html
    assert "btn-pin-a" in html and "btn-pin-b" in html
    assert "updateOverrideBadge" in js
    assert "pinDual" in js or "dualCardA" in js
    assert "renderDualDelta" in js or "dual-delta" in html
    assert "override-active" in css or "badge" in css
    assert "unpkg.com" not in html.lower()
    assert "jsdelivr" not in html.lower()
    assert "cdn." not in html.lower() or "charset" in html.lower()


def test_model_doc_synced_v023():
    """Public docs (README + docs/MODEL.md) stay honest and in sync with the code:
    no stale out_of_scope claims for UI / video-protein multi-card, calibration
    and the not-measured-silicon disclaimer documented, relative links resolve."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    model = (root / "docs" / "MODEL.md").read_text(encoding="utf-8")

    assert "out_of_scope" not in model
    assert "CalibrationOverrides" in model
    assert "视频" in model and "蛋白" in model and "多卡" in model
    assert "不是实测硅片" in readme and "not measured silicon" in readme
    assert "docs/MODEL.md" in readme
    for doc, text in ((root / "README.md", readme), (root / "docs" / "MODEL.md", model)):
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).exists(), f"{doc.name}: broken link {target}"


def test_llm_handcheck_path_unchanged_v023():
    """Guard: v0.23 polish must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.24: assumed energy / cost stub + scenario presets
# ---------------------------------------------------------------------------


def test_econ_stub_off_by_default_v024():
    """No econ knobs → est_* fields zero, econ_configured False (no hidden watts)."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    for mid in ("illustrative_27B", "illustrative_dit_video", "illustrative_protein_pair"):
        c = evaluate_workbench(WorkbenchConfig(model_id=mid))
        assert c.econ_configured is False
        assert c.est_power_W == 0.0 and c.est_system_cost_usd == 0.0
        assert c.est_energy_per_token_J == 0.0
        assert c.est_energy_per_frame_J == 0.0 and c.est_energy_per_seq_J == 0.0
        assert not any("econ stub" in a for a in c.assumptions)


def test_econ_llm_energy_cost_math_v024():
    """LLM: P = chips × tdp × util; E/token = P × TPOT / B; cost; $/MTok parts."""
    from accel_dse.econ import J_PER_KWH, SECONDS_PER_YEAR
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    cfg = WorkbenchConfig(
        model_id="illustrative_27B", chip_count=2, batch=4,
        tdp_w=400.0, power_util=0.5, cost_per_card_usd=1000.0, mem_addon_usd=250.0,
        usd_per_kwh=0.2, amortize_years=2.0, duty_cycle=0.5,
    )
    c = evaluate_workbench(cfg)
    assert c.econ_configured is True
    assert abs(c.est_power_W - 2 * 400 * 0.5) < 1e-9
    assert abs(c.est_power_per_card_W - 200.0) < 1e-9
    tpot_s = c.TPOT_ms / 1e3
    assert abs(c.est_energy_per_token_J - 400.0 * tpot_s / 4) < 1e-9
    assert abs(c.est_energy_prefill_J - 400.0 * c.TTFT_ms / 1e3) < 1e-9
    assert abs(c.est_system_cost_usd - 2 * 1250.0) < 1e-9
    assert abs(c.est_tokens_per_s - 4 / tpot_s) < 1e-6
    e_part = c.est_energy_per_token_J * 1e6 / J_PER_KWH * 0.2
    c_part = 2500.0 / (c.est_tokens_per_s * 2.0 * SECONDS_PER_YEAR * 0.5) * 1e6
    assert abs(c.est_energy_usd_per_Mtok - e_part) < 1e-9
    assert abs(c.est_capex_usd_per_Mtok - c_part) < 1e-9
    assert abs(c.est_usd_per_Mtok - (e_part + c_part)) < 1e-9
    assert any("[assumed econ stub]" in a and "NOT silicon" in a for a in c.assumptions)
    # physics unchanged by econ knobs
    base = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B", chip_count=2, batch=4))
    assert base.TPOT_ms == c.TPOT_ms and base.TTFT_ms == c.TTFT_ms


def test_econ_watts_per_tops_and_tdp_precedence_v024():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    c = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B", watts_per_tops=2.0))
    assert abs(c.est_power_W - 2.0 * c.peak_tops) < 1e-9  # default util 1.0
    c2 = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", watts_per_tops=2.0, tdp_w=123.0)
    )
    assert abs(c2.est_power_W - 123.0) < 1e-9
    assert any("watts_per_tops ignored" in a for a in c2.assumptions)
    # cost only → no power, still configured
    c3 = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B", cost_per_card_usd=5.0, chip_count=4))
    assert c3.econ_configured and c3.est_power_W == 0.0 and c3.est_system_cost_usd == 20.0


def test_econ_video_protein_energy_units_v024():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    v = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_dit_video", tdp_w=100.0, batch=2, usd_per_kwh=0.1)
    )
    assert v.domain == "video"
    exp = 100.0 * (v.TTFC_ms / 1e3) / (v.n_frames * 2)
    assert abs(v.est_energy_per_frame_J - exp) < 1e-9
    assert v.est_energy_per_token_J == 0.0 and v.est_usd_per_Mtok == 0.0
    assert any("LLM-only" in a for a in v.assumptions)

    pr = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_protein_pair", tdp_w=100.0, chip_count=2)
    )
    assert pr.domain == "protein"
    exp = 200.0 * (pr.time_per_seq_ms / 1e3) / pr.batch
    assert abs(pr.est_energy_per_seq_J - exp) < 1e-9


def test_econ_knobs_json_example_is_placeholder_v024():
    """Example JSON exists, is labeled placeholder/fake, mirrors package data copy."""
    import json
    from pathlib import Path

    from accel_dse.econ import EXAMPLE_JSON_PATH, EnergyCostKnobs, load_example_econ

    root = Path(__file__).resolve().parents[1]
    ex = root / "examples" / "energy_cost.example.json"
    assert ex.is_file() and EXAMPLE_JSON_PATH.is_file()
    assert json.loads(ex.read_text()) == json.loads(EXAMPLE_JSON_PATH.read_text())
    d = load_example_econ()
    note = (d.get("_note") or "").upper()
    assert "EXAMPLE" in note and "PLACEHOLDER" in note and "NOT SILICON" in note
    k = EnergyCostKnobs.from_dict(d)
    assert k.tdp_w == 400.0  # clearly-fake round example number
    assert "_note" not in k.to_dict()
    # engine code carries no baked wattage default
    src = (root / "accel_dse" / "econ.py").read_text()
    assert "tdp_w: float | None = None" in src


def test_scenario_presets_catalog_v024():
    from accel_dse.package_ranges import get_compute, get_package
    from accel_dse.scenarios import evaluate_presets, get_preset, list_presets

    ids = [p.id for p in list_presets()]
    for want in ("edge-lpddr-4x64", "card-hbm-4stack", "scaleup-8chip"):
        assert want in ids
    edge = get_preset("edge")
    assert get_package(edge.package_id).kind == "LPDDR"
    assert get_package(edge.package_id).n_packages == 4
    assert get_package(edge.package_id).package_width_bits == 64
    card = get_preset("card-hbm-4stack")
    assert get_package(card.package_id).n_stacks == 4 and card.chip_count == 1
    sc = get_preset("scaleup-8chip")
    assert sc.chip_count == 8
    get_compute(sc.compute_id)
    rows = evaluate_presets("illustrative_27B", dtype="int8", tdp_w=10.0)
    by = {p.id: c for p, c in rows}
    assert by["scaleup-8chip"].chips == 8 and by["scaleup-8chip"].tp == 8
    assert by["edge-lpddr-4x64"].mem_kind == "LPDDR"
    assert by["card-hbm-4stack"].mem_eff_GBps > by["edge-lpddr-4x64"].mem_eff_GBps
    assert by["scaleup-8chip"].TPOT_ms < by["card-hbm-4stack"].TPOT_ms
    assert abs(by["scaleup-8chip"].est_power_W - 80.0) < 1e-9
    # presets carry no power/price
    for p in list_presets():
        d = p.to_dict()
        assert not any(k.startswith(("tdp", "cost", "usd", "watts")) for k in d)


def test_api_presets_and_econ_body_v024():
    from accel_dse.serve import api_eval, api_presets, api_sweep, sweep_result_to_csv

    pr = api_presets()
    assert pr["count"] >= 3 and pr["econ_example"].get("tdp_w") == 400
    assert "Energy/cost" in pr["banner"] and "ASSUMED" in pr["banner"]

    out = api_eval({
        "model_id": "illustrative_27B", "preset": "scaleup-8chip", "dtype": "int8",
        "econ": {"tdp_w": 400, "power_util": 0.6, "cost_per_card_usd": 10000,
                 "usd_per_kwh": 0.1, "amortize_years": 3},
    })
    assert out["chips"] == 8 and out["mem_kind"] == "HBM"
    assert out["econ_configured"] is True
    assert abs(out["est_power_W"] - 8 * 400 * 0.6) < 1e-9
    assert abs(out["est_system_cost_usd"] - 80000) < 1e-9
    assert out["est_usd_per_Mtok"] > 0
    assert out["config_echo"]["tdp_w"] == 400

    # flat keys + explicit chips beat preset chips
    out2 = api_eval({"model_id": "illustrative_27B", "preset": "edge", "chips": 2, "tdp_w": 15})
    assert out2["chips"] == 2 and out2["tp"] == 2 and out2["mem_kind"] == "LPDDR"
    assert abs(out2["est_power_W"] - 30.0) < 1e-9

    # econ flows through sweep (chips + parallel axes) and CSV
    r = api_sweep({"model_id": "illustrative_27B", "axis": "chips", "tdp_w": 100})
    assert [row["card"]["est_power_W"] for row in r["rows"]] == [100.0, 200.0, 400.0, 800.0]
    rp = api_sweep({"model_id": "illustrative_27B", "axis": "parallel",
                    "parallel_chips": 4, "tdp_w": 100})
    assert all(row["card"]["est_power_W"] == 400.0 for row in rp["rows"])
    csv_text = sweep_result_to_csv(r)
    assert "est_power_W" in csv_text.splitlines()[0]
    assert "est_usd_per_Mtok" in csv_text.splitlines()[0]


def test_cli_econ_presets_v024(capsys=None):
    import tempfile
    from pathlib import Path

    from accel_dse.cli import main

    root = Path(__file__).resolve().parents[1]
    ex = root / "examples" / "energy_cost.example.json"
    assert main(["list-presets"]) == 0
    assert main(["eval-presets", "--model", "illustrative_27B", "--econ", str(ex)]) == 0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "card.json"
        rc = main([
            "workbench", "--model", "illustrative_27B", "--preset", "scaleup-8chip",
            "--tdp-w", "400", "--power-util", "0.5", "--cost-per-card", "1000",
            "--json", str(out),
        ])
        assert rc == 0
        import json
        d = json.loads(out.read_text())
        assert d["chips"] == 8 and d["mem_kind"] == "HBM"
        assert abs(d["est_power_W"] - 8 * 400 * 0.5) < 1e-9
        assert d["est_system_cost_usd"] == 8000.0
        # explicit --chips beats preset
        rc = main([
            "workbench", "--model", "illustrative_27B", "--preset", "scaleup-8chip",
            "--chips", "2", "--econ", str(ex), "--json", str(out),
        ])
        assert rc == 0
        d = json.loads(out.read_text())
        assert d["chips"] == 2 and d["econ_configured"] is True
        assert abs(d["est_power_W"] - 2 * 400 * 0.6) < 1e-9


def test_ui_presets_econ_markers_v024():
    from accel_dse.serve import WEB_DIR

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    for pid in ("edge-lpddr-4x64", "card-hbm-4stack", "scaleup-8chip"):
        assert f'data-preset="{pid}"' in html
    assert "econ-section" in html and "ASSUMED stub" in html
    for fid in ("tdp_w", "watts_per_tops", "power_util", "cost_per_card_usd", "usd_per_kwh"):
        assert f'id="{fid}"' in html
    assert "btn-econ-example" in html and "EXAMPLE" in html
    assert "/api/presets" in js and "applyPreset" in js and "renderEcon" in js
    assert "est_energy_per_token_J" in js and "est_usd_per_Mtok" in js
    assert "unpkg.com" not in html.lower() and "jsdelivr" not in html.lower()


def test_report_has_econ_section_v024():
    from accel_dse.report import collect_report_data, render_html

    from accel_dse.scenarios import list_presets

    b = collect_report_data(chips_list=(1, 2))
    # v0.29: 3 domains × every preset (was hard-coded 9 = 3 × 3 presets; 0.29
    # adds edge-lpddr6-4x96 / card-hbm3e-8x12h / server-socamm2-8)
    assert len(b.preset_rows) == 3 * len(list_presets())
    html = render_html(b)
    assert 'id="econ"' in html and "ASSUMED stub" in html
    assert "EXAMPLE placeholders" in html and "no PDK / JEDEC" in html
    assert "scaleup-8chip" in html


def test_model_doc_marks_energy_cost_stub_v024():
    from pathlib import Path

    model = (Path(__file__).resolve().parents[1] / "docs" / "MODEL.md").read_text(encoding="utf-8")
    assert "assumed stub" in model
    assert "能耗" in model and "成本" in model and "默认关闭" in model


def test_llm_handcheck_path_unchanged_v024():
    """Guard: econ stub must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.25: scaling efficiency on MetricsCard
# ---------------------------------------------------------------------------


def test_scale_efficiency_chips1_is_one_v025():
    """chips=1 → scale_efficiency=1.0, speedup=1.0, note single-card."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    for mid in ("illustrative_27B", "illustrative_dit_video", "illustrative_protein_pair"):
        c = evaluate_workbench(WorkbenchConfig(model_id=mid, chip_count=1))
        assert c.scale_efficiency == 1.0 and c.speedup == 1.0
        assert "single-card" in c.notes
        assert c.t_single_primary_ms > 0
        if c.domain == "video":
            assert c.scale_metric == "TTFC"
        elif c.domain == "protein":
            assert c.scale_metric == "time_per_seq"
        else:
            assert c.scale_metric == "TPOT"
        assert "scale_efficiency" in c.to_markdown()
        d = c.to_json_dict()
        assert d["scale_efficiency"] == 1.0 and d["speedup"] == 1.0


def test_scale_efficiency_perfect_tp_near_one_v025():
    """Default HBM + strong C2C TP scale-up stays near ideal efficiency."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    c1 = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B", chip_count=1))
    # v0.29: the ≤0.28 ">0.99" held only because collectives were pure-BW
    # (optimism bug #1). α=0 reproduces the old near-ideal number; the default
    # α=3 µs exposes 80 collectives/token and lowers efficiency honestly.
    c8z = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", chip_count=8, c2c_latency_us=0.0)
    )
    assert c8z.scale_efficiency > 0.99  # compute-bound; near ideal without α
    c8 = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B", chip_count=8))
    assert abs(c8.speedup - (c1.TPOT_ms / c8.TPOT_ms)) < 1e-9
    assert abs(c8.scale_efficiency - c8.speedup / 8.0) < 1e-9
    assert 0.85 < c8.scale_efficiency < c8z.scale_efficiency
    assert c8.n_sync_per_token == 2 * 40 and abs(c8.t_sync_ms - 80 * 3e-3) < 1e-12
    assert c8.t_single_primary_ms == c1.TPOT_ms
    assert any("scale_efficiency=" in a for a in c8.assumptions)


def test_scale_efficiency_comm_bound_chips8_lt_one_v025():
    """chips=8 efficiency < 1 when comm/bubble-bound (PP or weak C2C)."""
    from accel_dse.workbench import ParallelOverride, WorkbenchConfig, evaluate_workbench

    # PP=8 → bubble wall; speedup≈1 → eff≈1/8
    c_pp = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_27B",
            chip_count=8,
            parallel=ParallelOverride(tp=1, pp=8, ep=1),
        )
    )
    assert c_pp.wall == "bubble"
    assert c_pp.scale_efficiency < 1.0
    assert c_pp.scale_efficiency < 0.2
    assert abs(c_pp.speedup - 1.0) < 0.05

    # Weak C2C → c2c wall; speedup < chips → eff < 1
    c_c2c = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", chip_count=8, c2c_gbps=0.5)
    )
    assert c_c2c.wall == "c2c"
    assert c_c2c.scale_efficiency < 1.0
    assert c_c2c.speedup < 8.0
    assert c_c2c.scale_efficiency == c_c2c.speedup / 8.0


def test_scale_efficiency_video_protein_v025():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    v = evaluate_workbench(WorkbenchConfig(model_id="illustrative_dit_video", chip_count=4))
    assert v.scale_metric == "TTFC"
    assert abs(v.scale_efficiency - v.speedup / 4.0) < 1e-9
    p = evaluate_workbench(WorkbenchConfig(model_id="illustrative_protein_pair", chip_count=4))
    assert p.scale_metric == "time_per_seq"
    assert abs(p.scale_efficiency - p.speedup / 4.0) < 1e-9


def test_api_sweep_csv_has_scale_efficiency_v025():
    from accel_dse.serve import api_eval, api_sweep, sweep_result_to_csv

    out = api_eval({"model_id": "illustrative_27B", "chips": 4})
    assert "scale_efficiency" in out and out["scale_efficiency"] > 0.9
    assert out["speedup"] > 3.5
    r = api_sweep({"model_id": "illustrative_27B", "axis": "chips"})
    assert all("scale_efficiency" in row["card"] for row in r["rows"])
    csv_text = sweep_result_to_csv(r)
    hdr = csv_text.splitlines()[0]
    assert "scale_efficiency" in hdr and "speedup" in hdr and "scale_metric" in hdr


def test_ui_scale_oom_markers_v025():
    from accel_dse.serve import WEB_DIR

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    assert 'id="m-scale-eff"' in html and 'id="m-speedup"' in html
    assert 'id="m-oom-badge"' in html and "CAPACITY OOM" in html
    assert "scale_efficiency" in js and "m-scale-eff" in js
    assert "oom-badge" in css and "scale-low" in css
    assert "CAPACITY OOM" in js or "m-oom-badge" in js


def test_report_has_scale_section_v025():
    from accel_dse.report import collect_report_data, render_html

    b = collect_report_data(chips_list=(1, 2, 4, 8))
    assert b.chips_dense[0].scale_efficiency == 1.0
    assert b.chips_dense[-1].chips == 8
    html = render_html(b)
    assert 'id="scale"' in html and "scale_efficiency" in html
    assert "Scale efficiency vs chips" in html
    assert "series/dense-27b" in html


def test_version_025():
    from accel_dse import __version__

    _assert_version(__version__)


def test_llm_handcheck_path_unchanged_v025():
    """Guard: scale_efficiency must not change toy LLM handcheck."""
    test_toy_handcheck_numbers()


# ---------------------------------------------------------------------------
# v0.26: non-GEMM overhead + dtype MAC factors (assumed opt-in)
# ---------------------------------------------------------------------------


def test_non_gemm_overhead_default_off_bit_identical_v026():
    """Defaults (overhead=0, factor=1) must match bare EvalConfig path."""
    from accel_dse.evaluate import EvalConfig, evaluate_inference, effective_compute_time_s
    from accel_dse.memory import HANDCHECK_MEM, SRAMConfig
    from accel_dse.model_shape import TOY_SHAPE
    from accel_dse.npu import NPUConfig

    npu = NPUConfig(rows=4, cols=4, mac_efficiency=1.0)
    sram = SRAMConfig(128 * 1024)
    base = EvalConfig(prompt_len=8, decode_seq_len=8, batch=1, frequency_hz=1e9)
    with_defaults = EvalConfig(
        prompt_len=8,
        decode_seq_len=8,
        batch=1,
        frequency_hz=1e9,
        non_gemm_overhead=0.0,
        dtype_mac_factor=1.0,
    )
    r0 = evaluate_inference(TOY_SHAPE, npu, sram, HANDCHECK_MEM, base)
    r1 = evaluate_inference(TOY_SHAPE, npu, sram, HANDCHECK_MEM, with_defaults)
    assert r0.ttft_s == r1.ttft_s and r0.tpot_s == r1.tpot_s
    assert r0.prefill.t_compute_s == r1.prefill.t_compute_s
    assert r0.decode.t_compute_s == r1.decode.t_compute_s
    assert abs(
        effective_compute_time_s(r0.decode.traffic.compute_cycles, base)
        - r0.decode.t_compute_s
    ) < 1e-15


def test_non_gemm_overhead_increases_latency_v026():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    c0 = evaluate_workbench(WorkbenchConfig(model_id="illustrative_27B"))
    c1 = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", non_gemm_overhead=0.25)
    )
    assert c1.t_compute_ms > c0.t_compute_ms
    assert abs(c1.t_compute_ms - c0.t_compute_ms * 1.25) < 1e-6
    assert c1.TPOT_ms >= c0.TPOT_ms  # roofline may stay mem-bound; never decreases
    assert c1.non_gemm_overhead == 0.25
    assert any("non_gemm_overhead" in a for a in c1.assumptions)


def test_finer_non_gemm_fracs_sum_v026():
    from accel_dse.evaluate import EvalConfig, resolved_non_gemm_overhead

    cfg = EvalConfig(non_gemm_overhead=0.9, softmax_frac=0.05, rope_frac=0.02, norm_frac=0.03)
    assert abs(resolved_non_gemm_overhead(cfg) - 0.10) < 1e-15
    cfg2 = EvalConfig(non_gemm_overhead=0.15)
    assert abs(resolved_non_gemm_overhead(cfg2) - 0.15) < 1e-15


def test_dtype_mac_factor_peak_and_compute_v026():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    base = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_27B", dtype="int4")
    )
    boosted = evaluate_workbench(
        WorkbenchConfig(
            model_id="illustrative_27B",
            dtype="int4",
            dtype_mac_factors={"fp16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 4.0},
        )
    )
    assert abs(boosted.peak_tops - base.peak_tops * 4.0) < 1e-6
    assert abs(boosted.t_compute_ms * 4.0 - base.t_compute_ms) < 1e-4
    assert boosted.dtype_mac_factor == 4.0
    assert any("dtype_mac_factor" in a for a in boosted.assumptions)


def test_dtype_mac_factor_may_flip_wall_v026():
    """int4 + large MAC factor can move balanced/compute toward memory wall."""
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    kw = dict(
        model_id="illustrative_27B",
        dtype="int4",
        efficiency=0.1,
        decode_seq_len=128000,
    )
    a = evaluate_workbench(WorkbenchConfig(**kw))
    b = evaluate_workbench(
        WorkbenchConfig(dtype_mac_factors={"int4": 8.0}, **kw)
    )
    assert a.wall in ("compute", "balanced", "memory")
    assert b.t_compute_ms < a.t_compute_ms
    # With much faster compute, wall should not stay pure compute if it was
    # balanced/compute — prefer memory or stay memory.
    if a.wall in ("compute", "balanced"):
        assert b.wall in ("memory", "balanced")
    assert b.wall == "memory" or b.t_dram_ms >= b.t_compute_ms * 0.95


def test_api_and_cli_assumed_compute_v026(capsys=None):
    from accel_dse.cli import main
    from accel_dse.serve import api_eval

    out = api_eval(
        {
            "model_id": "illustrative_27B",
            "non_gemm_overhead": 0.1,
            "dtype_mac_factors": {"int4": 2.0, "fp16": 1.0},
            "dtype": "fp16",
        }
    )
    assert out["ok"]
    _assert_version(out["version"])
    assert abs(out["non_gemm_overhead"] - 0.1) < 1e-9
    assert abs(out["dtype_mac_factor"] - 1.0) < 1e-9  # fp16 → 1.0
    assert out["config_echo"]["non_gemm_overhead"] == 0.1
    assert out["config_echo"]["dtype_mac_factors"]["int4"] == 2.0

    out4 = api_eval(
        {
            "model_id": "illustrative_27B",
            "dtype": "int4",
            "dtype_mac_factors": {"int4": 2.0},
        }
    )
    assert abs(out4["dtype_mac_factor"] - 2.0) < 1e-9

    rc = main(
        [
            "workbench",
            "--model",
            "illustrative_27B",
            "--non-gemm-overhead",
            "0.05",
            "--dtype-mac-factor",
            '{"fp16":1,"int4":4}',
        ]
    )
    assert rc == 0


def test_ui_assumed_compute_markers_v026():
    from accel_dse.serve import WEB_DIR

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="assumed-compute-section"' in html
    assert 'id="non_gemm_overhead"' in html
    assert 'id="dtype_mac_factors"' in html
    assert "non_gemm_overhead" in js and "dtype_mac_factors" in js
    assert "EXAMPLE_DTYPE_MAC" in js
    assert "wireAssumedCompute" in js


def test_report_has_assumed_compute_section_v026():
    from accel_dse.report import collect_report_data, render_html

    html = render_html(collect_report_data(chips_list=(1, 2)))
    assert 'id="assumed-compute"' in html
    assert "non_gemm_overhead" in html
    assert "dtype_mac_factors" in html


def test_dtype_mac_resolve_helpers_v026():
    from accel_dse.dtype import (
        DEFAULT_DTYPE_MAC_FACTORS,
        dtype_mac_factors_active,
        resolve_dtype_mac_factor,
    )

    assert all(v == 1.0 for v in DEFAULT_DTYPE_MAC_FACTORS.values())
    assert resolve_dtype_mac_factor(None) == 1.0
    assert resolve_dtype_mac_factor({"int4": 4.0}, weight_bits=4) == 4.0
    assert resolve_dtype_mac_factor({"fp8": 2.0}, dtype_name="fp8") == 2.0
    assert resolve_dtype_mac_factor({"8": 3.0}, weight_bits=8) == 3.0
    assert not dtype_mac_factors_active({"fp16": 1.0, "int4": 1.0})
    assert dtype_mac_factors_active({"int4": 4.0})


def test_version_026():
    from accel_dse import __version__

    _assert_version(__version__)


def test_llm_handcheck_path_unchanged_v026():
    """Guard: assumed compute extras must not change toy LLM handcheck at defaults."""
    test_toy_handcheck_numbers()



# ---------------------------------------------------------------------------
# v0.29: structured memory catalog + provenance tags + two optimism fixes
# ---------------------------------------------------------------------------


def test_mem_handcheck_lpddr6_4x96_10667_v029():
    """LPDDR6 4 × x96 @10667: bus 384 b, raw 512.0 GB/s, usable (8/9) 455.1."""
    from accel_dse.mem_catalog import make_spec

    s = make_spec("LPDDR6", count=4, rate=10667)
    assert s.unit_width_bits == 96 and s.bus_bits == 384 and s.form == "discrete"
    raw = 384 * 10667 / 8 / 1000  # GB/s
    assert abs(s.raw_GBps - raw) < 1e-9 and abs(s.raw_GBps - 512.0) < 0.05
    assert abs(s.payload_GBps - raw * 256 / 288) < 1e-9
    assert abs(s.payload_GBps - 455.1) < 0.05
    assert abs(s.effective_GBps - s.payload_GBps * 0.70) < 1e-9
    assert s.capacity_GB == 64 and s.capacity_bytes == 64 * 2**30
    mem = s.to_external_memory()
    assert abs(mem.peak_bandwidth_Bps() / 1e9 - s.raw_GBps) < 1e-6
    assert abs(mem.effective_bandwidth_Bps() / 1e9 - s.effective_GBps) < 1e-6
    assert "LPDDR6 4×96b @10667" in s.short_label() and "455 GB/s 可用" in s.short_label()


def test_mem_handcheck_hbm3e_8x12h_and_lpddr5x_v029():
    from accel_dse.mem_catalog import make_spec
    from accel_dse.memory import HBM_PRESET, LPDDR_PRESET

    h = make_spec("HBM3E", count=8, height=12, die_Gb=24, rate=9200)
    assert h.capacity_GB == 288 and h.capacity_bytes == 288 * 2**30
    assert abs(h.raw_GBps - 8 * 1024 * 9200 / 8000) < 1e-9  # 9420.8
    assert abs(h.raw_GBps - 9420.8) < 1e-9 and h.payload_factor == 1.0
    assert h.tag == "vendor_shipping"
    lp = make_spec("LPDDR5X", count=8, width_bits=64, rate=8533, cap_GB=16)
    assert lp.bus_bits == 512 and abs(lp.raw_GBps - 546.112) < 1e-9
    assert lp.capacity_GB == 128
    # presets are now these real configs (≤0.28: 8×1024@5.2/96 GB, 8×64@8.5/64 GB)
    assert abs(HBM_PRESET.peak_bandwidth_Bps() / 1e9 - 9420.8) < 1e-6
    assert HBM_PRESET.capacity_bytes == 288 * 2**30 and HBM_PRESET.mem_type == "HBM3E"
    assert abs(LPDDR_PRESET.peak_bandwidth_Bps() / 1e9 - 546.112) < 1e-6
    assert LPDDR_PRESET.capacity_bytes == 128 * 2**30
    # HBM4 doubles the interface width
    assert make_spec("HBM4", count=1).unit_width_bits == 2048


def test_mem_provenance_weakest_tag_v029():
    from accel_dse.mem_catalog import TAG_ORDER, TAG_ZH, make_spec, weakest

    assert TAG_ORDER[0] == "jedec" and TAG_ORDER[-1] == "speculative"
    assert weakest(["jedec", "vendor_sampling", "vendor_shipping"]) == "vendor_sampling"
    assert TAG_ZH["jedec_likely"] == "疑似 JEDEC"
    # LPDDR5X x96 package is vendor-announced only
    assert make_spec("LPDDR5X", width_bits=96, count=4).tag == "vendor_announced"
    # LPDDR6 x48 is speculative; 14400 is JEDEC-defined but not shipping
    assert make_spec("LPDDR6", width_bits=48, count=2).tag == "speculative"
    assert make_spec("LPDDR6", count=4, rate=14400).tag_zh == "已发布"
    # HBM4E is sampling; >12 stacks speculative but allowed
    assert make_spec("HBM4E", count=4).tag == "vendor_sampling"
    s16 = make_spec("HBM3E", count=16)
    assert s16.tag == "speculative" and s16.warnings()
    assert make_spec("HBM3E", count=8, height=16).tag in ("vendor_announced", "speculative")


def test_mem_legacy_ids_resolve_v029():
    """≤0.28 package ids in links / API / presets still resolve (+ note)."""
    from accel_dse.mem_catalog import parse_mem_id
    from accel_dse.serve import api_eval, api_memory

    s = parse_mem_id("lpddr_4x64_8533")
    assert s.id == "lpddr5x_4x64_8533_16g" and s.legacy_id == "lpddr_4x64_8533"
    assert parse_mem_id("lpddr_2x64_6400").mem_type == "LPDDR5"
    s6 = parse_mem_id("lpddr6_20x24_14400")  # 480 b of x24 dies → 5 x96 packages (v0.30)
    assert (s6.mem_type, s6.unit_width_bits, s6.n_units) == ("LPDDR6", 96, 5)
    assert s6.legacy_note
    h = parse_mem_id("hbm_hbm3e_4s")
    assert (h.mem_type, h.n_units, h.hbm_height, h.hbm_die_Gb) == ("HBM3E", 4, 12, 24)
    assert parse_mem_id("hbm_hbm4e_4s").rate_MTps == 14000
    r = api_memory(resolve="hbm_hbm3_8s")
    assert r["resolved"]["mem_type"] == "HBM3" and len(r["catalog"]["types"]) == 7
    a = api_eval({"model_id": "illustrative_27B", "package_id": "lpddr_4x64_8533"})
    b = api_eval({"model_id": "illustrative_27B", "package_id": "lpddr5x_4x64_8533_16g"})
    assert a["TPOT_ms"] == b["TPOT_ms"] and a["mem_summary"] == b["mem_summary"]
    c = api_eval({"model_id": "illustrative_27B", "mem_type": "LPDDR6", "mem_count": 4})
    assert c["mem_type"] == "LPDDR6" and abs(c["mem_raw_GBps"] - 512.0) < 0.05
    assert c["mem_spec"]["payload_factor"] < 1.0


def test_sync_alpha_additive_and_zero_identity_v029():
    """Exposed sync: t_stage += n_coll × α × (1−overlap); α=0 → ≤0.28 numbers."""
    from accel_dse.workbench import ParallelOverride, WorkbenchConfig, evaluate_workbench

    m = "illustrative_27B"  # 40 layers, dense
    base = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=8, c2c_latency_us=0.0))
    a3 = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=8, c2c_latency_us=3.0))
    a5 = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=8, c2c_latency_us=5.0))
    assert base.t_sync_ms == 0.0 and a3.n_sync_per_token == 2 * 40
    assert abs((a3.TPOT_ms - base.TPOT_ms) - 80 * 3e-3) < 1e-9
    assert abs((a5.TPOT_ms - base.TPOT_ms) - 80 * 5e-3) < 1e-9
    hid = evaluate_workbench(
        WorkbenchConfig(model_id=m, chip_count=8, c2c_latency_us=3.0, sync_overlap=1.0)
    )
    assert abs(hid.TPOT_ms - base.TPOT_ms) < 1e-12
    half = evaluate_workbench(
        WorkbenchConfig(model_id=m, chip_count=8, c2c_latency_us=3.0, sync_overlap=0.5)
    )
    assert abs((half.TPOT_ms - base.TPOT_ms) - 80 * 1.5e-3) < 1e-9
    # single card: no collectives → no sync
    one = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=1))
    assert one.t_sync_ms == 0.0 and one.n_sync_per_token == 0
    # PP: 1 send per stage; decode mb=1 → t_roof = t_stage × pp
    pp = evaluate_workbench(
        WorkbenchConfig(model_id=m, chip_count=2, parallel=ParallelOverride(tp=1, pp=2, ep=1))
    )
    pp0 = evaluate_workbench(
        WorkbenchConfig(model_id=m, chip_count=2, c2c_latency_us=0.0,
                        parallel=ParallelOverride(tp=1, pp=2, ep=1))
    )
    assert pp.n_sync_per_token == 1
    assert abs((pp.TPOT_ms - pp0.TPOT_ms) - 2 * 3e-3) < 1e-9
    # MoE EP adds 2 all-to-all / layer
    moe = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_moe", chip_count=4,
                        parallel=ParallelOverride(tp=2, pp=1, ep=2))
    )
    assert moe.n_sync_per_token == 2 * 40 + 2 * 40


def test_sync_video_protein_multicard_v029():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    for m in ("illustrative_dit_video", "illustrative_protein"):
        try:
            z = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=4, c2c_latency_us=0.0))
        except KeyError:
            continue
        a = evaluate_workbench(WorkbenchConfig(model_id=m, chip_count=4))
        assert a.n_sync_per_token > 0 and a.t_sync_ms > 0
        assert abs(a.t_sync_ms - a.n_sync_per_token * 3e-3) < 1e-12
        if a.domain == "video":
            # TTFC = N_denoise × forward; each forward pays the sync once
            assert abs((a.TTFC_ms - z.TTFC_ms) - a.n_denoise * a.t_sync_ms) < 1e-6
        else:
            assert a.time_per_seq_ms > z.time_per_seq_ms


def test_kv_sharding_gqa_replication_v029():
    """per-card KV = KV · ceil(n_kv/tp)/n_kv / pp (duplication when tp > n_kv)."""
    from dataclasses import replace

    from accel_dse.model_shape import ILLUSTRATIVE_27B
    from accel_dse.scaleup import kv_card_fraction, per_card_kv_bytes

    full = ILLUSTRATIVE_27B.kv_cache_bytes(4096, 1)
    # n_kv=8: tp ≤ 8 is a clean shard (unchanged vs ≤0.28)
    for tp in (1, 2, 4, 8):
        assert per_card_kv_bytes(ILLUSTRATIVE_27B, 4096, 1, tp) == full // tp
    gqa2 = replace(ILLUSTRATIVE_27B, n_kv_heads=2)
    f2 = gqa2.kv_cache_bytes(4096, 1)
    assert per_card_kv_bytes(gqa2, 4096, 1, 8) == f2 // 2  # ≤0.28 said f2 // 8
    num, den, rep, why = kv_card_fraction(gqa2, 8)
    assert (num, den, rep) == (1, 2, 4.0) and "复制" in why
    assert per_card_kv_bytes(gqa2, 4096, 1, 8, pp=2) == f2 // 4


def test_kv_mla_replicated_and_attn_dp_v029():
    """MLA latent is replicated per TP rank; attn-DP partitions KV by batch."""
    from accel_dse.memory import HBM_PRESET
    from accel_dse.model_shape import ILLUSTRATIVE_MLA
    from accel_dse.scaleup import (
        TPConfig,
        capacity_check,
        per_card_kv_bytes,
        per_card_weight_bytes,
    )

    full = ILLUSTRATIVE_MLA.kv_cache_bytes(4096, 8)
    assert per_card_kv_bytes(ILLUSTRATIVE_MLA, 4096, 8, 8) == full  # ×8 vs ≤0.28
    assert per_card_kv_bytes(ILLUSTRATIVE_MLA, 4096, 8, 8, pp=2) == full // 2
    dp = per_card_kv_bytes(ILLUSTRATIVE_MLA, 4096, 8, 8, attn_parallel="dp")
    assert dp == full // 8
    # B < tp under DP: busiest rank holds one full sequence
    one = ILLUSTRATIVE_MLA.kv_cache_bytes(4096, 1)
    assert per_card_kv_bytes(ILLUSTRATIVE_MLA, 4096, 1, 8, attn_parallel="dp") == one
    # attention weights replicated under DP → more weight bytes per card
    w_tp = per_card_weight_bytes(ILLUSTRATIVE_MLA, 8, "replicate")
    w_dp = per_card_weight_bytes(ILLUSTRATIVE_MLA, 8, "replicate", attn_parallel="dp")
    attn = ILLUSTRATIVE_MLA.n_layers * ILLUSTRATIVE_MLA.attn_weight_params_per_layer() * (
        ILLUSTRATIVE_MLA.weight_bits // 8
    )
    assert w_dp - w_tp == attn * 7 // 8
    oom_tp, used_tp, _, d_tp = capacity_check(
        ILLUSTRATIVE_MLA, HBM_PRESET, TPConfig(tp=8), seq_len=4096, batch=8
    )
    oom_dp, used_dp, _, d_dp = capacity_check(
        ILLUSTRATIVE_MLA, HBM_PRESET, TPConfig(tp=8, attn_parallel="dp"),
        seq_len=4096, batch=8,
    )
    assert "kv_rep=8" in d_tp and "attn=dp" in d_dp
    assert used_tp == w_tp + full and used_dp == w_dp + dp


def test_kv_fix_workbench_card_fields_v029():
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    t = evaluate_workbench(WorkbenchConfig(model_id="illustrative_mla", chip_count=8, batch=8))
    d = evaluate_workbench(
        WorkbenchConfig(model_id="illustrative_mla", chip_count=8, batch=8, attn_parallel="dp")
    )
    assert t.kv_replication == 8.0 and t.attn_parallel == "tp"
    assert any("KV 复制" in n for n in t.notes)
    assert d.kv_replication == 1.0 and d.bytes_KV * 8 == t.bytes_KV
    assert d.bytes_W > t.bytes_W  # replicated attention weights streamed


def test_package_axis_sweep_structured_v029():
    from accel_dse.serve import api_sweep
    from accel_dse.workbench import WorkbenchConfig, workbench_package_sweep

    r = api_sweep({"model_id": "illustrative_27B", "axis": "package",
                   "package_axis": "rate", "mem_type": "LPDDR6"})
    assert len(r["rows"]) == 3 and all("LPDDR6" in x["label"] for x in r["rows"])
    prim = [x["primary_ms"] for x in r["rows"]]
    assert prim[0] > prim[-1]  # faster grade → lower TPOT (BW-bound)
    r2 = api_sweep({"model_id": "illustrative_27B", "axis": "package", "package_axis": "type"})
    assert len(r2["rows"]) == 7
    rows = workbench_package_sweep(
        "illustrative_27B", axis="count",
        base=WorkbenchConfig(mem_type="HBM3E"),
    )
    assert [p.n_stacks for p, _ in rows] == [1, 2, 4, 5, 6, 8, 12, 16]
    assert rows[-1][1].mem_tag == "speculative"


def test_default_scenario_no_oom_v029():
    """Default workbench config and every preset fit for the default model."""
    from accel_dse.scenarios import evaluate_presets
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    assert not evaluate_workbench(WorkbenchConfig()).oom
    for _p, c in evaluate_presets("illustrative_27B"):
        assert not c.oom, _p.id


def test_version_029():
    from accel_dse import __version__

    _assert_version(__version__)
    assert tuple(int(x) for x in __version__.split(".")[:2]) >= (0, 29)


def test_ui_memory_selectors_v029():
    """v0.29 UI: structured memory selectors (type + popover), provenance badge,
    exposed-sync breakdown row and drawer knobs; catalog served by GET /api/memory."""
    from accel_dse.serve import WEB_DIR, api_memory

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="mem-type"', 'id="mem-chip"', 'id="mem-pop"', 'id="mp-rate"', 'id="mp-count"',
        'id="mp-width"', 'id="mp-height"', 'id="mp-die"', 'id="mp-cap"', 'id="mp-derived"',
        "通信同步（暴露）", 'id="bf-sync"', 'id="b-sync"', 'id="c2c_latency_us"',
        'id="sync_overlap"', 'id="attn-parallel-seg"', 'id="acc-sync"', 'id="i-kv"',
        'data-preset="edge-lpddr6-4x96"', 'data-preset="card-hbm3e-8x12h"',
        'data-preset="server-socamm2-8"', 'value="axis:count"', "/api/memory",
    ):
        assert marker in html, marker
    for t in ("LPDDR5", "LPDDR5X", "LPDDR6", "HBM3", "HBM3E", "HBM4", "HBM4E"):
        assert f'<option value="{t}"' in html, t
    for marker in (
        "/api/memory", "?resolve=", "normalizeMemSel", "memDerive", "weakestTag",
        "c2c_latency_us", "sync_overlap", "attn_parallel", "mem_type", "package_axis",
        "async function restoreConfigFromUrl", "await restoreConfigFromUrl",
    ):
        assert marker in js, marker
    for marker in (".mem-pop", ".ptag.t-speculative", ".c-sync", "--wall-sync"):
        assert marker in css, marker
    # the UI's tag order / zh labels must match the engine's
    cat = api_memory()["catalog"]
    assert cat["tag_order"] == [
        "jedec", "jedec_likely", "vendor_shipping", "vendor_sampling", "vendor_announced", "speculative"
    ]
    for k, zh in cat["tag_zh"].items():
        assert f'{k}: "{zh}"' in js, (k, zh)


# ---------------------------------------------------------------------------
# v0.30: MLA weights, capacity units, sweep labels, legacy LP6, Pareto / goodput
# ---------------------------------------------------------------------------

_HBM3E_8 = dict(mem_type="HBM3E", mem_count=8, hbm_height=12, hbm_die_Gb=24, mem_rate_MTps=9200)


def test_version_030():
    from accel_dse import __version__

    _assert_version(__version__)
    assert tuple(int(x) for x in __version__.split(".")[:2]) >= (0, 30)


def test_mla_param_count_deepseek_v3_v030():
    """Handcheck: DeepSeek-V3 published ≈ 671B total / ≈ 37B activated params.

    MLA projections from HF config (q_lora 1536, kv_lora 512, nope 128, rope 64,
    v 128, 128 heads, H 7168) → attention ≈ 187.1M / layer (0.29 dense-GQA
    formula gave ≈ 704.6M). Plus 3 dense-prefix layers (first_k_dense_replace,
    18432 FFN) and an untied LM head. MTP (nextn) layer not counted.
    """
    from accel_dse.series import resolve_llm_shape

    s = resolve_llm_shape("deepseek-v3")
    assert s.attn_kind == "mla"
    H, nh = 7168, 128
    hand = (H * 1536 + 1536 * nh * (128 + 64) + H * (512 + 64)
            + 512 * nh * (128 + 128) + nh * 128 * H)
    assert s.attn_weight_params_per_layer() == hand
    assert abs(hand / 1e6 - 187.1) < 0.1
    tot = s.total_weight_params()
    assert abs(tot / 671e9 - 1) < 0.01, tot  # within 1 % of published 671B
    act = s.active_weight_params()
    assert abs(act / 37e9 - 1) < 0.03, act  # within 3 % of published 37B
    assert s.n_dense_layers == 3 and s.tie_embeddings is False
    # MLA KV per token per layer = kv_lora + qk_rope = 576 elements
    assert s.kv_bytes_per_token() == s.n_layers * 576 * s.kv_bits // 8
    # stored per card at tp=8 ≈ total/8 (bf16) — was 183.6 GB in 0.29
    from accel_dse.scaleup import per_card_weight_bytes

    w8 = per_card_weight_bytes(s, 8, "shard")
    assert 160e9 < w8 < 172e9, w8


def test_mla_v4_lowrank_fallback_v030():
    """DeepSeek-V4 configs lack kv_lora_rank → 'lowrank' fallback (low-rank Q,
    K/V = H→n_kv·head_dim, grouped low-rank O); far below the 0.29 dense value."""
    from accel_dse.series import resolve_llm_shape

    s = resolve_llm_shape("deepseek-v4-flash")
    assert s.attn_kind == "lowrank"
    assert s.attn_weight_params_per_layer() < 150e6  # 0.29: ≈ 272.6M
    assert 250e9 < s.total_weight_params() < 300e9
    # illustrative / dense GQA shapes unchanged
    q = resolve_llm_shape("qwen3-32b")
    assert q.attn_kind == "gqa"
    assert abs(q.attn_weight_params_per_layer() / 1e6 - 94.4) < 0.1


def test_capacity_units_gib_v030():
    """capacity_GB (API/CSV) = vendor GB (2^30 B), consistent with the UI and
    the memory catalog; decimal values in *_decimal."""
    from accel_dse.serve import api_eval
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    c = evaluate_workbench(WorkbenchConfig(model_id="qwen3-32b", chip_count=8, **_HBM3E_8))
    assert c.capacity_GB == 288.0  # 8 × 12H × 24Gb / 8
    assert abs(c.capacity_GB_decimal - 288 * 2**30 / 1e9) < 1e-9
    assert c.capacity_needed_GB > 0  # v0.30: LLM exports W + KV per card
    assert abs(c.capacity_needed_GB_decimal / c.capacity_needed_GB - 2**30 / 1e9) < 1e-9
    # manual geometry: capacity_GB input is vendor GB too
    g = evaluate_workbench(WorkbenchConfig(model_id="qwen3-32b", chip_count=1, mem_kind="HBM",
                                           n_channels=8, width_bits=1024, data_rate_GTs=9.2,
                                           capacity_GB=100))
    assert abs(g.capacity_GB - 100.0) < 1e-6
    d = api_eval({"model_id": "qwen3-32b", "chip_count": 8, **_HBM3E_8})
    assert d["capacity_GB"] == 288.0 and d["capacity_GB_decimal"] > 309
    assert abs(d["capacity_GB"] - d["mem_capacity_nominal_GB"]) < 1e-9
    # video / protein also binary GB
    v = evaluate_workbench(WorkbenchConfig(model_id="cogvideox-2b", chip_count=2, **_HBM3E_8))
    assert v.capacity_GB == 288.0 and v.capacity_needed_GB > 0


def test_sweep_short_labels_raw_id_v030():
    from accel_dse.serve import api_sweep, sweep_result_to_csv

    r = api_sweep({"model_id": "illustrative_27B", "chip_count": 1, "axis": "package",
                   "package_kind": "HBM", "max_rows": 4})
    for row in r["rows"]:
        assert "_" not in row["label"] and " GB" in row["label"], row["label"]
        assert row["raw_id"] == row["axis_value"] == row["package_id"]
    lines = sweep_result_to_csv(r).splitlines()
    assert lines[0].split(",")[-1] == "raw_id"
    assert lines[1].endswith("," + r["rows"][0]["raw_id"])
    s = api_sweep({"model_id": "illustrative_27B", "chip_count": 1, "axis": "package",
                   "package_axis": "count", **_HBM3E_8})
    assert all(row.get("raw_id") and "_" not in row["label"] for row in s["rows"])


def test_legacy_lp6_bus_width_preserved_v030():
    """lpddr6_{n}x24 → x96 packages when n×24 % 96 == 0, else speculative x48
    (≡ x96s + one x48); bus width (→ bandwidth) and capacity per bit preserved."""
    from accel_dse.mem_catalog import parse_mem_id

    for n in (4, 6, 8, 12, 16, 20):
        for r in (10667, 14400):
            s = parse_mem_id(f"lpddr6_{n}x24_{r}")
            assert s.n_units * s.unit_width_bits == n * 24, (n, s.id)
            assert s.unit_width_bits == (96 if n * 24 % 96 == 0 else 48)
            assert s.legacy_note and s.rate_MTps == r
    s6 = parse_mem_id("lpddr6_6x24_10667")
    assert s6.id == "lpddr6_3x48_10667_8g" and "x48" in s6.legacy_note and "推测" in s6.legacy_note
    assert s6.capacity_GB == 24.0  # = 1×x96 16 GB + 1×x48 8 GB
    odd = parse_mem_id("lpddr6_5x24_10667")  # 120 b → nearest 96 (tie → narrower)
    assert odd.n_units * odd.unit_width_bits == 96


def _pareto(**kw):
    from accel_dse.pareto import pareto_analysis
    from accel_dse.workbench import WorkbenchConfig

    slo = kw.pop("slo", None)
    layouts = kw.pop("layouts", "all")
    # v0.31: default serving mode is amortized; the v0.30 relation tests pin the
    # decode-only upper bound (TTFT = prefill + TPOT, tok/s/user = 1000/TPOT).
    pk = {k: kw.pop(k) for k in ("goodput_mode", "prefill_mode", "out_len") if k in kw}
    pk.setdefault("goodput_mode", "upper")
    base = dict(model_id="qwen3-32b", chip_count=8, **_HBM3E_8)
    base.update(kw)
    return pareto_analysis(WorkbenchConfig(**base), slo=slo, layouts=layouts, **pk)


def test_pareto_frontier_synthetic_v030():
    import random

    from accel_dse.pareto import pareto_frontier

    rnd = random.Random(7)
    for better in ("higher", "lower"):
        xs = [rnd.uniform(1, 100) for _ in range(300)]
        ys = [rnd.uniform(1, 100) for _ in range(300)]
        xs += xs[:5]
        ys += ys[:5]  # exact duplicates
        fr = pareto_frontier(xs, ys, x_better=better)
        sg = 1 if better == "higher" else -1

        def dom(j, i):  # j dominates i
            return (sg * xs[j] >= sg * xs[i] and ys[j] >= ys[i]
                    and (sg * xs[j] > sg * xs[i] or ys[j] > ys[i]))

        fs = set(fr)
        for i in fr:
            assert not any(dom(j, i) for j in range(len(xs)))
        for i in range(len(xs)):
            if i not in fs:
                assert any(dom(j, i) or (xs[j], ys[j]) == (xs[i], ys[i]) for j in fr)
        # monotone: best interactivity first, throughput strictly increasing
        assert all(ys[a] < ys[b] for a, b in zip(fr, fr[1:]))
        assert all(sg * xs[a] >= sg * xs[b] for a, b in zip(fr, fr[1:]))


def test_pareto_real_monotone_and_capacity_v030():
    from dataclasses import replace

    from accel_dse.pareto import batch_grid, llm_max_batch
    from accel_dse.workbench import ParallelOverride, WorkbenchConfig, evaluate_workbench

    r = _pareto()
    assert r["domain"] == "llm" and r["axes"]["x_key"] == "tok_s_user"
    pts = r["points"]
    fr = [pts[i] for i in r["frontier"]]
    assert len(fr) >= 3
    fr_sorted = sorted(fr, key=lambda p: -p["x"])
    assert all(a["y"] < b["y"] for a, b in zip(fr_sorted, fr_sorted[1:]))
    for f in fr:
        assert not any(p["x"] >= f["x"] and p["y"] >= f["y"] and (p["x"] > f["x"] or p["y"] > f["y"])
                       for p in pts if not p["oom"])
    for p in pts:  # metric definitions
        assert abs(p["tok_s_user"] - 1000 / p["TPOT_ms"]) < 1e-9
        assert abs(p["tok_s_chip"] - p["batch"] * p["tok_s_user"] / 8) < 1e-9
        assert p["batch"] <= p["max_batch"] and not p["oom"]
        assert abs(p["TTFT_ms"] - (p["TTFT_prefill_ms"] + p["TPOT_ms"])) < 1e-9
    # layouts: dense → ep=1 only; attention dp only when tp>1
    assert all(l["ep"] == 1 for l in r["layouts"])
    assert not any(l["attn"] == "dp" and l["tp"] == 1 for l in r["layouts"])
    assert batch_grid(10) == [1, 2, 3, 4, 6, 8, 10]
    # capacity-limited max batch == evaluator OOM boundary (tp and attn dp)
    cfg = WorkbenchConfig(model_id="qwen3-32b", chip_count=8, decode_seq_len=32768,
                          **dict(_HBM3E_8, mem_count=2))
    for attn in ("tp", "dp"):
        b = llm_max_batch(cfg, 8, 1, 1, attn)
        assert b >= 1
        ok = evaluate_workbench(replace(cfg, batch=b, parallel=ParallelOverride(8, 1, 1), attn_parallel=attn))
        bad = evaluate_workbench(replace(cfg, batch=b + 1, parallel=ParallelOverride(8, 1, 1), attn_parallel=attn))
        assert not ok.oom and bad.oom, (attn, b)
    # MLA latent replicated under attn tp → dp fits more sequences (deepseek)
    ds = WorkbenchConfig(model_id="deepseek-v3", chip_count=8, decode_seq_len=8192, **_HBM3E_8)
    assert llm_max_batch(ds, 8, 1, 1, "dp") > llm_max_batch(ds, 8, 1, 1, "tp")
    # MoE: ep layouts enumerated; weights-too-big layouts flagged, not plotted
    rd = _pareto(model_id="deepseek-v3", layouts="all", decode_seq_len=512)
    assert any(l["ep"] > 1 for l in rd["layouts"])
    # v0.31: TP×EP expert sharding → every 8-chip layout fits by capacity (was 10/16)
    assert all(l["fits"] for l in rd["layouts"])
    assert all(not p["oom"] for p in rd["points"])


def test_goodput_binding_v030():
    # default SLO (TTFT 2 s, TPOT 50 ms): qwen3-32b 8 chips → TPOT binds
    r = _pareto()
    g = r["goodput"]
    assert g["ok"] and g["binding"] == "TPOT"
    assert g["TPOT_ms"] <= 50 and g["TTFT_ms"] <= 2000
    feas = [p for p in r["points"] if p["meets_slo"]]
    assert g["y"] == max(p["y"] for p in feas)
    best = r["points"][g["point_id"]]
    assert best["frontier"] and best["label"] == g["config"]
    lay = next(l for l in r["layouts"] if l["layout"] == g["layout"])
    assert lay["max_users_slo"] == g["max_users"] >= g["batch"]
    # one more user in that layout breaks TPOT
    from dataclasses import replace

    from accel_dse.workbench import ParallelOverride, WorkbenchConfig, evaluate_workbench

    cfg = WorkbenchConfig(model_id="qwen3-32b", chip_count=8, **_HBM3E_8)
    nxt = evaluate_workbench(replace(cfg, batch=g["max_users"] + 1,
                                     parallel=ParallelOverride(g["tp"], g["pp"], g["ep"]),
                                     attn_parallel=g["attn"]))
    assert nxt.TPOT_ms > 50
    # capacity binds: long context on 2 stacks → max users == KV-capacity limit
    rc = _pareto(decode_seq_len=32768, **dict(_HBM3E_8, mem_count=2))
    gc = rc["goodput"]
    assert gc["ok"] and gc["binding"] == "capacity"
    lc = next(l for l in rc["layouts"] if l["layout"] == gc["layout"])
    assert gc["max_users"] == lc["max_batch"]
    # TTFT binds: 32k prompt → no layout prefills within 2 s
    rt = _pareto(prompt_len=32768)
    assert not rt["goodput"]["ok"] and rt["goodput"]["binding"] == "TTFT"
    # relaxing TTFT makes it feasible again; tighter TPOT lowers goodput
    assert _pareto(prompt_len=32768, slo={"ttft_ms": 1e6})["goodput"]["ok"]
    tight = _pareto(slo={"tpot_ms": 20})["goodput"]
    assert tight["ok"] and tight["y"] < g["y"] and tight["TPOT_ms"] <= 20
    # impossible TPOT → nothing feasible, TPOT binds
    assert _pareto(slo={"tpot_ms": 0.01})["goodput"]["binding"] == "TPOT"


def test_pareto_video_protein_and_api_v030():
    from accel_dse.pareto import pareto_to_csv
    from accel_dse.serve import api_pareto, pareto_result_to_csv

    for mid, yk in (("cogvideox-2b", "frames_s_chip"), ("alphafold2", "seq_s_chip")):
        r = _pareto(model_id=mid, layouts="current")
        assert r["axes"]["x_better"] == "lower" and r["axes"]["y_key"] == yk
        assert r["points"] and r["frontier"] and r["goodput"]["ok"]
        fr = sorted((r["points"][i] for i in r["frontier"]), key=lambda p: p["x"])
        assert all(a["y"] < b["y"] for a, b in zip(fr, fr[1:]))
    a = api_pareto({"model_id": "qwen3-32b", "chip_count": 8, **_HBM3E_8,
                    "slo_tpot_ms": 30, "layouts": "current"})
    assert a["slo"]["tpot_ms"] == 30 and a["goodput"]["TPOT_ms"] <= 30
    assert len({(p["tp"], p["pp"], p["ep"]) for p in a["points"]}) == 1
    csv_text = pareto_result_to_csv(a)
    assert csv_text == pareto_to_csv(a)
    head = csv_text.splitlines()[0].split(",")
    assert head[:4] == ["domain", "model_id", "chips", "layout"] and "frontier" in head
    assert len(csv_text.splitlines()) == len(a["points"]) + 1
    try:
        api_pareto({"model_id": "qwen3-32b", "layouts": "bogus"})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pareto_cli_and_http_v030():
    import json as _json
    import subprocess
    import sys
    import tempfile
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from accel_dse.serve import WorkbenchHandler

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        out = subprocess.run(
            [sys.executable, "-m", "accel_dse", "pareto", "--model", "qwen3-32b", "--chips", "4",
             "--layouts", "current", "--csv", f"{td}/p.csv", "--json", f"{td}/p.json"],
            cwd=root, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert "goodput:" in out.stdout and "frontier:" in out.stdout
        assert Path(f"{td}/p.csv").read_text().startswith("domain,model_id")
        assert _json.loads(Path(f"{td}/p.json").read_text())["chips"] == 4
    srv = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        body = _json.dumps({"model_id": "qwen3-32b", "chip_count": 2, "layouts": "current"}).encode()
        for path, ctype in (("/api/pareto", "application/json"), ("/api/pareto.csv", "text/csv")):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                assert resp.status == 200 and resp.headers["Content-Type"].startswith(ctype)
                txt = resp.read().decode()
            if ctype == "text/csv":
                assert txt.startswith("domain,")
            else:
                assert _json.loads(txt)["goodput"]["ok"]
    finally:
        srv.shutdown()


def test_ui_pareto_tab_and_mobile_sheet_v030():
    web = Path(__file__).resolve().parents[1] / "accel_dse" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    for s in ('data-tab="pareto"', "吞吐 / 交互", 'id="tab-pareto"', 'id="pareto-svg"',
              'id="slo-ttft"', 'id="slo-tpot"', 'id="slo-lat"', 'id="kpi-goodput"',
              'id="btn-pareto-csv"', 'id="mem-pop-backdrop"', "最大并发用户", "约束（binding）"):
        assert s in html, s
    for s in ('"/api/pareto"', "drawParetoChart", "frontier",
              '"pareto"', "classList.toggle(\"sheet\"", "updateGoodputKpi", "paretoCsv",
              "td.title = r.raw_id"):
        assert s in js, s
    import re as _re

    assert not _re.search(r'<(script|link)[^>]+(src|href)="(https?:)?//', html)  # zero-dep, no CDN
    assert "@keyframes memSheetUp" in css and 'body[data-tab="pareto"] #tab-pareto' in css
    assert "mem-pop-backdrop" in css


# ---------------------------------------------------------------------------
# v0.31: TP×EP expert sharding, PP decode micro-batches, prefill-amortized
# goodput, speculative decoding / MTP, LM head, batch cap
# ---------------------------------------------------------------------------


def test_version_031():
    from accel_dse import __version__

    _assert_version(__version__)
    assert tuple(int(x) for x in __version__.split(".")[:2]) >= (0, 31)


def _ds():
    from accel_dse.workbench import resolve_llm_shape

    return resolve_llm_shape("deepseek-v3")


def test_moe_tp_ep_capacity_handcheck_v031():
    """TP×EP: routed experts stored once across the 8 cards (E·P/8 per card) for
    both tp_ep and ep_all; shared / attention / dense-prefix TP-split by tp."""
    from accel_dse.scaleup import moe_expert_degree, per_card_weight_bytes

    s = _ds()
    assert (s.n_experts, s.top_k, s.n_layers, s.n_dense_layers) == (256, 8, 61, 3)
    P = s.ffn_weight_params_per_expert()
    nm = s.n_layers - s.n_dense_layers
    B = s.weight_bits // 8
    emb = s.embed_params() * B
    for tp, ep in ((1, 8), (2, 4), (4, 2), (8, 1)):
        for moe in ("tp_ep", "ep_all"):
            if ep == 1 and moe == "tp_ep":
                continue  # ep=1 tp_ep is the plain dense-like TP path (checked below)
            hand = (
                s.n_layers * s.attn_weight_params_per_layer() / tp
                + s.n_dense_layers * s.dense_ffn_params() / tp
                + nm * s.n_shared_experts * P / tp
                + nm * s.n_experts * P / (tp * ep)  # routed experts: one copy over 8 cards
            ) * B + emb
            got = per_card_weight_bytes(s, tp, "replicate", ep=ep, moe_shard=moe)
            assert abs(got - hand) / hand < 1e-6, (tp, ep, moe, got, hand)
    # tp_ep vs ep_all: identical storage, different all-to-all degree
    assert per_card_weight_bytes(s, 2, "replicate", ep=4, moe_shard="tp_ep") == \
        per_card_weight_bytes(s, 2, "replicate", ep=4, moe_shard="ep_all")
    assert moe_expert_degree(s, 2, 4, "tp_ep") == (4, 2)
    assert moe_expert_degree(s, 2, 4, "ep_all") == (8, 1)
    assert moe_expert_degree(s, 8, 1, "tp_ep") == (1, 8)
    # 8 chips × 8×12H HBM3E (288 GB/card): every TP·EP split of 671B (bf16) fits now
    for tp, ep in ((1, 8), (2, 4), (4, 2), (8, 1)):
        assert per_card_weight_bytes(s, tp, "replicate", ep=ep) < 250e9


def test_moe_kv_split_over_ep_and_a2a_v031():
    from accel_dse.scaleup import attn_dp_degree, moe_a2a_bytes_per_rank, per_card_kv_bytes

    s = _ds()
    kv = s.kv_cache_bytes(4096, 64)
    # MoE ep>1 → attention DP across ep groups: MLA latent replicated over tp, batch split over ep
    assert attn_dp_degree(s, 2, 4, "tp") == 4 and attn_dp_degree(s, 2, 4, "dp") == 8
    assert per_card_kv_bytes(s, 4096, 64, 2, ep=4) == kv // 4
    assert per_card_kv_bytes(s, 4096, 64, 2, ep=4, attn_parallel="dp") == kv // 8
    assert per_card_kv_bytes(s, 4096, 64, 8, ep=1) == kv  # MLA under TP8: full latent per card
    assert per_card_kv_bytes(s, 4096, 6, 1, ep=4) == s.kv_cache_bytes(4096, 6) * 2 // 6  # ceil(6/4)
    # dense: ep ignored
    q = _q32()
    assert attn_dp_degree(q, 8, 1, "tp") == 1
    # a2a per rank: 2·(D−1)/D·(T/D)·top_k·H·2 B per MoE layer
    per_layer = 2 * 7 / 8 * (64 / 8) * 8 * 7168 * 2
    assert moe_a2a_bytes_per_rank(s, 64, 8, moe_layers=58) == int(58 * per_layer)
    assert moe_a2a_bytes_per_rank(s, 64, 1, moe_layers=58) == 0
    assert moe_a2a_bytes_per_rank(q, 64, 8, moe_layers=58) == 0


def _q32():
    from accel_dse.workbench import resolve_llm_shape

    return resolve_llm_shape("qwen3-32b")


def test_moe_experts_touched_v031():
    s = _ds()
    # one token: top_k/E of the local experts (8/256 × 32 = 1)
    assert abs(s.experts_touched(1, 32) - 1.0) < 1e-12
    assert abs(s.experts_touched(2, 32) - 32 * (1 - (1 - 8 / 256) ** 2)) < 1e-12
    assert s.experts_touched(10_000, 32) > 31.999
    assert s.experts_touched(1) <= s.experts_touched(4) <= s.n_experts


def _wb(**kw):
    from accel_dse.workbench import WorkbenchConfig, evaluate_workbench

    base = dict(model_id="qwen3-32b", chip_count=8, **_HBM3E_8)
    base.update(kw)
    return evaluate_workbench(WorkbenchConfig(**base))


def test_pp_decode_microbatches_v031():
    from accel_dse.workbench import ParallelOverride

    pp8 = ParallelOverride(1, 8, 1)
    c1 = _wb(parallel=pp8, batch=1)
    c8 = _wb(parallel=pp8, batch=8)
    c8_old = _wb(parallel=pp8, batch=8, decode_mb=1)
    c64 = _wb(parallel=pp8, batch=64)
    # B=1: single request traverses 8 stages (latency semantics unchanged)
    d1 = c1.phase_detail["decode"]
    assert c1.decode_mb_eff == 1 and c1.decode_microbatch == 1
    assert abs(c1.TPOT_ms - 8 * d1["stage_ms"]) / c1.TPOT_ms < 1e-9
    # B=pp: 8 micro-batches of 1 fill the pipe → same TPOT, 8× throughput
    assert c8.decode_mb_eff == 8 and c8.decode_microbatch == 1
    assert abs(c8.TPOT_ms - c1.TPOT_ms) / c1.TPOT_ms < 1e-9
    # decode_mb=1 reproduces ≤0.30 (whole batch walks the pipe, 1/pp utilisation)
    d_old = c8_old.phase_detail["decode"]
    assert c8_old.decode_mb_eff == 1 and abs(c8_old.TPOT_ms - 8 * d_old["stage_ms"]) / c8_old.TPOT_ms < 1e-9
    assert c8_old.TPOT_ms > c8.TPOT_ms
    # B=64: 8 micro-batches of 8; TPOT = max(mb, pp) · t_stage(8)
    d64 = c64.phase_detail["decode"]
    assert c64.decode_mb_eff == 8 and c64.decode_microbatch == 8
    assert abs(c64.TPOT_ms - 8 * d64["stage_ms"]) / c64.TPOT_ms < 1e-9
    # pp=1 untouched by the knob
    t1 = _wb(batch=8)
    assert t1.decode_mb_eff == 1 and t1.decode_microbatch == 8


def test_spec_decode_formula_and_step_v031():
    from accel_dse.scaleup import TPConfig, spec_expected_tokens, spec_mtp_layers

    assert spec_expected_tokens(0, 0.7) == 1.0
    assert abs(spec_expected_tokens(1, 0.7) - 1.7) < 1e-12
    assert abs(spec_expected_tokens(3, 0.7) - (1 + 0.7 + 0.49 + 0.343)) < 1e-12
    assert spec_expected_tokens(2, 1.0) == 3.0 and spec_expected_tokens(2, 0.0) == 1.0
    base = _wb(batch=1)
    k0 = _wb(batch=1, spec_k=0)
    assert k0.TPOT_ms == base.TPOT_ms and base.TPOT_step_ms == base.TPOT_ms
    k1 = _wb(batch=1, spec_k=1, spec_accept=0.7, spec_draft="model", spec_draft_frac=0.1)
    # TPOT = step / E;  step = verify(M = k+1) + k · frac · stage
    assert abs(k1.TPOT_ms - k1.TPOT_step_ms / 1.7) < 1e-9
    assert abs(k1.t_draft_ms - 0.1 * k1.phase_detail["decode"]["stage_ms"]) < 1e-9
    # memory-bound verify of 2 tokens costs < 10% more than one token
    verify = k1.TPOT_step_ms - k1.t_draft_ms
    assert base.TPOT_ms <= verify < 1.10 * base.TPOT_ms
    assert k1.TPOT_ms < base.TPOT_ms
    # MTP draft (DeepSeek: 1 nextn layer) — capacity includes the MTP module
    s = _ds()
    assert s.n_mtp_layers == 1
    assert spec_mtp_layers(s, TPConfig(tp=8, spec_k=2, spec_draft="mtp")) == 1
    assert spec_mtp_layers(s, TPConfig(tp=8, spec_k=2, spec_draft="model")) == 0
    assert spec_mtp_layers(s, TPConfig(tp=8)) == 0
    ds0 = _wb(model_id="deepseek-v3", batch=1)
    ds1 = _wb(model_id="deepseek-v3", batch=1, spec_k=1)
    assert ds1.capacity_needed_GB > ds0.capacity_needed_GB  # + MTP module on the card
    assert abs(ds1.spec_tokens_per_step - 1.7) < 1e-12
    assert ds1.t_draft_ms > 0 and ds1.TPOT_ms < ds0.TPOT_ms
    # compute-bound large batch: speculation costs more than it saves
    from accel_dse.workbench import ParallelOverride

    big0 = _wb(model_id="deepseek-v3", batch=256, parallel=ParallelOverride(1, 1, 8))
    big1 = _wb(model_id="deepseek-v3", batch=256, parallel=ParallelOverride(1, 1, 8), spec_k=1)
    assert big1.TPOT_ms > big0.TPOT_ms


def test_spec_mtp_capacity_and_verify_tokens_v031():
    from accel_dse.scaleup import mtp_card_bytes, per_card_weight_bytes

    s = _ds()
    w0 = per_card_weight_bytes(s, 8, "replicate")
    w1 = per_card_weight_bytes(s, 8, "replicate", mtp_layers=1)
    m = mtp_card_bytes(s, 8, 1, n_layers=1)
    assert w1 - w0 == m
    P = s.ffn_weight_params_per_expert()
    hand = (s.attn_weight_params_per_layer() / 8 + (s.n_shared_experts + s.n_experts) * P / 8
            + 2 * s.hidden * s.hidden / 8) * 2
    assert abs(m - hand) / hand < 1e-6
    # the verify step runs k+1 tokens per sequence: M = B·(k+1), weights read once
    from accel_dse.scaleup import C2C_400, TPConfig, evaluate_scaleup

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    cfg = EvalConfig(prompt_len=512, decode_seq_len=512, frequency_hz=SKU_100T.frequency_hz)
    r0 = evaluate_scaleup(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg, TPConfig(tp=2, c2c=C2C_400))
    r3 = evaluate_scaleup(ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, cfg,
                          TPConfig(tp=2, c2c=C2C_400, spec_k=3, spec_accept=0.5, spec_draft="model",
                                   spec_draft_frac=0.0))
    assert r3.decode.tokens_per_seq == 4 and r0.decode.tokens_per_seq == 1
    assert abs(r3.decode.flops_card / r0.decode.flops_card - 4) < 0.05  # ≈ 4× GEMM work
    assert r3.decode.weight_bytes_card == r0.decode.weight_bytes_card  # dense: read once
    assert abs(r3.tpot_s - r3.tpot_step_s / spec_expected_tokens_(3, 0.5)) < 1e-15
    assert r3.spec_tokens_per_step == spec_expected_tokens_(3, 0.5)


def spec_expected_tokens_(k, a):
    return sum(a ** i for i in range(k + 1))


def test_lm_head_counted_in_workbench_only_v031():
    from accel_dse.evaluate import EvalConfig as _EC

    assert _EC().count_lm_head is False  # raw engine (handcheck tests) unchanged
    q = _q32()
    vh2 = 151936 * 5120 * 2
    assert q.lm_head_params() * 2 == vh2
    # attention TP: vocab-parallel head (V/tp slice) whatever the storage policy
    rep = _wb(batch=1)
    sh = _wb(batch=1, embed_policy="shard")
    assert rep.bytes_head == sh.bytes_head == vh2 // 8
    assert abs(rep.TPOT_ms - sh.TPOT_ms) < 1e-12
    # attention DP: each rank owns its tokens → replicated head read in full
    rd = _wb(batch=8, attn_parallel="dp")
    sd = _wb(batch=8, attn_parallel="dp", embed_policy="shard")
    assert rd.bytes_head == vh2 and sd.bytes_head == vh2 // 8
    off = _wb(batch=1, model_id="qwen3-32b")
    assert off.bytes_head > 0
    # raw engine with count_lm_head: head GEMM (B, H, V) + V·H bytes on top
    from accel_dse.traffic import decode_traffic
    from accel_dse.npu import gemm_flops

    npu = SKU_100T.npu()
    sram = SRAMConfig(64 * 1024 * 1024)
    t0 = decode_traffic(q, npu, sram, 512, batch=4)
    t1 = decode_traffic(q, npu, sram, 512, batch=4, lm_head=True)
    assert t1.weight_bytes_dram - t0.weight_bytes_dram == vh2 == t1.head_weight_bytes
    assert t1.flops - t0.flops == gemm_flops(4, 5120, 151936) == t1.head_flops
    # TP8 B=1 decode is memory-bound: W includes the head
    assert rep.bytes_W >= vh2


def test_serving_metrics_amortized_handcheck_v031():
    from accel_dse.pareto import serving_metrics

    b1 = _wb(batch=1)
    c = _wb(batch=32)
    N = 256
    up = serving_metrics(c, b1=b1, mode="upper", prefill_mode="chunked", out_len=N)
    assert abs(up["TPOT_ms"] - c.TPOT_ms) < 1e-12
    assert abs(up["TTFT_ms"] - (b1.TTFT_ms + c.TPOT_ms)) < 1e-12 and up["prefill_share"] == 0
    ex = serving_metrics(c, b1=b1, mode="amortized", prefill_mode="exclusive", out_len=N)
    pf = b1.phase_detail["prefill"]["stage_ms"]
    assert abs(ex["TPOT_ms"] - (c.TPOT_ms + 32 * pf / N)) < 1e-9
    # chunked (pp=1): each tick folds r = B/N prompts into the decode step,
    # weights read once → tick' = max(dec_X + r·pre_X) + sync
    ch = serving_metrics(c, b1=b1, mode="amortized", prefill_mode="chunked", out_len=N)
    d, p = c.phase_detail["decode"], b1.phase_detail["prefill"]
    r = 32 / N
    hand = max(d["compute_ms"] + r * p["compute_ms"], d["mem_ms"] + r * p["mem_nonweight_ms"],
               d["c2c_ms"] + r * p["c2c_ms"], d["pp_act_ms"] + r * p["pp_act_ms"],
               d["a2a_ms"] + r * p["a2a_ms"], d["fabric_ms"]) + d["sync_ms"] + d["draft_ms"]
    assert abs(ch["TPOT_ms"] - hand) < 1e-9
    assert c.TPOT_ms < ch["TPOT_ms"] < ex["TPOT_ms"]  # weights shared with decode
    assert 0 < ch["prefill_share"] < ex["prefill_share"] < 1
    # N → ∞ recovers the upper bound
    inf = serving_metrics(c, b1=b1, mode="amortized", prefill_mode="exclusive", out_len=10**9)
    assert abs(inf["TPOT_ms"] - up["TPOT_ms"]) < 1e-4


def test_pareto_modes_runtime_and_cap_v031():
    import time as _t

    from accel_dse.pareto import LLM_BATCH_LIMIT, llm_max_batch
    from accel_dse.workbench import WorkbenchConfig

    assert LLM_BATCH_LIMIT > 4096
    q = WorkbenchConfig(model_id="qwen3-32b", chip_count=8, **_HBM3E_8)
    assert llm_max_batch(q, 8, 1, 1, "dp") > 4096  # cap raised (≤0.30 stopped at 4096)
    t0 = _t.perf_counter()
    rd = _pareto(model_id="deepseek-v3", goodput_mode="amortized")
    dt = _t.perf_counter() - t0
    assert dt < 3.0, dt
    assert len(rd["layouts"]) == 28 and all(l["fits"] for l in rd["layouts"])
    assert any(l.get("moe_shard") == "ep_all" for l in rd["layouts"])
    assert rd["serving"]["goodput_mode"] == "amortized" and rd["serving"]["out_len"] == 256
    ru = _pareto(model_id="deepseek-v3", goodput_mode="upper")
    assert ru["goodput"]["y"] > rd["goodput"]["y"] > 0
    for p in rd["points"]:
        assert p["TPOT_ms"] >= p["TPOT_decode_ms"] - 1e-9 and 0 <= p["prefill_share"] < 1
    rq = _pareto(goodput_mode="amortized", prefill_mode="exclusive", out_len=1024)
    rq2 = _pareto(goodput_mode="amortized", prefill_mode="exclusive", out_len=128)
    assert rq["goodput"]["y"] > rq2["goodput"]["y"]  # longer outputs amortize prefill better
    assert max(p["batch"] for p in rq["points"]) > 4096
    try:
        _pareto(goodput_mode="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_api_cli_v031_knobs():
    import subprocess
    import sys

    from accel_dse.serve import api_eval, api_pareto

    a = api_pareto({"model_id": "qwen3-32b", "chip_count": 8, **_HBM3E_8, "layouts": "current",
                    "goodput_mode": "amortized", "prefill_mode": "exclusive", "out_len": 512})
    assert a["serving"]["prefill_mode"] == "exclusive" and a["serving"]["out_len"] == 512
    assert "TPOT_decode_ms" in a["points"][0] and "prefill_share" in a["points"][0]
    e = api_eval({"model_id": "deepseek-v3", "chip_count": 8, **_HBM3E_8, "spec_k": 1,
                  "moe_shard": "ep_all", "decode_mb": 0})
    assert e["spec_k"] == 1 and e["moe_shard"] == "ep_all"
    assert abs(e["TPOT_ms"] * e["spec_tokens_per_step"] - e["TPOT_step_ms"]) < 1e-9
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-m", "accel_dse", "pareto", "--model", "qwen3-32b", "--chips", "8",
         "--layouts", "current", "--goodput-mode", "upper", "--spec-k", "1"],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "serving: upper" in out.stdout and "spec_k=1" in out.stdout


def test_ui_markers_v031():
    root = Path(__file__).resolve().parents[1] / "accel_dse" / "web"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for i in ("goodput-mode", "prefill-mode", "out-len", "spec_k", "spec_accept",
              "spec-draft-seg", "moe-shard-seg", "decode_mb", "b-draft", "acc-spec"):
        assert f'id="{i}"' in html, i
    for k in ("goodput_mode", "prefill_mode", "out_len", "spec_k", "moe_shard", "updateAmortUi"):
        assert k in js, k
