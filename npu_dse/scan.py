"""Parameter sweeps: SRAM size, memory preset, PE peak, SKU templates."""

from __future__ import annotations

from .evaluate import EvalConfig, InferenceResult, evaluate_inference
from .memory import (
    DEFAULT_SRAM,
    HBM_PRESET,
    LPDDR_PRESET,
    ExternalMemory,
    SRAMConfig,
)
from .model_shape import (
    ILLUSTRATIVE_27B,
    ILLUSTRATIVE_MLA,
    ILLUSTRATIVE_MOE,
    TOY_SHAPE,
    ModelShape,
)
from .npu import DEFAULT_NPU, NPUConfig
from .sku import SKU_100T, SKU_1P, SKU_BASELINE, SKUPreset


def _fmt_row(r: InferenceResult) -> str:
    sku = r.npu.sku_name or ""
    tops = r.peak_tops()
    return (
        f"{r.shape.name:18s} {r.mem.kind:6s} "
        f"SRAM={r.sram.capacity_bytes/2**20:7.1f}MiB "
        f"PE={r.npu.rows}x{r.npu.cols}x{r.npu.n_engines:<3d} "
        f"{tops:8.1f}T "
        f"{sku:12s} "
        f"TTFT={r.ttft_s*1e3:10.3f}ms [{r.prefill.wall:7s}] "
        f"(c={r.prefill.t_compute_s*1e3:.2f}/m={r.prefill.t_memory_s*1e3:.2f}) "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.2f}/m={r.decode.t_memory_s*1e3:.2f}) "
        f"dec_util={r.decode.traffic.mean_gemm_util:.4f} "
        f"Wpath={r.decode.traffic.weight_path} "
        f"R={r.decode.traffic.resident_layers}/{r.shape.n_layers}"
    )


def run_default_scan(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
) -> list[InferenceResult]:
    """Default DSE scan: toy + 27B-ish × HBM/LPDDR × SRAM / PE / SKU points."""
    cfg = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=1.0e9,
        pin_weights=False,
        contention_mode="serialize",
    )
    shapes: list[ModelShape] = [TOY_SHAPE, ILLUSTRATIVE_27B, ILLUSTRATIVE_MOE]
    mems: list[ExternalMemory] = [HBM_PRESET, LPDDR_PRESET]
    sram_sizes = [
        4 * 1024 * 1024,      # 4 MiB
        64 * 1024 * 1024,     # 64 MiB
        256 * 1024 * 1024,    # 256 MiB
    ]
    pe_cfgs = [
        NPUConfig(rows=32, cols=32, n_engines=1),
        NPUConfig(rows=64, cols=64, n_engines=1),
        NPUConfig(rows=128, cols=128, n_engines=1),
        NPUConfig(rows=64, cols=64, n_engines=16, sku_name="pe64x16eng"),
    ]

    results: list[InferenceResult] = []
    if verbose:
        print("npu_dse default scan (assumed/uncalibrated HW knobs)")
        print(f"Eval: prompt={cfg.prompt_len} decode_ctx={cfg.decode_seq_len} "
              f"batch={cfg.batch} freq={cfg.frequency_hz/1e9:.2f}GHz "
              f"contention={cfg.contention_mode}")
        print("-" * 140)

    for shape in shapes:
        for mem in mems:
            r = evaluate_inference(shape, DEFAULT_NPU, DEFAULT_SRAM, mem, cfg)
            results.append(r)
            if verbose:
                print(_fmt_row(r))

    if verbose:
        print("-" * 140)
        print("SRAM sweep (illustrative_27B @ HBM, PE=64x64x1):")
    for sz in sram_sizes:
        r = evaluate_inference(
            ILLUSTRATIVE_27B, DEFAULT_NPU, SRAMConfig(sz), HBM_PRESET, cfg
        )
        results.append(r)
        if verbose:
            print(_fmt_row(r))

    if verbose:
        print("-" * 140)
        print("PE / engine sweep (illustrative_27B @ LPDDR, SRAM=64MiB):")
    for pe in pe_cfgs:
        r = evaluate_inference(
            ILLUSTRATIVE_27B, pe, DEFAULT_SRAM, LPDDR_PRESET, cfg
        )
        results.append(r)
        if verbose:
            print(_fmt_row(r))

    if verbose:
        print("-" * 140)
        print("SKU presets @ illustrative_27B × HBM/LPDDR (see also scan-sku):")
    for sku in (SKU_BASELINE, SKU_100T, SKU_1P):
        for mem in mems:
            r = evaluate_inference(
                ILLUSTRATIVE_27B, sku.npu(), DEFAULT_SRAM, mem,
                EvalConfig(
                    prompt_len=cfg.prompt_len,
                    decode_seq_len=cfg.decode_seq_len,
                    batch=cfg.batch,
                    frequency_hz=sku.frequency_hz,
                    pin_weights=cfg.pin_weights,
                    contention_mode=cfg.contention_mode,
                ),
            )
            results.append(r)
            if verbose:
                print(_fmt_row(r))

    if verbose:
        print("-" * 140)
        print("Detail dump (default SRAM+PE, HBM only, each shape once):")
        seen: set[tuple[str, str]] = set()
        for r in results:
            key = (r.shape.name, r.mem.kind)
            if (
                r.mem.kind == "HBM"
                and r.sram.capacity_bytes == DEFAULT_SRAM.capacity_bytes
                and r.npu.rows == DEFAULT_NPU.rows
                and r.npu.n_engines == DEFAULT_NPU.n_engines
                and key not in seen
            ):
                seen.add(key)
                for line in r.summary_lines():
                    print(line)
                print()
    return results


def run_sku_scan(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    skus: list[SKUPreset] | None = None,
) -> list[InferenceResult]:
    """Scan 100T-class (and 1P) SKUs; highlight decode mem-wall crossover on LPDDR."""
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=1.0e9,
        pin_weights=False,
        contention_mode="serialize",
    )
    skus = skus or [SKU_BASELINE, SKU_100T, SKU_1P]
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []

    if verbose:
        print("npu_dse SKU scan — single-card peak templates")
        print("Honesty: process_nm is a label; TOPS from PE×engines×freq×2flop/MAC.")
        print("1P is a bigger single-die peak for sensitivity (multi-chip stub later).")
        for s in skus:
            print(f"  {s.summary()}")
        print(f"Eval: prompt={base.prompt_len} ctx={base.decode_seq_len} "
              f"contention={base.contention_mode} SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB")
        print("-" * 140)

    for sku in skus:
        for mem in mems:
            ecfg = EvalConfig(
                prompt_len=base.prompt_len,
                decode_seq_len=base.decode_seq_len,
                batch=base.batch,
                frequency_hz=sku.frequency_hz,
                pin_weights=base.pin_weights,
                contention_mode=base.contention_mode,
            )
            r = evaluate_inference(
                ILLUSTRATIVE_27B, sku.npu(), DEFAULT_SRAM, mem, ecfg
            )
            results.append(r)
            if verbose:
                print(_fmt_row(r))

    if verbose:
        print("-" * 140)
        print("Bandwidth-wall crossover (illustrative_27B decode @ LPDDR):")
        print("  baseline (~8T) often compute-bound (OS util≈1/R + tiny peak);")
        print("  sku_100t / sku_1p should flip TPOT to memory-bound (weight stream).")
        lpddr_rows = [r for r in results if r.mem.kind == "LPDDR"]
        for r in lpddr_rows:
            ratio = r.decode.t_memory_s / max(r.decode.t_compute_s, 1e-30)
            print(
                f"  {r.npu.sku_name:12s} wall={r.decode.wall:7s} "
                f"t_comp={r.decode.t_compute_s*1e3:.3f}ms "
                f"t_mem={r.decode.t_memory_s*1e3:.3f}ms "
                f"mem/comp={ratio:.2f}x "
                f"W_DRAM={r.decode.traffic.weight_bytes_dram/1e9:.2f}GB "
                f"path={r.decode.traffic.weight_path}"
            )
        # Detail one 100T LPDDR point
        for r in results:
            if r.npu.sku_name == SKU_100T.name and r.mem.kind == "LPDDR":
                print()
                for line in r.summary_lines():
                    print(line)
                break
    return results


def sweep_sram(
    shape: ModelShape,
    mem: ExternalMemory,
    npu: NPUConfig,
    sram_bytes_list: list[int],
    cfg: EvalConfig | None = None,
) -> list[InferenceResult]:
    cfg = cfg or EvalConfig()
    return [
        evaluate_inference(shape, npu, SRAMConfig(s), mem, cfg)
        for s in sram_bytes_list
    ]


def find_mem_bound_crossover(
    shape: ModelShape,
    mem: ExternalMemory,
    sram: SRAMConfig,
    engine_counts: list[int],
    rows: int = 64,
    cols: int = 64,
    cfg: EvalConfig | None = None,
) -> list[InferenceResult]:
    """Scale n_engines at fixed lane geometry; return results for crossover plots."""
    cfg = cfg or EvalConfig(
        prompt_len=512, decode_seq_len=512, frequency_hz=1e9,
        contention_mode="serialize",
    )
    out: list[InferenceResult] = []
    for ne in engine_counts:
        npu = NPUConfig(rows=rows, cols=cols, n_engines=ne, sku_name=f"x{ne}eng")
        out.append(evaluate_inference(shape, npu, sram, mem, cfg))
    return out


# ---------------------------------------------------------------------------
# v0.3 sweeps: dtype (bytes-only) and context length
# ---------------------------------------------------------------------------

DEFAULT_CTX_SWEEP: tuple[int, ...] = (512, 2048, 8192, 32768, 128000)


def _fmt_dtype_row(r: InferenceResult, dtype_name: str) -> str:
    w = r.decode.traffic.weight_bytes_dram
    kv = r.decode.traffic.kv_bytes_dram
    kv_r = r.decode.traffic.kv_bytes_read
    ratio = kv / max(w, 1)
    if ratio >= 1.0:
        rival = "KV>W"
    elif ratio >= 0.5:
        rival = "KV~W"
    elif ratio >= 0.25:
        rival = "KV^  "
    else:
        rival = "W>>KV"
    return (
        f"{dtype_name:8s} {r.mem.kind:6s} "
        f"Wbits={r.shape.weight_bits:<3d} KVbits={r.shape.kv_bits:<3d} "
        f"W={w/1e9:7.3f}GB KV={kv/1e9:7.4f}GB (r={kv_r/1e9:.4f}) "
        f"KV/W={ratio:5.2f} [{rival:5s}] "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f}) "
        f"Wpath={r.decode.traffic.weight_path}"
    )


def _fmt_ctx_row(r: InferenceResult) -> str:
    w = r.decode.traffic.weight_bytes_dram
    kv = r.decode.traffic.kv_bytes_dram
    kv_r = r.decode.traffic.kv_bytes_read
    kv_w = r.decode.traffic.kv_bytes_write
    ratio = kv / max(w, 1)
    if ratio >= 1.0:
        tag = "KV>W"
    elif ratio >= 0.5:
        tag = "KV~W"
    elif ratio >= 0.25:
        tag = "KV^  "
    else:
        tag = "W>>KV"
    parts = r.decode.traffic.partitions
    kv_hit = (
        parts is not None
        and parts.kv_scratch_bytes >= r.shape.kv_cache_bytes(r.cfg.decode_seq_len, r.cfg.batch)
        and r.shape.kv_cache_bytes(r.cfg.decode_seq_len, r.cfg.batch) > 0
    )
    return (
        f"ctx={r.cfg.decode_seq_len:7d} {r.mem.kind:6s} "
        f"W={w/1e9:7.3f}GB KV={kv/1e9:8.4f}GB "
        f"(r={kv_r/1e9:.4f}/w={kv_w/1e6:.3f}MB) "
        f"KV/W={ratio:5.2f} [{tag:5s}] "
        f"kv_hit={str(kv_hit):5s} "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f})"
    )


def run_dtype_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
) -> list[InferenceResult]:
    """Sweep weight/KV storage bits on sku_100t @ HBM and LPDDR.

    Bytes-only: MAC peak / FLOPs unchanged (conservative for low-precision).
    """
    from .dtype import DTYPE_SWEEP_DEFAULT

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    ecfg = EvalConfig(
        prompt_len=base.prompt_len,
        decode_seq_len=base.decode_seq_len,
        batch=base.batch,
        frequency_hz=sku.frequency_hz,
        pin_weights=base.pin_weights,
        contention_mode=base.contention_mode,
    )
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []

    if verbose:
        print("npu_dse dtype sweep — storage/traffic bits only (MAC rate unchanged)")
        print(
            f"SKU={sku.name} ({sku.achieved_tops():.1f} TOPS) shape={shape.name} "
            f"ctx={ecfg.decode_seq_len} SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB"
        )
        print(
            "Honesty: formats are storage/traffic bits; FLOPs & peak MAC rate fixed "
            "(conservative vs real low-precision MAC arrays)."
        )
        for p in DTYPE_SWEEP_DEFAULT:
            print(f"  {p.summary()}")
        print("-" * 140)

    for preset in DTYPE_SWEEP_DEFAULT:
        shaped = preset.apply(shape)  # weight + KV bits; act stays
        for mem in mems:
            r = evaluate_inference(
                shaped, sku.npu(), DEFAULT_SRAM, mem, ecfg
            )
            results.append(r)
            if verbose:
                print(_fmt_dtype_row(r, preset.name))

    if verbose:
        print("-" * 140)
        print("Decode TPOT vs dtype (LPDDR, memory-wall expected for sku_100t):")
        for r in results:
            if r.mem.kind != "LPDDR":
                continue
            bits = r.shape.weight_bits
            print(
                f"  {bits:2d}-bit  TPOT={r.tpot_s*1e3:.4f}ms  "
                f"W_DRAM={r.decode.traffic.weight_bytes_dram/1e9:.3f}GB  "
                f"t_mem={r.decode.t_memory_s*1e3:.3f}ms  "
                f"t_comp={r.decode.t_compute_s*1e3:.3f}ms  "
                f"wall={r.decode.wall}"
            )
        # Sanity: lower bits → fewer weight bytes → lower or equal mem time
        lpddr = [r for r in results if r.mem.kind == "LPDDR"]
        if len(lpddr) >= 2:
            w0 = lpddr[0].decode.traffic.weight_bytes_dram
            w1 = lpddr[-1].decode.traffic.weight_bytes_dram
            print(
                f"  check: weight_bytes {lpddr[0].shape.weight_bits}b→"
                f"{lpddr[-1].shape.weight_bits}b : {w0/1e9:.3f}→{w1/1e9:.3f} GB "
                f"(scale≈{w1/max(w0,1):.3f}, expect ~"
                f"{lpddr[-1].shape.weight_bits/lpddr[0].shape.weight_bits:.3f})"
            )
    return results


def run_ctx_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    ctx_list: list[int] | None = None,
) -> list[InferenceResult]:
    """Sweep decode context length; show when KV traffic rivals weight stream."""
    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    ctxs = list(ctx_list) if ctx_list is not None else list(DEFAULT_CTX_SWEEP)
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []

    if verbose:
        print("npu_dse context-length sweep — when KV rivals weight stream")
        print(
            f"SKU={sku.name} shape={shape.name} "
            f"W={shape.weight_bits}b KV={shape.kv_bits}b "
            f"SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB "
            f"contention={base.contention_mode}"
        )
        print(
            f"  body weights≈{shape.weight_bytes()/1e9:.3f}GB  "
            f"kv/token={shape.kv_bytes_per_token()} B"
        )
        print("-" * 140)

    for ctx in ctxs:
        if ctx < 0:
            raise ValueError(f"ctx must be non-negative, got {ctx}")
        # Prefill prompt capped at ctx for consistency (or keep base prompt)
        prompt = min(base.prompt_len, ctx) if ctx > 0 else base.prompt_len
        ecfg = EvalConfig(
            prompt_len=max(prompt, 1),
            decode_seq_len=ctx,
            batch=base.batch,
            frequency_hz=sku.frequency_hz,
            pin_weights=base.pin_weights,
            contention_mode=base.contention_mode,
        )
        for mem in mems:
            r = evaluate_inference(
                shape, sku.npu(), DEFAULT_SRAM, mem, ecfg
            )
            results.append(r)
            if verbose:
                print(_fmt_ctx_row(r))

    if verbose:
        print("-" * 140)
        print("KV vs weight (decode @ HBM; same bytes on LPDDR, different wall time):")
        hbm_rows = [r for r in results if r.mem.kind == "HBM"]
        first_25 = first_50 = None
        for r in hbm_rows:
            w = r.decode.traffic.weight_bytes_dram
            kv = r.decode.traffic.kv_bytes_dram
            ratio = kv / max(w, 1)
            mark = ""
            if ratio >= 0.25 and first_25 is None:
                first_25 = r.cfg.decode_seq_len
                mark = "  ← KV ≥25% of weight stream"
            if ratio >= 0.5 and first_50 is None:
                first_50 = r.cfg.decode_seq_len
                mark = "  ← KV ≥50% of weight stream (rivals)"
            print(
                f"  ctx={r.cfg.decode_seq_len:7d}  "
                f"W={w/1e9:.3f}GB  KV={kv/1e9:.4f}GB  "
                f"KV/W={ratio:.3f}{mark}"
            )
        if first_25 is not None:
            print(f"  first ctx with KV/W≥0.25: {first_25}")
        if first_50 is not None:
            print(f"  first ctx with KV/W≥0.5 (rivals): {first_50}")
        else:
            print(
                "  (KV stays <50% of weight stream over this fp16 sweep; "
                "weight-only quant would make KV dominate sooner)"
            )
        # Monotonic KV bytes check printed for honesty
        kv_seq = [r.decode.traffic.kv_bytes_dram for r in hbm_rows]
        mono = all(a <= b for a, b in zip(kv_seq, kv_seq[1:]))
        print(f"  KV DRAM bytes monotone non-decreasing with ctx: {mono}")
    return results


# ---------------------------------------------------------------------------
# v0.4 sweeps: SRAM layer-fit knee + independent weight/KV quant
# ---------------------------------------------------------------------------

DEFAULT_SRAM_SWEEP_MIB: tuple[float, ...] = (
    # Below R=1, around R=1, then R≈L/2 and R=L for illustrative_27B @ fp16
    # (W_layer=1320 MiB, L=40 → R1≈1320, R20≈26400, R40≈52800 MiB)
    64, 256, 512, 1024, 1320, 2048, 4096, 13200, 26400, 52800,
)


def _fmt_sram_row(r: InferenceResult) -> str:
    w = r.decode.traffic.weight_bytes_dram
    kv = r.decode.traffic.kv_bytes_dram
    parts = r.decode.traffic.partitions
    w_stage = parts.weight_staging_bytes if parts else 0
    w_layer = r.decode.traffic.layer_weight_bytes
    R = r.decode.traffic.resident_layers
    L = r.shape.n_layers
    ratio = kv / max(w, 1) if w > 0 else (float("inf") if kv > 0 else 0.0)
    if w == 0 and kv > 0:
        riv = "KV>W"
    elif ratio >= 1.0:
        riv = "KV>W"
    elif ratio >= 0.5:
        riv = "KV~W"
    else:
        riv = "W>>KV"
    mib = r.sram.capacity_bytes / 2**20
    return (
        f"SRAM={mib:8.1f}MiB {r.mem.kind:6s} "
        f"W_layer={w_layer/2**20:7.1f}MiB W_part={w_stage/2**20:8.1f}MiB "
        f"R={R:3d}/{L:<3d} "
        f"path={r.decode.traffic.weight_path:24s} "
        f"dbl={str(r.decode.traffic.double_buffer_eligible):5s} "
        f"W={w/1e9:7.3f}GB KV={kv/1e9:7.4f}GB [{riv:5s}] "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f}) "
        f"hide={r.cfg.weight_hide_factor:.1f} pol={r.cfg.sram_policy}"
    )


def run_sram_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    sram_mib_list: list[float] | None = None,
    weight_hide_factor: float = 0.0,
    sram_policy: str = "weight_resident",
) -> list[InferenceResult]:
    """Sweep SRAM capacity; show resident knees (R≥1, R≥L/2, R≥L).

    Honesty (v0.5): staging-only (R=0) does NOT reduce decode W DRAM bytes —
    still L*W_layer per token. Byte drops happen at resident knees:
      R≥1     → W DRAM = (L-1)*W_layer
      R≥L/2   → W DRAM ≈ L/2 * W_layer
      R≥L     → W DRAM = 0 (on_die_resident, steady-state TPOT)

    For illustrative_27B @ fp16: W_layer≈1320 MiB, L=40 →
      R1≈1320 MiB, R20≈26400 MiB, R40≈52800 MiB (full body on-die).

    Optional weight_hide_factor (assumed) reduces weight *mem time* only when
    staging dbl-buf eligible (R=0 and staging≥2×W) — not a byte win.
    """
    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    mibs = list(sram_mib_list) if sram_mib_list is not None else list(DEFAULT_SRAM_SWEEP_MIB)
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
        weight_hide_factor=weight_hide_factor,
        sram_policy=sram_policy,  # type: ignore[arg-type]
    )
    ecfg = EvalConfig(
        prompt_len=base.prompt_len,
        decode_seq_len=base.decode_seq_len,
        batch=base.batch,
        frequency_hz=sku.frequency_hz,
        pin_weights=base.pin_weights,
        contention_mode=base.contention_mode,
        weight_hide_factor=weight_hide_factor,
        sram_policy=sram_policy,  # type: ignore[arg-type]
    )
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []
    w_layer = shape.weight_bytes_per_layer()
    L = shape.n_layers
    knee_r1 = w_layer / 2**20
    knee_r_half = (L // 2) * w_layer / 2**20 if L > 0 else 0
    knee_r_full = L * w_layer / 2**20 if L > 0 else 0

    if verbose:
        print("npu_dse SRAM sweep — resident knees (R≥1 / R≥L/2 / R≥L)")
        print(
            f"SKU={sku.name} shape={shape.name} "
            f"W={shape.weight_bits}b KV={shape.kv_bits}b "
            f"ctx={ecfg.decode_seq_len} hide={weight_hide_factor:.1f}[assumed] "
            f"policy={sram_policy}"
        )
        print(
            f"  W_layer={w_layer/2**20:.1f} MiB  L={L}  "
            f"knee_R≥1 ≈ {knee_r1:.0f} MiB  "
            f"knee_R≥L/2 ≈ {knee_r_half:.0f} MiB  "
            f"knee_R≥L (full on-die) ≈ {knee_r_full:.0f} MiB"
        )
        print(
            "  Honesty: staging (R=0) keeps W DRAM = L*W_layer; only R≥1 drops bytes. "
            "dbl-buf hide (if >0) cuts weight mem time only, not bytes."
        )
        print("-" * 160)

    first_r1: dict[str, float] = {}
    first_r_half: dict[str, float] = {}
    first_r_full: dict[str, float] = {}
    first_dbl: dict[str, float] = {}
    for mib in mibs:
        if mib < 0:
            raise ValueError(f"SRAM MiB must be non-negative, got {mib}")
        sram = SRAMConfig(int(mib * 1024 * 1024), policy=sram_policy)  # type: ignore[arg-type]
        for mem in mems:
            r = evaluate_inference(shape, sku.npu(), sram, mem, ecfg)
            results.append(r)
            if verbose:
                print(_fmt_sram_row(r))
            R = r.decode.traffic.resident_layers
            if R >= 1 and mem.kind not in first_r1:
                first_r1[mem.kind] = mib
            if L > 0 and R >= L // 2 and mem.kind not in first_r_half:
                first_r_half[mem.kind] = mib
            if L > 0 and R >= L and mem.kind not in first_r_full:
                first_r_full[mem.kind] = mib
            if r.decode.traffic.double_buffer_eligible and mem.kind not in first_dbl:
                first_dbl[mem.kind] = mib

    if verbose:
        print("-" * 160)
        for kind in ("HBM", "LPDDR"):
            rows = [r for r in results if r.mem.kind == kind]
            if not rows:
                continue
            print(
                f"  {kind}: first R≥1 @ {first_r1.get(kind, 'n/a')} MiB; "
                f"R≥L/2 @ {first_r_half.get(kind, 'n/a')} MiB; "
                f"R≥L @ {first_r_full.get(kind, 'n/a')} MiB; "
                f"dbl_buf @ {first_dbl.get(kind, 'n/a')} MiB"
            )
            for r in rows:
                R = r.decode.traffic.resident_layers
                print(
                    f"    SRAM={r.sram.capacity_bytes/2**20:8.1f}MiB "
                    f"R={R:3d}/{L:<3d} "
                    f"path={r.decode.traffic.weight_path:24s} "
                    f"W={r.decode.traffic.weight_bytes_dram/1e9:7.3f}GB "
                    f"TPOT={r.tpot_s*1e3:.4f}ms "
                    f"t_mem={r.decode.t_memory_s*1e3:.3f}ms "
                    f"Wfrac={r.decode.contention.t_weight_s/max(r.decode.contention.t_serialize_s,1e-30):.1%} "
                    f"KVfrac={r.decode.contention.t_kv_s/max(r.decode.contention.t_serialize_s,1e-30):.1%}"
                )
    return results


def _fmt_quant_row(r: InferenceResult, quant_name: str) -> str:
    w = r.decode.traffic.weight_bytes_dram
    kv = r.decode.traffic.kv_bytes_dram
    ratio = kv / max(w, 1)
    if ratio >= 1.0:
        riv = "KV>W"
    elif ratio >= 0.5:
        riv = "KV~W"
    elif ratio >= 0.25:
        riv = "KV^  "
    else:
        riv = "W>>KV"
    return (
        f"{quant_name:8s} ctx={r.cfg.decode_seq_len:7d} {r.mem.kind:6s} "
        f"Wbits={r.shape.weight_bits:<3d} KVbits={r.shape.kv_bits:<3d} "
        f"W={w/1e9:7.3f}GB KV={kv/1e9:8.4f}GB KV/W={ratio:6.3f} [{riv:5s}] "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f}) "
        f"Wpath={r.decode.traffic.weight_path}"
    )


def run_quant_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    ctx_list: list[int] | None = None,
    quant_names: list[str] | None = None,
) -> list[InferenceResult]:
    """Sweep independent weight_bits/kv_bits; show when KV overtakes weight stream.

    Default presets: w16k16, w8k16, w4k16, w8k8, w4k8 @ ctx in {512, 128000}.
    """
    from .dtype import QUANT_REGISTRY, QUANT_SWEEP_DEFAULT, get_quant

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    ctxs = list(ctx_list) if ctx_list is not None else [512, 128000]
    presets = (
        [get_quant(n) for n in quant_names]
        if quant_names is not None
        else list(QUANT_SWEEP_DEFAULT)
    )
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []

    if verbose:
        print("npu_dse quant sweep — weight-only vs full quant (bytes-only)")
        print(
            f"SKU={sku.name} base_shape={shape.name} "
            f"SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB "
            f"contention={base.contention_mode}"
        )
        print("Honesty: MAC rate / FLOPs unchanged; only storage/traffic bytes scale.")
        for q in presets:
            print(f"  {q.summary()}")
        print("-" * 150)

    for q in presets:
        shaped = q.apply(shape)
        for ctx in ctxs:
            prompt = min(base.prompt_len, ctx) if ctx > 0 else base.prompt_len
            ecfg = EvalConfig(
                prompt_len=max(prompt, 1),
                decode_seq_len=ctx,
                batch=base.batch,
                frequency_hz=sku.frequency_hz,
                pin_weights=base.pin_weights,
                contention_mode=base.contention_mode,
                weight_hide_factor=base.weight_hide_factor,
            )
            for mem in mems:
                r = evaluate_inference(
                    shaped, sku.npu(), DEFAULT_SRAM, mem, ecfg
                )
                results.append(r)
                if verbose:
                    print(_fmt_quant_row(r, q.name))

    if verbose:
        print("-" * 150)
        print("KV overtakes weight (KV/W ≥ 1) — HBM rows:")
        for r in results:
            if r.mem.kind != "HBM":
                continue
            w = r.decode.traffic.weight_bytes_dram
            kv = r.decode.traffic.kv_bytes_dram
            ratio = kv / max(w, 1)
            mark = "  ← KV overtakes W" if ratio >= 1.0 else ""
            qname = r.shape.name.split("@")[-1] if "@" in r.shape.name else "?"
            print(
                f"  {qname:8s} ctx={r.cfg.decode_seq_len:7d} "
                f"W={w/1e9:.3f}GB KV={kv/1e9:.4f}GB KV/W={ratio:.3f}{mark}"
            )
        # Highlight weight-only vs full at long ctx
        print("Weight-only vs full at ctx=128000 (HBM):")
        long_hbm = [
            r for r in results
            if r.mem.kind == "HBM" and r.cfg.decode_seq_len == 128000
        ]
        for r in long_hbm:
            qname = r.shape.name.split("@")[-1]
            ratio = r.decode.traffic.kv_bytes_dram / max(
                r.decode.traffic.weight_bytes_dram, 1
            )
            print(
                f"  {qname:8s} W={r.shape.weight_bits}b KV={r.shape.kv_bits}b "
                f"KV/W={ratio:.3f} TPOT={r.tpot_s*1e3:.4f}ms"
            )
    return results


# ---------------------------------------------------------------------------
# v0.6 sweeps: batch (OS util vs M) + MoE active-W vs dense
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def _fmt_batch_row(r: InferenceResult) -> str:
    b = r.cfg.batch
    return (
        f"batch={b:3d} M={r.decode.traffic.m_eff:3d} {r.mem.kind:6s} "
        f"util={r.decode.traffic.mean_gemm_util:.4f} "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f}) "
        f"W={r.decode.traffic.weight_bytes_dram/1e9:7.3f}GB "
        f"KV={r.decode.traffic.kv_bytes_dram/1e9:7.4f}GB "
        f"FLOPs={r.decode.traffic.flops/1e9:.2f}G"
    )


def run_batch_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    batch_list: list[int] | None = None,
) -> list[InferenceResult]:
    """Sweep decode batch (=M); OS util rises with M; wall may flip.

    At sku_100t decode, M=1 util≈1/R; larger batch fills rows → higher util,
    lower t_compute per token-group wall, while weight bytes stay ~L*W_layer
    (decode streams weights once per step regardless of batch — same W DRAM;
    KV scales with batch).
    """
    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    batches = list(batch_list) if batch_list is not None else list(DEFAULT_BATCH_SWEEP)
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[InferenceResult] = []

    if verbose:
        print("npu_dse batch sweep — OS util vs M=batch @ decode")
        print(
            f"SKU={sku.name} shape={shape.name} "
            f"ctx={base.decode_seq_len} "
            f"SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB "
            f"PE={sku.rows}x{sku.cols}x{sku.n_engines} (R={sku.rows})"
        )
        print(
            f"  OS: util(M=1)≈1/R={1/sku.rows:.4f} when N%effC==0; "
            f"util rises toward 1 as M fills rows."
        )
        print("-" * 140)

    for b in batches:
        if b < 1:
            raise ValueError(f"batch must be >= 1, got {b}")
        ecfg = EvalConfig(
            prompt_len=base.prompt_len,
            decode_seq_len=base.decode_seq_len,
            batch=b,
            frequency_hz=sku.frequency_hz,
            pin_weights=base.pin_weights,
            contention_mode=base.contention_mode,
            weight_hide_factor=base.weight_hide_factor,
            sram_policy=base.sram_policy,
        )
        for mem in mems:
            r = evaluate_inference(
                shape, sku.npu(), DEFAULT_SRAM, mem, ecfg
            )
            results.append(r)
            if verbose:
                print(_fmt_batch_row(r))

    if verbose:
        print("-" * 140)
        print("Util / wall vs batch (LPDDR):")
        prev_u = -1.0
        for r in results:
            if r.mem.kind != "LPDDR":
                continue
            u = r.decode.traffic.mean_gemm_util
            mono = "↑" if u > prev_u + 1e-15 else ("=" if abs(u - prev_u) < 1e-15 else "↓")
            print(
                f"  batch={r.cfg.batch:3d} util={u:.4f} {mono} "
                f"wall={r.decode.wall:7s} "
                f"t_comp={r.decode.t_compute_s*1e3:.3f}ms "
                f"t_mem={r.decode.t_memory_s*1e3:.3f}ms "
                f"TPOT={r.tpot_s*1e3:.4f}ms"
            )
            prev_u = u
        lpddr = [r for r in results if r.mem.kind == "LPDDR"]
        if len(lpddr) >= 2:
            assert_mono = all(
                lpddr[i].decode.traffic.mean_gemm_util
                <= lpddr[i + 1].decode.traffic.mean_gemm_util + 1e-15
                for i in range(len(lpddr) - 1)
            )
            print(f"  util non-decreasing with batch: {assert_mono}")
    return results


def _fmt_moe_row(r: InferenceResult, tag: str) -> str:
    s = r.shape
    return (
        f"{tag:22s} {r.mem.kind:6s} "
        f"total≈{s.total_weight_params()/1e9:5.1f}B "
        f"active≈{s.active_weight_params()/1e9:5.1f}B "
        f"stream_W={r.decode.traffic.weight_bytes_dram/1e9:7.3f}GB "
        f"KV={r.decode.traffic.kv_bytes_dram/1e9:7.4f}GB "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f}) "
        f"util={r.decode.traffic.mean_gemm_util:.4f}"
    )


DEFAULT_EP_SWEEP = (1, 2, 4, 8)


def run_moe_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    ep_list: list[int] | None = None,
    c2c_gbps: float = 400.0,
) -> list:
    """Compare illustrative_moe active W vs dense; optional EP scale-up.

    When ``ep_list`` is None or ``[1]`` only: single-card InferenceResult rows
    (legacy). When ep>1 values are included: also emit ScaleupResult rows for
    MoE × ep on HBM/LPDDR with Megatron-MoE A2A estimate.
    """
    from .scaleup import C2CLink, TPConfig, evaluate_scaleup, get_c2c

    sku = sku or SKU_100T
    ep_list = list(ep_list) if ep_list is not None else [1]
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    ecfg = EvalConfig(
        prompt_len=base.prompt_len,
        decode_seq_len=base.decode_seq_len,
        batch=base.batch,
        frequency_hz=sku.frequency_hz,
        pin_weights=base.pin_weights,
        contention_mode=base.contention_mode,
    )
    shapes = [
        ("dense_27B", ILLUSTRATIVE_27B),
        ("moe_E8k2", ILLUSTRATIVE_MOE),
    ]
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list = []

    if verbose:
        print("npu_dse MoE sweep — active W stream vs dense 27B (+ optional EP)")
        print(
            f"SKU={sku.name} ctx={ecfg.decode_seq_len} "
            f"SRAM={DEFAULT_SRAM.capacity_bytes/2**20:.0f}MiB ep_list={ep_list}"
        )
        print(
            "Honesty: MoE traffic = attn + shared + top_k FFN (balanced); "
            "EP shards E/ep stored experts; A2A = "
            "2*(ep-1)/ep*B*S*H*act_bytes*(top_k/E_local_adjust), "
            "E_local_adjust=1 default."
        )
        for tag, sh in shapes:
            print(f"  {tag}: {sh.summary()}")
            print(
                f"    stored_W/layer={sh.weight_bytes_per_layer()/2**20:.1f}MiB "
                f"stream_W/layer={sh.stream_weight_bytes_per_layer()/2**20:.1f}MiB "
                f"ratio_stream/stored="
                f"{sh.stream_weight_bytes_per_layer()/max(sh.weight_bytes_per_layer(),1):.3f}"
            )
        print("-" * 150)

    # Single-card baseline (ep=1 path)
    for tag, sh in shapes:
        for mem in mems:
            r = evaluate_inference(sh, sku.npu(), DEFAULT_SRAM, mem, ecfg)
            results.append(r)
            if verbose:
                print(_fmt_moe_row(r, tag))

    # EP scale-up rows for MoE
    need_ep = [e for e in ep_list if e > 1]
    if need_ep:
        try:
            c2c = get_c2c(c2c_gbps)
        except KeyError:
            c2c = C2CLink(f"c2c_{int(c2c_gbps)}", float(c2c_gbps))
        if verbose:
            print("-" * 150)
            print(
                f"MoE EP scale-up (C2C={c2c.summary()}; "
                f"A2A on C2C BW; cards=tp*pp*ep with tp=pp=1):"
            )
        for ep in ep_list:
            if ILLUSTRATIVE_MOE.n_experts % ep != 0:
                if verbose:
                    print(f"  skip ep={ep}: E not divisible")
                continue
            tp_cfg = TPConfig(tp=1, pp=1, ep=ep, c2c=c2c, kv_fabric="none")
            for mem in mems:
                r = evaluate_scaleup(
                    ILLUSTRATIVE_MOE, sku.npu(), DEFAULT_SRAM, mem, ecfg, tp_cfg
                )
                results.append(r)
                if verbose:
                    print(
                        f"moe_ep={ep:<2d} cards={r.tp_cfg.n_cards:<3d} "
                        f"{r.mem.kind:6s} "
                        f"Wcard={r.decode.weight_bytes_card/1e9:7.3f}GB "
                        f"a2a={r.decode.a2a_bytes/1e6:8.3f}MB "
                        f"t_a2a={r.decode.t_a2a_s*1e3:8.4f}ms "
                        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:8s}] "
                        f"(c={r.decode.t_compute_s*1e3:.3f}/"
                        f"d={r.decode.t_memory_s*1e3:.3f}/"
                        f"a2a={r.decode.t_a2a_s*1e3:.3f})"
                    )

    if verbose:
        print("-" * 150)
        print("Active stream W vs dense (LPDDR decode, single-card):")
        dense_w = moe_w = None
        for r in results:
            if not isinstance(r, InferenceResult):
                continue
            if r.mem.kind != "LPDDR":
                continue
            w = r.decode.traffic.weight_bytes_dram
            if r.shape.name.startswith("illustrative_27B"):
                dense_w = w
                label = "dense_27B"
            elif r.shape.is_moe:
                moe_w = w
                label = "moe_E8k2"
            else:
                continue
            print(
                f"  {label:12s} stream_W={w/1e9:.3f}GB "
                f"TPOT={r.tpot_s*1e3:.4f}ms wall={r.decode.wall} "
                f"t_mem={r.decode.t_memory_s*1e3:.3f}ms"
            )
        if dense_w and moe_w:
            print(
                f"  MoE/dense stream_W ratio ≈ {moe_w/dense_w:.3f} "
                f"(MoE activates top_k={ILLUSTRATIVE_MOE.top_k}/"
                f"E={ILLUSTRATIVE_MOE.n_experts} FFN)"
            )
    return results


# ---------------------------------------------------------------------------
# Scale-up: TP × mem × C2C (v0.7)
# ---------------------------------------------------------------------------

DEFAULT_TP_SWEEP = (1, 2, 4, 8)


def _fmt_tp_row(r: "ScaleupResult") -> str:  # noqa: F821 — runtime import
    tp = r.tp_cfg.tp
    pp = r.tp_cfg.pp
    oom = "OOM" if r.oom else "ok"
    fab = r.tp_cfg.kv_fabric
    return (
        f"tp={tp:<2d} pp={pp:<2d} {r.mem.kind:6s} "
        f"C2C={r.tp_cfg.c2c.effective_bw_GBps:4.0f}GB/s "
        f"fab={fab:7s} "
        f"TTFT={r.ttft_s*1e3:10.3f}ms [{r.prefill.wall:8s}] "
        f"(c={r.prefill.t_compute_s*1e3:.2f}/d={r.prefill.t_memory_s*1e3:.2f}/"
        f"c2c={r.prefill.t_c2c_s*1e3:.2f}/pp={r.prefill.t_pp_act_s*1e3:.2f}/"
        f"fab={r.prefill.t_fabric_s*1e3:.2f}) "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:8s}] "
        f"(c={r.decode.t_compute_s*1e3:.2f}/d={r.decode.t_memory_s*1e3:.2f}/"
        f"c2c={r.decode.t_c2c_s*1e3:.2f}/pp={r.decode.t_pp_act_s*1e3:.2f}/"
        f"fab={r.decode.t_fabric_s*1e3:.2f}) "
        f"coll={r.decode.collective_bytes/1e6:8.2f}MB "
        f"Wcard={r.decode.weight_bytes_card/1e9:.2f}GB "
        f"cap={oom}"
    )


def run_tp_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    tp_list: list[int] | None = None,
    c2c_gbps: float = 400.0,
    c2c_hide: float = 0.0,
    embed_policy: str = "replicate",
) -> list:
    """Sweep tp∈{1,2,4,8} × {HBM,LPDDR} @ sku_100t for dense 27B scale-up.

    Reports TTFT/TPOT with max(compute, DRAM, C2C) breakdown and OOM flags.
    """
    from .scaleup import (
        C2CLink,
        ScaleupResult,
        TPConfig,
        evaluate_scaleup,
        get_c2c,
    )

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    tp_list = list(tp_list) if tp_list is not None else list(DEFAULT_TP_SWEEP)
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    ecfg = EvalConfig(
        prompt_len=base.prompt_len,
        decode_seq_len=base.decode_seq_len,
        batch=base.batch,
        frequency_hz=sku.frequency_hz,
        pin_weights=base.pin_weights,
        contention_mode=base.contention_mode,
        weight_hide_factor=base.weight_hide_factor,
        sram_policy=base.sram_policy,
    )
    try:
        c2c = get_c2c(c2c_gbps)
    except KeyError:
        c2c = C2CLink(f"c2c_{int(c2c_gbps)}", float(c2c_gbps))

    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[ScaleupResult] = []

    if verbose:
        print("npu_dse TP / scale-up sweep (assumed C2C BW; kv_fabric=none)")
        print(
            f"shape={shape.name} SKU={sku.name} "
            f"prompt={ecfg.prompt_len} ctx={ecfg.decode_seq_len} "
            f"C2C={c2c.summary()} hide={c2c_hide:.2f} embed={embed_policy}"
        )
        print(
            "Sharding: col-parallel Q/K/V/gate/up; row-parallel O/down; "
            "2× ring all-reduce/layer; bytes=2*(tp-1)/tp*volume"
        )
        print("-" * 160)

    for tp in tp_list:
        tp_cfg = TPConfig(
            tp=tp,
            c2c=c2c,
            c2c_hide=c2c_hide,
            kv_fabric="none",
            embed_policy=embed_policy,  # type: ignore[arg-type]
        )
        for mem in mems:
            r = evaluate_scaleup(
                shape, sku.npu(), DEFAULT_SRAM, mem, ecfg, tp_cfg
            )
            results.append(r)
            if verbose:
                print(_fmt_tp_row(r))

    if verbose:
        print("-" * 160)
        print("Detail (HBM rows):")
        for r in results:
            if r.mem.kind == "HBM":
                for line in r.summary_lines():
                    print(line)
                print()
    return results


def run_scaleup_scan(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
) -> list:
    """Alias: scan-scaleup — same as run_tp_sweep with default C2C=400 GB/s."""
    return run_tp_sweep(verbose=verbose, cfg=cfg, sku=sku)


# ---------------------------------------------------------------------------
# v0.8: KV fabric sweep + PP / parallel sweep
# ---------------------------------------------------------------------------

DEFAULT_PP_SWEEP = (1, 2, 4, 8)
DEFAULT_KV_FABRIC_PRESETS = (
    "none",
    "roce_100g",
    "roce_200g",
    "roce_400g",
    "ib_100g",
    "ib_200g",
    "ib_400g",
)
DEFAULT_KV_CTX_SWEEP = (512, 8192, 32768, 128000)


def _fmt_kv_fabric_row(r: "ScaleupResult", preset: str) -> str:  # noqa: F821
    fab = r.tp_cfg.fabric_link()
    bw = fab.effective_bw_GBps if fab else 0.0
    lat = fab.latency_us if fab else 0.0
    return (
        f"{preset:10s} ctx={r.cfg.decode_seq_len:7d} {r.mem.kind:6s} "
        f"BW={bw:5.1f}GB/s lat={lat:4.1f}us "
        f"remote_KV={r.decode.remote_kv_bytes/1e9:8.4f}GB "
        f"t_fab={r.decode.t_fabric_s*1e3:10.4f}ms "
        f"t_xfer={r.t_kv_xfer_s*1e3:10.4f}ms "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:8s}] "
        f"(c={r.decode.t_compute_s*1e3:.2f}/d={r.decode.t_memory_s*1e3:.2f}/"
        f"fab={r.decode.t_fabric_s*1e3:.2f}) "
        f"TTFT={r.ttft_s*1e3:10.3f}ms"
    )


def run_kv_fabric_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    ctx_list: list[int] | None = None,
    fabric_presets: list[str] | None = None,
    remote_kv_frac: float = 1.0,
    pd_kv_xfer: bool = True,
    tp: int = 1,
    c2c_gbps: float = 400.0,
) -> list:
    """Sweep KV fabric presets × ctx @ sku_100t dense (disagg decode assumed).

    ``remote_kv_frac=1.0`` (default for this sweep): all KV reads from fabric.
    ``pd_kv_xfer=True``: report t_kv_xfer for full-KV prefill→decode transfer.
    ``none`` rows must match the no-fabric baseline (t_fabric=0, t_xfer=0).
    """
    from .scaleup import (
        C2CLink,
        KV_FABRIC_PRESETS,
        ScaleupResult,
        TPConfig,
        evaluate_scaleup,
        get_c2c,
        get_kv_fabric,
    )

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    ctxs = list(ctx_list) if ctx_list is not None else list(DEFAULT_KV_CTX_SWEEP)
    presets = (
        list(fabric_presets)
        if fabric_presets is not None
        else list(DEFAULT_KV_FABRIC_PRESETS)
    )
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    try:
        c2c = get_c2c(c2c_gbps)
    except KeyError:
        c2c = C2CLink(f"c2c_{int(c2c_gbps)}", float(c2c_gbps))

    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[ScaleupResult] = []

    if verbose:
        print("npu_dse KV fabric sweep — IB/RoCE assumed BW+latency (analytical MVP)")
        print(
            f"shape={shape.name} SKU={sku.name} tp={tp} "
            f"remote_kv_frac={remote_kv_frac} pd_kv_xfer={pd_kv_xfer} "
            f"C2C={c2c.summary()}"
        )
        print(
            "Honesty: Gbps→GB/s via /8 * 0.80 eff; fixed µs latency/message; "
            "t_fabric=lat+bytes/BW; wall=max(compute,local_dram,c2c,fabric)."
        )
        for p in presets:
            if p == "none":
                print("  none: no fabric (baseline)")
            else:
                link = get_kv_fabric(p)
                print(f"  {link.summary()}")
        print("-" * 170)

    for preset in presets:
        if preset == "none":
            mode = "none"
            gbps = 200.0
            frac = 0.0
            do_xfer = False
        else:
            link = get_kv_fabric(preset)
            assert link is not None
            mode = link.mode
            gbps = link.link_gbps
            frac = remote_kv_frac
            do_xfer = pd_kv_xfer
        for ctx in ctxs:
            prompt = min(base.prompt_len, ctx) if ctx > 0 else base.prompt_len
            ecfg = EvalConfig(
                prompt_len=max(prompt, 1),
                decode_seq_len=ctx,
                batch=base.batch,
                frequency_hz=sku.frequency_hz,
                pin_weights=base.pin_weights,
                contention_mode=base.contention_mode,
                weight_hide_factor=base.weight_hide_factor,
                sram_policy=base.sram_policy,
            )
            tp_cfg = TPConfig(
                tp=tp,
                pp=1,
                c2c=c2c,
                kv_fabric=mode,  # type: ignore[arg-type]
                fabric_gbps=gbps,
                remote_kv_frac=frac,
                pd_kv_xfer=do_xfer,
                add_xfer_to_ttft=False,
            )
            for mem in mems:
                r = evaluate_scaleup(
                    shape, sku.npu(), DEFAULT_SRAM, mem, ecfg, tp_cfg
                )
                results.append(r)
                if verbose:
                    print(_fmt_kv_fabric_row(r, preset))

    if verbose:
        print("-" * 170)
        print("128k HBM highlight (remote decode + PD xfer):")
        for r in results:
            if r.mem.kind == "HBM" and r.cfg.decode_seq_len == 128000:
                if r.tp_cfg.kv_fabric == "none":
                    preset = "none"
                else:
                    prefix = "roce" if r.tp_cfg.kv_fabric == "roce_v2" else "ib"
                    preset = f"{prefix}_{int(r.tp_cfg.fabric_gbps)}g"
                print(
                    f"  {preset:10s} t_fab={r.decode.t_fabric_s*1e3:.4f}ms "
                    f"t_xfer={r.t_kv_xfer_s*1e3:.4f}ms "
                    f"TPOT={r.tpot_s*1e3:.4f}ms wall={r.decode.wall} "
                    f"remote_KV={r.decode.remote_kv_bytes/1e9:.4f}GB"
                )
    return results


def _fmt_pp_row(r: "ScaleupResult") -> str:  # noqa: F821
    return (
        f"tp={r.tp_cfg.tp:<2d} pp={r.tp_cfg.pp:<2d} mb={r.tp_cfg.mb:<2d} "
        f"cards={r.tp_cfg.n_cards:<3d} {r.mem.kind:6s} "
        f"bubble={r.decode.bubble_frac:.3f} "
        f"TTFT={r.ttft_s*1e3:10.3f}ms [{r.prefill.wall:8s}] "
        f"(c={r.prefill.t_compute_s*1e3:.2f}/d={r.prefill.t_memory_s*1e3:.2f}/"
        f"c2c={r.prefill.t_c2c_s*1e3:.2f}/pp={r.prefill.t_pp_act_s*1e3:.2f}) "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:8s}] "
        f"(c={r.decode.t_compute_s*1e3:.2f}/d={r.decode.t_memory_s*1e3:.2f}/"
        f"c2c={r.decode.t_c2c_s*1e3:.2f}/pp={r.decode.t_pp_act_s*1e3:.2f}) "
        f"stage={r.decode.t_stage_s*1e3:.3f}ms "
        f"Wcard={r.decode.weight_bytes_card/1e9:.2f}GB"
    )


def run_parallel_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    shape: ModelShape | None = None,
    tp_list: list[int] | None = None,
    pp_list: list[int] | None = None,
    c2c_gbps: float = 400.0,
    c2c_hide: float = 0.0,
    embed_policy: str = "replicate",
    decode_mb: int = 1,
    prefill_mb: int | None = None,
) -> list:
    """Sweep tp × pp (total cards = tp*pp, DP=1) @ sku_100t dense.

    Decode uses ``decode_mb`` (default 1 → bubble=(pp-1)/pp).
    Prefill uses ``prefill_mb`` defaulting to max(pp, decode_mb) to cut bubbles.
    Note: evaluator takes one mb from TPConfig; we set mb per-eval to the
    phase-appropriate value by running decode-oriented config (mb=decode_mb)
    and reporting prefill bubble with a note when prefill_mb differs.

    Limitation: single ``mb`` on TPConfig applies to both phases' bubble
    formula via phase-aware ``bubble_fraction`` — for prefill we temporarily
    build a config with mb=prefill_mb when it differs.
    """
    from .scaleup import (
        C2CLink,
        ScaleupResult,
        TPConfig,
        evaluate_scaleup,
        get_c2c,
    )

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_27B
    tp_list = list(tp_list) if tp_list is not None else [1, 2, 4]
    pp_list = list(pp_list) if pp_list is not None else list(DEFAULT_PP_SWEEP)
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    ecfg = EvalConfig(
        prompt_len=base.prompt_len,
        decode_seq_len=base.decode_seq_len,
        batch=base.batch,
        frequency_hz=sku.frequency_hz,
        pin_weights=base.pin_weights,
        contention_mode=base.contention_mode,
        weight_hide_factor=base.weight_hide_factor,
        sram_policy=base.sram_policy,
    )
    try:
        c2c = get_c2c(c2c_gbps)
    except KeyError:
        c2c = C2CLink(f"c2c_{int(c2c_gbps)}", float(c2c_gbps))

    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list[ScaleupResult] = []

    if verbose:
        print("npu_dse parallel sweep — TP × PP (assumed C2C; bubble model crude)")
        print(
            f"shape={shape.name} SKU={sku.name} "
            f"prompt={ecfg.prompt_len} ctx={ecfg.decode_seq_len} "
            f"C2C={c2c.summary()} decode_mb={decode_mb} "
            f"prefill_mb={prefill_mb if prefill_mb is not None else 'max(pp,mb)'}"
        )
        print(
            "Limitation: decode mb=1 → bubble=(pp-1)/pp (often bubble-heavy); "
            "prefill mb≥pp reduces bubble to (pp-1)/(mb+pp-1)."
        )
        print("-" * 170)

    for tp in tp_list:
        for pp in pp_list:
            pref_mb = prefill_mb if prefill_mb is not None else max(pp, decode_mb)
            # Use decode_mb on the shared config; re-eval prefill bubble via
            # a second call only when we want accurate prefill mb — for MVP
            # stash decode-oriented result and overlay prefill from pref_mb run
            # when different.
            dec_cfg = TPConfig(
                tp=tp,
                pp=pp,
                mb=decode_mb,
                c2c=c2c,
                c2c_hide=c2c_hide,
                kv_fabric="none",
                embed_policy=embed_policy,  # type: ignore[arg-type]
            )
            for mem in mems:
                r_dec = evaluate_scaleup(
                    shape, sku.npu(), DEFAULT_SRAM, mem, ecfg, dec_cfg
                )
                if pref_mb != decode_mb:
                    pref_cfg = TPConfig(
                        tp=tp,
                        pp=pp,
                        mb=pref_mb,
                        c2c=c2c,
                        c2c_hide=c2c_hide,
                        kv_fabric="none",
                        embed_policy=embed_policy,  # type: ignore[arg-type]
                    )
                    r_pref = evaluate_scaleup(
                        shape, sku.npu(), DEFAULT_SRAM, mem, ecfg, pref_cfg
                    )
                    # Hybrid: decode wall from decode_mb, prefill from pref_mb
                    hybrid = ScaleupResult(
                        shape=r_dec.shape,
                        npu=r_dec.npu,
                        sram=r_dec.sram,
                        mem=r_dec.mem,
                        cfg=r_dec.cfg,
                        tp_cfg=dec_cfg,
                        prefill=r_pref.prefill,
                        decode=r_dec.decode,
                        oom=r_dec.oom,
                        per_card_bytes=r_dec.per_card_bytes,
                        capacity_detail=r_dec.capacity_detail,
                        single_card=r_dec.single_card,
                        t_kv_xfer_s=r_dec.t_kv_xfer_s,
                        kv_xfer_bytes=r_dec.kv_xfer_bytes,
                    )
                    r = hybrid
                else:
                    r = r_dec
                results.append(r)
                if verbose:
                    print(_fmt_pp_row(r))

    if verbose:
        print("-" * 170)
        print("Sample HBM TP×PP grid (TPOT ms):")
        # Print compact grid for tp_list × pp_list
        print(f"  {'pp\\tp':8s}", end="")
        for tp in tp_list:
            print(f"  tp={tp:<4d}", end="")
        print()
        for pp in pp_list:
            print(f"  pp={pp:<4d}  ", end="")
            for tp in tp_list:
                cell = next(
                    (
                        x
                        for x in results
                        if x.mem.kind == "HBM"
                        and x.tp_cfg.tp == tp
                        and x.tp_cfg.pp == pp
                    ),
                    None,
                )
                if cell:
                    print(f"  {cell.tpot_s*1e3:8.3f}", end="")
                else:
                    print(f"  {'n/a':>8s}", end="")
            print()
    return results


def run_pp_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    pp_list: list[int] | None = None,
    tp: int = 1,
    **kwargs,
) -> list:
    """Convenience: fix tp, sweep pp."""
    return run_parallel_sweep(
        verbose=verbose,
        cfg=cfg,
        sku=sku,
        tp_list=[tp],
        pp_list=pp_list,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# v0.9: MLA compressed KV + CSV export helpers
# ---------------------------------------------------------------------------


def _fmt_mla_row(r: InferenceResult, tag: str) -> str:
    kv = r.decode.traffic.kv_bytes_dram
    w = r.decode.traffic.weight_bytes_dram
    return (
        f"{tag:22s} {r.mem.kind:6s} ctx={r.cfg.decode_seq_len:<7d} "
        f"kv/tok={r.shape.kv_bytes_per_token()/1024:7.1f}KiB "
        f"KV_DRAM={kv/1e9:8.4f}GB W={w/1e9:7.3f}GB "
        f"KV/W={kv/max(w,1):.4f} "
        f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:7s}] "
        f"(c={r.decode.t_compute_s*1e3:.3f}/m={r.decode.t_memory_s*1e3:.3f})"
    )


def run_mla_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    ctx_list: list[int] | None = None,
    remote_kv_frac: float = 1.0,
    fabric_preset: str = "roce_200g",
) -> list:
    """Compare illustrative_mla vs GQA 27B at long ctx (HBM/LPDDR + remote fabric).

    MLA KV scales with kv_lora_rank (joint latent); GQA uses full 2*n_kv*d.
    """
    from .scaleup import (
        C2C_400,
        TPConfig,
        evaluate_scaleup,
        get_kv_fabric,
    )

    sku = sku or SKU_100T
    ctxs = list(ctx_list) if ctx_list is not None else [512, 128000]
    base = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        pin_weights=False,
        contention_mode="serialize",
    )
    shapes = [
        ("gqa_27B", ILLUSTRATIVE_27B),
        ("mla_r512", ILLUSTRATIVE_MLA),
    ]
    mems = [HBM_PRESET, LPDDR_PRESET]
    results: list = []

    if verbose:
        print("npu_dse MLA sweep — compressed KV vs GQA 27B @ long ctx")
        print(
            f"SKU={sku.name} fabric={fabric_preset} "
            f"remote_kv_frac={remote_kv_frac}"
        )
        print(
            "Honesty: MLA kv_bytes = L * kv_lora_rank * kv_bytes "
            "(joint latent); GQA = 2*L*n_kv*d*kv_bytes. Assumed illustrative."
        )
        for tag, sh in shapes:
            print(f"  {tag}: {sh.summary()}")
        print("-" * 160)

    for ctx in ctxs:
        prompt = min(base.prompt_len, ctx) if ctx > 0 else base.prompt_len
        ecfg = EvalConfig(
            prompt_len=max(prompt, 1),
            decode_seq_len=ctx,
            batch=base.batch,
            frequency_hz=sku.frequency_hz,
            pin_weights=base.pin_weights,
            contention_mode=base.contention_mode,
        )
        for tag, sh in shapes:
            for mem in mems:
                r = evaluate_inference(sh, sku.npu(), DEFAULT_SRAM, mem, ecfg)
                results.append(r)
                if verbose:
                    print(_fmt_mla_row(r, tag))

    # Remote fabric compare at 128k
    link = get_kv_fabric(fabric_preset)
    if link is not None:
        if verbose:
            print("-" * 160)
            print(f"Remote KV fabric @ 128k ({link.summary()}):")
        ctx = 128000
        ecfg = EvalConfig(
            prompt_len=min(base.prompt_len, ctx),
            decode_seq_len=ctx,
            batch=base.batch,
            frequency_hz=sku.frequency_hz,
            contention_mode=base.contention_mode,
        )
        for tag, sh in shapes:
            tp_cfg = TPConfig(
                tp=1,
                c2c=C2C_400,
                kv_fabric=link.mode,  # type: ignore[arg-type]
                fabric_gbps=link.link_gbps,
                remote_kv_frac=remote_kv_frac,
                pd_kv_xfer=True,
            )
            r = evaluate_scaleup(
                sh, sku.npu(), DEFAULT_SRAM, HBM_PRESET, ecfg, tp_cfg
            )
            results.append(r)
            if verbose:
                print(
                    f"{tag:22s} fab remote_KV={r.decode.remote_kv_bytes/1e9:8.4f}GB "
                    f"t_fab={r.decode.t_fabric_s*1e3:10.4f}ms "
                    f"t_xfer={r.t_kv_xfer_s*1e3:10.4f}ms "
                    f"TPOT={r.tpot_s*1e3:10.4f}ms [{r.decode.wall:8s}] "
                    f"kv/tok={sh.kv_bytes_per_token()/1024:.1f}KiB"
                )
        if verbose:
            gqa = ILLUSTRATIVE_27B.kv_bytes_per_token()
            mla = ILLUSTRATIVE_MLA.kv_bytes_per_token()
            print(
                f"  MLA/GQA kv_bytes_per_token ratio = {mla/gqa:.4f} "
                f"({mla} / {gqa})"
            )
    return results


def export_csv_bundle(
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> list[Path]:
    """Write key sweep CSVs under ``out_dir`` (sku, dtype, ctx, sram, tp, kv-fabric, moe)."""
    import csv
    from pathlib import Path as P

    from .dtype import DTYPE_REGISTRY
    from .scaleup import TPConfig, evaluate_scaleup, get_c2c, get_kv_fabric
    from .sku import SKU_REGISTRY

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[P] = []
    sku = SKU_100T
    npu = sku.npu()
    sram = DEFAULT_SRAM
    base = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        contention_mode="serialize",
    )

    def _write(name: str, rows: list[dict]) -> P:
        path = out / name
        if not rows:
            path.write_text("")
            written.append(path)
            return path
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        written.append(path)
        if verbose:
            print(f"  wrote {path} ({len(rows)} rows)")
        return path

    if verbose:
        print(f"npu_dse export-csv → {out}")

    # sku
    rows = []
    for sname, s in SKU_REGISTRY.items():
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_inference(
                ILLUSTRATIVE_27B,
                s.npu(),
                sram,
                mem,
                EvalConfig(
                    prompt_len=512,
                    decode_seq_len=512,
                    frequency_hz=s.frequency_hz,
                ),
            )
            rows.append({
                "sweep": "sku",
                "sku": sname,
                "mem": mem.kind,
                "tops": f"{r.peak_tops():.2f}",
                "ttft_ms": f"{r.ttft_s*1e3:.6f}",
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
                "w_dram_gb": f"{r.decode.traffic.weight_bytes_dram/1e9:.6f}",
                "kv_dram_gb": f"{r.decode.traffic.kv_bytes_dram/1e9:.6f}",
            })
    _write("sweep_sku.csv", rows)

    # dtype
    rows = []
    for dname, d in DTYPE_REGISTRY.items():
        sh = d.apply(ILLUSTRATIVE_27B)
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_inference(sh, npu, sram, mem, base)
            rows.append({
                "sweep": "dtype",
                "dtype": dname,
                "mem": mem.kind,
                "weight_bits": sh.weight_bits,
                "kv_bits": sh.kv_bits,
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
                "w_dram_gb": f"{r.decode.traffic.weight_bytes_dram/1e9:.6f}",
                "kv_dram_gb": f"{r.decode.traffic.kv_bytes_dram/1e9:.6f}",
            })
    _write("sweep_dtype.csv", rows)

    # ctx
    rows = []
    for ctx in DEFAULT_CTX_SWEEP:
        ecfg = EvalConfig(
            prompt_len=min(512, ctx),
            decode_seq_len=ctx,
            frequency_hz=sku.frequency_hz,
        )
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, mem, ecfg)
            w = r.decode.traffic.weight_bytes_dram
            kv = r.decode.traffic.kv_bytes_dram
            rows.append({
                "sweep": "ctx",
                "ctx": ctx,
                "mem": mem.kind,
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
                "w_dram_gb": f"{w/1e9:.6f}",
                "kv_dram_gb": f"{kv/1e9:.6f}",
                "kv_over_w": f"{kv/max(w,1):.6f}",
            })
    _write("sweep_ctx.csv", rows)

    # sram knee
    rows = []
    for mib in DEFAULT_SRAM_SWEEP_MIB:
        sc = SRAMConfig(int(mib * 1024 * 1024))
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_inference(ILLUSTRATIVE_27B, npu, sc, mem, base)
            rows.append({
                "sweep": "sram",
                "sram_mib": mib,
                "mem": mem.kind,
                "resident_layers": r.decode.traffic.resident_layers,
                "n_layers": ILLUSTRATIVE_27B.n_layers,
                "weight_path": r.decode.traffic.weight_path,
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
                "w_dram_gb": f"{r.decode.traffic.weight_bytes_dram/1e9:.6f}",
            })
    _write("sweep_sram.csv", rows)

    # tp
    rows = []
    c2c = get_c2c(400)
    for tp in DEFAULT_TP_SWEEP:
        tp_cfg = TPConfig(tp=tp, c2c=c2c, kv_fabric="none")
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_scaleup(
                ILLUSTRATIVE_27B, npu, sram, mem, base, tp_cfg
            )
            rows.append({
                "sweep": "tp",
                "tp": tp,
                "pp": 1,
                "ep": 1,
                "cards": r.tp_cfg.n_cards,
                "mem": mem.kind,
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "ttft_ms": f"{r.ttft_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
                "t_compute_ms": f"{r.decode.t_compute_s*1e3:.6f}",
                "t_dram_ms": f"{r.decode.t_memory_s*1e3:.6f}",
                "t_c2c_ms": f"{r.decode.t_c2c_s*1e3:.6f}",
                "coll_mb": f"{r.decode.collective_bytes/1e6:.6f}",
                "oom": int(r.oom),
            })
    _write("sweep_tp.csv", rows)

    # kv-fabric
    rows = []
    for preset in ("none", "roce_100g", "roce_200g", "roce_400g", "ib_200g", "ib_400g"):
        for ctx in (512, 128000):
            if preset == "none":
                mode, gbps, frac, xfer = "none", 200.0, 0.0, False
            else:
                link = get_kv_fabric(preset)
                assert link is not None
                mode, gbps, frac, xfer = link.mode, link.link_gbps, 1.0, True
            ecfg = EvalConfig(
                prompt_len=min(512, ctx),
                decode_seq_len=ctx,
                frequency_hz=sku.frequency_hz,
            )
            tp_cfg = TPConfig(
                tp=1,
                c2c=c2c,
                kv_fabric=mode,  # type: ignore[arg-type]
                fabric_gbps=gbps,
                remote_kv_frac=frac,
                pd_kv_xfer=xfer,
            )
            r = evaluate_scaleup(
                ILLUSTRATIVE_27B, npu, sram, HBM_PRESET, ecfg, tp_cfg
            )
            rows.append({
                "sweep": "kv_fabric",
                "preset": preset,
                "ctx": ctx,
                "mem": "HBM",
                "remote_kv_gb": f"{r.decode.remote_kv_bytes/1e9:.6f}",
                "t_fabric_ms": f"{r.decode.t_fabric_s*1e3:.6f}",
                "t_kv_xfer_ms": f"{r.t_kv_xfer_s*1e3:.6f}",
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
            })
    _write("sweep_kv_fabric.csv", rows)

    # moe (+ ep)
    rows = []
    for ep in (1, 2, 4, 8):
        if ILLUSTRATIVE_MOE.n_experts % ep != 0:
            continue
        tp_cfg = TPConfig(tp=1, pp=1, ep=ep, c2c=c2c, kv_fabric="none")
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_scaleup(
                ILLUSTRATIVE_MOE, npu, sram, mem, base, tp_cfg
            )
            rows.append({
                "sweep": "moe_ep",
                "shape": ILLUSTRATIVE_MOE.name,
                "ep": ep,
                "cards": r.tp_cfg.n_cards,
                "mem": mem.kind,
                "w_card_gb": f"{r.decode.weight_bytes_card/1e9:.6f}",
                "a2a_mb": f"{r.decode.a2a_bytes/1e6:.6f}",
                "t_a2a_ms": f"{r.decode.t_a2a_s*1e3:.6f}",
                "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                "wall_decode": r.decode.wall,
            })
    # dense compare single-card
    for mem in (HBM_PRESET, LPDDR_PRESET):
        r = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, mem, base)
        rows.append({
            "sweep": "moe_ep",
            "shape": ILLUSTRATIVE_27B.name,
            "ep": 1,
            "cards": 1,
            "mem": mem.kind,
            "w_card_gb": f"{r.decode.traffic.weight_bytes_dram/1e9:.6f}",
            "a2a_mb": "0",
            "t_a2a_ms": "0",
            "tpot_ms": f"{r.tpot_s*1e3:.6f}",
            "wall_decode": r.decode.wall,
        })
    _write("sweep_moe.csv", rows)

    # mla @ 128k
    rows = []
    for tag, sh in (("gqa_27B", ILLUSTRATIVE_27B), ("mla_r512", ILLUSTRATIVE_MLA)):
        for ctx in (512, 128000):
            ecfg = EvalConfig(
                prompt_len=min(512, ctx),
                decode_seq_len=ctx,
                frequency_hz=sku.frequency_hz,
            )
            for mem in (HBM_PRESET, LPDDR_PRESET):
                r = evaluate_inference(sh, npu, sram, mem, ecfg)
                rows.append({
                    "sweep": "mla",
                    "shape": tag,
                    "ctx": ctx,
                    "mem": mem.kind,
                    "kv_lora_rank": sh.kv_lora_rank,
                    "kv_per_tok_kib": f"{sh.kv_bytes_per_token()/1024:.4f}",
                    "kv_dram_gb": f"{r.decode.traffic.kv_bytes_dram/1e9:.6f}",
                    "w_dram_gb": f"{r.decode.traffic.weight_bytes_dram/1e9:.6f}",
                    "tpot_ms": f"{r.tpot_s*1e3:.6f}",
                    "wall_decode": r.decode.wall,
                })
    _write("sweep_mla.csv", rows)

    # video + protein (v0.10 multi-domain)
    written.extend(export_video_protein_csv(str(out), verbose=verbose))
    written.extend(export_compare_domains_csv(str(out), verbose=verbose))

    if verbose:
        print(f"export-csv done: {len(written)} files")
    return written


# ---------------------------------------------------------------------------
# Video / protein multi-domain sweeps (v0.10)
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_DENOISE_SWEEP = (20, 50, 100)
DEFAULT_PROTEIN_L_SWEEP = (256, 512, 1024, 2048)


def run_video_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    denoise_list: list[int] | None = None,
    shape: "object | None" = None,
) -> list:
    """Sweep illustrative DiT-video × HBM/LPDDR @ sku_100t.

    Reports time-to-first-clip ≈ N_denoise × forward, frames/s, wall.
    """
    from .workloads import ILLUSTRATIVE_DIT_VIDEO, VideoShape, evaluate_video

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_DIT_VIDEO
    assert isinstance(shape, VideoShape)
    denoise_list = list(denoise_list or DEFAULT_VIDEO_DENOISE_SWEEP)
    cfg = cfg or EvalConfig(
        frequency_hz=sku.frequency_hz,
        contention_mode="serialize",
    )
    npu = sku.npu()
    sram = DEFAULT_SRAM
    results = []
    if verbose:
        print(
            f"npu_dse sweep-video (assumed/uncalibrated)  "
            f"sku={sku.name}  shape={shape.name}  "
            f"T={shape.n_tokens} F={shape.n_frames}"
        )
        print(
            f"{'N_den':>6} {'mem':6s} {'fwd_ms':>10} {'TTFC_ms':>12} "
            f"{'fps':>10} {'wall':8s} {'oom':>4} "
            f"{'W_GB':>8} {'FLOPs_fwd':>14}"
        )
        print("-" * 100)
    for nd in denoise_list:
        sh = shape.with_denoise(nd)
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_video(sh, npu, sram, mem, cfg)
            results.append(r)
            if verbose:
                print(
                    f"{nd:6d} {mem.kind:6s} "
                    f"{r.forward.t_roofline_s*1e3:10.3f} "
                    f"{r.time_to_first_clip_s*1e3:12.3f} "
                    f"{r.frames_per_s:10.4f} "
                    f"{r.forward.wall:8s} "
                    f"{'YES' if r.oom else 'no':>4} "
                    f"{r.forward.traffic.weight_bytes_dram/1e9:8.3f} "
                    f"{r.forward.traffic.flops:14d}"
                )
    if verbose and results:
        print("-" * 100)
        print("Detail (default N_denoise @ HBM):")
        for r in results:
            if r.mem.kind == "HBM" and r.n_denoise == shape.n_denoise:
                for line in r.summary_lines():
                    print(line)
                break
    return results


def run_protein_sweep(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
    seq_list: list[int] | None = None,
    shape: "object | None" = None,
) -> list:
    """Sweep illustrative protein × L × HBM/LPDDR @ sku_100t; flag OOM."""
    from .workloads import ILLUSTRATIVE_PROTEIN_PAIR, ProteinShape, evaluate_protein

    sku = sku or SKU_100T
    shape = shape or ILLUSTRATIVE_PROTEIN_PAIR
    assert isinstance(shape, ProteinShape)
    seq_list = list(seq_list or DEFAULT_PROTEIN_L_SWEEP)
    cfg = cfg or EvalConfig(
        frequency_hz=sku.frequency_hz,
        contention_mode="serialize",
    )
    npu = sku.npu()
    sram = DEFAULT_SRAM
    results = []
    if verbose:
        print(
            f"npu_dse sweep-protein (assumed/uncalibrated)  "
            f"sku={sku.name}  shape={shape.name}  "
            f"pair_dim={shape.pair_dim} layers={shape.n_layers}"
        )
        print(
            f"{'L_aa':>6} {'mem':6s} {'t_ms':>10} {'wall':8s} "
            f"{'pair_GB':>10} {'need_GB':>10} {'cap_GB':>8} "
            f"{'oom':>4} {'FLOPs':>14}"
        )
        print("-" * 100)
    for L in seq_list:
        sh = shape.with_seq_len(L)
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_protein(sh, npu, sram, mem, cfg)
            results.append(r)
            if verbose:
                print(
                    f"{L:6d} {mem.kind:6s} "
                    f"{r.time_per_sequence_s*1e3:10.3f} "
                    f"{r.forward.wall:8s} "
                    f"{r.pair_bytes/1e9:10.4f} "
                    f"{r.capacity_needed_bytes/1e9:10.4f} "
                    f"{mem.capacity_bytes/1e9:8.1f} "
                    f"{'YES' if r.oom else 'no':>4} "
                    f"{r.forward.traffic.flops:14d}"
                )
    if verbose and results:
        print("-" * 100)
        print("Detail (L=512 @ HBM):")
        for r in results:
            if r.mem.kind == "HBM" and r.shape.seq_len == 512:
                for line in r.summary_lines():
                    print(line)
                break
    return results


def export_video_protein_csv(
    out_dir: str,
    *,
    verbose: bool = True,
) -> list:
    """Write sweep_video.csv and sweep_protein.csv."""
    import csv
    from pathlib import Path as P

    from .workloads import ILLUSTRATIVE_DIT_VIDEO, ILLUSTRATIVE_PROTEIN_PAIR, evaluate_video, evaluate_protein

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sku = SKU_100T
    npu = sku.npu()
    sram = DEFAULT_SRAM
    cfg = EvalConfig(frequency_hz=sku.frequency_hz, contention_mode="serialize")
    written = []

    rows = []
    for nd in DEFAULT_VIDEO_DENOISE_SWEEP:
        sh = ILLUSTRATIVE_DIT_VIDEO.with_denoise(nd)
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_video(sh, npu, sram, mem, cfg)
            rows.append({
                "sweep": "video",
                "shape": sh.name,
                "n_denoise": nd,
                "n_frames": sh.n_frames,
                "n_tokens": sh.n_tokens,
                "mem": mem.kind,
                "fwd_ms": f"{r.forward.t_roofline_s*1e3:.6f}",
                "ttfc_ms": f"{r.time_to_first_clip_s*1e3:.6f}",
                "frames_per_s": f"{r.frames_per_s:.6f}",
                "wall": r.forward.wall,
                "oom": r.oom,
                "w_dram_gb": f"{r.forward.traffic.weight_bytes_dram/1e9:.6f}",
                "flops_fwd": r.forward.traffic.flops,
            })
    path = out / "sweep_video.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    written.append(path)
    if verbose:
        print(f"  wrote {path} ({len(rows)} rows)")

    rows = []
    for L in DEFAULT_PROTEIN_L_SWEEP:
        sh = ILLUSTRATIVE_PROTEIN_PAIR.with_seq_len(L)
        for mem in (HBM_PRESET, LPDDR_PRESET):
            r = evaluate_protein(sh, npu, sram, mem, cfg)
            rows.append({
                "sweep": "protein",
                "shape": sh.name,
                "seq_len": L,
                "pair_dim": sh.pair_dim,
                "mem": mem.kind,
                "t_ms": f"{r.time_per_sequence_s*1e3:.6f}",
                "wall": r.forward.wall,
                "pair_gb": f"{r.pair_bytes/1e9:.6f}",
                "need_gb": f"{r.capacity_needed_bytes/1e9:.6f}",
                "cap_gb": f"{mem.capacity_bytes/1e9:.1f}",
                "oom": r.oom,
                "flops": r.forward.traffic.flops,
            })
    path = out / "sweep_protein.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    written.append(path)
    if verbose:
        print(f"  wrote {path} ({len(rows)} rows)")
    return written


# ---------------------------------------------------------------------------
# Cross-domain compare (v0.11)
# ---------------------------------------------------------------------------

DOMINANT_COST_VALUES = ("weight_stream", "denoise_attn", "pair_L2", "compute_gemm")


def _llm_dominant_cost(wall: str) -> str:
    """LLM decode: memory wall → weight stream; else OS/GEMM compute."""
    if wall == "memory":
        return "weight_stream"
    return "compute_gemm"


def run_compare_domains(
    verbose: bool = True,
    cfg: EvalConfig | None = None,
    sku: SKUPreset | None = None,
) -> list[dict]:
    """Cross-domain one-pager: LLM / video / protein @ sku_100t × HBM/LPDDR.

    Rows (per mem):
      - illustrative_27B decode TPOT (fp16, ctx=512, batch=1)
      - illustrative_mla same
      - illustrative_dit_video TTFC @ N_denoise=50 (+ fps in notes)
      - illustrative_protein_pair t/seq @ L=512

    Columns: domain, mem, metric_name, metric_ms, wall, dominant_cost, notes.
    """
    from .workloads import (
        ILLUSTRATIVE_DIT_VIDEO,
        ILLUSTRATIVE_PROTEIN_PAIR,
        evaluate_protein,
        evaluate_video,
    )

    sku = sku or SKU_100T
    cfg = cfg or EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        contention_mode="serialize",
    )
    # Force compare defaults
    cfg = EvalConfig(
        prompt_len=512,
        decode_seq_len=512,
        batch=1,
        frequency_hz=sku.frequency_hz,
        contention_mode=cfg.contention_mode,
        pin_weights=cfg.pin_weights,
        weight_hide_factor=cfg.weight_hide_factor,
        sram_policy=cfg.sram_policy,
    )
    npu = sku.npu()
    sram = DEFAULT_SRAM
    rows: list[dict] = []

    if verbose:
        print(
            f"npu_dse compare-domains (assumed/uncalibrated)  "
            f"sku={sku.name}  SRAM={sram.capacity_bytes/2**20:.0f}MiB  "
            f"fp16 ctx=512 batch=1"
        )
        print(
            f"{'domain':8s} {'mem':6s} {'metric':16s} {'metric_ms':>12} "
            f"{'wall':8s} {'dominant_cost':14s}  notes"
        )
        print("-" * 120)

    for mem in (HBM_PRESET, LPDDR_PRESET):
        # LLM 27B
        r27 = evaluate_inference(ILLUSTRATIVE_27B, npu, sram, mem, cfg)
        dom = _llm_dominant_cost(r27.decode.wall)
        row = {
            "domain": "llm",
            "mem": mem.kind,
            "metric_name": "TPOT",
            "metric_ms": r27.tpot_s * 1e3,
            "wall": r27.decode.wall,
            "dominant_cost": dom,
            "notes": (
                f"shape=illustrative_27B fp16 ctx=512 batch=1; "
                f"W_dram={r27.decode.traffic.weight_bytes_dram/1e9:.2f}GB "
                f"KV={r27.decode.traffic.kv_bytes_dram/1e9:.4f}GB"
            ),
            "shape": "illustrative_27B",
        }
        rows.append(row)

        # LLM MLA
        r_mla = evaluate_inference(ILLUSTRATIVE_MLA, npu, sram, mem, cfg)
        dom = _llm_dominant_cost(r_mla.decode.wall)
        row = {
            "domain": "llm",
            "mem": mem.kind,
            "metric_name": "TPOT",
            "metric_ms": r_mla.tpot_s * 1e3,
            "wall": r_mla.decode.wall,
            "dominant_cost": dom,
            "notes": (
                f"shape=illustrative_mla fp16 ctx=512 batch=1; "
                f"kv_lora_rank={ILLUSTRATIVE_MLA.kv_lora_rank}; "
                f"W_dram={r_mla.decode.traffic.weight_bytes_dram/1e9:.2f}GB "
                f"KV={r_mla.decode.traffic.kv_bytes_dram/1e9:.4f}GB"
            ),
            "shape": "illustrative_mla",
        }
        rows.append(row)

        # Video DiT
        vshape = ILLUSTRATIVE_DIT_VIDEO.with_denoise(50)
        rv = evaluate_video(vshape, npu, sram, mem, cfg)
        row = {
            "domain": "video",
            "mem": mem.kind,
            "metric_name": "TTFC",
            "metric_ms": rv.time_to_first_clip_s * 1e3,
            "wall": rv.forward.wall,
            "dominant_cost": "denoise_attn",
            "notes": (
                f"shape=illustrative_dit_video N_denoise=50 T={vshape.n_tokens}; "
                f"fps={rv.frames_per_s:.4f}; "
                f"fwd_ms={rv.forward.t_roofline_s*1e3:.3f}"
            ),
            "shape": "illustrative_dit_video",
        }
        rows.append(row)

        # Protein
        pshape = ILLUSTRATIVE_PROTEIN_PAIR.with_seq_len(512)
        rp = evaluate_protein(pshape, npu, sram, mem, cfg)
        row = {
            "domain": "protein",
            "mem": mem.kind,
            "metric_name": "t_seq",
            "metric_ms": rp.time_per_sequence_s * 1e3,
            "wall": rp.forward.wall,
            "dominant_cost": "pair_L2",
            "notes": (
                f"shape=illustrative_protein_pair L=512 pair_dim={pshape.pair_dim}; "
                f"pair_GB={rp.pair_bytes/1e9:.4f}; oom={rp.oom}"
            ),
            "shape": "illustrative_protein_pair",
        }
        rows.append(row)

    if verbose:
        for row in rows:
            print(
                f"{row['domain']:8s} {row['mem']:6s} {row['metric_name']:16s} "
                f"{row['metric_ms']:12.3f} {row['wall']:8s} "
                f"{row['dominant_cost']:14s}  {row['notes']}"
            )
        print("-" * 120)
        print(
            "dominant_cost ∈ {weight_stream, denoise_attn, pair_L2, compute_gemm} "
            "(assumed labels for DSE narrative)"
        )
    return rows


def export_compare_domains_csv(
    out_dir: str,
    *,
    verbose: bool = True,
) -> list:
    """Write compare_domains.csv from run_compare_domains."""
    import csv
    from pathlib import Path as P

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = run_compare_domains(verbose=False)
    fieldnames = [
        "domain",
        "shape",
        "mem",
        "metric_name",
        "metric_ms",
        "wall",
        "dominant_cost",
        "notes",
    ]
    path = out / "compare_domains.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["metric_ms"] = f"{row['metric_ms']:.6f}"
            w.writerow(out_row)
    if verbose:
        print(f"  wrote {path} ({len(rows)} rows)")
    return [path]
