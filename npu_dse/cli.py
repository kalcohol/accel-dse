"""CLI: python -m npu_dse.cli  or  python -m npu_dse"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_OUT = Path(__file__).resolve().parents[1] / "out"

from .evaluate import EvalConfig, evaluate_inference
from .memory import HBM_PRESET, LPDDR_PRESET, SRAMConfig
from .model_shape import ILLUSTRATIVE_27B, ILLUSTRATIVE_MLA, ILLUSTRATIVE_MOE, TOY_SHAPE
from .npu import NPUConfig, gemm_utilization, utilization_vs_m
from .scan import (
    DEFAULT_BATCH_SWEEP,
    DEFAULT_CTX_SWEEP,
    DEFAULT_KV_CTX_SWEEP,
    DEFAULT_KV_FABRIC_PRESETS,
    DEFAULT_PP_SWEEP,
    DEFAULT_SRAM_SWEEP_MIB,
    DEFAULT_TP_SWEEP,
    run_batch_sweep,
    run_ctx_sweep,
    run_default_scan,
    run_dtype_sweep,
    run_kv_fabric_sweep,
    run_moe_sweep,
    run_mla_sweep,
    run_parallel_sweep,
    run_pp_sweep,
    run_quant_sweep,
    run_sku_scan,
    run_sram_sweep,
    run_tp_sweep,
    run_scaleup_scan,
    export_csv_bundle,
    DEFAULT_EP_SWEEP,
    run_video_sweep,
    run_protein_sweep,
    DEFAULT_VIDEO_DENOISE_SWEEP,
    DEFAULT_PROTEIN_L_SWEEP,
    export_video_protein_csv,
    run_compare_domains,
    export_compare_domains_csv,
)
from .sku import SKU_REGISTRY, get_sku

from .workbench import (
    CalibrationOverrides,
    DEFAULT_CHIP_SWEEP,
    DEFAULT_MEM_CHANNEL_SWEEP,
    DEFAULT_MEM_RATE_SWEEP_HBM,
    DEFAULT_MEM_RATE_SWEEP_LPDDR,
    ParallelOverride,
    WorkbenchConfig,
    evaluate_workbench,
    export_metrics_card_json,
    export_metrics_card_md,
    export_scan_csv,
    format_compute_sweep_table,
    format_mem_sweep_table,
    format_package_sweep_table,
    format_parallel_table,
    format_scan_table,
    workbench_compute_sweep,
    workbench_mem_sweep,
    workbench_package_sweep,
    workbench_parallel_matrix,
    workbench_scan,
)
from .series import list_series, list_series_rows
from .report import write_report
from .package_ranges import (
    DEFAULT_COMPUTE_SWEEP,
    DEFAULT_PACKAGE_SWEEP,
    format_compute_table,
    format_package_table,
    get_package,
    list_compute_primaries,
    list_packages,
    package_counts,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="npu_dse",
        description="Analytical NPU+SRAM+HBM/LPDDR multi-domain (LLM/video/protein) inference DSE",
    )
    sub = p.add_subparsers(dest="cmd")

    scan_p = sub.add_parser("scan", help="Run default DSE scan table (includes SKU rows)")
    scan_p.add_argument("--prompt", type=int, default=512)
    scan_p.add_argument("--ctx", type=int, default=512)
    scan_p.add_argument("--freq-ghz", type=float, default=1.0)
    scan_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    sku_p = sub.add_parser(
        "scan-sku",
        help="Scan 100T / 1P-class single-card peak templates; show mem-wall crossover",
    )
    sku_p.add_argument("--prompt", type=int, default=512)
    sku_p.add_argument("--ctx", type=int, default=512)
    sku_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    dtype_p = sub.add_parser(
        "sweep-dtype",
        help="Sweep fp16/fp8/int8/int4 storage bits on sku_100t @ HBM+LPDDR (bytes-only)",
    )
    dtype_p.add_argument("--prompt", type=int, default=512)
    dtype_p.add_argument("--ctx", type=int, default=512)
    dtype_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    dtype_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    ctx_p = sub.add_parser(
        "sweep-ctx",
        help="Sweep decode ctx lengths; show when KV rivals weight stream",
    )
    ctx_p.add_argument("--prompt", type=int, default=512)
    ctx_p.add_argument(
        "--ctx-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_CTX_SWEEP),
        help=f"context lengths (default: {' '.join(map(str, DEFAULT_CTX_SWEEP))})",
    )
    ctx_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    ctx_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    sram_p = sub.add_parser(
        "sweep-sram",
        help="Sweep SRAM MiB; show resident knees (R≥1 / R≥L/2 / R≥L)",
    )
    sram_p.add_argument("--prompt", type=int, default=512)
    sram_p.add_argument("--ctx", type=int, default=512)
    sram_p.add_argument(
        "--sram-list",
        type=float,
        nargs="+",
        default=list(DEFAULT_SRAM_SWEEP_MIB),
        help=f"SRAM MiB points (default: {' '.join(str(x) for x in DEFAULT_SRAM_SWEEP_MIB)})",
    )
    sram_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    sram_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    sram_p.add_argument(
        "--weight-hide",
        type=float,
        default=0.0,
        help="assumed weight hide factor when dbl-buf eligible (0 or 0.5)",
    )
    sram_p.add_argument(
        "--sram-policy",
        choices=["weight_resident", "kv_first", "balanced"],
        default="weight_resident",
        help="SRAM partition policy (default: weight_resident maximizes R)",
    )
    sram_p.add_argument(
        "--dtype",
        choices=["fp16", "fp8", "e4m3", "int8", "int4"],
        default="fp16",
        help="uniform dtype on weight+KV for the SRAM knee shape",
    )

    quant_p = sub.add_parser(
        "sweep-quant",
        help="Sweep W/KV bit pairs (w16k16..w4k8); show when KV overtakes W",
    )
    quant_p.add_argument("--prompt", type=int, default=512)
    quant_p.add_argument(
        "--ctx-list",
        type=int,
        nargs="+",
        default=[512, 128000],
        help="context lengths (default: 512 128000)",
    )
    quant_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    quant_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    quant_p.add_argument(
        "--quant",
        nargs="+",
        default=None,
        help="subset of w16k16 w8k16 w4k16 w8k8 w4k8 (default: all)",
    )

    batch_p = sub.add_parser(
        "sweep-batch",
        help="Sweep batch={1,2,4,8,16,32} @ sku_100t decode; util rises with M",
    )
    batch_p.add_argument("--prompt", type=int, default=512)
    batch_p.add_argument("--ctx", type=int, default=512)
    batch_p.add_argument(
        "--batch-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SWEEP),
        help=f"batch sizes (default: {' '.join(map(str, DEFAULT_BATCH_SWEEP))})",
    )
    batch_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    batch_p.add_argument(
        "--shape",
        choices=["27b", "moe", "toy"],
        default="27b",
    )
    batch_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    moe_p = sub.add_parser(
        "sweep-moe",
        help="MoE active-W stream vs dense; optional --ep expert-parallel",
    )
    moe_p.add_argument("--prompt", type=int, default=512)
    moe_p.add_argument("--ctx", type=int, default=512)
    moe_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    moe_p.add_argument(
        "--ep",
        type=int,
        nargs="+",
        default=[1],
        dest="ep_list",
        help=f"EP degrees to sweep (default: 1; try {' '.join(map(str, DEFAULT_EP_SWEEP))})",
    )
    moe_p.add_argument("--c2c-gbps", type=float, default=400.0)
    moe_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    sweep_p = sub.add_parser(
        "sweep",
        help="Unified sweep: --what dtype|ctx|sram|quant|batch|moe",
    )
    sweep_p.add_argument(
        "--what",
        choices=["dtype", "ctx", "sram", "quant", "batch", "moe", "mla"],
        required=True,
        help="which sweep to run",
    )
    sweep_p.add_argument("--prompt", type=int, default=512)
    sweep_p.add_argument("--ctx", type=int, default=512)
    sweep_p.add_argument(
        "--ctx-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_CTX_SWEEP),
    )
    sweep_p.add_argument(
        "--sram-list",
        type=float,
        nargs="+",
        default=list(DEFAULT_SRAM_SWEEP_MIB),
    )
    sweep_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    sweep_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    sweep_p.add_argument("--weight-hide", type=float, default=0.0)
    sweep_p.add_argument(
        "--sram-policy",
        choices=["weight_resident", "kv_first", "balanced"],
        default="weight_resident",
    )
    sweep_p.add_argument(
        "--dtype",
        choices=["fp16", "fp8", "e4m3", "int8", "int4"],
        default="fp16",
    )
    sweep_p.add_argument("--quant", nargs="+", default=None)
    sweep_p.add_argument(
        "--batch-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SWEEP),
    )
    sweep_p.add_argument(
        "--shape",
        choices=["27b", "moe", "toy"],
        default="27b",
        help="shape for batch/moe-related sweeps",
    )

    eval_p = sub.add_parser("eval", help="Evaluate one configuration")
    eval_p.add_argument("--shape", choices=["toy", "27b", "moe", "mla"], default="toy")
    eval_p.add_argument("--mem", choices=["hbm", "lpddr"], default="hbm")
    eval_p.add_argument("--sram-mib", type=float, default=64.0)
    eval_p.add_argument("--pe", type=int, nargs=2, default=None, metavar=("R", "C"))
    eval_p.add_argument("--engines", type=int, default=1)
    eval_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default=None,
        help="Use a named SKU preset (overrides --pe/--engines/--freq-ghz)",
    )
    eval_p.add_argument("--prompt", type=int, default=512)
    eval_p.add_argument("--ctx", type=int, default=512)
    eval_p.add_argument("--freq-ghz", type=float, default=1.0)
    eval_p.add_argument(
        "--weight-bits",
        type=int,
        default=None,
        help="Override weight storage bits (bytes-only; MAC rate unchanged)",
    )
    eval_p.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Override KV storage bits (bytes-only)",
    )
    eval_p.add_argument(
        "--dtype",
        choices=["fp16", "fp8", "e4m3", "int8", "int4"],
        default=None,
        help="Apply dtype preset to weight+KV bits",
    )
    eval_p.add_argument(
        "--quant",
        choices=["w16k16", "w8k16", "w4k16", "w8k8", "w4k8"],
        default=None,
        help="Independent W/KV bit preset (overrides --dtype for bits)",
    )
    eval_p.add_argument(
        "--weight-hide",
        type=float,
        default=0.0,
        help="assumed weight hide when dbl-buf eligible (0 or 0.5)",
    )
    eval_p.add_argument(
        "--sram-policy",
        choices=["weight_resident", "kv_first", "balanced"],
        default="weight_resident",
        help="SRAM partition policy",
    )
    eval_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    eval_p.add_argument("--pin-weights", action="store_true")


    tp_p = sub.add_parser(
        "sweep-tp",
        help="Sweep tp={1,2,4,8} × HBM/LPDDR @ sku_100t; C2C collectives (scale-up MVP)",
    )
    tp_p.add_argument("--prompt", type=int, default=512)
    tp_p.add_argument("--ctx", type=int, default=512)
    tp_p.add_argument(
        "--tp-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_TP_SWEEP),
        help=f"TP degrees (default: {' '.join(map(str, DEFAULT_TP_SWEEP))})",
    )
    tp_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    tp_p.add_argument(
        "--c2c-gbps",
        type=float,
        default=400.0,
        help="assumed C2C effective BW GB/s (100/200/400/800 presets)",
    )
    tp_p.add_argument(
        "--c2c-hide",
        type=float,
        default=0.0,
        help="assumed C2C/compute overlap fraction [0,1] (default 0)",
    )
    tp_p.add_argument(
        "--embed-policy",
        choices=["replicate", "shard"],
        default="replicate",
        help="embedding capacity: replicate (default) or shard across TP",
    )
    tp_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    scale_p = sub.add_parser(
        "scan-scaleup",
        help="Alias for sweep-tp (dense 27B × TP × HBM/LPDDR × sku_100t)",
    )
    scale_p.add_argument("--prompt", type=int, default=512)
    scale_p.add_argument("--ctx", type=int, default=512)
    scale_p.add_argument(
        "--tp-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_TP_SWEEP),
    )
    scale_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    scale_p.add_argument("--c2c-gbps", type=float, default=400.0)
    scale_p.add_argument("--c2c-hide", type=float, default=0.0)
    scale_p.add_argument(
        "--embed-policy",
        choices=["replicate", "shard"],
        default="replicate",
    )
    scale_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )


    kv_p = sub.add_parser(
        "sweep-kv-fabric",
        help="Sweep IB/RoCE KV fabric presets × ctx @ sku_100t (remote KV + PD xfer)",
    )
    kv_p.add_argument("--prompt", type=int, default=512)
    kv_p.add_argument(
        "--ctx-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_KV_CTX_SWEEP),
        help=f"context lengths (default: {' '.join(map(str, DEFAULT_KV_CTX_SWEEP))})",
    )
    kv_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    kv_p.add_argument(
        "--fabric-list",
        nargs="+",
        default=list(DEFAULT_KV_FABRIC_PRESETS),
        help="none | roce_{100,200,400}g | ib_{100,200,400}g",
    )
    kv_p.add_argument(
        "--remote-kv-frac",
        type=float,
        default=1.0,
        help="fraction of KV reads from fabric (default 1.0 = disagg decode)",
    )
    kv_p.add_argument(
        "--no-pd-xfer",
        action="store_true",
        help="disable prefill→decode full-KV transfer metric",
    )
    kv_p.add_argument("--tp", type=int, default=1)
    kv_p.add_argument("--c2c-gbps", type=float, default=400.0)
    kv_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    par_p = sub.add_parser(
        "sweep-parallel",
        help="Sweep tp×pp (cards=tp*pp) @ sku_100t; decode bubble=(pp-1)/pp for mb=1",
    )
    par_p.add_argument("--prompt", type=int, default=512)
    par_p.add_argument("--ctx", type=int, default=512)
    par_p.add_argument(
        "--tp-list",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    par_p.add_argument(
        "--pp-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_PP_SWEEP),
    )
    par_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    par_p.add_argument("--c2c-gbps", type=float, default=400.0)
    par_p.add_argument("--c2c-hide", type=float, default=0.0)
    par_p.add_argument(
        "--embed-policy",
        choices=["replicate", "shard"],
        default="replicate",
    )
    par_p.add_argument(
        "--decode-mb",
        type=int,
        default=1,
        help="decode microbatches (default 1 → bubble=(pp-1)/pp)",
    )
    par_p.add_argument(
        "--prefill-mb",
        type=int,
        default=None,
        help="prefill microbatches (default max(pp, decode_mb))",
    )
    par_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    pp_p = sub.add_parser(
        "sweep-pp",
        help="Sweep pp={1,2,4,8} at fixed --tp (default 1); alias of sweep-parallel",
    )
    pp_p.add_argument("--prompt", type=int, default=512)
    pp_p.add_argument("--ctx", type=int, default=512)
    pp_p.add_argument("--tp", type=int, default=1)
    pp_p.add_argument(
        "--pp-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_PP_SWEEP),
    )
    pp_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    pp_p.add_argument("--c2c-gbps", type=float, default=400.0)
    pp_p.add_argument("--c2c-hide", type=float, default=0.0)
    pp_p.add_argument(
        "--embed-policy",
        choices=["replicate", "shard"],
        default="replicate",
    )
    pp_p.add_argument("--decode-mb", type=int, default=1)
    pp_p.add_argument("--prefill-mb", type=int, default=None)
    pp_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    mla_p = sub.add_parser(
        "sweep-mla",
        help="MLA compressed KV vs GQA 27B @ ctx (HBM/LPDDR + remote fabric)",
    )
    mla_p.add_argument("--prompt", type=int, default=512)
    mla_p.add_argument(
        "--ctx-list",
        type=int,
        nargs="+",
        default=[512, 128000],
    )
    mla_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    mla_p.add_argument(
        "--fabric",
        default="roce_200g",
        help="KV fabric preset for 128k remote compare",
    )
    mla_p.add_argument("--remote-kv-frac", type=float, default=1.0)
    mla_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    csv_p = sub.add_parser(
        "export-csv",
        help="Write key sweep CSVs (sku/dtype/ctx/sram/tp/kv-fabric/moe/mla/video/protein/compare)",
    )
    csv_p.add_argument(
        "--out",
        type=str,
        default=str(_REPO_OUT) + "/",
        help="output directory for CSV files",
    )

    util_p = sub.add_parser("util-curve", help="Print M=1..R utilization curve")
    util_p.add_argument("--pe", type=int, nargs=2, default=[64, 64], metavar=("R", "C"))
    util_p.add_argument("--engines", type=int, default=1)
    util_p.add_argument("--k", type=int, default=64)
    util_p.add_argument("--n", type=int, default=64)

    sub.add_parser("handcheck", help="Run toy hand-check config (see examples/handcheck.md)")


    video_p = sub.add_parser(
        "sweep-video",
        help="Sweep DiT-video N_denoise × HBM/LPDDR @ sku_100t; TTFC + frames/s",
    )
    video_p.add_argument(
        "--shape",
        choices=["dit_video", "large", "toy"],
        default="dit_video",
        help="video shape: dit_video (default) | large (illustrative_large_dit) | toy",
    )
    video_p.add_argument(
        "--denoise-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_VIDEO_DENOISE_SWEEP),
        help=f"N_denoise steps (default: {' '.join(map(str, DEFAULT_VIDEO_DENOISE_SWEEP))})",
    )
    video_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    video_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    prot_p = sub.add_parser(
        "sweep-protein",
        help="Sweep protein L_aa × HBM/LPDDR @ sku_100t; pair L^2 + OOM flags",
    )
    prot_p.add_argument(
        "--seq-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROTEIN_L_SWEEP),
        help=f"sequence lengths (default: {' '.join(map(str, DEFAULT_PROTEIN_L_SWEEP))})",
    )
    prot_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    prot_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    cmp_p = sub.add_parser(
        "compare-domains",
        help="Cross-domain one-pager: LLM TPOT / video TTFC / protein t/seq @ sku_100t × HBM/LPDDR",
    )
    cmp_p.add_argument(
        "--sku",
        choices=sorted(SKU_REGISTRY.keys()),
        default="sku_100t",
    )
    cmp_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    cmp_p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="optional directory to also write compare_domains.csv",
    )

    sub.add_parser("list-shapes", help="List LLM / video / protein illustrative shapes")

    list_p = sub.add_parser("list-sku", help="Print SKU preset peak math")
    _ = list_p  # noqa: F841 — registered for side effect

    lp_pkg = sub.add_parser(
        "list-packages",
        help="LPDDR/HBM package geometry options (assumed DSE ranges) + peak GB/s",
    )
    lp_pkg.add_argument(
        "--kind",
        choices=["hbm", "lpddr", "all", "lpddr5", "lpddr5x", "lpddr6",
                 "hbm3", "hbm3e", "hbm4", "hbm4e"],
        default="all",
        help="filter by memory kind or type (v0.29)",
    )

    lp_cmp = sub.add_parser(
        "list-compute",
        help="Core (4–16T) / cluster (64–256T) compute options [assumed 1 GHz]",
    )
    lp_cmp.add_argument(
        "--level",
        choices=["core", "cluster", "all"],
        default="all",
        help="filter core vs cluster (default: primaries of both)",
    )
    lp_cmp.add_argument(
        "--all-aliases",
        action="store_true",
        dest="all_aliases",
        help="include cluster decomposition aliases",
    )

    list_d = sub.add_parser("list-dtype", help="Print dtype presets (storage bits)")
    _ = list_d

    list_q = sub.add_parser("list-quant", help="Print W/KV quant presets (independent bits)")
    _ = list_q

    wb_p = sub.add_parser(
        "workbench",
        help="Unified WorkbenchConfig → MetricsCard (product layer)",
    )
    wb_p.add_argument(
        "--model",
        default="illustrative_27B",
        help="series/shape id (e.g. illustrative_27B, series/dense-27b)",
    )
    wb_p.add_argument("--chips", type=int, default=1, help="chip_count (≥1); default tp=chips")
    wb_p.add_argument("--tp", type=int, default=None, help="explicit TP (requires --pp/--ep product == chips)")
    wb_p.add_argument("--pp", type=int, default=1)
    wb_p.add_argument("--ep", type=int, default=1)
    wb_p.add_argument("--mem", choices=["hbm", "lpddr"], default="hbm")
    wb_p.add_argument("--cores", type=int, default=16, help="n_cores (product language)")
    wb_p.add_argument("--tops-per-core", type=float, default=6.25, dest="tops_per_core")
    wb_p.add_argument("--pe", type=int, nargs=2, default=None, metavar=("R", "C"))
    wb_p.add_argument("--engines", type=int, default=None)
    wb_p.add_argument("--sku", choices=sorted(SKU_REGISTRY.keys()), default=None)
    wb_p.add_argument("--freq-ghz", type=float, default=1.0, dest="freq_ghz")
    wb_p.add_argument("--prompt", type=int, default=512)
    wb_p.add_argument("--ctx", type=int, default=512)
    wb_p.add_argument("--batch", type=int, default=1)
    wb_p.add_argument("--sram-mib", type=float, default=64.0, dest="sram_mib")
    wb_p.add_argument("--dtype", choices=["fp16", "fp8", "e4m3", "int8", "int4"], default=None)
    wb_p.add_argument("--quant", choices=["w16k16", "w8k16", "w4k16", "w8k8", "w4k8"], default=None)
    wb_p.add_argument("--n-packages", type=int, default=None, dest="n_packages")
    wb_p.add_argument("--n-channels", type=int, default=None, dest="n_channels")
    wb_p.add_argument("--n-ranks", type=int, default=None, dest="n_ranks")
    wb_p.add_argument("--data-rate-gts", type=float, default=None, dest="data_rate_gts")
    wb_p.add_argument("--width-bits", type=int, default=None, dest="width_bits")
    wb_p.add_argument("--efficiency", type=float, default=None)
    wb_p.add_argument(
        "--mem-efficiency",
        type=float,
        default=None,
        dest="mem_efficiency",
        help="alias for --efficiency (DRAM sustained/peak)",
    )
    wb_p.add_argument(
        "--weight-hide",
        type=float,
        default=None,
        dest="weight_hide",
        help="assumed weight hide / mac_efficiency overlap (0..0.9)",
    )
    wb_p.add_argument(
        "--calib",
        type=str,
        default=None,
        help="JSON CalibrationOverrides path (mem_efficiency/mac_efficiency/frequency_hz)",
    )
    wb_p.add_argument("--capacity-gb", type=float, default=None, dest="capacity_gb",
                      help="per-card capacity in vendor GB (= 2^30 B, v0.30)")
    _add_struct_mem_args(wb_p)
    _add_sync_kv_args(wb_p)
    wb_p.add_argument(
        "--preset",
        default=None,
        help="scenario preset (edge-lpddr-4x64 | card-hbm-4stack | scaleup-8chip); "
        "explicit --mem/--n-*/--cores/--chips flags still win",
    )
    _add_econ_args(wb_p)
    _add_assumed_compute_args(wb_p)
    wb_p.add_argument("--kv-fabric", default="none", dest="kv_fabric")
    wb_p.add_argument("--c2c-gbps", type=float, default=400.0, dest="c2c_gbps")
    wb_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    wb_p.add_argument(
        "--json",
        type=str,
        default=None,
        dest="json_out",
        help="write MetricsCard JSON to path",
    )
    wb_p.add_argument(
        "--md",
        type=str,
        default=None,
        dest="md_out",
        help="write MetricsCard Markdown to path",
    )

    wbs_p = sub.add_parser(
        "workbench-scan",
        help="Sweep chips / parallel / --sweep-mem / --sweep-package / --sweep-compute",
    )
    wbs_p.add_argument("--model", default="illustrative_27B")
    wbs_p.add_argument(
        "--chips-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHIP_SWEEP),
        dest="chips_list",
    )
    wbs_p.add_argument("--mem", choices=["hbm", "lpddr"], default="hbm")
    wbs_p.add_argument("--cores", type=int, default=16)
    wbs_p.add_argument("--tops-per-core", type=float, default=6.25, dest="tops_per_core")
    wbs_p.add_argument("--sku", choices=sorted(SKU_REGISTRY.keys()), default=None)
    wbs_p.add_argument("--prompt", type=int, default=512)
    wbs_p.add_argument("--ctx", type=int, default=512)
    wbs_p.add_argument("--batch", type=int, default=1)
    wbs_p.add_argument("--sram-mib", type=float, default=64.0, dest="sram_mib")
    wbs_p.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    wbs_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )
    wbs_p.add_argument(
        "--mode",
        choices=["chips", "parallel"],
        default="chips",
        help="chips: sweep chip counts; parallel: tp×pp×ep matrix for --chips",
    )
    wbs_p.add_argument(
        "--chips",
        type=int,
        default=8,
        help="chip count for --mode parallel (default 8)",
    )
    wbs_p.add_argument(
        "--sort-by",
        choices=["TPOT", "TTFT", "tpot", "ttft"],
        default="TPOT",
        dest="sort_by",
        help="sort parallel matrix by TPOT or TTFT",
    )
    wbs_p.add_argument(
        "--sweep-mem",
        action="store_true",
        help="sweep DRAM n_channels / data_rate geometry (BW wall flips)",
    )
    wbs_p.add_argument(
        "--mem-vary",
        choices=["channels", "rate", "both"],
        default="both",
        dest="mem_vary",
        help="which geometry knobs to vary with --sweep-mem",
    )
    wbs_p.add_argument(
        "--channels-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_MEM_CHANNEL_SWEEP),
        dest="channels_list",
    )
    wbs_p.add_argument(
        "--rate-list",
        type=float,
        nargs="+",
        default=None,
        dest="rate_list",
        help="data_rate_GTs points (default: HBM or LPDDR preset grid)",
    )

    wbs_p.add_argument(
        "--sweep-package",
        action="store_true",
        dest="sweep_package",
        help="sweep named LPDDR/HBM PackageOption catalog (assumed DSE ranges)",
    )
    wbs_p.add_argument(
        "--package-list",
        type=str,
        nargs="+",
        default=None,
        dest="package_list",
        help="package ids (default: LPDDR 2/4/8×64 + LPDDR6 sample + HBM 2/4/8 stacks × gens)",
    )
    wbs_p.add_argument(
        "--package-kind",
        choices=["hbm", "lpddr", "all"],
        default="all",
        dest="package_kind",
        help="filter --sweep-package by kind when --package-list omitted",
    )
    wbs_p.add_argument(
        "--package-axis",
        choices=["type", "count", "rate"],
        default=None,
        dest="package_axis",
        help="v0.29 structured sweep around the base memory (--mem-type …): "
        "all types / unit counts / rate grades of that generation",
    )
    _add_struct_mem_args(wbs_p)
    _add_sync_kv_args(wbs_p)
    wbs_p.add_argument(
        "--sweep-compute",
        action="store_true",
        dest="sweep_compute",
        help="sweep core 4–16T and/or cluster 64–256T compute catalog",
    )
    wbs_p.add_argument(
        "--compute-list",
        type=str,
        nargs="+",
        default=None,
        dest="compute_list",
        help="compute ids (default: core_4t..16t + cluster_64t..256t)",
    )
    wbs_p.add_argument(
        "--compute-level",
        choices=["core", "cluster", "all"],
        default="all",
        dest="compute_level",
        help="filter --sweep-compute level when --compute-list omitted",
    )

    wbp_p = sub.add_parser(
        "workbench-parallel",
        help="Enumerate tp×pp×ep = --chips; MetricsCard table sorted by TPOT/TTFT",
    )
    wbp_p.add_argument("--model", default="illustrative_27B")
    wbp_p.add_argument("--chips", type=int, default=8)
    wbp_p.add_argument("--mem", choices=["hbm", "lpddr"], default="hbm")
    wbp_p.add_argument("--cores", type=int, default=16)
    wbp_p.add_argument("--tops-per-core", type=float, default=6.25, dest="tops_per_core")
    wbp_p.add_argument("--sku", choices=sorted(SKU_REGISTRY.keys()), default=None)
    wbp_p.add_argument("--prompt", type=int, default=512)
    wbp_p.add_argument("--ctx", type=int, default=512)
    wbp_p.add_argument("--batch", type=int, default=1)
    wbp_p.add_argument("--sram-mib", type=float, default=64.0, dest="sram_mib")
    wbp_p.add_argument(
        "--sort-by",
        choices=["TPOT", "TTFT", "tpot", "ttft"],
        default="TPOT",
        dest="sort_by",
    )
    wbp_p.add_argument("--csv", type=str, default=None)
    wbp_p.add_argument(
        "--contention",
        choices=["serialize", "share_fair", "lower_bound"],
        default="serialize",
    )

    par_p = sub.add_parser(
        "pareto",
        help="v0.30 throughput–interactivity Pareto (batch × layouts, KV-capacity "
        "limited) + SLO goodput; --csv / --json export",
    )
    par_p.add_argument("--model", default="qwen3-32b")
    par_p.add_argument("--chips", type=int, default=8)
    par_p.add_argument("--mem", choices=["hbm", "lpddr"], default="hbm")
    par_p.add_argument("--cores", type=int, default=16)
    par_p.add_argument("--tops-per-core", type=float, default=6.25, dest="tops_per_core")
    par_p.add_argument("--sku", choices=sorted(SKU_REGISTRY.keys()), default=None)
    par_p.add_argument("--prompt", type=int, default=512)
    par_p.add_argument("--ctx", type=int, default=512, help="decode context (KV length)")
    par_p.add_argument("--batch", type=int, default=1, help="current-scenario batch (marked)")
    par_p.add_argument("--slo-ttft-ms", type=float, default=None, dest="slo_ttft_ms",
                       help="LLM TTFT SLO (default 2000)")
    par_p.add_argument("--slo-tpot-ms", type=float, default=None, dest="slo_tpot_ms",
                       help="LLM TPOT SLO (default 50)")
    par_p.add_argument("--slo-latency-ms", type=float, default=None, dest="slo_latency_ms",
                       help="video TTFC / protein time-per-seq SLO")
    par_p.add_argument("--layouts", choices=["all", "current"], default="all")
    par_p.add_argument("--batch-limit", type=int, default=None, dest="batch_limit")
    par_p.add_argument("--goodput-mode", choices=["amortized", "upper"], default="amortized",
                       dest="goodput_mode",
                       help="amortized (prefill deducted, steady state; default) | upper (decode only, ≤0.30)")
    par_p.add_argument("--prefill-mode", choices=["chunked", "exclusive"], default="chunked",
                       dest="prefill_mode", help="amortized: chunked/mixed (default) | exclusive")
    par_p.add_argument("--out-len", type=int, default=256, dest="out_len",
                       help="output tokens per request N (prefill amortisation; default 256)")
    par_p.add_argument("--csv", type=str, default=None)
    par_p.add_argument("--json", type=str, default=None)
    _add_struct_mem_args(par_p)
    _add_sync_kv_args(par_p)

    report_p = sub.add_parser(
        "report",
        help="Write self-contained HTML Metrics/wall report (inline CSS)",
    )
    report_p.add_argument(
        "--out",
        type=str,
        default=str(_REPO_OUT / "report.html"),
        help="output HTML path (default: out/report.html)",
    )

    ls_p = sub.add_parser("list-series", help="List HF-backed + illustrative series packs")
    ls_p.add_argument(
        "--product",
        action="store_true",
        help="only real HF-backed product packs (exclude illustrative/toy)",
    )

    sub.add_parser(
        "list-presets",
        help="List scenario presets (package + compute + chips bundles; assumed)",
    )
    evp = sub.add_parser(
        "eval-presets",
        help="Evaluate all scenario presets for one model (+ optional assumed energy/cost)",
    )
    evp.add_argument("--model", default="illustrative_27B")
    evp.add_argument("--presets", nargs="+", default=None, help="subset of preset ids")
    evp.add_argument("--prompt", type=int, default=512)
    evp.add_argument("--ctx", type=int, default=512)
    evp.add_argument("--batch", type=int, default=1)
    _add_econ_args(evp)

    serve_p = sub.add_parser(
        "serve",
        help="Launch interactive workbench web UI (local API + static knobs dashboard)",
    )
    serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host (default 127.0.0.1)",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="bind port (default 8765)",
    )
    serve_p.add_argument(
        "--stdlib",
        action="store_true",
        help="force stdlib http.server (skip FastAPI/uvicorn even if installed)",
    )

    return p


def _run_dtype(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_dtype_sweep(verbose=True, cfg=cfg, sku=sku)
    return 0


def _run_ctx(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=512,
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_ctx_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        ctx_list=list(args.ctx_list),
    )
    return 0


def _run_sram(args: argparse.Namespace) -> int:
    from .dtype import get_dtype
    from .model_shape import ILLUSTRATIVE_27B

    sku = get_sku(args.sku)
    shape = get_dtype(getattr(args, "dtype", "fp16")).apply(ILLUSTRATIVE_27B)
    policy = getattr(args, "sram_policy", "weight_resident")
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
        weight_hide_factor=getattr(args, "weight_hide", 0.0),
        sram_policy=policy,
    )
    run_sram_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        shape=shape,
        sram_mib_list=list(args.sram_list),
        weight_hide_factor=getattr(args, "weight_hide", 0.0),
        sram_policy=policy,
    )
    return 0


def _run_quant(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=512,
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    ctx_list = list(args.ctx_list) if hasattr(args, "ctx_list") else [512, 128000]
    # Unified sweep --what quant may pass default ctx sweep; prefer short+long
    if getattr(args, "what", None) == "quant" and ctx_list == list(DEFAULT_CTX_SWEEP):
        ctx_list = [512, 128000]
    run_quant_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        ctx_list=ctx_list,
        quant_names=getattr(args, "quant", None),
    )
    return 0



def _run_batch(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    shape_name = getattr(args, "shape", "27b")
    if shape_name == "toy":
        shape = TOY_SHAPE
    elif shape_name == "moe":
        shape = ILLUSTRATIVE_MOE
    else:
        shape = ILLUSTRATIVE_27B
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_batch_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        shape=shape,
        batch_list=list(getattr(args, "batch_list", DEFAULT_BATCH_SWEEP)),
    )
    return 0


def _run_moe(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_moe_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        ep_list=list(getattr(args, "ep_list", [1])),
        c2c_gbps=getattr(args, "c2c_gbps", 400.0),
    )
    return 0


def _run_mla(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=512,
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_mla_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        ctx_list=list(args.ctx_list),
        remote_kv_frac=getattr(args, "remote_kv_frac", 1.0),
        fabric_preset=getattr(args, "fabric", "roce_200g"),
    )
    return 0


def _run_export_csv(args: argparse.Namespace) -> int:
    export_csv_bundle(args.out, verbose=True)
    return 0


def _run_tp(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_tp_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        tp_list=list(getattr(args, "tp_list", DEFAULT_TP_SWEEP)),
        c2c_gbps=getattr(args, "c2c_gbps", 400.0),
        c2c_hide=getattr(args, "c2c_hide", 0.0),
        embed_policy=getattr(args, "embed_policy", "replicate"),
    )
    return 0



def _run_kv_fabric(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=512,
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_kv_fabric_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        ctx_list=list(args.ctx_list),
        fabric_presets=list(args.fabric_list),
        remote_kv_frac=getattr(args, "remote_kv_frac", 1.0),
        pd_kv_xfer=not getattr(args, "no_pd_xfer", False),
        tp=getattr(args, "tp", 1),
        c2c_gbps=getattr(args, "c2c_gbps", 400.0),
    )
    return 0


def _run_parallel(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_parallel_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        tp_list=list(getattr(args, "tp_list", [1, 2, 4])),
        pp_list=list(getattr(args, "pp_list", DEFAULT_PP_SWEEP)),
        c2c_gbps=getattr(args, "c2c_gbps", 400.0),
        c2c_hide=getattr(args, "c2c_hide", 0.0),
        embed_policy=getattr(args, "embed_policy", "replicate"),
        decode_mb=getattr(args, "decode_mb", 1),
        prefill_mb=getattr(args, "prefill_mb", None),
    )
    return 0


def _run_pp(args: argparse.Namespace) -> int:
    sku = get_sku(args.sku)
    cfg = EvalConfig(
        prompt_len=args.prompt,
        decode_seq_len=getattr(args, "ctx", 512),
        contention_mode=args.contention,
        frequency_hz=sku.frequency_hz,
    )
    run_pp_sweep(
        verbose=True,
        cfg=cfg,
        sku=sku,
        tp=getattr(args, "tp", 1),
        pp_list=list(getattr(args, "pp_list", DEFAULT_PP_SWEEP)),
        c2c_gbps=getattr(args, "c2c_gbps", 400.0),
        c2c_hide=getattr(args, "c2c_hide", 0.0),
        embed_policy=getattr(args, "embed_policy", "replicate"),
        decode_mb=getattr(args, "decode_mb", 1),
        prefill_mb=getattr(args, "prefill_mb", None),
    )
    return 0



def _add_assumed_compute_args(sp: argparse.ArgumentParser) -> None:
    """v0.26 opt-in assumed non-GEMM overhead + dtype MAC factors (default off)."""
    g = sp.add_argument_group(
        "assumed compute knobs (v0.26 — default off / 1.0; not silicon)"
    )
    g.add_argument(
        "--non-gemm-overhead",
        type=float,
        default=0.0,
        dest="non_gemm_overhead",
        help="fraction of GEMM compute time for Softmax/RoPE/LN/misc [0..1]; default 0=off",
    )
    g.add_argument("--softmax-frac", type=float, default=None, dest="softmax_frac",
                   help="optional finer Softmax fraction (with rope/norm replaces coarse)")
    g.add_argument("--rope-frac", type=float, default=None, dest="rope_frac",
                   help="optional finer RoPE fraction")
    g.add_argument("--norm-frac", type=float, default=None, dest="norm_frac",
                   help="optional finer LN/RMSNorm fraction")
    g.add_argument(
        "--dtype-mac-factor",
        type=str,
        default=None,
        dest="dtype_mac_factor_json",
        help="JSON map of assumed MAC peak factors, e.g. {\"int4\":4,\"fp8\":2} (default all 1.0 = bytes-only; user/assumed, not silicon)",
    )


def _assumed_compute_kwargs_from_args(args: argparse.Namespace) -> dict:
    import json as _json
    kw: dict = {}
    oh = getattr(args, "non_gemm_overhead", None)
    if oh is not None and float(oh) != 0.0:
        kw["non_gemm_overhead"] = float(oh)
    for name in ("softmax_frac", "rope_frac", "norm_frac"):
        v = getattr(args, name, None)
        if v is not None:
            kw[name] = float(v)
    raw = getattr(args, "dtype_mac_factor_json", None)
    if raw:
        s = str(raw).strip()
        if s.startswith("{"):
            data = _json.loads(s)
        else:
            data = _json.loads(Path(s).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("--dtype-mac-factor must be a JSON object")
        kw["dtype_mac_factors"] = {str(k): float(v) for k, v in data.items()}
    return kw


STRUCT_MEM_FLAGS = (
    "--package", "--mem-type", "--mem-form", "--mem-width", "--mem-rate",
    "--mem-count", "--mem-cap-gb", "--hbm-height", "--hbm-die-gb",
)


def _add_struct_mem_args(p: argparse.ArgumentParser) -> None:
    """v0.29 structured memory selectors (mem_catalog.make_spec)."""
    g = p.add_argument_group("structured memory (v0.29; wins over --mem/--n-* geometry)")
    g.add_argument("--package", default=None, dest="package_id",
                   help="catalog id (lpddr6_4x96_10667_16g, hbm3e_8s_12h24g_9200) or "
                   "legacy ≤0.28 id (lpddr_4x64_8533, hbm_hbm3e_4s)")
    g.add_argument("--mem-type", default=None, dest="mem_type",
                   choices=["LPDDR5", "LPDDR5X", "LPDDR6", "HBM3", "HBM3E", "HBM4", "HBM4E",
                            "lpddr5", "lpddr5x", "lpddr6", "hbm3", "hbm3e", "hbm4", "hbm4e"])
    g.add_argument("--mem-form", default=None, dest="mem_form",
                   help="LPDDR form: discrete | SOCAMM2 | LPCAMM2")
    g.add_argument("--mem-width", type=int, default=None, dest="mem_width_bits",
                   help="package / module width bits (x32/x64/x96/x48/128)")
    g.add_argument("--mem-rate", type=float, default=None, dest="mem_rate_MTps",
                   help="MT/s (8533) or GT/s (8.533); snapped to the catalog grade")
    g.add_argument("--mem-count", type=int, default=None, dest="mem_count",
                   help="packages / modules / HBM stacks")
    g.add_argument("--mem-cap-gb", type=float, default=None, dest="mem_cap_GB",
                   help="GB per LPDDR package / module")
    g.add_argument("--hbm-height", type=int, default=None, dest="hbm_height")
    g.add_argument("--hbm-die-gb", type=int, default=None, dest="hbm_die_Gb",
                   help="HBM die density in Gb (16/24/32)")


def _add_sync_kv_args(p: argparse.ArgumentParser) -> None:
    """v0.29 optimism-fix knobs (assumed)."""
    g = p.add_argument_group("sync / KV sharding (v0.29, assumed)")
    g.add_argument("--c2c-latency-us", type=float, default=None, dest="c2c_latency_us",
                   help="per-collective latency α in µs (default 3; 0 = ≤0.28 pure-BW)")
    g.add_argument("--sync-overlap", type=float, default=None, dest="sync_overlap",
                   help="fraction of α hidden (default 0 = fully exposed)")
    g.add_argument("--attn-parallel", choices=["tp", "dp"], default=None,
                   dest="attn_parallel",
                   help="attention parallel: tp (KV by head; MLA replicated) | dp (KV by batch)")
    g2 = p.add_argument_group("engine v0.31 (MoE sharding / PP decode micro-batches / spec decode)")
    g2.add_argument("--moe-shard", choices=["tp_ep", "ep_all"], default=None, dest="moe_shard",
                    help="MoE experts: tp_ep (over ep groups, TP-split by tp; default) | "
                    "ep_all (over all tp·ep ranks)")
    g2.add_argument("--pp-decode-mb", type=int, default=None, dest="pp_decode_mb",
                    help="PP decode micro-batches (0 = auto min(B, pp))")
    g2.add_argument("--spec-k", type=int, default=None, dest="spec_k",
                    help="speculative draft tokens per step (0 = off)")
    g2.add_argument("--spec-accept", type=float, default=None, dest="spec_accept",
                    help="per-token acceptance rate a (assumed; default 0.7)")
    g2.add_argument("--spec-draft", choices=["mtp", "model"], default=None, dest="spec_draft",
                    help="draft cost: mtp (one layer + LM head per draft token) | model")
    g2.add_argument("--spec-draft-frac", type=float, default=None, dest="spec_draft_frac",
                    help="draft=model: draft step cost as a fraction of the target step")


def _struct_mem_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Structured memory + sync/KV kwargs for WorkbenchConfig (only given flags)."""
    kw: dict = {}
    pid = getattr(args, "package_id", None)
    if pid:
        kw.update(get_package(pid).workbench_kwargs())
    for k in ("mem_form", "mem_width_bits", "mem_rate_MTps", "mem_count",
              "mem_cap_GB", "hbm_height", "hbm_die_Gb"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v
    t = getattr(args, "mem_type", None)
    if t:
        t = t.upper()
        if kw.get("mem_type") and kw["mem_type"] != t:
            for k in ("mem_form", "mem_width_bits", "mem_rate_MTps", "mem_count",
                      "mem_cap_GB", "hbm_height", "hbm_die_Gb"):
                if getattr(args, k, None) is None:
                    kw.pop(k, None)
        kw["mem_type"] = t
    if kw.get("mem_type"):
        from .mem_catalog import mem_kind_of

        r = kw.get("mem_rate_MTps")
        if r is not None and float(r) < 100:
            kw["mem_rate_MTps"] = float(r) * 1000.0
        kw["mem_kind"] = mem_kind_of(kw["mem_type"])
        for k in ("n_channels", "width_bits", "data_rate_GTs", "capacity_GB", "n_packages"):
            kw[k] = None
    elif any(getattr(args, k, None) is not None for k in (
        "mem_form", "mem_width_bits", "mem_rate_MTps", "mem_count", "mem_cap_GB",
        "hbm_height", "hbm_die_Gb",
    )):
        raise SystemExit("structured memory flags need --mem-type or --package")
    for k in ("c2c_latency_us", "sync_overlap", "attn_parallel",
              "moe_shard", "spec_k", "spec_accept", "spec_draft", "spec_draft_frac"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v
    if getattr(args, "pp_decode_mb", None) is not None:
        kw["decode_mb"] = int(args.pp_decode_mb)
    return kw


def _add_econ_args(sp: argparse.ArgumentParser) -> None:
    """v0.24 assumed energy / cost stub flags (all default None → off)."""
    g = sp.add_argument_group("energy / cost stub (ASSUMED — not silicon)")
    g.add_argument("--tdp-w", type=float, default=None, dest="tdp_w",
                   help="per-card power basis in W (assumed; wins over --watts-per-tops)")
    g.add_argument("--watts-per-tops", type=float, default=None, dest="watts_per_tops",
                   help="per-card W per peak TOPS (assumed)")
    g.add_argument("--power-util", type=float, default=None, dest="power_util",
                   help="avg/peak power factor (default 1.0 = TDP upper bound)")
    g.add_argument("--cost-per-card", type=float, default=None, dest="cost_per_card_usd",
                   help="USD per card (assumed)")
    g.add_argument("--mem-addon-usd", type=float, default=None, dest="mem_addon_usd",
                   help="USD per card memory package add-on (assumed)")
    g.add_argument("--usd-per-kwh", type=float, default=None, dest="usd_per_kwh",
                   help="electricity price for LLM $/MTok energy part (assumed)")
    g.add_argument("--amortize-years", type=float, default=None, dest="amortize_years",
                   help="capex amortization for LLM $/MTok (assumed)")
    g.add_argument("--duty-cycle", type=float, default=None, dest="duty_cycle",
                   help="serving duty cycle for capex amortization (default 1.0)")
    g.add_argument("--econ", type=str, default=None,
                   help="JSON energy/cost knobs (see examples/energy_cost.example.json)")


def _econ_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Merge --econ JSON then explicit flags (flags win)."""
    from .econ import ECON_FIELD_NAMES, EnergyCostKnobs

    kw: dict = {}
    path = getattr(args, "econ", None)
    if path:
        knobs = EnergyCostKnobs.load_json(path)
        kw = knobs.apply_to_kwargs(kw)
        print(f"[econ] loaded {path} active={knobs.active_fields()} [ASSUMED placeholders]")
    for n in ECON_FIELD_NAMES:
        v = getattr(args, n, None)
        if v is not None:
            kw[n] = float(v)
    return kw


def _flag_given(argv: list[str], *flags: str) -> bool:
    for a in argv:
        for f in flags:
            if a == f or a.startswith(f + "="):
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    # Default to scan if no subcommand
    if not argv:
        argv = ["scan"]
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "scan" or args.cmd is None:
        cfg = EvalConfig(
            prompt_len=getattr(args, "prompt", 512),
            decode_seq_len=getattr(args, "ctx", 512),
            frequency_hz=getattr(args, "freq_ghz", 1.0) * 1e9,
            contention_mode=getattr(args, "contention", "serialize"),
        )
        run_default_scan(verbose=True, cfg=cfg)
        return 0

    if args.cmd == "scan-sku":
        cfg = EvalConfig(
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            contention_mode=args.contention,
        )
        run_sku_scan(verbose=True, cfg=cfg)
        return 0

    if args.cmd == "sweep-dtype":
        return _run_dtype(args)

    if args.cmd == "sweep-ctx":
        return _run_ctx(args)

    if args.cmd == "sweep-sram":
        return _run_sram(args)

    if args.cmd == "sweep-quant":
        return _run_quant(args)

    if args.cmd == "sweep-batch":
        return _run_batch(args)

    if args.cmd == "sweep-moe":
        return _run_moe(args)

    if args.cmd == "sweep-mla":
        return _run_mla(args)

    if args.cmd == "export-csv":
        return _run_export_csv(args)

    if args.cmd in ("sweep-tp", "scan-scaleup"):
        return _run_tp(args)

    if args.cmd == "sweep-kv-fabric":
        return _run_kv_fabric(args)

    if args.cmd == "sweep-parallel":
        return _run_parallel(args)

    if args.cmd == "sweep-pp":
        return _run_pp(args)

    if args.cmd == "sweep":
        if args.what == "dtype":
            return _run_dtype(args)
        if args.what == "ctx":
            return _run_ctx(args)
        if args.what == "sram":
            return _run_sram(args)
        if args.what == "batch":
            return _run_batch(args)
        if args.what == "moe":
            return _run_moe(args)
        if args.what == "mla":
            return _run_mla(args)
        return _run_quant(args)

    if args.cmd == "sweep-video":
        from .workloads import (
            ILLUSTRATIVE_DIT_VIDEO,
            ILLUSTRATIVE_LARGE_DIT,
            TOY_VIDEO,
        )
        sku = get_sku(args.sku)
        cfg = EvalConfig(
            contention_mode=args.contention,
            frequency_hz=sku.frequency_hz,
        )
        shape_key = getattr(args, "shape", "dit_video")
        if shape_key == "large":
            vshape = ILLUSTRATIVE_LARGE_DIT
        elif shape_key == "toy":
            vshape = TOY_VIDEO
        else:
            vshape = ILLUSTRATIVE_DIT_VIDEO
        run_video_sweep(
            verbose=True,
            cfg=cfg,
            sku=sku,
            denoise_list=list(args.denoise_list),
            shape=vshape,
        )
        return 0

    if args.cmd == "sweep-protein":
        sku = get_sku(args.sku)
        cfg = EvalConfig(
            contention_mode=args.contention,
            frequency_hz=sku.frequency_hz,
        )
        run_protein_sweep(
            verbose=True,
            cfg=cfg,
            sku=sku,
            seq_list=list(args.seq_list),
        )
        return 0

    if args.cmd == "compare-domains":
        sku = get_sku(args.sku)
        cfg = EvalConfig(
            prompt_len=512,
            decode_seq_len=512,
            batch=1,
            contention_mode=args.contention,
            frequency_hz=sku.frequency_hz,
        )
        run_compare_domains(verbose=True, cfg=cfg, sku=sku)
        if getattr(args, "csv", None):
            export_compare_domains_csv(args.csv, verbose=True)
        return 0

    if args.cmd == "list-shapes":
        from .workloads import list_shapes
        print(f"{'domain':8s}  {'name':28s}  summary")
        print("-" * 120)
        for domain, name, summary in list_shapes():
            print(f"{domain:8s}  {name:28s}")
            print(f"          {summary}")
        return 0

    if args.cmd == "list-sku":
        for s in SKU_REGISTRY.values():
            print(s.summary())
        return 0

    if args.cmd == "list-packages":
        kind = None if args.kind == "all" else args.kind.upper()
        pkgs = list_packages(kind=kind)
        print(format_package_table(pkgs))
        counts = package_counts()
        print(
            f"catalog: LPDDR={counts['lpddr']} HBM={counts['hbm']} "
            f"total={counts['total']} [assumed / uncalibrated]"
        )
        return 0

    if args.cmd == "list-compute":
        from .package_ranges import list_compute as _list_compute

        if args.all_aliases:
            opts = _list_compute(level=args.level)
        elif args.level == "all":
            opts = list_compute_primaries()
        else:
            opts = [c for c in list_compute_primaries() if c.level == args.level]
        print(format_compute_table(opts, primaries_only=False))
        return 0

    if args.cmd == "list-dtype":
        from .dtype import DTYPE_REGISTRY

        for d in DTYPE_REGISTRY.values():
            print(d.summary())
        return 0

    if args.cmd == "list-quant":
        from .dtype import QUANT_REGISTRY

        for q in QUANT_REGISTRY.values():
            print(q.summary())
        return 0

    if args.cmd == "eval":
        if args.shape == "toy":
            shape = TOY_SHAPE
        elif args.shape == "moe":
            shape = ILLUSTRATIVE_MOE
        elif args.shape == "mla":
            shape = ILLUSTRATIVE_MLA
        else:
            shape = ILLUSTRATIVE_27B
        if args.dtype:
            from .dtype import get_dtype

            shape = get_dtype(args.dtype).apply(shape)
        if getattr(args, "quant", None):
            from .dtype import get_quant

            shape = get_quant(args.quant).apply(shape)
        if args.weight_bits is not None or args.kv_bits is not None:
            shape = shape.with_bits(
                weight_bits=args.weight_bits,
                kv_bits=args.kv_bits,
            )
        mem = HBM_PRESET if args.mem == "hbm" else LPDDR_PRESET
        sram = SRAMConfig(int(args.sram_mib * 1024 * 1024))
        if args.sku:
            sku = get_sku(args.sku)
            npu = sku.npu()
            freq = sku.frequency_hz
        else:
            pe = args.pe or [64, 64]
            npu = NPUConfig(rows=pe[0], cols=pe[1], n_engines=args.engines)
            freq = args.freq_ghz * 1e9
        cfg = EvalConfig(
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            frequency_hz=freq,
            contention_mode=args.contention,
            pin_weights=args.pin_weights,
            weight_hide_factor=getattr(args, "weight_hide", 0.0),
            sram_policy=getattr(args, "sram_policy", "weight_resident"),
        )
        r = evaluate_inference(shape, npu, sram, mem, cfg)
        for line in r.summary_lines():
            print(line)
        return 0

    if args.cmd == "util-curve":
        npu = NPUConfig(rows=args.pe[0], cols=args.pe[1], n_engines=args.engines)
        ms = list(range(1, npu.rows + 1)) + [npu.rows * 2, npu.rows * 4]
        print(
            f"OS util curve: PE={npu.rows}x{npu.cols}x{npu.n_engines} "
            f"eff_cols={npu.effective_cols} K={args.k} N={args.n}"
        )
        print(f"{'M':>6}  {'util':>8}  note")
        for m, u in utilization_vs_m(ms, args.k, args.n, npu):
            note = ""
            if m == 1:
                note = f"  decode M=1; ideal={1/npu.rows:.4f} if N%effC==0"
            if m == npu.rows:
                note = "  M fills rows"
            print(f"{m:6d}  {u:8.4f}{note}")
        u1 = gemm_utilization(1, args.k, args.n, npu)
        uL = gemm_utilization(npu.rows * 4, args.k, args.n, npu)
        print(f"check: util(M=1)={u1:.6f} < util(M={npu.rows*4})={uL:.6f}  [{u1 < uL}]")
        return 0

    if args.cmd == "handcheck":
        from .memory import HANDCHECK_MEM
        npu = NPUConfig(rows=4, cols=4)
        sram = SRAMConfig(128 * 1024)
        cfg = EvalConfig(
            prompt_len=8,
            decode_seq_len=8,
            batch=1,
            frequency_hz=1e9,
            pin_weights=False,
            contention_mode="serialize",
        )
        r = evaluate_inference(TOY_SHAPE, npu, sram, HANDCHECK_MEM, cfg)
        print("Hand-check config (examples/handcheck.md): PE=4x4, SRAM=128KiB, BW=16GB/s")
        for line in r.summary_lines():
            print(line)
        return 0

    if args.cmd == "list-presets":
        from .scenarios import format_preset_table, list_presets

        print(format_preset_table())
        print()
        for pre in list_presets():
            print("  " + pre.summary())
        print("\n(presets bundle catalog package + compute + chips only; no power/price "
              "numbers — set --tdp-w / --cost-per-card or --econ yourself) [assumed]")
        return 0

    if args.cmd == "eval-presets":
        from .scenarios import evaluate_presets, format_preset_eval_table

        econ_kw = _econ_kwargs_from_args(args)
        rows = evaluate_presets(
            args.model,
            preset_ids=args.presets,
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            batch=args.batch,
            **econ_kw,
        )
        print(f"eval-presets model={args.model} econ={'on' if econ_kw else 'off'}")
        print(format_preset_eval_table(rows))
        return 0

    if args.cmd == "workbench":
        parallel = None
        if getattr(args, "tp", None) is not None:
            parallel = ParallelOverride(tp=args.tp, pp=args.pp, ep=args.ep)
        pe_rows = pe_cols = None
        if getattr(args, "pe", None) is not None:
            pe_rows, pe_cols = args.pe[0], args.pe[1]
        eff = args.efficiency
        if getattr(args, "mem_efficiency", None) is not None:
            eff = args.mem_efficiency
        wh = getattr(args, "weight_hide", None)
        cfg = WorkbenchConfig(
            model_id=args.model,
            chip_count=args.chips,
            parallel=parallel,
            mem_kind=args.mem.upper(),
            n_cores=None if args.sku or args.pe else args.cores,
            tops_per_core=None if args.sku or args.pe else args.tops_per_core,
            pe_rows=pe_rows,
            pe_cols=pe_cols,
            n_engines=getattr(args, "engines", None),
            sku=args.sku,
            frequency_hz=args.freq_ghz * 1e9,
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            batch=args.batch,
            sram_mib=args.sram_mib,
            dtype=args.dtype,
            quant=args.quant,
            n_packages=args.n_packages,
            n_channels=args.n_channels,
            n_ranks=args.n_ranks,
            data_rate_GTs=args.data_rate_gts,
            width_bits=args.width_bits,
            efficiency=eff,
            capacity_GB=args.capacity_gb,
            kv_fabric=args.kv_fabric,
            c2c_gbps=args.c2c_gbps,
            contention_mode=args.contention,
            weight_hide_factor=float(wh) if wh is not None else 0.0,
            **_econ_kwargs_from_args(args),
            **_assumed_compute_kwargs_from_args(args),
        )
        smk = _struct_mem_kwargs_from_args(args)
        if smk:
            from dataclasses import replace as _dc_replace0

            cfg = _dc_replace0(cfg, **smk)
        if getattr(args, "preset", None):
            from dataclasses import replace as _dc_replace

            from .scenarios import get_preset

            pre = get_preset(args.preset)
            pkw = pre.workbench_kwargs()
            mem_keys = ("mem_kind", "n_channels", "width_bits", "data_rate_GTs",
                        "efficiency", "capacity_GB", "n_packages", "mem_type",
                        "mem_form", "mem_width_bits", "mem_rate_MTps", "mem_count",
                        "mem_cap_GB", "hbm_height", "hbm_die_Gb", "mem_legacy_id")
            comp_keys = ("n_cores", "tops_per_core", "frequency_hz", "sku",
                         "pe_rows", "pe_cols", "n_engines")
            if _flag_given(argv, "--mem", "--n-channels", "--width-bits",
                           "--data-rate-gts", "--n-packages", "--capacity-gb",
                           *STRUCT_MEM_FLAGS):
                for k in mem_keys:
                    pkw.pop(k, None)
            elif _flag_given(argv, "--efficiency", "--mem-efficiency"):
                pkw.pop("efficiency", None)
            if _flag_given(argv, "--cores", "--tops-per-core", "--pe", "--sku", "--engines"):
                for k in comp_keys:
                    pkw.pop(k, None)
            elif _flag_given(argv, "--freq-ghz"):
                pkw.pop("frequency_hz", None)
            if _flag_given(argv, "--chips", "--tp"):
                pkw.pop("chip_count", None)
                pkw.pop("parallel", None)
            cfg = _dc_replace(cfg, **pkw)
            print(f"[preset] {pre.summary()}")
        if getattr(args, "calib", None):
            cal = CalibrationOverrides.load_json(args.calib)
            cfg = cal.apply_to_config(cfg)
            print(
                f"[calib] loaded {args.calib} active={cal.active_fields()}"
                + (f" notes={cal.notes!r}" if cal.notes else "")
            )
        card = evaluate_workbench(cfg)
        for line in card.summary_lines():
            print(line)
        if getattr(args, "json_out", None):
            p = export_metrics_card_json(card, args.json_out)
            print(f"wrote JSON {p}")
        if getattr(args, "md_out", None):
            p = export_metrics_card_md(card, args.md_out)
            print(f"wrote Markdown {p}")
        return 0

    if args.cmd == "workbench-scan":
        common_kw = dict(
            mem_kind=args.mem.upper(),
            n_cores=None if args.sku else args.cores,
            tops_per_core=None if args.sku else args.tops_per_core,
            sku=args.sku,
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            batch=args.batch,
            sram_mib=args.sram_mib,
            contention_mode=args.contention,
        )
        smk = _struct_mem_kwargs_from_args(args)
        smk_nokind = {k: v for k, v in smk.items() if k != "mem_kind"}
        common_kw.update(smk)  # structured memory + sync/KV knobs for all scan modes
        if getattr(args, "sweep_package", False) or getattr(args, "package_axis", None):
            kind = None if getattr(args, "package_kind", "all") == "all" else args.package_kind.upper()
            rows = workbench_package_sweep(
                args.model,
                package_ids=getattr(args, "package_list", None),
                kind=kind,
                axis=getattr(args, "package_axis", None),
                chips=getattr(args, "chips", 1),
                **smk_nokind,
                n_cores=common_kw.get("n_cores"),
                tops_per_core=common_kw.get("tops_per_core"),
                sku=common_kw.get("sku"),
                prompt_len=common_kw["prompt_len"],
                decode_seq_len=common_kw["decode_seq_len"],
                batch=common_kw["batch"],
                sram_mib=common_kw["sram_mib"],
                contention_mode=common_kw["contention_mode"],
            )
            print(
                f"workbench-scan --sweep-package model={args.model} "
                f"chips={getattr(args, 'chips', 1)} "
                f"kind={getattr(args, 'package_kind', 'all')} "
                f"({len(rows)} packages) [assumed]"
            )
            print(format_package_sweep_table(rows))
            cards = [c for _, c in rows]
        elif getattr(args, "sweep_compute", False):
            rows = workbench_compute_sweep(
                args.model,
                compute_ids=getattr(args, "compute_list", None),
                level=getattr(args, "compute_level", "all"),
                chips=getattr(args, "chips", 1),
                mem_kind=smk.get("mem_kind", args.mem.upper()),
                prompt_len=common_kw["prompt_len"],
                decode_seq_len=common_kw["decode_seq_len"],
                batch=common_kw["batch"],
                sram_mib=common_kw["sram_mib"],
                contention_mode=common_kw["contention_mode"],
                **smk_nokind,
            )
            print(
                f"workbench-scan --sweep-compute model={args.model} "
                f"mem={args.mem} chips={getattr(args, 'chips', 1)} "
                f"level={getattr(args, 'compute_level', 'all')} "
                f"({len(rows)} options) [assumed]"
            )
            print(format_compute_sweep_table(rows))
            cards = [c for _, c in rows]
        elif getattr(args, "sweep_mem", False):
            cards = workbench_mem_sweep(
                args.model,
                chips=getattr(args, "chips", 1),
                vary=getattr(args, "mem_vary", "both"),
                channel_list=list(getattr(args, "channels_list", DEFAULT_MEM_CHANNEL_SWEEP)),
                rate_list=getattr(args, "rate_list", None),
                **common_kw,
            )
            print(
                f"workbench-scan --sweep-mem model={args.model} "
                f"mem={args.mem} chips={getattr(args, 'chips', 1)} "
                f"vary={getattr(args, 'mem_vary', 'both')}"
            )
            print(format_mem_sweep_table(cards))
        elif getattr(args, "mode", "chips") == "parallel":
            cards = workbench_parallel_matrix(
                args.model,
                chips=int(args.chips),
                sort_by=getattr(args, "sort_by", "TPOT"),
                **common_kw,
            )
            print(
                f"workbench-scan --mode parallel model={args.model} "
                f"chips={args.chips} mem={args.mem} "
                f"sort={getattr(args, 'sort_by', 'TPOT')}"
            )
            print(format_parallel_table(cards))
        else:
            cards = workbench_scan(
                args.model,
                chips_list=list(args.chips_list),
                **common_kw,
            )
            print(f"workbench-scan model={args.model} mem={args.mem}")
            print(format_scan_table(cards))
        if getattr(args, "csv", None):
            out = export_scan_csv(cards, args.csv)
            print(f"wrote {out}")
        return 0

    if args.cmd == "workbench-parallel":
        cards = workbench_parallel_matrix(
            args.model,
            chips=int(args.chips),
            sort_by=getattr(args, "sort_by", "TPOT"),
            mem_kind=args.mem.upper(),
            n_cores=None if args.sku else args.cores,
            tops_per_core=None if args.sku else args.tops_per_core,
            sku=args.sku,
            prompt_len=args.prompt,
            decode_seq_len=args.ctx,
            batch=args.batch,
            sram_mib=args.sram_mib,
            contention_mode=args.contention,
        )
        print(
            f"workbench-parallel model={args.model} chips={args.chips} "
            f"mem={args.mem} sort={getattr(args, 'sort_by', 'TPOT')} "
            f"({len(cards)} combos)"
        )
        print(format_parallel_table(cards))
        if getattr(args, "csv", None):
            out = export_scan_csv(cards, args.csv)
            print(f"wrote {out}")
        return 0

    if args.cmd == "pareto":
        import json as _json

        from .pareto import format_pareto_text, pareto_analysis, pareto_to_csv
        from .workbench import WorkbenchConfig as _WC

        mkw = {"mem_kind": args.mem.upper(), **_struct_mem_kwargs_from_args(args)}
        cfg = _WC(
            model_id=args.model, chip_count=int(args.chips),
            n_cores=None if args.sku else args.cores,
            tops_per_core=None if args.sku else args.tops_per_core, sku=args.sku,
            prompt_len=args.prompt, decode_seq_len=args.ctx, batch=args.batch, **mkw,
        )
        slo = {k: v for k, v in (("ttft_ms", args.slo_ttft_ms), ("tpot_ms", args.slo_tpot_ms),
                                 ("latency_ms", args.slo_latency_ms)) if v is not None}
        res = pareto_analysis(cfg, slo=slo, layouts=args.layouts, batch_limit=args.batch_limit,
                              goodput_mode=args.goodput_mode, prefill_mode=args.prefill_mode,
                              out_len=args.out_len)
        print(format_pareto_text(res))
        if args.csv:
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            Path(args.csv).write_text(pareto_to_csv(res), encoding="utf-8")
            print(f"wrote {args.csv}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(_json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
            print(f"wrote {args.json}")
        return 0

    if args.cmd == "report":
        out = write_report(args.out, verbose=True)
        print(f"report ready: {out}")
        return 0

    if args.cmd == "list-series":
        product_only = bool(getattr(args, "product", False))
        rows = list_series(product_only=product_only)
        print(f"{'id':28s}  {'family':10s}  {'domain':8s}  shape / metadata")
        print("-" * 120)
        last_section = None
        for e in rows:
            section = "HF-backed (public config)" if e.is_hf_backed else "illustrative / toy"
            if not product_only and section != last_section:
                print(f"\n## {section}")
                last_section = section
            shape_name = getattr(e.shape, "name", "?")
            print(f"{e.id:28s}  {e.family:10s}  {e.domain:8s}  → {shape_name}")
            meta = e.metadata if len(e.metadata) < 110 else e.metadata[:107] + "..."
            print(f"{'':28s}  {meta}")
        print(f"\n({len(rows)} series shown; use --product for HF-backed only)")
        return 0

    if args.cmd == "serve":
        from .serve import run_server

        run_server(
            host=getattr(args, "host", "127.0.0.1"),
            port=int(getattr(args, "port", 8765)),
            prefer_fastapi=not bool(getattr(args, "stdlib", False)),
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
