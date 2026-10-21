"""0.297: public MoE kernel timings vs the GPU reference path — is grouped-GEMM efficiency at small per-expert row
counts the missing mechanism behind the 0.296 shared signature (GPU step grows with tokens faster than modelled:
Drama 3 c64–256, LFM2.5 DSpark c8–64)?  Diagnostic only; nothing is fitted, nothing feeds the cost model.

Data 「来源」 (data/ext_kernels_moe.json):
  grouped_h800 — DeepGEMM m_grouped_gemm_fp8_fp8_bf16_nt_masked on H800, kineto kernel time, groups 1 … 18 ×
                 m 4 … 2048 rows per group (DeepSeek_Simulator @ 7c69101 results/H800_group_gemm.csv)
  module_b200  — TensorRT-LLM tech blog 24, B200, DeepSeek-V3-style NVFP4 routed MoE at TP8 (hidden 7168,
                 intermediate 256 per rank, 256 experts, top-8), TRTLLM-Gen per-expert grouped GEMM, module latency
                 (router GEMM, TopK, alpha, FC1, FC2, quantization) vs num_tokens 1 … 384
Model 「推导」 (the extval default variant's chip, the same op pricing as evaluate):
    grouped GEMM time = max(array time of `groups` GEMMs of m × k × n, operand bytes / achieved DRAM bandwidth)
    (+ the trace-sourced per-kernel floor ``kernel_us`` per kernel: 1 for a grouped GEMM; MODULE_KERNELS for the module)
    module: uniform routing (ir.moe_routing) → hit experts × m_e rows, FC1 (m_e × 7168 × 512) + FC2 (m_e × 256 × 7168)
    in NVFP4 (4 bits + one e4m3 scale per 16 「来源」 NVFP4 format → 4.5 bits), + tokens × hidden bf16 in / out.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "ext_kernels_moe.json"
MODULE_KERNELS = 6       # 「假设」 one per listed stage: router GEMM, TopK, alpha generation, FC1, FC2, quantization
NVFP4_BITS = 4.5


def data() -> dict:
    return json.loads(DATA.read_text())


@lru_cache(maxsize=8)
def _sys(hw: str, variant: str = "default"):
    from . import extval as X
    from .catalog import get_model
    from .hardware import System
    sc = X.scenario(dict(hw=hw, model="qwen3-8b", tp=1, dtype="release"), variant, phase="decode", batch=1, ctx=1024)
    return System(sc.chip, sc.mem_id), get_model("qwen3-8b")


def grouped_time(sysm, model, groups: int, m: int, k: int, n: int, fmt: str = "fp8", w_bits: float = 8.0,
                 a_bytes: float = 1.0, out_bytes: float = 2.0, act: bool = True) -> dict:
    """Array / DRAM seconds of `groups` independent m × k × n GEMMs (one grouped kernel)."""
    from .evaluate import _op_seconds
    from .ir import Op
    op = Op("e", "gemm", 0, m, k, n, count=groups, w_fmt=fmt, a_fmt=fmt, w_bits=w_bits, w_params=groups * k * n)
    ta = _op_seconds(op, sysm, "reconf", model)[0]
    by = groups * (k * n * w_bits / 8 + (m * k * a_bytes + m * n * out_bytes if act else 0.0))
    return {"array": ta, "dram": by / (sysm.dram_GBps * 1e9), "bytes": by}


def grouped_table(variant: str = "default") -> list[dict]:
    d = data()["grouped_h800"]
    sysm, model = _sys(d["hw"], variant)
    ku = (sysm.chip.kernel_us or 0.0) * 1e-6
    out = []
    for r in d["rows"]:
        t = grouped_time(sysm, model, r["groups"], r["m"], r["k"], r["n"])
        pred = max(t["array"], t["dram"])
        meas = r["time_us"] * 1e-6
        out.append({"groups": r["groups"], "m": r["m"], "n": r["n"], "k": r["k"], "meas_us": r["time_us"],
                    "pred_us": round(pred * 1e6, 2), "ratio": round(pred / meas, 3),
                    "ratio_floor": round((pred + ku) / meas, 3),
                    "meas_GBps": round(t["bytes"] / meas / 1e9), "bound": "array" if t["array"] > t["dram"] else "dram"})
    return out


def grouped_by_m(t: list[dict]) -> list[dict]:
    """Geometric-mean model / measured per rows-per-group bin (both N × K shapes pooled)."""
    out = []
    for m in sorted({x["m"] for x in t}):
        g = [x for x in t if x["m"] == m]
        gm = lambda key: round(math.exp(sum(math.log(x[key]) for x in g) / len(g)), 3)
        out.append({"m": m, "n": len(g), "ratio": gm("ratio"), "ratio_floor": gm("ratio_floor"),
                    "meas_GBps": round(sum(x["meas_GBps"] for x in g) / len(g))})
    return out


def module_table(variant: str = "default") -> list[dict]:
    from .ir import moe_routing
    d = data()["module_b200"]
    sysm, model = _sys("B200", variant)
    ku = (sysm.chip.kernel_us or 0.0) * 1e-6
    H, I, E, K = d["hidden"], d["intermediate"], d["experts"], d["top_k"]
    out = []
    for r in d["rows"]:
        T = r["tokens"]
        ro = moe_routing(E, K, T, 1)
        fc1 = grouped_time(sysm, model, ro.hit, ro.m_e, H, 2 * I, "nvfp4", NVFP4_BITS, 0.5625, 0.5625, act=False)
        fc2 = grouped_time(sysm, model, ro.hit, ro.m_e, I, H, "nvfp4", NVFP4_BITS, 0.5625, 2.0, act=False)
        io = T * H * 2 * 2 / (sysm.dram_GBps * 1e9)                     # bf16 hidden in / out
        pred = max(fc1["array"] + fc2["array"], fc1["dram"] + fc2["dram"] + io)
        meas = r["time_us"] * 1e-6
        out.append({"tokens": T, "hit": ro.hit, "m_e": ro.m_e, "meas_us": r["time_us"], "pred_us": round(pred * 1e6, 2),
                    "pred_floor_us": round((pred + MODULE_KERNELS * ku) * 1e6, 2),
                    "ratio_floor": round((pred + MODULE_KERNELS * ku) / meas, 3)})
    return out


def slopes(mt: list[dict], lo: int = 256, hi: int = 384) -> dict:
    """Marginal µs per token once every expert is hit (union saturated) 「推导」."""
    a = next(x for x in mt if x["tokens"] == lo)
    b = next(x for x in mt if x["tokens"] == hi)
    return {"tokens": f"{lo}-{hi}", "meas_us_per_token": round((b["meas_us"] - a["meas_us"]) / (hi - lo), 4),
            "pred_us_per_token": round((b["pred_us"] - a["pred_us"]) / (hi - lo), 4)}


def lfm_share(slope_us_per_token: float, top_k_ref: int = 8, top_k: int = 4, moe_layers: int = 40,
              mac_ref: float = 3 * 7168 * 256, mac: float = 3 * 2048 * 1536) -> float:
    """Upper-bound transfer of a measured post-saturation module slope to LFM2.5-24B-A2B 「近似」: per token-expert
    pair, scaled by the expert's MACs, × top-k × MoE layers → µs per token per step (every LFM layer has a MoE FFN;
    40 counts them all — an upper bound)."""
    per_pair = slope_us_per_token / top_k_ref * (mac / mac_ref)
    return per_pair * top_k * moe_layers


def summary(variant: str = "default") -> dict:
    gt, mt = grouped_table(variant), module_table(variant)
    bm = grouped_by_m(gt)
    small = [x for x in gt if x["m"] <= 128]
    big = [x for x in gt if x["m"] >= 256]
    gm = lambda g, k: round(math.exp(sum(math.log(x[k]) for x in g) / len(g)), 3)
    sl = slopes(mt)
    return {"grouped_small_m_ratio": gm(small, "ratio"), "grouped_small_m_ratio_floor": gm(small, "ratio_floor"),
            "grouped_large_m_ratio": gm(big, "ratio"), "grouped_small_m_GBps": [min(x["meas_GBps"] for x in small),
                                                                               max(x["meas_GBps"] for x in small)],
            "grouped_by_m": bm, "module_ratio_floor": [x["ratio_floor"] for x in mt], "module_slope": sl,
            "lfm_slope_bound_us_per_token": round(lfm_share(sl["meas_us_per_token"]), 2)}


@lru_cache(maxsize=8)
def _npu_sys(chip: str):
    from .. import api
    from .catalog import get_model
    from .hardware import System
    sc = api.scenario_from_body({"scenario": {"model": "lfm2.5-24b-a2b-fp8", "chip": chip,
                                              "serving": {"phase": "decode", "batch": 8}}})
    return System(sc.chip, sc.mem_id), get_model("lfm2.5-24b-a2b-fp8")


def npu_rows_curve(chip: str = "1P", groups: int = 64, k: int = 2048, n: int = 3072,
                   ms=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) -> list[dict]:
    """0.297 NPU check: the same grouped expert GEMM (LFM2.5 gate_up shape, FP8) on a catalog design — array vs
    DRAM time and the array's useful fraction as rows per expert grow 「推导」."""
    sysm, model = _npu_sys(chip)
    out = []
    for m in ms:
        t = grouped_time(sysm, model, groups, m, k, n)
        ideal = 2 * groups * m * k * n / (2 * sysm.chip.rows * sysm.chip.cols * sysm.chip.engines *
                                          (sysm.chip.formats.rate("fp8") or 1.0) * sysm.chip.freq_ghz * 1e9)
        out.append({"m": m, "array_us": round(t["array"] * 1e6, 3), "dram_us": round(t["dram"] * 1e6, 3),
                    "array_util": round(ideal / t["array"], 3) if t["array"] else 0.0,
                    "bound": "array" if t["array"] > t["dram"] else "dram"})
    return out
