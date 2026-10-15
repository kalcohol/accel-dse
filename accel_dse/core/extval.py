"""V6 — external validation against PUBLISHED measurements (0.66).

Data: accel_dse/data/ext_measurements.json (transcribed, cited per row).  Hardware: core/refhw.py (参考硬件,
datasheet peaks, nothing tuned).  Metrics and how accel-dse predicts them:

  ttft_ms            batch-B prefill of ISL tokens: ``evaluate(phase=prefill).ttft``
  static_tok_s_gpu   static batching (all B requests prefilled together, then OSL − 1 decode steps):
                     B·OSL / (TTFT(B, ISL) + Σ_steps TPOT(B, ctx)) / tp; Σ by the mean TPOT over 4 mid-point contexts
  max_tok_s_total    in-flight batching under infinite load, batch chosen by the runtime: the max over B (powers of two,
                     must fit) of B·OSL / (TTFT(B, ISL) + (OSL − 1)·mean TPOT) — a static-batch proxy (an upper bound
                     on the decode side, since in-flight batching interleaves prefill chunks with decode)
  max_tok_s_gpu      (0.91) the same max-load search ÷ the replica's cards (tp · dp): TRT-LLM ≥ 1.0 "Per GPU Output
                     Throughput"; a DEP-N column is attention-DP N + EP N (Layout tp=1, dp=N, ep=N)

Variants (``eff``):  "catalog" = DRAM efficiency of the memory catalog (0.7 「假设」), everything else as shipped;
"peak" = DRAM efficiency 1.0 (pure datasheet peak).  An explicit efficiency factor is NOT applied by default; see
``fit_eff`` for the held-out calibration study.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .catalog import get_model
from .evaluate import evaluate
from .parallel import Layout
from .hardware import Link
from .refhw import REF_HW
from .scenario import Scenario, Serving

DATA = Path(__file__).resolve().parents[1] / "data" / "ext_measurements.json"
FP8 = (("attn", "fp8"), ("mlp", "fp8"), ("expert", "fp8"), ("act", "fp8"), ("kv", "fp8"))   # 0.91: + routed experts
NVFP4 = (("attn", "nvfp4"), ("mlp", "nvfp4"), ("expert", "nvfp4"), ("act", "nvfp4"), ("kv", "fp8"))   # 0.91: + experts   # 0.69: TRT-LLM FP4 ckpt W4A4 + FP8 KV 「假设」
MXFP4 = (("kv", "fp8"),)   # 0.91: gpt-oss release checkpoint (MXFP4 experts) as published; TRT-LLM 1.2 config FP8 KV
# 0.93 (public docs, per checkpoint / hardware — not fitted):
#  * gpt-oss on Blackwell: MoE runs MXFP4 weights × MXFP8 activations (W4A8, UTCQMMA; TRT-LLM examples/models/core/
#    gpt_oss/README.md and the GPT-OSS Blackwell deployment guide); attention / router stay bf16 (released checkpoint)
#  * nvidia/DeepSeek-R1-0528-NVFP4-v2: only MoE linears + attention wo are FP4 (model card: v2 "additionally quantizes
#    the wo module"); the other attention projections keep the native FP8 block-scaled checkpoint.  wo is taken FP8
#    with the rest of attention (one role; ≈1 % of FLOPs)
MXFP4_BLACKWELL = MXFP4 + (("act_expert", "fp8"),)
DSR1_NVFP4_V2 = (("attn", "fp8"), ("mlp", "nvfp4"), ("expert", "nvfp4"), ("shared_expert", "nvfp4"), ("act", "fp8"), ("act_mlp", "nvfp4"),
                 ("act_expert", "nvfp4"), ("act_shared_expert", "nvfp4"), ("kv", "fp8"))
FMT_FIX = True     # 0.93 per-checkpoint formats above (False = 0.92 dtype-only overrides)
# 0.93: per-layer software overhead grounded on a PUBLIC kernel trace (not fitted): deepseek-ai/profile-data decode.json
# (EP128 decode step, two micro-batches): 1,734 small elementwise / norm / cast / rope / gate kernels = 7.6 ms over
# 61 layers × 2 micro-batches → 62.3 µs per layer per forward (≈ 14 kernels × 4.4 µs).  DeepSeek's own stack; diagnostic
# only (overhead_variant(base, TRACE_LAYER_US)) — not a default (MODEL §39)
TRACE_LAYER_US = 7.6e3 / 61 / 2


def _formats(r: dict) -> tuple:
    d = r["dtype"]
    if FMT_FIX and d == "mxfp4" and REF_HW[r["hw"]].chip.formats.native("mxfp4"):
        return MXFP4_BLACKWELL
    if FMT_FIX and d == "nvfp4" and r["model"] == "deepseek-r1":
        return DSR1_NVFP4_V2
    return {"fp8": FP8, "nvfp4": NVFP4, "mxfp4": MXFP4}.get(d, ())
EFF = {"catalog": None, "peak": 1.0}


# Hypothesis variants (diagnostics, never defaults of the cost model):
#   overlap "stage" = the model (a stage's compute, DRAM and link time overlap: max(...)); "none" = the sum (kernel-serial
#   execution, every op's traffic and compute exposed) — an upper bound on the stage-level roofline.
#   kv_reuse False = decode attention loads K/V once per *query* head (no GQA sharing in the kernel), i.e. KV traffic
#   and capacity × n_q / n_kv; projections unchanged.
VARIANTS = {
    "catalog": dict(eff="catalog", overlap="stage", kv_reuse=True),
    "peak": dict(eff="peak", overlap="stage", kv_reuse=True),
    "peak_serial": dict(eff="peak", overlap="none", kv_reuse=True),
    "peak_noreuse": dict(eff="peak", overlap="stage", kv_reuse=False),
    "peak_serial_noreuse": dict(eff="peak", overlap="none", kv_reuse=False),
    "catalog_class": dict(eff="catalog", overlap="class", kv_reuse=True),
    "peak_class": dict(eff="peak", overlap="class", kv_reuse=True),
    "catalog_kernel": dict(eff="catalog", overlap="kernel", kv_reuse=True),
    "peak_kernel": dict(eff="peak", overlap="kernel", kv_reuse=True),
    "catalog_serial": dict(eff="catalog", overlap="serial", kv_reuse=True),
    "peak_serial_m": dict(eff="peak", overlap="serial", kv_reuse=True),
    "catalog_tbo": dict(eff="catalog", overlap="tbo", kv_reuse=True),
    "peak_tbo": dict(eff="peak", overlap="tbo", kv_reuse=True),
}


@lru_cache(maxsize=None)
def _spec(model: str, kv_reuse: bool):
    m = get_model(model)
    if kv_reuse:
        return m
    lay = tuple(replace(L, core=replace(L.core, n_kv=L.core.n_q)) if L.core.kind == "gqa" and L.core.n_kv else L
                for L in m.layers)
    return replace(m, id=m.id + "#noreuse", layers=lay)


class _Pt:
    """ttft / tpot / fits of one evaluation under a variant."""

    def __init__(self, res, overlap):
        st = res.stages[res.heaviest_stage]
        t = st.time
        k = 1.0 if overlap != "none" or t.total <= 0 else (t.t_compute + t.t_dram + t.t_slc + t.t_link + t.t_sync) / t.total
        self.ttft, self.tpot, self.fits, self.bound = res.ttft * k, res.tpot * k, res.fits, res.bound


DATA_MOE = DATA.with_name("ext_measurements_moe.json")
DATA_R3 = DATA.with_name("ext_measurements_r3.json")
DATA_R4 = DATA.with_name("ext_measurements_r4.json")   # 0.91: TensorRT-LLM 1.2.0 perf-overview (B200 FP4 / H200 FP8)


def rows(path: Path = DATA) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def scenario(r: dict, eff: str = "catalog", **sv) -> Scenario:
    v = VARIANTS.get(eff, {})
    ov = v.get("overlap") if v.get("overlap") in ("class", "kernel", "serial", "tbo", "pipe") else "stage"
    e = v.get("eff", eff)
    h = REF_HW[r["hw"]]
    chip = h.chip if v.get("mac_eff") is None else replace(h.chip, mac_eff=v["mac_eff"])
    if v.get("split"):                 # 0.70: SMs run independent GEMM instances (Chip.split_instances)
        chip = replace(chip, split_instances=True)
    if v.get("sched"):                 # 0.91: Chip.instance_sched (reference chips ship "wide")
        chip = replace(chip, instance_sched=v["sched"])
    if v.get("kernel_us") is not None:  # 0.92: per-kernel fixed cost
        chip = replace(chip, kernel_us=v["kernel_us"])
    if v.get("fmt_ceiling") and chip.name.split("-")[0] in HOPPER:   # 0.94: Hopper only
        chip = replace(chip, fmt_ceiling=v["fmt_ceiling"])
    if v.get("pipe_cross_op") is not None:   # 0.98: exec_overlap "pipe" — GPU kernel boundaries (no cross-op prefetch)
        chip = replace(chip, pipe_cross_op=v["pipe_cross_op"])
    if v.get("sfu") and h.sfu_per_sm:  # 0.103: transcendentals on the SFU pipe (NVIDIA, CUDA Programming Guide)
        chip = replace(chip, sfu_per_cycle=h.sfu_per_sm * chip.engines)
    if v.get("fa3") and chip.name.split("-")[0].split(" ")[0] in FA3_HW:   # 0.104: FA3-style softmax/MMA overlap
        chip = replace(chip, attn_vec_overlap=True)
    if v.get("attn_sched"):            # 0.93: attention-only schedule
        chip = replace(chip, attn_sched=v["attn_sched"])
    if v.get("tbo_pf") and sv.get("phase") == "prefill" and r.get("src") in TBO_PREFILL_SRC:
        ov = "tbo"                     # 0.109: the source's engine runs prefill as two overlapped micro-batches
    mem_eff = e if isinstance(e, float) else EFF[e]
    extra = {}
    if r.get("node_cards"):            # 0.67 multi-node rows: cards per node + scale-out NIC per card
        extra = dict(node_cards=r["node_cards"], net=Link(r["net_GBps"], r.get("net_alpha_us", 5.0)))
    if r.get("spec_k"):                # 0.70: MTP as speculative decoding (published acceptance, not fitted)
        sv = {**sv, "spec_k": r["spec_k"], "spec_accept": r["spec_accept"]}
    if r.get("moe_skew"):
        sv = {**sv, "moe_skew": r["moe_skew"]}
    if v.get("rank_max"):              # 0.133 opt-in: slowest-rank random imbalance (Serving.moe_rank_max)
        sv = {**sv, "moe_rank_max": v["rank_max"]}
    extra.update(_mem_runtime(r, h) if v.get("mem_rt", MEM_RT) else {})
    if v.get("kv_eff"):                # 0.108: KV-cache streams at the decode-attention kernels' published efficiency
        extra["kv_mem_eff"] = v["kv_eff"]
    return Scenario(model=r["model"], chip=chip, mem_id=h.mem_id, mem_eff=mem_eff, link=h.link, mapping="reconf",
                    exec_overlap=ov,
                    layout=Layout(tp=r["tp"], dp=r.get("dp", 1), ep=r.get("ep", 1), etp=r.get("etp", 1)),
                    formats_override=_formats(r),
                    serving=Serving(**sv), **extra, **({"layer_overhead_us": v["layer_us"]} if v.get("layer_us") else {}))


def _ev(r, var, **sv):
    v = VARIANTS[var]
    return _Pt(evaluate(scenario(r, var, **sv), _spec(r["model"], v["kv_reuse"])), v["overlap"])


def _ttft(r, b, isl, eff):
    return _ev(r, eff, phase="prefill", batch=b, prompt=isl, out_len=max(1, r["osl"]))


def _decode(r, b, isl, osl, eff, n=4):
    res = [_ev(r, eff, phase="decode", batch=b, ctx=isl + max(1, round(osl * (k + 0.5) / n)), out_len=osl)
           for k in range(n)]
    return sum(x.tpot for x in res) / n, all(x.fits for x in res)


def _static(r, b, eff, inflight: bool = False):
    p = _ttft(r, b, r["isl"], eff)
    t, fits = _decode(r, b, r["isl"], r["osl"], eff)
    total = p.ttft + (r["osl"] - 1) * t
    pf = p.fits
    if inflight and INFLIGHT_FIT.get(eff, INFLIGHT_DEFAULT):
        # 0.92: in-flight batching prefills in chunks of ≤ CB_CHUNK tokens per iteration, so the activation working
        # set is one chunk's, not all b prompts' at once (the static proxy's time is kept: same tokens, same work).
        # KV of the b resident requests is still checked by the decode points (up to ISL + 7/8·OSL — TRT-LLM's
        # default GUARANTEED_NO_EVICT reserves each request's full length)
        q = min(r["isl"], CB_CHUNK)
        pf = _ev(r, eff, phase="prefill", batch=max(1, CB_CHUNK // q), prompt=q, out_len=max(1, r["osl"])).fits
    return b * r["osl"] / total, fits and pf, {"ttft_s": p.ttft, "tpot_s": t, "prefill_share": p.ttft / total}


def predict(r: dict, eff: str = "catalog") -> dict:
    m = r["metric"]
    if m == "ttft_ms":
        p = _ttft(r, r["batch"], r["isl"], eff)
        return {"pred": p.ttft * 1e3, "fits": p.fits, "bound": p.bound}
    if m == "static_tok_s_gpu":
        v, fits, det = _static(r, r["batch"], eff)
        return {"pred": v / r["tp"], "fits": fits, **det}
    if m == "static_latency_s":        # 0.69: one static batch end to end (vllm bench latency)
        p = _ttft(r, r["batch"], r["isl"], eff)
        t, fits = _decode(r, r["batch"], r["isl"], r["osl"], eff)
        return {"pred": p.ttft + (r["osl"] - 1) * t, "fits": fits and p.fits, "ttft_s": p.ttft, "tpot_s": t}
    if m == "static_tok_s_total":
        v, fits, det = _static(r, r["batch"], eff)
        return {"pred": v, "fits": fits, **det}
    if m == "prefill_tok_s_node":      # 0.67: B = dp · tokens_per_gpu / ISL requests prefilled together
        b = r["dp"] * r["tokens_per_gpu"] // r["isl"]
        p = _ttft(r, b, r["isl"], eff)
        return {"pred": b * r["isl"] / p.ttft / r["nodes"], "fits": p.fits, "bound": p.bound, "ttft_s": p.ttft}
    if m == "decode_tok_s_node":       # 0.67: B = dp · per-GPU batch at KV length kv_len, one token per step
        b = r["dp"] * r["per_gpu_batch"]
        p = _ev(r, eff, phase="decode", batch=b, ctx=r["kv_len"], out_len=max(1, r["osl"]))
        return {"pred": b / p.tpot / r["nodes"], "fits": p.fits, "bound": p.bound, "tpot_s": p.tpot}
    if m in ("max_tok_s_total", "max_tok_s_gpu") and VARIANTS.get(eff, {}).get("est"):
        est = estimator(r)                 # 0.105: per-source batching estimator (methodology, not fitted)
        if est["kind"] == "cb":
            best = predict_cb({**r, "cb_chunk": est["chunk"]}, eff, max_seqs=est["max_seqs"])
            return best                     # predict_cb already divides max_tok_s_gpu by the cards
    if m == "max_tok_s_total":
        best = None
        for k in range(0, 14):
            b = 2 ** k
            v, fits, det = _static(r, b, eff, inflight=True)
            if not fits:
                break
            if best is None or v > best["pred"]:
                best = {"pred": v, "fits": True, "batch": b, **det}
        return best or {"pred": float("nan"), "fits": False}
    if m == "max_tok_s_gpu":           # 0.91: same max-load search, per GPU of the replica (TRT-LLM ≥ 1.0 tables)
        best = predict({**r, "metric": "max_tok_s_total"}, eff)
        return {**best, "pred": best["pred"] / _cards(r)}
    raise ValueError(m)


def _cards(r: dict) -> int:
    return r["tp"] * r.get("dp", 1)


def table(eff: str = "catalog", path: Path = DATA) -> list[dict]:
    out = []
    for r in rows(path):
        p = predict(r, eff)
        out.append({**r, **p, "ratio": p["pred"] / r["value"]})
    return out


def summary(t: list[dict]) -> dict:
    """Per metric: n, geometric-mean ratio pred/meas, mean |log error| (as %), worst ratio."""
    out = {}
    for m in sorted({x["metric"] for x in t}):
        # 0.124: a zero / infinite ratio (prediction 0 or no fit) is counted, not logged (math.log raised ValueError)
        xs = [x["ratio"] for x in t if x["metric"] == m and x["ratio"] == x["ratio"]]
        lr = [math.log(r) for r in xs if 0 < r < math.inf]
        if not lr:
            out[m] = {"n": 0, "n_invalid": len(xs)}
            continue
        out[m] = {"n": len(lr), **({"n_invalid": len(xs) - len(lr)} if len(xs) > len(lr) else {}),
                  "gmean_ratio": math.exp(sum(lr) / len(lr)),
                  "mean_abs_err_pct": 100 * (math.exp(sum(abs(v) for v in lr) / len(lr)) - 1),
                  "min_ratio": math.exp(min(lr)), "max_ratio": math.exp(max(lr))}
    return out


def split_variant(base: str) -> str:
    """0.70: ``base`` variant with Chip.split_instances (GEMM instances spread over SMs)."""
    name = base + "+S"
    VARIANTS.setdefault(name, {**VARIANTS.get(base, {"eff": base}), "split": True})
    return name


def sched_variant(base: str, sched: str = "auto") -> str:
    """0.91: ``base`` variant with Chip.instance_sched = ``sched``.  "wide" (reference default) runs one GEMM as a single
    array spanning all SMs' columns, so N < engines·cols idles most of the chip (large prefill GEMMs ≈ 12–24 % on
    H100 / B200); "auto" also splits M / N over SM groups — how a GPU tiles CTAs."""
    name = f"{base}+{'A' if sched == 'auto' else sched}"
    VARIANTS.setdefault(name, {**VARIANTS.get(base, {"eff": base}), "sched": sched})
    return name


def gpu_variant(base: str, kernel_us: float = 0.53, sched: str = "cta", inflight: bool = True) -> str:
    """0.92: ``base`` with the GPU kernel model (instance_sched cta + kernel_us) and the in-flight memory check
    switched independently (diagnostics for the adoption study)."""
    name = f"{base}+G{sched}:{kernel_us:g}:{int(inflight)}"
    VARIANTS.setdefault(name, {**VARIANTS.get(base, {"eff": base}), "sched": sched, "kernel_us": kernel_us})
    INFLIGHT_FIT[name] = inflight
    return name


HOPPER = ("H100", "H800", "H200")
# 0.94 memory as the runtime sees it (public specs / documented defaults, not fitted):
#  * usable capacity = the vendor's published per-GPU figure (RefHW.hbm_GB: H200 141 GB, B200 180 GB — the catalog
#    stack configs give 144 / 192 GiB = 154.6 / 206.2 GB)
#  * TensorRT-LLM grants the KV cache free_gpu_memory_fraction = 0.9 of the memory left after weights / activations
#    (KvCacheConfig default; trtllm-bench and the 0.x engine flow use the same 0.9)
#  * vLLM: gpu_memory_utilization = 0.9 of total memory for weights + activations + KV (EngineArgs default)
MEM_RT = False     # opt-in (variant key "mem_rt"); in candidate_variant — MODEL §40
KV_FRAC = {"trtllm": 0.9}
TOTAL_FRAC = {"rocm-vllm": 0.9}


def _mem_runtime(r: dict, h) -> dict:
    src = r.get("src", "")
    cap = h.hbm_GB * next((f for k, f in TOTAL_FRAC.items() if src.startswith(k)), 1.0)
    return dict(mem_cap_GB=cap, kv_mem_frac=next((f for k, f in KV_FRAC.items() if src.startswith(k)), 1.0))
# 0.94: DeepGEMM README (2025-04-18): FP8 block-scaled GEMM "up to 1550 TFLOPS on H800" (dense FP8 peak 1979) — the
# best public sustained Hopper FP8 kernel rate (WGMMA FP8 accumulators are promoted to FP32 on CUDA cores every
# 128-K block, plus clocks under full tensor load).  Not fitted to any validation row.
HOPPER_FP8_CEIL = 1550 / 1979
# FlashMLA README (2025-04-22): dense MLA decoding "up to 660 TFlops on H800" (bf16 dense peak 989.5) — attention ops
HOPPER_ATTN_CEIL = 660 / 989.5
HOPPER_CEILING = (("fp8", HOPPER_FP8_CEIL), ("attn:bf16", HOPPER_ATTN_CEIL))


def candidate_variant(base: str = "catalog_kernel") -> str:
    """0.93 refhw candidate (NOT default — fails the no-set-worse rule on one single-row group, MODEL §39): attention
    ops on the GPU thread-block schedule (attn_sched "cta": FMHA / FlashMLA tile (request, head, q-block) over SMs;
    DeepSeek trace §25.9: MLA 31 ms vs wide 88) + per-layer software overhead from the public DeepSeek trace
    (TRACE_LAYER_US); GEMMs keep wide."""
    name = base + "+cand093"
    VARIANTS.setdefault(name, {**VARIANTS[base], "attn_sched": "cta", "layer_us": TRACE_LAYER_US,
                               # 0.94: runtime memory view + Hopper kernel ceilings (public, not fitted)
                               "mem_rt": True, "fmt_ceiling": HOPPER_CEILING,
                               # 0.109: prefill two-batch overlap for the sources that document it (TBO_PREFILL_SRC)
                               "tbo_pf": True})
    return name


# 0.105: batching estimator per source, from each source's published methodology (not fitted). Every max-load
# source is an in-flight / continuous-batching engine fed the whole dataset at once, so the continuous-batching proxy
# (predict_cb: chunked prefill fused into decode iterations, best running batch that fits) is the faithful one; the
# static-batch max search (predict "max_tok_s_total") models no engine. Chunk budget and running-sequence cap are the
# engine defaults:
#   trtllm-0.17 / 0.21 / 0.21-b200 / 1.2.0 — trtllm-bench throughput, in-flight batching (perf-overview pages);
#       max_num_tokens 8192 (TensorRT-LLM LlmArgs / BuildConfig default); running cap = memory (max_batch_size 2048
#       default not binding below the KV limit at these lengths 「假设」)
#   rocm-vllm-0.11.2 — vLLM offline throughput (LLM class): V1 continuous batching + chunked prefill;
#       max_num_batched_tokens 16384 and max_num_seqs 1024 on H100 / MI300X (vllm/engine/arg_utils.py,
#       get_batch_defaults, UsageContext.LLM_CLASS)
SOURCE_ESTIMATOR = {
    "trtllm-0.17": dict(kind="cb", chunk=8192, max_seqs=None),
    "trtllm-0.21": dict(kind="cb", chunk=8192, max_seqs=None),
    "trtllm-0.21-b200": dict(kind="cb", chunk=8192, max_seqs=None),
    "trtllm-1.2.0": dict(kind="cb", chunk=8192, max_seqs=None),
    "rocm-vllm-0.11.2": dict(kind="cb", chunk=16384, max_seqs=1024),
}


def estimator(r: dict) -> dict:
    """0.105: batching estimator of a max-load row (SOURCE_ESTIMATOR; unknown sources keep the static search)."""
    return SOURCE_ESTIMATOR.get(r.get("src"), dict(kind="static", chunk=CB_CHUNK, max_seqs=None))


def rankmax_variant(base: str = "catalog") -> str:
    """0.133: base + serving.moe_rank_max = "expected" (MODEL §79)."""
    name = base + "+rankmax"
    VARIANTS.setdefault(name, {**VARIANTS[base], "rank_max": "expected"})
    return name


def est_variant(base: str = "catalog") -> str:
    """0.105: ``base`` with the per-source batching estimator on max-load rows."""
    name = f"{base}+est"
    VARIANTS.setdefault(name, {**VARIANTS[base], "est": True})
    return name


def pipe_variant(base: str = "catalog", cross_op: bool = False) -> str:
    """0.98: ``base`` with the hardware-derived overlap (exec_overlap "pipe"). GPUs run one kernel at a time, so by
    default there is no prefetch across ops (cross_op False)."""
    name = f"{base}+pipe{'x' if cross_op else ''}"
    VARIANTS.setdefault(name, {**VARIANTS[base], "overlap": "pipe", "pipe_cross_op": cross_op})
    return name


def sfu_variant(base: str = "catalog") -> str:
    """0.103: ``base`` with transcendentals on a separate SFU pipe at the public rate (RefHW.sfu_per_sm)."""
    name = f"{base}+sfu"
    VARIANTS.setdefault(name, {**VARIANTS[base], "sfu": True})
    return name


FA3_HW = ("H100", "H200", "H800", "B200")   # Hopper (FlashAttention-3) / Blackwell (async warpgroup MMA kernels)


# 0.108: FlashMLA README (deepseek-ai/FlashMLA): "up to 3000 GB/s in memory-bound configuration" on H800 SXM5
# (3.35 TB/s HBM3) — the KV-streaming efficiency of a long-context decode-attention kernel (not fitted)
KV_STREAM_EFF = 3000 / 3350


def kveff_variant(base: str = "catalog_kernel", eff: float = KV_STREAM_EFF) -> str:
    """0.108: ``base`` with KV-cache read / write streams at ``eff`` of peak DRAM bandwidth (Scenario.kv_mem_eff)."""
    name = f"{base}+kv{eff:.3f}"
    VARIANTS.setdefault(name, {**VARIANTS[base], "kv_eff": eff})
    return name


# 0.109: sources whose published prefill configuration overlaps EP dispatch / combine with compute by two-batch
# overlap (one micro-batch's all-to-all during the other's GEMMs / attention):
#   sglang-ep-2025   — LMSYS "Deploying DeepSeek with PD Disaggregation and Large-scale EP" (2025-05-05): prefill
#                      nodes run two-batch overlap (TBO), DeepEP normal kernels
#   deepseek-profile — deepseek-ai/profile-data README: prefill "two micro-batches ... overlapping computation and
#                      all-to-all communication"
# Decode rows keep the refhw overlap (no change), so does every other source.
TBO_PREFILL_SRC = frozenset({"sglang-ep-2025", "deepseek-profile"})


def tbopf_variant(base: str = "catalog_kernel") -> str:
    """0.109: ``base`` with two-batch-overlap prefill for the sources in ``TBO_PREFILL_SRC``."""
    name = f"{base}+tbopf"
    VARIANTS.setdefault(name, {**VARIANTS[base], "tbo_pf": True})
    return name


def fa3_variant(base: str = "catalog_kernel") -> str:
    """0.104: ``base`` with the flash kernel's softmax / exp overlapped with its MMAs on Hopper / Blackwell
    (Chip.attn_vec_overlap; FA3, Shah et al. 2024). A100 (FA2) and MI300X keep the serial softmax."""
    name = f"{base}+fa3"
    VARIANTS.setdefault(name, {**VARIANTS[base], "fa3": True})
    return name


def overhead_variant(base: str, us: float) -> str:
    """0.69: ``base`` variant + a fixed per-layer overhead of ``us`` µs (Scenario.layer_overhead_us)."""
    name = f"{base}+L{us:g}"
    VARIANTS.setdefault(name, {**VARIANTS[base], "layer_us": float(us)})
    return name


def calib_variant(mem_eff: float, mac_eff: float, overlap: str) -> str:
    """Register (and name) a calibration variant: explicit DRAM and MAC efficiencies, one overlap mode."""
    name = f"cal:{overlap}:{mem_eff:.2f}:{mac_eff:.2f}"
    VARIANTS.setdefault(name, dict(eff=float(mem_eff), mac_eff=float(mac_eff), overlap=overlap, kv_reuse=True))
    return name


SPLITS = {   # held-out designs: (fit predicate, test predicate)
    "hw": (lambda r: r["hw"] == "H100-SXM", lambda r: r["hw"] != "H100-SXM"),
    "model": (lambda r: not r["model"].endswith("70b"), lambda r: r["model"].endswith("70b")),
    "source": (lambda r: r["src"] == "trtllm-0.21", lambda r: r["src"] != "trtllm-0.21"),
}


def mean_abs_log(rs: list[dict]) -> float:
    lr = [abs(math.log(x["ratio"])) for x in rs if x["ratio"] == x["ratio"] and x["ratio"] > 0]
    return sum(lr) / len(lr) if lr else float("nan")


# ---------------------------------------------------------------- 0.69: continuous batching (in-flight) proxy
CB_CHUNK = 8192     # TensorRT-LLM default max_num_tokens (chunked-prefill token budget per iteration) 「假设」
INFLIGHT_DEFAULT = True    # 0.92: max-load rows check prefill memory per chunk (see _static)
INFLIGHT_FIT: dict = {}    # per-variant override (False = the ≤ 0.91 whole-batch prefill check)


def _fused(dec, pre1, f: float, rr: float) -> float:
    """One iteration carrying dec's batch + an f share of pre1's prompt — ``pdqueue._fused_step`` (0.101: that now
    handles every overlap mode the way this function did since 0.70; identical numbers)."""
    from .pdqueue import _fused_step
    return _fused_step(dec, pre1, f, rr)


def _cb_at(r: dict, b: int, eff: str, C: int = CB_CHUNK):
    """Fluid in-flight batching at max load with running batch b: each request needs n chunk iterations (C tokens)
    and osl decode steps; a share x = n·b/osl of iterations carries a chunk (x ≤ 1), else prefill limits the batch to
    osl/n.  Returns (output tok/s, fits)."""
    from .pdqueue import _chunk_plan
    isl, osl = r["isl"], max(1, r["osl"])
    n, f, rr = _chunk_plan(isl, 0, C)
    lf = max(0.0, isl - (n - 1) * C) / isl
    pre1 = evaluate(scenario(r, eff, phase="prefill", batch=1, prompt=isl, out_len=osl), _spec(r["model"], True))

    def at(k):
        dec = evaluate(scenario(r, eff, phase="decode", batch=k, ctx=isl + osl // 2, out_len=osl), _spec(r["model"], True))
        full = evaluate(scenario(r, eff, phase="decode", batch=k, ctx=isl + osl, out_len=osl), _spec(r["model"], True))
        t1 = ((n - 1) * _fused(dec, pre1, f, rr) + _fused(dec, pre1, lf, rr)) / n
        return dec, t1, full.fits and pre1.fits
    x = n * b / osl
    if x <= 1:
        dec, t1, fits = at(b)
        t = (1 - x) * dec.step + x * t1
        return b / t, fits
    k = max(1, int(osl / n))
    dec, t1, fits = at(k)
    return k / t1, fits


def predict_cb(r: dict, eff: str = "catalog", max_seqs: int | None = None) -> dict:
    """max_tok_s_total (and 0.91 max_tok_s_gpu, ÷ cards) rows under the continuous-batching proxy: best power-of-two running batch that fits."""
    best = None
    for k in range(0, 14):
        if max_seqs and 2 ** k > max_seqs:    # 0.105: the engine's running-sequence cap
            break
        v, fits = _cb_at(r, 2 ** k, eff, r.get("cb_chunk", CB_CHUNK))
        if not fits:
            break
        if best is None or v > best["pred"]:
            best = {"pred": v, "fits": True, "batch": 2 ** k}
    if best and r["metric"] == "max_tok_s_gpu":
        best["pred"] /= _cards(r)
    return best or {"pred": float("nan"), "fits": False}
