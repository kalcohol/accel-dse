"""V6 — external validation against PUBLISHED measurements (0.66).

Data: accel_dse/data/ext_measurements.json (transcribed, cited per row).  Hardware: core/refhw.py (参考硬件,
datasheet peaks, nothing tuned).  Metrics and how accel-dse predicts them:

  ttft_ms            batch-B prefill of ISL tokens: ``evaluate(phase=prefill).ttft``
  static_tok_s_gpu   static batching (all B requests prefilled together, then OSL − 1 decode steps):
                     B·OSL / (TTFT(B, ISL) + Σ_steps TPOT(B, ctx)) / tp; Σ by the mean TPOT over 4 mid-point contexts
  max_tok_s_total    in-flight batching under infinite load, batch chosen by the runtime: the max over B (powers of two,
                     must fit) of B·OSL / (TTFT(B, ISL) + (OSL − 1)·mean TPOT) — a static-batch proxy (an upper bound
                     on the decode side, since in-flight batching interleaves prefill chunks with decode)

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
FP8 = (("attn", "fp8"), ("mlp", "fp8"), ("act", "fp8"), ("kv", "fp8"))
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


def rows(path: Path = DATA) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def scenario(r: dict, eff: str = "catalog", **sv) -> Scenario:
    v = VARIANTS.get(eff, {})
    ov = v.get("overlap") if v.get("overlap") in ("class", "kernel", "serial", "tbo") else "stage"
    e = v.get("eff", eff)
    h = REF_HW[r["hw"]]
    chip = h.chip if v.get("mac_eff") is None else replace(h.chip, mac_eff=v["mac_eff"])
    mem_eff = e if isinstance(e, float) else EFF[e]
    extra = {}
    if r.get("node_cards"):            # 0.67 multi-node rows: cards per node + scale-out NIC per card
        extra = dict(node_cards=r["node_cards"], net=Link(r["net_GBps"], r.get("net_alpha_us", 5.0)))
    if r.get("moe_skew"):
        sv = {**sv, "moe_skew": r["moe_skew"]}
    return Scenario(model=r["model"], chip=chip, mem_id=h.mem_id, mem_eff=mem_eff, link=h.link, mapping="reconf",
                    exec_overlap=ov,
                    layout=Layout(tp=r["tp"], dp=r.get("dp", 1), ep=r.get("ep", 1), etp=r.get("etp", 1)),
                    formats_override=FP8 if r["dtype"] == "fp8" else (),
                    serving=Serving(**sv), **extra)


def _ev(r, var, **sv):
    v = VARIANTS[var]
    return _Pt(evaluate(scenario(r, var, **sv), _spec(r["model"], v["kv_reuse"])), v["overlap"])


def _ttft(r, b, isl, eff):
    return _ev(r, eff, phase="prefill", batch=b, prompt=isl, out_len=max(1, r["osl"]))


def _decode(r, b, isl, osl, eff, n=4):
    res = [_ev(r, eff, phase="decode", batch=b, ctx=isl + max(1, round(osl * (k + 0.5) / n)), out_len=osl)
           for k in range(n)]
    return sum(x.tpot for x in res) / n, all(x.fits for x in res)


def _static(r, b, eff):
    p = _ttft(r, b, r["isl"], eff)
    t, fits = _decode(r, b, r["isl"], r["osl"], eff)
    total = p.ttft + (r["osl"] - 1) * t
    return b * r["osl"] / total, fits and p.fits, {"ttft_s": p.ttft, "tpot_s": t, "prefill_share": p.ttft / total}


def predict(r: dict, eff: str = "catalog") -> dict:
    m = r["metric"]
    if m == "ttft_ms":
        p = _ttft(r, r["batch"], r["isl"], eff)
        return {"pred": p.ttft * 1e3, "fits": p.fits, "bound": p.bound}
    if m == "static_tok_s_gpu":
        v, fits, det = _static(r, r["batch"], eff)
        return {"pred": v / r["tp"], "fits": fits, **det}
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
    if m == "max_tok_s_total":
        best = None
        for k in range(0, 14):
            b = 2 ** k
            v, fits, det = _static(r, b, eff)
            if not fits:
                break
            if best is None or v > best["pred"]:
                best = {"pred": v, "fits": True, "batch": b, **det}
        return best or {"pred": float("nan"), "fits": False}
    raise ValueError(m)


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
        lr = [math.log(x["ratio"]) for x in t if x["metric"] == m and x["ratio"] == x["ratio"]]
        out[m] = {"n": len(lr), "gmean_ratio": math.exp(sum(lr) / len(lr)),
                  "mean_abs_err_pct": 100 * (math.exp(sum(abs(v) for v in lr) / len(lr)) - 1),
                  "min_ratio": math.exp(min(lr)), "max_ratio": math.exp(max(lr))}
    return out


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


def _fused(dec, pre1, f: float, rr: float) -> float:
    """One iteration carrying dec's batch + an f share of pre1's prompt.  exec_overlap="stage" uses the tool's own
    ``pdqueue._fused_step``; the kernel-serial modes add the same per-stage components and take that mode's total."""
    from .pdqueue import _fused_step
    if f <= 0:
        return dec.step
    if dec.stages[0].time.overlap == "stage":
        return _fused_step(dec, pre1, f, rr)
    worst = 0.0
    for sd, sp in zip(dec.stages, pre1.stages):
        td, tp = sd.time, sp.time
        bw = td.dram_bytes / td.t_dram if td.t_dram > 0 else math.inf
        nonw = sp.dram["total"] - sp.dram["weights"]
        t = replace(td, t_array=td.t_array + f * tp.t_array, t_mac=td.t_mac + f * tp.t_mac,
                    t_feed=td.t_feed + f * tp.t_feed, t_vector=td.t_vector + f * tp.t_vector,
                    t_arr_attn=td.t_arr_attn + f * tp.t_arr_attn,
                    t_dram=td.t_dram + (f * nonw + rr * sp.dram["kv_write"]) / bw,
                    t_dram_kv=td.t_dram_kv + (f * tp.t_dram_kv if tp.t_dram > 0 else 0.0) + rr * sp.dram["kv_write"] / bw,
                    t_link=td.t_link + f * tp.t_link, t_sync=max(td.t_sync, tp.t_sync))
        worst = max(worst, t.total)
    return dec.step * worst / dec.tick if dec.tick > 0 else math.inf


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


def predict_cb(r: dict, eff: str = "catalog") -> dict:
    """max_tok_s_total rows under the continuous-batching proxy: best power-of-two running batch that fits."""
    best = None
    for k in range(0, 14):
        v, fits = _cb_at(r, 2 ** k, eff)
        if not fits:
            break
        if best is None or v > best["pred"]:
            best = {"pred": v, "fits": True, "batch": 2 ** k}
    return best or {"pred": float("nan"), "fits": False}
