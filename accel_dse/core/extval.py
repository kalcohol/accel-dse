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


def rows(path: Path = DATA) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def scenario(r: dict, eff: str = "catalog", **sv) -> Scenario:
    v = VARIANTS.get(eff, {})
    ov = v.get("overlap") if v.get("overlap") in ("class", "kernel", "serial") else "stage"
    e = v.get("eff", eff)
    h = REF_HW[r["hw"]]
    chip = h.chip if v.get("mac_eff") is None else replace(h.chip, mac_eff=v["mac_eff"])
    mem_eff = e if isinstance(e, float) else EFF[e]
    return Scenario(model=r["model"], chip=chip, mem_id=h.mem_id, mem_eff=mem_eff, link=h.link, mapping="reconf",
                    exec_overlap=ov,
                    layout=Layout(tp=r["tp"]), formats_override=FP8 if r["dtype"] == "fp8" else (),
                    serving=Serving(**sv))


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
