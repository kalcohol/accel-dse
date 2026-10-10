"""Validation tiers for core v2.

V0  invariants / metamorphic relations            (tests/test_core_metamorphic.py)
V1  params / active params / FLOPs vs releases     (tests/test_core_model.py)
V2  GPU-like trend sanity (caveated)               v2_trends()
V3  cross-tool: GenZ-LLM on identical abstract systems   v3_genz()
V4  serving: closed-form queueing (pdqueue) vs request-level DES (pdsim)   v4_serving() / v4_grid()   (0.54)
V5  silicon measurements                           n/a (no NPU exists — design guidance only)

V4 is model-vs-model: both sides share the per-step costs (prefill time, decode step vs batch, KV transfer), so it
measures the queueing / batching approximations only, not the cost model.  The DES has its own 「假设」 (FCFS, Poisson,
static prefill batching, whole-prefix LRU, no preemption).

V2 caveat: "H100-like" is OUR abstraction (128×128×16 MACs @1.83 GHz ≈ 989 dense
bf16 TFLOPS, HBM3 5 stacks @5.2 Gbps = 3.33 TB/s raw, mem_eff 0.8 「假设」).
Bands are roofline-plausibility bands informed by public serving reports
(e.g. vLLM / ClusterBid H100 Llama-3.1-8B: TTFT ≈ 92 ms for a 4 K prompt at
batch 1), not calibration targets.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import evaluate
from .hardware import CHIP_100T, CHIP_H100_LIKE, Link
from .parallel import Layout
from .scenario import Scenario, Serving, Workload

H100_MEM = "hbm3_5s_8h16g_5200"
LPDDR_191 = "lpddr5x_4x64_8533_16g"
HBM_6600 = "hbm3e_8s_12h24g_9200"
REF = Path(__file__).resolve().parents[1] / "data" / "genz_reference.json"


def _h100(model: str, **sv) -> Scenario:
    return Scenario(model=model, chip=CHIP_H100_LIKE, mem_id=H100_MEM, mem_eff=0.8, mapping="reconf",
                    link=Link(GBps=450.0, alpha_us=3.0), serving=Serving(**sv))


def v2_trends() -> list[dict]:
    rows = []

    def add(name, value, lo, hi, unit, note=""):
        rows.append({"check": name, "value": value, "lo": lo, "hi": hi, "unit": unit, "ok": lo <= value <= hi,
                     "note": note})

    b1 = evaluate(_h100("llama-3.1-8b", batch=1, ctx=1024))
    add("Llama-3.1-8B bf16 H100-like TPOT @B1", b1.tpot * 1e3, 4.5, 9.0, "ms",
        "bandwidth roofline 16 GB / 2.7 TB/s ≈ 6 ms")
    b16 = evaluate(_h100("llama-3.1-8b", batch=16, ctx=1024))
    add("TPOT(B16)/TPOT(B1) (bandwidth-bound regime)", b16.tpot / b1.tpot, 1.0, 1.5, "×")
    fp8 = evaluate(_h100("qwen3-8b-fp8", batch=1, ctx=1024))
    bf = evaluate(_h100("qwen3-8b", batch=1, ctx=1024))
    add("Qwen3-8B FP8/BF16 TPOT @B1", fp8.tpot / bf.tpot, 0.45, 0.65, "×", "weights halve; KV + head stay bf16")
    p = evaluate(_h100("llama-3.1-8b", phase="prefill", batch=1, prompt=4096))
    add("Llama-3.1-8B TTFT 4K prompt @B1", p.ttft * 1e3, 50.0, 200.0, "ms", "public H100 reports ≈ 92 ms")
    tp1 = evaluate(_h100("qwen3-32b", batch=1, ctx=1024))
    tp2 = evaluate(_h100("qwen3-32b", batch=1, ctx=1024).replace("layout", Layout(tp=2)))
    add("Qwen3-32B TP2 speed-up @B1", tp1.tpot / tp2.tpot, 1.5, 2.0, "×")
    big = evaluate(_h100("llama-3.1-8b", batch=256, ctx=1024))
    add("Llama-3.1-8B tok/s @B256 (compute regime)", big.throughput, 4000.0, 25000.0, "tok/s",
        "public H100 vLLM ≈ 9 k tok/s @B128")
    return rows


def v3_genz(path: Path = REF) -> list[dict]:
    ref = json.loads(path.read_text())
    out = []
    for r in ref["rows"]:
        mem = LPDDR_191 if r["mem_GBps"] < 1000 else HBM_6600
        for org in (("os", "reconf") if mem == HBM_6600 else ("os",)):
            ours = evaluate(Scenario(model="llama-3.1-8b", chip=CHIP_100T, mem_id=mem, mapping=org,
                                     serving=Serving(batch=r["batch"], ctx=r["ctx"]))).tpot * 1e3
            comparable = mem == LPDDR_191 or org == "reconf"
            out.append({**r, "mapping": org, "ours_ms": ours, "ratio": ours / r["tpot_ms"], "comparable": comparable})
    return out


def v2_domain() -> list[dict]:
    """Video / protein sanity rows (0.41).  Only the workload / FLOP model is checked against a public number; no
    hardware calibration.  Wan2.1 README: T2V-1.3B makes a 5 s 480P clip on one RTX 4090 in about 4 minutes
    (T5 + VAE + CPU offload included); a 4090 has ≈ 165 TFLOPS dense bf16 with fp32 accumulate.  0.44: the FLOP count
    includes the umT5 encoder and the Wan-VAE decode (the README time does too)."""
    r = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM_6600))
    flops = r.domain_summary()["tflop_per_request"] * 1e12
    eff = flops / 240.0 / 165e12
    p = r.pipeline or {}
    return [{"check": "Wan2.1-1.3B 480P pipeline FLOP/clip ÷ 240 s ÷ RTX 4090 165 TFLOPS", "value": eff, "lo": 0.4,
             "hi": 1.0, "unit": "×", "ok": 0.4 <= eff <= 1.0,
             "note": f"{flops / 1e15:.1f} PFLOP per clip (DiT 50 steps × CFG 2 × 32,760 tokens + umT5 + Wan-VAE decode "
                     f"{p.get('tflop', 0) / 1e3:.2f} PFLOP); README: ~4 min on a 4090 incl. T5 + VAE"},
            _opensora_row(), _esmfold_row()]


def _esmfold_row() -> dict:
    """0.43: ESMFold paper (Lin et al., Science 2023 / bioRxiv 2022.07.20.500902): "On a single NVIDIA V100 GPU, ESMFold
    makes a prediction on a protein with 384 residues in 14.2 seconds".  The folding trunk (≈96 % of the FLOPs) runs
    in fp32 in the reference implementation (ESM-2 in fp16), so the denominator is the V100 SXM2 fp32 peak 15.7
    TFLOPS.  Guards the pair-representation op model (triangle multiplication / attention N³ terms, recycles)."""
    r = evaluate(Scenario(model="esmfold", mem_id=HBM_6600, workload=Workload(seq_len=384)))
    flops = r.domain_summary()["tflop_per_request"] * 1e12
    eff = flops / 14.2 / 15.7e12
    return {"check": "ESMFold 384 残基 FLOP/request ÷ 14.2 s ÷ V100 fp32 15.7 TFLOPS", "value": eff, "lo": 0.05,
            "hi": 0.8, "unit": "×", "ok": 0.05 <= eff <= 0.8,
            "note": f"{flops / 1e12:.1f} TFLOP per request (ESM-2 3B + 48 折叠块 × 4 次主干前向 + 结构模块); "
                    "论文：单 V100 14.2 s（主干 fp32）"}


def _opensora_row() -> dict:
    """0.42: Open-Sora 1.2 README (Gradio section, 80 GB H100): 720p 4 s takes 130 s end to end (T5 + VAE included).
    The H100 has 989 TFLOPS dense bf16.  STDiT3 uses factorized spatial / temporal attention as released; if we had
    modelled full 3D attention the implied utilisation would exceed 100 %, so this row also guards the span model."""
    r = evaluate(Scenario(model="opensora-stdit3", mem_id=HBM_6600))
    flops = r.domain_summary()["tflop_per_request"] * 1e12
    eff = flops / 130.0 / 989e12
    p = r.pipeline or {}
    return {"check": "Open-Sora STDiT3 720p 4s pipeline FLOP/clip ÷ 130 s ÷ H100 989 TFLOPS", "value": eff, "lo": 0.05,
            "hi": 0.6, "unit": "×", "ok": 0.05 <= eff <= 0.6,
            "note": f"{flops / 1e15:.1f} PFLOP per clip (30 steps × CFG 2 × 108,000 tokens, factorized S/T attention; "
                    f"T5 + VAE decode {p.get('tflop', 0) / 1e3:.2f} PFLOP); README: 130 s on one H100 incl. T5 + VAE"}



# ---------------------------------------------------------------- V4 serving (0.54)
V4_LOADS = (0.3, 0.6, 0.85)
V4_CVS = (0.0, 0.5, 1.0)
V4_MODES = ("pd", "coloc_prefill_first", "coloc_chunked")
# tolerance bands per metric (|err| ≤ band counts as within band); documented in MODEL.md §18.4
V4_BANDS = {"ttft_p50": 0.15, "ttft_p90": 0.20, "ttft_p99": 0.30, "tpot_mean": 0.10, "tpot_p90": 0.20, "itl_max": 0.30,
            "slo_goodput": 0.20, "prefix_hit": 0.03}


# 0.55 — scenario families (model, layout, prefill cards, decode cards, TTFT / TPOT SLO ms).  dense8b = the 0.54 grid;
# moe30b = a live MoE (Qwen3-30B-A3B, TP2 × EP2: expert all-to-all on the card link); tp4_32b = a 4-card TP replica
# (Qwen3-32B, TP4: heavier per-layer all-reduce, 3 colocated / 1 + 2 PD replicas).  All on 1P + HBM3E 6.6 Gbps.
V4_FAMILIES = {
    "dense8b": ("qwen3-8b", dict(tp=2), 2, 6, 400, 10),
    "moe30b": ("qwen3-30b-a3b", dict(tp=2, ep=2), 2, 6, 400, 15),
    "tp4_32b": ("qwen3-32b", dict(tp=4), 4, 8, 600, 15),
}


def v4_scenario(load: float, cv: float = 0.0, prefix: bool = False, family: str = "dense8b") -> Scenario:
    from .hardware import CHIPS
    from .scenario import PDConfig
    model, lay, pc, dc, ttft_slo, tpot_slo = V4_FAMILIES[family]
    pd = dict(enabled=True, prefill_layout=Layout(**lay), prefill_cards=pc, decode_cards=dc, load=load,
              prompt_cv=cv, out_cv=cv)
    if prefix:
        pd.update(prefix_len=2048, prefix_count=20000)
    return Scenario(model=model, chip=CHIPS["1P"], mem_id=HBM_6600, layout=Layout(**lay),
                    serving=Serving(batch=64, prompt=4096, out_len=512, ttft_slo_ms=ttft_slo, tpot_slo_ms=tpot_slo),
                    pd=PDConfig(**pd))


def _summ(rows: list[dict]) -> list[dict]:
    import math
    out = []
    fams = sorted({r.get("family", "dense8b") for r in rows}, key=lambda f: list(V4_FAMILIES).index(f))
    for fam in [None] + (fams if len(fams) > 1 else []):
        sel = [r for r in rows if fam is None or r.get("family", "dense8b") == fam]
        for mode in V4_MODES:
            for metric, band in V4_BANDS.items():
                pairs = [(r["modes"][mode]["err"].get(metric), r["modes"][mode].get("noise", {}).get(metric))
                         for r in sel if mode in r.get("modes", {})]
                pairs = [(e, nz) for e, nz in pairs if e is not None and math.isfinite(e)]
                if not pairs:
                    continue
                errs = sorted(e for e, _ in pairs)
                nzs = sorted(nz for _, nz in pairs if nz is not None and math.isfinite(nz))
                row = {"family": fam or "all", "mode": mode, "metric": metric, "n": len(errs), "min": errs[0],
                       "median": errs[len(errs) // 2], "max": errs[-1],
                       "mean_abs": sum(abs(e) for e in errs) / len(errs),
                       "within_band": sum(abs(e) <= band for e in errs) / len(errs), "band": band}
                if nzs:
                    # DES seed-to-seed spread of one run (sd / mean), median over the points
                    row["noise_median"] = nzs[len(nzs) // 2]
                out.append(row)
    return out


def _grid_point(args):
    load, cv, prefix, family, n_req, slo, seeds = args[:7]
    slo_n = args[7] if len(args) > 7 else max(600, n_req // 2)
    import time
    from .pdsim import compare
    t0 = time.time()
    r = compare(v4_scenario(load, cv, prefix, family), n_req=n_req, warmup=n_req // 4, seeds=seeds, slo=slo,
                slo_n=slo_n, slo_tol=0.01)
    r.update(load=load, cv=cv, prefix=prefix, family=family, n_req=n_req, elapsed_s=round(time.time() - t0, 1))
    return r


# 0.55: the two extra families run a reduced grid (load × CV 0 / 1, prefix off)
V4_EXTRA_GRID = dict(loads=V4_LOADS, cvs=(0.0, 1.0), prefixes=(False,))


def v4_grid(n_req: int = 3000, slo: bool = False, loads=V4_LOADS, cvs=V4_CVS, prefixes=(False, True),
            seed: int = 11, progress: bool = False, seeds: int = 3, families=tuple(V4_FAMILIES),
            jobs: int = 1, n_high: int | None = None, high_load: float = 0.85, partial_dir: str | None = None,
            only: int | None = None) -> dict:
    """Full V4 grid (scripts/v4_serving.py).  0.55: ``seeds`` independent DES runs per point (metrics averaged,
    spread reported as ``noise``), the dense8b family on the full grid and the others on V4_EXTRA_GRID; ``jobs``
    worker processes.  0.57: ``n_high`` requests per DES run at load ≥ ``high_load`` (the slot / prompt queues relax
    slowly there; the SLO bisection keeps ``n_req // 2``).  0.65.1: ``partial_dir`` (sequential only) writes one JSON
    per point and skips points already there, so an interrupted run resumes; ``only`` = run at most that many new
    points and return (rows / summary then cover the points done so far)."""
    seed_t = tuple(seed + i for i in range(max(1, seeds)))
    pts = []
    for fam in families:
        g = dict(loads=loads, cvs=cvs, prefixes=prefixes) if fam == "dense8b" else V4_EXTRA_GRID
        for load in g["loads"]:
            for cv in g["cvs"]:
                for prefix in g["prefixes"]:
                    n_pt = n_high if (n_high and load >= high_load) else n_req
                    pts.append((load, cv, prefix, fam, n_pt, slo, seed_t, max(600, n_req // 2)))
    rows = []
    pdir = None
    if partial_dir:
        import json
        from pathlib import Path
        pdir = Path(partial_dir)
        pdir.mkdir(parents=True, exist_ok=True)

    def _pfile(a):
        load, cv, prefix, fam, n_pt = a[:5]
        return pdir / f"{fam}_l{load}_cv{cv}_p{int(prefix)}_n{n_pt}_s{len(seed_t)}_slo{int(bool(slo))}.json"

    def show(r):
        if progress:
            print(f"{r['family']} load {r['load']} cv {r['cv']} prefix {int(r['prefix'])}: " + " | ".join(
                f"{m}: " + " ".join(f"{k} {v:+.2f}" for k, v in x["err"].items())
                for m, x in r.get("modes", {}).items()), flush=True)
    if jobs > 1:
        import multiprocessing as mp
        order = sorted(range(len(pts)), key=lambda i: -pts[i][4])      # longest runs first (load balance)
        got = {}
        with mp.get_context("fork").Pool(jobs) as pool:
            for i, r in zip(order, pool.imap(_grid_point, [pts[i] for i in order])):
                got[i] = r
                show(r)
        rows = [got[i] for i in range(len(pts))]
    else:
        for a in pts:
            if pdir is not None and _pfile(a).exists():          # 0.65.1: resume from per-point files
                rows.append(json.loads(_pfile(a).read_text()))
                continue
            if only is not None and only <= 0:
                continue
            r = _grid_point(a)
            if only is not None:
                only -= 1
            if pdir is not None:
                _pfile(a).with_suffix(".tmp").write_text(json.dumps(r, default=float))
                _pfile(a).with_suffix(".tmp").replace(_pfile(a))
            rows.append(r)
            show(r)
    return {"rows": rows, "summary": _summ(rows), "n_req": n_req, "n_high": n_high, "high_load": high_load,
            "seed": seed, "seeds": list(seed_t),
            "families": {f: V4_FAMILIES[f][0] for f in families}, "bands": V4_BANDS}


def v4_serving(n_req: int = 1000) -> list[dict]:
    """Live V4 subset for ``accel-dse validate`` (≈ 10–20 s): medium load, CV 0 and 1, prefix on, three modes."""
    from .pdsim import compare
    rows = []
    for load, cv, prefix in ((0.6, 0.0, False), (0.6, 1.0, False), (0.6, 0.0, True)):
        r = compare(v4_scenario(load, cv, prefix), n_req=n_req, warmup=n_req // 4, seed=11)
        for mode, x in r.get("modes", {}).items():
            for metric, e in x["err"].items():
                band = V4_BANDS.get(metric)
                if band is None:
                    continue
                rows.append({"scenario": f"load {load} CV {cv} prefix {'on' if prefix else 'off'}", "mode": mode,
                             "metric": metric, "sim": x["sim"].get(metric), "ana": x["ana"].get(metric), "err": e,
                             "band": band, "ok": e == e and abs(e) <= band})
    return rows
