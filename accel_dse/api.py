"""JSON API over core v2 (used by the web UI, the HTTP server and tests).

Every request carries a (partial) Scenario dict that is merged onto the
default scenario and then parsed **strictly** (unknown keys / non-finite
numbers rejected).  All sweeps are controlled replacements of that scenario.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from . import __version__, mem_catalog
from .core.catalog import DOMAINS, catalog_listing, get_model, labels, list_models, offline_entries, unlisted_models
from .core.evaluate import Result, evaluate
from .core.hardware import CHIPS, Chip
from .core.mapping import ORG_LABEL, ORGS
from .core.parallel import Layout, enumerate_layouts
from .core.scenario import Scenario
from .core.search import best_batch, search_layouts, tpot_throughput_front
from .core.serving import goodput
from .core.stability import ranking_stability

HONESTY = ("所有硬件参数（频率、阵列几何、SRAM 端口、DRAM 效率、链路 α/β、MAC 效率）均为「假设」，未经硅片标定；"
           "本工具用于设计指导与相对比较，不是性能承诺。")
MAX_CARDS = 64
SWEEP_PATHS = {
    "serving.batch": int, "serving.ctx": int, "serving.prompt": int, "serving.spec_k": int,
    "chip.sram_mib": float, "chip.sram_port_Bpc": float, "chip.freq_ghz": float, "chip.mac_eff": float,
    "chip.gemv_macs": int, "mem_eff": float, "link.GBps": float, "link.alpha_us": float,
    "workload.frames": int, "workload.steps": int, "workload.height": int, "workload.width": int,
    "workload.seq_len": int, "workload.msa": int, "workload.recycles": int, "workload.samples": int,
}


class ApiError(ValueError):
    pass


_POOL: ProcessPoolExecutor | None = None


def _worker_init(parent: int) -> None:
    """Worker processes exit when the server dies (even on SIGKILL, which skips pool shutdown)."""
    import threading
    import time

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def enable_pool(workers: int | None = None) -> int:
    """Run the heavy searches (compare: one job per mapping; stability: one job per request) in
    worker processes.  Enabled by the HTTP server only; library / test calls stay in-process."""
    global _POOL
    n = workers if workers is not None else min(5, os.cpu_count() or 1)
    if n > 1 and _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_worker_init, initargs=(os.getpid(),))
    return n if _POOL is not None else 1


def shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def clean(x: Any) -> Any:
    """JSON-safe copy: non-finite floats (e.g. TTFT of an infeasible prefill) → None."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


# ------------------------------------------------------------------ scenario parsing
def _merge(base: dict, over: dict, where: str = "scenario") -> dict:
    out = dict(base)
    for k, v in over.items():
        if k not in base:
            raise ApiError(f"{where}: unknown key {k!r}")
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}.{k}")
        else:
            out[k] = v
    return out


def _finite(x: Any, where: str = "body") -> None:
    if isinstance(x, float) and not math.isfinite(x):
        raise ApiError(f"{where}: non-finite number")
    if isinstance(x, dict):
        for k, v in x.items():
            _finite(v, f"{where}.{k}")
    if isinstance(x, list):
        for i, v in enumerate(x):
            _finite(v, f"{where}[{i}]")


def scenario_from_body(body: dict) -> Scenario:
    if not isinstance(body, dict):
        raise ApiError("body must be a JSON object")
    _finite(body)
    sc = body.get("scenario", {})
    if not isinstance(sc, dict):
        raise ApiError("scenario must be an object")
    preset = body.get("chip_preset", "100T")
    if preset not in CHIPS:
        raise ApiError(f"chip_preset must be one of {sorted(CHIPS)}")
    base = Scenario(chip=CHIPS[preset]).to_dict()
    try:
        d = _merge(base, sc)
        scn = Scenario.from_dict(d)
        get_model(scn.model)
        mem_catalog.parse_mem_id(scn.mem_id)
    except ApiError:
        raise
    except (ValueError, TypeError, KeyError) as e:
        raise ApiError(str(e).strip("'\"")) from None
    if scn.layout.cards > MAX_CARDS:
        raise ApiError(f"at most {MAX_CARDS} cards per replica")
    return scn


def _cards(body: dict) -> int:
    n = body.get("cards", 8)
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= MAX_CARDS):
        raise ApiError(f"cards must be an integer in [1, {MAX_CARDS}]")
    return n


def _objective(body: dict) -> str:
    o = body.get("objective", "decode")
    if o not in ("decode", "goodput"):
        raise ApiError("objective must be decode|goodput")
    if o == "goodput":
        mid = (body.get("scenario") or {}).get("model") if isinstance(body.get("scenario"), dict) else None
        if mid and not get_model(mid).kv_cache:
            raise ApiError("goodput 目标仅适用于 LLM / VLM；视频 / 蛋白质模型请用 decode 目标（单位/s/卡）")
    return o


# ------------------------------------------------------------------ serialisation
def _stage_dict(s) -> dict:
    t = s.time
    return {"index": s.index, "layers": list(s.layers), "bound": t.bound, "array_util": t.array_util,
            "t_ms": {"array": t.t_array * 1e3, "mac": t.t_mac * 1e3, "feed": t.t_feed * 1e3,
                     "vector": t.t_vector * 1e3, "dram": t.t_dram * 1e3, "link": t.t_link * 1e3,
                     "sync": t.t_sync * 1e3, "total": t.total * 1e3},
            "dram_GB": {k: v / 1e9 for k, v in s.dram.items()},
            "mem": {"stored_w_GiB": s.mem.stored_w / 2**30, "kv_GiB": s.mem.kv_total / 2**30,
                    "state_GiB": s.mem.state_total / 2**30, "need_GiB": s.mem.dram_need / 2**30,
                    "cap_GiB": s.mem.dram_cap / 2**30, "fits": s.mem.fits, "residency": s.mem.residency,
                    "kv_sram_MiB": s.mem.kv_sram / 2**20, "staging_MiB": s.mem.staging / 2**20,
                    "pipe_w_GiB": s.mem.pipe_w / 2**30},
            "convert_Melems": s.convert_elems / 1e6, "tflops": t.flops / 1e12}


def result_dict(r: Result, *, with_goodput: bool = False) -> dict:
    s = r.scenario
    out = {"summary": r.summary(), "stages": [_stage_dict(x) for x in r.stages],
           "model": {"id": r.model.id, "hf_id": r.model.hf_id, **labels(r.model)},
           "hash": s.hash(), "scenario": s.to_dict()}
    if with_goodput and s.serving.phase == "decode" and r.workload is None:
        g = goodput(r)
        out["goodput"] = {"tok_s": g.goodput_tok_s, "tok_s_card": g.goodput_per_card, "ttft_ms": g.ttft_ms,
                          "ttft_ok": g.ttft_ok, "prefill_batch": g.prefill_batch, "decode_share": g.decode_share,
                          "prefill_tok_s": g.prefill_tok_s}
    return out


def _row(r, objective: str) -> dict:
    res = r.result
    st = res.stages[res.heaviest_stage] if res else None
    d = {"layout": r.layout.label, "layout_obj": r.layout.__dict__, "mapping": r.mapping, "batch": r.batch,
         "tok_s_card": r.per_card, "score": r.score(objective),
         "tpot_ms": res.tpot * 1e3 if (res and res.workload is None) else None,
         "latency_ms": res.latency * 1e3 if res else None, "unit": res.workload.unit if (res and res.workload) else "token",
         "bound": res.bound if res else None,
         "array_util": st.time.array_util if st else None, "fits": res.fits if res else False}
    if r.goodput is not None:
        d.update(goodput_card=r.goodput.goodput_per_card, ttft_ms=r.goodput.ttft_ms, ttft_ok=r.goodput.ttft_ok)
    return d


# ------------------------------------------------------------------ endpoints
def api_health() -> dict:
    return {"ok": True, "version": __version__, "core": "v2", "domains": ["llm", "vlm", "gen", "protein"]}


def api_models() -> dict:
    return {"models": list_models(), "offline": offline_entries(), "catalog": [m["id"] for m in catalog_listing()],
            "unlisted": unlisted_models(), "domains": DOMAINS,
            "note": "按厂商 → 系列排列；视频生成（DiT）与蛋白质（ESM-2 编码器、ESMFold / AlphaFold 2 / OpenFold / "
                    "Boltz-1 / Protenix 结构预测）已接入 v2 可评估；AlphaFold 3 权重需申请，仍为离线条目；"
                    "见 docs/MODEL.md §10–§12"}


def api_catalog() -> dict:
    chips = {k: {**c.__dict__, "formats": [list(x) for x in c.formats.rates], "peak_tflops_bf16": c.peak_tflops_bf16,
                 "port_GBps": c.port_GBps, "gemv": c.gemv, "lanes": c.lanes}
             for k, c in CHIPS.items()}
    return {"chips": chips, "mappings": [{"id": o, "label": ORG_LABEL[o]} for o in ORGS],
            "memory": mem_catalog.catalog_dict(), "defaults": Scenario().to_dict(), "honesty": HONESTY,
            "sweep_paths": sorted(SWEEP_PATHS), "max_cards": MAX_CARDS}


def api_memory(body: dict) -> dict:
    """Resolve a memory configuration (structured fields or an ``id``) to a spec."""
    _finite(body)
    allowed = {"id", "type", "form", "width", "rate", "count", "cap", "height", "die", "meta_mode"}
    bad = set(body) - allowed
    if bad:
        raise ApiError(f"memory: unknown keys {sorted(bad)}")
    try:
        if "id" in body:
            s = mem_catalog.parse_mem_id(str(body["id"]))
        else:
            s = mem_catalog.make_spec(body.get("type", "LPDDR5X"), form=body.get("form"), width_bits=body.get("width"),
                                      rate=body.get("rate"), count=body.get("count"), cap_GB=body.get("cap"),
                                      height=body.get("height"), die_Gb=body.get("die"),
                                      meta_mode=bool(body.get("meta_mode", False)))
    except (ValueError, KeyError, TypeError) as e:
        raise ApiError(str(e).strip("'\"")) from None
    return {"id": s.id, "kind": s.kind, "raw_GBps": s.raw_GBps, "eff_GBps": s.effective_GBps,
            "efficiency": s.efficiency, "capacity_GiB": s.capacity_GB,
            "nominal_capacity_GiB": s.nominal_capacity_GB,
            "tag": s.tag, "tag_zh": s.tag_zh,
            "spec_status": s.spec_status, "product_status": s.product_status,
            "spec_zh": mem_catalog.SPEC_ZH[s.spec_status],
            "product_zh": mem_catalog.PRODUCT_ZH[s.product_status],
            "meta_mode": s.meta_mode, "meta_reserve_frac": s.meta_reserve_frac,
            "warnings": s.warnings(),
            "fields": {"type": s.mem_type, "form": s.form, "width": s.unit_width_bits, "rate": s.rate_MTps,
                       "count": s.n_units, "cap": s.cap_per_unit_GB, "height": s.hbm_height, "die": s.hbm_die_Gb,
                       "meta_mode": s.meta_mode}}


def api_eval(body: dict) -> dict:
    scn = scenario_from_body(body)
    if body.get("best_batch"):
        bb = best_batch(scn)
        if bb.batch:
            scn = scn.replace("serving.batch", bb.batch)
    try:
        r = evaluate(scn)
    except ValueError as e:
        raise ApiError(str(e)) from None
    return result_dict(r, with_goodput=bool(body.get("goodput", True)))


def api_layouts(body: dict) -> dict:
    scn = scenario_from_body(body)
    obj = _objective(body)
    stats: dict = {}
    rows = search_layouts(scn, _cards(body), objective=obj, top=16, stats=stats)
    return {"objective": obj, "mapping": scn.mapping, "rows": [_row(r, obj) for r in rows],
            "n_layouts": stats["layouts"], "stats": stats}


def _compare_row(body: dict, org: str) -> dict:
    scn = scenario_from_body(body).replace("mapping", org)
    obj = _objective(body)
    rows = search_layouts(scn, _cards(body), objective=obj, top=2)
    top = rows[0]
    d = _row(top, obj)
    d["label"] = ORG_LABEL[org]
    d["runner_up"] = _row(rows[1], obj) if len(rows) > 1 else None
    if obj == "decode" and top.result is not None and scn.serving.phase == "decode" and top.result.workload is None:
        g = goodput(top.result)
        d.update(goodput_card=g.goodput_per_card, ttft_ms=g.ttft_ms, ttft_ok=g.ttft_ok)
    return d


def api_compare(body: dict) -> dict:
    """Best layout per mapping organisation (the mapping comparison view)."""
    scenario_from_body(body)                    # validate in-process (ApiError → 400)
    obj, cards = _objective(body), _cards(body)
    if _POOL is not None:
        out = list(_POOL.map(_compare_row, [body] * len(ORGS), ORGS))
    else:
        out = [_compare_row(body, org) for org in ORGS]
    best = max(out, key=lambda d: d["score"])
    return {"objective": obj, "cards": cards, "rows": out, "best_mapping": best["mapping"]}


def _mem_candidates(mem_id: str) -> list:
    """Same memory type / form / rate, more or larger units, ordered by capacity (then fewer units)."""
    cur = mem_catalog.parse_mem_id(mem_id)
    t = next(x for x in mem_catalog.catalog_dict()["types"] if x["id"] == cur.mem_type)
    f = next(x for x in t["forms"] if x["id"] == cur.form)
    out = []
    for c in f["counts"]:
        if c["n"] < cur.n_units:
            continue
        if t["kind"] == "HBM":
            sizes = [dict(height=x["height"], die_Gb=x["die_Gb"]) for x in f["cap_tags"]]
        else:
            sizes = [dict(cap_GB=x["GB"]) for x in f["caps"].get(str(cur.unit_width_bits), [])]
        for sz in sizes:
            try:
                sp = mem_catalog.make_spec(cur.mem_type, form=cur.form, width_bits=cur.unit_width_bits,
                                           rate=cur.rate_MTps, count=c["n"], meta_mode=cur.meta_mode, **sz)
            except (ValueError, KeyError):
                continue
            if sp.capacity_GB > cur.capacity_GB:
                out.append(sp)
    out.sort(key=lambda sp: (sp.capacity_GB, sp.n_units))
    return out


def api_fit(body: dict) -> dict:
    """Capacity check with concrete fixes: fewest cards that hold the model, or the smallest larger memory."""
    scn = scenario_from_body(body)
    m = get_model(scn.model)
    r = evaluate(scn)
    r1 = r if scn.serving.batch == 1 else evaluate(scn.replace("serving.batch", 1))
    heavy = r1.stages[max(range(len(r1.stages)), key=lambda i: r1.stages[i].mem.dram_need)]
    out = {"fits": r.fits, "fits_batch1": r1.fits, "need_GiB": heavy.mem.dram_need / 2**30,
           "cap_GiB": heavy.mem.dram_cap / 2**30, "cards": scn.layout.cards, "weights_GiB": (heavy.mem.stored_w + heavy.mem.pipe_w) / 2**30,
           "pipe_w_GiB": heavy.mem.pipe_w / 2**30}
    if r.fits:
        return out
    if r1.fits:   # KV of a large batch overflows: largest batch that fits (capacity is monotone in batch)
        lo, hi = 1, scn.serving.batch
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if evaluate(scn.replace("serving.batch", mid)).fits else (lo, mid)
        out["max_batch"] = lo
        return out
    cards = scn.layout.cards
    for n in (2, 4, 8, 16, 32, 64):
        if n <= cards:
            continue
        fit = [lay for lay in enumerate_layouts(n, m.n_layers, m.is_moe, full=not m.kv_cache, pair=m.is_pair)
               if evaluate(scn.replace("layout", lay).replace("serving.batch", 1)).fits]
        if not fit:
            continue
        rows = search_layouts(scn, n, top=1)
        if rows and rows[0].batch:
            lay, batch = rows[0].layout, rows[0].batch
        else:   # fits but misses the TPOT SLO everywhere: smallest footprint layout at batch 1
            lay = min(fit, key=lambda l: max(s.mem.dram_need for s in
                                             evaluate(scn.replace("layout", l).replace("serving.batch", 1)).stages))
            batch = 1
        rr = evaluate(scn.replace("layout", lay).replace("serving.batch", batch))
        out["min_cards"] = {"cards": n, "layout": lay.label, "layout_obj": lay.__dict__, "batch": batch,
                            "need_GiB": max(s.mem.dram_need for s in rr.stages) / 2**30,
                            "meets_slo": bool(rows and rows[0].batch)}
        break
    for sp in _mem_candidates(scn.mem_id):
        if evaluate(scn.replace("mem_id", sp.id).replace("serving.batch", 1)).fits:
            out["min_mem"] = {"id": sp.id, "count": sp.n_units, "capacity_GiB": sp.capacity_GB,
                              "raw_GBps": sp.raw_GBps, "tag_zh": sp.tag_zh, "label": sp.short_label()}
            break
    return out


def api_stability(body: dict) -> dict:
    scenario_from_body(body), _cards(body), _objective(body)     # validate in-process
    if _POOL is not None:
        return _POOL.submit(_stability, body).result()
    return _stability(body)


def _stability(body: dict) -> dict:
    scn = scenario_from_body(body)
    st = ranking_stability(scn, _cards(body), include_mapping=bool(body.get("include_mapping", False)),
                           objective=_objective(body))
    return {"base_top": st.base_top, "stable": st.stable, "agree": st.agree, "cases": st.cases,
            "rule": "≥90% 扰动下 top-1 不变或与新 top-1 相差 ≤5%"}


def api_pareto(body: dict) -> dict:
    scn = scenario_from_body(body)
    return {"front": tpot_throughput_front(scn), "layout": scn.layout.label}


def api_sweep(body: dict) -> dict:
    scn = scenario_from_body(body)
    path = body.get("path")
    if path not in SWEEP_PATHS:
        raise ApiError(f"path must be one of {sorted(SWEEP_PATHS)}")
    vals = body.get("values")
    if not isinstance(vals, list) or not (1 <= len(vals) <= 32):
        raise ApiError("values must be a list of 1..32 numbers")
    typ = SWEEP_PATHS[path]
    rows = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ApiError("values must be numbers")
        if typ is int and float(v) != int(v):
            raise ApiError(f"{path} takes integers")
        try:
            r = evaluate(scn.replace(path, typ(v)))
        except (ValueError, TypeError) as e:
            raise ApiError(f"{path}={v}: {e}") from None
        dec = r.scenario.serving.phase == "decode" and r.workload is None
        llm = r.workload is None
        rows.append({"value": v, "tpot_ms": r.tpot * 1e3 if dec else None,
                     "ttft_ms": r.ttft * 1e3 if (llm and not dec) else None,
                     "latency_ms": None if llm else r.latency * 1e3, "tok_s": r.throughput,
                     "tok_s_card": r.per_card, "bound": r.bound, "fits": r.fits,
                     "array_util": r.stages[r.heaviest_stage].time.array_util})
    return {"path": path, "rows": rows, "base_hash": scn.hash()}
