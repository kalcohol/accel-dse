"""JSON API over core v2 (used by the web UI, the HTTP server and tests).

Every request carries a (partial) Scenario dict that is merged onto the
default scenario and then parsed **strictly** (unknown keys / non-finite
numbers rejected).  All sweeps are controlled replacements of that scenario.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from . import __version__, mem_catalog
from .core.catalog import DOMAINS, catalog_listing, get_model, labels, list_models, offline_entries, unlisted_models
from .core.budget import Budget, budget_report
from .core.disagg import disagg_report
from .core.energy import EnergyTable, energy_report, scaleup_bytes
from dataclasses import replace

from .core.evaluate import Result, evaluate
from .core.hardware import ACC_KIB_PER_ENGINE, CHIPS, Chip
from .core.mapping import ORG_LABEL, ORGS
from .core.parallel import Layout, enumerate_layouts
from .core.d2d_catalog import D2D_DEFAULT_STD, D2D_DEFAULT_UNITS, D2D_STANDARDS
from .core.fabric import fabric_report
from .core.scenario import Scenario, upgrade_legacy
from .core.search import best_batch, search_layouts, tpot_throughput_front
from .core.serving import goodput
from .core.vision import text_view
from .core.stability import ranking_stability

HONESTY = ("所有硬件参数（频率、阵列几何、SRAM 端口、DRAM 效率、链路 α/β、MAC 效率）均为「假设」，未经硅片标定；"
           "本工具用于设计指导与相对比较，不是性能承诺。")
MAX_CARDS = 8192          # 0.60: was 64 (= core.parallel.MAX_REPLICA_CARDS)
SWEEP_PATHS = {
    "serving.batch": int, "serving.ctx": int, "serving.prompt": int, "serving.spec_k": int,
    "serving.images": int, "serving.image_w": int, "serving.image_h": int,
    "serving.videos": int, "serving.video_frames": int, "serving.audio_s": float,
    "chip.sram_mib": float, "chip.sram_port_Bpc": float, "chip.freq_ghz": float, "chip.mac_eff": float,
    "chip.gemv_macs": int, "mem_eff": float, "link.GBps": float, "link.alpha_us": float,
    "serving.moe_skew": float, "chip.slc_mib": float, "chip.slc_GBps": float, "d2d.GBps": float, "d2d.alpha_us": float, "package_cards": int,
    "d2d_units": int, "net.GBps": float, "net.alpha_us": float, "node_cards": int, "fabric.oversub": float,
    "fabric.oversub_spine": float, "fabric.pod_nodes": int, "fabric.hop_spine_us": float, "fabric.hop_core_us": float,
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
    global _POOL, _POOL_N
    n = workers if workers is not None else min(8, os.cpu_count() or 1)    # 0.61: 8 (was 5)
    _POOL_N = n
    if n > 1 and _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_worker_init, initargs=(os.getpid(),))
    return n if _POOL is not None else 1


_POOL_N = 0
_POOL_LOCK = threading.Lock()


def reset_pool(broken) -> None:
    """0.129: a worker killed abruptly (OOM, signal) breaks a ProcessPoolExecutor for good — every later compare /
    layouts / fit / stability request was a 500 until restart.  Replace the broken pool (once per breakage)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is broken and _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None
            enable_pool(_POOL_N)


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


_COMMON_KEYS = {"chip_preset", "scenario", "energy", "budget"}
_ROUTE_KEYS = {"eval": {"best_batch", "goodput", "sensitivity"}, "fit": set(), "layouts": {"cards", "objective"},
               "compare": {"cards", "objective"}, "stability": {"cards", "objective", "include_mapping"},
               "pareto": {"sensitivity"}, "sweep": {"path", "values", "sensitivity"}, "sensitivity": set(),
               "insight": {"models"}, "insight_path": {"models", "metric", "steps", "max_cost", "beam"},
               "insight_sizing": {"models", "resources", "joint", "unpriced", "energy_tie"}}


def _check_keys(body, route: str) -> None:
    """0.85: reject unknown top-level body keys (README: strict parsing) — they were silently ignored, so a typo
    such as "objetive" ran the default objective."""
    if not isinstance(body, dict):
        raise ApiError("body must be a JSON object")
    bad = set(body) - _COMMON_KEYS - _ROUTE_KEYS[route]
    if bad:
        raise ApiError(f"{route}: unknown keys {sorted(bad)}; allowed {sorted(_COMMON_KEYS | _ROUTE_KEYS[route])}")


def scenario_from_body(body: dict) -> Scenario:
    if not isinstance(body, dict):
        raise ApiError("body must be a JSON object")
    _finite(body)
    sc = body.get("scenario", {})
    if not isinstance(sc, dict):
        raise ApiError("scenario must be an object")
    preset = body.get("chip_preset", "100T")
    if not isinstance(preset, str) or preset not in CHIPS:
        raise ApiError(f"chip_preset must be one of {sorted(CHIPS)}")
    base = Scenario(chip=CHIPS[preset]).to_dict(full=True)
    try:
        d = _merge(base, upgrade_legacy(sc))
        scn = Scenario.from_dict(d)
        get_model(scn.model)
        mem_catalog.parse_mem_id(scn.mem_id)
        if scn.pd.prefill_mem_id is not None:
            mem_catalog.parse_mem_id(scn.pd.prefill_mem_id)
    except ApiError:
        raise
    except (ValueError, TypeError, KeyError) as e:
        raise ApiError(str(e).strip("'\"")) from None
    if scn.layout.cards > MAX_CARDS or scn.pd.prefill_layout.cards > MAX_CARDS:
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
                     "sync": t.t_sync * 1e3, "total": t.total * 1e3, "slc": t.t_slc * 1e3},
            "dram_GB": {k: v / 1e9 for k, v in s.dram.items() if k != "slc_parts"},
            "slc_GB": {k: v / 1e9 for k, v in s.dram.get("slc_parts", {}).items()},
            "link_GB": {"total": t.link_bytes / 1e9, "d2d": t.d2d_bytes / 1e9, "net": t.net_bytes / 1e9,
                        "scaleup": scaleup_bytes(t) / 1e9,     # payload basis (timing); 0.78: wire = sent bytes
                        "wire": {"d2d": t.wire_d2d / 1e9, "link": t.wire_link / 1e9, "net": t.wire_net / 1e9}},
            "mem": {"stored_w_GiB": s.mem.stored_w / 2**30, "kv_GiB": s.mem.kv_total / 2**30,
                    "state_GiB": s.mem.state_total / 2**30, "need_GiB": s.mem.dram_need / 2**30,
                    "cap_GiB": s.mem.dram_cap / 2**30, "fits": s.mem.fits, "residency": s.mem.residency,
                    "kv_sram_MiB": s.mem.kv_sram / 2**20, "staging_MiB": s.mem.staging / 2**20,
                    "pipe_w_GiB": s.mem.pipe_w / 2**30, "slc_MiB": s.mem.slc / 2**20,
                    "slc_policy": s.mem.slc_policy, "slc_residency": s.mem.slc_residency, "slc_all": s.mem.slc_all,
                    "slc_w_MiB": (s.mem.slc_hot + s.mem.slc_expert) / 2**20, "slc_kv_MiB": s.mem.slc_kv / 2**20},
            "convert_Melems": s.convert_elems / 1e6, "tflops": t.flops / 1e12}


def _api_summary(r: Result) -> dict:
    """0.109 (API audit): the summary with the latency metrics that do not apply to the result as null, the rule the
    sweep rows already use — decode: ttft_ms / latency_ms; LLM prefill: tpot_ms / latency_ms; non-LLM workloads:
    tpot_ms / ttft_ms.  The core summary (and the fingerprints) still carries 0.0 for them; exported JSON / CSV of
    eval and sweep now agree instead of reporting "0 ms"."""
    s = r.summary()
    if r.workload is not None:
        na = ("tpot_ms", "ttft_ms")
    elif r.scenario.serving.phase == "decode":
        na = ("ttft_ms", "latency_ms")
    else:
        na = ("tpot_ms", "latency_ms")
    for k in na:
        s[k] = None
    return s


def result_dict(r: Result, *, with_goodput: bool = False) -> dict:
    s = r.scenario
    out = {"summary": _api_summary(r), "stages": [_stage_dict(x) for x in r.stages],
           "model": {"id": r.model.id, "hf_id": r.model.hf_id, **labels(r.model)},
           "hash": text_view(s).hash(), "scenario": text_view(s).to_dict()}   # 0.62: as entered (image tokens not folded in)
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
    return {"ok": True, "version": __version__, "core": "v2", "domains": ["llm", "vlm", "gen", "protein", "genomics"]}


def api_models() -> dict:
    return {"models": list_models(), "offline": offline_entries(), "catalog": [m["id"] for m in catalog_listing()],
            "unlisted": unlisted_models(), "domains": DOMAINS,
            "note": "按厂商 → 系列排列；图像 / 视频生成（DiT）与蛋白质（ESM-2 编码器、ESMFold / AlphaFold 2 / OpenFold / "
                    "Boltz-1 / Protenix 结构预测）已接入 v2 可评估；AlphaFold 3 权重需申请，仍为离线条目；"
                    "见 docs/MODEL.md §10–§12"}


def _density_presets() -> dict:
    from .core.budget import DENSITY_PRESETS
    return {k: {"label": v["label"], "fields": {f: {"value": x, "status": st, "source": src}
                                                for f, (x, st, src) in v["fields"].items()}}
            for k, v in DENSITY_PRESETS.items()}


def _energy_presets() -> dict:
    from .core.energy import ENERGY_PRESETS
    return {k: {"label": v["label"], "fields": {f: {"value": x, "status": st, "source": src}
                                                for f, (x, st, src) in v["fields"].items()}}
            for k, v in ENERGY_PRESETS.items()}


def _tco_presets() -> dict:
    from .core.budget import TCO_PRESETS
    return {k: {"label": v["label"], "fields": {f: {"value": x, "status": st, "source": src}
                                                for f, (x, st, src) in v["fields"].items()}}
            for k, v in TCO_PRESETS.items()}


def _electricity_presets() -> dict:
    from .core.budget import ELECTRICITY_PRESETS
    return {k: {"label": v["label"], "fields": {f: {"value": x, "status": st, "source": src}
                                                for f, (x, st, src) in v["fields"].items()}}
            for k, v in ELECTRICITY_PRESETS.items()}


def api_catalog() -> dict:
    from .core.dtypes import MIXED_PRESETS
    chips = {k: {**c.__dict__, "formats": [list(x) for x in c.formats.rates], "mixed": [list(x) for x in c.formats.mixed], "peak_tflops_bf16": c.peak_tflops_bf16,
                 "port_GBps": c.port_GBps, "gemv": c.gemv, "lanes": c.lanes,
                 "acc_total_kib": c.acc_total_kib, "acc_kib_per_engine_default": ACC_KIB_PER_ENGINE}
             for k, c in CHIPS.items()}
    return {"chips": chips, "mappings": [{"id": o, "label": ORG_LABEL[o]} for o in ORGS],
            "memory": mem_catalog.catalog_dict(), "defaults": Scenario().to_dict(full=True), "honesty": HONESTY,
            "sweep_paths": sorted(SWEEP_PATHS), "max_cards": MAX_CARDS,
            "d2d_standards": [{"id": k, **v} for k, v in D2D_STANDARDS.items()], "d2d_default_std": D2D_DEFAULT_STD,
            "d2d_default_units": D2D_DEFAULT_UNITS,
            "density_presets": _density_presets(),
            "energy_presets": _energy_presets(),
            "electricity_presets": _electricity_presets(),
            "tco_presets": _tco_presets(),
            "mixed_presets": {k: {"pairs": [list(p) for p in v[0]], "source": v[1]} for k, v in MIXED_PRESETS.items()}}


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


def _energy_table(body: dict) -> EnergyTable:
    try:
        d = body.get("energy")
        if isinstance(d, dict) and "energy_preset" in d:
            from .core.energy import apply_energy_preset
            d = apply_energy_preset(d)
        # 0.188: the card TDP for idle_frac_tdp comes from the budget's tdp_W_card when the energy table has none
        bud = body.get("budget")
        if isinstance(d, dict) and d.get("tdp_W") is None and isinstance(bud, dict) \
                and isinstance(bud.get("tdp_W_card"), (int, float)) and not isinstance(bud.get("tdp_W_card"), bool):
            d = {**d, "tdp_W": float(bud["tdp_W_card"])}
        return EnergyTable.from_dict(d)
    except (ValueError, TypeError) as e:
        raise ApiError(str(e)) from None


def _budget(body: dict) -> Budget | None:
    """Optional resource / area budget (0.49): body top-level ``budget`` (not part of the scenario)."""
    if body.get("budget") is None:
        return None
    try:
        d = body["budget"]
        if isinstance(d, dict):
            from .core.budget import apply_preset, apply_electricity_preset
            if "density_preset" in d:
                d = apply_preset(d)
            if "electricity_preset" in d:
                d = apply_electricity_preset(d)
            if "tco_preset" in d:
                from .core.budget import apply_tco_preset
                d = apply_tco_preset(d)
        return Budget.from_dict(d)
    except (ValueError, TypeError) as e:
        raise ApiError(str(e)) from None


def _budget_of(res, b: Budget | None, table: EnergyTable) -> dict | None:
    if b is None or res is None:
        return None
    return budget_report(res, b, energy_report(res, table) if b.power_W_card is not None else None)


def _flag_rows(dicts: list[dict], results: list, body: dict) -> list[dict]:
    """Budget flags on search / sweep rows; rows that break the budget are ranked after the rest (stable)."""
    b = _budget(body)
    if b is None:
        return dicts
    table = _energy_table(body)
    for d, res in zip(dicts, results):
        rep = _budget_of(res, b, table)
        d["budget_ok"] = rep["ok"] if rep else None
        d["budget_violations"] = rep["violations"] if rep else []
    return sorted(dicts, key=lambda d: d.get("budget_ok") is False)


def _pd_prefill_budget(scn, pd_rep: dict, bud: Budget, table: EnergyTable) -> dict:
    """0.61.4: the budget against the PD prefill pool — its own chip / memory / layout at its operating batch."""
    pd = scn.pd
    ps = scn.colocated().replace("layout", pd.prefill_layout).replace("serving.phase", "prefill") \
        .replace("serving.batch", max(1, int(pd_rep["prefill"]["batch"] or 1)))
    if pd.prefill_chip is not None:
        ps = ps.replace("chip", pd.prefill_chip)
    if pd.prefill_mem_id is not None:
        ps = ps.replace("mem_id", pd.prefill_mem_id)
    pr = evaluate(ps)
    rep = budget_report(pr, bud, energy_report(pr, table) if bud.power_W_card is not None else None)
    rep["pool"] = {"chip": ps.chip.name, "mem_id": ps.mem_id, "layout": ps.layout.label, "batch": ps.serving.batch}
    return rep


SENS_HINGE = 0.15     # 0.96: |Δ| above which a result "hinges" on the assumption 「假设」 (display threshold only)


def _lanes_q(s: Scenario) -> Scenario:
    ch = s.chip
    return s.replace("chip", replace(ch, vector_lanes=max(1, ch.lanes // 4)))


# 0.97: one definition shared by eval / sweep / Pareto. (key, short label, transform) — the transform is applied to
# each evaluated point, so a swept chip keeps "¼ of ITS lanes" and a swept memory keeps its own capacity.
# 0.99/0.100: overlap is bracketed — pipe (hardware-derived, MODEL §44; the default since 0.100) is the realistic
# estimate, stage (full overlap) the upper bound on throughput, serial the lower bound
SENS_CASES = (("pipe", "pipe（现实估计）", lambda s: None if s.exec_overlap in ("pipe", "serial") else s.replace("exec_overlap", "pipe")),
              ("exec_overlap", "全重叠（上界）", lambda s: None if s.exec_overlap == "stage" else s.replace("exec_overlap", "stage")),
              ("exec_overlap", "无重叠（下界）", lambda s: None if s.exec_overlap == "serial" else s.replace("exec_overlap", "serial")),
              ("mem_eff", "DRAM 效率 0.5", lambda s: s.replace("mem_eff", 0.5)),
              ("mem_eff", "DRAM 效率 0.9", lambda s: s.replace("mem_eff", 0.9)),
              ("vector_lanes", "向量 lane ÷4", _lanes_q))
SENS_SWEEP_SKIP = {"exec_overlap": "exec_overlap", "mem_eff": "mem_eff", "chip.vector_lanes": "vector_lanes"}


def sensitivity_report(scn: Scenario, r: Result) -> dict:
    """0.96: the same scenario under the three 「假设」 constants that drive NPU design conclusions (MODEL §41):
    no compute / memory / link overlap, DRAM efficiency 0.5 / 0.9, a quarter of the vector lanes."""
    eff = scn.mem_eff if scn.mem_eff is not None else evaluate_mem_eff(scn)
    cases = []
    ov = scn.exec_overlap
    if ov not in ("pipe", "serial"):     # 0.98/0.99: the hardware-derived overlap — the realistic alternative
        cases.append(("pipe", "pipe：硬件推导重叠（双缓冲 + 数据依赖，现实估计）", scn.replace("exec_overlap", "pipe")))
    if ov != "stage":
        cases.append(("exec_overlap", "stage：全重叠（吞吐上界）", scn.replace("exec_overlap", "stage")))
    if ov != "serial":
        cases.append(("exec_overlap", "serial：无重叠（吞吐下界）", scn.replace("exec_overlap", "serial")))
    for e in (0.5, 0.9):
        if abs(e - eff) > 1e-9:
            cases.append(("mem_eff", f"DRAM 效率 {e:g}（当前 {eff:g}）", scn.replace("mem_eff", e)))
    ch = scn.chip
    cases.append(("vector_lanes", f"向量 lane ÷4（{ch.lanes} → {max(1, ch.lanes // 4)}）", _lanes_q(scn)))
    base = r.throughput
    rows = []
    for key, label, s in cases:
        try:
            q = evaluate(s)
        except ValueError as e:
            rows.append({"key": key, "label": label, "error": str(e)})
            continue
        ratio = q.throughput / base if base else None
        rows.append({"key": key, "label": label, "tok_s": q.throughput, "ratio": ratio, "bound": q.bound,
                     "fits": q.fits, "bound_flip": q.bound != r.bound,
                     "hinge": bool(ratio is not None and (abs(ratio - 1) > SENS_HINGE or q.bound != r.bound))})
    hinges = sorted({x["key"] for x in rows if x.get("hinge")})
    return {"base_tok_s": base, "base_bound": r.bound, "base_label": SENS_BASE_LABEL.get(ov, ov), "threshold": SENS_HINGE, "rows": rows, "hinges": hinges,
            "domain": r.model.domain,
            # 0.97: what was re-evaluated (PD pools / budgets are not)
            "scope": "合并部署（PD 池未复核）" if scn.pd.enabled and r.model.kv_cache else "本场景"}


SENS_BASE_LABEL = {"stage": "stage 全重叠（吞吐上界）", "pipe": "pipe 硬件推导（现实估计）", "serial": "serial 无重叠（吞吐下界）"}
SENS_TIE = 0.01     # 0.97: points within 1 % are ties (a flat roofline is not a ranking) 「假设」
SENS_KNEE = 0.05    # 0.97: knee = first swept value within 5 % of the sweep's best ("enough SRAM / clock") 「假设」


def _knee(vals: list):
    m = max((v for v in vals if v is not None), default=None)
    return None if m is None else next(i for i, v in enumerate(vals) if v is not None and v >= m * (1 - SENS_KNEE))


def _near_best(vals: list) -> set:
    m = max((v for v in vals if v is not None), default=None)
    return set() if m is None else {i for i, v in enumerate(vals) if v is not None and v >= m * (1 - SENS_TIE)}


def _discordant(a: list, b: list) -> list:
    """Pairs (i, j) that are ordered one way in a and the other way in b, each by more than SENS_TIE."""
    def gt(x, y):
        return x > y * (1 + SENS_TIE)
    out = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if None in (a[i], a[j], b[i], b[j]):
                continue
            if (gt(a[i], a[j]) and gt(b[j], b[i])) or (gt(a[j], a[i]) and gt(b[i], b[j])):
                out.append((i, j))
    return out


def sweep_sensitivity(scns: list, base_rows: list, path: str, key_of=lambda d: d["tok_s_card"]) -> dict:
    """0.97: re-run every sweep point under each SENS_CASES alternative. A conclusion flips when the best value
    changes (no overlap between the base and alternative near-best sets, ties within SENS_TIE), when a pair of
    points swaps order by more than SENS_TIE (rank flip), or when a point's bottleneck changes."""
    skip = SENS_SWEEP_SKIP.get(path)
    base = [key_of(d) if d.get("fits") else None for d in base_rows]
    b0, k0 = _near_best(base), _knee(base)
    cases, row_flags = [], [[] for _ in base_rows]
    for key, label, f in SENS_CASES:
        if key == skip or (skip == "exec_overlap" and key == "pipe"):
            continue
        ts = [f(s) for s in scns]
        if all(t is None for t in ts):
            continue                                  # case does not apply (already serial)
        alt, bounds = [], []
        for t in ts:
            if t is None:
                alt.append(None); bounds.append(None)
                continue
            try:
                q = evaluate(t)
                alt.append(q.per_card if q.fits else None); bounds.append(q.bound)
            except ValueError:
                alt.append(None); bounds.append(None)
        b1, disc, k1 = _near_best(alt), _discordant(base, alt), _knee(alt)
        inpair = {k for p in disc for k in p}
        rows = []
        for i, d in enumerate(base_rows):
            rf = i in inpair
            bf = bounds[i] is not None and bounds[i] != d["bound"]
            if rf or bf:
                row_flags[i].append(label)
            rows.append({"value": d["value"], "thr_card": alt[i], "ratio": (alt[i] / base[i]) if alt[i] and base[i] else None,
                         "bound": bounds[i], "rank_flip": rf, "bound_flip": bf})
        cases.append({"key": key, "label": label, "best_values": [base_rows[i]["value"] for i in sorted(b1)],
                      "best_changed": bool(b0 and b1 and not (b0 & b1)), "flips": len(disc),
                      "knee_value": base_rows[k1]["value"] if k1 is not None else None,
                      "knee_changed": k0 is not None and k1 is not None and k1 != k0, "rows": rows})
    return {"best_values": [base_rows[i]["value"] for i in sorted(b0)], "tie": SENS_TIE, "knee_tol": SENS_KNEE,
            "knee_value": base_rows[k0]["value"] if k0 is not None else None, "cases": cases, "row_flags": row_flags,
            "hinges": sorted({c["key"] for c in cases if c["best_changed"] or c["flips"] or c["knee_changed"]})}


def pareto_sensitivity(scn: Scenario, front: list) -> dict:
    """0.97: the latency–throughput front under each alternative; a base front point that is dominated there is
    flagged (its "best trade-off" status hinges on that assumption)."""
    from .core.search import tpot_throughput_front
    batches = [p["batch"] for p in front]
    cases, flags = [], [[] for _ in front]
    for key, label, f in SENS_CASES:
        t = f(scn)
        if t is None:
            continue
        try:
            alt = tpot_throughput_front(t)
        except ValueError as e:
            cases.append({"key": key, "label": label, "error": str(e)})
            continue
        on = {p["batch"] for p in alt}
        off = [b for b in batches if b not in on]
        for i, b in enumerate(batches):
            if b not in on:
                flags[i].append(label)
        cases.append({"key": key, "label": label, "front_batches": sorted(on), "off_front": off,
                      "new_on_front": sorted(on - set(batches))})
    return {"cases": cases, "point_flags": flags,
            "hinges": sorted({c["key"] for c in cases if c.get("off_front") or c.get("new_on_front")})}


def evaluate_mem_eff(scn: Scenario) -> float:
    from .core.hardware import System
    return System(scn.chip, scn.mem_id, None).mem.efficiency


def api_eval(body: dict) -> dict:
    _check_keys(body, "eval")
    scn = scenario_from_body(body)
    bud = _budget(body)
    bb = None
    if body.get("best_batch"):
        bb = best_batch(scn)
        if bb.batch:
            scn = scn.replace("serving.batch", bb.batch)
    table = _energy_table(body)
    try:
        r = evaluate(scn)
    except ValueError as e:
        raise ApiError(str(e)) from None
    out = result_dict(r, with_goodput=bool(body.get("goodput", True)))
    if bb is not None and not bb.batch:     # 0.61.3: say so instead of silently evaluating the entered batch
        out["summary"]["warnings"].append(f"最佳 batch：没有既放得下又满足 SLO 的 batch，按输入的 batch {scn.serving.batch} 评估")
    out["energy"] = energy_report(r, table)      # 0.47.1: action counts always; J only for user-supplied entries
    if body.get("sensitivity", True):
        out["sensitivity"] = sensitivity_report(scn, r)
    if bud is not None:
        out["budget"] = budget_report(r, bud, out["energy"])
    if scn.fabric.enabled:  # 0.59: per-collective algorithm / topology breakdown
        out["fabric"] = fabric_report(scn, r)
    if scn.pd.enabled and r.model.kv_cache:   # 0.50 PD next to colocated; 0.90: video / protein → warning only (no pd block)
        try:
            out["pd"] = disagg_report(scn, r, _energy_table(body))
        except ValueError as e:
            out["pd"] = {"error": str(e)}
        if bud is not None and bud.provided:
            out["budget"]["scope_note"] = ("上表核对合并部署 / PD decode 池（芯片 " + scn.chip.name + "、" + scn.layout.label + "）"
                                           + ("；PD prefill 池见 pd.budget" if "error" not in out["pd"] else ""))
            if "error" not in out["pd"]:
                out["pd"]["budget"] = _pd_prefill_budget(scn, out["pd"], bud, table)
                # the deployment is within budget only if both pools are
                oks = (out["budget"]["ok"], out["pd"]["budget"]["ok"])
                out["budget"]["ok_with_pd"] = False if False in oks else (None if None in oks else True)
    return out


def api_layouts(body: dict) -> dict:
    _check_keys(body, "layouts")
    scn = scenario_from_body(body)
    obj = _objective(body)
    stats: dict = {}
    rows = search_layouts(scn, _cards(body), objective=obj, top=16, stats=stats, pool=_POOL)
    return {"objective": obj, "mapping": scn.mapping,
            "rows": _flag_rows([_row(r, obj) for r in rows], [r.result for r in rows], body),
            "n_layouts": stats["layouts"], "stats": stats}


def _compare_row(body: dict, org: str) -> dict:
    scn = scenario_from_body(body).replace("mapping", org)
    obj = _objective(body)
    bud = _budget(body)
    rows = search_layouts(scn, _cards(body), objective=obj, top=2 if bud is None else 8, pool=_POOL)
    if bud is not None:     # best layout within the budget (searched without it; flagged rows ranked last)
        table = _energy_table(body)
        ok = [r for r in rows if (_budget_of(r.result, bud, table) or {}).get("ok") is not False]
        rows = ok + [r for r in rows if r not in ok]
    top = rows[0]
    d = _row(top, obj)
    d["label"] = ORG_LABEL[org]
    d["runner_up"] = _row(rows[1], obj) if len(rows) > 1 else None
    if bud is not None:
        _flag_rows([d], [top.result], body)
        if d["runner_up"]:
            _flag_rows([d["runner_up"]], [rows[1].result], body)
    if obj == "decode" and top.result is not None and scn.serving.phase == "decode" and top.result.workload is None:
        g = goodput(top.result)
        d.update(goodput_card=g.goodput_per_card, ttft_ms=g.ttft_ms, ttft_ok=g.ttft_ok)
    return d


def api_compare(body: dict) -> dict:
    """Best layout per mapping organisation (the mapping comparison view)."""
    _check_keys(body, "compare")
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


FIT_SCAN = 32     # 0.127: non-power-of-two card counts scanned below the first fitting power of two


def api_fit(body: dict) -> dict:
    """Capacity check with concrete fixes: fewest cards that hold the model, or the smallest larger memory."""
    _check_keys(body, "fit")
    scn = scenario_from_body(body)
    m = get_model(scn.model)
    try:                                  # 0.74: invalid layout / scenario → 400 (was an uncaught ValueError → 500)
        r = evaluate(scn)
        r1 = r if scn.serving.batch == 1 else evaluate(scn.replace("serving.batch", 1))
    except ValueError as e:
        raise ApiError(str(e)) from None
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
    bud = _budget(body)
    scn = scn.colocated()                 # 0.56: other layouts need not divide pd.decode_cards
    def fits_on(n: int) -> list:
        return [lay for lay in enumerate_layouts(n, m.n_layers, m.is_moe, full=not m.kv_cache, pair=m.is_pair)
                if evaluate(scn.replace("layout", lay).replace("serving.batch", 1)).fits]

    prev = cards
    for n in [2 ** i for i in range(1, MAX_CARDS.bit_length())]:    # 0.60: up to MAX_CARDS (was ≤ 64)
        if n <= cards:
            continue
        if bud is not None and bud.cards is not None and n > bud.cards:
            out["budget_cards_limit"] = int(bud.cards)      # more cards would break the card budget
            break
        fit = fits_on(n)
        if not fit:
            prev = n
            continue
        # 0.127 review: the fewest cards need not be a power of two (Llama-3.3-70B: TP3 holds it, 4 was reported) —
        # scan the card counts between the last failing power of two and n (bounded: ≤ FIT_SCAN counts)
        pow2 = n
        for n2 in range(prev + 1, min(n, prev + 1 + FIT_SCAN)):
            f2 = fits_on(n2)
            if f2:
                n, fit = n2, f2
                break
        rows = search_layouts(scn, n, top=1, pool=_POOL)
        if rows and rows[0].batch:
            lay, batch = rows[0].layout, rows[0].batch
        else:   # fits but misses the TPOT SLO everywhere: smallest footprint layout at batch 1
            lay = min(fit, key=lambda l: max(s.mem.dram_need for s in
                                             evaluate(scn.replace("layout", l).replace("serving.batch", 1)).stages))
            batch = 1
        rr = evaluate(scn.replace("layout", lay).replace("serving.batch", batch))
        out["min_cards"] = {"cards": n, "layout": lay.label, "layout_obj": lay.__dict__, "batch": batch,
                            "need_GiB": max(s.mem.dram_need for s in rr.stages) / 2**30,
                            "meets_slo": bool(rows and rows[0].batch), "pow2_cards": pow2}
        break
    for sp in _mem_candidates(scn.mem_id):
        if evaluate(scn.replace("mem_id", sp.id).replace("serving.batch", 1)).fits:
            out["min_mem"] = {"id": sp.id, "count": sp.n_units, "capacity_GiB": sp.capacity_GB,
                              "raw_GBps": sp.raw_GBps, "tag_zh": sp.tag_zh, "label": sp.short_label()}
            break
    return out


def api_stability(body: dict) -> dict:
    _check_keys(body, "stability")
    scenario_from_body(body), _cards(body), _objective(body)     # validate in-process
    return _stability(body)


def _stability(body: dict) -> dict:
    scn = scenario_from_body(body)
    # 0.61: the perturbation cases (and the base search's brackets) fan out over the pool (was one job per request)
    st = ranking_stability(scn, _cards(body), include_mapping=bool(body.get("include_mapping", False)),
                           objective=_objective(body), pool=_POOL)
    return {"base_top": st.base_top, "stable": st.stable, "agree": st.agree, "cases": st.cases,
            "rule": "≥90% 扰动下 top-1 不变或与新 top-1 相差 ≤5%"}


def api_pareto(body: dict) -> dict:
    _check_keys(body, "pareto")
    scn = scenario_from_body(body)
    out = {"front": tpot_throughput_front(scn), "layout": scn.layout.label}
    if body.get("sensitivity"):          # 0.97 opt-in (≈ 4× the cost)
        out["sensitivity"] = pareto_sensitivity(scn, out["front"])
    return out


def _sweep_inapplicable(path: str, r) -> str | None:
    """0.129: a knob the model ignores swept to a flat line with HTTP 200 (e.g. workload.frames on an LLM,
    serving.moe_skew on a dense model, serving.spec_k without MTP / draft) — read as "insensitive"."""
    m = r.model
    if path.startswith("workload.") and r.workload is None:
        return "仅适用于视频 / 蛋白质 / 基因组工作负载模型（该模型为 LLM）"
    if path in ("serving.images", "serving.image_w", "serving.image_h") and m.vision is None:
        return "该模型没有视觉塔（纯文本）"
    if path == "serving.audio_s" and getattr(m.vision, "audio", None) is None:
        return "该模型没有音频编码器"
    if path in ("serving.videos", "serving.video_frames"):
        from .core.vision import video_pre
        if video_pre(m) is None:
            return "该模型没有已建模的视频处理器（Qwen3VLVideoProcessor 系）"
    if path == "serving.moe_skew" and not m.is_moe:
        return "仅适用于 MoE 模型"
    sc = r.scenario
    if path == "serving.moe_skew" and sc.layout.ep <= 1 and not (sc.pd.enabled and sc.pd.prefill_layout.ep > 1):
        return "专家负载倾斜只在 EP > 1 时起作用（当前 EP = 1）"      # 0.130
    if path in ("serving.ctx", "serving.prompt") and r.workload is not None:
        return "仅适用于 LLM / VLM（该模型为固定工作负载）"
    if path == "serving.prompt" and sc.serving.phase == "decode" and not sc.pd.enabled:
        return "decode 阶段的结果不读取 prompt 长度（改 ctx，或切到 prefill 阶段）"
    if path == "serving.spec_k" and not (m.mtp_layers or m.draft):
        return "该模型没有 MTP / 草稿模块（spec_k 被忽略）"
    w = m.workload
    if r.workload is not None and w is not None and path.startswith("workload."):     # 0.130: per workload kind (frames on an image DiT, …)
        k = path.split(".", 1)[1]
        pair = bool(m.is_pair)
        ok = {"frames": w.kind == "gen" and w.frames > 1, "height": w.kind == "gen", "width": w.kind == "gen",
              "steps": w.kind == "gen" or (pair and bool(w.diff_steps)),
              "seq_len": w.kind in ("protein", "genomics"), "msa": pair and bool(w.msa or w.xmsa),
              "recycles": pair, "samples": pair and bool(w.diff_steps)}.get(k, True)
        if not ok:
            return f"该模型的工作负载（{w.kind}）不读取 {k}"
    return None


def api_sweep(body: dict) -> dict:
    _check_keys(body, "sweep")
    scn = scenario_from_body(body)
    path = body.get("path")
    if not isinstance(path, str) or path not in SWEEP_PATHS:
        raise ApiError(f"path must be one of {sorted(SWEEP_PATHS)}")
    vals = body.get("values")
    if not isinstance(vals, list) or not (1 <= len(vals) <= 32):
        raise ApiError("values must be a list of 1..32 numbers")
    typ = SWEEP_PATHS[path]
    rows, results, scns = [], [], []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ApiError("values must be numbers")
        if typ is int and float(v) != int(v):
            raise ApiError(f"{path} takes integers")
        try:
            s_v = scn.replace(path, typ(v))
            if path == "package_cards" and v > 1 and not s_v.d2d_enabled:
                s_v = s_v.replace("d2d_enabled", True)       # sweeping the package size = chiplets with D2D
            elif path == "d2d.GBps" and s_v.d2d_std != "custom":
                s_v = s_v.replace("d2d_std", "custom")       # a swept D2D figure replaces the catalog grade
            elif path == "d2d_units" and not s_v.d2d_enabled:
                raise ApiError("d2d_units sweep needs d2d_enabled = true and package_cards > 1")
            r = evaluate(s_v)
        except (ValueError, TypeError) as e:
            raise ApiError(f"{path}={v}: {e}") from None
        if not results and (why := _sweep_inapplicable(path, r)):
            raise ApiError(f"{path}: {why}")
        scns.append(s_v)
        dec = r.scenario.serving.phase == "decode" and r.workload is None
        llm = r.workload is None
        rows.append({"value": v, "tpot_ms": r.tpot * 1e3 if dec else None,
                     "ttft_ms": r.ttft * 1e3 if (llm and not dec) else None,
                     "latency_ms": None if llm else r.latency * 1e3, "tok_s": r.throughput,
                     "tok_s_card": r.per_card, "bound": r.bound, "fits": r.fits,
                     "array_util": r.stages[r.heaviest_stage].time.array_util})
        results.append(r)
    if _budget(body) is not None:       # flag only — sweep rows keep their order
        order = {id(d): i for i, d in enumerate(rows)}
        rows = sorted(_flag_rows(rows, results, body), key=lambda d: order[id(d)])
    out = {"path": path, "rows": rows, "base_hash": scn.hash()}
    if body.get("sensitivity"):          # 0.97 opt-in (≈ 4× the cost)
        out["sensitivity"] = sweep_sensitivity(scns, rows, path)
    return out


def api_sensitivity(body: dict) -> dict:
    """0.97: the eval sensitivity block alone, so the UI can render the main result first and fill the 敏感度
    panel afterwards (DiT points cost ≈ 5× the plain eval)."""
    _check_keys(body, "sensitivity")
    scn = scenario_from_body(body)
    try:
        r = evaluate(scn)
    except ValueError as e:
        raise ApiError(str(e)) from None
    return {"sensitivity": sensitivity_report(scn, r), "hash": scn.hash()}


def _insight_models(body: dict) -> list:
    """0.151: any catalog domain; the set must share one domain (llm / gen / protein / genomics)."""
    from .core.insight import DEFAULT_MODELS, DEFAULT_MODELS_BY_DOMAIN, MAX_MODELS, insight_domain
    models = body.get("models")
    if models is None:
        models = list(DEFAULT_MODELS)
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models) or not 1 <= len(models) <= MAX_MODELS:
        raise ApiError(f"models must be a list of 1..{MAX_MODELS} model ids")
    for m in models:
        try:
            get_model(m)
        except (KeyError, ValueError):
            raise ApiError(f"unknown model {m!r}") from None
    try:
        insight_domain(models)          # raises ValueError on mixed / unsupported
    except ValueError as e:
        raise ApiError(str(e)) from None
    return models


def api_insight_path(body: dict) -> dict:
    """0.148: greedy multi-step upgrade path (gain per area / 「假设」 relative cost), never worse than the start."""
    from .core.insight import upgrade_path
    _check_keys(body, "insight_path")
    scn = scenario_from_body(body)
    models = _insight_models(body)
    try:
        return {**upgrade_path(scn, models, body.get("metric"), _budget(body), body.get("steps", 6),
                               body.get("max_cost"), _energy_table(body), body.get("beam", 1)), "hash": scn.hash()}
    except ValueError as e:
        raise ApiError(str(e)) from None


def api_insight_sizing(body: dict) -> dict:
    """0.157: datapath sizing headroom — smallest dequant rate / vector width / SRAM keeping every model within ±1 %."""
    from .core.insight import sizing_report
    _check_keys(body, "insight_sizing")
    scn = scenario_from_body(body)
    models = _insight_models(body)
    res = body.get("resources")
    if res is not None and (not isinstance(res, list) or not all(isinstance(x, str) for x in res)):
        raise ApiError("resources must be a list of strings")
    try:
        return {**sizing_report(scn, models, _budget(body), res, joint=bool(body.get("joint", False)), unpriced=body.get("unpriced", "last"),
                                     table=_energy_table(body), energy_tie=body.get("energy_tie", False) is True), "hash": scn.hash()}
    except ValueError as e:
        raise ApiError(str(e)) from None


def api_insight(body: dict) -> dict:
    """0.146/0.151: NPU design insight — which hardware knob moves the domain's native metrics most for a model set
    (LLM: TTFT/TPOT/tok/s; gen: latency/frames/s; protein: latency/seq/s; genomics: latency/windows/s), per area /
    power proxy.  See core/insight.py."""
    from .core.insight import DEFAULT_MODELS, DEFAULT_MODELS_BY_DOMAIN, insight_report
    _check_keys(body, "insight")
    scn = scenario_from_body(body)
    models = _insight_models(body)
    try:
        return {**insight_report(scn, models, _budget(body), _energy_table(body)), "hash": scn.hash(),
                "default_models": list(DEFAULT_MODELS),
                "default_models_by_domain": {k: list(v) for k, v in DEFAULT_MODELS_BY_DOMAIN.items()}}
    except ValueError as e:
        raise ApiError(str(e)) from None
