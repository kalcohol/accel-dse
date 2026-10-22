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
from . import mem_catalog
from .core.energy import EnergyTable, energy_report, scaleup_bytes
from dataclasses import replace

from .core.evaluate import Result, evaluate
from .core.hardware import ACC_KIB_PER_ENGINE, CHIPS, Chip
from .core.mapping import ORG_LABEL, ORGS
from .core.parallel import Layout, enumerate_layouts
from .core.schedule import tree_tokens
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
    "chip.sram_mib": float, "chip.state_sram_mib": float, "chip.clk_ungated_share": float, "chip.sram_port_Bpc": float, "chip.freq_ghz": float, "chip.mac_eff": float,
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
_ROUTE_KEYS = {"eval": {"best_batch", "goodput", "sensitivity", "power_cap", "audio_out_s", "cooling", "power_scope", "tco"}, "fit": set(), "layouts": {"cards", "objective"},
               "compare": {"cards", "objective"}, "stability": {"cards", "objective", "include_mapping"},
               "pareto": {"sensitivity"}, "sweep": {"path", "values", "sensitivity"}, "sensitivity": {"power_cap_W"},
               "insight": {"models"}, "insight_path": {"models", "metric", "steps", "max_cost", "beam"},
               "insight_sizing": {"models", "resources", "joint", "unpriced", "energy_tie"},
               "insight_slc": {"models", "policy", "gibs", "temps"},
               "insight_speech": {"models", "audio_s", "verify", "dispatch"},
               "insight_speech_streams": {"models", "audio_s", "slc_gib"},
               "insight_dispatch": {"models", "batches", "ctx"},
               "insight_noc": {"topologies", "phases", "prompt", "noc_link_Bpc", "noc_routing"},
               "insight_dse": {"mac_scales", "sram_scales", "sides", "topologies", "power_mode", "curve", "phase",
                               "prompt", "tie", "mems", "mem_types", "mem_count_scales", "mem_caps", "slc_mibs",
                               "mix_out_tokens", "slo_tpot_ms", "slo_ttft_ms", "mem_usd_per_GB", "max_mem_usd", "f_scales",
                               "noc_link_Bpc", "noc_routing", "noc_links", "dies", "d2d_stds", "d2d_units",
                               "die_split", "reticle_mm2", "pkg_mm2", "d0_per_cm2", "mono_clusters",
                               "mem_edges_shoreline", "clusters", "cost", "include_unbuildable", "shore_strict",
                               "cooling", "power_scope", "tco", "tco_sens_row"}}


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
        sc = upgrade_legacy(sc)
        if isinstance(sc.get("chip"), str) and sc["chip"] in CHIPS and "mem_id" not in sc:
            from .core.hardware import default_mem
            sc = {**sc, "mem_id": default_mem(sc["chip"])}     # 0.205: a preset name brings its memory pairing
        d = _merge(base, sc)
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
                    "kv_sram_MiB": s.mem.kv_sram / 2**20, "state_sram_MiB": max(s.mem.state_sram, 0.0) / 2**20, "staging_MiB": s.mem.staging / 2**20,
                    "pipe_w_GiB": s.mem.pipe_w / 2**30, "slc_MiB": s.mem.slc / 2**20,
                    "slc_policy": s.mem.slc_policy, "slc_residency": s.mem.slc_residency, "slc_all": s.mem.slc_all,
                    "slc_frac": s.mem.slc_frac, "slc_frac_exp": s.mem.slc_frac_exp, "slc_frac_exp_bytes": s.mem.slc_frac_exp_bytes,
                    "slc_w_MiB": (s.mem.slc_hot + s.mem.slc_expert) / 2**20, "slc_kv_MiB": s.mem.slc_kv / 2**20,
                    # 0.286: with a measured expert load, the share of expert reads the SRAM-pinned experts serve
                    **({"sram_exp_hit": s.mem.sram_exp_hit} if s.mem.sram_exp_hit >= 0 else {}),
                    # 0.293 MoVA value experts (routed hot class): pinned bytes and the read shares SRAM / SLC serve
                    **({"vexp_sram_MiB": s.mem.pinned_rt / 2**20, "sram_vexp_hit": s.mem.sram_rt_hit,
                        "slc_vexp_MiB": s.mem.slc_rt / 2**20, "slc_frac_vexp": s.mem.slc_frac_rt}
                       if s.mem.sram_rt_hit >= 0 else {})},
            "convert_Melems": s.convert_elems / 1e6, "tflops": t.flops / 1e12,
            # 0.290: weight-dequant vector ms and whether it decides a VECTOR-bound stage (≥ half its vector time)
            "dequant_ms": s.conv_w * 1e3, "dequant_bound": dequant_bound(s)}


def dequant_bound(s) -> bool:
    """0.290: a VECTOR-bound stage whose vector time is mostly weight dequant (conversion of packed / codebook weights
    on the vector path, StageResult.conv_w) — the UI tags it 「反量化」."""
    return s.time.bound == "VECTOR" and s.conv_w >= 0.5 * s.time.t_vector > 0


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
    out = {"summary": _api_summary(r), "stages": [_stage_dict(x) for x in r.stages], "walls": r.walls(), "balance": r.balance(),
           "dispatch_sens": r.dispatch_sens(),     # 0.284 per-op dispatch sensitivity of the step
           "model": {"id": r.model.id, "hf_id": r.model.hf_id, **labels(r.model)},
           "hash": text_view(s).hash(), "scenario": text_view(s).to_dict()}   # 0.62: as entered (image tokens not folded in)
    out["shoreline"] = r.shoreline     # 0.312: buildability (per-edge memory / D2D / I/O shoreline)
    if r.spec:
        out["spec"] = r.spec        # 0.225: τ of the spec step (+ tree band and published-reference check)
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
    # 0.269: derived from the shared catalog.DOMAINS (was a hard-coded list that missed asr / tts)
    return {"ok": True, "version": __version__, "core": "v2", "domains": list(DOMAINS)}


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
    from .core.hardware import MEM_LEGACY_0204, default_mem
    from .core.dtypes import DEQUANT_OPS, MIXED_PRESETS
    chips = {k: {**c.__dict__, "formats": [list(x) for x in c.formats.rates], "mixed": [list(x) for x in c.formats.mixed], "peak_tflops_bf16": c.peak_tflops_bf16,
                 "port_GBps": c.port_GBps, "gemv": c.gemv, "lanes": c.lanes,
                 "acc_total_kib": c.acc_total_kib, "acc_kib_per_engine_default": ACC_KIB_PER_ENGINE,
                 "default_mem": default_mem(c)}
             for k, c in CHIPS.items()}
    return {"chips": chips, "mappings": [{"id": o, "label": ORG_LABEL[o]} for o in ORGS],
            "memory": mem_catalog.catalog_dict(), "defaults": Scenario().to_dict(full=True), "honesty": HONESTY,
            "sweep_paths": sorted(SWEEP_PATHS), "max_cards": MAX_CARDS, "mem_legacy_default": MEM_LEGACY_0204,
            "d2d_standards": [{"id": k, **v} for k, v in D2D_STANDARDS.items()], "d2d_default_std": D2D_DEFAULT_STD,
            "d2d_default_units": D2D_DEFAULT_UNITS,
            "density_presets": _density_presets(),
            "energy_presets": _energy_presets(),
            "electricity_presets": _electricity_presets(),
            "tco_presets": _tco_presets(),
            "mixed_presets": {k: {"pairs": [list(p) for p in v[0]], "source": v[1]} for k, v in MIXED_PRESETS.items()},
            "dequant_ops": dict(DEQUANT_OPS)}


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


def sensitivity_report(scn: Scenario, r: Result, power_cap_W: float | None = None) -> dict:
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
    if not ch.dispatch and not ch.kernel_us and not ch.gpu:   # 0.203: NPU dispatch is 「假设」 0 by default — scan the modes
        from .core.hardware import DISPATCH_US
        for mode, lab in (("static", "每步静态图启动 + 同步"), ("trace", "每算子设备回放派发"), ("host", "每算子主机派发")):
            cases.append(("dispatch", f"派发 {mode}：{lab} {DISPATCH_US[mode]:g} µs「二手」",
                          scn.replace("chip", replace(ch, dispatch=mode))))
    base = r.throughput

    def row(q):
        ratio = q.throughput / base if base else None
        return {"tok_s": q.throughput, "ratio": ratio, "bound": q.bound, "fits": q.fits, "bound_flip": q.bound != r.bound,
                "hinge": bool(ratio is not None and (abs(ratio - 1) > SENS_HINGE or q.bound != r.bound))}
    rows = _eval_cases(cases, row)
    hinges = sorted({x["key"] for x in rows if x.get("hinge")})
    return {"base_tok_s": base, "base_bound": r.bound, "base_label": SENS_BASE_LABEL.get(ov, ov), "threshold": SENS_HINGE, "rows": rows, "hinges": hinges,
            "domain": r.model.domain,
            # 0.97: what was re-evaluated (PD pools / budgets are not)
            "scope": "合并部署（PD 池未复核）" if scn.pd.enabled and r.model.kv_cache else "本场景",
            "balance_whatif": balance_whatif(scn, r, power_cap_W)}


def _eval_cases(cases, row_fn) -> list:
    """0.208: the sensitivity scan's case loop, shared by sensitivity_report and balance_whatif — each (key, label,
    scenario) is evaluated once; a rejected case becomes an error row; row_fn(result) adds the per-panel fields."""
    rows = []
    for key, label, s in cases:
        try:
            q = evaluate(s)
        except ValueError as e:
            rows.append({"key": key, "label": label, "error": str(e)})
            continue
        rows.append({"key": key, "label": label, **row_fn(q)})
    return rows


WHATIF_GAIN = 0.10    # 0.208: a lever "works" when it makes balance reachable, moves the asymptote / batch limit up or TPOT down by > 10 % 「假设」


def _mem_x2(mem_id: str):
    """0.208: the same memory with twice the channels / stacks (2× raw bandwidth); None if the id is not of that form."""
    import re
    m = re.match(r"^([a-z0-9]+)_(\d+)(x\d+|s)_(.*)$", mem_id)
    return None if not m else f"{m.group(1)}_{2 * int(m.group(2))}{m.group(3)}_{m.group(4)}"


# 0.211: lever cost (area / power) on the sourced presets — the same accounting as the budget / energy panels
WHATIF_DENSITY = "n5_derived"      # 1P is a 5 nm-class design point 「假设」; SRAM density 推导 (TPUv4i CMEM × N5 bitcell)
WHATIF_ENERGY = "tpuv4i_7nm"       # newest energy preset (推导 / 来源 per field)
RETICLE_MM2 = 26 * 33              # 858 mm² single-exposure reticle field (26 × 33 mm, ASML / industry standard) 「来源」


def _tok_s_card(q) -> float | None:
    """0.217: decode output tokens / s per card (batch / TPOT / cards) — J/token × this ≈ W_card."""
    return q.scenario.serving.batch / q.tpot / q.scenario.layout.cards if q.tpot else None


def _whatif_table() -> EnergyTable:
    from .core.energy import ENERGY_PRESETS
    return EnergyTable.from_dict({k: v for k, (v, _, _) in ENERGY_PRESETS[WHATIF_ENERGY]["fields"].items()
                                  if k in EnergyTable.__dataclass_fields__ and k not in ("idle_W", "idle_frac_tdp", "tdp_W")})


def lever_power_cap(q, c: dict, cap_W: float) -> dict:
    """0.227: does a cap bind at this operating point (W/card of the what-if energy basis), and what is left of the
    throughput: dfs (clock only — analytic (cap − P_static) / P_dynamic without a per-cycle term; re-evaluated with
    one, 0.228 — the headline) and the rx5700xt_n7 V–f
    curve (re-evaluated, only when the cap binds) — core/powercap.py."""
    from .core.powercap import dvfs_cap
    W = c.get("W_card")
    if W is None:
        return {"error": "无能耗"}
    ps = (c.get("leak_W") or 0.0) + (c.get("hbm_W") or 0.0)
    tok = _tok_s_card(q)
    if ps >= cap_W:
        return {"cap_W": cap_W, "W_card": W, "P_static_W": ps, "binds": True, "feasible": False,
                "tf_dfs": 0.0, "tok_s_card_dfs": 0.0, "tf_vf": 0.0, "tok_s_card_vf": 0.0}
    binds = W > cap_W
    tf = min(1.0, (cap_W - ps) / (W - ps)) if W > ps else 1.0
    out = {"cap_W": cap_W, "W_card": W, "P_static_W": ps, "binds": binds, "feasible": True,
           "tf_dfs": tf, "tok_s_card_dfs": tok * tf, "tf_vf": 1.0, "tok_s_card_vf": tok, "s_vf": 1.0}
    if binds and _whatif_table().pJ_clk_mac_cycle is not None:
        # 0.228: with a per-cycle term the clock-only point is no longer the duty-cycle ratio (clocked power falls with
        # f while a memory-set op rate does not) — re-evaluate it too
        d = dvfs_cap(q, _whatif_table(), cap_W, "dfs")
        if not d.get("error"):
            out.update(tf_dfs=d["throughput_factor"], tok_s_card_dfs=tok * d["throughput_factor"], s_dfs=d["s"])
    if binds:
        d = dvfs_cap(q, _whatif_table(), cap_W, "rx5700xt_n7")
        if not d.get("error"):
            out.update(tf_vf=d["throughput_factor"], tok_s_card_vf=tok * d["throughput_factor"], s_vf=d["s"],
                       bound_vf=d["bound"])
    return out


def _lever_cost(q) -> dict:
    """Die area (budget.full_area: SRAM incl. the state buffer + DRAM PHY pins) and energy per output token / average
    card power (energy_report, dynamic only: no idle_W).  SRAM leakage is not modelled (static power is idle_W, a
    product figure) — reported in cost_basis.note."""
    from .core.budget import DENSITY_PRESETS, Budget, full_area
    b = Budget(**{k: v for k, (v, _, _) in DENSITY_PRESETS[WHATIF_DENSITY]["fields"].items() if k in Budget.__dataclass_fields__})
    a = full_area(q.scenario, b)
    e = energy_report(q, _whatif_table())
    from .core.budget import shoreline_check
    return {"area_mm2": a.get("mm2"), "J_per_tok": e.get("J_per_unit"), "W_card": e.get("avg_W_per_card"),
            "leak_W": e.get("sram_leak_W_card"), "hbm_W": e.get("hbm_static_W_card"), "shoreline": shoreline_check(q.scenario, a.get("mm2")),
            # 0.214: the D2D share (two-die lever): PHY area per die, energy per token, exposed collective latency
            "d2d_phy_mm2": a["terms"].get("d2d_phy"), "d2d_J_per_tok": (e.get("J_by_action") or {}).get("d2d"),
            "dies": q.scenario.layout.cards if q.scenario.d2d_enabled else 1,
            "coll_lat_ms": q.walls()["exposed_ms"].get("collective_lat") if q.walls() else None}


SPEC_ROLLBACK_BATCHES = (1, 32)     # 0.223: what-if comparison batches (user-requested design points)


def spec_rollback_compare(scn: Scenario) -> list:
    """0.223: TPOT / tok/s per card / DRAM need of both spec rollback modes (snapshot, replay) at batch 1 and 32."""
    from dataclasses import replace as rp
    out = []
    for b in SPEC_ROLLBACK_BATCHES:
        for mode in ("snapshot", "replay"):
            try:
                q = evaluate(rp(scn, serving=rp(scn.serving, batch=b, spec_rollback=mode)))
            except ValueError as e:
                out.append({"batch": b, "mode": mode, "error": str(e)})
                continue
            out.append({"batch": b, "mode": mode, "tpot_ms": q.tpot * 1e3, "tok_s_card": _tok_s_card(q),
                        "dram_need_GiB": max(st.mem.dram_need for st in q.stages) / 2**30, "fits": q.fits})
    return out


SPEC_TREE_BATCHES = (1, 8, 32)     # 0.225: tree-vs-chain what-if batches
SPEC_TREE_TOPK = 4                  # 0.225: tree offered when the scenario is a chain (SGLang auto topk for Llama 「来源」)


def spec_tree_alt(sv) -> dict:
    """0.225: the other drafting shape — a chain becomes a topk-4 tree with auto draft tokens (1 + k·topk, ≤ 64);
    a tree becomes the chain of the same depth.  0.226: the tree keeps the scenario's acceptance model (default
    chain, the headline); the chain resets it to the default."""
    if sv.spec_topk > 1:
        return {"spec_topk": 1, "spec_draft_tokens": 0, "spec_tree_accept_model": "chain"}
    return {"spec_topk": SPEC_TREE_TOPK, "spec_draft_tokens": 0}


TREE_MODELS = ("chain", "rank", "indep")


def spec_tree_compare(scn: Scenario) -> dict:
    """0.225: chain vs tree (same depth spec_k) per batch.  0.226: the headline is the scenario's acceptance model
    (default chain = guaranteed lower bound); every row also carries the tree under chain / rank / indep — the band
    [chain, indep] — and the crossover per model (batch intervals where the winner flips; rows carry
    tree_wins_<model>).  The published EAGLE-1 matched pairs: spec_refs.tree_check; nothing is fitted."""
    from dataclasses import replace as rp
    sv0 = scn.serving
    tree = {"spec_topk": sv0.spec_topk, "spec_draft_tokens": sv0.spec_draft_tokens} if sv0.spec_topk > 1 \
        else spec_tree_alt(sv0)
    tree = {"spec_topk": tree["spec_topk"], "spec_draft_tokens": tree["spec_draft_tokens"]}
    rows = []
    for b in SPEC_TREE_BATCHES:
        x = {"batch": b}
        variants = [("chain", {"spec_topk": 1, "spec_draft_tokens": 0, "spec_tree_accept_model": "chain"})]
        variants += [("tree_" + mm, {**tree, "spec_tree_accept_model": mm}) for mm in TREE_MODELS]
        for name, kw in variants:
            try:
                q = evaluate(rp(scn, serving=rp(sv0, batch=b, **kw)))
            except ValueError as e:
                x[name] = {"error": str(e)}
                continue
            x[name] = {"tpot_ms": q.tpot * 1e3, "tok_s_card": _tok_s_card(q), "tau": q.tokens_per_step,
                       "dram_need_GiB": max(st.mem.dram_need for st in q.stages) / 2**30, "fits": q.fits}
        rows.append(x)

    def cross(name):
        """[[b_prev, b], …] where the winner (tree faster than chain or not) flips between listed batches."""
        prev, out = None, []
        for x in rows:
            c, t = x["chain"], x[name]
            if "error" in c or "error" in t:
                continue
            win = t["tpot_ms"] < c["tpot_ms"]
            x["tree_wins_" + name[5:]] = win
            if prev is not None and win != prev[1]:
                out.append([prev[0], x["batch"]])
            prev = (x["batch"], win)
        return out
    from .core.spec_refs import tree_check
    chk = {mm: tree_check(mm) for mm in TREE_MODELS}
    out = {"k": sv0.spec_k, "topk": tree["spec_topk"], "headline": sv0.spec_tree_accept_model,
           "draft_tokens": tree_tokens(sv0.spec_k, tree["spec_topk"], tree["spec_draft_tokens"]), "rows": rows,
           "check": {**{mm: [c["err_min"], c["err_max"]] for mm, c in chk.items()},
                     "gain_published": [min(chk["chain"]["gain_published"]), max(chk["chain"]["gain_published"])],
                     "src": chk["chain"]["rows"][0]["src"]}}
    for mm in TREE_MODELS:
        out["crossover_" + mm] = cross("tree_" + mm)
    out["crossover"] = out["crossover_" + sv0.spec_tree_accept_model]
    return out


def balance_whatif(scn: Scenario, r: Result, power_cap_W: float | None = None) -> dict | None:
    """0.208: the machine-balance levers, one variable each, on the sensitivity scan's case loop: fp8 KV, bf16
    recurrent state, half the context, 2× SRAM, 2× DRAM bandwidth (twice the channels / stacks at the SAME capacity,
    Scenario.mem_cap_GB).  Each row: critical batch, batch limit (KV capacity / TPOT SLO), saturation intensity and
    TPOT at the configured batch — so the designer sees which lever reaches the balance point.  Decode LLM / VLM
    only (the critical batch is a decode notion)."""
    from dataclasses import replace as rp
    if scn.serving.phase != "decode" or r.model.domain not in ("llm", "vlm"):
        return None
    b0 = r.balance()
    if b0.get("reachable") is None:
        return None
    m, ch = r.model, scn.chip
    fo = dict(scn.formats_override)
    cases = []
    if "kv" in (b0["asymptote_by"] or "") and m.kv_fmt not in ("fp8", "int8") and fo.get("kv") not in ("fp8", "int8"):
        cases.append(("kv_fp8", f"KV fp8（{fo.get('kv', m.kv_fmt)} → fp8）", rp(scn, formats_override=tuple({**fo, "kv": "fp8"}.items()))))
    if "state" in (b0["asymptote_by"] or "") and fo.get("state", m.state_fmt) == "fp32":
        cases.append(("state_bf16", "状态 bf16（fp32 → bf16）", rp(scn, formats_override=tuple({**fo, "state": "bf16"}.items()))))
    if scn.serving.ctx > 1:
        cases.append(("ctx_half", f"ctx ÷2（{scn.serving.ctx} → {scn.serving.ctx // 2}）",
                      rp(scn, serving=rp(scn.serving, ctx=max(1, scn.serving.ctx // 2)))))
    if "state" in (b0["asymptote_by"] or ""):
        # 0.209: a dedicated state buffer sized to hold N sequences' recurrent state (Chip.state_sram_mib), N = the
        # critical batch with all state on chip (probe: an unbounded buffer), capped at the batch limit
        st1 = r.stages[r.heaviest_stage].mem.state_total / max(1, scn.serving.batch)
        if st1 > 0:
            big = evaluate(rp(scn, chip=rp(ch, state_sram_mib=1e9))).balance()
            n_seq = big["critical_batch"] or big["batch_limit"] or 1
            mib = math.ceil(n_seq * st1 / 2**20)
            cases.append(("state_sram", f"状态 SRAM {mib / 1024:.2f} GiB（容纳 {n_seq} 序列 × 每序列状态容量 {st1 / 2**20:.1f} MiB；未驻留时读+写 {2 * st1 / 1e6:.1f} MB/步）",
                          rp(scn, chip=rp(ch, state_sram_mib=mib))))
    cases.append(("sram_x2", f"SRAM ×2（{ch.sram_mib:g} → {2 * ch.sram_mib:g} MiB）", rp(scn, chip=rp(ch, sram_mib=2 * ch.sram_mib))))
    mx = _mem_x2(scn.mem_id)
    if mx:
        from .core.hardware import System
        cap = scn.mem_cap_GB or System(ch, scn.mem_id, scn.mem_eff).dram_bytes / 1e9
        cases.append(("dram_bw_x2", f"DRAM 带宽 ×2（{scn.mem_id} → {mx}，容量不变）", rp(scn, mem_id=mx, mem_cap_GB=cap)))
        # 0.214: the same total (compute, SRAM, 2× stacks, capacity) as two dies in one package (B200 approach): each die
        # has half the engines / SRAM / state SRAM and the base memory (half of the 2× stacks) at half the capacity,
        # TP ×2 across the dies over the D2D tier (the scenario's d2d_std / d2d_units; default UCIe-A 48G × 4) — the
        # existing multi-card + D2D model, not a new one
        if ch.engines >= 2 and scn.layout.cards * 2 <= 64:
            try:        # a scenario that cannot take TP ×2 (e.g. PD decode_cards not a multiple) has no two-die row
                half = rp(ch, engines=ch.engines // 2, sram_mib=ch.sram_mib / 2, state_sram_mib=ch.state_sram_mib / 2)
                two = rp(scn, chip=half, mem_cap_GB=cap / 2, layout=rp(scn.layout, tp=scn.layout.tp * 2),
                         package_cards=max(2, scn.package_cards), d2d_enabled=True)
            except ValueError:
                two = None
            if two is not None:
                cases.append(("two_die", f"双晶粒封装（2 × 半算力，各 {scn.mem_id}，TP×2 经 D2D {two.d2d_std}×{two.d2d_units}）", two))

    # 0.215: capacity at the same bandwidth — taller HBM stacks (same stack count, rate, width).  The catalog's
    # heights / densities carry the JEDEC / shipping status (HBM3E 12-Hi 36 GB shipping, 16-Hi JEDEC-allowed;
    # HBM4 16-Hi × 24 Gb = 48 GB sampling).  Stack height is fixed by JEDEC (720 µm HBM3/3E, 775 µm HBM4) and the
    # footprint is the same, so shoreline / PHY area do not change 「二手」; DRAM pJ/bit unchanged 「假设」;
    # refresh / standby of the extra dies is not modelled.  Only when capacity binds the batch.
    if b0.get("limited_by") == "kv_capacity" and not scn.mem_cap_GB:       # no capacity override (0 / None)
        try:
            spec = mem_catalog.parse_mem_id(scn.mem_id)
        except ValueError:
            spec = None
        if spec is not None and spec.kind == "HBM":
            cat = mem_catalog.HBM_CATALOG[spec.mem_type]
            opts = sorted({(h * d, h, d) for h in cat["heights"] for d in cat["densities"]
                           if h * d > spec.hbm_height * spec.hbm_die_Gb and h >= spec.hbm_height})
            for _, hh, dd in opts[-2:] if len(opts) > 2 else opts:
                try:
                    hs = mem_catalog.make_spec(spec.mem_type, count=spec.n_units, height=hh, die_Gb=dd, rate=spec.rate_MTps)
                except ValueError:
                    continue
                cases.append((f"hbm_{hh}h{dd}g", f"HBM 堆栈加高（{spec.hbm_height}H×{spec.hbm_die_Gb}Gb → {hh}H×{dd}Gb，"
                              f"{spec.capacity_GB:g} → {hs.capacity_GB:g} GB，带宽不变 · {hs.tag_zh}）", rp(scn, mem_id=hs.id)))

    # 0.223: spec-verify rollback strategy on hybrid / linear layers — snapshot (SGLang default) ↔ ReplaySSM (opt-in);
    # the row also carries both modes at batch 1 and 32 (spec_rollback_compare)
    sv0 = scn.serving
    spec_on = sv0.spec_k > 0 and (m.mtp_layers or sv0.draft_model)
    has_lin = any(L.core is not None and L.core.kind == "linear" and L.core.lin not in ("hyena", "rwkv7") for L in m.layers)
    roll_key = None
    if spec_on and has_lin:
        alt = "snapshot" if sv0.spec_rollback == "replay" else "replay"
        roll_key = f"spec_rollback_{alt}"
        cases.append((roll_key, f"投机回滚 {sv0.spec_rollback} → {alt}（{'ReplaySSM 环形缓冲重放，无 q× 中间状态' if alt == 'replay' else '逐草稿中间状态快照'} · SGLang 「来源」）",
                       rp(scn, serving=rp(sv0, spec_rollback=alt))))

    # 0.225: drafting shape — chain ↔ tree (same depth); the row carries chain / tree (both acceptance bounds) per
    # batch and the crossover (spec_tree_compare)
    tree_key = None
    if spec_on and sv0.spec_k and not (m.draft == "dspark" and not sv0.draft_model):
        alt_t = spec_tree_alt(sv0)
        tree_key = "spec_tree_chain" if alt_t["spec_topk"] == 1 else "spec_tree_tree"
        lab = ("树 → 线性链（同深度 k）" if alt_t["spec_topk"] == 1
               else f"线性链 → 树（topk {alt_t['spec_topk']}，N = {tree_tokens(sv0.spec_k, alt_t['spec_topk'], 0)}；"
                    f"主值接受率 {sv0.spec_tree_accept_model}，区间 [chain, indep] 见对比）")
        cases.append((tree_key, f"投机草稿形状 {lab} · EAGLE / SGLang 「来源」", rp(scn, serving=rp(sv0, **alt_t))))

    def pick(q, b):
        return {"critical_batch": b["critical_batch"], "batch_limit": b["batch_limit"], "limited_by": b["limited_by"],
                "reachable": b["reachable"], "ai_asymptote": b["ai_asymptote"], "asymptote_by": b["asymptote_by"],
                "chip_flop_per_byte": b["chip_flop_per_byte"], "tpot_ms": q.tpot * 1e3}
    base = pick(r, b0)
    c0 = _lever_cost(r)
    base.update(c0)
    # 0.218: the base row is priced at the configured batch — stated, like every lever row's cost_batch
    base.update(cost_batch=scn.serving.batch, cost_basis="configured", tok_s_card_at_cost_batch=_tok_s_card(r))
    if power_cap_W:     # 0.227: does the user's cap bind at the base point
        base["power_cap"] = lever_power_cap(r, c0, power_cap_W)

    def cost(x, q):
        # 0.211: compare at the lever's balance point — its critical batch when reachable, else the configured
        # batch — with the base chip at that same batch (a lever's value shows at the batch it unlocks)
        # 0.217: an unreachable lever was costed at the configured batch (default 1) — J/token at batch 1 (0.48 J for
        # qwen3-8b) next to a batch-limit throughput read as an energy error.  Now: its batch limit (the operating
        # point it unlocks); the base at the same batch, capped at the base's own limit (it cannot hold more KV)
        if x["reachable"] and x["critical_batch"]:
            bb, basis = x["critical_batch"], "critical_batch"
        elif x.get("batch_limit"):
            bb, basis = int(x["batch_limit"]), "batch_limit"
        else:
            bb, basis = scn.serving.batch, "configured"
        if x.get("batch_limit") and bb > int(x["batch_limit"]):     # 0.218 invariant: never price past the limit
            bb, basis = int(x["batch_limit"]), "batch_limit"
        b_base = min(bb, int(base["batch_limit"])) if basis == "batch_limit" and base.get("batch_limit") else bb
        x["cost_batch"], x["cost_basis"], x["base_cost_batch"] = bb, basis, b_base
        try:
            qb = q if bb == q.scenario.serving.batch else evaluate(q.scenario.replace("serving.batch", bb))
            rb = r if b_base == scn.serving.batch else evaluate(scn.replace("serving.batch", b_base))
        except ValueError:
            return
        c, cb = _lever_cost(qb), _lever_cost(rb)
        x.update(c)
        x["tpot_ms_at_cost_batch"] = qb.tpot * 1e3
        x["tok_s_card_at_cost_batch"] = _tok_s_card(qb)
        x["base_J_per_tok_at_cost_batch"] = cb["J_per_tok"]
        x["d_area_mm2"] = None if c["area_mm2"] is None or c0["area_mm2"] is None else c["area_mm2"] - c0["area_mm2"]
        x["d_J_per_tok"] = None if c["J_per_tok"] is None or cb["J_per_tok"] is None else c["J_per_tok"] - cb["J_per_tok"]
        x["over_reticle"] = bool(c["area_mm2"] and c["area_mm2"] > RETICLE_MM2)
        x["d_leak_W"] = None if c["leak_W"] is None and c0["leak_W"] is None else (c["leak_W"] or 0.0) - (c0["leak_W"] or 0.0)
        # 0.216: HBM standby + refresh (static, per stack × die), vs the base
        x["d_hbm_W"] = None if c["hbm_W"] is None and c0["hbm_W"] is None else (c["hbm_W"] or 0.0) - (c0["hbm_W"] or 0.0)
        if power_cap_W:     # 0.227: the lever under the cap — vs the base under the same cap at the same batch
            pc, pb = lever_power_cap(qb, c, power_cap_W), lever_power_cap(rb, cb, power_cap_W)
            if not pc.get("error") and not pb.get("error"):
                pc["base_tok_s_card_dfs"], pc["base_tok_s_card_vf"] = pb["tok_s_card_dfs"], pb["tok_s_card_vf"]
                pc["base_binds"] = pb["binds"]
                pc["gain_dfs"] = pc["tok_s_card_dfs"] / pb["tok_s_card_dfs"] if pb["tok_s_card_dfs"] else None
                pc["gain_vf"] = pc["tok_s_card_vf"] / pb["tok_s_card_vf"] if pb["tok_s_card_vf"] else None
            x["power_cap"] = pc

    def row(q):
        x = pick(q, q.balance())
        cost(x, q)
        g = lambda k: (x[k] or 0) > (base[k] or 0) * (1 + WHATIF_GAIN)
        x["tpot_ratio"] = x["tpot_ms"] / base["tpot_ms"] if base["tpot_ms"] else None
        x["works"] = bool((x["reachable"] and not base["reachable"]) or g("ai_asymptote") or g("batch_limit")
                          or (x["tpot_ratio"] is not None and x["tpot_ratio"] < 1 - WHATIF_GAIN))
        return x
    rows = _eval_cases(cases, row)
    if roll_key:
        for x in rows:
            if x["key"] == roll_key and not x.get("error"):
                x["spec_rollback_compare"] = spec_rollback_compare(scn)
    if tree_key:
        for x in rows:
            if x["key"] == tree_key and not x.get("error"):
                x["spec_tree_compare"] = spec_tree_compare(scn)
    capd = {}
    if power_cap_W:
        capd = {"power_cap_W": power_cap_W,
                "works_capped": [x["key"] for x in rows if (x.get("power_cap") or {}).get("gain_dfs") is not None
                                 and x["power_cap"]["gain_dfs"] > 1 + WHATIF_GAIN],
                "binds": [x["key"] for x in rows if (x.get("power_cap") or {}).get("binds")],
                "cap_basis": ("封顶判定用 what-if 能耗基准（tpuv4i_7nm 预设）在各杠杆计价 batch 的 W/卡；主值 dfs = 仅降频"
                              "（电压不变，≡ 0.189 占空比吞吐，偏保守）；vf = RX 5700 XT（N7）V–f 曲线「二手」；静态功耗不随电压降（偏保守）")}
    return {"base": base, "rows": rows, "gain": WHATIF_GAIN, "works": [x["key"] for x in rows if x.get("works")], **capd,
            "cost_basis": {"density_preset": WHATIF_DENSITY, "energy_preset": WHATIF_ENERGY, "reticle_mm2": RETICLE_MM2,
                           "note": ("面积 = SRAM（含专用状态 SRAM，同宏密度）+ DRAM PHY 引脚（budget.full_area，n5_derived 密度 「推导」）；"
                                    "能耗 = 动态 J/token + SRAM 泄漏 + DRAM 待机/刷新（tpuv4i_7nm 预设，不含 idle_W），在各杠杆自己的临界 batch 与基线同 batch 比较；"
                                    "HBM 岸线按面积代理的方形晶粒 · area = SRAM incl. the state buffer + DRAM PHY pins; dynamic J/token at the "
                                    "lever's critical batch vs the base at the same batch; SRAM leakage 230.7 µW/MiB (3 nm, 25 °C, lower bound) "
                                    "in W/卡 and Δ泄漏; HBM standby 48 mW/die + refresh 1.95 mW/Gb (DRAMsim3 HBM2 IDD proxy, 二手/推导), LPDDR IDD2N 32.2 mW per 8 Gb channel + refresh 0.765 mW/Gb (Micron Y52Q, 来源/推导) in Δ DRAM; HBM shoreline 11 mm/stack on a square die of the area proxy, 2 sides typical")}}


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


def _capped_point(e: dict, pkg_scope: bool) -> dict | None:
    """0.323: the clock-capped operating point of an eval under a binding cap — energy.power_cap.dvfs.dfs, the same
    dvfs_cap call (curve dfs, CAP_ITERS) the energy DSE makes for a row; static at the capped clock in package scope,
    the nominal static in card scope (energy_dse._run).  None when no cap binds."""
    from .core.powercap import static_W
    pc = e.get("power_cap") or {}
    d = (pc.get("dvfs") or {}).get("dfs") or {}
    if pc.get("error") or not pc.get("capped") or d.get("error") or not d.get("feasible", True) or "W_card" not in d:
        return None
    return {"throughput_factor": d["throughput_factor"], "W_card": d["W_card"], "pkg_W": d.get("pkg_W", d["W_card"]),
            "static_W": d.get("static_W", static_W(e)) if pkg_scope else static_W(e), "J_per_unit": d["J_per_unit"],
            "s": d.get("s"), "curve": "dfs"}


def api_eval(body: dict) -> dict:
    from .core.speech import speech_kpis
    _check_keys(body, "eval")
    scn = scenario_from_body(body)
    bud = _budget(body)
    bb = None
    if body.get("best_batch"):
        bb = best_batch(scn)
        if bb.batch:
            scn = scn.replace("serving.batch", bb.batch)
    table = _energy_table(body)
    # 0.323: the user's die (budget die_mm2 / densities) decides the auto memory-attach edges, as in the energy DSE
    # (hbm_mem_edges on the row's die) — the Scenario resolves them on the preset's published area (H100-like 814 mm²)
    # while the shoreline / thermal / cost checks below use the budget die, so one eval priced NoC energy on one die
    # and judged buildability on another
    su = None
    if bud is not None and (bud.die_mm2 or bud.mm2_per_kmac is not None):     # 0.312: the user's die area / densities
        from .core.budget import hbm_mem_edges, shoreline_report
        su = shoreline_report(scn, die_mm2=bud.die_mm2, b=None if bud.die_mm2 else bud)
        ch = scn.chip
        if su.get("die_mm2") and ch.noc_mem_edges == 0 and ch.noc_mem_attach != "none":
            me = hbm_mem_edges(scn, su["die_mm2"])
            if me != ch.mem_edges:
                import dataclasses as _dc
                scn = scn.replace("chip", _dc.replace(ch, noc_mem_edges=me))
                su = shoreline_report(scn, die_mm2=bud.die_mm2, b=None if bud.die_mm2 else bud)
    try:
        r = evaluate(scn)
    except ValueError as e:
        raise ApiError(str(e)) from None
    out = result_dict(r, with_goodput=bool(body.get("goodput", True)))
    if bb is not None and not bb.batch:     # 0.61.3: say so instead of silently evaluating the entered batch
        out["summary"]["warnings"].append(f"最佳 batch：没有既放得下又满足 SLO 的 batch，按输入的 batch {scn.serving.batch} 评估")
    pc = body.get("power_cap", False)
    if not isinstance(pc, bool):
        raise ApiError("power_cap must be true or false")
    sh = out["shoreline"]
    if su is not None and su.get("die_mm2"):
        sh = out["shoreline"] = {**su, "die_src": "budget"}
    if sh.get("feasible") is False:
        out["summary"]["warnings"].append("不可建（岸线）：" + "；".join(sh["reasons"]))
    if getattr(scn.chip, "hbm_base", "std") == "custom":       # 0.324: custom HBM base die — what it changed
        from .core import hbm_base as _hb
        from .core.budget import auto_area_budget as _aab
        out["hbm_base"] = _hb.report(scn, bud if bud is not None else _aab())
        out["summary"]["warnings"] += [n for n in out["hbm_base"]["notes"] if n.startswith("hbm_base=custom")]
    # 0.316: the TDP cap compares the package power (thermal.cap_power_fn → package_power, the function the thermal
    # check below uses) per card over the shoreline die; power_scope "card" = 0.315
    from .core.thermal import POWER_SCOPES, cap_power_fn
    psc = body.get("power_scope", "package")
    if psc not in POWER_SCOPES:
        raise ApiError(f"power_scope must be one of {list(POWER_SCOPES)}")
    pkf = cap_power_fn(sh.get("die_mm2"), 1, psc) if pc else None
    # 0.323: the cap is budget.power_W_card, else energy.tdp_W — the energy DSE's rule (eval read tdp_W only)
    cap_W = (bud.power_W_card if bud is not None and bud.power_W_card else None) or table.tdp_W
    out["energy"] = energy_report(r, table, power_cap=pc, pk=pkf, cap_W=cap_W)      # 0.47.1: action counts always; J only for user-supplied entries
    # 0.323: the operating point under a binding cap = the DVFS clock-capped point (curve dfs), the energy DSE's
    # default (power_mode dvfs, curve dfs); thermal and TCO are judged / priced there.  The duty-cycle headline
    # (energy.power_cap.throughput_factor, 0.189) stays as the labelled conservative view.
    if isinstance(out["energy"].get("power_cap"), dict):     # 0.323: where the cap came from (the DSE's power_cap_src)
        out["energy"]["power_cap"]["cap_src"] = ("budget.power_W_card" if bud is not None and bud.power_W_card
                                                 else "energy.tdp_W" if table.tdp_W else None)
    opc = _capped_point(out["energy"], pkf is not None)
    # 0.313: thermal / power buildability beside the shoreline — reported whenever the energy table gives a power;
    # judged only with body.cooling (a thermal.COOLING preset)
    from .core.powercap import static_W as _static_W
    from .core.thermal import COOLING, dram_dev_W, thermal_report
    cool = body.get("cooling")
    if cool is not None and cool not in COOLING:
        raise ApiError(f"cooling must be one of {sorted(COOLING)}")
    ew = out["energy"].get("avg_W_per_card")
    pk = scn.package_cards if scn.d2d_enabled else 1
    ddw, stw = dram_dev_W(out["energy"]), _static_W(out["energy"])
    if opc is not None:                     # 0.323: at the capped operating point (as the DSE row)
        ddw = ddw * opc["W_card"] / ew if ew else ddw
        ew, stw = opc["W_card"], opc["static_W"]
    th = thermal_report(ew * pk if ew is not None else None, sh.get("die_mm2"), pk,
                        ddw * pk, cool, stw * pk,     # 0.315: complete power
                        out["energy"]["idle_W_used"] * pk if out["energy"].get("idle_W_used") is not None else None)
    if opc is not None:
        th["operating_point"] = "capped"
    out["thermal"] = th
    if th.get("feasible") is False:
        out["summary"]["warnings"].append("不可建（热 / 功耗）：" + "；".join(th["reasons"]))
    if body.get("tco") is not None:          # 0.317: opt-in $/M unit (core/tco.py)
        from .core.tco import eval_tco
        try:
            out["tco"] = eval_tco(body["tco"], scn=scn, r=r, energy=out["energy"], die_mm2=sh.get("die_mm2"),
                                  unit=out["energy"].get("unit") or "token", op=opc)
        except ValueError as e:
            raise ApiError(str(e)) from None
        if out["tco"].get("usd_per_Munit") is not None and (th.get("feasible") is False or sh.get("feasible") is False):
            out["tco"]["buildable"] = False
            out["tco"]["notes"].append("该设计不可建（岸线 / 热）：$/M 仅供参考，不能作为成本结论")
    if body.get("sensitivity", True):
        out["sensitivity"] = sensitivity_report(scn, r, cap_W if pc and cap_W else None)   # 0.227 (0.323: same cap)
    if bud is not None:
        out["budget"] = budget_report(r, bud, out["energy"])
    if scn.fabric.enabled:  # 0.59: per-collective algorithm / topology breakdown
        out["fabric"] = fabric_report(scn, r)
    # 0.270 speech serving convention (core/speech.py): end-to-end per audio second — ASR when audio_s > 0 on a model
    # with an audio encoder; TTS for a TTS head (audio_out_s seconds out, default 10 「假设」); 0.271 Omni (Talker +
    # token2wav speech output, core/omni.py) always
    aos = body.get("audio_out_s", 10.0)
    if isinstance(aos, bool) or not isinstance(aos, (int, float)) or not 0 < aos <= 3600:
        raise ApiError("audio_out_s must be a number in (0, 3600]")
    if r.tts is not None or getattr(r.model, "omni", None) is not None or getattr(r.model, "cosy", None) is not None or getattr(r.model, "echo", None) is not None or getattr(r.model, "kimi", None) is not None or getattr(r.model, "moshi", None) is not None or (scn.serving.audio_s and getattr(getattr(r.model, "vision", None), "audio", None) is not None):
        try:
            out["speech"] = speech_kpis(scn, float(aos))
        except ValueError as e:
            out["speech"] = {"error": str(e)}
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
    if path in ("serving.images", "serving.image_w", "serving.image_h") and (m.vision is None or m.vision.family == "audio"):
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
    if path == "serving.spec_k" and not (m.mtp_layers or m.draft or sc.serving.draft_model):
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
    cap = body.get("power_cap_W")      # 0.227: user's per-card power cap (no default — the TDP is a product decision)
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap < 1e5):
        raise ApiError("power_cap_W must be a number in (0, 100000) W")
    try:
        r = evaluate(scn)
    except ValueError as e:
        raise ApiError(str(e)) from None
    return {"sensitivity": sensitivity_report(scn, r, float(cap) if cap is not None else None), "hash": scn.hash()}


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



def api_insight_slc(body: dict) -> dict:
    """0.262: J/token-optimal SLC size under energy.sram_temp_C (hot-die crossover)."""
    from .core.insight import slc_energy_report
    _check_keys(body, "insight_slc")
    scn = scenario_from_body(body)
    # default = the scenario's model (a GiB sweep × N models is expensive; insight defaults stay unused)
    models = _insight_models(body) if body.get("models") is not None else [scn.model]
    gibs = body.get("gibs")
    if gibs is not None and (not isinstance(gibs, list) or not all(isinstance(x, (int, float)) for x in gibs)):
        raise ApiError("gibs must be a list of numbers")
    temps = body.get("temps")
    if temps is not None and (not isinstance(temps, list) or not all(isinstance(x, (int, float)) for x in temps)):
        raise ApiError("temps must be a list of numbers")
    pol = body.get("policy", "dip")
    if not isinstance(pol, str):
        raise ApiError("policy must be a string")
    try:
        return {**slc_energy_report(scn, _energy_table(body), models, policy=pol,
                                    gibs=tuple(gibs) if gibs is not None else None,
                                    temps=tuple(temps) if temps is not None else None), "hash": scn.hash()}
    except ValueError as e:
        raise ApiError(str(e)) from None


def _noc_link(body: dict):
    """0.303: optional NoC link bandwidth (B / cycle / link / direction) applied to every insight / DSE point."""
    v = body.get("noc_link_Bpc")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1e6):
        raise ApiError("noc_link_Bpc must be a number in [0, 1e6) bytes / cycle / link")
    return None if v is None else float(v)


def _noc_routing(body: dict):
    """0.304: optional NoC contention model ("bisection" = 0.303, "xy" = busiest link under XY routing)."""
    from .core.hardware import NOC_ROUTINGS
    v = body.get("noc_routing")
    if v is not None and v not in NOC_ROUTINGS:
        raise ApiError(f"noc_routing must be one of {NOC_ROUTINGS}")
    return v


def api_insight_noc(body: dict) -> dict:
    """0.299: NoC / on-chip data-movement energy vs engine count × topology (core/noc_insight.py)."""
    from .core.hardware import NOC_TOPOLOGIES
    from .core.noc_insight import noc_design_report
    _check_keys(body, "insight_noc")
    scn = scenario_from_body(body)
    topo = body.get("topologies", list(NOC_TOPOLOGIES))
    if not isinstance(topo, list) or not topo or not all(isinstance(x, str) and x in NOC_TOPOLOGIES for x in topo):
        raise ApiError(f"topologies must be a non-empty list of {NOC_TOPOLOGIES}")
    ph = body.get("phases", ["decode", "prefill"])
    if not isinstance(ph, list) or not ph or not all(x in ("decode", "prefill") for x in ph):
        raise ApiError("phases must be a non-empty list of 'decode' / 'prefill'")
    pr = body.get("prompt")
    if pr is not None and (isinstance(pr, bool) or not isinstance(pr, int) or not 1 <= pr <= 1 << 20):
        raise ApiError("prompt must be an integer in [1, 1048576]")
    nl, rt = _noc_link(body), _noc_routing(body)
    try:
        return {**noc_design_report(scn, _energy_table(body), tuple(topo), tuple(ph), pr, nl, rt), "hash": scn.hash()}
    except ValueError as e:
        raise ApiError(str(e)) from None


def api_insight_dse(body: dict) -> dict:
    """0.300: energy-aware design search — throughput vs J/token (perf/W) Pareto front under the budget's die area /
    card power (core/energy_dse.py)."""
    from .core.energy_dse import energy_design_search

    _check_keys(body, "insight_dse")
    scn = scenario_from_body(body)

    def nums(k, lo, hi, n=6, integer=False):
        v = body.get(k)
        if v is None:
            return None
        if (not isinstance(v, list) or not 1 <= len(v) <= n or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and lo <= x <= hi for x in v) or
                (integer and not all(float(x).is_integer() for x in v))):
            raise ApiError(f"{k} must be a list of 1–{n} {'integers' if integer else 'numbers'} in [{lo}, {hi}]")
        return tuple(int(x) if integer else float(x) for x in v)
    kw = {}
    for k, lo, hi, it in (("mac_scales", 0.0625, 16, False), ("sram_scales", 0.0625, 16, False), ("sides", 4, 1024, True)):
        v = nums(k, lo, hi, integer=it)
        if v is not None:
            kw[k] = v
    for k, lo, hi, it in (("mem_count_scales", 0.0625, 16, False), ("mem_caps", 1, 1024, False),
                          ("slc_mibs", 0, 1 << 20, False),          # 0.301: memory-side axes
                          ("f_scales", 0.15, 1, False)):            # 0.302: DVFS operating point (f / f_max)
        v = nums(k, lo, hi, n=8, integer=it)
        if v is not None:
            kw[k] = v
    for k in ("mems", "mem_types"):
        v = body.get(k)
        if v is not None:
            if not isinstance(v, list) or not 1 <= len(v) <= 12 or not all(isinstance(x, str) and 0 < len(x) <= 64 for x in v):
                raise ApiError(f"{k} must be a list of 1–12 strings")
            kw[k] = tuple(v)
    for k in ("slo_tpot_ms", "slo_ttft_ms", "max_mem_usd"):
        v = body.get(k)
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1e12:
                raise ApiError(f"{k} must be a positive number")
            kw[k] = float(v)
    if (nl := _noc_link(body)) is not None:      # 0.303: NoC congestion on every grid point
        kw["noc_link_Bpc"] = nl
    if (rt := _noc_routing(body)) is not None:   # 0.304: per-link XY contention
        kw["noc_routing"] = rt
    if (v := body.get("noc_links")) is not None:  # 0.306: NoC link width as a search axis
        if (not isinstance(v, list) or not 1 <= len(v) <= 6 or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < 1e6 for x in v)):
            raise ApiError("noc_links must be a list of 1–6 link widths in (0, 1e6) bytes / cycle")
        if "noc_link_Bpc" in kw:
            raise ApiError("noc_links (axis) and noc_link_Bpc (single value) are exclusive")
        kw["noc_links"] = tuple(float(x) for x in v)
    mo = body.get("mix_out_tokens")
    if mo is not None:
        if isinstance(mo, bool) or not isinstance(mo, int) or not 1 <= mo <= 1 << 20:
            raise ApiError("mix_out_tokens must be an integer in [1, 1048576]")
        kw["mix_out_tokens"] = mo
    # 0.308: chiplet partitioning — dies per package × D2D grade, split axis, reticle / interposer / yield
    if (v := nums("dies", 1, 16, n=5, integer=True)) is not None:
        kw["dies"] = v
    if (v := body.get("d2d_stds")) is not None:
        if not isinstance(v, list) or not 1 <= len(v) <= 6 or not all(isinstance(x, str) and x in D2D_STANDARDS for x in v):
            raise ApiError(f"d2d_stds must be a list of 1–6 of {sorted(D2D_STANDARDS)}")
        kw["d2d_stds"] = tuple(v)
    if (v := body.get("d2d_units")) is not None:
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 64:
            raise ApiError("d2d_units must be an integer in [1, 64]")
        kw["d2d_units"] = v
    if (v := body.get("die_split")) is not None:
        if v not in ("tp", "pp", "ep"):
            raise ApiError("die_split must be 'tp', 'pp' or 'ep'")
        kw["die_split"] = v
    if (v := nums("clusters", 1, 64, n=5, integer=True)) is not None:      # 0.310: NoC-cluster axis
        kw["clusters"] = v
    if (v := body.get("cost")) is not None:                                # 0.310: cost per good package
        from .core.energy_dse import _cost_params
        try:
            _cost_params(v)
        except ValueError as e:
            raise ApiError(str(e)) from None
        kw["cost"] = v
    if (v := body.get("cooling")) is not None:                             # 0.313: thermal preset
        from .core.thermal import COOLING
        if v not in COOLING:
            raise ApiError(f"cooling must be one of {sorted(COOLING)}")
        kw["cooling"] = v
    if (v := body.get("power_scope")) is not None:                         # 0.316: cap scope (card = 0.315)
        from .core.thermal import POWER_SCOPES
        if v not in POWER_SCOPES:
            raise ApiError(f"power_scope must be one of {list(POWER_SCOPES)}")
        kw["power_scope"] = v
    if (v := body.get("tco")) is not None:                                 # 0.317: $/M unit axis (needs cost)
        from .core.tco import tco_params
        try:
            tco_params(v)
        except ValueError as e:
            raise ApiError(str(e)) from None
        if kw.get("cost") is None:
            raise ApiError("tco needs cost (package cost per good unit, n5_cowos_s / n5_cowos_l)")
        kw["tco"] = v
    if (v := body.get("tco_sens_row")) is not None:                        # 0.318: sensitivity of a chosen row
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ApiError("tco_sens_row must be a row index ≥ 0")
        if kw.get("tco") is None:
            raise ApiError("tco_sens_row needs tco")
        kw["tco_sens_row"] = v
    for k in ("mono_clusters", "mem_edges_shoreline", "include_unbuildable", "shore_strict"):   # 0.309 / 0.312 / 0.313
        if (v := body.get(k)) is not None:
            if not isinstance(v, bool):
                raise ApiError(f"{k} must be a boolean")
            kw[k] = v
    for k, lo, hi in (("reticle_mm2", 0, 1e5), ("d0_per_cm2", 1e-6, 10)):
        v = body.get(k)
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v < hi:
                raise ApiError(f"{k} must be a number in [{lo:g}, {hi:g})")
            kw[k] = float(v)
    if (v := body.get("pkg_mm2")) is not None:
        from .core.energy_dse import PKG_LIMITS
        if isinstance(v, str):
            if v not in PKG_LIMITS:
                raise ApiError(f"pkg_mm2 must be a number or one of {sorted(PKG_LIMITS)}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1e6:
            raise ApiError("pkg_mm2 must be a number in (0, 1e6) mm² or a package preset")
        kw["pkg_mm2"] = v if isinstance(v, str) else float(v)
    up = body.get("mem_usd_per_GB")
    if up is not None:
        if (not isinstance(up, dict) or len(up) > 8 or
                not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < 1e6 for v in up.values())):
            raise ApiError("mem_usd_per_GB must be an object {memory type: $/GB ≥ 0}")
        kw["mem_usd_per_GB"] = dict(up)
    for k, typ in (("topologies", list), ("power_mode", str), ("curve", str), ("phase", str)):
        if body.get(k) is not None:
            if not isinstance(body[k], typ):
                raise ApiError(f"{k} must be a {'list' if typ is list else 'string'}")
            kw[k] = tuple(body[k]) if typ is list else body[k]
    pr = body.get("prompt")
    if pr is not None and (isinstance(pr, bool) or not isinstance(pr, int) or not 1 <= pr <= 1 << 20):
        raise ApiError("prompt must be an integer in [1, 1048576]")
    tie = body.get("tie")
    if tie is not None and (isinstance(tie, bool) or not isinstance(tie, (int, float))):
        raise ApiError("tie must be a number in [0, 0.5)")
    try:
        return {**energy_design_search(scn, _energy_table(body), _budget(body), prompt=pr, tie=tie, **kw),
                "hash": scn.hash()}
    except ValueError as e:
        raise ApiError(str(e)) from None


def api_tco_sens(body: dict) -> dict:
    """0.318: TCO sensitivity of one design of an energy DSE (body = insight_dse body with tco + cost; optional
    tco_sens_row, default the cheapest $/M row): the tornado over the sourced ranges, the mono / split $/M verdicts
    with the inputs that flip them (and where), and the ranges with their sources."""
    from .core.tco import MEM_RANGES
    from .core.tco_sens import LABELS, RANGES
    if not isinstance(body, dict) or body.get("tco") is None:
        raise ApiError("tco_sens needs an insight_dse body with tco (and cost)")
    r = api_insight_dse(body)
    t = r["tco"] or {}
    cmp_ = [{k: e.get(k) for k in ("mac_x", "sram_x", "side", "topology", "mem_id", "slc_mib", "f_op", "verdict_tco",
                                   "d_tco", "best_tco_mono", "best_tco_split", "tco_flips", "tco_corners")}
            for e in ((r.get("chiplet") or {}).get("compare") or []) if e.get("verdict_tco")]
    return {"sens": t.get("sens"), "row": (t.get("sens") or {}).get("row"),
            "design": None if t.get("sens") is None else
            {k: r["rows"][t["sens"]["row"]].get(k) for k in ("dies", "mac_x", "sram_x", "mem_id", "topology", "thr", "unit")},
            "compare": cmp_, "preset": t.get("preset"), "n_ranked": t.get("n_ranked"),
            "ranges": {"inputs": {k: {"label": LABELS[k], "lo": v[0], "hi": v[1], "tag_lo": v[2], "tag_hi": v[3], "src": v[4]}
                                  for k, v in RANGES.items()},
                       "mem_usd_per_GB": {k: {"lo": v[0], "hi": v[1], "tag_lo": v[2], "tag_hi": v[3], "src": v[4]}
                                          for k, v in MEM_RANGES.items()}},
            "hash": r.get("hash")}


def api_report(body: dict) -> dict:
    """0.320: design verdict report of one design of an energy DSE (body = insight_dse body; optional report_row,
    default the cheapest $/M row, else the best perf/W row; report_format json | markdown | both): performance at SLO,
    energy, shoreline, thermal / power, yield and cost per good package, TCO with sensitivity and joint corners, and
    near-tie warnings — every number with its label and source, missing inputs listed as missing (core/report.py).
    0.321: ``report_rows`` [2 – 3 row indices] or ``report_compare`` true → side-by-side comparison (compare_report)."""
    from .core.report import compare_markdown, compare_report, design_report, to_markdown
    if not isinstance(body, dict):
        raise ApiError("body must be a JSON object")
    fmt = body.get("report_format", "both")
    if fmt not in ("json", "markdown", "both"):
        raise ApiError("report_format must be json, markdown or both")
    cmp_ = body.get("report_compare", False)
    if not isinstance(cmp_, bool):
        raise ApiError("report_compare must be true or false")
    if "report_rows" in body and "report_row" in body:
        raise ApiError("give report_row (one design) or report_rows (comparison), not both")
    b = {k: v for k, v in body.items() if k not in ("report_row", "report_rows", "report_compare", "report_format")}
    if cmp_ or "report_rows" in body:
        # 0.321: side-by-side comparison of 2 – 3 rows (report_rows; default: the report row + its cheapest rival)
        r = api_insight_dse(b)
        try:
            rep = compare_report(r, b, body.get("report_rows"))
        except ValueError as e:
            raise ApiError(str(e)) from None
        out = {"rows": rep["rows"], "hash": rep["hash"]}
        if fmt in ("json", "both"):
            out["compare"] = rep
        if fmt in ("markdown", "both"):
            out["markdown"] = compare_markdown(rep)
        return out
    r = api_insight_dse(b)
    try:
        rep = design_report(r, b, body.get("report_row"))
    except ValueError as e:
        raise ApiError(str(e)) from None
    out = {"row": rep["row"], "hash": rep["hash"]}
    if fmt in ("json", "both"):
        out["report"] = rep
    if fmt in ("markdown", "both"):
        out["markdown"] = to_markdown(rep)
    return out


def api_insight_speech(body: dict) -> dict:
    """0.274: which stage bounds real time / first audio for each catalog speech model on the scenario's chip + memory,
    and the DRAM bandwidth or pinned SLC that would unlock real time (core/speech_insight.py)."""
    from .core.speech_insight import SPEECH_MODELS, speech_bounds
    _check_keys(body, "insight_speech")
    scn = scenario_from_body(body)
    models = body.get("models", list(SPEECH_MODELS))
    if not isinstance(models, list) or not models or not all(isinstance(x, str) for x in models):
        raise ApiError("models must be a non-empty list of model ids")
    a = body.get("audio_s", 10.0)
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 < a <= 600:
        raise ApiError("audio_s must be a number in (0, 600]")
    v = body.get("verify", True)
    if not isinstance(v, bool):
        raise ApiError("verify must be true or false")
    dp = body.get("dispatch", True)          # 0.283: per-op dispatch sensitivity
    if not isinstance(dp, bool):
        raise ApiError("dispatch must be true or false")
    try:
        return {**speech_bounds(scn, models, float(a), verify=v, dispatch=dp), "hash": scn.hash()}
    except (KeyError, ValueError) as e:
        raise ApiError(str(e)) from None


def api_insight_dispatch(body: dict) -> dict:
    """0.284: per-op dispatch sensitivity of LLM decode at small batch (core/dispatch_insight.dispatch_sensitivity)."""
    from .core.dispatch_insight import BATCHES, CTX, LLM_MODELS, dispatch_sensitivity
    _check_keys(body, "insight_dispatch")
    scn = scenario_from_body(body)
    models = body.get("models", list(LLM_MODELS))
    if not isinstance(models, list) or not models or len(models) > 24 or not all(isinstance(x, str) for x in models):
        raise ApiError("models must be a list of 1–24 model ids")
    bs = body.get("batches", list(BATCHES))
    if not isinstance(bs, list) or not 0 < len(bs) <= 8 or not all(
            isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 4096 for x in bs):
        raise ApiError("batches must be a list of 1–8 integers in [1, 4096]")
    ctx = body.get("ctx", CTX)
    if isinstance(ctx, bool) or not isinstance(ctx, int) or not 1 <= ctx <= 1 << 20:
        raise ApiError("ctx must be an integer in [1, 1048576]")
    try:
        return {**dispatch_sensitivity(scn, models, tuple(bs), ctx), "hash": scn.hash()}
    except (KeyError, ValueError) as e:
        raise ApiError(str(e)) from None


def api_insight_speech_streams(body: dict) -> dict:
    """0.276: real-time streams per card vs SLC size for catalog speech models (core/speech_insight.speech_streams)."""
    from .core.speech_insight import SLC_GIB, SPEECH_MODELS, speech_streams
    _check_keys(body, "insight_speech_streams")
    scn = scenario_from_body(body)
    models = body.get("models", list(SPEECH_MODELS))
    if not isinstance(models, list) or not models or not all(isinstance(x, str) for x in models):
        raise ApiError("models must be a non-empty list of model ids")
    a = body.get("audio_s", 10.0)
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 < a <= 600:
        raise ApiError("audio_s must be a number in (0, 600]")
    g = body.get("slc_gib", list(SLC_GIB))
    if not isinstance(g, list) or not 0 < len(g) <= 12 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 64 for x in g):
        raise ApiError("slc_gib must be a list of 1–12 numbers in [0, 64]")
    try:
        return {**speech_streams(scn, models, float(a), tuple(float(x) for x in g)), "hash": scn.hash()}
    except (KeyError, ValueError) as e:
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
