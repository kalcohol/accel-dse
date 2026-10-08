"""Local interactive workbench HTTP API + static UI (v0.25; memory v0.29).

Zero hard dependency: uses stdlib ``http.server`` by default.
If ``fastapi`` + ``uvicorn`` are installed, those are preferred.

Endpoints:
  GET  /                 → static index.html
  GET  /api/health
  GET  /api/series[?domain=&product_only=]
  GET  /api/packages[?kind=]   (kind = HBM | LPDDR | LPDDR5X | HBM3E | …)
  GET  /api/memory[?resolve=<id>] → v0.29 structured memory selector catalog
                           (types → forms → widths/rates/counts/caps + provenance
                           tags); ``resolve`` maps a canonical or legacy package
                           id to its structured spec (+ legacy_note)
  GET  /api/compute[?level=]
  POST /api/eval         → MetricsCard JSON from WorkbenchConfig knobs
                           (llm / video / protein)
  POST /api/sweep        → small DSE grid along one axis → list of MetricsCards
                           (axis: chips|package|compute|parallel|series; ≤32 rows)
  POST /api/pareto       → v0.30 throughput–interactivity Pareto (batch × layouts,
                           capacity-limited) + SLO goodput (``slo_ttft_ms`` /
                           ``slo_tpot_ms`` / ``slo_latency_ms``); ``/api/pareto.csv``
                           returns the points as CSV
  GET  /api/presets      → scenario presets (package+compute+chips) + EXAMPLE econ
                           placeholders (v0.25)

Energy / cost (v0.25, ASSUMED stub): body ``tdp_w`` | ``watts_per_tops``,
``power_util``, ``cost_per_card_usd``, ``mem_addon_usd``, ``usd_per_kwh``,
``amortize_years``, ``duty_cycle`` (flat) or nested ``econ: {...}``.
Scenario preset: body ``preset: "edge-lpddr-4x64"`` (explicit package/compute/
chips keys in the same body win).

Memory (v0.29): ``package_id`` (canonical e.g. ``lpddr6_4x96_10667_16g`` /
``hbm3e_8s_12h24g_9200`` or legacy ``lpddr_4x64_8533`` / ``hbm_hbm3e_4s``) and/or
structured fields ``mem_type, mem_form, mem_width_bits, mem_rate_MTps,
mem_count, mem_cap_GB, hbm_height, hbm_die_Gb`` (structured fields win).
Optimism fixes (assumed knobs): ``c2c_latency_us`` (α per collective, default
3), ``sync_overlap`` (0 = exposed), ``attn_parallel`` (tp | dp).
Sweep ``axis=package`` + ``package_axis=type|count|rate`` iterates the
structured catalog around the base memory.

Honesty: BW / efficiency / freq remain assumed / uncalibrated.
"""

from __future__ import annotations

import json
import mimetypes
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .package_ranges import (
    DEFAULT_COMPUTE_SWEEP,
    DEFAULT_COMPUTE_SWEEP_CLUSTER,
    DEFAULT_COMPUTE_SWEEP_CORE,
    DEFAULT_PACKAGE_SWEEP_HBM,
    DEFAULT_PACKAGE_SWEEP_LPDDR,
    get_compute,
    get_package,
    list_compute,
    list_compute_primaries,
    list_packages,
)
from .econ import ECON_FIELD_NAMES, EnergyCostKnobs, load_example_econ
from .mem_catalog import catalog_dict, make_spec, parse_mem_id, sweep_specs
from .scenarios import get_preset, list_presets
from .series import list_series
from .workbench import (
    DEFAULT_CHIP_SWEEP,
    CalibrationOverrides,
    ParallelOverride,
    WorkbenchConfig,
    evaluate_workbench,
    workbench_parallel_matrix,
)

WEB_DIR = Path(__file__).resolve().parent / "web"

HONESTY_BANNER = (
    "All absolute bandwidth, efficiency, and frequency numbers are "
    "assumed / uncalibrated DSE labels — not measured silicon. "
    "MAC/TOPS peak math is derived from cores × tops_per_core @ assumed freq. "
    "MetricsCard covers llm (TTFT/TPOT), video (TTFC/frames/s), and protein "
    "(time/seq + pair memory). Video/protein multi-card: tp/pp/ep with "
    "tp*pp*ep==chips; TP act collectives (ring|tree), PP bubble (denoise "
    "pipelines poorly), EP no-op without MoE experts. "
    "Energy/cost (est_*) fields are an ASSUMED stub from user knobs "
    "(tdp_w / watts_per_tops / cost_per_card_usd) — not silicon power."
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _series_to_dict(e: Any) -> dict[str, Any]:
    shape = e.shape
    shape_name = getattr(shape, "name", "?")
    hf_id = None
    meta = e.metadata or ""
    m = re.search(r"hf:([^\s|]+)", meta)
    if m:
        hf_id = m.group(1)
    out: dict[str, Any] = {
        "id": e.id,
        "family": e.family,
        "domain": e.domain,
        "shape_name": shape_name,
        "metadata": meta,
        "hf_id": hf_id,
        "is_hf_backed": bool(e.is_hf_backed),
        "is_illustrative": bool(e.is_illustrative),
    }
    # Domain defaults for UI knobs
    if e.domain == "video":
        out["defaults"] = {
            "n_denoise": getattr(shape, "n_denoise", 50),
            "n_frames": getattr(shape, "n_frames", 16),
            "batch": getattr(shape, "batch", 1),
        }
    elif e.domain == "protein":
        out["defaults"] = {
            "seq_len": getattr(shape, "seq_len", 512),
            "pair_dim": getattr(shape, "pair_dim", 0),
            "batch": getattr(shape, "batch", 1),
        }
    elif e.domain == "llm":
        out["defaults"] = {"batch": 1}
    return out


def _package_to_dict(p: Any) -> dict[str, Any]:
    return {
        "id": p.id,
        "kind": p.kind,
        "n_channels": p.n_channels,
        "width_bits": p.width_bits,
        "data_rate_GTs": p.data_rate_GTs,
        "efficiency": p.efficiency,
        "capacity_GB": p.capacity_GB,
        "generation": p.generation,
        "n_stacks": p.n_stacks,
        "n_packages": p.n_packages,
        "package_width_bits": p.package_width_bits,
        "bus_width_bits": p.bus_width_bits,
        "note": p.note,
        "peak_GBps": p.peak_bandwidth_GBps(),
        "eff_GBps": p.effective_bandwidth_GBps(),
        # v0.29 structured view
        "payload_GBps": p.payload_bandwidth_GBps(),
        "mem_type": p.mem_spec.mem_type,
        "form": p.mem_spec.form,
        "tag": p.mem_spec.tag,
        "tag_zh": p.mem_spec.tag_zh,
        "label": p.mem_spec.short_label(),
        "spec": p.mem_spec.to_dict(),
    }


def _compute_to_dict(c: Any) -> dict[str, Any]:
    return {
        "id": c.id,
        "level": c.level,
        "peak_tops": c.peak_tops,
        "n_cores": c.n_cores,
        "tops_per_core": c.tops_per_core,
        "frequency_hz": c.frequency_hz,
        "note": c.note,
    }


def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "banner": HONESTY_BANNER,
    }


def api_series(
    *,
    domain: str | None = None,
    product_only: bool = False,
) -> dict[str, Any]:
    rows = list_series(product_only=product_only)
    if domain:
        d = domain.strip().lower()
        rows = [e for e in rows if e.domain == d]
    return {
        "count": len(rows),
        "banner": HONESTY_BANNER,
        "series": [_series_to_dict(e) for e in rows],
    }


def api_packages(*, kind: str | None = None) -> dict[str, Any]:
    k = None
    if kind and kind.strip().lower() not in ("", "all"):
        k = kind.strip().upper()
    pkgs = list_packages(kind=k)  # type: ignore[arg-type]
    return {
        "count": len(pkgs),
        "banner": HONESTY_BANNER,
        "packages": [_package_to_dict(p) for p in pkgs],
    }


def api_memory(*, resolve: str | None = None) -> dict[str, Any]:
    """v0.29 structured memory catalog (+ optional id resolution)."""
    out: dict[str, Any] = {"banner": HONESTY_BANNER, "catalog": catalog_dict()}
    if resolve:
        spec = parse_mem_id(str(resolve))
        out["resolved"] = spec.to_dict()
        out["workbench_kwargs"] = {
            k: v for k, v in spec.workbench_kwargs().items() if v is not None
        }
    return out


STRUCT_MEM_KEYS = (
    "mem_type",
    "mem_form",
    "mem_width_bits",
    "mem_rate_MTps",
    "mem_count",
    "mem_cap_GB",
    "hbm_height",
    "hbm_die_Gb",
)
_STRUCT_MEM_CAST = {
    "mem_type": str,
    "mem_form": str,
    "mem_width_bits": int,
    "mem_rate_MTps": float,
    "mem_count": int,
    "mem_cap_GB": float,
    "hbm_height": int,
    "hbm_die_Gb": int,
}


def _apply_struct_mem(body: dict[str, Any], kw: dict[str, Any]) -> None:
    """Structured memory body fields override package-derived ones (v0.29)."""
    given = {k: body[k] for k in STRUCT_MEM_KEYS if body.get(k) not in (None, "")}
    if not given:
        return
    new_t = str(given.get("mem_type") or kw.get("mem_type") or "").upper()
    old_t = str(kw.get("mem_type") or "").upper()
    if not new_t:
        raise ValueError("structured memory fields need mem_type (or a package_id)")
    if old_t and new_t != old_t:
        # type switch: drop package-derived details not explicitly given
        for k in STRUCT_MEM_KEYS:
            if k not in given:
                kw[k] = None
    if given.get("mem_form") and str(given["mem_form"]) != str(kw.get("mem_form") or ""):
        for k in ("mem_width_bits", "mem_cap_GB", "mem_rate_MTps", "mem_count"):
            if k not in given:
                kw[k] = None
    for k, v in given.items():
        kw[k] = _STRUCT_MEM_CAST[k](v)
    kw["mem_type"] = new_t
    # legacy manual geometry must not shadow the structured spec
    for k in ("n_channels", "width_bits", "data_rate_GTs", "capacity_GB", "n_packages"):
        kw[k] = None
    spec = make_spec(
        new_t,
        form=kw.get("mem_form"),
        width_bits=kw.get("mem_width_bits"),
        rate=kw.get("mem_rate_MTps"),
        count=kw.get("mem_count"),
        cap_GB=kw.get("mem_cap_GB"),
        height=kw.get("hbm_height"),
        die_Gb=kw.get("hbm_die_Gb"),
    )  # validates + snaps (raises ValueError on impossible combos)
    kw["mem_kind"] = spec.kind


def api_compute(*, level: str = "all", all_aliases: bool = False) -> dict[str, Any]:
    lvl = (level or "all").strip().lower()
    if all_aliases:
        opts = list_compute(level=lvl if lvl != "all" else "all")
    elif lvl == "all":
        opts = list_compute_primaries()
    else:
        opts = [c for c in list_compute_primaries() if c.level == lvl]
    return {
        "count": len(opts),
        "banner": HONESTY_BANNER,
        "compute": [_compute_to_dict(c) for c in opts],
    }


def _build_workbench_from_body(body: dict[str, Any]) -> WorkbenchConfig:
    """Map JSON body → WorkbenchConfig. Supports package_id / compute_id shortcuts."""
    kw: dict[str, Any] = {}

    # v0.25 scenario preset: fill package/compute/chips unless body sets them
    preset_id = body.get("preset") or body.get("scenario")
    if preset_id:
        pb = get_preset(str(preset_id)).body()
        merged = dict(pb)
        merged.update({k: v for k, v in body.items() if v is not None})
        # alias keys in body beat preset canonical keys
        if body.get("chips") is not None and body.get("chip_count") is None:
            merged["chip_count"] = body["chips"]
        if body.get("package") and not body.get("package_id"):
            merged["package_id"] = body["package"]
        if body.get("compute") and not body.get("compute_id"):
            merged["compute_id"] = body["compute"]
        # explicit manual geometry / cores in body suppress preset ids
        if any(body.get(k) is not None for k in ("n_channels", "mem_kind", "mem")) and not body.get("package_id"):
            merged.pop("package_id", None)
        if body.get("mem_type") and not body.get("package_id") and not body.get("package"):
            merged.pop("package_id", None)
        if any(body.get(k) is not None for k in ("n_cores", "cores", "sku")) and not body.get("compute_id"):
            merged.pop("compute_id", None)
        if body.get("chip_count") is not None or body.get("chips") is not None:
            if body.get("tp") is None:
                for k in ("tp", "pp", "ep"):
                    merged.pop(k, None)
        body = merged

    model_id = (
        body.get("model_id")
        or body.get("model")
        or body.get("series")
        or body.get("series_id")
        or "illustrative_27B"
    )
    kw["model_id"] = str(model_id)

    chip_count = int(body.get("chip_count", body.get("chips", 1)))
    kw["chip_count"] = chip_count

    tp = body.get("tp")
    pp = body.get("pp", 1)
    ep = body.get("ep", 1)
    if tp is not None:
        kw["parallel"] = ParallelOverride(tp=int(tp), pp=int(pp), ep=int(ep))

    # Memory: package preset OR manual geometry
    package_id = body.get("package_id") or body.get("package")
    if package_id:
        pkg = get_package(str(package_id))
        kw.update(pkg.workbench_kwargs())
    else:
        mem_kind = body.get("mem_kind") or body.get("mem") or "HBM"
        kw["mem_kind"] = str(mem_kind).upper()
        for src, dst in (
            ("n_channels", "n_channels"),
            ("n_packages", "n_packages"),
            ("n_ranks", "n_ranks"),
            ("width_bits", "width_bits"),
            ("data_rate_GTs", "data_rate_GTs"),
            ("efficiency", "efficiency"),
            ("capacity_GB", "capacity_GB"),
        ):
            if body.get(src) is not None:
                kw[dst] = body[src]
        # aliases
        if body.get("data_rate_gts") is not None and "data_rate_GTs" not in kw:
            kw["data_rate_GTs"] = body["data_rate_gts"]
        if body.get("capacity_gb") is not None and "capacity_GB" not in kw:
            kw["capacity_GB"] = body["capacity_gb"]
    _apply_struct_mem(body, kw)
    # Slider / body efficiency always wins over package default (assumed knob)
    if body.get("efficiency") is not None:
        kw["efficiency"] = float(body["efficiency"])
    if body.get("mem_efficiency") is not None:
        kw["efficiency"] = float(body["mem_efficiency"])

    # Compute: compute_id preset OR manual cores×tops
    compute_id = body.get("compute_id") or body.get("compute")
    if compute_id:
        opt = get_compute(str(compute_id))
        kw.update(opt.workbench_kwargs())
    else:
        if body.get("n_cores") is not None:
            kw["n_cores"] = int(body["n_cores"])
        elif body.get("cores") is not None:
            kw["n_cores"] = int(body["cores"])
        if body.get("tops_per_core") is not None:
            kw["tops_per_core"] = float(body["tops_per_core"])
        if body.get("sku"):
            kw["sku"] = str(body["sku"])

    # Workload / extras
    if body.get("prompt_len") is not None:
        kw["prompt_len"] = int(body["prompt_len"])
    elif body.get("prompt") is not None:
        kw["prompt_len"] = int(body["prompt"])
    if body.get("decode_seq_len") is not None:
        kw["decode_seq_len"] = int(body["decode_seq_len"])
    elif body.get("ctx") is not None:
        kw["decode_seq_len"] = int(body["ctx"])
    if body.get("batch") is not None:
        kw["batch"] = int(body["batch"])
    if body.get("sram_mib") is not None:
        kw["sram_mib"] = float(body["sram_mib"])
    if body.get("dtype") is not None:
        kw["dtype"] = str(body["dtype"])
    if body.get("quant") is not None:
        kw["quant"] = str(body["quant"])
    if body.get("weight_bits") is not None:
        kw["weight_bits"] = int(body["weight_bits"])
    if body.get("kv_bits") is not None:
        kw["kv_bits"] = int(body["kv_bits"])
    if body.get("act_bits") is not None:
        kw["act_bits"] = int(body["act_bits"])
    if body.get("mac_efficiency") is not None:
        # product alias: weight_hide-style assumed MAC/mem hide — map to weight_hide_factor
        # Keep honest: this is an assumed overlap knob, not calibrated util.
        kw["weight_hide_factor"] = float(body["mac_efficiency"])
    if body.get("weight_hide_factor") is not None:
        kw["weight_hide_factor"] = float(body["weight_hide_factor"])
    if body.get("contention_mode") is not None:
        kw["contention_mode"] = str(body["contention_mode"])
    if body.get("frequency_hz") is not None:
        kw["frequency_hz"] = float(body["frequency_hz"])
    elif body.get("freq_ghz") is not None:
        kw["frequency_hz"] = float(body["freq_ghz"]) * 1e9
    if body.get("c2c_gbps") is not None:
        kw["c2c_gbps"] = float(body["c2c_gbps"])
    if body.get("kv_fabric") is not None:
        kw["kv_fabric"] = str(body["kv_fabric"])
    # v0.29 optimism-fix knobs (assumed)
    if body.get("c2c_latency_us") is not None:
        kw["c2c_latency_us"] = float(body["c2c_latency_us"])
    elif body.get("alpha_us") is not None:
        kw["c2c_latency_us"] = float(body["alpha_us"])
    if body.get("sync_overlap") is not None:
        kw["sync_overlap"] = float(body["sync_overlap"])
    if body.get("attn_parallel") is not None:
        kw["attn_parallel"] = str(body["attn_parallel"]).strip().lower()
    # v0.31 engine knobs: MoE sharding + speculative decoding / MTP
    if body.get("moe_shard") not in (None, ""):
        kw["moe_shard"] = str(body["moe_shard"]).strip().lower()
    if body.get("spec_k") not in (None, ""):
        kw["spec_k"] = int(body["spec_k"])
    if body.get("spec_accept") not in (None, ""):
        kw["spec_accept"] = float(body["spec_accept"])
    if body.get("spec_draft") not in (None, ""):
        kw["spec_draft"] = str(body["spec_draft"]).strip().lower()
    if body.get("spec_draft_frac") not in (None, ""):
        kw["spec_draft_frac"] = float(body["spec_draft_frac"])

    # Domain-specific knobs (video / protein)
    if body.get("n_denoise") is not None:
        kw["n_denoise"] = int(body["n_denoise"])
    if body.get("n_frames") is not None:
        kw["n_frames"] = int(body["n_frames"])
    elif body.get("frames") is not None:
        kw["n_frames"] = int(body["frames"])
    if body.get("seq_len") is not None:
        kw["seq_len"] = int(body["seq_len"])
    elif body.get("seq") is not None:
        kw["seq_len"] = int(body["seq"])
    if body.get("decode_mb") is not None:
        kw["decode_mb"] = int(body["decode_mb"])
    elif body.get("mb") is not None:
        kw["decode_mb"] = int(body["mb"])

    # Nested calib / calibration object wins last (optional measured inject)
    calib_raw = body.get("calib") if body.get("calib") is not None else body.get("calibration")
    if calib_raw is not None:
        if isinstance(calib_raw, str):
            # JSON string or filesystem path
            s = calib_raw.strip()
            if s.startswith("{"):
                import json as _json
                cal = CalibrationOverrides.from_dict(_json.loads(s))
            else:
                cal = CalibrationOverrides.load_json(s)
        elif isinstance(calib_raw, dict):
            cal = CalibrationOverrides.from_dict(calib_raw)
        else:
            raise TypeError("calib must be object, JSON string, or path")
        kw = cal.apply_to_kwargs(kw)

    # v0.25 energy / cost stub knobs (flat keys, then nested econ object wins)
    flat_econ = {k: body[k] for k in body if k in ECON_FIELD_NAMES or k in (
        "tdp", "cost_per_card", "w_per_tops", "util_factor", "mem_addon",
    )}
    if flat_econ:
        kw = EnergyCostKnobs.from_dict(flat_econ).apply_to_kwargs(kw)
    econ_raw = body.get("econ") if body.get("econ") is not None else body.get("energy_cost")
    if econ_raw is not None:
        if isinstance(econ_raw, str):
            s_ = econ_raw.strip()
            if s_.startswith("{"):
                econ = EnergyCostKnobs.from_dict(json.loads(s_))
            else:
                econ = EnergyCostKnobs.load_json(s_)
        elif isinstance(econ_raw, dict):
            econ = EnergyCostKnobs.from_dict(econ_raw)
        else:
            raise TypeError("econ must be object, JSON string, or path")
        kw = econ.apply_to_kwargs(kw)

    # v0.26 opt-in assumed non-GEMM overhead + dtype MAC factors
    if body.get("non_gemm_overhead") is not None:
        kw["non_gemm_overhead"] = float(body["non_gemm_overhead"])
    for src, dst in (
        ("softmax_frac", "softmax_frac"),
        ("rope_frac", "rope_frac"),
        ("norm_frac", "norm_frac"),
    ):
        if body.get(src) is not None:
            kw[dst] = float(body[src])
    dmf = body.get("dtype_mac_factors")
    if dmf is None:
        dmf = body.get("dtype_mac_factor")
    if dmf is not None:
        if isinstance(dmf, str):
            s = dmf.strip()
            if s.startswith("{"):
                dmf = json.loads(s)
            else:
                dmf = json.loads(Path(s).read_text(encoding="utf-8"))
        if not isinstance(dmf, dict):
            raise TypeError("dtype_mac_factors must be a JSON object / dict")
        kw["dtype_mac_factors"] = {str(k): float(v) for k, v in dmf.items()}

    return WorkbenchConfig(**kw)


def api_presets() -> dict[str, Any]:
    """Scenario presets + EXAMPLE econ placeholders (fake; labeled)."""
    presets = list_presets()
    return {
        "count": len(presets),
        "banner": HONESTY_BANNER,
        "presets": [dict(p.to_dict(), body=p.body()) for p in presets],
        "econ_example": load_example_econ(),
        "econ_fields": list(ECON_FIELD_NAMES),
    }


def api_eval(body: dict[str, Any]) -> dict[str, Any]:
    """Evaluate WorkbenchConfig → MetricsCard (+ honesty banner)."""
    cfg = _build_workbench_from_body(body)
    card = evaluate_workbench(cfg)
    out = card.to_json_dict()
    out["banner"] = HONESTY_BANNER
    out["ok"] = True
    out["version"] = __version__
    # Surface config echo for UI
    out["config_echo"] = {
        "model_id": cfg.model_id,
        "domain": out.get("domain", "llm"),
        "chip_count": cfg.chip_count,
        "mem_kind": str(cfg.mem_kind),
        "n_cores": cfg.n_cores,
        "tops_per_core": cfg.tops_per_core,
        "n_channels": cfg.n_channels,
        "width_bits": cfg.width_bits,
        "data_rate_GTs": cfg.data_rate_GTs,
        "efficiency": cfg.efficiency,
        "batch": cfg.batch,
        "prompt_len": cfg.prompt_len,
        "decode_seq_len": cfg.decode_seq_len,
        "n_denoise": cfg.n_denoise,
        "n_frames": cfg.n_frames,
        "seq_len": cfg.seq_len,
        "frequency_hz": cfg.frequency_hz,
        "weight_hide_factor": cfg.weight_hide_factor,
        "mac_efficiency": cfg.mac_efficiency,
        "non_gemm_overhead": cfg.non_gemm_overhead,
        "softmax_frac": cfg.softmax_frac,
        "rope_frac": cfg.rope_frac,
        "norm_frac": cfg.norm_frac,
        "dtype_mac_factors": cfg.dtype_mac_factors,
        "c2c_latency_us": cfg.c2c_latency_us,
        "sync_overlap": cfg.sync_overlap,
        "attn_parallel": cfg.attn_parallel,
        "moe_shard": cfg.moe_shard,
        "spec_k": cfg.spec_k,
        "spec_accept": cfg.spec_accept,
        "spec_draft": cfg.spec_draft,
        "decode_mb": cfg.decode_mb,
        **{k: getattr(cfg, k) for k in STRUCT_MEM_KEYS},
        "mem_legacy_id": cfg.mem_legacy_id,
        **{k: getattr(cfg, k) for k in ECON_FIELD_NAMES},
    }
    from .workbench import build_mem_spec, default_spec_for_preset, build_memory

    spec = build_mem_spec(cfg)
    if spec is None:
        mem, _ = build_memory(cfg)
        spec = default_spec_for_preset(mem)
    out["mem_spec"] = spec.to_dict() if spec is not None else None
    return out


SWEEP_MAX_ROWS = 32
SWEEP_AXES = ("chips", "package", "compute", "parallel", "series")


def _primary_metric_key(domain: str) -> str:
    d = (domain or "llm").lower()
    if d == "video":
        return "TTFC_ms"
    if d == "protein":
        return "time_per_seq_ms"
    return "TPOT_ms"


def _primary_metric_value(card: dict[str, Any]) -> float:
    key = _primary_metric_key(str(card.get("domain") or "llm"))
    if key == "TTFC_ms":
        return float(card.get("TTFC_ms") or card.get("TTFT_ms") or 0.0)
    if key == "time_per_seq_ms":
        return float(card.get("time_per_seq_ms") or card.get("TTFT_ms") or 0.0)
    return float(card.get("TPOT_ms") or 0.0)


def _card_from_metrics(card_obj: Any) -> dict[str, Any]:
    out = card_obj.to_json_dict()
    out["banner"] = HONESTY_BANNER
    out["ok"] = True
    out["version"] = __version__
    out["primary_ms"] = _primary_metric_value(out)
    return out


def _short_pkg_label(short_label: str, raw_id: str) -> str:
    """v0.30: compact sweep-row label — '<type geometry @rate> · <cap>' from
    MemSpec.short_label() (drops bandwidth / maturity which are columns already).
    Falls back to the raw id."""
    parts = [x.strip() for x in str(short_label or "").split(" · ") if x.strip()]
    if not parts:
        return str(raw_id)
    cap = next((x for x in parts[1:] if x.endswith(" GB")), "")
    return f"{parts[0]} · {cap}" if cap else parts[0]

def api_sweep(body: dict[str, Any]) -> dict[str, Any]:
    """Run a small DSE grid along one knob axis; return MetricsCards (≤32).

    Axes:
      chips     — chip_count list (default 1/2/4/8); tp=chips
      package   — named package catalog (kind filter optional)
      compute   — core / cluster / all compute presets
      parallel  — tp×pp×ep matrix for chips ∈ {4,8} (or body chip_count)
      series    — series ids in domain (cap ≤32)

    Body accepts the same base knobs as /api/eval plus:
      axis, values / chips_list / package_ids / compute_ids / series_ids,
      package_kind, compute_level, parallel_chips, max_rows.
    """
    axis_raw = str(body.get("axis") or body.get("sweep") or "chips").strip().lower()
    axis_aliases = {
        "chip": "chips",
        "chip_count": "chips",
        "packages": "package",
        "mem": "package",
        "pkg": "package",
        "cores": "compute",
        "cluster": "compute",
        "tp_pp": "parallel",
        "tp/pp": "parallel",
        "matrix": "parallel",
        "models": "series",
        "model": "series",
    }
    axis = axis_aliases.get(axis_raw, axis_raw)
    if axis not in SWEEP_AXES:
        raise ValueError(
            f"axis must be one of {SWEEP_AXES}, got {axis_raw!r}"
        )

    max_rows = int(body.get("max_rows", SWEEP_MAX_ROWS))
    max_rows = max(1, min(max_rows, SWEEP_MAX_ROWS))

    # Base knobs shared with /api/eval (strip sweep-only keys)
    base_body = dict(body)
    for k in (
        "axis",
        "sweep",
        "values",
        "chips_list",
        "package_ids",
        "compute_ids",
        "series_ids",
        "package_kind",
        "compute_level",
        "parallel_chips",
        "max_rows",
        "package_axis",
        "mem_axis",
    ):
        base_body.pop(k, None)

    rows: list[dict[str, Any]] = []
    requested = 0
    domain_hint = "llm"

    if axis == "chips":
        vals = body.get("values") or body.get("chips_list") or list(DEFAULT_CHIP_SWEEP)
        vals = [int(v) for v in vals]
        requested = len(vals)
        vals = vals[:max_rows]
        for n in vals:
            b = dict(base_body)
            b["chip_count"] = int(n)
            b.pop("tp", None)
            b.pop("pp", None)
            b.pop("ep", None)
            card = api_eval(b)
            domain_hint = str(card.get("domain") or domain_hint)
            rows.append(
                {
                    "label": f"chips={n}",
                    "axis_value": int(n),
                    "axis_key": "chips",
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    elif axis == "package" and (body.get("package_axis") or body.get("mem_axis")):
        pax = str(body.get("package_axis") or body.get("mem_axis")).strip().lower()
        pax = {"类型": "type", "数量": "count", "速率": "rate", "gen": "type",
               "stacks": "count", "packages": "count", "speed": "rate"}.get(pax, pax)
        if pax not in ("type", "count", "rate"):
            raise ValueError(f"package_axis must be type|count|rate, got {pax!r}")
        base_cfg = _build_workbench_from_body(base_body)
        from .workbench import build_mem_spec, build_memory, default_spec_for_preset

        spec = build_mem_spec(base_cfg)
        if spec is None:
            spec = default_spec_for_preset(build_memory(base_cfg)[0]) or make_spec(
                "HBM3E" if str(base_cfg.mem_kind).upper() == "HBM" else "LPDDR5X"
            )
        specs = sweep_specs(spec, pax)
        requested = len(specs)
        for sp in specs[:max_rows]:
            b = dict(base_body)
            for mk in (
                "package_id", "package", "mem_kind", "mem", "n_channels",
                "width_bits", "data_rate_GTs", "data_rate_gts", "n_packages",
                "capacity_GB", "capacity_gb", *STRUCT_MEM_KEYS,
            ):
                b.pop(mk, None)
            b.update({k: v for k, v in sp.workbench_kwargs().items()
                      if k in STRUCT_MEM_KEYS and v is not None})
            card = api_eval(b)
            domain_hint = str(card.get("domain") or domain_hint)
            rows.append(
                {
                    "label": _short_pkg_label(sp.short_label(), sp.id),
                    "raw_id": sp.id,
                    "axis_value": sp.id,
                    "axis_key": "package",
                    "package_axis": pax,
                    "package_id": sp.id,
                    "n_packages": sp.n_units,
                    "package_width_bits": sp.unit_width_bits,
                    "mem_tag": sp.tag,
                    "mem_tag_zh": sp.tag_zh,
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    elif axis == "package":
        kind = body.get("package_kind") or body.get("kind")
        if body.get("values") or body.get("package_ids"):
            ids = list(body.get("values") or body.get("package_ids"))
        elif kind:
            k = str(kind).upper()
            if k == "LPDDR":
                ids = list(DEFAULT_PACKAGE_SWEEP_LPDDR)
            elif k == "HBM":
                ids = list(DEFAULT_PACKAGE_SWEEP_HBM)
            else:
                ids = [p.id for p in list_packages(kind=k)]  # type: ignore[arg-type]
        else:
            # Prefer current mem kind from body / package_id, else compact HBM set
            mem_kind = str(base_body.get("mem_kind") or "").upper()
            pid = base_body.get("package_id") or base_body.get("package")
            if pid:
                try:
                    mem_kind = get_package(str(pid)).kind
                except KeyError:
                    pass
            if mem_kind == "LPDDR":
                ids = list(DEFAULT_PACKAGE_SWEEP_LPDDR)
            else:
                ids = list(DEFAULT_PACKAGE_SWEEP_HBM)
        requested = len(ids)
        ids = [str(x) for x in ids][:max_rows]
        for pid in ids:
            pkg = get_package(pid)
            b = dict(base_body)
            b["package_id"] = pid
            # Drop manual geometry / structured fields so package wins
            for mk in (
                "mem_kind",
                "n_channels",
                "width_bits",
                "data_rate_GTs",
                "data_rate_gts",
                "n_packages",
                "capacity_GB",
                "capacity_gb",
                *STRUCT_MEM_KEYS,
            ):
                b.pop(mk, None)
            card = api_eval(b)
            domain_hint = str(card.get("domain") or domain_hint)
            gran = pkg.package_width_bits or pkg.width_bits
            npk = pkg.n_packages or pkg.n_stacks or pkg.n_channels
            # v0.30: short human label; raw catalog id kept in raw_id (tooltip / CSV)
            label = _short_pkg_label(pkg.mem_spec.short_label(), pid)
            rows.append(
                {
                    "label": label,
                    "raw_id": pid,
                    "axis_value": pid,
                    "axis_key": "package",
                    "package_id": pid,
                    "n_packages": npk,
                    "package_width_bits": gran,
                    "mem_tag": pkg.mem_spec.tag,
                    "mem_tag_zh": pkg.mem_spec.tag_zh,
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    elif axis == "compute":
        level = str(body.get("compute_level") or body.get("level") or "all").lower()
        if body.get("values") or body.get("compute_ids"):
            ids = list(body.get("values") or body.get("compute_ids"))
        elif level == "core":
            ids = list(DEFAULT_COMPUTE_SWEEP_CORE)
        elif level == "cluster":
            ids = list(DEFAULT_COMPUTE_SWEEP_CLUSTER)
        else:
            ids = list(DEFAULT_COMPUTE_SWEEP)
        requested = len(ids)
        ids = [str(x) for x in ids][:max_rows]
        for cid in ids:
            opt = get_compute(cid)
            b = dict(base_body)
            b["compute_id"] = cid
            for mk in ("n_cores", "tops_per_core", "cores", "sku"):
                b.pop(mk, None)
            card = api_eval(b)
            domain_hint = str(card.get("domain") or domain_hint)
            rows.append(
                {
                    "label": f"{cid} ({opt.peak_tops:g}T)",
                    "axis_value": cid,
                    "axis_key": "compute",
                    "compute_id": cid,
                    "peak_tops": opt.peak_tops,
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    elif axis == "parallel":
        if body.get("parallel_chips") is not None:
            chips = int(body["parallel_chips"])
        elif isinstance(body.get("values"), list) and body.get("values"):
            chips = int(body["values"][0])
        else:
            chips = int(body.get("chip_count") or body.get("chips") or 4)
        if chips < 1:
            raise ValueError("parallel_chips must be >= 1")
        model_id = (
            base_body.get("model_id")
            or base_body.get("model")
            or base_body.get("series")
            or base_body.get("series_id")
            or "illustrative_27B"
        )
        # Build WorkbenchConfig once for shared knobs
        cfg = _build_workbench_from_body(
            {**base_body, "chip_count": chips, "model_id": model_id}
        )
        # drop parallel override — matrix sets it
        cfg_kw = {
            "mem_kind": cfg.mem_kind,
            "n_packages": cfg.n_packages,
            "n_ranks": cfg.n_ranks,
            "n_channels": cfg.n_channels,
            "data_rate_GTs": cfg.data_rate_GTs,
            "width_bits": cfg.width_bits,
            "efficiency": cfg.efficiency,
            "capacity_GB": cfg.capacity_GB,
            "n_cores": cfg.n_cores,
            "tops_per_core": cfg.tops_per_core,
            "prompt_len": cfg.prompt_len,
            "decode_seq_len": cfg.decode_seq_len,
            "batch": cfg.batch,
            "dtype": cfg.dtype,
            "quant": cfg.quant,
            "sram_mib": cfg.sram_mib,
            "weight_hide_factor": cfg.weight_hide_factor,
            "n_denoise": cfg.n_denoise,
            "n_frames": cfg.n_frames,
            "seq_len": cfg.seq_len,
            "decode_mb": cfg.decode_mb,
            "c2c_gbps": cfg.c2c_gbps,
            "kv_fabric": cfg.kv_fabric,
            "contention_mode": cfg.contention_mode,
            **{k: getattr(cfg, k) for k in ECON_FIELD_NAMES},
        }
        # Remove Nones
        cfg_kw = {k: v for k, v in cfg_kw.items() if v is not None}
        cards = workbench_parallel_matrix(
            str(model_id),
            int(chips),
            sort_by=str(body.get("sort_by") or "TPOT"),
            **cfg_kw,
        )
        requested = len(cards)
        cards = cards[:max_rows]
        for c in cards:
            card = _card_from_metrics(c)
            domain_hint = str(card.get("domain") or domain_hint)
            label = f"tp={c.tp} pp={c.pp} ep={c.ep}"
            rows.append(
                {
                    "label": label,
                    "axis_value": f"{c.tp}x{c.pp}x{c.ep}",
                    "axis_key": "parallel",
                    "tp": c.tp,
                    "pp": c.pp,
                    "ep": c.ep,
                    "chips": c.chips,
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    elif axis == "series":
        domain = body.get("domain")
        if not domain:
            # infer from first series id or leave None
            mid = (
                base_body.get("model_id")
                or base_body.get("model")
                or base_body.get("series")
            )
            if mid:
                try:
                    from .series import get_series as _gs

                    domain = _gs(str(mid)).domain
                except Exception:
                    domain = None
        product_only = bool(body.get("product_only", False))
        if body.get("values") or body.get("series_ids"):
            ids = [str(x) for x in (body.get("values") or body.get("series_ids"))]
        else:
            catalog = api_series(
                domain=str(domain) if domain else None,
                product_only=product_only,
            )
            ids = [s["id"] for s in catalog["series"]]
        requested = len(ids)
        ids = ids[:max_rows]
        for sid in ids:
            b = dict(base_body)
            b["model_id"] = sid
            card = api_eval(b)
            domain_hint = str(card.get("domain") or domain_hint)
            rows.append(
                {
                    "label": sid,
                    "axis_value": sid,
                    "axis_key": "series",
                    "model_id": sid,
                    "card": card,
                    "primary_ms": _primary_metric_value(card),
                }
            )

    metric_key = _primary_metric_key(domain_hint)
    return {
        "ok": True,
        "axis": axis,
        "count": len(rows),
        "requested": requested,
        "capped": requested > len(rows),
        "max_rows": max_rows,
        "metric_key": metric_key,
        "domain": domain_hint,
        "banner": HONESTY_BANNER,
        "version": __version__,
        "rows": rows,
    }




def delta_pct_vs_baseline(value: float, baseline: float | None) -> float | None:
    """Relative Δ% = (value - baseline) / baseline * 100. None if baseline missing/0."""
    if baseline is None:
        return None
    try:
        b = float(baseline)
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not (b == b) or b == 0.0:  # NaN or zero
        return None
    return (v - b) / b * 100.0


def sweep_result_to_json(result: dict[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a /api/sweep response dict to JSON text (stable keys)."""
    return json.dumps(result, indent=indent, default=str, ensure_ascii=False)


def sweep_result_to_csv(result: dict[str, Any]) -> str:
    """Flatten sweep rows to CSV (one row per config; primary + key MetricsCard fields).

    Client-side UI also builds CSV from the same last `/api/sweep` payload; this
    helper is the server-side reference + test surface.
    """
    import csv
    import io

    rows = result.get("rows") or []
    metric_key = str(result.get("metric_key") or "primary_ms")
    domain = str(result.get("domain") or "llm")
    baseline = result.get("baseline_ms")
    buf = io.StringIO()
    fieldnames = [
        "axis",
        "label",
        "axis_value",
        "domain",
        "metric_key",
        "primary_ms",
        "delta_pct",
        "TTFT_ms",
        "TPOT_ms",
        "TTFC_ms",
        "time_per_seq_ms",
        "frames_per_s",
        "pair_bytes",
        "wall",
        "util",
        "peak_tops",
        "mem_eff_GBps",
        "tp",
        "pp",
        "ep",
        "chips",
        "oom",
        "scale_efficiency",
        "speedup",
        "t_single_primary_ms",
        "scale_metric",
        "model_id",
        "est_power_W",
        "est_energy_per_token_J",
        "est_energy_per_frame_J",
        "est_energy_per_seq_J",
        "est_system_cost_usd",
        "est_usd_per_Mtok",
        "raw_id",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        card = r.get("card") or {}
        primary = r.get("primary_ms")
        if primary is None:
            primary = _primary_metric_value(card) if card else None
        dlt = r.get("delta_pct")
        if dlt is None and baseline is not None and primary is not None:
            dlt = delta_pct_vs_baseline(float(primary), float(baseline))
        w.writerow(
            {
                "axis": result.get("axis"),
                "label": r.get("label"),
                "raw_id": r.get("raw_id") or r.get("axis_value"),
                "axis_value": r.get("axis_value"),
                "domain": card.get("domain") or domain,
                "metric_key": metric_key,
                "primary_ms": primary,
                "delta_pct": "" if dlt is None else round(float(dlt), 4),
                "TTFT_ms": card.get("TTFT_ms"),
                "TPOT_ms": card.get("TPOT_ms"),
                "TTFC_ms": card.get("TTFC_ms"),
                "time_per_seq_ms": card.get("time_per_seq_ms"),
                "frames_per_s": card.get("frames_per_s"),
                "pair_bytes": card.get("pair_bytes"),
                "wall": card.get("wall"),
                "util": card.get("util"),
                "peak_tops": card.get("peak_tops"),
                "mem_eff_GBps": card.get("mem_eff_GBps"),
                "tp": card.get("tp"),
                "pp": card.get("pp"),
                "ep": card.get("ep"),
                "chips": card.get("chips"),
                "oom": card.get("oom"),
                "scale_efficiency": card.get("scale_efficiency"),
                "speedup": card.get("speedup"),
                "t_single_primary_ms": card.get("t_single_primary_ms"),
                "scale_metric": card.get("scale_metric"),
                "model_id": card.get("model_id") or r.get("model_id"),
                "est_power_W": card.get("est_power_W"),
                "est_energy_per_token_J": card.get("est_energy_per_token_J"),
                "est_energy_per_frame_J": card.get("est_energy_per_frame_J"),
                "est_energy_per_seq_J": card.get("est_energy_per_seq_J"),
                "est_system_cost_usd": card.get("est_system_cost_usd"),
                "est_usd_per_Mtok": card.get("est_usd_per_Mtok"),
            }
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Stdlib HTTP server
# ---------------------------------------------------------------------------


PARETO_BODY_KEYS = ("slo_ttft_ms", "slo_tpot_ms", "slo_latency_ms", "layouts", "batch_limit", "slo",
                    "goodput_mode", "prefill_mode", "out_len")


def api_pareto(body: dict[str, Any]) -> dict[str, Any]:
    """v0.30 throughput–interactivity Pareto + SLO goodput for the scenario in ``body``.

    Body = /api/eval body plus optional ``slo_ttft_ms`` / ``slo_tpot_ms`` (LLM),
    ``slo_latency_ms`` (video TTFC / protein time-per-seq), ``layouts``
    (``all`` | ``current``) and ``batch_limit`` (≤ LLM_BATCH_LIMIT).
    v0.31 (LLM): ``goodput_mode`` (``amortized`` default | ``upper``),
    ``prefill_mode`` (``chunked`` default | ``exclusive``), ``out_len`` (N output
    tokens per request, default 256) — prefill-amortised steady-state goodput.
    """
    from .pareto import LLM_BATCH_LIMIT, pareto_analysis

    body = dict(body or {})
    extra = {k: body.pop(k) for k in PARETO_BODY_KEYS if k in body}
    cfg = _build_workbench_from_body(body)
    slo: dict[str, float] = {}
    nested = extra.get("slo") or {}
    for src, dst in (("slo_ttft_ms", "ttft_ms"), ("slo_tpot_ms", "tpot_ms"), ("slo_latency_ms", "latency_ms")):
        v = extra.get(src, nested.get(dst) if isinstance(nested, dict) else None)
        if v is not None and v != "":
            fv = float(v)
            if not fv > 0:
                raise ValueError(f"{src} must be > 0")
            slo[dst] = fv
    layouts = str(extra.get("layouts") or "all")
    if layouts not in ("all", "current"):
        raise ValueError("layouts must be all|current")
    bl = extra.get("batch_limit")
    bl_i = None if bl in (None, "") else max(1, min(int(bl), LLM_BATCH_LIMIT))
    gm = str(extra.get("goodput_mode") or "amortized").strip().lower()
    pm = str(extra.get("prefill_mode") or "chunked").strip().lower()
    ol = extra.get("out_len")
    ol_i = None if ol in (None, "") else int(ol)
    out = pareto_analysis(cfg, slo=slo, layouts=layouts, batch_limit=bl_i,
                          goodput_mode=gm, prefill_mode=pm, out_len=ol_i)
    out["version"] = __version__
    return out


def pareto_result_to_csv(result: dict[str, Any]) -> str:
    from .pareto import pareto_to_csv

    return pareto_to_csv(result)


def _text_response(handler: BaseHTTPRequestHandler, status: int, text: str, ctype: str) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)

def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _serve_static(handler: BaseHTTPRequestHandler, rel: str) -> None:
    rel = rel.lstrip("/")
    if not rel or rel == "/":
        rel = "index.html"
    # Prevent path traversal
    target = (WEB_DIR / rel).resolve()
    if not str(target).startswith(str(WEB_DIR.resolve())):
        handler.send_error(403, "Forbidden")
        return
    if not target.is_file():
        handler.send_error(404, f"Not found: {rel}")
        return
    ctype, _ = mimetypes.guess_type(str(target))
    data = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


class WorkbenchHandler(BaseHTTPRequestHandler):
    """ThreadingHTTPServer handler for workbench API + static UI."""

    server_version = f"accel_dse/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quieter default; still useful on stderr
        sys_stderr = __import__("sys").stderr
        print(f"[serve] {self.address_string()} {fmt % args}", file=sys_stderr)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path in ("/api/health", "/api/version"):
                _json_response(self, 200, api_health())
                return
            if path == "/api/series":
                domain = (qs.get("domain") or [None])[0]
                po = (qs.get("product_only") or ["0"])[0].lower() in (
                    "1",
                    "true",
                    "yes",
                )
                _json_response(
                    self, 200, api_series(domain=domain, product_only=po)
                )
                return
            if path == "/api/packages":
                kind = (qs.get("kind") or [None])[0]
                _json_response(self, 200, api_packages(kind=kind))
                return
            if path == "/api/memory":
                res = (qs.get("resolve") or [None])[0]
                try:
                    _json_response(self, 200, api_memory(resolve=res))
                except (KeyError, ValueError) as exc:
                    _json_response(self, 400, {"ok": False, "error": str(exc)})
                return
            if path == "/api/compute":
                level = (qs.get("level") or ["all"])[0]
                aliases = (qs.get("all_aliases") or ["0"])[0].lower() in (
                    "1",
                    "true",
                    "yes",
                )
                _json_response(
                    self, 200, api_compute(level=level, all_aliases=aliases)
                )
                return
            if path == "/api/presets":
                _json_response(self, 200, api_presets())
                return
            # Static
            if path in ("/", "/index.html"):
                _serve_static(self, "index.html")
                return
            if path.startswith("/static/") or path.startswith("/web/"):
                rel = path.split("/", 2)[-1]
                _serve_static(self, rel)
                return
            # Direct asset names under web/
            if path.lstrip("/") in ("app.js", "style.css", "favicon.ico"):
                _serve_static(self, path.lstrip("/"))
                return
            self.send_error(404, f"Unknown path {path}")
        except Exception as exc:  # noqa: BLE001
            _json_response(
                self,
                500,
                {"ok": False, "error": str(exc), "trace": traceback.format_exc()},
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/eval":
                body = _read_json_body(self)
                result = api_eval(body)
                _json_response(self, 200, result)
                return
            if path == "/api/sweep":
                body = _read_json_body(self)
                result = api_sweep(body)
                _json_response(self, 200, result)
                return
            if path in ("/api/pareto", "/api/pareto.csv"):
                body = _read_json_body(self)
                result = api_pareto(body)
                if path.endswith(".csv"):
                    _text_response(self, 200, pareto_result_to_csv(result), "text/csv; charset=utf-8")
                else:
                    _json_response(self, 200, result)
                return
            self.send_error(404, f"Unknown path {path}")
        except NotImplementedError as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
        except (KeyError, ValueError, TypeError) as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            _json_response(
                self,
                500,
                {"ok": False, "error": str(exc), "trace": traceback.format_exc()},
            )


def create_fastapi_app():
    """Build a FastAPI app if fastapi is installed; else raise ImportError."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="accel_dse workbench",
        version=__version__,
        description=HONESTY_BANNER,
    )

    @app.get("/api/health")
    def health():
        return api_health()

    @app.get("/api/series")
    def series(
        domain: str | None = None,
        product_only: bool = False,
    ):
        return api_series(domain=domain, product_only=product_only)

    @app.get("/api/packages")
    def packages(kind: str | None = None):
        return api_packages(kind=kind)

    @app.get("/api/memory")
    def memory(resolve: str | None = None):
        try:
            return api_memory(resolve=resolve)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/compute")
    def compute(level: str = "all", all_aliases: bool = False):
        return api_compute(level=level, all_aliases=all_aliases)

    @app.get("/api/presets")
    def presets():
        return api_presets()

    @app.post("/api/eval")
    def eval_endpoint(body: dict[str, Any]):
        try:
            return api_eval(body)
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sweep")
    def sweep_endpoint(body: dict[str, Any]):
        try:
            return api_sweep(body)
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pareto")
    def pareto_endpoint(body: dict[str, Any]):
        try:
            return api_pareto(body)
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pareto.csv")
    def pareto_csv_endpoint(body: dict[str, Any]):
        from fastapi.responses import PlainTextResponse

        try:
            return PlainTextResponse(pareto_result_to_csv(api_pareto(body)), media_type="text/csv")
        except (KeyError, ValueError, TypeError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/")
    def index():
        index_path = WEB_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="index.html missing")
        return FileResponse(index_path)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/app.js")
        def app_js():
            return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

        @app.get("/style.css")
        def style_css():
            return FileResponse(WEB_DIR / "style.css", media_type="text/css")

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    prefer_fastapi: bool = True,
) -> None:
    """Start the workbench server (blocking). Prefer FastAPI+uvicorn if present."""
    if prefer_fastapi:
        try:
            import uvicorn

            app = create_fastapi_app()
            print(
                f"accel_dse workbench v{__version__} (FastAPI) "
                f"http://{host}:{port}/",
                flush=True,
            )
            print(f"  honesty: {HONESTY_BANNER[:80]}…", flush=True)
            uvicorn.run(app, host=host, port=port, log_level="info")
            return
        except ImportError:
            pass

    httpd = ThreadingHTTPServer((host, port), WorkbenchHandler)
    print(
        f"accel_dse workbench v{__version__} (stdlib http.server) "
        f"http://{host}:{port}/",
        flush=True,
    )
    print(f"  honesty: {HONESTY_BANNER[:80]}…", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down")
    finally:
        httpd.server_close()


__all__ = [
    "HONESTY_BANNER",
    "WEB_DIR",
    "SWEEP_MAX_ROWS",
    "SWEEP_AXES",
    "api_health",
    "api_series",
    "api_packages",
    "api_compute",
    "api_eval",
    "api_sweep",
    "delta_pct_vs_baseline",
    "sweep_result_to_csv",
    "sweep_result_to_json",
    "create_fastapi_app",
    "run_server",
    "WorkbenchHandler",
    "CalibrationOverrides",
]
