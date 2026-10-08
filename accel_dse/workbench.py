"""Unified workbench: one WorkbenchConfig → one MetricsCard.

Reuses evaluate / scaleup / traffic / workloads physics — no forked roofline.
Product layer: series registry, chip_count→parallel (llm/video/protein tp/pp/ep),
DRAM geometry, cores×tops_per_core, stable MetricsCard fields for all domains.
"""

from __future__ import annotations

GIB = 1 << 30  # v0.30: vendor/JEDEC "GB" (binary) used for all capacity_GB fields

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .dtype import dtype_mac_factors_active, get_dtype, get_quant, resolve_dtype_mac_factor
from .econ import apply_energy_cost, econ_kwargs_from_obj
from .evaluate import EvalConfig, evaluate_inference, resolved_non_gemm_overhead
from .memory import (
    HBM_PRESET,
    LPDDR_PRESET,
    ExternalMemory,
    MemKind,
    SRAMConfig,
    memory_from_geometry,
)
from .model_shape import ModelShape
from .npu import NPUConfig, cores_from_npu, npu_from_cores, npu_from_pe
from .mem_catalog import MemSpec, make_spec
from .scaleup import (
    DEFAULT_C2C_LATENCY_US,
    DEFAULT_SYNC_OVERLAP,
    TPConfig,
    sync_collectives_per_token,
    evaluate_scaleup,
    get_c2c,
    get_kv_fabric,
    scale_domain_parallel,
)
from .series import (
    get_series,
    resolve_llm_shape,
    resolve_protein_shape,
    resolve_video_shape,
)
from .sku import get_sku
from .workloads import (
    ProteinShape,
    VideoShape,
    evaluate_protein,
    evaluate_video,
)

ContentionMode = Literal["serialize", "share_fair", "lower_bound"]
DEFAULT_CHIP_SWEEP = (1, 2, 4, 8)


@dataclass(frozen=True)
class ParallelOverride:
    """Explicit {tp, pp, ep}; must satisfy tp*pp*ep == chip_count."""

    tp: int = 1
    pp: int = 1
    ep: int = 1


# Defaults for assumed / override knobs (UI badge + calib honesty)
DEFAULT_MEM_EFFICIENCY = 0.70
DEFAULT_WEIGHT_HIDE = 0.0
DEFAULT_FREQUENCY_HZ = 1.0e9
DEFAULT_NPU_MAC_EFFICIENCY = 1.0


@dataclass(frozen=True)
class CalibrationOverrides:
    """Optional measured / assumed efficiency injection (non-blocking).

    JSON / API schema (all fields optional)::

        {
          "mem_efficiency": 0.65,      // → WorkbenchConfig.efficiency
          "mac_efficiency": 0.0,       // product alias → weight_hide_factor
          "weight_hide": 0.0,          // explicit weight_hide_factor
          "npu_mac_efficiency": 1.0,   // true NPUConfig.mac_efficiency
          "frequency_hz": 1.0e9,       // or freq_ghz
          "notes": "optional label"
        }

    Missing keys leave WorkbenchConfig defaults untouched. CLI: ``--calib path.json``.
    API: body ``calib: {...}`` (also accepts top-level ``efficiency`` / ``freq_ghz``).
    """

    mem_efficiency: float | None = None
    mac_efficiency: float | None = None  # alias → weight_hide_factor
    weight_hide: float | None = None
    npu_mac_efficiency: float | None = None
    frequency_hz: float | None = None
    freq_ghz: float | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CalibrationOverrides":
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise TypeError("calib must be a JSON object")
        mem = d.get("mem_efficiency", d.get("efficiency"))
        mac = d.get("mac_efficiency")
        wh = d.get("weight_hide", d.get("weight_hide_factor"))
        npu_mac = d.get("npu_mac_efficiency")
        freq = d.get("frequency_hz")
        freq_ghz = d.get("freq_ghz")
        notes = d.get("notes") or d.get("note")
        return cls(
            mem_efficiency=float(mem) if mem is not None else None,
            mac_efficiency=float(mac) if mac is not None else None,
            weight_hide=float(wh) if wh is not None else None,
            npu_mac_efficiency=float(npu_mac) if npu_mac is not None else None,
            frequency_hz=float(freq) if freq is not None else None,
            freq_ghz=float(freq_ghz) if freq_ghz is not None else None,
            notes=str(notes) if notes is not None else None,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CalibrationOverrides":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"calib file must be a JSON object: {p}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.mem_efficiency is not None:
            out["mem_efficiency"] = self.mem_efficiency
        if self.mac_efficiency is not None:
            out["mac_efficiency"] = self.mac_efficiency
        if self.weight_hide is not None:
            out["weight_hide"] = self.weight_hide
        if self.npu_mac_efficiency is not None:
            out["npu_mac_efficiency"] = self.npu_mac_efficiency
        if self.frequency_hz is not None:
            out["frequency_hz"] = self.frequency_hz
        elif self.freq_ghz is not None:
            out["freq_ghz"] = self.freq_ghz
        if self.notes is not None:
            out["notes"] = self.notes
        return out

    def resolved_frequency_hz(self) -> float | None:
        if self.frequency_hz is not None:
            return float(self.frequency_hz)
        if self.freq_ghz is not None:
            return float(self.freq_ghz) * 1e9
        return None

    def resolved_weight_hide(self) -> float | None:
        if self.weight_hide is not None:
            return float(self.weight_hide)
        if self.mac_efficiency is not None:
            return float(self.mac_efficiency)
        return None

    def active_fields(self) -> list[str]:
        keys: list[str] = []
        if self.mem_efficiency is not None:
            keys.append("mem_efficiency")
        if self.mac_efficiency is not None:
            keys.append("mac_efficiency")
        if self.weight_hide is not None:
            keys.append("weight_hide")
        if self.npu_mac_efficiency is not None:
            keys.append("npu_mac_efficiency")
        if self.frequency_hz is not None or self.freq_ghz is not None:
            keys.append("frequency_hz")
        return keys

    def apply_to_kwargs(self, kw: dict[str, Any]) -> dict[str, Any]:
        """Mutate/return kwargs dict for WorkbenchConfig construction."""
        out = dict(kw)
        if self.mem_efficiency is not None:
            out["efficiency"] = float(self.mem_efficiency)
        wh = self.resolved_weight_hide()
        if wh is not None:
            out["weight_hide_factor"] = wh
        if self.npu_mac_efficiency is not None:
            out["mac_efficiency"] = float(self.npu_mac_efficiency)
        freq = self.resolved_frequency_hz()
        if freq is not None:
            out["frequency_hz"] = freq
        return out

    def apply_to_config(self, cfg: "WorkbenchConfig") -> "WorkbenchConfig":
        kw = self.apply_to_kwargs({})
        if not kw:
            return cfg
        return replace(cfg, **kw)


@dataclass
class WorkbenchConfig:
    """First-class unified product config (name flexible; this is the type).

    chip_count (≥1): default interpret as tp=chip_count when pp=ep=1.
    Pass ``parallel`` for explicit {tp,pp,ep} with tp*pp*ep==chip_count.
    """

    model_id: str = "illustrative_27B"
    chip_count: int = 1
    parallel: ParallelOverride | None = None
    # Memory
    mem_kind: MemKind | str = "HBM"
    n_packages: int | None = None
    n_ranks: int | None = None
    n_channels: int | None = None
    data_rate_GTs: float | None = None
    width_bits: int | None = None
    efficiency: float | None = None
    capacity_GB: float | None = None
    # v0.29 structured memory (mem_catalog.make_spec). When mem_type is set it
    # takes precedence over the legacy geometry knobs above (efficiency still
    # applies). Units: LPDDR packages/modules or HBM stacks.
    mem_type: str | None = None  # LPDDR5 | LPDDR5X | LPDDR6 | HBM3 | HBM3E | HBM4 | HBM4E
    mem_form: str | None = None  # discrete | SOCAMM2 | LPCAMM2 (LPDDR only)
    mem_width_bits: int | None = None  # per-package / module width
    mem_rate_MTps: float | None = None
    mem_count: int | None = None  # packages / modules / stacks
    mem_cap_GB: float | None = None  # per package / module (LPDDR)
    hbm_height: int | None = None  # 8 / 12 / 16-high
    hbm_die_Gb: int | None = None  # 16 / 24 / 32 Gb
    mem_legacy_id: str | None = None  # echo of resolved ≤0.28 package id
    # NPU product language OR PE
    n_cores: int | None = 16
    tops_per_core: float | None = 6.25
    pe_rows: int | None = None
    pe_cols: int | None = None
    n_engines: int | None = None
    frequency_hz: float = 1.0e9
    mac_efficiency: float | None = None  # NPU array mac_efficiency; None → 1.0
    sku: str | None = None  # optional named SKU overrides cores/pe
    # Workload
    prompt_len: int = 512
    decode_seq_len: int = 512
    batch: int = 1
    weight_bits: int | None = None
    kv_bits: int | None = None
    act_bits: int | None = None
    dtype: str | None = None
    quant: str | None = None
    # Optional extras
    sram_mib: float = 64.0
    kv_fabric: str = "none"
    remote_kv_frac: float = 0.0
    c2c_gbps: float = 400.0
    c2c_hide: float = 0.0
    contention_mode: ContentionMode = "serialize"
    sram_policy: str = "weight_resident"
    weight_hide_factor: float = 0.0
    embed_policy: str = "replicate"
    # v0.31: LLM decode micro-batches through the PP pipeline (0 = auto min(B, pp));
    # video/protein (and LLM prefill) use max(1, decode_mb) microbatches.
    decode_mb: int = 0
    # Domain-specific workload knobs (video / protein); ignored for LLM
    n_denoise: int | None = None
    n_frames: int | None = None
    seq_len: int | None = None
    # v0.24 energy / cost stub knobs (ASSUMED; all None → stub off). See econ.py.
    tdp_w: float | None = None  # per-card average-power basis (W)
    watts_per_tops: float | None = None  # alt: W per peak TOPS (per card)
    power_util: float | None = None  # avg/peak power factor; None → 1.0 (TDP bound)
    cost_per_card_usd: float | None = None
    mem_addon_usd: float | None = None  # per-card memory package add-on
    usd_per_kwh: float | None = None  # LLM $/MTok energy part
    amortize_years: float | None = None  # LLM $/MTok capex part
    duty_cycle: float | None = None  # None → 1.0
    # v0.26 opt-in assumed knobs (defaults off / 1.0 → handcheck unchanged)
    non_gemm_overhead: float = 0.0  # fraction of GEMM compute; Softmax/RoPE/LN/misc
    softmax_frac: float | None = None  # finer; if any set, sum replaces coarse
    rope_frac: float | None = None
    norm_frac: float | None = None
    # map e.g. {"fp16":1.0,"fp8":2.0,"int8":2.0,"int4":4.0}; None/all-1.0 = bytes-only
    dtype_mac_factors: dict | None = None
    # v0.29 optimism fixes (ASSUMED knobs; see DECISIONS D-0.29-2/3)
    c2c_latency_us: float = DEFAULT_C2C_LATENCY_US  # per-collective α (µs)
    sync_overlap: float = DEFAULT_SYNC_OVERLAP  # 0 = fully exposed
    attn_parallel: str = "tp"  # tp | dp (attention data-parallel)
    # v0.31 MoE expert sharding: tp_ep (experts over ep, TP-split by tp) | ep_all
    moe_shard: str = "tp_ep"
    # v0.31 speculative decoding / MTP (spec_k = 0 → off; acceptance 「假设」)
    spec_k: int = 0
    spec_accept: float = 0.7
    spec_draft: str = "mtp"  # mtp (one extra layer + head per draft token) | model
    spec_draft_frac: float = 0.1  # draft="model": draft step / target step

    def __post_init__(self) -> None:
        if self.chip_count < 1:
            raise ValueError("chip_count must be >= 1")
        if self.decode_mb < 0:
            raise ValueError("decode_mb must be >= 0 (0 = auto)")
        if self.moe_shard not in ("tp_ep", "ep_all"):
            raise ValueError("moe_shard must be 'tp_ep' or 'ep_all'")
        if self.spec_draft not in ("mtp", "model"):
            raise ValueError("spec_draft must be 'mtp' or 'model'")


@dataclass
class MetricsCard:
    """Stable product metrics card from evaluate_workbench."""

    TTFT_ms: float
    TPOT_ms: float
    wall: str
    # breakdown (decode-centric; prefill noted in assumptions)
    t_compute_ms: float
    t_dram_ms: float
    t_c2c_ms: float
    t_fabric_ms: float
    t_bubble_ms: float
    bytes_W: int
    bytes_KV: int
    bytes_coll: int
    util: float
    oom: bool
    chips: int
    peak_tops: float
    mem_eff_GBps: float
    # identity / context
    model_id: str = ""
    mem_kind: str = ""
    tp: int = 1
    pp: int = 1
    ep: int = 1
    n_cores: int = 0
    tops_per_core: float = 0.0
    pe_rows: int = 0
    pe_cols: int = 0
    n_engines: int = 0
    sram_mib: float = 0.0
    prompt_len: int = 0
    decode_seq_len: int = 0
    batch: int = 1
    # Domain identity + video/protein metrics (LLM leaves these at defaults)
    domain: str = "llm"
    TTFC_ms: float = 0.0
    frames_per_s: float = 0.0
    time_per_seq_ms: float = 0.0
    pair_bytes: int = 0
    # v0.30: capacity in vendor/JEDEC GB (= GiB, 2^30 B) — same unit as the
    # memory catalog (HBM3E 12H×24Gb = 36 GB/stack) and the UI. The decimal
    # (10^9 B) values are kept in *_decimal for anyone who needs SI GB.
    capacity_needed_GB: float = 0.0
    capacity_GB: float = 0.0
    capacity_needed_GB_decimal: float = 0.0
    capacity_GB_decimal: float = 0.0
    n_denoise: int = 0
    n_frames: int = 0
    seq_len: int = 0
    # v0.24 energy / cost stub (ASSUMED from user knobs; 0 = knob not set)
    econ_configured: bool = False
    est_power_W: float = 0.0  # system (all chips), cards only
    est_power_per_card_W: float = 0.0
    est_energy_per_token_J: float = 0.0  # llm (decode)
    est_energy_prefill_J: float = 0.0  # llm (TTFT × P)
    est_energy_per_frame_J: float = 0.0  # video
    est_energy_per_seq_J: float = 0.0  # protein
    est_system_cost_usd: float = 0.0
    est_tokens_per_s: float = 0.0  # llm decode throughput (derived when econ on)
    est_energy_usd_per_Mtok: float = 0.0
    est_capex_usd_per_Mtok: float = 0.0
    est_usd_per_Mtok: float = 0.0
    econ_basis: str = ""
    # v0.25 scaling efficiency vs single-card baseline (same config, chips=tp=pp=ep=1)
    scale_efficiency: float = 1.0  # (t_single / t_multi) / chip_count; ideal 1.0
    speedup: float = 1.0  # t_single / t_multi
    t_single_primary_ms: float = 0.0  # baseline primary latency (ms)
    scale_metric: str = ""  # TPOT | TTFC | time_per_seq
    # v0.26 assumed knob echo (defaults preserve pre-v0.26)
    non_gemm_overhead: float = 0.0
    dtype_mac_factor: float = 1.0
    # v0.29 memory identity (structured catalog; empty for manual geometry)
    mem_type: str = ""
    mem_form: str = ""
    mem_summary: str = ""  # e.g. "LPDDR6 4×96b @10667 · 455 GB/s 可用 · 64 GB · 厂商量产"
    mem_id: str = ""
    mem_tag: str = ""  # weakest provenance tag (jedec … speculative)
    mem_tag_zh: str = ""
    mem_bus_bits: int = 0
    mem_raw_GBps: float = 0.0
    mem_payload_GBps: float = 0.0
    mem_capacity_nominal_GB: float = 0.0
    # v0.29 optimism fixes (exposed sync + honest KV sharding)
    t_sync_ms: float = 0.0  # decode / forward exposed collective latency
    n_sync_per_token: int = 0
    c2c_latency_us: float = 0.0
    sync_overlap: float = 0.0
    attn_parallel: str = "tp"
    kv_replication: float = 1.0
    # v0.31 engine: TP×EP sharding, PP decode micro-batches, spec decode, LM head
    moe_shard: str = ""
    decode_mb_eff: int = 1  # micro-batches in flight (PP decode)
    decode_microbatch: int = 0  # micro-batch size the decode step ran at
    TPOT_step_ms: float = 0.0  # time between verify/decode steps for one user
    spec_k: int = 0
    spec_accept: float = 0.0
    spec_draft: str = ""
    spec_tokens_per_step: float = 1.0  # E[accepted+bonus] = (1−a^(k+1))/(1−a)
    t_draft_ms: float = 0.0  # draft cost per pipeline tick
    bytes_head: int = 0  # LM head bytes read per decode step (per card)
    experts_read: float = 0.0  # expected local routed experts read per step
    # v0.31 per-phase stage components (ms, one card of the slowest stage) used
    # by the Pareto prefill-amortised goodput model; empty for video/protein.
    phase_detail: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def econ_lines(self) -> list[str]:
        if not self.econ_configured:
            return []
        if self.domain == "video":
            e = f"E/frame={self.est_energy_per_frame_J:.4g} J"
        elif self.domain == "protein":
            e = f"E/seq={self.est_energy_per_seq_J:.4g} J"
        else:
            e = (
                f"E/token={self.est_energy_per_token_J:.4g} J  "
                f"tok/s={self.est_tokens_per_s:.4g}"
            )
            if self.est_usd_per_Mtok > 0:
                e += (
                    f"  $/MTok={self.est_usd_per_Mtok:.4g} "
                    f"(energy {self.est_energy_usd_per_Mtok:.4g} + "
                    f"capex {self.est_capex_usd_per_Mtok:.4g})"
                )
        return [
            f"  econ [ASSUMED stub]: P={self.est_power_W:.4g} W  {e}  "
            f"system_cost=${self.est_system_cost_usd:,.0f}",
        ]

    def summary_lines(self) -> list[str]:
        notes = "; ".join(self.notes) if self.notes else ""
        lines = [
            f"=== MetricsCard | {self.domain} | {self.model_id} | chips={self.chips} "
            f"tp/pp/ep={self.tp}/{self.pp}/{self.ep} | {self.mem_kind} "
            f"| peak={self.peak_tops:.2f} TOPS | mem_eff={self.mem_eff_GBps:.1f} GB/s ===",
        ]
        if self.domain == "video":
            lines.append(
                f"  TTFC={self.TTFC_ms:.4f} ms  frames/s={self.frames_per_s:.4f}  "
                f"wall={self.wall}  util={self.util:.4f}  oom={self.oom}"
            )
            lines.append(
                f"  workload: F={self.n_frames} N_denoise={self.n_denoise} B={self.batch}  "
                f"cap_need={self.capacity_needed_GB:.3f} GB / cap={self.capacity_GB:.1f} GB"
            )
        elif self.domain == "protein":
            lines.append(
                f"  time/seq={self.time_per_seq_ms:.4f} ms  "
                f"wall={self.wall}  util={self.util:.4f}  oom={self.oom}"
            )
            lines.append(
                f"  workload: L={self.seq_len} B={self.batch}  "
                f"pair_bytes={self.pair_bytes}  "
                f"cap_need={self.capacity_needed_GB:.3f} GB / cap={self.capacity_GB:.1f} GB"
            )
        else:
            lines.append(
                f"  TTFT={self.TTFT_ms:.4f} ms  TPOT={self.TPOT_ms:.4f} ms/tok  "
                f"wall={self.wall}  util={self.util:.4f}  oom={self.oom}"
            )
            lines.append(
                f"  workload S={self.prompt_len}/ctx={self.decode_seq_len}/B={self.batch}"
            )
            if self.spec_k > 0 or self.decode_mb_eff > 1 or self.moe_shard:
                lines.append(
                    f"  v0.31: TPOT_step={self.TPOT_step_ms:.4f} ms  E[tok/step]={self.spec_tokens_per_step:.4f}"
                    f"  draft={self.t_draft_ms:.4f} ms  pp_mb={self.decode_mb_eff}×{self.decode_microbatch}"
                    f"  moe={self.moe_shard or '-'}  experts_read/card={self.experts_read:.2f}"
                )
        lines.extend(
            [
                f"  breakdown: compute={self.t_compute_ms:.4f} ms  "
                f"dram={self.t_dram_ms:.4f} ms  c2c={self.t_c2c_ms:.4f} ms  "
                f"sync={self.t_sync_ms:.4f} ms  "
                f"fabric={self.t_fabric_ms:.4f} ms  bubble={self.t_bubble_ms:.4f} ms",
                f"  bytes: W={self.bytes_W}  KV={self.bytes_KV}  coll={self.bytes_coll}",
                f"  npu: cores={self.n_cores} × {self.tops_per_core:.4g} T/core "
                f"→ PE={self.pe_rows}x{self.pe_cols}×{self.n_engines}  "
                f"SRAM={self.sram_mib:.1f} MiB",
            ]
        )
        lines.extend(self.econ_lines())
        if self.scale_metric:
            lines.append(
                f"  scale: eff={self.scale_efficiency:.4f}  speedup={self.speedup:.4f}×  "
                f"(metric={self.scale_metric}  t_single={self.t_single_primary_ms:.4f} ms)"
            )
        if notes:
            lines.append(f"  notes: {notes}")
        if self.assumptions:
            lines.append("  assumptions:")
            for a in self.assumptions:
                lines.append(f"    - {a}")
        return lines

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = "|".join(self.notes)
        d["assumptions"] = "|".join(self.assumptions)
        return d

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (lists kept as lists)."""
        d = asdict(self)
        return d

    def to_markdown(self) -> str:
        """Markdown document for MetricsCard."""
        lines = [
            f"# MetricsCard — `{self.model_id}`",
            "",
            f"| field | value |",
            f"|---|---|",
            f"| chips | {self.chips} |",
            f"| tp / pp / ep | {self.tp} / {self.pp} / {self.ep} |",
            f"| mem | {self.mem_summary or self.mem_kind} ({self.mem_eff_GBps:.1f} GB/s eff) |",
            f"| mem raw / payload GB/s | {self.mem_raw_GBps:.1f} / {self.mem_payload_GBps:.1f} |",
            f"| mem tag | {self.mem_tag} |",
            f"| peak TOPS | {self.peak_tops:.2f} |",
            f"| domain | {self.domain} |",
            f"| TTFT_ms | {self.TTFT_ms:.4f} |",
            f"| TPOT_ms | {self.TPOT_ms:.4f} |",
            f"| TTFC_ms | {self.TTFC_ms:.4f} |",
            f"| frames_per_s | {self.frames_per_s:.4f} |",
            f"| time_per_seq_ms | {self.time_per_seq_ms:.4f} |",
            f"| pair_bytes | {self.pair_bytes} |",
            f"| capacity_needed_GB | {self.capacity_needed_GB:.4f} |",
            f"| capacity_GB | {self.capacity_GB:.4f} |",
            f"| wall | {self.wall} |",
            f"| util | {self.util:.4f} |",
            f"| oom | {self.oom} |",
            f"| t_compute_ms | {self.t_compute_ms:.4f} |",
            f"| t_dram_ms | {self.t_dram_ms:.4f} |",
            f"| t_c2c_ms | {self.t_c2c_ms:.4f} |",
            f"| t_sync_ms (exposed, α={self.c2c_latency_us:g} µs × {self.n_sync_per_token}) | {self.t_sync_ms:.4f} |",
            f"| attn_parallel / kv_replication | {self.attn_parallel} / {self.kv_replication:g} |",
            f"| moe_shard / decode_mb_eff (micro-batch) | {self.moe_shard or '-'} / {self.decode_mb_eff} ({self.decode_microbatch}) |",
            f"| spec_k / accept / E[tok/step] | {self.spec_k} / {self.spec_accept:g} / {self.spec_tokens_per_step:.4f} |",
            f"| TPOT_step_ms / t_draft_ms | {self.TPOT_step_ms:.4f} / {self.t_draft_ms:.4f} |",
            f"| t_fabric_ms | {self.t_fabric_ms:.4f} |",
            f"| t_bubble_ms | {self.t_bubble_ms:.4f} |",
            f"| bytes_W | {self.bytes_W} |",
            f"| bytes_KV | {self.bytes_KV} |",
            f"| bytes_coll | {self.bytes_coll} |",
            f"| n_cores × tops/core | {self.n_cores} × {self.tops_per_core:.4g} |",
            f"| PE | {self.pe_rows}x{self.pe_cols}×{self.n_engines} |",
            f"| SRAM_MiB | {self.sram_mib:.1f} |",
            f"| prompt / ctx / batch | {self.prompt_len} / {self.decode_seq_len} / {self.batch} |",
        ]
        if self.econ_configured:
            lines.extend(
                [
                    f"| est_power_W (assumed) | {self.est_power_W:.4g} |",
                    f"| est_energy_per_token_J (assumed) | {self.est_energy_per_token_J:.6g} |",
                    f"| est_energy_per_frame_J (assumed) | {self.est_energy_per_frame_J:.6g} |",
                    f"| est_energy_per_seq_J (assumed) | {self.est_energy_per_seq_J:.6g} |",
                    f"| est_system_cost_usd (assumed) | {self.est_system_cost_usd:.2f} |",
                    f"| est_usd_per_Mtok (assumed) | {self.est_usd_per_Mtok:.6g} |",
                    f"| econ_basis | {self.econ_basis} |",
                ]
            )
        lines.extend(
            [
                f"| scale_efficiency | {self.scale_efficiency:.6g} |",
            f"| non_gemm_overhead | {self.non_gemm_overhead:.6g} |",
            f"| dtype_mac_factor | {self.dtype_mac_factor:.6g} |",
                f"| speedup | {self.speedup:.6g} |",
                f"| t_single_primary_ms | {self.t_single_primary_ms:.4f} |",
                f"| scale_metric | {self.scale_metric} |",
            ]
        )
        lines.append("")
        if self.notes:
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        if self.assumptions:
            lines.append("## Assumptions")
            for a in self.assumptions:
                lines.append(f"- {a}")
            lines.append("")
        lines.append("```")
        lines.extend(self.summary_lines())
        lines.append("```")
        lines.append("")
        return "\n".join(lines)


def resolve_parallel(
    chip_count: int,
    parallel: ParallelOverride | None,
) -> tuple[int, int, int]:
    """Default: tp=chip_count, pp=ep=1. Explicit must satisfy product == chips."""
    if parallel is None:
        return int(chip_count), 1, 1
    tp, pp, ep = int(parallel.tp), int(parallel.pp), int(parallel.ep)
    if tp * pp * ep != chip_count:
        raise ValueError(
            f"parallel tp*pp*ep={tp}*{pp}*{ep}={tp*pp*ep} != chip_count={chip_count}"
        )
    return tp, pp, ep


def build_npu(cfg: WorkbenchConfig) -> tuple[NPUConfig, float, list[str]]:
    """Return (npu, frequency_hz, assumption notes)."""
    notes: list[str] = []
    freq = cfg.frequency_hz
    if cfg.sku:
        sku = get_sku(cfg.sku)
        notes.append(f"NPU from named SKU {sku.name} (overrides cores/pe) [assumed]")
        npu = sku.npu()
        freq = sku.frequency_hz
    elif cfg.pe_rows is not None and cfg.pe_cols is not None:
        eng = cfg.n_engines if cfg.n_engines is not None else (
            cfg.n_cores if cfg.n_cores is not None else 1
        )
        npu = npu_from_pe(cfg.pe_rows, cfg.pe_cols, eng)
        notes.append(
            f"NPU from explicit PE {cfg.pe_rows}x{cfg.pe_cols}×{eng} [assumed]"
        )
    elif cfg.n_cores is not None and cfg.tops_per_core is not None:
        npu = npu_from_cores(
            cfg.n_cores, cfg.tops_per_core, frequency_hz=freq
        )
        notes.append(
            "NPU cores×tops_per_core → n_engines=n_cores, near-square PE/core "
            f"({cfg.n_cores}×{cfg.tops_per_core:g} T → "
            f"{npu.rows}x{npu.cols}×{npu.n_engines}) [assumed mapping]"
        )
    else:
        # Fallback: sku_100t-equivalent product language
        npu = npu_from_cores(16, 6.25, frequency_hz=freq)
        notes.append(
            "NPU default cores=16 × 6.25 T/core (~100 TOPS class, assumed)"
        )
    if cfg.mac_efficiency is not None:
        npu = replace(npu, mac_efficiency=float(cfg.mac_efficiency))
        notes.append(
            f"NPU mac_efficiency={cfg.mac_efficiency:g} [assumed/override]"
        )
    return npu, freq, notes


def build_mem_spec(cfg: WorkbenchConfig) -> MemSpec | None:
    """Structured MemSpec when ``cfg.mem_type`` is set (else None)."""
    if not cfg.mem_type:
        return None
    return make_spec(
        cfg.mem_type,
        form=cfg.mem_form,
        width_bits=cfg.mem_width_bits,
        rate=cfg.mem_rate_MTps,
        count=cfg.mem_count,
        cap_GB=cfg.mem_cap_GB,
        height=cfg.hbm_height,
        die_Gb=cfg.hbm_die_Gb,
        efficiency=cfg.efficiency if cfg.efficiency is not None else 0.70,
    )


def build_memory(cfg: WorkbenchConfig) -> tuple[ExternalMemory, list[str]]:
    notes: list[str] = []
    spec = build_mem_spec(cfg)
    if spec is not None:
        mem = spec.to_external_memory()
        notes.append(
            f"DRAM {spec.summary()} [rates/caps per catalog tag; efficiency assumed]"
        )
        if cfg.mem_legacy_id and spec.legacy_note:
            notes.append(spec.legacy_note)
        for w in spec.warnings():
            notes.append(f"mem warning: {w}")
        return mem, notes
    kind = str(cfg.mem_kind).upper()
    overrides = any(
        v is not None
        for v in (
            cfg.n_packages,
            cfg.n_ranks,
            cfg.n_channels,
            cfg.data_rate_GTs,
            cfg.width_bits,
            cfg.efficiency,
            cfg.capacity_GB,
        )
    )
    if overrides:
        mem = memory_from_geometry(
            kind,
            n_channels=cfg.n_channels,
            n_packages=cfg.n_packages,
            n_ranks=cfg.n_ranks,
            data_rate_GTs=cfg.data_rate_GTs,
            width_bits=cfg.width_bits,
            efficiency=cfg.efficiency,
            capacity_GB=cfg.capacity_GB,
        )
        notes.append(
            f"DRAM from geometry knobs on {kind} preset defaults [assumed]"
        )
    else:
        mem = HBM_PRESET if kind == "HBM" else LPDDR_PRESET
        notes.append(f"DRAM preset {mem.summary()} [assumed]")
    return mem, notes


def _mem_card_fields(cfg: WorkbenchConfig, mem: ExternalMemory) -> dict[str, Any]:
    """MetricsCard memory identity fields (structured spec or preset/manual)."""
    spec = build_mem_spec(cfg)
    if spec is None and mem.mem_type and mem in (HBM_PRESET, LPDDR_PRESET):
        spec = default_spec_for_preset(mem)
    if spec is not None:
        return {
            "mem_type": spec.mem_type,
            "mem_form": spec.form,
            "mem_summary": spec.short_label(),
            "mem_id": spec.id,
            "mem_tag": spec.tag,
            "mem_tag_zh": spec.tag_zh,
            "mem_bus_bits": spec.bus_bits,
            "mem_raw_GBps": spec.raw_GBps,
            "mem_payload_GBps": spec.payload_GBps,
            "mem_capacity_nominal_GB": spec.capacity_GB,
        }
    raw = mem.peak_bandwidth_Bps() / 1e9
    return {
        "mem_type": "",
        "mem_form": "manual",
        "mem_summary": (
            f"{mem.kind} 手动几何 {mem.n_channels}×{mem.width_bits}b @"
            f"{mem.data_rate_gt_s*1000:.0f} · {raw:.0f} GB/s · "
            f"{mem.capacity_bytes/1e9:.0f} GB · 推测"
        ),
        "mem_id": "",
        "mem_tag": "speculative",
        "mem_tag_zh": "推测",
        "mem_bus_bits": mem.n_channels * mem.width_bits,
        "mem_raw_GBps": raw,
        "mem_payload_GBps": mem.payload_bandwidth_Bps() / 1e9,
        "mem_capacity_nominal_GB": mem.capacity_bytes / 1e9,
    }


def default_spec_for_preset(mem: ExternalMemory) -> MemSpec | None:
    if mem is HBM_PRESET or mem == HBM_PRESET:
        return make_spec("HBM3E", count=8, height=12, die_Gb=24, rate=9200)
    if mem is LPDDR_PRESET or mem == LPDDR_PRESET:
        return make_spec("LPDDR5X", count=8, width_bits=64, rate=8533, cap_GB=16)
    return None



def _resolved_dtype_mac_factor(
    cfg: WorkbenchConfig,
    *,
    weight_bits: int,
    dtype_name: str | None = None,
) -> float:
    """Resolve assumed dtype MAC peak factor; default 1.0."""
    return resolve_dtype_mac_factor(
        cfg.dtype_mac_factors,
        dtype_name=dtype_name or cfg.dtype,
        weight_bits=weight_bits,
    )


def _make_eval_config(
    cfg: WorkbenchConfig,
    freq: float,
    *,
    weight_bits: int,
    dtype_name: str | None = None,
    prompt_len: int | None = None,
    decode_seq_len: int | None = None,
    batch: int | None = None,
) -> tuple[EvalConfig, float]:
    """Build EvalConfig with v0.26 assumed knobs; return (eval_cfg, mac_factor)."""
    mac_f = _resolved_dtype_mac_factor(cfg, weight_bits=weight_bits, dtype_name=dtype_name)
    ec = EvalConfig(
        prompt_len=cfg.prompt_len if prompt_len is None else prompt_len,
        decode_seq_len=cfg.decode_seq_len if decode_seq_len is None else decode_seq_len,
        batch=cfg.batch if batch is None else batch,
        frequency_hz=freq,
        contention_mode=cfg.contention_mode,
        weight_hide_factor=cfg.weight_hide_factor,
        sram_policy=cfg.sram_policy,  # type: ignore[arg-type]
        non_gemm_overhead=float(cfg.non_gemm_overhead),
        softmax_frac=cfg.softmax_frac,
        rope_frac=cfg.rope_frac,
        norm_frac=cfg.norm_frac,
        dtype_mac_factor=mac_f,
    )
    return ec, mac_f


def _append_assumed_compute_notes(
    assumptions: list[str],
    notes: list[str],
    eval_cfg: EvalConfig,
    mac_f: float,
    *,
    dtype_mac_factors: dict | None,
) -> None:
    """Honesty notes for non-GEMM overhead / dtype MAC factor (assumed)."""
    oh = resolved_non_gemm_overhead(eval_cfg)
    if oh > 0.0:
        if any(
            v is not None
            for v in (eval_cfg.softmax_frac, eval_cfg.rope_frac, eval_cfg.norm_frac)
        ):
            detail = (
                f"softmax={eval_cfg.softmax_frac or 0:g} "
                f"rope={eval_cfg.rope_frac or 0:g} "
                f"norm={eval_cfg.norm_frac or 0:g}"
            )
            assumptions.append(
                f"non_gemm_overhead={oh:g} (finer {detail}) "
                f"[ASSUMED coarse Softmax/RoPE/LN; not cycle-accurate; "
                f"t_compute*=(1+oh)]"
            )
        else:
            assumptions.append(
                f"non_gemm_overhead={oh:g} [ASSUMED coarse Softmax/RoPE/LN/misc; "
                f"not cycle-accurate; t_compute*=(1+oh); default 0=off]"
            )
        notes.append(f"non_gemm_overhead={oh:g} [assumed]")
    if abs(mac_f - 1.0) > 1e-15 or dtype_mac_factors_active(dtype_mac_factors):
        assumptions.append(
            f"dtype_mac_factor={mac_f:g} [ASSUMED user table — not silicon; "
            f"peak_TOPS*=factor, t_compute/=factor; default all 1.0=bytes-only]"
        )
        if abs(mac_f - 1.0) > 1e-15:
            notes.append(f"dtype_mac_factor={mac_f:g} [assumed]")


def _attn_parallel(v: str | None) -> str:
    a = (v or "tp").strip().lower()
    if a in ("tp", "head", "heads"):
        return "tp"
    if a in ("dp", "attn_dp", "attention_dp", "data"):
        return "dp"
    raise ValueError(f"attn_parallel must be 'tp' or 'dp', got {v!r}")


def _sync_assumption(cfg: WorkbenchConfig, *, tp: int, pp: int, ep: int) -> str:
    if tp * pp * ep <= 1 or cfg.c2c_latency_us <= 0:
        return ""
    return (
        f"exposed sync: α={cfg.c2c_latency_us:g} µs per collective × "
        f"(TP 2 all-reduce/layer, EP 2 all-to-all/layer, PP 1 send/stage) × "
        f"(1−overlap {cfg.sync_overlap:g}) added outside max(compute, mem, c2c) "
        f"[ASSUMED α; v0.29]"
    )


def _fabric_mode(name: str) -> str:
    n = (name or "none").strip().lower()
    if n in ("none", "", "off"):
        return "none"
    if n.startswith("roce") or n == "roce_v2":
        return "roce_v2"
    if n.startswith("ib"):
        return "ib"
    # preset name → mode via get_kv_fabric
    link = get_kv_fabric(n)
    return link.mode if link else "none"


def _fabric_gbps(name: str, default: float = 200.0) -> float:
    n = (name or "none").strip().lower()
    if n in ("none", "", "off"):
        return default
    try:
        link = get_kv_fabric(n)
        if link:
            return link.link_gbps
    except KeyError:
        pass
    return default


def _common_hw(
    cfg: WorkbenchConfig,
) -> tuple[Any, float, Any, SRAMConfig, list[str], list[str]]:
    """Shared NPU + memory + SRAM build for all domains."""
    assumptions: list[str] = []
    notes: list[str] = []
    npu, freq, npu_notes = build_npu(cfg)
    assumptions.extend(npu_notes)
    mem, mem_notes = build_memory(cfg)
    assumptions.extend(mem_notes)
    sram = SRAMConfig(int(cfg.sram_mib * 1024 * 1024))
    return npu, freq, mem, sram, assumptions, notes


def _apply_video_knobs(shape: VideoShape, cfg: WorkbenchConfig, notes: list[str]) -> VideoShape:
    from dataclasses import replace as dc_replace

    kw: dict[str, Any] = {}
    if cfg.n_denoise is not None:
        kw["n_denoise"] = int(cfg.n_denoise)
    if cfg.n_frames is not None:
        kw["n_frames"] = int(cfg.n_frames)
    if cfg.batch is not None and cfg.batch != shape.batch:
        kw["batch"] = int(cfg.batch)
    if cfg.weight_bits is not None:
        kw["weight_bits"] = int(cfg.weight_bits)
    if cfg.act_bits is not None:
        kw["act_bits"] = int(cfg.act_bits)
    if kw:
        shape = dc_replace(shape, **kw)
        notes.append(
            "video knobs: "
            + ", ".join(f"{k}={v}" for k, v in kw.items())
        )
    return shape


def _apply_protein_knobs(
    shape: ProteinShape, cfg: WorkbenchConfig, notes: list[str]
) -> ProteinShape:
    from dataclasses import replace as dc_replace

    kw: dict[str, Any] = {}
    if cfg.seq_len is not None:
        kw["seq_len"] = int(cfg.seq_len)
    if cfg.batch is not None and cfg.batch != shape.batch:
        kw["batch"] = int(cfg.batch)
    if cfg.weight_bits is not None:
        kw["weight_bits"] = int(cfg.weight_bits)
    if cfg.act_bits is not None:
        kw["act_bits"] = int(cfg.act_bits)
    if kw:
        shape = dc_replace(shape, **kw)
        notes.append(
            "protein knobs: "
            + ", ".join(f"{k}={v}" for k, v in kw.items())
        )
    return shape


def llm_shape_for_config(cfg: WorkbenchConfig, notes: list[str] | None = None) -> ModelShape:
    """Resolve the LLM ModelShape incl. dtype / quant / bit overrides (v0.30: shared
    by the evaluator and the pareto capacity search)."""
    notes = notes if notes is not None else []
    shape = resolve_llm_shape(cfg.model_id)
    if cfg.dtype:
        shape = get_dtype(cfg.dtype).apply(shape)
        notes.append(f"dtype={cfg.dtype} (bytes-only)")
    if cfg.quant:
        shape = get_quant(cfg.quant).apply(shape)
        notes.append(f"quant={cfg.quant} (bytes-only)")
    if cfg.weight_bits is not None or cfg.kv_bits is not None or cfg.act_bits is not None:
        shape = shape.with_bits(
            weight_bits=cfg.weight_bits,
            kv_bits=cfg.kv_bits,
            act_bits=cfg.act_bits,
        )
        notes.append(
            f"bits W/KV/A="
            f"{shape.weight_bits}/{shape.kv_bits}/{shape.act_bits}"
        )
    return shape

def _evaluate_workbench_llm(cfg: WorkbenchConfig) -> MetricsCard:
    """LLM path: evaluate_scaleup → MetricsCard (unchanged physics)."""
    assumptions: list[str] = []
    notes: list[str] = []

    series = get_series(cfg.model_id)
    assumptions.append(series.metadata)
    shape = llm_shape_for_config(cfg, notes)

    tp, pp, ep = resolve_parallel(cfg.chip_count, cfg.parallel)
    if cfg.parallel is None:
        assumptions.append(
            f"chip_count={cfg.chip_count} → tp={tp}, pp=1, ep=1 (default mapping)"
        )
    else:
        assumptions.append(
            f"explicit parallel tp/pp/ep={tp}/{pp}/{ep} "
            f"(product==chip_count={cfg.chip_count})"
        )

    npu, freq, mem, sram, hw_a, hw_n = _common_hw(cfg)
    assumptions.extend(hw_a)
    notes.extend(hw_n)

    eval_cfg, mac_f = _make_eval_config(
        cfg, freq, weight_bits=shape.weight_bits, dtype_name=cfg.dtype,
    )
    # v0.31: product path counts the LM head (V·H read + GEMM) every step
    eval_cfg = replace(eval_cfg, count_lm_head=True)
    _append_assumed_compute_notes(
        assumptions, notes, eval_cfg, mac_f, dtype_mac_factors=cfg.dtype_mac_factors,
    )

    fab_mode = _fabric_mode(cfg.kv_fabric)
    fab_gbps = _fabric_gbps(cfg.kv_fabric)
    tp_cfg = TPConfig(
        tp=tp,
        pp=pp,
        ep=ep,
        mb=max(int(cfg.decode_mb), 1),
        decode_mb=int(cfg.decode_mb),
        moe_shard=str(cfg.moe_shard or "tp_ep"),  # type: ignore[arg-type]
        spec_k=int(cfg.spec_k),
        spec_accept=float(cfg.spec_accept),
        spec_draft=str(cfg.spec_draft or "mtp"),  # type: ignore[arg-type]
        spec_draft_frac=float(cfg.spec_draft_frac),
        c2c=get_c2c(cfg.c2c_gbps),
        c2c_hide=cfg.c2c_hide,
        kv_fabric=fab_mode,  # type: ignore[arg-type]
        fabric_gbps=fab_gbps,
        remote_kv_frac=cfg.remote_kv_frac,
        embed_policy=cfg.embed_policy,  # type: ignore[arg-type]
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        attn_parallel=_attn_parallel(cfg.attn_parallel),  # type: ignore[arg-type]
    )
    if dec_sync_note := _sync_assumption(cfg, tp=tp, pp=pp, ep=ep):
        assumptions.append(dec_sync_note)
    if fab_mode != "none":
        assumptions.append(
            f"kv_fabric={cfg.kv_fabric} mode={fab_mode} "
            f"remote_frac={cfg.remote_kv_frac} [assumed BW]"
        )

    result = evaluate_scaleup(shape, npu, sram, mem, eval_cfg, tp_cfg)

    dec = result.decode
    pre = result.prefill
    t_tick = dec.t_tick_s or dec.t_stage_s
    t_roof = dec.t_roofline_s
    t_bubble = max(t_roof - t_tick, 0.0)
    e_tok = result.spec_tokens_per_step
    if shape.is_moe and (ep > 1 or tp_cfg.moe_shard == "ep_all"):
        assumptions.append(
            f"MoE sharding {tp_cfg.moe_shard}: "
            + ("experts over ep groups, each expert TP-split by tp; " if tp_cfg.moe_shard == "tp_ep"
               else "experts over all tp·ep ranks (unsplit); ")
            + "attention data-parallel across ep groups (KV split by batch over ep) [v0.31]"
        )
    if shape.is_moe:
        assumptions.append(
            f"MoE decode weight stream: expected touched experts for {dec.moe_tokens:g} tokens/step "
            f"= {dec.experts_read_card:.2f} local routed experts/card (uniform routing) [assumed]"
        )
    if dec.head_bytes_card > 0:
        assumptions.append(
            f"LM head counted every decode step: {dec.head_bytes_card / 1e9:.3f} GB/card read + GEMM "
            "(vocab-parallel under attention TP; replicated head read in full per rank under "
            "attention DP; PP stages balanced around the head → amortised /pp) [v0.31, assumed]"
        )
    if pp > 1 and cfg.batch > 1:
        assumptions.append(
            f"PP decode: {dec.mb_eff} micro-batch(es) of {dec.batch_eff} in flight; "
            f"TPOT_step = max(mb, pp) × tick [v0.31]"
        )
    if tp_cfg.spec_k > 0:
        assumptions.append(
            f"speculative decoding k={tp_cfg.spec_k} accept={tp_cfg.spec_accept:g} draft={tp_cfg.spec_draft}: "
            f"verify M=B×{tp_cfg.spec_k + 1}; E[tokens/step]={e_tok:.3f}; "
            f"TPOT = step / E [assumed acceptance]"
        )
    if dec.kv_note:
        notes.append(f"KV 复制 ×{dec.kv_replication:g}: {dec.kv_note}")
    if tp_cfg.attn_parallel == "dp" and tp > 1:
        assumptions.append(
            "attn_parallel=dp: KV partitioned by batch across the TP group; "
            "attention weights replicated per rank (capacity + decode W stream); "
            "attention compute split by batch ≈ same FLOPs/card; DP↔TP re-shard "
            "collectives approximated by the same 2/layer count [assumed]"
        )

    n_cores, tpc = cores_from_npu(npu, freq)
    peak = npu.peak_tops_at(freq) * mac_f
    mem_eff = mem.effective_bandwidth_Bps() / 1e9

    phase_detail = {
        "decode": {
            "compute_ms": dec.t_compute_s * 1e3, "mem_ms": dec.t_memory_s * 1e3,
            "c2c_ms": dec.t_c2c_s * 1e3, "pp_act_ms": dec.t_pp_act_s * 1e3,
            "a2a_ms": dec.t_a2a_s * 1e3, "fabric_ms": dec.t_fabric_s * 1e3,
            "sync_ms": dec.t_sync_s * 1e3, "draft_ms": dec.t_draft_s * 1e3,
            "stage_ms": dec.t_stage_s * 1e3, "tick_ms": t_tick * 1e3,
            "mb": dec.mb_eff, "microbatch": dec.batch_eff, "pp": pp,
            "slots": max(dec.mb_eff, pp) if pp > 1 else 1,
            "tokens_per_step": e_tok,
        },
        "prefill": {
            "compute_ms": pre.t_compute_s * 1e3,
            "mem_nonweight_ms": (pre.contention.t_kv_s + pre.contention.t_act_s) * 1e3,
            "mem_ms": pre.t_memory_s * 1e3,
            "c2c_ms": pre.t_c2c_s * 1e3, "pp_act_ms": pre.t_pp_act_s * 1e3,
            "a2a_ms": pre.t_a2a_s * 1e3, "sync_ms": pre.t_sync_s * 1e3,
            "stage_ms": pre.t_stage_s * 1e3, "ttft_ms": result.ttft_s * 1e3,
            "batch": cfg.batch,
        },
    }

    return MetricsCard(
        TTFT_ms=result.ttft_s * 1e3,
        TPOT_ms=result.tpot_s * 1e3,
        wall=dec.wall,
        t_compute_ms=dec.t_compute_s * 1e3,
        t_dram_ms=dec.t_memory_s * 1e3,
        t_c2c_ms=(dec.t_c2c_s + dec.t_pp_act_s + dec.t_a2a_s) * 1e3,
        t_fabric_ms=dec.t_fabric_s * 1e3,
        t_bubble_ms=t_bubble * 1e3,
        bytes_W=dec.weight_bytes_card,
        bytes_KV=dec.kv_bytes_card,
        bytes_coll=dec.collective_bytes + dec.a2a_bytes + dec.pp_act_bytes,
        util=result.single_card.decode.traffic.mean_gemm_util,
        oom=result.oom,
        chips=cfg.chip_count,
        peak_tops=peak,
        mem_eff_GBps=mem_eff,
        model_id=cfg.model_id,
        mem_kind=str(mem.kind),
        tp=tp,
        pp=pp,
        ep=ep,
        n_cores=n_cores,
        tops_per_core=tpc,
        pe_rows=npu.rows,
        pe_cols=npu.cols,
        n_engines=npu.n_engines,
        sram_mib=cfg.sram_mib,
        prompt_len=cfg.prompt_len,
        decode_seq_len=cfg.decode_seq_len,
        batch=cfg.batch,
        domain="llm",
        capacity_GB=mem.capacity_bytes / GIB,
        capacity_GB_decimal=mem.capacity_bytes / 1e9,
        # v0.30: LLM per-card W + KV (was not exported before; UI showed cap only)
        capacity_needed_GB=result.per_card_bytes / GIB,
        capacity_needed_GB_decimal=result.per_card_bytes / 1e9,
        non_gemm_overhead=resolved_non_gemm_overhead(eval_cfg),
        dtype_mac_factor=mac_f,
        t_sync_ms=dec.t_sync_s * 1e3,
        n_sync_per_token=dec.n_sync,
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        attn_parallel=tp_cfg.attn_parallel,
        kv_replication=dec.kv_replication,
        moe_shard=tp_cfg.moe_shard if shape.is_moe else "",
        decode_mb_eff=dec.mb_eff,
        decode_microbatch=dec.batch_eff,
        TPOT_step_ms=result.tpot_step_s * 1e3,
        spec_k=tp_cfg.spec_k,
        spec_accept=tp_cfg.spec_accept if tp_cfg.spec_k > 0 else 0.0,
        spec_draft=tp_cfg.spec_draft if tp_cfg.spec_k > 0 else "",
        spec_tokens_per_step=e_tok,
        t_draft_ms=dec.t_draft_s * 1e3,
        bytes_head=dec.head_bytes_card,
        experts_read=dec.experts_read_card,
        phase_detail=phase_detail,
        **_mem_card_fields(cfg, mem),
        notes=notes,
        assumptions=assumptions,
    )


def _evaluate_workbench_video(cfg: WorkbenchConfig) -> MetricsCard:
    """Video path: evaluate_video + TP/PP/EP domain scale-up → MetricsCard."""
    assumptions: list[str] = []
    notes: list[str] = []
    series = get_series(cfg.model_id)
    shape = resolve_video_shape(cfg.model_id)
    assumptions.append(series.metadata)
    shape = _apply_video_knobs(shape, cfg, notes)
    if cfg.dtype or cfg.quant:
        notes.append("dtype/quant ignored for video (use weight_bits/act_bits)")

    tp, pp, ep = resolve_parallel(cfg.chip_count, cfg.parallel)
    if cfg.parallel is None:
        assumptions.append(
            f"chip_count={cfg.chip_count} → tp={tp}, pp=1, ep=1 (default mapping)"
        )
    else:
        assumptions.append(
            f"explicit parallel tp/pp/ep={tp}/{pp}/{ep} "
            f"(product==chip_count={cfg.chip_count})"
        )

    npu, freq, mem, sram, hw_a, hw_n = _common_hw(cfg)
    assumptions.extend(hw_a)
    notes.extend(hw_n)

    eval_cfg, mac_f = _make_eval_config(
        cfg, freq, weight_bits=shape.weight_bits, dtype_name=cfg.dtype,
        batch=cfg.batch,
    )
    _append_assumed_compute_notes(
        assumptions, notes, eval_cfg, mac_f, dtype_mac_factors=cfg.dtype_mac_factors,
    )
    result = evaluate_video(shape, npu, sram, mem, eval_cfg)
    # Residual-stream activation volume for collectives: B · T · H · act_bytes
    act_vol = (
        shape.batch * shape.n_tokens * shape.hidden * (shape.act_bits // 8)
    )
    scaled = scale_domain_parallel(
        result.forward,
        mem=mem,
        cfg=eval_cfg,
        tp=tp,
        pp=pp,
        ep=ep,
        mb=max(int(cfg.decode_mb), 1),
        act_volume_bytes=act_vol,
        n_layers=shape.n_layers,
        capacity_weight_bytes=shape.weight_bytes(),
        capacity_other_bytes=shape.act_bytes_working(),
        shard_other=False,
        c2c=get_c2c(cfg.c2c_gbps),
        c2c_hide=cfg.c2c_hide,
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        has_moe_experts=False,
        phase="denoise",
        domain="video",
    )
    assumptions.extend(scaled.assumptions)

    n_cores, tpc = cores_from_npu(npu, freq)
    peak = npu.peak_tops_at(freq) * mac_f
    mem_eff = mem.effective_bandwidth_Bps() / 1e9
    # TTFC = N_denoise × scaled forward wall (includes PP bubble)
    t_fwd = scaled.t_roofline_s
    ttfc_s = shape.n_denoise * t_fwd
    ttfc_ms = ttfc_s * 1e3
    t_frame_ms = (ttfc_s / max(shape.n_frames, 1)) * 1e3
    frames_per_s = shape.n_frames / ttfc_s if ttfc_s > 0 else float("inf")
    t_bubble_ms = (scaled.t_roofline_s - scaled.t_stage_s) * 1e3 if scaled.bubble_frac > 0 else 0.0
    t_c2c_ms = (scaled.t_c2c_s + scaled.t_pp_act_s + scaled.t_a2a_s) * 1e3

    return MetricsCard(
        TTFT_ms=ttfc_ms,  # alias: TTFC as TTFT-analogue for shared UI slots
        TPOT_ms=t_frame_ms,  # alias: time/frame as TPOT-analogue
        wall=scaled.wall,
        t_compute_ms=scaled.t_compute_s * 1e3,
        t_dram_ms=scaled.t_memory_s * 1e3,
        t_c2c_ms=t_c2c_ms,
        t_fabric_ms=0.0,
        t_bubble_ms=t_bubble_ms,
        bytes_W=scaled.weight_bytes_card,
        bytes_KV=scaled.kv_bytes_card,
        bytes_coll=scaled.collective_bytes + scaled.pp_act_bytes + scaled.a2a_bytes,
        util=scaled.mean_gemm_util,
        oom=scaled.oom,
        chips=cfg.chip_count,
        peak_tops=peak,
        mem_eff_GBps=mem_eff,
        model_id=cfg.model_id,
        mem_kind=str(mem.kind),
        tp=tp,
        pp=pp,
        ep=ep,
        n_cores=n_cores,
        tops_per_core=tpc,
        pe_rows=npu.rows,
        pe_cols=npu.cols,
        n_engines=npu.n_engines,
        sram_mib=cfg.sram_mib,
        prompt_len=0,
        decode_seq_len=0,
        batch=shape.batch,
        domain="video",
        TTFC_ms=ttfc_ms,
        frames_per_s=frames_per_s,
        capacity_needed_GB=scaled.capacity_needed_bytes / GIB,
        capacity_needed_GB_decimal=scaled.capacity_needed_bytes / 1e9,
        capacity_GB=mem.capacity_bytes / GIB,
        capacity_GB_decimal=mem.capacity_bytes / 1e9,
        n_denoise=shape.n_denoise,
        n_frames=shape.n_frames,
        non_gemm_overhead=resolved_non_gemm_overhead(eval_cfg),
        dtype_mac_factor=mac_f,
        t_sync_ms=scaled.t_sync_s * 1e3,
        n_sync_per_token=scaled.n_sync,
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        **_mem_card_fields(cfg, mem),
        notes=notes,
        assumptions=assumptions,
    )


def _evaluate_workbench_protein(cfg: WorkbenchConfig) -> MetricsCard:
    """Protein path: evaluate_protein + TP/PP/EP domain scale-up → MetricsCard."""
    assumptions: list[str] = []
    notes: list[str] = []
    series = get_series(cfg.model_id)
    shape = resolve_protein_shape(cfg.model_id)
    assumptions.append(series.metadata)
    shape = _apply_protein_knobs(shape, cfg, notes)
    if cfg.dtype or cfg.quant:
        notes.append("dtype/quant ignored for protein (use weight_bits/act_bits)")

    tp, pp, ep = resolve_parallel(cfg.chip_count, cfg.parallel)
    if cfg.parallel is None:
        assumptions.append(
            f"chip_count={cfg.chip_count} → tp={tp}, pp=1, ep=1 (default mapping)"
        )
    else:
        assumptions.append(
            f"explicit parallel tp/pp/ep={tp}/{pp}/{ep} "
            f"(product==chip_count={cfg.chip_count})"
        )
    if shape.has_pair:
        assumptions.append(
            f"pair_dim={shape.pair_dim}: L²·C_z pair act ≈ "
            f"{shape.pair_activation_bytes()/1e9:.4f} GB (capacity stress; "
            f"sharded / (tp*pp) when multi-card — crude)"
        )
    else:
        assumptions.append("seq-only protein (pair_dim=0)")

    npu, freq, mem, sram, hw_a, hw_n = _common_hw(cfg)
    assumptions.extend(hw_a)
    notes.extend(hw_n)

    eval_cfg, mac_f = _make_eval_config(
        cfg, freq, weight_bits=shape.weight_bits, dtype_name=cfg.dtype,
        batch=cfg.batch,
    )
    _append_assumed_compute_notes(
        assumptions, notes, eval_cfg, mac_f, dtype_mac_factors=cfg.dtype_mac_factors,
    )
    result = evaluate_protein(shape, npu, sram, mem, eval_cfg)
    act_vol = shape.seq_activation_bytes()  # B · L · H · act_bytes
    pair_b = result.pair_bytes
    scaled = scale_domain_parallel(
        result.forward,
        mem=mem,
        cfg=eval_cfg,
        tp=tp,
        pp=pp,
        ep=ep,
        mb=max(int(cfg.decode_mb), 1),
        act_volume_bytes=act_vol,
        n_layers=shape.n_layers,
        capacity_weight_bytes=shape.weight_bytes(),
        capacity_other_bytes=shape.seq_activation_bytes() + pair_b,
        shard_other=True,  # crude pair/seq act shard for capacity
        c2c=get_c2c(cfg.c2c_gbps),
        c2c_hide=cfg.c2c_hide,
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        has_moe_experts=False,
        phase="forward",
        domain="protein",
    )
    assumptions.extend(scaled.assumptions)

    n_cores, tpc = cores_from_npu(npu, freq)
    peak = npu.peak_tops_at(freq) * mac_f
    mem_eff = mem.effective_bandwidth_Bps() / 1e9
    t_seq_ms = scaled.t_roofline_s * 1e3
    n_shard = tp * pp  # EP no-op without MoE
    pair_card = pair_b // n_shard if n_shard > 1 else pair_b
    t_bubble_ms = (scaled.t_roofline_s - scaled.t_stage_s) * 1e3 if scaled.bubble_frac > 0 else 0.0
    t_c2c_ms = (scaled.t_c2c_s + scaled.t_pp_act_s + scaled.t_a2a_s) * 1e3

    return MetricsCard(
        TTFT_ms=t_seq_ms,  # alias: time/seq as TTFT-analogue
        TPOT_ms=0.0,
        wall=scaled.wall,
        t_compute_ms=scaled.t_compute_s * 1e3,
        t_dram_ms=scaled.t_memory_s * 1e3,
        t_c2c_ms=t_c2c_ms,
        t_fabric_ms=0.0,
        t_bubble_ms=t_bubble_ms,
        bytes_W=scaled.weight_bytes_card,
        bytes_KV=pair_card,  # reuse KV slot as pair footprint flag
        bytes_coll=scaled.collective_bytes + scaled.pp_act_bytes + scaled.a2a_bytes,
        util=scaled.mean_gemm_util,
        oom=scaled.oom,
        chips=cfg.chip_count,
        peak_tops=peak,
        mem_eff_GBps=mem_eff,
        model_id=cfg.model_id,
        mem_kind=str(mem.kind),
        tp=tp,
        pp=pp,
        ep=ep,
        n_cores=n_cores,
        tops_per_core=tpc,
        pe_rows=npu.rows,
        pe_cols=npu.cols,
        n_engines=npu.n_engines,
        sram_mib=cfg.sram_mib,
        prompt_len=0,
        decode_seq_len=shape.seq_len,
        batch=shape.batch,
        domain="protein",
        time_per_seq_ms=t_seq_ms,
        pair_bytes=pair_card,
        capacity_needed_GB=scaled.capacity_needed_bytes / GIB,
        capacity_needed_GB_decimal=scaled.capacity_needed_bytes / 1e9,
        capacity_GB=mem.capacity_bytes / GIB,
        capacity_GB_decimal=mem.capacity_bytes / 1e9,
        seq_len=shape.seq_len,
        non_gemm_overhead=resolved_non_gemm_overhead(eval_cfg),
        dtype_mac_factor=mac_f,
        t_sync_ms=scaled.t_sync_s * 1e3,
        n_sync_per_token=scaled.n_sync,
        c2c_latency_us=float(cfg.c2c_latency_us),
        sync_overlap=float(cfg.sync_overlap),
        **_mem_card_fields(cfg, mem),
        notes=notes,
        assumptions=assumptions,
    )


def primary_latency_ms(card: MetricsCard) -> tuple[float, str]:
    """Domain primary latency (ms) used for scaling efficiency.

    LLM → TPOT; video → TTFC; protein → time_per_seq.
    """
    if card.domain == "video":
        return float(card.TTFC_ms or card.TTFT_ms or 0.0), "TTFC"
    if card.domain == "protein":
        return float(card.time_per_seq_ms or card.TTFT_ms or 0.0), "time_per_seq"
    return float(card.TPOT_ms or 0.0), "TPOT"


def apply_scale_efficiency(card: MetricsCard, cfg: WorkbenchConfig) -> MetricsCard:
    """Fill scale_efficiency / speedup vs single-card baseline (chips=tp=pp=ep=1).

    ``scale_efficiency = (t_single / t_multi) / chip_count`` (ideal 1.0).
    ``speedup = t_single / t_multi``. On chips=1: both 1.0 with note "single-card".
    Honesty: ignores host/NIC non-ideal; C2C (and PP bubble / EP A2A) already in t_multi.
    """
    t_multi, metric = primary_latency_ms(card)
    chips = int(card.chips)
    n_parallel = int(card.tp) * int(card.pp) * int(card.ep)
    # Prefer chip_count; fall back to product if somehow mismatched
    n = max(chips, n_parallel, 1)

    honesty = (
        "scale_efficiency=(t_single/t_multi)/chips on domain primary "
        f"({metric}); ideal 1.0; ignores host/NIC non-ideal; "
        "C2C/PP-bubble/EP already in t_multi [derived]"
    )

    if n <= 1:
        card.scale_efficiency = 1.0
        card.speedup = 1.0
        card.t_single_primary_ms = t_multi
        card.scale_metric = metric
        if "single-card" not in card.notes:
            card.notes.append("single-card")
        if honesty not in card.assumptions:
            card.assumptions.append(honesty)
        return card

    # Same config except chips=1, tp=pp=ep=1 (default parallel mapping)
    base_cfg = replace(cfg, chip_count=1, parallel=None)
    base = evaluate_workbench(base_cfg)
    t_single, _ = primary_latency_ms(base)
    if t_multi <= 0.0 or not (t_multi == t_multi):  # zero or NaN
        speedup = 0.0
        eff = 0.0
    else:
        speedup = t_single / t_multi
        eff = speedup / float(n)

    card.scale_efficiency = eff
    card.speedup = speedup
    card.t_single_primary_ms = t_single
    card.scale_metric = metric
    note = (
        f"scale vs chips=1: t_single={t_single:.4f} ms → speedup={speedup:.4f}× "
        f"eff={eff:.4f} (chips={n})"
    )
    if note not in card.notes:
        card.notes.append(note)
    if honesty not in card.assumptions:
        card.assumptions.append(honesty)
    return card


def evaluate_workbench(cfg: WorkbenchConfig, *, scale_baseline: bool = True) -> MetricsCard:
    """cfg → MetricsCard for llm / video / protein domains.

    LLM uses evaluate_scaleup (multi-chip). Video/protein use evaluate_video /
    evaluate_protein plus TP/PP/EP domain scale-up (C2C collectives + PP bubble).
    Attaches scale_efficiency vs single-card baseline (v0.25) unless
    ``scale_baseline=False`` (v0.31: Pareto sweeps skip the extra chips=1 eval).
    """
    series = get_series(cfg.model_id)
    if series.domain == "llm":
        card = apply_energy_cost(_evaluate_workbench_llm(cfg), cfg)
    elif series.domain == "video":
        card = apply_energy_cost(_evaluate_workbench_video(cfg), cfg)
    elif series.domain == "protein":
        card = apply_energy_cost(_evaluate_workbench_protein(cfg), cfg)
    else:
        raise NotImplementedError(
            f"unsupported domain={series.domain!r} for {cfg.model_id!r}"
        )
    if not scale_baseline:
        return card
    return apply_scale_efficiency(card, cfg)


def workbench_scan(
    model_id: str,
    *,
    chips_list: list[int] | tuple[int, ...] = DEFAULT_CHIP_SWEEP,
    base: WorkbenchConfig | None = None,
    **overrides: Any,
) -> list[MetricsCard]:
    """Sweep chip_count over chips_list; return MetricsCards.

    Chip sweep always re-defaults parallel to tp=chip_count (pp=ep=1).
    Pass per-chip explicit parallel via a custom loop instead.
    """
    cards: list[MetricsCard] = []
    base_kw: dict[str, Any] = {}
    if base is not None:
        base_kw = {
            "mem_kind": base.mem_kind,
            "n_packages": base.n_packages,
            "n_ranks": base.n_ranks,
            "n_channels": base.n_channels,
            "data_rate_GTs": base.data_rate_GTs,
            "width_bits": base.width_bits,
            "efficiency": base.efficiency,
            "capacity_GB": base.capacity_GB,
            "n_cores": base.n_cores,
            "tops_per_core": base.tops_per_core,
            "pe_rows": base.pe_rows,
            "pe_cols": base.pe_cols,
            "n_engines": base.n_engines,
            "frequency_hz": base.frequency_hz,
            "mac_efficiency": base.mac_efficiency,
            "sku": base.sku,
            "prompt_len": base.prompt_len,
            "decode_seq_len": base.decode_seq_len,
            "batch": base.batch,
            "weight_bits": base.weight_bits,
            "kv_bits": base.kv_bits,
            "act_bits": base.act_bits,
            "dtype": base.dtype,
            "quant": base.quant,
            "sram_mib": base.sram_mib,
            "kv_fabric": base.kv_fabric,
            "remote_kv_frac": base.remote_kv_frac,
            "c2c_gbps": base.c2c_gbps,
            "c2c_hide": base.c2c_hide,
            "contention_mode": base.contention_mode,
            "sram_policy": base.sram_policy,
            "weight_hide_factor": base.weight_hide_factor,
            "embed_policy": base.embed_policy,
            "decode_mb": base.decode_mb,
            "mem_type": base.mem_type,
            "mem_form": base.mem_form,
            "mem_width_bits": base.mem_width_bits,
            "mem_rate_MTps": base.mem_rate_MTps,
            "mem_count": base.mem_count,
            "mem_cap_GB": base.mem_cap_GB,
            "hbm_height": base.hbm_height,
            "hbm_die_Gb": base.hbm_die_Gb,
            "c2c_latency_us": base.c2c_latency_us,
            "sync_overlap": base.sync_overlap,
            "attn_parallel": base.attn_parallel,
            "moe_shard": base.moe_shard,
            "spec_k": base.spec_k,
            "spec_accept": base.spec_accept,
            "spec_draft": base.spec_draft,
            "spec_draft_frac": base.spec_draft_frac,
            **econ_kwargs_from_obj(base),
        }
    for n in chips_list:
        kw = {**base_kw, **overrides}
        kw["model_id"] = model_id
        kw["chip_count"] = int(n)
        kw["parallel"] = None  # tp=chips default
        cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    return cards


def format_scan_table(cards: list[MetricsCard]) -> str:
    hdr = (
        f"{'chips':>5}  {'tp':>3}  {'TTFT_ms':>12}  {'TPOT_ms':>12}  "
        f"{'wall':>8}  {'util':>8}  {'peak_T':>8}  {'mem_GBps':>9}  "
        f"{'oom':>3}  {'scale_eff':>9}  {'speedup':>8}  "
        f"{'W_bytes':>12}  {'KV_bytes':>12}"
    )
    lines = [hdr, "-" * len(hdr)]
    for c in cards:
        lines.append(
            f"{c.chips:5d}  {c.tp:3d}  {c.TTFT_ms:12.4f}  {c.TPOT_ms:12.4f}  "
            f"{c.wall:>8}  {c.util:8.4f}  {c.peak_tops:8.2f}  {c.mem_eff_GBps:9.1f}  "
            f"{str(c.oom):>3}  {c.scale_efficiency:9.4f}  {c.speedup:8.4f}  "
            f"{c.bytes_W:12d}  {c.bytes_KV:12d}"
        )
    return "\n".join(lines)


def export_scan_csv(
    cards: list[MetricsCard],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cards:
        path.write_text("", encoding="utf-8")
        return path
    rows = [c.to_row() for c in cards]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def single_card_reference(
    model_id: str = "illustrative_27B",
    *,
    mem_kind: str = "HBM",
    sram_mib: float = 64.0,
    prompt_len: int = 512,
    decode_seq_len: int = 512,
    n_cores: int = 16,
    tops_per_core: float = 6.25,
) -> tuple[MetricsCard, Any]:
    """Helper: workbench chips=1 vs raw evaluate_inference for tolerance tests."""
    cfg = WorkbenchConfig(
        model_id=model_id,
        chip_count=1,
        mem_kind=mem_kind,
        sram_mib=sram_mib,
        prompt_len=prompt_len,
        decode_seq_len=decode_seq_len,
        n_cores=n_cores,
        tops_per_core=tops_per_core,
    )
    card = evaluate_workbench(cfg)
    shape = resolve_llm_shape(model_id)
    npu, freq, _ = build_npu(cfg)
    mem, _ = build_memory(cfg)
    sram = SRAMConfig(int(sram_mib * 1024 * 1024))
    raw = evaluate_inference(
        shape,
        npu,
        sram,
        mem,
        EvalConfig(
            prompt_len=prompt_len,
            decode_seq_len=decode_seq_len,
            frequency_hz=freq,
            count_lm_head=True,  # v0.31: workbench counts the LM head
        ),
    )
    return card, raw


# ---------------------------------------------------------------------------
# v0.13: parallel combination matrix + mem geometry sweep + card export
# ---------------------------------------------------------------------------

# v0.29: units are LPDDR x64 packages / HBM stacks; rate sweeps are tied to the
# preset generation (HBM_PRESET = HBM3E, LPDDR_PRESET = LPDDR5X). ≤0.28 mixed
# HBM3/3E/4/4E "labels" (6.4/9.2/8.0/10.0) on one HBM3-shaped stack — invalid.
# With a structured base (mem_type set) the sweep uses that type's catalog.
DEFAULT_MEM_CHANNEL_SWEEP = (2, 4, 6, 8)  # HBM stacks / LPDDR packages
DEFAULT_MEM_RATE_SWEEP_HBM = (8.0, 9.2, 9.6, 9.8)  # HBM3E grades (GT/s)
DEFAULT_MEM_RATE_SWEEP_LPDDR = (7.5, 8.533, 9.6, 10.667)  # LPDDR5X grades (GT/s)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def enumerate_parallel_combos(chips: int) -> list[tuple[int, int, int]]:
    """Enumerate tp×pp×ep = chips with reasonable degree bounds.

    Each factor is drawn from divisors of ``chips``. When ``chips`` is a
    power of two, restrict to {1, 2, 4, 8, …}; otherwise use all positive
    divisors (so non-pow2 chip counts still get a matrix).
    """
    if chips < 1:
        raise ValueError(f"chips must be >= 1, got {chips}")
    divisors = [d for d in range(1, chips + 1) if chips % d == 0]
    if _is_pow2(chips):
        cands = [d for d in divisors if _is_pow2(d)]
    else:
        cands = divisors
    cand_set = set(cands)
    out: list[tuple[int, int, int]] = []
    for tp in cands:
        for pp in cands:
            prod = tp * pp
            if prod > chips or chips % prod != 0:
                continue
            ep = chips // prod
            if ep in cand_set:
                out.append((tp, pp, ep))
    return out


def _base_kw_from_overrides(
    base: WorkbenchConfig | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    base_kw: dict[str, Any] = {}
    if base is not None:
        base_kw = {
            "mem_kind": base.mem_kind,
            "n_packages": base.n_packages,
            "n_ranks": base.n_ranks,
            "n_channels": base.n_channels,
            "data_rate_GTs": base.data_rate_GTs,
            "width_bits": base.width_bits,
            "efficiency": base.efficiency,
            "capacity_GB": base.capacity_GB,
            "n_cores": base.n_cores,
            "tops_per_core": base.tops_per_core,
            "pe_rows": base.pe_rows,
            "pe_cols": base.pe_cols,
            "n_engines": base.n_engines,
            "frequency_hz": base.frequency_hz,
            "mac_efficiency": base.mac_efficiency,
            "sku": base.sku,
            "prompt_len": base.prompt_len,
            "decode_seq_len": base.decode_seq_len,
            "batch": base.batch,
            "weight_bits": base.weight_bits,
            "kv_bits": base.kv_bits,
            "act_bits": base.act_bits,
            "dtype": base.dtype,
            "quant": base.quant,
            "sram_mib": base.sram_mib,
            "kv_fabric": base.kv_fabric,
            "remote_kv_frac": base.remote_kv_frac,
            "c2c_gbps": base.c2c_gbps,
            "c2c_hide": base.c2c_hide,
            "contention_mode": base.contention_mode,
            "sram_policy": base.sram_policy,
            "weight_hide_factor": base.weight_hide_factor,
            "embed_policy": base.embed_policy,
            "decode_mb": base.decode_mb,
            "mem_type": base.mem_type,
            "mem_form": base.mem_form,
            "mem_width_bits": base.mem_width_bits,
            "mem_rate_MTps": base.mem_rate_MTps,
            "mem_count": base.mem_count,
            "mem_cap_GB": base.mem_cap_GB,
            "hbm_height": base.hbm_height,
            "hbm_die_Gb": base.hbm_die_Gb,
            "c2c_latency_us": base.c2c_latency_us,
            "sync_overlap": base.sync_overlap,
            "attn_parallel": base.attn_parallel,
            "moe_shard": base.moe_shard,
            "spec_k": base.spec_k,
            "spec_accept": base.spec_accept,
            "spec_draft": base.spec_draft,
            "spec_draft_frac": base.spec_draft_frac,
            **econ_kwargs_from_obj(base),
        }
    return {**base_kw, **overrides}


def workbench_parallel_matrix(
    model_id: str,
    chips: int,
    *,
    sort_by: str = "TPOT",
    base: WorkbenchConfig | None = None,
    **overrides: Any,
) -> list[MetricsCard]:
    """Evaluate MetricsCard for every tp×pp×ep factorization of ``chips``.

    Sorted by TPOT (default) or TTFT ascending. MoE models (`series/moe-*`)
    make EP degrees meaningful via A2A traffic. Dense/MLA shapes force ep=1
    (scaleup rejects ep>1 on non-MoE).
    """
    combos = enumerate_parallel_combos(chips)
    # Filter EP: dense/MLA → ep=1 only; MoE → ep must divide n_experts
    series = get_series(model_id)
    shape = series.shape if series.domain == "llm" else None
    is_moe = bool(getattr(shape, "is_moe", False))
    n_experts = int(getattr(shape, "n_experts", 1) or 1)
    filtered: list[tuple[int, int, int]] = []
    for tp, pp, ep in combos:
        if not is_moe and ep != 1:
            continue
        if is_moe and ep > 1 and (n_experts % ep != 0):
            continue
        filtered.append((tp, pp, ep))
    combos = filtered
    kw0 = _base_kw_from_overrides(base, overrides)
    cards: list[MetricsCard] = []
    for tp, pp, ep in combos:
        kw = dict(kw0)
        kw["model_id"] = model_id
        kw["chip_count"] = int(chips)
        kw["parallel"] = ParallelOverride(tp=tp, pp=pp, ep=ep)
        cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    key = sort_by.strip().upper()
    if key in ("TTFT", "TTFT_MS"):
        cards.sort(key=lambda c: (c.TTFT_ms, c.TPOT_ms, c.tp, c.pp, c.ep))
    else:
        cards.sort(key=lambda c: (c.TPOT_ms, c.TTFT_ms, c.tp, c.pp, c.ep))
    return cards


def format_parallel_table(cards: list[MetricsCard]) -> str:
    hdr = (
        f"{'rank':>4}  {'chips':>5}  {'tp':>3}  {'pp':>3}  {'ep':>3}  "
        f"{'TTFT_ms':>12}  {'TPOT_ms':>12}  {'wall':>8}  "
        f"{'t_c2c_ms':>10}  {'t_bub_ms':>10}  {'util':>8}  {'oom':>3}"
    )
    lines = [hdr, "-" * len(hdr)]
    for i, c in enumerate(cards, 1):
        lines.append(
            f"{i:4d}  {c.chips:5d}  {c.tp:3d}  {c.pp:3d}  {c.ep:3d}  "
            f"{c.TTFT_ms:12.4f}  {c.TPOT_ms:12.4f}  {c.wall:>8}  "
            f"{c.t_c2c_ms:10.4f}  {c.t_bubble_ms:10.4f}  "
            f"{c.util:8.4f}  {str(c.oom):>3}"
        )
    return "\n".join(lines)


def workbench_mem_sweep(
    model_id: str,
    *,
    chips: int = 1,
    mem_kind: str = "HBM",
    vary: str = "both",
    channel_list: list[int] | tuple[int, ...] | None = None,
    rate_list: list[float] | tuple[float, ...] | None = None,
    base: WorkbenchConfig | None = None,
    **overrides: Any,
) -> list[MetricsCard]:
    """Small geometry grid on n_channels and/or data_rate_GTs.

    Fixed chips/model; shows BW-wall flips as effective GB/s changes.
    ``vary`` ∈ {channels, rate, both}.
    """
    kw_probe = _base_kw_from_overrides(base, overrides)
    if kw_probe.get("mem_type"):
        return _structured_mem_sweep(
            model_id, chips=chips, vary=vary, channel_list=channel_list,
            rate_list=rate_list, kw0=kw_probe,
        )
    kind = str(mem_kind).upper()
    chans = list(channel_list) if channel_list is not None else list(DEFAULT_MEM_CHANNEL_SWEEP)
    if rate_list is not None:
        rates = list(rate_list)
    elif kind == "LPDDR":
        rates = list(DEFAULT_MEM_RATE_SWEEP_LPDDR)
    else:
        rates = list(DEFAULT_MEM_RATE_SWEEP_HBM)

    vary_l = vary.strip().lower()
    if vary_l not in ("channels", "rate", "both"):
        raise ValueError(f"vary must be channels|rate|both, got {vary!r}")

    kw0 = _base_kw_from_overrides(base, overrides)
    kw0["model_id"] = model_id
    kw0["chip_count"] = int(chips)
    kw0["parallel"] = None
    kw0["mem_kind"] = kind

    cards: list[MetricsCard] = []
    if vary_l == "channels":
        for ch in chans:
            kw = dict(kw0)
            kw["n_channels"] = int(ch)
            cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    elif vary_l == "rate":
        for rate in rates:
            kw = dict(kw0)
            kw["data_rate_GTs"] = float(rate)
            cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    else:
        for ch in chans:
            for rate in rates:
                kw = dict(kw0)
                kw["n_channels"] = int(ch)
                kw["data_rate_GTs"] = float(rate)
                cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    return cards


def _structured_mem_sweep(
    model_id: str,
    *,
    chips: int,
    vary: str,
    channel_list: Any,
    rate_list: Any,
    kw0: dict[str, Any],
) -> list[MetricsCard]:
    """mem sweep on a structured base: counts / rates from that type's catalog."""
    from .mem_catalog import counts_for, rates_for

    t, form = kw0["mem_type"], kw0.get("mem_form")
    counts = list(channel_list) if channel_list is not None else counts_for(t, form)
    if rate_list is not None:
        rates = [float(r) * (1000.0 if float(r) < 100 else 1.0) for r in rate_list]
    else:
        rates = [float(r) for r in rates_for(t, form)]
    v = vary.strip().lower()
    if v not in ("channels", "rate", "both"):
        raise ValueError(f"vary must be channels|rate|both, got {vary!r}")
    kw0 = dict(kw0, model_id=model_id, chip_count=int(chips), parallel=None)
    grid: list[tuple[Any, Any]]
    if v == "channels":
        grid = [(n, None) for n in counts]
    elif v == "rate":
        grid = [(None, r) for r in rates]
    else:
        grid = [(n, r) for n in counts for r in rates]
    cards: list[MetricsCard] = []
    for n, r in grid:
        kw = dict(kw0)
        if n is not None:
            kw["mem_count"] = int(n)
        if r is not None:
            kw["mem_rate_MTps"] = float(r)
        cards.append(evaluate_workbench(WorkbenchConfig(**kw)))
    return cards


def format_mem_sweep_table(cards: list[MetricsCard]) -> str:
    hdr = (
        f"{'mem':>7}  {'eff_GBps':>10}  {'TTFT_ms':>12}  {'TPOT_ms':>12}  "
        f"{'wall':>8}  {'t_comp_ms':>10}  {'t_dram_ms':>10}  "
        f"{'flip':>6}  {'chips':>5}"
    )
    lines = [hdr, "-" * len(hdr)]
    prev_wall: str | None = None
    for c in cards:
        flip = ""
        if prev_wall is not None and c.wall != prev_wall:
            flip = "FLIP"
        prev_wall = c.wall
        lines.append(
            f"{(c.mem_type or c.mem_kind):>7}  {c.mem_eff_GBps:10.1f}  "
            f"{c.TTFT_ms:12.4f}  {c.TPOT_ms:12.4f}  "
            f"{c.wall:>8}  {c.t_compute_ms:10.4f}  {c.t_dram_ms:10.4f}  "
            f"{flip:>6}  {c.chips:5d}"
        )
    return "\n".join(lines)


def export_metrics_card_json(card: MetricsCard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(card.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def export_metrics_card_md(card: MetricsCard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.to_markdown(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# v0.16: package-geometry + compute-hierarchy sweeps (assumed DSE ranges)
# ---------------------------------------------------------------------------


def workbench_package_sweep(
    model_id: str,
    *,
    package_ids: list[str] | tuple[str, ...] | None = None,
    kind: str | None = None,
    chips: int = 1,
    base: WorkbenchConfig | None = None,
    axis: str | None = None,
    **overrides: Any,
) -> list[tuple[Any, MetricsCard]]:
    """Sweep memory configs → (PackageOption, MetricsCard) rows.

    v0.29 ``axis`` ∈ {type, count, rate}: iterate the structured catalog around
    the base memory (base.mem_type or the HBM/LPDDR preset): all 7 types at
    their defaults, all unit counts, or all rate grades of that generation.
    Otherwise ``package_ids`` (canonical or legacy ids) or the default list
    (DEFAULT_PACKAGE_SWEEP). Efficiencies remain assumed / uncalibrated.
    """
    if axis:
        return _package_axis_sweep(
            model_id, axis=axis, chips=chips, base=base, kind=kind, **overrides
        )
    from .package_ranges import (
        DEFAULT_PACKAGE_SWEEP,
        DEFAULT_PACKAGE_SWEEP_HBM,
        DEFAULT_PACKAGE_SWEEP_LPDDR,
        get_package,
        list_packages,
    )

    if package_ids is not None:
        ids = list(package_ids)
    elif kind is not None:
        k = str(kind).upper()
        if k == "LPDDR":
            ids = list(DEFAULT_PACKAGE_SWEEP_LPDDR)
        elif k == "HBM":
            ids = list(DEFAULT_PACKAGE_SWEEP_HBM)
        else:
            ids = [p.id for p in list_packages(kind=k)]  # type: ignore[arg-type]
    else:
        ids = list(DEFAULT_PACKAGE_SWEEP)

    kw0 = _base_kw_from_overrides(base, overrides)
    kw0["model_id"] = model_id
    kw0["chip_count"] = int(chips)
    kw0["parallel"] = None

    rows: list[tuple[Any, MetricsCard]] = []
    for pid in ids:
        pkg = get_package(pid)
        kw = dict(kw0)
        kw.update(pkg.workbench_kwargs())
        card = evaluate_workbench(WorkbenchConfig(**kw))
        card.notes = list(card.notes) + [f"package={pkg.id} [assumed]"]
        rows.append((pkg, card))
    return rows


def _package_axis_sweep(
    model_id: str,
    *,
    axis: str,
    chips: int,
    base: WorkbenchConfig | None,
    kind: str | None = None,
    **overrides: Any,
) -> list[tuple[Any, MetricsCard]]:
    from .mem_catalog import sweep_specs
    from .package_ranges import PackageOption

    kw0 = _base_kw_from_overrides(base, overrides)
    probe = WorkbenchConfig(**{**kw0, "model_id": model_id})
    spec = build_mem_spec(probe)
    if spec is None:
        k = str(kind or probe.mem_kind).upper()
        spec = default_spec_for_preset(HBM_PRESET if k == "HBM" else LPDDR_PRESET)
    assert spec is not None
    kw0["model_id"] = model_id
    kw0["chip_count"] = int(chips)
    kw0["parallel"] = None
    rows: list[tuple[Any, MetricsCard]] = []
    for sp in sweep_specs(spec, axis):
        pkg = PackageOption.from_spec(sp)
        kw = dict(kw0)
        kw.update(pkg.workbench_kwargs())
        card = evaluate_workbench(WorkbenchConfig(**kw))
        card.notes = list(card.notes) + [f"package={pkg.id} axis={axis} [assumed]"]
        rows.append((pkg, card))
    return rows


def format_package_sweep_table(rows: list[tuple[Any, MetricsCard]]) -> str:
    hdr = (
        f"{'package':34s}  {'kind':5s}  {'n':>3}  {'w':>4}  {'GT/s':>6}  "
        f"{'peak_GBps':>10}  {'eff_GBps':>10}  {'TTFT_ms':>10}  "
        f"{'TPOT_ms':>10}  {'wall':>8}  {'flip':>6}  tag"
    )
    lines = [hdr, "-" * len(hdr)]
    prev_wall: str | None = None
    for pkg, c in rows:
        flip = ""
        if prev_wall is not None and c.wall != prev_wall:
            flip = "FLIP"
        prev_wall = c.wall
        lines.append(
            f"{pkg.id:34s}  {pkg.kind:5s}  {pkg.n_channels:3d}  "
            f"{pkg.width_bits:4d}  {pkg.data_rate_GTs:6.3f}  "
            f"{pkg.peak_bandwidth_GBps():10.1f}  {c.mem_eff_GBps:10.1f}  "
            f"{c.TTFT_ms:10.4f}  {c.TPOT_ms:10.4f}  {c.wall:>8}  {flip:>6}  "
            f"{c.mem_tag_zh}"
        )
    return "\n".join(lines)


def workbench_compute_sweep(
    model_id: str,
    *,
    compute_ids: list[str] | tuple[str, ...] | None = None,
    level: str = "all",
    chips: int = 1,
    mem_kind: str = "HBM",
    base: WorkbenchConfig | None = None,
    **overrides: Any,
) -> list[tuple[Any, MetricsCard]]:
    """Sweep core (4–16T) and/or cluster (64–256T) compute catalog options."""
    from .package_ranges import (
        DEFAULT_COMPUTE_SWEEP,
        DEFAULT_COMPUTE_SWEEP_CLUSTER,
        DEFAULT_COMPUTE_SWEEP_CORE,
        get_compute,
    )

    if compute_ids is not None:
        ids = list(compute_ids)
    else:
        lvl = str(level).strip().lower()
        if lvl == "core":
            ids = list(DEFAULT_COMPUTE_SWEEP_CORE)
        elif lvl == "cluster":
            ids = list(DEFAULT_COMPUTE_SWEEP_CLUSTER)
        else:
            ids = list(DEFAULT_COMPUTE_SWEEP)

    kw0 = _base_kw_from_overrides(base, overrides)
    kw0["model_id"] = model_id
    kw0["chip_count"] = int(chips)
    kw0["parallel"] = None
    kw0["mem_kind"] = str(mem_kind).upper()
    # Clear PE/SKU so cores×tops win
    kw0["sku"] = None
    kw0["pe_rows"] = None
    kw0["pe_cols"] = None
    kw0["n_engines"] = None

    rows: list[tuple[Any, MetricsCard]] = []
    for cid in ids:
        opt = get_compute(cid)
        kw = dict(kw0)
        kw.update(opt.workbench_kwargs())
        card = evaluate_workbench(WorkbenchConfig(**kw))
        card.notes = list(card.notes) + [
            f"compute={opt.id} peak={opt.peak_tops:g}T [assumed]"
        ]
        rows.append((opt, card))
    return rows


def format_compute_sweep_table(rows: list[tuple[Any, MetricsCard]]) -> str:
    hdr = (
        f"{'compute':28s}  {'level':8s}  {'peak_T':>8}  {'cores':>5}  "
        f"{'T/core':>7}  {'TTFT_ms':>10}  {'TPOT_ms':>10}  "
        f"{'wall':>8}  {'util':>7}  {'flip':>6}"
    )
    lines = [hdr, "-" * len(hdr)]
    prev_wall: str | None = None
    for opt, c in rows:
        flip = ""
        if prev_wall is not None and c.wall != prev_wall:
            flip = "FLIP"
        prev_wall = c.wall
        lines.append(
            f"{opt.id:28s}  {opt.level:8s}  {opt.peak_tops:8.1f}  "
            f"{opt.n_cores:5d}  {opt.tops_per_core:7.2f}  "
            f"{c.TTFT_ms:10.4f}  {c.TPOT_ms:10.4f}  {c.wall:>8}  "
            f"{c.util:7.4f}  {flip:>6}"
        )
    return "\n".join(lines)
