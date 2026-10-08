"""TTFT / TPOT evaluation: compute wall vs memory wall.

Roofline-style per phase:
  t_compute = cycles / frequency_Hz
  t_memory  = contention result (serialize by default)
  t_phase   = max(t_compute, t_memory)   # perfect overlap bound
  also report t_sum = t_compute + t_memory (no overlap)

frequency_Hz is *assumed / uncalibrated*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .memory import ExternalMemory, SRAMConfig, SramPolicy
from .model_shape import ModelShape
from .npu import NPUConfig
from .traffic import (
    ContentionResult,
    PhaseTraffic,
    contend,
    decode_traffic,
    prefill_traffic,
)

ContentionMode = Literal["serialize", "share_fair", "lower_bound"]


@dataclass(frozen=True)
class EvalConfig:
    prompt_len: int = 1024
    decode_seq_len: int = 1024   # KV length at the measured decode step
    batch: int = 1
    frequency_hz: float = 1.0e9  # assumed 1 GHz
    pin_weights: bool = False
    contention_mode: ContentionMode = "serialize"
    # Assumed overlap when double_buffer_eligible (staging R=0, ≥2×W):
    # effective weight mem time *= (1 - hide). 0.0 = no hide (default).
    weight_hide_factor: float = 0.0
    sram_policy: SramPolicy = "weight_resident"
    # v0.26 opt-in assumed knobs (defaults preserve handcheck / conservation):
    # Coarse non-GEMM operator overhead as fraction of GEMM compute time.
    # t_compute' = t_compute * (1 + overhead). Not cycle-accurate.
    non_gemm_overhead: float = 0.0
    # Optional finer Softmax / RoPE / LN fractions; if any set, sum replaces
    # the coarse knob (still assumed, not silicon).
    softmax_frac: float | None = None
    rope_frac: float | None = None
    norm_frac: float | None = None
    # Resolved dtype MAC peak factor (user/assumed table). Default 1.0 =
    # bytes-only conservative behavior. peak_TOPS *= factor; t_compute /= factor.
    dtype_mac_factor: float = 1.0
    # v0.31: count the LM head (V·H weight read + GEMM) in prefill/decode.
    # Raw engine default off (handcheck arithmetic); the workbench turns it on.
    count_lm_head: bool = False


@dataclass(frozen=True)
class PhaseResult:
    traffic: PhaseTraffic
    contention: ContentionResult
    t_compute_s: float
    t_memory_s: float
    t_roofline_s: float   # max(compute, mem)
    t_sum_s: float        # compute + mem
    wall: str             # "compute" | "memory" | "balanced"


@dataclass(frozen=True)
class InferenceResult:
    shape: ModelShape
    npu: NPUConfig
    sram: SRAMConfig
    mem: ExternalMemory
    cfg: EvalConfig
    prefill: PhaseResult
    decode: PhaseResult

    @property
    def ttft_s(self) -> float:
        return self.prefill.t_roofline_s

    @property
    def tpot_s(self) -> float:
        return self.decode.t_roofline_s

    def peak_tops(self) -> float:
        return self.npu.peak_tops_at(self.cfg.frequency_hz)

    def summary_lines(self) -> list[str]:
        sku_tag = f"sku={self.npu.sku_name} " if self.npu.sku_name else ""
        parts = self.decode.traffic.partitions
        part_s = parts.summary() if parts else "n/a"
        tops = self.peak_tops()
        tops_s = f"{tops:.2f}" if tops < 10 else f"{tops:.1f}"
        R = self.decode.traffic.resident_layers
        L = self.shape.n_layers
        lines = [
            f"=== {self.shape.name} | {self.mem.kind} | SRAM={self.sram.capacity_bytes/2**20:.1f}MiB "
            f"| {sku_tag}PE={self.npu.rows}x{self.npu.cols}×{self.npu.n_engines} "
            f"@ {self.cfg.frequency_hz/1e9:.2f}GHz → {tops_s} TOPS (assumed geom) ===",
            f"  shape: {self.shape.summary()}",
            f"  mem:   {self.mem.summary()}",
            f"  SRAM partitions (derived): {part_s}",
            f"  sram_policy={self.cfg.sram_policy}  resident_layers R={R}/{L}",
            f"  TTFT (prefill S={self.cfg.prompt_len}): {self.ttft_s*1e3:.4f} ms "
            f"[{self.prefill.wall}-bound]  "
            f"compute={self.prefill.t_compute_s*1e3:.4f} ms  "
            f"mem({self.cfg.contention_mode})={self.prefill.t_memory_s*1e3:.4f} ms",
            f"    DRAM bytes: W={self.prefill.traffic.weight_bytes_dram} "
            f"KV={self.prefill.traffic.kv_bytes_dram} "
            f"(r={self.prefill.traffic.kv_bytes_read}/w={self.prefill.traffic.kv_bytes_write}) "
            f"Act={self.prefill.traffic.act_bytes_dram}  "
            f"FLOPs={self.prefill.traffic.flops}  "
            f"mean_GEMM_util={self.prefill.traffic.mean_gemm_util:.4f} (M={self.prefill.traffic.m_eff})",
            f"    weight_path={self.prefill.traffic.weight_path} "
            f"R={self.prefill.traffic.resident_layers}",
            f"    BW breakdown (serialize frac of mem time): "
            f"W={self.prefill.contention.t_weight_s/max(self.prefill.contention.t_serialize_s,1e-30):.2%} "
            f"KV={self.prefill.contention.t_kv_s/max(self.prefill.contention.t_serialize_s,1e-30):.2%} "
            f"Act={self.prefill.contention.t_act_s/max(self.prefill.contention.t_serialize_s,1e-30):.2%}",
            f"  TPOT (decode ctx={self.cfg.decode_seq_len}): {self.tpot_s*1e3:.4f} ms/tok "
            f"[{self.decode.wall}-bound]  "
            f"compute={self.decode.t_compute_s*1e3:.4f} ms  "
            f"mem({self.cfg.contention_mode})={self.decode.t_memory_s*1e3:.4f} ms",
            f"    DRAM bytes: W={self.decode.traffic.weight_bytes_dram} "
            f"KV={self.decode.traffic.kv_bytes_dram} "
            f"(r={self.decode.traffic.kv_bytes_read}/w={self.decode.traffic.kv_bytes_write}) "
            f"Act={self.decode.traffic.act_bytes_dram}  "
            f"FLOPs={self.decode.traffic.flops}  "
            f"mean_GEMM_util={self.decode.traffic.mean_gemm_util:.4f} (M={self.decode.traffic.m_eff})",
            f"    weight_path={self.decode.traffic.weight_path} "
            f"R={R}/{L} "
            f"dbl_buf={self.decode.traffic.double_buffer_eligible} "
            f"hide={self.cfg.weight_hide_factor:.2f}[assumed]",
            f"    BW breakdown: "
            f"W={self.decode.contention.t_weight_s/max(self.decode.contention.t_serialize_s,1e-30):.2%} "
            f"KV={self.decode.contention.t_kv_s/max(self.decode.contention.t_serialize_s,1e-30):.2%} "
            f"Act={self.decode.contention.t_act_s/max(self.decode.contention.t_serialize_s,1e-30):.2%}",
            f"    contention: serialize={self.decode.contention.t_serialize_s*1e3:.4f} ms  "
            f"share_fair={self.decode.contention.t_share_fair_s*1e3:.4f} ms  "
            f"lower_bound={self.decode.contention.t_lower_bound_s*1e3:.4f} ms",
        ]
        return lines



def resolved_non_gemm_overhead(cfg: EvalConfig) -> float:
    """Return assumed non-GEMM overhead fraction (≥0).

    If any of softmax_frac / rope_frac / norm_frac is provided, their sum is
    used; otherwise ``non_gemm_overhead`` (default 0 = off).
    """
    finer = (cfg.softmax_frac, cfg.rope_frac, cfg.norm_frac)
    if any(v is not None for v in finer):
        return max(0.0, sum(float(v or 0.0) for v in finer))
    return max(0.0, float(cfg.non_gemm_overhead))


def effective_compute_time_s(compute_cycles: float, cfg: EvalConfig) -> float:
    """GEMM compute wall with optional dtype MAC factor + non-GEMM overhead.

    Defaults (dtype_mac_factor=1.0, overhead=0) → cycles / frequency_hz
    (bit-identical to pre-v0.26). Opt-in:
      t = cycles / (freq * dtype_mac_factor)
      t' = t * (1 + non_gemm_overhead)
    Factors are user/assumed — not silicon calibration.
    """
    mac_f = float(cfg.dtype_mac_factor)
    if mac_f <= 0.0:
        raise ValueError("dtype_mac_factor must be > 0")
    t = float(compute_cycles) / (cfg.frequency_hz * mac_f)
    return t * (1.0 + resolved_non_gemm_overhead(cfg))


def _mem_time(c: ContentionResult, mode: ContentionMode) -> float:
    if mode == "serialize":
        return c.t_serialize_s
    if mode == "share_fair":
        return c.t_share_fair_s
    return c.t_lower_bound_s


def _mem_time_with_hide(
    c: ContentionResult,
    mode: ContentionMode,
    *,
    hide_factor: float,
    double_buffer_eligible: bool,
) -> float:
    """Apply assumed weight hide when staging ≥ 2*W_layer (R=0 only).

    hide_factor in [0, 1]: fraction of weight stream time overlapped/hidden.
    Labeled assumed — not a claim of a real schedule.
    Does NOT reduce DRAM bytes — only effective mem time.
    """
    h = max(0.0, min(float(hide_factor), 1.0))
    if not double_buffer_eligible or h <= 0.0:
        return _mem_time(c, mode)
    tw = c.t_weight_s * (1.0 - h)
    tk = c.t_kv_s
    ta = c.t_act_s
    if mode == "serialize":
        return tw + tk + ta
    if mode == "share_fair":
        return max(2.0 * tw, 2.0 * tk) + ta
    return max(tw, tk) + ta


def _phase_result(
    traffic: PhaseTraffic,
    mem: ExternalMemory,
    cfg: EvalConfig,
) -> PhaseResult:
    cont = contend(traffic, mem)
    t_comp = effective_compute_time_s(traffic.compute_cycles, cfg)
    t_mem = _mem_time_with_hide(
        cont,
        cfg.contention_mode,
        hide_factor=cfg.weight_hide_factor,
        double_buffer_eligible=traffic.double_buffer_eligible,
    )
    t_roof = max(t_comp, t_mem)
    if abs(t_comp - t_mem) / max(t_roof, 1e-30) < 0.05:
        wall = "balanced"
    elif t_comp >= t_mem:
        wall = "compute"
    else:
        wall = "memory"
    return PhaseResult(
        traffic=traffic,
        contention=cont,
        t_compute_s=t_comp,
        t_memory_s=t_mem,
        t_roofline_s=t_roof,
        t_sum_s=t_comp + t_mem,
        wall=wall,
    )


def evaluate_inference(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig | None = None,
) -> InferenceResult:
    cfg = cfg or EvalConfig()
    # Propagate policy onto SRAM if using derived partitions
    sram_eff = sram
    if sram.partitions is None and sram.policy != cfg.sram_policy:
        sram_eff = SRAMConfig(
            capacity_bytes=sram.capacity_bytes,
            n_banks=sram.n_banks,
            partitions=None,
            policy=cfg.sram_policy,
        )
    pref = prefill_traffic(
        shape, npu, sram_eff, cfg.prompt_len, cfg.batch, cfg.pin_weights,
        sram_policy=cfg.sram_policy, lm_head=cfg.count_lm_head,
    )
    dec = decode_traffic(
        shape, npu, sram_eff, cfg.decode_seq_len, cfg.batch, cfg.pin_weights,
        sram_policy=cfg.sram_policy, lm_head=cfg.count_lm_head,
    )
    return InferenceResult(
        shape=shape,
        npu=npu,
        sram=sram_eff,
        mem=mem,
        cfg=cfg,
        prefill=_phase_result(pref, mem, cfg),
        decode=_phase_result(dec, mem, cfg),
    )


def evaluate_decode_phase(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig,
    *,
    batch: int,
    tokens_per_seq: int = 1,
) -> PhaseResult:
    """Single-card decode step at an arbitrary micro-batch / verify width (v0.31).

    Used by the scale-up evaluator for PP decode micro-batches (batch =
    ceil(B / decode_mb)) and speculative verify steps (tokens_per_seq = 1 + k).
    """
    sram_eff = sram
    if sram.partitions is None and sram.policy != cfg.sram_policy:
        sram_eff = SRAMConfig(
            capacity_bytes=sram.capacity_bytes,
            n_banks=sram.n_banks,
            partitions=None,
            policy=cfg.sram_policy,
        )
    dec = decode_traffic(
        shape, npu, sram_eff, cfg.decode_seq_len, batch, cfg.pin_weights,
        sram_policy=cfg.sram_policy, tokens_per_seq=tokens_per_seq,
        lm_head=cfg.count_lm_head,
    )
    return _phase_result(dec, mem, cfg)
