"""Compute FLOPs and external DRAM traffic for prefill / decode.

Weight path (v0.5+ honesty — staging vs resident; MoE = active stream):
  - Weights live in HBM/LPDDR by default.
  - **Staging** (R=0): weight_partition is a 1×/2× W_layer (or tile) buffer.
    Decode still reads each layer once/token → W DRAM ≈ L * W_layer
    (or L * W_layer * tiles if tiled). Double-buffer (staging≥2×W) only
    enables optional weight_hide_factor — does NOT cut bytes.
  - **Resident** (R≥1): R layers stay on-die across tokens.
    Steady-state decode W DRAM ≈ (L − R) * W_layer (0 if R≥L).
    Prefill / cold-start still loads all layers once (L * W_layer).
  - Labels:
      hbm_stream_all_layers  — R=0, full model stream (staging or tile)
      hbm_stream_tiled       — R=0, tiles>1 reload amplification
      hbm_stream_miss_layers — 0<R<L, only miss layers stream
      on_die_resident        — R≥L, decode W DRAM=0 (steady state)
      sram_pinned_body       — pin_weights and body fits (legacy pin path)
  - double_buffer_eligible is a *separate* flag (staging-only).

KV path (capacity-derived):
  - Prefill: always write-through KV to external (populate persistent cache).
  - Decode read: if kv_scratch >= full KV working set → on-die hit (0 read
    DRAM); else full KV read from external. New-token KV always write-through.

Contention modes:
  - "serialize": t = (Bw+Bkv+Bact)/BW
  - "share_fair": concurrent 50/50 split → t = max(2*Bw/BW, 2*Bkv/BW) + Bact/BW
  - Lower bound: t >= max(Bw, Bkv)/BW + Bact/BW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .memory import ExternalMemory, SRAMConfig, SRAMPartitions, SramPolicy
from .model_shape import ModelShape
from .npu import NPUConfig, gemm_compute_cycles, gemm_flops, gemm_utilization

WeightPath = Literal[
    "hbm_stream_all_layers",
    "hbm_stream_tiled",
    "hbm_stream_miss_layers",
    "on_die_resident",
    "sram_pinned_body",
]


@dataclass(frozen=True)
class PhaseTraffic:
    """One inference phase (prefill or one decode step)."""

    name: str
    flops: int
    weight_bytes_dram: int
    kv_bytes_dram: int
    act_bytes_dram: int
    compute_cycles: float
    mean_gemm_util: float
    m_eff: int  # effective M used (B*S for prefill, B for decode)
    weight_path: WeightPath = "hbm_stream_all_layers"
    partitions: SRAMPartitions | None = None
    # Fine-grained KV accounting (optional clarity)
    kv_bytes_read: int = 0
    kv_bytes_write: int = 0
    # Staging >= 2*W_layer and R==0 → double-buffer eligible (hide optional)
    double_buffer_eligible: bool = False
    layer_weight_bytes: int = 0
    resident_layers: int = 0
    # v0.31: LM head (V·H) bytes read this phase — included in weight_bytes_dram
    head_weight_bytes: int = 0
    # v0.31: query tokens per sequence in this decode step (1 + spec_k when verifying)
    tokens_per_seq: int = 1
    # v0.31: LM head share of ``flops`` / ``compute_cycles`` (split separately by scale-up)
    head_flops: int = 0
    head_cycles: float = 0.0

    @property
    def total_dram_bytes(self) -> int:
        return self.weight_bytes_dram + self.kv_bytes_dram + self.act_bytes_dram


@dataclass(frozen=True)
class ContentionResult:
    """Memory time under different weight/KV contention policies."""

    t_weight_s: float
    t_kv_s: float
    t_act_s: float
    t_serialize_s: float       # (Bw+Bkv+Bact)/BW
    t_share_fair_s: float      # max(2*Bw, 2*Bkv)/BW + Bact/BW  (act after)
    t_lower_bound_s: float     # max(Bw,Bkv)/BW + Bact/BW
    bandwidth_Bps: float

    def utilization_breakdown(self, wall_s: float) -> dict[str, float]:
        """Fraction of wall time attributable to each stream if serialized."""
        if wall_s <= 0:
            return {"weight": 0.0, "kv": 0.0, "act": 0.0, "idle_or_compute": 1.0}
        return {
            "weight": self.t_weight_s / wall_s,
            "kv": self.t_kv_s / wall_s,
            "act": self.t_act_s / wall_s,
            "mem_serialize_frac": self.t_serialize_s / wall_s,
        }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def weight_dram_bytes_for_layer(
    layer_weight_bytes: int,
    weight_staging_bytes: int,
    n_output_tiles: int,
) -> int:
    """Derive external weight traffic from staging capacity vs footprint.

    If layer fits in weight_staging: load once → layer_weight_bytes.
    Else: must reload weights for each output tile pass:
          bytes = layer_weight_bytes * n_output_tiles
    n_output_tiles >= 1.
    """
    if layer_weight_bytes <= 0:
        return 0
    if n_output_tiles < 1:
        n_output_tiles = 1
    if layer_weight_bytes <= weight_staging_bytes:
        return layer_weight_bytes
    return layer_weight_bytes * n_output_tiles


def estimate_output_tiles(
    out_rows: int,
    out_cols: int,
    weight_staging_bytes: int,
    layer_weight_bytes: int,
    act_elem_bytes: int,
) -> int:
    """How many weight-reload passes if weights do not fit in staging.

    Simple rule (documented, hand-checkable):
      If weights fit in staging: tiles = 1
      Else: tiles = ceil(out_rows / max(1, staging // max(act_elem*out_cols, 1)))
      with a floor of 1 and a cap of out_rows.
    """
    if layer_weight_bytes <= weight_staging_bytes:
        return 1
    row_bytes = max(act_elem_bytes * max(out_cols, 1), 1)
    rows_per_pass = max(weight_staging_bytes // row_bytes, 1)
    return min(out_rows, _ceil_div(out_rows, rows_per_pass))


def _layer_gemm_stats(
    shape: ModelShape,
    m: int,
    npu: NPUConfig,
) -> tuple[int, float, float]:
    """Per-layer FLOPs, compute cycles, mean util over major GEMMs.

    GEMMs modeled (attention projections per ``ModelShape.attn_proj_shapes`` + SwiGLU):
      Q:  (M, H, q_dim)
      K:  (M, H, kv_dim)
      V:  (M, H, kv_dim)
      O:  (M, q_dim, H)
      gate/up/down × (n_shared + top_k)   # MoE: active experts only
    Attention score/value matmuls added by caller (depend on seq).

    MoE honesty: FLOPs use top_k (+shared), not all E (balanced routing).
    """
    h = shape.hidden
    # v0.30: attention projections from the shape (GQA: Q, K, V, O as before;
    # MLA: q_a/q_b, kv_a/kv_b, o; low-rank Q/O for DeepSeek-V4-style configs).
    attn_gemms = [(m, k, n) for _name, k, n in shape.attn_proj_shapes()]

    def _stats(gemms: list[tuple[int, int, int]]) -> tuple[int, float, float]:
        fl = 0
        cy = 0.0
        ua = 0.0
        for mm, kk, nn in gemms:
            fl += gemm_flops(mm, kk, nn)
            cy += gemm_compute_cycles(mm, kk, nn, npu)
            ua += gemm_utilization(mm, kk, nn, npu)
        return fl, cy, ua / len(gemms)

    f = shape.intermediate
    n_ffn = shape.n_shared_experts + shape.top_k
    gemms = list(attn_gemms)
    for _ in range(n_ffn):
        gemms.extend([(m, h, f), (m, h, f), (m, f, h)])
    flops, cycles, mean_util = _stats(gemms)
    nd = shape.n_dense_layers if shape.is_moe else 0
    if nd > 0:
        # Dense-prefix layers (HF first_k_dense_replace): one dense SwiGLU of
        # width dense_intermediate. Return the layer-weighted average.
        fd = shape.dense_intermediate or f
        dg = list(attn_gemms) + [(m, h, fd), (m, h, fd), (m, fd, h)]
        dfl, dcy, du = _stats(dg)
        L = shape.n_layers
        nm = L - nd
        flops = (nd * dfl + nm * flops) // L
        cycles = (nd * dcy + nm * cycles) / L
        mean_util = (nd * du + nm * mean_util) / L
    return flops, cycles, mean_util


def _attn_score_flops(shape: ModelShape, m: int, seq_ctx: int) -> int:
    """QK^T + Attn·V FLOPs for `m` queries against `seq_ctx` keys (causal approx).

    Per head: QK = 2*m*seq_ctx*d, AV = 2*m*seq_ctx*d → 4*m*seq_ctx*d per head.
    Total: 4 * n_heads * m * seq_ctx * head_dim
    For prefill with causal mask we approximate with seq_ctx = S and m = S
    (full rectangle — slightly pessimistic vs triangular 0.5 factor; we apply
    0.5 for prefill when m == seq_ctx).
    """
    return 4 * shape.n_heads * m * seq_ctx * shape.head_dim


def _resolve(
    shape: ModelShape,
    sram: SRAMConfig,
    seq_for_kv: int,
    batch: int,
    policy: SramPolicy | None = None,
) -> SRAMPartitions:
    kv_needed = shape.kv_bytes_per_token() * seq_for_kv * batch
    return sram.resolve_partitions(
        shape, kv_bytes_needed=kv_needed, policy=policy
    )


def _weight_accounting(
    shape: ModelShape,
    parts: SRAMPartitions,
    *,
    m: int,
    act_b: int,
    pin_weights: bool,
    steady_state_decode: bool,
    w_layer_stream: int | None = None,
) -> tuple[int, WeightPath, bool, int]:
    """Return (weight_bytes_dram, path, dbl_eligible, resident_layers).

    Prefill / cold-start: always pay L * (per-layer stream) to fill residents.
    Decode steady-state: (L − R) * per-miss-layer stream; 0 if R ≥ L.
    """
    # Stream unit = active layer (attn + shared + top_k FFN). Dense: = stored.
    w_layer = shape.stream_weight_bytes_per_layer() if w_layer_stream is None else int(w_layer_stream)
    w_stored = shape.weight_bytes_per_layer()
    L = shape.n_layers
    R = max(0, min(parts.resident_layers, L))
    w_part = parts.weight_staging_bytes
    body_w = L * w_stored  # pin / capacity: all experts
    dbl = bool(parts.double_buffer_eligible and R == 0)

    if pin_weights and body_w <= w_part:
        if steady_state_decode:
            return 0, "sram_pinned_body", False, L
        return body_w, "sram_pinned_body", False, L

    # Staging capacity available for miss-layer streaming:
    # when R≥1, leftover after R*W_layer (< W_layer) may force tiles on misses.
    if R >= 1:
        leftover = w_part - R * w_layer
        stage_for_miss = max(leftover, w_part // 4 if leftover == 0 else leftover)
        # Prefer a honest small tile buffer for miss layers
        if leftover < w_layer:
            stage_for_miss = max(leftover, 1) if leftover > 0 else max(w_part // (R + 1), 1)
    else:
        stage_for_miss = w_part

    tiles = estimate_output_tiles(
        out_rows=m,
        out_cols=shape.hidden,
        weight_staging_bytes=stage_for_miss if R >= 1 else w_part,
        layer_weight_bytes=w_layer,
        act_elem_bytes=act_b,
    )
    # When R≥1 and leftover < W_layer, miss layers still counted once each
    # for tiles==1 decode (M=1); tiling only amplifies when stage tiny + large M.
    per_miss = weight_dram_bytes_for_layer(w_layer, stage_for_miss if R >= 1 else w_part, tiles)

    if R >= L and L > 0:
        # Full on-die resident
        if steady_state_decode:
            return 0, "on_die_resident", False, R
        # Prefill cold-start: load all layers once into resident SRAM
        return body_w, "on_die_resident", False, R

    if R >= 1:
        miss = L - R
        if steady_state_decode:
            # Steady state: only miss layers stream each token
            # Use once-per-layer for miss when decode M small (tiles forced ≥1);
            # if stage_for_miss >= W_layer somehow, still once; else allow tiles.
            if stage_for_miss >= w_layer:
                w_bytes = miss * w_layer
            else:
                # Decode M=1 typically tiles=1 via estimate when? layer doesn't fit
                # → tiles may be >1. For honesty on M=1 with tiny leftover:
                # still stream each miss layer once (sequential layer eval with
                # a small rolling tile buffer over one layer's worth of reads
                # equals W_layer bytes once). Cap at once-per-layer for M=1.
                if m <= 1:
                    w_bytes = miss * w_layer
                else:
                    w_bytes = miss * per_miss
            return w_bytes, "hbm_stream_miss_layers", False, R
        # Prefill cold-start: load residents once + stream miss once
        return body_w, "hbm_stream_miss_layers", False, R

    # R == 0: staging-only — full model stream every phase call
    w_bytes = L * weight_dram_bytes_for_layer(w_layer, w_part, tiles)
    if tiles > 1:
        path: WeightPath = "hbm_stream_tiled"
    else:
        path = "hbm_stream_all_layers"
    return w_bytes, path, dbl, 0


def prefill_traffic(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    prompt_len: int,
    batch: int = 1,
    pin_weights: bool = False,
    sram_policy: SramPolicy | None = None,
    lm_head: bool = False,
) -> PhaseTraffic:
    """Prefill: large-M GEMMs, M = batch * prompt_len. Cold-start weight load.

    ``lm_head`` (v0.31, workbench on): add the LM head GEMM for the last prompt
    token of each sequence and its V·H weight read."""
    if prompt_len < 0 or batch < 0:
        raise ValueError("prompt_len and batch must be non-negative")
    m = batch * prompt_len
    act_b = shape.act_bits // 8
    w_layer = shape.stream_weight_bytes_per_layer()
    parts = _resolve(shape, sram, prompt_len, batch, policy=sram_policy)

    weight_bytes, w_path, dbl, R = _weight_accounting(
        shape,
        parts,
        m=m,
        act_b=act_b,
        pin_weights=pin_weights,
        steady_state_decode=False,
    )

    # KV write-through for all prompt tokens (populate external cache)
    kv_write = shape.kv_bytes_per_token() * prompt_len * batch
    kv_read = 0
    kv_bytes = kv_write

    # Activation DRAM vs act partition
    act_ws = m * shape.hidden * act_b
    if act_ws > parts.act_bytes:
        act_bytes = shape.n_layers * 2 * act_ws  # spill read+write per layer
    else:
        act_bytes = 0

    # v0.31: per-layer stats are identical across layers → compute once
    L = shape.n_layers
    lf, lc, lu = _layer_gemm_stats(shape, m, npu)
    attn_f = _attn_score_flops(shape, m, prompt_len) // 2
    attn_cycles = attn_f / max(npu.peak_flops_per_cycle(), 1)
    flops = L * (lf + attn_f)
    cycles = L * (lc + attn_cycles)
    mean_util = lu
    # v0.31: LM head for the last prompt token of each sequence (M = batch)
    head_b, hf, hc = _lm_head_cost(shape, batch, npu) if lm_head else (0, 0, 0.0)
    flops += hf
    cycles += hc
    weight_bytes += head_b

    return PhaseTraffic(
        name="prefill",
        flops=flops,
        weight_bytes_dram=weight_bytes,
        kv_bytes_dram=kv_bytes,
        act_bytes_dram=act_bytes,
        compute_cycles=cycles,
        mean_gemm_util=mean_util,
        m_eff=m,
        weight_path=w_path,
        partitions=parts,
        kv_bytes_read=kv_read,
        kv_bytes_write=kv_write,
        double_buffer_eligible=dbl,
        layer_weight_bytes=w_layer,
        resident_layers=R,
        head_weight_bytes=head_b,
    
        head_flops=hf,
        head_cycles=hc,
    )


def _lm_head_cost(shape: ModelShape, m: int, npu: NPUConfig) -> tuple[int, int, float]:
    """(weight bytes, FLOPs, cycles) of the LM head GEMM (m, H, V) — v0.31.

    The head matrix (V·H) is read once per phase call (decode: once per step)."""
    if m <= 0:
        return 0, 0, 0.0
    b = shape.lm_head_params() * shape.weight_bits // 8
    k, n = shape.hidden, shape.vocab
    return b, gemm_flops(m, k, n), float(gemm_compute_cycles(m, k, n, npu))


def decode_traffic(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    seq_len: int,
    batch: int = 1,
    pin_weights: bool = False,
    sram_policy: SramPolicy | None = None,
    tokens_per_seq: int = 1,
    lm_head: bool = False,
) -> PhaseTraffic:
    """One decode step (steady-state): M = batch × tokens_per_seq.

    Resident R layers skip HBM. v0.31:
      * MoE weight stream is batch-aware: expected touched experts for the
        step's M tokens (「假设」uniform routing, ``ModelShape.experts_touched``).
      * ``lm_head``: the LM head (V·H) is read once per step and its GEMM
        (M, H, V) is counted (workbench / UI / Pareto on; raw engine default off
        so examples/handcheck.md arithmetic is unchanged).
      * ``tokens_per_seq`` = 1 + spec_k for a speculative verify step: M grows,
        KV context is read once per sequence, k+1 KV entries are written.
    """
    if seq_len < 0 or batch < 0:
        raise ValueError("seq_len and batch must be non-negative")
    q = max(int(tokens_per_seq), 1)
    m = batch * q
    act_b = shape.act_bits // 8
    w_layer = shape.stream_weight_bytes_per_layer_tokens(m)
    parts = _resolve(shape, sram, seq_len, batch, policy=sram_policy)

    weight_bytes, w_path, dbl, R = _weight_accounting(
        shape,
        parts,
        m=m,
        act_b=act_b,
        pin_weights=pin_weights,
        steady_state_decode=True,
        w_layer_stream=w_layer,
    )

    # KV: capacity-derived read hit; always write-through new token(s)
    kv_full = shape.kv_bytes_per_token() * seq_len * batch
    kv_write = shape.kv_bytes_per_token() * batch * q
    if parts.kv_scratch_bytes >= kv_full and kv_full > 0:
        kv_read = 0
    else:
        kv_read = kv_full
    kv_bytes = kv_read + kv_write

    act_ws = m * shape.hidden * act_b
    if act_ws > parts.act_bytes:
        act_bytes = shape.n_layers * 2 * act_ws
    else:
        act_bytes = 0

    L = shape.n_layers
    lf, lc, lu = _layer_gemm_stats(shape, m, npu)
    attn_f = _attn_score_flops(shape, m, seq_len)
    attn_cycles = attn_f / max(npu.peak_flops_per_cycle(), 1)
    flops = L * (lf + attn_f)
    cycles = L * (lc + attn_cycles)
    mean_util = lu
    head_b, hf, hc = _lm_head_cost(shape, m, npu) if lm_head else (0, 0, 0.0)
    flops += hf
    cycles += hc
    weight_bytes += head_b

    return PhaseTraffic(
        name="decode",
        flops=flops,
        weight_bytes_dram=weight_bytes,
        kv_bytes_dram=kv_bytes,
        act_bytes_dram=act_bytes,
        compute_cycles=cycles,
        mean_gemm_util=mean_util,
        m_eff=m,
        weight_path=w_path,
        partitions=parts,
        kv_bytes_read=kv_read,
        kv_bytes_write=kv_write,
        double_buffer_eligible=dbl,
        layer_weight_bytes=w_layer,
        resident_layers=R,
        head_weight_bytes=head_b,
        tokens_per_seq=q,
    
        head_flops=hf,
        head_cycles=hc,
    )


def contend(
    traffic: PhaseTraffic,
    mem: ExternalMemory,
) -> ContentionResult:
    """Map byte streams to time under serialize / fair-share / lower-bound."""
    bw = mem.effective_bandwidth_Bps()
    if bw <= 0:
        raise ValueError("effective bandwidth must be positive")
    tw = traffic.weight_bytes_dram / bw
    tk = traffic.kv_bytes_dram / bw
    ta = traffic.act_bytes_dram / bw
    t_ser = tw + tk + ta
    t_fair = max(2.0 * tw, 2.0 * tk) + ta
    t_lo = max(tw, tk) + ta
    return ContentionResult(
        t_weight_s=tw,
        t_kv_s=tk,
        t_act_s=ta,
        t_serialize_s=t_ser,
        t_share_fair_s=t_fair,
        t_lower_bound_s=t_lo,
        bandwidth_Bps=bw,
    )
