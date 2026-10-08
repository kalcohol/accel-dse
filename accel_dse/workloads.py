"""Video (DiT-like) and protein workload shapes + phase programs.

These are **illustrative workload shapes** mapped onto the same single-card
NPU GEMM / roofline engine used for LLM inference. They do **not** claim any
vendor model quality, checkpoint dims, or calibrated silicon timing.

Honesty:
  - Labels: assumed / uncalibrated absolute numbers; derived FLOPs/bytes/time.
  - Approximations away: Softmax/LN/RoPE/VAE/decode head; diffusion sampler
    bookkeeping; MSA attention depth; Evoformer triangle ops (protein pair
    update is a single L×L GEMM family, not AlphaFold).
  - DiT: full spatial-temporal self-attn recomputed each denoise step
    (no cross-step KV cache — common DiT pattern). Documented below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .evaluate import EvalConfig, PhaseResult, _phase_result
from .memory import ExternalMemory, SRAMConfig, SRAMPartitions
from .npu import NPUConfig, gemm_compute_cycles, gemm_flops, gemm_utilization
from .traffic import PhaseTraffic


# ---------------------------------------------------------------------------
# Video — DiT-like diffusion transformer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoShape:
    """Illustrative DiT / latent-video diffusion transformer.

    Pipeline (phase program):
      patchify latent frames → transformer blocks (self-attn + MLP)
      × N_denoise forward passes (each denoise step = one full forward).

    Tokens:
      T = n_frames * (latent_h // patch) * (latent_w // patch)

    Attention (assumed DiT default):
      Full self-attn over all T tokens every denoise step — **recomputed**,
      not KV-cached across denoise steps (and not causal across frames unless
      noted). Temporal/causal cached variants are out of scope for this MVP.

    Approximated away:
      VAE encode/decode, noise schedule, CFG double-forward, patch embed
      detail, AdaLN / timestep MLP (folded into negligible vs body GEMMs).
    """

    name: str
    n_frames: int
    latent_h: int
    latent_w: int
    patch_size: int
    hidden: int
    n_layers: int
    n_heads: int
    head_dim: int
    intermediate: int
    n_denoise: int
    batch: int = 1
    weight_bits: int = 16
    act_bits: int = 16

    def __post_init__(self) -> None:
        if self.n_frames < 1 or self.n_layers < 1 or self.n_denoise < 1:
            raise ValueError("n_frames, n_layers, n_denoise must be >= 1")
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.latent_h % self.patch_size or self.latent_w % self.patch_size:
            raise ValueError("latent_h/w must be divisible by patch_size")
        if self.n_heads < 1 or self.head_dim < 1:
            raise ValueError("heads/head_dim must be positive")
        if self.head_dim * self.n_heads != self.hidden:
            pass  # allow toy mismatch
        if self.weight_bits <= 0 or self.act_bits <= 0:
            raise ValueError("bit widths must be positive")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")

    @property
    def n_tokens(self) -> int:
        """Tokens after patchify: F * (H/p) * (W/p)."""
        return (
            self.n_frames
            * (self.latent_h // self.patch_size)
            * (self.latent_w // self.patch_size)
        )

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    def attn_weight_params_per_layer(self) -> int:
        """Q+K+V+O (full attention; no GQA in this illustrative DiT)."""
        h, qd = self.hidden, self.q_dim
        return h * qd + h * qd + h * qd + qd * h  # = 4 * H * H when qd=H

    def mlp_weight_params_per_layer(self) -> int:
        """SwiGLU-style MLP: gate+up+down (same counting as LLM FFN)."""
        h, f = self.hidden, self.intermediate
        return 2 * h * f + f * h

    def weight_params_per_layer(self) -> int:
        return self.attn_weight_params_per_layer() + self.mlp_weight_params_per_layer()

    def total_weight_params(self) -> int:
        """Body + patch embed (approx VAE-latent → H) + final proj."""
        patch_in = 4 * self.patch_size * self.patch_size  # assume 4-ch latent
        embed = patch_in * self.hidden + self.hidden * patch_in
        return self.n_layers * self.weight_params_per_layer() + embed

    def weight_bytes(self) -> int:
        return self.total_weight_params() * self.weight_bits // 8

    def weight_bytes_per_layer(self) -> int:
        return self.weight_params_per_layer() * self.weight_bits // 8

    def act_bytes_working(self) -> int:
        """Activation working set ≈ B * T * H * act_bytes (residual stream)."""
        return self.batch * self.n_tokens * self.hidden * (self.act_bits // 8)

    def attn_score_bytes_peak(self) -> int:
        """Peak attn score buffer (illustrative; often tiled away).

        B * n_heads * T * T * act_bytes — can dominate for large T.
        """
        return (
            self.batch
            * self.n_heads
            * self.n_tokens
            * self.n_tokens
            * (self.act_bits // 8)
        )

    def with_denoise(self, n_denoise: int) -> "VideoShape":
        return replace(self, n_denoise=int(n_denoise))

    def summary(self) -> str:
        p = self.total_weight_params()
        ps = f"{p/1e9:.3f}B" if p >= 1e9 else f"{p/1e6:.3f}M"
        return (
            f"{self.name}: F={self.n_frames} lat={self.latent_h}x{self.latent_w} "
            f"patch={self.patch_size} → T={self.n_tokens} "
            f"L={self.n_layers} H={self.hidden} heads={self.n_heads} "
            f"F_mlp={self.intermediate} N_denoise={self.n_denoise} B={self.batch} "
            f"| params≈{ps} W≈{self.weight_bytes()/1e9:.3f}GB "
            f"(W={self.weight_bits}/A={self.act_bits}-bit, assumed; "
            f"full attn recomputed each step, no denoise KV cache)"
        )


# Illustrative DiT-video: ~0.7B-class body, 16 frames, 32×32 latent, patch=2 → T=4096
# NOT a Sora/CogVideo/Latte claim — placeholder for denoise-step DSE.
ILLUSTRATIVE_DIT_VIDEO = VideoShape(
    name="illustrative_dit_video",
    n_frames=16,
    latent_h=32,
    latent_w=32,
    patch_size=2,
    hidden=1024,
    n_layers=28,
    n_heads=16,
    head_dim=64,
    intermediate=4096,
    n_denoise=50,
    batch=1,
    weight_bits=16,
    act_bits=16,
)

# Larger slow generative DiT: bigger T (more frames + resolution), more layers,
# N_denoise in 50–100 → TTFC minutes-scale on sku_100t (~3 min @ N=50, ~5–6 min @ N=80).
# Illustrative placeholder for "100T narrative" heavy generative models — NOT a
# MiniMax-H3 (or any vendor) checkpoint claim.
ILLUSTRATIVE_LARGE_DIT = VideoShape(
    name="illustrative_large_dit",
    n_frames=32,
    latent_h=64,
    latent_w=64,
    patch_size=2,
    hidden=1536,
    n_layers=40,
    n_heads=24,
    head_dim=64,
    intermediate=6144,
    n_denoise=80,
    batch=1,
    weight_bits=16,
    act_bits=16,
)

# Tiny video shape for unit tests / hand arithmetic
TOY_VIDEO = VideoShape(
    name="toy_video",
    n_frames=2,
    latent_h=4,
    latent_w=4,
    patch_size=2,
    hidden=64,
    n_layers=2,
    n_heads=4,
    head_dim=16,
    intermediate=128,
    n_denoise=4,
    batch=1,
    weight_bits=16,
    act_bits=16,
)


@dataclass(frozen=True)
class VideoResult:
    """One video-generation evaluation (clip forward × N_denoise)."""

    shape: VideoShape
    npu: NPUConfig
    sram: SRAMConfig
    mem: ExternalMemory
    cfg: EvalConfig
    forward: PhaseResult  # one denoise-step forward
    n_denoise: int
    oom: bool
    capacity_needed_bytes: int

    @property
    def time_to_first_clip_s(self) -> float:
        """TTFT-like: N_denoise × roofline forward time."""
        return self.n_denoise * self.forward.t_roofline_s

    @property
    def time_per_frame_s(self) -> float:
        return self.time_to_first_clip_s / max(self.shape.n_frames, 1)

    @property
    def frames_per_s(self) -> float:
        t = self.time_to_first_clip_s
        if t <= 0:
            return float("inf")
        return self.shape.n_frames / t

    @property
    def clips_per_s(self) -> float:
        t = self.time_to_first_clip_s
        if t <= 0:
            return float("inf")
        return 1.0 / t

    def peak_tops(self) -> float:
        return self.npu.peak_tops_at(self.cfg.frequency_hz)

    def summary_lines(self) -> list[str]:
        s = self.shape
        tops = self.peak_tops()
        tops_s = f"{tops:.2f}" if tops < 10 else f"{tops:.1f}"
        oom_s = " OOM" if self.oom else ""
        return [
            f"=== VIDEO {s.name} | {self.mem.kind} | "
            f"SRAM={self.sram.capacity_bytes/2**20:.1f}MiB | "
            f"PE={self.npu.rows}x{self.npu.cols}×{self.npu.n_engines} "
            f"@ {self.cfg.frequency_hz/1e9:.2f}GHz → {tops_s} TOPS "
            f"(assumed; uncalibrated){oom_s} ===",
            f"  shape: {s.summary()}",
            f"  mem:   {self.mem.summary()}",
            f"  forward (1 denoise step): {self.forward.t_roofline_s*1e3:.4f} ms "
            f"[{self.forward.wall}-bound]  "
            f"compute={self.forward.t_compute_s*1e3:.4f} ms  "
            f"mem({self.cfg.contention_mode})={self.forward.t_memory_s*1e3:.4f} ms",
            f"    DRAM: W={self.forward.traffic.weight_bytes_dram} "
            f"KV={self.forward.traffic.kv_bytes_dram} "
            f"Act={self.forward.traffic.act_bytes_dram}  "
            f"FLOPs={self.forward.traffic.flops}  "
            f"mean_GEMM_util={self.forward.traffic.mean_gemm_util:.4f} "
            f"(M={self.forward.traffic.m_eff})",
            f"    note: full attn recomputed; kv_bytes here = 0 "
            f"(no denoise-step KV cache)",
            f"  time_to_first_clip ≈ N_denoise×forward = "
            f"{self.n_denoise}×{self.forward.t_roofline_s*1e3:.4f} = "
            f"{self.time_to_first_clip_s*1e3:.4f} ms",
            f"  throughput: {self.frames_per_s:.4f} frames/s  "
            f"({self.clips_per_s:.4f} clips/s)  "
            f"time/frame={self.time_per_frame_s*1e3:.4f} ms",
            f"  capacity_needed≈{self.capacity_needed_bytes/1e9:.3f} GB "
            f"(W+act; attn-score peak not all resident) "
            f"vs mem.cap={self.mem.capacity_bytes/1e9:.1f} GB",
        ]


def _video_forward_traffic(
    shape: VideoShape,
    npu: NPUConfig,
    sram: SRAMConfig,
) -> PhaseTraffic:
    """One DiT forward (one denoise step): stream all layers, full attn."""
    m = shape.batch * shape.n_tokens
    w_layer = shape.weight_bytes_per_layer()
    L = shape.n_layers
    h = shape.hidden
    qd = shape.q_dim
    f = shape.intermediate
    T = shape.n_tokens

    # Staging-only by default (R=0): full weight stream each forward.
    # Reuse a simple partition: weight gets most of SRAM as staging.
    cap = sram.capacity_bytes
    w_stage = min(cap * 3 // 4, max(w_layer, cap // 2))
    act_part = max(cap - w_stage, 0) // 2
    kv_part = cap - w_stage - act_part
    parts = SRAMPartitions(
        weight_staging_bytes=w_stage,
        kv_scratch_bytes=kv_part,
        act_bytes=act_part,
        resident_layers=0,
        double_buffer_eligible=(w_stage >= 2 * w_layer),
        layer_weight_bytes=w_layer,
        n_layers=L,
        policy="weight_resident",
    )
    # Weight DRAM: stream all layers once per forward (staging R=0)
    weight_bytes = L * w_layer + (
        4 * shape.patch_size * shape.patch_size * h * (shape.weight_bits // 8) * 2
    )

    # No KV cache across denoise steps (DiT recompute)
    kv_bytes = 0

    act_ws = shape.act_bytes_working()
    if act_ws > parts.act_bytes:
        act_bytes = L * 2 * act_ws
    else:
        act_bytes = 0

    flops = 0
    cycles = 0.0
    util_sum = 0.0
    n_gemms = 0
    for _ in range(L):
        gemms = [
            (m, h, qd),  # Q
            (m, h, qd),  # K
            (m, h, qd),  # V
            (m, qd, h),  # O
            (m, h, f),   # gate
            (m, h, f),   # up
            (m, f, h),   # down
        ]
        for mm, kk, nn in gemms:
            flops += gemm_flops(mm, kk, nn)
            cycles += gemm_compute_cycles(mm, kk, nn, npu)
            util_sum += gemm_utilization(mm, kk, nn, npu)
            n_gemms += 1
        # Full self-attn QK^T + Attn·V: B independent T×T maps (no causal 1/2).
        # DiT recomputes every denoise step — no KV cache.
        attn_f = 4 * shape.n_heads * shape.batch * T * T * shape.head_dim
        flops += attn_f
        cycles += attn_f / max(npu.peak_flops_per_cycle(), 1)

    mean_util = util_sum / max(n_gemms, 1)
    return PhaseTraffic(
        name="video_forward",
        flops=flops,
        weight_bytes_dram=weight_bytes,
        kv_bytes_dram=kv_bytes,
        act_bytes_dram=act_bytes,
        compute_cycles=cycles,
        mean_gemm_util=mean_util,
        m_eff=m,
        weight_path="hbm_stream_all_layers",
        partitions=parts,
        kv_bytes_read=0,
        kv_bytes_write=0,
        double_buffer_eligible=parts.double_buffer_eligible,
        layer_weight_bytes=w_layer,
        resident_layers=0,
    )


def evaluate_video(
    shape: VideoShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig | None = None,
) -> VideoResult:
    """Evaluate DiT-like video generation on the single-card NPU model."""
    cfg = cfg or EvalConfig(frequency_hz=1.0e9, contention_mode="serialize")
    traffic = _video_forward_traffic(shape, npu, sram)
    forward = _phase_result(traffic, mem, cfg)
    needed = shape.weight_bytes() + shape.act_bytes_working()
    oom = needed > mem.capacity_bytes
    return VideoResult(
        shape=shape,
        npu=npu,
        sram=sram,
        mem=mem,
        cfg=cfg,
        forward=forward,
        n_denoise=shape.n_denoise,
        oom=oom,
        capacity_needed_bytes=needed,
    )


# ---------------------------------------------------------------------------
# Protein — transformer encoder + optional L×L pair GEMM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinShape:
    """Illustrative protein structure / sequence encoder MVP.

    Phase program:
      1. Transformer encoder on L residue tokens (ESM-style).
      2. Optional per-layer L×L pair-update GEMM (pair_dim > 0).

    This is a **heavy approximation** of Evoformer / AF2 pair representation:
      - Real Evoformer has triangle multiplicative/attention updates, MSA
        row/column attention, etc. Here we only count:
          (a) standard seq self-attn + MLP on L tokens
          (b) one pair transition GEMM: (L², C_z) × (C_z, C_z) per layer
          (c) one seq→pair outer-product-like GEMM: FLOPs ≈ 2·L·L·C_z
      - MSA depth is a label only in MVP (msa_depth=1 → single sequence).

    Memory stress:
      pair activations ≈ L² · pair_dim · act_bytes — explodes with L.
    """

    name: str
    seq_len: int
    hidden: int
    n_layers: int
    n_heads: int
    head_dim: int
    intermediate: int
    pair_dim: int = 0
    msa_depth: int = 1
    batch: int = 1
    weight_bits: int = 16
    act_bits: int = 16

    def __post_init__(self) -> None:
        if self.seq_len < 1 or self.n_layers < 1:
            raise ValueError("seq_len and n_layers must be >= 1")
        if self.n_heads < 1 or self.head_dim < 1 or self.hidden < 1:
            raise ValueError("dims must be positive")
        if self.pair_dim < 0:
            raise ValueError("pair_dim must be non-negative")
        if self.msa_depth < 1:
            raise ValueError("msa_depth must be >= 1")
        if self.weight_bits <= 0 or self.act_bits <= 0:
            raise ValueError("bit widths must be positive")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")

    @property
    def has_pair(self) -> bool:
        return self.pair_dim > 0

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    def attn_weight_params_per_layer(self) -> int:
        h, qd = self.hidden, self.q_dim
        return h * qd + h * qd + h * qd + qd * h

    def mlp_weight_params_per_layer(self) -> int:
        h, f = self.hidden, self.intermediate
        return 2 * h * f + f * h

    def pair_weight_params_per_layer(self) -> int:
        """Pair transition Linear(C_z, C_z) — illustrative."""
        if self.pair_dim <= 0:
            return 0
        return self.pair_dim * self.pair_dim

    def weight_params_per_layer(self) -> int:
        return (
            self.attn_weight_params_per_layer()
            + self.mlp_weight_params_per_layer()
            + self.pair_weight_params_per_layer()
        )

    def total_weight_params(self) -> int:
        """Body + residue embed (20 aa + gap ≈ 32) + optional pair init."""
        aa = 32
        embed = aa * self.hidden
        pair_init = self.pair_dim * self.pair_dim if self.pair_dim > 0 else 0
        return self.n_layers * self.weight_params_per_layer() + embed + pair_init

    def weight_bytes(self) -> int:
        return self.total_weight_params() * self.weight_bits // 8

    def weight_bytes_per_layer(self) -> int:
        return self.weight_params_per_layer() * self.weight_bits // 8

    def pair_activation_bytes(self) -> int:
        """L×L×C_z pair activation footprint (can dominate capacity)."""
        if self.pair_dim <= 0:
            return 0
        return (
            self.batch
            * self.seq_len
            * self.seq_len
            * self.pair_dim
            * (self.act_bits // 8)
        )

    def seq_activation_bytes(self) -> int:
        return self.batch * self.seq_len * self.hidden * (self.act_bits // 8)

    def with_seq_len(self, seq_len: int) -> "ProteinShape":
        return replace(self, seq_len=int(seq_len))

    def summary(self) -> str:
        p = self.total_weight_params()
        ps = f"{p/1e9:.3f}B" if p >= 1e9 else f"{p/1e6:.3f}M"
        pair = (
            f" pair_dim={self.pair_dim} pair_act≈{self.pair_activation_bytes()/1e9:.3f}GB"
            if self.has_pair
            else " (seq-only, no pair)"
        )
        return (
            f"{self.name}: L_aa={self.seq_len} MSA_depth={self.msa_depth} "
            f"layers={self.n_layers} H={self.hidden} heads={self.n_heads} "
            f"F={self.intermediate}{pair} | params≈{ps} "
            f"W≈{self.weight_bytes()/1e6:.2f}MB "
            f"(W={self.weight_bits}/A={self.act_bits}-bit, assumed; "
            f"Evoformer heavily approximated)"
        )


# Illustrative protein: ESM-ish encoder + pair_dim=128 for L² stress.
# NOT an ESM/AlphaFold/OpenFold claim — placeholder for L / L² DSE.
ILLUSTRATIVE_PROTEIN_PAIR = ProteinShape(
    name="illustrative_protein_pair",
    seq_len=512,
    hidden=1024,
    n_layers=24,
    n_heads=16,
    head_dim=64,
    intermediate=4096,
    pair_dim=128,
    msa_depth=1,
    batch=1,
    weight_bits=16,
    act_bits=16,
)

TOY_PROTEIN = ProteinShape(
    name="toy_protein",
    seq_len=32,
    hidden=64,
    n_layers=2,
    n_heads=4,
    head_dim=16,
    intermediate=128,
    pair_dim=16,
    msa_depth=1,
    batch=1,
    weight_bits=16,
    act_bits=16,
)


@dataclass(frozen=True)
class ProteinResult:
    shape: ProteinShape
    npu: NPUConfig
    sram: SRAMConfig
    mem: ExternalMemory
    cfg: EvalConfig
    forward: PhaseResult
    oom: bool
    capacity_needed_bytes: int
    pair_bytes: int

    @property
    def time_per_sequence_s(self) -> float:
        return self.forward.t_roofline_s

    def peak_tops(self) -> float:
        return self.npu.peak_tops_at(self.cfg.frequency_hz)

    def summary_lines(self) -> list[str]:
        s = self.shape
        tops = self.peak_tops()
        tops_s = f"{tops:.2f}" if tops < 10 else f"{tops:.1f}"
        oom_s = " OOM" if self.oom else ""
        return [
            f"=== PROTEIN {s.name} | {self.mem.kind} | "
            f"SRAM={self.sram.capacity_bytes/2**20:.1f}MiB | "
            f"PE={self.npu.rows}x{self.npu.cols}×{self.npu.n_engines} "
            f"@ {self.cfg.frequency_hz/1e9:.2f}GHz → {tops_s} TOPS "
            f"(assumed; uncalibrated){oom_s} ===",
            f"  shape: {s.summary()}",
            f"  mem:   {self.mem.summary()}",
            f"  time/seq: {self.time_per_sequence_s*1e3:.4f} ms "
            f"[{self.forward.wall}-bound]  "
            f"compute={self.forward.t_compute_s*1e3:.4f} ms  "
            f"mem({self.cfg.contention_mode})={self.forward.t_memory_s*1e3:.4f} ms",
            f"    DRAM: W={self.forward.traffic.weight_bytes_dram} "
            f"KV={self.forward.traffic.kv_bytes_dram} "
            f"Act={self.forward.traffic.act_bytes_dram}  "
            f"FLOPs={self.forward.traffic.flops}  "
            f"mean_GEMM_util={self.forward.traffic.mean_gemm_util:.4f} "
            f"(M={self.forward.traffic.m_eff})",
            f"  pair_act_bytes={self.pair_bytes} "
            f"({self.pair_bytes/1e9:.4f} GB) ~ L²·C_z",
            f"  capacity_needed≈{self.capacity_needed_bytes/1e9:.3f} GB "
            f"(W+seq_act+pair_act) vs mem.cap={self.mem.capacity_bytes/1e9:.1f} GB",
        ]


def _protein_forward_traffic(
    shape: ProteinShape,
    npu: NPUConfig,
    sram: SRAMConfig,
) -> PhaseTraffic:
    """One protein forward: encoder on L + optional pair GEMMs."""
    L_aa = shape.seq_len
    m = shape.batch * L_aa
    w_layer = shape.weight_bytes_per_layer()
    nL = shape.n_layers
    h = shape.hidden
    qd = shape.q_dim
    f = shape.intermediate
    cz = shape.pair_dim

    cap = sram.capacity_bytes
    w_stage = min(cap * 3 // 4, max(w_layer, cap // 2))
    rem = max(cap - w_stage, 0)
    # Prefer keeping some pair scratch if pair exists
    pair_need = min(shape.pair_activation_bytes(), rem // 2) if cz > 0 else 0
    act_part = max(rem - pair_need, 0)
    parts = SRAMPartitions(
        weight_staging_bytes=w_stage,
        kv_scratch_bytes=pair_need,  # reuse kv slot as pair scratch label
        act_bytes=act_part,
        resident_layers=0,
        double_buffer_eligible=(w_stage >= 2 * w_layer),
        layer_weight_bytes=w_layer,
        n_layers=nL,
        policy="weight_resident",
    )

    weight_bytes = nL * w_layer + 32 * h * (shape.weight_bits // 8)

    # Seq self-attn has no persistent KV across sequences; within forward we
    # recompute (encoder). Count KV DRAM as 0 (scores tiled / not written back).
    kv_bytes = 0

    seq_act = shape.seq_activation_bytes()
    pair_act = shape.pair_activation_bytes()
    # Act DRAM: spill seq if needed; pair always counted as external traffic
    # when it does not fit in pair scratch (kv_scratch reuse).
    act_bytes = 0
    if seq_act > parts.act_bytes:
        act_bytes += nL * 2 * seq_act
    if pair_act > parts.kv_scratch_bytes:
        # Read+write pair tensor once per layer (conservative spill)
        act_bytes += nL * 2 * pair_act

    flops = 0
    cycles = 0.0
    util_sum = 0.0
    n_gemms = 0
    for _ in range(nL):
        gemms = [
            (m, h, qd),
            (m, h, qd),
            (m, h, qd),
            (m, qd, h),
            (m, h, f),
            (m, h, f),
            (m, f, h),
        ]
        for mm, kk, nn in gemms:
            flops += gemm_flops(mm, kk, nn)
            cycles += gemm_compute_cycles(mm, kk, nn, npu)
            util_sum += gemm_utilization(mm, kk, nn, npu)
            n_gemms += 1
        # Seq self-attn over L residues (full, encoder)
        attn_f = 4 * shape.n_heads * shape.batch * L_aa * L_aa * shape.head_dim
        flops += attn_f
        cycles += attn_f / max(npu.peak_flops_per_cycle(), 1)

        if cz > 0:
            # (1) seq→pair outer-product-like: FLOPs ≈ 2 * L * L * C_z * B
            op_f = 2 * shape.batch * L_aa * L_aa * cz
            flops += op_f
            cycles += op_f / max(npu.peak_flops_per_cycle(), 1)
            # (2) pair transition GEMM timed on NPU:
            #     (L², C_z) × (C_z, C_z)  →  M=L², K=C_z, N=C_z
            mm = shape.batch * L_aa * L_aa
            flops += gemm_flops(mm, cz, cz)
            cycles += gemm_compute_cycles(mm, cz, cz, npu)
            util_sum += gemm_utilization(mm, cz, cz, npu)
            n_gemms += 1

    mean_util = util_sum / max(n_gemms, 1)
    return PhaseTraffic(
        name="protein_forward",
        flops=flops,
        weight_bytes_dram=weight_bytes,
        kv_bytes_dram=kv_bytes,
        act_bytes_dram=act_bytes,
        compute_cycles=cycles,
        mean_gemm_util=mean_util,
        m_eff=m,
        weight_path="hbm_stream_all_layers",
        partitions=parts,
        kv_bytes_read=0,
        kv_bytes_write=0,
        double_buffer_eligible=parts.double_buffer_eligible,
        layer_weight_bytes=w_layer,
        resident_layers=0,
    )


def evaluate_protein(
    shape: ProteinShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig | None = None,
) -> ProteinResult:
    """Evaluate protein encoder (+ optional pair) on the single-card NPU model."""
    cfg = cfg or EvalConfig(frequency_hz=1.0e9, contention_mode="serialize")
    traffic = _protein_forward_traffic(shape, npu, sram)
    forward = _phase_result(traffic, mem, cfg)
    pair_b = shape.pair_activation_bytes()
    needed = shape.weight_bytes() + shape.seq_activation_bytes() + pair_b
    oom = needed > mem.capacity_bytes
    return ProteinResult(
        shape=shape,
        npu=npu,
        sram=sram,
        mem=mem,
        cfg=cfg,
        forward=forward,
        oom=oom,
        capacity_needed_bytes=needed,
        pair_bytes=pair_b,
    )


# ---------------------------------------------------------------------------
# Shape registry (LLM + video + protein)
# ---------------------------------------------------------------------------

def list_shapes() -> list[tuple[str, str, str]]:
    """Return (domain, name, one-line summary) for built-in + HF catalog shapes."""
    from .model_shape import (
        ILLUSTRATIVE_27B,
        ILLUSTRATIVE_MLA,
        ILLUSTRATIVE_MOE,
        TOY_SHAPE,
    )

    rows: list[tuple[str, str, str]] = [
        ("llm", TOY_SHAPE.name, TOY_SHAPE.summary()),
        ("llm", ILLUSTRATIVE_27B.name, ILLUSTRATIVE_27B.summary()),
        ("llm", ILLUSTRATIVE_MOE.name, ILLUSTRATIVE_MOE.summary()),
        ("llm", ILLUSTRATIVE_MLA.name, ILLUSTRATIVE_MLA.summary()),
        ("video", TOY_VIDEO.name, TOY_VIDEO.summary()),
        ("video", ILLUSTRATIVE_DIT_VIDEO.name, ILLUSTRATIVE_DIT_VIDEO.summary()),
        ("video", ILLUSTRATIVE_LARGE_DIT.name, ILLUSTRATIVE_LARGE_DIT.summary()),
        ("protein", TOY_PROTEIN.name, TOY_PROTEIN.summary()),
        ("protein", ILLUSTRATIVE_PROTEIN_PAIR.name, ILLUSTRATIVE_PROTEIN_PAIR.summary()),
    ]
    seen = {r[1] for r in rows}
    try:
        from .series import list_series
        for e in list_series():
            name = getattr(e.shape, "name", e.id)
            if name in seen:
                continue
            seen.add(name)
            rows.append((e.domain, name, e.shape.summary()))
    except Exception:
        pass
    return rows
