"""On-die SRAM (partitioned) and external memory (HBM or LPDDR) bandwidth models.

Bandwidth formula (derived from unit geometry; v0.29):
  peak_Bps (raw)  = n_units * (unit_width_bits / 8) * data_rate_GT/s * 1e9
  payload_Bps     = raw * payload_factor      # LPDDR6 = 256/288 = 8/9; else 1.0
  effective_Bps   = payload * efficiency      # efficiency is *assumed*

``n_channels`` is kept as the field name for backward compatibility but means
**units**: LPDDR packages / modules or HBM stacks (see ``unit_kind``). Use
``mem_catalog.make_spec`` for structured LPDDR5/5X/6 + HBM3/3E/4/4E configs.

SRAM partitions (capacities sum ≤ total):
  weight_partition — staging (R=0) OR persistent resident layers (R≥1)
  kv_scratch       — on-die KV window / attention working set
  act              — activation / residual scratch

Honesty (v0.5):
  Staging (1× or 2× W_layer) does NOT reduce decode HBM weight *bytes* —
  each layer is still read once per token. Double-buffer only enables optional
  weight_hide_factor (overlap). True byte savings require R ≥ 1 resident layers
  kept across tokens: steady-state decode W DRAM ≈ (L − R) * W_layer.

No fake PDK area/power. Capacities and rates are assumed presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .model_shape import ModelShape

MemKind = Literal["HBM", "LPDDR"]
SramPolicy = Literal["weight_resident", "kv_first", "balanced"]


@dataclass(frozen=True)
class SRAMPartitions:
    """Three-way split of on-die SRAM. Sum must be ≤ parent capacity.

    weight_staging_bytes is the weight *partition* budget:
      - R == 0: used as layer staging / tile buffer (no resident savings)
      - R >= 1: holds R full layers persistently (resident across tokens)
    """

    weight_staging_bytes: int
    kv_scratch_bytes: int
    act_bytes: int
    resident_layers: int = 0
    # Staging-only double-buffer (R==0 and staging >= 2*W_layer)
    double_buffer_eligible: bool = False
    layer_weight_bytes: int = 0
    n_layers: int = 0
    policy: SramPolicy = "weight_resident"

    def __post_init__(self) -> None:
        for name, v in (
            ("weight_staging_bytes", self.weight_staging_bytes),
            ("kv_scratch_bytes", self.kv_scratch_bytes),
            ("act_bytes", self.act_bytes),
            ("resident_layers", self.resident_layers),
        ):
            if v < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bytes(self) -> int:
        return (
            self.weight_staging_bytes
            + self.kv_scratch_bytes
            + self.act_bytes
        )

    @property
    def weight_partition_bytes(self) -> int:
        """Alias: weight side of the split (staging or resident)."""
        return self.weight_staging_bytes

    def summary(self) -> str:
        t = max(self.total_bytes, 1)
        r = self.resident_layers
        mode = (
            f"R={r}/{self.n_layers}"
            if self.n_layers > 0
            else f"R={r}"
        )
        return (
            f"W_part={self.weight_staging_bytes/2**20:.2f}MiB "
            f"({self.weight_staging_bytes/t:.0%}) [{mode}] "
            f"KV_scratch={self.kv_scratch_bytes/2**20:.2f}MiB "
            f"({self.kv_scratch_bytes/t:.0%}) "
            f"act={self.act_bytes/2**20:.2f}MiB "
            f"({self.act_bytes/t:.0%}) "
            f"policy={self.policy} dbl={self.double_buffer_eligible}"
        )


def _fix_sum(w: int, kv: int, act: int, cap: int) -> tuple[int, int, int]:
    used = w + kv + act
    if used < cap:
        act += cap - used
    elif used > cap:
        overflow = used - cap
        act = max(0, act - overflow)
        used = w + kv + act
        if used > cap:
            kv = max(0, kv - (used - cap))
            used = w + kv + act
            if used > cap:
                w = max(0, w - (used - cap))
    return w, kv, act


def _staging_only(
    cap: int,
    w_layer: int,
    kv_need: int,
    n_layers: int,
    policy: SramPolicy,
) -> SRAMPartitions:
    """R=0: staging / tile buffer — does NOT cut decode weight bytes."""
    if w_layer > 0 and (2 * w_layer) <= int(0.80 * cap):
        w_stage = 2 * w_layer
        dbl = True
    elif w_layer > 0 and w_layer <= int(0.80 * cap):
        w_stage = w_layer
        dbl = False
    else:
        w_stage = cap // 4
        dbl = False
    rem = cap - w_stage
    if kv_need > 0:
        kv = min(rem // 2, kv_need)
    else:
        kv = rem // 2
    act = rem - kv
    w_stage, kv, act = _fix_sum(w_stage, kv, act, cap)
    return SRAMPartitions(
        weight_staging_bytes=w_stage,
        kv_scratch_bytes=kv,
        act_bytes=act,
        resident_layers=0,
        double_buffer_eligible=dbl,
        layer_weight_bytes=w_layer,
        n_layers=n_layers,
        policy=policy,
    )


def derive_sram_partitions(
    capacity_bytes: int,
    layer_weight_bytes: int = 0,
    kv_bytes_needed: int = 0,
    n_layers: int = 0,
    policy: SramPolicy = "weight_resident",
) -> SRAMPartitions:
    """Capacity-derived partition policy (no user hit-rate).

    Policies (assumed):
      weight_resident (default): maximize R = floor(weight_budget / W_layer)
        when capacity ≥ W_layer; leftover → KV then act. Else staging-only.
      kv_first: satisfy KV working set first; remainder → maximize R / staging.
      balanced: after a small act floor, split remainder ~50/50 weight vs KV,
        then R = floor(weight_part / W_layer).

    R ≥ 1 means that many layers stay on-die across tokens (true byte savings).
    R = 0 with staging ≥ W_layer is staging-only: bytes still L*W_layer/token.
    """
    if capacity_bytes < 0:
        raise ValueError("capacity must be non-negative")
    if policy not in ("weight_resident", "kv_first", "balanced"):
        raise ValueError(f"unknown sram policy: {policy}")
    cap = capacity_bytes
    w_layer = max(layer_weight_bytes, 0)
    kv_need = max(kv_bytes_needed, 0)
    n_layers = max(n_layers, 0)

    # Tiny / no-layer: staging-only path
    if w_layer <= 0 or cap < w_layer:
        return _staging_only(cap, w_layer, kv_need, n_layers, policy)

    min_act = max(cap // 16, 0)

    if policy == "weight_resident":
        # Maximize R = floor(cap / W_layer) (clamp to L). Remainder → KV then act.
        # No min_act floor that would block R — full on-die when cap ≥ L*W_layer.
        r_max = cap // w_layer
        if n_layers > 0:
            r_max = min(r_max, n_layers)
        if r_max < 1:
            return _staging_only(cap, w_layer, kv_need, n_layers, policy)
        w_part = r_max * w_layer
        rem = cap - w_part
        if kv_need > 0:
            kv = min(rem, kv_need)
        else:
            kv = rem // 2 if rem > 0 else 0
        act = rem - kv
        w_part, kv, act = _fix_sum(w_part, kv, act, cap)
        r = w_part // w_layer
        if n_layers > 0:
            r = min(r, n_layers)
        return SRAMPartitions(
            weight_staging_bytes=w_part,
            kv_scratch_bytes=kv,
            act_bytes=act,
            resident_layers=r,
            double_buffer_eligible=False,
            layer_weight_bytes=w_layer,
            n_layers=n_layers,
            policy=policy,
        )

    if policy == "kv_first":
        # Satisfy KV first (up to need), leave at least one tile for weights
        min_w_tile = min(w_layer, cap // 4) if w_layer > 0 else cap // 4
        kv = min(kv_need, max(cap - min_w_tile - min_act, 0)) if kv_need > 0 else 0
        rem = cap - kv
        # From rem, maximize R (leave min_act)
        budget = max(rem - min_act, 0)
        r_max = budget // w_layer if w_layer > 0 else 0
        if n_layers > 0:
            r_max = min(r_max, n_layers)
        if r_max >= 1:
            w_part = r_max * w_layer
            act = rem - w_part
            w_part, kv, act = _fix_sum(w_part, kv, act, cap)
            r = min(w_part // w_layer, n_layers) if n_layers > 0 else w_part // w_layer
            return SRAMPartitions(
                weight_staging_bytes=w_part,
                kv_scratch_bytes=kv,
                act_bytes=act,
                resident_layers=r,
                double_buffer_eligible=False,
                layer_weight_bytes=w_layer,
                n_layers=n_layers,
                policy=policy,
            )
        # No resident room — staging from rem
        staging_cap = rem
        # Run staging split on staging_cap, then add kv back
        if w_layer > 0 and (2 * w_layer) <= int(0.80 * staging_cap):
            w_stage = 2 * w_layer
            dbl = True
        elif w_layer > 0 and w_layer <= int(0.80 * staging_cap):
            w_stage = w_layer
            dbl = False
        else:
            w_stage = max(staging_cap // 4, 0)
            dbl = False
        act = staging_cap - w_stage
        w_stage, kv, act = _fix_sum(w_stage, kv, act, cap)
        return SRAMPartitions(
            weight_staging_bytes=w_stage,
            kv_scratch_bytes=kv,
            act_bytes=act,
            resident_layers=0,
            double_buffer_eligible=dbl,
            layer_weight_bytes=w_layer,
            n_layers=n_layers,
            policy=policy,
        )

    # balanced: 50/50 weight vs KV after act floor
    act0 = min_act
    rem = cap - act0
    half = rem // 2
    if kv_need > 0:
        kv = min(half, kv_need)
    else:
        kv = half
    w_budget = rem - kv
    r_max = w_budget // w_layer if w_layer > 0 else 0
    if n_layers > 0:
        r_max = min(r_max, n_layers)
    if r_max >= 1:
        w_part = r_max * w_layer
        # leftover from w_budget goes to act (or kv if under-need)
        leftover = rem - w_part - kv
        if leftover < 0:
            kv = max(0, kv + leftover)
            leftover = rem - w_part - kv
        act = act0 + max(leftover, 0)
        w_part, kv, act = _fix_sum(w_part, kv, act, cap)
        r = min(w_part // w_layer, n_layers) if n_layers > 0 else w_part // w_layer
        return SRAMPartitions(
            weight_staging_bytes=w_part,
            kv_scratch_bytes=kv,
            act_bytes=act,
            resident_layers=r,
            double_buffer_eligible=False,
            layer_weight_bytes=w_layer,
            n_layers=n_layers,
            policy=policy,
        )
    return _staging_only(cap, w_layer, kv_need, n_layers, policy)


@dataclass(frozen=True)
class SRAMConfig:
    """On-die scratchpad with optional explicit partitions.

    If partitions is None, traffic/evaluate call derive_sram_partitions using
    the model shape (capacity-derived; no user hit-rate).
    """

    capacity_bytes: int
    n_banks: int = 1
    partitions: SRAMPartitions | None = None
    policy: SramPolicy = "weight_resident"

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError("SRAM capacity must be non-negative")
        if self.partitions is not None:
            if self.partitions.total_bytes > self.capacity_bytes:
                raise ValueError(
                    f"partition sum {self.partitions.total_bytes} exceeds "
                    f"capacity {self.capacity_bytes}"
                )

    def resolve_partitions(
        self,
        shape: "ModelShape | None" = None,
        kv_bytes_needed: int = 0,
        policy: SramPolicy | None = None,
    ) -> SRAMPartitions:
        if self.partitions is not None:
            return self.partitions
        # MoE: stage/reside on *active* (stream) layer footprint, not all-E.
        # Expert-parallel residency of unused experts is not modeled.
        w_layer = (
            shape.stream_weight_bytes_per_layer() if shape is not None else 0
        )
        n_layers = shape.n_layers if shape is not None else 0
        pol: SramPolicy = policy if policy is not None else self.policy
        return derive_sram_partitions(
            self.capacity_bytes,
            layer_weight_bytes=w_layer,
            kv_bytes_needed=kv_bytes_needed,
            n_layers=n_layers,
            policy=pol,
        )


@dataclass(frozen=True)
class ExternalMemory:
    """Pure HBM OR pure LPDDR (selectable). Per-card external DRAM.

    Multi-card TP/PP/C2C + KV fabric lives in ``scaleup.py`` (assumed link BW).
    Legacy ``chiplet_c2c_bw_Bps`` / ``ib_bw_Bps`` fields remain unused stubs.
    """

    kind: MemKind
    n_channels: int          # = n_units (LPDDR packages/modules or HBM stacks)
    width_bits: int          # per unit (x64 / x96 package, 128-bit module, 1024/2048 stack)
    data_rate_gt_s: float    # Giga-transfers per second per pin
    efficiency: float        # assumed sustained / payload (uncalibrated)
    capacity_bytes: int      # per-card DRAM capacity
    # unused legacy stubs (real model: scaleup.C2CLink / KvFabricLink)
    chiplet_c2c_bw_Bps: float | None = None
    ib_bw_Bps: float | None = None
    # v0.29 structured-catalog metadata (defaults keep ≤0.28 behaviour)
    payload_factor: float = 1.0  # LPDDR6 = 8/9 (in-band metadata + DBI/ECC)
    mem_type: str = ""           # LPDDR5 | LPDDR5X | LPDDR6 | HBM3 | HBM3E | HBM4 | HBM4E
    unit_kind: str = "unit"      # package | module | stack | unit (manual geometry)
    spec_tag: str = ""           # weakest provenance tag (mem_catalog.TAG_ORDER)
    capacity_nominal_GB: float = 0.0  # vendor GB (= GiB); 0 → capacity_bytes/1e9

    @property
    def n_units(self) -> int:
        return self.n_channels

    def peak_bandwidth_Bps(self) -> float:
        """Derived raw pin BW: units × (width/8) × rate × 1e9."""
        return (
            self.n_channels
            * (self.width_bits / 8.0)
            * self.data_rate_gt_s
            * 1e9
        )

    def payload_bandwidth_Bps(self) -> float:
        """Usable data BW = raw × payload_factor (LPDDR6 8/9)."""
        return self.peak_bandwidth_Bps() * self.payload_factor

    def effective_bandwidth_Bps(self) -> float:
        return self.payload_bandwidth_Bps() * self.efficiency

    def summary(self) -> str:
        peak = self.peak_bandwidth_Bps()
        eff = self.effective_bandwidth_Bps()
        typ = f" {self.mem_type}" if self.mem_type else ""
        pay = (
            f" payload×{self.payload_factor:.4f}"
            if abs(self.payload_factor - 1.0) > 1e-12
            else ""
        )
        return (
            f"{self.kind}{typ}: {self.n_channels}×{self.width_bits}b ({self.unit_kind}) × "
            f"{self.data_rate_gt_s} GT/s{pay} × eff={self.efficiency} "
            f"| raw={peak/1e9:.1f} GB/s eff={eff/1e9:.1f} GB/s "
            f"cap={self.capacity_bytes/1e9:.1f} GB (assumed)"
        )


# ---------------------------------------------------------------------------
# Assumed presets (uncalibrated — geometry-derived BW, efficiency assumed)
# ---------------------------------------------------------------------------

# v0.29: presets are real catalog configs (docs/research/memory_specs_2026-10 §6.3).
# ≤0.28 used HBM 8×1024b @5.2 GT/s / 96 GB (no such grade; 12 GB/stack does not
# exist) and LPDDR 8×64b @8.5 GT/s / 64 GB (8.5 is not a grade).
#
# HBM3E 8 stacks × 12-high × 24 Gb @ 9200 MT/s (MI355X / B300-class geometry):
#   raw = 8 × 1024 × 9200 / 8000 = 9420.8 GB/s; ×0.70 → 6594.6 GB/s (assumed eff)
#   capacity = 8 × 12 × 24 / 8 = 288 GB (vendor GB = 2^30 B)
HBM_PRESET = ExternalMemory(
    kind="HBM",
    n_channels=8,
    width_bits=1024,
    data_rate_gt_s=9.2,
    efficiency=0.70,
    capacity_bytes=288 * 2**30,
    mem_type="HBM3E",
    unit_kind="stack",
    spec_tag="vendor_shipping",
    capacity_nominal_GB=288.0,
)

# LPDDR5X 8 packages × x64 @ 8533 MT/s, 16 GB/package:
#   raw = 512 × 8533 / 8000 = 546.1 GB/s; ×0.70 → 382.3 GB/s; capacity 128 GB
LPDDR_PRESET = ExternalMemory(
    kind="LPDDR",
    n_channels=8,
    width_bits=64,
    data_rate_gt_s=8.533,
    efficiency=0.70,
    capacity_bytes=128 * 2**30,
    mem_type="LPDDR5X",
    unit_kind="package",
    spec_tag="vendor_shipping",
    capacity_nominal_GB=128.0,
)

# Tiny memory for hand-check (easy round numbers)
HANDCHECK_MEM = ExternalMemory(
    kind="HBM",
    n_channels=1,
    width_bits=64,
    data_rate_gt_s=2.0,
    efficiency=1.0,          # ideal for calculator
    capacity_bytes=10**9,
)

DEFAULT_SRAM = SRAMConfig(capacity_bytes=64 * 1024 * 1024)  # 64 MiB assumed
HANDCHECK_SRAM = SRAMConfig(capacity_bytes=4096)  # 4 KiB — forces streaming in toy


# ---------------------------------------------------------------------------
# Geometry knobs → ExternalMemory (presets become defaults)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemGeometry:
    """Manual DRAM geometry knobs (legacy / free-form; assumed unless derived BW).

    Prefer ``mem_catalog.make_spec`` for real LPDDR/HBM configs. Here
    ``n_channels`` = units of ``width_bits`` (packages / stacks).

    Bandwidth (derived)::

        peak_Bps = n_channels * (width_bits / 8) * data_rate_GTs * 1e9
        # optional: n_packages multiplies unit count (assumed mapping)
        effective_Bps = peak_Bps * efficiency       # payload_factor = 1

    ``n_ranks`` is accepted as an alias for ``n_channels`` when channels are
    unspecified (assumed 1:1 rank↔channel for DSE knobs — needs_lock later).
    """

    kind: MemKind
    n_channels: int
    width_bits: int
    data_rate_GTs: float
    efficiency: float
    capacity_GB: float
    n_packages: int = 1
    n_ranks: int | None = None  # informational / alias source

    def effective_channels(self) -> int:
        """Assumed: packages multiply channel count."""
        return int(self.n_channels) * max(int(self.n_packages), 1)

    def to_external_memory(self) -> ExternalMemory:
        ch = self.effective_channels()
        return ExternalMemory(
            kind=self.kind,
            n_channels=ch,
            width_bits=self.width_bits,
            data_rate_gt_s=float(self.data_rate_GTs),
            efficiency=float(self.efficiency),
            # v0.30: GB = vendor/JEDEC binary GB (2^30 B), same as the structured catalog
            capacity_bytes=int(round(self.capacity_GB * 2**30)),
            capacity_nominal_GB=float(self.capacity_GB),
        )

    def peak_bandwidth_GBps(self) -> float:
        return self.to_external_memory().peak_bandwidth_Bps() / 1e9

    def effective_bandwidth_GBps(self) -> float:
        return self.to_external_memory().effective_bandwidth_Bps() / 1e9

    def summary(self) -> str:
        mem = self.to_external_memory()
        return (
            f"MemGeometry {self.kind}: pkgs={self.n_packages} "
            f"ch={self.n_channels} (eff_ch={self.effective_channels()}) "
            f"× {self.width_bits}b × {self.data_rate_GTs} GT/s "
            f"× eff={self.efficiency} → peak={mem.peak_bandwidth_Bps()/1e9:.1f} "
            f"GB/s eff={mem.effective_bandwidth_Bps()/1e9:.1f} GB/s "
            f"cap={self.capacity_GB:.1f} GB [assumed geom]"
        )


def _preset_for(kind: MemKind) -> ExternalMemory:
    if kind == "HBM":
        return HBM_PRESET
    if kind == "LPDDR":
        return LPDDR_PRESET
    raise ValueError(f"unknown mem kind {kind!r}")


def memory_from_geometry(
    kind: MemKind | str = "HBM",
    *,
    n_channels: int | None = None,
    n_packages: int | None = None,
    n_ranks: int | None = None,
    data_rate_GTs: float | None = None,
    width_bits: int | None = None,
    efficiency: float | None = None,
    capacity_GB: float | None = None,
) -> ExternalMemory:
    """Build ExternalMemory from geometry knobs; missing fields ← preset defaults.

    Mapping assumed:
      - ``n_ranks`` fills ``n_channels`` when channels omitted (1:1 alias).
      - ``n_packages`` multiplies channel count (see MemGeometry).
    """
    k: MemKind = kind.upper() if isinstance(kind, str) else kind  # type: ignore[assignment]
    if k not in ("HBM", "LPDDR"):
        raise ValueError(f"mem kind must be HBM|LPDDR, got {kind!r}")
    base = _preset_for(k)
    ch = n_channels
    if ch is None and n_ranks is not None:
        ch = int(n_ranks)
    if ch is None:
        ch = base.n_channels
    pkgs = 1 if n_packages is None else int(n_packages)
    if pkgs < 1:
        raise ValueError("n_packages must be >= 1")
    geom = MemGeometry(
        kind=k,
        n_channels=int(ch),
        width_bits=int(width_bits if width_bits is not None else base.width_bits),
        data_rate_GTs=float(
            data_rate_GTs if data_rate_GTs is not None else base.data_rate_gt_s
        ),
        efficiency=float(
            efficiency if efficiency is not None else base.efficiency
        ),
        capacity_GB=float(
            capacity_GB
            if capacity_GB is not None
            else base.capacity_bytes / 2**30
        ),
        n_packages=pkgs,
        n_ranks=n_ranks,
    )
    return geom.to_external_memory()


def peak_bandwidth_GBps_from_geometry(
    n_channels: int,
    width_bits: int,
    data_rate_GTs: float,
    *,
    n_packages: int = 1,
) -> float:
    """Derived peak GB/s: pkgs * ch * (width/8) * GT/s."""
    return (
        max(n_packages, 1)
        * n_channels
        * (width_bits / 8.0)
        * data_rate_GTs
    )
