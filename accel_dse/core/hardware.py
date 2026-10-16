"""L3a — hardware description.  **No existing NPU is presupposed**: every
datapath property below is a design variable; defaults are 「假设」.

Chip
  rows×cols×engines      MAC array (bf16 MACs/cycle = R·C·E); engines tile along N
  formats                native compute formats + per-format MAC-rate multiplier
  vector_lanes           elementwise element-ops / cycle (norm, softmax, rope, act, dequant)
  gemv_macs              optional dedicated vector/GEMV MAC unit (MACs / cycle, bf16)
  sram_mib               on-chip SRAM capacity
  sram_port_Bpc          SRAM ↔ datapath port bandwidth (bytes / cycle) — the *feed* bound
  ws_load_width          weight-stationary load width (elements / cycle):
                         "edge" = C·E (shift in from one edge), "broadside" = R·C·E
  slc_mib / slc_GBps     optional system-level cache between SRAM and DRAM (0.48; 0 = none, the default)
  slc_policy             "pin": software-managed partition — after SRAM, the hottest stage data are pinned in the SLC
                         (hot weights → routed experts → KV / state), served at slc_GBps; streamed activations bypass
                         "lru": hardware cache under the per-step cyclic sweep of an inference step — everything hits if
                         the step's DRAM working set fits, otherwise nothing does (LRU thrash bound)
System = Chip + external memory (mem_catalog.MemSpec) + a three-tier interconnect (0.50):
  d2d    optional die-to-die tier between the ``package_cards`` dies of one package (chiplet stacking); off (the
         default) = monolithic die, every card its own package.  Default grade when on: UCIe-A x64 @ 48 GT/s × 4
         modules = 1536 GB/s per direction (core/d2d_catalog.py)
  link   in-node scale-up between packages (the pre-0.50 single link: 400 GB/s, 3 µs 「假设」)
  net    cross-node scale-out (IB / RoCEv2 class) between nodes of ``node_cards`` cards; 0 = the whole system is one
         node (the default → the tier is unused).  Default 50 GB/s per card (one 400 Gb/s NIC), 5 µs 「假设」
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .. import mem_catalog
from .dtypes import FormatSupport


SLC_POLICIES = ("pin", "lru", "pin_bw")   # 0.91 pin_bw: pin + bandwidth-balanced hit ratio (opt-in)


ACC_KIB_PER_ENGINE = 64.0     # 0.141: = the ≤ 0.140 1024 KiB chip default / 16 engines 「假设」
INSTANCE_SCHEDS = ("wide", "split", "auto", "cta")   # 0.92 cta: GPU thread-block tiling (reference hardware)
NOC_TOPOLOGIES = ("crossbar", "ring", "mesh")


@dataclass(frozen=True)
class Chip:
    name: str = "chip"
    freq_ghz: float = 1.0
    rows: int = 56
    cols: int = 56
    engines: int = 16
    formats: FormatSupport = FormatSupport()
    vector_lanes: int | None = None       # 「假设」 None → 4·cols·engines
    # 0.103 (opt-in): transcendental (exp / rsqrt / reciprocal) results per cycle on a separate SFU pipe that runs
    # alongside the lanes; None = transcendentals at the lane rate (pre-0.103). NVIDIA: 16 / clk / SM (CUDA Programming
    # Guide, native arithmetic throughput, cc 7.x–10.0) vs 64–128 FP32 lanes / SM
    sfu_per_cycle: float | None = field(default=None, metadata={"omit_default": True})
    # 0.104 (opt-in): the flash-attention kernel overlaps softmax / exp with its MMAs (FlashAttention-3, Shah et al.
    # 2024: warpgroup ping-pong + intra-warpgroup softmax/GEMM pipelining) — kernel / tbo overlap modes only
    attn_vec_overlap: bool = field(default=False, metadata={"omit_default": True})
    gemv_macs: int | None = None          # 「假设」 None → R·C·E / 8 (only used by mappings with a vector unit)
    sram_mib: float = 64.0                # 「假设」
    # 0.141: per-core default — every engine feeds its own R × C array: None → 4·(R + C)·2 B/cycle per engine
    # (≤ 0.140: 4·(R + C·E)·2 for the whole chip; saved ≤ 0.140 scenarios are migrated, scenario.upgrade_legacy)
    sram_port_Bpc: float | None = field(default=None, metadata={"omit_default": True})   # chip-wide absolute when set
    mac_eff: float = 1.0                  # sustained / peak MAC utilisation inside a tile 「假设」
    # WS partial-sum accumulator (fp32) 「假设」; overflow spills to SRAM.  0.141: per core by default
    # (ACC_KIB_PER_ENGINE × engines); an explicit acc_kib is the chip total (legacy), acc_kib_per_engine scales with E
    acc_kib: float | None = field(default=None, metadata={"omit_default": True})
    slc_mib: float = 0.0                  # system-level cache (0.48) 「假设」; 0 = none
    slc_GBps: float = 2000.0              # SLC bandwidth 「假设」 (used only when slc_mib > 0)
    slc_policy: str = "pin"               # pin | lru (see module doc) 「假设」
    # 0.70: engines may each run an independent GEMM instance (batched attention heads / requests, grouped experts):
    # a ``count``-instance op then also costs ceil(count / engines) waves on one R×C engine each (cheaper of that and
    # the chip-wide array).  GPU-like (SMs are independent); False = the 0.69 chip-wide array only 「假设」
    split_instances: bool = field(default=False, metadata={"omit_default": True})   # 0.70.1: hash-stable
    # 0.71: how a GEMM with count > 1 independent instances (per-request / per-head attention QK·PV, absorbed MLA
    # per-head products, per-expert GEMMs) is scheduled over the engines (cores):
    #   "wide"  — all engines ganged as one R × (C·E) array, instances back to back (≤ 0.70 behaviour; a lock-step
    #             single-sequencer array, or a GPU reference where the 0.70 calibration holds);
    #   "split" — one instance per engine, ceil(count / E) waves (= split_instances=True);
    #   "auto"  — (0.71 default: independent core sequencers) cheapest of wide and every core-group size g | E (E/g groups of R × (C·g), each with g/E of the
    #             SRAM port and accumulator), per op.  Total SRAM / DRAM bytes are unchanged (operands streamed);
    #             each group needs its own instruction stream / sequencer 「假设」.
    instance_sched: str = field(default="auto", metadata={"omit_default": True})   # 0.71 default auto
    # 0.72 (optional, default off): on-chip NoC latency between cores.  A schedule spanning s cores pays
    # noc_hop_cycles × diameter(topology, s) once per op per sequencer (operand distribution / systolic hand-off);
    # topology "crossbar" (1 hop), "ring" (⌊s/2⌋), "mesh" (2·(⌈√s⌉−1)).  0 = ideal NoC (≤ 0.71) 「假设」
    noc_topology: str = field(default="mesh", metadata={"omit_default": True})
    noc_hop_cycles: float = field(default=0.0, metadata={"omit_default": True})
    # 0.72 (optional, default 0): sequencer launch / barrier overhead (cycles).  Every op pays one launch; when ≥ 2
    # core groups run instances in lock-step waves, each further wave adds one barrier (a single sequencer — wide,
    # or one active group — streams its instances back to back) 「假设」
    wave_overhead_cycles: float = field(default=0.0, metadata={"omit_default": True})
    # 0.92 (optional, default 0): fixed device-side cost per kernel (µs), paid serially once per GEMM / attention op
    # (elementwise ops are taken as fused into GEMM prologues / epilogues).  GPU reference hardware: 0.53 µs = NVIDIA's
    # repeat-launch device runtime of a 100-node straight-line CUDA graph of empty kernels (53 µs, CUDA 12.6)
    kernel_us: float = field(default=0.0, metadata={"omit_default": True})
    # 0.93 (optional, default "" = instance_sched): schedule for attention ops only (QK / PV, incl. MLA latent), e.g.
    # "cta" — attention kernels spread (request, head) tiles over SMs (FlashMLA / FMHA) even where GEMMs keep "wide"
    attn_sched: str = field(default="", metadata={"omit_default": True})
    # 0.94 (optional): sustained fraction of peak MAC rate per exec format, ((fmt, frac), ...) — a kernel-level ceiling
    # from a public kernel benchmark (e.g. Hopper FP8 block-scaled GEMM, DeepGEMM), applied as max(t, ideal / frac);
    # a "<kind>:<fmt>" key (e.g. "attn:bf16") takes precedence for that op kind
    fmt_ceiling: tuple = field(default=(), metadata={"omit_default": True})
    # 0.96 (opt-in): attention-core compute format. "" = bf16 math with the KV cache up-cast on the vector path (the
    # default); "kv" = native low-precision attention when the KV format is native (Q and P quantised to it — the
    # FlashAttention-3 FP8 / FP8-FMHA recipe): no KV up-cast, MAC rate of the KV format, Q / P quantisation charged
    attn_fmt: str = field(default="", metadata={"omit_default": True})
    # 0.96: SRAM port width per engine (B/cycle) — scales with `engines` (sweeps / die splits); exclusive with the
    # chip-wide absolute sram_port_Bpc
    sram_port_Bpc_per_engine: float | None = field(default=None, metadata={"omit_default": True})
    # 0.98 exec_overlap "pipe" (core/overlap.py): DMA runs ahead into the next ops (False = GPU kernel boundaries);
    # optional DMA issue limit by Little's law — engines × outstanding KiB / DRAM latency (all three or none)
    pipe_cross_op: bool = field(default=True, metadata={"omit_default": True})
    dma_engines: int | None = field(default=None, metadata={"omit_default": True})
    dma_outstanding_KiB: float | None = field(default=None, metadata={"omit_default": True})
    dram_latency_ns: float | None = field(default=None, metadata={"omit_default": True})
    # 0.135 (opt-in, instance_sched "cta"): "dram" = each output tile also pays its pipeline prologue (one K-stage of
    # A and B, 128 B of K per row: the TMA swizzle width) and its C writeback at the SM's share of DRAM bandwidth,
    # not overlapped with the main loop (MODEL §81).  tile_io_Bpc: chip DRAM bytes / cycle, filled by evaluate.
    tile_io: str = field(default="", metadata={"omit_default": True})
    # 0.195 tile_io "occupancy": per-SM resources for resident CTAs (CUDA occupancy rules): shared memory KiB, register
    # (or Blackwell TMEM) KiB for the fp32 accumulator tile, max resident threads
    occ_smem_kib: float = field(default=0.0, metadata={"omit_default": True})
    occ_reg_kib: float = field(default=0.0, metadata={"omit_default": True})
    occ_threads: int = field(default=0, metadata={"omit_default": True})
    smem_Bpc_sm: float = field(default=0.0, metadata={"omit_default": True})   # 0.196: SMEM B/clk per SM (cta main loop)
    # 0.193: GEMM fmt_ceiling applied as a multiplicative achieved-peak fraction on top of the tile/wave-quantised
    # time (instance_sched "cta"), instead of a floor (max(t, ideal / cap)).  The fraction must already have the
    # wave quantisation of its measured shape divided out (extval.net_ceiling)
    gemm_ceil_mul: bool = field(default=False, metadata={"omit_default": True})
    # 0.193: ≤ 0.192 "wide" padding (N padded to E·cols) for this chip — kept by the historical extval variants
    wide_pad: bool = field(default=False, metadata={"omit_default": True})
    # 0.136 (opt-in): stand-alone vector ops (norms, rope, activation, residual — not the softmax inside the fused
    # attention kernel) are kernels too and pay kernel_us each (the GPU floor of a small kernel)
    kernel_vec: bool = field(default=False, metadata={"omit_default": True})
    # 0.177 (opt-in): GPU-runtime kernels the op list otherwise omits (activation quant casts, MLA q-norm, split-KV combine,
    # router top-k + split-K reduce, shared-expert activation, sampling passes; EP all-to-all as send + recv kernels) —
    # ir.glue_ops; each pays its own work and, with kernel_us / kernel_vec, the per-kernel fixed cost
    gpu_glue: bool = field(default=False, metadata={"omit_default": True})
    tile_io_Bpc: float = field(default=0.0, metadata={"omit_default": True})
    acc_kib_per_engine: float | None = field(default=None, metadata={"omit_default": True})   # 0.141

    def __post_init__(self):
        # 0.94: JSON round-trips give lists — keep the field hashable and the scenario hash stable
        object.__setattr__(self, "fmt_ceiling", tuple((str(k), float(v)) for k, v in self.fmt_ceiling))
        for k, v in self.fmt_ceiling:
            if not (0 < v <= 1):
                raise ValueError(f"chip.fmt_ceiling[{k!r}] must be in (0, 1]")
        if self.noc_topology not in NOC_TOPOLOGIES:
            raise ValueError(f"chip.noc_topology must be one of {NOC_TOPOLOGIES} (got {self.noc_topology!r})")
        for k in ("noc_hop_cycles", "wave_overhead_cycles", "kernel_us"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= v < 1e9):
                raise ValueError(f"chip.{k} must be finite and >= 0 (got {v})")
        for k in ("freq_ghz", "rows", "cols", "engines", "sram_mib", "mac_eff"):
            v = getattr(self, k)
            if not (v == v) or v <= 0 or v == float("inf"):
                raise ValueError(f"chip.{k} must be finite and > 0 (got {v})")
        if self.mac_eff > 1.0:      # 0.129: sustained / peak utilisation above 1 was accepted (silently ≈ 1)
            raise ValueError(f"chip.mac_eff must be in (0, 1] (got {self.mac_eff})")
        for k, lo in (("slc_mib", 0.0), ("slc_GBps", 1e-9)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v < 1e7):
                raise ValueError(f"chip.{k} must be finite and ≥ {lo:g}")
        dma = (self.dma_engines, self.dma_outstanding_KiB, self.dram_latency_ns)
        if any(x is not None for x in dma):
            if any(x is None for x in dma):
                raise ValueError("chip.dma_engines, dma_outstanding_KiB and dram_latency_ns go together (Little's law)")
            for nm, x in zip(("dma_engines", "dma_outstanding_KiB", "dram_latency_ns"), dma):
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not (math.isfinite(x) and x > 0):
                    raise ValueError(f"chip.{nm} must be finite and > 0")
        if self.sram_port_Bpc_per_engine is not None:
            v = self.sram_port_Bpc_per_engine
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v > 0):
                raise ValueError("chip.sram_port_Bpc_per_engine must be finite and > 0")
            if self.sram_port_Bpc:
                raise ValueError("set either chip.sram_port_Bpc (chip-wide) or chip.sram_port_Bpc_per_engine, not both")
        for k in ("acc_kib", "acc_kib_per_engine"):
            v = getattr(self, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v >= 0)):
                raise ValueError(f"chip.{k} must be finite and >= 0")
        if self.acc_kib is not None and self.acc_kib_per_engine is not None:
            raise ValueError("set either chip.acc_kib (chip-wide) or chip.acc_kib_per_engine, not both")
        if self.attn_fmt not in ("", "kv"):
            raise ValueError("chip.attn_fmt must be '' (bf16 + KV up-cast) or 'kv' (native KV-format attention)")
        if self.tile_io not in ("", "dram", "persistent", "occupancy"):
            raise ValueError("chip.tile_io must be empty, dram, persistent or occupancy")
        if self.attn_sched and self.attn_sched not in INSTANCE_SCHEDS:
            raise ValueError(f"chip.attn_sched must be empty or one of {', '.join(INSTANCE_SCHEDS)}")
        if self.instance_sched not in INSTANCE_SCHEDS:
            raise ValueError(f"chip.instance_sched must be one of {', '.join(INSTANCE_SCHEDS)}")
        if self.slc_policy not in SLC_POLICIES:
            raise ValueError(f"chip.slc_policy must be one of {', '.join(SLC_POLICIES)}")

    @property
    def slc_bytes(self) -> float:
        return self.slc_mib * 2**20

    def noc_cycles(self, cores: int | None = None) -> float:
        """0.72: NoC distribution latency (cycles) across ``cores`` (default all engines)."""
        if not self.noc_hop_cycles:
            return 0.0
        s_ = cores or self.engines
        if s_ <= 1:
            return 0.0
        d = {"crossbar": 1, "ring": s_ // 2, "mesh": 2 * math.isqrt(s_ - 1)}[self.noc_topology]
        return self.noc_hop_cycles * d

    @property
    def c_eff(self) -> int:
        return self.cols * self.engines

    @property
    def macs(self) -> int:
        return self.rows * self.cols * self.engines

    @property
    def peak_tflops_bf16(self) -> float:
        return 2.0 * self.macs * self.freq_ghz / 1e3

    @property
    def lanes(self) -> int:
        return self.vector_lanes or 4 * self.cols * self.engines

    @property
    def gemv(self) -> int:
        return self.gemv_macs or max(1, self.macs // 8)

    @property
    def port_Bpc(self) -> float:
        if self.sram_port_Bpc_per_engine:
            return self.sram_port_Bpc_per_engine * self.engines
        return self.sram_port_Bpc or 4.0 * (self.rows + self.cols) * 2.0 * self.engines

    @property
    def acc_total_kib(self) -> float:
        """0.141: chip-total accumulator KiB (explicit total, else per engine × engines)."""
        if self.acc_kib is not None:
            return self.acc_kib
        return (self.acc_kib_per_engine if self.acc_kib_per_engine is not None else ACC_KIB_PER_ENGINE) * self.engines

    @property
    def port_GBps(self) -> float:
        return self.port_Bpc * self.freq_ghz

    @property
    def acc_rows(self) -> int:
        """Output rows of a full-width (C·E) tile the WS accumulator can hold."""
        return int(self.acc_total_kib * 1024 // (4 * self.c_eff))

    @property
    def sram_bytes(self) -> float:
        return self.sram_mib * 2**20


LINK_TOPOLOGIES = ("switch", "ring", "full_mesh", "torus2d")


@dataclass(frozen=True)
class Link:
    """Scale-up link per rank.  α-β: t = α + bytes/β (per collective step)."""
    GBps: float = 400.0       # 「假设」 per-rank uni-directional bandwidth
    alpha_us: float = 3.0     # 「假设」 per-collective latency (sync + launch)
    topology: str = "switch"  # switch | ring | full_mesh | torus2d — used only with fabric.enabled (0.59)

    def __post_init__(self):
        if not (self.GBps > 0 and self.GBps != float("inf")) or not (self.alpha_us >= 0):
            raise ValueError("link GBps must be finite > 0, alpha_us ≥ 0")
        if self.topology not in LINK_TOPOLOGIES:
            raise ValueError(f"link topology must be one of {', '.join(LINK_TOPOLOGIES)}")


NET_TOPOLOGIES = ("fat_tree", "rail")
FABRIC_ALGOS = ("auto", "auto_overlap", "ring", "tree", "hier")
OVERLAP_MODES = ("sum", "ports")
NET_TIERS = (2, 3)
INNET_MODES = ("off", "net", "net+link")
PROTOCOLS = ("off", "auto", "LL", "LL128", "Simple")    # 0.61 NCCL-like wire protocols (off = the 0.60 model)


@dataclass(frozen=True)
class Fabric:
    """Topology-aware collectives (0.59, 「假设」; off = the 0.50 three-tier α-β model, bit for bit).  See core/fabric.py.

    net_topology  cross-node tier: "fat_tree" (two-level leaf-spine; ``leaf_nodes`` whole nodes under one leaf, leaf
                  uplinks oversubscribed ``oversub``:1, spine non-blocking) or "rail" (rail-optimised: NIC j of every
                  node on rail switch j, ``leaf_nodes`` nodes per rail switch, cross-rail traffic first hops over the
                  in-node scale-up (PXN)).  leaf_nodes = 0 → derived from ``switch_radix``: down ports
                  = radix·r/(1 + r); fat_tree: down ports // node_cards nodes, rail: down ports nodes.
    algo          auto (cheapest of ring / tree / hier per collective) | ring (flat ring) | tree (NCCL double binary
                  tree across nodes, chains inside) | hier (reduce-scatter inside, allreduce on top, all-gather back).
    hop_*_us      per-step latency of each tier (ring / tree steps); defaults = NCCL tuner defaults for the LL protocol
                  (NVLink ring 0.6 µs, NET ring 2.7 µs, NET tree 5.0 µs; D2D 0.1 µs 「假设」) — reference order of
                  magnitude, not a measurement of any system.
    innet_reduce  in-network reduction, vendor option 「假设」: "net" = switch aggregation on the cross-node tier (SHARP /
                  CollNet class), "net+link" also on a switched scale-up tier (NVLS class).
    contention    PD KV hand-off shares the NICs with the pools' collectives 「假设」.
    torus_x       X extent of a ``link.topology = torus2d`` scale-up domain (0 = the divisor nearest √n).
    0.60 「假设」 (all default to the 0.59 behaviour):
    net_tiers     2 = leaf-spine (0.59) | 3 = leaf / spine / core fat-tree: ``pod_nodes`` nodes per pod (0 = leaf_nodes ×
                  spine down ports, spine down ports = radix·r₂/(1 + r₂)), spine → core uplinks oversubscribed
                  ``oversub_spine``:1 on top of the leaf's ``oversub`` (fat_tree only; rail keeps two levels).
    overlap       "sum" (0.59: a stage's collectives' bandwidth seconds add up) | "ports": the D2D PHY, scale-up ports
                  and NIC of a card are separate ports — a stage's link time is the busiest port's Σ busy seconds (each
                  collective contributes its per-tier busy time), floored at the longest single collective.
    algo          + "auto_overlap": per stage, the algorithm combination minimising the stage time
                  max(compute / DRAM window, link) + Σα (exposed time) instead of each collective's bw + α alone.
    kv_feedback   PD: the KV stream's share of the decode pool's NIC slows the decode pool's network collectives
                  (u_kv; decode TPOT recomputed) — 0.59 only reported the collectives' slowdown on the KV stream.
    channels / channel_GBps  0.124 「假设」 (0 = off): a collective is split over ``channels`` rings / trees, each
                  driven by one copy engine (an SM / DMA queue) that moves at most ``channel_GBps``; a scale-up or
                  network tier then delivers min(link GBps, channels × channel_GBps) to the collective (NCCL nChannels).
    kv_feedback_prefill  0.121 PD (with kv_feedback): the same u_kv also takes the prefill pool's NICs (the sending
                  side of each pair carries the same bytes), so the prefill pool's network collectives run on
                  (1 − u_kv)·net.GBps too; prefill TTFT / request rate recomputed inside the same fixed point.
    0.61 「假设」 (defaults = the 0.60 behaviour):
    hop_spine_us  extra per-step latency of a network step that leaves the leaf (leaf → spine → leaf: two more switch
                  traversals); hop_net_us / hop_net_tree_us stay the same-leaf step.
    hop_core_us   extra again for a step that leaves the pod (three-tier fat-tree: spine → core → spine).
    protocol      "off" (0.60: hop_* latency at full link bandwidth) | "auto" (per collective the fastest of LL / LL128 /
                  Simple) | one of them: bandwidth efficiency LL ½ (4 B flag per 8 B), LL128 120/128, Simple 1;
                  per-step latency × the NCCL tuner's protocol ratios (hop_* are the LL values).
    innet_reduce  "net+link" also covers all-gather / reduce-scatter on the switched scale-up tier (NVLS-class
                  multicast / in-switch reduction): same bytes per port (reduce-scatter sends n shards instead of n − 1),
                  one switch round trip instead of n − 1 steps; SHARP-class "net" stays allreduce-only."""
    enabled: bool = False
    algo: str = "auto_overlap"     # 0.141 default (stage re-pick; monotone in bandwidth); ≤ 0.140 "auto"
    net_topology: str = "fat_tree"
    oversub: float = 1.0
    leaf_nodes: int = 0
    switch_radix: int = 64
    torus_x: int = 0
    hop_d2d_us: float = 0.1
    hop_link_us: float = 0.6
    hop_net_us: float = 2.7
    hop_net_tree_us: float = 5.0
    innet_reduce: str = "off"
    contention: bool = True
    net_tiers: int = 2
    pod_nodes: int = 0
    oversub_spine: float = 1.0
    overlap: str = "sum"
    kv_feedback: bool = False
    hop_spine_us: float = 0.0
    hop_core_us: float = 0.0
    protocol: str = "off"
    kv_feedback_prefill: bool = field(default=False, metadata={"omit_default": True})   # 0.121
    channels: int = field(default=0, metadata={"omit_default": True})          # 0.124 (0 = off)
    channel_GBps: float = field(default=0.0, metadata={"omit_default": True})  # 0.124

    def __post_init__(self):
        if not all(isinstance(getattr(self, k), bool) for k in ("enabled", "contention", "kv_feedback",
                                                                 "kv_feedback_prefill")):
            raise ValueError("fabric.enabled / fabric.contention / fabric.kv_feedback / fabric.kv_feedback_prefill "
                             "must be booleans")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or not 0 <= self.channels <= 1024:
            raise ValueError("fabric.channels must be an integer in [0, 1024]")
        if isinstance(self.channel_GBps, bool) or not isinstance(self.channel_GBps, (int, float)) \
                or not 0 <= self.channel_GBps <= 1e5 or (self.channels > 0) != (self.channel_GBps > 0):
            raise ValueError("fabric.channels and fabric.channel_GBps are set together (both > 0) or both 0")
        if self.net_tiers not in NET_TIERS or isinstance(self.net_tiers, bool):
            raise ValueError("fabric.net_tiers must be 2 (leaf-spine) or 3 (leaf / spine / core)")
        if isinstance(self.oversub_spine, bool) or not isinstance(self.oversub_spine, (int, float)) \
                or not 1.0 <= self.oversub_spine <= 64:
            raise ValueError("fabric.oversub_spine must be in [1, 64] (down:up ratio of the spine → core uplinks)")
        if isinstance(self.pod_nodes, bool) or not isinstance(self.pod_nodes, int) or not 0 <= self.pod_nodes <= 65536:
            raise ValueError("fabric.pod_nodes must be an integer in [0, 65536]")
        if self.pod_nodes and self.leaf_nodes and self.pod_nodes % self.leaf_nodes:
            raise ValueError("fabric.pod_nodes must be a multiple of fabric.leaf_nodes (whole leaves per pod)")
        for k, opts in (("algo", FABRIC_ALGOS), ("net_topology", NET_TOPOLOGIES), ("innet_reduce", INNET_MODES),
                        ("overlap", OVERLAP_MODES), ("protocol", PROTOCOLS)):
            if getattr(self, k) not in opts:
                raise ValueError(f"fabric.{k} must be one of {', '.join(opts)}")
        if isinstance(self.oversub, bool) or not isinstance(self.oversub, (int, float)) or not 1.0 <= self.oversub <= 64:
            raise ValueError("fabric.oversub must be in [1, 64] (down:up ratio of the leaf uplinks)")
        for k, lo, hi in (("leaf_nodes", 0, 4096), ("switch_radix", 4, 1024), ("torus_x", 0, 1024)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"fabric.{k} must be an integer in [{lo}, {hi}]")
        for k in ("hop_d2d_us", "hop_link_us", "hop_net_us", "hop_net_tree_us", "hop_spine_us", "hop_core_us"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1e4:
                raise ValueError(f"fabric.{k} must be in [0, 1e4] µs")


D2D_DEFAULT = Link(1536.0, 0.5)   # UCIe-A x64 @ 48 GT/s × 4 modules (raw, per direction); α = collective sync 「假设」
NET_DEFAULT = Link(50.0, 5.0)     # cross-node: one 400 Gb/s IB NDR / 400GbE RoCEv2 NIC per card; α 「假设」


@dataclass(frozen=True)
class System:
    chip: Chip
    mem_id: str = "lpddr5x_4x64_8533_16g"
    mem_eff: float | None = None
    link: Link = Link()                 # in-node scale-up between packages
    d2d: Link = D2D_DEFAULT             # between the dies of one package (effective only with package_cards > 1)
    package_cards: int = 1              # dies per package on the D2D tier; 1 = monolithic (D2D off)
    net: Link = NET_DEFAULT             # cross-node scale-out
    node_cards: int = 0                 # cards per node; 0 = one node (cross-node tier unused)
    fabric: Fabric = Fabric()           # 0.59 topology-aware collectives (off = 0.50 model)
    cards: int = 0                      # cards of the evaluated replica (scale-up domain when node_cards = 0; 0.59)
    cap_GB: float = 0.0                 # 0.94: usable capacity override (GB); 0 = stack capacity
    kv_frac: float = 1.0                # 0.94: KV / state share of post-weight free memory
    kv_eff: float | None = None         # 0.108 (opt-in): DRAM efficiency of KV-cache streams (None = mem_eff)

    @property
    def kv_time_scale(self) -> float:
        """0.108: time of one KV-cache byte relative to an average DRAM byte (1.0 unless kv_eff is set)."""
        if self.kv_eff is None:
            return 1.0
        return self.mem.effective_GBps / _mem_spec(self.mem_id, self.kv_eff).effective_GBps

    @property
    def mem(self) -> mem_catalog.MemSpec:
        return _mem_spec(self.mem_id, self.mem_eff)

    @property
    def dram_GBps(self) -> float:
        c = self.chip
        if c.dma_engines:     # 0.98 (opt-in): DMA issue limit, Little's law — engines × outstanding bytes / latency
            return min(self.mem.effective_GBps, c.dma_engines * c.dma_outstanding_KiB * 1024 / (c.dram_latency_ns * 1e-9) / 1e9)
        return self.mem.effective_GBps

    @property
    def dram_bytes(self) -> float:
        return self.cap_GB * 1e9 if self.cap_GB else float(self.mem.capacity_bytes)


@lru_cache(maxsize=4096)
def _mem_spec(mem_id: str, eff: float | None) -> mem_catalog.MemSpec:
    return mem_catalog.parse_mem_id(mem_id, efficiency=eff)


# Presets (geometry from the 0.3x SKU templates; all 「假设」)
CHIP_100T = Chip("100T", 1.0, 56, 56, 16)            # 100.35 TFLOPS bf16
CHIP_1P = Chip("1P", 1.0, 128, 128, 32, sram_mib=256.0)   # 1048.6 TFLOPS bf16
CHIP_H100_LIKE = Chip("H100-like", 1.83, 128, 128, 16,      # ≈ 959 TFLOPS dense bf16
                      formats=FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0))),
                      sram_mib=50.0, vector_lanes=16384, sram_port_Bpc=None)
CHIPS = {c.name: c for c in (CHIP_100T, CHIP_1P, CHIP_H100_LIKE)}
