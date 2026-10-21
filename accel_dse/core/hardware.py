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
                         "dip" (0.258): hardware cache with thrash-resistant insertion (DIP / BIP, Qureshi et al. ISCA
                         2007): everything hits if the working set fits, otherwise the cyclic-sweep hit rate
                         (K − ε·(T − K)) / T of the retained share, ε = 1/32
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


SLC_POLICIES = ("pin", "lru", "pin_bw", "dip")   # 0.91 pin_bw: pin + bandwidth-balanced hit ratio; 0.258 dip


ACC_KIB_PER_ENGINE = 64.0     # 0.141: = the ≤ 0.140 1024 KiB chip default / 16 engines 「假设」
INSTANCE_SCHEDS = ("wide", "split", "auto", "cta")   # 0.92 cta: GPU thread-block tiling (reference hardware)
NOC_TOPOLOGIES = ("crossbar", "ring", "mesh")
# 0.304: NoC contention model when noc_link_Bpc is set: "bisection" (0.303: payload ÷ bisection links) or "xy"
# (per-link load of dimension-ordered multicast trees / reduction trees, MODEL §249.2)
NOC_ROUTINGS = ("bisection", "xy")
NOC_MEM_ATTACH = ("edge", "none")      # 0.305: DRAM ingress into the NoC (Chip.noc_mem_attach)
NOC_MEM_BALANCE = ("interleave", "pooled")   # 0.311: Chip.noc_mem_balance
NOC_EJECT = ("off", "link")            # 0.307: engine local (eject) port model (Chip.noc_eject)
# 0.303: reference NoC link bandwidth — Tenstorrent Blackhole: one 512-bit flit per cycle per link per NoC (64 B /
# cycle; Wormhole 256 bit = 32 B), tt-isa-documentation BlackholeA0/NoC/README.md 「来源」.  Measured multicast
# throughput on Blackhole is 24 B / cycle (tt-low-level-documentation, data-movement report) — not used, cross-check.
NOC_LINK_BPC_REF = 64.0


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
    # 0.209 (design variable): a dedicated on-chip recurrent-state buffer (MiB), filled before the shared SRAM; 0 = none
    state_sram_mib: float = field(default=0.0, metadata={"omit_default": True})
    # 0.141: per-core default — every engine feeds its own R × C array: None → 4·(R + C)·2 B/cycle per engine
    # (≤ 0.140: 4·(R + C·E)·2 for the whole chip; saved ≤ 0.140 scenarios are migrated, scenario.upgrade_legacy)
    sram_port_Bpc: float | None = field(default=None, metadata={"omit_default": True})   # chip-wide absolute when set
    mac_eff: float = 1.0                  # sustained / peak MAC utilisation inside a tile 「假设」
    # WS partial-sum accumulator (fp32) 「假设」; overflow spills to SRAM.  0.141: per core by default
    # (ACC_KIB_PER_ENGINE × engines); an explicit acc_kib is the chip total (legacy), acc_kib_per_engine scales with E
    acc_kib: float | None = field(default=None, metadata={"omit_default": True})
    slc_mib: float = 0.0                  # system-level cache (0.48) 「假设」; 0 = none
    slc_GBps: float = 2000.0              # SLC bandwidth 「假设」 (used only when slc_mib > 0)
    slc_policy: str = "pin"               # pin | lru | pin_bw | dip (see module doc) 「假设」
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
    # 0.303 (optional, default 0 = unlimited, ≤ 0.302): NoC link bandwidth, bytes per core cycle per link per
    # direction.  The dataflow's NoC payload (0.299: operand multicast, cross-group copies, split-K partial sums) must
    # cross the topology's bisection: an op takes ≥ payload / (bisection links × noc_link_Bpc) cycles, overlapped with
    # the array (max), so a time penalty appears only when the bisection is exceeded.  Reference NOC_LINK_BPC_REF.
    noc_link_Bpc: float = field(default=0.0, metadata={"omit_default": True})
    # 0.304: "xy" prices the busiest link of the dataflow's multicast / reduction trees under dimension-ordered
    # (X then Y) routing instead of the bisection average, and adds the cross-group copies to the in-group traffic
    # they share links with (no effect while noc_link_Bpc = 0).  0.305: default "xy" (was "bisection") — XY is the
    # physical routing of a mesh, and the ring / crossbar formulas are their per-link physical limits (MODEL §250.1)
    noc_routing: str = field(default="xy", metadata={"omit_default": True})
    # 0.305: where DRAM enters the NoC (only while noc_link_Bpc > 0).  "edge" (default): memory tiles in two columns,
    # one attach point per engine row in each (Tenstorrent Blackhole NoC 0: GDDR tiles at x = 0 and x = 9 「来源」), each
    # NIU → router one flit per cycle → DRAM ingress ≤ 2·⌊√E⌋ links × noc_link_Bpc (mesh / crossbar ports alike), ring:
    # two memory stops injecting both ways (4 links).  "none": DRAM ingress not limited by the NoC (≤ 0.304).
    noc_mem_attach: str = field(default="edge", metadata={"omit_default": True})
    # 0.307 (opt-in; only with noc_link_Bpc > 0, noc_routing "xy"): "link" = each engine's local (eject) port is one
    # link wide, so (a) a multicast's per-engine ingest (s − 1)/s · payload / g bounds the op like any link, and
    # (b) the DRAM stream shares that port (D / E per engine) and, on a ring, the links next to the two memory stops
    # (D / 4 each) with the dataflow — the stage needs ≥ (Σ op load on the shared port + DRAM share) / (L · f).
    # "off" = ≤ 0.306 (only network links; DRAM ingress bound separately).  MODEL §252.2
    noc_eject: str = field(default="off", metadata={"omit_default": True})
    # 0.309: this chip object is one of ``noc_clusters`` NoC clusters (domains) of ONE monolithic die — a hierarchical
    # NoC (A100 / H100: "the L2 cache is divided into two partitions … each partition caches data for memory accesses
    # from SMs in the GPCs directly connected to the partition" 「来源」).  The scenario then models the die as
    # noc_clusters cards on an on-die cluster tier (package_cards = noc_clusters, d2d_enabled): the tier is the NoC's
    # own boundary links (no PHY, no reticle relief), its bytes are NoC hops + one link pitch; DRAM attach and DRAM
    # hops keep the whole die's geometry (memory columns on the die's west / east edges, shared by the clusters).
    # 1 = a flat (single-domain) die (≤ 0.308).  MODEL §254
    noc_clusters: int = field(default=1, metadata={"omit_default": True})
    # 0.309: die edges with memory attach (noc_mem_attach "edge"): 2 = the west / east memory columns (0.305,
    # Blackhole); 3 / 4 add a memory row on the north / south edge — HBM stacks that need the 3rd / 4th die edge
    # (budget.shoreline_check "tight") attach there too, C routers per row.  Memory links and DRAM hops follow.
    # 0.310: 0 (default) = auto — the Scenario sets noc_mem_edges_auto from the memory's shoreline occupancy
    # (budget.mem_edges_auto: HBM stacks × 11 mm, LPDDR GB/s ÷ beachfront density, over the die proxy's side);
    # 2 / 3 / 4 = fixed (2 reproduces ≤ 0.309 outside the chiplet DSE).  A bare Chip (no Scenario) uses 2.
    noc_mem_edges: int = field(default=0, metadata={"omit_default": True})
    noc_mem_edges_auto: int = field(default=2, metadata={"internal": True})
    # 0.311: die edges kept for the host interface (PCIe / card link): memory may take at most 4 − D2D edges − this
    # (floor 2) — budget.mem_sides_max.  1 (default) = one edge always left for I/O 「假设」 (shipping packages keep
    # two: GH100 NVLink + PCIe, B200 NV-HBI + I/O 「二手」); 0 = the 0.310 rule (memory may take every non-D2D edge).
    # 0.312: -1 (default) = sized from the mechanism — the host (PCIe) and scale-up SerDes lanes × a PHY edge per
    # lane (io_* below) take shoreline wherever memory leaves room on the non-D2D edges (budget.io_need /
    # shoreline_report); memory may then take every non-D2D edge.  1 = the 0.311 whole reserved edge, 0 = 0.310.
    noc_mem_io_edges: int = field(default=-1, metadata={"omit_default": True})
    # 0.313: io_su_lane_Gbps default 200 = the shipping scale-up lane of the HBM3E era (NVLink5 / B200: 18 links × 2
    # pairs per direction × 200 Gb/s, 224G PAM4 = 900 GB/s per direction, NVIDIA Blackwell whitepaper 「来源」);
    # 100 = NVLink4 (H100, the H100-like preset) = the 0.312 rule; 50 = NVLink3 (A100) — budget.SU_LANE_GENERATIONS.
    # 0.313: io_su_lane_mm 0 (default) = by lane rate (budget.SU_LANE_MM): 224G ≈ 0.5 mm per duplex lane ("DSP/SERDES
    # beachfront costs ~4 mm per Rx/Tx 8-lane port", OIF 448G Signaling for AI workshop, MaxLinear, Apr 2025, slide 14;
    # slide 15: 32 ports × 8 lanes in 128 mm 「来源」, approximate); ≤ 112G keeps the PCIe5 0.285 mm 「假设」.  A
    # positive value pins it.
    # 0.312: host / scale-up I/O lanes and their PHY edge (budget.io_need).  Host: PCIe x16 per package (H100 SXM /
    # PCIe: PCIe Gen5 x16, NVIDIA datasheet 「来源」); per duplex lane 0.285 mm of die edge (Intel 32 Gb/s PCIe Gen5
    # SerDes, 10 nm, CICC 2020: "die edge usage per lane of 285 µm" 「来源」).  Scale-up: the card's Link.GBps per
    # direction ÷ the lane rate (100 Gb/s PAM4: NVLink4 18 links × 2 lanes × 100 G = 450 GB/s per direction 「二手」)
    # duplex lanes, each at the PCIe lane's edge (no public long-reach 112G die-edge figure; 112G XSR 0.8–2 Tb/s/mm =
    # 0.05–0.14 mm/lane is the short-reach floor 「二手」) 「假设」
    io_host_lanes: int = field(default=16, metadata={"omit_default": True})
    io_lane_mm: float = field(default=0.285, metadata={"omit_default": True})
    io_su_lane_Gbps: float = field(default=200.0, metadata={"omit_default": True})
    io_su_lane_mm: float = field(default=0.0, metadata={"omit_default": True})
    # 0.311: how the memory-attach links share the DRAM traffic.  "interleave" (default): addresses are interleaved
    # uniformly over the memory units (HBM stacks / LPDDR channels), so edge e carries the share u_e / N of the bytes
    # its units serve and the NoC delivers at most min_e links_e · N / u_e link-widths (the most loaded edge binds);
    # units are placed on the edges by water-filling against each edge's links and length (budget.mem_edge_load).
    # "pooled" = ≤ 0.310 (every attach link carries an equal share, whatever memory sits behind it).
    noc_mem_balance: str = field(default="interleave", metadata={"omit_default": True})
    # 0.311: memory units per memory edge (west, east, north, south)[: mem_edges] — set by the Scenario
    # (budget.mem_load_auto) for HBM / LPDDR; () = unknown → pooled
    noc_mem_load: tuple = field(default=(), metadata={"internal": True})
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
    # 0.202: GPU reference hardware (refhw._chip).  Gate for GPU-kernel mechanisms that are otherwise reachable from an
    # NPU schedule (mapping.WIDE_WAVES); NPU presets / user chips leave it False
    gpu: bool = field(default=False, metadata={"omit_default": True})
    # 0.203: NPU dispatch model (DISPATCH_MODES above); dispatch_us None = the mode's sourced default
    dispatch: str = field(default="", metadata={"omit_default": True})
    dispatch_us: float | None = field(default=None, metadata={"omit_default": True})
    acc_kib_per_engine: float | None = field(default=None, metadata={"omit_default": True})   # 0.141
    # 0.229 (design lever): share of the per-cycle clock power (energy pJ_clk_mac_cycle) that is NOT clock-gated when
    # the MAC arrays are idle: P_clk = P_full · (g + (1 − g) · u), u = useful MAC slot-cycles / clocked slot-cycles.
    # 1.0 (default) = no gating benefit — the tpuv4i_7nm value is an H100 measurement at u ≈ 0 (already the stalled
    # floor), and Eyeriss v1 (JSSC 2017 Fig. 16) keeps its clock-network power flat (g ≈ 0.96, 推导) while zero-gating
    # cuts MAC activity ×0.24.  Aggressive fine-grained latch gating: g ≈ 0.25 (POWER5-like, HC17 tutorial, 推导)
    clk_ungated_share: float = field(default=1.0, metadata={"omit_default": True})
    # 0.265 (design lever): when True, the gated share (1−g) of P_clk tracks mean(u_mac, u_vec, u_sram_port)
    # instead of u_mac alone — closes MODEL §176 "vector / SRAM ports not gated separately". Equal domain weight
    # 「假设」; default False keeps 0.229 fingerprints.
    clk_datapath_gate: bool = field(default=False, metadata={"omit_default": True})

    def dispatch_s(self) -> tuple[float, float]:
        """0.203: (seconds per step per stage, seconds per op) of the dispatch model."""
        if not self.dispatch:
            return 0.0, 0.0
        us = (DISPATCH_US[self.dispatch] if self.dispatch_us is None else self.dispatch_us) * 1e-6
        return (us, 0.0) if self.dispatch == "static" else (0.0, us)

    def __post_init__(self):
        # 0.94: JSON round-trips give lists — keep the field hashable and the scenario hash stable
        object.__setattr__(self, "fmt_ceiling", tuple((str(k), float(v)) for k, v in self.fmt_ceiling))
        for k, v in self.fmt_ceiling:
            if not (0 < v <= 1):
                raise ValueError(f"chip.fmt_ceiling[{k!r}] must be in (0, 1]")
        if self.dispatch not in DISPATCH_MODES:
            raise ValueError(f"chip.dispatch must be one of {DISPATCH_MODES}")
        if self.dispatch_us is not None and (isinstance(self.dispatch_us, bool) or not 0.0 <= self.dispatch_us < 1e9):
            raise ValueError("chip.dispatch_us must be finite and >= 0")
        if isinstance(self.noc_mem_edges, bool) or self.noc_mem_edges not in (0, 2, 3, 4):
            raise ValueError("chip.noc_mem_edges must be 0 (auto), 2, 3 or 4")
        if self.noc_mem_edges_auto not in (2, 3, 4):
            raise ValueError("chip.noc_mem_edges_auto must be 2, 3 or 4")
        if isinstance(self.noc_mem_io_edges, bool) or self.noc_mem_io_edges not in (-1, 0, 1):
            raise ValueError("chip.noc_mem_io_edges must be -1 (sized), 0 or 1")
        if isinstance(self.io_host_lanes, bool) or not isinstance(self.io_host_lanes, int) \
                or not 0 <= self.io_host_lanes <= 256:
            raise ValueError("chip.io_host_lanes must be an integer in [0, 256]")
        for nm, hi in (("io_lane_mm", 10.0), ("io_su_lane_mm", 10.0), ("io_su_lane_Gbps", 10000.0)):
            v = getattr(self, nm)
            if nm == "io_su_lane_mm" and not isinstance(v, bool) and v == 0:
                continue                    # 0.313: 0 = by lane rate (budget.SU_LANE_MM)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= hi:
                raise ValueError(f"chip.{nm} must be in (0, {hi:g}]")
        if self.noc_mem_balance not in NOC_MEM_BALANCE:
            raise ValueError(f"chip.noc_mem_balance must be one of {NOC_MEM_BALANCE}")
        if not isinstance(self.noc_mem_load, tuple) or any(isinstance(u, bool) or not isinstance(u, int) or u < 0
                                                           for u in self.noc_mem_load):
            raise ValueError("chip.noc_mem_load must be a tuple of unit counts >= 0")
        if isinstance(self.noc_clusters, bool) or not isinstance(self.noc_clusters, int) or not 1 <= self.noc_clusters <= 64:
            raise ValueError("chip.noc_clusters must be an integer in [1, 64]")
        if self.noc_eject not in NOC_EJECT:
            raise ValueError(f"chip.noc_eject must be one of {NOC_EJECT} (got {self.noc_eject!r})")
        if self.noc_mem_attach not in NOC_MEM_ATTACH:
            raise ValueError(f"chip.noc_mem_attach must be one of {NOC_MEM_ATTACH} (got {self.noc_mem_attach!r})")
        if self.noc_routing not in NOC_ROUTINGS:
            raise ValueError(f"chip.noc_routing must be one of {NOC_ROUTINGS} (got {self.noc_routing!r})")
        if self.noc_topology not in NOC_TOPOLOGIES:
            raise ValueError(f"chip.noc_topology must be one of {NOC_TOPOLOGIES} (got {self.noc_topology!r})")
        for k in ("noc_hop_cycles", "wave_overhead_cycles", "kernel_us", "noc_link_Bpc"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= v < 1e9):
                raise ValueError(f"chip.{k} must be finite and >= 0 (got {v})")
        for k in ("freq_ghz", "rows", "cols", "engines", "sram_mib", "mac_eff"):
            v = getattr(self, k)
            if not (v == v) or v <= 0 or v == float("inf"):
                raise ValueError(f"chip.{k} must be finite and > 0 (got {v})")
        g = self.clk_ungated_share
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not (0.0 <= g <= 1.0):
            raise ValueError(f"chip.clk_ungated_share must be in [0, 1] (got {g})")
        if not isinstance(self.clk_datapath_gate, bool):
            raise ValueError(f"chip.clk_datapath_gate must be bool (got {self.clk_datapath_gate!r})")
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

    def noc_bisection_links(self, cores: int | None = None) -> float:
        """0.303: links per direction across the minimum bisection of ``cores`` engines (default all): mesh ⌊√s⌋ (a
        k × k grid cut in half crosses k links), ring 2 (a cut severs two links; traffic may go either way round),
        crossbar s / 2 (a non-blocking switch: half the ports on each side) 「推导」."""
        s_ = cores or self.engines
        if s_ <= 1:
            return math.inf
        return {"crossbar": s_ / 2, "ring": 2.0, "mesh": float(math.isqrt(s_))}[self.noc_topology]

    def noc_bw_cycles(self, payload: float, cores: int | None = None) -> float:
        """0.303: cycles to carry ``payload`` bytes across the bisection of ``cores`` engines (0 when unlimited)."""
        if not self.noc_link_Bpc or payload <= 0:
            return 0.0
        return payload / (self.noc_bisection_links(cores) * self.noc_link_Bpc)

    def noc_block_links(self, g: int) -> float:
        """0.304: links on the boundary of a block of ``g`` engines that a tree edge between blocks can spread over:
        mesh ⌊√g⌋ (one per row of the block's side), ring 1 (a contiguous arc), crossbar g (a port per engine)."""
        if g <= 1:
            return 1.0
        return {"crossbar": float(g), "ring": 1.0, "mesh": float(math.isqrt(g))}[self.noc_topology]

    def noc_xy_load(self, payload: float, s: int, g: int = 1, pattern: str = "allgather") -> float:
        """0.304: bytes on the busiest link (one direction) when ``payload`` moves among ``s`` blocks of ``g`` engines
        under dimension-ordered routing 「推导」:
          allgather  every block needs the whole payload, sharded over the blocks (0.299: outputs stay local and
                     become the next op's A).  Mesh R × C blocks (R = ⌊√s⌋, X then Y): a column link below row r
                     carries every shard of rows ≤ r → (R − 1)/R · payload (a 1 × C row: (C − 1)/C); ring
                     (shortest way round): ⌊s/2⌋/s · payload; crossbar: an output port ingests (s − 1)/s · payload.
          reduce     s partial tiles of ``payload`` summed over a tree mirroring the multicast: every tree edge
                     carries one tile → payload.
        divided by the block-boundary links a tree edge spreads over (noc_block_links(g))."""
        if s <= 1 or payload <= 0:
            return 0.0
        if pattern == "reduce":
            f = 1.0
        elif self.noc_topology == "mesh":
            r = math.isqrt(s)
            c = -(-s // r)
            f = (r - 1) / r if r >= 2 else (c - 1) / c
        elif self.noc_topology == "ring":
            f = (s // 2) / s
        else:
            f = (s - 1) / s
        return payload * f / self.noc_block_links(g)

    @property
    def mem_edges(self) -> int:
        """0.310: memory-attach edges in effect — the fixed noc_mem_edges, else the Scenario-resolved auto value."""
        return self.noc_mem_edges or self.noc_mem_edges_auto

    def noc_mem_links(self) -> float:
        """0.305: NoC links DRAM traffic can enter through (noc_mem_attach "edge"): mesh / crossbar 2·⌊√E⌋ (a memory
        column on each side, one attach per engine row), ring 4 (two memory stops, both directions); inf when the
        link width is unset or attach is "none"."""
        if not self.noc_link_Bpc or self.noc_mem_attach == "none":
            return math.inf
        k = self.noc_clusters        # 0.309: a cluster gets its share of the whole die's memory-column attach points
        m = self.mem_edges           # 0.309: 3 / 4 edges add a memory row on the north / south side (C attach each)
        per = self.noc_mem_edge_links()
        ld = self.noc_mem_load
        if self.noc_mem_balance == "interleave" and len(ld) == m and sum(ld) > 0:
            # 0.311: uniform interleave over the units → edge e carries u_e / N of the bytes on links_e links
            n = sum(ld)
            return min(per[i] * n / u for i, u in enumerate(ld) if u > 0) / k
        return sum(per) / k

    def noc_mem_edge_links(self) -> list[float]:
        """0.311: attach links per memory edge (west, east, north, south)[: mem_edges] of the whole die: mesh /
        crossbar R = ⌊√E⌋ on the west / east columns, C = ⌈E/R⌉ on the north / south rows; ring 2 per memory stop."""
        m = self.mem_edges
        if self.noc_topology == "ring":
            return [2.0] * m
        E = self.engines * self.noc_clusters
        r_ = max(1, math.isqrt(E))
        c_ = -(-E // r_)
        return [float(r_), float(r_), float(c_), float(c_)][:m]

    def noc_mem_hops(self) -> tuple[float, float]:
        """0.306: (router traversals, link pitches) per DRAM byte from the memory columns to the engines (noc_mem_attach
        "edge"; independent of the link width).  Mesh R × C (R = ⌊√E⌋, C = ⌈E/R⌉) with a memory column beside
        column 1 and column C: an engine in column c is min(c, C + 1 − c) hops (and pitches) from the nearer one;
        averaged over the columns 「推导」.  Ring: two memory stops splitting the ring into arcs of E/2 engines, the
        same nearest-stop distance along an arc.  Crossbar: one switch traversal; the wire runs the same floor-plan
        distance as the mesh 「假设」.  (0, 0) for "none" or a single engine."""
        if self.noc_mem_attach == "none" or self.engines * self.noc_clusters <= 1:
            return 0.0, 0.0
        if self.noc_clusters > 1:    # 0.309: memory columns are the die's edges — the whole die's distances 「推导」
            return replace(self, engines=self.engines * self.noc_clusters, noc_clusters=1).noc_mem_hops()

        def mean_d(n: int) -> float:
            n = max(1, n)
            return sum(min(i, n + 1 - i) for i in range(1, n + 1)) / n
        m = self.mem_edges
        if self.noc_topology == "ring":
            d = mean_d(-(-self.engines // m))     # m memory stops (0.309; 2 = ≤ 0.308)
            return d, d
        r_ = max(1, math.isqrt(self.engines))
        c_ = -(-self.engines // r_)
        if m == 2:
            d = mean_d(c_)
        else:   # 0.309: nearest of the west / east columns and the north (m ≥ 3) / south (m = 4) memory rows 「推导」
            d = sum(min(c, c_ + 1 - c, r, *((r_ + 1 - r,) if m == 4 else ()))
                    for r in range(1, r_ + 1) for c in range(1, c_ + 1)) / (r_ * c_)
        return (1.0, d) if self.noc_topology == "crossbar" else (d, d)

    def noc_mem_GBps(self) -> float:
        """0.305: DRAM bandwidth the NoC can deliver into the array: memory-attach links × noc_link_Bpc × clock."""
        n = self.noc_mem_links()
        return math.inf if math.isinf(n) else n * self.noc_link_Bpc * self.freq_ghz

    def noc_d2d_links(self, edges: int) -> float:
        """0.308: NoC links D2D traffic can enter through on ``edges`` die edges given to D2D PHYs (d2d_catalog.d2d_edges)
        「推导」: the PHY sits on the north / south edge, one attach router per engine column (the memory columns are
        the west / east edges, 0.305) → C = ⌈E/R⌉ per edge (mesh; crossbar as the mesh, like noc_mem_links); ring: one
        stop per edge, both directions → 2.  inf when the link width is unset, attach is "none", or no edge."""
        if not self.noc_link_Bpc or self.noc_mem_attach == "none" or edges <= 0:
            return math.inf
        if self.noc_topology == "ring":
            return 2.0 * edges
        r_ = max(1, math.isqrt(self.engines))
        return float(edges * -(-self.engines // r_))

    def noc_d2d_GBps(self, edges: int) -> float:
        """0.308: D2D bandwidth (per direction) the NoC can carry to / from the die edge: attach links × width × clock."""
        n = self.noc_d2d_links(edges)
        return math.inf if math.isinf(n) else n * self.noc_link_Bpc * self.freq_ghz

    def noc_d2d_hops(self) -> tuple[float, float]:
        """0.308: (router traversals, link pitches) per D2D byte between the die edge and an engine (one end) 「推导」:
        mesh R rows below the D2D edge, an engine in row i is i hops from its column's attach router → (R + 1)/2;
        ring: a stop on the ring, nearest way round → mean over E engines of min(i, E − i), ≈ E/4; crossbar: one
        switch traversal, the mesh's wire distance 「假设」.  With two D2D edges each edge's traffic travels
        perpendicular to its own edge (the same mean).  (0, 0) when attach is "none" or a single engine."""
        if self.noc_mem_attach == "none" or self.engines <= 1:
            return 0.0, 0.0
        E = self.engines
        if self.noc_topology == "ring":
            d = sum(min(i, E - i) for i in range(E)) / E
            return d, d
        r_ = max(1, math.isqrt(E))
        d = (r_ + 1) / 2.0
        return (1.0, d) if self.noc_topology == "crossbar" else (d, d)

    def noc_eject_on(self) -> bool:
        """0.307: the local-port model applies (noc_eject "link", a link width set, XY routing)."""
        return bool(self.noc_link_Bpc) and self.noc_eject == "link" and self.noc_routing == "xy"

    def noc_eject_load(self, payload: float, s: int, g: int = 1, pattern: str = "allgather") -> float:
        """0.307: bytes through one engine's local (eject) port when ``payload`` moves among ``s`` blocks of ``g``
        engines 「推导」: all-gather — a block ingests the other blocks' shards, (s − 1)/s · payload, through its g
        engines' ports (the in-block multicast is the sub-op's own traffic); reduce — a tree node ingests at least one
        partial tile, payload / g (lower bound: XY tree nodes with 2–3 children ingest more).  The crossbar's output
        port is the same port (equal to its noc_xy_load)."""
        if s <= 1 or payload <= 0:
            return 0.0
        f = 1.0 if pattern == "reduce" else (s - 1) / s
        return payload * f / max(1, g)

    def noc_port_cycles(self, payload: float, s: int, g: int = 1, pattern: str = "allgather") -> tuple[float, float]:
        """0.307: (busiest network-link cycles, local-port cycles) of one dataflow payload under XY with the local-port
        model on; (0, 0) otherwise.  The first is noc_xy_load / L (≤ 0.306's figure)."""
        if not self.noc_eject_on() or payload <= 0 or s * g <= 1:
            return 0.0, 0.0
        L = self.noc_link_Bpc
        return self.noc_xy_load(payload, s, g, pattern) / L, self.noc_eject_load(payload, s, g, pattern) / L

    def noc_share_seconds(self, eject_s: float, link_s: float, dram_bytes: float) -> float:
        """0.307: stage seconds the DRAM stream and the dataflow need on the resources they share (local-port model
        on, noc_mem_attach "edge"): every engine's port carries Σ op port load + D / E; on a ring the link next to a
        memory stop carries Σ op link load + D / 4 (two stops, both directions).  Mesh row links never exceed
        max(ingress, busiest link) (MODEL §251.3) — no term.  0 when off."""
        if not self.noc_eject_on() or self.noc_mem_attach == "none" or self.engines <= 1:
            return 0.0
        bw = self.noc_link_Bpc * self.freq_ghz * 1e9
        t = eject_s + dram_bytes / self.engines / bw
        if self.noc_topology == "ring":
            t = max(t, link_s + dram_bytes / 4.0 / bw)
        return t

    def noc_traffic_cycles(self, payload: float, s: int, g: int = 1, pattern: str = "allgather") -> float:
        """0.304: NoC cycles of one dataflow payload among ``s`` blocks of ``g`` engines: noc_routing "bisection"
        → payload ÷ (bisection links of s·g engines × noc_link_Bpc) (0.303); "xy" → busiest-link load ÷ noc_link_Bpc.
        0 when noc_link_Bpc is unset."""
        if not self.noc_link_Bpc or payload <= 0 or s * g <= 1:
            return 0.0
        if self.noc_routing == "xy":
            ld = self.noc_xy_load(payload, s, g, pattern)
            if self.noc_eject == "link":          # 0.307: the local port is a link too
                ld = max(ld, self.noc_eject_load(payload, s, g, pattern))
            return ld / self.noc_link_Bpc
        return payload / (self.noc_bisection_links(s * g) * self.noc_link_Bpc)

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
# 0.203: NPU dispatch / sync model (Chip.dispatch).  "" = ideal (no dispatch cost; ≤ 0.202, the default).
#   static  one compiled program per step: host launch + host↔device token sync once per step per stage (decode is
#           serial: the sampled token feeds the next step, so the launch cannot run ahead).  Default 10 µs / step:
#           JAX jit dispatch microbenchmark jit_simple_dispatch 9.1–10.1 µs per call (XLA commit 660c49a, host CPU path
#           only; the TPU program launch / sync is not in it) 「二手」 — a lower bound for TPU-style static graphs.
#   trace   device-replayed command stream (Tenstorrent Metal Trace, Ascend ACLGraph replay): per-op device dispatch.
#           Default 1.5 µs / op: tt-metal issue #20681 (TT dev: kernel load "up to around 1.5us" before launch on busy
#           cores in trace mode; tracy PROFILER_OP_TO_OP_OVERHEAD_NANO_SEC = 1500) 「二手」.
#   host    host-launched per op (eager / piecewise graph).  Default 20 µs / op: vllm-ascend issue #13021, Ascend 910B
#           profile, ~20 µs gaps per piecewise-subgraph boundary 「二手」; MindSpore ACLGraph doc uses 20 µs / operator
#           as its worked example 「假设」 in the source.
#   No public figure was found for Gaudi recipe launch, Inferentia / Trainium NEFF launch, Groq or Cerebras: their
#   static schedules are covered by "static" with dispatch_us as the user's figure 「假设」.
DISPATCH_MODES = ("", "static", "trace", "host")
DISPATCH_US = {"static": 10.0, "trace": 1.5, "host": 20.0}     # per step (static) / per op (trace, host)



@dataclass(frozen=True)
class Link:
    """Scale-up link per rank.  α-β: t = α + bytes/β (per collective step)."""
    GBps: float = 400.0       # 「假设」 per-rank uni-directional bandwidth
    alpha_us: float = 3.0     # 「假设」 per-collective latency (sync + launch)
    topology: str = "switch"  # switch | ring | full_mesh | torus2d — used only with fabric.enabled (0.59)
    # 0.201: all-reduce latency model.  "" = α = alpha_us (the user's own interconnect: NPU design points);
    # "gpu" = NVLink / NVSwitch GPU node (refhw): CUDA custom all-reduce floor / NCCL Ring-LL (schedule.py);
    # 0.202 "rocm" = MI300X xGMI node: vLLM ROCm custom all-reduce measured floor (schedule.ROCM_AR_US)
    ar_lat: str = field(default="", metadata={"omit_default": True})

    def __post_init__(self):
        if not (self.GBps > 0 and self.GBps != float("inf")) or not (self.alpha_us >= 0):
            raise ValueError("link GBps must be finite > 0, alpha_us ≥ 0")
        if self.topology not in LINK_TOPOLOGIES:
            raise ValueError(f"link topology must be one of {', '.join(LINK_TOPOLOGIES)}")
        if self.ar_lat not in ("", "gpu", "rocm"):
            raise ValueError("link ar_lat must be '', 'gpu' or 'rocm'")


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
            return min(self.mem.effective_GBps, c.dma_engines * c.dma_outstanding_KiB * 1024 / (c.dram_latency_ns * 1e-9) / 1e9,
                       c.noc_mem_GBps())
        return min(self.mem.effective_GBps, c.noc_mem_GBps())      # 0.305: NoC DRAM ingress (inf unless link set)

    @property
    def dram_bytes(self) -> float:
        return self.cap_GB * 1e9 if self.cap_GB else float(self.mem.capacity_bytes)


@lru_cache(maxsize=4096)
def _mem_spec(mem_id: str, eff: float | None) -> mem_catalog.MemSpec:
    return mem_catalog.parse_mem_id(mem_id, efficiency=eff)


# Presets (geometry from the 0.3x SKU templates; all 「假设」)
# 0.205 machine balance (bf16 FLOP per DRAM byte, raw bandwidth) — MODEL §152.  Same-class public chips span ≈ 200–560
# (H100 SXM 295, B200 281, MI300X 247, TPU v5e 229, v6e 560, Ascend 910B 200–250, 910C 250, Gaudi 3 496).  ≤ 0.204
# every preset defaulted to LPDDR5X 4 × 64 bit (273 GB/s): 100T 368 (in class, kept); 1P 3850 and H100-like 3510
# (7–15× above the class: the extra compute idles on DRAM).  New defaults 「推导」: 1P → 4 × HBM3E 8-high @ 9.2 Gb/s
# (4.71 TB/s, 223 FLOP/B: the class median ≈ 250–280 at the nearest whole stack count — 3 stacks 297 / 4 stacks 223;
# 4 kept for a B200-like half: half the compute of a B200 on half its 8 stacks); H100-like → the H100 SXM memory
# (5 × HBM3 @ 5.2 Gb/s, 3.35 TB/s, H100 datasheet 「来源」 — the preset is named after it).  The ≤ 0.204 pairings stay as
# legacy presets "1P-lp4x64" / "H100-like-lp4x64".
MEM_LEGACY_0204 = "lpddr5x_4x64_8533_16g"
CHIP_100T = Chip("100T", 1.0, 56, 56, 16)            # 100.35 TFLOPS bf16
CHIP_1P = Chip("1P", 1.0, 128, 128, 32, sram_mib=256.0)   # 1048.6 TFLOPS bf16
CHIP_H100_LIKE = Chip("H100-like", 1.83, 128, 128, 16,      # ≈ 959 TFLOPS dense bf16
                      formats=FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0))),
                      sram_mib=50.0, vector_lanes=16384, sram_port_Bpc=None,
                      io_su_lane_Gbps=100.0)   # 0.313: NVLink4 lanes (H100: 18 × 2 pairs × 100 Gb/s/dir 「来源」)
CHIP_1P_L = replace(CHIP_1P, name="1P-lp4x64")
CHIP_H100_LIKE_L = replace(CHIP_H100_LIKE, name="H100-like-lp4x64")
CHIPS = {c.name: c for c in (CHIP_100T, CHIP_1P, CHIP_H100_LIKE, CHIP_1P_L, CHIP_H100_LIKE_L)}
# default memory per chip name (not a Chip field: the chip object and its hash stay what they were)
PRESET_MEM = {"1P": "hbm3e_4s_8h24g_9200", "H100-like": "hbm3_5s_8h16g_5234"}


# 0.310: published die area of a preset's namesake (not a Chip field: hashes unchanged) — used by the auto memory edges
# instead of the density proxy while the chip keeps the preset's area-relevant geometry.  H100 SXM: 814 mm²
# (NVIDIA H100 whitepaper 「来源」).
PRESET_DIE_MM2 = {"H100-like": 814.0, "H100-like-lp4x64": 814.0}
_AREA_KEYS = ("engines", "rows", "cols", "sram_mib", "slc_mib", "state_sram_mib", "vector_lanes", "noc_clusters")


# 0.311: published long-edge length of a preset's namesake die.  GH100 carries its 6 HBM sites 3 per long side
# (NVIDIA Hopper whitepaper "6 HBM3 or HBM2e stacks"; die shots 「二手」) → ≥ 3 × 11 mm = 33 mm of usable long edge
# (stack pitch budget.HBM_STACK_EDGE_MM, stacks overhanging the die corners on the interposer) 「推导」; the short edge
# is 814 / 33 = 24.7 mm.  Every other die is square (0.212).  0.313 audit: GH100's outline is not published (only
# 814 mm²); the 33 mm follows from the 11 mm stack rule itself (Micron HBM3E 11 × 11 mm 「来源」), so it cannot
# independently confirm that rule — the host-side HBM3 PHY is narrower (8 × 3 mm, HBM4 8 × 4 mm, Samsung via
# inlevel9 「二手」): the stack, not the PHY, sets the edge a stack occupies (I/O cannot face a stack).
PRESET_DIE_EDGE_MM = {"H100-like": 33.0, "H100-like-lp4x64": 33.0}


def die_edges(chip, die_mm2: float) -> list[float]:
    """0.311: the four edge lengths of the die, long edges first: the preset's published long edge when the die is the
    preset's published area (preset_die_mm2), else a square of ``die_mm2``."""
    lg = PRESET_DIE_EDGE_MM.get(getattr(chip, "name", ""))
    if lg and preset_die_mm2(chip) == die_mm2 and lg * lg >= die_mm2:
        return [lg, lg, die_mm2 / lg, die_mm2 / lg]
    s = die_mm2 ** 0.5
    return [s, s, s, s]


def preset_die_mm2(chip) -> float | None:
    """0.310: PRESET_DIE_MM2 for a chip that still has its preset's area-relevant geometry, else None."""
    a = PRESET_DIE_MM2.get(chip.name)
    p = CHIPS.get(chip.name)
    if a is None or p is None or any(getattr(chip, k) != getattr(p, k) for k in _AREA_KEYS):
        return None
    return a


# 0.218: provenance of every preset number (catalog audit).  Chip geometries are design points, not products.
CHIP_SOURCES: dict[str, tuple[str, str]] = {
    "100T": ("假设", "0.3x SKU template geometry (56×56 MAC, 16 cores, 1 GHz); design point, not a product"),
    "1P": ("假设", "0.3x SKU template (128×128 MAC, 32 cores, 1 GHz, 256 MiB SRAM); memory 4 × HBM3E 8H @ 9.2 推导 (MODEL §152)"),
    "H100-like": ("推导/来源", "1.83 GHz × 128² × 16 ≈ 959 TFLOPS dense bf16 matched to H100 SXM 来源; 50 MiB L2 + 5 × HBM3 @ 5.2 Gb/s from the H100 datasheet 来源; MAC geometry 假设"),
    "1P-lp4x64": ("假设", "1P with the ≤ 0.204 LPDDR5X 4 × 64 default"),
    "H100-like-lp4x64": ("假设", "H100-like with the ≤ 0.204 LPDDR5X 4 × 64 default"),
}
LINK_SOURCES: dict[str, tuple[str, str]] = {
    "D2D_DEFAULT": ("来源/假设", "UCIe-A x64 @ 48 GT/s × 4 modules = 1536 GB/s (d2d_catalog, 来源); 4 modules and α 0.5 µs 假设"),
    "NET_DEFAULT": ("二手/假设", "one 400 Gb/s IB NDR / 400GbE NIC per card = 50 GB/s (IEEE 802.3 / IBTA line rate, 二手); α 5 µs 假设"),
    "Link()": ("假设", "in-node scale-up default"),
}


def default_mem(chip) -> str:
    """0.205: the default memory of a chip (by preset name); anything else → the ≤ 0.204 global default."""
    return PRESET_MEM.get(getattr(chip, "name", chip), MEM_LEGACY_0204)
