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

from dataclasses import dataclass, field, replace
from functools import lru_cache

from .. import mem_catalog
from .dtypes import FormatSupport


SLC_POLICIES = ("pin", "lru")


@dataclass(frozen=True)
class Chip:
    name: str = "chip"
    freq_ghz: float = 1.0
    rows: int = 56
    cols: int = 56
    engines: int = 16
    formats: FormatSupport = FormatSupport()
    vector_lanes: int | None = None       # 「假设」 None → 4·cols·engines
    gemv_macs: int | None = None          # 「假设」 None → R·C·E / 8 (only used by mappings with a vector unit)
    sram_mib: float = 64.0                # 「假设」
    sram_port_Bpc: float | None = None    # 「假设」 None → 4·(R + C·E)·2 B/cycle
    mac_eff: float = 1.0                  # sustained / peak MAC utilisation inside a tile 「假设」
    acc_kib: float = 1024.0               # WS partial-sum accumulator (fp32) 「假设」; overflow spills to SRAM
    slc_mib: float = 0.0                  # system-level cache (0.48) 「假设」; 0 = none
    slc_GBps: float = 2000.0              # SLC bandwidth 「假设」 (used only when slc_mib > 0)
    slc_policy: str = "pin"               # pin | lru (see module doc) 「假设」

    def __post_init__(self):
        for k in ("freq_ghz", "rows", "cols", "engines", "sram_mib", "mac_eff"):
            v = getattr(self, k)
            if not (v == v) or v <= 0 or v == float("inf"):
                raise ValueError(f"chip.{k} must be finite and > 0 (got {v})")
        for k, lo in (("slc_mib", 0.0), ("slc_GBps", 1e-9)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v < 1e7):
                raise ValueError(f"chip.{k} must be finite and ≥ {lo:g}")
        if self.slc_policy not in SLC_POLICIES:
            raise ValueError(f"chip.slc_policy must be one of {', '.join(SLC_POLICIES)}")

    @property
    def slc_bytes(self) -> float:
        return self.slc_mib * 2**20

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
        return self.sram_port_Bpc or 4.0 * (self.rows + self.c_eff) * 2.0

    @property
    def port_GBps(self) -> float:
        return self.port_Bpc * self.freq_ghz

    @property
    def acc_rows(self) -> int:
        """Output rows of a full-width (C·E) tile the WS accumulator can hold."""
        return int(self.acc_kib * 1024 // (4 * self.c_eff))

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
FABRIC_ALGOS = ("auto", "ring", "tree", "hier")
INNET_MODES = ("off", "net", "net+link")


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
    torus_x       X extent of a ``link.topology = torus2d`` scale-up domain (0 = the divisor nearest √n)."""
    enabled: bool = False
    algo: str = "auto"
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

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.contention, bool):
            raise ValueError("fabric.enabled / fabric.contention must be booleans")
        for k, opts in (("algo", FABRIC_ALGOS), ("net_topology", NET_TOPOLOGIES), ("innet_reduce", INNET_MODES)):
            if getattr(self, k) not in opts:
                raise ValueError(f"fabric.{k} must be one of {', '.join(opts)}")
        if isinstance(self.oversub, bool) or not isinstance(self.oversub, (int, float)) or not 1.0 <= self.oversub <= 64:
            raise ValueError("fabric.oversub must be in [1, 64] (down:up ratio of the leaf uplinks)")
        for k, lo, hi in (("leaf_nodes", 0, 4096), ("switch_radix", 4, 1024), ("torus_x", 0, 1024)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"fabric.{k} must be an integer in [{lo}, {hi}]")
        for k in ("hop_d2d_us", "hop_link_us", "hop_net_us", "hop_net_tree_us"):
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

    @property
    def mem(self) -> mem_catalog.MemSpec:
        return _mem_spec(self.mem_id, self.mem_eff)

    @property
    def dram_GBps(self) -> float:
        return self.mem.effective_GBps

    @property
    def dram_bytes(self) -> float:
        return float(self.mem.capacity_bytes)


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
