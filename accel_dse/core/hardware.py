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
System = Chip + external memory (mem_catalog.MemSpec) + scale-up link.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .. import mem_catalog
from .dtypes import FormatSupport


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

    def __post_init__(self):
        for k in ("freq_ghz", "rows", "cols", "engines", "sram_mib", "mac_eff"):
            v = getattr(self, k)
            if not (v == v) or v <= 0 or v == float("inf"):
                raise ValueError(f"chip.{k} must be finite and > 0 (got {v})")

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


@dataclass(frozen=True)
class Link:
    """Scale-up link per rank.  α-β: t = α + bytes/β (per collective step)."""
    GBps: float = 400.0       # 「假设」 per-rank uni-directional bandwidth
    alpha_us: float = 3.0     # 「假设」 per-collective latency (sync + launch)
    topology: str = "switch"  # switch | ring

    def __post_init__(self):
        if not (self.GBps > 0 and self.GBps != float("inf")) or not (self.alpha_us >= 0):
            raise ValueError("link GBps must be finite > 0, alpha_us ≥ 0")


@dataclass(frozen=True)
class System:
    chip: Chip
    mem_id: str = "lpddr5x_4x64_8533_16g"
    mem_eff: float | None = None
    link: Link = Link()

    @property
    def mem(self) -> mem_catalog.MemSpec:
        return mem_catalog.parse_mem_id(self.mem_id, efficiency=self.mem_eff)

    @property
    def dram_GBps(self) -> float:
        return self.mem.effective_GBps

    @property
    def dram_bytes(self) -> float:
        return float(self.mem.capacity_bytes)


# Presets (geometry from the 0.3x SKU templates; all 「假设」)
CHIP_100T = Chip("100T", 1.0, 56, 56, 16)            # 100.35 TFLOPS bf16
CHIP_1P = Chip("1P", 1.0, 128, 128, 32, sram_mib=256.0)   # 1048.6 TFLOPS bf16
CHIP_H100_LIKE = Chip("H100-like", 1.83, 128, 128, 16,      # ≈ 959 TFLOPS dense bf16
                      formats=FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0))),
                      sram_mib=50.0, vector_lanes=16384, sram_port_Bpc=None)
CHIPS = {c.name: c for c in (CHIP_100T, CHIP_1P, CHIP_H100_LIKE)}
