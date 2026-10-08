"""Single-card peak-compute SKU presets.

100T / 1P labels are *single-card peak FLOPS templates* for sensitivity.
Assumed: 1P may later map to multi-chip; for now it is a bigger single-die
peak so decode can cross into the memory wall on LPDDR.

Peak math (derived):
  peak_FLOP/s = R * C * n_engines * frequency_Hz * 2   # 2 flop / MAC
  peak_TOPS   = peak_FLOP/s / 1e12

We pick n_engines and a near-square (R,C) so that peak_TOPS ≈ target at the
assumed GHz, rather than leaving a lone 64×64 (~8 TOPS @1GHz) as the only
default for 27B-class models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .npu import NPUConfig


@dataclass(frozen=True)
class SKUPreset:
    """Named single-card compute template."""

    name: str
    target_tops: float          # desired peak TOPS at frequency_hz
    frequency_hz: float
    n_engines: int
    rows: int
    cols: int
    process_nm: float = 4.0     # label only
    note: str = ""

    def npu(self) -> NPUConfig:
        return NPUConfig(
            rows=self.rows,
            cols=self.cols,
            n_engines=self.n_engines,
            mac_efficiency=1.0,
            process_nm=self.process_nm,
            sku_name=self.name,
        )

    def achieved_tops(self) -> float:
        return self.npu().peak_tops_at(self.frequency_hz)

    def summary(self) -> str:
        ach = self.achieved_tops()
        return (
            f"{self.name}: target={self.target_tops:.1f} TOPS → "
            f"PE={self.rows}x{self.cols} × {self.n_engines} eng @ "
            f"{self.frequency_hz/1e9:.2f}GHz → {ach:.2f} TOPS (derived) "
            f"| process_nm={self.process_nm} (label) {self.note}"
        )


def macs_needed_for_tops(target_tops: float, frequency_hz: float) -> float:
    """MAC units required: TOPS*1e12 / (2 flop/MAC) / Hz."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    return target_tops * 1e12 / (2.0 * frequency_hz)


def derive_pe_geometry(
    target_tops: float,
    frequency_hz: float = 1.0e9,
    n_engines: int = 1,
    prefer_multiple_of: int = 8,
) -> tuple[int, int, int]:
    """Choose near-square (R,C) and n_engines to hit target TOPS.

    Returns (rows, cols, n_engines).
    Side length rounded up to prefer_multiple_of for tidy tiling.
    """
    if n_engines <= 0:
        raise ValueError("n_engines must be positive")
    macs = macs_needed_for_tops(target_tops, frequency_hz)
    per_eng = macs / n_engines
    side = max(1, int(math.ceil(math.sqrt(per_eng))))
    if prefer_multiple_of > 1:
        side = int(math.ceil(side / prefer_multiple_of) * prefer_multiple_of)
    return side, side, n_engines


def make_sku(
    name: str,
    target_tops: float,
    frequency_hz: float = 1.0e9,
    n_engines: int = 16,
    process_nm: float = 4.0,
    note: str = "",
) -> SKUPreset:
    r, c, e = derive_pe_geometry(target_tops, frequency_hz, n_engines)
    return SKUPreset(
        name=name,
        target_tops=target_tops,
        frequency_hz=frequency_hz,
        n_engines=e,
        rows=r,
        cols=c,
        process_nm=process_nm,
        note=note,
    )


# ---------------------------------------------------------------------------
# Assumed presets (uncalibrated peak templates)
# ---------------------------------------------------------------------------

# ~8.2 TOPS @1GHz — legacy tiny default; decode often compute-bound on LPDDR.
SKU_BASELINE = SKUPreset(
    name="sku_baseline",
    target_tops=8.192,
    frequency_hz=1.0e9,
    n_engines=1,
    rows=64,
    cols=64,
    process_nm=4.0,
    note="[assumed] tiny PE; util(M=1)=1/64",
)

# ~100 TOPS class: 16 engines × 56×56 @1GHz
# macs = 16*56*56 = 50176; FLOP/s = 50176*2*1e9 = 1.00352e14 → 100.35 TOPS
SKU_100T = SKUPreset(
    name="sku_100t",
    target_tops=100.0,
    frequency_hz=1.0e9,
    n_engines=16,
    rows=56,
    cols=56,
    process_nm=4.0,
    note="[assumed] single-card ~100T peak template",
)

# ~1 PFLOPS = 1000 TOPS class as *single-die peak for sensitivity*
# (later may become multi-chip; not modeled yet).
# 64 eng × 88×88 = 495616 MACs; *2*1e9 / 1e12 = 991.23 TOPS ≈ 0.99 P
SKU_1P = SKUPreset(
    name="sku_1p",
    target_tops=1000.0,
    frequency_hz=1.0e9,
    n_engines=64,
    rows=88,
    cols=88,
    process_nm=4.0,
    note="[assumed] single-die 1P-class peak for sensitivity (not multi-chip yet)",
)

SKU_REGISTRY: dict[str, SKUPreset] = {
    SKU_BASELINE.name: SKU_BASELINE,
    SKU_100T.name: SKU_100T,
    SKU_1P.name: SKU_1P,
}


def get_sku(name: str) -> SKUPreset:
    key = name.strip().lower().replace("-", "_")
    if key not in SKU_REGISTRY:
        raise KeyError(f"unknown SKU {name!r}; choose from {list(SKU_REGISTRY)}")
    return SKU_REGISTRY[key]
