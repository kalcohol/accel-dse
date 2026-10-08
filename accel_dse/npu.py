"""NPU PE array and GEMM utilization model.

Dataflow choice: **output-stationary (OS)** systolic-style array.
  - An R×C array of MAC units per engine; each PE accumulates one output element.
  - n_engines replicate (or column-parallelize) the array — SKU peak scales with engines.
  - Ideal cycle count for GEMM (M×K)·(K×N) → (M×N):
        cycles = ceil(M / R) * ceil(N / (C * n_engines)) * K
    (n_engines=1 recovers the single-array formula.)
  - Peak MACs/cycle = R * C * n_engines  (assumed; uncalibrated silicon).
  - Utilization = (M * K * N) / (R * C * n_engines * cycles)

Decode M=1: utilization collapses roughly toward 1/R when N fills the
effective width (explicit small-M curve — see `gemm_utilization` / README).
Absolute decode throughput still scales with C * n_engines * freq even when
util≈1/R — that is how 100T-class SKUs expose the memory wall.

This is an analytical bound, NOT cycle-accurate RTL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NPUConfig:
    """Assumed NPU PE geometry and efficiency knobs.

    All fields are *assumed / uncalibrated* unless noted.
    process_nm is a label only — no PDK-derived area/power.
    """

    rows: int = 64          # R: OS tile along M
    cols: int = 64          # C: OS tile along N (per engine)
    n_engines: int = 1      # parallel engines (column-parallel along N)
    # Extra efficiency factor on top of geometric util (pipeline bubbles, etc.)
    # Assumed; default 1.0 = pure geometric model.
    mac_efficiency: float = 1.0
    process_nm: float = 4.0  # label only
    dataflow: str = "output_stationary"
    # Optional human label (e.g. sku name); not used in math
    sku_name: str = ""

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.n_engines <= 0:
            raise ValueError("n_engines must be positive")

    @property
    def peak_macs_per_cycle(self) -> int:
        return self.rows * self.cols * self.n_engines

    @property
    def effective_cols(self) -> int:
        """Column width after engine parallelization along N."""
        return self.cols * self.n_engines

    def peak_flops_per_cycle(self) -> int:
        """FLOPs ≈ 2 × MAC (multiply-add counted as 2 flops)."""
        return 2 * self.peak_macs_per_cycle

    def peak_tops_at(self, frequency_hz: float) -> float:
        """Peak Tera-OPS (FLOP/s / 1e12) at assumed frequency."""
        return self.peak_flops_per_cycle() * frequency_hz / 1e12

    def peak_macs_per_s(self, frequency_hz: float) -> float:
        return self.peak_macs_per_cycle * frequency_hz


def gemm_cycles(m: int, k: int, n: int, npu: NPUConfig) -> int:
    """Ideal OS systolic cycles for one GEMM (M×K)×(K×N).

    Engines expand effective columns: ceil(N / (C * n_engines)).
    """
    if m <= 0 or k <= 0 or n <= 0:
        return 0
    return (
        math.ceil(m / npu.rows)
        * math.ceil(n / npu.effective_cols)
        * k
    )


def gemm_macs(m: int, k: int, n: int) -> int:
    return m * k * n


def gemm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def gemm_utilization(m: int, k: int, n: int, npu: NPUConfig) -> float:
    """Geometric OS utilization in [0, 1].

    Formula:
      util = (M*K*N) / (R*C*n_engines * cycles)
           = (M * N) / (R * C * n_engines * ceil(M/R) * ceil(N/(C*n_engines)))

    Explicit M=1 curve (K cancels, N multiple of effective cols):
      util(M=1) = 1 / R
      util(large M) → 1.0 when M % R == 0 and N % (C*n_engines) == 0

    So decode (M=1) util is strictly less than large-M util for R > 1.
    """
    cycles = gemm_cycles(m, k, n, npu)
    if cycles == 0:
        return 0.0
    peak = npu.peak_macs_per_cycle * cycles
    return (gemm_macs(m, k, n) / peak) * npu.mac_efficiency


def gemm_compute_cycles(m: int, k: int, n: int, npu: NPUConfig) -> float:
    """Cycles after applying mac_efficiency (assumed bubble factor)."""
    raw = gemm_cycles(m, k, n, npu)
    if npu.mac_efficiency <= 0:
        raise ValueError("mac_efficiency must be > 0")
    return raw / npu.mac_efficiency


def utilization_vs_m(
    m_values: list[int],
    k: int,
    n: int,
    npu: NPUConfig,
) -> list[tuple[int, float]]:
    """Return (M, util) pairs — the explicit small-M utilization curve."""
    return [(m, gemm_utilization(m, k, n, npu)) for m in m_values]


# Default assumed PE (uncalibrated). 64×64 × 1 engine = 4096 MAC/cycle ≈ 8.2 TOPS @1GHz.
DEFAULT_NPU = NPUConfig(rows=64, cols=64, n_engines=1, mac_efficiency=1.0, process_nm=4.0)


# ---------------------------------------------------------------------------
# Product language: cores × tops_per_core ↔ PE / engines
# ---------------------------------------------------------------------------

# Assumed mapping (documented; needs_lock for real micro-arch):
#   - One "core" ≡ one OS engine (n_engines = n_cores).
#   - Each core has a near-square PE array sized so
#       peak_TOPS_per_core = 2 * rows * cols * frequency_Hz / 1e12
#     ≈ tops_per_core (derived geometry).
#   - Total peak_TOPS = n_cores * tops_per_core.
#   - Inverse: given PE rows/cols/engines → tops_per_core =
#       peak_TOPS / n_engines at the assumed frequency.


def _near_square_side(macs_per_engine: float, prefer_multiple_of: int = 8) -> int:
    import math

    side = max(1, int(math.ceil(math.sqrt(max(macs_per_engine, 1.0)))))
    if prefer_multiple_of > 1:
        side = int(math.ceil(side / prefer_multiple_of) * prefer_multiple_of)
    return side


def npu_from_cores(
    n_cores: int,
    tops_per_core: float,
    *,
    frequency_hz: float = 1.0e9,
    mac_efficiency: float = 1.0,
    process_nm: float = 4.0,
    sku_name: str = "",
    prefer_multiple_of: int = 8,
) -> NPUConfig:
    """Build NPUConfig from product language ``n_cores × tops_per_core``.

    Assumed mapping: ``n_engines = n_cores``; PE rows=cols derived so each
    engine's peak TOPS ≈ ``tops_per_core`` at ``frequency_hz``.
    """
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")
    if tops_per_core <= 0:
        raise ValueError("tops_per_core must be > 0")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    # macs_per_engine = tops_per_core * 1e12 / (2 * freq)
    macs = tops_per_core * 1e12 / (2.0 * frequency_hz)
    side = _near_square_side(macs, prefer_multiple_of)
    name = sku_name or f"cores_{n_cores}x{tops_per_core:g}T"
    return NPUConfig(
        rows=side,
        cols=side,
        n_engines=int(n_cores),
        mac_efficiency=mac_efficiency,
        process_nm=process_nm,
        sku_name=name,
    )


def cores_from_npu(
    npu: NPUConfig,
    frequency_hz: float = 1.0e9,
) -> tuple[int, float]:
    """Inverse: (n_cores, tops_per_core) from PE/engines layout.

    Assumed: n_cores = n_engines; tops_per_core = total_peak_TOPS / n_engines.
    """
    total = npu.peak_tops_at(frequency_hz)
    n_cores = npu.n_engines
    return n_cores, total / n_cores


def npu_from_pe(
    rows: int,
    cols: int,
    n_engines: int = 1,
    *,
    mac_efficiency: float = 1.0,
    process_nm: float = 4.0,
    sku_name: str = "",
) -> NPUConfig:
    """Direct PE rows/cols/engines constructor (keeps legacy path)."""
    return NPUConfig(
        rows=rows,
        cols=cols,
        n_engines=n_engines,
        mac_efficiency=mac_efficiency,
        process_nm=process_nm,
        sku_name=sku_name or f"pe_{rows}x{cols}x{n_engines}",
    )
