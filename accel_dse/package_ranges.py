"""Design-space memory package catalog + compute hierarchy ranges (NOT silicon calib).

v0.29 — memory side is now built from the structured, provenance-tagged catalog
in ``mem_catalog.py`` (source: ``research/memory_specs_2026-10.{md,json}``).
All efficiencies remain **assumed / uncalibrated**; rates / widths / capacities
carry provenance tags (JEDEC / 疑似 JEDEC / 厂商量产 / 送样 / 已发布 / 推测).

------------------------------------------------------------------------
Modelling units (granule)
------------------------------------------------------------------------
  - LPDDR5 / LPDDR5X: **package** x32 (2×16) / **x64 (4×16, default)** / x96
    (6×16, LPDDR5X vendor-extended). SoC bus = n_packages × width.
  - LPDDR6: **package x96 = 4 × x24 channels = 8 × x12 sub-channels**
    (JESD209-6 defines the x24 *die*; shipping PoP packages — CXMT 16 GB 1295-ball,
    Xiaomi XRING O3 4×24-bit @10.667 = 113.8 GB/s — are x96). x48 = speculative.
    LPDDR6 payload factor **8/9** (BL24: 288 bit = 256 data + 16 meta + 16 DBI/ECC),
    kept separate from ``efficiency``.
  - SOCAMM2 (JESD328) / LPCAMM2 (JESD318): 128-bit **modules**.
  - HBM3 / HBM3E: 1024-bit **stack**; HBM4 / HBM4E: 2048-bit stack.
    Capacity = stacks × height × die_Gb / 8 (HBM3 16/24 GB, HBM3E 24/36/48,
    HBM4 24–64, HBM4E 32/48/64 GB per stack).

``ExternalMemory.n_channels`` keeps its name for API compatibility but means
**units** (packages / modules / stacks) — see ``ExternalMemory.unit_kind``.

Legacy (≤0.28) ids still resolve via ``get_package`` → ``mem_catalog.parse_mem_id``
(``lpddr_{n}x64_{rate}``, ``lpddr6_{n}x24_{rate}``, ``hbm_{gen}_{n}s``) with a
``legacy_note`` explaining the mapping.

Bandwidth (derived, same as memory.py)::

    raw_Bps       = n_units * (width_bits / 8) * data_rate_GTs * 1e9
    effective_Bps = raw_Bps * payload_factor * efficiency   # efficiency assumed

Compute hierarchy (assumed 1 GHz unless noted):
  - Single core peak TOPS options: 4, 8, 12, 16
  - Single cluster peak TOPS: 64, 128, 192, 256
    (= n_cores × tops_per_core at ASSUMED_FREQ_HZ)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

from .memory import ExternalMemory, MemKind, MemGeometry
from .mem_catalog import (
    HBM_CATALOG,
    LPDDR_CATALOG,
    MEM_TYPES,
    TAG_EN,
    MemSpec,
    make_spec,
    parse_mem_id,
)

ASSUMED_EFFICIENCY = 0.70  # sustained/payload placeholder — uncalibrated knob
ASSUMED_FREQ_HZ = 1.0e9

# Rate grades per generation (GT/s) — tied to generation (no "HBM3 @10" combos)
LPDDR5_RATE_BAND_GTS: tuple[float, ...] = tuple(
    r / 1000 for r in LPDDR_CATALOG["LPDDR5"]["forms"]["discrete"]["rates"]
)
LPDDR5X_RATE_BAND_GTS: tuple[float, ...] = tuple(
    r / 1000 for r in LPDDR_CATALOG["LPDDR5X"]["forms"]["discrete"]["rates"]
)
LPDDR_RATE_BAND_GTS = LPDDR5X_RATE_BAND_GTS  # legacy alias
LPDDR6_RATE_BAND_GTS: tuple[float, ...] = tuple(
    r / 1000 for r in LPDDR_CATALOG["LPDDR6"]["forms"]["discrete"]["rates"]
)

# HBM generation → (stack width, default GT/s, note) — legacy-shaped view
HBM_GEN_LABELS: dict[str, tuple[int, float, str]] = {
    g: (
        c["stack_width"],
        c["default_rate"] / 1000.0,
        f"{g} {c['stack_width']}b stack; grades "
        + "/".join(str(r) for r in c["rates"])
        + f" MT/s; {c['jedec_doc']}",
    )
    for g, c in HBM_CATALOG.items()
}

HBM_STACK_COUNTS: tuple[int, ...] = (2, 4, 6, 8)  # curated catalog subset

# LPDDR5/5X: n_packages of 64-bit packages (default granule)
LPDDR5X_PACKAGE_WIDTH_BITS = 64
LPDDR5X_N_PACKAGES: tuple[int, ...] = (2, 4, 8)
# LPDDR6: n_packages of 96-bit (4×24) packages
LPDDR6_PACKAGE_WIDTH_BITS = 96
LPDDR6_N_PACKAGES: tuple[int, ...] = (1, 2, 4, 6)

CORE_TOPS_GRID: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0)
CLUSTER_TOPS_GRID: tuple[float, ...] = (64.0, 128.0, 192.0, 256.0)


@dataclass(frozen=True)
class PackageOption:
    """Named memory configuration (thin view over ``MemSpec``).

    Field names are kept from ≤0.28 for callers; semantics (v0.29):
      - ``n_channels`` = units (packages / modules / stacks)
      - ``width_bits`` = unit width (x32/x64/x96 package, 128 module, 1024/2048 stack)
      - ``generation`` = mem_type (LPDDR5 / LPDDR5X / LPDDR6 / HBM3 / HBM3E / HBM4 / HBM4E)
      - ``capacity_GB`` = nominal vendor GB (= GiB)
    """

    id: str
    kind: MemKind
    n_channels: int
    width_bits: int
    data_rate_GTs: float
    efficiency: float = ASSUMED_EFFICIENCY
    capacity_GB: float = 64.0
    generation: str = ""
    n_stacks: int | None = None
    n_packages: int | None = None
    package_width_bits: int | None = None
    note: str = ""
    spec: MemSpec | None = None

    @classmethod
    def from_spec(cls, spec: MemSpec, *, pid: str | None = None) -> "PackageOption":
        hbm = spec.kind == "HBM"
        note = (
            f"{spec.form} {spec.n_units}×{spec.unit_width_bits}b @{spec.rate_MTps} MT/s; "
            f"tag={TAG_EN[spec.tag]}"
        )
        if spec.legacy_note:
            note += f"; {spec.legacy_note}"
        return cls(
            id=pid or spec.id,
            kind=spec.kind,  # type: ignore[arg-type]
            n_channels=spec.n_units,
            width_bits=spec.unit_width_bits,
            data_rate_GTs=spec.rate_MTps / 1000.0,
            efficiency=spec.efficiency,
            capacity_GB=spec.capacity_GB,
            generation=spec.mem_type,
            n_stacks=spec.n_units if hbm else None,
            n_packages=None if hbm else spec.n_units,
            package_width_bits=None if hbm else spec.unit_width_bits,
            note=note,
            spec=spec,
        )

    @property
    def mem_spec(self) -> MemSpec:
        if self.spec is not None:
            return self.spec
        return parse_mem_id(self.id)

    @property
    def bus_width_bits(self) -> int:
        return int(self.n_channels) * int(self.width_bits)

    @property
    def tag(self) -> str:
        return self.mem_spec.tag

    def to_mem_geometry(self) -> MemGeometry:
        return MemGeometry(
            kind=self.kind,
            n_channels=self.n_channels,
            width_bits=self.width_bits,
            data_rate_GTs=self.data_rate_GTs,
            efficiency=self.efficiency,
            capacity_GB=self.capacity_GB,
            n_packages=1,
        )

    def to_external_memory(self) -> ExternalMemory:
        return self.mem_spec.to_external_memory()

    def peak_bandwidth_GBps(self) -> float:
        """Raw pin bandwidth (before payload / efficiency)."""
        return self.mem_spec.raw_GBps

    def payload_bandwidth_GBps(self) -> float:
        return self.mem_spec.payload_GBps

    def effective_bandwidth_GBps(self) -> float:
        return self.mem_spec.effective_GBps

    def workbench_kwargs(self) -> dict:
        """Structured memory knobs for WorkbenchConfig (v0.29)."""
        kw = self.mem_spec.workbench_kwargs()
        if self.mem_spec.legacy_id:
            kw["mem_legacy_id"] = self.mem_spec.legacy_id
        return kw

    def summary(self) -> str:
        return self.mem_spec.summary() + (f" — {self.note}" if self.note else "")


@dataclass(frozen=True)
class ComputeOption:
    """Named core or cluster compute option (assumed 1 GHz peak math)."""

    id: str
    level: Literal["core", "cluster"]
    peak_tops: float
    n_cores: int
    tops_per_core: float
    frequency_hz: float = ASSUMED_FREQ_HZ
    note: str = ""

    def workbench_kwargs(self) -> dict:
        return {
            "n_cores": self.n_cores,
            "tops_per_core": self.tops_per_core,
            "frequency_hz": self.frequency_hz,
            "sku": None,
            "pe_rows": None,
            "pe_cols": None,
            "n_engines": None,
        }

    def summary(self) -> str:
        return (
            f"{self.id}: {self.level} peak={self.peak_tops:g} TOPS = "
            f"{self.n_cores} × {self.tops_per_core:g} T/core @ "
            f"{self.frequency_hz/1e9:.2f} GHz [assumed] — {self.note}"
        )


def _opts(specs: Iterable[MemSpec]) -> list[PackageOption]:
    return [PackageOption.from_spec(s) for s in specs]


def build_lpddr5_packages(
    n_packages: Sequence[int] = (2, 4, 8),
    efficiency: float = ASSUMED_EFFICIENCY,
) -> list[PackageOption]:
    """LPDDR5 x64 packages (12 GB each) × {5500, 6400}."""
    rates = LPDDR_CATALOG["LPDDR5"]["forms"]["discrete"]["rates"]
    return _opts(
        make_spec("LPDDR5", count=n, rate=r, efficiency=efficiency)
        for n in n_packages
        for r in rates
    )


def build_lpddr5x_packages(
    n_packages: Sequence[int] = LPDDR5X_N_PACKAGES,
    rates: Sequence[float] | None = None,
    efficiency: float = ASSUMED_EFFICIENCY,
) -> list[PackageOption]:
    """LPDDR5X x64 packages (16 GB each) × {7500, 8533, 9600, 10667}."""
    rr = rates or tuple(LPDDR_CATALOG["LPDDR5X"]["forms"]["discrete"]["rates"])
    return _opts(
        make_spec("LPDDR5X", count=n, rate=r, efficiency=efficiency)
        for n in n_packages
        for r in rr
    )


def build_lpddr6_packages(
    n_packages: Sequence[int] = LPDDR6_N_PACKAGES,
    rates: Sequence[float] | None = None,
    efficiency: float = ASSUMED_EFFICIENCY,
) -> list[PackageOption]:
    """LPDDR6 **x96 packages** (4×24-bit ch, 8×12-bit sub-ch; 16 GB) × rates.

    Payload 8/9 applied on top of raw (separate from ``efficiency``).
    """
    rr = rates or tuple(LPDDR_CATALOG["LPDDR6"]["forms"]["discrete"]["rates"])
    return _opts(
        make_spec("LPDDR6", count=n, rate=r, efficiency=efficiency)
        for n in n_packages
        for r in rr
    )


def build_lpddr_module_packages(efficiency: float = ASSUMED_EFFICIENCY) -> list[PackageOption]:
    """SOCAMM2 (JESD328) and LPCAMM2 (JESD318) LPDDR5X modules."""
    out = [
        make_spec("LPDDR5X", form="SOCAMM2", count=n, rate=r, efficiency=efficiency)
        for n in (4, 8)
        for r in (8533, 9600)
    ]
    out += [
        make_spec("LPDDR5X", form="LPCAMM2", count=n, efficiency=efficiency)
        for n in (1, 2)
    ]
    return _opts(out)


def build_lpddr_packages(
    *,
    include_lpddr5x: bool = True,
    include_lpddr6: bool = True,
    include_lpddr5: bool = True,
    include_modules: bool = True,
    efficiency: float = ASSUMED_EFFICIENCY,
    rates: Sequence[float] | None = None,  # legacy, ignored
    ch_width: Sequence[tuple[int, int, str]] | None = None,  # legacy, ignored
) -> list[PackageOption]:
    """Curated LPDDR catalog: LPDDR5 / LPDDR5X / LPDDR6 packages + modules."""
    _ = rates, ch_width
    out: list[PackageOption] = []
    if include_lpddr5:
        out.extend(build_lpddr5_packages(efficiency=efficiency))
    if include_lpddr5x:
        out.extend(build_lpddr5x_packages(efficiency=efficiency))
    if include_lpddr6:
        out.extend(build_lpddr6_packages(efficiency=efficiency))
    if include_modules:
        out.extend(build_lpddr_module_packages(efficiency=efficiency))
    return out


def build_hbm_packages(
    stacks: Sequence[int] = HBM_STACK_COUNTS,
    generations: Sequence[str] = tuple(HBM_CATALOG.keys()),
    efficiency: float = ASSUMED_EFFICIENCY,
    *,
    extra_rates: bool = True,
) -> list[PackageOption]:
    """HBM stacks × gen defaults (12-high, default die) + 8-stack rate variants."""
    out: list[MemSpec] = []
    for gen in generations:
        g = str(gen).upper()
        if g not in HBM_CATALOG:
            raise KeyError(f"unknown HBM generation {gen!r}")
        for n in stacks:
            out.append(make_spec(g, count=n, efficiency=efficiency))
        if extra_rates:
            for r in HBM_CATALOG[g]["rates"]:
                if r != HBM_CATALOG[g]["default_rate"]:
                    out.append(make_spec(g, count=8, rate=r, efficiency=efficiency))
    return _opts(out)


def build_package_catalog(
    *,
    include_lpddr: bool = True,
    include_hbm: bool = True,
) -> dict[str, PackageOption]:
    pkgs: list[PackageOption] = []
    if include_lpddr:
        pkgs.extend(build_lpddr_packages())
    if include_hbm:
        pkgs.extend(build_hbm_packages())
    return {p.id: p for p in pkgs}


PACKAGE_CATALOG: dict[str, PackageOption] = build_package_catalog()


def get_package(package_id: str) -> PackageOption:
    """Canonical catalog id, any parseable structured id, or legacy ≤0.28 id."""
    key = package_id.strip().lower().replace("-", "_")
    if key in PACKAGE_CATALOG:
        return PACKAGE_CATALOG[key]
    try:
        spec = parse_mem_id(key)
    except (KeyError, ValueError) as exc:
        raise KeyError(
            f"unknown package {package_id!r}; "
            f"choose from list-packages ({len(PACKAGE_CATALOG)} options) or a structured id "
            f"like lpddr6_4x96_10667_16g / hbm3e_8s_12h24g_9200"
        ) from exc
    return PackageOption.from_spec(spec, pid=spec.legacy_id or spec.id)


def list_packages(
    *,
    kind: MemKind | str | None = None,
) -> list[PackageOption]:
    rows = list(PACKAGE_CATALOG.values())
    if kind is not None:
        k = str(kind).upper()
        if k in MEM_TYPES:
            rows = [p for p in rows if p.generation == k]
        else:
            rows = [p for p in rows if p.kind == k]
    return rows


def format_package_table(packages: Iterable[PackageOption] | None = None) -> str:
    pkgs = list(packages) if packages is not None else list_packages()
    hdr = (
        f"{'id':34s}  {'type':7s}  {'form':8s}  {'n':>3}  {'w':>5}  {'bus':>6}  "
        f"{'MT/s':>6}  {'raw_GBps':>9}  {'pay_GBps':>9}  {'eff_GBps':>9}  "
        f"{'cap_GB':>7}  tag"
    )
    lines = [hdr, "-" * len(hdr)]
    for p in pkgs:
        s = p.mem_spec
        lines.append(
            f"{p.id:34s}  {s.mem_type:7s}  {s.form:8s}  {s.n_units:3d}  "
            f"{s.unit_width_bits:5d}  {s.bus_bits:6d}  {s.rate_MTps:6d}  "
            f"{s.raw_GBps:9.1f}  {s.payload_GBps:9.1f}  {s.effective_GBps:9.1f}  "
            f"{s.capacity_GB:7.0f}  {TAG_EN[s.tag]}"
        )
    lines.append(
        "\n(units: LPDDR5/5X x64 packages, LPDDR6 x96 packages (payload 8/9), "
        "SOCAMM2/LPCAMM2 128-bit modules, HBM stacks; GB = 2^30 B; "
        "efficiency assumed — not silicon calib. peak_GBps = raw_GBps)"
    )
    lines.append(f"({len(pkgs)} packages)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compute hierarchy catalog
# ---------------------------------------------------------------------------


def _cluster_decompositions(cluster_tops: float) -> list[tuple[int, float]]:
    """Example (n_cores, tops_per_core) that hit cluster_tops at 1 GHz.

    Prefer tidy factors from CORE_TOPS_GRID and powers-of-two core counts.
    """
    out: list[tuple[int, float]] = []
    for tpc in CORE_TOPS_GRID:
        if cluster_tops % tpc == 0:
            n = int(round(cluster_tops / tpc))
            if n >= 1:
                out.append((n, float(tpc)))
    # Also allow tops_per_core not on the core grid if needed (e.g. 192/24)
    if not out:
        for n in (4, 8, 12, 16, 32, 48, 64):
            tpc = cluster_tops / n
            if tpc >= 1.0:
                out.append((n, float(tpc)))
    # Dedup preserving order
    seen: set[tuple[int, float]] = set()
    uniq: list[tuple[int, float]] = []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    return uniq


def build_compute_catalog(
    core_tops: Sequence[float] = CORE_TOPS_GRID,
    cluster_tops: Sequence[float] = CLUSTER_TOPS_GRID,
    frequency_hz: float = ASSUMED_FREQ_HZ,
) -> dict[str, ComputeOption]:
    cat: dict[str, ComputeOption] = {}
    for t in core_tops:
        cid = f"core_{int(t) if float(t).is_integer() else t}t"
        cat[cid] = ComputeOption(
            id=cid,
            level="core",
            peak_tops=float(t),
            n_cores=1,
            tops_per_core=float(t),
            frequency_hz=frequency_hz,
            note="single-core peak TOPS option [assumed]",
        )
    for ct in cluster_tops:
        decomps = _cluster_decompositions(float(ct))
        # Primary catalog entry uses preferred decomposition (16 cores if possible)
        preferred = None
        for n, tpc in decomps:
            if n == 16:
                preferred = (n, tpc)
                break
        if preferred is None:
            preferred = decomps[0]
        n0, tpc0 = preferred
        cid = f"cluster_{int(ct) if float(ct).is_integer() else ct}t"
        decomp_str = ", ".join(f"{n}×{tpc:g}" for n, tpc in decomps)
        cat[cid] = ComputeOption(
            id=cid,
            level="cluster",
            peak_tops=float(ct),
            n_cores=n0,
            tops_per_core=tpc0,
            frequency_hz=frequency_hz,
            note=(
                f"cluster peak; primary {n0}×{tpc0:g}; "
                f"alts=[{decomp_str}] @ {frequency_hz/1e9:.0f} GHz [assumed]"
            ),
        )
        # Also register explicit decomposition aliases
        for n, tpc in decomps:
            aid = f"cluster_{int(ct)}t_{n}x{int(tpc) if float(tpc).is_integer() else tpc}"
            if aid in cat:
                continue
            cat[aid] = ComputeOption(
                id=aid,
                level="cluster",
                peak_tops=float(ct),
                n_cores=n,
                tops_per_core=tpc,
                frequency_hz=frequency_hz,
                note=f"cluster {ct:g}T via {n}×{tpc:g} [assumed]",
            )
    return cat


COMPUTE_CATALOG: dict[str, ComputeOption] = build_compute_catalog()


def get_compute(compute_id: str) -> ComputeOption:
    key = compute_id.strip().lower().replace("-", "_")
    if key not in COMPUTE_CATALOG:
        raise KeyError(
            f"unknown compute option {compute_id!r}; "
            f"choose from list-compute ({len(COMPUTE_CATALOG)} options)"
        )
    return COMPUTE_CATALOG[key]


def list_compute(
    *,
    level: Literal["core", "cluster", "all"] = "all",
) -> list[ComputeOption]:
    rows = list(COMPUTE_CATALOG.values())
    if level == "core":
        rows = [c for c in rows if c.level == "core"]
    elif level == "cluster":
        # Primary cluster entries only (id == cluster_Nt without _NxM suffix
        # after the tops tag) — keep aliases but sort cores first then clusters
        rows = [c for c in rows if c.level == "cluster"]
    return rows


def list_compute_primaries() -> list[ComputeOption]:
    """Core grid + one primary cluster entry per cluster_tops (no decomp aliases)."""
    cores = [c for c in COMPUTE_CATALOG.values() if c.level == "core"]
    clusters = []
    for ct in CLUSTER_TOPS_GRID:
        cid = f"cluster_{int(ct)}t"
        if cid in COMPUTE_CATALOG:
            clusters.append(COMPUTE_CATALOG[cid])
    return cores + clusters


def format_compute_table(
    options: Iterable[ComputeOption] | None = None,
    *,
    primaries_only: bool = True,
) -> str:
    opts = (
        list(options)
        if options is not None
        else (list_compute_primaries() if primaries_only else list_compute())
    )
    hdr = (
        f"{'id':28s}  {'level':8s}  {'peak_T':>8}  {'n_cores':>7}  "
        f"{'T/core':>8}  {'GHz':>5}"
    )
    lines = [hdr, "-" * len(hdr)]
    for c in opts:
        lines.append(
            f"{c.id:28s}  {c.level:8s}  {c.peak_tops:8.1f}  "
            f"{c.n_cores:7d}  {c.tops_per_core:8.2f}  "
            f"{c.frequency_hz/1e9:5.2f}"
        )
    lines.append(
        f"\n({len(opts)} compute options; peak math at assumed "
        f"{ASSUMED_FREQ_HZ/1e9:.0f} GHz — uncalibrated)"
    )
    return "\n".join(lines)


# Default sweep grids derived from catalogs (for workbench-scan / API)
# LPDDR: LPDDR5X 2/4/8 × x64 @8533 + LPDDR6 2/4 × x96 @10667 + SOCAMM2 8 × @9600
DEFAULT_PACKAGE_SWEEP_LPDDR: tuple[str, ...] = tuple(
    [make_spec("LPDDR5X", count=n).id for n in (2, 4, 8)]
    + [make_spec("LPDDR5X", count=8, rate=r).id for r in (7500, 9600, 10667)]
    + [make_spec("LPDDR6", count=n).id for n in (2, 4, 6)]
    + [make_spec("LPDDR5X", form="SOCAMM2").id]
)

DEFAULT_PACKAGE_SWEEP_HBM: tuple[str, ...] = tuple(
    make_spec(g, count=n).id for g in ("HBM3", "HBM3E", "HBM4", "HBM4E") for n in (2, 4, 8)
)

DEFAULT_PACKAGE_SWEEP: tuple[str, ...] = (
    DEFAULT_PACKAGE_SWEEP_LPDDR + DEFAULT_PACKAGE_SWEEP_HBM
)

DEFAULT_COMPUTE_SWEEP_CORE: tuple[str, ...] = tuple(
    f"core_{int(t)}t" for t in CORE_TOPS_GRID
)
DEFAULT_COMPUTE_SWEEP_CLUSTER: tuple[str, ...] = tuple(
    f"cluster_{int(t)}t" for t in CLUSTER_TOPS_GRID
)
DEFAULT_COMPUTE_SWEEP: tuple[str, ...] = (
    DEFAULT_COMPUTE_SWEEP_CORE + DEFAULT_COMPUTE_SWEEP_CLUSTER
)

# Geometry sweep defaults (``--sweep-mem`` without --sweep-package).
# v0.29: rates are tied to the preset generation (HBM_PRESET = HBM3E,
# LPDDR_PRESET = LPDDR5X) — no more "HBM3 @10.0" style invalid combos.
DEFAULT_MEM_CHANNEL_SWEEP_LPDDR: tuple[int, ...] = (2, 4, 6, 8)  # x64 packages
DEFAULT_MEM_WIDTH_SWEEP_LPDDR: tuple[int, ...] = (64,)  # granule = x64 package
DEFAULT_MEM_RATE_SWEEP_LPDDR_V16: tuple[float, ...] = LPDDR5X_RATE_BAND_GTS
DEFAULT_MEM_CHANNEL_SWEEP_HBM: tuple[int, ...] = HBM_STACK_COUNTS  # stacks
DEFAULT_MEM_RATE_SWEEP_HBM_V16: tuple[float, ...] = tuple(
    r / 1000 for r in HBM_CATALOG["HBM3E"]["rates"]
)


def package_counts() -> dict[str, int]:
    lp = list_packages(kind="LPDDR")
    hb = list_packages(kind="HBM")
    by = {t: len([p for p in lp + hb if p.generation == t]) for t in MEM_TYPES}
    return {
        "lpddr": len(lp),
        "lpddr5": by["LPDDR5"],
        "lpddr5x": by["LPDDR5X"],
        "lpddr6": by["LPDDR6"],
        "hbm": len(hb),
        "hbm3": by["HBM3"],
        "hbm3e": by["HBM3E"],
        "hbm4": by["HBM4"],
        "hbm4e": by["HBM4E"],
        "total": len(lp) + len(hb),
        "compute_core": len([c for c in COMPUTE_CATALOG.values() if c.level == "core"]),
        "compute_cluster_primary": len(CLUSTER_TOPS_GRID),
        "compute_total": len(COMPUTE_CATALOG),
    }
