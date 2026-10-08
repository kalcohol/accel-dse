"""Structured external-memory catalog (v0.29) — LPDDR5/5X/6 + HBM3/3E/4/4E.

Source of truth: ``docs/research/memory_specs_2026-10.{md,json}`` (verified 2026-10-08,
§6 "proposed_selectors"). This module keeps a compact, data-driven copy of the
selector table so the engine has no file-system dependency at import time.

Modelling unit (granule) — *not* the JEDEC die channel:
  - LPDDR5 / LPDDR5X discrete: **package** (x32 = 2×16, **x64 = 4×16 default**,
    x96 = 6×16 vendor-extended LPDDR5X)
  - LPDDR6 discrete: **package x96 = 4 × x24 channels = 8 × x12 sub-channels**
    (JESD209-6 defines the x24 *die*; shipping PoP packages are x96). x48 is
    speculative only.
  - SOCAMM2 / LPCAMM2: **module** (128-bit; LPDDR6 CAMM2 192-bit speculative)
  - HBM: **stack** (1024-bit HBM3/3E; 2048-bit HBM4/4E)

Derived (no hidden constants)::

    bus_bits      = n_units × unit_width_bits
    raw_GBps      = bus_bits × MT/s / 8 / 1000
    payload_GBps  = raw_GBps × payload_factor      # LPDDR6 = 256/288 = 8/9
    effective     = payload_GBps × efficiency      # efficiency: assumed knob
    capacity      = n_units × GB/unit              # HBM: stacks × height × die_Gb / 8
                    (vendor "GB" = 2^30 B, JEDEC binary)

Provenance tag of a combo = the **weakest** of its component tags
(speed grade, unit width, capacity / stack-height×density, unit count):

    JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测

Values outside the catalog lists are accepted (DSE what-ifs) but tagged 推测.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

GIB = 2**30

TAG_ORDER: tuple[str, ...] = (
    "jedec",
    "jedec_likely",
    "vendor_shipping",
    "vendor_sampling",
    "vendor_announced",
    "speculative",
)
TAG_ZH: dict[str, str] = {
    "jedec": "JEDEC",
    "jedec_likely": "疑似 JEDEC",
    "vendor_shipping": "厂商量产",
    "vendor_sampling": "送样",
    "vendor_announced": "已发布",
    "speculative": "推测",
}
TAG_EN: dict[str, str] = {
    "jedec": "JEDEC",
    "jedec_likely": "JEDEC?",
    "vendor_shipping": "vendor-shipping",
    "vendor_sampling": "vendor-sampling",
    "vendor_announced": "vendor-announced",
    "speculative": "speculative",
}

LPDDR6_PAYLOAD = 256.0 / 288.0  # BL24: 288 bit = 256 data + 16 meta + 16 DBI/ECC

MEM_TYPES: tuple[str, ...] = (
    "LPDDR5", "LPDDR5X", "LPDDR6", "HBM3", "HBM3E", "HBM4", "HBM4E",
)
FORM_ZH: dict[str, str] = {
    "discrete": "板载封装",
    "SOCAMM2": "SOCAMM2 模组",
    "LPCAMM2": "LPCAMM2 模组",
    "stack": "HBM 堆栈",
}
UNIT_ZH: dict[str, str] = {"package": "颗", "module": "条", "stack": "堆"}


def weakest(*tags: Any) -> str:
    """Weakest provenance tag among ``tags`` (unknown → speculative).

    Accepts ``weakest("jedec", "vendor_shipping")`` or one iterable of tags.
    """
    if len(tags) == 1 and not isinstance(tags[0], str):
        tags = tuple(tags[0])
    worst = 0
    for t in tags:
        i = TAG_ORDER.index(t) if t in TAG_ORDER else len(TAG_ORDER) - 1
        worst = max(worst, i)
    return TAG_ORDER[worst]


# ---------------------------------------------------------------------------
# Catalog data (compact copy of docs/research/memory_specs_2026-10.json §6)
# ---------------------------------------------------------------------------
# Each LPDDR form: unit, widths{bits: tag}, rates{MTps: tag}, counts, caps{width:{GB:tag}}
# Defaults chosen per research §6.2 (bold values).

_LP5_CAPS = {3: "vendor_shipping", 6: "vendor_shipping", 8: "vendor_shipping", 12: "vendor_shipping"}

LPDDR_CATALOG: dict[str, dict[str, Any]] = {
    "LPDDR5": {
        "jedec_doc": "JESD209-5 / 5C",
        "payload_factor": 1.0,
        "clock": "lp5",
        "forms": {
            "discrete": {
                "unit": "package",
                "widths": {32: "vendor_shipping", 64: "vendor_shipping"},
                "default_width": 64,
                "rates": {5500: "jedec", 6400: "jedec"},
                "default_rate": 6400,
                "counts": (1, 2, 4, 6, 8),
                "default_count": 4,
                "caps": {32: dict(_LP5_CAPS), 64: dict(_LP5_CAPS)},
                "default_cap": {32: 6, 64: 12},
            },
        },
    },
    "LPDDR5X": {
        "jedec_doc": "JESD209-5B/5C (≤8533 明文；9600/10667 疑似)",
        "payload_factor": 1.0,
        "clock": "lp5",
        "forms": {
            "discrete": {
                "unit": "package",
                # x96 (6×16) = Micron "6-channel" (secondary report) → 已发布
                "widths": {32: "vendor_shipping", 64: "vendor_shipping", 96: "vendor_announced"},
                "default_width": 64,
                "rates": {7500: "jedec_likely", 8533: "jedec", 9600: "jedec_likely", 10667: "jedec_likely"},
                "default_rate": 8533,
                "counts": (1, 2, 3, 4, 6, 8, 10, 12, 16),
                "default_count": 8,
                "caps": {
                    32: {8: "vendor_shipping", 16: "vendor_shipping", 32: "vendor_shipping", 64: "vendor_shipping"},
                    64: {4: "vendor_shipping", 6: "vendor_shipping", 8: "vendor_shipping",
                         12: "vendor_shipping", 16: "vendor_shipping", 24: "vendor_shipping",
                         32: "vendor_announced"},
                    96: {12: "vendor_announced", 16: "vendor_announced", 24: "vendor_announced"},
                },
                "default_cap": {32: 16, 64: 16, 96: 16},
            },
            "SOCAMM2": {
                "unit": "module",
                "widths": {128: "jedec"},  # JESD328 (2026-06)
                "default_width": 128,
                "rates": {8533: "vendor_shipping", 9600: "vendor_shipping"},
                "default_rate": 9600,
                "counts": (1, 2, 4, 6, 8),
                "default_count": 8,
                "caps": {128: {48: "vendor_shipping", 64: "vendor_shipping", 96: "vendor_shipping",
                               128: "vendor_shipping", 192: "vendor_shipping", 256: "vendor_shipping"}},
                "default_cap": {128: 192},
            },
            "LPCAMM2": {
                "unit": "module",
                "widths": {128: "jedec"},  # JESD318 CAMM2
                "default_width": 128,
                "rates": {7500: "vendor_shipping", 8533: "vendor_shipping", 9600: "speculative"},
                "default_rate": 8533,
                "counts": (1, 2),
                "default_count": 2,
                "caps": {128: {16: "vendor_shipping", 32: "vendor_shipping", 64: "vendor_shipping",
                               96: "vendor_announced"}},
                "default_cap": {128: 32},
            },
        },
    },
    "LPDDR6": {
        "jedec_doc": "JESD209-6 (2025-07)",
        "payload_factor": LPDDR6_PAYLOAD,
        "clock": "lp6",
        "forms": {
            "discrete": {
                "unit": "package",
                "widths": {96: "vendor_shipping", 48: "speculative"},
                "default_width": 96,
                # 14400 = JEDEC 最高定义档，但仅论文硅 / 产品页上限 → 已发布（未量产）
                "rates": {10667: "jedec", 12800: "jedec_likely", 14400: "vendor_announced"},
                "default_rate": 10667,
                "counts": (1, 2, 3, 4, 6, 8),
                "default_count": 4,
                "caps": {
                    96: {8: "speculative", 12: "vendor_announced", 16: "vendor_shipping",
                         24: "speculative", 32: "speculative"},
                    48: {4: "speculative", 8: "speculative"},
                },
                "default_cap": {96: 16, 48: 8},
            },
            "LPCAMM2": {  # LPDDR6 CAMM2: JEDEC in development → speculative
                "unit": "module",
                "widths": {192: "speculative"},
                "default_width": 192,
                "rates": {10667: "speculative", 12800: "speculative", 14400: "speculative"},
                "default_rate": 10667,
                "counts": (1, 2),
                "default_count": 1,
                "caps": {192: {32: "speculative", 64: "speculative", 128: "speculative"}},
                "default_cap": {192: 64},
            },
        },
    },
}

HBM_STACK_COUNTS: tuple[int, ...] = (1, 2, 4, 5, 6, 8, 12, 16)

HBM_CATALOG: dict[str, dict[str, Any]] = {
    "HBM3": {
        "jedec_doc": "JESD238 / JESD238B.01",
        "stack_width": 1024,
        "rates": {6400: "jedec"},
        "default_rate": 6400,
        "heights": (8, 12),
        "default_height": 12,
        "densities": (16, 24),
        "default_density": 16,
        # (height, die_Gb) → tag; other in-range combos → JEDEC-allowed
        "cap_tags": {(8, 16): "vendor_shipping", (12, 16): "vendor_shipping"},
        "cap_default_tag": "jedec",
    },
    "HBM3E": {
        "jedec_doc": "无独立 JEDEC 文档（HBM3 厂商扩展）",
        "stack_width": 1024,
        "rates": {8000: "vendor_shipping", 9200: "vendor_shipping", 9600: "vendor_shipping",
                  9800: "vendor_announced"},
        "default_rate": 9200,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (24,),
        "default_density": 24,
        "cap_tags": {(8, 24): "vendor_shipping", (12, 24): "vendor_shipping",
                     (16, 24): "vendor_announced"},
        "cap_default_tag": "speculative",
    },
    "HBM4": {
        "jedec_doc": "JESD270-4 (2025-04)",
        "stack_width": 2048,
        "rates": {8000: "jedec", 10000: "vendor_shipping", 11000: "vendor_shipping",
                  11700: "vendor_shipping", 13000: "vendor_announced"},
        "default_rate": 8000,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (24, 32),
        "default_density": 24,
        "cap_tags": {(8, 24): "jedec", (8, 32): "jedec", (12, 24): "vendor_shipping",
                     (12, 32): "vendor_sampling", (16, 24): "vendor_sampling",
                     (16, 32): "jedec"},
        "cap_default_tag": "jedec",
    },
    "HBM4E": {
        "jedec_doc": "无 JEDEC 标准（厂商定义）",
        "stack_width": 2048,
        "rates": {12800: "speculative", 14000: "vendor_sampling", 16000: "vendor_sampling"},
        "default_rate": 14000,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (32,),
        "default_density": 32,
        "cap_tags": {(8, 32): "vendor_announced", (12, 32): "vendor_sampling",
                     (16, 32): "vendor_announced"},
        "cap_default_tag": "speculative",
    },
}
HBM_DEFAULT_STACKS = 8

# Bus width above which an on-board (discrete) LPDDR config is flagged
# "beyond typical SoC beachfront" (research: >8 × x64 or >6 × x96).
LPDDR_BEACHFRONT_WARN_BITS = 576


def mem_kind_of(mem_type: str) -> str:
    return "HBM" if mem_type.upper().startswith("HBM") else "LPDDR"


def normalize_type(mem_type: str) -> str:
    t = str(mem_type).strip().upper().replace("-", "").replace("_", "")
    if t not in MEM_TYPES:
        raise ValueError(f"unknown mem_type {mem_type!r}; choose from {MEM_TYPES}")
    return t


def normalize_form(mem_type: str, form: str | None) -> str:
    if mem_kind_of(mem_type) == "HBM":
        return "stack"
    if not form:
        return "discrete"
    f = str(form).strip()
    low = f.lower().replace("-", "").replace("_", "")
    alias = {
        "discrete": "discrete", "package": "discrete", "onboard": "discrete", "板载封装": "discrete",
        "socamm2": "SOCAMM2", "socamm": "SOCAMM2",
        "lpcamm2": "LPCAMM2", "camm2": "LPCAMM2", "lpcamm": "LPCAMM2",
    }
    if low not in alias:
        raise ValueError(f"unknown LPDDR form {form!r} (discrete|SOCAMM2|LPCAMM2)")
    out = alias[low]
    forms = LPDDR_CATALOG[mem_type]["forms"]
    if out not in forms:
        raise ValueError(f"{mem_type} has no form {out!r}; choose from {tuple(forms)}")
    return out


def clocks_MHz(mem_type: str, rate_MTps: float) -> dict[str, float]:
    """LPDDR clock relations (JEDEC common knowledge; LPDDR6 CK approx)."""
    if mem_kind_of(mem_type) == "HBM":
        return {"gbps_per_pin": rate_MTps / 1000.0}
    if mem_type == "LPDDR6":
        return {"wck_MHz": rate_MTps / 2.0, "ck_MHz": rate_MTps / 4.0}
    return {"wck_MHz": rate_MTps / 2.0, "ck_MHz": rate_MTps / 8.0}


# ---------------------------------------------------------------------------
# Structured spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemSpec:
    """One structured external-memory configuration (per card)."""

    mem_type: str  # LPDDR5 | LPDDR5X | LPDDR6 | HBM3 | HBM3E | HBM4 | HBM4E
    form: str  # discrete | SOCAMM2 | LPCAMM2 | stack
    unit_width_bits: int
    rate_MTps: int
    n_units: int
    cap_per_unit_GB: float  # LPDDR: per package/module; HBM: derived per stack
    hbm_height: int = 0
    hbm_die_Gb: int = 0
    efficiency: float = 0.70
    payload_factor: float = 1.0
    legacy_id: str = ""
    legacy_note: str = ""

    # --- derived -----------------------------------------------------------
    @property
    def kind(self) -> str:
        return mem_kind_of(self.mem_type)

    @property
    def unit_kind(self) -> str:
        if self.kind == "HBM":
            return "stack"
        return "package" if self.form == "discrete" else "module"

    @property
    def bus_bits(self) -> int:
        return int(self.n_units) * int(self.unit_width_bits)

    @property
    def raw_GBps(self) -> float:
        return self.bus_bits * float(self.rate_MTps) / 8.0 / 1000.0

    @property
    def payload_GBps(self) -> float:
        return self.raw_GBps * self.payload_factor

    @property
    def effective_GBps(self) -> float:
        return self.payload_GBps * self.efficiency

    @property
    def capacity_GB(self) -> float:
        """Nominal vendor GB (= GiB, JEDEC binary)."""
        return float(self.n_units) * float(self.cap_per_unit_GB)

    @property
    def capacity_bytes(self) -> int:
        return int(round(self.capacity_GB * GIB))

    @property
    def id(self) -> str:
        t = self.mem_type.lower()
        cap = f"{self.cap_per_unit_GB:g}g"
        if self.kind == "HBM":
            return f"{t}_{self.n_units}s_{self.hbm_height}h{self.hbm_die_Gb}g_{self.rate_MTps}"
        if self.form == "discrete":
            return f"{t}_{self.n_units}x{self.unit_width_bits}_{self.rate_MTps}_{cap}"
        return f"{t}_{self.form.lower()}_{self.n_units}x{self.unit_width_bits}_{self.rate_MTps}_{cap}"

    # --- provenance --------------------------------------------------------
    def component_tags(self) -> dict[str, str]:
        t = self.mem_type
        if self.kind == "HBM":
            cat = HBM_CATALOG[t]
            rate_tag = cat["rates"].get(self.rate_MTps, "speculative")
            in_range = (
                self.hbm_height in cat["heights"] and self.hbm_die_Gb in cat["densities"]
            )
            cap_tag = (
                cat["cap_tags"].get((self.hbm_height, self.hbm_die_Gb), cat["cap_default_tag"])
                if in_range
                else "speculative"
            )
            width_tag = "jedec" if self.unit_width_bits == cat["stack_width"] else "speculative"
            if self.n_units in HBM_STACK_COUNTS and self.n_units <= 12:
                count_tag = "jedec"  # SoC choice, not a memory-spec claim
            else:
                count_tag = "speculative"
            return {"rate": rate_tag, "width": width_tag, "capacity": cap_tag, "count": count_tag}
        form = LPDDR_CATALOG[t]["forms"][self.form]
        rate_tag = form["rates"].get(self.rate_MTps, "speculative")
        width_tag = form["widths"].get(self.unit_width_bits, "speculative")
        caps = form["caps"].get(self.unit_width_bits, {})
        cap_key = int(self.cap_per_unit_GB) if float(self.cap_per_unit_GB).is_integer() else self.cap_per_unit_GB
        cap_tag = caps.get(cap_key, "speculative")
        count_tag = "jedec" if self.n_units in form["counts"] else "speculative"
        return {"rate": rate_tag, "width": width_tag, "capacity": cap_tag, "count": count_tag}

    @property
    def tag(self) -> str:
        return weakest(*self.component_tags().values())

    @property
    def tag_zh(self) -> str:
        return TAG_ZH[self.tag]

    def warnings(self) -> list[str]:
        w: list[str] = []
        if self.kind == "LPDDR" and self.form == "discrete" and self.bus_bits > LPDDR_BEACHFRONT_WARN_BITS:
            w.append(
                f"板载 LPDDR 总线 {self.bus_bits}-bit 超出常见 SoC 边长（≈≤512–576 bit；"
                f"Grace≈480-bit，Vera 用 SOCAMM2 模组到 1024-bit）"
            )
        if self.kind == "HBM" and self.n_units > 12:
            w.append(f"{self.n_units} 堆 HBM 超出已知产品（MI455X = 12 堆）— 推测")
        for comp, tg in self.component_tags().items():
            if tg == "speculative":
                w.append(f"{_COMP_ZH[comp]}为推测值")
        if self.legacy_note:
            w.append(self.legacy_note)
        return w

    # --- presentation ------------------------------------------------------
    def short_label(self) -> str:
        """Compact summary used in the scenario bar chip."""
        if self.kind == "HBM":
            geo = f"{self.n_units}×{self.hbm_height}H×{self.hbm_die_Gb}Gb"
        elif self.form == "discrete":
            geo = f"{self.n_units}×{self.unit_width_bits}b"
        else:
            geo = f"{self.form} {self.n_units}×{self.unit_width_bits}b"
        return (
            f"{self.mem_type} {geo} @{self.rate_MTps} · "
            f"{_fmt_bw(self.payload_GBps)} 可用 · {self.capacity_GB:g} GB · {self.tag_zh}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            kind=self.kind,
            unit_kind=self.unit_kind,
            id=self.id,
            bus_bits=self.bus_bits,
            raw_GBps=self.raw_GBps,
            payload_GBps=self.payload_GBps,
            effective_GBps=self.effective_GBps,
            capacity_GB=self.capacity_GB,
            capacity_bytes=self.capacity_bytes,
            tag=self.tag,
            tag_zh=self.tag_zh,
            component_tags=self.component_tags(),
            clocks=clocks_MHz(self.mem_type, self.rate_MTps),
            warnings=self.warnings(),
            summary=self.short_label(),
        )
        return d

    def summary(self) -> str:
        return (
            f"{self.id}: {self.mem_type} {self.form} {self.n_units}×{self.unit_width_bits}b "
            f"(bus {self.bus_bits}b) @{self.rate_MTps} MT/s → raw {self.raw_GBps:.1f} GB/s, "
            f"payload×{self.payload_factor:.4f} = {self.payload_GBps:.1f}, "
            f"×eff {self.efficiency:g} = {self.effective_GBps:.1f} GB/s; "
            f"cap {self.capacity_GB:g} GB; tag={TAG_EN[self.tag]}"
        )


_COMP_ZH = {"rate": "速率档", "width": "位宽", "capacity": "容量", "count": "数量"}


def _fmt_bw(gbps: float) -> str:
    if gbps >= 1000:
        return f"{gbps / 1000:.3g} TB/s" if gbps < 10000 else f"{gbps / 1000:.1f} TB/s"
    return f"{gbps:.0f} GB/s" if gbps >= 100 else f"{gbps:.1f} GB/s"


def _snap_rate(rate: float, choices: tuple[int, ...] | list[int]) -> int:
    """Accept GT/s (8.533) or MT/s (8533); snap within 0.5% of a catalog grade."""
    r = float(rate)
    if r < 100:  # GT/s
        r *= 1000.0
    for c in choices:
        if abs(r - c) / c < 0.005:
            return int(c)
    return int(round(r))


def make_spec(
    mem_type: str,
    *,
    form: str | None = None,
    width_bits: int | None = None,
    rate: float | None = None,
    count: int | None = None,
    cap_GB: float | None = None,
    height: int | None = None,
    die_Gb: int | None = None,
    efficiency: float | None = None,
    payload_factor: float | None = None,
    legacy_id: str = "",
    legacy_note: str = "",
) -> MemSpec:
    """Build a MemSpec; missing fields ← per-type/form catalog defaults."""
    t = normalize_type(mem_type)
    eff = 0.70 if efficiency is None else float(efficiency)
    if not (0.0 < eff <= 1.0):
        raise ValueError(f"efficiency must be in (0, 1], got {eff}")
    if mem_kind_of(t) == "HBM":
        cat = HBM_CATALOG[t]
        rr = _snap_rate(rate, tuple(cat["rates"])) if rate is not None else cat["default_rate"]
        h = int(height) if height is not None else cat["default_height"]
        d = int(die_Gb) if die_Gb is not None else cat["default_density"]
        n = int(count) if count is not None else HBM_DEFAULT_STACKS
        w = int(width_bits) if width_bits is not None else cat["stack_width"]
        if n < 1 or h < 1 or d < 1:
            raise ValueError("HBM stacks/height/density must be >= 1")
        return MemSpec(
            mem_type=t, form="stack", unit_width_bits=w, rate_MTps=rr, n_units=n,
            cap_per_unit_GB=h * d / 8.0, hbm_height=h, hbm_die_Gb=d, efficiency=eff,
            payload_factor=1.0 if payload_factor is None else float(payload_factor),
            legacy_id=legacy_id, legacy_note=legacy_note,
        )
    fm = normalize_form(t, form)
    cat_t = LPDDR_CATALOG[t]
    f = cat_t["forms"][fm]
    w = int(width_bits) if width_bits is not None else f["default_width"]
    rr = _snap_rate(rate, tuple(f["rates"])) if rate is not None else f["default_rate"]
    n = int(count) if count is not None else f["default_count"]
    if cap_GB is not None:
        c = float(cap_GB)
    else:
        c = float(f["default_cap"].get(w, next(iter(f["default_cap"].values()))))
    if n < 1 or w < 1 or c <= 0:
        raise ValueError("LPDDR count/width/capacity must be positive")
    pf = cat_t["payload_factor"] if payload_factor is None else float(payload_factor)
    if not (0.0 < pf <= 1.0):
        raise ValueError("payload_factor must be in (0, 1]")
    return MemSpec(
        mem_type=t, form=fm, unit_width_bits=w, rate_MTps=rr, n_units=n,
        cap_per_unit_GB=c, efficiency=eff, payload_factor=pf,
        legacy_id=legacy_id, legacy_note=legacy_note,
    )


def default_spec(mem_type: str, form: str | None = None) -> MemSpec:
    return make_spec(mem_type, form=form)


# ---------------------------------------------------------------------------
# Ids: canonical parse + legacy (≤0.28) package ids
# ---------------------------------------------------------------------------

_RE_HBM = re.compile(r"^(hbm3|hbm3e|hbm4|hbm4e)_(\d+)s_(\d+)h(\d+)g_(\d+)$")
_RE_LP = re.compile(
    r"^(lpddr5|lpddr5x|lpddr6)_(?:(socamm2|lpcamm2)_)?(\d+)x(\d+)_(\d+)_([\d.]+)g$"
)
_RE_OLD_LP5X = re.compile(r"^lpddr_(\d+)x64_(\d+)$")
_RE_OLD_LP6 = re.compile(r"^lpddr6_(\d+)x24_(\d+)$")
_RE_OLD_HBM = re.compile(r"^hbm_(hbm3|hbm3e|hbm4|hbm4e)_(\d+)s$")


def _nearest_count(target: float, counts: tuple[int, ...]) -> int:
    """Nearest allowed count; ties → lower (conservative)."""
    best = min(counts, key=lambda c: (abs(c - target), c))
    return int(best)


def _lp6_legacy_layout(bits: int) -> tuple[int, int, str]:
    """v0.30: legacy x24-die LPDDR6 bus → (package width, count, note).

    Preserve the old bus width (hence bandwidth) as closely as possible:
      * multiple of 96 b → n × x96 (shipping package);
      * else multiple of 48 b → n × x48 (speculative). Bandwidth- and capacity-
        equivalent to ⌊bits/96⌋×x96 + 1×x48 (x48 = 8 GB = half an x96 16 GB),
        which a single-width MemSpec cannot express;
      * odd die counts (bits not a multiple of 48) → nearest of the x96/x48
        layouts, ties → narrower (conservative), x96 preferred at equal width.
    """
    if bits % 96 == 0:
        return 96, bits // 96, "位宽完全保留（x96 量产封装）"
    if bits % 48 == 0:
        k96 = bits // 96
        return 48, bits // 48, (
            f"位宽完全保留；等效 {k96}×x96 + 1×x48，x48 为推测封装" if k96
            else "位宽完全保留；x48 为推测封装"
        )
    cands: list[tuple[int, int, int]] = []  # (bits, width, count)
    for wdt in (96, 48):
        lo = max(1, bits // wdt)
        for c in (lo, lo + 1):
            cands.append((c * wdt, wdt, c))
    b, wdt, c = min(cands, key=lambda t: (abs(t[0] - bits), t[0], t[1] != 96))
    return wdt, c, f"奇数 die 数无法整除，取最近位宽 {b}-bit（相差 {b - bits:+d}-bit）"

def parse_mem_id(mem_id: str, *, efficiency: float | None = None) -> MemSpec:
    """Canonical v0.29 id or legacy ≤0.28 package id → MemSpec.

    Legacy mapping (with ``legacy_note``):
      lpddr_{n}x64_6400        → LPDDR5 n×x64 @6400 (6400 is LPDDR5, not 5X), 12 GB/pkg
      lpddr_{n}x64_{8533|9600} → LPDDR5X n×x64, 16 GB/pkg
      lpddr6_{n}x24_{rate}     → LPDDR6 bus n×24 b kept: x96 packages if /96, else x48
                                 (speculative; ≡ x96s + one x48), odd n → nearest width
      hbm_{gen}_{n}s           → gen defaults (rate / 12H / die) × n stacks
    """
    key = str(mem_id).strip().lower().replace("-", "_")
    m = _RE_HBM.match(key)
    if m:
        t, n, h, d, r = m.groups()
        return make_spec(t.upper(), count=int(n), height=int(h), die_Gb=int(d), rate=int(r),
                         efficiency=efficiency)
    m = _RE_LP.match(key)
    if m:
        t, form, n, w, r, c = m.groups()
        return make_spec(t.upper(), form=form or "discrete", count=int(n), width_bits=int(w),
                         rate=int(r), cap_GB=float(c), efficiency=efficiency)
    m = _RE_OLD_LP5X.match(key)
    if m:
        n, r = int(m.group(1)), int(m.group(2))
        if r <= 6400:
            return make_spec(
                "LPDDR5", count=n, width_bits=64, rate=r, efficiency=efficiency, legacy_id=key,
                legacy_note=f"旧 id {key} → LPDDR5 {n}×x64 @{r}（6400 属 LPDDR5 上限，非 LPDDR5X；容量改为 12 GB/颗）",
            )
        return make_spec(
            "LPDDR5X", count=n, width_bits=64, rate=r, efficiency=efficiency, legacy_id=key,
            legacy_note=f"旧 id {key} → LPDDR5X {n}×x64 @{r}（容量改为 16 GB/颗，不再随位宽线性）",
        )
    m = _RE_OLD_LP6.match(key)
    if m:
        n_die, r = int(m.group(1)), int(m.group(2))
        bits = n_die * 24
        w, n_pkg, how = _lp6_legacy_layout(bits)
        spec = make_spec("LPDDR6", count=n_pkg, width_bits=w, rate=r, efficiency=efficiency,
                         legacy_id=key)
        return replace(spec, legacy_note=(
            f"旧 id {key}（{n_die}×24-bit die = {bits}-bit）→ LPDDR6 {n_pkg}×x{w} "
            f"（{n_pkg * w}-bit；{how}）；另计 8/9 payload"
        ))
    m = _RE_OLD_HBM.match(key)
    if m:
        gen, n = m.group(1).upper(), int(m.group(2))
        spec = make_spec(gen, count=n, efficiency=efficiency, legacy_id=key)
        old_rate = {"HBM3": 6400, "HBM3E": 9200, "HBM4": 8000, "HBM4E": 10000}[gen]
        note = (
            f"旧 id {key} → {gen} {n}×{spec.hbm_height}H×{spec.hbm_die_Gb}Gb @{spec.rate_MTps}"
            f"（容量按 层数×die 密度 计算：{spec.capacity_GB:g} GB）"
        )
        if old_rate != spec.rate_MTps:
            note += f"；旧速率 {old_rate} 为过时设计目标，改用 {spec.rate_MTps}"
        return replace(spec, legacy_note=note)
    raise KeyError(f"unknown memory id {mem_id!r}")


# ---------------------------------------------------------------------------
# Catalog export for UI / API
# ---------------------------------------------------------------------------


def catalog_dict() -> dict[str, Any]:
    """JSON-ready selector catalog (types → forms → widths/rates/counts/caps + tags)."""
    types: list[dict[str, Any]] = []
    for t in MEM_TYPES:
        if mem_kind_of(t) == "HBM":
            c = HBM_CATALOG[t]
            types.append({
                "id": t, "kind": "HBM", "jedec_doc": c["jedec_doc"], "payload_factor": 1.0,
                "forms": [{
                    "id": "stack", "label": FORM_ZH["stack"], "unit": "stack",
                    "widths": [{"bits": c["stack_width"], "tag": "jedec"}],
                    "default_width": c["stack_width"],
                    "rates": [{"MTps": r, "tag": tg, "clocks": clocks_MHz(t, r)} for r, tg in c["rates"].items()],
                    "default_rate": c["default_rate"],
                    "counts": [{"n": n, "tag": "jedec" if n <= 12 else "speculative"} for n in HBM_STACK_COUNTS],
                    "default_count": HBM_DEFAULT_STACKS,
                    "heights": list(c["heights"]), "default_height": c["default_height"],
                    "densities": list(c["densities"]), "default_density": c["default_density"],
                    "cap_tags": [
                        {"height": h, "die_Gb": d, "GB": h * d / 8,
                         "tag": c["cap_tags"].get((h, d), c["cap_default_tag"])}
                        for h in c["heights"] for d in c["densities"]
                    ],
                }],
                "default_form": "stack",
            })
            continue
        c = LPDDR_CATALOG[t]
        forms = []
        for fid, f in c["forms"].items():
            forms.append({
                "id": fid, "label": FORM_ZH[fid], "unit": f["unit"],
                "widths": [{"bits": w, "tag": tg} for w, tg in f["widths"].items()],
                "default_width": f["default_width"],
                "rates": [{"MTps": r, "tag": tg, "clocks": clocks_MHz(t, r)} for r, tg in f["rates"].items()],
                "default_rate": f["default_rate"],
                "counts": [{"n": n, "tag": "jedec"} for n in f["counts"]],
                "default_count": f["default_count"],
                "caps": {str(w): [{"GB": g, "tag": tg} for g, tg in caps.items()] for w, caps in f["caps"].items()},
                "default_cap": {str(w): g for w, g in f["default_cap"].items()},
            })
        types.append({
            "id": t, "kind": "LPDDR", "jedec_doc": c["jedec_doc"],
            "payload_factor": c["payload_factor"], "forms": forms, "default_form": "discrete",
        })
    return {
        "types": types,
        "tag_order": list(TAG_ORDER),
        "tag_zh": TAG_ZH,
        "beachfront_warn_bits": LPDDR_BEACHFRONT_WARN_BITS,
        "formula": (
            "bus = 数量×位宽；raw = bus×MT/s/8000 GB/s；可用 = raw×payload（LPDDR6 = 8/9）；"
            "有效 = 可用×efficiency（假设）；容量 = 数量×单颗（HBM：堆数×层数×die Gb/8），GB = 2^30 B"
        ),
        "source": "docs/research/memory_specs_2026-10.md §6 (2026-10-08)",
    }


# Rates per type (for sweeps tied to generation — fixes "HBM3 @10.0" combos)
def rates_for(mem_type: str, form: str | None = None) -> list[int]:
    t = normalize_type(mem_type)
    if mem_kind_of(t) == "HBM":
        return list(HBM_CATALOG[t]["rates"])
    return list(LPDDR_CATALOG[t]["forms"][normalize_form(t, form)]["rates"])


def counts_for(mem_type: str, form: str | None = None) -> list[int]:
    t = normalize_type(mem_type)
    if mem_kind_of(t) == "HBM":
        return list(HBM_STACK_COUNTS)
    return list(LPDDR_CATALOG[t]["forms"][normalize_form(t, form)]["counts"])


def sweep_specs(base: MemSpec, axis: str) -> list[MemSpec]:
    """Structured package sweep: axis ∈ {type, count, rate}."""
    a = axis.strip().lower()
    if a in ("rate", "speed", "mem_rate"):
        return [replace(base, rate_MTps=r, legacy_id="", legacy_note="") for r in rates_for(base.mem_type, base.form)]
    if a in ("count", "n", "mem_count", "stacks", "packages"):
        cs = counts_for(base.mem_type, base.form)
        return [replace(base, n_units=n, legacy_id="", legacy_note="") for n in cs]
    if a in ("type", "mem_type", "gen"):
        return [make_spec(t, efficiency=base.efficiency) for t in MEM_TYPES]
    raise ValueError(f"package_axis must be type|count|rate, got {axis!r}")


_ = math  # keep import for future geometry helpers
_ = field
