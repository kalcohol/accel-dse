"""Structured external-memory catalog — LPDDR5/5X/6 + HBM3/3E/4/4E.

Source of truth: ``docs/research/memory_specs_2026-10.{md,json}`` (verified 2026-10-08)
plus the 2026-10-08 audit (``MEMORY_AUDIT``; corrections folded into 0.40.1).
This module keeps a compact, data-driven copy of the selector table so the engine
has no file-system dependency at import time.

Modelling unit (granule) — *not* the JEDEC die channel:
  - LPDDR5 / LPDDR5X discrete: **package** (x32 = 2×16, **x64 = 4×16 default**).
    LPDDR5X x96 (6×16) exists only as an Apple-custom part and is intentionally
    not offered.
  - LPDDR6 discrete: **package x96 = 4 × x24 channels = 8 × x12 sub-channels**
    (JESD209-6 defines the x24 *die*; shipping PoP packages are x96). x48 has no
    public part number (Samsung 357/518-ball widths unpublished) → 推测.
  - SOCAMM2 / LPCAMM2: **module** (128-bit; LPDDR6 CAMM2 192-bit speculative)
  - HBM: **stack** (1024-bit HBM3/3E; 2048-bit HBM4/4E)

Derived (no hidden constants)::

    bus_bits      = n_units × unit_width_bits
    raw_GBps      = bus_bits × MT/s / 8 / 1000
    payload_GBps  = raw_GBps × payload_factor      # LPDDR6 = 256/288 = 8/9 (fixed BL24 format)
    effective     = payload_GBps × efficiency      # efficiency: assumed knob
    capacity      = n_units × GB/unit              # HBM: stacks × height × die_Gb / 8
                    (vendor "GB" = 2^30 B, JEDEC binary)
                    × (1 − 1/16) if LPDDR6 meta mode (optional, 「假设」)

Provenance is two independent axes per component (speed grade, unit width,
capacity / stack-height×density):

    spec status    : JEDEC > 疑似 JEDEC > 超规格定制 (beyond-spec custom) > 无规范
    product status : 量产 (shipping) > 送样 (sampling) > 已发布 (announced) > 无产品

A combo takes the weakest value on each axis independently. Package / stack
*counts* are SoC design choices: they carry no spec status; a count outside the
known-product list only sets product status to 无产品.

The legacy single tag (``tag``; JEDEC > 疑似 JEDEC > 厂商量产 > 送样 > 已发布 > 推测)
is derived from the two axes for backward compatibility.

Values outside the catalog lists are accepted (DSE what-ifs) but tagged 无规范 · 无产品.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

GIB = 2**30

# --- legacy single-axis tag (derived) ---------------------------------------
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

# --- dual axes (0.40.1, memory audit M-2) -----------------------------------
SPEC_ORDER: tuple[str, ...] = ("jedec", "jedec_likely", "custom", "unspecified")
SPEC_ZH: dict[str, str] = {
    "jedec": "JEDEC", "jedec_likely": "疑似 JEDEC", "custom": "超规格定制", "unspecified": "无规范",
}
SPEC_EN: dict[str, str] = {
    "jedec": "JEDEC", "jedec_likely": "JEDEC?", "custom": "beyond-spec custom", "unspecified": "unspecified",
}
SPEC_HINT: dict[str, str] = {
    "jedec": "JEDEC 文档或讲稿明文定义",
    "jedec_likely": "间接证据指向 JEDEC 定义（规范原文未读或付费）",
    "custom": "厂商超出 JEDEC 定义（如 HBM3E/HBM4 高速档）",
    "unspecified": "没有找到规范依据（含规范在研）",
}
PRODUCT_ORDER: tuple[str, ...] = ("shipping", "sampling", "announced", "none")
PRODUCT_ZH: dict[str, str] = {"shipping": "量产", "sampling": "送样", "announced": "已发布", "none": "无产品"}
PRODUCT_EN: dict[str, str] = {
    "shipping": "shipping", "sampling": "sampling", "announced": "announced", "none": "no product",
}

LPDDR6_PAYLOAD = 256.0 / 288.0  # BL24: 288 bit = 256 data + 16 meta + 16 DBI/Link-ECC (fixed format)
# LPDDR6 System / carve-out meta mode (optional, default off). Capacity reserve
# not published → 「假设」: metadata persisted at the wire ratio 16 bit / 256 bit.
LPDDR6_META_CARVEOUT = 16.0 / 256.0

MEM_TYPES: tuple[str, ...] = (
    "LPDDR5", "LPDDR5X", "LPDDR6", "HBM3", "HBM3E", "HBM4", "HBM4E",
)
# Brand / marketing aliases → (type, default rate). LPDDR5T = SK hynix 2023 name
# for 9.6 Gbps LPDDR5X (later merged into LPDDR5X-9600).
TYPE_ALIASES: dict[str, tuple[str, int]] = {"LPDDR5T": ("LPDDR5X", 9600)}
FORM_ZH: dict[str, str] = {
    "discrete": "板载封装",
    "SOCAMM2": "SOCAMM2 模组",
    "LPCAMM2": "LPCAMM2 模组",
    "stack": "HBM 堆栈",
}
UNIT_ZH: dict[str, str] = {"package": "颗", "module": "条", "stack": "堆"}


def weakest(*tags: Any) -> str:
    """Weakest legacy provenance tag among ``tags`` (unknown → speculative).

    Accepts ``weakest("jedec", "vendor_shipping")`` or one iterable of tags.
    """
    if len(tags) == 1 and not isinstance(tags[0], str):
        tags = tuple(tags[0])
    worst = 0
    for t in tags:
        i = TAG_ORDER.index(t) if t in TAG_ORDER else len(TAG_ORDER) - 1
        worst = max(worst, i)
    return TAG_ORDER[worst]


def _weakest_on(order: tuple[str, ...], vals: Any) -> str:
    worst = 0
    for v in vals:
        worst = max(worst, order.index(v) if v in order else len(order) - 1)
    return order[worst]


def weakest_spec(*vals: str) -> str:
    return _weakest_on(SPEC_ORDER, vals)


def weakest_product(*vals: str) -> str:
    return _weakest_on(PRODUCT_ORDER, vals)


def legacy_tag(spec: str, product: str) -> str:
    """Two axes → legacy single tag (compat): no product ⇒ JEDEC-allowed or 推测."""
    if product == "none":
        return "jedec" if spec == "jedec" else "speculative"
    if product == "shipping":
        return {"jedec": "jedec", "jedec_likely": "jedec_likely"}.get(spec, "vendor_shipping")
    return f"vendor_{product}"


def status_zh(spec: str, product: str) -> str:
    """Concise dual label, e.g. 「JEDEC · 量产」, 「JEDEC 允许 · 无产品」."""
    s = "JEDEC 允许" if (spec == "jedec" and product == "none") else SPEC_ZH.get(spec, spec)
    return f"{s} · {PRODUCT_ZH.get(product, product)}"


def status_en(spec: str, product: str) -> str:
    return f"{SPEC_EN.get(spec, spec)} / {PRODUCT_EN.get(product, product)}"


# ---------------------------------------------------------------------------
# Catalog data (compact copy of docs/research/memory_specs_2026-10.json §6,
# corrected per the 2026-10-08 audit). Every entry = (spec status, product status).
# ---------------------------------------------------------------------------
_JS = ("jedec", "shipping")
_JM = ("jedec", "sampling")
_JA = ("jedec", "announced")
_JN = ("jedec", "none")            # JEDEC-allowed, no product
_LS = ("jedec_likely", "shipping")
_LM = ("jedec_likely", "sampling")
_LA = ("jedec_likely", "announced")
_LN = ("jedec_likely", "none")
_CS = ("custom", "shipping")
_CA = ("custom", "announced")
_UM = ("unspecified", "sampling")
_UA = ("unspecified", "announced")
_UN = ("unspecified", "none")      # = legacy 推测

# Each LPDDR form: unit, widths{bits: st}, rates{MTps: st}, counts, caps{width:{GB: st}}
# st = (spec, product). Package widths: JESD209 defines dies + package ballouts
# (paid chapters, unread) → 疑似 JEDEC. Capacities within JEDEC die densities → JEDEC.

LPDDR_CATALOG: dict[str, dict[str, Any]] = {
    "LPDDR5": {
        "jedec_doc": "JESD209-5 / 5C",
        "payload_factor": 1.0,
        "clock": "lp5",
        "forms": {
            "discrete": {
                "unit": "package",
                "widths": {32: _LS, 64: _LS},
                "default_width": 64,
                "rates": {5500: _JS, 6400: _JS},
                "default_rate": 6400,
                "counts": (1, 2, 4, 6, 8),
                "default_count": 4,
                # x32 12 GB not verified (audit L5-3) → JEDEC 允许 · 无产品
                "caps": {32: {3: _JS, 6: _JS, 8: _JS, 12: _JN}, 64: {3: _JS, 6: _JS, 8: _JS, 12: _JS}},
                "default_cap": {32: 6, 64: 12},
            },
        },
    },
    "LPDDR5X": {
        "jedec_doc": "JESD209-5C (≤8533 明文；9600/10667 疑似)；LPDDR5T = LPDDR5X-9600",
        "payload_factor": 1.0,
        "clock": "lp5",
        "forms": {
            "discrete": {
                "unit": "package",
                # x96 (6×16): Apple-custom only (Micron "6-channel"), intentionally not offered.
                "widths": {32: _LS, 64: _LS},
                "default_width": 64,
                "rates": {7500: _LS, 8533: _JS, 9600: _LS, 10667: _LS},
                "default_rate": 8533,
                "counts": (1, 2, 3, 4, 6, 8, 10, 12, 16),
                "default_count": 8,
                "caps": {
                    # 64 GB x32 = 256 GB SOCAMM2 component, sampling since 2026-03 (audit L5X-9)
                    32: {8: _JS, 16: _JS, 32: _JS, 64: _JM},
                    64: {4: _JS, 6: _JS, 8: _JS, 12: _JS, 16: _JS, 24: _JS, 32: _JA},
                },
                "default_cap": {32: 16, 64: 16},
            },
            "SOCAMM2": {
                "unit": "module",
                "widths": {128: _JS},  # JESD328 (2026-06)
                "default_width": 128,
                "rates": {8533: _JS, 9600: _LS},
                "default_rate": 9600,
                "counts": (1, 2, 4, 6, 8),
                "default_count": 8,
                # 256 GB: Micron sampling 2026-03; mass production unconfirmed (audit S-4)
                "caps": {128: {48: _JS, 64: _JS, 96: _JS, 128: _JS, 192: _JS, 256: _JM}},
                "default_cap": {128: 192},
            },
            "LPCAMM2": {
                "unit": "module",
                "widths": {128: _JS},  # JESD318 CAMM2
                "default_width": 128,
                # 9600: Micron product page + Samsung 96 GB 9600 module (audit C-3)
                "rates": {7500: _LS, 8533: _JS, 9600: _LS},
                "default_rate": 8533,
                "counts": (1, 2),
                "default_count": 2,
                "caps": {128: {16: _JS, 32: _JS, 64: _JS, 96: _JA}},
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
                # x96: CXMT 1295-ball PoP, Samsung PDBEREAGY0B1, Synopsys "up to 96 bits per package".
                # x48: no public PN; Samsung lists 357/518-FBGA with width unpublished →
                # 推测, upgradable once a datasheet / PN is supplied.
                "widths": {96: _LS, 48: _UN},
                "default_width": 96,
                # 10667 / 11733 / 12800 / 14400 = WCK 5333/5866/6400/7200 MHz (JEDEC Takahashi).
                # 12800: CXMT mass production. 14400: JEDEC highest grade, no product.
                "rates": {10667: _JS, 11733: _LN, 12800: _JS, 14400: _JA},
                "default_rate": 10667,
                "counts": (1, 2, 3, 4, 6, 8),
                "default_count": 4,
                "caps": {
                    # 8 / 12 GB: inside Samsung's 16–128 Gb range, no SKU → unified (audit P-9)
                    # 24 / 32 GB: need 24/32 Gb dies or the next-rev x6 sub-channel mode
                    96: {8: _LN, 12: _LN, 16: _JS, 24: _UN, 32: _UN},
                    48: {4: _UN, 8: _UN},
                },
                "default_cap": {96: 16, 48: 8},
            },
            "LPCAMM2": {  # LPDDR6 CAMM2: JEDEC in development (2026-04 focus = LPDDR6 SOCAMM2)
                "unit": "module",
                "widths": {192: _UN},
                "default_width": 192,
                "rates": {10667: _UN, 12800: _UN, 14400: _UN},
                "default_rate": 10667,
                "counts": (1, 2),
                "default_count": 1,
                "caps": {192: {32: _UN, 64: _UN, 128: _UN}},
                "default_cap": {192: 64},
            },
        },
    },
}

HBM_STACK_COUNTS: tuple[int, ...] = (1, 2, 4, 5, 6, 8, 12, 16)
HBM_MAX_KNOWN_STACKS = 12  # MI455X

HBM_CATALOG: dict[str, dict[str, Any]] = {
    "HBM3": {
        "jedec_doc": "JESD238 / JESD238B.01",
        "stack_width": 1024,
        "width_status": _JS,
        "rates": {6400: _JS},
        "default_rate": 6400,
        "heights": (8, 12),
        "default_height": 12,
        "densities": (16, 24),
        "default_density": 16,
        # (height, die_Gb) → status; other in-range combos → cap_default
        "cap_tags": {(8, 16): _JS, (12, 16): _JS},
        "cap_default": _JN,
    },
    "HBM3E": {
        "jedec_doc": "无独立 JEDEC 文档（HBM3 厂商扩展）",
        "stack_width": 1024,
        "width_status": _JS,
        # speeds beyond JESD238's 6.4 Gb/s → 超规格定制; 8000 = system-level run rate
        "rates": {8000: _CS, 9200: _CS, 9600: _CS, 9800: _CA},
        "default_rate": 9200,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (24,),
        "default_density": 24,
        "cap_tags": {(8, 24): _JS, (12, 24): _JS, (16, 24): _JA},
        "cap_default": _UN,
    },
    "HBM4": {
        "jedec_doc": "JESD270-4 (2025-04) / 270-4A",
        "stack_width": 2048,
        "width_status": _JS,
        "rates": {8000: _JS, 10000: _CS, 11000: _CS, 11700: _CS, 13000: _CA},
        "default_rate": 8000,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (24, 32),
        "default_density": 24,
        # 48 GB samples are 16-high × 24 Gb; 12-high × 32 Gb is JEDEC-allowed only (audit H4-5)
        "cap_tags": {(8, 24): _JN, (8, 32): _JN, (12, 24): _JS,
                     (12, 32): _JN, (16, 24): _JM, (16, 32): _JN},
        "cap_default": _JN,
    },
    "HBM4E": {
        "jedec_doc": "无 JEDEC 标准（厂商定义）",
        "stack_width": 2048,
        "width_status": _UM,
        "rates": {12800: _UN, 14000: _UM, 16000: _UM},
        "default_rate": 14000,
        "heights": (8, 12, 16),
        "default_height": 12,
        "densities": (32,),
        "default_density": 32,
        "cap_tags": {(8, 32): _UA, (12, 32): _UM, (16, 32): _UA},
        "cap_default": _UN,
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
    if t in TYPE_ALIASES:
        return TYPE_ALIASES[t][0]
    if t not in MEM_TYPES:
        raise ValueError(f"unknown mem_type {mem_type!r}; choose from {MEM_TYPES} (alias: LPDDR5T)")
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
    meta_mode: bool = False  # LPDDR6 System / carve-out meta mode (optional, default off)
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
    def nominal_capacity_GB(self) -> float:
        """Nominal vendor GB (= GiB, JEDEC binary)."""
        return float(self.n_units) * float(self.cap_per_unit_GB)

    @property
    def meta_reserve_frac(self) -> float:
        return LPDDR6_META_CARVEOUT if (self.meta_mode and self.mem_type == "LPDDR6") else 0.0

    @property
    def capacity_GB(self) -> float:
        """Usable GB: nominal minus the LPDDR6 meta carve-out when meta mode is on (「假设」)."""
        return self.nominal_capacity_GB * (1.0 - self.meta_reserve_frac)

    @property
    def capacity_bytes(self) -> int:
        return int(round(self.capacity_GB * GIB))

    @property
    def id(self) -> str:
        t = self.mem_type.lower()
        cap = f"{self.cap_per_unit_GB:g}g" + ("_meta" if self.meta_mode else "")
        if self.kind == "HBM":
            return f"{t}_{self.n_units}s_{self.hbm_height}h{self.hbm_die_Gb}g_{self.rate_MTps}"
        if self.form == "discrete":
            return f"{t}_{self.n_units}x{self.unit_width_bits}_{self.rate_MTps}_{cap}"
        return f"{t}_{self.form.lower()}_{self.n_units}x{self.unit_width_bits}_{self.rate_MTps}_{cap}"

    # --- provenance (dual axes) ---------------------------------------------
    def component_status(self) -> dict[str, dict[str, str]]:
        """Per component {spec, product}. ``count`` is an SoC choice: spec = soc_choice."""
        t = self.mem_type
        un = _UN
        if self.kind == "HBM":
            cat = HBM_CATALOG[t]
            rate = cat["rates"].get(self.rate_MTps, un)
            in_range = self.hbm_height in cat["heights"] and self.hbm_die_Gb in cat["densities"]
            cap = cat["cap_tags"].get((self.hbm_height, self.hbm_die_Gb), cat["cap_default"]) if in_range else un
            width = cat["width_status"] if self.unit_width_bits == cat["stack_width"] else un
            known = self.n_units in HBM_STACK_COUNTS and self.n_units <= HBM_MAX_KNOWN_STACKS
        else:
            form = LPDDR_CATALOG[t]["forms"][self.form]
            rate = form["rates"].get(self.rate_MTps, un)
            width = form["widths"].get(self.unit_width_bits, un)
            caps = form["caps"].get(self.unit_width_bits, {})
            ck = int(self.cap_per_unit_GB) if float(self.cap_per_unit_GB).is_integer() else self.cap_per_unit_GB
            cap = caps.get(ck, un)
            known = self.n_units in form["counts"]
        out = {k: {"spec": v[0], "product": v[1]} for k, v in (("rate", rate), ("width", width), ("capacity", cap))}
        out["count"] = {"spec": "soc_choice", "product": "shipping" if known else "none"}
        return out

    @property
    def spec_status(self) -> str:
        return weakest_spec(*(v["spec"] for k, v in self.component_status().items() if k != "count"))

    @property
    def product_status(self) -> str:
        return weakest_product(*(v["product"] for v in self.component_status().values()))

    @property
    def status_zh(self) -> str:
        return status_zh(self.spec_status, self.product_status)

    def component_tags(self) -> dict[str, str]:
        """Legacy single-axis tag per component (derived from the dual axes)."""
        out = {}
        for k, v in self.component_status().items():
            if k == "count":
                out[k] = "jedec" if v["product"] != "none" else "speculative"  # neutral unless unknown
            else:
                out[k] = legacy_tag(v["spec"], v["product"])
        return out

    @property
    def tag(self) -> str:
        return legacy_tag(self.spec_status, self.product_status)

    @property
    def tag_zh(self) -> str:
        """Concise dual label shown in the UI, e.g. 「疑似 JEDEC · 量产」."""
        return self.status_zh

    def warnings(self) -> list[str]:
        w: list[str] = []
        if self.kind == "LPDDR" and self.form == "discrete" and self.bus_bits > LPDDR_BEACHFRONT_WARN_BITS:
            w.append(
                f"板载 LPDDR 总线 {self.bus_bits}-bit 超出常见 SoC 边长（≈≤512–576 bit；"
                f"Grace≈480-bit，Vera 用 SOCAMM2 模组到 1024-bit）"
            )
        if self.kind == "HBM" and self.n_units > 12:
            w.append(f"{self.n_units} 堆 HBM 超出已知产品（MI455X = 12 堆）— 推测")
        for comp, st in self.component_status().items():
            if comp == "count":
                if st["product"] == "none" and self.kind != "HBM":
                    w.append(f"{self.n_units} {UNIT_ZH[self.unit_kind]}超出已知产品配置")
            elif st["product"] == "none" and st["spec"] != "jedec":
                w.append(f"{_COMP_ZH[comp]}为推测值（{status_zh(st['spec'], st['product'])}）")
        if self.meta_reserve_frac:
            w.append(f"LPDDR6 meta 模式：容量预留 {self.meta_reserve_frac:.2%}（假设，规范未公开）；"
                     f"Meta RD/WR 吞吐损失未建模")
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
            + (" · meta" if self.meta_mode else "")
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
            nominal_capacity_GB=self.nominal_capacity_GB,
            capacity_bytes=self.capacity_bytes,
            tag=self.tag,
            tag_zh=self.tag_zh,
            spec_status=self.spec_status,
            product_status=self.product_status,
            spec_zh=SPEC_ZH[self.spec_status],
            product_zh=PRODUCT_ZH[self.product_status],
            component_tags=self.component_tags(),
            component_status=self.component_status(),
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
            f"cap {self.capacity_GB:g} GB; spec={SPEC_EN[self.spec_status]}, "
            f"product={PRODUCT_EN[self.product_status]}"
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
    meta_mode: bool = False,
    legacy_id: str = "",
    legacy_note: str = "",
) -> MemSpec:
    """Build a MemSpec; missing fields ← per-type/form catalog defaults.

    ``LPDDR5T`` is accepted as an alias of LPDDR5X (default rate 9600).
    ``meta_mode`` (LPDDR6 only) reserves the meta carve-out from capacity.
    """
    raw_t = str(mem_type).strip().upper().replace("-", "").replace("_", "")
    t = normalize_type(mem_type)
    if raw_t in TYPE_ALIASES:
        if rate is None:
            rate = TYPE_ALIASES[raw_t][1]
        legacy_note = legacy_note or f"{raw_t} = SK hynix 对 LPDDR5X-9600 的品牌名，按 LPDDR5X 建模"
    if meta_mode and t != "LPDDR6":
        raise ValueError("meta_mode applies to LPDDR6 only")
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
    if t == "LPDDR5X" and fm == "discrete" and w == 96:
        # LPDDR5X x96 = Apple-custom only, not offered → same bus width in x64 packages
        n96 = n
        n = _nearest_count(n96 * 96 / 64, f["counts"])
        w = 64
        if cap_GB is not None and int(float(cap_GB)) not in f["caps"][64]:
            cap_GB = None
        legacy_note = legacy_note or (
            f"LPDDR5X x96 仅为 Apple 定制件，不提供；{n96}×x96 改为 {n}×x64（{n * 64}-bit）"
        )
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
        cap_per_unit_GB=c, efficiency=eff, payload_factor=pf, meta_mode=bool(meta_mode),
        legacy_id=legacy_id, legacy_note=legacy_note,
    )


def default_spec(mem_type: str, form: str | None = None) -> MemSpec:
    return make_spec(mem_type, form=form)


# ---------------------------------------------------------------------------
# Ids: canonical parse + legacy (≤0.28) package ids
# ---------------------------------------------------------------------------

_RE_HBM = re.compile(r"^(hbm3|hbm3e|hbm4|hbm4e)_(\d+)s_(\d+)h(\d+)g_(\d+)$")
_RE_LP = re.compile(
    r"^(lpddr5|lpddr5x|lpddr5t|lpddr6)_(?:(socamm2|lpcamm2)_)?(\d+)x(\d+)_(\d+)_([\d.]+)g(_meta)?$"
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
        t, form, n, w, r, c, meta = m.groups()
        return make_spec(t.upper(), form=form or "discrete", count=int(n), width_bits=int(w),
                         rate=int(r), cap_GB=float(c), efficiency=efficiency, meta_mode=bool(meta))
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
                    "widths": [_opt({"bits": c["stack_width"]}, c["width_status"])],
                    "default_width": c["stack_width"],
                    "rates": [_opt({"MTps": r, "clocks": clocks_MHz(t, r)}, st) for r, st in c["rates"].items()],
                    "default_rate": c["default_rate"],
                    "counts": [_count_opt(n, n <= HBM_MAX_KNOWN_STACKS) for n in HBM_STACK_COUNTS],
                    "default_count": HBM_DEFAULT_STACKS,
                    "heights": list(c["heights"]), "default_height": c["default_height"],
                    "densities": list(c["densities"]), "default_density": c["default_density"],
                    "cap_tags": [
                        _opt({"height": h, "die_Gb": d, "GB": h * d / 8}, c["cap_tags"].get((h, d), c["cap_default"]))
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
                "widths": [_opt({"bits": w}, st) for w, st in f["widths"].items()],
                "default_width": f["default_width"],
                "rates": [_opt({"MTps": r, "clocks": clocks_MHz(t, r)}, st) for r, st in f["rates"].items()],
                "default_rate": f["default_rate"],
                "counts": [_count_opt(n, True) for n in f["counts"]],
                "default_count": f["default_count"],
                "caps": {str(w): [_opt({"GB": g}, st) for g, st in caps.items()] for w, caps in f["caps"].items()},
                "default_cap": {str(w): g for w, g in f["default_cap"].items()},
            })
        entry = {
            "id": t, "kind": "LPDDR", "jedec_doc": c["jedec_doc"],
            "payload_factor": c["payload_factor"], "forms": forms, "default_form": "discrete",
        }
        if t == "LPDDR6":
            entry["meta_mode"] = {
                "default": False, "carveout_frac": LPDDR6_META_CARVEOUT,
                "note": "System / carve-out meta 模式：从阵列划出 meta 区持久保存每 32B 的 16 bit metadata；"
                        "容量预留 1/16（假设，规范与厂商未公开），Meta RD/WR 吞吐损失未建模。"
                        "8/9 payload 是 BL24 格式固定开销，与此开关无关。",
            }
        types.append(entry)
    return {
        "types": types,
        "tag_order": list(TAG_ORDER),
        "tag_zh": TAG_ZH,
        "spec_order": list(SPEC_ORDER), "spec_zh": SPEC_ZH, "spec_en": SPEC_EN, "spec_hint": SPEC_HINT,
        "product_order": list(PRODUCT_ORDER), "product_zh": PRODUCT_ZH, "product_en": PRODUCT_EN,
        "aliases": {k: {"type": v[0], "rate": v[1]} for k, v in TYPE_ALIASES.items()},
        "not_offered": ["LPDDR5X x96（6×16）：仅 Apple 定制件，其他客户无法采购，故不提供"],
        "status_note": "两个标签：规范状态（JEDEC / 疑似 JEDEC / 超规格定制 / 无规范）× 产品状态"
                       "（量产 / 送样 / 已发布 / 无产品）；组合各轴分别取最弱项。颗数 / 堆数是 SoC 设计选择，不打规范标签。",
        "beachfront_warn_bits": LPDDR_BEACHFRONT_WARN_BITS,
        "formula": (
            "bus = 数量×位宽；raw = bus×MT/s/8000 GB/s；可用 = raw×payload（LPDDR6 = 8/9）；"
            "有效 = 可用×efficiency（假设）；容量 = 数量×单颗（HBM：堆数×层数×die Gb/8），GB = 2^30 B"
        ),
        "source": "docs/research/memory_specs_2026-10.md §6 + MEMORY_AUDIT (2026-10-08)",
    }


def _opt(d: dict[str, Any], st: tuple[str, str]) -> dict[str, Any]:
    sp, pr = st
    d.update(spec=sp, product=pr, tag=legacy_tag(sp, pr), status_zh=status_zh(sp, pr))
    return d


def _count_opt(n: int, known: bool) -> dict[str, Any]:
    pr = "shipping" if known else "none"
    return {"n": n, "spec": "soc_choice", "product": pr, "tag": "jedec" if known else "speculative",
            "status_zh": "SoC 设计选择" if known else "超出已知产品"}


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
