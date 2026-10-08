"""Dtype / bit-width presets for storage and DRAM traffic (bytes-only).

Honesty contract (v0.3):
  - Presets change **weight_bits** and **kv_bits** (and optionally act_bits).
  - DRAM / SRAM byte footprints scale with bits; FLOPs and MAC array peak
    rate are **unchanged**. Keeping compute FLOPs/MAC rate fixed is
    *conservative for low-precision*: real low-precision MAC arrays are often
    denser/faster, which this model does **not** claim yet.
  - Labels are storage/traffic bits only — not a claim of native e4m3/int4
    MAC throughput curves.

Presets (assumed) — uniform bits on weight+KV:
  fp16     → 16 bits
  fp8      → 8 bits   (alias: e4m3)
  int8     → 8 bits
  int4     → 4 bits

Independent weight/KV quant presets (v0.4):
  w16k16, w8k16, w4k16, w8k8, w4k8
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .model_shape import ModelShape


@dataclass(frozen=True)
class DTypePreset:
    """Named storage/traffic bit-width template (assumed)."""

    name: str
    bits: int
    note: str = ""

    def apply(
        self,
        shape: ModelShape,
        *,
        weight: bool = True,
        kv: bool = True,
        act: bool = False,
    ) -> ModelShape:
        """Return a copy of `shape` with selected bit fields set to `bits`.

        Default: scale weight + KV bytes; leave act_bits alone (activations
        often stay wider). Pass act=True to scale activations too.
        """
        kw: dict[str, int | str] = {
            "name": f"{shape.name}@{self.name}",
        }
        if weight:
            kw["weight_bits"] = self.bits
        if kv:
            kw["kv_bits"] = self.bits
        if act:
            kw["act_bits"] = self.bits
        return replace(shape, **kw)  # type: ignore[arg-type]

    def summary(self) -> str:
        return (
            f"{self.name}: {self.bits}-bit storage/traffic "
            f"(MAC rate unchanged — bytes-only) {self.note}"
        )


# ---------------------------------------------------------------------------
# Assumed presets
# ---------------------------------------------------------------------------

DTYPE_FP16 = DTypePreset("fp16", 16, note="[assumed] baseline fp16 storage")
DTYPE_FP8 = DTypePreset("fp8", 8, note="[assumed] fp8/e4m3-class 8-bit storage")
DTYPE_E4M3 = DTypePreset("e4m3", 8, note="[assumed] alias of fp8 (e4m3)")
DTYPE_INT8 = DTypePreset("int8", 8, note="[assumed] int8 storage")
DTYPE_INT4 = DTypePreset("int4", 4, note="[assumed] int4 storage")

# Primary sweep order (e4m3 omitted as duplicate of fp8 bits)
DTYPE_SWEEP_DEFAULT: tuple[DTypePreset, ...] = (
    DTYPE_FP16,
    DTYPE_FP8,
    DTYPE_INT8,
    DTYPE_INT4,
)

DTYPE_REGISTRY: dict[str, DTypePreset] = {
    DTYPE_FP16.name: DTYPE_FP16,
    DTYPE_FP8.name: DTYPE_FP8,
    DTYPE_E4M3.name: DTYPE_E4M3,
    DTYPE_INT8.name: DTYPE_INT8,
    DTYPE_INT4.name: DTYPE_INT4,
}


@dataclass(frozen=True)
class QuantPreset:
    """Independent weight_bits / kv_bits template (bytes-only; MAC unchanged).

    Used for weight-only vs full quant comparisons (e.g. w4k16 vs w4k8).
    """

    name: str
    weight_bits: int
    kv_bits: int
    note: str = ""

    def apply(self, shape: ModelShape) -> ModelShape:
        return shape.with_bits(
            weight_bits=self.weight_bits,
            kv_bits=self.kv_bits,
            name_suffix=self.name,
        )

    def summary(self) -> str:
        return (
            f"{self.name}: W={self.weight_bits}-bit KV={self.kv_bits}-bit "
            f"storage/traffic (MAC rate unchanged) {self.note}"
        )


# Weight-only vs full quant presets (assumed)
QUANT_W16K16 = QuantPreset("w16k16", 16, 16, note="[assumed] full fp16")
QUANT_W8K16 = QuantPreset("w8k16", 8, 16, note="[assumed] weight-only 8-bit")
QUANT_W4K16 = QuantPreset("w4k16", 4, 16, note="[assumed] weight-only 4-bit")
QUANT_W8K8 = QuantPreset("w8k8", 8, 8, note="[assumed] full 8-bit")
QUANT_W4K8 = QuantPreset("w4k8", 4, 8, note="[assumed] W4 + KV8")

QUANT_SWEEP_DEFAULT: tuple[QuantPreset, ...] = (
    QUANT_W16K16,
    QUANT_W8K16,
    QUANT_W4K16,
    QUANT_W8K8,
    QUANT_W4K8,
)

QUANT_REGISTRY: dict[str, QuantPreset] = {
    q.name: q for q in QUANT_SWEEP_DEFAULT
}


def get_quant(name: str) -> QuantPreset:
    key = name.strip().lower().replace("-", "_")
    if key not in QUANT_REGISTRY:
        raise KeyError(
            f"unknown quant {name!r}; choose from {list(QUANT_REGISTRY)}"
        )
    return QUANT_REGISTRY[key]

def get_dtype(name: str) -> DTypePreset:
    key = name.strip().lower().replace("-", "_")
    # accept fp8_e4m3 etc.
    if key in ("fp8_e4m3", "e4m3_fp8"):
        key = "e4m3"
    if key not in DTYPE_REGISTRY:
        raise KeyError(
            f"unknown dtype {name!r}; choose from {list(DTYPE_REGISTRY)}"
        )
    return DTYPE_REGISTRY[key]


def with_bits(
    shape: ModelShape,
    *,
    weight_bits: int | None = None,
    kv_bits: int | None = None,
    act_bits: int | None = None,
    name_suffix: str | None = None,
) -> ModelShape:
    """Explicit bit override helper (bytes-only; FLOPs unchanged)."""
    wb = shape.weight_bits if weight_bits is None else weight_bits
    kb = shape.kv_bits if kv_bits is None else kv_bits
    ab = shape.act_bits if act_bits is None else act_bits
    for label, v in (("weight_bits", wb), ("kv_bits", kb), ("act_bits", ab)):
        if v <= 0 or v % 1 != 0:
            raise ValueError(f"{label} must be a positive integer, got {v}")
        if v % 8 != 0 and v not in (4,):  # allow nibble packing
            # still allow any positive int; bytes use integer division
            pass
    suffix = name_suffix or f"w{wb}kv{kb}"
    return replace(
        shape,
        name=f"{shape.name}@{suffix}",
        weight_bits=int(wb),
        kv_bits=int(kb),
        act_bits=int(ab),
    )


def iter_dtype_shapes(
    base: ModelShape,
    presets: Iterable[DTypePreset] | None = None,
) -> list[tuple[DTypePreset, ModelShape]]:
    presets = list(presets) if presets is not None else list(DTYPE_SWEEP_DEFAULT)
    return [(p, p.apply(base)) for p in presets]


# ---------------------------------------------------------------------------
# Optional dtype MAC peak factors (v0.26 — assumed / opt-in; default all 1.0)
# ---------------------------------------------------------------------------
#
# Honesty: these multiply reported peak TOPS and divide GEMM compute time.
# They are **user/assumed**, not silicon. Default map is all 1.0 so behavior
# stays bytes-only conservative (identical to pre-v0.26 handcheck).

DEFAULT_DTYPE_MAC_FACTORS: dict[str, float] = {
    "fp16": 1.0,
    "fp8": 1.0,
    "e4m3": 1.0,
    "int8": 1.0,
    "int4": 1.0,
}

# weight_bits → candidate keys (first hit in user map wins)
_BITS_TO_DTYPE_KEYS: dict[int, tuple[str, ...]] = {
    16: ("fp16", "16"),
    8: ("fp8", "int8", "e4m3", "8"),
    4: ("int4", "4"),
}


def normalize_dtype_mac_factors(
    factors: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """Return a lowercased factor map, or None if empty / all-default."""
    if not factors:
        return None
    out = {
        str(k).strip().lower().replace("-", "_"): float(v)
        for k, v in factors.items()
    }
    return out or None


def resolve_dtype_mac_factor(
    factors: Mapping[str, float] | None,
    *,
    dtype_name: str | None = None,
    weight_bits: int | None = None,
) -> float:
    """Look up assumed MAC peak factor for a weight dtype / bit-width.

    Lookup order: explicit ``dtype_name`` (fp16/fp8/int8/int4/…), then
    ``weight_bits`` as digit string or canonical name. Missing → **1.0**
    (conservative bytes-only). Factors are user/assumed, not silicon.
    """
    m = normalize_dtype_mac_factors(factors)
    if not m:
        return 1.0
    if dtype_name:
        key = str(dtype_name).strip().lower().replace("-", "_")
        if key in ("fp8_e4m3", "e4m3_fp8"):
            key = "e4m3"
        # quant presets like w4k16 → extract weight width hint
        if key.startswith("w") and "k" in key[1:]:
            # e.g. w4k16 → prefer bits path below unless exact key present
            if key in m:
                return m[key]
        elif key in m:
            return m[key]
    if weight_bits is not None:
        wb = int(weight_bits)
        bkey = str(wb)
        if bkey in m:
            return m[bkey]
        for cand in _BITS_TO_DTYPE_KEYS.get(wb, ()):
            if cand in m:
                return m[cand]
    return 1.0


def dtype_mac_factors_active(factors: Mapping[str, float] | None) -> bool:
    """True if any factor differs from 1.0 (opt-in in use)."""
    m = normalize_dtype_mac_factors(factors)
    if not m:
        return False
    return any(abs(v - 1.0) > 1e-15 for v in m.values())
