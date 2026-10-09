"""Numeric formats (storage + compute) and the hardware native-support matrix.

Models run **as released** (v0.40): each weight / KV / activation tensor keeps
the format of the official release; a hardware ``FormatSupport`` table says
which compute formats are native and at what MAC-rate multiplier. An op whose
operands are not natively supported is executed in the nearest native wider
format and pays a conversion (dequant / upcast) cost on the vector path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    name: str
    bits: float            # storage bits per element incl. amortised scales
    family: str            # float | int | mx
    compute_bits: int      # element width used for compute-format matching
    note: str = ""

    @property
    def bytes(self) -> float:
        return self.bits / 8.0


FORMATS: dict[str, Format] = {f.name: f for f in (
    Format("fp32", 32, "float", 32),
    Format("bf16", 16, "float", 16),
    Format("fp16", 16, "float", 16),
    # DeepSeek-style block FP8: e4m3 + one fp32 scale per 128×128 block
    Format("fp8", 8 + 32 / (128 * 128), "float", 8, "e4m3, block-128 fp32 scale"),
    Format("fp8_tensor", 8, "float", 8, "e4m3, per-tensor/channel scale"),
    # OCP MXFP4: e2m1 + one e8m0 scale per 32 elements
    Format("mxfp4", 4 + 8 / 32, "mx", 4, "e2m1 + e8m0 per 32"),
    Format("nvfp4", 4 + 8 / 16, "mx", 4, "e2m1 + e4m3 per 16"),
    Format("int8", 8, "int", 8),
    # AWQ / GPTQ / compressed-tensors int4: group 128, fp16 scale (+ 4-bit zero)
    Format("int4", 4 + (16 + 4) / 128, "int", 4, "group-128 scale+zero"),
)}

ACT_DEFAULT = "bf16"


def fmt(name: str) -> Format:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown format {name!r}; known: {sorted(FORMATS)}") from None


@dataclass(frozen=True)
class FormatSupport:
    """Native compute formats → MAC-rate multiplier vs the bf16 array rate.

    Design variable (「假设」default: bf16/fp16 ×1, fp8 ×2, int8 ×2; 4-bit and
    fp32 not native). ``dequant_elems_per_cycle`` = conversion throughput of
    the vector/dequant path (elements per cycle per chip; None → = vector lanes).
    """

    rates: tuple[tuple[str, float], ...] = (("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0))
    dequant_elems_per_cycle: float | None = None

    def __post_init__(self):
        # 0.61.3: rates as (str, float) so equal chips serialise / hash identically (["fp8", 2] vs ["fp8", 2.0])
        try:
            rs = tuple((str(n), float(r)) for n, r in self.rates)
        except (TypeError, ValueError):
            return          # left for the scenario validation to report
        object.__setattr__(self, "rates", rs)
        d = self.dequant_elems_per_cycle
        if isinstance(d, int) and not isinstance(d, bool):
            object.__setattr__(self, "dequant_elems_per_cycle", float(d))

    def rate(self, name: str) -> float | None:
        base = "fp8" if name.startswith("fp8") else name
        for n, r in self.rates:
            if n == base:
                return r
        return None

    def native(self, name: str) -> bool:
        return self.rate(name) is not None


def gemm_exec(w_fmt: str, a_fmt: str, support: FormatSupport) -> tuple[str, float, str, int]:
    """Execution format for a GEMM with weight format ``w_fmt`` × activation ``a_fmt``.

    Returns (exec_format, mac_rate_multiplier, conversion, elements_converted_flag)
    conversion ∈ {"none", "dequant_w", "upcast_both"}; flag bit0 = weights, bit1 = activations.
      * same native format (e.g. W8A8 fp8 on fp8 hardware) → native rate
      * weight-only quant (W lower than A): dequant W to A's format (if A native)
      * nothing native: upcast both to bf16 (if bf16 native) else fp16/fp32
    """
    w, a = fmt(w_fmt), fmt(a_fmt)
    if w.compute_bits == a.compute_bits and support.native(w_fmt) and support.native(a_fmt):
        return w_fmt, float(support.rate(w_fmt)), "none", 0
    if w.compute_bits < a.compute_bits and support.native(a_fmt):
        return a_fmt, float(support.rate(a_fmt)), "dequant_w", 1
    for wide in ("bf16", "fp16", "fp32"):
        if support.native(wide):
            flag = (0 if w_fmt == wide else 1) | (0 if a_fmt == wide else 2)
            return wide, float(support.rate(wide)), ("upcast_both" if flag else "none"), flag
    raise ValueError("hardware supports none of bf16/fp16/fp32")
