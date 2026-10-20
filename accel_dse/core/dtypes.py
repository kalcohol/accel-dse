"""Numeric formats (storage + compute) and the hardware native-support matrix.

Models run **as released** (v0.40): each weight / KV / activation tensor keeps
the format of the official release; a hardware ``FormatSupport`` table says
which compute formats are native and at what MAC-rate multiplier. An op whose
operands are not natively supported is executed in the nearest native wider
format and pays a conversion (dequant / upcast) cost on the vector path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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
    # 0.289 ggml block formats (ggml-common.h 「来源」), weight-only (dequantised to the activation format in the kernel):
    # IQ2_XXS = fp16 scale + 32 × uint16 (E8-lattice grid index + signs) per 256 weights = 66 B; Q8_0 = fp16 scale + 32 int8
    Format("iq2_xxs", 66 * 8 / 256, "int", 2, "ggml IQ2_XXS: 66 B per 256 (fp16 d + 8×grid/sign words)"),
    Format("q8_0", 34 * 8 / 32, "int", 8, "ggml Q8_0: fp16 d + 32 int8"),
)}

ACT_DEFAULT = "bf16"

# 0.290: vector element-ops per converted element.  Every format is charged the 0.40 convention of 2 (unpack / convert
# + one scale multiply 「假设」, evaluate._op_seconds) — except the ggml IQ2_XXS codebook format, whose reference dequant
# does more per weight 「推导」 from ggml dequantize_row_iq2_xxs (ggml-quants.c) / dequantize_iq2_xxs (ggml-cuda
# dequantize.cuh, llama.cpp master 839a572a4c, 2026-10-21 JST):
#   per weight      2      the convention (grid byte → float, × the sub-block scale db)
#                 + 1      sign: (ksigns_iq2xs[...] & kmask_iq2xs[j]) ? -1 : 1
#   per 8 weights + 4 / 8  two table lookups (iq2xxs_grid: 256 × uint64 = 2 KiB; ksigns_iq2xs: 128 B) + two index ops
#                          (sign index (aux32[1] >> 7·l) & 127; the grid index is a byte of the 32-bit word)
#   per 32        + 5 / 32 sub-block scale db = d · (0.5 + (aux32[1] >> 28)) · 0.25: shift, int → float, add, 2 mul
#   = 3.65625.  The fp16 block scale d (1 per 256) is not charged — the convention does not charge stored scales
#   (nvfp4 e4m3 per 16, int4 fp16 per 128) either.  Q8_0 (y = qs[j] · d, ggml dequantize_row_q8_0) is the convention.
#   A lookup is charged as one lane op (the 2 KiB + 128 B tables sit in each core's local memory「假设」).
#   A chip that lists the (format, activation) pair in FormatSupport.mixed decodes in the MAC datapath: no dequant.
DEQUANT_OPS = {"iq2_xxs": 2 + 1 + 4 / 8 + 5 / 32}


def dequant_ops(name: str) -> float:
    """0.290: vector element-ops to convert one element of ``name`` to the execution format (see DEQUANT_OPS)."""
    return DEQUANT_OPS.get(name, 2.0)

# 0.147: mixed-precision MAC presets (W, A, rate vs bf16) for the UI / insight knobs.  W4A8 at the FP8 rate is sourced
# (NVIDIA PTX ISA 8.7, tcgen05.mma kind::f8f6f4: fp4 / fp6 / fp8 operand mixes at the f8f6f4 = FP8 rate); W4A16 / W8A16
# at the bf16 rate are 「假设」 (the 16-bit operand sets the multiplier; the gain is the removed dequant only).
MIXED_PRESETS = {
    "w4a8": ((("mxfp4", "fp8", 2.0), ("int4", "int8", 2.0), ("nvfp4", "fp8", 2.0)), "PTX ISA 8.7 tcgen05.mma kind::f8f6f4（W4A8 = FP8 速率）"),
    "w4a16": ((("mxfp4", "bf16", 1.0), ("int4", "bf16", 1.0), ("int4", "fp16", 1.0), ("nvfp4", "bf16", 1.0)), "「假设」W4A16 = bf16 速率（只省反量化）"),
    "w8a16": ((("fp8", "bf16", 1.0), ("int8", "bf16", 1.0), ("int8", "fp16", 1.0)), "「假设」W8A16 = bf16 速率（只省反量化）"),
}


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
    # 0.147: native mixed-precision MAC pairs (weight format, activation format, MAC-rate multiplier vs bf16) — a design
    # variable, empty by default (no shipped chip changes).  A GEMM whose (W, A) pair is listed runs on the array at
    # that rate with no weight dequant (gemm_exec); otherwise the pre-0.147 rules apply.  Rate guidance: Blackwell
    # tcgen05.mma kind::f8f6f4 takes e2m1 × e4m3 operands at the FP8 rate (PTX ISA 8.7) → W4A8 = fp8 rate; W4A16 /
    # W8A16 at the 16-bit rate 「假设」 (the wider operand sets the multiplier array).  MIXED_PRESETS below.
    mixed: tuple[tuple[str, str, float], ...] = field(default=(), metadata={"omit_default": True})

    def __post_init__(self):
        # 0.61.3: rates as (str, float) so equal chips serialise / hash identically (["fp8", 2] vs ["fp8", 2.0])
        try:
            rs = tuple((str(n), float(r)) for n, r in self.rates)
        except (TypeError, ValueError):
            return          # left for the scenario validation to report
        object.__setattr__(self, "rates", rs)
        try:
            mx = tuple((str(w), str(a), float(r)) for w, a, r in self.mixed)
        except (TypeError, ValueError):
            return
        object.__setattr__(self, "mixed", mx)
        for w, a, r in mx:
            fmt(w), fmt(a)
            if not r > 0 or not math.isfinite(r):
                raise ValueError(f"formats.mixed rate for {w}×{a} must be a finite number > 0")
        d = self.dequant_elems_per_cycle
        if isinstance(d, int) and not isinstance(d, bool):
            object.__setattr__(self, "dequant_elems_per_cycle", float(d))

    def pair_rate(self, w_fmt: str, a_fmt: str) -> float | None:
        """0.147: native mixed-precision MAC rate for weights ``w_fmt`` × activations ``a_fmt`` (None = not native)."""
        b = lambda n: "fp8" if n.startswith("fp8") else n          # noqa: E731
        for w, a, r in self.mixed:
            if b(w) == b(w_fmt) and b(a) == b(a_fmt):
                return r
        return None

    def rate(self, name: str) -> float | None:
        if "*" in name:                                           # 0.147 mixed exec label "<w>*<a>"
            w, a = name.split("*", 1)
            return self.pair_rate(w, a)
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
    conversion ∈ {"none", "dequant_w", "upcast_both", "downcast"}; flag bit0 = weights, bit1 = activations.
      * same native format (e.g. W8A8 fp8 on fp8 hardware) → native rate; same width, other type → as below
      * weight-only quant (W lower than A): dequant W to A's format (if A native)
      * nothing native: upcast both to bf16 (if bf16 native) else fp16/fp32
    """
    w, a = fmt(w_fmt), fmt(a_fmt)
    base = lambda n: "fp8" if n.startswith("fp8") else n
    # 0.61.4: the native path needs one compute format — same width *and* the same element type (fp8 × int8 or
    # bf16 × fp16 have no common MAC path: both go to the widest native format below, was taken as native)
    if base(w_fmt) == base(a_fmt) and support.native(w_fmt) and support.native(a_fmt):
        return w_fmt, float(support.rate(w_fmt)), "none", 0
    pr = support.pair_rate(w_fmt, a_fmt) if support.mixed else None
    if pr is not None:      # 0.147: native mixed-precision MAC — no dequant, the pair's own rate
        return f"{w_fmt}*{a_fmt}", float(pr), "none", 0
    if w.compute_bits < a.compute_bits and support.native(a_fmt):
        return a_fmt, float(support.rate(a_fmt)), "dequant_w", 1
    for wide in ("bf16", "fp16", "fp32"):
        if support.native(wide):
            flag = (0 if w_fmt == wide else 1) | (0 if a_fmt == wide else 2)
            # 0.113: an input wider than the exec format (fp32 on hardware without fp32) is narrowed — labelled
            # "downcast", was "upcast_both" (label only; the conversion cost is the same element count)
            down = (flag & 1 and w.compute_bits > fmt(wide).compute_bits) or (flag & 2 and a.compute_bits > fmt(wide).compute_bits)
            return wide, float(support.rate(wide)), ("downcast" if down else "upcast_both" if flag else "none"), flag
    raise ValueError("hardware supports none of bf16/fp16/fp32")
