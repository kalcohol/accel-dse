"""0.102 — catalog-level sanity of the vector-lane budget and the dequant rate against public chips.

Two ratios, each computed the same way for the public datapoints and for a preset:

* ``mac_per_lane`` = dense 16-bit matrix MACs / cycle ÷ elementwise vector lanes / cycle (the model's
  ``Chip.lanes``: one element-op per lane per cycle; FP32 lanes for GPUs — FP32 is what norm / softmax / rope run in).
* ``dequant_headroom`` = conversion elements / s ÷ int4 weight elements streamed / s at the paired DRAM
  bandwidth (2 elements per byte).  ≥ 1 keeps W4A16 decode memory-bound when one conversion costs one lane-op, as
  Marlin (Frantar et al., arXiv 2408.11743) measures on A100 (near-ideal 4-bit speedup up to batch 16–32).

Public datapoints (peak numbers from the vendor documents cited):

* NVIDIA H100 SXM5 — 989.4 TFLOPS dense BF16 tensor, 66.9 TFLOPS FP32 (132 SMs × 128 FP32 lanes, 1.98 GHz),
  3.35 TB/s HBM3 (H100 Tensor Core GPU datasheet / Hopper architecture whitepaper).
* NVIDIA A100 SXM — 312 TFLOPS dense BF16, 19.5 TFLOPS FP32 (108 × 64 lanes, 1.41 GHz), 2.04 TB/s (A100 datasheet).
* AMD MI300X — 1307.4 TFLOPS dense BF16, 163.4 TFLOPS FP32 vector, 5.3 TB/s (AMD Instinct MI300X datasheet).
* Google TPU v4 — per TensorCore 4 MXU × 128×128 MACs; VPU 8 sublanes × 128 lanes × 2 ALUs; 1.05 GHz; 2 cores;
  1.2 TB/s (Cloud TPU system architecture; "How to Scale Your Model", jax-ml scaling book, TPU chapter).
* Google TPU v5p — 4 MXU × 128×128, VPU 8 × 128 × 4 ALUs (~30× less than the MXUs per core; scaling book); 2.76 TB/s.
* Huawei Ascend 910 — 32 DaVinci cores × 4096 FP16 MACs (16³ cube), 2048-bit vector = 128 FP16 / 64 FP32 lanes,
  ~1.0 GHz (256 TFLOPS FP16), 1.2 TB/s HBM2 (Hot Chips 31, Liao et al., "DaVinci"; Ascend C NPU architecture docs).

Caveat (documented, not modelled): GPUs run exp on SFUs at 16 / cycle / SM (1/8 of the FP32 lanes; FA3, Shah et al.
2024, quotes 3.9 TFLOPS exp vs 989 TFLOPS matmul on H100) — the model charges exp at the lane rate.
"""
from __future__ import annotations

from .hardware import CHIPS

# name: (16-bit MACs / cycle, vector lanes / cycle, GHz, DRAM GB/s, source)
PUBLIC = {
    "H100 SXM5": (989.4e12 / 2 / 1.98e9, 132 * 128, 1.98, 3350, "H100 datasheet / Hopper whitepaper"),
    "A100 SXM": (312e12 / 2 / 1.41e9, 108 * 64, 1.41, 2039, "A100 datasheet"),
    "MI300X": (1307.4e12 / 2 / 2.1e9, 163.4e12 / 2 / 2.1e9, 2.1, 5300, "MI300X datasheet"),
    "TPU v4": (2 * 4 * 128 * 128, 2 * 8 * 128 * 2, 1.05, 1200, "Cloud TPU docs; scaling book"),
    "TPU v5p": (2 * 4 * 128 * 128, 2 * 8 * 128 * 4, 1.75, 2765, "scaling book (VPU ~30x below MXU)"),
    "Ascend 910 (FP16 vec)": (32 * 4096, 32 * 128, 1.0, 1200, "Hot Chips 31 DaVinci"),
    "Ascend 910 (FP32 vec)": (32 * 4096, 32 * 64, 1.0, 1200, "Hot Chips 31 DaVinci"),
}


def ratios(macs: float, lanes: float, ghz: float, dram_GBps: float, conv_lanes: float | None = None) -> dict:
    conv = (conv_lanes or lanes) * ghz * 1e9
    return {"mac_per_lane": macs / lanes, "dequant_headroom": conv / (dram_GBps * 1e9 * 2)}


def public_band() -> dict:
    """min / max of each ratio over the public datapoints."""
    rs = [ratios(*v[:4]) for v in PUBLIC.values()]
    hd = [ratios(*v[:4])["dequant_headroom"] for k, v in PUBLIC.items() if "FP32 vec" not in k]  # Ascend dequant: FP16
    return {"mac_per_lane": (min(r["mac_per_lane"] for r in rs), max(r["mac_per_lane"] for r in rs)),
            "dequant_headroom": (min(hd), max(hd))}


def check(chip, dram_GBps: float) -> dict:
    """The preset's two ratios with ``ok`` flags against the public band."""
    macs = chip.rows * chip.cols * chip.engines
    r = ratios(macs, chip.lanes, chip.freq_ghz, dram_GBps, chip.formats.dequant_elems_per_cycle)
    band = public_band()
    return {k: (v, band[k][0] <= v <= band[k][1] if k == "mac_per_lane" else v >= band[k][0]) for k, v in r.items()}


def catalog_report(mems: dict) -> list[dict]:
    """Every catalog NPU preset × every memory in ``mems`` ({mem_id: effective GB/s})."""
    out = []
    for name, ch in CHIPS.items():
        for mid, bw in mems.items():
            c = check(ch, bw)
            out.append({"chip": name, "mem": mid, **{k: v for k, (v, _) in c.items()},
                        **{k + "_ok": ok for k, (_, ok) in c.items()}})
    return out
