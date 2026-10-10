"""参考硬件 — validation-only catalog of public GPUs (0.66).

NOT design presets: these exist only so external validation (core/extval.py) can run accel-dse on the same
configuration as a published measurement.  They are deliberately kept out of ``hardware.CHIPS``.

What is taken from vendor datasheets (cited per entry): dense peak tensor FLOPS per format, HBM bandwidth and
capacity, scale-up link bandwidth (bidirectional figure / 2 = per direction).  Everything else is the model's own
「假设」 and is NOT tuned to the measurements: MAC geometry (rows × cols × engines chosen so that R·C·E·2·f equals
the datasheet peak; engines = SM count), L2 as ``sram_mib``, vector lanes = FP32 CUDA cores, link α, mapping
"reconf", DRAM efficiency (catalog default 0.7, or 1.0 in the pure-peak variant).

A100-80GB: stand-in ``hbm3_5s_8h16g_3186`` (5 × 1024 b × 3.186 Gb/s = 2039 GB/s, 80 GiB).
A100-40GB: accel-dse's memory catalog has no HBM2; the stand-in id ``hbm3_5s_4h16g_2430`` has the same bus width,
pin rate, bandwidth (1555 GB/s) and capacity (40 GiB) — only the energy tables differ (unused here).
H200: the catalog's 6 × 24 GiB HBM3e stacks give 144 GiB vs the marketed 141 GB (affects capacity checks only).
"""
from __future__ import annotations

from dataclasses import dataclass

from .dtypes import FormatSupport
from .hardware import Chip, Link

H100_URL = "https://www.nvidia.com/en-us/data-center/h100/"
H200_URL = "https://www.nvidia.com/en-us/data-center/h200/"
H800_URL = ("https://lenovopress.lenovo.com/lp1814.pdf (H800 PCIe sheet: NVLink 400 GB/s, BF16 1,513 sparse); "
            "https://arxiv.org/abs/2412.19437 (DeepSeek-V3 report §3.1: H800 SXM nodes, NVLink, 50 GB/s IB per GPU)")
A100_URL = ("https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/"
            "nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")

MI300X_URL = "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf"
B200_URL = "https://www.nvidia.com/en-us/data-center/dgx-b200/ (8-GPU totals ÷ 8; sparse figures halved)"
_CDNA3 = FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0)))
_BLACKWELL = FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0), ("nvfp4", 4.0), ("mxfp4", 4.0)))
_HOPPER = FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0)))
_AMPERE = FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0), ("int8", 2.0)))


@dataclass(frozen=True)
class RefHW:
    name: str
    chip: Chip
    mem_id: str
    link: Link
    peak_bf16_tflops: float     # datasheet, dense (sparsity figures halved)
    peak_fp8_tflops: float      # 0 = no FP8 tensor cores
    hbm_GBps: float
    hbm_GB: float
    link_GBps_bidir: float
    url: str
    note: str = ""


def _chip(name, sms, mac_per_sm, peak_tflops, fmts, l2_mib, lanes):
    rows, cols = 32, mac_per_sm // 32
    f = peak_tflops * 1e12 / (2 * rows * cols * sms) / 1e9
    return Chip(name, f, rows, cols, sms, formats=fmts, sram_mib=l2_mib, vector_lanes=lanes,
                instance_sched="wide")     # 0.71: GPU references keep the 0.70 calibrated wide schedule (MODEL §25.9 / §26)


REF_HW = {h.name: h for h in (
    RefHW("MI300X", _chip("MI300X 参考硬件", 304, 1024, 1307.4, _CDNA3, 32.0, 19456),
          "hbm3_8s_8h24g_5200", Link(448.0, 3.0), 1307.4, 2614.9, 5300.0, 192.0, 896.0, MI300X_URL,
          "datasheet: 304 CUs, BF16 1,307.4 / FP8 2,614.9 TFLOPS dense, 192 GB HBM3 5.3 TB/s, 7 × 128 GB/s Infinity "
          "Fabric (bidir) — aggregate taken as the scale-up link 「假设」(a ring over the mesh uses less); L2 8 × 4 MB as "
          "sram (256 MB Infinity Cache NOT modelled)"),
    RefHW("B200", _chip("B200 参考硬件", 148, 4096, 2250.0, _BLACKWELL, 126.0, 18944),
          "hbm3e_8s_8h24g_7800", Link(900.0, 3.0), 2250.0, 4500.0, 8000.0, 180.0, 1800.0, B200_URL,
          "DGX B200: 72 PF FP8 / 144 PF FP4 (sparse, 8 GPUs) → 4.5 / 9 PF dense per GPU, BF16 2.25 PF dense 「假设」"
          "(half of FP8), 1,440 GB HBM3e / 8 = 180 GB (catalog 192 GiB stacks), 64 TB/s / 8 = 8 TB/s, NVLink5 1.8 TB/s"),
    RefHW("H800-SXM", _chip("H800-SXM 参考硬件", 132, 2048, 989.5, _HOPPER, 50.0, 16896),
          "hbm3_5s_8h16g_5234", Link(200.0, 3.0), 989.5, 1979.0, 3350.0, 80.0, 400.0, H800_URL,
          "H100-SXM die (same SM count / clocks / HBM3 80 GB 3.35 TB/s 「假设」: NVIDIA publishes no SXM H800 sheet) with "
          "NVLink capped at 400 GB/s bidirectional (PCIe sheet + DeepSeek-V3 report)"),
    RefHW("H100-SXM", _chip("H100-SXM 参考硬件", 132, 2048, 989.5, _HOPPER, 50.0, 16896),
          "hbm3_5s_8h16g_5234", Link(450.0, 3.0), 989.5, 1979.0, 3350.0, 80.0, 900.0, H100_URL,
          "datasheet: BF16 1,979 / FP8 3,958 TFLOPS with sparsity; 80 GB; 3.35 TB/s; NVLink 900 GB/s"),
    RefHW("H200-SXM", _chip("H200-SXM 参考硬件", 132, 2048, 989.5, _HOPPER, 50.0, 16896),
          "hbm3e_6s_8h24g_6250", Link(450.0, 3.0), 989.5, 1979.0, 4800.0, 141.0, 900.0, H200_URL,
          "datasheet: same tensor FLOPS as H100 SXM; 141 GB; 4.8 TB/s; NVLink 900 GB/s"),
    RefHW("A100-SXM-40GB", _chip("A100-40GB 参考硬件", 108, 1024, 312.0, _AMPERE, 40.0, 6912),
          "hbm3_5s_4h16g_2430", Link(300.0, 3.0), 312.0, 0.0, 1555.0, 40.0, 600.0, A100_URL,
          "datasheet: BF16 312 TFLOPS dense; 40 GB HBM2 1,555 GB/s; NVLink 600 GB/s"),
    RefHW("A100-SXM-80GB", _chip("A100-80GB 参考硬件", 108, 1024, 312.0, _AMPERE, 40.0, 6912),
          "hbm3_5s_8h16g_3186", Link(300.0, 3.0), 312.0, 0.0, 2039.0, 80.0, 600.0, A100_URL,
          "datasheet: BF16 312 TFLOPS dense; 80 GB HBM2e 2,039 GB/s (SXM); NVLink 600 GB/s"),
)}
