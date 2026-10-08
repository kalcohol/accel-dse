"""Die-to-die (D2D) interconnect catalog (0.50) — published figures only, each with its source.

Bandwidth is **raw, per direction, per unit** (a UCIe module, a BoW slice, or the whole published link for the
proprietary entries), before flit / protocol framing; the model's D2D tier uses ``per_unit_GBps × units`` per card
(die).  How many units a die carries on its package-facing edge is a floorplan choice → ``d2d_units`` 「假设」.
Energy figures are the published targets / measurements, shown for reference only — the energy table stays
user-supplied (core/energy.py).  Older interfaces (e.g. CCIX) are deliberately not listed.

Sources
  UCIe   UCIe Consortium: UCIe 2.0 overview (FMS 2024, D. Das Sharma) and Hot Chips 2023 UCIe tutorial — Standard
         package (UCIe-S, 2D): x16 module, 100–130 µm bump pitch, ≤ 25 mm reach, 0.5 pJ/bit target, 64 GB/s per
         module per direction at 32 GT/s; Advanced package (UCIe-A, 2.5D): x64 module, 25–55 µm pitch, ≤ 2 mm reach,
         0.25 pJ/bit target, 256 GB/s per module per direction at 32 GT/s; Tx+Rx latency < 2 ns; rates 4–32 GT/s;
         1, 2 or 4 modules per link.  UCIe 3.0 (Aug 2025, uciexpress.org/specifications) adds 48 and 64 GT/s.
         48 / 64 GT/s bandwidth = lanes × rate assuming 1 bit per transfer per lane (verified by the consortium's own
         arithmetic only at 32 GT/s) 「假设」; pJ/bit targets for 48 / 64 GT/s not published → the 2.0 target is shown.
  BoW    OCP Bunch of Wires PHY spec 2.0: 16-wire slice, BoW-256 = 16 Gb/s/wire (256 Gb/s per slice per direction),
         BoW-512 = 32 Gb/s/wire; < 0.5–1 pJ/bit doubly terminated (< 0.25–0.5 unterminated); < 2–4 ns without FEC.
  NVLink-C2C  NVIDIA Grace Hopper whitepaper: 900 GB/s total = 450 GB/s per direction, 1.3 pJ/bit.
Researched but not in the selector (user-approved list = UCIe-A 48 / 64, UCIe-S 32, BoW, NVLink-C2C): NVIDIA NV-HBI
(Blackwell, "10 TB/s", direction not stated), AMD Infinity Fabric AP (MI300X IOD↔IOD 2.4 / 3.0 TB/s per direction),
TSMC LIPINCON (VLSI 2019 test chip, 8 Gb/s/pin, 320 GB/s, 0.56 pJ/bit).  Use d2d_std = "custom" for any other figure.
"""

from __future__ import annotations

D2D_STANDARDS: dict[str, dict] = {      # the user-approved selector (0.50); order = UI order
    "ucie-a-48": dict(label="UCIe-A x64 @ 48 GT/s（UCIe 3.0 Advanced 2.5D）· 常用", family="UCIe", unit="x64 module",
                      per_unit_GBps=384.0, pJ_bit=0.25, reach_mm=2, pitch_um="25–55", open=True,
                      note="48 GT/s × 64 lanes ÷ 8；1 bit/transfer「假设」；pJ/bit 为 UCIe 2.0 目标（3.0 目标未公开）"),
    "ucie-a-64": dict(label="UCIe-A x64 @ 64 GT/s（UCIe 3.0 Advanced 2.5D）", family="UCIe", unit="x64 module",
                      per_unit_GBps=512.0, pJ_bit=0.25, reach_mm=2, pitch_um="25–55", open=True,
                      note="1 bit/transfer「假设」；64 GT/s BER 目标 1e-12（48 GT/s 为 1e-15）；pJ/bit 为 UCIe 2.0 目标"),
    "ucie-s-32": dict(label="UCIe-S x16 @ 32 GT/s（Standard 2D 有机基板）", family="UCIe", unit="x16 module",
                      per_unit_GBps=64.0, pJ_bit=0.5, reach_mm=25, pitch_um="100–130", open=True,
                      note="64 GB/s 每模块每方向（UCIe 联盟教程原值）"),
    "bow-256": dict(label="BoW-256（OCP Bunch of Wires 2.0，16 Gb/s/wire）", family="BoW", unit="16-wire slice",
                    per_unit_GBps=32.0, pJ_bit=0.5, reach_mm=25, pitch_um="—", open=True,
                    note="pJ/bit：双端端接 < 0.5–1，非端接 < 0.25–0.5（列 0.5 作参考）"),
    "bow-512": dict(label="BoW-512（OCP Bunch of Wires 2.0，32 Gb/s/wire）", family="BoW", unit="16-wire slice",
                    per_unit_GBps=64.0, pJ_bit=0.5, reach_mm=25, pitch_um="—", open=True, note="同上"),
    "nvlink-c2c": dict(label="NVIDIA NVLink-C2C（厂商专有，Grace Hopper）", family="vendor", unit="link",
                       per_unit_GBps=450.0, pJ_bit=1.3, reach_mm=None, pitch_um="—", open=False,
                       note="900 GB/s 双向合计 = 450 GB/s 每方向（NVIDIA 白皮书）；厂商专有接口，仅作参照"),
}
D2D_DEFAULT_STD = "ucie-a-48"
D2D_DEFAULT_UNITS = 4          # 「假设」: four x64 modules on the die's package-facing edge


def d2d_GBps(std: str, units: int) -> float:
    return D2D_STANDARDS[std]["per_unit_GBps"] * units
