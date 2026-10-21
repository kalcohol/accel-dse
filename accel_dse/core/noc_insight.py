"""0.299: NoC / on-chip data-movement energy as a design question (insight).

For the scenario's chip, the same MAC count and SRAM are re-cut into E engines of r × r arrays (many small vs few big
cores) on each NoC topology, and every point is evaluated (time from the cost model, which also re-derives the SRAM
port, accumulators and vector lanes from the engine shape) and priced with the energy table.  When the table has no
NoC numbers the insight fills the reference values (energy.NOC_HOP_PJ_BIT_7NM router, Keckler 10 nm wire 0.078
pJ/bit/mm) and says so.  "SRAM placement": the local-slice layout the model prices vs one central SRAM block, priced
analytically on the same counts: every SRAM-port byte then crosses the mesh from the centre (mean ½·√E hops 「推导」)
and pays the wire of the whole array instead of the slice.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .energy import (NOC_HOP_PJ_BIT_7NM, EnergyTable, energy_report, noc_pitch_mm, slc_wire_mm, sram_pJ_bit)
from .evaluate import evaluate
from .hardware import NOC_TOPOLOGIES

WIRE_REF = 0.078        # Keckler et al. IEEE Micro 2011: 10 nm projection 200 pJ / 256 bit / 10 mm 「来源/推导」
SIDES = (16, 32, 64, 128, 256, 512)


def engine_cuts(macs: int) -> list[tuple[int, int]]:
    """(engines, side) with engines · side² = macs, side a power of two in SIDES."""
    out = []
    for s_ in SIDES:
        e, rem = divmod(int(macs), s_ * s_)
        if rem == 0 and 1 <= e <= 4096:
            out.append((e, s_))
    return sorted(out)


def _fill(table: EnergyTable) -> tuple[EnergyTable, list[str]]:
    notes, kw = [], {}
    if table.pJ_bit_hop_noc is None:
        kw["pJ_bit_hop_noc"] = NOC_HOP_PJ_BIT_7NM
        notes.append(f"能耗表未填 pJ_bit_hop_noc：用参考值 {NOC_HOP_PJ_BIT_7NM:.3f}（Park DAC 2012 45 nm 推导 ÷ 2.6）")
    if table.pJ_bit_mm_onchip is None:
        kw["pJ_bit_mm_onchip"] = WIRE_REF
        notes.append(f"能耗表未填 pJ_bit_mm_onchip：用参考值 {WIRE_REF}（Keckler 2011 10 nm 投影）")
    return (replace(table, **kw) if kw else table), notes


def _central(e: dict, chip, table: EnergyTable) -> dict:
    """Central-SRAM variant on the same counts: SRAM bytes × ½√E mesh hops × (router + pitch wire), and the array
    wire of the whole SRAM instead of one slice; minus nothing else (the A multicast tree is the same size)."""
    per = e["counts_per_unit"]
    sb = per.get("sram", 0.0) * 8
    hops = 0.5 * math.sqrt(chip.engines) if chip.engines > 1 else 0.0
    hop_e = table.pJ_bit_hop_noc + table.pJ_bit_mm_onchip * noc_pitch_mm(table, chip)
    d_wire = table.pJ_bit_mm_onchip * (slc_wire_mm(table, chip.sram_mib) - slc_wire_mm(table, chip.sram_mib / chip.engines))
    extra = sb * (hops * hop_e + d_wire) * 1e-12
    return {"J_per_unit": e["J_per_unit"] + extra, "extra_J": extra, "hops": hops,
            "sram_pJ_bit_local": sram_pJ_bit(table, chip), "basis": "中心 SRAM：SRAM 端口字节 × ½√E 跳 ×（路由 + 间距导线）+ 整块阵列导线"}


def noc_design_report(scn, table: EnergyTable, topologies: tuple = NOC_TOPOLOGIES, phases: tuple = ("decode", "prefill"),
                      prompt: int | None = None, noc_link_Bpc: float | None = None) -> dict:
    if table is None or table.pJ_mac is None:
        raise ValueError("NoC 洞察需要能耗表（左侧「能耗」组或预设，至少 pJ_mac）")
    t, notes = _fill(table)
    ch = scn.chip
    if ch.instance_sched == "cta" or getattr(ch, "gpu", False):
        notes.append("GPU 参考芯片（cta 调度）：NoC 流量未建模（SM 经 L2 交叉开关），本洞察只对 NPU 有意义")
    if noc_link_Bpc is not None and not 0 <= noc_link_Bpc < 1e6:
        raise ValueError("noc_link_Bpc must be in [0, 1e6) bytes / cycle / link")
    L = ch.noc_link_Bpc if noc_link_Bpc is None else float(noc_link_Bpc)
    if L:
        notes.append(f"NoC 拥塞：链路 {L:g} B/周期/方向，数据流 NoC 载荷须穿过拓扑二分（mesh ⌊√E⌋ / ring 2 / crossbar E/2 条），"
                     "超出时按二分带宽计时（与阵列重叠取 max，MODEL §248）；tok_s_free = 不限带宽时的吞吐")
    cuts = engine_cuts(ch.macs)
    if (ch.engines, ch.rows) not in cuts and ch.rows == ch.cols:
        cuts = sorted(set(cuts) | {(ch.engines, ch.rows)})
    rows = []
    for ph in phases:
        sv = scn.serving if ph == scn.serving.phase else replace(scn.serving, phase=ph)
        if ph == "prefill":
            sv = replace(sv, batch=1, prompt=prompt or sv.prompt)
        for e_, side in cuts:
            for topo in topologies:
                c = replace(ch, engines=e_, rows=side, cols=side, noc_topology=topo, noc_link_Bpc=L)
                row = {"phase": ph, "engines": e_, "array": f"{side}×{side}", "topology": topo,
                       "base": e_ == ch.engines and side == ch.rows and topo == ch.noc_topology}
                try:
                    r = evaluate(scn.replace("chip", c).replace("serving", sv))
                    er = energy_report(r, t)
                except ValueError as ex:
                    rows.append({**row, "error": str(ex)})
                    continue
                if er.get("J_per_unit") is None:
                    rows.append({**row, "error": er.get("note") or "能耗表为空"})
                    continue
                b = er["breakdown"]
                row.update(J_per_unit=er["J_per_unit"], unit=er["unit"], tok_s=r.throughput,
                           breakdown=b["J"], share=b["share"], pitch_mm=er["noc"]["pitch_mm"] if "noc" in er else None,
                           byte_hops=er["counts_per_unit"].get("noc", 0.0), sram_B=er["counts_per_unit"].get("sram", 0.0),
                           noc_share=(b["J"]["noc"] + er["J_by_action"].get("noc_wire", 0.0)) / er["J_per_unit"])
                if L:            # 0.303: the same point with an unlimited NoC → the congestion penalty
                    rf = evaluate(scn.replace("chip", replace(c, noc_link_Bpc=0.0)).replace("serving", sv))
                    row["tok_s_free"] = rf.throughput
                    row["noc_penalty"] = 1.0 - r.throughput / rf.throughput if rf.throughput > 0 else 0.0
                if topo == "mesh":
                    row["central"] = _central(er, c, t)
                rows.append(row)
    best = {}
    for ph in phases:
        ok = [x for x in rows if x["phase"] == ph and "error" not in x]
        if ok:
            m = min(ok, key=lambda x: x["J_per_unit"])
            best[ph] = {"engines": m["engines"], "array": m["array"], "topology": m["topology"], "J_per_unit": m["J_per_unit"]}
    return {"chip": ch.name, "model": scn.model, "macs": ch.macs, "sram_mib": ch.sram_mib, "rows": rows, "best": best,
            "notes": notes, "noc_link_Bpc": L, "pJ_bit_hop_noc": t.pJ_bit_hop_noc, "pJ_bit_mm_onchip": t.pJ_bit_mm_onchip,
            "basis": "同 MAC 总数 / SRAM 切成 E 个 r×r 引擎 × 拓扑；时间来自代价模型（端口 / 累加器 / 向量随引擎形状推导），"
                     "能耗 = 能耗表（NoC 由数据流推导，见 MODEL §244）"}
