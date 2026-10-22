"""0.326: HBM package option — SPHBM4 on an organic substrate (Chip.hbm_pkg "sphbm4"; "si" = HBM on a silicon
interposer, ≤ 0.325).  Opt-in; HBM4 only.

What JEDEC JESD330-4 (Standard Package HBM4, v1.0 Jun 2026; PR 2026-07-13) defines 「来源」:
  * the same DRAM core dies as HBM4 (JESD270-4) on a different buffer / base die → capacity per stack unchanged;
  * 512 external DQ (32 × 16-bit channels in 8 Quad Channels) with 4:1 serialisation, each channel "four times faster
    than the corresponding HBM4 channel (of 64 data bits)" → stack bandwidth = the HBM4 stack at pin rate ÷ 4;
  * bump pitch > 90 µm and channels up to 20 mm long → standard organic substrate routing, no silicon interposer.
  * Rates 22.4 – 46.0 GT/s per pin, 2.944 TB/s per stack at 46 (Tom's Hardware / Igor's Lab on JESD330-4 「二手」)
    → an HBM4-equivalent rate of 5600 – 11500 MT/s 「推导」; outside it the option is ignored (warning).

What the model changes when it applies (mechanism + source):
  shoreline / stack       11 mm (silicon: stack abuts the host) → 25 / 3 = 8.33 mm.  Eliyan (Electronic Design
                          2026-02-18): on silicon "two per 25-mm die edge"; on organic "three memory devices can be
                          connected along the same edge" (angled routing within the longer reach) 「二手」.  Eliyan's own
                          silicon figure is 12.5 mm/stack vs the model's 11 (Micron 11 × 11 footprint), so the gain
                          here is 11 / 8.33 = 1.32×, not Eliyan's 1.5× — the organic figure is taken as published,
                          not scaled.
  interposer area         the stacks are off the interposer: the package-area limit (CoWoS-S / -L) and the 0.313
                          bounding box count the logic dies only (JEDEC: "mounting on standard organic substrates rather
                          than silicon substrates" 「来源」).
Unchanged, with the reason (NOT_MODELLED): no public figure for the SPHBM4 host PHY area or energy, the organic
substrate $/mm², the flip-chip attach yield, or an SPHBM4 $/GB.  Package $ keeps the stack footprint at the CoWoS
rate (an upper bound); report() gives the most that dropping it could save."""
from __future__ import annotations

PKG_TYPES = ("si", "sphbm4")
SPHBM4_TYPES = ("HBM4",)          # JESD330-4 is defined on HBM4 (JESD270-4) stacks

SPHBM4_DQ = 512                                     # 来源 JEDEC PR 2026-07-13
SPHBM4_SERDES = 4                                   # 来源 JEDEC: 4:1 serialisation / "four times faster"
SPHBM4_GTPS = (22.4, 46.0)                          # 二手 Tom's Hardware / Igor's Lab on JESD330-4
SPHBM4_BUMP_PITCH_UM_MIN = 90                       # 来源 JEDEC PR: "> 90 µm"
SPHBM4_REACH_MM = 20                                # 来源 JEDEC PR: "up to 20 mm"
SPHBM4_STACKS_PER_25MM = 3                          # 二手 Eliyan, Electronic Design 2026-02-18
SI_STACKS_PER_25MM = 2                              # 二手 same article (silicon substrate)
SPHBM4_EDGE_MM = 25.0 / SPHBM4_STACKS_PER_25MM     # 8.333 mm per stack 「二手」

SRC = {
    "edge": "Eliyan（Electronic Design 2026-02-18）：硅基板每 25 mm 晶粒边 2 堆；有机基板 SPHBM4 同一边 3 堆 → 25 / 3 = 8.33 mm/堆「二手」",
    "iface": "JEDEC JESD330-4（2026-06；PR 2026-07-13）：同 HBM4 DRAM 芯、512 DQ、4:1 串行化、凸点 > 90 µm、通道 ≤ 20 mm「来源」",
    "rate": "速率 22.4 – 46.0 GT/s、46 GT/s 时每栈 2.944 TB/s（Tom's Hardware / Igor's Lab 转述 JESD330-4「二手」）",
}

NOT_MODELLED = (
    "SPHBM4 主机侧 PHY 面积：无公开数字（Eliyan 仅定性称“往往不更大”）——沿用 HBM4 PHY 面积",
    "SPHBM4 接口 pJ/bit：无公开测量（Igor's Lab：串行化 / 均衡 / 纠错可能增加功耗与时延，“无可靠产品测量”）——沿用 HBM4 PHY 能耗",
    "有机基板 $/mm² 与 SPHBM4 倒装贴装良率：无公开数字（SemiAnalysis 称需 20 – 28+ 层高端 ABF，无价格）——封装 $ 仍按 CoWoS 费率计堆栈占地（上界）、贴装良率不变",
    "SPHBM4 $/GB：无公开价格（Tom's Hardware：基础晶粒“可能比 HBM4 的更贵”）——沿用 HBM4 $/GB",
    "有机基板最大尺寸 / 层数上限、20 mm 走线的布线可行性：无公开数字——不检查",
    "SPHBM4 时延（SerDes / 训练 / 纠错）：无公开数字——不计",
)


def rate_range_MTps() -> tuple[float, float]:
    """HBM4-equivalent pin rates SPHBM4 can carry: GT/s range × 1000 ÷ 4 「推导」."""
    lo, hi = SPHBM4_GTPS
    return lo * 1000 / SPHBM4_SERDES, hi * 1000 / SPHBM4_SERDES


def sphbm4_applies(chip, spec) -> bool:
    if getattr(chip, "hbm_pkg", "si") != "sphbm4" or spec is None:
        return False
    if spec.mem_type.upper() not in SPHBM4_TYPES:
        return False
    lo, hi = rate_range_MTps()
    return lo - 1e-9 <= spec.rate_MTps <= hi + 1e-9


def ignored_note(chip, spec) -> str | None:
    if getattr(chip, "hbm_pkg", "si") != "sphbm4" or sphbm4_applies(chip, spec):
        return None
    lo, hi = rate_range_MTps()
    if spec is None or spec.mem_type.upper() not in SPHBM4_TYPES:
        return (f"hbm_pkg=sphbm4 只对 HBM4 堆栈有效（JESD330-4 基于 HBM4 DRAM 芯）："
                f"{getattr(spec, 'mem_type', None) or '该外存'} 按硅中介层计")
    return (f"hbm_pkg=sphbm4 不适用：{spec.rate_MTps} MT/s × 4 = {spec.rate_MTps * 4 / 1000:g} GT/s 超出 JESD330-4 "
            f"22.4 – 46.0 GT/s（HBM4 等效 {lo:g} – {hi:g} MT/s）——按硅中介层计")


def stack_edge_mm(chip, spec) -> float:
    """Die edge one HBM stack takes: 8.33 mm with SPHBM4 on organic, else the 11 mm stack footprint."""
    from .budget import HBM_STACK_EDGE_MM
    return SPHBM4_EDGE_MM if sphbm4_applies(chip, spec) else HBM_STACK_EDGE_MM


def report(scn, sh: dict | None = None, cost_params: dict | None = None) -> dict:
    """What the option changed for this scenario (eval): applicability, edge per stack, interface, stack bandwidth,
    the interposer area it frees, and the package-$ bound."""
    from .. import mem_catalog
    from .budget import HBM_STACK_EDGE_MM
    from .energy_dse import HBM_FOOTPRINT_DEFAULT, HBM_FOOTPRINT_MM2
    sp = mem_catalog.parse_mem_id(scn.mem_id)
    on = sphbm4_applies(scn.chip, sp)
    lo, hi = rate_range_MTps()
    out = {"mode": getattr(scn.chip, "hbm_pkg", "si"), "applies": on, "mem_type": sp.mem_type,
           "edge_per_stack_mm": SPHBM4_EDGE_MM if on else HBM_STACK_EDGE_MM, "si_edge_per_stack_mm": HBM_STACK_EDGE_MM,
           "dq": SPHBM4_DQ, "pin_GTps": sp.rate_MTps * SPHBM4_SERDES / 1000, "gtps_range": list(SPHBM4_GTPS),
           "hbm4_equiv_MTps_range": [lo, hi], "bump_pitch_um_min": SPHBM4_BUMP_PITCH_UM_MIN,
           "reach_mm": SPHBM4_REACH_MM, "src": dict(SRC), "not_modelled": list(NOT_MODELLED), "notes": []}
    n = ignored_note(scn.chip, sp)
    if n:
        out["notes"].append(n)
    if on:
        stack_GBps = sp.unit_width_bits * sp.rate_MTps / 8 / 1000
        out["stack_GBps"] = stack_GBps                  # = 512 × pin rate ÷ 8 「推导」
        out["need_mm"] = sp.n_units * SPHBM4_EDGE_MM
        out["si_need_mm"] = sp.n_units * HBM_STACK_EDGE_MM
        fp = sp.n_units * HBM_FOOTPRINT_MM2.get(sp.mem_type, HBM_FOOTPRINT_DEFAULT)
        out["interposer_freed_mm2"] = fp
        if cost_params and cost_params.get("pkg_usd_per_mm2") is not None:
            out["pkg_usd_bound"] = fp * cost_params["pkg_usd_per_mm2"]
        out["notes"].append(
            f"hbm_pkg=sphbm4：每栈岸线 {HBM_STACK_EDGE_MM:g} → {SPHBM4_EDGE_MM:.2f} mm（{SRC['edge']}）；"
            f"{sp.n_units} 栈需 {out['si_need_mm']:.0f} → {out['need_mm']:.1f} mm；接口 {SPHBM4_DQ} DQ × "
            f"{out['pin_GTps']:g} GT/s = 每栈 {stack_GBps:.0f} GB/s（与 HBM4 @{sp.rate_MTps} 相同）；堆栈移出中介层 "
            f"{fp:.0f} mm²（中介层面积上限只计逻辑晶粒）")
        if "pkg_usd_bound" in out:
            out["notes"].append(f"封装 $ 仍按 CoWoS 费率计堆栈占地（有机基板 $/mm² 无公开数字）：若该面积不计费，最多省 "
                                f"${out['pkg_usd_bound']:.0f}/封装「推导·上界」——未计入")
    if sh and sh.get("mem"):
        out["shoreline_status"] = sh.get("status")
        if on and sh.get("die_mm2"):
            from dataclasses import replace
            from .budget import shoreline_report
            si = shoreline_report(scn.replace("chip", replace(scn.chip, hbm_pkg="si")), sh["die_mm2"])
            e0, e1 = (si.get("mem") or {}).get("edges"), sh["mem"].get("edges")
            out["si_shoreline_status"], out["si_mem_edges"], out["mem_edges"] = si.get("status"), e0, e1
            if e0 is not None and e1 is not None and e1 != e0:
                out["notes"].append(
                    f"外存接入边 {e0} → {e1}：自动规则取能放下堆栈的最少边（0.309 / 0.312），堆栈更窄 → 接入边更少 → "
                    f"DRAM 经 NoC 的跳数 / 接入链路随之变化——这是放置规则的结果，不是 SPHBM4 本身；设 noc_mem_edges = {e0} "
                    "则 NoC 与硅中介层相同")
    return out
