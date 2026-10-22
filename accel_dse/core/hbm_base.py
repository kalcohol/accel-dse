"""0.324: custom HBM base die — a design option (Chip.hbm_base "custom"; "std" = the JEDEC HBM4 interface, ≤ 0.323).

From HBM4 on the base die is made on a logic process (TSMC N12 for SK hynix HBM4, Samsung SF4, Micron in-house CMOS;
custom HBM4E / NVHBM on TSMC N3-class).  A *custom* base die takes the memory controller off the host and replaces
the 2048-DQ HBM PHY with a narrow die-to-die link (Samsung Hot Chips 2026; Marvell custom HBM; NVIDIA NVHBM).  What
the model changes, each number labelled:

  host PHY area / stack   std  2048 DQ × mm2_per_dq_hbm (Samsung HC 2026: conventional PHY > 8 × 4 mm 「二手」)
                          custom 8.5 × 1.5 mm D2D block (same talk, memory side 「二手」; host side taken equal — the
                          step the std PHY already makes 「推导」)
  memory controller       moved to the base die: the host term mm2_per_unit_mc (user field, no preset) drops out
  host PHY energy / bit   × 0.55 — Marvell: interface power −45 … −70 % "depending on scenarios" (EE Times part 2;
                          Marvell PR "up to 70 %") 「二手」; the low end is the default (vendor claims, best case unknown)
  memory $/GB             × 1.10 — custom HBM4E premium "5 % to 10 % overhead tops" (analyst speculation, EE Times
                          part 1) 「二手（推测）」; the upper end is the default
  shoreline / stack       unchanged: the stack's 11 mm footprint sets the edge, and the D2D block (8.5 mm) is no wider
                          than the PHY (8 mm) was 「推导」 — the custom die shrinks PHY *depth* (4 → 1.5 mm), i.e. area,
                          not the edge a stack needs.  Marvell's "+33 % stacks" has no published mechanism: not modelled
  stack bandwidth         unchanged (the DRAM stack is JEDEC HBM4 / 4E).  NVHBM "+30 % bandwidth" comes with a
                          faster interface spec, Marvell's custom stack is 2.048 TB/s (< 3.072 of a 12 GT/s HBM4E):
                          neither is a base-die property, so neither is applied
  near-memory compute     Samsung phase 2 (functions on the base die) — no public figure: listed, not modelled
  base-die process energy N12 0.8 V → N3P 0.75 V (TSMC via HardwareLUXX / TrendForce 「二手」) — the base die's share of
                          the device pJ/bit is unpublished: listed, not modelled
Only HBM4 / HBM4E carry a logic base die; on other memory the switch is ignored with a note.  MODEL §269.
"""
from __future__ import annotations

HBM_BASES = ("std", "custom")
CUSTOM_TYPES = ("HBM4", "HBM4E")

_HC26 = ("Samsung, S. Han, 'HBM Base Die: How HBM Will Evolve Using Advanced Logic Processes', Hot Chips 2026 "
         "tutorial (via Tom's Hardware / ServeTheHome 2026-08)")
CUSTOM = {
    "d2d_mm2_per_stack": (8.5 * 1.5, "二手/推导", _HC26 + ": custom HBM D2D block ≈ 8.5 × 1.5 mm on the 11 × 12.8 mm "
                          "HBM4 base die (conventional PHY > 8 × 4 mm; channel depth 5.5 → 2 mm); host side taken equal "
                          "to the memory side (推导, as for the std PHY)"),
    "phy_pJ_factor": (0.55, "二手", "Marvell custom HBM: interface power −45 % to −70 % depending on scenarios (EE Times "
                      "'The Industry is Striving for Custom Memory (Part 2)'); 'up to 70 %' (Marvell PR 2024-12) — low end "
                      "taken (× 0.55)"),
    "usd_per_GB_factor": (1.10, "二手/假设", "custom HBM4E premium '5 % to 10 % overhead tops' (analyst speculation, EE "
                          "Times Part 1); HBM4 base die already 3–4 × a core die's cost (Herald Business via TrendForce "
                          "2026-08-31) — upper end taken (× 1.10)"),
    "process": ("TSMC N3P（C-HBM4E）/ N3（NVHBM）", "二手", "TSMC OIP 2025 via HardwareLUXX / TrendForce 2025-12-01 (C-HBM4E "
                "N3P, 0.75 V); NVIDIA NVHBM base dies on TSMC 3 nm (Seoul Economic Daily via TrendForce 2026-10-19)"),
    "std_process": ("TSMC N12（SK hynix HBM4）/ Samsung SF4 / Micron 自有 CMOS", "二手", "Chosun Daily via TrendForce "
                    "2026-03-20 (SK hynix TSMC 12 nm, Samsung 4 nm); Micron (EE Times Part 1: in-house CMOS base die)"),
}
PJ_FACTOR_RANGE = (0.30, 0.55)      # −70 % … −45 % 「二手」
USD_FACTOR_RANGE = (1.05, 1.10)     # +5 % … +10 % 「二手（推测）」
NOT_MODELLED = ("近存计算 / 卸载（Samsung 第 2 阶段，基础晶粒上加功能）：无公开数字，未建模",
                "基础晶粒工艺能耗（N12 0.8 V → N3P 0.75 V）：基础晶粒在器件 pJ/bit 中的份额未公开，未建模",
                "Marvell「多 33 % 堆栈」：无公开机理（堆栈 11 mm 占地不变），未建模",
                "定制接口功率密度更高（Samsung 以 Heat Path Block 降峰温 > 35 %）：热点未建模")


def custom_applies(chip, mem_type: str | None) -> bool:
    return getattr(chip, "hbm_base", "std") == "custom" and bool(mem_type) and mem_type.upper() in CUSTOM_TYPES


def ignored_note(chip, mem_type: str | None) -> str | None:
    if getattr(chip, "hbm_base", "std") == "custom" and not custom_applies(chip, mem_type):
        return f"hbm_base=custom 只对 HBM4 / HBM4E（逻辑工艺基础晶粒）有效：{mem_type or '该外存'} 按标准接口计"
    return None


def report(scn, b=None) -> dict:
    """What the switch does for this scenario (eval / report block)."""
    from .. import mem_catalog
    sp = mem_catalog.parse_mem_id(scn.mem_id)
    on = custom_applies(scn.chip, sp.mem_type)
    out = {"mode": getattr(scn.chip, "hbm_base", "std"), "applies": on, "mem_type": sp.mem_type,
           "stacks": sp.n_units if "HBM" in sp.mem_type.upper() else 0,
           "process": CUSTOM["process" if on else "std_process"][0], "edge_per_stack_mm": None, "notes": []}
    if "HBM" in sp.mem_type.upper():
        from .budget import HBM_STACK_EDGE_MM
        out["edge_per_stack_mm"] = HBM_STACK_EDGE_MM
        dq = getattr(b, "mm2_per_dq_hbm", None) if b is not None else None
        std = sp.bus_bits / sp.n_units * dq if dq is not None else None
        out.update(std_phy_mm2_per_stack=std, custom_phy_mm2_per_stack=CUSTOM["d2d_mm2_per_stack"][0])
    n = ignored_note(scn.chip, sp.mem_type)
    if n:
        out["notes"].append(n)
    if on:
        out.update(phy_pJ_factor=CUSTOM["phy_pJ_factor"][0], usd_per_GB_factor=CUSTOM["usd_per_GB_factor"][0],
                   src={k: CUSTOM[k][2] for k in ("d2d_mm2_per_stack", "phy_pJ_factor", "usd_per_GB_factor", "process")},
                   tags={k: CUSTOM[k][1] for k in ("d2d_mm2_per_stack", "phy_pJ_factor", "usd_per_GB_factor", "process")})
        out["notes"] += ["岸线：每栈仍占 11 mm（堆栈占地；D2D 8.5 mm 不宽于原 PHY 8 mm）——定制基础晶粒缩小 PHY 深度 / 面积，"
                         "不减少每栈所需边长「推导」；主晶粒变小时边长随之变短"] + list(NOT_MODELLED)
    return out
