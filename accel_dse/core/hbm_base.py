"""0.324: custom HBM base die — a design option (Chip.hbm_base "custom"; "std" = the JEDEC HBM4 interface, ≤ 0.323).

From HBM4 on the base die is made on a logic process (TSMC N12 for SK hynix HBM4, Samsung SF4, Micron in-house CMOS;
custom HBM4E / NVHBM on TSMC N3-class).  A *custom* base die takes the memory controller off the host and replaces
the 2048-DQ HBM PHY with a narrow die-to-die link (Samsung Hot Chips 2026; Marvell custom HBM; NVIDIA NVHBM).  What
the model changes, each number labelled:

  host PHY area / stack   std  2048 DQ × mm2_per_dq_hbm (Samsung HC 2026: conventional PHY > 8 × 4 mm 「二手」)
                          custom (0.325) ceil(stack GB/s ÷ 384) UCIe-A x64 modules × 0.3888 × 1.043 mm (d2d_catalog,
                          UCIe geometry 「来源」) 「推导」 — HBM4 10 Gb/s: 7 modules ≈ 2.84 mm² (0.324: 12.75, Samsung's
                          base-die block taken for the host side; that is the memory side)
  memory controller       moved to the base die: the host term mm2_per_unit_mc / mm2_per_hbm_mc drops out (0.328: on
                          by default in n5_derived for HBM4 / 4E at 7.15 mm²; 0.327:
                          opt-in mc_preset 7.15 / 14.3 mm² per stack = Samsung HC 2026 "about 5-10% of the SoC
                          area" 「二手」 × its 26 × 33 mm, 6-stack reference xPU 「推导」; budget.MC_PRESETS)
  interface energy / bit  (0.325) both ends: std host PHY 0.29 「来源」 + memory-side PHY in the device pJ/bit (taken
                          = host PHY, 「推导」) → custom UCIe-A link 0.25 pJ/bit split half host / half base die 「推导」;
                          ratio 0.43, inside Marvell's −45 … −75 % 「二手」 (0.324: host PHY × 0.55, memory side unchanged)
  memory $/GB             × 1.10 — custom HBM4E premium "5 % to 10 % overhead tops" (analyst speculation, EE Times
                          part 1) 「二手（推测）」; the upper end is the default
  shoreline / stack       unchanged on a silicon interposer: a short-reach link needs the stack adjacent to the host
                          edge, so the stack's 11 mm footprint sets the edge (0.325 reason; Eliyan: 2 stacks per 25 mm
                          on silicon, 3 on organic with a 20 mm-reach narrow interface — another package, not modelled)
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
    # 0.325: the host-side D2D area is derived from the model's own UCIe-A geometry (hbm_base.host_d2d_mm2); the
    # Samsung 8.5 × 1.5 mm block is the *memory-side* (base-die) figure and is kept only as a reference
    "base_die_d2d_mm2": (8.5 * 1.5, "二手", _HC26 + ": custom HBM D2D block ≈ 8.5 × 1.5 mm on the 11 × 12.8 mm HBM4 base "
                         "die (conventional PHY > 8 × 4 mm; channel depth 5.5 → 2 mm) — memory side, reference only"),
    # 0.325: the interface energy is the D2D link (both ends) replacing both HBM PHY ends (hbm_base.iface_pJ)
    "link_std": (None, "推导", "d2d_catalog ucie-a-48 (UCIe-A x64 @ 48 GT/s, the catalog default): 384 GB/s per module, "
                 "0.3888 × 1.043 mm (UCIe 2.0 geometry 「来源」), 0.25 pJ/bit per link 「来源/假设」"),
    "usd_per_GB_factor": (1.10, "二手/假设", "custom HBM4E premium '5 % to 10 % overhead tops' (analyst speculation, EE "
                          "Times Part 1); HBM4 base die already 3–4 × a core die's cost (Herald Business via TrendForce "
                          "2026-08-31) — upper end taken (× 1.10)"),
    "process": ("TSMC N3P（C-HBM4E）/ N3（NVHBM）", "二手", "TSMC OIP 2025 via HardwareLUXX / TrendForce 2025-12-01 (C-HBM4E "
                "N3P, 0.75 V); NVIDIA NVHBM base dies on TSMC 3 nm (Seoul Economic Daily via TrendForce 2026-10-19)"),
    "std_process": ("TSMC N12（SK hynix HBM4）/ Samsung SF4 / Micron 自有 CMOS", "二手", "Chosun Daily via TrendForce "
                    "2026-03-20 (SK hynix TSMC 12 nm, Samsung 4 nm); Micron (EE Times Part 1: in-house CMOS base die)"),
}
D2D_STD = "ucie-a-48"               # 0.325: the UCIe-like link of a custom base die (d2d_catalog default)
# Vendor interface-power claims, cross-check only (0.324 used 0.55 on the host PHY alone): custom link ÷ (both HBM PHY
# ends) = 1 − saving.  Marvell HC 2025 (Kuemerle): "75 % lower HBM & main die I/O power" 「二手」; EE Times part 2:
# −45 … −70 % "depending on scenarios" 「二手」.
VENDOR_IFACE_RATIO = (0.25, 0.55)
USD_FACTOR_RANGE = (1.05, 1.10)     # +5 % … +10 % 「二手（推测）」
# 0.325: shoreline / stack — unchanged on a silicon interposer: short-reach links (UCIe-A ≤ 2 mm) need the stack
# immediately adjacent to the host edge, so the stack's 11 mm footprint sets the edge, not the link's beachfront
# (≈ 2.7 mm for 7 UCIe-A modules).  Eliyan (Electronic Design 2026): "on silicon substrates [stacks] must be placed
# immediately adjacent to the XPU, limiting the number of stacks to two per 25-mm die edge"; on organic substrate with
# a long-reach narrow interface (SPHBM4: 512 DQ, ≥ 90 µm bumps, 20 mm reach, JESD330-4) "three memory devices can be
# connected along the same edge" (≈ 8.3 mm per stack) 「二手」 — a different package (organic, no interposer): 0.326
# models it as Chip.hbm_pkg = "sphbm4" (core/hbm_pkg.py), mutually exclusive with a custom base die.
ORGANIC_STACKS_PER_25MM = (3, 2)
NOT_MODELLED = ("近存计算 / 卸载（Samsung 第 2 阶段、SK hynix StreamDQ 基础晶粒反量化）：无厂商公布的基础晶粒算力 / 能耗 / 功耗预算；"
                "且基础晶粒算力不提高每栈带宽（TSV 带宽 = 接口带宽，AMMA arXiv 2026「PNM alone does not increase per-cube "
                "bandwidth」）；上界「推导」：注意力卸载只省 KV 字节的接口能耗 ≤ KV 字节占比 × 整条链路能耗"
                "（1P qwen3-8b decode b64 custom：ctx 4K ≈ 3.4% J、32K ≈ 5.3% J，未扣基础晶粒计算能耗）——未建模",
                "内存控制器功耗：标准设计中主晶粒也未单列控制器功耗（无可分离的公开数字；SC'25 GPU 实测把控制 + 接口 + 片上互连合计 "
                "8.5–13.5 pJ/bit），故定制时无项可移；方向：若计入，定制主晶粒发热更低",
                "基础晶粒工艺能耗（N12 0.8 V → N3P 0.75 V）：基础晶粒在器件 pJ/bit 中的份额未公开，未建模",
                "D2D 接口功率密度更高（Samsung 以 Heat Path Block 降峰温 > 35 %）：热点仅报告（hotspot_W_mm2），不判定")


def stack_GBps(spec) -> float:
    """Peak GB/s of one stack (pins × MT/s ÷ 8)."""
    return spec.unit_width_bits * spec.rate_MTps / 8 / 1000.0


def host_d2d_units(spec) -> int:
    """0.325: UCIe-A x64 modules the host needs per stack — the stack's peak bandwidth in one direction (a read burst)."""
    import math
    from .d2d_catalog import D2D_STANDARDS
    return max(1, math.ceil(stack_GBps(spec) / D2D_STANDARDS[D2D_STD]["per_unit_GBps"] - 1e-9))


def host_d2d_mm2(spec) -> float:
    """0.325: host-side D2D PHY area per stack 「推导」 = modules × the UCIe-A module footprint (d2d_catalog geometry)."""
    from .d2d_catalog import d2d_phy_mm2
    return host_d2d_units(spec) * d2d_phy_mm2(D2D_STD)


def host_d2d_edge_mm(spec) -> float:
    from .d2d_catalog import d2d_edge_mm
    return host_d2d_units(spec) * d2d_edge_mm(D2D_STD)


def iface_pJ(table) -> dict | None:
    """0.325: per-bit interface energy with a custom base die, both ends.  std = host PHY (pJ_bit_phy_hbm, 「来源」) +
    the memory-side PHY inside the device pJ/bit, taken equal to the host PHY (a PHY pair, 「推导」); custom = the
    D2D link (d2d_catalog pJ_bit, booked once) split half host / half base die 「推导」.  None when the table has no
    PHY field (std then has no interface line either)."""
    from .d2d_catalog import D2D_STANDARDS
    phy = getattr(table, "pJ_bit_phy_hbm", None)
    if phy is None:
        return None
    link = D2D_STANDARDS[D2D_STD]["pJ_bit"]
    return {"host": link / 2, "dev_delta": link / 2 - phy, "link": link, "std_pair": 2 * phy,
            "ratio": link / (2 * phy)}


def std_equiv_delta(scn, b) -> float:
    """0.325: host-die mm² a standard interface would add back (std die − custom die, same budget): the card-level
    flux precedent (TDP ÷ logic-die area of shipped parts) counts its HBM PHY / controller ring in the area, so a
    custom die is judged on that definition (thermal.thermal_report card_area_mm2).  0 when the switch is off."""
    from dataclasses import replace
    from .. import mem_catalog
    if b is None or not custom_applies(scn.chip, mem_catalog.parse_mem_id(scn.mem_id).mem_type):
        return 0.0
    from .budget import full_area
    try:
        a = full_area(scn, b)["mm2"]
        s = full_area(scn.replace("chip", replace(scn.chip, hbm_base="std")), b)["mm2"]
    except Exception:           # noqa: BLE001 — densities unset: no area, no adjustment
        return 0.0
    return max(0.0, (s or 0.0) - (a or 0.0)) if a and s else 0.0


def custom_applies(chip, mem_type: str | None) -> bool:
    return getattr(chip, "hbm_base", "std") == "custom" and bool(mem_type) and mem_type.upper() in CUSTOM_TYPES


def ignored_note(chip, mem_type: str | None) -> str | None:
    if getattr(chip, "hbm_base", "std") == "custom" and not custom_applies(chip, mem_type):
        return f"hbm_base=custom 只对 HBM4 / HBM4E（逻辑工艺基础晶粒）有效：{mem_type or '该外存'} 按标准接口计"
    return None


_FLOOR_MEMO: dict = {}


def shore_floor_mm2(scn, die_mm2: float | None) -> float | None:
    """0.326: memoised on (scenario, die) — the energy DSE asks for the same floor from full_area and again from
    std_equiv_delta (180 bisections, 66 distinct, on a 216-point custom DSE); the bisection itself is unchanged."""
    try:
        key = (scn, die_mm2)
        hit = _FLOOR_MEMO.get(key, _FLOOR_MEMO)
    except TypeError:
        return _shore_floor_mm2(scn, die_mm2)
    if hit is not _FLOOR_MEMO:
        return hit
    r = _shore_floor_mm2(scn, die_mm2)
    if len(_FLOOR_MEMO) > 20000:
        _FLOOR_MEMO.clear()
    _FLOOR_MEMO[key] = r
    return r


def _shore_floor_mm2(scn, die_mm2: float | None) -> float | None:
    """0.325: the smallest die (mm², budget.shoreline_report geometry) whose shoreline is not "over" — the floor a
    custom-base-die host cannot shrink below while its stacks abut it (silicon interposer).  None when ``die_mm2`` is
    already not over, or the geometry is unknown."""
    from .budget import shoreline_report
    if not die_mm2:
        return None
    if shoreline_report(scn, die_mm2).get("status") != "over":
        return None
    lo, hi = die_mm2, die_mm2
    for _ in range(12):
        hi *= 1.5
        if shoreline_report(scn, hi).get("status") not in ("over", "unknown"):
            break
    else:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if shoreline_report(scn, mid).get("status") == "over":
            lo = mid
        else:
            hi = mid
    return hi


def report(scn, b=None, table=None, die_mm2=None) -> dict:
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
        out.update(std_phy_mm2_per_stack=std, custom_phy_mm2_per_stack=host_d2d_mm2(sp),
                   d2d_std=D2D_STD, d2d_units_per_stack=host_d2d_units(sp), d2d_edge_mm_per_stack=host_d2d_edge_mm(sp),
                   stack_GBps=stack_GBps(sp), base_die_d2d_mm2=CUSTOM["base_die_d2d_mm2"][0])
    n = ignored_note(scn.chip, sp.mem_type)
    if n:
        out["notes"].append(n)
    if on:
        ip = iface_pJ(table) if table is not None else None
        out.update(usd_per_GB_factor=CUSTOM["usd_per_GB_factor"][0], iface_pJ=ip,
                   std_equiv_delta_mm2=std_equiv_delta(scn, b),
                   src={k: CUSTOM[k][2] for k in ("base_die_d2d_mm2", "link_std", "usd_per_GB_factor", "process")},
                   tags={k: CUSTOM[k][1] for k in ("base_die_d2d_mm2", "link_std", "usd_per_GB_factor", "process")})
        # 0.327: the controller area that leaves the host (budget.mc_mm2_per_unit), or the sourced range when unset
        from .budget import MC_PRESETS, mc_mm2_per_unit
        mc = mc_mm2_per_unit(b, sp.mem_type)
        fixed = b is not None and getattr(b, "mm2_fixed", None) is not None
        out["mc_mm2_per_stack"] = mc
        out["mc_reclaim_mm2"] = out["stacks"] * mc if mc is not None and not fixed else None
        if mc is None and not fixed:
            lo, hi = (MC_PRESETS[k]["fields"]["mm2_per_hbm_mc"][0] for k in ("samsung_hc2026_lo", "samsung_hc2026_hi"))
            out["mc_reclaim_range_mm2"] = (out["stacks"] * lo, out["stacks"] * hi)
            out["notes"].append(f"内存控制器面积未计：Samsung HC 2026「约占 SoC 面积 5–10 %」「二手」→ 每栈 {lo:.2f}–{hi:.1f} mm²"
                                f"（其参考 xPU 26 × 33 mm、6 栈「推导」）；定制基础晶粒从主晶粒移走的 {out['stacks'] * lo:.0f}–"
                                f"{out['stacks'] * hi:.0f} mm² 未计入定制 / 标准比较（budget.mc_preset 打开）")
        fl = shore_floor_mm2(scn, die_mm2)
        out["shore_floor_mm2"] = fl
        pad = None
        if b is not None:
            from .budget import full_area
            try:
                pad = (full_area(scn, b).get("terms") or {}).get("shore_pad")
            except Exception:       # noqa: BLE001 — densities unset
                pad = None
        out["shore_pad_mm2"] = pad
        if pad:
            out["notes"].insert(0, f"岸线下限：逻辑面积低于紧贴 {out['stacks']} 栈所需边长，主晶粒保持在下限（+{pad:.0f} mm² 填充硅，"
                                   "计成本 / 良率 / 热，不计算力）「推导」")
        if fl:
            out["notes"].insert(0, f"主晶粒 {die_mm2:.0f} mm² 小于 {out['stacks']} 栈 + I/O 所需岸线下限 ≈ {fl:.0f} mm²（堆栈须紧贴，"
                                   "硅中介层）：按岸线判为不可建；实际设计会把晶粒保持在下限（多出的面积给算力 / SRAM，模型未计入）")
        if ip is not None:
            out["notes"].append(f"接口能耗：标准 = 主机 PHY {ip['std_pair'] / 2:g} + 器件侧 PHY {ip['std_pair'] / 2:g}（取对称「推导」）"
                                f"= {ip['std_pair']:g} pJ/bit；定制 = {D2D_STD} 链路 {ip['link']:g} pJ/bit（主机 / 基础晶粒各半「推导」），"
                                f"比 {ip['ratio']:.2f}——落在厂商口径 {VENDOR_IFACE_RATIO[0]:g}–{VENDOR_IFACE_RATIO[1]:g}"
                                "（Marvell HC 2025「主晶粒与 HBM I/O 功耗低 75 %」、EE Times −45…−70 %「二手」）内，作交叉检验")
        out["notes"] += ["岸线：每栈仍占 11 mm——硅中介层上短距 D2D（UCIe-A ≤ 2 mm）要求堆栈紧贴主晶粒边，边长由堆栈占地决定，"
                         f"不由链路宽度决定（{host_d2d_units(sp)} 个 UCIe-A 模块 ≈ {host_d2d_edge_mm(sp):.1f} mm）"
                         "「推导」；Eliyan：硅基板每 25 mm 边 2 栈、有机基板长距窄接口 3 栈「二手」（另一种封装：hbm_pkg=sphbm4，"
                         "0.326，与定制基础晶粒互斥）",
                         "热：卡级热流先例（出货件 TDP ÷ 逻辑晶粒面积）的面积含 HBM PHY / 控制器环，定制晶粒按同口径"
                         "（加回标准接口面积）判卡级热流；晶粒热流按实际面积（物理口径）"] + list(NOT_MODELLED)
    return out
