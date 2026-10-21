"""Resource / area budgets as optional design constraints (0.49) — headroom against user-entered limits.

Nothing is built in: every limit and every density is ``None`` until the user enters one, and a budget with no entry
changes nothing.  Two kinds of entries:

  direct limits   quantities the model already knows exactly for the evaluated point — per card: SRAM MiB, SLC MiB,
                  MAC units (rows·cols·engines), peak bf16 TFLOPS, DRAM capacity GiB; per replica: cards; and the
                  average power per card (from the user's energy table, core/energy.py)
  area proxy      die area per card ≈ SRAM MiB · mm²/MiB + SLC MiB · mm²/MiB + MACs / 1024 · mm²/kMAC + fixed mm²
                  (I/O, PHY, vector unit, NoC, …).  The densities are the user's 「假设」 (a process / library figure
                  they trust).  0.149: optional cited per-node presets (DENSITY_PRESETS, each value labelled 来源 / 推导 /
                  二手 / 假设 with its source) fill unset densities; interface PHY area (DRAM pins, D2D units,
                  C2C GB/s) is added by full_area() when those densities are set.  Checked against a per-card die limit and a
                  replica silicon limit (cards × die).  0.153: optional ``usd_per_kWh`` + ``tdp_W_card`` turn insight ΔJ (or
constant-TDP × time) into Δ$; electricity presets are labelled regional averages, never silent defaults.

Each limit gives value, limit, headroom = 1 − value / limit and ok = True / False / None.  ``None`` means the tool
cannot decide: an area density the estimate needs is missing, or the power is only a lower bound (energy table
incomplete) that is still under the limit.  A lower bound above the limit is a definite violation.
The batch / layout searches stay exact without the budget; the API flags rows that break it and ranks them after
the rows that fit (see api.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

_LIMITS = {   # key → (label, unit)
    "sram_mib": ("SRAM / 卡", "MiB"), "slc_mib": ("SLC / 卡", "MiB"), "macs": ("MAC 单元 / 卡", "MAC"),
    "tflops": ("峰值 bf16 / 卡", "TFLOPS"), "dram_GiB": ("DRAM 容量 / 卡", "GiB"), "cards": ("卡数 / 副本", "卡"),
    "power_W_card": ("平均功耗 / 卡", "W"), "die_mm2": ("面积代理 / 卡", "mm²"),
    "system_mm2": ("硅面积代理 / 副本", "mm²"),
}
_DENS = ("mm2_per_mib_sram", "mm2_per_mib_slc", "mm2_per_kmac", "mm2_fixed")
_PHY_DENS = ("mm2_per_dq_hbm", "mm2_per_dq_lpddr", "mm2_per_d2d_unit", "mm2_per_GBps_c2c",
             "uncore_frac", "mm2_host_io", "mm2_per_unit_mc")      # 0.213: uncore goes through full_area too

# 0.149: cited density presets per process node.  Each value: (number, status, source).  status 「来源」 = read from
# the cited text; 「推导」 = arithmetic on cited figures (the scaling step itself is an assumption); 「二手」 = number
# reported by a secondary source, original closed; 「假设」 = no public figure found.  All are upper/lower bounds as
# noted, and every field stays overridable (a user-entered budget field wins over the preset).
_TPUV1 = "Jouppi et al., ISCA 2017 (arXiv 1704.04760) §2 / Fig. 2; die < 330 mm² per Jouppi et al., ISCA 2021 Table 1"
_TPUV4I = "Jouppi et al., 'Ten Lessons From Three Generations Shaped Google's TPUv4i', ISCA 2021, Fig. 6 / §7.A / Table 1"
_BITCELL = ("TSMC N7 HD 6T bitcell 0.027 µm² (Wu et al., IEDM 2016); N5 0.021 µm² (Yeap et al., IEDM 2019; "
            "WikiChip / SemiAnalysis); N5 logic density 1.8× N7 (TSMC, WikiChip 'TSMC Details 5 nm')")
_PHY = {
    # 0.185: was 14/1024 「二手/假设」 (H100 die-shot estimates, unverified).  Now Samsung Hot Chips 2026 (S. Han, via
    # Tom's Hardware 2026-08 report): on the 11 × 12.8 mm HBM4 base die the conventional PHY occupies "more than 8 × 4 mm"
    # for the 2048-DQ interface (SF4), and the XPU-side interface mirrors it ("the matching interface on the XPU shrinks
    # too").  Host-side = memory-side is the 推导 step; ">" makes it a lower bound.
    "mm2_per_dq_hbm": (8 * 4 / 2048, "推导/二手", "Samsung Hot Chips 2026 HBM4 base die: conventional HBM PHY > 8 × 4 mm "
                       "for 2048 DQ (SF4), reported by Tom's Hardware / ServeTheHome → ≥ 0.0156 mm²/DQ; host-side PHY taken "
                       "equal to the memory-side one (推导, lower bound). Old: ≈14 mm²/1024-bit HBM3 from H100 die-shot "
                       "estimates (0.0137)"),
    # 0.184: was 0.03 「假设」 (no public LPDDR5X PHY area).  Now the Samsung SF2 LPDDR6 PHY (ISSCC 2026 paper 37.3, as
    # reported by SemiAnalysis 2026-04-15: 0.695 mm², 2.32 mm shoreline, 55.3 GB/s/mm² ⇒ 38.4 GB/s = one x24 channel
    # at 14.4 Gb/s incl. the 32/36 BL24 payload) → 0.695 / 24 DQ.  Includes the channel's CA / clock overhead.
    "mm2_per_dq_lpddr": (0.695 / 24, "推导/二手", "Samsung SF2 LPDDR6 PHY 0.695 mm² for one x24 channel (ISSCC 2026 37.3 via "
                         "SemiAnalysis 'ISSCC 2026 roundup'; x24 inferred from its 55.3 GB/s/mm² × 0.695 mm² = 38.4 GB/s "
                         "= x24 @ 14.4 Gb/s payload) → 0.0290 mm²/DQ; 2 nm LPDDR6 — LPDDR5X / older nodes are larger 「假设」"),
    "mm2_per_d2d_unit": (0.3888 * 1.043, "来源", "UCIe x64 Advanced module 388.8 µm × 1043 µm (45 µm bump pitch), Hot Chips "
                         "2023 UCIe tutorial 'Electrical, Form Factor and Compliance' table; UCIe-S x16 is 571.5 × 1540 µm "
                         "= 0.88 mm² at 24/32 GT/s — override for UCIe-S / BoW"),
    "mm2_per_GBps_c2c": (1.806 * 0.825 / 4 / 14.0, "二手", "quad 112 Gb/s DSP SerDes 1806 × 825 µm in 5 nm (Varzaghani et "
                         "al., VLSI 2022; area as reported secondary, paper closed) → 0.37 mm² per lane ÷ 14 GB/s per lane per "
                         "direction"),
}
# 0.162: vector-lane and dequant-unit densities for insight sizing (sizing only — not added to the die-area proxy, whose
# mm2_fixed already covers "vector unit").  No public NPU floorplan splits out the vector unit (TPUv4i ISCA 2021 Fig. 6
# gives CMEM 28 % and MXUs 11 % only), so they are derived per node from that node's MAC density × an operator-area
# ratio from Horowitz's 45 nm table (ISSCC 2014 plenary "Computing's energy problem", area column as reproduced in
# Sze et al. / MIT EEMS and NYCU AAML slides — 二手): 16b FP mult 1640 µm², 32b FP add 4184, 32b FP mult 7700, 8b add 36.
#   bf16 MAC (16b FP mult + 32b FP add)        = 5824 µm²   (the unit the MXU densities count)
#   vector lane (32b FP mult + 32b FP add)     = 11884 µm²  → ×2.0405 of a MAC   (elementwise fp32 FMA datapath)
#   dequant elem/cycle (16b FP mult + 8b add)  = 1676 µm²   → ×0.2878 of a MAC   (int → bf16 × scale + zero point)
# Datapath operators only (no register file / control / shuffle network): lower bounds for a real lane.
_HZ = "Horowitz ISSCC 2014 45 nm operator areas (二手: area table via Sze et al. / MIT EEMS slides)"
_LANE_R = (7700 + 4184) / (1640 + 4184)
_DQ_R = (1640 + 36) / (1640 + 4184)


def _vec(kmac: float, node_src: str) -> dict:
    return {"mm2_per_klane": (kmac * _LANE_R, "推导/二手", f"{node_src} MAC density × (FP32 mult + FP32 add)/(FP16 mult + FP32 add)"
                              f" = 11884/5824 µm² per lane, {_HZ}; datapath only (lower bound)"),
            "mm2_per_kdq": (kmac * _DQ_R, "推导/二手", f"{node_src} MAC density × (FP16 mult + 8b add)/(FP16 mult + FP32 add)"
                            f" = 1676/5824 µm² per dequant element/cycle, {_HZ}; datapath only (lower bound)")}


_UNCORE = {
    # 0.213: TPUv1 floor plan (Jouppi et al., ISCA 2017, Fig. 2 caption): data buffers 37 %, compute 30 %, I/O 10 %,
    # control 2 % — the remaining 21 % is unlabelled (routing / white space; NeuroMeter, MICRO 2021, also leaves ≈ 21 %
    # of TPUv1 "unknown").  I/O is modelled separately (DRAM PHY pins + host IO), so uncore = control 2 % + 21 %.
    "uncore_frac": (0.23, "来源/推导", "TPUv1 floor plan (Jouppi et al., ISCA 2017 Fig. 2): control 2 % + unlabelled "
                    "21 % (100 − 37 buffers − 30 compute − 10 I/O − 2 control); one design's share used for all 「假设」"),
    "mm2_host_io": (0.018 * 331, "二手", "NeuroMeter (Shao et al., MICRO 2021) TPUv1 die-shot breakdown: PCIe interface "
                    "1.8 % of a < 331 mm² 28 nm die ≈ 6.0 mm²; not scaled by node (analog PHY shrinks little) 「假设」"),
}


DENSITY_PRESETS: dict[str, dict] = {
    "n28_tpuv1": {"label": "28 nm（TPUv1 平面图）", "fields": {
        "mm2_per_mib_sram": (330 / 3 / 24, "来源", _TPUV1 + ": 24 MiB Unified Buffer ≈ 1/3 of the die (upper bound)"),
        "mm2_per_mib_slc": (330 / 3 / 24, "来源", "= SRAM (same macro class) 「假设」 on " + _TPUV1),
        "mm2_per_kmac": (330 / 4 / 64, "来源", _TPUV1 + ": 256×256 int8 MXU = 1/4 of the die (upper bound; int8 MAC — a "
                         "bf16 MAC unit is larger, so for a bf16 array this is optimistic)"),
        **_vec(330 / 4 / 64, "TPUv1 (int8 MAC — ratio is for bf16, so optimistic)"),
        **_PHY, **_UNCORE}},
    "n7_tpuv4i": {"label": "7 nm（TPUv4i 平面图）", "fields": {
        "mm2_per_mib_sram": (400 * 0.28 / 128, "来源", _TPUV4I + ": 128 MiB CMEM = 28 % of a < 400 mm² die (upper bound)"),
        "mm2_per_mib_slc": (400 * 0.28 / 128, "来源", "= CMEM density (" + _TPUV4I + ")"),
        "mm2_per_kmac": (400 * 0.11 / 64, "来源", _TPUV4I + ": 4 × 128×128 bf16 MXUs = 11 % of a < 400 mm² die (upper bound)"),
        **_vec(400 * 0.11 / 64, "TPUv4i"),
        **_PHY, **_UNCORE}},
    "n5_derived": {"label": "5 nm / N3E（由 7 nm 推导）", "fields": {
        "mm2_per_mib_sram": (400 * 0.28 / 128 * 0.021 / 0.027, "推导", "TPUv4i CMEM × bitcell 0.021/0.027 (paper says 128 MB; taken as MiB, ≤ 5 %); " + _BITCELL
                             + "; array-only lower bound 31.8 Mib/mm² = 0.25 mm²/MiB (WikiChip, IEDM 2022 N3E)"),
        "mm2_per_mib_slc": (400 * 0.28 / 128 * 0.021 / 0.027, "推导", "= SRAM density (TPUv4i CMEM × bitcell ratio)"),
        "mm2_per_kmac": (400 * 0.11 / 64 / 1.8, "推导", "TPUv4i MXU ÷ 1.8 logic density; " + _BITCELL),
        **_vec(400 * 0.11 / 64 / 1.8, "N5 (TPUv4i ÷ 1.8)"),
        **_PHY, **_UNCORE}},
}


# 0.153: labelled regional electricity averages for insight $/unit.  Never applied unless the user picks a preset or
# types usd_per_kWh.  status labels match DENSITY_PRESETS.  FX for non-USD sources is itself 「假设」.
_EIA_2024 = ("U.S. EIA Electric Power Annual Table 2.10 / Electric Power Monthly Table 5.3: Average Price of "
             "Electricity to Ultimate Customers by End-Use Sector, 2024 (cents/kWh → $/kWh)")
_CN_PRICE = ("中国价格信息网（chinaprice.cn）2024-09-25：36 大中城市普通工业用电（35 kV 及以上）总平均价 0.62 元/kWh；"
             "USD 换算用中间价 7.1 CNY/USD「假设」（非牌价锁定）")
ELECTRICITY_PRESETS: dict[str, dict] = {
    "eia_us_industrial_2024": {
        "label": "美国工业用电均价 2024（EIA）",
        "fields": {
            "usd_per_kWh": (0.0813, "来源", _EIA_2024 + ": industrial 8.13 ¢/kWh → 0.0813 $/kWh"),
        },
    },
    "eia_us_commercial_2024": {
        "label": "美国商业用电均价 2024（EIA）",
        "fields": {
            "usd_per_kWh": (0.1275, "来源", _EIA_2024 + ": commercial 12.75 ¢/kWh → 0.1275 $/kWh"),
        },
    },
    "cn_36city_industrial_2024_09": {
        "label": "中国 36 城工业用电 2024-09（价信网）",
        "fields": {
            "usd_per_kWh": (0.62 / 7.1, "二手/假设", _CN_PRICE + " → ≈ 0.0873 $/kWh"),
        },
    },
}


# 0.154: PUE presets only (no published card price / amortisation figure is adopted — those stay user inputs).
_UPTIME = "Uptime Institute Global Data Center Survey 2024 (n=526), Figure 2: industry average annual PUE"
TCO_PRESETS: dict[str, dict] = {
    "uptime_2024_avg": {"label": "PUE 行业均值 2024（Uptime）", "fields": {
        "pue": (1.56, "来源", _UPTIME + " 1.56 (per-site average, incl. legacy sites)")}},
    "uptime_2023_capw": {"label": "PUE 按容量加权 2023（Uptime）", "fields": {
        "pue": (1.47, "来源", "Uptime Institute Journal, 'Large data centers are mostly more efficient, analysis "
                "confirms': 2023 capacity-weighted PUE 1.47 (per-site 1.58)")}},
}


def apply_tco_preset(d: dict) -> dict:
    """Body ``budget`` with ``tco_preset``: fill only unset fields (user values win)."""
    d = dict(d)
    pid = d.pop("tco_preset", None)
    if pid is None:
        return d
    if pid not in TCO_PRESETS:
        raise ValueError(f"budget.tco_preset must be one of {sorted(TCO_PRESETS)}")
    for k, (v, _, _) in TCO_PRESETS[pid]["fields"].items():
        if d.get(k) is None:
            d[k] = v
    return d


def apply_electricity_preset(d: dict) -> dict:
    """Body ``budget`` with ``electricity_preset``: fill usd_per_kWh only if the user left it unset."""
    d = dict(d)
    pid = d.pop("electricity_preset", None)
    if pid is None:
        return d
    if pid not in ELECTRICITY_PRESETS:
        raise ValueError(f"budget.electricity_preset must be one of {sorted(ELECTRICITY_PRESETS)}")
    for k, (v, _, _) in ELECTRICITY_PRESETS[pid]["fields"].items():
        if d.get(k) is None:
            d[k] = v
    return d


def apply_preset(d: dict) -> dict:
    """Body ``budget`` with ``density_preset``: fill every density the user left unset from the preset."""
    d = dict(d)
    pid = d.pop("density_preset", None)
    if pid is None:
        return d
    if pid not in DENSITY_PRESETS:
        raise ValueError(f"budget.density_preset must be one of {sorted(DENSITY_PRESETS)}")
    for k, (v, _, _) in DENSITY_PRESETS[pid]["fields"].items():
        if d.get(k) is None:
            d[k] = v
    return d


def phy_area(scn, b: Budget) -> dict:
    """0.149: interface PHY area per card from the PHY densities that are set: DRAM pins, D2D units, C2C GB/s."""
    from .. import mem_catalog
    terms, notes = {}, []
    spec = mem_catalog.parse_mem_id(scn.mem_id)
    hbm = "HBM" in spec.mem_type.upper()
    dq = b.mm2_per_dq_hbm if hbm else b.mm2_per_dq_lpddr
    if dq is not None:
        terms["dram_phy"] = spec.bus_bits * dq
    if scn.d2d_enabled and scn.package_cards > 1 and b.mm2_per_d2d_unit is not None:
        if scn.d2d_std == "custom":
            notes.append("d2d_std = custom：单元数未知，D2D PHY 未计")
        else:
            terms["d2d_phy"] = scn.d2d_units * b.mm2_per_d2d_unit
    pk = scn.package_cards if scn.d2d_enabled else 1
    if b.mm2_per_GBps_c2c is not None:
        if scn.layout.cards > pk:          # 0.150: the scale-up link only leaves the package when cards > dies/package
            terms["c2c_phy"] = scn.link.GBps * b.mm2_per_GBps_c2c
        elif scn.layout.cards > 1:
            notes.append("全部卡在同一封装内（D2D）：卡间链路 PHY 不计，避免与 D2D 重复")
    if scn.node_cards:
        notes.append("跨节点网卡（net）PHY 不在面积代理内")
    if terms and b.mm2_fixed is not None:
        notes.append("已单独计入接口 PHY：固定面积 mm2_fixed 应不含这些 PHY（否则重复计算）")
    return {"mm2": sum(terms.values()), "terms": terms, "notes": notes}


# 0.212: die shoreline for HBM PHYs.  Each stack faces the host die with its 11 mm side (HBM4 base die 11 × 12.8 mm,
# Samsung Hot Chips 2026 via Tom's Hardware 「二手」; HBM3E stack ≈ 11 mm wide 「二手」), so one stack takes ≥ 11 mm of
# die edge.  Shipping packages put HBM on the two long sides and keep the other two for D2D / PCIe / NVLink: GH100
# (6 sites, 3 per long side, 814 mm²), B200 (4 per die on 2 dies, 8 per package), MI300X (8 stacks on the long sides of
# 4 IODs) 「二手」 — so 2 sides is the typical budget, 4 sides the geometric limit 「假设」.  The die is taken as square.
HBM_STACK_EDGE_MM = 11.0
HBM_SIDES_TYPICAL, HBM_SIDES_MAX = 2, 4
KNOWN_MAX_STACKS = {"B200 (CoWoS-L, 2 dies)": 8, "MI300X (CoWoS-S, 4 IODs)": 8}    # vendor specs 「来源」


def shoreline_check(scn, die_mm2: float | None) -> dict | None:
    """HBM stacks vs the die edge: None for non-HBM memory or an unknown die area."""
    from .. import mem_catalog
    spec = mem_catalog.parse_mem_id(scn.mem_id)
    if "HBM" not in spec.mem_type.upper() or not die_mm2:
        return None
    side = die_mm2 ** 0.5
    need = spec.n_units * HBM_STACK_EDGE_MM
    # 0.214: a die in a multi-die package gives one edge to the D2D link (B200: the inner edge to NV-HBI)
    d2d_edges = 1 if (getattr(scn, "d2d_enabled", False) and getattr(scn, "package_cards", 1) > 1) else 0
    sides_max = HBM_SIDES_MAX - d2d_edges
    typ, mx = min(HBM_SIDES_TYPICAL, sides_max) * side, sides_max * side
    status = "ok" if need <= typ else ("tight" if need <= mx else "over")
    out = {"stacks": spec.n_units, "need_mm": need, "edge_typical_mm": typ, "edge_max_mm": mx, "die_mm2": die_mm2,
           "status": status, "fit_typical": int(typ // HBM_STACK_EDGE_MM), "fit_max": int(mx // HBM_STACK_EDGE_MM),
           "known_max_stacks": KNOWN_MAX_STACKS, "d2d_edges": d2d_edges, "sides_max": sides_max}
    if status != "ok":
        out["warning"] = (f"HBM 岸线：{spec.n_units} 堆栈需 {need:.0f} mm 晶粒边缘，方形晶粒 {die_mm2:.0f} mm² 两长边仅 "
                          f"{typ:.0f} mm（{'四' if sides_max == 4 else '三'}边 {mx:.0f} mm）" + ("；四边也放不下 — 需多晶粒封装（如 B200 2 晶粒 8 堆栈）"
                                                              if status == "over" else "；需占用第三、四边")
                          + f" · HBM shoreline: {spec.n_units} stacks need {need:.0f} mm of die edge")
    return out


def noc_router_mm2(chip, b) -> float | None:
    """0.307: area of the chip's NoC routers (one per engine, mesh / ring, link width set): E × the Park DAC 2012
    reference router scaled to the link width (energy.noc_router_scale) × the node factor (the user's MAC density ÷
    Horowitz's 45 nm bf16 MAC, 「推导/二手」).  None without mm2_per_kmac or without routers to price."""
    from .energy import NOC_MAC_UM2_45, noc_router_scale
    sc = noc_router_scale(chip.noc_link_Bpc, chip.noc_topology)
    if sc is None or chip.engines <= 1 or b.mm2_per_kmac is None:
        return None
    return chip.engines * sc["um2_45"] * 1e-6 * (b.mm2_per_kmac / 1024 * 1e6 / NOC_MAC_UM2_45)


def full_area(scn, b: Budget, noc_routers: bool = False) -> dict:
    """Die area proxy incl. interface PHYs (0.149): area_estimate(chip) + phy_area(scenario).
    0.307 ``noc_routers`` (the energy DSE's NoC-width axis): + the NoC routers, counted once — after the uncore share
    (uncore_frac is TPUv1's control + unlabelled share; TPUv1 has no NoC: its blocks are joined by dedicated 256-byte
    paths, so the share holds wiring / white space, not routers; the NoC's link wires play the role of those paths
    and stay inside the share), and not at all when mm2_fixed is given (it already means "everything else, incl. NoC")."""
    a = area_estimate(scn.chip, b)
    p = phy_area(scn, b)
    terms = {**a["terms"], **p["terms"]}
    notes = list(p["notes"])
    mm2 = None if a["mm2"] is None else a["mm2"] + p["mm2"]
    # 0.213: uncore — only when the user gave no mm2_fixed (which already means "everything else")
    if b.mm2_fixed is None and mm2 is not None:
        if b.mm2_host_io is not None:
            terms["host_io"] = b.mm2_host_io
            mm2 += b.mm2_host_io
        if b.mm2_per_unit_mc is not None:
            from .. import mem_catalog
            spec = mem_catalog.parse_mem_id(scn.mem_id)
            terms["mem_ctrl"] = spec.n_units * b.mm2_per_unit_mc
            mm2 += terms["mem_ctrl"]
        if b.uncore_frac:
            u = mm2 * b.uncore_frac / (1 - b.uncore_frac)      # share of the final die
            terms["uncore"] = u
            mm2 += u
    elif b.mm2_fixed is not None and (b.uncore_frac or b.mm2_host_io or b.mm2_per_unit_mc):
        notes.append("已填固定面积 mm2_fixed：uncore 项不再另计（避免重复）")
    if noc_routers and mm2 is not None:
        nr = noc_router_mm2(scn.chip, b)
        if nr is not None:
            if b.mm2_fixed is None:
                terms["noc_routers"] = nr
                mm2 += nr
            else:
                notes.append("已填固定面积 mm2_fixed（含 NoC）：路由器面积不再另计（0.307，避免重复）")
    return {**a, "mm2": mm2, "terms": terms, "phy_notes": notes}


@dataclass(frozen=True)
class Budget:
    """User-entered limits and area densities (all ``None`` = not set)."""
    sram_mib: float | None = None
    slc_mib: float | None = None
    macs: float | None = None
    tflops: float | None = None
    dram_GiB: float | None = None
    cards: int | None = None
    power_W_card: float | None = None
    die_mm2: float | None = None
    system_mm2: float | None = None
    mm2_per_mib_sram: float | None = None     # 「假设」 SRAM macro + periphery density
    mm2_per_mib_slc: float | None = None      # 「假设」
    mm2_per_kmac: float | None = None         # 「假设」 per 1024 MAC units incl. local registers / wiring
    mm2_fixed: float | None = None            # 「假设」 everything else (I/O, PHY, vector unit, NoC)
    # 0.162: insight sizing only (not in the die-area proxy): per 1024 vector lanes / per 1024 dequant elements per cycle
    mm2_per_klane: float | None = None
    mm2_per_kdq: float | None = None
    # 0.149: interface PHY area (optional; None = not counted).  Presets in DENSITY_PRESETS carry the sources.
    mm2_per_dq_hbm: float | None = None       # per HBM data pin (bus bit) of the DRAM interface
    mm2_per_dq_lpddr: float | None = None     # per data pin of any non-HBM DRAM interface (LPDDR / DDR / GDDR)
    mm2_per_d2d_unit: float | None = None     # per D2D unit (UCIe module / BoW slice) on the die
    mm2_per_GBps_c2c: float | None = None     # per GB/s (per direction) of card-to-card SerDes link
    # 0.213: uncore (not counted elsewhere): control / NoC / routing / white space as a share of the die, host IO
    # (PCIe) as a fixed area, memory controllers per DRAM channel.  Not added when mm2_fixed is set (it already
    # covers "everything else").
    uncore_frac: float | None = None          # share of the final die, 0 ≤ f < 1
    mm2_host_io: float | None = None          # host interface (PCIe x16 class) incl. its PHY
    mm2_per_unit_mc: float | None = None   # memory controller per DRAM unit (HBM stack / DRAM package; no public figure: no preset)
    # 0.153: absolute operating cost (optional; None = insight does not rank by $).
    usd_per_kWh: float | None = None           # electricity rate; presets in ELECTRICITY_PRESETS
    tdp_W_card: float | None = None            # nameplate / design TDP for J≈W·t when energy table has no J
    # 0.154: optional TCO inputs (None = no CapEx+OpEx ranking).  Never defaulted silently.
    card_usd: float | None = None              # purchase price per card ($)
    pue: float | None = None                   # facility PUE (≥ 1); presets in TCO_PRESETS
    amort_years: float | None = None           # straight-line amortisation of card_usd

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1e9:
                raise ValueError(f"budget.{f.name} must be a finite number in [0, 1e9) or null")
            if f.name in _LIMITS and v <= 0:
                raise ValueError(f"budget.{f.name} must be > 0")
        if self.uncore_frac is not None and not self.uncore_frac < 1:
            raise ValueError("budget.uncore_frac must be < 1")
        if self.pue is not None and self.pue < 1:
            raise ValueError("budget.pue must be ≥ 1")
        if self.amort_years is not None and self.amort_years <= 0:
            raise ValueError("budget.amort_years must be > 0")
        if self.cards is not None and float(self.cards) != int(self.cards):
            raise ValueError("budget.cards must be an integer")

    @property
    def provided(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    @staticmethod
    def from_dict(d: dict | None) -> "Budget":
        if d is None:
            return Budget()
        if not isinstance(d, dict):
            raise ValueError("budget: expected an object")
        bad = set(d) - {f.name for f in fields(Budget)}
        if bad:
            raise ValueError(f"budget: unknown keys {sorted(bad)}")
        return Budget(**d)


def area_estimate(chip, b: Budget) -> dict:
    """Per-card die-area proxy from the user's densities: {mm2 | None, terms, missing}."""
    terms, missing = {}, []
    # 0.211: the dedicated recurrent-state buffer (Chip.state_sram_mib) is the same SRAM macro class → same density
    for name, qty, dens in (("sram", chip.sram_mib, b.mm2_per_mib_sram),
                            ("state_sram", getattr(chip, "state_sram_mib", 0.0), b.mm2_per_mib_sram),
                            ("slc", chip.slc_mib, b.mm2_per_mib_slc),
                            ("mac", chip.macs / 1024, b.mm2_per_kmac)):
        if qty <= 0:
            continue
        if dens is None:
            missing.append({"sram": "mm2_per_mib_sram", "state_sram": "mm2_per_mib_sram", "slc": "mm2_per_mib_slc", "mac": "mm2_per_kmac"}[name])
        else:
            terms[name] = qty * dens
    if b.mm2_fixed is not None:
        terms["fixed"] = b.mm2_fixed
    return {"mm2": None if missing else sum(terms.values()), "terms": terms, "missing": missing,
            "fixed_given": b.mm2_fixed is not None}


_AUTO = object()


def _item(key: str, value, limit, ok=_AUTO, note: str = "") -> dict:
    lab, unit = _LIMITS[key]
    if ok is _AUTO:
        ok = None if value is None else value <= limit * (1 + 1e-12)
    return {"key": key, "label": lab, "unit": unit, "value": value, "limit": limit,
            "headroom": (1.0 - value / limit) if value is not None else None, "ok": ok, "note": note}


def budget_report(r, b: Budget, energy: dict | None = None) -> dict:
    """Headroom of one evaluated result against the budget.  ``energy``: energy_report(r, table) for the power limit."""
    ch, lay = r.scenario.chip, r.scenario.layout
    items = []
    # 0.211: sram_mib = on-die SRAM incl. the dedicated state buffer
    direct = {"sram_mib": ch.sram_mib + getattr(ch, "state_sram_mib", 0.0), "slc_mib": ch.slc_mib, "macs": float(ch.macs),
              "tflops": ch.peak_tflops_bf16, "cards": float(lay.cards),
              "dram_GiB": r.stages[0].mem.dram_cap / 2**30 if r.stages else None}
    for k, v in direct.items():
        lim = getattr(b, k)
        if lim is not None and v is not None:
            items.append(_item(k, v, lim))
    if b.power_W_card is not None:
        e = energy or {}
        if "avg_W_per_card" not in e:
            items.append(_item("power_W_card", None, b.power_W_card, None, "需要能耗表（左侧「能耗」组）才能算功耗"))
        else:
            w = e["avg_W_per_card"]
            cnt = e.get("counts_per_unit", {})
            key = {"pJ_mac": "mac", "pJ_vec": "vec", "pJ_bit_sram": "sram", "pJ_bit_dram": "dram", "pJ_bit_link": "link",
                   "pJ_bit_slc": "slc", "pJ_bit_d2d": "d2d", "pJ_bit_net": "net", "idle_W": "idle_card_s",
                   "tdp_W": "idle_card_s"}
            e = {**e, "missing": [m for m in e.get("missing", []) if cnt.get(key.get(m, ""), 1) > 0]}  # zero-count entries don't matter
            if e["missing"]:
                ok = False if w > b.power_W_card else None
                items.append(_item("power_W_card", w, b.power_W_card, ok,
                                   f"能耗表未填 {', '.join(e['missing'])}：{w:.3g} W 只是下界"))
            else:
                items.append(_item("power_W_card", w, b.power_W_card))
    area = area_estimate(ch, b) if any(getattr(b, k) is not None for k in _DENS + _PHY_DENS + ("die_mm2", "system_mm2")) else None
    if area is not None and any(getattr(b, k) is not None for k in _PHY_DENS):      # 0.149: interface PHYs
        area = full_area(r.scenario, b)
    for k, mult in (("die_mm2", 1), ("system_mm2", lay.cards)):
        lim = getattr(b, k)
        if lim is None:
            continue
        if area["mm2"] is None:
            # 0.61.2: the terms that are known already bound the area from below — above the limit is a definite
            # violation (as for the power lower bound and the missing fixed area); otherwise undecidable
            lb = sum(area["terms"].values()) * mult
            if lb > lim:
                items.append(_item(k, lb, lim, False, f"面积代理缺密度 {', '.join(area['missing'])}「假设」："
                                                      f"已知项 {lb:.4g} mm² 只是下界，已超限"))
            else:
                items.append(_item(k, None, lim, None, f"面积代理缺密度 {', '.join(area['missing'])}「假设」"))
        else:
            v = area["mm2"] * mult
            if area["fixed_given"]:
                items.append(_item(k, v, lim))
            else:       # without the fixed part the proxy is a lower bound (enter 0 if the densities already cover it)
                items.append(_item(k, v, lim, False if v > lim else None,
                                   "未填固定面积 mm2_fixed：估计只是下界（已计入密度时填 0）"))
    oks = [i["ok"] for i in items]
    ok = False if False in oks else (None if None in oks else True)
    # 0.212: HBM shoreline on the die limit (die_mm2) when given, else the area proxy — a warning, not a budget item
    shore = shoreline_check(r.scenario, b.die_mm2 or (area or {}).get("mm2"))
    return {"ok": ok, "items": items, "area": area, "provided": b.provided, "shoreline": shore,
            "violations": [i["key"] for i in items if i["ok"] is False],
            "basis": "用户输入的限额与面积密度「假设」；工具不内置工艺库，只报告余量"}
