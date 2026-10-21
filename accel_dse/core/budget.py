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
import copy
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
    if getattr(scn.chip, "noc_clusters", 1) > 1:
        notes.append("片上 NoC 簇：簇间链路无 D2D PHY（0.309）")
    elif scn.d2d_enabled and scn.package_cards > 1 and b.mm2_per_d2d_unit is not None:
        from .d2d_catalog import d2d_phy_mm2
        if scn.d2d_std == "custom":
            notes.append("d2d_std = custom：单元数未知，D2D PHY 未计")
        elif scn.d2d_std.startswith("ucie-a"):      # the density field is the UCIe-A module (preset 「来源」)
            terms["d2d_phy"] = scn.d2d_units * b.mm2_per_d2d_unit
        elif d2d_phy_mm2(scn.d2d_std) is not None:   # 0.308: UCIe-S / BoW — their own published footprint
            terms["d2d_phy"] = scn.d2d_units * d2d_phy_mm2(scn.d2d_std)
            notes.append(f"D2D PHY 按 {scn.d2d_std} 自身单元尺寸计（d2d_catalog.D2D_PHY_GEOM，0.308）")
        else:
            notes.append(f"{scn.d2d_std}：无公开 PHY 尺寸，D2D PHY 未计")
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
# Samsung Hot Chips 2026 via Tom's Hardware 「二手」; HBM3E stack 11 × 11 × 0.72 mm, Micron HBM3E product brief
# 「来源」 — 0.313), so one stack takes ≥ 11 mm of
# die edge.  Shipping packages put HBM on the two long sides and keep the other two for D2D / PCIe / NVLink: GH100
# (6 sites, 3 per long side, 814 mm²), B200 (4 per die on 2 dies, 8 per package), MI300X (8 stacks on the long sides of
# 4 IODs) 「二手」 — so 2 sides is the typical budget, 4 sides the geometric limit 「假设」.  The die is taken as square.
HBM_STACK_EDGE_MM = 11.0
HBM_STACK_DEPTH_MM = {"HBM4": 12.8, "HBM4E": 12.8}     # 0.313: depth away from the die edge (HBM4 11 × 12.8 「二手」)
HBM_SIDES_TYPICAL, HBM_SIDES_MAX = 2, 4
KNOWN_MAX_STACKS = {"B200 (CoWoS-L, 2 dies)": 8, "MI300X (CoWoS-S, 4 IODs)": 8}    # vendor specs 「来源」


def _d2d_edges_of(scn) -> int:
    from .d2d_catalog import d2d_edges as _d2d_edges
    return _d2d_edges(getattr(scn, "package_cards", 1)) if getattr(scn, "d2d_enabled", False) else 0


def mem_sides_max(scn) -> int:
    """0.311: die edges the memory interface may occupy = 4 − D2D edges − host-I/O edges (Chip.noc_mem_io_edges),
    floor 2 (a middle die of a ≥ 3-die package reaches the host through its neighbours 「假设」).  0.310: 4 − D2D."""
    if getattr(scn.chip, "noc_mem_io_edges", 0) == 1:
        return max(2, HBM_SIDES_MAX - _d2d_edges_of(scn) - 1)
    # 0.312 sized I/O (-1): memory may take every non-D2D edge; the I/O lanes must fit in what it leaves
    # (shoreline_report) — the same edge budget as 0.310 (0), plus that check
    return HBM_SIDES_MAX - _d2d_edges_of(scn)


def discrete_fit(scn) -> bool:
    """0.312: memory units fit edge by edge (⌊edge / unit edge⌋ whole HBM stacks / LPDDR channels per edge) — the
    sized-I/O mechanism (noc_mem_io_edges = -1).  The 0.310 / 0.311 switches (0 / 1) keep their summed-edge-length
    test, so they reproduce those versions."""
    return getattr(scn.chip, "noc_mem_io_edges", -1) == -1


def _fit_units(lengths, umm) -> int:
    import math
    return sum(math.floor(x / umm + 1e-9) for x in lengths)


def shoreline_check(scn, die_mm2: float | None) -> dict | None:
    """HBM stacks vs the die edge: None for non-HBM memory or an unknown die area.  0.309: a NoC-clustered scenario is
    checked as its whole die (all stacks on its edges, no D2D edge)."""
    from .. import mem_catalog
    scn = whole_die(scn)
    spec = mem_catalog.parse_mem_id(scn.mem_id)
    if "HBM" not in spec.mem_type.upper() or not die_mm2:
        return None
    from .hardware import die_edges
    ed = die_edges(scn.chip, die_mm2)            # 0.311: long edges first (GH100 published long edge), else square
    side = die_mm2 ** 0.5
    need = spec.n_units * HBM_STACK_EDGE_MM
    # 0.214: a die in a multi-die package gives one edge to the D2D link (B200: the inner edge to NV-HBI)
    # 0.308: two inner edges for packages of ≥ 3 dies (d2d_catalog.d2d_edges)
    # 0.311: and one edge stays for the host I/O (Chip.noc_mem_io_edges, mem_sides_max)
    d2d_edges = _d2d_edges_of(scn)
    sides_max = mem_sides_max(scn)
    typ, mx = sum(ed[:min(HBM_SIDES_TYPICAL, sides_max)]), sum(ed[:sides_max])
    if discrete_fit(scn):           # 0.312: whole stacks per edge (⌊edge / 11 mm⌋ each), not summed edge length
        ft, fm = _fit_units(ed[:min(HBM_SIDES_TYPICAL, sides_max)], HBM_STACK_EDGE_MM), \
            _fit_units(ed[:sides_max], HBM_STACK_EDGE_MM)
        status = "ok" if spec.n_units <= ft else ("tight" if spec.n_units <= fm else "over")
    else:
        ft, fm = int(typ // HBM_STACK_EDGE_MM), int(mx // HBM_STACK_EDGE_MM)
        status = "ok" if need <= typ else ("tight" if need <= mx else "over")
    out = {"stacks": spec.n_units, "need_mm": need, "edge_typical_mm": typ, "edge_max_mm": mx, "die_mm2": die_mm2,
           "status": status, "fit_typical": ft, "fit_max": fm, "discrete": discrete_fit(scn),
           "known_max_stacks": KNOWN_MAX_STACKS, "d2d_edges": d2d_edges, "sides_max": sides_max}
    if status != "ok":
        io = getattr(scn.chip, "noc_mem_io_edges", 0)
        keep = "、".join(x for x in (f"D2D {d2d_edges} 边" if d2d_edges else "", "主机 I/O 1 边" if io == 1 else "")
                         if x)
        nz = "二三四"[sides_max - 2] if 2 <= sides_max <= 4 else str(sides_max)
        out["warning"] = (f"HBM 岸线：{spec.n_units} 堆栈需 {need:.0f} mm 晶粒边缘，{'方形' if ed[0] == ed[2] else ''}晶粒 "
                          f"{die_mm2:.0f} mm² 两长边仅 {typ:.0f} mm（可用 {nz}边 {mx:.0f} mm"
                          + (f"，每边整堆栈只放得下 {fm} 个" if discrete_fit(scn) else "")
                          + (f"，另留 {keep}" if keep else "") + "）"
                          + (f"；可用 {nz}边也放不下 — 需多晶粒封装（如 B200 2 晶粒 8 堆栈）" if status == "over"
                             else "；需占用第三" + ("、四边" if sides_max >= 4 else "边"))
                          + f" · HBM shoreline: {spec.n_units} stacks need {need:.0f} mm of die edge"
                          + f" ({sides_max} usable edges, {mx:.0f} mm)")
    return out


# 0.310: conventional LPDDR5X beachfront 0.18 Tb/s per mm of die edge (Credo Weaver brief, vs its 2 Tb/s/mm
# SerDes fan-out 「二手」) at the brief's 9.6 GT/s LPDDR5X PHY → 18.75 data pins / mm 「推导」; used for every LPDDR
# generation (pins, not rate, set the edge) 「近似」.  DDR / GDDR: no figure (2 edges kept).
LPDDR_TBps_PER_MM, LPDDR_REF_GTps = 0.18, 9.6
LPDDR_PINS_PER_MM = LPDDR_TBps_PER_MM * 1000 / LPDDR_REF_GTps


def mem_edge_need_mm(spec) -> float | None:
    """0.310: die edge the DRAM interface needs: HBM stacks × 11 mm; LPDDR bus bits ÷ 18.75 pins / mm; else None."""
    t = spec.mem_type.upper()
    if "HBM" in t:
        return spec.n_units * HBM_STACK_EDGE_MM
    if "LPDDR" in t:
        return spec.bus_bits / LPDDR_PINS_PER_MM
    return None


def hbm_mem_edges(scn, die_mm2: float | None, lpddr: bool = True) -> int:
    """0.309: die edges the HBM stacks occupy = the fewest edges (long first) whose length covers stacks × 11 mm,
    in [2, sides_max]; 2 for an unknown area or a stack count beyond sides_max (shoreline_check "over").  0.310
    (``lpddr``): LPDDR interfaces too (mem_edge_need_mm), same clamp; ``lpddr=False`` = the 0.309 HBM-only rule.
    0.311: sides_max keeps the host-I/O edge (mem_sides_max) and the edges are the die's (hardware.die_edges: the
    GH100 long edge, else square — = ⌈need / side⌉ as before).  Feeds Chip.noc_mem_edges: the memory attach points
    sit where the interface PHYs are 「推导」."""
    from .. import mem_catalog
    from .hardware import die_edges
    if not die_mm2:
        return 2
    w = whole_die(scn)
    if discrete_fit(w) and lpddr:   # 0.312: the sized layout's memory edges (2 when it cannot hold the memory)
        try:
            L = sized_layout(scn, die_mm2)
        except (ValueError, KeyError, TypeError):
            return 2
        if not L["units"] or L["mem_status"] == "over":
            return 2
        return max(2, min(4, L["mem_edges"]))
    sh = shoreline_check(scn, die_mm2)
    if sh:
        if sh["status"] == "over":
            return 2
        need, smax = sh["need_mm"], int(sh["sides_max"])
    else:
        if not lpddr:
            return 2
        spec = mem_catalog.parse_mem_id(w.mem_id)
        if "LPDDR" not in spec.mem_type.upper():
            return 2
        need, smax = mem_edge_need_mm(spec), mem_sides_max(w)
    ed = die_edges(w.chip, die_mm2)
    u = mem_edge_units(mem_catalog.parse_mem_id(w.mem_id)) if discrete_fit(w) else None
    for m in range(2, smax + 1):
        if (_fit_units(ed[:m], u[1]) >= u[0]) if u else (sum(ed[:m]) >= need - 1e-9):
            return m
    return 2


def mem_edge_units(spec) -> tuple[int, float] | None:
    """0.311: the interleave units of a memory and the die edge each takes: HBM stacks × 11 mm; LPDDR channels (x16
    LPDDR4X / 5 / 5X, x24 LPDDR6 — mem_catalog, JEDEC 「来源」) × channel bits ÷ 18.75 pins / mm 「推导」; None else."""
    t = spec.mem_type.upper()
    if "HBM" in t:
        return int(spec.n_units), HBM_STACK_EDGE_MM
    if "LPDDR" in t:
        cb = 24 if "LPDDR6" in t else 16
        return max(1, spec.bus_bits // cb), cb / LPDDR_PINS_PER_MM
    return None


def mem_edge_load(scn, die_mm2: float | None, m: int) -> tuple:
    """0.311: memory units per memory edge (west, east, north, south)[:m] — water-filled one unit at a time onto the
    edge with the lowest load per attach link after the add, within each edge's length (⌊edge / unit edge⌋, the die's
    west / east = its long edges); length ignored when the units cannot fit (an over-full shoreline is
    shoreline_check's warning) or the die is unknown.  () for memory without units (DDR / GDDR)."""
    import math
    from .. import mem_catalog
    from .hardware import die_edges
    w = whole_die(scn)
    u = mem_edge_units(mem_catalog.parse_mem_id(w.mem_id))
    if u is None or m < 2:
        return ()
    n, umm = u
    from dataclasses import replace as _rp
    per = (w.chip if w.chip.mem_edges == m else _rp(w.chip, noc_mem_edges=m)).noc_mem_edge_links()
    caps = [math.floor(x / umm + 1e-9) for x in die_edges(w.chip, die_mm2)][:m] if die_mm2 else None
    if caps is not None and sum(caps) < n:
        caps = None
    load = [0] * m
    for _ in range(n):
        best = min((e for e in range(m) if caps is None or load[e] < caps[e]),
                   key=lambda e: ((load[e] + 1) / per[e], e))
        load[best] += 1
    return tuple(load)


AUTO_EDGE_DENSITY = "n5_derived"     # 0.310: die-area proxy for the auto memory edges when no densities are given
_AUTO_B: list = []
_AUTO_CACHE: dict = {}


def auto_area_budget() -> "Budget":
    if not _AUTO_B:
        _AUTO_B.append(Budget(**{k: v for k, (v, _, _) in DENSITY_PRESETS[AUTO_EDGE_DENSITY]["fields"].items()
                                 if k in Budget.__dataclass_fields__}))
    return _AUTO_B[0]


def mem_edges_auto(scn, b: "Budget | None" = None) -> int:
    """0.310: the auto memory-attach edges of a scenario — hbm_mem_edges over the die area: a preset's published
    area (hardware.preset_die_mm2, H100-like 814 mm²), else the die-area proxy (full_area) from ``b``'s densities,
    else the n5_derived preset (5 nm-class 「近似」 for other nodes).  2 when the proxy is
    incomplete.  Cached per (chip, memory, package, layout) when ``b`` is None."""
    key = None
    if b is None:
        try:
            key = (scn.chip, scn.mem_id, scn.d2d_enabled, scn.package_cards, scn.d2d_std, scn.d2d_units,
                   scn.layout, scn.node_cards, scn.link, scn.mem_cap_GB)
            hit = _AUTO_CACHE.get(key)
        except TypeError:
            key, hit = None, None
        if hit is not None:
            return hit
    try:
        from .hardware import preset_die_mm2
        die = preset_die_mm2(scn.chip) if b is None else None
        if die is None:
            die = full_area(scn, b if b is not None else auto_area_budget()).get("mm2")
        m = hbm_mem_edges(scn, die) if die else 2
    except (ValueError, KeyError, TypeError):
        m = 2
    if key is not None:
        if len(_AUTO_CACHE) > 20000:
            _AUTO_CACHE.clear()
        _AUTO_CACHE[key] = m
    return m


def auto_die_mm2(scn, b: "Budget | None" = None) -> float | None:
    """0.311: the die area the memory-edge rules use — the preset's published area (hardware.preset_die_mm2) when
    ``b`` is None, else the area proxy (full_area) from ``b`` or the n5_derived preset."""
    from .hardware import preset_die_mm2
    die = preset_die_mm2(scn.chip) if b is None else None
    if die is None:
        die = full_area(scn, b if b is not None else auto_area_budget()).get("mm2")
    return die


_LOAD_CACHE: dict = {}


def mem_load_auto(scn) -> tuple:
    """0.311: Chip.noc_mem_load of a scenario — mem_edge_load over auto_die_mm2 for the chip's memory edges.
    Cached per (chip, memory, package, layout)."""
    try:
        key = (scn.chip, scn.mem_id, scn.d2d_enabled, scn.package_cards, scn.layout, scn.mem_cap_GB)
        hit = _LOAD_CACHE.get(key)
    except TypeError:
        key, hit = None, None
    if hit is not None:
        return hit
    try:
        die = auto_die_mm2(scn)
        ld = ()
        if die and discrete_fit(whole_die(scn)):      # 0.312: the sized layout's load when its edge count is the chip's
            L = sized_layout(scn, die)
            if len(L["load"]) == scn.chip.mem_edges and L["mem_status"] != "over":
                ld = tuple(L["load"])
        if not ld:
            ld = mem_edge_load(scn, die, scn.chip.mem_edges)
    except (ValueError, KeyError, TypeError):
        ld = ()
    if key is not None:
        if len(_LOAD_CACHE) > 20000:
            _LOAD_CACHE.clear()
        _LOAD_CACHE[key] = ld
    return ld


def d2d_shoreline(scn, die_mm2: float | None) -> dict | None:
    """0.308: D2D PHY units vs the die edges they get (d2d_catalog.d2d_edges of a square die): units × the unit's edge
    length ≤ edges × side.  None when D2D is off, the grade has no published footprint, or the die area is unknown."""
    from .d2d_catalog import d2d_edge_mm, d2d_edges
    if not (getattr(scn, "d2d_enabled", False) and scn.package_cards > 1) or not die_mm2 \
            or getattr(scn.chip, "noc_clusters", 1) > 1:
        return None
    w = d2d_edge_mm(scn.d2d_std)
    if w is None:
        return None
    ed = d2d_edges(scn.package_cards)
    need, have = scn.d2d_units * w, ed * die_mm2 ** 0.5
    return {"units": scn.d2d_units, "unit_edge_mm": w, "need_mm": need, "edges": ed, "edge_mm": have,
            "status": "ok" if need <= have else "over"}


EDGE_NAMES = ("W", "E", "N", "S")      # 0.312: die edges in hardware.die_edges order (long first = W / E)
IO_MACRO_LANES = 4                     # 0.312: SerDes macros of 4 duplex lanes (Synopsys PCIe5 PHY x4, Cadence 112G
                                       # quad macro 「来源」); a macro is never split across two edges 「假设」


# 0.313: scale-up SerDes lane generations (per-direction pair rate of the shipping scale-up links, NVIDIA A100 / H100 /
# Blackwell whitepapers 「来源」).  The lane rate is an era choice (a design uses the fastest lane of its generation,
# not the slowest that reaches its bandwidth): Chip.io_su_lane_Gbps, default 200 (HBM3E era), H100-like 100.
# Sensitivity of the verdicts: MODEL §258.2.
SU_LANE_GENERATIONS = {50.0: "NVLink3 (A100, HBM2E): 12 links × 4 pairs × 50 Gb/s = 300 GB/s/dir 「来源」",
                       100.0: "NVLink4 (H100, HBM3): 18 links × 2 pairs × 100 Gb/s = 450 GB/s/dir 「来源」",
                       200.0: "NVLink5 (B200, HBM3E): 18 links × 2 pairs × 200 Gb/s (224G PAM4) = 900 GB/s/dir 「来源」"}

# 0.313: die edge per duplex scale-up lane by rate (Chip.io_su_lane_mm = 0): 224G ≈ 0.5 mm (OIF 448G workshop,
# MaxLinear Apr 2025: ~4 mm per Rx/Tx 8-lane port 「来源」, "~"); 50 / 112G: no public figure → the PCIe5 lane's
# 0.285 mm (Intel CICC 2020) 「假设」 (the 0.312 value).  Per Tb/s: 224G 2.5 mm, 100G 2.85 mm — similar, so the lane
# rate alone does not rescue a tight I/O edge.
SU_LANE_MM = {50.0: 0.285, 100.0: 0.285, 200.0: 0.5}


def su_lane_mm(chip) -> float:
    v = getattr(chip, "io_su_lane_mm", 0.0) or 0.0
    if v > 0:
        return float(v)
    r = float(chip.io_su_lane_Gbps)
    return SU_LANE_MM.get(r, 0.5 if r > 112.0 else 0.285)


def io_need(scn) -> dict:
    """0.312: host + scale-up I/O lanes of one die and the die edge they need (Chip.io_*; mechanism and sources at
    the fields).  Host: io_host_lanes per package, split over its dies (⌈lanes / dies⌉ to x4 macros 「假设」).
    Scale-up: only when the replica's cards leave the package (layout.cards > dies per package — the c2c PHY rule of
    phy_area), Link.GBps per direction × 8 ÷ io_su_lane_Gbps duplex lanes per card (× the clusters of a NoC-clustered
    die), to x4 macros.  The network NIC hangs off PCIe (not on the die)."""
    import math
    ch = scn.chip
    k = getattr(ch, "noc_clusters", 1)
    pk = scn.package_cards if scn.d2d_enabled else 1
    dies = 1 if k > 1 else pk
    q = IO_MACRO_LANES
    host = math.ceil(ch.io_host_lanes / dies / q) * q if ch.io_host_lanes else 0
    su_cards = k if k > 1 else 1
    su = 0
    rate = float(ch.io_su_lane_Gbps)
    basis = SU_LANE_GENERATIONS.get(rate, "pinned 「假设」")
    if scn.layout.cards > pk:
        su = math.ceil(scn.link.GBps * 8 * su_cards / rate / q) * q
    smm = su_lane_mm(ch)
    macros = [q * ch.io_lane_mm] * (host // q) + [q * smm] * (su // q)
    return {"host_lanes": host, "scaleup_lanes": su, "lane_mm": ch.io_lane_mm, "su_lane_mm": smm,
            "su_lane_Gbps": rate, "su_lane_basis": basis, "need_mm": host * ch.io_lane_mm + su * smm,
            "macros_mm": macros, "dies": dies}


# 0.312: die shape under the sized-I/O mechanism.  A die with no published geometry may be drawn as any rectangle of
# its area inside the scanner exposure field 26 × 33 mm (the 858 mm² reticle limit already used, 「来源」) up to an
# aspect of 2 : 1 「假设」 (shipping ASICs: GH100 33 × 24.7 = 1.34, B200-class dies ≈ 1.2–1.4 「二手」); the squarest
# shape whose shoreline holds memory + D2D + I/O is taken (0.25 mm steps).  D2D sits on the short or the long
# edge(s) — whichever lets the rest fit (a row of strips joins on its long edges, B200 NV-HBI 「二手」).
RETICLE_FIELD_MM = (26.0, 33.0)
DIE_ASPECT_MAX = 2.0
SHAPE_STEP_MM = 0.25
_SIZED_CACHE: dict = {}


def die_shapes(chip, die_mm2: float) -> list:
    """0.312: candidate edge lengths [long, long, short, short], squarest first (a published rectangle: only it)."""
    from .hardware import die_edges
    ed = die_edges(chip, die_mm2)
    if abs(ed[0] - ed[2]) > 1e-9:
        return [list(ed)]
    out, L = [list(ed)], ed[0]
    if die_mm2 > RETICLE_FIELD_MM[0] * RETICLE_FIELD_MM[1] * (1 + 1e-9):
        return out
    while True:
        L += SHAPE_STEP_MM
        S = die_mm2 / L
        if L > RETICLE_FIELD_MM[1] + 1e-9 or L / S > DIE_ASPECT_MAX + 1e-9:
            return out
        if S <= RETICLE_FIELD_MM[0] + 1e-9:
            out.append([L, L, S, S])


_D2D_POS = {0: [()], 1: [(3,), (1,)], 2: [(2, 3), (0, 1)]}


def _place(ed, dpos, units, links4, macros, d2d_need) -> dict:
    """0.312: one shape × D2D position: memory units on the fewest non-D2D edges (long first) that hold them whole,
    water-filled against each edge's attach links within ⌊edge / unit⌋; D2D PHY on dpos; I/O x4 macros first-fit
    into what memory leaves on the non-D2D edges."""
    import math
    non = [i for i in range(4) if i not in dpos]
    mem_mm, mem_u, io_mm = [0.0] * 4, [0] * 4, [0.0] * 4
    d2d_mm = [0.0] * 4
    d2d_st = None
    if dpos:
        d2d_st = "unknown"
        if d2d_need is not None:
            for i in dpos:
                d2d_mm[i] = d2d_need / len(dpos)
            d2d_st = "ok" if d2d_need <= sum(ed[i] for i in dpos) + 1e-9 else "over"
    mst, k, fitk, load = "unknown", min(2, len(non)), None, ()
    if units:
        n, umm = units
        cap = {i: math.floor(ed[i] / umm + 1e-9) for i in non}
        lo = min(2, len(non))
        k = next((j for j in range(lo, len(non) + 1) if sum(cap[i] for i in non[:j]) >= n), len(non))
        fitk = sum(cap[i] for i in non[:k])
        mst = "ok" if n <= sum(cap[i] for i in non[:lo]) else ("tight" if n <= fitk else "over")
        med = non[:k]
        ld = [0] * k
        for _ in range(n):
            ok = [j for j in range(k) if mst == "over" or ld[j] < cap[med[j]]]
            j = min(ok, key=lambda j: ((ld[j] + 1) / links4[j], j))
            ld[j] += 1
        load = tuple(ld)
        for j, i in enumerate(med):
            mem_u[i], mem_mm[i] = ld[j], ld[j] * umm
    free = [max(0.0, ed[i] - mem_mm[i]) if i in non else 0.0 for i in range(4)]
    left = 0
    order = sorted(non, key=lambda j: (-free[j], j))
    for mac in sorted(macros, reverse=True):
        i = next((j for j in order if free[j] - io_mm[j] >= mac - 1e-9), None)
        if i is None:
            left += 1
        else:
            io_mm[i] += mac
    io_st = "ok" if not left else "over"
    bad = sum(st == "over" for st in (mst, d2d_st, io_st))
    return {"ed": list(ed), "dpos": list(dpos), "non": non, "mem_edges": k, "mem_status": mst, "fit_units": fitk,
            "load": load, "mem_u": mem_u, "mem_mm": mem_mm, "d2d_mm": d2d_mm, "d2d_status": d2d_st, "io_mm": io_mm,
            "io_status": io_st, "io_free_mm": sum(free), "unplaced_macros": left, "bad": bad,
            "free_total": sum(ed) - sum(mem_mm) - sum(d2d_mm) - sum(io_mm)}


def sized_layout(scn, die_mm2: float) -> dict:
    """0.312: the buildable layout of one die under the sized-I/O mechanism — the squarest die shape (die_shapes)
    × D2D position (short / long edges) whose memory, D2D and I/O all fit (fewer memory edges, then more free edge,
    on ties); else the square die's least-violating placement.  Cached."""
    from .. import mem_catalog
    w = whole_die(scn)
    try:
        key = (w.chip, w.mem_id, scn.d2d_enabled, scn.package_cards, scn.d2d_std, scn.d2d_units, scn.layout,
               scn.link, scn.mem_cap_GB, round(die_mm2, 6))
        hit = _SIZED_CACHE.get(key)
    except TypeError:
        key, hit = None, None
    if hit is not None:
        return hit
    units = mem_edge_units(mem_catalog.parse_mem_id(w.mem_id))
    from dataclasses import replace as _rp
    links4 = (_rp(w.chip, noc_mem_edges=4)).noc_mem_edge_links()
    nz = io_need(scn)
    ds = d2d_shoreline(scn, die_mm2)
    nd = _d2d_edges_of(w)
    best = None
    for si, ed in enumerate(die_shapes(w.chip, die_mm2)):
        cands = [_place(ed, dp, units, links4, nz["macros_mm"], ds["need_mm"] if ds else None)
                 for dp in _D2D_POS.get(nd, [tuple(range(4 - nd, 4))])]
        ok = [c for c in cands if not c["bad"]]
        if ok:
            best = min(ok, key=lambda c: (c["mem_edges"], -c["free_total"]))
            best["shape_index"] = si
            break
        if si == 0:
            best = min(cands, key=lambda c: (c["bad"], c["mem_edges"], -c["free_total"]))
            best["shape_index"] = 0
    best["io_need"] = nz
    best["d2d"] = ds
    best["units"] = units
    if key is not None:
        if len(_SIZED_CACHE) > 20000:
            _SIZED_CACHE.clear()
        _SIZED_CACHE[key] = best
    return best


def _sized_report(scn, die: float, out: dict) -> dict:
    from .. import mem_catalog
    w = whole_die(scn)
    L = sized_layout(scn, die)
    ed, nd = L["ed"], _d2d_edges_of(w)
    spec = mem_catalog.parse_mem_id(w.mem_id)
    need = mem_edge_need_mm(spec)
    if L["units"]:
        n, umm = L["units"]
        mem = {"kind": spec.kind, "units": n, "unit_mm": umm, "need_mm": need, "edges": L["mem_edges"],
               "sides_max": len(L["non"]), "avail_mm": sum(ed[i] for i in L["non"]), "fit_units": L["fit_units"],
               "load": list(L["load"]), "status": L["mem_status"]}
    else:
        mem = {"kind": spec.kind, "units": None, "need_mm": None, "edges": w.chip.mem_edges, "sides_max": len(L["non"]),
               "status": "unknown"}
        out["reasons"].append(f"{spec.mem_type} 无公开岸线密度：外存岸线未判定")
    ds = L["d2d"]
    d2d = None
    if nd:
        d2d = {"edges": nd, "status": L["d2d_status"], "at": [EDGE_NAMES[i] for i in L["dpos"]]}
        if ds:
            d2d.update(units=ds["units"], unit_edge_mm=ds["unit_edge_mm"], need_mm=ds["need_mm"],
                       avail_mm=sum(ed[i] for i in L["dpos"]))
    nz = L["io_need"]
    io = {"mode": "sized", "status": L["io_status"], "avail_mm": L["io_free_mm"], "unplaced_macros": L["unplaced_macros"],
          **{k: nz[k] for k in ("host_lanes", "scaleup_lanes", "lane_mm", "su_lane_mm", "need_mm")}}
    rows = []
    for i in range(4):
        used = L["mem_mm"][i] + L["d2d_mm"][i] + L["io_mm"][i]
        role = "+".join(r for r, v in (("mem", L["mem_mm"][i]), ("d2d", L["d2d_mm"][i]), ("io", L["io_mm"][i]))
                        if v > 0) or "free"
        rows.append({"edge": EDGE_NAMES[i], "len_mm": ed[i], "mem_units": L["mem_u"][i], "mem_mm": L["mem_mm"][i],
                     "d2d_mm": L["d2d_mm"][i], "io_mm": L["io_mm"][i], "used_mm": used, "free_mm": ed[i] - used,
                     "role": role})
    if mem["status"] == "over":
        if spec.kind == "HBM":
            out["reasons"].append(f"HBM 岸线：{mem['units']} 堆栈需 {need:.0f} mm，非 D2D 的 {mem['sides_max']} 边"
                                  f"（{'/'.join(f'{ed[i]:.1f}' for i in L['non'])} mm，任意 ≤ 2:1 长宽比）每边整堆栈只放得下 "
                                  f"{mem['fit_units']} 个")
        else:
            out["reasons"].append(f"{spec.mem_type} 岸线：{mem['units']} 通道需 {need:.1f} mm > 可用 {mem['sides_max']} 边 "
                                  f"{mem['avail_mm']:.1f} mm（可放 {mem['fit_units']}）")
    if d2d and d2d["status"] == "over":
        out["reasons"].append(f"D2D 岸线：{d2d['units']} 单元需 {d2d['need_mm']:.1f} mm > {nd} 边 {d2d['avail_mm']:.1f} mm")
    if io["status"] == "over":
        out["reasons"].append(f"I/O 岸线：主机 {io['host_lanes']} + scale-up {io['scaleup_lanes']} 通道需 "
                              f"{io['need_mm']:.1f} mm，外存 / D2D 之外剩余 {io['avail_mm']:.1f} mm 放不下 "
                              f"{io['unplaced_macros']} 个 x4 宏")
    over = [st for st in (mem["status"], (d2d or {}).get("status"), io["status"]) if st == "over"]
    # 0.313: interposer footprint of this die + its stacks as placed — the bounding box (stacks stand out from their
    # edge by their depth: HBM3/3E 11 mm, Micron 「来源」; HBM4 12.8 mm 「二手」) vs the plain area sum; a stack row on
    # 3–4 sides leaves the corner squares empty, so Σ areas understates the interposer (0.308 – 0.312 used Σ)
    if spec.kind == "HBM" and L["units"]:
        dep = HBM_STACK_DEPTH_MM.get(spec.mem_type, HBM_STACK_EDGE_MM)
        dz = [dep if L["mem_u"][i] else 0.0 for i in range(4)]
        bbox = (ed[2] + dz[0] + dz[1]) * (ed[0] + dz[2] + dz[3])
        ssum = die + L["units"][0] * HBM_STACK_EDGE_MM * dep
        out["footprint"] = {"bbox_mm2": max(bbox, ssum), "sum_mm2": ssum, "stack_depth_mm": dep}
    out.update({"edges": rows, "mem": mem, "d2d": d2d, "io": io, "feasible": not over,
                "status": "over" if over else ("tight" if mem["status"] == "tight" else "ok"),
                "used_mm": sum(r["used_mm"] for r in rows), "perimeter_mm": sum(ed),
                "shape": {"long_mm": ed[0], "short_mm": ed[2], "aspect": ed[0] / ed[2],
                          "reshaped": L["shape_index"] > 0, "aspect_max": DIE_ASPECT_MAX,
                          "field_mm": list(RETICLE_FIELD_MM)}})
    return out


_SHORE_CACHE: dict = {}


def shoreline_report(scn, die_mm2: float | None = None, b: "Budget | None" = None) -> dict:
    """0.312: shoreline_report_uncached, cached per (chip, memory, package, layout, link) for the auto die area."""
    key = None
    if die_mm2 is None and b is None:
        try:
            key = (scn.chip, scn.mem_id, scn.d2d_enabled, scn.package_cards, scn.d2d_std, scn.d2d_units,
                   scn.layout, scn.link, scn.mem_cap_GB)
            hit = _SHORE_CACHE.get(key)
        except TypeError:
            key, hit = None, None
        if hit is not None:
            return copy.deepcopy(hit)
    try:
        out = shoreline_report_uncached(scn, die_mm2, b)
    except (ValueError, KeyError, TypeError) as e:
        out = {"die_mm2": die_mm2, "feasible": None, "status": "unknown", "reasons": [f"岸线不可判定：{e}"]}
    if key is not None:
        if len(_SHORE_CACHE) > 20000:
            _SHORE_CACHE.clear()
        _SHORE_CACHE[key] = copy.deepcopy(out)
    return out


def shoreline_report_uncached(scn, die_mm2: float | None = None, b: "Budget | None" = None) -> dict:
    """0.312: buildability of one die's shoreline — per edge (W, E, N, S = hardware.die_edges, long first) the memory
    units placed (mem_edge_load), the D2D PHY (d2d_shoreline, on the last D2D edges), and the host / scale-up I/O
    (io_need: x4 macros placed greedily on the non-D2D edges' free length in sized mode; the whole last non-D2D edge
    with noc_mem_io_edges = 1; not checked with 0).  Status per part ok / tight (memory beyond the 2 long edges) /
    over; ``feasible`` = no part over (None when the die area is unknown).  ``die_mm2`` None = auto_die_mm2."""
    from .. import mem_catalog
    from .hardware import die_edges
    w = whole_die(scn)
    die = die_mm2 or auto_die_mm2(scn, b)
    mode = {-1: "sized", 0: "none", 1: "edge"}[getattr(scn.chip, "noc_mem_io_edges", -1)]
    out = {"die_mm2": die, "io_mode": mode, "feasible": None, "status": "unknown", "reasons": []}
    if not die:
        out["reasons"].append("晶粒面积未知：岸线不可判定")
        return out
    if discrete_fit(w):             # 0.312: the sized mechanism (shape × D2D position search)
        return _sized_report(scn, die, out)
    ed = list(die_edges(w.chip, die))
    nd = _d2d_edges_of(w)
    smax = mem_sides_max(w)
    m = min(w.chip.mem_edges, smax) if w.chip.mem_edges else 2
    spec = mem_catalog.parse_mem_id(w.mem_id)
    u = mem_edge_units(spec)
    need = mem_edge_need_mm(spec)
    mem_mm = [0.0] * 4
    mem_u = [0] * 4
    if u:
        n, umm = u
        load = mem_edge_load(w, die, m)
        for i, x in enumerate(load):
            mem_u[i], mem_mm[i] = x, x * umm
        fit2, fitm = _fit_units(ed[:min(2, smax)], umm), _fit_units(ed[:smax], umm)
        if discrete_fit(w):
            mst = "ok" if n <= fit2 else ("tight" if n <= fitm else "over")
        else:                       # 0.310 / 0.311 switches: summed edge length (their rule)
            t2, tm = sum(ed[:min(2, smax)]), sum(ed[:smax])
            mst = "ok" if need <= t2 + 1e-9 else ("tight" if need <= tm + 1e-9 else "over")
        mem = {"kind": spec.kind, "units": n, "unit_mm": umm, "need_mm": need, "edges": m, "sides_max": smax,
               "avail_mm": sum(ed[:smax]), "fit_units": fitm, "load": list(load), "status": mst}
    else:
        mem = {"kind": spec.kind, "units": None, "need_mm": None, "edges": m, "sides_max": smax, "status": "unknown"}
        out["reasons"].append(f"{spec.mem_type} 无公开岸线密度：外存岸线未判定")
    d2d_mm = [0.0] * 4
    ds = d2d_shoreline(scn, die)
    d2d = None
    if nd:
        d2d = {"edges": nd, "status": "unknown"}
        if ds:
            for i in range(4 - nd, 4):
                d2d_mm[i] = ds["need_mm"] / nd
            d2d = {"edges": nd, "units": ds["units"], "unit_edge_mm": ds["unit_edge_mm"], "need_mm": ds["need_mm"],
                   "avail_mm": sum(ed[4 - nd:]), "status": ds["status"]}
    io_mm = [0.0] * 4
    io = {"mode": mode, "status": "off"}
    if mode != "none":
        nz = io_need(scn)
        io.update({k: nz[k] for k in ("host_lanes", "scaleup_lanes", "lane_mm", "su_lane_mm", "need_mm")})
        if mode == "edge":
            e = 4 - nd - 1
            if e < m:               # floor 2: a middle die reaches the host through its neighbours (0.311)
                io.update(status="via_neighbour", avail_mm=0.0)
            else:                   # 0.311 rule: the reserved edge is assumed to suffice (reported, not checked)
                io_mm[e] = min(nz["need_mm"], ed[e])
                io.update(status="reserved", avail_mm=ed[e], edge=EDGE_NAMES[e],
                          fits_edge=nz["need_mm"] <= ed[e] + 1e-9)
        else:
            free = [max(0.0, ed[i] - mem_mm[i]) if i < 4 - nd else 0.0 for i in range(4)]
            placed, left = [0.0] * 4, 0
            order = sorted(range(4), key=lambda j: (-free[j], j))      # first fit, the freest edge first
            for mac in sorted(nz["macros_mm"], reverse=True):
                i = next((j for j in order if free[j] - placed[j] >= mac - 1e-9), None)
                if i is None:
                    left += 1
                else:
                    placed[i] += mac
            io_mm = placed
            io.update(status="ok" if not left else "over", avail_mm=sum(free), unplaced_macros=left)
    rows = []
    for i in range(4):
        used = mem_mm[i] + d2d_mm[i] + io_mm[i]
        if mode == "edge" and io.get("edge") == EDGE_NAMES[i]:
            used = max(used, ed[i])
        role = "+".join(r for r, v in (("mem", mem_mm[i]), ("d2d", d2d_mm[i]), ("io", io_mm[i])) if v > 0) or "free"
        rows.append({"edge": EDGE_NAMES[i], "len_mm": ed[i], "mem_units": mem_u[i], "mem_mm": mem_mm[i],
                     "d2d_mm": d2d_mm[i], "io_mm": io_mm[i], "used_mm": used, "free_mm": ed[i] - used, "role": role})
    parts = [("外存", mem["status"]), ("D2D", (d2d or {}).get("status")), ("I/O", io["status"])]
    over = [p for p, st in parts if st == "over"]
    if mem["status"] == "over":
        if spec.kind == "HBM":
            out["reasons"].append(f"HBM 岸线：{mem['units']} 堆栈需 {need:.0f} mm > 可用 {smax} 边 {mem['avail_mm']:.0f} mm"
                                  + (f"（每边整堆栈，只放得下 {mem['fit_units']} 个）" if discrete_fit(w) else ""))
        else:
            out["reasons"].append(f"{spec.mem_type} 岸线：{mem['units']} 通道需 {need:.1f} mm > 可用 {smax} 边 "
                                  f"{mem['avail_mm']:.1f} mm（可放 {mem['fit_units']}）")
    if d2d and d2d["status"] == "over":
        out["reasons"].append(f"D2D 岸线：{d2d['units']} 单元需 {d2d['need_mm']:.1f} mm > {nd} 边 {d2d['avail_mm']:.1f} mm")
    if io["status"] == "over":
        out["reasons"].append(f"I/O 岸线：主机 {io['host_lanes']} + scale-up {io['scaleup_lanes']} 通道需 "
                              f"{io['need_mm']:.1f} mm > 外存 / D2D 之外剩余 {io['avail_mm']:.1f} mm")
    out.update({"edges": rows, "mem": mem, "d2d": d2d, "io": io, "feasible": not over,
                "status": "over" if over else ("tight" if mem["status"] == "tight" else "ok"),
                "used_mm": sum(r["used_mm"] for r in rows), "perimeter_mm": sum(ed)})
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


def whole_die(scn):
    """0.309: the monolithic die a NoC-clustered scenario (chip.noc_clusters = K > 1, K cluster cards on the on-die
    tier) stands for: engines / SRAM / SLC / state SRAM × K, memory units × K, one flat NoC domain, D2D off, layout
    unchanged (area / shoreline only).  Any other scenario is returned unchanged."""
    k = getattr(scn.chip, "noc_clusters", 1)
    if k <= 1:
        return scn
    from dataclasses import replace as _rp
    from .. import mem_catalog as mc
    ch = scn.chip
    c = _rp(ch, engines=ch.engines * k, sram_mib=ch.sram_mib * k, slc_mib=ch.slc_mib * k,
            state_sram_mib=ch.state_sram_mib * k, noc_clusters=1)
    sp = mc.parse_mem_id(scn.mem_id)
    hbm = sp.kind == "HBM"
    kw = {"height": sp.hbm_height, "die_Gb": sp.hbm_die_Gb} if hbm else {"cap_GB": sp.cap_per_unit_GB}
    mid = mc.make_spec(sp.mem_type, form=None if hbm else sp.form, width_bits=sp.unit_width_bits, rate=sp.rate_MTps,
                       count=sp.n_units * k, **kw).id
    lay = scn.layout                  # the clusters were cards along one axis: the die is 1/K of the replica's cards
    for ax in ("tp", "pp", "dp", "cp", "sp"):
        if getattr(lay, ax) % k == 0:
            lay = _rp(lay, **{ax: getattr(lay, ax) // k})
            break
    return _rp(scn, chip=c, mem_id=mid, d2d_enabled=False, package_cards=1, mem_cap_GB=scn.mem_cap_GB * k, layout=lay)


def full_area(scn, b: Budget, noc_routers: bool = False) -> dict:
    """Die area proxy incl. interface PHYs (0.149): area_estimate(chip) + phy_area(scenario).
    0.307 ``noc_routers`` (the energy DSE's NoC-width axis): + the NoC routers, counted once — after the uncore share
    (uncore_frac is TPUv1's control + unlabelled share; TPUv1 has no NoC: its blocks are joined by dedicated 256-byte
    paths, so the share holds wiring / white space, not routers; the NoC's link wires play the role of those paths
    and stay inside the share), and not at all when mm2_fixed is given (it already means "everything else, incl. NoC").
    0.309: a NoC-clustered scenario (chip.noc_clusters > 1) is priced as its whole monolithic die (whole_die: no D2D
    PHY — the cluster tier is the NoC's own links); "mm2" is the die, "clusters" says so."""
    k = getattr(scn.chip, "noc_clusters", 1)
    if k > 1:
        out = full_area(whole_die(scn), b, noc_routers)
        out["phy_notes"] = out["phy_notes"] + [f"片上 {k} 个 NoC 簇：按整片单晶粒计面积（簇间链路即 NoC 链路，无 D2D PHY，0.309）"]
        return {**out, "clusters": k}
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
    kc = getattr(ch, "noc_clusters", 1)
    if area is not None and (kc > 1 or any(getattr(b, k) is not None for k in _PHY_DENS)):      # 0.149: interface PHYs
        area = full_area(r.scenario, b)        # 0.309: clustered → the whole die
    for k, mult in (("die_mm2", 1), ("system_mm2", lay.cards / kc)):
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
