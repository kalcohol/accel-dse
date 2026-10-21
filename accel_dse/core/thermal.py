"""0.313: buildability from power density and package power (opt-in limits; the figures are always reported).
0.314: audited against shipped parts (MODEL §259) — one flux definition on both sides, limits re-sourced.

Flux judged (``flux_W_mm2``) = package (card) power ÷ the package's logic-die area (dies × die mm²): the definition a
shipped part's figure can be formed in (TDP / TBP ÷ die area; neither its HBM power nor its VR loss is published).
0.313 judged the die-level figure (card W − DRAM-device W, ÷ area) against a limit equal to H100's card-level figure
— two definitions.  The die-level figure stays reported (``die_flux_W_mm2``; die heat = (card W − DRAM devices) ÷
dies, equal split 「近似」; the host-side DRAM PHY stays on the die).  Both are averages over the die; hot spots are
not modelled — a shipped part's average already carries its own hot-spot ratio, so the comparison holds for a design
of similar uniformity 「近似」.

What the modelled card power contains (energy_report): dynamic MAC / vector / SRAM / SLC / NoC (+ wire) / clock /
DRAM-device / DRAM-PHY / D2D / link energy, SRAM (+ NoC router) leakage, HBM standby; whole-card static (logic
leakage, uncore) only when the energy table has idle_W or idle_frac_tdp × tdp_W (``static_included``).  Not modelled:
voltage-regulator loss on the module (inside a shipped TBP).  Both omissions make the modelled power low against a
shipped TDP, so an "over" verdict is robust and an "ok" within that margin is not 「近似」.

Limits (``COOLING``): rule = the larger of (a) the highest average flux / package power of a shipped part with that
cooling and (b) the HIR 2021 Table 1 *thermal-budget* figure (50 °C junction-to-inlet, 4 cm² chip, demonstrated
hardware); the HIR *uniform-heat-flux* idealisations (air 0.85, water 2.50 W/mm²) are not used (0.313 used both).
  air_oam  1000 W: AMD Instinct MI350X, OAM, air-cooled, 1000 W TBP (AMD product page / ROCm MI350 docs) 「来源」;
           Intel Gaudi 3 HL-325L OAM 900 W air (Hot Chips 2024, Intel) 「来源」.  (0.313: 600 W, the OCP OAM r2.0
           air recommendation — exceeded by both shipped OAM parts.)
  air_hgx  1000 W: HGX B200, "configurable up to 1,000 W", air-cooled HGX B200 racks (NVIDIA Blackwell datasheet) 「来源」.
  air flux 0.86 W/mm²: H100 / H200 SXM 700 W air-cooled (Dell XE9680 spec sheet: 8 × H200 700 W, air, 35 °C inlet
           「来源」) on GH100 814 mm² 「二手」 → 0.86 「推导」 — above HIR's 0.55 (thermal budget 50 °C) 「来源」.
  liquid   1400 W: AMD Instinct MI355X, OAM, direct liquid cooling, 1400 W TBP (AMD MI355X datasheet) 「来源」;
           HGX B300 1400 W bin, liquid (Supermicro) 「二手」.
  liquid flux 1.70 W/mm²: HIR 2021 Table 1 water-cooled module cold plate at a 50 °C budget, 4 cm² chip 「来源」 — above
           the shipped liquid parts (MI355X 1400 W on ≈ 1464 mm² of IOD footprint 「二手」 → 0.96).  (0.313: 2.5, the
           uniform-flux idealisation.)
Power delivery: the die current at an assumed 0.75 V core rail is reported; no public per-bump current limit for
current micro-bump / hybrid-bond pitches was found (the 0.2 A per C4 bump guideline is a legacy figure for
50–100 µm bumps 「二手」), so power delivery is reported, not judged.
"""

COOLING = {
    "air_oam": {"pkg_W": 1000.0, "flux_W_mm2": 0.86,
                "src": "MI350X OAM 风冷 1000 W TBP（AMD）「来源」、Gaudi 3 HL-325L OAM 风冷 900 W（Hot Chips 2024）「来源」；"
                       "热流 0.86 W/mm² = H100/H200 SXM 风冷 700 W ÷ 814 mm²「推导」（高于 HIR 2021 表 1 风冷 0.55 W/mm²「来源」）"},
    "air_hgx": {"pkg_W": 1000.0, "flux_W_mm2": 0.86,
                "src": "HGX B200 风冷，最高 1000 W / GPU（NVIDIA Blackwell 数据手册）「来源」；热流同风冷 0.86 W/mm²「推导」"},
    "liquid": {"pkg_W": 1400.0, "flux_W_mm2": 1.70,
               "src": "MI355X OAM 直接液冷 1400 W TBP（AMD 数据手册）「来源」；热流 1.70 W/mm²：HIR 2021 表 1 水冷板、"
                      "50 °C 温度预算、4 cm² 芯片「来源」（出货液冷件 ≤ 0.96 W/mm²「二手」）"},
}
# Shipped parts: card-level average flux = TDP / TBP ÷ logic-die area (the heat-exiting footprint for 3D stacks).
# area None = not published (package power only).  The judged limits above come from these and HIR (rule above).
KNOWN_PARTS = (
    {"part": "H100 SXM", "W": 700.0, "die_mm2": 814.0, "dies": 1, "cooling": "air",
     "src": "NVIDIA 700 W（风冷 HGX）；GH100 814 mm²「二手」 → 0.86 W/mm²（出货风冷最高）"},
    {"part": "H200 SXM", "W": 700.0, "die_mm2": 814.0, "dies": 1, "cooling": "air",
     "src": "Dell XE9680 规格表：8 × H200 700 W 风冷、35 °C 进风「来源」；同 GH100 晶粒 → 0.86 W/mm²"},
    {"part": "B200 (HGX, air)", "W": 1000.0, "die_mm2": 858.0, "dies": 2, "cooling": "air",
     "src": "NVIDIA 数据手册 1000 W「来源」；两晶粒近光罩上限（≤ 858 mm²，面积未公布）「二手」 → ≥ 0.58 W/mm²"},
    {"part": "B200 (liquid)", "W": 1200.0, "die_mm2": 858.0, "dies": 2, "cooling": "liquid",
     "src": "HGX B200 液冷 1200 W 档「二手」 → ≥ 0.70 W/mm²"},
    {"part": "MI300X", "W": 750.0, "die_mm2": 370.0, "dies": 4, "cooling": "air",
     "src": "AMD 750 W TBP「来源」、Dell XE9680 风冷「来源」；4 × IOD 370 mm²（XCD 叠于其上；ASCII.jp 实测）「二手」 → 0.51 W/mm²"},
    {"part": "MI350X", "W": 1000.0, "die_mm2": 732.2, "dies": 2, "cooling": "air",
     "src": "AMD 1000 W TBP 风冷「来源」；2 × IOD 732 mm²（ASCII.jp 由照片推算）「二手」 → 0.68 W/mm²"},
    {"part": "MI355X", "W": 1400.0, "die_mm2": 732.2, "dies": 2, "cooling": "liquid",
     "src": "AMD 1400 W TBP 直接液冷「来源」；同 MI350X 面积「二手」 → 0.96 W/mm²"},
    {"part": "Gaudi 3 HL-325L", "W": 900.0, "die_mm2": None, "dies": 2, "cooling": "air",
     "src": "Intel Hot Chips 2024：OAM 风冷 900 W / 液冷 1200 W「来源」；晶粒面积未公布"},
    {"part": "TPU v5p / v6e", "W": None, "die_mm2": None, "dies": 1, "cooling": "liquid / air",
     "src": "Google 未公布单芯片 TDP 与面积（第三方估计不采用）"},
)
PD_VDD_V = 0.75          # 「假设」 core rail for the reported die current only (not judged)


def cooling_names() -> tuple:
    return tuple(COOLING)


def thermal_report(pkg_W: float | None, die_mm2: float | None, dies: int = 1, dram_dev_W: float = 0.0,
                   cooling: str | None = None, static_included: bool | None = None) -> dict:
    """Package power and heat flux vs a cooling preset.  ``cooling`` None = report only (``feasible`` None, status
    "unchecked"); a preset name = judged (``feasible`` False when the package power or the card-level flux exceeds it).
    ``static_included``: whether ``pkg_W`` holds the whole-card static floor (energy table idle_W / idle_frac_tdp);
    False adds a note (the modelled power is then low against a shipped TDP)."""
    if cooling is not None and cooling not in COOLING:
        raise ValueError(f"cooling must be one of {sorted(COOLING)}")
    out = {"pkg_W": pkg_W, "dies": dies, "dram_dev_W": dram_dev_W, "die_mm2": die_mm2, "cooling": cooling,
           "status": "unknown", "feasible": None, "reasons": [], "notes": [], "static_included": static_included,
           "flux_basis": "封装功耗 ÷ 逻辑晶粒总面积（出货件同口径：TDP ÷ 晶粒面积）；平均值，不含热点"}
    if pkg_W is None or not dies:
        out["reasons"].append("无功耗（需能耗表）：热 / 供电未判定")
        return out
    if static_included is False:
        out["notes"].append("功耗未含整卡静态（能耗表无 idle_W / idle_frac_tdp）且不含稳压器损耗：低于同口径出货 TDP，"
                            "「超限」可信，「可建」余量不足时不可信「近似」")
    die_W = max(0.0, pkg_W - (dram_dev_W or 0.0)) / dies
    out["die_W"] = die_W
    area = die_mm2 * dies if die_mm2 else None
    flux = pkg_W / area if area else None
    out["flux_W_mm2"] = flux
    out["die_flux_W_mm2"] = die_W / die_mm2 if die_mm2 else None
    out["pd"] = {"vdd_V": PD_VDD_V, "I_A": die_W / PD_VDD_V,
                 "A_per_mm2": (die_W / PD_VDD_V / die_mm2) if die_mm2 else None, "judged": False}
    lim = {}
    for k, c in COOLING.items():
        lim[k] = {"pkg_ok": pkg_W <= c["pkg_W"] * (1 + 1e-9),
                  "flux_ok": None if flux is None else flux <= c["flux_W_mm2"] * (1 + 1e-9),
                  "pkg_W": c["pkg_W"], "flux_W_mm2": c["flux_W_mm2"],
                  "flux_ratio": None if flux is None else flux / c["flux_W_mm2"]}
    out["limits"] = lim
    if cooling is None:
        out["status"] = "unchecked"
        return out
    c, l = COOLING[cooling], lim[cooling]
    out["flux_ratio"] = l["flux_ratio"]
    if not l["pkg_ok"]:
        out["reasons"].append(f"封装功耗 {pkg_W:.0f} W > {cooling} 上限 {c['pkg_W']:g} W")
    if l["flux_ok"] is False:
        out["reasons"].append(f"热流密度 {flux:.2f} W/mm² > {cooling} 上限 {c['flux_W_mm2']:g} W/mm²")
    if flux is None:
        out["reasons"].append("晶粒面积未知：热流密度未判定")
    out["feasible"] = not (l["pkg_ok"] is False or l["flux_ok"] is False)
    out["status"] = "ok" if out["feasible"] and flux is not None else ("over" if not out["feasible"] else "unknown")
    out["src"] = c["src"]
    return out


def dram_dev_W(er: dict) -> float:
    """DRAM-device power per card from an energy report: (dram + hbm_static J) × the card's average W / total J."""
    j = er.get("J_by_action") or {}
    tot = sum(j.values())
    if not tot or er.get("avg_W_per_card") is None:
        return 0.0
    return er["avg_W_per_card"] * (j.get("dram", 0.0) + j.get("hbm_static", 0.0)) / tot
