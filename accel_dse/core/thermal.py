"""0.313: buildability from power density and package power (opt-in limits; the figures are always reported).

Heat that the die's cooling path must carry = package (card) power minus the DRAM devices' own power (HBM / LPDDR
array access + standby / refresh: it is dissipated in the stacks, not the logic die); the host-side DRAM PHY stays
on the die.  Per die = that ÷ dies in the package (equal split 「近似」); the whole-card static / idle figure is counted
on the die (conservative 「近似」).

Cooling presets (``COOLING``) — package power and die heat flux limits:
  air_oam  600 W / module: OCP OAM r2.0 base spec ("an air-cooled solution recommends TDP ≤ 600 W") 「来源」
           (OAM v1.5: ≈ 450 W in an 8-module shadowed layout 「来源」);
           0.85 W/mm²: air-cooled limit 0.55–0.85 W/mm² for a ≈ 4 cm² die at a 50 °C budget, IEEE Heterogeneous
           Integration Roadmap 2021 ch. 20 (thermal) 「来源」 — upper end.
  air_hgx  1000 W / GPU: HGX B200 air-cooled 1000 W configuration (NVIDIA Blackwell datasheet; Lenovo HGX B200
           1000 W product guide) 「二手」; HGX B300 air 1100 W (8U) 「二手」; flux as air_oam.
  liquid   1400 W / GPU: HGX B300 1400 W bin needs direct liquid cooling (Supermicro HGX B300 datasheet) 「二手」;
           2.5 W/mm²: single-phase water cold plate 1.7–2.5 W/mm² (HIR 2021 ch. 20) 「来源」 — upper end.
Known parts (consistency, 「二手」): H100 SXM 700 W air on 814 mm² → ≤ 0.86 W/mm² card-level (HBM included, so the die
figure is lower) — at the air flux limit; B200 1000 W air on two reticle-class dies (≤ 858 mm² each) → ≥ 0.58 W/mm².
Power delivery: the die current at an assumed 0.75 V core rail is reported; no public per-bump current limit for
current micro-bump / hybrid-bond pitches was found (the 0.2 A per C4 bump guideline is a legacy figure for
50–100 µm bumps 「二手」), so power delivery is reported, not judged.
"""

COOLING = {
    "air_oam": {"pkg_W": 600.0, "flux_W_mm2": 0.85,
                "src": "OCP OAM r2.0：风冷建议 ≤ 600 W「来源」；IEEE HIR 2021 第 20 章：风冷 0.55–0.85 W/mm²（取上限）「来源」"},
    "air_hgx": {"pkg_W": 1000.0, "flux_W_mm2": 0.85,
                "src": "HGX B200 风冷 1000 W / GPU（NVIDIA 数据手册、Lenovo 产品指南）「二手」；热流密度同风冷上限 0.85 W/mm²「来源」"},
    "liquid": {"pkg_W": 1400.0, "flux_W_mm2": 2.5,
               "src": "HGX B300 1400 W 档需直接液冷（Supermicro 数据手册）「二手」；单相水冷板 1.7–2.5 W/mm²（HIR 2021）取上限「来源」"},
}
KNOWN_PARTS = (
    {"part": "H100 SXM", "W": 700.0, "die_mm2": 814.0, "dies": 1, "cooling": "air", "src": "NVIDIA 数据手册 / 白皮书「二手」"},
    {"part": "B200 (HGX)", "W": 1000.0, "die_mm2": 858.0, "dies": 2, "cooling": "air",
     "src": "两个光罩级晶粒（≤ 858 mm²）「二手」：热流 ≥ 0.58 W/mm²"},
)
PD_VDD_V = 0.75          # 「假设」 core rail for the reported die current only (not judged)


def cooling_names() -> tuple:
    return tuple(COOLING)


def thermal_report(pkg_W: float | None, die_mm2: float | None, dies: int = 1, dram_dev_W: float = 0.0,
                   cooling: str | None = None) -> dict:
    """Package power and per-die heat flux vs a cooling preset.  ``cooling`` None = report only (``feasible`` None,
    status "unchecked"); a preset name = judged (``feasible`` False when the package power or the flux exceeds it)."""
    if cooling is not None and cooling not in COOLING:
        raise ValueError(f"cooling must be one of {sorted(COOLING)}")
    out = {"pkg_W": pkg_W, "dies": dies, "dram_dev_W": dram_dev_W, "die_mm2": die_mm2, "cooling": cooling,
           "status": "unknown", "feasible": None, "reasons": []}
    if pkg_W is None or not dies:
        out["reasons"].append("无功耗（需能耗表）：热 / 供电未判定")
        return out
    die_W = max(0.0, pkg_W - (dram_dev_W or 0.0)) / dies
    out["die_W"] = die_W
    flux = die_W / die_mm2 if die_mm2 else None
    out["flux_W_mm2"] = flux
    out["pd"] = {"vdd_V": PD_VDD_V, "I_A": die_W / PD_VDD_V,
                 "A_per_mm2": (die_W / PD_VDD_V / die_mm2) if die_mm2 else None, "judged": False}
    lim = {}
    for k, c in COOLING.items():
        lim[k] = {"pkg_ok": pkg_W <= c["pkg_W"] * (1 + 1e-9),
                  "flux_ok": None if flux is None else flux <= c["flux_W_mm2"] * (1 + 1e-9),
                  "pkg_W": c["pkg_W"], "flux_W_mm2": c["flux_W_mm2"]}
    out["limits"] = lim
    if cooling is None:
        out["status"] = "unchecked"
        return out
    c, l = COOLING[cooling], lim[cooling]
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
