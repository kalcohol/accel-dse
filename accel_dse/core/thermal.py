"""0.313: buildability from power density and package power (opt-in limits; the figures are always reported).
0.314: audited against shipped parts (MODEL §259) — one flux definition on both sides, limits re-sourced.

Flux judged (``flux_W_mm2``) = package (card) power ÷ the package's logic-die area (dies × die mm²): the definition a
shipped part's figure can be formed in (TDP / TBP ÷ die area; neither its HBM power nor its VR loss is published).
0.313 judged the die-level figure (card W − DRAM-device W, ÷ area) against a limit equal to H100's card-level figure
— two definitions.  The die-level figure stays reported (``die_flux_W_mm2``; die heat = (card W − DRAM devices) ÷
dies, equal split 「近似」; the host-side DRAM PHY stays on the die).  Both are averages over the die; hot spots are
not modelled — a shipped part's average already carries its own hot-spot ratio, so the comparison holds for a design
of similar uniformity 「近似」.

0.315 (MODEL §260): the package power judged is complete — ``package_power``:
  card W (energy_report: dynamic MAC / vector / SRAM / SLC / NoC (+ wire) / clock / DRAM-device / host DRAM-PHY / D2D /
  link, SRAM (+ NoC router) leakage, HBM standby; + the table's idle_W when set)
  + whole-card static floor when the table has no idle figure: STATIC_FLOOR_W_MM2 × logic-die area × VR_EFF (delivered
    side), less the static already charged (SRAM / NoC leakage, HBM standby — a measured idle floor contains them;
    the same no-double-count rule as energy.py) — always on (0.314: only when the table had idle)
  + voltage-regulator / PDN loss: delivered W × (1 / VR_EFF − 1) (an idle_W figure is board-level already, so it carries
    no second loss).  A shipped TDP / TBP is the module's input power (OCP OAM r2.0 §8.4: TDP at the 44–59.5 V
    input) — the judged package power now has the same scope.
  Where each part sits: HBM / DRAM devices (dram + hbm_static) are in the package, beside the die (out of the die heat);
  the host-side DRAM PHY is on the logic die (in the die heat); VR loss is on the module (in the package power, out
  of the die heat).  die_W = (card W + static add − DRAM devices) ÷ dies.
  Remaining bound (not a blanket caveat): the floor is measured at idle temperature (MI300X idle 34–47 °C 「二手」 →
  STATIC_T_IDLE_C 40 「近似」); leakage at operating temperature is higher.  ``pkg_W_hot`` = the static (charged +
  floor) × leak_temp_factor(85) / leak_temp_factor(40) (the TSMC 3 nm SRAM ratio applied to all static, an upper
  bound 「假设」); an "ok" row whose hot bound is over carries ``hot_ok`` False and a note — only those rows are marginal.

Limits (``COOLING``): rule = buildable when within (a) the highest average flux / package power of a shipped part with
that cooling (card-level: TDP ÷ die area, judged on ``flux_W_mm2``) or (b) the HIR 2021 Table 1 *thermal-budget*
figure (50 °C junction-to-inlet, 4 cm² chip, demonstrated hardware — a chip heat flux, judged on ``die_flux_W_mm2``,
0.315: each precedent against its own definition; 0.314 took the larger number for one card-level figure, which put
HIR's chip-level 1.70 against card-level power).  The HIR *uniform-heat-flux* idealisations (air 0.85, water 2.50)
are not used.
  air_oam  1000 W: AMD Instinct MI350X, OAM, air-cooled, 1000 W TBP (AMD product page / ROCm MI350 docs) 「来源」;
           Intel Gaudi 3 HL-325L OAM 900 W air (Hot Chips 2024, Intel) 「来源」.  (0.313: 600 W, the OCP OAM r2.0
           air recommendation — exceeded by both shipped OAM parts.)
  air_hgx  1000 W: HGX B200, "configurable up to 1,000 W", air-cooled HGX B200 racks (NVIDIA Blackwell datasheet) 「来源」.
  air flux 0.86 W/mm²: H100 / H200 SXM 700 W air-cooled (Dell XE9680 spec sheet: 8 × H200 700 W, air, 35 °C inlet
           「来源」) on GH100 814 mm² 「二手」 → 0.86 「推导」 — above HIR's 0.55 (thermal budget 50 °C) 「来源」.
  liquid   1400 W: AMD Instinct MI355X, OAM, direct liquid cooling, 1400 W TBP (AMD MI355X datasheet) 「来源」;
           HGX B300 1400 W bin, liquid (Supermicro) 「二手」.
  liquid flux 0.96 W/mm² card-level: MI355X 1400 W on ≈ 1464 mm² of IOD footprint 「二手」 「推导」; die heat 1.70 W/mm²:
           HIR 2021 Table 1 water-cooled module cold plate at a 50 °C budget, 4 cm² chip 「来源」.  (0.313: 2.5, the
           uniform-flux idealisation; 0.314: 1.70 card-level.)
  air die heat 0.55 W/mm²: HIR 2021 Table 1 air, thermal budget 50 °C 「来源」 (below the card-level 0.86, which decides).
Power delivery: the die current at an assumed 0.75 V core rail is reported; no public per-bump current limit for
current micro-bump / hybrid-bond pitches was found (the 0.2 A per C4 bump guideline is a legacy figure for
50–100 µm bumps 「二手」), so power delivery is reported, not judged.
"""

COOLING = {
    "air_oam": {"pkg_W": 1000.0, "flux_W_mm2": 0.86, "die_flux_W_mm2": 0.55,
                "src": "MI350X OAM 风冷 1000 W TBP（AMD）「来源」、Gaudi 3 HL-325L OAM 风冷 900 W（Hot Chips 2024）「来源」；"
                       "热流 0.86 W/mm² = H100/H200 SXM 风冷 700 W ÷ 814 mm²「推导」（高于 HIR 2021 表 1 风冷 0.55 W/mm²「来源」）"},
    "air_hgx": {"pkg_W": 1000.0, "flux_W_mm2": 0.86, "die_flux_W_mm2": 0.55,
                "src": "HGX B200 风冷，最高 1000 W / GPU（NVIDIA Blackwell 数据手册）「来源」；热流同风冷 0.86 W/mm²「推导」"},
    "liquid": {"pkg_W": 1400.0, "flux_W_mm2": 0.96, "die_flux_W_mm2": 1.70,
               "src": "MI355X OAM 直接液冷 1400 W TBP（AMD 数据手册）「来源」；封装热流 0.96 W/mm² = 1400 W ÷ 1464 mm²"
                      "「推导」；晶粒热流 1.70 W/mm²：HIR 2021 表 1 水冷板、50 °C 温度预算、4 cm² 芯片「来源」"},
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
# 0.315: module voltage-regulator / PDN efficiency, 48 V input → core rails, at load 「近似」 (range 0.77–0.90):
VR_EFF = 0.85
VR_SRC = ("48 V → 1 V 单级 VPD 变换器满载系统效率 84.4 %（1.0 V / 400 A）、77.2 %（0.65 V / 600 A）（ECCE 2024，"
          "doi 10.1109/ECCE55643.2024.10861778）「来源」；两级 48 → 1.8 V 母线级 ≤ 95 %（APEC 2024，doi "
          "10.1109/APEC48139.2024.10509453）「来源」 × 第二级；OCP OAM r2.0 §8.4：TDP 为 44–59.5 V 输入侧模块功耗「来源」 → 0.85「近似」")
# 0.315: whole-card static floor, card-level W per mm² of logic-die footprint (idle, no workload, idle clocks):
STATIC_FLOOR_W_MM2 = 0.088
STATIC_SRC = ("H100 SXM 无 CUDA 上下文空闲 71.8 W（lo0 tech blog）「二手」÷ GH100 814 mm²「二手」= 0.088；MI300X 空闲（SCLK 132 MHz、"
              "0 % 利用）131–133 W socket（AMD 技术文章 rocm-smi 输出）「来源」÷ 4 × IOD 370 mm²「二手」= 0.089 → 0.088 W/mm²「推导」"
              "（两家一致；含 HBM 自刷新与非核心，故扣除已计的 SRAM / NoC 泄漏与 HBM 待机）")
STATIC_T_IDLE_C, STATIC_T_HOT_C = 40.0, 85.0    # 「近似」 MI300X idle junction 34–47 °C 「二手」; 85 °C hot reference


def cooling_names() -> tuple:
    return tuple(COOLING)


def _hot_k() -> float:
    from .energy import leak_temp_factor
    return leak_temp_factor(STATIC_T_HOT_C) / leak_temp_factor(STATIC_T_IDLE_C)


def package_power(card_W: float | None, die_mm2: float | None, dies: int = 1, static_W: float = 0.0,
                  idle_W: float | None = None) -> dict:
    """0.315: complete package (module-input) power from the modelled card W (energy_report scope, see module doc).
    ``static_W`` = static already in card_W (powercap.static_W: idle + SRAM / NoC leakage + HBM standby);
    ``idle_W`` = the table's idle figure when one is charged (board-level: no floor, no second VR loss)."""
    out = {"card_W": card_W, "vr_eff": VR_EFF, "static_floor_W_mm2": STATIC_FLOOR_W_MM2, "static_add_W": 0.0,
           "vr_loss_W": 0.0, "pkg_W": card_W, "pkg_W_hot": card_W, "static_basis": "none"}
    if card_W is None:
        return out
    area = die_mm2 * dies if die_mm2 else None
    static_W = static_W or 0.0
    add, board = 0.0, 0.0
    if idle_W is not None:
        out["static_basis"], board = "idle", idle_W
    elif area:
        out["static_basis"] = "floor"
        out["static_floor_W"] = STATIC_FLOOR_W_MM2 * area * VR_EFF            # delivered side
        add = max(0.0, out["static_floor_W"] - static_W)
    deliv = card_W + add
    vr = max(0.0, deliv - board) * (1.0 / VR_EFF - 1.0)
    stat = static_W + add                                                   # all static, delivered side
    out.update(static_add_W=add, delivered_W=deliv, vr_loss_W=vr, pkg_W=deliv + vr, static_W=stat,
               pkg_W_hot=deliv + vr + stat * (_hot_k() - 1.0) / VR_EFF)
    return out


def thermal_report(pkg_W: float | None, die_mm2: float | None, dies: int = 1, dram_dev_W: float = 0.0,
                   cooling: str | None = None, static_W: float = 0.0, idle_W: float | None = None,
                   basis: str = "model") -> dict:
    """Package power and heat flux vs a cooling preset.  ``cooling`` None = report only (``feasible`` None, status
    "unchecked"); a preset name = judged (``feasible`` False when the package power is over, or the flux is over both
    the shipped card-level limit and the HIR die-heat limit).
    ``basis`` "model" (default): ``pkg_W`` is the modelled card W and is completed by ``package_power`` (static floor,
    VR loss; 0.315); "package": ``pkg_W`` is already a module-input power (a shipped TDP / TBP) and is judged as is."""
    if cooling is not None and cooling not in COOLING:
        raise ValueError(f"cooling must be one of {sorted(COOLING)}")
    if basis not in ("model", "package"):
        raise ValueError("basis must be 'model' or 'package'")
    if basis == "model":
        pp = package_power(pkg_W, die_mm2, dies, static_W, idle_W)
    else:
        pp = {"card_W": None, "pkg_W": pkg_W, "pkg_W_hot": pkg_W, "vr_loss_W": None, "static_add_W": 0.0,
              "static_basis": "package", "delivered_W": None}
    P = pp["pkg_W"]
    out = {"pkg_W": P, "card_W": pp["card_W"], "power": pp, "dies": dies, "dram_dev_W": dram_dev_W, "die_mm2": die_mm2,
           "cooling": cooling, "status": "unknown", "feasible": None, "reasons": [], "notes": [],
           "static_included": pp["static_basis"] in ("idle", "floor", "package"), "static_basis": pp["static_basis"],
           "vr_loss_W": pp["vr_loss_W"], "static_add_W": pp["static_add_W"], "pkg_W_hot": pp["pkg_W_hot"],
           "hot_ok": None,
           "flux_basis": "封装功耗（含整卡静态与稳压器损耗，模块输入侧）÷ 逻辑晶粒总面积（出货件同口径：TDP ÷ 晶粒面积）；"
                         "晶粒热流 = 晶粒发热（扣 DRAM 器件与稳压器损耗）÷ 面积；平均值，不含热点"}
    if P is None or not dies:
        out["reasons"].append("无功耗（需能耗表）：热 / 供电未判定")
        return out
    if pp["static_basis"] == "none":
        out["notes"].append("晶粒面积未知：整卡静态底功耗（∝ 面积）未计入，功耗偏低「近似」")
    # die heat: delivered power less the DRAM devices (VR loss is on the module, not through the die)
    deliv = pp["delivered_W"] if pp.get("delivered_W") is not None else P
    die_W = max(0.0, deliv - (dram_dev_W or 0.0)) / dies
    out["die_W"] = die_W
    area = die_mm2 * dies if die_mm2 else None

    def _lim(Pp, dW):
        fl = Pp / area if area else None
        dfl = dW / die_mm2 if die_mm2 else None
        L = {}
        for k, c in COOLING.items():
            rc = None if fl is None else fl / c["flux_W_mm2"]
            rd = None if dfl is None else dfl / c["die_flux_W_mm2"]
            r = None if rc is None else (min(rc, rd) if rd is not None else rc)
            L[k] = {"pkg_ok": Pp <= c["pkg_W"] * (1 + 1e-9), "flux_ok": None if r is None else r <= 1 + 1e-9,
                    "pkg_W": c["pkg_W"], "flux_W_mm2": c["flux_W_mm2"], "die_flux_W_mm2": c["die_flux_W_mm2"],
                    "flux_ratio": r, "card_ratio": rc, "die_ratio": rd}
        return fl, dfl, L
    flux, dflux, lim = _lim(P, die_W)
    out["flux_W_mm2"], out["die_flux_W_mm2"] = flux, dflux
    out["pd"] = {"vdd_V": PD_VDD_V, "I_A": die_W / PD_VDD_V,
                 "A_per_mm2": (die_W / PD_VDD_V / die_mm2) if die_mm2 else None, "judged": False}
    out["limits"] = lim
    if cooling is None:
        out["status"] = "unchecked"
        return out
    c, l = COOLING[cooling], lim[cooling]
    out["flux_ratio"] = l["flux_ratio"]
    if not l["pkg_ok"]:
        out["reasons"].append(f"封装功耗 {P:.0f} W > {cooling} 上限 {c['pkg_W']:g} W")
    if l["flux_ok"] is False:
        out["reasons"].append(f"热流密度 {flux:.2f} W/mm² > {cooling} 上限 {c['flux_W_mm2']:g} W/mm²"
                              + (f"（晶粒 {dflux:.2f} > {c['die_flux_W_mm2']:g}）" if dflux is not None else ""))
    if flux is None:
        out["reasons"].append("晶粒面积未知：热流密度未判定")
    out["feasible"] = not (l["pkg_ok"] is False or l["flux_ok"] is False)
    out["status"] = "ok" if out["feasible"] and flux is not None else ("over" if not out["feasible"] else "unknown")
    out["src"] = c["src"]
    if basis == "model" and out["status"] == "ok":
        Ph = pp["pkg_W_hot"]
        hW = die_W + (Ph - P) * VR_EFF / dies
        _, _, Lh = _lim(Ph, hW)
        out["hot_ok"] = bool(Lh[cooling]["pkg_ok"] and Lh[cooling]["flux_ok"] is not False)
        if not out["hot_ok"]:
            out["notes"].append(f"余量不足：静态按 {STATIC_T_HOT_C:g} °C 泄漏上界（× {_hot_k():.2f}「假设」）时封装 {Ph:.0f} W "
                                f"超 {cooling} 上限——此点「可建」依赖工作温度下的泄漏")
    return out


def dram_dev_W(er: dict) -> float:
    """DRAM-device power per card from an energy report: (dram + hbm_static J) × the card's average W / total J."""
    j = er.get("J_by_action") or {}
    tot = sum(j.values())
    if not tot or er.get("avg_W_per_card") is None:
        return 0.0
    return er["avg_W_per_card"] * (j.get("dram", 0.0) + j.get("hbm_static", 0.0)) / tot
