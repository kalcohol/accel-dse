"""0.317: cost per unit of work (TCO) for one package (card) at its operating point — opt-in.

    $/M unit = (annual capex charge + annual energy cost) ÷ (units served per year) × 1e6

    capex per package   C = usd_good_pkg (0.310 chiplet economics: Σ good dies + packaging + test; HBM devices NOT
                            inside — only their interposer footprint) + memory $ (GB × $/GB, the only place memory
                            devices are priced) + system_usd_per_card (host / networking share; 0 in "accel")
    annual capex charge = C / life (straight line) or C · r / (1 − (1 + r)^−life) with capital_rate r > 0
    IT power            = util · (P_pkg + P_sys) + (1 − util) · (P_pkg,idle + P_sys)
                            P_pkg = the complete module-input package power (thermal.package_power, 0.315/0.316: card W
                            incl. DRAM devices + static floor + VR / PDN loss) at the capped operating point
                            P_pkg,idle = package_power of the static alone (same function); P_sys = system_W_per_card
    annual energy cost  = IT power × PUE × 8760 h × $/kWh        (PUE covers cooling / power delivery outside IT once)
    units per year      = thr (at the scenario's batch; SLO-violating points are already infeasible) × util × 3600 × 8760

Also reported: tok/s/W = thr ÷ P_pkg (module input, design metric) and per facility W (÷ (P_pkg + P_sys) · PUE).
Costs are manufacturing cost (no vendor margin) 「假设」 — the designer's view; a buyer's TCO would use prices.
Every default is labelled 来源 / 二手 / 推导 / 假设 below; the user's values win."""
from __future__ import annotations

HOURS_PER_YEAR = 8760.0

# memory $/GB defaults (used only when the user gives no mem_usd_per_GB for that type); value, tag, source
MEM_USD_PER_GB = {
    "HBM3E": (17.5, "二手", "HBM3E 长约 $17–18/GB（TrendForce 引 Fubon Research，2026-08-13，经 MoneyToday / GPUQuant 转述「二手」），取中值"),
    "HBM4": (31.5, "二手", "HBM4 $31–32/GB（同上，TrendForce 2026-08-13「二手」），取中值"),
    "LPDDR5X": (19.5, "推导", "LPDDR5X 2Q26 合约约 $19.3–19.8/GB（SemiAnalysis 约 $10/GB 基数 × TrendForce 2Q26 +93–98 %，经 Wccftech "
                             "转述「二手」）「推导」，取中值"),
}
# 0.318 additions — public, dated, labelled; LPDDR6 still unpriced (only a rumoured +20 % premium over LPDDR5X, no $/GB)
MEM_USD_PER_GB.update({
    "HBM3": (13.0, "二手", "HBM3 2025 合约 $13.00/GB（区间 12.0–13.6；Silicon Analysts 模型估计，基于 TrendForce 2024-05 所述 "
                         "2025 涨幅，低置信度「二手」；页面 2026-10 查）"),
    "HBM4E": (28.0, "二手", "HBM4E 2027 预测 $3.5/Gb = $28/GB（UBS，2026-06-22，经新浪财经转述「二手」）——是预测，不是成交价"),
    "LPDDR5": (10.0, "二手", "LPDDR5 合约约 $10/GB（SemiAnalysis，2026-05，经 Wccftech 转述「二手」；2Q26 涨价前水平，"
                           "原文 LPDDR5 / 5X 口径不严）"),
})
# 0.318: sourced $/GB range per type for the sensitivity view: (lo, hi, tag_lo, tag_hi, source)
MEM_RANGES = {
    "HBM3E": (10.28, 18.0, "二手", "二手",
              "HBM3E $10.28–18.0/GB：下端 2026-08-31 合约低值（GPUQuant「二手」）；上端 Fubon $17–18（TrendForce 2026-08-13「二手」）；"
              "Silicon Analysts 每栈成交 $12.5–13.9（首尔经济 2025-12-26「二手」）在区间内"),
    "HBM4": (15.56, 32.0, "二手", "二手",
             "HBM4 $15.56–32/GB：下端 SK hynix 对 NVIDIA 约 $560 / 36 GB 栈（首尔经济 2025-11-05，经 Silicon Analysts「二手」）；"
             "上端 Fubon $31–32（TrendForce 2026-08-13「二手」）；2027 供货预期约 $36（首尔经济 2026-10-01）不计入 2026 区间"),
    "LPDDR5X": (7.8, 21.0, "二手", "二手", "LPDDR5X $7.8–21/GB：长约下限 / 上限（Wccftech 2026-05「二手」）"),
    "HBM3": (12.0, 13.6, "二手", "二手", "HBM3 $12.0–13.6/GB（Silicon Analysts 2025 合约模型区间「二手」）"),
    "HBM4E": (28.0, 40.0, "二手", "推导",
              "HBM4E 2027 预测：下端 UBS $3.5/Gb = $28/GB「二手」；上端 台湾分析师 2027 HBM4 $4.16/Gb × HBM4E +20 %（三立新闻「二手」）"
              "≈ $5/Gb = $40/GB「推导」"),
    "LPDDR5": (10.0, 19.8, "二手", "推导", "LPDDR5 $10–19.8/GB：SemiAnalysis 约 $10「二手」→ × TrendForce 2Q26 +93–98 %「推导」上端"),
}
# LPDDR6: unpriced unless the user gives one (never silently 0); no sourced range
_BASE = {
    "life_years": 5.0, "pue": 1.54, "usd_per_kWh": 0.0891, "util": 1.0, "capital_rate": 0.0,
    "system_usd_per_card": 0.0, "system_W_per_card": 0.0,
}
_SRC = {
    "life_years": "5 年：Amazon 2025 起部分服务器 6 → 5 年（2025 10-K「来源」）；Meta 5.5、Alphabet 6（10-K「来源」）；取区间下端「假设」",
    "pue": "PUE 1.54：Uptime Institute 2025 全球数据中心调查加权平均（n = 681「来源」；近 5 年新建 1.48、> 20 MW 1.44）",
    "usd_per_kWh": "$0.0891/kWh：EIA 电力月报 Table 5.3 工业电价，截至 2026-07 滚动 12 个月 8.91 ¢「来源」（大型数据中心多按工业电价）",
    "util": "利用率 1.0：设计点按满载运行「假设」（Meta 24k H100 分析取 90 %「二手」）",
    "capital_rate": "资金成本 0：直线摊销「假设」（Meta 分析 WACC 9.33 %「二手」可填）",
    "system_usd_per_card": "系统分摊 $0：只算加速器封装 + 内存（accel 口径）",
    "system_W_per_card": "系统功耗 0 W（accel 口径）",
}
TCO_PRESETS = {
    "accel_2026": {**_BASE, "src": _SRC},
    "server_2026": {**_BASE, "system_usd_per_card": 7500.0, "system_W_per_card": 575.0,
                    "src": {**_SRC,
                            "system_usd_per_card": "$7,500 / 卡：SemiAnalysis H100 服务器约 $190k、含存储 / 网络等全口径 $250k / 8 GPU"
                                                   "「二手」→ (250k − 190k) ÷ 8「推导」；不含服务器内 CPU / 主机内存 / 机箱（下界）",
                            "system_W_per_card": "575 W / 卡：DGX H100 最大 10.2 kW「来源」− 8 × 700 W GPU TDP「来源」→ 4.6 kW ÷ 8「推导」"
                                                 "（铭牌最大值，上界）"}},
}
TCO_KEYS = tuple(_BASE)


def tco_params(tco) -> dict | None:
    """None | preset name | dict (``preset`` + TCO_KEYS overrides + optional ``mem_usd_per_GB``) → validated terms."""
    if tco is None:
        return None
    if isinstance(tco, str):
        tco = {"preset": tco}
    if not isinstance(tco, dict):
        raise ValueError(f"tco must be a preset name ({sorted(TCO_PRESETS)}) or an object")
    pr = tco.get("preset", "accel_2026")
    if pr not in TCO_PRESETS:
        raise ValueError(f"tco preset must be one of {sorted(TCO_PRESETS)}")
    out = {**TCO_PRESETS[pr], "preset": pr, "src": dict(TCO_PRESETS[pr]["src"]), "mem_prices": {}}
    for k, v in tco.items():
        if k == "preset":
            continue
        if k == "mem_usd_per_GB":
            out["mem_prices"] = _mem_prices(v)
            continue
        if k not in TCO_KEYS:
            raise ValueError(f"unknown tco key {k!r} (one of {TCO_KEYS + ('mem_usd_per_GB',)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1e7:
            raise ValueError(f"tco.{k} must be a number in [0, 1e7)")
        out[k] = float(v)
        out["src"][k] = f"用户输入 {v:g}"
    if not out["life_years"] > 0:
        raise ValueError("tco.life_years must be > 0")
    if not out["pue"] >= 1:
        raise ValueError("tco.pue must be ≥ 1 (total facility ÷ IT power)")
    if not 0 < out["util"] <= 1:
        raise ValueError("tco.util must be in (0, 1]")
    if not out["capital_rate"] < 1:
        raise ValueError("tco.capital_rate must be < 1 (a fraction per year)")
    return out


def _mem_prices(d) -> dict:
    from .. import mem_catalog as mc
    if not isinstance(d, dict):
        raise ValueError("tco.mem_usd_per_GB must be an object {mem_type: $/GB}")
    out = {}
    for k, v in d.items():
        try:
            t = mc.normalize_type(k)
        except (ValueError, KeyError):
            raise ValueError(f"tco.mem_usd_per_GB: unknown memory type {k!r}") from None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("tco.mem_usd_per_GB values must be ≥ 0 ($/GB)")
        out[t] = float(v)
    return out


def mem_price(p: dict, mem_type: str, user: dict | None = None):
    """$/GB and its provenance: user (DSE mem_usd_per_GB, then tco.mem_usd_per_GB) → sourced default → None."""
    for d, tag in ((user or {}, "用户输入"), (p.get("mem_prices") or {}, "用户输入")):
        if mem_type in d:
            return d[mem_type], tag
    if mem_type in MEM_USD_PER_GB:
        v, tag, _ = MEM_USD_PER_GB[mem_type]
        return v, tag
    return None, None


def capex_per_year(p: dict, capex_usd: float) -> float:
    L, r = p["life_years"], p["capital_rate"]
    if r > 0:
        return capex_usd * r / (1 - (1 + r) ** -L)
    return capex_usd / L


def tco_point(p: dict, *, pkg_usd: float, mem_usd: float, thr: float, pkg_W: float, pkg_W_idle: float,
              unit: str = "token") -> dict:
    """One design point.  ``pkg_usd`` excludes memory devices; ``mem_usd`` is the only memory term; ``pkg_W`` is the
    complete package power at the operating point (never card W)."""
    sys_usd, sys_W = p["system_usd_per_card"], p["system_W_per_card"]
    capex = pkg_usd + mem_usd + sys_usd
    cap_yr = capex_per_year(p, capex)
    it_W = p["util"] * (pkg_W + sys_W) + (1 - p["util"]) * (pkg_W_idle + sys_W)
    kwh_yr = it_W * p["pue"] * HOURS_PER_YEAR / 1000.0
    en_yr = kwh_yr * p["usd_per_kWh"]
    units_yr = thr * p["util"] * 3600.0 * HOURS_PER_YEAR
    tot = cap_yr + en_yr
    per = tot / units_yr * 1e6 if units_yr > 0 else float("inf")
    share = lambda x: x / tot if tot > 0 else 0.0    # noqa: E731
    f = cap_yr / capex if capex > 0 else 0.0
    return {"usd_per_Munit": per, "unit": unit, "capex_usd": capex, "capex_usd_per_yr": cap_yr,
            "energy_usd_per_yr": en_yr, "kWh_per_yr": kwh_yr, "it_W": it_W, "facility_W": it_W * p["pue"],
            "share": {"package": share(pkg_usd * f), "memory": share(mem_usd * f), "system": share(sys_usd * f),
                      "energy": share(en_yr)},
            "units_per_s_per_W": thr / pkg_W if pkg_W > 0 else None,
            "units_per_s_per_facility_W": thr / ((pkg_W + sys_W) * p["pue"]) if pkg_W + sys_W > 0 else None}


def basis_note(p: dict) -> str:
    s = p["src"]
    return ("$/M 单位（0.317，TCO，可选）：(年化资本 + 年电费) ÷ 年服务量；资本 = 每好封装成本（不含 HBM 器件）+ 内存 GB × $/GB"
            "（内存只在此计一次）+ 系统分摊；电 = 完整封装功耗（与热判定 / 功耗预算同一函数）× PUE × $/kWh；"
            f"{s['life_years']}；{s['pue']}；{s['usd_per_kWh']}；{s['util']}；{s['capital_rate']}；{s['system_usd_per_card']}；"
            f"{s['system_W_per_card']}；成本按制造成本（无厂商毛利）「假设」；只有可建点进 $/M 前沿")


EVAL_KEYS = ("cost", "pkg_usd", "d0_per_cm2")


def eval_tco(spec, *, scn, r, energy: dict, die_mm2, unit: str) -> dict:
    """0.317: TCO of one evaluated package.  ``spec`` = tco_params input plus eval-only keys: ``pkg_usd`` (package $
    without memory devices, e.g. a known part) or ``cost`` (COST_PRESETS name / object, priced on the shoreline die with
    Poisson yield at ``d0_per_cm2``, default 0.10 /cm² 「二手」 N5).  Power = the complete package power at the
    operating point (capped when the TDP cap binds), throughput likewise."""
    import math
    from .. import mem_catalog as mc
    from .energy_dse import _cost_params, _hbm_footprint, package_cost
    from .hardware import default_mem
    from .powercap import static_W
    from .thermal import package_power
    if isinstance(spec, str):
        spec = {"preset": spec}
    if not isinstance(spec, dict):
        raise ValueError("tco must be a preset name or an object")
    p = tco_params({k: v for k, v in spec.items() if k not in EVAL_KEYS})
    pk = scn.package_cards if scn.d2d_enabled else 1
    mid = scn.mem_id or default_mem(scn.chip)
    sp = mc.parse_mem_id(mid)
    notes, miss = [], []
    cp, d0 = None, None
    if spec.get("pkg_usd") is not None:
        v = spec["pkg_usd"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1e8:
            raise ValueError("tco.pkg_usd must be a number in [0, 1e8)")
        pkg_usd, pkg_src = float(v), "用户输入（不含内存器件）"
    elif spec.get("cost") is not None:
        cp = _cost_params(spec["cost"])
        d0 = spec.get("d0_per_cm2", 0.1)
        if isinstance(d0, bool) or not isinstance(d0, (int, float)) or not 0 <= d0 < 10:
            raise ValueError("tco.d0_per_cm2 must be a number in [0, 10)")
        if not die_mm2:
            miss.append("晶粒面积（岸线报告无面积）")
            pkg_usd, pkg_src = None, None
        else:
            y = math.exp(-die_mm2 * d0 / 100.0)
            cc = package_cost(cp, die_mm2, pk, y, pk * die_mm2 + _hbm_footprint(mid))
            pkg_usd, pkg_src = cc["usd_good_pkg"], f"{cp['preset']} × 岸线晶粒 {die_mm2:.0f} mm²，D0 {d0:g}，良率 {y:.3f}"
    else:
        raise ValueError("tco in eval needs pkg_usd (package $ without memory) or cost (a COST_PRESETS name)")
    pr, ptag = mem_price(p, sp.mem_type)
    if pr is None:
        miss.append(f"{sp.mem_type} $/GB（无公开价，请填 tco.mem_usd_per_GB）")
    ew = energy.get("avg_W_per_card")
    if ew is None:
        miss.append("功耗（能耗表）")
    if miss:
        return {"usd_per_Munit": None, "missing": miss, "preset": p["preset"]}
    st = static_W(energy) * pk
    iw = energy["idle_W_used"] * pk if energy.get("idle_W_used") is not None else None
    thr = r.per_card * pk
    card_W = ew * pk
    pc = energy.get("power_cap") or {}
    if pc.get("capped"):                    # the sustained, capped point: throughput × factor at the capped card W
        thr *= pc["throughput_factor"]
        card_W = pc["cap_card_W"] * pk
        notes.append("按功耗封顶后的工作点计（吞吐 × 封顶系数，封装功耗 = 封顶点）")
    pw = package_power(card_W, die_mm2, pk, st, iw)["pkg_W"]
    idle = package_power(st, die_mm2, pk, st, iw)["pkg_W"]
    mem_usd = sp.capacity_GB * pk * pr
    t = tco_point(p, pkg_usd=pkg_usd, mem_usd=mem_usd, thr=thr, pkg_W=pw, pkg_W_idle=idle, unit=unit)
    t.update(preset=p["preset"], pkg_usd=pkg_usd, pkg_src=pkg_src, mem_usd=mem_usd, mem_GB=sp.capacity_GB * pk,
             mem_usd_per_GB=pr, mem_price_src=ptag, pkg_W=pw, pkg_W_idle=idle, thr=thr, src=p["src"],
             notes=notes + [basis_note(p)])
    # 0.318: one-at-a-time sensitivity over the sourced ranges (cost inputs only when priced from a cost preset)
    from .tco_sens import tornado
    rw = {"mm2": die_mm2, "dies": pk, "si_mm2": (die_mm2 or 0.0) * pk, "mem_id": mid, "mem_type": sp.mem_type,
          "mem_GB": sp.capacity_GB * pk, "thr": thr, "unit": unit, "tco": t}
    t["sens"] = tornado(rw, {"tco": p, "cost": cp, "d0": d0, "prices": None})
    return t
