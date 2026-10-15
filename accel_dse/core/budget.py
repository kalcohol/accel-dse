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
                  replica silicon limit (cards × die).

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
_PHY_DENS = ("mm2_per_dq_hbm", "mm2_per_dq_lpddr", "mm2_per_d2d_unit", "mm2_per_GBps_c2c")

# 0.149: cited density presets per process node.  Each value: (number, status, source).  status 「来源」 = read from
# the cited text; 「推导」 = arithmetic on cited figures (the scaling step itself is an assumption); 「二手」 = number
# reported by a secondary source, original closed; 「假设」 = no public figure found.  All are upper/lower bounds as
# noted, and every field stays overridable (a user-entered budget field wins over the preset).
_TPUV1 = "Jouppi et al., ISCA 2017 (arXiv 1704.04760) §2 / Fig. 2; die < 330 mm² per Jouppi et al., ISCA 2021 Table 1"
_TPUV4I = "Jouppi et al., 'Ten Lessons From Three Generations Shaped Google's TPUv4i', ISCA 2021, Fig. 6 / §7.A / Table 1"
_BITCELL = ("TSMC N7 HD 6T bitcell 0.027 µm² (Wu et al., IEDM 2016); N5 0.021 µm² (Yeap et al., IEDM 2019; "
            "WikiChip / SemiAnalysis); N5 logic density 1.8× N7 (TSMC, WikiChip 'TSMC Details 5 nm')")
_PHY = {
    "mm2_per_dq_hbm": (14.0 / 1024, "二手/假设", "≈ 14 mm² per 1024-bit HBM3 PHY: public H100 die-shot estimates (secondary, "
                       "unverified); lower bound RX-only 0.0056 mm²/DQ (Chae et al., ISSCC 2023, 4 nm HBM3, secondary)"),
    "mm2_per_dq_lpddr": (0.03, "假设", "no public LPDDR5X PHY area found (A-SSCC 2024 4 nm 9.6 Gb/s PHY reports none in the "
                         "abstract) — ≈ 1.9 mm² per x64 「假设」"),
    "mm2_per_d2d_unit": (0.3888 * 1.043, "来源", "UCIe x64 Advanced module 388.8 µm × 1043 µm (45 µm bump pitch), Hot Chips "
                         "2023 UCIe tutorial 'Electrical, Form Factor and Compliance' table; UCIe-S x16 is 571.5 × 1540 µm "
                         "= 0.88 mm² at 24/32 GT/s — override for UCIe-S / BoW"),
    "mm2_per_GBps_c2c": (1.806 * 0.825 / 4 / 14.0, "二手", "quad 112 Gb/s DSP SerDes 1806 × 825 µm in 5 nm (Varzaghani et "
                         "al., VLSI 2022; area as reported secondary, paper closed) → 0.37 mm² per lane ÷ 14 GB/s per lane per "
                         "direction"),
}
DENSITY_PRESETS: dict[str, dict] = {
    "n28_tpuv1": {"label": "28 nm（TPUv1 平面图）", "fields": {
        "mm2_per_mib_sram": (330 / 3 / 24, "来源", _TPUV1 + ": 24 MiB Unified Buffer ≈ 1/3 of the die (upper bound)"),
        "mm2_per_mib_slc": (330 / 3 / 24, "来源", "= SRAM (same macro class) 「假设」 on " + _TPUV1),
        "mm2_per_kmac": (330 / 4 / 64, "来源", _TPUV1 + ": 256×256 int8 MXU = 1/4 of the die (upper bound; int8 MAC)"),
        **_PHY}},
    "n7_tpuv4i": {"label": "7 nm（TPUv4i 平面图）", "fields": {
        "mm2_per_mib_sram": (400 * 0.28 / 128, "来源", _TPUV4I + ": 128 MiB CMEM = 28 % of a < 400 mm² die (upper bound)"),
        "mm2_per_mib_slc": (400 * 0.28 / 128, "来源", "= CMEM density (" + _TPUV4I + ")"),
        "mm2_per_kmac": (400 * 0.11 / 64, "来源", _TPUV4I + ": 4 × 128×128 bf16 MXUs = 11 % of a < 400 mm² die (upper bound)"),
        **_PHY}},
    "n5_derived": {"label": "5 nm / N3E（由 7 nm 推导）", "fields": {
        "mm2_per_mib_sram": (400 * 0.28 / 128 * 0.021 / 0.027, "推导", "TPUv4i CMEM × bitcell 0.021/0.027; " + _BITCELL
                             + "; array-only lower bound 31.8 Mib/mm² = 0.25 mm²/MiB (WikiChip, IEDM 2022 N3E)"),
        "mm2_per_mib_slc": (400 * 0.28 / 128 * 0.021 / 0.027, "推导", "= SRAM density (TPUv4i CMEM × bitcell ratio)"),
        "mm2_per_kmac": (400 * 0.11 / 64 / 1.8, "推导", "TPUv4i MXU ÷ 1.8 logic density; " + _BITCELL),
        **_PHY}},
}


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
    if scn.layout.cards > 1 and b.mm2_per_GBps_c2c is not None:
        terms["c2c_phy"] = scn.link.GBps * b.mm2_per_GBps_c2c
    return {"mm2": sum(terms.values()), "terms": terms, "notes": notes}


def full_area(scn, b: Budget) -> dict:
    """Die area proxy incl. interface PHYs (0.149): area_estimate(chip) + phy_area(scenario)."""
    a = area_estimate(scn.chip, b)
    p = phy_area(scn, b)
    return {**a, "mm2": None if a["mm2"] is None else a["mm2"] + p["mm2"], "terms": {**a["terms"], **p["terms"]},
            "phy_notes": p["notes"]}


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
    # 0.149: interface PHY area (optional; None = not counted).  Presets in DENSITY_PRESETS carry the sources.
    mm2_per_dq_hbm: float | None = None       # per HBM data pin (bus bit) of the DRAM interface
    mm2_per_dq_lpddr: float | None = None     # per data pin of any non-HBM DRAM interface (LPDDR / DDR / GDDR)
    mm2_per_d2d_unit: float | None = None     # per D2D unit (UCIe module / BoW slice) on the die
    mm2_per_GBps_c2c: float | None = None     # per GB/s (per direction) of card-to-card SerDes link

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1e9:
                raise ValueError(f"budget.{f.name} must be a finite number in [0, 1e9) or null")
            if f.name in _LIMITS and v <= 0:
                raise ValueError(f"budget.{f.name} must be > 0")
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
    for name, qty, dens in (("sram", chip.sram_mib, b.mm2_per_mib_sram), ("slc", chip.slc_mib, b.mm2_per_mib_slc),
                            ("mac", chip.macs / 1024, b.mm2_per_kmac)):
        if qty <= 0:
            continue
        if dens is None:
            missing.append({"sram": "mm2_per_mib_sram", "slc": "mm2_per_mib_slc", "mac": "mm2_per_kmac"}[name])
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
    direct = {"sram_mib": ch.sram_mib, "slc_mib": ch.slc_mib, "macs": float(ch.macs),
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
                   "pJ_bit_slc": "slc", "pJ_bit_d2d": "d2d", "pJ_bit_net": "net", "idle_W": "idle_card_s"}
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
    return {"ok": ok, "items": items, "area": area, "provided": b.provided,
            "violations": [i["key"] for i in items if i["ok"] is False],
            "basis": "用户输入的限额与面积密度「假设」；工具不内置工艺库，只报告余量"}
