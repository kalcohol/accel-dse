"""Resource / area budgets as optional design constraints (0.49) — headroom against user-entered limits.

Nothing is built in: every limit and every density is ``None`` until the user enters one, and a budget with no entry
changes nothing.  Two kinds of entries:

  direct limits   quantities the model already knows exactly for the evaluated point — per card: SRAM MiB, SLC MiB,
                  MAC units (rows·cols·engines), peak bf16 TFLOPS, DRAM capacity GiB; per replica: cards; and the
                  average power per card (from the user's energy table, core/energy.py)
  area proxy      die area per card ≈ SRAM MiB · mm²/MiB + SLC MiB · mm²/MiB + MACs / 1024 · mm²/kMAC + fixed mm²
                  (I/O, PHY, vector unit, NoC, …).  The densities are the user's 「假设」 (a process / library figure
                  they trust); the tool ships no process library.  Checked against a per-card die limit and a
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
                   "pJ_bit_slc": "slc", "pJ_bit_d2d": "d2d", "idle_W": "idle_card_s"}
            e = {**e, "missing": [m for m in e.get("missing", []) if cnt.get(key.get(m, ""), 1) > 0]}  # zero-count entries don't matter
            if e["missing"]:
                ok = False if w > b.power_W_card else None
                items.append(_item("power_W_card", w, b.power_W_card, ok,
                                   f"能耗表未填 {', '.join(e['missing'])}：{w:.3g} W 只是下界"))
            else:
                items.append(_item("power_W_card", w, b.power_W_card))
    area = area_estimate(ch, b) if any(getattr(b, k) is not None for k in _DENS + ("die_mm2", "system_mm2")) else None
    for k, mult in (("die_mm2", 1), ("system_mm2", lay.cards)):
        lim = getattr(b, k)
        if lim is None:
            continue
        if area["mm2"] is None:
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
