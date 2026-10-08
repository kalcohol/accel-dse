"""Ranking stability: does the best layout survive plausible assumption changes?

Perturbation set (one-at-a-time around the base, plus two joint corners):
  mapping organisation   os / ws_edge / ws_broad / os_vec / reconf
  DRAM efficiency        0.6 / 0.7 / 0.85
  link α                 1 / 3 / 5 µs
  MAC efficiency         0.7 / 1.0
Stable ⇔ in ≥ 90 % of perturbed cases the base top-1 layout is still top-1 or
within 5 % of the perturbed top-1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mapping import ORGS
from .scenario import Scenario
from .search import search_layouts


def perturbations(base: Scenario, include_mapping: bool = True) -> list[tuple[str, Scenario]]:
    out = []
    if include_mapping:
        for org in ORGS:
            if org != base.mapping:
                out.append((f"mapping={org}", base.replace("mapping", org)))
    for e in (0.6, 0.7, 0.85):
        if e != (base.mem_eff or 0.7):
            out.append((f"mem_eff={e}", base.replace("mem_eff", e)))
    for a in (1.0, 3.0, 5.0):
        if a != base.link.alpha_us:
            out.append((f"alpha={a}us", base.replace("link.alpha_us", a)))
    if base.chip.mac_eff != 0.7:
        out.append(("mac_eff=0.7", base.replace("chip.mac_eff", 0.7)))
    out.append(("corner: mem_eff=0.6, α=5, mac_eff=0.7",
                base.replace("mem_eff", 0.6).replace("link.alpha_us", 5.0).replace("chip.mac_eff", 0.7)))
    out.append(("corner: mem_eff=0.85, α=1", base.replace("mem_eff", 0.85).replace("link.alpha_us", 1.0)))
    return out


@dataclass
class Stability:
    base_top: str
    stable: bool
    agree: float
    cases: list[dict]


def ranking_stability(base: Scenario, cards: int, include_mapping: bool = True, max_tp: int | None = None) -> Stability:
    rows = search_layouts(base, cards, max_tp=max_tp)
    top = rows[0]
    key = lambda r: (r.layout, )
    cases, ok = [], 0
    for name, scn in perturbations(base, include_mapping):
        pr = search_layouts(scn, cards, max_tp=max_tp)
        best = pr[0]
        mine = next((r for r in pr if r.layout == top.layout), None)
        mine_v = mine.per_card if mine else 0.0
        same = best.layout == top.layout
        close = best.per_card > 0 and mine_v >= 0.95 * best.per_card
        ok += same or close
        cases.append({"case": name, "top": best.layout.label, "top_tok_s_card": best.per_card,
                      "base_top_tok_s_card": mine_v, "same": same, "within5": close})
    agree = ok / len(cases) if cases else 1.0
    return Stability(top.layout.label, agree >= 0.9, agree, cases)
