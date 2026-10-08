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
from .search import best_batch, search_layouts


def perturbations(base: Scenario, include_mapping: bool = True) -> list[tuple[str, Scenario]]:
    out = []
    if include_mapping:
        for org in ORGS:
            if org != base.mapping:
                out.append((f"映射 = {org}", base.replace("mapping", org)))
    for e in (0.6, 0.7, 0.85):
        if e != (base.mem_eff or 0.7):
            out.append((f"DRAM 效率 = {e}", base.replace("mem_eff", e)))
    for a in (1.0, 3.0, 5.0):
        if a != base.link.alpha_us:
            out.append((f"同步 α = {a:g} µs", base.replace("link.alpha_us", a)))
    if base.chip.mac_eff != 0.7:
        out.append(("MAC 效率 = 0.7", base.replace("chip.mac_eff", 0.7)))
    out.append(("组合：DRAM 效率 0.6 + α 5 µs + MAC 效率 0.7",
                base.replace("mem_eff", 0.6).replace("link.alpha_us", 5.0).replace("chip.mac_eff", 0.7)))
    out.append(("组合：DRAM 效率 0.85 + α 1 µs", base.replace("mem_eff", 0.85).replace("link.alpha_us", 1.0)))
    return out


@dataclass
class Stability:
    base_top: str
    stable: bool
    agree: float
    cases: list[dict]


def _score_of(scn: Scenario, objective: str) -> float:
    """Exact score of one fixed layout (best batch; goodput if requested)."""
    from .serving import goodput
    bb = best_batch(scn)
    if not bb.batch:
        return 0.0
    if objective == "goodput":
        return goodput(bb.result).goodput_per_card
    return bb.result.per_card


def ranking_stability(base: Scenario, cards: int, include_mapping: bool = True, max_tp: int | None = None,
                      objective: str = "decode", progress=None) -> Stability:
    """Top-1 layout under each perturbation (exact top-1 search) vs the base top-1."""
    rows = search_layouts(base, cards, max_tp=max_tp, objective=objective, top=1)
    top = rows[0]
    cases, ok = [], 0
    perts = perturbations(base, include_mapping)
    for i, (name, scn) in enumerate(perts):
        # the base top-1 layout's exact score is the incumbent: only layouts that strictly beat it are solved
        mine_v = _score_of(scn.replace("layout", top.layout), objective)
        better = [r for r in search_layouts(scn, cards, max_tp=max_tp, objective=objective, top=1, floor=mine_v) if r.batch]
        if better:
            best_lab, best_v = better[0].layout.label, better[0].score(objective)
        else:
            best_lab, best_v = top.layout.label, mine_v
        same = not better
        close = best_v > 0 and mine_v >= 0.95 * best_v
        ok += same or close
        cases.append({"case": name, "top": best_lab, "top_tok_s_card": best_v,
                      "base_top_tok_s_card": mine_v, "same": same, "within5": close})
        if progress:
            progress(i + 1, len(perts))
    agree = ok / len(cases) if cases else 1.0
    return Stability(top.layout.label, agree >= 0.9, agree, cases)
