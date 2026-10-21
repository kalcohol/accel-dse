"""0.300: energy-aware design search — Pareto front of throughput vs J/token (≡ perf/W) under area / power budgets.

The scenario's chip is varied on a small grid of design variables, every point is evaluated by the cost model (time,
fit) and priced by the energy table (MAC / SRAM / wire / NoC / DRAM / static, incl. the 0.299 dataflow NoC):
  MAC count   × mac_scales  (engines × r² re-cut; r from `sides`, default the current side ÷2, ×1, ×2)
  SRAM        × sram_scales (the per-engine slice follows; area and the 0.298 distance term follow)
  NoC         topologies    (default: the chip's)
Constraints (budget, all optional; nothing is defaulted silently):
  area   die area proxy (budget.full_area with the user's densities) ≤ die_mm2 — a hard limit (area cannot be throttled)
  power  avg card power ≤ power_W_card (or energy.tdp_W): over-budget points either run clock-capped (power_mode
         "dvfs", the 0.227 DVFS model on curve `curve`; throughput and J/token at the capped clock) or are dropped
         ("reject")
  fit    the model must fit (memory) — infeasible points are kept in the table, marked
perf/W = units/s ÷ W = 1 / (J/unit) for a given workload, so the 2-D front (throughput ↑, J/unit ↓) is also the
(throughput, perf/W) front; `front3` adds die area ↓ as a third objective.
"""
from __future__ import annotations

from dataclasses import replace

from .energy import EnergyTable, energy_report
from .evaluate import evaluate
from .hardware import NOC_TOPOLOGIES
from .noc_insight import _fill

MAX_POINTS = 96
MAC_SCALES = (0.5, 1.0, 2.0)
SRAM_SCALES = (0.5, 1.0, 2.0)
POWER_MODES = ("dvfs", "reject")


def dominates(a: dict, b: dict, keys, tie: float = 0.0) -> bool:
    """a dominates b on keys [(name, +1 maximise | −1 minimise)]: within ``tie`` (relative) of b or better on every
    key, and better than b by more than ``tie`` on at least one.  tie = 0 is plain Pareto dominance."""
    def no_worse(x, y, s):
        return x >= y * (1 - tie) if s > 0 else x <= y * (1 + tie)

    def better(x, y, s):
        return x > y * (1 + tie) if s > 0 else x < y * (1 - tie)
    return all(no_worse(a[k], b[k], s) for k, s in keys) and any(better(a[k], b[k], s) for k, s in keys)


def pareto(rows: list[dict], keys, tie: float = 0.0) -> list[int]:
    """Indices of the non-dominated rows.  With tie > 0 (differences within tie are not a ranking, as insight.TIE)
    the relation is not transitive; an empty result falls back to the exact (tie = 0) front."""
    f = [i for i, r in enumerate(rows) if not any(dominates(o, r, keys, tie) for j, o in enumerate(rows) if j != i)]
    return f if f or not tie else pareto(rows, keys, 0.0)


def _sides(ch, sides) -> list[int]:
    if sides:
        return sorted({int(s) for s in sides})
    cur = int(ch.rows)
    return sorted({s for s in (cur // 2, cur, cur * 2) if s >= 4})


def design_grid(ch, mac_scales=MAC_SCALES, sram_scales=SRAM_SCALES, sides=None, topologies=None) -> list[dict]:
    """Design points: (macs, engines, side, sram_mib, topology).  A (macs, side) pair is kept when macs / side² is
    a whole engine count ≥ 1."""
    topo = list(topologies) if topologies else [ch.noc_topology]
    out = []
    for ms in mac_scales:
        macs = int(round(ch.macs * ms))
        for sd in _sides(ch, sides):
            e, rem = divmod(macs, sd * sd)
            if rem or not 1 <= e <= 4096:
                continue
            for ss in sram_scales:
                for t in topo:
                    out.append({"mac_x": ms, "sram_x": ss, "macs": macs, "engines": e, "side": sd,
                                "sram_mib": ch.sram_mib * ss, "topology": t,
                                "base": ms == 1.0 and ss == 1.0 and sd == ch.rows and t == ch.noc_topology})
    return out


def energy_design_search(scn, table: EnergyTable, budget=None, mac_scales=MAC_SCALES, sram_scales=SRAM_SCALES,
                         sides=None, topologies=None, power_mode: str = "dvfs", curve: str = "dfs",
                         phase: str | None = None, prompt: int | None = None, tie: float | None = None) -> dict:
    from .budget import full_area
    from .insight import TIE
    from .powercap import CAP_CURVES, dvfs_cap
    tie = TIE if tie is None else float(tie)
    if not 0 <= tie < 0.5:
        raise ValueError("tie must be in [0, 0.5)")
    if table is None or table.pJ_mac is None:
        raise ValueError("能耗设计搜索需要能耗表（左侧「能耗」组或预设，至少 pJ_mac）")
    if power_mode not in POWER_MODES:
        raise ValueError(f"power_mode must be one of {POWER_MODES}")
    if curve not in CAP_CURVES:
        raise ValueError(f"curve must be one of {CAP_CURVES}")
    if topologies and not all(t in NOC_TOPOLOGIES for t in topologies):
        raise ValueError(f"topologies must be a list of {NOC_TOPOLOGIES}")
    if phase is not None and phase != scn.serving.phase:
        if phase not in ("decode", "prefill"):
            raise ValueError("phase must be 'decode' or 'prefill'")
        sv = replace(scn.serving, phase=phase)
        if phase == "prefill":
            sv = replace(sv, batch=1, prompt=prompt or sv.prompt)
        scn = scn.replace("serving", sv)
    elif prompt is not None:
        scn = scn.replace("serving", replace(scn.serving, prompt=prompt))
    ch = scn.chip
    if ch.rows != ch.cols:
        raise ValueError("能耗设计搜索按方阵引擎 r×r 重切：当前芯片阵列非方阵")
    t, notes = _fill(table)
    grid = design_grid(ch, mac_scales, sram_scales, sides, topologies)
    if not grid:
        raise ValueError("设计网格为空（MAC 总数 / 边长² 须为整数个引擎）")
    if len(grid) > MAX_POINTS:
        raise ValueError(f"设计网格 {len(grid)} 点 > 上限 {MAX_POINTS}：减少 mac_scales / sram_scales / sides / topologies")
    b = budget
    area_lim = getattr(b, "die_mm2", None) if b is not None else None
    cap = (getattr(b, "power_W_card", None) if b is not None else None) or t.tdp_W
    cap_src = "budget.power_W_card" if b is not None and getattr(b, "power_W_card", None) else ("energy.tdp_W" if t.tdp_W else None)
    dens_ok = b is not None and b.mm2_per_kmac is not None and b.mm2_per_mib_sram is not None
    if not dens_ok:
        notes.append("预算未填面积密度（mm2_per_kmac / mm2_per_mib_sram）：不算面积、不做面积约束（可选密度预设）")
    if area_lim is not None and not dens_ok:
        notes.append("die_mm2 已填但缺面积密度：面积约束未生效")
    if cap is None:
        notes.append("未填功耗预算（budget.power_W_card 或 energy.tdp_W）：不做功耗约束")
    if ch.instance_sched == "cta" or getattr(ch, "gpu", False):
        notes.append("GPU 参考芯片（cta 调度）：NoC 流量未建模，重切引擎的结论只对 NPU 有意义")
    rows = []
    for p in grid:
        c = replace(ch, engines=p["engines"], rows=p["side"], cols=p["side"], sram_mib=p["sram_mib"],
                    noc_topology=p["topology"])
        s = scn.replace("chip", c)
        row = {**p, "array": f'{p["side"]}×{p["side"]}'}
        try:
            r = evaluate(s)
            er = energy_report(r, t)
        except ValueError as ex:
            rows.append({**row, "feasible": False, "why": str(ex)})
            continue
        if er.get("J_per_unit") is None:
            rows.append({**row, "feasible": False, "why": er.get("note") or "能耗表为空"})
            continue
        row.update(unit=er["unit"], fits=bool(r.fits), bound=r.bound, thr_nominal=r.per_card,
                   J_nominal=er["J_per_unit"], W_nominal=er["avg_W_per_card"], share=er["breakdown"]["share"],
                   thr=r.per_card, J_per_unit=er["J_per_unit"], W_card=er["avg_W_per_card"], f_scale=1.0, capped=False)
        if dens_ok:
            a = full_area(s, b)
            row["mm2"] = a["mm2"]
        why = []
        if not r.fits:
            why.append("放不下（内存）")
        if area_lim is not None and dens_ok and row.get("mm2") is not None and row["mm2"] > area_lim * (1 + 1e-9):
            why.append(f"面积 {row['mm2']:.0f} > {area_lim:g} mm²")
        if cap is not None and row["W_card"] > cap * (1 + 1e-9) and not why:
            if power_mode == "reject":
                why.append(f"功耗 {row['W_card']:.0f} > {cap:g} W")
            else:
                d = dvfs_cap(r, t, cap, curve, iters=12)
                if d.get("error") or not d.get("feasible", True):
                    why.append(d.get("error") or "功耗封顶不可行")
                else:
                    row.update(thr=r.per_card * d["throughput_factor"], J_per_unit=d["J_per_unit"], W_card=d["W_card"],
                               f_scale=d["s"], capped=True)
        row["feasible"] = not why
        if why:
            row["why"] = "；".join(why)
        row["perf_W"] = 1.0 / row["J_per_unit"] if row["J_per_unit"] > 0 else None
        rows.append(row)
    ok = [i for i, x in enumerate(rows) if x.get("feasible")]
    sub = [rows[i] for i in ok]
    f2 = [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1)), tie)]
    for i in f2:
        rows[i]["front"] = True
    if dens_ok:
        f3 = [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("mm2", -1)), tie)]
        for i in f3:
            rows[i]["front3"] = True
    best = {}
    if sub:
        def pick(i):
            x = rows[i]
            return {"index": i, "engines": x["engines"], "array": x["array"], "macs": x["macs"], "sram_mib": x["sram_mib"],
                    "topology": x["topology"], "thr": x["thr"], "J_per_unit": x["J_per_unit"], "mm2": x.get("mm2")}
        best["perf_W"] = pick(min(ok, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"])))
        best["thr"] = pick(max(ok, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"])))
    base = next((i for i, x in enumerate(rows) if x["base"]), None)
    return {"chip": ch.name, "model": scn.model, "phase": scn.serving.phase, "rows": rows, "front": sorted(f2, key=lambda i: rows[i]["thr"]),
            "best": best, "base": base, "notes": notes, "power_cap_W": cap, "power_cap_src": cap_src, "power_mode": power_mode,
            "curve": curve, "die_mm2": area_lim, "tie": tie, "area": dens_ok, "n_feasible": len(ok),
            "basis": "设计网格（MAC 总数 × 引擎边长 × SRAM × NoC 拓扑）逐点评估：时间 / 放得下来自代价模型，J/单位 来自能耗表"
                     "（MAC / SRAM / 导线 / NoC / DRAM / 静态）；perf/W = 1 / (J/单位)；超功耗点按 DVFS 封顶运行（或剔除），"
                     "超面积点剔除；前沿 = 吞吐↑ 与 J/单位↓ 的非支配集，差 ≤ tie（默认 1 %）视为持平（MODEL §245）"}
