"""0.300: energy-aware design search — Pareto front of throughput vs J/token (≡ perf/W) under area / power budgets.
0.301: memory side (DRAM type / unit count / per-unit capacity, SLC size) as joint axes; workload mix; SLO limits.
0.302: DVFS operating point (clock / voltage) as an axis; LPDDR background power and HBM4 / LPDDR6 pJ/bit in the table.

The scenario's chip is varied on a small grid of design variables, every point is evaluated by the cost model (time,
fit) and priced by the energy table (MAC / SRAM / wire / NoC / DRAM / static, incl. the 0.299 dataflow NoC):
  MAC count   × mac_scales  (engines × r² re-cut; r from `sides`, default the current side ÷2, ×1, ×2)
  SRAM        × sram_scales (the per-engine slice follows; area and the 0.298 distance term follow)
  NoC         topologies    (default: the chip's)
  DRAM        `mems` (explicit catalog ids) or `mem_types` × `mem_count_scales` × `mem_caps` (0.301; default: the
              scenario's memory).  Bandwidth / capacity / fit from the cost model; energy from the per-kind pJ/bit
              (pJ_bit_dram_hbm / _lpddr, 0.302 _hbm4 / _lpddr6, + PHY) and DRAM standby / refresh; die area adds the DRAM PHY (bus bits ×
              mm2_per_dq_*); HBM stacks are checked against the die shoreline (budget.shoreline_check, "over" = drop)
  SLC         `slc_mibs` (0.301; absolute MiB, default the chip's): area mm2_per_mib_slc, SLC access + wire energy
  NoC link    `noc_link_Bpc` (0.303; default the chip's, 0 = unlimited): bisection congestion on every point;
              `noc_routing` (0.304; "bisection" / "xy"): busiest-link load under dimension-ordered routing
              `noc_links` (0.306): link width as an axis — every point also pays router area + width-dependent
              static power (noc_router_cost, Park DAC 2012 「来源」 scaled 「推导/假设」), MODEL §251.2
  clock       `f_scales` (0.302; f/f_max of the chip's nominal clock, default 1): the DVFS operating point — the
              chip is re-evaluated at freq_ghz·f, core-rail action energies × V(f)², SRAM leakage × V(f), V(f) from the
              `curve` (powercap._point, the same mechanism as the power cap).  More MACs at a lower clock vs fewer at
              f_max is then a front trade under the power budget.  Under "dfs" (V flat) a lower clock never lowers J/unit.
Constraints (budget, all optional; nothing is defaulted silently):
  area   die area proxy (budget.full_area with the user's densities) ≤ die_mm2 — a hard limit (area cannot be throttled)
  power  avg card power ≤ power_W_card (or energy.tdp_W): over-budget points either run clock-capped (power_mode
         "dvfs", the 0.227 DVFS model on curve `curve`; throughput and J/token at the capped clock) or are dropped
         ("reject")
  fit    the model must fit (memory) — infeasible points are kept in the table, marked
  SLO    slo_tpot_ms / slo_ttft_ms (0.301): decode TPOT / prefill TTFT at the (capped) clock
  $      mem_usd_per_GB {type: $/GB} (user input only, no preset) → memory $ per card; max_mem_usd limit
Workload mix (0.301, `mix_out_tokens` D): one request = prompt P (serving.prompt) prefill tokens + D decode tokens on the
same card, time-shared: card-s/request = P / thr_prefill + D / thr_decode, J/request = P·J_prefill + D·J_decode
「推导」 (each phase at its own batch and its own capped clock; queueing / PD interference not modelled).
perf/W = units/s ÷ W = 1 / (J/unit) for a given workload, so the 2-D front (throughput ↑, J/unit ↓) is also the
(throughput, perf/W) front; `front3` adds die area ↓, `front_usd` memory $ ↓ (when priced).
"""
from __future__ import annotations

from dataclasses import replace

from .energy import EnergyTable, energy_report
from .evaluate import evaluate
from .hardware import NOC_TOPOLOGIES
from .noc_insight import _fill

MAX_POINTS = 240                       # 0.301: 96 → 240 (memory / SLC axes)

# 0.306: NoC router cost per link width, for the `noc_links` axis (constants and scaling live in energy.py since 0.307).
from .energy import (NOC_COST_SRC, NOC_MAC_UM2_45, NOC_PORTS, NOC_ROUTER_CLK_PJ_45,  # noqa: E402,F401
                     NOC_ROUTER_LEAK_MW_45, NOC_ROUTER_REF_BITS, NOC_ROUTER_UM2_45, NOC_XBAR_UM2_45, noc_router_scale)
NOC_NODE_E = 2.6        # 45 → 7 nm energy factor of the tpuv4i_7nm preset 「推导」


def noc_router_cost(link_Bpc: float, topology: str, f_ghz: float, mm2_per_kmac: float | None) -> dict | None:
    """0.306: one router's area (mm²) and width-dependent static power (W) for a link of ``link_Bpc`` bytes / cycle
    (energy.noc_router_scale; clock floor ∝ ports·width, leakage ∝ area 「假设」).  Node: area × (the user's MAC density
    ÷ Horowitz's 45 nm bf16 MAC area) — logic scales like the MAC datapath 「推导/假设」; energy / leakage ÷ 2.6 (the
    preset's 45 → 7 nm factor) 「推导」.  Crossbar topology (one switch for all engines): None."""
    sc = noc_router_scale(link_Bpc, topology)
    if sc is None:
        return None
    node = (mm2_per_kmac / 1024 * 1e6 / NOC_MAC_UM2_45) if mm2_per_kmac else None
    W = (NOC_ROUTER_CLK_PJ_45 * sc["lin"] * f_ghz * 1e-3 + NOC_ROUTER_LEAK_MW_45 * sc["area_x"] * 1e-3) / NOC_NODE_E
    return {"mm2": sc["um2_45"] * 1e-6 * node if node else None, "W": W, "area_x": sc["area_x"]}


# 0.308: chiplet partitioning — package / yield limits (each with its status; every one overridable by argument)
RETICLE_MM2 = 26 * 33                  # 858 mm² single-exposure field (26 × 33 mm, ASML / industry standard) 「来源」
PKG_LIMITS = {    # interposer area (dies + HBM stacks must fit) — package technology presets
    "cowos-s": (2700.0, "来源", "TSMC 3DFabric CoWoS page: CoWoS-S interposer up to 3.3× reticle (~2700 mm²)"),
    "cowos-l": (5.5 * RETICLE_MM2, "来源/推导", "TSMC 2026 North America Technology Symposium press release: "
                "'now producing 5.5-reticle size CoWoS' → 5.5 × 858 = 4719 mm²"),
}
HBM_FOOTPRINT_MM2 = {"HBM4": 11.0 * 12.8, "HBM4E": 11.0 * 12.8}   # HBM4 base die 11 × 12.8 mm (Samsung HC 2026) 「二手」
HBM_FOOTPRINT_DEFAULT = 11.0 * 11.0     # HBM3 / 3E: 11 mm edge (budget.HBM_STACK_EDGE_MM) squared 「近似」
D0_NOTE = ("缺陷密度参考：TSMC N5 量产爬坡时约 0.10 /cm²、N7 约 0.09 /cm²（TSMC 技术论坛 / VLSI 2019，经 AnandTech 转述「二手」）；"
           "良率模型 Y = exp(−A·D0)（Poisson「假设」；Murphy / 负二项更乐观）；晶粒在封装前测试（KGD）「假设」，封装 / 键合良率未计")
DIE_SPLITS = ("tp", "pp", "ep")


def _mem_split(mem_id: str, n: int) -> str | None:
    """0.308: the per-die memory of a package memory split over n dies (unit count ÷ n, same unit), or None when the
    count does not divide or the catalog has no such count."""
    from .. import mem_catalog as mc
    if n == 1:
        return mem_id
    sp = mc.parse_mem_id(mem_id)
    if sp.n_units % n:
        return None
    hbm = sp.kind == "HBM"
    kw = {"height": sp.hbm_height, "die_Gb": sp.hbm_die_Gb} if hbm else {"cap_GB": sp.cap_per_unit_GB}
    try:
        return mc.make_spec(sp.mem_type, form=None if hbm else sp.form, width_bits=sp.unit_width_bits,
                            rate=sp.rate_MTps, count=sp.n_units // n, **kw).id
    except (KeyError, ValueError):
        return None


def _split_layout(lay, n: int, split: str, moe: bool):
    """0.308: the replica's layout with its package split over n dies — the cross-die traffic is the existing
    collective model of that axis on the D2D tier (schedule: a package's consecutive cards talk over D2D):
      tp  TP × n (MoE: expert TP × n too, ep·etp = tp·dp) — per-layer all-reduce / all-gather over D2D
      pp  PP × n — stage-boundary activations over D2D (and the pipeline bubble of the micro-batches)
      ep  MoE only: attention DP × n, experts EP × n — dispatch / combine all-to-all over D2D."""
    from .parallel import Layout
    if split == "tp":
        return Layout(lay.pp, lay.tp * n, lay.dp, lay.ep, lay.etp * n if moe else lay.etp, lay.sp, lay.cp)
    if split == "pp":
        return Layout(lay.pp * n, lay.tp, lay.dp, lay.ep, lay.etp, lay.sp, lay.cp)
    if not moe:
        raise ValueError("die_split ep 只用于 MoE 模型")
    return Layout(lay.pp, lay.tp, lay.dp * n, lay.ep * n, lay.etp, lay.sp, lay.cp)


def _hbm_footprint(mem_id: str) -> float:
    from .. import mem_catalog as mc
    sp = mc.parse_mem_id(mem_id)
    if sp.kind != "HBM":
        return 0.0
    return sp.n_units * HBM_FOOTPRINT_MM2.get(sp.mem_type, HBM_FOOTPRINT_DEFAULT)


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


def _snap(n: float, allowed) -> int:
    return int(min(allowed, key=lambda c: (abs(c - n), c)))


def mem_axis(base_id: str, mems=None, mem_types=None, mem_count_scales=(1.0,), mem_caps=None) -> list[str]:
    """0.301: candidate memory ids.  Explicit `mems` win; else each type in `mem_types` × unit-count scale × per-unit
    capacity (GB; LPDDR: a catalog package capacity, HBM: height × die density / 8).  The count scale applies to the
    scenario's own count when the type matches, else to the type's catalog default; counts snap to the catalog's.
    Combinations the catalog does not offer are skipped."""
    from .. import mem_catalog as mc
    if mems:
        return list(dict.fromkeys(mc.parse_mem_id(m).id for m in mems))
    if not mem_types:
        return [mc.parse_mem_id(base_id).id]
    base = mc.parse_mem_id(base_id)
    out = []
    for t in mem_types:
        tt = mc.normalize_type(t)
        ref = base if base.mem_type == tt else mc.default_spec(tt)
        hbm = mc.mem_kind_of(tt) == "HBM"
        if hbm:
            cat = mc.HBM_CATALOG[tt]
            counts = tuple(range(1, mc.HBM_MAX_KNOWN_STACKS + 1))
        else:
            f = mc.LPDDR_CATALOG[tt]["forms"][ref.form]
            counts = tuple(f["counts"])
        for cs in mem_count_scales or (1.0,):
            n = _snap(ref.n_units * float(cs), counts)
            for cap in (mem_caps or (None,)):
                kw = {}
                if cap is not None:
                    if hbm:
                        hd = [(h, d) for h in cat["heights"] for d in cat["densities"] if abs(h * d / 8.0 - cap) < 1e-9]
                        if not hd:
                            continue
                        kw = {"height": hd[0][0], "die_Gb": hd[0][1]}
                    else:
                        if int(cap) != cap or int(cap) not in f["caps"].get(ref.unit_width_bits, {}):
                            continue
                        kw = {"cap_GB": float(cap)}
                elif hbm:
                    kw = {"height": ref.hbm_height, "die_Gb": ref.hbm_die_Gb}
                else:
                    kw = {"cap_GB": ref.cap_per_unit_GB}
                sp = mc.make_spec(tt, form=None if hbm else ref.form, width_bits=ref.unit_width_bits,
                                  rate=ref.rate_MTps, count=n, **kw)
                out.append(sp.id)
    return list(dict.fromkeys(out))


def _mem_info(mem_id: str) -> dict:
    from .. import mem_catalog as mc
    sp = mc.parse_mem_id(mem_id)
    return {"mem_id": sp.id, "mem_type": sp.mem_type, "mem_units": sp.n_units, "mem_GB": sp.capacity_GB,
            "mem_GBps": sp.raw_GBps, "mem_bus_bits": sp.bus_bits, "mem_kind": sp.kind}


# 0.301: which preset pJ/bit / PHY figures stand in for a memory type they were not measured on (labelled in notes)
MEM_PROXY = {      # 0.302: HBM4 / LPDDR6 pJ/bit sourced (pJ_bit_dram_hbm4 / _lpddr6); PHY energy still borrowed
    "HBM4": "HBM4 pJ/bit = pJ_bit_dram_hbm4（预设 2.46：HBM3E ÷ 1.4，Samsung / SK hynix 厂商口径「推导」）；PHY 仍用 HBM3 0.29 pJ/bit「近似（假设）」",
    "HBM4E": "HBM4E 沿用 HBM4 的 pJ/bit（pJ_bit_dram_hbm4）与 HBM3 PHY 0.29 pJ/bit「近似（假设）」",
    "HBM3": "HBM3 用 HBM3E 的 pJ/bit（HBM3E = HBM2E −20 %，HBM3 介于两者）「近似（假设）」",
    "LPDDR6": "LPDDR6 pJ/bit = pJ_bit_dram_lpddr6（预设 3.31：LPDDR5X ÷ 1.21，Samsung ISSCC 2026「推导」）；PHY 仍用 LPDDR5X 1.3 pJ/bit、"
              "后台功耗沿用 LPDDR5X（Micron Y52Q）「近似（假设）」",
    "LPDDR5": "LPDDR5 用 LPDDR5X 的 pJ/bit 与后台功耗「近似（假设）」",
}
PHY_AREA_PROXY = {"HBM": "HBM PHY 面积按 HBM4 base die 的常规 PHY（> 8 × 4 mm / 2048 bit）每 bit 计，HBM3E 窄总线同密度「近似」",
                  "LPDDR": "LPDDR PHY 面积按 SF2 LPDDR6 x24 通道 0.695 mm² 每 bit 计，LPDDR5X 同密度「近似」"}


def _phase_scn(scn, phase, prompt):
    if phase == scn.serving.phase:
        return scn if prompt is None or phase != "prefill" else scn.replace("serving", replace(scn.serving, prompt=prompt))
    sv = replace(scn.serving, phase=phase)
    if phase == "prefill":
        sv = replace(sv, batch=1, prompt=prompt or sv.prompt)
    return scn.replace("serving", sv)


def _run(s, t, cap, power_mode, curve, f_op: float = 1.0, n_dies: int = 1) -> dict:
    """0.308: a package of ``n_dies`` dies (one card each in the cost model): every die runs under cap / n and the
    per-card figures are scaled to the package (throughput, W, static W × n; J/unit and latency unchanged)."""
    o = _run1(s, t, None if cap is None else cap / n_dies, power_mode, curve, f_op)
    if n_dies > 1:
        for k in ("thr", "thr_nominal", "W_card", "W_nominal", "hbm_static_W", "noc_W"):
            if o.get(k) is not None:
                o[k] *= n_dies
        o["why"] = [f"每晶粒（预算 ÷ {n_dies}）{w}" for w in o["why"]]
    return o


def _run1(s, t, cap, power_mode, curve, f_op: float = 1.0) -> dict:
    """One phase of one design point: at the operating point f_op (0.302; clock f_op·f_max, voltage V(f_op) on the
    curve, core-rail energies × V², leakage × V — powercap._point), then capped / rejected by power.  A point over the
    cap is capped from f_max like any other: power rises with f, so the cap clock is below f_op and equals the f_op = 1
    point's capped result."""
    from .powercap import _point, dvfs_cap
    r0 = evaluate(s)
    r, er = (r0, energy_report(r0, t)) if f_op == 1.0 else _point(r0, t, curve, f_op)
    if er.get("J_per_unit") is None:
        raise ValueError(er.get("note") or "能耗表为空")
    dec = s.serving.phase == "decode"
    lat = (r.tpot if dec else r.ttft) or 0.0
    o = {"unit": er["unit"], "fits": bool(r.fits), "bound": r.bound, "thr_nominal": r.per_card,
         "J_nominal": er["J_per_unit"], "W_nominal": er["avg_W_per_card"], "share": er["breakdown"]["share"],
         "thr": r.per_card, "J_per_unit": er["J_per_unit"], "W_card": er["avg_W_per_card"], "f_scale": f_op,
         "capped": False, "lat_ms": lat * 1e3, "hbm_static_W": er.get("hbm_static_W_card"), "why": [],
         "noc_W": (er.get("noc_router") or {}).get("W")}      # 0.307: router static W at the operating point
    r = r0
    if cap is not None and o["W_card"] > cap * (1 + 1e-9) and r.fits:
        if power_mode == "reject":
            o["why"].append(f"功耗 {o['W_card']:.0f} > {cap:g} W")
        else:
            d = dvfs_cap(r, t, cap, curve, iters=12)
            if d.get("error") or not d.get("feasible", True):
                o["why"].append(d.get("error") or "功耗封顶不可行")
            else:
                k = d["throughput_factor"]
                o.update(thr=r.per_card * k, J_per_unit=d["J_per_unit"], W_card=d["W_card"], f_scale=d["s"],
                         capped=True, lat_ms=o["lat_ms"] / k if k > 0 else o["lat_ms"])
    return o


def _split_compare(rows: list[dict], tie: float = 0.01) -> list[dict]:
    """0.308: per design (same package totals: MACs, SRAM, engine side, topology, memory, SLC, clock, NoC link), the
    monolithic die (dies = 1) against its best feasible split (by throughput and by J/unit).  Verdicts: only_split
    (the monolithic die is infeasible — reticle / area / power / shoreline — and a split is feasible), split_thr /
    split_J / split_both (better by more than ``tie``), mono (no split beats it), none (nothing feasible)."""
    groups: dict = {}
    for i, x in enumerate(rows):
        k = (x["mac_x"], x["sram_x"], x["side"], x["topology"], x["mem_id"], x["slc_mib"], x["f_op"], x["noc_link_Bpc"])
        groups.setdefault(k, []).append(i)
    out = []
    for k, ix in groups.items():
        mono = next((i for i in ix if rows[i].get("dies", 1) == 1), None)
        sp = [i for i in ix if rows[i].get("dies", 1) > 1 and rows[i].get("feasible")]
        if not any(rows[i].get("dies", 1) > 1 for i in ix):
            continue
        e = {"mac_x": k[0], "sram_x": k[1], "side": k[2], "topology": k[3], "mem_id": k[4], "slc_mib": k[5],
             "f_op": k[6], "noc_link_Bpc": k[7], "mono": mono,
             "mono_feasible": bool(mono is not None and rows[mono].get("feasible")),
             "mono_why": rows[mono].get("why") if mono is not None else None}
        if not sp:
            e["verdict"] = "mono" if e["mono_feasible"] else "none"
            out.append(e)
            continue
        bt = max(sp, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"]))
        bj = min(sp, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"]))
        e.update(best_thr=bt, best_J=bj)
        if not e["mono_feasible"]:
            e["verdict"] = "only_split"
        else:
            m = rows[mono]
            e["d_thr"] = rows[bt]["thr"] / m["thr"] - 1 if m["thr"] > 0 else None
            e["d_J"] = rows[bj]["J_per_unit"] / m["J_per_unit"] - 1 if m["J_per_unit"] > 0 else None
            bthr = e["d_thr"] is not None and e["d_thr"] > tie
            bJ = e["d_J"] is not None and e["d_J"] < -tie
            e["verdict"] = "split_both" if bthr and bJ else "split_thr" if bthr else "split_J" if bJ else "mono"
        out.append(e)
    return out


def energy_design_search(scn, table: EnergyTable, budget=None, mac_scales=MAC_SCALES, sram_scales=SRAM_SCALES,
                         sides=None, topologies=None, power_mode: str = "dvfs", curve: str = "dfs",
                         phase: str | None = None, prompt: int | None = None, tie: float | None = None,
                         mems=None, mem_types=None, mem_count_scales=None, mem_caps=None, slc_mibs=None,
                         mix_out_tokens: int | None = None, slo_tpot_ms: float | None = None,
                         slo_ttft_ms: float | None = None, mem_usd_per_GB: dict | None = None,
                         max_mem_usd: float | None = None, f_scales=None, noc_link_Bpc: float | None = None,
                         noc_routing: str | None = None, noc_links=None, dies=None, d2d_stds=None,
                         d2d_units: int | None = None, die_split: str = "tp", reticle_mm2: float | None = None,
                         pkg_mm2=None, d0_per_cm2: float | None = None) -> dict:
    from .. import mem_catalog as mc
    from .budget import full_area, shoreline_check
    from .hardware import default_mem
    from .insight import TIE
    from .powercap import CAP_CURVES, F_MIN, VF_CURVES
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
    if phase is not None and phase not in ("decode", "prefill"):
        raise ValueError("phase must be 'decode' or 'prefill'")
    if mix_out_tokens is not None and not (isinstance(mix_out_tokens, int) and 1 <= mix_out_tokens <= 1 << 20):
        raise ValueError("mix_out_tokens must be an integer in [1, 1048576]")
    for k, v in (("slo_tpot_ms", slo_tpot_ms), ("slo_ttft_ms", slo_ttft_ms), ("max_mem_usd", max_mem_usd)):
        if v is not None and not v > 0:
            raise ValueError(f"{k} must be > 0")
    prices = {}
    for k, v in (mem_usd_per_GB or {}).items():
        try:
            tt = mc.normalize_type(k)
        except (ValueError, KeyError):
            raise ValueError(f"mem_usd_per_GB: unknown memory type {k!r}") from None
        if not (isinstance(v, (int, float)) and v >= 0):
            raise ValueError("mem_usd_per_GB values must be ≥ 0 ($/GB)")
        prices[tt] = float(v)
    if max_mem_usd is not None and not prices:
        raise ValueError("max_mem_usd needs mem_usd_per_GB (no memory price is built in)")
    if noc_link_Bpc is not None and not 0 <= noc_link_Bpc < 1e6:
        raise ValueError("noc_link_Bpc must be in [0, 1e6) bytes / cycle / link")
    from .hardware import NOC_ROUTINGS
    if noc_routing is not None and noc_routing not in NOC_ROUTINGS:
        raise ValueError(f"noc_routing must be one of {NOC_ROUTINGS}")
    if noc_links is not None and (not noc_links or not all(0 < float(x) < 1e6 for x in noc_links)):
        raise ValueError("noc_links must be a non-empty list of link widths in (0, 1e6) bytes / cycle")
    if noc_links is not None and noc_link_Bpc is not None:
        raise ValueError("noc_links (axis) and noc_link_Bpc (single value) are exclusive")
    fs = sorted({float(x) for x in f_scales}, reverse=True) if f_scales else [1.0]
    if not all(F_MIN <= x <= 1.0 for x in fs):
        raise ValueError(f"f_scales must be in [{F_MIN:g}, 1] (fraction of the chip's nominal clock)")
    # 0.308: chiplet partitioning axes
    from .d2d_catalog import D2D_STANDARDS, d2d_edges
    chip_mode = dies is not None
    Ns = sorted({int(x) for x in dies}) if chip_mode else [1]
    if chip_mode and (not dies or not all(float(x).is_integer() and 1 <= int(x) <= 16 for x in dies)):
        raise ValueError("dies must be a non-empty list of integers in [1, 16]")
    if chip_mode and scn.d2d_enabled and scn.package_cards > 1:
        raise ValueError("dies 轴从单片设计出发：场景已是多晶粒封装（d2d_enabled / package_cards > 1）")
    if d2d_stds is not None and (not d2d_stds or not all(x in D2D_STANDARDS for x in d2d_stds)):
        raise ValueError(f"d2d_stds must be a non-empty list of {sorted(D2D_STANDARDS)}")
    if (d2d_stds is not None or d2d_units is not None) and not chip_mode:
        raise ValueError("d2d_stds / d2d_units need the dies axis")
    if d2d_units is not None and not (isinstance(d2d_units, int) and 1 <= d2d_units <= 64):
        raise ValueError("d2d_units must be an integer in [1, 64]")
    if die_split not in DIE_SPLITS:
        raise ValueError(f"die_split must be one of {DIE_SPLITS}")
    stds = list(dict.fromkeys(d2d_stds)) if d2d_stds else [scn.d2d_std if scn.d2d_std != "custom" else "ucie-a-48"]
    units = d2d_units if d2d_units is not None else scn.d2d_units
    if reticle_mm2 is None and chip_mode:
        reticle_mm2 = float(RETICLE_MM2)
    if reticle_mm2 is not None and not reticle_mm2 >= 0:
        raise ValueError("reticle_mm2 must be ≥ 0 (0 = off)")
    pkg_src = None
    if isinstance(pkg_mm2, str):
        if pkg_mm2 not in PKG_LIMITS:
            raise ValueError(f"pkg_mm2 must be a number or one of {sorted(PKG_LIMITS)}")
        pkg_src = PKG_LIMITS[pkg_mm2][2]
        pkg_mm2 = PKG_LIMITS[pkg_mm2][0]
    if pkg_mm2 is not None and not pkg_mm2 > 0:
        raise ValueError("pkg_mm2 must be > 0")
    if d0_per_cm2 is not None and not 0 < d0_per_cm2 < 10:
        raise ValueError("d0_per_cm2 must be in (0, 10) defects / cm²")
    from .catalog import get_model
    moe = bool(get_model(scn.model).is_moe) if chip_mode else False
    pk_axis = [(1, None)] + [(nn, sd) for nn in Ns if nn > 1 for sd in stds] if chip_mode else [(1, None)]
    if chip_mode and 1 not in Ns:
        pk_axis = pk_axis[1:]
    mix = mix_out_tokens is not None
    phases = ("prefill", "decode") if mix else (phase or scn.serving.phase,)
    scn_ph = {ph: _phase_scn(scn, ph, prompt) for ph in phases}
    s0 = scn_ph[phases[-1]]
    P = scn_ph["prefill"].serving.prompt if mix else None
    ch = scn.chip
    if ch.rows != ch.cols:
        raise ValueError("能耗设计搜索按方阵引擎 r×r 重切：当前芯片阵列非方阵")
    t, notes = _fill(table)
    grid = design_grid(ch, mac_scales, sram_scales, sides, topologies)
    if not grid:
        raise ValueError("设计网格为空（MAC 总数 / 边长² 须为整数个引擎）")
    base_mem = mc.parse_mem_id(scn.mem_id or default_mem(ch)).id
    try:
        mem_ids = mem_axis(base_mem, mems, mem_types, mem_count_scales or (1.0,), mem_caps)
    except (KeyError, ValueError) as ex:
        raise ValueError(f"内存轴无效：{ex}") from None
    if not mem_ids:
        raise ValueError("内存轴为空（所选容量在目录中不存在）")
    slcs = sorted({float(x) for x in slc_mibs}) if slc_mibs else [float(ch.slc_mib)]
    if any(x < 0 for x in slcs):
        raise ValueError("slc_mibs must be ≥ 0")
    Ls = sorted({float(x) for x in noc_links}) if noc_links is not None else None
    n_pts = len(grid) * len(mem_ids) * len(slcs) * len(fs) * (len(Ls) if Ls else 1) * len(pk_axis)
    if n_pts > MAX_POINTS:
        raise ValueError(f"设计网格 {n_pts} 点 > 上限 {MAX_POINTS}：减少 mac_scales / sram_scales / sides / topologies / 内存 / SLC / 频率档 / NoC 链路 / 晶粒数 / D2D 接口")
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
    infos = {m: _mem_info(m) for m in mem_ids}
    if len(mem_ids) > 1 or slc_mibs:
        kinds = {i["mem_type"] for i in infos.values()}
        for k in sorted(kinds):
            if k in MEM_PROXY and MEM_PROXY[k] not in notes:
                notes.append(MEM_PROXY[k])
        if dens_ok:
            for kk in sorted({i["mem_kind"] for i in infos.values()}):
                fld = "mm2_per_dq_hbm" if kk == "HBM" else "mm2_per_dq_lpddr"
                if getattr(b, fld, None) is None:
                    notes.append(f"预算未填 {fld}：{kk} PHY 面积未计（密度预设含该项）")
                else:
                    notes.append(PHY_AREA_PROXY[kk])
            if any(x > 0 for x in slcs) and b.mm2_per_mib_slc is None:
                notes.append("预算未填 mm2_per_mib_slc：SLC 面积未计")
        if any(i["mem_kind"] != "HBM" for i in infos.values()) and t.mW_lpddr_ch_standby is None \
                and t.mW_lpddr_Gb_refresh is None:
            notes.append("能耗表未填 LPDDR 后台 / 刷新（mW_lpddr_ch_standby / mW_lpddr_Gb_refresh）：LPDDR 点的静态功耗偏低")
    L = ch.noc_link_Bpc if noc_link_Bpc is None else float(noc_link_Bpc)
    RT = ch.noc_routing if noc_routing is None else noc_routing
    if Ls:
        L = Ls[-1]                                   # notes below quote the widest link of the axis
        kd = b.mm2_per_kmac if b is not None else None
        notes.append(f"NoC 链路宽度轴（0.306）：{{{', '.join(f'{x:g}' for x in Ls)}}} B/周期/方向；每点计路由器面积与静态功耗——"
                     "交叉开关 ∝ (端口·位宽)²、缓冲 / 数据通路 ∝ 端口·位宽「推导/假设」，时钟底功耗 ∝ 位宽、泄漏 ∝ 面积「假设」；"
                     + NOC_COST_SRC + "；工艺：面积 × (用户 MAC 密度 ÷ Horowitz 45 nm bf16 MAC 5824 µm²)「推导/二手」，功耗 ÷ 2.6「推导」"
                     + ("" if kd else "；未填 mm2_per_kmac：NoC 面积未计（只计功耗）")
                     + "。NoC 路由器时钟底功耗（核心轨 V²、× f）与泄漏（∝ V）进入能耗表（pJ_noc_router_cycle / mW_noc_router_leak），"
                     "在功耗封顶 / DVFS 之前计入（0.307）；面积：路由器单独计一次，不乘 uncore_frac，链路导线已在 uncore 的布线份额内，"
                     "mm2_fixed 已含 NoC 时不再加（MODEL §252.1）；crossbar 拓扑不计（单个大开关，无此模型）")
        # 0.307: router static power enters the energy table → priced before the power cap / DVFS (user values win)
        t = replace(t, pJ_noc_router_cycle=(t.pJ_noc_router_cycle if t.pJ_noc_router_cycle is not None
                                            else NOC_ROUTER_CLK_PJ_45 / NOC_NODE_E),
                    mW_noc_router_leak=(t.mW_noc_router_leak if t.mW_noc_router_leak is not None
                                        else NOC_ROUTER_LEAK_MW_45 / NOC_NODE_E))
    if L:
        notes.append(f"NoC 拥塞：链路 {L:g} B/周期/方向，数据流 NoC 载荷超出拓扑二分带宽时计时（MODEL §248）" if RT == "bisection"
                     else f"NoC 拥塞（逐链路 XY）：链路 {L:g} B/周期/方向，按维序路由最忙链路负载计时（MODEL §249.2）")
    if L and ch.noc_mem_attach == "edge":
        notes.append(f"DRAM 经 NoC 进入阵列（0.305，noc_mem_attach=edge：两列存储 tile，每行引擎各一个接入点，Tenstorrent Blackhole「来源」）："
                     f"可送达 DRAM 带宽 ≤ 2·⌊√E⌋ × {L:g} B/周期 × 频率（ring 4 条），低于存储带宽时 decode 受 NoC 限制（MODEL §250.1）")
    if fs != [1.0]:
        notes.append(f"频率轴（DVFS 工作点）：f/f_max ∈ {{{', '.join(f'{x:g}' for x in fs)}}}，按 {curve} 曲线取 V(f)，核心轨动作能耗 × V²、"
                     f"SRAM 泄漏 × V（与功耗封顶同一机制，MODEL §221/§247）；{VF_CURVES[curve]['label']}「{VF_CURVES[curve]['tag']}」")
        if curve == "dfs":
            notes.append("曲线 dfs 电压不变：降频不降每动作能耗，只有在功耗预算下才可能有收益（选 rx5700xt_n7 看电压收益）")
    if mix:
        notes.append(f"负载混合：每请求 prefill {P} token + decode {mix_out_tokens} token，同卡分时；"
                     "J/请求 = P·J_prefill + D·J_decode，卡秒/请求 = P/吞吐_prefill + D/吞吐_decode「推导」")
    if chip_mode:
        notes.append(f"晶粒切分轴（0.308）：每封装 {{{', '.join(map(str, Ns))}}} 个晶粒 × D2D 接口 {{{', '.join(stds)}}}，每晶粒 {units} 个单元；"
                     f"同一封装总量（MAC / SRAM / SLC / 外存堆栈）平分到各晶粒，按 {die_split.upper()} 切到晶粒上——跨晶粒流量即该并行轴的"
                     "既有集合通信（封装内卡间字节走 D2D 层，schedule），D2D 带宽 = 单元数 × 目录带宽「来源/假设」并受晶粒 NoC 边缘接入限制（0.308）；"
                     "吞吐 / 功耗按封装计（每晶粒 × 晶粒数），功耗预算按封装、每晶粒 ÷ 晶粒数；面积预算 die_mm2 限制封装逻辑硅总面积（Σ 晶粒）")
        if d2d_stds:
            notes.append("D2D pJ/bit 取所选接口目录值（UCIe-A 0.25 / UCIe-S 0.5 / BoW 0.5「来源」；48 / 64 GT/s 沿用 UCIe 2.0 目标「假设」），"
                         "覆盖能耗表 pJ_bit_d2d")
        if reticle_mm2:
            notes.append(f"单晶粒面积 ≤ 光罩 {reticle_mm2:g} mm²（26 × 33 mm「来源」；拼接 / 多次曝光未建模「近似」）")
        if pkg_mm2:
            notes.append(f"封装（中介层）面积 ≤ {pkg_mm2:g} mm²：Σ 晶粒 + HBM 堆栈占地（HBM3/3E 11 × 11 mm「近似」，HBM4 11 × 12.8 mm「二手」），"
                         "间距 / 无源器件未计「近似」" + (f"；{pkg_src}" if pkg_src else ""))
        if d0_per_cm2:
            notes.append(f"良率：D0 = {d0_per_cm2:g} /cm²；" + D0_NOTE + "；kgd_mm2 = 每好封装消耗的晶圆面积 Σ A / Y(A)")
        if L and ch.noc_mem_attach != "none":
            notes.append("D2D 经晶粒边缘接入 NoC（0.308）：D2D 边每列引擎一个接入路由器（mesh ⌈E/R⌉ 条链路 / 边，ring 每边 2），"
                         "可承载 D2D 带宽 ≤ 接入链路 × 链路宽度 × 频率；D2D 字节两端各穿过半个晶粒的 NoC（能耗计入 NoC）")
    rows = []
    for p in grid:
        for mid in mem_ids:
            for slc, fo, lw, (nd, sd) in ((x, y, z, w) for x in slcs for y in fs for z in (Ls or [None]) for w in pk_axis):
                lk = lw if lw is not None else (ch.noc_link_Bpc if noc_link_Bpc is None else float(noc_link_Bpc))
                row = {**p, "array": f'{p["side"]}×{p["side"]}', **infos[mid], "slc_mib": slc, "f_op": fo,
                       "f_ghz": ch.freq_ghz * fo, "noc_link_Bpc": lk,
                       "base": p["base"] and mid == base_mem and slc == float(ch.slc_mib) and fo == 1.0
                       and (lw is None or lw == ch.noc_link_Bpc) and nd == 1}
                if chip_mode:
                    row.update(dies=nd, d2d_std=sd, d2d_units=units if nd > 1 else None, die_split=die_split if nd > 1 else None)
                mid_d = _mem_split(mid, nd)
                if p["engines"] % nd or mid_d is None:
                    rows.append({**row, "feasible": False,
                                 "why": f"{p['engines']} 引擎 / {infos[mid]['mem_units']} 个外存单元不能平分到 {nd} 个晶粒"})
                    continue
                c = replace(ch, engines=p["engines"] // nd, rows=p["side"], cols=p["side"], sram_mib=p["sram_mib"] / nd,
                            noc_topology=p["topology"], slc_mib=slc / nd, noc_link_Bpc=lk, noc_routing=RT,
                            state_sram_mib=ch.state_sram_mib / nd)

                def die_scn(sx, c=c, mid_d=mid_d, nd=nd, sd=sd):
                    sx = sx.replace("chip", c).replace("mem_id", mid_d)
                    if nd == 1:
                        return sx
                    sx = sx.replace("layout", _split_layout(sx.layout, nd, die_split, moe))
                    sx = replace(sx, package_cards=nd, d2d_enabled=True, d2d_std=sd, d2d_units=units,
                                 mem_cap_GB=sx.mem_cap_GB / nd)
                    return sx
                td = replace(t, pJ_bit_d2d=D2D_STANDARDS[sd]["pJ_bit"]) if (sd and d2d_stds) else t
                try:
                    res = {ph: _run(die_scn(scn_ph[ph]), td, cap, power_mode, curve, fo, nd) for ph in phases}
                except ValueError as ex:
                    rows.append({**row, "feasible": False, "why": str(ex)})
                    continue
                why = [w for ph in phases for w in res[ph]["why"]]
                if mix:
                    pf, dc = res["prefill"], res["decode"]
                    ts = P / pf["thr"] + mix_out_tokens / dc["thr"] if pf["thr"] > 0 and dc["thr"] > 0 else float("inf")
                    J = P * pf["J_per_unit"] + mix_out_tokens * dc["J_per_unit"]
                    shares = {k: (P * pf["J_per_unit"] * pf["share"].get(k, 0) + mix_out_tokens * dc["J_per_unit"]
                                  * dc["share"].get(k, 0)) / J for k in pf["share"]} if J > 0 else pf["share"]
                    row.update(unit="请求", fits=pf["fits"] and dc["fits"], bound=f'{pf["bound"]}/{dc["bound"]}',
                               thr=1.0 / ts if ts > 0 else 0.0, J_per_unit=J, W_card=J / ts if ts > 0 else 0.0,
                               share=shares, f_scale=min(pf["f_scale"], dc["f_scale"]),
                               capped=pf["capped"] or dc["capped"], ttft_ms=pf["lat_ms"], tpot_ms=dc["lat_ms"],
                               phases={ph: {k: res[ph][k] for k in ("thr", "J_per_unit", "W_card", "f_scale", "capped",
                                                                    "lat_ms", "bound")} for ph in phases},
                               hbm_static_W=dc["hbm_static_W"])
                    row["thr_nominal"] = 1.0 / (P / pf["thr_nominal"] + mix_out_tokens / dc["thr_nominal"])
                else:
                    o = res[phases[0]]
                    row.update({k: v for k, v in o.items() if k not in ("why", "lat_ms", "noc_W")})
                    row["tpot_ms" if phases[0] == "decode" else "ttft_ms"] = o["lat_ms"]
                if res[phases[-1]].get("noc_W") is not None:
                    row["noc_W"] = res[phases[-1]]["noc_W"]
                s_full = die_scn(s0)
                if dens_ok:
                    fa = full_area(s_full, b, noc_routers=Ls is not None)    # 0.307: routers counted once (budget.py)
                    row["mm2"] = fa["mm2"]
                    if "noc_routers" in fa["terms"]:
                        row["noc_mm2"] = fa["terms"]["noc_routers"] * nd
                    row["si_mm2"] = fa["mm2"] * nd            # 0.308: logic silicon per package (Σ dies)
                    if "d2d_phy" in fa["terms"]:
                        row["d2d_phy_mm2"] = fa["terms"]["d2d_phy"] * nd
                    if d0_per_cm2:
                        import math as _m
                        y = _m.exp(-fa["mm2"] * d0_per_cm2 / 100.0)
                        row["die_yield"] = y
                        row["kgd_mm2"] = nd * fa["mm2"] / y
                if chip_mode and nd > 1:
                    lk_d = s_full.d2d_link
                    row["d2d_GBps"] = lk_d.GBps
                    from .d2d_catalog import d2d_GBps as _dg
                    row["d2d_noc_capped"] = lk_d.GBps < _dg(sd, units) * (1 - 1e-12)
                    ph_ = res[phases[-1]]
                    row["io_share"] = (ph_.get("share") or {}).get("io")     # D2D (+ link / net when the replica spans packages)
                if prices:
                    pr = prices.get(row["mem_type"])
                    row["mem_usd"] = row["mem_GB"] * pr if pr is not None else None
                if not row["fits"]:
                    why.insert(0, "放不下（内存）")
                if area_lim is not None and dens_ok and row.get("si_mm2") is not None and row["si_mm2"] > area_lim * (1 + 1e-9):
                    why.append(f"面积 {row['si_mm2']:.0f} > {area_lim:g} mm²" + (f"（{nd} 晶粒合计）" if nd > 1 else ""))
                if reticle_mm2 and dens_ok and row.get("mm2") is not None and row["mm2"] > reticle_mm2 * (1 + 1e-9):
                    why.append(f"单晶粒 {row['mm2']:.0f} > 光罩 {reticle_mm2:g} mm²")
                if pkg_mm2 and dens_ok and row.get("si_mm2") is not None:
                    row["pkg_mm2"] = row["si_mm2"] + _hbm_footprint(mid)
                    if row["pkg_mm2"] > pkg_mm2 * (1 + 1e-9):
                        why.append(f"封装 {row['pkg_mm2']:.0f} > 中介层 {pkg_mm2:g} mm²")
                if chip_mode and nd > 1 and dens_ok and row.get("mm2"):
                    from .budget import d2d_shoreline
                    ds = d2d_shoreline(s_full, row["mm2"])
                    if ds and ds["status"] == "over":
                        why.append(f"D2D 岸线：{ds['units']} 单元需 {ds['need_mm']:.1f} mm > {ds['edges']} 边 {ds['edge_mm']:.1f} mm")
                if dens_ok and row["mem_kind"] == "HBM" and row.get("mm2"):
                    sh = shoreline_check(s_full, row["mm2"])
                    if sh:
                        row["shoreline"] = sh["status"]      # "tight" = needs the 3rd / 4th die edge (kept, flagged)
                    if sh and sh["status"] == "over":
                        why.append(f"HBM 岸线：{sh['stacks']} 堆栈需 {sh['need_mm']:.0f} mm > 四边 {sh['edge_max_mm']:.0f} mm")
                if slo_tpot_ms is not None and row.get("tpot_ms") is not None and row["tpot_ms"] > slo_tpot_ms * (1 + 1e-9):
                    why.append(f"TPOT {row['tpot_ms']:.1f} > {slo_tpot_ms:g} ms")
                if slo_ttft_ms is not None and row.get("ttft_ms") is not None and row["ttft_ms"] > slo_ttft_ms * (1 + 1e-9):
                    why.append(f"TTFT {row['ttft_ms']:.0f} > {slo_ttft_ms:g} ms")
                if max_mem_usd is not None:
                    if row.get("mem_usd") is None:
                        why.append(f"{row['mem_type']} 未填 $/GB")
                    elif row["mem_usd"] > max_mem_usd * (1 + 1e-9):
                        why.append(f"内存 ${row['mem_usd']:.0f} > ${max_mem_usd:g}")
                row["feasible"] = not why
                if why:
                    row["why"] = "；".join(dict.fromkeys(why))
                row["perf_W"] = 1.0 / row["J_per_unit"] if row["J_per_unit"] > 0 else None
                rows.append(row)
    ok = [i for i, x in enumerate(rows) if x.get("feasible")]
    sub = [rows[i] for i in ok]
    f2 = [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1)), tie)]
    for i in f2:
        rows[i]["front"] = True
    if dens_ok:
        for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("si_mm2", -1)), tie)]:
            rows[i]["front3"] = True        # 0.308: Σ die area per package (= mm2 for a monolithic die)
        if d0_per_cm2:
            for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("kgd_mm2", -1)), tie)]:
                rows[i]["front_kgd"] = True
    priced = bool(prices) and sub and all(x.get("mem_usd") is not None for x in sub)
    if priced:
        for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("mem_usd", -1)), tie)]:
            rows[i]["front_usd"] = True
    best = {}
    if sub:
        def pick(i):
            x = rows[i]
            return {"index": i, "engines": x["engines"], "array": x["array"], "macs": x["macs"], "sram_mib": x["sram_mib"],
                    "topology": x["topology"], "mem_id": x["mem_id"], "slc_mib": x["slc_mib"], "f_op": x["f_op"],
                    "noc_link_Bpc": x["noc_link_Bpc"], "dies": x.get("dies", 1), "d2d_std": x.get("d2d_std"),
                    "thr": x["thr"],
                    "J_per_unit": x["J_per_unit"], "mm2": x.get("mm2")}
        best["perf_W"] = pick(min(ok, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"])))
        best["thr"] = pick(max(ok, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"])))
    base = next((i for i, x in enumerate(rows) if x["base"]), None)
    split = _split_compare(rows) if chip_mode else None
    return {"chiplet": {"dies": Ns, "d2d_stds": stds, "d2d_units": units, "die_split": die_split,
                        "reticle_mm2": reticle_mm2, "pkg_mm2": pkg_mm2, "d0_per_cm2": d0_per_cm2,
                        "compare": split} if chip_mode else None,
            "chip": ch.name, "model": scn.model, "phase": "mix" if mix else phases[0], "rows": rows,
            "front": sorted(f2, key=lambda i: rows[i]["thr"]), "best": best, "base": base, "notes": notes,
            "power_cap_W": cap, "power_cap_src": cap_src, "power_mode": power_mode, "curve": curve,
            "die_mm2": area_lim, "tie": tie, "area": dens_ok, "n_feasible": len(ok), "mems": mem_ids, "slc_mibs": slcs, "f_scales": fs, "noc_link_Bpc": None if Ls else L, "noc_routing": RT,
            "noc_links": Ls,
            "mix": {"prompt": P, "out_tokens": mix_out_tokens} if mix else None,
            "slo": {"tpot_ms": slo_tpot_ms, "ttft_ms": slo_ttft_ms}, "priced": bool(priced),
            "basis": "设计网格（MAC 总数 × 引擎边长 × SRAM × NoC 拓扑 × 外存 × SLC × 频率工作点）逐点评估：时间 / 放得下来自代价模型，"
                     "J/单位 来自能耗表（MAC / SRAM / 导线 / NoC / DRAM+PHY / 静态）；perf/W = 1 / (J/单位)；超功耗点按 DVFS "
                     "封顶运行（或剔除），超面积 / HBM 岸线 / SLO / 内存 $ 的点剔除；前沿 = 吞吐↑ 与 J/单位↓ 的非支配集，"
                     "差 ≤ tie（默认 1 %）视为持平（MODEL §245–247）"}
