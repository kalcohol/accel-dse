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
              mm2_per_dq_*); buildability (0.312) = budget.shoreline_report per die (0.313: + thermal.thermal_report with a cooling
              preset; shore_strict = memory on ≤ 2 edges): memory units, D2D PHY and host +
              scale-up I/O lanes per edge; an infeasible die is flagged (row.buildable / why) and kept off the fronts
              unless include_unbuildable=True
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
HBM_FOOTPRINT_DEFAULT = 11.0 * 11.0     # HBM3 / 3E: 11 × 11 mm (Micron HBM3E product brief 「来源」, 0.313; was 「近似」)
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


# 0.310: chiplet economics — cost per good package = Σ good-die cost + packaging + test / assembly (HBM excluded:
# the same stacks in every point of one memory; added as mem_usd when priced).  Every figure is an estimate:
#   wafer: TSMC N5 300 mm foundry sale price $16,988 (CSET 2020 "AI Chips" model 「二手」);
#   dies per wafer: π R² / A − π · 2R / √(2A), R = 150 mm (standard gross-die approximation 「来源」; no scribe /
#     edge exclusion 「近似」); good die = die / Y (Poisson Y, d0_per_cm2; KGD-tested dies, so bad dies are not packaged);
#   packaging per mm² of the package area (Σ dies + HBM footprints): CoWoS-S $750 per H100 (Silicon Analysts 2026
#     「二手」) over 814 + 6 × 121 mm² → $0.49 / mm²; CoWoS-L $1,100 per B200 over 2 × 800 + 8 × 121 mm² → $0.43 / mm²
#     「推导」 (interposer yield is inside these figures);
#   test / assembly $920 per package (H100, Silicon Analysts 「二手」); multi-die packages +22.5 % (KGD testing adds
#     15–30 % to test cost, same source, midpoint 「二手」);
#   assembly yield 1.0 (bonding loss of good dies not modelled 「假设」).
COST_PRESETS = {
    "n5_cowos_s": {"wafer_usd": 16988.0, "pkg_usd_per_mm2": 750.0 / (814 + 6 * 121), "test_usd": 920.0,
                   "kgd_test_extra": 0.225, "asm_yield": 1.0,
                   "src": "N5 晶圆 $16,988（CSET 2020「二手」）；CoWoS-S $750 / H100（Silicon Analysts「二手」）÷ (814 + 6×121) mm²「推导」；"
                          "测试 / 组装 $920 / 封装、多晶粒 KGD +22.5 %（同源 15–30 % 中值「二手」）"},
    "n5_cowos_l": {"wafer_usd": 16988.0, "pkg_usd_per_mm2": 1100.0 / (2 * 800 + 8 * 121), "test_usd": 920.0,
                   "kgd_test_extra": 0.225, "asm_yield": 1.0,
                   "src": "N5 晶圆 $16,988（CSET 2020「二手」）；CoWoS-L $1,100 / B200（Silicon Analysts「二手」）÷ (2×800 + 8×121) mm²「推导」；"
                          "测试 / 组装 $920 / 封装、多晶粒 KGD +22.5 %「二手」"},
}
COST_KEYS = ("wafer_usd", "pkg_usd_per_mm2", "test_usd", "kgd_test_extra", "asm_yield")
WAFER_R_MM = 150.0


def _cost_params(cost):
    """None | preset name | dict (a ``preset`` key plus any COST_KEYS overrides) → the cost terms."""
    if cost is None:
        return None
    if isinstance(cost, str):
        cost = {"preset": cost}
    if not isinstance(cost, dict):
        raise ValueError(f"cost must be a preset name ({sorted(COST_PRESETS)}) or an object")
    pr = cost.get("preset", "n5_cowos_s")
    if pr not in COST_PRESETS:
        raise ValueError(f"cost preset must be one of {sorted(COST_PRESETS)}")
    out = dict(COST_PRESETS[pr], preset=pr)
    for k, v in cost.items():
        if k == "preset":
            continue
        if k not in COST_KEYS:
            raise ValueError(f"unknown cost key {k!r} (one of {COST_KEYS})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1e7:
            raise ValueError(f"cost.{k} must be a number in [0, 1e7)")
        out[k] = float(v)
    if not 0 < out["asm_yield"] <= 1:
        raise ValueError("cost.asm_yield must be in (0, 1]")
    if not out["wafer_usd"] > 0:
        raise ValueError("cost.wafer_usd must be > 0")
    if k_ := [k for k in cost if k != "preset"]:
        out["src"] = out["src"] + "；用户覆盖：" + "、".join(k_)
    return out


def dies_per_wafer(mm2: float) -> float:
    """Gross dies per 300 mm wafer: π R² / A − π · 2R / √(2A) (≥ 0)."""
    import math
    return max(0.0, math.pi * WAFER_R_MM ** 2 / mm2 - math.pi * 2 * WAFER_R_MM / math.sqrt(2 * mm2))


def package_cost(cp: dict, die_mm2: float, n_dies: int, die_yield: float, pkg_area_mm2: float) -> dict:
    """0.310: cost per good package (USD) from the cost terms ``cp`` (see COST_PRESETS)."""
    dpw = dies_per_wafer(die_mm2)
    die = cp["wafer_usd"] / dpw if dpw > 0 else float("inf")
    good = die / die_yield if die_yield > 0 else float("inf")
    pkg = cp["pkg_usd_per_mm2"] * pkg_area_mm2
    test = cp["test_usd"] * (1 + (cp["kgd_test_extra"] if n_dies > 1 else 0.0))
    tot = (n_dies * good + pkg + test) / cp["asm_yield"]
    return {"dpw": dpw, "die_usd": die, "good_die_usd": good, "si_usd": n_dies * good, "pkg_usd": pkg,
            "test_usd": test, "usd_good_pkg": tot}


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


def _run(s, t, cap, power_mode, curve, f_op: float = 1.0, n_dies: int = 1, pk=None) -> dict:
    """0.308: a package of ``n_dies`` dies (one card each in the cost model): every die runs under cap / n and the
    per-card figures are scaled to the package (throughput, W, static W × n; J/unit and latency unchanged)."""
    o = _run1(s, t, None if cap is None else cap / n_dies, power_mode, curve, f_op, pk)
    if n_dies > 1:
        for k in ("thr", "thr_nominal", "W_card", "W_nominal", "hbm_static_W", "noc_W", "dram_dev_W", "static_W",
                  "idle_W", "cap_pkg_W"):
            if o.get(k) is not None:
                o[k] *= n_dies
        o["why"] = [f"每晶粒（预算 ÷ {n_dies}）{w}" for w in o["why"]]
    return o


def _run1(s, t, cap, power_mode, curve, f_op: float = 1.0, pk=None) -> dict:
    """One phase of one design point: at the operating point f_op (0.302; clock f_op·f_max, voltage V(f_op) on the
    curve, core-rail energies × V², leakage × V — powercap._point), then capped / rejected by power.  A point over the
    cap is capped from f_max like any other: power rises with f, so the cap clock is below f_op and equals the f_op = 1
    point's capped result."""
    from .powercap import _point, dvfs_cap, static_W
    from .thermal import dram_dev_W
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
         "noc_W": (er.get("noc_router") or {}).get("W"),      # 0.307: router static W at the operating point
         "dram_dev_W": dram_dev_W(er),                         # 0.313: DRAM-device W (heat in the stacks, thermal)
         "static_W": static_W(er), "idle_W": er.get("idle_W_used")}   # 0.315: static in W_card (thermal.package_power)
    r = r0
    # 0.316: the cap compares the package power (thermal.cap_power_fn → package_power, the thermal check's function)
    # unless power_scope "card" (pk None, 0.315)
    o["cap_pkg_W"] = o["W_card"] if pk is None else pk(o["W_card"], o["static_W"], o["idle_W"])
    if cap is not None and o["cap_pkg_W"] > cap * (1 + 1e-9) and r.fits:
        if power_mode == "reject":
            o["why"].append(f"功耗 {o['cap_pkg_W']:.0f} > {cap:g} W" + ("（封装口径）" if pk is not None else ""))
        else:
            d = dvfs_cap(r, t, cap, curve, iters=12, pk=pk)
            if d.get("error") or not d.get("feasible", True):
                o["why"].append(d.get("error") or "功耗封顶不可行")
            else:
                k = d["throughput_factor"]
                if o["W_card"] > 0:     # 0.313: the DRAM-device share of the capped power (same traffic per unit)
                    o["dram_dev_W"] = o["dram_dev_W"] * d["W_card"] / o["W_card"]
                o.update(thr=r.per_card * k, J_per_unit=d["J_per_unit"], W_card=d["W_card"], f_scale=d["s"],
                         capped=True, lat_ms=o["lat_ms"] / k if k > 0 else o["lat_ms"], cap_pkg_W=d.get("pkg_W", d["W_card"]),
                         static_W=d.get("static_W", o["static_W"]) if pk is not None else o["static_W"])
                # 0.316: package scope carries the static at the capped clock (the cap and the thermal check then see
                # the same package power); card scope keeps the nominal static (0.315)
    return o


def _verdict(rows, sp_thr, sp_J, monos, tie):
    """Best split (by thr, by J) against the best of ``monos`` (each by its own metric)."""
    if not monos:
        return "only_split", None, None, None, None
    mt = max(monos, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"]))
    mj = min(monos, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"]))
    d_thr = rows[sp_thr]["thr"] / rows[mt]["thr"] - 1 if rows[mt]["thr"] > 0 else None
    d_J = rows[sp_J]["J_per_unit"] / rows[mj]["J_per_unit"] - 1 if rows[mj]["J_per_unit"] > 0 else None
    bthr = d_thr is not None and d_thr > tie
    bJ = d_J is not None and d_J < -tie
    return ("split_both" if bthr and bJ else "split_thr" if bthr else "split_J" if bJ else "mono"), d_thr, d_J, mt, mj


def _split_compare(rows: list[dict], tie: float = 0.01) -> list[dict]:
    """0.308: per design (same package totals: MACs, SRAM, engine side, topology, memory, SLC, clock, NoC link), the
    monolithic die (dies = 1) against its best feasible split (by throughput and by J/unit).  Verdicts: only_split
    (the monolithic die is infeasible — reticle / area / power / shoreline — and a split is feasible), split_thr /
    split_J / split_both (better by more than ``tie``), mono (no split beats it), none (nothing feasible).
    0.309: the monolithic side is the best feasible dies = 1 row — flat or split into K NoC clusters (mono_clusters) —
    by each metric; ``verdict_flat`` / ``d_thr_flat`` / ``d_J_flat`` keep the 0.308 flat-die-only comparison."""
    groups: dict = {}
    for i, x in enumerate(rows):
        k = (x["mac_x"], x["sram_x"], x["side"], x["topology"], x["mem_id"], x["slc_mib"], x["f_op"], x["noc_link_Bpc"])
        groups.setdefault(k, []).append(i)
    out = []
    for k, ix in groups.items():
        mono = next((i for i in ix if rows[i].get("dies", 1) == 1 and rows[i].get("clusters", 1) == 1), None)
        monos = [i for i in ix if rows[i].get("dies", 1) == 1 and rows[i].get("feasible")]
        sp = [i for i in ix if rows[i].get("dies", 1) > 1 and rows[i].get("feasible")]
        if not any(rows[i].get("dies", 1) > 1 for i in ix):
            continue
        flat_ok = bool(mono is not None and rows[mono].get("feasible"))
        e = {"mac_x": k[0], "sram_x": k[1], "side": k[2], "topology": k[3], "mem_id": k[4], "slc_mib": k[5],
             "f_op": k[6], "noc_link_Bpc": k[7], "mono": mono, "mono_feasible": bool(monos),
             "mono_flat_feasible": flat_ok, "mono_why": rows[mono].get("why") if mono is not None else None}
        if not sp:
            e["verdict"] = e["verdict_flat"] = "mono" if monos else "none"
            if not flat_ok:
                e["verdict_flat"] = "none"
            out.append(e)
            continue
        bt = max(sp, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"]))
        bj = min(sp, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"]))
        e.update(best_thr=bt, best_J=bj)
        v, dt, dj, mt, mj = _verdict(rows, bt, bj, monos, tie)
        e.update(verdict=v, mono_thr=mt, mono_J=mj)
        if dt is not None or dj is not None:
            e.update(d_thr=dt, d_J=dj)
        vf, dtf, djf, _, _ = _verdict(rows, bt, bj, [mono] if flat_ok else [], tie)
        e.update(verdict_flat=vf, d_thr_flat=dtf, d_J_flat=djf)
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
                         pkg_mm2=None, d0_per_cm2: float | None = None, mono_clusters: bool = True,
                         mem_edges_shoreline: bool = True, clusters=None, cost=None,
                         include_unbuildable: bool = False, shore_strict: bool = False,
                         cooling: str | None = None, power_scope: str = "package") -> dict:
    from .. import mem_catalog as mc
    from .budget import full_area, shoreline_report
    from .thermal import COOLING, thermal_report
    from .energy import idle_W_of
    from .thermal import STATIC_FLOOR_W_MM2, VR_EFF
    if cooling is not None and cooling not in COOLING:
        raise ValueError(f"cooling must be one of {sorted(COOLING)}")
    from .thermal import POWER_SCOPES, cap_power_fn
    if power_scope not in POWER_SCOPES:
        raise ValueError(f"power_scope must be one of {POWER_SCOPES}")
    static_in = idle_W_of(table) is not None      # 0.314: the table's idle figure (0.315: else the thermal static floor)
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
    # 0.310: NoC clusters of the monolithic die as an axis of its own (also without the dies axis)
    if clusters is not None:
        if not clusters or len(clusters) > 5 or not all(not isinstance(x, bool) and float(x).is_integer()
                                                        and 1 <= int(x) <= 64 for x in clusters):
            raise ValueError("clusters must be a list of 1–5 integers in [1, 64]")
        if scn.d2d_enabled and scn.package_cards > 1:
            raise ValueError("clusters 轴从单片设计出发：场景已是多晶粒封装（d2d_enabled / package_cards > 1）")
    Ks = sorted({int(x) for x in clusters if int(x) > 1}) if clusters is not None else \
        ([nn for nn in Ns if nn > 1] if (chip_mode and mono_clusters) else [])
    clu_mode = chip_mode or clusters is not None
    from .catalog import get_model
    moe = bool(get_model(scn.model).is_moe) if clu_mode else False
    # (dies, D2D standard, NoC clusters per die).  0.309: with the die axis on, a monolithic die is also searched
    # split into K NoC clusters for every K > 1 in the die axis (same partition the K-die package gets; mono_clusters)
    pk_axis = ([(1, None, 1)] + [(1, None, k) for k in Ks]
               + ([(nn, sd, 1) for nn in Ns if nn > 1 for sd in stds] if chip_mode else []))
    if chip_mode and 1 not in Ns:
        pk_axis = [x for x in pk_axis if x[0] > 1]
    mem_edges_on = bool(chip_mode and mem_edges_shoreline)
    # 0.310: memory edges are a mechanism default (Chip.noc_mem_edges = 0 = auto): every row resolves them from its
    # own die (the search's densities when complete, else the Scenario's proxy); a fixed chip value is kept, except
    # that the chiplet comparison applies its 0.309 HBM rule (mem_edges_shoreline) and False forces 2 (0.308)
    auto_edges = scn.chip.noc_mem_edges == 0
    cost_p = _cost_params(cost)
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
    elif power_scope == "package":
        notes.append(f"功耗预算 {cap:g} W 按封装功耗比较（0.316）：与热判定同一函数 thermal.package_power（能耗表功耗 + "
                     + ("能耗表 idle 静态（板级，不再计稳压器损耗）" if static_in else "整卡静态底功耗 + 稳压器 / PDN 损耗")
                     + "），DVFS 封顶点的封装功耗 ≤ 预算；power_scope=card 复现 0.315（只比能耗表卡功耗）")
    else:
        notes.append(f"功耗预算 {cap:g} W 按能耗表卡功耗比较（power_scope=card，0.315 口径；不含静态底功耗与稳压器损耗）")
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
        if 1 in Ns and mono_clusters and any(nn > 1 for nn in Ns):
            notes.append("单片对照（0.309）：单片也按同一 K 划成片上 NoC 簇（层次化 NoC；A100/H100 L2 两分区、GPC 直连本分区「来源」）——"
                         "簇间层 = 片上全局链路：带宽 = 簇边界接入链路 × 链路宽度 × 频率（链路宽度未设 = 不限），能耗按每端半跨 NoC "
                         "+ 一跳一节距（pJ_bit_hop_noc / pJ_bit_mm_onchip，Keckler 2011「来源」），无 D2D PHY 能耗与面积；"
                         "面积 / 良率 / 光罩 / 岸线按整片计；单片判定取平面与分簇中各指标最优者，verdict_flat = 0.308 仅平面对照")
        if mem_edges_on:
            notes.append("外存接入边（0.309）：HBM 堆栈占用几条晶粒边（⌈堆栈 × 11 mm / 边长⌉，≤ 4 − D2D 边），NoC 就在几条边接入外存"
                         "（第 3 / 4 边各加一行接入路由器）——单片与拆分按同一岸线规则，DRAM 跳数 / 接入链路随之「推导」")
        if L and ch.noc_mem_attach != "none":
            notes.append("D2D 经晶粒边缘接入 NoC（0.308）：D2D 边每列引擎一个接入路由器（mesh ⌈E/R⌉ 条链路 / 边，ring 每边 2），"
                         "可承载 D2D 带宽 ≤ 接入链路 × 链路宽度 × 频率；D2D 字节两端各穿过半个晶粒的 NoC（能耗计入 NoC）")
    if clusters is not None and Ks:
        notes.append(f"NoC 簇轴（0.310）：单片按 K = {{{', '.join(map(str, Ks))}}} 划成片上 NoC 簇（层次化 NoC，0.309 机理），"
                     f"簇间按 {die_split.upper()} 切分；面积 / 良率按整片")
    if auto_edges and not (chip_mode and not mem_edges_shoreline):
        notes.append("外存接入边（0.310 机理默认）：每个设计点按其晶粒的外存岸线占用（HBM 堆栈 × 11 mm；LPDDR 位宽 ÷ 18.75 引脚 / mm，"
                     "Credo 0.18 Tb/s/mm @ 9.6 GT/s「二手/推导」）决定几条边接入 NoC；芯片 noc_mem_edges = 2 复现 0.309。"
                     "0.312：主机 I/O 按通道定长（PCIe x16 × 0.285 mm + scale-up 链路 ÷ 通道速率 × 每通道边长；0.313 默认 200G × 0.5 mm「来源」，100G × 0.285 mm = 0.312），外存按整堆栈 / 整通道放边，"
                     "晶粒在 26 × 33 曝光场内可取长宽比 ≤ 2 的矩形（noc_mem_io_edges = 1 复现 0.311 整边保留，0 复现 0.310），"
                     "地址均匀交织 → 各边按其上外存单元份额承载流量，最重的边限制 NoC 接入（noc_mem_balance = pooled 复现 0.310）")
    if shore_strict:
        notes.append("岸线严格（0.313，shore_strict）：外存占第 3 / 4 边（tight）视为不可建——出货晶粒（GH100、B200、MI300X IOD、"
                     "TPU v5p、Trainium2、MI250X GCD）外存不超过两条边「二手」；第三边仅见于专利（Google US 12,315,860 B2 "
                     "权利要求 14）「来源」，无出货先例；默认（关）沿用 0.310 起的四边几何上限「假设」")
    notes.append("热 / 功耗（0.313；0.314 口径统一；0.315 封装功耗完整）：每点报告封装功耗（模块输入侧，与出货件 TDP 同口径："
                 "能耗表功耗" + ("（含能耗表 idle 整卡静态）" if static_in else
                              f" + 整卡静态底功耗 {STATIC_FLOOR_W_MM2:g} W/mm² × 晶粒面积（扣已计泄漏 / HBM 待机）")
                 + f" + 稳压器 / PDN 损耗（效率 {VR_EFF:g}「近似」））、热流密度（判定用：封装功耗 ÷ 逻辑晶粒总面积，与出货件 "
                 "TDP ÷ 晶粒面积同口径；晶粒发热（扣 DRAM 器件与稳压器损耗）对 HIR 晶粒热流上限）与 0.75 V「假设」下的电流密度；"
                 "静态按空闲温度实测，85 °C 泄漏上界下超限的「可建」点标记余量不足（hot_ok）；"
                 + (f"按 {cooling} 判定（{COOLING[cooling]['pkg_W']:g} W / {COOLING[cooling]['flux_W_mm2']:g} W/mm²"
                    f"（晶粒 {COOLING[cooling]['die_flux_W_mm2']:g}）；"
                    + COOLING[cooling]["src"] + "）；热超限点与岸线不可建点同样移出 Pareto 前沿（0.316，"
                    "front_thermal_out 标出本在前沿的点；include_unbuildable 保留）"
                    if cooling else "未选冷却方案（cooling）：只报告不判定，不影响前沿")
                 + "；供电（凸点电流）无公开上限，只报告")
    if cost_p is not None:
        notes.append("成本 / 好封装（0.310）：Σ 晶粒 × 晶圆价 / DPW / Y + 封装 $/mm² × (Σ 晶粒 + HBM 占地) + 测试 / 组装（多晶粒 + KGD）"
                     "，HBM 不计（同一外存档相同）；DPW = πR²/A − π·2R/√(2A)（R = 150 mm「来源」，无划片 / 边缘排除「近似」）；"
                     + cost_p["src"] + ("" if d0_per_cm2 else "；未设 D0：良率按 1 计"))
    rows = []
    for p in grid:
        for mid in mem_ids:
            for slc, fo, lw, (nd, sd, kc) in ((x, y, z, w) for x in slcs for y in fs for z in (Ls or [None]) for w in pk_axis):
                npart = nd if nd > 1 else kc        # cards the evaluator sees: dies, or the clusters of one die
                lk = lw if lw is not None else (ch.noc_link_Bpc if noc_link_Bpc is None else float(noc_link_Bpc))
                row = {**p, "array": f'{p["side"]}×{p["side"]}', **infos[mid], "slc_mib": slc, "f_op": fo,
                       "f_ghz": ch.freq_ghz * fo, "noc_link_Bpc": lk,
                       "base": p["base"] and mid == base_mem and slc == float(ch.slc_mib) and fo == 1.0
                       and (lw is None or lw == ch.noc_link_Bpc) and nd == 1 and kc == 1}
                if chip_mode:
                    row.update(dies=nd, d2d_std=sd, d2d_units=units if nd > 1 else None,
                               die_split=die_split if npart > 1 else None, clusters=kc)
                elif clu_mode:
                    row.update(clusters=kc, die_split=die_split if kc > 1 else None)
                mid_d = _mem_split(mid, npart)
                if p["engines"] % npart or mid_d is None:
                    rows.append({**row, "feasible": False,
                                 "why": f"{p['engines']} 引擎 / {infos[mid]['mem_units']} 个外存单元不能平分到 {npart} 个"
                                        + ("晶粒" if nd > 1 else " NoC 簇")})
                    continue
                c = replace(ch, engines=p["engines"] // npart, rows=p["side"], cols=p["side"],
                            sram_mib=p["sram_mib"] / npart, noc_topology=p["topology"], slc_mib=slc / npart,
                            noc_link_Bpc=lk, noc_routing=RT, state_sram_mib=ch.state_sram_mib / npart, noc_clusters=kc)
                if chip_mode and not mem_edges_shoreline:
                    c = replace(c, noc_mem_edges=2)                  # the 0.308 view

                def die_scn(sx, c, mid_d=mid_d, npart=npart, nd=nd, sd=sd):
                    if npart == 1:
                        return sx.replace("chip", c).replace("mem_id", mid_d)
                    # clusters: the chip goes in last — noc_clusters > 1 needs d2d_enabled / package_cards = K first
                    sx = sx.replace("chip", replace(c, noc_clusters=1)).replace("mem_id", mid_d)
                    sx = sx.replace("layout", _split_layout(sx.layout, npart, die_split, moe))
                    # 0.312: the package keeps its scale-up bandwidth — each die / cluster card gets Link.GBps / npart
                    # (≤ 0.311 gave every part the whole link: K× the package's off-package bandwidth and SerDes)
                    sx = replace(sx, package_cards=npart, d2d_enabled=True, mem_cap_GB=sx.mem_cap_GB / npart,
                                 link=replace(sx.link, GBps=sx.link.GBps / npart),
                                 **({"d2d_std": sd, "d2d_units": units} if nd > 1 else {}))
                    return replace(sx, chip=c) if c.noc_clusters > 1 else sx
                try:
                    s_full = die_scn(s0, c)
                except ValueError as ex:
                    rows.append({**row, "feasible": False, "why": str(ex)})
                    continue
                if dens_ok:
                    fa = full_area(s_full, b, noc_routers=Ls is not None)    # 0.307: routers counted once (budget.py)
                    if mem_edges_on or (auto_edges and not chip_mode):
                        # 0.309: memory attach on every die edge the HBM stacks occupy (0.310: + LPDDR when auto)
                        from .budget import hbm_mem_edges
                        me = hbm_mem_edges(s_full, fa["mm2"], lpddr=auto_edges)
                        if me != c.noc_mem_edges:
                            c = replace(c, noc_mem_edges=me)
                            s_full = die_scn(s0, c)
                row["mem_edges"] = s_full.chip.mem_edges
                td = replace(t, pJ_bit_d2d=D2D_STANDARDS[sd]["pJ_bit"]) if (sd and d2d_stds) else t
                # 0.316: the cap's package power per die / cluster card over the die the thermal check uses (the
                # search densities, else the shoreline proxy) — one area for both
                sr0 = None if dens_ok else shoreline_report(s_full, None)
                die_a = fa["mm2"] if dens_ok else (sr0 or {}).get("die_mm2")
                pk = cap_power_fn(die_a * nd / npart if die_a else None, 1, power_scope) if cap is not None else None
                try:
                    res = {ph: _run(die_scn(scn_ph[ph], c), td, cap, power_mode, curve, fo, npart, pk) for ph in phases}
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
                               hbm_static_W=dc["hbm_static_W"], static_W=dc["static_W"], idle_W=dc["idle_W"])
                    fr = {ph: (res[ph]["dram_dev_W"] / res[ph]["W_card"] if res[ph]["W_card"] > 0 else 0.0) for ph in phases}
                    row["dram_dev_W"] = row["W_card"] * ((P * pf["J_per_unit"] * fr["prefill"] + mix_out_tokens
                                                          * dc["J_per_unit"] * fr["decode"]) / J if J > 0 else 0.0)
                    row["thr_nominal"] = 1.0 / (P / pf["thr_nominal"] + mix_out_tokens / dc["thr_nominal"])
                else:
                    o = res[phases[0]]
                    row.update({k: v for k, v in o.items() if k not in ("why", "lat_ms", "noc_W")})
                    row["tpot_ms" if phases[0] == "decode" else "ttft_ms"] = o["lat_ms"]
                if res[phases[-1]].get("noc_W") is not None:
                    row["noc_W"] = res[phases[-1]]["noc_W"]
                if dens_ok:
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
                if cost_p is not None and dens_ok and row.get("mm2"):
                    cc = package_cost(cost_p, row["mm2"], nd, row.get("die_yield", 1.0),
                                      row["si_mm2"] + _hbm_footprint(mid))
                    row.update({k: cc[k] for k in ("dpw", "good_die_usd", "si_usd", "pkg_usd", "test_usd",
                                                   "usd_good_pkg")})
                if prices:
                    pr = prices.get(row["mem_type"])
                    row["mem_usd"] = row["mem_GB"] * pr if pr is not None else None
                if not row["fits"]:
                    why.insert(0, "放不下（内存）")
                if area_lim is not None and dens_ok and row.get("si_mm2") is not None and row["si_mm2"] > area_lim * (1 + 1e-9):
                    why.append(f"面积 {row['si_mm2']:.0f} > {area_lim:g} mm²" + (f"（{nd} 晶粒合计）" if nd > 1 else ""))
                if reticle_mm2 and dens_ok and row.get("mm2") is not None and row["mm2"] > reticle_mm2 * (1 + 1e-9):
                    why.append(f"单晶粒 {row['mm2']:.0f} > 光罩 {reticle_mm2:g} mm²")
                sr = sr0 if sr0 is not None else shoreline_report(s_full, row.get("mm2") if dens_ok else None)
                if pkg_mm2 and dens_ok and row.get("si_mm2") is not None:
                    row["pkg_mm2"] = row["si_mm2"] + _hbm_footprint(mid)
                    if nd == 1 and (sr.get("footprint") or {}).get("bbox_mm2"):
                        # 0.313: one die — its placed stacks' bounding box (corner squares of a 3–4-sided stack row)
                        row["pkg_mm2"] = max(row["pkg_mm2"], sr["footprint"]["bbox_mm2"])
                    if row["pkg_mm2"] > pkg_mm2 * (1 + 1e-9):
                        why.append(f"封装 {row['pkg_mm2']:.0f} > 中介层 {pkg_mm2:g} mm²")
                # 0.312: buildability is a first-class result — one shoreline report per die (memory units, D2D PHY,
                # host + scale-up I/O per edge; budget.shoreline_report) over the row's die (the search densities, else
                # the n5_derived proxy 「近似」).  Unbuildable rows leave every front unless include_unbuildable.
                feas = sr.get("feasible")
                if shore_strict and feas and sr.get("status") == "tight":
                    # 0.313: no shipping die carries HBM / memory on more than its two long edges (GH100, B200,
                    # MI300X IODs 「二手」) — strict counts a 3–4-sided memory row as unbuildable
                    feas = False
                    sr["reasons"] = sr.get("reasons", []) + [
                        f"外存占 {sr['mem'].get('edges')} 边（> 2 边无出货先例，shore_strict）"]
                row["buildable"] = feas
                row["shoreline"] = sr.get("status")       # ok / tight (memory beyond 2 edges) / over / unknown
                if sr.get("edges"):
                    row["shore"] = {"mem": sr["mem"]["status"], "io": sr["io"]["status"],
                                    "d2d": (sr.get("d2d") or {}).get("status"), "io_mm": sr["io"].get("need_mm"),
                                    "used_mm": sr["used_mm"], "perimeter_mm": sr["perimeter_mm"],
                                    "die_src": "dse" if dens_ok else "proxy"}
                if feas is False:
                    sw = "；".join(sr["reasons"])
                    if include_unbuildable:
                        row["why_shore"] = sw
                    else:
                        why.append(sw)
                # 0.313: thermal / power buildability (always reported; judged only with a cooling preset)
                th = thermal_report(row.get("W_card"), row.get("mm2") if dens_ok else sr.get("die_mm2"), nd,
                                    row.get("dram_dev_W") or 0.0, cooling, row.get("static_W") or 0.0, row.get("idle_W"))
                # 0.314: flux_W_mm2 = card-level (judged, shipped-part definition); die_flux_W_mm2 = DRAM devices out
                # 0.315: pkg_W = complete package power (static floor + VR loss, thermal.package_power)
                row["thermal"] = {k: th.get(k) for k in ("status", "feasible", "pkg_W", "card_W", "die_W", "flux_W_mm2",
                                                         "die_flux_W_mm2", "flux_ratio", "static_included",
                                                         "static_basis", "static_add_W", "vr_loss_W", "pkg_W_hot",
                                                         "hot_ok")}
                row["thermal"]["A_per_mm2"] = (th.get("pd") or {}).get("A_per_mm2")
                tw_added = None
                if th.get("feasible") is False:
                    row["buildable"] = False
                    tw = "；".join(th["reasons"])
                    if include_unbuildable:
                        row["why_thermal"] = tw
                    else:
                        why.append(tw)
                        tw_added = tw       # 0.316: off the front only because of the cooling limit?
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
                row["_ok_wo_thermal"] = (not why) or (tw_added is not None and all(w == tw_added for w in why))
                if why:
                    row["why"] = "；".join(dict.fromkeys(why))
                row["perf_W"] = 1.0 / row["J_per_unit"] if row["J_per_unit"] > 0 else None
                rows.append(row)
    ok = [i for i, x in enumerate(rows) if x.get("feasible")]
    sub = [rows[i] for i in ok]
    f2 = [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1)), tie)]
    for i in f2:
        rows[i]["front"] = True
    # 0.316: points the thermal check took off the front — the front without the cooling filter, less the buildable
    # front (only with a cooling preset; include_unbuildable keeps them on the front anyway)
    n_th_out = 0
    if cooling is not None and not include_unbuildable:
        okw = [i for i, x in enumerate(rows) if x.pop("_ok_wo_thermal", False)]
        for i in [okw[j] for j in pareto([rows[i] for i in okw], (("thr", 1), ("J_per_unit", -1)), tie)]:
            if not rows[i].get("feasible"):
                rows[i]["front_thermal_out"] = True
                n_th_out += 1
    for x in rows:
        x.pop("_ok_wo_thermal", None)
    if dens_ok:
        for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("si_mm2", -1)), tie)]:
            rows[i]["front3"] = True        # 0.308: Σ die area per package (= mm2 for a monolithic die)
        if d0_per_cm2:
            for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("kgd_mm2", -1)), tie)]:
                rows[i]["front_kgd"] = True
    costed = cost_p is not None and bool(sub) and all(x.get("usd_good_pkg") is not None for x in sub)
    if costed:      # 0.310: cost axis — throughput ↑, J ↓, cost per good package ↓
        for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("usd_good_pkg", -1)), tie)]:
            rows[i]["front_cost"] = True
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
            "cost": ({k: cost_p[k] for k in (*COST_KEYS, "preset", "src")} if cost_p else None), "costed": costed,
            "clusters": Ks if clusters is not None else None,
            "include_unbuildable": bool(include_unbuildable),
            "n_unbuildable": sum(1 for x in rows if x.get("buildable") is False),
            "shore_strict": bool(shore_strict), "cooling": cooling,
            "n_thermal_over": sum(1 for x in rows if (x.get("thermal") or {}).get("feasible") is False),
            "n_front_thermal_out": n_th_out, "power_scope": power_scope,
            "basis": "设计网格（MAC 总数 × 引擎边长 × SRAM × NoC 拓扑 × 外存 × SLC × 频率工作点）逐点评估：时间 / 放得下来自代价模型，"
                     "J/单位 来自能耗表（MAC / SRAM / 导线 / NoC / DRAM+PHY / 静态）；perf/W = 1 / (J/单位)；超功耗点按 DVFS "
                     "封顶运行（或剔除），超面积 / 岸线不可建（外存 / D2D / I/O，0.312；include_unbuildable 可纳入）/ SLO / 内存 $ 的点剔除；前沿 = 吞吐↑ 与 J/单位↓ 的非支配集，"
                     "差 ≤ tie（默认 1 %）视为持平（MODEL §245–247）"}
