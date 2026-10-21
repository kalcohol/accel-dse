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


def _run(s, t, cap, power_mode, curve, f_op: float = 1.0) -> dict:
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
         "capped": False, "lat_ms": lat * 1e3, "hbm_static_W": er.get("hbm_static_W_card"), "why": []}
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


def energy_design_search(scn, table: EnergyTable, budget=None, mac_scales=MAC_SCALES, sram_scales=SRAM_SCALES,
                         sides=None, topologies=None, power_mode: str = "dvfs", curve: str = "dfs",
                         phase: str | None = None, prompt: int | None = None, tie: float | None = None,
                         mems=None, mem_types=None, mem_count_scales=None, mem_caps=None, slc_mibs=None,
                         mix_out_tokens: int | None = None, slo_tpot_ms: float | None = None,
                         slo_ttft_ms: float | None = None, mem_usd_per_GB: dict | None = None,
                         max_mem_usd: float | None = None, f_scales=None) -> dict:
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
    fs = sorted({float(x) for x in f_scales}, reverse=True) if f_scales else [1.0]
    if not all(F_MIN <= x <= 1.0 for x in fs):
        raise ValueError(f"f_scales must be in [{F_MIN:g}, 1] (fraction of the chip's nominal clock)")
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
    n_pts = len(grid) * len(mem_ids) * len(slcs) * len(fs)
    if n_pts > MAX_POINTS:
        raise ValueError(f"设计网格 {n_pts} 点 > 上限 {MAX_POINTS}：减少 mac_scales / sram_scales / sides / topologies / 内存 / SLC / 频率档")
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
    if fs != [1.0]:
        notes.append(f"频率轴（DVFS 工作点）：f/f_max ∈ {{{', '.join(f'{x:g}' for x in fs)}}}，按 {curve} 曲线取 V(f)，核心轨动作能耗 × V²、"
                     f"SRAM 泄漏 × V（与功耗封顶同一机制，MODEL §221/§247）；{VF_CURVES[curve]['label']}「{VF_CURVES[curve]['tag']}」")
        if curve == "dfs":
            notes.append("曲线 dfs 电压不变：降频不降每动作能耗，只有在功耗预算下才可能有收益（选 rx5700xt_n7 看电压收益）")
    if mix:
        notes.append(f"负载混合：每请求 prefill {P} token + decode {mix_out_tokens} token，同卡分时；"
                     "J/请求 = P·J_prefill + D·J_decode，卡秒/请求 = P/吞吐_prefill + D/吞吐_decode「推导」")
    rows = []
    for p in grid:
        for mid in mem_ids:
            for slc, fo in ((x, y) for x in slcs for y in fs):
                c = replace(ch, engines=p["engines"], rows=p["side"], cols=p["side"], sram_mib=p["sram_mib"],
                            noc_topology=p["topology"], slc_mib=slc)
                row = {**p, "array": f'{p["side"]}×{p["side"]}', **infos[mid], "slc_mib": slc, "f_op": fo,
                       "f_ghz": ch.freq_ghz * fo,
                       "base": p["base"] and mid == base_mem and slc == float(ch.slc_mib) and fo == 1.0}
                try:
                    res = {ph: _run(scn_ph[ph].replace("chip", c).replace("mem_id", mid), t, cap, power_mode, curve, fo)
                           for ph in phases}
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
                    row.update({k: v for k, v in o.items() if k not in ("why", "lat_ms")})
                    row["tpot_ms" if phases[0] == "decode" else "ttft_ms"] = o["lat_ms"]
                s_full = s0.replace("chip", c).replace("mem_id", mid)
                if dens_ok:
                    row["mm2"] = full_area(s_full, b)["mm2"]
                if prices:
                    pr = prices.get(row["mem_type"])
                    row["mem_usd"] = row["mem_GB"] * pr if pr is not None else None
                if not row["fits"]:
                    why.insert(0, "放不下（内存）")
                if area_lim is not None and dens_ok and row.get("mm2") is not None and row["mm2"] > area_lim * (1 + 1e-9):
                    why.append(f"面积 {row['mm2']:.0f} > {area_lim:g} mm²")
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
        for i in [ok[i] for i in pareto(sub, (("thr", 1), ("J_per_unit", -1), ("mm2", -1)), tie)]:
            rows[i]["front3"] = True
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
                    "thr": x["thr"],
                    "J_per_unit": x["J_per_unit"], "mm2": x.get("mm2")}
        best["perf_W"] = pick(min(ok, key=lambda i: (rows[i]["J_per_unit"], -rows[i]["thr"])))
        best["thr"] = pick(max(ok, key=lambda i: (rows[i]["thr"], -rows[i]["J_per_unit"])))
    base = next((i for i, x in enumerate(rows) if x["base"]), None)
    return {"chip": ch.name, "model": scn.model, "phase": "mix" if mix else phases[0], "rows": rows,
            "front": sorted(f2, key=lambda i: rows[i]["thr"]), "best": best, "base": base, "notes": notes,
            "power_cap_W": cap, "power_cap_src": cap_src, "power_mode": power_mode, "curve": curve,
            "die_mm2": area_lim, "tie": tie, "area": dens_ok, "n_feasible": len(ok), "mems": mem_ids, "slc_mibs": slcs, "f_scales": fs,
            "mix": {"prompt": P, "out_tokens": mix_out_tokens} if mix else None,
            "slo": {"tpot_ms": slo_tpot_ms, "ttft_ms": slo_ttft_ms}, "priced": bool(priced),
            "basis": "设计网格（MAC 总数 × 引擎边长 × SRAM × NoC 拓扑 × 外存 × SLC × 频率工作点）逐点评估：时间 / 放得下来自代价模型，"
                     "J/单位 来自能耗表（MAC / SRAM / 导线 / NoC / DRAM+PHY / 静态）；perf/W = 1 / (J/单位)；超功耗点按 DVFS "
                     "封顶运行（或剔除），超面积 / HBM 岸线 / SLO / 内存 $ 的点剔除；前沿 = 吞吐↑ 与 J/单位↓ 的非支配集，"
                     "差 ≤ tie（默认 1 %）视为持平（MODEL §245–247）"}
