"""0.227: power-capped operation by clock / voltage scaling (opt-in; the cap value is the user's TDP decision, §73).

Mechanism: card power P(f) = E_core(V)/t(f) + E_other/t(f) + P_static.  Lowering the core clock f stretches only the
clocked work (MAC / vector / SFU / SRAM-port time; DRAM, links and host keep their own clocks), so the step is
re-evaluated at chip.freq_ghz · s.  Core-rail action energies (mac, vec, sram, slc) scale with V(f)² (P ∝ C·V²·f,
「来源」 arXiv 2607.00819 eq. 1); DRAM array / PHY, D2D / link / network bits are held.  Static power (0.260): the
core-rail leakage term (SRAM macro leakage, `uW_per_mib_sram_leak`) scales linearly with V — P_static = n·V
(Kandiah et al., 'AccelWattch', MICRO 2021, §4.2 eq. 2: chip and memory static b·V + b′·V, 「来源」); linear is the
conservative 「近似」 (sub-threshold / DIBL leakage falls faster than V).  The idle floor (measured at the idle P-state,
i.e. already at low V) and HBM standby / refresh (memory rail, its clock is held) stay held 「假设」.  Curves (V/V_max against f/f_max):
  dfs          clock only at nominal voltage — energy per action unchanged.  Without a per-cycle term the capped
               throughput equals the 0.189 duty-cycle model (t = E_dyn / (cap − P_static)); 0.228: with pJ_clk_mac_cycle
               the clocked power macs·f·pJ falls with f even when the op rate is memory-set, so dfs ≥ duty cycling
               (duty cycling gates the clock off-period: clock energy per unit unchanged).  Headline (no voltage benefit).
  rx5700xt_n7  AMD RX 5700 XT (TSMC N7) default OD_VDDC_CURVE 800 MHz 704 mV, 1409 MHz 819 mV, 2019 MHz 1185 mV
               「二手」 (amdgpu pp_od_clk_voltage dump, sibradzic/amdgpu-clocks issue #39); flat below the first point.
               Cross-check (not used): H100 power turns superlinear at f_t ≈ 0.51 f_max (2607.00819, FIRESTARTER).
Lowest clock F_MIN = 0.15 f_max (A100 0.21 / 1.41 GHz, H100 0.345 / 1.98 GHz settable range, 2607.00819 Table 2
「来源」).  Below F_MIN the remaining excess is removed by duty cycling (0.189)."""
from __future__ import annotations

from dataclasses import replace

VF_CURVES = {
    "dfs": {"points": ((0.0, 1.0), (1.0, 1.0)), "tag": "推导",
            "label": "仅降频（电压不变，每动作能耗不变 ≡ 0.189 占空比吞吐）"},
    "rx5700xt_n7": {"points": ((800 / 2019, 704 / 1185), (1409 / 2019, 819 / 1185), (1.0, 1.0)), "tag": "二手",
                    "label": "AMD RX 5700 XT（TSMC N7）默认 V–f 曲线：800 MHz 704 mV / 1409 MHz 819 mV / 2019 MHz 1185 mV"},
}
CAP_CURVES = ("dfs", "rx5700xt_n7")
F_MIN = 0.15
CORE_ACTIONS = {"mac": "pJ_mac", "vec": "pJ_vec", "sram": "pJ_bit_sram", "slc": "pJ_bit_slc",
                "clock": "pJ_clk_mac_cycle"}      # 0.228: per-cycle term on the core rail (V²; f via the re-evaluation)


def volt(curve: str, s: float) -> float:
    """V / V_max at f / f_max = s (piecewise linear through the curve points; flat below the first)."""
    pts = VF_CURVES[curve]["points"]
    if s <= pts[0][0]:
        return pts[0][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if s <= f1:
            return v0 + (v1 - v0) * (s - f0) / (f1 - f0)
    return pts[-1][1]


def static_W(e: dict) -> float:
    """Static W per card of an energy report: idle + SRAM leakage + HBM standby / refresh."""
    return (e.get("idle_W_used") or 0.0) + (e.get("sram_leak_W_card") or 0.0) + (e.get("hbm_static_W_card") or 0.0)


def _units_s_card(e: dict, cards: int) -> float:
    return e["units_per_window"] / e["window_s"] / cards if e.get("window_s") else 0.0


def _point(r, table, curve: str, s: float):
    from .energy import energy_report
    from .evaluate import evaluate
    scn = r.scenario
    q = r if s == 1.0 else evaluate(scn.replace("chip", replace(scn.chip, freq_ghz=scn.chip.freq_ghz * s)))
    v = volt(curve, s)
    t = replace(table, **{f: getattr(table, f) * v * v for f in CORE_ACTIONS.values() if getattr(table, f) is not None})
    if t.uW_per_mib_sram_leak is not None and v != 1.0:     # 0.260: core-rail leakage ∝ V (AccelWattch eq. 2)
        t = replace(t, uW_per_mib_sram_leak=t.uW_per_mib_sram_leak * v)
    return q, energy_report(q, t)


def dvfs_cap(r, table, cap_W: float, curve: str = "dfs", iters: int = 22) -> dict:
    """Highest clock (fraction s of the nominal) whose card power fits cap_W; then the throughput / energy there."""
    from .energy import energy_report
    cards = r.scenario.layout.cards
    e1 = energy_report(r, table)
    if e1.get("avg_W_per_card") is None:
        return {"error": "功耗封顶需要能耗表（J/单位）"}
    u1 = _units_s_card(e1, cards)
    P1 = e1["avg_W_per_card"]
    base = {"curve": curve, "tag": VF_CURVES[curve]["tag"], "cap_W": cap_W, "P_nominal_W": P1, "f_nominal_ghz": r.scenario.chip.freq_ghz}
    if P1 <= cap_W * (1 + 1e-12):
        return {**base, "binds": False, "s": 1.0, "f_ghz": r.scenario.chip.freq_ghz, "v": 1.0, "throughput_factor": 1.0,
                "W_card": P1, "J_per_unit": e1["J_per_unit"], "bound": r.bound, "duty": 1.0}
    lo_q, lo_e = _point(r, table, curve, F_MIN)
    if lo_e["avg_W_per_card"] > cap_W:
        ps = static_W(lo_e)
        if ps >= cap_W:
            return {**base, "binds": True, "feasible": False, "error": "静态功耗 ≥ 封顶：任何吞吐都超功耗"}
        slow = (lo_e["avg_W_per_card"] - ps) / (cap_W - ps)
        J = lo_e["J_per_unit"] + ps * cards / _units_s_card(lo_e, cards) / cards * (slow - 1.0)
        return {**base, "binds": True, "feasible": True, "s": F_MIN, "f_ghz": lo_q.scenario.chip.freq_ghz,
                "v": volt(curve, F_MIN), "duty": 1.0 / slow, "throughput_factor": _units_s_card(lo_e, cards) / slow / u1,
                "W_card": cap_W, "J_per_unit": J, "bound": lo_q.bound}
    lo, hi = F_MIN, 1.0
    best = (lo_q, lo_e, lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        q, e = _point(r, table, curve, mid)
        if e["avg_W_per_card"] <= cap_W:
            lo, best = mid, (q, e, mid)
        else:
            hi = mid
    q, e, s = best
    return {**base, "binds": True, "feasible": True, "s": s, "f_ghz": q.scenario.chip.freq_ghz, "v": volt(curve, s),
            "duty": 1.0, "throughput_factor": _units_s_card(e, cards) / u1, "W_card": e["avg_W_per_card"],
            "J_per_unit": e["J_per_unit"], "bound": q.bound, "tpot_ms": q.tpot * 1e3 if q.tpot else None}
