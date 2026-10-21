#!/usr/bin/env python3
"""Chiplet-partitioning example of MODEL §253.2 / §254.2 (0.308 / 0.309), self-contained.

Energy-aware DSE on the 1P chip (TPUv4i 7 nm energy table, n5_derived density, 700 W, 8 × HBM3E 8-high 24 GB,
CoWoS-L 4719 mm², D0 0.1 /cm²) over MAC ×{1, 2}, 1 / 2 / 4 dies × {UCIe-A 48G, UCIe-S 32G, BoW-256}.

    python3 scripts/chiplet_example.py            # 0.309 view: monolithic die also as 2 / 4 NoC clusters, HBM attach
                                                  # on every edge its stacks occupy; prints verdict and the 0.308 one
    python3 scripts/chiplet_example.py --flat     # 0.308 view (flat monolithic die, two memory columns)
    python3 scripts/chiplet_example.py --case 0   # one case (index into CASES)
    python3 scripts/chiplet_example.py --cost n5_cowos_l --d0 0.2   # 0.310: cost per good package (+ D0)
    python3 scripts/chiplet_example.py --io-edges 0   # 0.310 rule: no die edge kept for host I/O
    python3 scripts/chiplet_example.py --io-edges 1   # 0.311 rule: one whole die edge kept for host I/O
                                                      # (0.312 default -1: PCIe + scale-up lanes × 0.285 mm sized)
    python3 scripts/chiplet_example.py --mem hbm3e_6s_8h24g_9200 --tp 8 --link 900   # 0.312: other memory; a
                                                      # replica over 8 packages (scale-up SerDes on every die)
    python3 scripts/chiplet_example.py --sweep        # 0.312: buildable-only verdicts over memory × D0 × deployment
    python3 scripts/chiplet_example.py --sweep --lane 100   # 0.313: the 0.312 scale-up lane rate (default 200G)
    python3 scripts/chiplet_example.py --sweep --strict     # 0.313: memory on > 2 edges counts as unbuildable
    python3 scripts/chiplet_example.py --case 0 --cost n5_cowos_l --cooling air_oam   # 0.313: thermal judged
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from accel_dse.core import d2d_catalog  # noqa: E402
from accel_dse.core.budget import DENSITY_PRESETS, Budget  # noqa: E402
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable  # noqa: E402
from accel_dse.core.energy_dse import energy_design_search  # noqa: E402
from accel_dse.core.hardware import CHIPS, Link  # noqa: E402
from accel_dse.core.parallel import Layout  # noqa: E402
from accel_dse.core.scenario import Scenario, Serving  # noqa: E402

# (model, phase, die_split, NoC link width B/cycle (0 = unset), UCIe-A 48G pJ/bit override or None)
CASES = [
    ("qwen3-8b", "decode", "tp", 0, None),
    ("qwen3-8b", "prefill", "tp", 0, None),
    ("qwen3-8b", "decode", "tp", 32, None),
    ("qwen3-8b", "prefill", "tp", 64, None),
    ("qwen3-30b-a3b", "decode", "ep", 0, None),
    ("qwen3-30b-a3b", "decode", "tp", 0, None),
    ("qwen3-8b", "prefill", "tp", 0, 0.75),
]


def table() -> EnergyTable:
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    return EnergyTable.from_dict({k: v for k, (v, _, _) in f.items() if k in EnergyTable.__dataclass_fields__
                                  and k not in ("idle_W", "idle_frac_tdp", "tdp_W")})


def budget() -> Budget:
    f = DENSITY_PRESETS["n5_derived"]["fields"]
    return Budget(**{k: v for k, (v, _, _) in f.items() if k in Budget.__dataclass_fields__}, power_W_card=700)


def run(model, ph, split="tp", L=0, pj=None, flat=False, out=print, cost=None, d0=0.1, io_edges=-1,
        mem="hbm3e_8s_8h24g_9200", tp=1, link=400.0, macs=(1.0, 2.0), include_unbuildable=False, lane=None,
        shore_strict=False, cooling=None) -> dict:
    std = d2d_catalog.D2D_STANDARDS["ucie-a-48"]
    old = std["pJ_bit"]
    if pj:
        std["pJ_bit"] = pj
    try:
        chip = CHIPS["1P"] if io_edges == -1 else replace(CHIPS["1P"], noc_mem_io_edges=io_edges)
        if lane:                    # 0.313: scale-up lane rate (Gb/s per pair; default the chip's 200)
            chip = replace(chip, io_su_lane_Gbps=float(lane))
        s = Scenario(model=model, chip=chip, mem_id=mem, layout=Layout(tp=tp), link=Link(GBps=link),
                     serving=Serving(phase=ph, batch=64 if ph == "decode" else 1, prompt=2048, ctx=4096))
        kw = dict(noc_link_Bpc=L, noc_routing="xy") if L else {}
        o = energy_design_search(s, table(), budget(), mac_scales=macs, sram_scales=(1.0,), sides=(128,),
                                 dies=(1, 2, 4), d2d_stds=("ucie-a-48", "ucie-s-32", "bow-256"), die_split=split,
                                 d0_per_cm2=d0, pkg_mm2="cowos-l", mono_clusters=not flat,
                                 mem_edges_shoreline=not flat, include_unbuildable=include_unbuildable,
                                 shore_strict=shore_strict, cooling=cooling,
                                 **({"cost": cost} if cost else {}), **kw)
    finally:
        std["pJ_bit"] = old
    out(f"== {model} {ph} {split} L={L} pj={pj} {'(0.308 flat)' if flat else '(0.309)'} io_edges={io_edges} "
        f"mem={mem} tp={tp} link={link:g}")
    for x in o["rows"]:
        kind = x.get("d2d_std") or ("mono" if x.get("clusters", 1) == 1 else f"mono×{x['clusters']}簇")
        out(f'  MAC×{x["mac_x"]:g} N={x["dies"]} {kind:10s} thr={x.get("thr", 0):8.0f} '
            f'J={x.get("J_per_unit", 0) * 1e3:7.2f}mJ W={x.get("W_card", 0):5.0f} die={x.get("mm2", 0):5.0f} '
            f'Σ={x.get("si_mm2", 0):5.0f} kgd={x.get("kgd_mm2", 0):5.0f} memE={x.get("mem_edges", 2)} '
            f'd2d={x.get("d2d_GBps") or 0:6.0f}{"*" if x.get("d2d_noc_capped") else " "} '
            f'{"F" if x.get("front") else " "}{"K" if x.get("front_kgd") else " "}'
            + (f' ${x["usd_good_pkg"]:6.0f}{"C" if x.get("front_cost") else " "}' if x.get("usd_good_pkg") is not None else '')
            + f' {"B" if x.get("buildable") else ("b" if x.get("buildable") is False else "?")}'
            + f' {x.get("why", "")}{x.get("why_shore", "")}')
    for e in o["chiplet"]["compare"]:
        pc = lambda v: None if v is None else round(v * 100, 1)  # noqa: E731
        out(f'  -> MAC×{e["mac_x"]:g} verdict={e["verdict"]} d_thr={pc(e.get("d_thr"))}% d_J={pc(e.get("d_J"))}% '
            f'best_thr={e.get("best_thr")} best_J={e.get("best_J")} mono_thr={e.get("mono_thr")} '
            f'mono_J={e.get("mono_J")} | flat-only verdict={e.get("verdict_flat")}')
    return o


SWEEP_MEMS = ("hbm3e_4s_8h24g_9200", "hbm3e_6s_8h24g_9200", "hbm3e_8s_8h24g_9200")
SWEEP_D0 = (0.05, 0.1, 0.2, 0.3)
SWEEP_DEPLOY = ((1, 400.0), (8, 400.0), (8, 900.0))     # (packages per replica, scale-up GB/s per direction)


def sweep(out=print, io_edges=-1, cost="n5_cowos_l", lane=None, shore_strict=False) -> list:
    """0.312: buildable-only economics — qwen3-8b decode, MAC ×1 / ×2: for each memory × deployment × D0 the cheapest
    feasible (buildable) monolithic die (flat or NoC-clustered) against the cheapest feasible split, $ per good
    package (n5_cowos_l).  "mono buildable" = some dies = 1 row passes the shoreline report."""
    res = []
    out(f"== sweep io_edges={io_edges} cost={cost} lane={lane or 'chip'} strict={shore_strict}: mem tp/link D0 MAC× | mono buildable · mono $ | best split $ (N std) "
        "| cheaper")
    for mem in SWEEP_MEMS:
        for tp, lk in SWEEP_DEPLOY:
            for d0 in SWEEP_D0:
                o = run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None, cost=cost, d0=d0, io_edges=io_edges,
                        mem=mem, tp=tp, link=lk, lane=lane, shore_strict=shore_strict)
                for mx in (1.0, 2.0):
                    rs = [x for x in o["rows"] if x["mac_x"] == mx]
                    mono = [x for x in rs if x.get("dies", 1) == 1]
                    mb = any(x.get("buildable") for x in mono)
                    mf = [x for x in mono if x.get("feasible")]
                    sp = [x for x in rs if x.get("dies", 1) > 1 and x.get("feasible")]
                    m_ = min(mf, key=lambda x: x["usd_good_pkg"]) if mf else None
                    s_ = min(sp, key=lambda x: x["usd_good_pkg"]) if sp else None
                    who = ("mono" if m_ and (not s_ or m_["usd_good_pkg"] <= s_["usd_good_pkg"]) else
                           ("split" if s_ else "none"))
                    r = {"mem": mem, "tp": tp, "link": lk, "d0": d0, "mac_x": mx, "mono_buildable": mb,
                         "mono_feasible": bool(mf), "mono_usd": m_ and m_["usd_good_pkg"],
                         "mono_why": None if mf else (mono[0].get("why") if mono else None),
                         "split_usd": s_ and s_["usd_good_pkg"], "split": s_ and (s_["dies"], s_["d2d_std"]),
                         "cheaper": who}
                    res.append(r)
                    out(f"  {mem.split('_')[1]:>3s} tp{tp}/{lk:4.0f} D0 {d0:4.2f} ×{mx:g} | "
                        f"{'Y' if mb else 'N'} · {('$%5.0f' % m_['usd_good_pkg']) if m_ else '   —  '} | "
                        f"{('$%5.0f (%d %s)' % (s_['usd_good_pkg'], s_['dies'], s_['d2d_std'])) if s_ else '—':22s} | {who}"
                        + ("" if mf else f"  [{(r['mono_why'] or '')[:60]}]"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="0.308 view: no NoC-cluster rows, two memory columns")
    ap.add_argument("--case", type=int, action="append", help="case index (repeatable); default all")
    ap.add_argument("--cost", choices=("n5_cowos_s", "n5_cowos_l"), help="0.310: cost per good package")
    ap.add_argument("--d0", type=float, default=0.1, help="defect density /cm² (default 0.1)")
    ap.add_argument("--io-edges", type=int, choices=(-1, 0, 1), default=-1,
                    help="host I/O shoreline: -1 = sized from the PCIe / scale-up lanes (0.312 default); 1 = one whole "
                         "edge (0.311); 0 = none (0.310)")
    ap.add_argument("--mem", default="hbm3e_8s_8h24g_9200", help="memory id (default 8 × HBM3E 8-high 24 GB)")
    ap.add_argument("--tp", type=int, default=1, help="packages per replica (TP); > 1 puts scale-up SerDes on each die")
    ap.add_argument("--link", type=float, default=400.0, help="scale-up GB/s per direction per card (default 400)")
    ap.add_argument("--include-unbuildable", action="store_true", help="0.312: keep unbuildable rows on the fronts")
    ap.add_argument("--lane", type=float, choices=(50.0, 100.0, 200.0),
                    help="0.313: scale-up lane Gb/s (default the chip's: 200; 100 = the 0.312 rule)")
    ap.add_argument("--strict", action="store_true", help="0.313: memory on > 2 edges counts as unbuildable")
    ap.add_argument("--cooling", choices=("air_oam", "air_hgx", "liquid"), help="0.313: judge thermal buildability")
    ap.add_argument("--sweep", action="store_true", help="0.312: buildable-only verdict table (memory × D0 × deployment)")
    a = ap.parse_args(argv)
    if a.sweep:
        sweep(io_edges=a.io_edges, cost=a.cost or "n5_cowos_l", lane=a.lane, shore_strict=a.strict)
        return 0
    for i in (a.case or range(len(CASES))):
        run(*CASES[i], flat=a.flat, cost=a.cost, d0=a.d0, io_edges=a.io_edges, mem=a.mem, tp=a.tp, link=a.link,
            include_unbuildable=a.include_unbuildable, lane=a.lane, shore_strict=a.strict, cooling=a.cooling)
    return 0


if __name__ == "__main__":
    sys.exit(main())
