#!/usr/bin/env python3
"""Chiplet-partitioning example of MODEL §253.2 / §254.2 (0.308 / 0.309), self-contained.

Energy-aware DSE on the 1P chip (TPUv4i 7 nm energy table, n5_derived density, 700 W, 8 × HBM3E 8-high 24 GB,
CoWoS-L 4719 mm², D0 0.1 /cm²) over MAC ×{1, 2}, 1 / 2 / 4 dies × {UCIe-A 48G, UCIe-S 32G, BoW-256}.

    python3 scripts/chiplet_example.py            # 0.309 view: monolithic die also as 2 / 4 NoC clusters, HBM attach
                                                  # on every edge its stacks occupy; prints verdict and the 0.308 one
    python3 scripts/chiplet_example.py --flat     # 0.308 view (flat monolithic die, two memory columns)
    python3 scripts/chiplet_example.py --case 0   # one case (index into CASES)
    python3 scripts/chiplet_example.py --cost n5_cowos_l --d0 0.2   # 0.310: cost per good package (+ D0)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from accel_dse.core import d2d_catalog  # noqa: E402
from accel_dse.core.budget import DENSITY_PRESETS, Budget  # noqa: E402
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable  # noqa: E402
from accel_dse.core.energy_dse import energy_design_search  # noqa: E402
from accel_dse.core.hardware import CHIPS  # noqa: E402
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


def run(model, ph, split="tp", L=0, pj=None, flat=False, out=print, cost=None, d0=0.1) -> dict:
    std = d2d_catalog.D2D_STANDARDS["ucie-a-48"]
    old = std["pJ_bit"]
    if pj:
        std["pJ_bit"] = pj
    try:
        s = Scenario(model=model, chip=CHIPS["1P"], mem_id="hbm3e_8s_8h24g_9200",
                     serving=Serving(phase=ph, batch=64 if ph == "decode" else 1, prompt=2048, ctx=4096))
        kw = dict(noc_link_Bpc=L, noc_routing="xy") if L else {}
        o = energy_design_search(s, table(), budget(), mac_scales=(1.0, 2.0), sram_scales=(1.0,), sides=(128,),
                                 dies=(1, 2, 4), d2d_stds=("ucie-a-48", "ucie-s-32", "bow-256"), die_split=split,
                                 d0_per_cm2=d0, pkg_mm2="cowos-l", mono_clusters=not flat,
                                 mem_edges_shoreline=not flat, **({"cost": cost} if cost else {}), **kw)
    finally:
        std["pJ_bit"] = old
    out(f"== {model} {ph} {split} L={L} pj={pj} {'(0.308 flat)' if flat else '(0.309)'}")
    for x in o["rows"]:
        kind = x.get("d2d_std") or ("mono" if x.get("clusters", 1) == 1 else f"mono×{x['clusters']}簇")
        out(f'  MAC×{x["mac_x"]:g} N={x["dies"]} {kind:10s} thr={x.get("thr", 0):8.0f} '
            f'J={x.get("J_per_unit", 0) * 1e3:7.2f}mJ W={x.get("W_card", 0):5.0f} die={x.get("mm2", 0):5.0f} '
            f'Σ={x.get("si_mm2", 0):5.0f} kgd={x.get("kgd_mm2", 0):5.0f} memE={x.get("mem_edges", 2)} '
            f'd2d={x.get("d2d_GBps") or 0:6.0f}{"*" if x.get("d2d_noc_capped") else " "} '
            f'{"F" if x.get("front") else " "}{"K" if x.get("front_kgd") else " "}'
            + (f' ${x["usd_good_pkg"]:6.0f}{"C" if x.get("front_cost") else " "}' if x.get("usd_good_pkg") is not None else '')
            + f' {x.get("why", "")}')
    for e in o["chiplet"]["compare"]:
        pc = lambda v: None if v is None else round(v * 100, 1)  # noqa: E731
        out(f'  -> MAC×{e["mac_x"]:g} verdict={e["verdict"]} d_thr={pc(e.get("d_thr"))}% d_J={pc(e.get("d_J"))}% '
            f'best_thr={e.get("best_thr")} best_J={e.get("best_J")} mono_thr={e.get("mono_thr")} '
            f'mono_J={e.get("mono_J")} | flat-only verdict={e.get("verdict_flat")}')
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="0.308 view: no NoC-cluster rows, two memory columns")
    ap.add_argument("--case", type=int, action="append", help="case index (repeatable); default all")
    ap.add_argument("--cost", choices=("n5_cowos_s", "n5_cowos_l"), help="0.310: cost per good package")
    ap.add_argument("--d0", type=float, default=0.1, help="defect density /cm² (default 0.1)")
    a = ap.parse_args(argv)
    for i in (a.case or range(len(CASES))):
        run(*CASES[i], flat=a.flat, cost=a.cost, d0=a.d0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
