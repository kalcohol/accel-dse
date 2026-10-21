#!/usr/bin/env python3
"""0.318 TCO sensitivity (MODEL §263) on the chiplet example (scripts/chiplet_example.py, qwen3-8b on the 1P chip).

    python3 scripts/tco_sens.py                       # tornado: cheapest $/Mtok design of CASES[0] (decode, tp)
    python3 scripts/tco_sens.py --case 1 --tco server_2026
    python3 scripts/tco_sens.py --study               # do the 0.317 $/M verdicts flip inside the sourced ranges?
                                                      # (CASES mono/split, SWEEP mono/split, 4 / 6 / 8-stack choice)

One input at a time across its sourced range (core/tco_sens.RANGES, tco.MEM_RANGES); throughput and package power do
not move.  In the SWEEP / memory-choice cells D0 is a grid axis (0.05–0.3), so it is not scanned there."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import chiplet_example as C  # noqa: E402
from accel_dse.core.energy_dse import _cost_params, _tco_ok  # noqa: E402
from accel_dse.core.tco import tco_params  # noqa: E402
from accel_dse.core.tco_sens import (COST_INPUTS, INPUTS, LABELS, base_values, compare_flips, input_range,  # noqa: E402
                                     row_usd, scan_flip, tornado)


def _base(cost, tco, d0):
    return {"tco": tco_params(tco), "cost": _cost_params(cost), "d0": d0, "prices": None}


def _fmt(k, x):
    return f"{100 * x:.2f} %" if k == "capital_rate" else f"{x:.4g}"


def show_tornado(t, out=print):
    out(f"  base ${t['usd_per_Munit']:.5f} / M {t['unit']}")
    for b in t["bars"]:
        out(f"  {b['label']:12s} {_fmt(b['input'], b['lo']):>9s}「{b['tag_lo']}」→ {_fmt(b['input'], b['hi']):>9s}「{b['tag_hi']}」 "
            f"{100 * b['d_lo']:+6.1f}% / {100 * b['d_hi']:+6.1f}%")
    for s in t["skipped"]:
        out(f"  (skipped) {s}")


def case_tornado(ci=0, cost="n5_cowos_l", tco="accel_2026", d0=0.1, out=print) -> dict:
    o = C.run(*C.CASES[ci], out=lambda *_: None, cost=cost, d0=d0, tco=tco)
    b = o["tco"]["best"]
    if b is None:
        raise SystemExit("no buildable, priced row")
    r = o["rows"][b]
    t = tornado(r, _base(cost, tco, d0))
    out(f"== case {ci} {C.CASES[ci][:3]} cheapest $/M row {b}: {r['dies']} die(s) MAC×{r['mac_x']:g} {r['mem_id']}")
    show_tornado(t, out)
    return t


def _scan_all(verdict, mem_type, base, skip=()):
    bv = base_values(base, mem_type)
    res = []
    for k in INPUTS:
        if k in skip or (k in COST_INPUTS and base.get("cost") is None):
            continue
        rg = input_range(k, mem_type)
        if rg is not None:
            res.append(scan_flip(verdict, k, rg, bv.get(k), n=32, iters=40))
    return res


def _thr(s):
    sh = lambda v: v.split("_")[1] if v.startswith("hbm") else v      # noqa: E731
    return ", ".join(f"{s['label']} {_fmt(s['input'], t['at'])}（{sh(t['from'])}→{sh(t['to'])}）" for t in s["thresholds"])


def study(cost="n5_cowos_l", tco="accel_2026", out=print, cases=None, sweep_cells=None) -> dict:
    """Flip analysis of the 0.317 verdicts.  ``cases`` / ``sweep_cells`` restrict the work (tests)."""
    res = {"cases": [], "sweep": [], "mem": []}
    out(f"== tco_sens study tco={tco} cost={cost}")
    for ci in (range(len(C.CASES)) if cases is None else cases):
        o = C.run(*C.CASES[ci], out=lambda *_: None, cost=cost, tco=tco)
        base = _base(cost, tco, 0.1)
        for f in compare_flips(o["rows"], o["chiplet"]["compare"], base):
            r = {"case": ci, **f}
            res["cases"].append(r)
            out(f"  case {ci} MAC×{f['mac_x']:g}: $/M {f['verdict_tco']:5s} | "
                + ("; ".join(_thr(s) for s in f["flips"]) if f["flips"] else f"no flip in {f['n_inputs']} inputs"))
    cells = [(m, tp, lk, d0) for m in C.SWEEP_MEMS for tp, lk in C.SWEEP_DEPLOY for d0 in C.SWEEP_D0]
    if sweep_cells is not None:
        cells = [c for c in cells if (c[1], c[3]) in sweep_cells]
    by = {}
    for mem, tp, lk, d0 in cells:
        o = C.run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None, cost=cost, d0=d0, mem=mem, tp=tp, link=lk,
                  tco=tco)
        base = _base(cost, tco, d0)
        for mx in (1.0, 2.0):
            rs = [x for x in o["rows"] if x["mac_x"] == mx and _tco_ok(x)]
            if not rs:
                continue
            by.setdefault((tp, lk, d0, mx), {})[mem] = (rs, base)
            mono = [x for x in rs if x.get("dies", 1) == 1]
            sp = [x for x in rs if x.get("dies", 1) > 1]
            if not mono or not sp:
                continue

            def who(ov, mono=mono, sp=sp, base=base):
                return "mono" if min(row_usd(x, base, ov) for x in mono) <= min(row_usd(x, base, ov) for x in sp) else "split"
            scans = _scan_all(who, rs[0]["mem_type"], base, skip=("d0",))
            fl = [s for s in scans if s["flips"]]
            res["sweep"].append({"mem": mem, "tp": tp, "link": lk, "d0": d0, "mac_x": mx, "cheaper_tco": who({}),
                                 "flips": fl})
            if fl:
                out(f"  sweep {mem.split('_')[1]} tp{tp}/{lk:.0f} D0 {d0:.2f} ×{mx:g}: {who({})} | " + "; ".join(_thr(s) for s in fl))
    for k, d in sorted(by.items()):
        if len(d) < 2:
            continue

        def pick(ov, d=d):
            return min(d, key=lambda m: min(row_usd(x, d[m][1], ov) for x in d[m][0]))
        mt = next(iter(d.values()))[0][0]["mem_type"]
        scans = _scan_all(pick, mt, next(iter(d.values()))[1], skip=("d0",))
        fl = [s for s in scans if s["flips"]]
        res["mem"].append({"tp": k[0], "link": k[1], "d0": k[2], "mac_x": k[3], "by_tco": pick({}), "options": sorted(d),
                           "flips": fl})
        out(f"  mem choice tp{k[0]}/{k[1]:.0f} D0 {k[2]:.2f} ×{k[3]:g}: $/M→{pick({}).split('_')[1]} | "
            + ("; ".join(_thr(s) for s in fl) if fl else "no flip"))
    n_c = sum(1 for r in res["cases"] if r["flips"])
    n_s = sum(1 for r in res["sweep"] if r["flips"])
    n_m = sum(1 for r in res["mem"] if r["flips"])
    out(f"  flips: cases {n_c}/{len(res['cases'])}, sweep {n_s}/{len(res['sweep'])}, memory choice {n_m}/{len(res['mem'])}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--cost", choices=("n5_cowos_s", "n5_cowos_l"), default="n5_cowos_l")
    ap.add_argument("--tco", choices=("accel_2026", "server_2026"), default="accel_2026")
    ap.add_argument("--d0", type=float, default=0.1)
    ap.add_argument("--study", action="store_true")
    a = ap.parse_args(argv)
    if a.study:
        study(cost=a.cost, tco=a.tco)
    else:
        case_tornado(a.case, cost=a.cost, tco=a.tco, d0=a.d0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
