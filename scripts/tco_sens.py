#!/usr/bin/env python3
"""0.318 TCO sensitivity (MODEL §263) on the chiplet example (scripts/chiplet_example.py, qwen3-8b on the 1P chip).

    python3 scripts/tco_sens.py                       # tornado: cheapest $/Mtok design of CASES[0] (decode, tp)
    python3 scripts/tco_sens.py --case 1 --tco server_2026
    python3 scripts/tco_sens.py --study               # do the 0.317 $/M verdicts flip inside the sourced ranges?
                                                      # (CASES mono/split, SWEEP mono/split, 4 / 6 / 8-stack choice;
                                                      # 0.319: and at the joint adverse / favourable corners)
    python3 scripts/tco_sens.py --asm-compare         # 0.319: verdicts under the 0.318 flat assembly yield → component
    python3 scripts/tco_sens.py --study --asm flat    # 0.318 basis

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
from accel_dse.core.tco_sens import (INPUTS, applicable, base_values, compare_flips, corner,  # noqa: E402
                                     input_range, row_usd, scan_flip, tornado)

SIDES = ("adverse", "favourable")


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
    c = t.get("corners") or {}
    if c:
        out(f"  joint corners: adverse {100 * c['adverse']['d']:+.1f}% · favourable {100 * c['favourable']['d']:+.1f}%")


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
        if k in skip or not applicable(k, base):
            continue
        rg = input_range(k, mem_type, base)
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
                + ("; ".join(_thr(s) for s in f["flips"]) if f["flips"] else f"no flip in {f['n_inputs']} inputs")
                + f" | corners adverse {f['corners']['adverse']}, favourable {f['corners']['favourable']}")
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
            cor = {sd: who(corner(base, sd, skip=("d0",))) for sd in SIDES}
            res["sweep"].append({"mem": mem, "tp": tp, "link": lk, "d0": d0, "mac_x": mx, "cheaper_tco": who({}),
                                 "flips": fl, "corners": cor})
            if fl or any(v != who({}) for v in cor.values()):
                out(f"  sweep {mem.split('_')[1]} tp{tp}/{lk:.0f} D0 {d0:.2f} ×{mx:g}: {who({})} | " + "; ".join(_thr(s) for s in fl)
                    + f" | corners {cor['adverse']} / {cor['favourable']}")
    for k, d in sorted(by.items()):
        if len(d) < 2:
            continue

        def pick(ov, d=d):
            return min(d, key=lambda m: min(row_usd(x, d[m][1], ov) for x in d[m][0]))
        mt = next(iter(d.values()))[0][0]["mem_type"]
        b0 = next(iter(d.values()))[1]
        scans = _scan_all(pick, mt, b0, skip=("d0",))
        fl = [s for s in scans if s["flips"]]
        cor = {sd: pick(corner(b0, sd, skip=("d0",))) for sd in SIDES}
        res["mem"].append({"tp": k[0], "link": k[1], "d0": k[2], "mac_x": k[3], "by_tco": pick({}), "options": sorted(d),
                           "flips": fl, "corners": cor})
        out(f"  mem choice tp{k[0]}/{k[1]:.0f} D0 {k[2]:.2f} ×{k[3]:g}: $/M→{pick({}).split('_')[1]} | "
            + ("; ".join(_thr(s) for s in fl) if fl else "no flip")
            + f" | corners {cor['adverse'].split('_')[1]} / {cor['favourable'].split('_')[1]}")
    n_c = sum(1 for r in res["cases"] if r["flips"])
    n_s = sum(1 for r in res["sweep"] if r["flips"])
    n_m = sum(1 for r in res["mem"] if r["flips"])
    out(f"  flips: cases {n_c}/{len(res['cases'])}, sweep {n_s}/{len(res['sweep'])}, memory choice {n_m}/{len(res['mem'])}")
    for sd in SIDES:     # 0.319: joint corners
        cc = sum(1 for r in res["cases"] if r["corners"][sd] != r["verdict_tco"])
        cs = sum(1 for r in res["sweep"] if r["corners"][sd] != r["cheaper_tco"])
        cm = sum(1 for r in res["mem"] if r["corners"][sd] != r["by_tco"])
        res.setdefault("corner_flips", {})[sd] = {"cases": cc, "sweep": cs, "mem": cm}
        out(f"  joint corner {sd}: cases {cc}/{len(res['cases'])}, sweep {cs}/{len(res['sweep'])}, memory choice {cm}/{len(res['mem'])}")
    return res


def asm_compare(out=print, tco="accel_2026", preset="n5_cowos_l") -> dict:
    """0.319: the 0.317 / 0.318 verdicts under the flat (0.318) and the component assembly-yield model, old → new:
    CASES $/M mono/split, SWEEP cheaper by $/good package and by $/Mtok, memory choice by capex and by $/Mtok."""
    r = {}
    for m in ("flat", "component"):
        r[m] = C.tco_study(out=lambda *_: None, tco=tco, cost={"preset": preset, "asm_model": m})
    ch = {"cases": [], "sweep": [], "mem": []}
    for a, b in zip(r["flat"]["cases"], r["component"]["cases"]):
        if a["verdict_tco"] != b["verdict_tco"]:
            ch["cases"].append({**{k: b[k] for k in ("case", "mac_x")}, "old": a["verdict_tco"], "new": b["verdict_tco"]})
    for a, b in zip(r["flat"]["sweep"], r["component"]["sweep"]):
        for k in ("cheaper_pkg", "cheaper_tco"):
            if a[k] != b[k]:
                ch["sweep"].append({**{x: b[x] for x in ("mem", "tp", "link", "d0", "mac_x")}, "verdict": k, "old": a[k], "new": b[k]})
    for a, b in zip(r["flat"]["mem"], r["component"]["mem"]):
        for k in ("by_thr", "by_capex", "by_tco"):
            if a[k] != b[k]:
                ch["mem"].append({**{x: b[x] for x in ("tp", "link", "d0", "mac_x")}, "verdict": k, "old": a[k], "new": b[k]})
    out(f"== asm_compare {preset} {tco}: flat (0.318) → component (0.319)")
    for c in ch["cases"]:
        out(f"  case {c['case']} MAC×{c['mac_x']:g}: $/M {c['old']} → {c['new']}")
    for c in ch["sweep"]:
        out(f"  sweep {c['mem'].split('_')[1]} tp{c['tp']}/{c['link']:.0f} D0 {c['d0']:.2f} ×{c['mac_x']:g}: {c['verdict']} {c['old']} → {c['new']}")
    for c in ch["mem"]:
        out(f"  mem tp{c['tp']}/{c['link']:.0f} D0 {c['d0']:.2f} ×{c['mac_x']:g}: {c['verdict']} {c['old'].split('_')[1]} → {c['new'].split('_')[1]}")
    out(f"  changed: cases {len(ch['cases'])}/{len(r['component']['cases'])}, sweep {len(ch['sweep'])} "
        f"(of {len(r['component']['sweep'])} cells × 2), memory {len(ch['mem'])} (of {len(r['component']['mem'])} cells × 3)")
    return {"changes": ch, "runs": r}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--cost", choices=("n5_cowos_s", "n5_cowos_l"), default="n5_cowos_l")
    ap.add_argument("--tco", choices=("accel_2026", "server_2026"), default="accel_2026")
    ap.add_argument("--d0", type=float, default=0.1)
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--asm", choices=("component", "flat"), default="component",
                    help="0.319 assembly-yield model (flat = 0.318: asm_yield 1, no scrap)")
    ap.add_argument("--asm-compare", action="store_true", help="0.319: verdicts flat (0.318) → component, old → new")
    a = ap.parse_args(argv)
    cost = {"preset": a.cost, "asm_model": a.asm}
    if a.asm_compare:
        asm_compare(tco=a.tco, preset=a.cost)
    elif a.study:
        study(cost=cost, tco=a.tco)
    else:
        case_tornado(a.case, cost=cost, tco=a.tco, d0=a.d0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
