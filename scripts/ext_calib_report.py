#!/usr/bin/env python3
"""Analyse ext_calib.py output: per split and overlap mode, fit (mem_eff, mac_eff) on the fit rows, report fit and
held-out mean |log error| (as %), next to the un-tuned baselines (catalog 0.7 / 1.0 and peak 1.0 / 1.0)."""
import json
import math
import sys

from accel_dse.core import extval as X

rows = {r["id"]: r for r in X.rows()}
grid = [json.loads(l) for l in open(sys.argv[1])]


def err(g, pred):
    lr = [abs(math.log(v)) for i, v in g["ratios"] if pred(rows[i]) and v == v and v > 0]
    return 100 * (math.exp(sum(lr) / len(lr)) - 1), len(lr)


def gm(g, pred):
    lr = [math.log(v) for i, v in g["ratios"] if pred(rows[i]) and v == v and v > 0]
    return math.exp(sum(lr) / len(lr))


def at(ov, me, ma):
    return next(g for g in grid if g["ov"] == ov and abs(g["mem"] - me) < 1e-9 and abs(g["mac"] - ma) < 1e-9)


out = {}
for sp, (fit, test) in X.SPLITS.items():
    for ov in ("stage", "class", "kernel", "serial"):
        cand = [g for g in grid if g["ov"] == ov]
        best = min(cand, key=lambda g: err(g, fit)[0])
        base_c, base_p = at(ov, 0.7, 1.0), at(ov, 1.0, 1.0)
        out[f"{sp}/{ov}"] = dict(
            fit_mem=best["mem"], fit_mac=best["mac"], n_fit=err(best, fit)[1], n_test=err(best, test)[1],
            fit_err=err(best, fit)[0], test_err=err(best, test)[0], test_gmean=gm(best, test),
            untuned_catalog_test_err=err(base_c, test)[0], untuned_peak_test_err=err(base_p, test)[0])
for k, v in out.items():
    print(f"{k:14s} fit mem {v['fit_mem']:.2f} mac {v['fit_mac']:.2f} | fit err {v['fit_err']:5.1f}% (n{v['n_fit']}) "
          f"| held-out err {v['test_err']:5.1f}% gmean {v['test_gmean']:.2f} (n{v['n_test']}) | untuned on held-out: "
          f"catalog {v['untuned_catalog_test_err']:5.1f}%  peak {v['untuned_peak_test_err']:5.1f}%")
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
