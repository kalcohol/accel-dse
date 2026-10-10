#!/usr/bin/env python3
"""0.71: every headline number that changes if Chip.instance_sched defaulted to "auto" (catalog models, default chip,
the fingerprint serving points, 1 card / TP8 / DP8·EP8, and the three catalog chips)."""
import json, sys
from dataclasses import replace
from accel_dse.core.catalog import entries
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

PTS = {"decode b1": Serving(), "prefill b2 8k": Serving(phase="prefill", batch=2, prompt=8192),
       "decode b16 ctx32k": Serving(batch=16, ctx=32768)}
LAYS = {"1 card": {}, "TP8": {"layout": Layout(tp=8)}, "DP8·EP8": {"layout": Layout(dp=8, ep=8)}}
rows = []
chips = sys.argv[2].split(",") if len(sys.argv) > 2 else [None]
for cn in chips:
    for e in entries():
        for pn, sv in PTS.items():
            for ln, lay in LAYS.items():
                kw = dict(model=e["id"], serving=sv, **lay)
                if cn:
                    kw["chip"] = CHIPS[cn]
                try:
                    s = Scenario(**kw)
                    a = evaluate(s); b = evaluate(replace(s, chip=replace(s.chip, instance_sched="auto")))
                except Exception as ex:          # invalid layout for this model (e.g. EP on dense)
                    continue
                for k in ("tpot", "ttft", "throughput"):
                    x, y = getattr(a, k), getattr(b, k)
                    if x and y and x == x and y == y and x != y:
                        rows.append(dict(chip=s.chip.name, model=e["id"], point=pn, layout=ln, metric=k, wide=x, auto=y,
                                         rel=y / x - 1, bound=(a.bound, b.bound)))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(len(rows), "changed numbers")
