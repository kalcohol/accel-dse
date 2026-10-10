#!/usr/bin/env python3
"""Summarise instance_sched_diff.py output: per chip/point/metric n, min/median/max rel change; bottleneck flips; CSV."""
import csv, json, statistics, sys
from collections import Counter, defaultdict
rows = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, ["chip", "model", "point", "layout", "metric", "wide", "auto", "rel", "bound_wide", "bound_auto"])
    w.writeheader()
    for r in rows:
        w.writerow({**{k: r[k] for k in ("chip", "model", "point", "layout", "metric", "wide", "auto", "rel")},
                    "bound_wide": r["bound"][0], "bound_auto": r["bound"][1]})
flips = Counter()
seen = set()
for r in rows:
    key = (r["chip"], r["model"], r["point"], r["layout"])
    if r["bound"][0] != r["bound"][1] and key not in seen:
        seen.add(key); flips[f'{r["bound"][0]}→{r["bound"][1]}'] += 1
g = defaultdict(list)
for r in rows:
    g[(r["chip"], r["point"], r["metric"])].append(r["rel"])
print(f'changed numbers: {len(rows)} (models {len({r["model"] for r in rows})}); bottleneck flips: {sum(flips.values())} — '
      + ", ".join(f"{k} {v}" for k, v in flips.most_common()))
print("\n| chip | point | metric | n | min | median | max |\n|---|---|---|---|---|---|---|")
for (c, p, m), v in sorted(g.items()):
    print(f"| {c} | {p} | {m} | {len(v)} | {min(v):+.1%} | {statistics.median(v):+.1%} | {max(v):+.1%} |")
