#!/usr/bin/env python3
"""0.66 held-out calibration study (resumable).  Grid over DRAM efficiency × MAC efficiency × overlap mode; one JSON
line per grid point with every row's predicted / measured ratio.  Usage: ext_calib.py OUT.jsonl"""
import json
import sys
from pathlib import Path

from accel_dse.core import extval as X

out = Path(sys.argv[1])
done = set()
if out.exists():
    for ln in out.read_text().splitlines():
        try:
            done.add(json.loads(ln)["key"])
        except Exception:
            pass
rows = X.rows()
MEM = [round(0.5 + 0.05 * i, 2) for i in range(11)]
MAC = [round(0.4 + 0.1 * i, 2) for i in range(7)]
for ov in ("stage", "class", "kernel", "serial"):
    for me in MEM:
        for ma in MAC:
            key = f"{ov}:{me}:{ma}"
            if key in done:
                continue
            name = X.calib_variant(me, ma, ov)
            res = []
            for r in rows:
                p = X.predict(r, name)
                res.append([r["id"], p["pred"] / r["value"]])
            with out.open("a") as f:
                f.write(json.dumps({"key": key, "ov": ov, "mem": me, "mac": ma, "ratios": res}) + "\n")
            print(key, flush=True)
