#!/usr/bin/env python3
"""0.66 V6: run accel-dse on every published measurement (data/ext_measurements.json) under the reported variants
and write data/ext_validation.json (per-row predictions + per-metric summary).  Optional: --calib REPORT.json (from
ext_calib_report.py) is embedded as the held-out calibration study."""
import argparse
import json
from pathlib import Path

from accel_dse import __version__
from accel_dse.core import extval as X

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(X.DATA).with_name("ext_validation.json")))
ap.add_argument("--calib", default=None)
a = ap.parse_args()
REPORT = ("default", "catalog", "peak", "catalog_class", "catalog_kernel", "catalog_kernel_fixed", "catalog_serial", "peak_serial_m", "peak_noreuse")
res = {"generated": __version__, "variants": {k: X.VARIANTS[k] for k in REPORT}, "summary": {}, "rows": {}}
for v in REPORT:
    t = X.table(v)
    res["summary"][v] = X.summary(t)
    res["rows"][v] = [{"id": x["id"], "pred": round(x["pred"], 4), "ratio": round(x["ratio"], 4), "fits": x["fits"],
                       **({"batch": x["batch"]} if x["metric"] == "max_tok_s_total" else {})} for x in t]
# 0.282: speech diagnostic (core/extspeech.py) — Drama 3 on one H200, SGLang-Omni cookbook; not in the LLM summary
from accel_dse.core import extspeech as XS
res["speech"] = {}
for v in ("default", "catalog"):
    t1, t2 = XS.table(v), XS.table(v, t2=True)
    res["speech"][v] = {"summary": XS.summary(t1), "rows": t1, "rows_t2": t2}
# 0.295: speculative-decoding diagnostic (core/extspec.py) — LFM2.5-24B-A2B FP8 + DSpark on one B300; not in the LLM summary
from accel_dse.core import extspec as XP
res["spec"] = {}
for v in ("default", "catalog"):
    t = XP.table(v)
    res["spec"][v] = {"summary": XP.summary(t), "by_concurrency": XP.by_concurrency(t), "rows": t}
if a.calib:
    res["calibration"] = json.load(open(a.calib))
json.dump(res, open(a.out, "w"), indent=1)
for v, s in res["summary"].items():
    print(v, {m: (x["n"], round(x["gmean_ratio"], 2), round(x["mean_abs_err_pct"], 1)) for m, x in s.items()})
for v, s in res["speech"].items():
    print("speech", v, s["summary"])
for v, s in res["spec"].items():
    print("spec", v, s["summary"], [(x["c"], x["speedup_meas"], x["speedup_pred"]) for x in s["by_concurrency"]])
