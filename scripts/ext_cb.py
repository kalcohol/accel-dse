#!/usr/bin/env python3
"""0.69 V6.3 (a): in-flight-batching rows (max_tok_s_total) scored with the continuous-batching proxy
(extval.predict_cb) vs the static 'prefill-all-then-decode' proxy (extval.predict).  Writes JSON + a markdown table."""
import json, math, sys
from accel_dse.core import extval as X

VS = ("catalog", "catalog_tbo", "catalog_kernel", "catalog_serial", "peak_serial_m")
R = [r for p in (X.DATA, X.DATA_MOE) for r in X.rows(p) if r["metric"] == "max_tok_s_total"]
out = {"rows": {}}
for v in VS:
    out["rows"][v] = [{"id": r["id"], "src": r["src"], "hw": r["hw"], "isl": r["isl"], "osl": r["osl"],
                       "static": X.predict(r, v)["pred"] / r["value"], "cb": X.predict_cb(r, v)["pred"] / r["value"]}
                      for r in R]
json.dump(out, open(sys.argv[1], "w"), indent=1)


def st(xs):
    l = [math.log(x) for x in xs if x == x]
    return f"{math.exp(sum(l) / len(l)):.2f} / {100 * (math.exp(sum(abs(a) for a in l) / len(l)) - 1):.0f}%"


groups = {"all": lambda x: True, "prefill-heavy (ISL ≥ 4·OSL)": lambda x: x["isl"] >= 4 * x["osl"],
          "decode-heavy": lambda x: x["isl"] < 4 * x["osl"]}
print("| group | n | " + " | ".join(f"{v} static | {v} CB" for v in VS) + " |")
for g, f in groups.items():
    xs = {v: [x for x in out["rows"][v] if f(x)] for v in VS}
    print(f"| {g} | {len(xs[VS[0]])} | " + " | ".join(f"{st([x['static'] for x in xs[v]])} | {st([x['cb'] for x in xs[v]])}"
                                              for v in VS) + " |")
