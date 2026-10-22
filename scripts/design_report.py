#!/usr/bin/env python3
"""0.320: design verdict report of one design of an energy DSE (core/report.py; same as POST /api/report).

    python3 scripts/design_report.py --example                      # 1P qwen3-8b decode, mono vs 2 dies, CoWoS-S, accel TCO
    python3 scripts/design_report.py --body dse.json --row 3 --md r.md --json r.json

The body is an insight_dse body (the UI's DSE request).  Without --row: the cheapest $/M row (TCO set), else the best
perf/W row.  Markdown goes to stdout unless --md is given.  Every number carries its label and source; inputs the
body does not give are listed as missing (never 0).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from accel_dse import api  # noqa: E402

EXAMPLE = {"chip_preset": "1P",
           "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
           "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"},
           "mac_scales": [1], "sram_scales": [1], "dies": [1, 2], "cost": "n5_cowos_s", "tco": "accel_2026",
           "d0_per_cm2": 0.1, "slo_tpot_ms": 50, "cooling": "air_hgx"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--body", help="insight_dse body (JSON file)")
    g.add_argument("--example", action="store_true", help="built-in example body")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--md", help="write Markdown here (default: stdout)")
    ap.add_argument("--json", help="write the JSON report here")
    a = ap.parse_args(argv)
    body = EXAMPLE if a.example else json.load(open(a.body, encoding="utf-8"))
    try:
        out = api.api_report({**body, **({"report_row": a.row} if a.row is not None else {})})
    except api.ApiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out["report"], f, ensure_ascii=False, indent=1)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(out["markdown"])
    else:
        sys.stdout.write(out["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
