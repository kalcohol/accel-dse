"""V4 serving validation grid (0.54): closed-form queueing model (core/pdqueue) vs the request-level DES (core/pdsim).

    PYTHONPATH=. python3 scripts/v4_serving.py [--n 3000] [--n-high 20000] [--seeds 3] [--jobs 8] [--slo] [--families dense8b,moe30b,tp4_32b]
                                               [--out accel_dse/data/v4_serving.json]

Grid: Qwen3-8B on 1P + HBM3E (TP2, batch 64, prompt 4096, out 512, SLO TTFT 400 / TPOT 10 ms, PD prefill 2 + decode 6),
load 0.3 / 0.6 / 0.85 × length CV 0 / 0.5 / 1 × prefix cache off / on (capacity mode: 20000 prefixes × 2048 tokens,
Zipf 1) × modes PD / colocated prefill-first / colocated chunked 512.  Errors are (closed form − DES) / DES, > 0 =
closed form pessimistic (for SLO goodput the sign is flipped so > 0 still means pessimistic).
0.55: + MoE (Qwen3-30B-A3B, TP2×EP2) and 4-card TP (Qwen3-32B, TP4) families on a reduced grid (load × CV 0 / 1);
each point averages ``--seeds`` DES runs and records their spread (``noise``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accel_dse.core.validation import V4_FAMILIES, v4_grid  # noqa: E402


def _clean(x):
    """Strict JSON: non-finite floats → null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3, help="independent DES runs per point (0.55)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (0.55)")
    ap.add_argument("--families", default=",".join(V4_FAMILIES), help="comma list of V4_FAMILIES (0.55)")
    ap.add_argument("--slo", action="store_true", help="also bisect the DES SLO rate (slower)")
    ap.add_argument("--n-high", type=int, default=None, help="requests per DES run at load ≥ --high-load (0.57)")
    ap.add_argument("--high-load", type=float, default=0.85)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "accel_dse" / "data" / "v4_serving.json"))
    a = ap.parse_args()
    t0 = time.time()
    res = v4_grid(n_req=a.n, slo=a.slo, progress=True, seeds=a.seeds, jobs=a.jobs,
                  families=tuple(f for f in a.families.split(",") if f), n_high=a.n_high, high_load=a.high_load)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False, allow_nan=False))
    print("wrote", a.out, res["elapsed_s"], "s")
    for row in res["summary"]:
        print(row)


if __name__ == "__main__":
    main()
