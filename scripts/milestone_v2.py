#!/usr/bin/env python3
"""Phase-1 milestone numbers for core v2 (writes JSON to stdout / file).

  * Qwen3-32B and DeepSeek-V3 on 8 cards: best layout under each mapping organisation
    (100T chip, HBM3E 8 stacks, ctx 4096, TPOT SLO 50 ms) + ranking stability
  * 100T + LPDDR5X (191 GB/s) Qwen3-8B: TPOT vs batch per mapping, best batch @ SLO 100 ms
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accel_dse.core.evaluate import evaluate  # noqa: E402
from accel_dse.core.mapping import ORGS  # noqa: E402
from accel_dse.core.scenario import Scenario, Serving  # noqa: E402
from accel_dse.core.search import best_batch, search_layouts  # noqa: E402
from accel_dse.core.serving import goodput  # noqa: E402
from accel_dse.core.stability import ranking_stability  # noqa: E402

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "layouts": {}, "lpddr_qwen3_8b": {}}

for mid in ("qwen3-32b", "deepseek-v3"):
    per = {}
    for org in ORGS:
        base = Scenario(model=mid, mem_id=HBM, mapping=org, serving=Serving(ctx=4096, prompt=4096, out_len=1024,
                                                                           tpot_slo_ms=50.0))
        rows = search_layouts(base, 8)
        top = rows[0]
        g = goodput(top.result) if top.batch else None
        st = ranking_stability(base, 8, include_mapping=False)
        per[org] = {
            "top3": [{"layout": r.layout.label, "batch": r.batch, "tok_s_card": r.per_card,
                      "tpot_ms": r.result.tpot * 1e3 if r.result else None,
                      "bound": r.result.bound if r.result else None} for r in rows[:3]],
            "goodput_card": g.goodput_per_card if g else 0.0,
            "stability": {"stable": st.stable, "agree": st.agree,
                          "flips": [c for c in st.cases if not (c["same"] or c["within5"])]},
        }
        print(mid, org, per[org]["top3"][0], f"stable={st.stable} agree={st.agree:.2f}", flush=True)
    out["layouts"][mid] = per

for ctx in (1024, 4096):
    rows = {}
    for org in ORGS:
        base = Scenario(model="qwen3-8b", mem_id=LP, mapping=org, serving=Serving(ctx=ctx, tpot_slo_ms=100.0))
        t = {b: evaluate(base.replace("serving.batch", b)) for b in (1, 8, 64)}
        bb = best_batch(base)
        rows[org] = {"tpot_ms": {b: r.tpot * 1e3 for b, r in t.items()}, "bound": {b: r.bound for b, r in t.items()},
                     "best_batch@100ms": bb.batch, "tok_s@best": bb.result.throughput if bb.batch else 0.0}
    out["lpddr_qwen3_8b"][ctx] = rows
    print("qwen3-8b LPDDR ctx", ctx, rows["os"], flush=True)

dst = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if dst:
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
