#!/usr/bin/env python3
"""0.71: attention / expert GEMM utilisation vs array size × core count × instance schedule (MODEL §26)."""
import json, sys
from dataclasses import replace
from accel_dse.core import evaluate as E
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving

base = CHIPS["1P"]                       # 128 × 128 × 32 = 524,288 MACs
GEOM = [(512, 512, 2), (256, 256, 8), (128, 128, 32), (64, 64, 128), (32, 32, 512), (128, 256, 16), (64, 128, 64)]
CASES = {
    "DS-V3 MLA decode b64 ctx4k": ("deepseek-v3", dict(phase="decode", batch=64, ctx=4096)),
    "DS-V3 MLA decode b8 ctx32k": ("deepseek-v3", dict(phase="decode", batch=8, ctx=32768)),
    "Qwen3-32B GQA decode b64 ctx4k": ("qwen3-32b", dict(phase="decode", batch=64, ctx=4096)),
    "Qwen3-32B GQA decode b8 ctx128k": ("qwen3-32b", dict(phase="decode", batch=8, ctx=131072)),
    "Qwen3-32B prefill b1 4k": ("qwen3-32b", dict(phase="prefill", batch=1, prompt=4096)),
    "DS-V3 prefill b1 4k": ("deepseek-v3", dict(phase="prefill", batch=1, prompt=4096)),
    "Qwen3-30B-A3B MoE decode b64 ctx4k": ("qwen3-30b-a3b", dict(phase="decode", batch=64, ctx=4096)),
}
log = []
o_s = E._op_seconds
o_sum = E._sum_ops
def sum_ops(ops_, sys_, org, model, memos=None):
    for o in ops_:
        if o.kind in ("attn", "gemm"):
            log.append((o, o_s(o, sys_, org, model)))
    return o_sum(ops_, sys_, org, model, memos)
E._sum_ops = sum_ops
out = []
for case, (mid, sv) in CASES.items():
    for (r, c, e) in GEOM:
        for sched in ("wide", "split", "auto"):
            ch = replace(base, rows=r, cols=c, engines=e, instance_sched=sched, name=f"{r}x{c}x{e}")
            log.clear()
            res = E.evaluate(Scenario(model=mid, chip=ch, serving=Serving(**sv)))
            agg = {}
            for o, x in log:
                cls = "attn" if o.kind == "attn" else ("expert" if o.role == "expert" else "dense_gemm")
                a = agg.setdefault(cls, [0.0, 0.0]); a[0] += x[0]; a[1] += x[5]
            t = res.stages[0].time
            out.append(dict(case=case, geom=f"{r}x{c}x{e}", sched=sched, step_ms=res.step * 1e3,
                            t_arr_attn_ms=sum(s.time.t_arr_attn for s in res.stages) * 1e3,
                            t_array_ms=sum(s.time.t_array for s in res.stages) * 1e3,
                            t_dram_ms=sum(s.time.t_dram for s in res.stages) * 1e3, bound=res.bound,
                            util={k: (v[1] / v[0] if v[0] else None) for k, v in agg.items()}))
json.dump(out, open(sys.argv[1], "w"), indent=1)
for case in CASES:
    print(f"\n### {case}\n| geometry R×C×E | sched | attn util | expert util | dense util | attn array ms | step ms | bound |\n|---|---|---|---|---|---|---|---|")
    for x in out:
        if x["case"] == case:
            u = x["util"]; f = lambda k: f"{100*u[k]:.0f}%" if u.get(k) else "—"
            print(f"| {x['geom']} | {x['sched']} | {f('attn')} | {f('expert')} | {f('dense_gemm')} | {x['t_arr_attn_ms']:.2f} | {x['step_ms']:.2f} | {x['bound']} |")
