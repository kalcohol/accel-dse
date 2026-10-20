"""0.89: CP work KPI exact (DSA top-k + LM head); export helpers in app.js; fabric auto vs auto_overlap monotonicity."""
from __future__ import annotations

import collections
import random
import re
from dataclasses import replace
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, _cdiv, build_rank_ops
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

H = "hbm3e_8s_12h24g_9200"


def _work(r):
    L = r.scenario.layout
    return sum(s.flops_u for s in r.stages) * L.tp * L.sp * L.dp * L.cp / r.scenario.serving.batch


def test_dsa_topk_share_exact():
    m = get_model("deepseek-v3.2")
    for S, cp in ((131072, 8), (131072, 3), (8192, 4), (1048576, 8)):
        q = _cdiv(S, cp); c = round(S * S / (2 * cp * q) - q / 2)

        def agg(ph):
            d = collections.Counter()
            for o in build_rank_ops(m, 3, ph, Shard()):
                if o.kind != "comm":
                    d[o.name] += (o.flops + o.vec) * o.share
            return d
        g, k = agg(Phase("prefill", 1, q=S)), agg(Phase("prefill", cp, q=q, ctx=c, cp=cp, cp_len=S))
        for n in ("qk", "pv", "softmax"):
            assert abs(k[n] / g[n] - 1) < 1e-9, (S, cp, n, k[n] / g[n])


def test_cp_ttft_unchanged_by_share_fix():
    r = evaluate(Scenario(model="deepseek-v3.2", layout=Layout(cp=8, ep=8), mem_id=H, node_cards=8,
                          serving=Serving(phase="prefill", batch=1, prompt=131072), exec_overlap="stage"))
    assert abs(r.ttft - 13.9215) < 5e-3       # 0.88 value 13.921 (the share fix touches the work KPI, not time); 0.193 13.9062 (WIDE_WAVES: no N padding); 0.202 13.9215 = the 0.88 value again (GPU wave refill no longer on NPU)


def test_cp_whole_model_work_exact_incl_lm_head():
    for mid, L, L1 in (("llama-3.1-8b", Layout(tp=4, cp=8), Layout(tp=4)), ("qwen3-1.7b", Layout(pp=2, tp=2, cp=8), Layout(pp=2, tp=2)),
                       ("qwen3-30b-a3b", Layout(tp=4, cp=4, ep=16), Layout(tp=4, ep=4))):
        for B in (1, 3):
            sv = Serving(phase="prefill", batch=B, prompt=4096)
            a, b = (evaluate(Scenario(model=mid, layout=x, mem_id=H, serving=sv)) for x in (L, L1))
            assert abs(_work(a) / _work(b) - 1) < 1e-9, (mid, B, _work(a) / _work(b))


def test_fabric_auto_overlap_monotone_in_link_bw():
    """auto picks per collective (bw + α, like a runtime tuner): a faster link can pick a higher-α algorithm whose
    bandwidth hides under compute — ≤ 0.4 % non-monotone (0.89 audit, MODEL §35).  auto_overlap minimises the stage
    time and is monotone."""
    rnd = random.Random(5)
    n = 0
    for _ in range(40):
        mid = rnd.choice(["qwen3-8b", "qwen3-30b-a3b", "internlm2-5-20b"])
        tp = rnd.choice([2, 4, 8, 16]); cp = rnd.choice([1, 2, 4]); moe = mid == "qwen3-30b-a3b"
        d = {"model": mid, "mem_id": H, "package_cards": rnd.choice([2, 4]), "d2d_enabled": True,
             "fabric": {"enabled": True, "algo": "auto_overlap"}, "node_cards": rnd.choice([0, 4, 8]),
             "layout": {"pp": rnd.choice([1, 2]), "tp": tp, "ep": tp * cp if moe else 1, "cp": cp},
             "serving": {"phase": "prefill", "batch": rnd.choice([1, 2, 8]), "prompt": rnd.choice([4096, 32768])}}
        try:
            s = Scenario.from_dict(d)
            a = evaluate(s)
        except ValueError:
            continue
        b = evaluate(replace(s, link=replace(s.link, GBps=s.link.GBps / 4)))
        assert b.ttft >= a.ttft * (1 - 1e-12), d
        n += 1
    assert n >= 30


def test_ui_export_wiring():
    js, html = W.read(), W.read(W.INDEX_HTML)          # 0.286: parsed (ids / call sites / string literals)
    ids = W.html_ids(html)
    for box in ("cmp", "sw"):
        for f in ("csv", "json"):
            assert f"{box}-exp-{f}" in ids
    assert W.count(js, "setExport('cmp'") == 1 and W.count(js, "setExport('sw'") == 2      # compare; sweep + Pareto
    assert W.has(W.functions(js)["expCSV"], "return '\\ufeff' +")                      # BOM first (Excel reads UTF-8)
    assert W.has(W.functions(js)["csvCell"], "if (/^[=+\\-@\\t\\r]/.test(t)) t = \"'\" + t;")   # formula-injection guard
    for lab in ("有效吞吐 goodput（tok/s/卡）", "TPOT（ms）", "有效 MAC 利用率（%）", "prefill TTFT（ms）"):
        assert lab in W.strings(js), lab
    assert W.count(js, "clearExport('cmp')") >= 1 and W.count(js, "clearExport('sw')") == 2  # no stale export mid-run
