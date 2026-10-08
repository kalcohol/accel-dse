"""Throughput–interactivity Pareto + SLO goodput (v0.30; v0.31 serving model).

For the *current scenario* (a WorkbenchConfig) sweep the decode batch from 1 up
to the per-card KV-capacity limit, across every parallel layout valid for the
chip count (tp×pp×ep = chips; attention tp|dp; MoE expert sharding tp_ep|ep_all
for LLM), and report

* LLM:     x = tokens/s/user (= 1000 / TPOT_ms), y = tokens/s/chip
           (= B × 1000 / TPOT_ms / chips)
* video:   x = TTFC_ms (lower better), y = frames/s/chip
* protein: x = time/seq_ms (lower better), y = seq/s/chip

plus the Pareto frontier (non-dominated points) and the **SLO goodput**: the
best y meeting the latency SLO, the config achieving it, max concurrent users
and which constraint binds (TTFT / TPOT / capacity).

LLM serving model (「假设」, DECISIONS §19 D-0.30-3 / §20 D-0.31-3):
  * decode runs at batch B continuously; TPOT_step(B) is the engine's per-user
    step interval (PP: max(mb, pp) × tick; spec decode: verify + draft).
    TPOT_decode = TPOT_step / E[tokens/step] (spec decode; E = 1 when off).
  * ``goodput_mode="upper"`` (≤0.30 behaviour): prefill not deducted;
        TPOT = TPOT_decode,  TTFT = TTFT_prefill(1) + TPOT_step
  * ``goodput_mode="amortized"`` (default): every request has P prompt tokens
    and generates N output tokens (``out_len``, default 256) → per decoded
    token the system also owes P/N prefill tokens. Steady state, queueing-free:
      - ``prefill_mode="exclusive"`` (独占): prefills pause decode;
            TPOT = TPOT_decode + B · t_prefill_req / N
            (t_prefill_req = prefill stage time of one prompt, pipelined)
            TTFT = TTFT_prefill(1) + TPOT_step
      - ``prefill_mode="chunked"`` (分块混合, default): prefill chunks ride in
        the decode ticks; per tick r = b·E/N prompts of compute / KV-write /
        collectives are added, weights are read once (shared):
            tick' = max(c + r·c_p, m + r·m_p, comm + r·comm_p, …) + sync + draft
            TPOT  = max(mb, pp) · tick' / E
            TTFT  = pp · max(c + c_p, m + m_p, …) + tick'   (one prompt in one
                    pass while decode continues, plus the in-flight tick)
  * capacity: per-card weights (+ MTP when spec draft = mtp) + KV(B, ctx) ≤
    capacity, same sharding as the evaluator. Activations are not counted.
Video / protein: batch B requests processed together; latency = batch wall
(TTFC / time-per-seq of the batch), throughput = B × units / latency / chips.
"""
from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import replace
from typing import Any, Callable

from .scaleup import TPConfig, per_card_kv_bytes, per_card_weight_bytes, spec_mtp_layers
from .series import get_series, resolve_protein_shape, resolve_video_shape
from .workbench import (
    ParallelOverride,
    WorkbenchConfig,
    build_memory,
    enumerate_parallel_combos,
    evaluate_workbench,
    llm_shape_for_config,
    resolve_parallel,
)

LLM_BATCH_LIMIT = 65536  # v0.31: capacity search cap (was 4096); grid stays log-spaced
DEFAULT_OUT_LEN = 256  # v0.31 output tokens per request (prefill amortisation)
GOODPUT_MODES = ("amortized", "upper")
PREFILL_MODES = ("chunked", "exclusive")
GOODPUT_MODE_ZH = {"amortized": "摊销 prefill（稳态）", "upper": "上界（仅 decode）"}
PREFILL_MODE_ZH = {"chunked": "分块混合", "exclusive": "独占"}
AUX_BATCH_LIMIT = 256  # video / protein (latency grows ~linearly with batch)

DEFAULT_SLO: dict[str, dict[str, float]] = {
    "llm": {"ttft_ms": 2000.0, "tpot_ms": 50.0},
    "video": {"latency_ms": 30000.0},
    "protein": {"latency_ms": 10000.0},
}

AXES: dict[str, dict[str, str]] = {
    "llm": {
        "x_key": "tok_s_user", "x_label": "交互性 tokens/s/用户（= 1/TPOT）", "x_better": "higher",
        "y_key": "tok_s_chip", "y_label": "吞吐 tokens/s/芯片",
        "x_unit": "tok/s", "y_unit": "tok/s",
    },
    "video": {
        "x_key": "latency_ms", "x_label": "TTFC（越小越好）", "x_better": "lower",
        "y_key": "frames_s_chip", "y_label": "吞吐 帧/s/芯片",
        "x_unit": "ms", "y_unit": "帧/s",
    },
    "protein": {
        "x_key": "latency_ms", "x_label": "单批时间 time/seq（越小越好）", "x_better": "lower",
        "y_key": "seq_s_chip", "y_label": "吞吐 序列/s/芯片",
        "x_unit": "ms", "y_unit": "seq/s",
    },
}

BINDING_ZH = {
    "TTFT": "TTFT（首 token 时延）",
    "TPOT": "TPOT（每 token 时延）",
    "latency": "时延 SLO",
    "capacity": "容量（权重 + KV 放不下更大批量）",
    "batch_limit": "搜索批量上限",
    "TTFT+TPOT": "TTFT 与 TPOT 同时超限",
    "none": "—",
}

ASSUMPTIONS_LLM = [
    "假设：稳态 decode 以批量 B 连续运行；TPOT(B) 为引擎 decode 步时延",
    "假设：新请求最多等待一个在途 decode 步，然后单独 prefill（不与 decode 交织 / 不分块）："
    "TTFT_eff(B) = TTFT_prefill(单条 prompt，同布局) + TPOT(B)",
    "假设：吞吐只计 decode（未扣除 prefill 占用）→ tokens/s/芯片 为上界",
    "容量：每卡 权重 + KV(B, ctx=decode_seq_len) ≤ 容量（KV 切分同评估器；未计激活）",
    "PP 布局：decode 以 mb = min(B, pp) 个微批填满流水线（v0.31）；单用户 TPOT 仍含 pp 级遍历",
]
# v0.31 prefill-amortised serving assumptions (replace the "upper bound" lines)
ASSUMPTIONS_LLM_AMORT = {
    "exclusive": [
        "假设：每个请求 prompt P = prompt_len、输出 N 个 token（N = 输出长度）；稳态、无排队",
        "假设（独占）：prefill 暂停 decode；TPOT = TPOT_decode + B × t_prefill(单条) / N；"
        "TTFT = TTFT_prefill(单条) + 一个在途 decode 步",
        "吞吐已扣除 prefill 占用（非上界）",
    ],
    "chunked": [
        "假设：每个请求 prompt P = prompt_len、输出 N 个 token（N = 输出长度）；稳态、无排队",
        "假设（分块混合）：prefill 分块并入 decode 步，权重只读一次；每步额外承担 b·E/N 条 prompt 的"
        "算力 / KV 写 / 集合通信；TTFT = 一条 prompt 与 decode 同步混跑遍历 pp 级 + 一个在途步",
        "吞吐已扣除 prefill 占用（非上界）",
    ],
}
ASSUMPTIONS_AUX = [
    "假设：B 个请求成批处理；时延 = 该批 wall（TTFC / 单批时间），吞吐 = B × 单位 / 时延 / 芯片数",
    "视频 / 蛋白多为算力受限：增大批量几乎不提升每芯片吞吐，曲线主要区分并行布局",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def batch_grid(bmax: int) -> list[int]:
    """1, 2, 3, 4, 6, 8, 12, 16, 24, … ≤ bmax, plus bmax itself."""
    if bmax < 1:
        return []
    out = {1}
    k = 1
    while k <= bmax:
        out.add(k)
        if k * 3 // 2 <= bmax and k >= 2:
            out.add(k * 3 // 2)
        k *= 2
    out.add(bmax)
    return sorted(x for x in out if 1 <= x <= bmax)


def pareto_frontier(xs: list[float], ys: list[float], *, x_better: str = "higher") -> list[int]:
    """Indices of non-dominated points (maximise y; x higher/lower better).

    A point is dominated if another is ≥ in both (after orienting x) and > in
    one. Exact duplicates keep the first index only. Returned sorted by x from
    best interactivity to worst (y strictly increasing along that order).
    """
    sgn = 1.0 if x_better == "higher" else -1.0
    order = sorted(range(len(xs)), key=lambda i: (-sgn * xs[i], -ys[i], i))
    out: list[int] = []
    best_y = -math.inf
    for i in order:
        y = ys[i]
        if y > best_y * (1 + 1e-12) + 1e-15 if best_y > 0 else y > best_y:
            out.append(i)
            best_y = y
    return out


def _domain(cfg: WorkbenchConfig) -> str:
    return str(get_series(cfg.model_id).domain)


Layout = tuple[int, int, int, str, str]  # (tp, pp, ep, attn, moe_shard)


def _cur_moe(cfg: WorkbenchConfig, dom: str, tp: int) -> str:
    """Scenario's MoE sharding ('' for dense / non-LLM; ep_all only meaningful tp>1)."""
    if dom != "llm":
        return ""
    sh = llm_shape_for_config(cfg)
    if not sh.is_moe:
        return ""
    m = str(cfg.moe_shard or "tp_ep")
    return m if tp > 1 else "tp_ep"


def valid_layouts(cfg: WorkbenchConfig, *, mode: str = "all") -> list[Layout]:
    """(tp, pp, ep, attn, moe_shard) layouts valid for cfg.chip_count.

    ``mode="current"`` → only the scenario's own layout. ep>1 only for MoE
    (ep | n_experts); pp ≤ n_layers; attn dp only for LLM with tp>1.
    v0.31 MoE: tp>1 layouts also enumerate ``ep_all`` (experts over all tp·ep
    ranks, needs tp·ep | n_experts) next to the default ``tp_ep``. Dense /
    video / protein layouts carry moe_shard = "".
    """
    dom = _domain(cfg)
    if mode == "current":
        tp, pp, ep = resolve_parallel(cfg.chip_count, cfg.parallel)
        attn = str(cfg.attn_parallel or "tp") if dom == "llm" and tp > 1 else "tp"
        return [(tp, pp, ep, attn, _cur_moe(cfg, dom, tp))]
    n_exp, n_layers = 1, 10**9
    if dom == "llm":
        sh = llm_shape_for_config(cfg)
        n_exp, n_layers = int(sh.n_experts), int(sh.n_layers)
    elif dom == "video":
        n_layers = int(getattr(resolve_video_shape(cfg.model_id), "n_layers", 10**9) or 10**9)
    else:
        n_layers = int(getattr(resolve_protein_shape(cfg.model_id), "n_layers", 10**9) or 10**9)
    out: list[Layout] = []
    moe = dom == "llm" and n_exp > 1
    for tp, pp, ep in enumerate_parallel_combos(cfg.chip_count):
        if ep > 1 and (n_exp <= 1 or n_exp % ep != 0):
            continue
        if pp > n_layers:
            continue
        attns = ["tp", "dp"] if dom == "llm" and tp > 1 else ["tp"]
        shards = [""]
        if moe:
            shards = ["tp_ep"] + (["ep_all"] if tp > 1 and n_exp % (tp * ep) == 0 else [])
        for a in attns:
            for m in shards:
                out.append((tp, pp, ep, a, m))
    return out


def layout_label(tp: int, pp: int, ep: int, attn: str, moe: str = "") -> str:
    s = f"TP{tp}·PP{pp}·EP{ep}"
    s += " · 注意力 DP" if attn == "dp" else ""
    return s + (" · EP全卡" if moe == "ep_all" else "")


def llm_max_batch(
    cfg: WorkbenchConfig, tp: int, pp: int, ep: int, attn: str, *, moe: str | None = None,
    limit: int = LLM_BATCH_LIMIT,
) -> int:
    """Largest batch whose per-card weights + KV fit the per-card capacity
    (same accounting as scaleup.capacity_check). 0 → weights alone don't fit."""
    shape = llm_shape_for_config(cfg)
    mem, _ = build_memory(cfg)
    cap = int(mem.capacity_bytes)
    ms = str(moe or cfg.moe_shard or "tp_ep")
    mtp = spec_mtp_layers(shape, TPConfig(spec_k=int(cfg.spec_k), spec_draft=str(cfg.spec_draft or "mtp")))  # type: ignore[arg-type]
    w = per_card_weight_bytes(shape, tp, cfg.embed_policy, pp=pp, ep=ep, attn_parallel=attn,  # type: ignore[arg-type]
                              moe_shard=ms, mtp_layers=mtp)

    def fits(b: int) -> bool:
        kv = per_card_kv_bytes(shape, cfg.decode_seq_len, b, tp, pp=pp, ep=ep, attn_parallel=attn)
        return w + kv <= cap

    if not fits(1):
        return 0
    if fits(limit):
        return limit
    lo, hi = 1, 2
    while hi < limit and fits(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, limit)
    while hi - lo > 1:  # invariant: fits(lo), not fits(hi)
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def serving_metrics(
    c: Any, *, b1: Any, mode: str, prefill_mode: str, out_len: int,
) -> dict[str, float]:
    """LLM per-user TPOT / TTFT under the chosen serving model (see module doc).

    ``c`` = MetricsCard at batch B, ``b1`` = MetricsCard at B=1 (same layout;
    provides the single-prompt prefill). Returns TPOT_ms, TTFT_ms, TPOT_step_ms,
    TPOT_decode_ms, TTFT_prefill_ms, prefill_share.
    """
    step = float(c.TPOT_step_ms or c.TPOT_ms)
    e = max(float(c.spec_tokens_per_step or 1.0), 1e-12)
    dec_tpot = step / e
    ttft1 = float(b1.TTFT_ms)
    pf = (b1.phase_detail or {}).get("prefill") or {}
    d = (c.phase_detail or {}).get("decode") or {}
    B = int(c.batch)
    N = max(int(out_len), 1)
    if mode == "upper" or not pf or not d:
        tpot, ttft = dec_tpot, ttft1 + step
    elif prefill_mode == "exclusive":
        t_req = float(pf.get("stage_ms", ttft1))
        tpot = dec_tpot + B * t_req / N
        ttft = ttft1 + step
    else:  # chunked / mixed
        r = float(d.get("microbatch", B)) * e / N  # prompts folded into each tick
        keys = (("compute_ms", "compute_ms"), ("mem_ms", "mem_nonweight_ms"),
                ("c2c_ms", "c2c_ms"), ("pp_act_ms", "pp_act_ms"), ("a2a_ms", "a2a_ms"))
        extra = float(d.get("sync_ms", 0.0)) + float(d.get("draft_ms", 0.0))
        fab = float(d.get("fabric_ms", 0.0))
        tick = max([float(d.get(k, 0.0)) + r * float(pf.get(kp, 0.0)) for k, kp in keys] + [fab]) + extra
        slots = int(d.get("slots", 1)) or 1
        tpot = slots * tick / e
        tick_p = max([float(d.get(k, 0.0)) + float(pf.get(kp, 0.0)) for k, kp in keys] + [fab]) + extra
        ttft = int(d.get("pp", 1) or 1) * tick_p + tick
    share = max(0.0, 1.0 - dec_tpot / tpot) if tpot > 0 else 0.0
    return {"TPOT_ms": tpot, "TTFT_ms": ttft, "TPOT_step_ms": step, "TPOT_decode_ms": dec_tpot,
            "TTFT_prefill_ms": ttft1, "prefill_share": share}


def pareto_analysis(
    cfg: WorkbenchConfig,
    *,
    slo: dict[str, float] | None = None,
    layouts: str = "all",
    batch_limit: int | None = None,
    goodput_mode: str = "amortized",
    prefill_mode: str = "chunked",
    out_len: int | None = None,
) -> dict[str, Any]:
    """Throughput–interactivity sweep + Pareto frontier + SLO goodput (JSON-ready)."""
    t0 = time.perf_counter()
    dom = _domain(cfg)
    ax = AXES[dom]
    if goodput_mode not in GOODPUT_MODES:
        raise ValueError(f"goodput_mode must be one of {GOODPUT_MODES}")
    if prefill_mode not in PREFILL_MODES:
        raise ValueError(f"prefill_mode must be one of {PREFILL_MODES}")
    n_out = int(out_len) if out_len is not None else DEFAULT_OUT_LEN
    if n_out < 1:
        raise ValueError("out_len must be >= 1")
    s = dict(DEFAULT_SLO[dom])
    for k, v in (slo or {}).items():
        if v is not None and k in s:
            s[k] = float(v)
    chips = int(cfg.chip_count)
    cache: dict[tuple[int, int, int, str, str, int], Any] = {}
    n_evals = 0

    def ev(tp: int, pp: int, ep: int, attn: str, moe: str, b: int):
        nonlocal n_evals
        key = (tp, pp, ep, attn, moe, b)
        if key not in cache:
            c2 = replace(cfg, batch=int(b), parallel=ParallelOverride(tp=tp, pp=pp, ep=ep),
                         attn_parallel=attn, moe_shard=moe or cfg.moe_shard or "tp_ep")
            cache[key] = evaluate_workbench(c2, scale_baseline=False)
            n_evals += 1
        return cache[key]

    lays = valid_layouts(cfg, mode=layouts)
    points: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    cur_tp, cur_pp, cur_ep = resolve_parallel(cfg.chip_count, cfg.parallel)
    cur_attn = str(cfg.attn_parallel or "tp") if cur_tp > 1 else "tp"
    cur_moe = _cur_moe(cfg, dom, cur_tp)

    def make_point(tp: int, pp: int, ep: int, attn: str, moe: str, b: int, bmax: int) -> dict[str, Any]:
        c = ev(tp, pp, ep, attn, moe, b)
        p: dict[str, Any] = {
            "layout": layout_label(tp, pp, ep, attn, moe), "tp": tp, "pp": pp, "ep": ep, "attn": attn,
            "moe_shard": moe, "batch": b, "chips": chips, "max_batch": bmax, "wall": c.wall,
            "capacity_needed_GB": c.capacity_needed_GB, "capacity_GB": c.capacity_GB,
            "oom": bool(c.oom),
        }
        if dom == "llm":
            m = serving_metrics(c, b1=ev(tp, pp, ep, attn, moe, 1), mode=goodput_mode,
                                prefill_mode=prefill_mode, out_len=n_out)
            tpot = m["TPOT_ms"]
            p.update(m)
            p.update({
                "tok_s_user": 1000.0 / tpot if tpot > 0 else 0.0,
                "tok_s_chip": b * 1000.0 / tpot / chips if tpot > 0 else 0.0,
                "latency_ms": tpot,
                "decode_mb": int(c.decode_mb_eff),
                "spec_tokens_per_step": float(c.spec_tokens_per_step),
            })
            viol = []
            if p["TTFT_ms"] > s["ttft_ms"]:
                viol.append("TTFT")
            if tpot > s["tpot_ms"]:
                viol.append("TPOT")
        else:
            lat = float(c.TTFC_ms if dom == "video" else c.time_per_seq_ms)
            p["latency_ms"] = lat
            if dom == "video":
                fps_req = float(c.frames_per_s)  # n_frames / TTFC (per request)
                p["frames_s_chip"] = b * fps_req / chips
                p["TTFC_ms"] = lat
            else:
                p["seq_s_chip"] = b * 1000.0 / lat / chips if lat > 0 else 0.0
                p["time_per_seq_ms"] = lat
            viol = ["latency"] if lat > s["latency_ms"] else []
        if c.oom:
            viol.append("capacity")
        p["violates"] = viol
        p["meets_slo"] = not viol
        p["x"] = float(p[ax["x_key"]])
        p["y"] = float(p[ax["y_key"]])
        p["current"] = (tp, pp, ep, attn, moe, b) == (cur_tp, cur_pp, cur_ep, cur_attn, cur_moe, int(cfg.batch))
        p["label"] = f"{p['layout']} · B={b}"
        return p

    best: dict[str, Any] | None = None
    blimit = int(batch_limit or (LLM_BATCH_LIMIT if dom == "llm" else AUX_BATCH_LIMIT))
    for tp, pp, ep, attn, moe in lays:
        if dom == "llm":
            bmax = llm_max_batch(cfg, tp, pp, ep, attn, moe=moe or None, limit=blimit)
        else:
            bmax = _aux_max_batch(lambda b: ev(tp, pp, ep, attn, moe, b), blimit)
        row: dict[str, Any] = {"layout": layout_label(tp, pp, ep, attn, moe), "tp": tp, "pp": pp,
                               "ep": ep, "attn": attn, "moe_shard": moe, "max_batch": bmax}
        if bmax < 1:
            row.update({"fits": False, "goodput": None, "binding": "capacity"})
            if dom == "llm":
                row["why_not"] = _why_not_fit(cfg, tp, pp, ep, attn, moe)
            layout_rows.append(row)
            continue
        row["fits"] = True
        row["TTFT_prefill_ms"] = float(ev(tp, pp, ep, attn, moe, 1).TTFT_ms) if dom == "llm" else None
        grid = batch_grid(bmax)
        lay_pts: dict[int, dict[str, Any]] = {}

        def pt(b: int) -> dict[str, Any]:
            if b not in lay_pts:
                lay_pts[b] = make_point(tp, pp, ep, attn, moe, b, bmax)
            return lay_pts[b]

        for g in grid:  # full log-spaced grid (frontier spans beyond the SLO)
            pt(g)
        sampled = sorted(lay_pts)
        bf = 0
        if lay_pts[1]["meets_slo"]:
            lo = 1
            for g in sampled:
                if lay_pts[g]["meets_slo"]:
                    lo = g
                else:
                    break
            nxt = [g for g in grid if g > lo]
            if nxt:
                hi = nxt[0]
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if pt(mid)["meets_slo"]:
                        lo = mid
                    else:
                        hi = mid
            bf = lo
        row["max_users_slo"] = bf
        if bf >= 1:
            cand = [p for b, p in lay_pts.items() if b <= bf and p["meets_slo"]]
            # best y (exact ties → larger batch); max_users reported separately (bf)
            gp = max(cand, key=lambda p: (p["y"], p["batch"]))
            if bf >= bmax:
                binding = "batch_limit" if bmax >= blimit else "capacity"
            else:
                binding = "+".join(v for v in pt(bf + 1)["violates"] if v != "capacity") or "capacity"
            row["goodput"] = gp["y"]
            row["goodput_batch"] = gp["batch"]
            row["binding"] = binding
            cand_best = {"point": gp, "max_users": bf, "binding": binding}
            if best is None or (gp["y"], gp["x"] if ax["x_better"] == "higher" else -gp["x"]) > (
                best["point"]["y"],
                best["point"]["x"] if ax["x_better"] == "higher" else -best["point"]["x"],
            ):
                best = cand_best
        else:
            row["goodput"] = None
            row["binding"] = "+".join(v for v in lay_pts[1]["violates"]) or "none"
        points.extend(lay_pts[b] for b in sorted(lay_pts))
        layout_rows.append(row)

    # frontier over capacity-feasible points
    ok_idx = [i for i, p in enumerate(points) if not p["oom"]]
    fr = pareto_frontier([points[i]["x"] for i in ok_idx], [points[i]["y"] for i in ok_idx],
                         x_better=ax["x_better"])
    front = [ok_idx[j] for j in fr]
    fset = set(front)
    for i, p in enumerate(points):
        p["frontier"] = i in fset
        p["id"] = i

    goodput = _goodput_summary(dom, s, best, layout_rows, ax)
    serving = None
    assumptions = ASSUMPTIONS_AUX
    if dom == "llm":
        serving = {
            "goodput_mode": goodput_mode, "goodput_mode_zh": GOODPUT_MODE_ZH[goodput_mode],
            "prefill_mode": prefill_mode, "prefill_mode_zh": PREFILL_MODE_ZH[prefill_mode],
            "out_len": n_out, "prompt_len": int(cfg.prompt_len),
            "spec_k": int(cfg.spec_k), "spec_accept": float(cfg.spec_accept),
        }
        goodput.update({"goodput_mode": goodput_mode, "prefill_mode": prefill_mode, "out_len": n_out})
        if goodput_mode == "upper":
            assumptions = list(ASSUMPTIONS_LLM)
        else:
            assumptions = ASSUMPTIONS_LLM_AMORT[prefill_mode] + ASSUMPTIONS_LLM[3:]
        if int(cfg.spec_k) > 0:
            assumptions = assumptions + [
                f"假设：投机解码 k={int(cfg.spec_k)}、接受率 a={float(cfg.spec_accept):g}（i.i.d.），"
                "每步期望产出 (1−a^(k+1))/(1−a) 个 token；TPOT = 步时延 / 期望 token 数"
            ]
    return {
        "ok": True,
        "domain": dom,
        "model_id": cfg.model_id,
        "chips": chips,
        "axes": ax,
        "slo": s,
        "points": points,
        "frontier": front,
        "layouts": layout_rows,
        "goodput": goodput,
        "serving": serving,
        "assumptions": assumptions,
        "batch_limit": blimit,
        "n_evals": n_evals,
        "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        "current": {"tp": cur_tp, "pp": cur_pp, "ep": cur_ep, "attn": cur_attn, "moe_shard": cur_moe,
                    "batch": int(cfg.batch)},
    }


def _why_not_fit(cfg: WorkbenchConfig, tp: int, pp: int, ep: int, attn: str, moe: str) -> str:
    """Short reason a layout does not fit at B=1 (weights vs capacity, GiB)."""
    shape = llm_shape_for_config(cfg)
    mem, _ = build_memory(cfg)
    mtp = spec_mtp_layers(shape, TPConfig(spec_k=int(cfg.spec_k), spec_draft=str(cfg.spec_draft or "mtp")))  # type: ignore[arg-type]
    w = per_card_weight_bytes(shape, tp, cfg.embed_policy, pp=pp, ep=ep, attn_parallel=attn,  # type: ignore[arg-type]
                              moe_shard=moe or "tp_ep", mtp_layers=mtp)
    kv = per_card_kv_bytes(shape, cfg.decode_seq_len, 1, tp, pp=pp, ep=ep, attn_parallel=attn)
    return (f"每卡权重 {w / 2**30:.1f} GB + KV(B=1) {kv / 2**30:.2f} GB > 容量 "
            f"{mem.capacity_bytes / 2**30:.0f} GB")


def _aux_max_batch(ev1: Callable[[int], Any], limit: int) -> int:
    """Video / protein: largest batch without OOM (evaluator flag), ≤ limit."""
    if ev1(1).oom:
        return 0
    if not ev1(limit).oom:
        return limit
    lo, hi = 1, 2
    while hi < limit and not ev1(hi).oom:
        lo, hi = hi, hi * 2
    return _bisect(ev1, lo, min(hi, limit))


def _bisect(ev1: Callable[[int], Any], lo: int, hi: int) -> int:
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ev1(mid).oom:
            hi = mid
        else:
            lo = mid
    return lo


def _goodput_summary(dom: str, slo: dict[str, float], best: dict[str, Any] | None,
                     rows: list[dict[str, Any]], ax: dict[str, str]) -> dict[str, Any]:
    if best is not None:
        p = best["point"]
        out = {
            "ok": True,
            "y": p["y"], "x": p["x"],
            "config": p["label"], "layout": p["layout"],
            "tp": p["tp"], "pp": p["pp"], "ep": p["ep"], "attn": p["attn"],
            "moe_shard": p.get("moe_shard", ""), "batch": p["batch"],
            "max_users": best["max_users"],
            "binding": best["binding"],
            "binding_zh": BINDING_ZH.get(best["binding"], best["binding"]),
            "point_id": p.get("id"),
        }
        if dom == "llm":
            out.update({"tok_s_chip": p["tok_s_chip"], "tok_s_user": p["tok_s_user"],
                        "TTFT_ms": p["TTFT_ms"], "TPOT_ms": p["TPOT_ms"],
                        "TPOT_step_ms": p.get("TPOT_step_ms"), "TPOT_decode_ms": p.get("TPOT_decode_ms"),
                        "prefill_share": p.get("prefill_share"), "decode_mb": p.get("decode_mb"),
                        "spec_tokens_per_step": p.get("spec_tokens_per_step")})
        else:
            out.update({ax["y_key"]: p["y"], "latency_ms": p["latency_ms"]})
        return out
    fits = [r for r in rows if r.get("fits")]
    if not fits:
        binding = "capacity"
    else:
        sets = [set(str(r.get("binding") or "").split("+")) for r in fits]
        if dom != "llm":
            binding = "latency"
        elif all("TPOT" in x for x in sets):
            binding = "TPOT"
        elif all("TTFT" in x for x in sets):
            binding = "TTFT"
        else:
            binding = "TTFT+TPOT"
    return {"ok": False, "y": 0.0, "max_users": 0, "binding": binding,
            "binding_zh": BINDING_ZH.get(binding, binding), "config": None}


CSV_FIELDS = [
    "domain", "model_id", "chips", "layout", "tp", "pp", "ep", "attn", "batch", "max_batch",
    "x_key", "x", "y_key", "y", "TPOT_ms", "TTFT_ms", "TTFT_prefill_ms", "latency_ms",
    "capacity_needed_GB", "capacity_GB", "wall", "frontier", "meets_slo", "violates", "current",
    # v0.31 (appended to keep column order stable)
    "moe_shard", "TPOT_step_ms", "TPOT_decode_ms", "prefill_share", "decode_mb", "spec_tokens_per_step",
    "goodput_mode", "prefill_mode", "out_len",
]


def pareto_to_csv(result: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    ax = result.get("axes") or {}
    for p in result.get("points") or []:
        row = dict(p)
        sv = result.get("serving") or {}
        row.update({
            "domain": result.get("domain"), "model_id": result.get("model_id"),
            "x_key": ax.get("x_key"), "y_key": ax.get("y_key"),
            "violates": "+".join(p.get("violates") or []),
            "goodput_mode": sv.get("goodput_mode", ""), "prefill_mode": sv.get("prefill_mode", ""),
            "out_len": sv.get("out_len", ""),
        })
        w.writerow(row)
    return buf.getvalue()


def format_pareto_text(result: dict[str, Any]) -> str:
    ax = result["axes"]
    g = result["goodput"]
    lines = [
        f"pareto  model={result['model_id']} chips={result['chips']} domain={result['domain']} "
        f"points={len(result['points'])} frontier={len(result['frontier'])} evals={result['n_evals']}",
        f"  x={ax['x_key']} ({ax['x_better']} better)  y={ax['y_key']}",
        f"  SLO {result['slo']}",
    ]
    sv = result.get("serving")
    if sv:
        lines.append(f"  serving: {sv['goodput_mode']} prefill={sv['prefill_mode']} "
                     f"P={sv['prompt_len']} N={sv['out_len']} spec_k={sv['spec_k']}")
    if g.get("ok"):
        lines.append(
            f"  goodput: {ax['y_key']}={g['y']:.4g} @ {g['config']}  max_users={g['max_users']}  "
            f"binding={g['binding']}"
        )
    else:
        lines.append(f"  goodput: none meets SLO (binding={g['binding']})")
    lines.append("  frontier:")
    for i in result["frontier"]:
        p = result["points"][i]
        lines.append(f"    {p['label']:<40s} x={p['x']:.4g} y={p['y']:.4g}"
                     f"{'  ✓SLO' if p['meets_slo'] else ''}")
    for a in result["assumptions"]:
        lines.append(f"  [{a}]")
    return "\n".join(lines)
