"""Prefill / decode disaggregation (PD) — steady-state fluid model, first cut (0.50).  Off by default.

Colocated serving (today's goodput, core/serving.py) time-shares each replica between prefill and decode.  With PD
(DistServe / Splitwise / Mooncake style) a prefill pool and a decode pool run on separate cards, possibly with
different layouts, and each request's KV cache moves prefill → decode.

Pools (per scenario; the decode pool uses ``scenario.layout``, the prefill pool ``pd.prefill_layout``)
  prefill   N_p cards = r_p replicas of c_p cards; each runs the largest prefill batch (powers of two ≤ 64) that meets
            the TTFT SLO — the same choice as colocated goodput (best_prefill).  Capacity λ_p = r_p · R_p / S requests/s
            (R_p prompt tokens/s per replica, S prompt length).
  decode    N_d cards = r_d replicas of c_d cards at the scenario's decode batch.  λ_d = r_d · R_d / out_len.
  KV        per request = the logical KV / indexer / recurrent state of the whole model at S tokens (as released dtype,
            no TP replication — fan-out inside the decode pool is not charged) 「假设」.  Tier between the pools
            (0.50 three tiers, 「假设」): the cross-node network ``net`` when ``node_cards`` > 0 (pools on different
            nodes), else the in-node scale-up ``link`` (one scale-up domain).  Per-request bandwidth
            β_req = min(c_p, c_d) · β (card-pair streams in parallel), pool capacity λ_kv = min(N_p, N_d) · β / KV;
            β = ``pd.kv_GBps`` or that tier's GBps, α = that tier's α per transfer (per layer chunk when layer-wise).  Transfer time t_kv = α_tot + KV / β_req.
            Exposed in TTFT: all of t_kv, or with ``pd.kv_layerwise`` only what prefill compute cannot hide:
            max(α + KV/β_req/L, t_kv − TTFT_p·(L−1)/L).
Metrics
  TTFT = TTFT_p + exposed KV transfer;  TPOT = the decode pool's TPOT (no prefill interference)
  goodput = min(λ_p, λ_d, λ_kv) · out_len output tokens/s over N_p + N_d cards; the minimum names the bottleneck,
  the other pools' utilisation = λ / their capacity.
Colocated comparison on the same total cards with the scenario's layout: floor(N / c_d) replicas × colocated goodput;
its TTFT is the colocated prefill latency, its TPOT the pure decode step and the *effective* TPOT = TPOT / decode share
(prefills pause decoding on a time-shared replica: the fluid average inter-token time) 「假设」.
Split search: every N_p (multiple of c_p) with N_d = N − N_p a multiple of c_d, same total N, best goodput per card.

0.52: with a request-length mix or a prefix-cache hit rate (core/lengths.py, ``pd.prefix_hit``) the fluid capacities
use the mix — prefill λ_p = r_p / Σ w_i·TTFT(b, S_i, p_i)/b at the pool's batch, decode λ_d = r_d·R_d(ctx_eff) / E[out],
KV λ_kv = min(N_p, N_d)·β / E[KV transferred] — and the colocated comparison the same per-request prefill time; the
TTFT shown is that of a representative request (S = round(E[S])).  Defaults (fixed lengths, no prefix) take the 0.50
code path unchanged.

Not modelled (first cut): queueing / arrival burstiness and its TTFT tail, continuous-batching dynamics and chunked
prefill, KV transfer contending with the pools' own collectives, prefix caching, different chips / memories per
pool, energy of the PD system, and searching the pools' layouts (both are inputs; only the split is searched).
"""

from __future__ import annotations

import math
from dataclasses import replace

from .catalog import get_model
from .evaluate import Result, evaluate
from .ir import Shard
from .parallel import enumerate_layouts
from .memplan import stage_storage
from .scenario import PDConfig, Scenario
from .energy import EnergyTable
from .lengths import lengths_of
from .pdqueue import _Pool, _pd_mode, _slo_rate, prefix_tokens, queue_report
from .serving import Goodput, best_prefill, goodput


def kv_bytes_per_request(model, prompt: int) -> float:
    """Logical KV + indexer keys + recurrent state of the whole model after a ``prompt``-token prefill."""
    st = stage_storage(model, 0, model.n_layers, True, True, Shard(), prompt, 0)
    return st.kv_per_seq + st.idx_per_seq + st.state_per_seq


def _pool(cards: int, per: int) -> tuple[int, int]:
    n = cards or per
    return n, n // per


def disagg_report(scn: Scenario, decode: Result | None = None, energy: EnergyTable | None = None,
                  queue: bool = True) -> dict:
    """PD metrics for an LLM scenario with ``scn.pd.enabled`` (decode pool = scn.layout at scn.serving.batch)."""
    m = get_model(scn.model)
    if not m.kv_cache:
        raise ValueError("PD 分离只适用于 LLM / VLM（视频 / 蛋白质模型没有 prefill / decode 两阶段）")
    pd, sv = scn.pd, scn.serving
    lens, h = lengths_of(pd, sv), pd.prefix_hit
    plain = lens.trivial and not h and sv.prefix_cached == 0
    S_rep = max(1, round(lens.mean_S))
    p_rep = prefix_tokens(S_rep, h)
    ctx_eff = sv.ctx if lens.trivial else max(1, round(sv.ctx * lens.ctx_ratio))
    dscn = scn.replace("serving.phase", "decode")
    if not plain:
        dscn = dscn.replace("serving.prefix_cached", 0).replace("serving.prompt", S_rep) \
            .replace("serving.prefix_cached", p_rep).replace("serving.ctx", ctx_eff)
    dec = decode if (plain and decode is not None and decode.scenario.serving.phase == "decode") else evaluate(dscn)
    pscn = dscn.replace("layout", pd.prefill_layout)
    ppool, dpool = _Pool(pscn), _Pool(dscn)
    pb, pr = best_prefill(pscn)
    p_ok = pr is not None
    if pr is None:
        pb, pr = 1, evaluate(pscn.replace("serving.phase", "prefill").replace("serving.batch", 1))
    c_p, c_d = pd.prefill_layout.cards, scn.layout.cards
    n_p, r_p = _pool(pd.prefill_cards, c_p)
    n_d, r_d = _pool(pd.decode_cards, c_d)
    S, out = (sv.prompt, sv.out_len) if lens.trivial else (S_rep, lens.mean_out)
    pts = [(w, Si, prefix_tokens(Si, h)) for w, Si in lens.prompts()]
    kv_full = {(Si, pi): kv_bytes_per_request(m, Si) for _, Si, pi in pts}
    kv_full.setdefault((S_rep, p_rep), kv_bytes_per_request(m, S_rep))
    kv_new = {k: (v * (k[0] - k[1]) / k[0] if k[1] else v) for k, v in kv_full.items()}   # KV of the uncached tokens
    kv_xfer = kv_new if pd.prefix_on_decode else kv_full                       # 「假设」 decode holds the prefix
    kv = kv_bytes_per_request(m, S) if plain else sum(w * kv_xfer[(Si, pi)] for w, Si, pi in pts)
    # KV hand-off tier (「假设」): the two pools sit on different nodes when the cross-node tier is modelled
    # (node_cards > 0) → network link; otherwise one scale-up domain → the in-node link.  pd.kv_GBps overrides β.
    kv_link = scn.net if scn.node_cards > 0 else scn.link
    kv_tier = "net" if scn.node_cards > 0 else "link"
    beta = (pd.kv_GBps or kv_link.GBps) * 1e9
    alpha = kv_link.alpha_us * 1e-6
    L = m.n_layers
    b_req = min(c_p, c_d) * beta

    def kv_time(ttft_p: float) -> tuple[float, float]:
        if pd.kv_layerwise:
            t = L * alpha + kv / b_req
            return t, max(alpha + kv / b_req / L, t - ttft_p * (L - 1) / L)
        t = alpha + kv / b_req
        return t, t

    t_kv, exposed = kv_time(pr.ttft)
    rp = pr.throughput if pr.fits else 0.0
    rd = dec.throughput if dec.fits else 0.0
    if not plain:      # requests/s per prefill replica over the prompt mix at the pool's batch pb
        t_req = sum(w * ppool.run("prefill", pb, Si, pi).ttft for w, Si, pi in pts) / pb
        rq_p = 1.0 / t_req if pr.fits and t_req > 0 else 0.0

    def rates(np_: int, nd_: int) -> dict:
        lp = (np_ // c_p) * rp / S if plain else (np_ // c_p) * rq_p
        ld = (nd_ // c_d) * rd / out
        lk = min(np_, nd_) * beta / kv if kv > 0 else math.inf
        lam = min(lp, ld, lk)
        bott = "prefill" if lam == lp else "decode" if lam == ld else "kv"
        util = {k: (lam / v if v > 0 and math.isfinite(v) else 0.0) for k, v in (("prefill", lp), ("decode", ld), ("kv", lk))}
        return {"req_s": lam, "goodput_tok_s": lam * out, "goodput_per_card": lam * out / (np_ + nd_),
                "bottleneck": bott, "cap_req_s": {"prefill": lp, "decode": ld, "kv": lk}, "util": util}

    main = rates(n_p, n_d)
    ttft = pr.ttft + exposed
    total = n_p + n_d
    splits = []
    for k in range(1, total // c_p + 1):
        np_ = k * c_p
        nd_ = total - np_
        if nd_ < c_d or nd_ % c_d:
            continue
        x = rates(np_, nd_)
        splits.append({"prefill_cards": np_, "decode_cards": nd_, "goodput_per_card": x["goodput_per_card"],
                       "bottleneck": x["bottleneck"]})
    best = max(splits, key=lambda d: d["goodput_per_card"]) if splits else None
    # colocated on the same cards with the scenario's layout
    g = goodput(dec)
    reps = total // c_d
    cpool = _Pool(dscn)
    if not plain and g.decode_tok_s > 0 and g.prefill_tok_s > 0:     # colocated: same mix, per-request prefill time
        t_c = sum(w * cpool.run("prefill", g.prefill_batch, Si, pi).ttft for w, Si, pi in pts) / g.prefill_batch
        rd_c = g.decode_tok_s
        gg = 1.0 / (1.0 / rd_c + t_c / out)
        g = Goodput(rd_c, g.prefill_tok_s, g.prefill_batch, g.ttft_ms, gg, gg / scn.layout.cards,
                    (1.0 / rd_c) / (1.0 / rd_c + t_c / out), g.ttft_ok)
    co = {"cards": total, "replicas": reps, "layout": scn.layout.label, "ttft_ms": g.ttft_ms,
          "tpot_ms": dec.tpot * 1e3, "tpot_eff_ms": dec.tpot * 1e3 / g.decode_share if g.decode_share > 0 else math.inf,
          "decode_share": g.decode_share, "goodput_tok_s": reps * g.goodput_tok_s,
          "goodput_per_card": reps * g.goodput_tok_s / total if total else 0.0, "ttft_ok": g.ttft_ok,
          "tpot_eff_ok": dec.tpot * 1e3 / g.decode_share <= sv.tpot_slo_ms if g.decode_share > 0 else False,
          "idle_cards": total - reps * c_d}
    warnings = []
    if not p_ok:
        warnings.append("prefill 池：单条请求的 prefill 也超过 TTFT SLO 或放不下（按 batch 1 计）")
    if not dec.fits:
        warnings.append("decode 池：容量不足")
    if ttft * 1e3 > sv.ttft_slo_ms:
        warnings.append(f"PD 的 TTFT {ttft * 1e3:.0f} ms（prefill {pr.ttft * 1e3:.0f} + KV 传输 {exposed * 1e3:.1f}）超过 SLO "
                        f"{sv.ttft_slo_ms:g} ms")
    if co["idle_cards"]:
        warnings.append(f"合并对照：{total} 卡不是解码布局 {c_d} 卡的整数倍，{co['idle_cards']} 张卡闲置")
    q = ctx = None
    if queue or pd.search_layouts:
        ctx = {"ppool": ppool, "dpool": dpool, "cpool": cpool, "r_p": r_p, "r_d": r_d, "n_p": n_p,
               "n_d": n_d, "r_c": reps, "out": out, "B": sv.batch, "tier": kv_tier, "beta": beta,
               "alpha": alpha, "pair": min(c_p, c_d), "layerwise": pd.kv_layerwise, "L": L, "C": pd.chunk_tokens,
               "shared": pd.kv_GBps is None, "len": lens, "pts": pts, "rep": (S_rep, p_rep), "kv_xfer": kv_xfer,
               "kv_new": kv_new, "out_cs2": lens.out_cs2}
        dpool.memo[("decode", sv.batch, None, 0)] = dec
        cpool.memo = dpool.memo                           # same layout → same evaluations
    if queue:
        q = queue_report(ctx, main["req_s"], pd, sv, energy)
    ls = None
    if pd.search_layouts:
        ls = _search_layouts(m, scn, pscn, dscn, total, (c_p, c_d), pts, plain, S, out, kv, beta, ctx,
                             queue, main["req_s"])
    return {
        "layout_search": ls,
        "queue": q,
        "lengths": {**lens.summary(), "prefix_hit": h, "prefix_on_decode": pd.prefix_on_decode,
                    "prefix_tokens_rep": p_rep, "decode_ctx": ctx_eff, "plain": plain},
        "prefill": {"cards": n_p, "replicas": r_p, "layout": pd.prefill_layout.label, "batch": pb,
                    "ttft_ms": pr.ttft * 1e3, "tok_s_replica": rp, "fits": pr.fits, "ttft_ok": p_ok, "bound": pr.bound},
        "decode": {"cards": n_d, "replicas": r_d, "layout": scn.layout.label, "batch": sv.batch,
                   "tpot_ms": dec.tpot * 1e3, "tok_s_replica": rd, "fits": dec.fits, "bound": dec.bound},
        "kv": {"bytes_per_req": kv, "GBps_req": b_req / 1e9, "t_ms": t_kv * 1e3, "exposed_ms": exposed * 1e3,
               "layerwise": pd.kv_layerwise, "tier": kv_tier,
               "source": "pd.kv_GBps" if pd.kv_GBps else ("net.GBps（跨节点层）" if kv_tier == "net" else "link.GBps（节点内互联层）")},
        "ttft_ms": ttft * 1e3, "tpot_ms": dec.tpot * 1e3,
        "ttft_ok": ttft * 1e3 <= sv.ttft_slo_ms and p_ok, "tpot_ok": dec.tpot * 1e3 <= sv.tpot_slo_ms,
        **main, "cards": total, "splits": splits, "best_split": best, "coloc": co, "warnings": warnings,
        "basis": "稳态流体模型「假设」：无排队 / 到达波动、无连续批处理动态；KV 经"
                 + ("跨节点网络（两池在不同节点）" if kv_tier == "net" else "节点内 scale-up（两池同一 scale-up 域）") + "传输",
    }


SEARCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
SEARCH_CAP = 64          # candidate layouts per pool (smallest PP first) — keeps the opt-in search interactive
SEARCH_SLO_TOP = 4       # best fluid layout pairs whose SLO goodput (queueing model) is computed …
SEARCH_SLO_MAX = 24      # … plus the best pair of every decode / prefill layout, up to this many


def _search_layouts(m, scn, pscn, dscn, total, cur, pts, plain, S, out, kv, beta, ctx, queue, lam0) -> dict:
    """Pool layouts as design variables (0.52, opt-in ``pd.search_layouts``): every prefill layout × decode layout
    (replica sizes 1, 2, 4, …, 64 ≤ total, plus the current ones) × card split of the same total.  Fluid goodput per
    card for all; the SLO goodput (queueing model, p90 TTFT / TPOT) for the best fluid pairs.  The decode batch per
    replica stays ``serving.batch``; the prefill batch is best_prefill per layout (TTFT SLO)."""
    pscn, dscn = pscn.replace("pd", PDConfig()), dscn.replace("pd", PDConfig())   # pool card checks off
    sizes = sorted({c for c in SEARCH_SIZES if c <= total} | set(cur))
    lays = [lay for c in sizes for lay in enumerate_layouts(c, m.n_layers, m.is_moe)]
    lays.sort(key=lambda l: (l.pp, l.cards, l.tp))
    truncated = len(lays) > SEARCH_CAP
    lays = lays[:SEARCH_CAP] + [l for l in (pscn.layout, dscn.layout) if l not in lays[:SEARCH_CAP]]
    P, D = [], []
    for lay in lays:
        ps = pscn.replace("layout", lay)
        b_, r_ = best_prefill(ps)
        if r_ is not None and r_.fits:
            pool = _Pool(ps)
            if plain:
                rq = r_.throughput / S
            else:
                rq = b_ / sum(w * pool.run("prefill", b_, Si, pi).ttft for w, Si, pi in pts)
            P.append((lay, b_, rq, pool, r_.ttft))
        ds = dscn.replace("layout", lay)
        rd_ = evaluate(ds)
        if rd_.fits and rd_.throughput > 0:
            D.append((lay, rd_.throughput / out, rd_))
    pairs = []
    for lp_, bp, rqp, pool, ttft_p in P:
        for ld_, rqd, rdx in D:
            cp, cd = lp_.cards, ld_.cards
            best = None
            for k in range(1, total // cp + 1):
                np_, nd_ = k * cp, total - k * cp
                if nd_ < cd or nd_ % cd:
                    continue
                caps = {"prefill": (np_ // cp) * rqp, "decode": (nd_ // cd) * rqd,
                        "kv": min(np_, nd_) * beta / kv if kv > 0 else math.inf}
                lam = min(caps.values())
                row = {"prefill_layout": lp_.label, "decode_layout": ld_.label, "prefill_cards": np_,
                       "decode_cards": nd_, "prefill_batch": bp, "req_s": lam, "goodput_per_card": lam * out / total,
                       "bottleneck": min(caps, key=caps.get), "prefill_ttft_ms": ttft_p * 1e3,
                       "decode_tpot_ms": rdx.tpot * 1e3, "_k": (lp_, ld_, np_, nd_, pool, rdx),
                       "_tie": (-caps["prefill"], ttft_p, rdx.tpot)}
                if best is None or row["req_s"] > best["req_s"]:
                    best = row
            if best is not None:
                pairs.append(best)
    # fluid goodput first; ties (the non-bottleneck pool) → more prefill headroom, lower TTFT, lower TPOT first
    pairs.sort(key=lambda r: (-round(r["req_s"], 9), *r["_tie"]))
    cur_key = (pscn.layout.label, dscn.layout.label)
    if queue and ctx is not None:
        sv = scn.serving
        # SLO candidates: the best fluid rows, the best row of every decode layout (sets TPOT) and of every prefill
        # layout (sets TTFT), and the current layouts — fluid order alone would hide TPOT / TTFT differences
        todo, seen = [], set()

        def take(r):
            key = (r["prefill_layout"], r["decode_layout"])
            if key not in seen and len(todo) < SEARCH_SLO_MAX:
                seen.add(key)
                todo.append(r)
        for r in pairs[:SEARCH_SLO_TOP]:
            take(r)
        for side in ("decode_layout", "prefill_layout"):
            firsts: dict = {}
            for r in pairs:
                firsts.setdefault(r[side], r)
            for r in firsts.values():
                take(r)
        for r in pairs:
            if (r["prefill_layout"], r["decode_layout"]) == cur_key:
                seen.discard(cur_key)
                todo = [t for t in todo if (t["prefill_layout"], t["decode_layout"]) != cur_key]
                todo.append(r)
                seen.add(cur_key)
        for r in todo:
            lp_, ld_, np_, nd_, pool, rdx = r["_k"]
            dpool = _Pool(dscn.replace("layout", ld_))
            dpool.memo[("decode", sv.batch, None, 0)] = rdx
            cx = {**ctx, "ppool": pool, "dpool": dpool, "n_p": np_, "n_d": nd_, "r_p": np_ // lp_.cards,
                  "r_d": nd_ // ld_.cards, "pair": min(lp_.cards, ld_.cards)}
            rate = _slo_rate(_pd_mode, cx, max(r["req_s"], lam0, 1e-9), sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3)
            r["slo_rate_rps"] = rate
            r["slo_goodput_per_card"] = rate * out / total
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in pairs]
    slo = [r for r in rows if "slo_goodput_per_card" in r]
    return {"cards": total, "candidates": len(lays), "truncated": truncated, "pairs": len(rows),
            "rows": rows[:12], "best_fluid": rows[0] if rows else None,
            "best_slo": max(slo, key=lambda r: r["slo_goodput_per_card"]) if slo else None,
            "current_layouts": next((r for r in rows if (r["prefill_layout"], r["decode_layout"]) == cur_key), None),
            "basis": "池布局搜索「假设」：两池各自的布局（每副本卡数 1/2/4/…/64 与当前布局）× 同总卡数的切分；"
                     "decode 每副本 batch 不变，prefill batch 按 TTFT SLO 取；流体 goodput 全部排序，"
                     f"前 {SEARCH_SLO_TOP} 对、每种 decode / prefill 布局的最佳一对与当前布局另算排队模型下的 SLO goodput（至多 {SEARCH_SLO_MAX} 对）"}
