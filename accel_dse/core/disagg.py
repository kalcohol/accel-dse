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

0.53: with ``pd.prefix_len`` > 0 (and ``pd.prefix_hit`` = 0) the hit rate is emergent from an LRU / Zipf / Che model
(core/prefixcache.py) driven by free DRAM or ``pd.prefix_cache_GB``; an explicit ``pd.prefix_hit`` still overrides.
``pd.prefill_chip`` / ``pd.prefill_mem_id`` let the prefill pool use another chip / memory (colocated stays on the
decode scenario).  ``pd.search_decode_batch`` (with ``search_layouts``) also picks each decode layout's batch in
{B/2, B, 2B, 4B} under the TPOT SLO.  Defaults (no prefix_len, same chip/mem, no decode-batch search) take the 0.52
path unchanged.
"""

from __future__ import annotations

import contextvars
import math
from dataclasses import replace

from .catalog import get_model
from .evaluate import Result, evaluate
from .ir import Shard
from .parallel import MAX_REPLICA_CARDS, enumerate_layouts
from .memplan import cache_new_bytes, stage_storage
from .scenario import PDConfig, Scenario
from .energy import EnergyTable
from .lengths import lengths_of
from .vision import expand_images
from .pdqueue import _Pool, _pd_mode, _slo_rate, prefix_tokens, queue_report
from .prefixcache import (both_hit, capacity, hit_of, per_card_bytes, pool_hits, tree_both_tail, tree_cum_tokens,
                          tree_depth_law, tree_depth_tail, tree_groups, tree_resident, zipf_groups)
from .serving import Goodput, best_prefill, goodput

# 0.63: pdsim.capture_ctx collects the live queueing ctx here (per context; was a monkey-patch of queue_report, which
# leaked across threads of the HTTP server)
CTX_SINK: contextvars.ContextVar = contextvars.ContextVar("accel_dse_pd_ctx_sink", default=None)


def kv_bytes_per_request(model, prompt: int) -> float:
    """Logical KV + indexer keys + recurrent state of the whole model after a ``prompt``-token prefill."""
    st = stage_storage(model, 0, model.n_layers, True, True, Shard(), prompt, 0)
    return st.kv_per_seq + st.idx_per_seq + st.state_per_seq


def kv_token_capacity(model, result, cap_GB: float | None) -> dict:
    """KV tokens one replica can hold (0.55, pd.kv_policy): ``pd.kv_capacity_GB`` per replica, else the DRAM left after
    weights + runtime reserve (the evaluation's own active-batch KV is given back), per pipeline stage, ÷ the per-token
    KV slope of that stage (KV + indexer; recurrent state is per sequence, not per token, and is ignored 「假设」)."""
    lay = result.scenario.layout
    t1, t2 = 1024, 2048
    rg = [s.layers for s in result.stages]
    b1, b2 = per_card_bytes(model, lay, t1, rg), per_card_bytes(model, lay, t2, rg)
    slope = [(y - x) / (t2 - t1) for x, y in zip(b1, b2)]
    cps = lay.cards // lay.pp
    per_rep = sum(sl * cps / lay.dp for sl in slope)          # bytes per token over one replica
    if per_rep <= 0:
        return {"tokens": None, "source": "no_kv", "bytes_per_token": 0.0}
    if cap_GB is not None:
        return {"tokens": int(cap_GB * 1e9 // per_rep), "source": "pd.kv_capacity_GB", "capacity_GB": cap_GB,
                "bytes_per_token": per_rep}
    free = [max(0.0, st.mem.dram_cap - (st.mem.dram_need - st.mem.kv_total - st.mem.state_total))
            for st in result.stages]
    toks = [math.floor(f / sl) for f, sl in zip(free, slope) if sl > 0]
    K = min(toks) * lay.dp if toks else 0
    return {"tokens": int(K), "source": "derived_free_dram", "capacity_GB": K * per_rep / 1e9,
            "bytes_per_token": per_rep}


def _pool(cards: int, per: int) -> tuple[int, int]:
    n = cards or per
    return n, n // per


def disagg_report(scn: Scenario, decode: Result | None = None, energy: EnergyTable | None = None,
                  queue: bool = True) -> dict:
    """PD metrics for an LLM scenario with ``scn.pd.enabled`` (decode pool = scn.layout at scn.serving.batch)."""
    m = get_model(scn.model)
    if not m.kv_cache:
        raise ValueError("PD 分离只适用于 LLM / VLM（视频 / 蛋白质模型没有 prefill / decode 两阶段）")
    scn = expand_images(scn, m)      # 0.62: VLM image tokens are prompt tokens of every request (idempotent)
    pd, sv = scn.pd, scn.serving
    lens, h = lengths_of(pd, sv), pd.prefix_hit
    tree = pd.prefix_tree                            # 0.58 radix / partial prefix matching
    cap_mode = (pd.prefix_len > 0 or bool(tree)) and not h   # 0.53 capacity model (explicit prefix_hit overrides)
    plain = lens.trivial and not h and sv.prefix_cached == 0 and not cap_mode
    S_rep = max(1, round(lens.mean_S))
    p_rep = prefix_tokens(S_rep, h)
    ctx_eff = sv.ctx if lens.trivial else max(1, round(sv.ctx * lens.ctx_ratio))
    c_p, c_d = pd.prefill_layout.cards, scn.layout.cards
    n_p, r_p = _pool(pd.prefill_cards, c_p)
    n_d, r_d = _pool(pd.decode_cards, c_d)
    total = n_p + n_d
    reps = total // c_d

    def scns(pp_rep: int, pc_rep: int) -> tuple[Scenario, Scenario]:
        d = scn.replace("serving.phase", "decode")
        if not plain:
            d = d.replace("serving.prefix_cached", 0).replace("serving.prompt", S_rep) \
                .replace("serving.prefix_cached", pc_rep).replace("serving.ctx", ctx_eff)
        # 0.61.4: drop the PD block before swapping in the prefill layout — pd.decode_cards is validated against the
        # scenario's layout, so a prefill layout whose cards do not divide decode_cards (prefill TP4, decode 2 × TP1)
        # raised "decode_cards must be a multiple of the (decode) layout's 4 cards"; the pools' evaluations never
        # read the PD block
        p = d.colocated().replace("layout", pd.prefill_layout).replace("serving.prefix_cached", pp_rep if not plain else 0)
        if pd.prefill_chip is not None:              # 0.53 heterogeneous pools
            p = p.replace("chip", pd.prefill_chip)
        if pd.prefill_mem_id is not None:
            p = p.replace("mem_id", pd.prefill_mem_id)
        return d, p

    def prefill_of(ps: Scenario):
        b_, r_ = best_prefill(ps)
        if r_ is None:
            return 1, evaluate(ps.replace("serving.phase", "prefill").replace("serving.batch", 1)), False
        return b_, r_, True

    pcache = None
    if cap_mode and tree:
        # 0.58 radix tree: one token-capacity LRU per replica shared by all levels; Che per node (core/prefixcache)
        dscn, pscn = scns(0, 0)
        dec0 = evaluate(dscn)
        tg, cum = tree_groups(tree), tree_cum_tokens(tree)
        Lp = cum[-1]

        def tree_pool(cap, reps_):
            C = cap["K_float"] * Lp * (reps_ if pd.prefix_affinity else 1)
            hh = tree_resident(tree, C, tg)
            tail = tree_depth_tail(tg, hh)
            return hh, tail, tree_depth_law(tail), C

        def tok_of(law, S_):
            return sum(q * min(c, S_ - 1) for q, c in zip(law, cum))
        cap_d = capacity(m, dec0, Lp, pd.prefix_cache_GB)
        hd, tail_d, law_d, C_d = tree_pool(cap_d, r_d)
        hc, tail_c, law_c, C_c = tree_pool(cap_d, reps)
        pb, pr, p_ok = prefill_of(pscn)
        for _ in range(3):
            cap_p = capacity(m, pr, Lp, pd.prefix_cache_GB)
            hp, tail_p, law_p, C_p = tree_pool(cap_p, r_p)
            pp_rep = int(tok_of(law_p, S_rep))
            nb, nr, nok = prefill_of(scns(pp_rep, 0)[1])
            same = nb == pb
            pb, pr, p_ok = nb, nr, nok
            if same:
                break
        pc_rep = int(tok_of(law_c, S_rep))
        dscn, pscn = scns(pp_rep, pc_rep)
        dec = replace(dec0, scenario=dscn)
        # decode side holds min(prefill depth, decode depth): P(both hold the level-k node) per level
        both = tree_both_tail(tg, hp, hd)
        e_p = sum(L * t for (L, _, _), t in zip(tree, tail_p))
        q_dec = (sum(L * t for (L, _, _), t in zip(tree, both)) / e_p if e_p > 0 else 0.0) if pd.prefix_on_decode else 0.0
        p_rep = pp_rep
        H_p = sum(q * c for q, c in zip(law_p, cum)) / Lp      # token hit ratio
        H_c = sum(q * c for q, c in zip(law_c, cum)) / Lp

        def lvl(law, tail, C):
            return {"token_hit": sum(q * c for q, c in zip(law, cum)) / Lp, "level_hit": tail, "depth_law": law,
                    "capacity_tokens": C}
        pcache = {"policy": "LRU radix 树（Che 逐节点近似）", "tree": [list(x) for x in tree], "prefix_len": Lp,
                  "affinity": pd.prefix_affinity,
                  "prefill": {**cap_p, "replicas": r_p, "hit": H_p, **lvl(law_p, tail_p, C_p)},
                  "decode": {**cap_d, "replicas": r_d, "hit": sum(q * c for q, c in zip(law_d, cum)) / Lp,
                             "holds_given_prefill_hit": q_dec, **lvl(law_d, tail_d, C_d)},
                  "coloc": {**cap_d, "replicas": reps, "hit": H_c, **lvl(law_c, tail_c, C_c)},
                  "basis": "radix 前缀树「假设」：每层（token 数、分支数、Zipf）独立选子节点（IRM）；节点级 LRU，容量按 token "
                           "计、各层共享；部分命中节省匹配到的层；命中按 Che 逐节点近似（树性质：子节点驻留 ⇒ 父节点驻留）；"
                           + ("前缀感知路由：总容量近似（DES 按根节点分配副本）" if pd.prefix_affinity else "随机路由：每个副本各自缓存")}
    elif cap_mode:
        dscn, pscn = scns(0, 0)
        dec0 = evaluate(dscn)                        # decode results do not depend on the cached prefix
        groups = zipf_groups(pd.prefix_count, pd.prefix_zipf)
        Lp = pd.prefix_len
        cap_d = capacity(m, dec0, Lp, pd.prefix_cache_GB)
        hd = pool_hits(groups, cap_d["K"], r_d, pd.prefix_affinity)
        hc = pool_hits(groups, cap_d["K"], reps, pd.prefix_affinity)
        H_c = hit_of(groups, hc)
        pb, pr, p_ok = prefill_of(pscn)
        for _ in range(3):                           # the prefill batch sets the free memory, the hit rate the batch
            cap_p = capacity(m, pr, Lp, pd.prefix_cache_GB)
            hp = pool_hits(groups, cap_p["K"], r_p, pd.prefix_affinity)
            H_p = hit_of(groups, hp)
            pp_rep = int(H_p * min(Lp, S_rep - 1))
            nb, nr, nok = prefill_of(scns(pp_rep, 0)[1])
            same = nb == pb
            pb, pr, p_ok = nb, nr, nok
            if same:
                break
        pc_rep = int(H_c * min(Lp, S_rep - 1))
        dscn, pscn = scns(pp_rep, pc_rep)
        dec = replace(dec0, scenario=dscn)
        H_both = both_hit(groups, hp, hd)
        q_dec = (H_both / H_p if H_p > 0 else 0.0) if pd.prefix_on_decode else 0.0
        p_rep = pp_rep
        pcache = {"policy": "LRU（Che 近似）", "prefix_len": Lp, "prefix_count": pd.prefix_count,
                  "zipf": pd.prefix_zipf, "affinity": pd.prefix_affinity,
                  "prefill": {**cap_p, "replicas": r_p, "hit": H_p},
                  "decode": {**cap_d, "replicas": r_d, "hit": hit_of(groups, hd), "holds_given_prefill_hit": q_dec},
                  "coloc": {**cap_d, "replicas": reps, "hit": H_c},
                  "basis": "前缀缓存容量模型「假设」：N 个共享前缀、Zipf 流行度、独立请求（IRM）；整前缀 LRU，命中率按 Che 近似；"
                           "容量 = 权重 + 活跃 batch KV 之后剩余的 DRAM（或 pd.prefix_cache_GB）；"
                           + ("前缀感知路由：副本分摊前缀（总容量）" if pd.prefix_affinity else "随机路由：每个副本各自缓存全体前缀")}
    else:
        dscn, pscn = scns(p_rep, p_rep)
        dec = decode if (plain and decode is not None and decode.scenario.serving.phase == "decode") else evaluate(dscn)
        pb, pr, p_ok = prefill_of(pscn)
    ppool, dpool = _Pool(pscn), _Pool(dscn)
    S, out = (sv.prompt, sv.out_len) if lens.trivial else (S_rep, lens.mean_out)

    def mix_pts(H: float) -> list:
        if tree:                                     # 0.58: matched depth law → one class per depth
            law = H
            res = []
            for w, Si in lens.prompts():
                acc: dict[int, float] = {}
                for q, c in zip(law, cum):
                    if q > 0:
                        pi = min(c, Si - 1)
                        acc[pi] = acc.get(pi, 0.0) + w * q
                res += [(x, Si, pi) for pi, x in sorted(acc.items())]
            return res
        res = []
        for w, Si in lens.prompts():
            pi = min(pd.prefix_len, Si - 1)
            if pi <= 0 or H <= 0:
                res.append((w, Si, 0))
            elif H >= 1:
                res.append((w, Si, pi))
            else:
                res += [(w * H, Si, pi), (w * (1 - H), Si, 0)]
        return res

    if cap_mode:
        pts, pts_c = (mix_pts(law_p), mix_pts(law_c)) if tree else (mix_pts(H_p), mix_pts(H_c))
    else:
        pts = pts_c = [(w, Si, prefix_tokens(Si, h)) for w, Si in lens.prompts()]
        q_dec = 1.0 if pd.prefix_on_decode else 0.0
    kv_full = {(Si, pi): kv_bytes_per_request(m, Si) for _, Si, pi in pts + pts_c}
    kv_full.setdefault((S_rep, p_rep), kv_bytes_per_request(m, S_rep))
    # KV / state the decode side lacks given the cached prefix (0.62: per layer kind -- was (S − p) / S of the total)
    kv_new = {k: (cache_new_bytes(m, k[0], k[1]) if k[1] else v) for k, v in kv_full.items()}
    # 「假设」 the decode pool holds the prefix (0.53: with probability q_dec given a prefill hit)
    kv_xfer = {k: kv_full[k] - q_dec * (kv_full[k] - kv_new[k]) if q_dec != 1.0 else kv_new[k] for k in kv_full}
    kv = kv_bytes_per_request(m, S) if plain else sum(w * kv_xfer[(Si, pi)] for w, Si, pi in pts)
    # KV hand-off tier (「假设」): the two pools sit on different nodes when the cross-node tier is modelled
    # (node_cards > 0) → network link; otherwise one scale-up domain → the in-node link.  pd.kv_GBps overrides β.
    kv_link = scn.net if scn.node_cards > 0 else scn.link
    kv_tier = "net" if scn.node_cards > 0 else "link"
    beta = (pd.kv_GBps or kv_link.GBps) * 1e9
    alpha = kv_link.alpha_us * 1e-6
    kv_fab = None
    if scn.fabric.enabled:      # 0.59 「假设」: oversubscribed leaf uplinks + NICs shared with the pools' collectives
        from .fabric import _hop, kv_factor, kv_hop_extra
        fac = kv_factor(scn.fabric, scn.node_cards, total) if kv_tier == "net" else 1.0
        bn = kv_link.GBps * 1e9

        def _u(r) -> float:      # busy share of the pool's own collectives on this tier (heaviest stage)
            return max(((st.time.net_bytes if kv_tier == "net" else st.time.link_bytes - st.time.d2d_bytes
                         - st.time.net_bytes) / bn / st.time.total if st.time.total > 0 else 0.0) for st in r.stages)
        u_c = min(0.95, max(_u(pr), _u(dec))) if scn.fabric.contention else 0.0
        beta_raw = beta
        beta = beta / fac * (1.0 - u_c)
        alpha += _hop(scn.fabric, kv_tier)
        if kv_tier == "net":    # 0.61: hop_spine / hop_core of the pairs that leave the leaf / pod
            alpha += kv_hop_extra(scn.fabric, scn.node_cards, total)
        kv_fab = {"leaf_factor": fac, "u_coll": u_c, "GBps_raw": beta_raw / 1e9, "GBps_eff": beta / 1e9}
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

    kv_fb = None
    if kv_fab is not None and scn.fabric.kv_feedback and kv_tier == "net" and kv > 0 and dec.fits:
        # 0.60 「假设」: the KV stream takes u_kv = λ·kv / (min(n_p, n_d)·β) of the decode pool's NICs, so the decode
        # pool's network collectives run on (1 − u_kv)·net.GBps; λ falls with the slower decode → fixed point
        tpot0, u, net0 = dec.tpot, 0.0, dscn.net.GBps
        for _ in range(6):
            lam_ = rates(n_p, n_d)["req_s"]
            u_new = min(0.95, lam_ * kv / (min(n_p, n_d) * kv_fab["GBps_raw"] * 1e9))
            if abs(u_new - u) < 1e-4:
                break
            u = u_new
            dscn = dscn.replace("net.GBps", net0 * (1.0 - u))
            dec = evaluate(dscn)
            rd = dec.throughput if dec.fits else 0.0
        dpool = _Pool(dscn)
        kv_fb = {"u_kv": u, "tpot_ms_before": tpot0 * 1e3, "tpot_ms_after": dec.tpot * 1e3,
                 "net_GBps_decode": net0 * (1.0 - u)}
    main = rates(n_p, n_d)
    ttft = pr.ttft + exposed
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
    cpool = _Pool(dscn)
    if not plain and g.decode_tok_s > 0 and g.prefill_tok_s > 0:     # colocated: same mix, per-request prefill time
        t_c = sum(w * cpool.run("prefill", g.prefill_batch, Si, pi).ttft for w, Si, pi in pts_c) / g.prefill_batch
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
    if pd.prefix_len and h:
        warnings.append(f"pd.prefix_hit = {h:g} 显式给定：覆盖前缀缓存容量模型（prefix_len = {pd.prefix_len} 未用于命中率）")
    if pcache and pcache["prefill"]["K"] == 0:
        warnings.append("前缀缓存：prefill 池每副本放不下一个前缀（剩余 DRAM 不足或容量为 0）→ 命中率 0")
    if co["idle_cards"]:
        warnings.append(f"合并对照：{total} 卡不是解码布局 {c_d} 卡的整数倍，{co['idle_cards']} 张卡闲置")
    q = ctx = None
    if queue or pd.search_layouts:
        ctx = {"ppool": ppool, "dpool": dpool, "cpool": cpool, "r_p": r_p, "r_d": r_d, "n_p": n_p,
               "n_d": n_d, "r_c": reps, "out": out, "B": sv.batch, "tier": kv_tier, "beta": beta,
               "alpha": alpha, "pair": min(c_p, c_d), "layerwise": pd.kv_layerwise, "L": L, "C": pd.chunk_tokens,
               "shared": pd.kv_GBps is None, "len": lens, "pts": pts, "rep": (S_rep, p_rep), "kv_xfer": kv_xfer,
               "kv_new": kv_new, "kv_full": kv_full, "out_cs2": lens.out_cs2, "pts_c": pts_c,
               **({"prefix_hits": (H_p, H_c)} if cap_mode else {})}
        if pd.kv_policy != "off":       # 0.55: decode KV capacity (decode pool and colocated replicas share the layout)
            ctx["kv_policy"] = pd.kv_policy
            ctx["kv_cap"] = kv_token_capacity(m, dec, pd.kv_capacity_GB)
            ctx["kv_admit"] = pd.kv_admit
            if pd.kv_policy == "swap":      # host link per replica = per card × cards per replica 「假设」 (one link each)
                gbps = pd.swap_GBps if pd.swap_GBps is not None else scn.workload.host_GBps
                ctx["swap"] = {"GBps_card": gbps, "source": "pd.swap_GBps" if pd.swap_GBps is not None
                               else "workload.host_GBps", "Bps_replica": gbps * 1e9 * scn.layout.cards}
        dpool.memo[("decode", sv.batch, None, 0)] = dec
        cpool.memo = dpool.memo                           # same layout → same evaluations
    if queue:
        sink = CTX_SINK.get()
        if sink is not None:
            sink["ctx"] = ctx
        q = queue_report(ctx, main["req_s"], pd, sv, energy)
        if q and "modes" in q and pd.simulate:
            from . import pdsim as _ps
            sim_modes = {}
            pc = pcache
            lru = (dict(prefix_tree=tree, affinity=pd.prefix_affinity) if tree else
                   dict(prefix_len=pd.prefix_len, prefix_n=pd.prefix_count, prefix_alpha=pd.prefix_zipf,
                        affinity=pd.prefix_affinity)) if pc else {}
            for mode, ana in q["modes"].items():
                if not ana.get("stable"):
                    continue
                kw = dict(lru)
                if pc:
                    ck = "capacity_tokens" if tree else "K"
                    kw["prefix_K"] = pc["prefill"][ck] if mode == "pd" else pc["coloc"][ck]
                    if mode == "pd":
                        kw["prefix_K_dec"] = pc["decode"][ck]
                s = _ps.simulate(ctx, q["lambda_rps"], mode, n_req=1500, warmup=400, seed=1, **kw)
                err = {name: _ps.rel_err(s[k][qk], _ps._ana_value(ana, name)) for name, k, qk in _ps.METRICS}
                err = {kk: (v if math.isfinite(v) else None) for kk, v in err.items()}
                sim_modes[mode] = {"ttft_ms": s["ttft_ms"], "tpot_ms": s["tpot_ms"], "itl_max_ms": s["itl_max_ms"],
                                   "prefix_hit": s.get("prefix_hit"), "prefix_hit_decode": s.get("prefix_hit_decode"),
                                   "n": s["n"], "complete": s["complete"], "err": err}
            q = {**q, "sim": {"modes": sim_modes, "n_req": 1500, "warmup": 400, "seed": 1},
                 "sim_basis": "request-level DES (core/pdsim) 「假设」; n=1500 after 400 warmup; "
                 "same per-step costs as the closed form; err = (analytic − DES) / DES"}
    ls = None
    if pd.search_layouts:
        ls = _search_layouts(m, scn, pscn, dscn, total, (c_p, c_d), pts, plain, S, out, kv, beta, ctx,
                             queue, main["req_s"])
    return {
        "layout_search": ls,
        "queue": q,
        "lengths": {**lens.summary(), "prefix_hit": h if not cap_mode else pcache["prefill"]["hit"],
                    "prefix_hit_source": "capacity" if cap_mode else "explicit" if h else "off",
                    "prefix_on_decode": pd.prefix_on_decode,
                    "prefix_tokens_rep": p_rep, "decode_ctx": ctx_eff, "plain": plain},
        "prefix_cache": pcache,
        "prefill": {"cards": n_p, "replicas": r_p, "layout": pd.prefill_layout.label, "batch": pb,
                    "chip": pscn.chip.name, "mem_id": pscn.mem_id, "hetero": pscn.chip != scn.chip or pscn.mem_id != scn.mem_id,
                    "ttft_ms": pr.ttft * 1e3, "tok_s_replica": rp, "fits": pr.fits, "ttft_ok": p_ok, "bound": pr.bound},
        "decode": {"cards": n_d, "replicas": r_d, "layout": scn.layout.label, "batch": sv.batch,
                   "chip": scn.chip.name, "mem_id": scn.mem_id,
                   "tpot_ms": dec.tpot * 1e3, "tok_s_replica": rd, "fits": dec.fits, "bound": dec.bound},
        "kv": {"bytes_per_req": kv, "GBps_req": b_req / 1e9, "t_ms": t_kv * 1e3, "exposed_ms": exposed * 1e3,
               "layerwise": pd.kv_layerwise, "tier": kv_tier,
               **({"fabric": {**kv_fab, "coll_slowdown": 1.0 / max(0.05, 1.0 - main["req_s"] * kv
                                                                 / (min(n_p, n_d) * kv_fab["GBps_raw"] * 1e9)),
                              **({"feedback": kv_fb} if kv_fb else {})}}
                  if kv_fab else {}),
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
DECODE_BATCH_MULT = (0.5, 1, 2, 4)   # pd.search_decode_batch: decode batch candidates × serving.batch (0.53)


def _search_layouts(m, scn, pscn, dscn, total, cur, pts, plain, S, out, kv, beta, ctx, queue, lam0) -> dict:
    """Pool layouts as design variables (0.52, opt-in ``pd.search_layouts``): every prefill layout × decode layout
    (replica sizes 1, 2, 4, …, 64 ≤ total, plus the current ones) × card split of the same total.  Fluid goodput per
    card for all; the SLO goodput (queueing model, p90 TTFT / TPOT) for the best fluid pairs.  The decode batch per
    replica stays ``serving.batch``; the prefill batch is best_prefill per layout (TTFT SLO)."""
    pscn, dscn = pscn.replace("pd", PDConfig()), dscn.replace("pd", PDConfig())   # pool card checks off
    big = total > SEARCH_SIZES[-1]          # 0.60: totals above 64 cards also try replicas of 128, 256, … ≤ total
    sizes = sorted({c for c in SEARCH_SIZES if c <= total} | set(cur)
                   | ({1 << i for i in range(7, MAX_REPLICA_CARDS.bit_length()) if (1 << i) <= total} if big else set()))
    lays = [lay for c in sizes for lay in enumerate_layouts(c, m.n_layers, m.is_moe)]
    lays.sort(key=lambda l: (l.pp, l.cards, l.tp))
    truncated = len(lays) > SEARCH_CAP
    if big and truncated:     # 0.60: equal quota per replica size (smallest PP / TP first within a size) so large
        q = -(-SEARCH_CAP // len(sizes))    # replicas are not crowded out by the many small ones
        lays = [l for c in sizes for l in [x for x in lays if x.cards == c][:q]][:SEARCH_CAP]
    lays = lays[:SEARCH_CAP] + [l for l in (pscn.layout, dscn.layout) if l not in lays[:SEARCH_CAP]]
    P, D = [], []
    B0, tpot_slo = scn.serving.batch, scn.serving.tpot_slo_ms
    search_b = scn.pd.search_decode_batch
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
        bd = B0
        if search_b:          # 0.53: largest-throughput batch in {B/2, B, 2B, 4B} meeting the TPOT SLO (fluid TPOT)
            for b_ in DECODE_BATCH_MULT:
                b_ = max(1, int(B0 * b_))
                if b_ == bd or b_ > max(4096, 64 * lay.cards):     # 0.60: cap scales with replica cards
                    continue
                x = rd_ if b_ == B0 else evaluate(ds.replace("serving.batch", b_))
                ok_x = x.fits and x.tpot * 1e3 <= tpot_slo
                ok_c = rd_.fits and rd_.tpot * 1e3 <= tpot_slo
                if ok_x and (not ok_c or x.throughput > rd_.throughput):
                    rd_, bd = x, b_
        if rd_.fits and rd_.throughput > 0:
            D.append((lay, rd_.throughput / out, rd_, bd))
    pairs = []
    for lp_, bp, rqp, pool, ttft_p in P:
        for ld_, rqd, rdx, bd in D:
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
                       "decode_cards": nd_, "prefill_batch": bp, "decode_batch": bd, "req_s": lam, "goodput_per_card": lam * out / total,
                       "bottleneck": min(caps, key=caps.get), "prefill_ttft_ms": ttft_p * 1e3,
                       "decode_tpot_ms": rdx.tpot * 1e3, "_k": (lp_, ld_, np_, nd_, pool, rdx, bd),
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
            lp_, ld_, np_, nd_, pool, rdx, bd = r["_k"]
            dpool = _Pool(dscn.replace("layout", ld_).replace("serving.batch", bd))
            dpool.memo[("decode", bd, None, 0)] = rdx
            cx = {**ctx, "ppool": pool, "dpool": dpool, "B": bd, "n_p": np_, "n_d": nd_, "r_p": np_ // lp_.cards,
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
            "decode_batch_searched": search_b,
            "basis": "池布局搜索「假设」：两池各自的布局（每副本卡数 1/2/4/…/64 与当前布局）× 同总卡数的切分；"
                     + ("decode 每副本 batch 在 B/2、B、2B、4B 中取满足 TPOT SLO（满批流体 TPOT）的最高吞吐，" if search_b else
                        "decode 每副本 batch 不变，") + "prefill batch 按 TTFT SLO 取；流体 goodput 全部排序，"
                     f"前 {SEARCH_SLO_TOP} 对、每种 decode / prefill 布局的最佳一对与当前布局另算排队模型下的 SLO goodput（至多 {SEARCH_SLO_MAX} 对）"}
