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
from .memplan import stage_storage
from .scenario import Scenario
from .serving import best_prefill, goodput


def kv_bytes_per_request(model, prompt: int) -> float:
    """Logical KV + indexer keys + recurrent state of the whole model after a ``prompt``-token prefill."""
    st = stage_storage(model, 0, model.n_layers, True, True, Shard(), prompt, 0)
    return st.kv_per_seq + st.idx_per_seq + st.state_per_seq


def _pool(cards: int, per: int) -> tuple[int, int]:
    n = cards or per
    return n, n // per


def disagg_report(scn: Scenario, decode: Result | None = None) -> dict:
    """PD metrics for an LLM scenario with ``scn.pd.enabled`` (decode pool = scn.layout at scn.serving.batch)."""
    m = get_model(scn.model)
    if not m.kv_cache:
        raise ValueError("PD 分离只适用于 LLM / VLM（视频 / 蛋白质模型没有 prefill / decode 两阶段）")
    pd, sv = scn.pd, scn.serving
    dscn = scn.replace("serving.phase", "decode")
    dec = decode if (decode is not None and decode.scenario.serving.phase == "decode") else evaluate(dscn)
    pscn = dscn.replace("layout", pd.prefill_layout)
    pb, pr = best_prefill(pscn)
    p_ok = pr is not None
    if pr is None:
        pb, pr = 1, evaluate(pscn.replace("serving.phase", "prefill").replace("serving.batch", 1))
    c_p, c_d = pd.prefill_layout.cards, scn.layout.cards
    n_p, r_p = _pool(pd.prefill_cards, c_p)
    n_d, r_d = _pool(pd.decode_cards, c_d)
    S, out = sv.prompt, sv.out_len
    kv = kv_bytes_per_request(m, S)
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

    def rates(np_: int, nd_: int) -> dict:
        lp = (np_ // c_p) * rp / S
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
    return {
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
