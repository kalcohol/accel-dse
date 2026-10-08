"""Queueing, continuous batching, chunked prefill, KV-link contention and energy for the PD report (0.51).

Everything here is analytic on top of evaluated steps (no event simulator); all of it is 「假设」 and labelled so.
Offered load λ (requests/s, Poisson) = ``pd.rate_rps`` or ``pd.load`` × the PD fluid capacity; the colocated
system on the same cards sees the same λ.

PD (prefill pool r_p replicas, decode pool r_d replicas; random split → Poisson λ/r per replica)
  prefill   M/D/1 per replica (DistServe's model for a prefill instance).  Static batch cap b ∈ {1, 2, 4, …, 64}:
            per-request service τ_b = TTFT(b) / b (the fluid rate at b), latency floor TTFT(b); the cap minimising
            mean TTFT among stable ones is used (b > 1 is the batch-server approximation of a dynamic batcher).
  KV        one transfer stream per prefill replica, M/D/1 with service KV / β_req'; β' = β·(1 − u_coll), u_coll =
            the busiest pool's own collective traffic on the KV tier (bytes / β / tick)
  decode    continuous batching with B = serving.batch slots per replica: the mean occupancy is the fixed point
            n̄ = λ_d · out_len · TPOT(⌈n̄⌉) (Little's law) — the batch actually running at this load, faster than the
            full-batch TPOT; occupancy spread ~ Poisson(n̄) → TPOT p90 / p99 = TPOT(Poisson⁻¹(q)); a free slot
            = M/D/c (c = B slots, each held out_len·TPOT(n̄)) via Erlang C × ½.  KV arriving at λ·KV / N_d per decode card
            stretches the decode step's KV-tier time by 1 / (1 − u_kv) (and the prefill pool's by λ·KV / N_p).
  TTFT_q    = w_prefill,q + TTFT(b) + w_kv,q + exposed KV transfer (quantiles added: an upper-side estimate)
Colocated, same cards (⌊N / c_d⌋ replicas of the scenario layout)
  prefill-first   vLLM's default without chunking: prefills (M/D/1, same batch-cap rule on the decode layout)
                  pre-empt decode iterations; decode gets 1 − ρ_p of the time → occupancy n̄ = λ_c·out·TPOT/(1 − ρ_p),
                  mean TPOT_eff = TPOT(n̄)/(1 − ρ_p); a token interval that meets a prefill stalls for TTFT(b)
                  ("generation stall", share ≈ (λ_c / b)·TPOT_eff); request-average TPOT quantile = TPOT(n̄_q) +
                  Poisson⁻¹_q((λ_c / b)·out·TPOT_eff)·TTFT(b) / out; TTFT adds the residual decode step
  chunked         Sarathi-Serve style: every iteration carries the running decodes plus up to C = ``pd.chunk_tokens``
                  prompt tokens.  Iteration time per stage = max(compute_d + f·compute_p, DRAM_d + f·(prefill DRAM
                  except weights) + prefix-KV re-read, SLC_d, link_d + f·link_p) + max(sync) with f = C / S
                  (weights read once for both — the point of piggybacking); prefill is FCFS at C tokens / iteration →
                  M/D/1 with τ = (S / C)·T_iter, TTFT = w + ⌈S / C⌉·T_iter; mean TPOT = ρ·T(n̄, C) + (1 − ρ)·T(n̄, 0);
                  request-average quantile: j ~ Poisson(ρ · out) of its iterations carry a chunk
TPOT quantiles are request-average TPOT (DistServe's SLO metric); ``itl_max`` = the longest single token gap.
SLO capacity: the largest λ whose p90 TTFT ≤ TTFT SLO and p90 TPOT ≤ TPOT SLO (DistServe-style SLO goodput), by
bisection, per mode; ÷ cards → requests/s/card and output tokens/s/card.  For PD the same is searched over every
prefill / decode card split of the same total (``pd_slo_best_split``; layouts stay the inputs).
Energy per output token at this load: prefill counts / prompt token × S + decode counts / token (at the running
batch n̄) × out, + KV bytes on its tier (PD), ÷ out; card·s = cards / λ / out (every provisioned card, busy or not).
Chunked prefill: the prompt's weight reads are shared with the decode iteration (dropped) and each chunk re-reads
the prefix KV (+ KV·(S − C)/(2C) DRAM bytes per request).  J only for the energy-table entries the user gave.
Not modelled: arrival burstiness beyond Poisson, heavy-tailed prompt / output lengths (all requests are S / out_len),
pre-emption and KV-cache eviction, prefix caching, central-queue load balancing (random split is pessimistic).
"""

from __future__ import annotations

import math

from .energy import ACTIONS, EnergyTable, _BITS, _UNIT_PJ, action_counts, scaleup_bytes
from .evaluate import Result, evaluate
from .queueing import mdc_wait, md1, poisson_quantile
from .scenario import Scenario

QS = (0.5, 0.9, 0.99)
B_CAPS = (1, 2, 4, 8, 16, 32, 64)


def _ms(d: dict) -> dict:
    return {k: (v * 1e3 if k in ("mean", "p50", "p90", "p99") else v) for k, v in d.items()}


def _tier_util(r: Result, tier: str, beta: float) -> float:
    """Busiest stage's own traffic on the KV tier: bytes / β / tick (per card)."""
    u = 0.0
    for st in r.stages:
        t = st.time
        b = t.net_bytes if tier == "net" else scaleup_bytes(t)
        if r.tick > 0:
            u = max(u, b / beta / r.tick)
    return min(u, 1.0)


def _contended(r: Result, tier: str, beta: float, u_kv: float) -> float:
    """tick'/tick when the KV tier carries a u_kv share of extra traffic (the tier's time × 1 / (1 − u_kv))."""
    if u_kv <= 0 or r.tick <= 0:
        return 1.0
    if u_kv >= 1:
        return math.inf
    worst = 0.0
    for st in r.stages:
        t = st.time
        b = t.net_bytes if tier == "net" else scaleup_bytes(t)
        link = t.t_link + b / beta * (1 / (1 - u_kv) - 1)
        worst = max(worst, max(t.t_compute, t.t_dram, t.t_slc, link) + t.t_sync)
    return max(1.0, worst / r.tick)


class _Pool:
    """Memoised evaluations of one layout at different batches / phases."""

    def __init__(self, scn: Scenario):
        self.scn = scn.replace("serving.phase", "decode")
        self.memo: dict = {}

    def run(self, phase: str, b: int) -> Result:
        k = (phase, b)
        if k not in self.memo:
            self.memo[k] = evaluate(self.scn.replace("serving.phase", phase).replace("serving.batch", b))
        return self.memo[k]


def _prefill_server(pool: _Pool, lam: float, scale: float = 1.0) -> dict | None:
    """Batch cap minimising mean TTFT (M/D/1, τ_b = TTFT(b)/b); None if no cap is stable."""
    best = None
    for b in B_CAPS:
        r = pool.run("prefill", b)
        if not r.fits:
            break
        lat = r.ttft * scale
        w = md1(lam, lat / b, QS)
        if not w["stable"]:
            continue
        if best is None or w["mean"] + lat < best["wait"]["mean"] + best["lat"]:
            best = {"b": b, "lat": lat, "wait": w, "r": r}
    return best


def _decode_fixed_point(pool: _Pool, lam: float, out: int, B: int, share: float = 1.0,
                        scale=lambda r: 1.0) -> dict | None:
    """Continuous batching: smallest k ≤ B with λ·(out / e)·step(k)/share ≤ k; None if even k = B does not keep up."""
    def need(k: int) -> tuple[float, Result, float]:
        r = pool.run("decode", k)
        st = r.step * scale(r) / share
        return lam * out / r.tokens_per_step * st, r, st

    nB, rB, _ = need(B)
    if not rB.fits or nB > B:
        return None
    lo, hi = 1, B
    while lo < hi:
        mid = (lo + hi) // 2
        if need(mid)[0] <= mid:
            hi = mid
        else:
            lo = mid + 1
    n, r, st = need(lo)
    return {"k": lo, "occupancy": n, "r": r, "tpot": st / r.tokens_per_step, "step": st}


def _tpot_at(pool: _Pool, k: int, share: float = 1.0, scale=lambda r: 1.0) -> float:
    r = pool.run("decode", max(1, k))
    return r.step * scale(r) / share / r.tokens_per_step


def _pd_mode(ctx: dict, lam: float) -> dict:
    pp, dp = ctx["ppool"], ctx["dpool"]
    r_p, r_d, n_p, n_d = ctx["r_p"], ctx["r_d"], ctx["n_p"], ctx["n_d"]
    S, out, B = ctx["S"], ctx["out"], ctx["B"]
    kv, tier, beta, alpha = ctx["kv"], ctx["tier"], ctx["beta"], ctx["alpha"]
    lam_p, lam_d = lam / r_p, lam / r_d
    shared = ctx.get("shared", True)      # KV rides the pools' own collective tier (no pd.kv_GBps override)
    u_kv_d = lam * kv / (n_d * beta) if kv > 0 and shared else 0.0
    u_kv_p = lam * kv / (n_p * beta) if kv > 0 and shared else 0.0
    sc_d = lambda r: _contended(r, tier, beta, u_kv_d)
    pre = _prefill_server(pp, lam_p, _contended(pp.run("prefill", 1), tier, beta, u_kv_p))
    dec = _decode_fixed_point(dp, lam_d, out, B, scale=sc_d)
    res = {"lambda_rps": lam, "stable": pre is not None and dec is not None and u_kv_d < 1 and u_kv_p < 1}
    if not res["stable"]:
        res["why"] = ("prefill 池" if pre is None else "decode 池" if dec is None else "KV 链路") + "在此负载下不稳定" \
            + ("（KV 与池内集合通信共用互联层，争用计入）" if max(u_kv_d, u_kv_p) > 0.3 else "")
        return res
    u_coll = max(_tier_util(pre["r"], tier, beta), _tier_util(dec["r"], tier, beta)) if shared else 0.0
    beta_av = beta * (1 - u_coll)
    b_req = ctx["pair"] * beta_av
    t_bw = kv / b_req if b_req > 0 else math.inf
    kvq = md1(lam_p, t_bw, QS)
    if ctx["layerwise"]:
        L = ctx["L"]
        t_kv = L * alpha + t_bw
        exposed = max(alpha + t_bw / L, t_kv - pre["lat"] * (L - 1) / L)
    else:
        exposed = alpha + t_bw
    if not kvq["stable"]:
        return {**res, "stable": False, "why": "KV 链路（扣除池内集合通信后）在此负载下不稳定"}
    ttft = {q: pre["wait"][q] + pre["lat"] + kvq[q] + exposed for q in ("mean", "p50", "p90", "p99")}
    k90 = min(B, poisson_quantile(dec["occupancy"], 0.9))
    slot = mdc_wait(lam_d, out * dec["tpot"], B, QS)
    tp90 = _tpot_at(dp, max(dec["k"], k90), scale=sc_d)
    k99 = min(B, poisson_quantile(dec["occupancy"], 0.99))
    return {**res, "ttft": ttft, "tpot_mean": dec["tpot"], "tpot_p90": tp90,
            "tpot_p99": _tpot_at(dp, max(dec["k"], k99), scale=sc_d),
            "itl_max": _tpot_at(dp, max(dec["k"], k99), scale=sc_d),
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"batch_cap": pre["b"], "ttft_b_ms": pre["lat"] * 1e3, "wait_ms": _ms(pre["wait"]),
                        "kv_slowdown": pre["lat"] / pre["r"].ttft},
            "kv": {"wait_ms": _ms(kvq), "exposed_ms": exposed * 1e3, "shared_tier": shared, "u_coll": u_coll,
                   "GBps_req_avail": b_req / 1e9,
                   "u_kv_decode": u_kv_d, "u_kv_prefill": u_kv_p},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B,
                       "kv_slowdown": sc_d(dec["r"]), "slot_wait_ms": _ms(slot), "running_p90": max(dec["k"], k90)},
            "_pre": pre, "_dec": dec}


def _coloc_prefill_first(ctx: dict, lam: float) -> dict:
    cp, r_c, S, out, B = ctx["cpool"], ctx["r_c"], ctx["S"], ctx["out"], ctx["B"]
    lam_c = lam / r_c
    pre = _prefill_server(cp, lam_c)
    res = {"lambda_rps": lam, "stable": pre is not None}
    if pre is None:
        return {**res, "why": "prefill 在此负载下不稳定"}
    share = 1 - pre["wait"]["rho"]
    dec = _decode_fixed_point(cp, lam_c, out, B, share=share)
    if dec is None or share <= 0:
        return {**res, "stable": False, "why": "decode 在此负载下不稳定（prefill 占用后剩余时间不够）"}
    tp = dec["r"].step / dec["r"].tokens_per_step
    resid = dec["r"].step
    ttft = {q: pre["wait"][q] + pre["lat"] + (resid / 2 if q == "mean" else resid if q in ("p90", "p99") else resid / 2)
            for q in ("mean", "p50", "p90", "p99")}
    stall_share = min(1.0, lam_c / pre["b"] * dec["tpot"])
    # request-average TPOT: pure decode steps at the running batch + the prefill stalls met during its lifetime
    # (prefill batches start at λ_c / b, Poisson → count ~ Poisson((λ_c / b) · lifetime))
    life = out * dec["tpot"]
    m_st = lam_c / pre["b"] * life
    k90 = min(B, poisson_quantile(dec["occupancy"], 0.9))
    tp90 = _tpot_at(cp, max(dec["k"], k90))
    req_q = {q: tp90 + poisson_quantile(m_st, q) * pre["lat"] / out for q in (0.9, 0.99)}
    slot = mdc_wait(lam_c, out * dec["tpot"], B, QS)
    return {**res, "ttft": ttft, "tpot_mean": dec["tpot"], "tpot_p90": req_q[0.9], "tpot_p99": req_q[0.99],
            "itl_max": tp90 + pre["lat"],
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"batch_cap": pre["b"], "ttft_b_ms": pre["lat"] * 1e3, "wait_ms": _ms(pre["wait"])},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B, "share": share,
                       "tpot_pure_ms": tp * 1e3, "slot_wait_ms": _ms(slot)},
            "stall_ms": pre["lat"] * 1e3, "stall_share": stall_share, "_pre": pre, "_dec": dec}


def _fused_step(dec: Result, pre1: Result, f: float, reread_frac: float) -> float:
    """One iteration carrying dec's batch plus an f·S-token prompt chunk (stage-level fusion, see module doc)."""
    if f <= 0:
        return dec.step
    worst = 0.0
    for sd, sp in zip(dec.stages, pre1.stages):
        td, tp = sd.time, sp.time
        bw = td.dram_bytes / td.t_dram if td.t_dram > 0 else math.inf
        nonw = sp.dram["total"] - sp.dram["weights"]
        dram = td.t_dram + (f * nonw + reread_frac * sp.dram["kv_write"]) / bw
        t = max(td.t_compute + f * tp.t_compute, dram, td.t_slc, td.t_link + f * tp.t_link) + max(td.t_sync, tp.t_sync)
        worst = max(worst, t)
    return dec.step * worst / dec.tick if dec.tick > 0 else math.inf


def _coloc_chunked(ctx: dict, lam: float) -> dict:
    cp, r_c, S, out, B, C = ctx["cpool"], ctx["r_c"], ctx["S"], ctx["out"], ctx["B"], ctx["C"]
    lam_c = lam / r_c
    pre1 = cp.run("prefill", 1)
    res = {"lambda_rps": lam, "stable": False, "chunk_tokens": C}
    if not pre1.fits:
        return {**res, "why": "单条 prefill 放不下"}
    f = C / S
    reread = max(0.0, (S - C) / (2 * S))
    n_chunks = math.ceil(S / C)

    def at(k: int):
        r = cp.run("decode", k)
        t1 = _fused_step(r, pre1, f, reread)
        rho = lam_c * S * t1 / C
        tbar = rho * t1 + (1 - rho) * r.step
        return r, t1, rho, tbar, lam_c * out / r.tokens_per_step * tbar

    rB, t1B, rhoB, _, nB = at(B)
    if not rB.fits or rhoB >= 1 or nB > B:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    lo, hi = 1, B
    while lo < hi:
        mid = (lo + hi) // 2
        r, t1, rho, tbar, n = at(mid)
        if rho < 1 and n <= mid:
            hi = mid
        else:
            lo = mid + 1
    r, t1, rho, tbar, n = at(lo)
    w = md1(lam_c, n_chunks * t1, QS)
    if not w["stable"]:
        return {**res, "why": "分块 prefill 队列不稳定"}
    ttft = {q: w[q] + n_chunks * t1 for q in ("mean", "p50", "p90", "p99")}
    k90 = min(B, poisson_quantile(n, 0.9))
    r90 = cp.run("decode", max(lo, k90))
    t1_90, t0_90 = _fused_step(r90, pre1, f, reread), r90.step
    e = r.tokens_per_step
    iters = max(1, round(out / e))
    def req_tpot(q: float) -> float:      # request-average: j of its iterations carry a chunk, j ~ Poisson(ρ·iters)
        j = min(iters, poisson_quantile(rho * iters, q))
        return (t1_90 * j + t0_90 * (iters - j)) / iters / e
    slot = mdc_wait(lam_c, out / e * tbar, B, QS)
    return {**res, "stable": True, "ttft": ttft, "tpot_mean": tbar / e,
            "tpot_p90": req_tpot(0.9), "tpot_p99": req_tpot(0.99), "itl_max": t1_90 / e,
            "e2e_mean": ttft["mean"] + slot["mean"] + out * tbar / e,
            "prefill": {"chunk_tokens": C, "chunks": n_chunks, "iter_ms": t1 * 1e3, "rho": rho, "wait_ms": _ms(w)},
            "decode": {"running_batch": lo, "occupancy": n, "slots": B, "iter_ms_no_chunk": r.step * 1e3,
                       "slot_wait_ms": _ms(slot)},
            "_dec_r": r, "_pre1": pre1}


def _slo_rate(fn, ctx: dict, start: float, ttft_slo: float, tpot_slo: float) -> float:
    """Largest λ meeting p90 TTFT and p90 TPOT SLOs (bisection; feasibility is taken as monotone in λ)."""
    def ok(lam: float) -> bool:
        x = fn(ctx, lam)
        return x["stable"] and x["ttft"]["p90"] <= ttft_slo and x["tpot_p90"] <= tpot_slo
    if start <= 0:
        return 0.0
    lo, hi = 0.0, start
    for _ in range(40):
        if not ok(hi):
            break
        lo, hi = hi, hi * 2
    else:
        return lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4 * hi:
            break
    return lo


def _public(x: dict) -> dict:
    out = {k: v for k, v in x.items() if not k.startswith("_")}
    if "ttft" in out:
        out["ttft_ms"] = {k: v * 1e3 for k, v in out.pop("ttft").items()}
    for k in ("tpot_mean", "tpot_p90", "tpot_p99", "itl_max", "e2e_mean"):
        if k in out:
            out[k + "_ms"] = out.pop(k) * 1e3
    return out


def _per_unit(r: Result) -> dict:
    a = action_counts(r)
    u = a["units"] or 1.0
    return {k: a["counts"][k] / u for k in ACTIONS if k != "idle"}


def _energy(counts_tok: dict, card_s_tok: float, table: EnergyTable | None) -> dict:
    out = {"counts_per_token": counts_tok, "card_s_per_token": card_s_tok}
    if table is None or not table.provided:
        return out
    j = {}
    for k, attr in _UNIT_PJ.items():
        e = getattr(table, attr)
        if e is not None:
            j[k] = counts_tok.get(k, 0.0) * e * 1e-12 * (8 if k in _BITS else 1)
    if table.idle_W is not None:
        j["idle"] = table.idle_W * card_s_tok
    out.update(J_per_token=sum(j.values()), J_by_action=j)
    return out


def _mix(pre: Result, dec: Result, S: int, out: int, extra: dict, shared_weights: bool = False) -> dict:
    p, d = _per_unit(pre), _per_unit(dec)
    if shared_weights:      # chunked prefill piggybacks on the decode iteration's weight read
        w = sum(st.dram["weights"] for st in pre.stages)
        tot = sum(st.dram["total"] for st in pre.stages)
        p["dram"] *= (1 - w / tot) if tot > 0 else 1.0
    c = {k: (p[k] * S + d[k] * out) / out for k in p}
    for k, v in extra.items():
        c[k] = c.get(k, 0.0) + v / out
    return c


def queue_report(ctx: dict, lam_fluid: float, pd, sv, table: EnergyTable | None = None) -> dict:
    lam = pd.rate_rps if pd.rate_rps else pd.load * lam_fluid
    out = {"lambda_rps": lam, "load": None if pd.rate_rps else pd.load, "chunk_tokens": pd.chunk_tokens,
           "basis": "排队与连续批处理的解析近似「假设」：泊松到达、请求长度固定、M/D/1 / Erlang C、阶段级融合的分块 prefill；"
                    "分位数逐项相加（偏保守）"}
    if lam <= 0:
        out["error"] = "PD 稳态容量为 0（放不下或 SLO 下无可行 prefill），不做排队估计"
        return out
    modes = {"pd": _pd_mode(ctx, lam)}
    if ctx["r_c"] > 0:
        modes["coloc_prefill_first"] = _coloc_prefill_first(ctx, lam)
        modes["coloc_chunked"] = _coloc_chunked(ctx, lam)
    cards = ctx["n_p"] + ctx["n_d"]
    ttft_slo, tpot_slo = sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3
    fns = {"pd": _pd_mode, "coloc_prefill_first": _coloc_prefill_first, "coloc_chunked": _coloc_chunked}
    for name, x in modes.items():
        rate = _slo_rate(fns[name], ctx, max(lam_fluid, lam), ttft_slo, tpot_slo)
        x["slo_rate_rps"] = rate
        x["slo_goodput_per_card"] = rate * ctx["out"] / cards
        if x["stable"]:
            x["ttft_p90_ok"] = x["ttft"]["p90"] <= ttft_slo
            x["tpot_p90_ok"] = x["tpot_p90"] <= tpot_slo
    # SLO-goodput split search for PD (same total cards, the pools' layouts fixed): DistServe-style placement
    best = None
    c_p, c_d = ctx["n_p"] // ctx["r_p"], ctx["n_d"] // ctx["r_d"]
    splits = []
    for k in range(1, cards // c_p + 1):
        np_, nd_ = k * c_p, cards - k * c_p
        if nd_ < c_d or nd_ % c_d:
            continue
        cx = {**ctx, "n_p": np_, "n_d": nd_, "r_p": np_ // c_p, "r_d": nd_ // c_d}
        rate = _slo_rate(_pd_mode, cx, max(lam_fluid, lam), ttft_slo, tpot_slo)
        row = {"prefill_cards": np_, "decode_cards": nd_, "slo_rate_rps": rate,
               "slo_goodput_per_card": rate * ctx["out"] / cards}
        splits.append(row)
        if best is None or rate > best["slo_rate_rps"]:
            best = row
    out["pd_slo_splits"] = splits
    out["pd_slo_best_split"] = best
    # energy per output token at this load (the evaluated action counts of the operating points)
    en = {}
    S, o = ctx["S"], ctx["out"]
    x = modes["pd"]
    if x["stable"]:
        kv_extra = {ctx["tier"] if ctx["tier"] == "net" else "link": ctx["kv"]}
        en["pd"] = _energy(_mix(x["_pre"]["r"], x["_dec"]["r"], S, o, kv_extra), cards / lam / o, table)
    for name in ("coloc_prefill_first", "coloc_chunked"):
        y = modes.get(name)
        if y and y["stable"]:
            pre = y["_pre"]["r"] if name == "coloc_prefill_first" else y["_pre1"]
            dec = y["_dec"]["r"] if name == "coloc_prefill_first" else y["_dec_r"]
            C = ctx["C"]
            extra = {"dram": ctx["kv"] * (S - C) / (2 * C)} if name == "coloc_chunked" and C < S else {}
            en[name] = _energy(_mix(pre, dec, S, o, extra, shared_weights=name == "coloc_chunked"), cards / lam / o,
                               table)
    out["modes"] = {k: _public(v) for k, v in modes.items()}
    out["energy"] = en
    return out
