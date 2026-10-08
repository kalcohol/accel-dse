"""Queueing, continuous batching, chunked prefill, KV-link contention and energy for the PD report (0.51;
0.52: request-length spread, prefix cache, corrected chunked-iteration mean; 0.54: birth–death decode, TTFT
convolution, prefill-first worst gap — all checked against the request-level DES in core/pdsim, V4 serving).

Everything here is analytic on top of evaluated steps; all of it is 「假设」 and labelled so.  core/pdsim simulates
the same system event by event with the same step costs (validation, and opt-in ``pd.simulate``).
Offered load λ (requests/s, Poisson) = ``pd.rate_rps`` or ``pd.load`` × the PD fluid capacity; the colocated
system on the same cards sees the same λ.

PD (prefill pool r_p replicas, decode pool r_d replicas; random split → Poisson λ/r per replica)
  prefill   M/D/1 per replica (DistServe's model for a prefill instance).  Static batch cap b ∈ {1, 2, 4, …, 64}:
            per-request service τ_b = TTFT(b) / b (the fluid rate at b), latency floor TTFT(b); the cap minimising
            mean TTFT among stable ones is used (b > 1 is the batch-server approximation of a dynamic batcher).
  KV        one transfer stream per prefill replica, M/D/1 with service KV / β_req'; β' = β·(1 − u_coll), u_coll =
            the busiest pool's own collective traffic on the KV tier (bytes / β / tick)
  decode    continuous batching with B = serving.batch slots per replica (0.54: birth–death over step(k)):
            μ(n) = min(n,B)·e/(out·step(min(n,B))); π from detailed balance; E[TPOT] = E[k]/(λ·out) (Little on the
            running set).  Replaces the 0.51–0.53 fixed point n̄ = λ·out·TPOT(⌈n̄⌉), which ignored occupancy
            fluctuations (≈ 30 % optimistic mean TPOT at 80 % load in the §18.1 example).  The step a running request
            sees is weighted π(k)·k/step(k) (its mean equals the Little mean); request-average TPOT_q = (seen mean +
            z_q·0.858·seen sd)/e (OU window 「假设」); itl_max = the 0.99 point of the seen step law (discrete).
            A free slot = M/D/c via Erlang C × ½.  KV arriving at λ·KV / N_d per decode card
            stretches the decode step's KV-tier time by 1 / (1 − u_kv) (and the prefill pool's by λ·KV / N_p).
  TTFT_q    = (w_prefill ⊕ own latency)_q + w_kv,q + exposed KV transfer.  One service value: quantiles add exactly;
            a service mix (0.54): the q-point of W + L_i with W the M/G/1 Cramér–Lundberg tail and L_i the
            request's own latency (queueing.mg1_sum_quantile).  The KV-queue quantile is still added (small).
Colocated, same cards (⌊N / c_d⌋ replicas of the scenario layout)
  prefill-first   vLLM's default without chunking: prefills (M/D/1, same batch-cap rule on the decode layout)
                  pre-empt decode iterations; decode gets 1 − ρ_p of the time → birth–death with step(k)/(1 − ρ_p);
                  request-average TPOT_q = mean + √(decode-occupancy spread² + (z_q·√m·TTFT(b)/out)²), m = prefill
                  batches met in a lifetime (0.54: root-sum-square; 0.51–0.53 added the stall quantile on top of a
                  stretched TPOT — double count).  itl_max = one decode step + the 0.99 point over the request's
                  lifetime of the longest prefill busy period (busy periods start at λ(1 − ρ); batches per period
                  Borel(λ·lat) for cap 1 / geometric for cap > 1, durations from the latency mix 「假设」).
                  TTFT adds the residual decode step
  chunked         Sarathi-Serve style: every iteration carries the running decodes plus up to C = ``pd.chunk_tokens``
                  prompt tokens.  Iteration time per stage = max(compute_d + f·compute_p, DRAM_d + f·(prefill DRAM
                  except weights) + prefix-KV re-read, SLC_d, link_d + f·link_p) + max(sync) with f = C / S
                  (weights read once for both — the point of piggybacking); prefill is FCFS at C tokens / iteration →
                  M/D/1 with τ = ⌈S / C⌉·T_iter at the seen running batch, TTFT = w ⊕ ⌈S / C⌉·T_iter; decode =
                  birth–death over the per-iteration mean tbar(k) = T₀(k) / (1 − ν(T₁(k) − T₀(k))) (0.54; 0.52 fixed
                  the share); itl_max = the fused iteration at the seen batch's 0.99 point
TPOT quantiles are request-average TPOT (DistServe's SLO metric); ``itl_max`` = the longest single token gap.
SLO capacity: the largest λ whose p90 TTFT ≤ TTFT SLO and p90 TPOT ≤ TPOT SLO (DistServe-style SLO goodput), by
bisection, per mode; ÷ cards → requests/s/card and output tokens/s/card.  For PD the same is searched over every
prefill / decode card split of the same total (``pd_slo_best_split``; layouts stay the inputs).
Energy per output token at this load: prefill counts / prompt token × S + decode counts / token (at the running
batch n̄) × out, + KV bytes on its tier (PD), ÷ out; card·s = cards / λ / out (every provisioned card, busy or not).
Chunked prefill: the prompt's weight reads are shared with the decode iteration (dropped) and each chunk re-reads
the prefix KV (+ KV·(S − C)/(2C) DRAM bytes per request).  J only for the energy-table entries the user gave.
0.52 — variable lengths (core/lengths.py: fixed by default, else a discrete mix of prompt values S_i with weights w_i):
  every M/D/1 above becomes M/G/1 over the per-request service times of the S_i (queueing.mg1; one value → M/D/1);
  a batch of b mixed requests takes τ_i + (b − 1)·τ̄ for a request of length S_i; TTFT quantiles = wait quantile +
  the discrete quantile of (own latency + exposed KV), which are comonotone in S_i (exact sum) + KV-queue quantile;
  decode uses E[out] in Little's law (the PS birth–death occupancy is insensitive to the output-length law), the length-
  biased context (serving.ctx × ctx_ratio) and Allen–Cunneen (1 + c_s²)/2 for the slot wait.
  Prefix cache (``pd.prefix_hit`` = h): each request's first ⌊h·S_i⌋ tokens are already cached → the prefill runs
  only the rest, attending to the cached prefix (serving.prefix_cached → Phase(prefill, q = S − p, ctx = p): fewer
  GEMM rows, prefix KV read); the KV hand-off moves the uncached fraction (1 − p/S) when ``pd.prefix_on_decode``
  (the decode side holds the same prefix, e.g. a shared system prompt), else all of it.  Hit rate is an input (or,
  0.53, from the Che LRU capacity model); decode is unchanged (each request still attends its full context).
Chunked mean iteration (0.52 fix): ν = λ·Σ w_i n_i chunk iterations/s occupy ρ = λ·Σ w_i n_i T₁,i of the time, so the
  per-iteration mean is T₀ / (1 − ρ + ν T₀) and a share x = ν·that of iterations carry a chunk (0.51 used the
  time-average ρ·T₁ + (1 − ρ)·T₀ and share ρ, which is length-biased upward).
Not modelled: arrival burstiness beyond Poisson, pre-emption and KV-cache eviction, central-queue load balancing
(random split is pessimistic), length-aware scheduling (FCFS everywhere).  Known bias (V4, MODEL.md §18.4): colocated
TPOT tails are optimistic (median −14 … −18 % at p90, up to −43 % at load 0.85 with CV 1) because prefill stalls make
decode arrivals bursty, which a Poisson-fed birth–death does not see.
"""



from __future__ import annotations

import math

from .energy import ACTIONS, EnergyTable, _BITS, _UNIT_PJ, action_counts, scaleup_bytes
from .evaluate import Result, evaluate
from .queueing import dquantile, mdc_wait, mg1, mg1_sum_quantile
from .scenario import Scenario

QS = (0.5, 0.9, 0.99)
B_CAPS = (1, 2, 4, 8, 16, 32, 64)
_QK = ("mean", "p50", "p90", "p99")


def _ms(d: dict) -> dict:
    return {k: (v * 1e3 if k in _QK else v) for k, v in d.items()}


def _wsum(ws, xs) -> float:
    return sum(w * x for w, x in zip(ws, xs))


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
    """Memoised evaluations of one layout at different batches / phases / prompt lengths / cached prefixes."""

    def __init__(self, scn: Scenario):
        self.scn = scn.replace("serving.phase", "decode")
        self.memo: dict = {}

    def run(self, phase: str, b: int, S: int | None = None, p: int = 0) -> Result:
        k = (phase, b, S, p)
        if k not in self.memo:
            s = self.scn.replace("serving.phase", phase).replace("serving.batch", b)
            if S is not None and (S != s.serving.prompt or p != s.serving.prefix_cached):
                s = s.replace("serving.prefix_cached", 0).replace("serving.prompt", S).replace("serving.prefix_cached", p)
            self.memo[k] = evaluate(s)
        return self.memo[k]


def prefix_tokens(S: int, h: float) -> int:
    """Cached prefix of an S-token prompt at hit rate h (at least one new token is always computed)."""
    return min(S - 1, int(h * S)) if h > 0 else 0


def _prefill_server(pool: _Pool, lam: float, pts, scale: float = 1.0) -> dict | None:
    """Batch cap minimising mean TTFT (M/G/1 over the prompt mix, τ_i = TTFT(b, S_i)/b); None if none is stable."""
    best = None
    ws = [w for w, _, _ in pts]
    for b in B_CAPS:
        rs = [pool.run("prefill", b, S, p) for _, S, p in pts]
        if not all(r.fits for r in rs):
            break
        taus = [r.ttft * scale / b for r in rs]
        w = mg1(lam, taus, ws, QS)
        if not w["stable"]:
            continue
        tbar = _wsum(ws, taus)
        lats = [t + (b - 1) * tbar for t in taus] if len(taus) > 1 else [rs[0].ttft * scale]
        lat = _wsum(ws, lats)
        if best is None or w["mean"] + lat < best["wait"]["mean"] + best["lat"]:
            best = {"b": b, "lat": lat, "lats": lats, "wait": w, "rs": rs, "ws": ws, "scale": scale, "taus": taus,
                    "lam": lam}
    return best


_Z = {0.5: 0.0, 0.9: 1.2815515655446004, 0.99: 2.3263478740408408}
_WIN = math.sqrt(2.0 * math.exp(-1.0))   # OU time-average over one correlation time: σ factor ≈ 0.858


def _decode_birth_death(pool: _Pool, lam: float, out: float, B: int, share: float = 1.0,
                        scale=lambda r: 1.0, n_max: int = 4096, step_fn=None) -> dict | None:
    """Continuous-batching decode as a birth–death process (0.54; validated against core/pdsim).

    State n = requests at the replica (running + waiting for a slot).  Every running sequence advances one engine
    step per iteration, so the replica is a processor-sharing server with state-dependent capacity: departure rate
    μ(n) = k·e / (out · step(k)), k = min(n, B).  Stationary π_n ∝ Π λ/μ(j) (exact for PS, insensitive to the
    output-length law); E[TPOT] = E[k] / (λ·out) by Little on the running set.  The 0.51–0.53 Little fixed point
    n̄ = λ·out·TPOT(⌈n̄⌉) ignored occupancy fluctuations and is optimistic when step(k) grows with k (KV reads) —
    e.g. −30 % mean TPOT at 80 % load in the §18.1 example.
    Request-average TPOT quantiles: a running request sees the size-biased batch (mean m_s, sd σ_s); over its
    lifetime (≈ one correlation time of the occupancy) the average batch has sd ≈ 0.86·σ_s (OU window) 「假设」.
    ``step_fn(k) → (step, Result)`` overrides the pool step (chunked prefill uses its mean iteration)."""
    steps: dict = {}

    def step_of(k: int):
        if k not in steps:
            if step_fn is not None:
                steps[k] = step_fn(k)
            else:
                r = pool.run("decode", k)
                steps[k] = (r.step * scale(r) / share, r)
        return steps[k]

    stB, rB = step_of(B)
    if not rB.fits or not math.isfinite(stB):
        return None
    e = rB.tokens_per_step
    if lam <= 0 or out <= 0:
        st1, r1 = step_of(1)
        return {"k": 1, "occupancy": 0.0, "r": r1, "tpot": st1 / e, "step": st1, "running_mean": 1.0,
                "pi_run": (1.0,), "seen_mean": 1.0, "seen_sd": 0.0, "step_of": step_of, "e": e}
    if lam >= B * e / (out * stB):       # even a full batch cannot keep up
        return None
    logp = [0.0]
    for n in range(1, n_max + 1):
        k = min(n, B)
        st, _ = step_of(k)
        if not math.isfinite(st):
            return None
        logp.append(logp[-1] + math.log(lam * out * st / (k * e)))
        if n > B + 8 and logp[-1] < max(logp) - 40:
            break
    m = max(logp)
    w = [math.exp(x - m) for x in logp]
    Z = sum(w)
    pi = [x / Z for x in w]
    EN = sum(n * p for n, p in enumerate(pi))
    run_w = [0.0] * (B + 1)
    for n, p in enumerate(pi):
        run_w[min(n, B)] += p
    Ek = sum(k * p for k, p in enumerate(run_w))               # unconditional mean running batch
    busy = 1.0 - run_w[0]
    pi_run = tuple(x / busy for x in run_w[1:]) if busy > 0 else (1.0,)
    k_mean = Ek / busy if busy > 0 else 1.0
    # what a running request sees per iteration: weight ∝ π(k)·k / step(k) (iterations at k occur at rate 1/step;
    # the request is in k of them).  Its mean step = E[k] / E[k/step] = the Little mean exactly.
    wsee = [run_w[k] * k / step_of(k)[0] if k else 0.0 for k in range(B + 1)]
    ws_tot = sum(wsee) or 1.0
    wsee = [x / ws_tot for x in wsee]
    seen_mean = sum(k * x for k, x in enumerate(wsee))
    seen_sd = math.sqrt(max(0.0, sum(k * k * x for k, x in enumerate(wsee)) - seen_mean ** 2))
    st_mean = sum(step_of(k)[0] * x for k, x in enumerate(wsee) if x)
    st_sd = math.sqrt(max(0.0, sum(step_of(k)[0] ** 2 * x for k, x in enumerate(wsee) if x) - st_mean ** 2))
    k_hat = max(1, min(B, int(round(seen_mean))))
    st, r = step_of(k_hat)
    return {"k": k_hat, "occupancy": EN, "r": r, "tpot": Ek / (lam * out), "step": st, "running_mean": k_mean,
            "pi_run": pi_run, "seen_mean": seen_mean, "seen_sd": seen_sd, "w_seen": tuple(wsee), "step_seen_mean": st_mean,
            "step_seen_sd": st_sd, "step_of": step_of, "e": e}




def _tpot_req_q(dec: dict, q: float) -> float:
    """Request-average TPOT quantile: mean seen step + z·0.86·sd (lifetime ≈ one occupancy correlation time)."""
    return (dec["step_seen_mean"] + _Z[q] * _WIN * dec["step_seen_sd"]) / dec["e"]


def _gap_q(dec: dict, q: float) -> float:
    """Single-iteration token gap at the q-quantile of the batch a running request sees (discrete, no averaging)."""
    acc = 0.0
    for k, x in enumerate(dec["w_seen"]):
        acc += x
        if x and acc >= q:
            return dec["step_of"](k)[0] / dec["e"]
    return dec["step_of"](len(dec["w_seen"]) - 1)[0] / dec["e"]


def _busy_max_q(lam: float, lat: float, cap: int, life: float, rho: float, q: float) -> int:
    """q-quantile of the longest run of back-to-back prefill batches a request meets during ``life`` (prefill-first:
    decode stalls for the whole prefill busy period).  Busy periods start at λ(1 − ρ); batches per busy period:
    Borel(λ·lat) for cap 1 (every arrival needs its own service), geometric with continuation 1 − e^{−λ·lat} for
    cap > 1 (one batch absorbs the arrivals) 「假设」."""
    a = lam * lat
    m = max(0.0, lam * (1 - rho) * life)
    if a <= 0 or m <= 0:
        return 1
    F = 0.0
    for n in range(1, 400):
        if cap <= 1:
            if a >= 1:
                return 400
            pn = math.exp(-a * n + (n - 1) * math.log(a * n) - math.lgamma(n + 1))
        else:
            c = -math.expm1(-a)
            pn = (1 - c) * c ** (n - 1)
        F += pn
        if F >= 1 or F ** m >= q:
            return n
    return 400


def _busy_dur_max_q(lam: float, lats_w, cap: int, life: float, rho: float, q: float) -> float:
    """Duration version of _busy_max_q for a mixed prompt law (0.54): busy period = Σ_{j≤N} L_j with N as in
    _busy_max_q (count from the mean latency) and L_j iid from the discrete latency mix (independence of N and the
    L_j 「假设」).  Returns d with P(D ≤ d)^m = q.  A single latency value reduces exactly to n_q · lat."""
    pts = [(w, l) for w, l in lats_w if w > 0]
    tot = sum(w for w, _ in pts)
    pts = [(w / tot, l) for w, l in pts]
    lat = sum(w * l for w, l in pts)
    if len({l for _, l in pts}) <= 1:
        return _busy_max_q(lam, lat, cap, life, rho, q) * lat
    a = lam * lat
    m = max(0.0, lam * (1 - rho) * life)
    if a <= 0 or m <= 0:
        return max(l for _, l in pts)
    if cap <= 1 and a >= 1:
        return math.inf
    target = q ** (1.0 / m)                       # per-busy-period CDF level
    dt = min(l for _, l in pts) / 8.0
    idx = [(w, max(1, int(round(l / dt)))) for w, l in pts]
    cur = {0: 1.0}                                # pmf of Σ_{j≤n} L_j on the grid
    dist: dict = {}
    F_n = 0.0
    for n in range(1, 400):
        nxt: dict = {}
        for k, pk in cur.items():
            for w, i in idx:
                nxt[k + i] = nxt.get(k + i, 0.0) + pk * w
        cur = {k: v for k, v in nxt.items() if v > 1e-14}
        if cap <= 1:
            pn = math.exp(-a * n + (n - 1) * math.log(a * n) - math.lgamma(n + 1))
        else:
            c = -math.expm1(-a)
            pn = (1 - c) * c ** (n - 1)
        for k, v in cur.items():
            dist[k] = dist.get(k, 0.0) + pn * v
        F_n += pn
        if 1 - F_n < (1 - target) * 1e-3:
            break
    acc = 0.0
    for k in sorted(dist):
        acc += dist[k]
        if acc >= target:
            return k * dt
    return max(dist) * dt


def _decode_fixed_point(pool: _Pool, lam: float, out: float, B: int, share: float = 1.0,
                        scale=lambda r: 1.0) -> dict | None:
    """Continuous batching (0.54 = birth–death; name kept for call sites)."""
    return _decode_birth_death(pool, lam, out, B, share, scale)




def _tpot_at(pool: _Pool, k: int, share: float = 1.0, scale=lambda r: 1.0) -> float:
    r = pool.run("decode", max(1, k))
    return r.step * scale(r) / share / r.tokens_per_step


def _ttft(wait: dict, lats_w, extra: dict | None = None, add: float = 0.0, conv: tuple | None = None) -> dict:
    """TTFT quantiles = prefill-wait ⊕ own latency (+ ``extra`` queue quantile + ``add``).  ``conv`` = (λ, taus)
    of the prefill M/G/1: with a mixed service law the wait and the request's own latency are combined as a proper
    sum of independent variables (0.54, mg1_sum_quantile); otherwise (one service value) the quantiles add exactly."""
    ws = [w for w, _ in lats_w]
    out = {"mean": wait["mean"] + _wsum(ws, [x for _, x in lats_w]) + add + (extra["mean"] if extra else 0.0)}
    for q in QS:
        k = f"p{q * 100:g}"
        core = None
        if conv is not None:
            core = mg1_sum_quantile(conv[0], conv[1], ws, [x for _, x in lats_w], q)
        if core is None:
            core = wait[k] + dquantile(lats_w, q)
        out[k] = core + add + (extra[k] if extra else 0.0)
    return out


def _pd_mode(ctx: dict, lam: float) -> dict:
    pp, dp = ctx["ppool"], ctx["dpool"]
    r_p, r_d, n_p, n_d = ctx["r_p"], ctx["r_d"], ctx["n_p"], ctx["n_d"]
    pts, out, B = ctx["pts"], ctx["out"], ctx["B"]
    tier, beta, alpha = ctx["tier"], ctx["beta"], ctx["alpha"]
    ws = [w for w, _, _ in pts]
    xfer = [ctx["kv_xfer"][(S, p)] for _, S, p in pts]
    kv = _wsum(ws, xfer)
    lam_p, lam_d = lam / r_p, lam / r_d
    shared = ctx.get("shared", True)      # KV rides the pools' own collective tier (no pd.kv_GBps override)
    u_kv_d = lam * kv / (n_d * beta) if kv > 0 and shared else 0.0
    u_kv_p = lam * kv / (n_p * beta) if kv > 0 and shared else 0.0
    sc_d = lambda r: _contended(r, tier, beta, u_kv_d)
    rep = pp.run("prefill", 1, *ctx["rep"])
    pre = _prefill_server(pp, lam_p, pts, _contended(rep, tier, beta, u_kv_p))
    dec = _decode_fixed_point(dp, lam_d, out, B, scale=sc_d)
    res = {"lambda_rps": lam, "stable": pre is not None and dec is not None and u_kv_d < 1 and u_kv_p < 1}
    if not res["stable"]:
        res["why"] = ("prefill 池" if pre is None else "decode 池" if dec is None else "KV 链路") + "在此负载下不稳定" \
            + ("（KV 与池内集合通信共用互联层，争用计入）" if max(u_kv_d, u_kv_p) > 0.3 else "")
        return res
    u_coll = max(_wsum(ws, [_tier_util(r, tier, beta) for r in pre["rs"]]), _tier_util(dec["r"], tier, beta)) \
        if shared else 0.0
    beta_av = beta * (1 - u_coll)
    b_req = ctx["pair"] * beta_av
    t_bw = [x / b_req if b_req > 0 else math.inf for x in xfer]
    kvq = mg1(lam_p, t_bw, ws, QS)
    if not kvq["stable"]:
        return {**res, "stable": False, "why": "KV 链路（扣除池内集合通信后）在此负载下不稳定"}
    if ctx["layerwise"]:
        L = ctx["L"]
        exposed = [max(alpha + t / L, L * alpha + t - lat * (L - 1) / L) for t, lat in zip(t_bw, pre["lats"])]
    else:
        exposed = [alpha + t for t in t_bw]
    ttft = _ttft(pre["wait"], [(w, lat + e) for w, lat, e in zip(ws, pre["lats"], exposed)], kvq,
                 conv=(pre["lam"], pre["taus"]))
    slot = mdc_wait(lam_d, out * dec["tpot"], B, QS, cs2=ctx["out_cs2"])
    tp90, tp99 = _tpot_req_q(dec, 0.9), _tpot_req_q(dec, 0.99)
    return {**res, "ttft": ttft, "tpot_mean": dec["tpot"], "tpot_p90": tp90,
            "tpot_p99": tp99, "itl_max": _gap_q(dec, 0.99),
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"batch_cap": pre["b"], "ttft_b_ms": pre["lat"] * 1e3, "wait_ms": _ms(pre["wait"]),
                        "kv_slowdown": pre["scale"]},
            "kv": {"wait_ms": _ms(kvq), "exposed_ms": _wsum(ws, exposed) * 1e3, "bytes_per_req": kv,
                   "shared_tier": shared, "u_coll": u_coll, "GBps_req_avail": b_req / 1e9,
                   "u_kv_decode": u_kv_d, "u_kv_prefill": u_kv_p},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B,
                       "kv_slowdown": sc_d(dec["r"]), "slot_wait_ms": _ms(slot),
                       "running_p90": max(1, int(round(dec["seen_mean"] + _Z[0.9] * dec["seen_sd"])))},
            "_pre": pre, "_dec": dec}


def _coloc_prefill_first(ctx: dict, lam: float) -> dict:
    cp, r_c, pts, out, B = ctx["cpool"], ctx["r_c"], ctx.get("pts_c", ctx["pts"]), ctx["out"], ctx["B"]
    lam_c = lam / r_c
    pre = _prefill_server(cp, lam_c, pts)
    res = {"lambda_rps": lam, "stable": pre is not None}
    if pre is None:
        return {**res, "why": "prefill 在此负载下不稳定"}
    share = 1 - pre["wait"]["rho"]
    dec = _decode_fixed_point(cp, lam_c, out, B, share=share)
    if dec is None or share <= 0:
        return {**res, "stable": False, "why": "decode 在此负载下不稳定（prefill 占用后剩余时间不够）"}
    tp = dec["r"].step / dec["r"].tokens_per_step
    resid = dec["r"].step
    lats_w = list(zip(pre["ws"], pre["lats"]))
    ttft = _ttft(pre["wait"], lats_w, conv=(pre["lam"], pre["taus"]))
    ttft["mean"] += resid / 2
    ttft["p50"] += resid / 2
    ttft["p90"] += resid
    ttft["p99"] += resid
    stall_share = min(1.0, lam_c / pre["b"] * dec["tpot"])
    # request-average TPOT: pure decode steps at the running batch + the prefill stalls met during its lifetime
    # (prefill batches start at λ_c / b, Poisson → count ~ Poisson((λ_c / b) · lifetime)), each of mean length
    life = out * dec["tpot"]
    m_st = lam_c / pre["b"] * life
    # request-average TPOT: the birth–death mean already contains the stalls (steps ÷ share); the spread combines the
    # decode-occupancy deviation and the Poisson count of stalls met (independent, root-sum-square) — 0.54; the
    # 0.51–0.53 form added the stall quantile on top of a stretched TPOT (double count).
    pure = {q: (_tpot_req_q(dec, q) - _tpot_req_q(dec, 0.5)) for q in (0.9, 0.99)}
    req_q = {q: dec["tpot"] + math.sqrt(pure[q] ** 2 + (_Z[q] * math.sqrt(m_st) * pre["lat"] / out) ** 2)
             for q in (0.9, 0.99)}
    slot = mdc_wait(lam_c, out * dec["tpot"], B, QS, cs2=ctx["out_cs2"])
    # longest token gap (p99 over requests of each request's worst gap): one decode step + the longest prefill busy
    # period met in its lifetime (prefill-first stalls decode for the whole busy period)
    d99 = _busy_dur_max_q(lam_c, lats_w, pre["b"], life, pre["wait"]["rho"], 0.99)
    return {**res, "ttft": ttft, "tpot_mean": dec["tpot"], "tpot_p90": req_q[0.9], "tpot_p99": req_q[0.99],
            "itl_max": dec["step_seen_mean"] * share / dec["e"] + d99,
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"batch_cap": pre["b"], "ttft_b_ms": pre["lat"] * 1e3, "wait_ms": _ms(pre["wait"])},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B, "share": share,
                       "tpot_pure_ms": tp * 1e3, "slot_wait_ms": _ms(slot)},
            "stall_ms": pre["lat"] * 1e3, "stall_share": stall_share, "_pre": pre, "_dec": dec}


def _fused_step(dec: Result, pre1: Result, f: float, reread_frac: float) -> float:
    """One iteration carrying dec's batch plus an f share of pre1's prompt (stage-level fusion, see module doc);
    ``reread_frac`` = KV re-read per chunk as a fraction of pre1's KV write."""
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


def _chunk_plan(S: int, p: int, C: int) -> tuple[int, float, float]:
    """(chunks, f = share of the (uncached) prefill per chunk, KV re-read per chunk / the prefill's KV write)."""
    new = S - p
    n = math.ceil(new / C)
    f = min(1.0, C / new)
    reread = (p * (1 - f) + max(0.0, (new - C) / 2)) / new      # cached prefix + the earlier chunks, on average
    return n, f, reread


def _coloc_chunked(ctx: dict, lam: float) -> dict:
    cp, r_c, pts, out, B, C = ctx["cpool"], ctx["r_c"], ctx.get("pts_c", ctx["pts"]), ctx["out"], ctx["B"], ctx["C"]
    lam_c = lam / r_c
    ws = [w for w, _, _ in pts]
    res = {"lambda_rps": lam, "stable": False, "chunk_tokens": C}
    pre1 = [cp.run("prefill", 1, S, p) for _, S, p in pts]
    if not all(r.fits for r in pre1):
        return {**res, "why": "单条 prefill 放不下"}
    plans = [_chunk_plan(S, p, C) for _, S, p in pts]
    ns = [n for n, _, _ in plans]
    nu = lam_c * _wsum(ws, ns)                  # chunk iterations / s

    # Mean iteration at running batch k: a share x of iterations carries a chunk (ν chunk iterations / s) →
    # tbar = T₀ / (1 − ν·(T₁ − T₀)), x = ν·tbar (0.52 per-iteration mean); birth–death uses tbar as step(k).
    memo: dict = {}

    def mean_step(k: int):
        if k not in memo:
            r = cp.run("decode", k)
            t1 = [_fused_step(r, p1, f, rr) for p1, (_, f, rr) in zip(pre1, plans)]
            t1m = _wsum(ws, [m * t for m, t in zip(ns, t1)]) / _wsum(ws, ns)
            den = 1 - nu * (t1m - r.step)
            tbar = r.step / den if den > 0 else math.inf
            memo[k] = (tbar, r, t1, min(1.0, nu * tbar), t1m)
        return memo[k]

    tB, rB, _, xB, _ = mean_step(B)
    if not rB.fits or not math.isfinite(tB) or xB >= 1:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    dec = _decode_birth_death(cp, lam_c, out, B, step_fn=lambda k: mean_step(k)[:2])
    if dec is None:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    k_s = max(1, min(B, int(round(dec["seen_mean"]))))
    _, r, t1, x, t1m = mean_step(k_s)
    serv = [m * t for m, t in zip(ns, t1)]          # prefill service = chunks × fused iteration at the seen batch
    w = mg1(lam_c, serv, ws, QS)
    if not w["stable"]:
        return {**res, "why": "分块 prefill 队列不稳定"}
    ttft = _ttft(w, list(zip(ws, serv)), conv=(lam_c, serv))
    k99 = max(1, min(B, int(math.ceil(dec["seen_mean"] + _Z[0.99] * dec["seen_sd"]))))
    t1_99 = mean_step(k99)[2]
    slot = mdc_wait(lam_c, out * dec["tpot"], B, QS, cs2=ctx["out_cs2"])
    return {**res, "stable": True, "ttft": ttft, "tpot_mean": dec["tpot"],
            "tpot_p90": _tpot_req_q(dec, 0.9), "tpot_p99": _tpot_req_q(dec, 0.99),
            "itl_max": max(t1_99) / dec["e"],
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"chunk_tokens": C, "chunks": _wsum(ws, ns), "iter_ms": t1m * 1e3,
                        "rho": lam_c * _wsum(ws, serv), "chunk_share": x, "wait_ms": _ms(w)},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B,
                       "iter_ms_no_chunk": r.step * 1e3, "slot_wait_ms": _ms(slot)},
            "_dec_r": r, "_pre1": pre1, "_plans": plans}


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


def _mix(pres, pts, dec: Result, out: float, extra: dict, shared_weights: bool = False) -> dict:
    """Counts per output token: Σ_i w_i · (prefill counts per prompt token at S_i) · S_i + decode counts · out, ÷ out.
    (Prefill units are full prompt tokens, so a cached prefix shows up as fewer counts per unit.)"""
    d = _per_unit(dec)
    c = {k: d[k] * out for k in d}
    for r, (w, S, _) in zip(pres, pts):
        p = _per_unit(r)
        if shared_weights:      # chunked prefill piggybacks on the decode iteration's weight read
            wt = sum(st.dram["weights"] for st in r.stages)
            tot = sum(st.dram["total"] for st in r.stages)
            p["dram"] *= (1 - wt / tot) if tot > 0 else 1.0
        for k in c:
            c[k] += w * p[k] * S
    c = {k: v / out for k, v in c.items()}
    for k, v in extra.items():
        c[k] = c.get(k, 0.0) + v / out
    return c


def queue_report(ctx: dict, lam_fluid: float, pd, sv, table: EnergyTable | None = None) -> dict:
    lam = pd.rate_rps if pd.rate_rps else pd.load * lam_fluid
    out = {"lambda_rps": lam, "load": None if pd.rate_rps else pd.load, "chunk_tokens": pd.chunk_tokens,
           "basis": "排队与连续批处理的解析近似「假设」：泊松到达、" + ("请求长度固定" if ctx["len"].trivial else
                    "请求长度按离散分布（M/G/1）") + ("、前缀缓存命中率 " + f"{pd.prefix_hit:.0%}" if pd.prefix_hit else "")
                    + ("、前缀缓存容量模型（LRU / Che：PD prefill 命中 {:.0%}、合并 {:.0%}；命中 / 未命中两类请求按 M/G/1）"
                       .format(*ctx["prefix_hits"]) if ctx.get("prefix_hits") else "")
                    + "、M/D/1 / Erlang C、decode 连续批处理按 birth–death（0.54）、阶段级融合的分块 prefill；"
                    "服务时间混合时 TTFT = 等待 ⊕ 自身时延（卷积）；经请求级 DES 对照（V4，见建模说明 §18.4）"}
    if lam <= 0:
        out["error"] = "PD 稳态容量为 0（放不下或 SLO 下无可行 prefill），不做排队估计"
        return out
    modes = {"pd": _pd_mode(ctx, lam)}
    if ctx["r_c"] > 0:
        modes["coloc_prefill_first"] = _coloc_prefill_first(ctx, lam)
        modes["coloc_chunked"] = _coloc_chunked(ctx, lam)
    cards = ctx["n_p"] + ctx["n_d"]
    o = ctx["out"]
    ttft_slo, tpot_slo = sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3
    fns = {"pd": _pd_mode, "coloc_prefill_first": _coloc_prefill_first, "coloc_chunked": _coloc_chunked}
    for name, x in modes.items():
        rate = _slo_rate(fns[name], ctx, max(lam_fluid, lam), ttft_slo, tpot_slo)
        x["slo_rate_rps"] = rate
        x["slo_goodput_per_card"] = rate * o / cards
        x["stable_rate_rps"] = _slo_rate(fns[name], ctx, max(lam_fluid, lam), math.inf, math.inf)
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
        row = {"prefill_cards": np_, "decode_cards": nd_, "slo_rate_rps": rate, "slo_goodput_per_card": rate * o / cards}
        splits.append(row)
        if best is None or rate > best["slo_rate_rps"]:
            best = row
    out["pd_slo_splits"] = splits
    out["pd_slo_best_split"] = best
    # energy per output token at this load (the evaluated action counts of the operating points)
    en = {}
    pts = ctx["pts"]
    ws = [w for w, _, _ in pts]
    x = modes["pd"]
    if x["stable"]:
        kv = _wsum(ws, [ctx["kv_xfer"][(S, p)] for _, S, p in pts])
        en["pd"] = _energy(_mix(x["_pre"]["rs"], pts, x["_dec"]["r"], o, {ctx["tier"] if ctx["tier"] == "net" else "link": kv}),
                           cards / lam / o, table)
    pts = ctx.get("pts_c", pts)                 # colocated replicas: their own prefix-cache hit mix (0.53)
    ws = [w for w, _, _ in pts]
    y = modes.get("coloc_prefill_first")
    if y and y["stable"]:
        en["coloc_prefill_first"] = _energy(_mix(y["_pre"]["rs"], pts, y["_dec"]["r"], o, {}), cards / lam / o, table)
    y = modes.get("coloc_chunked")
    if y and y["stable"]:
        reread = _wsum(ws, [n * rr * ctx["kv_new"][(S, p)] for (n, _, rr), (_, S, p) in zip(y["_plans"], pts)])
        en["coloc_chunked"] = _energy(_mix(y["_pre1"], pts, y["_dec_r"], o, {"dram": reread}, shared_weights=True),
                                      cards / lam / o, table)
    out["modes"] = {k: _public(v) for k, v in modes.items()}
    out["energy"] = en
    return out
