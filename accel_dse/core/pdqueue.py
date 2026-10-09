"""Queueing, continuous batching, chunked prefill, KV-link contention and energy for the PD report (0.51;
0.52: request-length spread, prefix cache, corrected chunked-iteration mean; 0.54: birth–death decode, TTFT
convolution, prefill-first worst gap — all checked against the request-level DES in core/pdsim, V4 serving;
0.55: colocated decode burstiness (reduced-clock batch chain), occupancy autocorrelation window, quasi-static
chunked TTFT, worst gap from up-crossings, decode KV capacity admission / preemption; 0.56: vLLM admission order
with the wait in TTFT, preemption mixture in the TPOT / gap tails, swap preemption).

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
            z_q·win·seen sd)/e, win = √(2(x − 1 + e^{−x})/x²), x = life/τ_int with τ_int the occupancy's integrated
            autocorrelation time solved exactly from the chain (0.55; exponential correlation 「假设」; 0.54 fixed
            win = 0.858 — near saturation τ_int ≫ life and win → 1); itl_max = one step at the 0.99 point of the
            largest batch met over the life (0.55: up-crossing rate π_{m+1}μ_{m+1}, Poisson clumping 「假设」).
            A free slot = M/D/c via Erlang C × ½.  KV arriving at λ·KV / N_d per decode card
            stretches the decode step's KV-tier time by 1 / (1 − u_kv) (and the prefill pool's by λ·KV / N_p).
  TTFT_q    = (w_prefill ⊕ own latency)_q + w_kv,q + exposed KV transfer.  One service value: quantiles add exactly;
            a service mix (0.54): the q-point of W + L_i with W the M/G/1 Cramér–Lundberg tail and L_i the
            request's own latency (queueing.mg1_sum_quantile).  The KV-queue quantile is still added (small).
Colocated, same cards (⌊N / c_d⌋ replicas of the scenario layout)
  prefill-first   vLLM's default without chunking: prefills (M/D/1, same batch-cap rule on the decode layout)
                  pre-empt decode iterations.  0.55 — reduced clock: leave out the prefill time; there a prefill busy
                  period is an instant that hands its X requests (M/G/1 busy-period count, Takács / Borel) to decode
                  as one batch → batch-Poisson birth–death (level crossing, exact for a skip-free-downward chain),
                  wall TPOT = E[k]/(λ·out), seen step stretched by 1/(1 − ρ_p).  The batch's pair term is scaled by
                  κ = 2∫P(O > t)²dt/E[O] (members that join together leave together: 1 for exponential outputs, 2 for
                  a fixed length) by thinning / pairwise merging of batches at the same mean rate.  0.54 ran a
                  Poisson-fed chain at step/(1 − ρ_p) (TPOT p90 median −18 %).  TPOT_q = mean + √(occupancy spread² +
                  (z_q·√(n_bp·E[V²])/out)²), n_bp = busy periods met over the life, E[V²] = E[S²]/(1 − ρ)³.  itl_max =
                  one decode step + the 0.99 point of the longest busy period over the life (exact Takács durations
                  for cap 1, geometric for cap > 1).  TTFT adds the residual decode step
  chunked         Sarathi-Serve style: every iteration carries the running decodes plus up to C = ``pd.chunk_tokens``
                  prompt tokens.  Iteration time per stage = max(compute_d + f·compute_p, DRAM_d + f·(prefill DRAM
                  except weights) + prefix-KV re-read, SLC_d, link_d + f·link_p) + max(sync) with f = C / S
                  (weights read once for both — the point of piggybacking); the last chunk carries its real share
                  (0.55).  Decode (0.55): reduced clock without the chunk excess n_c·(T₁ − T₀); busy periods of the
                  chunk server still advance decode for their T₀ part, so their joins are spread — renewal–reward on
                  the joint (X, W) busy-period moments (branching identity) gives the long-run dispersion, matched by
                  binomial thinning of the batch (× κ as above).  TTFT (0.55): the running batch drifts far slower
                  than a prefill busy period → quasi-static mixture over 8 equal-mass batch environments of M/G/1
                  queues (exact M/D/1 for one service value), each with a decode-only iteration as vacation
                  (Fuhrmann–Cooper: W ⊕ U(0, T₀)); itl_max = the fused iteration at the largest batch met over the
                  life (0.99 point)
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
Decode KV capacity (0.55, ``pd.kv_policy``, default off): B_kv = ⌊K / mean running footprint⌋ (wait: S + out
  reserved, out size-biased E[o²]/E[o]; recompute / swap: S + generated, E[o²]/(2E[o])); the mode runs with
  min(B, B_kv) slots; admission wait = the chain's queue behind the slots (Little) × (1 + c_s²)/2 (Lee–Longton).
  0.56 — admission order ``pd.kv_admit``: before_prefill (vLLM, default) reserves the slot + KV before the prefill
  (PD: before the KV pull), so when the capacity binds the wait W (P(wait) = p) is in TTFT: the TTFT law (rebuilt
  from its p50 / p90 / p99) ⊕ (atom 1 − p, Exp(W/p)) — and so in the SLO goodput; after_prefill = 0.55 (reported apart).
  Preemption (recompute / swap): ν = ν_Rice (up-crossings of K by the running footprint, Normal per batch size) +
  ν_sat (an admission at a full replica fills to a headroom ~U(0, S): rate p·λ_r·E[(E[o]/S)(1 − e^{−S/E[o]})])
  「假设」; a victim's gap = T_fix + Exp(1/λ_r) (stall + re-queue at the head), mixed into the TPOT and worst-gap
  quantiles with weight p_v = preemptions / request; the stalls take a share f = ν·T_stall of the replica (≥ 1 →
  unstable).  Recompute: T_stall = re-prefill of S̄ + ḡ; swap: 2·(S̄ + ḡ)·bytes/token ÷ (host GB/s per card × cards
  per replica) (``pd.swap_GBps``, default workload.host_GBps 「假设」).
Not modelled: arrival burstiness beyond Poisson, central-queue load balancing (random split is pessimistic),
length-aware scheduling (FCFS everywhere), chunked prefill inside the PD prefill pool.  Known bias (V4 0.56, MODEL.md
§18.6): TPOT / tails within the DES seed noise at ≤ 0.85 load without a KV cap; with a binding KV cap the folded TTFT
is conservative (p90 +20…40 %) and recompute at high load / CV 1 misses the DES's preemption churn (optimistic).
"""



from __future__ import annotations

import bisect
import math

from .energy import ACTIONS, EnergyTable, _BITS, _UNIT_PJ, action_counts, scaleup_bytes
from .evaluate import Result, evaluate
from .queueing import mg1_mix_sum_quantile, dquantile, erlang_c, mdc_wait, mg1, mg1_sum_quantile
from .scenario import Scenario

QS = (0.5, 0.9, 0.99)
B_CAPS = (1, 2, 4, 8, 16, 32, 64)
SPLIT_EXACT = 32    # 0.60: PD SLO split search evaluates every split up to this many candidates, else coarse → fine
INTERP_B = 1024     # 0.60: decode batch above which the birth–death chain interpolates step(k) (see _decode_birth_death)
FAST_B, FAST_NL = 256, 20000   # 0.61: chains with B ≥ FAST_B and N·L > FAST_NL use C-level dot products
                    # (math.sumprod) for the batch-arrival convolutions in _pi / _occ_tau (same sums, rounding differs
                    # ~1e-16 rel); other chains keep the 0.60 loops bit-for-bit

try:
    _dot = math.sumprod                       # Python ≥ 3.12
except AttributeError:                        # pragma: no cover
    import operator as _op

    def _dot(a, b):
        return math.fsum(map(_op.mul, a, b))
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
                        scale=lambda r: 1.0, n_max: int | None = None, step_fn=None, batch=None) -> dict | None:
    """Continuous-batching decode as a birth–death process (0.54; validated against core/pdsim).

    State n = requests at the replica (running + waiting for a slot).  Every running sequence advances one engine
    step per iteration, so the replica is a processor-sharing server with state-dependent capacity: departure rate
    μ(n) = k·e / (out · step(k)), k = min(n, B).  Stationary π_n ∝ Π λ/μ(j) (exact for PS, insensitive to the
    output-length law); E[TPOT] = E[k] / (λ·out) by Little on the running set.  The 0.51–0.53 Little fixed point
    n̄ = λ·out·TPOT(⌈n̄⌉) ignored occupancy fluctuations and is optimistic when step(k) grows with k (KV reads) —
    e.g. −30 % mean TPOT at 80 % load in the §18.1 example.
    Request-average TPOT quantiles: a running request sees the size-biased batch (mean m_s, sd σ_s); over its
    lifetime (≈ one correlation time of the occupancy) the average batch has sd ≈ 0.86·σ_s (OU window) 「假设」.
    ``step_fn(k) → (step, Result)`` overrides the pool step (chunked prefill uses its mean iteration).
    ``batch = (λ_b, X, stretch)`` (0.55, colocated): the chain runs in a *reduced clock* that leaves out the time
    the prefill work adds (prefill-first: the whole prefill; chunked: n_c·(T₁ − T₀)).  In that clock a prefill busy
    period is an instant that hands its X requests to decode at once, so arrivals are batch-Poisson (rate λ_b, size
    pmf X[0] = P(X = 1), …) instead of Poisson.  The chain is skip-free downward, so level crossing is exact:
    π(n)·μ(n) = λ_b·Σ_{i<n} π(i)·P(X ≥ n − i).  Wall TPOT = E[k]/(λ·out) with λ the wall arrival rate; a seen step
    is stretched by ``stretch`` = 1/(1 − excess share) on average."""
    steps: dict = {}
    if n_max is None:
        n_max = max(4096, B + 4096)     # 0.60: was a fixed 4096 (truncated the chain for B ≥ 4096 at large card counts)

    def exact(k: int):
        if k not in steps:
            if step_fn is not None:
                steps[k] = step_fn(k)
            else:
                r = pool.run("decode", k)
                steps[k] = (r.step * scale(r) / share, r)
        return steps[k]

    if B <= INTERP_B:
        step_of = exact
    else:   # 0.60 「假设」: B > 1024 (large replicas) — step(k) exact for k ≤ 64, linear between geometric grid points
        grid = sorted({k for k in range(1, 65)} | {min(B, int(round(64 * 2 ** (j / 8)))) for j in range(1, 200)
                                                   if 64 * 2 ** ((j - 1) / 8) < B} | {B})
        interp: dict = {}

        def step_of(k: int):
            if k in interp:
                return interp[k]
            if k <= 64 or k >= B or k in grid:
                v = exact(min(k, B)) if k <= B else exact(B)
            else:
                i = bisect.bisect_left(grid, k)
                a, b = grid[i - 1], grid[i]
                (sa, _), (sb, rb) = exact(a), exact(b)
                v = (sa + (sb - sa) * (k - a) / (b - a), rb)
            interp[k] = v
            return v

    stB, rB = step_of(B)
    if not rB.fits or not math.isfinite(stB):
        return None
    e = rB.tokens_per_step
    if lam <= 0 or out <= 0:
        st1, r1 = step_of(1)
        return {"k": 1, "occupancy": 0.0, "r": r1, "tpot": st1 / e, "step": st1, "running_mean": 1.0,
                "pi_run": (1.0,), "seen_mean": 1.0, "seen_sd": 0.0, "step_of": step_of, "e": e}
    stretch = 1.0 if batch is None else batch[2]
    # 0.61 (B ≥ FAST_B): stop the chain once it is past its mode and below ~1e-20 of the peak instead of always
    # running to n > B (+ L): the dropped mass is < 1e-17 relative — at B = 4096 a lightly loaded chain went 4.9k
    # states deep for a mode in the hundreds, and every state k ≤ B cost an evaluate() when B ≤ INTERP_B
    trunc = B >= FAST_B

    def _pi(slow: float = 1.0):
        """Stationary π_n with every service rate divided by ``slow`` (0.57: restore stalls) — None if unstable."""
        if batch is None:
            lam_s = lam * slow
            if lam_s >= B * e / (out * stB):       # even a full batch cannot keep up
                return None
            logp = [0.0]
            m = 0.0                               # 0.61: running max (was max(logp) per step — O(N²))
            for n in range(1, n_max + 1):
                k = min(n, B)
                st, _ = step_of(k)
                if not math.isfinite(st):
                    return None
                x = logp[-1] + math.log(lam_s * out * st / (k * e))
                logp.append(x)
                if x > m:
                    m = x
                if n > B + 8 and x < m - 40:
                    break
                if trunc and x < m - 46 and x < logp[-2]:   # 0.61: past the mode and < e^-46 of it (large B only)
                    break
            w = [math.exp(x - m) for x in logp]
            Z = sum(w)
            return [x / Z for x in w]
        lam_b, X, _ = batch
        lam_b = lam_b * slow
        EX = sum((i + 1) * x for i, x in enumerate(X))
        if lam_b * EX >= B * e / (out * stB):
            return None
        tail = [0.0] * (len(X) + 1)          # tail[m] = P(X ≥ m), m = 1 … len(X)
        acc = 0.0
        for m_ in range(len(X), 0, -1):
            acc += X[m_ - 1]
            tail[m_] = acc
        L = len(X)
        w = [1.0]
        peak = 1.0
        fast = B >= FAST_B and L * (B + L) > FAST_NL
        below = 0
        rt = tail[::-1]                       # rt[j] = tail[L − j]: Σ_i w[i]·tail[n − i] = dot(w[lo:n], rt[L − n + lo:L])
        for n in range(1, n_max + 1):
            k = min(n, B)
            st, _ = step_of(k)
            if not math.isfinite(st):
                return None
            if fast:
                lo = max(0, n - L)
                up = _dot(w[lo:n], rt[L - n + lo:L])
            else:
                up = 0.0
                for i in range(max(0, n - L), n):
                    up += w[i] * tail[n - i]
            w.append(lam_b * up * out * st / (k * e))
            if w[-1] > peak:
                peak = w[-1]
            if peak > 1e250:
                w = [x / peak for x in w]
                peak = 1.0
            if n > B + L + 8 and w[-1] < peak * 1e-17:
                break
            if trunc:                         # 0.61: L straight weights < 1e-20 of the peak → every later up-sum is too
                below = below + 1 if w[-1] < peak * 1e-20 else 0
                if below > L:
                    break
        Z = sum(w)
        return [x / Z for x in w]
    pi = _pi(1.0)
    if pi is None:
        return None
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
    kmax = min(B, len(pi) - 1)                  # 0.61: step list once (was step_of() per k per pass, 4 passes)
    sv = [0.0] + [step_of(k)[0] for k in range(1, kmax + 1)]
    wsee = [run_w[k] * k / sv[k] if k else 0.0 for k in range(kmax + 1)] + [0.0] * (B - kmax)
    ws_tot = sum(wsee) or 1.0
    wsee = [x / ws_tot for x in wsee]
    seen_mean = sum(k * x for k, x in enumerate(wsee))
    seen_sd = math.sqrt(max(0.0, sum(k * k * x for k, x in enumerate(wsee)) - seen_mean ** 2))
    st_mean = sum(sv[k] * x for k, x in enumerate(wsee) if x)
    st_sd = math.sqrt(max(0.0, sum(sv[k] ** 2 * x for k, x in enumerate(wsee) if x) - st_mean ** 2))
    k_hat = max(1, min(B, int(round(seen_mean))))
    st, r = step_of(k_hat)
    # 0.55: lifetime-averaging factor from the occupancy's integrated autocorrelation time (exact for this chain)
    tau = _occ_tau(pi, B, (lam, [1.0]) if batch is None else batch[:2],
                   lambda k: k * e / (out * sv[k]))
    life = out * st_mean / e
    win = _WIN
    if tau and tau > 0 and life > 0:
        x_ = life / tau
        win = math.sqrt(2 * (x_ - 1 + math.exp(-x_)) / (x_ * x_)) if x_ > 1e-6 else 1.0
    return {"k": k_hat, "occupancy": EN, "r": r, "tpot": Ek / (lam * out), "step": st, "running_mean": k_mean,
            "tau_int": tau, "win": win, "pi_n": tuple(pi), "life": life,
            "mu": lambda n: min(n, B) * e / (out * step_of(min(n, B))[0]),
            "pi_run": pi_run, "seen_mean": seen_mean, "seen_sd": seen_sd, "w_seen": tuple(wsee), "step_seen_mean": st_mean,
            "pi_at": _pi,
            "step_seen_sd": st_sd, "step_of": step_of, "e": e, "stretch": stretch, "run_w": tuple(run_w)}




def _occ_tau(pi: list, B: int, arr, mu_of) -> float | None:
    """Integrated autocorrelation time of the running batch k = min(n, B) for the decode chain (0.55).

    Asymptotic variance of ∫k dt is 2·Σ_n d(n)·G(n), d(n) = h(n) − h(n−1) from the Poisson equation Qh = −(k − E k),
    G(n) = Σ_{i≥n} π_i (k_i − E k).  Skip-free downward (one departure at a time) → a cut at n gives
        π_n μ_n d(n) = G(n) − λ_b·Σ_{m>n} d(m)·Σ_{i<n} π_i P(X ≥ m − i)
    (birth–death: the coupling vanishes, d = G/(π μ)); solved from the top down in O(N·L) with the coupling sum
    carried by a recurrence.  τ = Σ d·G / Var(k)."""
    lam_b, X = arr
    N, L = len(pi), len(X)
    if N < 3:
        return None
    tail = [0.0] * (L + 2)
    acc = 0.0
    for m in range(L, 0, -1):
        acc += X[m - 1]
        tail[m] = acc
    kv = [min(n, B) for n in range(N)]
    Ek = sum(k * p for k, p in zip(kv, pi))
    var = sum(k * k * p for k, p in zip(kv, pi)) - Ek * Ek
    if var <= 1e-12:
        return None
    G = [0.0] * (N + 1)
    acc = 0.0
    for n in range(N - 1, -1, -1):
        acc += pi[n] * (kv[n] - Ek)
        G[n] = acc
    # 0.61.2: below the mean the top-down sum cancels to rounding noise (~1e-17), which d = G/(π·μ) then
    # divides by a left-tail π (e^-256 at a mean occupancy of 256) → τ_int ~1e80.  Σ_n π_n (k_n − E k) = 0, so
    # G(n) = −Σ_{i<n} π_i (k_i − E k) exactly — every term of that sum has one sign below the mean
    acc = 0.0
    for n in range(1, N):
        acc -= pi[n - 1] * (kv[n - 1] - Ek)
        if kv[n] >= Ek:
            break
        G[n] = acc
    d = [0.0] * (N + 1)
    T = 0.0                      # T(n) = Σ_{m>n} d(m) Σ_{i<n} π_i tail(m − i)
    sig = 0.0
    if L == 1:                   # 0.61: Poisson arrivals — the coupling sums are empty (T ≡ 0), same arithmetic as below
        for n in range(N - 1, 0, -1):
            mu = mu_of(min(n, B))
            if pi[n] > 1e-280 and mu > 0:
                d[n] = G[n] / (pi[n] * mu)
            sig += d[n] * G[n]
        return sig / var if sig > 0 else None
    if B >= FAST_B and N * L > FAST_NL:     # 0.61: the two O(L) sums as C-level dot products on slices (hand-vectorized)
        rt = tail[L + 1::-1]     # rt[j] = tail[L + 1 − j]
        for n in range(N - 1, 0, -1):
            if n < N - 1:
                hi = min(N, n + L + 1)
                if hi > n + 2:
                    T -= pi[n] * _dot(d[n + 2:hi], tail[2:hi - n])
                lo = max(0, n + 1 - L)
                if n > lo:
                    T += d[n + 1] * _dot(pi[lo:n], rt[L - n + lo:L])
            mu = mu_of(min(n, B))
            if pi[n] > 1e-280 and mu > 0:
                d[n] = (G[n] - lam_b * T) / (pi[n] * mu)
            sig += d[n] * G[n]
        return sig / var if sig > 0 else None
    for n in range(N - 1, 0, -1):
        if n < N - 1:            # T(n) from T(n+1)
            T -= pi[n] * sum(d[m] * tail[m - n] for m in range(n + 2, min(N, n + L + 1)))
            T += d[n + 1] * sum(pi[i] * tail[n + 1 - i] for i in range(max(0, n + 1 - L), n))
        mu = mu_of(min(n, B))
        if pi[n] > 1e-280 and mu > 0:
            d[n] = (G[n] - lam_b * T) / (pi[n] * mu)
        sig += d[n] * G[n]
    return sig / var if sig > 0 else None


def _tpot_req_q(dec: dict, q: float) -> float:
    """Request-average TPOT quantile: mean seen step + z·win·sd, win = lifetime-averaging factor of the occupancy
    (0.55: from its integrated autocorrelation time τ, exponential correlation 「假设」, √(2(x − 1 + e^{−x})/x²),
    x = life/τ; 0.54 used a fixed 0.86 = one correlation time), × the reduced-clock stretch (1 outside the 0.55
    colocated batch chain)."""
    return (dec["step_seen_mean"] + _Z[q] * dec.get("win", _WIN) * dec["step_seen_sd"]) * dec.get("stretch", 1.0) \
        / dec["e"]


def _mg1_busy(lam: float, svc_w, want_dur: bool = False, tol: float = 1e-9, n_cap: int = 400):
    """M/G/1 busy period with a discrete service law [(w, s)] (0.55, Takács): requests served X and duration V.
    P(X = n, V ∈ dt) = e^{−λt}(λt)^{n−1}/n! · dG^{*n}(t).  Returns (X pmf with X[0] = P(X = 1), duration law as a
    sorted [(t, mass)] list or None).  One service value → Borel(λs) exactly with V = n·s.  Mixed values: the n-fold
    service sum on a grid of mean/8 (each bin keeps its exact mean time) for n ≤ 60; beyond, a normal T_n (Simpson over ±6σ) (counts only; durations lumped at the mean, n ≫ 1 carries ≪ tol of the mass at the ρ
    the colocated modes run at).  None if ρ ≥ 1."""
    pts = [(w, s_) for w, s_ in svc_w if w > 0 and s_ > 0]
    tot = sum(w for w, _ in pts)
    pts = [(w / tot, s_) for w, s_ in pts]
    m1 = sum(w * s_ for w, s_ in pts)
    if lam <= 0 or m1 <= 0:
        return [1.0], ([(m1, 1.0)] if want_dur else None)
    if lam * m1 >= 1:
        return None, None
    X, dur = [], {} if want_dur else None

    def ker(t, n):
        return math.exp(-lam * t + (n - 1) * math.log(lam * t) - math.lgamma(n + 1)) if t > 0 else 0.0
    if len({s_ for _, s_ in pts}) == 1:
        s0 = pts[0][1]
        F = 0.0
        for n in range(1, n_cap + 1):
            p = ker(n * s0, n)
            X.append(p)
            if want_dur:
                dur[n * s0] = p
            F += p
            if 1 - F < tol:
                break
    else:
        dt = m1 / 8.0
        idx = [(w, s_, int(round(s_ / dt))) for w, s_ in pts]
        imax = max(i for _, _, i in idx)
        cm, ct, lo, hi = [1.0], [0.0], 0, 1       # bin k → mass, mass-weighted exact time (flat lists, 0.56)
        F = 0.0
        m2 = sum(w * s_ * s_ for w, s_ in pts)
        var = max(0.0, m2 - m1 * m1)
        # Simpson over ±6σ of a normal T_n (41 nodes)
        zs = [-6 + 12 * j / 40 for j in range(41)]
        zw = [(1 if j in (0, 40) else 4 if j % 2 else 2) * (12 / 40) / 3 * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
              for j, z in enumerate(zs)]
        for n in range(1, n_cap + 1):
            if n <= 60:
                nm = [0.0] * (hi + imax)
                nt = [0.0] * (hi + imax)
                rng_k = [k for k in range(lo, hi) if cm[k] > 0.0]
                for w, s_, i in idx:
                    for k in rng_k:
                        pk = cm[k]
                        nm[k + i] += pk * w
                        nt[k + i] += w * (ct[k] + pk * s_)
                cm, ct = nm, nt
                p = 0.0
                lo, hi = len(cm), 0
                for k in range(len(cm)):
                    v = cm[k]
                    if v <= 1e-13:
                        cm[k] = 0.0
                        continue
                    lo, hi = min(lo, k), k + 1
                    t_ = ct[k] / v
                    q_ = v * ker(t_, n)
                    p += q_
                    if want_dur and q_ > 0:
                        dur[t_] = dur.get(t_, 0.0) + q_
                if hi == 0:
                    lo = 0
            else:
                mu_n, sd_n = n * m1, math.sqrt(n * var)
                p = sum(wt * ker(max(1e-12, mu_n + sd_n * z), n) for z, wt in zip(zs, zw))
                if want_dur:
                    dur[mu_n] = dur.get(mu_n, 0.0) + p
            X.append(p)
            F += p
            if 1 - F < tol:
                break
    Z = sum(X)
    X = [x / Z for x in X]
    if want_dur:
        dur = sorted((t, v / Z) for t, v in dur.items())
    return X, dur


def _busy_spread(lam: float, svc_rew, X: list, lam_b: float) -> float:
    """Pair-term factor a² of reduced-clock arrivals when part of a busy period still advances decode (0.55, chunked).

    ``svc_rew = [(w, S, R)]``: per-request wall service S and the part R of it that is *not* excess (decode still
    advances, n_c·T₀).  A busy period then lasts W = ΣR > 0 in the reduced clock and its X joins are spread over it, so
    the input is a modulated Poisson process, not instantaneous batches.  Renewal–reward over cycles C = idle (Exp λ)
    + W gives its long-run variance rate σ² = Var(X − r·C)/E[C] (r = mean reduced rate), with the joint busy-period
    moments of (X, W) from the branching identity  Y = y₀ + Σ_{j ≤ N(S)} Y_j, N(S) ~ Poisson(λS):
        (1 − ρ)·E[Y_a Y_b] = E[y₀a y₀b] + λE[y₀a S]E[Y_b] + λE[y₀b S]E[Y_a] + λ²E[S²]E[Y_a]E[Y_b].
    a² = (σ² − r)/(λ_b·E[X(X−1)]) is the share of the instantaneous-batch pair term that survives (near-critical
    decode responds to the long-run dispersion 「假设」).  R ≡ 0 (prefill-first) → a² = 1."""
    tot = sum(w for w, _, _ in svc_rew)
    ES = sum(w * S for w, S, _ in svc_rew) / tot
    ES2 = sum(w * S * S for w, S, _ in svc_rew) / tot
    ER = sum(w * R for w, _, R in svc_rew) / tot
    ER2 = sum(w * R * R for w, _, R in svc_rew) / tot
    ERS = sum(w * R * S for w, S, R in svc_rew) / tot
    rho = lam * ES
    fact2 = sum((i + 1) * i * x for i, x in enumerate(X))
    if rho >= 1 or ER <= 0 or fact2 <= 0:
        return 1.0
    g = 1 - rho
    EX, EW = 1 / g, ER / g
    EX2 = (1 + 2 * lam * ES * EX + lam * lam * ES2 * EX * EX) / g
    EXW = (ER + lam * ES * EW + lam * ERS * EX + lam * lam * ES2 * EX * EW) / g
    EW2 = (ER2 + 2 * lam * ERS * EW + lam * lam * ES2 * EW * EW) / g
    EC = 1 / lam + EW
    r = EX / EC
    var = (EX2 - EX * EX) - 2 * r * (EXW - EX * EW) + r * r * (EW2 - EW * EW + 1 / lam ** 2)
    return min(1.0, max(0.0, (var / EC - r) / (lam_b * fact2)))


def _pair_kappa(out_w) -> float:
    """κ = 2∫P(O > t)²dt / E[O] for the output-length law [(w, O)] (0.55): how long two requests that join decode
    together stay together (processor sharing: all running sequences advance at the same token rate).  1 for an
    exponential law (what the birth–death chain implies), 2 for a fixed length (a batch leaves together)."""
    tot = sum(w for w, _ in out_w)
    pts = sorted((o, w / tot) for w, o in out_w if w > 0)
    if not pts:
        return 1.0
    mean = sum(o * w for o, w in pts)
    surv, prev, acc = 1.0, 0.0, 0.0
    for o, w in pts:
        acc += (o - prev) * surv * surv
        surv -= w
        prev = o
    return 2 * acc / mean if mean > 0 else 1.0


def _shape_batch(lam_b: float, X: list, rate: float, a2: float):
    """Batch-Poisson input (λ_b, X) re-shaped so its pair term λ_b·E[X(X−1)] scales by a² at the same mean rate
    ``rate`` (0.55).  a² < 1: binomial thinning (keep prob a) + Poisson singletons; a² > 1: a share p of batches merge
    with an independent copy, p = (a² − 1)F / (2E[X]² − (a² − 1)F) (capped at 1).  Returns (rate_b, pmf)."""
    if abs(a2 - 1) < 1e-9:
        return lam_b, X
    if a2 < 1:
        a = math.sqrt(max(0.0, a2))
        Y = [0.0] * (len(X) + 1)
        for i, x in enumerate(X):
            n = i + 1
            if x <= 0:
                continue
            if a <= 0:
                Y[0] += x
                continue
            pm = (1 - a) ** n                 # P(m = 0)
            Y[0] += x * pm
            for m in range(1, n + 1):
                pm *= (n - m + 1) / m * (a / (1 - a))
                Y[m] += x * pm
        nb, ns = lam_b * (1 - Y[0]), rate * (1 - a)
        nu = nb + ns
        pmf = [lam_b * y / nu for y in Y[1:]]
        pmf[0] += ns / nu
        return nu, pmf
    EX = sum((i + 1) * x for i, x in enumerate(X))
    F = sum((i + 1) * i * x for i, x in enumerate(X))
    den = 2 * EX * EX - (a2 - 1) * F
    p = 1.0 if den <= 0 else min(1.0, (a2 - 1) * F / den)
    L = len(X)
    pmf = [(1 - p) * x for x in X] + [0.0] * L
    for i, x in enumerate(X):
        if x <= 0:
            continue
        for j, y in enumerate(X):
            pmf[i + j + 1] += p * x * y          # size (i+1)+(j+1) → index i+j+1
    while len(pmf) > 1 and pmf[-1] < 1e-300:
        pmf.pop()
    return lam_b / (1 + p), pmf


def _excess_var(lam: float, svc_w, ratio: float, rho: float) -> float:
    """E[V²] of the excess time one busy period adds (V = ratio × busy-period duration; M/G/1: E[S²]/(1 − ρ)³)."""
    m2 = sum(w * s_ * s_ for w, s_ in svc_w) / max(1e-300, sum(w for w, _ in svc_w))
    return ratio * ratio * m2 / (1 - rho) ** 3 if rho < 1 else math.inf


def _kmax_life_q(dec: dict, q: float) -> int:
    """q-quantile of the largest running batch a request meets over its life (0.55).

    The chain moves down one request at a time, so the rate of up-crossings of level m (n ≤ m → n > m) is exactly
    π_{m+1}·μ_{m+1} (level-crossing balance).  Poisson clumping 「假设」: P(max over life ≤ m) ≈ F_seen(m)·
    exp(−L·π_{m+1}μ_{m+1}/F(m)), L = the request's life in the chain's clock.  0.54 took the seen law's q-point,
    which ignores the excursions during the life (V4: worst gap −12…−19 %)."""
    ws, pi, B = dec["w_seen"], dec.get("pi_n"), len(dec["w_seen"]) - 1
    if not pi:
        acc = 0.0
        for k, x in enumerate(ws):
            acc += x
            if x and acc >= q:
                return k
        return B
    L, mu = dec["life"], dec["mu"]
    Fs, F = 0.0, 0.0
    for m in range(0, B):
        Fs += ws[m]
        F += pi[m]
        if m + 1 >= len(pi):
            return m
        if F <= 0:
            continue
        up = pi[m + 1] * mu(m + 1)
        if Fs * math.exp(-L * up / F) >= q:
            return max(1, m)
    return B


def _gap_q(dec: dict, q: float) -> float:
    """Worst single token gap of a request at the q-quantile: one step at the largest batch it meets over its life."""
    return dec["step_of"](_kmax_life_q(dec, q))[0] / dec["e"]


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


def _pd_mode_base(ctx: dict, lam: float) -> dict:
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


def _busy_q_from(dur, m: float, q: float) -> float:
    """d with P(V ≤ d)^m = q from a discrete duration law [(t, mass)] (the longest of m busy periods)."""
    target = q ** (1.0 / m) if m > 0 else 0.0
    acc = 0.0
    for t, v in dur:
        acc += v
        if acc >= target - 1e-12:
            return t
    return dur[-1][0] if dur else 0.0


def _coloc_prefill_first_base(ctx: dict, lam: float) -> dict:
    cp, r_c, pts, out, B = ctx["cpool"], ctx["r_c"], ctx.get("pts_c", ctx["pts"]), ctx["out"], ctx["B"]
    lam_c = lam / r_c
    pre = _prefill_server(cp, lam_c, pts)
    res = {"lambda_rps": lam, "stable": pre is not None}
    if pre is None:
        return {**res, "why": "prefill 在此负载下不稳定"}
    rho = pre["wait"]["rho"]
    share = 1 - rho
    # 0.55: decode in the reduced clock (prefill time left out) with batch arrivals — each prefill busy period hands
    # its X requests (M/G/1 busy-period count over the per-request prefill service, Takács) to decode at once.
    # Busy periods start at λ(1 − ρ) in wall time = λ per reduced second.  0.54 used Poisson arrivals with
    # step/(1 − ρ), which misses this burstiness (DES: mean TPOT −10 %, p90 −13 … −43 %).
    svc = list(zip(pre["ws"], pre["taus"]))
    X, _ = _mg1_busy(lam_c, svc)
    dec = None
    if X is not None and share > 0:
        kappa = _pair_kappa([(w_, o_) for w_, _, o_ in ctx["len"].points])
        EXb = sum((i + 1) * x_ for i, x_ in enumerate(X))
        nu_b, Xb = _shape_batch(lam_c, X, lam_c * EXb, kappa)
        dec = _decode_birth_death(cp, lam_c, out, B, batch=(nu_b, Xb, 1 / share))
    if dec is None:
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
    life = out * dec["tpot"]
    # request-average TPOT spread (root-sum-square of independent parts): the occupancy deviation of the steps it
    # sees (stretched) and the stall time it meets — busy periods arrive at λ(1 − ρ) over its life and each adds V,
    # E[V²] = E[S²]/(1 − ρ)³ for cap 1 (clustered: 0.54 counted single batches, E[V²] = lat²).
    if pre["b"] <= 1:
        ev2 = _excess_var(lam_c, lats_w, 1.0, rho)
    else:
        c = -math.expm1(-lam_c * pre["lat"])
        if 1 - c < 1e-12:     # 0.60: c rounds to 1 at extreme λ (stable-rate bisection at large card counts) → was 1/0
            return {**res, "stable": False, "why": "prefill 批几乎从不空闲（到达间隔 ≪ prefill 延迟）"}
        ev2 = pre["lat"] ** 2 * (1 + c) / (1 - c) ** 2
    n_bp = lam_c * share * life
    pure = {q: (_tpot_req_q(dec, q) - _tpot_req_q(dec, 0.5)) for q in (0.9, 0.99)}
    req_q = {q: dec["tpot"] + math.sqrt(pure[q] ** 2 + (_Z[q] * math.sqrt(n_bp * ev2) / out) ** 2)
             for q in (0.9, 0.99)}
    slot = mdc_wait(lam_c, out * dec["tpot"], B, QS, cs2=ctx["out_cs2"])
    # longest token gap (p99 over requests of each request's worst gap): one decode step + the longest prefill busy
    # period met in its life.  Cap 1: exact M/G/1 busy-period duration (Takács, 0.55); cap > 1: batch count
    # geometric × latency mix (0.54).
    if pre["b"] <= 1:
        _, dur = _mg1_busy(lam_c, lats_w, want_dur=True)
        d99 = _busy_q_from(dur, n_bp, 0.99) if dur else math.inf
    else:
        d99 = _busy_dur_max_q(lam_c, lats_w, pre["b"], life, rho, 0.99)
    return {**res, "ttft": ttft, "tpot_mean": dec["tpot"], "tpot_p90": req_q[0.9], "tpot_p99": req_q[0.99],
            "itl_max": dec["step_seen_mean"] / dec["e"] + d99,
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"batch_cap": pre["b"], "ttft_b_ms": pre["lat"] * 1e3, "wait_ms": _ms(pre["wait"])},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B, "share": share,
                       "tpot_pure_ms": tp * 1e3, "slot_wait_ms": _ms(slot),
                       "burst_mean": sum((i + 1) * x for i, x in enumerate(X))},
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


def _coloc_chunked_base(ctx: dict, lam: float) -> dict:
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
    lasts = [max(0.0, (S - p) - (n - 1) * C) / max(1, S - p) for (_, S, p), n in zip(pts, ns)]

    # Mean iteration at running batch k: a share x of iterations carries a chunk (ν chunk iterations / s) →
    # tbar = T₀ / (1 − ν·(T₁ − T₀)), x = ν·tbar (0.52 per-iteration mean); birth–death uses tbar as step(k).
    memo: dict = {}

    def mean_step(k: int):
        if k not in memo:
            r = cp.run("decode", k)
            t1 = [_fused_step(r, p1, f, rr) for p1, (_, f, rr) in zip(pre1, plans)]
            # 0.55: per-prompt chunk sum — n−1 full chunks + the partial last one (as the engine runs it)
            tsum = [(m - 1) * t + _fused_step(r, p1, lf, rr) for p1, m, t, lf, rr in
                    zip(pre1, ns, t1, lasts, [pl[2] for pl in plans])]
            t1m = _wsum(ws, tsum) / _wsum(ws, ns)
            den = 1 - nu * (t1m - r.step)
            tbar = r.step / den if den > 0 else math.inf
            memo[k] = (tbar, r, t1, min(1.0, nu * tbar), t1m, tsum)
        return memo[k]

    tB, rB, _, xB, _, _ = mean_step(B)
    if not rB.fits or not math.isfinite(tB) or xB >= 1:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    dec0 = _decode_birth_death(cp, lam_c, out, B, step_fn=lambda k: mean_step(k)[:2])
    if dec0 is None:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    # 0.55: reduced clock without the chunk excess n_c·(T₁ − T₀).  In it a prefill busy period (the chunk server is
    # busy ρ_s = λ·Σw·n_c·T₁ of the wall time) hands its X requests to decode as one batch (M/G/1 busy-period
    # count, Takács); decode runs at its pure step T₀(k).  Batch rate = λ(1 − ρ_s)/(1 − f), f = λ·Σw·n_c·(T₁ − T₀).
    # T₁ − T₀ is evaluated at the batch the decodes see (two fixed-point passes from the 0.54 mean-iteration chain).
    k_s = max(1, min(B, int(round(dec0["seen_mean"]))))
    kappa = _pair_kappa([(w_, o_) for w_, _, o_ in ctx["len"].points])
    dec = None
    for _ in range(2):
        _, r, t1, x, t1m, tsum = mean_step(k_s)
        T0 = r.step
        serv = tsum
        exc = [t - m * T0 for m, t in zip(ns, tsum)]
        rho_s, f = lam_c * _wsum(ws, serv), lam_c * _wsum(ws, exc)
        if rho_s >= 1 or f >= 1:
            return {**res, "why": "分块 prefill 队列不稳定"}
        X, _ = _mg1_busy(lam_c, list(zip(ws, serv)))
        d = None
        if X is not None:
            lam_b = lam_c * (1 - rho_s) / (1 - f)
            a_thin = kappa * _busy_spread(lam_c, [(w_, sv, sv - ex) for w_, sv, ex in zip(ws, serv, exc)], X, lam_b)
            nu_b, Xb = _shape_batch(lam_b, X, lam_c / (1 - f), a_thin)
            d = _decode_birth_death(cp, lam_c, out, B, batch=(nu_b, Xb, 1 / (1 - f)))
        if d is None:
            break
        dec = d
        k_s = max(1, min(B, int(round(dec["seen_mean"]))))
    if dec is None:
        return {**res, "why": "分块 prefill + decode 在此负载下不稳定"}
    _, r, t1, x, t1m, tsum = mean_step(k_s)
    serv = tsum                                     # prefill service = Σ fused chunk iterations
    # TTFT: the chunk server's speed follows the running batch, which drifts slowly — mix the service over the
    # time-weighted batch law (8 equal-mass bins; 0.55, V4 showed the single seen-batch service misses the tail)
    run_w = dec["run_w"]
    kbins, acc_w, edges = [], 0.0, [(j + 0.5) / 8 for j in range(8)]
    ei = 0
    for k, pk in enumerate(run_w):
        acc_w += pk
        while ei < 8 and acc_w >= edges[ei] - 1e-12:
            kbins.append(k)
            ei += 1
    while len(kbins) < 8:
        kbins.append(len(run_w) - 1)
    serv_mix, ws_mix, envs = [], [], []
    for kraw in kbins:
        kb = max(1, kraw)
        tb = mean_step(kb)[5]
        # no running decode (k = 0) → no decode-only iteration to wait out
        envs.append((1 / 8, [(w_, t, t) for w_, t in zip(ws, tb)], mean_step(kb)[1].step if kraw > 0 else 0.0))
        for w_, t in zip(ws, tb):
            serv_mix.append(t)
            ws_mix.append(w_ / 8)
    w = mg1(lam_c, serv_mix, ws_mix, QS)
    if not w["stable"]:
        return {**res, "why": "分块 prefill 队列不稳定"}
    ttft = _ttft(w, list(zip(ws_mix, serv_mix)), conv=(lam_c, serv_mix))
    # 0.55: the batch drifts far slower than a prefill busy period (τ_int ≫ E[S]/(1 − ρ)) → quasi-static mixture of
    # per-batch M/G/1 queues (a slow batch is a slow server for a whole busy period), not one M/G/1 over the mix
    qmix = {q: mg1_mix_sum_quantile(lam_c, envs, q) for q in QS}
    if all(v is not None for v in qmix.values()):
        mean_mix = 0.0
        for _, pts_b, vac in envs:
            m1b = sum(w_ * t for w_, t, _ in pts_b)
            m2b = sum(w_ * t * t for w_, t, _ in pts_b)
            mean_mix += (lam_c * m2b / (2 * (1 - lam_c * m1b)) + m1b + vac / 2) / 8
        ttft = {"mean": mean_mix, **{f"p{q * 100:g}": v for q, v in qmix.items()}}
    # worst single gap: the chunk iteration at the largest batch met over the life (0.99 point, up-crossing clumping)
    t1_99 = mean_step(max(1, _kmax_life_q(dec, 0.99)))[2]
    slot = mdc_wait(lam_c, out * dec["tpot"], B, QS, cs2=ctx["out_cs2"])
    # request-average TPOT spread: occupancy deviation (stretched) ⊕ the chunk excess met over its life
    T0 = r.step
    exc = [t - m * T0 for m, t in zip(ns, tsum)]
    rho_s, f = lam_c * _wsum(ws, serv), lam_c * _wsum(ws, exc)
    ratio = _wsum(ws, exc) / max(1e-300, _wsum(ws, serv))
    ev2 = _excess_var(lam_c, list(zip(ws, serv)), ratio, rho_s)
    life = out * dec["tpot"]
    n_bp = lam_c * (1 - rho_s) * life
    pure = {q: (_tpot_req_q(dec, q) - _tpot_req_q(dec, 0.5)) for q in (0.9, 0.99)}
    req_q = {q: dec["tpot"] + math.sqrt(pure[q] ** 2 + (_Z[q] * math.sqrt(n_bp * ev2) / out) ** 2)
             for q in (0.9, 0.99)}
    return {**res, "stable": True, "ttft": ttft, "tpot_mean": dec["tpot"],
            "tpot_p90": req_q[0.9], "tpot_p99": req_q[0.99],
            "itl_max": max(t1_99) / dec["e"],
            "e2e_mean": ttft["mean"] + slot["mean"] + out * dec["tpot"],
            "prefill": {"chunk_tokens": C, "chunks": _wsum(ws, ns), "iter_ms": t1m * 1e3,
                        "iter_ms_env": [min(mean_step(max(1, k_))[4] for k_ in kbins) * 1e3,
                                        max(mean_step(max(1, k_))[4] for k_ in kbins) * 1e3],
                        "rho": lam_c * _wsum(ws, serv), "chunk_share": x, "wait_ms": _ms(w)},
            "decode": {"running_batch": dec["k"], "occupancy": dec["occupancy"], "slots": B,
                       "iter_ms_no_chunk": r.step * 1e3, "slot_wait_ms": _ms(slot), "excess_share": f,
                       "batch_pair": a_thin},
            "_dec_r": r, "_pre1": pre1, "_plans": plans, "_dec": dec}


def _kv_slots(ctx: dict, B: int) -> tuple[int, dict | None]:
    """Decode slots under ``pd.kv_policy`` (0.55): B_kv = ⌊K / footprint⌋, footprint = the mean KV tokens of a
    *running* sequence — a running set is size-biased by output length (it stays ∝ out): wait reserves S + out →
    E[S] + E[o²]/E[o]; recompute holds S + generated → E[S] + E[o²]/(2E[o]) 「假设」 (prompt and output independent)."""
    pol = ctx.get("kv_policy", "off")
    cap = ctx.get("kv_cap") or {}
    if pol == "off" or cap.get("tokens") is None:
        return B, None
    pts = ctx["len"].points
    tot = sum(w for w, _, _ in pts)
    ES = sum(w * S for w, S, _ in pts) / tot
    Eo = sum(w * o for w, _, o in pts) / tot
    Eo2 = sum(w * o * o for w, _, o in pts) / tot
    foot = ES + (Eo2 / Eo if pol == "wait" else Eo2 / (2 * Eo))
    K = cap["tokens"]
    Bkv = max(1, int(K // foot)) if foot > 0 else B
    Eo3 = sum(w * o ** 3 for w, _, o in pts) / tot
    VS = sum(w * S * S for w, S, _ in pts) / tot - ES * ES
    Eg = Eo2 / (2 * Eo)
    var_f = max(0.0, VS + Eo3 / (3 * Eo) - Eg * Eg)           # running footprint S + g, g ~ U(0, o), o size-biased
    return min(B, Bkv), {"policy": pol, "capacity_tokens": K, "capacity_GB": cap.get("capacity_GB"),
                         "source": cap.get("source"), "footprint_tokens": foot, "slots_kv": Bkv, "binds": Bkv < B,
                         "ES": ES, "Eg": Eg, "var_f": var_f}


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (series below a + 1, Lentz continued fraction above)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, d = a, 1.0 / a
        tot = d
        for _ in range(500):
            ap += 1
            d *= x / ap
            tot += d
            if abs(d) < abs(tot) * 1e-14:
                break
        return max(0.0, 1.0 - tot * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _wait_mix_sum_q(tq: dict, p: float, m_cond: float, c2: float = 1.0) -> dict:
    """TTFT ⊕ admission wait (0.56) 「假设」: W = 0 w.p. 1 − p, else a conditional wait of mean m_cond — exponential
    (c2 = 1) or, 0.58, Gamma with squared CV ``c2`` (tabulated once, 512 points).  The TTFT law is rebuilt from its
    quantiles (log-survival interpolated between p50 / p90 / p99 and extrapolated past p99 with the p90→p99 slope;
    linear from 0.6·p50 below the median) on a 64-point grid, then P(T + W > t) = S_T(t) + p·E[S_W(t − T); T ≤ t]
    is solved for each quantile."""
    if p <= 0 or m_cond <= 0:
        return dict(tq)
    if abs(c2 - 1.0) < 1e-9:
        def sw(y: float) -> float:
            return math.exp(-y / m_cond)
    else:
        k_, th = 1.0 / c2, m_cond * c2
        n_g = 512
        y_hi = th * (k_ + 12 * math.sqrt(k_) + 30)
        dy = y_hi / n_g
        tab = [_gammq(k_, i * dy / th) for i in range(n_g + 1)]

        def sw(y: float) -> float:
            if y <= 0:
                return 1.0
            u = y / dy
            i = int(u)
            if i >= n_g:
                return tab[-1] * math.exp(-(y - y_hi) / th)
            return tab[i] + (tab[i + 1] - tab[i]) * (u - i)
    p50, p90, p99 = tq["p50"], tq["p90"], tq["p99"]
    lo = 0.6 * p50

    def Q(u: float) -> float:
        if u <= 0.5:
            return lo + (p50 - lo) * u / 0.5
        ls = -math.log(1 - u)
        a, b = math.log(2), math.log(10)
        if ls <= b:
            return p50 + (p90 - p50) * (ls - a) / (b - a)
        return p90 + (p99 - p90) * (ls - b) / (math.log(100) - b)
    n = 64
    us = [(k + 0.5) / n for k in range(n - 1)]
    pts = [(1.0 / n, Q(u)) for u in us]
    tail = [(1.0 / n / 8, Q(1 - (1.0 / n) * (k + 0.5) / 8)) for k in range(8)]   # finer in the last cell
    pts += tail

    def surv(t: float) -> float:
        acc = 0.0
        for w, x in pts:
            acc += w * (1.0 if x > t else p * sw(t - x))
        return acc
    out = {"mean": tq["mean"] + p * m_cond}
    for q in QS:
        k = f"p{q * 100:g}"
        a, b = 0.0, max(x for _, x in pts) + m_cond * max(1.0, math.log(max(p, 1e-12) / (1 - q))) + 1e-9
        while surv(b) > 1 - q:
            b *= 2
        for _ in range(100):
            mid = 0.5 * (a + b)
            if surv(mid) > 1 - q:
                a = mid
            else:
                b = mid
        out[k] = b
    return out


def _kv_wrap(base, ctx: dict, lam: float, pool_key: str, reps_key: str) -> dict:
    """Run a mode with the KV-limited slot count (0.55; 0.56 adds the admission order, swap, the saturation preemption
    rate and the victims' tail) 「假设」.

    * Admission wait: the chain's queue behind B_kv (Little × Lee–Longton).  ``kv_admit`` = before_prefill (vLLM:
      KV is allocated before the prefill / the KV pull, so the wait is part of TTFT and of SLO goodput — folded in
      when the capacity binds) | after_prefill (0.55: reported apart).
    * recompute / swap preemptions: ν = ν_Rice (footprint up-crossings of K, 0.55) + ν_sat.  ν_sat: while requests
      queue for KV the scheduler refills greedily, leaving headroom h ~ U(0, S_next) for growth; growth at
      G = λ_r·E[o] tokens/s exhausts it before the next departure (rate λ_r) with probability e^{−h/E[o]} →
      ν_sat = p_wait·λ_r·E_S[(E[o]/S)(1 − e^{−S/E[o]})].
    * Each preemption stalls the replica for T_stall (recompute: re-prefill S̄ + ḡ; swap: KV out + in over the host
      link) → TPOT × 1/(1 − ν·T_stall).  A victim's gap = T_fix + Exp(1/λ_r) (waits for a departure to free room;
      T_fix = re-prefill, or swap-out + swap-in); per-request probability p_v = ν/λ_r enters the worst-gap and TPOT
      tails as a mixture: q-quantile ≥ T_fix + ln(p_v/(1 − q))/λ_r when p_v > 1 − q."""
    B0 = ctx["B"]
    B, kvi = _kv_slots(ctx, B0)
    if kvi is None:
        return base(ctx, lam)
    x = base({**ctx, "B": B} if B != B0 else ctx, lam)
    x["kv_cap"] = kvi
    if not x.get("stable"):
        return x
    dec = x.get("_dec") or {}
    pi0 = dec.get("pi_n") or ()
    lam_r = lam / ctx[reps_key]
    run0 = dec.get("tpot", 0.0) * lam_r * ctx["out"]                   # E[running] (Little)
    ek0 = sum(min(n, B) * q for n, q in enumerate(pi0))
    rs = run0 / ek0 if ek0 > 0 else 1.0                                 # chain clock → wall (1 for the PD chain)
    cs2 = ctx.get("out_cs2", 1.0)
    admit = ctx.get("kv_admit", "before_prefill")
    hold = kvi["binds"] and admit == "before_prefill" and pool_key == "cpool" and lam_r > 0 and run0 > 0
    # 0.58: the slot is taken at the boundary where the prompt starts prefilling (the queueing for the replica happens
    # before admission, at the head of the prompt queue) → hold = prefill service = TTFT mean − prefill queue wait
    # (0.57 used the whole TTFT mean)
    t_pre = max(0.0, x["ttft"].get("mean", 0.0) - ((x.get("prefill") or {}).get("wait_ms") or {}).get("mean", 0.0) / 1e3) \
        if hold else 0.0

    def slot_stats(slow: float):
        """(P(wait), mean wait W, E[running]) with the decode service slowed by ``slow`` (restore stalls, 0.57);
        a str = the reason it is unstable."""
        pi = pi0 if slow == 1.0 else (dec["pi_at"](slow) if dec.get("pi_at") else None)
        if not pi:
            return "KV 容量不足：恢复停顿（重算 / 换入换出）拖慢 decode 后槽位队列不稳定" if slow != 1.0 else (0.0, 0.0, 0.0)
        pw = sum(pi[B:]) if len(pi) > B else 0.0
        occ = sum(n * q for n, q in enumerate(pi))
        ek = sum(min(n, B) * q for n, q in enumerate(pi))
        run = run0 if slow == 1.0 else ek * rs
        w = (occ - (run0 if slow == 1.0 else ek)) / lam_r if lam_r > 0 else 0.0
        # The chain's queue behind the slots is M/M/c-like (memoryless lives).  0.55 / 0.56: Lee–Longton (1 + c_s²)/2.
        # 0.57: two-moment M/G/c — W ≈ c_s²·W_M/M/c + (1 − c_s²)·W_M/D/c for c_s² ≤ 1, with Cosmetatos' M/D/c
        # W_M/D/c ≈ ½·W_M/M/c / (1 + (1 − ρ)(c − 1)(√(4 + 5c) − 2)/(16ρc)) (many slots drain a queue of near-equal
        # lives faster than one server would); Lee–Longton above c_s² = 1.  ρ = E[running]/B.  c_s² = the output
        # length's (the slot is held ∝ out) 「假设」.
        rho_s = min(0.999, max(1e-6, run / B)) if B > 0 else 0.999
        cosm = 1.0 + (1 - rho_s) * (B - 1) * (math.sqrt(4 + 5 * B) - 2) / (16 * rho_s * B)
        w = max(0.0, cs2 * w + (1 - cs2) * 0.5 * w / cosm if cs2 <= 1 else w * (1 + cs2) / 2)
        if hold:
            # 0.57: a colocated request holds its slot from admission through its prefill queueing + prefill, not
            # just its decode life → slot load a′ = a + λ_r·T_pre (T_pre = this mode's mean TTFT without the slot
            # wait).  The chain (decode only) sees a; scale P(wait) and W by the M/M/c (Erlang C) ratios 「假设」.
            # (PD: the slot is taken just before the KV pull, a few ms — negligible.)
            a2 = run + lam_r * t_pre
            if a2 >= B:
                return "KV 槽位不足：先准入后 prefill 时，prefill 期间也占着 KV 槽"
            c1, c2 = erlang_c(B, run), erlang_c(B, a2)
            if c1 > 0:
                pw = min(1.0, pw * c2 / c1)
                w *= (c2 / c1) * (B - run) / (B - a2) * (run + lam_r * t_pre) / run
        return pw, w, run

    ss = slot_stats(1.0)
    if isinstance(ss, str):
        return {**x, "stable": False, "why": ss}
    p_wait, W, run = ss
    if hold:
        kvi["slot_hold_prefill_ms"] = t_pre * 1e3
    pol = kvi["policy"]
    if pol in ("recompute", "swap"):
        pool = ctx[pool_key]
        S_re = max(1, int(round(kvi["ES"] + kvi["Eg"])))
        if pol == "swap":
            sw = ctx.get("swap") or {}
            bps = sw.get("Bps_replica", 0.0)
            t_io = S_re * (ctx.get("kv_cap") or {}).get("bytes_per_token", 0.0) / bps if bps > 0 else math.inf
            t_stall, t_fix = 2 * t_io, 2 * t_io
            kvi.update(swap_GBps_card=sw.get("GBps_card"), swap_source=sw.get("source"), swap_ms=t_io * 1e3)
        else:
            # re-prefill of the mean footprint S̄ + ḡ (0.57 check: the DES's victims are young — mean footprint ≈ 0.8 of
            # S̄ + ḡ — which offsets the convexity of prefill in length; restore 92 ms DES vs 92 ms here, dense8b CV 1)
            t_stall = t_fix = pool.run("prefill", 1, S_re).ttft
            kvi.update(recompute_ms=t_stall * 1e3)
        K, m_f, v_f = kvi["capacity_tokens"], kvi["ES"] + kvi["Eg"], kvi["var_f"]
        nu_rice = 0.0
        for k_, pk in enumerate(dec.get("run_w") or ()):
            if k_ < 1 or pk <= 0:
                continue
            sd = math.sqrt(k_ * v_f) if v_f > 0 else 0.0
            if sd <= 0:
                continue
            z = (K - k_ * m_f) / sd
            if z > 12:
                continue
            nu_rice += pk * math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi)) * k_ * dec["e"] / dec["step_of"](k_)[0]
        pts = ctx["len"].points
        tot = sum(w for w, _, _ in pts)
        Eo = sum(w * o for w, _, o in pts) / tot
        p_ev = sum(w * (Eo / S) * (1 - math.exp(-S / Eo)) for w, S, _ in pts) / tot if Eo > 0 else 0.0
        # 0.57: the departure that refills also frees its own grown output o_d, so the headroom walks
        # h → h + o_d − G (not a fresh U(0, S) draw against G): P(G − o_d > h) = the 0.56 term × E_o[e^{−o/E[o]}]
        # (G ~ Exp(E[o]), memoryless) — ≈ 0.37 at fixed outputs, ≈ 0.5 for exponential ones.
        e_od = sum(w * math.exp(-o / Eo) for w, _, o in pts) / tot if Eo > 0 else 1.0
        p_ev *= e_od
        if base is _coloc_chunked_base:
            # 0.58 「假设」: chunked prefill admits one prompt at a time, when the chunk server frees up — a departure
            # met while a prompt is prefilling (probability ρ_pre) is not followed by a tight greedy refill (growth
            # first eats into the freed room), so only (1 − ρ_pre) of the refills leave the U(0, S) headroom
            rho_pre = min(0.95, max(0.0, (x.get("prefill") or {}).get("rho", 0.0)))
            p_ev *= 1 - rho_pre
            kvi["refill_tight_share"] = 1 - rho_pre
        # 0.57 recompute cascade: restores take a share f of the replica, every slot is held 1/(1 − f) longer → more
        # queueing for KV → more saturation preemptions → larger f.  Iterate f ← ν(f)·T_stall on the chain slowed by
        # 1/(1 − f) (smallest fixed point; diverges → unstable).
        f_r, nu_sat = 0.0, 0.0
        for _ in range(400):
            ss = slot_stats(1.0 / (1.0 - f_r))
            if isinstance(ss, str):
                return {**x, "stable": False, "why": ss}
            p_wait, W, run = ss
            nu_sat = p_wait * lam_r * p_ev if kvi["binds"] else 0.0
            f_new = (nu_rice + nu_sat) * t_stall
            if f_new >= 0.995:
                return {**x, "stable": False, "why": f"KV 容量不足：{pol} 抢占的恢复（重算 / 换入换出）占满副本"}
            if abs(f_new - f_r) < 1e-6:
                f_r = f_new
                break
            f_r = f_new
        nu = nu_rice + nu_sat
        p_v = min(1.0, nu / lam_r) if lam_r > 0 else 0.0                 # preemptions per request
        st = 1 / (1 - f_r)
        g_mean = t_fix + (1 / lam_r if lam_r > 0 else 0.0)
        tp_mean0 = x["tpot_mean"] * st
        x["tpot_mean"] = tp_mean0 + p_v * g_mean / ctx["out"]
        for k_, q in (("tpot_p90", 0.9), ("tpot_p99", 0.99)):
            v = x[k_] * st + p_v * g_mean / ctx["out"]
            if p_v > 1 - q and lam_r > 0:
                v = max(v, tp_mean0 + (t_fix + math.log(p_v / (1 - q)) / lam_r) / ctx["out"])
            x[k_] = v
        if p_v > 0.01 and lam_r > 0:
            x["itl_max"] = max(x["itl_max"], t_fix + math.log(p_v / 0.01) / lam_r)
        x["e2e_mean"] = x.get("e2e_mean", 0.0) + ctx["out"] * (x["tpot_mean"] - x["tpot_mean"] / st)
        kvi.update(preempt_per_req=p_v, preempt_rice=nu_rice / lam_r if lam_r > 0 else 0.0,
                   preempt_sat=nu_sat / lam_r if lam_r > 0 else 0.0, restore_share=f_r,
                   victim_gap_mean_ms=g_mean * 1e3, refill_offset_factor=e_od, slot_slowdown=st)
        if pol == "recompute":
            kvi["recompute_share"] = f_r
    else:
        kvi["preempt_per_req"] = 0.0
    kvi.update(p_wait=p_wait, slot_wait_mean_ms=W * 1e3, admit=admit,
               in_ttft=bool(kvi["binds"] and admit == "before_prefill"))
    if kvi["in_ttft"] and p_wait > 0 and W > 0:
        # 0.58 wait shape 「假设」 (DES-calibrated): PD's conditional wait is exponential-like (DES c² ≈ 0.6, but the
        # Exp fold scores better there); colocated slot waits come in clumps (a prefill batch / chunked prompt takes
        # slots together) and spread with the hold variability: DES c² ≈ 1.1 at CV 0, 1.6–1.7 at CV 1 → 1 + 0.7·c_s².
        c2w = 1.0 if pool_key == "dpool" else 1.0 + 0.7 * min(2.0, max(0.0, cs2))
        x["ttft"] = _wait_mix_sum_q(x["ttft"], min(1.0, p_wait), W / min(1.0, p_wait), c2w)
        kvi["wait_c2"] = c2w
        x["e2e_mean"] = x.get("e2e_mean", 0.0) + W
    return x


def _pd_mode(ctx: dict, lam: float) -> dict:
    return _kv_wrap(_pd_mode_base, ctx, lam, "dpool", "r_d")


def _coloc_prefill_first(ctx: dict, lam: float) -> dict:
    return _kv_wrap(_coloc_prefill_first_base, ctx, lam, "cpool", "r_c")


def _coloc_chunked(ctx: dict, lam: float) -> dict:
    return _kv_wrap(_coloc_chunked_base, ctx, lam, "cpool", "r_c")


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


def _energy(counts_tok: dict, card_s_tok: float, table: EnergyTable | None, prefill_card_s_tok: float = 0.0) -> dict:
    """Dynamic energy = action counts × the table; static energy = W × card-seconds per output token at this load
    (wall-clock, busy or idle — the cards are provisioned either way).  ``prefill_card_s_tok``: the PD prefill pool's
    share of ``card_s_tok``, charged at ``idle_W_prefill`` when given (its chip may differ, 0.56); nothing defaulted."""
    out = {"counts_per_token": counts_tok, "card_s_per_token": card_s_tok}
    if table is None or not table.provided:
        return out
    j = {}
    for k, attr in _UNIT_PJ.items():
        e = getattr(table, attr)
        if e is not None:
            j[k] = counts_tok.get(k, 0.0) * e * 1e-12 * (8 if k in _BITS else 1)
    split = prefill_card_s_tok > 0 and table.idle_W_prefill is not None     # else one rate for every card (0.51)
    if split:
        j["idle_prefill"] = table.idle_W_prefill * prefill_card_s_tok
    if table.idle_W is not None:
        j["idle"] = table.idle_W * (card_s_tok - (prefill_card_s_tok if split else 0.0))
    if table.idle_W is None and table.idle_W_prefill is not None:
        out["static_note"] = ("只计 PD prefill 池的静态能耗：未给 idle_W，decode 池未计" if prefill_card_s_tok > 0 else
                              "未给 idle_W（idle_W_prefill 只用于 PD prefill 池），静态能耗未计")
    if not j:
        return out
    tot = sum(j.values())
    idle = j.get("idle", 0.0) + j.get("idle_prefill", 0.0)
    out.update(J_per_token=tot, J_by_action=j, tok_per_J=1 / tot if tot > 0 else None,
               static_share=idle / tot if tot > 0 else None)
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
    ks = [k for k in range(1, cards // c_p + 1) if cards - k * c_p >= c_d and not (cards - k * c_p) % c_d]
    done: dict = {}

    def run(k: int):
        nonlocal best
        if k in done:
            return done[k]
        np_, nd_ = k * c_p, cards - k * c_p
        cx = {**ctx, "n_p": np_, "n_d": nd_, "r_p": np_ // c_p, "r_d": nd_ // c_d}
        rate = _slo_rate(_pd_mode, cx, max(lam_fluid, lam), ttft_slo, tpot_slo)
        row = done[k] = {"prefill_cards": np_, "decode_cards": nd_, "slo_rate_rps": rate, "slo_goodput_per_card": rate * o / cards}
        if best is None or rate > best["slo_rate_rps"] or (rate == best["slo_rate_rps"] and np_ < best["prefill_cards"]):
            best = row
        return row

    if len(ks) <= SPLIT_EXACT:
        for k in ks:
            splits.append(run(k))
    else:   # 0.60 (large pools, many small replicas): 16 evenly spaced splits, then every split between the best
        idx = sorted({round(i * (len(ks) - 1) / 15) for i in range(16)})     # one's neighbours 「假设」 unimodal
        for i in idx:
            run(ks[i])
        j = ks.index(best["prefill_cards"] // c_p)
        lo_, hi_ = max(0, j - (len(ks) - 1) // 15 - 1), min(len(ks) - 1, j + (len(ks) - 1) // 15 + 1)
        for i in range(lo_, hi_ + 1):
            run(ks[i])
        splits = [done[k] for k in sorted(done)]
        out["pd_slo_splits_sampled"] = {"evaluated": len(done), "candidates": len(ks)}
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
                           cards / lam / o, table, ctx["n_p"] / lam / o)
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
