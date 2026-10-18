"""Closed-form queueing helpers for the PD report (0.51) — standard results, no simulator.

M/D/1 (Poisson arrivals λ, deterministic service τ, ρ = λτ < 1)
  mean wait            W̄ = ρτ / (2(1 − ρ))                         (Pollaczek–Khinchine, exact)
  P(W > 0)             = ρ                                           (exact)
  tail                 P(W > t) ≈ C·e^(−θt),  θτ = x > 0 root of ρ(e^x − 1) = x,  C = (1 − ρ) / (ρe^x − 1)
                       (Cramér–Lundberg asymptote of the M/G/1 waiting time: the dominant pole of the P-K transform;
                       tests check p90 / p99 against a Lindley-recursion simulation)
  quantile             w_q = 0 if ρ ≤ 1 − q, else max(0, ln(C / (1 − q)) / θ)
M/D/c slot wait (decode slots) — Erlang C of M/M/c with the Allen–Cunneen factor (c_a² + c_s²)/2 = ½ for
deterministic service: P(wait) ≈ C(c, a), W̄ ≈ ½·C(c, a) / (cμ − λ); conditional wait ~ exponential with that
mean 「假设」.
Poisson occupancy: in-service count of an M/G/∞ (continuous-batching slots far from full) ~ Poisson(n̄).
M/G/1 with a discrete service distribution {τ_i, w_i} (0.52, variable request lengths)
  mean wait            W̄ = λ·E[τ²] / (2(1 − ρ))                     (Pollaczek–Khinchine, exact)
  tail                 P(W > t) ≈ C·e^(−θt),  θ > 0 root of λ(E[e^(θτ)] − 1) = θ,  C = (1 − ρ) / (λ·E[τ e^(θτ)] − 1)
                       (same Cramér–Lundberg asymptote; one point → M/D/1 exactly — delegated to ``md1``)
M/G/c slot wait: Allen–Cunneen factor (1 + c_s²) / 2 with c_s² of the slot holding time.
"""

from __future__ import annotations

import math


def md1_theta(rho: float) -> float:
    """x = θτ > 0 with ρ(e^x − 1) = x (0 < ρ < 1)."""
    f = lambda x: rho * math.expm1(x) - x
    lo, hi = 0.0, 1.0
    while f(hi) <= 0:
        lo, hi = hi, hi * 2
        if hi > 1e4:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


def md1(lam: float, tau: float, qs: tuple = (0.5, 0.9, 0.99)) -> dict:
    """Waiting time (before service) of an M/D/1 queue; ``stable`` False when ρ ≥ 1 (then waits are inf)."""
    rho = lam * tau
    if lam <= 0 or tau <= 0:
        return {"rho": max(rho, 0.0), "stable": True, "mean": 0.0, "p_wait": 0.0, **{f"p{_pct(q)}": 0.0 for q in qs}}
    if rho >= 1:
        return {"rho": rho, "stable": False, "mean": math.inf, "p_wait": 1.0, **{f"p{_pct(q)}": math.inf for q in qs}}
    x = md1_theta(rho)
    theta = x / tau
    c = (1 - rho) / (rho * math.exp(x) - 1)
    out = {"rho": rho, "stable": True, "mean": rho * tau / (2 * (1 - rho)), "p_wait": rho}
    for q in qs:
        if rho <= 1 - q:
            out[f"p{_pct(q)}"] = 0.0
        elif md1_cdf(lam, tau, MD1_EXACT_SPAN * tau) >= q:
            # 0.55: near the wait atom the asymptotic c·e^{−θx} is poor (V4: +20 … +30 % TTFT p90 at ρ ≈ 1 − q);
            # use Erlang's exact CDF there
            lo, hi = 0.0, MD1_EXACT_SPAN * tau
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if md1_cdf(lam, tau, mid) >= q:
                    hi = mid
                else:
                    lo = mid
            out[f"p{_pct(q)}"] = hi
        else:
            out[f"p{_pct(q)}"] = max(0.0, math.log(c / (1 - q)) / theta)
    return out


MD1_EXACT_SPAN = 6      # exact CDF used for waits ≤ 6 service times (alternating sum stays well-conditioned)


def md1_cdf(lam: float, tau: float, x: float) -> float:
    """Erlang's exact M/D/1 waiting-time CDF: P(W ≤ x) = (1 − ρ)·Σ_{k=0}^{⌊x/τ⌋} [λ(kτ − x)]^k / k! · e^{−λ(kτ − x)}."""
    rho = lam * tau
    if x < 0:
        return 0.0
    acc = 0.0
    for k in range(int(x // tau) + 1):
        u = lam * (k * tau - x)
        acc += (u ** k) / math.factorial(k) * math.exp(-u)
    return min(1.0, max(0.0, (1 - rho) * acc))


def mg1(lam: float, taus, weights, qs: tuple = (0.5, 0.9, 0.99)) -> dict:
    """Waiting time of an M/G/1 queue whose service time takes value taus[i] with probability weights[i]."""
    pts = [(w, t) for w, t in zip(weights, taus) if w > 0]
    tot = sum(w for w, _ in pts)
    pts = [(w / tot, t) for w, t in pts]
    if len({t for _, t in pts}) == 1:
        return md1(lam, pts[0][1], qs)
    m1 = sum(w * t for w, t in pts)
    rho = lam * m1
    if lam <= 0 or m1 <= 0:
        return {"rho": max(rho, 0.0), "stable": True, "mean": 0.0, "p_wait": 0.0, **{f"p{_pct(q)}": 0.0 for q in qs}}
    if rho >= 1:
        return {"rho": rho, "stable": False, "mean": math.inf, "p_wait": 1.0, **{f"p{_pct(q)}": math.inf for q in qs}}
    m2 = sum(w * t * t for w, t in pts)
    tmax = max(t for _, t in pts)

    def f(th: float) -> float:
        return lam * (sum(w * math.exp(th * t) for w, t in pts) - 1) - th
    lo, hi = 0.0, 1.0 / tmax
    while f(hi) <= 0:
        lo, hi = hi, hi * 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13 * hi:
            break
    th = 0.5 * (lo + hi)
    out = {"rho": rho, "stable": True, "mean": lam * m2 / (2 * (1 - rho)), "p_wait": rho}
    sw, x_end, _ = _pk_surv(lam, pts)       # 0.56: exact law (was the Cramér–Lundberg tail c·e^{−θx})
    for q in qs:
        out[f"p{_pct(q)}"] = 0.0 if rho <= 1 - q else _surv_inv(sw, x_end, th, 1 - q)
    return out


def _surv_inv(sw, x_end: float, th: float, p: float) -> float:
    """x with P(W > x) = p for a survival from ``_pk_surv`` (decreasing; exponential beyond x_end)."""
    tail = sw(x_end)
    if tail <= p:
        a, b = 0.0, x_end
        for _ in range(100):
            mid = 0.5 * (a + b)
            if sw(mid) > p:
                a = mid
            else:
                b = mid
            if b - a < 1e-12 * max(b, 1e-12):
                break
        return b
    return x_end + math.log(tail / p) / th



_PK_CACHE: dict = {}
PK_GRID = 800             # grid points over [0, 4·τ_max] for the exact M/G/1 wait law (0.56)


def _pk_surv(lam: float, pts):
    """Exact M/G/1 waiting-time survival P(W > x) for a discrete service law [(w, τ)] (0.56, Pollaczek–Khinchine).

    W = sum of a Geometric(ρ) number of equilibrium residuals, density P(S > x)/E[S] — a step function for a discrete
    law — so its density solves the renewal equation g(x) = λ·Σ_i w_i·[G(x) − G(x − τ_i)] (G with the atom 1 − ρ at
    0), marched on a grid of ``PK_GRID`` points up to 4·τ_max (trapezoid, implicit in G(x); G(x − τ_i) interpolated).
    Beyond the grid the Cramér–Lundberg decay e^{−θx} continues from the grid's last value.  The 0.54 / 0.55 law
    min(ρ, c·e^{−θx}) is exact only in the tail; when the service times spread over ~100× (CV 1 lengths) it puts too
    little mass at short waits (behind short jobs), which biased TTFT p50 up by 15–30 % (V4 0.56 with 15 bins).
    Returns (surv, x_end, θ) or None when unstable.  Cached per (λ, law)."""
    key = (lam, tuple(pts))
    hit = _PK_CACHE.get(key)
    if hit is not None:
        return hit
    cl = _cl_params(lam, pts)
    if cl is None:
        return None
    rho, th, _ = cl
    tmax = max(t for _, t in pts)
    m1 = sum(w * t for w, t in pts)
    x_end = 4.0 * tmax
    n = PK_GRID
    h = x_end / n
    lh = lam * h
    if lh >= 0.5:                          # keep the implicit step well conditioned (never for real service laws)
        n = int(math.ceil(2 * lam * x_end)) + 1
        h = x_end / n
        lh = lam * h
    G = [0.0] * (n + 1)
    G[0] = 1.0 - rho
    # G = (1 − ρ)·1[x ≥ 0] + H(x), H continuous with H(0) = 0.  Over [x_{k−1}, x_k]:
    #   ∫g = λ·[trap(G) − trap(Σ_i w_i H(x − τ_i)) − (1 − ρ)·Σ_i w_i·|[x_{k−1}, x_k] ∩ [τ_i, ∞)|]
    # (the atom's step is integrated exactly, so the trapezoid only sees continuous integrands → 2nd order).
    # x_k − τ_i = (k − c_i + f_i)·h with c_i = ⌈τ_i/h⌉, f_i = c_i − τ_i/h.
    a0 = 1.0 - rho
    sh = []
    for w, t in pts:
        c = max(1, math.ceil(t / h - 1e-12))
        sh.append((w, c, min(1.0, max(0.0, c - t / h)), t))
    den = 1.0 - 0.5 * lh
    hs_prev = 0.0                          # Σ_i w_i H(x_{k−1} − τ_i)
    for k in range(1, n + 1):
        hs = 0.0
        step = 0.0
        xk = k * h
        for w, c, f, t in sh:
            j = k - c
            if j >= 0:
                g0 = G[j] - a0 if j > 0 else 0.0
                g1 = (G[j + 1] if j + 1 < k else G[k - 1]) - a0
                hs += w * (g0 + (g1 - g0) * f)
            step += w * min(h, max(0.0, xk - t))
        G[k] = (G[k - 1] * (1.0 + 0.5 * lh) - 0.5 * lh * (hs_prev + hs) - lam * a0 * step) / den
        hs_prev = hs
    tail = max(0.0, 1.0 - G[n])

    def surv(x: float) -> float:
        if x < 0:
            return 1.0
        if x >= x_end:
            return tail * math.exp(-th * (x - x_end))
        y = x / h
        j = min(int(y), n - 1)            # 0.238: x just below x_end can round to y = n
        return max(0.0, 1.0 - (G[j] + (G[j + 1] - G[j]) * (y - j))) if x > 0 else rho
    out = (surv, x_end, th)
    if len(_PK_CACHE) > 4096:
        _PK_CACHE.clear()
    _PK_CACHE[key] = out
    return out


def mg1_sum_quantile(lam: float, taus, weights, lats, q: float) -> float | None:
    """q-quantile of W + L_i (0.54): W = M/G/1 wait over service mix ``taus`` (0.56: exact Pollaczek–Khinchine law,
    ``_pk_surv``; 0.54 / 0.55 used the Cramér–Lundberg tail min(ρ, c·e^{−θx})), L_i = the request's own latency,
    drawn with the SAME index i as its service (one prompt length → its service and its latency).  W is independent
    of the arriving request's own i, so P(T > t) = Σ w_i·P(W > t − L_i).  Returns None when the service mix is a
    single value (caller keeps the exact wait-quantile + latency sum) or the queue is unstable."""
    pts = [(w, t, l) for w, t, l in zip(weights, taus, lats) if w > 0]
    if len({t for _, t, _ in pts}) <= 1 or lam <= 0:
        return None
    tot = sum(w for w, _, _ in pts)
    pts = [(w / tot, t, l) for w, t, l in pts]
    pk = _pk_surv(lam, [(w, t) for w, t, _ in pts])
    if pk is None:
        return None
    sw, x_end, th = pk

    def surv(t: float) -> float:
        return sum(w * sw(t - l) for w, _, l in pts)
    a = min(l for _, _, l in pts)
    tail = sw(x_end)
    b = max(l for _, _, l in pts) + x_end + (max(0.0, math.log(tail / (1 - q)) / th) if tail > 0 else 0.0) + 1e-9
    if surv(a) <= 1 - q:
        return a
    for _ in range(200):
        mid = 0.5 * (a + b)
        if surv(mid) > 1 - q:
            a = mid
        else:
            b = mid
        if b - a < 1e-12 * max(b, 1e-12):
            break
    return b


def _corr_surv(lam: float, svc, rows, theta: float = math.inf):
    """0.238: the survival P(W + L > t) of mg1_corr_sum_quantile as a function (and its bracket) — shared by the
    single-environment and the drifting-batch mixture forms.  → (surv(t, u), lo, hi) or None when unstable."""
    pts = [(w, t) for w, t in svc if w > 0]
    tot = sum(w for w, _ in pts)
    pts = [(w / tot, t) for w, t in pts]
    rho = lam * sum(w * t for w, t in pts)
    if lam <= 0 or rho >= 1:
        return None
    rows = [r if len(r) >= 6 else (*r, 0, False) for r in rows]
    mt = sum(r[0] for r in rows)
    ty_m, lead_m, fol_m = {}, {}, {}
    for m, a, c, sp, i, fo in rows:
        ty_m[i] = ty_m.get(i, 0.0) + m
        (fol_m if fo else lead_m)[i] = (fol_m if fo else lead_m).get(i, 0.0) + m
    # per class: (weight within its W-region, a, c, S′, region) — region 0: all w, 1: w ≤ θ, 2: w > θ
    rs = []
    for m, a, c, sp, i, fo in rows:
        if m <= 0:
            continue
        wi = ty_m[i] / mt
        if not fol_m.get(i) or not math.isfinite(theta):
            if fo:
                continue
            rs.append((wi * m / lead_m[i], a, c, sp, 0))
        elif fo:
            rs.append((wi * m / fol_m[i], a, c, sp, 2))
        else:
            rs.append((wi * m / lead_m[i], a, c, sp, 1))
    if len({t for _, t in pts}) == 1:
        pk = None
    else:
        pk = _pk_surv(lam, pts)
        if pk is None:
            return None
    # discrete wait law: the atom 1 − ρ at 0 + grid masses out to P(W > x) < 1e-8
    if pk is None:
        tau = pts[0][1]

        cl_md1 = _cl_params(lam, pts)

        def sw(x: float) -> float:
            if x < 0:
                return 1.0
            if x <= MD1_EXACT_SPAN * tau:
                return 1.0 - md1_cdf(lam, tau, x)
            # 0.239: beyond 6 τ Erlang's alternating series loses all precision (P(W > 40τ) came out 1.0 at ρ = 0.95,
            # long before the 0.238 overflow) → the Cramér–Lundberg tail, as mg1_mix_sum_quantile does
            r_, t_, c_ = cl_md1
            return min(r_, c_ * math.exp(-t_ * x))
        th = None
        x_hi = tau * 4
        while sw(x_hi) > 1e-8 and x_hi < tau * 400:
            x_hi *= 1.5
    else:
        sw, x_end, th = pk
        tail = sw(x_end)
        x_hi = x_end + (math.log(tail / 1e-8) / th if tail > 1e-8 else 0.0)
    n = 1500
    h = x_hi / n
    xs, mw = [0.0], [1.0 - rho]                  # the atom at 0, then cell midpoints
    prev = rho
    for k in range(1, n + 1):
        cur = sw(k * h)
        xs.append((k - 0.5) * h)
        mw.append(max(0.0, prev - cur))
        prev = cur
    # suffix sums: S1[j] = P(W ≥ xs[j]) on the grid, SE[j] = E[e^{−λW}; W ≥ xs[j]]
    S1, SE = [0.0] * (n + 2), [0.0] * (n + 2)
    for j in range(n, -1, -1):
        S1[j] = S1[j + 1] + mw[j]
        SE[j] = SE[j + 1] + mw[j] * math.exp(-lam * xs[j])

    def idx(y: float) -> int:                    # first grid point with x > y
        if y < 0:
            return 0
        return min(n + 1, int(y / h + 0.5) + 1)

    jt = idx(theta) if math.isfinite(theta) else n + 1     # grid points with x > θ start at jt

    def part(arr, j, region):                # Σ over grid points ≥ j inside the class's W-region
        if region == 1:
            return arr[min(j, jt)] - arr[jt]
        if region == 2:
            return arr[max(j, jt)]
        return arr[j]

    def surv(t: float, u: float = 0.0) -> float:
        # u: a vacation residual added to the wait (it counts for topping, not for follower status)
        t -= u
        acc = 0.0
        for m, a, c, sp, rg in rs:
            ja = idx(t - a)
            if not math.isfinite(sp):
                acc += m * part(S1, ja, rg)
                continue
            e_sp = math.exp(-lam * (sp + u))
            jc = idx(t - c)
            acc += m * (e_sp * part(SE, ja, rg) + part(S1, jc, rg) - e_sp * part(SE, jc, rg))
        return acc
    lo = min(min(a, c) for _, a, c, _, _ in rs)
    hi = max(max(a, c) for _, a, c, _, _ in rs) + x_hi + 1e-9
    return surv, lo, hi


def mg1_corr_sum_quantile(lam: float, svc, rows, q: float, theta: float = math.inf) -> float | None:
    """0.237: q-quantile of W + L when the request's own latency depends on its own wait.  A prompt's last iteration
    is topped up with the next prompt's head iff someone arrived behind it before that iteration starts — during its
    wait W and its first iterations S′ (FCFS: everyone arriving after it waits behind it) — so given W = w the shared
    latency has probability π(w) = 1 − e^{−λ(w + S′)} (Poisson arrivals).  Its mean over the M/G/1 wait law is the
    0.235 P-K LST π, E[e^{−λW}] = (1 − ρ)/E[e^{−λS}], so the marginals are unchanged — only the correlation is added
    (independent convolution, mg1_sum_quantile, puts the long shared latencies on short waits as often as on long).
    Followers (a prompt that starts inside its predecessor's last iteration) are those whose wait exceeds the
    predecessor's last-piece share ``theta``: given W = w a prompt of type i is a leader (its φ = 0 rows) for w ≤ θ
    and a follower (its φ > 0 rows, in their stationary proportions) for w > θ; a type without follower rows keeps
    its leader rows at every w.
    ``svc`` = [(w, τ)] the server-share law (the wait), ``rows`` = [(m, a, c, S′, type, follower)] per class: mass,
    latency alone, latency topped up, S′ (∞: never topped), prompt type, phase > 0.  None when unstable."""
    built = _corr_surv(lam, svc, rows, theta)
    if built is None:
        return None
    surv, a0, b0 = built
    if surv(a0) <= 1 - q:
        return a0
    for _ in range(80):
        mid = 0.5 * (a0 + b0)
        if surv(mid) > 1 - q:
            a0 = mid
        else:
            b0 = mid
        if b0 - a0 < 1e-9 * max(b0, 1e-12):
            break
    return b0


def mg1_corr_mix_sum_quantile(lam: float, bins, q: float) -> float | None:
    """0.238: the wait-correlated TTFT under a slowly drifting running batch (fused chunked prefill: vLLM,
    sglang_mixed) — the quasi-static mixture of mg1_mix_sum_quantile with each environment's survival from
    _corr_surv.  ``bins`` = [(w_b, svc, rows, θ, v_b)]; the vacation residual U(0, v_b) (a decode-only iteration in
    flight when the prompt queue is empty, Fuhrmann–Cooper) adds to the wait for topping (arrivals during it queue
    behind the prompt) but not to the follower test (an empty queue makes a leader) 「近似」."""
    envs = []
    for wb, svc, rows, theta, v in bins:
        if wb <= 0:
            continue
        built = _corr_surv(lam, svc, rows, theta)
        if built is None:
            return None
        envs.append((wb, built, v))
    if not envs:
        return None
    wtot = sum(e[0] for e in envs)
    U = 8

    def surv(t: float) -> float:
        acc = 0.0
        for wb, (sf, _, _), v in envs:
            if v > 0:
                acc += wb * sum(sf(t, v * (j + 0.5) / U) for j in range(U)) / U
            else:
                acc += wb * sf(t)
        return acc / wtot
    a = min(e[1][1] for e in envs)
    b = max(e[1][2] for e in envs) + max(e[2] for e in envs)
    if surv(a) <= 1 - q:
        return a
    for _ in range(80):
        mid = 0.5 * (a + b)
        if surv(mid) > 1 - q:
            a = mid
        else:
            b = mid
        if b - a < 1e-9 * max(b, 1e-12):
            break
    return b


def _cl_params(lam: float, pts) -> tuple[float, float, float] | None:
    """(ρ, θ, c) of the M/G/1 Cramér–Lundberg wait tail P(W > x) ≈ c·e^{−θx} for service law [(w, τ)]."""
    m1 = sum(w * t for w, t in pts)
    rho = lam * m1
    if rho >= 1 or m1 <= 0:
        return None
    tmax = max(t for _, t in pts)

    def f(th: float) -> float:
        return lam * (sum(w * math.exp(th * t) for w, t in pts) - 1) - th
    lo, hi = 0.0, 1.0 / tmax
    while f(hi) <= 0:
        lo, hi = hi, hi * 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13 * hi:
            break
    th = 0.5 * (lo + hi)
    return rho, th, (1 - rho) / (lam * sum(w * t * math.exp(th * t) for w, t in pts) - 1)


def mg1_mix_sum_quantile(lam: float, bins, q: float) -> float | None:
    """q-quantile of TTFT = W + L under a slowly drifting server speed (0.55, colocated chunked prefill).

    ``bins = [(w_b, [(w_i, τ_bi, l_bi)])]`` or ``(w_b, pts, v_b)`` with a vacation length v_b (multiple vacations of
    fixed length: the server runs a decode-only iteration whenever the prompt queue is empty, so by Fuhrmann–Cooper
    W = W_M/G/1 ⊕ Uniform(0, v_b)).  In environment b (the running decode batch, which drifts on a time scale
    much longer than a prefill busy period) the queue is M/G/1 with service law τ_b·; quasi-static mixture
    P(T > t) = Σ_b w_b Σ_i w_i·P(W_b > t − l_bi).  Per environment: exact M/D/1 (Erlang) for one service value
    within 6τ, else the Cramér–Lundberg tail; a service mix uses the exact Pollaczek–Khinchine law (0.56,
    ``_pk_surv``).  None if some environment is unstable."""
    envs = []
    for bn in bins:
        wb, pts = bn[0], bn[1]
        vac = bn[2] if len(bn) > 2 else 0.0
        if wb <= 0:
            continue
        tot = sum(w for w, _, _ in pts)
        pts = [(w / tot, t, l) for w, t, l in pts if w > 0]
        cl = _cl_params(lam, [(w, t) for w, t, _ in pts])
        if cl is None:
            return None
        single = pts[0][1] if len({t for _, t, _ in pts}) == 1 else None
        pk = None if single is not None else _pk_surv(lam, [(w, t) for w, t, _ in pts])     # 0.56: exact law
        envs.append((wb, pts, (cl, pk), single, vac))
    if not envs:
        return None
    wtot = sum(e[0] for e in envs)
    U = 8                                             # midpoints of the uniform vacation residual

    def sw(x: float, clpk, single) -> float:
        if x < 0:
            return 1.0
        (rho, th, c), pk = clpk
        if pk is not None:
            return pk[0](x)
        if single is not None and x <= MD1_EXACT_SPAN * single:
            return 1.0 - md1_cdf(lam, single, x)
        return min(rho, c * math.exp(-th * x))

    def w_hi(clpk) -> float:              # a wait beyond the q-quantile in this environment
        (rho, th, c), pk = clpk
        if pk is None:
            return max(0.0, math.log(max(c, 1e-300) / (1 - q)) / th)
        tail = pk[0](pk[1])
        return pk[1] + (max(0.0, math.log(tail / (1 - q)) / th) if tail > 0 else 0.0)

    def surv(t: float) -> float:
        acc = 0.0
        for wb, pts, cl, sg, v in envs:
            if v > 0:
                acc += wb * sum(w * sw(t - l - v * (j + 0.5) / U, cl, sg) for w, _, l in pts for j in range(U)) / U
            else:
                acc += wb * sum(w * sw(t - l, cl, sg) for w, _, l in pts)
        return acc / wtot
    a = min(l for _, pts, _, _, _ in envs for _, _, l in pts)
    b = max(l for _, pts, _, _, _ in envs for _, _, l in pts) + max(e[4] for e in envs) + max(
        w_hi(cl) for _, _, cl, _, _ in envs) + 1e-9
    if surv(a) <= 1 - q:
        return a
    for _ in range(200):
        mid = 0.5 * (a + b)
        if surv(mid) > 1 - q:
            a = mid
        else:
            b = mid
        if b - a < 1e-12 * max(b, 1e-12):
            break
    return b


def dquantile(pts, q: float) -> float:
    """q-quantile of a discrete distribution [(weight, value)] (weights need not be normalised)."""
    pts = sorted(((w, v) for w, v in pts if w > 0), key=lambda x: x[1])
    tot = sum(w for w, _ in pts)
    acc = 0.0
    for w, v in pts:
        acc += w / tot
        if acc >= q - 1e-12:
            return v
    return pts[-1][1]


def _pct(q: float) -> str:
    s = f"{q * 100:g}"
    return s.replace(".", "_")


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in M/M/c with offered load a = λ/μ (a < c)."""
    if a <= 0:
        return 0.0
    if a >= c:
        return 1.0
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return c * b / (c - a * (1 - b))


def mdc_wait(lam: float, service: float, c: int, qs: tuple = (0.5, 0.9, 0.99), cs2: float = 0.0) -> dict:
    """Slot wait of c servers with mean holding time ``service`` (Allen–Cunneen (1 + c_s²)/2 × Erlang C; c_s² = 0
    deterministic); quantiles from an exponential tail."""
    a = lam * service
    if lam <= 0:
        return {"rho": 0.0, "stable": True, "mean": 0.0, "p_wait": 0.0, **{f"p{_pct(q)}": 0.0 for q in qs}}
    den = c / service - lam          # 0.53 fix: a hair below c can round to den = 0 (λ at exactly the fluid capacity)
    if a >= c or den <= 0:
        return {"rho": a / c, "stable": False, "mean": math.inf, "p_wait": 1.0, **{f"p{_pct(q)}": math.inf for q in qs}}
    pw = erlang_c(c, a)
    cond = 0.5 * (1 + cs2) / den   # mean conditional wait (M/M/c: 1/(cμ − λ)) × (1 + c_s²)/2
    out = {"rho": a / c, "stable": True, "mean": pw * cond, "p_wait": pw}
    for q in qs:
        out[f"p{_pct(q)}"] = 0.0 if pw <= 1 - q else cond * math.log(pw / (1 - q))
    return out


def poisson_quantile(mean: float, q: float) -> int:
    """Smallest k with P(Poisson(mean) ≤ k) ≥ q."""
    if mean <= 0:
        return 0
    if mean > 500:          # normal approximation (continuity-corrected)
        z = {0.5: 0.0, 0.9: 1.2815515655446004, 0.99: 2.3263478740408408}.get(q)
        if z is None:
            raise ValueError("poisson_quantile: q must be 0.5, 0.9 or 0.99 for large means")
        return max(0, math.ceil(mean + z * math.sqrt(mean) - 0.5))
    p = math.exp(-mean)
    cdf, k = p, 0
    while cdf < q:
        k += 1
        p *= mean / k
        cdf += p
        if k > 10 * mean + 100:
            break
    return k
