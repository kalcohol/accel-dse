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
    c = (1 - rho) / (lam * sum(w * t * math.exp(th * t) for w, t in pts) - 1)
    out = {"rho": rho, "stable": True, "mean": lam * m2 / (2 * (1 - rho)), "p_wait": rho}
    for q in qs:
        out[f"p{_pct(q)}"] = 0.0 if rho <= 1 - q else max(0.0, math.log(c / (1 - q)) / th)
    return out



def mg1_sum_quantile(lam: float, taus, weights, lats, q: float) -> float | None:
    """q-quantile of W + L_i (0.54): W = M/G/1 wait over service mix ``taus`` (Cramér–Lundberg tail
    P(W > x) ≈ min(ρ, c·e^{−θx})), L_i = the request's own latency, drawn with the SAME index i as its service (one
    prompt length → its service and its latency).  W is independent of the arriving request's own i, so
    P(T > t) = Σ w_i·P(W > t − L_i).  Returns None when the service mix is a single value (caller keeps the exact
    wait-quantile + latency sum) or the queue is unstable."""
    pts = [(w, t, l) for w, t, l in zip(weights, taus, lats) if w > 0]
    if len({t for _, t, _ in pts}) <= 1 or lam <= 0:
        return None
    tot = sum(w for w, _, _ in pts)
    pts = [(w / tot, t, l) for w, t, l in pts]
    m1 = sum(w * t for w, t, _ in pts)
    rho = lam * m1
    if rho >= 1:
        return None
    tmax = max(t for _, t, _ in pts)

    def f(th: float) -> float:
        return lam * (sum(w * math.exp(th * t) for w, t, _ in pts) - 1) - th
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
    c = (1 - rho) / (lam * sum(w * t * math.exp(th * t) for w, t, _ in pts) - 1)

    def surv(t: float) -> float:
        acc = 0.0
        for w, _, l in pts:
            x = t - l
            acc += w * (1.0 if x < 0 else min(rho, c * math.exp(-th * x)))
        return acc
    a = min(l for _, _, l in pts)
    b = max(l for _, _, l in pts) + max(0.0, math.log(max(c, 1e-300) / (1 - q)) / th) + 1e-9
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
    within 6τ, else the Cramér–Lundberg tail.  None if some environment is unstable."""
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
        envs.append((wb, pts, cl, single, vac))
    if not envs:
        return None
    wtot = sum(e[0] for e in envs)
    U = 8                                             # midpoints of the uniform vacation residual

    def sw(x: float, cl, single) -> float:
        if x < 0:
            return 1.0
        rho, th, c = cl
        if single is not None and x <= MD1_EXACT_SPAN * single:
            return 1.0 - md1_cdf(lam, single, x)
        return min(rho, c * math.exp(-th * x))

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
        max(0.0, math.log(max(cl[2], 1e-300) / (1 - q)) / cl[1]) for _, _, cl, _, _ in envs) + 1e-9
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
