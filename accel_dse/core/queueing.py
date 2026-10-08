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
        out[f"p{_pct(q)}"] = 0.0 if rho <= 1 - q else max(0.0, math.log(c / (1 - q)) / theta)
    return out


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


def mdc_wait(lam: float, service: float, c: int, qs: tuple = (0.5, 0.9, 0.99)) -> dict:
    """Slot wait of c deterministic servers (Allen–Cunneen ½ × Erlang C); quantiles from an exponential tail."""
    a = lam * service
    if lam <= 0:
        return {"rho": 0.0, "stable": True, "mean": 0.0, "p_wait": 0.0, **{f"p{_pct(q)}": 0.0 for q in qs}}
    if a >= c:
        return {"rho": a / c, "stable": False, "mean": math.inf, "p_wait": 1.0, **{f"p{_pct(q)}": math.inf for q in qs}}
    pw = erlang_c(c, a)
    cond = 0.5 / (c / service - lam)          # mean conditional wait (M/M/c: 1/(cμ − λ)) × ½
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
