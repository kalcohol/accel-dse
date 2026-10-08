"""Request-length distribution for the PD report (0.52) 「假设」.

Default: every request has prompt S = ``serving.prompt`` and output ``serving.out_len`` (fixed lengths, as in 0.51).
``pd.prompt_cv`` / ``pd.out_cv`` (coefficient of variation > 0): independent lognormal prompt / output lengths with
those means, discretised into N = 15 equal-probability bins (0.56; was 8), each represented by its conditional mean
  E[X | bin k] = N · mean · (Φ(z_{k+1} − σ) − Φ(z_k − σ)),  z_k = Φ⁻¹(k / N),  σ² = ln(1 + cv²)
(exact for the lognormal; the mean is kept, the within-bin spread is dropped, so the variance is slightly low).
N = 15 puts no bin boundary at the reported quantiles (0.5·N, 0.9·N, 0.99·N are not integers): with N = 8,
P(S ≥ s₅) = 0.5 exactly, so TTFT p50 sat on a jump of the distribution and moved by tens of percent for a 0.03
change in probability (V4 0.55).
``pd.length_mix``: an explicit discrete joint mix of (weight, prompt, out_len) rows (may correlate S and out).
Each prompt value is evaluated separately (prefill time is not linear in S: attention), so nothing assumes linearity
in the prompt length; output lengths enter through their moments (Little's law, Erlang C / Allen–Cunneen c_s²).
Decode context: a running request is seen in proportion to its output length (length bias), so the time-average
context is E[out·(S + out/2)] / E[out].  ``serving.ctx`` is read as the context of a representative fixed-length
request and rescaled by that over (E[S] + E[out]/2) (= 1 for fixed lengths).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

BINS = 15
_N = NormalDist()


def lognormal_bins(mean: float, cv: float, n: int = BINS) -> list[tuple[float, int]]:
    """[(weight, integer value)] — n equal-probability bins of a lognormal (mean, cv), bin conditional means."""
    if cv <= 0:
        return [(1.0, max(1, round(mean)))]
    sig = math.sqrt(math.log1p(cv * cv))
    z = [-math.inf] + [_N.inv_cdf(k / n) for k in range(1, n)] + [math.inf]
    out = []
    for k in range(n):
        lo = _N.cdf(z[k] - sig) if math.isfinite(z[k]) else 0.0
        hi = _N.cdf(z[k + 1] - sig) if math.isfinite(z[k + 1]) else 1.0
        out.append((1.0 / n, max(1, round(n * mean * (hi - lo)))))
    return out


@dataclass(frozen=True)
class Lengths:
    points: tuple[tuple[float, int, int], ...]     # (weight, prompt, out_len), weights sum to 1
    source: str                                    # fixed | cv | mix

    @property
    def trivial(self) -> bool:
        return len(self.points) == 1

    def prompts(self) -> list[tuple[float, int]]:
        """Marginal of the prompt length: [(weight, S)] sorted by S."""
        acc: dict = {}
        for w, s, _ in self.points:
            acc[s] = acc.get(s, 0.0) + w
        return sorted(((w, s) for s, w in acc.items()), key=lambda x: x[1])

    @property
    def mean_S(self) -> float:
        return sum(w * s for w, s, _ in self.points)

    @property
    def mean_out(self) -> float:
        return sum(w * o for w, _, o in self.points)

    @property
    def out_cs2(self) -> float:
        """Squared coefficient of variation of the output length."""
        m = self.mean_out
        return max(0.0, sum(w * o * o for w, _, o in self.points) / (m * m) - 1.0) if m > 0 else 0.0

    @property
    def prompt_cs2(self) -> float:
        m = self.mean_S
        return max(0.0, sum(w * s * s for w, s, _ in self.points) / (m * m) - 1.0) if m > 0 else 0.0

    @property
    def ctx_ratio(self) -> float:
        if self.trivial:
            return 1.0
        m = self.mean_out
        t = sum(w * o * (s + o / 2) for w, s, o in self.points) / m
        return t / (self.mean_S + m / 2)

    def summary(self) -> dict:
        return {"source": self.source, "points": [list(p) for p in self.points], "mean_prompt": self.mean_S,
                "mean_out": self.mean_out,
                "prompt_cv_eff": math.sqrt(self.prompt_cs2), "out_cv_eff": math.sqrt(self.out_cs2), "ctx_ratio": self.ctx_ratio,
                "prompts": [list(p) for p in self.prompts()]}


def lengths_of(pd, sv) -> Lengths:
    if pd.length_mix:
        tot = sum(r[0] for r in pd.length_mix)
        return Lengths(tuple((w / tot, s, o) for w, s, o in pd.length_mix), "mix")
    if pd.prompt_cv or pd.out_cv:
        ps, os_ = lognormal_bins(sv.prompt, pd.prompt_cv), lognormal_bins(sv.out_len, pd.out_cv)
        return Lengths(tuple((a * b, s, o) for a, s in ps for b, o in os_), "cv")
    return Lengths(((1.0, sv.prompt, sv.out_len),), "fixed")
