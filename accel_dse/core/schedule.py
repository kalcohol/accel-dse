"""L5 — per-stage time composition, collectives, pipeline and speculative decode.

Collectives (α-β, per rank payload B, group g, link β, latency α):
  allreduce (ring)   2(g−1)/g · B/β      + α
  alltoall           (g−1)/g · B/β       + α
  allgather          (g−1)/g · B·g/β ... payload B is the per-rank share → (g−1)·B/β + α
  p2p (PP send)      B/β                 + α
α is *exposed* synchronisation: added after the overlapped max (「假设」 3 µs).

Two-tier interconnect (0.48, ``tiered_collective``): cards are numbered TP-fastest, then SP, DP, PP; the
``package_cards`` consecutive cards of a package talk over the die-to-die link (β_d, α_d), packages over the
scale-up / network link (β_n, α_n).  A group of g ranks with stride s keeps k = clamp(P // s, 1, g) members in one
package and spans m = g / k packages:
  k = g        all on D2D (the formulas above with β_d, α_d)
  k = 1        all on the network (the formulas above — exactly the single-tier result)
  otherwise    hierarchical (NCCL-style two-level):
    allreduce  intra reduce-scatter + inter allreduce of B/k + intra all-gather:
               2(k−1)/k·B/β_d + 2(m−1)/m·(B/k)/β_n,      α = 2α_d + α_n
    allgather  inter all-gather among packages, then intra:  (m−1)·B/β_n + (k−1)·m·B/β_d,   α = α_n + α_d
    alltoall   the (k−1)/g share over D2D and the (g−k)/g share over the network at once:
               max((k−1)/g·B/β_d, (g−k)/g·B/β_n),        α = max(α_d, α_n)
The share of the bytes each rank sends that stays on D2D is returned for the energy counts.

Stage step:  t_stage = max(t_compute, t_dram, t_slc, t_link) + t_sync   (t_slc: 0.48 SLC port, 0 without one)
  t_compute = max(Σ array-op time, Σ vector time)      (array ‖ vector, 「假设」)
Decode step with mb micro-batches over pp stages: T_step = max(mb, pp)·max_s t_stage
Prefill:     TTFT = (mb + pp − 1)·max_s t_stage
Speculative decode (k drafts, acceptance a, i.i.d. 「假设」):
  E[tokens/step] = (1 − a^(k+1)) / (1 − a);   verify runs q = 1+k tokens
  MTP drafts run sequentially on the last stage (k extra MTP-module passes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hardware import Link


def collective_seconds(kind: str, payload: float, group: int, link: Link) -> tuple[float, float]:
    """(bandwidth seconds, exposed-latency seconds)."""
    if group <= 1 or payload <= 0:
        return 0.0, 0.0
    beta = link.GBps * 1e9
    a = link.alpha_us * 1e-6
    if kind == "allreduce":
        bw = 2 * (group - 1) / group * payload / beta
    elif kind == "alltoall":
        bw = (group - 1) / group * payload / beta
    elif kind == "allgather":
        bw = (group - 1) * payload / beta
    else:  # p2p
        bw = payload / beta
    return bw, a


def tiered_collective(kind: str, payload: float, group: int, link: Link, d2d: Link, package: int,
                      stride: int = 1) -> tuple[float, float, float]:
    """(bandwidth s, exposed-latency s, D2D share of the sent bytes) of one collective on the two-tier fabric."""
    if group <= 1 or payload <= 0:
        return 0.0, 0.0, 0.0
    k = 1 if package <= 1 else max(1, min(group, package // max(1, stride)))
    k = math.gcd(k, group)
    if k == 1:
        return (*collective_seconds(kind, payload, group, link), 0.0)
    if k == group:
        return (*collective_seconds(kind, payload, group, d2d), 1.0)
    m = group // k
    bd, bn = d2d.GBps * 1e9, link.GBps * 1e9
    ad, an = d2d.alpha_us * 1e-6, link.alpha_us * 1e-6
    if kind == "allreduce":
        intra, inter = 2 * (k - 1) / k * payload, 2 * (m - 1) / m * payload / k
        return intra / bd + inter / bn, 2 * ad + an, intra / (intra + inter)
    if kind == "allgather":
        intra, inter = (k - 1) * m * payload, (m - 1) * payload
        return intra / bd + inter / bn, ad + an, intra / (intra + inter)
    if kind == "alltoall":
        intra, inter = (k - 1) / group * payload, (group - k) / group * payload
        return max(intra / bd, inter / bn), max(ad, an), intra / (intra + inter)
    return (*collective_seconds(kind, payload, group, link), 0.0)


def p2p_crosses(stage: int, stage_cards: int, package: int) -> bool:
    """Does the PP hand-off from ``stage`` to the next stage leave the package?  (cards numbered TP, SP, DP, PP)"""
    if package <= 1:
        return True
    return (stage * stage_cards) // package != ((stage + 1) * stage_cards) // package


def spec_expected_tokens(k: int, a: float) -> float:
    if k <= 0:
        return 1.0
    if not (0.0 <= a <= 1.0):
        raise ValueError("acceptance must be in [0,1]")
    if a >= 1.0:
        return float(k + 1)
    return (1.0 - a ** (k + 1)) / (1.0 - a)


@dataclass(frozen=True)
class StageTime:
    t_array: float
    t_mac: float            # array time spent in MAC-bound ops
    t_feed: float           # array time spent in feed(SRAM-port)-bound ops
    t_vector: float
    t_dram: float
    t_link: float
    t_sync: float
    dram_bytes: float
    link_bytes: float
    flops: float
    t_ideal: float = 0.0    # array time at 100 % MAC utilisation (same work)
    t_slc: float = 0.0      # system-level-cache time (0.48): SLC hits / slc_GBps, a port parallel to DRAM 「假设」
    slc_bytes: float = 0.0  # bytes served by the SLC
    d2d_bytes: float = 0.0  # share of link_bytes carried by the die-to-die tier

    @property
    def array_util(self) -> float:
        """Useful MAC fraction while the array is busy (tile padding / small-M waste shows here)."""
        return self.t_ideal / self.t_array if self.t_array > 0 else 0.0

    @property
    def t_compute(self) -> float:
        return max(self.t_array, self.t_vector)

    @property
    def total(self) -> float:
        return max(self.t_compute, self.t_dram, self.t_slc, self.t_link) + self.t_sync

    @property
    def bound(self) -> str:
        """Which bound binds: MAC | FEED | VECTOR | DRAM | SLC | LINK (| SYNC if α dominates)."""
        core = max(self.t_compute, self.t_dram, self.t_slc, self.t_link)
        if self.t_sync > core:
            return "SYNC"
        if core == self.t_dram and self.t_dram > 0:
            return "DRAM"
        if core == self.t_slc and self.t_slc > 0:
            return "SLC"
        if core == self.t_link and self.t_link > 0:
            return "LINK"
        if self.t_vector > self.t_array:
            return "VECTOR"
        return "FEED" if self.t_feed > self.t_mac else "MAC"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(t_compute=self.t_compute, total=self.total, bound=self.bound, array_util=self.array_util)
        return d
