"""L5 — per-stage time composition, collectives, pipeline and speculative decode.

Collectives (α-β, per rank payload B, group g, link β, latency α):
  allreduce (ring)   2(g−1)/g · B/β      + α
  alltoall           (g−1)/g · B/β       + α
  allgather          (g−1)/g · B·g/β ... payload B is the per-rank share → (g−1)·B/β + α
  p2p (PP send)      B/β                 + α
α is *exposed* synchronisation: added after the overlapped max (「假设」 3 µs).

Stage step:  t_stage = max(t_compute, t_dram, t_link) + t_sync
  t_compute = max(Σ array-op time, Σ vector time)      (array ‖ vector, 「假设」)
Decode step with mb micro-batches over pp stages: T_step = max(mb, pp)·max_s t_stage
Prefill:     TTFT = (mb + pp − 1)·max_s t_stage
Speculative decode (k drafts, acceptance a, i.i.d. 「假设」):
  E[tokens/step] = (1 − a^(k+1)) / (1 − a);   verify runs q = 1+k tokens
  MTP drafts run sequentially on the last stage (k extra MTP-module passes).
"""

from __future__ import annotations

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

    @property
    def array_util(self) -> float:
        """Useful MAC fraction while the array is busy (tile padding / small-M waste shows here)."""
        return self.t_ideal / self.t_array if self.t_array > 0 else 0.0

    @property
    def t_compute(self) -> float:
        return max(self.t_array, self.t_vector)

    @property
    def total(self) -> float:
        return max(self.t_compute, self.t_dram, self.t_link) + self.t_sync

    @property
    def bound(self) -> str:
        """Which bound binds: MAC | FEED | VECTOR | DRAM | LINK (| SYNC if α dominates)."""
        core = max(self.t_compute, self.t_dram, self.t_link)
        if self.t_sync > core:
            return "SYNC"
        if core == self.t_dram and self.t_dram > 0:
            return "DRAM"
        if core == self.t_link and self.t_link > 0:
            return "LINK"
        if self.t_vector > self.t_array:
            return "VECTOR"
        return "FEED" if self.t_feed > self.t_mac else "MAC"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(t_compute=self.t_compute, total=self.total, bound=self.bound, array_util=self.array_util)
        return d
