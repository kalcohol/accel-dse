"""L5 — per-stage time composition, collectives, pipeline and speculative decode.

Collectives (α-β, per rank payload B, group g, link β, latency α):
  allreduce (ring)   2(g−1)/g · B/β      + α
  alltoall           (g−1)/g · B/β       + α
  allgather          (g−1)/g · B·g/β ... payload B is the per-rank share → (g−1)·B/β + α
  p2p (PP send)      B/β                 + α
α is *exposed* synchronisation: added after the overlapped max (「假设」 3 µs).

Three-tier interconnect (0.50, ``fabric_collective``; 0.48 two-tier = its special case ``tiered_collective``):
cards are numbered TP-fastest, then SP, DP, PP; the ``package`` consecutive dies of a package talk over the
die-to-die tier (β_d, α_d — only with D2D on), the ``node`` consecutive cards of a node over the in-node scale-up
link (β_n, α_n), nodes over the cross-node network (β_x, α_x; node = 0 → one node, tier unused).  A group of g ranks
with stride s keeps k₁ = gcd(clamp(P // s, 1, g), g) members in one package and k₂ (same rule with N, a multiple of
k₁) in one node → level sizes n₀ = k₁ (D2D), n₁ = k₂ / k₁ (scale-up), n₂ = g / k₂ (network); size-1 levels drop out
and a single remaining level is exactly the flat formula on that tier.  Hierarchical (NCCL-style) with
B₍ᵢ₎ = B / Π_{j<i} n_j:
    allreduce  reduce-scatter up the levels, allreduce on the top one, all-gather back down:
               Σᵢ 2(nᵢ−1)/nᵢ · B₍ᵢ₎ / βᵢ,                     α = Σ_{i<top} 2αᵢ + α_top
    allgather  outermost level first, then inwards:  Σᵢ (nᵢ−1) · B · Π_{j>i} n_j / βᵢ,   α = Σᵢ αᵢ
    alltoall   every tier at once, each carrying the destinations it owns ((nᵢ−1)·Π_{j<i} n_j / g of B):
               maxᵢ share_i · B / βᵢ,                         α = maxᵢ αᵢ
With two levels (D2D + scale-up) these are the 0.48 formulas, bit for bit.  The share of the bytes each rank sends
on the D2D and on the network tier is returned for the energy counts (the rest is scale-up link).

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


def _members(cap: int, group: int, stride: int) -> int:
    if cap <= 1:
        return 1
    return math.gcd(max(1, min(group, cap // max(1, stride))), group)


def fabric_collective(kind: str, payload: float, group: int, link: Link, d2d: Link | None = None, package: int = 1,
                      net: Link | None = None, node: int = 0, stride: int = 1, k_pkg: int | None = None,
                      k_node: int | None = None) -> tuple[float, float, float, float]:
    """(bandwidth s, exposed-latency s, D2D share, network share of the sent bytes) on the three-tier fabric.
    ``k_pkg`` / ``k_node``: members of the group inside one package / node when the group is not a uniform stride."""
    if group <= 1 or payload <= 0:
        return 0.0, 0.0, 0.0, 0.0
    k1 = k_pkg if k_pkg is not None else (1 if d2d is None else _members(package, group, stride))
    k1 = math.gcd(max(1, k1), group)
    if net is None or node <= 0:
        k2 = group
    else:
        k2 = math.gcd(max(1, k_node if k_node is not None else _members(node, group, stride)), group)
        if k2 % k1:
            k2 = k1
    levels = [(n, ln, t) for n, ln, t in ((k1, d2d, "d2d"), (k2 // k1, link, "link"), (group // k2, net, "net"))
              if n > 1]
    if len(levels) == 1:
        n, ln, t = levels[0]
        bw, a = collective_seconds(kind, payload, group, ln)
        return bw, a, float(t == "d2d"), float(t == "net")
    if kind not in ("allreduce", "allgather", "alltoall"):         # p2p-like: over the outermost tier
        n, ln, t = levels[-1]
        bw, a = collective_seconds(kind, payload, group, ln)
        return bw, a, float(t == "d2d"), float(t == "net")
    sizes = [n for n, _, _ in levels]
    vol = []
    for i, (n, ln, t) in enumerate(levels):
        pre, post = math.prod(sizes[:i]), math.prod(sizes[i + 1:])
        if kind == "allreduce":
            v = 2 * (n - 1) / n * payload if pre == 1 else 2 * (n - 1) / n * payload / pre
        elif kind == "allgather":
            v = (n - 1) * post * payload
        else:
            v = (n - 1) / group * payload if pre == 1 else (n - 1) * pre / group * payload
        vol.append(v)
    secs = [v / (ln.GBps * 1e9) for v, (_, ln, _) in zip(vol, levels)]
    alphas = [ln.alpha_us * 1e-6 for _, ln, _ in levels]
    if kind == "alltoall":
        bw, a = max(secs), max(alphas)
    else:
        bw = secs[0]
        for x in secs[1:]:
            bw += x
        if kind == "allreduce":
            a = 2 * alphas[0]
            for x in alphas[1:-1]:
                a += 2 * x
            a += alphas[-1]
        else:
            a = alphas[0]
            for x in alphas[1:]:
                a += x
    tot = sum(vol)
    share = {t: v / tot for v, (_, _, t) in zip(vol, levels)}
    return bw, a, share.get("d2d", 0.0), share.get("net", 0.0)


def tiered_collective(kind: str, payload: float, group: int, link: Link, d2d: Link, package: int,
                      stride: int = 1) -> tuple[float, float, float]:
    """(bandwidth s, exposed-latency s, D2D share of the sent bytes) on the 0.48 two-tier fabric (one node)."""
    bw, a, fd, _ = fabric_collective(kind, payload, group, link, d2d if package > 1 else None, package, stride=stride)
    return bw, a, fd


def p2p_tier(stage: int, stage_cards: int, package: int, node: int) -> str:
    """Tier of the PP hand-off ``stage`` → next: "d2d" (same package), "link" (same node) or "net"."""
    if not p2p_crosses(stage, stage_cards, package):
        return "d2d"
    if node <= 0 or not p2p_crosses(stage, stage_cards, node):
        return "link"
    return "net"


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
    net_bytes: float = 0.0  # share of link_bytes carried by the cross-node network tier (0.50)
    t_arr_attn: float = 0.0 # 0.66: array time of attention-core ops (part of t_array)
    t_dram_kv: float = 0.0  # 0.66: KV / state traffic time (part of t_dram)
    overlap: str = "stage"  # 0.66: stage | class | kernel | serial (see total)

    @property
    def array_util(self) -> float:
        """Useful MAC fraction while the array is busy (tile padding / small-M waste shows here)."""
        return self.t_ideal / self.t_array if self.t_array > 0 else 0.0

    @property
    def t_compute(self) -> float:
        return max(self.t_array, self.t_vector)

    @property
    def total(self) -> float:
        """Overlap inside a stage step (``Scenario.exec_overlap``; 0.66 modes 「假设」):
          stage   max(t_compute, t_dram, t_slc, t_link) + t_sync — everything in the stage overlaps (pre-0.66)
          class   kernel-serial GPU-like execution with a roofline per op class: GEMMs ‖ weight traffic, attention ‖
                  KV traffic, the classes add, vector work overlaps that sum, collectives exposed:
                  max(t_vector, max(t_gemm, t_dram_w) + max(t_attn, t_dram_kv), t_slc) + t_link + t_sync
          kernel  class, with the vector work as kernels of its own (added, not overlapped)
          serial  no overlap at all: t_array + t_vector + max(t_dram, t_slc) + t_link + t_sync (upper bound)
          tbo     (0.67) kernel-serial compute as in ``kernel``, but collectives hidden behind it — two-micro-batch
                  overlap (DeepSeek / SGLang "two-batch overlap": one micro-batch's all-to-all runs during the other's
                  compute): max(kernel core, t_slc, t_link) + t_sync
        Every mode is ≥ the stage value; stage ≤ tbo ≤ kernel ≤ serial."""
        if self.overlap != "stage":
            if self.overlap == "serial":
                return self.t_array + self.t_vector + max(self.t_dram, self.t_slc) + self.t_link + self.t_sync
            core = max(self.t_array - self.t_arr_attn, self.t_dram - self.t_dram_kv) + max(self.t_arr_attn, self.t_dram_kv)
            core = core + self.t_vector if self.overlap in ("kernel", "tbo") else max(self.t_vector, core)
            if self.overlap == "tbo":
                return max(core, self.t_slc, self.t_link) + self.t_sync
            return max(core, self.t_slc) + self.t_link + self.t_sync
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
