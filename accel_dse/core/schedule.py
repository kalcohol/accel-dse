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

import functools

import math
from dataclasses import dataclass

from .hardware import Link


# 0.199: small-message all-reduce latency inside one NVLink / NVSwitch node 「来源」 NCCL's own cost model
# (src/graph/tuning.cc, v2.19.3): Ring + LL protocol = baseLat 6.6 µs + nsteps × hwLat[NVLINK][Ring][LL] 0.6 µs,
# nsteps = 2(n − 1) for all-reduce → n = 2: 7.8, 4: 10.2, 8: 15.0 µs (Tree LL: 6.8 + 2(n − 1)·0.6 — the same).
# Measured cross-check 「二手」: nccl-tests issue #123 (DGX-A100, 8 GPUs, LL) ≈ 21 µs total incl. launch; NCCL dev
# reply: ~7.5 µs base on 2 × H100.  ≤ 0.198: a flat Link.alpha_us = 3 µs per collective, any group size.
# 「假设」: the engines' collectives run at the NCCL latency — vLLM / SGLang custom all-reduce and TRT-LLM's
# all-reduce plugin can be faster for small messages; no per-engine measurement is applied.
NCCL_AR_LAT = True
# 0.200: the engines behind the validation rows run TP all-reduce on a custom one-/two-shot kernel inside an NVLink node
# (≤ 8 GPUs), not NCCL 「来源」: TensorRT-LLM v0.8.0 tensorrt_llm/plugin/plugin.py PluginConfig.use_custom_all_reduce =
# True (the 0.8 perf-doc builds do not override it); vLLM disable_custom_all_reduce = False (csrc/custom_all_reduce.cuh,
# PR #2192: kernels "very similar to TensorRT-LLM's customAllReduceKernels"); SGLang the same kernel.  Latency floor:
# 6.8 µs at 2 KB on 8 × H100 with CUDA graphs (vLLM PR #33839, benchmark_device_communicators.py) 「二手」; the same
# floor is used for TP 2 / 4 and A100 「假设」.  Larger messages keep the bandwidth term.  CUSTOM_AR False = 0.199 (NCCL).
CUSTOM_AR = True
# 0.201: only links marked ar_lat = "gpu" (the reference GPUs) take the CUDA all-reduce latency; any other link keeps its
# own alpha_us.  ≤ 0.200 every switch link got max(alpha_us, 6.8 µs) — NPU design points silently inherited the CUDA
# custom all-reduce floor.  False = ≤ 0.200.
GPU_AR_ONLY = True
CUSTOM_AR_US = 6.8
# 0.202: MI300X (ar_lat "rocm") — vLLM's ROCm custom all-reduce (the same one-/two-shot kernel on HIP, PR #14125; used
# below 16 MB, QuickReduce off by default) measured on 8 × MI300X 「来源」: vLLM PR #19744 kernel benchmark, "baseline =
# custom allreduce when the data size is less than 16MB": TP2 7.26 µs (2 KB), TP4 7.18 µs (2 KB), TP8 12.29 µs (min,
# 32 KB; 14.07 at 4 KB).  Floor = the smallest measured value per TP.  Groups 3 / 5–7 take the next listed TP 「假设」;
# > 8 keeps the NCCL Ring-LL model 「假设」 (no RCCL small-message source applied).  ROCM_AR False = ≤ 0.201 (CUDA 6.8 µs).
ROCM_AR = True
ROCM_AR_US = {2: 7.26, 4: 7.18, 8: 12.29}
NCCL_BASE_LAT_US, NCCL_NVL_HW_LAT_US = 6.6, 0.6


def nccl_ring_ll_us(n: int) -> float:
    return NCCL_BASE_LAT_US + 2 * (n - 1) * NCCL_NVL_HW_LAT_US if n > 1 else 0.0


def collective_seconds(kind: str, payload: float, group: int, link: Link) -> tuple[float, float]:
    """(bandwidth seconds, exposed-latency seconds)."""
    if group <= 1 or payload <= 0:
        return 0.0, 0.0
    beta = link.GBps * 1e9
    a = link.alpha_us * 1e-6
    if kind == "allreduce" and NCCL_AR_LAT and link.topology == "switch" and (link.ar_lat in ("gpu", "rocm") or not GPU_AR_ONLY):
        if link.ar_lat == "rocm" and ROCM_AR and CUSTOM_AR and group <= 8:
            fl = ROCM_AR_US[min(g for g in ROCM_AR_US if g >= group)]
        else:
            fl = CUSTOM_AR_US if CUSTOM_AR and group <= 8 else nccl_ring_ll_us(group)
        a = max(a, fl * 1e-6)
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


# 0.143: DeepEP normal (node-mode) all-to-all, achieved rates from DeepEP's published H800 table (README, 2025-06,
# "~160 GB/s NVLink maximum", CX7 400 Gb/s ≈ 50 GB/s): intranode NVLink dispatch 153 / combine 158 GB/s against the
# 200 GB/s nominal NVLink of the model's H800 → 0.78; internode RDMA 43 / 58 / 51 GB/s at EP16 / 32 / 64 — at or above
# the 50 GB/s line rate on the logical byte count (EP32 / 64), so no RDMA derating is applied 「假设」.
DEEPEP_NVL_EFF = 155.5 / 200.0


def _node_fwd(vol: list, levels: list, payload: float, group: int, net_scale: float) -> list:
    """0.143: node-mode all-to-all forwarding.  A remote token-expert pair arrives over RDMA at the same-local-rank GPU
    of the destination node and is forwarded over NVLink to the expert's GPU ((k − 1) / k of them), so the scale-up
    tier carries the remote pairs as well, not only the node-local ones."""
    if net_scale >= 1.0 or not any(t == "net" for *_, t in levels):
        return vol
    k = math.prod(lv[0] for lv in levels if lv[-1] != "net")
    li = next((i for i, lv in enumerate(levels) if lv[-1] == "link"), None)
    if li is None or k <= 1:
        return vol
    vol = list(vol)
    vol[li] += (group - k) / group * payload * (k - 1) / k
    return vol


def fabric_collective(kind: str, payload: float, group: int, link: Link, d2d: Link | None = None, package: int = 1,
                      net: Link | None = None, node: int = 0, stride: int = 1, k_pkg: int | None = None,
                      k_node: int | None = None, net_scale: float = 1.0) -> tuple[float, float, float, float]:
    """(bandwidth s, exposed-latency s, D2D share, network share of the sent bytes) on the three-tier fabric.
    ``net_scale`` (0.134): factor on an all-to-all's network-tier volume (DeepEP normal kernels send a token once per
    destination node, Op.comm_net).
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
            if t == "net":
                v *= net_scale
        vol.append(v)
    eff = [1.0] * len(levels)
    if kind == "alltoall" and net_scale < 1.0:
        vol = _node_fwd(vol, levels, payload, group, net_scale)
        eff = [DEEPEP_NVL_EFF if t == "link" else 1.0 for _, _, t in levels]
    secs = [v / (ln.GBps * 1e9 * e) for v, (_, ln, _), e in zip(vol, levels, eff)]
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


def wire_single(kind: str, payload: float, group: int) -> float:
    """0.78: bytes one rank puts on the wire for one collective on a single tier (the α-β volumes above): ring
    all-reduce 2(n−1)/n · payload, all-gather / reduce-scatter (n−1) · shard, all-to-all (n−1)/n · payload, p2p payload."""
    if group <= 1 or payload <= 0:
        return 0.0
    if kind == "allreduce":
        return 2 * (group - 1) / group * payload
    if kind == "alltoall":
        return (group - 1) / group * payload
    if kind in ("allgather", "reducescatter"):
        return (group - 1) * payload
    return payload


def collective_wire(kind: str, payload: float, group: int, d2d: bool = False, package: int = 1, node: int = 0,
                    stride: int = 1, k_pkg: int | None = None, k_node: int | None = None,
                    net_scale: float = 1.0) -> dict:
    """0.78: {tier: bytes one rank sends} of ``fabric_collective`` (same levels and per-level volumes)."""
    if group <= 1 or payload <= 0:
        return {}
    k1 = k_pkg if k_pkg is not None else (1 if not d2d else _members(package, group, stride))
    k1 = math.gcd(max(1, k1), group)
    if node <= 0:
        k2 = group
    else:
        k2 = math.gcd(max(1, k_node if k_node is not None else _members(node, group, stride)), group)
        if k2 % k1:
            k2 = k1
    levels = [(n, t) for n, t in ((k1, "d2d"), (k2 // k1, "link"), (group // k2, "net")) if n > 1]
    if not levels:
        return {}
    if len(levels) == 1 or kind not in ("allreduce", "allgather", "alltoall"):
        return {levels[-1][1] if len(levels) > 1 else levels[0][1]: wire_single(kind, payload, group)}
    sizes = [n for n, _ in levels]
    out = {}
    for i, (n, t) in enumerate(levels):
        pre, post = math.prod(sizes[:i]), math.prod(sizes[i + 1:])
        if kind == "allreduce":
            v = 2 * (n - 1) / n * payload / pre
        elif kind == "allgather":
            v = (n - 1) * post * payload
        else:
            v = (n - 1) * pre / group * payload * (net_scale if t == "net" else 1.0)
        out[t] = v
    if kind == "alltoall" and net_scale < 1.0 and "net" in out and "link" in out:   # 0.143: NVLink forwarding
        k = math.prod(n for n, t in levels if t != "net")
        out["link"] += (group - k) / group * payload * (k - 1) / k
    return out


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


def tree_tokens(k: int, topk: int, n: int) -> int:
    """0.224: verified tokens per sequence of a draft tree (root + nodes); chain (topk 1): 1 + k."""
    if topk <= 1:
        return 1 + k
    return n if n else min(64, 1 + k * topk)


TREE_ACCEPT_MODELS = ("chain", "rank", "indep")      # 0.226: default chain (lower bound); band [chain, indep]

# 0.226: Sequoia (arXiv 2402.12374) App. E acceptance vector — probability that the j-th ranked child of an accepted
# token is accepted (positional acceptance assumption, Def. 3.1); Llama3-70B-Instruct target, Llama3-8B-Instruct draft,
# CNN DailyMail 「来源」 (temperature not stated)
SEQUOIA_ACCEPT_VECTOR = (0.7732, 0.1039, 0.0402, 0.0206, 0.0128, 0.0081, 0.0064, 0.0043, 0.0035, 0.0026, 0.0025, 0.0021,
                         0.0016, 0.0014, 0.0010, 0.0010, 0.0010, 0.0007, 0.0007, 0.0006, 0.0007, 0.0006, 0.0004, 0.0004,
                         0.0005, 0.0006, 0.0004, 0.0003, 0.0002, 0.0004, 0.0001)


def rank_accept(a: float, j: int) -> float:
    """0.226: acceptance of the j-th ranked child (0-based) for a top-1 acceptance a 「推导」: p_0 = a, p_j = P_j ·
    min(a / P_0, (1 − a) / (1 − P_0)) — the smaller of "scale every rank with a" and "keep the shape of the miss mass",
    so Σ p ≤ 1 at any a; at a = P_0 both equal the published vector."""
    P = SEQUOIA_ACCEPT_VECTOR
    if j == 0:
        return a
    if j >= len(P):
        return 0.0
    return P[j] * min(a / P[0], (1.0 - a) / (1.0 - P[0]))


@functools.lru_cache(maxsize=4096)
def rank_tree_tau(a: float, k: int, topk: int, n: int) -> float:
    """0.226: τ of an EAGLE-2 / SGLang-style draft tree under positional acceptance: each of k draft steps expands the
    topk best leaves with topk children each and keeps the topk best (SGLang eagle_utils select_top_k_tokens 「来源」);
    the verified tree is the n − 1 best-scoring candidates (score f(v) = Π child-rank acceptance on the path, which
    under positional acceptance equals the draft-confidence order 「假设」); τ = 1 + Σ f(v) (Sequoia Prop. 3.4)."""
    cands, front = [], [1.0]
    for d in range(k):
        nxt = sorted(((f * rank_accept(a, r), d) for f in front for r in range(topk)), key=lambda x: -x[0])
        cands += nxt
        front = [x[0] for x in nxt[:topk]]
    cands.sort(key=lambda x: (-x[0], x[1]))
    return 1.0 + sum(x[0] for x in cands[:n - 1])


def spec_tree_accept(a: float, k: int, topk: int, n: int, model: str = "indep") -> float:
    """Per-depth acceptance of a draft tree (serving.spec_tree_accept_model):
    "indep" (0.224 「假设」, upper bound): the accepted path at each depth has w = min(topk, (n − 1) / k) candidate
      children, each accepted i.i.d. with probability a → 1 − (1 − a)^w.  0.225 check vs the published matched
      chain / tree pairs (core/spec_refs.py, EAGLE-1 Table 5): +49 … +63 % over the published tree τ — siblings are
      neither independent nor equally likely (greedy verify accepts at most one, lower-ranked candidates match less).
    "chain" (lower bound): the tree's top-1 path is the chain of the same depth (EAGLE / SGLang trees keep the top-1
      child at every expanded node 「来源」 EAGLE choices.py), so τ_tree ≥ τ_chain → a.  Published: −17 … −19 %."""
    if topk <= 1 or k <= 0 or model == "chain":
        return a
    w = min(float(topk), (tree_tokens(k, topk, n) - 1) / k)
    return 1.0 - (1.0 - a) ** max(1.0, w)


def spec_tree_tau(a: float, k: int, topk: int, n: int, model: str = "indep") -> float:
    """0.225: expected tokens per verify of a chain / tree.  A tree of n verified tokens is at most n − 1 deep, so its
    accepted path is capped at min(k, n − 1) drafts (0.224 counted k depths even when n − 1 < k).
    0.226 "rank": the positional-acceptance tree (rank_tree_tau), held inside the band [chain, indep]."""
    d = min(k, tree_tokens(k, topk, n) - 1) if topk > 1 else k
    if model == "rank" and topk > 1 and k > 0:
        lo, hi = spec_tree_tau(a, k, topk, n, "chain"), spec_tree_tau(a, k, topk, n, "indep")
        return min(hi, max(lo, rank_tree_tau(a, k, topk, tree_tokens(k, topk, n))))
    return spec_expected_tokens(d, spec_tree_accept(a, k, topk, n, "chain" if model == "rank" else model))


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
    wire_d2d: float = 0.0   # 0.78: bytes one rank sends on the wire per tier (per algorithm; energy) — the *_bytes
    wire_link: float = 0.0  #       fields above stay payload (timing / KV-transfer shares)
    wire_net: float = 0.0
    t_pipe: float = 0.0
    t_kern: float = 0.0     # 0.137: part of t_sync that is GPU-stream kernel overhead (kernel_us, layer_overhead_us)
    t_disp: float = 0.0     # 0.203: part of t_sync that is NPU dispatch (Chip.dispatch)
    t_fixed: float = 0.0    # 0.203 (0.204 renamed from t_static): fixed per-layer overhead time in t_sync (layer_overhead_us) — not static power
    # 0.104 (opt-in Chip.attn_vec_overlap): vector seconds of the attention core (softmax / exp) and whether the
    # flash kernel overlaps them with its MMAs (FlashAttention-3 ping-pong + intra-warpgroup pipelining)
    t_vec_attn: float = 0.0
    attn_ovl: bool = False     # 0.98: exec_overlap "pipe" — op-sequence makespan on DMA + compute (core/overlap.py)

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
        if self.overlap == "pipe":       # 0.98: bracketed by the full-overlap and the serial bounds
            lo = max(self.t_compute, self.t_dram, self.t_slc, self.t_link) + self.t_sync
            hi = self.t_array + self.t_vector + max(self.t_dram, self.t_slc) + self.t_link + self.t_sync
            return min(max(self.t_pipe + self.t_sync, lo), hi)
        if self.overlap != "stage":
            if self.overlap == "serial":
                return self.t_array + self.t_vector + max(self.t_dram, self.t_slc) + self.t_link + self.t_sync
            if self.attn_ovl and self.overlap in ("kernel", "tbo"):
                # 0.104: exp / softmax inside the flash kernel overlap its MMAs (FA3); other vector kernels stay serial
                core = (max(self.t_array - self.t_arr_attn, self.t_dram - self.t_dram_kv)
                        + max(self.t_arr_attn, self.t_vec_attn, self.t_dram_kv) + self.t_vector - self.t_vec_attn)
            else:
                core = max(self.t_array - self.t_arr_attn, self.t_dram - self.t_dram_kv) + max(self.t_arr_attn, self.t_dram_kv)
                core = core + self.t_vector if self.overlap in ("kernel", "tbo") else max(self.t_vector, core)
            if self.overlap == "tbo":
                # 0.137: per-kernel / per-layer overhead runs on the compute stream, so the other micro-batch's
                # all-to-all hides it like the kernels themselves (was added after the max: a link-bound TBO stage paid it)
                return max(core + self.t_kern, self.t_slc, self.t_link) + self.t_sync - self.t_kern
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
