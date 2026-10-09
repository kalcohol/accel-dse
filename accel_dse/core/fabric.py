"""L5b — topology-aware collectives on the three-tier fabric (0.59, 「假设」; off by default: ``scenario.fabric``).

With ``fabric.enabled`` = false every collective takes the 0.50 path (``schedule.fabric_collective``) bit for bit.
With it on, each collective of g ranks (levels n₀ D2D · n₁ scale-up · n₂ network as in schedule.py; K_i = Π_{j<i} n_j
ranks of the group inside one unit of tier i) is costed by α-β per algorithm and the cheapest is taken
(``fabric.algo = auto``) or the user's choice:

  allreduce, payload B per rank (NCCL tuner model: ring busBw·n/(2(n−1)), tree busBw/2):
    ring  (flat, multi-channel: tier i's K_i ports carry one ring channel each)
          bw = 2(g−1)/g · B · maxᵢ 1/(K_i·β_i)                 steps 2(g−1), of which 2(U_i − 1) − Σ_{j>i} cross tier i
    tree  (NCCL: chains inside the node, double binary tree across nodes; on one node the tree is a chain)
          bw = 2B · max( max_{i<top} 1/β_i , 1/(K_top·β_top) )  latency 2(K_top − 1) chain + 2⌈log₂ n_net⌉ tree steps
          (one node: bw = 2B · maxᵢ 1/β_i, latency 2(g − 1) chain steps)
    hier  reduce-scatter up, allreduce on the top level (ring / tree / in-network), all-gather down
          bw = Σ_{i<top} 2(n_i−1)/n_i · B/K_i / β_i + top(B/K_top)
          top: ring 2(n−1)/n·b/β, tree 2b/β (network top only), in-network b/β (switch aggregation: each rank sends and receives b once)
  allgather: ring (g−1)·B·maxᵢ 1/(K_i·β_i), steps g−1; hier = the 0.50 formula, steps Σ(n_i − 1).
  alltoall:  direct (every tier at once, the 0.50 shares) with topology factors; latency one hop.
  α = α_launch (max of the tiers' per-collective α) + Σ steps × hop_tier.

Topology factors divide the tier's β (≥ 1):
  scale-up (``link.topology``, domain n packages = node_cards / package, or the replica's cards on one node):
    switch 1;  full_mesh (n−1)/(k−1) (only the k−1 direct links among the k members are usable);
    ring   ring-like patterns 1 if the members wrap the ring else 2 (a line), × stride (concurrent groups share links);
           alltoall from the maximum directed-link load (shortest path) of the concurrent groups;
    torus2d (X × Y, 4 ports of β/4): ring-like 4 / usable ports (2 per dimension the members wrap, 1 per dimension
           they only partly span) × stride along the dimension; alltoall from dimension-order routing link loads.
  network (``fabric.net_topology``; oversubscription r = down:up of the leaf uplinks, spine non-blocking):
    all leaf cards communicate at once (SPMD), so a pattern whose per-card share f of traffic leaves the leaf runs
    at max(1, r·f):  ring f = 1/m (m = group nodes per leaf, n₂ > m),  tree f = cut tree edges / (n₂ − 1),
    alltoall f = (n₂ − m)/(n₂ − 1), in-network reduction f = 0.  Rail: same with m = nodes per rail switch;
    alltoall's cross-rail share (k₂ − 1)/k₂ of the network bytes also crosses the scale-up tier first (PXN).
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

from .hardware import Fabric, Link

_LOG: dict | None = None        # collector for fabric_report (None = off)
_MULT = 1                       # layers sharing the op list being summed (fabric_report counts)


def leaf_nodes(fab: Fabric, node_cards: int) -> int:
    """Nodes under one leaf (fat_tree) / per rail switch (rail)."""
    if fab.leaf_nodes:
        return fab.leaf_nodes
    down = int(fab.switch_radix * fab.oversub / (1.0 + fab.oversub))
    return max(1, down // max(1, node_cards)) if fab.net_topology == "fat_tree" else max(1, down)


def torus_dims(n: int, x: int = 0) -> tuple[int, int]:
    if x and n % x == 0:
        return x, n // x
    x = max(d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0)
    return n // x, x


def _dim_ports(extent: int, full: int) -> int:
    if extent <= 1:
        return 0
    return 2 if extent == full and full > 2 else 1


@lru_cache(maxsize=4096)
def su_ring_factor(topo: str, n: int, k: int, s: int, tx: int = 0) -> float:
    """β divisor of a ring / chain / tree pattern over k members (stride s) of an n-package scale-up domain."""
    if k <= 1 or topo == "switch" or n <= 1:
        return 1.0
    if topo == "full_mesh":
        return (n - 1) / (k - 1)
    if topo == "ring":
        return (1.0 if k * s >= n else 2.0) * s
    X, Y = torus_dims(n, tx)
    pos = [(m * s) % n for m in range(k)]
    ex, ey = len({p % X for p in pos}), len({p // X for p in pos})
    ports = _dim_ports(ex, X) + _dim_ports(ey, Y)
    share = s if s < X else max(1, s // X)
    return (4.0 / ports if ports else 1.0) * share


def _ring_route(a: int, b: int, n: int):
    d = (b - a) % n
    step = 1 if (d < n - d or (d == n - d and a % 2 == 0)) else -1
    x = a
    while x != b:
        y = (x + step) % n
        yield (x, y)
        x = y


def _torus_route(a: int, b: int, X: int, Y: int):
    ax, ay, bx, by = a % X, a // X, b % X, b // X
    for x, y in _ring_route(ax, bx, X):           # X first, in row ay
        yield ("x", ay, x, y)
    for x, y in _ring_route(ay, by, Y):           # then Y, in column bx
        yield ("y", bx, x, y)


@lru_cache(maxsize=4096)
def su_a2a_factor(topo: str, n: int, k: int, s: int, tx: int = 0) -> float:
    """β divisor of an all-to-all over k members (stride s): max directed-link load of the s concurrent groups of the
    first k·s packages under shortest-path (ring) / dimension-order (torus) routing, vs a non-blocking switch."""
    if k <= 1 or topo == "switch" or n <= 1:
        return 1.0
    if topo == "full_mesh":
        return (n - 1) / (k - 1)
    load: dict = defaultdict(int)
    X, Y = torus_dims(n, tx) if topo == "torus2d" else (n, 1)
    for j in range(s):
        grp = [(j + m * s) % n for m in range(k)]
        for a in grp:
            for b in grp:
                if a != b:
                    for ln in (_ring_route(a, b, n) if topo == "ring" else _torus_route(a, b, X, Y)):
                        load[ln] += 1
    ports = 2 if topo == "ring" else 4
    t = max(load.values()) * ports / k            # × B/β: message B/k on a port of β/ports
    return max(1.0, t / ((k - 1) / k))


def _btree_parent(rank: int, n: int) -> int:
    """NCCL ncclGetBtree parent (−1 for the root)."""
    if rank == 0:
        return -1
    bit = 1
    while bit < n and not (bit & rank):
        bit <<= 1
    up = (rank ^ bit) | (bit << 1)
    if up >= n:
        up = rank ^ bit
    return up


def _dtree_parent2(rank: int, n: int) -> int:
    """Parent in NCCL's second tree (ncclGetDtree): the mirror (n even) / one-shift (n odd) of the first."""
    if n % 2:
        u = _btree_parent((rank - 1) % n, n)
        return -1 if u == -1 else (u + 1) % n
    u = _btree_parent(n - 1 - rank, n)
    return -1 if u == -1 else n - 1 - u


@lru_cache(maxsize=4096)
def tree_cut_share(n: int, m: int) -> float:
    """Share of the NCCL double-binary-tree edges over n nodes (m consecutive nodes per leaf) that leave a leaf.
    0.61: both trees, each carrying half of the bytes (0.59 / 0.60 counted the first tree only; same value whenever
    the mirror maps leaves onto leaves, i.e. m | n with n even)."""
    if n <= 1 or m >= n:
        return 0.0
    cut1 = sum(1 for r in range(1, n) if r // m != _btree_parent(r, n) // m)
    cut2 = sum(1 for r in range(n) if (q := _dtree_parent2(r, n)) != -1 and r // m != q // m)
    return 0.5 * (cut1 + cut2) / (n - 1)


def pod_nodes(fab: Fabric, node_cards: int) -> int | None:
    """Nodes per pod of a three-tier (leaf / spine / core) fat-tree (0.60); None for two tiers / rail."""
    if fab.net_tiers != 3 or fab.net_topology != "fat_tree":
        return None
    if fab.pod_nodes:
        return fab.pod_nodes
    spine_down = int(fab.switch_radix * fab.oversub_spine / (1.0 + fab.oversub_spine))
    return leaf_nodes(fab, node_cards) * max(1, spine_down)


def _share(pattern: str, n2: int, m: int) -> float:
    """Per-card share of a pattern's network traffic that leaves a block of m consecutive group nodes."""
    if m >= n2:
        return 0.0
    return {"ring": 1.0 / m, "tree": tree_cut_share(n2, m), "alltoall": (n2 - m) / (n2 - 1)}.get(pattern, 1.0)


def _block(n2: int, size: int, s_n: int) -> int:
    """Group nodes inside one block of ``size`` consecutive nodes (consecutive group nodes s_n apart)."""
    return max(1, min(n2, size // max(1, s_n))) if size >= s_n else 1


def net_shares(fab: Fabric, pattern: str, n2: int, s_n: int, node_cards: int) -> tuple[float, float]:
    """(share leaving the leaf, share leaving the pod) of a pattern over n2 nodes (0.60 report; pod share 0 for
    two tiers)."""
    if n2 <= 1 or pattern == "innet":
        return 0.0, 0.0
    f1 = _share(pattern, n2, _block(n2, leaf_nodes(fab, node_cards), s_n))
    P = pod_nodes(fab, node_cards)
    return f1, (_share(pattern, n2, _block(n2, P, s_n)) if P else 0.0)


def net_factor(fab: Fabric, pattern: str, n2: int, s_n: int, node_cards: int) -> float:
    """β divisor of the network tier for a pattern over n2 nodes whose consecutive nodes are s_n nodes apart.
    Two tiers: max(1, r·f_leaf).  Three tiers (0.60): max(1, r₁·f_leaf, r₁·r₂·f_pod) — the spine → core uplinks
    carry 1/(r₁·r₂) of a card's bandwidth."""
    P = pod_nodes(fab, node_cards)
    r2 = fab.oversub_spine if P else 1.0
    if (fab.oversub <= 1.0 and r2 <= 1.0) or n2 <= 1 or pattern == "innet":
        return 1.0
    L = leaf_nodes(fab, node_cards)
    m = max(1, min(n2, L // max(1, s_n))) if L >= s_n else 1
    out = 1.0
    if m < n2:
        f = {"ring": 1.0 / m, "tree": tree_cut_share(n2, m), "alltoall": (n2 - m) / (n2 - 1)}.get(pattern, 1.0)
        out = max(1.0, fab.oversub * f)
    if P:
        p = _block(n2, P, s_n)
        if p < n2:
            out = max(out, fab.oversub * r2 * _share(pattern, n2, p))
    return out


def _levels(group, sys, stride, k_pkg, k_node):
    from .schedule import _members
    d2d = sys.d2d if sys.package_cards > 1 else None
    net = sys.net if sys.node_cards > 0 else None
    k1 = k_pkg if k_pkg is not None else (1 if d2d is None else _members(sys.package_cards, group, stride))
    k1 = math.gcd(max(1, k1), group)
    if net is None:
        k2 = group
    else:
        k2 = math.gcd(max(1, k_node if k_node is not None else _members(sys.node_cards, group, stride)), group)
        if k2 % k1:
            k2 = k1
    lv = [(n, ln, t) for n, ln, t in ((k1, d2d, "d2d"), (k2 // k1, sys.link, "link"), (group // k2, net, "net"))
          if n > 1]
    return lv, k1, k2


def _hop(fab: Fabric, tier: str, tree: bool = False) -> float:
    return 1e-6 * {"d2d": fab.hop_d2d_us, "link": fab.hop_link_us,
                   "net": fab.hop_net_tree_us if tree else fab.hop_net_us}[tier]


def _steps_split(levels, total_steps_of):
    """Ring steps per tier: steps crossing tier i = f(U_i) − Σ_{j>i}, U_i = Π_{j≥i} n_j (units of tier i−1)."""
    out = {}
    acc = 0
    for i in range(len(levels) - 1, -1, -1):
        U = math.prod(n for n, _, _ in levels[i:])
        st = total_steps_of(U) - acc
        out[levels[i][2]] = st
        acc += st
    return out


PROTOS = ("LL", "LL128", "Simple")
_PROTO_EFF = {"LL": 0.5, "LL128": 120.0 / 128.0, "Simple": 1.0}
# 0.61 「假设」: per-step latency ratios LL : LL128 : Simple from the NCCL tuner's hwLat table (tuning.cc, open-source
# defaults; NVLink ring .6 / 1.9 / 3.4 µs, NET ring 2.7 / 4.0 / 14 µs, NET tree 5.0 / 8.5 / 14 µs) applied to the
# user's hop_* (= the LL values); the D2D tier takes the scale-up ratios
_PROTO_LAT = {"link": (1.0, 1.9 / 0.6, 3.4 / 0.6), "net": (1.0, 4.0 / 2.7, 14.0 / 2.7),
              "tree": (1.0, 8.5 / 5.0, 14.0 / 5.0)}


def base_algo(name: str) -> str:
    """Algorithm of a candidate name ("ring/LL128" → "ring"; 0.61 protocol suffix)."""
    return name.split("/", 1)[0]


def _protos(fab: Fabric) -> tuple:
    p = getattr(fab, "protocol", "off")
    return (None,) if p == "off" else PROTOS if p == "auto" else (p,)


def _hop_p(fab: Fabric, tier: str, proto, tree: bool = False) -> float:
    h = _hop(fab, tier, tree)
    if proto is None:
        return h
    return h * _PROTO_LAT["tree" if tree else ("net" if tier == "net" else "link")][PROTOS.index(proto)]


def switch_extra(fab: Fabric, n_net: int, s_net: int, node_cards: int) -> tuple[float, float, int]:
    """0.61 「假设」: extra seconds per network step from the switch tiers it crosses, for a pattern over n_net group
    nodes (consecutive s_net nodes apart).  Returns (ring / flat step: + hop_spine if the group spans leaves,
    + hop_core if it spans pods;  binary tree: Σ over the ⌈log₂ n⌉ levels — level j joins ranks 2^j apart, so it
    leaves the leaf once 2^j ≥ m (m group nodes per leaf): max(0, ⌈log₂ n⌉ − ⌈log₂ m⌉) levels pay hop_spine, likewise
    pods;  in-network reduction levels (1 leaf, 2 spine, 3 core))."""
    if n_net <= 1 or node_cards <= 0 or (not fab.hop_spine_us and not fab.hop_core_us):
        return 0.0, 0.0, 0
    m = _block(n_net, leaf_nodes(fab, node_cards), s_net)
    P = pod_nodes(fab, node_cards)
    p = _block(n_net, P, s_net) if P else n_net
    ring = 1e-6 * (fab.hop_spine_us * (n_net > m) + fab.hop_core_us * (n_net > p))
    L = math.ceil(math.log2(n_net))
    tree = 1e-6 * (fab.hop_spine_us * max(0, L - math.ceil(math.log2(m))) +
                   fab.hop_core_us * max(0, L - math.ceil(math.log2(p))))
    return ring, tree, (3 if n_net > p else 2 if n_net > m else 1)


def candidates(kind: str, payload: float, group: int, sys, stride: int = 1, k_pkg: int | None = None,
               k_node: int | None = None) -> tuple[dict, dict, list]:
    """Every algorithm of one collective (0.60): {name: (bw s, α s, {tier: busy s})}, {name: {tier: byte share}},
    and the levels.  bw = the collective's own bandwidth time (max over tiers for single-phase ring / tree /
    all-to-all, sum of the phases for hier); busy = the seconds each tier's ports are occupied.
    0.61: ``fabric.protocol`` ≠ off → one candidate per algorithm × protocol ("ring/LL128"): bw and busy ÷ the
    protocol's efficiency, per-step latency × its ratio; hop_spine / hop_core added to the network steps that leave
    the leaf / pod; NVLS / SHARP-class all-gather / reduce-scatter ("innet") with ``innet_reduce``."""
    fab: Fabric = sys.fabric
    lv, k1, k2 = _levels(group, sys, stride, k_pkg, k_node)
    if not lv:
        return {}, {}, lv
    pkg = sys.package_cards if sys.package_cards > 1 else 1
    dom = max(1, (sys.node_cards if sys.node_cards > 0 else max(sys.cards, group * stride)) // pkg)
    s_link = max(1, (stride * k1) // pkg)
    s_net = max(1, (stride * k2) // sys.node_cards) if sys.node_cards > 0 else 1
    topo = sys.link.topology
    a_launch = max(ln.alpha_us for _, ln, _ in lv) * 1e-6
    K = [math.prod(n for n, _, _ in lv[:i]) for i in range(len(lv))]
    T = [t for _, _, t in lv]

    def beta(i: int, pattern: str) -> float:
        n, ln, t = lv[i]
        b = ln.GBps * 1e9
        if t == "link":
            f = su_a2a_factor(topo, dom, n, s_link, fab.torus_x) if pattern == "alltoall" else \
                1.0 if pattern == "innet" else su_ring_factor(topo, dom, n, s_link, fab.torus_x)
        elif t == "net":
            f = net_factor(fab, pattern, n, s_net, sys.node_cards)
        else:
            f = 1.0
        return b / f

    def single(busy: list) -> tuple:          # one phase, every tier at once: bw = slowest tier
        return max(busy), dict(zip(T, busy))

    top = len(lv) - 1
    nt, _, tt = lv[top]
    x_ring, x_tree, x_lvl = switch_extra(fab, nt, s_net, sys.node_cards) if tt == "net" else (0.0, 0.0, 0)
    # 0.61 NVLS class (all-gather / reduce-scatter): the switched scale-up tier with innet_reduce = net+link
    # (SHARP-class network aggregation stays allreduce-only, as in 0.59)
    inn = [t == "link" and fab.innet_reduce == "net+link" and topo == "switch" for _, _, t in lv]
    cands: dict = {}
    tops_pick = ""
    for pr in _protos(fab):
        def H(t: str, tree: bool = False) -> float:     # per-step latency of tier t (switch tiers on the network)
            h = _hop_p(fab, t, pr, tree)
            if t == "net" and (x_ring or x_tree):
                h += x_tree if tree else x_ring
            return h
        c: dict = {}
        if kind == "allreduce":
            _, busy = single([2 * (group - 1) / group * payload / (K[i] * beta(i, "ring")) for i in range(len(lv))])
            bw = 2 * (group - 1) / group * payload * max(1.0 / (K[i] * beta(i, "ring")) for i in range(len(lv)))
            st = _steps_split(lv, lambda U: 2 * (U - 1))
            c["ring"] = (bw, a_launch + sum(st[t] * H(t) for t in st), busy)
            if tt == "net":     # NCCL tree: chains inside the node, double binary tree across nodes
                _, busy = single([2 * payload / beta(i, "ring") for i in range(top)]
                                 + [2 * payload / (K[top] * beta(top, "tree"))])
                bw = 2 * payload * max([1.0 / beta(i, "ring") for i in range(top)] + [1.0 / (K[top] * beta(top, "tree"))])
                stl = _steps_split(lv[:top], lambda U: 2 * (U - 1)) if top else {}
                lat = sum(stl[t] * H(t) for t in stl) + 2 * math.ceil(math.log2(nt)) * _hop_p(fab, tt, pr, tree=True) \
                    + (2 * x_tree if x_tree else 0.0)
            else:               # one node (nNodes = 1): the NCCL tree is a chain through all ranks
                _, busy = single([2 * payload / beta(i, "ring") for i in range(len(lv))])
                bw = 2 * payload * max(1.0 / beta(i, "ring") for i in range(len(lv)))
                stl = _steps_split(lv, lambda U: 2 * (U - 1))
                lat = sum(stl[t] * H(t) for t in stl)
            c["tree"] = (bw, a_launch + lat, busy)
            lo = [2 * (lv[i][0] - 1) / lv[i][0] * payload / K[i] / beta(i, "ring") for i in range(top)]
            lo_lat = sum(2 * (lv[i][0] - 1) * H(lv[i][2]) for i in range(top))
            b_top = payload / K[top]
            tops = {"ring": (2 * (nt - 1) / nt * b_top / beta(top, "ring"), 2 * (nt - 1) * H(tt))}
            if tt == "net":
                tops["tree"] = (2 * b_top / beta(top, "tree"), 2 * math.ceil(math.log2(nt)) * _hop_p(fab, tt, pr, tree=True)
                                + (2 * x_tree if x_tree else 0.0))
            if (tt == "net" and fab.innet_reduce != "off") or \
                    (tt == "link" and fab.innet_reduce == "net+link" and topo == "switch"):
                P = pod_nodes(fab, sys.node_cards) if tt == "net" else None
                lvl = (3 if P and nt > P else 2 if nt > leaf_nodes(fab, sys.node_cards) else 1) if tt == "net" else 1
                xi = 0.0
                if tt == "net" and (fab.hop_spine_us or fab.hop_core_us):
                    xi = 2e-6 * (fab.hop_spine_us * (x_lvl >= 2) + fab.hop_core_us * (x_lvl >= 3))
                tops["innet"] = (b_top / beta(top, "innet"), 2 * lvl * _hop_p(fab, tt, pr) + xi)
            tn = min(tops, key=lambda x: sum(tops[x]))
            lo_bw = sum(lo)
            c["hier"] = (lo_bw + tops[tn][0], a_launch + lo_lat + tops[tn][1], dict(zip(T, lo + [tops[tn][0]])))
            tops_pick = tn
        elif kind in ("allgather", "reducescatter"):
            # payload = one rank's shard (all-gather input / reduce-scatter output): every rank moves (g − 1) shards
            _, busy = single([(group - 1) * payload / (K[i] * beta(i, "ring")) for i in range(len(lv))])
            bw = (group - 1) * payload * max(1.0 / (K[i] * beta(i, "ring")) for i in range(len(lv)))
            st = _steps_split(lv, lambda U: U - 1)
            c["ring"] = (bw, a_launch + sum(st[t] * H(t) for t in st), busy)
            ph = [(n - 1) * math.prod(m for m, _, _ in lv[i + 1:]) * payload / beta(i, "ring") for i, (n, _, _) in enumerate(lv)]
            c["hier"] = (sum(ph), a_launch + sum((n - 1) * H(t) for n, _, t in lv), dict(zip(T, ph)))
            if any(inn):        # 0.61 NVLS class: the switched scale-up level is one switch round trip
                rs = kind == "reducescatter"   # in-switch reduction: each rank sends its n shards once (n vs n − 1)
                phi, lat = [], 0.0             # all-gather multicast: each rank still receives n − 1 shards
                for i, (n, _, t) in enumerate(lv):
                    M = math.prod(m for m, _, _ in lv[i + 1:])
                    if inn[i]:
                        phi.append((n if rs else n - 1) * M * payload / beta(i, "innet"))
                        lat += 2 * _hop_p(fab, t, pr)
                    else:
                        phi.append(ph[i])
                        lat += (n - 1) * H(t)
                c["innet"] = (sum(phi), a_launch + lat, dict(zip(T, phi)))
            tops_pick = "ring"
        elif kind == "alltoall":
            vols = [(n - 1) * K[i] / group * payload for i, (n, ln, t) in enumerate(lv)]
            if fab.net_topology == "rail" and tt == "net" and k2 > 1:
                for i, (_, _, t) in enumerate(lv):
                    if t == "link":             # PXN: cross-rail network bytes first hop over the scale-up tier
                        vols[i] += vols[top] * (k2 - 1) / k2
            bw, busy = single([v / beta(i, "alltoall") for i, v in enumerate(vols)])
            c["direct"] = (bw, a_launch + max(H(t) for _, _, t in lv), busy)
        else:                                   # p2p-like: the outermost tier
            x = payload / beta(top, "ring")
            c["p2p"] = (x, a_launch + H(tt), {tt: x})
        for name, (bw, a, busy) in c.items():
            if pr is None:
                cands[name] = (bw, a, busy)
            else:
                e = _PROTO_EFF[pr]
                cands[f"{name}/{pr}"] = (bw / e, a, {t: x / e for t, x in busy.items()})
    # byte shares for the energy counts: hierarchical volumes (alltoall: per-tier destinations; ring: link mix)
    if kind == "alltoall":
        vol = [(n - 1) * K[i] / group for i, (n, _, _) in enumerate(lv)]
    elif kind in ("allgather", "reducescatter"):
        vol = [(n - 1) * math.prod(m for m, _, _ in lv[i + 1:]) for i, (n, _, _) in enumerate(lv)]
    else:
        vol = [2 * (n - 1) / n / K[i] for i, (n, _, _) in enumerate(lv)]
    tot = sum(vol) or 1.0
    hshare = {t: v / tot for v, (_, _, t) in zip(vol, lv)}
    shares = {}
    for name in cands:
        if kind in ("allreduce", "allgather", "reducescatter") and base_algo(name) == "ring":
            U = [math.prod(n for n, _, _ in lv[i:]) for i in range(len(lv))] + [1]
            hops = [U[i] - U[i + 1] for i in range(top)] + [nt]         # ring hops per tier (sum = g)
            shares[name] = {t: h / group for h, (_, _, t) in zip(hops, lv)}
        else:
            shares[name] = hshare
    meta = {"levels": lv, "top": tops_pick, "s_net": s_net, "nt": nt, "tt": tt}
    return cands, shares, meta


_PATTERN = {"ring": "ring", "tree": "tree", "direct": "alltoall", "p2p": "ring", "innet": "innet"}


def pick(cands: dict, fab: Fabric) -> str:
    if fab.algo not in ("auto", "auto_overlap"):
        if fab.algo in cands:
            return fab.algo
        sub = [k for k in cands if base_algo(k) == fab.algo]     # 0.61: the user's algorithm, fastest protocol
        if sub:
            return min(sub, key=lambda x: cands[x][0] + cands[x][1])
    return min(cands, key=lambda x: cands[x][0] + cands[x][1])


def collective_full(kind: str, payload: float, group: int, sys, stride: int = 1, k_pkg: int | None = None,
                    k_node: int | None = None):
    """(bw, α, D2D share, net share, {tier: busy}, candidates | None) — candidates (with shares) are returned for
    ``algo = auto_overlap`` so the stage can re-pick (evaluate._stage_link)."""
    if group <= 1 or payload <= 0:
        return 0.0, 0.0, 0.0, 0.0, {}, None
    fab: Fabric = sys.fabric
    cands, shares, meta = candidates(kind, payload, group, sys, stride, k_pkg, k_node)
    if not cands:
        return 0.0, 0.0, 0.0, 0.0, {}, None
    name = pick(cands, fab)
    bw, a, busy = cands[name]
    sh = shares[name]
    key = (kind, group, stride, k_pkg, k_node, round(payload))
    if _LOG is not None:
        lv = meta["levels"]
        e = _LOG.setdefault(key, {"kind": kind, "group": group, "bytes": payload, "count": 0,
                                  "levels": [[t, n] for n, _, t in lv], "algo": name,
                                  "top": meta["top"] if base_algo(name) == "hier" else "",
                                  "cands": {k: [v[0], v[1]] for k, v in cands.items()},
                                  "busy": {k: dict(v[2]) for k, v in cands.items()},
                                  "net_frac": {k: v.get("net", 0.0) for k, v in shares.items()}, "hier_top": meta["top"]})
        if meta["tt"] == "net":
            e["net_tiers"] = {k: net_shares(fab, "tree" if (base_algo(k) == "hier" and meta["top"] == "tree") else
                                            "innet" if (base_algo(k) == "hier" and meta["top"] == "innet") else
                                            _PATTERN.get(base_algo(k), "ring"), meta["nt"], meta["s_net"], sys.node_cards)
                              for k in cands}
        e["count"] += _MULT
    full = None
    if fab.algo == "auto_overlap" and len(cands) > 1:
        full = (key, payload, name,
                {k: (v[0], v[1], v[2], shares[k].get("d2d", 0.0), shares[k].get("net", 0.0)) for k, v in cands.items()})
    return bw, a, sh.get("d2d", 0.0), sh.get("net", 0.0), busy, full


def collective(kind: str, payload: float, group: int, sys, stride: int = 1, k_pkg: int | None = None,
               k_node: int | None = None) -> tuple[float, float, float, float]:
    """(bandwidth s, exposed-latency s, D2D share, network share) of one collective with ``sys.fabric`` on."""
    return collective_full(kind, payload, group, sys, stride, k_pkg, k_node)[:4]


def _p2p_link_factor(topo: str, n: int, S: int, tx: int = 0) -> float:
    """β divisor of a PP hand-off inside a ring / torus / full-mesh scale-up domain of n packages (0.60): every
    package a of a stage sends to a + S at once (no wrap past the domain, 「假设」 conservative), shortest-path (ring) /
    dimension-order (torus) routing, ports of β/2 (ring) or β/4 (torus); full mesh: one direct link of β/(n − 1)."""
    if topo == "switch" or n <= 1 or S <= 0 or S >= n:
        return 1.0
    if topo == "full_mesh":
        return float(n - 1)
    return _p2p_shift_factor(topo, n, S, tx)


@lru_cache(maxsize=4096)
def _p2p_shift_factor(topo: str, n: int, S: int, tx: int) -> float:
    load: dict = defaultdict(int)
    X, Y = torus_dims(n, tx) if topo == "torus2d" else (n, 1)
    for a in range(n - S):
        for ln in (_ring_route(a, a + S, n) if topo == "ring" else _torus_route(a, a + S, X, Y)):
            load[ln] += 1
    return max(1.0, max(load.values()) * (2 if topo == "ring" else 4)) if load else 1.0


def p2p(payload: float, tier: str, ln: Link, sys, src_card: int, dst_card: int, stage_cards: int = 0) -> float:
    """Bandwidth seconds of a PP hand-off on ``tier``: the network tier pays r when the two cards sit under
    different leaves (all cards of the stage send at once, so every flow shares the oversubscribed uplinks);
    0.60: r₁·r₂ across pods (three tiers); on the scale-up tier a ring / torus / full-mesh domain pays its hop
    load (``_p2p_link_factor``)."""
    b = ln.GBps * 1e9
    fab = sys.fabric
    if tier == "net" and sys.node_cards > 0:
        P = pod_nodes(fab, sys.node_cards)
        r2 = fab.oversub_spine if P else 1.0
        if fab.oversub > 1 or r2 > 1:
            L = leaf_nodes(fab, sys.node_cards)
            ns, nd = src_card // sys.node_cards, dst_card // sys.node_cards
            if P and ns // P != nd // P:
                b /= fab.oversub * r2
            elif ns // L != nd // L:
                b /= fab.oversub
    elif tier == "link" and stage_cards and sys.link.topology != "switch":
        pkg = sys.package_cards if sys.package_cards > 1 else 1
        dom = max(1, (sys.node_cards if sys.node_cards > 0 else sys.cards) // pkg)
        b /= _p2p_link_factor(sys.link.topology, dom, max(1, stage_cards // pkg), fab.torus_x)
    return payload / b


def p2p_alpha(payload_s: float, tier: str, ln: Link, sys, src_card: int, dst_card: int,
              with_proto: bool = False) -> tuple:
    """(bw s, α s) of a PP hand-off whose bandwidth seconds are ``payload_s`` (0.61): α = launch + one step of the
    tier, + hop_spine / hop_core when the two cards' nodes sit under different leaves / pods; with
    ``fabric.protocol`` the fastest (or the chosen) protocol: bw ÷ efficiency, step × its latency ratio."""
    fab = sys.fabric
    x = 0.0
    if tier == "net" and sys.node_cards > 0 and (fab.hop_spine_us or fab.hop_core_us):
        ns, nd = src_card // sys.node_cards, dst_card // sys.node_cards
        L = leaf_nodes(fab, sys.node_cards)
        P = pod_nodes(fab, sys.node_cards)
        x = 1e-6 * (fab.hop_spine_us * (ns // L != nd // L) + fab.hop_core_us * bool(P and ns // P != nd // P))
    best = None
    for pr in _protos(fab):
        bw = payload_s if pr is None else payload_s / _PROTO_EFF[pr]
        a = ln.alpha_us * 1e-6 + _hop_p(fab, tier, pr) + (x if x else 0.0)
        if best is None or bw + a < best[0] + best[1]:
            best = (bw, a, pr)
    return best if with_proto else best[:2]


def kv_hop_extra(fab: Fabric, node_cards: int, total_cards: int) -> float:
    """0.61 「假设」: mean switch-tier latency of a PD KV transfer — random prefill → decode pairs leave the leaf with
    share 1 − m/n (hop_spine) and the pod with 1 − P/n (hop_core), n nodes."""
    if node_cards <= 0 or (not fab.hop_spine_us and not fab.hop_core_us):
        return 0.0
    n = max(1, -(-total_cards // node_cards))
    m = leaf_nodes(fab, node_cards)
    P = pod_nodes(fab, node_cards)
    return 1e-6 * (fab.hop_spine_us * max(0.0, 1.0 - m / n) + fab.hop_core_us * (max(0.0, 1.0 - P / n) if P else 0.0))


def kv_factor(fab: Fabric, node_cards: int, total_cards: int) -> float:
    """β divisor of the PD KV hand-off over the network: random prefill → decode card pairs across n nodes; the share
    1 − m/n of pairs leaves the leaf (m nodes per leaf / rail switch), all cards streaming at once.  0.60 three tiers:
    also 1 − P/n leaves the pod at r₁·r₂."""
    P = pod_nodes(fab, node_cards) if node_cards > 0 else None
    r2 = fab.oversub_spine if P else 1.0
    if (fab.oversub <= 1.0 and r2 <= 1.0) or node_cards <= 0:
        return 1.0
    n = max(1, -(-total_cards // node_cards))
    m = leaf_nodes(fab, node_cards)
    out = 1.0
    if m < n:
        out = max(1.0, fab.oversub * (1.0 - m / n))
    if P and P < n:
        out = max(out, fab.oversub * r2 * (1.0 - P / n))
    return out


_ALGO_ZH = {"ring": "ring（扁平环）", "tree": "tree（双二叉树）", "hier": "hier（分层）", "direct": "direct（直接 all-to-all）",
            "p2p": "p2p", "innet": "innet（交换机内归约 / 多播，NVLS 类）"}
_KIND_ZH = {"allreduce": "allreduce", "allgather": "allgather", "reducescatter": "reduce-scatter", "alltoall": "all-to-all",
            "p2p": "p2p"}


def fabric_report(scn, res=None) -> dict | None:
    """Per-collective breakdown of one evaluation with ``scn.fabric`` on (UI / CLI table): every distinct collective
    (kind, group, payload) with its tier levels, the chosen algorithm, its bandwidth / latency seconds and the other
    candidates; plus the step time of the same scenario with the fabric model off."""
    global _LOG
    if not scn.fabric.enabled:
        return None
    import dataclasses
    from .evaluate import evaluate
    _LOG = {}
    try:
        r = evaluate(scn)
        log = _LOG
    finally:
        _LOG = None
    try:
        off = evaluate(dataclasses.replace(scn, fabric=dataclasses.replace(scn.fabric, enabled=False)))
        off_ms = off.step * 1e3
    except ValueError:
        off_ms = None
    fab = scn.fabric
    stages = log.pop("_stages", [])
    rows = []
    leaf_b = pod_b = net_b = 0.0
    for e in sorted(log.values(), key=lambda x: -x["count"] * sum(x["cands"][x["algo"]])):
        bw, a = e["cands"][e["algo"]]
        row = {"kind": e["kind"], "group": e["group"], "bytes": e["bytes"], "count": e["count"],
               "levels": e["levels"], "algo": e["algo"], "top": e["hier_top"] if base_algo(e["algo"]) == "hier" and "hier_top" in e
               else e["top"], "bw_us": bw * 1e6, "alpha_us": a * 1e6,
               "cands": {k: {"bw_us": v[0] * 1e6, "alpha_us": v[1] * 1e6} for k, v in e["cands"].items()}}
        if "busy" in e:
            row["busy_us"] = {t: x * 1e6 for t, x in e["busy"][e["algo"]].items()}
        nb = e["count"] * e["bytes"] * (e.get("net_frac", {}).get(e["algo"], 1.0 if e["kind"] == "p2p" and
                                                                  e["levels"][0][0] == "net" else 0.0))
        if "net_tiers" in e:     # 0.60: share of this collective's network traffic leaving the leaf / the pod
            f1, f2 = e["net_tiers"][e["algo"]]
            row["net_leaf_share"], row["net_pod_share"] = f1, f2
            leaf_b += nb * f1
            pod_b += nb * f2
        net_b += nb
        rows.append(row)
    pkg = scn.package_eff
    dom = (scn.node_cards if scn.node_cards > 0 else scn.layout.cards) // max(1, pkg)
    P = pod_nodes(fab, scn.node_cards) if scn.node_cards > 0 else None
    basis = (f"跨节点 {('三层 fat-tree（leaf / spine / core）' if P else 'fat-tree / leaf-spine') if fab.net_topology == 'fat_tree' else 'rail-optimized'}，"
             f"leaf 上行收敛比 {fab.oversub:g}:1，每个 {'leaf' if fab.net_topology == 'fat_tree' else 'rail 交换机'} "
             f"{leaf_nodes(fab, scn.node_cards)} 节点" + (f"；spine → core 收敛比 {fab.oversub_spine:g}:1，每个 pod {P} 节点" if P else "")
             + "；" if scn.node_cards > 0 else "单节点（跨节点层未用）；") + \
        ("各层端口并发（同端口串行，级链路时间 = 最忙端口）；" if fab.overlap == "ports" else "") + \
        ("算法按级暴露时间最小选取（auto_overlap）；" if fab.algo == "auto_overlap" else "") + \
        (f"节点内 scale-up 拓扑 {scn.link.topology}（域内 {dom} 个封装）；算法 {fab.algo}；"
         f"每步时延 D2D / scale-up / 网络 ring / 网络 tree = {fab.hop_d2d_us:g} / {fab.hop_link_us:g} / "
         f"{fab.hop_net_us:g} / {fab.hop_net_tree_us:g} µs（数量级参考 NCCL tuner 默认值，LL 协议）；"
         f"网内归约 {fab.innet_reduce}" + ("（厂商选项「假设」）" if fab.innet_reduce != "off" else "") + "。「假设」")
    return {"rows": rows, "step_ms": r.step * 1e3, "step_ms_off": off_ms, "leaf_nodes": leaf_nodes(fab, scn.node_cards)
            if scn.node_cards > 0 else None, "scaleup_domain": dom, "basis": basis,
            "pod_nodes": P, "net_tiers": 3 if P else 2, "overlap": fab.overlap,
            "net_traffic": ({"net_MB": net_b / 1e6, "leaf_up_share": leaf_b / net_b, "pod_up_share": pod_b / net_b}
                            if net_b > 0 else None),
            "stages": stages}
