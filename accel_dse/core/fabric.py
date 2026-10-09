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


@lru_cache(maxsize=4096)
def tree_cut_share(n: int, m: int) -> float:
    """Share of the NCCL binary-tree edges over n nodes (m consecutive nodes per leaf) that leave a leaf."""
    if n <= 1 or m >= n:
        return 0.0
    cut = sum(1 for r in range(1, n) if r // m != _btree_parent(r, n) // m)
    return cut / (n - 1)


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


def candidates(kind: str, payload: float, group: int, sys, stride: int = 1, k_pkg: int | None = None,
               k_node: int | None = None) -> tuple[dict, dict, list]:
    """Every algorithm of one collective (0.60): {name: (bw s, α s, {tier: busy s})}, {name: {tier: byte share}},
    and the levels.  bw = the collective's own bandwidth time (max over tiers for single-phase ring / tree /
    all-to-all, sum of the phases for hier); busy = the seconds each tier's ports are occupied."""
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
    cands: dict = {}
    tops_pick = ""
    if kind == "allreduce":
        _, busy = single([2 * (group - 1) / group * payload / (K[i] * beta(i, "ring")) for i in range(len(lv))])
        bw = 2 * (group - 1) / group * payload * max(1.0 / (K[i] * beta(i, "ring")) for i in range(len(lv)))
        st = _steps_split(lv, lambda U: 2 * (U - 1))
        cands["ring"] = (bw, a_launch + sum(st[t] * _hop(fab, t) for t in st), busy)
        if tt == "net":     # NCCL tree: chains inside the node, double binary tree across nodes
            _, busy = single([2 * payload / beta(i, "ring") for i in range(top)]
                             + [2 * payload / (K[top] * beta(top, "tree"))])
            bw = 2 * payload * max([1.0 / beta(i, "ring") for i in range(top)] + [1.0 / (K[top] * beta(top, "tree"))])
            stl = _steps_split(lv[:top], lambda U: 2 * (U - 1)) if top else {}
            lat = sum(stl[t] * _hop(fab, t) for t in stl) + 2 * math.ceil(math.log2(nt)) * _hop(fab, tt, tree=True)
        else:               # one node (nNodes = 1): the NCCL tree is a chain through all ranks
            _, busy = single([2 * payload / beta(i, "ring") for i in range(len(lv))])
            bw = 2 * payload * max(1.0 / beta(i, "ring") for i in range(len(lv)))
            stl = _steps_split(lv, lambda U: 2 * (U - 1))
            lat = sum(stl[t] * _hop(fab, t) for t in stl)
        cands["tree"] = (bw, a_launch + lat, busy)
        lo = [2 * (lv[i][0] - 1) / lv[i][0] * payload / K[i] / beta(i, "ring") for i in range(top)]
        lo_lat = sum(2 * (lv[i][0] - 1) * _hop(fab, lv[i][2]) for i in range(top))
        b_top = payload / K[top]
        tops = {"ring": (2 * (nt - 1) / nt * b_top / beta(top, "ring"), 2 * (nt - 1) * _hop(fab, tt))}
        if tt == "net":
            tops["tree"] = (2 * b_top / beta(top, "tree"), 2 * math.ceil(math.log2(nt)) * _hop(fab, tt, tree=True))
        if (tt == "net" and fab.innet_reduce != "off") or \
                (tt == "link" and fab.innet_reduce == "net+link" and topo == "switch"):
            P = pod_nodes(fab, sys.node_cards) if tt == "net" else None
            lvl = (3 if P and nt > P else 2 if nt > leaf_nodes(fab, sys.node_cards) else 1) if tt == "net" else 1
            tops["innet"] = (b_top / beta(top, "innet"), 2 * lvl * _hop(fab, tt))
        tn = min(tops, key=lambda x: sum(tops[x]))
        lo_bw = sum(lo)
        cands["hier"] = (lo_bw + tops[tn][0], a_launch + lo_lat + tops[tn][1], dict(zip(T, lo + [tops[tn][0]])))
        tops_pick = tn
    elif kind == "allgather":
        _, busy = single([(group - 1) * payload / (K[i] * beta(i, "ring")) for i in range(len(lv))])
        bw = (group - 1) * payload * max(1.0 / (K[i] * beta(i, "ring")) for i in range(len(lv)))
        st = _steps_split(lv, lambda U: U - 1)
        cands["ring"] = (bw, a_launch + sum(st[t] * _hop(fab, t) for t in st), busy)
        ph = [(n - 1) * math.prod(m for m, _, _ in lv[i + 1:]) * payload / beta(i, "ring") for i, (n, _, _) in enumerate(lv)]
        cands["hier"] = (sum(ph), a_launch + sum((n - 1) * _hop(fab, t) for n, _, t in lv), dict(zip(T, ph)))
        tops_pick = "ring"
    elif kind == "alltoall":
        vols = [(n - 1) * K[i] / group * payload for i, (n, ln, t) in enumerate(lv)]
        if fab.net_topology == "rail" and tt == "net" and k2 > 1:
            for i, (_, _, t) in enumerate(lv):
                if t == "link":             # PXN: cross-rail network bytes first hop over the scale-up tier
                    vols[i] += vols[top] * (k2 - 1) / k2
        bw, busy = single([v / beta(i, "alltoall") for i, v in enumerate(vols)])
        cands["direct"] = (bw, a_launch + max(_hop(fab, t) for _, _, t in lv), busy)
    else:                                   # p2p-like: the outermost tier
        x = payload / beta(top, "ring")
        cands["p2p"] = (x, a_launch + _hop(fab, tt), {tt: x})
    # byte shares for the energy counts: hierarchical volumes (alltoall: per-tier destinations; ring: link mix)
    if kind == "alltoall":
        vol = [(n - 1) * K[i] / group for i, (n, _, _) in enumerate(lv)]
    elif kind == "allgather":
        vol = [(n - 1) * math.prod(m for m, _, _ in lv[i + 1:]) for i, (n, _, _) in enumerate(lv)]
    else:
        vol = [2 * (n - 1) / n / K[i] for i, (n, _, _) in enumerate(lv)]
    tot = sum(vol) or 1.0
    hshare = {t: v / tot for v, (_, _, t) in zip(vol, lv)}
    shares = {}
    for name in cands:
        if kind in ("allreduce", "allgather") and name == "ring":
            U = [math.prod(n for n, _, _ in lv[i:]) for i in range(len(lv))] + [1]
            hops = [U[i] - U[i + 1] for i in range(top)] + [nt]         # ring hops per tier (sum = g)
            shares[name] = {t: h / group for h, (_, _, t) in zip(hops, lv)}
        else:
            shares[name] = hshare
    meta = {"levels": lv, "top": tops_pick, "s_net": s_net, "nt": nt, "tt": tt}
    return cands, shares, meta


_PATTERN = {"ring": "ring", "tree": "tree", "direct": "alltoall", "p2p": "ring"}


def pick(cands: dict, fab: Fabric) -> str:
    if fab.algo not in ("auto", "auto_overlap") and fab.algo in cands:
        return fab.algo
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
                                  "top": meta["top"] if name == "hier" else "",
                                  "cands": {k: [v[0], v[1]] for k, v in cands.items()},
                                  "busy": {k: dict(v[2]) for k, v in cands.items()},
                                  "net_frac": {k: v.get("net", 0.0) for k, v in shares.items()}, "hier_top": meta["top"]})
        if meta["tt"] == "net":
            e["net_tiers"] = {k: net_shares(fab, "tree" if (k == "hier" and meta["top"] == "tree") else
                                            "innet" if (k == "hier" and meta["top"] == "innet") else
                                            _PATTERN.get(k, "ring"), meta["nt"], meta["s_net"], sys.node_cards)
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
            "p2p": "p2p"}
_KIND_ZH = {"allreduce": "allreduce", "allgather": "allgather", "alltoall": "all-to-all", "p2p": "p2p"}


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
               "levels": e["levels"], "algo": e["algo"], "top": e["hier_top"] if e["algo"] == "hier" and "hier_top" in e
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
