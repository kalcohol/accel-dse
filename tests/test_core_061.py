"""0.61 — performance (PD queue chains: C-level dot products, running max, past-mode truncation, step list once;
evaluate: per-op / per-collective memo; layout search: brackets on a process pool, perturbation cases on the pool),
per-step latency of the switch tiers crossed (hop_spine / hop_core), NVLS-class all-gather / reduce-scatter,
NCCL's second tree in the leaf cut share, LL / LL128 / Simple protocols.  All 0.61 model options default to the 0.60
behaviour (「假设」)."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

from accel_dse.core import fabric as F
from accel_dse.core import pdqueue as PQ
from accel_dse.core import search as SR
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Fabric, Link, System
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

US = 1e-6


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def _sys(nc=8, cards=256, topo="switch", net=25.0, **kw):
    return System(CHIPS["100T"], link=Link(400.0, 3.0, topo), net=Link(net, 5.0), node_cards=nc,
                  fabric=Fabric(enabled=True, **kw), cards=cards)


# ---------------------------------------------------------------- options / defaults

def test_fabric_validation_061():
    for bad in ({"hop_spine_us": -1}, {"hop_core_us": 2e4}, {"protocol": "ll"}, {"protocol": "fast"},
                {"hop_spine_us": True}):
        try:
            Fabric(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    f = Fabric()
    assert (f.hop_spine_us, f.hop_core_us, f.protocol) == (0.0, 0.0, "off")
    # defaults: plain algorithm names, no protocol suffix, no extra latency
    c, _, _ = F.candidates("allreduce", 4 << 20, 256, _sys())
    assert set(c) == {"ring", "tree", "hier"}
    assert F.switch_extra(Fabric(), 32, 1, 8) == (0.0, 0.0, 0)


# ---------------------------------------------------------------- item 2: switch tiers crossed

def test_switch_extra_hand_calc():
    # 32 group nodes (stride 1), 2 per leaf, 8 per pod, hop_spine 1 µs, hop_core 2 µs
    fab = Fabric(net_tiers=3, leaf_nodes=2, pod_nodes=8, hop_spine_us=1.0, hop_core_us=2.0)
    ring, tree, lvl = F.switch_extra(fab, 32, 1, 8)
    assert _close(ring, 3 * US)                                   # every ring step leaves leaf and pod
    # tree: ⌈log₂32⌉ = 5 levels; leave the leaf when 2^j ≥ 2 → 5 − 1 = 4 levels; the pod when 2^j ≥ 8 → 5 − 3 = 2
    assert _close(tree, (4 * 1.0 + 2 * 2.0) * US)
    assert lvl == 3
    # inside one leaf: nothing; inside one pod: spine only
    assert F.switch_extra(fab, 2, 1, 8) == (0.0, 0.0, 1)
    r2, t2, l2 = F.switch_extra(fab, 8, 1, 8)
    assert _close(r2, 1 * US) and _close(t2, 2 * US) and l2 == 2   # 3 levels − ⌈log₂2⌉ = 2 levels pay hop_spine


def test_collective_latency_with_switch_tiers():
    # 256 cards = 32 nodes × 8; ring allreduce: 2·(32 − 1) = 62 network steps → + 62 × 3 µs;
    # tree: 2 × (4·1 + 2·2) µs = + 16 µs; all-to-all: one hop → + 3 µs; bandwidth unchanged
    kw = dict(net_tiers=3, leaf_nodes=2, pod_nodes=8)
    base, _, _ = F.candidates("allreduce", 8 << 20, 256, _sys(**kw))
    ext, _, _ = F.candidates("allreduce", 8 << 20, 256, _sys(hop_spine_us=1.0, hop_core_us=2.0, **kw))
    assert _close(ext["ring"][1] - base["ring"][1], 62 * 3 * US)
    assert _close(ext["tree"][1] - base["tree"][1], 16 * US)
    for k in base:
        assert ext[k][0] == base[k][0]
    a0, _, _ = F.candidates("alltoall", 1 << 20, 256, _sys(**kw))
    a1, _, _ = F.candidates("alltoall", 1 << 20, 256, _sys(hop_spine_us=1.0, hop_core_us=2.0, **kw))
    assert _close(a1["direct"][1] - a0["direct"][1], 3 * US)
    # 2 nodes under one leaf: no change
    s0, _, _ = F.candidates("allreduce", 8 << 20, 16, _sys(**kw))
    s1, _, _ = F.candidates("allreduce", 8 << 20, 16, _sys(hop_spine_us=1.0, hop_core_us=2.0, **kw))
    assert s0 == s1


def test_p2p_and_kv_switch_latency():
    sysx = _sys(net_tiers=3, leaf_nodes=2, pod_nodes=4, hop_spine_us=1.5, hop_core_us=4.0)
    ln = sysx.net
    base = ln.alpha_us * US + F._hop(sysx.fabric, "net")
    # cards 15 → 16: nodes 1 → 2, leaves 0 → 1, same pod
    bw, a = F.p2p_alpha(1e-3, "net", ln, sysx, 15, 16)
    assert bw == 1e-3 and _close(a, base + 1.5 * US)
    # cards 31 → 32: nodes 3 → 4: other pod
    _, a = F.p2p_alpha(1e-3, "net", ln, sysx, 31, 32)
    assert _close(a, base + 5.5 * US)
    _, a = F.p2p_alpha(1e-3, "net", ln, sysx, 7, 8)                 # nodes 0 → 1, same leaf
    assert _close(a, base)
    # KV: 64 nodes, 4 per leaf, 16 per pod → (1 − 4/64)·1 + (1 − 16/64)·2 µs
    fab = Fabric(net_tiers=3, leaf_nodes=4, pod_nodes=16, hop_spine_us=1.0, hop_core_us=2.0)
    assert _close(F.kv_hop_extra(fab, 8, 512), (0.9375 + 0.75 * 2) * US)
    assert F.kv_hop_extra(Fabric(), 8, 512) == 0.0


# ---------------------------------------------------------------- item 3: NVLS-class AG / RS, second tree

def test_nvls_allgather_reducescatter_hand_calc():
    # one node of 8 on a switched scale-up tier (β = 400 GB/s, hop 0.6 µs, α 3 µs); shard p = 1 MiB
    p = 1 << 20
    s_off = _sys(nc=8, cards=8)
    s_on = _sys(nc=8, cards=8, innet_reduce="net+link")
    c0, _, _ = F.candidates("allgather", p, 8, s_off)
    c1, _, _ = F.candidates("allgather", p, 8, s_on)
    assert "innet" not in c0 and "innet" in c1
    b = 400e9
    assert _close(c1["innet"][0], 7 * p / b)                   # multicast: still 7 shards received per rank
    assert _close(c1["innet"][1], 3 * US + 2 * 0.6 * US)       # one switch round trip
    assert _close(c1["ring"][1], 3 * US + 7 * 0.6 * US)        # vs 7 ring steps
    r1, _, _ = F.candidates("reducescatter", p, 8, s_on)
    assert _close(r1["innet"][0], 8 * p / b)                   # in-switch reduction: 8 shards sent once
    assert _close(r1["ring"][0], 7 * p / b)
    assert F.pick(c1, s_on.fabric) == "innet"                  # latency-bound small message
    # SHARP-class "net" stays allreduce-only; ring scale-up has no switch to reduce in
    cn, _, _ = F.candidates("allgather", p, 64, _sys(innet_reduce="net"))
    assert "innet" not in cn
    cr, _, _ = F.candidates("allgather", p, 8, _sys(nc=8, cards=8, topo="ring", innet_reduce="net+link"))
    assert "innet" not in cr
    # evaluated end to end: logits all-gather of a TP 8 decode picks innet
    from accel_dse.core.fabric import fabric_report
    s = Scenario(model="qwen3-8b", layout=Layout(1, 8, 1, 1, 1), node_cards=8, serving=Serving(batch=8),
                 fabric=Fabric(enabled=True, innet_reduce="net+link"))
    rep = fabric_report(s, evaluate(s))
    assert any(x["kind"] == "allgather" and x["algo"] == "innet" for x in rep["rows"])


def test_double_tree_cut_share_hand_calc():
    # n = 5 nodes, 2 per leaf (blocks {0,1} {2,3} {4}).  Tree 1 (ncclGetBtree): 1→2, 2→4, 3→2, 4→0: cuts 1→2,
    # 2→4, 4→0 = 3.  Tree 2 (n odd: shift by one, root 1): 0→1, 2→3, 3→0, 4→3: cuts 3→0, 4→3 = 2.
    # Each tree carries half the bytes: (3 + 2)/2 / 4 edges = 0.625 (0.60: 3/4 = 0.75)
    assert [F._dtree_parent2(r, 5) for r in range(5)] == [1, -1, 3, 0, 3]
    assert _close(F.tree_cut_share(5, 2), 0.625)
    # n = 8 (even: mirror, root 7), 4 per leaf: both trees cut 2 of 7 edges → unchanged 2/7
    assert [F._dtree_parent2(r, 8) for r in range(8)] == [1, 3, 1, 7, 5, 3, 5, -1]
    assert _close(F.tree_cut_share(8, 4), 2 / 7)
    assert F.tree_cut_share(4, 4) == 0.0


# ---------------------------------------------------------------- item 4: protocols

def test_protocol_hand_calc():
    big, small = 64 << 20, 4096
    off, _, _ = F.candidates("allreduce", big, 256, _sys())
    au, _, _ = F.candidates("allreduce", big, 256, _sys(protocol="auto"))
    assert set(au) == {f"{a}/{p}" for a in ("ring", "tree", "hier") for p in ("LL", "LL128", "Simple")}
    # LL: half bandwidth, the hop_* latency; LL128: 128/120 bandwidth; Simple: full bandwidth
    assert _close(au["ring/LL"][0], 2 * off["ring"][0]) and _close(au["ring/LL"][1], off["ring"][1])
    assert _close(au["ring/LL128"][0], off["ring"][0] * 128 / 120)
    assert _close(au["ring/Simple"][0], off["ring"][0])
    # Simple ring latency: launch 5 µs + 448 scale-up steps × 0.6·(3.4/0.6) + 62 network steps × 2.7·(14/2.7)
    assert _close(au["ring/Simple"][1], 5 * US + 448 * 3.4 * US + 62 * 14.0 * US)
    s_auto = _sys(protocol="auto")
    cb, _, _ = F.candidates("allreduce", big, 256, s_auto)
    cs, _, _ = F.candidates("allreduce", small, 256, s_auto)
    assert F.pick(cb, s_auto.fabric).endswith("/Simple") or F.pick(cb, s_auto.fabric).endswith("/LL128")
    assert F.pick(cs, s_auto.fabric).endswith("/LL")
    # a forced algorithm keeps its protocol choice; a forced protocol keeps the algorithm choice
    s_ring = _sys(protocol="auto", algo="ring")
    assert F.pick(cs, s_ring.fabric) == "ring/LL"
    one, _, _ = F.candidates("allreduce", small, 256, _sys(protocol="LL128"))
    assert set(one) == {"ring/LL128", "tree/LL128", "hier/LL128"}
    # p2p: the fastest protocol for the message
    s = _sys(protocol="auto")
    bw, a = F.p2p_alpha(1e-3, "net", s.net, s, 0, 8)          # 1 ms of bytes: Simple (full bandwidth)
    assert bw == 1e-3 and _close(a, 5 * US + 14.0 * US)
    bw, a = F.p2p_alpha(1e-8, "net", s.net, s, 0, 8)          # tiny: LL
    assert _close(bw, 2e-8) and _close(a, 5 * US + 2.7 * US)


def test_protocol_end_to_end_slower_or_equal():
    # protocol = off is the optimistic 0.60 model (LL latency at full bandwidth): auto can only add time
    for lay in (Layout(1, 8, 4, 32, 1), Layout(2, 16, 1, 1, 1)):
        mid = "deepseek-v3" if lay.ep > 1 else "llama-3.3-70b"
        s0 = Scenario(model=mid, layout=lay, node_cards=8, serving=Serving(batch=64), fabric=Fabric(enabled=True))
        s1 = s0.replace("fabric", Fabric(enabled=True, protocol="auto"))
        assert evaluate(s1).step >= evaluate(s0).step * (1 - 1e-12)


# ---------------------------------------------------------------- item 1: performance (exactness)

def test_occ_tau_fast_matches_loop():
    random.seed(7)
    for N, L, r in ((300, 40, 0.98), (2000, 300, 0.997)):
        X = [random.random() * 0.97 ** i for i in range(L)]
        z = sum(X)
        X = [x / z for x in X]
        pi = [r ** n * (1 + 0.3 * random.random()) for n in range(N)]
        z = sum(pi)
        pi = [p / z for p in pi]
        mu = lambda k: 0.5 + k * 0.01
        old = PQ.FAST_NL
        try:
            PQ.FAST_NL = 10 ** 12
            a = PQ._occ_tau(pi, 2 * N, (0.01, X), mu)          # B ≥ FAST_B: fast path when N·L > FAST_NL
            PQ.FAST_NL = 0
            b = PQ._occ_tau(pi, 2 * N, (0.01, X), mu)
        finally:
            PQ.FAST_NL = old
        assert _close(a, b, 1e-12)
        # Poisson arrivals (L = 1): closed form d = G/(π·μ), the coupling sums are empty
        c = PQ._occ_tau(pi, N // 3, (0.01, [1.0]), mu)
        assert c is None or c > 0


def test_pd_queue_truncation_bounded():
    # B ≥ FAST_B chains stop past the mode below 1e-20 of the peak: same report to ~1e-15 vs the full chain
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(1, 1, 1, 1, 1),
                 serving=Serving(batch=64, prompt=2048, out_len=256), pd=PDConfig(enabled=True, simulate=False))
    from accel_dse.core.disagg import disagg_report
    old = PQ.FAST_B
    try:
        PQ.FAST_B = 32                                  # truncated chains (and dot products) at B = 64
        a = disagg_report(s)["queue"]["modes"]
        PQ.FAST_B = 10 ** 9                             # full chains, 0.60 loops
        b = disagg_report(s)["queue"]["modes"]
    finally:
        PQ.FAST_B = old
    assert any(isinstance(v.get("slo_rate_rps"), float) and v["slo_rate_rps"] > 0 for v in a.values())
    for k in a:
        for kk in ("slo_rate_rps", "tpot_p90_ms", "stable_rate_rps"):
            x, y = a[k].get(kk), b[k].get(kk)
            if isinstance(x, float) and isinstance(y, float):
                assert _close(x, y, 1e-9), (k, kk, x, y)
            else:
                assert x == y


def test_search_pool_same_ranking():
    # brackets on a pool (threads here; the server uses spawn processes) → the same exact top-k
    base = Scenario(model="qwen3-30b-a3b", layout=Layout(1, 1, 1, 1, 1), node_cards=8,
                    fabric=Fabric(enabled=True, oversub=2.0))
    old = SR.PAR_MIN
    try:
        SR.PAR_MIN = 1
        st0, st1 = {}, {}
        r0 = SR.search_layouts(base, 32, top=4, stats=st0)
        with ThreadPoolExecutor(4) as ex:
            r1 = SR.search_layouts(base, 32, top=4, stats=st1, pool=ex)
    finally:
        SR.PAR_MIN = old
    assert [(r.layout, r.batch, r.result.per_card) for r in r0] == [(r.layout, r.batch, r.result.per_card) for r in r1]
    assert all(not isinstance(r.result, SR._Lite) for r in r1)
    assert st0["solved"] == st1["solved"] and st0["evals"] == st1["evals"]


def test_stability_pool_same_cases():
    from accel_dse.core.stability import ranking_stability
    base = Scenario(model="qwen3-8b", layout=Layout(1, 1, 1, 1, 1))
    a = ranking_stability(base, 4, include_mapping=False)
    with ThreadPoolExecutor(3) as ex:
        b = ranking_stability(base, 4, include_mapping=False, pool=ex)
    assert (a.base_top, a.stable, a.agree, a.cases) == (b.base_top, b.stable, b.agree, b.cases)


def test_op_memo_pure():
    # memoised per-op / per-collective costs: evaluation order does not matter
    s1 = Scenario(model="deepseek-v3", layout=Layout(1, 8, 4, 32, 1), node_cards=8, serving=Serving(batch=96),
                  fabric=Fabric(enabled=True, oversub=2.0))
    s2 = s1.replace("serving.batch", 32)
    a1, a2 = evaluate(s1).step, evaluate(s2).step
    from accel_dse.core import evaluate as E
    from accel_dse.core.memo import model_cache
    E._COLL_MEMO.clear()
    model_cache(get_model("deepseek-v3")).clear()
    b2, b1 = evaluate(s2).step, evaluate(s1).step
    assert (a1, a2) == (b1, b2)
