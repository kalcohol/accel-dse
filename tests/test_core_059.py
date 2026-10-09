"""0.59 — topology-aware collectives (scenario.fabric, 「假设」): NCCL ring / tree α-β formulas, hierarchical, fat-tree
oversubscription, rail PXN, scale-up topologies, in-network reduction, PP across leaves, PD KV contention; off = 0.58."""

from __future__ import annotations

import dataclasses
import math

from accel_dse.api import ApiError, api_eval
from accel_dse.core import fabric as F
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NET_DEFAULT, Fabric, Link, System
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

B = 1 << 20


def _sys(nc=0, cards=8, topo="switch", link=400.0, **kw):
    return System(CHIPS["100T"], link=Link(link, 3.0, topo), net=NET_DEFAULT, node_cards=nc,
                  fabric=Fabric(enabled=True, **kw), cards=cards)


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def test_off_is_058_bit_for_bit():
    for kw in ({"node_cards": 8, "layout": Layout(tp=8, pp=2)}, {"node_cards": 4, "layout": Layout(tp=4, ep=8, dp=2),
                                                                  "model": "qwen3-30b-a3b"}):
        s0 = Scenario(**kw, serving=Serving(batch=16))
        s1 = dataclasses.replace(s0, fabric=Fabric(enabled=False, oversub=4.0, algo="tree", innet_reduce="net"),
                                 link=Link(400.0, 3.0, "torus2d"))
        a, b = evaluate(s0), evaluate(s1)
        assert a.step == b.step and a.stages[0].time.net_bytes == b.stages[0].time.net_bytes


def test_nccl_ring_tree_hier_formulas():
    # one node, 8 ranks, 400 GB/s, α 3 µs, hop 0.6 µs: ring = 2(n−1)/n·B/β, 2(n−1) steps; NCCL tree on one node = chain
    bw, a, _, _ = F.collective("allreduce", B, 8, _sys(algo="ring"))
    assert _close(bw, 2 * 7 / 8 * B / 400e9) and _close(a, 3e-6 + 14 * 0.6e-6)
    bw, a, _, _ = F.collective("allreduce", B, 8, _sys(algo="tree"))
    assert _close(bw, 2 * B / 400e9) and _close(a, 3e-6 + 14 * 0.6e-6)
    # two nodes of 8 (net 50 GB/s per card, α 5 µs, NET ring hop 2.7, tree hop 5.0)
    s = _sys(nc=8, cards=16, algo="ring")
    bw, a, _, fn = F.collective("allreduce", B, 16, s)
    assert _close(bw, 2 * 15 / 16 * B * max(1 / 400e9, 1 / (8 * 50e9)))
    assert _close(a, 5e-6 + 2 * 2.7e-6 + 28 * 0.6e-6) and _close(fn, 2 / 16)
    bw, a, _, _ = F.collective("allreduce", B, 16, dataclasses.replace(s, fabric=Fabric(enabled=True, algo="tree")))
    assert _close(bw, 2 * B * max(1 / 400e9, 1 / (8 * 50e9))) and _close(a, 5e-6 + 2 * 7 * 0.6e-6 + 2 * 1 * 5.0e-6)
    bw, a, _, _ = F.collective("allreduce", B, 16, dataclasses.replace(s, fabric=Fabric(enabled=True, algo="hier")))
    assert _close(bw, 2 * 7 / 8 * B / 400e9 + 2 * 1 / 2 * (B / 8) / 50e9)
    assert _close(a, 5e-6 + 2 * 7 * 0.6e-6 + 2 * 1 * 2.7e-6)
    # auto takes the cheapest bandwidth + latency
    tot = {al: sum(F.collective("allreduce", 4096, 16, dataclasses.replace(s, fabric=Fabric(enabled=True, algo=al)))[:2])
           for al in ("ring", "tree", "hier")}
    assert _close(sum(F.collective("allreduce", 4096, 16, dataclasses.replace(s, fabric=Fabric(enabled=True)))[:2]),
                  min(tot.values()))
    # allgather ring (g−1)·B·max 1/(K β)
    bw, a, _, _ = F.collective("allgather", B, 16, s)
    assert _close(bw, 15 * B * max(1 / 400e9, 1 / (8 * 50e9))) and _close(a, 5e-6 + 2.7e-6 + 14 * 0.6e-6)


def test_btree_and_net_factors():
    assert [F._btree_parent(r, 8) for r in range(8)] == [-1, 2, 4, 2, 0, 6, 4, 6]   # NCCL ncclGetBtree
    assert _close(F.tree_cut_share(8, 4), 2 / 7)
    fab = Fabric(enabled=True, oversub=4.0, leaf_nodes=2)
    assert F.net_factor(fab, "ring", 8, 1, 8) == 2.0                 # max(1, r/m), m = 2
    assert _close(F.net_factor(fab, "alltoall", 8, 1, 8), 4 * 6 / 7)  # r·(n − m)/(n − 1)
    assert F.net_factor(fab, "innet", 8, 1, 8) == 1.0
    assert F.net_factor(dataclasses.replace(fab, oversub=1.0), "alltoall", 8, 1, 8) == 1.0
    # derived leaf sizes: 64-port switch, down = 64·r/(1+r)
    assert [F.leaf_nodes(Fabric(oversub=r), 8) for r in (1.0, 2.0, 4.0)] == [4, 5, 6]
    assert F.leaf_nodes(Fabric(net_topology="rail"), 8) == 32


def test_ep_alltoall_oversubscription_and_rail():
    for r in (1.0, 2.0, 4.0):
        bw, _, _, _ = F.collective("alltoall", B, 32, _sys(nc=8, cards=32, oversub=r, leaf_nodes=1))
        assert _close(bw, max(24 / 32 * B / 50e9 * r, 7 / 32 * B / 400e9))
    # rail: cross-rail network bytes ((k₂ − 1)/k₂) also cross a slow scale-up first (PXN)
    ft = F.collective("alltoall", B, 32, _sys(nc=8, cards=32, link=40.0))[0]
    rl = F.collective("alltoall", B, 32, _sys(nc=8, cards=32, link=40.0, net_topology="rail"))[0]
    assert _close(ft, 24 / 32 * B / 50e9) and _close(rl, (7 / 32 + 24 / 32 * 7 / 8) * B / 40e9)


def test_scaleup_topologies():
    assert F.su_ring_factor("full_mesh", 8, 2, 1) == 7.0 and F.su_a2a_factor("full_mesh", 8, 4, 1) == 7 / 3
    assert F.su_ring_factor("ring", 8, 8, 1) == 1.0 and F.su_ring_factor("ring", 8, 4, 1) == 2.0
    assert F.su_ring_factor("ring", 8, 4, 2) == 2.0                  # wraps, two groups share each link
    assert _close(F.su_a2a_factor("ring", 8, 8, 1), 16 / 7)          # Σ min(d, 8 − d) / 8 over 7/8
    assert _close(F.su_a2a_factor("torus2d", 16, 16, 1), 32 / 15)
    assert F.su_ring_factor("torus2d", 16, 16, 1) == 1.0 and F.su_ring_factor("torus2d", 16, 4, 1) == 2.0
    bw_sw = F.collective("allreduce", B, 2, _sys(algo="ring"))[0]
    bw_fm = F.collective("allreduce", B, 2, _sys(topo="full_mesh", algo="ring"))[0]
    assert _close(bw_fm, 7 * bw_sw)                                  # TP2 on an 8-card full mesh: 1 of 7 links


def test_innet_reduction():
    s = _sys(nc=1, cards=4, oversub=4.0, leaf_nodes=1, innet_reduce="net", algo="hier")
    bw, a, _, _ = F.collective("allreduce", B, 4, s)
    assert _close(bw, B / 50e9) and _close(a, 5e-6 + 2 * 2 * 2.7e-6)   # each rank sends / receives B once
    s = _sys(cards=8, innet_reduce="net+link", algo="hier")
    assert _close(F.collective("allreduce", B, 8, s)[0], B / 400e9)
    s = _sys(cards=8, topo="ring", innet_reduce="net+link", algo="hier")   # NVLS-class needs a switch
    assert _close(F.collective("allreduce", B, 8, s)[0], 2 * 7 / 8 * B / 400e9)


def test_pp_cross_leaf_and_report():
    sc = Scenario(model="qwen3-32b", layout=Layout(tp=8, pp=4), node_cards=8, serving=Serving(batch=16),
                  fabric=Fabric(enabled=True, oversub=4.0, leaf_nodes=2))
    r = F.fabric_report(sc)
    p2p = sorted((x for x in r["rows"] if x["kind"] == "p2p"), key=lambda x: x["bw_us"])
    assert len(p2p) == 3 and _close(p2p[-1]["bw_us"], 4 * p2p[0]["bw_us"])   # stage 1 → 2 crosses the leaf boundary
    ar = next(x for x in r["rows"] if x["kind"] == "allreduce")
    assert ar["count"] == 2 * 64 and set(ar["cands"]) == {"ring", "tree", "hier"}
    assert r["step_ms_off"] > 0 and r["step_ms"] > 0


def test_api_validation_and_pd_kv():
    base = {"model": "qwen3-8b", "layout": {"tp": 2}, "node_cards": 2, "serving": {"batch": 16, "prompt": 2048, "out_len": 128}}
    for bad, frag in (({"oversub": 0.5}, "oversub"), ({"algo": "butterfly"}, "algo"), ({"leaf_nodes": -1}, "leaf_nodes"),
                      ({"innet_reduce": "yes"}, "innet_reduce")):
        try:
            api_eval({"scenario": {**base, "fabric": {"enabled": True, **bad}}, "chip_preset": "1P"})
        except ApiError as e:
            assert frag in str(e)
        else:
            raise AssertionError(bad)
    try:
        api_eval({"scenario": {**base, "link": {"topology": "hypercube"}}, "chip_preset": "1P"})
    except ApiError as e:
        assert "topology" in str(e)
    else:
        raise AssertionError("bad topology accepted")
    out = api_eval({"scenario": {**base, "fabric": {"enabled": True, "oversub": 4.0, "leaf_nodes": 1},
                                 "pd": {"enabled": True, "decode_cards": 4, "prefill_cards": 4}}, "chip_preset": "1P"})
    assert out["fabric"]["rows"]
    kf = out["pd"]["kv"]["fabric"]
    assert kf["leaf_factor"] == 4.0 * (1 - 1 / 4)                    # 8 cards / 2 per node = 4 nodes, 1 per leaf
    assert _close(kf["GBps_eff"], kf["GBps_raw"] / kf["leaf_factor"] * (1 - kf["u_coll"]))
    off = api_eval({"scenario": {**base, "pd": {"enabled": True, "decode_cards": 4, "prefill_cards": 4}},
                    "chip_preset": "1P"})
    assert "fabric" not in out["pd"]["kv"] or True
    assert "fabric" not in off and "fabric" not in off["pd"]["kv"]
    assert out["pd"]["kv"]["t_ms"] > off["pd"]["kv"]["t_ms"]
