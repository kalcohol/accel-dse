"""0.60 — large card counts (API cap 8192, batch cap 64 × cards, PD search / queue at scale), three-tier fat-tree
(leaf / spine / core, per-tier oversubscription), per-port overlap of tiers, exposed-time algorithm choice
(auto_overlap), multi-hop PP on ring / torus / full-mesh scale-up, PD KV contention fed back into decode TPOT.
All 0.60 options default to the 0.59 behaviour (「假设」)."""

from __future__ import annotations

import dataclasses

from accel_dse.api import ApiError, api_catalog, api_eval, api_layouts, MAX_CARDS
from accel_dse.core import fabric as F
from accel_dse.core import pdqueue as PQ
from accel_dse.core.disagg import disagg_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NET_DEFAULT, Fabric, Link, System
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.core.search import B_CAP, b_cap_for, _BatchSearch

MB = 1 << 20
HBM = "hbm3e_8s_12h24g_9200"


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def _sys(nc=8, cards=256, topo="switch", net=25.0, **kw):
    return System(CHIPS["100T"], link=Link(400.0, 3.0, topo), net=Link(net, 5.0), node_cards=nc,
                  fabric=Fabric(enabled=True, **kw), cards=cards)


def test_fabric_validation_060():
    for bad in ({"net_tiers": 4}, {"oversub_spine": 0.5}, {"oversub_spine": 65}, {"pod_nodes": -1},
                {"leaf_nodes": 4, "pod_nodes": 6}, {"overlap": "max"}, {"algo": "fastest"}, {"kv_feedback": 1}):
        try:
            Fabric(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    f = Fabric()
    assert (f.net_tiers, f.pod_nodes, f.oversub_spine, f.overlap, f.kv_feedback) == (2, 0, 1.0, "sum", False)


def test_pod_nodes_derivation():
    # radix 64: leaf down ports 64·r/(1+r); 8 cards / node → r=1: 32 ports = 4 nodes per leaf;
    # spine down ports 64·r2/(1+r2): r2=1 → 32 leaves per pod = 128 nodes; r2=3 → 48 leaves = 192 nodes
    assert F.leaf_nodes(Fabric(), 8) == 4
    assert F.pod_nodes(Fabric(), 8) is None                                   # two tiers
    assert F.pod_nodes(Fabric(net_tiers=3), 8) == 128
    assert F.pod_nodes(Fabric(net_tiers=3, oversub_spine=3.0), 8) == 192
    assert F.pod_nodes(Fabric(net_tiers=3, leaf_nodes=2, pod_nodes=16), 8) == 16
    assert F.pod_nodes(Fabric(net_tiers=3, net_topology="rail"), 8) is None   # rail keeps two levels


def test_three_tier_net_factor_hand_calc():
    # 32 nodes, 2 per leaf, 8 per pod, r1 = 2, r2 = 4
    fab = Fabric(net_tiers=3, leaf_nodes=2, pod_nodes=8, oversub=2.0, oversub_spine=4.0)
    two = Fabric(leaf_nodes=2, oversub=2.0)
    # all-to-all: f_leaf = 30/31, f_pod = 24/31 → max(1, 2·30/31, 8·24/31) = 192/31
    assert _close(F.net_factor(fab, "alltoall", 32, 1, 8), 8 * 24 / 31)
    assert _close(F.net_factor(two, "alltoall", 32, 1, 8), 2 * 30 / 31)
    # ring: f_leaf = 1/2, f_pod = 1/8 → max(1, 1, 1) = 1
    assert F.net_factor(fab, "ring", 32, 1, 8) == 1.0
    # tree: NCCL btree cut shares at m = 2 and m = 8
    exp = max(1.0, 2 * F.tree_cut_share(32, 2), 8 * F.tree_cut_share(32, 8))
    assert _close(F.net_factor(fab, "tree", 32, 1, 8), exp)
    # r2 = 1 → identical to two tiers for every pattern (f_pod ≤ f_leaf)
    same = dataclasses.replace(fab, oversub_spine=1.0)
    for pat in ("ring", "tree", "alltoall"):
        assert F.net_factor(same, pat, 32, 1, 8) == F.net_factor(two, pat, 32, 1, 8)
    # group inside one pod: no core penalty
    assert _close(F.net_factor(fab, "alltoall", 8, 1, 8), 2 * 6 / 7)
    # r1 = 1 but r2 > 1 still penalises cross-pod traffic: 1·4·24/31
    assert _close(F.net_factor(dataclasses.replace(fab, oversub=1.0), "alltoall", 32, 1, 8), 4 * 24 / 31)
    assert F.net_shares(fab, "alltoall", 32, 1, 8) == (30 / 31, 24 / 31)


def test_three_tier_kv_and_p2p():
    fab = Fabric(enabled=True, net_tiers=3, leaf_nodes=2, pod_nodes=8, oversub=2.0, oversub_spine=4.0)
    # 256 cards / 8 = 32 nodes: max(2·(1 − 2/32), 8·(1 − 8/32)) = 6
    assert _close(F.kv_factor(fab, 8, 256), 6.0)
    assert _close(F.kv_factor(dataclasses.replace(fab, net_tiers=2), 8, 256), 2 * (1 - 2 / 32))
    sys = System(CHIPS["100T"], net=Link(50.0, 5.0), node_cards=8, fabric=fab, cards=256)
    b = 50e9
    assert _close(F.p2p(MB, "net", sys.net, sys, 7, 8), MB / b)                 # same leaf
    assert _close(F.p2p(MB, "net", sys.net, sys, 15, 16), MB / (b / 2))         # other leaf, same pod: r1
    assert _close(F.p2p(MB, "net", sys.net, sys, 63, 64), MB / (b / 8))         # other pod: r1·r2


def test_three_tier_r2_1_equals_two_tier_end_to_end():
    lay = Layout(1, 1, 64, 64, 1)
    base = Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=lay, node_cards=8,
                    serving=Serving(batch=512), fabric=Fabric(enabled=True, oversub=2.0))
    three = dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, net_tiers=3, pod_nodes=4))
    assert evaluate(base).step == evaluate(three).step
    worse = dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, net_tiers=3, pod_nodes=4, oversub_spine=4.0))
    assert evaluate(worse).stages[0].time.t_link > evaluate(base).stages[0].time.t_link


def test_candidates_busy_hand_calc():
    # ring allreduce, g = 16 over 2 nodes of 8: busy_link = 2·15/16·B/(1·400e9), busy_net = 2·15/16·B/(8·25e9)
    sys = _sys(cards=16)
    cands, shares, meta = F.candidates("allreduce", MB, 16, sys)
    bw, a, busy = cands["ring"]
    assert _close(busy["link"], 1.875 * MB / 400e9) and _close(busy["net"], 1.875 * MB / 200e9)
    assert _close(bw, max(busy.values()))
    # hier: phases add up — bw = Σ busy
    bw, a, busy = cands["hier"]
    assert _close(bw, sum(busy.values()))
    assert _close(busy["link"], 2 * 7 / 8 * MB / 400e9)


def test_ports_overlap_bounds():
    for lay, ph in ((Layout(1, 8, 32, 256, 1), "decode"), (Layout(1, 16, 16, 256, 1), "prefill")):
        s = Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=lay, node_cards=8,
                     serving=Serving(phase=ph, batch=256, prompt=2048),
                     fabric=Fabric(enabled=True, oversub=2.0))
        p = dataclasses.replace(s, fabric=dataclasses.replace(s.fabric, overlap="ports"))
        rs, rp = evaluate(s), evaluate(p)
        rep = F.fabric_report(p, rp)
        st = rep["stages"][0]
        assert _close(rp.stages[0].time.t_link * 1e6, st["link_us"])
        assert _close(st["link_us"], max(max(st["busy_us"].values()), max(x["bw_us"] for x in rep["rows"])))
        # ports never exceeds the sum and never undercuts the busiest port
        assert rp.stages[0].time.t_link <= rs.stages[0].time.t_link * (1 + 1e-12)
        assert rp.stages[0].time.t_link >= max(st["busy_us"].values()) * 1e-6 * (1 - 1e-12)
        assert rp.step <= rs.step * (1 + 1e-12)


def test_auto_overlap_minimises_stage_time():
    # TP32 decode across 4 nodes: auto (bw + α per collective) picks tree; with compute hiding the bandwidth, the
    # lower-latency hier wins the stage (exposed α)
    s = Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=Layout(1, 32, 8, 256, 1), node_cards=8,
                 serving=Serving(batch=2048), fabric=Fabric(enabled=True, oversub=2.0, net_tiers=3, pod_nodes=16,
                                                            oversub_spine=4.0))
    o = dataclasses.replace(s, fabric=dataclasses.replace(s.fabric, algo="auto_overlap"))
    ra, ro = evaluate(s), evaluate(o)
    assert ro.step < ra.step
    rep = F.fabric_report(o, ro)
    ar = [x for x in rep["rows"] if x["kind"] == "allreduce"][0]
    assert ar["algo"] == "hier"
    assert F.fabric_report(s, ra)["rows"][0]["algo"] in ("tree", "hier", "direct")
    # never worse than auto / any forced algorithm, both overlap modes
    for ov in ("sum", "ports"):
        for lay in (Layout(1, 16, 16, 256, 1), Layout(2, 8, 16, 128, 1)):
            base = dataclasses.replace(s, layout=lay, fabric=dataclasses.replace(s.fabric, overlap=ov))
            best = evaluate(dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, algo="auto_overlap"))).step
            for algo in ("auto", "ring", "tree", "hier"):
                assert best <= evaluate(dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, algo=algo))).step \
                    * (1 + 1e-12)


def test_p2p_multi_hop_scaleup():
    # ring of 8 packages, stage shift S = 2: flows a → a+2 (a = 0..5) clockwise; a link carries ≤ 2 flows on a β/2
    # port → factor 4.  Full mesh: one direct link of β/7 → 7.  Torus 4×2, S = 4: one Y hop per column → 4 / 1·1 = 4
    assert F._p2p_link_factor("ring", 8, 2) == 4.0
    assert F._p2p_link_factor("full_mesh", 8, 2) == 7.0
    assert F._p2p_link_factor("torus2d", 8, 4) == 4.0
    assert F._p2p_link_factor("switch", 8, 2) == 1.0
    # end to end: PP4 · TP2 on one 8-card ring node pays the hop load; switch is unchanged
    for topo, fac in (("switch", 1.0), ("ring", 4.0)):
        sys = System(CHIPS["100T"], link=Link(400.0, 3.0, topo), node_cards=8, fabric=Fabric(enabled=True), cards=8)
        assert _close(F.p2p(MB, "link", sys.link, sys, 1, 2, 2), MB / (400e9 / fac))


def test_kv_feedback_into_decode_tpot():
    lay = Layout(1, 1, 64, 64, 1)
    s = Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=lay, node_cards=8, net=Link(12.5, 5.0),
                 serving=Serving(batch=512, prompt=2048, out_len=16, ctx=2048, ttft_slo_ms=20000),
                 pd=PDConfig(enabled=True, simulate=False, prefill_layout=lay, prefill_cards=960, decode_cards=64),
                 fabric=Fabric(enabled=True, oversub=4.0, leaf_nodes=1))
    off = disagg_report(s)
    assert "feedback" not in off["kv"]["fabric"]
    on = disagg_report(dataclasses.replace(s, fabric=dataclasses.replace(s.fabric, kv_feedback=True)))
    fb = on["kv"]["fabric"]["feedback"]
    assert 0 < fb["u_kv"] <= 1 - on["kv"]["fabric"]["u_coll"] + 1e-9
    assert _close(fb["tpot_ms_before"], off["tpot_ms"])
    assert fb["tpot_ms_after"] > fb["tpot_ms_before"] and _close(on["tpot_ms"], fb["tpot_ms_after"])
    assert _close(fb["net_GBps_decode"], 12.5 * (1 - fb["u_kv"]))


def test_large_card_counts_api():
    assert MAX_CARDS == 8192 and api_catalog()["max_cards"] == 8192
    sc = {"model": "deepseek-v3", "mem_id": HBM, "layout": {"pp": 4, "tp": 1, "dp": 256, "ep": 256, "etp": 1},
          "node_cards": 8, "serving": {"batch": 4096}, "fabric": {"enabled": True, "net_tiers": 3, "pod_nodes": 16,
                                                                   "oversub": 2, "oversub_spine": 2}}
    r = api_eval({"chip_preset": "1P", "scenario": sc})
    assert r["summary"]["fits"] and r["fabric"]["pod_nodes"] == 16 and r["fabric"]["net_traffic"]["pod_up_share"] > 0
    try:
        api_eval({"chip_preset": "1P", "scenario": {**sc, "layout": {"pp": 1, "tp": 1, "dp": 8200, "ep": 8200, "etp": 1}}})
        raise AssertionError("accepted 8200 cards")
    except ApiError:
        pass
    try:
        api_layouts({"chip_preset": "1P", "scenario": sc, "cards": 8193})
        raise AssertionError("accepted cards 8193")
    except ApiError:
        pass
    # batch cap scales with cards: 64 × cards (4096 = 64 × 64 at the old limit)
    assert b_cap_for(64) == B_CAP == 4096 and b_cap_for(1024) == 65536
    bs = _BatchSearch(Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=Layout(1, 1, 1024, 1024, 1)))
    assert bs.b_cap == 65536


def test_pd_split_sampling_matches_exact():
    lay = Layout(1, 1, 1, 1, 1)
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HBM, layout=lay,
                 serving=Serving(batch=32, prompt=1024, out_len=128),
                 pd=PDConfig(enabled=True, simulate=False, prefill_layout=lay, prefill_cards=12, decode_cards=12))
    old = PQ.SPLIT_EXACT
    try:
        PQ.SPLIT_EXACT = 1000
        exact = disagg_report(s)["queue"]
        PQ.SPLIT_EXACT = 4
        samp = disagg_report(s)["queue"]
    finally:
        PQ.SPLIT_EXACT = old
    assert "pd_slo_splits_sampled" not in exact and samp["pd_slo_splits_sampled"]["candidates"] == 23
    assert samp["pd_slo_best_split"] == exact["pd_slo_best_split"]
    assert len(samp["pd_slo_splits"]) < len(exact["pd_slo_splits"])


def test_decode_chain_interpolation_large_batch():
    # B > 1024: step(k) interpolated on a geometric grid — mean TPOT within 0.5 % of the exact chain
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HBM, layout=Layout(1, 1, 1, 1, 1),
                 serving=Serving(batch=1536, ctx=1024))
    pool = PQ._Pool(s)
    st = evaluate(s).step
    lam = 0.7 * 1536 / (128 * st)
    a = PQ._decode_birth_death(pool, lam, 128, 1536)
    old = PQ.INTERP_B
    try:
        PQ.INTERP_B = 10 ** 6
        b = PQ._decode_birth_death(PQ._Pool(s), lam, 128, 1536)
    finally:
        PQ.INTERP_B = old
    assert a is not None and b is not None
    assert abs(a["tpot"] / b["tpot"] - 1) < 5e-3
