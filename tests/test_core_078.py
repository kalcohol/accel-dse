"""0.78: link energy on physical wire bytes per algorithm and tier (d2d / scale-up / network each its own pJ/bit);
timing fields (link_bytes, d2d_bytes, net_bytes = payload shares) unchanged."""
from accel_dse.core import fabric
from accel_dse.core.energy import action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Fabric, System
from accel_dse.core.scenario import Layout, Scenario
from accel_dse.core.schedule import collective_wire, wire_single

P = 1e6


def _r(d):
    return {k: round(v / P, 6) for k, v in d.items()}


def test_single_tier_formulas():
    assert wire_single("allreduce", P, 8) == 2 * 7 / 8 * P
    assert wire_single("allgather", P, 8) == 7 * P            # payload = one rank's shard
    assert wire_single("alltoall", P, 8) == 7 / 8 * P
    assert wire_single("p2p", P, 2) == P and wire_single("allreduce", P, 1) == 0.0


def test_tiered_hierarchical_volumes():
    # 16 ranks = 2 dies × 2 packages × 4 nodes
    assert _r(collective_wire("allreduce", P, 16, True, 2, 4)) == {"d2d": 1.0, "link": 0.5, "net": 0.375}
    assert _r(collective_wire("allgather", P, 16, True, 2, 4)) == {"d2d": 8.0, "link": 4.0, "net": 3.0}
    assert _r(collective_wire("alltoall", P, 16, True, 2, 4)) == {"d2d": 0.0625, "link": 0.125, "net": 0.75}


def test_fabric_algorithms_wire():
    def ar(algo, innet="off"):
        s = System(CHIPS["100T"], package_cards=2, node_cards=4, cards=16,
                   fabric=Fabric(enabled=True, algo=algo, innet_reduce=innet))
        return _r(fabric.collective_full("allreduce", P, 16, s)[6])
    ring = {"d2d": 0.9375, "link": 0.46875, "net": 0.46875}            # 2·15/16 split by ring hops 8 / 4 / 4
    assert ar("ring") == ring and ar("tree") == ring                  # reduce + broadcast over g − 1 edges
    assert ar("hier") == {"d2d": 1.0, "link": 0.5, "net": 0.375}
    assert ar("hier", "net") == {"d2d": 1.0, "link": 0.5, "net": 0.25}  # in-network: each node sends its shard once
    s = System(CHIPS["100T"], package_cards=2, node_cards=4, cards=16, fabric=Fabric(enabled=True))
    assert abs(sum(fabric.collective_full("alltoall", P, 16, s)[6].values()) / P - 15 / 16) < 1e-12
    assert _r(fabric.collective_full("p2p", P, 16, s)[6]) == {"net": 1.0}


def test_energy_counts_use_wire_and_timing_unchanged():
    for tp, k in ((2, 1.0), (4, 1.5), (8, 1.75)):
        r = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=tp)))
        t = r.stages[0].time
        assert abs(t.wire_link / t.link_bytes - k) < 1e-3                       # head's 8-byte gather aside
        c = action_counts(r)["counts"]
        assert abs(c["link"] - t.wire_link * tp) <= 1e-9 * c["link"]          # per token, every rank
        assert t.wire_d2d == t.wire_net == 0.0
    h = evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=4), d2d_enabled=True, package_cards=2)).stages[0].time
    assert abs(h.wire_d2d / h.link_bytes - 1.0) < 1e-3 and abs(h.wire_link / h.link_bytes - 0.5) < 1e-3
    assert abs(h.d2d_bytes / h.link_bytes - 2 / 3) < 1e-9                      # timing share unchanged (payload)


def test_des_overload_detected_when_run_shorter_than_request_lifetime():
    # 0.78: 800 requests at 1.1–1.8× colocated capacity arrive in < 2 request lifetimes (512 tokens × ~50 ms): drift and
    # utilisation tests missed the backlog; the decode-capacity bound (prefill share included) must flag it
    from dataclasses import replace
    from accel_dse.core import pdsim
    from accel_dse.core.validation import v4_scenario
    for rate, want in ((10.8, False), (18.0, False), (9.5, True)):
        s = v4_scenario(0.5)
        s = replace(s, pd=replace(s.pd, rate_rps=rate))
        rep, ctx = pdsim.capture_ctx(s)
        for mode in ("pd", "coloc_prefill_first", "coloc_chunked"):
            if want and mode == "pd":
                continue                       # 9.5 > PD fluid capacity 9.02: PD is over, colocated (9.76) is not
            r = pdsim.simulate(ctx, rate, mode, n_req=800, warmup=200, seed=3)
            assert r["stable"] == want == rep["queue"]["modes"][mode]["stable"], (rate, mode, r["stable"])
            assert r["over_capacity"] == (not want)


from tests.pin_legacy import pin_module  # noqa: E402  (0.173)
pin_module(globals())


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
