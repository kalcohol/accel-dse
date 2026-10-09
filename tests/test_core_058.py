"""0.58 — colocated KV slot hold = prefill service, colocated wait shape (Gamma, c² = 1 + 0.7·c_s²), chunked
refill-tight share, radix / partial prefix tree (Che per node, token capacity shared across levels; DES RadixLRU)."""

from __future__ import annotations

import dataclasses
import math
import random

from accel_dse.api import ApiError, api_eval
from accel_dse.core import pdqueue as q
from accel_dse.core import pdsim as ps
from accel_dse.core.disagg import disagg_report
from accel_dse.core.prefixcache import (tree_cum_tokens, tree_depth_law, tree_depth_tail, tree_groups,
                                        tree_resident)
from accel_dse.core.validation import v4_scenario

TREE = ((512, 4, 1.0), (1536, 200, 1.0), (1024, 100, 0.0))


def _tree_scn(load=0.6, cv=0.0, gb=20.0):
    s = v4_scenario(load, cv, False, "dense8b")
    return dataclasses.replace(s, pd=dataclasses.replace(s.pd, prefix_tree=TREE, prefix_cache_GB=gb))


def test_gammq():
    for a in (0.5, 1.0, 2.5):
        for x in (0.2, 1.0, 4.0):
            n, hi = 20000, 60.0
            h = (hi - x) / n
            num = sum(math.exp(-(x + (i + .5) * h) + (a - 1) * math.log(x + (i + .5) * h) - math.lgamma(a))
                      for i in range(n)) * h
            assert abs(q._gammq(a, x) - num) < 1e-5
    assert abs(q._gammq(1.0, 2.0) - math.exp(-2)) < 1e-12


def test_tree_che_vs_radix_lru():
    lv = ((1024, 8, 1.0), (2048, 50, 0.8))
    g = tree_groups(lv)
    tot = sum(L * sum(c for c, _ in gk) for (L, _, _), gk in zip(lv, g))
    C = 0.1 * tot
    H = tree_depth_tail(g, tree_resident(lv, C, g))
    rng = random.Random(3)
    c = ps.RadixLRU(C, [1024, 2048])
    for p in ps._tree_paths(rng, lv, 20000):
        c.access(p)
    c.reset_stats()
    for p in ps._tree_paths(rng, lv, 60000):
        c.access(p)
    sim = [sum(c.depth_n[k:]) / c.looks for k in (1, 2)]
    assert max(abs(a - b) for a, b in zip(H, sim)) < 0.01, (H, sim)
    law = tree_depth_law(H)
    assert abs(sum(law) - 1) < 1e-12 and all(x >= 0 for x in law)
    assert tree_cum_tokens(lv) == [0, 1024, 3072]
    assert all(x == 1.0 for gk in tree_resident(lv, 2 * tot, g) for x in gk)


def test_prefix_tree_scenario_typing():
    sc = {"model": "qwen3-8b", "serving": {"batch": 8, "prompt": 8192, "out_len": 64},
          "pd": {"enabled": True, "prefix_tree": [[1024, 8, 1], [4096, 100, 1.0]]}}
    r = api_eval({"scenario": sc, "chip_preset": "1P"})
    assert r["scenario"]["pd"]["prefix_tree"] == [[1024, 8, 1.0], [4096, 100, 1.0]]
    for bad, frag in (([[1024, 8]], "prefix_tree"), ([[1024, 8.5, 1]], "prefix_tree[0][1]"), ([[1, 2, 5]], "zipf")):
        try:
            api_eval({"scenario": {**sc, "pd": {"enabled": True, "prefix_tree": bad}}, "chip_preset": "1P"})
        except ApiError as e:
            assert frag in str(e)
        else:
            raise AssertionError(bad)
    try:
        api_eval({"scenario": {**sc, "pd": {"enabled": True, "prefix_tree": [[1024, 8, 1]], "prefix_len": 512}},
                  "chip_preset": "1P"})
    except ApiError as e:
        assert "alternatives" in str(e)
    else:
        raise AssertionError("prefix_tree + prefix_len accepted")


def test_tree_report_and_des():
    scn = _tree_scn()
    rep = disagg_report(scn)
    pc = rep["prefix_cache"]
    assert pc["tree"] and len(pc["prefill"]["level_hit"]) == 3
    lh = pc["prefill"]["level_hit"]
    assert lh[0] >= lh[1] >= lh[2] and 0.2 < lh[1] < 0.6           # partial hits at the document level
    assert abs(pc["prefill"]["token_hit"] - sum(L * h for (L, _, _), h in zip(TREE, lh)) / 3072) < 1e-9
    rep2, ctx = ps.capture_ctx(scn)
    assert sorted(p for _, _, p in ctx["pts"]) == [0, 512, 2048, 3072]     # one class per matched depth
    assert abs(sum(w for w, _, _ in ctx["pts"]) - 1) < 1e-9
    lam = rep2["queue"]["lambda_rps"]
    s = ps.simulate(ctx, lam, "pd", n_req=800, warmup=200, seed=3, prefix_tree=TREE,
                    prefix_K=pc["prefill"]["capacity_tokens"], prefix_K_dec=pc["decode"]["capacity_tokens"])
    assert abs(s["prefix_hit"] - pc["prefill"]["hit"]) < 0.05
    assert abs(s["prefix_level_hit"][1] - lh[1]) < 0.06


def test_coloc_slot_hold_wait_shape_chunked_refill():
    s = v4_scenario(0.85, 1.0, False, "dense8b")
    rep, _ = ps.capture_ctx(dataclasses.replace(s, pd=dataclasses.replace(s.pd, kv_policy="recompute",
                                                                           kv_capacity_GB=12.0)))
    m = rep["queue"]["modes"]
    for k in ("coloc_prefill_first", "coloc_chunked"):
        c = m[k]["kv_cap"]
        assert 0 < c["slot_hold_prefill_ms"] < m[k]["ttft_ms"]["mean"]
        assert 1.3 < c["wait_c2"] < 1.8
    assert m["pd"]["kv_cap"]["wait_c2"] == 1.0
    ch = m["coloc_chunked"]
    assert abs(ch["kv_cap"]["refill_tight_share"] - (1 - ch["prefill"]["rho"])) < 1e-9
    assert "refill_tight_share" not in m["coloc_prefill_first"]["kv_cap"]
