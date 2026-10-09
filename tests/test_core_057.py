"""0.57 — KV admission wait as an M/G/c slot queue (Cosmetatos M/D/c blend, colocated slot hold through prefill),
recompute / swap preemption: refill offset (the departing request frees its own output), restore-share fixed point
on the slowed slot chain, DES restore accounting; V4 grid per-load DES length."""

from __future__ import annotations

import dataclasses
import inspect
import math
import random

from accel_dse.core import pdqueue as q
from accel_dse.core import pdsim as ps
from accel_dse.core.validation import v4_grid, v4_scenario


def _kv(load, cv, pol, gb=12.0, mode=None):
    s = v4_scenario(load, cv, False, "dense8b")
    return dataclasses.replace(s, pd=dataclasses.replace(s.pd, kv_policy=pol, kv_capacity_GB=gb))


_CACHE: dict = {}


def _rep(load, cv, pol):
    k = (load, cv, pol)
    if k not in _CACHE:
        _CACHE[k] = ps.capture_ctx(_kv(load, cv, pol))
    return _CACHE[k]


def test_slot_chain_rebuild_and_slowdown():
    rep, ctx = _rep(0.85, 1.0, "wait")
    lam = rep["queue"]["lambda_rps"]
    B, kvi = q._kv_slots(ctx, ctx["B"])
    assert kvi["binds"]
    d = q._pd_mode_base({**ctx, "B": B}, lam)["_dec"]
    pi1 = d["pi_at"](1.0)
    assert max(abs(a - b) for a, b in zip(pi1, d["pi_n"])) < 1e-12
    pw = [sum((d["pi_at"](s) or [0] * (B + 1))[B:]) for s in (1.0, 1.02, 1.05)]
    assert pw[0] < pw[1] < pw[2]
    assert d["pi_at"](10.0) is None                    # slower than the arrival rate → unstable


def test_mdc_blend_and_colocated_slot_hold():
    rep, _ = _rep(0.85, 0.0, "wait")
    m = rep["queue"]["modes"]
    pd, pf = m["pd"]["kv_cap"], m["coloc_prefill_first"]["kv_cap"]
    assert pd["in_ttft"] and pd["p_wait"] > 0 and pd["slot_wait_mean_ms"] > 0
    assert "slot_hold_prefill_ms" not in pd               # PD takes the slot just before the KV pull
    assert pf["slot_hold_prefill_ms"] > 0                 # colocated before_prefill holds it through prefill


def test_refill_offset_and_cascade():
    rep0, _ = _rep(0.85, 0.0, "recompute")
    k0 = rep0["queue"]["modes"]["pd"]["kv_cap"]
    assert abs(k0["refill_offset_factor"] - math.exp(-1)) < 0.02        # fixed outputs: e^{-1}
    rep1, _ = _rep(0.85, 1.0, "recompute")
    k1 = rep1["queue"]["modes"]["pd"]["kv_cap"]
    assert 0.3 < k1["refill_offset_factor"] < 0.7
    assert k1["slot_slowdown"] > 1 and abs(k1["slot_slowdown"] - 1 / (1 - k1["restore_share"])) < 1e-9
    reps, _ = _rep(0.85, 1.0, "swap")
    ks = reps["queue"]["modes"]["pd"]["kv_cap"]
    # recompute restores stall the replica far longer than swap → slower slot chain → more queueing for KV
    assert k1["restore_share"] > ks["restore_share"] and k1["p_wait"] > ks["p_wait"]
    for mode in ("pd", "coloc_prefill_first", "coloc_chunked"):         # 0.57 counts (DES: 0.065 / 0.050 / 0.038)
        assert 0.02 < rep1["queue"]["modes"][mode]["kv_cap"]["preempt_per_req"] < 0.12


def test_des_restore_holds_kv_and_slot():
    eng = ps.Engine(random.Random(1))
    r = ps._Replica(eng, None, slots=2)
    r.kv_policy, r.kv_cap = "recompute", 1000
    a = ps.Req.__new__(ps.Req)
    a.S, a.tokens_done = 600, 100
    r.restoring = a
    b = ps.Req.__new__(ps.Req)
    b.S, b.out = 400, 10
    assert r._kv_used() == 700 and r._n_held() == 1
    assert not r._can_admit(b)                            # 700 + 400 > 1000: the restoring sequence keeps its room
    r.restoring = None
    assert r._can_admit(b)


def test_v4_grid_n_high_option():
    sig = inspect.signature(v4_grid)
    assert sig.parameters["n_high"].default is None and sig.parameters["high_load"].default == 0.85
