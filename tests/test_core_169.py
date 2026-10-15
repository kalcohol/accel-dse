"""0.169: host L2 prefix tier on the radix-tree / multi-turn models; card → host backup write traffic (own PCIe
direction, own capacity)."""
import dataclasses
from types import SimpleNamespace

from accel_dse.core.disagg import _host_entry, _reach_cap, disagg_report
from accel_dse.core.prefixcache import tree_cum_tokens

from test_core_058 import TREE, _tree_scn


def _run(**kw):
    s = _tree_scn()
    return disagg_report(dataclasses.replace(s, pd=dataclasses.replace(s.pd, **kw)))


def test_reach_capacity_and_write_fraction():
    cap = {"K": 3, "K_float": 3.5, "footprint_MB": 100.0}
    wt = SimpleNamespace(prefix_host_GB=1.0, prefix_host_policy="write_through")      # 10 prefixes
    wb = SimpleNamespace(prefix_host_GB=1.0, prefix_host_policy="write_back")
    assert _reach_cap(wt, cap)["K_float"] == 10.0 and abs(_reach_cap(wb, cap)["K_float"] - 13.5) < 1e-12
    off = SimpleNamespace(prefix_host_GB=0.0, prefix_host_policy="write_through")
    assert _reach_cap(off, cap) is cap
    # write_through backs up every newly computed token (1 − H); write_back demotes L1 evictions (1 − H1)
    assert abs(_host_entry(wt, cap, 0.3, 0.5, 0.3, 0.5)["write_frac"] - 0.5) < 1e-12
    assert abs(_host_entry(wb, cap, 0.3, 0.5, 0.3, 0.5)["write_frac"] - 0.7) < 1e-12


def test_tree_mode_host_hits_and_write_cap():
    base, r = _run(), _run(prefix_host_GB=100.0)
    pc, h = r["prefix_cache"], r["prefix_cache"]["host"]
    Lp = tree_cum_tokens(TREE)[-1]
    assert pc["prefix_len"] == Lp and "host" not in base["prefix_cache"]
    assert abs(h["hit_device"] - base["prefix_cache"]["prefill"]["hit"]) < 1e-9
    assert h["hit_host"] > 0 and abs(pc["prefill"]["hit"] - (h["hit_device"] + h["hit_host"])) < 1e-12
    ld = h["load"]
    assert abs(ld["write_MB_card_req"] - h["write_frac"] * ld["bytes_per_card_MB"]) < 1e-9
    assert abs(ld["write_req_s_replica"] - ld["host_GBps_card"] * 1e9 / (ld["write_MB_card_req"] * 1e6)) < 1e-6
    assert "host_write" in r["cap_req_s"] and "host_write" in r["util"] and "host_write" not in base["cap_req_s"]
    slow = _run(prefix_host_GB=100.0, swap_GBps=0.05)                     # slow host link → it binds
    assert slow["bottleneck"] in ("host", "host_write") and slow["req_s"] < r["req_s"]


def test_multi_turn_host_tier():
    r = _run(prefix_host_GB=100.0, prefix_turns=3, prefix_turn_user=200)
    h = r["prefix_cache"]["host"]
    assert "turns" in r["prefix_cache"] and h["hit_host"] > 0 and 0 < h["write_frac"] < 1
