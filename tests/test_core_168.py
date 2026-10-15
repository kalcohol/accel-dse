"""0.168: host-memory (L2) tier of the whole-prefix cache (SGLang HiCache write_through / write_back, layer-wise
load-back overlapped with prefill)."""
import math
from types import SimpleNamespace

from accel_dse.core.disagg import _host_load, _k_host, _k_tot, disagg_report
from accel_dse.core.prefixcache import hit_of, pool_hits, zipf_groups
from accel_dse.core.scenario import PDConfig

from test_core_053 import sc


def test_union_capacity_by_policy():
    cap = {"K": 100, "footprint_MB": 300.0}
    wt = SimpleNamespace(prefix_host_GB=60.0, prefix_host_policy="write_through")     # 200 prefixes on host
    wb = SimpleNamespace(prefix_host_GB=60.0, prefix_host_policy="write_back")
    off = SimpleNamespace(prefix_host_GB=0.0, prefix_host_policy="write_through")
    assert _k_host(wt, cap) == 200
    assert _k_tot(wt, cap, None, 0) == 200 and _k_tot(wb, cap, None, 0) == 300 and _k_tot(off, cap, None, 0) == 100
    small = SimpleNamespace(prefix_host_GB=15.0, prefix_host_policy="write_through")  # 50 < K1: inclusive adds nothing
    assert _k_tot(small, cap, None, 0) == 100
    for bad in (dict(prefix_host_GB=-1.0), dict(prefix_host_policy="lru"), dict(prefix_host_GB=True)):
        try:
            PDConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_report_hits_load_and_ttft():
    base = disagg_report(sc(prefix_len=2048, prefix_count=100_000, prefix_zipf=1.0))
    r = disagg_report(sc(prefix_len=2048, prefix_count=100_000, prefix_zipf=1.0, prefix_host_GB=200.0,
                         prefix_host_policy="write_back"))
    h = r["prefix_cache"]["host"]
    K1 = r["prefix_cache"]["prefill"]["K"]
    # footprint: qwen3-8b KV 36 layers · 8 kv heads · 128 · 2 (K, V) · 2 B = 147456 B / token → 2048 tokens = 302 MB
    assert abs(r["prefix_cache"]["prefill"]["footprint_MB"] - 147456 * 2048 / 1e6) < 1e-6
    assert h["K_host"] == int(200e9 // (147456 * 2048))
    g = zipf_groups(100_000, 1.0)
    r_p = r["prefix_cache"]["prefill"]["replicas"]
    H1 = hit_of(g, pool_hits(g, K1, r_p, False))
    H12 = hit_of(g, pool_hits(g, K1 + h["K_host"], r_p, False))
    assert abs(h["hit_device"] - H1) < 1e-12 and abs(h["hit_host"] - (H12 - H1)) < 1e-12 and h["hit_host"] > 0
    assert r["prefix_cache"]["prefill"]["hit"] > base["prefix_cache"]["prefill"]["hit"]
    ld = h["load"]
    # each of the 2 TP cards loads its half (151 MB) at 50 GB/s 「假设」 host link → 3.02 ms; layer-wise overlap
    assert abs(ld["bytes_per_card_MB"] - 147456 * 2048 / 2 / 1e6) < 1e-6 and abs(ld["t_load_ms"] - 3.01989888) < 1e-6
    assert abs(ld["mean_added_ms"] - h["hit_host"] * ld["exposed_ms"]) < 1e-12
    assert abs(ld["req_s_replica"] - 50e9 / (h["hit_host"] * ld["bytes_per_card_MB"] * 1e6)) < 1e-6
    assert "host" in r["cap_req_s"] and "host" not in base["cap_req_s"] and "host" not in base["util"]


def test_exposed_rule_and_ignore_warning():
    pd = SimpleNamespace(swap_GBps=None, prefix_len=1000)
    m = SimpleNamespace(n_layers=10)
    import accel_dse.core.disagg as D
    orig = D.per_card_bytes
    D.per_card_bytes = lambda *a, **k: [1e9]
    try:
        scn = SimpleNamespace(pd=pd, workload=SimpleNamespace(host_GBps=10.0))
        pr = SimpleNamespace(scenario=SimpleNamespace(layout=SimpleNamespace(pp=1)), stages=[])
        ld = _host_load(scn, m, pr, {"host": {"hit_host": 0.5}})
    finally:
        D.per_card_bytes = orig
    assert abs(ld["t_load_ms"] - 100.0) < 1e-9                       # 1 GB / 10 GB/s
    assert abs(ld["exposed_s"](0.0) - 0.1) < 1e-12                   # nothing to hide behind
    assert abs(ld["exposed_s"](1.0) - 0.01) < 1e-12                  # floor: one layer's load
    assert abs(ld["exposed_s"](0.05) - (0.1 - 0.045)) < 1e-12
    assert abs(ld["req_s_replica"] - 10e9 / (0.5 * 1e9)) < 1e-9
    w = disagg_report(sc(prefix_hit=0.3, prefix_host_GB=100.0))
    assert any("prefix_host_GB 已忽略" in x for x in w["warnings"])
