"""0.53 — prefix-cache capacity (LRU / Che) + heterogeneous PD pools + decode-batch layout search."""

from __future__ import annotations

import math
import random
import collections

from accel_dse.core.disagg import disagg_report
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.prefixcache import capacity, lru_hit, zipf_groups, che_T, resident
from accel_dse.core.queueing import mdc_wait
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.api import scenario_from_body

SV = Serving(batch=64, prompt=4096, out_len=512, ttft_slo_ms=400, tpot_slo_ms=10)
BASE = dict(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=2), serving=SV)
PD = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6)


def sc(**pd) -> Scenario:
    return Scenario(**{**BASE, "pd": PDConfig(**{**PD, **pd})})


def test_che_uniform_exact():
    """Uniform Zipf(0): H = K/N for K ≤ N."""
    for n, K in ((200, 50), (1000, 100), (1000, 1000)):
        assert abs(lru_hit(n, 0.0, K) - min(1.0, K / n)) < 1e-9


def test_che_vs_sim():
    """Che ≈ discrete LRU under IRM (absolute error < 1 % after warm-up)."""
    def sim(n, a, K, R=300_000, seed=7):
        rng = random.Random(seed)
        w = [i ** -a for i in range(1, n + 1)]
        reqs = rng.choices(range(n), weights=w, k=R)
        od = collections.OrderedDict()
        hit = 0
        warm = R // 5
        for i, x in enumerate(reqs):
            if x in od:
                od.move_to_end(x)
                hit += i >= warm
            else:
                od[x] = 1
                if len(od) > K:
                    od.popitem(last=False)
        return hit / (R - warm)
    for n, a, K in ((1000, 0.8, 100), (1000, 1.0, 50), (2000, 0.6, 400)):
        assert abs(lru_hit(n, a, K) - sim(n, a, K)) < 0.01


def test_che_monotone_in_K():
    h = [lru_hit(5000, 0.9, K) for K in (0, 10, 100, 500, 2000, 5000)]
    assert h[0] == 0.0 and h[-1] == 1.0
    assert all(a <= b + 1e-12 for a, b in zip(h, h[1:]))


def test_che_binning_matches_exact():
    """Log-binned ranks for N > 4096 agree with exact within 1e-5."""
    from accel_dse.core import prefixcache as pc
    for n, a, K in ((20000, 0.9, 2000), (100000, 0.5, 20000)):
        b = pc.lru_hit(n, a, K)
        was = pc.EXACT_N
        pc.EXACT_N = 10 ** 7
        e = pc.lru_hit(n, a, K)
        pc.EXACT_N = was
        assert abs(b - e) < 1e-5


def test_prefix_capacity_emergent_hit():
    r = disagg_report(sc(prefix_len=2048, prefix_count=100_000, prefix_zipf=1.0))
    pc = r["prefix_cache"]
    assert pc is not None and r["lengths"]["prefix_hit_source"] == "capacity"
    assert 0 < pc["prefill"]["hit"] < 1
    assert pc["prefill"]["K"] > 0 and pc["prefill"]["footprint_MB"] > 0
    assert abs(r["lengths"]["prefix_hit"] - pc["prefill"]["hit"]) < 1e-12
    # larger capacity → higher hit
    r2 = disagg_report(sc(prefix_len=2048, prefix_count=100_000, prefix_cache_GB=2000))
    assert r2["prefix_cache"]["prefill"]["hit"] >= pc["prefill"]["hit"] - 1e-9


def test_prefix_hit_overrides_capacity():
    r = disagg_report(sc(prefix_len=2048, prefix_hit=0.3))
    assert r["prefix_cache"] is None
    assert r["lengths"]["prefix_hit"] == 0.3 and r["lengths"]["prefix_hit_source"] == "explicit"
    assert any("覆盖" in w for w in r["warnings"])


def test_prefix_affinity_raises_hit():
    kw = dict(prefix_len=2048, prefix_count=100_000, prefix_zipf=0.8)
    a = disagg_report(sc(**kw))["prefix_cache"]["decode"]["hit"]
    b = disagg_report(sc(**kw, prefix_affinity=True))["prefix_cache"]["decode"]["hit"]
    assert b >= a - 1e-12


def test_hetero_prefill_chip_mem():
    r = disagg_report(sc(prefill_chip=CHIPS["100T"], prefill_mem_id="lpddr5x_4x64_8533_16g"))
    assert r["prefill"]["hetero"] and r["prefill"]["chip"] == "100T"
    assert r["prefill"]["mem_id"] == "lpddr5x_4x64_8533_16g"
    assert r["decode"]["chip"] == "1P" and r["decode"]["mem_id"] == "hbm3e_8s_12h24g_9200"
    # colocated comparison stays on the decode (scenario) chip
    assert r["coloc"]["layout"] == "PP1·TP2"


def test_default_identical_to_052():
    """Inactive 0.53 knobs leave the PD report unchanged (the 0.52 code path; see also the fingerprint check)."""
    r = disagg_report(sc())
    assert r["prefix_cache"] is None and r["lengths"]["plain"] and not r["prefill"]["hetero"]
    # prefix-cache knobs without prefix_len (and decode-batch search without layout search) change nothing
    r2 = disagg_report(sc(prefix_count=7, prefix_zipf=2.0, prefix_affinity=True, prefix_cache_GB=1.0,
                          search_decode_batch=True))
    assert r == r2


def test_mdc_wait_at_capacity():
    """Hairline ρ → 1 no longer ZeroDivisionError (0.53)."""
    s = mdc_wait(1.0, 1.0, 1)
    assert not s["stable"] and math.isinf(s["mean"])


def test_api_prefill_chip_name_and_mem():
    body = {"chip_preset": "1P", "scenario": {
        "model": "qwen3-8b", "mem_id": "hbm3e_8s_12h24g_9200", "layout": {"tp": 2},
        "serving": {"batch": 64, "prompt": 4096, "out_len": 512},
        "pd": {"enabled": True, "prefill_layout": {"tp": 2}, "prefill_cards": 2, "decode_cards": 6,
               "prefill_chip": "100T", "prefill_mem_id": "lpddr5x_4x64_8533_16g", "prefix_len": 1024}}}
    s = scenario_from_body(body)
    assert s.pd.prefill_chip.name == "100T" and s.pd.prefill_mem_id.startswith("lpddr")
    r = disagg_report(s)
    assert r["prefill"]["chip"] == "100T" and r["prefix_cache"]["prefix_len"] == 1024


def test_search_decode_batch_field():
    r = disagg_report(sc(search_layouts=True, search_decode_batch=True))
    ls = r["layout_search"]
    assert ls["decode_batch_searched"] and all("decode_batch" in row for row in ls["rows"])


def test_from_dict_roundtrip():
    s = sc(prefix_len=512, prefix_count=99, prefix_zipf=0.5, prefix_cache_GB=12.5,
           prefix_affinity=True, prefill_chip=CHIPS["100T"], search_decode_batch=True)
    d = s.to_dict()
    assert d["pd"]["prefix_len"] == 512 and d["pd"]["prefill_chip"]["name"] == "100T"
    s2 = Scenario.from_dict(d)
    assert s2.pd.prefix_affinity and s2.pd.prefill_chip.name == "100T"
    assert s2.pd.prefix_cache_GB == 12.5 and s2.pd.search_decode_batch
