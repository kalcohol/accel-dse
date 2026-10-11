"""0.81: prefix cache (Che / radix) vs the DES caches on random workloads, radix node-granularity slack, chip preset
names in the constructors, PD energy identities, speculative-decoding energy scaling."""
import random
from dataclasses import replace

from accel_dse.core.disagg import disagg_report, kv_bytes_per_request
from accel_dse.core.catalog import get_model
from accel_dse.core.energy import EnergyTable, action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.pdsim import LRUCache, RadixLRU, _tree_paths, _zipf_ids
from accel_dse.core.prefixcache import lru_hit, tree_depth_tail, tree_groups, tree_resident
from accel_dse.core.scenario import Layout, PDConfig, Scenario, Serving
from accel_dse.core.validation import v4_scenario


def _tree(levels, C):
    g = tree_groups(levels)
    return tree_depth_tail(g, tree_resident(levels, C, g))


def test_radix_single_length_is_whole_nodes():
    # 308 tokens hold two 128-token nodes (the LRU evicts whole nodes): uniform over 10 → H = 0.2 (was 0.24)
    assert abs(_tree([(128, 10, 0.0)], 308)[0] - 0.2) < 1e-9
    assert _tree([(128, 10, 0.0)], 100)[0] == 0.0
    assert abs(_tree([(128, 10, 0.0)], 1280)[0] - 1.0) < 1e-12


def test_radix_che_vs_radix_lru_random():
    rng, errs = random.Random(21), []
    for i in range(12):
        m = rng.choice([1, 2])
        levels = [(rng.choice([16, 128, 1024]), rng.choice([3, 10, 30]), rng.choice([0.0, 0.8, 1.2])) for _ in range(m)]
        tot, cnt = 0, 1
        for L, n, _ in levels:
            cnt *= n
            tot += L * cnt
        C = rng.uniform(0.05, 0.9) * tot
        ana = _tree(levels, C)
        c = RadixLRU(C, [L for L, _, _ in levels])
        P = _tree_paths(random.Random(i), levels, 24000)
        for p in P[:8000]:
            c.access(p)
        c.reset_stats()
        for p in P[8000:]:
            c.access(p)
        T = sum(c.depth_n)
        sim = [sum(c.depth_n[k:]) / T for k in range(1, m + 1)]
        errs.append(max(abs(a - b) for a, b in zip(ana, sim)))
    assert sum(errs) / len(errs) < 0.03 and max(errs) < 0.15, errs


def test_whole_prefix_lru_vs_che_random():
    rng = random.Random(22)
    for i in range(10):
        n, a = rng.choice([50, 500, 3000]), rng.choice([0.0, 0.8, 1.2])
        K = rng.randint(1, n - 1)
        c = LRUCache(K)
        ids = _zipf_ids(random.Random(i), n, a, 40 * n)
        for p in ids[:10 * n]:
            c.access(p)
        c.hits = c.looks = 0
        for p in ids[10 * n:]:
            c.access(p)
        ana = lru_hit(n, a, K)
        assert abs(ana - c.hit_rate) < 0.02 + 0.03 * ana, (n, a, K, ana, c.hit_rate)


def test_chip_preset_names_in_constructors():
    a, b = Scenario(model="qwen3-8b", chip="1P"), Scenario(model="qwen3-8b", chip=CHIPS["1P"])
    assert a.chip is CHIPS["1P"] and a.hash() == b.hash() and evaluate(a).tpot == evaluate(b).tpot
    assert PDConfig(prefill_chip="100T").prefill_chip is CHIPS["100T"]
    for bad in ("nope", 3):
        try:
            Scenario(chip=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_pd_energy_identities():
    s = v4_scenario(0.6)
    T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=4.0, pJ_bit_link=10.0, idle_W=100.0,
                    idle_W_prefill=60.0)
    q = disagg_report(s, energy=T)["queue"]
    lam, o = q["lambda_rps"], s.serving.out_len
    n_p, n_d = s.pd.prefill_cards, s.pd.decode_cards
    e = q["energy"]
    assert abs(e["pd"]["J_by_action"]["idle"] - 100 * n_d / lam / o) < 1e-12
    assert abs(e["pd"]["J_by_action"]["idle_prefill"] - 60 * n_p / lam / o) < 1e-12
    assert abs(e["coloc_prefill_first"]["J_by_action"]["idle"] - 100 * (n_p + n_d) / lam / o) < 1e-12
    # the PD-only link bytes = the KV hand-off, once per request
    kv = kv_bytes_per_request(get_model(s.model), s.serving.prompt) / o
    d = e["pd"]["counts_per_token"]["link"] - e["coloc_prefill_first"]["counts_per_token"]["link"]
    assert abs(d - kv) < 1e-6 * kv
    q2 = disagg_report(s, energy=replace(T, pJ_mac=1.0))["queue"]["energy"]
    for m in e:
        assert abs(q2[m]["J_by_action"]["mac"] / e[m]["J_by_action"]["mac"] - 2.0) < 1e-12


def test_spec_decode_energy_per_token():
    def per_tok(k, a):
        r = evaluate(Scenario(model="glm-4.5-air", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200",
                              layout=Layout(tp=8, ep=8), serving=Serving(batch=32, spec_k=k, spec_accept=a)))
        c = action_counts(r)
        return c["units"], c["counts"]["mac"] / c["units"], c["counts"]["dram"] / c["units"]
    u0, m0, d0 = per_tok(0, 0.7)
    u2, m2, d2 = per_tok(2, 0.7)
    assert abs(u2 / u0 - (1 + 0.7 + 0.49)) < 1e-9          # units = accepted tokens: E = Σ a^i
    uz, mz, _ = per_tok(2, 0.0)
    assert uz == u0 and 2.9 < mz / m0 < 3.4                 # nothing accepted: 3 verified positions + drafts
    assert d2 < d0                                          # weights amortised over E tokens
    try:                                                    # no MTP head → warned and ignored
        w = evaluate(Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200",
                              serving=Serving(spec_k=2))).warnings
        assert any("spec_k" in x for x in w)
    except ValueError:
        raise AssertionError("no-MTP spec_k must warn, not fail")


def test_des_nonbinding_kv_cap_equals_cap_off():
    # 0.81: before_prefill KV-arrived requests join at the iteration boundary — a cap that never binds must measure
    # the same TPOT / worst gap as no cap (was: worst-gap p99 +43 %, TPOT inflated by the in-flight remainder)
    from accel_dse.core.pdsim import capture_ctx, simulate
    s0 = replace(v4_scenario(0.3, 0.0, False, "moe30b"), pd=replace(v4_scenario(0.3, 0.0, False, "moe30b").pd,
                 kv_policy="recompute", kv_capacity_GB=8.0, kv_admit="before_prefill"))
    s1 = replace(s0, pd=replace(s0.pd, kv_policy="off", kv_capacity_GB=None))
    out = []
    for s in (s0, s1):
        rep, ctx = capture_ctx(s)
        d = simulate(ctx, rep["queue"]["lambda_rps"], "pd", n_req=1500, warmup=300, seed=3)
        out.append((d["itl_max"], d["tpot"], d.get("preempt_per_req", 0.0)))
    assert out[0][2] == 0.0
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1], out
