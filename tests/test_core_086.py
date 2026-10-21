"""0.86 adversarial round — standing invariants for feature combinations (PD + D2D chiplets + SLC + MoE skew +
speculative decoding + prefix cache) and numerical extremes (1 → 1024 cards, 1M context, chip parameter extremes).

Acceptance criteria: hard capacity (fits ⇔ every stage's DRAM need ≤ capacity), sum of parts = totals (energy by
action, PD fluid capacities), work conservation over TP, monotonicity in skew / acceptance / cached prefix / pinned
SLC, and the colocated fluid bound: with the prefill pool on the decode layout, PD goodput per card ≤ colocated
time-sharing goodput (a PD prefill pool on a cheaper layout, e.g. TP1 without collectives, may legitimately exceed it)."""
from __future__ import annotations

import math
import random
from dataclasses import replace

from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=5.0, pJ_bit_link=10.0, idle_W=100.0,
                pJ_bit_slc=1.0, pJ_bit_d2d=2.0, pJ_bit_net=20.0)
MOE = ("qwen3-30b-a3b", "gpt-oss-20b")
H = "hbm3e_8s_12h24g_9200"


def _rand(rnd):
    mid = rnd.choice(("qwen3-8b", "qwen3-30b-a3b", "gpt-oss-20b", "llama-3.1-8b"))
    moe = mid in MOE
    tp = rnd.choice([1, 2, 4]); dp = rnd.choice([1, 2]) if moe else 1
    chip = replace(CHIPS[rnd.choice(["100T", "1P"])], slc_mib=rnd.choice([0, 64, 256]), slc_policy=rnd.choice(["pin", "lru"]))
    k = rnd.choice([0, 2, 3])
    pr = rnd.choice([2048, 8192])
    sv = Serving(phase=rnd.choice(["decode", "prefill"]), batch=rnd.choice([1, 16, 64]), prompt=pr, out_len=rnd.choice([64, 512]),
                 moe_skew=rnd.choice([1.0, 1.3]) if moe else 1.0, spec_k=k, spec_accept=0.5 if k else 0.0,
                 prefix_cached=rnd.choice([0, 128]))
    d2d = rnd.random() < 0.5
    return Scenario(model=mid, chip=chip, layout=Layout(tp=tp, dp=dp, ep=tp * dp if moe else 1), serving=sv, node_cards=8,
                    d2d_enabled=d2d, package_cards=rnd.choice([2, 4]) if d2d else 1, mem_id=rnd.choice(["lpddr5x_4x64_8533_16g", H]))


def test_combo_evaluate_invariants():
    rnd = random.Random(86)
    n = 0
    while n < 16:
        s = _rand(rnd)
        try:
            r = evaluate(s)
        except ValueError:
            continue
        n += 1
        case = (s.model, s.layout, s.serving, s.chip.slc_mib, s.d2d_enabled)
        assert r.fits == all(st.mem.dram_need <= st.mem.dram_cap * (1 + 1e-9) for st in r.stages), case
        for st in r.stages:
            t = st.time
            comps = (t.t_mac, t.t_vector, t.t_dram, t.t_link)
            assert all(math.isfinite(c) and c >= 0 for c in comps) and t.total >= max(comps) * (1 - 1e-9), case
        if not r.fits:
            continue
        e = energy_report(r, T)
        J = e["J_by_action"]
        assert abs(sum(J.values()) - e["J_per_unit"]) <= 1e-9 * e["J_per_unit"], case
        assert abs(energy_report(r, replace(T, pJ_bit_dram=10.0))["J_by_action"]["dram"] - 2 * J["dram"]) <= 1e-9 * J["dram"], case
        if not s.d2d_enabled:
            assert J.get("d2d", 0.0) == 0.0, case
        if s.chip.slc_mib == 0:
            assert J.get("slc", 0.0) == 0.0, case
        sv = s.serving
        if s.model in MOE:
            r2 = evaluate(replace(s, serving=replace(sv, moe_skew=2.0)))
            assert not r2.fits or r2.step >= r.step * (1 - 1e-9), case
        if sv.spec_k and sv.phase == "decode":
            r2 = evaluate(replace(s, serving=replace(sv, spec_accept=0.8)))
            assert not r2.fits or r2.throughput >= r.throughput * (1 - 1e-9), case
        if sv.phase == "prefill":
            r2 = evaluate(replace(s, serving=replace(sv, prefix_cached=1024)))
            assert not r2.fits or r2.latency <= r.latency * (1 + 1e-9) and r2.step <= r.step * (1 + 1e-9), case
        if s.layout.tp > 1:
            r1 = evaluate(replace(s, layout=Layout(tp=1, dp=s.layout.dp, ep=s.layout.dp if s.model in MOE else 1)))

            def w(x):
                L = x.scenario.layout
                return sum(st.flops_u for st in x.stages) * L.tp * L.sp * L.dp / -(-x.scenario.serving.batch // x.microbatches)
            assert abs(w(r) / w(r1) - 1) < 1e-6, case


def _pd_identities(r, out):
    d, cap = r["decode"], r["cap_req_s"]
    assert abs(r["req_s"] - min(cap.values())) <= 1e-9 * max(1e-12, r["req_s"])
    assert all(-1e-12 <= u <= 1 + 1e-9 for u in r["util"].values())
    if d["fits"] and d["tok_s_replica"] > 0:
        assert abs(cap["decode"] - d["replicas"] * d["tok_s_replica"] / out) <= 1e-9 * cap["decode"]
    assert abs(r["goodput_tok_s"] - r["req_s"] * out) <= 1e-9 * max(1e-12, r["goodput_tok_s"])
    for e in (r["queue"].get("energy") or {}).values():
        if e and e.get("J_by_action"):
            assert abs(sum(e["J_by_action"].values()) - e["J_per_token"]) <= 1e-9 * e["J_per_token"]


def test_pd_colocated_fluid_bound_same_layout():
    for mid, L, slc, d2d, k in (("qwen3-8b", Layout(tp=2), 64, True, 2), ("qwen3-30b-a3b", Layout(tp=2, ep=2), 0, False, 0),
                                ("llama-3.1-8b", Layout(tp=4), 256, True, 3)):
        sv = Serving(batch=16, prompt=2048, out_len=256, spec_k=k, spec_accept=0.6 if k else 0.0,
                     moe_skew=1.3 if mid in MOE else 1.0)
        for pc, dc in ((1, 1), (2, 1), (1, 2)):
            s = Scenario(model=mid, chip=replace(CHIPS["100T"], slc_mib=slc), layout=L, serving=sv, node_cards=8, mem_id=H,
                         d2d_enabled=d2d, package_cards=2 if d2d else 1,
                         pd=PDConfig(enabled=True, prefill_layout=L, prefill_cards=pc * L.cards, decode_cards=dc * L.cards))
            r = disagg_report(s, energy=T)
            _pd_identities(r, 256)
            co = r["coloc"]["goodput_per_card"]
            assert co > 0 and r["goodput_per_card"] <= co * (1 + 1e-6), (mid, pc, dc, r["goodput_per_card"], co)
            bs = r["best_split"]
            assert r["goodput_per_card"] * (1 - 1e-9) <= bs["goodput_per_card"] <= co * (1 + 1e-6), (mid, pc, dc)


def test_pd_combo_kv_policy_prefix_tree_hetero():
    sv = Serving(batch=32, prompt=2048, out_len=256, spec_k=2, spec_accept=0.7, moe_skew=1.3)
    base = Scenario(model="qwen3-30b-a3b", layout=Layout(tp=2, ep=2), serving=sv, node_cards=8, mem_id=H,
                    chip=replace(CHIPS["100T"], slc_mib=64), d2d_enabled=True, package_cards=2)
    for kw in (dict(kv_policy="recompute", kv_capacity_GB=8.0),
               dict(kv_policy="swap", kv_capacity_GB=8.0),
               dict(prefix_tree=((512, 1, 0.0), (256, 4, 1.0)), prefix_cache_GB=0.5),
               dict(prefill_chip=CHIPS["1P"], prefix_hit=0.5)):
        r = disagg_report(replace(base, pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2, **kw)), energy=T)
        _pd_identities(r, 256)
        for m, x in r["queue"]["modes"].items():
            for key in ("ttft_ms", "tpot_mean_ms", "itl_max_ms"):
                v = x.get(key)
                v = v.get("mean") if isinstance(v, dict) else v
                assert v is None or (math.isfinite(v) and v >= 0), (kw, m, key, v)


def test_numerical_extremes():
    # 1M context: fits on 8 HBM cards, finite, warns beyond max_position_embeddings, tpot non-decreasing in ctx
    for mid, L in (("qwen3-8b", Layout(tp=8)), ("deepseek-v3.2", Layout(tp=8, ep=8)), ("qwen3-next-80b-a3b", Layout(tp=8, ep=8))):
        prev = 0.0
        for ctx in (4096, 131072, 1048576):
            r = evaluate(Scenario(model=mid, layout=L, mem_id=H, node_cards=8, serving=Serving(phase="decode", batch=1, ctx=ctx)))
            assert r.fits and math.isfinite(r.step) and r.step >= prev * (1 - 1e-9), (mid, ctx)
            prev = r.step
        assert any("max_position_embeddings" in w for w in r.warnings), mid
    # 1 → 1024 cards, batch 1 → 65536: finite, fits ⇔ capacity
    for L in (Layout(tp=8, ep=8), Layout(tp=8, dp=128, ep=1024)):
        for b in (1, 65536):
            r = evaluate(Scenario(model="deepseek-v3", layout=L, mem_id=H, node_cards=8, serving=Serving(phase="decode", batch=b, ctx=4096)))
            assert all(math.isfinite(x) and x >= 0 for x in (r.step, r.throughput, r.per_card)), (L, b)
            assert r.fits == all(st.mem.dram_need <= st.mem.dram_cap * (1 + 1e-9) for st in r.stages), (L, b)
    r = evaluate(Scenario(model="qwen3-0.6b", serving=Serving(phase="decode", batch=1, ctx=1024)))
    assert r.fits and r.throughput > 0
    # chip parameter extremes stay finite
    for fq in (0.1, 10.0):
        for sram in (1, 16384):
            chip = replace(CHIPS["100T"], freq_ghz=fq, sram_mib=sram)
            for ph in ("decode", "prefill"):
                r = evaluate(Scenario(model="qwen3-8b", chip=chip, serving=Serving(phase=ph, batch=8, prompt=2048, ctx=2048)))
                assert math.isfinite(r.step) and r.step > 0, (fq, sram, ph)
