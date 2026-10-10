"""0.73 adversarial audit — randomized standing invariants (deterministic seeds) for areas without property tests:
energy accounting, SLC policies, capacity partitions, fabric collectives, pipeline schedules, video / protein
schedules, search vs known feasible points, PD fluid / energy identities, length-law means.

Acceptance criteria: per-rank work conservation, legal schedule dependencies, hard capacity constraints, search
never worse than a known feasible point."""
from __future__ import annotations

import math
import random
from dataclasses import replace

from accel_dse.core import fabric as F
from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Fabric, System
from accel_dse.core.lengths import lognormal_bins
from accel_dse.core.parallel import Layout, enumerate_layouts
from accel_dse.core.catalog import get_model
from accel_dse.core.scenario import Scenario, Serving, Workload
from accel_dse.core.search import _feasible, best_batch, brute_force_batch, search_layouts

T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=5.0, pJ_bit_link=10.0, idle_W=100.0,
                pJ_bit_slc=1.0, pJ_bit_d2d=2.0, pJ_bit_net=20.0)
MOE = ("qwen3-30b-a3b", "deepseek-v3")
DOM = ("wan2.1-1.3b", "esm2-650m", "esmfold", "cogvideox-2b")


def _work(r):
    L = r.scenario.layout
    return sum(st.flops_u for st in r.stages) * L.tp * L.sp * L.dp / -(-r.scenario.serving.batch // r.microbatches)


def _rand_scn(rnd, models, pp=(1,)):
    mid = rnd.choice(models)
    tp, dp = rnd.choice([1, 2, 4]), rnd.choice([1, 2])
    if mid in DOM:
        lay = rnd.choice([Layout(), Layout(sp=2), Layout(dp=2)])
    elif mid in MOE:
        lay = Layout(tp=tp, dp=dp, ep=tp * dp, pp=rnd.choice(pp))
    else:
        lay = Layout(tp=tp, pp=rnd.choice(pp))
    sv = Serving(phase=rnd.choice(["decode", "prefill"]), batch=rnd.choice([1, 3, 16, 64]), prompt=rnd.choice([512, 4096]))
    return Scenario(model=mid, chip=CHIPS[rnd.choice(["100T", "1P"])], layout=lay, serving=sv, node_cards=4)


def test_energy_capacity_slc_work():
    rnd = random.Random(73)
    models = ["qwen3-8b", "qwen3-30b-a3b", "deepseek-v3", "qwen3-32b", "wan2.1-1.3b", "esm2-650m"]
    n = 0
    while n < 40:
        s = _rand_scn(rnd, models, pp=(1, 2))
        try:
            r = evaluate(s)
        except ValueError:
            continue
        n += 1
        case = (s.model, s.layout, s.serving.phase, s.serving.batch, s.chip.name)
        # hard capacity: fits ⇔ every stage's DRAM need ≤ capacity; SRAM / SLC partitions within capacity
        assert r.fits == all(st.mem.dram_need <= st.mem.dram_cap * (1 + 1e-9) for st in r.stages), case
        for st in r.stages:
            m = st.mem
            assert m.pinned_hot + m.pinned_expert + m.kv_sram + m.staging <= m.sram * (1 + 1e-9), case
            assert m.slc_hot + m.slc_expert + m.slc_kv <= m.slc * (1 + 1e-9) + 1, case
        # energy: J = Σ actions, linear per action, counts independent of the table, idle = cards × window
        e = energy_report(r, T)
        c, J = e["counts_per_unit"], e["J_by_action"]
        assert all(math.isfinite(v) and v >= 0 for v in c.values()), case
        assert abs(sum(J.values()) - e["J_per_unit"]) <= 1e-9 * max(1.0, e["J_per_unit"]), case
        assert abs(c["idle_card_s"] - s.layout.cards * e["window_s"] / e["units_per_window"]) <= 1e-9 * c["idle_card_s"], case
        e2 = energy_report(r, replace(T, pJ_bit_dram=10.0))
        assert e2["counts_per_unit"] == c and abs(e2["J_by_action"]["dram"] - 2 * J["dram"]) <= 1e-9 * J["dram"], case
        # SLC: never slower, never more DRAM bytes, fits unchanged
        for pol in ("pin", "lru"):
            r2 = evaluate(replace(s, chip=replace(s.chip, slc_mib=64.0, slc_policy=pol)))
            assert r2.step <= r.step * (1 + 1e-9) and r2.fits == r.fits, (case, pol)
            assert energy_report(r2, T)["counts_per_unit"]["dram"] <= c["dram"] * (1 + 1e-9), (case, pol)
        # schedule dependencies: a micro-batch crosses every stage in order; all micro-batches cross the bottleneck
        ts = [st.time.total for st in r.stages]
        assert r.step >= sum(ts) * (1 - 1e-9) and r.step >= r.microbatches * max(ts) * (1 - 1e-9), case
        cov = sorted(i for st in r.stages for i in range(*st.layers))
        assert cov == list(range(r.model.n_layers)), case
        # per-rank work conservation: request work independent of TP / PP (dense, LLM)
        if s.model not in MOE + DOM:
            assert abs(_work(r) / _work(evaluate(replace(s, layout=Layout()))) - 1) < 1e-6, case


def test_fabric_collective_properties():
    rnd = random.Random(7)
    n = 0
    while n < 600:
        total, nc = rnd.choice([8, 16, 64, 256]), rnd.choice([4, 8])
        fab = Fabric(enabled=True, oversub=rnd.choice([1.0, 4.0]), algo=rnd.choice(["auto", "ring", "tree", "hier", "auto_overlap"]),
                     net_topology=rnd.choice(["fat_tree", "rail"]), innet_reduce=rnd.choice(["off", "net", "net+link"]))
        s = Scenario(model="qwen3-8b", node_cards=nc, fabric=fab)
        mk = lambda f: System(s.chip, s.mem_id, s.mem_eff, s.link, s.d2d_link, s.package_eff, s.net, s.node_cards, f, total)
        kind = rnd.choice(["allreduce", "allgather", "alltoall"])
        g = rnd.choice([d for d in (2, 4, 8, 16, 64, 256) if d <= total])
        p = rnd.choice([1e3, 1e6, 1e9])
        bw, a, d2d, net = F.collective(kind, p, g, mk(fab))
        bw2, a2, _, _ = F.collective(kind, 2 * p, g, mk(fab))
        bw3, a3, _, _ = F.collective(kind, p, g, mk(replace(fab, oversub=fab.oversub * 2)))
        case = (kind, g, total, nc, fab, p)
        assert all(math.isfinite(x) and x >= 0 for x in (bw, a)) and 0 <= d2d <= 1 + 1e-9 and 0 <= net <= 1 + 1e-9, case
        assert bw2 + a2 >= (bw + a) * (1 - 1e-9) and bw2 <= 2 * bw * (1 + 1e-6) + 1e-15, case
        assert bw3 + a3 >= (bw + a) * (1 - 1e-9), case
        fastest = max(x.GBps for x in (s.link, s.d2d_link, s.net) if x is not None) * 1e9
        vol = {"allreduce": 2 * (g - 1) / g, "allgather": g - 1, "alltoall": (g - 1) / g}[kind]
        assert bw >= vol * p / fastest * (0.5 if fab.innet_reduce != "off" and kind == "allreduce" else 1) * (1 - 1e-6), case
        n += 1


def test_video_protein_schedules():
    rnd = random.Random(11)
    n = 0
    while n < 40:
        if rnd.random() < 0.6:
            mid = rnd.choice(["wan2.1-1.3b", "cogvideox-2b", "ltx-video", "hunyuanvideo", "mochi-1", "opensora-stdit3"])
            w = Workload(frames=rnd.choice([0, 9, 17]), steps=rnd.choice([0, 4]), te_cpu=rnd.choice([False, True]),
                         overlap=rnd.choice([False, True]), placement=rnd.choice(["auto", "resident", "offload", "shard"]),
                         vae_tiling=rnd.choice([False, True]), vae_parallel=rnd.choice([False, True]))
        else:
            mid = rnd.choice(["esm2-650m", "esmfold", "openfold", "boltz-1"])
            w = Workload(seq_len=rnd.choice([0, 128, 384]), recycles=rnd.choice([0, 1, 3]), samples=rnd.choice([0, 1, 5]))
        lay = rnd.choice([Layout(), Layout(sp=2), Layout(dp=2)])
        b = rnd.choice([1, 2])
        try:
            s = Scenario(model=mid, chip=CHIPS[rnd.choice(["100T", "1P"])], workload=w, layout=lay, serving=Serving(batch=b))
            r = evaluate(s)
        except ValueError:
            continue
        n += 1
        case = (mid, w, lay, b)
        d = r.domain_summary()
        p = r.pipeline or {}
        te, dn, de = p.get("te_s", 0.0), p.get("denoise_s", 0.0), p.get("decode_s", 0.0)
        assert r.latency >= (te + dn + de) * (1 - 1e-9), case                  # encode → denoise → decode
        assert max(te, dn) * (1 - 1e-9) <= d["period_s"] <= r.latency * (1 + 1e-9), case
        if p:
            assert abs(dn - r.workload.steps * d["step_ms"] / 1e3) <= 1e-6 * dn, case
        r1 = evaluate(replace(s, serving=Serving(batch=1)))
        assert r.latency >= r1.latency * (1 - 1e-9), case
        assert abs(r._tflop_per_request() / r1._tflop_per_request() - 1) < 1e-6, case
        r2 = evaluate(replace(s, layout=Layout()))
        assert abs(r._tflop_per_request() / r2._tflop_per_request() - 1) < 1e-6, case   # work over SP / DP


def test_search_never_worse_than_known_points():
    rnd = random.Random(5)
    for _ in range(6):
        mid = rnd.choice(["qwen3-8b", "llama-3.2-3b", "qwen3-30b-a3b", "esm2-650m"])
        chip = CHIPS[rnd.choice(["100T", "1P", "H100-like"])]
        cards = rnd.choice([2, 4])
        s = Scenario(model=mid, chip=chip, node_cards=8, serving=Serving(batch=1, tpot_slo_ms=rnd.choice([20, 100])))
        best = max((x.per_card for x in search_layouts(s, cards, top=rnd.choice([None, 1]))), default=0.0)
        m = get_model(mid)
        lays = list(enumerate_layouts(cards, m.n_layers, m.is_moe, full=not m.kv_cache, pair=m.is_pair))
        for _ in range(4):
            x = evaluate(s.replace("layout", rnd.choice(lays)).replace("serving.batch", rnd.choice([1, 4, 16, 64])))
            assert not _feasible(x) or x.per_card <= best * (1 + 1e-9), (mid, chip.name, cards)
        s1 = s.replace("layout", rnd.choice(lays))
        bb, (bf, val) = best_batch(s1, 128), brute_force_batch(s1, 128)
        assert (bb.result.throughput if bb.result is not None else 0.0) >= val * (1 - 1e-9), (mid, chip.name)


def test_length_bins_keep_the_mean():
    rnd = random.Random(3)
    for _ in range(3000):
        mean = rnd.choice([3, 8, 64, 100, 512, 4096, 8192]) * rnd.choice([1, 1.37])
        cv = rnd.choice([0.1, 0.5, 0.7, 0.9, 2.0])
        v = [x for _, x in lognormal_bins(mean, cv)]
        assert v == sorted(v) and min(v) >= 1, (mean, cv)
        assert sum(v) == round(len(v) * mean), (mean, cv, sum(v) / len(v))


def test_pd_length_law_keeps_means():
    # 0.73 bug: per-bin rounding drifted the PD length-law means (out 64, cv 0.9 → 64.13), so the PD goodput was
    # not req/s × out_len; checked at the source (the full cv queue report costs minutes)
    from accel_dse.core.lengths import lengths_of
    from accel_dse.core.scenario import PDConfig
    for pr, out, pcv, ocv in ((8192, 64, 0.7, 0.9), (1024, 256, 0.5, 2.0), (4000, 1000, 1.0, 0.3)):
        L = lengths_of(PDConfig(enabled=True, prompt_cv=pcv, out_cv=ocv), Serving(prompt=pr, out_len=out))
        assert abs(L.mean_out - out) < 1e-9 and abs(L.mean_S - pr) < 1e-9, (pr, out, L.mean_S, L.mean_out)


def test_pd_fluid_and_energy_identities():
    from accel_dse.core.disagg import disagg_report
    from accel_dse.core.scenario import PDConfig
    s = Scenario(model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", node_cards=8,
                 serving=Serving(batch=32, prompt=1024, out_len=256),
                 pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2))
    r = disagg_report(s, energy=replace(T, idle_W_prefill=60.0))
    assert abs(r["goodput_per_card"] - r["req_s"] * 256 / r["cards"]) <= 1e-9 * r["goodput_per_card"]
    assert abs(r["req_s"] - min(r["cap_req_s"].values())) <= 1e-9 * r["req_s"]
    assert all(0 <= u <= 1 + 1e-9 for u in r["util"].values())
    assert r["best_split"]["goodput_per_card"] >= r["goodput_per_card"] * (1 - 1e-9)
    q = r["queue"]
    assert q["modes"]["pd"]["slo_rate_rps"] <= r["req_s"] * (1 + 1e-6)
    for e in q["energy"].values():
        assert abs(sum(e["J_by_action"].values()) - e["J_per_token"]) <= 1e-9 * e["J_per_token"]
