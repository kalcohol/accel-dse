"""0.63 — external review of 0.62.1: MLA absorbed decode GEMMs, CFG step dependency, SLO-rate search, radix path
capacity, uneven TP heads, per-pair PP routes, work conservation (padding is time, not work), DES speculative tokens,
per-context fabric collector, LLM activation streaming.  Plus the permanent property tests the review asked for:
work conservation, schedule legality, hard capacity constraints, search ≥ any known feasible point."""

from __future__ import annotations
from dataclasses import replace

import dataclasses
import itertools
import math
import random
import threading

from accel_dse import api
from accel_dse.core import evaluate as ev
from accel_dse.core import fabric
from accel_dse.core.catalog import get_model
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import _cfg_chain, evaluate
from accel_dse.core.hardware import CHIPS, Fabric, System
from accel_dse.core.ir import Phase, PairDims, Shard, build_rank_ops, full_io_ops, gqa_local, head_ops
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.parallel import Layout
from accel_dse.core.prefixcache import tree_depth_tail, tree_groups, tree_resident
from accel_dse.core.scenario import Scenario, Serving, Workload

HBM = "hbm3e_8s_12h24g_9200"


# ------------------------------------------------------------------ 1. MLA absorbed decode
def test_mla_decode_kv_b_per_head_gemms():
    m = get_model("deepseek-v3")
    li = 5
    ops = {o.name: o for o in build_rank_ops(m, li, Phase("decode", 1, 1, 4096))}
    assert "attn.kv_b" not in ops
    uk, uv = ops["attn.kv_b.uk"], ops["attn.kv_b.uv"]
    assert (uk.m, uk.k, uk.n, uk.count) == (1, 128, 512, 128) and (uv.m, uv.k, uv.n, uv.count) == (1, 512, 128, 128)
    pre = {o.name: o for o in build_rank_ops(m, li, Phase("prefill", 1, 64, 0))}["attn.kv_b"]
    assert uk.w_params + uv.w_params == pre.w_params == 512 * 128 * 256
    assert abs(uk.flops + uv.flops - 2 * 512 * 128 * 256) < 1e-6
    ch = replace(CHIPS["100T"], instance_sched="wide")  # 0.71: hand-checked numbers are for the wide schedule (auto default changes them)
    kw = dict(w_fmt=uk.w_fmt, a_fmt=uk.a_fmt)
    c = gemm_cost(ch, "os", 1, 128, 512, count=128, **kw).cycles + gemm_cost(ch, "os", 1, 512, 128, count=128, **kw).cycles
    assert c == 40960 and gemm_cost(ch, "os", 1, 512, 32768, **kw).cycles == 9472
    # TP4: 32 heads per rank, params / FLOPs conserved over the ranks
    t4 = {o.name: o for o in build_rank_ops(m, li, Phase("decode", 1, 1, 4096), Shard(tp=4, ep=4))}
    assert t4["attn.kv_b.uk"].count == 32 and 4 * (t4["attn.kv_b.uk"].w_params + t4["attn.kv_b.uv"].w_params) == pre.w_params


# ------------------------------------------------------------------ 2. CFG dependency
def test_cfg_forwards_of_a_request_are_one_dependent_chain():
    assert [_cfg_chain(s, mb, c) for s, mb, c in ((2, 2, 2), (4, 2, 2), (4, 4, 2), (6, 4, 2), (3, 2, 1), (6, 3, 3), (6, 4, 3))] == \
        [2, 1, 2, 1, 1, 3, 3]
    wl = Workload(steps=1, pipeline=False)
    r1 = evaluate(Scenario(model="wan2.1-t2v-1.3b", layout=Layout(pp=1), workload=wl))
    r2 = evaluate(Scenario(model="wan2.1-t2v-1.3b", layout=Layout(pp=2), workload=wl))
    assert abs(r2.step / r2.tick - 3.0) < 1e-9          # cond → uncond → latent update: (2 + 2 − 1) ticks, was 2
    assert r2.step > 1.4 * r1.step / 2                  # CFG 2 is not two independent requests
    rb = evaluate(Scenario(model="wan2.1-t2v-1.3b", layout=Layout(pp=2), serving=Serving(batch=2), workload=wl))
    assert abs(rb.step / rb.tick - 2.0) < 1e-9          # each request's CFG pair in one micro-batch: max(mb, pp)


def test_schedule_legality_video_denoise():
    """Property: per denoise step ≥ the stages' busy ticks (mb) and ≥ one chain's critical path (k + pp − 1)."""
    for pp, b, mb in itertools.product((1, 2, 4), (1, 2, 3), (0, 1, 2, 4)):
        s = Scenario(model="wan2.1-t2v-1.3b", layout=Layout(pp=pp), workload=Workload(steps=2, pipeline=False),
                     serving=Serving(batch=b, microbatches=mb))
        try:
            r = evaluate(s)
        except ValueError:
            continue
        seqs = b * 2
        k = _cfg_chain(seqs, r.microbatches, 2)
        per = r.step / 2 / r.tick
        assert per >= r.microbatches - 1e-9 and per >= k + pp - 1 - 1e-9 * (k > 1), (pp, b, mb, per, k)


# ------------------------------------------------------------------ 3. SLO rate ≥ any known feasible point
_PD = {"model": "qwen3-next-80b-a3b", "mem_id": HBM, "layout": {"tp": 2, "ep": 2},
       "serving": {"batch": 16, "prompt": 1024, "out_len": 512, "ttft_slo_ms": 85, "tpot_slo_ms": 100},
       "pd": {"enabled": True, "prefill_layout": {"tp": 1}, "prefix_hit": 0.4}}


def _feasible_scan(scn_body, n=120):
    from accel_dse.core import pdqueue as pq
    from accel_dse.core.pdsim import capture_ctx
    s = api.scenario_from_body({"scenario": scn_body, "chip_preset": "1P"})
    rep, ctx = capture_ctx(s)
    q = rep["queue"]
    sv = s.serving
    out = {}
    for name, fn in (("pd", pq._pd_mode), ("coloc_prefill_first", pq._coloc_prefill_first), ("coloc_chunked", pq._coloc_chunked)):
        if name not in q["modes"]:
            continue
        lam_s = q["modes"][name]["stable_rate_rps"]
        feas = [lam_s * i / n for i in range(1, n + 1)
                if (lambda x: x["stable"] and x["ttft"]["p90"] <= sv.ttft_slo_ms / 1e3 and x["tpot_p90"] <= sv.tpot_slo_ms / 1e3)(fn(ctx, lam_s * i / n))]
        out[name] = (q["modes"][name]["slo_rate_rps"], max(feas, default=0.0))
    return out


def test_slo_rate_not_below_feasible_points():
    res = _feasible_scan(_PD)
    rate, feas = res["pd"]
    # 0.63: was 1.724 with 2.834 feasible; 0.65 exact bulk queue: feasible ~1.735 (DES ~1.60, fluid 2.834 was optimistic)
    # Unreleased: TTFT SLO 80 → 85 ms (chunked GDN prefill +5.5 %); feasible ~1.47
    assert feas >= 1.4 and rate >= feas * (1 - 1e-9), res
    for body in (dict(_PD, model="qwen3-8b", layout={"tp": 2}), dict(_PD, serving=dict(_PD["serving"], ttft_slo_ms=300))):
        for name, (rate, feas) in _feasible_scan(body, 60).items():
            assert rate >= feas * (1 - 1e-9), (body["model"], name, rate, feas)


# ------------------------------------------------------------------ 4. radix capacity
def test_radix_path_must_fit_with_its_ancestors():
    from accel_dse.core.pdsim import RadixLRU, _tree_paths
    lv = [(60, 1, 0.0), (60, 1, 0.0)]
    g = tree_groups(lv)
    assert tree_depth_tail(g, tree_resident(lv, 100, g)) == [1.0, 0.0]          # was [0.833, 0.833]
    c = RadixLRU(100, [60, 60])
    for _ in range(50):
        c.access((0, 0))
    assert c.depth_n[2] == 0 and c.used <= 100


def test_capacity_constraints_radix_property():
    """Property: tails non-increasing; a level whose cumulative path exceeds the capacity has no hits; analytic
    within a few points of the radix-LRU simulation."""
    from accel_dse.core.pdsim import RadixLRU, _tree_paths
    rng = random.Random(7)
    for _ in range(25):
        lv = [(rng.choice((64, 256, 1024, 4000)), rng.choice((1, 3, 8, 20)), rng.choice((0.0, 0.8, 1.2)))
              for _ in range(rng.choice((1, 2, 3)))]
        C = rng.choice((100, 1000, 5000, 20000, 80000))
        g = tree_groups(lv)
        tail = tree_depth_tail(g, tree_resident(lv, C, g))
        assert all(a >= b - 1e-12 for a, b in zip([1.0] + tail, tail))
        cum = 0
        for k, (L, _, _) in enumerate(lv):
            cum += L
            if cum > C:
                assert all(t == 0.0 for t in tail[k:]), (lv, C, tail)
                break


# ------------------------------------------------------------------ 5. uneven TP heads + work conservation
def test_uneven_tp_keeps_every_head():
    assert gqa_local(32, 8, 3) == (3, 12) and gqa_local(32, 8, 4) == (2, 8) and gqa_local(32, 8, 16) == (1, 2)
    assert gqa_local(64, 8, 12) == (1, 8) and gqa_local(28, 4, 3) == (2, 10) and gqa_local(40, 10, 8) == (2, 5)
    assert gqa_local(40, 10, 4) == (3, 10)
    for q, kv in ((32, 8), (40, 10), (64, 8), (28, 4), (64, 4), (48, 8)):     # every head on some rank
        for tp in range(1, 17):
            k_, h_ = gqa_local(q, kv, tp)
            assert h_ * tp >= q and k_ * tp >= kv and h_ >= -(-q // tp), (q, kv, tp)
    m = get_model("qwen3-8b")
    ph = Phase("decode", 8, 1, 2048)
    qk = next(o for o in build_rank_ops(m, 0, ph, Shard(tp=3)) if o.name == "qk")
    # busiest rank: groups whole (3 KV heads × 4) beats heads dealt (11 query heads touching 4 KV groups)
    assert qk.count == 8 * 3 and qk.m == 4
    k = next(o for o in build_rank_ops(m, 0, ph, Shard(tp=3)) if o.name == "attn.k")
    assert k.n == 3 * 128


def _conserved(m, ph, sh, ops_fn):
    loc, glob = ops_fn(sh), ops_fn(Shard())
    n = sh.tp * sh.dp * sh.sp
    f_loc = sum(o.flops * o.share / max(o.replicated, 1e-12) for o in loc if o.kind != "comm") * n
    # useful work: an op's own share < 1 only for padded expert rows (hit · m_e > token-expert pairs)
    f_glob = sum(o.flops * o.share for o in glob if o.kind != "comm")
    v_loc = sum(o.vec * o.share for o in loc if o.kind == "vector") * n
    v_glob = sum(o.vec * o.share for o in glob if o.kind == "vector")
    return f_loc, f_glob, v_loc, v_glob


def test_work_conservation_property():
    """Property: Σ over ranks of effective work (FLOPs, vector ops) = the unsharded model's, for any TP / DP / EP /
    SP split and uneven batch / heads / tokens; the busiest rank's ops (time) are separate."""
    rng = random.Random(3)
    cases = []
    for mid in ("qwen3-8b", "qwen3-30b-a3b", "deepseek-v3", "qwen3-next-80b-a3b", "glm-4.6"):
        m = get_model(mid)
        for _ in range(6):
            tp = rng.choice((1, 2, 3, 4, 6, 8))
            dp = rng.choice((1, 2, 3)) if m.is_moe else 1
            ep = tp * dp if m.is_moe else 1
            ph = Phase(rng.choice(("decode", "prefill")), rng.choice((1, 3, 7, 16)), 1, 1024)
            if ph.kind == "prefill":
                ph = Phase("prefill", ph.batch, rng.choice((100, 513)), rng.choice((0, 256)))
            li = rng.randrange(m.n_layers)
            cases.append((m, ph, Shard(tp=tp, dp=dp, ep=ep), lambda sh, m=m, li=li, ph=ph: build_rank_ops(m, li, ph, sh)))
            cases.append((m, ph, Shard(tp=tp, dp=dp, ep=ep), lambda sh, m=m, ph=ph: head_ops(m, ph, sh)))
    for mid in ("wan2.1-t2v-1.3b", "esm2-650m"):
        m = get_model(mid)
        w = ev.resolve_workload(m, Workload())
        for tp, sp, dp in ((1, 3, 1), (3, 1, 1), (2, 3, 2), (1, 1, 3)):
            if m.domain == "protein" and tp > 1:
                continue
            ph = Phase("full", rng.choice((1, 3)), w.tokens, w.ctx, frames=w.frames, aux=w.aux, pair=w.pair)
            cases.append((m, ph, Shard(tp=tp, sp=sp, dp=dp), lambda sh, m=m, ph=ph: build_rank_ops(m, 3, ph, sh)))
            cases.append((m, ph, Shard(tp=tp, sp=sp, dp=dp), lambda sh, m=m, ph=ph: full_io_ops(m, ph, sh, "pre")))
    for m, ph, sh, fn in cases:
        fl, fg, vl, vg = _conserved(m, ph, sh, fn)
        assert abs(fl - fg) <= 1e-9 * max(fg, 1.0), (m.id, ph, sh, fl, fg)
        assert abs(vl - vg) <= 1e-9 * max(vg, 1.0), (m.id, ph, sh, vl, vg)


def test_energy_work_invariant_to_split_and_padding():
    """MAC / vector counts per output unit do not depend on how the work is split or padded (no replicated ops)."""
    ref = None
    for lay, b, mb in ((Layout(), 3, 0), (Layout(tp=3), 3, 0), (Layout(pp=2), 3, 0), (Layout(pp=3), 3, 2),
                       (Layout(tp=2, pp=2), 3, 0), (Layout(tp=8), 3, 0)):
        r = evaluate(Scenario(model="qwen3-8b", mem_id=HBM, layout=lay, serving=Serving(batch=b, microbatches=mb)))
        e = energy_report(r)["counts_per_unit"]
        ref = ref or e
        assert abs(e["mac"] / ref["mac"] - 1) < 1e-9 and abs(e["vec"] / ref["vec"] - 1) < 1e-9, (lay.label, e["mac"], ref["mac"])
    a = energy_report(evaluate(Scenario(model="esm2-650m", serving=Serving(batch=3))))["counts_per_unit"]["mac"]
    b = energy_report(evaluate(Scenario(model="esm2-650m", layout=Layout(pp=2), serving=Serving(batch=3))))["counts_per_unit"]["mac"]
    assert abs(b / a - 1) < 1e-9                        # was +33 % (last micro-batch padding)


def test_vision_padding_is_not_work():
    base = {"model": "qwen3.8-27b", "mem_id": HBM, "serving": {"phase": "prefill", "batch": 1, "images": 3, "prompt": 4096}}
    r1 = _vis(dict(base, layout={"tp": 1}))
    r2 = _vis(dict(base, layout={"tp": 2}))
    assert abs(r2["tflop"] / r1["tflop"] - 1) < 1e-9 and r2["per_card"] == 2      # TP2: 2 images on the busiest card, 3 of work


def _vis(body):
    return api.api_eval({"scenario": body})["summary"]["vision"]


# ------------------------------------------------------------------ 6. PP hand-off over every rank pair
def test_pp_handoff_slowest_rank_pair():
    scn = Scenario(model="llama-3.3-70b", layout=Layout(pp=2, tp=64), node_cards=8, fabric=Fabric(enabled=True, oversub=4.0))
    sys_ = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                  scn.fabric, scn.layout.cards)
    bw, a, fd, fn = ev._p2p(sys_, scn.net.GBps * 1e9 * 1e-3, 0, 64)
    assert abs(bw - 4e-3) < 1e-12 and fn == 1.0         # rank 0 → 64 crosses the 4:1 leaf (was 1 ms via 63 → 64)
    # mixed tiers: 3 cards per stage in 4-card nodes → pairs 0→3 in-node, 1→4 / 2→5 cross; slowest = network
    s2 = Scenario(model="qwen3-8b", layout=Layout(pp=2, tp=3), node_cards=4)
    sy2 = System(s2.chip, s2.mem_id, s2.mem_eff, s2.link, s2.d2d_link, s2.package_eff, s2.net, s2.node_cards, s2.fabric, 6)
    bw2, _, _, fn2 = ev._p2p(sy2, 1e9, 0, 3)
    assert abs(fn2 - 2 / 3) < 1e-12 and abs(bw2 - 1e9 / (s2.net.GBps * 1e9)) < 1e-12


# ------------------------------------------------------------------ 8. DES speculative tokens
def test_des_spec_tokens_are_drawn_with_the_right_mean():
    from accel_dse.core.pdsim import Engine
    import random as _r

    class R:
        _spec = (3, 0.6)
        eng = Engine(_r.Random(1), 5)
    from accel_dse.core.pdsim import _Replica
    xs = [_Replica._draw(R) for _ in range(60000)]
    mean = sum(xs) / len(xs)
    exp = (1 - 0.6 ** 4) / (1 - 0.6)
    assert abs(mean / exp - 1) < 0.01 and set(xs) <= {1, 2, 3, 4}


# ------------------------------------------------------------------ 9. per-context fabric collector
def test_fabric_report_is_thread_safe():
    scn = Scenario(model="deepseek-v3", layout=Layout(dp=16, ep=16), node_cards=8, serving=Serving(batch=64),
                   fabric=Fabric(enabled=True, oversub=2.0))
    ref = fabric.fabric_report(scn)
    out, errs = [], []

    def run():
        try:
            for _ in range(3):
                out.append(fabric.fabric_report(scn))
                evaluate(dataclasses.replace(scn, serving=Serving(batch=32)))
        except Exception as e:      # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=run) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs and len(out) == 18
    key = lambda rep: [(r["kind"], r["group"], r["count"], r["algo"]) for r in rep["rows"]]
    assert all(key(r) == key(ref) for r in out)
    assert fabric.log() is None


# ------------------------------------------------------------------ 10. LLM activation streaming
def test_llm_prefill_activations_spill_with_small_sram():
    acts, prev = [], None
    for sram in (16, 64, 256, 4096):
        ch = dataclasses.replace(CHIPS["100T"], sram_mib=sram)
        r = evaluate(Scenario(model="qwen3-8b", chip=ch, serving=Serving(phase="prefill", batch=4, prompt=4096)))
        st = r.stages[0]
        assert st.mem.staging <= max(2 * 2**20, ch.sram_bytes / 2) + 1
        acts.append(st.dram.get("act", 0.0))
    # 0.64 (2-D blocking): a nest may trade activation re-reads for weight re-reads, so the activation part alone
    # is only non-increasing in SRAM (64 / 256 MiB: 96.6 GB both; total spill 143.8 → 96.6 GB)
    assert acts[0] > acts[1] >= acts[2] > acts[3] >= 0 and acts[1] > 1e10
    d = evaluate(Scenario(model="qwen3-8b", chip=dataclasses.replace(CHIPS["100T"], sram_mib=64),
                          serving=Serving(phase="decode", batch=8, ctx=4096)))
    assert d.stages[0].dram.get("act", 0.0) == 0.0            # decode activations stay on chip


def test_expert_padding_is_not_work():
    """Useful expert FLOPs = token-expert pairs × per-pair FLOPs, independent of the padded rows (hit · m_e)."""
    for mid in ("qwen3-30b-a3b", "mixtral-8x22b", "deepseek-v3"):
        m = get_model(mid)
        li = m.n_layers - 1
        f = m.layers[li].ffn
        for b in (1, 2, 3, 7):
            ph = Phase("decode", b, 1, 1024)
            ex = [o for o in build_rank_ops(m, li, ph) if o.name.startswith("expert.") and o.kind == "gemm"]
            useful = sum(o.flops * o.share for o in ex)
            per_pair = sum(o.flops / (o.count * o.m) for o in ex)
            assert abs(useful / (b * f.top_k * per_pair) - 1) < 1e-9, (mid, b)


def test_moe_energy_mac_invariant_to_split():
    """MoE MAC count per token does not depend on PP micro-batching / DP·EP (no replicated expert work)."""
    ref = None
    for lay in (Layout(), Layout(pp=2), Layout(dp=2, ep=2), Layout(pp=3)):
        r = evaluate(Scenario(model="mixtral-8x22b", mem_id=HBM, layout=lay, serving=Serving(batch=3)))
        mac = energy_report(r)["counts_per_unit"]["mac"]
        ref = ref or mac
        assert abs(mac / ref - 1) < 1e-9, (lay.label, mac, ref)
