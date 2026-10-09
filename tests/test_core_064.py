"""0.64 — closing the 0.63 "not done" list: SLO-aware prefill batch cap (PD / prefill-first) with the DES SLO search
on the same scan + refine, 2-D blocking in the activation-spill model, fractional running batch in the closed form,
weight dequant under uneven TP, DRAM bytes of idle DP ranks under EP.  Plus extended invariant property tests."""

from __future__ import annotations

import math
import random

from accel_dse import api
from accel_dse.core import pdqueue as pq
from accel_dse.core import pdsim
from accel_dse.core.catalog import get_model
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Op, Phase, Shard, build_rank_ops
from accel_dse.core.memplan import act_stream, gemm_blocking
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
MiB = 2 ** 20
_QN = {"model": "qwen3-next-80b-a3b", "mem_id": HBM, "layout": {"tp": 2, "ep": 2},
       "serving": {"batch": 16, "prompt": 1024, "out_len": 512, "ttft_slo_ms": 80, "tpot_slo_ms": 100},
       "pd": {"enabled": True, "prefill_layout": {"tp": 1}, "prefix_hit": 0.0, "load": 0.6}}


def _ctx(body):
    s = api.scenario_from_body({"scenario": body, "chip_preset": "1P"})
    return pdsim.capture_ctx(s)


# ------------------------------------------------------------------ 1. SLO-aware prefill batch cap
def test_slo_aware_cap_meets_slo_when_some_cap_does():
    """Property: at every λ, if any fixed cap meets both SLOs the mode's result meets them (and is stable whenever
    the default is); the chosen cap has the smallest mean TTFT among the SLO-meeting caps."""
    rep, ctx = _ctx(_QN)
    slo = ctx["slo"]
    bases = {"pd": lambda c, l: pq._kv_wrap(pq._pd_mode_base, c, l, "dpool", "r_d"),
             "coloc_prefill_first": lambda c, l: pq._kv_wrap(pq._coloc_prefill_first_base, c, l, "cpool", "r_c")}
    modes = {"pd": pq._pd_mode, "coloc_prefill_first": pq._coloc_prefill_first}
    picked = 0
    for name, base in bases.items():
        lam_s = rep["queue"]["modes"][name]["stable_rate_rps"]
        for i in range(1, 13):
            lam = lam_s * i / 12
            x = modes[name](ctx, lam)
            per = [base({**ctx, "_cap": b}, lam) for b in pq.B_CAPS]
            ok = [p for p in per if pq._meets(p, slo)]
            if ok and (base(ctx, lam)["stable"] or name != "pd"):
                assert pq._meets(x, slo), (name, lam)
                assert x["ttft"]["mean"] <= min(p["ttft"]["mean"] for p in ok) * (1 + 1e-9), (name, lam)
            if x.get("prefill", {}).get("cap_rule") == "slo":
                picked += 1
    assert picked > 0


def test_slo_aware_cap_raises_pd_slo_rate():
    rep, _ = _ctx(_QN)
    m = rep["queue"]["modes"]["pd"]
    assert m["slo_rate_rps"] > 2.5, m["slo_rate_rps"]                   # 0.63: 1.580 (cap 1 only meets 80 ms below)
    without = dict(_QN, serving=dict(_QN["serving"], ttft_slo_ms=10 ** 6, tpot_slo_ms=10 ** 6))
    r2, _ = _ctx(without)
    assert "cap_rule" not in (r2["queue"]["modes"]["pd"].get("prefill") or {})    # no SLO → 0.63's min-mean cap


def test_slo_rate_not_below_any_feasible_point_property():
    """Search ≥ any feasible grid point (the scan + refine is monotone-safe under the union-of-caps predicate)."""
    rep, ctx = _ctx(dict(_QN, pd=dict(_QN["pd"], prefix_hit=0.4)))
    slo = ctx["slo"]
    for name, fn in (("pd", pq._pd_mode), ("coloc_prefill_first", pq._coloc_prefill_first)):
        q = rep["queue"]["modes"][name]
        lam_s = q["stable_rate_rps"]
        feas = [lam_s * i / 40 for i in range(1, 41) if pq._meets(fn(ctx, lam_s * i / 40), slo)]
        assert q["slo_rate_rps"] >= max(feas, default=0.0) * (1 - 1e-9), (name, q["slo_rate_rps"], feas[-3:])


def test_des_slo_rate_scan_refine():
    _, ctx = _ctx(_QN)
    r = pdsim.slo_rate(ctx, "pd", 0.08, 0.1, 1.0, n_req=400, warmup=80)
    assert r > 1.0
    x = pdsim.simulate(ctx, r, "pd", n_req=400, warmup=80)
    assert x["complete"] and x["ttft"]["p90"] <= 0.08 and x["tpot"]["p90"] <= 0.1


# ------------------------------------------------------------------ 2. 2-D blocking
def test_gemm_blocking_hand_calc():
    # 16384×4096 · 4096×12288 bf16, 32 MiB: weight-stationary bn = 32 Mi / (4096·2) = 4096 → A read ⌈12288/4096⌉ =
    # 3×, W once: 3·128 + 96 = 480 MiB (activation-stationary bm 4096: 128 + 4·96 = 512; output-stationary bm 2048,
    # bn 4096: 3·128 + 8·96 = 1152).  0.63: min(A + C + W·⌈512/32⌉, W + A·⌈96/32⌉ + C) → A ×3 as well.
    m, k, n = 16384, 4096, 12288
    assert gemm_blocking(m, k, n, m * k * 2.0, k * n * 2.0, 32 * MiB) == (3, 1)
    # square 8192³ bf16, 4 MiB: output-stationary bm 1024 → bn = 4 Mi / (1024·4) = 1024 → (8, 8): 2048 MiB, vs
    # weight-stationary bn 256 → 32·128 + 128 = 4224 MiB (0.63's best one-sided choice)
    s = 8192
    assert gemm_blocking(s, s, s, s * s * 2.0, s * s * 2.0, 4 * MiB) == (8, 8)
    op = Op("ffn.up", "gemm", 0, m=m, k=k, n=n, w_params=k * n, w_bits=16, w_fmt="bf16", stream=True)
    a, wx = act_stream(op, 2.0, 64 * MiB)                 # budget = SRAM / 2
    assert abs(a - (3 * 128 + 384) * MiB) < 1 and wx == 0.0   # A ×3 + C once = 768 MiB


def _one_sided_063(ai, ao, w, budget):
    act_chunked = ai + ao + w * math.ceil((ai + ao) / budget)
    w_chunked = w + ai * math.ceil(w / budget) + ao
    return min(act_chunked, w_chunked)


def test_gemm_blocking_monotone_and_bounded_property():
    rng = random.Random(64)
    for _ in range(300):
        m, k, n = (rng.choice((1, 7, 64, 513, 2048, 16384)) for _ in range(3))
        ea, ew = rng.choice((1.0, 2.0)), rng.choice((0.5, 1.0, 2.0))
        A, W = m * k * ea, k * n * ew
        prev = math.inf
        for mib in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 512):
            ra, rw = gemm_blocking(m, k, n, A, W, mib * MiB)
            t = A * ra + W * rw
            assert ra >= 1 and rw >= 1 and t >= A + W - 1e-6
            assert t <= prev * (1 + 1e-12), (m, k, n, mib, t, prev)
            C = m * n * ea
            assert t + C <= _one_sided_063(A, C, W, mib * MiB) * (1 + 1e-12)     # never above 0.63's choice
            prev = t


def test_spill_monotone_in_sram():
    m = get_model("qwen3-8b")
    ops = [o for o in build_rank_ops(m, 3, Phase("prefill", 1, 16384, 0)) if o.stream]
    prev = math.inf
    for mib in (4, 8, 16, 32, 64, 128, 256):
        tot = sum(sum(act_stream(o, 2.0, mib * MiB)) for o in ops)
        assert tot <= prev * (1 + 1e-12), (mib, tot, prev)
        prev = tot


# ------------------------------------------------------------------ 3a. fractional running batch
def test_running_batch_is_fractional_and_continuous():
    _, ctx = _ctx(dict(_QN, serving=dict(_QN["serving"], ttft_slo_ms=10 ** 6, tpot_slo_ms=10 ** 6)))
    xs = [pq._coloc_prefill_first(ctx, 0.5 + 0.02 * i) for i in range(30)]
    ks = [x["decode"]["running_batch"] for x in xs]
    assert any(abs(k - round(k)) > 1e-3 for k in ks)
    dec = xs[10]["_dec"]
    lo, hi = math.floor(dec["seen_mean"]), math.floor(dec["seen_mean"]) + 1
    s_lo, s_hi = dec["step_of"](lo)[0], dec["step_of"](hi)[0]
    assert min(s_lo, s_hi) - 1e-15 <= dec["step"] <= max(s_lo, s_hi) + 1e-15
    assert abs(sum(w for w, _ in dec["mix"]) - 1) < 1e-12
    # TTFT mean moves smoothly with λ (0.63: jumps where round(n̄) stepped)
    t = [x["ttft"]["mean"] for x in xs]
    d = [b - a for a, b in zip(t, t[1:])]
    assert max(d) < 4 * (sum(d) / len(d)) + 1e-6, d


# ------------------------------------------------------------------ 3b / 4. split invariance of energy work
def test_dequant_work_invariant_to_uneven_tp():
    ref = None
    for tp in (1, 2, 3, 6):
        r = evaluate(Scenario(model="qwen3-32b-awq", mem_id=HBM, layout=Layout(tp=tp), serving=Serving(batch=4)))
        e = energy_report(r)["counts_per_unit"]
        ref = ref or e
        assert abs(e["vec"] / ref["vec"] - 1) < 1e-9 and abs(e["mac"] / ref["mac"] - 1) < 1e-9, (tp, e["vec"], ref["vec"])


def test_weight_share_property():
    """Σ over the TP ranks of the weight elements converted = the unsharded weights × copies (mean-rank share)."""
    rng = random.Random(7)
    for mid in ("qwen3-8b", "qwen3-32b-awq", "qwen3-30b-a3b", "deepseek-v3"):
        m = get_model(mid)
        for _ in range(4):
            tp = rng.choice((2, 3, 5, 6))
            sh = Shard(tp=tp, ep=tp) if m.is_moe else Shard(tp=tp)
            for o in build_rank_ops(m, rng.randrange(m.n_layers), Phase("decode", 4, 1, 2048), sh):
                assert 0 < o.wshare <= 1.0
                if o.w_params and o.wshare < 1:
                    assert o.wshare >= 0.5, (mid, tp, o.name, o.wshare)


# ------------------------------------------------------------------ 3c. idle DP ranks under EP
def test_idle_ep_ranks_move_only_expert_bytes():
    per_tok = []
    for b in (1, 2, 3, 4):
        r = evaluate(Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(dp=4, ep=4), serving=Serving(batch=b)))
        st = r.stages[0]
        assert 0 < st.dram["exp_dram"] < st.time.dram_bytes
        per_tok.append(energy_report(r)["counts_per_unit"]["dram"])
    # few expert hits per rank → expert bytes ∝ tokens: the per-token DRAM is flat for b ≤ dp (0.63: 14.8 → 6.4 GB
    # per token from b = 1 to 4 — idle ranks were charged the busy rank's attention / dense / KV bytes)
    assert max(per_tok) / min(per_tok) - 1 < 1e-6, per_tok
    a = energy_report(evaluate(Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(dp=4, ep=4),
                                        serving=Serving(batch=8))))["counts_per_unit"]["dram"]
    assert a < per_tok[0]
