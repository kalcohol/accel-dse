"""0.61.4 — round-4 audit fixes (DP idle-rank energy, PD prefill-pool budget / divisibility, swap host-link energy,
mixed fp8×int8 formats, Pareto beyond batch 512 / NaN, context-length and PP-imbalance warnings)."""

from __future__ import annotations

import math

from accel_dse import api
from accel_dse.core.dtypes import FormatSupport, gemm_exec
from accel_dse.core.search import pareto


def _eval(scn, **kw):
    return api.api_eval({"scenario": scn, **kw})


def _mac_tflop(r):
    return r["energy"]["counts_per_unit"]["mac"] * 2 / 1e12


def test_dp_idle_ranks_do_not_add_energy_or_tflop():
    """batch < dp (or not a multiple) leaves ranks idle; they must not be charged MACs / TFLOP."""
    for mid in ("esmfold", "boltz-1", "wan2.1-1.3b"):
        ref = None
        for b in (1, 3, 5):
            for dp in (1, 2, 4):
                r = _eval({"model": mid, "layout": {"dp": dp}, "serving": {"batch": b}})
                v = (_mac_tflop(r), r["summary"]["gen"]["tflop_per_request"])
                ref = ref or v
                assert abs(v[0] / ref[0] - 1) < 1e-9 and abs(v[1] / ref[1] - 1) < 1e-9, (mid, b, dp, v, ref)


def test_batch_invariant_work_per_unit_video_protein():
    for mid in ("wan2.1-1.3b", "esmfold", "alphafold2", "boltz-1"):
        vals = set()
        for b, lay, mb in ((1, {}, None), (3, {}, None), (4, {"pp": 2}, 2)):
            sv = {"batch": b, **({"microbatches": mb} if mb else {})}
            r = _eval({"model": mid, "layout": lay, "serving": sv})
            vals.add(round(_mac_tflop(r), 6))
        assert len(vals) == 1, (mid, vals)


def _pd(**pd):
    return {"model": "qwen3-8b", "serving": {"batch": 16, "prompt": 2048, "out_len": 256},
            "pd": {"enabled": True, **pd}}


def test_pd_prefill_layout_not_dividing_decode_cards():
    """Prefill TP4 with a 2-card decode pool of TP1 replicas used to raise 'decode_cards % layout.cards'."""
    r = _eval({**_pd(prefill_layout={"tp": 4}, prefill_cards=4, decode_cards=2), "layout": {"tp": 1}})
    assert "error" not in r["pd"], r["pd"]
    assert math.isfinite(r["summary"]["tpot_ms"])


def test_pd_prefill_pool_budget():
    scn = {**_pd(prefill_layout={"tp": 4}, prefill_cards=4, decode_cards=2), "layout": {"tp": 1}}
    r = _eval(scn, budget={"cards": 2})
    assert "pd" in r and "budget" in r["pd"]
    assert r["pd"]["budget"]["ok"] is False
    assert r["budget"]["ok_with_pd"] is False
    assert "prefill" in r["budget"]["scope_note"]
    r2 = _eval(scn, budget={"cards": 64})
    assert r2["pd"]["budget"]["ok"] is True and r2["budget"]["ok_with_pd"] is r2["budget"]["ok"]


def test_swap_host_link_energy_needs_user_value():
    """kv_policy=swap moves the preempted KV out and back over the host link: charged only with a user pJ_bit_host."""
    from dataclasses import replace
    from accel_dse.core.disagg import disagg_report
    from accel_dse.core.energy import EnergyTable
    from accel_dse.core.scenario import PDConfig, Scenario, Serving
    T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=5.0, pJ_bit_link=10.0, idle_W=100.0)
    assert T.pJ_bit_host is None
    scn = Scenario(model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", serving=Serving(batch=64, prompt=4096, out_len=512),
                   node_cards=8, pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2, kv_policy="swap",
                                             kv_capacity_GB=8.0, load=0.5, prompt_cv=0.7, out_cv=0.9))
    q0 = disagg_report(scn, energy=T)["queue"]["energy"]
    q1 = disagg_report(scn, energy=replace(T, pJ_bit_host=20.0))["queue"]["energy"]
    hit = [m for m, e in q0.items() if e.get("host_note")]
    assert hit, q0
    for m in hit:
        assert "host" not in q0[m]["J_by_action"]
        assert q1[m]["J_by_action"]["host"] > 0 and not q1[m].get("host_note")
        assert q1[m]["J_per_token"] > q0[m]["J_per_token"]


def test_mixed_fp8_int8_upcasts():
    sup = FormatSupport((("bf16", 1.0), ("fp16", 1.0), ("fp8", 2.0), ("int8", 2.0)))
    assert gemm_exec("fp8", "fp8", sup)[2] == "none"
    ex, rate, conv, _ = gemm_exec("int8", "fp8", sup)
    assert conv != "none" and rate <= 1.0, (ex, rate, conv)
    ex, rate, conv, _ = gemm_exec("fp16", "bf16", sup)
    assert conv != "none", (ex, rate, conv)


def test_pareto_drops_nan_and_extends_past_512():
    pts = [(1.0, 1.0, "a"), (float("nan"), 5.0, "n"), (2.0, float("inf"), "i"), (2.0, 3.0, "b"), (3.0, 2.0, "c")]
    front = pareto(pts)
    assert [p[2] for p in front] == ["a", "b"]
    from accel_dse.core.scenario import Scenario
    from accel_dse.core.parallel import Layout
    from accel_dse.core.search import tpot_throughput_front
    f = tpot_throughput_front(Scenario(model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", layout=Layout(tp=8)))
    assert max(p["batch"] for p in f) > 512


def test_context_beyond_max_position_warns():
    w = lambda ctx: [x for x in _eval({"model": "qwen3-8b", "serving": {"ctx": ctx, "phase": "decode"}})
                     ["summary"]["warnings"] if "max_position_embeddings" in x]
    assert w(65536) and not w(32768)


def test_pp_imbalance_warning_only_for_heterogeneous_stacks():
    w = lambda mid: [x for x in _eval({"model": mid, "layout": {"pp": 2}})["summary"]["warnings"] if "流水级按层数" in x]
    assert w("esmfold") and not w("qwen3-8b")
