"""L0/L6/L7 — scenario identity, strict input, exact batch search vs brute force, Pareto."""
from __future__ import annotations

import json
import math

from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.search import best_batch, brute_force_batch, pareto, search_layouts
from accel_dse.core.serving import goodput

HBM = "hbm3e_8s_12h24g_9200"


def test_identity_replace_same_value():
    s = Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(tp=4), serving=Serving(batch=16))
    for path, v in (("serving.batch", 16), ("chip.sram_mib", s.chip.sram_mib), ("mapping", "os"),
                    ("link.alpha_us", 3.0), ("layout.tp", 4)):
        t = s.replace(path, v)
        assert t == s and t.hash() == s.hash()
        assert evaluate(t).tpot == evaluate(s).tpot


def test_sweep_is_controlled_replacement():
    s = Scenario(model="qwen3-8b")
    base = s.to_dict()
    for b in (1, 2, 8):
        d = s.replace("serving.batch", b).to_dict()
        diff = {k for k in base if base[k] != d[k]}
        assert diff <= {"serving"}
        assert {k for k in base["serving"] if base["serving"][k] != d["serving"][k]} <= {"batch"}


def test_strict_json_round_trip_and_rejections():
    s = Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(tp=8, ep=8), mapping="reconf")
    d = json.loads(json.dumps(s.to_dict()))
    assert Scenario.from_dict(d) == s
    bad = [
        {**d, "bogus": 1},
        {**d, "serving": {**d["serving"], "batch": -1}},
        {**d, "serving": {**d["serving"], "spec_accept": 1.5}},
        {**d, "chip": {**d["chip"], "sram_mib": float("nan")}},
        {**d, "chip": {**d["chip"], "rows": float("inf")}},
        {**d, "mapping": "magic"},
        {**d, "layout": {**d["layout"], "pp": 0}},
        {**d, "serving": {**d["serving"], "tpot_slo_ms": float("inf")}},
    ]
    for b in bad:
        try:
            Scenario.from_dict(b)
            raise AssertionError(f"accepted {b}")
        except (ValueError, TypeError, KeyError):
            pass
    try:
        s.replace("chip.nope", 1)
        raise AssertionError("unknown path accepted")
    except KeyError:
        pass


def test_invalid_layout_for_model_rejected():
    try:
        evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=2, dp=2, ep=4)))
        raise AssertionError
    except ValueError:
        pass
    try:
        evaluate(Scenario(model="deepseek-v3", layout=Layout(tp=4, ep=2)))
        raise AssertionError
    except ValueError:
        pass


def test_exact_batch_search_matches_brute_force():
    cases = [
        Scenario(model="qwen3-8b", serving=Serving(ctx=1024, tpot_slo_ms=120)),
        Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(tp=4), mapping="reconf", serving=Serving(ctx=4096, tpot_slo_ms=50)),
        Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(pp=2, tp=4), mapping="os", serving=Serving(ctx=2048, tpot_slo_ms=40)),
        Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(tp=8, ep=8), mapping="ws_broad", serving=Serving(ctx=4096, tpot_slo_ms=50)),
        Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(tp=1, dp=2, ep=2), mapping="os_vec", serving=Serving(ctx=2048, tpot_slo_ms=30)),
        Scenario(model="gpt-oss-120b", mem_id=HBM, layout=Layout(tp=2, ep=2), mapping="reconf", serving=Serving(ctx=2048, tpot_slo_ms=25)),
    ]
    for scn in cases:
        bb = best_batch(scn)
        cap = max(bb.b_max + 8, 16)
        bf_b, bf_thr = brute_force_batch(scn, cap)
        assert bb.batch == bf_b, (scn.model, bb.batch, bf_b)
        if bf_b:
            assert abs(bb.result.throughput - bf_thr) < 1e-9 * bf_thr


def test_step_monotone_and_feasibility_monotone_bruteforce():
    """Certify (P1) step non-decreasing in batch, (P2) feasibility monotone, on a family sample."""
    for scn in (Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(tp=4, dp=2, ep=8), mapping="os"),
                Scenario(model="qwen3-next-80b-a3b", mem_id=HBM, layout=Layout(tp=2, ep=2), mapping="reconf"),
                Scenario(model="qwen3-32b-fp8", mem_id=HBM, layout=Layout(tp=2), mapping="ws_edge"),
                Scenario(model="gpt-oss-20b", mapping="os_vec")):
        prev_step, prev_feas = 0.0, True
        for b in range(1, 200):
            r = evaluate(scn.replace("serving.batch", b))
            assert r.step >= prev_step - 1e-15, (scn.model, b)
            feas = r.fits and r.tpot * 1e3 <= scn.serving.tpot_slo_ms
            assert not (feas and not prev_feas), (scn.model, b)
            prev_step, prev_feas = r.step, feas


def test_pareto_synthetic():
    pts = [(1, 1, "a"), (2, 3, "b"), (2, 2, "c"), (3, 3, "d"), (4, 5, "e"), (0.5, 0.2, "f")]
    assert [p[2] for p in pareto(pts)] == ["f", "a", "b", "e"]


def test_layout_search_returns_sorted_valid_rows():
    rows = search_layouts(Scenario(model="qwen3-32b", mem_id=HBM, serving=Serving(ctx=2048)), 4)
    assert rows and all(r.layout.cards == 4 for r in rows)
    vals = [r.per_card for r in rows]
    assert vals == sorted(vals, reverse=True)


def test_goodput_includes_prefill_amortisation():
    scn = Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(tp=4), mapping="reconf",
                   serving=Serving(batch=32, ctx=4096, prompt=4096, out_len=512))
    r = evaluate(scn)
    g = goodput(r)
    assert 0 < g.goodput_tok_s < r.throughput
    exp = 1 / (1 / g.decode_tok_s + (4096 / 512) / g.prefill_tok_s)
    assert abs(g.goodput_tok_s - exp) < 1e-9 * exp
    g2 = goodput(evaluate(scn.replace("serving.out_len", 4096)))
    assert g2.goodput_tok_s > g.goodput_tok_s          # longer outputs amortise prefill better


def test_goodput_flags_unmet_ttft_instead_of_zero():
    scn = Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(tp=1, dp=8, ep=8), mapping="reconf",
                   serving=Serving(batch=64, ctx=4096, prompt=4096, ttft_slo_ms=500.0))
    g = goodput(evaluate(scn))
    assert not g.ttft_ok and g.goodput_tok_s > 0 and g.ttft_ms > 500


def test_hash_ignores_int_vs_float_spelling():
    s = Scenario(model="qwen3-8b")
    t = s.replace("chip.sram_mib", 64).replace("link.GBps", 400)
    assert t == s and t.hash() == s.hash()
    assert Scenario.from_dict(json.loads(json.dumps(t.to_dict()))).hash() == s.hash()
