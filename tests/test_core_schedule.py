"""L5 — collectives, pipeline, speculative decode / MTP, LM head."""
from __future__ import annotations

from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import Link
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.schedule import collective_seconds, spec_expected_tokens

HBM = "hbm3e_8s_12h24g_9200"


def test_collective_alpha_beta():
    L = Link(GBps=100.0, alpha_us=3.0)
    bw, a = collective_seconds("allreduce", 1e6, 4, L)
    assert abs(bw - 2 * 3 / 4 * 1e6 / 1e11) < 1e-15 and a == 3e-6
    bw, a = collective_seconds("alltoall", 1e6, 8, L)
    assert abs(bw - 7 / 8 * 1e6 / 1e11) < 1e-15
    assert collective_seconds("allreduce", 1e6, 1, L) == (0.0, 0.0)


def test_spec_expected_tokens():
    assert spec_expected_tokens(0, 0.7) == 1.0
    assert abs(spec_expected_tokens(2, 0.7) - (1 - 0.7**3) / 0.3) < 1e-12
    assert spec_expected_tokens(3, 1.0) == 4.0
    assert spec_expected_tokens(3, 0.0) == 1.0


def test_pipeline_decode_and_prefill_formulas():
    base = Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(pp=4), serving=Serving(batch=16, ctx=2048))
    r = evaluate(base)
    assert r.microbatches == 4
    assert abs(r.step - max(4, 4) * r.tick) < 1e-15
    r2 = evaluate(base.replace("serving.microbatches", 2))
    assert abs(r2.step - 4 * r2.tick) < 1e-15            # mb < pp → bubbles: pp·tick
    p = evaluate(base.replace("serving.phase", "prefill").replace("serving.batch", 4).replace("serving.prompt", 1024))
    assert abs(p.ttft - (4 + 4 - 1) * p.tick) < 1e-15


def test_mtp_draft_runs_on_last_stage_only():
    base = Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(pp=8), serving=Serving(batch=8, ctx=2048))
    a = evaluate(base)
    b = evaluate(base.replace("serving.spec_k", 1))
    assert b.tokens_per_step > 1.0
    # verify (q=2) slows every stage a little; the MTP module adds weights/compute to the last stage only
    assert b.stages[-1].mem.stored_w > a.stages[-1].mem.stored_w
    assert all(abs(b.stages[i].mem.stored_w - a.stages[i].mem.stored_w) < 1 for i in range(7))


def test_spec_k_ignored_without_mtp():
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(batch=1, spec_k=2)))
    assert r.tokens_per_step == 1.0 and any("spec_k 已忽略" in w for w in r.warnings)


def test_sync_exposed_and_link_bound():
    base = Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(tp=8), serving=Serving(batch=1, ctx=512))
    a = evaluate(base)
    b = evaluate(base.replace("link.alpha_us", 10.0))
    n_coll = 2 * 64 + 1                                  # attn + mlp all-reduce per layer + logits gather
    assert abs((b.stages[0].time.t_sync - a.stages[0].time.t_sync) - n_coll * 7e-6) < 1e-9
