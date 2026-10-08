"""V0 — invariants and metamorphic relations of the v2 core."""
from __future__ import annotations

import itertools

from accel_dse.core.dtypes import FormatSupport
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIP_100T, Chip
from accel_dse.core.mapping import ORGS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
CASES = [
    Scenario(model="qwen3-8b", serving=Serving(batch=4, ctx=2048)),
    Scenario(model="qwen3-32b", mem_id=HBM, layout=Layout(tp=4), serving=Serving(batch=32, ctx=4096)),
    Scenario(model="deepseek-v3", mem_id=HBM, layout=Layout(tp=4, dp=2, ep=8), serving=Serving(batch=64, ctx=4096)),
    Scenario(model="qwen3-next-80b-a3b", mem_id=HBM, layout=Layout(tp=2, ep=2), serving=Serving(batch=16, ctx=8192)),
    Scenario(model="gpt-oss-120b", mem_id=HBM, layout=Layout(tp=2, ep=2), serving=Serving(batch=16, ctx=4096)),
]


def _t(s):
    return evaluate(s).tpot


def test_more_dram_bandwidth_never_slower():
    for s, org in itertools.product(CASES, ORGS):
        s = s.replace("mapping", org)
        assert _t(s.replace("mem_eff", 0.9)) <= _t(s.replace("mem_eff", 0.6)) + 1e-15


def test_more_sram_never_slower():
    for s, org in itertools.product(CASES, ORGS):
        s = s.replace("mapping", org)
        assert _t(s.replace("chip.sram_mib", 512.0)) <= _t(s) + 1e-15


def test_longer_context_never_faster():
    for s in CASES:
        assert _t(s.replace("serving.ctx", s.serving.ctx * 2)) >= _t(s) - 1e-15


def test_faster_link_and_lower_alpha_never_slower():
    for s in CASES:
        assert _t(s.replace("link.GBps", 1600.0)) <= _t(s) + 1e-15
        assert _t(s.replace("link.alpha_us", 1.0)) <= _t(s) + 1e-15


def test_wider_sram_port_never_slower():
    for s, org in itertools.product(CASES, ORGS):
        s = s.replace("mapping", org)
        assert _t(s.replace("chip.sram_port_Bpc", 4 * s.chip.port_Bpc)) <= _t(s) + 1e-15


def test_reconf_never_slower_than_single_dataflow():
    for s in CASES:
        r = _t(s.replace("mapping", "reconf"))
        for org in ("os", "ws_edge", "ws_broad", "os_vec"):
            assert r <= _t(s.replace("mapping", org)) + 1e-15


def test_removing_native_fp8_never_faster_for_fp8_release():
    for mid in ("deepseek-v3", "qwen3-32b-fp8"):
        s = Scenario(model=mid, mem_id=HBM, layout=Layout(tp=8, ep=8) if mid.startswith("deepseek") else Layout(tp=2),
                     serving=Serving(batch=64, ctx=2048), mapping="reconf")
        no8 = s.replace("chip.formats", FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0))))
        assert _t(no8) >= _t(s)


def test_single_card_is_default_layout():
    s = Scenario(model="qwen3-8b")
    assert s.layout == Layout() and evaluate(s).scenario.layout.cards == 1


def test_results_finite_and_positive():
    for s, org in itertools.product(CASES, ORGS):
        r = evaluate(s.replace("mapping", org))
        for v in (r.tpot, r.step, r.throughput, r.tick):
            assert v > 0 and v < float("inf")


def test_what_if_override_labelled():
    s = Scenario(model="qwen3-32b", formats_override=(("mlp", "int4"), ("attn", "int4")))
    r = evaluate(s)
    assert r.model.what_if and r.model.fmt("mlp").fmt == "int4"
    assert r.tpot < evaluate(Scenario(model="qwen3-32b")).tpot
