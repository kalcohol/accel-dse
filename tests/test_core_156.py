"""0.156 standing datapath / residency / non-LLM multi-card guards (randomized suite: val5/v0156/dpfuzz.py)."""
import dataclasses as dc

from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _chip(s, **kw):
    return s.replace("chip", dc.replace(s.chip, **kw))


def _t(r):
    return r.latency if r.domain != "llm" else r.tpot


def test_dequant_and_lanes_monotone():
    for m in ("qwen3-8b-awq", "gpt-oss-20b"):
        s = Scenario().replace("model", m).replace("serving.batch", 16)
        r = evaluate(s)
        d0 = s.chip.formats.dequant_elems_per_cycle or s.chip.lanes
        up = evaluate(_chip(s, formats=dc.replace(s.chip.formats, dequant_elems_per_cycle=d0 * 4)))
        dn = evaluate(_chip(s, formats=dc.replace(s.chip.formats, dequant_elems_per_cycle=max(1, d0 // 4))))
        assert _t(up) <= _t(r) * (1 + 1e-9) and _t(dn) >= _t(r) * (1 - 1e-9)
        slow = evaluate(_chip(s, formats=dc.replace(s.chip.formats, dequant_elems_per_cycle=1)))
        assert _t(slow) > _t(r) * 10                                       # dequant path is actually exercised
        assert _t(evaluate(_chip(s, vector_lanes=s.chip.lanes * 2))) <= _t(r) * (1 + 1e-9)


def test_sram_and_batch_residency():
    s = Scenario().replace("model", "llama-3.1-8b").replace("serving.batch", 16)
    r = evaluate(s)
    a = evaluate(_chip(s, sram_mib=s.chip.sram_mib * 2))
    assert _t(a) <= _t(r) * (1 + 1e-9) and a.stages[0].mem.pinned_hot >= r.stages[0].mem.pinned_hot
    b = evaluate(s.replace("serving.batch", 64))
    m0, m1 = r.stages[0].mem, b.stages[0].mem
    assert m1.stored_w == m0.stored_w and m1.pinned_hot <= m0.pinned_hot          # staging grows with batch
    assert b.stages[0].dram["weights"] / 64 <= r.stages[0].dram["weights"] / 16    # amortised per sequence


def test_non_llm_multicard():
    for m in ("wan2.1-1.3b", "esm2-650m", "carbon-a-1.2b"):
        s = Scenario().replace("model", m)
        r = evaluate(s)
        f1 = sum(st.flops_u for st in r.stages)
        for key in ("tp", "sp"):
            x = evaluate(s.replace("layout." + key, 2))
            assert abs(sum(st.flops_u for st in x.stages) * 2 - f1) <= 1e-3 * f1, (m, key)
            assert max(st.mem.stored_w for st in x.stages) <= max(st.mem.stored_w for st in r.stages) * (1 + 1e-9)
        d = evaluate(s.replace("layout.dp", 2))
        if r.domain == "gen":
            assert d.latency <= r.latency * (1 + 1e-9)        # DP = CFG cond / uncond split (MODEL §layouts)
        else:
            assert abs(d.latency - r.latency) <= 1e-9 * r.latency
