"""0.103: SFU transcendental rate (opt-in); extra MoE experts touched by a co-located chunk."""
from dataclasses import replace

from accel_dse.core.evaluate import evaluate, extra_expert_bytes
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _fused_step
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving
import accel_dse.core.extval as X

HB = "hbm3e_8s_12h24g_9200"


def test_sfu_default_off_and_binding():
    s = Scenario(chip=CHIPS["1P"], mem_id=HB, serving=Serving(phase="prefill", batch=1, prompt=32768))
    a = evaluate(s)
    b = evaluate(s.replace("chip", replace(CHIPS["1P"], sfu_per_cycle=10 ** 9)))   # huge SFU: lanes bind
    c = evaluate(s.replace("chip", replace(CHIPS["1P"], sfu_per_cycle=64)))        # tiny SFU: exp binds
    assert a.step == b.step
    assert c.stages[0].time.t_vector > a.stages[0].time.t_vector
    assert c.step >= a.step


def test_refhw_sfu_rates():
    assert REF_HW["MI300X"].sfu_per_sm == 0.0
    for k in ("H100-SXM", "H200-SXM", "B200", "A100-SXM-80GB"):
        assert REF_HW[k].sfu_per_sm == 16.0
    v = X.sfu_variant("catalog")
    r = next(r for r in X.rows(X.DATA) if r["hw"] == "H100-SXM")
    assert X.scenario(r, v).chip.sfu_per_cycle == 16.0 * 132
    assert X.scenario(r, "catalog").chip.sfu_per_cycle is None


def _pair(model, lay, ov, k):
    b = Scenario(model=model, chip=CHIPS["1P"], mem_id=HB, layout=lay, exec_overlap=ov)
    return (evaluate(b.replace("serving", Serving(phase="decode", batch=k, ctx=4096))),
            evaluate(b.replace("serving", Serving(phase="prefill", batch=1, prompt=4096))))


def test_chunk_extra_experts():
    d, p = _pair("qwen3-8b", Layout(tp=2), "stage", 4)
    assert extra_expert_bytes(d.stages[0], p.stages[0], 0.125) == 0.0           # dense
    d, p = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), "stage", 1)
    x1 = extra_expert_bytes(d.stages[0], p.stages[0], 1 / 512)
    x2 = extra_expert_bytes(d.stages[0], p.stages[0], 1 / 32)
    assert 0 < x1 <= x2
    d128, _ = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), "stage", 128)
    assert x2 <= d128.stages[0].dram["exp_dram"] * 1.001                          # at most every expert
    for ov in ("stage", "pipe", "serial"):
        d, p = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), ov, 1)
        assert _fused_step(d, p, 1 / 32, 0.0) > 2.0 * d.step                       # the chunk streams ~all experts
    ds, ps = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), "stage", 16)
    dp, pp = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), "pipe", 16)
    assert _fused_step(ds, ps, 0.125, 0.2) <= _fused_step(dp, pp, 0.125, 0.2) * (1 + 1e-9)
