"""0.95 NPU design-space review: dequant rate honoured; monotone resource sweeps; bottleneck flips at the roofline."""
from dataclasses import replace
from accel_dse.core.dtypes import FormatSupport
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving

C = CHIPS["1P"]
HBM, LP6 = "hbm3e_8s_12h24g_9200", "lpddr6_8x96_10667_16g"
DEC = Serving(phase="decode", batch=64, ctx=4096)
PF = Serving(phase="prefill", batch=1, prompt=4096)


def _ev(chip=C, mem=HBM, sv=DEC, model="qwen3-8b", **kw):
    return evaluate(Scenario(model=model, chip=chip, mem_id=mem, serving=sv, **kw))


def test_dequant_rate_is_used():
    ov = (("attn", "int4"), ("mlp", "int4"))
    base = _ev(formats_override=ov).stages[0].time.t_vector
    same = _ev(chip=replace(C, formats=FormatSupport(C.formats.rates, float(C.lanes))), formats_override=ov)
    assert abs(same.stages[0].time.t_vector - base) <= 1e-12 * max(1.0, base)      # = lanes → unchanged
    slow = _ev(chip=replace(C, formats=FormatSupport(C.formats.rates, 64.0)), formats_override=ov)
    assert slow.stages[0].time.t_vector > 10 * base


def test_monotone_resource_sweeps():
    for sv, mem in ((DEC, HBM), (PF, LP6), (Serving(phase="decode", batch=1, ctx=4096), HBM)):
        for key, vals in (("sram_mib", (16.0, 64.0, 256.0, 1024.0)), ("engines", (8, 16, 32, 64)),
                          ("freq_ghz", (0.5, 1.0, 2.0))):
            t = [_ev(chip=replace(C, **{key: v}), mem=mem, sv=sv).throughput for v in vals]
            assert all(b >= a * (1 - 1e-9) for a, b in zip(t, t[1:])), (key, sv.phase, t)
    t = [_ev(mem=m).throughput for m in ("lpddr5x_8x64_8533_16g", LP6, "hbm3e_4s_12h24g_9200", HBM)]
    assert t == sorted(t)


def test_decode_flip_at_array_weight_load_rate():
    # B = 1 decode on a weight-stationary array: each weight tile loads in `rows` cycles, so the array's weight-intake
    # rate is engines · cols · 2 B · f — the DRAM → MAC flip sits where it crosses the HBM bandwidth (MODEL §41)
    sv = Serving(phase="decode", batch=1, ctx=4096)
    b16, b32 = _ev(chip=replace(C, engines=16), sv=sv), _ev(chip=replace(C, engines=32), sv=sv)
    bw = _ev(sv=sv).scenario  # noqa
    intake = lambda e: e * C.cols * 2 * C.freq_ghz          # GB/s
    from accel_dse.core.hardware import System
    hbm = System(C, HBM).dram_GBps
    assert intake(16) < hbm < intake(32)
    assert b16.bound == "MAC" and b32.bound == "DRAM"
