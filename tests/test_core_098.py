"""0.98: exec_overlap "pipe" — hardware-derived overlap (double buffering, kernel boundaries, DMA, dependencies)."""
import random
from dataclasses import replace
from accel_dse.core import overlap as O
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving

C = CHIPS["1P"]
HB, LP = "hbm3e_8s_12h24g_9200", "lpddr6_8x96_10667_16g"
F8 = (("attn", "fp8"), ("mlp", "fp8"), ("act", "fp8"), ("kv", "fp8"))


def _t(s):
    r = evaluate(s)
    return r.stages[r.heaviest_stage].time


def test_flow_shop_basics():
    # one op: c = m = 1 s over k tiles → max + one tile of exposure
    k = O.PIPE_MIN_TILES
    tiles = [(1.0, 1.0 / k, 1.0 / k)] * k
    assert abs(O.makespan(tiles, buf=1e9) - (1 + 1 / k)) < 1e-12
    # no buffer room beyond one tile → load and compute alternate (serial)
    assert abs(O.makespan(tiles, buf=1.0) - 2.0) < 1e-12
    random.seed(3)
    for _ in range(300):                     # monotone in times, sizes and buffer
        t = [(random.choice([0, .5, 1]), random.random(), random.random()) for _ in range(8)]
        t = [(s, m if s else 0.0, c) for s, m, c in t]
        b = random.choice([1.0, 2.0, 4.0])
        base = O.makespan(t, b)
        i = random.randrange(8)
        u = list(t); u[i] = (u[i][0], u[i][1] * .5, u[i][2] * .5)
        assert O.makespan(u, b) <= base + 1e-12 and O.makespan(t, 2 * b) <= base + 1e-12
        assert max(sum(x[1] for x in t), sum(x[2] for x in t)) - 1e-12 <= base <= sum(x[1] + x[2] for x in t) + 1e-12


def test_between_stage_and_serial():
    cases = [("qwen3-8b", HB, Serving(phase="decode", batch=64, ctx=4096), ()),
             ("qwen3-8b", LP, Serving(phase="prefill", batch=1, prompt=8192), ()),
             ("qwen3-32b", HB, Serving(phase="decode", batch=128, ctx=8192), F8),
             ("wan2.1-14b", HB, Serving(), ())]
    for model, mem, sv, fo in cases:
        s = Scenario(model=model, chip=C, mem_id=mem, serving=sv, formats_override=fo, exec_overlap="stage")
        lo, hi = _t(s).total, _t(s.replace("exec_overlap", "serial")).total
        p = _t(s.replace("exec_overlap", "pipe"))
        raw = p.t_pipe + p.t_sync
        assert lo * (1 - 1e-9) <= raw <= hi * (1 + 1e-9), model        # not clamped
    s = Scenario(model="deepseek-v3", chip=C, mem_id=HB, formats_override=F8, exec_overlap="stage",
                 serving=Serving(phase="decode", batch=256, ctx=4096)).replace("layout.tp", 8).replace("layout.ep", 8)
    st, pp = _t(s).total, _t(s.replace("exec_overlap", "pipe")).total
    assert 1.05 < pp / st < 1.3                                          # exposed collectives, MoE overlap with shared


def test_monotone_in_sram():
    for model, mem, sv in (("qwen3-30b-a3b", HB, Serving(phase="decode", batch=64, ctx=4096)),
                           ("qwen3-30b-a3b", LP, Serving(phase="prefill", batch=1, prompt=8192)),
                           ("qwen3-8b", HB, Serving(phase="decode", batch=1, ctx=2048))):
        for xop in (True, False):
            prev = 0.0
            for mib in (1, 2, 4, 16, 64, 256, 1024):
                r = evaluate(Scenario(model=model, chip=replace(C, sram_mib=mib, pipe_cross_op=xop), mem_id=mem,
                                      serving=sv, exec_overlap="pipe"))
                assert r.throughput >= prev * (1 - 1e-12), (model, xop, mib)
                prev = r.throughput


def test_kernel_boundaries_and_dma():
    s = Scenario(model="qwen3-8b", chip=C, mem_id=HB, serving=Serving(phase="decode", batch=64, ctx=4096), exec_overlap="pipe")
    a = _t(s).total
    assert _t(s.replace("chip", replace(C, pipe_cross_op=False))).total > a * 1.01   # every op pays its prologue
    lim = s.replace("chip", replace(C, dma_engines=4, dma_outstanding_KiB=64, dram_latency_ns=500))   # 524 GB/s
    assert _t(lim).total > a * 1.5
    assert abs(_t(lim.replace("exec_overlap", "stage")).t_dram * 524.288e9 - _t(s).dram_bytes) < 1e-6 * _t(s).dram_bytes
    for bad in (dict(dma_engines=4), dict(dma_engines=0, dma_outstanding_KiB=64, dram_latency_ns=500)):
        try:
            replace(C, **bad); assert False
        except ValueError:
            pass


def test_default_unchanged_and_extval_variant():
    s = Scenario(model="qwen3-8b", chip=C)
    assert "pipe_cross_op" not in str(s.to_dict()) and s.exec_overlap == "pipe"     # 0.100: pipe is the default
    import accel_dse.core.extval as X
    v = X.pipe_variant("catalog")
    assert X.VARIANTS[v]["overlap"] == "pipe" and X.VARIANTS[v]["pipe_cross_op"] is False
    r = X.rows(X.DATA)[0]
    sc = X.scenario(r, v, phase="decode", batch=8, ctx=1024, out_len=8)
    assert sc.exec_overlap == "pipe" and sc.chip.pipe_cross_op is False


from tests.pin_legacy import pin_module  # noqa: E402  (0.173)
pin_module(globals())


from tests.pin_legacy import pin_module_unfused  # noqa: E402  (0.198: per-projection checks on the unfused IR)
pin_module_unfused(globals(), only={'test_kernel_boundaries_and_dma'})
