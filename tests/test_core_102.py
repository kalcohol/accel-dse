"""0.102: op-level co-located chunk iteration under pipe; vector-lane / dequant catalog sanity."""
from accel_dse.core.evaluate import evaluate, pipe_fused_time
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _fused_step
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core import vecsanity as V

HB = "hbm3e_8s_12h24g_9200"


def _pair(model, lay, ov="pipe", k=16):
    b = Scenario(model=model, chip=CHIPS["1P"], mem_id=HB, layout=lay, exec_overlap=ov)
    return (evaluate(b.replace("serving", Serving(phase="decode", batch=k, ctx=4096))),
            evaluate(b.replace("serving", Serving(phase="prefill", batch=1, prompt=4096))))


def test_fused_pipe_is_op_level_and_bracketed():
    for model, lay in (("qwen3-8b", Layout(tp=2)), ("qwen3-30b-a3b", Layout(tp=2, ep=2)),
                       ("deepseek-v3", Layout(tp=8, ep=8))):
        _check_bracket(model, lay)


def _check_bracket(model, lay):
    d, p = _pair(model, lay)
    ds, ps = _pair(model, lay, "stage")
    dz, pz = _pair(model, lay, "serial")
    for f, rr in ((0.125, 0.4), (0.5, 0.2)):
        assert pipe_fused_time(d.stages[0], p.stages[0], f, rr, d.stages[0].time.t_dram) is not None
        t = _fused_step(d, p, f, rr)
        assert _fused_step(ds, ps, f, rr) <= t * (1 + 1e-9) and t <= _fused_step(dz, pz, f, rr) * 1.01
        assert _fused_step(d, p, f, rr) == t          # memoised replay is deterministic


def test_fused_pipe_grows_with_chunk():
    d, p = _pair("qwen3-8b", Layout(tp=2))
    ts = [_fused_step(d, p, f, 0.0) for f in (0.05, 0.1, 0.25, 0.5, 1.0)]
    assert ts == sorted(ts)


def test_vector_lane_presets_within_public_band():
    lo, hi = V.public_band()["mac_per_lane"]
    assert abs(lo - 8.0) < 8e-3 and hi == 64.0
    for name, ch in CHIPS.items():
        r = V.check(ch, 1000.0)
        assert r["mac_per_lane"][1], name


def test_dequant_warning():
    ov = (("attn", "int4"), ("mlp", "int4"))
    s = Scenario(chip=CHIPS["100T"], mem_id=HB, serving=Serving(phase="decode", batch=1, ctx=4096), formats_override=ov)
    assert any("反量化" in w for w in evaluate(s).warnings)
    assert not any("反量化" in w for w in evaluate(s.replace("formats_override", ())).warnings)
