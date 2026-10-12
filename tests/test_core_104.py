"""0.104: FA3-style softmax / MMA overlap in kernel modes (Chip.attn_vec_overlap, opt-in)."""
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
import accel_dse.core.extval as X


def _s(ov, flag, sfu=None):
    ch = replace(CHIPS["1P"], attn_vec_overlap=flag, sfu_per_cycle=sfu)
    return evaluate(Scenario(chip=ch, mem_id="hbm3e_8s_12h24g_9200", exec_overlap=ov,
                             serving=Serving(phase="prefill", batch=1, prompt=16384)))


def test_default_off_identical():
    for ov in ("stage", "pipe", "kernel", "class", "serial"):
        a = evaluate(Scenario(chip=CHIPS["1P"], mem_id="hbm3e_8s_12h24g_9200", exec_overlap=ov,
                              serving=Serving(phase="prefill", batch=1, prompt=16384)))
        assert a.step == _s(ov, False).step


def test_overlap_only_kernel_modes_and_bracketed():
    for ov in ("stage", "pipe", "class", "serial"):
        assert _s(ov, True).step == _s(ov, False).step
    for ov in ("kernel", "tbo"):
        for sfu in (None, 256):
            on, off, st = _s(ov, True, sfu), _s(ov, False, sfu), _s("stage", True, sfu)
            assert st.step <= on.step * (1 + 1e-9) <= off.step * (1 + 1e-9)
    assert _s("kernel", True, 256).stages[0].time.t_vec_attn > 0


def test_fa3_variant_hopper_blackwell_only():
    v = X.fa3_variant("catalog_kernel")
    seen = set()
    for r in X.rows(X.DATA) + X.rows(X.DATA_R4):
        on = X.scenario(r, v).chip.attn_vec_overlap
        assert on == any(r["hw"].startswith(h) for h in X.FA3_HW)
        seen.add((r["hw"][:4], on))
    assert any(o for _, o in seen) and any(not o for _, o in seen)
