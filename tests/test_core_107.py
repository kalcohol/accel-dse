"""0.107: genomics adversarial audit — bidirectional work conserved under CP / TP / PP / SP; window accounting."""
import math
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _work(r):
    lay = r.scenario.layout
    return sum(st.flops_u for st in r.stages) * lay.tp * lay.sp * lay.dp * lay.cp / math.ceil(r.scenario.serving.batch / r.microbatches)


def test_bidir_work_conserved_under_cp():
    """attn_mode = bidir with cp > 1 charged the causal zigzag share S²/(2cp) per rank (cp 4 lost up to 30 % of the work
    at 32K); every rank's queries must see all S keys."""
    for mid, S in (("carbon-500m", 32768), ("qwen3-8b", 8192)):
        sc = {"model": mid, "serving": {"phase": "prefill", "batch": 1, "prompt": S}, "attn_mode": "bidir"}
        w1 = _work(evaluate(Scenario.from_dict(sc)))
        for lay in ({"cp": 2}, {"cp": 4}, {"tp": 2, "cp": 2}, {"tp": 2}, {"pp": 2}):
            r = evaluate(Scenario.from_dict({**sc, "layout": lay}))
            assert abs(_work(r) / w1 - 1) < 1e-6, (mid, lay, _work(r) / w1)
        # causal CP unchanged
        c1 = _work(evaluate(Scenario.from_dict({**sc, "attn_mode": "reference"})))
        c4 = _work(evaluate(Scenario.from_dict({**sc, "attn_mode": "reference", "layout": {"cp": 4}})))
        assert abs(c4 / c1 - 1) < 1e-6 and w1 > c1


def test_carbon_a_work_conserved_and_window_accounting():
    base = {"model": "carbon-a-1.2b", "serving": {"batch": 2}}
    w0 = _work(evaluate(Scenario.from_dict(base)))
    for lay in ({"tp": 2}, {"tp": 4}, {"pp": 2}, {"sp": 2}, {"sp": 4}, {"dp": 2}, {"tp": 2, "sp": 2}):
        assert abs(_work(evaluate(Scenario.from_dict({**base, "layout": lay}))) / w0 - 1) < 1e-6, lay
    for L in (1, 2, 1024, 1025, 16384):
        g = evaluate(Scenario.from_dict({**base, "workload": {"seq_len": L}})).summary()["gen"]
        w = g["workload"]
        assert 0 <= w["overlap_tokens"] < L and w["new_bp_per_window"] == (L - w["overlap_tokens"]) * 6
        assert abs(g["bp_per_s_card"] - g["windows_per_s_card"] * w["new_bp_per_window"]) <= 1e-9 * g["bp_per_s_card"] + 1e-12
