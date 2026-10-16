"""0.192: wide-gang wave quantisation (opt-in, prefill scaling invariant), state-first SRAM pinning, AutoThink thought decode."""

from accel_dse.core import refhw
from accel_dse.core.evaluate import evaluate, model_flags
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.scenario import Scenario

WAVES = dict(WIDE_WAVES=True)


def _near(a, b, rel):
    return abs(a - b) <= rel * abs(b)


def _raises(f):
    try:
        f()
    except ValueError:
        return True
    return False

P = (4096, 8192, 16384, 32768)     # beyond the efficiency knee (fixed per-forward costs cancel in the marginal)


def _hw(name):
    s = Scenario()
    if name:
        h = refhw.REF_HW[name]
        s = s.replace("chip", h.chip).replace("mem_id", h.mem_id)
    return s


def _marginal(s):
    t = [evaluate(s.replace("serving.prompt", p)).step for p in P]
    return [(t[i + 1] - t[i]) / (P[i + 1] - P[i]) for i in range(len(P) - 1)]


def test_prefill_marginal_cost_per_token_never_falls():
    """Adversarial invariant (under WIDE_WAVES): for a fixed model, the marginal prefill time per extra prompt token
    must not decrease as the prompt grows past the knee (linear terms constant, attention terms grow).  The default
    wide gang pads N to E·cols (MODEL §119 calibration), and the 0.173 orientation swap fills it only once M ≳ E·cols:
    H200 Qwen3.5-9B 4096 → 8192 costs 1.5×, marginal per token halves (pinned below as the known default artefact)."""
    with model_flags(**WAVES):
        _invariant()


def _invariant():
    for hw, mid in [(h, m) for h in ("H200-SXM", "B200", "MI300X", "A100-SXM-80GB", None) for m in
                    ("qwen3.5-9b", "qwen3-8b", "rwkv7-g1k-7.2b", "startlux-decision-v1-27b", "startlux-decision-v1-35b-a3b")]:
        mc = _marginal(_hw(hw).replace("model", mid).replace("serving.phase", "prefill"))
        assert all(mc[i + 1] >= mc[i] * 0.97 for i in range(len(mc) - 1)), (hw, mid, mc)


def test_wide_waves_fix_and_legacy_record():
    s = _hw("H200-SXM").replace("model", "qwen3.5-9b").replace("serving.phase", "prefill")
    o = {p: evaluate(s.replace("serving.prompt", p)).step for p in (4096, 8192)}
    assert o[8192] / o[4096] < 1.6                     # default (padding) artefact, pinned
    ch = refhw.REF_HW["H200-SXM"].chip
    assert gemm_cost(ch, "os", 4096, 12288, 4096).cycles == gemm_cost(ch, "os", 4096, 4096, 24576).cycles  # ½ the FLOPs, same time
    with model_flags(**WAVES):
        t = {p: evaluate(s.replace("serving.prompt", p)).step for p in (4096, 8192)}
        a, b = gemm_cost(ch, "os", 4096, 12288, 4096), gemm_cost(ch, "os", 4096, 4096, 24576)
    assert 1.9 <= t[8192] / t[4096] <= 2.3 and t[4096] < o[4096]
    assert _near(b.cycles, 2 * a.cycles, 0.05)         # cycles ∝ FLOPs once tiles fill the gang


def test_state_first_pinning():
    for mid in ("rwkv7-g1k-1.5b", "rwkv7-g1k-7.2b"):
        for sram in (64.0, 256.0, 1024.0):
            s = Scenario().replace("model", mid).replace("serving.batch", 32).replace("serving.ctx", 4096) \
                .replace("chip.sram_mib", sram)
            new = evaluate(s)
            with model_flags(STATE_FIRST=False):
                old = evaluate(s)
            assert new.step <= old.step * (1 + 1e-9), (mid, sram)


def _d(**kw):
    s = Scenario().replace("model", "startlux-decision-v1.1-9b-autothink").replace("serving.phase", "prefill") \
        .replace("serving.prompt", 1024).replace("serving.labels", 3)
    for k, v in kw.items():
        s = s.replace("serving." + k, v)
    return s


def test_autothink_total_latency():
    base = evaluate(_d()).decision()
    th = evaluate(_d(think=True)).decision()
    assert base["think_ms"] == 0 and th["think_tokens"] == 596 and "二手" in th["think_note"]
    assert _near(th["readout_ms"], base["latency_ms"], 1e-12)
    assert _near(th["latency_ms"], th["readout_ms"] + th["think_ms"], 1e-12)
    dec = evaluate(Scenario().replace("model", "startlux-decision-v1.1-9b-autothink").replace("serving.ctx", 1322)).step
    assert _near(th["think_ms"], 596 * dec * 1e3, 0.05)
    t2 = evaluate(_d(think=True, think_tokens=1192)).decision()
    assert t2["think_ms"] > 1.9 * th["think_ms"] and "用户" in t2["think_note"]
    assert th["decisions_per_s"] < base["decisions_per_s"]


def test_think_validation():
    for bad in (0, 4097, True, 1.5):
        assert _raises(lambda: _d(think_tokens=bad)), bad
    assert _raises(lambda: Scenario().replace("serving.think", True))        # needs labels
    assert "think" not in Scenario().to_dict()["serving"]


def test_state_pinned_before_weights_in_plan():
    from accel_dse.core import memplan as MP
    s = Scenario().replace("model", "rwkv7-g1k-7.2b").replace("serving.batch", 8).replace("chip.sram_mib", 256.0)
    r = evaluate(s)
    mp = r.stages[0].mem
    assert mp.kv_sram == min(mp.state_total, mp.sram - mp.staging)        # all state (or all free SRAM) first
    assert MP.STATE_FIRST
