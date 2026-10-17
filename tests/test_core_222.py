"""0.222: spec-decode verify-step mechanisms (hybrid-state snapshot, sequential DSpark Markov head, static dispatch of
the separate draft program) and the generic independent draft model (serving.draft_model), plus a seeded fuzzer."""
import random

from accel_dse import api
from accel_dse.core import catalog as C
from accel_dse.core import ir
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.memo import model_cache
from accel_dse.core.memplan import stage_storage  # noqa: F401  (signature check below)


def _lin_layer(m):
    for li in range(m.n_layers):
        ops = ir.build_rank_ops(m, li, Phase("decode", 4, 1, 2048), Shard())
        if any(o.name == "state_update" and o.state_rw > 0 for o in ops):
            return li
    raise AssertionError("no linear layer")


def _state_rw(m, li, q):
    ops = ir.build_rank_ops(m, li, Phase("decode", 4, q, 2048), Shard())
    return sum(o.state_rw for o in ops if o.name == "state_update")


def test_snapshot_state_traffic_q_plus_3():
    m = C.get_model("qwen3.8-27b")
    li = _lin_layer(m)
    one = _state_rw(m, li, 1)
    assert one > 0
    for q in (2, 4, 8):
        assert abs(_state_rw(m, li, q) / one - (q + 3) / 2) < 1e-9
    old = ir.SPEC_STATE_SNAPSHOT
    try:
        ir.SPEC_STATE_SNAPSHOT = False
        assert abs(_state_rw(m, li, 8) - one) < 1e-9
    finally:
        ir.SPEC_STATE_SNAPSHOT = old


def test_dspark_markov_head_sequential():
    m = C.get_model("qwen3.8-27b-ready")
    ops = ir.dspark_ops(m, Phase("decode", 4, 8, 2048), Shard())
    heads = [o for o in ops if o.name.startswith("dspark.markov_head")]
    samp = [o for o in ops if o.name.startswith("dspark_sample")]
    assert len(heads) >= 2 and len(heads) == len(samp)
    assert all(o.m == 4 for o in heads)      # one token per sequence per sequential Markov step
    assert len({o.name for o in heads}) == len(heads)


def _ev(model, k=0, a=0.7, dm="", b=8, ctx=4096, tp=1, chip="1P"):
    sv = {"batch": b, "ctx": ctx, "spec_k": k, "spec_accept": a}
    if dm:
        sv["draft_model"] = dm
    return api.api_eval({"chip_preset": chip, "scenario": {"model": model, "layout": {"tp": tp}, "serving": sv}})


def test_spec_state_memory_scratch():
    s0 = _ev("qwen3.8-27b", 0, b=32)["summary"]
    s1 = _ev("qwen3.8-27b", 1, b=32)["summary"]
    s3 = _ev("qwen3.8-27b", 3, b=32)["summary"]
    assert s0["dram_need_GiB"] < s1["dram_need_GiB"] < s3["dram_need_GiB"]
    # dense (no recurrent state) target: verify adds no state scratch
    d0 = _ev("qwen3-32b", 0, dm="qwen3-0.6b")["summary"]["dram_need_GiB"]
    d4 = _ev("qwen3-32b", 4, dm="qwen3-0.6b")["summary"]["dram_need_GiB"]
    base = _ev("qwen3-32b", 0)["summary"]["dram_need_GiB"]
    assert abs(d0 - base) < 1e-9 and d4 > base


def test_draft_model_step_and_tokens():
    plain = _ev("qwen3-32b")["summary"]
    s = _ev("qwen3-32b", 4, 0.7, "qwen3-0.6b")["summary"]
    s0 = _ev("qwen3-32b", 4, 0.0, "qwen3-0.6b")["summary"]
    s9 = _ev("qwen3-32b", 4, 0.9, "qwen3-0.6b")["summary"]
    assert s["step_ms"] > plain["step_ms"]
    assert abs(s0["step_ms"] - s["step_ms"]) < 1e-9          # acceptance is a throughput lever only
    e = (1 - 0.7 ** 5) / (1 - 0.7)
    assert abs(s["tpot_ms"] - s["step_ms"] / e) / s["tpot_ms"] < 1e-6
    assert s9["tpot_ms"] < s["tpot_ms"] < s0["tpot_ms"]
    assert abs(s0["tpot_ms"] - s0["step_ms"]) < 1e-9
    big = _ev("qwen3-32b", 4, 0.7, "qwen3-8b")["summary"]
    assert big["step_ms"] > s["step_ms"] and big["dram_need_GiB"] > s["dram_need_GiB"]


def test_draft_scales_with_k():
    st = [_ev("qwen3-32b", k, 0.7, "qwen3-0.6b")["summary"]["step_ms"] for k in (1, 2, 4)]
    assert st[0] < st[1] < st[2]


def test_spec_k_ignored_without_draft_warns():
    w = _ev("qwen3-32b", 4)["summary"]["warnings"]
    assert any("serving.draft_model" in x for x in w)


def test_draft_model_validation():
    from accel_dse.core.scenario import Serving
    try:
        Serving(draft_model=3)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("non-str draft_model accepted")
    try:
        _ev("qwen3-32b", 4, 0.7, "no-such-model-xyz")
    except Exception:
        pass
    else:
        raise AssertionError("unknown draft model accepted")


def test_fuzz_draft_and_snapshot_paths():
    rng = random.Random(222)
    targets = ["qwen3-32b", "qwen3.8-27b", "qwen3-8b"]
    drafts = ["qwen3-0.6b", "qwen3-1.7b"]
    for _ in range(10):
        t, d = rng.choice(targets), rng.choice(drafts + [""])
        k, a = rng.randint(0, 6), rng.choice([0.0, 0.3, 0.7, 0.95])
        b, tp = rng.choice([1, 4, 16]), rng.choice([1, 2])
        s = _ev(t, k, a, d, b=b, tp=tp)["summary"]
        for key in ("tpot_ms", "step_ms", "tok_s", "dram_need_GiB"):
            v = s[key]
            assert v == v and v > 0 and v != float("inf"), (t, d, k, a, b, tp, key, v)
        assert s["tpot_ms"] <= s["step_ms"] * (1 + 1e-9)
        if d and k:
            base = _ev(t, 0, a, "", b=b, tp=tp)["summary"]
            assert s["dram_need_GiB"] > base["dram_need_GiB"]
            assert s["step_ms"] > base["step_ms"]
