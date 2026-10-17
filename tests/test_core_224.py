"""0.224: ReplaySSM GDN cursor cycle (early flush) + KDA fold-every-commit; radix-track state copies; tree drafting
(EAGLE-style topk / num_draft_tokens); seeded fuzzer incl. the J/token × tok/s/card = W/card invariant."""
import random
import sys
from dataclasses import replace as rp

from accel_dse.core import catalog as C
from accel_dse.core import ir
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.schedule import spec_expected_tokens, spec_tree_accept, tree_tokens

sys.path.insert(0, "tests")


def _sc(model, b=4, k=0, a=0.7, dm="", roll="", topk=1, n=0, track=0, ctx=4096):
    s = Scenario(chip=CHIPS["1P"], model=model)
    for f, v in (("batch", b), ("spec_k", k), ("spec_accept", a), ("ctx", ctx), ("spec_topk", topk),
                 ("spec_draft_tokens", n), ("mamba_track_interval", track)):
        s = s.replace("serving." + f, v)
    if dm:
        s = s.replace("serving.draft_model", dm)
    if roll:
        s = s.replace("serving.spec_rollback", roll)
    return s


def _need(r):
    return max(s.mem.dram_need for s in r.stages)


# ---- 1. ReplaySSM cursor cycle / KDA fold / track
def test_ring_len_power_of_two():
    for q in range(2, 10):
        L = ir.replay_ring_len(q)
        assert L & (L - 1) == 0 and L >= max(16, 2 * q)


def test_replay_cycle_follows_sglang_cursor_rule():
    # L = 16, q = 2: flush when write_pos + 4 > 16
    h, fr, fn = ir.replay_cycle(16, 2, 1.0)
    # deterministic E = 1: write_pos 1..12 then flush (13 + 4 > 16 ⇒ flag at wp = 13): cycle of 13 steps
    assert abs(fr - 1 / 13) < 0.01 and abs(fn - 1.0) < 0.02       # every committed record folded once
    assert 5 < h < 7
    # zero headroom (L = 2q): fold every step, no history
    h, fr, fn = ir.replay_cycle(16, 8, 3.0)
    assert h == 0 and abs(fr - 1) < 1e-3 and abs(fn - 3.0) < 1e-2
    # fold volume per step = committed tokens per step (long run)
    for E in (1.0, 2.5, 4.0):
        assert abs(ir.replay_cycle(16, 4, E)[2] - E) < 0.05 * E


def _op(m, q, **kw):
    m = C.get_model(m)
    for li in range(m.n_layers):
        ops = ir.build_rank_ops(m, li, Phase("decode", 4, q, 2048, **kw), Shard())
        o = [o for o in ops if o.name == "state_update" and o.state_rw]
        if o:
            return o[0]


def test_kda_fold_every_commit():
    sn = _op("glm-5.3-flash", 4)
    rr = _op("glm-5.3-flash", 4, roll="replay", acc=0.7)
    one = _op("glm-5.3-flash", 1)
    st = one.state_rw / 2
    E = ir.spec_expected(4, 0.7)
    assert rr.state_rw > 3 * st and rr.state_rw < sn.state_rw         # 3 states + ring, < (q + 3) states
    assert rr.vec > sn.vec
    assert E > 1


def test_track_adds_state_copies():
    a = _op("qwen3.8-27b", 1)
    b = _op("qwen3.8-27b", 1, track=256)
    st = a.state_rw / 2
    assert abs(b.state_rw - a.state_rw - st / 256) < 1e-6 * st
    s4 = _op("qwen3.8-27b", 4, acc=0.7, track=256)
    s4_0 = _op("qwen3.8-27b", 4)
    assert abs(s4.state_rw - s4_0.state_rw - st * ir.spec_expected(4, 0.7) / 256) < 1e-6 * st
    r4 = _op("qwen3.8-27b", 4, roll="replay", acc=0.7, track=256)
    r4_0 = _op("qwen3.8-27b", 4, roll="replay", acc=0.7)
    assert r4.state_rw > r4_0.state_rw and r4.vec >= r4_0.vec


def test_replay_falls_back_for_non_gdn_kda():
    r = evaluate(_sc("minimax-text-01", 1, 3, dm="qwen3-0.6b", roll="replay"))
    assert any("lightning" in w for w in r.warnings)


def test_track_field_validation():
    for bad in (-1, True, 1.5):
        try:
            Serving(mamba_track_interval=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


# ---- 2. tree drafting
def test_tree_tokens_and_acceptance():
    assert tree_tokens(3, 1, 0) == 4 and tree_tokens(3, 4, 16) == 16 and tree_tokens(3, 4, 0) == 13
    assert spec_tree_accept(0.7, 3, 1, 0) == 0.7
    a4 = spec_tree_accept(0.7, 3, 4, 16)
    assert 0.7 < a4 < 1 and abs(a4 - (1 - 0.3 ** 4)) < 1e-12
    assert abs(spec_tree_accept(0.7, 4, 4, 5) - 0.7) < 1e-12        # one candidate per depth = chain
    for bad in ({"spec_topk": 0}, {"spec_topk": 11}, {"spec_draft_tokens": 1}, {"spec_draft_tokens": 65}):
        try:
            Serving(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_tree_verify_cost_and_tokens():
    ch = evaluate(_sc("qwen3.8-27b", 1, 3))
    tr = evaluate(_sc("qwen3.8-27b", 1, 3, topk=4, n=16).replace("serving.spec_tree_accept_model", "indep"))  # 0.226
    assert tr.step > ch.step and tr.tokens_per_step > ch.tokens_per_step
    assert abs(tr.tokens_per_step - spec_expected_tokens(3, spec_tree_accept(0.7, 3, 4, 16))) < 1e-12
    assert _need(tr) > _need(ch)                       # hybrid: N intermediate states (SGLang num_draft_tokens)
    # replay is linear-chain only → tree uses snapshot (same result as snapshot) + warning
    trr = evaluate(_sc("qwen3.8-27b", 1, 3, topk=4, n=16, roll="replay"))
    assert trr.step == tr.step and any("树形" in w for w in trr.warnings)
    # dense target + draft model: steps 2..k at topk × batch rows
    d1 = evaluate(_sc("qwen3-32b", 4, 3, dm="qwen3-0.6b"))
    d4 = evaluate(_sc("qwen3-32b", 4, 3, dm="qwen3-0.6b", topk=4, n=16))
    assert d4.draft["s"] > d1.draft["s"] and d4.draft["pass_s"] == d1.draft["pass_s"]
    # spec_draft_tokens beyond the candidate tree
    try:
        evaluate(_sc("qwen3.8-27b", 1, 2, topk=2, n=40))
    except ValueError:
        pass
    else:
        raise AssertionError("oversized tree accepted")
    # DSpark: no tree
    r = evaluate(_sc("qwen3.8-27b-ready", 1, 7, topk=4, n=16))
    assert any("DSpark" in w for w in r.warnings)


# ---- 3. fuzz + invariant over every spec mode
def test_fuzz_spec_modes_invariant():
    from test_core_217 import _table
    t, rnd = _table(), random.Random(224)
    for _ in range(14):
        m = rnd.choice(["qwen3.8-27b", "qwen3-32b", "glm-5.3-flash", "qwen3.8-27b-ready"])
        dm = "qwen3-0.6b" if m == "qwen3-32b" else ""
        k = rnd.randint(1, 4)
        topk = rnd.choice([1, 1, 2, 4])
        n = rnd.choice([0, 1 + k * topk]) if topk > 1 else 0
        roll, a = rnd.choice(["snapshot", "replay"]), rnd.choice([0.0, 0.5, 0.9, 1.0])
        b, track = rnd.choice([1, 8, 32]), rnd.choice([0, 256])
        r = evaluate(_sc(m, b, k, a, dm, roll, topk, n, track))
        for v in (r.step, r.tpot, r.throughput):
            assert v == v and 0 < v < float("inf"), (m, k, topk, n, roll, a, b, track)
        assert 1 - 1e-12 <= r.tokens_per_step <= tree_tokens(k, topk, n) + 1e-9
        e = energy_report(r, t)
        if e.get("J_per_unit"):
            assert abs(e["J_per_unit"] * (b / r.tpot / r.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-3
