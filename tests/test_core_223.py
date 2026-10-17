"""0.223: independent draft energy; spec rollback strategy (snapshot | ReplaySSM) with its what-if row; static
dispatch of speculative steps; seeded fuzzer over the new paths."""
import math
import random
from dataclasses import replace as rp

from accel_dse import api
from accel_dse.core import ir
from accel_dse.core import catalog as C
from accel_dse.core.energy import _counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.scenario import Scenario, Serving


def _sc(model, b=4, k=0, a=0.7, dm="", roll="", chip=None, ctx=4096):
    s = Scenario(chip=chip or CHIPS["1P"], model=model).replace("serving.batch", b).replace("serving.spec_k", k) \
        .replace("serving.spec_accept", a).replace("serving.ctx", ctx)
    if dm:
        s = s.replace("serving.draft_model", dm)
    if roll:
        s = s.replace("serving.spec_rollback", roll)
    return s


# ---- 1. draft energy
def test_draft_acts_counted_once_per_pass():
    r = evaluate(_sc("qwen3-32b", 8, 4, dm="qwen3-0.6b"))
    assert r.draft and set(r.draft["acts"]) >= {"mac", "dram"}
    with_d, without = _counts(r)["counts"], _counts(rp(r, draft=None))["counts"]
    for a, v in r.draft["acts"].items():
        assert abs(with_d.get(a, 0.0) - without.get(a, 0.0) - v) <= 1e-6 * max(1.0, v)
    one = evaluate(_sc("qwen3-32b", 8, 1, dm="qwen3-0.6b"))
    assert abs(r.draft["acts"]["dram"] / one.draft["acts"]["dram"] - 4) < 1e-9


def test_no_draft_energy_unchanged():
    r = evaluate(_sc("qwen3-32b", 8, 0))
    assert r.draft is None


# ---- 2. rollback strategy
def _lin_li(m):
    for li in range(m.n_layers):
        if any(o.name == "state_update" and o.state_rw for o in ir.build_rank_ops(m, li, Phase("decode", 4, 1, 2048))):
            return li


def _st(m, li, q, roll="", acc=0.0):
    ops = ir.build_rank_ops(m, li, Phase("decode", 4, q, 2048, roll=roll, acc=acc), Shard())
    o = next(o for o in ops if o.name == "state_update")
    return o.state_rw, o.vec


def test_replay_ring_len_and_expected():
    assert ir.replay_ring_len(2) == 16 and ir.replay_ring_len(8) == 16 and ir.replay_ring_len(9) == 18
    assert ir.spec_expected(5, 0.0) == 1.0 and ir.spec_expected(5, 1.0) == 5.0
    assert abs(ir.spec_expected(5, 0.7) - (1 - 0.7 ** 5) / 0.3) < 1e-12


def test_replay_traffic_below_snapshot_and_compute_above():
    m = C.get_model("qwen3.8-27b")
    li = _lin_li(m)
    for q in (4, 8):
        s_rw, s_vec = _st(m, li, q)
        r_rw, r_vec = _st(m, li, q, "replay", 0.7)
        assert r_rw < s_rw and r_vec > s_vec
    # q = 1 (no speculation): replay is not engaged
    assert _st(m, li, 1, "replay", 0.7) == _st(m, li, 1)
    # more acceptance → more folds and replayed tokens
    lo, hi = _st(m, li, 8, "replay", 0.1), _st(m, li, 8, "replay", 0.95)
    assert hi[0] > lo[0] and hi[1] > lo[1]


def test_replay_memory_and_step():
    sn = evaluate(_sc("qwen3.8-27b", 32, 3))
    rp_ = evaluate(_sc("qwen3.8-27b", 32, 3, roll="replay"))
    ar = evaluate(_sc("qwen3.8-27b", 32, 0))
    need = lambda r: max(s.mem.dram_need for s in r.stages)
    assert need(ar) < need(rp_) < need(sn)
    assert rp_.step < sn.step
    # snapshot step does not depend on acceptance; replay's does (fold / replay over expected accepted tokens)
    assert evaluate(_sc("qwen3.8-27b", 32, 3, a=0.1)).step == sn.step
    assert evaluate(_sc("qwen3.8-27b", 32, 3, a=0.95, roll="replay")).step > evaluate(_sc("qwen3.8-27b", 32, 3, a=0.1, roll="replay")).step
    # dense target: rollback mode is irrelevant
    d1 = evaluate(_sc("qwen3-32b", 8, 4, dm="qwen3-0.6b"))
    d2 = evaluate(_sc("qwen3-32b", 8, 4, dm="qwen3-0.6b", roll="replay"))
    assert d1.step == d2.step and need(d1) == need(d2)
    # no speculation: replay is a no-op
    assert evaluate(_sc("qwen3.8-27b", 32, 0, roll="replay")).step == ar.step


def test_spec_rollback_validation():
    try:
        Serving(spec_rollback="undo")
    except ValueError:
        pass
    else:
        raise AssertionError("bad spec_rollback accepted")


def test_whatif_rollback_row():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3.8-27b",
                             "serving": {"batch": 1, "ctx": 4096, "spec_k": 3}}})["sensitivity"]["balance_whatif"]
    row = next(x for x in w["rows"] if x["key"] == "spec_rollback_replay")
    cmp_ = row["spec_rollback_compare"]
    assert [(c["batch"], c["mode"]) for c in cmp_] == [(1, "snapshot"), (1, "replay"), (32, "snapshot"), (32, "replay")]
    b32 = {c["mode"]: c for c in cmp_ if c["batch"] == 32}
    assert b32["replay"]["dram_need_GiB"] < b32["snapshot"]["dram_need_GiB"]
    assert abs(row["J_per_tok"] * row["tok_s_card_at_cost_batch"] / row["W_card"] - 1) < 1e-3
    w2 = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3.8-27b",
                              "serving": {"batch": 1, "ctx": 4096, "spec_k": 0}}})["sensitivity"]["balance_whatif"]
    assert not any(x["key"].startswith("spec_rollback") for x in w2["rows"])


# ---- 4. static dispatch of speculative steps
def test_static_dispatch_spec_doubles_head_stage():
    ps = rp(CHIPS["1P"], dispatch="static").dispatch_s()[0]
    assert ps > 0
    d = lambda **kw: evaluate(_sc(chip=rp(CHIPS["1P"], dispatch="static"), **kw)).step - evaluate(_sc(**kw)).step
    assert math.isclose(d(model="qwen3.8-27b"), ps, rel_tol=1e-6)
    assert math.isclose(d(model="qwen3.8-27b", k=1), 2 * ps, rel_tol=1e-6)        # MTP draft program + verify
    assert math.isclose(d(model="qwen3-32b", k=1), ps, rel_tol=1e-6)              # spec ignored (no MTP)
    # independent draft: one target program + k draft passes, each its own program (no double charge)
    assert math.isclose(d(model="qwen3-32b", k=3, dm="qwen3-0.6b"), 4 * ps, rel_tol=1e-6)


# ---- 5. fuzz
def test_fuzz_rollback_and_draft_energy():
    from accel_dse.core.energy import energy_report
    import sys
    sys.path.insert(0, "tests")
    from test_core_217 import _table
    t = _table()
    rnd = random.Random(223)
    for _ in range(12):
        m = rnd.choice(["qwen3.8-27b", "qwen3.8-27b-ready", "qwen3-32b"])
        dm = rnd.choice(["", "qwen3-0.6b"]) if m == "qwen3-32b" else rnd.choice(["", "", "qwen3-0.6b"])
        k, a, b = rnd.randint(0, 7), rnd.choice([0.0, 0.4, 0.8, 1.0]), rnd.choice([1, 4, 32])
        if m == "qwen3.8-27b-ready" and not dm:
            k = max(k, 1)
        roll = rnd.choice(["snapshot", "replay"])
        r = evaluate(_sc(m, b, k, a, dm, roll))
        for v in (r.step, r.tpot, r.throughput):
            assert v == v and 0 < v < float("inf"), (m, dm, k, a, b, roll)
        assert r.tokens_per_step >= 1 - 1e-12
        e = energy_report(r, t)
        if e.get("J_per_unit"):
            assert abs(e["J_per_unit"] * (b / r.tpot / r.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-3
        if roll == "replay" and k:
            sn = evaluate(_sc(m, b, k, a, dm, "snapshot"))
            assert max(s.mem.dram_need for s in r.stages) <= max(s.mem.dram_need for s in sn.stages) + 1
