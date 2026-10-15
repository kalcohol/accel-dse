"""0.142: the pipe replay uses the auto_overlap picks and chooses them on the pipe total; perfect-EPLB rows."""
from accel_dse.core import fabric
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario

_S = {"model": "qwen3-next-80b-a3b", "layout": {"tp": 8, "ep": 8},
      "serving": {"phase": "prefill", "batch": 8, "ctx": 1024, "prompt": 256},
      "package_cards": 2, "d2d_enabled": True, "fabric": {"enabled": True}}


def test_more_d2d_units_never_slower_under_pipe_auto_overlap():
    """sysfuzz 0.141 seed 41: 4 → 8 D2D units was +0.009 % because the 4-unit stage mixed the re-picked hier α with
    the default ring bandwidth in the flow-shop replay."""
    s = Scenario.from_dict(_S)
    assert s.exec_overlap == "pipe" and s.fabric.algo == "auto_overlap"
    t = [evaluate(s.replace("d2d_units", u)).ttft for u in (2, 4, 8, 16)]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(t, t[1:])), t


def test_auto_overlap_never_worse_than_auto_in_pipe():
    for u in (2, 4, 8):
        s = Scenario.from_dict(_S).replace("d2d_units", u)
        assert evaluate(s).ttft <= evaluate(s.replace("fabric.algo", "auto")).ttft * (1 + 1e-12)


def test_pipe_replay_uses_the_stage_picks():
    """The collectives the flow shop replays carry the same algorithm the stage accounts for."""
    seen = {}
    orig = fabric.collective_full

    def spy(kind, payload, group, sys_, *a, **k):
        r = orig(kind, payload, group, sys_, *a, **k)
        if fabric._PICK_CV.get():
            seen[(kind, round(payload))] = fabric._PICK_CV.get()
        return r
    fabric.collective_full = spy
    try:
        evaluate(Scenario.from_dict(_S).replace("d2d_units", 4))
    finally:
        fabric.collective_full = orig
    assert seen


def test_perfect_eplb_rows_have_no_rank_max():
    from accel_dse.core import extval as X
    r = next(r for r in X.rows(X.DATA_MOE) if r.get("eplb") == "perfect")
    assert X.scenario(r).serving.moe_rank_max == "off"
    d = next(r for r in X.rows(X.DATA_MOE) if r.get("eplb") == "default" and r["metric"] == "prefill_tok_s_node")
    assert X.scenario(d).serving.moe_rank_max == "expected"
