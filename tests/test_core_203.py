"""0.203: NPU dispatch model (Chip.dispatch) and the per-step wall summary (Result.walls)."""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, DISPATCH_US, Chip
from accel_dse.core.scenario import Scenario


def _s(chip="100T", **kw):
    s = Scenario(chip=CHIPS[chip])
    return replace(s, chip=replace(s.chip, **kw)) if kw else s


def test_default_ideal_and_hash_stable():
    assert CHIPS["100T"].dispatch == "" and CHIPS["1P"].dispatch == ""
    assert _s().hash() == _s(dispatch="").hash()
    assert evaluate(_s()).tpot == evaluate(_s(dispatch="")).tpot


def test_modes_order_and_exact_accounting():
    base = evaluate(_s())
    t = {m: evaluate(_s(dispatch=m)) for m in ("static", "trace", "host")}
    assert base.tpot < t["static"].tpot < t["trace"].tpot < t["host"].tpot
    # static: exactly one launch per step per stage on a DRAM-bound step (t_sync is serial)
    st = t["static"].stages[0].time
    assert abs(st.t_disp - DISPATCH_US["static"] * 1e-6) < 1e-15
    # per-op modes scale linearly with the figure
    h2 = evaluate(_s(dispatch="host", dispatch_us=40.0)).stages[0].time.t_disp
    assert abs(h2 - 2 * t["host"].stages[0].time.t_disp) < 1e-12
    assert abs(t["host"].stages[0].time.t_disp / t["trace"].stages[0].time.t_disp - 20 / 1.5) < 1e-9


def test_dispatch_validation():
    for bad in (dict(dispatch="cuda"), dict(dispatch="host", dispatch_us=-1.0)):
        try:
            replace(CHIPS["100T"], **bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_walls_sum_to_stage_and_name_the_wall():
    for kw in ({}, dict(dispatch="host")):
        for ov in ("stage", "pipe", "serial"):
            s = replace(_s(**kw), exec_overlap=ov, layout=replace(_s().layout, tp=4))
            r = evaluate(s)
            w = r.walls()
            assert abs(sum(w["exposed_ms"].values()) - w["stage_ms"]) < 1e-9 * max(1.0, w["stage_ms"]) or ov != "stage"
            assert w["exact"] == (ov == "stage")
            assert w["wall"] in w["exposed_ms"]
    w = evaluate(replace(_s(dispatch="host"), exec_overlap="stage")).walls()
    assert w["exposed_ms"]["dispatch"] > 0 and w["dispatch_mode"] == "host"


def test_fixed_overhead_split_from_layer_overhead():
    s = replace(_s(dispatch="static"), layer_overhead_us=5.0, exec_overlap="stage")
    w = evaluate(s).walls()
    L = evaluate(s).model.n_layers
    assert abs(w["exposed_ms"]["fixed_overhead"] - (L - 1) * 5e-3) < 1e-9 or abs(w["exposed_ms"]["fixed_overhead"] - L * 5e-3) < 1e-9
    assert abs(w["exposed_ms"]["dispatch"] - 10e-3) < 1e-12


def test_api_eval_and_sensitivity_carry_it():
    from accel_dse import api
    r = api.api_eval({"scenario": {"model": "qwen3-8b"}})
    assert r["walls"]["wall"] == "dram"
    rows = api.api_sensitivity({"scenario": {"model": "qwen3-8b"}})["sensitivity"]["rows"]
    assert [x["key"] for x in rows].count("dispatch") == 3
    g = api.api_sensitivity({"scenario": {"model": "qwen3-8b", "chip": {"dispatch": "host"}}})["sensitivity"]["rows"]
    assert all(x["key"] != "dispatch" for x in g)
