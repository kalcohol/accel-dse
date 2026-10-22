"""0.210: adversarial review (multi-card PP, walls, preset memory pairing), cross-route fuzz with a timing guard,
pipe-mode per-op KV / state miss split."""
import math
import random
from dataclasses import replace

from accel_dse import api
from accel_dse.core import memplan
from accel_dse.core.evaluate import _kv_state_miss, evaluate
from accel_dse.core.hardware import CHIPS, DISPATCH_MODES, default_mem
from accel_dse.core.scenario import Scenario

from tests.workmeter import guard  # 0.332 deterministic work guard

RWKV, LUX, NEXT = "rwkv7-g1k-7.2b", "startlux-decision-v1-35b-a3b", "qwen3-next-80b-a3b"
WALL_KEYS = {"compute", "dram", "collective_bw", "collective_lat", "dispatch", "fixed_overhead"}
GUARD_S = 20.0          # per request; 0.209 found a 127 s prefill (overlap open-window replay)


def _timed(fn, body, bound: float, label: str = "210"):
    """Run one in-process request under a runtime guard that load cannot trip (0.332, tests/workmeter.py): the
    request's deterministic work (calls in accel_dse code) must stay within what ``bound`` quiet-box CPU seconds buy —
    the same threshold as 0.304's CPU-time bound, on a measure that does not inflate under -j 8 (CPU seconds did, 9.5×
    at rt278b).  ≤ 0.331 also bounded freeze-aware wall × a contention factor (0.304 / 0.316); a sleeping / blocking
    hang never returns, which the runner's per-file timeout reports (no in-process I/O on this path)."""
    with guard(bound, label):
        return fn(body)


def _nan(x):
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, dict):
        return any(_nan(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_nan(v) for v in x)
    return False


def _walls_ok(r, body):
    w = r["walls"]
    if not w or not w["stage_ms"] > 0:
        return
    assert set(w["share"]) <= WALL_KEYS, body
    assert abs(sum(w["share"].values()) - 1) < 1e-6, body
    assert 0 <= w["stage"] < len(r["stages"]), body
    assert w["stage_ms"] <= w["step_ms"] * (1 + 1e-9), body
    assert all(v >= -1e-12 for v in w["exposed_ms"].values()), body


def test_multicard_pp_walls_and_balance():
    for model in ("qwen3-8b", LUX, RWKV):
        for pp in (2, 3):
            for disp in [""] + list(DISPATCH_MODES):
                body = {"chip_preset": "1P", "scenario": {"model": model, "layout": {"pp": pp},
                                                          "serving": {"batch": 16}, "chip": {"dispatch": disp}},
                        "sensitivity": False}
                try:
                    r = api.api_eval(body)
                except api.ApiError:
                    continue
                assert not _nan(r), body
                assert len(r["stages"]) == pp, body
                _walls_ok(r, body)
                b = r["balance"]
                if b and b.get("reachable"):
                    assert 1 <= b["critical_batch"] <= b["batch_limit"], body


def test_preset_memory_pairing_consistent():
    cat = api.api_catalog()["chips"]
    for k in CHIPS:
        assert cat[k]["default_mem"] == default_mem(k) == api.scenario_from_body({"chip_preset": k}).mem_id
        m = api.api_memory({"id": default_mem(k)})
        assert m["eff_GBps"] > 0 and m["capacity_GiB"] > 0 and not _nan(m)


def _body(rnd):
    sv = {"batch": rnd.choice([1, 2, 7, 64, 4096, 100000]), "ctx": rnd.choice([1, 4096, 131072, 1000000]),
          "prompt": rnd.choice([1, 512, 8192]), "phase": rnd.choice(["decode", "decode", "prefill"]),
          "tpot_slo_ms": rnd.choice([50.0, 1e9, 1e-3])}
    return {"chip_preset": rnd.choice(sorted(CHIPS)),
            "scenario": {"model": rnd.choice(["qwen3-8b", RWKV, LUX, NEXT, "llama-3.2-1b"]), "serving": sv,
                         "layout": {"tp": rnd.choice([1, 2, 3]), "pp": rnd.choice([1, 2, 3])},
                         "chip": {"dispatch": rnd.choice([""] + list(DISPATCH_MODES)),
                                  "state_sram_mib": rnd.choice([0.0, 64.0, 1e5])}}}


def test_cross_route_fuzz_with_timing_guard():
    rnd = random.Random(210)
    routes = [("eval", lambda b: api.api_eval({**b, "sensitivity": False})),
              ("sensitivity", api.api_sensitivity),
              ("fit", api.api_fit),
              ("sweep", lambda b: api.api_sweep({**b, "path": "serving.batch", "values": [1, 8, 64]}))]
    for i in range(16):
        body = _body(rnd)
        name, fn = routes[i % len(routes)]
        try:
            r = _timed(fn, body, GUARD_S * (4 if name in ("sensitivity", "fit", "sweep") else 1), f"210 fuzz {i} {name}")
        except api.ApiError:
            continue
        assert not _nan(r), (name, body)
        if name == "eval":
            _walls_ok(r, body)


def test_huge_batch_timing_guard():
    for model, phase in ((LUX, "prefill"), (NEXT, "prefill"), (RWKV, "decode"), ("qwen3-8b", "prefill")):
        body = {"chip_preset": "1P-lp4x64", "scenario": {"model": model, "layout": {"pp": 2},
                "serving": {"phase": phase, "batch": 100000, "prompt": 8192}}, "sensitivity": False}
        try:
            _timed(api.api_eval, body, GUARD_S, f"210 huge {model} {phase}")
        except api.ApiError:
            pass


def test_pipe_split_matches_step_dram():
    s = Scenario(chip=replace(CHIPS["1P"], state_sram_mib=1e9), model=LUX)
    s = replace(s, serving=replace(s.serving, batch=8))
    r = evaluate(s)
    for st in r.stages:
        f = _kv_state_miss(st.mem)
        assert f["state"] == 0.0                   # all state resident: no state DRAM weight in pipe mode
        assert 0.0 <= f["kv"] <= 1.0
    old = memplan.STATE_SPLIT
    try:
        memplan.STATE_SPLIT = False
        f = _kv_state_miss(r.stages[0].mem)
        assert f["kv"] == f["state"]               # legacy: pooled
    finally:
        memplan.STATE_SPLIT = old


def test_state_size_capacity_vs_traffic_labels():
    """RWKV-7 7.2B fp32: capacity 33.0 MiB = 34.6 MB per sequence; traffic read + write ≈ 69.2 MB per step."""
    r = evaluate(Scenario(chip=CHIPS["1P"], model=RWKV))
    st1 = max(x.mem.state_total for x in r.stages)
    assert abs(st1 / 2**20 - 33.0) < 0.05 and abs(2 * st1 / 1e6 - 69.2) < 0.1
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]["balance_whatif"]
    lab = [x["label"] for x in w["rows"] if x.get("key") == "state_sram"][0]
    assert "33.0 MiB" in lab and "69.2 MB" in lab and "读+写" in lab
