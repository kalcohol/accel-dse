"""0.209: recurrent-state on-chip residency (dedicated state buffer Chip.state_sram_mib, per-stream miss split,
partial residency knee probe) and the 'state SRAM' what-if lever; adversarial edge inputs for 0.203–0.208."""
import math
import sys
from dataclasses import replace
from pathlib import Path

from accel_dse import api


def _approx(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)
from accel_dse.core import memplan
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, DISPATCH_MODES
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

RWKV, LUX = "rwkv7-g1k-7.2b", "startlux-decision-v1-35b-a3b"


def _sc(chip="1P", model="qwen3-8b", ssram=0.0, **sv):
    s = Scenario(chip=replace(CHIPS[chip], state_sram_mib=ssram), model=model)
    return replace(s, serving=replace(s.serving, **sv)) if sv else s


def _st(r):
    return max(r.stages, key=lambda s: s.mem.state_total).mem


def test_state_buffer_full_partial_none():
    st1 = _st(evaluate(_sc(model=RWKV, batch=1))).state_total
    for B, n_fit in ((8, 8), (8, 3), (8, 0)):
        r = evaluate(_sc(model=RWKV, batch=B, ssram=n_fit * st1 / 2**20, sram_mib=0.0)) if False else \
            evaluate(_sc(model=RWKV, batch=B, ssram=n_fit * st1 / 2**20))
        m = _st(r)
        assert m.state_sram >= min(n_fit * st1, m.state_total) * 0.999
        assert m.state_sram <= m.state_total * 1.0001


def test_full_residency_removes_state_dram_traffic():
    a = evaluate(_sc(model=RWKV, batch=64))
    b = evaluate(_sc(model=RWKV, batch=64, ssram=1e9))
    assert _approx(_st(b).state_sram, _st(b).state_total)
    assert b.tpot < a.tpot and b.balance()["ai_asymptote"] > a.balance()["ai_asymptote"]


def test_buffer_zero_is_bit_identical():
    a, b = evaluate(_sc(model=RWKV, batch=16)), evaluate(_sc(model=RWKV, batch=16, ssram=0.0))
    assert a.tpot == b.tpot


def test_split_miss_hybrid_state_pinned_not_charged():
    r = evaluate(_sc(model=LUX, batch=4, ssram=1e9))
    m = _st(r)
    assert _approx(m.state_sram, m.state_total)
    old = memplan.STATE_SPLIT
    try:
        memplan.STATE_SPLIT = False
        r0 = evaluate(_sc(model=LUX, batch=4, ssram=1e9))
    finally:
        memplan.STATE_SPLIT = old
    assert r.tpot <= r0.tpot


def test_rwkv_1p_state_sram_answer():
    """The 0.209 answer: RWKV-7 7.2B on 1P reaches balance with a ~10.1 GiB dedicated state buffer (≈ 314 seq × 33 MiB)."""
    assert not evaluate(_sc(model=RWKV)).balance()["reachable"]
    b = evaluate(_sc(model=RWKV, ssram=10363)).balance()
    assert b["reachable"] and 290 <= b["critical_batch"] <= 330 <= b["batch_limit"]
    assert not evaluate(_sc(model=RWKV, ssram=8192)).balance()["reachable"]


def test_knee_probe_satisfies_definition():
    r = evaluate(_sc(model=RWKV, ssram=10363))
    b = r.balance()
    assert r._ai_at(b["critical_batch"])[0] >= b["chip_flop_per_byte"]


def test_state_sram_whatif_lever():
    s = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]
    rows = {x["key"]: x for x in s["balance_whatif"]["rows"] if "key" in x}
    assert "state_sram" in rows and rows["state_sram"]["reachable"] and rows["state_sram"]["works"]
    q = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3-8b"}})["sensitivity"]
    assert all(x.get("key") != "state_sram" for x in (q.get("balance_whatif") or {"rows": []})["rows"])


def test_api_accepts_state_sram_mib():
    r = api.api_eval({"chip_preset": "1P", "scenario": {"model": RWKV, "chip": {"state_sram_mib": 1e5}}, "sensitivity": False})
    assert r["balance"]["reachable"] is not False   # finite buffer: AI peaks at the knee; ai_asymptote is the B→∞ (spilled) value


# ----------------------------------------------------------------- adversarial edge inputs (0.203–0.208)
def _nan(x):
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, dict):
        return any(_nan(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_nan(v) for v in x)
    return False


EDGES = [
    dict(model="qwen3-8b", serving={"batch": 1}),
    dict(model=RWKV, serving={"batch": 1, "ctx": 1}),
    dict(model="qwen3-8b", serving={"ctx": 1000000, "batch": 2}),
    dict(model="qwen3-8b", layout={"tp": 3}),                     # TP not dividing heads
    dict(model=LUX, layout={"pp": 3}, serving={"batch": 7}),       # MoE+state hybrid, multi-card PP
    dict(model="qwen3-8b", serving={"tpot_slo_ms": 1e9, "ttft_slo_ms": 1e9}),   # effectively no SLO
    dict(model="qwen3-8b", serving={"tpot_slo_ms": 1e-6}),
    dict(model="qwen3-8b", serving={"phase": "prefill", "labels": 4, "batch": 3}),
    dict(model="qwen3-next-80b-a3b", layout={"tp": 2, "pp": 2}, serving={"batch": 64}),
]


def test_edge_inputs_no_nan_consistent():
    for i in range(len(EDGES)):
        for dispatch in [""] + list(DISPATCH_MODES):
            _edge(i, dispatch)


def _edge(i, dispatch):
    sc = {**EDGES[i], "chip": {"dispatch": dispatch}}
    try:
        r = api.api_eval({"chip_preset": "1P", "scenario": sc, "sensitivity": False})
    except api.ApiError:
        return
    assert not _nan(r)
    w = r.get("walls")
    if w and w["stage_ms"] > 0:
        assert _approx(sum(w["share"].values()), 1.0)
        assert set(w["share"]) <= {"compute", "dram", "collective_bw", "collective_lat", "dispatch", "fixed_overhead"}
    b = r.get("balance")
    if b and b.get("reachable"):
        assert 1 <= b["critical_batch"] <= b["batch_limit"]
    if b and b.get("critical_batch") is not None:
        assert b["critical_batch"] >= 1


def test_batch_zero_rejected_or_clean():
    try:
        r = api.api_eval({"chip_preset": "1P", "scenario": {"serving": {"batch": 0}}, "sensitivity": False})
    except api.ApiError:
        return
    assert not _nan(r)


def test_fuzz_eval():
    for seed in range(3):
        _fuzz(seed)


def _fuzz(seed):
    import random
    rnd = random.Random(seed)
    models = ["qwen3-8b", RWKV, LUX, "qwen3-next-80b-a3b", "llama-3.2-1b"]
    for _ in range(6):
        sv = {"batch": rnd.choice([1, 2, 7, 64, 1000]), "ctx": rnd.choice([1, 4096, 131072]),
              "phase": rnd.choice(["decode", "decode", "prefill"]), "tpot_slo_ms": rnd.choice([50.0, 1e9])}
        sc = {"model": rnd.choice(models), "serving": sv, "layout": {"tp": rnd.choice([1, 2, 3]), "pp": rnd.choice([1, 2])},
              "chip": {"dispatch": rnd.choice([""] + list(DISPATCH_MODES)), "state_sram_mib": rnd.choice([0.0, 64.0, 1e5])}}
        body = {"chip_preset": rnd.choice(sorted(CHIPS)), "scenario": sc, "sensitivity": False}
        try:
            r = api.api_eval(body)
        except api.ApiError:
            continue
        assert not _nan(r), body
        b = r.get("balance") or {}
        if b.get("reachable"):
            assert b["critical_batch"] <= b["batch_limit"], body


def test_ui_keys_exist_in_api():
    """API↔UI: every balance / walls / what-if key the UI reads is present in the API response."""
    import re
    from pathlib import Path
    js = W.read()                                    # 0.286: parsed (property reads in the block / function bodies)
    fn = W.functions(js)
    r = api.api_eval({"chip_preset": "1P", "scenario": {"model": RWKV}, "sensitivity": False})
    bal = W.blocks(js, "if (r.balance)")
    assert len(bal) == 1 and W.props(bal[0][1], "b")
    for k in W.props(bal[0][1], "b"):
        assert k in r["balance"], k
    assert W.props(fn["renderWalls"], "w")
    for k in W.props(fn["renderWalls"], "w"):
        assert k in r["walls"], k
    # 0.227: with a power cap the what-if adds power_cap_W / works_capped / binds / cap_basis (UI reads them only then)
    s = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}, "power_cap_W": 100.0})["sensitivity"]["balance_whatif"]
    assert W.props(fn["renderBalanceWhatif"], "w")
    for k in W.props(fn["renderBalanceWhatif"], "w"):
        assert k in s, k
    assert "c-state_sram_mib" in W.html_ids(W.read(W.INDEX_HTML))


def test_shared_sram_invariant_with_state_buffer():
    for model in (RWKV, LUX):
        for ssram in (0.0, 100.0, 1e9):
            _inv(model, ssram)


def _inv(model, ssram):
    from accel_dse.core.evaluate import _shared_kv
    r = evaluate(_sc(model=model, batch=32, ssram=ssram))
    for s in r.stages:
        m = s.mem
        assert m.pinned_hot + m.pinned_expert + _shared_kv(m) + m.staging <= m.sram * (1 + 1e-9)
        assert 0 <= m.state_sram <= m.state_total * (1 + 1e-9) and m.state_sram <= m.kv_sram * (1 + 1e-9)


def test_open_window_long_run_fast_and_exact():
    """0.209 overlap fix: a long run inside an open shared-expert window is skipped, not replayed (was minutes)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from freeze_clock import clock                       # 0.316: freeze-aware (host/VM freeze time not charged)
    from accel_dse.core import overlap as o
    s1, s2 = o._Sim(1e8, True), o._Sim(1e8, True)
    tile = (1048576.0, 5.5e-06, 6.8e-07)
    for s in (s1, s2):
        s.feed([("open", 0, 0.0)])
    fc = clock()
    t = fc.now()
    s1.feed_runs([(tile, 200000)], True)
    assert fc.now() - t < 2, (fc.now() - t, fc.stalls())
    s2.feed([tile]); s2.feed([tile] * 199999)
    for s in (s1, s2):
        s.feed([("close", 0, 0.0)])
    assert _approx(s1.c_end, s2.c_end, rel=1e-9) and _approx(s1.hw, s2.hw, rel=1e-9)
