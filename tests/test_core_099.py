"""0.99: exact pipe replay (no extrapolation), tile ≤ buffer/2, bracketed sensitivity labels."""
import sys
from pathlib import Path

from accel_dse import api
from accel_dse.core import overlap as O
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving

C = CHIPS["1P"]


def test_periodic_jump_is_exact():
    items = [(2e-4, {"hot": 3e6}, 0.0), (1e-5, {}, 0.0), (0.0, {}, 5e-5), (4e-4, {"exp": 9e6}, 0.0)]
    f, spb = {}, 1e-11
    for buf in (2 ** 21, 2 ** 23):
        for xop in (True, False):
            tl = []
            for it in items * 40:
                tl.append(O._tiles([it], f, spb, 1.0, min(O.PIPE_TILE, buf / 2)))
            sim = O._Sim(buf, xop)
            for t in tl:                                  # brute force: every tile fed one by one
                first = True
                for (tile, k) in t:
                    for _ in range(k):
                        sim.feed([tile], first); first = False
            a = O.stage_pipe_time([(items, 40)], f, spb, buf, xop)
            assert abs(a - sim.c_end) <= 1e-9 * sim.c_end


def test_dit_exact_and_fast():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from freeze_clock import clock                       # 0.316: freeze-aware (host/VM freeze time not charged)
    fc = clock()
    t = fc.now()
    r = evaluate(Scenario(model="wan2.1-14b", chip=C, exec_overlap="pipe"))
    assert fc.now() - t < 20 and r.throughput > 0, (fc.now() - t, fc.stalls())


def test_tile_capped_by_buffer():
    s = Scenario(model="qwen3-8b", chip=C, mem_id="hbm3e_8s_12h24g_9200", serving=Serving(phase="decode", batch=1, ctx=2048),
                 exec_overlap="pipe")
    t0 = O.PIPE_TILE
    try:
        a = evaluate(s).throughput
        O.PIPE_TILE = 4 * t0                      # > staging / 2: capped, still double-buffered
        assert abs(evaluate(s).throughput / a - 1) < 1e-3
    finally:
        O.PIPE_TILE = t0


def test_sensitivity_bracket_labels():
    body = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "exec_overlap": "stage"}}   # 0.100: pipe is the default
    s = api.api_sensitivity(body)["sensitivity"]
    assert s["base_label"].startswith("stage") and "上界" in s["base_label"]
    labels = [x["label"] for x in s["rows"]]
    assert "现实" in labels[0] and "下界" in labels[1]
    p = api.api_sensitivity({**body, "scenario": {**body["scenario"], "exec_overlap": "pipe"}})["sensitivity"]
    assert "现实" in p["base_label"] and [x["key"] for x in p["rows"]][:2] == ["exec_overlap", "exec_overlap"]
    assert "上界" in p["rows"][0]["label"] and "下界" in p["rows"][1]["label"]
    sw = api.api_sweep({**body, "path": "chip.sram_mib", "values": [16, 64], "sensitivity": True})["sensitivity"]
    assert [c["label"] for c in sw["cases"]][:2] == ["pipe（现实估计）", "无重叠（下界）"]
    d = api.api_sweep({"chip_preset": "1P", "scenario": {"model": "qwen3-8b"}, "path": "chip.sram_mib", "values": [16, 64],
                       "sensitivity": True})["sensitivity"]                    # default (pipe): bracket both ways
    assert [c["label"] for c in d["cases"]][:2] == ["全重叠（上界）", "无重叠（下界）"]
