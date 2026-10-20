"""0.97: sweep / Pareto sensitivity, /api/sensitivity, per-engine port under the engine-group split, FA3 bands."""
from dataclasses import replace
from accel_dse import api
from accel_dse.core.evaluate import evaluate, _sum_ops
from accel_dse.core.hardware import CHIPS, System
from accel_dse.core.ir import Op
from accel_dse.core.catalog import get_model
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

C = CHIPS["1P"]
HBM = "hbm3e_8s_12h24g_9200"


def test_per_engine_port_evaluates_and_scales():
    # 0.96 bug: the engine-group split re-built the chip with both port fields → ValueError on every GEMM
    pe = 4.0 * (C.rows + C.cols * C.engines) * 2.0 / C.engines / 8    # the ≤ 0.140 default port / 8 (feed-bound)
    sv = Serving(phase="prefill", batch=1, prompt=8192)
    a = evaluate(Scenario(model="qwen3-32b", chip=replace(C, sram_port_Bpc=pe * C.engines), mem_id=HBM, serving=sv))
    b = evaluate(Scenario(model="qwen3-32b", chip=replace(C, sram_port_Bpc_per_engine=pe), mem_id=HBM, serving=sv))
    assert abs(a.throughput - b.throughput) < 1e-9 * a.throughput
    # die split at fixed silicon: per-engine port keeps the total, the absolute port inflates it (the trap)
    tot = []
    for pc in (1, 2, 4):
        E = C.engines // pc
        for tag, ch in (("abs", replace(C, engines=E, sram_port_Bpc=pe * C.engines)),
                        ("pe", replace(C, engines=E, sram_port_Bpc_per_engine=pe))):
            s = Scenario(model="qwen3-32b", chip=ch, mem_id=HBM, serving=sv)
            if pc > 1:
                s = s.replace("d2d_enabled", True).replace("package_cards", pc).replace("layout.tp", pc)
            tot.append((tag, pc, evaluate(s).throughput))
    pe_t = [t for g, _, t in tot if g == "pe"]
    abs_t = [t for g, _, t in tot if g == "abs"]
    assert max(pe_t) / min(pe_t) < 1.15          # same silicon, same answer (minus D2D traffic)
    assert abs_t[-1] / abs_t[0] > 2.0            # the trap: per-die port = whole-chip port


def test_fa3_ratio_band():
    """FP8 / BF16 attention speed-up on the H100-like preset brackets FA3's measured ≈ 1.6× (740 → ~1.2 PF, hd 256):
    serial (no softmax overlap) ≤ measured ≤ full overlap (2×, exp still on the vector path)."""
    ch, M = CHIPS["H100-like"], get_model("qwen3-8b")
    s, hd, h = 8192, 256, 8
    t = {}
    for af, kv in (("", "bf16"), ("kv", "fp8")):
        ops = [Op("qk", "attn", 0, m=s, k=hd, n=s, count=2 * h, w_fmt=kv, a_fmt="bf16", causal=1.0),
               Op("pv", "attn", 0, m=s, k=s, n=hd, count=2 * h, w_fmt=kv, a_fmt="bf16", causal=1.0),
               Op("softmax", "vector", 0, vec=2 * h * s * s * 5)]
        r = _sum_ops(ops, System(replace(ch, attn_fmt=af), HBM), "os", M)
        t[af] = (max(r["t_arr"], r["t_vec"]), r["t_arr"] + r["t_vec"])
    over, ser = t[""][0] / t["kv"][0], t[""][1] / t["kv"][1]
    assert 1.6 <= over <= 2.0 + 1e-9 and 1.4 <= ser <= 1.9


def test_sweep_sensitivity_flags():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-32b", "mem_id": "lpddr6_8x96_10667_16g",
                                           "serving": {"phase": "prefill", "batch": 1, "prompt": 16384},
                                           "exec_overlap": "stage"},     # 0.100: base pinned to stage
         "path": "chip.sram_mib", "values": [16, 32, 64, 128, 256]}
    plain = api.api_sweep(b)
    assert "sensitivity" not in plain
    o = api.api_sweep({**b, "sensitivity": True})
    assert [r["tok_s_card"] for r in o["rows"]] == [r["tok_s_card"] for r in plain["rows"]]
    s = o["sensitivity"]
    assert s["knee_value"] == 64
    by = {c["label"]: c for c in s["cases"]}
    assert by["无重叠（下界）"]["knee_value"] == 256 and by["无重叠（下界）"]["knee_changed"]       # "64 MiB is enough" hinges on overlap
    assert by["DRAM 效率 0.5"]["knee_value"] == 128
    assert "exec_overlap" in s["hinges"] and "mem_eff" in s["hinges"]
    assert len(s["row_flags"]) == 5
    # sweeping an assumption itself skips that alternative
    o2 = api.api_sweep({**b, "path": "mem_eff", "values": [0.5, 0.7, 0.9], "sensitivity": True})
    assert all(c["key"] != "mem_eff" for c in o2["sensitivity"]["cases"])


def test_ties_are_not_flips():
    from accel_dse.api import _discordant, _near_best
    assert _discordant([100, 100.5, 101], [103, 102, 101]) == []          # within 1 %: ties
    assert _discordant([100, 120], [130, 110]) == [(0, 1)]
    assert _near_best([100, 99.5, 90]) == {0, 1}


def test_pareto_and_eval_sensitivity():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "lpddr6_8x96_10667_16g",
                                           "serving": {"phase": "decode", "ctx": 8192}}}
    o = api.api_pareto({**b, "sensitivity": True})
    s = o["sensitivity"]
    assert len(s["point_flags"]) == len(o["front"]) and {c["key"] for c in s["cases"]} >= {"mem_eff", "vector_lanes"}
    assert "sensitivity" not in api.api_pareto(b)
    e = api.api_eval({**b, "best_batch": True})
    bb = e["summary"]["batch"]
    x = api.api_sensitivity({**b, "scenario": {**b["scenario"], "serving": {**b["scenario"]["serving"], "batch": bb}}})
    assert abs(x["sensitivity"]["base_tok_s"] - e["sensitivity"]["base_tok_s"]) < 1e-9 * e["summary"]["tok_s"]
    assert abs(e["sensitivity"]["base_tok_s"] - e["summary"]["tok_s"]) < 1e-9 * e["summary"]["tok_s"]
    pd = api.api_eval({**b, "scenario": {**b["scenario"], "pd": {"enabled": True}}})
    assert "PD" in pd["sensitivity"]["scope"]
    try:
        api.api_sensitivity({**b, "best_batch": True}); assert False
    except api.ApiError:
        pass


def test_ui_wiring():
    import pathlib
    root = pathlib.Path(api.__file__).parent
    js, html = W.read(), W.read(W.INDEX_HTML)          # 0.286: parsed
    for t in ("sw-sens", "sw-sens-box", "sw-sens-tbl", "sw-sens-note"):
        assert t in W.html_ids(html), t
    assert "/api/sensitivity" in {c["path"] for c in W.req_calls(js)} and W.has(js, "sensitivity: false")
    assert {"renderSweepSens", "renderParetoSens"} <= set(W.functions(js))
    assert '"/api/sensitivity"' in (root / "serve.py").read_text()
