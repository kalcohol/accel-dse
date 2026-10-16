"""0.208: machine-balance what-if levers (on the sensitivity case loop); fixed-state (RWKV-7) and hybrid
(StartLux MoE) critical batch; decision readout critical batch."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario

RWKV, LUX = "rwkv7-g1k-7.2b", "startlux-decision-v1-35b-a3b"


def _sc(chip="1P", model="qwen3-8b", mem=None, **sv):
    s = Scenario(chip=CHIPS[chip], model=model, **({"mem_id": mem} if mem else {}))
    return replace(s, serving=replace(s.serving, **sv)) if sv else s


def test_fixed_state_asymptote_independent_of_ctx():
    a = [evaluate(_sc(model=RWKV, ctx=c)).balance() for c in (256, 4096, 32768)]
    assert all(x["asymptote_by"] == "state" for x in a)
    assert max(x["ai_asymptote"] for x in a) / min(x["ai_asymptote"] for x in a) < 1.01
    q = [evaluate(_sc(ctx=c)).balance()["ai_asymptote"] for c in (256, 4096)]   # KV model: falls with ctx
    assert q[0] > 5 * q[1]


def test_fixed_state_reachable_when_chip_balance_below_asymptote():
    b = evaluate(_sc(model=RWKV)).balance()          # 1P: 318 FLOP/B > fp32-state asymptote ≈ 181
    assert not b["reachable"] and b["critical_batch"] is None and "state-bound" in b["hint"] and b["warning"] is None
    b = evaluate(_sc(model=RWKV, mem="hbm3e_8s_8h24g_9200")).balance()     # 2× bandwidth: 159 FLOP/B
    assert b["reachable"] and 1 < b["critical_batch"] <= b["batch_limit"]
    r = evaluate(_sc(model=RWKV, mem="hbm3e_8s_8h24g_9200"))
    bs = b["critical_batch"]
    assert r._ai_at(bs)[0] >= b["chip_flop_per_byte"] > r._ai_at(bs - 1)[0]
    b = evaluate(replace(_sc(model=RWKV), formats_override=(("state", "bf16"),))).balance()
    assert b["reachable"] and b["ai_asymptote"] > 300


def test_hybrid_and_moe_labels():
    assert evaluate(_sc(model=LUX)).balance()["asymptote_by"] == "kv+state"
    assert evaluate(_sc()).balance()["asymptote_by"] == "kv"


def test_decision_critical_batch():
    b = evaluate(_sc(model=LUX, phase="prefill", prompt=512, labels=4)).balance()
    assert b["reachable"] and b["critical_batch"] > 1 and b["warning"] is None
    b = evaluate(_sc(chip="100T", model="startlux-decision-v1-9b", phase="prefill", prompt=512, labels=4)).balance()
    assert b["limited_by"] in ("ttft_slo", "kv_capacity", None)
    w = evaluate(_sc(model=LUX, phase="prefill", prompt=512, labels=4)).walls()
    assert w["wall"] in ("dram", "compute")


def test_whatif_levers_single_variable():
    w = api.api_sensitivity({"chip_preset": "1P"})["sensitivity"]["balance_whatif"]
    keys = [x["key"] for x in w["rows"]]
    assert keys == ["kv_fp8", "ctx_half", "sram_x2", "dram_bw_x2"]
    by = {x["key"]: x for x in w["rows"]}
    assert by["kv_fp8"]["ai_asymptote"] > 1.9 * w["base"]["ai_asymptote"] and by["kv_fp8"]["works"]
    assert not by["sram_x2"]["works"]
    assert abs(by["dram_bw_x2"]["chip_flop_per_byte"] - w["base"]["chip_flop_per_byte"] / 2) < 1
    assert by["dram_bw_x2"]["batch_limit"] == w["base"]["batch_limit"]        # same capacity
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": RWKV}})["sensitivity"]["balance_whatif"]
    by = {x["key"]: x for x in w["rows"]}
    assert "kv_fp8" not in by and by["state_bf16"]["reachable"] and not by["ctx_half"]["works"]
    assert api.api_sensitivity({"chip_preset": "1P", "scenario": {"serving": {"phase": "prefill"}}})["sensitivity"]["balance_whatif"] is None


def test_shared_case_loop_and_ui():
    import inspect
    import pathlib
    assert "_eval_cases(" in inspect.getsource(api.sensitivity_report) and "_eval_cases(" in inspect.getsource(api.balance_whatif)
    root = pathlib.Path(api.__file__).parent
    assert "renderBalanceWhatif" in (root / "web/app.js").read_text()
    assert "balance what-if table" in (root.parent / "scripts/ui_headless_smoke.py").read_text()
