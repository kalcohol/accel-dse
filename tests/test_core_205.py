"""0.205: balanced default memory per NPU preset; machine-balance KPI / warning."""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse import api
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, MEM_LEGACY_0204
from accel_dse.core.scenario import Scenario

# public same-class balance, bf16 dense FLOP / raw DRAM byte (MODEL §152)
CLASS = {"H100 SXM": 989 / 3.35, "B200": 2250 / 8.0, "MI300X": 1307.4 / 5.3, "TPU v5e": 197 / (800 * 1.073741824 / 1e3),
         "TPU v6e": 918 / 1.638, "Gaudi 3": 1835 / 3.7, "Ascend 910C": 800 / 3.2}


def _bal(n):
    return evaluate(Scenario(chip=CHIPS[n])).balance()["chip_flop_per_byte_raw"]


def test_defaults_and_legacy():
    assert Scenario().mem_id == MEM_LEGACY_0204 and Scenario().hash() == "723894c17df5d7bb"   # 100T unchanged
    assert Scenario(chip=CHIPS["1P"]).mem_id == "hbm3e_4s_8h24g_9200"
    assert Scenario(chip=CHIPS["H100-like"]).mem_id == "hbm3_5s_8h16g_5234"
    for n in ("1P-lp4x64", "H100-like-lp4x64"):
        assert Scenario(chip=CHIPS[n]).mem_id == MEM_LEGACY_0204
    assert Scenario(chip=CHIPS["1P"], mem_id=MEM_LEGACY_0204).mem_id == MEM_LEGACY_0204   # explicit wins
    assert api.scenario_from_body({"chip_preset": "1P"}).mem_id == "hbm3e_4s_8h24g_9200"


def test_presets_in_public_class_range():
    lo, hi = min(CLASS.values()), max(CLASS.values())
    for n in ("100T", "1P", "H100-like"):
        assert lo * 0.9 <= _bal(n) <= hi, (n, _bal(n), lo, hi)
    for n in ("1P-lp4x64", "H100-like-lp4x64"):
        assert _bal(n) > 5 * hi                  # the ≤ 0.204 pairings: compute idles on DRAM


def test_balance_warning_sides():
    b = evaluate(Scenario(chip=CHIPS["1P-lp4x64"])).balance()
    assert b["idle"] == "compute" and "compute idle" in b["warning"]
    s = replace(Scenario(chip=CHIPS["100T"], mem_id="hbm3e_8s_12h24g_9200"),
                serving=replace(Scenario().serving, phase="prefill", batch=8, prompt=8192))
    b = evaluate(s).balance()
    assert b["idle"] == "bandwidth" and "bandwidth idle" in b["warning"], b
    r = api.api_eval({"chip_preset": "1P", "scenario": {"model": "qwen3-8b"}})
    assert set(r["balance"]) >= {"chip_flop_per_byte", "workload_flop_per_byte", "ratio", "idle", "warning"}


def test_ui_wiring():
    import pathlib
    root = pathlib.Path(api.__file__).parent
    js = (root / "web/app.js").read_text()
    assert "machine balance" in js and "default_mem" in js and "balance-warn" in js
    assert "1P-lp4x64" in (root.parent / "scripts/ui_headless_smoke.py").read_text()
