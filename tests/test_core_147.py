"""0.147: mixed-precision native MAC pairs (FormatSupport.mixed) as a design variable — W4A8 / W4A16 / W8A16."""
from dataclasses import replace

from accel_dse import api as A
from accel_dse.core.dtypes import MIXED_PRESETS, FormatSupport, gemm_exec
from accel_dse.core.evaluate import evaluate
from accel_dse.core.insight import knobs
from accel_dse.core.scenario import Scenario


def _mx(s, *names, dq=256):
    pairs = tuple(p for n in names for p in MIXED_PRESETS[n][0])
    return s.replace("chip", replace(s.chip, formats=replace(s.chip.formats, mixed=pairs, dequant_elems_per_cycle=dq)))


def test_default_chip_and_fingerprint_unchanged():
    assert Scenario().chip.formats.mixed == ()
    assert Scenario().hash() == "723894c17df5d7bb"
    assert "mixed" not in Scenario().to_dict()["chip"].get("formats", {})


def test_gemm_exec_native_only_when_pair_listed():
    d = Scenario().chip.formats
    assert gemm_exec("mxfp4", "fp8", d)[2] == "dequant_w"
    m = replace(d, mixed=MIXED_PRESETS["w4a8"][0])
    assert gemm_exec("mxfp4", "fp8", m) == ("mxfp4*fp8", 2.0, "none", 0) and m.rate("mxfp4*fp8") == 2.0
    assert gemm_exec("mxfp4", "bf16", m)[2] == "dequant_w"          # W4A16 not listed → dequant still charged
    assert gemm_exec("fp8", "fp8", m)[0] == "fp8"                   # same-format path untouched


def test_mixed_removes_dequant_bottleneck():
    for mid, preset in (("gpt-oss-120b", "w4a16"), ("mimo-v2.6-flash", "w4a8")):
        s = Scenario().replace("model", mid).replace("serving.batch", 64)
        base, other = _mx(s), _mx(s, "w8a16")                    # unrelated pair: no change
        assert abs(evaluate(other).tpot / evaluate(base).tpot - 1) < 1e-9
        assert evaluate(_mx(s, preset)).tpot < evaluate(base).tpot


def test_bad_mixed_entries_rejected():
    for bad in ([["mxfp4", "bf16", 0]], [["foo", "bf16", 1]], [["mxfp4", "bf16"]], "x",
                [["mxfp4", "bf16", True]], [["mxfp4", "bf16", float("inf")]]):
        try:
            A.api_eval({"scenario": {"chip": {"formats": {"mixed": bad}}}})
        except A.ApiError:
            continue
        raise AssertionError(bad)


def test_insight_knobs_and_catalog():
    keys = {k: c for k, _, _, c in knobs(Scenario())}
    assert {"mixed_w4a8", "mixed_w4a16", "mixed_w8a16"} <= set(keys)
    cat = A.api_catalog()
    assert set(cat["mixed_presets"]) == {"w4a8", "w4a16", "w8a16"} and cat["chips"]["100T"]["mixed"] == []
    assert "假设" in cat["mixed_presets"]["w4a16"]["source"] and "PTX" in cat["mixed_presets"]["w4a8"]["source"]


def test_ui_wiring():
    import pathlib
    web = pathlib.Path(A.__file__).parent / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    assert 'id="chip-mixed"' in html and "function paintMixed" in js and "mixed_presets" in js
