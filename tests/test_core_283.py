"""0.283: components (speech cards, codecs, VAE / text encoders) pay the per-op dispatch / kernel cost like stages, and
the speech bottleneck insight reports per-op dispatch sensitivity (x(u) = x(0) + n·u, exact)."""
import dataclasses

from accel_dse.api import ApiError, api_insight_speech
from accel_dse.core.catalog import get_model
from accel_dse.core.drama3 import dac_ops
from accel_dse.core.evaluate import _component_time
from accel_dse.core.hardware import CHIPS, DISPATCH_US, System
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech_insight import _with_dispatch, speech_bounds

LP = "lpddr5x_4x64_8533_16g"
B = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id=LP)


def _dac(chip):
    fs = get_model("drama-3").fish
    return _component_time([(dac_ops(fs, 10, 0, 0, 1), 3)], System(chip, LP, 0.8), "os", "fp32")


def test_component_default_unchanged_and_dispatch_charged():
    c0 = _dac(B.chip)
    assert "sync_s" not in c0 and "dispatch_s" not in c0              # default: no new keys (summaries unchanged)
    tr = _dac(dataclasses.replace(B.chip, dispatch="trace", dispatch_us=2.0))
    n_ops = tr["dispatch_s"] / 2e-6
    assert n_ops > 3 * 50 and abs(n_ops - round(n_ops)) < 1e-6            # 3 executions × many small ops
    assert abs(tr["s"] - c0["s"] - tr["dispatch_s"]) < 1e-12               # serial term (stage overlap)
    st = _dac(dataclasses.replace(B.chip, dispatch="static"))
    assert abs(st["dispatch_s"] - 3 * DISPATCH_US["static"] * 1e-6) < 1e-15  # one program launch per execution
    ku = _dac(dataclasses.replace(B.chip, kernel_us=3.0))
    assert ku["s"] > c0["s"] and ku["dispatch_s"] == 0.0 and ku["sync_s"] > 0


def test_dispatch_sensitivity_exact():
    M = ("drama-3", "cosyvoice2-0.5b", "moshiko-7b")
    a = {r["model"]: r for r in speech_bounds(B, M, verify=False)["rows"]}
    h = {r["model"]: r for r in speech_bounds(_with_dispatch(B, "host", None), M, verify=False)["rows"]}
    for m in M:
        d = a[m]["dispatch"]
        assert d["x_ideal"] == a[m]["x"] and abs(d["x_at"]["host"] - h[m]["x"]) < 1e-9
        assert abs(h[m]["dispatch"]["x_ideal"] - a[m]["x"]) < 1e-12        # chip dispatch stripped for the reference
        assert all(v > 0 for k, v in d["ops_per_audio_s"].items())
        assert d["us_eq"] > 0 and d["x_at"]["host"] > d["x_at"]["trace"] > d["x_ideal"]
    cv = a["cosyvoice2-0.5b"]["dispatch"]
    assert cv["ops_per_audio_s"]["speech_card"] > 0                       # flow + HiFT component now dispatched
    dr = a["drama-3"]["dispatch"]
    assert a["drama-3"]["realtime"] and dr["us_rt"] > 0
    at = speech_bounds(_with_dispatch(B, "trace", dr["us_rt"]), ("drama-3",), verify=False, dispatch=False)
    assert abs(at["rows"][0]["x"] - 1.0) < 1e-9                            # real-time limit is exact
    assert a["moshiko-7b"]["dispatch"]["us_rt"] == 0.0                     # not real time even when ideal


def test_api_dispatch_flag():
    body = {"scenario": {"model": "qwen3-8b", "mem_id": LP, "chip": {"name": "100T"}}, "models": ["cosyvoice2-0.5b"], "verify": False}
    r = api_insight_speech({**body, "dispatch": False})
    assert "dispatch" not in r["rows"][0]
    assert "dispatch" in api_insight_speech(body)["rows"][0]
    try:
        api_insight_speech({**body, "dispatch": "yes"})
        raise AssertionError
    except ApiError:
        pass
