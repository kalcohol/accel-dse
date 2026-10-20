"""0.275: speech bottleneck panel (UI for /api/insight_speech), static build routing, scenario isolation."""
import re
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech_insight import _scn, speech_bounds
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"


def test_ui_panel_wired():
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: parsed (call site, body builder, progress label)
    ids = W.html_ids(html)
    for i in ("ins-sp-run", "ins-sp-tbl", "ins-sp-summary", "ins-sp-verify", "ins-sp-audio", "ins-sp-status"):
        assert i in ids, i
    c = [c for c in W.req_calls(js) if c["path"] == "/api/insight_speech"]
    assert [x["fn"] for x in c] == ["runInsightSpeech"] and c[0]["body"][:2] == [("id", "body"), ("punct", "(")]
    assert W.object_dict(W.literal(js, "STAGE"))["insight_speech"] == "语音瓶颈"


def test_static_routes_and_slow_lane():
    slow = W.list_values(W.read(W.SHIM_JS), "slowRoutes")      # 0.286: parsed
    assert {"/api/insight_speech", "/api/insight_slc"} <= slow
    st, payload = serve.respond("POST", "/api/insight_speech",
                                b'{"scenario": {"model": "qwen3-8b", "mem_id": "' + LP.encode() + b'", "chip": {"name": "100T"}},'
                                b' "models": ["ming-3.0-tts-tiny"], "verify": false}')
    assert st == 200 and b'"binding"' in payload
    assert serve.respond("POST", "/api/insight_speech", b"[1]")[0] == 400


def test_rows_ignore_ui_model_settings():
    """A draft model / PD / big batch on the UI's model must not reach the speech rows."""
    plain = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id=LP)
    busy = plain.replace("serving", Serving(batch=8, prompt=2048, ctx=4096, spec_k=3, draft_model="qwen3-0.6b"))
    s = _scn(busy, "moshiko-7b", 10.0)
    assert s.serving == Serving(batch=1, prompt=64, out_len=64) and s.chip == plain.chip
    a = speech_bounds(plain, ("moshiko-7b",), verify=False)["rows"][0]["x"]
    b = speech_bounds(busy, ("moshiko-7b",), verify=False)["rows"][0]["x"]
    assert a == b > 1.0


def test_verify_adds_to_existing_slc():
    chip = CHIPS["100T"].__class__(**{**CHIPS["100T"].__dict__, "slc_mib": 256.0, "slc_policy": "pin"})
    r = speech_bounds(Scenario(model="qwen3-8b", chip=chip, mem_id=LP), ("kimi-audio-7b-instruct",))["rows"][0]
    assert r["verify"]["slc_mib"] > 256 and r["verify"]["realtime"]
