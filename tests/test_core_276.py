"""0.276: real-time streams per card vs SLC size (speech_insight.speech_streams, /api/insight_speech_streams)."""
import re
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import MAX_STREAMS, _lm_ok, _max_streams, _scn, _stages, _with_slc, speech_streams

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
BASE = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id=LP)


def test_max_streams_search_exact():
    for n in (0, 1, 2, 3, 17, 100, 511, 512, 1023, 1024, 5000):
        for g in (1, 3, 64, 700, 1024):
            calls = []
            r = _max_streams(lambda b: calls.append(b) or b <= n, g)
            assert r == min(n, MAX_STREAMS), (n, g, r)
            assert len(set(calls)) <= 25


def test_moshi_streams_vs_slc_verified():
    """Moshi needs ~1.3 GB pinned for one stream (0.274); the curve is engine reruns: b streams real time, b + 1 not."""
    out = speech_streams(BASE, ("moshiko-7b",), 10.0, (0.0, 2.0, 8.0))
    c = {x["slc_gib"]: x for x in out["rows"][0]["curve"]}
    assert c[0.0]["streams"] == 0 and c[2.0]["streams"] >= 1 and c[8.0]["streams"] > c[2.0]["streams"]
    assert out["rows"][0]["first_realtime_slc_gib"] == 2.0
    n = c[8.0]["streams_lm"]
    sb = _with_slc(BASE, 8.0)
    assert _lm_ok(sb, "moshiko-7b", 10.0, n)[0] and not _lm_ok(sb, "moshiko-7b", 10.0, n + 1)[0]
    assert c[8.0]["lm_limit"] == "realtime" and c[8.0]["cards"] == 1 and c[8.0]["speech_cards"] == 0
    assert abs(c[8.0]["streams_lm_pred"] - n) <= 1          # memory-bound regime: the linear read-out holds


def test_speech_card_sizing():
    """Kimi: the batch-1 detokenizer card carries floor(1 / x_sc(1)) streams; speech cards = ceil(N_lm / N_sc)."""
    out = speech_streams(BASE, ("kimi-audio-7b-instruct",), 10.0, (4.0,))
    c = out["rows"][0]["curve"][0]
    k = speech_kpis(_scn(_with_slc(BASE, 4.0), "kimi-audio-7b-instruct", 10.0), 10.0)
    assert c["streams_speech_card"] == k["streams_realtime"] == int(1 / _stages(k, 10.0)[0]["speech_card"])
    assert c["speech_cards"] == -(-c["streams_lm"] // c["streams_speech_card"])
    assert abs(c["streams_per_card"] - c["streams_lm"] / (1 + c["speech_cards"])) < 1e-12
    assert c["streams_lm"] > 0


def test_api_route_ui_and_shim():
    assert "/api/insight_speech_streams" in serve.POST_ROUTES
    out = api.api_insight_speech_streams({"scenario": {"model": "qwen3-8b", "mem_id": LP, "chip": {"name": "100T"}},
                                          "models": ["whisper-large-v3-turbo"], "slc_gib": [0]})
    assert out["rows"][0]["curve"][0]["streams"] > 0 and "hash" in out
    for bad in ({"slc_gib": []}, {"slc_gib": [-1]}, {"slc_gib": [True]}, {"models": "x"}, {"audio_s": 0}):
        try:
            api.api_insight_speech_streams({"scenario": {"model": "qwen3-8b", "mem_id": LP, "chip": {"name": "100T"}}, **bad})
        except api.ApiError:
            continue
        raise AssertionError(bad)
    js = (ROOT / "accel_dse/web/app.js").read_text()
    html = (ROOT / "accel_dse/web/index.html").read_text()
    assert "req('insight_speech_streams', '/api/insight_speech_streams', body(" in js
    for i in ("ins-sps-run", "ins-sps-gib", "ins-sps-tbl", "ins-sps-asr"):
        assert f'id="{i}"' in html
    shim = (ROOT / "web-static/pyodide-shim.js").read_text()
    assert "/api/insight_speech_streams" in re.search(r"slowRoutes \|\| \[([^\]]*)\]", shim).group(1)


def _fr(gib):
    return speech_kpis(_scn(_with_slc(BASE, gib), "fireredtts3-base", 10.0), 10.0)


def test_firered_dit_pin_reuses_rr_rule():
    """The fp32 DiT (flow_params · 4 B, read 10× per step) is the 0.269 re-read class: SRAM first, then SLC; each DRAM
    byte the SLC removes costs the same seconds (memory-bound step) — the TPOT drop is linear in the pinned bytes."""
    k0, k5, k1, k2 = (_fr(g) for g in (0.0, 0.5, 1.0, 2.0))
    p0 = k0["dit_pin"]
    assert abs(p0["dit_GB"] - 293723200 * 4 / 1e9) < 1e-9 and p0["reads_per_step"] == 10
    assert p0["slc_GB"] == 0 and 0 < p0["sram_pinned_GB"] < 0.1 and p0["miss"] > 0.9
    assert abs(p0["dram_GB_per_step"] - p0["miss"] * p0["dit_GB"] * 10) < 1e-9
    assert k2["dit_pin"]["miss"] == 0 and k2["dit_pin"]["dram_GB_per_step"] == 0
    assert 1.0 < p0["slc_full_GiB"] < 1.1
    s_a = (k0["tpot_s"] - k5["tpot_s"]) / (p0["dram_GB_per_step"] - k5["dit_pin"]["dram_GB_per_step"])
    s_b = (k5["tpot_s"] - k1["tpot_s"]) / (k5["dit_pin"]["dram_GB_per_step"] - k1["dit_pin"]["dram_GB_per_step"])
    assert abs(s_a / s_b - 1) < 0.02
    assert k0["tpot_s"] > k5["tpot_s"] > k1["tpot_s"] > k2["tpot_s"]


def test_firered_bf16_recast_option():
    """autocast(bf16) over fp32 backbone weights: +4 B per non-embedding backbone param per step, not in tpot_s."""
    from accel_dse.core.catalog import get_model
    k = _fr(0.0)
    r, m = k["bf16_recast"], get_model("fireredtts3-base")
    n = m.params() - (m.embed_params() if callable(m.embed_params) else m.embed_params)
    assert r["params"] == n and abs(r["extra_GB_per_step"] - 4 * n / 1e9) < 1e-9 and r["counted"] is False
    assert abs(r["tpot_s"] - k["tpot_s"] - r["extra_s_per_step"]) < 1e-12 and r["rtf"] > k["rtf"]
