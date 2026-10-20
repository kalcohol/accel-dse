"""0.277: speech_streams speedup — identical results (lm_only KPIs, shared rerun memo, near search, memoised sub-chip,
memoised TTS step FLOPs / FireRed fixed components); progressive per-model UI requests."""
import math
import re
from pathlib import Path

from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import (MAX_STREAMS, SPEECH_CARD, SPEECH_MODELS, _lm_load, _max_streams, _scn,
                                           _stages, _with_slc, speech_streams)
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
BASE = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g")


def _ref_point(sb, mid, a):
    """The 0.276 algorithm verbatim: full KPIs for every rerun, no memo, doubling search from the linear prediction."""
    def ok(b):
        k = speech_kpis(_scn(sb, mid, a).replace("serving.batch", b), a)
        lm = {s: v for s, v in _stages(k, a)[0].items() if s not in SPEECH_CARD}
        return (bool(k["fits"]) and max(lm.values()) <= 1.0), lm, k
    ok1, lm1, k1 = ok(1)
    _, lm2, _ = ok(2)
    x1, x2 = max(lm1.values()), max(lm2.values())
    d = x2 - x1
    pred = (max(0, math.floor((1.0 - (x1 - d)) / d + 1e-9)) if d > 1e-12 else (MAX_STREAMS if x1 <= 1 else 0)) \
        if x1 <= 1 else 0
    n = _max_streams(lambda b: ok(b)[0], max(1, pred)) if ok1 else 0
    if n >= MAX_STREAMS:
        lim = "cap"
    else:
        _, lmn, kn = ok(n + 1)
        lim = "realtime" if max(lmn.values()) > 1.0 else "memory" if not kn["fits"] else "realtime"
    sc = int(k1.get("streams_realtime") or 0) if "speech_card" in _stages(k1, a)[0] else None
    return n, pred, lim, sc, x1


def test_streams_identical_to_reference_algorithm():
    models, gibs = ("moshiko-7b", "fun-cosyvoice3-0.5b-2512", "kimi-audio-7b-instruct"), (0.0, 4.0, 8.0)
    out = speech_streams(BASE, models, 10.0, gibs)
    for row, mid in zip(out["rows"], models):
        for c, g in zip(row["curve"], gibs):
            n, pred, lim, sc, x1 = _ref_point(_with_slc(BASE, g), mid, 10.0)
            assert (c["streams_lm"], c["streams_lm_pred"], c["lm_limit"], c["streams_speech_card"], c["x1"]) == \
                (n, min(pred, MAX_STREAMS), lim, sc, x1), (mid, g)


def test_lm_only_matches_full_kpis():
    """lm_only returns the LM-card loads and fit verdict by the same expressions — exact equality, every model."""
    for mid in SPEECH_MODELS:
        for g in (0.0, 2.0):
            for b in (1, 5):
                s = _scn(_with_slc(BASE, g), mid, 10.0).replace("serving.batch", b)
                full, lo = speech_kpis(s, 10.0), speech_kpis(s, 10.0, lm_only=True)
                assert _lm_load(full, 10.0) == _lm_load(lo, 10.0) and full["fits"] == lo["fits"], (mid, g, b)


def test_near_search_exact():
    for n in (0, 1, 2, 3, 17, 100, 511, 1023, 1024, 5000):
        for g in (1, 3, 64, 700, 1024):
            memo = {}
            assert _max_streams(lambda b: b <= n, g, memo, near=True) == min(n, MAX_STREAMS)
            assert all(v == (b <= n) for b, v in memo.items())


def test_ui_progressive_and_model_lists():
    js = W.read()                                      # 0.286: parsed literals / function body
    lists = {name: [v for k, v in W.literal(js, name) if k == "str"] for name in ("SPS_OUT", "SPS_ASR")}
    assert set(lists["SPS_OUT"]) | set(lists["SPS_ASR"]) == set(SPEECH_MODELS)
    assert not set(lists["SPS_OUT"]) & set(lists["SPS_ASR"])
    fn = W.functions(js)[W.snapshot_request(js, "/api/insight_speech_streams")["fn"]]
    assert W.has(fn, "const run = ++spsRun") and W.has(fn, "v0 = S.ver")          # supersede token + scenario version
    after = fn[W.index(fn, "req('insight_speech_streams'"):]
    assert W.has(after, "run !== spsRun) return") and W.has(after, "staleTag(v0)")   # checked after each awaited row
