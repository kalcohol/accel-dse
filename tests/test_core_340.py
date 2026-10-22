"""0.340: adversarial 0.339 speech adds + Qwen3-TTS prefill; Ming-3.0-tts-0.5B via tts_spec; OrcaCyber still pending."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.qwen3_tts import code_predictor_ops
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS

CHIP, MEM = CHIPS["100T"], "lpddr5x_4x64_8533_16g"
NEW339 = ("qwen2.5-omni-3b", "step-audio-2-mini-base", "step-audio-2-mini-think", "moshika-7b")


def test_adversarial_339_param_eval_speech():
    ents = {e["id"]: e for e in list_models()}
    for mid in NEW339:
        assert get_model(mid).param_check()["rel_err"] == 0.0, mid
        assert ents[mid]["domain"] == "omni"
        sc = Scenario(model=mid, chip=CHIP, mem_id=MEM, serving=Serving(batch=1, prompt=64, out_len=32))
        assert evaluate(sc).fits
        assert speech_kpis(sc).get("rtf") is not None


def test_qwen3_tts_prefill_held():
    q = get_model("qwen3-tts-0.6b-base").qwen3_tts
    ops = code_predictor_ops(q, 1)
    assert any(o.name.startswith("cp.prefill.") for o in ops)
    assert sum(1 for o in ops if o.name.endswith(".head") and o.kind == "gemm") == 15
    assert next(o for o in ops if o.name == "cp.prefill.0.q").m == 2


def test_ming_05b_catalog_and_tts():
    m = get_model("ming-3.0-tts-0.5b")
    assert m.param_check()["rel_err"] == 0.0
    assert m.tts is not None
    assert m.tts.flow_depth == 8 and m.tts.flow_steps == 10 and m.tts.cfg_scale == 2.0
    assert m.hidden == 896 and len(m.layers) == 24
    ents = {e["id"]: e for e in list_models()}
    assert ents["ming-3.0-tts-0.5b"]["domain"] == "tts"
    assert ents["ming-3.0-tts-0.5b"]["coverage"] == "full"
    assert "ming-3.0-tts-0.5b" in SPEECH_MODELS


def test_ming_05b_speech_kpis():
    k = speech_kpis(Scenario(model="ming-3.0-tts-0.5b", chip=CHIP, mem_id=MEM,
                             serving=Serving(batch=1, prompt=64, out_len=64)))
    assert k["kind"] == "tts"
    assert k["rtf"] < 1.0


def test_ming_tiny_still_ok():
    assert get_model("ming-3.0-tts-tiny").param_check()["rel_err"] == 0.0
    assert get_model("ming-3.0-tts-tiny").tts is not None


def test_speech_count_340():
    assert len(SPEECH_MODELS) == 34


def test_changelog_no_ui_patch_340():
    cl = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert "## [0.340.0]" in cl
    sec = cl.split("## [0.340.0]")[1].split("## [")[0]
    assert "UI patch not applied" in sec
