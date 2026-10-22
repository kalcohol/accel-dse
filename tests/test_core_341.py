"""0.341: adversarial Ming-3.0-tts-0.5B DiTAR; Ming-omni-tts-0.5B via tts_spec; Qwen3-Omni/MOSS/Orca/pocket pending."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS
from accel_dse.core.tts import tts_head_ops

CHIP, MEM = CHIPS["100T"], "lpddr5x_4x64_8533_16g"


def test_ming30_05b_adversarial_ditar():
    m = get_model("ming-3.0-tts-0.5b")
    assert m.param_check()["rel_err"] == 0.0
    ts = m.tts
    assert ts.flow_depth == 8 and ts.flow_steps == 10 and ts.cfg_scale == 2.0
    assert ts.rate == 44100 and abs(ts.s_per_step - 0.08) < 1e-9
    ops = tts_head_ops(ts, 1, 0)
    assert any("flow0." in o.name for o in ops) and any("flow9." in o.name for o in ops)
    sc = Scenario(model="ming-3.0-tts-0.5b", chip=CHIP, mem_id=MEM,
                  serving=Serving(batch=1, prompt=64, out_len=64))
    assert evaluate(sc).fits
    k = speech_kpis(sc)
    assert k["kind"] == "tts" and k["rtf"] < 1.0


def test_ming_omni_tts_05b():
    m = get_model("ming-omni-tts-0.5b")
    assert m.param_check()["rel_err"] == 0.0
    assert m.tts.patch == 4 and m.tts.history == 32
    assert abs(m.tts.s_per_step - 0.32) < 1e-9
    ents = {e["id"]: e for e in list_models()}
    assert ents["ming-omni-tts-0.5b"]["domain"] == "tts"
    assert ents["ming-omni-tts-0.5b"]["coverage"] == "full"
    assert "ming-omni-tts-0.5b" in SPEECH_MODELS
    sc = Scenario(model="ming-omni-tts-0.5b", chip=CHIP, mem_id=MEM,
                  serving=Serving(batch=1, prompt=64, out_len=32))
    assert evaluate(sc).fits
    assert speech_kpis(sc)["kind"] == "tts"


def test_speech_count_341():
    assert len(SPEECH_MODELS) == 33


def test_changelog_no_ui_patch_341():
    cl = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert "## [0.341.0]" in cl
    sec = cl.split("## [0.341.0]")[1].split("## [")[0]
    assert "UI patch not applied" in sec
