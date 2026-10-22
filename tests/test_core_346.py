"""0.346: adversarial Qwen3-Omni (code2wav keys + Talker projection); MOSS-TTS Delay; no visual UI."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.moss_tts import FRAME_HZ, moss_audio_head_ops, moss_tts_spec
from accel_dse.core.omni import omni_spec
from accel_dse.core.qwen3_omni import code2wav_ops, talker_projection_ops
from accel_dse.core.qwen3_tts import qwen3_tts_spec
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"


def test_qwen3_omni_adversarial_code2wav_keys_and_projection():
    m = get_model("qwen3-omni-30b-a3b")
    o = m.qwen3_omni
    assert o is not None and o.c2_window == 72
    # streaming one frame attends sliding_window keys (not min(1, window)=1)
    ops = code2wav_ops(o, 1, frames=1)
    qk = [x for x in ops if x.name.endswith(".attn.qk")]
    assert qk and all(x.n == 72 for x in qk)
    # Thinker→Talker ResizeMLP once at setup
    proj = talker_projection_ops(o, 1, 128)
    assert [(x.name, x.k, x.n) for x in proj if x.kind == "gemm"] == [
        ("talker.proj.fc1", 2048, 2048), ("talker.proj.fc2", 2048, 1024)]
    s = Scenario(model="qwen3-omni-30b-a3b", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    k = speech_kpis(s, 2.0)
    assert k["kind"] == "qwen3_omni"
    assert "talker_prefill_s" in k["split_first"]
    assert k["split_first"]["talker_prefill_s"] > 0


def test_moss_tts_builder_and_gates():
    m = get_model("moss-tts")
    assert m.moss_tts is not None
    assert m.omni is None and m.qwen3_tts is None and m.qwen3_omni is None and m.moshi is None
    o = m.moss_tts
    assert o.n_vq == 32 and o.audio_vocab == 1024 and o.frame_hz == FRAME_HZ
    assert o.delay_frames == 32 and o.audio_head_params == 134348800
    assert m.release_params == m.params()
    assert abs(m.params() - m.release_params) == 0
    assert m.coverage == "partial"
    assert any("tokenizer" in r for r in m.coverage_reasons)
    from accel_dse.core.model import load_release
    rel = load_release("OpenMOSS-Team/MOSS-TTS")
    assert moss_tts_spec(rel["config"], rel) is not None
    assert omni_spec(rel["config"]) is None
    assert qwen3_tts_spec(rel["config"], rel) is None
    ids = {e["id"]: e for e in list_models()}
    assert ids["moss-tts"]["domain"] == "tts" and ids["moss-tts"]["param_err"] == 0.0


def test_moss_tts_speech_kpis_and_ops():
    assert len(moss_audio_head_ops(get_model("moss-tts").moss_tts, 1)) >= 3
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    k = speech_kpis(s, 2.0)
    assert k["kind"] == "moss_tts"
    assert k["n_vq"] == 32 and k["delay_frames"] == 32
    assert "backbone_s" in k["split_frame"] and "audio_heads_s" in k["split_frame"]
    assert k["frame_s"] > 0 and k["first_audio_s"] > k["ttft_s"]
    lo = speech_kpis(s, 2.0, lm_only=True)
    assert lo["kind"] == "moss_tts" and "frame" in lo["lm_stages"]


def test_moss_in_speech_models_and_no_break_others():
    assert "moss-tts" in SPEECH_MODELS
    assert len(SPEECH_MODELS) == 34
    assert get_model("qwen3-omni-30b-a3b").moss_tts is None
    assert get_model("qwen2.5-omni-7b").omni is not None
    assert get_model("moshiko-7b").moshi is not None


def test_changelog_346():
    assert "## [0.346.0]" in CL
    sec = CL.split("## [0.346.0]")[1].split("## [")[0]
    assert "Qwen3-Omni" in sec or "code2wav" in sec.lower()
    assert "MOSS" in sec or "moss" in sec.lower()
    assert "restyle" not in sec.lower() or "no visual" in sec.lower() or "未" in sec
