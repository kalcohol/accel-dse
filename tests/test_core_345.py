"""0.345: clean Qwen3-Omni dual-MoE + code2wav builder; NPU skip; no visual UI."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.omni import omni_spec
from accel_dse.core.qwen3_omni import FRAME_HZ, code2wav_ops, code_predictor_ops, qwen3_omni_spec, talker_ops
from accel_dse.core.qwen3_tts import qwen3_tts_spec
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"


def test_qwen3_omni_builder_not_thin_wrap():
    m = get_model("qwen3-omni-30b-a3b")
    assert m.qwen3_omni is not None
    assert m.omni is None and m.qwen3_tts is None
    o = m.qwen3_omni
    assert o.t_experts == 128 and o.t_topk == 6 and o.t_shared_inter == 768
    assert o.cp_steps == 15 and o.total_upsample == 1920 and o.frame_hz == FRAME_HZ
    assert o.talker_params > 3e9 and o.code2wav_params > 2e8
    assert m.release_params == m.params()
    ids = {e["id"] for e in list_models()}
    assert "qwen3-omni-30b-a3b" in ids


def test_qwen3_omni_gates_other_specs():
    from accel_dse.core.model import load_release
    rel = load_release("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    cfg = rel["config"]
    assert omni_spec(cfg) is None
    assert qwen3_tts_spec(cfg, rel) is None
    assert qwen3_omni_spec(cfg, rel) is not None
    # other omni models unchanged
    assert get_model("qwen2.5-omni-7b").omni is not None
    assert get_model("qwen2.5-omni-7b").qwen3_omni is None


def test_qwen3_omni_ops_and_speech_kpis():
    m = get_model("qwen3-omni-30b-a3b")
    o = m.qwen3_omni
    assert len(talker_ops(o, 1, 1, 128)) > 50
    assert len(code_predictor_ops(o, 1)) > 50
    assert len(code2wav_ops(o, 1, 1)) > 20
    s = Scenario(model="qwen3-omni-30b-a3b", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    k = speech_kpis(s, 2.0)
    assert k["kind"] == "qwen3_omni"
    assert k["cp_steps"] == 15
    assert "talker_s" in k["split_frame"] and "code2wav_s" in k["split_frame"]
    assert k["frame_s"] > 0 and k["first_audio_s"] > 0


def test_qwen3_omni_aut_audio():
    m = get_model("qwen3-omni-30b-a3b")
    assert m.vision is not None and m.vision.audio is not None
    assert m.vision.audio.kind == "aut"
    assert m.vision.extra > 0  # ViT storage-only surplus


def test_changelog_345():
    assert "## [0.345.0]" in CL
    sec = CL.split("## [0.345.0]")[1].split("## [")[0]
    assert "Qwen3-Omni" in sec or "qwen3-omni" in sec
    assert "code2wav" in sec.lower()
    assert "NPU" in sec or "npu" in sec.lower() or "skip" in sec.lower()
