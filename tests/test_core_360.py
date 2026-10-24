"""0.360: speech siblings whose public configs drop into existing builders.

Whisper .en (vocab 51864), Distil-Whisper (fewer decoder layers, header dtype),
Qwen3-ASR-0.6B, Qwen2-Audio-7B, Step-Audio-R1.1 (no token2wav), Qwen3-Omni
Thinker-only, Kimi-Audio-7B. Shapes and dtypes come from the release headers.
"""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"

# (id, params, audio_or_vision_audio, layers, hidden, vocab, weight fmt, domain, family, coverage)
ROWS = (
    ("qwen3-asr-0.6b", 596_049_920, 186_376_192, 28, 1024, 151_936, "bf16", "asr", "Qwen3-ASR", "full"),
    ("qwen2-audio-7b", 7_754_878_976, 642_215_936, 32, 4096, 156_032, "bf16", "asr", "Qwen2-Audio", "full"),
    ("whisper-tiny.en", 37_760_256, 8_208_384, 4, 384, 51_864, "fp32", "asr", "Whisper", "full"),
    ("whisper-base.en", 72_593_408, 20_590_592, 6, 512, 51_864, "fp32", "asr", "Whisper", "full"),
    ("whisper-small.en", 241_734_144, 88_154_112, 12, 768, 51_864, "fp32", "asr", "Whisper", "full"),
    ("whisper-medium.en", 763_856_896, 307_216_384, 24, 1024, 51_864, "fp32", "asr", "Whisper", "full"),
    ("distil-whisper-small.en", 166_132_224, 88_154_112, 4, 768, 51_864, "fp16", "asr", "Distil-Whisper", "full"),
    ("distil-whisper-medium.en", 394_375_168, 307_216_384, 2, 1024, 51_864, "fp16", "asr", "Distil-Whisper", "full"),
    ("distil-whisper-large-v2", 756_220_160, 636_784_640, 2, 1280, 51_865, "fp16", "asr", "Distil-Whisper", "full"),
    ("distil-whisper-large-v3", 756_405_760, 636_968_960, 2, 1280, 51_866, "fp16", "asr", "Distil-Whisper", "full"),
    ("distil-whisper-large-v3.5", 756_405_760, 636_968_960, 2, 1280, 51_866, "fp32", "asr", "Distil-Whisper", "full"),
    ("step-audio-2-r1.1", 32_832_033_792, 654_999_808, 64, 5120, 158_720, "bf16", "asr", "Step-Audio-2", "full"),
    ("qwen3-omni-30b-a3b-thinking", 30_532_646_912, 647_927_168, 48, 2048, 152_064, "bf16", "omni", "Qwen3-Omni", "partial"),
    ("qwen3-omni-30b-a3b-captioner", 30_532_646_912, 647_927_168, 48, 2048, 152_064, "bf16", "omni", "Qwen3-Omni", "partial"),
    ("kimi-audio-7b", 9_735_124_992, 668_178_432, 34, 3584, 168_448, "bf16", "omni", "Kimi-Audio", "partial"),
)
NEW = tuple(r[0] for r in ROWS)


def _fmt(m):
    return dict(m.formats)["attn"].fmt


def test_release_shapes_dtype_and_coverage():
    ents = {e["id"]: e for e in list_models()}
    assert len(SPEECH_MODELS) == 55
    for mid, params, audio, layers, hidden, vocab, fmt, domain, family, cov in ROWS:
        m = get_model(mid)
        e = ents[mid]
        assert m.param_check()["rel_err"] == 0.0, mid
        assert m.params() == params == m.release_params, mid
        assert m.vision.audio.params == audio, mid
        assert m.n_layers == layers and m.hidden == hidden and m.vocab == vocab, mid
        assert _fmt(m) == fmt, (mid, _fmt(m))
        assert e["domain"] == domain and e["family"] == family and e["coverage"] == cov, mid
        assert mid in SPEECH_MODELS
        if m.enc_ctx:
            assert m.enc_ctx == 1500 and m.tie_embeddings
            assert sum(L.cross is not None for L in m.layers) == layers
            assert m.vision_params == audio
        else:
            assert m.enc_ctx == 0


def test_whisper_en_is_one_vocab_row():
    """English checkpoints share the multilingual encoder; the tied embedding loses one row (51865 → 51864)."""
    for en, multi in (("whisper-tiny.en", "whisper-tiny"), ("whisper-base.en", "whisper-base"),
                      ("whisper-small.en", "whisper-small"), ("whisper-medium.en", "whisper-medium")):
        a, b = get_model(en), get_model(multi)
        assert b.vocab - a.vocab == 1 and a.vocab == 51864
        assert b.params() - a.params() == a.hidden
        assert a.vision_params == b.vision_params and a.n_layers == b.n_layers


def test_same_shape_siblings_and_thinker_only():
    base, inst = get_model("qwen2-audio-7b"), get_model("qwen2-audio-7b-instruct")
    assert base.params() == inst.params() and base.vision_params == inst.vision_params
    assert base.tie_embeddings is False and _fmt(base) == "bf16"
    kb, ki = get_model("kimi-audio-7b"), get_model("kimi-audio-7b-instruct")
    assert kb.params() == ki.params() and kb.vision_params == ki.vision_params
    assert kb.kimi.dit_params == ki.kimi.dit_params == 1_584_023_120
    assert kb.kimi.voc_params == 241_118_777 and kb.kimi.depth == 16 and kb.kimi.token_rate == 12.5
    assert any("GLM-4-Voice" in r for r in kb.coverage_reasons) and len(kb.coverage_reasons) == 1
    think, cap, omni = (get_model(i) for i in
                        ("qwen3-omni-30b-a3b-thinking", "qwen3-omni-30b-a3b-captioner", "qwen3-omni-30b-a3b"))
    assert think.qwen3_omni is None and cap.qwen3_omni is None and omni.qwen3_omni is not None
    assert think.params() == cap.params() == omni.params()
    assert think.vision.audio.params == cap.vision.audio.params == omni.vision.audio.params
    assert think.vision_params == cap.vision_params == 1_186_558_576
    assert ents_family()
    rel = load_release("stepfun-ai/Step-Audio-R1.1")
    assert "token2wav" not in rel
    step = get_model("step-audio-2-r1.1")
    assert step.kimi is None and step.qwen3_omni is None and step.tie_embeddings is False


def ents_family():
    ents = {e["id"]: e for e in list_models()}
    assert ents["qwen3-omni-30b-a3b"]["family"] == "Qwen3-Omni"
    assert all(len(get_model(i).coverage_reasons) == 1 and "thinker.visual" in get_model(i).coverage_reasons[0]
               for i in ("qwen3-omni-30b-a3b-thinking", "qwen3-omni-30b-a3b-captioner"))
    return True


def test_speech_kinds():
    def kind(mid):
        s = Scenario(model=mid, chip=CHIPS["1P"], mem_id=LP,
                     serving=Serving(batch=1, prompt=64, out_len=16, audio_s=2.0))
        return speech_kpis(s, 1.0)["kind"]
    for mid in ("qwen3-asr-0.6b", "whisper-tiny.en", "distil-whisper-large-v3.5",
                "step-audio-2-r1.1", "qwen3-omni-30b-a3b-thinking"):
        assert kind(mid) == "asr", mid
    assert kind("kimi-audio-7b") == "kimi"


def test_changelog_360():
    assert "## [0.360.0]" in CL
    sec = CL.split("## [0.360.0]")[1].split("## [")[0]
    for needle in ("Distil-Whisper", "Qwen3-ASR-0.6B", "Step-Audio-R1.1", "Kimi-Audio-7B", "CosyVoice-300M"):
        assert needle in sec
