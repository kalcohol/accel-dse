"""0.338: Qwen3-TTS 12 Hz builder; Carbon-X / OmniASR fp32 adversarial; OrcaCyber still pending."""
from __future__ import annotations

import unittest

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS
from accel_dse.core.vision import audio_ops
from accel_dse.core.qwen3_tts import FRAME_HZ, code_predictor_ops


IDS = (
    "qwen3-tts-0.6b-base",
    "qwen3-tts-1.7b-base",
    "qwen3-tts-0.6b-custom-voice",
    "qwen3-tts-1.7b-custom-voice",
    "qwen3-tts-1.7b-voice-design",
)


class TestQwen3Tts338(unittest.TestCase):
    def test_released_params_and_domain(self):
        ents = {e["id"]: e for e in list_models()}
        for mid in IDS:
            self.assertIn(mid, ents)
            self.assertEqual(ents[mid]["domain"], "tts")
            m = get_model(mid)
            self.assertEqual(m.param_check()["rel_err"], 0.0, mid)
            self.assertIsNotNone(m.qwen3_tts)
            self.assertEqual(m.qwen3_tts.cp_steps, 15)
            self.assertEqual(m.qwen3_tts.frame_hz, FRAME_HZ)

    def test_base_has_speaker_custom_does_not(self):
        self.assertGreater(get_model("qwen3-tts-0.6b-base").qwen3_tts.spk_params, 0)
        self.assertEqual(get_model("qwen3-tts-0.6b-custom-voice").qwen3_tts.spk_params, 0)

    def test_code_predictor_ops_and_speech_kpis(self):
        m = get_model("qwen3-tts-0.6b-base")
        ops = code_predictor_ops(m.qwen3_tts, 1)
        self.assertTrue(any(o.name.startswith("cp.prefill.") for o in ops))  # 0.339: HF prefill(2)+15 decode
        s = Scenario(model="qwen3-tts-0.6b-base", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g",
                     serving=Serving(batch=1, prompt=64, out_len=64))
        k = speech_kpis(s)
        self.assertEqual(k["kind"], "qwen3_tts")
        self.assertIn("code_predictor_s", k["split_frame"])
        self.assertLess(k["rtf"], 1.0)

    def test_speech_models_include_qwen3_tts(self):
        for mid in IDS:
            self.assertIn(mid, SPEECH_MODELS)
        self.assertEqual(len(SPEECH_MODELS), 31)  # 0.340 +ming-0.5b  # 0.339: +moshika +step-base/think +omni-3b


class TestAdversarial338(unittest.TestCase):
    def test_carbon_x_still_qwen3(self):
        m = get_model("carbon-x")
        self.assertEqual(m.param_check()["rel_err"], 0.0)
        self.assertEqual(m.hidden, 5120)
        self.assertEqual(len(m.layers), 40)
        s = Scenario(model="carbon-x", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g",
                     serving=Serving(batch=1, prompt=128, out_len=32, phase="prefill"))
        from accel_dse.core.evaluate import evaluate
        pre = evaluate(s)
        self.assertTrue(pre.fits)

    def test_omniasr_fp32_weights_held(self):
        for mid in ("omniasr-llm-300m-v2", "omniasr-llm-7b-v2"):
            a = get_model(mid).vision.audio
            self.assertEqual(a.w_bits, 32.0)
            self.assertEqual(a.w_fmt, "fp32")
            gemm = [o for o in audio_ops(a, 10.0, 1) if o.kind == "gemm"]
            self.assertTrue(gemm)
            self.assertTrue(all(o.w_bits == 32.0 and o.w_fmt == "fp32" for o in gemm))
        w = get_model("whisper-large-v3").vision.audio
        self.assertEqual(w.w_bits, 16.0)


if __name__ == "__main__":
    unittest.main()
