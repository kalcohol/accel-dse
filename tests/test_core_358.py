"""0.358: Whisper encoder-decoder params() includes encoder (637M) + decoder.

Bug: whisper-large-v3 params() reported 906M (decoder only), expected 1.54B (encoder + decoder).
Fix: Model.params() now includes vision_params for encoder-decoder models (enc_ctx > 0)."""
from __future__ import annotations

import unittest

from accel_dse.core.catalog import get_model


class TestWhisperParamCount358(unittest.TestCase):
    def test_whisper_large_v3_params_include_encoder(self):
        m = get_model("whisper-large-v3")
        # encoder-decoder: params() = decoder + encoder
        self.assertEqual(m.enc_ctx, 1500, "whisper-large-v3 is encoder-decoder")
        self.assertEqual(m.vision_params, 636968960, "encoder params")
        # Before 0.358: params() = 906521600 (decoder only)
        # After  0.358: params() = 906521600 + 636968960 = 1543490560
        self.assertEqual(m.params(), 1543490560, "params() includes encoder + decoder")
        # release_params should match
        self.assertEqual(m.release_params, 1543490560)
        pc = m.param_check()
        self.assertEqual(pc["rel_err"], 0.0, "params match release")

    def test_whisper_turbo_params_include_encoder(self):
        m = get_model("whisper-large-v3-turbo")
        self.assertEqual(m.enc_ctx, 1500, "whisper-turbo is encoder-decoder")
        self.assertEqual(m.vision_params, 636968960, "encoder params (same as v3)")
        # Before 0.358: params() = 171909120 (decoder only)
        # After  0.358: params() = 171909120 + 636968960 = 808878080
        self.assertEqual(m.params(), 808878080, "params() includes encoder + decoder")
        self.assertEqual(m.release_params, 808878080)
        pc = m.param_check()
        self.assertEqual(pc["rel_err"], 0.0, "params match release")

    def test_other_asr_models_unaffected(self):
        # Qwen2-Audio, Qwen3-ASR: LLM + audio tower, encoder already in params_llm
        for mid in ("qwen2-audio-7b-instruct", "qwen3-asr-1.7b"):
            m = get_model(mid)
            self.assertEqual(m.enc_ctx, 0, f"{mid} is not encoder-decoder")
            pc = m.param_check()
            self.assertEqual(pc["rel_err"], 0.0, f"{mid} params match release")


if __name__ == "__main__":
    unittest.main()
