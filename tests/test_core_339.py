"""0.339: Qwen3-TTS code-predictor prefill fix; speech freshness (Omni-3B, Step Base/Think, moshika); OrcaCyber still pending."""
from __future__ import annotations

import unittest

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.qwen3_tts import code_predictor_ops
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS

NEW = (
    "qwen2.5-omni-3b",
    "step-audio-2-mini-base",
    "step-audio-2-mini-think",
    "moshika-7b",
)
CHIP = CHIPS["100T"]
MEM = "lpddr5x_4x64_8533_16g"


class TestQwen3TtsPredictor339(unittest.TestCase):
    def test_prefill_then_cp_steps_heads(self):
        q = get_model("qwen3-tts-0.6b-base").qwen3_tts
        ops = code_predictor_ops(q, 1)
        self.assertTrue(any(o.name.startswith("cp.prefill.") for o in ops))
        heads = [o for o in ops if o.name.endswith(".head") and o.kind == "gemm"]
        self.assertEqual(len(heads), q.cp_steps)  # 15 = 1 prefill + 14 decode
        self.assertEqual(sum(1 for o in ops if o.name.endswith(".attn.qk") and o.name.startswith("cp.prefill.")),
                         q.cp_layers)

    def test_speech_kpis_still_realtime_on_100t(self):
        k = speech_kpis(Scenario(model="qwen3-tts-0.6b-base", chip=CHIP, mem_id=MEM,
                                 serving=Serving(batch=1, prompt=64, out_len=64)))
        self.assertEqual(k["kind"], "qwen3_tts")
        self.assertIn("code_predictor_s", k["split_frame"])
        self.assertLess(k["rtf"], 1.0)


class TestSpeechFreshness339(unittest.TestCase):
    def test_new_models_param_check_and_domain(self):
        ents = {e["id"]: e for e in list_models()}
        for mid in NEW:
            self.assertIn(mid, ents)
            self.assertEqual(ents[mid]["domain"], "omni")
            self.assertEqual(get_model(mid).param_check()["rel_err"], 0.0, mid)
            self.assertIn(mid, SPEECH_MODELS)

    def test_omni3b_talker_token2wav(self):
        m = get_model("qwen2.5-omni-3b")
        self.assertIsNotNone(m.omni)
        k = speech_kpis(Scenario(model="qwen2.5-omni-3b", chip=CHIP, mem_id=MEM,
                                 serving=Serving(batch=1, prompt=64, out_len=64)))
        self.assertIsNotNone(k.get("rtf"))

    def test_step_base_think_token2wav(self):
        for mid in ("step-audio-2-mini-base", "step-audio-2-mini-think"):
            k = speech_kpis(Scenario(model=mid, chip=CHIP, mem_id=MEM,
                                     serving=Serving(batch=1, prompt=64, out_len=64)))
            self.assertIn("flow_s", k)

    def test_moshika_matches_moshiko_dims(self):
        ka, ko = get_model("moshika-7b"), get_model("moshiko-7b")
        self.assertEqual(ka.hidden, ko.hidden)
        self.assertEqual(len(ka.layers), len(ko.layers))
        self.assertEqual(ka.param_check()["ours"], ko.param_check()["ours"])
        self.assertIsNotNone(ka.moshi)

    def test_speech_count(self):
        self.assertEqual(len(SPEECH_MODELS), 33)  # 0.341  # 0.340 +ming-0.5b


class TestDocs339(unittest.TestCase):
    def test_changelog_and_no_ui_patch(self):
        from pathlib import Path
        cl = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
        self.assertIn("## [0.339.0]", cl)
        sec = cl.split("## [0.339.0]")[1].split("## [")[0]
        self.assertIn("UI patch not applied", sec)


if __name__ == "__main__":
    unittest.main()
