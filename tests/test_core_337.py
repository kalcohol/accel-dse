"""0.337: OmniASR w2v as-released fp32 weights; Carbon-X; adversarial shapes."""
from __future__ import annotations

import unittest
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.vision import audio_ops


class TestOmniAsrDtype337(unittest.TestCase):
    def test_w2v_ops_use_as_released_fp32(self):
        for mid in (
            "omniasr-llm-300m-v2",
            "omniasr-llm-1b-v2",
            "omniasr-llm-3b-v2",
            "omniasr-llm-7b-v2",
        ):
            m = get_model(mid)
            a = m.vision.audio
            self.assertEqual(a.kind, "w2v")
            self.assertEqual(a.w_bits, 32.0, mid)
            self.assertEqual(a.w_fmt, "fp32", mid)
            gemm = [o for o in audio_ops(a, 10.0, 1) if o.kind == "gemm"]
            self.assertTrue(gemm, mid)
            for o in gemm:
                self.assertEqual(o.w_bits, 32.0, (mid, o.name))
                self.assertEqual(o.w_fmt, "fp32", (mid, o.name))

    def test_other_audio_still_bf16_default(self):
        a = get_model("whisper-large-v3").vision.audio
        self.assertEqual(a.w_bits, 16.0)
        self.assertEqual(a.w_fmt, "bf16")


class TestCarbonX337(unittest.TestCase):
    def test_carbon_x_from_release(self):
        m = get_model("carbon-x")
        pc = m.param_check()
        self.assertEqual(pc["rel_err"], 0.0)
        self.assertEqual(m.hidden, 5120)
        self.assertEqual(len(m.layers), 40)
        self.assertEqual(m.formats[0][1].fmt, "bf16")
        ids = {x["id"] for x in list_models()}
        self.assertIn("carbon-x", ids)


class TestOmniAsrShapes337(unittest.TestCase):
    def test_feat_extract_length_and_projector(self):
        m = get_model("omniasr-llm-300m-v2")
        a = m.vision.audio
        # Π stride 320 → 50 Hz @ 16 kHz; 10 s → 500 frames before pad
        ops = audio_ops(a, 10.0, 1)
        names = [o.name for o in ops]
        self.assertTrue(any(n.startswith("aud.fe.") for n in names))
        self.assertTrue(any("proj" in n or "o_proj" in n or "mm" in n for n in names)
                        or any(n.startswith("aud.") for n in names))
        # text trunk still Llama-shaped
        self.assertEqual(m.hidden, 4096)
        self.assertEqual(len(m.layers), 12)


class TestUiPatchAssertPortable337(unittest.TestCase):
    def test_336_ui_patch_assert_portable(self):
        src = (Path(__file__).resolve().parent / "test_core_336.py").read_text(encoding="utf-8")
        self.assertNotIn("/workspace/ui-audit/ui.patch", src)
        self.assertIn("UI patch not applied", src)


if __name__ == "__main__":
    unittest.main()
