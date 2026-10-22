"""0.344: adversarial 0.343 UI; Qwen3-Omni not thin-wrapped; NPU still no public absolute; freshness."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import list_models
from accel_dse.core.omni import omni_spec
from accel_dse.core.qwen3_tts import qwen3_tts_spec
from tests import webstruct as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()

# Public HF Qwen/Qwen3-Omni-30B-A3B-Instruct config.json dims 「来源」 (fetched 2026-10-23 JST).
# Not enough to reuse omni_spec (token2wav DiT+BigVGAN) or qwen3_tts (model_type qwen3_tts).
_QWEN3_OMNI_MIN = {
    "model_type": "qwen3_omni_moe",
    "enable_audio_output": True,
    "thinker_config": {
        "text_config": {
            "hidden_size": 2048, "num_hidden_layers": 48, "num_experts": 128,
            "num_experts_per_tok": 8, "moe_intermediate_size": 768, "vocab_size": 152064,
        },
    },
    "talker_config": {
        "text_config": {
            "hidden_size": 1024, "num_hidden_layers": 20, "num_experts": 128,
            "num_experts_per_tok": 6, "moe_intermediate_size": 384, "vocab_size": 3072,
            "num_attention_heads": 16,
        },
        "code_predictor_config": {
            "model_type": "qwen3_omni_moe_talker_code_predictor",
            "hidden_size": 1024, "num_hidden_layers": 5, "num_code_groups": 16,
            "num_attention_heads": 16, "intermediate_size": 3072,
        },
    },
    "code2wav_config": {
        "hidden_size": 1024, "num_hidden_layers": 8, "num_attention_heads": 16,
        "num_quantizers": 16, "codebook_dim": 512, "decoder_dim": 1536,
        "sliding_window": 72, "upsample_rates": [8, 5, 4, 3],
    },
}


def test_row_flex_still_present_no_visual_tokens():
    # style.css checks must inline the path (test_core_286 lint)
    assert ".toolbar, .row" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "display: flex" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "flex-wrap: wrap" in (ROOT / "accel_dse/web/style.css").read_text()
    assert ".scroll:has(> table:empty)" in (ROOT / "accel_dse/web/style.css").read_text()
    assert "padding-bottom: 0" in (ROOT / "accel_dse/web/style.css").read_text().split(
        ".scroll:has(> table:empty)")[1].split("}")[0]
    assert "--fs-kpi" not in (ROOT / "accel_dse/web/style.css").read_text()
    assert "--ctl-h" not in (ROOT / "accel_dse/web/style.css").read_text()
    assert "visual harmonisation" not in (ROOT / "accel_dse/web/style.css").read_text()


def test_wl_kind_clear_else_branch():
    body = W.functions(W.APP_JS.read_text())["paintDomain"]
    assert W.has(body, "$('wl-kind').textContent = ''")
    # token stream ends with else $('wl-kind').textContent = '' (0.343 no-workload branch)
    assert ("id", "else") in body
    i = body.index(("id", "else"))
    assert ("str", "wl-kind") in body[i:i + 8]


def test_qwen3_omni_not_thin_wrapped():
    assert omni_spec(_QWEN3_OMNI_MIN) is None
    assert qwen3_tts_spec(_QWEN3_OMNI_MIN) is None
    ids = {e["id"] for e in list_models()}
    assert "qwen3-omni-30b-a3b" not in ids
    assert not any("qwen3-omni" in i for i in ids)
    assert "code2wav_config" in _QWEN3_OMNI_MIN and "token2wav_config" not in _QWEN3_OMNI_MIN
    c2 = _QWEN3_OMNI_MIN["code2wav_config"]
    assert c2["num_quantizers"] == 16 and c2["upsample_rates"] == [8, 5, 4, 3]
    talker = _QWEN3_OMNI_MIN["talker_config"]["text_config"]
    assert talker["num_experts"] == 128 and talker["num_experts_per_tok"] == 6


def test_changelog_344_adversarial_pending():
    assert "## [0.344.0]" in CL
    sec = CL.split("## [0.344.0]")[1].split("## [")[0]
    assert "0.343" in sec or ".row" in sec
    assert "Qwen3-Omni" in sec
    assert "code2wav" in sec.lower() or "thin" in sec.lower() or "pending" in sec.lower()
    assert "visual" in sec.lower()
    assert "NPU" in sec or "npu" in sec
