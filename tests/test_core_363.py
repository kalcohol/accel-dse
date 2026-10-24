"""0.363: public 2025-2026 LLM / MoE / Wan2.2-TI2V-5B releases on existing builders.

Shapes and dtypes come from the release headers. Nemotron-H configs that only
publish hybrid_override_pattern expand with the Lightning alphabet.
"""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.domain import resolve_workload
from accel_dse.core.model import _nemotron_block_types
from accel_dse.core.pipeline import pipeline_for
from accel_dse.core.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()

# id, params, mtp, layers, hidden, vocab, weight fmt, bits (6 dp), act, domain, family, moe
ROWS = (
    ("qwen3-14b", 14_768_307_200, 0, 40, 5120, 151_936, "bf16", 16.0, "bf16", "llm", "Qwen3 稠密", False),
    ("qwen3-14b-fp8", 14_768_307_200, 0, 40, 5120, 151_936, "fp8", 8.000977, "fp8", "llm", "Qwen3 稠密", False),
    ("qwen3-coder-480b-a35b", 480_154_875_392, 0, 62, 6144, 151_936, "bf16", 16.0, "bf16", "llm", "Qwen3-Coder", True),
    ("qwen3-coder-480b-a35b-fp8", 480_154_875_392, 0, 62, 6144, 151_936, "fp8", 8.000977, "fp8", "llm", "Qwen3-Coder", True),
    # 0.365: attn bits are the fp8 bucket alone (block scales), not the bf16 conv / in_proj_ba smear
    ("qwen3-coder-next-fp8", 79_674_391_296, 0, 48, 2048, 151_936, "fp8", 8.000977, "fp8", "llm", "Qwen3-Next", True),
    ("qwen3.8-27b-fp8", 26_895_998_464, 424_699_392, 64, 5120, 248_320, "fp8", 8.000977, "fp8", "vlm", "Qwen3.8", False),
    ("nemotron-nano-9b-v2", 8_888_227_328, 0, 56, 4480, 131_072, "bf16", 16.0, "bf16", "llm", "Nemotron-Nano", False),
    ("nemotron-3-nano-4b-bf16", 3_973_556_832, 0, 42, 3136, 131_072, "bf16", 16.0, "bf16", "llm", "Nemotron-3", False),
    ("nemotron-3-ultra-550b-a55b-bf16", 549_308_993_536, 11_215_610_368, 108, 8192, 131_072, "bf16", 16.0, "bf16",
     "llm", "Nemotron-3", True),
    ("hy3", 295_033_543_488, 3_752_612_288, 80, 4096, 120_832, "bf16", 16.0, "bf16", "llm", "Hy3", True),
    ("glm-4.7-flash", 29_943_393_920, 1_278_094_656, 47, 2048, 154_880, "bf16", 16.0, "bf16", "llm", "GLM-4.7", True),
)


def _fmt(m):
    return dict(m.formats)["attn"]


def test_release_shapes_dtype_and_family():
    ents = {e["id"]: e for e in list_models()}
    for mid, params, mtp, layers, hidden, vocab, fmt, bits, act, domain, family, moe in ROWS:
        m = get_model(mid)
        e = ents[mid]
        pc = m.param_check()
        assert pc["rel_err"] == 0.0, mid
        assert m.params() == params == m.release_params, mid
        assert pc["mtp_ours"] == pc["mtp_release"] == mtp, mid
        assert m.n_layers == layers and m.hidden == hidden and m.vocab == vocab, mid
        w = _fmt(m)
        assert w.fmt == fmt and round(w.bits, 6) == bits, (mid, w.fmt, w.bits)
        assert m.act_fmt == act and m.is_moe is moe, mid
        assert e["domain"] == domain and e["family"] == family and e["coverage"] == "full", mid
        assert m.coverage_reasons == (), mid


def test_same_shape_siblings_and_new_stacks():
    nxt, coder = get_model("qwen3-next-80b-a3b"), get_model("qwen3-coder-next-fp8")
    assert nxt.params() == coder.params() and nxt.n_layers == coder.n_layers
    assert _fmt(nxt).fmt == "bf16" and _fmt(coder).fmt == "fp8"
    base, fp8 = get_model("qwen3.8-27b"), get_model("qwen3.8-27b-fp8")
    assert base.params() == fp8.params() and base.vision_params == fp8.vision_params == 460_730_096
    assert base.vision.params == fp8.vision.params and fp8.vision.wfmt == () and fp8.vision.w_Bpp == 2.0
    assert fp8.vision.depth == 27 and fp8.vision.hidden == 1152
    assert get_model("qwen3-14b").n_layers != get_model("qwen3-32b").n_layers
    moe = next(L.ffn for L in get_model("qwen3-coder-480b-a35b").layers if L.ffn.kind == "moe")
    assert (moe.n_experts, moe.top_k, moe.d_expert, moe.n_shared) == (160, 8, 2560, 0)
    hy = next(L.ffn for L in get_model("hy3").layers if L.ffn.kind == "moe")
    assert (hy.n_experts, hy.top_k, hy.d_expert) == (192, 8, 1536)
    flash = get_model("glm-4.7-flash")
    core = next(L.core for L in flash.layers if L.core.kind == "mla")
    assert core.kv_lora == 512 and core.idx_heads == 0
    ffn = next(L.ffn for L in flash.layers if L.ffn.kind == "moe")
    assert (ffn.n_experts, ffn.top_k, ffn.d_expert) == (64, 4, 1536)


def test_nemotron_pattern_alphabet():
    """hybrid_override_pattern uses the alphabet Lightning already publishes as layers_block_type."""
    nano = get_model("nemotron-nano-9b-v2")
    note = next(n for n in nano.notes if n.startswith("Nemotron-H"))
    assert "27 层 Mamba-2" in note and "4 层 GQA" in note and "25 层 relu² MLP" in note
    small = get_model("nemotron-3-nano-4b-bf16")
    note = next(n for n in small.notes if n.startswith("Nemotron-H"))
    assert "21 层 Mamba-2" in note and "4 层 GQA" in note and "17 层 relu² MLP" in note
    ultra = get_model("nemotron-3-ultra-550b-a55b-bf16")
    note = next(n for n in ultra.notes if n.startswith("Nemotron-H"))
    assert "48 层 latent MoE" in note and "层 relu² MLP" not in note
    lit = get_model("nemotron-3.5-lightning-30b-a3b-bf16")
    assert lit.param_check()["rel_err"] == 0.0 and lit.params() == 31_577_940_288 and lit.n_layers == 52
    assert "层 relu² MLP" not in "".join(lit.notes)
    bcs, types = _nemotron_block_types({"hybrid_override_pattern": "M-E*"})
    assert bcs == [] and types == ["mamba", "mlp", "moe", "attention"]
    try:
        _nemotron_block_types({"hybrid_override_pattern": "MX"})
    except ValueError as ex:
        assert "unknown chars" in str(ex)
    else:
        raise AssertionError("unknown pattern character was accepted")


def test_wan22_ti2v_5b_dit_only():
    """Dense Wan DiT. Wan2.2-VAE (4x16x16) is not the catalog Wan2.1 VAE, so the pipeline stays off."""
    m = get_model("wan2.2-ti2v-5b")
    e = next(x for x in list_models() if x["id"] == "wan2.2-ti2v-5b")
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 4_999_787_712
    assert m.n_layers == 30 and m.hidden == 3072
    assert _fmt(m).fmt == "fp32" and m.act_fmt == "bf16"
    assert e["family"] == "Wan2.2" and e["domain"] == "gen" and e["coverage"] == "partial"
    assert len(m.coverage_reasons) == 1
    assert "文本编码器（11.4 GB）" in m.coverage_reasons[0] and "VAE（2.8 GB）" in m.coverage_reasons[0]
    assert pipeline_for("wan2.2-ti2v-5b") is None
    w = m.workload
    assert (w.frames, w.height, w.width, w.fps, w.steps, w.cfg) == (121, 704, 1280, 24, 50, 2)
    assert (w.vae_t, w.vae_s, w.patch, w.text_tokens, w.vae_frames) == (4, 16, (1, 2, 2), 512, "wan")
    rw = resolve_workload(m, Scenario(model="wan2.2-ti2v-5b").workload)
    assert rw.warnings == ()
    assert rw.frames == 31 and rw.tokens == 27280 and rw.ctx == 512 and rw.steps == 50 and rw.seqs_per_request == 2
    assert rw.info["latent"] == [31, 44, 80] and rw.info["grid"] == [31, 22, 40]


def test_changelog_363():
    assert "## [0.363.0]" in CL
    sec = CL.split("## [0.363.0]")[1].split("## [")[0]
    for needle in ("Qwen3-14B", "Qwen3-Coder-480B", "Qwen3-Coder-Next-FP8", "Qwen3.8-27B-FP8",
                   "NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-3-Ultra-550B", "Hy3", "GLM-4.7-Flash",
                   "Wan2.2-TI2V-5B", "未纳入"):
        assert needle in sec, needle
