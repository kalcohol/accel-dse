"""0.278: NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 — Nemotron-H hybrid (Mamba-2 SSD / GQA / latent MoE, one mixer
per layer, 1 MTP module) + C-RADIO ViT-H/16 vision with the processor's dynamic single-tile rule."""
import importlib.util
import json
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.scenario import Scenario
from accel_dse.core.evaluate import evaluate
from accel_dse.core.vision import image_grid, radio_patches

ROOT = Path(__file__).resolve().parents[1]
MID = "nemotron-3.5-super-vl-120b-a12b-nvfp4"
REL = ROOT / "accel_dse/data/releases/nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.json"


def _sr():
    spec = importlib.util.spec_from_file_location("summarize_release", ROOT / "scripts/summarize_release.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_params_exact_against_release_headers():
    m, rel = get_model(MID), json.loads(REL.read_text())
    assert m.release_params == m.params() == rel["params_llm"]
    assert m.mtp_params() == rel["params_mtp"]
    assert m.vision is not None and m.vision_params == rel["params_vision"] == 842_362_886
    # RADIO tower + projector modelled exactly; the stored-only remainder = video_embedder (3·2·16²·1280) + the 6 fp32
    # input-normalizer buffers
    assert m.vision.extra == 3 * 2 * 16 * 16 * 1280 + 6
    assert "other" not in rel["roles"]           # backbone.norm_f → norm


def test_layer_pattern_and_formats():
    m = get_model(MID)
    assert len(m.layers) == 88 and len(m.mtp_layers) == 1
    kinds = [("mamba" if l.core.kind == "linear" else "attn" if l.core.kind != "none" else "moe") for l in m.layers]
    assert kinds.count("mamba") == 40 and kinds.count("attn") == 8 and kinds.count("moe") == 40
    mam = next(l for l in m.layers if l.core.kind == "linear").core
    assert (mam.lin, mam.n_state_heads, mam.state_dk, mam.state_dv, mam.state_groups, mam.lin_chunk) == ("ssd", 128, 128, 64, 8, 128)
    moe = next(l for l in m.layers if l.core.kind == "none").ffn
    assert moe.n_experts == 512 and moe.top_k == 22
    rel = json.loads(REL.read_text())
    assert set(rel["roles"]["expert"]) == {"nvfp4"} and set(rel["roles"]["lm_head"]) == {"fp8"}
    assert rel["quantization_config"]["kv_cache_scheme"]["num_bits"] == 8


def test_ssd_cb_product_shared_per_group():
    """mamba_chunk_scan_combined forms C·Bᵀ once per (chunk, group): count = b · ⌈G / tp⌉ · chunks, not per head."""
    m = get_model(MID)
    li = next(i for i, l in enumerate(m.layers) if l.core.kind == "linear")
    for tp in (1, 2, 16):
        ops = build_rank_ops(m, li, Phase("prefill", 1, 1024, 0), Shard(tp=tp, ep=tp))
        aqk = [o for o in ops if o.name == "lin_aqk"]
        intra = [o for o in ops if o.name == "lin_intra"]
        assert len(aqk) == 1 and aqk[0].count == -(-8 // tp) * (1024 // 128)
        assert intra[0].count == -(-128 // tp) * (1024 // 128)


def test_radio_patch_rule_matches_processor():
    # (w, h) → (w_patches, h_patches): processor rule, budget min((16384 − 4)·4, 13312), floor 1024, sides even
    assert radio_patches(1024, 768, 16, 2, 1024, 13312, 16384) == (64, 48)
    assert radio_patches(100, 100, 16, 2, 1024, 13312, 16384) == (32, 32)
    wp, hp = radio_patches(4000, 3000, 16, 2, 1024, 13312, 16384)
    assert wp * hp <= 13312 and wp % 2 == 0 and hp % 2 == 0 and (wp, hp) == (132, 100)
    g = image_grid(get_model(MID).vision, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens) == (64, 64, 1024, 1026)    # <img> … </img>


def test_evaluate_with_image():
    s = Scenario(model=MID, chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g").replace("layout.tp", 2).replace("layout.ep", 2)
    r0 = evaluate(s.replace("serving.phase", "prefill"))
    r1 = evaluate(s.replace("serving.phase", "prefill").replace("serving.images", 1))
    assert r1.fits and r1.ttft > r0.ttft and r1.vision["tokens_per_image"] == 1026
    # tower FLOPs ≈ 32 · 2 · N · (4h² + 2·h·4h) + 32 · 4 · N² · h + projector, N = 4096
    N, h = 4096, 1280
    est = 32 * (2 * N * (4 * h * h + 8 * h * h) + 4 * N * N * h) + 2 * (N // 4) * (4 * h * 20480 + 20480 * 4096)
    assert abs(r1.vision["tflop_per_image"] * 1e12 / est - 1) < 0.05


def test_summarize_helpers():
    sr = _sr()
    f = sr._nemotron_h_omni_role
    assert f("vision_model.radio_model.blocks.0.attn.qkv.weight", 88) == "vision"
    assert f("mlp1.1.weight", 88) == "vision"
    assert f("language_model.backbone.layers.1.mixer.fc1_latent_proj.weight", 88) == "mlp"
    assert f("language_model.backbone.norm_f.weight", 88) == "norm"
    qc = {"quant_method": "modelopt_mixed",
          "quantized_layers": {"a": {"quant_algo": "FP8"}, "b": {"quant_algo": "NVFP4", "group_size": 16},
                               "c": {"quant_algo": "NVFP4", "group_size": 16}},
          "ignore": ["x", "y"]}
    c = sr._compact_qc(qc)
    assert c["quantized_layers_by_algo"] == {"FP8": 1, "NVFP4/g16": 2} and c["ignore_count"] == 2
    assert "quantized_layers" not in c and "ignore" not in c
    assert sr._compact_qc({"quant_method": "fp8", "ignore": ["x"]}) == {"quant_method": "fp8", "ignore": ["x"]}


def test_dac_dram_split_weights_vs_activations():
    """Scope 3 finding: DAC DRAM traffic is activation-dominated, so SLC-pinning the 0.74 GB fp32 DAC weights is not
    mechanism-justified; the split adds up to the costed DRAM bytes of each window."""
    from accel_dse.core.drama3 import dac_ops, dac_windows
    from accel_dse.core.evaluate import _component_time
    from accel_dse.core.hardware import System
    from accel_dse.core.speech import _dac_window_dram, speech_kpis
    s = Scenario(model="drama-3", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g")
    k1, k8 = (speech_kpis(s.replace("serving.batch", b))["dac"] for b in (1, 8))
    assert abs(k1["dram_weights_GB"] - k8["dram_weights_GB"]) < 1e-9          # weights: once per window, any batch
    assert k1["dram_act_GB"] > 2 * k1["dram_weights_GB"] and k8["dram_act_GB"] > 20 * k8["dram_weights_GB"]
    sys = System(s.chip, s.mem_id, s.mem_eff, s.link, s.d2d_link, s.package_eff, s.net, s.node_cards, s.fabric, 1,
                 s.mem_cap_GB, s.kv_mem_frac, s.kv_mem_eff)
    fs = get_model("drama-3").fish
    for n, _, done in dac_windows(216)[:3]:
        w, a = _dac_window_dram(fs, n, 0, done, 2, sys, s.mapping)
        ref = _component_time([(dac_ops(fs, n, 0, done, 2), 1)], sys, s.mapping, "fp32")["dram_GB"]
        assert abs(w + a - ref) < 1e-9 * max(1.0, ref)
