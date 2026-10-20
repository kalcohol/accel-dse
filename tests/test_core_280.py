"""0.280: nvidia Qwen3.8-Flash-Next-FP4-Local (ModelOpt MIXED_PRECISION as released: NVFP4 csf experts, MXFP8 attention,
per-role activation formats); Index-Echo in the speech bounds / streams views; CosyVoice3 frontend from its ONNX graphs."""
import importlib.util
import json
import re
from pathlib import Path

from accel_dse.core.catalog import get_model, labels, list_models
from accel_dse.core.echo import ECHO_REASON
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release
from accel_dse.core.onnxfront import front_flops, front_ops
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS, _lm_load, _scn, _stages, speech_bounds

ROOT = Path(__file__).resolve().parents[1]
MEM = "lpddr5x_4x64_8533_16g"
FP4, BASE = "qwen3.8-flash-next-fp4-local", "qwen3.8-flash-next"
REL = ROOT / "accel_dse/data/releases/nvidia__Qwen3.8-Flash-Next-FP4-Local.json"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fp4_local_params_and_as_released_bits():
    m, b, rel = get_model(FP4), get_model(BASE), json.loads(REL.read_text())
    assert m.release_params == m.params() == b.params() == rel["params_llm"] == 176_943_899_520
    assert rel["params_mtp"] == 2_607_150_848 and rel["params_vision"] == 448_931_056 and rel["n_tensors"] == 375_487
    f = dict(m.formats)
    ex = rel["roles"]["expert"]["nvfp4"]
    assert f["expert"].fmt == "nvfp4" and abs(f["expert"].bits - 8 * ex["bytes"] / ex["params"]) < 1e-12
    assert 4.25 < f["expert"].bits < 4.3          # csf scales (fixed u8 + u32 exceptions) < 4.5 of plain NVFP4 / 16
    assert f["attn"].fmt == "fp8" and abs(f["attn"].bits - 8.25) < 0.01      # MXFP8: e4m3 + e8m0 per 32
    assert f["shared_expert"].fmt == "fp8" and f["lm_head"].fmt == "bf16" and f["other"].fmt == "bf16"
    assert (f["act_expert"].fmt, f["act_attn"].fmt, f["act_shared_expert"].fmt) == ("nvfp4", "fp8", "fp8")
    assert "act_mtp" not in f                      # W4A16_NVFP4 MTP experts stay weight-only
    assert m.layers == b.layers and m.release_bytes < 0.3 * b.release_bytes


def test_fp4_local_quant_config_compact():
    qc = json.loads(REL.read_text())["quantization_config"]
    assert "quantized_layers" not in qc and qc["quantized_layers_by_algo"] == \
        {"MXFP8/g32": 438, "NVFP4/g16": 1, "NVFP4/g16/csf": 48, "W4A16_NVFP4/g16": 58}
    assert qc["algo_by_role"]["expert"] == ["NVFP4"] and qc["algo_by_role"]["attn"] == ["MXFP8"]
    sr = _mod("summarize_release", "scripts/summarize_release.py")
    assert sr.SCALE_RE.search("model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale.nvfp4_csf_exceptions")
    assert sr.SCALE_RE.search("model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale.nvfp4_csf_fixed")
    assert sr.classify("model.language_model.layers.3.attn_hyper_connection.input_mix_weight_up.weight") == "other"
    assert sr.classify("model.language_model.layers.3.attn_hyper_connection.hc_norm.weight") == "norm"


def test_gated_residual_and_ple_keep_bf16():
    m = get_model(FP4)
    ops = build_rank_ops(m, 1, Phase("decode", 1, 1, 1024), Shard())
    by = {o.name: o for o in ops if o.kind == "gemm"}
    assert by["attn.gr_a_down"].w_fmt == "bf16" and by["attn.gr_a_down"].a_fmt == "bf16"
    assert by["attn.ple_key"].w_fmt == "bf16"
    assert all((o.w_fmt, o.a_fmt) == ("fp8", "fp8") for n, o in by.items() if n.startswith("attn.lin_"))
    assert (by["expert.gate_up"].w_fmt, by["expert.gate_up"].a_fmt) == ("nvfp4", "nvfp4")
    base = get_model(BASE)
    assert dict(base.formats)["other"].fmt == "bf16"     # all-bf16 release: identical costs (full-model hashes unchanged)
    rel = load_release("Qwen/Qwen3.8-Flash-Next")
    assert rel["roles"]["attn"]["bf16"]["params"] == 2_703_854_976 and rel["roles"]["other"]["bf16"]["params"] == 672_471_040


def test_fp4_local_listing_and_speed():
    x = {e["id"]: e for e in list_models()}[FP4]
    assert x["provider"] == "nvidia" and x["family"] == "Qwen3.8（NVIDIA 量化）" and x["domain"] == "vlm"
    assert x["param_err"] == 0.0 and x["vision"]["video"]
    assert labels(get_model(FP4))["dtype"] == "W fp8/专家 nvfp4 · A fp8/专家 nvfp4 · KV bf16"
    assert labels(get_model("beam-1.0-501b-a23b-nvfp4"))["dtype"] == "W bf16/专家 nvfp4 · A bf16/专家 nvfp4 · KV bf16"
    s = lambda i: Scenario(model=i, chip=CHIPS["100T"], mem_id=MEM).replace("layout.tp", 8).replace("layout.ep", 8)
    r4, rb = evaluate(s(FP4)), evaluate(s(BASE))
    assert r4.fits and r4.tpot < rb.tpot


def test_nemotron_mixed_activation_formats():
    m = get_model("nemotron-3.5-super-vl-120b-a12b-nvfp4")
    f = dict(m.formats)
    assert (f["act_attn"].fmt, f["act_expert"].fmt, f["act_shared_expert"].fmt, f["act_lm_head"].fmt) == \
        ("fp8", "nvfp4", "nvfp4", "fp8")
    assert "act_mlp" not in f                     # bf16 latent projections are not quantized


def test_echo_in_speech_views():
    assert {"index-echo-s2st-2b", "index-echo-s2st-9b"} <= set(SPEECH_MODELS)
    B = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id=MEM)
    rows = {r["model"]: r for r in speech_bounds(B, ("index-echo-s2st-2b", "index-echo-s2st-9b"), verify=False)["rows"]}
    a, b = rows["index-echo-s2st-2b"], rows["index-echo-s2st-9b"]
    assert a["binding"] == "pipeline" and a["realtime"] and a["first_audio_dominant"] == "cv3_decode_s"
    assert b["first_audio_dominant"] == "st_decode_s" and abs(b["rereads_per_audio_s"] - 6.2) < 1e-12
    for mid in ("index-echo-s2st-2b",):
        for n in (1, 4):
            s = _scn(B, mid, 10.0).replace("serving.batch", n)
            full, lo = speech_kpis(s, 10.0), speech_kpis(s, 10.0, lm_only=True)
            assert _lm_load(full, 10.0) == _lm_load(lo, 10.0) == {"pipeline": full["rtf"]} and full["fits"] == lo["fits"]
            assert full["clips"] == n
    k1 = speech_kpis(_scn(B, "index-echo-s2st-2b", 10.0), 10.0)
    k4 = speech_kpis(_scn(B, "index-echo-s2st-2b", 10.0).replace("serving.batch", 4), 10.0)
    assert k1["total_s"] < k4["total_s"] < 4 * k1["total_s"]            # batched LM steps, per-clip flow / HiFT
    assert abs(k4["split"]["flow_s"] - 4 * k1["split"]["flow_s"]) < 1e-12
    assert abs(sum(k4["split"].values()) - k4["total_s"]) < 1e-12 and k1["split"]["repin_s"] > 0
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert "pipeline: 'S2ST 全流程（串行）'" in js


def test_speech_tokenizer_flops_from_graph():
    fe = load_release("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")["onnx_frontend"]
    st, cp = fe["speech_tokenizer"], fe["campplus"]
    assert st["params"] == 242_009_608 and cp["params"] == 6_907_680 and st["dtypes"] == ["F32"]
    assert (st["provider"], cp["provider"], st["max_s"], st["bins"], cp["bins"]) == ("cuda", "cpu", 30, 128, 80)
    # 10 s → 1000 mel frames → conv1 s2 (500) → conv2 s2 (250); 12 blocks q/k/v/o 1280², MLP 1280↔5120, full attention
    # 20 × 64 over 250 positions (QK + PV), FSQ project_in 1280 → 8
    t1, t = 500, 250
    want = 2 * 384 * 1280 * t1 + 2 * 3840 * 1280 * t + 12 * (4 * 2 * 1280 * 1280 * t + 2 * 2 * 1280 * 5120 * t
                                                             + 4 * t * t * 1280) + 2 * 1280 * 8 * t
    assert front_flops(st, 10.0) == want
    assert sum(o.w_params for o in front_ops(st, 10.0)) == st["params"]
    assert sum(o.w_params for o in front_ops(cp, 10.0)) == cp["params"]
    assert [o for o in cp["ops"] if o[0] == "gemm" and o[4] == 0]          # the dense layer after statistics pooling


def test_echo_frontend_in_kpis():
    s = Scenario(model="index-echo-s2st-2b", chip=CHIPS["100T"], mem_id=MEM).replace("serving.audio_s", 10.0)
    p = speech_kpis(s, 10.0)
    assert p["split"]["speech_tokenizer_s"] > 0 and abs(sum(p["split"].values()) - p["total_s"]) < 1e-12
    assert not any("campplus" in k for k in p["split"])                  # host CPU: reported only
    assert p["frontend"]["campplus_on"] == "host CPU" and p["frontend"]["campplus_gflop_host"] > 0
    assert "CampPlus" in ECHO_REASON and "speech_tokenizer_v3" not in ECHO_REASON
    m = get_model("index-echo-s2st-2b")
    assert m.coverage == "partial" and tuple(m.coverage_reasons) == (ECHO_REASON,)


def test_onnx_wire_walker():
    fo = _mod("fetch_onnx_graph", "scripts/fetch_onnx_graph.py")
    # TensorProto: dims [3, 4] (packed), data_type 1, name "w", raw_data 48 bytes
    raw = bytes(48)
    t = bytes([0x0A, 2, 3, 4, 0x10, 1, 0x42, 1]) + b"w" + bytes([0x4A, 48]) + raw
    assert fo.tensor(t, (0, len(t))) == ("w", "F32", [3, 4])
    assert fo.varint(bytes([0xAC, 0x02]), 0) == (300, 2)
