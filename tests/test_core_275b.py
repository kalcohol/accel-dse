"""0.275 FireRedTTS3-Base: backbone config from the code, exact params, per-step head ops from the released shapes,
non-streaming first audio."""
import json
import math
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import _component_time, evaluate
from accel_dse.core.fireredtts3 import (fire_head_ops, prompt_ops, prompt_patches, redae_decode_ops)
from accel_dse.core.hardware import CHIPS, System
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
M = "fireredtts3-base"
REL = json.loads((ROOT / "accel_dse/data/releases/FireRedTeam__FireRedTTS3.json").read_text())
W = lambda ops, pre: sum(o.w_params * o.count for o in ops if o.kind == "gemm" and o.name.startswith(pre))


def test_backbone_config_matches_header():
    """Qwen3_1_7B_ConfigDict is a dict in fireredtts3_base.py (not in the HF config); the header confirms it."""
    c, st = REL["config"], REL["speech_tensors"]
    b = c["backbone"]
    assert "fireredtts3_base.py" in c["_source"] and c["model_type"] == "fireredtts3"
    m = get_model(M)
    assert (m.hidden, m.n_layers, m.vocab, m.tie_embeddings) == (b["hidden_size"], b["num_hidden_layers"], b["vocab_size"], True)
    assert m.params() == REL["params_llm"] == 1_720_574_976 and m.coverage == "full" and m.domain == "llm"
    assert st["dit_head.weight"] == [c["core"]["dit_hidden_size"], b["hidden_size"]]
    assert st["dit.in_proj.weight"] == [1024, c["core"]["redae_dim"] + c["core"]["spk_in_dim"] + 1024]
    assert st["redae.decoder.qwen3.layers.0.self_attn.q_proj.weight"] == [14 * 128, c["redae"]["dec_hidden_size"]]


def test_exact_params_and_head_weights():
    t = get_model(M).tts
    assert t.head_params == REL["params_tts"] == 400_057_921
    assert t.redae_params == REL["params_codec"] == 943_776_322
    assert REL["params_llm"] + REL["params_tts"] == 2_120_632_897
    ops = fire_head_ops(t, 1, 0)
    # one Euler evaluation reads every DiT weight except the RMSNorm weights (3 per block)
    assert t.flow_params - W(ops, "tts.flow0.") == t.dit_depth * 3 * t.dit_d
    assert W(ops, "tts.flow") == t.flow_steps * W(ops, "tts.flow0.")
    assert t.agg_params - W(ops, "tts.pe.") == t.pe_depth * 2 * t.pe_d + 2 * t.pe_d          # norms + cls token
    assert W(ops, "tts.stop") + W(ops, "tts.dit_head") == t.stop_params
    dec = redae_decode_ops(t, 100)
    assert t.redae_dec_params - W(dec, "redae.dec") == 18 * (2 * 896 + 2 * 128) + 896 + 1920   # norms + window
    assert t.s_per_step == 0.16 and t.cfg_mult == 2 and t.flow_rows == 12
    assert prompt_patches(t, 5.0) == 32


def test_firered_kpis_formula():
    s = Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=64, out_len=64))
    k = speech_kpis(s, 10.0)
    assert k["label"] == "FireRedTTS3" and k["steps"] == 63 and k["prompt_audio_s"] == 5.0
    f = k["split_first"]
    assert abs(k["first_audio_s"] - sum(f.values())) < 1e-12 and k["first_audio_s"] == k["total_s"]   # non-streaming
    assert abs(f["steps_s"] - 63 * k["tpot_s"]) < 1e-12
    t = get_model(M).tts
    sys = System(s.chip, s.mem_id, s.mem_eff)
    assert abs(f["redae_decode_s"] - _component_time([(redae_decode_ops(t, 252), 1)], sys, s.mapping, "fp32")["s"]) < 1e-12
    assert abs(f["prompt_s"] - _component_time([(prompt_ops(t, 5.0), 1)], sys, s.mapping, "fp32")["s"]) < 1e-12
    assert abs(k["rtf"] - (f["steps_s"] + f["redae_decode_s"]) / 10.0) < 1e-12
    # 100T + LPDDR5X: real time (RTF ≈ 0.57) but first audio ≈ the whole segment
    assert 0.4 < k["rtf"] < 0.8 and k["first_audio_s"] > 5.0
    dec = evaluate(s.replace("serving.phase", "decode").replace("serving.prompt", 64 + 1 + 32).replace("serving.ctx", 97 + 31))
    assert abs(dec.tpot - k["tpot_s"]) < 1e-12 and dec.tts["head_w_GB_per_step"] > 10   # 10 × 1.17 GB fp32 DiT re-read


def test_moshi_layer_overhead_reuses_scenario_knob():
    """0.275: Scenario.layer_overhead_us (0.69 per-layer fixed overhead) also charges the depformer's 48 serial
    one-row layer passes and the 8 + 8 Mimi transformer layers; off by default → identical to 0.274."""
    import dataclasses
    s = Scenario(model="moshiko-7b", chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=1, prompt=1, out_len=1))
    a = speech_kpis(s)
    b = speech_kpis(dataclasses.replace(s, layer_overhead_us=10.0))
    assert a["layer_passes"] == {"depformer_s": 48, "mimi_encode_s": 8, "mimi_decode_s": 8} and a["layer_overhead_s"] == 0
    for key, n in a["layer_passes"].items():
        assert abs(b["split_frame"][key] - a["split_frame"][key] - n * 10e-6) < 1e-12
    dt = b["split_frame"]["temporal_s"] - a["split_frame"]["temporal_s"]
    assert abs(dt - 32 * 10e-6) < 2e-6                 # the temporal step's own 32 layers, from evaluate
    assert abs(b["layer_overhead_s"] - 64 * 10e-6) < 1e-12
