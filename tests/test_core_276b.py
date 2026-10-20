"""0.276: Fish Audio Drama 3 (core/drama3.py) — release roles, tower as decoder layers, streaming speech KPIs."""
import json
import math
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import SPEECH_MODELS, _scn, _stages

ROOT = Path(__file__).resolve().parents[1]
REL = json.loads((ROOT / "accel_dse/data/releases/fishaudio__drama-3.json").read_text())
M = "drama-3"
BASE = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g")


def test_release_roles_and_resident_params():
    r = REL["roles"]
    assert REL["params_vision"] == 0 and set(r["expert"]) == {"fp8"} and set(r["attn"]) == {"bf16"}
    # tower = 13 × (wqkv 9216×2048 + wo 2048×4096 + eh_proj 2048×4096) + backbone attention
    m = get_model(M)
    assert len(m.layers) == 40 + 13 and m.vocab == 4096 and m.fish.tower_layers == 13
    t = m.layers[40:]
    assert all(L.core.kind == "gqa" and L.core.n_kv == 2 and L.core.qk_dim == 256 and L.core.window is None for L in t)
    assert sum(l.params for l in t[0].attn_linears) == 4096 * 2048 + 9216 * 2048 + 4096 * 2048
    assert sum(l.params for l in t[1].attn_linears) == 2048 * 2048 + 9216 * 2048 + 4096 * 2048
    assert sum(l.params for l in t[0].ffn_linears) == 3 * 6144 * 2048
    # resident = release − non-candidate lm_head rows − eh_proj embedding halves of tower layers ≥ 1 (vllm-omni)
    drop = (252288 - 4096) * 2048 + 12 * 2048 * 2048
    assert m.release_params == REL["params_llm"] - drop and m.params() == m.release_params
    assert m.fmt("expert").fmt == "fp8" and m.lookup_bits == 16
    x = {e["id"]: e for e in list_models()}[M]
    assert x["domain"] == "tts" and abs(x["param_err"]) <= 0.005


def test_streaming_kpis():
    k = speech_kpis(_scn(BASE, M, 10.0), 10.0)
    frame = 2048 / 44100
    assert k["steps"] == math.ceil(10.0 / frame) and abs(k["frame_ms"] - frame * 1e3) < 1e-9
    assert k["ref_rows"] == math.ceil(10.0 * 44100 / 2048) and k["prompt_rows"] == 64 + k["ref_rows"]
    assert abs(k["lm_rtf"] - k["tpot_s"] / frame) < 1e-12 and abs(k["rtf"] - k["lm_rtf"] - k["dac"]["load"]) < 1e-12   # 0.277
    # 0.277: + the DAC's first 6-frame window (colocated; tests/test_core_277b.py)
    assert abs(k["first_audio_s"] - k["ttft_s"] - 6 * k["tpot_s"] - k["split_first"]["dac_window0_s"]) < 1e-12
    assert k["dac_costed"] is True
    st, r, dom, fa = _stages(k, 10.0)
    assert set(st) == {"lm"} and dom == "prefill_s" and fa == k["first_audio_s"]
    assert M in SPEECH_MODELS


def test_decode_is_weight_read_bound_at_batch_1():
    """b = 1 on LPDDR5X: TPOT ≈ (bf16 non-expert + tower + heads + 8 fp8 experts × 40 layers) / effective BW 「推导」."""
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.hardware import System
    m = get_model(M)
    r = REL["roles"]
    nonexp = 2 * (sum(v["params"] for k in ("attn", "mlp", "shared_expert", "router") for v in r[k].values()) - 12 * 2048 * 2048)
    heads = 2 * (4096 + 9 * 4096) * 2048
    exp = 40 * 8 * 3 * 512 * 2048
    s = _scn(BASE, M, 10.0)
    d = evaluate(s.replace("serving.phase", "decode").replace("serving.prompt", 280).replace("serving.ctx", 388))
    bw = System(BASE.chip, BASE.mem_id, BASE.mem_eff).dram_GBps * 1e9
    assert 0.9 < d.tpot / ((nonexp + heads + exp) / bw) < 1.15
    assert m.fish.candidates == 4096
