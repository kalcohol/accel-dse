"""0.336: OmniASR LLM v2 Wav2Vec2-style audio builder (asr domain); CTC kept pending; 0.335 LPDDR pins hold."""
import os

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.energy import ENERGY_PRESETS, _LPDDR_ACTIVE_PJ
from accel_dse.core.energy_dse import MEM_PROXY
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.vision import audio_ops, audio_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = ("omniasr-llm-300m-v2", "omniasr-llm-1b-v2", "omniasr-llm-3b-v2", "omniasr-llm-7b-v2")
WANT = {  # (params_llm, params_vision) from released safetensors headers
    "omniasr-llm-300m-v2": (1_318_817_792, 319_635_968),
    "omniasr-llm-1b-v2": (1_318_817_792, 967_742_976),
    "omniasr-llm-3b-v2": (1_318_817_792, 3_068_711_424),
    "omniasr-llm-7b-v2": (1_318_817_792, 6_493_073_920),
}


def test_omniasr_llm_params_and_domain():
    ents = {e["id"]: e for e in list_models()}
    for mid, (llm, vis) in WANT.items():
        m = get_model(mid)
        assert m.params() == llm and m.param_check()["rel_err"] == 0.0, mid
        assert m.vision.audio.kind == "w2v" and m.vision.audio.params == vis == m.vision_params, mid
        assert ents[mid]["domain"] == "asr" and ents[mid]["family"] == "OmniASR", mid
        assert ents[mid]["coverage"] == "full", mid
    # shared Llama text trunk; audio tower scales with the size name
    assert len({get_model(i).params() for i in IDS}) == 1
    assert get_model("omniasr-llm-7b-v2").vision_params > get_model("omniasr-llm-300m-v2").vision_params


def test_omniasr_w2v_frames_and_ops():
    from accel_dse.core.model import load_release
    a = get_model("omniasr-llm-300m-v2").vision.audio
    # 16 kHz / Π stride 320 → 50 Hz; (L − k)//s + 1 over 10 s = 499 frames
    assert a.tokens(10) == 499 == a.enc_positions(10) == a.prompt_tokens(10)
    assert a.rate == 16000 and a.ds[0] == (10, 5, 0) and a.lin == ((1024, 4096),)
    ops = audio_ops(a, 10.0, 1)
    assert any(o.name.startswith("aud.fe.") for o in ops) and any(o.name == "aud.pos" for o in ops)
    assert any(o.name == "aud.proj.linear0" for o in ops)
    fl = sum(2 * o.m * o.k * o.n * o.count for o in ops if o.kind in ("gemm", "attn"))
    assert fl > 0
    # CTC config builds the same audio tower but is not catalogued (no AR LM for asr domain)
    ac = load_release("bezzam/omniasr-llm-300m-v2")["config"]["audio_config"]
    c = audio_spec({"model_type": "omniasr_ctc", "audio_config": ac, "vocab_size": 10288})
    assert c.kind == "w2v" and c.lin == () and c.out == ac["hidden_size"]
    assert not any(e["id"].startswith("omniasr-ctc") for e in list_models())


def test_omniasr_speech_kpis_smoke():
    s = Scenario(model="omniasr-llm-300m-v2", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g",
                 serving=Serving(phase="prefill", batch=1, prompt=1, ctx=4096, audio_s=10.0))
    k = speech_kpis(s)
    assert k["kind"] == "asr" and k["enc_positions"] == 499 and k["fits"] is True


def test_lpddr_0335_pins_hold():
    F = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    v, tag, src = F["pJ_bit_dram_lpddr"]
    assert abs(v - _LPDDR_ACTIVE_PJ) < 1e-12 and tag == "推导" and "IDD4R" in src and "Y52Q" in src
    assert abs(F["pJ_bit_dram_lpddr6"][0] - _LPDDR_ACTIVE_PJ / 1.21) < 1e-12
    assert abs(F["pJ_bit_dram_hbm"][0] - 5.8017342581848474) < 1e-9
    assert "2.67" in MEM_PROXY["LPDDR6"]


def test_docs_and_no_ui_patch():
    md = open(os.path.join(ROOT, "docs", "MODEL.md"), encoding="utf-8").read()
    cl = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    assert "§281" in md and "## [0.336.0]" in cl and "OmniASR" in cl.split("## [0.336.0]")[1].split("## [")[0]
    assert os.path.exists("/workspace/ui-audit/ui.patch")  # present → not applied this version
