"""0.62.1 — round-6 convergence audit: doc examples re-derived from the code (MODEL.md §2 / §9 / §12 / §19 / §19.5),
neutral-value identities of the optional features, eval / sweep agreement with images."""

from __future__ import annotations
from dataclasses import replace

import glob
import json
import os

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models, unlisted_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Fabric
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"


def test_release_count_and_param_error():
    """MODEL §2 / §9 V1: 99 releases (77 LLM / VLM incl. 3 validation-only Llama-2-7B/70B + Mistral-7B (0.66), Carbon 500M/3B/8B (0.106), LFM2.5-24B-A2B (+FP8) and Ling-3.1-flash (0.114), Step-5-Preview, MiMo-V2.6-Flash, Intern-S2-397B, MiniCPM-V-4.7 (0.144), MiMo-V2.6-Pro, Ling-3.0-flash-VL (0.145), Evo 2 40B / 7B / 1B (0.170), Muse Spark Open BF16 / FP8 / MXFP4 (0.174) + NVFP4 (0.175) + 22 image / video / protein / genomics incl. Qwen-Image-2.1-Turbo, Qwen-Image, FLUX.2-klein-4B (0.115), LingBot-Video MoE-DMD (0.144)), all within ±0.5 %."""
    files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "accel_dse", "data", "releases", "*.json"))
    doms = [json.load(open(f)).get("domain", "llm") for f in files]
    # 0.191: + Qwen3.5-9B, 14 StartLux-Decision, 4 RWKV-7 G1k; 0.251: + Hy-Memory-30B-A3B (+FP8), Step-5-Preview-NVFP4,
    # UserRepGen-4B, flux-3-image-klein-pup (llm) + FLUX 3 Image Klein / Klein Base (gen)  118 → 125, llm 96 → 101
    # 0.253: + Dory1.0-4B BF16 / MXFP8 / NVFP4, MAI-Gym-9B (llm)  125 → 129, llm 101 → 105
    # 0.254: + Beam-1.0-501B-A23B BF16 / FP8 / NVFP4 (llm)  129 → 132, llm 105 → 108
    # 0.257: + MOSS-Image-Base / -SFT (gen)  132 → 134, llm 108
    # 0.268: + Ming-3.0-asr-tiny (llm release, listing domain asr)  134 → 135, llm 109
    # 0.269: + Ming-3.0-tts-tiny (llm release, listing domain tts)  135 → 136, llm 110
    # 0.270: + Whisper-large-v3 / -turbo, Qwen2-Audio-7B-Instruct, Qwen3-ASR-1.7B (llm releases, listing domain asr)  136 → 140, llm 114
    # 0.271: + Qwen2.5-Omni-7B (listing domain omni), CosyVoice2-0.5B, Fun-CosyVoice3-0.5B-2512 (tts)  140 → 143, llm 117
    # 0.272: + Fun-ASR-Nano-2512-hf, Step-Audio-2-mini (llm releases, listing domain asr)  143 → 145, llm 119
    # 0.273: + Kimi-Audio-7B-Instruct (llm release, listing domain omni)  145 → 146, llm 120
    # 0.274: + Kyutai Moshiko (temporal transformer = llm release, listing domain omni)  146 → 147, llm 121
    # 0.275: + FireRedTTS3-Base (backbone = llm release, listing domain tts)  147 → 148, llm 122
    # 0.276: + Fish Audio Drama 3 (Qwen3.5-MoE backbone + codebook tower = llm release, listing domain tts)  148 → 149, llm 123
    # 0.278: + NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (llm release, listing domain vlm)  149 → 150, llm 124
    # 0.279: + Index-Echo-S2ST-2B / -9B (ST LM = llm release, listing domain omni)  150 → 152, llm 126
    # 0.280: + nvidia Qwen3.8-Flash-Next-FP4-Local (llm release, listing domain vlm)  152 → 153, llm 127
    # 0.287: + ornith-ai/Ornith-1.5-122B (qwen3_5_moe VLM = llm release, listing domain vlm)  153 → 154, llm 128
    # 0.288: + 11 Ornith releases (3 FP8 listed, 8 merged; all llm releases, listing domain vlm)  154 → 165, llm 139
    # 0.289: + 5 Nemotron 3.5 releases (Bolt 24GB / BF16, Lightning NVFP4 / BF16, Super-VL BF16; llm releases)  165 → 170, llm 144
    # 0.291: + 14 releases (4 pplx-decider, 5 Jais 3 BF16 + 4 FP8, Mellum2.1; all llm releases)  170 → 184, llm 158
    # 0.292: + 2 (Jais-3-MoVA BF16 + FP8)  184 → 186, llm 158 → 160
    # 0.311: + 2 (Ling-3.0-tiny, Ling-3.0-flash; bailing_hybrid llm releases)  186 → 188, llm 160 → 162
    # 0.314: + ATH-MaaS/OvisOCR3 (qwen3_5 VLM = llm release, listing domain vlm)  188 → 189, llm 162 → 163
    # 0.315: + RecursiveSI/flint (flint MoE llm release)  189 → 190, llm 163 → 164
    # 0.320: + KRAFTON/Raon-1-mini-preview (raon1_mini MoE llm release)  190 → 191, llm 164 → 165
    # 0.331: + ATH-MaaS__PixText_models__PixText-Teacher (second release file of the same repo)  192 → 193
    # 0.330: + ATH-MaaS/PixText (PixText-MeanFlow, gen release)  191 → 192, llm 165
    # 0.338: + 5 Qwen3-TTS (release domain default llm; listing domain tts)  198 → 203, llm 170 → 175
    # 0.337: + Carbon-X (HuggingFaceBio; Qwen3 DNA LM)  197 → 198, llm 169 → 170
    # 0.336: + 4 OmniASR LLM v2 (bezzam; llm releases, listing domain asr)  193 → 197, llm 165 → 169
    # 0.339: + Qwen2.5-Omni-3B, Step-Audio-2-mini-Base/Think, moshika (llm releases)  203 → 207, llm 175 → 179
    # 0.340: + Ming-3.0-tts-0.5B (llm release, listing domain tts)  207 → 208, llm 179 → 180
    # 0.341: + Ming-omni-tts-0.5B  208 → 209, llm 180 → 181
    # 0.345: + Qwen3-Omni-30B-A3B  209 → 210, llm 181 → 182
    # 0.346: + OpenMOSS-Team/MOSS-TTS (llm release, listing domain tts)  210 → 211, llm 182 → 183
    # 0.355: + MiniMax M3.1-Flash (llm release)  211 → 212, llm 183 → 184
    # 0.356: + MiniMax M3, M2.7 (llm releases)  212 → 214, llm 184 → 186
    # 0.357: + 6 Whisper variants (asr releases)  214 → 220, asr +6
    # 0.360: + 9 Whisper .en / Distil-Whisper (asr) + 6 speech siblings (llm releases)  220 → 235, llm 186 → 192
    # 0.361: + BlinkDL rwkv-8-pile rc00 and rc00-hybrid (llm)  235 → 237, llm 192 → 194
    # 0.363: + 11 LLM / MoE releases + Wan2.2-TI2V-5B (release domain video)  237 → 249, llm 194 → 205
    # 0.366: + 12 Qwen3-VL Instruct + FP8 (llm releases, listing domain vlm)  249 → 261, llm 205 → 217
    # 0.368: + 6 Gemma-4 (31B, 26B-A4B, 12B, each base + it; llm releases, listing domain vlm)  261 → 267, llm 217 → 223
    # 0.369: + 4 Gemma-4 E2B / E4B (each base + it; llm releases, listing domain vlm)  267 → 271, llm 223 → 227
    # 0.370: + 6 Granite-4.1 (3B / 8B / 30B, each bf16 + FP8; llm releases)  271 → 277, llm 227 → 233
    # 0.371: + 6 Olmo-3 (7B base/Think/Instruct, 32B base/Think, 3.1-32B-Instruct) + 3 EXAONE-4
    #        (4.0-32B, 4.0.1-32B, 4.0-1.2B)  277 → 286, llm 233 → 242
    # 0.372: + MiMo-V2-Flash (llm)  286 → 287, llm 242 → 243
    # 0.373: + Ling-flash-2.0 (bailing_moe llm release)  287 → 288, llm 243 → 244
    # 0.374: + Step-3.5-Flash (step3p5 llm release)  288 → 289, llm 244 → 245
    # 0.375: + LongCat-Flash-Chat (llm release)  289 → 290, llm 245 → 246
    # 0.376: + Hunyuan-A13B-Instruct (hunyuan_v1_moe llm release)  290 → 291, llm 246 → 247
    # 0.377: + ERNIE-4.5-21B-A3B-PT (ernie4_5_moe llm release)  291 → 292, llm 247 → 248
    # 0.378: + ERNIE-4.5-21B-A3B-Base-PT (ernie4_5_moe llm release, no MTP)  292 → 293, llm 248 → 249
    assert len(files) == 293 and doms.count("llm") == 249
    listed = {x["id"]: x for x in list_models()}
    assert len(listed) + len(unlisted_models()) == 301   # 0.378: +1 ernie-4.5-21b-a3b-base-pt (listed); 0.377: +1 ernie-4.5-21b-a3b-pt (listed); 0.376: +1 hunyuan-a13b-instruct (listed); 0.375: +1 longcat-flash-chat (listed); 0.374: +1 step-3.5-flash (listed); 0.373: +1 ling-flash-2.0 (listed); 0.372: +1 mimo-v2-flash (listed); 0.371: +9 (6 olmo-3, 3 exaone-4; all listed); 0.370: +6 granite-4.1 (all listed); 0.369: +4 gemma-4 E2B/E4B (all listed); 0.368: +6 gemma-4 (all listed); 0.366: +12 qwen3-vl (all listed); 0.363: +12 listed (11 LLM/MoE + wan2.2-ti2v-5b); 0.361: + rwkv8-pile, rwkv8-pile-hybrid (listed); 0.360: +15 speech (4 whisper.en, 5 distil, qwen3-asr-0.6b, qwen2-audio-7b, step-audio-2-r1.1, 2 qwen3-omni thinker, kimi-audio-7b); 0.357: +6 whisper (tiny/base/small/medium/large/large-v2); 0.356: +minimax-m3, +minimax-m2.7; 0.355: +minimax-m3.1-flash; 0.346: +moss-tts; 0.345: +qwen3-omni-30b-a3b; 0.341: +ming-omni; 0.340: +ming-0.5b; 0.339: +4 (moshika, step-base/think, omni-3b); 0.338: + 5 qwen3-tts (listed) 206 → 211; 0.337: + carbon-x (listed) 205 → 206; 0.336: + 4 omniasr-llm-*-v2 (listed) 201 → 205; 0.331: + pixtext-teacher (listed) 200 → 201; 0.330: + pixtext-meanflow (listed) 199 → 200; 0.320: + raon-1-mini-preview (listed); 0.315: + flint (listed); 0.314: + ovisocr3 (merged into startlux-decision-v1-0.8b); 0.311: + ling-3.0-tiny, ling-3.0-flash, ling-3.0-tiny-ubtree (target + draft); 0.294: + lfm2.5-24b-a2b-fp8-dspark (target + draft); 0.292: + 2 jais-3-mova (both listed); 0.291: + 14 (4 pplx-decider, 9 jais-3, mellum2.1; all listed); 0.289: + 5 nemotron-3.5 (all listed: bolt-24gb / -bf16, lightning-nvfp4 / -bf16, super-vl-bf16); 0.288: + 11 ornith (3 listed FP8, 8 merged); 0.287: + ornith-1.5-122b; 0.280: + qwen3.8-flash-next-fp4-local; 0.279: + index-echo-s2st-2b, -9b; 0.278: + nemotron-3.5-super-vl-120b-a12b-nvfp4; 0.276: + drama-3; 0.275: + fireredtts3-base; 0.274: + moshiko-7b; 0.273: + kimi-audio-7b-instruct; 0.272: + fun-asr-nano-2512, step-audio-2-mini; 0.271: + qwen2.5-omni-7b, cosyvoice2-0.5b, fun-cosyvoice3-0.5b-2512; 0.270: + 4 speech (whisper ×2, qwen2-audio, qwen3-asr); 0.269: + ming-3.0-tts-tiny; 0.268: + ming-3.0-asr-tiny; 0.257: + moss-image-base, moss-image-sft (merged); 0.254: + 3 Beam + 2 DFlash2 pairs; 0.253: + 4; 0.191: + 19 releases + microsoft-decision-1 proxy; 0.221: + qwen3.8-27b-ready (target + draft); 0.251: + 7; 0.252: + qwen3-8b-dspine
    assert all(abs(x["param_err"]) <= 0.005 for x in listed.values())
    assert all(abs(x["param_err"]) < 1e-5 for x in listed.values() if get_model(x["id"]).domain in ("gen", "protein", "genomics"))


def test_esmfold_fp32_activation_example():
    """MODEL §12: ESMFold 512 residues, 100T + LPDDR5X, 5.0 → 10.0 s (4 trunk passes since 0.61.2)."""
    s = Scenario(chip=CHIPS["100T"], model="esmfold", mem_id=LP, exec_overlap="stage")   # 0.100: hand-calc pins the stage path
    assert round(evaluate(s).latency, 1) == 5.0
    assert round(evaluate(s.replace("formats_override", (("act", "fp32"),))).latency, 1) == 10.0


def test_fabric_examples_after_bf16_combine():
    """MODEL §19 example A and §19.5 (re-derived: combine / all-reduce wire bytes ≥ bf16 since 0.61.1)."""
    def run(model, lay, phase, batch, fab=Fabric()):
        return evaluate(Scenario(model=model, chip=replace(CHIPS["1P"], instance_sched="wide"),  # 0.71: §19 examples at the wide schedule
                                 mem_id=HBM, layout=lay, node_cards=8,
                                 serving=Serving(phase=phase, batch=batch, moe_rank_max="off"),   # 0.141: pinned
                                 fabric=replace(fab, algo="auto"), exec_overlap="stage"))
    r = run("deepseek-v3", Layout(dp=64, ep=64), "prefill", 64)
    assert round(r.step * 1e3, 1) == 715.9 and r.bound == "LINK"
    r = run("deepseek-v3", Layout(dp=64, ep=64), "prefill", 64, Fabric(enabled=True, oversub=4.0, leaf_nodes=2))
    assert round(r.step * 1e3) == 2453
    r = run("kimi-k2", Layout(dp=1024, ep=1024), "decode", 8192, Fabric(enabled=True, oversub=4.0))
    # 0.63: per-head MLA absorption (W_UK / W_UV) makes decode compute 7.26 ms > link 6.30 ms (0.62.1: 7.22, LINK)
    assert round(r.step * 1e3, 2) == 7.26 and r.bound == "MAC"


def test_neutral_values_are_identities():
    """images = 0 ignores the image size; slc_mib = 0 ignores slc_GBps / policy; fabric off ignores its knobs;
    spec_k = 0 ignores spec_accept; pp_split is inert at PP = 1; package_cards is inert with D2D off."""
    s = Scenario(model="qwen3.8-27b", mem_id=HBM, layout=Layout(tp=2), serving=Serving(batch=8))

    def sig(x):
        d = evaluate(x).summary()
        d.pop("warnings")
        return json.dumps(d, sort_keys=True, default=str)
    base = sig(s)
    assert sig(s.replace("serving.image_w", 333).replace("serving.image_h", 777)) == base
    assert sig(s.replace("chip.slc_GBps", 999.0).replace("chip.slc_policy", "lru")) == base
    assert sig(s.replace("fabric", Fabric(enabled=False, algo="tree", overlap="ports", net_tiers=3, oversub=4.0))) == base
    assert sig(s.replace("serving.spec_accept", 0.3)) == base
    assert sig(s.replace("pp_split", "layers")) == base
    assert sig(s.replace("package_cards", 4)) == base


def test_sweep_images_matches_eval():
    sc = {"model": "kimi-k2.5", "mem_id": HBM, "layout": {"tp": 4, "dp": 2, "ep": 8},
          "serving": {"phase": "prefill", "batch": 2, "image_w": 1920, "image_h": 1080}}
    sw = api.api_sweep({"scenario": sc, "path": "serving.images", "values": [0, 1, 3]})
    s = api.scenario_from_body({"scenario": sc})
    for row, v in zip(sw["rows"], (0, 1, 3)):
        r = evaluate(s.replace("serving.images", v))
        assert row["ttft_ms"] == r.ttft * 1e3 and row["fits"] == r.fits
    assert sw["rows"][0]["ttft_ms"] < sw["rows"][1]["ttft_ms"] < sw["rows"][2]["ttft_ms"]


from tests.pin_legacy import pin_module  # noqa: E402  (0.173)
pin_module(globals())
