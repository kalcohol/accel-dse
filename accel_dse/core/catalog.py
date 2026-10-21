"""Model catalog for core v2: catalog ids → official HF releases → ModelSpec.

Sources: LLM entries of ``data/series_catalog.json`` (labels / aliases) plus
official quantised releases as *separate* entries (as-released dtype) and a
few small references.  Gated repos (meta-llama) use public byte-identical
mirrors (unsloth/*), labelled as mirror provenance.  Video generation and protein entries of the same file are
listed as catalog-only rows (not yet on core v2).  Listing order: vendor (厂商) → family → size.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .domain import from_domain_release
from .model import DATA, ModelSpec, NativeWorkload, from_release, load_release, spec_accept_from_tau

EXTRA = [
    # id, hf_id  (listing label = official repo name)
    ("qwen3-32b-fp8", "Qwen/Qwen3-32B-FP8"),
    ("qwen3-8b-fp8", "Qwen/Qwen3-8B-FP8"),
    ("qwen3-30b-a3b-fp8", "Qwen/Qwen3-30B-A3B-FP8"),
    ("qwen3-235b-a22b-fp8", "Qwen/Qwen3-235B-A22B-FP8"),
    ("qwen3-32b-awq", "Qwen/Qwen3-32B-AWQ"),
    ("qwen3-8b-awq", "Qwen/Qwen3-8B-AWQ"),
    ("qwen3-4b", "Qwen/Qwen3-4B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("llama-3.1-8b", "unsloth/Llama-3.1-8B-Instruct"),
    ("llama-3.3-70b", "unsloth/Llama-3.3-70B-Instruct"),
    ("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct"),
    ("llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct"),
    # 0.66 external validation only (published TensorRT-LLM / Databricks measurements; core/extval.py)
    ("llama-2-7b", "NousResearch/Llama-2-7b-hf"),
    ("llama-2-70b", "NousResearch/Llama-2-70b-hf"),
    ("mistral-7b", "mistralai/Mistral-7B-v0.1"),
]
# 0.106 genomics: Carbon DNA foundation models (stock LlamaForCausalLM, 6-mer + Qwen3 BPE hybrid vocab, BF16).
# Appended after the domain releases so the order of every earlier entry (and the fingerprints over it) is unchanged.
EXTRA_TAIL = [
    ("carbon-8b", "HuggingFaceBio/Carbon-8B"),
    ("carbon-3b", "HuggingFaceBio/Carbon-3B"),
    ("carbon-500m", "HuggingFaceBio/Carbon-500M"),
    # 0.114 coverage refresh (official releases 2026-10-12 / 13; public config.json + safetensors headers)
    ("lfm2.5-24b-a2b", "LiquidAI/LFM2.5-24B-A2B"),
    ("lfm2.5-24b-a2b-fp8", "LiquidAI/LFM2.5-24B-A2B-FP8"),
    ("ling-3.1-flash", "inclusionAI/Ling-3.1-flash"),
    # 0.144 catalog refresh (official releases 2026-09 / 10; public config.json + safetensors headers, as released)
    ("step-5-preview", "stepfun-ai/Step-5-Preview"),
    ("mimo-v2.6-flash", "XiaomiMiMo/MiMo-V2.6-Flash-RL"),
    ("intern-s2-397b", "internlm/Intern-S2-397B"),
    ("minicpm-v-4.7-35b-a3b", "openbmb/MiniCPM-V-4.7-35B-A3B"),
    # 0.145 (official releases 2026-09-04 / 2026-10; public config.json + safetensors headers, as released)
    ("mimo-v2.6-pro", "XiaomiMiMo/MiMo-V2.6-Pro-RL"),
    ("ling-3.0-flash-vl", "inclusionAI/Ling-3.0-flash-VL"),
    # 0.170 genomics: Arc Institute Evo 2 (StripedHyena 2, single-nucleotide byte tokens, causal DNA LM); PyTorch zip
    # checkpoints read over HTTP range + the evo2 repo inference configs (scripts/build_evo2_release.py)
    ("evo2-40b", "arcinstitute/evo2_40b"),
    ("evo2-7b", "arcinstitute/evo2_7b"),
    ("evo2-1b", "arcinstitute/evo2_1b_base"),
    # 0.174 Meta Muse Spark Open (2026-10-15).  meta-models/* is gated: the BF16 entry is derived from the public AMD
    # release headers (same tensors; release JSON "derived"); AMD FP8 (PTPC) / MXFP4 releases read as published.
    ("muse-spark-open", "meta-models/Muse-Spark-Open-816B-A42B"),
    ("muse-spark-open-fp8", "amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8"),
    ("muse-spark-open-mxfp4", "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"),
    ("muse-spark-open-nvfp4", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"),   # 0.175 推导 (gated; see release "derived")
    # 0.175: the blog's serve command (NVFP4 target + Inferact DSpark draft, num_speculative_tokens 7)
    ("muse-spark-open-nvfp4-dspark", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4+Inferact/Muse-Spark-Open-DSpark"),
    # 0.221: Qwen3.8-27B target + StartLux ReaDy DSpark draft with the recurrent-state reader (both public, 2026-10-17)
    ("qwen3.8-27b-ready", "Qwen/Qwen3.8-27B+startlux-models/ReaDy-Qwen3.8-27B"),
    # 0.294: LFM2.5-24B-A2B FP8 target + LiquidAI DSpark draft (the README's SGLang serve command; repo 2026-10-20 JST)
    ("lfm2.5-24b-a2b-fp8-dspark", "LiquidAI/LFM2.5-24B-A2B-FP8+LiquidAI/LFM2.5-24B-A2B-DSpark"),
    # 0.252: Qwen3-8B target + tencent DSpine (DFlash anchor-row block draft + selected-prefix code bridge, 2026-10-16)
    ("qwen3-8b-dspine", "Qwen/Qwen3-8B+tencent/Qwen3-8B-DSpine-b16"),
    # 0.191 decision models ("System One": one forward, letter logits over the listed options; serving.labels).
    # StartLux Labs, CC BY-NC-4.0, Qwen3.5 fine-tunes, bf16 as released (public config.json + safetensors headers)
    ("qwen3.5-9b", "Qwen/Qwen3.5-9B"),
    ("startlux-decision-v1-27b", "startlux-models/StartLux-Decision-v1-27B"),
    ("startlux-decision-v1-9b", "startlux-models/StartLux-Decision-v1-9B"),
    ("startlux-decision-v1-4b", "startlux-models/StartLux-Decision-v1-4B"),
    ("startlux-decision-v1-2b", "startlux-models/StartLux-Decision-v1-2B"),
    ("startlux-decision-v1-0.8b", "startlux-models/StartLux-Decision-v1-0.8B"),
    ("startlux-decision-v1-35b-a3b", "startlux-models/StartLux-Decision-v1-35B-A3B"),
    ("startlux-decision-v1.1-27b-instant", "startlux-models/StartLux-Decision-v1.1-27B-Instant"),
    ("startlux-decision-v1.1-27b-autothink", "startlux-models/StartLux-Decision-v1.1-27B-AutoThink"),
    ("startlux-decision-v1.1-9b-instant", "startlux-models/StartLux-Decision-v1.1-9B-Instant"),
    ("startlux-decision-v1.1-9b-autothink", "startlux-models/StartLux-Decision-v1.1-9B-AutoThink"),
    ("startlux-decision-v1.1-4b-instant", "startlux-models/StartLux-Decision-v1.1-4B-Instant"),
    ("startlux-decision-v1.1-4b-autothink", "startlux-models/StartLux-Decision-v1.1-4B-AutoThink"),
    ("startlux-decision-v1.1-2b-instant", "startlux-models/StartLux-Decision-v1.1-2B-Instant"),
    ("startlux-decision-v1.1-0.8b-instant", "startlux-models/StartLux-Decision-v1.1-0.8B-Instant"),
    # 0.191 RWKV-7 G1k (BlinkDL, Apache-2.0, 2026-09-30): rwkv7 linear recurrence, no KV cache
    ("rwkv7-g1k-13.3b", "RWKV/RWKV7-G1k-13.3B-20260930"),
    ("rwkv7-g1k-7.2b", "RWKV/RWKV7-G1k-7.2B-20260930"),
    ("rwkv7-g1k-2.9b", "RWKV/RWKV7-G1k-2.9B-20260930"),
    ("rwkv7-g1k-1.5b", "RWKV/RWKV7-G1k-1.5B-20260930"),
    # 0.251 catalog freshness sweep (official releases 2026-10-14 … 16; public config.json + safetensors headers, as released)
    ("hy-memory-30b-a3b", "tencent/Hy-Memory-30B-A3B"),
    ("hy-memory-30b-a3b-fp8", "tencent/Hy-Memory-30B-A3B-FP8"),
    ("step-5-preview-nvfp4", "stepfun-ai/Step-5-Preview-NVFP4"),
    # same architecture as a catalog entry (merged: resolvable + validated, not listed separately)
    ("userrepgen-4b", "microsoft/UserRepGen-4B"),
    ("flux-3-image-klein-pup", "black-forest-labs/flux-3-image-klein-pup"),
    # 0.253 freshness sweep (official releases 2026-10-15 … 17)
    ("dory1.0-4b", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"),
    ("dory1.0-4b-mxfp8", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8"),
    ("dory1.0-4b-nvfp4", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4"),
    ("mai-gym-9b", "Tongyi-MAI/MAI-Gym-9B"),
    # 0.254 Reflection Beam-1.0 (2026-10-07 / 10-10; BF16 / FP8 / NVFP4 as released) + its DFlash2 draft (10-13)
    ("beam-1.0-501b-a23b", "reflection/Beam-1.0-501B-A23B"),
    ("beam-1.0-501b-a23b-fp8", "reflection/Beam-1.0-501B-A23B-FP8"),
    ("beam-1.0-501b-a23b-nvfp4", "reflection/Beam-1.0-501B-A23B-NVFP4"),
    ("beam-1.0-501b-a23b-nvfp4-dflash2", "reflection/Beam-1.0-501B-A23B-NVFP4+reflection/Beam-1.0-501B-A23B-DFlash2"),
    # 0.254: closes the 0.253 deferral (§200.4) — the conv / selector code is public (vLLM qwen3_dflash2.py)
    ("muse-spark-open-nvfp4-dflash2", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4+nvidia/Muse-Spark-Open-816B-A42B-DFlash2"),
    # 0.268 freshness: inclusionAI Ming-3.0-asr-tiny (2026-10-14; BailingMoeV3 KDA/MLA decoder + Whisper encoder, bf16)
    ("ming-3.0-asr-tiny", "inclusionAI/Ming-3.0-asr-tiny"),
    # 0.269: inclusionAI Ming-3.0-tts-tiny (2026-09-22; same decoder + flow-matching DiT head + streaming AudioVAE)
    ("ming-3.0-tts-tiny", "inclusionAI/Ming-3.0-tts-tiny"),
    # 0.270 speech round: Whisper-large-v3 / -turbo (encoder-decoder, cross-attention decode, 30 s padded windows),
    # Qwen2-Audio-7B-Instruct (Whisper-v3 encoder + pool + Qwen-7B), Qwen3-ASR-1.7B (AuT encoder + Qwen3-1.7B)
    ("whisper-large-v3", "openai/whisper-large-v3"),
    ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
    ("qwen2-audio-7b-instruct", "Qwen/Qwen2-Audio-7B-Instruct"),
    ("qwen3-asr-1.7b", "Qwen/Qwen3-ASR-1.7B"),
    # 0.271 speech round 2: Qwen2.5-Omni-7B (Thinker Qwen2.5-7B + Whisper-v3 audio tower; Talker + token2wav speech out)
    ("qwen2.5-omni-7b", "Qwen/Qwen2.5-Omni-7B"),
    # 0.271: CosyVoice2-0.5B / Fun-CosyVoice3-0.5B-2512 (Qwen2.5-0.5B LM → 25 Hz tokens → chunked flow matching → HiFT;
    # torch .pt checkpoints read by scripts/fetch_pt_release.py)
    ("cosyvoice2-0.5b", "FunAudioLLM/CosyVoice2-0.5B"),
    ("fun-cosyvoice3-0.5b-2512", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
    # 0.272 speech round 3: Fun-ASR-Nano-2512 (transformers fun_asr_nano: SAN-M 70-layer encoder + adaptor + Qwen3-0.6B)
    ("fun-asr-nano-2512", "FunAudioLLM/Fun-ASR-Nano-2512-hf"),
    # 0.272: Step-Audio-2-mini (Whisper-style encoder + AvgPool + conv Adaptor → Qwen2.5-7B-shaped LLM; audio in)
    ("step-audio-2-mini", "stepfun-ai/Step-Audio-2-mini"),
    # 0.273 speech round 4: Kimi-Audio-7B-Instruct (Whisper-large-v3 encoder + adaptor → Qwen2-7B trunk with a 6-layer
    # audio branch; text + 12.5 Hz audio tokens per step → chunked flow-matching DiT detokenizer → BigVGAN)
    ("kimi-audio-7b-instruct", "moonshotai/Kimi-Audio-7B-Instruct"),
    # 0.274 speech round 5: Kyutai Moshi (moshiko, bf16) — full-duplex: temporal 7B + depth transformer (8 codebooks
    # per 12.5 Hz frame) + Mimi codec; architecture from the moshi package loaders.py (no HF config)
    ("moshiko-7b", "kyutai/moshiko-pytorch-bf16"),
    # 0.275: FireRedTTS3-Base (fp32) — Qwen3-1.7B backbone → per-step DiT flow head (10 steps × CFG 2) + patch encoder,
    # RedAE decode after generation; backbone config from the code (fireredtts3_base.py), see core/fireredtts3.py
    ("fireredtts3-base", "FireRedTeam/FireRedTTS3"),
    # 0.276: Fish Audio Drama 3 — Qwen3.5-35B-A3B-shaped MoE backbone (FP8 experts) + 13-layer codebook-MTP tower (one
    # row per backbone position, own KV) → 10-codebook frame per step @ 44.1 kHz / 2048; DAC codec not costed
    ("drama-3", "fishaudio/drama-3"),
    # 0.278: NVIDIA Nemotron-3.5-Super-VL (2026-10-12, as released NVFP4 / FP8 mixed) — Nemotron-H hybrid: 40 Mamba-2 +
    # 8 GQA + 40 latent-MoE layers (512 experts top-22, relu²), 1 MTP block pair, RADIO ViT-H vision
    ("nemotron-3.5-super-vl-120b-a12b-nvfp4", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"),
    # 0.279: bilibili Index-Echo S2ST (2026-10-12 v4.1) — speech → translated speech: Qwen3-Omni AuT + connector +
    # Qwen3.5 ST decoder (2B / 9B) → Hidden2CV mapper (fp32) → CosyVoice3 (= Fun-CosyVoice3-0.5B-2512 files)
    ("index-echo-s2st-2b", "IndexTeam/Index-Echo-S2ST-2B"),
    ("index-echo-s2st-9b", "IndexTeam/Index-Echo-S2ST-9B"),
    # 0.280: nvidia Qwen3.8-Flash-Next-FP4-Local (2026-10-08, ModelOpt MIXED_PRECISION as released: NVFP4 W4A4 routed
    # experts with compressed (csf) scales, MXFP8 attention / linear-attention / shared experts, W4A16 NVFP4 MTP experts)
    ("qwen3.8-flash-next-fp4-local", "nvidia/Qwen3.8-Flash-Next-FP4-Local"),
    # 0.287: Ornith AI Ornith-1.5-122B (repo 2026-10-16, made public with its FP8 / GGUF / MLX siblings 2026-10-20; MIT)
    # — Qwen3.5-based post-train, qwen3_5_moe as released bf16: 48 layers (3 Gated-DeltaNet : 1 gated GQA), 256 experts
    # top-8 + shared expert, hidden 3072, 1 MTP layer, Qwen3.5 ViT (27 blocks).  No Qwen3.5-122B-A10B entry exists, so
    # this is a new size point, not a merge.  The FP8 sibling ships no config.json / index yet (HTTP 404) → not added.
    ("ornith-1.5-122b", "ornith-ai/Ornith-1.5-122B"),
    # 0.288: the rest of the Ornith family (ornith-ai, MIT; Qwen3.5 post-trains, public config.json + safetensors headers).
    # Official FP8 = compressed-tensors W8A8 float (per-channel weights, dynamic per-token activations) on the routed /
    # shared experts and full-attention projections; linear attention, router, embeddings, head, MTP, ViT stay bf16.
    # No Qwen3.5 FP8 release is catalogued → listed (new as-released dtype points).
    ("ornith-1.5-122b-fp8", "ornith-ai/Ornith-1.5-122B-FP8"),
    ("ornith-1.5-397b-fp8", "ornith-ai/Ornith-1.5-397B-FP8"),
    ("ornith-1.5-35b-a3b-fp8", "ornith-ai/Ornith-1.5-35B-A3B-FP8"),
    # bf16 sizes: config and header shapes identical to a listed entry (MERGED); 1.0 = 1.5 without the MTP head
    ("ornith-1.5-397b", "ornith-ai/Ornith-1.5-397B"),
    ("ornith-1.5-35b-a3b", "ornith-ai/Ornith-1.5-35B-A3B"),
    ("ornith-1.5-9b", "ornith-ai/Ornith-1.5-9B"),
    ("ornith-1.0-397b", "ornith-ai/Ornith-1.0-397B"),
    ("ornith-1.0-397b-fp8", "ornith-ai/Ornith-1.0-397B-FP8"),
    ("ornith-1.0-35b", "ornith-ai/Ornith-1.0-35B"),
    ("ornith-1.0-35b-fp8", "ornith-ai/Ornith-1.0-35B-FP8"),
    ("ornith-1.0-9b", "ornith-ai/Ornith-1.0-9B"),
    # 0.289: NVIDIA Nemotron 3.5 family (nvidia, OpenMDW-1.1).  Bolt 74B-A9B (repos 2026-10-07 / 09, model card release
    # 10/15) = Star-Elastic compression of Super-VL 120B-A12B: nemotron_h_puzzle — same 88-layer Mamba-2 / GQA / latent-MoE
    # layout, per-layer routed experts 240–496 and widths 1536–2560 (block_configs), top-9, no MTP; same RADIO ViT-H.
    # 24GB = as served (vLLM): ModelOpt MIXED_PRECISION — IQ2_XXS routed + shared experts, NVFP4 W4A16 Mamba in / out,
    # Q8_0 attention q/k/v/o, latent projections and lm_head, bf16 embeddings / router / norms / ViT.
    ("nemotron-3.5-bolt-74b-a9b-24gb", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"),
    ("nemotron-3.5-bolt-74b-a9b-bf16", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16"),
    # Lightning 30B-A3B (2026-08-01 / 04): nemotron_h text-only, 52 layers, 128 experts top-6 (no latent), 1 MTP layer;
    # NVFP4 = ModelOpt MIXED_PRECISION (W4A16 NVFP4 experts / shared / lm_head, FP8 attention, fp8 KV)
    ("nemotron-3.5-lightning-30b-a3b-nvfp4", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"),
    ("nemotron-3.5-lightning-30b-a3b-bf16", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
    ("nemotron-3.5-super-vl-120b-a12b-bf16", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"),
    # 0.291 Perplexity pplx-decider (Apache-2.0; decision models: Qwen3_5Model backbone + readout.safetensors [255, h],
    # no lm_head).  v1 (2026-10-02 JST) causal, last-token readout; v1.1 (10-05) noncausal full attention; v1.2 / Flash
    # v1.2 (10-15 / 10-16) noncausal + joint requests (one readout per question marker, serving.questions)
    ("pplx-decider-v1.2", "perplexity-ai/pplx-decider-v1.2"),
    ("pplx-decider-flash-v1.2", "perplexity-ai/pplx-decider-flash-v1.2"),
    ("pplx-decider-v1.1-27b", "perplexity-ai/pplx-decider-v1.1-27b"),
    ("pplx-decider-v1-27b", "perplexity-ai/pplx-decider-v1-27b"),
    # 0.291 Jais 3 (Inception / MBZUAI IFM, Apache-2.0; k2_horizon, 2026-09-29 JST): BF16 originals (inception42/*) and
    # IFM's FP8 releases (compressed-tensors W8A8 float, 128×128 block weights + dynamic per-group-128 activations; the
    # 375B-A23B keeps attention, shared experts, routers and the first 3 dense layers bf16).  MoVA-36B-A4B: added in 0.292
    # (routed value experts modelled, below); not added: the 0.9B FP8 (no such release), Uno adapters
    ("jais-3-375b-a23b", "inception42/Jais-3-375B-A23B"),
    ("jais-3-32b", "inception42/Jais-3-32B"),
    ("jais-3-7b", "inception42/Jais-3-7B"),
    ("jais-3-3.7b", "inception42/Jais-3-3.7B"),
    ("jais-3-0.9b", "inception42/Jais-3-0.9B"),
    ("jais-3-375b-a23b-fp8", "IFM/Jais-3-375B-A23B-FP8"),
    ("jais-3-32b-fp8", "IFM/Jais-3-32B-FP8"),
    ("jais-3-7b-fp8", "IFM/Jais-3-7B-FP8"),
    ("jais-3-3.7b-fp8", "IFM/Jais-3-3.7B-FP8"),
    # 0.292 Jais-3-MoVA-36B-A4B (inception42, 2026-09-29 JST; IFM FP8 sibling): k2_horizon with MoVA attention in the
    # 45 sparse layers (64 routed value experts top-4 replace v_proj) and a softplus attention output gate in every
    # layer; 100 experts top-8 + 1 shared.  As released: the inception42 safetensors are fp32 (config dtype bf16);
    # the IFM FP8 (quant_method fp8, 128 × 128 blocks, dynamic activations) quantises only the routed FFN experts —
    # value experts, v_router, attention, shared expert, dense layers, embeddings and head stay bf16
    ("jais-3-mova-36b-a4b", "inception42/Jais-3-MoVA-36B-A4B"),
    ("jais-3-mova-36b-a4b-fp8", "IFM/Jais-3-MoVA-36B-A4B-FP8"),
    # 0.291 JetBrains Mellum2.1 Thinking (Apache-2.0, 2026-09-20 JST): transformers mellum (Qwen3-MoE attention with
    # q / k norm, 3:1 sliding 1024 / full, 64 experts top-8), bf16; the card's MTP head is not released yet
    ("mellum2.1-12b-a2.5b-thinking", "JetBrains/Mellum2.1-12B-A2.5B-Thinking"),
    # 0.311 inclusionAI Ling-3.0 text releases (2026-08-10 / 08-03 JST; bailing_hybrid KDA + MLA, the Ling-3.1-flash
    # builder) + Ling-3.0-tiny-UBTree (2026-10-21 22:49 JST: DFlash block backbone + bigram selector, tree verify)
    ("ling-3.0-tiny", "inclusionAI/Ling-3.0-tiny"),
    ("ling-3.0-flash", "inclusionAI/Ling-3.0-flash"),
    ("ling-3.0-tiny-ubtree", "inclusionAI/Ling-3.0-tiny+inclusionAI/Ling-3.0-tiny-UBTree"),
    # 0.314 ATH-MaaS OvisOCR3 (Apache-2.0, 2026-10-21 11:36 JST; document OCR / parsing): Qwen3.5-0.8B post-train,
    # transformers qwen3_5 VLM as released bf16 — same config / header shapes as StartLux-Decision-v1-0.8B except the
    # MTP head it does not ship → MERGED (resolvable + validated, not listed)
    ("ovisocr3", "ATH-MaaS/OvisOCR3"),
]
# 0.191 architecture proxies: closed weights, architecture stated by the vendor → evaluated on the named public config
# (id, config release, provider key, label, note)
PROXIES = [
    ("microsoft-decision-1", "Qwen/Qwen3.5-9B", "microsoft", "Microsoft-Decision-1（架构代理 · Qwen3.5-9B）",
     "「架构代理」：Microsoft-Decision-1（2026-10-09，闭源）仅公开说明由 Qwen3.5-9B 后训练"
     "（commandline.microsoft.com/microsoft-decision-1-model-foundry/）→ 按 Qwen3.5-9B 发布配置与 bf16 推断；"
     "实际部署精度 / 结构改动未公开"),
]
DECISION_IDS = {"microsoft-decision-1",      # + every startlux-decision-* id (decision readout, serving.labels)
                "pplx-decider-v1.2", "pplx-decider-flash-v1.2", "pplx-decider-v1.1-27b", "pplx-decider-v1-27b"}   # 0.291
MIRRORS = {"unsloth/": "meta-llama (gated) → public mirror unsloth/*, same safetensors",
           "NousResearch/": "meta-llama (gated) → public mirror NousResearch/*, same safetensors"}

# Non-autoregressive releases on core v2 (0.41 / 0.42): id, hf_id, builder, native workload (official config / README).
# The remaining protein entries of series_catalog.json stay catalog-only until modelled.
_WAN_SRC = ("Wan2.1 官方 README / generate.py：t2v {res}、81 帧、16 fps、sample_steps 50、guide_scale 5.0（CFG）、"
            "VAE stride (4, 8, 8)、patch (1, 2, 2)、text_len 512")
_COG_SRC = ("transformer/config.json：sample_frames 49、潜空间 60×90（×8 = 480×720）、temporal_compression_ratio 4；"
            "diffusers CogVideoXPipeline 默认 50 步、guidance_scale 6（CFG）、8 fps")
_WAN22_SRC = ("Wan2.2 官方 README / generate.py（t2v-A14B）：1280×720、81 帧、16 fps、sample_steps 40、guide_scale (3.0, 4.0)"
              "（CFG）、boundary 0.875（高/低噪声专家切换）、Wan2.1 VAE stride (4, 8, 8)、patch (1, 2, 2)")
_HY_SRC = ("HunyuanVideo 官方 sample_video.py：--video-size 720 1280、--video-length 129、--infer-steps 50、"
           "--embedded-cfg-scale 6.0（guidance 蒸馏，无 CFG）、text_len 256、24 fps；VAE 4×8×8、patch 1×2×2")
_LTX_SRC = ("diffusers LTXPipeline 默认：704×512、161 帧、frame_rate 25、50 步、guidance_scale 3（CFG）、"
            "max_sequence_length 128；VAE 32×32 空间 × 8 时间、patch 1")
_MOCHI_SRC = ("genmo/mochi demos/cli.py 默认：848×480、163 帧、num_steps 64、cfg_scale 6.0（CFG）；30 fps；"
              "VAE 6× 时间 × 8× 空间、patch 2×2、T5 256 token")
_OS_SRC = ("Open-Sora 1.2 README 推理示例：--num-frames 4s（102 帧）--resolution 720p --aspect-ratio 9:16（720×1280）；"
           "rflow 30 步、cfg_scale 7.0（CFG）、24 fps；VAE 17 帧 → 5 潜帧、8× 空间、patch (1, 2, 2)、T5 300 token")
_H3_SRC = ("diffusers MiniMax-H3 文档 / pipeline：画布 1344×768（短边 768）、num_frames 124（17n+5）、24 fps、"
           "SGLang 默认 num_inference_steps 50（49 次前向）、CFG 蒸馏；VisualVAE f16t4d24 + patch (1, 2, 2)、"
           "音频 40 latent/s × 立体声；文本 512 token「假设」")
_ESM_SRC = "config.json：max_position_embeddings 1026（1022 残基 + <cls>/<eos>）；默认 512 残基为工作负载「假设」"
_CARBON_A_SRC = ("Carbon-A 模型卡：16,384 token 窗口（98,304 bp，6-mer）、长记录按重叠 1,024 token（6,144 bp）的窗口读入；"
                 "推理 BF16 + FlashAttention-2")
_ESMFOLD_SRC = ("esmfold_v1 config.json（esmfold_config.trunk：48 块、max_recycles 4、结构模块 8 次迭代）；"
                "默认 512 残基为工作负载「假设」")
_OF_SRC = ("openfold/config.py：data.predict max_msa_clusters 512、max_extra_msa 1024、max_templates 4；"
           "max_recycling_iters 3（主干 4 遍）；默认 512 残基「假设」")
_AF2_SRC = ("alphafold/model/config.py：model_1_ptm max_extra_msa 5120、reduce_msa_clusters_by_max_templates（512 − 4 = "
            "508 行 MSA 聚类）、max_templates 4、num_recycle 3（主干 4 遍）、num_ensemble 1；默认 512 残基「假设」")
_BOLTZ_SRC = ("boltz v0.4.1 main.py predict：recycling_steps 3、sampling_steps 200、diffusion_samples 1、"
              "max_msa_seqs 4096；默认 512 残基、每残基 8 个重原子「假设」")
_PX_SRC = ("Protenix v0.5.0 configs_base.py：N_cycle 4、sample_diffusion N_step 200 / N_sample 5；configs_data.py："
           "MSA sample_cutoff 2048、模板关闭；默认 512 残基、每残基 8 个重原子「假设」")
_QI21_SRC = ("Qwen/Qwen-Image-2.1-Turbo README：QwenImage21Pipeline、保存的 8 步 sample_sigmas、CFG=1；分辨率预设 1:1 = 2048×2048"
             "（文本 256 token「假设」）")
_QI_SRC = ("Qwen/Qwen-Image README：aspect_ratios 1:1 = 1328×1328、num_inference_steps 50、true_cfg_scale 4.0（CFG）"
           "（文本 256 token「假设」）")
_FLUX2K_SRC = ("black-forest-labs/FLUX.2-klein-4B README：1024×1024、num_inference_steps 4、guidance_scale 1.0（蒸馏，无 CFG）；"
               "Flux2KleinPipeline max_sequence_length 512（padding=max_length）")
_LBV_SRC = ("Robbyant/lingbot-video README（MoE-DMD）：--scheduler dmd_student --steps 8 --guidance_scale 1、832×480、"
            "121 帧、24 fps；pipeline_lingbot_video.py：VAE ×4 时间 / ×8 空间（AutoencoderKLWan），文本 = Qwen3-VL 隐状态"
            "（裁掉系统模板，padding=longest），按示例结构化 JSON 提示 ≈ 1k token 计「假设」")
_FLUX3_SRC = ("black-forest-labs/flux-3-image-klein README / config.json：sampler 1024×1024、steps 16、guidance null（flow-map 蒸馏）；"
              "Base：50 步、guidance 3.5；Qwen3-VL-4B 文本（8 层隐状态拼接、pad_multiple 80，提示上采样后的完整描述 ≈ 320 token「假设」）；"
              "AutoencoderKLFlux2 ×8 + 2×2 打包")
_MOSS_SRC = ("OpenMOSS-Team/MOSS-Image-Base README / pipeline_moss_image.py：1024×1024、num_inference_steps 50、guidance_scale 3"
             "（CFG，两次前向；uncond 跳过第 22 块）、shift 5；Qwen3.5-9B 文本（末层隐状态，逐样本真实长度，≈ 256 token「假设」）；"
             "AutoencoderKLFlux2 ×8（FLUX.2 small decoder，fp32）+ 2×2 打包")
DOMAIN_RELEASES = [
    ("wan2.1-14b", "Wan-AI/Wan2.1-T2V-14B", "wan",
     NativeWorkload("gen", frames=81, height=720, width=1280, fps=16, vae_frames="wan", source=_WAN_SRC.format(res="720P（1280×720）"))),
    ("wan2.1-1.3b", "Wan-AI/Wan2.1-T2V-1.3B", "wan",
     NativeWorkload("gen", frames=81, height=480, width=832, fps=16, vae_frames="wan", source=_WAN_SRC.format(res="480P（832×480）"))),
    ("cogvideox-5b", "zai-org/CogVideoX-5b", "cogvideox",
     NativeWorkload("gen", frames=49, height=480, width=720, fps=8, source=_COG_SRC)),
    ("cogvideox-2b", "zai-org/CogVideoX-2b", "cogvideox",
     NativeWorkload("gen", frames=49, height=480, width=720, fps=8, source=_COG_SRC)),
    ("wan2.2-a14b", "Wan-AI/Wan2.2-T2V-A14B", "wan22",
     NativeWorkload("gen", frames=81, height=720, width=1280, fps=16, vae_frames="wan", steps=40, source=_WAN22_SRC)),
    ("hunyuanvideo", "hunyuanvideo-community/HunyuanVideo", "hunyuan",
     NativeWorkload("gen", frames=129, height=720, width=1280, fps=24, steps=50, cfg=1, text_tokens=256, source=_HY_SRC)),
    ("ltx-video", "Lightricks/LTX-Video", "ltx",
     NativeWorkload("gen", frames=161, height=512, width=704, fps=25, vae_t=8, vae_s=32, steps=50, cfg=2,
                    text_tokens=128, source=_LTX_SRC)),
    ("mochi-1", "genmo/mochi-1-preview", "mochi",
     NativeWorkload("gen", frames=163, height=480, width=848, fps=30, vae_t=6, vae_s=8, steps=64, cfg=2,
                    source=_MOCHI_SRC)),
    ("opensora-stdit3", "hpcai-tech/OpenSora-STDiT-v3", "stdit",
     NativeWorkload("gen", frames=102, height=1280, width=720, fps=24, vae_t=4, vae_s=8, vae_frames="chunk17",
                    steps=30, cfg=2, source=_OS_SRC)),
    ("minimax-h3", "MiniMaxAI/MiniMax-H3", "h3",
     NativeWorkload("gen", frames=124, height=768, width=1344, fps=24, vae_t=4, vae_s=16, vae_frames="h3",
                    audio_per_s=40, audio_channels=2, steps=49, cfg=1, text_tokens=512, source=_H3_SRC)),
    ("esm2-3b", "facebook/esm2_t36_3B_UR50D", "esm", NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_ESM_SRC)),
    ("esm2-650m", "facebook/esm2_t33_650M_UR50D", "esm", NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_ESM_SRC)),
    # 0.43 structure models (checkpoint headers read over HTTP; see scripts/fetch_torch_ckpt.py)
    ("esmfold", "facebook/esmfold_v1", "esmfold", NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_ESMFOLD_SRC)),
    ("openfold", "gh:aqlaboratory/openfold", "openfold",
     NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_OF_SRC)),
    ("alphafold2", "gh:google-deepmind/alphafold", "alphafold2",
     NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_AF2_SRC)),
    ("boltz-1", "boltz-community/boltz-1", "boltz1",
     NativeWorkload("protein", seq_len=512, steps=1, cfg=1, atoms_per_res=8.0, source=_BOLTZ_SRC)),
    ("protenix", "gh:bytedance/Protenix", "protenix",
     NativeWorkload("protein", seq_len=512, steps=1, cfg=1, atoms_per_res=8.0, source=_PX_SRC)),
    # 0.106 genomics: bidirectional token-classification encoder (gene annotation)
    ("carbon-a-1.2b", "HuggingFaceBio/Carbon-A-1.2B", "generanno",
     NativeWorkload("genomics", seq_len=16384, steps=1, cfg=1, kmer=6, overlap=1024, source=_CARBON_A_SRC)),
    # 0.115 image DiT (gen domain, one latent frame; appended last so earlier entries keep their order)
    ("qwen-image-2.1-turbo", "Qwen/Qwen-Image-2.1-Turbo", "qwenimage21",
     NativeWorkload("gen", frames=1, height=2048, width=2048, vae_s=16, steps=8, cfg=1, source=_QI21_SRC)),
    ("qwen-image", "Qwen/Qwen-Image", "qwenimage",
     NativeWorkload("gen", frames=1, height=1328, width=1328, vae_s=8, steps=50, cfg=2, source=_QI_SRC)),
    ("flux.2-klein-4b", "black-forest-labs/FLUX.2-klein-4B", "flux2",
     NativeWorkload("gen", frames=1, height=1024, width=1024, vae_s=8, steps=4, cfg=1, source=_FLUX2K_SRC)),
    # 0.144 MoE video DiT (transformer/ subfolder of the diffusers release)
    ("lingbot-video-moe-dmd", "robbyant/lingbot-video-moe-dmd-30b-a3b", "lingbot",
     NativeWorkload("gen", frames=121, height=480, width=832, fps=24, vae_frames="wan", steps=8, cfg=1,
                    text_tokens=1024, source=_LBV_SRC)),
    # 0.251 FLUX 3 Image Klein (2026-10-13, gated "auto" → read from the byte-identical public mirror unsloth/*, 2026-10-16)
    ("flux-3-image-klein", "black-forest-labs/flux-3-image-klein", "flux3",
     NativeWorkload("gen", frames=1, height=1024, width=1024, vae_s=8, steps=16, cfg=1, text_tokens=320, source=_FLUX3_SRC)),
    ("flux-3-image-klein-base", "black-forest-labs/flux-3-image-klein-base", "flux3",
     NativeWorkload("gen", frames=1, height=1024, width=1024, vae_s=8, steps=50, cfg=2, text_tokens=320, source=_FLUX3_SRC)),
    # 0.257 MOSS-Image (OpenMOSS-Team, 2026-10-11, public diffusers repo + custom pipeline); SFT = same transformer
    # shapes / dtypes and the same pipeline (README: "same architecture and inference pipeline") → merged into Base
    ("moss-image-base", "OpenMOSS-Team/MOSS-Image-Base", "moss",
     NativeWorkload("gen", frames=1, height=1024, width=1024, vae_s=8, steps=50, cfg=2, text_tokens=256, source=_MOSS_SRC)),
    ("moss-image-sft", "OpenMOSS-Team/MOSS-Image-SFT", "moss",
     NativeWorkload("gen", frames=1, height=1024, width=1024, vae_s=8, steps=50, cfg=2, text_tokens=256, source=_MOSS_SRC)),
]
_DOMAIN_IDS = {i for i, *_ in DOMAIN_RELEASES}
# GitHub-only releases: the repo name alone does not say which checkpoint is evaluated
DOMAIN_LABELS = {"alphafold2": "AlphaFold 2 · model_1_ptm", "openfold": "OpenFold · finetuning_ptm_2",
                 "protenix": "Protenix v0.5.0"}
# catalog-only rows that stay off core v2, with the reason shown in the listing
OFFLINE_REASONS = {
    "alphafold3": "权重需向 Google DeepMind 申请（表单审批、条款禁止再分发），没有可公开读取的发布文件来核对张量形状 → "
                  "不按论文尺寸臆造；同类架构可用 Protenix / Boltz-1（AF3 复现，公开权重）评估",
}

# ---------------------------------------------------------------- listing: vendor (厂商) → family; domain is only a badge
DOMAINS = {"llm": "LLM（文本）", "vlm": "VLM（多模态：语言主干 + 视觉编码器）", "asr": "语音识别（语言主干 + 音频编码器）",
           "tts": "语音合成（语言主干 + 流匹配 + 音频 VAE / 声码器）",
           "omni": "全模态（语言主干 + 音频编码器；语音输出：Omni Talker + token2wav / Step-Audio-2 token2wav）", "gen": "图像 / 视频生成（DiT）", "protein": "蛋白质",
           "genomics": "基因组（DNA 编码器）"}
OFFLINE_DOMAINS = {"video": "gen", "protein": "protein"}     # series_catalog domain → listing domain (not on core v2)
PROVIDERS = [   # key, label, orgs (HF org, or gh:<owner>/ for GitHub-only releases) — listing order
    ("alibaba", "阿里（Qwen · Wan · 通义）", ("Qwen/", "Wan-AI/", "Tongyi-MAI/", "FunAudioLLM/")), ("deepseek", "DeepSeek（深度求索）", ("deepseek-ai/",)),
    ("kimi", "Kimi（月之暗面）", ("moonshotai/",)), ("zhipu", "智谱（GLM · CogVideoX）", ("zai-org/", "THUDM/")),
    ("minimax", "MiniMax（稀宇）", ("MiniMaxAI/",)),
    ("meta", "Meta（Llama · ESM · Muse）", ("unsloth/", "meta-llama/", "facebook/", "meta-models/", "amd/Muse-Spark")),
    ("openai", "OpenAI（gpt-oss · Whisper）", ("openai/",)), ("mistral", "Mistral AI", ("mistralai/",)),
    ("microsoft", "微软（Phi）", ("microsoft/",)),
    ("bytedance", "字节跳动（Seed · Protenix）", ("ByteDance-Seed/", "gh:bytedance/")),
    ("yi", "零一万物（Yi）", ("01-ai/",)), ("internlm", "上海 AI 实验室（InternLM）", ("internlm/",)),
    # vendors with catalog-only (not yet on core v2) releases
    ("tencent", "腾讯（HunyuanVideo · Hy）", ("hunyuanvideo-community/", "tencent/")), ("lightricks", "Lightricks（LTX-Video）", ("Lightricks/",)),
    ("genmo", "Genmo（Mochi）", ("genmo/",)), ("hpcai", "HPC-AI Tech（Open-Sora）", ("hpcai-tech/",)),
    ("deepmind", "Google DeepMind（AlphaFold）", ("gh:google-deepmind/",)), ("boltz", "Boltz（MIT）", ("gh:jwohlwend/", "boltz-community/")),
    ("openfold", "OpenFold（aqlaboratory）", ("gh:aqlaboratory/",)),
    ("hfbio", "Hugging Face Bio（Carbon 基因组）", ("HuggingFaceBio/",)),
    ("arc", "Arc Institute（Evo 2 基因组）", ("arcinstitute/",)),
    ("liquid", "Liquid AI（LFM）", ("LiquidAI/",)), ("ant", "蚂蚁集团（Ling）", ("inclusionAI/",)),
    ("bfl", "Black Forest Labs（FLUX）", ("black-forest-labs/",)),
    ("stepfun", "阶跃星辰（Step）", ("stepfun-ai/",)), ("xiaomi", "小米（MiMo）", ("XiaomiMiMo/",)),
    ("openbmb", "面壁智能 / OpenBMB（MiniCPM）", ("openbmb/",)), ("robbyant", "Robbyant（LingBot-Video）", ("robbyant/",)),
    ("startlux", "StartLux Labs（决策模型）", ("startlux-models/",)), ("rwkv", "RWKV（BlinkDL）", ("RWKV/",)),
    ("nvidia", "NVIDIA（Nemotron）", ("nvidia/",)),
    ("reflection", "Reflection AI（Beam）", ("reflection/",)),
    ("openmoss", "复旦 OpenMOSS（MOSS）", ("OpenMOSS-Team/",)),
    ("kyutai", "Kyutai（Moshi）", ("kyutai/",)),     # 0.274
    ("bilibili", "哔哩哔哩 Index（Index-Echo）", ("IndexTeam/",)),     # 0.279
    ("firered", "小红书 FireRedTeam（FireRedTTS）", ("FireRedTeam/",)),     # 0.275
    ("fishaudio", "Fish Audio（Drama）", ("fishaudio/",)),     # 0.276
    ("ornith", "Ornith AI（Ornith）", ("ornith-ai/",)),     # 0.287
    ("perplexity", "Perplexity（决策模型）", ("perplexity-ai/",)),     # 0.291
    ("inception", "Inception / MBZUAI IFM（Jais）", ("inception42/", "IFM/")),     # 0.291
    ("jetbrains", "JetBrains（Mellum）", ("JetBrains/",)),     # 0.291
]
FAMILIES = [    # provider, family, id prefixes — newest family first; within a family: size ↓, base → FP8 → AWQ
    ("alibaba", "Fun-ASR-Nano", ("fun-asr-nano",)), ("alibaba", "Qwen3-ASR", ("qwen3-asr",)), ("alibaba", "Qwen2.5-Omni", ("qwen2.5-omni",)),
    ("alibaba", "Fun-CosyVoice3", ("fun-cosyvoice3",)), ("alibaba", "CosyVoice2", ("cosyvoice2",)),
    ("alibaba", "Qwen2-Audio", ("qwen2-audio",)),
    ("alibaba", "Qwen3.8", ("qwen3.8",)), ("alibaba", "Qwen3.5", ("qwen3.5",)), ("alibaba", "Qwen3-Next", ("qwen3-next",)),
    ("alibaba", "Qwen3 MoE", ("qwen3-235b", "qwen3-30b")), ("alibaba", "Qwen3 稠密", ("qwen3-",)),
    ("alibaba", "Qwen2.5", ("qwen2.5",)), ("alibaba", "Wan2.2", ("wan2.2",)), ("alibaba", "Wan2.1", ("wan2.1",)),
    ("deepseek", "DeepSeek-V4", ("deepseek-v4",)), ("deepseek", "DeepSeek-V3", ("deepseek-v3", "deepseek-r1")),
    ("kimi", "Kimi-Audio", ("kimi-audio",)), ("kimi", "Kimi-K3", ("kimi-k3",)), ("kimi", "Kimi-K2", ("kimi-k2",)),
    ("zhipu", "GLM-5", ("glm-5",)), ("zhipu", "GLM-4.5 / 4.6", ("glm-4",)), ("zhipu", "CogVideoX", ("cogvideox",)),
    ("minimax", "MiniMax-M1 / Text-01", ("minimax-m1", "minimax-text")), ("minimax", "MiniMax-H", ("minimax-h",)),
    ("meta", "Llama 3", ("llama-3",)), ("meta", "ESM-2", ("esm2",)), ("meta", "ESMFold", ("esmfold",)),
    ("openai", "gpt-oss", ("gpt-oss",)), ("openai", "Whisper", ("whisper",)),
    ("mistral", "Mistral Large", ("mistral-large",)), ("mistral", "Mixtral", ("mixtral",)), ("mistral", "Magistral", ("magistral",)),
    ("microsoft", "Phi-4", ("phi-4",)), ("bytedance", "Seed-OSS", ("seed-oss",)), ("bytedance", "Protenix", ("protenix",)),
    ("tencent", "HunyuanVideo", ("hunyuanvideo",)), ("lightricks", "LTX-Video", ("ltx-video",)),
    ("genmo", "Mochi 1", ("mochi",)), ("hpcai", "Open-Sora", ("opensora",)),
    ("deepmind", "AlphaFold", ("alphafold",)), ("boltz", "Boltz", ("boltz",)), ("openfold", "OpenFold", ("openfold",)),
    ("hfbio", "Carbon-A（基因注释编码器）", ("carbon-a",)), ("hfbio", "Carbon（DNA 因果语言模型）", ("carbon-",)),
    ("arc", "Evo 2（StripedHyena 2 DNA 语言模型）", ("evo2-",)),
    ("meta", "Muse Spark Open", ("muse-spark",)),
    ("liquid", "LFM2.5", ("lfm2.5",)), ("ant", "Ling-3.1", ("ling-3.1",)), ("ant", "Ling-3.0-VL", ("ling-3.0-flash-vl",)),
    ("ant", "Ling-3.0", ("ling-3.0",)),   # 0.311: Ling-3.0-tiny / -flash text + the tiny UBTree pair
    ("ant", "Ming-3.0-ASR", ("ming-3.0-asr",)), ("ant", "Ming-3.0-TTS", ("ming-3.0-tts",)),
    ("alibaba", "Qwen-Image", ("qwen-image",)), ("bfl", "FLUX.2", ("flux.2",)),
    ("stepfun", "Step-Audio-2", ("step-audio-2",)), ("stepfun", "Step-5", ("step-5",)), ("xiaomi", "MiMo-V2.6", ("mimo-v2.6",)), ("internlm", "Intern-S2", ("intern-s2",)),
    ("openbmb", "MiniCPM-V 4.7", ("minicpm-v-4.7",)), ("robbyant", "LingBot-Video", ("lingbot-video",)),
    ("startlux", "StartLux-Decision v1.1", ("startlux-decision-v1.1",)), ("startlux", "StartLux-Decision v1", ("startlux-decision-v1-",)),
    ("microsoft", "Microsoft-Decision", ("microsoft-decision",)), ("rwkv", "RWKV-7 G1k", ("rwkv7-",)),
    ("tencent", "Hy-Memory（Hy3 MoE）", ("hy-memory",)), ("bfl", "FLUX 3", ("flux-3",)),
    ("microsoft", "UserRepGen", ("userrepgen",)),
    ("nvidia", "Nemotron-Labs Dory（循环 LM）", ("dory",)), ("nvidia", "Nemotron-3.5（Super VL · Bolt · Lightning）", ("nemotron-3.5",)), ("nvidia", "Qwen3.8（NVIDIA 量化）", ("qwen3.8",)), ("alibaba", "MAI-Gym", ("mai-gym",)),
    ("reflection", "Beam-1.0", ("beam-",)),
    ("openmoss", "MOSS-Image", ("moss-image",)),
    ("kyutai", "Moshi", ("moshi",)),     # 0.274
    ("bilibili", "Index-Echo S2ST", ("index-echo",)),     # 0.279
    ("firered", "FireRedTTS3", ("fireredtts3",)),     # 0.275
    ("fishaudio", "Drama 3", ("drama-3",)),     # 0.276
    ("ornith", "Ornith-1.5", ("ornith-",)),     # 0.287
    ("perplexity", "pplx-decider v1.2", ("pplx-decider-v1.2", "pplx-decider-flash-v1.2")),     # 0.291
    ("perplexity", "pplx-decider v1 / v1.1", ("pplx-decider-v1",)),
    ("inception", "Jais 3", ("jais-3-",)), ("jetbrains", "Mellum2.1", ("mellum2.1",)),
]
_PROV_RANK = {k: i for i, (k, _, _) in enumerate(PROVIDERS)}
# Same architecture as a listed entry → resolvable by id and validated (V1), but not listed separately.
MERGED = {"deepseek-v3.1": "deepseek-v3", "deepseek-r1": "deepseek-v3", "kimi-k2.7-code": "kimi-k2.5",
          "glm-4.5": "glm-4.6", "minimax-text-01": "minimax-m1-80k",
          # 0.251: Qwen3.5 fine-tunes, config identical to a listed entry except the MTP head they do not ship
          "userrepgen-4b": "startlux-decision-v1-4b", "flux-3-image-klein-pup": "qwen3.5-9b",
          # 0.253: MAI-Gym-9B (GUI-agent Qwen3.5-9B fine-tune, no MTP head)
          "mai-gym-9b": "qwen3.5-9b",
          # 0.257: MOSS-Image-SFT (fine-tune of MOSS-Image-Base: identical header shapes / dtypes and pipeline)
          "moss-image-sft": "moss-image-base",
          # 0.288: Ornith (ornith-ai) Qwen3.5 post-trains — same config / header shapes as the listed entry (dtypes as
          # released: all-bf16 where the Qwen base keeps a few fp32 norms / A_log); 1.0 ships no MTP head
          "ornith-1.5-397b": "qwen3.5-397b-a17b", "ornith-1.5-35b-a3b": "startlux-decision-v1-35b-a3b",
          "ornith-1.5-9b": "qwen3.5-9b", "ornith-1.0-397b": "qwen3.5-397b-a17b",
          "ornith-1.0-35b": "startlux-decision-v1-35b-a3b", "ornith-1.0-9b": "qwen3.5-9b",
          "ornith-1.0-397b-fp8": "ornith-1.5-397b-fp8", "ornith-1.0-35b-fp8": "ornith-1.5-35b-a3b-fp8",
          # 0.314: OvisOCR3 (ATH-MaaS Qwen3.5-0.8B post-train; no MTP head; all-bf16 where StartLux keeps fp32 norms)
          "ovisocr3": "startlux-decision-v1-0.8b"}
# 0.289: the Nemotron 3.5 bf16 releases (Bolt, Lightning, Super-VL) are listed, not merged — MERGED needs identical
# params and dtypes, and they differ in dtype from the as-served 24GB / NVFP4 releases
# Generic dense GQA models whose sizes are already covered by Qwen3 / Llama 3: no extra design insight.
UNLISTED = {"yi-1.5-34b": "与 Qwen3-32B 同类（稠密 GQA）", "internlm3-8b": "与 Qwen3-8B / Llama-3.1-8B 同类",
            "internlm2-5-20b": "与 Qwen3-32B / Magistral-Small 同类", "qwen2.5-3b": "与 Qwen3-4B 同类",
            "qwen2.5-1.5b": "与 Qwen3-1.7B 同类", "opensora-stdit2": "已由 STDiT3 取代（同尺寸）",
            "llama-2-7b": "仅用于外部校核（0.66 参考硬件）", "llama-2-70b": "仅用于外部校核（0.66 参考硬件）",
            "mistral-7b": "仅用于外部校核（0.66 参考硬件）"}
_VARIANT = {"": 0, "fp8": 1, "awq": 2}


def _video_pre(spec):
    from .vision import video_pre
    return video_pre(spec)


def provider_of(hf_id: str) -> tuple[str, str]:
    for key, label, orgs in PROVIDERS:
        if hf_id.startswith(orgs):
            return key, label
    return "other", hf_id.split("/")[0]


def family_of(provider: str, model_id: str) -> tuple[int, str]:
    for i, (prov, fam, prefixes) in enumerate(FAMILIES):
        if prov == provider and model_id.startswith(prefixes):
            return i, fam
    return len(FAMILIES), ""


def _variant(model_id: str) -> int:
    tail = model_id.rsplit("-", 1)[-1]
    return _VARIANT.get(tail if tail in _VARIANT else "", 0)


def _repo_name(hf_id: str) -> str:
    return hf_id.split("/", 1)[1]


@lru_cache(maxsize=1)
def entries() -> list[dict]:
    """Every fetched LLM / VLM release (resolvable by id; all are validated in V1)."""
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        if e.get("domain") != "llm" or not e.get("hf_id"):
            continue
        out.append({"id": e["id"], "hf_id": e["hf_id"], "aliases": tuple(e.get("aliases") or ())})
    for i, h in EXTRA:
        out.append({"id": i, "hf_id": h, "aliases": (h,)})
    old = {e["id"]: e for e in raw["entries"]}
    for i, h, b, wl in DOMAIN_RELEASES:
        al = tuple(old.get(i, {}).get("aliases") or ()) + (h,)
        out.append({"id": i, "hf_id": h, "aliases": al, "builder": b, "workload": wl})
    for i, h in EXTRA_TAIL:
        out.append({"id": i, "hf_id": h, "aliases": (h,)})
    for i, h, prov, lab, note in PROXIES:
        out.append({"id": i, "hf_id": h, "aliases": (), "proxy_note": note, "proxy_label": lab, "proxy_provider": prov})
    out = [e for e in out if load_release(e["hf_id"]) is not None]
    for e in out:
        e["label"] = e.get("proxy_label") or DOMAIN_LABELS.get(e["id"]) or _repo_name(e["hf_id"])
        e["provider"], e["provider_label"] = provider_of(e["hf_id"])
        if e.get("proxy_provider"):
            e["provider"] = e["proxy_provider"]
            e["provider_label"] = next(l for k, l, _ in PROVIDERS if k == e["provider"])
        e["decision"] = e["id"] in DECISION_IDS or e["id"].startswith("startlux-decision")
        e["family_rank"], e["family"] = family_of(e["provider"], e["id"])
        if e.get("builder"):
            e["domain"] = e["workload"].kind
        else:
            rel = load_release(e["hf_id"])
            cf = rel.get("config") or {}
            # 0.268: an audio-only release (audio_config, no vision_config) is speech recognition, not a VLM
            # 0.270: any release with a modelled audio encoder and no vision tower (Whisper keeps it outside audio_config)
            from .vision import audio_spec
            # 0.271: Qwen2.5-Omni (speech in + speech out) → omni; 0.273: Step-Audio-2-mini with its token2wav too
            e["domain"] = "tts" if cf.get("model_type") in ("lingv3_tts", "cosyvoice2", "cosyvoice3", "fireredtts3", "fish_omni") else "omni" if cf.get("model_type") == "qwen2_5_omni" or (cf.get("model_type") == "step_audio_2" and rel.get("token2wav")) or rel.get("kimi_audio") or cf.get("model_type") in ("moshi", "index_echo_s2st") else \
                ("asr" if (cf.get("audio_config") or audio_spec(cf) is not None) and not cf.get("vision_config")
                 else "vlm") if (rel.get("params_vision") or 0) > 0 else "llm"
        e["merged_into"] = MERGED.get(e["id"])
        e["unlisted"] = UNLISTED.get(e["id"])
        e["listed"] = not e["merged_into"] and not e["unlisted"]
    return out


def _meta(e: dict) -> dict[str, str]:
    """``metadata`` string of series_catalog.json ('k:v | k:v | …') → dict."""
    out = {}
    for part in (e.get("metadata") or "").split(" | "):
        k, _, v = part.partition(":")
        if v:
            out[k.strip()] = v.strip()
    return out


def _org_key(e: dict, meta: dict) -> str:
    """HF id, or ``gh:<owner>/<repo>`` for releases published on GitHub only (AlphaFold, Boltz, …)."""
    if e.get("hf_id"):
        return e["hf_id"]
    src = meta.get("source", "")
    if src.startswith("https://github.com/"):
        return "gh:" + "/".join(src[len("https://github.com/"):].split("/")[:2])
    return ""


@lru_cache(maxsize=1)
def offline_entries() -> list[dict]:
    """Video generation (DiT) and protein entries of series_catalog.json: listed in the catalog under their vendor,
    but not yet on core v2 (not evaluable).  Dims are the historical catalog's (public configs / papers)."""
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        dom = OFFLINE_DOMAINS.get(e.get("domain"))
        if dom is None or e["id"] in UNLISTED or e["id"] in _DOMAIN_IDS:
            continue
        meta, sh = _meta(e), e.get("shape") or {}
        org = _org_key(e, meta)
        prov, plab = provider_of(org)
        rank, fam = family_of(prov, e["id"])
        dims = f"{sh.get('n_layers', '?')} 层 · hidden {sh.get('hidden', '?')}"
        if sh.get("pair_dim"):
            dims += f" · pair {sh['pair_dim']}"
        out.append({"id": e["id"], "hf_id": e.get("hf_id"), "source": meta.get("source", ""),
                    "label": _repo_name(e["hf_id"]) if e.get("hf_id") else e.get("user_label") or e["id"],
                    "domain": dom, "domain_label": DOMAINS[dom], "provider": prov, "provider_label": plab,
                    "family": fam or plab, "family_rank": rank, "status": "暂未接入 v2", "evaluable": False,
                    "reason": OFFLINE_REASONS.get(e["id"], "尚无可核对的发布权重 / 配置接入 core v2"),
                    "arch": ("DiT · " if dom == "gen" else "") + dims, "arch_detail": meta.get("arch", ""),
                    "_size": (sh.get("n_layers") or 0) * (sh.get("hidden") or 0) ** 2})
    out.sort(key=lambda d: (_PROV_RANK.get(d["provider"], 99), d["family_rank"], -d["_size"]))
    for d in out:
        del d["_size"]
    return out


def _index() -> dict[str, dict]:
    idx = {}
    for e in entries():
        idx[e["id"]] = e
        for a in e["aliases"]:
            idx.setdefault(a.lower(), e)
    return idx


@lru_cache(maxsize=None)
def get_model(model_id: str) -> ModelSpec:
    e = _index().get(model_id) or _index().get(model_id.lower())
    if e is None:
        raise KeyError(f"unknown model {model_id!r}")
    if e.get("builder"):
        return from_domain_release(e["id"], e["hf_id"], e["builder"], e["workload"])
    spec = from_release(e["id"], e["hf_id"])
    if e.get("proxy_note"):
        from dataclasses import replace as _rp
        spec = _rp(spec, coverage="proxy", notes=spec.notes + (e["proxy_note"],),
                   coverage_reasons=spec.coverage_reasons + ("架构代理：按所述基座的公开配置推断",))
    return spec


def provenance(spec: ModelSpec) -> str:
    for k, v in MIRRORS.items():
        if spec.hf_id.startswith(k):
            return "mirror"
    return "official"


def dtype_label(spec: ModelSpec) -> str:
    f = dict(spec.formats)
    w = f.get("expert") or f.get("mlp") or f.get("attn")
    parts = []
    if "expert" in f and "attn" in f and f["expert"].fmt != f["attn"].fmt:
        parts.append(f"W {f['attn'].fmt}/专家 {f['expert'].fmt}")
    elif w:
        parts.append(f"W {w.fmt}")
    aa, ae = spec.act_fmt_for("attn"), spec.act_fmt_for("expert")
    # 0.280: per-role activation formats (W4A4 NVFP4 experts, W8A8 MXFP8 / FP8 attention) shown like the weights
    parts.append(f"A {aa}/专家 {ae}" if "expert" in f and ae != aa else f"A {aa}")
    if spec.kv_cache:
        parts.append(f"KV {spec.kv_fmt}")
    return " · ".join(parts)


_COVER_FILTER = ("「架构代理」", "混合注意力：", "DSA lightning", "attention sink")


def labels(spec: ModelSpec) -> dict:
    """Three-axis model label: provenance × coverage (with concrete reasons) × as-released dtype."""
    pc = spec.param_check()
    return {
        "provenance": provenance(spec),
        "coverage": spec.coverage,
        "coverage_reasons": list(spec.coverage_reasons),
        "proxy_badge": spec.coverage == "proxy",
        "dtype": dtype_label(spec),
        "act_fmt": spec.act_fmt,
        "params_B": pc["ours"] / 1e9,
        "active_B": spec.active_params() / 1e9,
        "release_params_B": (pc["release"] or 0) / 1e9,
        "param_err": pc["rel_err"],
        "vision_params_B": spec.vision_params / 1e9,
        "vision": ({"label": spec.vision.label, "patch": spec.vision.patch, "merge": spec.vision.merge,
                    "depth": spec.vision.depth, "hidden": spec.vision.hidden, "params_B": spec.vision.params / 1e9,
                    "video": _video_pre(spec) is not None}
                   if spec.vision else None),
        "notes": [n for n in spec.notes if not n.startswith(_COVER_FILTER)],
        "arch": spec.arch,
        "what_if": spec.what_if,
        "roles": {r: f.fmt for r, f in spec.formats},
        "is_moe": spec.is_moe,
        "is_pair": spec.is_pair,
        "mtp_layers": len(spec.mtp_layers),
        "draft": spec.draft,
        "vocab": spec.vocab,        # 0.223: UI draft-model picker (same tokenizer check)
        # 0.221: published acceptance → the i.i.d. spec_accept it implies (a lever value, not a default)
        "spec_published": ({"tau": spec.spec_published[0], "k": spec.spec_published[1],
                            "accept_iid": round(spec_accept_from_tau(*spec.spec_published), 4)}
                           if spec.spec_published else None),
        # 0.253: published τ at temperature 1 (DSpine) → serving.spec_accept_src = published with spec_sampling = sampled
        "spec_published_t1": spec.spec_published_t1 or None,
        # 0.253 looped LM (Dory): release loop count (0 = not a looped model) → UI workload.loops / loop_kv
        "loop_default": spec.loop_spec[2] if spec.loop_spec else 0,
        "n_layers": spec.n_layers,
        "model_domain": spec.domain,
        "kv_cache": spec.kv_cache,
        "workload": _workload_dict(spec),
        "heads": spec.layers[0].core.n_q if spec.layers else 0,
    }


def _pipeline_dict(model_id: str) -> list[dict] | None:
    from .pipeline import component_data, pipeline_for, stored_bytes
    pl = pipeline_for(model_id)
    if pl is None:
        return None
    out = [{"role": "text_encoder", "label": t.label, "GB": stored_bytes(t.key) / 1e9,
            "dtype": component_data(t.key)["dtype"], "tokens": t.tokens} for t in pl.text]
    out += [{"role": r, "label": v.label, "GB": stored_bytes(v.key) / 1e9, "dtype": component_data(v.key)["dtype"]}
            for r, v in (("vae", pl.vae), ("audio_vae", pl.audio)) if v]
    return out


def _workload_dict(spec: ModelSpec) -> dict | None:
    w = spec.workload
    if w is None:
        return None
    if w.kind == "gen":
        return {"kind": "gen", "image": w.frames == 1 and not w.fps,     # 0.115 image DiT (one latent frame)
                "frames": w.frames, "height": w.height, "width": w.width, "fps": w.fps,
                "steps": w.steps, "cfg": w.cfg, "patch": list(w.patch), "vae": [w.vae_t, w.vae_s, w.vae_s],
                "text_tokens": w.text_tokens, "joint_text": bool(w.prefix_tokens), "source": w.source,
                "vae_frames": w.vae_frames, "audio_per_s": w.audio_per_s, "audio_channels": w.audio_channels,
                "attention": ("factorized" if any(l.core.span in ("spatial", "temporal") for l in spec.layers)
                              else "per_mode" if any(l.core.span == "mode" for l in spec.layers) else "full"),
                "max_refs": w.max_refs,
                "uncond": ({"text_tokens": w.uncond[0], "skip_layers": list(w.uncond[1])} if w.uncond else None),   # 0.260
                "cfg_distilled": w.cfg == 1, "pipeline": _pipeline_dict(spec.id)}
    if w.kind == "genomics":
        return {"kind": "genomics", "seq_len": w.seq_len, "max_seq": w.max_seq, "kmer": w.kmer, "overlap": w.overlap,
                "source": w.source, "structure": False}
    d = {"kind": "protein", "seq_len": w.seq_len, "max_seq": w.max_seq, "source": w.source, "structure": spec.is_pair}
    if spec.is_pair:
        d.update({"msa": w.msa, "xmsa": w.xmsa, "templates": w.templates, "recycles": w.recycles,
                  "diff_steps": w.diff_steps, "samples": w.samples, "atoms_per_res": w.atoms_per_res,
                  "pair_dim": w.pair_dim})
    return d


def list_models() -> list[dict]:
    """Listed (evaluable) models, ordered vendor → family → size ↓ → base / FP8 / AWQ.  LLM and VLM releases of
    one vendor (even of one family) sit together; ``domain`` is only a badge."""
    es = entries()
    by_id = {e["id"]: e for e in es}
    merged: dict[str, list[str]] = {}
    for e in es:
        if e["merged_into"]:
            merged.setdefault(e["merged_into"], []).append(e["id"])
    out = []
    for e in es:
        if not e["listed"]:
            continue
        s = get_model(e["id"])
        same = []
        for mid in merged.get(e["id"], []):
            o = get_model(mid)
            same.append({"id": mid, "label": by_id[mid]["label"],
                         "identical": o.params() == s.params() and dtype_label(o) == dtype_label(s)})
        out.append({"id": e["id"], "hf_id": e["hf_id"], "label": e["label"], "domain": e["domain"],
                    "domain_label": DOMAINS[e["domain"]], "provider": e["provider"],
                    "provider_label": e["provider_label"], "family": e["family"], "same_as": same, "evaluable": True,
                    **labels(s), "_key": (_PROV_RANK.get(e["provider"], 99), e["family_rank"], -s.params(),
                                          _variant(e["id"]))})
    out.sort(key=lambda d: d["_key"])
    for d in out:
        del d["_key"]
    return out


def catalog_listing() -> list[dict]:
    """The full catalog in display order: vendor → (evaluable families, then catalog-only families) → size ↓.
    Evaluable rows are ``list_models()`` items; catalog-only rows (video generation / protein, not yet on core v2)
    are ``offline_entries()`` items with ``evaluable: False``."""
    ms, off = list_models(), offline_entries()
    pos = {m["id"]: i for i, m in enumerate(ms)}
    opos = {o["id"]: i for i, o in enumerate(off)}
    rows = [(_PROV_RANK.get(m["provider"], 99), 0, pos[m["id"]], m) for m in ms]
    rows += [(_PROV_RANK.get(o["provider"], 99), 1, opos[o["id"]], o) for o in off]
    return [r[-1] for r in sorted(rows, key=lambda r: r[:3])]


def unlisted_models() -> list[dict]:
    """Fetched releases not listed separately, with the reason (merged into a same-structure entry, or redundant)."""
    by_id = {e["id"]: e for e in entries()}
    out = []
    for e in entries():
        if e["merged_into"]:
            a, b = get_model(e["id"]), get_model(e["merged_into"])
            d = a.params() / b.params() - 1
            rd = (a.release_params or 0) / (b.release_params or 1) - 1
            same = "同结构、同 dtype" + ("" if abs(rd) < 1e-4 else f"（发布参数差 {rd:+.2%}）")
            out.append({"id": e["id"], "label": e["label"],
                        "reason": f"与 {by_id[e['merged_into']]['label']} {same}，已合并（仍可按 id 评估）"})
    out += [{"id": e["id"], "label": e["label"], "reason": e["unlisted"] + "，尺寸已覆盖，不单列（仍可按 id 评估）"}
            for e in entries() if e["unlisted"]]
    return out
