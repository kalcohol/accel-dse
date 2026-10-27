"""VLM vision encoders (0.62): vision tower + projector / merger of every VLM in the catalog, built from the released
``vision_config`` and the released image-processor settings, run at prefill.

Each released family (checked against the reference modules: transformers ``qwen3_5`` / ``qwen4_exp`` / ``glm5_next``
/ ``kimi_k25`` vision models, the Kimi-K3 remote code ``modeling_kimi_k3.py`` and DeepSeek-V4.1 ``inference/vision.py``;
docs/MODEL.md §20):

  qwen   Qwen3.5 / Qwen3.8: Conv3d patch embed (temporal_patch_size 2 — a still image is repeated to 2 frames, one
         temporal patch), learned 2-D position table (bilinear gather), ``depth`` pre-LN blocks (fused qkv + proj,
         GELU MLP fc1 / fc2), 2×2 patch merger (LN → fc1 4h→4h → GELU → fc2 4h→out).  Resize: Qwen2-VL
         ``smart_resize`` to multiples of patch·merge = 32 px within [shortest_edge, longest_edge] pixels.
         Qwen3-VL / Qwen3-VL-MoE (``qwen3_vl`` / ``qwen3_vl_moe``): the same tower.  ``deepstack_visual_indexes``
         is one extra merger per index (transformers ``Qwen3VLVisionPatchMerger(use_postshuffle_norm=True)``:
         LN over 4h, then the same fc1 / fc2).  The index only picks which block's hidden state feeds that merger;
         every block still runs.  Each merger output is added onto visual-token rows after LLM layers
         ``0 .. len(indexes)-1`` (``Qwen3VLTextModel._deepstack_process``).  A non-empty index list on any other
         Qwen ViT type is still refused.
         0.419 bills those ``nn.LayerNorm`` calls (fp32 reduction) and ``nn.GELU()`` on each merger
         (the erf form, after fc1, on merged rows and width 4h). The block MLP stays
         ``ACT2FN[hidden_act]`` = ``gelu_pytorch_tanh`` and is not billed. RoPE and the full
         residual adds stay in the tail.
  glm    GLM-5.3-Flash: Conv3d patch embed (T 2), ``depth`` RMSNorm blocks (qkv / proj with bias, q / k RMSNorm,
         SwiGLU MLP with bias), post RMSNorm, 2×2 Conv2d downsample h → out (an implicit GEMM K = 4h), merger
         proj out→out → LN → GELU → SwiGLU out→projection_intermediate→out.  Resize: Glm5Next ``smart_resize`` to
         multiples of 28 px within [min_image_tokens, max_image_tokens] tokens (padded canvas).
  kimi   Kimi-K2.5 (MoonViT): Conv2d patch embed, learned 64×64 position grid (bicubic), LN blocks (q / k / v / proj
         with bias, GELU MLP), final LN, 2×2 patch merger (LN → 4h→4h → GELU → 4h→text hidden).
         Kimi-K3: no biases, RMSNorm, attention width ``qkv_hidden_size``, merger v2 (4h→4h → GELU → 4h→text,
         post RMSNorm).  Resize: ``navit_resize_image`` (≤ in_patch_limit patches, ≤ 512 patches per side, padded to
         multiples of patch·merge = 28 px).
  ds     DeepSeek-V4.1-Flash: Linear patch embed (3·14² → h), RMSNorm blocks (fused qkv with bias, SiLU-gated MLP w1 h→2I
         without bias, w2), final RMSNorm, aligner 3×3 unfold (9h → text dim → GELU → text dim).  Resize:
         ``plan_image_grid`` (≥ min_pixels, ≤ vision_max_n_token LLM positions incl. one newline token per row and
         start / end).  The aligner sits outside ``vision`` in the release role table (``aligner.*`` → other); its
         73.41 M params are added here.

  gemma4 Gemma-4 ViT (gemma4_vision): Linear patch embed (3·16² → h, no bias), factorised position table
         (2, position_embedding_size, h), depth pre-norm blocks (separate q / k / v / o, q / k RMSNorm,
         gated MLP). The processor pads to max_soft_tokens · pooling² patches; those rows are what the
         tower GEMMs and the bidirectional attention see. Average-pool (no parameters) then strips the
         pad; the projector Linear(h → text, no bias) runs on the remaining soft tokens. std_bias and
         std_scale are stored buffers when standardize is set. use_clipped_linears adds four bf16
         scalar buffers on each of q, k, v, o, gate, up and down; the clamp does not change the GEMM.
         Video uses the same tower with the video processor's max_soft_tokens (70 on the shipped
         configs, not the image 280): one batch row per frame, padded to 70·pooling² patches.
  gemma4u Gemma-4 12B encoder-free embedder (gemma4_unified_vision): LN → Linear(48²·3 → mm) → LN →
         factorised position table → LN → RMSNorm (no scale) → Linear(mm → text). Both GEMMs see the
         padded max_soft_tokens rows; the LLM prompt keeps only the real soft tokens. Video merges
         patches first, then pads each frame to the video max_soft_tokens (70) before those GEMMs.
         The unified audio
         path is one Linear(audio_embed_dim → text) when audio_config.model_type is gemma4_unified_audio.
         gemma4_audio (E2B / E4B) is a conformer: two stride-2 Conv2d, 12 local-attention blocks,
         then Linear into the text embedder. Each block's content scores use the chunk context;
         every padded query also dots the context//2+1 relative keys. Prompt tokens follow the
         processor formula; the tower GEMMs see the waveform padded to a multiple of 128.
         The conformer also bills the subsampling mask multiplies, bias-free LayerNorms and
         ReLUs, the with-scale RMSNorms, SiLU, GLU on the kept half, and the 0.5 residual.
         The scale-free embed norm, the logit softcap, and the plain residual adds stay in the tail.
  gspeech Granite Speech (granite_speech / granite_speech_plus): 16 kHz mel, hop 160, pairs stacked
         to input_dim 160. A conformer (two half-residual FFNs, Shaw block attention, GLU depthwise
         conv) then a BLIP-2 Q-Former that does not own text embeddings. Query and key projections
         of the conformer see the context-padded frames; the output projection, FFNs, conv, and the
         mid CTC linears see the unpadded frames. The Q-Former pads to whole windows and emits
         window//downsample queries per window. Speech Plus concatenates selected encoder states
         on the feature axis, which widens only the cross-attention K/V. Weights are bf16 as stored.
         Each conformer block also bills its LayerNorms, SiLU, GLU, BatchNorm and the 0.5 residual.
         The Q-Former bills its erf-GELU and the LayerNorms the query path runs. Eval dropout is not an op.
  pixtral Pixtral ViT (Mistral3 vision_config, model_type pixtral): Conv2d patch embed (kernel = stride =
         patch, no bias), RMSNorm, depth pre-norm blocks (separate q / k / v / o, all width hidden, SwiGLU).
         The projector RMSNorms the patch sequence, unfolds spatial_merge_size**2, then Linear(4h -> h),
         Linear(h -> text), GELU, Linear(text -> text). vision_feature_layer as an int selects one hidden
         state (the tower still runs every block); a list concatenates those states on the first projector
         GEMM. Resize is PixtralProcessor: the image processor is called with patch_size * spatial_merge_size,
         longest edge is vision_config.image_size, and each merged row adds one break token (the last is
         the end token). Attention is one bidirectional block per image.
         0.418 bills the weight RMSNorms (ln_pre, per-layer attention and FFN norms, projector norm),
         ACT2FN["silu"] on the gate, the plain product with up, and ACT2FN["gelu"] (erf) after linear_1.
         RoPE, the full residual adds, and the block attention mask stay in the tail.
  ernievl ERNIE-4.5-VL DFNRope (ernie4_5_moe_vl): Linear patch embed (3·14² → 1280, no bias),
         32 pre-LN blocks (fused qkv and proj with bias, quick_gelu MLP fc1 / fc2 with bias, not
         gated). attn_sep is stored and is not read by VisionAttention, so there is no second
         attention stage. The tower is fp16 and the resampler is bf16; both are 2 bytes per
         parameter, so the GEMM label stays bf16. Image post-GEMMs see the spatially merged rows:
         one frame is duplicated and concatenated before the temporal linear, so the row count
         stays put. Video (Ernie4_5_VLProcessor) resizes each frame with the video pixel limits,
         copies the last frame when the count is odd, and pairs frames: the tower attends inside
         one frame, spatial GEMMs see every frame, and the temporal GEMMs plus the projector see
         half the rows. Resize is the processor smart_resize (factor 28). Image limits are min
         3136 px and max 6177·28². Video limits are min 299·28² and max 1196·28². LLM positions
         are the rows that enter the language model plus the start and end markers.

Counted per image (all images of a request share the size): every GEMM above at M = patches (tower) or merged rows
(merger / aligner); bidirectional attention within the image (QKᵀ and PV, ``heads`` × patches²); norm / activation /
residual element-ops 3 per GEMM output 「假设」, softmax 5 per score.  LongCat-ViT also bills the RMSNorm
modules and LayerScales ``LongCatVisionEncoderLayer`` runs, the projector LayerNorm, the SwiGLU SiLU and the
product it multiplies, the two projector erf-GELUs, rope2d cos/sin, the per-forward ``torch.outer`` that
builds ``inv_freq`` angles, and the cos/sin ``repeat`` that widens those angles to ``head_dim``. The FSMN
encoder bills its ReLU6 calls, the int32 mask multiply, and the projector GELU. OmniASR bills
``ACT2FN["gelu"]`` (``nn.functional.gelu``, the erf form) on every feature-extractor conv output, on the
positional conv after the even-kernel pad slice, and on each FFN intermediate. It also bills the
``nn.LayerNorm`` calls the same forward runs: one on every conv output before that GELU, one on the
conv channels before the projection, one before attention and one before the FFN in each block, and
one after the blocks. Those norms reduce in fp32. The positional conv has no LayerNorm. Eval dropout
is not an op.
d1 FastConformer (``audio.py`` ``ConformerLayer``) bills each block's LayerNorms, ``F.silu``,
``F.glu`` on the kept half, BatchNorm and ``x + ff * 0.5``. The adapter and the residual bill
``F.gelu`` (the default erf form). Those activations stay fp32. Subsampling bills three
``nn.ReLU`` calls (one clip, not ReLU6) and the ``_time_mask`` multiplies. The depthwise
``masked_fill``, the pad and the plain residual adds stay in the tail. The sinusoid table
is not an op.
Fun-ASR-Nano (kind ``sanm``) bills each block's input and post LayerNorm, ``ACT2FN["relu"]``
(``nn.ReLU``, one clip) on the MLP intermediate, and the FSMN mask multiply. The output
LayerNorm runs only on the two published layers. The projector ReLU and the adaptor's
LayerNorms and ReLU use the same counts. The adaptor has no FSMN. Eval dropout is not an
op. The ``sqrt(d)`` scale and the sinusoid add are not ops.
The shared Whisper encoder (kind ``whisper``) bills ``nn.functional.gelu`` / ``ACT2FN["gelu"]``
/ ``nn.GELU()`` as the erf form, the pre-attention and pre-MLP LayerNorms, and the encoder
LayerNorm. Step-Audio's adaptor GELU and ReLU, Kimi's adaptor SiLU and LayerNorm, and Ming's
projector GELU use the same recipes. Those activations follow the vision-role dtype and do
not follow the requested GEMM activation format. LayerNorm stays fp32. The decoder's
GELU-as-SiLU approximation is unchanged.
The AuT encoder (kind ``aut``: Qwen3-ASR, Qwen3-Omni, Index-Echo) bills ``F.gelu`` on
each padded Conv2d output, ``ACT2FN["gelu"]`` on each FFN intermediate and on the
projector, and the pre-attention, pre-MLP and encoder LayerNorms. Index-Echo 2B
also bills ``F.gelu`` on the connector ``w1`` output. Those are the erf form.
The connector scalars and the 9B projection have no activation. The sinusoid add
and the plain residual adds are not ops.
Pixtral (family ``pixtral``) bills ``PixtralRMSNorm`` / ``Mistral3RMSNorm`` (fp32
``pow(2).mean``, one ``rsqrt``, times the weight) on the conv output, before
attention, before the SwiGLU, and on the patch sequence before the merger.
``ACT2FN["silu"]`` is ``x * sigmoid(x)`` on the gate only. The product with ``up``
is one multiply on that same width. ``ACT2FN["gelu"]`` after ``linear_1`` is the
erf form, on merged rows and the text width. Those three stay at the vision-role
dtype. The sinusoid RoPE, the full residual adds, and the block mask are not ops.
Qwen3-VL and WeVisOne (``ln_mode`` ``qwen3vl``) bill ``nn.LayerNorm`` before
attention, before the MLP, on the patch sequence before the final merger's view,
and on the merged width of each deepstack merger. ``nn.GELU()`` after each merger
fc1 is the erf form. The MLP's ``gelu_pytorch_tanh`` is not an op. GELU follows
the vision-role dtype, except a tower stored as fp32 whose activations stay the
model activation format. LayerNorm stays fp32.
LFM2.5-VL (family ``lfm2vl``, including d1-3B) bills ``nn.LayerNorm`` before
attention, before the MLP, and after the blocks, all on the padded patch rows.
``ACT2FN["gelu"]`` sits between the projector linears on the unshuffled valid
rows. The tower MLP stays ``gelu_pytorch_tanh`` and is not billed. A projector
LayerNorm is billed only when ``projector_use_layernorm`` is true.
The shared 3-wide tail stays on every GEMM. Softmax is not given a second SFU.  Image tokens into the LLM = merged positions
(+ DeepSeek's processor-inserted newline / start / end positions); chat-template delimiters around an image are text
tokens of the prompt.  The encoder weights are bf16 as released; activations bf16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .ir import Op

VEC_PER_OUT = 3.0
DEFAULT_PX = (1024, 1024)          # 「假设」 typical image when serving.images > 0 and no resolution is given

# 0.369 Gemma-4 E2B / E4B. processor_config.json is identical on the four repos (2026-10-24):
# sampling_rate 16000, hop_length 160, frame_length 320, feature_size 128, audio_seq_length 750.
# Gemma4AudioFeatureExtractor.__call__ pads to a multiple of 128 and truncates at 480000 samples.
_G4_AUDIO_RATE = 16000
_G4_AUDIO_HOP = 160
_G4_AUDIO_FRAME = 320
_G4_AUDIO_MELS = 128
_G4_AUDIO_CAP = 750
_G4_AUDIO_MAX_SAMPLES = 480_000
_G4_AUDIO_PAD = 128

# 0.288: Ornith repos besides 1.5-122B (image + video preprocessor configs identical to Qwen3.5's, each checked)
ORNITH_REPOS = ("1.5-122B-FP8", "1.5-397B-FP8", "1.5-35B-A3B-FP8", "1.5-397B", "1.5-35B-A3B", "1.5-9B", "1.0-397B",
                "1.0-397B-FP8", "1.0-35B", "1.0-35B-FP8", "1.0-9B")

# 0.366 Qwen3-VL Instruct + FP8.  preprocessor_config.json / video_preprocessor_config.json checked on each repo:
# Qwen2VLImageProcessorFast shortest 65536 / longest 16777216; Qwen3VLVideoProcessor shortest 4096 / longest 25165824;
# patch 16, temporal 2, merge 2.  Same smart_resize as the Qwen3.5 path.
QWEN3_VL_REPOS = (
    "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct", "Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct-FP8", "Qwen/Qwen3-VL-4B-Instruct-FP8", "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "Qwen/Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
)
# 0.388 WeVisOne-VL (tencent, 2026-10-22). processor_config.json on both repos: image processor is
# Qwen2VLImageProcessor (shortest 65536 / longest 16777216), video processor is Qwen3VLVideoProcessor
# (shortest 4096 / longest 25165824, fps 2, min_frames 4, max_frames 768). 2B sets max_video_tokens 768;
# 4B omits it and Qwen3VLVideoProcessor.max_video_tokens defaults to 768. Patch 16, merge 2, temporal 2.
WEVIS_REPOS = (
    "tencent/WeVisOne-VL-2B",
    "tencent/WeVisOne-VL-4B",
)

# released image-processor settings (preprocessor_config.json / processor_config.json / inference/config.json)
_PREPROC = {
    # 0.278 Nemotron-3.5-Super-VL (preprocessor_config.json; NemotronH_Omni_Reasoning_V3ImageProcessor)
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": {"min_patches": 1024, "max_patches": 13312, "max_len": 16384},
    # 0.289 Bolt 74B-A9B (24GB / BF16) and Super-VL BF16: identical preprocessor_config.json
    **{f"nvidia/NVIDIA-Nemotron-3.5-{r}": {"min_patches": 1024, "max_patches": 13312, "max_len": 16384}
       for r in ("Bolt-74B-A9B-24GB", "Bolt-74B-A9B-BF16", "Super-VL-120B-A12B-BF16")},
    "Qwen/Qwen3.8-Flash-Next": {"min_px": 65536, "max_px": 16777216},
    "nvidia/Qwen3.8-Flash-Next-FP4-Local": {"min_px": 65536, "max_px": 16777216},   # 0.280: same preprocessor_config
    "Qwen/Qwen3.8-27B": {"min_px": 65536, "max_px": 16777216},
    "Qwen/Qwen3.8-27B-FP8": {"min_px": 65536, "max_px": 16777216},   # 0.363: preprocessor_config.json identical
    "Qwen/Qwen3.5-397B-A17B": {"min_px": 65536, "max_px": 16777216},
    "ornith-ai/Ornith-1.5-122B": {"min_px": 65536, "max_px": 16777216},   # 0.287: preprocessor_config.json (Qwen2VLImageProcessorFast)
    # 0.288: the other Ornith repos — each preprocessor_config.json checked (same size / patch 16 / merge 2)
    **{f"ornith-ai/Ornith-{v}": {"min_px": 65536, "max_px": 16777216} for v in ORNITH_REPOS},
    # 0.291 pplx-decider: processor_config.json image_processor size (Qwen2VLImageProcessor, patch 16, merge 2), each
    # checked; v1.2 / Flash = decision_config max_pixels 1638400 (card: 1.6 MP); v1.1 4194304; v1 262144 (512 × 512)
    **{f"perplexity-ai/pplx-decider-{v}": {"min_px": 65536, "max_px": x} for v, x in (
        ("v1.2", 1638400), ("flash-v1.2", 1638400), ("v1.1-27b", 4194304), ("v1-27b", 262144))},
    # 0.355 MiniMax M3.1-Flash: patch_merge spatial 2 / temporal 2, image 672, patch 14 → 1296 base patches (48×27), merged to 576
    "MiniMaxAI/MiniMax-M3.1-Flash": {"min_px": 65536, "max_px": 16777216},
    # 0.356 MiniMax M3 (non-Flash): same vision config as M3.1-Flash (image 2016, patch 14, merged to 576)
    "MiniMaxAI/MiniMax-M3": {"min_px": 65536, "max_px": 16777216},
    "zai-org/GLM-5.3-Flash": {"min_tok": 16, "max_tok": 8000},
    "moonshotai/Kimi-K2.5": {"in_patch_limit": 16384, "side": 512},
    "moonshotai/Kimi-K2.7-Code": {"in_patch_limit": 16384, "side": 512},   # same vision_config / image processor as K2.5
    "moonshotai/Kimi-K3": {"in_patch_limit": 65536, "side": 512},
    "deepseek-ai/DeepSeek-V4.1-Flash": {},
    # 0.144
    "internlm/Intern-S2-397B": {"min_px": 65536, "max_px": 16777216},          # Qwen2VLImageProcessorFast size
    "XiaomiMiMo/MiMo-V2.6-Flash-RL": {"min_px": 8192, "max_px": 8388608},      # processor_config image_min/max_pixels
    # Step5VisionConfig image_size (overview); 0.255: + Step3p7ImageProcessor crops (patch_size 504, max_image_size 3024)
    "stepfun-ai/Step-5-Preview": {"side": 728, "crop": 504, "max_img": 3024},
    "stepfun-ai/Step-5-Preview-NVFP4": {"side": 728, "crop": 504, "max_img": 3024},   # 0.251: config identical but quantization
    "openbmb/MiniCPM-V-4.7-35B-A3B": {"side": 448, "max_slice": 9},           # MiniCPMV4_6ImageProcessor defaults
    # 0.145
    "XiaomiMiMo/MiMo-V2.6-Pro-RL": {"min_px": 8192, "max_px": 8388608},        # processor_config image_min/max_pixels
    "inclusionAI/Ling-3.0-flash-VL": {"min_px": 4096, "max_px": 4194304},      # preprocessor_config min/max_pixels
    # 0.174 Muse Spark Open: processor_config image_processor max_image_tokens 4096 (merged), patch 14, merge 2.
    # 0.255: resize = transformers muse_glimmer/image_processing_muse_glimmer.py smart_resize (auto map muse_spark →
    # MuseGlimmerImageProcessor; amd/…-Quark-MXFP4 processor_config.json image_processor_type MuseGlimmerImageProcessor)
    **{k: {"muse_tok": 4096} for k in (
        "meta-models/Muse-Spark-Open-816B-A42B", "amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8",
        "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4")},
    # 0.191 Qwen3.5-9B and the StartLux-Decision fine-tunes: preprocessor_config.json Qwen2VLImageProcessorFast size
    # shortest 65536 / longest 16777216 px, patch 16, merge 2 (each repo checked)
    # 0.251 black-forest-labs/flux-3-image-klein-pup: same preprocessor_config.json (prompt upsampler, Qwen3.5-9B fine-tune);
    # microsoft/UserRepGen-4B ships the Qwen3.5-4B vision tower but no preprocessor_config.json → base-model processor 「假设」
    **{k: {"min_px": 65536, "max_px": 16777216} for k in ("Qwen/Qwen3.5-9B", "black-forest-labs/flux-3-image-klein-pup",
                                                         "microsoft/UserRepGen-4B", "Tongyi-MAI/MAI-Gym-9B",
                                                         "ATH-MaaS/OvisOCR3") + tuple(   # 0.314: checked identical
        f"startlux-models/StartLux-Decision-{v}" for v in (
            "v1-27B", "v1-9B", "v1-4B", "v1-2B", "v1-0.8B", "v1-35B-A3B", "v1.1-27B-Instant", "v1.1-27B-AutoThink",
            "v1.1-9B-Instant", "v1.1-9B-AutoThink", "v1.1-4B-Instant", "v1.1-4B-AutoThink", "v1.1-2B-Instant",
            "v1.1-0.8B-Instant")) + (
        # 0.390 Intern-Decision: preprocessor_config.json matches StartLux-Decision-v1-4B
        # (Qwen2VLImageProcessorFast, shortest 65536, longest 16777216, patch 16, merge 2).
        "internlm/Intern-Decision-4B", "internlm/Intern-Decision-2B", "internlm/Intern-Decision-0.8B",
    )},
    # 0.366 Qwen3-VL: same image processor bounds as Qwen3.5 (checked per repo, see QWEN3_VL_REPOS)
    **{k: {"min_px": 65536, "max_px": 16777216} for k in QWEN3_VL_REPOS},
    # 0.388 WeVisOne: same pixel bounds (processor_config.json, both sizes)
    **{k: {"min_px": 65536, "max_px": 16777216} for k in WEVIS_REPOS},
}

# 0.165 — video input.  「来源」 each repo's video_preprocessor_config.json (Qwen3VLVideoProcessor): size shortest_edge
# 4096 / longest_edge 25165824 (a pixel budget over t_bar·h·w of the whole clip), patch 16, temporal_patch 2, merge 2.
_VIDEO = {k: {"min_px": 4096, "max_px": 25165824} for k in (
    "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-Flash-Next", "internlm/Intern-S2-397B",
    # 0.390 Intern-Decision: video_preprocessor_config.json is Qwen3VLVideoProcessor
    # (shortest 4096, longest 25165824, patch 16, temporal 2, merge 2). Each size checked.
    "internlm/Intern-Decision-4B", "internlm/Intern-Decision-2B", "internlm/Intern-Decision-0.8B",
    "nvidia/Qwen3.8-Flash-Next-FP4-Local",
    "ornith-ai/Ornith-1.5-122B") + tuple(f"ornith-ai/Ornith-{v}" for v in ORNITH_REPOS)
    + QWEN3_VL_REPOS   # 0.287 Ornith; 0.366 Qwen3-VL: video_preprocessor_config.json (Qwen3VLVideoProcessor, same size / patch / merge)
    + WEVIS_REPOS}     # 0.388 WeVisOne: processor_config.json video_processor (same size / patch / merge)
# 0.166 — Kimi K2.5 / K2.7-Code video chunks.  「来源」 preprocessor_config.json media_proc_cfg: temporal_merge_kernel_size 4,
# in_patch_limit_each_frame 4096, in_patch_limit_video null (K2.5) / 655360 (K2.7-Code), patch_limit_on_one_side 512;
# config.json vision_config video_attn_type spatial_temporal, merge_type sd2_tpool (modeling_kimi_k25 identical vision code).
_VIDEO.update({k: {"chunk": 4, "frame_lim": 4096, "total": t} for k, t in (
    ("moonshotai/Kimi-K2.5", None), ("moonshotai/Kimi-K2.7-Code", 655360))})
# 0.255 Muse Spark Open video.  「来源」 processor_config.json video_processor (MuseGlimmerVideoProcessor):
# max_video_frame_tokens 256, num_frames 600, fps 2, temporal_patch 2; per-frame smart_resize to ≤ 256 merged tokens
_VIDEO.update({k: {"muse_frame_tok": 256, "num_frames": 600} for k in (
    "meta-models/Muse-Spark-Open-816B-A42B", "amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8",
    "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4", "meta-models/Muse-Spark-Open-816B-A42B-NVFP4")})
# 0.279 Nemotron-3.5-Super-VL video.  「来源」 preprocessor_config.json video_target_num_patches 1024 (aspect kept);
# config.json video_temporal_patch_size 2, video_pruning_rate 0.7 (EVS, modeling_nemotron_h_omni.compute_retention_mask)
_VIDEO["nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"] = {"radio_frame": 1024, "tp": 2, "evs": 0.7}
for _r in ("Bolt-74B-A9B-24GB", "Bolt-74B-A9B-BF16", "Super-VL-120B-A12B-BF16"):    # 0.289 same video fields
    _VIDEO[f"nvidia/NVIDIA-Nemotron-3.5-{_r}"] = {"radio_frame": 1024, "tp": 2, "evs": 0.7}
# 0.382 Gemma-4 video. processor_config.json video_processor on each google/gemma-4-* repo (2026-10-24):
# Gemma4VideoProcessor (ViT) or Gemma4UnifiedVideoProcessor (12B). Same numbers: do_sample_frames, num_frames 32,
# max_soft_tokens 70, patch 16, pooling 3. Image max_soft_tokens stays 280. The sampler emits exactly num_frames
# and rejects a shorter clip (video_utils.default_sample_indices_fn).
_GEMMA4_VIDEO = {"gemma4_soft": 70, "num_frames": 32, "patch": 16, "pool": 3}
for _g4 in (
    "google/gemma-4-31B-it", "google/gemma-4-31B",
    "google/gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B",
    "google/gemma-4-12B-it", "google/gemma-4-12B",
    "google/gemma-4-E4B-it", "google/gemma-4-E4B",
    "google/gemma-4-E2B-it", "google/gemma-4-E2B",
    # 0.389 Nemotron-3.5-Bolt-E4B-BF16: processor_config.json matches google/gemma-4-E4B
    # (Gemma4VideoProcessor, max_soft_tokens 70, num_frames 32, patch 16, pool 3).
    "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16",
):
    _VIDEO[_g4] = dict(_GEMMA4_VIDEO)
# 0.385 ERNIE-4.5-VL video. processing_ernie4_5_vl.py Ernie4_5_VLProcessor (no processor_config.json):
# video_min_pixels 299*28*28, video_max_pixels 1196*28*28, temporal_conv_size 2, spatial_conv_size 2.
# preprocessor_config max_pixels is the image processor's own field; _add_video passes these limits into
# get_smarted_resize. An odd sampled count copies the last frame so the resampler's pairs have equal rows.
# fps / min_frames / max_frames only choose samples from a file. Timestamps are drawn into the pixels.
_ERNIE_VL_VIDEO = {"ernie_temporal": 2, "min_px": 299 * 28 * 28, "max_px": 1196 * 28 * 28}
for _ev in (
    "baidu/ERNIE-4.5-VL-28B-A3B-PT", "baidu/ERNIE-4.5-VL-424B-A47B-PT",
    # 0.396 Thinking uses the same Ernie4_5_VLProcessor limits (temporal 2, video pixel caps).
    "baidu/ERNIE-4.5-VL-28B-A3B-Thinking",
    # 0.397: non-PT repos ship processing_ernie4_5_vl.py with the same defaults
    # (video_min_pixels 299*28*28, video_max_pixels 1196*28*28, temporal_conv_size 2).
    "baidu/ERNIE-4.5-VL-28B-A3B", "baidu/ERNIE-4.5-VL-28B-A3B-Base",
    "baidu/ERNIE-4.5-VL-424B-A47B",
):
    _VIDEO[_ev] = dict(_ERNIE_VL_VIDEO)
# 「来源」 transformers qwen3_vl/video_processing_qwen3_vl.py: fps 2, min_frames 4, max_frames 768, max_video_tokens 768
VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, VIDEO_MAX_TOK = 2, 4, 768, 768


@dataclass(frozen=True)
class VisionSpec:
    family: str                 # qwen | glm | kimi | ds
    label: str
    patch: int
    tpatch: int                 # temporal patch (Conv3d over a repeated still image); 1 = 2-D
    hidden: int
    inter: int
    depth: int
    heads: int
    qkv: int                    # attention width (heads · head_dim)
    gated: bool                 # SwiGLU / SiLU-gated MLP (3 weight matrices' worth) vs GELU fc1 / fc2
    merge: int                  # spatial merge side (2; DeepSeek aligner 3)
    post: tuple[tuple[int, int], ...]   # (K, N) GEMMs per merged position (downsample conv, merger / aligner)
    params: int                 # vision tower + projector params (bf16)
    pre: tuple                  # resize parameters ((key, value), …) — hashable
    bias_qkv: bool = True
    # 0.144 (defaults keep every earlier encoder unchanged)
    kv_heads: int = 0           # GQA tower (MiMo): qkv GEMM N = (heads + 2·kv_heads)·head_dim
    insert: int = 0             # in-tower 2×2 merger after this many blocks (MiniCPM-V-4.7 insert_layer_id): later
                                # blocks run on N/4 rows; ``mid`` GEMMs run there
    mid: tuple[tuple[int, int, int], ...] = ()   # (K, N, row divisor vs patches) GEMMs at the insert point
    extra: int = 0              # stored but not run for images (MiMo audio encoder + speech embeddings)
    approx: str = ""            # coverage reason when the encoder / resize is approximated
    # 0.145 MiMo windowed ViT layers (modeling_mimo_v2 MiMoVisionAttention._build_window_mask): every block not in
    # ``full_idx`` masks |i − j| > ``win`` over the per-image patch sequence (row order for vit_window_attn_types 0,
    # merge-unit column order for 1: same key count); ``sink`` = learned per-head bias on key column 0 of those blocks;
    # ``perm`` = row ↔ column re-index gathers of the hidden states per pass (apply_index)
    win: int = 0
    full_idx: tuple[int, ...] = ()
    sink: bool = False
    perm: int = 0
    # 0.175 Muse Spark NaViT (vLLM PR #61879 vision.py _get_sparse_attention_perm): blocks not in ``full_idx`` attend only
    # within tile × tile patch tiles of the per-image patch grid (tile = pos_emb_height = 32; edge tiles are partial)
    tile: int = 0
    audio: "AudioSpec | None" = None    # 0.175 Muse Spark audio encoder (costed when serving.audio_s > 0)
    # 0.285 quantized vision tower (release ``vision_fmt``; default () = bf16 everywhere, every earlier encoder
    # unchanged): ((kind, w_fmt, w_bits, a_fmt), …) per tower linear kind attn (qkv / o), mlp, merger (post GEMMs),
    # patch (patch embed).  ``w_Bpp`` = stored bytes per tower parameter (release vision bytes / params; 2.0 = bf16).
    wfmt: tuple = ()
    w_Bpp: float = 2.0
    # 0.366 Qwen3-VL: number of deepstack mergers (len(deepstack_visual_indexes)).  0 for every earlier tower.
    # Each one appends the same two post GEMMs as the final merger; the LLM add is a prefill vector op.
    deepstack: int = 0
    # 0.418 Pixtral SiLU, the SwiGLU product, and the projector erf-GELU. The one
    # float safetensors vision role. RMSNorm stays fp32. Empty on every earlier tower.
    # 0.419 Qwen3-VL merger erf-GELU and LFM2.5-VL projector erf-GELU use the same field.
    # LayerNorm stays fp32.
    vec_fmt: str = ""
    # 0.419: "qwen3vl" bills Qwen3VLVisionBlock / Qwen3VLVisionPatchMerger LayerNorms
    # and nn.GELU(). Empty on every other tower, including Qwen3.5.
    ln_mode: str = ""
    # 0.419: deepstack block indexes, in merger order. Empty when deepstack is 0.
    ds_idx: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.qkv // self.heads

    @property
    def rz(self) -> dict:
        return dict(self.pre)


@dataclass(frozen=True)
class AudioSpec:
    """0.175 Muse Spark audio encoder (vLLM PR #61879 audio.py AutoEncoder): raw 24 kHz mono → 4 stages of
    [Patchifier (patchify p, Linear+bias, RMSNorm) + n × Block (depthwise causal conv k, RMSNorm, Linear d→4d GELU →
    d)], RMSNorm, o_proj → out (+ dual RMSNorm).  One LLM token per Π patch_sizes samples (1920 = 80 ms)."""
    rate: int
    patches: tuple[int, ...]
    dims: tuple[int, ...]
    depths: tuple[int, ...]
    kernel: int
    mlp: float
    out: int
    params: int
    # 0.268 Ming-3.0-asr-tiny (kind "whisper"; defaults keep Muse Spark unchanged): 16 kHz log-mel (hop = patches[0]
    # = 160 samples, n_mels bins) → Whisper encoder [conv1 k3 s1 + GELU, conv2 k3 s2 + GELU, sinusoid pos, depths[0] ×
    # pre-LN block (heads, bidirectional MHA, GELU MLP mlp·d), ln_post] → projector [Conv1d (k, s, p) per ``ds`` + GELU,
    # Linear out → out].  The feature extractor never pads to 30 s; mel frames are split into ``chunk``-frame chunks
    # and each chunk runs its own encoder pass (attention within a chunk) — modeling_ming_v3_asr_encoder.py
    # encode_audio_clips / audio_spec.py MingAudioSpec.token_count 「来源」
    kind: str = "muse"
    heads: int = 0
    n_mels: int = 0
    chunk: int = 0
    ds: tuple[tuple[int, int, int], ...] = ()
    # 0.270 speech round.  ``pad``: the feature extractor pads every pass to a 30 s window of ``chunk`` frames
    # (WhisperFeatureExtractor padding="max_length", n_samples 480000) — the encoder always runs 1500 positions;
    # ``trunc``: audio past the first window is dropped (Qwen2-Audio processor, truncation=True) instead of encoded as
    # further windows (Whisper long-form: sequential 30 s windows); ``cross``: encoder-decoder (Whisper) — the encoder
    # output feeds decoder cross-attention, no audio positions in the decoder prompt; ``pool``: AvgPool1d(pool) after
    # the encoder layers (Qwen2-Audio); ``lin``: projector Linears after the encoder (None = the Ming conv projector
    # path, () = none); ``win``: kind "aut" (Qwen3-ASR AuT) attention window in post-CNN positions; ``ch``: AuT conv2d
    # channels (downsample_hidden_size)
    pad: bool = False
    trunc: bool = False
    cross: bool = False
    pool: int = 1
    lin: tuple[tuple[int, int], ...] | None = None
    win: int = 0
    ch: int = 0
    # 0.271 Qwen2.5-Omni: the processor pads to 30 s (padding="max_length", truncation) but the encoder runs only the
    # valid frames in 2·n_window chunks — ``maxf`` = the truncation cap in mel frames (0 = none) with the padded
    # extractor's ⌈samples / hop⌉ frame count; ``stored``: params kept in the release but not run (audio_bos_eos_token)
    maxf: int = 0
    stored: int = 0
    # 0.272 Fun-ASR-Nano (kind "sanm"): Kaldi fbank (25 ms / 10 ms, snip_edges) → LFR stacking ``lfr`` = (m, n): m
    # frames stacked, stride n (80 × 7 = 560 inputs, 16.7 LFR frames / s); ``ad`` = adaptor (layers, d, heads, ffn)
    # after the projector Linears; ``kernel`` = FSMN depthwise kernel; ``maxf`` (sanm) = LFR positions available
    # (positional_embedding rows − 1: positions counted from one).
    # 0.414: ``sanm_final`` is the encoder layers whose output LayerNorm is not Identity.
    lfr: tuple[int, int] = ()
    ad: tuple[int, int, int, int] = ()
    # 0.272 Step-Audio-2-mini: ``mpad`` samples appended before the STFT (log_mel_spectrogram padding=479; frames =
    # (samples + mpad) // hop); ``ds`` + ``lin`` together = AvgPool → Adaptor conv (GELU) → Linear, ReLU, Linear
    mpad: int = 0
    # 0.273 Kimi-Audio: ``stack`` consecutive encoder positions concatenated (1280 × 4 = 5120) into one LLM position
    # (12.5 Hz) before the adaptor Linears; sequential 30 s windows (pad, no truncation)
    stack: int = 0
    # 0.336 OmniASR (kind "w2v"): Wav2Vec2-style tower — ``ds`` = ((kernel, stride, 0), …) of the feature-extractor
    # Conv1d stack (config conv_kernel / conv_stride); ``n_mels`` reused as the last conv channel (projection in);
    # ``ch`` = num_conv_pos_embedding_groups; positional conv kernel = ``kernel`` when kind is w2v
    # 0.337: as-released weight format for the audio tower (OmniASR headers are fp32; other kinds keep bf16 default)
    w_bits: float = 16.0
    w_fmt: str = "bf16"
    # 0.369 Gemma-4 conformer (kind "g4conf"): STFT frame length in samples, and the chunked-attention
    # context_left / context_right from Gemma4AudioConfig. 0 leaves every other kind unchanged.
    frame: int = 0
    ctx_l: int = 0
    ctx_r: int = 0
    # 0.392 Granite Speech (kind "gspeech"). win is the Q-Former window in encoder frames.
    # chunk is the Shaw attention context. gs_q is queries per window. gs_cross is the
    # Q-Former K/V width (encoder hidden, or hidden times one plus each exported state).
    # gs_mid is the CTC output width. gs_pos is max_pos_emb (the Shaw table has 2*gs_pos+1 rows).
    # gs_ff / gs_inner are the conformer FFN and conv widths. gs_freq is cross_attention_frequency.
    # gs_ql / gs_qh / gs_qheads / gs_qff are the Q-Former depth, width, heads, and FFN width.
    gs_q: int = 0
    gs_cross: int = 0
    gs_mid: int = 0
    gs_pos: int = 0
    gs_ff: int = 0
    gs_inner: int = 0
    gs_freq: int = 0
    gs_ql: int = 0
    gs_qh: int = 0
    gs_qheads: int = 0
    gs_qff: int = 0
    # 0.393 Granite Speech 5 CTC (kind "gsctc"): conformer layers whose conv is stride-2.
    # Each listed layer keeps T // 2 frames (residual pool). Empty for every other kind.
    gs_sub: tuple[int, ...] = ()
    # 0.395 OmniASR CTC (kind "w2v"): OmniASRForCTC.ctc_head is Linear(hidden, vocab) with bias,
    # applied to every encoder frame. Prompt tokens stay 0. LLM ports leave this false.
    ctc: bool = False
    # 0.414 Fun-ASR-Nano: encoder layers that construct an output LayerNorm. Empty
    # for every other kind. Identity layers are not in this tuple.
    sanm_final: tuple[int, ...] = ()
    # 0.415 whisper-family activation dtype from the safetensors vision role
    # (fp16 / bf16 / fp32). Empty falls back to w_fmt. GEMM w_fmt in _whisper_ops
    # stays the historical bf16 billing. LayerNorm is fp32 either way.
    # 0.416 AuT uses the same field for GELU. LayerNorm stays fp32.
    vec_fmt: str = ""
    # 0.416 AuT projector-linear indices followed by erf GELU. Index 0 is proj1.
    # Index-Echo 2B also lists the connector w1 index. Empty on every other kind.
    gelu_lin: tuple[int, ...] = ()

    @property
    def samples_per_token(self) -> int:
        return math.prod(self.patches)

    def lfr_frames(self, seconds: float) -> int:
        """0.272 sanm: LFR frames = ⌈fbank frames / n⌉, fbank frames = 1 + (samples − 400) // 160 (kaldi snip_edges)
        — FunAsrNanoFeatureExtractor._apply_lfr 「来源」; capped at ``maxf`` (the sinusoid table)."""
        n = round(seconds * self.rate)
        fb = max(1, 1 + (n - 400) // self.patches[0]) if n >= 400 else 1
        t = -(-fb // self.lfr[1])
        return min(t, self.maxf) if self.maxf else t

    def mel_frames(self, seconds: float) -> int:
        """Mel frames of the clip.  Whisper STFT (center=True, last frame dropped) → ⌊samples / hop⌋; the padded
        extractors count the attention mask rescaled by hop → ⌈samples / hop⌉ (≤ one window when truncating)."""
        n = round(seconds * self.rate)
        if self.kind == "sanm":
            return self.lfr_frames(seconds) * self.lfr[1]      # fbank frames actually encoded (after the cap)
        if self.mpad:
            return max(1, min(self.maxf or 10 ** 9, (n + self.mpad) // self.patches[0]))
        if self.maxf:
            return max(1, min(self.maxf, -(-n // self.patches[0])))
        if self.pad:
            return max(1, -(-n // self.patches[0]))
        return max(1, n // self.patches[0])

    def windows(self, seconds: float) -> int:
        """0.270 padded 30 s windows per clip (Whisper long-form: sequential windows; truncating processors: one)."""
        return 1 if self.trunc else max(1, -(-self.mel_frames(seconds) // self.chunk))

    def chunks(self, seconds: float) -> tuple[int, ...]:
        """0.268 whisper: mel frames per encoder pass (torch.stft center=True drops the last frame → ⌊samples / hop⌋).
        0.270 padded kinds: ``windows`` passes of the full ``chunk`` frames; AuT: ``chunk``-frame CNN chunks."""
        if self.pad:
            return (self.chunk,) * self.windows(seconds)
        mel = self.mel_frames(seconds)
        full, rem = divmod(mel, self.chunk)
        return (self.chunk,) * full + ((rem,) if rem else ())

    def aut_len(self, n: int) -> int:
        """0.270 Qwen3-ASR post-CNN positions of an n-frame chunk: three (k3, s2, p1) convs (_post_cnn_length)."""
        for _ in range(3):
            n = (n - 1) // 2 + 1 if n > 0 else 0
        return n

    def enc_positions(self, seconds: float) -> int:
        """0.270 encoder output positions per clip (Whisper: 1500 per 30 s window — the decoder cross-attention keys)."""
        if self.kind == "gspeech":
            return self._gs_lens(seconds)[0]
        if self.kind == "gsctc":
            return self.tokens(seconds)
        if self.kind == "aut":
            return sum(self.aut_len(n) for n in self.chunks(seconds))
        if self.kind == "sanm":
            return self.lfr_frames(seconds)
        if self.kind == "w2v":
            return self.tokens(seconds)        # 0.336: one encoder / LLM position per post-conv frame (50 Hz)
        if self.kind == "g4conf":
            return self._g4_audio_lens(seconds)[0]
        if self.kind != "whisper":
            return self.tokens(seconds)        # Muse Spark: one output position per token (no chunking)
        return sum(self.chunk_len(n)[0] for n in self.chunks(seconds))

    def prompt_tokens(self, seconds: float) -> int:
        """0.270 audio positions inserted into the LLM prompt (0 for an encoder-decoder: cross-attention instead)."""
        if self.kind in ("gsctc", "d1fc"):
            # d1 prefix embeddings are not text tokens. expand_images must not add them.
            return 0
        if self.kind == "w2v" and self.ctc:
            # OmniASRForCTC has no decoder. Encoder frames are not prompt tokens.
            return 0
        return 0 if self.cross else self.tokens(seconds)

    def chunk_len(self, n: int) -> tuple[int, int]:
        """(encoder positions after conv2, projector tokens) of an n-frame chunk (MingAudioSpec.embedding_length)."""
        t = (n + 1) // 2
        o = t // self.pool if self.ds and self.pool > 1 else t      # 0.272 Step-Audio-2: AvgPool before the adaptor conv
        for k, st, pd in self.ds:
            o = (o + 2 * pd - k) // st + 1
        return t, o

    def _g4_mel(self, samples: int) -> int:
        """Gemma4AudioFeatureExtractor semicausal unfold (frame + 1, hop)."""
        if samples <= 0 or self.frame <= 0:
            return 0
        pad_left = self.frame // 2
        unfold = self.frame + 1
        return (samples + pad_left - unfold) // self.patches[0] + 1

    def _g4_conv_len(self, samples: int) -> int:
        """Two stride-2 Conv2d (kernel 3, pad 1) after the mel frames. Gemma4Processor._compute_audio_num_tokens."""
        mel = self._g4_mel(samples)
        if mel <= 0:
            return 0
        t = mel
        for _ in range(2):
            t = (t + 2 - 3) // 2 + 1
        return t

    def _g4_audio_lens(self, seconds: float) -> tuple[int, int]:
        """(tower time steps, prompt tokens). The tower sees the 128-sample pad; the prompt uses the mask."""
        n = min(_G4_AUDIO_MAX_SAMPLES, max(0, round(seconds * self.rate)))
        prompt = self._g4_conv_len(n)
        if self.maxf:
            prompt = min(prompt, self.maxf)
        padded = ((n + _G4_AUDIO_PAD - 1) // _G4_AUDIO_PAD) * _G4_AUDIO_PAD if n else 0
        return self._g4_conv_len(padded), prompt

    def _gs_lens(self, seconds: float) -> tuple[int, int, int]:
        """(encoder frames, Shaw-padded frames, LLM prompt tokens).

        GraniteSpeechFeatureExtractor: mel length is samples // hop + 1, an odd frame is
        dropped, and pairs are stacked. The projector pads to whole windows and emits
        gs_q queries from every window, including a partial last window.
        """
        if self.patches[0] <= 0 or self.win <= 0 or self.gs_q <= 0 or self.chunk <= 0:
            return 0, 0, 0
        n = max(0, round(seconds * self.rate))
        enc = (n // self.patches[0] + 1) // 2
        if enc <= 0:
            return 0, 0, 0
        nblocks = -(-enc // self.win)
        pad = -(-enc // self.chunk) * self.chunk
        return enc, pad, nblocks * self.gs_q

    def _gsctc_frames(self, seconds: float) -> tuple[int, ...]:
        """Encoder-frame length entering each conformer layer, then the CTC length.

        GraniteSpeech5FeatureExtractor: mel frames are samples // hop, then pairs are
        stacked (ceil). GraniteSpeech5EncoderSubsamplingBlock keeps T // 2. The last
        entry is the CTC head length. Empty when the clip has no mel frame.
        """
        if self.kind != "gsctc" or self.patches[0] <= 0 or self.chunk <= 0 or not self.depths:
            return ()
        n = max(0, round(seconds * self.rate))
        mel = n // self.patches[0]
        if mel <= 0:
            return ()
        t = -(-mel // 2)
        sub = set(self.gs_sub)
        lengths = []
        for i in range(self.depths[0]):
            lengths.append(t)
            if i in sub:
                t = t // 2
        lengths.append(t)
        return tuple(lengths)

    def _lcafsmn_frames(self, seconds: float) -> int:
        """LongCat ``AudioFrameLengthCalculator``: kaldi snip_edges, then ceil-divide by the splice stride.

        ``frame`` is the window in samples, ``patches[0]`` is the hop, ``chunk`` is the splice stride.
        Left and right context pad the feature width. They do not change this length.
        """
        if seconds <= 0 or self.rate <= 0 or self.frame <= 0 or not self.patches or self.patches[0] <= 0 or self.chunk <= 0:
            return 0
        n = round(seconds * self.rate)
        fb = (n - self.frame) // self.patches[0] + 1
        if fb <= 0:
            return 0
        return (fb + self.chunk - 1) // self.chunk

    def tokens(self, seconds: float) -> int:
        if self.kind == "lcafsmn":
            return self._lcafsmn_frames(seconds)
        if self.kind == "d1fc":
            return _d1_audio_lens(seconds)[1]
        if self.kind == "gsctc":
            frames = self._gsctc_frames(seconds)
            return frames[-1] if frames else 0
        if self.kind == "gspeech":
            return self._gs_lens(seconds)[2]
        if self.kind == "g4conf":
            return self._g4_audio_lens(seconds)[1]
        if self.kind == "sanm":
            return self.lfr_frames(seconds)     # processor: num_audio_tokens = input_features_mask.sum (one per LFR frame)
        if self.kind == "aut":
            return self.enc_positions(seconds)
        if self.kind == "whisper" and self.stack:
            # 0.273 Kimi-Audio (kimia_infer whisper_Lv3/whisper.py): per 30 s segment of L samples token_len =
            # (L − 1) // (160 · 8) + 1, the encoder output sliced to token_len · 4 positions → token_len LLM positions
            n = max(1, round(seconds * self.rate))
            seg = self.chunk * self.patches[0]
            full, rem = divmod(n, seg)
            step = self.patches[0] * 2 * self.stack
            return full * ((seg - 1) // step + 1) + ((rem - 1) // step + 1 if rem else 0)
        if self.kind == "whisper" and self.pad:
            # Qwen2-Audio processor: num_audio_tokens = (((L − 1)//2 + 1) − 2)//2 + 1 over the unpadded frames L
            # (conv2 stride 2, AvgPool1d 2); Whisper (cross): encoder positions of every window
            if self.cross:
                return self.enc_positions(seconds)
            L = min(self.mel_frames(seconds), self.chunk)
            t = (L - 1) // 2 + 1
            return (t - 2) // self.pool + 1 if self.pool > 1 else t
        if self.kind == "whisper" and self.ds and self.pool > 1:
            # 0.272 Step-Audio-2 forward: feat_lens = ((L + 1)//2//2 − 1)//2 + 1 (conv2, AvgPool 2, adaptor conv s2)
            return sum(self.chunk_len(n)[1] for n in self.chunks(seconds))
        if self.kind == "whisper" and self.pool > 1:
            # 0.271 Qwen2.5-Omni processor: (((L − 1)//2 + 1) − 2)//2 + 1 over the valid frames L (conv2 stride 2,
            # stride-2 average pool over the clip's concatenated chunk outputs) 「来源」 processing_qwen2_5_omni.py
            t = (self.mel_frames(seconds) - 1) // 2 + 1
            return (t - 2) // self.pool + 1
        if self.kind == "whisper":
            return sum(self.chunk_len(n)[1] for n in self.chunks(seconds))
        if self.kind == "w2v":
            # 0.336 OmniASR / Wav2Vec2 _get_feat_extract_output_lengths 「来源」: (L − k) // s + 1 per Conv1d
            n = round(seconds * self.rate)
            for k, st, _ in self.ds:
                n = (n - k) // st + 1 if n >= k else 0
            return max(0, n)
        return math.ceil(round(seconds * self.rate) / self.samples_per_token)


def _gemma4_audio_spec(cfg: dict, ac: dict) -> AudioSpec | None:
    """gemma4_audio conformer plus embed_audio. None when a width has no header path."""
    text_h = (cfg.get("text_config") or {}).get("hidden_size")
    h = ac.get("hidden_size") or 0
    layers = ac.get("num_hidden_layers") or 0
    heads = ac.get("num_attention_heads") or 0
    chans = list(ac.get("subsampling_conv_channels") or [])
    out_d = ac.get("output_proj_dims") or 0
    kernel = ac.get("conv_kernel_size") or 0
    chunk = ac.get("attention_chunk_size") or 0
    ctx_l = int(ac.get("attention_context_left") or 0)
    ctx_r = int(ac.get("attention_context_right") or 0)
    if not (text_h and h and layers and heads and out_d and kernel and chunk and ctx_l >= 1):
        return None
    if h % heads or not ac.get("use_clipped_linears"):
        return None
    # Released hidden_act is silu. residual_weight is the FFN branch scale; the class
    # default is 0.5 and every catalogued gemma4_audio checkpoint stores 0.5.
    # A different activation or scale is not the SiLU / half-residual recipe.
    act = ac.get("hidden_act", "silu")
    scale = ac.get("residual_weight", 0.5)
    if act != "silu":
        raise ValueError("gemma4_audio hidden_act is not silu")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or float(scale) != 0.5:
        raise ValueError("gemma4_audio residual_weight is not 0.5")
    if len(chans) != 2 or int(chans[0]) != _G4_AUDIO_MELS:
        return None
    c0, c1 = int(chans[0]), int(chans[1])
    # Gemma4AudioSubSampleConvProjection: proj_in = (c0 // 4) * c1. That is two stride-2
    # reductions of a mel axis whose width equals the first conv's channel count.
    proj_in = (c0 // 4) * c1
    if proj_in != h:
        return None
    hd = h // heads
    # Conv2d k3 s2 p1, no bias, plus LayerNorm weight. Linear(proj_in, h), no bias.
    sub = c0 * 9 + c0 + c1 * c0 * 9 + c1 + proj_in * h
    # Two FFNs. Each has two clipped linears (four bf16 scalars) and two RMSNorms. Width is 4h.
    one_ff = 2 * (h * 4 * h) + 2 * h + 8
    # GLU linear_start h to 2h, causal depthwise conv, linear_end h to h, two RMSNorms. Both linears are clipped.
    lconv = h * (2 * h) + 4 + h * h + 4 + h * kernel + 2 * h
    # q, k, v, post are clipped. relative_k_proj is a plain Linear. per_dim_scale is [head_dim]. Three RMSNorms.
    attn = 4 * (h * h + 4) + h * h + hd + 3 * h
    tower = sub + layers * (2 * one_ff + lconv + attn) + h * out_d + out_d
    # embed_audio: Linear(out_d, text), no bias. The pre-norm has with_scale=False.
    params = tower + out_d * text_h
    return AudioSpec(
        _G4_AUDIO_RATE, (_G4_AUDIO_HOP,), (h, c0, c1), (layers,), kernel, 4.0, text_h, params,
        kind="g4conf", heads=heads, n_mels=_G4_AUDIO_MELS, chunk=chunk, lin=((h, out_d),),
        maxf=_G4_AUDIO_CAP, frame=_G4_AUDIO_FRAME, ctx_l=ctx_l, ctx_r=ctx_r)


def _gs_idx(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _granite_speech_audio_spec(cfg: dict) -> AudioSpec | None:
    """GraniteSpeechForConditionalGeneration / GraniteSpeechPlusForConditionalGeneration.

    The tower is GraniteSpeechCTCEncoder (or the Plus encoder) plus a BLIP-2 Q-Former
    that is called with query embeds only. use_qformer_text_input false means the text
    FFN is not constructed. Shaw relative embeddings are a table; the einsum with the
    queries is the extra matmul. Returns None when a width has no header path.
    """
    mt = cfg.get("model_type")
    enc = cfg.get("encoder_config") if isinstance(cfg.get("encoder_config"), dict) else {}
    pc = cfg.get("projector_config") if isinstance(cfg.get("projector_config"), dict) else {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    pairs = {"granite_speech": "granite_speech_encoder", "granite_speech_plus": "granite_speech_plus_encoder"}
    if pairs.get(mt) != enc.get("model_type") or pc.get("model_type") != "blip_2_qformer":
        return None
    if cfg.get("has_lora_adapter") is not False or pc.get("use_qformer_text_input") is not False:
        return None
    h = enc.get("hidden_dim")
    layers = enc.get("num_layers")
    heads = enc.get("num_heads")
    dim_head = enc.get("dim_head")
    ff_mult = enc.get("feedforward_mult")
    exp = enc.get("conv_expansion_factor")
    kern = enc.get("conv_kernel_size")
    ind = enc.get("input_dim")
    mid = enc.get("output_dim")
    ctx = enc.get("context_size")
    pos = enc.get("max_pos_emb")
    win = cfg.get("window_size")
    down = cfg.get("downsample_rate")
    qh = pc.get("hidden_size")
    qheads = pc.get("num_attention_heads")
    ql = pc.get("num_hidden_layers")
    qff = pc.get("intermediate_size")
    freq = pc.get("cross_attention_frequency")
    cross = pc.get("encoder_hidden_size")
    text_h = text.get("hidden_size")
    if not all(_pos_int(x) for x in (
            h, layers, heads, dim_head, ff_mult, exp, kern, ind, mid, ctx, pos, win, down,
            qh, qheads, ql, qff, freq, cross, text_h)):
        return None
    if ctx > pos or dim_head * heads != h or qh % qheads or win % down:
        return None
    cat = enc.get("cat_hidden_layers") or []
    if not isinstance(cat, list) or any(not _gs_idx(i) or i > layers for i in cat):
        return None
    # The speech encoder has no cat path. A non-empty list there is not this module.
    if enc.get("model_type") == "granite_speech_encoder" and cat:
        return None
    exported = set(cat)
    if cross != h * (1 + len(exported)):
        return None
    ff = h * ff_mult
    inner = h * exp
    inner_dim = dim_head * heads
    ff1 = 2 * h + _lin(h, ff) + _lin(ff, h)
    attn = (2 * h + h * inner_dim + h * (2 * inner_dim) + _lin(inner_dim, h)
            + (2 * pos + 1) * dim_head)
    conv = 2 * h + _lin(h, inner * 2) + inner * kern + 2 * inner + _lin(inner, h)
    block = 2 * ff1 + attn + conv + 2 * h
    params = layers * block + _lin(ind, h) + _lin(h, mid) + _lin(mid, h)
    for i in range(ql):
        params += 4 * _lin(qh, qh) + 2 * qh
        if i % freq == 0:
            params += 2 * _lin(qh, qh) + 2 * _lin(cross, qh) + 2 * qh
        params += _lin(qh, qff) + _lin(qff, qh) + 2 * qh
    params += 2 * qh + (win // down) * qh + _lin(qh, text_h)
    return AudioSpec(16000, (160,), (h,), (layers,), kern, float(ff_mult), text_h, params, kind="gspeech",
                     heads=heads, n_mels=ind, chunk=ctx, win=win, gs_q=win // down, gs_cross=cross, gs_mid=mid,
                     gs_pos=pos, gs_ff=ff, gs_inner=inner, gs_freq=freq, gs_ql=ql, gs_qh=qh, gs_qheads=qheads,
                     gs_qff=qff)


def _conv_same(length: int, kernel: int, stride: int) -> int:
    """Conv1d length with padding (kernel - 1) // 2. Stride 1 keeps length when kernel is odd."""
    pad = (kernel - 1) // 2
    return (length + 2 * pad - kernel) // stride + 1


def _granite_speech5_ctc_audio_spec(cfg: dict) -> AudioSpec | None:
    """GraniteSpeech5ForCTC: the conformer encoder and a CTC head tied to encoder.out.

    There is no granite LM and no Q-Former. Q/K/V are bias-free. o_proj, both FFN linears,
    the input linear, and both CTC linears have bias (attention_bias is true). Shaw's table
    is an embedding of (2 * max_position_embeddings + 1, head_dim). The distance buffer is
    not stored. BatchNorm affine is a parameter; running stats are not. Returns None when
    a width has no header path.
    """
    if cfg.get("model_type") != "granite_speech5_ctc":
        return None
    enc = cfg.get("encoder_config") if isinstance(cfg.get("encoder_config"), dict) else {}
    if enc.get("model_type") != "granite_speech5_encoder":
        return None
    if cfg.get("tie_word_embeddings") is not True or enc.get("attention_bias") is not True:
        return None
    if enc.get("hidden_act") != "silu":
        return None
    h = enc.get("hidden_size")
    layers = enc.get("num_hidden_layers")
    heads = enc.get("num_attention_heads")
    kv = enc.get("num_key_value_heads")
    hd = enc.get("head_dim")
    ff = enc.get("intermediate_size")
    exp = enc.get("conv_expansion_factor")
    kern = enc.get("conv_kernel_size")
    ctx = enc.get("context_size")
    pos = enc.get("max_position_embeddings")
    mel = enc.get("num_mel_bins")
    vocab = enc.get("vocab_size")
    sub = enc.get("subsample_layers")
    if not all(_pos_int(x) for x in (h, layers, heads, kv, hd, ff, exp, kern, ctx, pos, mel, vocab)):
        return None
    if kv != heads or hd * heads != h or ctx > pos or kern % 2 == 0:
        return None
    if cfg.get("vocab_size") != vocab:
        return None
    if not isinstance(sub, list) or len(set(sub)) != len(sub) or any(not _gs_idx(i) or i >= layers for i in sub):
        return None
    ind = mel * 4
    inner = h * exp
    ff1 = _lin(h, ff) + _lin(ff, h)
    attn = 3 * (h * h) + _lin(h, h) + (2 * pos + 1) * hd
    conv = _lin(h, inner * 2) + inner * kern + 2 * inner + _lin(inner, h)
    block = 2 * ff1 + attn + conv + 5 * (2 * h)
    params = layers * block + _lin(ind, h) + _lin(h, vocab) + _lin(vocab, h)
    return AudioSpec(16000, (160,), (h,), (layers,), kern, float(ff) / h, vocab, params, kind="gsctc",
                     heads=heads, n_mels=ind, chunk=ctx, gs_mid=vocab, gs_pos=pos, gs_ff=ff, gs_inner=inner,
                     gs_sub=tuple(sub))


_D1_RATE = 16000
_D1_HOP = 160
_D1_MIN_SAMPLES = 8000
_D1_MAX_SAMPLES = 30 * _D1_RATE
_D1_PAD = 1024
_D1_TILE = 512
_D1_PX = 32
_D1_TEXT_FLOOR = 64


def _d1_conv_len(n: int) -> int:
    """Conv2d k3 s2 p1 length: (n + 2 - 3) // 2 + 1."""
    return (n - 1) // 2 + 1 if n > 0 else 0


def _d1_audio_lens(seconds: float) -> tuple[int, int]:
    """(mel frames, FastConformer prefix tokens).

    waveform() drops a missing clip, pads under 0.5 s up to 8000 samples, and cuts at 30 s.
    n_fft is even, so the centered STFT frame count is samples // hop. Three stride-2 convs
    then keep (L - 1) // 2 + 1.
    """
    if seconds <= 0:
        return 0, 0
    n = min(_D1_MAX_SAMPLES, max(0, round(seconds * _D1_RATE)))
    if n <= 0:
        return 0, 0
    if n < _D1_MIN_SAMPLES:
        n = _D1_MIN_SAMPLES
    mel = n // _D1_HOP
    t = mel
    for _ in range(3):
        t = _d1_conv_len(t)
    return mel, t


def _d1_ff_width(intermediate: int, mult: float, multiple: int) -> int:
    """Lfm2MLP auto-adjust, always on for this encoder (no block_auto_adjust_ff_dim flag)."""
    hidden = int(mult * int(2 * intermediate / 3))
    return multiple * ((hidden + multiple - 1) // multiple)


def _d1_head_params(d: int, layers: int) -> int:
    """DecisionHead: type embedding, pre-norm TransformerEncoderLayer, scorer LN-Linear-GELU-Linear."""
    ff = 4 * d
    # in_proj (3d, d) + bias, out_proj, fc1, fc2, two LayerNorms. ReLU and GELU have no parameters.
    layer = 4 * d * d + 2 * ff * d + ff + 9 * d
    scorer = 2 * d + (d * d + d) + (d + 1)
    return layers * layer + 3 * d + scorer


def _d1_audio_spec(cfg: dict) -> AudioSpec | None:
    """FastConformer prefix tower. None unless the released widths match audio.py."""
    ac = cfg.get("audio_config") or {}
    tc = cfg.get("text_config") or {}
    d = ac.get("d_model")
    layers = ac.get("n_layers")
    heads = ac.get("n_heads")
    feat = ac.get("feat_in")
    ch = ac.get("subsampling_conv_channels")
    ff_exp = ac.get("ff_expansion_factor")
    kernel = ac.get("conv_kernel_size")
    res_w = ac.get("residual_width")
    text_h = tc.get("hidden_size")
    if not all(_pos_int(x) for x in (d, layers, heads, feat, ch, kernel, res_w, text_h)):
        return None
    if feat != 128 or ff_exp != 4 or kernel % 2 == 0 or d % heads:
        return None
    ff = d * ff_exp
    # Conv2d / Linear biases are on. Depthwise conv is groups=channels. BatchNorm affine is
    # stored; running stats are buffers and are not parameters.
    sub = (ch * 9 + ch) + 2 * (ch * 9 + ch) + 2 * (ch * ch + ch)
    freq = feat
    for _ in range(3):
        freq = _d1_conv_len(freq)
    sub += _lin(ch * freq, d, True)
    attn = 4 * _lin(d, d) + d * d + 2 * heads * (d // heads)
    conv = _lin(d, 2 * d) + (d * kernel + d) + 2 * d + _lin(d, d)
    norms = 5 * 2 * d
    block = 2 * ( _lin(d, ff) + _lin(ff, d)) + attn + conv + norms
    adapter = 2 * d + _lin(d, text_h) + _lin(text_h, text_h)
    residual = 2 * text_h + _lin(text_h, res_w) + _lin(res_w, text_h)
    params = sub + layers * block + adapter + residual
    return AudioSpec(_D1_RATE, (_D1_HOP,), (d,), (layers,), kernel, float(ff_exp), text_h, params,
                     kind="d1fc", heads=heads, n_mels=feat, ch=ch, gs_ff=ff, gs_inner=res_w,
                     w_bits=32.0, w_fmt="fp32")


def _d1_trunk_params(tc: dict) -> tuple[int, int, int, tuple[str, ...]] | None:
    """(trunk params, ffn width, head dim, layer types). None when the encoder is not this class."""
    h = tc.get("hidden_size")
    vocab = tc.get("vocab_size")
    n = tc.get("num_hidden_layers")
    nq = tc.get("num_attention_heads")
    nkv = tc.get("num_key_value_heads")
    types = tc.get("layer_types")
    cache = tc.get("conv_L_cache")
    mult = tc.get("block_ffn_dim_multiplier")
    multiple = tc.get("block_multiple_of")
    inter = tc.get("intermediate_size")
    if not all(_pos_int(x) for x in (h, vocab, n, nq, nkv, cache, multiple, inter)):
        return None
    if isinstance(mult, bool) or not isinstance(mult, (int, float)) or mult <= 0:
        return None
    if cache != 3 or h % nq or nkv > nq or nq % nkv or not isinstance(types, list) or len(types) != n:
        return None
    if any(k not in ("conv", "full_attention") for k in types):
        return None
    hd = h // nq
    ff = _d1_ff_width(inter, float(mult), multiple)
    # No biases. q / k RMSNorm is head_dim. Two RMSNorms per block. embedding_norm is one more.
    conv = h * 3 * h + h * 3 + h * h + 2 * h * ff + ff * h + 2 * h
    attn = h * h + 2 * (nkv * hd) * h + h * h + 2 * hd + 2 * h * ff + ff * h + 2 * h
    n_conv = sum(k == "conv" for k in types)
    n_attn = n - n_conv
    params = vocab * h + h + n_conv * conv + n_attn * attn
    return params, ff, hd, tuple(types)


def audio_spec(cfg: dict) -> AudioSpec | None:
    ac = cfg.get("audio_config") or {}
    mt = cfg.get("model_type")
    if mt == "d1_omni":
        return _d1_audio_spec(cfg)
    if mt == "ming_v3_asr" and ac.get("n_audio_state"):
        # modeling_whisper_encoder.py hardcodes F.gelu / nn.GELU(). Refuse a contradictory field.
        _keep_erf_gelu("ming", ac)
        return _whisper_audio_spec(cfg, ac)
    if mt == "whisper":
        _require_erf_gelu("whisper", cfg.get("activation_function"))
        return _whisper_v3_spec(cfg, cfg["encoder_layers"], cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"],
                                cfg["num_mel_bins"], cfg["max_source_positions"], cross=True)
    if mt == "qwen2_audio" and ac.get("model_type") == "qwen2_audio_encoder":
        _require_erf_gelu("qwen2_audio", ac.get("activation_function"))
        out = (cfg.get("text_config") or {}).get("hidden_size") or 4096     # Qwen2Config default (text_config omits it)
        return _whisper_v3_spec({"d_model": ac["d_model"]}, ac["encoder_layers"], ac["encoder_attention_heads"],
                                ac["encoder_ffn_dim"], ac["num_mel_bins"], ac["max_source_positions"], pool=2,
                                trunc=True, lin=((ac["d_model"], out),))
    tk = cfg.get("thinker_config") or {}
    if mt == "qwen2_5_omni" and (tk.get("audio_config") or {}).get("model_type") == "qwen2_5_omni_audio_encoder":
        _require_erf_gelu("qwen2_5_omni", (tk.get("audio_config") or {}).get("activation_function"))
        return _omni_audio_spec(tk["audio_config"])
    kw = cfg.get("_kimi_whisper") or {}
    if kw:
        # The bundled whisper dict often omits activation_function. The copy is WhisperModel.encoder.
        _keep_erf_gelu("kimi", kw)
        return _kimi_audio_spec(kw, cfg)
    sa = cfg.get("audio_encoder_config") or {}
    if mt == "step_audio_2" and sa.get("n_audio_state"):
        _keep_erf_gelu("step_audio", sa)
        return _step_audio_spec(sa)
    if mt == "fun_asr_nano" and ac.get("model_type") == "fun_asr_nano_encoder":
        return _sanm_spec(ac, cfg.get("adaptor_config") or {})
    if mt == "qwen3_asr" and (tk.get("audio_config") or {}).get("model_type") == "qwen3_asr_audio_encoder":
        return _aut_spec(tk["audio_config"])
    if mt == "qwen3_omni_moe" and (tk.get("audio_config") or {}).get("model_type") == "qwen3_omni_moe_audio_encoder":
        # 0.345 Qwen3-Omni Thinker AuT (same encoder type as Index-Echo) 「来源」 configuration_qwen3_omni_moe.py
        return _aut_spec(tk["audio_config"])
    if mt == "index_echo_s2st" and ac.get("model_type") == "qwen3_omni_moe_audio_encoder":
        # 0.279 Index-Echo ST LM (code/stlm/audio_model.py 「来源」): the Qwen3-Omni AuT (same encoder as Qwen3-ASR:
        # WhisperFeatureExtractor 128 mel with n_samples raised to 300 s, only the valid frames encoded) → connector.
        # 2B AudioConnector.forward is exp(log_alpha) * (h + beta * w2(F.gelu(w1(h)))). F.gelu is the erf form.
        # The two scalars and the plain residual add are not ops. 9B AudioConnector9B.forward is proj(h) only;
        # the gain is folded into the stored weight. Shapes come from the release (stlm_ckpt).
        a = _aut_spec(ac)
        ct = cfg.get("_echo_connector") or {}
        pairs = [(n, sh) for n, sh in sorted(ct.items()) if len(sh) == 2]
        names = [n for n, _sh in pairs]
        mats = tuple((sh[1], sh[0]) for _n, sh in pairs)
        extra: tuple[int, ...] = ()
        if names == ["w1.weight", "w2.weight"]:
            extra = (len(a.lin),)
        elif names == ["proj.weight"] or not names:
            extra = ()
        else:
            raise ValueError("index echo connector is not w1/w2 or proj")
        return replace(a, lin=a.lin + mats, out=mats[-1][1] if mats else a.out,
                       params=a.params + sum(math.prod(sh) for sh in ct.values()),
                       gelu_lin=a.gelu_lin + extra)
    if mt in ("omniasr", "omniasr_ctc") and ac.get("model_type") == "omniasr_audio":
        # 0.336 OmniASR LLM / CTC (bezzam/*-v2 HF ports of facebook/omniASR): Wav2Vec2-style audio_tower +
        # multi_modal_projector (LLM) or CTC lm_head (CTC — encoder only; language decode not modelled as asr)
        return _omniasr_audio_spec(cfg, ac)
    if ac.get("model_type") == "gemma4_unified_audio":
        # 12B: one projection. The E2B / E4B conformer is gemma4_audio, below.
        d_in = ac.get("output_proj_dims")
        if not d_in or d_in != ac.get("audio_embed_dim"):
            return None
        spt = ac.get("audio_samples_per_token") or 0
        text_h = (cfg.get("text_config") or {}).get("hidden_size")
        if spt <= 0 or not text_h:
            return None
        return AudioSpec(16000, (spt,), (d_in,), (0,), 1, 1.0, text_h, d_in * text_h, kind="g4lin")
    if ac.get("model_type") == "gemma4_audio":
        return _gemma4_audio_spec(cfg, ac)
    if mt in ("granite_speech", "granite_speech_plus"):
        return _granite_speech_audio_spec(cfg)
    if mt == "granite_speech5_ctc":
        return _granite_speech5_ctc_audio_spec(cfg)
    if ac.get("model_type") != "muse_spark_audio":
        return None
    pt, dm, dp, k, r = tuple(ac["patch_sizes"]), tuple(ac["hidden_sizes"]), tuple(ac["depths"]), ac["conv_kernel_size"], ac["mlp_ratio"]
    out = ac["out_hidden_size"]
    fe = cfg.get("_feature_extractor") or {}
    din = (1,) + dm[:-1]
    params = 0
    for i in range(len(dm)):
        d, hd = dm[i], int(dm[i] * r)
        params += _lin(din[i] * pt[i], d) + d + dp[i] * (d * k + d + d + _lin(d, hd) + _lin(hd, d))
    params += dm[-1] + _lin(dm[-1], out) + out
    return AudioSpec(fe.get("sampling_rate", 24000), pt, dm, dp, k, r, out, params)


def _whisper_audio_spec(cfg: dict, ac: dict) -> AudioSpec:
    """0.268 Ming-3.0-asr-tiny audio tower + linear_proj_audio (released header: conv1 [1280, 128, 3], conv2
    [1280, 1280, 3], positional_embedding [3000, 1280] stored, key without bias, projector conv [1536, 1280, 3] +
    Linear [1536, 1536]).  The checkpoint's second projector ``linear_proj`` (4096 → 1536 → 1536) is ignored on load
    (_keys_to_ignore_on_load_unexpected) → storage only (``VisionSpec.extra``)."""
    d, L, nh, mel = ac["n_audio_state"], ac["n_audio_layer"], ac["n_audio_head"], ac.get("n_mels", 128)
    out = (cfg.get("text_config") or {}).get("hidden_size") or cfg.get("hidden_size")
    ks = list(ac.get("downsampler_kernel_size_list") or [3])
    ss = list(ac.get("downsampler_stride_list") or [2])
    ps = list(ac.get("downsampler_padding_list") or [k // 2 for k in ks])
    blk = 4 * d * d + 3 * d + 2 * d + _lin(d, 4 * d) + _lin(4 * d, d) + 2 * d
    params = _lin(3 * mel, d) + _lin(3 * d, d) + ac.get("n_audio_ctx", 3000) * d + L * blk + 2 * d
    din = d
    for k in ks:
        params += _lin(k * din, out)
        din = out
    params += _lin(out, out)
    return AudioSpec(16000, (160,), (d,), (L,), 3, 4.0, out, params, kind="whisper", heads=nh, n_mels=mel,
                     chunk=ac.get("chunk_frames", 3000), ds=tuple(zip(ks, ss, ps)))


def _whisper_v3_spec(cfg: dict, L: int, nh: int, ffn: int, mel: int, pos: int, *, cross: bool = False, pool: int = 1,
                     trunc: bool = False, lin: tuple = ()) -> AudioSpec:
    """0.270 Whisper-large-v3 encoder (transformers WhisperEncoder; Qwen2AudioEncoder = the same + AvgPool1d(2, 2)
    before layer_norm): conv1 [d, mel, 3] + conv2 [d, d, 3, stride 2] (GELU), positions [max_source_positions, d]
    (Whisper: sinusoid buffer saved in the checkpoint; Qwen2-Audio: nn.Embedding), L pre-LN blocks (q / v / o with bias,
    k without, GELU fc1 / fc2 with bias), final layer_norm; optional projector Linears with bias.  Inputs padded to 30 s
    (3000 frames → 1500 positions) by the feature extractor 「来源」."""
    d = cfg["d_model"]
    blk = 4 * d * d + 3 * d + 2 * d + _lin(d, ffn) + _lin(ffn, d) + 2 * d
    params = _lin(3 * mel, d) + _lin(3 * d, d) + pos * d + L * blk + 2 * d + sum(_lin(k, n) for k, n in lin)
    out = lin[-1][1] if lin else d
    return AudioSpec(16000, (160,), (d,), (L,), 3, ffn / d, out, params, kind="whisper", heads=nh, n_mels=mel,
                     chunk=2 * pos, pad=True, trunc=trunc, cross=cross, pool=pool, lin=tuple(lin))


def _omni_audio_spec(ac: dict) -> AudioSpec:
    """0.271 Qwen2.5-Omni audio tower (transformers Qwen2_5OmniAudioEncoder, modeling_qwen2_5_omni.py 「来源」): the
    Whisper-v3 encoder (conv1 k3 + conv2 k3 s2, GELU; L pre-LN blocks, k without bias) run per chunk of 2·n_window =
    200 mel frames (chunk_and_pad_features; attention within a chunk, cu_seqlens); sinusoid positions are a buffer (not
    in the checkpoint); pair average pool, then ln_post, then proj d → output_dim; audio_bos_eos_token Embedding(2, output_dim)
    stored (looked up for the two marker positions).  Header: 639,647,232 params."""
    d, out = ac["d_model"], ac["output_dim"]
    a = _whisper_v3_spec({"d_model": d}, ac["encoder_layers"], ac["encoder_attention_heads"], ac["encoder_ffn_dim"],
                         ac["num_mel_bins"], 0, pool=2, lin=((d, out),))
    return replace(a, params=a.params + 2 * out, chunk=2 * ac["n_window"], pad=False,
                   maxf=2 * ac["max_source_positions"], stored=2 * out)


def _aut_spec(ac: dict) -> AudioSpec:
    """0.270 Qwen3-ASR audio encoder (transformers Qwen3ASREncoder, modeling_qwen3_asr.py 「来源」): mel frames padded to
    chunks of 2·n_window (100 frames = 1 s); per chunk three Conv2d (k3, s2, p1; 1 → C → C → C channels, GELU) over
    (mel, time), conv_out Linear C·⌈⌈⌈mel/2⌉/2⌉/2⌉ → d (no bias), sinusoid positions; only the valid post-CNN positions
    (13 per full chunk) enter the L blocks (q / k / v / o with bias, GELU fc1 / fc2), attention within windows of
    13 · n_window_infer / (2·n_window) positions (cu_seqlens), ln_post; projector Linear d → d, GELU, Linear d → out.

    0.416: ``activation_function`` must be ``gelu``. Conv ``F.gelu`` and ``ACT2FN['gelu']`` are the erf form.
    A different value is refused. The projector GELU is linear index 0.
    """
    _require_erf_gelu("aut", ac.get("activation_function"))
    d, L, nh, ffn, mel = ac["d_model"], ac["encoder_layers"], ac["encoder_attention_heads"], ac["encoder_ffn_dim"], ac["num_mel_bins"]
    C, nw = ac["downsample_hidden_size"], ac["n_window"]
    f = mel
    for _ in range(3):
        f = (f + 1) // 2
    blk = 4 * d * d + 4 * d + 2 * d + _lin(d, ffn) + _lin(ffn, d) + 2 * d
    lin = ((d, d), (d, ac["output_dim"]))
    params = (_lin(9, C) + 2 * _lin(9 * C, C) + C * f * d + L * blk + 2 * d + sum(_lin(k, n) for k, n in lin))
    a = AudioSpec(16000, (160,), (d,), (L,), 3, ffn / d, ac["output_dim"], params, kind="aut", heads=nh, n_mels=mel,
                  chunk=2 * nw, lin=lin, ch=C, gelu_lin=(0,))
    return replace(a, win=a.aut_len(2 * nw) * (ac["n_window_infer"] // (2 * nw)))


def _kimi_audio_spec(w: dict, cfg: dict) -> AudioSpec:
    """0.273 Kimi-Audio-7B-Instruct audio input (kimia_infer whisper_Lv3/whisper.py + modeling_moonshot_kimia.py
    VQAdaptor 「来源」): the repo's whisper-large-v3 encoder (WhisperModel.encoder, bf16) on every 30 s segment padded
    to 3000 mel frames (1500 positions; sequential segments, no truncation); output sliced to 4 · token_len positions,
    reshaped 4 × 1280 → 5120 per 12.5 Hz position; VQAdaptor = Linear 5120 → hidden + SiLU, Linear hidden → hidden,
    LayerNorm (bias).  The continuous features are added to the discrete GLM-4-Voice semantic-token embeddings at the
    same positions — that tokenizer (THUDM/glm-4-voice-tokenizer, a separate repo) is not costed (coverage reason)."""
    d, h = w["d_model"], cfg["hidden_size"]
    st = cfg.get("kimia_adaptor_input_dim", 4 * d) // d
    a = _whisper_v3_spec({"d_model": d}, w["encoder_layers"], w["encoder_attention_heads"], w["encoder_ffn_dim"],
                         w["num_mel_bins"], w["max_source_positions"], lin=((st * d, h), (h, h)))
    return replace(a, params=a.params + 2 * h, stack=st)


def _step_audio_spec(sa: dict) -> AudioSpec:
    """0.272 Step-Audio-2-mini audio encoder (modeling_step_audio_2.py AudioEncoder + Adaptor 「来源」): log-mel 128
    (audio padded by 479 samples, hop 160), conv1 k3 + conv2 k3 s2 (GELU), learned positions Embedding(n_audio_ctx,
    d) (stored), L Whisper blocks (query / value / out with bias, key without; GELU MLP 4d), AvgPool1d(2, 2), LayerNorm;
    Adaptor: Conv1d(d, d, k, stride, padding 1) + GELU, Linear d → 2048, ReLU, Linear 2048 → llm_dim.  One pass over
    the whole clip (no chunking, full attention); conv2 positions ≤ n_audio_ctx → at most 2·n_ctx − 1 mel frames."""
    d, L, nh, mel, ctx = sa["n_audio_state"], sa["n_audio_layer"], sa["n_audio_head"], sa["n_mels"], sa["n_audio_ctx"]
    k, st = sa.get("kernel_size", 3), sa.get("adapter_stride", 2)
    lin = ((d, 2048), (2048, sa["llm_dim"]))
    a = _whisper_v3_spec({"d_model": d}, L, nh, 4 * d, mel, ctx, pool=2, lin=lin)
    return replace(a, params=a.params + _lin(k * d, d), pad=False, chunk=2 * ctx - 1, maxf=2 * ctx - 1,
                   ds=((k, st, 1),), mpad=479)


def _sanm_spec(ac: dict, ad: dict) -> AudioSpec:
    """0.272 Fun-ASR-Nano audio tower (transformers FunAsrNanoEncoder / FunAsrNanoMultiModalProjector,
    modeling_fun_asr_nano.py 「来源」; SenseVoice SAN-M): input = LFR features (80 mel × 7 stacked = 560, stride 6),
    × √d + sinusoid positions (buffer, from position 1); L pre-LN blocks [LayerNorm, q / k / v / o Linear with bias
    (layer 0 maps 560 → d, no residual), FSMN depthwise Conv1d k (no bias) on the unsplit values added to the attention
    output, LayerNorm, ReLU MLP fc1 / fc2 with bias]; bidirectional attention over the whole utterance (no chunking);
    final LayerNorm on layers L − n_tp − 1 and L − 1 only.  All L layers (incl. the n_tp timestamp-prediction layers)
    run in forward.  Projector: Linear d → P, ReLU, Linear P → D, then the adaptor's n blocks (d = D, no FSMN, ReLU MLP
    D/4).  One LLM token per LFR frame.  Header: 233,741,920 params.

    0.414 bills those LayerNorms, the ReLU, and the FSMN mask multiply in ``_sanm_ops``.
    ``hidden_act`` / ``projector_hidden_act`` must be ``relu``. The two output-LayerNorm
    layers must be distinct encoder indices. Anything else is refused.
    """
    if ac.get("hidden_act") != "relu" or ad.get("hidden_act") != "relu" or ad.get("projector_hidden_act") != "relu":
        raise ValueError("fun_asr_nano hidden_act is not relu")
    d, L, nh, ffn, k = ac["hidden_size"], ac["num_hidden_layers"], ac["num_attention_heads"], ac["intermediate_size"], ac["fsmn_kernel_size"]
    m = ac.get("num_stacked_frames", 7)
    din = ac["num_mel_bins"] * m
    D, P, nl = ad["hidden_size"], ad["projector_hidden_size"], ad["num_hidden_layers"]
    adh, adf = ad["num_attention_heads"], ad["intermediate_size"]
    n_tp = ac.get("num_timestamp_prediction_layers")
    # Output LayerNorm exists only at (L - n_tp - 1) and (L - 1). The param total
    # counts two of them (2 * 2 * d). A repeated or out-of-range index is refused.
    if isinstance(n_tp, bool) or not isinstance(n_tp, int):
        raise ValueError("fun_asr_nano timestamp layers are missing")
    i0, i1 = L - n_tp - 1, L - 1
    if not (0 <= i0 < i1 < L):
        raise ValueError("fun_asr_nano final LayerNorm layers are not two encoder layers")

    def blk(i: int, dd: int, f: int, fsmn: bool) -> int:
        return 2 * i + 3 * _lin(i, dd) + _lin(dd, dd) + (dd * k if fsmn else 0) + 2 * dd + _lin(dd, f) + _lin(f, dd)
    params = (blk(din, d, ffn, True) + (L - 1) * blk(d, d, ffn, True) + 2 * 2 * d
              + _lin(d, P) + _lin(P, D) + nl * blk(D, D, adf, False))
    return AudioSpec(16000, (160,), (d,), (L,), k, ffn / d, D, params, kind="sanm", heads=nh, n_mels=din,
                     lin=((d, P), (P, D)), maxf=ac["max_position_embeddings"] - 1, lfr=(m, 6), ad=(nl, D, adh, adf),
                     sanm_final=(i0, i1))


def _omniasr_ctc_ready(cfg: dict, ac: dict) -> bool:
    """True when the config is OmniASRForCTC: uniform conv channels, biased convs, GELU, and a vocab."""
    vocab = cfg.get("vocab_size")
    d = ac.get("hidden_size")
    layers = ac.get("num_hidden_layers")
    heads = ac.get("num_attention_heads")
    ff = ac.get("intermediate_size")
    groups = ac.get("num_conv_pos_embedding_groups")
    pk = ac.get("num_conv_pos_embeddings")
    cd = ac.get("conv_dim")
    ck = ac.get("conv_kernel")
    cs = ac.get("conv_stride")
    if ac.get("conv_bias") is not True or ac.get("hidden_act") != "gelu":
        return False
    if not all(_pos_int(x) for x in (vocab, d, layers, heads, ff, groups, pk)):
        return False
    if d % heads or d % groups:
        return False
    if not isinstance(cd, list) or not isinstance(ck, list) or not isinstance(cs, list):
        return False
    if not cd or len(cd) != len(ck) or len(cd) != len(cs):
        return False
    if any(not _pos_int(x) for x in (*cd, *ck, *cs)):
        return False
    if any(x != cd[0] for x in cd):
        return False
    return True


def _omniasr_audio_spec(cfg: dict, ac: dict) -> AudioSpec | None:
    """0.336 OmniASR audio tower (bezzam/omniasr-*-v2 config + safetensors headers; Wav2Vec2-shaped
    transformers OmniASR audio): raw 16 kHz waveform -> 7 Conv1d (bias) + LayerNorm per layer (config conv_dim /
    conv_kernel / conv_stride) -> LayerNorm -> Linear last_conv -> hidden -> grouped positional Conv1d
    (num_conv_pos_embeddings x groups) -> L pre-LN blocks (q / k / v / o with bias, GELU intermediate / output with
    bias, 2 x LayerNorm) -> final LayerNorm. LLM releases add multi_modal_projector Linear hidden -> text_hidden
    (bias). 0.395 CTC releases add ctc_head Linear hidden -> vocab (bias, nn.Linear default) on every encoder
    frame and no projector. Frame length is Wav2Vec2 (L - k) // s + 1 per Conv1d. Returns None for a CTC config
    whose widths are not this class."""
    ctc = cfg.get("model_type") == "omniasr_ctc"
    if ctc and not _omniasr_ctc_ready(cfg, ac):
        return None
    d, L, nh, ff = ac["hidden_size"], ac["num_hidden_layers"], ac["num_attention_heads"], ac["intermediate_size"]
    cd, ck, cs = list(ac["conv_dim"]), list(ac["conv_kernel"]), list(ac["conv_stride"])
    g = ac.get("num_conv_pos_embedding_groups", 16)
    pk = ac.get("num_conv_pos_embeddings", 128)
    params = cd[0] * 1 * ck[0] + cd[0] + 2 * cd[0]
    for i in range(1, len(cd)):
        params += cd[i] * cd[i - 1] * ck[i] + cd[i] + 2 * cd[i]
    params += 2 * cd[-1] + _lin(cd[-1], d)                                  # post-conv LN + projection
    params += d * (d // g) * pk + d                                         # grouped positional conv + bias
    blk = 4 * _lin(d, d) + _lin(d, ff) + _lin(ff, d) + 4 * d
    params += L * blk + 2 * d
    out = d
    lin: tuple[tuple[int, int], ...] = ()
    tc = cfg.get("text_config") or {}
    if ctc:
        out = int(cfg["vocab_size"])
        lin = ((d, out),)
        params += _lin(d, out)
    elif cfg.get("model_type") == "omniasr" and tc.get("hidden_size"):
        out = tc["hidden_size"]
        lin = ((d, out),)
        params += _lin(d, out)
    ds = tuple((ck[i], cs[i], 0) for i in range(len(ck)))
    # as-released: bezzam OmniASR safetensors are float32 (release roles.vision fp32)
    dt = (cfg.get("dtype") or cfg.get("torch_dtype") or "").lower()
    wb, wf = (32.0, "fp32") if dt in ("float32", "fp32") else (16.0, "bf16")
    return AudioSpec(16000, (math.prod(cs),), (d,), (L,), pk, ff / d, out, params, kind="w2v", heads=nh,
                     n_mels=cd[-1], ds=ds, lin=lin, ch=g, w_bits=wb, w_fmt=wf, ctc=ctc)


def _w2v_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """0.336 OmniASR / Wav2Vec2 encoder ops: each Conv1d as an implicit GEMM over every output frame; positional
    conv as depthwise-ish GEMM; L bidirectional full-attention blocks; optional projector Linear.

    0.410: ``hidden_act`` is ``gelu``. ``ACT2FN["gelu"]`` is ``nn.functional.gelu`` (erf, not tanh).
    ``OmniASRLayerNormConvLayer`` runs it on every conv output, ``OmniASRPositionalConvEmbedding`` runs it
    after the even-kernel ``[:, :, :-1]`` slice, and ``OmniASRFeedForward`` runs it on the intermediate width.
    The class does not cast, so the GELU dtype is the released weight format. Requested GEMM activation
    format does not change it. Eval ``Dropout`` and ``layerdrop`` are identity. The projector and the CTC
    head have no GELU.

    0.417: the same classes run ``nn.LayerNorm`` (affine weight and bias, already in the parameter total).
    Each conv layer normalizes the conv channels after the transpose and before GELU.
    ``OmniASREncoderSubsamplingConv1D`` normalizes those channels again, then the projection.
    Each block normalizes ``hidden_size`` before attention and again before the FFN.
    ``OmniASRAudioModel`` normalizes ``hidden_size`` after the blocks. PyTorch reduces in fp32,
    one rsqrt per row. The positional conv, the plain residual adds, and the padding ``masked_fill``
    are not this op. The shared GEMM tail is not reduced.
    """
    d, dh = a.dims[0], a.dims[0] // a.heads
    T = a.tokens(seconds)
    R = T * clips
    ops: list[Op] = []

    def gemm(name: str, m: int, k: int, n: int, li: int = 0, wp: int | None = None) -> Op:
        return Op(name, "gemm", li, m=m, k=k, n=n, role="vision", w_params=(wp if wp is not None else k * n + n),
                  w_bits=a.w_bits, w_fmt=a.w_fmt, a_fmt=af, act_bytes=m * (k + n) * ab, stream=True,
                  vec=m * n * VEC_PER_OUT)

    # feature-extractor lengths per layer (same recurrence as tokens())
    n = round(seconds * a.rate)
    lens = []
    for k, st, _ in a.ds:
        n = (n - k) // st + 1 if n >= k else 0
        lens.append(max(0, n))
    din = 1
    for i, ((k, st, _), cd, Lout) in enumerate(zip(a.ds, [a.n_mels] * len(a.ds), lens)):
        # first layer in_ch=1, later in_ch=out_ch=n_mels (all OmniASR conv_dim equal)
        cin = 1 if i == 0 else a.n_mels
        cout = a.n_mels
        rows = Lout * clips
        ops.append(gemm(f"aud.fe.{i}", rows, k * cin, cout, 0, wp=cout * cin * k + cout))
        # Transpose, LayerNorm over conv channels, transpose back, then ACT2FN["gelu"].
        ops.append(_ln_vec(f"aud.fe.{i}.ln", 0, rows, cout))
        ops.append(_gelu_erf_vec(f"aud.fe.{i}.gelu", 0, rows * cout, a.w_fmt))
        din = cout
    if T:
        # Post-conv LayerNorm is still the conv width. The projection is the next GEMM.
        ops.append(_ln_vec("aud.fe.ln", 0, R, a.n_mels))
        ops.append(gemm("aud.fe.proj", R, a.n_mels, d, 0))
        # grouped positional conv: weight [d, d/g, pk] → treat as GEMM R × (pk·d/g) → d.
        # padding kernel//2 on an even kernel emits T+1, then [:, :, :-1] drops one frame.
        # GELU reads the sliced length, which is the encoder frame count. No LayerNorm here.
        ops.append(gemm("aud.pos", R, a.kernel * (d // a.ch), d, 0, wp=d * (d // a.ch) * a.kernel + d))
        ops.append(_gelu_erf_vec("aud.pos.gelu", 0, R * d, a.w_fmt))
        ff = int(a.mlp * d)
        for li in range(a.depths[0]):
            ops.append(_ln_vec(f"aud.{li}.attn_ln", li, R, d))
            ops.append(gemm(f"aud.{li}.qkv", R, d, 3 * d, li))
            ops += [Op(f"aud.{li}.qk", "attn", li, m=T, k=dh, n=T, count=clips * a.heads, causal=1.0,
                       act_bytes=clips * a.heads * T * dh * ab, stream=True, orient=True),
                    Op(f"aud.{li}.pv", "attn", li, m=T, k=T, n=dh, count=clips * a.heads, causal=1.0,
                       act_bytes=clips * a.heads * T * dh * ab, orient=True),
                    Op(f"aud.{li}.softmax", "vector", li, vec=clips * a.heads * T * T * 5.0)]
            ops.append(gemm(f"aud.{li}.o", R, d, d, li))
            ops.append(_ln_vec(f"aud.{li}.mlp_ln", li, R, d))
            ops.append(gemm(f"aud.{li}.fc1", R, d, ff, li))
            ops.append(_gelu_erf_vec(f"aud.{li}.gelu", li, R * ff, a.w_fmt))
            ops.append(gemm(f"aud.{li}.fc2", R, ff, d, li))
        ops.append(_ln_vec("aud.ln", a.depths[0], R, d))
        for j, (k, n) in enumerate(a.lin or ()):
            # CTC: one Linear on every frame (OmniASRForCTC.ctc_head). LLM: multi_modal_projector.
            name = "aud.ctc" if a.ctc else f"aud.proj.linear{j}"
            ops.append(gemm(name, R, k, n, a.depths[0]))
    return ops


def _sanm_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """0.272 Fun-ASR-Nano SAN-M encoder + projector + adaptor ops (see _sanm_spec): every layer on all T LFR positions
    of a clip (one pass per clip, bidirectional attention T × T per head); FSMN as a depthwise conv (2·k per output
    element).

    0.414: ``FunAsrNanoEncoderLayer`` (transformers ``modeling_fun_asr_nano.py``, sha ``a672eb5``)
    runs input LayerNorm, attention, then the FSMN. ``FunAsrNanoFSMN`` multiplies by the
    feature mask (cast to the activation dtype) before the depthwise conv. Post LayerNorm
    and ``ACT2FN["relu"]`` follow. ``ACT2FN["relu"]`` is ``nn.ReLU``: ``max(x, 0)``, one
    clip, no transcendental. Not ``nn.ReLU6``. The output LayerNorm is constructed only
    for ``sanm_final``; other layers store ``nn.Identity``. Layer 0's input LayerNorm is
    ``input_size`` (stacked mel), and its value projection is already ``hidden``. The
    projector runs one ReLU between its two linears. Adaptor layers are the same block
    with ``use_fsmn=False`` and ``add_norm=False``: no mask, no output LayerNorm.
    LayerNorm reduces in fp32. ReLU and the mask do not upcast; the release is bf16.
    A requested GEMM activation format does not change those dtypes. Eval dropout is
    identity. The ``sqrt(d)`` scale, the sinusoid add, the pad and the plain residual
    adds are not ops. The shared GEMM tail is not reduced. Softmax is not given an SFU.
    """
    final = a.sanm_final
    depth = a.depths[0]
    if len(final) != 2 or final[0] == final[1] or any(i < 0 or i >= depth for i in final):
        raise ValueError("fun_asr_nano final LayerNorm layers are not two encoder layers")
    d, dh = a.dims[0], a.dims[0] // a.heads
    T = a.lfr_frames(seconds)
    R = T * clips
    ops: list[Op] = []
    # Released checkpoint dtype. The class does not cast ReLU or the mask to the request.
    act = "bf16"

    def gemm(name: str, m: int, k: int, n: int, li: int = 0) -> Op:
        return Op(name, "gemm", li, m=m, k=k, n=n, role="vision", w_params=k * n + n, w_bits=16.0, w_fmt="bf16",
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    def attn(p: str, li: int, heads: int, hd: int) -> list[Op]:
        cnt = clips * heads
        return [Op(f"{p}{li}.qk", "attn", li, m=T, k=hd, n=T, count=cnt, causal=1.0, act_bytes=cnt * T * hd * ab,
                   stream=True, orient=True),
                Op(f"{p}{li}.pv", "attn", li, m=T, k=T, n=hd, count=cnt, causal=1.0, act_bytes=cnt * T * hd * ab,
                   orient=True),
                Op(f"{p}{li}.softmax", "vector", li, vec=cnt * T * T * 5.0)]
    ff = int(a.mlp * d)
    for li in range(depth):
        # Layer 0 normalizes the stacked mel width. Later layers normalize hidden.
        i = a.n_mels if li == 0 else d
        ops.append(_ln_vec(f"aud.{li}.in_ln", li, R, i))
        ops.append(gemm(f"aud.{li}.qkv", R, i, 3 * d, li))
        ops += attn("aud.", li, a.heads, dh)
        ops.append(gemm(f"aud.{li}.o", R, d, d, li))
        # Mask multiply reads v_proj, which is hidden wide even on layer 0.
        ops.append(Op(f"aud.{li}.mask", "vector", li, vec=R * d, a_fmt=act))
        ops.append(Op(f"aud.{li}.fsmn", "vector", li, vec=R * d * 2.0 * a.kernel, act_bytes=2 * R * d * ab,
                      w_params=d * a.kernel, w_bits=16.0))
        ops.append(_ln_vec(f"aud.{li}.post_ln", li, R, d))
        ops.append(gemm(f"aud.{li}.fc1", R, d, ff, li))
        ops.append(_relu_vec(f"aud.{li}.relu", li, R * ff, act))
        ops.append(gemm(f"aud.{li}.fc2", R, ff, d, li))
        if li in final:
            ops.append(_ln_vec(f"aud.{li}.out_ln", li, R, d))
    L = depth
    if len(a.lin) != 2:
        raise ValueError("fun_asr_nano projector is not two linears")
    for j, (k, n) in enumerate(a.lin):
        ops.append(gemm(f"aud.proj.linear{j}", R, k, n, L))
        if j == 0:
            ops.append(_relu_vec("aud.proj.relu", L, R * n, act))
    nl, D, h2, f2 = a.ad
    for j in range(nl):
        li = L + 1 + j
        ops.append(_ln_vec(f"aud.ad{j}.in_ln", li, R, D))
        ops.append(gemm(f"aud.ad{j}.qkv", R, D, 3 * D, li))
        ops += attn(f"aud.ad{j}.", li, h2, D // h2)
        ops.append(gemm(f"aud.ad{j}.o", R, D, D, li))
        ops.append(_ln_vec(f"aud.ad{j}.post_ln", li, R, D))
        ops.append(gemm(f"aud.ad{j}.fc1", R, D, f2, li))
        ops.append(_relu_vec(f"aud.ad{j}.relu", li, R * f2, act))
        ops.append(gemm(f"aud.ad{j}.fc2", R, f2, D, li))
    return ops


def _aut_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """AuT encoder GEMMs plus the activations the published forward actually runs.

    Qwen3ASREncoder and Qwen3OmniMoeAudioEncoder (transformers main ``c0d024f``) run
    ``F.gelu`` on each of the three padded Conv2d outputs, then the conv_out linear
    with no activation. Each block is pre-LN attention, then pre-LN MLP. The MLP
    activation is ``ACT2FN['gelu']``. ``ln_post`` follows the blocks. The projector
    is Linear, the same activation, Linear. ``F.gelu`` and ``ACT2FN['gelu']`` are the
    erf form, not ``gelu_pytorch_tanh``. Conv GELU reads the padded chunk grid. The
    LayerNorms, the FFN GELU and the projector read the valid encoder positions.
    Index-Echo 2B's ``AudioConnector.forward`` is ``F.gelu`` on ``w1``; 9B's connector
    is one linear. GELU does not upcast: ``a_fmt`` is ``vec_fmt`` (the vision-role
    dtype). LayerNorm reduces in fp32. The shared GEMM tail is not reduced. A
    requested GEMM activation format does not change these vector dtypes. GEMM
    ``w_fmt`` stays bf16. Eval dropout is not an op. The sinusoid add, the fp16
    clamp, the connector scalars and the plain residual adds are not ops.
    """
    d, dh, C = a.dims[0], a.dims[0] // a.heads, a.ch
    ff = a.mlp * d
    if ff != int(ff):
        raise ValueError("aut ffn width is not an integer")
    ff = int(ff)
    if not a.lin or len(a.lin) < 2:
        raise ValueError("aut projector is missing")
    (k0, n0), (k1, _n1) = a.lin[0], a.lin[1]
    if k0 != d or n0 != d or k1 != d:
        raise ValueError("aut projector is not linear, gelu, linear")
    gelu_at = tuple(a.gelu_lin)
    if 0 not in gelu_at or any(i < 0 or i >= len(a.lin) for i in gelu_at):
        raise ValueError("aut projector gelu is missing")
    act = a.vec_fmt or a.w_fmt
    if act not in ("fp16", "bf16", "fp32"):
        raise ValueError("aut activation dtype is not a released float format")
    ops: list[Op] = []

    def gemm(name: str, m: int, k: int, n: int, li: int = 0, w: int | None = None) -> Op:
        return Op(name, "gemm", li, m=m, k=k, n=n, role="vision", w_params=k * n if w is None else w, w_bits=16.0,
                  w_fmt="bf16", a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)
    nch = len(a.chunks(seconds)) * clips
    f, t = a.n_mels, a.chunk
    k_in = 1
    for i in range(3):
        f, t = (f + 1) // 2, (t + 1) // 2
        rows = nch * f * t
        ops.append(gemm(f"aud.conv2d{i + 1}", rows, 9 * k_in, C))
        # F.gelu on the padded conv output. The class does not upcast.
        ops.append(_gelu_erf_vec(f"aud.conv2d{i + 1}.gelu", 0, rows * C, act))
        k_in = C
    P = a.enc_positions(seconds) * clips
    ops.append(gemm("aud.conv_out", nch * t, C * f, d))
    wins = []
    for _ in range(clips):
        _full, rem = divmod(a.enc_positions(seconds), a.win)
        wins += [a.win] * _full + ([rem] if rem else [])
    depth = a.depths[0]
    for li in range(depth):
        ops.append(_ln_vec(f"aud.{li}.attn_ln", li, P, d))
        ops.append(gemm(f"aud.{li}.qkv", P, d, 3 * d, li))
        for wi, w in enumerate(sorted(set(wins), reverse=True)):
            cnt = wins.count(w) * a.heads
            sfx = f".w{wi}" if wi else ""
            ops += [Op(f"aud.{li}.qk{sfx}", "attn", li, m=w, k=dh, n=w, count=cnt, causal=1.0, act_bytes=cnt * w * dh * ab,
                       stream=True, orient=True),
                    Op(f"aud.{li}.pv{sfx}", "attn", li, m=w, k=w, n=dh, count=cnt, causal=1.0, act_bytes=cnt * w * dh * ab,
                       orient=True),
                    Op(f"aud.{li}.softmax{sfx}", "vector", li, vec=cnt * w * w * 5.0)]
        ops.append(gemm(f"aud.{li}.o", P, d, d, li))
        ops.append(_ln_vec(f"aud.{li}.mlp_ln", li, P, d))
        ops.append(gemm(f"aud.{li}.fc1", P, d, ff, li))
        ops.append(_gelu_erf_vec(f"aud.{li}.gelu", li, P * ff, act))
        ops.append(gemm(f"aud.{li}.fc2", P, ff, d, li))
    ops.append(_ln_vec("aud.ln", depth, P, d))
    for i, (k, n) in enumerate(a.lin):
        ops.append(gemm(f"aud.proj.linear{i}", P, k, n, depth))
        if i in gelu_at:
            # proj1 is aud.proj.gelu. The 2B connector w1 is the only other site.
            name = "aud.proj.gelu" if i == 0 else "aud.conn.gelu"
            ops.append(_gelu_erf_vec(name, depth, P * n, act))
    return ops


def _require_erf_gelu(label: str, value: object) -> None:
    """Shared encoder MLP activation. ``gelu`` is the erf form. Anything else is refused."""
    if value != "gelu":
        raise ValueError(f"{label} activation_function is not gelu")


def _keep_erf_gelu(label: str, block: dict) -> None:
    """Step, Ming and Kimi hardcode the erf form. A missing field stays. A different field is refused."""
    if isinstance(block, dict) and "activation_function" in block:
        _require_erf_gelu(label, block.get("activation_function"))


def _whisper_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """Whisper-family encoder GEMMs plus the activations the published forward actually runs.

    conv1 / conv2 use ``nn.functional.gelu`` (default ``approximate='none'``, the erf form).
    Each block is pre-LN attention, then pre-LN MLP. The MLP activation is ``ACT2FN['gelu']``
    (transformers Whisper / Qwen2-Audio / Qwen2.5-Omni) or ``nn.GELU()`` / the same ``F.gelu``
    (Step-Audio, Ming, and the Kimi copy of WhisperModel.encoder). Both are the erf form,
    not ``gelu_pytorch_tanh``. The encoder LayerNorm sits after the blocks. Qwen2-Audio and
    Qwen2.5-Omni run it after the stride-2 pool. Step-Audio's ``after_norm`` does too, then
    the adaptor is ``F.gelu(conv)``, Linear, ``nn.ReLU``, Linear. Kimi LayerNorm is on the
    encoder positions, before the 4-wide stack; ``VQAdaptor`` is Linear, ``nn.SiLU``,
    ``nn.Dropout(0)``, Linear, LayerNorm. Ming's projector is Conv1d, ``nn.GELU()``, Linear.
    GELU, SiLU and ReLU do not upcast: ``a_fmt`` is ``vec_fmt`` (the vision-role dtype).
    LayerNorm reduces in fp32. The shared GEMM tail is not reduced. A requested GEMM
    activation format does not change these vector dtypes. GEMM ``w_fmt`` stays bf16.
    Eval dropout is not an op. The fp16 clamp, the Omni ``* long`` mask, and Ming
    ``F.normalize`` are not ops here.
    """
    d, dh = a.dims[0], a.dims[0] // a.heads
    ff = a.mlp * d
    if ff != int(ff):
        raise ValueError("whisper ffn width is not an integer")
    ff = int(ff)
    act = a.vec_fmt or a.w_fmt
    if act not in ("fp16", "bf16", "fp32"):
        raise ValueError("whisper activation dtype is not a released float format")
    ops: list[Op] = []

    def gemm(name: str, m: int, k: int, n: int, li: int = 0) -> Op:
        return Op(name, "gemm", li, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=16.0, w_fmt="bf16", a_fmt=af,
                  act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    groups: dict[int, int] = {}
    for n in a.chunks(seconds):
        groups[n] = groups.get(n, 0) + clips
    for gi, (n, cnt) in enumerate(sorted(groups.items(), reverse=True)):
        t, o = a.chunk_len(n)
        p = f"aud.c{gi}." if gi else "aud."
        rows, ch = cnt * t, cnt * a.heads
        mel = cnt * n
        # k3 s1 p1 keeps the mel length. k3 s2 p1 is ``t`` positions.
        ops.append(gemm(p + "conv1", mel, 3 * a.n_mels, d))
        ops.append(_gelu_erf_vec(p + "conv1.gelu", 0, mel * d, act))
        ops.append(gemm(p + "conv2", rows, 3 * d, d))
        ops.append(_gelu_erf_vec(p + "conv2.gelu", 0, rows * d, act))
        depth = a.depths[0]
        for li in range(depth):
            # Attention pre-norm, then the MLP pre-norm. GELU reads the FFN width.
            ops.append(_ln_vec(f"{p}{li}.attn_ln", li, rows, d))
            ops.append(gemm(f"{p}{li}.qkv", rows, d, 3 * d, li))
            ops += [Op(f"{p}{li}.qk", "attn", li, m=t, k=dh, n=t, count=ch, causal=1.0, act_bytes=ch * t * dh * ab,
                       stream=True, orient=True),
                    Op(f"{p}{li}.pv", "attn", li, m=t, k=t, n=dh, count=ch, causal=1.0, act_bytes=ch * t * dh * ab,
                       orient=True),
                    Op(f"{p}{li}.softmax", "vector", li, vec=ch * t * t * 5.0)]
            ops.append(gemm(f"{p}{li}.o", rows, d, d, li))
            ops.append(_ln_vec(f"{p}{li}.mlp_ln", li, rows, d))
            ops.append(gemm(f"{p}{li}.fc1", rows, d, ff, li))
            ops.append(_gelu_erf_vec(f"{p}{li}.gelu", li, rows * ff, act))
            ops.append(gemm(f"{p}{li}.fc2", rows, ff, d, li))
        if a.lin and a.ds:
            # Step-Audio-2: pool, after_norm, adaptor conv + F.gelu, Linear, nn.ReLU, Linear.
            if len(a.lin) != 2:
                raise ValueError("step audio adaptor is not conv, gelu, linear, relu, linear")
            pooled = cnt * (t // a.pool)
            ops.append(Op(f"{p}pool", "vector", depth, vec=rows * d * 1.0))
            ops.append(_ln_vec(f"{p}ln", depth, pooled, d))
            (kk, _, _), = a.ds
            ops.append(gemm(f"{p}proj.conv0", cnt * o, kk * d, d, depth))
            ops.append(_gelu_erf_vec(f"{p}proj.gelu", depth, cnt * o * d, act))
            for i, (k, n_) in enumerate(a.lin):
                ops.append(gemm(f"{p}proj.linear{i}", cnt * o, k, n_, depth))
                if i == 0:
                    # nn.ReLU is one clip. The width is 2048, not d.
                    ops.append(_relu_vec(f"{p}proj.relu", depth, cnt * o * n_, act))
            continue
        if a.lin is not None:
            # Empty ``lin`` is the bare Whisper encoder (final LayerNorm on ``t``).
            # Qwen2-Audio / Qwen2.5-Omni: LayerNorm after the pool, then a Linear with no GELU.
            # Kimi: encoder LayerNorm before the stack. VQAdaptor is Linear, SiLU, Linear, LayerNorm.
            if a.stack:
                if len(a.lin) != 2:
                    raise ValueError("kimi adaptor is not linear, silu, linear, layernorm")
                (k0, n0), (k1, n1) = a.lin
                if k0 != a.stack * d or n0 != n1 or k1 != n0:
                    raise ValueError("kimi adaptor widths do not match the published linears")
                ops.append(_ln_vec(f"{p}ln", depth, rows, d))
            rp = rows // (a.stack or a.pool)
            if a.pool > 1:
                ops.append(Op(f"{p}pool", "vector", depth, vec=rows * d * 1.0))
            if not a.stack:
                ops.append(_ln_vec(f"{p}ln", depth, rp, d))
            for i, (k, n_) in enumerate(a.lin):
                ops.append(gemm(f"{p}proj.linear{i}", rp, k, n_, depth))
                if a.stack and i == 0:
                    ops.append(_silu_vec(f"{p}proj.silu", depth, rp * n_, act))
                if a.stack and i == len(a.lin) - 1:
                    ops.append(_ln_vec(f"{p}proj.ln", depth, rp, n_))
            continue
        # Ming: ln_post, then each downsampler Conv1d + nn.GELU(), then Linear. No ReLU.
        ops.append(_ln_vec(f"{p}ln", depth, rows, d))
        din, m_ = d, rows
        for i, (k, st, pd) in enumerate(a.ds):
            m_ = cnt * o if i == len(a.ds) - 1 else m_
            ops.append(gemm(f"{p}proj.conv{i}", m_, k * din, a.out, depth))
            gelu_name = f"{p}proj.gelu" if len(a.ds) == 1 else f"{p}proj.gelu{i}"
            ops.append(_gelu_erf_vec(gelu_name, depth, m_ * a.out, act))
            din = a.out
        ops.append(gemm(f"{p}proj.linear", cnt * o, a.out, a.out, depth))
    return ops


def _g4conf_ops(a: AudioSpec, seconds: float, clips: int, af: str) -> list[Op]:
    """Conformer GEMMs at the padded tower length. Attention is on chunk-padded blocks.

    Gemma4AudioSubSampleConvProjectionLayer (transformers ``modular_gemma4.py``,
    sha ``2df500f``) multiplies by the feature mask, then Conv2d, then
    ``nn.ReLU(LayerNorm(..., bias=False))``. Gemma4AudioFeedForward is RMSNorm,
    ``ACT2FN['silu']``, RMSNorm, then ``*= residual_weight`` (0.5) and the residual
    add. Gemma4AudioLightConv1d is RMSNorm, ``F.glu`` on the kept half, the causal
    depthwise conv, RMSNorm, SiLU, then a full residual add. The block adds three
    more with-scale RMSNorms (pre-attn, post-attn, out). ``Gemma4RMSNorm`` is
    ``Gemma3nRMSNorm``: fp32, ``pow(2).mean``, one ``pow(-0.5)``, times the weight.
    ``embed_audio`` uses ``with_scale=False`` and is not that op. The logit softcap,
    ``softplus(per_dim_scale)``, the clamps, and the sinusoid table are not ops
    here. The shared GEMM tail is not reduced. A requested GEMM activation format
    does not change these vector dtypes.
    """
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    tower, _prompt = a._g4_audio_lens(seconds)
    if tower <= 0 or clips <= 0:
        return []
    h, c0, c1 = a.dims
    hd = h // a.heads
    context = a.chunk + (a.ctx_l - 1) + a.ctx_r
    rel = context // 2 + 1
    q_pad = -(-tower // a.chunk) * a.chunk
    wf, wb = a.w_fmt, a.w_bits
    n = min(_G4_AUDIO_MAX_SAMPLES, max(0, round(seconds * a.rate)))
    padded = ((n + _G4_AUDIO_PAD - 1) // _G4_AUDIO_PAD) * _G4_AUDIO_PAD
    mel = a._g4_mel(padded)

    def conv_out(length: int) -> int:
        return (length + 2 - 3) // 2 + 1

    t0, f0 = conv_out(mel), conv_out(a.n_mels)
    f1 = conv_out(f0)
    out_d = a.lin[0][1]

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0, count: int = 1) -> Op:
        return Op(name, "gemm", layer, m=m, k=k, n=n, count=count, role="vision", w_params=k * n * count,
                  w_bits=wb, w_fmt=wf, a_fmt=af, act_bytes=m * (k + n) * count * ab, stream=True,
                  vec=m * n * count * VEC_PER_OUT)

    t = tower * clips
    # Released weights are bf16. SiLU, GLU, ReLU and the 0.5 scale do not upcast.
    # The first mask multiply reads the feature extractor's float32 mel, before the
    # conv casts to the weight dtype. The second multiply reads the bf16 ReLU output.
    act = a.w_fmt

    def one(name: str, n: int, fmt: str) -> Op:
        return Op(name, "vector", 0, vec=n, a_fmt=fmt)

    # Channel 1 on the way in. nn.ReLU is one clip (max(x, 0)), not ReLU6.
    # LayerNorm has no bias. The reduction is one row per time-frequency cell.
    ops = [
        one("aud.sub.mask0", clips * mel * a.n_mels, "fp32"),
        gemm("aud.conv0", t0 * f0 * clips, 9, c0),
        _ln_vec("aud.sub.ln0", 0, t0 * f0 * clips, c0),
        one("aud.sub.relu0", t0 * f0 * clips * c0, act),
        one("aud.sub.mask1", t0 * f0 * clips * c0, act),
        gemm("aud.conv1", tower * f1 * clips, 9 * c0, c1),
        _ln_vec("aud.sub.ln1", 0, tower * f1 * clips, c1),
        one("aud.sub.relu1", tower * f1 * clips * c1, act),
        gemm("aud.sub_proj", t, f1 * c1, h),
    ]
    for li in range(a.depths[0]):
        # Feed-forward: pre RMSNorm, Linear, SiLU, Linear, post RMSNorm, *= 0.5, add.
        # SiLU width is 4h. The scale width is h.
        ops.append(_rms_vec(f"aud.{li}.ff1_rms", li, t, h))
        ops.append(gemm(f"aud.{li}.ff1_up", t, h, 4 * h, li))
        ops.append(_silu_vec(f"aud.{li}.ff1_silu", li, t * 4 * h, act))
        ops.append(gemm(f"aud.{li}.ff1_down", t, 4 * h, h, li))
        ops.append(_rms_vec(f"aud.{li}.ff1_post", li, t, h))
        ops.append(_half_vec(f"aud.{li}.ff1_half", li, t * h, act))
        ops.append(_rms_vec(f"aud.{li}.attn_rms", li, t, h))
        ops += [gemm(f"aud.{li}.q", t, h, h, li), gemm(f"aud.{li}.k", t, h, h, li),
                gemm(f"aud.{li}.v", t, h, h, li)]
        # relative_k_proj is one Linear on the batch-1 sinusoid (context//2+1, h). Gemma4AudioAttention
        # then does queries_flat @ relative_keys: every padded query dots those rel keys (matrix_bd).
        # rel_shift reindexes that score to context and does not matmul again. The projection does not
        # grow with the clip batch; the query-key matmul does. The sin/cos table is not an op.
        ops.append(gemm(f"aud.{li}.rel", rel, h, h, li))
        ops += [Op(f"aud.{li}.qk", "attn", li, m=q_pad, k=hd, n=context, count=a.heads * clips, causal=1.0,
                   act_bytes=a.heads * clips * q_pad * hd * ab, stream=True, orient=True),
                Op(f"aud.{li}.relqk", "attn", li, m=q_pad, k=hd, n=rel, count=a.heads * clips, causal=1.0,
                   act_bytes=a.heads * clips * q_pad * hd * ab, stream=True, orient=True),
                Op(f"aud.{li}.pv", "attn", li, m=q_pad, k=context, n=hd, count=a.heads * clips, causal=1.0,
                   act_bytes=a.heads * clips * q_pad * hd * ab, orient=True),
                Op(f"aud.{li}.softmax", "vector", li, vec=a.heads * clips * q_pad * context * 5.0)]
        # post linear, then norm_post_attn. The following residual add is a full add, not * 0.5.
        ops.append(gemm(f"aud.{li}.post", t, h, h, li))
        ops.append(_rms_vec(f"aud.{li}.attn_out", li, t, h))
        # GLU keeps h, not the 2h pointwise width. SiLU reads the depthwise output.
        ops.append(_rms_vec(f"aud.{li}.lconv_rms", li, t, h))
        ops.append(gemm(f"aud.{li}.lconv_start", t, h, 2 * h, li))
        ops.append(_glu_vec(f"aud.{li}.glu", li, t * h, act))
        ops.append(Op(f"aud.{li}.lconv", "vector", li, vec=t * h * a.kernel, act_bytes=2 * t * h * ab))
        ops.append(_rms_vec(f"aud.{li}.lconv_norm", li, t, h))
        ops.append(_silu_vec(f"aud.{li}.lconv_silu", li, t * h, act))
        ops.append(gemm(f"aud.{li}.lconv_end", t, h, h, li))
        ops.append(_rms_vec(f"aud.{li}.ff2_rms", li, t, h))
        ops.append(gemm(f"aud.{li}.ff2_up", t, h, 4 * h, li))
        ops.append(_silu_vec(f"aud.{li}.ff2_silu", li, t * 4 * h, act))
        ops.append(gemm(f"aud.{li}.ff2_down", t, 4 * h, h, li))
        ops.append(_rms_vec(f"aud.{li}.ff2_post", li, t, h))
        ops.append(_half_vec(f"aud.{li}.ff2_half", li, t * h, act))
        ops.append(_rms_vec(f"aud.{li}.out_rms", li, t, h))
    ops.append(gemm("aud.out", t, h, out_d, a.depths[0]))
    # embed_audio sees the tower rows. Its pre-norm has with_scale=False, so it is
    # not the weight-multiplying RMSNorm. The prompt mask is applied after this Linear.
    ops.append(gemm("aud.embed", t, out_d, a.out, a.depths[0]))
    return ops


def _silu_vec(name: str, layer: int, n: int, af: str) -> Op:
    """x * sigmoid(x). sigmoid is exp, one add and a reciprocal. The class does not upcast."""
    return Op(name, "vector", layer, vec=n * 4, sfu=n * 2, a_fmt=af)


def _glu_vec(name: str, layer: int, n: int, af: str) -> Op:
    """a * sigmoid(b) on the kept half. The split is a view. Same sigmoid count as SiLU."""
    return Op(name, "vector", layer, vec=n * 4, sfu=n * 2, a_fmt=af)


def _gelu_erf_vec(name: str, layer: int, n: int, af: str) -> Op:
    """0.5 * x * (1 + erf(x / sqrt(2))). Four arithmetic ops and one erf."""
    return Op(name, "vector", layer, vec=n * 5, sfu=float(n), a_fmt=af)


def _ln_vec(name: str, layer: int, rows: int, width: int) -> Op:
    """PyTorch layer_norm reduces in fp32. One rsqrt per row. The affine is already in the parameter total."""
    return Op(name, "vector", layer, vec=rows * width * 4, sfu=float(rows), a_fmt="fp32")


def _relu_vec(name: str, layer: int, n: int, af: str) -> Op:
    """``nn.ReLU`` / ``ACT2FN["relu"]`` is ``max(x, 0)``. One clip, no transcendental.

    ``nn.ReLU6`` is two clips (``vec = n * 2``). This is not that op. The class does not upcast.
    """
    return Op(name, "vector", layer, vec=n, a_fmt=af)


def _rms_vec(name: str, layer: int, rows: int, width: int) -> Op:
    """Gemma3nRMSNorm with a weight, the same count as LongCat RMSNorm.

    The forward upcasts to fp32, does ``pow(2).mean``, one ``pow(-0.5)`` (the rsqrt),
    and multiplies the weight. ``with_scale=False`` skips that multiply and is not this op.
    """
    return Op(name, "vector", layer, vec=rows * width * 4, sfu=float(rows), a_fmt="fp32")


def _half_vec(name: str, layer: int, n: int, af: str) -> Op:
    """residual + 0.5 * branch. One multiply and one add. No transcendental."""
    return Op(name, "vector", layer, vec=n * 2, a_fmt=af)


def _bn_vec(name: str, layer: int, elems: int, channels: int, af: str) -> Op:
    """Eval BatchNorm: (x - mean) * rsqrt(var + eps) * weight + bias.

    rsqrt runs once per channel. Running stats do not depend on the batch, so the
    transcendental does not grow with the clip count. The four arithmetic ops do.
    """
    return Op(name, "vector", layer, vec=elems * 4, sfu=float(channels), a_fmt=af)


def _gspeech_ops(a: AudioSpec, seconds: float, clips: int, af: str) -> list[Op]:
    """Conformer GEMMs at the unpadded frame count. Shaw Q/K and SDPA see the padded blocks.

    The Q-Former attends inside one window. Cross-attention K/V see the padded window,
    including a partial last window (the projector passes no encoder mask).

    GraniteSpeechConformerBlock (and the Plus copy) runs LayerNorm, nn.SiLU, nn.GLU,
    BatchNorm1d and ``0.5 * ff + residual`` on the unpadded frames. Attention's
    pre-norm is before the context pad. The Q-Former query path runs one erf-GELU
    per layer (``hidden_act`` is ``gelu``) and the LayerNorms in the query output
    modules. ``nn.Dropout`` is identity in eval, including when the config probability
    is not zero, so it is not an op. The shared GEMM tail is not reduced.
    """
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    enc, pad, _prompt = a._gs_lens(seconds)
    if enc <= 0 or clips <= 0:
        return []
    h = a.dims[0]
    hd = h // a.heads
    qhd = a.gs_qh // a.gs_qheads
    nblocks = -(-enc // a.win)
    blocks = pad // a.chunk
    wf, wb = a.w_fmt, a.w_bits
    t = enc * clips
    tp = pad * clips
    nq = nblocks * a.gs_q * clips
    nk = nblocks * a.win * clips

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op(name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=wb, w_fmt=wf,
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    def attn(name: str, m: int, k: int, n: int, count: int, layer: int) -> Op:
        return Op(name, "attn", layer, m=m, k=k, n=n, count=count, causal=1.0,
                  act_bytes=count * m * k * ab, stream=True, orient=True)

    # Released conformer weights are bf16. nn.SiLU / nn.GLU / the 0.5 scale do not upcast.
    # BatchNorm affine is bf16 on these checkpoints (no fp32 parameter role).
    act = "bf16"
    inner = a.gs_inner
    ops = [gemm("aud.in", t, a.n_mels, h)]
    for li in range(a.depths[0]):
        # GraniteSpeechConformerFeedForward: LayerNorm, Linear, SiLU, Linear.
        # The block then does 0.5 * ff + residual. Dropout is eval identity.
        ops.append(_ln_vec(f"aud.{li}.ff1_ln", li, t, h))
        ops.append(gemm(f"aud.{li}.ff1_up", t, h, a.gs_ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff1_silu", li, t * a.gs_ff, act))
        ops.append(gemm(f"aud.{li}.ff1_down", t, a.gs_ff, h, li))
        ops.append(_half_vec(f"aud.{li}.ff1_half", li, t * h, act))
        # pre_norm runs before the context pad, so the reduction is the unpadded length.
        ops.append(_ln_vec(f"aud.{li}.attn_ln", li, t, h))
        ops += [gemm(f"aud.{li}.q", tp, h, h, li), gemm(f"aud.{li}.kv", tp, h, 2 * h, li),
                gemm(f"aud.{li}.o", t, h, h, li)]
        # rel_pos_emb is a gather of the (context, context) distance buffer. The einsum
        # with the queries is the matmul. SDPA then runs QK and PV on the same padded block.
        cnt = a.heads * blocks * clips
        ops += [attn(f"aud.{li}.shaw", a.chunk, hd, a.chunk, cnt, li),
                attn(f"aud.{li}.qk", a.chunk, hd, a.chunk, cnt, li),
                attn(f"aud.{li}.pv", a.chunk, a.chunk, hd, cnt, li),
                Op(f"aud.{li}.softmax", "vector", li, vec=cnt * a.chunk * a.chunk * 5.0)]
        # Conv module: LayerNorm, pointwise to 2 * inner, GLU (keeps inner), depthwise,
        # BatchNorm, SiLU, pointwise back. Stride 1 keeps the unpadded length.
        ops.append(_ln_vec(f"aud.{li}.conv_ln", li, t, h))
        ops.append(gemm(f"aud.{li}.conv_up", t, h, 2 * inner, li))
        ops.append(_glu_vec(f"aud.{li}.glu", li, t * inner, act))
        ops.append(Op(f"aud.{li}.dconv", "vector", li, vec=t * inner * a.kernel,
                      act_bytes=2 * t * inner * ab))
        ops.append(_bn_vec(f"aud.{li}.bn", li, t * inner, inner, act))
        ops.append(_silu_vec(f"aud.{li}.conv_silu", li, t * inner, act))
        ops.append(gemm(f"aud.{li}.conv_down", t, inner, h, li))
        ops.append(_ln_vec(f"aud.{li}.ff2_ln", li, t, h))
        ops.append(gemm(f"aud.{li}.ff2_up", t, h, a.gs_ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff2_silu", li, t * a.gs_ff, act))
        ops.append(gemm(f"aud.{li}.ff2_down", t, a.gs_ff, h, li))
        ops.append(_half_vec(f"aud.{li}.ff2_half", li, t * h, act))
        ops.append(_ln_vec(f"aud.{li}.out_ln", li, t, h))
    mid_l = a.depths[0] // 2
    if mid_l >= 1:
        ops += [gemm("aud.mid.out", t, h, a.gs_mid, mid_l),
                gemm("aud.mid.back", t, a.gs_mid, h, mid_l),
                Op("aud.mid.softmax", "vector", mid_l, vec=t * a.gs_mid * 5.0)]
    # Blip2QFormerModel casts queries to the LayerNorm weight dtype (bf16 here) and
    # normalizes them once. use_qformer_text_input is false, so only the query FFN runs.
    ops.append(_ln_vec("aud.qf.ln", 0, nq, a.gs_qh))
    for li in range(a.gs_ql):
        ops += [gemm(f"aud.qf.{li}.q", nq, a.gs_qh, a.gs_qh, li),
                gemm(f"aud.qf.{li}.k", nq, a.gs_qh, a.gs_qh, li),
                gemm(f"aud.qf.{li}.v", nq, a.gs_qh, a.gs_qh, li),
                gemm(f"aud.qf.{li}.o", nq, a.gs_qh, a.gs_qh, li)]
        ops.append(_ln_vec(f"aud.qf.{li}.attn_ln", li, nq, a.gs_qh))
        cnt = a.gs_qheads * nblocks * clips
        ops += [attn(f"aud.qf.{li}.qk", a.gs_q, qhd, a.gs_q, cnt, li),
                attn(f"aud.qf.{li}.pv", a.gs_q, a.gs_q, qhd, cnt, li),
                Op(f"aud.qf.{li}.softmax", "vector", li, vec=cnt * a.gs_q * a.gs_q * 5.0)]
        if li % a.gs_freq == 0:
            ops += [gemm(f"aud.qf.{li}.cq", nq, a.gs_qh, a.gs_qh, li),
                    gemm(f"aud.qf.{li}.ck", nk, a.gs_cross, a.gs_qh, li),
                    gemm(f"aud.qf.{li}.cv", nk, a.gs_cross, a.gs_qh, li),
                    gemm(f"aud.qf.{li}.co", nq, a.gs_qh, a.gs_qh, li)]
            ops.append(_ln_vec(f"aud.qf.{li}.cross_ln", li, nq, a.gs_qh))
            ops += [attn(f"aud.qf.{li}.cqk", a.gs_q, qhd, a.win, cnt, li),
                    attn(f"aud.qf.{li}.cpv", a.gs_q, a.win, qhd, cnt, li),
                    Op(f"aud.qf.{li}.csoftmax", "vector", li, vec=cnt * a.gs_q * a.win * 5.0)]
        # ACT2FN["gelu"] is the erf form, on the query rows and the intermediate width.
        ops.append(gemm(f"aud.qf.{li}.fc1", nq, a.gs_qh, a.gs_qff, li))
        ops.append(_gelu_erf_vec(f"aud.qf.{li}.gelu", li, nq * a.gs_qff, act))
        ops.append(gemm(f"aud.qf.{li}.fc2", nq, a.gs_qff, a.gs_qh, li))
        ops.append(_ln_vec(f"aud.qf.{li}.out_ln", li, nq, a.gs_qh))
    ops.append(gemm("aud.proj", nq, a.gs_qh, a.out))
    return ops


def _gsctc_ops(a: AudioSpec, seconds: float, clips: int, af: str) -> list[Op]:
    """Shaw conformer with stride-2 subsampling. No Q-Former and no LM prompt.

    q / k / v and the Shaw matmul see each block padded to context_size. o_proj, both
    FFNs, and the stride-1 conv see the unpadded length. A subsample layer's depthwise
    conv and pointwise_lin2 see the conv output length; ff2 and the residual see T // 2.
    Mid CTC (out, then out_mid) runs once after layer num_layers // 2. The final CTC
    head is the same tensor as out and is read again.

    GraniteSpeech5EncoderBlock runs five LayerNorms, ``ACT2FN['silu']`` on each FFN,
    ``F.glu`` on the pointwise output, BatchNorm then ``F.silu`` on the depthwise
    output, and ``residual + 0.5 * ff``. A subsampling block pools the residual with
    ``unfold(1, 2, 2).mean`` before adding the conv tail. The masked_fill before the
    depthwise conv is not a separate op. activation_dropout and attention_dropout are
    0. The BatchNorm affine is fp32 and ``F.silu`` reads that output with no cast
    back, so the conv SiLU stays fp32. FFN SiLU and GLU read bf16 linears and stay
    bf16. The shared GEMM tail is not reduced.
    """
    from .dtypes import fmt as _fmt
    frames = a._gsctc_frames(seconds)
    if not frames or clips <= 0:
        return []
    ab = _fmt(af).bytes
    h = a.dims[0]
    hd = h // a.heads
    wf, wb = a.w_fmt, a.w_bits
    sub = set(a.gs_sub)
    n_layers = a.depths[0]

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op(name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=wb, w_fmt=wf,
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    def attn(name: str, m: int, k: int, n: int, count: int, layer: int) -> Op:
        return Op(name, "attn", layer, m=m, k=k, n=n, count=count, causal=1.0,
                  act_bytes=count * m * k * ab, stream=True, orient=True)

    # FFN SiLU and GLU sit on bf16 linears and do not upcast. BatchNorm affine is fp32.
    act = "bf16"
    inner = a.gs_inner
    ops = [gemm("aud.in", frames[0] * clips, a.n_mels, h)]
    for li in range(n_layers):
        t = frames[li]
        blocks = -(-t // a.chunk) if t else 0
        pad = blocks * a.chunk
        tp = pad * clips
        tm = t * clips
        ops.append(_ln_vec(f"aud.{li}.ff1_ln", li, tm, h))
        ops.append(gemm(f"aud.{li}.ff1_up", tm, h, a.gs_ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff1_silu", li, tm * a.gs_ff, act))
        ops.append(gemm(f"aud.{li}.ff1_down", tm, a.gs_ff, h, li))
        ops.append(_half_vec(f"aud.{li}.ff1_half", li, tm * h, act))
        ops.append(_ln_vec(f"aud.{li}.attn_ln", li, tm, h))
        ops += [gemm(f"aud.{li}.q", tp, h, h, li), gemm(f"aud.{li}.k", tp, h, h, li),
                gemm(f"aud.{li}.v", tp, h, h, li), gemm(f"aud.{li}.o", tm, h, h, li)]
        cnt = a.heads * blocks * clips
        if cnt:
            ops += [attn(f"aud.{li}.shaw", a.chunk, hd, a.chunk, cnt, li),
                    attn(f"aud.{li}.qk", a.chunk, hd, a.chunk, cnt, li),
                    attn(f"aud.{li}.pv", a.chunk, a.chunk, hd, cnt, li),
                    Op(f"aud.{li}.softmax", "vector", li, vec=cnt * a.chunk * a.chunk * 5.0)]
        stride = 2 if li in sub else 1
        cout = _conv_same(t, a.kernel, stride) if t else 0
        kept = t // 2 if li in sub else t
        cm = cout * clips
        km = kept * clips
        # GLU keeps the pre-conv length. BatchNorm and conv SiLU see the conv output.
        ops.append(_ln_vec(f"aud.{li}.conv_ln", li, tm, h))
        ops.append(gemm(f"aud.{li}.conv_up", tm, h, 2 * inner, li))
        ops.append(_glu_vec(f"aud.{li}.glu", li, tm * inner, act))
        ops.append(Op(f"aud.{li}.dconv", "vector", li, vec=cm * inner * a.kernel,
                      act_bytes=2 * cm * inner * ab))
        ops.append(_bn_vec(f"aud.{li}.bn", li, cm * inner, inner, "fp32"))
        ops.append(_silu_vec(f"aud.{li}.conv_silu", li, cm * inner, "fp32"))
        ops.append(gemm(f"aud.{li}.conv_down", cm, inner, h, li))
        if li in sub:
            # unfold(1, 2, 2).mean drops a trailing odd frame. Add and multiply by 1/2.
            ops.append(Op(f"aud.{li}.pool", "vector", li, vec=km * h * 2, a_fmt=act))
        ops.append(_ln_vec(f"aud.{li}.ff2_ln", li, km, h))
        ops.append(gemm(f"aud.{li}.ff2_up", km, h, a.gs_ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff2_silu", li, km * a.gs_ff, act))
        ops.append(gemm(f"aud.{li}.ff2_down", km, a.gs_ff, h, li))
        ops.append(_half_vec(f"aud.{li}.ff2_half", li, km * h, act))
        ops.append(_ln_vec(f"aud.{li}.out_ln", li, km, h))
        if li + 1 == n_layers // 2:
            mid = frames[li + 1] * clips
            ops += [gemm("aud.mid.out", mid, h, a.gs_mid, li + 1),
                    gemm("aud.mid.back", mid, a.gs_mid, h, li + 1),
                    Op("aud.mid.softmax", "vector", li + 1, vec=mid * a.gs_mid * 5.0)]
    final = frames[-1] * clips
    ops.append(gemm("aud.ctc", final, h, a.gs_mid, n_layers))
    return ops


def _d1_audio_ops(a: AudioSpec, seconds: float, clips: int) -> list[Op]:
    """FastConformer prefix. Activations stay fp32: the released class does not autocast.

    linear_pos reads the 2T-1 sinusoid once per batch, not once per clip. The relative
    matmul is T by 2T-1 and is then sliced; QK, PV, and softmax see the T by T slice.
    Depthwise convs are vector taps.

    ``ConformerLayer`` (LiquidAI/d1-omni-600M ``audio.py``) runs LayerNorm, ``F.silu``,
    ``x + ff * 0.5``, ``F.glu`` on the kept half and eval BatchNorm. Those use the same
    vector recipes as Granite Speech. ``F.glu``'s split is a view. BatchNorm's rsqrt is
    once per channel. The adapter and the residual run ``F.gelu`` with the default
    ``approximate='none'`` (erf). The class does not cast, so a requested GEMM format
    does not change these ops or the fp32 GEMMs. The shared GEMM tail is not reduced.
    ``ConvSubsampling`` runs ``nn.ReLU`` three times (one clip; ReLU6 is two) and
    multiplies by ``_time_mask`` before every subsampling module and once more before
    the linear. The depthwise ``masked_fill`` is a where, not that multiply, so it
    stays in the tail with the pad, the plain residual adds and the sinusoid table.
    """
    mel, t = _d1_audio_lens(seconds)
    if t <= 0 or clips <= 0:
        return []
    from .dtypes import fmt as _fmt
    af = "fp32"
    ab = _fmt(af).bytes
    d = a.dims[0]
    hds = a.heads
    hd = d // hds
    ch = a.ch
    ff = a.gs_ff
    ker = a.kernel

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op(name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=32.0, w_fmt="fp32",
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    # ConvSubsampling.forward multiplies by _time_mask, then runs the module, and
    # updates lengths only after a stride-2 Conv2d. nn.ReLU(inplace=True) is
    # max(x, 0): one clip, no transcendental. ReLU6 is two clips (vec = n * 2).
    # The mask is one multiply in the activation dtype. Building it (arange, compare,
    # cast) is not an op. The depthwise masked_fill stays in the tail.
    def one(name: str, elems: int) -> Op:
        return Op(name, "vector", 0, vec=elems, a_fmt=af)

    chans = 1
    length = mel
    freq = a.n_mels
    ops: list[Op] = []
    stages = (
        ("conv", "aud.sub.0"),
        ("relu", "aud.sub.relu0"),
        ("dw", "aud.sub.dw0"),
        ("pw", "aud.sub.pw0"),
        ("relu", "aud.sub.relu1"),
        ("dw", "aud.sub.dw1"),
        ("pw", "aud.sub.pw1"),
        ("relu", "aud.sub.relu2"),
    )
    mask_i = 0
    for kind, name in stages:
        ops.append(one(f"aud.sub.mask{mask_i}", clips * chans * length * freq))
        mask_i += 1
        if kind == "conv":
            length, freq = _d1_conv_len(length), _d1_conv_len(freq)
            ops.append(gemm(name, length * freq * clips, 9, ch))
            chans = ch
        elif kind == "dw":
            length, freq = _d1_conv_len(length), _d1_conv_len(freq)
            ops.append(Op(name, "vector", 0, vec=length * freq * clips * ch * 9,
                          act_bytes=2 * length * freq * clips * ch * ab))
        elif kind == "pw":
            ops.append(gemm(name, length * freq * clips, ch, ch))
        else:
            ops.append(one(name, clips * chans * length * freq))
    ops.append(one(f"aud.sub.mask{mask_i}", clips * ch * length * freq))
    ops.append(gemm("aud.sub.out", length * clips, ch * freq, d))
    # Released weights are fp32. F.silu, F.glu, the 0.5 scale, BatchNorm and F.gelu do not upcast.
    act = "fp32"
    pos = 2 * t - 1
    for li in range(a.depths[0]):
        tm = t * clips
        # FeedForward: LayerNorm, Linear, F.silu, Linear. The block then does x + ff * 0.5.
        ops.append(_ln_vec(f"aud.{li}.ff1_ln", li, tm, d))
        ops.append(gemm(f"aud.{li}.ff1_up", tm, d, ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff1_silu", li, tm * ff, act))
        ops.append(gemm(f"aud.{li}.ff1_down", tm, ff, d, li))
        ops.append(_half_vec(f"aud.{li}.ff1_half", li, tm * d, act))
        ops.append(_ln_vec(f"aud.{li}.attn_ln", li, tm, d))
        ops += [gemm(f"aud.{li}.q", tm, d, d, li), gemm(f"aud.{li}.k", tm, d, d, li),
                gemm(f"aud.{li}.v", tm, d, d, li), gemm(f"aud.{li}.o", tm, d, d, li)]
        # One position row per batch, shared by every clip of this length.
        ops.append(gemm(f"aud.{li}.pos", pos, d, d, li))
        cnt = hds * clips
        ops += [Op(f"aud.{li}.ac", "attn", li, m=t, k=hd, n=t, count=cnt, causal=1.0,
                   act_bytes=cnt * t * hd * ab, stream=True, orient=True),
                Op(f"aud.{li}.bd", "attn", li, m=t, k=hd, n=pos, count=cnt, causal=1.0,
                   act_bytes=cnt * t * hd * ab, stream=True, orient=True),
                Op(f"aud.{li}.pv", "attn", li, m=t, k=t, n=hd, count=cnt, causal=1.0,
                   act_bytes=cnt * t * hd * ab, orient=True),
                Op(f"aud.{li}.softmax", "vector", li, vec=cnt * t * t * 5.0)]
        # ConvModule: LayerNorm, pointwise to 2 * d, F.glu (keeps d), depthwise,
        # BatchNorm, F.silu, pointwise back. The pad and masked_fill are not ops.
        ops.append(_ln_vec(f"aud.{li}.conv_ln", li, tm, d))
        ops.append(gemm(f"aud.{li}.pw1", tm, d, 2 * d, li))
        ops.append(_glu_vec(f"aud.{li}.glu", li, tm * d, act))
        ops.append(Op(f"aud.{li}.dw", "vector", li, vec=tm * d * ker, act_bytes=2 * tm * d * ab))
        ops.append(_bn_vec(f"aud.{li}.bn", li, tm * d, d, act))
        ops.append(_silu_vec(f"aud.{li}.conv_silu", li, tm * d, act))
        ops.append(gemm(f"aud.{li}.pw2", tm, d, d, li))
        ops.append(_ln_vec(f"aud.{li}.ff2_ln", li, tm, d))
        ops.append(gemm(f"aud.{li}.ff2_up", tm, d, ff, li))
        ops.append(_silu_vec(f"aud.{li}.ff2_silu", li, tm * ff, act))
        ops.append(gemm(f"aud.{li}.ff2_down", tm, ff, d, li))
        ops.append(_half_vec(f"aud.{li}.ff2_half", li, tm * d, act))
        ops.append(_ln_vec(f"aud.{li}.out_ln", li, tm, d))
    rows = t * clips
    # Adapter: LayerNorm(d), Linear(d -> out), F.gelu, Linear(out -> out).
    ops.append(_ln_vec("aud.adapter.ln", 0, rows, d))
    ops.append(gemm("aud.adapter.1", rows, d, a.out))
    ops.append(_gelu_erf_vec("aud.adapter.gelu", 0, rows * a.out, act))
    ops.append(gemm("aud.adapter.2", rows, a.out, a.out))
    # Residual: LayerNorm(out), Linear(out -> width), F.gelu, Linear(width -> out), then x + branch.
    # The trailing add is a full residual, not the 0.5 scale, so it is not a half op.
    ops.append(_ln_vec("aud.residual.ln", 0, rows, a.out))
    ops.append(gemm("aud.residual.down", rows, a.out, a.gs_inner))
    ops.append(_gelu_erf_vec("aud.residual.gelu", 0, rows * a.gs_inner, act))
    ops.append(gemm("aud.residual.up", rows, a.gs_inner, a.out))
    return ops


def _lcafsmn_ops(a: AudioSpec, seconds: float, clips: int) -> list[Op]:
    """LongCatAudioEncoder plus the two-linear projector. GEMM weights and activations stay bf16.

    The output linear is executed and then multiplied by zero (``hidden + 0.0 * output.sum()``).
    Its GEMM is billed. Each FSMN layer runs two ``LayerNorm(proj)`` modules (memory, then the FFN).
    Those are billed as fp32 reductions. The released ``activation`` is ``relu6``: two clips, no
    transcendental, on the FFN hidden width. The encoder multiplies by an int32 mask before every
    layer and once more before the DNN; bf16 times int32 is fp32. The projector GELU is ``nn.GELU``
    (erf) on the text hidden width. ``dropout`` in the released config is 0, and ``emb_mult`` is the
    constant 1, so neither is a separate op.
    """
    from .dtypes import fmt as _fmt
    af = "bf16"
    ab = _fmt(af).bytes
    t = a.tokens(seconds) * clips
    if t <= 0:
        return []
    proj, hidden, inp = a.dims
    nlayer, n_right, ndnn = a.depths
    if ndnn < 1:
        raise ValueError("longcat audio ndnn is not positive")

    def gemm(name: str, k: int, n: int, layer: int = 0) -> Op:
        return Op(name, "gemm", layer, m=t, k=k, n=n, role="vision", w_params=k * n, w_bits=16.0,
                  w_fmt="bf16", a_fmt=af, act_bytes=t * (k + n) * ab, stream=True, vec=t * n * VEC_PER_OUT)

    # Released audio/config.json activation is relu6. nn.ReLU6 is min(max(x, 0), 6):
    # two clips, no transcendental, and the class does not upcast. The width is the
    # FFN hidden size. The pre-net, every FSMN FFN, and every DNN linear run it.
    def relu6(name: str, layer: int) -> Op:
        return Op(name, "vector", layer, vec=t * hidden * 2, a_fmt="bf16")

    # bf16 hidden states times the int32 mask. PyTorch promotes that product to fp32.
    # One multiply on the proj width. No rsqrt.
    def mask(name: str, layer: int) -> Op:
        return Op(name, "vector", layer, vec=t * proj, a_fmt="fp32")

    ops = [gemm("aud.pre", inp, hidden), relu6("aud.pre_relu", 0), gemm("aud.pre_down", hidden, proj)]
    for i in range(nlayer):
        right = a.ctx_r if i >= nlayer - n_right else 0
        ker = a.ctx_l + right + 1
        # LongCatAudioEncoder.forward multiplies by the mask, then the layer runs
        # LayerNorm(proj) before the depthwise conv. The FFN Sequential starts with
        # another LayerNorm(proj), then Linear, ReLU6, Linear. PyTorch layer_norm
        # reduces in fp32 and applies one rsqrt per frame. The width is proj, not
        # the FFN hidden size. Weight and bias are already in the encoder total.
        ops.append(mask(f"aud.{i}.mask", i))
        ops.append(Op(f"aud.{i}.mem_ln", "vector", i, vec=t * proj * 4, sfu=float(t), a_fmt="fp32"))
        ops.append(Op(f"aud.{i}.dw", "vector", i, vec=t * proj * ker, act_bytes=2 * t * proj * ab))
        ops.append(Op(f"aud.{i}.ffn_ln", "vector", i, vec=t * proj * 4, sfu=float(t), a_fmt="fp32"))
        ops.append(gemm(f"aud.{i}.ffn_up", proj, hidden, i))
        ops.append(relu6(f"aud.{i}.relu", i))
        ops.append(gemm(f"aud.{i}.ffn_down", hidden, proj, i))
    ops.append(mask("aud.dnn_mask", 0))
    ops.append(gemm("aud.dnn.0", proj, hidden))
    ops.append(relu6("aud.dnn.0.relu", 0))
    for j in range(1, ndnn):
        ops.append(gemm(f"aud.dnn.{j}", hidden, hidden))
        ops.append(relu6(f"aud.dnn.{j}.relu", 0))
    ops.append(gemm("aud.out", hidden, a.out))
    for j, (k, n) in enumerate(a.lin or ()):
        ops.append(gemm(f"aud.proj.{j}", k, n))
        if j == 0:
            # nn.GELU() default is the erf form, not the tanh approximation.
            # 0.5 * x * (1 + erf(x / sqrt(2))): four arithmetic ops and one erf.
            # The class does not upcast. The width is the text hidden size.
            ops.append(Op("aud.proj_gelu", "vector", 0, vec=t * n * 5, sfu=float(t * n), a_fmt="bf16"))
    return ops


def audio_ops(a: AudioSpec, seconds: float, clips: int, af: str = "bf16") -> list[Op]:
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    if a.kind == "lcafsmn":
        return _lcafsmn_ops(a, seconds, clips)
    if a.kind == "gspeech":
        return _gspeech_ops(a, seconds, clips, af)
    if a.kind == "gsctc":
        return _gsctc_ops(a, seconds, clips, af)
    if a.kind == "d1fc":
        return _d1_audio_ops(a, seconds, clips)
    if a.kind == "g4conf":
        return _g4conf_ops(a, seconds, clips, af)
    if a.kind == "g4lin":
        # Gemma4UnifiedMultimodalEmbedder: one Linear(audio_embed_dim → text), no bias. RMSNorm has no scale.
        t = a.tokens(seconds) * clips
        k, n = a.dims[0], a.out
        return [Op("aud.proj", "gemm", 0, m=t, k=k, n=n, role="vision", w_params=k * n, w_bits=a.w_bits,
                   w_fmt=a.w_fmt, a_fmt=af, act_bytes=t * (k + n) * ab, stream=True, vec=t * n * VEC_PER_OUT)]
    if a.kind == "whisper":
        return _whisper_ops(a, seconds, clips, ab, af)
    if a.kind == "aut":
        return _aut_ops(a, seconds, clips, ab, af)
    if a.kind == "sanm":
        return _sanm_ops(a, seconds, clips, ab, af)
    if a.kind == "w2v":
        return _w2v_ops(a, seconds, clips, ab, af)
    t = a.tokens(seconds) * a.samples_per_token * clips       # padded samples (all clips)
    ops: list[Op] = []
    din = 1
    for i, (pt, d, n) in enumerate(zip(a.patches, a.dims, a.depths)):
        t //= pt
        k_, hd = din * pt, int(d * a.mlp)
        ops.append(Op(f"aud.{i}.patch", "gemm", 0, m=t, k=k_, n=d, role="vision", w_params=k_ * d, w_bits=16.0, w_fmt="bf16",
                      a_fmt=af, act_bytes=t * (k_ + d) * ab, stream=True, vec=t * d * VEC_PER_OUT))
        for j in range(n):
            ops.append(Op(f"aud.{i}.{j}.conv", "vector", 0, vec=t * d * 2 * a.kernel, act_bytes=2 * t * d * ab))
            ops.append(Op(f"aud.{i}.{j}.fc1", "gemm", 0, m=t, k=d, n=hd, role="vision", w_params=d * hd, w_bits=16.0,
                          w_fmt="bf16", a_fmt=af, act_bytes=t * (d + hd) * ab, stream=True, vec=t * hd * VEC_PER_OUT))
            ops.append(Op(f"aud.{i}.{j}.fc2", "gemm", 0, m=t, k=hd, n=d, role="vision", w_params=d * hd, w_bits=16.0,
                          w_fmt="bf16", a_fmt=af, act_bytes=t * (d + hd) * ab, stream=True, vec=t * d * VEC_PER_OUT))
        din = d
    ops.append(Op("aud.o_proj", "gemm", 0, m=t, k=din, n=a.out, role="vision", w_params=din * a.out, w_bits=16.0,
                  w_fmt="bf16", a_fmt=af, act_bytes=t * (din + a.out) * ab, stream=True, vec=t * a.out * VEC_PER_OUT))
    return ops


def _lin(i: int, o: int, b: bool = True) -> int:
    return i * o + (o if b else 0)


def _pixtral_resize(height: int, width: int, longest: int, factor: int) -> tuple[int, int]:
    """PixtralImageProcessor.get_resize_output_image_size, with the processor patch set to
    ``patch_size * spatial_merge_size`` (PixtralProcessor._process_images)."""
    ratio = max(height / longest, width / longest)
    if ratio > 1:
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))
    nh = (height - 1) // factor + 1
    nw = (width - 1) // factor + 1
    return nh * factor, nw * factor


def _pixtral_vision_spec(cfg: dict, vc: dict) -> VisionSpec | None:
    """PixtralVisionModel + Mistral3MultiModalProjector. All linears in the tower are bias-free.
    The merger linear is bias-free. The two projector linears follow ``multimodal_projector_bias``.

    0.418: ``hidden_act`` must be ``silu`` and ``projector_hidden_act`` must be ``gelu``.
    ``ACT2FN["silu"]`` is ``x * sigmoid(x)``. ``ACT2FN["gelu"]`` is the erf form.
    Anything else is refused. The RMSNorms are billed in ``_tower_ops``.
    """
    if vc.get("hidden_act") != "silu":
        raise ValueError("pixtral hidden_act is not silu")
    if cfg.get("projector_hidden_act") != "gelu":
        raise ValueError("pixtral projector_hidden_act is not gelu")
    h = vc.get("hidden_size")
    inter = vc.get("intermediate_size")
    depth = vc.get("num_hidden_layers")
    heads = vc.get("num_attention_heads")
    patch = vc.get("patch_size")
    channels = vc.get("num_channels")
    image_size = vc.get("image_size")
    text_h = (cfg.get("text_config") or {}).get("hidden_size")
    merge = cfg.get("spatial_merge_size")
    vfl = cfg.get("vision_feature_layer", -1)
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
               for x in (h, inter, depth, heads, patch, channels, text_h, merge)):
        return None
    if h % heads:
        return None
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        return None
    if isinstance(vfl, bool) or isinstance(vfl, int):
        n_feat = 1
    elif isinstance(vfl, list) and vfl and all(isinstance(i, int) and not isinstance(i, bool) for i in vfl):
        n_feat = len(vfl)
    else:
        return None
    bias = bool(cfg.get("multimodal_projector_bias"))
    params = (channels * patch * patch * h + h
              + depth * (2 * h + 4 * h * h + 3 * h * inter)
              + h + (merge * merge * h) * h
              + _lin(h * n_feat, text_h, bias) + _lin(text_h, text_h, bias))
    pre = tuple(sorted({"longest": image_size, "factor": patch * merge}.items()))
    post = ((merge * merge * h, h), (h * n_feat, text_h), (text_h, text_h))
    return VisionSpec("pixtral", "Pixtral ViT + 2x2 patch merger", patch, 1, h, inter, depth, heads, h,
                      True, merge, post, params, pre, False)


def _gemma4_resize(height: int, width: int, patch: int, max_patches: int, pool: int) -> tuple[int, int]:
    """transformers get_aspect_ratio_preserving_size (gemma4 and gemma4_unified image processors)."""
    total_px = height * width
    target_px = max_patches * (patch ** 2)
    factor = math.sqrt(target_px / total_px)
    side = pool * patch
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    max_side = (max_patches // (pool * pool)) * side
    if th == 0 and tw == 0:
        return side, side
    if th == 0:
        th = side
        tw = min(int(math.floor(width / height)) * side, max_side) or side
    elif tw == 0:
        tw = side
        th = min(int(math.floor(height / width)) * side, max_side) or side
    return th, tw


def _gemma4_vision_spec(cfg: dict, vc: dict) -> VisionSpec | None:
    """Gemma-4 ViT or the 12B encoder-free embedder. None when a weight has no accounting path."""
    text_h = (cfg.get("text_config") or {}).get("hidden_size") or cfg.get("hidden_size")
    if not text_h:
        return None
    mt = vc.get("model_type")
    if mt == "gemma4_unified_vision":
        patch = vc.get("patch_size") or 0
        pool = vc.get("pooling_kernel_size") or 0
        soft = vc.get("num_soft_tokens") or vc.get("default_output_length") or 0
        mm = vc.get("mm_embed_dim") or 0
        pos = vc.get("mm_posemb_size") or 0
        side = vc.get("model_patch_size") or 0
        if not (patch and pool and soft and mm and pos and side == patch * pool):
            return None
        # LN1 (w+b) → Linear+bias → LN2 (w+b) → pos table (pos, 2, mm) → LN3 (w+b) → Linear(mm → text, no bias).
        patch_dim = side * side * 3
        params = (2 * patch_dim + patch_dim * mm + mm + 2 * mm + pos * 2 * mm + 2 * mm + mm * text_h)
        au = audio_spec(cfg)
        if au is not None:
            params += au.params
        pre = {"patch": patch, "pool": pool, "max_soft": soft, "mm": mm, "patch_dim": patch_dim, "text_h": text_h}
        return VisionSpec("gemma4u", "Gemma-4 unified embedder (no vision tower)", patch, 1, mm, 0, 0, 1, mm,
                          False, pool, ((mm, text_h),), params, tuple(sorted(pre.items())), audio=au)
    if mt != "gemma4_vision":
        return None
    hv = vc.get("hidden_size") or 0
    inter = vc.get("intermediate_size") or 0
    depth = vc.get("num_hidden_layers") or 0
    nh = vc.get("num_attention_heads") or 0
    nkv = vc.get("num_key_value_heads") or nh
    hd = vc.get("head_dim") or (hv // nh if nh else 0)
    patch = vc.get("patch_size") or 0
    pool = vc.get("pooling_kernel_size") or 0
    pos = vc.get("position_embedding_size") or 0
    soft = vc.get("default_output_length") or cfg.get("vision_soft_tokens_per_image") or 0
    if not (hv and inter and depth and nh and hd and patch and pool and pos and soft):
        return None
    if nh * hd != hv:
        return None
    qd, kvd = nh * hd, nkv * hd
    # input proj (no bias) + position table (2, pos, h) + per-layer q/k/v/o, q/k RMSNorm, 4 norms, gated MLP
    params = (3 * patch * patch * hv + 2 * pos * hv
              + depth * (hv * qd + 2 * hv * kvd + qd * hv + 2 * hd + 4 * hv + 3 * hv * inter))
    if vc.get("standardize", False):
        params += 2 * hv
    if vc.get("use_clipped_linears"):
        # Seven Gemma4ClippableLinear modules per block. Each stores four bf16 scalar buffers.
        # The clamp does not change the GEMM.
        params += depth * 7 * 4
    mm = vc.get("output_proj_dims") or hv
    params += mm * text_h
    au = audio_spec(cfg)
    if au is not None and au.kind != "g4conf":
        return None
    if au is not None:
        params += au.params
    pre = {"patch": patch, "pool": pool, "max_soft": soft, "qd": qd, "kvd": kvd, "text_h": text_h,
           "std": 1 if vc.get("standardize", False) else 0}
    return VisionSpec("gemma4", "Gemma-4 ViT + pool + projector", patch, 1, hv, inter, depth, nh, qd, True, pool,
                      ((mm, text_h),), params, tuple(sorted(pre.items())), audio=au)


def _gemma4_vision_ops(v: VisionSpec, g: ImageGrid, images: int, af: str) -> list[Op]:
    """Padded tower (or unified) GEMMs, then the projector on the rows the reference actually projects."""
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    pr = v.rz
    # Image grids leave pad_soft at 0 and use the image max_soft_tokens. A video frame sets pad_soft to the
    # video processor's max_soft_tokens (70), which is not the image pad (280).
    pool, soft = pr["pool"], (g.pad_soft or pr["max_soft"])
    wf, wb = "bf16", 16.0

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op("vis." + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=wb, w_fmt=wf,
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    if v.family == "gemma4u":
        # Both linears see the padded soft-token rows. The LLM keeps g.llm_tokens of them.
        rows = images * soft
        k_patch = pr["patch_dim"]
        mm, text_h = pr["mm"], pr["text_h"]
        return [gemm("patch_dense", rows, k_patch, mm), gemm("proj", rows, mm, text_h)]
    n_pad = soft * pool * pool
    rows = images * n_pad
    h = v.hidden
    ops = [gemm("patch_embed", rows, 3 * v.patch * v.patch, h)]
    qd, kvd = pr["qd"], pr["kvd"]
    cnt = images * v.heads
    dh = v.head_dim
    keys = n_pad * n_pad
    for li in range(v.depth):
        ops += [gemm(f"{li}.q", rows, h, qd, li), gemm(f"{li}.k", rows, h, kvd, li),
                gemm(f"{li}.v", rows, h, kvd, li), gemm(f"{li}.o", rows, qd, h, li)]
        ops += [gemm(f"{li}.mlp_in", rows, h, 2 * v.inter, li), gemm(f"{li}.mlp_out", rows, v.inter, h, li)]
        ops += [Op(f"vis.{li}.qk", "attn", li, m=n_pad, k=dh, n=n_pad, count=cnt, causal=1.0,
                   act_bytes=cnt * n_pad * dh * ab, stream=True, orient=True),
                Op(f"vis.{li}.pv", "attn", li, m=n_pad, k=n_pad, n=dh, count=cnt, causal=1.0,
                   act_bytes=cnt * n_pad * dh * ab, orient=True),
                Op(f"vis.{li}.softmax", "vector", li, vec=cnt * keys * 5.0)]
    # Pool is a position-weighted average (no parameters), then the projector on the unpadded soft tokens.
    ops.append(Op("vis.pool", "vector", v.depth, vec=rows * h, act_bytes=rows * h * ab))
    m_proj = images * g.llm_tokens
    k_proj, n_proj = v.post[0]
    ops.append(gemm("proj", m_proj, k_proj, n_proj, v.depth))
    if pr.get("std"):
        ops.append(Op("vis.std", "vector", v.depth, vec=m_proj * h * 2, act_bytes=m_proj * h * ab))
    return ops


def _ernie_smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """ERNIE-4.5-VL ``smart_resize`` (processing_ernie4_5_vl.py). Factor is patch * spatial merge."""
    max_ratio = 200
    if min(height, width) > 0 and max(height, width) / min(height, width) > max_ratio:
        if height > width:
            width = max(factor, int(round(width / factor)) * factor)
            height = int(math.floor(width * max_ratio / factor)) * factor
        else:
            height = max(factor, int(round(height / factor)) * factor)
            width = int(math.floor(height * max_ratio / factor)) * factor
    hb = max(factor, int(round(height / factor)) * factor)
    wb = max(factor, int(round(width / factor)) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        hb = int(math.floor(height / beta / factor)) * factor
        wb = int(math.floor(width / beta / factor)) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        hb = int(math.ceil(height * beta / factor)) * factor
        wb = int(math.ceil(width * beta / factor)) * factor
    if hb < factor or wb < factor or hb * wb < min_pixels or hb * wb > max_pixels:
        raise ValueError(f"ernie4_5_vl smart_resize failed for {height}x{width}")
    return hb, wb


def _ernie_vl_vision_spec(cfg: dict, vc: dict) -> VisionSpec | None:
    """DFNRope vision tower plus the variable-resolution resampler.

    Tower weights are fp16 (``_keep_in_fp16_modules``). The resampler is bf16.
    Both are 2 bytes per parameter, so the tower GEMMs keep the default bf16 label.
    ``attn_sep`` is stored on the vision config and is not read by ``VisionAttention``.
    Image post-GEMMs all see the spatially merged rows: a single frame is duplicated
    and concatenated before the temporal linear, so the row count does not change.
    Video is a separate grid (_ernie_video): frames are paired and the temporal GEMMs
    see half as many rows.
    """
    h = int(vc.get("embed_dim") or vc.get("hidden_size") or 0)
    if h < 1 or int(vc.get("hidden_size") or h) != h:
        return None
    depth = int(vc["depth"])
    heads = int(vc["num_heads"])
    patch = int(vc["patch_size"])
    merge = int(vc["spatial_merge_size"])
    ratio = float(vc.get("mlp_ratio") or 0)
    if depth < 1 or heads < 1 or h % heads or patch < 1 or merge < 1 or ratio <= 0:
        return None
    if int(vc.get("in_channels") or 3) != 3:
        return None
    text_h = int(cfg["hidden_size"])
    pix = int(cfg.get("pixel_hidden_size") or 0)
    sc = int(cfg.get("spatial_conv_size") or 0)
    tcv = int(cfg.get("temporal_conv_size") or 0)
    if pix != h or sc != merge or tcv != 2:
        return None
    if cfg.get("use_temporal_conv", True) is not True:
        return None
    inter = int(h * ratio)
    block = (4 * h + _lin(h, 3 * h, True) + _lin(h, h, True) + _lin(h, inter, True) + _lin(inter, h, True))
    tower = _lin(3 * patch * patch, h, False) + depth * block + 2 * h
    sd = h * sc * sc
    td = sd * tcv
    res = 2 * _lin(sd, sd, True) + 2 * sd + _lin(td, sd, True) + _lin(sd, sd, True) + 2 * sd + _lin(sd, text_h, True) + text_h
    # Processor defaults (Ernie4_5_VLProcessor). preprocessor_config max_pixels is the image processor's
    # own field; _add_image passes the processor's image_max_pixels into get_smarted_resize.
    pre = (("factor", patch * merge), ("min_px", 4 * 28 * 28), ("max_px", 6177 * 28 * 28))
    post = ((sd, sd), (sd, sd), (td, sd), (sd, sd), (sd, text_h))
    return VisionSpec("ernievl", "ERNIE-4.5-VL DFN ViT + resampler", patch, 1, h, inter, depth, heads, h, False,
                      merge, post, tower + res, pre)


def _lfm2_vl_round_by_factor(number: float, factor: int) -> int:
    """Lfm2VlImageProcessor.round_by_factor. Python round is half-to-even."""
    return round(number / factor) * factor


def _lfm2_vl_target_ratios(min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
    """Tile grids whose area is in [min_tiles, max_tiles]. Equal areas keep (width, height) ascending.

    The processor sorts by area only. Set order would otherwise decide equal-area ties.
    """
    ratios = {
        (w, h)
        for n in range(min_tiles, max_tiles + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if min_tiles <= w * h <= max_tiles
    }
    return sorted(ratios, key=lambda x: (x[0] * x[1], x[0], x[1]))


def _lfm2_vl_closest_ratio(aspect: float, ratios: list[tuple[int, int]], width: int, height: int,
                           tile: int) -> tuple[int, int]:
    """Lfm2VlImageProcessor.find_closest_aspect_ratio."""
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for rw, rh in ratios:
        diff = abs(aspect - (rw / rh))
        if diff < best_diff:
            best_diff = diff
            best = (rw, rh)
        elif diff == best_diff and area > 0.5 * tile * tile * rw * rh:
            best = (rw, rh)
    return best


def _lfm2_vl_smart_resize(height: int, width: int, factor: int, min_tokens: int, max_tokens: int,
                          patch: int) -> tuple[int, int]:
    """Lfm2VlImageProcessor.smart_resize. Returns (width, height)."""
    total = patch * factor
    min_px = min_tokens * patch * patch * factor * factor
    max_px = max_tokens * patch * patch * factor * factor
    h_bar = max(total, _lfm2_vl_round_by_factor(height, total))
    w_bar = max(total, _lfm2_vl_round_by_factor(width, total))
    if h_bar * w_bar > max_px:
        beta = math.sqrt((height * width) / max_px)
        h_bar = max(total, int(math.floor(height / beta / total)) * total)
        w_bar = max(total, int(math.floor(width / beta / total)) * total)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (height * width))
        h_bar = int(math.ceil(height * beta / total)) * total
        w_bar = int(math.ceil(width * beta / total)) * total
    return w_bar, h_bar


def _lfm2_vl_too_large(height: int, width: int, max_tokens: int, patch: int, factor: int, tol: float) -> bool:
    """Lfm2VlImageProcessor._is_image_too_large."""
    total = patch * factor
    h_bar = max(patch, _lfm2_vl_round_by_factor(height, total))
    w_bar = max(patch, _lfm2_vl_round_by_factor(width, total))
    return h_bar * w_bar > max_tokens * patch * patch * factor * factor * tol


def _lfm2_vl_down_tokens(height: int, width: int, patch: int, factor: int) -> int:
    """Lfm2VlProcessor._compute_tokens_for_image."""
    ph = math.ceil((height // patch) / factor)
    pw = math.ceil((width // patch) / factor)
    return ph * pw


def _lfm2_vl_tile_tokens(tile: int, patch: int, factor: int) -> int:
    """Lfm2VlProcessor._compute_tokens_per_tile."""
    n = tile // patch
    d = math.ceil(n / factor)
    return d * d


def _pos_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _lfm2_vl_vision_spec(cfg: dict, vc: dict) -> VisionSpec | None:
    """Siglip2VisionModel (vision_use_head false) plus Lfm2VlMultiModalProjector.

    The tower is a linear patch embed (3 * patch * patch, bias), a learned position table of
    ``num_patches`` rows, pre-LN blocks (separate q / k / v / o, GELU fc1 / fc2, all with bias),
    and a post LayerNorm. The projector pixel-unshuffles by ``downsample_factor`` and runs two
    linears. ``projector_use_layernorm`` adds a LayerNorm on the unshuffled width. The pooling
    head is absent only when ``vision_use_head`` is false; any other value is not this module.
    """
    if vc.get("vision_use_head") is not False:
        return None
    if (cfg.get("text_config") or {}).get("model_type") != "lfm2":
        return None
    h = vc.get("hidden_size")
    inter = vc.get("intermediate_size")
    depth = vc.get("num_hidden_layers")
    heads = vc.get("num_attention_heads")
    patch = vc.get("patch_size")
    channels = vc.get("num_channels")
    npos = vc.get("num_patches")
    text_h = (cfg.get("text_config") or {}).get("hidden_size")
    factor = cfg.get("downsample_factor")
    proj_h = cfg.get("projector_hidden_size")
    enc_patch = cfg.get("encoder_patch_size")
    tile = cfg.get("tile_size")
    min_tok = cfg.get("min_image_tokens")
    max_tok = cfg.get("max_image_tokens")
    min_tiles = cfg.get("min_tiles")
    max_tiles = cfg.get("max_tiles")
    if not all(_pos_int(x) for x in (
            h, inter, depth, heads, patch, channels, npos, text_h, factor, proj_h, enc_patch, tile,
            min_tok, max_tok, min_tiles, max_tiles)):
        return None
    if h % heads or enc_patch != patch or tile % (patch * factor) or min_tiles > max_tiles:
        return None
    side = int(math.isqrt(npos))
    if side * side != npos:
        return None
    if not isinstance(cfg.get("projector_bias"), bool):
        return None
    if not isinstance(cfg.get("projector_use_layernorm"), bool):
        return None
    if not isinstance(cfg.get("do_image_splitting"), bool):
        return None
    if not isinstance(cfg.get("use_thumbnail"), bool):
        return None
    if not isinstance(cfg.get("use_image_special_tokens"), bool):
        return None
    tol = cfg.get("max_pixels_tolerance")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        return None
    # 0.419: the tower MLP is gelu_pytorch_tanh (not billed). The projector is ACT2FN["gelu"] (erf).
    if vc.get("hidden_act") != "gelu_pytorch_tanh":
        raise ValueError("lfm2_vl hidden_act is not gelu_pytorch_tanh")
    if cfg.get("projector_hidden_act") != "gelu":
        raise ValueError("lfm2_vl projector_hidden_act is not gelu")
    bias = cfg["projector_bias"]
    in_ch = h * factor * factor
    block = (4 * h + 4 * _lin(h, h, True) + _lin(h, inter, True) + _lin(inter, h, True))
    params = (_lin(channels * patch * patch, h, True) + npos * h + depth * block + 2 * h)
    if cfg["projector_use_layernorm"]:
        params += 2 * in_ch
    params += _lin(in_ch, proj_h, bias) + _lin(proj_h, text_h, bias)
    split = cfg["do_image_splitting"]
    pad = max(max_tok * factor * factor, (tile // patch) ** 2 if split else 0)
    pre = {
        "factor": factor, "tile": tile, "min_tok": min_tok, "max_tok": max_tok,
        "min_tiles": min_tiles, "max_tiles": max_tiles,
        "thumb": 1 if cfg["use_thumbnail"] else 0,
        "special": 1 if cfg["use_image_special_tokens"] else 0,
        "split": 1 if split else 0, "tol": float(tol), "pad": pad,
        "proj_ln": 1 if cfg["projector_use_layernorm"] else 0,
    }
    return VisionSpec("lfm2vl", "SigLIP2 NaFlex + pixel-unshuffle projector", patch, 1, h, inter, depth, heads, h,
                      False, factor, ((in_ch, proj_h), (proj_h, text_h)), params, tuple(sorted(pre.items())))


def _lfm2_vl_image_grid(v: VisionSpec, width: int, height: int) -> ImageGrid:
    """One image under Lfm2VlImageProcessor plus Lfm2VlProcessor._build_image_tokens.

    A tile or a resized image is padded to ``max_num_patches`` before the tower. The projector
    and the LLM feature tokens see only the valid grid after the unshuffle. Special tokens are
    extra LLM positions and are not tower rows.
    """
    pr = v.rz
    patch = v.patch
    factor = int(pr["factor"])
    tile = int(pr["tile"])
    min_tok = int(pr["min_tok"])
    max_tok = int(pr["max_tok"])
    min_tiles = int(pr["min_tiles"])
    max_tiles = int(pr["max_tiles"])
    thumb = bool(pr["thumb"])
    special = bool(pr["special"])
    split = bool(pr["split"])
    tol = float(pr["tol"])
    if width < 1 or height < 1:
        raise ValueError("lfm2_vl image size must be positive")
    if not split:
        min_tiles, max_tiles = 1, 1
    new_w, new_h = _lfm2_vl_smart_resize(height, width, factor, min_tok, max_tok, patch)
    too = _lfm2_vl_too_large(height, width, max_tok, patch, factor, tol)
    do_split = not (min_tiles == max_tiles == 1)
    if too and do_split:
        cols, rows = _lfm2_vl_closest_ratio(width / height, _lfm2_vl_target_ratios(min_tiles, max_tiles),
                                            width, height, tile)
        multi = rows > 1 or cols > 1
        tile_side = tile // patch
        if tile_side % factor:
            raise ValueError("lfm2_vl tile is not divisible by downsample_factor")
        if not multi:
            # crop_image_to_patches still resizes to one tile, but _build_image_tokens takes the
            # single-image branch and counts the smart_resize grid. Refuse when those disagree.
            feat_tile = _lfm2_vl_tile_tokens(tile, patch, factor)
            feat_img = _lfm2_vl_down_tokens(new_h, new_w, patch, factor)
            if feat_tile != feat_img:
                raise ValueError("lfm2_vl 1x1 tile tokens do not match the encoder output")
            ph = pw = tile_side
            specs = 2 if special else 0
            return ImageGrid(ph, pw, feat_tile, feat_tile + specs, (tile, tile), ((ph, pw, 1),))
        views: list[tuple[int, int, int]] = [(tile_side, tile_side, rows * cols)]
        feat = rows * cols * _lfm2_vl_tile_tokens(tile, patch, factor)
        specs = (1 + rows * cols) if special else 0
        if thumb:
            th_ph, th_pw = new_h // patch, new_w // patch
            if th_ph < 1 or th_pw < 1 or th_ph % factor or th_pw % factor:
                raise ValueError("lfm2_vl thumbnail grid is not divisible by downsample_factor")
            if (th_ph, th_pw) == (tile_side, tile_side):
                views[0] = (tile_side, tile_side, rows * cols + 1)
            else:
                views.append((th_ph, th_pw, 1))
            feat += _lfm2_vl_down_tokens(new_h, new_w, patch, factor)
            if special:
                specs += 1
        if special:
            specs += 1
        return ImageGrid(tile_side, tile_side, feat, feat + specs, (tile * cols, tile * rows), tuple(views))
    ph, pw = new_h // patch, new_w // patch
    if ph < 1 or pw < 1 or ph % factor or pw % factor:
        raise ValueError("lfm2_vl resized grid is not divisible by downsample_factor")
    feat = _lfm2_vl_down_tokens(new_h, new_w, patch, factor)
    specs = 2 if special else 0
    return ImageGrid(ph, pw, feat, feat + specs, (new_w, new_h), ((ph, pw, 1),))


def _lfm2_vl_vision_ops(v: VisionSpec, g: ImageGrid, images: int, af: str) -> list[Op]:
    """Padded SigLIP2 tower, then the projector on the unshuffled valid rows.

    pixel_values are padded to max_num_patches and the encoder attention is the full square
    (an additive mask, not a shorter sequence). get_image_features drops the pad before the
    projector.
    """
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    pad = int(v.rz["pad"])
    factor = v.merge
    groups = g.views or ((g.ph, g.pw, 1),)
    qf = {k: (f, b, a) for k, f, b, a in v.wfmt}
    h = v.hidden
    dh = v.head_dim
    ops: list[Op] = []

    def gemm(pfx: str, name: str, m: int, k: int, n: int, layer: int, kind: str) -> Op:
        f, b, a = qf.get(kind, ("bf16", 16.0, af)) if qf else ("bf16", 16.0, af)
        return Op(pfx + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=b, w_fmt=f,
                  a_fmt=a, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    if v.vec_fmt not in ("fp16", "bf16", "fp32"):
        raise ValueError("lfm2_vl activation dtype is not a vision float")
    proj_ln = v.rz.get("proj_ln")
    if proj_ln not in (0, 1) or len(v.post) != 2:
        raise ValueError("lfm2_vl projector is not unshuffle, linear, gelu, linear")
    for gi, (ph, pw, cnt) in enumerate(groups):
        if ph % factor or pw % factor or ph < 1 or pw < 1:
            raise ValueError("lfm2_vl patch grid is not divisible by downsample_factor")
        pfx = "vis." if gi == 0 else f"vis.v{gi}."
        rows = images * cnt * pad
        heads = images * cnt * v.heads
        ops.append(gemm(pfx, "patch_embed", rows, 3 * v.patch * v.patch, h, 0, "patch"))
        for li in range(v.depth):
            # Siglip2EncoderLayer: layer_norm1, attention, residual, layer_norm2, MLP, residual.
            # The residual adds are the full branch. The MLP activation is gelu_pytorch_tanh.
            ops.append(_ln_vec(f"{pfx}{li}.attn_ln", li, rows, h))
            ops += [gemm(pfx, f"{li}.q", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.k", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.v", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.o", rows, h, h, li, "attn")]
            ops.append(_ln_vec(f"{pfx}{li}.mlp_ln", li, rows, h))
            ops += [gemm(pfx, f"{li}.fc1", rows, h, v.inter, li, "mlp"),
                    gemm(pfx, f"{li}.fc2", rows, v.inter, h, li, "mlp")]
            keys = pad * pad
            ops += [Op(f"{pfx}{li}.qk", "attn", li, m=pad, k=dh, n=pad, count=heads, causal=1.0,
                       act_bytes=heads * pad * dh * ab, stream=True, orient=True),
                    Op(f"{pfx}{li}.pv", "attn", li, m=pad, k=pad, n=dh, count=heads, causal=1.0,
                       act_bytes=heads * pad * dh * ab, orient=True),
                    Op(f"{pfx}{li}.softmax", "vector", li, vec=heads * keys * 5.0)]
        # post_layernorm runs before get_image_features drops the pad. vision_use_head is false,
        # so the pooling head (another LayerNorm) does not run.
        ops.append(_ln_vec(pfx + "post_ln", v.depth, rows, h))
        m_proj = images * cnt * (ph // factor) * (pw // factor)
        if proj_ln:
            # Optional LayerNorm on the unshuffled width, before linear_1. Released configs leave it off.
            ops.append(_ln_vec(pfx + "proj_ln", v.depth, m_proj, v.post[0][0]))
        ops.append(gemm(pfx, "merger.0", m_proj, v.post[0][0], v.post[0][1], v.depth, "merger"))
        # ACT2FN["gelu"] is the erf form. Width is projector_hidden_size, not the tower hidden size.
        ops.append(_gelu_erf_vec(pfx + "pgelu", v.depth, m_proj * v.post[0][1], v.vec_fmt))
        ops.append(gemm(pfx, "merger.1", m_proj, v.post[1][0], v.post[1][1], v.depth, "merger"))
    return ops


def _d1_tile_ratios() -> tuple[tuple[int, int], ...]:
    """Insertion order of vision.layout's ratio set, then a stable sort by area.

    The released set comprehension iterates in hash order before sorted(), so an equal-area
    tie follows PYTHONHASHSEED. Sizes this bill locks have a unique strict diff, so the
    winner does not depend on that order. Generator order keeps the grid deterministic.
    """
    seen: list[tuple[int, int]] = []
    got: set[tuple[int, int]] = set()
    for n in range(2, 11):
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                if 2 <= x * y <= 10 and (x, y) not in got:
                    got.add((x, y))
                    seen.append((x, y))
    return tuple(sorted(seen, key=lambda r: r[0] * r[1]))


def _d1_vision_spec(cfg: dict) -> VisionSpec | None:
    """SigLIP2 NaFlex, 2x2 pixel-unshuffle, FastConformer, bidirectional trunk, decision head.

    The trunk is not an lfm2 decoder: there is no causal mask and no lm_head. VisionSpec.params
    is the vision role (tower + projector + audio). The trunk and the head stay on the caller.
    """
    if (cfg.get("dtype") or cfg.get("torch_dtype")) != "float32":
        return None
    vc = cfg.get("vision_config") or {}
    if vc.get("model_type") != "siglip2_vision_model" or vc.get("vision_use_head") is not False:
        return None
    tc = cfg.get("text_config") or {}
    trunk = _d1_trunk_params(tc)
    au = _d1_audio_spec(cfg)
    if trunk is None or au is None:
        return None
    trunk_params, ff, hd, types = trunk
    h = vc.get("hidden_size")
    inter = vc.get("intermediate_size")
    depth = vc.get("num_hidden_layers")
    heads = vc.get("num_attention_heads")
    patch = vc.get("patch_size")
    channels = vc.get("num_channels")
    npos = vc.get("num_patches")
    text_h = tc.get("hidden_size")
    proj_h = cfg.get("projector_hidden_size")
    head_layers = cfg.get("head_layers")
    max_len = cfg.get("max_length")
    image_text = cfg.get("image_text_length")
    audio_text = cfg.get("audio_text_length")
    if not all(_pos_int(x) for x in (
            h, inter, depth, heads, patch, channels, npos, text_h, proj_h, head_layers, max_len,
            image_text, audio_text)):
        return None
    if patch != 16 or channels != 3 or h % heads or text_h % 64:
        return None
    if text_h != au.out or au.out != tc.get("hidden_size"):
        return None
    block = (4 * h + 4 * _lin(h, h, True) + _lin(h, inter, True) + _lin(inter, h, True))
    tower = _lin(channels * patch * patch, h, True) + npos * h + depth * block + 2 * h
    in_ch = h * 2 * 2
    proj = _lin(in_ch, proj_h, True) + _lin(proj_h, text_h, True)
    head_params = _d1_head_params(text_h, head_layers)
    # 17 BatchNorms: running_mean / running_var are fp32, num_batches_tracked is one I64 each.
    buf = au.depths[0] * au.dims[0] * 2 * 4 + au.depths[0] * 8
    pre = {
        "pad": _D1_PAD, "tile": _D1_TILE, "px": _D1_PX,
        "min_px": 64 * 1024, "max_px": 256 * 1024,
        "max_len": max_len, "image_text": image_text, "audio_text": audio_text,
        "text_floor": _D1_TEXT_FLOOR,
        "trunk": text_h, "ff": ff, "nq": tc["num_attention_heads"], "nkv": tc["num_key_value_heads"],
        "hd": hd, "ck": 3, "types": ",".join(types),
        "hl": head_layers, "hff": 4 * text_h, "hh": text_h // 64, "vocab": tc["vocab_size"],
        "trunk_params": trunk_params, "head_params": head_params, "buf": buf,
    }
    wfmt = tuple((k, "fp32", 32.0, "fp32") for k in ("patch", "attn", "mlp", "merger"))
    return VisionSpec(
        "d1omni", "SigLIP2 NaFlex + pixel-unshuffle + FastConformer (bidirectional trunk)",
        patch, 1, h, inter, depth, heads, h, False, 2, ((in_ch, proj_h), (proj_h, text_h)),
        tower + proj + au.params, tuple(sorted(pre.items())), audio=au, wfmt=wfmt, w_Bpp=4.0)


def _d1_image_grid(v: VisionSpec, width: int, height: int) -> ImageGrid:
    """vision.layout: smart-resize, then up to ten 512 px tiles plus a thumbnail.

    Prefix tokens are the unshuffled crops. The tower still pads every crop to 1024.
    There are no special image tokens.
    """
    pr = v.rz
    factor = int(pr["px"])
    tile = int(pr["tile"])
    maximum = int(pr["max_px"])
    minimum = int(pr["min_px"])
    patch = v.patch
    if width < 1 or height < 1:
        raise ValueError("d1 image size must be positive")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    large = (max(16, round(height / factor) * factor) * max(16, round(width / factor) * factor)
             > maximum * 2)
    views: list[tuple[int, int, int]] = []
    feat = 0
    if large:
        best = float("inf")
        grid = (1, 1)
        for ratio in _d1_tile_ratios():
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (diff == best and width * height > 0.5 * tile * tile * ratio[0] * ratio[1]):
                grid, best = ratio, diff
        gw, gh = grid
        side = tile // patch
        views.append((side, side, gw * gh))
        feat += gw * gh * (side * side // 4)
    ph, pw = h // patch, w // patch
    if ph < 1 or pw < 1 or ph % v.merge or pw % v.merge:
        raise ValueError("d1 thumbnail grid is not divisible by the unshuffle")
    thumb = ph * pw // 4
    if views and (ph, pw) == (views[0][0], views[0][1]):
        a, b, c = views[0]
        views[0] = (a, b, c + 1)
    else:
        views.append((ph, pw, 1))
    feat += thumb
    px = (tile * grid[0], tile * grid[1]) if large else (w, h)
    return ImageGrid(views[0][0], views[0][1], feat, feat, px, tuple(views))


def _d1_vision_ops(v: VisionSpec, g: ImageGrid, images: int, af: str) -> list[Op]:
    """Padded SigLIP2 square, then the projector on the valid unshuffled rows. Weights and activations are fp32."""
    del af
    from .dtypes import fmt as _fmt
    ab = _fmt("fp32").bytes
    pad = int(v.rz["pad"])
    factor = v.merge
    groups = g.views or ((g.ph, g.pw, 1),)
    qf = {k: (f, b, a) for k, f, b, a in v.wfmt}
    h = v.hidden
    dh = v.head_dim
    ops: list[Op] = []

    def gemm(pfx: str, name: str, m: int, k: int, n: int, layer: int, kind: str) -> Op:
        f, b, a = qf[kind]
        return Op(pfx + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=b, w_fmt=f,
                  a_fmt=a, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    for gi, (ph, pw, cnt) in enumerate(groups):
        if ph % factor or pw % factor or ph < 1 or pw < 1:
            raise ValueError("d1 patch grid is not divisible by downsample_factor")
        pfx = "vis." if gi == 0 else f"vis.v{gi}."
        rows = images * cnt * pad
        heads = images * cnt * v.heads
        ops.append(gemm(pfx, "patch_embed", rows, 3 * v.patch * v.patch, h, 0, "patch"))
        for li in range(v.depth):
            ops += [gemm(pfx, f"{li}.q", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.k", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.v", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.o", rows, h, h, li, "attn"),
                    gemm(pfx, f"{li}.fc1", rows, h, v.inter, li, "mlp"),
                    gemm(pfx, f"{li}.fc2", rows, v.inter, h, li, "mlp")]
            ops += [Op(f"{pfx}{li}.qk", "attn", li, m=pad, k=dh, n=pad, count=heads, causal=1.0,
                       act_bytes=heads * pad * dh * ab, stream=True, orient=True),
                    Op(f"{pfx}{li}.pv", "attn", li, m=pad, k=pad, n=dh, count=heads, causal=1.0,
                       act_bytes=heads * pad * dh * ab, orient=True),
                    Op(f"{pfx}{li}.softmax", "vector", li, vec=heads * pad * pad * 5.0)]
        m_proj = images * cnt * (ph // factor) * (pw // factor)
        for i, kn in enumerate(v.post):
            k, n = kn[:2]
            ops.append(gemm(pfx, f"merger.{i}", m_proj, k, n, v.depth, "merger"))
    return ops


def d1_text_len(v: VisionSpec, prompt: int, prefix: int, images: int, audio_s: float) -> int:
    """Text tokens after the class cap. Prefix embeddings are not text. Images and audio are exclusive."""
    if images and audio_s:
        raise ValueError("d1 一次请求只能带图像或音频")
    pr = v.rz
    if images:
        cap = int(pr["image_text"])
    elif audio_s:
        cap = int(pr["audio_text"])
    else:
        cap = int(pr["max_len"])
    max_len = min(cap, int(pr["max_len"]) - prefix)
    if max_len < int(pr["text_floor"]):
        raise ValueError(f"d1 媒体前缀占了 {prefix} 个位置，剩下的文本不足 {int(pr['text_floor'])}")
    return min(prompt, max_len)


def d1_trunk_ops(v: VisionSpec, sequences: int, text: int, prefix: int, labels: int,
                 af: str = "fp32") -> list[Op]:
    """One bidirectional trunk pass per question, then the decision head on text positions only.

    Attention is the full square (the additive mask is still scored). The centered 3-tap is a
    vector op, not a recurrent state. The scorer gathers marker rows only. Type embedding is a
    lookup, not a GEMM.
    """
    del af
    if sequences <= 0 or text < 0 or prefix < 0:
        return []
    if labels > text:
        raise ValueError("d1 的选项标记不能多于文本长度")
    from .dtypes import fmt as _fmt
    ab = _fmt("fp32").bytes
    pr = v.rz
    h = int(pr["trunk"])
    ff = int(pr["ff"])
    nq = int(pr["nq"])
    nkv = int(pr["nkv"])
    hd = int(pr["hd"])
    length = prefix + text
    m_trunk = sequences * length
    m_text = sequences * text
    types = str(pr["types"]).split(",")

    def gemm(name: str, m: int, k: int, n: int, layer: int) -> Op:
        return Op(name, "gemm", layer, m=m, k=k, n=n, role="attn", w_params=k * n, w_bits=32.0, w_fmt="fp32",
                  a_fmt="fp32", act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    ops: list[Op] = []
    if text:
        ops.append(Op("trunk.embed", "lookup", -1, kv_read=m_text * h * ab, vec=m_text * h))
    for i, kind in enumerate(types):
        if kind == "conv":
            ops.append(gemm(f"trunk.{i}.in", m_trunk, h, 3 * h, i))
            # Three taps at 2 flops each, plus the B*x and C*y gates.
            ops.append(Op(f"trunk.{i}.conv", "vector", i, vec=m_trunk * h * 8,
                          act_bytes=2 * m_trunk * h * ab))
            ops.append(gemm(f"trunk.{i}.out", m_trunk, h, h, i))
        else:
            ops += [gemm(f"trunk.{i}.q", m_trunk, h, nq * hd, i),
                    gemm(f"trunk.{i}.k", m_trunk, h, nkv * hd, i),
                    gemm(f"trunk.{i}.v", m_trunk, h, nkv * hd, i),
                    gemm(f"trunk.{i}.o", m_trunk, h, h, i)]
            cnt = sequences * nq
            if length:
                ops += [Op(f"trunk.{i}.qk", "attn", i, m=length, k=hd, n=length, count=cnt, causal=1.0,
                           act_bytes=cnt * length * hd * ab, stream=True, orient=True),
                        Op(f"trunk.{i}.pv", "attn", i, m=length, k=length, n=hd, count=cnt, causal=1.0,
                           act_bytes=cnt * length * hd * ab, orient=True),
                        Op(f"trunk.{i}.softmax", "vector", i, vec=cnt * length * length * 5.0)]
        ops += [gemm(f"trunk.{i}.w1", m_trunk, h, ff, i),
                gemm(f"trunk.{i}.w3", m_trunk, h, ff, i),
                gemm(f"trunk.{i}.w2", m_trunk, ff, h, i)]
    hl = int(pr["hl"])
    hh = int(pr["hh"])
    hff = int(pr["hff"])
    hhd = h // hh
    if text:
        ops.append(Op("head.type_emb", "lookup", -1, kv_read=sequences * h * ab, vec=sequences * h))
        for j in range(hl):
            ops += [gemm(f"head.{j}.in", m_text, h, 3 * h, j),
                    gemm(f"head.{j}.o", m_text, h, h, j)]
            cnt = sequences * hh
            ops += [Op(f"head.{j}.qk", "attn", j, m=text, k=hhd, n=text, count=cnt, causal=1.0,
                       act_bytes=cnt * text * hhd * ab, stream=True, orient=True),
                    Op(f"head.{j}.pv", "attn", j, m=text, k=text, n=hhd, count=cnt, causal=1.0,
                       act_bytes=cnt * text * hhd * ab, orient=True),
                    Op(f"head.{j}.softmax", "vector", j, vec=cnt * text * text * 5.0)]
            ops += [gemm(f"head.{j}.fc1", m_text, h, hff, j),
                    gemm(f"head.{j}.fc2", m_text, hff, h, j)]
    if labels > 0 and text:
        m_score = sequences * labels
        ops += [gemm("head.scorer.1", m_score, h, h, hl),
                gemm("head.scorer.3", m_score, h, 1, hl)]
    return ops


def _omni_int(meta: dict, key: str) -> int:
    v = meta.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"longcat omni field {key} is not an int")
    return v


def longcat_omni_counts(meta: dict) -> dict:
    """Parameter counts of the published Omni towers and audio heads.

    The formulas are ``LongCatVisionModel``, ``VisionProjector``, ``LongCatAudioEncoder``,
    ``LongCatOmniAudioAdaptor``, ``AudioEmbedding``, and ``AudioOutputLayer`` (``has_proj`` false).
    Each count must match the ``.pt`` header stored on the release. A mismatch raises.
    """
    if meta.get("qkv_bias") is not False:
        raise ValueError("longcat omni qkv bias is not false")
    if meta.get("has_proj") is not False:
        raise ValueError("longcat omni audio output has_proj is not false")
    if meta.get("hidden_act") != "SwiGLU":
        raise ValueError("longcat omni vision activation is not SwiGLU")
    hv = _omni_int(meta, "vision_hidden")
    layers = _omni_int(meta, "vision_layers")
    heads = _omni_int(meta, "vision_heads")
    inter_cfg = _omni_int(meta, "vision_intermediate")
    patch = _omni_int(meta, "patch")
    tpatch = _omni_int(meta, "tpatch")
    merge = _omni_int(meta, "spatial_merge")
    resize_factor = _omni_int(meta, "resize_factor")
    min_tokens = _omni_int(meta, "min_tokens")
    max_tokens = _omni_int(meta, "max_tokens")
    adaptor = _omni_int(meta, "adaptor_dim")
    text_h = _omni_int(meta, "text_hidden")
    a_in = _omni_int(meta, "audio_input")
    a_h = _omni_int(meta, "audio_hidden")
    a_proj = _omni_int(meta, "audio_proj")
    a_vocab = _omni_int(meta, "audio_vocab")
    a_layers = _omni_int(meta, "audio_layers")
    a_right = _omni_int(meta, "audio_right_layers")
    left_order = _omni_int(meta, "audio_left_order")
    right_order = _omni_int(meta, "audio_right_order")
    ndnn = _omni_int(meta, "audio_ndnn")
    rate = _omni_int(meta, "audio_rate")
    frame_ms = _omni_int(meta, "audio_frame_ms")
    shift_ms = _omni_int(meta, "audio_shift_ms")
    stride = _omni_int(meta, "audio_stride")
    n_in = _omni_int(meta, "codebooks")
    n_res = _omni_int(meta, "residual_codebooks")
    code_vocab = _omni_int(meta, "code_vocab")
    if hv <= 0 or heads <= 0 or hv % heads or layers <= 0 or patch <= 0 or tpatch <= 0:
        raise ValueError("longcat omni vision shape is not positive")
    if merge != 1:
        raise ValueError("longcat omni spatial_merge_size is not 1")
    if a_right < 0 or a_right > a_layers or left_order < 0 or right_order < 0 or ndnn < 1:
        raise ValueError("longcat omni audio layer shape is outside the encoder")
    if n_in < 1 or n_res < 0 or n_res >= n_in or code_vocab <= 0 or text_h <= 0:
        raise ValueError("longcat omni audio heads are not the released codebook set")
    if a_h != text_h:
        raise ValueError("longcat omni audio hidden size is not the text hidden size")
    inner = int(inter_cfg * 2 / 3)
    if inner <= 0:
        raise ValueError("longcat omni SwiGLU inner width is not positive")
    # Conv3d bias, qkv without bias, q/k RMSNorm, proj with bias, two RMSNorms, two LayerScales,
    # fc1 / fc3 / fc2 with bias, and the inner RMSNorm.
    one = (hv * 3 * hv + 2 * hv + hv * hv + hv + 2 * hv + 2 * hv
           + 2 * (hv * inner + inner) + inner + (inner * hv + hv))
    vision = 3 * tpatch * patch * patch * hv + hv + layers * one
    ed = hv * 2
    vproj = 2 * ed + (ed * adaptor + adaptor) + (adaptor * adaptor + adaptor) + (adaptor * text_h + text_h)
    # Pre linear with bias, projection without bias. Each FSMN layer: two LayerNorms, depthwise
    # conv without bias, and a biased up/down FFN. DNN is ndnn biased linears. Output is biased.
    pre = a_in * a_h + a_h + a_h * a_proj

    def lyr(k: int) -> int:
        return 2 * a_proj + a_proj * k + 2 * a_proj + (a_proj * a_h + a_h) + (a_h * a_proj + a_proj)

    enc_layers = (a_layers - a_right) * lyr(left_order + 1) + a_right * lyr(left_order + right_order + 1)
    dnn = a_proj * a_h + a_h + (ndnn - 1) * (a_h * a_h + a_h)
    out = a_h * a_vocab + a_vocab
    encoder = pre + enc_layers + dnn + out
    aproj = 2 * (a_h * a_h + a_h)
    emb = n_in * code_vocab * text_h
    residual = n_res * code_vocab * text_h
    heads_p = n_in * text_h * code_vocab
    expect = {
        "params_vision_model": vision,
        "params_vision_projector": vproj,
        "params_audio_encoder": encoder,
        "params_audio_projector": aproj,
        "params_audio_emb": emb,
        "params_audio_out": residual + heads_p,
    }
    for key, params in expect.items():
        if _omni_int(meta, key) != params:
            raise ValueError(f"longcat omni {key} {meta.get(key)} != class formula {params}")
        bkey = "bytes_" + key[len("params_"):]
        if _omni_int(meta, bkey) != 2 * params:
            raise ValueError(f"longcat omni {bkey} is not 2 bytes per parameter")
    window = rate * frame_ms // 1000
    hop = rate * shift_ms // 1000
    if window <= 0 or hop <= 0 or stride <= 0:
        raise ValueError("longcat omni audio frame geometry is not positive")
    return {
        "vision_model": vision, "vision_projector": vproj, "audio_encoder": encoder,
        "audio_projector": aproj, "audio_emb": emb, "audio_residual": residual,
        "audio_heads": heads_p, "towers": vision + vproj + encoder + aproj,
        "text_hidden": text_h, "vision_hidden": hv, "vision_layers": layers,
        "vision_heads": heads, "inner": inner, "patch": patch, "tpatch": tpatch,
        "adaptor": adaptor, "min_tokens": min_tokens, "max_tokens": max_tokens,
        "resize_factor": resize_factor, "audio_input": a_in, "audio_hidden": a_h,
        "audio_proj": a_proj, "audio_vocab": a_vocab, "audio_layers": a_layers,
        "audio_right": a_right, "left_order": left_order, "right_order": right_order,
        "ndnn": ndnn, "rate": rate, "window": window, "hop": hop, "stride": stride,
        "codebooks": n_in, "residual": n_res, "code_vocab": code_vocab,
    }


def _longcat_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """``LongCatImageTransform.smart_resize``, including the extra 14 / 28 clamps."""
    if factor <= 0:
        raise ValueError("longcat omni resize factor is not positive")
    if height < factor or width < factor:
        if height < factor:
            ratio = factor / height
            height, width = factor, int(ratio * width) + 1
        if width < factor:
            ratio = factor / width
            width, height = factor, int(ratio * height) + 1
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    if h_bar < factor:
        h_bar, w_bar = factor, max_pixels // factor
    if w_bar < factor:
        h_bar, w_bar = max_pixels // factor, factor
    if h_bar % 14 != 0:
        h_bar = h_bar // 14 * 14
    if w_bar % 28 != 0:
        w_bar = w_bar // 28 * 28
    return h_bar, w_bar


def _longcat_omni_vision_spec(cfg: dict) -> VisionSpec:
    meta = cfg.get("longcat_omni")
    if not isinstance(meta, dict):
        raise ValueError("longcat omni tower header is missing")
    counts = longcat_omni_counts(meta)
    text_h = cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size")
    if text_h != counts["text_hidden"]:
        raise ValueError("longcat omni text hidden does not match the language config")
    hv = counts["vision_hidden"]
    inner = counts["inner"]
    adaptor = counts["adaptor"]
    factor = counts["patch"] * counts["resize_factor"]
    pre = {
        "factor": factor,
        "min_px": counts["min_tokens"] * counts["patch"] * counts["patch"],
        "max_px": counts["max_tokens"] * counts["patch"] * counts["patch"],
    }
    au = AudioSpec(
        counts["rate"], (counts["hop"],),
        (counts["audio_proj"], counts["audio_hidden"], counts["audio_input"]),
        (counts["audio_layers"], counts["audio_right"], counts["ndnn"]),
        counts["left_order"], 0.0, counts["audio_vocab"],
        counts["audio_encoder"] + counts["audio_projector"],
        kind="lcafsmn", chunk=counts["stride"], frame=counts["window"],
        ctx_l=counts["left_order"], ctx_r=counts["right_order"],
        lin=((counts["audio_hidden"], text_h), (text_h, text_h)))
    post = ((hv * 2, adaptor), (adaptor, adaptor), (adaptor, text_h))
    return VisionSpec(
        "longcatvit", "LongCat ViT（rope2d，SwiGLU）+ 宽度拼接投影；FSMN 音频编码器",
        counts["patch"], counts["tpatch"], hv, inner, counts["vision_layers"],
        counts["vision_heads"], hv, True, 1, post,
        counts["towers"], tuple(sorted(pre.items())), False, audio=au)


def vision_spec(hf_id: str, cfg: dict) -> VisionSpec | None:
    """VisionSpec from the release's full config (``vision_config`` + text hidden size), or None."""
    vc = cfg.get("vision_config")
    hf_id = hf_id.split("+", 1)[0]              # 0.175: "<target>+<draft repo>" entries use the target's processor
    if "LongcatFlashOmniForCausalLM" in (cfg.get("architectures") or []) or cfg.get("model_type") == "longcat_flash_omni":
        return _longcat_omni_vision_spec(cfg)
    if cfg.get("model_type") == "qwen2_5_omni":
        # 0.271 Qwen2.5-Omni: the audio tower is modelled; the Qwen2.5-VL ViT (thinker.visual) is stored only (the
        # release params_vision surplus → VisionSpec.extra in from_release) — image / video prompts are not costed
        au = audio_spec(cfg)
        if au is not None:
            return VisionSpec("audio", "Qwen2.5-Omni 音频编码器（Whisper-v3，200 帧分块、只跑有效帧）+ 平均池化 + 线性投影",
                              0, 1, au.dims[0], int(au.mlp * au.dims[0]), au.depths[0], au.heads, au.dims[0], False, 1,
                              (), au.params, (), audio=au,
                              approx="视觉塔（Qwen2.5-VL ViT，thinker.visual）只计存储：图像 / 视频输入未建模")
        return None
    if cfg.get("model_type") == "qwen3_omni_moe":
        # 0.345 Qwen3-Omni: AuT audio modelled; ViT (thinker.visual) storage only — image/video prompts not costed
        au = audio_spec(cfg)
        if au is not None:
            return VisionSpec("audio", "Qwen3-Omni AuT 音频编码器（Conv2d ×3 下采样 8×，窗口注意力）+ MLP 投影",
                              0, 1, au.dims[0], int(au.mlp * au.dims[0]), au.depths[0], au.heads, au.dims[0], False, 1,
                              (), au.params, (), audio=au,
                              approx="视觉塔（thinker.visual）只计存储：图像 / 视频输入未建模")
        return None
    if not vc:
        au = audio_spec(cfg)
        if au is not None and au.kind == "gspeech":
            lab = ("Granite Speech Plus：Conformer（Shaw 分块相对位置，拼接中间层）+ Q-Former 查询桥" if
                   cfg.get("model_type") == "granite_speech_plus" else
                   "Granite Speech：Conformer（Shaw 分块相对位置）+ Q-Former 查询桥")
            return VisionSpec("audio", lab, 0, 1, au.dims[0], au.gs_ff, au.depths[0], au.heads, au.dims[0],
                              False, 1, (), au.params, (), audio=au)
        if au is not None and au.kind == "gsctc":
            return VisionSpec("audio", "Granite Speech 5 CTC：Shaw Conformer（时间下采样）+ 绑定的 CTC 头",
                              0, 1, au.dims[0], au.gs_ff, au.depths[0], au.heads, au.dims[0],
                              False, 1, (), au.params, (), audio=au)
        if au is not None and au.kind == "w2v" and au.ctc:
            return VisionSpec("audio", "OmniASR CTC：Wav2Vec2 编码器（双向）+ 线性 CTC 头",
                              0, 1, au.dims[0], int(au.mlp * au.dims[0]), au.depths[0], au.heads,
                              au.dims[0], False, 1, (), au.params, (), audio=au)
        if au is not None and au.kind in ("whisper", "aut", "sanm", "w2v"):
            # 0.268 audio-only release (Ming-3.0-asr-tiny): an encoder container with no image path (family "audio":
            # image_grid_of → None, serving.images ignored with a warning); costed when serving.audio_s > 0
            # 0.270: Whisper-large-v3 (encoder-decoder), Qwen2-Audio (Whisper-v3 encoder + pool + projector), Qwen3-ASR (AuT)
            lab = ("Whisper-large-v3 音频编码器（30 s 填充窗、逐窗）+ 4 帧拼接 → 适配器 MLP（12.5 Hz）" if au.stack else
                   "Whisper 式音频编码器（整段一次、≤ 30 s）+ 平均池化 + 卷积适配器（s2）+ MLP 投影" if au.ds and au.lin else
                   "Whisper-v3 音频编码器（30 s 填充窗，交叉注意力输入）" if au.cross else
                   "Whisper-v3 音频编码器（30 s 填充）+ 平均池化 + 线性投影" if au.lin is not None and au.kind == "whisper" else
                   "AuT 音频编码器（Qwen3-Omni，Conv2d ×3 下采样 8×，窗口注意力）+ MLP 投影 + ST 连接器" if au.kind == "aut" and cfg.get("model_type") == "index_echo_s2st" else
                   "AuT 音频编码器（Conv2d ×3 下采样 8×，窗口注意力）+ MLP 投影" if au.kind == "aut" else
                   "SAN-M 音频编码器（LFR 7/6 帧堆叠，70 层 FSMN 自注意力，整段双向）+ 投影 + 2 层适配器" if au.kind == "sanm" else
                   "OmniASR / Wav2Vec2 音频编码器（7 层 Conv1d 下采样 320× → 50 Hz，双向 Transformer）+ 线性投影" if au.kind == "w2v" else
                   "Whisper 音频编码器 + 卷积下采样投影")
            return VisionSpec("audio", lab, 0, 1, au.dims[0], int(au.mlp * au.dims[0]),
                              au.depths[0], au.heads, au.dims[0], False, 1, (), au.params, (), audio=au,
                              # 0.273 Kimi-Audio: the discrete input tokens come from a second encoder in another repo
                              approx=("输入离散语义 token：GLM-4-Voice tokenizer（THUDM/glm-4-voice-tokenizer，Whisper-v3 "
                                      "前 16 层 + 4× 池化 + VQ 16384，另一仓库）未计入音频编码耗时" if au.stack else ""))
        return None
    if cfg.get("model_type") == "d1_omni":
        return _d1_vision_spec(cfg)
    if cfg.get("model_type") == "lfm2_vl" and vc.get("model_type") == "siglip2_vision_model":
        return _lfm2_vl_vision_spec(cfg, vc)
    if vc.get("model_type") in ("gemma4_vision", "gemma4_unified_vision"):
        return _gemma4_vision_spec(cfg, vc)
    if cfg.get("model_type") == "mistral3" and vc.get("model_type") == "pixtral":
        return _pixtral_vision_spec(cfg, vc)
    if cfg.get("model_type") == "ernie4_5_moe_vl":
        return _ernie_vl_vision_spec(cfg, vc)
    if hf_id not in _PREPROC:
        return None
    pre = dict(_PREPROC[hf_id])
    text_h = (cfg.get("text_config") or {}).get("hidden_size") or cfg.get("hidden_size")
    mt = vc.get("model_type", "")
    if "vt_hidden_size" in vc:                      # MoonViT (Kimi)
        h, inter, L, nh = vc["vt_hidden_size"], vc["vt_intermediate_size"], vc["vt_num_hidden_layers"], vc["vt_num_attention_heads"]
        q = vc.get("qkv_hidden_size") or h
        v2 = vc.get("mm_projector_type") == "patchmergerv2"
        bias = vc.get("linear_bias", True) if v2 else True
        abias = vc.get("attn_bias", True) if v2 else True
        pbias = vc.get("patch_embed_proj_bias", True)
        p = vc["patch_size"]
        mk = vc["merge_kernel_size"][0]
        norm = h if vc.get("norm_type") == "rmsnorm" else 2 * h
        params = (_lin(3 * p * p, h, pbias) + vc["init_pos_emb_height"] * vc["init_pos_emb_width"] * h
                  + L * (2 * norm + _lin(h, 3 * q, abias) + _lin(q, h, abias) + _lin(h, inter, bias) + _lin(inter, h, bias))
                  + norm)
        hm = h * mk * mk
        params += (_lin(hm, hm, False) + _lin(hm, text_h, False) + text_h) if v2 else \
            (2 * h + _lin(hm, hm) + _lin(hm, text_h))
        return VisionSpec("kimi", "MoonViT" + (" v2" if v2 else ""), p, 1, h, inter, L, nh, q, False, mk,
                          ((hm, hm), (hm, text_h)), params, tuple(sorted(pre.items())), abias)
    if mt in ("qwen3_5", "qwen3_5_moe", "qwen4_exp", "qwen3_5_vision", "qwen3_5_moe_vision",
              "qwen3_vl", "qwen3_vl_moe", "qwen3_vl_vision"):
        # 0.191: transformers 5 names Qwen3.5 vision *_vision. 0.366: Qwen3-VL deepstack.
        # 0.388: transformers 5.12 Qwen3VLVisionConfig.model_type is qwen3_vl_vision. Older checkpoints
        # still say qwen3_vl; Qwen3VLConfig rewrites that name to qwen3_vl_vision. Same module.
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"]
        p, t, mk, out = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"], vc["out_hidden_size"]
        # 0.366: deepstack mergers exist only on Qwen3VLVisionModel (one ModuleList entry per index,
        # use_postshuffle_norm=True).  A non-empty list on Qwen3.5 / Qwen3.8 is still not that module.
        ds = vc.get("deepstack_visual_indexes") or ()
        if ds and mt not in ("qwen3_vl", "qwen3_vl_moe", "qwen3_vl_vision"):
            return None
        if any(type(i) is not int or i < 0 or i >= L for i in ds):
            return None
        hm = h * mk * mk
        n_ds = len(ds)
        # final merger: pre-shuffle LN(h) + Linear(4h, 4h) + Linear(4h, out)
        # each deepstack merger: post-shuffle LN(4h) + the same two linears
        params = (_lin(3 * t * p * p, h) + vc["num_position_embeddings"] * h
                  + L * (4 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + 2 * h + _lin(hm, hm) + _lin(hm, out)
                  + n_ds * (2 * hm + _lin(hm, hm) + _lin(hm, out)))
        pre["factor"] = p * mk
        post = ((hm, hm), (hm, out)) * (1 + n_ds)
        label = "Qwen ViT" if n_ds == 0 else "Qwen3-VL ViT + deepstack"
        # 0.419: a non-empty deepstack list is Qwen3VLVisionModel. The block MLP is
        # gelu_pytorch_tanh. The merger activation is nn.GELU(), independent of hidden_act.
        q3 = n_ds > 0 and mt in ("qwen3_vl", "qwen3_vl_moe", "qwen3_vl_vision")
        if q3 and vc.get("hidden_act") != "gelu_pytorch_tanh":
            raise ValueError("qwen3vl hidden_act is not gelu_pytorch_tanh")
        return VisionSpec("qwen", label, p, t, h, inter, L, nh, h, False, mk, post, params,
                          tuple(sorted(pre.items())), deepstack=n_ds,
                          ln_mode="qwen3vl" if q3 else "", ds_idx=tuple(ds) if q3 else ())
    if mt == "glm5_next_vision":
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"]
        p, t, mk, out, pi = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"], vc["out_hidden_size"], \
            vc["projection_intermediate_size"]
        b = vc.get("attention_bias", True)
        params = (3 * t * p * p * h + h
                  + L * (2 * h + _lin(h, 3 * h, b) + _lin(h, h, b) + 2 * (h // nh) + 2 * _lin(h, inter, b) + _lin(inter, h, b))
                  + h + _lin(h * mk * mk, out) + _lin(out, out, False) + 2 * out + 2 * _lin(out, pi, False) + _lin(pi, out, False))
        pre["factor"] = p * mk
        post = ((h * mk * mk, out), (out, out), (out, 2 * pi), (pi, out))
        return VisionSpec("glm", "GLM ViT", p, t, h, inter, L, nh, h, True, mk, post, params, tuple(sorted(pre.items())), b)
    if mt == "deepseek_v41_vision":
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["num_hidden_layers"], vc["num_attention_heads"]
        p, r = vc["patch_size"], vc["downsample_ratio"]
        params = (_lin(3 * p * p, h) + L * (2 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, 2 * inter, False)
                                            + _lin(inter, h, False)) + h
                  + _lin(r * r * h, text_h) + _lin(text_h, text_h) + 3 * text_h)
        pre.update(min_pixels=vc["min_pixels"], max_tok=vc["max_image_tokens"], ratio=r)
        return VisionSpec("ds", "DeepSeek ViT + aligner", p, 1, h, inter, L, nh, h, True, r,
                          ((r * r * h, text_h), (text_h, text_h)), params, tuple(sorted(pre.items())))
    if mt == "qwen3_moe_vit" and vc.get("disable_merger_proj"):   # 0.145 Ling-3.0-VL: Qwen3-VL ViT, merger = norm only
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"]
        p, t, mk = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
        hm = h * mk * mk
        # BailingMM linear_proj (mlp_depth 2): Linear(4·h → text_h), GELU, Linear(text_h → text_h), outside the tower
        params = (_lin(3 * t * p * p, h) + vc["num_position_embeddings"] * h
                  + L * (4 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + 2 * h + _lin(hm, text_h) + _lin(text_h, text_h))
        pre["factor"] = p * mk
        return VisionSpec("qwen", "Qwen3-VL ViT + linear_proj", p, t, h, inter, L, nh, h, False, mk,
                          ((hm, text_h), (text_h, text_h)), params, tuple(sorted(pre.items())))
    if "fullatt_block_indexes" in vc and vc.get("num_key_value_heads"):   # 0.144 MiMo-V2.6 ViT (Qwen2.5-VL style)
        h, inter, L, nh, kv = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"], vc["num_key_value_heads"]
        p, t, mk, out = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"], vc["out_hidden_size"]
        dh = vc.get("head_dim") or 64                      # qkv [3072, 1280] = (32 + 2·8) × 64 in the released header
        qd, kvd = nh * dh, kv * dh
        n_full = len(vc["fullatt_block_indexes"])
        hm = h * mk * mk
        params = (3 * t * p * p * h + L * (2 * h + _lin(h, qd + 2 * kvd) + _lin(qd, h) + 2 * _lin(h, inter) + _lin(inter, h))
                  + (L - n_full) * nh * (1 if vc.get("use_sink") else 0) + h + _lin(hm, hm, False) + _lin(hm, out, False))
        pre["factor"] = p * mk
        types = list(vc.get("vit_window_attn_types") or [-1] * L)
        perm = sum(1 for i in range(L) if i and (types[i] == 1) != (types[i - 1] == 1)) + (types[0] == 1)
        return VisionSpec("qwen", "MiMo ViT（GQA + 带状窗口注意力 + sink）", p, t, h, inter, L, nh, qd, True, mk,
                          ((hm, hm), (hm, out)), params, tuple(sorted(pre.items())), kv_heads=kv,
                          win=vc.get("visual_token_window_size") or 0, full_idx=tuple(vc["fullatt_block_indexes"]),
                          sink=bool(vc.get("use_sink")), perm=perm)
    if mt == "clip_vision_model" and cfg.get("model_type") == "minimax_m3_vl":    # 0.355 MiniMax M3.1-Flash
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["num_hidden_layers"], vc["num_attention_heads"]
        p, img_size = vc["patch_size"], vc["image_size"]
        mk_s = cfg.get("img_token_compression_config", {}).get("spatial_merge_size", 2)
        mk_t = cfg.get("img_token_compression_config", {}).get("temporal_patch_size", 2)
        mk = mk_s  # spatial merge
        proj_h = cfg.get("projector_hidden_size", text_h)
        # Conv2d patch embed + 3D RoPE + LN blocks (qkv / proj, GELU MLP) + LN + projector (h → proj_h, bias) + patch_merge MLP
        hm = h * mk * mk
        params = (3 * p * p * h + L * (2 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + h + _lin(h, proj_h) + _lin(hm, hm) + _lin(hm, hm))
        pre["factor"] = p * mk
        return VisionSpec("minimax", "MiniMax ViT (3D RoPE) + patch merger", p, mk_t, h, inter, L, nh, h, False, mk,
                          ((h, proj_h), (hm, hm), (hm, hm)), params, tuple(sorted(pre.items())))
    if mt == "muse_spark_vision":                         # 0.174 vLLM PR #61879 vision.py NaViT
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["num_hidden_layers"], vc["num_attention_heads"]
        p, t, mk = vc["patch_size"], vc["patch_temporal"], vc["merge_size"]
        hm, ph = h * mk * mk, cfg["projector_hidden_size"]
        # conv1_linear (no bias) + pos-emb grid + ln_pre/ln_post + blocks (LN×2, biased qkv/out/c_fc/c_proj, GELU MLP)
        # + NaViTAdapter c_fc / c_proj (no bias)
        params = (3 * t * p * p * h + vc["pos_emb_height"] * vc["pos_emb_width"] * h + 4 * h
                  + L * (4 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + _lin(hm, ph, False) + _lin(ph, ph, False))
        pre["factor"] = p * mk
        lt = list(vc.get("layer_types") or ["full_attention"] * L)
        full = tuple(i for i, x in enumerate(lt) if x != "window_attention")
        tile = vc["pos_emb_height"] if len(full) < L else 0
        au = audio_spec(cfg)
        return VisionSpec("qwen", "Muse Spark NaViT (Perception Encoder)", p, t, h, inter, L, nh, h, False, mk,
                          ((hm, ph), (ph, ph)), params, tuple(sorted(pre.items())), full_idx=full, tile=tile,
                          perm=2 if tile else 0, audio=au)
    if vc.get("model_type") == "perception_encoder":          # 0.144 Step-5 PE + 2× stride-2 3×3 downsamplers
        h, L, nh, p, side = vc["width"], vc["layers"], vc["heads"], vc["patch_size"], vc["image_size"]
        inter = 8960 if h == 1536 else int(h * vc.get("mlp_ratio", 8960 / 1536))
        g = side // p
        params = (3 * p * p * h + g * g * h + 2 * h
                  + L * (4 * h + 2 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + _lin(9 * h, 2 * h) + _lin(9 * 2 * h, 4 * h) + _lin(4 * h, text_h, False))
        return VisionSpec("fixed", "Perception Encoder + 下采样卷积", p, 1, h, inter, L, nh, h, False, 4,
                          ((9 * h, 2 * h, 4), (9 * 2 * h, 4 * h, 16), (4 * h, text_h, 16)), params,
                          tuple(sorted(pre.items())),
                          approx="视觉：裁剪规划取自同厂同塔 Step-3.7-Flash 的 transformers Step3p7ImageProcessor（728² 概览 + 504² 滑窗裁剪，"
                                 "「来源」代码）；Step-5-Preview 未随附处理器，套用于 Step-5 为「假设」（vision_config / 169 / 81 / "
                                 "image_token_id 全同，inference/validate_vision.py 以 configuration_step3p7 加载参考编码器）")
    if mt == "radio" and cfg.get("model_type") == "nemotron_h_omni":
        # 0.278 C-RADIO ViT-H/16 (modeling_radio.py 「来源」): conv-free patch embedder (Linear 768 → h, no bias) + CPE
        # position grid (16384 × h) + 10 cls / register tokens, 32 pre-LN blocks (biased qkv / proj, GELU MLP 4h); then
        # (MTP releases) a final LayerNorm, 2×2 pixel shuffle (downsample_ratio 0.5) and mlp1 = RMSNorm(4h) → Linear
        # 4h → projector_hidden_size → ReLU² → Linear → text h (no bias) 「来源」 modeling_nemotron_h_omni.py
        h, L, nh, p = vc["hidden_size"], vc["num_hidden_layers"], vc["num_attention_heads"], vc["patch_size"]
        inter = int(h * vc.get("mlp_ratio", 4.0))
        r = int(round(1 / cfg["downsample_ratio"]))
        ph_ = cfg["projector_hidden_size"]
        ncls = (vc.get("num_cls_tokens") or 0) + (vc.get("num_registers") or 0)
        grid = (vc.get("cpe_max_size") or 2048) // p
        fln = 2 * h if (tc_ := cfg.get("llm_config") or {}).get("num_nextn_predict_layers") else 0
        video = 3 * (vc.get("video_temporal_patch_size") or 1) * p * p * h
        params = (3 * p * p * h + grid * grid * h + ncls * h
                  + L * (4 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + fln + r * r * h + _lin(r * r * h, ph_, False) + _lin(ph_, text_h or tc_.get("hidden_size"), False))
        pre.update(patch=p, ratio=r)
        out = text_h or tc_.get("hidden_size")
        return VisionSpec("radio", "C-RADIO ViT-H/16（动态分辨率单图）+ 2×2 像素重排 + MLP 投影", p, 1, h, inter, L, nh, h,
                          False, r, ((r * r * h, ph_), (ph_, out)), params, tuple(sorted(pre.items())), extra=video,
                          approx="视觉：10 个 cls / register token 未计入注意力长度（≥ 1024 个 patch，< 1 %）「近似」；"
                                 "视频 tubelet 的 video_embedder（2 帧 patch）按单帧 patch 嵌入计「近似」，EVS 剪枝按保留数公式计")
    if vc.get("model_type") == "minicpmv4_7_vision":          # 0.144 SigLIP + in-tower merger (insert_layer_id) + merger
        h, inter, L, nh, p = vc["hidden_size"], vc["intermediate_size"], vc["num_hidden_layers"], vc["num_attention_heads"], vc["patch_size"]
        ins = cfg.get("insert_layer_id") or 0
        g = vc["image_size"] // p
        h4 = 4 * h
        mi = 17216 if h == 1152 else 4 * inter        # vit_merger MLP width (released header)
        params = (_lin(3 * p * p, h) + g * g * h + L * (4 * h + 4 * _lin(h, h) + _lin(h, inter) + _lin(inter, h)) + 2 * h
                  + 2 * h + 4 * _lin(h, h) + 2 * h4 + _lin(h4, mi) + _lin(mi, h)
                  + 2 * h4 + _lin(h4, h4) + _lin(h4, text_h))
        pre["patch"] = p
        return VisionSpec("minicpm", "SigLIP + 塔内 2×2 合并 + 2×2 merger（概览 + 切片）", p, 1, h, inter, L, nh, h, False, 4,
                          ((h4, h4, 16), (h4, text_h, 16)), params, tuple(sorted(pre.items())), insert=ins,
                          mid=((h, 3 * h, 1), (h, h, 1), (h4, mi, 4), (mi, h, 4)))
    return None


# ------------------------------------------------------------------ image → patch grid / LLM tokens
def _muse_resize(h: int, w: int, f: int, max_tok: int) -> tuple[int, int]:
    """0.255 transformers muse_glimmer image_processing / video_processing ``smart_resize``, verbatim: the integer
    (f-px) grid closest to the input aspect ratio among floor/ceil of the ideal grid, scaled down to ≤ max_tok cells
    (no lower bound, no rounding to the nearest multiple); f = patch·merge so one cell = one merged token."""
    ih, iw = h / f, w / f
    ratio = iw / ih if ih > 0 else 1.0
    if ih * iw > max_tok:
        ih = (max_tok / ratio) ** 0.5
        iw = ih * ratio
    cand = [(a, b) for a in {math.floor(ih), math.ceil(ih)} for b in {math.floor(iw), math.ceil(iw)}
            if a >= 1 and b >= 1 and a * b <= max_tok]
    if not cand:
        cand = [(max(1, round(ih)), max(1, round(iw)))]
    # min() keeps the first of equal keys; set order is unspecified upstream, so ties break (a, b) ascending here
    a, b = min(sorted(cand), key=lambda g: abs(g[0] / g[1] - h / w))
    return a * f, b * f


def _qwen_resize(h: int, w: int, f: int, lo: int, hi: int) -> tuple[int, int]:
    hb, wb = round(h / f) * f, round(w / f) * f
    if hb * wb > hi:
        beta = math.sqrt(h * w / hi)
        hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
    elif hb * wb < lo:
        beta = math.sqrt(lo / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return hb, wb


def _glm_resize(h: int, w: int, f: int, lo_tok: int, hi_tok: int, t: int = 2) -> tuple[int, int]:
    ppt = t * f * f
    lo, hi = lo_tok * ppt, hi_tok * ppt
    al = lambda v: math.ceil(v / f) * f          # noqa: E731
    ah, aw = al(h), al(w)
    if t * ah * aw < lo:
        s = math.sqrt(lo / (t * h * w))
        ah, aw = al(max(1, math.ceil(h * s))), al(max(1, math.ceil(w * s)))
    if t * ah * aw > hi:
        low, high, best = 1, h, (f, f)
        while low <= high:
            ch = (low + high) // 2
            cw = max(1, math.floor(w * ch / h))
            if t * al(ch) * al(cw) <= hi:
                best, low = (al(ch), al(cw)), ch + 1
            else:
                high = ch - 1
        ah, aw = best
    return ah, aw


def _kimi_resize(h: int, w: int, p: int, mk: int, lim: int, side: int) -> tuple[int, int]:
    s1 = math.sqrt(lim / (max(1.0, w // p) * max(1.0, h // p)))
    s = min(1.0, s1, side * p / w, side * p / h)
    nw, nh = min(max(1, int(w * s)), side * p), min(max(1, int(h * s)), side * p)
    f = mk * p
    return nh + (f - nh % f) % f, nw + (f - nw % f) % f


def _ds_grid(h: int, w: int, p: int, r: int, min_px: int, max_tok: int) -> tuple[int, int, int, int]:
    if 0 < w * h < min_px:
        ratio = (min_px / (w * h)) ** 0.5
        w, h = int(w * ratio), int(h * ratio)
    bw, bh = math.ceil(w / p) * p, math.ceil(h / p) * p
    grid = lambda bh_, bw_: (math.ceil((bh_ // p) / r), math.ceil((bw_ // p) / r))   # noqa: E731
    ntok = lambda gh, gw: gh * (gw + 1) + 2                                            # noqa: E731
    gh, gw = grid(bh, bw)
    if ntok(gh, gw) > max_tok:
        ar = h / w
        mw = math.sqrt((max_tok - 2) / ar + 0.25) - 0.5
        mh = mw * ar
        cell = p * r
        if mw < 1.0:
            bh, bw = (max_tok - 2) // 2 * cell, cell
        elif mh < 1.0:
            bh, bw = cell, (max_tok - 3) * cell
        else:
            beta = min(math.floor(mw) * cell / w, math.floor(mh) * cell / h)
            bh, bw = math.floor(h * beta / p) * p, math.floor(w * beta / p) * p
        gh, gw = grid(bh, bw)
    return bh // p, bw // p, gh, gw


# 0.145 MiniCPM-V-4.x slicing (transformers MiniCPMV4_6ImageProcessor, which minicpmv4_7 maps to)
def _ensure_divide(length: float, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def _cpm_best_resize(h: float, w: float, side: int, p: int, upscale: bool) -> tuple[int, int]:
    if h * w > side * side or upscale:
        ar = w / h
        h = int(side / math.sqrt(ar))
        w = int(h * ar)
    return _ensure_divide(h, p * 4), _ensure_divide(w, p * 4)


def _cpm_grid(h: int, w: int, max_slices: int, side: int) -> tuple[int, int] | None:
    log_ratio = math.log(w / h)
    multiple = min(math.ceil(w * h / (side * side)), max_slices)
    if multiple <= 1:
        return None
    best, err_min = (1, 1), float("inf")
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > max_slices:
            continue
        for r in range(1, n + 1):
            if n % r == 0:
                err = abs(log_ratio - math.log((n // r) / r))
                if err < err_min or (err == err_min and r > best[0]):
                    best, err_min = (r, n // r), err
    return best


def step_crops(width: int, height: int, side: int, crop: int, max_img: int) -> tuple[int, int]:
    """0.255 transformers step3p7/image_processing_step3p7.py Step3p7ImageProcessor._plan_patches, verbatim → (crops
    along width, along height); (0, 0) = overview only.  Extreme aspect (min side < 32 and ratio > 4) pads to a square;
    images over max_img on the long side scale down; window = short side if long ≤ side and ratio > 1.5 (else none),
    min(short, crop) if ratio > 4, else crop; each side snaps to a window multiple (up when the remainder > 20 %)."""
    if min(width, height) < 32 and max(width / height, height / width) > 4:
        width = height = max(width, height)
    if max(height, width) > max_img:
        sc = max_img / max(height, width)
        width, height = int(width * sc), int(height * sc)
    short, long_ = min(height, width), max(height, width)
    if long_ <= side:
        win = short if long_ / short > 1.5 else 0
    elif long_ / short > 4:
        win = min(short, crop)
    else:
        win = crop
    if win == 0:
        return 0, 0
    cw, ch = ((win * (d // win + (d % win > 0.2 * win)) if d >= win else d) for d in (width, height))
    return max(1, cw // win), max(1, ch // win)


def minicpm_views(width: int, height: int, side: int, p: int, max_slices: int):
    """(overview (ph, pw), slice (ph, pw) or None, (rows, cols)) as MiniCPMV4_6ImageProcessor._preprocess."""
    grid = _cpm_grid(height, width, max_slices, side)
    sh, sw = _cpm_best_resize(height, width, side, p, upscale=grid is None)
    if grid is None:
        return (sh // p, sw // p), None, (0, 0)
    gy, gx = grid
    rh, rw = _ensure_divide(height, gy), _ensure_divide(width, gx)
    bh, bw = _cpm_best_resize(rh / gy, rw / gx, side, p, upscale=True)
    return (sh // p, sw // p), (bh // p, bw // p), (gy, gx)


@dataclass(frozen=True)
class ImageGrid:
    ph: int                     # patch rows / cols into the tower
    pw: int
    merged: int                 # positions out of the merger / aligner
    llm_tokens: int             # positions the image occupies in the LLM sequence
    px: tuple[int, int]         # (width, height) the image is resized / padded to
    views: tuple = ()           # 0.145: ((ph, pw, count), …) when one image runs as several tower passes (MiniCPM)
    tframes: int = 1            # 0.166: frames attended jointly in one pass (Kimi video chunk: spatial-temporal
                                # attention over tframes·ph·pw, temporal mean before the merger)
    pad_soft: int = 0           # 0.382: video max_soft_tokens when it differs from the image pad (0 = image pad)


def image_grid(v: VisionSpec, width: int, height: int) -> ImageGrid:
    if v.family == "audio":                       # 0.268 audio-only encoder: no image path
        return ImageGrid(0, 0, 0, 0, (0, 0))
    pr = v.rz
    if v.family == "ernievl":
        # Merged patches plus IMAGE_START and IMAGE_END. Both markers are token type image.
        hb, wb = _ernie_smart_resize(height, width, pr["factor"], pr["min_px"], pr["max_px"])
        ph, pw = hb // v.patch, wb // v.patch
        m = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, m, m + 2, (wb, hb))
    if v.family in ("gemma4", "gemma4u"):
        # LLM tokens = real soft tokens after the pad is stripped. Tower GEMMs use the pad (vision_ops).
        max_patches = pr["max_soft"] * pr["pool"] * pr["pool"]
        th, tw = _gemma4_resize(height, width, pr["patch"], max_patches, pr["pool"])
        ph, pw = th // pr["patch"], tw // pr["patch"]
        soft = ph * pw // (pr["pool"] * pr["pool"])
        return ImageGrid(ph, pw, soft, soft, (tw, th))
    if v.family == "qwen" and "muse_tok" in pr:
        # 0.255 MuseSparkProcessor.replace_image_token: <|image_start|> + merged <|patch|> tokens + <|image_end|>
        hb, wb = _muse_resize(height, width, pr["factor"], pr["muse_tok"])
        ph, pw = hb // v.patch, wb // v.patch
        m = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, m, m + 2, (wb, hb))
    if v.family == "qwen":
        hb, wb = _qwen_resize(height, width, pr["factor"], pr["min_px"], pr["max_px"])
        ph, pw = hb // v.patch, wb // v.patch
        m = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, m, m, (wb, hb))
    if v.family == "glm":
        hb, wb = _glm_resize(height, width, pr["factor"], pr["min_tok"], pr["max_tok"], v.tpatch)
        ph, pw = hb // v.patch, wb // v.patch
        m = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, m, m, (wb, hb))
    if v.family == "kimi":
        hb, wb = _kimi_resize(height, width, v.patch, v.merge, pr["in_patch_limit"], pr["side"])
        ph, pw = hb // v.patch, wb // v.patch
        m = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, m, m, (wb, hb))
    if v.family == "fixed":                       # 0.144: one square view resized to ``side`` (overview only)
        ph = pw = pr["side"] // v.patch
        m = ph * pw // (v.merge * v.merge)
        if "crop" not in pr:
            return ImageGrid(ph, pw, m, m, (pr["side"], pr["side"]))
        # 0.255 Step3p7Processor.replace_image_token: per crop <patch_start> 81 <patch_end>, <patch_newline> after each
        # crop row but the last, then <im_start> 169 <im_end>
        nx, ny = step_crops(width, height, pr["side"], pr["crop"], pr["max_img"])
        cp = pr["crop"] // v.patch
        mc = cp * cp // (v.merge * v.merge)
        n = nx * ny
        views = ((ph, pw, 1),) + (((cp, cp, n),) if n else ())
        tok = m + 2 + n * (mc + 2) + (ny - 1 if n else 0)
        return ImageGrid(ph, pw, m + n * mc, tok, (pr["side"], pr["side"]), views)
    if v.family == "minicpm":
        # 0.145: overview + rows × cols slices; tokens = patches / 16 per view (MiniCPMV4_7Processor.replace_image_token)
        # + <image>/</image> + per-slice <slice>/</slice> + "\n" between slice rows + <image_id>0</image_id> (3,
        # use_image_id default True; the index renders as one token 「假设」 for < 10 images)
        (oh, ow), sl, (r, c) = minicpm_views(width, height, pr["side"], v.patch, pr["max_slice"])
        to = oh * ow // 16
        views = ((oh, ow, 1),)
        tok = 2 + to + 3
        merged = to
        if sl:
            ts = sl[0] * sl[1] // 16
            views += ((sl[0], sl[1], r * c),)
            tok += r * c * (ts + 2) + (r - 1)
            merged += r * c * ts
        return ImageGrid(oh, ow, merged, tok, (ow * v.patch, oh * v.patch), views)
    if v.family == "radio":
        # 0.278 one dynamic-resolution tile per image (NemotronH_Omni_Reasoning_V3ImageProcessor); prompt = <img> + tokens
        # + </img> (processing_nemotron_h_omni.py)
        wp, hp = radio_patches(width, height, pr["patch"], pr["ratio"], pr["min_patches"], pr["max_patches"], pr["max_len"])
        m = hp * wp // (pr["ratio"] ** 2)
        return ImageGrid(hp, wp, m, m + 2, (wp * pr["patch"], hp * pr["patch"]))
    if v.family == "lfm2vl":
        return _lfm2_vl_image_grid(v, width, height)
    if v.family == "d1omni":
        return _d1_image_grid(v, width, height)
    if v.family == "longcatvit":
        # Projector pairs width only. merged is ph * pw / 2, not a square merge.
        factor = pr["factor"]
        hb, wb = _longcat_resize(height, width, factor, pr["min_px"], pr["max_px"])
        ph, pw = hb // v.patch, wb // v.patch
        if pw % 2:
            raise ValueError("longcat omni patch width is not even")
        m = ph * pw // 2
        return ImageGrid(ph, pw, m, m, (wb, hb))
    if v.family == "pixtral":
        # PixtralProcessor.replace_image_token: one break/end token per merged row.
        factor = pr["factor"]
        hb, wb = _pixtral_resize(height, width, pr["longest"], factor)
        ph, pw = hb // v.patch, wb // v.patch
        rows, cols = ph // v.merge, pw // v.merge
        m = rows * cols
        return ImageGrid(ph, pw, m, (cols + 1) * rows, (wb, hb))
    ph, pw, gh, gw = _ds_grid(height, width, v.patch, pr["ratio"], pr["min_pixels"], pr["max_tok"])
    return ImageGrid(ph, pw, gh * gw, gh * (gw + 1) + 2, (pw * v.patch, ph * v.patch))


def radio_patches(width: int, height: int, p: int, r: int, lo: int, hi: int, max_len: int, n_images: int = 1) -> tuple[int, int]:
    """0.278 NemotronH_Omni_Reasoning_V3ImageProcessor._compute_target_patches (verbatim rule 「来源」): budget = max(min((
    max_model_len − 4) · r², max_num_patches), min_num_patches); scale the ⌊side / p + ½⌉ grid down to the budget, up to
    min_num_patches, then round each side to a multiple of r within the budget.  → (w_patches, h_patches)."""
    budget = max((max_len - 4) * r * r, lo * n_images)
    avail = max(min(budget, hi if hi and hi > 0 else budget), lo)
    ch, cw = round(height / p + 0.5), round(width / p + 0.5)
    f = min(math.sqrt(avail / (ch * cw)), 1.0)
    th, tw = math.floor(f * ch), math.floor(f * cw)
    if avail > lo and th * tw < lo:
        up = math.sqrt(lo / (th * tw))
        th, tw = math.ceil(up * th), math.ceil(up * tw)
    rh = th % r
    if rh:
        th = th + (r - rh) if (th + r - rh) * tw <= avail else max(r, th - rh)
    rw = tw % r
    if rw:
        tw = tw + (r - rw) if th * (tw + r - rw) <= avail else max(r, tw - rw)
    return tw, th


# ------------------------------------------------------------------ op graph
def tile_keys(ph: int, pw: int, t: int) -> int:
    """Σ over tiles of (patches in tile)² for a ph × pw grid cut into t × t tiles (edge tiles partial)."""
    rows = [min(t, ph - i) for i in range(0, ph, t)]
    cols = [min(t, pw - j) for j in range(0, pw, t)]
    return sum((r * c) ** 2 for r in rows for c in cols)


def band_keys(n: int, w: int) -> int:
    """Σ over queries of unmasked keys under |i − j| ≤ w in a sequence of n (w ≤ 0: full n²)."""
    if w <= 0 or w >= n - 1:
        return n * n
    return n + 2 * (w * n - w * (w + 1) // 2)


def vision_ops(v: VisionSpec, g: ImageGrid, images: int, af: str = "bf16") -> list[Op]:
    """Op list of one encoder pass over ``images`` images of grid ``g`` (bf16 weights).  0.145: an image of several
    views (MiniCPM overview + slices) runs one tower pass per view group (ops of later groups are named vis.v<i>.…);
    MiMo windowed blocks charge only the band's keys (``band_keys``)."""
    if images <= 0:
        return []
    if v.family == "longcatvit":
        # The tower .pt files are bf16 and are not in the language fp8 ignore list.
        af = "bf16"
    if v.family in ("gemma4", "gemma4u"):
        return _gemma4_vision_ops(v, g, images, af)
    if v.family == "lfm2vl":
        return _lfm2_vl_vision_ops(v, g, images, af)
    if v.family == "d1omni":
        return _d1_vision_ops(v, g, images, af)
    if v.family == "ernievl" and g.tframes > 1:
        return _ernie_vl_video_ops(v, g, images, af)
    if g.tframes > 1:
        # 0.166 Kimi chunk (modeling_kimi_k25): patch-embed + tower over every frame's patches with one cu_seqlens
        # span of t·h·w (spatial_temporal); tpool_patch_merger means over t, then the merger runs on h/2·w/2 rows
        n = g.ph * g.pw
        ops = _tower_ops(v, g.tframes * n, images, images * g.merged, "vis.", af)
        from .dtypes import fmt as _fmt
        i = next(k for k, o in enumerate(ops) if o.name.startswith("vis.merger"))
        ops.insert(i, Op("vis.tpool", "vector", v.depth, vec=images * g.tframes * n * v.hidden,
                         act_bytes=images * (g.tframes + 1) * n * v.hidden * _fmt(af).bytes))
        return ops
    groups = g.views or ((g.ph, g.pw, 1),)
    ops: list[Op] = []
    for gi, (ph, pw, cnt) in enumerate(groups):
        mrows = images * g.merged if len(groups) == 1 else None
        ops += _tower_ops(v, ph * pw, images * cnt, mrows, "vis." if gi == 0 else f"vis.v{gi}.", af, (ph, pw))
    return ops


def _tower_ops(v: VisionSpec, N: int, images: int, mrows: int | None, pfx: str, af: str,
               grid: tuple[int, int] | None = None) -> list[Op]:
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    N0 = N
    rows = images * N
    wf, wb = "bf16", 16.0
    ops: list[Op] = []
    qf = {k: (f, b, a) for k, f, b, a in v.wfmt}       # 0.285 quantized tower: per-kind formats (empty = bf16)

    def kind(name: str) -> str:
        return ("patch" if name == "patch_embed" else "merger" if name.startswith(("merger", "vit_merger")) else
                "attn" if name.endswith((".qkv", ".o")) else "mlp")

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        f, b, a = qf.get(kind(name), (wf, wb, af)) if qf else (wf, wb, af)
        return Op(pfx + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=b, w_fmt=f,
                  a_fmt=a, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    def attn(tag: str, li: int, N: int, keys: int, sink: bool) -> list[Op]:
        fr = keys / (N * N)
        return [Op(f"{pfx}{tag}.qk", "attn", li, m=N, k=dh, n=N, count=cnt, causal=fr, act_bytes=cnt * N * dh * ab,
                   stream=True, orient=True),
                Op(f"{pfx}{tag}.pv", "attn", li, m=N, k=N, n=dh, count=cnt, causal=fr, act_bytes=cnt * N * dh * ab,
                   orient=True),
                Op(f"{pfx}{tag}.softmax", "vector", li, vec=cnt * keys * (5.0 + (1.0 if sink else 0.0)))]

    ops.append(gemm("patch_embed", rows, 3 * v.tpatch * v.patch * v.patch, v.hidden))
    h, q, dh = v.hidden, v.qkv, v.head_dim
    cnt = images * v.heads
    qkv_n = (v.heads + 2 * v.kv_heads) * dh if v.kv_heads else 3 * q
    longcat = v.family == "longcatvit" and v.gated
    pixtral = v.family == "pixtral"
    q3 = v.ln_mode == "qwen3vl"
    if q3:
        # Qwen3VLVisionBlock LayerNorms and Qwen3VLVisionPatchMerger nn.GELU().
        # The block MLP stays gelu_pytorch_tanh. Merger fc1 is Linear(4h, 4h).
        if v.vec_fmt not in ("fp16", "bf16", "fp32"):
            raise ValueError("qwen3vl activation dtype is not a vision float")
        hm = h * v.merge * v.merge
        if (v.deepstack < 1 or len(v.ds_idx) != v.deepstack or len(set(v.ds_idx)) != v.deepstack
                or len(v.post) != 2 * (1 + v.deepstack)):
            raise ValueError("qwen3vl deepstack mergers are missing")
        if any(i < 0 or i >= v.depth for i in v.ds_idx):
            raise ValueError("qwen3vl deepstack index is outside the tower")
        for i in range(0, len(v.post), 2):
            if v.post[i] != (hm, hm) or v.post[i + 1][0] != hm:
                raise ValueError("qwen3vl merger is not linear, gelu, linear")
    if pixtral:
        # PixtralVisionModel.ln_pre is PixtralRMSNorm on the flattened patches,
        # before the block loop. One rsqrt per patch token, width hidden.
        if v.vec_fmt not in ("fp16", "bf16", "fp32"):
            raise ValueError("pixtral activation dtype is not a vision-role float")
        if (not v.gated or len(v.post) != 3 or v.post[1][0] != h):
            raise ValueError("pixtral projector is not norm, merger, linear, gelu, linear")
        ops.append(_rms_vec(pfx + "ln_pre", 0, rows, h))
    if longcat and grid is not None:
        # VisionRotaryEmbedding2D(head_dim // 2). inv_freq is arange(0, dim, 2)
        # in fp32 and is not persistent. Encoder.forward calls forward_ once,
        # before the layer loop: torch.outer(seq, inv_freq). seqlen is
        # max(t, h, w). VisionProjector requires t == 1, so the side is
        # max(ph, pw). The table is shared across layers and across images of
        # this grid; it is not a GEMM and it has no transcendental. The later
        # pos_ids gather is not a separate op.
        ph, pw = grid
        dim = dh // 2
        inv_len = (dim + 1) // 2
        side = max(ph, pw)
        ops.append(Op(pfx + "rope_outer", "vector", 0, vec=side * inv_len, a_fmt="fp32"))

    def lc_rms(name: str, li: int, width: int, n_rows: int) -> Op:
        # RMSNorm.forward / the same 4-elem + 1 rsqrt recipe as attn_norm. fp32 upcast.
        # The weight is already in the tower parameter total.
        return Op(pfx + name, "vector", li, vec=n_rows * width * 4, sfu=float(n_rows), a_fmt="fp32")

    if v.perm:
        # apply_index gathers (row ↔ merge-unit column order): read + write the hidden states per switch
        ops.append(Op(pfx + "win_perm", "vector", 0, vec=v.perm * rows * h, act_bytes=v.perm * 2 * rows * h * ab))
    for li in range(v.depth):
        if v.insert and li == v.insert:
            # 0.144 MiniCPM-V-4.7 vit_merger: attention at full resolution, 2×2 concat, MLP down to the tower width
            for i, (k, n, dv) in enumerate(v.mid):
                ops.append(gemm(f"vit_merger.{i}", images * (N // dv), k, n, li))
            ops += attn("vit_merger", li, N, N * N, False)
            N, rows = N // 4, rows // 4
        if longcat:
            # LongCatVisionEncoderLayer: norm1, then attention, then ls1, then
            # norm2, SwiGLU, ls2. All four block RMSNorms reduce embed_dim (hidden),
            # one rsqrt per patch token. q_norm / k_norm flatten heads before the
            # norm, so the width is embed_dim, not head_dim, and the rsqrt count is
            # tokens rather than tokens * heads.
            ops.append(lc_rms(f"{li}.norm1", li, h, rows))
        if pixtral:
            # PixtralAttentionLayer: attention_norm, then q/k/v. Width is hidden.
            ops.append(_rms_vec(pfx + f"{li}.attn_rms", li, rows, h))
        if q3:
            # norm1 is LayerNorm(hidden) on patch tokens, before the fused qkv.
            ops.append(_ln_vec(pfx + f"{li}.attn_ln", li, rows, h))
        ops.append(gemm(f"{li}.qkv", rows, h, qkv_n, li))
        if pixtral:
            # Attention runs before o_proj. One bidirectional block per image.
            # The block mask stays inside this softmax. RoPE is not an op here.
            ops += attn(str(li), li, N, N * N, False)
        if longcat:
            # apply_rotary_pos_emb_vision upcasts q and k to fp32. freqs.cos and
            # freqs.sin run once per call. freqs is [tokens, head_dim/2] (the
            # h/w pair flattened), not per head and not embed_dim. The function
            # is called for q and again for k, so the trig is not reused.
            # The rotate-multiply is the same 3 element ops per full-head element
            # as the language RoPE vector. Softmax stays the bidirectional score
            # op below and is not given a second SFU.
            half = dh // 2
            ops.append(Op(pfx + f"{li}.rope", "vector", li, a_fmt="fp32",
                          vec=2 * rows * (v.heads * dh * 3 + half * 2),
                          sfu=2 * rows * half * 2))
            # cos/sin are [tokens, 1, head_dim/2]. repeat(1, 1, 2) materializes
            # [tokens, 1, head_dim] for each of them. q and k each call the
            # function, so the copy is not shared. It is fp32 and has no
            # transcendental. unsqueeze is a view and is not counted.
            ops.append(Op(pfx + f"{li}.rope_repeat", "vector", li, a_fmt="fp32",
                          vec=4 * rows * dh))
            ops.append(lc_rms(f"{li}.q_norm", li, h, rows))
            ops.append(lc_rms(f"{li}.k_norm", li, h, rows))
        ops.append(gemm(f"{li}.o", rows, q, h, li))
        if pixtral:
            # Residual add is the full branch, not 0.5. ffn_norm then SwiGLU.
            ops.append(_rms_vec(pfx + f"{li}.ffn_rms", li, rows, h))
        if q3:
            # norm2 is LayerNorm(hidden) before the MLP. The residual add is the full branch.
            ops.append(_ln_vec(pfx + f"{li}.mlp_ln", li, rows, h))
        if longcat:
            # ls1 / ls2 are Parameter(embed_dim): branch * scale, then the residual add.
            # The class does not upcast. No rsqrt.
            ops.append(Op(pfx + f"{li}.ls1", "vector", li, vec=rows * h * 2, a_fmt="bf16"))
            ops.append(lc_rms(f"{li}.norm2", li, h, rows))
        ops.append(gemm(f"{li}.mlp_in", rows, h, (2 if v.gated else 1) * v.inter, li))
        if pixtral:
            # PixtralMLP: down(act(gate) * up). ACT2FN["silu"] is x * sigmoid(x)
            # on the gate only, so the width is intermediate, not the fused 2x GEMM.
            # The product with up is one multiply. Neither upcasts.
            ops.append(_silu_vec(pfx + f"{li}.silu", li, rows * v.inter, v.vec_fmt))
            ops.append(Op(pfx + f"{li}.mlp_mul", "vector", li, vec=rows * v.inter, a_fmt=v.vec_fmt))
        if longcat:
            # SwiGLU.forward is fc2(RMSNorm(silu(fc1(x)) * fc3(x))). ACT2FN["silu"]
            # is x * sigmoid(x): the language-stack count, 4 element ops with exp
            # and reciprocal. It runs on fc1 only, so the width is the inner size,
            # not the fused 2×inner GEMM. The class does not upcast.
            # The product of that SiLU and fc3 is one multiply on the same width.
            # The norm then reduces the inner width. Rows are patch tokens, before
            # the width-pair projector.
            ops.append(Op(pfx + f"{li}.silu", "vector", li, vec=rows * v.inter * 4,
                          sfu=rows * v.inter * 2, a_fmt="bf16"))
            ops.append(Op(pfx + f"{li}.mlp_mul", "vector", li, vec=rows * v.inter, a_fmt="bf16"))
            ops.append(lc_rms(f"{li}.mlp_norm", li, v.inter, rows))
        ops.append(gemm(f"{li}.mlp_out", rows, v.inter, h, li))
        if longcat:
            ops.append(Op(pfx + f"{li}.ls2", "vector", li, vec=rows * h * 2, a_fmt="bf16"))
        if pixtral:
            continue
        if v.tile and grid is not None and li not in v.full_idx:
            ops += attn(str(li), li, N, tile_keys(grid[0], grid[1], v.tile), False)
            continue
        windowed = bool(v.win) and li not in v.full_idx
        ops += attn(str(li), li, N, band_keys(N, v.win) if windowed else N * N, windowed and v.sink)

    def merger_rows(kn: tuple) -> int:
        if len(kn) > 2:
            return images * (N0 // kn[2])
        return mrows if mrows is not None else images * (N0 // (v.merge * v.merge))

    if longcat and v.post:
        # VisionProjector.p0 starts with LayerNorm(hidden * 2) on the width-paired rows
        # (ph * pw / 2), not on patch tokens. fp32 reduction, one rsqrt per row.
        kn0 = v.post[0]
        ops.append(lc_rms("proj_ln", v.depth, kn0[0], merger_rows(kn0)))
    if pixtral:
        # Mistral3MultiModalProjector.norm runs on the patch sequence, before the
        # unfold. Width is hidden, not the text width, and the row count is not merged.
        ops.append(_rms_vec(pfx + "proj_rms", v.depth, images * N0, h))
    if q3:
        # Final merger, use_postshuffle_norm false: LayerNorm(hidden) on the patch
        # sequence, then a view to 4h. The row count is not the merged count.
        ops.append(_ln_vec(pfx + "pre_ln", v.depth, images * N0, h))
    for i, kn in enumerate(v.post):
        k, n = kn[:2]
        m_ = merger_rows(kn)
        if q3 and i % 2 == 0 and i > 0:
            # Deepstack merger, use_postshuffle_norm true: view to 4h, then LayerNorm(4h).
            slot = i // 2 - 1
            ops.append(_ln_vec(pfx + f"ds{slot}.ln", v.ds_idx[slot], m_, k))
        ops.append(gemm(f"merger.{i}", m_, k, n, v.depth))
        if q3 and i % 2 == 0:
            # nn.GELU() default approximate is 'none': the erf form. Width is fc1's N (4h).
            # The block MLP's gelu_pytorch_tanh is not this op.
            if i == 0:
                ops.append(_gelu_erf_vec(pfx + "mgelu", v.depth, m_ * n, v.vec_fmt))
            else:
                slot = i // 2 - 1
                ops.append(_gelu_erf_vec(pfx + f"ds{slot}.gelu", v.ds_idx[slot], m_ * n, v.vec_fmt))
        if pixtral and i == 1:
            # ACT2FN["gelu"] after linear_1, before linear_2. Erf form. Rows are
            # the merged count. Width is linear_1's N (the text hidden size).
            ops.append(_gelu_erf_vec(pfx + "proj_gelu", v.depth, m_ * n, v.vec_fmt))
        if longcat and i < 2:
            # VisionProjector.p0 is LayerNorm, Linear, GELU, Linear, GELU.
            # nn.GELU() default is the erf form, not tanh:
            # 0.5 * x * (1 + erf(x / sqrt(2))). Four arithmetic ops and one erf.
            # The class does not upcast. Rows are the width-paired count, and the
            # width is adaptor_dim. The final proj linear has no GELU.
            ops.append(Op(pfx + f"proj_gelu{i}", "vector", v.depth,
                          vec=m_ * n * 5, sfu=float(m_ * n), a_fmt="bf16"))
    if v.deepstack and v.post:
        # Qwen3VLTextModel._deepstack_process: after each of the first len(indexes) decoder layers, add that
        # merger's output onto visual-token rows.  Decode positions are not visual, so this is prefill-only.
        # The reference clones the whole hidden state; the algorithm is the indexed add (read hidden, read
        # feature, write hidden).  Billed on the encoder pass that produced the features (full width, once).
        # 0.367: stream=True.  act_stream returns act_bytes with no SRAM absorb — later LLM layers consume the
        # feature, so the round trip is DRAM even when the bytes fit the activation half.  vision_act_peak
        # stays GEMM-only.
        out_h = v.post[-1][1]
        m_ = mrows if mrows is not None else images * (N0 // (v.merge * v.merge))
        ops.append(Op(pfx + "deepstack", "vector", 0, vec=v.deepstack * m_ * out_h,
                      act_bytes=v.deepstack * 3 * m_ * out_h * ab, stream=True))
    return ops


def _ernie_vl_video_ops(v: VisionSpec, g: ImageGrid, images: int, af: str) -> list[Op]:
    """ERNIE-4.5-VL video resampler (VariableResolutionResamplerModel).

    ``vision_forward`` expands ``grid_thw`` to one ``(1, h, w)`` row per frame, so
    attention stays inside a frame and the tower GEMMs see every frame. Spatial
    linears run on every frame's merged rows. The temporal linear concatenates each
    pair on the feature axis (K = 2 · spatial) and the GEMMs after that see half as
    many rows. ``VIDEO_START`` / ``VIDEO_END`` are LLM tokens, not merger rows.
    The ``index_select`` that builds the pairs is not billed.
    """
    if g.tframes < 2 or g.tframes % 2:
        raise ValueError("ernie4_5_vl video frames must be a positive even count")
    if not v.post:
        raise ValueError("ernie4_5_vl video resampler is missing")
    sd = v.post[0][0]
    wide = next((i for i, kn in enumerate(v.post) if kn[0] == sd * 2), None)
    if wide is None:
        raise ValueError("ernie4_5_vl video temporal linear is missing")
    ops = _tower_ops(replace(v, post=()), g.ph * g.pw, images * g.tframes, None, "vis.", af)
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    m_sp = images * g.tframes * g.merged
    m_tm = images * (g.tframes // 2) * g.merged
    for i, kn in enumerate(v.post):
        k, n = kn[:2]
        m_ = m_sp if i < wide else m_tm
        ops.append(Op(f"vis.merger.{i}", "gemm", v.depth, m=m_, k=k, n=n, role="vision",
                      w_params=k * n, w_bits=16.0, w_fmt="bf16", a_fmt=af,
                      act_bytes=m_ * (k + n) * ab, stream=True, vec=m_ * n * VEC_PER_OUT))
    return ops


def vision_flops(v: VisionSpec, g: ImageGrid, images: int = 1) -> float:
    return sum(o.flops for o in vision_ops(v, g, images))


def vision_act_peak(v: VisionSpec, g: ImageGrid, images: int, af: str = "bf16") -> float:
    """Largest single-op activation in+out (bytes) of an encoder pass — its DRAM working set."""
    return max((o.act_bytes for o in vision_ops(v, g, images, af) if o.kind == "gemm"), default=0.0)


# ------------------------------------------------------------------ scenario expansion
def image_grid_of(m, sv) -> ImageGrid | None:
    v = getattr(m, "vision", None)
    return image_grid(v, sv.image_w, sv.image_h) if v is not None and v.family != "audio" and sv.images else None


def expand_images(scn, m):
    """The scenario with each request's image tokens added to serving.prompt and serving.ctx (idempotent: marks
    serving.image_tokens).  Everything downstream (prefill length, KV, PD lengths / capacity, goodput) then sees the
    image positions as prompt tokens; the encoder itself is costed by evaluate at prefill."""
    sv = scn.serving
    if not (sv.images or getattr(sv, "videos", 0) or getattr(sv, "audio_s", 0)) or sv.image_tokens:
        return scn
    g = image_grid_of(m, sv)
    vg = video_grid_of(m, sv) if getattr(m, "vision", None) is not None else None
    au = getattr(getattr(m, "vision", None), "audio", None) if getattr(sv, "audio_s", 0) else None
    if g is None and vg is None and au is None:
        return scn
    import dataclasses
    t = (sv.images * g.llm_tokens if g is not None else 0) + (sv.videos * vg.llm_tokens if vg is not None else 0) \
        + (au.prompt_tokens(sv.audio_s) if au is not None else 0)
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt + t, ctx=sv.ctx + t, image_tokens=t))


def video_pre(m) -> dict | None:
    """Video processor settings of a model with a modelled video path (Qwen3VLVideoProcessor; 0.166 Kimi chunks;
    0.255 MuseGlimmerVideoProcessor; 0.382 Gemma4VideoProcessor / Gemma4UnifiedVideoProcessor;
    0.385 Ernie4_5_VLProcessor)."""
    v = getattr(m, "vision", None)
    pre = _VIDEO.get(getattr(m, "hf_id", "")) if v is not None else None
    if pre is None:
        return None
    if "gemma4_soft" in pre:
        if v.family not in ("gemma4", "gemma4u"):
            return None
        return pre
    if "ernie_temporal" in pre:
        if v.family != "ernievl":
            return None
        return pre
    if v.family != ("kimi" if "chunk" in pre else "radio" if "radio_frame" in pre else "qwen"):
        return None
    return pre


@dataclass(frozen=True)
class VideoGrid:
    frames: int                 # sampled frames into the processor
    t: int                      # temporal slices = ⌈frames / temporal_patch⌉ (frames padded to a multiple)
    slice: ImageGrid            # one temporal slice (the tower attends within a slice: cu_seqlens per t)
    llm_tokens: int             # t · merged positions in the LLM sequence (timestamp text tokens not counted)
    tail: ImageGrid | None = None   # 0.166 Kimi: shorter last chunk (F mod 4 frames), one of the t passes


def video_grid(v: VisionSpec, pre: dict, frames: int, width: int, height: int, cap: bool = False) -> VideoGrid:
    """transformers Qwen3VLVideoProcessor.resize + smart_resize, verbatim: the pixel budget bounds t_bar·h·w of the
    clip, so more frames → smaller frames.  cap = cap_pixels_per_frame (qwen-vl-utils: per-frame pixels ≤
    min(768·f², longest/F), floored at 1.05·shortest)."""
    if "chunk" in pre:
        return _kimi_video(v, pre, frames, width, height)
    if "muse_frame_tok" in pre:
        return _muse_video(v, pre, frames, width, height)
    if "radio_frame" in pre:
        return _radio_video(v, pre, frames, width, height)
    if "gemma4_soft" in pre:
        return _gemma4_video(v, pre, frames, width, height)
    if "ernie_temporal" in pre:
        return _ernie_video(v, pre, frames, width, height)
    tp, f = max(1, v.tpatch), v.patch * v.merge
    lo, hi = pre["min_px"], pre["max_px"]
    if frames < tp:
        raise ValueError(f"serving.video_frames must be ≥ temporal_patch_size {tp}")
    if cap:
        hi = max(min(VIDEO_MAX_TOK * f * f, hi // frames), int(lo * 1.05)) * frames
    h, w = height, width
    if h < f or w < f:
        sc = max(f / h, f / w)
        h, w = int(h * sc), int(w * sc)
    hb, wb = round(h / f) * f, round(w / f) * f
    tb = round(frames / tp) * tp
    if tb * hb * wb > hi:
        beta = math.sqrt(frames * h * w / hi)
        hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
    elif tb * hb * wb < lo:
        beta = math.sqrt(lo / (frames * h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    ph, pw = hb // v.patch, wb // v.patch
    mg = ph * pw // (v.merge * v.merge)
    t = -(-frames // tp)
    return VideoGrid(frames, t, ImageGrid(ph, pw, mg, mg, (wb, hb)), t * mg)


def _muse_video(v: VisionSpec, pre: dict, frames: int, width: int, height: int) -> VideoGrid:
    """0.255 transformers MuseGlimmerVideoProcessor, verbatim: sample_frames caps the count at num_frames and rounds
    down to a multiple of temporal_patch (≥ temporal_patch); each frame is smart_resized to ≤ max_video_frame_tokens
    merged tokens; grid_t = frames / temporal_patch and the tower attends per temporal slice (vision_utils
    get_vision_cu_seqlens, merge_temporal False).  LLM tokens (MuseSparkProcessor.replace_video_token):
    <|vid_start|> + per slice [timestamp text + h·w/4 <|video|>] joined by <|vid_frame_separator|> + <|vid_end|>;
    the "Time: x.xxs" text is not counted (as for Qwen video)."""
    tp = max(1, v.tpatch)
    if frames < tp:
        raise ValueError(f"serving.video_frames must be ≥ temporal_patch_size {tp}")
    n = max(tp, (min(frames, pre["num_frames"]) // tp) * tp)
    hb, wb = _muse_resize(height, width, v.patch * v.merge, pre["muse_frame_tok"])
    ph, pw = hb // v.patch, wb // v.patch
    mg = ph * pw // (v.merge * v.merge)
    t = n // tp
    return VideoGrid(n, t, ImageGrid(ph, pw, mg, mg, (wb, hb)), t * mg + t + 1)


def radio_video_patches(width: int, height: int, target: int, r: int) -> tuple[int, int]:
    """0.279 NemotronH_Omni_Reasoning_V3ImageProcessor._compute_target_patches_video, verbatim (aspect kept): ph =
    round(√(target / aspect)), pw = round(√(target · aspect)), each rounded to a multiple of r — up if ph·pw stays ≤
    target, else both down (≥ r).  → (w_patches, h_patches)."""
    a = width / max(height, 1)
    ph, pw = max(round(math.sqrt(target / a)), 1), max(round(math.sqrt(target * a)), 1)
    if r > 1:
        rh, rw = ph % r, pw % r
        hu, wu = ph + (r - rh if rh else 0), pw + (r - rw if rw else 0)
        if hu * wu <= target:
            ph, pw = hu, wu
        else:
            ph, pw = max(r, ph - rh), max(r, pw - rw)
    return pw, ph


def _radio_video(v: VisionSpec, pre: dict, frames: int, width: int, height: int) -> VideoGrid:
    """0.279 Nemotron-3.5-Super-VL video (processing / modeling_nemotron_h_omni.py 「来源」): every frame resized to the
    video rule (radio_video_patches, ≈ 1024 patches); frames grouped in tubelets of 2 (last padded by repeating the
    final frame) — each tubelet is one tower pass (video_embedder over 2 frames' patches) → final LN, pixel shuffle,
    mlp1; prompt: per tubelet a frame label + <img> + h·w/4 tokens + </img>; EVS keeps max(one tubelet, ⌊t · h·w/4 ·
    (1 − q)⌋) of the video tokens (q = video_pruning_rate; the encoder still runs on all).  Frame-label text tokens are
    not counted (as for Qwen video)."""
    tp = pre["tp"]
    if frames < 1:
        raise ValueError("serving.video_frames must be ≥ 1")
    r = v.merge
    pw, ph = radio_video_patches(width, height, pre["radio_frame"], r)
    mg = ph * pw // (r * r)
    t = -(-frames // tp)
    keep = max(mg, int(t * mg * (1 - pre["evs"])))
    return VideoGrid(frames, t, ImageGrid(ph, pw, mg, mg + 2, (pw * v.patch, ph * v.patch)), keep + 2 * t)


def _kimi_video(v: VisionSpec, pre: dict, frames: int, width: int, height: int) -> VideoGrid:
    """kimi_k25_vision_processing.split_video_chunks + navit_resize_video, verbatim: frames are cut into chunks of
    ``chunk`` (last one shorter); each chunk is resized per frame with in_patch_limit_each_frame (lowered to
    round(total / frames in chunk) when in_patch_limit_video is set) and the side limit; tokens per chunk =
    (h/28)·(w/28) after tpool (independent of the frame count)."""
    c, side = pre["chunk"], v.rz["side"]
    n = -(-frames // c)

    def g(nf):
        lim = pre["frame_lim"] if pre["total"] is None else min(round(pre["total"] / nf), pre["frame_lim"])
        hb, wb = _kimi_resize(height, width, v.patch, v.merge, lim, side)
        ph, pw = hb // v.patch, wb // v.patch
        mg = ph * pw // (v.merge * v.merge)
        return ImageGrid(ph, pw, mg, mg, (wb, hb), tframes=nf)
    full, r = g(c), frames % c
    tail = g(r) if r and n > 1 else None
    first = full if (r == 0 or n > 1) else g(r)
    return VideoGrid(frames, n, first, n * first.merged if tail is None else (n - 1) * first.merged + tail.merged, tail)


def _gemma4_video(v: VisionSpec, pre: dict, frames: int, width: int, height: int) -> VideoGrid:
    """Gemma4VideoProcessor / Gemma4UnifiedVideoProcessor, per frame.

    ``get_aspect_ratio_preserving_size`` is the image resize with ``max_patches = max_soft_tokens · pooling²``.
    The published sampler (``do_sample_frames``, ``num_frames``) returns exactly ``num_frames`` indices and
    raises when the clip is shorter, so a longer ``serving.video_frames`` still feeds ``num_frames`` rows.
    ViT pads each frame to ``max_soft_tokens · pooling²`` patches before the tower. Unified merges first and
    pads to ``max_soft_tokens`` rows; ``pad_soft`` selects that pad inside ``_gemma4_vision_ops``. LLM tokens
    are the unpadded soft tokens (``_get_num_multimodal_tokens``). ``mm:ss`` text and the boi / eoi wrappers
    are not counted, the same way the image path omits boi / eoi. There is no per-frame pixel cap.
    """
    if v.patch != pre["patch"] or v.merge != pre["pool"]:
        raise ValueError("gemma4 video processor patch/pool does not match the vision tower")
    n = int(pre["num_frames"])
    if frames < n:
        raise ValueError(
            f"serving.video_frames must be ≥ num_frames {n}: Gemma4VideoProcessor samples exactly {n} frames "
            "and rejects a shorter clip")
    soft_cap = int(pre["gemma4_soft"])
    pool = int(pre["pool"])
    max_patches = soft_cap * pool * pool
    th, tw = _gemma4_resize(height, width, int(pre["patch"]), max_patches, pool)
    ph, pw = th // int(pre["patch"]), tw // int(pre["patch"])
    real = ph * pw // (pool * pool)
    one = ImageGrid(ph, pw, real, real, (tw, th), pad_soft=soft_cap)
    return VideoGrid(n, n, one, n * real)


def _ernie_video(v: VisionSpec, pre: dict, frames: int, width: int, height: int) -> VideoGrid:
    """Ernie4_5_VLProcessor video grid.

    Each frame is ``smart_resize``d with the processor's video pixel limits. Those
    limits are per frame (``_add_video``), not a clip budget. An odd count copies
    the last frame so ``temporal_conv_size`` 2 has equal rows. LLM tokens are the
    paired rows plus ``VIDEO_START`` and ``VIDEO_END``. ``fps`` / ``min_frames`` /
    ``max_frames`` only choose samples from a file; ``serving.video_frames`` is
    that count and is not clamped. Timestamps are drawn into the pixels. One slice
    holds every padded frame so the resampler pairs them in a single encoder pass.
    """
    if frames < 1:
        raise ValueError("serving.video_frames must be ≥ 1")
    temporal = int(pre["ernie_temporal"])
    factor = v.patch * v.merge
    if temporal != 2 or factor != 28:
        raise ValueError("ernie4_5_vl video temporal merge is temporal_conv_size 2 and factor 28")
    t = frames + (frames % temporal)
    hb, wb = _ernie_smart_resize(height, width, factor, int(pre["min_px"]), int(pre["max_px"]))
    ph, pw = hb // v.patch, wb // v.patch
    per = ph * pw // (v.merge * v.merge)
    llm = (t // temporal) * per + 2
    one = ImageGrid(ph, pw, per, llm, (wb, hb), tframes=t)
    return VideoGrid(t, 1, one, llm)


def video_grid_of(m, sv) -> VideoGrid | None:
    if not getattr(sv, "videos", 0):
        return None
    pre = video_pre(m)
    if pre is None:
        raise ValueError("serving.videos：该模型没有已建模的视频处理器（Qwen3VLVideoProcessor 系：Qwen3.5-397B / "
                         "Qwen3.8 / Intern-S2；0.166 Kimi K2.5 / K2.7-Code 4 帧块；0.255 Muse Spark Open；"
                         "0.279 Nemotron-3.5-Super-VL RADIO tubelet；0.382 Gemma-4 逐帧 pad；"
                         "0.385 ERNIE-4.5-VL 帧配对）")
    return video_grid(m.vision, pre, sv.video_frames, sv.video_w, sv.video_h, sv.video_cap)


def text_view(scn):
    """Inverse of expand_images for echoing the user's scenario (prompt / ctx as entered)."""
    sv = scn.serving
    if not sv.image_tokens:
        return scn
    import dataclasses
    t = sv.image_tokens
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt - t, ctx=sv.ctx - t, image_tokens=0))
