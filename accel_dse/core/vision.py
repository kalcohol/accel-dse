"""VLM vision encoders (0.62): vision tower + projector / merger of every VLM in the catalog, built from the released
``vision_config`` and the released image-processor settings, run at prefill.

Each released family (checked against the reference modules: transformers ``qwen3_5`` / ``qwen4_exp`` / ``glm5_next``
/ ``kimi_k25`` vision models, the Kimi-K3 remote code ``modeling_kimi_k3.py`` and DeepSeek-V4.1 ``inference/vision.py``;
docs/MODEL.md §20):

  qwen   Qwen3.5 / Qwen3.8: Conv3d patch embed (temporal_patch_size 2 — a still image is repeated to 2 frames, one
         temporal patch), learned 2-D position table (bilinear gather), ``depth`` pre-LN blocks (fused qkv + proj,
         GELU MLP fc1 / fc2), 2×2 patch merger (LN → fc1 4h→4h → GELU → fc2 4h→out).  Resize: Qwen2-VL
         ``smart_resize`` to multiples of patch·merge = 32 px within [shortest_edge, longest_edge] pixels.
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

Counted per image (all images of a request share the size): every GEMM above at M = patches (tower) or merged rows
(merger / aligner); bidirectional attention within the image (QKᵀ and PV, ``heads`` × patches²); norm / activation /
residual element-ops 3 per GEMM output 「假设」, softmax 5 per score.  Image tokens into the LLM = merged positions
(+ DeepSeek's processor-inserted newline / start / end positions); chat-template delimiters around an image are text
tokens of the prompt.  The encoder weights are bf16 as released; activations bf16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .ir import Op

VEC_PER_OUT = 3.0
DEFAULT_PX = (1024, 1024)          # 「假设」 typical image when serving.images > 0 and no resolution is given

# released image-processor settings (preprocessor_config.json / processor_config.json / inference/config.json)
_PREPROC = {
    "Qwen/Qwen3.8-Flash-Next": {"min_px": 65536, "max_px": 16777216},
    "Qwen/Qwen3.8-27B": {"min_px": 65536, "max_px": 16777216},
    "Qwen/Qwen3.5-397B-A17B": {"min_px": 65536, "max_px": 16777216},
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
                                                         "microsoft/UserRepGen-4B", "Tongyi-MAI/MAI-Gym-9B") + tuple(
        f"startlux-models/StartLux-Decision-{v}" for v in (
            "v1-27B", "v1-9B", "v1-4B", "v1-2B", "v1-0.8B", "v1-35B-A3B", "v1.1-27B-Instant", "v1.1-27B-AutoThink",
            "v1.1-9B-Instant", "v1.1-9B-AutoThink", "v1.1-4B-Instant", "v1.1-4B-AutoThink", "v1.1-2B-Instant",
            "v1.1-0.8B-Instant"))},
}

# 0.165 — video input.  「来源」 each repo's video_preprocessor_config.json (Qwen3VLVideoProcessor): size shortest_edge
# 4096 / longest_edge 25165824 (a pixel budget over t_bar·h·w of the whole clip), patch 16, temporal_patch 2, merge 2.
_VIDEO = {k: {"min_px": 4096, "max_px": 25165824} for k in (
    "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next", "internlm/Intern-S2-397B")}
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
    # (positional_embedding rows − 1: positions counted from one)
    lfr: tuple[int, int] = ()
    ad: tuple[int, int, int, int] = ()
    # 0.272 Step-Audio-2-mini: ``mpad`` samples appended before the STFT (log_mel_spectrogram padding=479; frames =
    # (samples + mpad) // hop); ``ds`` + ``lin`` together = AvgPool → Adaptor conv (GELU) → Linear, ReLU, Linear
    mpad: int = 0

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
        if self.kind == "aut":
            return sum(self.aut_len(n) for n in self.chunks(seconds))
        if self.kind == "sanm":
            return self.lfr_frames(seconds)
        if self.kind != "whisper":
            return self.tokens(seconds)        # Muse Spark: one output position per token (no chunking)
        return sum(self.chunk_len(n)[0] for n in self.chunks(seconds))

    def prompt_tokens(self, seconds: float) -> int:
        """0.270 audio positions inserted into the LLM prompt (0 for an encoder-decoder: cross-attention instead)."""
        return 0 if self.cross else self.tokens(seconds)

    def chunk_len(self, n: int) -> tuple[int, int]:
        """(encoder positions after conv2, projector tokens) of an n-frame chunk (MingAudioSpec.embedding_length)."""
        t = (n + 1) // 2
        o = t // self.pool if self.ds and self.pool > 1 else t      # 0.272 Step-Audio-2: AvgPool before the adaptor conv
        for k, st, pd in self.ds:
            o = (o + 2 * pd - k) // st + 1
        return t, o

    def tokens(self, seconds: float) -> int:
        if self.kind == "sanm":
            return self.lfr_frames(seconds)     # processor: num_audio_tokens = input_features_mask.sum (one per LFR frame)
        if self.kind == "aut":
            return self.enc_positions(seconds)
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
        return math.ceil(round(seconds * self.rate) / self.samples_per_token)


def audio_spec(cfg: dict) -> AudioSpec | None:
    ac = cfg.get("audio_config") or {}
    mt = cfg.get("model_type")
    if mt == "ming_v3_asr" and ac.get("n_audio_state"):
        return _whisper_audio_spec(cfg, ac)
    if mt == "whisper":
        return _whisper_v3_spec(cfg, cfg["encoder_layers"], cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"],
                                cfg["num_mel_bins"], cfg["max_source_positions"], cross=True)
    if mt == "qwen2_audio" and ac.get("model_type") == "qwen2_audio_encoder":
        out = (cfg.get("text_config") or {}).get("hidden_size") or 4096     # Qwen2Config default (text_config omits it)
        return _whisper_v3_spec({"d_model": ac["d_model"]}, ac["encoder_layers"], ac["encoder_attention_heads"],
                                ac["encoder_ffn_dim"], ac["num_mel_bins"], ac["max_source_positions"], pool=2,
                                trunc=True, lin=((ac["d_model"], out),))
    tk = cfg.get("thinker_config") or {}
    if mt == "qwen2_5_omni" and (tk.get("audio_config") or {}).get("model_type") == "qwen2_5_omni_audio_encoder":
        return _omni_audio_spec(tk["audio_config"])
    sa = cfg.get("audio_encoder_config") or {}
    if mt == "step_audio_2" and sa.get("n_audio_state"):
        return _step_audio_spec(sa)
    if mt == "fun_asr_nano" and ac.get("model_type") == "fun_asr_nano_encoder":
        return _sanm_spec(ac, cfg.get("adaptor_config") or {})
    if mt == "qwen3_asr" and (tk.get("audio_config") or {}).get("model_type") == "qwen3_asr_audio_encoder":
        return _aut_spec(tk["audio_config"])
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
    in the checkpoint); ln_post, AvgPool1d(2, 2), proj d → output_dim; audio_bos_eos_token Embedding(2, output_dim)
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
    13 · n_window_infer / (2·n_window) positions (cu_seqlens), ln_post; projector Linear d → d, GELU, Linear d → out."""
    d, L, nh, ffn, mel = ac["d_model"], ac["encoder_layers"], ac["encoder_attention_heads"], ac["encoder_ffn_dim"], ac["num_mel_bins"]
    C, nw = ac["downsample_hidden_size"], ac["n_window"]
    f = mel
    for _ in range(3):
        f = (f + 1) // 2
    blk = 4 * d * d + 4 * d + 2 * d + _lin(d, ffn) + _lin(ffn, d) + 2 * d
    lin = ((d, d), (d, ac["output_dim"]))
    params = (_lin(9, C) + 2 * _lin(9 * C, C) + C * f * d + L * blk + 2 * d + sum(_lin(k, n) for k, n in lin))
    a = AudioSpec(16000, (160,), (d,), (L,), 3, ffn / d, ac["output_dim"], params, kind="aut", heads=nh, n_mels=mel,
                  chunk=2 * nw, lin=lin, ch=C)
    return replace(a, win=a.aut_len(2 * nw) * (ac["n_window_infer"] // (2 * nw)))


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
    D/4).  One LLM token per LFR frame.  Header: 233,741,920 params."""
    d, L, nh, ffn, k = ac["hidden_size"], ac["num_hidden_layers"], ac["num_attention_heads"], ac["intermediate_size"], ac["fsmn_kernel_size"]
    m = ac.get("num_stacked_frames", 7)
    din = ac["num_mel_bins"] * m
    D, P, nl = ad["hidden_size"], ad["projector_hidden_size"], ad["num_hidden_layers"]
    adh, adf = ad["num_attention_heads"], ad["intermediate_size"]

    def blk(i: int, dd: int, f: int, fsmn: bool) -> int:
        return 2 * i + 3 * _lin(i, dd) + _lin(dd, dd) + (dd * k if fsmn else 0) + 2 * dd + _lin(dd, f) + _lin(f, dd)
    params = (blk(din, d, ffn, True) + (L - 1) * blk(d, d, ffn, True) + 2 * 2 * d
              + _lin(d, P) + _lin(P, D) + nl * blk(D, D, adf, False))
    return AudioSpec(16000, (160,), (d,), (L,), k, ffn / d, D, params, kind="sanm", heads=nh, n_mels=din,
                     lin=((d, P), (P, D)), maxf=ac["max_position_embeddings"] - 1, lfr=(m, 6), ad=(nl, D, adh, adf))


def _sanm_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """0.272 Fun-ASR-Nano SAN-M encoder + projector + adaptor ops (see _sanm_spec): every layer on all T LFR positions
    of a clip (one pass per clip, bidirectional attention T × T per head); FSMN as a depthwise conv (2·k per output
    element); final LayerNorms folded into the vector ops."""
    d, dh = a.dims[0], a.dims[0] // a.heads
    T = a.lfr_frames(seconds)
    R = T * clips
    ops: list[Op] = []

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
    for li in range(a.depths[0]):
        i = a.n_mels if li == 0 else d
        ops += [gemm(f"aud.{li}.qkv", R, i, 3 * d, li), gemm(f"aud.{li}.o", R, d, d, li),
                Op(f"aud.{li}.fsmn", "vector", li, vec=R * d * 2.0 * a.kernel, act_bytes=2 * R * d * ab,
                   w_params=d * a.kernel, w_bits=16.0),
                gemm(f"aud.{li}.fc1", R, d, ff, li), gemm(f"aud.{li}.fc2", R, ff, d, li)]
        ops += attn("aud.", li, a.heads, dh)
    L = a.depths[0]
    for j, (k, n) in enumerate(a.lin):
        ops.append(gemm(f"aud.proj.linear{j}", R, k, n, L))
    nl, D, h2, f2 = a.ad
    for j in range(nl):
        li = L + 1 + j
        ops += [gemm(f"aud.ad{j}.qkv", R, D, 3 * D, li), gemm(f"aud.ad{j}.o", R, D, D, li),
                gemm(f"aud.ad{j}.fc1", R, D, f2, li), gemm(f"aud.ad{j}.fc2", R, f2, D, li)]
        ops += attn(f"aud.ad{j}.", li, h2, D // h2)
    return ops


def _aut_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    """0.270 Qwen3-ASR AuT encoder ops (see _aut_spec): Conv2d as implicit GEMMs over every padded chunk; blocks on
    the valid positions; attention per window (full windows + the tail window)."""
    d, dh, C = a.dims[0], a.dims[0] // a.heads, a.ch
    ops: list[Op] = []

    def gemm(name: str, m: int, k: int, n: int, li: int = 0, w: int | None = None) -> Op:
        return Op(name, "gemm", li, m=m, k=k, n=n, role="vision", w_params=k * n if w is None else w, w_bits=16.0,
                  w_fmt="bf16", a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)
    nch = len(a.chunks(seconds)) * clips
    f, t = a.n_mels, a.chunk
    k_in = 1
    for i in range(3):
        f, t = (f + 1) // 2, (t + 1) // 2
        ops.append(gemm(f"aud.conv2d{i + 1}", nch * f * t, 9 * k_in, C))
        k_in = C
    P = a.enc_positions(seconds) * clips
    ops.append(gemm("aud.conv_out", nch * t, C * f, d))
    wins = []
    for _ in range(clips):
        full, rem = divmod(a.enc_positions(seconds), a.win)
        wins += [a.win] * full + ([rem] if rem else [])
    for li in range(a.depths[0]):
        ops += [gemm(f"aud.{li}.qkv", P, d, 3 * d, li), gemm(f"aud.{li}.o", P, d, d, li),
                gemm(f"aud.{li}.fc1", P, d, int(a.mlp * d), li), gemm(f"aud.{li}.fc2", P, int(a.mlp * d), d, li)]
        for wi, w in enumerate(sorted(set(wins), reverse=True)):
            cnt = wins.count(w) * a.heads
            sfx = f".w{wi}" if wi else ""
            ops += [Op(f"aud.{li}.qk{sfx}", "attn", li, m=w, k=dh, n=w, count=cnt, causal=1.0, act_bytes=cnt * w * dh * ab,
                       stream=True, orient=True),
                    Op(f"aud.{li}.pv{sfx}", "attn", li, m=w, k=w, n=dh, count=cnt, causal=1.0, act_bytes=cnt * w * dh * ab,
                       orient=True),
                    Op(f"aud.{li}.softmax{sfx}", "vector", li, vec=cnt * w * w * 5.0)]
    for i, (k, n) in enumerate(a.lin):
        ops.append(gemm(f"aud.proj.linear{i}", P, k, n, a.depths[0]))
    return ops


def _whisper_ops(a: AudioSpec, seconds: float, clips: int, ab: float, af: str) -> list[Op]:
    d, dh = a.dims[0], a.dims[0] // a.heads
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
        ops += [gemm(p + "conv1", cnt * n, 3 * a.n_mels, d), gemm(p + "conv2", rows, 3 * d, d)]
        for li in range(a.depths[0]):
            ops += [gemm(f"{p}{li}.qkv", rows, d, 3 * d, li), gemm(f"{p}{li}.o", rows, d, d, li),
                    gemm(f"{p}{li}.fc1", rows, d, int(a.mlp * d), li), gemm(f"{p}{li}.fc2", rows, int(a.mlp * d), d, li),
                    Op(f"{p}{li}.qk", "attn", li, m=t, k=dh, n=t, count=ch, causal=1.0, act_bytes=ch * t * dh * ab,
                       stream=True, orient=True),
                    Op(f"{p}{li}.pv", "attn", li, m=t, k=t, n=dh, count=ch, causal=1.0, act_bytes=ch * t * dh * ab,
                       orient=True),
                    Op(f"{p}{li}.softmax", "vector", li, vec=ch * t * t * 5.0)]
        if a.lin and a.ds:
            # 0.272 Step-Audio-2: AvgPool1d(pool) → Adaptor conv (k, s) d → d as an implicit GEMM → Linears
            ops.append(Op(f"{p}pool", "vector", a.depths[0], vec=rows * d * 1.0))
            (kk, _, _), = a.ds
            ops.append(gemm(f"{p}proj.conv0", cnt * o, kk * d, d, a.depths[0]))
            for i, (k, n_) in enumerate(a.lin):
                ops.append(gemm(f"{p}proj.linear{i}", cnt * o, k, n_, a.depths[0]))
            continue
        if a.lin is not None:
            # 0.270 Whisper-v3 family: AvgPool1d(pool) (Qwen2-Audio; ½ add per pooled input) then the projector Linears
            # on every padded position (the processor's mask selects the valid ones afterwards)
            rp = rows // a.pool
            if a.pool > 1:
                ops.append(Op(f"{p}pool", "vector", a.depths[0], vec=rows * d * 1.0))
            for i, (k, n_) in enumerate(a.lin):
                ops.append(gemm(f"{p}proj.linear{i}", rp, k, n_, a.depths[0]))
            continue
        din, m_ = d, rows
        for i, (k, st, pd) in enumerate(a.ds):
            m_ = cnt * o if i == len(a.ds) - 1 else m_      # single-stage downsampler in the release
            ops.append(gemm(f"{p}proj.conv{i}", m_, k * din, a.out, a.depths[0]))
            din = a.out
        ops.append(gemm(f"{p}proj.linear", cnt * o, a.out, a.out, a.depths[0]))
    return ops


def audio_ops(a: AudioSpec, seconds: float, clips: int, af: str = "bf16") -> list[Op]:
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    if a.kind == "whisper":
        return _whisper_ops(a, seconds, clips, ab, af)
    if a.kind == "aut":
        return _aut_ops(a, seconds, clips, ab, af)
    if a.kind == "sanm":
        return _sanm_ops(a, seconds, clips, ab, af)
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


def vision_spec(hf_id: str, cfg: dict) -> VisionSpec | None:
    """VisionSpec from the release's full config (``vision_config`` + text hidden size), or None."""
    vc = cfg.get("vision_config")
    hf_id = hf_id.split("+", 1)[0]              # 0.175: "<target>+<draft repo>" entries use the target's processor
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
    if not vc:
        au = audio_spec(cfg)
        if au is not None and au.kind in ("whisper", "aut", "sanm"):
            # 0.268 audio-only release (Ming-3.0-asr-tiny): an encoder container with no image path (family "audio":
            # image_grid_of → None, serving.images ignored with a warning); costed when serving.audio_s > 0
            # 0.270: Whisper-large-v3 (encoder-decoder), Qwen2-Audio (Whisper-v3 encoder + pool + projector), Qwen3-ASR (AuT)
            lab = ("Whisper 式音频编码器（整段一次、≤ 30 s）+ 平均池化 + 卷积适配器（s2）+ MLP 投影" if au.ds and au.lin else
                   "Whisper-v3 音频编码器（30 s 填充窗，交叉注意力输入）" if au.cross else
                   "Whisper-v3 音频编码器（30 s 填充）+ 平均池化 + 线性投影" if au.lin is not None and au.kind == "whisper" else
                   "AuT 音频编码器（Conv2d ×3 下采样 8×，窗口注意力）+ MLP 投影" if au.kind == "aut" else
                   "SAN-M 音频编码器（LFR 7/6 帧堆叠，70 层 FSMN 自注意力，整段双向）+ 投影 + 2 层适配器" if au.kind == "sanm" else
                   "Whisper 音频编码器 + 卷积下采样投影")
            return VisionSpec("audio", lab, 0, 1, au.dims[0], int(au.mlp * au.dims[0]),
                              au.depths[0], au.heads, au.dims[0], False, 1, (), au.params, (), audio=au)
        return None
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
    if mt in ("qwen3_5", "qwen3_5_moe", "qwen4_exp", "qwen3_5_vision"):   # 0.191: transformers 5 names it *_vision
        h, inter, L, nh = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"]
        p, t, mk, out = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"], vc["out_hidden_size"]
        if vc.get("deepstack_visual_indexes"):
            return None                             # deepstack mergers not modelled (none in the catalog)
        hm = h * mk * mk
        params = (_lin(3 * t * p * p, h) + vc["num_position_embeddings"] * h
                  + L * (4 * h + _lin(h, 3 * h) + _lin(h, h) + _lin(h, inter) + _lin(inter, h))
                  + 2 * h + _lin(hm, hm) + _lin(hm, out))
        pre["factor"] = p * mk
        return VisionSpec("qwen", "Qwen ViT", p, t, h, inter, L, nh, h, False, mk, ((hm, hm), (hm, out)), params, tuple(sorted(pre.items())))
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


def image_grid(v: VisionSpec, width: int, height: int) -> ImageGrid:
    if v.family == "audio":                       # 0.268 audio-only encoder: no image path
        return ImageGrid(0, 0, 0, 0, (0, 0))
    pr = v.rz
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
    ph, pw, gh, gw = _ds_grid(height, width, v.patch, pr["ratio"], pr["min_pixels"], pr["max_tok"])
    return ImageGrid(ph, pw, gh * gw, gh * (gw + 1) + 2, (pw * v.patch, ph * v.patch))


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

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op(pfx + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=wb, w_fmt=wf,
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

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
        ops += [gemm(f"{li}.qkv", rows, h, qkv_n, li), gemm(f"{li}.o", rows, q, h, li)]
        ops.append(gemm(f"{li}.mlp_in", rows, h, (2 if v.gated else 1) * v.inter, li))
        ops.append(gemm(f"{li}.mlp_out", rows, v.inter, h, li))
        if v.tile and grid is not None and li not in v.full_idx:
            ops += attn(str(li), li, N, tile_keys(grid[0], grid[1], v.tile), False)
            continue
        windowed = bool(v.win) and li not in v.full_idx
        ops += attn(str(li), li, N, band_keys(N, v.win) if windowed else N * N, windowed and v.sink)
    for i, kn in enumerate(v.post):
        k, n = kn[:2]
        if len(kn) > 2:
            m_ = images * (N0 // kn[2])
        else:
            m_ = mrows if mrows is not None else images * (N0 // (v.merge * v.merge))
        ops.append(gemm(f"merger.{i}", m_, k, n, v.depth))
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
    0.255 MuseGlimmerVideoProcessor)."""
    v = getattr(m, "vision", None)
    pre = _VIDEO.get(getattr(m, "hf_id", "")) if v is not None else None
    if pre is None or v.family != ("kimi" if "chunk" in pre else "qwen"):
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


def video_grid_of(m, sv) -> VideoGrid | None:
    if not getattr(sv, "videos", 0):
        return None
    pre = video_pre(m)
    if pre is None:
        raise ValueError("serving.videos：该模型没有已建模的视频处理器（Qwen3VLVideoProcessor 系：Qwen3.5-397B / "
                         "Qwen3.8 / Intern-S2；0.166 Kimi K2.5 / K2.7-Code 4 帧块；0.255 Muse Spark Open）")
    return video_grid(m.vision, pre, sv.video_frames, sv.video_w, sv.video_h, sv.video_cap)


def text_view(scn):
    """Inverse of expand_images for echoing the user's scenario (prompt / ctx as entered)."""
    sv = scn.serving
    if not sv.image_tokens:
        return scn
    import dataclasses
    t = sv.image_tokens
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt - t, ctx=sv.ctx - t, image_tokens=0))
