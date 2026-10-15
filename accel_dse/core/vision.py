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
from dataclasses import dataclass

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
    "stepfun-ai/Step-5-Preview": {"side": 728},                                # Step5VisionConfig image_size (overview)
    "openbmb/MiniCPM-V-4.7-35B-A3B": {"side": 448, "max_slice": 9},           # MiniCPMV4_6ImageProcessor defaults
    # 0.145
    "XiaomiMiMo/MiMo-V2.6-Pro-RL": {"min_px": 8192, "max_px": 8388608},        # processor_config image_min/max_pixels
    "inclusionAI/Ling-3.0-flash-VL": {"min_px": 4096, "max_px": 4194304},      # preprocessor_config min/max_pixels
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

    @property
    def head_dim(self) -> int:
        return self.qkv // self.heads

    @property
    def rz(self) -> dict:
        return dict(self.pre)


def _lin(i: int, o: int, b: bool = True) -> int:
    return i * o + (o if b else 0)


def vision_spec(hf_id: str, cfg: dict) -> VisionSpec | None:
    """VisionSpec from the release's full config (``vision_config`` + text hidden size), or None."""
    vc = cfg.get("vision_config")
    if not vc or hf_id not in _PREPROC:
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
    if mt in ("qwen3_5", "qwen3_5_moe", "qwen4_exp"):
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
                          approx=f"视觉：每图只计 {side}² 概览（config.image_token_len=169 token）；开放权重 inference 接受外部预处理的 504² 裁剪（patch_token_len=81/块，num_patches），但裁剪块数算法未随 Step-5-Preview 发布（model.py: external image preprocessing）；平台 API 仅给 detail low/high 的最长边缩放建议（728 / 504 倍数），无公开 crop/overview 处理器 → 仍 partial")
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
    pr = v.rz
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
        return ImageGrid(ph, pw, m, m, (pr["side"], pr["side"]))
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
        ops += _tower_ops(v, ph * pw, images * cnt, mrows, "vis." if gi == 0 else f"vis.v{gi}.", af)
    return ops


def _tower_ops(v: VisionSpec, N: int, images: int, mrows: int | None, pfx: str, af: str) -> list[Op]:
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
    return image_grid(v, sv.image_w, sv.image_h) if v is not None and sv.images else None


def expand_images(scn, m):
    """The scenario with each request's image tokens added to serving.prompt and serving.ctx (idempotent: marks
    serving.image_tokens).  Everything downstream (prefill length, KV, PD lengths / capacity, goodput) then sees the
    image positions as prompt tokens; the encoder itself is costed by evaluate at prefill."""
    sv = scn.serving
    if not (sv.images or getattr(sv, "videos", 0)) or sv.image_tokens:
        return scn
    g = image_grid_of(m, sv)
    vg = video_grid_of(m, sv) if getattr(m, "vision", None) is not None else None
    if g is None and vg is None:
        return scn
    import dataclasses
    t = (sv.images * g.llm_tokens if g is not None else 0) + (sv.videos * vg.llm_tokens if vg is not None else 0)
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt + t, ctx=sv.ctx + t, image_tokens=t))


def video_pre(m) -> dict | None:
    """Video processor settings of a model with a modelled video path (Qwen3VLVideoProcessor; 0.166 Kimi chunks)."""
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
                         "Qwen3.8 / Intern-S2；0.166 Kimi K2.5 / K2.7-Code 4 帧块）")
    return video_grid(m.vision, pre, sv.video_frames, sv.video_w, sv.video_h, sv.video_cap)


def text_view(scn):
    """Inverse of expand_images for echoing the user's scenario (prompt / ctx as entered)."""
    sv = scn.serving
    if not sv.image_tokens:
        return scn
    import dataclasses
    t = sv.image_tokens
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt - t, ctx=sv.ctx - t, image_tokens=0))
