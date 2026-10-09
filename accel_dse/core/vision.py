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
    "deepseek-ai/DeepSeek-V4.1-Flash": {},          # min_pixels / max_image_tokens live in vision_config
}


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


@dataclass(frozen=True)
class ImageGrid:
    ph: int                     # patch rows / cols into the tower
    pw: int
    merged: int                 # positions out of the merger / aligner
    llm_tokens: int             # positions the image occupies in the LLM sequence
    px: tuple[int, int]         # (width, height) the image is resized / padded to


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
    ph, pw, gh, gw = _ds_grid(height, width, v.patch, pr["ratio"], pr["min_pixels"], pr["max_tok"])
    return ImageGrid(ph, pw, gh * gw, gh * (gw + 1) + 2, (pw * v.patch, ph * v.patch))


# ------------------------------------------------------------------ op graph
def vision_ops(v: VisionSpec, g: ImageGrid, images: int, af: str = "bf16") -> list[Op]:
    """Op list of one encoder pass over ``images`` images of grid ``g`` (bf16 weights)."""
    if images <= 0:
        return []
    from .dtypes import fmt as _fmt
    ab = _fmt(af).bytes
    N = g.ph * g.pw
    rows = images * N
    mrows = images * g.merged
    wf, wb = "bf16", 16.0
    ops: list[Op] = []

    def gemm(name: str, m: int, k: int, n: int, layer: int = 0) -> Op:
        return Op("vis." + name, "gemm", layer, m=m, k=k, n=n, role="vision", w_params=k * n, w_bits=wb, w_fmt=wf,
                  a_fmt=af, act_bytes=m * (k + n) * ab, stream=True, vec=m * n * VEC_PER_OUT)

    ops.append(gemm("patch_embed", rows, 3 * v.tpatch * v.patch * v.patch, v.hidden))
    h, q, dh = v.hidden, v.qkv, v.head_dim
    cnt = images * v.heads
    for li in range(v.depth):
        ops += [gemm(f"{li}.qkv", rows, h, 3 * q, li), gemm(f"{li}.o", rows, q, h, li)]
        ops.append(gemm(f"{li}.mlp_in", rows, h, (2 if v.gated else 1) * v.inter, li))
        ops.append(gemm(f"{li}.mlp_out", rows, v.inter, h, li))
        ops += [Op(f"vis.{li}.qk", "attn", li, m=N, k=dh, n=N, count=cnt, act_bytes=cnt * N * dh * ab, stream=True,
                   orient=True),
                Op(f"vis.{li}.pv", "attn", li, m=N, k=N, n=dh, count=cnt, act_bytes=cnt * N * dh * ab, orient=True),
                Op(f"vis.{li}.softmax", "vector", li, vec=cnt * N * N * 5.0)]
    for i, (k, n) in enumerate(v.post):
        ops.append(gemm(f"merger.{i}", mrows, k, n, v.depth))
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
    if not sv.images or sv.image_tokens:
        return scn
    g = image_grid_of(m, sv)
    if g is None:
        return scn
    import dataclasses
    t = sv.images * g.llm_tokens
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt + t, ctx=sv.ctx + t, image_tokens=t))


def text_view(scn):
    """Inverse of expand_images for echoing the user's scenario (prompt / ctx as entered)."""
    sv = scn.serving
    if not sv.image_tokens:
        return scn
    import dataclasses
    t = sv.image_tokens
    return dataclasses.replace(scn, serving=dataclasses.replace(sv, prompt=sv.prompt - t, ctx=sv.ctx - t, image_tokens=0))
