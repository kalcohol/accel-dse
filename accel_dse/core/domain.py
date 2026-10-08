"""L1c — non-autoregressive domains on core v2: video-generation DiT denoisers and protein language models.

Built **from the official release** like the LLMs: dims from the release config, io / conditioning dims from the
safetensors header shapes (``releases/*.json`` → ``shapes``), formats from the header dtypes.  Parameter counts are
checked against the header totals (V1).

Video DiT (one denoise step = one forward over every latent token of the clip, *full 3D attention*, bidirectional):
  Wan2.1     self-attention over all video tokens (3D RoPE, RMS QK-norm) → cross-attention to the text tokens
             (k/v from the projected umT5 output, recomputed every step as in the reference) → GELU FFN;
             AdaLN modulation from a shared timestep projection.
  CogVideoX  joint attention over [text ‖ video] tokens (expert AdaLN per stream, shared QKV / FFN weights),
             GELU FFN; 3D RoPE on 5B, sincos positions on 2B.
Protein (ESM-2): bidirectional pre-LN encoder with rotary attention, GELU FFN, tied MLM head.

What is *not* modelled is listed in ``coverage_reasons`` (text encoder / VAE of the video pipelines).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .model import AttnCore, Ffn, Layer, Linear, ModelSpec, RoleFormat, NativeWorkload, _release_formats, load_release

GiB = 2**30


# ---------------------------------------------------------------- shared helpers
def _mha(d: int, H: int, *, rows_kv: str = "tok", causal: bool = False, cross: bool = False,
         rope: int = 0) -> tuple[tuple[Linear, ...], AttnCore]:
    hd = d // H
    lin = (Linear("q", d, d, "col"), Linear("k", d, d, "col", rows=rows_kv), Linear("v", d, d, "col", rows=rows_kv),
           Linear("o", d, d, "row"))
    return lin, AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=rope, causal=causal, cross=cross)


def _gelu_ffn(d: int, f: int) -> tuple[Ffn, tuple[Linear, ...]]:
    return Ffn("dense", d_ff=f, gated=False), (Linear("up", d, f, "col"), Linear("down", f, d, "row"))


def _formats(rel: dict) -> tuple[tuple[str, RoleFormat], ...]:
    fm = _release_formats(rel)
    # domain roles not in the LLM role list: conditioning / io GEMMs share the attention format in every release
    for role in ("cond", "io"):
        v = rel.get("roles", {}).get(role)
        if v:
            fmt, _ = max(v.items(), key=lambda kv: kv[1]["params"])
            p = sum(x["params"] for x in v.values())
            b = sum(x["bytes"] for x in v.values())
            fm[role] = RoleFormat(fmt if fmt in ("fp32", "bf16", "fp16", "fp8") else "bf16", 8.0 * b / p)
    return tuple(sorted(fm.items()))


def _act_fmt(fm: dict) -> str:
    """Activations: fp16 releases run fp16 (CogVideoX-2b reference); bf16 otherwise — fp32 releases (Wan2.1, ESM-2)
    are run with bf16 activations as in the Wan reference (autocast bf16) 「假设」 for ESM-2."""
    w = fm.get("attn")
    return "fp16" if w and w.fmt == "fp16" else "bf16"


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _unmodelled(rel: dict, subfolder: str) -> dict[str, int]:
    """Weight bytes of the pipeline components outside the modelled denoiser (text encoder, VAE, …)."""
    out: dict[str, int] = {}
    files = (rel.get("source") or {}).get("repo_files") or {}
    for path, size in files.items():
        top = path.split("/", 1)[0] if "/" in path else path
        if subfolder and top == subfolder:
            continue
        if not subfolder and path.startswith("diffusion_pytorch_model"):
            continue
        key = ("text_encoder" if ("t5" in path.lower() or top == "text_encoder") else
               "vae" if ("vae" in path.lower() or top == "vae") else top)
        out[key] = out.get(key, 0) + size
    return out


# ---------------------------------------------------------------- Wan2.1
def build_wan(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    d, H, f, L = c["dim"], c["num_heads"], c["ffn_dim"], c["num_layers"]
    text_len = c.get("text_len", 512)
    pe = sh["patch_embedding.weight"]                       # [d, in, pt, ph, pw]
    in_dim, patch = pe[1], tuple(pe[2:5])
    text_dim = sh["text_embedding.0.weight"][1]
    freq = sh["time_embedding.0.weight"][1]
    n_mod = sh["time_projection.1.weight"][0]               # 6·d (shift/scale/gate × attn, ffn)
    out_p = sh["head.head.weight"][0]                       # out_dim · prod(patch)
    hd = d // H
    alin, core = _mha(d, H, rope=hd)
    xlin, xcore = _mha(d, H, rows_kv="ctx", cross=True)
    ffn, flin = _gelu_ffn(d, f)
    misc = 4 * d + 2 * d + 4 * d + 2 * d + 2 * d + f + d + 6 * d   # biases, RMS QK-norms, norm3 (affine), modulation
    layer = Layer(alin, core, ffn, flin, misc, cross_linears=xlin, cross=xcore)
    io_pre = (Linear("patch_embed", in_dim * math.prod(patch), d, "col", rows="img"),
              Linear("text_embed_1", text_dim, d, "col", rows="ctx"), Linear("text_embed_2", d, d, "col", rows="ctx"),
              Linear("time_embed_1", freq, d, "col", rows="seq"), Linear("time_embed_2", d, d, "col", rows="seq"),
              Linear("time_proj", d, n_mod, "col", rows="seq"))
    io_post = (Linear("head", d, out_p, "col", rows="img"),)
    io_misc = d + 2 * d + 2 * d + n_mod + out_p + 2 * d      # io biases + head modulation
    wl = replace(wl, patch=patch, text_tokens=text_len, prefix_tokens=0)
    return _finish(model_id, hf_id, rel, "Wan2.1 DiT（全 3D 注意力 + 文本跨注意力）", d, L, layer, io_pre, io_post,
                   io_misc, 0, wl, "")


# ---------------------------------------------------------------- CogVideoX
def build_cogvideox(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    H, hd, L = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"]
    d, te, text_dim = H * hd, c["time_embed_dim"], c["text_embed_dim"]
    p, in_c = c["patch_size"], c["in_channels"]
    out_p = sh["proj_out.weight"][0]
    inner = sh["transformer_blocks.0.ff.net.0.proj.weight"][0]
    rope = c.get("use_rotary_positional_embeddings", False)
    alin, core = _mha(d, H, rope=hd if rope else 0)
    ffn, flin = _gelu_ffn(d, inner)
    ada = (Linear("norm1_ada", te, 6 * d, "col", rows="seq"),)
    ada2 = (Linear("norm2_ada", te, 6 * d, "col", rows="seq"),)
    misc = 2 * d + 6 * d + 4 * d + 4 * hd + 2 * d + 6 * d + inner + d
    layer = Layer(ada + alin, core, ffn, ada2 + flin, misc)
    io_pre = (Linear("patch_embed", in_c * p * p, d, "col", rows="img"), Linear("text_proj", text_dim, d, "col", rows="ctx"),
              Linear("time_embed_1", d, te, "col", rows="seq"), Linear("time_embed_2", te, te, "col", rows="seq"))
    io_post = (Linear("norm_out_ada", te, 2 * d, "col", rows="seq"), Linear("proj_out", d, out_p, "col", rows="img"))
    io_misc = d + d + te + te + 2 * d + out_p + 2 * d + 2 * d   # biases + norm_final + norm_out.norm
    tl = c.get("max_text_seq_length", 226)
    vt = c.get("temporal_compression_ratio", 4)
    wl = replace(wl, patch=(1, p, p), text_tokens=tl, prefix_tokens=tl, vae_t=vt)
    arch = "CogVideoX DiT（[文本 ‖ 视频] 联合全 3D 注意力" + ("，3D RoPE）" if rope else "，sincos 位置）")
    return _finish(model_id, hf_id, rel, arch, d, L, layer, io_pre, io_post, io_misc, 0, wl, "")


# ---------------------------------------------------------------- ESM-2
def build_esm(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    d, H, f, L, V = c["hidden_size"], c["num_attention_heads"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    hd = d // H
    alin, core = _mha(d, H, rope=hd if c.get("position_embedding_type") == "rotary" else 0)
    ffn, flin = _gelu_ffn(d, f)
    misc = 4 * d + 2 * d + f + d + 2 * d                    # biases + attention LN + output LN
    layer = Layer(alin, core, ffn, flin, misc)
    io_post = (Linear("lm_dense", d, d, "col"),)
    pos = math.prod(sh.get("esm.embeddings.position_embeddings.weight", [0]))
    contact = math.prod(sh.get("esm.contact_head.regression.weight", [0])) + 1
    io_misc = pos + contact + d + 2 * d + V                  # unused abs-position table, contact head, lm biases + LN
    max_pos = c.get("max_position_embeddings", 1026)
    wl = replace(wl, max_seq=max_pos - 2, special_tokens=2)
    spec = _finish(model_id, hf_id, rel, "ESM-2 编码器（双向注意力 + RoPE）", d, L, layer, (), io_post, io_misc,
                   V, wl, "")
    head_copy = math.prod(sh.get("lm_head.decoder.weight", [0]))   # tied decoder stored again by the safetensors PR
    notes = list(spec.notes)
    if head_copy and spec.release_params:
        spec = replace(spec, release_params=spec.release_params - head_copy)
        notes.append(f"发布另存了一份 tied MLM decoder（{head_copy / 1e6:.2f}M），已去重")
    if pos:
        notes.append(f"绝对位置表（{pos / 1e6:.2f}M 参数）随发布存储，但 ESM-2 用 RoPE，推理不读取：只计存储")
    notes.append("contact head（逻辑回归，各层注意力图 → 接触图）只计参数；默认不输出接触图，其注意力图读写未计")
    return replace(spec, notes=tuple(notes))


# ---------------------------------------------------------------- common tail
def _finish(model_id, hf_id, rel, arch, d, L, layer, io_pre, io_post, io_misc, vocab, wl, _):
    fm = _formats(rel)
    fmd = dict(fm)
    dom = "gen" if wl.kind == "gen" else "protein"
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=d, vocab=vocab, layers=tuple([layer] * L),
        tie_embeddings=bool(vocab), formats=fm, act_fmt=_act_fmt(fmd), kv_fmt=_act_fmt(fmd),
        release_params=rel.get("params_llm"), release_bytes=rel.get("bytes_llm"), quantized_release=False,
        domain=dom, kv_cache=False, io_pre=io_pre, io_post=io_post, io_misc_params=io_misc,
        final_norm_params=(2 * d if dom == "protein" else 0), adaln=(dom == "gen"), workload=wl)
    notes, reasons = [], []
    w = fmd.get("attn")
    if w and w.fmt == "fp32":
        notes.append(f"发布权重为 fp32（4 B / 参数，存储与读流量按 fp32）；激活按 {spec.act_fmt}"
                     + ("（参考实现 autocast）" if dom == "gen" else "「假设」") + "；芯片无 fp32 原生 MAC → 逐 GEMM 把权重转换为 bf16，"
                     "转换开销计入向量单元；可用 dtype what-if 看 bf16 发布")
    if dom == "gen":
        sub = (rel.get("source") or {}).get("subfolder", "")
        um = _unmodelled(rel, sub)
        parts = []
        if um.get("text_encoder"):
            parts.append(f"文本编码器（{_gb(um['text_encoder'])}）")
        if um.get("vae"):
            parts.append(f"VAE（{_gb(um['vae'])}）")
        reasons.append("只评估 DiT 去噪主干：" + "、".join(parts or ["文本编码器与 VAE"]) +
                       "未建模——不计其权重存储与时间（文本编码每请求一次、VAE 解码每段一次）")
        notes.append(f"每个去噪步 {wl.cfg} 次前向（classifier-free guidance：cond + uncond，batch ×{wl.cfg}）；"
                     f"默认 {wl.steps} 步")
        if layer.cross is not None:
            notes.append("跨注意力的文本 K/V 按参考实现每步重算（可缓存，未作为优化建模）")
    pc = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
    if abs(pc) > 0.02:
        reasons.append(f"参数与发布相差 {pc:+.1%}")
    return replace(spec, notes=tuple(notes), coverage_reasons=tuple(reasons),
                   coverage="partial" if reasons else "full")


BUILDERS = {"wan": build_wan, "cogvideox": build_cogvideox, "esm": build_esm}


def from_domain_release(model_id: str, hf_id: str, builder: str, wl: NativeWorkload) -> ModelSpec:
    rel = load_release(hf_id)
    if rel is None:
        raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
    return BUILDERS[builder](model_id, hf_id, rel, wl)


# ---------------------------------------------------------------- workload resolution
@dataclass(frozen=True)
class ResolvedWorkload:
    kind: str
    tokens: int              # tokens per sequence through the trunk (incl. joint-attention text prefix)
    ctx: int                 # conditioning tokens per sequence (cross-attention K/V source)
    seqs_per_request: int    # forward sequences per request per step (CFG)
    steps: int               # forward passes per request (denoise steps; 1 for encoders)
    units: float             # output units per request (video frames; 1 sequence)
    unit: str                # frame | seq
    info: dict
    warnings: tuple[str, ...]


def resolve_workload(spec: ModelSpec, w) -> ResolvedWorkload:
    """Model defaults overridden by the scenario's ``workload`` block (0 = model default)."""
    d = spec.workload
    warns: list[str] = []
    if spec.domain == "gen":
        F, Hh, Ww = w.frames or d.frames, w.height or d.height, w.width or d.width
        steps, cfg = w.steps or d.steps, w.cfg or d.cfg
        pt, ph, pw = d.patch
        lt = (F - 1) // d.vae_t + 1
        if (F - 1) % d.vae_t:
            warns.append(f"帧数 {F} 不满足 4k+1：潜空间按 {lt} 帧计")
        lh, lw = math.ceil(Hh / d.vae_s), math.ceil(Ww / d.vae_s)
        gt, gh, gw = math.ceil(lt / pt), math.ceil(lh / ph), math.ceil(lw / pw)
        if Hh % (d.vae_s * ph) or Ww % (d.vae_s * pw):
            warns.append(f"分辨率 {Ww}×{Hh} 不是 {d.vae_s * pw} 的整数倍：按向上取整的 patch 网格计")
        vid = gt * gh * gw
        if (F, Hh, Ww) != (d.frames, d.height, d.width):
            warns.append(f"工作负载偏离发布默认（{d.width}×{d.height}，{d.frames} 帧）")
        info = {"frames": F, "height": Hh, "width": Ww, "fps": d.fps, "video_s": F / d.fps if d.fps else None,
                "latent": [lt, lh, lw], "grid": [gt, gh, gw], "video_tokens": vid, "text_tokens": d.text_tokens,
                "seq_tokens": vid + d.prefix_tokens, "steps": steps, "cfg": cfg,
                "default": {"frames": d.frames, "height": d.height, "width": d.width, "steps": d.steps, "cfg": d.cfg},
                "source": d.source}
        return ResolvedWorkload("gen", vid + d.prefix_tokens, d.text_tokens, cfg, steps, float(F), "frame", info,
                                tuple(warns))
    L = w.seq_len or d.seq_len
    if d.max_seq and L > d.max_seq:
        warns.append(f"序列长度 {L} 超过训练长度 {d.max_seq} 残基（RoPE 可外推，精度不在本工具范围）")
    info = {"seq_len": L, "tokens": L + d.special_tokens, "max_seq": d.max_seq,
            "default": {"seq_len": d.seq_len}, "source": d.source}
    return ResolvedWorkload("protein", L + d.special_tokens, 0, 1, 1, 1.0, "seq", info, tuple(warns))
