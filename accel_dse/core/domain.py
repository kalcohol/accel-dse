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
  Wan2.2     A14B = two Wan2.1-14B-shaped experts (high- / low-noise); each step runs one, both are stored.
  HunyuanVideo  20 dual-stream MMDiT blocks (per-stream weights, joint attention over [video ‖ text]) +
             40 single-stream parallel blocks (fused QKV+MLP in, fused out); guidance-distilled (1 forward / step).
  LTX-Video  self-attention + T5 cross-attention, 1×1×1 patch on a 32×32×8 VAE (few, wide tokens).
  Mochi-1    AsymmDiT: dual-stream joint attention with a narrower text stream (1536) and its own FFN.
  STDiT3     Open-Sora 1.2: *factorized* attention as released — spatial blocks attend within a latent frame,
             temporal blocks along time at one position; T5 cross-attention in every block.
  MiniMax-H3 single-stream omni transformer over one packed [text ‖ audio ‖ video] sequence (full attention);
             the 13 B AdaLN branch is precomputed per timestep at inference (README) and not loaded.
Protein (ESM-2): bidirectional pre-LN encoder with rotary attention, GELU FFN, tied MLM head.

What is *not* modelled is listed in ``coverage_reasons`` (text encoder / VAE of the video pipelines).
"""

from __future__ import annotations

import math
import re
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


def _c_dtype(key: str) -> str:
    from .pipeline import component_data
    return component_data(key)["dtype"]


def _unmodelled(rel: dict, subfolder: str, skip: tuple[str, ...] = ()) -> dict[str, int]:
    """Weight bytes of the pipeline components outside the modelled denoiser (text encoder, VAE …).
    ``skip``: repo folders that are other checkpoints of the same model (original-format copies, other tasks).
    Precision variants (``*.bf16.*``) and duplicate shard sets keep only the default / largest set."""
    files = (rel.get("source") or {}).get("repo_files") or {}
    mine = set(subfolder.split("+")) if subfolder else set()
    sets: dict[str, dict[str, int]] = {}
    for path, size in files.items():
        top = path.split("/", 1)[0] if "/" in path else ""
        if top in mine or top in skip or (not subfolder and path.startswith("diffusion_pytorch_model")):
            continue
        low = path.lower()
        if re.search(r"\.(bf16|fp16)\.", low) and any(re.sub(r"\.(bf16|fp16)\.", ".", low) == q.lower() for q in files):
            continue
        key = ("audio_vae" if "audio_vae" in low else
               "text_encoder" if ("t5" in low or top.startswith("text_encoder")) else
               "vae" if ("vae" in low or top == "vae") else top or path)
        m = re.search(r"-of-(\d+)", path)
        sets.setdefault(key, {}).setdefault(m.group(1) if m else path, 0)
        sets[key][m.group(1) if m else path] += size
    out = {}
    for key, groups in sets.items():
        shards = {g: v for g, v in groups.items() if g.isdigit()}
        single = sum(v for g, v in groups.items() if not g.isdigit())
        out[key] = single + (max(shards.values()) if shards else 0)
    return out


# ---------------------------------------------------------------- header-driven helpers (0.42 builders)
BLOCK_RE = re.compile(r"(blocks|layers|layer|transformer_blocks)\.(\d+)\.")


def _n(shape) -> int:
    return math.prod(shape) if shape else 1


def _lin(sh: dict, name: str, short: str = "", split: str = "col", rows: str = "tok", role: str = "") -> Linear:
    """Linear from a header weight ``[out, in, (kernel…)]`` (conv patch embedders flatten to in·kernel)."""
    w = sh[name + ".weight"]
    return Linear(short or name, _n(w[1:]), w[0], split, rows=rows, role=role)


def _misc(sh: dict, prefix: str, lins: list[str], cached: tuple[str, ...] = ()) -> tuple[int, int]:
    """(misc, cached) params of one block: every header tensor under ``prefix`` that is not a modelled GEMM weight
    (biases, norms, modulation tables).  A 2-D weight ≥ 256 × 256 that is neither modelled nor cached raises, so a
    GEMM can never silently become storage-only."""
    used = {prefix + n + ".weight" for n in lins}
    misc = cache = 0
    for t, shp in sh.items():
        if not t.startswith(prefix) or t in used:
            continue
        if any(t.startswith(prefix + c) for c in cached):
            cache += _n(shp)
            continue
        if t.endswith(".weight") and len(shp) >= 2 and min(shp[0], _n(shp[1:])) >= 256:
            raise ValueError(f"unmodelled GEMM weight {t} {shp}")
        misc += _n(shp)
    return misc, cache


def _io_misc(sh: dict, io: tuple[Linear, ...], names: list[str], allow: tuple[str, ...] = ()) -> int:
    """Params of the non-block tensors that are not io GEMM weights (biases, tables, buffers)."""
    used = {n + ".weight" for n in names}
    tot = 0
    for t, shp in sh.items():
        if BLOCK_RE.search(t) or t in used:
            continue
        if (t.endswith(".weight") and len(shp) >= 2 and min(shp[0], _n(shp[1:])) >= 256
                and not any(a in t for a in allow)):
            raise ValueError(f"unmodelled io GEMM weight {t} {shp}")
        tot += _n(shp)
    return tot


def _io(sh: dict, specs: list[tuple]) -> tuple[tuple[Linear, ...], list[str]]:
    """``[(tensor, short, rows[, role]), …]`` → io linears (column split) and the tensor names used."""
    lins = tuple(_lin(sh, t, s_, "col", rows=r, role=(x[0] if x else "")) for t, s_, r, *x in specs)
    return lins, [t for t, *_ in specs]


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


# ---------------------------------------------------------------- Wan2.2 A14B (two noise-level experts)
def build_wan22(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    spec = build_wan(model_id, hf_id, rel, wl)
    ex = rel.get("experts") or {}
    names = list(ex)
    standby = ex[names[1]] if len(names) > 1 else 0
    notes = list(spec.notes)
    notes.insert(0, f"MoE 去噪：{' / '.join(names)} 两个 14B 专家（结构同 Wan2.1-14B），按噪声级切换（boundary 0.875：高噪声"
                    "步用前者）——每步只算一个专家，两个都常驻存储（另一个计 standby 存储，每步不读）")
    spec = replace(spec, standby_params=standby, notes=tuple(notes),
                   arch="Wan2.2 A14B DiT（高/低噪声两专家，每步一个；全 3D 注意力 + 文本跨注意力）")
    return _recheck(spec)


def _recheck(spec: ModelSpec) -> ModelSpec:
    """Re-run the ±2 % parameter check after a builder changed the parameter split."""
    reasons = [r for r in spec.coverage_reasons if not r.startswith("参数与发布相差")]
    pc = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
    if abs(pc) > 0.02:
        reasons.append(f"参数与发布相差 {pc:+.1%}")
    return replace(spec, coverage_reasons=tuple(reasons), coverage="partial" if reasons else "full")


# ---------------------------------------------------------------- HunyuanVideo (dual-stream + single-stream MMDiT)
def build_hunyuan(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    H, hd = c["num_attention_heads"], c["attention_head_dim"]
    d, Ld, Ls = H * hd, c["num_layers"], c["num_single_layers"]
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    p = "transformer_blocks.0."
    a = [("attn.to_q", "img_q", "col", "img"), ("attn.to_k", "img_k", "col", "img"), ("attn.to_v", "img_v", "col", "img"),
         ("attn.to_out.0", "img_o", "row", "img"), ("attn.add_q_proj", "txt_q", "col", "txt"),
         ("attn.add_k_proj", "txt_k", "col", "txt"), ("attn.add_v_proj", "txt_v", "col", "txt"),
         ("attn.to_add_out", "txt_o", "row", "txt"), ("norm1.linear", "img_mod", "col", "seq"),
         ("norm1_context.linear", "txt_mod", "col", "seq")]
    f = [("ff.net.0.proj", "img_up", "col", "img"), ("ff.net.2", "img_down", "row", "img"),
         ("ff_context.net.0.proj", "txt_up", "col", "txt"), ("ff_context.net.2", "txt_down", "row", "txt")]
    dff = sh[p + "ff.net.0.proj.weight"][0]
    misc, _ = _misc(sh, p, [n for n, *_ in a + f])
    double = Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core, Ffn("dense", d_ff=dff, gated=False),
                   tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc)
    p = "single_transformer_blocks.0."
    a = [("attn.to_q", "q", "col", "tok"), ("attn.to_k", "k", "col", "tok"), ("attn.to_v", "v", "col", "tok"),
         ("norm.linear", "mod", "col", "seq")]
    f = [("proj_mlp", "mlp_up", "col", "tok"), ("proj_out", "fused_out", "row", "tok")]
    sff = sh[p + "proj_mlp.weight"][0]
    misc, _ = _misc(sh, p, [n for n, *_ in a + f])
    single = Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core, Ffn("dense", d_ff=sff, gated=False),
                   tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc, fused_out=True)
    specs = [("x_embedder.proj", "patch_embed", "img"), ("context_embedder.proj_in", "text_in", "txt")]
    for e in ("context_embedder.time_text_embed.timestep_embedder", "context_embedder.time_text_embed.text_embedder",
              "time_text_embed.timestep_embedder", "time_text_embed.guidance_embedder", "time_text_embed.text_embedder"):
        specs += [(e + ".linear_1", e.split(".")[-1] + "_1", "seq"), (e + ".linear_2", e.split(".")[-1] + "_2", "seq")]
    nref = c.get("num_refiner_layers", 2)
    for i in range(nref):
        r = f"context_embedder.token_refiner.refiner_blocks.{i}."
        specs += [(r + n, f"refiner{i}.{n.split('.')[-1] if n != 'attn.to_out.0' else 'o'}", "txt", "cond")
                  for n in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")]
        specs += [(r + "norm_out.linear", f"refiner{i}.mod", "seq", "cond")]
    io_pre, used = _io(sh, specs)
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    pe = sh["x_embedder.proj.weight"]
    wl = replace(wl, patch=tuple(pe[2:5]), text_tokens=wl.text_tokens or 256, prefix_tokens=wl.text_tokens or 256)
    spec = _finish(model_id, hf_id, rel, f"HunyuanVideo MMDiT（{Ld} 双流 + {Ls} 单流，[视频 ‖ 文本] 联合全 3D 注意力）",
                   d, 0, [double] * Ld + [single] * Ls, io_pre, io_post, io_misc, 0, wl, "",
                   skip=(), extra_notes=(
                       f"token refiner（{nref} 层，文本 {wl.text_tokens} token 上的自注意力 Transformer）的 GEMM 计入，"
                       "其 256×256 注意力核忽略不计",
                       "guidance 强度作为嵌入输入（embedded guidance 6.0，官方 sample_video.py 默认）",
                       f"文本按 {wl.text_tokens} token（text_len）计入联合序列；参考实现按 mask 跳过 padding「假设」全长"))
    return spec


# ---------------------------------------------------------------- LTX-Video
def build_ltx(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    H, hd, L = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"]
    d = H * hd
    p = "transformer_blocks.0."
    a = [("attn1.to_q", "q"), ("attn1.to_k", "k"), ("attn1.to_v", "v"), ("attn1.to_out.0", "o")]
    x = [("attn2.to_q", "xq", "tok"), ("attn2.to_k", "xk", "ctx"), ("attn2.to_v", "xv", "ctx"),
         ("attn2.to_out.0", "xo", "tok")]
    f = [("ff.net.0.proj", "up", "col"), ("ff.net.2", "down", "row")]
    misc, _ = _misc(sh, p, [n for n, *_ in a + x + f])
    alin = tuple(_lin(sh, p + n, s_, "row" if s_ == "o" else "col") for n, s_ in a)
    xlin = tuple(_lin(sh, p + n, s_, "row" if s_ == "xo" else "col", rows=r) for n, s_, r in x)
    flin = tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in f)
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    xcore = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, causal=False, cross=True)
    layer = Layer(alin, core, Ffn("dense", d_ff=sh[p + "ff.net.0.proj.weight"][0], gated=False), flin, misc,
                  cross_linears=xlin, cross=xcore)
    io_pre, used = _io(sh, [("proj_in", "patch_embed", "img"), ("caption_projection.linear_1", "caption_1", "ctx"),
                            ("caption_projection.linear_2", "caption_2", "ctx"),
                            ("time_embed.emb.timestep_embedder.linear_1", "time_1", "seq"),
                            ("time_embed.emb.timestep_embedder.linear_2", "time_2", "seq"),
                            ("time_embed.linear", "time_mod", "seq")])
    io_post, used2 = _io(sh, [("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    wl = replace(wl, patch=(c.get("patch_size_t", 1), c["patch_size"], c["patch_size"]), prefix_tokens=0)
    return _finish(model_id, hf_id, rel, "LTX-Video DiT（全 3D 注意力 + T5 跨注意力，32×32×8 VAE、1×1×1 patch）",
                   d, L, layer, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       "diffusers transformer/ = LTX-Video 2B v0.9（与 ltx-video-2b-v0.9.safetensors 去掉 VAE 后同尺寸）；"
                       "13B 0.9.7/0.9.8 为单文件原始格式，未单列",))


# ---------------------------------------------------------------- Mochi-1 (asymmetric dual-stream)
def build_mochi(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    H, hd, L = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"]
    d = H * hd
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)

    def block(i: int, last: bool) -> Layer:
        p = f"transformer_blocks.{i}."
        a = [("attn1.to_q", "img_q", "col", "img"), ("attn1.to_k", "img_k", "col", "img"),
             ("attn1.to_v", "img_v", "col", "img"), ("attn1.to_out.0", "img_o", "row", "img"),
             ("attn1.add_q_proj", "txt_q", "col", "txt"), ("attn1.add_k_proj", "txt_k", "col", "txt"),
             ("attn1.add_v_proj", "txt_v", "col", "txt"), ("norm1.linear", "img_mod", "col", "seq")]
        f = [("ff.net.0.proj", "img_gate_up", "col", "img"), ("ff.net.2", "img_down", "row", "img")]
        if last:        # context_pre_only: text tokens only feed the last attention (no text out / FFN)
            a += [("norm1_context.linear_1", "txt_mod", "col", "seq")]
        else:
            a += [("attn1.to_add_out", "txt_o", "row", "txt"), ("norm1_context.linear", "txt_mod", "col", "seq")]
            f += [("ff_context.net.0.proj", "txt_gate_up", "col", "txt"), ("ff_context.net.2", "txt_down", "row", "txt")]
        misc, _ = _misc(sh, p, [n for n, *_ in a + f])
        dff = sh[p + "ff.net.2.weight"][1]
        return Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core, Ffn("dense", d_ff=dff, gated=True),
                     tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc)

    layers = [block(0, False)] * (L - 1) + [block(L - 1, True)]
    io_pre, used = _io(sh, [("patch_embed.proj", "patch_embed", "img"),
                            ("time_embed.timestep_embedder.linear_1", "time_1", "seq"),
                            ("time_embed.timestep_embedder.linear_2", "time_2", "seq"),
                            ("time_embed.pooler.to_q", "pool_q", "seq"), ("time_embed.pooler.to_kv", "pool_kv", "ctx"),
                            ("time_embed.pooler.to_out", "pool_out", "seq"), ("time_embed.caption_proj", "text_in", "txt")])
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    pe = sh["patch_embed.proj.weight"]
    tl = c.get("max_sequence_length", 256)
    wl = replace(wl, patch=(1, pe[2], pe[3]), text_tokens=tl, prefix_tokens=tl)
    tw = sh["transformer_blocks.0.attn1.add_q_proj.weight"][1]
    return _finish(model_id, hf_id, rel, f"Mochi-1 AsymmDiT（{L} 层非对称双流：视频 {d} / 文本 {tw}，联合全 3D 注意力）",
                   d, 0, layers, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"文本流按 T5 {tl} token 全长计入联合序列（参考实现按有效长度裁剪：「假设」全长）",
                       "末层 context_pre_only：文本 token 只参与注意力，无文本输出投影 / FFN"))


# ---------------------------------------------------------------- Open-Sora STDiT3 (factorized spatial / temporal)
def build_stdit(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    d, H, L = c["hidden_size"], c["num_heads"], c["depth"]
    hd = d // H

    def block(kind: str) -> Layer:
        p = f"{kind}_blocks.0."
        a = [("attn.qkv", "qkv", "col", "tok"), ("attn.proj", "o", "row", "tok")]
        x = [("cross_attn.q_linear", "xq", "col", "tok"), ("cross_attn.kv_linear", "xkv", "col", "ctx"),
             ("cross_attn.proj", "xo", "row", "tok")]
        f = [("mlp.fc1", "up", "col", "tok"), ("mlp.fc2", "down", "row", "tok")]
        misc, _ = _misc(sh, p, [n for n, *_ in a + x + f])
        span = "spatial" if kind == "spatial" else "temporal"
        core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd if span == "temporal" else 0,
                        causal=False, span=span)
        xcore = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, causal=False, cross=True)
        return Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core,
                     Ffn("dense", d_ff=sh[p + "mlp.fc1.weight"][0], gated=False),
                     tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc,
                     cross_linears=tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in x), cross=xcore)

    sp_, tp_ = block("spatial"), block("temporal")
    layers = [sp_, tp_] * L
    io_pre, used = _io(sh, [("x_embedder.proj", "patch_embed", "img"), ("t_embedder.mlp.0", "time_1", "seq"),
                            ("t_embedder.mlp.2", "time_2", "seq"), ("t_block.1", "time_mod", "seq"),
                            ("fps_embedder.mlp.0", "fps_1", "seq"), ("fps_embedder.mlp.2", "fps_2", "seq"),
                            ("y_embedder.y_proj.fc1", "caption_1", "ctx"), ("y_embedder.y_proj.fc2", "caption_2", "ctx")])
    io_post, used2 = _io(sh, [("final_layer.linear", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2, allow=("y_embedding",))
    pe = sh["x_embedder.proj.weight"]
    wl = replace(wl, patch=tuple(pe[2:5]), text_tokens=c.get("model_max_length", 300), prefix_tokens=0)
    return _finish(model_id, hf_id, rel, f"Open-Sora STDiT3（{L} 空间块 + {L} 时间块分解注意力 + T5 跨注意力）",
                   d, 0, layers, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       "分解时空注意力（发布结构如此，非近似）：空间块在每个潜帧内 H·W token 上注意，时间块在每个空间位置沿 T 注意",
                       "y_embedding（300×4096 空文本表，CFG uncond 用）只计存储"),
                   ext_parts=("T5-XXL 文本编码器（DeepFloyd/t5-v1_1-xxl，独立仓库）", "VAE（hpcai-tech/OpenSora-VAE-v1.2，独立仓库）"))


# ---------------------------------------------------------------- MiniMax-H3 (single-stream omni, joint audio)
def build_h3(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    H, hd, L, d = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"], c["hidden_size"]
    p = "transformer_blocks.0."
    a = [("attn.to_q", "q", "col"), ("attn.to_k", "k", "col"), ("attn.to_v", "v", "col"), ("attn.to_out.0", "o", "row")]
    f = [("ff.net.0.proj", "gate_up", "col"), ("ff.net.2", "down", "row")]
    misc, cached = _misc(sh, p, [n for n, *_ in a + f], cached=("adaln_proj.",))
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=2 * 3 * c.get("rope_freq_dim", 16), causal=False)
    layer = Layer(tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in a), core,
                  Ffn("dense", d_ff=sh[p + "ff.net.2.weight"][1], gated=True),
                  tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in f), misc)
    specs = [("proj_in", "patch_embed", "img"), ("audio_proj_in", "audio_in", "aud"),
             ("context_embedder", "text_in", "txt", "cond"), ("time_embedder.linear_1", "time_1", "seq"),
             ("time_embedder.linear_2", "time_2", "seq")]
    nref = c.get("num_refiner_layers", 2)
    for i in range(nref):
        r = f"token_refiner.refiner_blocks.{i}."
        specs += [(r + n, f"refiner{i}.{n.split('.')[-1] if n != 'attn.to_out.0' else 'o'}", "txt", "cond")
                  for n in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")]
    io_pre, used = _io(sh, specs)
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq", "cond"), ("proj_out", "head", "img"),
                              ("audio_proj_out", "audio_head", "aud")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    wl = replace(wl, patch=tuple(c.get("patch_size", (1, 2, 2))), prefix_tokens=wl.text_tokens)
    spec = _finish(model_id, hf_id, rel, f"MiniMax-H3 单流 Omni Transformer（{L} 层，[文本 ‖ 音频 ‖ 视频] 打包序列全注意力）",
                   d, 0, [layer] * L, io_pre, io_post, io_misc, 0, wl, "", skip=("FL2VA", "Ref2VA", "transformer_ref"),
                   extra_notes=(
                       f"AdaLN 分支（每层 adaln_proj，共 {cached * L / 1e9:.1f}B 参数）按 README 在推理部署中按时间步预计算缓存、"
                       "不加载：不计存储与读流量（缓存的调制表约 50 步 × 层 × 18 × hidden，< 0.5 GB，未计）",
                       "num_inference_steps 50 = 49 次模型前向（终点 σ=0 不算，diffusers 文档）",
                       f"文本按 {wl.text_tokens} token 计入打包序列「假设」（H3-Context-IR 提示词长度不定）；"
                       f"立体声音频按 {wl.audio_per_s} latent/s × {wl.audio_channels} 声道计入序列",
                       "开源版只提供全注意力推理（稀疏注意力未发布）：按全注意力建模",
                       f"token refiner（{nref} 层整宽 Transformer）的 GEMM 计入，其文本自注意力核忽略不计"))
    return _recheck(replace(spec, cached_params=cached * L))


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
    # 1026 learned positions = 2 (padding_idx offset) + 1024 tokens incl. <cls>/<eos> → 1022 residues
    wl = replace(wl, max_seq=max_pos - 4, special_tokens=2)
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
def _finish(model_id, hf_id, rel, arch, d, L, layer, io_pre, io_post, io_misc, vocab, wl, _, skip=(), extra_notes=(),
            ext_parts=()):
    fm = _formats(rel)
    fmd = dict(fm)
    dom = "gen" if wl.kind == "gen" else "protein"
    layers = tuple(layer) if isinstance(layer, (list, tuple)) else tuple([layer] * L)
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=d, vocab=vocab, layers=layers,
        tie_embeddings=bool(vocab), formats=fm, act_fmt=_act_fmt(fmd), kv_fmt=_act_fmt(fmd),
        release_params=rel.get("params_llm"), release_bytes=rel.get("bytes_llm"), quantized_release=False,
        domain=dom, kv_cache=False, io_pre=io_pre, io_post=io_post, io_misc_params=io_misc,
        final_norm_params=(2 * d if dom == "protein" else 0), adaln=(dom == "gen"), workload=wl)
    notes, reasons = [], []
    w = fmd.get("attn")
    if w and w.fmt == "fp32":
        notes.append(f"发布权重为 fp32（4 B / 参数，存储与读流量按 fp32）；激活按 {spec.act_fmt}"
                     + ("（参考实现 autocast）" if dom == "gen" else "「假设」") + "；芯片无 fp32 原生 MAC → 逐 GEMM 把权重转换为 bf16，"
                     "转换开销计入向量单元；可用 dtype what-if 看 bf16 发布" + ("、用激活 dtype what-if 看 fp32 激活" if dom == "protein" else ""))
    if dom == "gen":
        sub = (rel.get("source") or {}).get("subfolder", "")
        um = _unmodelled(rel, sub, skip)
        parts = []
        if um.get("text_encoder"):
            parts.append(f"文本编码器（{_gb(um['text_encoder'])}）")
        if um.get("vae"):
            parts.append(f"VAE（{_gb(um['vae'])}）")
        if um.get("audio_vae"):
            parts.append(f"音频 VAE（{_gb(um['audio_vae'])}）")
        from .pipeline import pipeline_for, stored_bytes
        pl = pipeline_for(model_id)
        if pl is None:
            reasons.append("只评估 DiT 去噪主干：" + "、".join(parts or list(ext_parts) or ["文本编码器与 VAE"]) +
                           "未建模——不计其权重存储与时间（文本编码每请求一次、VAE 解码每段一次）")
        else:
            comps = [f"{t.label}（{_gb(stored_bytes(t.key))} {_c_dtype(t.key)}，{t.tokens} token/提示）" for t in pl.text]
            comps += [f"{v.label}（{_gb(stored_bytes(v.key))} {_c_dtype(v.key)}）" for v in (pl.vae, pl.audio) if v]
            notes.append("整条 pipeline 计入（默认，workload.pipeline=false 只看 DiT）：" + "、".join(comps) +
                         "——按发布检查点头的算子图建模，时间串行加在去噪前后（单卡执行「假设」），"
                         "权重按加载的整个检查点计入首 / 末流水级卡的存储；见 docs/MODEL.md §11.4")
            notes.extend(n for n in [t.note for t in pl.text] + [v.note for v in (pl.vae, pl.audio) if v] + [pl.note]
                         if n)
        if wl.cfg > 1:
            notes.append(f"每个去噪步 {wl.cfg} 次前向（classifier-free guidance：cond + uncond，batch ×{wl.cfg}）；"
                         f"默认 {wl.steps} 步")
        else:
            notes.append(f"每个去噪步 1 次前向（guidance 蒸馏，无 uncond 分支）；默认 {wl.steps} 步")
        if any(l.cross is not None for l in layers):
            notes.append("跨注意力的文本 K/V 按参考实现每步重算（可缓存，未作为优化建模）")
    notes.extend(extra_notes)
    pc = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
    if abs(pc) > 0.02:
        reasons.append(f"参数与发布相差 {pc:+.1%}")
    return replace(spec, notes=tuple(notes), coverage_reasons=tuple(reasons),
                   coverage="partial" if reasons else "full")


BUILDERS = {"wan": build_wan, "wan22": build_wan22, "cogvideox": build_cogvideox, "hunyuan": build_hunyuan,
            "ltx": build_ltx, "mochi": build_mochi, "stdit": build_stdit, "h3": build_h3, "esm": build_esm}
from .structure import BUILDERS as _STRUCTURE_BUILDERS  # noqa: E402  (0.43 protein structure models)
BUILDERS.update(_STRUCTURE_BUILDERS)


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
    frames: int = 0          # latent frames of the video grid (factorized attention)
    aux: int = 0             # joint audio rows in the sequence
    pair: object = None      # structure models: ir.PairDims (grids, recycles, diffusion steps, samples)


def _latent_frames(d: NativeWorkload, F: int, warns: list[str]) -> int:
    if d.vae_frames == "chunk17":           # Open-Sora 1.2 VAE: micro-batches of 17 frames → 5 latent frames each
        lt = math.ceil(F / 17) * ((17 - 1) // d.vae_t + 1)
        if F % 17:
            warns.append(f"帧数 {F} 不是 17 的整数倍：VAE 按 {math.ceil(F / 17)} 个 17 帧块（补齐）计")
        return lt
    if d.vae_frames == "h3":                # MiniMax-H3: frames snapped up to 17n+5 → 5n+2 latent frames
        Fa = F
        while Fa % 17 != 5:
            Fa += 1
        if Fa != F:
            warns.append(f"帧数 {F} 按 VAE 规则补齐到 17n+5 = {Fa}")
        return (Fa - 5) // 17 * 5 + 2
    lt = (F - 1) // d.vae_t + 1
    if (F - 1) % d.vae_t:
        warns.append(f"帧数 {F} 不满足 {d.vae_t}k+1：潜空间按 {lt} 帧计")
    return lt


def resolve_workload(spec: ModelSpec, w) -> ResolvedWorkload:
    """Model defaults overridden by the scenario's ``workload`` block (0 = model default)."""
    d = spec.workload
    warns: list[str] = []
    if spec.domain == "gen":
        F, Hh, Ww = w.frames or d.frames, w.height or d.height, w.width or d.width
        steps, cfg = w.steps or d.steps, w.cfg or d.cfg
        pt, ph, pw = d.patch
        lt = _latent_frames(d, F, warns)
        lh, lw = math.ceil(Hh / d.vae_s), math.ceil(Ww / d.vae_s)
        gt, gh, gw = math.ceil(lt / pt), math.ceil(lh / ph), math.ceil(lw / pw)
        if Hh % (d.vae_s * ph) or Ww % (d.vae_s * pw):
            warns.append(f"分辨率 {Ww}×{Hh} 不是 {d.vae_s * pw} 的整数倍：按向上取整的 patch 网格计")
        vid = gt * gh * gw
        if (F, Hh, Ww) != (d.frames, d.height, d.width):
            warns.append(f"工作负载偏离发布默认（{d.width}×{d.height}，{d.frames} 帧）")
        aud = round(F / d.fps * d.audio_per_s) * d.audio_channels if d.audio_per_s and d.fps else 0
        spans = {l.core.span for l in spec.layers}
        attn = "factorized" if spans - {"full"} else "full"
        info = {"frames": F, "height": Hh, "width": Ww, "fps": d.fps, "video_s": F / d.fps if d.fps else None,
                "latent": [lt, lh, lw], "grid": [gt, gh, gw], "video_tokens": vid, "text_tokens": d.text_tokens,
                "joint_text": d.prefix_tokens, "audio_tokens": aud, "attention": attn,
                "seq_tokens": vid + d.prefix_tokens + aud, "steps": steps, "cfg": cfg,
                "default": {"frames": d.frames, "height": d.height, "width": d.width, "steps": d.steps, "cfg": d.cfg},
                "source": d.source}
        if attn == "factorized":
            info["attn_groups"] = {"spatial": [gt, gh * gw], "temporal": [gh * gw, gt]}
        return ResolvedWorkload("gen", vid + d.prefix_tokens + aud, d.text_tokens, cfg,
                                steps, float(F), "frame", info, tuple(warns), frames=gt, aux=aud)
    L = w.seq_len or d.seq_len
    if d.max_seq and L > d.max_seq:
        warns.append(f"序列长度 {L} 超过训练长度 {d.max_seq} 残基" + ("（RoPE 可外推，精度不在本工具范围）"
                     if not spec.is_pair else "（裁剪长度；更长序列推理可行，精度不在本工具范围）"))
    info = {"seq_len": L, "tokens": L + d.special_tokens, "max_seq": d.max_seq,
            "default": {"seq_len": d.seq_len}, "source": d.source}
    pd = None
    if spec.is_pair:
        from .ir import PairDims
        msa = w.msa or d.msa
        rec = w.recycles or d.recycles
        steps = (w.steps or d.diff_steps) if d.diff_steps else 0
        smp = (w.samples or d.samples) if d.diff_steps else 1
        xmsa = d.xmsa
        if w.msa and d.xmsa and not d.msa:
            xmsa, msa = w.msa, 0
        atoms = math.ceil(L * d.atoms_per_res) if d.atoms_per_res else 0
        # 0.61.3: AF2 / OpenFold concatenate the template torsion-angle embeddings (T rows) to the MSA representation
        # (alphafold/model/modules.py EmbeddingsAndEvoformer; openfold model.py embed_templates → torch.cat([m, a]))
        msa_rows = msa + d.templates if d.tmpl_msa and msa else msa
        pd = PairDims(L, msa_rows, xmsa, d.templates, atoms, rec, steps, smp)
        info.update({"msa": msa, "msa_rows": msa_rows, "xmsa": xmsa, "templates": d.templates, "atoms": atoms, "recycles": rec,
                     "diff_steps": steps, "samples": smp, "pair_dim": d.pair_dim,
                     "default": {"seq_len": d.seq_len, "msa": d.msa or d.xmsa, "recycles": d.recycles,
                                 "steps": d.diff_steps, "samples": d.samples}})
    return ResolvedWorkload("protein", L + d.special_tokens, 0, 1, 1, 1.0, "seq", info, tuple(warns), pair=pd)
