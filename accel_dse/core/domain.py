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

What is *not* modelled is listed in ``coverage_reasons`` (e.g. MSA search / featurization of the structure models);
the video pipelines' text encoder and VAE decode are modelled (``pipeline``: te_s / decode_s) and not a reason.
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
                       f"token refiner（{nref} 层，文本 {wl.text_tokens} token 上的自注意力 Transformer）的 GEMM 与"
                       f"注意力核（{H} 头 × {hd}，双向，无 QK-norm / RoPE：diffusers HunyuanVideoIndividualTokenRefinerBlock）都计入；"
                       "参考实现按文本 mask 屏蔽 padding，按全长计「假设」",
                       "guidance 强度作为嵌入输入（embedded guidance 6.0，官方 sample_video.py 默认）",
                       f"文本按 {wl.text_tokens} token（text_len）计入联合序列；参考实现按 mask 跳过 padding「假设」全长"))
    # 0.260: the token refiner's self-attention core (heads from the refiner's to_q width / head dim, 「推导」)
    rh = sh["context_embedder.token_refiner.refiner_blocks.0.attn.to_q.weight"][0] // hd if nref else 0
    return replace(spec, io_attn=tuple((f"refiner{i}", "txt", rh, hd, 1, "") for i in range(nref)))


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
                       f"token refiner（{nref} 层整宽 Transformer）的 GEMM 与文本自注意力核（to_q 宽 / norm_q 头维 → "
                       "头数「推导」，QK-norm）都计入；无 RoPE、无 mask「来源」：diffusers transformer_minimax_h3.py "
                       "MiniMaxH3TokenRefinerBlock（'No AdaLN and no rotary embedding'，attn 不传 rotary_emb）"))
    # 0.260: refiner attention core — heads = to_q width / the per-head norm_q size (7168 / 128 = 56), 「推导」
    rp = "token_refiner.refiner_blocks.0.attn."
    rd = sh[rp + "norm_q.weight"][0] if rp + "norm_q.weight" in sh else hd
    io_attn = tuple((f"refiner{i}", "txt", sh[rp + "to_q.weight"][0] // rd, rd, 1,
                     "qknorm" if rp + "norm_q.weight" in sh else "") for i in range(nref)) if nref else ()
    return _recheck(replace(spec, cached_params=cached * L, io_attn=io_attn))


# ---------------------------------------------------------------- 0.115 image DiT (single 2-D latent frame)
def _image_wl(wl: NativeWorkload, patch: int, text: int) -> NativeWorkload:
    """Image = one latent frame: frames 1, temporal patch 1, no temporal attention; text joins the attention sequence."""
    return replace(wl, frames=1, fps=0, vae_t=1, patch=(1, patch, patch), text_tokens=text, prefix_tokens=text)


def build_qwenimage(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """Qwen-Image (QwenImageTransformer2DModel): 60 dual-stream MMDiT blocks (per-stream QKV / out / GELU-tanh MLP /
    AdaLN modulation, joint attention over [text ‖ image], RMS QK-norm, 3-axis MSRoPE)."""
    c, sh = rel["config"], rel["shapes"]
    H, hd, L = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"]
    d = H * hd
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    p = "transformer_blocks.0."
    a = [("attn.to_q", "img_q", "col", "img"), ("attn.to_k", "img_k", "col", "img"), ("attn.to_v", "img_v", "col", "img"),
         ("attn.to_out.0", "img_o", "row", "img"), ("attn.add_q_proj", "txt_q", "col", "txt"),
         ("attn.add_k_proj", "txt_k", "col", "txt"), ("attn.add_v_proj", "txt_v", "col", "txt"),
         ("attn.to_add_out", "txt_o", "row", "txt"), ("img_mod.1", "img_mod", "col", "seq"),
         ("txt_mod.1", "txt_mod", "col", "seq")]
    f = [("img_mlp.net.0.proj", "img_up", "col", "img"), ("img_mlp.net.2", "img_down", "row", "img"),
         ("txt_mlp.net.0.proj", "txt_up", "col", "txt"), ("txt_mlp.net.2", "txt_down", "row", "txt")]
    misc, _ = _misc(sh, p, [n for n, *_ in a + f])
    layer = Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core,
                  Ffn("dense", d_ff=sh[p + "img_mlp.net.0.proj.weight"][0], gated=False),
                  tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc)
    io_pre, used = _io(sh, [("img_in", "patch_embed", "img"), ("txt_in", "text_in", "txt", "cond"),
                            ("time_text_embed.timestep_embedder.linear_1", "time_1", "seq"),
                            ("time_text_embed.timestep_embedder.linear_2", "time_2", "seq")])
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq", "cond"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    wl = _image_wl(wl, c["patch_size"], wl.text_tokens or 256)
    return _finish(model_id, hf_id, rel, f"Qwen-Image MMDiT（{L} 双流，[文本 ‖ 图像] 联合注意力；单帧 2D 潜空间）",
                   d, 0, [layer] * L, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"图像 = 1 个潜帧（VAE 空间 ×{wl.vae_s}、2×2 patch 打包；无时间维 / 时间注意力）",
                       f"文本（Qwen2.5-VL 隐状态，去掉模板前缀）按 {wl.text_tokens} token 计入联合序列「假设」"
                       "（参考 pipeline 按批内最长提示补齐，长度不定）"))


def build_qwenimage21(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """Qwen-Image-2.1 (QwenImage21Transformer2DModel): 32 single-stream blocks over [text ‖ image] with shared
    weights (QKV / out / SwiGLU MLP), RMS QK-norm; AdaLN from one shared timestep modulation (no per-block tables)."""
    c, sh = rel["config"], rel["shapes"]
    H, hd, L = c["num_attention_heads"], c["attention_head_dim"], c["num_layers"]
    d = H * hd
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    p = "transformer_blocks.0."
    a = [("attn.to_q", "q", "col"), ("attn.to_k", "k", "col"), ("attn.to_v", "v", "col"), ("attn.to_out.0", "o", "row")]
    f = [("img_mlp.gate_layer", "gate", "col"), ("img_mlp.proj", "up", "col"), ("img_mlp.out", "down", "row")]
    misc, _ = _misc(sh, p, [n for n, *_ in a + f])
    layer = Layer(tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in a), core,
                  Ffn("dense", d_ff=sh[p + "img_mlp.proj.weight"][0], gated=True),
                  tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in f), misc)
    io_pre, used = _io(sh, [("img_in", "patch_embed", "img"), ("txt_in.in_layer", "text_in_1", "txt", "cond"),
                            ("txt_in.out_layer", "text_in_2", "txt", "cond"),
                            ("time_text_embed.timestep_embedder.linear_1", "time_1", "seq"),
                            ("time_text_embed.timestep_embedder.linear_2", "time_2", "seq"),
                            ("modulation.1", "time_mod", "seq", "cond")])
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq", "cond"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    wl = _image_wl(wl, c["patch_size"], wl.text_tokens or 256)
    cc = bool(c.get("causal_condition", False))
    # 0.116: causal_condition → block-causal mask (text causal over text, image over all) and, as QwenImage21Pipeline's
    # use_kv_cache=True default, the text-prefix K/V extracted at step 1 and reused by steps 2…n (workload.prefix_kv)
    wl = replace(wl, cond_causal=cc, prefix_kv=cc)
    return _finish(model_id, hf_id, rel, f"Qwen-Image-2.1 单流 DiT（{L} 层，[文本 ‖ 图像] 共享权重联合注意力；单帧 2D 潜空间）",
                   d, 0, [layer] * L, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"图像 = 1 个潜帧（VAE 空间 ×{wl.vae_s}、z=64、1×1 patch；无时间维 / 时间注意力）",
                       f"文本（Qwen3-VL 隐状态）按 {wl.text_tokens} token 计入联合序列「假设」；causal_condition：文本 token 只按因果"
                       "看文本、图像 token 看 [文本 ‖ 图像]（块因果掩码），文本前缀从 t=0 调制、与时间步无关 → 参考实现第 1 步"
                       "抽取前缀 K/V，后续步只算图像 token、注意力读缓存（默认开，workload.prefix_kv = off 关闭）",
                       "Turbo：8 步蒸馏采样（sample_sigmas 8 个），CFG = 1（README）"))


def build_flux2(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """FLUX.2 (Flux2Transformer2DModel): dual-stream MMDiT blocks + single-stream parallel blocks (fused
    QKV+MLP-in, fused out); shared modulation projections (no per-block tables), SwiGLU MLP, no biases."""
    c, sh = rel["config"], rel["shapes"]
    H, hd = c["num_attention_heads"], c["attention_head_dim"]
    d, Ld, Ls = H * hd, c["num_layers"], c["num_single_layers"]
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    p = "transformer_blocks.0."
    a = [("attn.to_q", "img_q", "col", "img"), ("attn.to_k", "img_k", "col", "img"), ("attn.to_v", "img_v", "col", "img"),
         ("attn.to_out.0", "img_o", "row", "img"), ("attn.add_q_proj", "txt_q", "col", "txt"),
         ("attn.add_k_proj", "txt_k", "col", "txt"), ("attn.add_v_proj", "txt_v", "col", "txt"),
         ("attn.to_add_out", "txt_o", "row", "txt")]
    f = [("ff.linear_in", "img_gate_up", "col", "img"), ("ff.linear_out", "img_down", "row", "img"),
         ("ff_context.linear_in", "txt_gate_up", "col", "txt"), ("ff_context.linear_out", "txt_down", "row", "txt")]
    dff = sh[p + "ff.linear_out.weight"][1]
    misc, _ = _misc(sh, p, [n for n, *_ in a + f])
    double = Layer(tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in a), core, Ffn("dense", d_ff=dff, gated=True),
                   tuple(_lin(sh, p + n, s_, sp, r) for n, s_, sp, r in f), misc)
    p = "single_transformer_blocks.0."
    w_in, w_out = sh[p + "attn.to_qkv_mlp_proj.weight"], sh[p + "attn.to_out.weight"]
    sff = (w_in[0] - 3 * d) // 2
    assert w_in[0] == 3 * d + 2 * sff and w_out[1] == d + sff, (w_in, w_out)
    misc, _ = _misc(sh, p, ["attn.to_qkv_mlp_proj", "attn.to_out"])
    single = Layer((Linear("qkv", d, 3 * d, "col"),), core, Ffn("dense", d_ff=sff, gated=True),
                   (Linear("mlp_gate_up", d, 2 * sff, "col"), Linear("fused_out", d + sff, d, "row")), misc, fused_out=True)
    io_pre, used = _io(sh, [("x_embedder", "patch_embed", "img"), ("context_embedder", "text_in", "txt", "cond"),
                            ("time_guidance_embed.timestep_embedder.linear_1", "time_1", "seq"),
                            ("time_guidance_embed.timestep_embedder.linear_2", "time_2", "seq"),
                            ("double_stream_modulation_img.linear", "img_mod", "seq", "cond"),
                            ("double_stream_modulation_txt.linear", "txt_mod", "seq", "cond"),
                            ("single_stream_modulation.linear", "single_mod", "seq", "cond")])
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "seq", "cond"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    # 0.116: the pipeline packs 2×2 latent pixels of the ×8 VAE into one 128-channel token (x_embedder in = 32·2·2), so the
    # latent grid is H/8 and the DiT patch is 2 (was vae_s 16 / patch 1: same tokens, wrong VAE input size)
    wl = _image_wl(wl, 2 if sh["x_embedder.weight"][1] == 4 * 32 else c.get("patch_size", 1), wl.text_tokens or 512)
    return _finish(model_id, hf_id, rel, f"FLUX.2 MMDiT（{Ld} 双流 + {Ls} 单流，[文本 ‖ 图像] 联合注意力；单帧 2D 潜空间）",
                   d, 0, [double] * Ld + [single] * Ls, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"图像 = 1 个潜帧（VAE ×{wl.vae_s} + 2×2 打包 = 每 token {2 * wl.vae_s}×{2 * wl.vae_s} px；无时间维 / 时间注意力）",
                       f"文本（Qwen3 第 9/18/27 层隐状态拼接）按 pipeline 补齐到 {wl.text_tokens} token 计入联合序列",
                       "klein：4 步蒸馏，guidance_scale 1.0（无 CFG，README）"))


def build_flux3(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """FLUX 3 Image Klein (Flux3Transformer2DModel).  0.251 read it from the safetensors header; 0.252 from the BFL
    reference code (github black-forest-labs/flux-3-image-klein src/flux3/dit.py, pipeline.py, sampling.py) and diffusers
    main (models/transformers/transformer_flux3.py), 「来源」:
      three token modes — noisy image, reference images (image_cond: VAE-encoded clean references, 2×2 packed latents,
      x_embedders.image_cond 128 → 3072) and text (context_embedder 20480 → 3072);
      5 PER-MODE blocks (mode_blocks.image / mode_blocks.image_cond / txt_mode_blocks): each mode self-attends WITHIN
      ITSELF only (all references of a sample jointly) — parallel block x + gate·(attn_out(attn) + mlp_out(SwiGLU)) from
      one LayerNorm + per-mode shift / scale / gate (early_stream_modulations.<mode>);
      28 single blocks: joint attention over [image ‖ image_cond ‖ text] per sample, same parallel block, per-token
      modulation by mode (single_stream_modulations.<mode>);
      modulation per token: time_in MLP on every token's timestep (image t, text and references 0), both ModeModulation
      tables per token, ref_time_in (flow map) per sample, final adaLN on the image tokens; only image tokens reach
      the head.  No cache of reference tokens across steps: sampling.py runs dit(packed) every step with the references
      in the pack (Base CFG: [uncond, cond] samples each carry them)."""
    c, sh = rel["config"], rel["shapes"]
    H, hd = c["num_attention_heads"], c["attention_head_dim"]
    d, Ld, Ls = H * hd, c["num_layers"], c["num_single_layers"]
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    streams = {"img": "mode_blocks.image.0.", "ref": "mode_blocks.image_cond.0.", "txt": "txt_mode_blocks.0."}
    a, f = [], []
    for r, p in streams.items():
        a += [_lin(sh, p + "attn.to_q", f"{r}_q", "col", r), _lin(sh, p + "attn.to_k", f"{r}_k", "col", r),
              _lin(sh, p + "attn.to_v", f"{r}_v", "col", r), _lin(sh, p + "attn.to_out.0", f"{r}_o", "row", r)]
        f += [_lin(sh, p + "ff.linear_in", f"{r}_gate_up", "col", r), _lin(sh, p + "ff.linear_out", f"{r}_down", "row", r)]
    dff = sh["mode_blocks.image.0.ff.linear_out.weight"][1]
    blk = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.linear_in", "ff.linear_out"]
    misc = sum(_misc(sh, p, blk)[0] for p in streams.values())
    # parallel attention + MLP: the two row-parallel outputs are summed before the gated residual → one TP all-reduce
    # per block (fused_out) 「推导」 (the reference itself runs on one GPU)
    mode = Layer(tuple(a), replace(core, span="mode"), Ffn("dense", d_ff=dff, gated=True), tuple(f), misc,
                 fused_out=True)
    p = "single_transformer_blocks.0."
    sff = sh[p + "ff.linear_out.weight"][1]
    misc, _ = _misc(sh, p, blk)
    single = Layer((_lin(sh, p + "attn.to_q", "q"), _lin(sh, p + "attn.to_k", "k"), _lin(sh, p + "attn.to_v", "v"),
                    _lin(sh, p + "attn.to_out.0", "o", "row")), core, Ffn("dense", d_ff=sff, gated=True),
                   (_lin(sh, p + "ff.linear_in", "gate_up"), _lin(sh, p + "ff.linear_out", "down", "row")), misc,
                   fused_out=True)
    io_spec = [("x_embedders.image", "patch_embed", "img"), ("x_embedders.image_cond", "ref_embed", "ref"),
               ("context_embedder", "text_in", "txt", "cond"),
               # BFL dit.py: vec[m] = time_in(timestep_embedding(timesteps[m])) — timesteps are per token, every mode
               ("timestep_embedder.linear_1", "time_1", "tok"), ("timestep_embedder.linear_2", "time_2", "tok"),
               ("early_stream_modulations.image.linear", "img_mod", "img", "cond"),
               ("early_stream_modulations.image_cond.linear", "ref_mod", "ref", "cond"),
               ("early_stream_modulations.txt.linear", "txt_mod", "txt", "cond"),
               ("single_stream_modulations.image.linear", "single_img_mod", "img", "cond"),
               ("single_stream_modulations.image_cond.linear", "single_ref_mod", "ref", "cond"),
               ("single_stream_modulations.txt.linear", "single_txt_mod", "txt", "cond")]
    flow_map = "target_timestep_embedder.linear_1.weight" in sh
    if flow_map:      # ref_time_in(timesteps_ref) per sample, then expanded over every token of the sample
        io_spec += [("target_timestep_embedder.linear_1", "target_time_1", "seq"),
                    ("target_timestep_embedder.linear_2", "target_time_2", "seq")]
    io_pre, used = _io(sh, io_spec)
    io_post, used2 = _io(sh, [("norm_out.linear", "norm_out_mod", "img", "cond"), ("proj_out", "head", "img")])
    # every remaining tensor (QK-norm scales of the blocks are in misc; norms / biases outside the blocks) → io_misc;
    # an unmodelled 2-D weight ≥ 256 raises, so coverage "full" means every header GEMM is in the op graph
    blocks = ("mode_blocks.image.", "mode_blocks.image_cond.", "txt_mode_blocks.", "single_transformer_blocks.")
    used_set = {n + ".weight" for n in used + used2}
    io_misc = 0
    for t, v in sh.items():
        if t.startswith(blocks) or t in used_set:
            continue
        if t.endswith(".weight") and len(v) >= 2 and min(v[0], _n(v[1:])) >= 256:
            raise ValueError(f"unmodelled io GEMM weight {t} {v}")
        io_misc += _n(v)
    # the summarised header keeps first / last block of BLOCK_RE stacks (single_transformer_blocks, txt_mode_blocks);
    # mode_blocks.<mode>.<i> does not match BLOCK_RE and is listed in full — all per-mode blocks share block 0's shapes
    for m_ in ("mode_blocks.image.", "mode_blocks.image_cond.", "txt_mode_blocks."):
        ids = {int(t[len(m_):].split(".")[0]) for t in sh if t.startswith(m_)}
        if max(ids) != Ld - 1:
            raise ValueError(f"{m_}: last block {max(ids)} ≠ num_layers {Ld} − 1")
    wl = replace(_image_wl(wl, 2 if sh["x_embedders.image.weight"][1] == 4 * 32 else 1, wl.text_tokens or 320),
                 max_refs=wl.max_refs or 10)
    steps_note = (f"distilled（flow map）：{wl.steps} 步、无 guidance（config sampler.steps 16、guidance null；README）"
                  if flow_map else f"Base（未蒸馏）：{wl.steps} 步、真 CFG（BFL pipeline.py MODELS base guidance 4.0）→ "
                  "每步 [无条件 ‖ 条件] 两个样本打包前向，两样本都带参考图 token")
    spec = _finish(model_id, hf_id, rel, f"FLUX 3 DiT（{Ld} 分模态块：图像 / 参考图 / 文本各自模态内自注意力；{Ls} 单流块："
                   "[图像 ‖ 参考图 ‖ 文本] 联合注意力；并行 attn + SwiGLU；单帧 2D 潜空间）",
                   d, 0, [mode] * Ld + [single] * Ls, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"图像 = 1 个潜帧（VAE ×{wl.vae_s} + 2×2 打包 = 每 token {2 * wl.vae_s}×{2 * wl.vae_s} px；无时间维）",
                       f"文本：Qwen3-VL-4B 第 4/8/…/32 层隐状态拼接（config text_encoder.layers，8 × 2560 = 20480 = "
                       f"context_embedder 输入）；pad_multiple 80 → 按 {wl.text_tokens} token（4 × 80，提示上采样后的完整场景描述）"
                       "计入联合序列「假设」",
                       steps_note,
                       "参考图（编辑 / 多参考，workload.ref_images ≤ 10，「来源」BFL flux3/pipeline.py）：每张 VAE 编码（fp32 AE，"
                       "编码器计入 pipeline）→ (w/16)(h/16) 个 token；1 张按 ≤ 4 MP 画布（16 px 网格，边 64–4096），≥ 2 张每张 "
                       "≤ 1 MP；尺寸默认 = 输出尺寸（发布默认输出尺寸取第一张参考图）「假设」，可设 ref_width / ref_height",
                       "参考图 token 每步重算（「来源」sampling.py 每步 dit(packed)；diffusers Flux3 无 KV 缓存参数）；时间步 0 调制；"
                       "RoPE (t, h, w, l)：h / w × image_rope_scale 0.5，l = −(1 + 序号) 区分参考图——只改位置编码，不改算量",
                       "调制按 token 计（BFL 参考实现：time_in 与两张 ModeModulation 表逐 token；diffusers 时间 MLP 按样本、"
                       "调制逐 token）；空模态补 1 个占位 token（文生图时参考图流跑 1 行，可忽略，未计）"))
    return spec



# ---------------------------------------------------------------- 0.257 MOSS-Image (OpenMOSS, single-stream joint DiT)
def build_moss(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """MOSS-Image (MOSSImageTransformer, the release's pipeline_moss_image.py, 「来源」):
      video_embedder Linear(128 → d) on the 2×2-packed FLUX.2 latents (×8 VAE → 128 channels per token);
      text_embedder = Linear(4096 → d) on the Qwen3.5-9B last hidden state + 2 text-only refiner blocks (fused qkv +
      proj with biases, RMS QK-norm, soft-capped gated MLP ``_situ``) — run every forward, on the text rows only;
      time_embedding MLP (freq 256 → d → d) → ONE shared_block_modulation (d → 6d: shift / scale / gate for attention
      and MLP) used by every block, and video_head_modulation (d → 2d) for the head;
      num_layers single-stream blocks over [image ‖ text] (bias-free q / k / v / o, RMS QK-norm, 4-axis RoPE over the
      whole head, ffn_up_gate d → 2·ffn, ffn_down) with the shared AdaLN; final RMSNorm + head on the image rows only.
    Each sample of a batch runs at its own text length (loop over the batch); CFG = two transformer calls per step
    (cond, then uncond with block 22 skipped — the release default ``uncond_skip_layers=(22,)``)."""
    c, sh = rel["config"], rel["shapes"]
    d, H, L, f = c["dim"], c["num_heads"], c["num_layers"], c["ffn_dim"]
    hd = d // H
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    p = "layers.0."
    a = [("proj.q", "q", "col"), ("proj.k", "k", "col"), ("proj.v", "v", "col"), ("proj.o", "o", "row")]
    fl = [("proj.ffn_up_gate", "gate_up", "col"), ("proj.ffn_down", "down", "row")]
    if sh[p + "proj.ffn_up_gate.weight"][0] != 2 * f or sh[p + "proj.ffn_down.weight"][1] != f:
        raise ValueError("MOSS-Image: ffn_up_gate / ffn_down do not match ffn_dim")
    misc, _ = _misc(sh, p, [n for n, *_ in a + fl])
    layer = Layer(tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in a), core, Ffn("dense", d_ff=f, gated=True),
                  tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in fl), misc)
    blocks = sorted({int(t.split(".")[1]) for t in sh if t.startswith("layers.")})
    if blocks[-1] != L - 1:
        raise ValueError(f"layers: last block {blocks[-1]} ≠ num_layers {L} − 1")
    refs = sorted({int(t.split(".")[2]) for t in sh if t.startswith("text_embedder.blocks.")})
    specs = [("video_embedder", "patch_embed", "img"), ("text_embedder.proj", "text_in", "txt", "cond")]
    for i in refs:
        r = f"text_embedder.blocks.{i}."
        specs += [(r + n, f"refiner{i}.{n}", "txt", "cond") for n in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")]
    specs += [("time_embedding.0", "time_1", "seq"), ("time_embedding.2", "time_2", "seq"),
              ("shared_block_modulation", "block_mod", "seq", "cond")]
    io_pre, used = _io(sh, specs)
    io_post, used2 = _io(sh, [("video_head_modulation", "norm_out_mod", "seq", "cond"),
                              ("final_linear_video", "head", "img")])
    # every non-block tensor that is not an io GEMM weight (refiner biases / norms, embedder biases, the final norm,
    # the fp32 RoPE band buffer) → io_misc; an unmodelled 2-D weight ≥ 256 raises (BLOCK_RE would also skip the
    # refiner's text_embedder.blocks.N tensors, so the walk is explicit here, as in build_flux3)
    used_set = {n + ".weight" for n in used + used2}
    io_misc = 0
    for t, v in sh.items():
        if t.startswith("layers.") or t in used_set:
            continue
        if t.endswith(".weight") and len(v) >= 2 and min(v[0], _n(v[1:])) >= 256:
            raise ValueError(f"unmodelled io GEMM weight {t} {v}")
        io_misc += _n(v)
    # the header summary keeps blocks 0 and 1 of the refiner (both) — 2 refiner blocks of block 0's shapes
    wl = _image_wl(wl, 2 if sh["video_embedder.weight"][1] == 4 * 32 else 1, wl.text_tokens or 256)
    # 0.258: the CFG negative branch as the release runs it (pipeline __call__ defaults, 「来源」): negative_prompt '' →
    # chat template '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n' = 10 tokens (4 special tokens +
    # user / assistant / 4 × '\n'; the Qwen pre-tokenizer splits letters from newlines, 「推导」), unpadded; and
    # uncond_skip_layers = (22,) with slg_start 0 / slg_stop 1 → block 22 skipped at every step
    if wl.cfg == 2 and not wl.uncond:
        wl = replace(wl, uncond=(_MOSS_NEG_TOKENS, (22,) if L > 22 else ()))
    spec = _finish(model_id, hf_id, rel, f"MOSS-Image 单流 DiT（{L} 层，[图像 ‖ 文本] 共享权重联合注意力，共享 AdaLN 调制；"
                   "单帧 2D 潜空间）", d, 0, [layer] * L, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"图像 = 1 个潜帧（FLUX.2 VAE ×{wl.vae_s} + 2×2 打包 = 每 token {2 * wl.vae_s}×{2 * wl.vae_s} px；无时间维）",
                       f"文本：Qwen3.5-9B 末层隐状态（聊天模板 + enable_thinking，逐样本真实长度、不补齐，上限 4096）"
                       f"按 {wl.text_tokens} token 计入联合序列「假设」；2 层文本 refiner 每次前向都跑（参考实现不缓存），"
                       "GEMM 与文本自注意力核（24 头 × 128，双向，QK-norm + RoPE）都计入",
                       "50 步 flow-matching Euler（shift 5，吸附到 1000 步训练网格后仍为 50 个时间步「推导」）、CFG 3 → 每步 cond + "
                       "uncond 两次 transformer 调用（README 默认），先后串行（两次调用，各读一遍权重）",
                       f"uncond 分支按发布默认：负提示为空串 → 聊天模板 {_MOSS_NEG_TOKENS} token「推导」、跳过第 22 块（41 块）；"
                       "两分支各按自己的形状计（workload.cfg_branches = same 回到同形状、一次打包前向的旧口径）",
                       "参考实现逐样本循环（batch 内每张图单独前向）：batch > 1 时按打包批计、权重读一次「近似」"
                       "（1024² 时 4096 + 文本行，计算受限，影响可忽略）",
                       "MLP 激活 _situ = 4·tanh(g/4)·σ(g) · 25·tanh(h/25)（软截断门控）：向量开销按 SwiGLU 计「近似」"))
    return replace(spec, io_attn=tuple((f"refiner{i}", "txt", H, hd, 1) for i in refs))


_MOSS_NEG_TOKENS = 10      # chat-templated empty negative prompt (see build_moss)


def _require_esm_encoder(c: dict) -> None:
    """Published ESM-2 encoder (transformers modeling_esm.py). The class hardcodes erf-GELU
    and nn.LayerNorm. A checkpoint whose config does not match that forward is refused."""
    if c.get("model_type") != "esm":
        raise ValueError("esm model_type is not esm")
    if c.get("hidden_act") != "gelu":
        raise ValueError("esm hidden_act is not gelu")
    if c.get("position_embedding_type") != "rotary":
        raise ValueError("esm position_embedding_type is not rotary")
    if c.get("emb_layer_norm_before") is not False:
        raise ValueError("esm emb_layer_norm_before is not false")
    if c.get("attention_probs_dropout_prob") not in (0, 0.0) or c.get("hidden_dropout_prob") not in (0, 0.0):
        raise ValueError("esm dropout is not 0")
    if c.get("is_folding_model"):
        raise ValueError("esm folding model is not the encoder")


# ---------------------------------------------------------------- ESM-2
def build_esm(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    c, sh = rel["config"], rel["shapes"]
    _require_esm_encoder(c)
    d, H, f, L, V = c["hidden_size"], c["num_attention_heads"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    hd = d // H
    alin, core = _mha(d, H, rope=hd)
    core = replace(core, qk_norm=False, q_scale=True)
    ffn, flin = _gelu_ffn(d, f)
    misc = 4 * d + 2 * d + f + d + 2 * d                    # biases + attention LN + output LN
    layer = Layer(alin, core, ffn, flin, misc, vec_kind="esm2")
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
    notes.append("0.437：块内两处 nn.LayerNorm 与编码器末层 emb_layer_norm_after 按 fp32 归约计"
                 "（vec = token×hidden×4，每行一次 rsqrt）；MLP 与 MLM 头的 gelu 是 "
                 "x·0.5·(1+erf(x/√2))（vec = 元素×5，一次 erf）。查询在 RoPE 前乘 head_dim**-0.5。"
                 "没有 QK-norm。token_dropout 的缩放取决于样本里的 mask 数，工作负载没有 mask 率，不单列")
    return replace(spec, notes=tuple(notes), vec_kind="esm2")


def _require_generanno(c: dict, sh: dict) -> None:
    """Published Carbon-A encoder (modeling_generanno.py). RMSNorm is T5LayerNorm.
    The MLP is down(silu(gate) * up). A checkpoint that does not match is refused."""
    if c.get("model_type") != "generanno":
        raise ValueError("generanno model_type is not generanno")
    if c.get("hidden_act") != "silu":
        raise ValueError("generanno hidden_act is not silu")
    if c.get("attention_bias") is not False:
        raise ValueError("generanno attention_bias is not false")
    if c.get("mlp_bias") is not False:
        raise ValueError("generanno mlp_bias is not false")
    if c.get("attention_dropout") not in (0, 0.0):
        raise ValueError("generanno attention_dropout is not 0")
    if c.get("rope_scaling") is not None:
        raise ValueError("generanno rope_scaling is not null")
    if int(c.get("pretraining_tp") or 1) != 1:
        raise ValueError("generanno pretraining_tp is not 1")
    if c.get("use_mlp_classifier"):
        raise ValueError("generanno use_mlp_classifier is not false")
    if int(c.get("num_prediction_heads") or 0) != 2:
        raise ValueError("generanno num_prediction_heads is not 2")
    d = int(c["hidden_size"])
    H = int(c["num_attention_heads"])
    KV = int(c["num_key_value_heads"])
    f = int(c["intermediate_size"])
    L = int(c["num_hidden_layers"])
    if H <= 0 or KV <= 0 or d % H or H % KV:
        raise ValueError("generanno head counts do not divide")
    hd = c.get("head_dim")
    if hd is not None and int(hd) != d // H:
        raise ValueError("generanno head_dim is not hidden/heads")
    kmer = int(c.get("k") or 0)
    labels = c.get("id2label") or {}
    if kmer != 6 or len(labels) != 2:
        raise ValueError("generanno score width is not k*2")

    def _need(name: str, shape: list) -> None:
        got = sh.get(name)
        if list(got or []) != shape:
            raise ValueError(f"generanno {name} shape is not {shape}")

    for li in (0, L - 1):
        p = f"model.layers.{li}."
        _need(p + "input_layernorm.weight", [d])
        _need(p + "post_attention_layernorm.weight", [d])
        _need(p + "mlp.gate_proj.weight", [f, d])
        _need(p + "mlp.up_proj.weight", [f, d])
        _need(p + "mlp.down_proj.weight", [d, f])
    _need("model.norm.weight", [d])
    _need("score.0.weight", [12, d])
    _need("score.1.weight", [12, d])
    if any("q_norm" in n or "k_norm" in n for n in sh):
        raise ValueError("generanno has q_norm")
    if any(n.startswith("score.") and n.endswith(".bias") for n in sh):
        raise ValueError("generanno score bias is present")


def build_generanno(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """0.106 Carbon-A (Generanno): Llama-style blocks (RMSNorm, GQA + RoPE, SwiGLU), bidirectional, no KV cache;
    two per-token strand heads score.{0,1} (hidden → k·2: per base of the 6-mer token, CDS / non-coding).
    0.438 bills the published RMSNorm and SiLU from modeling_generanno.py."""
    from .model import _ffn_dense, _gqa
    c, sh = rel["config"], rel["shapes"]
    _require_generanno(c, sh)
    d, H, KV, f, L, V = (c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"],
                         c["num_hidden_layers"], c["vocab_size"])
    hd = c.get("head_dim") or d // H
    alin, core = _gqa(d, H, KV, hd)
    core = replace(core, rope_dim=hd, causal=False, qk_norm=False)
    ffn, flin = _ffn_dense(d, f, gated=True)
    layer = Layer(alin, core, ffn, flin, 2 * d, vec_kind="generanno")  # input / post-attention RMSNorm
    heads = sorted(n for n in sh if n.startswith("score.") and n.endswith(".weight"))
    io_post = tuple(Linear(n.rsplit(".", 1)[0].replace(".", "_"), d, sh[n][0], "rep") for n in heads)
    wl = replace(wl, max_seq=c.get("max_position_embeddings", wl.seq_len), kmer=c.get("k", wl.kmer))
    spec = _finish(model_id, hf_id, rel, "Carbon-A 编码器（Generanno：双向 GQA + RoPE + SwiGLU，逐 token 链头）", d, L,
                   layer, (), io_post, 0, V, wl, "")
    # the card's inference setting (BF16 + FlashAttention-2, every reported result) — the safetensors are fp32
    bf = RoleFormat("bf16", 16.0)
    fm = tuple((k, bf) for k, _ in spec.formats)
    k_ = wl.kmer or 6
    notes = [n for n in spec.notes if "fp32" not in n] + [
        "发布 safetensors 为 fp32；按模型卡的推理设置（BF16 + FlashAttention-2，全部报告结果）计权重与激活",
        f"无词表 MLM 头：两个逐 token 链头（正 / 反链）{' + '.join(f'{d}→{l.n}' for l in io_post)}"
        f"（每 token {k_} 个碱基 × 2 类：CDS / 非编码）",
        f"窗口 {wl.seq_len} token（{wl.seq_len * k_:,} bp），相邻窗口重叠 {wl.overlap} token；每窗口新增 "
        f"{(wl.seq_len - wl.overlap) * k_:,} bp（注释流水线对重叠部分取平均）",
        "0.438：input_layernorm、post_attention_layernorm 和 model.norm 是 GenerannoRMSNorm"
        "（fp32 pow(2).mean，每行一次 rsqrt，再乘 weight）。MLP 是 down(silu(gate) * up)。"
        "没有 QK-norm。RoPE 打在 repeat_kv 之前的 q 和 k 上。cos/sin 的 fp32 表不单列"]
    reasons = [r for r in spec.coverage_reasons]
    return replace(spec, formats=fm, act_fmt="bf16", kv_fmt="bf16", final_norm_params=d, vocab_head=False,
                   notes=tuple(notes), coverage_reasons=tuple(reasons), coverage="partial" if reasons else "full",
                   vec_kind="generanno")


# ---------------------------------------------------------------- 0.330 PixText (pixel-space text-editing DiT)
def build_pixtext(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """PixText (ATH-MaaS, GroupJiT, github.com/ATH-MaaS/PixText src/pixtext/modeling/transformer.py, 「来源」):
    pixel space — no VAE, no text encoder.  One packed sequence per pass = the main RGB image (16×16×3 pixel patches,
    patch_projection 768 → d) ‖ for each edit item its RGB crop tokens ‖ the same number of glyph tokens (rendered
    16×16 glyph patches, glyph_projection 256 → d).  ``depth`` PackedJiT blocks: RMSNorm, fused qkv (+bias), RMS
    QK-norm, 3-axis RoPE over the whole head (axes_dims sum = head dim), proj; SwiGLU w12 (d → 2·⌊4d·2/3⌋) / w3;
    AdaLN ``adaLN_modulation`` (d → 6d) evaluated **per token** (conditioning_ids = arange(tokens): every token
    carries its own timestep embedding), as are timestep_embedder and — MeanFlow — interval_embedder, which the
    reference calls twice per token (f(r) − f(0)).  Final layer on the RGB tokens (per-image AdaLN d → 2d, linear d → 768).
    Workload mapping: main tokens = the image rows (frames 1, ``vae_s`` 1, patch 16); items + glyphs = the joint
    prefix rows (``text_tokens``)."""
    c, sh = rel["config"]["model"], rel["shapes"]
    d, H, L, P = c["hidden_size"], c["num_heads"], c["depth"], c["patch_size"]
    hd = d // H
    if sum(c["axes_dims"]) != hd:
        raise ValueError(f"PixText: RoPE axes {c['axes_dims']} do not cover head dim {hd}")
    f = int(int(d * c["mlp_ratio"]) * 2 / 3)          # SwiGLUFFN: hidden = int(mlp_hidden · 2 / 3)
    p = "blocks.0."
    if sh[p + "mlp.w12.weight"][0] != 2 * f or sh[p + "mlp.w3.weight"][1] != f:
        raise ValueError("PixText: w12 / w3 do not match int(4d·2/3)")
    if sh[p + "attn.qkv.weight"][0] != 3 * d or sh[p + "adaLN_modulation.1.weight"][0] != 6 * d:
        raise ValueError("PixText: qkv / adaLN shapes")
    blocks = sorted({int(t.split(".")[1]) for t in sh if t.startswith("blocks.")})
    if blocks[-1] != L - 1:
        raise ValueError(f"blocks: last block {blocks[-1]} ≠ depth {L} − 1")
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    a = [("attn.qkv", "qkv", "col"), ("attn.proj", "o", "row"), ("adaLN_modulation.1", "ada_mod", "col")]
    fl = [("mlp.w12", "gate_up", "col"), ("mlp.w3", "down", "row")]
    misc, _ = _misc(sh, p, [n for n, *_ in a + fl])
    layer = Layer(tuple(_lin(sh, p + n, s_, sp, role=("cond" if s_ == "ada_mod" else "")) for n, s_, sp in a), core,
                  Ffn("dense", d_ff=f, gated=True), tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in fl), misc)
    specs = [("patch_projection.proj", "patch_embed", "img"), ("glyph_projection.proj", "glyph_embed", "txt"),
             ("timestep_embedder.mlp.0", "time_1", "tok", "cond"), ("timestep_embedder.mlp.2", "time_2", "tok", "cond")]
    iv = "interval_embedder.mlp.0.weight" in sh
    if iv:
        specs += [("interval_embedder.mlp.0", "interval_1", "tok", "cond"), ("interval_embedder.mlp.2", "interval_2", "tok", "cond")]
    io_pre, used = _io(sh, specs)
    dup = ()
    if iv:   # IntervalEmbedder.forward = f(r) − f(0): the second call reuses the same weights (compute, no storage)
        dup = (_lin(sh, "interval_embedder.mlp.0", "interval_1_zero", rows="tok", role="cond"),
               _lin(sh, "interval_embedder.mlp.2", "interval_2_zero", rows="tok", role="cond"))
    io_post, used2 = _io(sh, [("final_layer.adaLN_modulation.1", "norm_out_mod", "seq", "cond"),
                              ("final_layer.linear", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2) - sum(x.params for x in dup)
    wl = _image_wl(wl, P, wl.text_tokens)
    n_crop = wl.text_tokens // 2
    # 0.331: the guidance branch is a second, serial model call without glyphs (sampling/flow.py _guided_prediction,
    # meanflow.py _guided_interval_prediction: glyph_conditions → None) → its sequence = main + item crops: the joint
    # prefix minus the glyph tokens (one glyph token per item crop token, 「来源」 branch tokenizer); no skipped blocks
    if wl.cfg == 2 and not wl.uncond:
        wl = replace(wl, uncond=(wl.text_tokens - n_crop, ()))
    g = guided_steps(wl.steps, wl.cfg_interval) if wl.cfg == 2 else 0
    solver = "meanflow_interval_euler，c = k/{0}".format(wl.steps) if iv else "Euler，t = k/{0}".format(wl.steps)
    fwd = (f"每请求 {wl.steps} 步（{solver}）+ guidance_scale 3.3 只在 guidance_interval "
           f"({wl.cfg_interval[0]}, {wl.cfg_interval[1]}) 内（严格不等，k/步数）的 {g} 步各多一次无字形分支前向 = "
           f"{wl.steps + g} 次 DiT 前向（参考实现两次串行调用「来源」"
           + ("；README：four MeanFlow updates use seven model forwards" if iv else "") + "）；无字形分支按自身形状"
           f"（{(wl.height // P) * (wl.width // P) + wl.text_tokens - n_crop} token：主图 + 编辑项，无字形）计（0.331，"
           "原按条件分支形状）") if wl.cfg_interval else ""
    return _finish(model_id, hf_id, rel, f"PixText GroupJiT 像素空间 DiT（{L} 层单流，[主图 ‖ 编辑项裁块 ‖ 字形] 联合注意力；"
                   f"16×16 像素 patch，无 VAE）", d, 0, [layer] * L, io_pre + dup, io_post, io_misc, 0, wl, "",
                   skip=("models",), standalone=(
                       "发布即完整推理：无文本编码器、无 OCR、无 VAE（README：MeanFlow needs no additional text encoder, OCR "
                       "model, VAE；Teacher 同一 GroupJiT 结构）——字形由字体光栅化（CPU，未计时）、输出直接是像素 patch"),
                   fwd_note=fwd, extra_notes=(
                       f"每次 pass = 主图（main_target_area 1048576 → 1024×1024 = {(wl.height // P) * (wl.width // P)} token）"
                       f" + 编辑项 RGB 裁块 {n_crop} token + 等量字形 token {n_crop}（发布参考用例 examples/pipeline_case 第 1 pass："
                       f"12 项，11 个 64×64 + 1 个 64×80 裁块 → Σ{n_crop}「来源」）；max_items_per_pass 12，"
                       f"更多项分多次 pass 串行（参考用例 13 项 = 2 pass，第 2 pass 仅 1 项 12 token）",
                       "注意力掩码（items_interread_glyph_reads_own_item_bidirectional_main）：主图看主图 + 全部编辑项、编辑项看主图 / 全部编辑项 / "
                       "自己的字形、字形只看自己的字形与所属编辑项（attention.py _allowed_routing_pair「来源」）——条件分支按稠密联合注意力计「近似」"
                       "（参考用例第 1 pass 允许 18,430,912 对 vs 稠密 4488² = 20,142,144，多算 ≈ 9.3%「推导」；未建模原因：字形只看"
                       "所属编辑项，代价取决于各项尺寸 Σnᵢ²，工作负载只有总 token 数）；无字形分支（主图 + 编辑项）在该掩码下本就稠密，按实计",
                       "AdaLN（d → 6d）、timestep / interval embedder 按参考实现逐 token 计算（conditioning_ids = arange(tokens)）；"
                       "patch_projection 按主图行、glyph_projection 按前缀行计（编辑项 RGB 行实为 patch_projection，"
                       "≈ 0.005% FLOP「近似」）"))




# ---------------------------------------------------------------- common tail
def _finish(model_id, hf_id, rel, arch, d, L, layer, io_pre, io_post, io_misc, vocab, wl, _, skip=(), extra_notes=(),
            ext_parts=(), standalone="", fwd_note=""):
    """``standalone`` (0.330): the release has no text encoder / VAE at all (PixText) — that sentence replaces the
    "DiT trunk only" coverage reason when no unmodelled component is found; ``fwd_note`` replaces the generic
    forwards-per-step note (PixText: guidance on a sub-interval of the steps, counted as forwards)."""
    fm = _formats(rel)
    fmd = dict(fm)
    dom = wl.kind if wl.kind in ("gen", "genomics") else "protein"
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
        if pl is None and standalone and not parts:
            notes.append(standalone)
        elif pl is None:
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
        if fwd_note:
            notes.append(fwd_note)
        elif wl.cfg > 1:
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



# ---------------------------------------------------------------- 0.144 LingBot-Video (MoE DiT)
def build_lingbot(model_id: str, hf_id: str, rel: dict, wl: NativeWorkload) -> ModelSpec:
    """LingBot-Video MoE (LingBotVideoTransformer3DModel, Robbyant/lingbot-video transformer_lingbot_video.py):
    single-stream blocks over the joint [video ‖ text] sequence (text_embedder projects the Qwen3-VL hidden states to
    the model width; no cross-attention), MHA with RMS QK-norm and 3-axis RoPE; AdaLN from a shared per-token time
    modulation (time_modulation 6·d, computed for every joint token in the reference) + a per-block scale_shift_table;
    sandwich RMSNorms on the attention / FFN outputs; FFN = sigmoid grouped-top-k MoE (num_experts, top-k,
    n_group / topk_group) + shared SwiGLU experts (mlp_only_layers stay dense)."""
    c, sh = rel["config"], rel["shapes"]
    d, H, L = c["hidden_size"], c["num_attention_heads"], c["depth"]
    hd = d // H
    E, k, de, nsh = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], c.get("n_shared_experts") or 0
    core = AttnCore("gqa", n_q=H, n_kv=H, qk_dim=hd, v_dim=hd, rope_dim=hd, causal=False)
    mlp_only = set(c.get("mlp_only_layers") or [])
    step = c.get("decoder_sparse_step") or 1
    p = "blocks.0."
    a = [("attn.to_q", "q", "col"), ("attn.to_k", "k", "col"), ("attn.to_v", "v", "col"), ("attn.to_out", "o", "row")]
    alin = tuple(_lin(sh, p + n, s_, sp) for n, s_, sp in a)
    flin = (Linear("router", d, E, "rep"), Linear("shared_gate_up", d, 2 * nsh * de, "col"),
            Linear("shared_down", nsh * de, d, "row"))
    misc, _ = _misc(sh, p, [n for n, *_ in a] + ["ffn.router", "ffn.shared_experts.gate_proj",
                                                  "ffn.shared_experts.up_proj", "ffn.shared_experts.down_proj"])
    misc -= 3 * E * de * d          # routed expert tensors (ffn.experts.w1/w2/w3, no .weight suffix) are the Ffn experts
    moe = Layer(alin, core, Ffn("moe", gated=True, n_experts=E, top_k=k, d_expert=de, n_shared=nsh, d_shared=de),
                flin, misc, post_norm=True)
    layers = []
    for i in range(L):
        if i in mlp_only or (i + 1) % step:
            raise ValueError("LingBot-Video dense blocks not in this release")   # every released block is MoE
        layers.append(moe)
    io_pre, used = _io(sh, [("patch_embedder", "patch_embed", "img"), ("text_embedder.linear_1", "text_in_1", "txt", "cond"),
                            ("text_embedder.linear_2", "text_in_2", "txt", "cond"),
                            ("time_embedder.linear_1", "time_1", "seq"), ("time_embedder.linear_2", "time_2", "seq"),
                            ("time_modulation.1", "time_mod", "tok", "cond")])
    io_post, used2 = _io(sh, [("norm_out_modulation.1", "norm_out_mod", "tok", "cond"), ("proj_out", "head", "img")])
    io_misc = _io_misc(sh, io_pre + io_post, used + used2)
    wl = replace(wl, patch=tuple(c["patch_size"]), prefix_tokens=wl.text_tokens)
    return _finish(model_id, hf_id, rel, f"LingBot-Video MoE DiT（{L} 层单流，[视频 ‖ 文本] 联合注意力；"
                   f"MoE {E} 专家 top-{k} + {nsh} 共享）", d, 0, layers, io_pre, io_post, io_misc, 0, wl, "", extra_notes=(
                       f"每块 FFN = sigmoid 分组路由（{c.get('n_group')} 组选 {c.get('topk_group')}）top-{k} MoE：路由专家按本卡"
                       "token 的分组 GEMM（参考实现无 EP：Ulysses SP 切 token、每卡持有全部专家），均匀路由期望命中专家数「假设」",
                       "时间调制 time_modulation / norm_out_modulation 按参考实现对联合序列每个 token 计算（广播后的同一 t_emb）",
                       f"文本（Qwen3-VL 隐状态，padding=longest，上限 TOKEN_LENGTH 37698）按 {wl.text_tokens} token 计入联合序列「假设」",
                       "DMD 蒸馏学生：8 步（dmd_student），guidance 1.0（无 CFG），README 默认 832×480、121 帧、24 fps"))

BUILDERS = {"wan": build_wan, "wan22": build_wan22, "cogvideox": build_cogvideox, "hunyuan": build_hunyuan,
            "ltx": build_ltx, "mochi": build_mochi, "stdit": build_stdit, "h3": build_h3, "esm": build_esm,
            "generanno": build_generanno, "qwenimage": build_qwenimage, "qwenimage21": build_qwenimage21,
            "flux2": build_flux2, "lingbot": build_lingbot, "flux3": build_flux3, "moss": build_moss,
            "pixtext": build_pixtext}
from .structure import BUILDERS as _STRUCTURE_BUILDERS  # noqa: E402  (0.43 protein structure models)
BUILDERS.update(_STRUCTURE_BUILDERS)


def from_domain_release(model_id: str, hf_id: str, builder: str, wl: NativeWorkload, release: str | None = None) -> ModelSpec:
    """``release`` (0.331): the release-file key when it differs from the public repo id (catalog.RELEASE_KEYS)."""
    rel = load_release(release or hf_id)
    if rel is None:
        raise ValueError(f"no release metadata for {release or hf_id}; run scripts/fetch_hf_release.py")
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
    ref: int = 0             # 0.252 reference-image rows in the sequence (FLUX 3 image_cond tokens)
    guided: int = -1         # 0.331 steps that run the CFG negative branch (cfg_interval); −1 = every step


def guided_steps(steps: int, interval: tuple) -> int:
    """0.331: steps k = 0 … steps − 1 whose progress k / steps lies strictly inside the guidance interval (lo, hi)
    (PixText GroupFlowSampler / GroupMeanFlowSampler: ``low < time_value < high`` with time_value = k / steps, 「来源」);
    an empty interval guides every step."""
    if not interval:
        return steps
    lo, hi = interval
    return sum(1 for k in range(steps) if lo < k / steps < hi)


def _latent_frames(d: NativeWorkload, F: int, warns: list[str]) -> int:
    if d.vae_frames == "chunk17":           # Open-Sora 1.2 VAE: micro-batches of 17 frames → 5 latent frames each
        # 0.118: VideoAutoencoderPipeline.get_latent_size — full 17-frame chunks give 5 latents each; the remainder r
        # frames is front-padded to a multiple of t (VAE_Temporal) → ⌈r / t⌉ latents (0.44–0.117 padded it to 17)
        r = F % 17
        lt = F // 17 * ((17 - 1) // d.vae_t + 1) + math.ceil(r / d.vae_t)
        if r:
            warns.append(f"帧数 {F} 不是 17 的整数倍：末段 {r} 帧 → {math.ceil(r / d.vae_t)} 潜帧")
        return lt
    if d.vae_frames == "h3":                # MiniMax-H3: frames snapped up to 17n+5 → 5n+2 latent frames
        Fa = F
        while Fa % 17 != 5:
            Fa += 1
        if Fa != F:
            warns.append(f"帧数 {F} 按 VAE 规则补齐到 17n+5 = {Fa}")
        return (Fa - 5) // 17 * 5 + 2
    if d.vae_frames == "wan" and F % d.vae_t != 1:
        # 0.117: diffusers WanPipeline snaps num_frames → F // t · t + 1 (warning "Rounding to the nearest number"); F ≡ 0
        # (mod t) therefore rounds *up* one latent frame vs (F − 1) // t + 1
        Fa = F // d.vae_t * d.vae_t + 1
        warns.append(f"帧数 {F} 不满足 {d.vae_t}k+1：参考 pipeline 取 {Fa} 帧")
        return (Fa - 1) // d.vae_t + 1
    lt = (F - 1) // d.vae_t + 1
    if (F - 1) % d.vae_t:
        warns.append(f"帧数 {F} 不满足 {d.vae_t}k+1：潜空间按 {lt} 帧计")
    return lt


_REF_MATCH_PX = 4 * 1024 * 1024      # flux3/pipeline.py MATCH_MAX_PIXELS (first / single reference, padded canvas)
_REF_MULTI_PX = 1024 * 1024          # MULTI_REFERENCE_MAX_PIXELS (each reference of a multi-reference edit)


def flux3_ref_size(n: int, w: int, h: int, px: int = 16) -> tuple[int, int]:
    """0.252: the size (w, h) px one reference reaches the DiT at, by the BFL release rules (flux3/pipeline.py, 「来源」):
      n = 1  reference_canvas: scale = min(1, √(4 MiB / wh), 4096 / max side); fitted = round(side · scale); canvas =
             fitted rounded up to 16 px, sides clamped to [64, 4096], area ≤ 4 MiB (else scale ·= 0.995) — the padded
             canvas is encoded
      n ≥ 2  prepare_multi_reference: wh > 1 MiB → sides int(side · √(1 MiB / wh)); centre-crop to the 16-px grid"""
    if n == 1:
        scale = min(1.0, math.sqrt(_REF_MATCH_PX / (w * h)), 4096 / max(w, h))
        while True:
            fit = (max(1, round(w * scale)), max(1, round(h * scale)))
            cv = tuple(min(4096, max(64, -(-v // px) * px)) for v in fit)
            if cv[0] * cv[1] <= _REF_MATCH_PX:
                return cv
            scale *= 0.995
    if w * h > _REF_MULTI_PX:
        s_ = math.sqrt(_REF_MULTI_PX / (w * h))
        w, h = int(w * s_), int(h * s_)
    return w // px * px, h // px * px


def tri_reference_chunk(model_id: str, n: int) -> int:
    """0.119: triangle-attention chunking of the release inference code at N tokens (−1 = whole logits materialised).
      esmfold     esm / HF EsmFoldingTrunk chunk_size None (set_chunk_size not called) → materialised
      openfold    config.py globals chunk_size 4 (inference)
      alphafold2  global_config.subbatch_size 4
      boltz-1     trunk.py: N > const.chunk_size_threshold (384) → 128, else 512
      protenix    configs_base infer_setting chunk_size_thresholds: ≤ 1024 none, ≤ 1536 512, ≤ 2048 256, ≤ 2560 128;
                  longer: 128 「假设」"""
    if model_id == "esmfold":
        return -1
    if model_id in ("openfold", "alphafold2"):
        return 4
    if model_id == "boltz-1":
        return 128 if n > 384 else 512
    if model_id == "protenix":
        return -1 if n <= 1024 else 512 if n <= 1536 else 256 if n <= 2048 else 128
    return -1


def resolve_workload(spec: ModelSpec, w) -> ResolvedWorkload:
    """Model defaults overridden by the scenario's ``workload`` block (0 = model default)."""
    d = spec.workload
    warns: list[str] = []
    if not spec.is_pair and (getattr(w, "tri_attn", "fused") != "fused" or getattr(w, "tri_chunk", 0)):
        warns.append("workload.tri_attn / tri_chunk 只用于带三角注意力的结构模型，已忽略")
    if spec.domain == "gen":
        F, Hh, Ww = w.frames or d.frames, w.height or d.height, w.width or d.width
        image = d.frames == 1 and not d.fps         # 0.115 image DiT: one latent frame, no temporal axis
        if image and F != 1:
            warns.append(f"图像模型只生成单帧：帧数 {F} 忽略，按 1 帧计")
            F = 1
        steps, cfg = w.steps or d.steps, w.cfg or d.cfg
        pt, ph, pw = d.patch
        lt = _latent_frames(d, F, warns)
        if d.vae_frames in ("causal", "wan") and not image:
            F = (lt - 1) * d.vae_t + 1                  # 0.117: the causal decoder emits (lt − 1)·t + 1 frames
        # 0.117: every gen pipeline floors H / W to a multiple of vae_s · patch (image: QwenImage / QwenImage21 / Flux2,
        # 0.116; video: diffusers WanPipeline rounds down with a warning, HunyuanVideo / LTX / Mochi / CogVideoX reject
        # other sizes or floor at latent // and the stride-p patch conv).  0.44–0.116 took a ceil patch grid for video.
        # image pipelines pack 2×2 latents even at DiT patch 1 (QwenImage21: latent = 2·⌊px / (2·vae_s)⌋)
        q_h, q_w = (d.vae_s * max(ph, 2), d.vae_s * max(pw, 2)) if image else (d.vae_s * ph, d.vae_s * pw)
        Hf, Wf = Hh // q_h * q_h, Ww // q_w * q_w
        if (Hf, Wf) != (Hh, Ww):
            warns.append(f"分辨率 {Ww}×{Hh} 不是 {q_w}×{q_h} 的整数倍：参考 pipeline 向下取整为 {Wf}×{Hf}")
        if not (Hf and Wf):
            raise ValueError(f"frame size below one latent patch ({q_w}×{q_h} px)")
        Hh, Ww = Hf, Wf
        lh, lw = Hh // d.vae_s, Ww // d.vae_s
        gt, gh, gw = math.ceil(lt / pt), lh // ph, lw // pw
        vid = gt * gh * gw
        if (F, Hh, Ww) != (d.frames, d.height, d.width):
            warns.append(f"工作负载偏离发布默认（{d.width}×{d.height}，{d.frames} 帧）")
        aud = round(F / d.fps * d.audio_per_s) * d.audio_channels if d.audio_per_s and d.fps else 0
        # 0.252 reference images (FLUX 3 edit / multi-reference): VAE-encoded, 2×2 packed latents → (w / 16)(h / 16) rows
        # each, joined to the sequence (own per-mode blocks, then the joint single blocks); size = workload ref_width /
        # ref_height (0 → the output size — the release's default output size is the first reference's) fitted by
        # flux3_ref_size
        nref = getattr(w, "ref_images", 0) or 0
        ref_rows, ref_px = 0, []
        if nref and not d.max_refs:
            warns.append("workload.ref_images 已忽略：该模型没有参考图输入流")
            nref = 0
        if nref > d.max_refs > 0:
            raise ValueError(f"workload.ref_images {nref} 超过发布上限 {d.max_refs} 张（MAX_REFERENCES）")
        if (getattr(w, "ref_width", 0) or getattr(w, "ref_height", 0)) and not nref:
            warns.append("workload.ref_width / ref_height 已忽略：没有参考图（ref_images = 0）")
        if nref:
            rw, rh = (getattr(w, "ref_width", 0) or Ww), (getattr(w, "ref_height", 0) or Hh)
            fw, fh = flux3_ref_size(nref, rw, rh, d.vae_s * 2)
            if not (fw and fh):
                raise ValueError(f"reference {rw}×{rh} px is below one latent patch")
            ref_px = [fw, fh]
            ref_rows = nref * (fw // (d.vae_s * 2)) * (fh // (d.vae_s * 2))
        spans = {l.core.span for l in spec.layers}
        attn = "factorized" if spans - {"full", "mode"} else "per_mode" if "mode" in spans else "full"
        info = {"frames": F, "height": Hh, "width": Ww, "fps": d.fps, "video_s": F / d.fps if d.fps else None,
                "latent": [lt, lh, lw], "grid": [gt, gh, gw], "video_tokens": vid, "text_tokens": d.text_tokens,
                "joint_text": d.prefix_tokens, "audio_tokens": aud, "attention": attn,
                "seq_tokens": vid + d.prefix_tokens + aud + ref_rows, "steps": steps, "cfg": cfg,
                "default": {"frames": d.frames, "height": d.height, "width": d.width, "steps": d.steps, "cfg": d.cfg},
                "source": d.source}
        g = -1
        if d.cfg_interval and cfg > 1:              # 0.331 only interval-guided models carry the keys
            g = guided_steps(steps, d.cfg_interval)
            info.update(cfg_interval=list(d.cfg_interval), guided_steps=g, forwards=steps + g)
        if image:                                   # only image models carry the key (video summaries unchanged)
            info["image"] = True
        if d.max_refs:                              # 0.252 only models with a reference stream carry the keys
            info.update(ref_images=nref, ref_px=ref_px, ref_tokens=ref_rows, max_refs=d.max_refs)
        if attn == "factorized":
            info["attn_groups"] = {"spatial": [gt, gh * gw], "temporal": [gh * gw, gt]}
        return ResolvedWorkload("gen", vid + d.prefix_tokens + aud + ref_rows, d.text_tokens, cfg,
                                steps, float(F), "frame", info, tuple(warns), frames=gt, aux=aud, ref=ref_rows, guided=g)
    if spec.domain == "genomics":                       # 0.106: one window of k-mer tokens per sequence
        L = w.seq_len or d.seq_len
        if d.max_seq and L > d.max_seq:
            warns.append(f"窗口 {L} token 超过训练长度 {d.max_seq}（RoPE 外推，精度不在本工具范围）")
        k_ = d.kmer or 1
        ov = d.overlap
        if ov >= L:
            # 0.118: a window no longer than the overlap cannot advance (0.106 clamped to L − 1 → 1 new token per
            # window); a record this short is read as one window with no overlap
            warns.append(f"窗口 {L} token 不长于重叠 {ov} token：按单窗口（无重叠）计")
            ov = 0
        info = {"seq_len": L, "tokens": L, "max_seq": d.max_seq, "kmer": k_, "bp_per_window": L * k_,
                "overlap_tokens": ov, "new_bp_per_window": (L - ov) * k_,
                "default": {"seq_len": d.seq_len}, "source": d.source}
        return ResolvedWorkload("genomics", L, 0, 1, 1, 1.0, "window", info, tuple(warns))
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
        cap_ = d.msa or d.xmsa
        if w.msa and cap_ and w.msa > cap_:
            # 0.79: the release pipelines cluster / subsample the MSA to their cap (AF2 / OpenFold max_msa_clusters,
            # Boltz / Protenix max_msa_seqs); the extra rows are evaluated as given (what-if), was silent
            warns.append(f"MSA {w.msa} 行超过发布实现的上限 {cap_} 行（发布代码会聚类 / 截断到该上限）；按所填行数计（what-if）")
        atoms = math.ceil(L * d.atoms_per_res) if d.atoms_per_res else 0
        # 0.61.3: AF2 / OpenFold concatenate the template torsion-angle embeddings (T rows) to the MSA representation
        # (alphafold/model/modules.py EmbeddingsAndEvoformer; openfold model.py embed_templates → torch.cat([m, a]))
        msa_rows = msa + d.templates if d.tmpl_msa and msa else msa
        tri = getattr(w, "tri_attn", "fused")
        ref = tri_reference_chunk(spec.id, L)
        if tri == "fused":
            tc = 0
        elif tri == "materialized":
            tc = -1
        elif tri == "chunked":
            tc = w.tri_chunk or (ref if ref > 0 else 128)
        else:                                       # reference
            tc = ref
        if w.tri_chunk and tri != "chunked":
            warns.append(f"workload.tri_chunk = {w.tri_chunk} 只用于 tri_attn = chunked，已忽略")
        pd = PairDims(L, msa_rows, xmsa, d.templates, atoms, rec, steps, smp, tri_chunk=tc)
        info["tri_attn"] = {"mode": tri, "chunk": tc, "reference_chunk": ref} if tri != "fused" else None
        if info["tri_attn"] is None:
            del info["tri_attn"]
        info.update({"msa": msa, "msa_rows": msa_rows, "xmsa": xmsa, "templates": d.templates, "atoms": atoms, "recycles": rec,
                     "diff_steps": steps, "samples": smp, "pair_dim": d.pair_dim,
                     "default": {"seq_len": d.seq_len, "msa": d.msa or d.xmsa, "recycles": d.recycles,
                                 "steps": d.diff_steps, "samples": d.samples}})
    return ResolvedWorkload("protein", L + d.special_tokens, 0, 1, 1, 1.0, "seq", info, tuple(warns), pair=pd)
