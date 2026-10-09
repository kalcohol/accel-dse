"""Video pipeline components outside the denoiser (0.44): text encoders and VAE decoders.

A text-to-video request runs   text encoder(s) → steps × DiT forward(s) → VAE decode (→ audio VAE decode).
Up to 0.43 only the DiT was evaluated.  From 0.44 the other components are op graphs too, built from their
released checkpoint headers (``accel_dse/data/pipeline/*.json``, see scripts/build_pipeline_data.py) and run on
the same mapping / memplan / schedule path as the denoiser (``evaluate._pipeline_cost``):

  text encoder  every header GEMM of the text transformer at M = prompts · padded text tokens, plus the
                attention score / value products (bidirectional for T5 / umT5, causal for the Llama / Qwen /
                CLIP text towers).  prompts = batch · cfg when the reference encodes a negative prompt for the
                uncond branch, else batch (Open-Sora: learned null embedding; guidance-distilled models).
  VAE decoder   every decoder conv as an implicit GEMM: M = output voxels at the conv's resolution level,
                K = C_in · kernel volume, N = C_out (2-D per-frame convs: kernel kh·kw; depth-to-space /
                transposed convs at their input / output rate).  Resolution levels follow each family's
                up-block schedule (causal temporal upsampling t → e·t − (e−1)); mid-block attention as
                attention ops (per frame for Wan / SD-VAE, full causal 3-D within a tile for HunyuanVideo);
                the MiniMax-H3 decoder is a ViT over 5 + 2-latent-frame clips, spatially tiled as released (0.47).  Norm / activation / residual:
                8 vector element-ops per output element 「假设」(BigVGAN anti-aliased activations: 60).
  storage       the whole loaded checkpoint of each component at its released dtype (a VAE's encoder and the
                H3 text encoder's vision tower / LM head are loaded with the pipeline though not run).

Placement 「假设」: the components run on one card, serially with the denoise loop (clip latency =
text + denoise + decode); optional multi-card tile-parallel decode (0.47, ``workload.vae_parallel``, the MiniMax-H3
release's ``parallel_tiling``) in ``evaluate._pipeline_cost``; text-encoder weights live on the
first pipeline stage's card, VAE weights on the last stage's card.  Activation peak of a decode = the largest
op of one decode chunk (one latent frame for the causal-cache decoders, a tile / chunk where the reference
tiles), not additive to the denoiser's (it runs after the loop).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .dtypes import fmt as _fmt
from .ir import Op

DATA = Path(__file__).resolve().parent.parent / "data" / "pipeline"
VEC_PER_OUT = 8.0          # norm + activation + residual element-ops per conv / linear output element 「假设」


@lru_cache(maxsize=None)
def component_data(key: str) -> dict:
    return json.loads((DATA / f"{key}.json").read_text())


@dataclass(frozen=True)
class TextEnc:
    key: str
    label: str
    tokens: int              # padded tokens per prompt
    heads: int
    head_dim: int
    causal: bool = False
    note: str = ""


@dataclass(frozen=True)
class Vae:
    key: str
    family: str
    label: str
    act: str = ""            # activation dtype of the reference decode ("" → the denoiser's)
    note: str = ""


@dataclass(frozen=True)
class Pipeline:
    text: tuple[TextEnc, ...]
    vae: Vae
    audio: Vae | None = None
    neg_prompt: bool = True  # reference encodes a negative / empty prompt for the CFG uncond branch
    note: str = ""


_UMT5 = TextEnc("umt5-xxl", "umT5-XXL 编码器", 512, 64, 64,
                note="Wan 参考实现 text_len = 512（padding 到 512）")
_WAN_VAE = Vae("wan-vae", "wan", "Wan-VAE", act="fp32",
               note="参考实现 WanVAE 以 float32 运行（diffusers 示例亦 torch_dtype=float32）：激活按 fp32")
_COG_NOTE = "CogVideoX 默认不开 VAE tiling（文档示例开启以省显存）：按整段 causal 解码（frame_batch_size = 2 潜帧）计"

PIPELINES: dict[str, Pipeline] = {
    "wan2.1-14b": Pipeline((_UMT5,), _WAN_VAE),
    "wan2.1-1.3b": Pipeline((_UMT5,), _WAN_VAE),
    "wan2.2-a14b": Pipeline((_UMT5,), _WAN_VAE),
    "cogvideox-5b": Pipeline((TextEnc("t5-v1_1-xxl.bf16", "T5 v1.1 XXL 编码器", 226, 64, 64),),
                             Vae("cogvideox-vae.5b", "cog", "CogVideoX VAE", note=_COG_NOTE)),
    "cogvideox-2b": Pipeline((TextEnc("t5-v1_1-xxl.fp16", "T5 v1.1 XXL 编码器", 226, 64, 64),),
                             Vae("cogvideox-vae.2b", "cog", "CogVideoX VAE", note=_COG_NOTE)),
    "hunyuanvideo": Pipeline(
        (TextEnc("llava-llama3-8b", "LLaVA-Llama-3-8B 文本塔", 351, 32, 128, causal=True,
                 note="提示模板 crop_start 95 + 256 token = 351 token 输入；取倒数第 3 层隐状态，参考实现仍跑满 32 层"),
         TextEnc("clip-vit-l-text", "CLIP ViT-L 文本塔", 77, 12, 64, causal=True)),
        Vae("hunyuan-vae", "hunyuan", "HunyuanVideo 3D VAE",
            note="中间块为整段 causal 3D 注意力，参考实现必须分块解码：按 diffusers 默认 tiling（空间 tile 256 px / "
                 "stride 192，时间 tile 16+1 帧 / stride 12）计，重叠区重复计算计入"),
        neg_prompt=False),
    "ltx-video": Pipeline((TextEnc("t5-v1_1-xxl.fp32", "T5 v1.1 XXL 编码器", 128, 64, 64),),
                          Vae("ltx-vae", "ltx", "LTX-Video VAE")),
    "mochi-1": Pipeline((TextEnc("t5-v1_1-xxl.mochi", "T5 v1.1 XXL 编码器", 256, 64, 64),),
                        Vae("mochi-vae", "mochi", "Mochi AsymmVAE",
                            note="激活峰值按逐潜帧解码（framewise decoding）计「假设」；diffusers 默认整段解码时峰值更高")),
    "opensora-stdit3": Pipeline((TextEnc("t5-v1_1-xxl.deepfloyd", "T5 v1.1 XXL 编码器（DeepFloyd）", 300, 64, 64),),
                                Vae("opensora-vae", "os", "Open-Sora VAE v1.2（SD-VAE 2D + 时间 VAE）"),
                                neg_prompt=False,
                                note="uncond 分支用学习到的 null 文本嵌入：每请求只编码 1 条提示"),
    "minimax-h3": Pipeline(
        (TextEnc("qwen3-vl-32b", "Qwen3-VL 文本塔", 512, 64, 128, causal=True,
                 note="参考实现读取 hidden_states[50]（64 层中第 50 层），HF 前向仍跑满 64 层：按 64 层计"),),
        Vae("h3-vae", "h3", "MiniMax-H3 ViT 视频解码器",
            note="ViT 解码器按发布配置：时间上 5n+2 潜帧 → n 段、每段 5+2 潜帧（clip_length 17 / token_drop 3）；"
                 "空间总是分块（vae_decoder_tiling = 1，tile 256 px、最小重叠 64 px），块内全注意力 + 5 个 register token"),
        audio=Vae("h3-audio-vae", "audio", "MiniMax-H3 音频 VAE（BigVGAN 型）",
                  note="按声道数逐路解码 40 Hz 潜变量 → 32 kHz 波形「假设」"),
        neg_prompt=False),
}


def pipeline_for(model_id: str) -> Pipeline | None:
    return PIPELINES.get(model_id)


def stored_bytes(key: str) -> float:
    d = component_data(key)
    return d["params_stored"] * _fmt(d["dtype"]).bytes


# ---------------------------------------------------------------- text encoders
_TE_LAYER = re.compile(r"(?:^|\.)(?:block|blocks|layers)\.(\d+)\.")
_TE_SKIP = re.compile(r"embed|shared\.|lm_head|relative_attention_bias|position|visual|norm|text_projection")


def text_ops(te: TextEnc, prompts: int, af: str) -> tuple[list[Op], float]:
    """Op list of one text-encoder pass over ``prompts`` padded prompts; (ops, activation peak bytes)."""
    d = component_data(te.key)
    wf = d["dtype"]
    wb = _fmt(wf).bits
    ab = _fmt(af).bytes
    rows = prompts * te.tokens
    ops: list[Op] = []
    layers = set()
    peak = 0.0
    for name, sh in sorted(d["tensors"].items()):
        mm = _TE_LAYER.search(name)
        if mm:
            layers.add(int(mm.group(1)))
        if len(sh) != 2 or _TE_SKIP.search(name):
            continue
        n, k = sh
        ops.append(Op("te." + name, "gemm", 0, m=rows, k=k, n=n, role="text_encoder", w_params=k * n, w_bits=wb,
                      w_fmt=wf, a_fmt=af, act_bytes=rows * (k + n) * ab, stream=True, vec=rows * n * 3.0))
        peak = max(peak, rows * (k + n) * ab)
    causal = 0.5 if te.causal else 1.0
    cnt = prompts * te.heads
    T, dh = te.tokens, te.head_dim
    for li in sorted(layers):
        ops += [Op(f"te.{li}.qk", "attn", li, m=T, k=dh, n=T, count=cnt, causal=causal,
                   act_bytes=cnt * T * dh * ab, stream=True, orient=True),
                Op(f"te.{li}.pv", "attn", li, m=T, k=T, n=dh, count=cnt, causal=causal,
                   act_bytes=cnt * T * dh * ab, orient=True),
                Op(f"te.{li}.softmax", "vector", li, vec=cnt * T * T * causal * 5)]
    return ops, peak


def te_layers(te: TextEnc) -> tuple[int, float]:
    """(transformer layers, bytes of the largest layer) of a text encoder — the unit an FSDP-sharded encoder
    all-gathers (Wan ``--t5_fsdp``)."""
    d = component_data(te.key)
    wb = _fmt(d["dtype"]).bytes
    per: dict = {}
    for name, sh in d["tensors"].items():
        mm = _TE_LAYER.search(name)
        if mm:
            per[int(mm.group(1))] = per.get(int(mm.group(1)), 0) + math.prod(sh) * wb
    return len(per), max(per.values(), default=0.0)


# ---------------------------------------------------------------- VAE decoders
@dataclass(frozen=True)
class _Fam:
    block_re: str                       # regex capturing the up-block index
    order: tuple[int, ...]              # execution order of up blocks
    ups: dict                           # block → (temporal factor, spatial factor)
    pre: tuple[str, ...] = ()           # upsampler tensors that run at the block's input resolution
    post: tuple[str, ...] = ()          # non-upsampler tensors that run after the block's upsample (output resolution)
    trule: str = "causal"               # causal: t → e·t − (e−1);  double: t → e·t
    attn: str = ""                      # frame | 3d | ""
    attn_dim: int = 0
    chunk: str = "frame"                # activation-peak chunk: frame (latent frame) | cog2 | whole | tile
    shift: tuple = ()                   # (tensor substring, k): those tensors' captured index + k is their block


_TAIL = ("conv_out", "norm_out", "block_out", "proj_out", "conv_post", "conv_norm_out")
_UPS = ("upsamplers", "conv_blocks", ".proj.")

FAMILIES = {
    "wan": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2, 3), {0: (2, 2), 1: (2, 2), 2: (1, 2)}, pre=("time_conv",),
                attn="frame", attn_dim=384),
    "cog": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2, 3), {0: (2, 2), 1: (2, 2), 2: (1, 2)}, chunk="cog2"),
    "hunyuan": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2, 3), {0: (1, 2), 1: (2, 2), 2: (2, 2)}, attn="3d",
                    attn_dim=512, chunk="tile"),
    # 0.61.2: diffusers LTXVideoUpBlock3d runs conv_in → upsamplers → resnets, so the resnets are at the block's
    # output resolution (0.44–0.61.1 put them at the input resolution: decode 2.5–3.3× low)
    "ltx": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2, 3), {1: (2, 2), 2: (2, 2), 3: (2, 2)}, pre=("upsamplers",),
                post=("resnets",), chunk="whole"),
    "mochi": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2), {0: (3, 2), 1: (2, 2), 2: (1, 2)}, pre=(".proj.",)),
    # Open-Sora VAE v1.2: temporal VAE (MAGVIT-v2 style, at latent spatial size) then the SD-VAE 2-D decoder per frame
    # 0.61.3: Decoder.forward runs conv_blocks[i − 1] at the end of block i (before its depth-to-space), so
    # conv_blocks.k belongs to block k + 1 at that block's input resolution (0.44–0.61.2 keyed it to block k:
    # conv_blocks.2 / .1 at twice their frame count, temporal decode +8.9 %)
    "os_t": _Fam(r"(?:block_res_blocks|conv_blocks)\.(\d+)\.", (3, 2, 1, 0), {3: (2, 1), 2: (2, 1)},
                 pre=("conv_blocks",), trule="double", chunk="whole", shift=("conv_blocks", 1)),
    "os_s": _Fam(r"up_blocks\.(\d+)\.", (0, 1, 2, 3), {0: (1, 2), 1: (1, 2), 2: (1, 2)}, attn="frame",
                 attn_dim=512),
}


def _up(res: tuple[int, int, int], f: tuple[int, int] | None, rule: str) -> tuple[int, int, int]:
    if not f:
        return res
    t, h, w = res
    et, es = f
    t2 = t * et if rule == "double" else t * et - (et - 1)
    return t2, h * es, w * es


def _conv_walk(tensors: dict, prefix: str, fam: _Fam, lat: tuple[int, int, int], wf: str, af: str, count: int,
               batch: int, tag: str) -> tuple[list[Op], float, tuple[int, int, int]]:
    """Implicit-GEMM ops of one conv decoder (``count`` identical tiles of latent size ``lat``, ``batch`` clips)."""
    wb = _fmt(wf).bits
    ab = _fmt(af).bytes
    pre_res, post_res, res = {}, {}, lat
    for i in fam.order:
        pre_res[i] = res
        res = _up(res, fam.ups.get(i), fam.trule)
        post_res[i] = res
    final = res
    bre = re.compile(fam.block_re)
    chunks = {"frame": lat[0], "cog2": math.ceil(lat[0] / 2), "whole": 1, "tile": 1}[fam.chunk]
    ops: list[Op] = []
    peak = 0.0
    for name, sh in sorted(tensors.items()):
        if not name.startswith(prefix) or len(sh) < 2 or not name.endswith(("weight", "weight_v")):
            continue
        if "norm" in name and "conv" not in name:
            continue
        mm = bre.search(name)
        bi = int(mm.group(1)) + (fam.shift[1] if fam.shift and fam.shift[0] in name else 0) if mm else -1
        if mm and bi in pre_res:
            i = bi
            up = any(u in name for u in _UPS)
            r = post_res[i] if (up and not any(p in name for p in fam.pre)) or any(p in name for p in fam.post) \
                else pre_res[i]
        elif any(t in name for t in _TAIL):
            r = final
        else:
            r = lat
        t, h, w = r
        vox = t * h * w * batch
        n, k = sh[0], math.prod(sh[1:])
        cin = sh[1]
        # input tensor at the conv's input resolution: = output voxels, except a depth-to-space conv (same res)
        # and the 2-D conv after an upsample (input at the pre-upsample res → ≤ output voxels; take output, 「假设」)
        ops.append(Op(f"{tag}.{name}", "gemm", 0, m=vox, k=k, n=n, count=count, role="vae", w_params=k * n * count,
                      w_bits=wb, w_fmt=wf, a_fmt=af, act_bytes=vox * (cin + n) * count * ab, stream=True,
                      vec=vox * n * count * VEC_PER_OUT, in_elems=vox * cin * count))
        peak = max(peak, vox / chunks * (cin + n) * ab)
    if fam.attn:
        t, h, w = lat
        dd = fam.attn_dim
        if fam.attn == "frame":
            groups, L, causal = t * batch * count, h * w, 1.0
        else:                              # full 3-D attention, causal by latent frame
            groups, L, causal = batch * count, t * h * w, (1 + 1 / t) / 2
        ops += [Op(f"{tag}.mid.qk", "attn", 0, m=L, k=dd, n=L, count=groups, causal=causal,
                   act_bytes=groups * L * dd * ab, stream=True, orient=True),
                Op(f"{tag}.mid.pv", "attn", 0, m=L, k=L, n=dd, count=groups, causal=causal,
                   act_bytes=groups * L * dd * ab, orient=True),
                Op(f"{tag}.mid.softmax", "vector", 0, vec=groups * L * L * causal * 5)]
        peak = max(peak, 4 * L * dd * ab)
    return ops, peak, final


def _hunyuan_tiles(lat: tuple[int, int, int]) -> list[list[tuple[int, int, int]]]:
    """diffusers AutoencoderKLHunyuanVideo tiled decode: temporal tiles of 4+1 latent frames every 3, spatial tiles of
    32×32 latent every 24 (tile_sample_min 256 px / stride 192 px, 16 / 12 frames) → decode rounds (one spatial
    ``tiled_decode`` call per temporal tile), each the row-major list of tile latent sizes."""
    T, H, W = lat
    ts = [len(range(T)[i:i + 5]) for i in range(0, T, 3)] if T > 4 else [T]
    tiled = H > 32 or W > 32
    hs = [len(range(H)[i:i + 32]) for i in range(0, H, 24)] if tiled else [H]
    ws = [len(range(W)[j:j + 32]) for j in range(0, W, 24)] if tiled else [W]
    return [[(a, b, c) for b in hs for c in ws] for a in ts]


# optional spatial tiling of diffusers ``enable_tiling()`` (0.45), latent units: (tile, stride) per axis (H, W)
#   CogVideoX  tile_sample_min = sample size / 2 = 240 × 360 px → 30 × 45 latent; overlap factors 1/6, 1/5 →
#              stride int(30·5/6) = 25, int(45·4/5) = 36   (AutoencoderKLCogVideoX)
#   Mochi      tile_sample_min 256 px, stride 192 px → 32 / 24 latent                (AutoencoderKLMochi)
#   Wan        tile_sample_min 256 px, stride 192 px → 32 / 24 latent (0.46)         (AutoencoderKLWan; the official
#              Wan repo decodes untiled — the option models the diffusers path)
#   LTX-Video  tile_sample_min 512 px, stride 448 px, spatial compression 32 → 16 / 14 latent (0.47)
#              (AutoencoderKLLTXVideo; framewise decoding stays off, as enable_tiling() leaves it)
# Open-Sora VAE v1.2 has no spatial tiling in its release (it micro-batches frames: 2-D VAE 4 frames, temporal VAE
# 17-frame micro-chunks — already how it is costed) and is not in diffusers.  MiniMax-H3 always tiles (below).
TILING = {"cog": ((30, 25), (45, 36)), "mochi": ((32, 24), (32, 24)), "wan": ((32, 24), (32, 24)),
          "ltx": ((16, 14), (16, 14))}


def _spatial_tiles(lat: tuple[int, int, int], spec: tuple) -> list[list[tuple[int, int, int]]]:
    """diffusers ``tiled_decode``: once either axis exceeds the tile, both axes step by the stride from 0 (a short
    axis then yields a full tile plus a sliver, as the reference loops do).  One decode round."""
    T, H, W = lat
    (th, sh), (tw, sw) = spec
    if H <= th and W <= tw:
        return [[(T, H, W)]]
    hs = [len(range(H)[i:i + th]) for i in range(0, H, sh)]
    ws = [len(range(W)[j:j + tw]) for j in range(0, W, sw)]
    return [[(T, b, c) for b in hs for c in ws]]


# MiniMax-H3 visual VAE decode (0.47), as released: ``FL2VA/video_vae/config.json`` vae_decoder_tiling = 1,
# vae_tile_size 256 px, vae_tile_overlap_min 64 px (diffusers AutoencoderKLMiniMaxH3: "spatial tiling is on by
# default … disabling tiling changes the output"); temporal clips of tokens_chunk_size + token_overlap latent frames
H3_TILE_PX, H3_OVERLAP_PX = 256, 64


def _h3_split(px: int, ratio: int) -> list[int]:
    """AutoencoderKLMiniMaxH3._split_tiles: the fewest full-size tiles whose overlaps stay ≥ the minimum → latent
    lengths of the tiles along one axis."""
    if H3_TILE_PX >= px:
        return [px // ratio]
    n = math.ceil(px / H3_TILE_PX)
    while H3_TILE_PX * n - H3_OVERLAP_PX * (n - 1) - px < 0:
        n += 1
    return [H3_TILE_PX // ratio] * n


def h3_geometry(lat: tuple[int, int, int], config: dict) -> dict:
    """Temporal clips and spatial tiles of one H3 decode (diffusers ``_decode``: 5n+2 latent frames → n clips of
    5 + 2 latent frames; each clip spatially tiled)."""
    T, H, W = lat
    ratio = math.prod(config.get("spatial_downsample_factors", [2, 2, 2, 2]))
    rt = math.prod(config.get("temporal_downsample_factors", [1, 2, 2]))
    chunk = math.ceil(config.get("clip_length", 17) / rt)
    drop = config.get("token_drop", 3)
    over = (-drop) % chunk
    pad = (-(T + drop)) % chunk
    clips = max(1, (T + drop + pad) // chunk - int(drop > 0))
    hs, ws = _h3_split(H * ratio, ratio), _h3_split(W * ratio, ratio)
    return {"clips": clips, "clip_t": chunk + over, "hs": hs, "ws": ws}


def _tiled(ten: dict, fam: str, rounds: list, wf: str, af: str, batch: int, lat: tuple, info: dict) -> tuple:
    count: dict = {}
    for rd in rounds:
        for tl in rd:
            count[tl] = count.get(tl, 0) + 1
    groups, peak, by = [], 0.0, {}
    for tl, c in sorted(count.items()):  # one op list per distinct tile size, executed c · batch times (weights re-read)
        o, p, _ = _conv_walk(ten, "", FAMILIES[fam], tl, wf, af, 1, 1, f"vae[{tl[0]}x{tl[1]}x{tl[2]}]")
        groups.append((o, c * batch))
        by[tl] = o
        peak = max(peak, p)
    return [], peak, _tile_info(info, rounds, groups, by, lat)


def _tile_info(info: dict, rounds: list, groups: list, by: dict, lat: tuple) -> dict:
    vol = sum(math.prod(tl) for rd in rounds for tl in rd)
    info["tiles"] = sum(len(rd) for rd in rounds)
    info["overlap"] = vol / math.prod(lat)
    info["groups"] = groups
    info["rounds"] = rounds          # decode order: [[tile latent size, …] per independent tiled call]
    info["tile_ops"] = by            # tile latent size → op list of one execution (batch 1)
    return info


def vae_ops(v: Vae, lat: tuple[int, int, int], frames: int, batch: int, af: str,
            audio_rows: int = 0, tiling: bool = False) -> tuple[list[Op], float, dict]:
    """Decode ops of one request batch; (ops, activation peak bytes, info).  A tiled decode returns its ops as
    ``info["groups"]`` = [(op list of one tile size, executions)] instead, plus ``info["rounds"]`` (decode order of
    the tiles, for the multi-card split) and ``info["tile_ops"]``.  ``tiling``: diffusers ``enable_tiling()`` for the
    families in ``TILING``; HunyuanVideo and MiniMax-H3 always tile, as their references do."""
    d = component_data(v.key)
    wf, ten = d["dtype"], d["tensors"]
    af = v.act or af
    info: dict = {"act": af}
    if v.family == "hunyuan":
        return _tiled(ten, "hunyuan", _hunyuan_tiles(lat), wf, af, batch, lat, info)
    if tiling and v.family in TILING:
        info["tiling"] = True
        return _tiled(ten, v.family, _spatial_tiles(lat, TILING[v.family]), wf, af, batch, lat, info)
    if v.family == "os":
        t0, h0, w0 = lat
        o1, p1, tfin = _conv_walk(ten, "temporal_vae.", FAMILIES["os_t"], (t0, h0, w0), wf, af, 1, batch, "vae.t")
        o2, p2, _ = _conv_walk(ten, "spatial_vae.", FAMILIES["os_s"], (frames, h0, w0), wf, af, 1, batch, "vae.s")
        p1 = p1 / max(1, math.ceil(t0 / 5))         # temporal VAE decodes 5-latent micro-chunks
        return o1 + o2, max(p1, p2), info
    if v.family == "h3":
        c = d["config"]
        heads, dh = c.get("decoder_num_attention_heads", 32), c.get("decoder_attention_head_dim", 64)
        regs = c.get("decoder_num_register_tokens", 4) + 1          # register tokens + cls token
        g = h3_geometry(lat, c)
        wb, ab = _fmt(wf).bits, _fmt(af).bytes
        ff = max(sh[0] for nm, sh in ten.items() if "ff.net.0" in nm and len(sh) == 2)
        layers = sorted({int(mm.group(1)) for nm in ten for mm in [re.search(r"transformer_blocks\.(\d+)\.", nm)] if mm})
        rounds = [[(g["clip_t"], a_, b_) for a_ in g["hs"] for b_ in g["ws"]] for _ in range(g["clips"])]
        by, groups, peak = {}, [], 0.0
        for tl in sorted(set(rounds[0])):
            L = math.prod(tl) + regs
            ops = []
            for name, sh in sorted(ten.items()):
                if len(sh) < 2 or not name.endswith("weight"):
                    continue
                n, k = sh[0], math.prod(sh[1:])
                ops.append(Op(f"vae[{tl[1]}x{tl[2]}]." + name, "gemm", 0, m=L, k=k, n=n, role="vae", w_params=k * n,
                              w_bits=wb, w_fmt=wf, a_fmt=af, act_bytes=L * (k + n) * ab, stream=True, vec=L * n * 3.0))
            for li in layers:
                ops += [Op(f"vae.{li}.qk", "attn", li, m=L, k=dh, n=L, count=heads, act_bytes=heads * L * dh * ab,
                           stream=True, orient=True),
                        Op(f"vae.{li}.pv", "attn", li, m=L, k=L, n=dh, count=heads, act_bytes=heads * L * dh * ab,
                           orient=True),
                        Op(f"vae.{li}.softmax", "vector", li, vec=heads * L * L * 5)]
            by[tl] = ops
            groups.append((ops, sum(rd.count(tl) for rd in rounds) * batch))
            peak = max(peak, L * (heads * dh + ff) * ab * batch)
        info.update(chunks=g["clips"], chunk_tokens=math.prod(rounds[0][0]) + regs, tiling=True)
        return [], peak, _tile_info(info, rounds, groups, by, lat)
    if v.family == "audio":
        c = d["config"]
        rates = c.get("decoder_rates", [])
        wb, ab = _fmt(wf).bits, _fmt(af).bytes
        n_res = len(c.get("resblock_kernel_sizes", [3, 7, 11]))
        lens = [audio_rows]
        for r in rates:
            lens.append(lens[-1] * r)
        ops, peak = [], 0.0
        for name, sh in sorted(ten.items()):
            if not name.endswith("weight_v") or len(sh) < 3:
                continue
            m_up = re.search(r"ups\.(\d+)\.", name)
            m_rb = re.search(r"resblocks\.(\d+)\.", name)
            if m_up:            # ConvTranspose1d [C_in, C_out, k] with stride = rate: MACs = L_in · C_in · C_out · k
                i = int(m_up.group(1))
                cin, n, kk = sh
                m, k = lens[i + 1], max(1, cin * kk // rates[i])
            else:
                lvl = (int(m_rb.group(1)) // n_res + 1) if m_rb else (len(rates) if "conv_post" in name else 0)
                n, cin, kk = sh
                m, k = lens[lvl], cin * kk
            ops.append(Op("avae." + name, "gemm", 0, m=m * batch, k=k, n=n, role="vae", w_params=math.prod(sh),
                          w_bits=wb, w_fmt=wf, a_fmt=af, act_bytes=m * batch * (cin + n) * ab, stream=True,
                          vec=m * batch * n * 60.0, in_elems=m * batch * cin))
            peak = max(peak, m * batch * (cin + n) * ab)
        info["samples"] = lens[-1]
        return ops, peak, info
    fam = FAMILIES[v.family]
    ops, peak, fin = _conv_walk(ten, "", fam, lat, wf, af, 1, batch, "vae")
    info["out"] = fin
    return ops, peak, info
