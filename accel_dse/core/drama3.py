"""0.276 Fish Audio Drama 3 TTS (docs/MODEL.md §221.3).

Reference code 「来源」 github vllm-project/vllm-omni@da572cd vllm_omni/model_executor/models/fish_omni/
(fish_omni_model.py, codebook_tower.py, pipeline.py; stage_input_processors/fish_omni.py) + HF fishaudio/drama-3
config.json / safetensors headers (README: 35B total, 3B active, FP8 MoE experts, rest bf16).

One forward = one sampled audio frame per request (Stage 0, ``fish_omni``):
  1. the input row ``[main, c0..c9]`` → main-token embedding + the backbone codebook's embedding, scaled
     1/√(1 + vq_backbone_codebooks) → Qwen3.5-MoE backbone (40 layers: 30 Gated-DeltaNet + 10 gated full attention,
     256 experts top-8 + shared expert, FP8 experts);
  2. the main token is drawn from the lm_head rows of end-of-speech + semantic codes 1..4095 only (vllm-omni keeps
     those 4096 rows — "cuts the per-frame head read from 1 GB to 16 MB");
  3. the codebook-MTP tower: 13 layers (4 shared + 9 per codebook), each mixes one code embedding into the hidden
     state (eh_proj; layers ≥ 1 fold the embedding half into a per-code table at load) and runs one gated-attention
     transformer block (GQA 16 / 2, head 256, SwiGLU 6144) whose causal attention covers the whole sequence — a paged
     KV cache with one row per backbone position; code k is drawn from codebook head k (9 × Linear 2048 → 4096).
  → modelled as 40 + 13 decoder layers on every row (prefill and decode) with the 9 heads as per-sequence rows
     「推导」: same FLOPs, weights and KV bytes; the serial sample between tower layers is not a separate latency 「近似」.
Stage 1 (``dac_decoder``, the Fish DAC of S2 Pro, 44.1 kHz, 2048 samples per frame), streamed in chunks ramping 6, 8,
12, 16, 24, 32 then 64 frames with 32 frames of left context 「来源」.  0.277: costed from codec.pth's tensor shapes
(scripts/fetch_fish_codec.py → release ``codec_tensors``) and fish-speech fish_speech/models/dac/{rvq,modded_dac}.py
「来源」 — see :func:`dac_ops`.  First audio = backbone prefill + 6 decode steps + the DAC on that first 6-frame window.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .ir import Op

CODEC_RATE = 44100      # fish_omni_model._CODEC_SAMPLE_RATE 「来源」
CODEC_HOP = 2048        # fish_omni_model._CODEC_HOP 「来源」
CHUNK_RAMP = (6, 8, 12, 16, 24, 32)     # stage_input_processors/fish_omni.DEFAULT_RAMP 「来源」
CHUNK = 64              # DEFAULT_CHUNK
LEFT_CONTEXT = 32       # DEFAULT_LEFT_CONTEXT
REF_S = 10.0            # voice-reference length (one prompt row per reference frame) 「假设」
DAC_WINDOW = 128        # post_module WindowLimitedTransformer window_size (fish-speech configs/modded_dac_vq.yaml) 「来源」;
                        # codec.pth carries no hyper-parameters, so the Drama 3 codec is taken to use the same 「假设」
DAC_HEAD_DIM = 64       # head_dim (same yaml); freqs_cis [4096, 32, 2] in codec.pth = 64 / 2 rotary pairs 「来源」
VEC_PER_OUT = 3.0       # epilogue element-ops per GEMM output (bias / residual / scale), as cosyvoice.py


@dataclass(frozen=True)
class FishSpec:
    n_codebooks: int = 10
    codebook_size: int = 4096
    shared_layers: int = 4
    tower_layers: int = 13
    dim: int = 2048
    n_head: int = 16
    n_kv: int = 2
    head_dim: int = 256
    inter: int = 6144
    candidates: int = 4096
    codec_params: int = 0       # 0.277: every weight tensor of codec.pth (encoder + quantizer + decoder)
    kind: str = "fish"
    codec: tuple = ()           # 0.277: ((name, shape), …) DAC decode path (release codec_tensors); () = not costed

    @property
    def frame_s(self) -> float:
        return CODEC_HOP / CODEC_RATE

    @property
    def frames_per_s(self) -> float:
        return CODEC_RATE / CODEC_HOP

    def ref_rows(self, ref_s: float) -> int:
        return max(0, -(-int(round(ref_s * CODEC_RATE)) // CODEC_HOP))


def is_fish(cfg: dict) -> bool:
    return (cfg or {}).get("model_type") == "fish_omni" and isinstance((cfg or {}).get("audio_decoder_config"), dict)


def n_candidates(cfg: dict) -> int:
    """fish_omni_model.main_candidate_ids: end of speech + semantic codes 1..end 「来源」."""
    return 1 + cfg["semantic_end_token_id"] - cfg["semantic_start_token_id"]


def fish_spec(cfg: dict, rel: dict | None = None) -> FishSpec | None:
    if not is_fish(cfg):
        return None
    a = cfg["audio_decoder_config"]
    q = a["num_codebooks"]
    ct = (rel or {}).get("codec_tensors") or {}
    return FishSpec(q, a["vocab_size"], a["shared_enrichment_layers"], a["shared_enrichment_layers"] + q - 1,
                    a["dim"], a["n_head"], a["n_local_heads"], a["head_dim"], a["intermediate_size"], n_candidates(cfg),
                    int((rel or {}).get("params_codec") or 0), "fish", tuple((n, tuple(v)) for n, v in ct.items()))


def dac_params(fs: FishSpec) -> int:
    """Decode-path weights resident on the speech card (fp32, codec.pth dtype 「来源」)."""
    return sum(math.prod(s) for _, s in fs.codec)


def dac_rates(fs: FishSpec) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(quantizer upsample factors, decoder strides) from the shapes 「推导」: a ConvTranspose1d weight is [in, out, k];
    the quantizer's upsample uses k = stride (rvq.DownsampleResidualVectorQuantize), the decoder's DecoderBlock
    k = 2 · stride (modded_dac.DecoderBlock).  Product = the codec hop (Drama 3: 2·2 · 8·8·4·2 = 2048)."""
    sh = dict(fs.codec)
    up = tuple(sh[f"quantizer.upsample.{i}.0.conv.weight"][2] for i in range(8) if f"quantizer.upsample.{i}.0.conv.weight" in sh)
    dec = []
    for i in range(1, 16):
        w = sh.get(f"decoder.model.{i}.block.1.conv.parametrizations.weight.original1")
        if w is None:
            break
        dec.append(w[2] // 2)
    return up, tuple(dec)


def dac_ops(fs: FishSpec, new: int, ctx: int = 0, hist: int = 0, batch: int = 1) -> list[Op]:
    """One DAC decode of a window: ``new`` frames decoded, plus ``ctx`` left-context frames re-decoded (vllm-omni's module
    decoder) — ``ctx`` = 0 with ``hist`` frames of kept state is the stateful path (fish_speech_dac_stateful: only the
    new frames, window-attention keys from the kept KV) 「来源」 recipe Fish-Audio-Drama-3.md.
    Decode = rvq.decode 「来源」: codebook lookups (semantic 4096 + 9 residual 1024, dim 8) → out_proj (8 → 1024) summed
    → post_module (8-layer causal window transformer, dim 1024, 16 × 64 heads, SwiGLU 3072, window 128) at the frame rate
    → upsample (2 × [ConvTranspose k = s = 2 + ConvNeXt block]) → decoder (conv7 1024 → 1536, 4 DecoderBlocks
    [Snake, ConvTranspose k = 2s, 3 ResidualUnits (Snake, conv7 dilated 1 / 3 / 9, Snake, conv1)], Snake, conv7 → 1).
    One fp32 GEMM per weight tensor with rows = the positions it runs at (a ConvTranspose per input position, n = out · k);
    weight-norm gains / biases / Snake α / norms ride with their module; Snake = 5 element-ops + 1 sin per input element;
    attention = QKᵀ / PV over min(window, position) keys 「近似」 (the window's last row, an upper bound).
    ``batch`` streams' windows in one forward (vllm-omni's per-step batch of the codec stage 「来源」 sip.py
    ``per_step_batch``): rows × batch, weights read once 「近似」 (the streams' windows taken as aligned)."""
    F = new + ctx
    up, dec = dac_rates(fs)
    sh = dict(fs.codec)
    cum_up = [math.prod(up[:i + 1]) for i in range(len(up))]
    base = cum_up[-1] if cum_up else 1
    T = [F * base]
    for s_ in dec:
        T.append(T[-1] * s_)

    def rows(n: str):
        if ".in_proj." in n or n.endswith("codebook.weight"):
            return None                                     # encode-only / lookups (below)
        if n.startswith(("quantizer.semantic_quantizer.", "quantizer.quantizer.", "quantizer.post_module.")):
            return F
        m_ = re.match(r"quantizer\.upsample\.(\d+)\.(\d)\.", n)
        if m_:
            i, j = int(m_.group(1)), int(m_.group(2))
            return F * (cum_up[i - 1] if i else 1) if j == 0 else F * cum_up[i]
        m_ = re.match(r"decoder\.model\.(\d+)\.", n)
        if m_:
            i = int(m_.group(1))
            if i == 0:
                return T[0]
            if i <= len(dec):
                return T[i - 1] if ".block.1." in n and ".block.1.block" not in n and n.count(".block.") == 1 else T[i]
            return T[-1]
        return F

    def module(n: str) -> str:
        return re.sub(r"\.(weight|bias|weight_g|weight_v|parametrizations\.weight\.original[01])$", "", n)

    def main_w(n: str, s) -> bool:
        return len(s) >= 2 and not n.endswith(("original0", "weight_g", ".alpha", "codebook.weight"))

    side_of: dict[str, int] = {}
    for n, s in fs.codec:
        if not main_w(n, s) and not n.endswith(".alpha"):
            side_of[module(n)] = side_of.get(module(n), 0) + math.prod(s)
    ops: list[Op] = []
    used = 0
    for n, s in fs.codec:
        if not main_w(n, s):
            continue
        r = rows(n)
        if r is None:
            continue
        r *= batch
        side = side_of.get(module(n), 0)
        numel = math.prod(s)
        tconv = bool(re.match(r"quantizer\.upsample\.\d+\.0\.", n)) or (
            n.startswith("decoder.model.") and n.count(".block.") == 1 and ".block.1." in n and n != "decoder.model.0")
        if tconv:                                            # ConvTranspose1d [in, out, k]: per input position
            k, nn_ = s[0], s[1] * s[2]
        else:                                                # Linear [out, in] / Conv1d [out, in / g, k]
            k, nn_ = numel // s[0], s[0]
        vec = r * nn_ * VEC_PER_OUT
        sfu = 0.0
        if n.startswith("decoder.model.") and not n.startswith("decoder.model.0."):
            cin = s[0] if tconv else s[1]                    # Snake on the conv input
            vec += 5.0 * r * cin
            sfu = r * cin
        ops.append(Op("dac." + n, "gemm", 0, m=r, k=k, n=nn_, role="speech", w_params=numel + side, w_bits=32.0,
                      w_fmt="fp32", a_fmt="fp32", act_bytes=r * (k + nn_) * 4.0, vec=vec, sfu=sfu, stream=True))
        used += numel + side
    # attention of the 8 window-transformer layers
    nl = len({m_.group(1) for n, _ in fs.codec for m_ in [re.match(r"quantizer\.post_module\.layers\.(\d+)\.", n)] if m_})
    if nl:
        d = sh["quantizer.post_module.layers.0.attention.wo.weight"][0]
        h = d // DAC_HEAD_DIM
        keys = min(DAC_WINDOW, hist + F)
        hb = h * nl * batch
        ops += [Op("dac.post_module.qk", "attn", 0, m=F, k=DAC_HEAD_DIM, n=keys, count=hb, a_fmt="fp32",
                   act_bytes=hb * (F + keys) * DAC_HEAD_DIM * 4.0,
                   kv_read=2.0 * hb * min(hist, DAC_WINDOW) * DAC_HEAD_DIM * 4.0, stream=True),
                Op("dac.post_module.pv", "attn", 0, m=F, k=keys, n=DAC_HEAD_DIM, count=hb, a_fmt="fp32",
                   act_bytes=hb * (F + keys) * DAC_HEAD_DIM * 4.0),
                Op("dac.post_module.softmax_norm_rope", "vector", 0, vec=hb * F * keys * 6.0 + nl * batch * F * d * 20.0,
                   sfu=hb * F * keys)]
    q = [s for n, s in fs.codec if n.endswith("codebook.weight")]
    ops.append(Op("dac.codebook_lookup", "lookup", 0, kv_read=batch * F * sum(s[1] for s in q) * 4.0))
    left = dac_params(fs) - used - sum(math.prod(s) for n, s in fs.codec if n.endswith("codebook.weight") or ".in_proj." in n)
    if left > 0:                                             # norms / layer scales not folded above: read once
        ops.append(Op("dac.vectors", "gemm", 0, m=1, k=left, n=1, role="speech", w_params=left, w_bits=32.0, w_fmt="fp32",
                      a_fmt="fp32"))
    return ops


def dac_windows(frames: int) -> list[tuple[int, int, int]]:
    """[(new, left context, frames already emitted)] of one stream's windows 「来源」 stage_input_processors/fish_omni.py:
    the ramp, then CHUNK frames per window, each with min(LEFT_CONTEXT, emitted) earlier frames."""
    out, done, i = [], 0, 0
    while done < frames:
        n = min(CHUNK_RAMP[i] if i < len(CHUNK_RAMP) else CHUNK, frames - done)
        out.append((n, min(LEFT_CONTEXT, done), done))
        done += n
        i += 1
    return out


def tower_layers(fs: FishSpec, Layer, Linear, gqa, ffn_dense) -> list:
    """The 13 codebook-tower layers as decoder layers (codebook_tower.TowerLayer): eh_proj (layer 0: [e; h] 2d → d;
    layers ≥ 1: the h half d → d after the code-table fold 「来源」), qkv with the sigmoid output gate (q ‖ gate), o,
    SwiGLU; norms enorm / hnorm / input / q / k / post / final as misc."""
    d = fs.dim
    out = []
    for i in range(fs.tower_layers):
        lin, core = gqa(d, fs.n_head, fs.n_kv, fs.head_dim, out_gate=True)
        eh = Linear("eh_proj", (2 if i == 0 else 1) * d, d, "rep")
        ffn, fl = ffn_dense(d, fs.inter)
        out.append(Layer((eh,) + tuple(lin), core, ffn, fl, misc_params=5 * d + 2 * fs.head_dim))
    return out


def dac_gflop_per_frame(fs: FishSpec, frames: int = CHUNK) -> float:
    """GEMM + attention GFLOP per decoded frame of a stateful ``frames``-frame window 「推导」 (2 · m · k · n)."""
    ops = dac_ops(fs, frames, 0, 0)
    return sum(2.0 * o.m * o.k * o.n * o.count for o in ops if o.kind in ("gemm", "attn") and o.name != "dac.vectors") / frames / 1e9
