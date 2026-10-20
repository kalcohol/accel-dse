"""0.274 Kyutai Moshi full-duplex speech (docs/MODEL.md §219).

Reference code 「来源」 (github kyutai-labs/moshi@e6a55d2): moshi/models/loaders.py (_lm_kwargs, _mimi_config,
get_mimi), moshi/models/lm.py (LMModel.forward_text / forward_depformer, LMGen._step / depformer_step),
moshi/modules/transformer.py (StreamingTransformer, weights_per_step, multi-linear in_proj), moshi/models/compression.py
(MimiModel encode / decode), moshi/modules/seanet.py, moshi/quantization/vq.py, moshi/server.py; HF
kyutai/moshiko-pytorch-bf16 model.safetensors + tokenizer-e351c8d8-checkpoint125.safetensors (Mimi) headers.

Per 80 ms frame (12.5 Hz), one conversation (server.py: streaming_forever(1), one connection under a lock):
  Mimi encode (user audio, 1920 samples @ 24 kHz, fp32 — get_mimi passes no dtype): SEANet encoder (convs at 1920 /
      480 / 96 / 16 / 2 positions per frame; ratios 4·5·6·8 = 960), 8-layer transformer (d512, 8 × 64 heads, 2
      positions per frame, context 250), stride-2 conv → 12.5 Hz, split RVQ: 8 codebooks used (num_codebooks 8) —
      nearest-code search over 2048 × 256 each.
  Temporal transformer (the catalog model, bf16): one step over the summed embeddings of text + 16 audio codebooks
      (8 user + 8 Moshi), text head; KV ring buffer of context 3000 frames.
  Depth transformer (depformer, bf16): dep_q = 8 sequential steps per frame, one per Moshi codebook; step k runs
      depformer_in[k] (4096 → 1024) + the previous token's embedding through 6 layers whose weights are step k's
      (weights_per_step: in_proj / out_proj slice k of the multi-linear, gating[k]) with attention over the k + 1
      positions of this frame, then linears[k] (1024 → 2048).  Every frame therefore reads each depformer weight once
      (8 disjoint slices, 8 × 6 serial layer passes on one row) and re-reads the whole depformer 12.5 times per second.
  Mimi decode (Moshi's 8 codebooks): dequantize, depthwise upsample ×2, transformer, SEANet decoder (transposed
      convs 8·6·5·4) → 1920 samples.
Real time ⇔ frame time ≤ 80 ms.  Response latency = 2 frames (the 80 ms input frame + the acoustic codebooks' delay
of one frame, _lm_kwargs delays) + one frame's compute — Kyutai's "160 ms theoretical" plus compute.
Ops come from the released tensor shapes (one GEMM per weight at its module rate; ConvTranspose per input position).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .ir import Op

VEC_PER_OUT = 3.0
FRAME_S = 0.08


@dataclass(frozen=True)
class MoshiSpec:
    tensors: tuple                 # ((name, shape), …) depformer* / linears.* / mimi.*
    dep_q: int = 8
    dep_layers: int = 6
    dep_dim: int = 1024
    dep_heads: int = 16
    dim: int = 4096
    frame_rate: float = 12.5
    sample_rate: int = 24000
    n_codebooks: int = 8           # Mimi codebooks per stream used by Moshi
    bins: int = 2048
    q_dim: int = 256
    mimi_dim: int = 512
    mimi_heads: int = 8
    mimi_ctx: int = 250
    ratios: tuple = (8, 6, 5, 4)
    delay_frames: int = 1          # max(_lm_kwargs delays)
    context: int = 3000

    def _p(self, pre) -> int:
        return sum(math.prod(s) for n, s in self.tensors if n.startswith(pre))

    @property
    def depth_params(self) -> int:
        return self._p(("depformer", "linears."))

    @property
    def mimi_params(self) -> int:
        return self._p("mimi.")

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    @property
    def enc_rate(self) -> int:
        """Mimi encoder-transformer positions per frame (SEANet hop 960 → 25 Hz; stride-2 conv → 12.5 Hz)."""
        return self.samples_per_frame // math.prod(self.ratios)

    @property
    def stored_bytes(self) -> float:
        return self.depth_params * 2.0 + self.mimi_params * 4.0


def moshi_spec(cfg: dict, rel: dict | None) -> MoshiSpec | None:
    if cfg.get("model_type") != "moshi" or not rel or not rel.get("speech_tensors"):
        return None
    lm, mi = cfg["lm"], cfg["mimi"]
    return MoshiSpec(tuple((n, tuple(s)) for n, s in rel["speech_tensors"].items()), lm["dep_q"],
                     lm["depformer_num_layers"], lm["depformer_dim"], lm["depformer_num_heads"], lm["dim"],
                     mi["frame_rate"], mi["sample_rate"], lm["dep_q"], mi["quantizer"]["bins"],
                     mi["quantizer"]["dimension"], mi["transformer"]["d_model"], mi["transformer"]["num_heads"],
                     mi["transformer"]["context"], tuple(mi["seanet"]["ratios"]), max(lm["delays"]), lm["context"])


def _g(name: str, m: int, k: int, n: int, bits: float, w: int | None = None, cnt: int = 1) -> Op:
    ab = bits / 8
    return Op(name, "gemm", 0, m=m, k=k, n=n, count=cnt, role="speech", w_params=k * n * cnt if w is None else w,
              w_bits=bits, w_fmt="fp32" if bits == 32 else "bf16", a_fmt="fp32" if bits == 32 else "bf16",
              act_bytes=m * (k + n) * ab * cnt, vec=m * n * VEC_PER_OUT * cnt, stream=True)


def _attn(name: str, q: int, keys: int, hd: int, heads: int, bits: float, b: int) -> list[Op]:
    ab = bits / 8
    c = b * heads
    return [Op(name + ".qk", "attn", 0, m=q, k=hd, n=keys, count=c, a_fmt="fp32" if bits == 32 else "bf16",
               act_bytes=c * (q + keys) * hd * ab, stream=True),
            Op(name + ".pv", "attn", 0, m=q, k=keys, n=hd, count=c, a_fmt="fp32" if bits == 32 else "bf16",
               act_bytes=c * (q + keys) * hd * ab),
            Op(name + ".softmax", "vector", 0, vec=c * q * keys * 6.0)]


def _shape(s: MoshiSpec, name: str) -> tuple:
    for n, sh in s.tensors:
        if n == name:
            return sh
    raise KeyError(name)


def depformer_ops(s: MoshiSpec, b: int) -> list[Op]:
    """One frame: dep_q serial steps × dep_layers layers on b rows; step k reads slice k of in_proj / out_proj and
    gating[k] (weights_per_step) — so the frame reads every depformer weight once."""
    D, hd = s.dep_dim, s.dep_dim // s.dep_heads
    ops: list[Op] = []
    for k in range(s.dep_q):
        ops.append(_g(f"dep.{k}.in", b, s.dim, D, 16))
        ops.append(Op(f"dep.{k}.emb", "lookup", 0, kv_read=b * D * 2.0, vec=b * D))
        for li in range(s.dep_layers):
            p = f"depformer.layers.{li}."
            fi, fo = _shape(s, p + f"gating.{k}.linear_in.weight"), _shape(s, p + f"gating.{k}.linear_out.weight")
            ops += [_g(f"dep.{k}.{li}.qkv", b, D, 3 * D, 16), _g(f"dep.{k}.{li}.o", b, D, D, 16),
                    _g(f"dep.{k}.{li}.ff_in", b, fi[1], fi[0], 16), _g(f"dep.{k}.{li}.ff_out", b, fo[1], fo[0], 16),
                    Op(f"dep.{k}.{li}.norm_act", "vector", li, vec=b * (D * 8 + fo[1] * 4))]
            ops += _attn(f"dep.{k}.{li}.attn", 1, k + 1, hd, s.dep_heads, 16, b)
        ops.append(_g(f"dep.{k}.head", b, D, s.bins, 16))
        ops.append(Op(f"dep.{k}.sample", "vector", 0, vec=b * s.bins * 3, sfu=b * s.bins))
    norms = sum(math.prod(sh) for n, sh in s.tensors if n.startswith("depformer.layers.") and n.endswith(".alpha"))
    ops.append(Op("dep.norm_params", "gemm", 0, m=1, k=norms, n=1, role="speech", w_params=norms, w_bits=16.0,
                  w_fmt="bf16", a_fmt="bf16"))
    return ops


def _seanet_rows(s: MoshiSpec, side: str) -> dict[int, tuple[int, int]]:
    """model index → (positions per frame of the module's GEMM rows, transposed?) for the SEANet encoder / decoder."""
    P = s.samples_per_frame
    out: dict[int, tuple[int, int]] = {}
    if side == "encoder":
        r, i = P, 0
        out[0] = (r, 0)
        i = 1
        for st in reversed(s.ratios):
            out[i] = (r, 0)              # residual block at the current rate
            r //= st
            out[i + 2] = (r, 0)          # strided conv → output positions
            i += 3
        out[i + 1] = (r, 0)              # final conv
        return out
    r = P // math.prod(s.ratios)
    out[0] = (r, 0)
    i = 2
    for st in s.ratios:
        out[i] = (r, 1)                  # ConvTranspose per input position
        r *= st
        out[i + 1] = (r, 0)
        i += 3
    out[i] = (r, 0)                      # final conv (index 14)
    return out


def _codec_side(s: MoshiSpec, side: str, b: int) -> list[Op]:
    rows = _seanet_rows(s, side)
    ops: list[Op] = []
    pre = f"mimi.{side}.model."
    for n, sh in s.tensors:
        if not n.startswith(pre) or not n.endswith(".weight"):
            continue
        idx = int(n[len(pre):].split(".")[0])
        r, tr = rows[idx]
        bias = math.prod(_shape(s, n[:-len("weight")] + "bias")) if any(m == n[:-len("weight")] + "bias" for m, _ in s.tensors) else 0
        if tr:
            ops.append(_g(n, b * r, sh[0], sh[1] * sh[2], 32, w=math.prod(sh) + bias))
        else:
            ops.append(_g(n, b * r, sh[1] * sh[2], sh[0], 32, w=math.prod(sh) + bias))
        ops.append(Op(n + ".elu", "vector", idx, vec=b * r * (sh[0] if not tr else sh[1]) * 4, sfu=b * r * sh[0]))
    # streaming transformer (2 positions per frame at 25 Hz over a 250-position KV cache)
    tp = f"mimi.{side}_transformer.transformer.layers."
    L = len({n[len(tp):].split(".")[0] for n, _ in s.tensors if n.startswith(tp)})
    q, d, hd = s.enc_rate, s.mimi_dim, s.mimi_dim // s.mimi_heads
    for li in range(L):
        p = f"{tp}{li}."
        f1 = _shape(s, p + "linear1.weight")
        side_p = sum(math.prod(sh) for n, sh in s.tensors if n.startswith(p) and len(sh) == 1)
        ops += [_g(p + "qkv", b * q, d, 3 * d, 32, w=3 * d * d + side_p), _g(p + "o", b * q, d, d, 32),
                _g(p + "ff1", b * q, d, f1[0], 32), _g(p + "ff2", b * q, f1[0], d, 32),
                Op(p + "norm_act", "vector", li, vec=b * q * (d * 10 + f1[0] * 8), sfu=b * q * f1[0])]
        ops += _attn(p + "attn", q, s.mimi_ctx, hd, s.mimi_heads, 32, b)
    return ops


def mimi_encode_ops(s: MoshiSpec, b: int) -> list[Op]:
    """User audio frame → 8 codes: SEANet encoder, transformer, stride-2 conv, split RVQ search (1 + 7 codebooks)."""
    ops = _codec_side(s, "encoder", b)
    dn = _shape(s, "mimi.downsample.conv.conv.conv.weight")
    ops.append(_g("mimi.downsample", b, dn[1] * dn[2], dn[0], 32))
    for grp in ("rvq_first", "rvq_rest"):
        ip = _shape(s, f"mimi.quantizer.{grp}.input_proj.weight")
        ops.append(_g(f"mimi.{grp}.in", b, ip[1], ip[0], 32))
    for c in range(s.n_codebooks):
        ops.append(_g(f"mimi.vq.{c}.search", b, s.q_dim, s.bins, 32, w=s.bins * (s.q_dim + 1)))   # codebook + usage
        ops.append(Op(f"mimi.vq.{c}.argmin", "vector", 0, vec=b * s.bins * 3))
    return ops


def mimi_decode_ops(s: MoshiSpec, b: int) -> list[Op]:
    """8 codes → 1920 samples: dequantize (lookup + output_proj ×2), depthwise ConvTranspose ×2, transformer, SEANet
    decoder."""
    ops = [Op("mimi.dequant", "lookup", 0, kv_read=b * s.n_codebooks * s.q_dim * 4.0, vec=b * s.n_codebooks * s.q_dim)]
    for grp in ("rvq_first", "rvq_rest"):
        op_ = _shape(s, f"mimi.quantizer.{grp}.output_proj.weight")
        ops.append(_g(f"mimi.{grp}.out", b, op_[1], op_[0], 32))
    up = _shape(s, "mimi.upsample.convtr.convtr.convtr.weight")      # [C, 1, k] groups = C
    ops.append(_g("mimi.upsample", b, 1, up[0] * up[2], 32))
    return ops + _codec_side(s, "decoder", b)


def depth_weight_bytes(s: MoshiSpec) -> float:
    return sum(o.w_params for o in depformer_ops(s, 1) if o.kind == "gemm") * 2.0


def mimi_weight_bytes(s: MoshiSpec) -> float:
    return sum(o.w_params * o.w_bits / 8 for o in mimi_encode_ops(s, 1) + mimi_decode_ops(s, 1) if o.kind == "gemm")
