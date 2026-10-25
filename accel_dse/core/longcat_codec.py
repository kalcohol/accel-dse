"""LongCat-Flash-Omni waveform decoder (0.403).

The released post-processor loads ``LongCatAudioCodecDecoder`` (``decoder_type``
``24k``) and calls ``forward``. Sources:

* class ``LongCatAudioCodecDecoder`` / ``MemDecoder`` / ``AutoGroupResidualVectorQuantize``
  in ``meituan-longcat/LongCat-Audio-Codec`` (``networks/semantic_codec/LongCatAudioCodec_model.py``,
  ``networks/decoders/mem_decoders.py``, ``networks/quantizers/auto_grvq.py``,
  ``networks/modules/causal_model.py``, ``networks/modules/modules.py``)
* Omni ``Codec2WavProcessor.codec2wav`` (semantic row 0, acoustic rows 1:4, then
  ``decoder(semantic - 32, acoustic - 32)``)
* checkpoint ``audio_codec/LongCatAudioCodec_decoder_24k_4codebooks_aug_sft.pt``
  plus ``audio_codec/config.yaml`` on both Omni repos (same ETag)

``load_decoder`` uses ``strict=True``. ``from_codes`` reads the codebooks and
``out_proj`` only. ``in_proj`` stays in the file and is not a decode GEMM.
The encoder checkpoint is not in the Omni root, so it is not billed.
Weights are fp32 as stored. The 24k path sets ``is_final_causal=True``,
``groups=1``, ``lstm_nums=1``, ``lookahead_frame=3``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Op

_LOOKAHEAD = 3
_FINAL_KERNEL = 7
_FP32_BYTES = 4
_VEC_PER_OUT = 3.0
# DAC Snake1d: x + 1/(alpha+eps) * sin(alpha*x)^2. Five element ops, one sin.
# Same count as the descript snake billed for the Fish DAC decoder.
_SNAKE_VEC = 5.0


@dataclass(frozen=True)
class LongcatCodecSpec:
    """Geometry of one released 24k decoder checkpoint."""

    latent: int
    semantic: int
    channels: int
    rates: tuple[int, ...]
    n_codebooks: int
    codebook_size: int
    codebook_dim: int
    semantic_tokens: int
    params: int
    params_decode: int
    params_in_proj: int
    params_lookup: int
    file_bytes: int
    n_tensors: int

    @property
    def width(self) -> int:
        return self.latent + self.semantic

    @property
    def hop(self) -> int:
        return math.prod(self.rates)

    @property
    def sample_rate(self) -> int:
        """``LongCatAudioCodecDecoder`` sets 24000 when ``decoder_type`` is ``24k``."""
        return 24000


def _need_int(meta: dict, key: str) -> int:
    v = meta.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"longcat codec field {key} is not a positive int")
    return v


def _wn_conv(out: dict, prefix: str, out_c: int, in_c: int, k: int) -> None:
    out[prefix + ".bias"] = (out_c,)
    out[prefix + ".weight_g"] = (out_c, 1, 1)
    out[prefix + ".weight_v"] = (out_c, in_c, k)


def _wn_tconv(out: dict, prefix: str, in_c: int, out_c: int, k: int) -> None:
    out[prefix + ".bias"] = (out_c,)
    out[prefix + ".weight_g"] = (in_c, 1, 1)
    out[prefix + ".weight_v"] = (in_c, out_c, k)


def _snake_tensor(out: dict, prefix: str, channels: int) -> None:
    out[prefix + ".alpha"] = (1, channels, 1)


def _residual(out: dict, prefix: str, dim: int) -> None:
    _snake_tensor(out, prefix + ".block.0", dim)
    _wn_conv(out, prefix + ".block.2", dim, dim, 7)
    _snake_tensor(out, prefix + ".block.3", dim)
    _wn_conv(out, prefix + ".block.4", dim, dim, 1)


def expected_tensors(spec: LongcatCodecSpec) -> dict[str, tuple[int, ...]]:
    """State-dict shapes of ``LongCatAudioCodecDecoder`` for this geometry.

    Names follow the modules. Weight-norm stores ``weight_g`` and ``weight_v``
    (legacy ``weight_norm``, the keys in the released pickle).
    """
    latent = spec.latent
    if latent % 2:
        raise ValueError("longcat codec latent width is not even")
    half = latent // 2
    width = spec.width
    if spec.channels % (2 ** len(spec.rates)):
        raise ValueError("longcat codec decoder width is not divisible by the rate product")
    out: dict[str, tuple[int, ...]] = {}
    gate = 4 * width
    for which in ("lstm_list1", "lstm_list2"):
        p = f"decoder.mem_layers.0.{which}.0"
        out[p + ".weight_ih_l0"] = (gate, width)
        out[p + ".weight_hh_l0"] = (gate, width)
        out[p + ".bias_ih_l0"] = (gate,)
        out[p + ".bias_hh_l0"] = (gate,)
    for ln in ("ln1", "ln2"):
        out[f"decoder.mem_layers.0.{ln}.weight"] = (width,)
        out[f"decoder.mem_layers.0.{ln}.bias"] = (width,)
    for i in range(spec.n_codebooks):
        p = f"acoustic_quantizer.quantizers.{i}"
        for side in ("a", "b"):
            _wn_conv(out, f"{p}.in_proj_{side}", spec.codebook_dim, latent, 1)
            _wn_conv(out, f"{p}.out_proj_{side}", half, spec.codebook_dim, 1)
            out[f"{p}.codebook_{side}.weight"] = (spec.codebook_size, spec.codebook_dim)
    kernel = 2 * _LOOKAHEAD + 1
    _wn_conv(out, "decoder.model.0", spec.channels, width, kernel)
    ch = spec.channels
    for i, stride in enumerate(spec.rates):
        in_c = ch // 2 ** i
        out_c = ch // 2 ** (i + 1)
        p = f"decoder.model.{i + 1}"
        _snake_tensor(out, p + ".block.0", in_c)
        _wn_tconv(out, p + ".block.1", in_c, out_c, stride)
        for j in range(3):
            _residual(out, f"{p}.block.{j + 2}", out_c)
    last = ch // 2 ** len(spec.rates)
    tail = 1 + len(spec.rates)
    _snake_tensor(out, f"decoder.model.{tail}", last)
    # ZeroPad1d is model.{tail+1} and has no parameters. The final conv is the next module.
    _wn_conv(out, f"decoder.model.{tail + 2}", 1, last, _FINAL_KERNEL)
    out["semantic_dequantizer.dequant.weight"] = (spec.semantic_tokens, spec.semantic)
    return out


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _split_params(tensors: dict[str, tuple[int, ...]]) -> tuple[int, int, int, int]:
    total = sum(_numel(s) for s in tensors.values())
    in_proj = sum(_numel(s) for n, s in tensors.items() if ".in_proj_" in n)
    lookup = sum(_numel(s) for n, s in tensors.items() if n.endswith("codebook_a.weight")
                 or n.endswith("codebook_b.weight") or n.endswith("dequant.weight"))
    return total, total - in_proj, in_proj, lookup


def codec_spec(meta: dict) -> LongcatCodecSpec:
    """Build a spec when the stored header matches the class formula."""
    if not isinstance(meta, dict):
        raise ValueError("longcat codec header is missing")
    if meta.get("decoder_type") != "24k":
        raise ValueError("longcat codec decoder_type is not 24k")
    if meta.get("dtype") != "F32":
        raise ValueError("longcat codec dtype is not F32")
    rates = meta.get("codec_dec_ratios")
    if not isinstance(rates, list) or not rates or any(isinstance(s, bool) or not isinstance(s, int) or s <= 0
                                                      for s in rates):
        raise ValueError("longcat codec codec_dec_ratios is not a list of positive ints")
    spec = LongcatCodecSpec(
        _need_int(meta, "codec_dimension"),
        _need_int(meta, "semantic_dim"),
        _need_int(meta, "decoder_dim"),
        tuple(rates),
        _need_int(meta, "codec_codebooks"),
        _need_int(meta, "codec_codebook_size"),
        _need_int(meta, "codec_codebook_search_dim"),
        _need_int(meta, "semantic_token_nums"),
        _need_int(meta, "params"),
        _need_int(meta, "params_decode"),
        _need_int(meta, "params_in_proj"),
        _need_int(meta, "params_lookup"),
        _need_int(meta, "file_bytes"),
        _need_int(meta, "n_tensors"),
    )
    expect = expected_tensors(spec)
    got = meta.get("tensors")
    if not isinstance(got, dict):
        raise ValueError("longcat codec tensors are missing")
    if len(got) != len(expect) or set(got) != set(expect):
        raise ValueError("longcat codec tensor names do not match the decoder class")
    for name, shape in expect.items():
        raw = got[name]
        if list(raw) != list(shape):
            raise ValueError(f"longcat codec tensor {name} shape does not match the class")
    total, decode, idle, lookup = _split_params(expect)
    if spec.n_tensors != len(expect):
        raise ValueError("longcat codec n_tensors does not match the class")
    if (spec.params, spec.params_decode, spec.params_in_proj, spec.params_lookup) != (total, decode, idle, lookup):
        raise ValueError("longcat codec param totals do not match the class")
    if spec.file_bytes < spec.params * _FP32_BYTES:
        raise ValueError("longcat codec file is smaller than the fp32 weights")
    return spec


def _gemm(name: str, m: int, k: int, n: int, w: int, in_elems: float = 0.0) -> Op:
    return Op(name, "gemm", 0, m=m, k=k, n=n, role="codec", w_params=w, w_bits=32.0, w_fmt="fp32",
              a_fmt="fp32", act_bytes=m * (k + n) * _FP32_BYTES, vec=m * n * _VEC_PER_OUT, stream=True,
              in_elems=in_elems)


def _conv_w(out_c: int, in_c: int, k: int) -> int:
    return out_c * in_c * k + out_c + out_c


def _tconv_w(in_c: int, out_c: int, k: int) -> int:
    return in_c * out_c * k + in_c + out_c


def codec_ops(spec: LongcatCodecSpec, frames: int, batch: int = 1) -> list[Op]:
    """One ``decoder.forward`` of ``frames`` codec frames.

    ``Codec2WavProcessor`` passes the rows that remain after ``shift_codec_back``.
    That shift drops the last row and realigns acoustic codes. It does not add a GEMM.
    ``frames`` is the length of the tensor the decoder receives.

    Decode reads semantic and acoustic codebooks, ``out_proj``, the GLSTM, and the
    causal upsampler. ``in_proj`` is not in this list.
    """
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
        raise ValueError("longcat codec frames is not a non-negative int")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("longcat codec batch is not a positive int")
    if frames == 0:
        return []
    b = batch
    t = frames
    h = spec.width
    gate = 4 * h
    half = spec.latent // 2
    ops: list[Op] = [
        Op("codec.semantic", "lookup", 0, kv_read=b * t * spec.semantic * _FP32_BYTES),
    ]
    for i in range(spec.n_codebooks):
        row = b * t * spec.codebook_dim * _FP32_BYTES
        ops.append(Op(f"codec.q{i}.a", "lookup", 0, kv_read=row))
        ops.append(Op(f"codec.q{i}.b", "lookup", 0, kv_read=row))
        w = _conv_w(half, spec.codebook_dim, 1)
        ops.append(_gemm(f"codec.q{i}.out_a", b * t, spec.codebook_dim, half, w, in_elems=b * t * spec.codebook_dim))
        ops.append(_gemm(f"codec.q{i}.out_b", b * t, spec.codebook_dim, half, w, in_elems=b * t * spec.codebook_dim))
    # Input projection is one GEMM over the frames. The recurrent projection is one
    # GEMM per frame; the MAC count is the same as one GEMM with m = frames.
    lstm_w = gate * h + gate
    for name in ("codec.lstm1.ih", "codec.lstm1.hh", "codec.lstm2.ih", "codec.lstm2.hh"):
        ops.append(_gemm(name, b * t, h, gate, lstm_w, in_elems=b * t * h))
    # PyTorch LSTM cell per hidden unit: sigmoid, sigmoid, tanh, sigmoid, tanh(c),
    # then f*c, i*g, their sum, and o*tanh(c). Two LSTMs.
    ops.append(Op("codec.lstm.act", "vector", 0, vec=b * t * h * 9.0 * 2.0, sfu=b * t * h * 5.0 * 2.0,
                  a_fmt="fp32"))
    # LayerNorm affine is a scale and a bias per width, read once. The element work is separate.
    ln = 4 * h
    ops.append(_gemm("codec.ln", 1, ln, 1, ln))
    ops.append(Op("codec.ln.act", "vector", 0, vec=b * t * h * 4.0 * 2.0, sfu=float(b * t * 2), a_fmt="fp32"))
    kernel = 2 * _LOOKAHEAD + 1
    ops.append(_gemm("codec.dec.conv0", b * t, h * kernel, spec.channels, _conv_w(spec.channels, h, kernel),
                     in_elems=b * t * h))
    length = t
    ch = spec.channels
    side = 0
    for i, stride in enumerate(spec.rates):
        in_c = ch // 2 ** i
        out_c = ch // 2 ** (i + 1)
        side += in_c
        ops.append(Op(f"codec.up{i}.snake", "vector", 0, vec=b * length * in_c * _SNAKE_VEC,
                      sfu=float(b * length * in_c), a_fmt="fp32"))
        ops.append(_gemm(f"codec.up{i}.trans", b * length, in_c, out_c * stride,
                         _tconv_w(in_c, out_c, stride), in_elems=b * length * in_c))
        length *= stride
        for j in range(3):
            side += 2 * out_c
            ops.append(Op(f"codec.up{i}.res{j}.snake0", "vector", 0, vec=b * length * out_c * _SNAKE_VEC,
                          sfu=float(b * length * out_c), a_fmt="fp32"))
            ops.append(_gemm(f"codec.up{i}.res{j}.conv7", b * length, out_c * 7, out_c,
                             _conv_w(out_c, out_c, 7), in_elems=b * length * out_c))
            ops.append(Op(f"codec.up{i}.res{j}.snake1", "vector", 0, vec=b * length * out_c * _SNAKE_VEC,
                          sfu=float(b * length * out_c), a_fmt="fp32"))
            ops.append(_gemm(f"codec.up{i}.res{j}.conv1", b * length, out_c, out_c,
                             _conv_w(out_c, out_c, 1), in_elems=b * length * out_c))
    last = ch // 2 ** len(spec.rates)
    if length != t * spec.hop or last <= 0:
        raise ValueError("longcat codec upsample length does not match the rate product")
    side += last
    ops.append(Op("codec.out.snake", "vector", 0, vec=b * length * last * _SNAKE_VEC,
                  sfu=float(b * length * last), a_fmt="fp32"))
    ops.append(_gemm("codec.out.conv", b * length, last * _FINAL_KERNEL, 1, _conv_w(1, last, _FINAL_KERNEL),
                     in_elems=b * length * last))
    # Snake alpha parameters. Counted once per forward, not as extra MACs.
    ops.append(_gemm("codec.snake.alpha", 1, side, 1, side))
    gemm_w = sum(op.w_params for op in ops if op.kind == "gemm")
    if gemm_w + spec.params_lookup != spec.params_decode:
        raise ValueError("longcat codec decode GEMMs do not cover the decode weights")
    return ops
