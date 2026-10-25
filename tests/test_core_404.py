"""0.404: adversarial checks on the released 24k LongCatAudioCodecDecoder.

The class (meituan-longcat/LongCat-Audio-Codec) is a causal MemDecoder:
4 codebooks are 1 semantic embedding plus 3 acoustic quantizers, each with
codebook_a and codebook_b. Upsample rates [8, 6, 6, 5] emit 1440 samples
per frame at 24 kHz. Snake is DAC sin^2 (alpha only). in_proj is stored and
not read by from_codes. The decoder stays out of params(). The 24k Sequential
ends with nn.Tanh, one per sample.
"""
from __future__ import annotations

import copy
import math

from accel_dse.core.catalog import get_model
from accel_dse.core.longcat_codec import codec_ops, codec_spec
from accel_dse.core.model import load_release

PARAMS = 560_664_980_480
AUDIO_IO = 555_810_816
PARAMS_DECODE = 150_852_930
PARAMS_IN_PROJ = 49_248
PARAMS_LOOKUP = 10_490_080
HOP = 1440
WIDTH = 1024 + 1280
RATES = (8, 6, 6, 5)


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc)
    else:
        raise AssertionError(f"accepted a release that should mention {needle}")


def _snake_elements(channels: int, rates: tuple[int, ...], frames: int) -> int:
    """Snake inputs of CausalDecoderBlock plus the final Snake1d.

    Each block is Snake, transposed conv, then three ResidualUnits. A residual
    is Snake, conv7, Snake, conv1. Dilation does not add a Snake.
    """
    total = 0
    length = frames
    ch = channels
    for _stride in rates:
        out_c = ch // 2
        total += length * ch
        length *= _stride
        total += 3 * 2 * length * out_c
        ch = out_c
    total += length * ch
    return total


def test_longcat_codec_decode_matches_the_released_24k_class():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    rel = load_release(m.hf_id)
    spec = m.longcat_codec
    tensors = rel["longcat_codec"]["tensors"]
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    assert m.params() == fp.params() == PARAMS + AUDIO_IO
    assert spec.params not in (m.params(), fp.params())
    assert spec.hop == math.prod(RATES) == HOP
    assert spec.sample_rate == 24000
    assert spec.sample_rate // 1000 * 60 == HOP
    assert spec.n_codebooks == 3 and spec.width == WIDTH
    assert spec.rates == RATES
    assert "nn.Tanh" in " ".join(m.notes)
    assert "不进 params()" in " ".join(m.notes)
    names = set(tensors)
    assert not any("beta" in name for name in names)
    assert not any(name.endswith("lstm_list1.1.weight_ih_l0") for name in names)
    ih = tensors["decoder.mem_layers.0.lstm_list1.0.weight_ih_l0"]
    assert ih == [4 * WIDTH, WIDTH]
    in_proj = sum(math.prod(shape) for name, shape in tensors.items() if ".in_proj_" in name)
    assert in_proj == PARAMS_IN_PROJ == spec.params_in_proj
    for rate, index in zip(RATES, (1, 2, 3, 4)):
        weight = tensors[f"decoder.model.{index}.block.1.weight_v"]
        assert weight[2] == rate and weight[2] != 2 * rate
    ops = codec_ops(spec, 1, 1)
    assert not any("in_proj" in op.name for op in ops)
    assert sum(op.w_params for op in ops if op.kind == "gemm") + PARAMS_LOOKUP == PARAMS_DECODE
    assert all(op.w_fmt == "fp32" and op.a_fmt == "fp32" for op in ops if op.kind == "gemm")
    lookups = [op for op in ops if op.kind == "lookup"]
    assert [op.name for op in lookups] == [
        "codec.semantic",
        "codec.q0.a", "codec.q0.b", "codec.q1.a", "codec.q1.b", "codec.q2.a", "codec.q2.b",
    ]
    assert lookups[0].kv_read == spec.semantic * 4
    assert all(op.kv_read == spec.codebook_dim * 4 for op in lookups[1:])
    snakes = [op for op in ops if op.kind == "vector" and "snake" in op.name]
    assert snakes and all(op.vec == op.sfu * 5 and op.a_fmt == "fp32" for op in snakes)
    assert sum(op.sfu for op in snakes) == _snake_elements(spec.channels, spec.rates, 1)
    convs = [op for op in ops if op.name.endswith(".conv7")]
    assert len(convs) == 4 * 3
    for i, rate in enumerate(RATES):
        out_c = spec.channels // 2 ** (i + 1)
        stage = [op for op in convs if op.name.startswith(f"codec.up{i}.")]
        assert len(stage) == 3 and all(op.k == out_c * 7 for op in stage)
        trans = next(op for op in ops if op.name == f"codec.up{i}.trans")
        assert trans.n == out_c * rate
    out_names = [op.name for op in ops]
    assert out_names.index("codec.out.conv") + 1 == out_names.index("codec.out.tanh")
    tanh = next(op for op in ops if op.name == "codec.out.tanh")
    assert tanh.kind == "vector" and tanh.w_params == 0 and tanh.a_fmt == "fp32"
    assert tanh.vec == tanh.sfu == HOP
    wide = codec_ops(spec, 3, 2)
    wide_tanh = next(op for op in wide if op.name == "codec.out.tanh")
    assert wide_tanh.vec == wide_tanh.sfu == 2 * 3 * HOP
    assert codec_ops(spec, 0, 1) == []
    lstm = next(op for op in ops if op.name == "codec.lstm.act")
    assert lstm.sfu == WIDTH * 5 * 2 and lstm.vec == WIDTH * 9 * 2


def test_longcat_codec_rejects_a_non_causal_kernel_and_a_snake_beta():
    m = get_model("longcat-flash-omni")
    rel = load_release(m.hf_id)
    kernel = copy.deepcopy(rel["longcat_codec"])
    name = "decoder.model.1.block.1.weight_v"
    kernel["tensors"] = dict(kernel["tensors"])
    kernel["tensors"][name] = [1536, 768, 2 * RATES[0]]
    _raises(lambda: codec_spec(kernel), name)
    beta = copy.deepcopy(rel["longcat_codec"])
    beta["tensors"] = dict(beta["tensors"])
    beta["tensors"]["decoder.model.5.beta"] = [1, 96, 1]
    beta["n_tensors"] = len(beta["tensors"])
    _raises(lambda: codec_spec(beta), "tensor names")
