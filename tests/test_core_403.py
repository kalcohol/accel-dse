"""0.403: LongCat-Flash-Omni waveform decoder matches the released 24k class.

The decoder pickle is fp32. Decode runs codebooks, out_proj, the GLSTM, and the
causal upsampler. in_proj is stored and not executed. Language rel_err stays 0.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model
from accel_dse.core.longcat_codec import codec_ops, codec_spec, expected_tensors
from accel_dse.core.model import from_release, load_release

PARAMS = 560_664_980_480
AUDIO_IO = 555_810_816
PARAMS_CODEC = 150_902_178
PARAMS_DECODE = 150_852_930
PARAMS_IN_PROJ = 49_248
PARAMS_LOOKUP = 10_490_080
FILE_BYTES = 603_637_648
N_TENSORS = 174
HOP = 1440
WIDTH = 1024 + 1280


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc)
    else:
        raise AssertionError(f"accepted a release that should mention {needle}")


def test_longcat_codec_matches_the_24k_class_and_stays_out_of_params():
    m = get_model("longcat-flash-omni")
    fp = get_model("longcat-flash-omni-fp8")
    rel = load_release(m.hf_id)
    frel = load_release(fp.hf_id)
    spec = m.longcat_codec
    assert m.param_check()["rel_err"] == 0.0 and fp.param_check()["rel_err"] == 0.0
    assert m.params() == fp.params() == PARAMS + AUDIO_IO
    assert spec is not None and fp.longcat_codec == spec
    assert spec.params == PARAMS_CODEC and spec.params_decode == PARAMS_DECODE
    assert spec.params_in_proj == PARAMS_IN_PROJ and spec.params_lookup == PARAMS_LOOKUP
    assert spec.file_bytes == FILE_BYTES and spec.n_tensors == N_TENSORS
    assert spec.hop == HOP and spec.sample_rate == 24000 and spec.width == WIDTH
    assert spec.rates == (8, 6, 6, 5) and spec.n_codebooks == 3
    assert rel["longcat_codec"] == frel["longcat_codec"]
    assert rel["longcat_codec"]["dtype"] == "F32"
    assert expected_tensors(spec) == {n: tuple(s) for n, s in rel["longcat_codec"]["tensors"].items()}
    assert "LongCat-Audio-Codec" in " ".join(m.notes)
    assert "不进 params()" in " ".join(m.notes)
    assert "编码器" in " ".join(m.notes) and "不计入" in " ".join(m.notes)
    ops = codec_ops(spec, 1, 1)
    assert sum(op.w_params for op in ops if op.kind == "gemm") + PARAMS_LOOKUP == PARAMS_DECODE
    assert not any("in_proj" in op.name for op in ops)
    assert all(op.w_fmt == "fp32" and op.a_fmt == "fp32" for op in ops if op.kind == "gemm")
    assert sum(op.kind == "gemm" for op in ops) == 42
    sem = next(op for op in ops if op.name == "codec.semantic")
    assert sem.kv_read == 1280 * 4
    books = [op for op in ops if op.name.startswith("codec.q") and op.kind == "lookup"]
    assert len(books) == 6 and all(op.kv_read == 8 * 4 for op in books)
    conv0 = next(op for op in ops if op.name == "codec.dec.conv0")
    assert (conv0.m, conv0.k, conv0.n) == (1, WIDTH * 7, 1536)
    trans = next(op for op in ops if op.name == "codec.up0.trans")
    assert (trans.m, trans.k, trans.n) == (1, 1536, 768 * 8)
    final = next(op for op in ops if op.name == "codec.out.conv")
    assert (final.m, final.k, final.n, final.w_fmt) == (HOP, 96 * 7, 1, "fp32")
    snake = next(op for op in ops if op.name == "codec.out.snake")
    assert snake.sfu == HOP * 96 and snake.vec == HOP * 96 * 5
    assert next(op for op in codec_ops(spec, 2, 1) if op.name == "codec.out.conv").m == 2 * HOP
    assert next(op for op in codec_ops(spec, 1, 2) if op.name == "codec.out.conv").m == 2 * HOP
    assert codec_ops(spec, 0, 1) == []
    assert fp.fmt("expert").fmt == "fp8" and fp.fmt("attn").fmt == "bf16"
    assert m.fmt("audio_head").fmt == "bf16"


def test_longcat_codec_rejects_a_header_that_is_not_the_class():
    m = get_model("longcat-flash-omni")
    rel = load_release(m.hf_id)
    missing = copy.deepcopy(rel)
    missing.pop("longcat_codec")
    _raises(lambda: from_release(m.id, m.hf_id, missing, missing["config"]), "codec header is missing")
    bad = copy.deepcopy(rel)
    bad["longcat_codec"] = copy.deepcopy(rel["longcat_codec"])
    bad["longcat_codec"]["params"] += 1
    _raises(lambda: from_release(m.id, m.hf_id, bad, bad["config"]), "param totals")
    shape = copy.deepcopy(rel)
    shape["longcat_codec"] = copy.deepcopy(rel["longcat_codec"])
    shape["longcat_codec"]["tensors"] = dict(rel["longcat_codec"]["tensors"])
    shape["longcat_codec"]["tensors"]["semantic_dequantizer.dequant.weight"] = [8192, 1281]
    _raises(lambda: from_release(m.id, m.hf_id, shape, shape["config"]), "semantic_dequantizer.dequant.weight")
    kind = copy.deepcopy(rel)
    kind["longcat_codec"] = dict(rel["longcat_codec"])
    kind["longcat_codec"]["decoder_type"] = "16k"
    _raises(lambda: codec_spec(kind["longcat_codec"]), "not 24k")
    dtype = copy.deepcopy(rel)
    dtype["longcat_codec"] = dict(rel["longcat_codec"])
    dtype["longcat_codec"]["dtype"] = "BF16"
    _raises(lambda: codec_spec(dtype["longcat_codec"]), "not F32")
    _raises(lambda: codec_ops(m.longcat_codec, -1, 1), "frames")
    _raises(lambda: codec_ops(m.longcat_codec, 1, 0), "batch")
