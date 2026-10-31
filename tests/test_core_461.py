"""0.461: Whisper and Distil-Whisper decoders bill published erf-GELU and LayerNorm.

ACT2FN["gelu"] is x * 0.5 * (1 + erf(x / sqrt(2))). It sits between fc1 and
fc2. The four LayerNorms reduce in fp32. The MLP is not gated, so there is
no product. The encoder stays on the vision graph. These 17 ids sit outside
the first 60 catalog rows. Vector flops stay 0. The generic .ew tail stays.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _tag_whisper_decoder_vectors, _tc, from_release, load_release
from accel_dse.core.vision import audio_ops


_ROWS = (
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
    "whisper-large-v2",
    "whisper-tiny.en",
    "whisper-base.en",
    "whisper-small.en",
    "whisper-medium.en",
    "distil-whisper-small.en",
    "distil-whisper-medium.en",
    "distil-whisper-large-v2",
    "distil-whisper-large-v3",
    "distil-whisper-large-v3.5",
)
_BANNED = ("silu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul", "mlp_mul")
_WIDE = ("whisper-large-v3", "whisper-large", "whisper-large-v2")


def _by(ops):
    found = {}
    for op in ops:
        if op.name in found and op.kind == "vector" and not op.name.endswith(".ew"):
            raise AssertionError(op.name)
        found[op.name] = op
    return found


def _gelu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == "gelu"
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _ln(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bare(model):
    return replace(
        model,
        layers=tuple(replace(L, vec_kind="") for L in model.layers),
        vec_kind="")


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def _bad(mid: str, edit):
    m = get_model(mid)
    rel = json.loads(json.dumps(load_release(m.hf_id)))
    edit(rel["config"])
    return lambda: from_release(mid, m.hf_id, rel)


def test_version_is_461():
    assert __version__ == "0.482.0"


def test_whisper_decoder_bills_erf_gelu_between_the_linears():
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    for mid in _ROWS:
        m = get_model(mid)
        assert ids.index(mid) >= 60, mid
        assert m.vec_kind == "whisper" and all(L.vec_kind == "whisper" for L in m.layers)
        assert m.act_fmt == "bf16" and m.enc_ctx == 1500
        assert m.final_norm_params == 2 * m.hidden
        assert m.param_check()["rel_err"] == 0.0
        assert any("0.461" in n for n in m.notes)
        assert "按 SiLU" not in " ".join(m.notes)
        L = m.layers[0]
        assert L.ffn.gated is False and [ln.name for ln in L.ffn_linears] == ["up", "down"]
        h, ff, depth = m.hidden, L.ffn.d_ff, m.n_layers
        wfmt = m.fmt("mlp").fmt
        assert wfmt in ("fp16", "fp32") and m.fmt("attn").fmt == wfmt
        ph = Phase("prefill", 1, 4096, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        op = _by(ops)
        _gelu(op["gelu"], 4096 * ff, 0, "bf16")
        _ln(op["attn_norm"], 4096, h, 0)
        _ln(op["cross_norm"], 4096, h, 0)
        _ln(op["ffn_norm"], 4096, h, 0)
        assert op["mlp.up"].m == 4096 and op["mlp.up"].k == h and op["mlp.up"].n == ff
        assert op["mlp.up"].w_fmt == wfmt and op["mlp.up"].a_fmt == "bf16"
        assert op["mlp.down"].m == 4096 and op["mlp.down"].k == ff and op["mlp.down"].n == h
        assert op["mlp.down"].w_fmt == wfmt and op["mlp.down"].flops == 2 * 4096 * ff * h
        names = [o.name for o in ops]
        assert names.index("attn_norm") < names.index("attn.qkv") < names.index("qk")
        assert names.index("rope") < names.index("cross_norm") < names.index("cross.q")
        assert names.index("ffn_norm") < names.index("mlp.up") < names.index("gelu")
        assert names.index("gelu") < names.index("mlp.down") < names.index("residual")
        assert "act" not in names
        for banned in _BANNED:
            assert banned not in names
        head = _by(head_ops(m, ph, Shard()))
        _ln(head["final_norm"], 4096, h, depth)
        total = sum(o.vec for li in range(depth)
                    for o in build_rank_ops(m, li, ph, Shard()) if o.name == "gelu")
        assert total == 4096 * ff * 5 * depth
        enc = _by(audio_ops(m.vision.audio, 1.0, 1, "bf16"))
        assert enc["aud.0.gelu"].a_fmt == m.vision.audio.vec_fmt
        assert enc["aud.0.attn_ln"].a_fmt == "fp32"
        assert "gelu" not in {o.name for o in audio_ops(m.vision.audio, 1.0, 1, "bf16")
                              if not o.name.startswith("aud.")}


def test_large_v3_shapes_tp_fp8_batch_and_decode():
    m = get_model("whisper-large-v3")
    assert m.n_layers == 32 and m.hidden == 1280 and m.layers[0].ffn.d_ff == 5120
    assert m.fmt("mlp").fmt == "fp16" and m.params() == m.release_params == 1_543_490_560
    ph = Phase("prefill", 1, 4096, 0)
    op = _by(build_rank_ops(m, 0, ph, Shard()))
    _gelu(op["gelu"], 4096 * 5120, 0, "bf16")
    assert op["attn.qkv"].w_fmt == "fp16" and op["cross.q"].w_fmt == "fp16"
    assert "cross.k" in op and "cross.v" in op
    two = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _gelu(two["gelu"], 2 * 4096 * 5120, 0, "bf16")
    _ln(two["cross_norm"], 2 * 4096, 1280, 0)
    short = _by(build_rank_ops(m, 0, Phase("prefill", 1, 128, 0), Shard()))
    _gelu(short["gelu"], 128 * 5120, 0, "bf16")
    assert short["mlp.up"].m == 128 and short["mlp.down"].n == 1280
    tp = _by(build_rank_ops(m, 0, ph, Shard(tp=2)))
    _gelu(tp["gelu"], 4096 * 2560, 0, "bf16")
    assert tp["mlp.up"].n == 2560 and tp["mlp.down"].k == 2560
    assert tp["attn_norm"].vec == 4096 * 1280 * 4 and tp["cross_norm"].vec == 4096 * 1280 * 4
    tnames = [o.name for o in build_rank_ops(m, 0, ph, Shard(tp=2))]
    assert tnames.index("mlp.down") < tnames.index("mlp_allreduce")
    assert tnames.index("cross.o") < tnames.index("cross_allreduce") < tnames.index("ffn_norm")
    fp8 = replace(m, act_fmt="fp8")
    fop = _by(build_rank_ops(fp8, 0, ph, Shard()))
    _gelu(fop["gelu"], 4096 * 5120, 0, "fp8")
    assert fop["attn_norm"].a_fmt == "fp32" and fop["cross_norm"].a_fmt == "fp32"
    assert fop["ffn_norm"].a_fmt == "fp32" and fop["mlp.up"].a_fmt == "fp8"
    assert fop["mlp.up"].w_fmt == "fp16" and fop["mlp.down"].w_fmt == "fp16"
    fhead = _by(head_ops(fp8, ph, Shard()))
    assert fhead["final_norm"].a_fmt == "fp32" and fhead["final_norm"].sfu == 4096.0
    dec = _by(build_rank_ops(m, 0, Phase("decode", 4, 1, 60), Shard()))
    _gelu(dec["gelu"], 4 * 5120, 0, "bf16")
    _ln(dec["cross_norm"], 4, 1280, 0)
    assert "cross.q" in dec and "cross.k" not in dec and "cross.v" not in dec
    tiny = get_model("whisper-tiny")
    top = _by(build_rank_ops(tiny, 0, ph, Shard()))
    assert tiny.fmt("mlp").fmt == "fp32" and tiny.n_layers == 4
    _gelu(top["gelu"], 4096 * 1536, 0, "bf16")
    assert top["mlp.up"].w_fmt == "fp32"
    dist = get_model("distil-whisper-large-v3")
    assert dist.n_layers == 2 and dist.fmt("mlp").fmt == "fp16"
    dhead = _by(head_ops(dist, ph, Shard()))
    assert dhead["final_norm"].layer == 2
    for mid in _WIDE:
        wide = get_model(mid)
        assert wide.n_layers == 32 and wide.layers[0].ffn.d_ff == 5120
    _raises(lambda: build_rank_ops(m, 0, Phase("full", 1, 8, 0), Shard()), "autoregressive")


def test_refusals_and_other_models():
    m = get_model("whisper-large-v3")
    c = dict(_tc(load_release(m.hf_id)["config"]))
    c["activation_function"] = "silu"
    _raises(lambda: _tag_whisper_decoder_vectors(m, c), "activation is not gelu")
    _raises(_bad("whisper-tiny", lambda c: c.__setitem__("dropout", 0.1)), "dropout is not 0")
    _raises(_bad("distil-whisper-large-v3", lambda c: c.__setitem__("scale_embedding", True)),
            "scale_embedding")
    _raises(_bad("whisper-base", lambda c: c.__setitem__("architectures", ["WhisperModel"])),
            "architecture is not whisper")
    m = get_model("whisper-small")
    c = _tc(load_release(m.hf_id)["config"])
    gated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=True))
    bad = replace(m, layers=(gated,) + m.layers[1:])
    _raises(lambda: _tag_whisper_decoder_vectors(bad, c), "not dense gelu")
    renamed = tuple(replace(ln, name="gate_up") if ln.name == "up" else ln
                    for ln in m.layers[0].ffn_linears)
    bad_names = replace(m, layers=(replace(m.layers[0], ffn_linears=renamed),) + m.layers[1:])
    _raises(lambda: _tag_whisper_decoder_vectors(bad_names, c), "not up then down")
    other = get_model("qwen2-audio-7b")
    assert other.vec_kind == "qwen2_text"
    assert not any("0.461" in n for n in other.notes)
    for mid in ("qwen3-8b", "qwen3-32b", "qwen3-1.7b", "llama-3.1-8b", "esm2-3b",
                "ernie-4.5-0.3b-pt", "moshiko-7b", "openfold"):
        assert get_model(mid).vec_kind != "whisper"
        assert not any("0.461" in n for n in get_model(mid).notes)


def test_non_vector_schedule_fingerprints_and_kpis():
    ph = Phase("prefill", 1, 256, 0)
    for mid in ("whisper-large-v3", "whisper-tiny", "distil-whisper-large-v3.5"):
        m = get_model(mid)
        cleared = _bare(m)
        for shard in (Shard(), Shard(tp=2), Shard(tp=4)):
            assert _sched(m, ph, shard) == _sched(cleared, ph, shard)
        bare = build_rank_ops(cleared, 0, ph, Shard())
        assert all(o.name != "gelu" for o in bare)
        assert any(o.name == "act" for o in bare)
    ids = [e["id"] for e in entries()]
    assert "whisper-large-v3" not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "6381be58dbab8f57"
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario

    def fpx(top):
        h = hashlib.sha256()
        for e in entries()[:60]:
            if e["id"].startswith("deepseek-v4"):
                continue
            for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}},
                          {"layout": {"tp": 2}}):
                try:
                    h.update(json.dumps(evaluate(Scenario.from_dict({"model": e["id"], **top, **extra})).summary(),
                                        sort_keys=True, default=str).encode())
                except Exception as ex:  # noqa: BLE001
                    h.update(type(ex).__name__.encode())
        return h.hexdigest()[:16]

    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "98a7a0911ec0eefc"
    assert "whisper-large-v3" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
