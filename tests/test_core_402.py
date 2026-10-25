"""0.402: LongCat-Flash-Omni bills the published text stack plus the tower headers.

The text class is LongcatFlashForCausalLM without model.mtp.*. Vision is the
rope2d SwiGLU tower and the width-pair projector. Audio is the FSMN encoder.
Codebook embeddings are lookups. The four output linears run on sampled rows.
Omni-FP8 keeps blocks 23-27 in bf16 and has no MTP. Towers stay bf16.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, embed_ops, head_ops
from accel_dse.core.model import from_release, load_release
from accel_dse.core.vision import audio_ops, image_grid, vision_ops

H = 6144
V = 128
PARAMS = 560_664_980_480
AUDIO_IO = 555_810_816
TOWERS = 1_338_069_380
CODE_VOCAB = 8224
N_IN = 4
N_RES = 3


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc)
    else:
        raise AssertionError(f"accepted a release that should mention {needle}")


def _lang(m) -> int:
    return m.params() - m.lookup_params - sum(lin.params for lin in m.io_post)


def test_longcat_omni_matches_headers():
    chat = get_model("longcat-flash-chat")
    m = get_model("longcat-flash-omni")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == PARAMS + AUDIO_IO
    assert _lang(m) == chat.params() == rel["params_llm"] == PARAMS
    assert m.mtp_params() == rel["params_mtp"] == 0 and m.mtp_layers == ()
    assert m.lookup_params == (N_IN + N_RES) * CODE_VOCAB * H
    assert m.lookup_bits == 16.0 and m.audio_codebooks == (N_IN, N_RES, CODE_VOCAB)
    assert len(m.io_post) == N_IN
    assert all(lin.k == H and lin.n == CODE_VOCAB and lin.role == "audio_head" for lin in m.io_post)
    assert sum(lin.params for lin in m.io_post) == N_IN * H * CODE_VOCAB
    assert m.vision_params == rel["params_vision"] == TOWERS
    assert rel["params_audio_io"] == AUDIO_IO
    assert rel["n_tensors"] == 43739 and rel["bytes_mtp"] == 0
    assert rel["bytes_llm"] == rel["bytes_total"] == 1_121_594_245_120
    assert m.coverage == "full" and m.coverage_reasons == ()
    assert len(m.layers) == 56 and m.layers[0].core.v_dim == V
    assert m.fmt("audio_head").fmt == "bf16" and m.act_fmt_for("audio_head") == "bf16"
    ph = Phase("decode", 1, 1, 64)
    assert [(op.name, op.k, op.n, op.w_fmt) for op in build_rank_ops(m, 0, ph, Shard())
            if op.kind == "gemm"] == [
        (op.name, op.k, op.n, op.w_fmt) for op in build_rank_ops(chat, 0, ph, Shard())
        if op.kind == "gemm"]
    assert next(op for op in build_rank_ops(m, 0, ph, Shard()) if op.name == "pv").n == V
    pre = embed_ops(m, Phase("prefill", 1, 16, 0), Shard())
    dec = embed_ops(m, ph, Shard())
    assert not any(op.name == "audio_code_lookup" for op in pre)
    look = next(op for op in dec if op.name == "audio_code_lookup")
    assert look.kv_read == N_IN * H * 2
    hd = head_ops(m, ph, Shard())
    assert any(op.name == "audio_residual_lookup" and op.kv_read == N_RES * H * 2 for op in hd)
    assert [op.n for op in hd if op.name.startswith("io.audio_head")] == [CODE_VOCAB] * N_IN
    assert all(op.w_fmt == "bf16" and op.a_fmt == "bf16" for op in hd if op.name.startswith("io.audio_head"))
    vs = m.vision
    assert vs is not None and vs.family == "longcatvit" and vs.audio is not None
    assert vs.audio.kind == "lcafsmn" and vs.inter == 3456 and vs.depth == 32
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.px) == (72, 72, 2592, 2592, (1008, 1008))
    vops = vision_ops(vs, g, 1, "fp8")
    patch = next(op for op in vops if op.name == "vis.patch_embed")
    assert (patch.k, patch.n, patch.a_fmt, patch.w_fmt) == (3 * 2 * 14 * 14, 1280, "bf16", "bf16")
    assert next(op for op in vops if op.name == "vis.0.mlp_in").n == 3456 * 2
    assert next(op for op in vops if op.name == "vis.merger.0").m == 2592
    assert vs.audio.tokens(1.0) == 13 and vs.audio.tokens(0.01) == 0
    aops = audio_ops(vs.audio, 1.0, 1, "fp8")
    assert all(op.a_fmt == "bf16" and op.w_fmt == "bf16" for op in aops if op.kind == "gemm")
    assert next(op for op in aops if op.name == "aud.pre").k == 1200
    assert next(op for op in aops if op.name == "aud.out").n == 5252
    assert next(op for op in aops if op.name == "aud.0.dw").vec == 13 * 1536 * 11
    assert next(op for op in aops if op.name == "aud.16.dw").vec == 13 * 1536 * 12
    text = " ".join(m.notes)
    assert "没有 model.mtp" in text and "不把 V 补到 qk_head_dim" in text
    assert "Audio-Codec" in text and "language_model_embedding" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-omni"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "omni" and row["param_err"] == 0.0


def test_longcat_omni_fp8_skips_late_blocks_and_has_no_mtp():
    m = get_model("longcat-flash-omni-fp8")
    base = get_model("longcat-flash-omni")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == PARAMS + AUDIO_IO and m.mtp_params() == 0
    assert m.vision_params == base.vision_params == TOWERS
    assert rel["n_tensors"] == 79205
    assert rel["bytes_llm"] == rel["bytes_total"] == 666_757_550_080
    assert rel["bytes_mtp"] == 0 and "mtp" not in rel["roles"]
    assert m.coverage == "full" and m.mtp_layers == ()
    assert m.fmt("attn").fmt == "bf16" and m.act_fmt_for("attn") == "bf16"
    assert m.fmt("expert").fmt == "fp8" and m.fmt("expert_bf16").fmt == "bf16"
    assert m.fmt("mlp").fmt == "fp8" and m.fmt("mlp_bf16").fmt == "bf16"
    assert m.act_fmt_for("audio_head") == "bf16"
    assert "mtp_fp8" not in dict(m.formats) and "mtp_kv_a" not in dict(m.formats)
    for i, layer in enumerate(m.layers):
        blk = i // 2
        if i % 2 == 0:
            assert layer.ffn.expert_role == ("expert_bf16" if blk >= 23 else "")
        roles = {lin.role for lin in layer.ffn_linears if lin.name in ("gate_up", "down")}
        assert roles == ({"mlp_bf16"} if blk >= 23 else {""})
    text = " ".join(m.notes)
    assert "23、24、25、26、27" in text and "没有 MTP" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-omni-fp8"]
    assert row["domain"] == "omni" and row["param_err"] == 0.0 and row["provider"] == "meituan"


def test_longcat_omni_rejects_a_broken_header():
    m = get_model("longcat-flash-omni")
    rel = load_release(m.hf_id)
    missing = copy.deepcopy(rel)
    missing.pop("longcat_omni")
    _raises(lambda: from_release(m.id, m.hf_id, missing, missing["config"]), "tower header is missing")
    bad = copy.deepcopy(rel)
    bad["longcat_omni"] = dict(rel["longcat_omni"])
    bad["longcat_omni"]["params_audio_encoder"] += 1
    _raises(lambda: from_release(m.id, m.hf_id, bad, bad["config"]), "params_audio_encoder")
    io = copy.deepcopy(rel)
    io["params_audio_io"] = AUDIO_IO - 1
    _raises(lambda: from_release(m.id, m.hf_id, io, io["config"]), "params_audio_io")

    fp = get_model("longcat-flash-omni-fp8")
    frel = load_release(fp.hf_id)
    with_mtp = copy.deepcopy(frel)
    with_mtp["params_mtp"] = 1
    with_mtp["roles"] = copy.deepcopy(frel["roles"])
    with_mtp["roles"]["mtp"] = {"bf16": {"params": 1, "bytes": 2}}
    _raises(lambda: from_release(fp.id, fp.hf_id, with_mtp, with_mtp["config"]), "no MTP")
    partial = copy.deepcopy(frel["config"])
    partial["quantization_config"] = dict(partial["quantization_config"])
    partial["quantization_config"]["ignored_layers"] = list(
        partial["quantization_config"]["ignored_layers"]) + [
            "model.layers.0.mlp.experts.0.gate_proj"]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, partial), "expert projections")
