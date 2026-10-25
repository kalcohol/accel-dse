"""0.400: LongCat-Flash-Lite n-gram embeddings, and Lite-FP8 block scales.

Lite is LongcatFlashNgramForCausalLM. The decoder is the Flash dual-sublayer
stack. N-gram tables are lookups. Projections stay bf16. HF and vLLM drop
model.mtp.*, so the checkpoint MTP module is not a layer.

Lite-FP8 quantizes experts, both dense MLPs, and attention. kv_a scale tiles
are not the uniform 128 ratio. Chat / Thinking bf16 still do not take the
0.399 per-block split.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, _mtp_ops, build_rank_ops, embed_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release

H = 3072
VOCAB = 131072
BLOCKS = 14
POOL = 384
PARAMS = 68_562_465_024
MTP = 510_870_528
TABLES = 31_406_985_216
PROJS = 9_437_184
ORDER = 4
SPLITS = 4
NUM = 12
DIM = 256
FP8_BITS = 8.0 * 4097 / 4096
KV_A_PARAMS = H * 576
KV_A_BYTES = KV_A_PARAMS + 4 * 5 * 24
KV_A_BITS = 8.0 * KV_A_BYTES / KV_A_PARAMS
BIAS_GAP = 2 * H + 2 * BLOCKS * POOL


def _weights(m):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0).weights


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc)
    else:
        raise AssertionError(f"accepted a release that should mention {needle}")


def test_longcat_flash_lite_ngram_is_exact():
    m = get_model("longcat-flash-lite")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == PARAMS
    assert m.mtp_params() == 0 and m.release_mtp_params == rel["params_mtp"] == MTP
    assert m.coverage == "full" and m.coverage_reasons == ()
    assert m.mtp_layers == () and len(m.layers) == 2 * BLOCKS
    assert m.layers[0].core.v_dim == 128 and m.layers[0].core.kind == "gqa"
    assert rel["n_tensors"] == 11160
    assert rel["bytes_llm"] == 137_157_970_944 and rel["bytes_mtp"] == 1_021_741_056
    assert rel["bytes_total"] == rel["bytes_llm"] + rel["bytes_mtp"] == 138_179_712_000
    assert "other" not in rel["roles"]
    assert rel["roles"]["embed"]["bf16"]["params"] == VOCAB * H + TABLES
    assert rel["roles"]["ngram"]["bf16"]["params"] == PROJS
    assert rel["roles"]["ngram"]["bf16"]["bytes"] == 2 * PROJS
    assert m.ngram_order == ORDER and m.ngram_splits == SPLITS and m.ngram_dim == DIM
    assert m.lookup_params == TABLES and m.lookup_bits == 16.0
    assert len(m.io_pre) == NUM
    assert all(lin.k == DIM and lin.n == H and lin.role == "ngram" and lin.split == "rep" for lin in m.io_pre)
    assert sum(lin.params for lin in m.io_pre) == PROJS
    assert m.act_fmt == "bf16" and m.act_fmt_for("ngram") == "bf16"
    assert m.fmt("ngram").fmt == "bf16" and m.fmt("ngram").bits == 16.0
    assert m.fmt("attn").bits == 16.0 and m.fmt("expert").bits == 16.0
    assert "kv_a" not in dict(m.formats)
    assert all(lin.role != "kv_a" for L in m.layers for lin in L.attn_linears)
    assert all(L.ffn.expert_role == "" for L in m.layers)
    assert rel["bytes_llm"] - _weights(m) == BIAS_GAP
    ph = Phase("decode", 1, 1, 64)
    ops = embed_ops(m, ph, Shard())
    look = next(op for op in ops if op.name == "ngram_lookup")
    assert look.kind == "lookup" and look.kv_read == NUM * DIM * 2 and look.vec == 63
    proj = next(op for op in ops if op.name == "ngram.proj")
    assert (proj.k, proj.n, proj.count, proj.w_bits, proj.w_fmt, proj.a_fmt, proj.role) == (
        DIM, H, NUM, 16.0, "bf16", "bf16", "ngram")
    mean = next(op for op in ops if op.name == "ngram_mean")
    assert mean.vec == H * (NUM + 1)
    assert _mtp_ops(m, ph, Shard()) == []
    layer = build_rank_ops(m, 0, ph, Shard())
    assert not any(op.name.startswith("ngram") for op in layer)
    assert next(op for op in layer if op.name == "pv").n == 128
    text = " ".join(m.notes)
    assert "丢掉 model.mtp" in text and "不把 n-1 个历史 token" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-lite"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_flash_lite_fp8_splits_kv_a():
    m = get_model("longcat-flash-lite-fp8")
    rel = load_release(m.hf_id)
    bf = load_release("meituan-longcat/LongCat-Flash-Lite")
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == PARAMS and m.mtp_params() == 0
    assert m.release_mtp_params == MTP and rel["params_llm"] == bf["params_llm"]
    assert rel["params_mtp"] == bf["params_mtp"] == MTP
    assert m.coverage == "full" and m.mtp_layers == ()
    assert rel["n_tensors"] == 22145
    assert rel["bytes_llm"] == 100_842_857_088 and rel["bytes_mtp"] == 913_567_584
    assert rel["bytes_total"] == rel["bytes_llm"] + rel["bytes_mtp"]
    assert rel["config"]["disable_quant_module"] == []
    assert rel["config"]["quantization_config"]["weight_block_size"] == [128, 128]
    assert m.act_fmt == "fp8" and m.act_fmt_for("ngram") == "bf16"
    assert m.act_fmt_for("attn") == "fp8" and m.act_fmt_for("kv_a") == "fp8"
    assert m.act_fmt_for("expert") == "fp8" and m.act_fmt_for("router") == "fp8"
    assert m.fmt("expert").bits == m.fmt("mlp").bits == m.fmt("attn").bits == FP8_BITS
    assert m.fmt("kv_a").fmt == "fp8" and m.fmt("kv_a").bits == KV_A_BITS
    assert m.fmt("kv_a").bits != FP8_BITS
    assert m.fmt("ngram").bits == 16.0 and m.fmt("router").fmt == "fp32"
    assert m.lookup_params == TABLES and len(m.io_pre) == NUM
    assert all(lin.name == "kv_a" and lin.role == "kv_a" or lin.name != "kv_a"
               for L in m.layers for lin in L.attn_linears)
    assert sum(lin.role == "kv_a" for L in m.layers for lin in L.attn_linears) == 2 * BLOCKS
    assert rel["roles"]["expert"] == {"fp8": {"params": 33_822_867_456, "bytes": 33_831_124_992}}
    assert rel["roles"]["mlp"]["fp8"]["params"] == 1_585_446_912
    assert "bf16" not in rel["roles"]["expert"] and "bf16" not in rel["roles"]["attn"]
    assert rel["bytes_llm"] - _weights(m) == BIAS_GAP
    ph = Phase("decode", 1, 1, 64)
    ops = build_rank_ops(m, 0, ph, Shard())
    kva = next(op for op in ops if op.name == "attn.kv_a")
    assert (kva.k, kva.n, kva.w_bits, kva.w_fmt, kva.a_fmt, kva.role) == (
        H, 576, KV_A_BITS, "fp8", "fp8", "kv_a")
    qa = next(op for op in ops if op.name == "attn.q_a")
    assert (qa.w_bits, qa.a_fmt, qa.role) == (FP8_BITS, "fp8", "attn")
    eg = next(op for op in ops if op.name == "expert.gate_up")
    assert (eg.w_fmt, eg.w_bits, eg.a_fmt, eg.role) == ("fp8", FP8_BITS, "fp8", "expert")
    proj = next(op for op in embed_ops(m, ph, Shard()) if op.name == "ngram.proj")
    assert (proj.w_fmt, proj.w_bits, proj.a_fmt) == ("bf16", 16.0, "bf16")
    assert next(op for op in ops if op.name == "pv").n == 128
    assert _mtp_ops(m, ph, Shard()) == []
    text = " ".join(m.notes)
    assert "kv_a" in text and "disable_quant_module 是空的" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-lite-fp8"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_ngram_rejects_a_mismatch():
    m = get_model("longcat-flash-lite")
    rel = load_release(m.hf_id)
    bad = copy.deepcopy(rel)
    bad["roles"]["embed"]["bf16"]["bytes"] += 2
    _raises(lambda: from_release(m.id, m.hf_id, bad, bad["config"]), "embed")
    ratio = copy.deepcopy(rel["config"])
    ratio["ngram_vocab_size_ratio"] = 77
    _raises(lambda: from_release(m.id, m.hf_id, rel, ratio), "embed")
    fp = get_model("longcat-flash-lite-fp8")
    frel = load_release(fp.hf_id)
    smeared = copy.deepcopy(frel)
    smeared["roles"]["expert"]["fp8"]["bytes"] += 1
    _raises(lambda: from_release(fp.id, fp.hf_id, smeared, smeared["config"]), "expert")
    scheme = copy.deepcopy(frel["config"])
    scheme["quantization_config"] = dict(scheme["quantization_config"])
    scheme["quantization_config"]["weight_block_size"] = [64, 64]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, scheme), "weight_block_size")
    missed = copy.deepcopy(frel["config"])
    missed["quantization_config"] = dict(missed["quantization_config"])
    missed["quantization_config"]["ignored_layers"] = [
        n for n in missed["quantization_config"]["ignored_layers"]
        if n != "model.ngram_embeddings.embedders.0"]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, missed), "ignored_layers")
    named = copy.deepcopy(frel["config"])
    named["quantization_config"] = dict(named["quantization_config"])
    named["quantization_config"]["ignored_layers"] = list(
        named["quantization_config"]["ignored_layers"]) + ["model.layers.0.mlp.experts.0.gate_proj"]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, named), "main-stack GEMM")


def test_longcat_fp8_dense_ignore_and_bf16_thinking():
    m = get_model("longcat-flash-chat-fp8")
    rel = load_release(m.hf_id)
    partial = copy.deepcopy(rel["config"])
    partial["quantization_config"] = dict(partial["quantization_config"])
    partial["quantization_config"]["ignored_layers"] = [
        n for n in partial["quantization_config"]["ignored_layers"]
        if n != "model.layers.0.mlps.0.gate_proj"]
    _raises(lambda: from_release(m.id, m.hf_id, rel, partial), "dense")
    bad = copy.deepcopy(rel)
    bad["roles"]["mlp"]["bf16"]["bytes"] += 2
    _raises(lambda: from_release(m.id, m.hf_id, bad, bad["config"]), "dense MLP")
    think = get_model("longcat-flash-thinking")
    assert think.param_check()["rel_err"] == 0.0
    assert think.mtp_eh_params == 0 and think.mtp_eh_role == ""
    assert "expert_bf16" not in dict(think.formats) and "mtp_fp8" not in dict(think.formats)
    assert "mlp_bf16" not in dict(think.formats)
    assert all(L.ffn.expert_role == "" for L in think.layers)
    assert all(lin.role == "" for L in think.layers for lin in L.ffn_linears)
    assert think.fmt("expert").fmt == "bf16" and think.fmt("mtp").bits == 16.0
    assert think.layers[0].core.v_dim == 128 and think.mtp_layers[0].core.v_dim == 192
