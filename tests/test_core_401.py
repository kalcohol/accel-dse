"""0.401: LongcatForCausalLM (Thinking-2601 / Prover) uses the published MLA stack.

The decoder matches LongCat-Flash: expanded KV, unpadded V in the cache.
deepseek_yarn scales cos/sin and does not change GEMM or KV width.
The HF class does not run MTP. The official guide runs checkpoint MTP with
vLLM method longcat_flash_mtp, which pads V to qk_head_dim. Prover has no
model.mtp.* tensors, so it has no draft.

Thinking-2601-FP8 uses the same ignored_layers partition as Chat-FP8.
No expert or dense name is ignored, so every block is block-fp8.
Attention stays bf16. MTP bytes match Chat-FP8.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, _mtp_ops, build_rank_ops
from accel_dse.core.model import from_release, load_release

H = 6144
HEADS = 64
QK = 192
V = 128
VOCAB = 131072
PARAMS = 560_664_980_480
MTP = 1_197_899_776
FP8_BITS = 8.0 * 16388 / 16384
KV_A_PARAMS = H * (512 + 64)
KV_A_BYTES = KV_A_PARAMS + 4 * 5 * 48
KV_A_BITS = 8.0 * KV_A_BYTES / KV_A_PARAMS


def _sig(ops):
    return [
        (op.name, op.kind, op.m, op.k, op.n, op.w_params, op.w_bits, op.w_fmt, op.kv_read, op.kv_write)
        for op in ops
    ]


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc)
    else:
        raise AssertionError(f"accepted a release that should mention {needle}")


def test_longcat_thinking_2601_matches_flash_and_runs_vllm_mtp():
    chat = get_model("longcat-flash-chat")
    m = get_model("longcat-flash-thinking-2601")
    rel = load_release(m.hf_id)
    crel = load_release(chat.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.param_check()["mtp_ours"] == m.param_check()["mtp_release"] == MTP
    assert m.params() == m.release_params == rel["params_llm"] == PARAMS
    assert m.mtp_params() == rel["params_mtp"] == MTP
    assert m.coverage == "full" and m.coverage_reasons == ()
    assert m.lookup_params == 0 and len(m.layers) == 56 and len(m.mtp_layers) == 1
    assert m.max_ctx == 983040 and chat.max_ctx == 131072
    assert rel["roles"] == crel["roles"]
    assert rel["n_tensors"] == crel["n_tensors"] == 43756
    assert rel["bytes_llm"] == crel["bytes_llm"] and rel["bytes_mtp"] == crel["bytes_mtp"]
    assert rel["bytes_total"] == rel["bytes_llm"] + rel["bytes_mtp"]
    assert rel["config"]["rope_scaling"]["rope_type"] == "deepseek_yarn"
    assert rel["config"]["rope_scaling"]["factor"] == 120
    assert rel["config"]["architectures"] == ["LongcatCausalLM"]
    assert m.layers[0].core.v_dim == V and m.mtp_layers[0].core.v_dim == QK
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("expert").fmt == "bf16"
    assert "expert_bf16" not in dict(m.formats) and "mtp_fp8" not in dict(m.formats)
    ph = Phase("decode", 1, 1, 64)
    assert _sig(build_rank_ops(m, 0, ph, Shard())) == _sig(build_rank_ops(chat, 0, ph, Shard()))
    assert _sig(build_rank_ops(m, 1, ph, Shard())) == _sig(build_rank_ops(chat, 1, ph, Shard()))
    assert _sig(_mtp_ops(m, ph, Shard())) == _sig(_mtp_ops(chat, ph, Shard()))
    assert next(op for op in build_rank_ops(m, 0, ph, Shard()) if op.name == "pv").n == V
    assert next(op for op in _mtp_ops(m, ph, Shard()) if op.name == "pv").n == QK
    text = " ".join(m.notes)
    assert "deepseek_yarn" in text and "longcat_flash_mtp" in text
    assert "未填充的 v_head_dim" in text and "显式传入 method" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-thinking-2601"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_prover_has_no_draft():
    base = get_model("longcat-flash-thinking-2601")
    m = get_model("longcat-flash-prover")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == PARAMS and m.mtp_params() == 0
    assert m.release_mtp_params == rel["params_mtp"] == 0
    assert m.coverage == "full" and m.mtp_layers == () and len(m.layers) == 56
    assert rel["n_tensors"] == 43739 and "mtp" not in rel["roles"]
    assert rel["bytes_llm"] == rel["bytes_total"] == 1_121_594_245_120
    assert rel["bytes_mtp"] == 0
    assert m.layers[0].core.v_dim == V and m.max_ctx == 983040
    ph = Phase("decode", 1, 1, 64)
    assert _sig(build_rank_ops(m, 0, ph, Shard())) == _sig(build_rank_ops(base, 0, ph, Shard()))
    assert _mtp_ops(m, ph, Shard()) == []
    text = " ".join(m.notes)
    assert "checkpoint 没有 model.mtp" in text and "不把 V 补到 qk_head_dim" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-prover"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_thinking_2601_fp8_quantizes_every_block():
    chat = get_model("longcat-flash-chat-fp8")
    m = get_model("longcat-flash-thinking-2601-fp8")
    rel = load_release(m.hf_id)
    crel = load_release(chat.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == PARAMS and m.mtp_params() == rel["params_mtp"] == MTP
    assert m.coverage == "full" and len(m.layers) == 56 and len(m.mtp_layers) == 1
    assert rel["n_tensors"] == 86941
    assert rel["bytes_llm"] == 567_880_007_680 and rel["bytes_mtp"] == crel["bytes_mtp"]
    assert rel["bytes_total"] == rel["bytes_llm"] + rel["bytes_mtp"]
    assert rel["roles"]["mtp"] == crel["roles"]["mtp"]
    assert rel["roles"]["attn"] == crel["roles"]["attn"]
    assert "bf16" not in rel["roles"]["expert"] and "bf16" not in rel["roles"]["mlp"]
    assert rel["roles"]["expert"]["fp8"]["params"] == 541_165_879_296
    assert rel["roles"]["mlp"]["fp8"]["params"] == 12_683_575_296
    assert m.act_fmt == "fp8" and m.act_fmt_for("attn") == "bf16"
    assert m.act_fmt_for("expert") == "fp8" and m.act_fmt_for("mlp") == "fp8"
    assert m.fmt("expert").bits == m.fmt("mlp").bits == FP8_BITS
    assert m.fmt("attn").fmt == "bf16" and m.fmt("mtp").bits == 16.0
    assert m.fmt("mtp_fp8").bits == FP8_BITS
    assert m.fmt("mtp_kv_a").bits == KV_A_BITS and m.fmt("mtp_kv_a").bits != FP8_BITS
    assert "expert_bf16" not in dict(m.formats) and "mlp_bf16" not in dict(m.formats)
    assert all(L.ffn.expert_role == "" for L in m.layers)
    assert all(lin.role == "" for L in m.layers for lin in L.ffn_linears)
    assert m.layers[0].core.v_dim == V and m.mtp_layers[0].core.v_dim == QK
    assert m.mtp_eh_role == "mtp_fp8" and m.mtp_eh_params == 2 * H * H
    ph = Phase("decode", 1, 1, 64)
    main = build_rank_ops(m, 0, ph, Shard())
    other = build_rank_ops(m, 2, ph, Shard())
    eg = next(op for op in main if op.name == "expert.gate_up")
    assert (eg.w_fmt, eg.w_bits, eg.a_fmt, eg.role) == ("fp8", FP8_BITS, "fp8", "expert")
    eg2 = next(op for op in other if op.name == "expert.gate_up")
    assert (eg2.w_fmt, eg2.a_fmt, eg2.role) == ("fp8", "fp8", "expert")
    mg = next(op for op in build_rank_ops(m, 1, ph, Shard()) if op.name == "mlp.gate_up")
    assert (mg.w_fmt, mg.w_bits, mg.a_fmt, mg.role) == ("fp8", FP8_BITS, "fp8", "mlp")
    assert next(op for op in main if op.name == "pv").n == V
    assert next(op for op in main if op.name == "attn.o").a_fmt == "bf16"
    ops = _mtp_ops(m, ph, Shard())
    assert next(op for op in ops if op.name == "pv").n == QK
    kva = next(op for op in ops if op.name == "attn.kv_a")
    assert (kva.w_bits, kva.w_fmt, kva.a_fmt) == (KV_A_BITS, "fp8", "fp8")
    text = " ".join(m.notes)
    assert "block （无）" in text and "self_attn" in text and "mtp_kv_a" in text
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-thinking-2601-fp8"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_yarn_does_not_change_gemms_and_rejects_other_stacks():
    m = get_model("longcat-flash-thinking-2601")
    rel = load_release(m.hf_id)
    cfg = copy.deepcopy(rel["config"])
    cfg["rope_scaling"] = dict(cfg["rope_scaling"])
    cfg["rope_scaling"]["factor"] = 8
    cfg["rope_scaling"]["mscale"] = 0.5
    cfg["max_position_embeddings"] = 4096
    cfg["rope_theta"] = 10000.0
    alt = from_release(m.id, m.hf_id, rel, cfg)
    assert alt.param_check()["rel_err"] == 0.0
    assert alt.mtp_params() == MTP and alt.params() == PARAMS
    assert alt.max_ctx == 4096
    assert alt.layers[0].core.v_dim == V and alt.mtp_layers[0].core.v_dim == QK
    ph = Phase("decode", 1, 1, 64)
    assert _sig(build_rank_ops(m, 0, ph, Shard())) == _sig(build_rank_ops(alt, 0, ph, Shard()))
    assert _sig(_mtp_ops(m, ph, Shard())) == _sig(_mtp_ops(alt, ph, Shard()))
    assert "factor 8" in " ".join(alt.notes)

    yarn = copy.deepcopy(rel["config"])
    yarn["rope_scaling"] = dict(yarn["rope_scaling"])
    yarn["rope_scaling"]["rope_type"] = "default"
    _raises(lambda: from_release(m.id, m.hf_id, rel, yarn), "deepseek_yarn")
    lsa = copy.deepcopy(rel["config"])
    lsa["attention_method"] = "LSA"
    _raises(lambda: from_release(m.id, m.hf_id, rel, lsa), "not MLA")
    idx = copy.deepcopy(rel["config"])
    idx["index_n_heads"] = 16
    _raises(lambda: from_release(m.id, m.hf_id, rel, idx), "index_n_heads")
    oe = copy.deepcopy(rel["config"])
    oe["oe_vocab_size_ratio"] = 78
    _raises(lambda: from_release(m.id, m.hf_id, rel, oe), "oe_vocab_size_ratio")
    nmtp = copy.deepcopy(rel["config"])
    nmtp["mtp_num_layers"] = 3
    _raises(lambda: from_release(m.id, m.hf_id, rel, nmtp), "mtp_num_layers")

    fp = get_model("longcat-flash-thinking-2601-fp8")
    frel = load_release(fp.hf_id)
    partial = copy.deepcopy(frel["config"])
    partial["quantization_config"] = dict(partial["quantization_config"])
    partial["quantization_config"]["ignored_layers"] = list(
        partial["quantization_config"]["ignored_layers"]) + [
            "model.layers.0.mlp.experts.0.gate_proj"]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, partial), "expert projections")
    scheme = copy.deepcopy(frel["config"])
    scheme["quantization_config"] = dict(scheme["quantization_config"])
    scheme["quantization_config"]["weight_block_size"] = [64, 64]
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, scheme), "weight_block_size")
    attn = copy.deepcopy(frel["config"])
    attn["disable_quant_module"] = []
    _raises(lambda: from_release(fp.id, fp.hf_id, frel, attn), "disable_quant_module")
    smeared = copy.deepcopy(frel)
    smeared["roles"] = copy.deepcopy(frel["roles"])
    smeared["roles"]["expert"] = copy.deepcopy(frel["roles"]["expert"])
    smeared["roles"]["expert"]["fp8"] = dict(smeared["roles"]["expert"]["fp8"])
    smeared["roles"]["expert"]["fp8"]["bytes"] += 1
    _raises(lambda: from_release(fp.id, fp.hf_id, smeared, smeared["config"]), "expert bytes")
