"""0.399: LongCat-Flash FP8 is billed per ignored block, not as a role average.

Chat-FP8 and Thinking-FP8 share headers. Experts and both dense MLPs stay bf16
only on the blocks named in ignored_layers. Other blocks are e4m3 with one
fp32 scale per 128x128 tile. Attention stays bf16. MTP eh_proj is fp8. kv_a
scale tiles are not the uniform ratio, so that GEMM has its own role.

The 0.398 draft cache is unchanged: use_mla does not select the pad, V is
padded only on the MTP layer, and the bf16 twins keep a single bf16 mtp role.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.memplan import _layer_storage
from accel_dse.core.model import from_release, load_release

H = 6144
HEADS = 64
QK = 192
V = 128
Q_LORA = 1536
KV_LORA = 512
D_FF = 12288
VOCAB = 131072
PARAMS = 560_664_980_480
MTP = 1_197_899_776
SKIP = {0, 2, 14, 17, 18, 24, 25, 26, 27}
FP8_BITS = 8.0 * 16388 / 16384
KV_A_PARAMS = H * (KV_LORA + 64)
KV_A_BYTES = KV_A_PARAMS + 4 * 5 * 48
KV_A_BITS = 8.0 * KV_A_BYTES / KV_A_PARAMS


def _bits(op):
    return (op.name, op.k, op.n, op.w_fmt, op.w_bits, op.a_fmt, op.role)


def test_longcat_flash_fp8_splits_ignored_blocks():
    m = get_model("longcat-flash-chat-fp8")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == PARAMS
    assert m.mtp_params() == rel["params_mtp"] == MTP
    assert m.coverage == "full" and m.lookup_params == 0 and len(m.layers) == 56
    assert len(m.mtp_layers) == 1
    assert rel["n_tensors"] == 73063
    assert rel["bytes_total"] == 747_862_918_848
    assert rel["bytes_llm"] + rel["bytes_mtp"] == rel["bytes_total"]
    assert m.act_fmt == "fp8"
    assert m.fmt("expert").fmt == "fp8" and m.fmt("expert").bits == FP8_BITS
    assert m.fmt("mlp").fmt == "fp8" and m.fmt("mlp").bits == FP8_BITS
    assert m.fmt("expert_bf16").bits == 16.0 and m.fmt("mlp_bf16").bits == 16.0
    assert m.fmt("attn").fmt == "bf16" and m.fmt("attn").bits == 16.0
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0
    assert m.fmt("mtp_fp8").bits == FP8_BITS
    assert m.fmt("mtp_kv_a").fmt == "fp8" and m.fmt("mtp_kv_a").bits == KV_A_BITS
    assert m.fmt("mtp_kv_a").bits != FP8_BITS
    assert m.fmt("router").fmt == "fp32" and m.fmt("embed").bits == 16.0
    assert m.act_fmt_for("attn") == "bf16"
    assert m.act_fmt_for("expert") == "fp8" and m.act_fmt_for("mlp") == "fp8"
    assert m.act_fmt_for("expert_bf16") == "bf16" and m.act_fmt_for("mlp_bf16") == "bf16"
    assert m.act_fmt_for("mtp_fp8") == "fp8" and m.act_fmt_for("mtp_kv_a") == "fp8"
    assert m.act_fmt_for("router") == "fp8"
    assert m.mtp_eh_params == 2 * H * H and m.mtp_eh_role == "mtp_fp8"
    assert m.mtp_extra_params - m.mtp_eh_params == VOCAB * H + H

    expert_blocks = {i // 2 for i, L in enumerate(m.layers) if L.ffn.expert_role == "expert_bf16"}
    dense_blocks = {i // 2 for i, L in enumerate(m.layers)
                    if any(lin.role == "mlp_bf16" for lin in L.ffn_linears)}
    assert expert_blocks == dense_blocks == SKIP
    assert sum(L.ffn.expert_role == "expert_bf16" for L in m.layers) == len(SKIP)
    assert m.layers[0].ffn.expert_role == "expert_bf16"
    assert m.layers[2].ffn.expert_role == ""
    assert all(lin.role != "mlp_bf16" for lin in m.layers[0].ffn_linears if lin.name == "router")
    assert all(lin.role == "" for lin in m.layers[2].ffn_linears)
    assert all(lin.role == "" for lin in m.layers[3].ffn_linears)

    exp = sum(_layer_storage(m, L, Shard())[1] for L in m.layers)
    roles = rel["roles"]
    assert exp == roles["expert"]["bf16"]["bytes"] + roles["expert"]["fp8"]["bytes"]
    mlp_b = 0.0
    attn_b = 0.0
    for L in m.layers:
        for lin in L.ffn_linears:
            if lin.name.startswith("router"):
                continue
            mlp_b += lin.params * m.fmt(lin.role or "mlp").bits / 8
        for lin in L.attn_linears:
            attn_b += lin.params * m.fmt(lin.role or "attn").bits / 8
    assert mlp_b == roles["mlp"]["bf16"]["bytes"] + roles["mlp"]["fp8"]["bytes"]
    assert attn_b == roles["attn"]["bf16"]["bytes"]
    hot, exp_m = _layer_storage(m, m.mtp_layers[0], Shard())
    eh_b = m.mtp_eh_params * m.fmt(m.mtp_eh_role).bits / 8
    rest = (m.mtp_extra_params - m.mtp_eh_params) * m.fmt("mtp").bits / 8
    assert exp_m == 0.0
    assert hot + eh_b + rest == rel["bytes_mtp"]

    ph = Phase("decode", 1, 1, 64)
    main = build_rank_ops(m, 0, ph, Shard())
    q = build_rank_ops(m, 2, ph, Shard())
    dense_bf = build_rank_ops(m, 1, ph, Shard())
    dense_fp = build_rank_ops(m, 3, ph, Shard())
    eg = next(op for op in main if op.name == "expert.gate_up")
    assert (eg.w_fmt, eg.w_bits, eg.a_fmt, eg.role) == ("bf16", 16.0, "bf16", "expert_bf16")
    egq = next(op for op in q if op.name == "expert.gate_up")
    assert (egq.w_fmt, egq.w_bits, egq.a_fmt, egq.role) == ("fp8", FP8_BITS, "fp8", "expert")
    mg = next(op for op in dense_bf if op.name == "mlp.gate_up")
    assert (mg.w_fmt, mg.w_bits, mg.a_fmt, mg.role) == ("bf16", 16.0, "bf16", "mlp_bf16")
    mq = next(op for op in dense_fp if op.name == "mlp.gate_up")
    assert (mq.w_fmt, mq.w_bits, mq.a_fmt, mq.role) == ("fp8", FP8_BITS, "fp8", "mlp")
    ao = next(op for op in main if op.name == "attn.o")
    assert (ao.k, ao.n, ao.w_fmt, ao.w_bits, ao.a_fmt) == (HEADS * V, H, "bf16", 16.0, "bf16")
    assert next(op for op in main if op.name == "pv").n == V
    assert next(op for op in main if op.name == "softmax").kv_read == 65 * HEADS * (QK + V) * 2

    ops = _mtp_ops(m, ph, Shard())
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_fmt, eh.w_bits, eh.a_fmt) == (
        2 * H, H, 2 * H * H, "fp8", FP8_BITS, "fp8")
    kva = next(op for op in ops if op.name == "attn.kv_a")
    assert (kva.k, kva.n, kva.w_bits, kva.w_fmt, kva.a_fmt) == (H, KV_LORA + 64, KV_A_BITS, "fp8", "fp8")
    mo = next(op for op in ops if op.name == "attn.o")
    assert (mo.k, mo.w_fmt, mo.w_bits, mo.a_fmt) == (HEADS * V, "fp8", FP8_BITS, "fp8")
    up = next(op for op in ops if op.name == "attn.kv_up")
    assert (up.k, up.n, up.w_bits) == (KV_LORA, HEADS * (V + V), FP8_BITS)
    assert next(op for op in ops if op.name == "pv").n == QK
    assert next(op for op in ops if op.name == "softmax").kv_read == 65 * HEADS * (QK + QK) * 2
    assert m.mtp_layers[0].core.v_dim == QK and m.layers[0].core.v_dim == V
    assert m.mtp_layers[0].ffn.kind == "dense" and m.mtp_layers[0].ffn.n_experts == 0
    assert rel["config"]["n_routed_experts"] == 512
    assert not any(op.kind == "lookup" or op.name == "moe.router" or op.name.startswith("expert.") for op in ops)
    text = " ".join(m.notes)
    assert "不把 bf16 和 fp8 平均" in text and "未填充的 V" in text and "mtp_kv_a" in text

    full = build_rank_ops(m, 0, Phase("full", 1, 4), Shard())
    fg = next(op for op in full if op.name == "expert.gate_up")
    assert (fg.role, fg.w_fmt, fg.a_fmt) == ("expert_bf16", "bf16", "bf16")
    full_q = build_rank_ops(m, 3, Phase("full", 1, 4), Shard())
    fmq = next(op for op in full_q if op.name == "mlp.gate_up")
    assert (fmq.role, fmq.w_fmt, fmq.a_fmt) == ("mlp", "fp8", "fp8")

    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-chat-fp8"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_flash_thinking_fp8_matches_chat():
    chat = get_model("longcat-flash-chat-fp8")
    m = get_model("longcat-flash-thinking-fp8")
    rel = load_release(m.hf_id)
    crel = load_release(chat.hf_id)
    assert rel["roles"] == crel["roles"]
    assert rel["config"] == crel["config"]
    assert rel["quantization_config"] == crel["quantization_config"]
    assert rel["bytes_total"] == crel["bytes_total"] and rel["n_tensors"] == 73063
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == chat.params() and m.mtp_params() == chat.mtp_params()
    assert m.mtp_eh_params == chat.mtp_eh_params and m.mtp_eh_role == chat.mtp_eh_role
    assert m.layers[0].core.v_dim == V and m.mtp_layers[0].core.v_dim == QK
    assert m.fmt("mtp_kv_a").bits == chat.fmt("mtp_kv_a").bits == KV_A_BITS
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-thinking-fp8"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_longcat_fp8_rejects_a_smeared_bucket_and_ignores_use_mla():
    m = get_model("longcat-flash-chat-fp8")
    rel = load_release(m.hf_id)
    ph = Phase("decode", 1, 1, 64)
    base = _mtp_ops(m, ph, Shard())
    for flag in (False, True):
        cfg = copy.deepcopy(rel["config"])
        cfg["use_mla"] = flag
        alt = from_release(m.id, m.hf_id, rel, cfg)
        assert alt.param_check()["rel_err"] == 0.0
        assert alt.layers[0].core.v_dim == V and alt.mtp_layers[0].core.v_dim == QK
        assert alt.mtp_eh_role == "mtp_fp8" and alt.fmt("expert").bits == FP8_BITS
        assert alt.layers[0].ffn.expert_role == "expert_bf16"
        assert alt.mtp_layers[0].ffn.n_experts == 0
        got = _mtp_ops(alt, ph, Shard())
        assert [_bits(op) for op in got if op.kind == "gemm"] == [_bits(op) for op in base if op.kind == "gemm"]
        main = build_rank_ops(alt, 0, ph, Shard())
        assert next(op for op in main if op.name == "pv").n == V

    bad = copy.deepcopy(rel)
    bad["roles"]["expert"]["bf16"]["bytes"] += 2
    try:
        from_release(m.id, m.hf_id, bad, bad["config"])
    except ValueError as exc:
        assert "expert" in str(exc)
    else:
        raise AssertionError("smeared expert bytes were accepted")

    partial = copy.deepcopy(rel)
    names = partial["config"]["quantization_config"]["ignored_layers"]
    partial["config"]["quantization_config"] = dict(partial["config"]["quantization_config"])
    dropped = [n for n in names if n != "model.layers.0.mlp.experts.0.down_proj"]
    partial["config"]["quantization_config"]["ignored_layers"] = dropped
    partial["quantization_config"] = partial["config"]["quantization_config"]
    try:
        from_release(m.id, m.hf_id, partial, partial["config"])
    except ValueError as exc:
        assert "expert" in str(exc)
    else:
        raise AssertionError("a partial ignored expert block was accepted")

    scheme = copy.deepcopy(rel["config"])
    scheme["quantization_config"] = dict(scheme["quantization_config"])
    scheme["quantization_config"]["weight_block_size"] = [64, 64]
    try:
        from_release(m.id, m.hf_id, rel, scheme)
    except ValueError as exc:
        assert "weight_block_size" in str(exc)
    else:
        raise AssertionError("a 64x64 fp8 block was accepted")

    narrow = copy.deepcopy(rel["config"])
    narrow["qk_nope_head_dim"] = 32
    narrow["qk_rope_head_dim"] = 32
    try:
        from_release(m.id, m.hf_id, rel, narrow)
    except ValueError as exc:
        assert "qk_head_dim" in str(exc)
    else:
        raise AssertionError("negative MTP V pad was accepted")


def test_bf16_longcat_mtp_is_unchanged_by_the_fp8_split():
    m = get_model("longcat-flash-chat")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.mtp_eh_params == 0 and m.mtp_eh_role == ""
    assert all(L.ffn.expert_role == "" for L in m.layers)
    assert all(lin.role == "" for L in m.layers for lin in L.ffn_linears)
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0
    assert "mtp_fp8" not in dict(m.formats)
    ph = Phase("decode", 1, 1, 64)
    ops = _mtp_ops(m, ph, Shard())
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_fmt, eh.w_bits, eh.role) == (2 * H, H, "bf16", 16.0, "mtp")
    assert m.layers[0].core.v_dim == V and m.mtp_layers[0].core.v_dim == QK
    assert m.mtp_layers[0].ffn.kind == "dense" and m.mtp_layers[0].ffn.n_experts == 0
    assert not any(op.kind == "lookup" for op in ops)
    for flag in (False, True):
        cfg = copy.deepcopy(rel["config"])
        cfg["use_mla"] = flag
        alt = from_release(m.id, m.hf_id, rel, cfg)
        assert alt.layers[0].core.v_dim == V and alt.mtp_layers[0].core.v_dim == QK
        assert alt.mtp_eh_params == 0
        assert _bits(next(op for op in _mtp_ops(alt, ph, Shard()) if op.name == "mtp.eh_proj")) == _bits(eh)
    assert (m.mtp_extra_params - 2 * H * H) == VOCAB * H + H
    assert next(l.n for l in m.mtp_layers[0].attn_linears if l.name == "q_a") == Q_LORA
