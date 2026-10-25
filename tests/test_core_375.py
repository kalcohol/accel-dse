"""0.375: Step-3.5-Flash review holds, and LongCat-Flash closes on expanded KV.

LongCat-Flash stores multi-head K/V after kv_b, not the latent (kv_lora + RoPE).
The router scores routed experts plus identity zero-experts. Only the routed
experts have GEMMs. An MTP-only o_proj match still retags MTP o and leaves the
main stack on attn.
"""
from __future__ import annotations

import copy

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release


def _weights(m, n_mtp=0):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=n_mtp).weights


def test_step35_review_and_mtp_only_peel():
    m = get_model("step-3.5-flash")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == 196_956_130_368 and m.mtp_params() == 2_428_171_008
    assert rel["params_llm"] == m.params() and rel["params_mtp"] == m.mtp_params()
    assert "o_proj" not in dict(m.formats)
    flash = get_model("mimo-v2-flash")
    assert all(l.role == "o_proj" for L in flash.layers + flash.mtp_layers for l in L.attn_linears if l.name == "o")
    broken = copy.deepcopy(load_release(flash.hf_id))
    broken["roles"]["attn"]["bf16"]["params"] += 1
    spec = from_release("synth", flash.hf_id, rel=broken, cfg=broken["config"])
    main_o = [l for L in spec.layers for l in L.attn_linears if l.name == "o"]
    mtp_o = [l for L in spec.mtp_layers for l in L.attn_linears if l.name == "o"]
    assert main_o and all(l.role != "o_proj" for l in main_o)
    assert mtp_o and all(l.role == "o_proj" for l in mtp_o)
    ops = _mtp_ops(spec, Phase("decode", 1, 1, 16), Shard())
    o = next(op for op in ops if op.name == "attn.o")
    assert o.w_fmt == "bf16" and o.w_bits == 16.0
    main = build_rank_ops(spec, 0, Phase("decode", 1, 1, 16), Shard())
    main_o_op = next(op for op in main if op.name == "attn.o")
    assert main_o_op.role != "o_proj"


def test_longcat_flash_expanded_kv_and_zero_experts():
    m = get_model("longcat-flash-chat")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 560_664_980_480
    assert m.coverage == "full" and m.vision_params == 0 and not m.tie_embeddings
    assert m.hidden == 6144 and m.vocab == 131072
    assert len(m.layers) == 56 and len(m.mtp_layers) == 1
    assert m.mtp_params() == rel["params_mtp"] == 1_197_899_776
    assert m.mtp_extra_params == m.vocab * m.hidden + 2 * m.hidden * m.hidden + m.hidden
    mt = m.mtp_layers[0]
    # 0.398: vLLM LongCatFlashMTP stores V padded to qk_head_dim. The main stack stays at v_head_dim.
    assert mt.core.kind == "gqa" and mt.core.kv_elems_per_token() == 64 * (192 + 192)
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == 12288 and mt.misc_params == 26624
    mtp_ops = _mtp_ops(m, Phase("decode", 1, 1, 16), Shard())
    eh = next(op for op in mtp_ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_bits) == (2 * m.hidden, m.hidden, 2 * m.hidden * m.hidden, 16.0)
    moe = [L for L in m.layers if L.ffn.kind == "moe"]
    dense = [L for L in m.layers if L.ffn.kind == "dense"]
    assert len(moe) == 28 and len(dense) == 28
    assert [L.ffn.kind for L in m.layers[:4]] == ["moe", "dense", "moe", "dense"]
    a, b = m.layers[0], m.layers[1]
    assert a.core.kind == "gqa" and b.core.kind == "gqa"
    assert a.core.kv_lora == 0 and b.core.kv_lora == 0
    assert (a.core.n_q, a.core.n_kv, a.core.qk_dim, a.core.v_dim, a.core.rope_dim) == (64, 64, 192, 128, 64)
    assert a.core.kv_elems_per_token() == 64 * (192 + 128)
    assert a.core.kv_elems_per_token() != 512 + 64
    q_a = next(l for l in a.attn_linears if l.name == "q_a")
    q = next(l for l in a.attn_linears if l.name == "q")
    kv_a = next(l for l in a.attn_linears if l.name == "kv_a")
    kv_up = next(l for l in a.attn_linears if l.name == "kv_up")
    o = next(l for l in a.attn_linears if l.name == "o")
    assert (q_a.k, q_a.n, q_a.split) == (6144, 1536, "rep")
    assert (q.k, q.n) == (1536, 64 * 192)
    assert (kv_a.k, kv_a.n) == (6144, 512 + 64)
    assert (kv_up.k, kv_up.n) == (512, 64 * (128 + 128))
    assert (o.k, o.n, o.split) == (64 * 128, 6144, "row")
    assert a.ffn.n_experts == 512 and a.ffn.zero_experts == 256 and a.ffn.top_k == 12
    assert a.ffn.d_expert == 2048 and a.ffn.d_ff == 12288 and a.ffn.n_shared == 0
    assert a.expert_params(m.hidden) == 512 * 3 * m.hidden * 2048
    assert a.misc_params == 2 * m.hidden + 1536 + 512 + 768
    assert b.ffn.d_ff == 12288 and b.ffn.zero_experts == 0
    assert b.misc_params == 2 * m.hidden + 1536 + 512
    router = next(l for l in a.ffn_linears if l.name == "router")
    assert (router.k, router.n, router.split) == (m.hidden, 768, "rep")
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    assert m.fmt("attn").fmt == "bf16" and m.fmt("expert").fmt == "bf16" and m.fmt("mlp").fmt == "bf16"
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"

    dec = build_rank_ops(m, 0, Phase("decode", 1, 1, 64), Shard())
    sm = next(op for op in dec if op.name == "softmax")
    assert sm.kv_read == 65 * 64 * (192 + 128) * 2
    assert sm.kv_write == 64 * (192 + 128) * 2
    up = next(op for op in dec if op.name == "attn.kv_up")
    assert (up.m, up.k, up.n) == (1, 512, 64 * 256)
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.w_fmt) == (m.hidden, 768, 32.0, "fp32")
    pre = build_rank_ops(m, 0, Phase("prefill", 1, 16, 100), Shard())
    pre_up = next(op for op in pre if op.name == "attn.kv_up")
    assert pre_up.m == 16

    tokens = 768
    routed = build_rank_ops(m, 0, Phase("decode", tokens, 1, 64), Shard())
    gate = next(op for op in routed if op.name == "expert.gate_up")
    useful = gate.count * gate.m * gate.share
    assert useful == tokens * 12 * 512 / 768
    assert useful == 6144
    assert any(op.name == "act" for op in routed)
    dense_ops = build_rank_ops(m, 1, Phase("decode", 1, 1, 64), Shard())
    assert any(op.name == "mlp.gate_up" for op in dense_ops)
    assert not any(op.name == "expert.gate_up" for op in dense_ops)
    # fp32 correction bias is LLM misc at 2 bytes (short 2 bytes/param). Final RMSNorm is
    # absent from stage storage.
    assert _weights(m) - rel["bytes_llm"] == -(m.hidden * 2 + 28 * 768 * 2)

    ids = {d["id"]: d for d in list_models()}
    row = ids["longcat-flash-chat"]
    assert row["family"] == "LongCat-Flash" and row["provenance"] == "official"
    assert row["provider"] == "meituan"
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "longcat-flash-chat", "layout": {"tp": 8, "ep": 8},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0
