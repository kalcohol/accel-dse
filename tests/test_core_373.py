"""0.373: MiMo o_proj / MTP eh_proj are the bf16 buckets, and Ling-flash-2.0 closes.

The 0.372 MiMo-V2-Flash stack (hybrid SWA, dense L0, MoE 256/top-8, MTP x3,
half-head rope vector, unread attention_chunk_size) stays. What changes is the
weight format: o_proj and the sinks are bf16, q/k/v stay on the block-fp8 bucket.
"""
from __future__ import annotations

from accel_dse import api as A
from accel_dse.core.catalog import entries, get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import load_release


MIMO = ("mimo-v2-flash", "mimo-v2.6-flash", "mimo-v2.6-pro")


def _weights(m, n_mtp=0):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=n_mtp).weights


def _mtp_bytes(m):
    n = len(m.mtp_layers)
    return _weights(m, n) - _weights(m, 0)


def _rope_vec(model_id: str, layer: int) -> int:
    m = get_model(model_id)
    ops = build_rank_ops(m, layer, Phase("decode", 1, 1, 64), Shard())
    ropes = [o for o in ops if o.name == "rope"]
    assert len(ropes) == 1
    return int(ropes[0].vec)


def test_mimo_oproj_and_eh_proj_are_bf16():
    for mid in MIMO:
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0
        attn = rel["roles"]["attn"]
        fp8 = attn["fp8"]
        assert m.fmt("attn").fmt == "fp8"
        assert m.fmt("attn").bits == 8.0 * fp8["bytes"] / fp8["params"]
        assert m.fmt("o_proj").fmt == "bf16" and m.fmt("o_proj").bits == 16.0
        assert m.act_fmt_for("o_proj") == "fp8" and m.act_fmt == "fp8"
        named = [l for L in m.layers + m.mtp_layers for l in L.attn_linears if l.name == "o"]
        assert named and all(l.role == "o_proj" for l in named)
        assert m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0
        eh = next(o for o in _mtp_ops(m, Phase("decode", 1, 1, 16), Shard()) if o.name == "mtp.eh_proj")
        assert eh.w_fmt == "bf16" and eh.w_bits == 16.0 and eh.a_fmt == "fp8"
        assert eh.w_params == 2 * m.hidden * m.hidden
        # eh_proj / o_proj / sinks / norms are 16-bit and match the bf16 bucket. The dense MLP
        # is billed at the mlp role, which matches the MTP fp8 scale ratio. q/k/v are billed at
        # the main-stack attn role. On Flash and V2.6-Flash that ratio sits slightly above the
        # MTP fp8 bucket (about 146 bytes). On V2.6-Pro the two sides meet.
        qkv = sum(l.params for L in m.mtp_layers for l in L.attn_linears if l.name != "o")
        ff = sum(l.params for L in m.mtp_layers for l in L.ffn_linears)
        mtp_fp8 = rel["roles"]["mtp"]["fp8"]
        lo_b = (qkv * m.fmt("attn").bits + ff * m.fmt("mlp").bits) / 8
        assert abs((_mtp_bytes(m) - rel["bytes_mtp"]) - (lo_b - mtp_fp8["bytes"])) < 1e-4

    flash = get_model("mimo-v2-flash")
    rel = load_release(flash.hf_id)
    ops = build_rank_ops(flash, 0, Phase("decode", 1, 1, 16), Shard())
    o = next(op for op in ops if op.name == "attn.o")
    qkv = next(op for op in ops if op.name == "attn.qkv")
    assert o.w_fmt == "bf16" and o.w_bits == 16.0 and o.a_fmt == "fp8"
    assert qkv.w_fmt == "fp8" and qkv.a_fmt == "fp8"
    assert qkv.w_bits == flash.fmt("attn").bits
    # LLM misc stays 2 bytes/param, so the fp32 router bias (47 x 256) is 2 bytes short
    # and the final RMSNorm is absent from stage storage.
    bias = 47 * 256
    assert _weights(flash) - rel["bytes_llm"] == -(flash.hidden * 2 + bias * 2)
    v26 = get_model("mimo-v2.6-flash")
    assert v26.params() == 308_778_780_864
    assert v26.params(include_mtp=True) - v26.params() == 987_820_224
    assert v26.fmt("expert").fmt == "mxfp4" and v26.fmt("attn").fmt == "fp8"
    assert all(L.core.rope_dim == 0 for L in v26.layers)
    pro = get_model("mimo-v2.6-pro")
    assert pro.fmt("expert").fmt == "mxfp4" and all(L.core.rope_dim == 0 for L in pro.layers)


def test_mimo_v2_flash_geometry_unchanged():
    m = get_model("mimo-v2-flash")
    assert m.params() == m.release_params == 308_778_780_864
    assert len(m.mtp_layers) == 3 and m.mtp_extra_params == 2 * m.hidden * m.hidden
    assert _rope_vec("mimo-v2-flash", 0) == 64 * (192 // 2) * 3
    assert _rope_vec("mimo-v2-flash", 1) == 64 * (192 // 2) * 3
    full = build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard())
    sw = build_rank_ops(m, 1, Phase("decode", 1, 1, 4096), Shard())
    # attention_chunk_size is 128 and is not read. Full attention sees ctx + the new token.
    # The cache stores the whole head (192 + 128), not the rotated 64.
    assert m.layers[0].core.window is None and m.layers[1].core.window == 128
    assert m.layers[1].core.n_kv == 8 and m.layers[0].core.n_kv == 4
    width = (192 + 128) * 2
    assert next(o for o in full if o.name == "softmax").kv_read == 4097 * 4 * width
    assert next(o for o in sw if o.name == "softmax").kv_read == 128 * 8 * width


def test_oproj_split_catalog():
    got = sorted(e["id"] for e in entries() if "o_proj" in dict(get_model(e["id"]).formats))
    assert got == ["mimo-v2-flash", "mimo-v2.6-flash", "mimo-v2.6-pro"]


def test_ling_flash_2_0_matches_headers():
    m = get_model("ling-flash-2.0")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 102_889_705_216
    assert rel["params_llm"] == 102_889_705_216 and rel["params_mtp"] == 0
    assert m.coverage == "full" and m.vision_params == 0 and not m.tie_embeddings
    assert m.hidden == 4096 and m.vocab == 157184 and len(m.layers) == 32
    assert not m.mtp_layers and m.mtp_extra_params == 0
    dense = [L for L in m.layers if L.ffn.kind == "dense"]
    moe = [L for L in m.layers if L.ffn.kind == "moe"]
    assert len(dense) == 1 and len(moe) == 31
    assert dense[0].ffn.d_ff == 9216
    assert moe[0].ffn.n_experts == 256 and moe[0].ffn.top_k == 8
    assert moe[0].ffn.n_shared == 1 and moe[0].ffn.d_shared == 1024
    assert moe[0].ffn.groups == (8, 4)
    assert not any(l.name == "shared_expert_gate" for L in m.layers for l in L.ffn_linears)
    # two hidden norms + q/k RMSNorm(head_dim); MoE layers also store expert_bias [E]
    assert dense[0].misc_params == 2 * m.hidden + 2 * 128
    assert moe[0].misc_params == 2 * m.hidden + 2 * 128 + 256
    assert all(L.core.window is None and L.core.rope_dim == 0 and not L.core.no_rope for L in m.layers)
    assert all(L.core.n_q == 32 and L.core.n_kv == 4 and L.core.qk_dim == 128 for L in m.layers)
    assert _rope_vec("ling-flash-2.0", 0) == 32 * (128 // 2) * 3
    assert m.fmt("attn").fmt == "bf16" and m.fmt("expert").fmt == "bf16"
    # expert_bias is fp32 inside the mlp role, so the dense GEMM carries that smear, and the
    # buffer itself is LLM misc at 2 bytes. The final RMSNorm is not in stage storage.
    mlp = rel["roles"]["mlp"]
    dense_p, bias_p = mlp["bf16"]["params"], mlp["fp32"]["params"]
    dense_b, bias_b = mlp["bf16"]["bytes"], mlp["fp32"]["bytes"]
    avg = 8.0 * (dense_b + bias_b) / (dense_p + bias_p)
    smear = dense_p * avg / 8 + bias_p * 2 - (dense_b + bias_b)
    assert abs((_weights(m) - rel["bytes_llm"]) - (smear - m.hidden * 2)) < 1e-3
    assert m.fmt("router").fmt == "bf16" and m.fmt("router").bits == 16.0
    ids = {d["id"]: d for d in list_models()}
    assert ids["ling-flash-2.0"]["family"] == "Ling-flash-2.0"
    assert ids["ling-flash-2.0"]["provenance"] == "official"
    assert any("bailing_moe" in n for n in m.notes)
    assert any("没有 MTP" in n for n in m.notes)
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "ling-flash-2.0", "layout": {"tp": 8, "ep": 8},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0
