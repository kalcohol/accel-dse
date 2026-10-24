"""0.372: re-check the 0.371 q/k-norm ledger, and land MiMo-V2-Flash.

Olmo-3 still norms the full Q/K projection and runs RoPE on both layer kinds.
EXAONE-4 still norms head_dim and drops RoPE only on hybrid full layers. The GQA
rope vector stays the historical query-head half-head width. MiMo-V2-Flash uses
the existing hybrid builder; partial RoPE does not change GEMM or KV width.
"""
from __future__ import annotations

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release


def _rope_vec(model_id: str, layer: int) -> int | None:
    m = get_model(model_id)
    ops = build_rank_ops(m, layer, Phase("decode", 1, 1, 64), Shard())
    ropes = [o for o in ops if o.name == "rope"]
    if not ropes:
        return None
    assert len(ropes) == 1
    return int(ropes[0].vec)


def test_olmo_exaone_rope_vector_stays_half_head():
    # 0.371 ledger, re-checked. The rope op bills query heads times head_dim // 2,
    # including on Olmo full layers. EXAONE hybrid full layers stay NoPE.
    # Decode, one token, TP 1: vec = n_q * (head_dim // 2) * 3.
    for mid, n_q, hd, sw, full in (
        ("olmo-3-7b-think", 32, 128, 0, 3),
        ("olmo-3-32b-think", 40, 128, 0, 3),
    ):
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0
        assert all(not L.core.no_rope and L.core.rope_dim == 0 for L in m.layers)
        want = n_q * (hd // 2) * 3
        assert _rope_vec(mid, sw) == want and _rope_vec(mid, full) == want
        # 7B MHA: both norms are hidden. 32B GQA: k_norm is KV * head_dim.
        qk = m.layers[0].core.n_q * hd + m.layers[0].core.n_kv * hd
        assert m.layers[0].misc_params == 2 * m.hidden + qk

    m = get_model("exaone-4.0-32b")
    assert m.param_check()["rel_err"] == 0.0
    assert _rope_vec("exaone-4.0-32b", 0) == 40 * (128 // 2) * 3
    assert _rope_vec("exaone-4.0-32b", 3) is None
    assert m.layers[3].core.no_rope and m.layers[3].core.qk_dim == 128
    sw = build_rank_ops(m, 0, Phase("decode", 1, 1, 64), Shard())
    full = build_rank_ops(m, 3, Phase("decode", 1, 1, 64), Shard())
    assert next(o for o in sw if o.name == "softmax").kv_read == next(
        o for o in full if o.name == "softmax").kv_read

    tiny = get_model("exaone-4.0-1.2b")
    assert tiny.param_check()["rel_err"] == 0.0 and tiny.tie_embeddings
    assert all(not L.core.no_rope and L.core.window is None for L in tiny.layers)
    assert _rope_vec("exaone-4.0-1.2b", 0) == 32 * (64 // 2) * 3


def test_mimo_v2_flash_matches_headers():
    m = get_model("mimo-v2-flash")
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 308_778_780_864
    assert m.params(include_mtp=True) - m.params() == m.release_mtp_params == 987_820_224
    assert m.mtp_extra_params == 2 * m.hidden * m.hidden and len(m.mtp_layers) == 3
    assert m.coverage == "full" and m.vision_params == 0 and not m.tie_embeddings
    assert m.act_fmt == "fp8" and m.kv_fmt == "bf16"
    swa = [L for L in m.layers if L.core.window == 128]
    full = [L for L in m.layers if L.core.window is None]
    assert len(swa) == 39 and len(full) == 9
    assert all(L.core.sink and L.core.n_kv == 8 and not L.core.no_rope for L in swa)
    assert all((not L.core.sink) and L.core.n_kv == 4 and L.core.rope_dim == 0 for L in full)
    assert all(L.core.n_q == 64 and L.core.qk_dim == 192 and L.core.v_dim == 128 for L in m.layers)
    assert m.layers[0].ffn.kind == "dense" and m.layers[0].ffn.d_ff == 16384
    assert m.layers[0].misc_params == 2 * m.hidden
    moe = [L for L in m.layers if L.ffn.kind == "moe"]
    assert len(moe) == 47
    assert moe[0].ffn.n_experts == 256 and moe[0].ffn.top_k == 8 and moe[0].ffn.n_shared == 0
    assert moe[0].ffn.groups == () and moe[0].ffn.d_expert == 2048
    # SWA MoE misc: two hidden norms, one sink logit per head, and the score-correction bias.
    # No q/k norm. The dense full layer (0) has only the two norms.
    assert swa[0].misc_params == 256 + 64 + 2 * m.hidden
    # MTP is the SWA block plus a dense FFN, five hidden norms, and eh_proj [h, 2h].
    mt = m.mtp_layers[0]
    assert mt.core.window == 128 and mt.core.sink and mt.ffn.kind == "dense"
    assert mt.misc_params == 64 + 5 * m.hidden
    # Historical query-head rope bill: half of 192, not the rotated 64.
    assert _rope_vec("mimo-v2-flash", 0) == 64 * (192 // 2) * 3
    assert _rope_vec("mimo-v2-flash", 1) == 64 * (192 // 2) * 3
    sw_ops = build_rank_ops(m, 1, Phase("decode", 1, 1, 64), Shard())
    full_ops = build_rank_ops(m, 0, Phase("decode", 1, 1, 64), Shard())
    assert next(o for o in sw_ops if o.name == "softmax").kv_read == 2 * next(
        o for o in full_ops if o.name == "softmax").kv_read
    rel = load_release(m.hf_id)
    attn = rel["roles"]["attn"]
    o_params = sum(lin.params for L in m.layers for lin in L.attn_linears if lin.name == "o")
    sinks = sum(L.core.n_q for L in m.layers if L.core.sink)
    assert attn["bf16"]["params"] == o_params + sinks
    assert attn["bf16"]["bytes"] == 2 * attn["bf16"]["params"]
    qkv = sum(lin.params for L in m.layers for lin in L.attn_linears if lin.name in ("q", "k", "v"))
    assert attn["fp8"]["params"] == qkv
    # 0.373: the attn role is the fp8 bucket (q/k/v + block scales). o_proj is its own bf16 role.
    fp8 = attn["fp8"]
    assert m.fmt("attn").fmt == "fp8" and m.fmt("attn").bits == 8.0 * fp8["bytes"] / fp8["params"]
    assert m.fmt("o_proj").fmt == "bf16" and m.fmt("o_proj").bits == 16.0
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * rel["roles"]["router"]["fp32"]["params"]
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    assert m.fmt("expert").fmt == "fp8" and m.fmt("embed").bits == 16.0
    # Block-128 scales sit in the expert byte bucket and stay out of the logical count.
    exp = rel["roles"]["expert"]["fp8"]
    assert exp["params"] == 47 * 256 * 3 * m.hidden * 2048
    assert exp["bytes"] > exp["params"]
    ids = {d["id"]: d for d in list_models()}
    assert ids["mimo-v2-flash"]["family"] == "MiMo-V2-Flash"
    assert ids["mimo-v2-flash"]["provenance"] == "official"
    assert any("MiMo-V2-Flash" in n for n in m.notes)
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "mimo-v2-flash", "layout": {"tp": 8, "ep": 8},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_mimo_v26_stack_is_unchanged():
    m = get_model("mimo-v2.6-flash")
    assert m.params() == 308_778_780_864
    assert m.params(include_mtp=True) - m.params() == 987_820_224
    assert m.fmt("expert").fmt == "mxfp4" and m.fmt("attn").fmt == "fp8"
    assert all(L.core.rope_dim == 0 for L in m.layers)
    assert not any("MiMo-V2-Flash" in n for n in m.notes)
