"""0.374: Step-3.5-Flash closes, and an MTP-only o_proj match retags MTP o.

The 0.373 MiMo both-match path (main stack and MTP o_proj) and Ling-flash-2.0
geometry stay. When only the MTP bf16 identity holds, MTP o moves to role
o_proj at bf16. The main stack stays on the attn role.
"""
from __future__ import annotations

import copy

from accel_dse import api as A
from accel_dse.core.catalog import entries, get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release


def _weights(m, n_mtp=0):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=n_mtp).weights


def _mtp_bytes(m):
    return _weights(m, len(m.mtp_layers)) - _weights(m, 0)


def _rope_vec(model_id: str, layer: int) -> int:
    m = get_model(model_id)
    ops = build_rank_ops(m, layer, Phase("decode", 1, 1, 64), Shard())
    ropes = [o for o in ops if o.name == "rope"]
    assert len(ropes) == 1
    return int(ropes[0].vec)


def test_step35_flash_matches_headers():
    m = get_model("step-3.5-flash")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 196_956_130_368
    assert m.mtp_params() == rel["params_mtp"] == 2_428_171_008
    assert m.coverage == "full" and m.vision_params == 0 and not m.tie_embeddings
    assert m.hidden == 4096 and m.vocab == 128896 and len(m.layers) == 45
    assert len(m.mtp_layers) == 3 and m.mtp_extra_params == 561_512_448
    full = [L for L in m.layers if L.core.window is None]
    sw = [L for L in m.layers if L.core.window == 512]
    assert len(full) == 12 and len(sw) == 33
    assert sum(L.ffn.kind == "dense" for L in m.layers) == 3
    assert sum(L.ffn.kind == "moe" for L in m.layers) == 42

    L0, L1, L3 = m.layers[0], m.layers[1], m.layers[3]
    assert L0.core.n_q == 64 and L0.core.n_kv == 8 and L0.core.qk_dim == 128
    assert L0.core.rope_dim == 64 and L0.core.v_dim == 128 and not L0.core.no_rope
    assert L0.ffn.kind == "dense" and L0.ffn.d_ff == 11264
    assert L0.misc_params == 2 * m.hidden + 2 * 128
    g0 = next(l for l in L0.attn_linears if l.name == "g_head")
    assert g0.n == 64 and g0.k == m.hidden
    assert L1.core.window == 512 and L1.core.n_q == 96 and L1.core.n_kv == 8
    assert L1.core.rope_dim == 128 and L1.core.qk_dim == 128
    assert L1.ffn.kind == "dense"
    assert next(l for l in L1.attn_linears if l.name == "g_head").n == 96
    assert L3.ffn.kind == "moe" and L3.ffn.n_experts == 288 and L3.ffn.top_k == 8
    assert L3.ffn.n_shared == 1 and L3.ffn.d_shared == 1280 and L3.ffn.groups == ()
    assert L3.misc_params == 2 * m.hidden + 2 * 128 + 288
    assert not any(l.name == "shared_expert_gate" for L in m.layers for l in L.ffn_linears)
    assert all(L.core.window == 512 and L.core.n_q == 96 and L.core.rope_dim == 128
               and L.ffn.kind == "dense" and L.ffn.d_ff == 11264 for L in m.mtp_layers)
    assert all(L.misc_params == 2 * 128 + 5 * m.hidden for L in m.mtp_layers)

    assert _rope_vec("step-3.5-flash", 0) == 64 * 64 * 3
    assert _rope_vec("step-3.5-flash", 1) == 96 * 128 * 3
    mtp_ops = _mtp_ops(m, Phase("decode", 1, 1, 64), Shard())
    assert int(next(o for o in mtp_ops if o.name == "rope").vec) == 96 * 128 * 3
    full_ops = build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard())
    sw_ops = build_rank_ops(m, 1, Phase("decode", 1, 1, 4096), Shard())
    width = (128 + 128) * 2
    assert next(o for o in full_ops if o.name == "softmax").kv_read == 4097 * 8 * width
    assert next(o for o in sw_ops if o.name == "softmax").kv_read == 512 * 8 * width

    for role in ("attn", "expert", "router", "mlp", "shared_expert"):
        assert m.fmt(role).fmt == "bf16" and m.fmt(role).bits == 16.0
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    assert "o_proj" not in dict(m.formats)
    # fp32 router_bias is LLM misc at 2 bytes (short 2 bytes/param) and the final RMSNorm
    # is absent from stage storage. MTP weights are entirely bf16.
    assert _weights(m) - rel["bytes_llm"] == -(m.hidden * 2 + 12096 * 2)
    assert _mtp_bytes(m) == rel["bytes_mtp"]
    eh = next(o for o in _mtp_ops(m, Phase("decode", 1, 1, 16), Shard()) if o.name == "mtp.eh_proj")
    head = next(o for o in _mtp_ops(m, Phase("decode", 1, 1, 16), Shard()) if o.name == "mtp.head")
    assert eh.w_fmt == "bf16" and eh.w_bits == 16.0 and eh.w_params == 2 * m.hidden * m.hidden
    assert head.w_fmt == "bf16" and head.n == m.vocab

    ids = {d["id"]: d for d in list_models()}
    assert ids["step-3.5-flash"]["family"] == "Step-3.5"
    assert ids["step-3.5-flash"]["provenance"] == "official"
    assert any("Step-3.5-Flash" in n for n in m.notes)
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "step-3.5-flash", "layout": {"tp": 8, "ep": 8},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_mtp_only_oproj_retag():
    flash = get_model("mimo-v2-flash")
    rel = load_release(flash.hf_id)
    broken = copy.deepcopy(rel)
    broken["roles"]["attn"]["bf16"]["params"] += 1
    spec = from_release("synth", flash.hf_id, rel=broken, cfg=rel["config"])
    main_o = [l for L in spec.layers for l in L.attn_linears if l.name == "o"]
    mtp_o = [l for L in spec.mtp_layers for l in L.attn_linears if l.name == "o"]
    assert main_o and all(l.role != "o_proj" for l in main_o)
    assert mtp_o and all(l.role == "o_proj" for l in mtp_o)
    assert spec.fmt("mtp").fmt == "bf16" and spec.fmt("mtp").bits == 16.0
    assert spec.fmt("o_proj").fmt == "bf16" and spec.fmt("o_proj").bits == 16.0
    ops = _mtp_ops(spec, Phase("decode", 1, 1, 16), Shard())
    o = next(op for op in ops if op.name == "attn.o")
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert o.w_fmt == "bf16" and o.w_bits == 16.0
    assert eh.w_fmt == "bf16" and eh.w_bits == 16.0
    # The catalog model is unchanged: both identities match, so every o is o_proj.
    assert all(l.role == "o_proj" for L in flash.layers + flash.mtp_layers
               for l in L.attn_linears if l.name == "o")
    got = sorted(e["id"] for e in entries() if "o_proj" in dict(get_model(e["id"]).formats))
    assert got == ["mimo-v2-flash", "mimo-v2.6-flash", "mimo-v2.6-pro"]


def test_ling_and_mimo_paths_unchanged():
    ling = get_model("ling-flash-2.0")
    assert ling.param_check()["rel_err"] == 0.0
    assert not ling.mtp_layers
    assert ling.layers[1].ffn.groups == (8, 4)
    mimo = get_model("mimo-v2-flash")
    assert mimo.param_check()["rel_err"] == 0.0
    assert mimo.fmt("o_proj").bits == 16.0 and mimo.fmt("mtp").bits == 16.0
