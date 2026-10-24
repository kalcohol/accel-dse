"""0.376: LongCat-Flash review holds, and Hunyuan-A13B closes on the text MoE.

LongCat's published class does not run model.mtp.*. The undeclared-MTP clone keeps
the private embedding in the remainder. Hunyuan-A13B is GQA with q/k RMSNorm, a
shared SwiGLU on every token, and an fp32 router. CLA, MLA, and the ViT are off.
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


def test_longcat_review_holds():
    m = get_model("longcat-flash-chat")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == 560_664_980_480
    assert m.mtp_params() == rel["params_mtp"] == 1_197_899_776
    assert m.lookup_params == 0 and m.vision_params == 0 and len(m.mtp_layers) == 1
    assert m.mtp_extra_params == m.vocab * m.hidden + 2 * m.hidden * m.hidden + m.hidden
    mt = m.mtp_layers[0]
    assert mt.core.kind == "gqa" and mt.core.kv_lora == 0
    assert mt.core.kv_elems_per_token() == 64 * (192 + 128)
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == 12288
    assert mt.misc_params == 2 * m.hidden + 1536 + 512 + 2 * m.hidden
    ops = _mtp_ops(m, Phase("decode", 1, 1, 16), Shard())
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params) == (2 * m.hidden, m.hidden, 2 * m.hidden * m.hidden)
    a = m.layers[0]
    assert a.ffn.n_experts == 512 and a.ffn.zero_experts == 256 and a.ffn.top_k == 12
    assert a.core.kv_elems_per_token() == 64 * (192 + 128)
    assert a.core.kv_elems_per_token() != 512 + 64
    router = next(l for l in a.ffn_linears if l.name == "router")
    assert (router.k, router.n) == (m.hidden, 768)
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    tokens = 768
    routed = build_rank_ops(m, 0, Phase("decode", tokens, 1, 64), Shard())
    gate = next(op for op in routed if op.name == "expert.gate_up")
    assert gate.count * gate.m * gate.share == tokens * 8


def test_hunyuan_a13b_text_moe():
    m = get_model("hunyuan-a13b-instruct")
    rel = load_release(m.hf_id)
    h, heads, kv, hd, layers = 4096, 32, 8, 128, 32
    experts, topk, width, vocab = 64, 8, 3072, 128167
    attn = 2 * h * heads * hd + 2 * h * kv * hd
    shared = 3 * h * width
    expert = experts * 3 * h * width
    router_p = h * experts
    misc = 2 * h + 2 * hd
    expect = layers * (attn + shared + expert + router_p + misc) + vocab * h + h
    assert expect == 80_393_183_232
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == expect
    assert m.mtp_params() == rel["params_mtp"] == 0 and m.mtp_layers == ()
    assert m.lookup_params == 0 and m.vision_params == 0 and m.vision is None
    assert m.coverage == "full" and m.tie_embeddings and m.hidden == h and m.vocab == vocab
    assert len(m.layers) == layers and all(L.ffn.kind == "moe" for L in m.layers)
    assert "lm_head" not in rel["roles"] and rel["params_vision"] == 0
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["params"] == layers * router_p
    assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * layers * router_p
    assert set(rel["roles"]["mlp"]) == {"bf16"}
    assert rel["roles"]["mlp"]["bf16"]["params"] == layers * shared
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    assert m.fmt("mlp").fmt == "bf16" and m.fmt("mlp").bits == 16.0
    assert m.fmt("expert").fmt == "bf16" and m.fmt("attn").fmt == "bf16"
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    L0 = m.layers[0]
    assert (L0.core.kind, L0.core.n_q, L0.core.n_kv, L0.core.qk_dim, L0.core.v_dim) == ("gqa", heads, kv, hd, hd)
    assert L0.core.kv_lora == 0 and L0.core.kv_elems_per_token() == kv * (hd + hd)
    assert L0.ffn.n_experts == experts and L0.ffn.top_k == topk and L0.ffn.d_expert == width
    assert L0.ffn.zero_experts == 0 and L0.ffn.n_shared == 1 and L0.ffn.d_shared == width
    assert L0.expert_params(h) == expert
    assert L0.misc_params == misc
    assert m.layers[-1].ffn.kind == "moe" and m.layers[-1].ffn.top_k == topk
    q = next(l for l in L0.attn_linears if l.name == "q")
    k = next(l for l in L0.attn_linears if l.name == "k")
    v = next(l for l in L0.attn_linears if l.name == "v")
    o = next(l for l in L0.attn_linears if l.name == "o")
    assert (q.k, q.n) == (h, heads * hd)
    assert (k.k, k.n) == (h, kv * hd) and (v.k, v.n) == (h, kv * hd)
    assert (o.k, o.n) == (heads * hd, h)
    router = next(l for l in L0.ffn_linears if l.name == "router")
    shared_up = next(l for l in L0.ffn_linears if l.name == "shared_gate_up")
    shared_down = next(l for l in L0.ffn_linears if l.name == "shared_down")
    assert (router.k, router.n, router.split) == (h, experts, "rep")
    assert (shared_up.k, shared_up.n) == (h, 2 * width)
    assert (shared_down.k, shared_down.n) == (width, h)
    assert not any(l.name == "shared_expert_gate" for l in L0.ffn_linears)
    # Active params keep the shared MLP and top-8 experts. The other 56 experts are idle.
    assert m.active_params() == expect - (experts - topk) * 3 * h * width * layers

    dec = build_rank_ops(m, 0, Phase("decode", 1, 1, 64), Shard())
    sm = next(op for op in dec if op.name == "softmax")
    assert sm.kv_read == 65 * kv * (hd + hd) * 2
    assert sm.kv_write == kv * (hd + hd) * 2
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.w_fmt, rt.a_fmt) == (h, experts, 32.0, "fp32", "bf16")
    sh = next(op for op in dec if op.name == "moe.shared_gate_up")
    assert (sh.m, sh.k, sh.n, sh.w_bits) == (1, h, 2 * width, 16.0)
    tokens = 128
    routed = build_rank_ops(m, 0, Phase("decode", tokens, 1, 32), Shard())
    gate = next(op for op in routed if op.name == "expert.gate_up")
    assert gate.count * gate.m * gate.share == tokens * topk
    assert _weights(m) - rel["bytes_llm"] == -(h * 2)

    ids = {d["id"]: d for d in list_models()}
    row = ids["hunyuan-a13b-instruct"]
    assert row["family"] == "Hunyuan-A13B" and row["provenance"] == "official"
    assert row["provider"] == "tencent"
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "hunyuan-a13b-instruct", "layout": {"tp": 2, "ep": 2},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_hunyuan_flags_and_refused_paths():
    rel = load_release("tencent/Hunyuan-A13B-Instruct")
    off = copy.deepcopy(rel["config"])
    off["use_mixed_mlp_moe"] = False
    off["use_qk_norm"] = False
    spec = from_release("synth", rel["repo"], rel=rel, cfg=off)
    layer = spec.layers[0]
    assert layer.ffn.n_shared == 0 and layer.ffn.d_shared == 0
    assert layer.misc_params == 2 * spec.hidden
    assert not any(l.name.startswith("shared") for l in layer.ffn_linears)
    assert layer.ffn.n_experts == 64 and layer.ffn.top_k == 8 and layer.ffn.d_expert == 3072
    for flag in ("use_cla", "use_mla"):
        bad = copy.deepcopy(rel["config"])
        bad[flag] = True
        try:
            from_release("synth", rel["repo"], rel=rel, cfg=bad)
        except ValueError as exc:
            assert flag in str(exc)
        else:
            raise AssertionError(flag)
