"""0.377: Hunyuan-A13B and LongCat review holds; ERNIE-4.5-21B-A3B closes.

Hunyuan stays a text MoE with an fp32 gate.wg router and a bf16 shared SwiGLU.
LongCat MTP is the released dense block with V padded to qk_head_dim. ERNIE layer 0 is dense, layers
1-27 are MoE with one shared SwiGLU, and the stored MTP block is that dense
shape rather than a clone of the last MoE layer.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release


def _weights(m, n_mtp=0):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=n_mtp).weights


def _summarize():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_release.py"
    spec = importlib.util.spec_from_file_location("summarize_release", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_review_376_accounts_hold():
    hun = get_model("hunyuan-a13b-instruct")
    hrel = load_release(hun.hf_id)
    h, heads, kv, hd, layers = 4096, 32, 8, 128, 32
    experts, topk, width, vocab = 64, 8, 3072, 128167
    attn = 2 * h * heads * hd + 2 * h * kv * hd
    expect = layers * (attn + 3 * h * width + experts * 3 * h * width + h * experts + 2 * h + 2 * hd) + vocab * h + h
    assert expect == 80_393_183_232
    assert hun.param_check()["rel_err"] == 0.0
    assert hun.params() == hun.release_params == hrel["params_llm"] == expect
    assert hun.mtp_params() == 0 and hun.vision_params == 0 and hun.vision is None
    assert set(hrel["roles"]["router"]) == {"fp32"}
    assert hrel["roles"]["router"]["fp32"]["params"] == layers * h * experts == 8_388_608
    assert hrel["roles"]["router"]["fp32"]["bytes"] == 4 * layers * h * experts
    assert set(hrel["roles"]["mlp"]) == {"bf16"}
    assert hrel["roles"]["mlp"]["bf16"]["params"] == layers * 3 * h * width
    assert hun.fmt("router").fmt == "fp32" and hun.fmt("mlp").fmt == "bf16" and hun.act_fmt == "bf16"
    L0 = hun.layers[0]
    assert (L0.core.n_q, L0.core.n_kv, L0.core.qk_dim) == (heads, kv, hd)
    assert L0.core.kv_elems_per_token() == kv * (hd + hd)
    assert L0.ffn.n_experts == experts and L0.ffn.top_k == topk and L0.ffn.d_expert == width
    assert L0.ffn.n_shared == 1 and L0.ffn.d_shared == width
    assert not any(l.name == "shared_expert_gate" for l in L0.ffn_linears)
    assert all(layer.ffn.kind == "moe" for layer in hun.layers)

    cat = get_model("longcat-flash-chat")
    crel = load_release(cat.hf_id)
    assert cat.param_check()["rel_err"] == 0.0
    assert cat.params() == 560_664_980_480
    assert cat.mtp_params() == crel["params_mtp"] == 1_197_899_776
    assert len(cat.mtp_layers) == 1 and cat.mtp_layers[0].ffn.kind == "dense"
    assert cat.mtp_layers[0].core.kv_lora == 0
    assert cat.mtp_extra_params == cat.vocab * cat.hidden + 2 * cat.hidden * cat.hidden + cat.hidden
    eh = next(op for op in _mtp_ops(cat, Phase("decode", 1, 1, 16), Shard()) if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params) == (2 * cat.hidden, cat.hidden, 2 * cat.hidden * cat.hidden)
    assert cat.fmt("router").fmt == "fp32" and cat.fmt("router").bits == 32.0

    role = _summarize().classify
    assert role("model.layers.0.mlp.gate.wg.weight", 32) == "router"
    assert role("model.mtp.layers.0.eh_proj.weight", 56) == "mtp"
    assert role("model.mtp.embed_tokens.weight", 56) == "mtp"


def test_ernie_21b_a3b_closes():
    m = get_model("ernie-4.5-21b-a3b-pt")
    rel = load_release(m.hf_id)
    h, heads, kv, hd, layers = 2560, 20, 4, 128, 28
    experts, topk, de, d_dense, vocab = 64, 6, 1536, 12288, 103424
    n_moe = 27
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    dense_ffn = 3 * h * d_dense
    shared = 3 * h * (2 * de)
    expert = experts * 3 * h * de
    router_w = h * experts
    dense_layer = attn + dense_ffn + 2 * h
    moe_layer = attn + expert + shared + router_w + experts + 2 * h
    expect = dense_layer + n_moe * moe_layer + vocab * h + h
    mtp_layer = attn + dense_ffn + 4 * h
    mtp_expect = mtp_layer + 2 * h * h
    assert attn == 15_728_640 and dense_ffn == 94_371_840
    assert expect == 21_825_437_888 and mtp_expect == 123_217_920
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == expect
    assert m.mtp_params() == rel["params_mtp"] == mtp_expect
    assert m.mtp_extra_params == 2 * h * h
    assert m.lookup_params == 0 and m.vision_params == 0 and m.vision is None
    assert m.coverage == "full" and m.tie_embeddings and m.hidden == h and m.vocab == vocab
    assert len(m.layers) == layers and len(m.mtp_layers) == 1
    assert "lm_head" not in rel["roles"] and rel["params_vision"] == 0
    assert rel["n_tensors"] == 5504 and rel["bytes_total"] == rel["bytes_llm"] + rel["bytes_mtp"]
    assert rel["torch_dtype"] == "bfloat16" and rel["quantization_config"] is None
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["params"] == n_moe * (router_w + experts) == 4_425_408
    assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * n_moe * (router_w + experts)
    assert set(rel["roles"]["mlp"]) == {"bf16"}
    assert rel["roles"]["mlp"]["bf16"]["params"] == dense_ffn
    assert set(rel["roles"]["shared_expert"]) == {"bf16"}
    assert rel["roles"]["shared_expert"]["bf16"]["params"] == n_moe * shared
    assert set(rel["roles"]["expert"]) == {"bf16"}
    assert rel["roles"]["expert"]["bf16"]["params"] == n_moe * expert
    assert set(rel["roles"]["mtp"]) == {"bf16"}
    assert rel["roles"]["mtp"]["bf16"]["params"] == mtp_expect
    assert rel["roles"]["mtp"]["bf16"]["bytes"] == 2 * mtp_expect
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    assert m.fmt("mlp").fmt == "bf16" and m.fmt("shared_expert").fmt == "bf16"
    assert m.fmt("expert").fmt == "bf16" and m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"

    dense, moe = m.layers[0], m.layers[1]
    assert dense.ffn.kind == "dense" and dense.ffn.d_ff == d_dense and dense.misc_params == 2 * h
    assert all(layer.ffn.kind == "moe" for layer in m.layers[1:])
    assert (moe.core.kind, moe.core.n_q, moe.core.n_kv, moe.core.qk_dim, moe.core.v_dim) == ("gqa", heads, kv, hd, hd)
    assert moe.core.kv_lora == 0 and moe.core.kv_elems_per_token() == kv * (hd + hd)
    assert moe.ffn.n_experts == experts and moe.ffn.top_k == topk and moe.ffn.d_expert == de
    assert moe.ffn.n_shared == 1 and moe.ffn.d_shared == 2 * de and moe.ffn.zero_experts == 0
    assert moe.misc_params == 2 * h + experts
    assert moe.expert_params(h) == expert
    assert not any(l.name == "shared_expert_gate" for l in moe.ffn_linears)
    q = next(l for l in moe.attn_linears if l.name == "q")
    k = next(l for l in moe.attn_linears if l.name == "k")
    o = next(l for l in moe.attn_linears if l.name == "o")
    assert (q.k, q.n) == (h, heads * hd)
    assert (k.k, k.n) == (h, kv * hd)
    assert (o.k, o.n) == (heads * hd, h)
    router = next(l for l in moe.ffn_linears if l.name == "router")
    shared_up = next(l for l in moe.ffn_linears if l.name == "shared_gate_up")
    shared_down = next(l for l in moe.ffn_linears if l.name == "shared_down")
    assert (router.k, router.n) == (h, experts)
    assert (shared_up.k, shared_up.n) == (h, 2 * (2 * de))
    assert (shared_down.k, shared_down.n) == (2 * de, h)

    mt = m.mtp_layers[0]
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == d_dense and mt.misc_params == 4 * h
    assert mt.core.kv_elems_per_token() == kv * (hd + hd)
    assert mt.params(h) == mtp_layer
    # A clone of the last MoE layer is several times the stored MTP block.
    assert m.layers[-1].params(h) > 5 * mt.params(h)
    ops = _mtp_ops(m, Phase("decode", 1, 1, 16), Shard())
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_bits, eh.w_fmt, eh.a_fmt) == (2 * h, h, 2 * h * h, 16.0, "bf16", "bf16")

    dec = build_rank_ops(m, 1, Phase("decode", 1, 1, 64), Shard())
    sm = next(op for op in dec if op.name == "softmax")
    assert sm.kv_read == 65 * kv * (hd + hd) * 2
    assert sm.kv_write == kv * (hd + hd) * 2
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.w_fmt, rt.a_fmt) == (h, experts, 32.0, "fp32", "bf16")
    sh = next(op for op in dec if op.name == "moe.shared_gate_up")
    assert (sh.m, sh.k, sh.n, sh.w_bits) == (1, h, 4 * de, 16.0)
    tokens = 96
    routed = build_rank_ops(m, 1, Phase("decode", tokens, 1, 32), Shard())
    gate = next(op for op in routed if op.name == "expert.gate_up")
    assert gate.count * gate.m * gate.share == tokens * topk
    dense_ops = build_rank_ops(m, 0, Phase("decode", 1, 1, 8), Shard())
    up = next(op for op in dense_ops if op.name.endswith("gate_up"))
    assert (up.k, up.n, up.w_bits) == (h, 2 * d_dense, 16.0)
    assert not any(op.name == "moe.router" for op in dense_ops)

    idle = (experts - topk) * 3 * h * de * n_moe
    assert m.active_params() == expect - idle == 3_352_150_208
    assert _weights(m) - rel["bytes_llm"] == -(n_moe * experts * 2 + h * 2)

    ids = {d["id"]: d for d in list_models()}
    row = ids["ernie-4.5-21b-a3b-pt"]
    assert row["family"] == "ERNIE-4.5" and row["provenance"] == "official"
    assert row["provider"] == "baidu"
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "ernie-4.5-21b-a3b-pt", "layout": {"tp": 2, "ep": 2},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0

    role = _summarize().classify
    assert role("model.mtp_block.0.self_attn.q_proj.weight", layers) == "mtp"
    assert role("model.mtp_linear_proj.0.weight", layers) == "mtp"
    assert role("model.mtp_emb_norm.0.weight", layers) == "mtp"
    assert role("model.mtp_hidden_norm.0.weight", layers) == "mtp"
    assert role("model.layers.1.mlp.moe_statics.e_score_correction_bias", layers) == "router"
    assert role("model.layers.1.mlp.gate.weight", layers) == "router"
    assert role("model.layers.0.mlp.gate_proj.weight", layers) == "mlp"
    assert role("model.layers.1.mlp.shared_experts.down_proj.weight", layers) == "shared_expert"
    assert role("model.layers.1.mlp.experts.3.gate_proj.weight", layers) == "expert"


def test_ernie_flags_and_refused_paths():
    rel = load_release("baidu/ERNIE-4.5-21B-A3B-PT")
    start = copy.deepcopy(rel["config"])
    start["moe_layer_start_index"] = 0
    spec = from_release("synth", rel["repo"], rel=rel, cfg=start)
    assert spec.layers[0].ffn.kind == "moe" and spec.layers[0].ffn.top_k == 6
    assert spec.layers[0].ffn.n_shared == 1 and spec.layers[0].ffn.d_shared == 3072

    bare = copy.deepcopy(rel["config"])
    bare["moe_num_shared_experts"] = 0
    spec = from_release("synth", rel["repo"], rel=rel, cfg=bare)
    layer = spec.layers[1]
    assert layer.ffn.kind == "moe" and layer.ffn.n_shared == 0 and layer.ffn.d_shared == 0
    assert layer.misc_params == 2 * spec.hidden + 64
    assert not any(l.name.startswith("shared") for l in layer.ffn_linears)
    assert layer.ffn.n_experts == 64 and layer.ffn.d_expert == 1536

    step = copy.deepcopy(rel["config"])
    step["moe_layer_interval"] = 2
    spec = from_release("synth", rel["repo"], rel=rel, cfg=step)
    moe_at = [i for i, layer in enumerate(spec.layers) if layer.ffn.kind == "moe"]
    assert moe_at == list(range(1, 28, 2))

    biased = copy.deepcopy(rel["config"])
    biased["use_bias"] = True
    try:
        from_release("synth", rel["repo"], rel=rel, cfg=biased)
    except ValueError as exc:
        assert "use_bias" in str(exc)
    else:
        raise AssertionError("use_bias")
