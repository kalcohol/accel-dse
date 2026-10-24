"""0.378: ERNIE-4.5-21B-A3B-PT review holds; Base-PT closes without an MTP module.

PT language geometry, router dtype, unused dense MTP block, and the n_mtp=0
stage gap stay as released. Base-PT has the same language tensors and no
mtp weights. Its config omits num_nextn_predict_layers, and a forced 1 is
dropped because the header summary has params_mtp == 0.
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


def test_review_377_ernie_pt_holds():
    m = get_model("ernie-4.5-21b-a3b-pt")
    rel = load_release(m.hf_id)
    h, heads, kv, hd = 2560, 20, 4, 128
    experts, topk, de, d_dense, vocab, n_moe = 64, 6, 1536, 12288, 103424, 27
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    dense_ffn = 3 * h * d_dense
    shared = 3 * h * (2 * de)
    expert = experts * 3 * h * de
    router = n_moe * (h * experts + experts)
    expect = (attn + dense_ffn + 2 * h) + n_moe * (attn + expert + shared + h * experts + experts + 2 * h)
    expect += vocab * h + h
    mtp_layer = attn + dense_ffn + 4 * h
    mtp_expect = mtp_layer + 2 * h * h
    assert attn == 15_728_640 and dense_ffn == 94_371_840 and shared == 23_592_960
    assert expect == 21_825_437_888 and mtp_expect == 123_217_920 and router == 4_425_408
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == expect
    assert m.mtp_params() == rel["params_mtp"] == mtp_expect
    assert m.mtp_extra_params == 2 * h * h
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["params"] == router
    assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * router
    assert set(rel["roles"]["mtp"]) == {"bf16"}
    assert rel["roles"]["mtp"]["bf16"]["params"] == mtp_expect
    assert rel["roles"]["mtp"]["bf16"]["bytes"] == 2 * mtp_expect
    assert m.fmt("router").bits == 32.0 and m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    assert m.layers[0].ffn.kind == "dense" and m.layers[0].ffn.d_ff == d_dense
    assert all(layer.ffn.kind == "moe" for layer in m.layers[1:])
    moe = m.layers[1]
    assert (moe.core.n_q, moe.core.n_kv, moe.core.qk_dim, moe.core.v_dim) == (heads, kv, hd, hd)
    assert moe.ffn.n_experts == experts and moe.ffn.top_k == topk and moe.ffn.d_expert == de
    assert moe.ffn.n_shared == 1 and moe.ffn.d_shared == 2 * de
    assert not any(lin.name == "shared_expert_gate" for lin in moe.ffn_linears)
    mt = m.mtp_layers[0]
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == d_dense and mt.params(h) == mtp_layer
    assert m.layers[-1].ffn.kind == "moe" and m.layers[-1].params(h) > 5 * mt.params(h)
    eh = next(op for op in _mtp_ops(m, Phase("decode", 1, 1, 16), Shard()) if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_bits, eh.w_fmt, eh.a_fmt) == (2 * h, h, 2 * h * h, 16.0, "bf16", "bf16")
    dec = build_rank_ops(m, 1, Phase("decode", 1, 1, 64), Shard())
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.w_fmt, rt.a_fmt) == (h, experts, 32.0, "fp32", "bf16")
    gap = n_moe * experts * 2 + h * 2
    assert gap == 8576
    assert _weights(m) - rel["bytes_llm"] == -gap

    biased = copy.deepcopy(rel["config"])
    biased["use_bias"] = True
    try:
        from_release("synth", rel["repo"], rel=rel, cfg=biased)
    except ValueError:
        pass
    else:
        raise AssertionError("use_bias")


def test_ernie_21b_a3b_base_pt_closes():
    m = get_model("ernie-4.5-21b-a3b-base-pt")
    rel = load_release(m.hf_id)
    pt = get_model("ernie-4.5-21b-a3b-pt")
    ptrel = load_release(pt.hf_id)
    h = 2560
    assert "num_nextn_predict_layers" not in rel["config"]
    assert rel["config"]["moe_layer_start_index"] == 1 and rel["config"]["moe_layer_end_index"] == 27
    assert rel["config"]["moe_num_shared_experts"] == 2 and rel["config"]["use_bias"] is False
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == pt.params() == 21_825_437_888
    assert m.mtp_params() == rel["params_mtp"] == rel["bytes_mtp"] == 0
    assert m.mtp_layers == () and m.mtp_extra_params == 0
    assert m.lookup_params == 0 and m.vision_params == 0 and m.vision is None
    assert m.coverage == "full" and m.tie_embeddings and m.hidden == h and m.vocab == 103424
    assert len(m.layers) == 28
    assert rel["n_tensors"] == 5492 and ptrel["n_tensors"] == 5504
    assert rel["bytes_total"] == rel["bytes_llm"] == ptrel["bytes_llm"] == 43_659_726_592
    assert rel["bytes_total"] == ptrel["bytes_total"] - ptrel["bytes_mtp"]
    assert "mtp" not in rel["roles"] and "lm_head" not in rel["roles"]
    for role in ("attn", "embed", "expert", "mlp", "norm", "router", "shared_expert"):
        assert rel["roles"][role] == ptrel["roles"][role]
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["params"] == 4_425_408
    assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * 4_425_408
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    assert m.fmt("expert").fmt == "bf16" and m.fmt("shared_expert").fmt == "bf16" and m.fmt("mlp").fmt == "bf16"
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    assert all(a.params(h) == b.params(h) and a.ffn.kind == b.ffn.kind for a, b in zip(m.layers, pt.layers))
    dense, moe = m.layers[0], m.layers[1]
    assert dense.ffn.kind == "dense" and dense.ffn.d_ff == 12288 and dense.misc_params == 2 * h
    assert moe.ffn.kind == "moe" and moe.ffn.n_experts == 64 and moe.ffn.top_k == 6
    assert moe.ffn.d_expert == 1536 and moe.ffn.n_shared == 1 and moe.ffn.d_shared == 3072
    assert moe.core.kv_elems_per_token() == 4 * (128 + 128)
    assert not any(lin.name == "shared_expert_gate" for lin in moe.ffn_linears)
    dec = build_rank_ops(m, 1, Phase("decode", 1, 1, 32), Shard())
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.a_fmt) == (h, 64, 32.0, "bf16")
    sh = next(op for op in dec if op.name == "moe.shared_gate_up")
    assert (sh.k, sh.n, sh.w_bits) == (h, 2 * 3072, 16.0)
    dense_ops = build_rank_ops(m, 0, Phase("decode", 1, 1, 8), Shard())
    up = next(op for op in dense_ops if op.name.endswith("gate_up"))
    assert (up.k, up.n, up.w_bits) == (h, 2 * 12288, 16.0)
    assert not any(op.name == "moe.router" for op in dense_ops)
    idle = (64 - 6) * 3 * h * 1536 * 27
    assert m.active_params() == pt.active_params() == 21_825_437_888 - idle == 3_352_150_208
    assert _weights(m) - rel["bytes_llm"] == -(27 * 64 * 2 + h * 2) == -8576

    invented = copy.deepcopy(rel["config"])
    invented["num_nextn_predict_layers"] = 1
    spec = from_release("synth", rel["repo"], rel=rel, cfg=invented)
    assert spec.mtp_layers == () and spec.mtp_params() == 0

    ids = {d["id"]: d for d in list_models()}
    row = ids["ernie-4.5-21b-a3b-base-pt"]
    assert row["family"] == "ERNIE-4.5" and row["provenance"] == "official"
    assert row["provider"] == "baidu" and row["mtp_layers"] == 0 and row["param_err"] == 0.0
    for ph in ("decode", "prefill"):
        s = A.api_eval({"scenario": {"model": "ernie-4.5-21b-a3b-base-pt",
                                     "layout": {"tp": 2, "ep": 2},
                                     "serving": {"phase": ph, "batch": 2}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0
