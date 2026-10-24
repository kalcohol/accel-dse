"""0.379: 0.378 ERNIE Base-PT review holds; 300B-A47B PT and Base-PT close.

Base-PT language tensors match the PT checkpoint with the MTP block removed.
300B uses the same public class: dense layers 0-2, MoE from layer 3, no shared
experts, untied lm_head. Base has no MTP weights. A forced next-n count is
dropped when the header summary has params_mtp == 0.
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


def test_review_378_base_pt_matches_pt_language():
    base = get_model("ernie-4.5-21b-a3b-base-pt")
    pt = get_model("ernie-4.5-21b-a3b-pt")
    brel = load_release(base.hf_id)
    prel = load_release(pt.hf_id)
    h, heads, kv, hd = 2560, 20, 4, 128
    experts, de, d_dense, vocab, n_moe = 64, 1536, 12288, 103424, 27
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    dense_ffn = 3 * h * d_dense
    shared = 3 * h * (2 * de)
    expert = experts * 3 * h * de
    router = n_moe * (h * experts + experts)
    language = (attn + dense_ffn + 2 * h) + n_moe * (attn + expert + shared + h * experts + experts + 2 * h)
    language += vocab * h + h
    mtp = attn + dense_ffn + 4 * h + 2 * h * h
    assert language == 21_825_437_888 and mtp == 123_217_920 and router == 4_425_408
    assert "num_nextn_predict_layers" not in brel["config"]
    assert prel["config"]["num_nextn_predict_layers"] == 1
    assert base.param_check()["rel_err"] == 0.0 and pt.param_check()["rel_err"] == 0.0
    assert base.params() == pt.params() == brel["params_llm"] == prel["params_llm"] == language
    assert base.mtp_params() == brel["params_mtp"] == brel["bytes_mtp"] == 0
    assert pt.mtp_params() == prel["params_mtp"] == mtp
    assert brel["n_tensors"] == 5492 and prel["n_tensors"] == 5504
    assert brel["bytes_total"] == brel["bytes_llm"] == prel["bytes_llm"] == 43_659_726_592
    assert prel["bytes_mtp"] == 2 * mtp == 246_435_840
    assert brel["bytes_total"] == prel["bytes_total"] - prel["bytes_mtp"]
    assert "mtp" not in brel["roles"]
    assert set(prel["roles"]["mtp"]) == {"bf16"}
    assert prel["roles"]["mtp"]["bf16"]["params"] == mtp
    assert prel["roles"]["router"]["fp32"]["params"] == brel["roles"]["router"]["fp32"]["params"] == router
    assert brel["roles"]["router"]["fp32"]["bytes"] == 4 * router
    for role in ("attn", "embed", "expert", "mlp", "norm", "router", "shared_expert"):
        assert brel["roles"][role] == prel["roles"][role]
    assert base.layers[0].ffn.kind == "dense" and base.layers[0].ffn.d_ff == d_dense
    assert all(layer.ffn.kind == "moe" and layer.ffn.n_shared == 1 and layer.ffn.d_shared == 2 * de
               for layer in base.layers[1:])
    assert all(a.params(h) == b.params(h) for a, b in zip(base.layers, pt.layers))
    assert pt.mtp_layers[0].ffn.kind == "dense" and pt.mtp_layers[0].ffn.d_ff == d_dense
    assert _weights(base) - brel["bytes_llm"] == _weights(pt) - prel["bytes_llm"] == -(n_moe * experts * 2 + h * 2)
    assert n_moe * experts * 2 + h * 2 == 8576
    assert base.fmt("router").bits == 32.0 and pt.fmt("router").bits == 32.0


def _expect_300b():
    h, heads, kv, hd = 8192, 64, 8, 128
    experts, topk, de, d_dense, vocab = 64, 8, 3584, 28672, 103424
    n_dense, n_moe, layers = 3, 51, 54
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    dense_ffn = 3 * h * d_dense
    expert = experts * 3 * h * de
    router = n_moe * (h * experts + experts)
    language = (layers * attn + n_dense * dense_ffn + n_moe * expert + router
                + layers * 2 * h + h + 2 * vocab * h)
    mtp_layer = attn + dense_ffn + 4 * h
    mtp = mtp_layer + 2 * h * h
    idle = (experts - topk) * 3 * h * de * n_moe
    return {
        "h": h, "heads": heads, "kv": kv, "hd": hd, "experts": experts, "topk": topk,
        "de": de, "d_dense": d_dense, "vocab": vocab, "n_dense": n_dense, "n_moe": n_moe,
        "layers": layers, "attn": attn, "router": router, "language": language,
        "mtp_layer": mtp_layer, "mtp": mtp, "idle": idle,
        "gap": n_moe * experts * 2 + h * 2,
    }


def test_ernie_300b_a47b_base_and_pt_close():
    g = _expect_300b()
    base = get_model("ernie-4.5-300b-a47b-base-pt")
    pt = get_model("ernie-4.5-300b-a47b-pt")
    brel = load_release(base.hf_id)
    prel = load_release(pt.hf_id)
    assert g["language"] == 299_484_163_264 and g["mtp"] == 989_888_512 and g["router"] == 26_741_952
    assert g["gap"] == 22912
    assert "num_nextn_predict_layers" not in brel["config"]
    assert prel["config"]["num_nextn_predict_layers"] == 1
    assert brel["config"]["moe_num_shared_experts"] == 0 and brel["config"]["moe_layer_start_index"] == 3
    assert brel["config"]["moe_layer_end_index"] == 53 and brel["config"]["tie_word_embeddings"] is False
    assert brel["config"]["use_bias"] is False and prel["config"]["moe_k"] == 8
    assert base.param_check()["rel_err"] == 0.0 and pt.param_check()["rel_err"] == 0.0
    assert base.params() == pt.params() == brel["params_llm"] == prel["params_llm"] == g["language"]
    assert base.mtp_params() == brel["params_mtp"] == brel["bytes_mtp"] == 0
    assert base.mtp_layers == () and base.mtp_extra_params == 0
    assert pt.mtp_params() == prel["params_mtp"] == g["mtp"]
    assert pt.mtp_extra_params == 2 * g["h"] * g["h"]
    assert brel["n_tensors"] == 10230 and prel["n_tensors"] == 10242
    assert brel["bytes_total"] == brel["bytes_llm"] == prel["bytes_llm"] == 599_021_810_432
    assert prel["bytes_mtp"] == 2 * g["mtp"] == 1_979_777_024
    assert prel["bytes_total"] == 601_001_587_456
    assert brel["bytes_total"] == prel["bytes_total"] - prel["bytes_mtp"]
    assert "mtp" not in brel["roles"] and "shared_expert" not in brel["roles"]
    assert "shared_expert" not in prel["roles"]
    assert set(prel["roles"]["mtp"]) == {"bf16"} and prel["roles"]["mtp"]["bf16"]["params"] == g["mtp"]
    assert set(brel["roles"]["router"]) == {"fp32"}
    assert brel["roles"]["router"]["fp32"]["params"] == prel["roles"]["router"]["fp32"]["params"] == g["router"]
    assert brel["roles"]["router"]["fp32"]["bytes"] == 4 * g["router"]
    assert brel["roles"]["lm_head"]["bf16"]["params"] == g["vocab"] * g["h"]
    for role in ("attn", "embed", "expert", "lm_head", "mlp", "norm", "router"):
        assert brel["roles"][role] == prel["roles"][role]
    assert len(base.layers) == g["layers"] == len(pt.layers)
    assert all(layer.ffn.kind == "dense" and layer.ffn.d_ff == g["d_dense"] and layer.misc_params == 2 * g["h"]
               for layer in base.layers[:g["n_dense"]])
    assert all(layer.ffn.kind == "moe" and layer.ffn.n_experts == g["experts"] and layer.ffn.top_k == g["topk"]
               and layer.ffn.d_expert == g["de"] and layer.ffn.n_shared == 0 and layer.ffn.d_shared == 0
               for layer in base.layers[g["n_dense"]:])
    assert all(a.params(g["h"]) == b.params(g["h"]) and a.ffn.kind == b.ffn.kind
               for a, b in zip(base.layers, pt.layers))
    moe = base.layers[3]
    assert (moe.core.n_q, moe.core.n_kv, moe.core.qk_dim, moe.core.v_dim) == (g["heads"], g["kv"], g["hd"], g["hd"])
    assert moe.core.kv_elems_per_token() == g["kv"] * (g["hd"] + g["hd"])
    assert not any(lin.name == "shared_expert_gate" for lin in moe.ffn_linears)
    assert base.tie_embeddings is False and base.vision is None and base.coverage == "full"
    assert base.fmt("router").fmt == "fp32" and base.fmt("expert").fmt == "bf16"
    assert base.fmt("lm_head").fmt == "bf16" and base.act_fmt == "bf16" and base.kv_fmt == "bf16"
    assert base.arch == "GQA MoE（无共享专家）"
    assert get_model("ernie-4.5-21b-a3b-base-pt").arch == "GQA + 共享 MLP 的 MoE"
    dec = build_rank_ops(base, 3, Phase("decode", 1, 1, 32), Shard())
    rt = next(op for op in dec if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_bits, rt.w_fmt, rt.a_fmt) == (g["h"], g["experts"], 32.0, "fp32", "bf16")
    assert not any(op.name.startswith("moe.shared") for op in dec)
    dense_ops = build_rank_ops(base, 2, Phase("decode", 1, 1, 8), Shard())
    up = next(op for op in dense_ops if op.name.endswith("gate_up"))
    assert (up.k, up.n, up.w_bits) == (g["h"], 2 * g["d_dense"], 16.0)
    assert not any(op.name == "moe.router" for op in dense_ops)
    assert base.active_params() == pt.active_params() == g["language"] - g["idle"] == 47_926_586_560
    assert _weights(base) - brel["bytes_llm"] == _weights(pt) - prel["bytes_llm"] == -g["gap"]
    mt = pt.mtp_layers[0]
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == g["d_dense"] and mt.params(g["h"]) == g["mtp_layer"]
    assert pt.layers[-1].ffn.kind == "moe" and pt.layers[-1].params(g["h"]) > 5 * mt.params(g["h"])
    eh = next(op for op in _mtp_ops(pt, Phase("decode", 1, 1, 16), Shard()) if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_bits, eh.w_fmt, eh.a_fmt) == (
        2 * g["h"], g["h"], 2 * g["h"] * g["h"], 16.0, "bf16", "bf16")

    invented = copy.deepcopy(brel["config"])
    invented["num_nextn_predict_layers"] = 1
    spec = from_release("synth", brel["repo"], rel=brel, cfg=invented)
    assert spec.mtp_layers == () and spec.mtp_params() == 0

    biased = copy.deepcopy(brel["config"])
    biased["use_bias"] = True
    try:
        from_release("synth", brel["repo"], rel=brel, cfg=biased)
    except ValueError:
        pass
    else:
        raise AssertionError("use_bias")

    ids = {d["id"]: d for d in list_models()}
    for mid, mtp_n in (("ernie-4.5-300b-a47b-base-pt", 0), ("ernie-4.5-300b-a47b-pt", 1)):
        row = ids[mid]
        assert row["family"] == "ERNIE-4.5" and row["provenance"] == "official"
        assert row["provider"] == "baidu" and row["mtp_layers"] == mtp_n and row["param_err"] == 0.0
        for ph in ("decode", "prefill"):
            s = A.api_eval({"scenario": {"model": mid,
                                         "layout": {"tp": 2, "ep": 2},
                                         "serving": {"phase": ph, "batch": 2}},
                            "sensitivity": False})["summary"]
            assert s["step_ms"] > 0 and s["tok_s"] > 0
