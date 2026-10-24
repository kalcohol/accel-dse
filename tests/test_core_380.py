"""0.380: 0.379 ERNIE-4.5-300B review holds; Mistral-Small-3.2 text + Pixtral vision close.

300B language geometry, unused MTP, and rel_err stay as billed in 0.379.
Mistral-Small-3.2 omits tie_word_embeddings; Mistral3Config defaults it to true
and the stored lm_head is the tied copy. Vision follows PixtralVisionModel and
Mistral3MultiModalProjector. Counts are the release headers (2026-10-24).
"""
from __future__ import annotations

import copy

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, head_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_ops


def _weights(m):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0).weights


def test_review_379_ernie_300b_unchanged():
    h, heads, kv, hd = 8192, 64, 8, 128
    experts, de, d_dense, vocab = 64, 3584, 28672, 103424
    n_dense, n_moe, layers = 3, 51, 54
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    language = (layers * attn + n_dense * 3 * h * d_dense + n_moe * experts * 3 * h * de
                + n_moe * (h * experts + experts) + layers * 2 * h + h + 2 * vocab * h)
    mtp = attn + 3 * h * d_dense + 4 * h + 2 * h * h
    base = get_model("ernie-4.5-300b-a47b-base-pt")
    pt = get_model("ernie-4.5-300b-a47b-pt")
    brel = load_release(base.hf_id)
    prel = load_release(pt.hf_id)
    assert language == 299_484_163_264 and mtp == 989_888_512
    assert "num_nextn_predict_layers" not in brel["config"]
    assert prel["config"]["num_nextn_predict_layers"] == 1
    assert brel["config"]["moe_num_shared_experts"] == 0
    assert brel["config"]["moe_layer_start_index"] == 3 and brel["config"]["moe_layer_end_index"] == 53
    assert base.param_check()["rel_err"] == 0.0 and pt.param_check()["rel_err"] == 0.0
    assert base.params() == pt.params() == brel["params_llm"] == prel["params_llm"] == language
    assert base.mtp_params() == brel["params_mtp"] == brel["bytes_mtp"] == 0
    assert pt.mtp_params() == prel["params_mtp"] == mtp
    assert brel["n_tensors"] == 10230 and prel["n_tensors"] == 10242
    assert brel["bytes_total"] == brel["bytes_llm"] == 599_021_810_432
    assert prel["bytes_total"] == 601_001_587_456 and prel["bytes_mtp"] == 2 * mtp
    assert all(layer.ffn.kind == "dense" and layer.ffn.d_ff == d_dense for layer in base.layers[:n_dense])
    assert all(layer.ffn.kind == "moe" and layer.ffn.n_experts == experts and layer.ffn.top_k == 8
               and layer.ffn.n_shared == 0 for layer in base.layers[n_dense:])
    assert len(base.layers) == layers and base.tie_embeddings is False and base.vision is None
    assert pt.mtp_layers[0].ffn.kind == "dense" and pt.mtp_layers[0].ffn.d_ff == d_dense
    assert (base.layers[0].core.n_q, base.layers[0].core.n_kv, base.layers[0].core.qk_dim) == (heads, kv, hd)


def test_mistral_small_3_2_text_and_pixtral_vision():
    h, heads, kv, hd, inter, vocab, layers = 5120, 32, 8, 128, 32768, 131072, 40
    q, kk = heads * hd, kv * hd
    attn = h * q + 2 * h * kk + q * h
    language = layers * (attn + 3 * h * inter + 2 * h) + vocab * h + h
    head = vocab * h
    vh, vi, vd, vp, vm, vheads, text_h = 1024, 4096, 24, 14, 2, 16, h
    vision = (3 * vp * vp * vh + vh + vd * (2 * vh + 4 * vh * vh + 3 * vh * vi)
              + vh + (vm * vm * vh) * vh + vh * text_h + text_h * text_h)
    assert language == 22_901_314_560 and head == 671_088_640 and vision == 438_958_080
    m = get_model("mistral-small-3.2-24b-instruct-2506")
    rel = load_release(m.hf_id)
    pc = m.param_check()
    assert pc["rel_err"] == 0.0 and m.params() == language == m.release_params
    assert rel["params_llm"] == language + head and rel["params_vision"] == vision
    assert rel["params_mtp"] == 0 and rel["bytes_mtp"] == 0 and rel["n_tensors"] == 585
    assert rel["bytes_llm"] == 47_144_806_400 and rel["bytes_total"] == 48_022_722_560
    assert rel["quantization_config"] is None and rel["torch_dtype"] == "bfloat16"
    assert "tie_word_embeddings" not in rel["config"] and "tie_word_embeddings" not in rel["config"]["text_config"]
    assert m.tie_embeddings is True and m.mtp_layers == () and m.mtp_params() == 0
    assert m.n_layers == layers and m.hidden == h and m.vocab == vocab
    assert m.arch == "GQA 稠密" and m.coverage == "full" and m.coverage_reasons == ()
    assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    assert m.fmt("attn").fmt == "bf16" and m.fmt("mlp").fmt == "bf16" and m.fmt("embed").fmt == "bf16"
    assert m.fmt("lm_head").fmt == "bf16" and m.fmt("lm_head").bits == 16.0
    assert all(L.ffn.kind == "dense" and L.ffn.d_ff == inter and L.core.window is None for L in m.layers)
    L0 = m.layers[0]
    assert (L0.core.n_q, L0.core.n_kv, L0.core.qk_dim, L0.core.v_dim) == (heads, kv, hd, hd)
    assert [(lin.name, lin.k, lin.n) for lin in L0.attn_linears] == [
        ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
    assert rel["roles"]["attn"]["bf16"]["params"] == layers * attn
    assert rel["roles"]["mlp"]["bf16"]["params"] == layers * 3 * h * inter
    assert rel["roles"]["embed"]["bf16"]["params"] == rel["roles"]["lm_head"]["bf16"]["params"] == head
    assert rel["roles"]["norm"]["bf16"]["params"] == layers * 2 * h + h
    assert set(rel["roles"]["vision"]) == {"bf16"} and rel["roles"]["vision"]["bf16"]["params"] == vision
    assert m.vision is not None and m.vision.params == vision == m.vision_params
    assert m.vision.family == "pixtral" and m.vision.depth == vd and m.vision.hidden == vh
    assert m.vision.heads == vheads and m.vision.patch == vp and m.vision.merge == vm
    assert m.vision.gated is True and m.vision.bias_qkv is False and m.vision.extra == 0
    assert m.vision.post == ((vm * vm * vh, vh), (vh, text_h), (text_h, text_h))
    assert dict(m.vision.pre) == {"factor": vp * vm, "longest": 1540}
    g = image_grid(m.vision, 1024, 1024)
    assert (g.px, g.ph, g.pw, g.merged, g.llm_tokens) == ((1036, 1036), 74, 74, 1369, 1406)
    wide = image_grid(m.vision, 1920, 1080)
    assert (wide.px, wide.ph, wide.pw, wide.merged, wide.llm_tokens) == ((1540, 868), 62, 110, 1705, 1736)
    ops = vision_ops(m.vision, g, 1)
    pe = next(o for o in ops if o.name == "vis.patch_embed")
    assert (pe.m, pe.k, pe.n) == (74 * 74, 3 * vp * vp, vh)
    qkv = next(o for o in ops if o.name == "vis.0.qkv")
    assert (qkv.m, qkv.k, qkv.n) == (74 * 74, vh, 3 * vh)
    mlp = next(o for o in ops if o.name == "vis.0.mlp_in")
    assert (mlp.m, mlp.k, mlp.n) == (74 * 74, vh, 2 * vi)
    for i, kn in enumerate(m.vision.post):
        gemm = next(o for o in ops if o.name == f"vis.merger.{i}")
        assert (gemm.m, gemm.k, gemm.n) == (g.merged, kn[0], kn[1])
    attn_op = next(o for o in ops if o.name == "vis.0.qk")
    assert (attn_op.m, attn_op.n, attn_op.count) == (74 * 74, 74 * 74, vheads)
    assert _weights(m) - rel["bytes_llm"] == -2 * (head + h)
    head_op = next(o for o in head_ops(m, Phase("decode", 1, 1, 8)) if o.name == "lm_head")
    assert head_op.role == "embed" and (head_op.k, head_op.n) == (h, vocab)
    untied = copy.deepcopy(rel["config"])
    untied["tie_word_embeddings"] = False
    spec = from_release("synth", rel["repo"], rel=rel, cfg=untied)
    assert spec.tie_embeddings is False and spec.params() == language + head
    assert spec.param_check()["rel_err"] == 0.0
    ids = {d["id"]: d for d in list_models()}
    row = ids["mistral-small-3.2-24b-instruct-2506"]
    assert row["family"] == "Mistral-Small" and row["provenance"] == "official"
    assert row["provider"] == "mistral" and row["domain"] == "vlm"
    assert row["mtp_layers"] == 0 and row["param_err"] == 0.0 and row["coverage"] == "full"
    r = evaluate(Scenario(model=m.id, serving=Serving(phase="prefill", prompt=128, images=1)))
    assert r.vision["image_tokens"] == 1406 and r.vision["resized_px"] == [1036, 1036]
    evaluate(Scenario(model=m.id, serving=Serving(phase="decode", batch=2, ctx=128)))
    for ph, extra in (("prefill", {"prompt": 64, "images": 1}), ("decode", {"batch": 2})):
        s = A.api_eval({"scenario": {"model": m.id, "serving": {"phase": ph, **extra}},
                        "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0
