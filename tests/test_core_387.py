"""0.387: 0.386 granite-4.2-30b review holds; LFM2.5-2.6B-Base closes.

granite-4.2-30b keeps the 4.1-30b GEMM shapes, an untied lm_head, and the
published multipliers. LFM2.5-2.6B-Base matches the Instruct 2.6B tensor
names, dtypes, and shapes. max_position_embeddings is 128000.
"""
from __future__ import annotations

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving


def test_review_386_granite_42_30b_holds():
    h, heads, kv, inter, vocab, layers = 4096, 32, 8, 32768, 100352, 64
    hd = h // heads
    q, kk = heads * hd, kv * hd
    attn = h * q + 2 * h * kk + q * h
    ffn = 3 * h * inter
    norms = layers * 2 * h + h
    language = layers * (attn + ffn) + norms + 2 * vocab * h
    assert language == 29_276_770_304
    mid = "granite-4.2-30b"
    m = get_model(mid)
    prev = get_model("granite-4.1-30b")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params
    assert m.params() == prev.params() + vocab * h
    assert rel["n_tensors"] == 579 and rel["params_mtp"] == 0 and rel["params_vision"] == 0
    assert rel["bytes_llm"] == rel["bytes_total"] == 58_553_540_608
    assert rel["torch_dtype"] == "bfloat16" and rel["quantization_config"] is None
    assert rel["config"]["tie_word_embeddings"] is False
    assert "head_dim" not in rel["config"] and "num_nextn_predict_layers" not in rel["config"]
    assert rel["config"]["attention_multiplier"] == 0.0078125
    assert rel["config"]["embedding_multiplier"] == 1.0
    assert rel["config"]["residual_multiplier"] == 1.0
    assert rel["config"]["logits_scaling"] == 1.0
    assert rel["config"]["attention_bias"] is False and rel["config"]["mlp_bias"] is False
    assert m.tie_embeddings is False and m.mtp_layers == () and m.vision is None and m.coverage == "full"
    assert m.n_layers == layers and m.hidden == h and m.kv_fmt == "bf16" and m.act_fmt == "bf16"
    assert "tied lm_head" not in " ".join(m.notes)
    assert prev.tie_embeddings is True and prev.param_check()["rel_err"] == 0.0
    assert all(L.core.kind == "gqa" and L.core.n_q == heads and L.core.n_kv == kv
               and L.core.qk_dim == hd and L.core.window is None
               and L.ffn.kind == "dense" and L.ffn.d_ff == inter and L.ffn.n_experts == 0
               for L in m.layers)
    assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].attn_linears] == [
        ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
    assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].attn_linears] == [
        (lin.name, lin.k, lin.n) for lin in prev.layers[0].attn_linears]
    assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].ffn_linears] == [
        ("gate_up", h, 2 * inter), ("down", inter, h)]
    assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].ffn_linears] == [
        (lin.name, lin.k, lin.n) for lin in prev.layers[0].ffn_linears]
    assert m.layers[0].misc_params == 2 * h
    ids = {d["id"]: d for d in list_models()}
    assert ids[mid]["family"] == "Granite-4.2"
    evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=32)))
    evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, prompt=32)))


def test_lfm25_2_6b_base_matches_instruct_shapes():
    h, heads, kv, inter, vocab, layers, kern = 2048, 32, 8, 10752, 128000, 30, 3
    hd = h // heads
    q, kk = heads * hd, kv * hd
    n_conv, n_attn = 22, 8
    conv = h * (3 * h) + h * h + h * kern
    attn = h * q + 2 * h * kk + q * h + 2 * hd
    ffn = 3 * h * inter
    norms = layers * 2 * h + h
    language = n_conv * (conv + ffn) + n_attn * (attn + ffn) + norms + vocab * h
    assert language == 2_697_198_592
    base = get_model("lfm2.5-2.6b-base")
    inst = get_model("lfm2.5-2.6b")
    rel = load_release(base.hf_id)
    old = load_release(inst.hf_id)
    assert base.param_check()["rel_err"] == 0.0 and base.params() == language == base.release_params
    assert base.params() == inst.params() and inst.param_check()["rel_err"] == 0.0
    assert rel["repo"] == "LiquidAI/LFM2.5-2.6B-Base" and rel["n_tensors"] == old["n_tensors"] == 266
    assert rel["params_llm"] == old["params_llm"] == language and rel["params_mtp"] == 0
    assert rel["bytes_llm"] == rel["bytes_total"] == old["bytes_total"] == 5_394_397_184
    assert rel["roles"] == old["roles"] and "lm_head" not in rel["roles"]
    assert rel["torch_dtype"] == "bfloat16" and rel["quantization_config"] is None
    cfg, icfg = rel["config"], old["config"]
    diffs = {k for k in set(cfg) | set(icfg) if cfg.get(k) != icfg.get(k)}
    assert diffs == {"max_position_embeddings", "transformers_version"}
    assert cfg["max_position_embeddings"] == 128000 and icfg["max_position_embeddings"] == 131072
    assert cfg["transformers_version"] == "5.9.0" and icfg["transformers_version"] == "5.2.0"
    assert cfg["model_type"] == "lfm2" and cfg["tie_word_embeddings"] is True
    assert cfg["conv_bias"] is False and cfg["block_auto_adjust_ff_dim"] is False
    assert "num_experts" not in cfg and "head_dim" not in cfg and "num_nextn_predict_layers" not in cfg
    assert cfg["layer_types"] == icfg["layer_types"]
    assert base.max_ctx == 128000 and inst.max_ctx == 131072
    assert base.tie_embeddings is True and base.mtp_layers == () and base.vision is None
    assert base.coverage == "full" and base.n_layers == layers and base.hidden == h
    assert base.kv_fmt == "bf16" and base.act_fmt == "bf16"
    assert len(base.layers) == len(inst.layers)
    for Lb, Li in zip(base.layers, inst.layers):
        assert Lb.core.kind == Li.core.kind and Lb.core.lin == Li.core.lin
        assert Lb.core.conv_kernel == Li.core.conv_kernel
        assert [(lin.name, lin.k, lin.n) for lin in Lb.attn_linears] == [
            (lin.name, lin.k, lin.n) for lin in Li.attn_linears]
        assert [(lin.name, lin.k, lin.n) for lin in Lb.ffn_linears] == [
            (lin.name, lin.k, lin.n) for lin in Li.ffn_linears]
        assert Lb.misc_params == Li.misc_params and Lb.ffn.d_ff == inter and Lb.ffn.n_experts == 0
    conv_l = [L for L in base.layers if L.core.lin == "conv"]
    attn_l = [L for L in base.layers if L.core.kind == "gqa"]
    assert len(conv_l) == n_conv and len(attn_l) == n_attn
    assert conv_l[0].core.conv_kernel == kern and conv_l[0].core.conv_channels == h
    assert [(lin.name, lin.k, lin.n) for lin in conv_l[0].attn_linears] == [
        ("conv_in", h, 3 * h), ("conv_out", h, h)]
    assert conv_l[0].misc_params == h * kern + 2 * h
    L = attn_l[0]
    assert (L.core.n_q, L.core.n_kv, L.core.qk_dim, L.core.window) == (heads, kv, hd, None)
    assert [(lin.name, lin.k, lin.n) for lin in L.attn_linears] == [
        ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
    assert L.misc_params == 2 * hd + 2 * h
    ids = {d["id"]: d for d in list_models()}
    assert ids["lfm2.5-2.6b-base"]["family"] == "LFM2.5"
    assert ids["lfm2.5-2.6b"]["family"] == "LFM2.5"
    s = A.api_eval({"scenario": {"model": "lfm2.5-2.6b-base", "serving": {"phase": "prefill", "batch": 1}},
                    "sensitivity": False})["summary"]
    assert s["step_ms"] > 0 and s["tok_s"] > 0
    evaluate(Scenario(model="lfm2.5-2.6b-base", serving=Serving(phase="decode", batch=1, ctx=32)))
