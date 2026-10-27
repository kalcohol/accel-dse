"""0.383: Gemma-4 video review holds; LFM2.5-2.6B and Granite-4.2 close.

The 0.382 video grid stays on the published processor (max_soft_tokens 70,
num_frames 32). Image pad stays 280. LFM2.5-2.6B is the dense lfm2 hybrid.
Granite-4.2-3b / 8b keep the 4.1 GEMM shapes and store a separate lm_head.
"""
from __future__ import annotations

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops

_G4 = (
    "gemma-4-31b-it", "gemma-4-31b",
    "gemma-4-26b-a4b-it", "gemma-4-26b-a4b",
    "gemma-4-12b-it", "gemma-4-12b",
    "gemma-4-e4b-it", "gemma-4-e4b",
    "gemma-4-e2b-it", "gemma-4-e2b",
)


def test_review_382_gemma4_video_unchanged():
    for mid in _G4:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0
        assert m.vision.rz["max_soft"] == 280
        assert video_pre(m) == {"gemma4_soft": 70, "num_frames": 32, "patch": 16, "pool": 3}
        assert m.vision.family == ("gemma4u" if "12b" in mid else "gemma4")
    vit = get_model("gemma-4-31b-it")
    pre = video_pre(vit)
    wide = video_grid(vit.vision, pre, 32, 1280, 720)
    assert (wide.slice.px, wide.slice.llm_tokens, wide.slice.pad_soft) == ((528, 288), 66, 70)
    assert wide.llm_tokens == 32 * 66
    assert video_grid(vit.vision, pre, 100, 1280, 720).llm_tokens == 32 * 66
    try:
        video_grid(vit.vision, pre, 8, 1280, 720)
    except ValueError as exc:
        assert "num_frames" in str(exc)
    else:
        raise AssertionError("a clip shorter than num_frames was accepted")
    ops = vision_ops(vit.vision, wide.slice, wide.frames)
    q = next(o for o in ops if o.name == "vis.0.q")
    proj = next(o for o in ops if o.name == "vis.proj")
    qk = next(o for o in ops if o.name == "vis.0.qk")
    assert q.m == 32 * 70 * 9 and proj.m == 32 * 66
    assert (qk.m, qk.n, qk.count) == (70 * 9, 70 * 9, 32 * vit.vision.heads)
    still = image_grid(vit.vision, 1024, 1024)
    assert still.pad_soft == 0 and still.llm_tokens == 256
    still_q = next(o for o in vision_ops(vit.vision, still, 1) if o.name == "vis.0.q")
    assert still_q.m == 280 * 9
    uni = get_model("gemma-4-12b-it")
    u = video_grid(uni.vision, video_pre(uni), 32, 1280, 720)
    uops = vision_ops(uni.vision, u.slice, u.frames)
    assert [o.name for o in uops] == ["vis.ln1", "vis.patch_dense", "vis.ln2", "vis.pos_ln", "vis.proj"]
    dense = next(o for o in uops if o.name == "vis.patch_dense")
    uproj = next(o for o in uops if o.name == "vis.proj")
    assert dense.m == uproj.m == 32 * 70
    assert u.llm_tokens == 32 * 66


def test_lfm25_2_6b_dense_hybrid_matches_header():
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
    m = get_model("lfm2.5-2.6b")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params
    assert rel["repo"] == "LiquidAI/LFM2.5-2.6B" and rel["n_tensors"] == 266
    assert rel["params_llm"] == language and rel["params_mtp"] == 0
    assert rel["bytes_llm"] == rel["bytes_total"] == 5_394_397_184
    assert rel["config"]["model_type"] == "lfm2" and rel["config"]["tie_word_embeddings"] is True
    assert rel["config"]["conv_bias"] is False and "num_experts" not in rel["config"]
    assert "num_nextn_predict_layers" not in rel["config"] and "head_dim" not in rel["config"]
    assert m.tie_embeddings is True and m.mtp_layers == () and m.vision is None and m.coverage == "full"
    assert m.n_layers == layers and m.hidden == h and m.kv_fmt == "bf16" and m.act_fmt == "bf16"
    conv_l = [L for L in m.layers if L.core.lin == "conv"]
    attn_l = [L for L in m.layers if L.core.kind == "gqa"]
    assert len(conv_l) == n_conv and len(attn_l) == n_attn
    assert conv_l[0].core.conv_kernel == kern and conv_l[0].core.conv_channels == h
    assert [(lin.name, lin.k, lin.n) for lin in conv_l[0].attn_linears] == [
        ("conv_in", h, 3 * h), ("conv_out", h, h)]
    assert conv_l[0].misc_params == h * kern + 2 * h
    assert all(L.ffn.kind == "dense" and L.ffn.d_ff == inter for L in m.layers)
    L = attn_l[0]
    assert (L.core.n_q, L.core.n_kv, L.core.qk_dim, L.core.v_dim, L.core.window) == (heads, kv, hd, hd, None)
    assert [(lin.name, lin.k, lin.n) for lin in L.attn_linears] == [
        ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
    assert L.misc_params == 2 * hd + 2 * h
    assert "短卷积" in " ".join(m.notes)
    ids = {d["id"]: d for d in list_models()}
    assert ids["lfm2.5-2.6b"]["family"] == "LFM2.5"
    s = A.api_eval({"scenario": {"model": "lfm2.5-2.6b", "serving": {"phase": "prefill", "batch": 1}},
                    "sensitivity": False})["summary"]
    assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_granite_42_untied_keeps_41_gemms():
    # Difference vs the tied 4.1 row is one lm_head (vocab * hidden). Scalars do not change GEMMs.
    rows = {
        "granite-4.2-3b": ("granite-4.1-3b", 40, 2560, 40, 8, 64, 8192, 3_659_737_600,
                           0.015625, 1.0, 1.0, 1.0, 7_319_475_200),
        "granite-4.2-8b": ("granite-4.1-8b", 40, 4096, 32, 8, 128, 12800, 8_791_592_960,
                           0.0078125, 1.0, 1.0, 1.0, 17_583_185_920),
    }
    ids = {d["id"]: d for d in list_models()}
    for mid, (old, n, h, heads, kv, hd, inter, params, attn_m, emb_m, res_m, logits, nbytes) in rows.items():
        m = get_model(mid)
        prev = get_model(old)
        rel = load_release(m.hf_id)
        vocab = rel["config"]["vocab_size"]
        assert vocab == 100352
        assert m.param_check()["rel_err"] == 0.0 and m.params() == params == m.release_params
        assert m.params() == prev.params() + vocab * h
        assert rel["n_tensors"] == 363 and rel["params_mtp"] == 0
        assert rel["bytes_llm"] == rel["bytes_total"] == nbytes
        assert rel["config"]["tie_word_embeddings"] is False and "head_dim" not in rel["config"]
        assert rel["config"]["attention_multiplier"] == attn_m
        assert rel["config"]["embedding_multiplier"] == emb_m
        assert rel["config"]["residual_multiplier"] == res_m
        assert rel["config"]["logits_scaling"] == logits
        assert m.tie_embeddings is False and m.mtp_layers == () and m.vision is None and m.coverage == "full"
        assert m.n_layers == n and m.hidden == h and m.kv_fmt == "bf16" and m.act_fmt == "bf16"
        assert "tied lm_head" not in " ".join(m.notes)
        assert prev.tie_embeddings is True
        assert all(L.core.kind == "gqa" and L.core.n_q == heads and L.core.n_kv == kv
                   and L.core.qk_dim == hd and L.core.window is None
                   and L.ffn.kind == "dense" and L.ffn.d_ff == inter for L in m.layers)
        q, kk = heads * hd, kv * hd
        assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].attn_linears] == [
            ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
        assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].attn_linears] == [
            (lin.name, lin.k, lin.n) for lin in prev.layers[0].attn_linears]
        assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].ffn_linears] == [
            (lin.name, lin.k, lin.n) for lin in prev.layers[0].ffn_linears]
        assert ids[mid]["family"] == "Granite-4.2"
        evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=32)))
    assert ids["granite-4.1-3b"]["family"] == "Granite-4.1"
    assert ids["granite-4.1-8b"]["family"] == "Granite-4.1"
