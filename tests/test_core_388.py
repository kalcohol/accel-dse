"""0.388: 0.387 LFM2.5-2.6B-Base review holds; WeVisOne-VL closes on Qwen3-VL.

Base and Instruct share tensor roles. max_position_embeddings does not change
GEMM or KV width. WeVisOne safetensors are fp32 while config dtype is bfloat16.
Activations stay bf16. The tower is the Qwen3-VL ViT, including deepstack.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model, labels, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid_of, vision_ops, vision_spec

NOTE = "safetensors 是 fp32（config dtype 为 bfloat16）。语言 GEMM 与视觉塔权重按 fp32 计，激活仍是 bf16。"

# id, llm, vision, layers, hidden, n_q, n_kv, inter, n_tensors, bytes_total, vision model_type, out
ROWS = (
    ("wevisone-vl-4b", "qwen3-vl-4b", 4_415_233_536, 415_347_712, 36, 2560, 32, 8, 9728,
     714, 19_322_324_992, "qwen3_vl_vision", 2560),
    ("wevisone-vl-2b", "qwen3-vl-2b", 2_034_787_328, 406_957_056, 28, 2048, 16, 8, 6144,
     626, 9_766_977_536, "qwen3_vl", 2048),
)


def test_review_387_lfm25_base_holds():
    base = get_model("lfm2.5-2.6b-base")
    inst = get_model("lfm2.5-2.6b")
    rel = load_release(base.hf_id)
    old = load_release(inst.hf_id)
    assert base.param_check()["rel_err"] == 0.0 and inst.param_check()["rel_err"] == 0.0
    assert base.params() == inst.params() == 2_697_198_592 == base.release_params
    assert rel["n_tensors"] == old["n_tensors"] == 266 and rel["roles"] == old["roles"]
    assert "lm_head" not in rel["roles"] and rel["params_mtp"] == 0 and rel["params_vision"] == 0
    assert rel["bytes_total"] == old["bytes_total"] == 5_394_397_184
    cfg, icfg = rel["config"], old["config"]
    diffs = {k for k in set(cfg) | set(icfg) if cfg.get(k) != icfg.get(k)}
    assert diffs == {"max_position_embeddings", "transformers_version"}
    assert cfg["max_position_embeddings"] == 128000 and icfg["max_position_embeddings"] == 131072
    assert base.max_ctx == 128000 and inst.max_ctx == 131072
    assert cfg["tie_word_embeddings"] is True and cfg["conv_bias"] is False
    assert cfg["block_auto_adjust_ff_dim"] is False
    assert "num_experts" not in cfg and "head_dim" not in cfg and "num_nextn_predict_layers" not in cfg
    conv = [L for L in base.layers if L.core.lin == "conv"]
    attn = [L for L in base.layers if L.core.kind == "gqa"]
    assert len(conv) == 22 and len(attn) == 8 and base.tie_embeddings is True and base.mtp_layers == ()
    for Lb, Li in zip(base.layers, inst.layers):
        assert [(x.name, x.k, x.n) for x in Lb.attn_linears] == [(x.name, x.k, x.n) for x in Li.attn_linears]
        assert [(x.name, x.k, x.n) for x in Lb.ffn_linears] == [(x.name, x.k, x.n) for x in Li.ffn_linears]
        assert Lb.core.kind == Li.core.kind and Lb.misc_params == Li.misc_params


def test_wevisone_vl_fp32_matches_qwen3_vl_shapes():
    ids = [e["id"] for e in list_models() if e["family"] == "WeVisOne"]
    assert ids == ["wevisone-vl-4b", "wevisone-vl-2b"]
    for mid, twin, params, vis, layers, hidden, nq, nkv, inter, nt, total, vmt, out in ROWS:
        m = get_model(mid)
        q = get_model(twin)
        rel = load_release(m.hf_id)
        pc = m.param_check()
        hd = hidden // nq
        qn, kn = nq * hd, nkv * hd
        assert pc["rel_err"] == 0.0 and m.params() == params == m.release_params == rel["params_llm"], mid
        assert m.vision.params == vis == rel["params_vision"] == q.vision.params, mid
        assert rel["bytes_llm"] == 4 * params and rel["bytes_total"] == total, mid
        assert rel["n_tensors"] == nt and rel["params_mtp"] == 0 and rel["quantization_config"] is None, mid
        assert rel["torch_dtype"] == "bfloat16" and rel["config"]["dtype"] == "bfloat16", mid
        assert set(rel["roles"]) == {"attn", "embed", "lm_head", "mlp", "norm", "vision"}, mid
        assert all(set(block) == {"fp32"} and block["fp32"]["bytes"] == 4 * block["fp32"]["params"]
                   for block in rel["roles"].values()), mid
        cfg = rel["config"]
        text, vc = cfg["text_config"], cfg["vision_config"]
        assert cfg["architectures"] == ["Qwen3VLForConditionalGeneration"] and cfg["model_type"] == "qwen3_vl", mid
        assert cfg["tie_word_embeddings"] is False and "num_nextn_predict_layers" not in cfg, mid
        assert text["vocab_size"] == 152680 and text["hidden_size"] == hidden and text["head_dim"] == 128, mid
        assert text["num_hidden_layers"] == layers and text["intermediate_size"] == inter, mid
        assert (text["num_attention_heads"], text["num_key_value_heads"]) == (nq, nkv), mid
        assert text["max_position_embeddings"] == 262144 and vc["model_type"] == vmt, mid
        assert vc["deepstack_visual_indexes"] == [5, 11, 17] and vc["out_hidden_size"] == out, mid
        assert (vc["depth"], vc["hidden_size"], vc["patch_size"], vc["spatial_merge_size"],
                vc["temporal_patch_size"]) == (24, 1024, 16, 2, 2), mid
        built = vision_spec(m.hf_id, cfg)
        assert built is not None and built.deepstack == 3 and built.params == vis, mid
        assert m.tie_embeddings is False and m.mtp_layers == () and m.vocab == 152680, mid
        assert m.n_layers == layers and m.hidden == hidden and m.max_ctx == 262144, mid
        assert m.coverage == "full" and m.coverage_reasons == () and m.act_fmt == "bf16" and m.kv_fmt == "bf16", mid
        assert NOTE in m.notes, mid
        assert m.vision.w_Bpp == 4.0 and m.vision.deepstack == 3 and m.vision.label == "Qwen3-VL ViT + deepstack", mid
        fm = {k: (f, b, a) for k, f, b, a in m.vision.wfmt}
        assert fm == {k: ("fp32", 32.0, "bf16") for k in ("attn", "merger", "mlp", "patch")}, mid
        for role in ("attn", "embed", "lm_head", "mlp"):
            assert m.fmt(role).fmt == "fp32" and m.fmt(role).bits == 32.0, (mid, role)
        L = m.layers[0]
        assert (L.core.n_q, L.core.n_kv, L.core.qk_dim, L.core.window) == (nq, nkv, hd, None), mid
        assert [(lin.name, lin.k, lin.n) for lin in L.attn_linears] == [
            ("q", hidden, qn), ("k", hidden, kn), ("v", hidden, kn), ("o", qn, hidden)], mid
        assert [(lin.name, lin.k, lin.n) for lin in L.ffn_linears] == [
            ("gate_up", hidden, 2 * inter), ("down", inter, hidden)], mid
        assert L.misc_params == 2 * hidden + 2 * hd, mid
        assert [(lin.name, lin.k, lin.n) for lin in L.attn_linears] == [
            (lin.name, lin.k, lin.n) for lin in q.layers[0].attn_linears], mid
        assert [(lin.name, lin.k, lin.n) for lin in L.ffn_linears] == [
            (lin.name, lin.k, lin.n) for lin in q.layers[0].ffn_linears], mid
        assert L.misc_params == q.layers[0].misc_params and q.tie_embeddings is True, mid
        assert m.params() == q.params() + (2 * 152680 - q.vocab) * hidden, mid
        g = image_grid(m.vision, 1024, 1024)
        gq = image_grid(q.vision, 1024, 1024)
        assert (g.ph, g.pw, g.merged, g.llm_tokens) == (gq.ph, gq.pw, gq.merged, gq.llm_tokens) == (64, 64, 1024, 1024)
        sv = Scenario(model=mid).replace("serving.videos", 1).replace("serving.video_frames", 4).serving
        clip = video_grid_of(m, sv)
        cq = video_grid_of(q, sv)
        assert clip.t == cq.t == 2 and clip.frames == 4 and clip.llm_tokens == cq.llm_tokens == 1760, mid
        assert (clip.slice.ph, clip.slice.pw, clip.slice.merged) == (44, 80, 880), mid
        ops = {o.name: o for o in vision_ops(m.vision, g, 1)}
        assert ops["vis.patch_embed"].w_fmt == "fp32" and ops["vis.patch_embed"].a_fmt == "bf16", mid
        assert ops["vis.deepstack"].vec == 3 * 1024 * out, mid
        gemm = [o for o in build_rank_ops(m, 0, Phase("prefill", 1, 32, 0), Shard()) if o.kind == "gemm" and o.w_params]
        assert gemm and all(o.w_fmt == "fp32" and o.w_bits == 32.0 and o.a_fmt == "bf16" for o in gemm), mid
        row = labels(m)
        assert row["dtype"] == "W fp32 · A bf16 · KV bf16" and row["vision"]["video"] is True, mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "WeVisOne" and listed["domain"] == "vlm" and listed["coverage"] == "full", mid
        pre = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, prompt=128, images=1)))
        assert pre.vision["image_tokens"] == 1024, mid
        assert pre.vision["weights_GB"] == m.vision.params * m.vision.w_Bpp / 1e9, mid
        assert pre.step > 0
        dec = evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=32)))
        assert dec.step > 0 and dec.vision["weights_GB"] == pre.vision["weights_GB"], mid
