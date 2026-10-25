"""0.389: Nemotron-3.5-Bolt-E4B on Gemma-4, LFM2.5-8B-A1B / 1.2B, Granite-4.2 FP8, Atria-Dawn.

Bolt-E4B-BF16 matches gemma-4-E4B tensor names, dtypes, and shapes. LFM2.5-1.2B
uses the published Lfm2MLP width adjust and the tie_embedding alias. Granite-4.2
FP8 is the existing compressed-tensors W8A8 bill. Atria-Dawn / FP8 match
GLM-5.2 / GLM-5.3 tensor for tensor; config head_dim is not the MLA width.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _build_gemma4, _build_lfm2, _lfm2_dense_ff, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre

_G4 = ("nemotron-3.5-bolt-e4b-bf16", "gemma-4-e4b")
_FP8 = (
    ("granite-4.2-3b-fp8", "granite-4.2-3b", 3_659_737_600, 4_175_754_240, 643),
    ("granite-4.2-8b-fp8", "granite-4.2-8b", 8_791_592_960, 9_617_203_200, 643),
    ("granite-4.2-30b-fp8", "granite-4.2-30b", 29_276_770_304, 30_109_605_888, 1027),
)


def _raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_review_388_wevisone_still_fp32():
    for mid, nt, params, vis, total in (
        ("wevisone-vl-4b", 714, 4_415_233_536, 415_347_712, 19_322_324_992),
        ("wevisone-vl-2b", 626, 2_034_787_328, 406_957_056, 9_766_977_536),
    ):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == params == rel["params_llm"], mid
        assert m.vision.params == vis == rel["params_vision"] and rel["n_tensors"] == nt, mid
        assert rel["bytes_total"] == total and rel["bytes_llm"] == 4 * params, mid
        assert m.fmt("attn").fmt == "fp32" and m.act_fmt == "bf16" and m.kv_fmt == "bf16", mid
        assert rel["quantization_config"] is None and rel["params_mtp"] == 0, mid


def test_nemotron_bolt_e4b_matches_gemma4_e4b():
    m = get_model("nemotron-3.5-bolt-e4b-bf16")
    e4 = get_model("gemma-4-e4b")
    it = get_model("gemma-4-e4b-it")
    rel, old = load_release(m.hf_id), load_release(e4.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and e4.param_check()["rel_err"] == 0.0
    assert m.params() == e4.params() == 7_518_069_034 == m.release_params == rel["params_llm"]
    assert m.vision.params == e4.vision.params == 478_088_384 == rel["params_vision"]
    assert rel["n_tensors"] == old["n_tensors"] == 2130 and rel["roles"] == old["roles"]
    assert rel["bytes_total"] == old["bytes_total"] == 15_992_314_836
    assert rel["bytes_llm"] == old["bytes_llm"] == 2 * rel["params_llm"]
    assert rel["quantization_config"] is None and rel["params_mtp"] == 0 and rel["torch_dtype"] == "bfloat16"
    assert all(set(block) == {"bf16"} for block in rel["roles"].values())
    cfg, base = rel["config"], old["config"]
    diffs = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
    assert diffs == {"eos_token_id", "text_config"}
    text_diffs = {k for k in set(cfg["text_config"]) | set(base["text_config"])
                  if cfg["text_config"].get(k) != base["text_config"].get(k)}
    assert text_diffs == {"max_position_embeddings"}
    assert cfg["text_config"]["max_position_embeddings"] == 65536
    assert base["text_config"]["max_position_embeddings"] == 131072
    assert m.max_ctx == 65536 and e4.max_ctx == 131072
    assert cfg["eos_token_id"] == load_release(it.hf_id)["config"]["eos_token_id"]
    assert cfg["model_type"] == "gemma4" and cfg["text_config"]["model_type"] == "gemma4_text"
    assert cfg["vision_config"]["model_type"] == "gemma4_vision"
    assert cfg["audio_config"]["model_type"] == "gemma4_audio"
    assert cfg["text_config"]["num_kv_shared_layers"] == 18
    assert cfg["text_config"]["hidden_size_per_layer_input"] == 256
    assert cfg["text_config"]["attention_k_eq_v"] is False
    assert cfg["tie_word_embeddings"] is True and m.tie_embeddings is True
    assert m.n_layers == 42 and m.hidden == 2560 and m.vocab == 262144
    assert m.cached_params == e4.cached_params == 55_055_616
    assert m.ple_packed == 42 * 256 and m.lookup_params == 262144 * 42 * 256
    assert m.coverage == "full" and m.coverage_reasons == () and m.act_fmt == "bf16" and m.kv_fmt == "bf16"
    for Lm, Le in zip(m.layers, e4.layers):
        assert Lm.kv_alias == Le.kv_alias and Lm.ple_in == Le.ple_in == 256
        assert [(x.name, x.k, x.n) for x in Lm.attn_linears] == [(x.name, x.k, x.n) for x in Le.attn_linears]
        assert [(x.name, x.k, x.n) for x in Lm.ffn_linears] == [(x.name, x.k, x.n) for x in Le.ffn_linears]
        assert Lm.core.qk_dim == Le.core.qk_dim and Lm.core.window == Le.core.window
        assert Lm.ffn.d_ff == Le.ffn.d_ff == 10240
    alias = [L for L in m.layers if L.kv_alias]
    assert len(alias) == 18
    assert all(not any(lin.name in ("k", "v") for lin in L.attn_linears) for L in alias)
    sm = next(o for o in build_rank_ops(m, 41, Phase("decode", 1, 1, 64), Shard()) if o.name == "softmax")
    assert sm.kv_write == 0.0 and sm.kv_read > 0.0
    assert video_pre(m) == video_pre(e4) == {"gemma4_soft": 70, "num_frames": 32, "patch": 16, "pool": 3}
    assert image_grid(m.vision, 1024, 1024) == image_grid(e4.vision, 1024, 1024)
    assert video_grid(m.vision, video_pre(m), 32, 1280, 720) == video_grid(e4.vision, video_pre(e4), 32, 1280, 720)
    assert m.vision.audio.kind == "g4conf" and m.vision.audio.tokens(1.0) == 25
    ids = [e["id"] for e in list_models() if e["family"].startswith("Nemotron-3.5")]
    assert "nemotron-3.5-bolt-e4b-bf16" in ids
    listed = next(e for e in list_models() if e["id"] == "nemotron-3.5-bolt-e4b-bf16")
    assert listed["domain"] == "vlm" and listed["coverage"] == "full"
    pre = evaluate(Scenario(model="nemotron-3.5-bolt-e4b-bf16",
                            serving=Serving(phase="prefill", batch=1, prompt=32, images=1)))
    assert pre.vision["image_tokens"] == 256 and pre.step > 0
    dec = evaluate(Scenario(model="nemotron-3.5-bolt-e4b-bf16",
                            serving=Serving(phase="decode", batch=1, ctx=32)))
    assert dec.step > 0 and dec.vision is None
    # A wider shared-KV cut is not a donor pattern this checkpoint uses.
    bad = dict(cfg["text_config"])
    bad["num_kv_shared_layers"] = bad["num_hidden_layers"]
    _raises(lambda: _build_gemma4(bad))


def test_lfm25_8b_a1b_and_1_2b():
    h, heads, kv, vocab = 2048, 32, 8, 65536
    hd = h // heads
    inter, layers, kern = 8192, 16, 3
    n_conv, n_attn = 10, 6
    qn, kn = heads * hd, kv * hd
    conv = h * (3 * h) + h * h + h * kern
    attn = h * qn + 2 * h * kn + qn * h + 2 * hd
    ffn = 3 * h * inter
    language = n_conv * (conv + ffn) + n_attn * (attn + ffn) + layers * 2 * h + h + vocab * h
    assert language == 1_170_340_608
    wide = n_conv * (conv + 3 * h * 12288) + n_attn * (attn + 3 * h * 12288) + layers * 2 * h + h + 2 * vocab * h
    assert wide != language
    for mid in ("lfm2.5-1.2b", "lfm2.5-1.2b-base"):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params, mid
        assert rel["n_tensors"] == 148 and rel["bytes_total"] == 2 * language and rel["params_mtp"] == 0, mid
        assert "lm_head" not in rel["roles"] and rel["quantization_config"] is None, mid
        assert all(set(block) == {"bf16"} for block in rel["roles"].values()), mid
        cfg = rel["config"]
        assert cfg["tie_embedding"] is True and "tie_word_embeddings" not in cfg, mid
        assert cfg["block_auto_adjust_ff_dim"] is True and cfg["intermediate_size"] == 12288, mid
        assert cfg["block_ffn_dim_multiplier"] == 1.0 and cfg["block_multiple_of"] == 256, mid
        assert m.tie_embeddings is True and m.mtp_layers == () and m.vision is None, mid
        assert m.n_layers == 16 and m.hidden == h and m.max_ctx == 128000 and m.act_fmt == "bf16", mid
        assert sum(1 for L in m.layers if L.core.lin == "conv") == 10, mid
        assert sum(1 for L in m.layers if L.core.kind == "gqa") == 6, mid
        assert all(L.ffn.d_ff == 8192 and L.ffn.kind == "dense" for L in m.layers), mid
        conv_l = next(L for L in m.layers if L.core.lin == "conv")
        assert [(lin.name, lin.k, lin.n) for lin in conv_l.attn_linears] == [
            ("conv_in", h, 3 * h), ("conv_out", h, h)], mid
        assert conv_l.core.conv_kernel == 3 and conv_l.misc_params == h * 3 + 2 * h, mid
        attn_l = next(L for L in m.layers if L.core.kind == "gqa")
        assert (attn_l.core.n_q, attn_l.core.n_kv, attn_l.core.qk_dim) == (32, 8, 64), mid
        assert [(lin.name, lin.k, lin.n) for lin in attn_l.attn_linears] == [
            ("q", h, qn), ("k", h, kn), ("v", h, kn), ("o", qn, h)], mid
        assert any("block_auto_adjust_ff_dim" in n for n in m.notes), mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "LFM2.5" and listed["domain"] == "llm" and listed["coverage"] == "full", mid
    inst, base = load_release("LiquidAI/LFM2.5-1.2B-Instruct"), load_release("LiquidAI/LFM2.5-1.2B-Base")
    assert inst["roles"] == base["roles"] and inst["config"] == base["config"]
    assert "block_auto_adjust_ff_dim" not in " ".join(get_model("lfm2.5-2.6b").notes)
    assert get_model("lfm2.5-2.6b").layers[0].ffn.d_ff == 10752

    m = get_model("lfm2.5-8b-a1b")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == 8_467_856_832 == m.release_params
    assert rel["n_tensors"] == 2302 and rel["params_mtp"] == 0 and rel["quantization_config"] is None
    bias = 22 * 32
    assert rel["roles"]["mlp"]["fp32"]["params"] == bias
    assert rel["bytes_total"] == 2 * m.params() + 2 * bias == 16_935_715_072
    cfg = rel["config"]
    assert cfg["model_type"] == "lfm2_moe" and cfg["tie_word_embeddings"] is True and m.tie_embeddings is True
    assert cfg["num_experts"] == 32 and cfg["num_experts_per_tok"] == 4 and cfg["num_dense_layers"] == 2
    assert cfg["moe_intermediate_size"] == 1792 and cfg["intermediate_size"] == 7168
    assert cfg["use_expert_bias"] is True and cfg["routed_scaling_factor"] == 1.0
    assert "block_auto_adjust_ff_dim" not in cfg and "num_nextn_predict_layers" not in cfg
    assert m.n_layers == 24 and m.hidden == 2048 and m.max_ctx == 128000 and m.vision is None
    assert sum(1 for L in m.layers if L.core.lin == "conv") == 18
    assert sum(1 for L in m.layers if L.core.kind == "gqa") == 6
    assert [L.ffn.kind for L in m.layers[:3]] == ["dense", "dense", "moe"]
    assert m.layers[0].ffn.d_ff == 7168 and m.layers[2].ffn.n_experts == 32 and m.layers[2].ffn.top_k == 4
    assert m.layers[2].ffn.d_expert == 1792 and m.layers[2].core.kind == "gqa"
    assert m.layers[2].misc_params == 2 * 64 + 2 * 2048 + 32
    assert m.fmt("expert").fmt == "bf16" and m.fmt("attn").fmt == "bf16" and m.act_fmt == "bf16"
    assert m.coverage == "full" and m.mtp_layers == ()
    listed = next(e for e in list_models() if e["id"] == "lfm2.5-8b-a1b")
    assert listed["family"] == "LFM2.5" and listed["domain"] == "llm"
    evaluate(Scenario(model="lfm2.5-8b-a1b", serving=Serving(phase="decode", batch=1, ctx=32)))
    evaluate(Scenario(model="lfm2.5-1.2b", serving=Serving(phase="prefill", batch=1, prompt=32)))


def test_lfm_auto_adjust_rejects_bad_width():
    cfg = dict(load_release("LiquidAI/LFM2.5-1.2B-Instruct")["config"])
    assert _lfm2_dense_ff(cfg) == 8192
    assert _lfm2_dense_ff({**cfg, "block_auto_adjust_ff_dim": False}) == 12288
    _raises(lambda: _lfm2_dense_ff({**cfg, "block_multiple_of": 0}))
    _raises(lambda: _lfm2_dense_ff({**cfg, "intermediate_size": 0}))
    bad = {**cfg, "layer_types": ["conv"] * 15 + ["moe"]}
    _raises(lambda: _build_lfm2(bad))
    short = {**cfg, "layer_types": ["conv", "full_attention"]}
    _raises(lambda: _build_lfm2(short))


def test_lfm25_350m_and_230m():
    inst, base = load_release("LiquidAI/LFM2.5-350M"), load_release("LiquidAI/LFM2.5-350M-Base")
    assert inst["roles"] == base["roles"] and inst["config"] == base["config"]
    h, heads, kv, vocab, hd = 1024, 16, 8, 65536, 64
    qn, kn, kern = heads * hd, kv * hd, 3
    conv = h * (3 * h) + h * h + h * kern
    attn = h * qn + 2 * h * kn + qn * h + 2 * hd
    inter = 4608
    ffn = 3 * h * inter
    language = 10 * (conv + ffn) + 6 * (attn + ffn) + 16 * 2 * h + h + vocab * h
    assert language == 354_483_968
    wide = 10 * (conv + 3 * h * 6656) + 6 * (attn + 3 * h * 6656) + 16 * 2 * h + h + vocab * h
    assert wide != language
    for mid in ("lfm2.5-350m", "lfm2.5-350m-base"):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params, mid
        assert rel["n_tensors"] == 148 and rel["bytes_total"] == 2 * language and rel["params_mtp"] == 0, mid
        assert "lm_head" not in rel["roles"] and m.tie_embeddings is True and m.vision is None, mid
        cfg = rel["config"]
        assert cfg["tie_embedding"] is True and "tie_word_embeddings" not in cfg, mid
        assert cfg["block_auto_adjust_ff_dim"] is True and cfg["intermediate_size"] == 6656, mid
        assert cfg["block_multiple_of"] == 256 and m.layers[0].ffn.d_ff == 4608, mid
        assert m.n_layers == 16 and m.hidden == h and m.max_ctx == 128000, mid
        assert sum(1 for L in m.layers if L.core.lin == "conv") == 10, mid
        assert sum(1 for L in m.layers if L.core.kind == "gqa") == 6, mid
        assert any("block_auto_adjust_ff_dim" in n for n in m.notes), mid
        assert m.coverage == "full" and m.mtp_layers == (), mid
    small = 2560
    language230 = 8 * (conv + 3 * h * small) + 6 * (attn + 3 * h * small) + 14 * 2 * h + h + vocab * h
    assert language230 == 229_693_184
    m = get_model("lfm2.5-230m")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == language230 == m.release_params
    assert rel["n_tensors"] == 132 and rel["bytes_total"] == 2 * language230 and rel["params_mtp"] == 0
    assert rel["config"]["block_auto_adjust_ff_dim"] is False and rel["config"]["tie_word_embeddings"] is True
    assert m.tie_embeddings is True and m.n_layers == 14 and m.layers[0].ffn.d_ff == 2560
    assert "block_auto_adjust_ff_dim" not in " ".join(m.notes)
    assert sum(1 for L in m.layers if L.core.lin == "conv") == 8
    assert sum(1 for L in m.layers if L.core.kind == "gqa") == 6
    assert m.coverage == "full" and m.vision is None and m.act_fmt == "bf16"
    ids = [e["id"] for e in list_models() if e["family"] == "LFM2.5"]
    assert ids.index("lfm2.5-350m") < ids.index("lfm2.5-350m-base") < ids.index("lfm2.5-230m")
    evaluate(Scenario(model="lfm2.5-230m", serving=Serving(phase="decode", batch=1, ctx=16)))


def test_granite_42_fp8_matches_bf16_shapes():
    ids = [e["id"] for e in list_models() if e["family"] == "Granite-4.2"]
    assert ids == [
        "granite-4.2-30b", "granite-4.2-30b-mxfp4", "granite-4.2-30b-nvfp4", "granite-4.2-30b-fp8",
        "granite-4.2-8b", "granite-4.2-8b-mxfp4", "granite-4.2-8b-nvfp4", "granite-4.2-8b-fp8",
        "granite-4.2-3b", "granite-4.2-3b-mxfp4", "granite-4.2-3b-nvfp4", "granite-4.2-3b-fp8",
    ]
    for mid, twin, params, total, nt in _FP8:
        m, b = get_model(mid), get_model(twin)
        rel, old = load_release(m.hf_id), load_release(b.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and b.param_check()["rel_err"] == 0.0, mid
        assert m.params() == b.params() == params == m.release_params, mid
        assert rel["n_tensors"] == nt and rel["bytes_total"] == rel["bytes_llm"] == total, mid
        assert rel["bytes_total"] < old["bytes_total"] and rel["params_mtp"] == 0, mid
        qc = rel["quantization_config"]
        assert qc["quant_method"] == "compressed-tensors" and qc["format"] == "float-quantized", mid
        assert qc["ignore"] == ["lm_head"] and qc.get("kv_cache_scheme") is None, mid
        group = next(iter(qc["config_groups"].values()))
        assert group["input_activations"]["num_bits"] == 8 and group["input_activations"]["type"] == "float", mid
        assert group["input_activations"]["strategy"] == "token" and group["input_activations"]["dynamic"] is True, mid
        assert group["weights"]["num_bits"] == 8 and group["weights"]["strategy"] == "channel", mid
        assert m.act_fmt == "fp8" and m.kv_fmt == "bf16" and b.act_fmt == "bf16", mid
        assert m.fmt("attn").fmt == "fp8" and m.fmt("mlp").fmt == "fp8", mid
        assert m.fmt("embed").fmt == "bf16" and m.fmt("lm_head").fmt == "bf16", mid
        assert m.fmt("attn").bits > 8.0 and m.fmt("attn").bits < 8.1, mid
        assert m.tie_embeddings is False and b.tie_embeddings is False and m.mtp_layers == (), mid
        assert m.coverage == "full" and m.vision is None, mid
        for Lm, Lb in zip(m.layers, b.layers):
            assert [(x.name, x.k, x.n) for x in Lm.attn_linears] == [(x.name, x.k, x.n) for x in Lb.attn_linears]
            assert [(x.name, x.k, x.n) for x in Lm.ffn_linears] == [(x.name, x.k, x.n) for x in Lb.ffn_linears]
            assert Lm.core.n_q == Lb.core.n_q and Lm.core.n_kv == Lb.core.n_kv and Lm.core.qk_dim == Lb.core.qk_dim
            assert Lm.ffn.d_ff == Lb.ffn.d_ff and Lm.misc_params == Lb.misc_params
        assert rel["config"]["attention_multiplier"] == old["config"]["attention_multiplier"], mid
        assert rel["config"]["embedding_multiplier"] == rel["config"]["residual_multiplier"] == 1.0, mid
        assert rel["config"]["logits_scaling"] == 1.0 and rel["config"]["tie_word_embeddings"] is False, mid
        gemm = [o for o in build_rank_ops(m, 0, Phase("prefill", 1, 16, 0), Shard()) if o.kind == "gemm" and o.w_params]
        names = {o.name: o for o in gemm}
        fused = ("attn.qkv", "attn.o", "mlp.gate_up", "mlp.down")
        assert set(names) >= set(fused), mid
        assert all(names[n].w_fmt == "fp8" and names[n].a_fmt == "fp8" for n in fused), mid
        head = [o for o in head_ops(m, Phase("prefill", 1, 16, 0)) if o.name == "lm_head"]
        assert head and all(o.w_fmt == "bf16" and o.a_fmt == "fp8" for o in head), mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "Granite-4.2" and listed["coverage"] == "full", mid
    evaluate(Scenario(model="granite-4.2-8b-fp8", serving=Serving(phase="decode", batch=1, ctx=32)))


def test_atria_dawn_matches_glm():
    rows = (
        ("atria-dawn", "glm-5.2", 59_585, 1_506_659_919_872, "bfloat16", None),
        ("atria-dawn-fp8", "glm-5.3", 118_629, 755_617_140_416, "bfloat16", "fp8"),
    )
    ids = [e["id"] for e in list_models() if e["family"] == "Atria"]
    assert ids == ["atria-dawn", "atria-dawn-fp8"]
    for mid, twin, nt, total, dtype, quant in rows:
        m, g = get_model(mid), get_model(twin)
        rel, old = load_release(m.hf_id), load_release(g.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and g.param_check()["rel_err"] == 0.0, mid
        assert m.params() == g.params() == 743_377_019_904 == rel["params_llm"] == old["params_llm"], mid
        assert rel["params_mtp"] == old["params_mtp"] == 9_952_920_576, mid
        assert rel["n_tensors"] == old["n_tensors"] == nt and rel["bytes_total"] == old["bytes_total"] == total, mid
        assert rel["roles"] == old["roles"] and rel["torch_dtype"] == dtype, mid
        assert m.act_fmt == g.act_fmt and m.kv_fmt == g.kv_fmt == "bf16", mid
        if quant is None:
            assert rel["quantization_config"] is None and m.act_fmt == "bf16", mid
        else:
            assert rel["quantization_config"]["quant_method"] == "fp8" and m.act_fmt == "fp8", mid
            assert rel["quantization_config"].get("scale_fmt") == "ue8m0", mid
            assert "scale_fmt" not in (old["quantization_config"] or {}), mid
            assert m.fmt("expert").fmt == "fp8" and m.fmt("embed").fmt == "bf16", mid
        cfg, gc = rel["config"], old["config"]
        assert cfg["model_type"] == "glm_moe_dsa" and cfg["head_dim"] == 64 and gc["head_dim"] == 192, mid
        assert (cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]) == (192, 64, 256), mid
        assert (cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]) == (
            gc["qk_nope_head_dim"], gc["qk_rope_head_dim"], gc["v_head_dim"]), mid
        assert cfg["indexer_types"].count("full") == 21 and cfg["indexer_types"].count("shared") == 57, mid
        assert cfg["indexer_types"] == gc["indexer_types"], mid
        assert set(cfg["layer_types"]) == {"deepseek_sparse_attention"} and "layer_types" not in gc, mid
        assert m.n_layers == g.n_layers == 78 and m.mtp_layers and len(m.mtp_layers) == len(g.mtp_layers), mid
        assert sum(1 for L in m.layers if L.core.idx_heads) == 21, mid
        assert m.layers[0].core.qk_dim == 256 and m.layers[0].core.v_dim == 256, mid
        assert m.layers[0].core.qk_dim != cfg["head_dim"], mid
        for Lm, Lg in zip(m.layers, g.layers):
            assert [(x.name, x.k, x.n) for x in Lm.attn_linears] == [(x.name, x.k, x.n) for x in Lg.attn_linears]
            assert [(x.name, x.k, x.n) for x in Lm.ffn_linears] == [(x.name, x.k, x.n) for x in Lg.ffn_linears]
            assert Lm.core.idx_heads == Lg.core.idx_heads and Lm.ffn.kind == Lg.ffn.kind
            assert Lm.misc_params == Lg.misc_params
        assert m.coverage == "full" and m.vision is None, mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "Atria" and listed["domain"] == "llm" and listed["provider"] == "internlm", mid
    evaluate(Scenario(model="atria-dawn", serving=Serving(phase="decode", batch=1, ctx=8)))
    evaluate(Scenario(model="atria-dawn-fp8", serving=Serving(phase="prefill", batch=1, prompt=16)))
