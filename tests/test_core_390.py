"""0.390: LFM2.5-VL SigLIP2, Mistral-Small twins, MiniMax-M2 q/k norm width, Intern-Decision.

LFM2.5-VL and d1-3B reuse the lfm2 text builder. The vision tower is Siglip2VisionModel
with vision_use_head false, plus the pixel-unshuffle projector. Magistral-Small-2509 and
Mistral-Small-3.1 match Mistral-Small-3.2. MiniMax-M2 / M2.1 / M2.5 match M2.7 shapes.
MiniMaxM2Attention stores full-width q/k RMSNorms. Intern-Decision matches StartLux of the
same size except the checkpoint has no mtp.* tensors.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import entries, get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops, vision_spec


def _raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _same_layers(a, b) -> bool:
    if len(a.layers) != len(b.layers):
        return False
    for la, lb in zip(a.layers, b.layers):
        if [(x.name, x.k, x.n) for x in la.attn_linears] != [(x.name, x.k, x.n) for x in lb.attn_linears]:
            return False
        if [(x.name, x.k, x.n) for x in la.ffn_linears] != [(x.name, x.k, x.n) for x in lb.ffn_linears]:
            return False
        if (la.ffn.kind, la.ffn.d_ff, la.ffn.n_experts, la.ffn.top_k, la.misc_params, la.core.kind) != (
                lb.ffn.kind, lb.ffn.d_ff, lb.ffn.n_experts, lb.ffn.top_k, lb.misc_params, lb.core.kind):
            return False
    return True


def test_lfm25_vl_siglip2_and_d1():
    rows = (
        ("lfm2.5-vl-1.6b", "lfm2.5-1.2b", 1_170_340_608, 426_285_296, 589, 3_193_251_808, 128000, 1152),
        ("lfm2.5-vl-3b", "lfm2.5-2.6b", 2_697_198_592, 426_285_296, 707, 6_246_967_776, 32768, 1152),
        ("lfm2.5-vl-450m", "lfm2.5-350m", 354_483_968, 94_234_880, 349, 897_437_696, 128000, 768),
        ("d1-3b", "lfm2.5-vl-3b", 2_697_198_592, 426_285_296, 707, 6_246_967_776, 32768, 1152),
    )
    for mid, twin, language, vis, nt, total, ctx, vh in rows:
        m, t = get_model(mid), get_model(twin)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params, mid
        assert m.vision is not None and m.vision.params == vis == rel["params_vision"], mid
        assert rel["n_tensors"] == nt and rel["bytes_total"] == total, mid
        assert rel["bytes_total"] == 2 * (language + vis) and rel["params_mtp"] == 0, mid
        assert rel["torch_dtype"] == "bfloat16" and rel["quantization_config"] is None, mid
        assert m.tie_embeddings is True and m.max_ctx == ctx and m.mtp_layers == (), mid
        assert m.coverage == "full" and m.act_fmt == "bf16" and m.kv_fmt == "bf16", mid
        assert m.vision.family == "lfm2vl" and m.vision.hidden == vh and m.vision.gated is False, mid
        assert m.vision.tpatch == 1 and dict(m.vision.pre)["pad"] == 1024, mid
        assert _same_layers(m, t), mid
        cfg = rel["config"]
        assert cfg["model_type"] == "lfm2_vl" and cfg["vision_config"]["model_type"] == "siglip2_vision_model", mid
        assert cfg["vision_config"]["vision_use_head"] is False, mid
        assert cfg["downsample_factor"] == 2 and cfg["projector_use_layernorm"] is False, mid
        assert cfg["projector_bias"] is True and cfg["do_image_splitting"] is True, mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["coverage"] == "full" and listed["domain"] == "vlm" and listed["provider"] == "liquid", mid
    small = get_model("lfm2.5-vl-1.6b")
    wide = get_model("lfm2.5-vl-3b")
    tiny = get_model("lfm2.5-vl-450m")
    d1 = get_model("d1-3b")
    assert small.n_layers == 16 and sum(1 for L in small.layers if L.core.kind == "gqa") == 6
    assert small.layers[0].ffn.d_ff == 8192 and small.hidden == 2048 and small.vocab == 65536
    tc = load_release(small.hf_id)["config"]
    assert "tie_word_embeddings" not in tc and "tie_embedding" not in tc["text_config"]
    assert "tie_word_embeddings" not in tc["text_config"]
    assert wide.n_layers == 30 and wide.layers[0].ffn.d_ff == 10752 and wide.vocab == 128000
    assert tiny.hidden == 1024 and tiny.vision.depth == 12 and tiny.vision.heads == 12
    assert tiny.vision.post == ((3072, 2048), (2048, 1024))
    assert small.vision.post == ((4608, 2048), (2048, 2048))
    assert d1.params() == wide.params() and d1.vision.params == wide.vision.params
    assert load_release(d1.hf_id)["roles"] == load_release(wide.hf_id)["roles"]
    assert load_release(d1.hf_id)["config"]["architectures"] == ["Lfm2VlForConditionalGeneration"]
    g = image_grid(small.vision, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.px, g.views) == (32, 32, 1280, 1287, (1024, 1024), ((32, 32, 5),))
    one = image_grid(small.vision, 512, 512)
    assert (one.merged, one.llm_tokens, one.views) == (256, 258, ((32, 32, 1),))
    up = image_grid(small.vision, 32, 32)
    assert (up.ph, up.pw, up.merged, up.llm_tokens, up.px) == (16, 16, 64, 66, (256, 256))
    assert image_grid(small.vision, 256, 256) == up
    wide_g = image_grid(small.vision, 1280, 720)
    assert (wide_g.merged, wide_g.llm_tokens, wide_g.px, wide_g.views) == (
        764, 769, (1024, 512), ((32, 32, 2), (24, 42, 1)))
    big = image_grid(small.vision, 2048, 1024)
    assert (big.merged, big.llm_tokens, big.views) == (2290, 2301, ((32, 32, 8), (22, 44, 1)))
    assert image_grid(wide.vision, 1024, 1024) == g
    assert image_grid(tiny.vision, 1024, 1024) == g
    ops = vision_ops(small.vision, up, 1)
    pe = next(o for o in ops if o.name == "vis.patch_embed")
    q = next(o for o in ops if o.name == "vis.0.q")
    qk = next(o for o in ops if o.name == "vis.0.qk")
    m0 = next(o for o in ops if o.name == "vis.merger.0")
    m1 = next(o for o in ops if o.name == "vis.merger.1")
    assert (pe.m, pe.k, pe.n) == (1024, 768, 1152)
    assert (q.m, q.k, q.n) == (1024, 1152, 1152)
    assert (qk.m, qk.k, qk.n, qk.count) == (1024, 72, 1024, 16)
    assert (m0.m, m0.k, m0.n) == (64, 4608, 2048)
    assert (m1.m, m1.k, m1.n) == (64, 2048, 2048)
    assert not any(o.name.endswith(".qkv") for o in ops)
    split = vision_ops(small.vision, wide_g, 1)
    assert next(o for o in split if o.name == "vis.patch_embed").m == 2048
    assert next(o for o in split if o.name == "vis.merger.0").m == 512
    assert next(o for o in split if o.name == "vis.v1.patch_embed").m == 1024
    assert next(o for o in split if o.name == "vis.v1.merger.0").m == 252
    tiny_ops = vision_ops(tiny.vision, image_grid(tiny.vision, 256, 256), 1)
    assert next(o for o in tiny_ops if o.name == "vis.patch_embed").n == 768
    assert (next(o for o in tiny_ops if o.name == "vis.merger.1").k,
            next(o for o in tiny_ops if o.name == "vis.merger.1").n) == (2048, 1024)
    pre = evaluate(Scenario(model="lfm2.5-vl-1.6b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1287 and pre.step > 0
    dec = evaluate(Scenario(model="d1-3b", serving=Serving(phase="decode", batch=1, ctx=16)))
    assert dec.step > 0 and dec.vision is None
    ids = [e["id"] for e in list_models() if e["family"] == "LFM2.5"]
    assert ids.index("lfm2.5-350m") < ids.index("lfm2.5-350m-base") < ids.index("lfm2.5-vl-450m") < ids.index("lfm2.5-230m")
    assert ids.index("lfm2.5-2.6b") < ids.index("lfm2.5-vl-3b")
    assert [e["id"] for e in list_models() if e["family"] == "d1"] == ["d1-3b", "d1-omni-600m"]


def test_lfm2_vl_rejects_a_different_tower():
    rel = load_release("LiquidAI/LFM2.5-VL-1.6B")
    head = copy.deepcopy(rel["config"])
    head["vision_config"]["vision_use_head"] = True
    assert vision_spec(rel["repo"], head) is None
    square = copy.deepcopy(rel["config"])
    square["vision_config"]["num_patches"] = 200
    assert vision_spec(rel["repo"], square) is None
    bias = copy.deepcopy(rel["config"])
    bias["projector_bias"] = 1
    assert vision_spec(rel["repo"], bias) is None
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("vision_use_head")
    assert vision_spec(rel["repo"], missing) is None
    _raises(lambda: image_grid(get_model("lfm2.5-vl-1.6b").vision, 0, 32))
    plain = copy.deepcopy(load_release("LiquidAI/LFM2.5-1.2B-Instruct"))
    plain["config"].pop("tie_embedding", None)
    plain["config"].pop("tie_word_embeddings", None)
    spec = from_release("synth", plain["repo"], rel=plain, cfg=plain["config"])
    assert spec.tie_embeddings is False


def test_magistral_and_mistral_31_match_32():
    base = get_model("mistral-small-3.2-24b-instruct-2506")
    old = load_release(base.hf_id)
    for mid in ("magistral-small-2509", "mistral-small-3.1-24b-instruct-2503"):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and base.param_check()["rel_err"] == 0.0, mid
        assert m.params() == base.params() == 22_901_314_560 == m.release_params, mid
        assert m.vision.params == base.vision.params == 438_958_080 == rel["params_vision"], mid
        assert rel["n_tensors"] == old["n_tensors"] == 585, mid
        assert rel["bytes_total"] == old["bytes_total"] == 48_022_722_560, mid
        assert rel["roles"] == old["roles"] and rel["params_mtp"] == 0, mid
        assert m.tie_embeddings is True and m.max_ctx == 131072 and m.mtp_layers == (), mid
        assert m.vision.family == "pixtral" and _same_layers(m, base), mid
        assert m.coverage == "full" and m.act_fmt == "bf16", mid
        g = image_grid(m.vision, 1024, 1024)
        assert g == image_grid(base.vision, 1024, 1024)
        assert (g.px, g.merged, g.llm_tokens) == ((1036, 1036), 1369, 1406)
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["domain"] == "vlm" and listed["coverage"] == "full" and listed["provider"] == "mistral", mid
    mag = load_release("mistralai/Magistral-Small-2509")["config"]
    small = old["config"]
    assert mag.get("dtype") == "bfloat16" and "torch_dtype" not in mag
    assert small.get("torch_dtype") == "bfloat16" and "dtype" not in small
    m31 = load_release("mistralai/Mistral-Small-3.1-24B-Instruct-2503")["config"]
    assert {k: v for k, v in m31.items() if k != "transformers_version"} == {
        k: v for k, v in small.items() if k != "transformers_version"}
    assert next(e["family"] for e in list_models() if e["id"] == "magistral-small-2509") == "Magistral"
    assert next(e["family"] for e in list_models() if e["id"] == "mistral-small-3.1-24b-instruct-2503") == "Mistral-Small"
    pre = evaluate(Scenario(model="magistral-small-2509", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1406 and pre.step > 0


def test_minimax_m2_full_width_qk_norm():
    h, heads, kv, hd, experts, layers = 3072, 48, 8, 128, 256, 62
    qk = heads * hd + kv * hd
    misc = 2 * h + qk + experts
    assert misc == 13568 and qk == 7168
    base = get_model("minimax-m2.7")
    assert base.param_check()["rel_err"] == 0.0
    assert base.params() == 228_689_764_864 == base.release_params
    assert base.layers[0].misc_params == misc
    assert base.max_ctx == 204800 and base.mtp_layers == () and base.vision is None
    rel_b = load_release(base.hf_id)
    assert rel_b["roles"]["norm"]["bf16"]["params"] == layers * (2 * h + qk) + h
    assert rel_b["roles"]["other"]["fp32"]["params"] == layers * experts
    lin = base.layers[0]
    assert [(x.name, x.k, x.n) for x in lin.attn_linears] == [
        ("q", h, heads * hd), ("k", h, kv * hd), ("v", h, kv * hd), ("o", heads * hd, h)]
    assert [(x.name, x.k, x.n) for x in lin.ffn_linears] == [("router", h, experts)]
    assert lin.ffn.kind == "moe" and lin.ffn.top_k == 8 and lin.ffn.d_expert == 1536
    assert lin.core.qk_dim == hd and lin.core.qk_dim != rel_b["config"]["rotary_dim"]
    for mid, ctx in (("minimax-m2", 196608), ("minimax-m2.1", 196608), ("minimax-m2.5", 196608)):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == base.params(), mid
        assert rel["n_tensors"] == rel_b["n_tensors"] == 96103, mid
        assert rel["bytes_total"] == rel_b["bytes_total"] == 230_121_630_720, mid
        assert rel["roles"] == rel_b["roles"] and rel["params_mtp"] == 0, mid
        assert m.max_ctx == ctx and base.max_ctx == 204800, mid
        assert m.act_fmt == "fp8" and m.kv_fmt == "bf16" and m.tie_embeddings is False, mid
        assert m.fmt("attn").fmt == "fp8" and m.fmt("expert").fmt == "fp8" and m.fmt("embed").fmt == "bf16", mid
        assert _same_layers(m, base) and m.coverage == "full" and m.vision is None, mid
        assert rel["config"]["use_qk_norm"] is True and rel["config"]["rotary_dim"] == 64, mid
        assert rel["config"]["quantization_config"]["quant_method"] == "fp8", mid
        assert any("整段投影宽度" in n for n in m.notes), mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "MiniMax-M2" and listed["domain"] == "llm", mid
    ids = [e["id"] for e in list_models() if e["family"] == "MiniMax-M2"]
    assert ids == ["minimax-m2.7", "minimax-m2", "minimax-m2.1", "minimax-m2.5"]
    rel = load_release("MiniMaxAI/MiniMax-M2.7")
    off = copy.deepcopy(rel["config"])
    off["use_qk_norm"] = False
    spec = from_release("synth", rel["repo"], rel=rel, cfg=off)
    assert spec.layers[0].misc_params == 2 * h + experts
    assert base.params() - spec.params() == layers * qk
    kind = copy.deepcopy(rel["config"])
    kind["qk_norm_type"] = "head"
    same = from_release("synth", rel["repo"], rel=rel, cfg=kind)
    assert same.layers[0].misc_params == misc and same.params() == base.params()
    flag = copy.deepcopy(rel["config"])
    flag["use_routing_bias"] = False
    still = from_release("synth", rel["repo"], rel=rel, cfg=flag)
    assert still.layers[0].misc_params == misc and still.params() == base.params()
    qn = get_model("qwen3-next-80b-a3b")
    assert qn.param_check()["rel_err"] == 0.0
    evaluate(Scenario(model="minimax-m2", serving=Serving(phase="decode", batch=1, ctx=8)))


def test_intern_decision_matches_startlux_without_mtp():
    rows = (
        ("intern-decision-4b", "startlux-decision-v1-4b", 4_205_751_296, 333_514_240, 723, 9_078_538_752, 120_599_552),
        ("intern-decision-2b", "startlux-decision-v1-2b", 1_881_825_088, 331_416_576, 617, 4_426_488_512, 60_828_160),
        ("intern-decision-0.8b", "startlux-decision-v1-0.8b", 752_393_024, 100_592_896, 473, 1_705_977_024, 20_452_864),
    )
    for mid, twin, language, vis, nt, total, mtp in rows:
        m, s = get_model(mid), get_model(twin)
        rel, old = load_release(m.hf_id), load_release(s.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and s.param_check()["rel_err"] == 0.0, mid
        assert m.params() == s.params() == language == rel["params_llm"] == old["params_llm"], mid
        assert m.vision.params == s.vision.params == vis == rel["params_vision"], mid
        assert rel["n_tensors"] == nt and old["n_tensors"] == nt + 15, mid
        assert rel["bytes_total"] == total and old["bytes_total"] == total + 2 * mtp, mid
        assert rel["params_mtp"] == 0 and old["params_mtp"] == mtp, mid
        assert m.mtp_layers == () and len(s.mtp_layers) == 1 and s.mtp_params() == mtp, mid
        assert _same_layers(m, s) and m.tie_embeddings is True and m.max_ctx == s.max_ctx == 262144, mid
        assert m.vision.family == "qwen" and image_grid(m.vision, 1024, 1024) == image_grid(s.vision, 1024, 1024), mid
        assert video_pre(s) is None and video_pre(m) == {"min_px": 4096, "max_px": 25165824}, mid
        clip = video_grid(m.vision, video_pre(m), 4, 1280, 720)
        assert (clip.frames, clip.t, clip.llm_tokens, clip.slice.ph, clip.slice.pw, clip.slice.merged) == (
            4, 2, 1760, 44, 80, 880)
        assert m.coverage == "full" and m.act_fmt == "bf16", mid
        assert {k: v for k, v in rel["config"].items() if k != "transformers_version"} == {
            k: v for k, v in old["config"].items() if k != "transformers_version"}, mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "Intern-Decision" and listed["domain"] == "vlm", mid
        assert listed["provider"] == "internlm", mid
        assert next(e for e in entries() if e["id"] == mid)["decision"] is False, mid
    assert [e["id"] for e in list_models() if e["family"] == "Intern-Decision"] == [
        "intern-decision-4b", "intern-decision-2b", "intern-decision-0.8b"]
    _raises(lambda: video_grid(get_model("intern-decision-4b").vision, video_pre(get_model("intern-decision-4b")),
                               1, 1280, 720))
    pre = evaluate(Scenario(model="intern-decision-4b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == evaluate(
        Scenario(model="startlux-decision-v1-4b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1))
    ).vision["image_tokens"]
    assert pre.step > 0
