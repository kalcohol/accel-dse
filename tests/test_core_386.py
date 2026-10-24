"""0.386: 0.385 ERNIE-4.5-VL video review holds; granite-4.2-30b closes.

The video grid stays on Ernie4_5_VLProcessor: per-frame smart_resize at
299*28^2 / 1196*28^2, an odd count copies the last frame, the tower attends
inside one frame, and the temporal GEMM (K = 10240) sees half the rows.
granite-4.2-30b keeps the 4.1-30b GEMM shapes and stores a separate lm_head.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops


def _pairs(op):
    return op.count * op.m * op.share


def test_review_385_ernie_vl_video_holds():
    pre_expect = {"ernie_temporal": 2, "min_px": 299 * 28 * 28, "max_px": 1196 * 28 * 28}
    specs = {
        "ernie-4.5-vl-28b-a3b-pt": (2560, 6),
        "ernie-4.5-vl-424b-a47b-pt": (8192, 8),
    }
    for mid, (text_n, k) in specs.items():
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0
        assert m.mtp_layers == () and m.mtp_params() == 0
        assert "num_nextn_predict_layers" not in rel["config"]
        assert m.vision.family == "ernievl"
        pre = video_pre(m)
        assert pre == pre_expect
        still = image_grid(m.vision, 1024, 1024)
        assert (still.merged, still.llm_tokens, still.tframes, still.px) == (1369, 1371, 1, (1036, 1036))
        still_ops = vision_ops(m.vision, still, 1)
        assert [o.m for o in still_ops if o.name.startswith("vis.merger")] == [1369] * 5
        assert next(o for o in still_ops if o.name == "vis.merger.2").k == 10240

        hd = video_grid(m.vision, pre, 32, 1280, 720)
        assert hd.frames == 32 and hd.t == 1 and hd.tail is None
        assert (hd.slice.ph, hd.slice.pw, hd.slice.merged, hd.slice.tframes) == (52, 92, 1196, 32)
        assert hd.slice.px == (1288, 728)
        assert hd.llm_tokens == 16 * 1196 + 2 == 19138
        ops = vision_ops(m.vision, hd.slice, 1)
        mergers = [o for o in ops if o.name.startswith("vis.merger")]
        assert [(o.m, o.k, o.n) for o in mergers] == [
            (32 * 1196, 5120, 5120),
            (32 * 1196, 5120, 5120),
            (16 * 1196, 10240, 5120),
            (16 * 1196, 5120, 5120),
            (16 * 1196, 5120, text_n),
        ]
        assert not any("index" in o.name or o.name == "vis.tpool" for o in ops)
        qk = [o for o in ops if o.name.endswith(".qk")]
        assert len(qk) == 32
        assert all(o.count == 32 * 16 and o.m == 52 * 92 and o.n == 52 * 92 for o in qk)
        odd = video_grid(m.vision, pre, 17, 1280, 720)
        assert odd.frames == 18 and odd.llm_tokens == 9 * 1196 + 2 == 10766

        moe = next(i for i, L in enumerate(m.layers) if L.ffn.n_experts_b)
        tok = hd.llm_tokens
        q = 32
        mix = build_rank_ops(m, moe, Phase("prefill", 1, tok + q, 0, mm=tok), Shard())
        assert _pairs(next(o for o in mix if o.name == "expert_b.gate_up")) == tok * k
        assert next(o for o in mix if o.name == "moe.router_b").m == tok + q
        dec = build_rank_ops(m, moe, Phase("decode", 1, 1, 16), Shard())
        assert not any("router_b" in o.name or "expert_b" in o.name for o in dec)

    m28 = "ernie-4.5-vl-28b-a3b-pt"
    m = get_model(m28)
    tok = video_grid(m.vision, video_pre(m), 32, 1280, 720).llm_tokens
    q = 32
    text = evaluate(Scenario(model=m28, serving=Serving(phase="prefill", batch=1, prompt=tok + q)))
    mix = evaluate(Scenario(model=m28, serving=Serving(
        phase="prefill", batch=1, prompt=tok + q, image_tokens=tok)))
    past = evaluate(Scenario(model=m28, serving=Serving(
        phase="prefill", batch=1, prompt=tok + q, prefix_cached=tok, image_tokens=tok)))
    bare = evaluate(Scenario(model=m28, serving=Serving(
        phase="prefill", batch=1, prompt=tok + q, prefix_cached=tok)))
    assert mix.stages[0].flops_u != text.stages[0].flops_u
    assert past.stages[0].flops_u == bare.stages[0].flops_u
    dec_v = evaluate(Scenario(model=m28, serving=Serving(phase="decode", batch=1, ctx=16, image_tokens=tok)))
    dec_t = evaluate(Scenario(model=m28, serving=Serving(phase="decode", batch=1, ctx=16)))
    assert dec_v.stages[0].flops_u == dec_t.stages[0].flops_u
    clip = evaluate(Scenario(model=m28, serving=Serving(
        phase="prefill", batch=1, prompt=32, videos=1, video_frames=32, video_w=1280, video_h=720)))
    assert clip.vision["image_tokens"] == 19138
    assert clip.vision["tokens_per_image"] == 1371
    assert clip.vision["video"]["tokens_per_video"] == 19138
    assert clip.stages[0].flops_u != text.stages[0].flops_u


def test_granite_42_30b_untied_keeps_41_gemms():
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
    assert rel["repo"] == "ibm-granite/granite-4.2-30b"
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
    old = load_release(prev.hf_id)
    assert old["config"]["attention_multiplier"] == 0.0078125
    assert old["config"]["embedding_multiplier"] == 12.0
    assert old["config"]["residual_multiplier"] == 0.175
    assert old["config"]["logits_scaling"] == 16.0
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
    assert ids["granite-4.1-30b"]["family"] == "Granite-4.1"
    evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=32)))
    evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, prompt=32)))
