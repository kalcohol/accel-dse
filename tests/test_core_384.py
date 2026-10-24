"""0.384: 0.383 LFM2.5 / Granite-4.2 review holds; ERNIE-4.5-VL dual-pool MoE closes.

LFM2.5-2.6B stays the dense hybrid (22 conv kernel 3 + 8 GQA). Granite-4.2-3b / 8b
stay untied copies of the 4.1 GEMMs, with embedding / residual / logits multipliers
at 1.0. ERNIE-4.5-VL bills the text expert pool and the vision expert pool from the
public ernie4_5_moe_vl class. A text step skips the vision pool. Image tokens,
including the start and end markers, use it.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import _mm_in_step, evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_pre, vision_ops


def _weights(m):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0).weights


def _summarize():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_release.py"
    spec = importlib.util.spec_from_file_location("summarize_release_384", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pairs(op):
    return op.count * op.m * op.share


class _Sv:
    def __init__(self, image_tokens, phase="prefill", prefix_cached=0):
        self.image_tokens = image_tokens
        self.phase = phase
        self.prefix_cached = prefix_cached


def test_review_383_lfm_and_granite_unchanged():
    lfm = get_model("lfm2.5-2.6b")
    assert lfm.param_check()["rel_err"] == 0.0
    assert lfm.params() == 2_697_198_592 and lfm.tie_embeddings is True
    assert sum(L.core.lin == "conv" and L.core.conv_kernel == 3 for L in lfm.layers) == 22
    assert sum(L.core.kind == "gqa" for L in lfm.layers) == 8
    assert lfm.mtp_layers == () and lfm.vision is None

    rows = {
        "granite-4.2-3b": ("granite-4.1-3b", 0.015625),
        "granite-4.2-8b": ("granite-4.1-8b", 0.0078125),
    }
    for mid, (old, attn_m) in rows.items():
        m = get_model(mid)
        prev = get_model(old)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.tie_embeddings is False
        assert prev.tie_embeddings is True
        assert m.params() == prev.params() + rel["config"]["vocab_size"] * m.hidden
        assert rel["config"]["attention_multiplier"] == attn_m
        assert rel["config"]["embedding_multiplier"] == 1.0
        assert rel["config"]["residual_multiplier"] == 1.0
        assert rel["config"]["logits_scaling"] == 1.0
        assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].attn_linears] == [
            (lin.name, lin.k, lin.n) for lin in prev.layers[0].attn_linears]
        assert [(lin.name, lin.k, lin.n) for lin in m.layers[0].ffn_linears] == [
            (lin.name, lin.k, lin.n) for lin in prev.layers[0].ffn_linears]


def test_ernie45_vl_dual_pool_matches_headers():
    ids = {d["id"]: d for d in list_models()}
    role = _summarize().classify
    specs = {
        "ernie-4.5-vl-28b-a3b-pt": dict(
            repo="baidu/ERNIE-4.5-VL-28B-A3B-PT", n_tensors=11105, h=2560, layers=28, heads=20, kv=4,
            d_dense=12288, vocab=103424, tie=True, k=6, de=1536, deb=512, n_shared=1, d_shared=3072,
            dense_n=1, moe_at=1, n_moe=27, params=28_624_635_776, vision=774_658_560,
            bytes_llm=57_266_973_184, bytes_total=58_816_290_304, text_h_out=2560),
        "ernie-4.5-vl-424b-a47b-pt": dict(
            repo="baidu/ERNIE-4.5-VL-424B-A47B-PT", n_tensors=20475, h=8192, layers=54, heads=64, kv=8,
            d_dense=28672, vocab=103424, tie=False, k=8, de=3584, deb=1536, n_shared=0, d_shared=0,
            dense_n=3, moe_at=3, n_moe=51, params=422_722_779_520, vision=803_505_664,
            bytes_llm=845_552_526_848, bytes_total=847_159_538_176, text_h_out=8192),
    }
    for mid, s in specs.items():
        m = get_model(mid)
        rel = load_release(m.hf_id)
        h, E = s["h"], 64
        hd = h // s["heads"]
        assert hd == 128 and "head_dim" not in rel["config"]
        assert "num_nextn_predict_layers" not in rel["config"]
        assert "moe_use_hard_gate" not in rel["config"]
        attn = 2 * h * h + 2 * h * s["kv"] * hd
        dense = attn + 3 * h * s["d_dense"] + 2 * h
        shared = 3 * h * s["d_shared"]
        expert = E * 3 * h * s["de"] + E * 3 * h * s["deb"]
        routers = 2 * h * E
        bias = 2 * E
        moe = attn + expert + shared + routers + bias + 2 * h
        head = s["vocab"] * h if s["tie"] else 2 * s["vocab"] * h
        expect = s["dense_n"] * dense + s["n_moe"] * moe + head + h
        assert expect == s["params"]
        assert m.param_check()["rel_err"] == 0.0
        assert m.params() == m.release_params == rel["params_llm"] == expect
        assert m.mtp_params() == rel["params_mtp"] == 0 and m.mtp_layers == ()
        assert m.coverage == "full" and m.coverage_reasons == ()
        assert m.tie_embeddings is s["tie"] and m.hidden == h and m.vocab == s["vocab"]
        assert m.n_layers == s["layers"] and m.vision_params == s["vision"] == rel["params_vision"]
        assert rel["n_tensors"] == s["n_tensors"] and rel["repo"] == s["repo"]
        assert rel["bytes_llm"] == s["bytes_llm"] and rel["bytes_total"] == s["bytes_total"]
        assert rel["torch_dtype"] == "bfloat16" and rel["quantization_config"] is None
        assert rel.get("domain") in (None, "llm")
        assert set(rel["roles"]["router"]) == {"fp32"}
        assert rel["roles"]["router"]["fp32"]["params"] == s["n_moe"] * (routers + bias)
        assert rel["roles"]["router"]["fp32"]["bytes"] == 4 * s["n_moe"] * (routers + bias)
        assert rel["roles"]["expert"]["bf16"]["params"] == s["n_moe"] * expert
        assert rel["roles"]["mlp"]["bf16"]["params"] == s["dense_n"] * 3 * h * s["d_dense"]
        assert rel["roles"]["vision"]["fp16"]["params"] == 630_433_280
        assert ("lm_head" in rel["roles"]) is (not s["tie"])
        if s["n_shared"]:
            assert rel["roles"]["shared_expert"]["bf16"]["params"] == s["n_moe"] * shared
        else:
            assert "shared_expert" not in rel["roles"]
        vis_b = sum(v["bytes"] for v in rel["roles"]["vision"].values())
        assert rel["bytes_total"] - rel["bytes_llm"] == vis_b == 2 * s["vision"]

        assert all(L.ffn.kind == "dense" and L.ffn.d_ff == s["d_dense"] for L in m.layers[:s["dense_n"]])
        assert all(L.ffn.kind == "moe" for L in m.layers[s["moe_at"]:])
        layer = m.layers[s["moe_at"]]
        assert (layer.core.n_q, layer.core.n_kv, layer.core.qk_dim) == (s["heads"], s["kv"], hd)
        assert layer.ffn.n_experts == E and layer.ffn.n_experts_b == E
        assert layer.ffn.d_expert == s["de"] and layer.ffn.d_expert_b == s["deb"]
        assert layer.ffn.top_k == s["k"] and layer.ffn.n_shared == s["n_shared"]
        assert layer.ffn.d_shared == s["d_shared"] and layer.misc_params == 2 * h + bias
        assert layer.expert_params(h) == expert
        names = [lin.name for lin in layer.ffn_linears]
        assert "router" in names and "router_b" in names
        if s["n_shared"]:
            up = next(lin for lin in layer.ffn_linears if lin.name == "shared_gate_up")
            down = next(lin for lin in layer.ffn_linears if lin.name == "shared_down")
            assert (up.k, up.n) == (h, 2 * s["d_shared"])
            assert (down.k, down.n) == (s["d_shared"], h)
        else:
            assert not any(lin.name.startswith("shared") for lin in layer.ffn_linears)
        assert m.fmt("router").fmt == "fp32" and m.fmt("expert").fmt == "bf16"
        assert m.vision.family == "ernievl" and m.vision.wfmt == () and m.vision.w_Bpp == 2.0
        assert m.vision.gated is False and m.vision.depth == 32 and m.vision.hidden == 1280
        # 0.385 wires Ernie4_5_VLProcessor. The still-image grid stays one frame.
        assert video_pre(m) == {"ernie_temporal": 2, "min_px": 299 * 28 * 28, "max_px": 1196 * 28 * 28}
        assert ids[mid]["family"] == "ERNIE-4.5-VL" and ids[mid]["domain"] == "vlm"
        assert ids[mid]["provider"] == "baidu"

        g = image_grid(m.vision, 1024, 1024)
        assert (g.ph, g.pw, g.merged, g.llm_tokens, g.px, g.tframes) == (74, 74, 1369, 1371, (1036, 1036), 1)
        tiny = image_grid(m.vision, 8, 8)
        assert (tiny.ph, tiny.pw, tiny.merged, tiny.llm_tokens, tiny.px) == (4, 4, 4, 6, (56, 56))
        ops = vision_ops(m.vision, g, 1)
        patch = next(o for o in ops if o.name == "vis.patch_embed")
        assert (patch.m, patch.k, patch.n, patch.w_fmt) == (74 * 74, 588, 1280, "bf16")
        mergers = [o for o in ops if o.name.startswith("vis.merger")]
        assert [(o.m, o.k, o.n) for o in mergers] == [
            (1369, 5120, 5120), (1369, 5120, 5120), (1369, 10240, 5120),
            (1369, 5120, 5120), (1369, 5120, s["text_h_out"])]
        assert sum(o.name.endswith(".qk") for o in ops) == 32
        mlp = next(o for o in ops if o.name == "vis.0.mlp_in")
        assert mlp.n == 5120

        li = s["moe_at"]
        dec = build_rank_ops(m, li, Phase("decode", 1, 1, 16), Shard())
        assert not any("router_b" in o.name or "expert_b" in o.name for o in dec)
        rt = next(o for o in dec if o.name == "moe.router")
        assert (rt.m, rt.k, rt.n, rt.w_fmt, rt.w_bits) == (1, h, E, "fp32", 32.0)
        gate = next(o for o in dec if o.name == "expert.gate_up")
        assert _pairs(gate) == s["k"] and (gate.k, gate.n) == (h, 2 * s["de"])

        mm, q = 8, 32
        pre = build_rank_ops(m, li, Phase("prefill", 1, q, 0, mm=mm), Shard())
        ra = next(o for o in pre if o.name == "moe.router")
        rb = next(o for o in pre if o.name == "moe.router_b")
        assert (ra.m, rb.m, ra.n, rb.n) == (q, q, E, E)
        ga = next(o for o in pre if o.name == "expert.gate_up")
        gb = next(o for o in pre if o.name == "expert_b.gate_up")
        assert _pairs(ga) == (q - mm) * s["k"] and (ga.k, ga.n) == (h, 2 * s["de"])
        assert _pairs(gb) == mm * s["k"] and (gb.k, gb.n) == (h, 2 * s["deb"])
        if s["n_shared"]:
            sh = next(o for o in pre if o.name == "moe.shared_gate_up")
            assert sh.m == q

        ep = build_rank_ops(m, li, Phase("prefill", 1, q, 0, mm=mm), Shard(tp=2, ep=2))
        assert any(o.name == "moe_dispatch" for o in ep)
        assert any(o.name == "moe_dispatch_b" for o in ep)
        assert any(o.name == "moe_combine_b" for o in ep)

        idle_a = (E - s["k"]) * 3 * h * s["de"] * s["n_moe"]
        pool_b = E * 3 * h * s["deb"] * s["n_moe"]
        assert m.active_params() == expect - idle_a - pool_b
        # fp32 bias is 4 bytes in the release and 2 bytes in LLM misc. The final RMSNorm
        # is in params() and is not in this LLM stage_storage weight sum (same h*2 as 0.377).
        assert _weights(m) - rel["bytes_llm"] == -(s["n_moe"] * bias * 2 + h * 2)

        text = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, prompt=q)))
        mix = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, prompt=q, image_tokens=mm)))
        past = evaluate(Scenario(model=mid, serving=Serving(
            phase="prefill", batch=1, prompt=q, prefix_cached=mm, image_tokens=mm)))
        bare = evaluate(Scenario(model=mid, serving=Serving(
            phase="prefill", batch=1, prompt=q, prefix_cached=mm)))
        assert mix.stages[0].flops_u != text.stages[0].flops_u
        assert past.stages[0].flops_u == bare.stages[0].flops_u
        one = (sum(o.flops * o.share for o in pre)
               - sum(o.flops * o.share for o in build_rank_ops(m, li, Phase("prefill", 1, q, 0), Shard())))
        assert mix.stages[0].flops_u - text.stages[0].flops_u == s["n_moe"] * one
        dec_img = evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=16, image_tokens=mm)))
        dec_txt = evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, ctx=16)))
        assert dec_img.stages[0].flops_u == dec_txt.stages[0].flops_u
        still = evaluate(Scenario(model=mid, serving=Serving(
            phase="prefill", batch=1, prompt=32, images=1, image_w=8, image_h=8)))
        assert still.vision["image_tokens"] == 6 and still.vision["tokens_per_image"] == 6
        assert still.step > 0

    assert _mm_in_step(_Sv(1371), 64) == 64
    assert _mm_in_step(_Sv(1371, prefix_cached=1371), 64) == 0
    assert _mm_in_step(_Sv(1371, prefix_cached=1360), 20) == 11
    assert _mm_in_step(_Sv(100, phase="decode"), 1) == 0
    assert _mm_in_step(_Sv(100), 10, cp=4, prompt=40) == 10
    assert _mm_in_step(_Sv(10), 10, cp=4, prompt=40) == 2

    rel = load_release("baidu/ERNIE-4.5-VL-28B-A3B-PT")
    bad = copy.deepcopy(rel["config"])
    bad["moe_layer_end_index"] = [28, 10]
    try:
        from_release("synth", rel["repo"], rel=rel, cfg=bad)
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("an identity expert pool was accepted")
    nxt = copy.deepcopy(rel["config"])
    nxt["num_nextn_predict_layers"] = 1
    try:
        from_release("synth", rel["repo"], rel=rel, cfg=nxt)
    except ValueError as exc:
        assert "MTP" in str(exc)
    else:
        raise AssertionError("a missing MTP module was invented")

    assert role("model.layers.1.mlp.gate.weight", 28) == "router"
    assert role("model.layers.1.mlp.gate.weight_1", 28) == "router"
    assert role("model.layers.1.mlp.moe_statics.e_score_correction_bias", 28) == "router"
    assert role("model.layers.1.mlp.experts.0.gate_proj.weight", 28) == "expert"
    assert role("model.layers.1.mlp.experts.64.down_proj.weight", 28) == "expert"
    assert role("model.layers.1.mlp.shared_experts.down_proj.weight", 28) == "shared_expert"
    assert role("model.layers.0.mlp.gate_proj.weight", 28) == "mlp"
    assert role("resampler_model.spatial_linear.0.weight", 28) == "vision"
    assert role("vision_model.blocks.0.attn.qkv.weight", 28) == "vision"
    ernie = get_model("ernie-4.5-21b-a3b-pt")
    assert ernie.layers[1].ffn.n_experts_b == 0 and ernie.param_check()["rel_err"] == 0.0
    assert ids["ernie-4.5-21b-a3b-pt"]["family"] == "ERNIE-4.5"
