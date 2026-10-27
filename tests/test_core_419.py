"""0.419: Qwen3-VL / WeVisOne LayerNorm and merger erf-GELU; LFM2.5-VL LayerNorm and projector erf-GELU.

Qwen3VLVisionPatchMerger uses nn.GELU() (erf). The block MLP stays gelu_pytorch_tanh.
Siglip2 LayerNorms and Lfm2VlMultiModalProjector ACT2FN["gelu"] are the same recipes.
The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid_of, vision_ops, vision_spec

QWEN = (
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-8b",
    "qwen3-vl-32b",
    "qwen3-vl-30b-a3b",
    "qwen3-vl-235b-a22b",
    "qwen3-vl-2b-fp8",
    "wevisone-vl-2b",
    "wevisone-vl-4b",
)
LFM = (
    "lfm2.5-vl-1.6b",
    "lfm2.5-vl-3b",
    "lfm2.5-vl-450m",
    "d1-3b",
)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _ln(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _gelu(op, rows: int, width: int, layer: int, fmt: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == fmt and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_qwen3vl_bills_layernorm_and_merger_erf_gelu():
    for mid in QWEN:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "qwen" and vs.ln_mode == "qwen3vl", mid
        assert vs.vec_fmt == "bf16" and vs.deepstack == 3 and len(vs.ds_idx) == 3, mid
        rel = load_release(m.hf_id)
        vc = rel["config"]["vision_config"]
        assert vc["hidden_act"] == "gelu_pytorch_tanh", mid
        assert tuple(vc["deepstack_visual_indexes"]) == vs.ds_idx, mid
        assert vs.ds_idx == ((5, 11, 17) if vs.depth == 24 else (8, 16, 24)), mid
        h, depth, inter = vs.hidden, vs.depth, vs.inter
        hm = h * 4
        g = image_grid(vs, 1024, 1024)
        assert (g.ph, g.pw, g.merged) == (64, 64, 1024), mid
        patches = 64 * 64
        ops = vision_ops(vs, g, 1, "bf16")
        by = _by(ops)
        _ln(by["vis.0.attn_ln"], patches, h, 0)
        _ln(by["vis.0.mlp_ln"], patches, h, 0)
        _ln(by[f"vis.{depth - 1}.attn_ln"], patches, h, depth - 1)
        _ln(by["vis.pre_ln"], patches, h, depth)
        assert by["vis.pre_ln"].vec != g.merged * h * 4
        assert by["vis.pre_ln"].vec != patches * hm * 4
        _gelu(by["vis.mgelu"], g.merged, hm, depth, "bf16")
        assert by["vis.mgelu"].vec != g.merged * vs.post[1][1] * 5
        assert by["vis.mgelu"].vec != patches * hm * 5
        for slot, layer in enumerate(vs.ds_idx):
            _ln(by[f"vis.ds{slot}.ln"], g.merged, hm, layer)
            _gelu(by[f"vis.ds{slot}.gelu"], g.merged, hm, layer, "bf16")
            assert by[f"vis.ds{slot}.ln"].vec != patches * hm * 4
        assert by["vis.0.qkv"].vec == patches * (3 * h) * 3
        assert by["vis.0.mlp_in"].vec == patches * inter * 3
        assert by["vis.0.mlp_in"].n == inter
        assert by["vis.merger.0"].m == g.merged and (by["vis.merger.0"].k, by["vis.merger.0"].n) == (hm, hm)
        assert by["vis.merger.1"].m == g.merged and by["vis.merger.1"].k == hm
        assert by["vis.merger.0"].vec == g.merged * hm * 3
        soft = by["vis.0.softmax"]
        assert soft.sfu == 0 and soft.vec == vs.heads * patches * patches * 5
        names = [op.name for op in ops]
        assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.mlp_ln")
        assert names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
        assert names.index("vis.pre_ln") < names.index("vis.merger.0") < names.index("vis.mgelu")
        assert names.index("vis.mgelu") < names.index("vis.merger.1") < names.index("vis.ds0.ln")
        assert names.index("vis.ds0.ln") < names.index("vis.merger.2") < names.index("vis.ds0.gelu")
        assert names.index("vis.ds2.gelu") < names.index("vis.deepstack")
        assert "vis.0.gelu" not in by and "vis.0.silu" not in by and "vis.proj_gelu" not in by
        assert "vis.proj_ln" not in by
        ln_names = [name for name in by if name.endswith(".attn_ln") or name.endswith(".mlp_ln")
                    or name == "vis.pre_ln" or name.endswith(".ln") and name.startswith("vis.ds")]
        assert len(ln_names) == 2 * depth + 1 + 3, mid
        gelu_names = [name for name in by if name == "vis.mgelu" or name.endswith(".gelu")]
        assert gelu_names == ["vis.mgelu", "vis.ds0.gelu", "vis.ds1.gelu", "vis.ds2.gelu"]
        assert sum(op.w_params for name in ln_names + gelu_names for op in (by[name],)) == 0
        text = " ".join(m.notes)
        for needle in ("nn.GELU()", "erf", "gelu_pytorch_tanh", "不单列", "fp32", "4*hidden", "尾项不减"):
            assert needle in text, (mid, needle)

        two = _by(vision_ops(vs, g, 2, "bf16"))
        assert two["vis.0.attn_ln"].vec == 2 * by["vis.0.attn_ln"].vec
        assert two["vis.0.attn_ln"].sfu == 2 * patches
        assert two["vis.pre_ln"].sfu == 2 * patches
        assert two["vis.mgelu"].vec == 2 * by["vis.mgelu"].vec
        assert two["vis.ds1.ln"].vec == 2 * by["vis.ds1.ln"].vec
        assert two["vis.0.qk"].m == patches and two["vis.0.qk"].count == 2 * vs.heads
        assert two["vis.0.softmax"].sfu == 0
        assert two["vis.0.mlp_in"].vec == 2 * by["vis.0.mlp_in"].vec

        wide = image_grid(vs, 1920, 1080)
        assert (wide.ph, wide.pw, wide.merged) == (68, 120, 2040), mid
        wops = _by(vision_ops(vs, wide, 1, "bf16"))
        wpatch = 68 * 120
        _ln(wops["vis.pre_ln"], wpatch, h, depth)
        _gelu(wops["vis.mgelu"], 2040, hm, depth, "bf16")
        _ln(wops["vis.ds2.ln"], 2040, hm, vs.ds_idx[2])
        assert wops["vis.0.qk"].m == wpatch and wops["vis.0.qk"].count == vs.heads

        forced = _by(vision_ops(vs, g, 1, "fp8"))
        if mid.startswith("wevisone"):
            assert set(rel["roles"]["vision"]) == {"fp32"}, mid
            assert by["vis.0.qkv"].w_fmt == "fp32" and by["vis.0.qkv"].w_bits == 32.0
            assert by["vis.0.qkv"].a_fmt == "bf16" and by["vis.merger.0"].w_fmt == "fp32"
            for name in ("vis.0.qkv", "vis.0.mlp_in", "vis.merger.0", "vis.merger.1"):
                assert forced[name].a_fmt == "bf16" and forced[name].w_fmt == "fp32", (mid, name)
                assert forced[name].vec == by[name].vec
        else:
            assert set(rel["roles"]["vision"]) == {"bf16"}, mid
            assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
            assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.qkv"].w_fmt == "bf16"
            assert forced["vis.0.mlp_in"].a_fmt == "fp8" and forced["vis.merger.0"].w_bits == 16.0
            assert forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
        for name in ("vis.mgelu", "vis.ds0.gelu", "vis.ds2.gelu"):
            assert forced[name].a_fmt == "bf16" and forced[name].vec == by[name].vec, (mid, name)
        for name in ("vis.0.attn_ln", "vis.0.mlp_ln", "vis.pre_ln", "vis.ds0.ln"):
            assert forced[name].a_fmt == "fp32" and forced[name].vec == by[name].vec, (mid, name)

    m2 = get_model("qwen3-vl-2b")
    clip = video_grid_of(m2, Scenario(model="qwen3-vl-2b").replace(
        "serving.videos", 1).replace("serving.video_frames", 4).serving)
    assert (clip.slice.ph, clip.slice.pw, clip.slice.merged) == (44, 80, 880)
    vops = _by(vision_ops(m2.vision, clip.slice, 1))
    _ln(vops["vis.pre_ln"], 44 * 80, 1024, 24)
    _gelu(vops["vis.mgelu"], 880, 4096, 24, "bf16")
    assert vops["vis.ds0.ln"].layer == 5 and vops["vis.ds2.gelu"].layer == 17
    pre = evaluate(Scenario(model="qwen3-vl-2b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1024 and pre.step > 0
    wpre = evaluate(Scenario(model="wevisone-vl-2b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert wpre.vision["image_tokens"] == 1024 and wpre.step > 0


def test_qwen3vl_refuses_other_acts_and_a_missing_dtype():
    rel = load_release("Qwen/Qwen3-VL-2B-Instruct")
    silu = copy.deepcopy(rel["config"])
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(rel["repo"], silu), "qwen3vl hidden_act is not gelu_pytorch_tanh")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("hidden_act")
    _raises(lambda: vision_spec(rel["repo"], missing), "qwen3vl hidden_act is not gelu_pytorch_tanh")
    bare = vision_spec(rel["repo"], rel["config"])
    assert bare is not None and bare.ln_mode == "qwen3vl" and bare.vec_fmt == ""
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "qwen3vl activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "qwen3vl activation dtype is not a vision float")
    old = get_model("qwen3.5-9b")
    # 0.424 bills this tower as ln_mode qwen35. It is still not the Qwen3-VL deepstack path.
    assert old.vision is not None and old.vision.ln_mode == "qwen35" and old.vision.deepstack == 0
    names = [op.name for op in vision_ops(old.vision, image_grid(old.vision, 1024, 1024), 1)]
    for needle in ("vis.ds0.ln", "vis.ds0.gelu", "vis.pgelu", "vis.post_ln"):
        assert needle not in names, needle
    assert "vis.pre_ln" in names and "vis.mgelu" in names and "vis.0.attn_ln" in names
    ernie = get_model("ernie-4.5-vl-28b-a3b")
    enames = [op.name for op in vision_ops(ernie.vision, image_grid(ernie.vision, 1024, 1024), 1)]
    # 0.433 bills ERNIE's own LayerNorm as attn_ln. The Qwen3-VL merger names stay off this tower.
    assert "vis.mgelu" not in enames and "vis.pre_ln" not in enames and "vis.ds0.ln" not in enames
    assert "vis.0.attn_ln" in enames and "vis.out_ln" in enames and "vis.sp_gelu" in enames


def test_lfm2_vl_bills_layernorm_and_projector_erf_gelu():
    for mid in LFM:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "lfm2vl" and vs.vec_fmt == "bf16", mid
        assert vs.gated is False and dict(vs.pre)["proj_ln"] == 0 and dict(vs.pre)["pad"] == 1024, mid
        rel = load_release(m.hf_id)
        assert rel["config"]["vision_config"]["hidden_act"] == "gelu_pytorch_tanh", mid
        assert rel["config"]["projector_hidden_act"] == "gelu", mid
        assert rel["config"]["projector_use_layernorm"] is False, mid
        assert set(rel["roles"]["vision"]) == {"bf16"}, mid
        h, depth = vs.hidden, vs.depth
        one = image_grid(vs, 256, 256)
        assert (one.ph, one.pw, one.merged, one.views) == (16, 16, 64, ((16, 16, 1),)), mid
        ops = vision_ops(vs, one, 1, "bf16")
        by = _by(ops)
        pad = 1024
        _ln(by["vis.0.attn_ln"], pad, h, 0)
        _ln(by["vis.0.mlp_ln"], pad, h, 0)
        _ln(by[f"vis.{depth - 1}.mlp_ln"], pad, h, depth - 1)
        _ln(by["vis.post_ln"], pad, h, depth)
        assert by["vis.post_ln"].vec != 64 * h * 4
        gelu_w = vs.post[0][1]
        _gelu(by["vis.pgelu"], 64, gelu_w, depth, "bf16")
        assert by["vis.pgelu"].vec != pad * gelu_w * 5
        assert by["vis.pgelu"].vec != 64 * h * 5
        assert by["vis.0.fc1"].vec == pad * vs.inter * 3
        assert by["vis.0.fc1"].w_fmt == "bf16" and by["vis.0.fc1"].w_bits == 16.0
        assert by["vis.0.q"].m == pad and by["vis.merger.0"].m == 64
        assert (by["vis.merger.0"].k, by["vis.merger.0"].n) == vs.post[0]
        assert by["vis.merger.1"].k == gelu_w
        assert by["vis.0.softmax"].sfu == 0
        names = [op.name for op in ops]
        assert names.index("vis.0.attn_ln") < names.index("vis.0.q") < names.index("vis.0.mlp_ln")
        assert names.index("vis.0.mlp_ln") < names.index("vis.0.fc1") < names.index("vis.post_ln")
        assert names.index("vis.post_ln") < names.index("vis.merger.0") < names.index("vis.pgelu")
        assert names.index("vis.pgelu") < names.index("vis.merger.1")
        assert "vis.proj_ln" not in by and "vis.0.gelu" not in by and "vis.mgelu" not in by
        assert len([name for name in by if name.endswith(".attn_ln") or name.endswith(".mlp_ln")]) == 2 * depth
        assert sum(by[name].w_params for name in by if name.endswith((".attn_ln", ".mlp_ln"))
                   or name in ("vis.post_ln", "vis.pgelu")) == 0
        text = " ".join(m.notes)
        for needle in ("ACT2FN['gelu']", "erf", "gelu_pytorch_tanh", "post_layernorm", "fp32", "尾项不减"):
            assert needle in text, (mid, needle)

        two = _by(vision_ops(vs, one, 2, "bf16"))
        assert two["vis.post_ln"].vec == 2 * by["vis.post_ln"].vec
        assert two["vis.post_ln"].sfu == 2 * pad
        assert two["vis.pgelu"].vec == 2 * by["vis.pgelu"].vec
        assert two["vis.0.qk"].m == pad and two["vis.0.qk"].count == 2 * vs.heads
        assert two["vis.0.fc1"].vec == 2 * by["vis.0.fc1"].vec

        forced = _by(vision_ops(vs, one, 1, "fp8"))
        for name in ("vis.patch_embed", "vis.0.q", "vis.0.fc1", "vis.merger.0", "vis.merger.1"):
            assert forced[name].a_fmt == "fp8" and forced[name].w_fmt == "bf16", (mid, name)
            assert forced[name].w_bits == 16.0 and forced[name].vec == by[name].vec
        assert forced["vis.pgelu"].a_fmt == "bf16" and forced["vis.post_ln"].a_fmt == "fp32"
        assert forced["vis.0.attn_ln"].vec == by["vis.0.attn_ln"].vec

    small = get_model("lfm2.5-vl-1.6b")
    wide = image_grid(small.vision, 1280, 720)
    assert wide.views == ((32, 32, 2), (24, 42, 1))
    split = _by(vision_ops(small.vision, wide, 1))
    _ln(split["vis.post_ln"], 2 * 1024, 1152, 27)
    _ln(split["vis.v1.post_ln"], 1024, 1152, 27)
    _gelu(split["vis.pgelu"], 512, 2048, 27, "bf16")
    _gelu(split["vis.v1.pgelu"], 252, 2048, 27, "bf16")
    assert split["vis.0.fc1"].m == 2048 and split["vis.v1.merger.0"].m == 252
    tiny = get_model("lfm2.5-vl-450m")
    tops = _by(vision_ops(tiny.vision, image_grid(tiny.vision, 256, 256), 1))
    _ln(tops["vis.post_ln"], 1024, 768, 12)
    _gelu(tops["vis.pgelu"], 64, 2048, 12, "bf16")
    assert tops["vis.pgelu"].vec != 64 * 1024 * 5
    assert len([name for name in tops if name.endswith(".attn_ln")]) == 12
    # 0.420 bills d1-omni's own SigLIP2 LayerNorms and projector F.gelu under these names.
    # The Qwen merger names stay off this tower.
    d1 = get_model("d1-omni-600m")
    dnames = [op.name for op in vision_ops(d1.vision, image_grid(d1.vision, 1024, 1024), 1)]
    for needle in ("vis.mgelu", "vis.pre_ln"):
        assert needle not in dnames, needle
    pre = evaluate(Scenario(model="lfm2.5-vl-1.6b", serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1287 and pre.step > 0


def test_lfm2_vl_refuses_other_acts_and_bills_projector_layernorm_when_enabled():
    rel = load_release("LiquidAI/LFM2.5-VL-1.6B")
    tanh = copy.deepcopy(rel["config"])
    tanh["projector_hidden_act"] = "gelu_pytorch_tanh"
    _raises(lambda: vision_spec(rel["repo"], tanh), "lfm2_vl projector_hidden_act is not gelu")
    quick = copy.deepcopy(rel["config"])
    quick["projector_hidden_act"] = "quick_gelu"
    _raises(lambda: vision_spec(rel["repo"], quick), "lfm2_vl projector_hidden_act is not gelu")
    tower = copy.deepcopy(rel["config"])
    tower["vision_config"]["hidden_act"] = "gelu"
    _raises(lambda: vision_spec(rel["repo"], tower), "lfm2_vl hidden_act is not gelu_pytorch_tanh")
    bare = vision_spec(rel["repo"], rel["config"])
    assert bare is not None and bare.vec_fmt == ""
    g = image_grid(bare, 256, 256)
    _raises(lambda: vision_ops(bare, g, 1), "lfm2_vl activation dtype is not a vision float")
    on = copy.deepcopy(rel["config"])
    on["projector_use_layernorm"] = True
    spec = from_release("synth", rel["repo"], rel=rel, cfg=on)
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16"
    assert dict(spec.vision.pre)["proj_ln"] == 1
    ops = _by(vision_ops(spec.vision, g, 1))
    _ln(ops["vis.proj_ln"], 64, spec.vision.post[0][0], spec.vision.depth)
    assert ops["vis.proj_ln"].vec != 1024 * spec.vision.post[0][0] * 4
    names = [op.name for op in vision_ops(spec.vision, g, 1)]
    assert names.index("vis.post_ln") < names.index("vis.proj_ln") < names.index("vis.merger.0")
