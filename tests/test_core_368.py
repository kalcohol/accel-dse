"""0.368: Gemma-4 hybrid sliding / full attention, ViT or unified embedder, MoE on 26B-A4B.

Counts are the release headers (2026-10-24). E2B / E4B landed in 0.369 (test_core_369).
This file keeps the six 0.368 rows as a prefix of the Gemma-4 family.
"""
from __future__ import annotations

import importlib.util
import math

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, image_grid, vision_act_peak, vision_ops

# id, llm, vision, layers, hidden, sliding, full, slide (n_kv, hd), full (n_kv, hd), moe
ROWS = (
    ("gemma-4-31b-it", 30_697_345_340, 575_743_536, 60, 5376, 50, 10, (16, 256), (4, 512), False),
    ("gemma-4-31b", 30_697_345_340, 575_743_536, 60, 5376, 50, 10, (16, 256), (4, 512), False),
    ("gemma-4-26b-a4b-it", 25_233_141_790, 572_794_416, 30, 2816, 25, 5, (8, 256), (2, 512), True),
    ("gemma-4-26b-a4b", 25_233_141_790, 572_794_416, 30, 2816, 25, 5, (8, 256), (2, 512), True),
    ("gemma-4-12b-it", 11_907_350_320, 52_379_904, 48, 3840, 40, 8, (8, 256), (1, 512), False),
    ("gemma-4-12b", 11_907_350_320, 52_379_904, 48, 3840, 40, 8, (8, 256), (1, 512), False),
)


def _kv(layer, ctx: int) -> float:
    c = layer.core
    n = min(ctx, c.window) if c.window else ctx
    return n * c.kv_elems_per_token()


def test_catalog_order_and_param_check():
    ids = [e["id"] for e in list_models() if e["family"] == "Gemma-4"]
    assert ids[:len(ROWS)] == [r[0] for r in ROWS]
    for mid, params, vis, layers, hidden, n_s, n_f, slide, full, moe in ROWS:
        m = get_model(mid)
        rel = load_release(m.hf_id)
        pc = m.param_check()
        assert pc["rel_err"] == 0.0 and m.params() == params == m.release_params, mid
        assert m.vision is not None and m.vision.params == vis == rel["params_vision"], mid
        assert m.vision.extra == 0 and m.coverage == "full" and m.coverage_reasons == (), mid
        assert m.n_layers == layers and m.hidden == hidden and m.tie_embeddings and m.vocab == 262144, mid
        assert m.act_fmt == "bf16" and m.fmt("attn").fmt == "bf16" and m.fmt("attn").bits == 16.0, mid
        sliding = [L for L in m.layers if L.core.window is not None]
        glob = [L for L in m.layers if L.core.window is None]
        assert len(sliding) == n_s and len(glob) == n_f, mid
        assert all(L.core.window == 1024 and (L.core.n_kv, L.core.qk_dim, L.core.v_dim) == (*slide, slide[1])
                   for L in sliding), mid
        assert all((L.core.n_kv, L.core.qk_dim, L.core.v_dim) == (*full, full[1]) for L in glob), mid
        assert all(any(lin.name == "v" for lin in L.attn_linears) for L in sliding), mid
        assert all(not any(lin.name == "v" for lin in L.attn_linears) for L in glob), mid
        ctx = 4096
        assert sum(_kv(L, ctx) for L in m.layers) < sum(ctx * L.core.kv_elems_per_token() for L in m.layers), mid
        if moe:
            f = m.layers[0].ffn
            assert m.is_moe and (f.n_experts, f.top_k, f.d_expert, f.n_shared, f.d_shared) == (128, 8, 704, 1, 2112), mid
            assert all(L.ffn.kind == "moe" for L in m.layers), mid
        else:
            assert not m.is_moe and all(L.ffn.kind == "dense" for L in m.layers), mid


def test_image_grid_and_padded_tower():
    m = get_model("gemma-4-31b-it")
    g = image_grid(m.vision, 1024, 1024)
    assert (g.px, g.ph, g.pw, g.llm_tokens) == ((768, 768), 48, 48, 256)
    wide = image_grid(m.vision, 1920, 1080)
    assert (wide.px, wide.llm_tokens) == ((1056, 576), 264)
    ops = vision_ops(m.vision, g, 1)
    q = next(o for o in ops if o.name == "vis.0.q")
    assert q.m == 280 * 9 and q.k == 1152 and q.n == 1152
    proj = next(o for o in ops if o.name == "vis.proj")
    assert proj.m == 256 and (proj.k, proj.n) == (1152, 5376)
    assert all(o.name != "vis.deepstack" for o in ops)
    peak = vision_act_peak(m.vision, g, 1)
    assert peak == max(o.act_bytes for o in ops if o.kind == "gemm")
    u = get_model("gemma-4-12b-it")
    ug = image_grid(u.vision, 1024, 1024)
    assert ug.llm_tokens == 256 and u.vision.family == "gemma4u" and u.vision.depth == 0
    uops = vision_ops(u.vision, ug, 1)
    assert [o.name for o in uops] == ["vis.ln1", "vis.patch_dense", "vis.ln2", "vis.pos_ln", "vis.proj"]
    dense = next(o for o in uops if o.name == "vis.patch_dense")
    proj_u = next(o for o in uops if o.name == "vis.proj")
    assert dense.m == 280 and (dense.k, dense.n) == (6912, 3840)
    assert (proj_u.k, proj_u.n) == (3840, 3840)
    assert u.vision.audio is not None and u.vision.audio.kind == "g4lin"
    assert u.vision.audio.tokens(1.0) == math.ceil(16000 / 640)
    aops = audio_ops(u.vision.audio, 1.0, 1)
    assert len(aops) == 1 and (aops[0].m, aops[0].k, aops[0].n) == (u.vision.audio.tokens(1.0), 640, 3840)
    assert get_model("gemma-4-31b-it").vision.audio is None


def test_smoke_eval_and_neighbors():
    r = evaluate(Scenario(model="gemma-4-31b-it", serving=Serving(phase="prefill", prompt=128, images=1)))
    assert r.vision["image_tokens"] == 256
    evaluate(Scenario(model="gemma-4-26b-a4b-it", serving=Serving(phase="decode", batch=2, ctx=2048)))
    evaluate(Scenario(model="gemma-4-12b-it", serving=Serving(phase="prefill", prompt=64, images=1)))
    q = get_model("qwen3-vl-8b")
    g = image_grid(q.vision, 1024, 1024)
    ops = vision_ops(q.vision, g, 1)
    ds = next(o for o in ops if o.name == "vis.deepstack")
    assert ds.stream is True and ds.kind == "vector"
    assert vision_act_peak(q.vision, g, 1) == max(o.act_bytes for o in ops if o.kind == "gemm")
    fp8 = get_model("qwen3-coder-next-fp8")
    assert fp8.fmt("lin_ba").fmt == "bf16" and fp8.fmt("lin_ba").bits == 16.0
    mtp = get_model("qwen3.8-27b-fp8")
    assert mtp.fmt("mtp").fmt == "bf16" and mtp.fmt("mtp").bits == 16.0


def test_router_and_std_scale_are_parameters():
    spec = importlib.util.spec_from_file_location("summarize_release", "scripts/summarize_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in (
        "model.language_model.layers.0.router.scale",
        "model.language_model.layers.0.router.per_expert_scale",
        "model.vision_tower.std_scale",
    ):
        assert mod.NOT_SCALE_RE.search(name) and mod.SCALE_RE.search(name), name
        # SCALE_RE still matches; NOT_SCALE_RE is what keeps the tensor as a parameter.
        assert mod._base(name) != name, name
    inv = "model.language_model.layers.0.mlp.experts.down_proj_scale_inv"
    assert mod.SCALE_RE.search(inv) and not mod.NOT_SCALE_RE.search(inv)
    assert mod._base(inv) == inv[: -len("_scale_inv")]
    rel = load_release("google/gemma-4-26B-A4B-it")
    assert "scale" not in rel["roles"]
    assert rel["roles"]["router"]["bf16"]["params"] == 30 * (2816 * 128 + 2816 + 128)
