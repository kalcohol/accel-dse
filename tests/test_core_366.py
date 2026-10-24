"""0.366: Qwen3-VL / Qwen3-VL-MoE on the existing Qwen ViT, plus one post-shuffle merger per deepstack index.

Counts are the release headers (2026-10-24). A non-empty deepstack list on Qwen3.5 stays refused
(tests/test_core_365.py).
"""
from __future__ import annotations

import importlib.util

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid_of, vision_ops, vision_spec

# id, llm params, vision params, layers, hidden, n_q, n_kv, ffn width, moe, tie, v_depth, v_hidden, v_out
ROWS = (
    ("qwen3-vl-2b", 1_720_574_976, 406_957_056, 28, 2048, 16, 8, 6144, False, True, 24, 1024, 2048),
    ("qwen3-vl-2b-fp8", 1_720_574_976, 406_957_056, 28, 2048, 16, 8, 6144, False, True, 24, 1024, 2048),
    ("qwen3-vl-4b", 4_022_468_096, 415_347_712, 36, 2560, 32, 8, 9728, False, True, 24, 1024, 2560),
    ("qwen3-vl-4b-fp8", 4_022_468_096, 415_347_712, 36, 2560, 32, 8, 9728, False, True, 24, 1024, 2560),
    ("qwen3-vl-8b", 8_190_735_360, 576_388_336, 36, 4096, 32, 8, 12288, False, False, 27, 1152, 4096),
    ("qwen3-vl-8b-fp8", 8_190_735_360, 576_388_336, 36, 4096, 32, 8, 12288, False, False, 27, 1152, 4096),
    ("qwen3-vl-32b", 32_762_123_264, 595_266_800, 64, 5120, 64, 8, 25600, False, False, 27, 1152, 5120),
    ("qwen3-vl-32b-fp8", 32_762_123_264, 595_266_800, 64, 5120, 64, 8, 25600, False, False, 27, 1152, 5120),
    ("qwen3-vl-30b-a3b", 30_532_122_624, 538_631_408, 48, 2048, 32, 4, 768, True, False, 27, 1152, 2048),
    ("qwen3-vl-30b-a3b-fp8", 30_532_122_624, 538_631_408, 48, 2048, 32, 4, 768, True, False, 27, 1152, 2048),
    ("qwen3-vl-235b-a22b", 235_093_634_560, 576_388_336, 94, 4096, 64, 4, 1536, True, False, 27, 1152, 4096),
    ("qwen3-vl-235b-a22b-fp8", 235_093_634_560, 576_388_336, 94, 4096, 64, 4, 1536, True, False, 27, 1152, 4096),
)


def test_catalog_order_and_release_counts():
    ents = {e["id"]: e for e in list_models()}
    ids = [e["id"] for e in list_models() if e["family"] == "Qwen3-VL"]
    assert ids == [r[0] for r in sorted(ROWS, key=lambda r: (-r[1], r[0].endswith("-fp8"), r[0]))]
    for mid, params, vis, layers, hidden, nq, nkv, width, moe, tie, depth, vh, out in ROWS:
        m = get_model(mid)
        e = ents[mid]
        rel = load_release(m.hf_id)
        pc = m.param_check()
        assert pc["rel_err"] == 0.0 and m.params() == params == m.release_params, mid
        assert m.vision.params == vis == rel["params_vision"] and m.vision.extra == 0, mid
        assert rel["roles"]["vision"]["bf16"]["bytes"] == 2 * vis, mid
        assert m.n_layers == layers and m.hidden == hidden and m.vocab == 151936 and m.tie_embeddings is tie, mid
        core = m.layers[0].core
        assert (core.n_q, core.n_kv) == (nq, nkv), mid
        assert m.is_moe is moe and e["family"] == "Qwen3-VL" and e["domain"] == "vlm" and e["coverage"] == "full", mid
        assert m.coverage_reasons == (), mid
        assert m.vision.deepstack == 3 and len(m.vision.post) == 8 and m.vision.label == "Qwen3-VL ViT + deepstack", mid
        assert m.vision.depth == depth and m.vision.hidden == vh and m.vision.post[-1] == (vh * 4, out), mid
        assert m.vision.post[0] == (vh * 4, vh * 4) and m.vision.wfmt == () and m.vision.w_Bpp == 2.0, mid
        if moe:
            assert all(L.ffn.kind == "moe" for L in m.layers), mid
            f = m.layers[0].ffn
            assert (f.n_experts, f.top_k, f.d_expert, f.n_shared) == (128, 8, width, 0), mid
        else:
            assert all(L.ffn.kind == "dense" and L.ffn.d_ff == width for L in m.layers), mid
        fp8 = mid.endswith("-fp8")
        assert m.act_fmt == ("fp8" if fp8 else "bf16"), mid
        w = dict(m.formats)["attn"]
        assert w.fmt == ("fp8" if fp8 else "bf16") and round(w.bits, 6) == (8.001953 if fp8 else 16.0), mid
        if fp8 and moe:
            role = rel["roles"]["expert"]
            assert list(role) == ["fp8"] and round(m.fmt("expert").bits, 6) == 8.001953, mid
        if tie and not fp8:
            assert "lm_head" not in rel["roles"], mid
        if tie and fp8:
            assert rel["roles"]["lm_head"]["bf16"]["params"] == m.vocab * m.hidden, mid
            assert rel["params_llm"] == params + m.vocab * m.hidden, mid
        if fp8:
            assert rel["roles"]["embed"]["bf16"]["params"] == m.vocab * m.hidden, mid


def test_deepstack_merger_and_prefill_add():
    """8B, 1024x1024: four mergers (final + 3 deepstack), add billed on the encoder pass."""
    m = get_model("qwen3-vl-8b")
    g = image_grid(m.vision, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens) == (64, 64, 1024, 1024)
    ops = vision_ops(m.vision, g, 1)
    mergers = [o for o in ops if o.name.startswith("vis.merger.")]
    assert len(mergers) == 8
    assert all(o.w_fmt == "bf16" and o.w_bits == 16.0 and o.m == 1024 for o in mergers)
    assert (mergers[0].k, mergers[0].n) == (4608, 4608) and (mergers[1].k, mergers[1].n) == (4608, 4096)
    ds = next(o for o in ops if o.name == "vis.deepstack")
    assert ds.kind == "vector" and ds.vec == 3 * 1024 * 4096
    assert ds.act_bytes == 3 * 3 * 1024 * 4096 * 2
    dec = build_rank_ops(m, 0, Phase("decode", 1, 1, 16), Shard())
    assert all("deepstack" not in o.name for o in dec)
    r = evaluate(Scenario(model="qwen3-vl-2b", serving=Serving(phase="prefill", prompt=128, images=1)))
    assert r.vision["image_tokens"] == image_grid(get_model("qwen3-vl-2b").vision, 1024, 1024).llm_tokens
    clip = video_grid_of(m, Scenario(model="qwen3-vl-8b").replace("serving.videos", 1).replace(
        "serving.video_frames", 4).serving)
    assert clip.t == 2 and clip.frames == 4


def test_deepstack_refused_outside_qwen3_vl_and_on_bad_indexes():
    old = get_model("qwen3.5-397b-a17b").vision
    assert old.deepstack == 0 and len(old.post) == 2 and old.label == "Qwen ViT"
    cfg = load_release("Qwen/Qwen3-VL-8B-Instruct")["config"]
    ok = vision_spec("Qwen/Qwen3-VL-8B-Instruct", cfg)
    assert ok is not None and ok.deepstack == 3
    boolean = {**cfg, "vision_config": {**cfg["vision_config"], "deepstack_visual_indexes": [True, 16, 24]}}
    assert vision_spec("Qwen/Qwen3-VL-8B-Instruct", boolean) is None


def test_moe_fp8_expert_scale_suffix():
    """experts.down_proj_scale_inv attaches to experts.down_proj, not a separate fp32 expert."""
    spec = importlib.util.spec_from_file_location("summarize_release", "scripts/summarize_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    name = "model.language_model.layers.0.mlp.experts.down_proj_scale_inv"
    assert mod.SCALE_RE.search(name) and mod._base(name) == name[: -len("_scale_inv")]
    attn = "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv"
    assert mod._base(attn) == attn[: -len(".weight_scale_inv")]
