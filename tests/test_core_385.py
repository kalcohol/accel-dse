"""0.385: 0.384 ERNIE-4.5-VL dual-pool review holds; video pairs frames.

The still-image grid is unchanged (1024 -> 1371, one duplicated frame). A video
uses Ernie4_5_VLProcessor: per-frame smart_resize at the video pixel limits, an
odd count copies the last frame, and the temporal GEMMs see half as many rows.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops


def _pairs(op):
    return op.count * op.m * op.share


def test_review_384_ernie_vl_holds():
    rows = {
        "ernie-4.5-vl-28b-a3b-pt": dict(
            k=6, n_shared=1, d_shared=3072, moe_at=1),
        "ernie-4.5-vl-424b-a47b-pt": dict(
            k=8, n_shared=0, d_shared=0, moe_at=3),
    }
    for mid, s in rows.items():
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0
        assert m.mtp_layers == () and m.mtp_params() == 0
        assert "num_nextn_predict_layers" not in rel["config"]
        layer = m.layers[s["moe_at"]]
        assert layer.ffn.n_shared == s["n_shared"] and layer.ffn.d_shared == s["d_shared"]
        assert m.vision.family == "ernievl"
        vis = rel["roles"]["vision"]
        assert set(vis) == {"fp16", "bf16"}
        assert vis["fp16"]["params"] == 630_433_280
        assert vis["bf16"]["params"] == m.vision_params - 630_433_280
        g = image_grid(m.vision, 1024, 1024)
        assert (g.merged, g.llm_tokens, g.tframes) == (1369, 1371, 1)
        still = vision_ops(m.vision, g, 1)
        assert [o.m for o in still if o.name.startswith("vis.merger")] == [1369] * 5

        li = s["moe_at"]
        dec = build_rank_ops(m, li, Phase("decode", 1, 1, 16), Shard())
        assert not any("router_b" in o.name or "expert_b" in o.name for o in dec)
        q, mm = 32, 8
        text = build_rank_ops(m, li, Phase("prefill", 1, q, 0), Shard())
        mix = build_rank_ops(m, li, Phase("prefill", 1, q, 0, mm=mm), Shard())
        assert not any(o.name == "moe.router_b" for o in text)
        assert next(o for o in mix if o.name == "moe.router").m == q
        assert next(o for o in mix if o.name == "moe.router_b").m == q
        assert _pairs(next(o for o in mix if o.name == "expert.gate_up")) == (q - mm) * s["k"]
        assert _pairs(next(o for o in mix if o.name == "expert_b.gate_up")) == mm * s["k"]


def test_ernie_vl_video_pairs_frames():
    pre_expect = {"ernie_temporal": 2, "min_px": 299 * 28 * 28, "max_px": 1196 * 28 * 28}
    specs = {
        "ernie-4.5-vl-28b-a3b-pt": 2560,
        "ernie-4.5-vl-424b-a47b-pt": 8192,
    }
    for mid, text_n in specs.items():
        m = get_model(mid)
        v = m.vision
        pre = video_pre(m)
        assert pre == pre_expect
        hd = video_grid(v, pre, 32, 1280, 720)
        assert hd.frames == 32 and hd.t == 1 and hd.tail is None
        assert (hd.slice.ph, hd.slice.pw, hd.slice.merged, hd.slice.tframes) == (52, 92, 1196, 32)
        assert hd.slice.px == (1288, 728)
        assert hd.llm_tokens == hd.slice.llm_tokens == 16 * 1196 + 2 == 19138
        odd = video_grid(v, pre, 17, 1280, 720)
        assert odd.frames == 18 and odd.slice.tframes == 18
        assert odd.llm_tokens == 9 * 1196 + 2 == 10766
        assert video_grid(v, pre, 17, 1280, 720, cap=True) == odd
        two = video_grid(v, pre, 2, 1024, 1024)
        assert (two.slice.ph, two.slice.pw, two.slice.merged, two.frames) == (68, 68, 1156, 2)
        assert two.llm_tokens == 1156 + 2
        assert image_grid(v, 1024, 1024).llm_tokens == 1371
        tiny = video_grid(v, pre, 2, 8, 8)
        assert (tiny.slice.ph, tiny.slice.pw, tiny.slice.merged, tiny.slice.px) == (36, 36, 324, (504, 504))
        wide = video_grid(v, pre, 2, 48, 4000)
        assert (wide.slice.px, wide.slice.merged) == ((56, 4424), 316)
        one = video_grid(v, pre, 1, 1280, 720)
        assert one.frames == 2 and one.llm_tokens == 1196 + 2
        try:
            video_grid(v, pre, 0, 1280, 720)
        except ValueError as exc:
            assert "video_frames" in str(exc)
        else:
            raise AssertionError("a zero-frame clip was accepted")

        ops = vision_ops(v, two.slice, 1)
        mergers = [o for o in ops if o.name.startswith("vis.merger")]
        assert [(o.m, o.k, o.n, o.w_fmt) for o in mergers] == [
            (2 * 1156, 5120, 5120, "bf16"),
            (2 * 1156, 5120, 5120, "bf16"),
            (1156, 10240, 5120, "bf16"),
            (1156, 5120, 5120, "bf16"),
            (1156, 5120, text_n, "bf16"),
        ]
        assert not any("index" in o.name or o.name == "vis.tpool" for o in ops)
        qk = [o for o in ops if o.name.endswith(".qk")]
        assert len(qk) == 32
        assert all(o.count == 2 * 16 and o.m == 68 * 68 and o.n == 68 * 68 for o in qk)
        patch = next(o for o in ops if o.name == "vis.patch_embed")
        assert (patch.m, patch.k, patch.n) == (2 * 68 * 68, 588, 1280)
        mlp = next(o for o in ops if o.name == "vis.0.mlp_in")
        assert mlp.m == 2 * 68 * 68 and mlp.n == 5120


def test_ernie_vl_video_step_uses_vision_pool():
    m28 = "ernie-4.5-vl-28b-a3b-pt"
    m = get_model(m28)
    pre = video_pre(m)
    tok = video_grid(m.vision, pre, 2, 1024, 1024).llm_tokens
    assert tok == 1158
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
    vid = clip.vision["video"]
    assert vid["frames"] == 32 and vid["slices"] == 1
    assert vid["frames_per_slice"] == 32
    assert vid["patches"] == [52, 92]
    assert vid["resized_px"] == [1288, 728]
    assert vid["tokens_per_video"] == vid["tokens_per_slice"] == 19138
    assert clip.vision.get("s", 0.0) > 0 and clip.step > 0
    same = evaluate(Scenario(model=m28, serving=Serving(phase="prefill", batch=1, prompt=32 + 19138)))
    assert clip.stages[0].flops_u != same.stages[0].flops_u

    small = evaluate(Scenario(model="ernie-4.5-vl-424b-a47b-pt", serving=Serving(
        phase="prefill", batch=1, prompt=32, videos=1, video_frames=2, video_w=1024, video_h=1024)))
    assert small.vision["image_tokens"] == 1158
    assert small.vision["video"]["frames"] == 2
    assert small.vision["video"]["tokens_per_video"] == 1158
    assert small.vision.get("s", 0.0) > 0 and small.step > 0
