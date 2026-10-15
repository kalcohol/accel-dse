"""0.166: Kimi K2.5 / K2.7-Code video chunks (4-frame spatial-temporal attention, temporal pooling); catalog flag."""
import math
from dataclasses import replace

from accel_dse.core.catalog import get_model, labels as model_info
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import video_grid, video_pre, vision_flops, vision_ops


def _navit(w, h, p=14, mk=2, lim=4096, side=512):
    # media_utils.navit_resize_image, written out independently
    s = min(1.0, math.sqrt(lim / (max(1.0, w // p) * max(1.0, h // p))), side * p / w, side * p / h)
    nw, nh = min(max(1, int(w * s)), side * p), min(max(1, int(h * s)), side * p)
    f = mk * p
    return (nh + (f - nh % f) % f) // f, (nw + (f - nw % f) % f) // f


def test_sourced_settings_and_hand_values():
    assert video_pre(get_model("kimi-k2.5")) == {"chunk": 4, "frame_lim": 4096, "total": None}
    assert video_pre(get_model("kimi-k2.7-code"))["total"] == 655360
    assert video_pre(get_model("kimi-k3")) is None                      # own processor / no video_attn_type: not modelled
    m = get_model("kimi-k2.5")
    g = video_grid(m.vision, video_pre(m), 32, 1280, 720)
    # 91·51 = 4641 patches > 4096 → s = √(4096/4641) → 1202×676 → pad to 28 → 1204×700 → 43×25 = 1075 tokens / chunk
    assert (g.t, g.slice.px, g.slice.tframes, g.slice.llm_tokens, g.llm_tokens, g.tail) == (8, (1204, 700), 4, 1075, 8600, None)
    for w, h in ((1920, 1080), (640, 480), (3840, 2160), (720, 1280)):
        th, tw = _navit(w, h)
        assert video_grid(m.vision, video_pre(m), 8, w, h).slice.llm_tokens == th * tw


def test_tail_chunk_and_token_count_independent_of_chunk_frames():
    m = get_model("kimi-k2.5")
    g = video_grid(m.vision, video_pre(m), 30, 1280, 720)
    assert g.t == 8 and g.tail is not None and g.tail.tframes == 2 and g.llm_tokens == 8 * 1075   # tpool: tokens ∤ t
    one = video_grid(m.vision, video_pre(m), 3, 1280, 720)
    assert one.t == 1 and one.tail is None and one.slice.tframes == 3


def test_spatial_temporal_attention_cost():
    m = get_model("kimi-k2.5")
    g = video_grid(m.vision, video_pre(m), 4, 1280, 720).slice
    sep = replace(g, tframes=1)
    ch, fr = vision_flops(m.vision, g), 4 * vision_flops(m.vision, sep)
    assert ch > 1.5 * fr                                              # attention over 4·4300 patches, not 4 × 4300
    ops = vision_ops(m.vision, g, 1)
    assert any(o.name == "vis.tpool" for o in ops)
    mg = [o for o in ops if o.name.startswith("vis.merger")]
    assert mg and all(o.m == g.merged for o in mg)                     # merger after the temporal mean


def test_evaluate_and_catalog_flag():
    r30 = evaluate(Scenario(model="kimi-k2.5", serving=Serving(phase="prefill", batch=1, prompt=1000, videos=1, video_frames=30)))
    r32 = evaluate(Scenario(model="kimi-k2.5", serving=Serving(phase="prefill", batch=1, prompt=1000, videos=1, video_frames=32)))
    assert r30.vision["image_tokens"] == r32.vision["image_tokens"] == 8600
    assert r30.vision["video"]["s"] < r32.vision["video"]["s"] and r32.vision["video"]["frames_per_slice"] == 4
    assert model_info(get_model("kimi-k2.5"))["vision"]["video"] is True
    assert model_info(get_model("glm-5.3-flash"))["vision"]["video"] is False
    q = evaluate(Scenario(model="qwen3.8-27b", serving=Serving(phase="prefill", videos=1)))
    assert q.vision["video"]["tokens_per_video"] == 11520 and q.vision["video"]["frames_per_slice"] == 2
