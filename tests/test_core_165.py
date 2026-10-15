"""0.165: VLM video input (Qwen3VLVideoProcessor models) — smart_resize over the clip, per-slice encoder passes."""
import math

from accel_dse import api as A
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import VIDEO_MAX_FRAMES, expand_images, image_grid, video_grid, video_pre, vision_flops

M = "qwen3.8-27b"


def _ref(frames, h, w, lo=4096, hi=25165824, f=32, tp=2):
    # transformers qwen3_vl smart_resize, written out independently (budget over t_bar·h·w)
    hb, wb, tb = round(h / f) * f, round(w / f) * f, round(frames / tp) * tp
    if tb * hb * wb > hi:
        b = math.sqrt(frames * h * w / hi)
        hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
    elif tb * hb * wb < lo:
        b = math.sqrt(lo / (frames * h * w))
        hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
    return hb, wb


def test_sourced_preprocessor_and_hand_values():
    m = get_model(M)
    assert video_pre(m) == {"min_px": 4096, "max_px": 25165824} and VIDEO_MAX_FRAMES == 768
    assert video_pre(get_model("glm-5.3-flash")) is None and video_pre(get_model("qwen3-8b")) is None
    g = video_grid(m.vision, video_pre(m), 32, 1280, 720)
    # 32·704·1280 > 25165824 → β = √(32·720·1280/25165824) → 640×1152; 40×72 patches → 720 merged × 16 slices
    assert (g.t, g.slice.px, g.slice.llm_tokens, g.llm_tokens) == (16, (1152, 640), 720, 11520)
    g8 = video_grid(m.vision, video_pre(m), 8, 1280, 720)
    assert (g8.t, g8.slice.px, g8.llm_tokens) == (4, (1280, 704), 3520)          # under budget: native-ish
    c8 = video_grid(m.vision, video_pre(m), 8, 1280, 720, cap=True)              # cap 768·32² px / frame
    assert c8.slice.px == (1152, 640) and c8.llm_tokens == 2880
    assert video_grid(m.vision, video_pre(m), 3, 1280, 720).t == 2               # frames padded to temporal_patch


def test_budget_spreads_over_frames_and_matches_reference():
    m = get_model(M)
    pre = video_pre(m)
    last = None
    for F in (4, 16, 64, 256, 768):
        for w, h in ((1280, 720), (1920, 1080), (640, 480), (3840, 2160)):
            g = video_grid(m.vision, pre, F, w, h)
            hb, wb = _ref(F, h, w)
            assert g.slice.px == (wb, hb)
            assert (math.ceil(F / 2) // 1) * hb * wb <= pre["max_px"] + 2 * hb * wb   # within budget (± tb rounding)
        tok = video_grid(m.vision, pre, F, 1920, 1080).llm_tokens
        assert last is None or tok >= 0.8 * last                                       # budget-bounded, not ∝ F
        last = tok
    assert video_grid(m.vision, pre, 768, 1920, 1080).llm_tokens <= pre["max_px"] // (2 * 32 * 32) + 384


def test_expansion_and_encoder_cost_per_slice():
    m = get_model(M)
    s = Scenario(model=M, serving=Serving(phase="prefill", batch=2, prompt=1000, videos=1))
    e = expand_images(s, m).serving
    assert e.image_tokens == 11520 and e.prompt == 1000 + 11520
    r = evaluate(s)
    v = r.vision["video"]
    assert v["slices"] == 16 and v["tokens_per_video"] == 11520 and r.vision["image_tokens"] == 11520
    # FLOPs = 16 slice passes of one 40×72 grid per video (attention within a slice, not over the clip)
    g = video_grid(m.vision, video_pre(m), 32, 1280, 720).slice
    assert abs(r.vision["tflop"] * 1e12 - 2 * 16 * vision_flops(m.vision, g)) / r.vision["tflop"] / 1e12 < 1e-9
    ri = evaluate(Scenario(model=M, serving=Serving(phase="prefill", batch=2, prompt=1000, images=1)))
    rb = evaluate(Scenario(model=M, serving=Serving(phase="prefill", batch=2, prompt=1000, images=1, videos=1)))
    assert abs(rb.vision["s"] - ri.vision["s"] - r.vision["s"]) < 1e-12
    assert rb.vision["image_tokens"] == 11520 + image_grid(m.vision, 1024, 1024).llm_tokens


def test_defaults_unchanged_and_validation():
    assert Scenario().hash() == Scenario(serving=Serving(videos=0, video_frames=32, video_w=1280, video_h=720)).hash()
    ri = evaluate(Scenario(model=M, serving=Serving(phase="prefill", images=1)))
    assert "video" not in ri.vision and "tflop_per_image" in ri.vision
    for kw in (dict(videos=17), dict(video_frames=1), dict(video_frames=769), dict(video_cap=1), dict(video_w=1, video_h=300)):
        try:
            Serving(**kw)
        except ValueError:
            continue
        raise AssertionError(kw)
    try:
        evaluate(Scenario(model="glm-5.3-flash", serving=Serving(phase="prefill", videos=1)))
    except ValueError as e:
        assert "视频处理器" in str(e)
    else:
        raise AssertionError("unsupported model must refuse videos")
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(videos=1)))
    assert any("已忽略" in w for w in r.warnings)


def test_decode_kv_and_cli_api():
    d0 = evaluate(Scenario(model=M, serving=Serving(batch=4, ctx=4096)))
    d1 = evaluate(Scenario(model=M, serving=Serving(batch=4, ctx=4096, videos=1)))
    assert d1.tpot > d0.tpot                                                # video tokens extend the decode context
    assert "serving.videos" in A.SWEEP_PATHS and "serving.video_frames" in A.SWEEP_PATHS
    import json, subprocess, sys
    out = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--model", M, "--phase", "prefill", "--videos", "1",
                          "--video-frames", "64", "--video-size", "1920x1080", "--video-cap", "--json"],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = d["summary"]["vision"]["video"]
    assert (v["frames"], v["px"], v["cap"]) == (64, [1920, 1080], True)
    assert d["scenario"]["serving"]["videos"] == 1 and d["scenario"]["serving"]["video_cap"] is True
