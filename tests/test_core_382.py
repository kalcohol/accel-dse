"""0.382: 0.381 ERNIE-4.5-0.3B review holds; Gemma-4 video grid closes.

ERNIE-4.5-0.3B-PT and Base-PT headers (2026-10-24) still match the dense class:
18-layer GQA 16/2, head_dim 128, tied embeddings, no MTP. Gemma-4 video uses the
published processor (max_soft_tokens 70, num_frames 32), not the image pad of 280.
"""
from __future__ import annotations

from accel_dse import api as A
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_ops

_G4 = (
    "gemma-4-31b-it", "gemma-4-31b",
    "gemma-4-26b-a4b-it", "gemma-4-26b-a4b",
    "gemma-4-12b-it", "gemma-4-12b",
    "gemma-4-e4b-it", "gemma-4-e4b",
    "gemma-4-e2b-it", "gemma-4-e2b",
)
_PRE = {"gemma4_soft": 70, "num_frames": 32, "patch": 16, "pool": 3}


def test_review_381_ernie_0_3b_unchanged():
    h, heads, kv, hd, inter, vocab, layers = 1024, 16, 2, 128, 3072, 103424, 18
    q, kk = heads * hd, kv * hd
    attn = h * q + 2 * h * kk + q * h
    language = layers * (attn + 3 * h * inter + 2 * h) + vocab * h + h
    assert language == 360_748_032
    assert h // heads == 64 and q == 2048 and kk == 256
    for mid, repo in (
        ("ernie-4.5-0.3b-pt", "baidu/ERNIE-4.5-0.3B-PT"),
        ("ernie-4.5-0.3b-base-pt", "baidu/ERNIE-4.5-0.3B-Base-PT"),
    ):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params
        assert rel["repo"] == repo and rel["n_tensors"] == 164
        assert rel["params_llm"] == language and rel["params_mtp"] == 0 and rel["bytes_mtp"] == 0
        assert rel["bytes_llm"] == rel["bytes_total"] == 721_496_064
        assert "num_nextn_predict_layers" not in rel["config"]
        assert rel["config"]["head_dim"] == 128 and rel["config"]["tie_word_embeddings"] is True
        assert m.mtp_layers == () and m.vision is None and m.tie_embeddings is True
        L0 = m.layers[0]
        assert (L0.core.n_q, L0.core.n_kv, L0.core.qk_dim, L0.core.v_dim) == (heads, kv, hd, hd)
        assert [(lin.name, lin.k, lin.n) for lin in L0.attn_linears] == [
            ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]


def test_gemma4_video_grid_and_pad():
    for mid in _G4:
        m = get_model(mid)
        assert video_pre(m) == _PRE
        assert m.param_check()["rel_err"] == 0.0
        assert m.vision.patch == 16 and m.vision.merge == 3
        assert m.vision.family == ("gemma4u" if "12b" in mid else "gemma4")
    vit = get_model("gemma-4-31b-it")
    pre = video_pre(vit)
    wide = video_grid(vit.vision, pre, 32, 1280, 720)
    assert wide.frames == 32 and wide.t == 32 and wide.tail is None
    assert (wide.slice.px, wide.slice.ph, wide.slice.pw, wide.slice.llm_tokens, wide.slice.pad_soft) == (
        (528, 288), 18, 33, 66, 70)
    assert wide.llm_tokens == 32 * 66
    assert video_grid(vit.vision, pre, 100, 1280, 720) == wide
    assert video_grid(vit.vision, pre, 32, 1280, 720, cap=True) == wide
    square = video_grid(vit.vision, pre, 32, 1024, 1024)
    assert (square.slice.px, square.slice.llm_tokens) == ((384, 384), 64)
    assert square.llm_tokens == 32 * 64
    full = video_grid(vit.vision, pre, 32, 4000, 48)
    assert full.slice.llm_tokens == 70 and full.slice.ph * full.slice.pw == 70 * 9
    try:
        video_grid(vit.vision, pre, 8, 1280, 720)
    except ValueError as exc:
        assert "num_frames" in str(exc)
    else:
        raise AssertionError("a clip shorter than num_frames was accepted")
    ops = vision_ops(vit.vision, wide.slice, wide.frames)
    q = next(o for o in ops if o.name == "vis.0.q")
    proj = next(o for o in ops if o.name == "vis.proj")
    assert q.m == 32 * 70 * 9 and q.k == 1152
    assert proj.m == 32 * 66 and (proj.k, proj.n) == (1152, 5376)
    qk = next(o for o in ops if o.name == "vis.0.qk")
    assert (qk.m, qk.n, qk.count) == (70 * 9, 70 * 9, 32 * vit.vision.heads)
    still = image_grid(vit.vision, 1024, 1024)
    assert still.pad_soft == 0 and still.llm_tokens == 256
    still_q = next(o for o in vision_ops(vit.vision, still, 1) if o.name == "vis.0.q")
    assert still_q.m == 280 * 9
    uni = get_model("gemma-4-12b-it")
    u = video_grid(uni.vision, video_pre(uni), 32, 1280, 720)
    assert u.llm_tokens == 32 * 66 and u.slice.pad_soft == 70
    uops = vision_ops(uni.vision, u.slice, u.frames)
    assert [o.name for o in uops] == ["vis.patch_dense", "vis.proj"]
    assert uops[0].m == uops[1].m == 32 * 70
    assert (uops[0].k, uops[0].n) == (6912, 3840)
    edge = video_grid(uni.vision, video_pre(uni), 32, 4000, 48)
    eops = vision_ops(uni.vision, edge.slice, 1)
    assert eops[0].m == 70 and edge.llm_tokens == 32 * 70
    e2 = get_model("gemma-4-e2b-it")
    e2q = next(o for o in vision_ops(e2.vision, wide.slice, 1) if o.name == "vis.0.q")
    assert e2q.m == 70 * 9 and e2q.k == e2.vision.hidden
    r = evaluate(Scenario(model="gemma-4-31b-it", serving=Serving(phase="prefill", prompt=16, videos=1)))
    assert r.vision["video"]["frames"] == 32
    assert r.vision["video"]["resized_px"] == [528, 288]
    assert r.vision["video"]["tokens_per_video"] == 32 * 66
    assert r.vision["image_tokens"] == 32 * 66
    assert r.vision["tflop"] > 0
    evaluate(Scenario(model="gemma-4-12b-it", serving=Serving(phase="decode", batch=1, ctx=32, videos=1)))
    s = A.api_eval({"scenario": {"model": "gemma-4-26b-a4b-it",
                                 "serving": {"phase": "prefill", "prompt": 16, "videos": 1}},
                    "sensitivity": False})["summary"]
    assert s["step_ms"] > 0 and s["tok_s"] > 0
