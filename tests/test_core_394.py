"""0.394: d1-omni bidirectional trunk, decision head, FastConformer, and SigLIP2.

The released class is not an lfm2 / lfm2_vl decoder. 0.393 CTC properties are re-checked
and left unchanged: the tied CTC weight is stored once, subsample_layers halves 50 to 12,
BatchNorm affine stays fp32 while GEMMs stay bf16, and StageTime stays off the JSON bill.
"""
from __future__ import annotations

import copy
import json

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.schedule import StageTime
from accel_dse.core.vision import audio_ops, d1_text_len, d1_trunk_ops, image_grid, vision_ops, vision_spec

MID = "d1-omni-600m"
HF = "LiquidAI/d1-omni-600M"
TRUNK = 380_732_161
VISION = 206_411_520
AUDIO = 112_176_640
HEAD = 26_248_193
BUF = 69_768
BYTES = 2_348_644_492
CTC = "granite-speech-5.0-470m-turboctc"
CTC_PARAMS = 472_928_256


def _ops(ops):
    by = {}
    for op in ops:
        assert op.name not in by, op.name
        by[op.name] = op
    return by


def _no_clock(obj):
    assert not isinstance(obj, StageTime)
    if isinstance(obj, dict):
        for value in obj.values():
            _no_clock(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _no_clock(value)


def test_d1_omni_catalog_matches_the_header():
    m = get_model(MID)
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == TRUNK == rel["params_llm"] == m.release_params == m.trunk_params
    assert m.vision_params == VISION == rel["params_vision"] == m.vision.params
    assert rel["params_mtp"] == 0 and rel["n_tensors"] == 1084 and rel["bytes_total"] == BYTES
    assert rel["bytes_llm"] == TRUNK * 4
    assert 4 * (TRUNK + VISION) + BUF == BYTES
    assert m.d1_buf == BUF == rel["roles"]["buffer"]["int64"]["bytes"]
    assert "vision_fmt" not in rel and "lm_head" not in rel["roles"] and "embed" in rel["roles"]
    assert rel["roles"]["embed"]["fp32"]["params"] == 65536 * 1024
    assert rel["roles"]["vision"]["fp32"]["params"] == VISION
    assert m.layers == () and m.n_layers == 0 and not m.kv_cache and m.coverage == "full"
    assert not m.coverage_reasons and m.domain == "vlm" and m.act_fmt == "fp32"
    assert m.fmt("attn").fmt == "fp32" and m.hidden == 1024 and m.vocab == 65536
    assert m.vision.family == "d1omni" and m.vision.w_Bpp == 4.0
    fm = {k: (f, b, a) for k, f, b, a in m.vision.wfmt}
    assert fm == {k: ("fp32", 32.0, "fp32") for k in ("patch", "attn", "mlp", "merger")}
    au = m.vision.audio
    assert au.kind == "d1fc" and au.params == AUDIO and au.w_fmt == "fp32" and au.w_bits == 32.0
    assert au.prompt_tokens(1.0) == 0
    assert int(m.vision.rz["trunk_params"]) + int(m.vision.rz["head_params"]) == TRUNK
    assert int(m.vision.rz["head_params"]) == HEAD and int(m.vision.rz["ff"]) == 4608
    assert m.vision.params - au.params == VISION - AUDIO
    listed = next(e for e in list_models() if e["id"] == MID)
    assert listed["domain"] == "vlm" and listed["family"] == "d1" and listed["coverage"] == "full"
    assert listed["dtype"] == "W fp32 · A fp32"
    assert "双向" in " ".join(m.notes) and "lm_head" in " ".join(m.notes)


def test_d1_audio_and_image_shapes_follow_the_class():
    m = get_model(MID)
    au, vis = m.vision.audio, m.vision
    assert [au.tokens(s) for s in (1.0, 0.5, 30.0, 0.0, 0.1, 30.5)] == [13, 7, 375, 0, 7, 375]
    grids = {
        (1024, 1024): (1280, ((32, 32, 5),)),
        (512, 512): (256, ((32, 32, 1),)),
        (256, 256): (64, ((16, 16, 1),)),
        (32, 32): (64, ((16, 16, 1),)),
        (100, 100): (64, ((16, 16, 1),)),
        (1280, 720): (764, ((32, 32, 2), (24, 42, 1))),
        (640, 480): (234, ((26, 36, 1),)),
        (2048, 2048): (2560, ((32, 32, 10),)),
        (768, 768): (1280, ((32, 32, 5),)),
    }
    for (w, h), (tokens, views) in grids.items():
        g = image_grid(vis, w, h)
        assert (g.llm_tokens, g.views) == (tokens, views), (w, h, g.llm_tokens, g.views)
    aud = _ops(audio_ops(au, 1.0, 1, "bf16"))
    assert (aud["aud.sub.0"].m, aud["aud.sub.0"].k, aud["aud.sub.0"].n) == (3200, 9, 256)
    assert aud["aud.sub.0"].w_fmt == "fp32" and aud["aud.sub.0"].a_fmt == "fp32"
    assert aud["aud.sub.dw0"].kind == "vector" and aud["aud.sub.dw0"].vec == 25 * 32 * 256 * 9
    assert aud["aud.0.pos"].m == 25 and aud["aud.0.bd"].n == 25
    assert (aud["aud.0.ac"].m, aud["aud.0.ac"].n, aud["aud.0.ac"].count) == (13, 13, 8)
    assert aud["aud.0.ac"].causal == 1.0 and aud["aud.0.dw"].kind == "vector"
    assert (aud["aud.adapter.1"].k, aud["aud.adapter.1"].n) == (512, 1024)
    assert (aud["aud.residual.down"].n, aud["aud.residual.up"].k) == (512, 512)
    two = _ops(audio_ops(au, 1.0, 2))
    assert two["aud.0.pos"].m == 25 and two["aud.sub.0"].m == 6400 and two["aud.0.ac"].count == 16
    assert all(op.kv_write == 0 and op.kind != "state" for op in aud.values())
    g = image_grid(vis, 1024, 1024)
    vops = _ops(vision_ops(vis, g, 1, "bf16"))
    assert (vops["vis.patch_embed"].m, vops["vis.patch_embed"].w_fmt, vops["vis.patch_embed"].a_fmt) == (5120, "fp32", "fp32")
    assert (vops["vis.0.qk"].m, vops["vis.0.qk"].count, vops["vis.0.qk"].causal) == (1024, 60, 1.0)
    assert vops["vis.merger.0"].m == 1280 and vops["vis.merger.1"].n == 1024
    assert d1_text_len(vis, 4096, g.llm_tokens, 1, 0.0) == 896
    assert d1_text_len(vis, 4096, au.tokens(1.0), 0, 1.0) == 4096
    assert d1_text_len(vis, 128, 0, 0, 0.0) == 128


def test_d1_forward_is_one_bidirectional_pass():
    m = get_model(MID)
    vis = m.vision
    g = image_grid(vis, 1024, 1024)
    text = d1_text_len(vis, 4096, g.llm_tokens, 1, 0.0)
    trunk = _ops(d1_trunk_ops(vis, 2, text, g.llm_tokens, 3))
    length = g.llm_tokens + text
    assert text == 896 and length == 2176
    assert (trunk["trunk.0.in"].m, trunk["trunk.0.in"].n) == (2 * length, 3072)
    assert trunk["trunk.2.q"].m == 2 * length and trunk["trunk.2.k"].n == 512
    qk = trunk["trunk.2.qk"]
    assert (qk.m, qk.n, qk.count, qk.causal) == (length, length, 32, 1.0)
    assert trunk["head.0.in"].m == 2 * text and trunk["head.scorer.3"].m == 6
    assert trunk["head.scorer.3"].n == 1 and trunk["head.type_emb"].kind == "lookup"
    assert trunk["trunk.embed"].kv_read == 2 * text * 1024 * 4
    assert all(op.kv_write == 0 and op.kind != "state" for op in trunk.values())
    assert all(op.w_fmt == "fp32" and op.a_fmt == "fp32" for op in trunk.values() if op.kind == "gemm")
    bare = _ops(d1_trunk_ops(vis, 1, 128, 0, 0))
    assert "head.scorer.1" not in bare and bare["head.0.in"].m == 128 and bare["trunk.0.in"].m == 128
    img = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=4096, images=1, labels=3)))
    assert img.asr is None and img.tpot == 0.0 and img.tokens_per_step == 1 and img.latency == img.step
    assert img.vision["image_tokens"] == 1280 and img.vision["audio"]["tokens"] == 0 and img.step > 0 and img.fits
    assert abs(img.vision["weights_GB"] * 1e9 - BYTES) < 1
    assert isinstance(img.stages[0].time, StageTime)
    blob = img.summary()
    json.dumps(blob)
    json.dumps(img.vision)
    _no_clock(blob)
    _no_clock(img.vision)
    aud = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=4096, audio_s=1.0)))
    assert aud.vision["audio"]["tokens"] == 13 and aud.vision["image_tokens"] == 0
    text_only = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=128)))
    assert text_only.vision["image_tokens"] == 0 and text_only.tokens_per_step == 1
    pair = evaluate(Scenario(model=MID, layout=Layout(dp=2),
                             serving=Serving(phase="prefill", batch=2, prompt=128, questions=2, labels=3)))
    one = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=128, questions=2, labels=3)))
    assert pair.step == one.step and abs(pair.vision["tflop"] / one.vision["tflop"] - 2) < 1e-9
    assert pair.throughput == pair.scenario.serving.batch * 2 / pair.step
    dec = evaluate(Scenario(model=MID, serving=Serving(phase="decode", batch=1, ctx=128)))
    assert any("没有自回归" in w for w in dec.warnings) and dec.tpot == 0.0 and dec.step > 0
    think = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=128, labels=3, think=True)))
    got = think.decision()
    assert got["think_ms"] == 0.0 and "思考不另计" in got["think_note"]
    try:
        evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, images=1, audio_s=1.0)))
        raise AssertionError("images and audio both billed")
    except ValueError as exc:
        assert "图像或音频" in str(exc)
    try:
        evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, images=64)))
        raise AssertionError("64 images of 1024 fit")
    except ValueError as exc:
        assert "文本不足" in str(exc)
    try:
        evaluate(Scenario(model=MID, layout=Layout(tp=2), serving=Serving(phase="prefill", batch=1, prompt=128)))
        raise AssertionError("tp 2 accepted")
    except ValueError as exc:
        assert "tp / sp" in str(exc)
    try:
        evaluate(Scenario(model=MID, layout=Layout(pp=2), serving=Serving(phase="prefill", batch=1, prompt=128)))
        raise AssertionError("pp 2 accepted")
    except ValueError as exc:
        assert "pp exceeds layer count" in str(exc)
    vid = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=128, videos=1)))
    assert any("没有视频" in w for w in vid.warnings)
    try:
        d1_trunk_ops(vis, 1, 32, 0, 64)
        raise AssertionError("labels longer than text accepted")
    except ValueError as exc:
        assert "选项标记" in str(exc)


def test_d1_refuses_a_different_tower():
    rel = load_release(HF)
    kept = vision_spec(HF, rel["config"])
    assert kept is not None and kept.params == VISION and kept.audio.params == AUDIO

    def gone(mut):
        cfg = copy.deepcopy(rel["config"])
        mut(cfg)
        assert vision_spec(HF, cfg) is None

    gone(lambda cfg: cfg["text_config"].__setitem__("conv_L_cache", 4))
    gone(lambda cfg: cfg["audio_config"].__setitem__("feat_in", 80))
    gone(lambda cfg: cfg["vision_config"].__setitem__("vision_use_head", True))
    gone(lambda cfg: cfg.__setitem__("dtype", "bfloat16"))
    gone(lambda cfg: cfg["audio_config"].__setitem__("ff_expansion_factor", 8))
    gone(lambda cfg: cfg["audio_config"].__setitem__("conv_kernel_size", 8))


def test_ctc_tied_weight_subsample_and_clock_stay_serializable():
    m = get_model(CTC)
    au = m.vision.audio
    assert au._gsctc_frames(1.0) == (50, 25) + (12,) * 15
    ops = _ops(audio_ops(au, 1.0, 1))
    tied = 1024 * 16384
    assert ops["aud.mid.out"].w_params == ops["aud.ctc"].w_params == tied
    assert ops["aud.mid.out"].w_params + ops["aud.ctc"].w_params == 2 * tied
    assert au.params == m.params() == CTC_PARAMS
    assert all(op.w_fmt == "bf16" and op.a_fmt == "bf16" for op in ops.values() if op.kind == "gemm")
    rel = load_release(m.hf_id)
    affine = rel["roles"]["vision"]["fp32"]["params"]
    assert affine == 16 * 2 * 2048 and rel["roles"]["vision"]["fp32"]["bytes"] == 4 * affine
    pre = evaluate(Scenario(model=CTC, serving=Serving(phase="prefill", batch=1, prompt=4, audio_s=1.0)))
    assert isinstance(pre.stages[0].time, StageTime)
    blob = pre.summary()
    json.dumps(blob)
    json.dumps(pre.vision)
    json.dumps(pre.asr)
    _no_clock(blob)
    _no_clock(pre.vision)
    _no_clock(pre.asr)
    assert pre.asr["enc_positions"] == 12 and "time" not in pre.vision and "time" not in pre.asr
