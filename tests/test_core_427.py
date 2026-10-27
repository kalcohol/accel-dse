"""0.427: Kimi MoonViT LayerNorm / RMSNorm and projector erf-GELU.

K2.5 and K2.7 bill nn.LayerNorm before attention, before the MLP, after the
blocks, and on the pre-view projector rows. K3 bills nn.RMSNorm at hidden
before attention, before the MLP, and after the blocks, plus a post RMSNorm
at the text width. Both bill nn.GELU() (erf) after the first merger linear.
The tower MLP stays PytorchGELUTanh. The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, video_grid, video_pre, vision_flops, vision_ops, vision_spec

K25 = "kimi-k2.5"
K27 = "kimi-k2.7-code"
K3 = "kimi-k3"
HF25 = "moonshotai/Kimi-K2.5"
HF3 = "moonshotai/Kimi-K3"
H25 = 1152
INTER25 = 4304
H3 = 1024
Q3 = 1536
INTER3 = 4096
DEPTH = 27
HEADS25 = 16
HEADS3 = 12
TEXT = 7168
ROWS = 74 * 74
MERGED = 1369
HM25 = 4 * H25
HM3 = 4 * H3


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


def _rms(op, rows: int, width: int, layer: int) -> None:
    _ln(op, rows, width, layer)


def _gelu(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == "bf16" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_k25_and_k27_bill_layernorm_and_projector_gelu():
    a = get_model(K25)
    b = get_model(K27)
    assert a.vision == b.vision
    assert a.param_check()["rel_err"] == 0.0 and b.param_check()["rel_err"] == 0.0
    assert a.param_check()["ours"] == a.param_check()["release"] == 1026408232448
    vs = a.vision
    assert vs is not None and vs.family == "kimi" and vs.ln_mode == "k25" and vs.vec_fmt == "bf16"
    assert vs.gated is False and vs.depth == DEPTH and vs.heads == HEADS25
    assert vs.hidden == H25 and vs.inter == INTER25 and vs.qkv == H25 and vs.merge == 2
    assert vs.post == ((HM25, HM25), (HM25, TEXT)) and vs.params == 471143920
    rel = load_release(a.hf_id)
    vc = rel["config"]["vision_config"]
    assert vc["mm_projector_type"] == "patchmerger" and vc["projector_hidden_act"] == "gelu"
    assert "norm_type" not in vc and "activation_func" not in vc and "qkv_hidden_size" not in vc
    assert vc["mm_hidden_size"] == H25 and set(rel["roles"]["vision"]) == {"bf16"}
    note = " ".join(a.notes)
    assert "PytorchGELUTanh" in note and "nn.GELU()" in note and "Kimi MoonViT" in note
    assert "PytorchGELUTanh" in " ".join(b.notes)
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.ph * g.pw, g.merged, g.llm_tokens) == (74, 74, ROWS, MERGED, MERGED)
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _ln(by["vis.0.attn_ln"], ROWS, H25, 0)
    _ln(by["vis.0.mlp_ln"], ROWS, H25, 0)
    _ln(by[f"vis.{DEPTH - 1}.attn_ln"], ROWS, H25, DEPTH - 1)
    _ln(by[f"vis.{DEPTH - 1}.mlp_ln"], ROWS, H25, DEPTH - 1)
    _ln(by["vis.out_ln"], ROWS, H25, DEPTH)
    _ln(by["vis.proj_ln"], ROWS, H25, DEPTH)
    # merged * 4 * hidden has the same element count. The rsqrt count is the patch rows.
    assert by["vis.proj_ln"].sfu == float(ROWS) and by["vis.proj_ln"].sfu != float(MERGED)
    _gelu(by["vis.mgelu"], MERGED, HM25, DEPTH)
    assert by["vis.mgelu"].vec != MERGED * TEXT * 5
    assert by["vis.0.qkv"].n == 3 * H25 and by["vis.0.qkv"].vec == ROWS * (3 * H25) * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.qkv"].a_fmt == "bf16"
    assert by["vis.0.mlp_in"].n == INTER25 and by["vis.0.mlp_in"].vec == ROWS * INTER25 * 3
    assert by["vis.0.mlp_out"].k == INTER25 and by["vis.0.mlp_out"].n == H25
    assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].k == HM25 and by["vis.merger.0"].n == HM25
    assert by["vis.merger.0"].vec == MERGED * HM25 * 3
    assert by["vis.merger.1"].k == HM25 and by["vis.merger.1"].n == TEXT
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS25
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS25 * ROWS * ROWS * 5
    names = [op.name for op in ops]
    assert names.index("vis.0.attn_ln") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.mlp_ln") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index("vis.0.qk") < names.index("vis.0.pv") < names.index("vis.0.softmax")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.out_ln")
    assert names.index("vis.out_ln") < names.index("vis.proj_ln") < names.index("vis.merger.0")
    assert names.index("vis.merger.0") < names.index("vis.mgelu") < names.index("vis.merger.1")
    lns = [op for op in ops if op.name.endswith(".attn_ln") or op.name.endswith(".mlp_ln")
           or op.name in ("vis.out_ln", "vis.proj_ln")]
    gels = [op for op in ops if "gelu" in op.name]
    assert len(lns) == 2 * DEPTH + 2 and len(gels) == 1 and gels[0].name == "vis.mgelu"
    for absent in ("vis.0.gelu", "vis.0.norm1", "vis.0.silu", "vis.out_rms", "vis.post_rms", "vis.0.relu"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - 8388.8464896) < 1e-6

    square = image_grid(vs, 448, 448)
    assert (square.ph, square.pw, square.ph * square.pw, square.merged) == (32, 32, 1024, 256)
    sb = _by(vision_ops(vs, square, 1, "bf16"))
    _ln(sb["vis.0.attn_ln"], 1024, H25, 0)
    _ln(sb["vis.proj_ln"], 1024, H25, DEPTH)
    _gelu(sb["vis.mgelu"], 256, HM25, DEPTH)
    assert abs(vision_flops(vs, square, 1) / 1e9 - 1001.502277632) < 1e-6

    wide = image_grid(vs, 640, 480)
    assert (wide.ph, wide.pw, wide.ph * wide.pw, wide.merged) == (36, 46, 1656, 414)
    wb = _by(vision_ops(vs, wide, 1, "bf16"))
    _ln(wb["vis.out_ln"], 1656, H25, DEPTH)
    _ln(wb["vis.proj_ln"], 1656, H25, DEPTH)
    _gelu(wb["vis.mgelu"], 414, HM25, DEPTH)
    assert abs(vision_flops(vs, wide, 1) / 1e9 - 1749.82975488) < 1e-6

    tall = image_grid(vs, 1024, 768)
    assert (tall.ph, tall.pw, tall.ph * tall.pw, tall.merged) == (56, 74, 4144, 1036)
    tb = _by(vision_ops(vs, tall, 1, "bf16"))
    _ln(tb["vis.0.mlp_ln"], 4144, H25, 0)
    _gelu(tb["vis.mgelu"], 1036, HM25, DEPTH)
    assert tb["vis.merger.0"].m == 1036 and tb["vis.merger.1"].n == TEXT
    assert abs(vision_flops(vs, tall, 1) / 1e9 - 5661.563830272) < 1e-6

    two = _by(vision_ops(vs, g, 2, "bf16"))
    _ln(two["vis.0.attn_ln"], 2 * ROWS, H25, 0)
    _ln(two["vis.0.mlp_ln"], 2 * ROWS, H25, 0)
    _ln(two["vis.out_ln"], 2 * ROWS, H25, DEPTH)
    _ln(two["vis.proj_ln"], 2 * ROWS, H25, DEPTH)
    _gelu(two["vis.mgelu"], 2 * MERGED, HM25, DEPTH)
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS25
    assert two["vis.merger.0"].m == 2 * MERGED

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.0.mlp_in"].a_fmt == "fp8"
    assert forced["vis.merger.0"].a_fmt == "fp8" and forced["vis.merger.1"].a_fmt == "fp8"
    assert forced["vis.merger.0"].w_fmt == "bf16" and forced["vis.0.qkv"].w_bits == 16.0
    assert forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
    assert forced["vis.mgelu"].a_fmt == "bf16" and forced["vis.mgelu"].vec == by["vis.mgelu"].vec
    assert forced["vis.0.attn_ln"].a_fmt == "fp32" and forced["vis.proj_ln"].a_fmt == "fp32"

    pre = evaluate(Scenario(model=K25, serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == MERGED


def test_k25_video_mean_sits_between_final_ln_and_projector_ln():
    m = get_model(K25)
    g = video_grid(m.vision, video_pre(m), 4, 1280, 720).slice
    assert (g.ph, g.pw, g.tframes, g.ph * g.pw, g.merged) == (50, 86, 4, 4300, 1075)
    ops = vision_ops(m.vision, g, 1, "bf16")
    by = _by(ops)
    _ln(by["vis.out_ln"], 17200, H25, DEPTH)
    _ln(by["vis.proj_ln"], 4300, H25, DEPTH)
    _gelu(by["vis.mgelu"], 1075, HM25, DEPTH)
    assert by["vis.merger.0"].m == 1075 and by["vis.merger.1"].m == 1075
    names = [op.name for op in ops]
    assert names.index("vis.out_ln") < names.index("vis.tpool") < names.index("vis.proj_ln")
    assert names.index("vis.proj_ln") < names.index("vis.merger.0") < names.index("vis.mgelu")
    assert names.index("vis.mgelu") < names.index("vis.merger.1")
    assert "vis.post_rms" not in by and "vis.0.gelu" not in by


def test_k3_bills_rmsnorm_and_projector_gelu():
    m = get_model(K3)
    pc = m.param_check()
    assert pc["ours"] == 2779481465440 and pc["release"] == 2779484478208
    assert pc["rel_err"] == (pc["ours"] - pc["release"]) / pc["release"]
    assert pc["rel_err"] != 0.0
    vs = m.vision
    assert vs is not None and vs.family == "kimi" and vs.ln_mode == "k3" and vs.vec_fmt == "bf16"
    assert vs.gated is False and vs.depth == DEPTH and vs.heads == HEADS3 and vs.qkv == Q3
    assert vs.hidden == H3 and vs.inter == INTER3 and vs.merge == 2
    assert vs.post == ((HM3, HM3), (HM3, TEXT)) and vs.params == 447358976
    rel = load_release(m.hf_id)
    vc = rel["config"]["vision_config"]
    assert vc["norm_type"] == "rmsnorm" and vc["activation_func"] == "gelu_pytorch_tanh"
    assert vc["mm_projector_type"] == "patchmergerv2" and vc["mlp_type"] == "mlp2"
    assert vc["projector_hidden_act"] == "gelu" and vc["qkv_hidden_size"] == Q3
    assert set(rel["roles"]["vision"]) == {"bf16"}
    note = " ".join(m.notes)
    assert "PytorchGELUTanh" in note and "qkv_hidden_size" in note and "post RMSNorm" in note
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.ph * g.pw, g.merged) == (74, 74, ROWS, MERGED)
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _rms(by["vis.0.norm1"], ROWS, H3, 0)
    _rms(by["vis.0.norm2"], ROWS, H3, 0)
    assert by["vis.0.norm1"].sfu == float(ROWS) and by["vis.0.norm1"].sfu != float(ROWS * HEADS3)
    _rms(by[f"vis.{DEPTH - 1}.norm1"], ROWS, H3, DEPTH - 1)
    _rms(by["vis.out_rms"], ROWS, H3, DEPTH)
    _gelu(by["vis.mgelu"], MERGED, HM3, DEPTH)
    _rms(by["vis.post_rms"], MERGED, TEXT, DEPTH)
    assert by["vis.post_rms"].vec != MERGED * HM3 * 4
    assert by["vis.0.qkv"].n == 3 * Q3 and by["vis.0.qkv"].vec == ROWS * (3 * Q3) * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.0.mlp_in"].n == INTER3 and by["vis.0.mlp_in"].vec == ROWS * INTER3 * 3
    assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].k == HM3 and by["vis.merger.0"].n == HM3
    assert by["vis.merger.1"].n == TEXT
    assert by["vis.0.qk"].m == ROWS and by["vis.0.qk"].count == HEADS3
    assert by["vis.0.softmax"].sfu == 0.0 and by["vis.0.softmax"].vec == HEADS3 * ROWS * ROWS * 5
    names = [op.name for op in ops]
    assert names.index("vis.0.norm1") < names.index("vis.0.qkv") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.norm2") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index(f"vis.{DEPTH - 1}.softmax") < names.index("vis.out_rms")
    assert names.index("vis.out_rms") < names.index("vis.merger.0") < names.index("vis.mgelu")
    assert names.index("vis.mgelu") < names.index("vis.merger.1") < names.index("vis.post_rms")
    norms = [op for op in ops if op.name.endswith(".norm1") or op.name.endswith(".norm2")
             or op.name in ("vis.out_rms", "vis.post_rms")]
    gels = [op for op in ops if "gelu" in op.name]
    assert len(norms) == 2 * DEPTH + 2 and len(gels) == 1
    for absent in ("vis.0.attn_ln", "vis.proj_ln", "vis.0.gelu", "vis.0.silu", "vis.out_ln"):
        assert absent not in by, absent
    assert abs(vision_flops(vs, g, 1) / 1e9 - 9448.28514304) < 1e-6

    square = image_grid(vs, 448, 448)
    assert (square.ph * square.pw, square.merged) == (1024, 256)
    assert abs(vision_flops(vs, square, 1) / 1e9 - 1010.550439936) < 1e-6
    wide = image_grid(vs, 640, 480)
    assert (wide.ph * wide.pw, wide.merged) == (1656, 414)
    assert abs(vision_flops(vs, wide, 1) / 1e9 - 1807.86659328) < 1e-6
    tall = image_grid(vs, 1024, 768)
    assert (tall.ph * tall.pw, tall.merged) == (4144, 1036)
    tb = _by(vision_ops(vs, tall, 1, "bf16"))
    _rms(tb["vis.out_rms"], 4144, H3, DEPTH)
    _gelu(tb["vis.mgelu"], 1036, HM3, DEPTH)
    _rms(tb["vis.post_rms"], 1036, TEXT, DEPTH)
    assert abs(vision_flops(vs, tall, 1) / 1e9 - 6234.383712256) < 1e-6

    two = _by(vision_ops(vs, g, 2, "bf16"))
    _rms(two["vis.0.norm1"], 2 * ROWS, H3, 0)
    _rms(two["vis.out_rms"], 2 * ROWS, H3, DEPTH)
    _gelu(two["vis.mgelu"], 2 * MERGED, HM3, DEPTH)
    _rms(two["vis.post_rms"], 2 * MERGED, TEXT, DEPTH)
    assert two["vis.0.qk"].m == ROWS and two["vis.0.qk"].count == 2 * HEADS3
    assert two["vis.merger.0"].m == 2 * MERGED

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    assert forced["vis.0.qkv"].a_fmt == "fp8" and forced["vis.merger.0"].a_fmt == "fp8"
    assert forced["vis.merger.0"].w_fmt == "bf16" and forced["vis.0.qkv"].w_bits == 16.0
    assert forced["vis.0.qkv"].vec == by["vis.0.qkv"].vec
    assert forced["vis.mgelu"].a_fmt == "bf16" and forced["vis.0.norm1"].a_fmt == "fp32"
    assert forced["vis.post_rms"].a_fmt == "fp32"

    pre = evaluate(Scenario(model=K3, serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == MERGED


def test_kimi_refuses_other_activations_and_a_missing_dtype():
    rel = load_release(HF25)
    for act in ("silu", "gelu_pytorch_tanh"):
        cfg = copy.deepcopy(rel["config"])
        cfg["vision_config"]["projector_hidden_act"] = act
        _raises(lambda: vision_spec(HF25, cfg), "kimi projector_hidden_act is not gelu")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("projector_hidden_act")
    _raises(lambda: vision_spec(HF25, missing), "kimi projector_hidden_act is not gelu")
    tanh = copy.deepcopy(rel["config"])
    tanh["vision_config"]["activation_func"] = "silu"
    _raises(lambda: vision_spec(HF25, tanh), "kimi moonvit activation_func is not gelu_pytorch_tanh")
    wide_q = copy.deepcopy(rel["config"])
    wide_q["vision_config"]["qkv_hidden_size"] = Q3
    _raises(lambda: vision_spec(HF25, wide_q), "kimi moonvit qkv width is not hidden")
    rms = copy.deepcopy(rel["config"])
    rms["vision_config"]["norm_type"] = "rmsnorm"
    _raises(lambda: vision_spec(HF25, rms), "kimi moonvit norm_type is not layernorm")

    bare = vision_spec(HF25, rel["config"])
    assert bare is not None and bare.vec_fmt == "" and bare.ln_mode == "k25"
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "kimi vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "kimi vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", gated=True), g, 1),
            "kimi vision block is not fused qkv plus a two-linear merger")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", post=()), g, 1),
            "kimi vision block is not fused qkv plus a two-linear merger")
    spec = from_release("synth", HF25, rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16" and spec.vision.ln_mode == "k25"

    rel3 = load_release(HF3)
    silu = copy.deepcopy(rel3["config"])
    silu["vision_config"]["activation_func"] = "silu"
    _raises(lambda: vision_spec(HF3, silu), "kimi v2 activation_func is not gelu_pytorch_tanh")
    layernorm = copy.deepcopy(rel3["config"])
    layernorm["vision_config"]["norm_type"] = "layernorm"
    _raises(lambda: vision_spec(HF3, layernorm), "kimi v2 norm_type is not rmsnorm")
    other_mlp = copy.deepcopy(rel3["config"])
    other_mlp["vision_config"]["mlp_type"] = "swiglu"
    _raises(lambda: vision_spec(HF3, other_mlp), "kimi v2 mlp_type is not mlp2")
    bare3 = vision_spec(HF3, rel3["config"])
    assert bare3 is not None and bare3.ln_mode == "k3" and bare3.vec_fmt == ""
    g3 = image_grid(bare3, 1024, 1024)
    _raises(lambda: vision_ops(bare3, g3, 1), "kimi vision activation dtype is not a vision float")
    spec3 = from_release("synth3", HF3, rel=rel3, cfg=rel3["config"])
    assert spec3.vision is not None and spec3.vision.vec_fmt == "bf16" and spec3.vision.ln_mode == "k3"
    assert spec3.param_check()["ours"] == 2779481465440


def test_other_towers_do_not_gain_kimi_moonvit_norms():
    for mid in ("qwen3.5-9b", "deepseek-v4.1-flash", "glm-5.3-flash"):
        m = get_model(mid)
        assert m.vision is not None and m.vision.ln_mode not in ("k25", "k3"), mid
        names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
        assert "vis.out_ln" not in names, mid
        assert "Kimi MoonViT" not in " ".join(m.notes)
        assert "Kimi-K3 MoonViT" not in " ".join(m.notes)
