"""0.418: Pixtral bills the RMSNorms, SiLU, product, and projector erf-GELU.

PixtralMLP is down(silu(gate) * up). The projector is RMSNorm, the merger,
linear_1, ACT2FN["gelu"], linear_2. RMSNorm reduces in fp32. SiLU, the product,
and erf-GELU follow the vision-role dtype. The GEMM tail is unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_ops, vision_spec

H = 1024
INTER = 4096
DEPTH = 24
HEADS = 16
TEXT = 5120
PATCHES = 74 * 74
MERGED = 1369
WIDE_PATCHES = 62 * 110
WIDE_MERGED = 1705
MODELS = (
    "mistral-small-3.2-24b-instruct-2506",
    "mistral-small-3.1-24b-instruct-2503",
    "magistral-small-2509",
)


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _names(depth: int) -> list[str]:
    names = ["vis.patch_embed", "vis.ln_pre"]
    for li in range(depth):
        p = f"vis.{li}."
        names += [p + "attn_rms", p + "qkv", p + "qk", p + "pv", p + "softmax", p + "o",
                  p + "ffn_rms", p + "mlp_in", p + "silu", p + "mlp_mul", p + "mlp_out"]
    names += ["vis.proj_rms", "vis.merger.0", "vis.merger.1", "vis.proj_gelu", "vis.merger.2"]
    return names


def _rms(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(cfg: dict, needle: str) -> None:
    try:
        vision_spec("mistralai/Mistral-Small-3.2-24B-Instruct-2506", cfg)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _ops_raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_pixtral_bills_rmsnorm_silu_and_projector_gelu():
    for mid in MODELS:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "pixtral" and vs.vec_fmt == "bf16", mid
        assert vs.hidden == H and vs.inter == INTER and vs.depth == DEPTH and vs.heads == HEADS
        assert vs.gated is True and vs.params == 438_958_080
        rel = load_release(m.hf_id)
        assert set(rel["roles"]["vision"]) == {"bf16"}
        assert rel["roles"]["vision"]["bf16"]["params"] == 438_958_080
        assert rel["roles"]["vision"]["bf16"]["bytes"] == 438_958_080 * 2
        g = image_grid(vs, 1024, 1024)
        assert (g.ph, g.pw, g.merged, g.llm_tokens) == (74, 74, MERGED, 1406)
        ops = vision_ops(vs, g, 1, "bf16")
        by = _by(ops)
        assert [op.name for op in ops] == _names(DEPTH), mid
        _rms(by["vis.ln_pre"], PATCHES, H, 0)
        _rms(by["vis.0.attn_rms"], PATCHES, H, 0)
        _rms(by["vis.0.ffn_rms"], PATCHES, H, 0)
        _rms(by[f"vis.{DEPTH - 1}.attn_rms"], PATCHES, H, DEPTH - 1)
        _rms(by[f"vis.{DEPTH - 1}.ffn_rms"], PATCHES, H, DEPTH - 1)
        proj = by["vis.proj_rms"]
        _rms(proj, PATCHES, H, DEPTH)
        assert proj.vec != MERGED * H * 4 and proj.vec != PATCHES * TEXT * 4
        silu = by["vis.0.silu"]
        assert silu.kind == "vector" and silu.layer == 0
        assert silu.vec == PATCHES * INTER * 4 and silu.sfu == PATCHES * INTER * 2
        assert silu.a_fmt == "bf16" and silu.w_params == 0 and silu.flops == 0
        assert silu.vec != PATCHES * (2 * INTER) * 4
        mul = by["vis.0.mlp_mul"]
        assert mul.kind == "vector" and mul.layer == 0 and mul.sfu == 0
        assert mul.vec == PATCHES * INTER and mul.a_fmt == "bf16"
        assert mul.w_params == 0 and mul.flops == 0
        assert mul.vec != PATCHES * (2 * INTER)
        gelu = by["vis.proj_gelu"]
        assert gelu.kind == "vector" and gelu.layer == DEPTH
        assert gelu.vec == MERGED * TEXT * 5 and gelu.sfu == float(MERGED * TEXT)
        assert gelu.a_fmt == "bf16" and gelu.w_params == 0 and gelu.flops == 0
        assert gelu.vec != MERGED * H * 5 and gelu.vec != PATCHES * TEXT * 5
        assert by["vis.0.qkv"].vec == PATCHES * (3 * H) * 3
        assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
        assert by["vis.0.mlp_in"].vec == PATCHES * (2 * INTER) * 3
        assert by["vis.0.mlp_in"].n == 2 * INTER and by["vis.0.mlp_in"].w_fmt == "bf16"
        assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].k == 4 * H
        assert by["vis.merger.1"].m == MERGED and (by["vis.merger.1"].k, by["vis.merger.1"].n) == (H, TEXT)
        assert by["vis.merger.2"].m == MERGED and (by["vis.merger.2"].k, by["vis.merger.2"].n) == (TEXT, TEXT)
        assert by["vis.merger.1"].vec == MERGED * TEXT * 3
        soft = by["vis.0.softmax"]
        assert soft.sfu == 0 and soft.vec == HEADS * PATCHES * PATCHES * 5
        assert by["vis.0.qk"].m == PATCHES and by["vis.0.qk"].count == HEADS
        rms_names = [name for name in by if name == "vis.ln_pre" or name.endswith(".attn_rms")
                     or name.endswith(".ffn_rms") or name == "vis.proj_rms"]
        assert len(rms_names) == 1 + 2 * DEPTH + 1
        assert sum(op.w_params for op in ops if op.name in rms_names or op.name.endswith((".silu", ".mlp_mul"))
                   or op.name == "vis.proj_gelu") == 0
        assert "vis.0.glu" not in by and "vis.proj_gelu0" not in by

        two = _by(vision_ops(vs, g, 2, "bf16"))
        assert two["vis.ln_pre"].vec == 2 * by["vis.ln_pre"].vec
        assert two["vis.ln_pre"].sfu == 2 * PATCHES
        assert two["vis.0.silu"].vec == 2 * silu.vec and two["vis.0.silu"].sfu == 2 * silu.sfu
        assert two["vis.0.mlp_mul"].vec == 2 * mul.vec
        assert two["vis.proj_rms"].vec == 2 * proj.vec and two["vis.proj_rms"].sfu == 2 * PATCHES
        assert two["vis.proj_gelu"].vec == 2 * gelu.vec
        assert two["vis.0.qk"].m == PATCHES and two["vis.0.qk"].count == 2 * HEADS
        assert two["vis.0.softmax"].sfu == 0
        assert two["vis.0.mlp_in"].vec == 2 * by["vis.0.mlp_in"].vec

        wide_g = image_grid(vs, 1920, 1080)
        assert (wide_g.ph, wide_g.pw, wide_g.merged, wide_g.llm_tokens) == (62, 110, WIDE_MERGED, 1736)
        wide = _by(vision_ops(vs, wide_g, 1, "bf16"))
        _rms(wide["vis.ln_pre"], WIDE_PATCHES, H, 0)
        _rms(wide["vis.proj_rms"], WIDE_PATCHES, H, DEPTH)
        assert wide["vis.proj_rms"].vec != WIDE_MERGED * H * 4
        assert wide["vis.0.silu"].vec == WIDE_PATCHES * INTER * 4
        assert wide["vis.proj_gelu"].vec == WIDE_MERGED * TEXT * 5
        assert wide["vis.0.qk"].m == WIDE_PATCHES and wide["vis.0.qk"].count == HEADS

        forced = _by(vision_ops(vs, g, 1, "fp8"))
        for name in ("vis.patch_embed", "vis.0.qkv", "vis.0.o", "vis.0.mlp_in", "vis.0.mlp_out",
                     "vis.merger.0", "vis.merger.1", "vis.merger.2"):
            assert forced[name].a_fmt == "fp8" and forced[name].w_fmt == "bf16", (mid, name)
            assert forced[name].w_bits == 16.0 and forced[name].vec == by[name].vec, (mid, name)
        for name in ("vis.0.silu", "vis.0.mlp_mul", "vis.proj_gelu"):
            assert forced[name].a_fmt == "bf16" and forced[name].vec == by[name].vec, (mid, name)
        for name in ("vis.ln_pre", "vis.0.attn_rms", "vis.0.ffn_rms", "vis.proj_rms"):
            assert forced[name].a_fmt == "fp32" and forced[name].vec == by[name].vec, (mid, name)

        text = " ".join(m.notes)
        for needle in ("silu", "不是 GLU", "erf", "fp32", "bf16", "torch_dtype", "尾项不减", "RoPE"):
            assert needle in text, (mid, needle)

    pre = evaluate(Scenario(model=MODELS[0], serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == 1406 and pre.step > 0


def test_pixtral_refuses_other_activations_and_a_missing_dtype():
    rel = load_release("mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    tanh = copy.deepcopy(rel["config"])
    tanh["vision_config"]["hidden_act"] = "gelu_pytorch_tanh"
    _raises(tanh, "pixtral hidden_act is not silu")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("hidden_act")
    _raises(missing, "pixtral hidden_act is not silu")
    quick = copy.deepcopy(rel["config"])
    quick["projector_hidden_act"] = "quick_gelu"
    _raises(quick, "pixtral projector_hidden_act is not gelu")

    bare = vision_spec(rel["repo"], rel["config"])
    assert bare is not None and bare.vec_fmt == ""
    g = image_grid(bare, 1024, 1024)
    _ops_raises(lambda: vision_ops(bare, g, 1), "pixtral activation dtype is not a vision-role float")
    _ops_raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
                "pixtral activation dtype is not a vision-role float")

    listed = copy.deepcopy(rel["config"])
    listed["vision_feature_layer"] = [-1, -2]
    spec = from_release("synth", rel["repo"], rel=rel, cfg=listed)
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16"
    assert spec.vision.post[1][0] == 2 * H
    g2 = image_grid(spec.vision, 1024, 1024)
    _ops_raises(lambda: vision_ops(spec.vision, g2, 1),
                "pixtral projector is not norm, merger, linear, gelu, linear")


def test_other_towers_do_not_gain_pixtral_names():
    for mid in ("qwen3-vl-2b", "lfm2.5-vl-1.6b", "d1-omni-600m", "longcat-flash-omni", "gemma-4-e2b-it"):
        m = get_model(mid)
        assert m.vision is not None and m.vision.family != "pixtral", mid
        names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
        for needle in ("vis.ln_pre", "vis.proj_rms", "vis.proj_gelu", "vis.0.attn_rms", "vis.0.ffn_rms"):
            assert needle not in names, (mid, needle)
