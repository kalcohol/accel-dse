"""0.422: GLM-5.3-Flash vision RMSNorm, gate SiLU, merger LayerNorm and erf-GELU.

Glm5NextRMSNorm reduces in fp32. q/k norms are per head. ACT2FN["silu"] is
x * sigmoid(x) on the gate only. nn.GELU() after merger.proj is the erf form.
The swiglu_limit clamps are not ops. The shared GEMM tail is unchanged.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

MID = "glm-5.3-flash"
HF = "zai-org/GLM-5.3-Flash"
H = 1024
INTER = 4096
DEPTH = 24
HEADS = 16
HD = 64
OUT = 4096
PI = 10240
PATCHES = 74 * 74
MERGED = 1369


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _rms(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _silu(op, n: int, layer: int, fmt: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == fmt and op.w_params == 0 and op.flops == 0


def _gelu(op, n: int, layer: int, fmt: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == fmt and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_glm5_vision_bills_rmsnorm_silu_and_merger_erf_gelu():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs is not None and vs.family == "glm" and vs.vec_fmt == "bf16"
    assert (vs.hidden, vs.inter, vs.depth, vs.heads, vs.head_dim) == (H, INTER, DEPTH, HEADS, HD)
    assert vs.post == ((H * 4, OUT), (OUT, OUT), (OUT, 2 * PI), (PI, OUT))
    rel = load_release(HF)
    vc = rel["config"]["vision_config"]
    assert vc["hidden_act"] == "silu" and vc["swiglu_limit"] == 10.0
    assert vc["attention_dropout"] == 0.0 and vc["rms_norm_eps"] == 1e-5
    g = image_grid(vs, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens, g.px) == (74, 74, MERGED, MERGED, (1036, 1036))
    assert g.ph * g.pw == PATCHES
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    _rms(by["vis.0.norm1"], PATCHES, H, 0)
    _rms(by["vis.0.norm2"], PATCHES, H, 0)
    _rms(by["vis.0.q_rms"], PATCHES * HEADS, HD, 0)
    _rms(by["vis.0.k_rms"], PATCHES * HEADS, HD, 0)
    assert by["vis.0.q_rms"].sfu == float(PATCHES * HEADS)
    assert by["vis.0.q_rms"].sfu != float(PATCHES)
    assert by["vis.0.k_rms"].sfu == by["vis.0.q_rms"].sfu
    assert by["vis.0.q_rms"].vec == by["vis.0.norm1"].vec
    assert by["vis.0.q_rms"].vec != PATCHES * HD * 4
    _rms(by[f"vis.{DEPTH - 1}.norm1"], PATCHES, H, DEPTH - 1)
    _rms(by[f"vis.{DEPTH - 1}.q_rms"], PATCHES * HEADS, HD, DEPTH - 1)
    _rms(by["vis.post_rms"], PATCHES, H, DEPTH)
    assert by["vis.post_rms"].sfu != float(MERGED)
    assert by["vis.post_rms"].vec != PATCHES * OUT * 4
    _silu(by["vis.0.silu"], PATCHES * INTER, 0, "bf16")
    assert by["vis.0.silu"].vec != PATCHES * (2 * INTER) * 4
    mul = by["vis.0.mlp_mul"]
    assert mul.kind == "vector" and mul.vec == PATCHES * INTER and mul.sfu == 0
    assert mul.a_fmt == "bf16" and mul.w_params == 0 and mul.flops == 0
    _gelu(by["vis.proj_gelu"], MERGED * OUT, DEPTH, "bf16")
    assert by["vis.proj_gelu"].vec != MERGED * H * 5
    assert by["vis.proj_gelu"].vec != MERGED * PI * 5
    _rms_ln = by["vis.proj_ln"]
    _rms(_rms_ln, MERGED, OUT, DEPTH)
    assert _rms_ln.a_fmt == "fp32"
    assert by["vis.proj_ln"].vec != PATCHES * OUT * 4
    assert by["vis.proj_ln"].vec != MERGED * H * 4
    _silu(by["vis.proj_silu"], MERGED * PI, DEPTH, "bf16")
    assert by["vis.proj_silu"].vec != MERGED * (2 * PI) * 4
    assert by["vis.proj_mul"].vec == MERGED * PI and by["vis.proj_mul"].sfu == 0
    assert by["vis.0.qkv"].vec == PATCHES * (3 * H) * 3
    assert by["vis.0.mlp_in"].n == 2 * INTER
    assert by["vis.0.mlp_in"].vec == PATCHES * (2 * INTER) * 3
    assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
    assert by["vis.merger.0"].m == MERGED and (by["vis.merger.0"].k, by["vis.merger.0"].n) == (4 * H, OUT)
    assert (by["vis.merger.1"].k, by["vis.merger.1"].n) == (OUT, OUT)
    assert by["vis.merger.2"].n == 2 * PI and by["vis.merger.3"].k == PI
    assert by["vis.merger.1"].vec == MERGED * OUT * 3
    soft = by["vis.0.softmax"]
    assert soft.sfu == 0 and soft.vec == HEADS * PATCHES * PATCHES * 5
    assert by["vis.0.qk"].m == PATCHES and by["vis.0.qk"].count == HEADS
    names = [op.name for op in ops]
    assert names.index("vis.0.norm1") < names.index("vis.0.qkv") < names.index("vis.0.q_rms")
    assert names.index("vis.0.q_rms") < names.index("vis.0.k_rms") < names.index("vis.0.o")
    assert names.index("vis.0.o") < names.index("vis.0.norm2") < names.index("vis.0.mlp_in")
    assert names.index("vis.0.mlp_in") < names.index("vis.0.silu") < names.index("vis.0.mlp_mul")
    assert names.index("vis.0.mlp_mul") < names.index("vis.0.mlp_out") < names.index("vis.0.qk")
    assert names.index(f"vis.{DEPTH - 1}.mlp_out") < names.index("vis.post_rms") < names.index("vis.merger.0")
    assert names.index("vis.merger.0") < names.index("vis.merger.1") < names.index("vis.proj_ln")
    assert names.index("vis.proj_ln") < names.index("vis.proj_gelu") < names.index("vis.merger.2")
    assert names.index("vis.merger.2") < names.index("vis.proj_silu") < names.index("vis.proj_mul")
    assert names.index("vis.proj_mul") < names.index("vis.merger.3")
    assert "vis.0.gelu" not in by and "vis.0.attn_ln" not in by and "vis.0.relu" not in by
    assert "vis.mgelu" not in by and "vis.proj_rms" not in by
    rms = [name for name in by if name.endswith((".norm1", ".norm2", ".q_rms", ".k_rms")) or name == "vis.post_rms"]
    assert len(rms) == 4 * DEPTH + 1
    assert [name for name in by if name.endswith(".ln") or name == "vis.proj_ln"] == ["vis.proj_ln"]
    assert len([name for name in by if name.endswith("silu")]) == DEPTH + 1
    assert len([name for name in by if name.endswith("_mul")]) == DEPTH + 1
    assert [name for name in by if "gelu" in name] == ["vis.proj_gelu"]
    billed = rms + ["vis.proj_ln", "vis.proj_gelu", "vis.proj_silu", "vis.proj_mul"]
    billed += [f"vis.{i}.silu" for i in range(DEPTH)] + [f"vis.{i}.mlp_mul" for i in range(DEPTH)]
    assert sum(by[name].w_params for name in billed) == 0
    assert sum(by[name].flops for name in billed) == 0

    wide = image_grid(vs, 640, 480)
    assert (wide.ph, wide.pw, wide.merged) == (36, 46, 414)
    wpatches = 36 * 46
    wb = _by(vision_ops(vs, wide, 1, "bf16"))
    _rms(wb["vis.post_rms"], wpatches, H, DEPTH)
    _silu(wb["vis.0.silu"], wpatches * INTER, 0, "bf16")
    _gelu(wb["vis.proj_gelu"], 414 * OUT, DEPTH, "bf16")
    _rms(wb["vis.0.q_rms"], wpatches * HEADS, HD, 0)
    assert wb["vis.0.qk"].m == wpatches and wb["vis.0.qk"].count == HEADS

    two = _by(vision_ops(vs, g, 2, "bf16"))
    assert two["vis.0.norm1"].vec == 2 * by["vis.0.norm1"].vec
    assert two["vis.0.q_rms"].sfu == 2 * by["vis.0.q_rms"].sfu
    assert two["vis.proj_gelu"].vec == 2 * by["vis.proj_gelu"].vec
    assert two["vis.post_rms"].sfu == 2 * by["vis.post_rms"].sfu
    assert two["vis.0.qk"].m == PATCHES and two["vis.0.qk"].count == 2 * HEADS
    assert two["vis.0.softmax"].sfu == 0
    assert two["vis.0.mlp_in"].vec == 2 * by["vis.0.mlp_in"].vec

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    for name in ("vis.patch_embed", "vis.0.qkv", "vis.0.o", "vis.0.mlp_in", "vis.0.mlp_out",
                 "vis.merger.0", "vis.merger.1", "vis.merger.2", "vis.merger.3"):
        assert forced[name].a_fmt == "fp8" and forced[name].w_fmt == "bf16", name
        assert forced[name].w_bits == 16.0 and forced[name].vec == by[name].vec, name
    for name in ("vis.0.silu", "vis.0.mlp_mul", "vis.proj_gelu", "vis.proj_silu", "vis.proj_mul"):
        assert forced[name].a_fmt == "bf16" and forced[name].vec == by[name].vec, name
    for name in ("vis.0.norm1", "vis.0.q_rms", "vis.0.k_rms", "vis.0.norm2", "vis.post_rms", "vis.proj_ln"):
        assert forced[name].a_fmt == "fp32" and forced[name].vec == by[name].vec, name

    text = " ".join(m.notes)
    for needle in ("silu", "head_dim", "erf", "fp32", "bf16", "torch_dtype", "尾项不减", "clamp", "RoPE"):
        assert needle in text, needle
    # 1024x768 is the published FLOP pin. Vector ops do not add GEMM FLOPs.
    g768 = image_grid(vs, 1024, 768)
    assert abs(vision_flops(vs, g768, 1) / 1e9 - 5365.561556992) <= 1e-6

    pre = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == MERGED and pre.step > 0


def test_glm5_refuses_other_activations_and_a_missing_dtype():
    rel = load_release(HF)
    gelu = copy.deepcopy(rel["config"])
    gelu["vision_config"]["hidden_act"] = "gelu"
    _raises(lambda: vision_spec(HF, gelu), "glm5 vision hidden_act is not silu")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("hidden_act")
    _raises(lambda: vision_spec(HF, missing), "glm5 vision hidden_act is not silu")
    tanh = copy.deepcopy(rel["config"])
    tanh["vision_config"]["hidden_act"] = "gelu_pytorch_tanh"
    _raises(lambda: vision_spec(HF, tanh), "glm5 vision hidden_act is not silu")

    bare = vision_spec(HF, rel["config"])
    assert bare is not None and bare.vec_fmt == "" and bare.family == "glm"
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "glm5 vision activation dtype is not a vision float")
    from dataclasses import replace
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "glm5 vision activation dtype is not a vision float")
    spec = from_release("synth", HF, rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16"
    assert spec.param_check()["rel_err"] == 0.0


def test_other_towers_do_not_gain_glm5_names():
    for mid in ("qwen3-vl-2b", "qwen3.5-9b", "mistral-small-3.2-24b-instruct-2506",
                "lfm2.5-vl-1.6b", "d1-omni-600m", "kimi-k2.5", "deepseek-v4.1-flash"):
        m = get_model(mid)
        assert m.vision is not None and m.vision.family != "glm", mid
        names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
        for needle in ("vis.post_rms", "vis.proj_silu", "vis.proj_mul", "vis.0.norm1"):
            # 0.426 bills DeepSeek Block.norm1. The GLM merger names stay absent.
            if mid == "deepseek-v4.1-flash" and needle == "vis.0.norm1":
                continue
            assert needle not in names, (mid, needle)
