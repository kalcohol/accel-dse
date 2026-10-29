"""0.436: DiT text-encoder RMSNorm, LayerNorm and SiLU.

Llama, Qwen2.5-VL, Qwen3 and Qwen3-VL text towers bill fp32 RMSNorm and the
checkpoint-dtype SiLU plus the gate*up multiply. T5 bills T5LayerNorm as
RMSNorm and leaves gelu_new unbilled. CLIP bills nn.LayerNorm and leaves
quick_gelu unbilled. Qwen3.5 bills SiLU only. umT5 RMSNorm is billed in 0.437.
Vector flops stay 0. GEMM tails stay at 3 element-ops per output.
"""
from __future__ import annotations

import json

import accel_dse.core.pipeline as P
from accel_dse import __version__
from accel_dse.core.pipeline import pipeline_for, text_ops


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


def _silu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 4 and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _mul(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _patch(key: str, edit):
    orig = P.component_data

    def fake(k):
        d = json.loads(json.dumps(orig(k)))
        if k == key:
            edit(d)
        return d

    P.component_data = fake
    return orig


def test_version_is_436():
    assert __version__ == "0.475.0"


def test_llama_bills_rmsnorm_and_silu_and_keeps_gemm_tail():
    te = pipeline_for("hunyuanvideo").text[0]
    ops = text_ops(te, 1, "fp16")[0]
    by = _by(ops)
    rows, h, inter, depth = 351, 4096, 14336, 32
    _rms(by["te.0.attn_rms"], rows, h, 0)
    _rms(by["te.0.mlp_rms"], rows, h, 0)
    _rms(by["te.31.attn_rms"], rows, h, 31)
    _rms(by["te.final_rms"], rows, h, depth)
    _silu(by["te.0.silu"], rows * inter, 0, "fp16")
    _mul(by["te.0.mlp_mul"], rows * inter, 0, "fp16")
    assert sum(1 for n in by if n.endswith(".attn_rms") or n.endswith(".mlp_rms") or n == "te.final_rms") == 65
    assert sum(1 for n in by if n.endswith(".silu")) == depth
    assert "te.0.q_rms" not in by and "te.0.k_rms" not in by
    assert "te.0.gelu" not in by and "te.0.quick_gelu" not in by and "te.0.attn_ln" not in by
    gemms = [o for o in ops if o.kind == "gemm"]
    assert gemms and all(o.vec == o.m * o.n * 3.0 and o.w_fmt == "fp16" and o.w_bits == 16 for o in gemms)
    assert all(o.flops == 0 for o in ops if o.kind == "vector" and not o.name.endswith(".softmax"))
    names = [o.name for o in ops]
    assert names.index("te.0.attn_rms") < names.index("te.0.qk") < names.index("te.0.softmax")
    assert names.index("te.0.softmax") < names.index("te.0.mlp_rms") < names.index("te.0.silu")
    assert names.index("te.0.mlp_mul") < names.index("te.1.attn_rms")
    assert names[-1] == "te.final_rms"
    assert all(names.index(g.name) < names.index("te.0.attn_rms") for g in gemms)


def test_llama_two_prompts_and_requested_gemm_dtype():
    te = pipeline_for("hunyuanvideo").text[0]
    ops = text_ops(te, 2, "fp8")[0]
    by = _by(ops)
    rows = 702
    _rms(by["te.0.attn_rms"], rows, 4096, 0)
    _rms(by["te.final_rms"], rows, 4096, 32)
    _silu(by["te.17.silu"], rows * 14336, 17, "fp16")
    _mul(by["te.31.mlp_mul"], rows * 14336, 31, "fp16")
    g = [o for o in ops if o.kind == "gemm"][0]
    assert g.a_fmt == "fp8" and g.m == rows and g.vec == rows * g.n * 3.0 and g.w_fmt == "fp16"


def test_qwen3_bills_qk_rmsnorm():
    te = pipeline_for("flux.2-klein-4b").text[0]
    ops = text_ops(te, 1, "bf16")[0]
    by = _by(ops)
    rows, h, inter, hd = 512, 2560, 9728, 128
    _rms(by["te.0.attn_rms"], rows, h, 0)
    _rms(by["te.0.q_rms"], rows * 32, hd, 0)
    _rms(by["te.0.k_rms"], rows * 8, hd, 0)
    _rms(by["te.35.k_rms"], rows * 8, hd, 35)
    _rms(by["te.final_rms"], rows, h, 36)
    _silu(by["te.0.silu"], rows * inter, 0, "bf16")
    _mul(by["te.0.mlp_mul"], rows * inter, 0, "bf16")
    assert sum(1 for n in by if n.endswith(".q_rms")) == 36
    assert sum(1 for n in by if n.endswith(".k_rms")) == 36
    assert "te.0.gelu" not in by and "te.0.attn_ln" not in by
    names = [o.name for o in ops]
    assert names.index("te.0.attn_rms") < names.index("te.0.q_rms") < names.index("te.0.k_rms")
    assert names.index("te.0.k_rms") < names.index("te.0.qk")
    fp8 = _by(text_ops(te, 1, "fp8")[0])
    assert fp8["te.0.silu"].a_fmt == "bf16" and fp8["te.0.q_rms"].a_fmt == "fp32"
    g = [o for o in text_ops(te, 1, "fp8")[0] if o.name.endswith("q_proj.weight")][0]
    assert g.a_fmt == "fp8" and g.w_fmt == "bf16" and g.w_bits == 16 and g.vec == g.m * g.n * 3.0


def test_qwen25_has_no_qk_norm_and_qwen3vl_does():
    te = pipeline_for("qwen-image").text[0]
    by = _by(text_ops(te, 1, "bf16")[0])
    rows, h, inter = 290, 3584, 18944
    _rms(by["te.0.attn_rms"], rows, h, 0)
    _rms(by["te.final_rms"], rows, h, 28)
    _silu(by["te.27.silu"], rows * inter, 27, "bf16")
    assert "te.0.q_rms" not in by and "te.0.k_rms" not in by
    assert sum(1 for n in by if n.endswith(".silu")) == 28

    te = pipeline_for("qwen-image-2.1-turbo").text[0]
    by = _by(text_ops(te, 1, "bf16")[0])
    rows = 270
    _rms(by["te.0.q_rms"], rows * 32, 128, 0)
    _rms(by["te.0.k_rms"], rows * 8, 128, 0)
    _silu(by["te.0.silu"], rows * 12288, 0, "bf16")
    _rms(by["te.final_rms"], rows, 4096, 36)

    te = pipeline_for("minimax-h3").text[0]
    by = _by(text_ops(te, 1, "bf16")[0])
    rows = 512
    _rms(by["te.0.attn_rms"], rows, 5120, 0)
    _rms(by["te.0.q_rms"], rows * 64, 128, 0)
    _rms(by["te.63.k_rms"], rows * 8, 128, 63)
    _silu(by["te.50.silu"], rows * 25600, 50, "bf16")
    _rms(by["te.final_rms"], rows, 5120, 64)
    assert sum(1 for n in by if n.endswith(".q_rms")) == 64

    for mid, tokens in (("lingbot-video-moe-dmd", 1024), ("flux-3-image-klein", 320)):
        te = pipeline_for(mid).text[0]
        by = _by(text_ops(te, 1, "bf16")[0])
        _rms(by["te.0.attn_rms"], tokens, 2560, 0)
        _rms(by["te.0.q_rms"], tokens * 32, 128, 0)
        _rms(by["te.0.k_rms"], tokens * 8, 128, 0)
        _silu(by["te.9.silu"], tokens * 9728, 9, "bf16")
        _rms(by["te.final_rms"], tokens, 2560, 36)


def test_t5_bills_rmsnorm_and_not_gelu_new():
    cases = (
        ("cogvideox-5b", 226, "bf16"),
        ("cogvideox-2b", 226, "fp16"),
        ("ltx-video", 128, "fp32"),
        ("mochi-1", 256, "fp32"),
        ("opensora-stdit3", 300, "fp32"),
    )
    for mid, tokens, af in cases:
        te = pipeline_for(mid).text[0]
        ops = text_ops(te, 1, "fp8")[0]
        by = _by(ops)
        _rms(by["te.0.attn_rms"], tokens, 4096, 0)
        _rms(by["te.0.mlp_rms"], tokens, 4096, 0)
        _rms(by["te.23.mlp_rms"], tokens, 4096, 23)
        _rms(by["te.final_rms"], tokens, 4096, 24)
        assert sum(1 for n in by if ".attn_rms" in n or ".mlp_rms" in n or n == "te.final_rms") == 49
        assert not any(n.endswith((".silu", ".mlp_mul", ".gelu", ".quick_gelu", ".q_rms")) for n in by)
        names = [o.name for o in ops]
        assert names.index("te.0.attn_rms") < names.index("te.0.qk") < names.index("te.0.mlp_rms")
        g = [o for o in ops if o.kind == "gemm"][0]
        assert g.a_fmt == "fp8" and g.w_fmt == af and g.vec == tokens * g.n * 3.0
        assert all(o.causal == 1.0 for o in ops if o.kind == "attn")


def test_clip_bills_layernorm_and_not_quick_gelu():
    te = pipeline_for("hunyuanvideo").text[1]
    ops = text_ops(te, 1, "fp8")[0]
    by = _by(ops)
    rows, h = 77, 768
    _rms(by["te.0.attn_ln"], rows, h, 0)
    _rms(by["te.0.mlp_ln"], rows, h, 0)
    _rms(by["te.11.mlp_ln"], rows, h, 11)
    _rms(by["te.final_ln"], rows, h, 12)
    assert sum(1 for n in by if n.endswith("_ln")) == 25
    assert not any("quick_gelu" in n or n.endswith(".silu") or n.endswith(".gelu") for n in by)
    names = [o.name for o in ops]
    assert names.index("te.0.attn_ln") < names.index("te.0.qk") < names.index("te.0.mlp_ln")
    assert names[-1] == "te.final_ln"
    g = [o for o in ops if o.kind == "gemm"][0]
    assert g.w_fmt == "fp16" and g.a_fmt == "fp8" and g.vec == rows * g.n * 3.0
    assert all(o.causal == 0.5 for o in ops if o.kind == "attn")


def test_qwen35_bills_silu_and_not_zero_centered_rmsnorm():
    te = pipeline_for("moss-image-base").text[0]
    ops = text_ops(te, 1, "fp8")[0]
    by = _by(ops)
    rows, inter = 256, 12288
    _silu(by["te.0.silu"], rows * inter, 0, "bf16")
    _mul(by["te.0.mlp_mul"], rows * inter, 0, "bf16")
    _silu(by["te.3.silu"], rows * inter, 3, "bf16")
    _silu(by["te.31.silu"], rows * inter, 31, "bf16")
    assert sum(1 for n in by if n.endswith(".silu")) == 32
    assert not any(n.endswith((".attn_rms", ".mlp_rms", ".q_rms", ".k_rms")) or n in ("te.final_rms", "te.final_ln")
                   for n in by)
    assert not any("sigmoid" in n or n.endswith(".gelu") for n in by)
    names = [o.name for o in ops]
    assert sum(n.endswith(".softmax") for n in names) == 8
    assert sum(n.endswith(".lin_akk") for n in names) == 24
    assert names.index("te.0.lin_vec") < names.index("te.0.silu") < names.index("te.0.mlp_mul")
    assert names.index("te.3.softmax") < names.index("te.3.silu")
    g = [o for o in ops if o.kind == "gemm"][0]
    assert g.w_fmt == "bf16" and g.a_fmt == "fp8" and g.vec == rows * g.n * 3.0


def test_umt5_rms_keeps_the_gemm_flops():
    te = pipeline_for("wan2.1-14b").text[0]
    ops = text_ops(te, 2, "bf16")[0]
    names = [o.name for o in ops]
    assert sum(n.endswith(".softmax") for n in names) == 24
    rows, d, f, L, T = 1024, 4096, 10240, 24, 512
    ref = 2 * rows * L * (4 * d * d + 3 * d * f) + L * 2 * 64 * 4 * T * T * 64
    assert abs(sum(o.flops for o in ops) / ref - 1) < 1e-9
    assert names.count("te.0.attn_rms") == 1 and "te.final_rms" in names


def test_text_encoder_flops_stay_the_closed_form():
    te = pipeline_for("minimax-h3").text[0]
    ops = text_ops(te, 1, "bf16")[0]
    assert all(o.flops == 0 for o in ops if ".silu" in o.name or ".rms" in o.name or o.name.endswith(".mlp_mul"))
    k = [o for o in ops if o.name.endswith("layers.0.self_attn.k_proj.weight")][0]
    assert (k.k, k.n) == (5120, 8 * 128) and k.vec == k.m * k.n * 3.0
    assert all(o.causal == 0.5 for o in ops if o.kind == "attn")


def test_refuses_a_different_activation_or_a_bad_dtype():
    te = pipeline_for("flux.2-klein-4b").text[0]
    orig = _patch("qwen3-4b", lambda d: d["config"].__setitem__("hidden_act", "gelu"))
    try:
        _raises(lambda: text_ops(te, 1, "bf16"), "hidden_act is not silu")
    finally:
        P.component_data = orig
    te = pipeline_for("cogvideox-5b").text[0]
    orig = _patch("t5-v1_1-xxl.bf16", lambda d: d["config"].__setitem__("dense_act_fn", "gelu"))
    try:
        _raises(lambda: text_ops(te, 1, "bf16"), "dense_act_fn is not gelu_new")
    finally:
        P.component_data = orig
    te = pipeline_for("hunyuanvideo").text[1]
    orig = _patch("clip-vit-l-text", lambda d: d["config"].__setitem__("hidden_act", "gelu"))
    try:
        _raises(lambda: text_ops(te, 1, "bf16"), "hidden_act is not quick_gelu")
    finally:
        P.component_data = orig
    te = pipeline_for("hunyuanvideo").text[0]
    orig = _patch("llava-llama3-8b", lambda d: d.__setitem__("dtype", "fp8"))
    try:
        _raises(lambda: text_ops(te, 1, "fp16"), "activation dtype is not fp16, bf16, or fp32")
    finally:
        P.component_data = orig
