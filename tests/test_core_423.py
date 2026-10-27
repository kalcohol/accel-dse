"""0.423: MiMo-V2.6 vision RMSNorm, gate SiLU, ln_q and merger erf-GELU.

nn.RMSNorm reduces in fp32, one rsqrt per token. ACT2FN["silu"] is
x * sigmoid(x) on the gate only. nn.LayerNorm (ln_q) runs on patch rows.
nn.GELU() after merger fc1 is the erf form at 4 * hidden. The shared GEMM
tail is unchanged. Family stays qwen.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import band_keys, image_grid, vision_flops, vision_ops, vision_spec

H = 1280
INTER = 4608
DEPTH = 28
HEADS = 32
KV = 8
HD = 64
HM = H * 4
PATCHES = 64 * 64
MERGED = 1024
FLASH = "mimo-v2.6-flash"
PRO = "mimo-v2.6-pro"
FLASH_HF = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
PRO_HF = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
CASES = (
    (FLASH, FLASH_HF, 4096, 989_721_600, 261_127_168, 4642.377302016),
    (PRO, PRO_HF, 6144, 1_033_761_792, 294_681_600, 4658.483429376),
)


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


def _ln(op, rows: int, width: int, layer: int) -> None:
    _rms(op, rows, width, layer)


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


def test_mimo_v26_vision_bills_rmsnorm_silu_and_merger_erf_gelu():
    for mid, hf, out_n, params, extra, gflop in CASES:
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        vs = m.vision
        assert vs is not None and vs.family == "qwen" and vs.ln_mode == "mimo", mid
        assert vs.vec_fmt == "bf16" and vs.gated and vs.kv_heads == KV, mid
        assert (vs.hidden, vs.inter, vs.depth, vs.heads, vs.head_dim) == (H, INTER, DEPTH, HEADS, HD)
        assert vs.post == ((HM, HM), (HM, out_n)), mid
        assert vs.params == params and vs.extra == extra, mid
        rel = load_release(hf)
        assert rel["config"]["vision_config"]["hidden_act"] == "silu"
        g = image_grid(vs, 1024, 1024)
        assert (g.ph, g.pw, g.merged, g.llm_tokens) == (64, 64, MERGED, MERGED), mid
        assert g.ph * g.pw == PATCHES
        ops = vision_ops(vs, g, 1, "bf16")
        by = _by(ops)
        _rms(by["vis.0.norm1"], PATCHES, H, 0)
        _rms(by["vis.0.norm2"], PATCHES, H, 0)
        assert by["vis.0.norm1"].sfu == float(PATCHES)
        assert by["vis.0.norm1"].sfu != float(PATCHES * HEADS)
        _rms(by[f"vis.{DEPTH - 1}.norm1"], PATCHES, H, DEPTH - 1)
        _rms(by[f"vis.{DEPTH - 1}.norm2"], PATCHES, H, DEPTH - 1)
        _silu(by["vis.0.silu"], PATCHES * INTER, 0, "bf16")
        assert by["vis.0.silu"].vec != PATCHES * (2 * INTER) * 4
        mul = by["vis.0.mlp_mul"]
        assert mul.kind == "vector" and mul.vec == PATCHES * INTER and mul.sfu == 0
        assert mul.a_fmt == "bf16" and mul.w_params == 0 and mul.flops == 0
        _ln(by["vis.ln_q"], PATCHES, H, DEPTH)
        assert by["vis.ln_q"].sfu != float(MERGED)
        assert by["vis.ln_q"].vec != PATCHES * HM * 4
        assert by["vis.ln_q"].vec != PATCHES * out_n * 4
        _gelu(by["vis.mgelu"], MERGED * HM, DEPTH, "bf16")
        assert by["vis.mgelu"].vec != MERGED * H * 5
        assert by["vis.mgelu"].vec != MERGED * out_n * 5
        assert by["vis.0.qkv"].n == (HEADS + 2 * KV) * HD
        assert by["vis.0.qkv"].vec == PATCHES * ((HEADS + 2 * KV) * HD) * 3
        assert by["vis.0.qkv"].w_fmt == "bf16" and by["vis.0.qkv"].w_bits == 16.0
        assert by["vis.0.mlp_in"].n == 2 * INTER
        assert by["vis.0.mlp_in"].vec == PATCHES * (2 * INTER) * 3
        assert by["vis.merger.0"].m == MERGED and (by["vis.merger.0"].k, by["vis.merger.0"].n) == (HM, HM)
        assert by["vis.merger.1"].m == MERGED and (by["vis.merger.1"].k, by["vis.merger.1"].n) == (HM, out_n)
        assert by["vis.merger.0"].vec == MERGED * HM * 3
        soft = by["vis.0.softmax"]
        assert soft.sfu == 0
        assert by["vis.0.qk"].m == PATCHES and by["vis.0.qk"].count == HEADS and by["vis.0.qk"].causal == 1.0
        names = [op.name for op in ops]
        assert names.index("vis.win_perm") < names.index("vis.0.norm1") < names.index("vis.0.qkv")
        assert names.index("vis.0.qkv") < names.index("vis.0.o") < names.index("vis.0.norm2")
        assert names.index("vis.0.norm2") < names.index("vis.0.mlp_in") < names.index("vis.0.silu")
        assert names.index("vis.0.silu") < names.index("vis.0.mlp_mul") < names.index("vis.0.mlp_out")
        assert names.index("vis.0.mlp_out") < names.index("vis.0.qk")
        assert names.index(f"vis.{DEPTH - 1}.mlp_out") < names.index("vis.ln_q") < names.index("vis.merger.0")
        assert names.index("vis.merger.0") < names.index("vis.mgelu") < names.index("vis.merger.1")
        for absent in ("vis.0.q_rms", "vis.0.k_rms", "vis.0.gelu", "vis.0.attn_ln",
                       "vis.pre_ln", "vis.proj_gelu", "vis.proj_ln", "vis.post_rms"):
            assert absent not in by, (mid, absent)
        rms = [name for name in by if name.endswith((".norm1", ".norm2"))]
        assert len(rms) == 2 * DEPTH
        assert len([name for name in by if name.endswith(".silu")]) == DEPTH
        assert len([name for name in by if name.endswith(".mlp_mul")]) == DEPTH
        assert [name for name in by if name.endswith(".ln") or name == "vis.ln_q"] == ["vis.ln_q"]
        assert [name for name in by if "gelu" in name] == ["vis.mgelu"]
        billed = rms + ["vis.ln_q", "vis.mgelu"]
        billed += [f"vis.{i}.silu" for i in range(DEPTH)] + [f"vis.{i}.mlp_mul" for i in range(DEPTH)]
        assert sum(by[name].w_params for name in billed) == 0
        assert sum(by[name].flops for name in billed) == 0

        wide = image_grid(vs, 640, 480)
        assert (wide.ph, wide.pw, wide.merged) == (30, 40, 300), mid
        wpatches = 30 * 40
        wb = _by(vision_ops(vs, wide, 1, "bf16"))
        _rms(wb["vis.0.norm1"], wpatches, H, 0)
        _ln(wb["vis.ln_q"], wpatches, H, DEPTH)
        _silu(wb["vis.0.silu"], wpatches * INTER, 0, "bf16")
        _gelu(wb["vis.mgelu"], 300 * HM, DEPTH, "bf16")
        assert wb["vis.0.qk"].m == wpatches and wb["vis.0.qk"].count == HEADS

        tall = image_grid(vs, 1024, 768)
        assert (tall.ph, tall.pw, tall.merged, tall.llm_tokens) == (48, 64, 768, 768), mid
        tb = _by(vision_ops(vs, tall, 1, "bf16"))
        n_tall = 48 * 64
        _ln(tb["vis.ln_q"], n_tall, H, DEPTH)
        _gelu(tb["vis.mgelu"], 768 * HM, DEPTH, "bf16")
        assert tb["vis.1.qk"].m == n_tall and tb["vis.1.qk"].causal == band_keys(n_tall, 64) / (n_tall * n_tall)
        assert tb["vis.1.softmax"].sfu == 0
        assert abs(vision_flops(vs, tall, 1) / 1e9 - gflop) <= 1e-6, mid

        two = _by(vision_ops(vs, g, 2, "bf16"))
        assert two["vis.0.norm1"].vec == 2 * by["vis.0.norm1"].vec
        assert two["vis.0.norm2"].sfu == 2 * by["vis.0.norm2"].sfu
        assert two["vis.0.silu"].vec == 2 * by["vis.0.silu"].vec
        assert two["vis.ln_q"].sfu == 2 * by["vis.ln_q"].sfu
        assert two["vis.mgelu"].vec == 2 * by["vis.mgelu"].vec
        assert two["vis.0.qk"].m == PATCHES and two["vis.0.qk"].count == 2 * HEADS
        assert two["vis.0.softmax"].sfu == 0
        assert two["vis.0.mlp_in"].vec == 2 * by["vis.0.mlp_in"].vec

        forced = _by(vision_ops(vs, g, 1, "fp8"))
        for name in ("vis.patch_embed", "vis.0.qkv", "vis.0.o", "vis.0.mlp_in", "vis.0.mlp_out",
                     "vis.merger.0", "vis.merger.1"):
            assert forced[name].a_fmt == "fp8" and forced[name].w_fmt == "bf16", (mid, name)
            assert forced[name].w_bits == 16.0 and forced[name].vec == by[name].vec, (mid, name)
        for name in ("vis.0.silu", "vis.0.mlp_mul", "vis.mgelu"):
            assert forced[name].a_fmt == "bf16" and forced[name].vec == by[name].vec, (mid, name)
        for name in ("vis.0.norm1", "vis.0.norm2", "vis.ln_q"):
            assert forced[name].a_fmt == "fp32" and forced[name].vec == by[name].vec, (mid, name)

        text = " ".join(m.notes)
        for needle in ("silu", "erf", "fp32", "bf16", "torch_dtype", "尾项不减", "RoPE", "整份残差"):
            assert needle in text, (mid, needle)
        assert "MiMo-V2-Flash" not in text, mid

    pre = evaluate(Scenario(model=FLASH, serving=Serving(phase="prefill", batch=1, prompt=16, images=1)))
    assert pre.vision["image_tokens"] == MERGED and pre.step > 0


def test_mimo_v26_refuses_other_activations_and_a_missing_dtype():
    rel = load_release(FLASH_HF)
    for act in ("gelu", "gelu_pytorch_tanh"):
        cfg = copy.deepcopy(rel["config"])
        cfg["vision_config"]["hidden_act"] = act
        _raises(lambda: vision_spec(FLASH_HF, cfg), "mimo vision hidden_act is not silu")
    missing = copy.deepcopy(rel["config"])
    missing["vision_config"].pop("hidden_act")
    _raises(lambda: vision_spec(FLASH_HF, missing), "mimo vision hidden_act is not silu")

    bare = vision_spec(FLASH_HF, rel["config"])
    assert bare is not None and bare.vec_fmt == "" and bare.ln_mode == "mimo" and bare.family == "qwen"
    g = image_grid(bare, 1024, 1024)
    _raises(lambda: vision_ops(bare, g, 1), "mimo vision activation dtype is not a vision float")
    from dataclasses import replace
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "mimo vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", post=()), g, 1),
            "mimo vision block is not gqa swiglu plus a two-linear merger")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="bf16", gated=False), g, 1),
            "mimo vision block is not gqa swiglu plus a two-linear merger")
    spec = from_release("synth", FLASH_HF, rel=rel, cfg=rel["config"])
    assert spec.vision is not None and spec.vision.vec_fmt == "bf16" and spec.vision.ln_mode == "mimo"
    assert spec.param_check()["rel_err"] == 0.0
    pro = from_release("synth-pro", PRO_HF, rel=load_release(PRO_HF))
    assert pro.vision is not None and pro.vision.ln_mode == "mimo" and pro.vision.post[1][1] == 6144


def test_other_towers_do_not_gain_mimo_names():
    for mid in ("qwen3-vl-2b", "qwen3.5-9b", "mistral-small-3.2-24b-instruct-2506",
                "lfm2.5-vl-1.6b", "d1-omni-600m", "kimi-k2.5", "deepseek-v4.1-flash",
                "glm-5.3-flash"):
        m = get_model(mid)
        assert m.vision is not None and m.vision.ln_mode != "mimo", mid
        names = [op.name for op in vision_ops(m.vision, image_grid(m.vision, 1024, 1024), 1)]
        assert "vis.ln_q" not in names, mid
        if mid not in ("glm-5.3-flash", "deepseek-v4.1-flash"):
            assert "vis.0.norm1" not in names, mid
        if mid not in ("mistral-small-3.2-24b-instruct-2506", "glm-5.3-flash", "deepseek-v4.1-flash"):
            assert "vis.0.silu" not in names, mid
