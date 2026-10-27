"""0.435: MiniCPM-V-4.7 vision LayerNorm and merger erf-GELU.

Each block's layer_norm1 / layer_norm2 are nn.LayerNorm. insert_layer_id still
runs at full resolution; vit_merger follows that layer. The final merger's
nn.GELU() is the erf form. Tower MLP and vit_merger gelu_pytorch_tanh are not
billed. Vector flops stay 0. GEMM tails stay in place.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid, vision_flops, vision_ops, vision_spec

MID = "minicpm-v-4.7-35b-a3b"
HF = "openbmb/MiniCPM-V-4.7-35B-A3B"
H, DEPTH, HEADS, INTER, TEXT, INSERT = 1152, 27, 16, 4304, 2048, 6
FULL, QUART, MERGED = 32 * 32, 256, 64
PARAMS = 552_986_032
VISION_BYTES = 1_105_972_064
LANG = 34_659_889_792
FLOP_448 = 485_816_795_136
FLOP_768 = 2_387_692_339_200
FLOP_1024 = 3_202_823_356_416
FLOP_480 = 1_549_428_572_160
FLOP_1080 = 4_388_973_281_280


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _norm(op, rows: int, width: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _gelu(op, rows: int, width: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 5 and op.sfu == float(rows * width)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_minicpm_v47_bills_layernorm_and_merger_erf_gelu():
    m = get_model(MID)
    vs = m.vision
    assert vs.family == "minicpm" and vs.ln_mode == "minicpm" and vs.vec_fmt == "bf16"
    assert vs.gated is False and vs.kv_heads == 0 and vs.deepstack == 0 and vs.insert == INSERT
    assert vs.hidden == H and vs.depth == DEPTH and vs.heads == HEADS and vs.inter == INTER
    assert vs.qkv == H and vs.patch == 14 and vs.merge == 4
    assert vs.params == PARAMS and vs.extra == 0 and vs.w_Bpp == 2.0
    assert abs(vs.params * vs.w_Bpp - VISION_BYTES) < 1
    assert vs.mid == ((H, 3 * H, 1), (H, H, 1), (4 * H, 4 * INTER, 4), (4 * INTER, H, 4))
    assert vs.post == ((4 * H, 4 * H, 16), (4 * H, TEXT, 16))
    assert m.params() == LANG and m.release_params == LANG
    assert m.param_check()["rel_err"] == 0.0 and m.coverage == "full"
    rel = load_release(HF)
    assert rel["config"]["vision_config"]["hidden_act"] == "gelu_pytorch_tanh"
    assert rel["config"]["text_config"]["hidden_act"] == "silu"
    assert rel["config"]["insert_layer_id"] == INSERT and rel["config"]["downsample_mode"] == "16x"
    assert "merger_times" not in rel["config"] and "window_kernel_size" not in rel["config"]["vision_config"]
    role = rel["roles"]["vision"]
    assert set(role) == {"bf16"}
    assert role["bf16"]["params"] == PARAMS and role["bf16"]["bytes"] == VISION_BYTES
    note = " ".join(m.notes)
    assert "nn.LayerNorm(hidden)" in note and "nn.GELU()" in note and "gelu_pytorch_tanh" in note
    assert "insert_layer_id" in note and "4*hidden" in note
    g = image_grid(vs, 448, 448)
    assert g.views == ((32, 32, 1),) and g.llm_tokens == 69 and g.merged == MERGED
    ops = vision_ops(vs, g, 1, "bf16")
    names = [o.name for o in ops]
    order = [
        "vis.6.attn_ln", "vis.6.qkv", "vis.6.o", "vis.6.mlp_ln", "vis.6.mlp_in", "vis.6.mlp_out",
        "vis.6.qk", "vis.6.pv", "vis.6.softmax",
        "vis.vit_merger.ln1", "vis.vit_merger.0", "vis.vit_merger.1", "vis.vit_merger.pre_ln",
        "vis.vit_merger.2", "vis.vit_merger.3", "vis.vit_merger.qk",
        "vis.7.attn_ln", "vis.post_ln", "vis.merger.pre_ln", "vis.merger.0", "vis.mgelu", "vis.merger.1",
    ]
    idx = [names.index(n) for n in order]
    assert idx == sorted(idx)
    assert names.index("vis.vit_merger.3") == names.index("vis.vit_merger.2") + 1
    by = _by(ops)
    _norm(by["vis.0.attn_ln"], FULL, H, 0)
    _norm(by["vis.0.mlp_ln"], FULL, H, 0)
    _norm(by["vis.6.attn_ln"], FULL, H, INSERT)
    _norm(by["vis.6.mlp_ln"], FULL, H, INSERT)
    _norm(by["vis.7.attn_ln"], QUART, H, 7)
    _norm(by["vis.7.mlp_ln"], QUART, H, 7)
    _norm(by["vis.26.attn_ln"], QUART, H, 26)
    _norm(by["vis.26.mlp_ln"], QUART, H, 26)
    _norm(by["vis.vit_merger.ln1"], FULL, H, INSERT)
    _norm(by["vis.vit_merger.pre_ln"], QUART, 4 * H, INSERT)
    _norm(by["vis.post_ln"], QUART, H, DEPTH)
    _norm(by["vis.merger.pre_ln"], MERGED, 4 * H, DEPTH)
    _gelu(by["vis.mgelu"], MERGED, 4 * H, DEPTH, "bf16")
    lns = [o for o in ops if o.a_fmt == "fp32"]
    assert len(lns) == DEPTH * 2 + 4
    assert [o.name for o in ops if o.name.endswith("gelu") or "mgelu" in o.name] == ["vis.mgelu"]
    assert not any(n.endswith(".gelu") or "silu" in n or "quick_gelu" in n for n in names)
    qkv = by["vis.6.qkv"]
    assert qkv.m == FULL and qkv.k == H and qkv.n == 3 * H and qkv.vec == FULL * 3 * H * 3
    assert qkv.w_fmt == "bf16" and qkv.w_bits == 16.0 and qkv.a_fmt == "bf16"
    assert by["vis.5.qkv"].m == FULL and by["vis.7.qkv"].m == QUART
    assert by["vis.7.qkv"].vec == QUART * 3 * H * 3 and by["vis.7.qkv"].w_fmt == "bf16"
    assert by["vis.6.qk"].m == FULL and by["vis.6.qk"].count == HEADS
    assert by["vis.7.qk"].m == QUART and by["vis.6.softmax"].sfu == 0.0
    assert by["vis.6.softmax"].vec == HEADS * FULL * FULL * 5
    assert by["vis.vit_merger.0"].m == FULL and by["vis.vit_merger.0"].n == 3 * H
    assert by["vis.vit_merger.2"].m == QUART and by["vis.vit_merger.2"].k == 4 * H
    assert by["vis.vit_merger.2"].n == 4 * INTER and by["vis.vit_merger.2"].vec == QUART * 4 * INTER * 3
    assert by["vis.vit_merger.qk"].m == FULL
    assert by["vis.merger.0"].m == MERGED and by["vis.merger.0"].n == 4 * H
    assert by["vis.merger.0"].vec == MERGED * 4 * H * 3
    assert by["vis.merger.1"].m == MERGED and by["vis.merger.1"].n == TEXT
    assert by["vis.patch_embed"].w_fmt == "bf16" and by["vis.patch_embed"].w_bits == 16.0
    assert abs(vision_flops(vs, g, 1) - FLOP_448) < 1.0
    tiny = image_grid(vs, 8, 8)
    assert tiny == image_grid(vs, 256, 256) == g
    wide = image_grid(vs, 1024, 768)
    assert wide.views == ((28, 36, 1), (28, 36, 4)) and wide.llm_tokens == 329
    wby = _by(vision_ops(vs, wide, 1))
    _norm(wby["vis.0.attn_ln"], 1008, H, 0)
    _norm(wby["vis.6.mlp_ln"], 1008, H, INSERT)
    _norm(wby["vis.7.attn_ln"], 252, H, 7)
    _norm(wby["vis.vit_merger.ln1"], 1008, H, INSERT)
    _norm(wby["vis.vit_merger.pre_ln"], 252, 4 * H, INSERT)
    _norm(wby["vis.post_ln"], 252, H, DEPTH)
    _norm(wby["vis.merger.pre_ln"], 63, 4 * H, DEPTH)
    _gelu(wby["vis.mgelu"], 63, 4 * H, DEPTH, "bf16")
    _norm(wby["vis.v1.0.attn_ln"], 4032, H, 0)
    _norm(wby["vis.v1.6.attn_ln"], 4032, H, INSERT)
    _norm(wby["vis.v1.7.attn_ln"], 1008, H, 7)
    _norm(wby["vis.v1.post_ln"], 1008, H, DEPTH)
    _gelu(wby["vis.v1.mgelu"], 252, 4 * H, DEPTH, "bf16")
    assert wby["vis.6.qkv"].m == 1008 and wby["vis.7.qkv"].m == 252
    assert wby["vis.v1.6.qkv"].m == 4032 and wby["vis.v1.7.qkv"].m == 1008
    assert wby["vis.v1.7.qk"].m == 252 and wby["vis.v1.7.qk"].count == 4 * HEADS
    assert wby["vis.merger.1"].m == 63 and wby["vis.v1.merger.1"].m == 252
    assert wby["vis.v1.vit_merger.qk"].m == 1008
    assert abs(vision_flops(vs, wide, 1) - FLOP_768) < 1.0
    square = image_grid(vs, 1024, 1024)
    assert square.views == ((32, 32, 1), (40, 24, 6)) and square.llm_tokens == 442
    sby = _by(vision_ops(vs, square, 1))
    _norm(sby["vis.6.attn_ln"], FULL, H, INSERT)
    _norm(sby["vis.v1.6.attn_ln"], 6 * 960, H, INSERT)
    _norm(sby["vis.v1.7.attn_ln"], 6 * 240, H, 7)
    _gelu(sby["vis.mgelu"], MERGED, 4 * H, DEPTH, "bf16")
    _gelu(sby["vis.v1.mgelu"], 6 * 60, 4 * H, DEPTH, "bf16")
    assert sby["vis.v1.7.qk"].m == 240 and sby["vis.v1.merger.1"].m == 360
    assert abs(vision_flops(vs, square, 1) - FLOP_1024) < 1.0
    mid = image_grid(vs, 640, 480)
    assert mid.views == ((28, 36, 1), (40, 28, 2)) and mid.llm_tokens == 212
    assert abs(vision_flops(vs, mid, 1) - FLOP_480) < 1.0
    hd = image_grid(vs, 1920, 1080)
    assert hd.views == ((24, 44, 1), (32, 32, 8)) and hd.llm_tokens == 600
    assert abs(vision_flops(vs, hd, 1) - FLOP_1080) < 1.0
    two = _by(vision_ops(vs, g, 2, "bf16"))
    _norm(two["vis.6.attn_ln"], 2 * FULL, H, INSERT)
    _norm(two["vis.7.mlp_ln"], 2 * QUART, H, 7)
    _norm(two["vis.vit_merger.pre_ln"], 2 * QUART, 4 * H, INSERT)
    _norm(two["vis.post_ln"], 2 * QUART, H, DEPTH)
    _gelu(two["vis.mgelu"], 2 * MERGED, 4 * H, DEPTH, "bf16")
    assert two["vis.6.qkv"].m == 2 * FULL and two["vis.6.qkv"].vec == 2 * FULL * 3 * H * 3
    assert two["vis.6.qk"].m == FULL and two["vis.6.qk"].count == 2 * HEADS
    assert two["vis.merger.0"].m == 2 * MERGED
    pair = _by(vision_ops(vs, g, 2, "fp8"))
    assert pair["vis.6.qkv"].a_fmt == "fp8" and pair["vis.6.qkv"].w_fmt == "bf16"
    assert pair["vis.6.qkv"].w_bits == 16.0 and pair["vis.6.qkv"].vec == 2 * FULL * 3 * H * 3
    assert pair["vis.6.mlp_in"].a_fmt == "fp8" and pair["vis.vit_merger.2"].a_fmt == "fp8"
    assert pair["vis.merger.1"].a_fmt == "fp8" and pair["vis.merger.1"].w_fmt == "bf16"
    assert pair["vis.patch_embed"].a_fmt == "fp8" and pair["vis.patch_embed"].w_fmt == "bf16"
    assert pair["vis.mgelu"].a_fmt == "bf16" and pair["vis.post_ln"].a_fmt == "fp32"
    assert pair["vis.6.softmax"].sfu == 0.0
    ev = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", prompt=32, images=1)))
    assert ev.vision["image_tokens"] == 442
    for other_id in ("qwen3-vl-2b", "minimax-m3", "ernie-4.5-vl-28b-a3b", "ling-3.0-flash-vl"):
        other = get_model(other_id)
        assert other.vision is not None and other.vision.family != "minicpm"
        assert other.vision.ln_mode != "minicpm"
        assert "vit_merger" not in " ".join(other.notes)


def test_minicpm_refuses_other_shapes():
    rel = load_release(HF)
    base = rel["config"]
    bare = vision_spec(HF, base)
    assert bare is not None and bare.ln_mode == "minicpm" and bare.vec_fmt == ""
    assert bare.params == PARAMS and bare.insert == INSERT
    g = image_grid(bare, 448, 448)
    _raises(lambda: vision_ops(bare, g, 1), "minicpm vision activation dtype is not a vision float")
    _raises(lambda: vision_ops(replace(bare, vec_fmt="fp8"), g, 1),
            "minicpm vision activation dtype is not a vision float")
    billed = replace(bare, vec_fmt="bf16")
    _raises(lambda: vision_ops(replace(billed, gated=True), g, 1),
            "minicpm vision block is not fused qkv plus the mergers")
    _raises(lambda: vision_ops(replace(billed, qkv=H * 2), g, 1),
            "minicpm vision block is not fused qkv plus the mergers")
    _raises(lambda: vision_ops(replace(billed, ln_mode=""), g, 1),
            "minicpm vision activation dtype is not a vision float")
    early = replace(billed, insert=0)
    eby = _by(vision_ops(early, g, 1))
    assert eby["vis.0.qkv"].m == FULL and eby["vis.1.qkv"].m == QUART
    assert eby["vis.vit_merger.ln1"].layer == 0 and eby["vis.post_ln"].sfu == float(QUART)
    last = replace(billed, insert=DEPTH - 1)
    lby = _by(vision_ops(last, g, 1))
    assert lby["vis.26.qkv"].m == FULL and lby["vis.post_ln"].sfu == float(QUART)
    assert "vis.27.qkv" not in lby
    silu = copy.deepcopy(base)
    silu["vision_config"]["hidden_act"] = "silu"
    _raises(lambda: vision_spec(HF, silu), "minicpm vision hidden_act is not gelu_pytorch_tanh")
    quick = copy.deepcopy(base)
    quick["vision_config"]["hidden_act"] = "quick_gelu"
    _raises(lambda: vision_spec(HF, quick), "minicpm vision hidden_act is not gelu_pytorch_tanh")
    four = copy.deepcopy(base)
    four["downsample_mode"] = "4x"
    _raises(lambda: vision_spec(HF, four), "minicpm downsample_mode is not 16x")
    times = copy.deepcopy(base)
    times["merger_times"] = 2
    _raises(lambda: vision_spec(HF, times), "minicpm merger_times is not 1")
    window = copy.deepcopy(base)
    window["vision_config"]["window_kernel_size"] = [3, 3]
    _raises(lambda: vision_spec(HF, window), "minicpm window_kernel_size is not (2, 2)")
    kernel = copy.deepcopy(base)
    kernel["merge_kernel_size"] = (1, 2)
    _raises(lambda: vision_spec(HF, kernel), "minicpm merge_kernel_size is not (2, 2)")
    outside = copy.deepcopy(base)
    outside["insert_layer_id"] = DEPTH
    _raises(lambda: vision_spec(HF, outside), "minicpm insert_layer_id is outside the tower")
    missing = copy.deepcopy(base)
    del missing["insert_layer_id"]
    del missing["downsample_mode"]
    built = vision_spec(HF, missing)
    assert built is not None and built.insert == INSERT and built.params == PARAMS
