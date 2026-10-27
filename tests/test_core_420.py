"""0.420: d1-omni SigLIP2 LayerNorm, projector erf-GELU, trunk RMSNorm and SiLU product.

Siglip2EncoderLayer and post_layernorm run on the padded crop. Projector.forward is
F.gelu (erf) after the 2x2 unshuffle. The tower MLP stays gelu_pytorch_tanh.
Trunk RMSNorm upcasts to fp32. q/k reduce over head_dim. F.silu(w1) * w3 uses the
FFN width. The scorer is LayerNorm and nn.GELU() on marker rows. The
TransformerEncoderLayer default activation stays in the GEMM tail.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import d1_text_len, d1_trunk_ops, image_grid, vision_ops, vision_spec

MID = "d1-omni-600m"
HF = "LiquidAI/d1-omni-600M"
ATTN = (2, 5, 8, 10, 12, 14)


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


def _gelu(op, n: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def test_d1_vision_bills_layernorm_and_projector_erf_gelu():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    vs = m.vision
    assert vs.family == "d1omni" and vs.hidden == 768 and vs.depth == 12 and vs.inter == 3072
    rel = load_release(HF)
    assert rel["config"]["vision_config"]["hidden_act"] == "gelu_pytorch_tanh"
    assert "projector_hidden_act" not in rel["config"]
    assert rel["config"]["projector_hidden_size"] == 2048
    g = image_grid(vs, 1024, 1024)
    assert g.views == ((32, 32, 5),) and g.llm_tokens == 1280
    ops = vision_ops(vs, g, 1, "bf16")
    by = _by(ops)
    pad_rows = 5 * 1024
    _ln(by["vis.0.attn_ln"], pad_rows, 768, 0)
    _ln(by["vis.0.mlp_ln"], pad_rows, 768, 0)
    _ln(by["vis.11.mlp_ln"], pad_rows, 768, 11)
    _ln(by["vis.post_ln"], pad_rows, 768, 12)
    assert by["vis.post_ln"].vec != 1280 * 768 * 4
    _gelu(by["vis.pgelu"], 1280 * 2048, 12)
    assert by["vis.pgelu"].vec != pad_rows * 2048 * 5
    assert by["vis.pgelu"].vec != 1280 * 768 * 5
    assert by["vis.pgelu"].vec != 1280 * 1024 * 5
    assert by["vis.0.fc1"].vec == pad_rows * 3072 * 3
    assert by["vis.0.fc1"].w_fmt == "fp32" and by["vis.0.fc1"].w_bits == 32.0
    assert by["vis.merger.0"].m == 1280 and (by["vis.merger.0"].k, by["vis.merger.0"].n) == (3072, 2048)
    assert by["vis.merger.1"].m == 1280 and (by["vis.merger.1"].k, by["vis.merger.1"].n) == (2048, 1024)
    assert by["vis.merger.0"].vec == 1280 * 2048 * 3
    assert by["vis.0.softmax"].sfu == 0 and by["vis.0.qk"].count == 60 and by["vis.0.qk"].m == 1024
    names = [op.name for op in ops]
    assert names.index("vis.0.attn_ln") < names.index("vis.0.q") < names.index("vis.0.mlp_ln")
    assert names.index("vis.0.mlp_ln") < names.index("vis.0.fc1") < names.index("vis.post_ln")
    assert names.index("vis.post_ln") < names.index("vis.merger.0") < names.index("vis.pgelu")
    assert names.index("vis.pgelu") < names.index("vis.merger.1")
    assert "vis.0.gelu" not in by and "vis.mgelu" not in by and "vis.pre_ln" not in by
    assert "vis.proj_ln" not in by and "vis.0.silu" not in by
    ln_names = [name for name in by if name.endswith(".attn_ln") or name.endswith(".mlp_ln")
                or name.endswith("post_ln")]
    assert len(ln_names) == 2 * 12 + 1
    assert sum(by[name].w_params for name in ln_names + ["vis.pgelu"]) == 0
    text = " ".join(m.notes)
    for needle in ("F.gelu", "erf", "gelu_pytorch_tanh", "post_layernorm", "projector_hidden_size",
                   "head_dim", "F.silu", "nn.GELU()", "F.relu", "尾项不减"):
        assert needle in text, needle

    two = _by(vision_ops(vs, g, 2, "fp32"))
    assert two["vis.post_ln"].vec == 2 * by["vis.post_ln"].vec
    assert two["vis.post_ln"].sfu == 2 * pad_rows
    assert two["vis.pgelu"].vec == 2 * by["vis.pgelu"].vec
    assert two["vis.0.qk"].m == 1024 and two["vis.0.qk"].count == 120
    assert two["vis.0.fc1"].vec == 2 * by["vis.0.fc1"].vec

    forced = _by(vision_ops(vs, g, 1, "fp8"))
    for name in ("vis.patch_embed", "vis.0.q", "vis.0.fc1", "vis.merger.0", "vis.merger.1"):
        assert forced[name].a_fmt == "fp32" and forced[name].w_fmt == "fp32", name
        assert forced[name].w_bits == 32.0 and forced[name].vec == by[name].vec
    assert forced["vis.pgelu"].a_fmt == "fp32" and forced["vis.post_ln"].a_fmt == "fp32"
    assert forced["vis.0.attn_ln"].vec == by["vis.0.attn_ln"].vec

    wide = image_grid(vs, 1280, 720)
    assert wide.views == ((32, 32, 2), (24, 42, 1))
    split = _by(vision_ops(vs, wide, 1, "fp32"))
    _ln(split["vis.post_ln"], 2 * 1024, 768, 12)
    _ln(split["vis.v1.post_ln"], 1024, 768, 12)
    _gelu(split["vis.pgelu"], 512 * 2048, 12)
    _gelu(split["vis.v1.pgelu"], 252 * 2048, 12)
    assert split["vis.v1.pgelu"].vec != 252 * 768 * 5
    assert split["vis.0.fc1"].m == 2048 and split["vis.v1.merger.0"].m == 252

    small = image_grid(vs, 256, 256)
    assert small.views == ((16, 16, 1),) and small.llm_tokens == 64
    tiny = _by(vision_ops(vs, small, 1))
    _ln(tiny["vis.post_ln"], 1024, 768, 12)
    _gelu(tiny["vis.pgelu"], 64 * 2048, 12)
    assert tiny["vis.pgelu"].vec != 64 * 1024 * 5
    pre = evaluate(Scenario(model=MID, serving=Serving(phase="prefill", batch=1, prompt=128, images=1)))
    assert pre.vision["image_tokens"] == 1280 and pre.step > 0 and pre.fits


def test_d1_trunk_bills_rmsnorm_silu_and_scorer_gelu():
    m = get_model(MID)
    vis = m.vision
    g = image_grid(vis, 1024, 1024)
    text = d1_text_len(vis, 4096, g.llm_tokens, 1, 0.0)
    assert text == 896
    length = g.llm_tokens + text
    seqs = 2
    ops = d1_trunk_ops(vis, seqs, text, g.llm_tokens, 3, "fp8")
    by = _by(ops)
    rows = seqs * length
    assert rows == 2 * 2176
    h, ff, nq, nkv, hd = 1024, 4608, 16, 8, 64
    _rms(by["trunk.0.op_rms"], rows, h, 0)
    _rms(by["trunk.0.ffn_rms"], rows, h, 0)
    _rms(by["trunk.15.ffn_rms"], rows, h, 15)
    _rms(by["trunk.out_rms"], rows, h, 16)
    assert by["trunk.out_rms"].sfu != float(seqs * text)
    assert "trunk.0.q_rms" not in by and "trunk.0.k_rms" not in by
    _rms(by["trunk.2.q_rms"], rows * nq, hd, 2)
    _rms(by["trunk.2.k_rms"], rows * nkv, hd, 2)
    assert by["trunk.2.q_rms"].sfu == float(rows * nq)
    assert by["trunk.2.k_rms"].sfu == float(rows * nkv)
    assert by["trunk.2.k_rms"].vec != by["trunk.2.q_rms"].vec
    assert by["trunk.2.k_rms"].vec != rows * h * 4
    assert by["trunk.2.q_rms"].sfu != float(rows)
    silu = by["trunk.0.silu"]
    assert silu.kind == "vector" and silu.layer == 0
    assert silu.vec == rows * ff * 4 and silu.sfu == rows * ff * 2
    assert silu.a_fmt == "fp32" and silu.w_params == 0 and silu.flops == 0
    assert silu.vec != rows * h * 4 and silu.vec != rows * (2 * ff) * 4
    mul = by["trunk.0.mlp_mul"]
    assert mul.vec == rows * ff and mul.sfu == 0 and mul.a_fmt == "fp32" and mul.w_params == 0
    assert by["trunk.2.silu"].vec == silu.vec and by["trunk.2.w1"].vec == rows * ff * 3
    assert by["trunk.2.w1"].w_fmt == "fp32" and by["trunk.2.w1"].a_fmt == "fp32"
    assert by["trunk.2.softmax"].sfu == 0
    _ln(by["head.scorer.ln"], seqs * 3, h, 2)
    assert by["head.scorer.ln"].sfu != float(seqs * text)
    _gelu(by["head.scorer.gelu"], seqs * 3 * h, 2)
    assert by["head.scorer.gelu"].vec != seqs * text * h * 5
    assert by["head.scorer.1"].m == 6 and by["head.scorer.3"].m == 6 and by["head.scorer.3"].n == 1
    assert by["head.scorer.1"].vec == 6 * h * 3
    names = [op.name for op in ops]
    assert names.index("trunk.0.op_rms") < names.index("trunk.0.in") < names.index("trunk.0.conv")
    assert names.index("trunk.0.out") < names.index("trunk.0.ffn_rms") < names.index("trunk.0.w1")
    assert names.index("trunk.0.w1") < names.index("trunk.0.w3") < names.index("trunk.0.silu")
    assert names.index("trunk.0.silu") < names.index("trunk.0.mlp_mul") < names.index("trunk.0.w2")
    assert names.index("trunk.2.q") < names.index("trunk.2.k") < names.index("trunk.2.q_rms")
    assert names.index("trunk.2.q_rms") < names.index("trunk.2.k_rms") < names.index("trunk.2.v")
    assert names.index("trunk.2.o") < names.index("trunk.2.qk") < names.index("trunk.2.ffn_rms")
    assert names.index("trunk.15.w2") < names.index("trunk.out_rms") < names.index("head.type_emb")
    assert names.index("head.scorer.ln") < names.index("head.scorer.1") < names.index("head.scorer.gelu")
    assert names.index("head.scorer.gelu") < names.index("head.scorer.3")
    assert "head.0.relu" not in by and "head.0.ln" not in by and "head.0.gelu" not in by
    assert "head.1.silu" not in by
    rms_names = [name for name in by if name.endswith("_rms")]
    assert len(rms_names) == 16 + 16 + 1 + len(ATTN) + len(ATTN)
    assert len([name for name in by if name.endswith(".silu")]) == 16
    assert sum(by[name].w_params for name in rms_names) == 0
    assert sum(by[name].w_params for name in ("head.scorer.ln", "head.scorer.gelu")) == 0

    bare = _by(d1_trunk_ops(vis, 1, 128, 0, 0))
    assert "head.scorer.ln" not in bare and "head.scorer.gelu" not in bare and "head.scorer.1" not in bare
    _rms(bare["trunk.out_rms"], 128, h, 16)
    assert bare["trunk.0.silu"].vec == 128 * ff * 4
    _rms(bare["trunk.2.k_rms"], 128 * nkv, hd, 2)
    assert bare["trunk.2.k_rms"].a_fmt == "fp32" and bare["trunk.0.mlp_mul"].a_fmt == "fp32"


def test_d1_refuses_another_tower_activation():
    rel = load_release(HF)
    kept = vision_spec(HF, rel["config"])
    assert kept is not None and kept.family == "d1omni"

    def bad(act):
        cfg = copy.deepcopy(rel["config"])
        if act is None:
            cfg["vision_config"].pop("hidden_act")
        else:
            cfg["vision_config"]["hidden_act"] = act
        return cfg

    _raises(lambda: vision_spec(HF, bad("gelu")), "d1_omni hidden_act is not gelu_pytorch_tanh")
    _raises(lambda: vision_spec(HF, bad("quick_gelu")), "d1_omni hidden_act is not gelu_pytorch_tanh")
    _raises(lambda: vision_spec(HF, bad("silu")), "d1_omni hidden_act is not gelu_pytorch_tanh")
    _raises(lambda: vision_spec(HF, bad(None)), "d1_omni hidden_act is not gelu_pytorch_tanh")
    lfm = get_model("lfm2.5-vl-1.6b")
    names = [op.name for op in vision_ops(lfm.vision, image_grid(lfm.vision, 256, 256), 1)]
    assert "trunk.out_rms" not in names and "head.scorer.gelu" not in names
