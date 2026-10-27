"""0.421: d1-omni DecisionHead LayerNorm and constructor-default F.relu.

TransformerEncoderLayer is built without activation, so PyTorch 2.6 uses F.relu.
norm_first runs one LayerNorm before attention and one before the FFN, both at
d_model on text rows. F.relu is max(x, 0) on the 4 * d width, not ReLU6.
Dropout 0 and the full residual adds stay off the graph. The scorer is unchanged.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model
from accel_dse.core.model import load_release
from accel_dse.core.vision import d1_text_len, d1_trunk_ops, image_grid, vision_spec

MID = "d1-omni-600m"
HF = "LiquidAI/d1-omni-600M"


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


def _relu(op, n: int, layer: int) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n and op.sfu == 0
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0


def test_d1_decision_head_bills_layernorm_and_default_relu():
    m = get_model(MID)
    assert m.param_check()["rel_err"] == 0.0
    vis = m.vision
    rel = load_release(HF)
    assert rel["config"]["head_layers"] == 2
    assert int(vis.rz["hl"]) == 2 and int(vis.rz["hff"]) == 4096 and int(vis.rz["hh"]) == 16
    g = image_grid(vis, 1024, 1024)
    text = d1_text_len(vis, 4096, g.llm_tokens, 1, 0.0)
    assert text == 896 and g.llm_tokens == 1280
    seqs = 2
    ops = d1_trunk_ops(vis, seqs, text, g.llm_tokens, 3, "fp8")
    by = _by(ops)
    rows = seqs * text
    assert rows == 1792
    h, hff = 1024, 4096
    _ln(by["head.0.attn_ln"], rows, h, 0)
    _ln(by["head.0.mlp_ln"], rows, h, 0)
    _ln(by["head.1.attn_ln"], rows, h, 1)
    _ln(by["head.1.mlp_ln"], rows, h, 1)
    assert by["head.0.attn_ln"].sfu != float(seqs * 3)
    assert by["head.0.attn_ln"].vec != (seqs * (g.llm_tokens + text)) * h * 4
    assert by["head.0.mlp_ln"].vec != rows * hff * 4
    _relu(by["head.0.relu"], rows * hff, 0)
    _relu(by["head.1.relu"], rows * hff, 1)
    assert by["head.0.relu"].vec != rows * hff * 2
    assert by["head.0.relu"].vec != rows * h
    assert by["head.0.relu"].vec != rows * h * 5
    assert by["head.0.fc1"].vec == rows * hff * 3
    assert by["head.0.fc1"].w_fmt == "fp32" and by["head.0.fc1"].w_bits == 32.0
    assert by["head.0.fc1"].a_fmt == "fp32" and by["head.0.in"].a_fmt == "fp32"
    assert by["head.0.in"].vec == rows * (3 * h) * 3
    assert by["head.0.qk"].m == text and by["head.0.qk"].count == seqs * 16
    assert by["head.0.softmax"].sfu == 0
    _ln(by["head.scorer.ln"], seqs * 3, h, 2)
    assert by["head.scorer.gelu"].vec == seqs * 3 * h * 5
    assert by["head.scorer.1"].m == 6 and by["head.scorer.3"].n == 1
    names = [op.name for op in ops]
    assert names.index("head.type_emb") < names.index("head.0.attn_ln") < names.index("head.0.in")
    assert names.index("head.0.in") < names.index("head.0.o") < names.index("head.0.qk")
    assert names.index("head.0.softmax") < names.index("head.0.mlp_ln") < names.index("head.0.fc1")
    assert names.index("head.0.fc1") < names.index("head.0.relu") < names.index("head.0.fc2")
    assert names.index("head.0.fc2") < names.index("head.1.attn_ln") < names.index("head.1.relu")
    assert names.index("head.1.fc2") < names.index("head.scorer.ln")
    ln_names = [name for name in by if name.startswith("head.") and (
        name.endswith(".attn_ln") or name.endswith(".mlp_ln"))]
    assert ln_names == ["head.0.attn_ln", "head.0.mlp_ln", "head.1.attn_ln", "head.1.mlp_ln"]
    assert [name for name in by if name.endswith(".relu")] == ["head.0.relu", "head.1.relu"]
    assert "head.0.ln" not in by and "head.0.gelu" not in by and "head.scorer.relu" not in by
    assert "head.0.silu" not in by and "head.1.gelu" not in by
    assert sum(by[name].w_params for name in ln_names + ["head.0.relu", "head.1.relu"]) == 0
    text_notes = " ".join(m.notes)
    for needle in ("F.relu", "norm_first", "4 * hidden", "dropout", "max(x, 0)", "尾项不减"):
        assert needle in text_notes, needle

    one = _by(d1_trunk_ops(vis, 1, text, g.llm_tokens, 3, "fp32"))
    assert one["head.0.relu"].vec * 2 == by["head.0.relu"].vec
    assert one["head.0.attn_ln"].sfu * 2 == by["head.0.attn_ln"].sfu
    assert one["head.0.qk"].m == text and one["head.0.qk"].count == 16
    assert one["head.0.fc1"].vec * 2 == by["head.0.fc1"].vec

    bare = _by(d1_trunk_ops(vis, 1, 128, 0, 0))
    _relu(bare["head.0.relu"], 128 * hff, 0)
    _ln(bare["head.0.attn_ln"], 128, h, 0)
    _ln(bare["head.1.mlp_ln"], 128, h, 1)
    assert "head.scorer.ln" not in bare and "head.scorer.gelu" not in bare
    assert "head.scorer.relu" not in bare
    assert bare["head.0.qk"].m == 128 and bare["trunk.out_rms"].sfu == 128.0

    empty = _by(d1_trunk_ops(vis, 1, 0, 16, 0))
    assert "head.0.relu" not in empty and "head.0.attn_ln" not in empty
    assert "head.type_emb" not in empty and empty["trunk.0.op_rms"].sfu == 16.0

    au = vis.audio
    audio_text = d1_text_len(vis, 4096, au.tokens(1.0), 0, 1.0)
    assert au.tokens(1.0) == 13 and audio_text == 4096
    heard = _by(d1_trunk_ops(vis, 1, audio_text, au.tokens(1.0), 1, "bf16"))
    _relu(heard["head.0.relu"], audio_text * hff, 0)
    _ln(heard["head.0.mlp_ln"], audio_text, h, 0)
    assert heard["head.0.relu"].vec != (audio_text + 13) * hff
    assert heard["trunk.out_rms"].sfu == float(audio_text + 13)
    assert heard["head.scorer.ln"].sfu == 1.0
    assert heard["head.0.in"].a_fmt == "fp32" and heard["head.0.relu"].a_fmt == "fp32"


def test_d1_decision_head_follows_head_layers():
    rel = load_release(HF)
    cfg = copy.deepcopy(rel["config"])
    cfg["head_layers"] = 1
    spec = vision_spec(HF, cfg)
    assert spec is not None and int(spec.rz["hl"]) == 1
    ops = _by(d1_trunk_ops(spec, 1, 64, 0, 2))
    _relu(ops["head.0.relu"], 64 * 4096, 0)
    _ln(ops["head.0.attn_ln"], 64, 1024, 0)
    _ln(ops["head.0.mlp_ln"], 64, 1024, 0)
    _ln(ops["head.scorer.ln"], 2, 1024, 1)
    assert "head.1.relu" not in ops and "head.1.attn_ln" not in ops and "head.1.mlp_ln" not in ops
    assert ops["head.0.fc1"].n == 4096 and ops["head.scorer.gelu"].layer == 1
