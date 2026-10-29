"""0.439: ESMFold language layers bill the ESM-2 width, not the folding trunk.

modeling_esm.py: pre-LN nn.LayerNorm, erf-GELU, query * head_dim**-0.5, RoPE, no QK-norm.
modeling_esmfold.py: the encoder runs at hidden_size (2560). emb_layer_norm_after runs once
on the last layer output. esm_s_mlp is LayerNorm(2560), Linear to sequence_state_dim (1024),
nn.ReLU, Linear, on residue rows after <cls>/<eos> are stripped. There is no EsmLMHead.
Vector flops stay 0. GEMM shapes stay put. ModelSpec.hidden stays the trunk width.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, Shard, build_rank_ops, full_io_ops
from accel_dse.core.model import NativeWorkload, load_release
from accel_dse.core.scenario import Workload
from accel_dse.core.structure import build_esmfold


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


def _gelu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _relu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release("facebook/esmfold_v1")))
    edit(rel)
    wl = NativeWorkload("protein", seq_len=8)
    return lambda: build_esmfold("x", "facebook/esmfold_v1", rel, wl)


def _phase(seq_len: int, batch: int = 1):
    m = get_model("esmfold")
    wl = resolve_workload(m, Workload(seq_len=seq_len))
    ph = Phase("full", batch, wl.tokens, 0, pair=wl.pair)
    return m, wl, ph


def test_version_is_439():
    assert __version__ == "0.471.0"


def test_esmfold_language_layers_use_esm2_width():
    m, wl, ph = _phase(512)
    assert m.vec_kind == "" and m.hidden == 1024 and m.params() == m.release_params == 3527665186
    assert m.param_check()["rel_err"] == 0.0
    assert wl.tokens == 514 and wl.pair.n == 512
    lang = [i for i, L in enumerate(m.layers) if L.stack == "esm"]
    assert lang == list(range(36))
    assert m.layers[0].vec_kind == "esm2" and m.layers[0].vec_width == 2560
    assert m.layers[0].emb_ln is False and m.layers[35].emb_ln is True
    assert m.layers[0].core.qk_norm is False and m.layers[0].core.q_scale is True
    assert m.layers[0].ffn.gated is False and m.act_fmt == "bf16"
    t, h, inter, heads, hd = 514, 2560, 10240, 40, 64
    ops = build_rank_ops(m, 0, ph, Shard())
    by = _by(ops)
    _ln(by["attn_norm"], t, h, 0)
    _ln(by["ffn_norm"], t, h, 0)
    _gelu(by["gelu"], t * inter, 0, "bf16")
    assert by["q_scale"].vec == t * h and by["q_scale"].sfu == 0.0 and by["q_scale"].a_fmt == "bf16"
    assert by["rope"].vec == t * h * 6 and by["rope"].sfu == 0.0 and by["rope"].a_fmt == "bf16"
    assert by["residual"].vec == t * h * 2
    assert "emb_ln" not in by and "act" not in by and "qk_norm_rope" not in by
    assert "silu" not in by and "quick_gelu" not in by and "gelu_new" not in by
    names = [o.name for o in ops if o.kind == "gemm"]
    assert names == ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.up", "mlp.down"]
    q = by["attn.q"]
    assert (q.m, q.k, q.n) == (t, h, h) and q.w_fmt == "fp16" and q.w_bits == 16.0
    assert q.a_fmt == "bf16" and q.vec == 0.0 and q.flops == 2 * t * h * h
    assert by["mlp.up"].n == inter and by["mlp.down"].k == inter and by["mlp.down"].n == h
    assert by["mlp.up"].w_fmt == "fp16" and by["mlp.down"].vec == 0.0
    order = [o.name for o in ops]
    assert order.index("mlp.up") < order.index("mlp.down") < order.index("gelu")
    assert order.index("gelu") < order.index("residual")
    assert all(o.flops == 0 for o in ops if o.kind == "vector")
    last_ops = build_rank_ops(m, 35, ph, Shard())
    last = _by(last_ops)
    _ln(last["attn_norm"], t, h, 35)
    _gelu(last["gelu"], t * inter, 35, "bf16")
    _ln(last["emb_ln"], t, h, 35)
    last_names = [o.name for o in last_ops]
    assert last_names.index("residual") < last_names.index("emb_ln")
    trunk = next(i for i, L in enumerate(m.layers) if L.stack == "trunk")
    tr = _by(build_rank_ops(m, trunk, ph, Shard()))
    assert "gelu" not in tr and "attn_norm" not in tr and "esm_s_ln" not in tr
    assert tr["trunk.seq_att0.qk"].m == 512
    post = _by(full_io_ops(m, ph, Shard(), "post"))
    assert post["final_norm"].vec == t * m.hidden * 4 and post["final_norm"].layer == m.n_layers
    assert post["final_norm"].a_fmt == "bf16" and post["final_norm"].sfu == 0.0
    assert "lm_gelu" not in post and "lm_ln" not in post and "mlm_decoder" not in post and "emb_ln" not in post


def test_esmfold_projection_is_residue_rows():
    m, wl, ph = _phase(512)
    proj = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_proj")
    assert m.layers[proj].vec_width == 2560 and m.layers[proj].repeat == ""
    ops = build_rank_ops(m, proj, ph, Shard())
    by = _by(ops)
    _ln(by["esm_s_ln"], 512, 2560, proj)
    _relu(by["esm_s_relu"], 512 * 1024, proj, "bf16")
    g1, g3 = by["esm_proj.mlp.1"], by["esm_proj.mlp.3"]
    assert (g1.m, g1.k, g1.n) == (512, 2560, 1024) and g1.w_fmt == "fp32" and g1.w_bits == 32.0
    assert (g3.m, g3.k, g3.n) == (512, 1024, 1024) and g3.vec == 0.0 and g3.flops == 2 * 512 * 1024 * 1024
    assert by["esm_proj.mlp.1.ew"].vec == 512 * (2560 + 1024) * 4
    assert by["esm_proj.mlp.3.ew"].vec == 512 * (1024 + 1024) * 4
    names = [o.name for o in ops]
    assert names.index("esm_s_ln") < names.index("esm_proj.mlp.1") < names.index("esm_s_relu")
    assert names.index("esm_s_relu") < names.index("esm_proj.mlp.3")
    assert "gelu" not in by and "silu" not in by
    short_m, short_wl, short = _phase(128)
    assert short_wl.tokens == 130 and short_wl.pair.n == 128
    sby = _by(build_rank_ops(short_m, 0, short, Shard()))
    _gelu(sby["gelu"], 130 * 10240, 0, "bf16")
    _ln(sby["attn_norm"], 130, 2560, 0)
    pby = _by(build_rank_ops(short_m, proj, short, Shard()))
    _ln(pby["esm_s_ln"], 128, 2560, proj)
    _relu(pby["esm_s_relu"], 128 * 1024, proj, "bf16")
    tby = _by(build_rank_ops(m, 0, Phase("full", 2, wl.tokens, 0, pair=wl.pair), Shard()))
    _ln(tby["attn_norm"], 1028, 2560, 0)
    _gelu(tby["gelu"], 1028 * 10240, 0, "bf16")
    p2 = _by(build_rank_ops(m, proj, Phase("full", 2, wl.tokens, 0, pair=wl.pair), Shard()))
    _ln(p2["esm_s_ln"], 1024, 2560, proj)
    _relu(p2["esm_s_relu"], 1024 * 1024, proj, "bf16")


def test_esmfold_tp_sp_and_fp8_keep_released_weight_dtype():
    m, wl, ph = _phase(512)
    proj = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_proj")
    t, h, inter = 514, 2560, 10240
    sh = Shard(tp=2, sp=2)
    rows = -(-t // 2)
    loc = _by(build_rank_ops(m, 0, ph, sh))
    _gelu(loc["gelu"], rows * (inter // 2), 0, "bf16")
    _ln(loc["attn_norm"], rows, h, 0)
    _ln(loc["ffn_norm"], rows, h, 0)
    assert loc["q_scale"].vec == rows * -(-40 // 2) * (h // 40)
    assert loc["rope"].vec == rows * (-(-40 // 2) + -(-40 // 2)) * (h // 40) * 3
    assert loc["residual"].vec == rows * h * 2
    assert loc["mlp.up"].n == inter // 2 and loc["attn.q"].n == h // 2
    assert loc["attn_allreduce"].comm_bytes == rows * h * 2
    assert loc["attn.q"].w_fmt == "fp16" and loc["attn.q"].w_bits == 16.0
    last = _by(build_rank_ops(m, 35, ph, sh))
    _ln(last["emb_ln"], rows, h, 35)
    dap = _by(build_rank_ops(m, proj, ph, Shard(sp=2)))
    _ln(dap["esm_s_ln"], 512, 2560, proj)
    _relu(dap["esm_s_relu"], 512 * 1024, proj, "bf16")
    assert dap["esm_s_ln"].replicated == 2 and dap["esm_proj.mlp.1"].replicated == 2
    assert dap["esm_proj.mlp.1"].m == 512
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, ph, Shard()))
    _gelu(op["gelu"], t * inter, 0, "fp8")
    _ln(op["attn_norm"], t, h, 0)
    assert op["q_scale"].a_fmt == "fp8" and op["rope"].a_fmt == "bf16"
    assert op["attn.q"].a_fmt == "fp8" and op["attn.q"].w_fmt == "fp16" and op["attn.q"].w_bits == 16.0
    pr = _by(build_rank_ops(fp8, proj, ph, Shard()))
    _relu(pr["esm_s_relu"], 512 * 1024, proj, "fp8")
    _ln(pr["esm_s_ln"], 512, 2560, proj)
    assert pr["esm_proj.mlp.1"].w_fmt == "fp32" and pr["esm_proj.mlp.1"].a_fmt == "fp8"


def test_other_structure_models_keep_their_tails():
    for mid in ("openfold", "alphafold2", "boltz-1", "protenix", "esm2-3b", "esm2-650m"):
        m = get_model(mid)
        assert all(L.vec_kind != "esm_proj" for L in m.layers)
        assert all(not L.emb_ln for L in m.layers)
    esm = get_model("esm2-3b")
    assert esm.vec_kind == "esm2" and esm.hidden == 2560 and esm.layers[0].vec_width == 0
    by = _by(build_rank_ops(esm, 0, Phase("full", 1, 514, 0), Shard()))
    _ln(by["attn_norm"], 514, 2560, 0)
    _gelu(by["gelu"], 514 * 10240, 0, "bf16")
    assert "emb_ln" not in by


def test_esmfold_config_mismatch_is_refused():
    _raises(_bad(lambda r: r["config"].__setitem__("hidden_act", "silu")), "hidden_act is not gelu")
    _raises(_bad(lambda r: r["config"].__setitem__("hidden_act", "gelu_new")), "hidden_act is not gelu")
    _raises(_bad(lambda r: r["config"].__setitem__("hidden_act", "quick_gelu")), "hidden_act is not gelu")
    _raises(_bad(lambda r: r["config"].__setitem__("position_embedding_type", "absolute")),
            "position_embedding_type is not rotary")
    _raises(_bad(lambda r: r["config"].__setitem__("emb_layer_norm_before", True)),
            "emb_layer_norm_before is not false")
    _raises(_bad(lambda r: r["config"].__setitem__("is_folding_model", False)),
            "is_folding_model is not true")
    _raises(_bad(lambda r: r["config"].__setitem__("attention_probs_dropout_prob", 0.1)),
            "dropout is not 0")
    _raises(_bad(lambda r: r["config"]["esmfold_config"].__setitem__("bypass_lm", True)),
            "bypass_lm is not false")
    _raises(_bad(lambda r: r["config"]["esmfold_config"].__setitem__("use_esm_attn_map", True)),
            "use_esm_attn_map is not false")
    _raises(_bad(lambda r: r["config"]["esmfold_config"].__setitem__("fp16_esm", True)),
            "fp16_esm is not false")
    _raises(_bad(lambda r: r["config"]["esmfold_config"].__setitem__("esm_type", "esm2_650M")),
            "esm_type is not esm2_3B")
    _raises(_bad(lambda r: r["shapes"].__setitem__("esm_s_mlp.1.weight", [1024, 1024])),
            "esm_s_mlp.1.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__("esm.encoder.layer.0.q_norm.weight", [64])),
            "has q_norm")
    _raises(_bad(lambda r: r["shapes"].__setitem__("lm_head.dense.weight", [2560, 2560])),
            "ESM-2 MLM head")
