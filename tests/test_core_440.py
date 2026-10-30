"""0.440: ESMFold trunk, recycle, structure module, and lddt bill published LayerNorm and ReLU.

TriangularSelfAttentionBlock: 11 LayerNorms and 2 ReLUs per block.
EsmFoldingTrunk: recycle_s_norm and recycle_z_norm once per recycle, then trunk2sm.
StructureModule._forward_monomer: layer_norm_s, layer_norm_z, linear_in once;
the block loop has layer_norm_ipa, transition ReLUs and LayerNorm, and angle-resnet ReLUs.
lddt_head is LayerNorm plus three linears, no ReLU, on structure['states'] (num_blocks copies,
last recycle only). Vector flops stay 0. The generic .ew tail stays. Sigmoid gates, IPA
softplus, and the angle clamp are not billed.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, Shard, build_rank_ops, full_io_ops, layer_repeat
from accel_dse.core.model import NativeWorkload, load_release
from accel_dse.core.scenario import Workload
from accel_dse.core.structure import build_esmfold


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _ln(op, rows: int, width: int, layer: int, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _relu(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


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


_TRUNK_LN = (
    "pair_to_seq_ln", "seq_ln", "mlp_seq_ln", "seq_to_pair_ln",
    "tri_mul_out_ln_in", "tri_mul_out_ln_out", "tri_mul_in_ln_in", "tri_mul_in_ln_out",
    "tri_att_start_ln", "tri_att_end_ln", "mlp_pair_ln",
)
_ANGLE_RELU = (
    "angle_init_relu", "angle_in_relu",
    "angle_b0_relu_1", "angle_b0_relu_2", "angle_b1_relu_1", "angle_b1_relu_2",
    "angle_out_relu",
)


def test_version_is_440():
    assert __version__ == "0.481.0"


def test_esmfold_trunk_bills_eleven_layernorms_and_two_relus():
    m, wl, ph = _phase(512)
    assert m.vec_kind == "" and m.hidden == 1024 and m.n_layers == 94
    assert m.params() == m.release_params == 3527665186 and m.param_check()["rel_err"] == 0.0
    assert wl.pair.n == 512 and wl.pair.recycles == 4
    trunks = [i for i, L in enumerate(m.layers) if L.vec_kind == "esm_trunk"]
    assert trunks == list(range(36, 84))
    r, p, cs, cz = 512, 512 * 512, 1024, 128
    for li in (trunks[0], trunks[-1]):
        layer = m.layers[li]
        assert layer.repeat == "recycle" and layer.iters == 1 and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        for name in ("seq_ln", "mlp_seq_ln", "seq_to_pair_ln"):
            _ln(by[name], r, cs, li)
        for name in ("pair_to_seq_ln", "tri_mul_out_ln_in", "tri_mul_out_ln_out", "tri_mul_in_ln_in",
                     "tri_mul_in_ln_out", "tri_att_start_ln", "tri_att_end_ln", "mlp_pair_ln"):
            _ln(by[name], p, cz, li)
        _relu(by["mlp_seq_relu"], r * 4 * cs, li, "bf16")
        _relu(by["mlp_pair_relu"], p * 4 * cz, li, "bf16")
        assert by["trunk.mlp_seq.mlp.1.ew"].vec == r * (cs + 4 * cs) * 4
        assert by["trunk.mlp_pair.mlp.1.ew"].vec == p * (cz + 4 * cz) * 4
        assert (by["trunk.seq_attention.proj"].m, by["trunk.seq_attention.proj"].k,
                by["trunk.seq_attention.proj"].n) == (r, cs, 3 * cs)
        assert by["trunk.seq_att0.qk"].m == r
        gemms = [o.name for o in ops if o.kind == "gemm"]
        assert gemms == [f"trunk.{ln.name}" for ln in layer.pair_linears]
        names = [o.name for o in ops]
        assert names.index("pair_to_seq_ln") < names.index("trunk.pair_to_sequence.linear")
        assert names.index("seq_ln") < names.index("trunk.seq_attention.proj")
        assert names.index("mlp_seq_ln") < names.index("trunk.mlp_seq.mlp.1") < names.index("mlp_seq_relu")
        assert names.index("mlp_seq_relu") < names.index("trunk.mlp_seq.mlp.1.ew") < names.index("trunk.mlp_seq.mlp.3")
        assert names.index("tri_mul_out_ln_in") < names.index("trunk.tri_mul_out.linear_a_g")
        assert names.index("trunk.tri_mul_out.linear_a_g") < names.index("tri_mul_out_ln_out")
        assert names.index("tri_mul_out_ln_out") < names.index("trunk.tri_mul_out.linear_z")
        assert names.index("mlp_pair_ln") < names.index("trunk.mlp_pair.mlp.1") < names.index("mlp_pair_relu")
        banned = ("gelu", "silu", "quick_gelu", "gelu_new", "sigmoid", "softplus", "clamp")
        assert all(o.name not in banned for o in ops)
        assert [n for n in names if n in _TRUNK_LN] and len([n for n in names if n in _TRUNK_LN]) == 11
    lang = _by(build_rank_ops(m, 0, ph, Shard()))
    assert lang["gelu"].vec == 514 * 10240 * 5 and lang["attn_norm"].vec == 514 * 2560 * 4
    assert "emb_ln" not in lang
    last = _by(build_rank_ops(m, 35, ph, Shard()))
    assert last["emb_ln"].vec == 514 * 2560 * 4
    proj = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_proj")
    pr = _by(build_rank_ops(m, proj, ph, Shard()))
    assert pr["esm_s_ln"].vec == 512 * 2560 * 4 and pr["esm_s_relu"].vec == 512 * 1024
    post = _by(full_io_ops(m, ph, Shard(), "post"))
    assert post["final_norm"].layer == m.n_layers and "lm_gelu" not in post


def test_esmfold_structure_recycle_and_lddt():
    m, wl, ph = _phase(512)
    r, p = 512, 512 * 512
    pre_i = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_sm_pre")
    sm_i = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_sm")
    rec_i = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_recycle")
    ld_i = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_lddt")
    assert (pre_i, sm_i, rec_i, ld_i) == (84, 85, 86, 91)
    pre, sm, rec, ld = m.layers[pre_i], m.layers[sm_i], m.layers[rec_i], m.layers[ld_i]
    assert pre.iters == 1 and pre.repeat == "recycle" and pre.vec_width == 128 and pre.misc_params == 0
    assert layer_repeat(pre, ph) == 4 and pre.pair_cores == ()
    assert sm.iters == 8 and sm.repeat == "recycle" and layer_repeat(sm, ph) == 32 and sm.misc_params == 6444
    assert sm.pair_cores and sm.pair_cores[0].heads == 12
    assert rec.repeat == "recycle" and rec.iters == 1 and layer_repeat(rec, ph) == 4
    assert ld.iters == 1 and ld.repeat == "" and layer_repeat(ld, ph) == 1
    pops = build_rank_ops(m, pre_i, ph, Shard())
    pb = _by(pops)
    _ln(pb["sm_s_ln"], r, 384, pre_i)
    _ln(pb["sm_z_ln"], p, 128, pre_i)
    lin = pb["structure.linear_in"]
    assert (lin.m, lin.k, lin.n) == (r, 384, 384) and lin.w_fmt == "fp32" and lin.w_bits == 32.0
    assert [o.name for o in pops].index("sm_s_ln") < [o.name for o in pops].index("sm_z_ln")
    assert [o.name for o in pops].index("sm_z_ln") < [o.name for o in pops].index("structure.linear_in")
    assert not any("relu" in o.name for o in pops)
    sops = build_rank_ops(m, sm_i, ph, Shard())
    sb = _by(sops)
    _ln(sb["ipa_ln"], r, 384, sm_i)
    _ln(sb["trans_ln"], r, 384, sm_i)
    for name in ("trans_relu_1", "trans_relu_2", "angle_init_relu", "angle_in_relu"):
        _relu(sb[name], r * 384, sm_i, "bf16")
    for name in ("angle_b0_relu_1", "angle_b0_relu_2", "angle_b1_relu_1", "angle_b1_relu_2", "angle_out_relu"):
        _relu(sb[name], r * 128, sm_i, "bf16")
    assert "structure.linear_in" not in sb
    assert sb["structure.ipa.linear_out"].k == 2112 and sb["structure.ipa.linear_out"].n == 384
    assert sb["structure.ipa.linear_out"].w_fmt == "fp32"
    assert sb["structure.angle_resnet.linear_out"].n == 14
    snames = [o.name for o in sops]
    assert snames.index("angle_init_relu") < snames.index("structure.angle_resnet.linear_initial")
    assert snames.index("angle_in_relu") < snames.index("structure.angle_resnet.linear_in")
    assert snames.index("structure.ipa.linear_out") < snames.index("ipa_ln")
    assert snames.index("structure.transition.layers.0.linear_1") < snames.index("trans_relu_1")
    assert snames.index("trans_relu_1") < snames.index("structure.transition.layers.0.linear_1.ew")
    assert snames.index("structure.transition.layers.0.linear_3") < snames.index("trans_ln")
    assert len([n for n in snames if n in _ANGLE_RELU or n.startswith("trans_relu")]) == 9
    rops = build_rank_ops(m, rec_i, ph, Shard())
    rb = _by(rops)
    _ln(rb["recycle_s_ln"], r, 1024, rec_i)
    _ln(rb["recycle_z_ln"], p, 128, rec_i)
    assert (rb["recycle.trunk2sm_s"].m, rb["recycle.trunk2sm_s"].k, rb["recycle.trunk2sm_s"].n) == (r, 1024, 384)
    assert (rb["recycle.trunk2sm_z"].m, rb["recycle.trunk2sm_z"].k, rb["recycle.trunk2sm_z"].n) == (p, 128, 128)
    rnames = [o.name for o in rops]
    assert rnames.index("recycle_s_ln") < rnames.index("recycle_z_ln") < rnames.index("recycle.trunk2sm_s")
    assert rnames.index("recycle.trunk2sm_s") < rnames.index("recycle.trunk2sm_z")
    lops = build_rank_ops(m, ld_i, ph, Shard())
    lb = _by(lops)
    _ln(lb["lddt_ln"], 8 * r, 384, ld_i)
    assert (lb["heads.1"].m, lb["heads.1"].k, lb["heads.1"].n) == (8 * r, 384, 128)
    assert (lb["heads.2"].m, lb["heads.2"].k, lb["heads.2"].n) == (8 * r, 128, 128)
    assert (lb["heads.3"].m, lb["heads.3"].k, lb["heads.3"].n) == (8 * r, 128, 1850)
    assert lb["heads.1"].w_fmt == "fp32" and lb["heads.1.ew"].vec == (8 * r) * (384 + 128) * 4
    assert not any("relu" in o.name for o in lops)
    assert [o.name for o in lops].index("lddt_ln") < [o.name for o in lops].index("heads.1")
    short_m, short_wl, short = _phase(128)
    assert short_wl.tokens == 130 and short_wl.pair.n == 128
    sby = _by(build_rank_ops(short_m, trunks_first(short_m), short, Shard()))
    _ln(sby["seq_ln"], 128, 1024, trunks_first(short_m))
    _ln(sby["mlp_pair_ln"], 128 * 128, 128, trunks_first(short_m))
    ldb = _by(build_rank_ops(short_m, ld_i, short, Shard()))
    _ln(ldb["lddt_ln"], 8 * 128, 384, ld_i)
    assert ldb["heads.3"].m == 8 * 128


def trunks_first(m) -> int:
    return next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_trunk")


def test_esmfold_batch_dap_and_fp8():
    m, wl, ph = _phase(512)
    trunk = trunks_first(m)
    pre = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_sm_pre")
    sm = next(i for i, L in enumerate(m.layers) if L.vec_kind == "esm_sm")
    two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
    b2 = _by(build_rank_ops(m, trunk, two, Shard()))
    _ln(b2["seq_ln"], 1024, 1024, trunk)
    _ln(b2["mlp_pair_ln"], 2 * 512 * 512, 128, trunk)
    _relu(b2["mlp_seq_relu"], 1024 * 4096, trunk, "bf16")
    dap = _by(build_rank_ops(m, trunk, ph, Shard(sp=2)))
    _ln(dap["seq_ln"], 512, 1024, trunk, rep=2)
    _ln(dap["mlp_pair_ln"], 256 * 512, 128, trunk, rep=1)
    _relu(dap["mlp_seq_relu"], 512 * 4096, trunk, "bf16", rep=2)
    _relu(dap["mlp_pair_relu"], (256 * 512) * 512, trunk, "bf16", rep=1)
    assert dap["trunk.mlp_seq.mlp.1"].m == 512 and dap["trunk.mlp_seq.mlp.1"].replicated == 2
    assert dap["trunk.mlp_pair.mlp.1"].m == 256 * 512 and dap["trunk.mlp_pair.mlp.1"].replicated == 1
    dpre = _by(build_rank_ops(m, pre, ph, Shard(sp=2)))
    _ln(dpre["sm_s_ln"], 512, 384, pre, rep=2)
    _ln(dpre["sm_z_ln"], 256 * 512, 128, pre, rep=1)
    assert dpre["structure.linear_in"].m == 512 and dpre["structure.linear_in"].replicated == 2
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, trunk, ph, Shard()))
    _relu(op["mlp_seq_relu"], 512 * 4096, trunk, "fp8")
    _relu(op["mlp_pair_relu"], (512 * 512) * 512, trunk, "fp8")
    _ln(op["seq_ln"], 512, 1024, trunk)
    _ln(op["mlp_pair_ln"], 512 * 512, 128, trunk)
    assert op["trunk.mlp_seq.mlp.1"].a_fmt == "fp8" and op["trunk.mlp_seq.mlp.1"].w_fmt == "fp32"
    sop = _by(build_rank_ops(fp8, sm, ph, Shard()))
    _relu(sop["trans_relu_1"], 512 * 384, sm, "fp8")
    _ln(sop["ipa_ln"], 512, 384, sm)
    assert sop["structure.ipa.linear_q"].w_fmt == "fp32" and sop["structure.ipa.linear_q"].a_fmt == "fp8"


def test_other_models_do_not_gain_esmfold_trunk_vectors():
    for mid in ("openfold", "alphafold2", "boltz-1", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert all(L.vec_kind not in ("esm_trunk", "esm_sm", "esm_sm_pre", "esm_recycle", "esm_lddt")
                   for L in other.layers)
    esm = get_model("esm2-3b")
    by = _by(build_rank_ops(esm, 0, Phase("full", 1, 514, 0), Shard()))
    assert "seq_ln" not in by and by["gelu"].vec == 514 * 10240 * 5


def test_esmfold_fold_config_mismatch_is_refused():
    trunk = ["config", "esmfold_config", "trunk"]
    sm = trunk + ["structure_module"]

    def set_path(rel, path, value):
        node = rel
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    _raises(_bad(lambda r: set_path(r, trunk + ["dropout"], 0.1)), "trunk dropout is not 0")
    _raises(_bad(lambda r: set_path(r, trunk + ["layer_drop"], 0.1)), "trunk dropout is not 0")
    _raises(_bad(lambda r: set_path(r, sm + ["dropout_rate"], 0.0)), "structure dropout_rate is not 0.1")
    _raises(_bad(lambda r: set_path(r, sm + ["num_transition_layers"], 2)), "num_transition_layers is not 1")
    _raises(_bad(lambda r: set_path(r, sm + ["num_resnet_blocks"], 1)), "num_resnet_blocks is not 2")
    _raises(_bad(lambda r: set_path(r, sm + ["num_angles"], 6)), "num_angles is not 7")
    _raises(_bad(lambda r: r["shapes"].__setitem__("trunk.blocks.0.layernorm_1.weight", [512])),
            "layernorm_1.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__("trunk.blocks.47.mlp_pair.mlp.0.weight", [64])),
            "mlp_pair.mlp.0.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].pop("trunk.blocks.0.pair_to_sequence.layernorm.weight")),
            "pair_to_sequence.layernorm.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__("lddt_head.3.weight", [128, 128])),
            "lddt_head.3.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__("lddt_head.4.weight", [128, 128])),
            "lddt head has a fourth linear")
    _raises(_bad(lambda r: set_path(r, trunk + ["sequence_head_width"], 3)),
            "trunk width does not divide")


def test_closed_form_flops_still_hold():
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario
    for n, ref in ((384, 50.24e12), (512, 97.32e12)):
        got = evaluate(Scenario(model="esmfold", mem_id="hbm3e_8s_12h24g_9200",
                                workload=Workload(seq_len=n))).domain_summary()["tflop_per_request"] * 1e12
        assert abs(got / ref - 1) < 1e-3, (n, got)
