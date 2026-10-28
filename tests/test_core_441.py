"""0.441: OpenFold and AlphaFold 2 bill published LayerNorm and ReLU.

Evoformer and extra MSA: 12 LayerNorms and 2 ReLUs per block.
Template pair block: 7 LayerNorms and 1 ReLU.
Structure module: layer norms and the input projection once per recycle;
the IPA loop keeps 8 iterations of LayerNorm and ReLU.
pLDDT is LayerNorm, ReLU, Linear, ReLU, Linear on the final single, once.
Recycling norms, the template-angle ReLU, and the template-stack LayerNorm
run once per recycle. Vector flops stay 0. The generic .ew tail stays.
Mask multiplies, sigmoid gates, IPA softplus, and the angle clamp are not billed.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, Shard, build_rank_ops, layer_repeat
from accel_dse.core.model import NativeWorkload, load_release
from accel_dse.core.scenario import Workload
from accel_dse.core.structure import build_alphafold2, build_openfold


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


def _phase(mid: str, seq_len: int, batch: int = 1):
    m = get_model(mid)
    wl = resolve_workload(m, Workload(seq_len=seq_len))
    ph = Phase("full", batch, wl.tokens, 0, pair=wl.pair)
    return m, wl, ph


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(who: str, edit):
    hf = "gh:aqlaboratory/openfold" if who == "openfold" else "gh:google-deepmind/alphafold"
    rel = json.loads(json.dumps(load_release(hf)))
    edit(rel)
    wl = NativeWorkload("protein", seq_len=8)
    build = build_openfold if who == "openfold" else build_alphafold2
    return lambda: build(who, hf, rel, wl)


_BLOCK_LN = (
    "msa_trans_ln", "opm_ln", "pair_trans_ln", "tri_att_end_ln", "tri_att_start_ln",
    "tri_mul_in_ln_in", "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out",
    "msa_col_ln", "msa_row_z_ln", "msa_row_m_ln",
)
_TMPL_LN = (
    "pair_trans_ln", "tri_att_end_ln", "tri_att_start_ln",
    "tri_mul_in_ln_in", "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out",
)
_ANGLE = (
    "angle_init_relu", "angle_in_relu",
    "angle_b0_relu_1", "angle_b0_relu_2", "angle_b1_relu_1", "angle_b1_relu_2",
    "angle_out_relu",
)


def _idx(mid: str, kind: str, stack: str | None = None) -> int:
    m = get_model(mid)
    return next(i for i, L in enumerate(m.layers) if L.vec_kind == kind and (stack is None or L.stack == stack))


def _gemm(layer, name: str) -> str:
    return f"{layer.stack}.{name}"


def _touch(ops, gemm: str, before: tuple[str, ...] = (), after: str | None = None) -> None:
    names = [o.name for o in ops]
    i = names.index(gemm)
    for k, want in enumerate(before):
        assert names[i - len(before) + k] == want, names[i - 2:i + 3]
    if after:
        assert names[i + 1] == after and names[i + 2] == gemm + ".ew"
    else:
        assert names[i + 1] == gemm + ".ew"


def test_version_is_441():
    assert __version__ == "0.446.0"


def test_openfold_and_alphafold2_block_layernorm_and_relu():
    for mid, msa_rows, xmsa_rows in (("openfold", 516 * 512, 1024 * 512), ("alphafold2", 512 * 512, 5120 * 512)):
        m, wl, ph = _phase(mid, 512)
        assert m.n_layers == 58 and m.params() == m.release_params == 93_237_338
        assert m.hidden == 384 and wl.pair.n == 512 and wl.pair.recycles == 4
        assert wl.pair.msa == msa_rows // 512 and wl.pair.xmsa == xmsa_rows // 512
        evo_i = [i for i, L in enumerate(m.layers) if L.stack == "evoformer"]
        extra_i = [i for i, L in enumerate(m.layers) if L.stack == "extra_msa"]
        tmpl_i = [i for i, L in enumerate(m.layers) if L.stack == "template"]
        assert evo_i == list(range(6, 54)) and extra_i == list(range(2, 6)) and tmpl_i == [0, 1]
        r, p = 512, 512 * 512
        for li in (evo_i[0], evo_i[-1]):
            layer = m.layers[li]
            assert layer.vec_kind == "of_block" and layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
            ops = build_rank_ops(m, li, ph, Shard())
            by = _by(ops)
            prefix = "core." if mid == "openfold" else ""
            for name in ("msa_trans_ln", "opm_ln", "msa_col_ln", "msa_row_m_ln"):
                _ln(by[name], msa_rows, 256, li)
            for name in ("pair_trans_ln", "tri_att_end_ln", "tri_att_start_ln", "tri_mul_in_ln_in",
                         "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out", "msa_row_z_ln"):
                _ln(by[name], p, 128, li)
            _relu(by["msa_trans_relu"], msa_rows * 1024, li, "bf16")
            _relu(by["pair_trans_relu"], p * 512, li, "bf16")
            msa_gemm = _gemm(layer, f"{prefix}msa_transition.linear_1")
            pair_gemm = _gemm(layer, f"{prefix}pair_transition.linear_1")
            assert by[msa_gemm].w_fmt == "fp32" and by[msa_gemm].w_bits == 32.0
            assert by[msa_gemm].flops == 2 * msa_rows * 256 * 1024
            assert by[msa_gemm + ".ew"].vec == msa_rows * (256 + 1024) * 4
            assert by[pair_gemm + ".ew"].vec == p * (128 + 512) * 4
            _touch(ops, msa_gemm, ("msa_trans_ln",), "msa_trans_relu")
            _touch(ops, pair_gemm, ("pair_trans_ln",), "pair_trans_relu")
            _touch(ops, _gemm(layer, f"{prefix}tri_mul_out.linear_a_g"), ("tri_mul_out_ln_in",))
            _touch(ops, _gemm(layer, f"{prefix}tri_mul_out.linear_z"), ("tri_mul_out_ln_out",))
            col = "msa_att_col._msa_att.mha.linear_g"
            _touch(ops, _gemm(layer, col), ("msa_col_ln",))
            assert len([n for n in (o.name for o in ops) if n in _BLOCK_LN]) == 12
            assert len([o for o in ops if o.name in ("msa_trans_relu", "pair_trans_relu")]) == 2
            banned = ("gelu", "silu", "sigmoid", "softplus", "clamp", "mask_mul")
            assert all(o.name not in banned for o in ops)
        for li in (extra_i[0], extra_i[-1]):
            layer = m.layers[li]
            ops = build_rank_ops(m, li, ph, Shard())
            by = _by(ops)
            prefix = "core." if mid == "openfold" else ""
            for name in ("msa_trans_ln", "opm_ln", "msa_col_ln", "msa_row_m_ln"):
                _ln(by[name], xmsa_rows, 64, li)
            _relu(by["msa_trans_relu"], xmsa_rows * 256, li, "bf16")
            _ln(by["pair_trans_ln"], p, 128, li)
            _ln(by["msa_row_z_ln"], p, 128, li)
            gate = _gemm(layer, "msa_att_col.global_attention.linear_g")
            query = _gemm(layer, "msa_att_col.global_attention.linear_q")
            _touch(ops, gate, ("msa_col_ln",))
            _touch(ops, query)
            assert by[query].m == r and by[query].replicated == 1
            assert by["msa_col_ln"].sfu == float(xmsa_rows)
            assert by[_gemm(layer, f"{prefix}msa_transition.linear_1") + ".ew"].vec == xmsa_rows * (64 + 256) * 4
        for li in tmpl_i:
            layer = m.layers[li]
            ops = build_rank_ops(m, li, ph, Shard())
            by = _by(ops)
            tmpl = 4 * p
            for name in _TMPL_LN:
                _ln(by[name], tmpl, 64, li)
            _relu(by["pair_trans_relu"], tmpl * 128, li, "bf16")
            assert "msa_trans_ln" not in by and "msa_trans_relu" not in by
            _touch(ops, _gemm(layer, "pair_transition.linear_1"), ("pair_trans_ln",), "pair_trans_relu")
            _touch(ops, _gemm(layer, "tri_mul_in.linear_z"), ("tri_mul_in_ln_out",))
            assert by[_gemm(layer, "pair_transition.linear_1") + ".ew"].vec == tmpl * (64 + 128) * 4
            assert len([n for n in (o.name for o in ops) if n in _TMPL_LN]) == 7


def test_structure_preamble_loop_heads_and_embed():
    r, p = 512, 512 * 512
    for mid, pre_gemm, plddt in (
        ("openfold", "structure.linear_in", ("heads.plddt.linear_1", "heads.plddt.linear_2", "heads.plddt.linear_3")),
        ("alphafold2", "structure.initial_projection", ("heads.plddt.act_0", "heads.plddt.act_1", "heads.plddt.logits")),
    ):
        m, wl, ph = _phase(mid, 512)
        pre_i, sm_i = _idx(mid, "of_sm_pre"), _idx(mid, "of_sm")
        head_i, emb_i = _idx(mid, "of_heads"), _idx(mid, "of_embed")
        assert (pre_i, sm_i, head_i, emb_i) == (54, 55, 56, 57)
        pre, sm, head, emb = m.layers[pre_i], m.layers[sm_i], m.layers[head_i], m.layers[emb_i]
        assert pre.iters == 1 and pre.repeat == "recycle" and pre.vec_width == 128 and pre.misc_params == 0
        assert pre.pair_cores == () and layer_repeat(pre, ph) == 4
        assert sm.iters == 8 and sm.repeat == "recycle" and sm.misc_params == 6444
        assert layer_repeat(sm, ph) == 32 and sm.pair_cores and sm.pair_cores[0].heads == 12
        assert head.iters == 1 and head.repeat == "" and layer_repeat(head, ph) == 1 and head.misc_params == 1262
        assert emb.repeat == "recycle" and layer_repeat(emb, ph) == 4 and emb.misc_params == 3072
        pops = build_rank_ops(m, pre_i, ph, Shard())
        pb = _by(pops)
        _ln(pb["sm_s_ln"], r, 384, pre_i)
        _ln(pb["sm_z_ln"], p, 128, pre_i)
        lin = pb[pre_gemm]
        assert (lin.m, lin.k, lin.n) == (r, 384, 384) and lin.w_fmt == "fp32" and lin.flops == 2 * r * 384 * 384
        _touch(pops, pre_gemm, ("sm_s_ln", "sm_z_ln"))
        assert not any("relu" in o.name for o in pops)
        sops = build_rank_ops(m, sm_i, ph, Shard())
        sb = _by(sops)
        assert pre_gemm not in sb
        _ln(sb["ipa_ln"], r, 384, sm_i)
        _ln(sb["trans_ln"], r, 384, sm_i)
        for name in ("trans_relu_1", "trans_relu_2", "angle_init_relu", "angle_in_relu"):
            _relu(sb[name], r * 384, sm_i, "bf16")
        for name in ("angle_b0_relu_1", "angle_b0_relu_2", "angle_b1_relu_1", "angle_b1_relu_2", "angle_out_relu"):
            _relu(sb[name], r * 128, sm_i, "bf16")
        assert sb["structure.ipa.linear_out"].k == 2112 and sb["structure.ipa.linear_out"].n == 384
        assert sb["structure.ipa.linear_out"].w_fmt == "fp32"
        if mid == "openfold":
            _touch(sops, "structure.angle_resnet.linear_initial", ("angle_init_relu",))
            _touch(sops, "structure.angle_resnet.linear_in", ("angle_in_relu",))
            _touch(sops, "structure.angle_resnet.linear_out", ("angle_out_relu",))
            _touch(sops, "structure.transition.layers.0.linear_1", (), "trans_relu_1")
            names = [o.name for o in sops]
            assert names.index("structure.transition.layers.0.linear_3") < names.index("trans_ln") < \
                names.index("structure.transition.layers.0.linear_3.ew")
            assert names.index("structure.ipa.linear_out") < names.index("ipa_ln") < \
                names.index("structure.ipa.linear_out.ew")
        else:
            _touch(sops, "structure.rigid_sidechain.input_projection", ("angle_in_relu",))
            _touch(sops, "structure.rigid_sidechain.input_projection_1", ("angle_init_relu",))
            _touch(sops, "structure.rigid_sidechain.unnormalized_angles", ("angle_out_relu",))
            _touch(sops, "structure.transition", (), "trans_relu_1")
            _touch(sops, "structure.transition_1", (), "trans_relu_2")
            names = [o.name for o in sops]
            assert names.index("structure.transition_2") < names.index("trans_ln") < names.index("structure.transition_2.ew")
        assert len([n for n in (o.name for o in sops) if n in _ANGLE or n.startswith("trans_relu")]) == 9
        hops = build_rank_ops(m, head_i, ph, Shard())
        hb = _by(hops)
        _ln(hb["plddt_ln"], r, 384, head_i)
        _relu(hb["plddt_relu_1"], r * 128, head_i, "bf16")
        _relu(hb["plddt_relu_2"], r * 128, head_i, "bf16")
        assert (hb[plddt[0]].m, hb[plddt[0]].k, hb[plddt[0]].n) == (r, 384, 128)
        assert (hb[plddt[1]].m, hb[plddt[1]].k, hb[plddt[1]].n) == (r, 128, 128)
        assert (hb[plddt[2]].m, hb[plddt[2]].k, hb[plddt[2]].n) == (r, 128, 50)
        assert hb[plddt[0]].m != 8 * r
        _touch(hops, plddt[0], ("plddt_ln",), "plddt_relu_1")
        _touch(hops, plddt[1], (), "plddt_relu_2")
        _touch(hops, plddt[2])
        assert hb["heads.distogram.linear"].m == p and "relu" not in "".join(
            o.name for o in hops if o.name.startswith("heads.distogram"))
        eops = build_rank_ops(m, emb_i, ph, Shard())
        eb = _by(eops)
        _ln(eb["recycle_m_ln"], r, 256, emb_i)
        _ln(eb["recycle_z_ln"], p, 128, emb_i)
        _relu(eb["tmpl_angle_relu"], (4 * r) * 256, emb_i, "bf16")
        _ln(eb["tmpl_out_ln"], 4 * p, 64, emb_i)
        _touch(eops, "embed.recycling_embedder.linear", ("recycle_m_ln", "recycle_z_ln"))
        _touch(eops, "embed.template_angle_embedder.linear_1", (), "tmpl_angle_relu")
        _touch(eops, "embed.template_angle_embedder.linear_2")
        _touch(eops, "embed.template_pointwise_att.mha.linear_k", ("tmpl_out_ln",))
        assert eb["embed.recycling_embedder.linear"].k == 15 and eb["embed.recycling_embedder.linear"].n == 128
        assert eb["embed.evoformer.linear"].m == r and eb["embed.evoformer.linear"].k == 256
    short_m, short_wl, short = _phase("openfold", 128)
    assert short_wl.pair.n == 128 and short_wl.pair.msa == 516
    evo = _idx("openfold", "of_block", "evoformer")
    sby = _by(build_rank_ops(short_m, evo, short, Shard()))
    _ln(sby["msa_trans_ln"], 516 * 128, 256, evo)
    _ln(sby["pair_trans_ln"], 128 * 128, 128, evo)
    pre = _idx("openfold", "of_sm_pre")
    pby = _by(build_rank_ops(short_m, pre, short, Shard()))
    _ln(pby["sm_s_ln"], 128, 384, pre)
    assert pby["structure.linear_in"].m == 128
    hby = _by(build_rank_ops(short_m, _idx("openfold", "of_heads"), short, Shard()))
    assert hby["heads.plddt.linear_3"].m == 128


def test_batch_dap_and_fp8():
    m, wl, ph = _phase("openfold", 512)
    evo = _idx("openfold", "of_block", "evoformer")
    extra = _idx("openfold", "of_block", "extra_msa")
    pre = _idx("openfold", "of_sm_pre")
    sm = _idx("openfold", "of_sm")
    emb = _idx("openfold", "of_embed")
    two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
    b2 = _by(build_rank_ops(m, evo, two, Shard()))
    _ln(b2["msa_trans_ln"], 2 * 516 * 512, 256, evo)
    _ln(b2["pair_trans_ln"], 2 * 512 * 512, 128, evo)
    _relu(b2["msa_trans_relu"], 2 * 516 * 512 * 1024, evo, "bf16")
    dap = _by(build_rank_ops(m, evo, ph, Shard(sp=2)))
    _ln(dap["msa_trans_ln"], 258 * 512, 256, evo, rep=1)
    _ln(dap["pair_trans_ln"], 256 * 512, 128, evo, rep=1)
    _relu(dap["pair_trans_relu"], (256 * 512) * 512, evo, "bf16", rep=1)
    assert dap["evoformer.core.msa_transition.linear_1"].m == 258 * 512
    assert dap["evoformer.core.msa_transition.linear_1"].replicated == 1
    dx = _by(build_rank_ops(m, extra, ph, Shard(sp=2)))
    _ln(dx["msa_col_ln"], 512 * 512, 64, extra, rep=1)
    assert dx["extra_msa.msa_att_col.global_attention.linear_q"].m == 512
    assert dx["extra_msa.msa_att_col.global_attention.linear_q"].replicated == 2
    dpre = _by(build_rank_ops(m, pre, ph, Shard(sp=2)))
    _ln(dpre["sm_s_ln"], 512, 384, pre, rep=2)
    _ln(dpre["sm_z_ln"], 256 * 512, 128, pre, rep=1)
    assert dpre["structure.linear_in"].m == 512 and dpre["structure.linear_in"].replicated == 2
    demb = _by(build_rank_ops(m, emb, ph, Shard(sp=2)))
    _ln(demb["recycle_m_ln"], 512, 256, emb, rep=2)
    _ln(demb["recycle_z_ln"], 256 * 512, 128, emb, rep=1)
    _relu(demb["tmpl_angle_relu"], (4 * 512) * 256, emb, "bf16", rep=2)
    _ln(demb["tmpl_out_ln"], 4 * 256 * 512, 64, emb, rep=1)
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, evo, ph, Shard()))
    _relu(op["msa_trans_relu"], 516 * 512 * 1024, evo, "fp8")
    _relu(op["pair_trans_relu"], (512 * 512) * 512, evo, "fp8")
    _ln(op["msa_trans_ln"], 516 * 512, 256, evo)
    assert op["evoformer.core.msa_transition.linear_1"].a_fmt == "fp8"
    assert op["evoformer.core.msa_transition.linear_1"].w_fmt == "fp32"
    sop = _by(build_rank_ops(fp8, sm, ph, Shard()))
    _relu(sop["trans_relu_1"], 512 * 384, sm, "fp8")
    _relu(sop["angle_out_relu"], 512 * 128, sm, "fp8")
    _ln(sop["ipa_ln"], 512, 384, sm)
    assert sop["structure.ipa.linear_q"].w_fmt == "fp32" and sop["structure.ipa.linear_q"].a_fmt == "fp8"
    af, af_wl, af_ph = _phase("alphafold2", 512)
    af_extra = _idx("alphafold2", "of_block", "extra_msa")
    ax = _by(build_rank_ops(af, af_extra, af_ph, Shard(sp=2)))
    _ln(ax["msa_col_ln"], 2560 * 512, 64, af_extra, rep=1)
    assert ax["extra_msa.msa_att_col.global_attention.linear_q"].m == 512
    assert ax["extra_msa.msa_att_col.global_attention.linear_q"].replicated == 2


def test_other_models_do_not_gain_openfold_vectors():
    for mid in ("boltz-1", "protenix", "esmfold", "esm2-3b"):
        other = get_model(mid)
        assert all(not L.vec_kind.startswith("of_") for L in other.layers)
    esm = get_model("esmfold")
    kinds = {L.vec_kind for L in esm.layers}
    assert {"esm_trunk", "esm_sm", "esm_sm_pre", "esm_recycle", "esm_lddt"} <= kinds
    of = get_model("openfold")
    assert all(L.vec_kind not in ("esm_trunk", "esm_sm", "esm_sm_pre", "esm_recycle", "esm_lddt") for L in of.layers)
    af = get_model("alphafold2")
    sig = lambda L: (L.stack, L.repeat, L.iters, sorted((l.k, l.n, l.rows) for l in L.pair_linears),
                     sorted(map(repr, L.pair_cores)), L.misc_params)
    assert [sig(L) for L in af.layers] == [sig(L) for L in of.layers]


def test_published_shape_mismatch_is_refused():
    _raises(_bad("openfold", lambda r: r["shapes"].pop("evoformer.blocks.0.core.msa_transition.layer_norm.weight")),
            "evoformer.blocks.0.core.msa_transition.layer_norm.weight shape is not")
    _raises(_bad("openfold", lambda r: r["shapes"].__setitem__(
        "evoformer.blocks.47.core.tri_mul_out.layer_norm_out.weight", [64])),
            "evoformer.blocks.47.core.tri_mul_out.layer_norm_out.weight shape is not")
    _raises(_bad("openfold", lambda r: r["shapes"].pop("evoformer.blocks.0.core.msa_transition.linear_1.weight")),
            "block linears are not the published set")
    _raises(_bad("openfold", lambda r: r["shapes"].__setitem__("aux_heads.plddt.linear_4.weight", [50, 128])),
            "plddt head has a fourth linear")
    _raises(_bad("openfold", lambda r: r["shapes"].__setitem__("structure_module.linear_in.weight", [128, 384])),
            "linear_in shape is not [384, 384] on res")
    _raises(_bad("openfold", lambda r: r["config"].__setitem__("no_ipa_blocks", 4)), "ipa blocks are not 8")
    _raises(_bad("openfold", lambda r: r["shapes"].__setitem__(
        "template_pair_stack.blocks.0.tri_mul_in.linear_z.weight", [64, 32])),
            "triangle multiply hidden width is not the projection width")
    _raises(_bad("alphafold2", lambda r: r["shapes"].__setitem__(
        "evoformer.blocks.0.msa_transition.input_layer_norm.scale", [64])),
            "evoformer.blocks.0.msa_transition.input_layer_norm.scale shape is not")
    _raises(_bad("alphafold2", lambda r: r["shapes"].__setitem__("aux_heads.plddt.act_2.weight", [128, 128])),
            "plddt head has a fourth linear")
    _raises(_bad("alphafold2", lambda r: r["shapes"].__setitem__(
        "structure_module.single_layer_norm.scale", [128])),
            "structure_module.single_layer_norm.scale shape is not")
    _raises(_bad("alphafold2", lambda r: r["shapes"].pop("extra_msa_stack.blocks.3.msa_att_col.query_norm.scale")),
            "extra_msa_stack.blocks.3.msa_att_col.query_norm.scale shape is not")

    m, wl, ph = _phase("openfold", 32)
    evo = _idx("openfold", "of_block", "evoformer")
    bad_layer = replace(m.layers[evo], pair_linears=m.layers[evo].pair_linears[:-1])
    broken = replace(m, layers=m.layers[:evo] + (bad_layer,) + m.layers[evo + 1:])
    _raises(lambda: build_rank_ops(broken, evo, ph, Shard()), "openfold block linears are not the published set")
    pre = _idx("alphafold2", "of_sm_pre")
    af, af_wl, af_ph = _phase("alphafold2", 32)
    bad_pre = replace(af.layers[pre], vec_width=0)
    broken_af = replace(af, layers=af.layers[:pre] + (bad_pre,) + af.layers[pre + 1:])
    _raises(lambda: build_rank_ops(broken_af, pre, af_ph, Shard()),
            "alphafold2 structure-module preamble shape is not the published projection")


def test_linear_in_is_once_per_recycle_and_flops_pin_holds():
    from tests.test_core_061_3 import _tflop
    for mid, ref in (("openfold", 221.443), ("alphafold2", 245.739)):
        t, _, wl = _tflop(mid, seq_len=384)
        assert abs(t / ref - 1) < 1e-3, (mid, t)
        assert wl.pair.recycles == 4
    t, _, _ = _tflop("alphafold2", seq_len=256)
    assert abs(t / 131.749 - 1) < 1e-3, t
    m, wl, ph = _phase("openfold", 512)
    pre = _by(build_rank_ops(m, _idx("openfold", "of_sm_pre"), ph, Shard()))
    saved = 7 * 4 * pre["structure.linear_in"].flops
    assert saved == 7 * 4 * 2 * 512 * 384 * 384
    assert abs(saved / 355.38e12) < 2e-4
