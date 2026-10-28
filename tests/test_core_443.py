"""0.443: Protenix bills published LayerNorm, Transition SiLU, and atom-encoder ReLU.

Pairformer: 10 LayerNorms, two SiLUs, two multiplies.
MSA blocks 0-2: 11 LayerNorms, two SiLUs, two multiplies. The last block has no
MSA stack: 8 LayerNorms, one SiLU, one multiply.
Recycle norms run once per cycle. Confidence input_strunk_ln runs once on the
unexpanded residue grid. Diffusion plain LayerNorms, plain Transitions, and
the atom-encoder ReLUs are billed. AdaLN and the diffusion SwiGLU are not.
Vector flops stay 0. The generic .ew tail stays. Sigmoid gates, mask
multiplies, and the OPM divide are not billed.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, Shard, build_rank_ops, layer_repeat
from accel_dse.core.model import NativeWorkload, load_release
from accel_dse.core.scenario import Workload
from accel_dse.core.structure import build_protenix


def _by(ops):
    found = {}
    for op in ops:
        if op.name in found and op.kind == "vector" and not op.name.endswith(".ew"):
            raise AssertionError(op.name)
        found[op.name] = op
    return found


def _ln(op, rows: int, width: int, layer: int, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == rows * width * 4 and op.sfu == float(rows)
    assert op.a_fmt == "fp32" and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _silu(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n * 4) and op.sfu == float(n * 2)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _mul(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _relu(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _phase(seq_len: int = 512, batch: int = 1, **kw):
    m = get_model("protenix")
    wl = resolve_workload(m, Workload(seq_len=seq_len, **kw))
    ph = Phase("full", batch, wl.tokens, 0, pair=wl.pair)
    return m, wl, ph


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release("gh:bytedance/Protenix")))
    edit(rel)
    wl = NativeWorkload("protein", seq_len=8)
    return lambda: build_protenix("protenix", "gh:bytedance/Protenix", rel, wl)


def _touch(ops, gemm: str, before: tuple[str, ...] = (), after: str | None = None) -> None:
    names = [o.name for o in ops]
    i = names.index(gemm)
    for k, want in enumerate(before):
        assert names[i - len(before) + k] == want, names[i - 2:i + 4]
    if after:
        assert names[i + 1] == after and names[i + 2] == gemm + ".ew"
    else:
        assert names[i + 1] == gemm + ".ew"


_PAIR_LN = (
    "attn_s_ln", "attn_z_ln", "trans_s_ln", "trans_z_ln",
    "tri_att_end_ln", "tri_att_start_ln",
    "tri_mul_in_ln_in", "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out",
)
_MSA_LN = (
    "msa_trans_ln", "opm_ln", "pair_trans_ln", "pwa_m_ln", "pwa_z_ln",
    "tri_att_end_ln", "tri_att_start_ln",
    "tri_mul_in_ln_in", "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out",
)
_MSA_LAST_LN = tuple(n for n in _MSA_LN if n not in ("msa_trans_ln", "pwa_m_ln", "pwa_z_ln"))


def test_version_is_443():
    assert __version__ == "0.445.0"


def test_pairformer_layernorm_and_silu():
    m, wl, ph = _phase()
    assert m.n_layers == 101 and m.params() == m.release_params == 368_088_191
    assert m.hidden == 384 and wl.pair.n == 512 and wl.pair.recycles == 4
    assert wl.pair.msa == 2048 and wl.pair.samples == 5 and wl.pair.diff_steps == 200
    assert wl.pair.atoms == 4096
    trunk = [i for i, L in enumerate(m.layers) if L.vec_kind == "px_pair" and L.stack == "pairformer"]
    conf = [i for i, L in enumerate(m.layers) if L.vec_kind == "px_pair" and L.stack == "confidence"]
    assert trunk == list(range(7, 55)) and conf == [85, 86, 87, 88]
    r, p = 512, 512 * 512
    for li in (trunk[0], trunk[-1]):
        layer = m.layers[li]
        assert layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        for name in ("attn_s_ln", "trans_s_ln"):
            _ln(by[name], r, 384, li)
        for name in ("attn_z_ln", "trans_z_ln", "tri_att_end_ln", "tri_att_start_ln",
                     "tri_mul_in_ln_in", "tri_mul_in_ln_out", "tri_mul_out_ln_in", "tri_mul_out_ln_out"):
            _ln(by[name], p, 128, li)
        _silu(by["trans_s_silu"], r * 1536, li, "bf16")
        _mul(by["trans_s_mul"], r * 1536, li, "bf16")
        _silu(by["trans_z_silu"], p * 512, li, "bf16")
        _mul(by["trans_z_mul"], p * 512, li, "bf16")
        fc1 = "pairformer.single_transition.linear_no_bias_a"
        assert by[fc1].w_fmt == "fp32" and by[fc1].w_bits == 32.0
        assert by[fc1].flops == 2 * r * 384 * 1536 and by[fc1].m == r
        assert by[fc1 + ".ew"].vec == r * (384 + 1536) * 4
        assert by["pairformer.pair_transition.linear_no_bias_a.ew"].vec == p * (128 + 512) * 4
        _touch(ops, fc1, ("trans_s_ln",), "trans_s_silu")
        _touch(ops, "pairformer.single_transition.linear_no_bias_b", (), "trans_s_mul")
        _touch(ops, "pairformer.single_transition.linear_no_bias")
        _touch(ops, "pairformer.pair_transition.linear_no_bias_a", ("trans_z_ln",), "trans_z_silu")
        _touch(ops, "pairformer.attention_pair_bias.attention.linear_q", ("attn_s_ln",))
        _touch(ops, "pairformer.attention_pair_bias.attention.linear_g")
        _touch(ops, "pairformer.attention_pair_bias.linear_nobias_z", ("attn_z_ln",))
        _touch(ops, "pairformer.tri_mul_in.linear_a_p", ("tri_mul_in_ln_in",))
        _touch(ops, "pairformer.tri_mul_in.linear_a_g")
        _touch(ops, "pairformer.tri_mul_in.linear_z", ("tri_mul_in_ln_out",))
        assert len([n for n in (o.name for o in ops) if n in _PAIR_LN]) == 10
        banned = ("gelu", "relu", "sigmoid", "softplus", "clamp", "mask_mul")
        assert all(o.name not in banned for o in ops)
    cli = conf[0]
    layer = m.layers[cli]
    assert layer.samples == "all" and layer.repeat == "" and layer_repeat(layer, ph) == 1
    cops = build_rank_ops(m, cli, ph, Shard())
    cby = _by(cops)
    _ln(cby["attn_s_ln"], 5 * r, 384, cli)
    _ln(cby["trans_z_ln"], 5 * p, 128, cli)
    _silu(cby["trans_z_silu"], 5 * p * 512, cli, "bf16")
    _touch(cops, "confidence.single_transition.linear_no_bias_a", ("trans_s_ln",), "trans_s_silu")
    assert cby["confidence.single_transition.linear_no_bias_a.ew"].vec == (5 * r) * (384 + 1536) * 4


def test_msa_recycle_embed_and_confidence():
    m, wl, ph = _phase()
    r, p, s, atoms = 512, 512 * 512, 2048 * 512, 4096
    msa_i = [i for i, L in enumerate(m.layers) if L.vec_kind == "px_msa"]
    assert msa_i == [3, 4, 5] and m.layers[6].vec_kind == "px_msa_last"
    for li in (msa_i[0], msa_i[-1]):
        layer = m.layers[li]
        assert layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        for name in ("msa_trans_ln", "opm_ln", "pwa_m_ln"):
            _ln(by[name], s, 64, li)
        _ln(by["pwa_z_ln"], p, 128, li)
        _ln(by["pair_trans_ln"], p, 128, li)
        _silu(by["msa_trans_silu"], s * 256, li, "bf16")
        _mul(by["msa_trans_mul"], s * 256, li, "bf16")
        _silu(by["pair_trans_silu"], p * 512, li, "bf16")
        _mul(by["pair_trans_mul"], p * 512, li, "bf16")
        assert by["msa.msa_stack.transition_m.linear_no_bias_a"].flops == 2 * s * 64 * 256
        assert by["msa.msa_stack.transition_m.linear_no_bias_a.ew"].vec == s * (64 + 256) * 4
        assert by["msa.outer_product_mean_msa.linear_out"].m == p
        _touch(ops, "msa.msa_stack.transition_m.linear_no_bias_a", ("msa_trans_ln",), "msa_trans_silu")
        _touch(ops, "msa.msa_stack.transition_m.linear_no_bias_b", (), "msa_trans_mul")
        _touch(ops, "msa.outer_product_mean_msa.linear_1", ("opm_ln",))
        _touch(ops, "msa.outer_product_mean_msa.linear_2")
        _touch(ops, "msa.msa_stack.msa_pair_weighted_averaging.linear_no_bias_mv", ("pwa_m_ln",))
        _touch(ops, "msa.msa_stack.msa_pair_weighted_averaging.linear_no_bias_mg")
        _touch(ops, "msa.msa_stack.msa_pair_weighted_averaging.linear_no_bias_z", ("pwa_z_ln",))
        assert len([n for n in (o.name for o in ops) if n in _MSA_LN]) == 11
        assert "attn_s_ln" not in by and "trans_s_silu" not in by
    last = build_rank_ops(m, 6, ph, Shard())
    lby = _by(last)
    assert layer_repeat(m.layers[6], ph) == 4
    _ln(lby["opm_ln"], s, 64, 6)
    _ln(lby["pair_trans_ln"], p, 128, 6)
    _silu(lby["pair_trans_silu"], p * 512, 6, "bf16")
    assert len([n for n in (o.name for o in last) if n in _MSA_LAST_LN]) == 8
    assert "pwa_m_ln" not in lby and "msa_trans_ln" not in lby and "msa_trans_silu" not in lby
    rec_z, rec_s = 90, 92
    assert m.layers[rec_z].vec_kind == m.layers[rec_s].vec_kind == "px_recycle"
    assert layer_repeat(m.layers[rec_z], ph) == 4
    assert build_rank_ops(m, 91, ph, Shard()) == []
    assert build_rank_ops(m, 93, ph, Shard()) == []
    assert build_rank_ops(m, 94, ph, Shard()) == []
    zops = build_rank_ops(m, rec_z, ph, Shard())
    sops = build_rank_ops(m, rec_s, ph, Shard())
    _ln(_by(zops)["z_ln"], p, 128, rec_z)
    _ln(_by(sops)["s_ln"], r, 384, rec_s)
    _touch(zops, "recycle.", ("z_ln",))
    _touch(sops, "recycle.", ("s_ln",))
    assert _by(zops)["recycle."].m == p and _by(sops)["recycle."].m == r
    emb = 89
    assert m.layers[emb].vec_kind == "px_embed" and layer_repeat(m.layers[emb], ph) == 1
    eops = build_rank_ops(m, emb, ph, Shard())
    eby = _by(eops)
    _relu(eby["cl_relu"], atoms * 128, emb, "bf16")
    _relu(eby["ck_relu"], 16384 * 128, emb, "bf16")
    _relu(eby["q_relu"], atoms * 384, emb, "bf16")
    apair = 524288
    for name in ("mlp_relu_1", "mlp_relu_2", "mlp_relu_3"):
        _relu(eby[name], apair * 16, emb, "bf16")
    assert eby["embed.atom_attention_encoder.linear_no_bias_cl"].m == atoms
    _touch(eops, "embed.atom_attention_encoder.linear_no_bias_cl", ("cl_relu",))
    _touch(eops, "embed.atom_attention_encoder.linear_no_bias_cm", ("ck_relu",))
    _touch(eops, "embed.atom_attention_encoder.linear_no_bias_q", (), "q_relu")
    _touch(eops, "embed.atom_attention_encoder.small_mlp.1", ("mlp_relu_1",))
    _touch(eops, "embed.atom_attention_encoder.small_mlp.5", ("mlp_relu_3",))
    conf = 98
    assert m.layers[conf].vec_kind == "px_conf" and m.layers[conf].samples == "all"
    assert layer_repeat(m.layers[conf], ph) == 1
    cops = build_rank_ops(m, conf, ph, Shard())
    cby = _by(cops)
    _ln(cby["input_s_ln"], r, 384, conf)
    _ln(cby["pae_ln"], 5 * p, 128, conf)
    _ln(cby["pde_ln"], 5 * p, 128, conf)
    _ln(cby["plddt_ln"], 5 * atoms, 384, conf)
    _ln(cby["resolved_ln"], 5 * atoms, 384, conf)
    _touch(cops, "confidence.linear_no_bias_s1", ("input_s_ln",))
    _touch(cops, "confidence.linear_no_bias_pae", ("pae_ln",))
    names = [o.name for o in cops]
    i = names.index("confidence.linear_no_bias_pde")
    assert names[i - 1:i + 4] == [
        "pde_ln", "confidence.linear_no_bias_pde", "plddt_ln", "resolved_ln",
        "confidence.linear_no_bias_pde.ew",
    ]
    assert cby["confidence.linear_no_bias_pae"].m == 5 * p
    assert cby["confidence.linear_no_bias_s1"].m == 5 * r
    assert cby["confidence.linear_no_bias_s1"].k == 449
    heads = build_rank_ops(m, 99, ph, Shard())
    assert all(o.name not in ("input_s_ln", "pae_ln", "plddt_ln") for o in heads)
    assert m.layers[95].vec_kind == "" and m.layers[96].vec_kind == ""


def test_diffusion_plain_norms_and_dit_pair_bias():
    m, wl, ph = _phase()
    r, p, atoms = 512, 512 * 512, 4096
    diff = 97
    assert m.layers[diff].vec_kind == "px_diff" and m.layers[diff].repeat == "diff"
    assert layer_repeat(m.layers[diff], ph) == 200
    dops = build_rank_ops(m, diff, ph, Shard())
    dby = _by(dops)
    _ln(dby["cond_z_ln"], p, 256, diff)
    _ln(dby["cond_s_ln"], r, 833, diff)
    _ln(dby["noise_ln"], 5, 256, diff)
    _ln(dby["z1_ln"], p, 128, diff)
    _silu(dby["z1_silu"], p * 256, diff, "bf16")
    _mul(dby["z1_mul"], p * 256, diff, "bf16")
    _ln(dby["z2_ln"], p, 128, diff)
    _ln(dby["s1_ln"], 5 * r, 384, diff)
    _silu(dby["s1_silu"], 5 * r * 768, diff, "bf16")
    _mul(dby["s1_mul"], 5 * r * 768, diff, "bf16")
    _ln(dby["s2_ln"], 5 * r, 384, diff)
    _ln(dby["enc_s_ln"], 5 * r, 384, diff)
    _ln(dby["enc_z_ln"], 5 * p, 128, diff)
    _ln(dby["tok_s_ln"], 5 * r, 384, diff)
    _ln(dby["a_ln"], 5 * r, 768, diff)
    _ln(dby["q_ln"], 5 * atoms, 128, diff)
    _relu(dby["cl_relu"], 5 * atoms * 128, diff, "bf16")
    _relu(dby["ck_relu"], 5 * 16384 * 128, diff, "bf16")
    _relu(dby["q_relu"], 5 * atoms * 768, diff, "bf16")
    for name in ("mlp_relu_1", "mlp_relu_2", "mlp_relu_3"):
        _relu(dby[name], 5 * 524288 * 16, diff, "bf16")
    zname = "diffusion.diffusion_conditioning.linear_no_bias_z"
    sname = "diffusion.diffusion_conditioning.linear_no_bias_s"
    nname = "diffusion.diffusion_conditioning.linear_no_bias_n"
    assert dby[zname].m == p and dby[zname].k == 256 and dby[zname].n == 128
    assert dby[sname].m == 5 * r and dby[sname].k == 833
    assert dby[nname].m == 5 * r and dby[nname].k == 256
    assert dby["diffusion.atom_attention_encoder.linear_no_bias_cl"].m == 5 * atoms
    assert dby["diffusion.atom_attention_encoder.linear_no_bias_s"].m == 5 * atoms
    assert dby["diffusion.atom_attention_encoder.linear_no_bias_z"].m == 5 * p
    assert dby["diffusion.atom_attention_decoder.linear_no_bias_a"].m == 5 * atoms
    assert dby["diffusion.atom_attention_decoder.linear_no_bias_a"].k == 768
    assert dby["diffusion.linear_no_bias_s"].m == 5 * r
    assert dby[zname + ".ew"].vec == p * (256 + 128) * 4
    assert dby["diffusion.diffusion_conditioning.transition_s1.linear_no_bias_a.ew"].vec == (5 * r) * (384 + 768) * 4
    assert dby["diffusion.diffusion_conditioning.transition_z1.linear_no_bias_a.ew"].vec == p * (128 + 256) * 4
    _touch(dops, zname, ("cond_z_ln",))
    _touch(dops, sname, ("cond_s_ln",))
    _touch(dops, nname, ("noise_ln",))
    _touch(dops, "diffusion.diffusion_conditioning.transition_s1.linear_no_bias_a", ("s1_ln",), "s1_silu")
    _touch(dops, "diffusion.diffusion_conditioning.transition_s1.linear_no_bias_b", (), "s1_mul")
    _touch(dops, "diffusion.diffusion_conditioning.transition_z1.linear_no_bias_a", ("z1_ln",), "z1_silu")
    _touch(dops, "diffusion.atom_attention_encoder.linear_no_bias_s", ("enc_s_ln",))
    _touch(dops, "diffusion.atom_attention_encoder.linear_no_bias_z", ("enc_z_ln",))
    _touch(dops, "diffusion.atom_attention_encoder.linear_no_bias_cl", ("cl_relu",))
    _touch(dops, "diffusion.atom_attention_encoder.linear_no_bias_q", (), "q_relu")
    _touch(dops, "diffusion.linear_no_bias_s", ("tok_s_ln",))
    _touch(dops, "diffusion.atom_attention_decoder.linear_no_bias_a", ("a_ln",))
    _touch(dops, "diffusion.atom_attention_decoder.linear_no_bias_out", ("q_ln",))
    dit = 58
    assert m.layers[dit].vec_kind == "px_dit" and m.layers[dit].stack == "diffusion"
    assert layer_repeat(m.layers[dit], ph) == 200
    tops = build_rank_ops(m, dit, ph, Shard())
    tby = _by(tops)
    _ln(tby["attn_z_ln"], 5 * p, 128, dit)
    _touch(tops, "diffusion.attention_pair_bias.linear_nobias_z", ("attn_z_ln",))
    assert tby["diffusion.attention_pair_bias.linear_nobias_z"].m == 5 * p
    assert "diffusion.conditioned_transition_block.adaln.linear_s" in tby
    assert "diffusion.conditioned_transition_block.linear_nobias_a1" in tby
    assert any(o.name.endswith(".softmax") for o in tops)
    assert all("silu" not in o.name and "relu" not in o.name and "gelu" not in o.name for o in tops)
    atom = build_rank_ops(m, 0, ph, Shard())
    aby = _by(atom)
    assert m.layers[0].vec_kind == "px_dit" and layer_repeat(m.layers[0], ph) == 1
    _ln(aby["attn_z_ln"], 524288, 16, 0)
    assert "atom_embed.attention_pair_bias.layernorm_kv.linear_s" in aby
    assert "atom_embed.conditioned_transition_block.adaln.linear_s" in aby
    assert all("silu" not in o.name and "relu" not in o.name for o in atom)
    kinds = {}
    for L in m.layers:
        if L.vec_kind:
            kinds[L.vec_kind] = kinds.get(L.vec_kind, 0) + 1
    assert kinds == {
        "px_pair": 52, "px_dit": 33, "px_msa": 3, "px_msa_last": 1,
        "px_recycle": 2, "px_embed": 1, "px_diff": 1, "px_conf": 1,
    }


def test_batch_samples_dap_fp8_and_short():
    m, wl, ph = _phase()
    two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
    b2 = _by(build_rank_ops(m, 7, two, Shard()))
    _ln(b2["attn_s_ln"], 2 * 512, 384, 7)
    _ln(b2["trans_z_ln"], 2 * 512 * 512, 128, 7)
    d2 = _by(build_rank_ops(m, 97, two, Shard()))
    _ln(d2["cond_s_ln"], 2 * 512, 833, 97)
    _ln(d2["noise_ln"], 2 * 5, 256, 97)
    _ln(d2["z1_ln"], 2 * 512 * 512, 128, 97)
    _ln(d2["s1_ln"], 2 * 5 * 512, 384, 97)
    many = replace(wl.pair, samples=2)
    ph2 = Phase("full", 1, wl.tokens, 0, pair=many)
    trunk = _by(build_rank_ops(m, 7, ph2, Shard()))
    _ln(trunk["attn_s_ln"], 512, 384, 7)
    _ln(trunk["trans_z_ln"], 512 * 512, 128, 7)
    conf_pf = _by(build_rank_ops(m, 85, ph2, Shard()))
    _ln(conf_pf["attn_s_ln"], 2 * 512, 384, 85)
    _ln(conf_pf["trans_z_ln"], 2 * 512 * 512, 128, 85)
    diff = _by(build_rank_ops(m, 97, ph2, Shard()))
    _ln(diff["cond_s_ln"], 512, 833, 97)
    _ln(diff["noise_ln"], 2, 256, 97)
    _ln(diff["z1_ln"], 512 * 512, 128, 97)
    _ln(diff["s1_ln"], 2 * 512, 384, 97)
    _ln(diff["enc_z_ln"], 2 * 512 * 512, 128, 97)
    _ln(diff["a_ln"], 2 * 512, 768, 97)
    conf = _by(build_rank_ops(m, 98, ph2, Shard()))
    _ln(conf["input_s_ln"], 512, 384, 98)
    _ln(conf["pae_ln"], 2 * 512 * 512, 128, 98)
    _ln(conf["plddt_ln"], 2 * 4096, 384, 98)
    assert conf["confidence.linear_no_bias_s1"].m == 2 * 512
    dap = _by(build_rank_ops(m, 7, ph, Shard(sp=2)))
    _ln(dap["attn_s_ln"], 512, 384, 7, rep=2)
    _silu(dap["trans_s_silu"], 512 * 1536, 7, "bf16", rep=2)
    _ln(dap["trans_z_ln"], 256 * 512, 128, 7, rep=1)
    _silu(dap["trans_z_silu"], (256 * 512) * 512, 7, "bf16", rep=1)
    assert dap["pairformer.single_transition.linear_no_bias_a"].m == 512
    assert dap["pairformer.single_transition.linear_no_bias_a"].replicated == 2
    assert dap["pairformer.pair_transition.linear_no_bias_a"].m == 256 * 512
    md = _by(build_rank_ops(m, 3, ph, Shard(sp=2)))
    _ln(md["msa_trans_ln"], 1024 * 512, 64, 3, rep=1)
    _ln(md["pwa_z_ln"], 256 * 512, 128, 3, rep=1)
    dd = _by(build_rank_ops(m, 97, ph, Shard(sp=2)))
    _ln(dd["cond_s_ln"], 512, 833, 97, rep=2)
    _ln(dd["noise_ln"], 5, 256, 97, rep=2)
    _ln(dd["s1_ln"], 5 * 512, 384, 97, rep=2)
    _silu(dd["s1_silu"], 5 * 512 * 768, 97, "bf16", rep=2)
    _ln(dd["z1_ln"], 256 * 512, 128, 97, rep=1)
    _ln(dd["enc_z_ln"], 5 * 256 * 512, 128, 97, rep=1)
    _relu(dd["cl_relu"], 5 * 4096 * 128, 97, "bf16", rep=2)
    assert dd["diffusion.diffusion_conditioning.linear_no_bias_s"].m == 5 * 512
    assert dd["diffusion.diffusion_conditioning.linear_no_bias_s"].replicated == 2
    assert dd["diffusion.diffusion_conditioning.linear_no_bias_z"].m == 256 * 512
    split = replace(wl.pair, split=True)
    phs = Phase("full", 1, wl.tokens, 0, pair=split)
    rep = 2 * 3 / 5
    sd = _by(build_rank_ops(m, 97, phs, Shard(sp=2)))
    _ln(sd["s1_ln"], 3 * 512, 384, 97, rep=rep)
    _silu(sd["s1_silu"], 3 * 512 * 768, 97, "bf16", rep=rep)
    _ln(sd["noise_ln"], 3, 256, 97, rep=rep)
    _relu(sd["cl_relu"], 3 * 4096 * 128, 97, "bf16", rep=rep)
    _ln(sd["a_ln"], 3 * 512, 768, 97, rep=rep)
    _ln(sd["z1_ln"], 256 * 512, 128, 97, rep=1)
    _ln(sd["cond_s_ln"], 512, 833, 97, rep=2)
    _ln(sd["enc_z_ln"], 5 * 256 * 512, 128, 97, rep=1)
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 7, ph, Shard()))
    _silu(op["trans_s_silu"], 512 * 1536, 7, "fp8")
    _mul(op["trans_z_mul"], (512 * 512) * 512, 7, "fp8")
    _ln(op["attn_s_ln"], 512, 384, 7)
    assert op["pairformer.single_transition.linear_no_bias_a"].a_fmt == "fp8"
    assert op["pairformer.single_transition.linear_no_bias_a"].w_fmt == "fp32"
    dop = _by(build_rank_ops(fp8, 97, ph, Shard()))
    _relu(dop["cl_relu"], 5 * 4096 * 128, 97, "fp8")
    _silu(dop["s1_silu"], 5 * 512 * 768, 97, "fp8")
    _ln(dop["a_ln"], 5 * 512, 768, 97)
    assert dop["diffusion.atom_attention_encoder.linear_no_bias_q"].a_fmt == "fp8"
    assert dop["diffusion.atom_attention_encoder.linear_no_bias_q"].w_fmt == "fp32"
    short_m, short_wl, short = _phase(128)
    assert short_wl.pair.n == 128 and short_wl.pair.msa == 2048 and short_wl.pair.atoms == 1024
    sby = _by(build_rank_ops(short_m, 7, short, Shard()))
    _ln(sby["attn_s_ln"], 128, 384, 7)
    _ln(sby["trans_z_ln"], 128 * 128, 128, 7)
    mby = _by(build_rank_ops(short_m, 3, short, Shard()))
    _ln(mby["msa_trans_ln"], 2048 * 128, 64, 3)
    one_m, one_wl, one = _phase(1)
    assert one_wl.pair.atoms == 8
    oby = _by(build_rank_ops(one_m, 89, one, Shard()))
    _relu(oby["cl_relu"], 32 * 128, 89, "bf16")
    _relu(oby["ck_relu"], 128 * 128, 89, "bf16")
    _relu(oby["mlp_relu_1"], 4096 * 16, 89, "bf16")
    assert oby["embed.atom_attention_encoder.linear_no_bias_cl"].m == 8


def test_non_vector_schedule_unchanged():
    m, wl, ph = _phase(64)
    bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers))

    def sched(model, shard):
        out = []
        ews = []
        for li, _L in enumerate(model.layers):
            for o in build_rank_ops(model, li, ph, shard):
                if o.name.endswith(".ew"):
                    ews.append((li, o.name, o.vec, o.replicated))
                elif o.kind != "vector":
                    out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.replicated))
        return out, ews

    for shard in (Shard(), Shard(sp=2)):
        got, gew = sched(m, shard)
        base, bew = sched(bare, shard)
        assert got == base
        assert gew == bew


def test_other_models_do_not_gain_protenix_vectors():
    for mid in ("boltz-1", "openfold", "alphafold2", "esmfold", "esm2-3b"):
        other = get_model(mid)
        assert all(not L.vec_kind.startswith("px_") for L in other.layers)
    px = get_model("protenix")
    assert all(not L.vec_kind.startswith("bz_") for L in px.layers)
    assert all(not L.vec_kind.startswith("of_") for L in px.layers)
    assert px.vec_kind == ""
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "protenix" not in ids[:60]
    from tests.test_core_coverage import _FULL_0651
    assert "protenix" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_shape_mismatch_is_refused():
    _raises(_bad(lambda r: r["shapes"].pop(
        "pairformer_stack.blocks.0.attention_pair_bias.layernorm_a.weight")),
            "pairformer_stack.blocks.0.attention_pair_bias.layernorm_a.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "pairformer_stack.blocks.47.pair_transition.layernorm1.weight", [64])),
            "pairformer_stack.blocks.47.pair_transition.layernorm1.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].pop(
        "pairformer_stack.blocks.0.single_transition.linear_no_bias_a.weight")),
            "protenix pairformer linears are not the published set")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "pairformer_stack.blocks.0.tri_mul_in.linear_z.weight", [64, 128])),
            "protenix tri_mul_in.linear_z shape is not the published hidden")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "diffusion_module.diffusion_transformer.blocks.0."
        "attention_pair_bias.layernorm_a.layernorm_a.weight", [384])),
            "is not the published scale-free AdaLN")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "diffusion_module.diffusion_conditioning.layernorm_s.bias", [833])),
            "diffusion_module.diffusion_conditioning.layernorm_s.bias is not in the published LayerNorm")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "msa_module.blocks.3.msa_stack.transition_m.layernorm1.weight", [64])),
            "last msa block still has an MSA stack")
    _raises(_bad(lambda r: r["config"].__setitem__("c_s", 128)),
            "protenix channel width is not the published pairformer")
    _raises(_bad(lambda r: r["config"].__setitem__("template", True)),
            "protenix template flag is not the published disabled setting")

    def shrink_pair(rel):
        rel["stacks"]["pairformer_stack.blocks"] = 47
        rel["config"]["pairformer_blocks"] = 47

    _raises(_bad(shrink_pair), "protenix pairformer blocks are not 48")

    def shrink_msa(rel):
        rel["stacks"]["msa_module.blocks"] = 3
        rel["config"]["msa_blocks"] = 3

    _raises(_bad(shrink_msa), "protenix msa blocks are not 4")
    m, wl, ph = _phase(32)
    bad_layer = replace(m.layers[7], pair_linears=m.layers[7].pair_linears[:-1])
    broken = replace(m, layers=m.layers[:7] + (bad_layer,) + m.layers[8:])
    _raises(lambda: build_rank_ops(broken, 7, ph, Shard()),
            "protenix pairformer linears are not the published set")
    bad_diff = replace(m.layers[97], pair_linears=m.layers[97].pair_linears[:-1])
    broken_d = replace(m, layers=m.layers[:97] + (bad_diff,) + m.layers[98:])
    _raises(lambda: build_rank_ops(broken_d, 97, ph, Shard()),
            "protenix diffusion linears are not the published set")


def test_closed_form_flops_still_hold():
    from tests.test_core_061_3 import _tflop
    t, st, _ = _tflop("protenix", seq_len=384)
    assert abs(t - 280.425) < 0.05, t
    assert abs(st["pairformer"] / 4 - 12.207) < 0.01
    assert abs(st["diffusion"] / 200 * 1e3 - 985.73) < 0.5
