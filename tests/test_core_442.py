"""0.442: Boltz-1 bills published LayerNorm and Transition SiLU.

Pairformer: 10 LayerNorms, two SiLUs, two multiplies.
MSA block: 11 LayerNorms, two SiLUs, two multiplies.
Trunk recycle norms run once per recycle. Confidence imitate-trunk norms run
once on the unexpanded grid. Diffusion plain Transitions and plain LayerNorms
are billed. AdaLN and the diffusion SwiGLU are not. Vector flops stay 0.
The generic .ew tail stays. Sigmoid gates, mask multiplies, and the OPM
divide are not billed.
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
from accel_dse.core.structure import build_boltz1


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


def _phase(seq_len: int = 512, batch: int = 1, **kw):
    m = get_model("boltz-1")
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
    rel = json.loads(json.dumps(load_release("boltz-community/boltz-1")))
    edit(rel)
    wl = NativeWorkload("protein", seq_len=8)
    return lambda: build_boltz1("boltz-1", "boltz-community/boltz-1", rel, wl)


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
_MSA_LN = _PAIR_LN[2:] + ("msa_trans_ln", "opm_ln", "pwa_m_ln", "pwa_z_ln", "z_trans_ln")


def test_version_is_442():
    assert __version__ == "0.482.0"


def test_pairformer_layernorm_and_silu():
    m, wl, ph = _phase()
    assert m.n_layers == 152 and m.params() == m.release_params == 592_009_341
    assert m.hidden == 384 and wl.pair.n == 512 and wl.pair.recycles == 4
    assert wl.pair.msa == 4096 and wl.pair.samples == 1 and wl.pair.diff_steps == 200
    trunk = [i for i, L in enumerate(m.layers) if L.vec_kind == "bz_pair" and L.stack == "pairformer"]
    conf = [i for i, L in enumerate(m.layers) if L.vec_kind == "bz_pair" and L.stack == "confidence"]
    assert trunk == list(range(7, 55)) and conf[0] == 93 and len(conf) == 48
    r, p = 512, 512 * 512
    for li in (trunk[0], trunk[-1]):
        layer = m.layers[li]
        assert layer.vec_kind == "bz_pair" and layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
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
        fc1 = "pairformer.transition_s.fc1"
        assert by[fc1].w_fmt == "fp32" and by[fc1].w_bits == 32.0
        assert by[fc1].flops == 2 * r * 384 * 1536 and by[fc1].m == r
        assert by[fc1 + ".ew"].vec == r * (384 + 1536) * 4
        assert by["pairformer.transition_z.fc1.ew"].vec == p * (128 + 512) * 4
        _touch(ops, fc1, ("trans_s_ln",), "trans_s_silu")
        _touch(ops, "pairformer.transition_s.fc2", (), "trans_s_mul")
        _touch(ops, "pairformer.transition_s.fc3")
        _touch(ops, "pairformer.transition_z.fc1", ("trans_z_ln",), "trans_z_silu")
        _touch(ops, "pairformer.attention.proj_q", ("attn_s_ln",))
        _touch(ops, "pairformer.attention.proj_g")
        _touch(ops, "pairformer.attention.proj_z.1", ("attn_z_ln",))
        _touch(ops, "pairformer.tri_mul_in.p_in", ("tri_mul_in_ln_in",))
        _touch(ops, "pairformer.tri_mul_in.g_in")
        _touch(ops, "pairformer.tri_mul_in.p_out", ("tri_mul_in_ln_out",))
        assert len([n for n in (o.name for o in ops) if n in _PAIR_LN]) == 10
        banned = ("gelu", "relu", "sigmoid", "softplus", "clamp", "mask_mul")
        assert all(o.name not in banned for o in ops)
    cli = conf[0]
    layer = m.layers[cli]
    assert layer.samples == "all" and layer.repeat == "" and layer_repeat(layer, ph) == 1
    cops = build_rank_ops(m, cli, ph, Shard())
    cby = _by(cops)
    _ln(cby["attn_s_ln"], r, 384, cli)
    _ln(cby["trans_z_ln"], p, 128, cli)
    _silu(cby["trans_z_silu"], p * 512, cli, "bf16")
    _touch(cops, "confidence.transition_s.fc1", ("trans_s_ln",), "trans_s_silu")
    assert cby["confidence.transition_s.fc1.ew"].vec == r * (384 + 1536) * 4


def test_msa_recycle_confidence_and_conditioners():
    m, wl, ph = _phase()
    r, p, s = 512, 512 * 512, 4096 * 512
    msa_i = [i for i, L in enumerate(m.layers) if L.vec_kind == "bz_msa" and L.stack == "msa"]
    cmsa = [i for i, L in enumerate(m.layers) if L.vec_kind == "bz_msa" and L.stack == "conf_msa"]
    assert msa_i == [3, 4, 5, 6] and cmsa == [89, 90, 91, 92]
    for li in (msa_i[0], msa_i[-1]):
        layer = m.layers[li]
        assert layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        for name in ("msa_trans_ln", "opm_ln", "pwa_m_ln"):
            _ln(by[name], s, 64, li)
        _ln(by["pwa_z_ln"], p, 128, li)
        _ln(by["z_trans_ln"], p, 128, li)
        _silu(by["msa_trans_silu"], s * 256, li, "bf16")
        _mul(by["msa_trans_mul"], s * 256, li, "bf16")
        _silu(by["z_trans_silu"], p * 512, li, "bf16")
        _mul(by["z_trans_mul"], p * 512, li, "bf16")
        assert by["msa.msa_transition.fc1"].flops == 2 * s * 64 * 256
        assert by["msa.msa_transition.fc1.ew"].vec == s * (64 + 256) * 4
        assert by["msa.outer_product_mean.proj_o"].m == p
        _touch(ops, "msa.msa_transition.fc1", ("msa_trans_ln",), "msa_trans_silu")
        _touch(ops, "msa.msa_transition.fc2", (), "msa_trans_mul")
        _touch(ops, "msa.outer_product_mean.proj_a", ("opm_ln",))
        _touch(ops, "msa.outer_product_mean.proj_b")
        _touch(ops, "msa.pair_weighted_averaging.proj_m", ("pwa_m_ln",))
        _touch(ops, "msa.pair_weighted_averaging.proj_g")
        _touch(ops, "msa.pair_weighted_averaging.proj_z", ("pwa_z_ln",))
        _touch(ops, "msa.z_transition.fc1", ("z_trans_ln",), "z_trans_silu")
        assert len([n for n in (o.name for o in ops) if n in _MSA_LN]) == 11
        assert "attn_s_ln" not in by and "trans_s_silu" not in by
    assert m.layers[cmsa[0]].samples == "all" and layer_repeat(m.layers[cmsa[0]], ph) == 1
    cby = _by(build_rank_ops(m, cmsa[0], ph, Shard()))
    _ln(cby["msa_trans_ln"], s, 64, cmsa[0])
    _ln(cby["pwa_z_ln"], p, 128, cmsa[0])
    rec_s, rec_z = 142, 143
    assert m.layers[rec_s].vec_kind == m.layers[rec_z].vec_kind == "bz_recycle"
    assert layer_repeat(m.layers[rec_s], ph) == 4 and m.layers[144].vec_kind == ""
    assert build_rank_ops(m, 144, ph, Shard()) == []
    sops = build_rank_ops(m, rec_s, ph, Shard())
    zops = build_rank_ops(m, rec_z, ph, Shard())
    _ln(_by(sops)["s_ln"], r, 384, rec_s)
    _ln(_by(zops)["z_ln"], p, 128, rec_z)
    _touch(sops, "recycle.", ("s_ln",))
    _touch(zops, "recycle.", ("z_ln",))
    assert _by(sops)["recycle."].m == r and _by(zops)["recycle."].m == p
    conf = next(i for i, L in enumerate(m.layers) if L.vec_kind == "bz_conf")
    assert conf == 149 and m.layers[conf].samples == "all" and layer_repeat(m.layers[conf], ph) == 1
    cops = build_rank_ops(m, conf, ph, Shard())
    cby = _by(cops)
    _ln(cby["conf_s_ln"], r, 384, conf)
    _ln(cby["conf_z_ln"], p, 128, conf)
    _ln(cby["s_diff_ln"], r, 768, conf)
    _ln(cby["final_s_ln"], r, 384, conf)
    _ln(cby["final_z_ln"], p, 128, conf)
    _touch(cops, "confidence.s_recycle", ("conf_s_ln",))
    _touch(cops, "confidence.z_recycle", ("conf_z_ln",))
    _touch(cops, "confidence.s_diffusion_to_s", ("s_diff_ln",))
    _touch(cops, "confidence.confidence_heads.to_plddt_logits", ("final_s_ln",))
    _touch(cops, "confidence.confidence_heads.to_resolved_logits")
    _touch(cops, "confidence.confidence_heads.to_pae_logits", ("final_z_ln",))
    assert cby["confidence.confidence_heads.to_pae_logits"].m == r
    assert cby["confidence.s_to_z_prod_in1"].m == r
    cache = next(i for i, L in enumerate(m.layers) if L.vec_kind == "bz_cache")
    assert cache == 55 and layer_repeat(m.layers[cache], ph) == 1
    kops = build_rank_ops(m, cache, ph, Shard())
    kby = _by(kops)
    _ln(kby["pair_init_ln"], p, 256, cache)
    _ln(kby["pair_trans0_ln"], p, 128, cache)
    _silu(kby["pair_trans0_silu"], p * 256, cache, "bf16")
    _mul(kby["pair_trans0_mul"], p * 256, cache, "bf16")
    _silu(kby["pair_trans1_silu"], p * 256, cache, "bf16")
    init = "diff_cache.diffusion.score_model.pairwise_conditioner.dim_pairwise_init_proj.1"
    t0 = "diff_cache.diffusion.score_model.pairwise_conditioner.transitions.0.fc1"
    _touch(kops, init, ("pair_init_ln",))
    _touch(kops, t0, ("pair_trans0_ln",), "pair_trans0_silu")
    assert kby[init].m == p and kby[init].k == 256 and kby[init].n == 128
    assert kby[t0 + ".ew"].vec == p * (128 + 256) * 4
    assert sum(1 for o in kops if o.name == "pair_init_ln") == 1
    diff = next(i for i, L in enumerate(m.layers) if L.vec_kind == "bz_diff")
    assert diff == 148 and m.layers[diff].repeat == "diff" and layer_repeat(m.layers[diff], ph) == 200
    dops = build_rank_ops(m, diff, ph, Shard())
    dby = _by(dops)
    _ln(dby["norm_single"], r, 839, diff)
    _ln(dby["fourier_ln"], 1, 256, diff)
    _ln(dby["single_trans0_ln"], r, 768, diff)
    _silu(dby["single_trans0_silu"], r * 1536, diff, "bf16")
    _mul(dby["single_trans0_mul"], r * 1536, diff, "bf16")
    _ln(dby["s_to_a_ln"], r, 768, diff)
    _ln(dby["a_ln"], r, 768, diff)
    _ln(dby["tok_upd_ln"], r, 768, diff)
    _ln(dby["tok_fourier_ln"], 1, 256, diff)
    _touch(dops, "diffusion.score_model.single_conditioner.single_embed", ("norm_single",))
    _touch(dops, "diffusion.score_model.single_conditioner.fourier_to_single", ("fourier_ln",))
    _touch(dops, "diffusion.score_model.single_conditioner.transitions.0.fc1",
           ("single_trans0_ln",), "single_trans0_silu")
    _touch(dops, "diffusion.score_model.s_to_a_linear.1", ("s_to_a_ln",))
    _touch(dops, "diffusion.score_model.atom_attention_decoder.a_to_q_trans", ("a_ln",))
    _touch(dops, "diffusion.out_token_feat_update.transition_block.a_to_b",
           ("tok_upd_ln", "tok_fourier_ln"))
    assert dby["diffusion.score_model.single_conditioner.fourier_to_single"].m == r
    assert dby["diffusion.score_model.atom_attention_decoder.a_to_q_trans"].m == 8 * r
    assert "single_trans0_silu" not in {o.name for o in build_rank_ops(m, 59, ph, Shard())}
    dit = build_rank_ops(m, 59, ph, Shard())
    assert m.layers[59].vec_kind == "" and m.layers[59].stack == "diffusion"
    assert all(o.name not in ("silu", "gelu", "relu", "norm_single", "a_ln") for o in dit)
    assert any(o.name == "diffusion.adaln.s_scale" for o in dit)
    assert any(o.name == "diffusion.transition.swish_gate.0" for o in dit)


def test_batch_samples_dap_fp8_and_short():
    m, wl, ph = _phase()
    two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
    b2 = _by(build_rank_ops(m, 7, two, Shard()))
    _ln(b2["attn_s_ln"], 2 * 512, 384, 7)
    _ln(b2["trans_z_ln"], 2 * 512 * 512, 128, 7)
    _silu(b2["trans_s_silu"], 2 * 512 * 1536, 7, "bf16")
    many = replace(wl.pair, samples=2)
    ph2 = Phase("full", 1, wl.tokens, 0, pair=many)
    trunk = _by(build_rank_ops(m, 7, ph2, Shard()))
    _ln(trunk["attn_s_ln"], 512, 384, 7)
    _ln(trunk["trans_z_ln"], 512 * 512, 128, 7)
    conf_pf = _by(build_rank_ops(m, 93, ph2, Shard()))
    _ln(conf_pf["attn_s_ln"], 2 * 512, 384, 93)
    _ln(conf_pf["trans_z_ln"], 2 * 512 * 512, 128, 93)
    _silu(conf_pf["trans_z_silu"], 2 * 512 * 512 * 512, 93, "bf16")
    cmsa = _by(build_rank_ops(m, 89, ph2, Shard()))
    _ln(cmsa["msa_trans_ln"], 2 * 4096 * 512, 64, 89)
    conf = _by(build_rank_ops(m, 149, ph2, Shard()))
    _ln(conf["conf_s_ln"], 512, 384, 149)
    _ln(conf["conf_z_ln"], 512 * 512, 128, 149)
    _ln(conf["final_s_ln"], 2 * 512, 384, 149)
    _ln(conf["final_z_ln"], 2 * 512 * 512, 128, 149)
    _ln(conf["s_diff_ln"], 2 * 512, 768, 149)
    assert conf["confidence.s_recycle"].m == 2 * 512
    assert conf["confidence.confidence_heads.to_pae_logits"].m == 2 * 512
    diff = _by(build_rank_ops(m, 148, ph2, Shard()))
    _ln(diff["fourier_ln"], 2, 256, 148)
    _ln(diff["tok_fourier_ln"], 2, 256, 148)
    _ln(diff["norm_single"], 2 * 512, 839, 148)
    _ln(diff["a_ln"], 2 * 512, 768, 148)
    _silu(diff["single_trans0_silu"], 2 * 512 * 1536, 148, "bf16")
    assert diff["diffusion.score_model.single_conditioner.fourier_to_single"].m == 2 * 512
    dap = _by(build_rank_ops(m, 7, ph, Shard(sp=2)))
    _ln(dap["attn_s_ln"], 512, 384, 7, rep=2)
    _silu(dap["trans_s_silu"], 512 * 1536, 7, "bf16", rep=2)
    _ln(dap["trans_z_ln"], 256 * 512, 128, 7, rep=1)
    _silu(dap["trans_z_silu"], (256 * 512) * 512, 7, "bf16", rep=1)
    assert dap["pairformer.transition_s.fc1"].m == 512 and dap["pairformer.transition_s.fc1"].replicated == 2
    assert dap["pairformer.transition_z.fc1"].m == 256 * 512
    md = _by(build_rank_ops(m, 3, ph, Shard(sp=2)))
    _ln(md["msa_trans_ln"], 2048 * 512, 64, 3, rep=1)
    _ln(md["pwa_z_ln"], 256 * 512, 128, 3, rep=1)
    _silu(md["msa_trans_silu"], (2048 * 512) * 256, 3, "bf16", rep=1)
    dd = _by(build_rank_ops(m, 148, ph, Shard(sp=2)))
    _ln(dd["fourier_ln"], 1, 256, 148, rep=2)
    _ln(dd["norm_single"], 512, 839, 148, rep=2)
    _ln(dd["a_ln"], 512, 768, 148, rep=2)
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 7, ph, Shard()))
    _silu(op["trans_s_silu"], 512 * 1536, 7, "fp8")
    _mul(op["trans_z_mul"], (512 * 512) * 512, 7, "fp8")
    _ln(op["attn_s_ln"], 512, 384, 7)
    assert op["pairformer.transition_s.fc1"].a_fmt == "fp8"
    assert op["pairformer.transition_s.fc1"].w_fmt == "fp32"
    short_m, short_wl, short = _phase(128)
    assert short_wl.pair.n == 128 and short_wl.pair.msa == 4096
    sby = _by(build_rank_ops(short_m, 7, short, Shard()))
    _ln(sby["attn_s_ln"], 128, 384, 7)
    _ln(sby["trans_z_ln"], 128 * 128, 128, 7)
    _silu(sby["trans_z_silu"], (128 * 128) * 512, 7, "bf16")
    mby = _by(build_rank_ops(short_m, 3, short, Shard()))
    _ln(mby["msa_trans_ln"], 4096 * 128, 64, 3)
    _ln(mby["opm_ln"], 4096 * 128, 64, 3)


def test_other_models_do_not_gain_boltz_vectors():
    for mid in ("protenix", "openfold", "alphafold2", "esmfold", "esm2-3b"):
        other = get_model(mid)
        assert all(not L.vec_kind.startswith("bz_") for L in other.layers)
    boltz = get_model("boltz-1")
    assert all(not L.vec_kind.startswith("of_") for L in boltz.layers)
    assert boltz.vec_kind == ""


def test_published_shape_mismatch_is_refused():
    _raises(_bad(lambda r: r["shapes"].pop("pairformer_module.layers.0.attention.norm_s.weight")),
            "pairformer_module.layers.0.attention.norm_s.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "pairformer_module.layers.47.transition_z.norm.weight", [64])),
            "pairformer_module.layers.47.transition_z.norm.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].pop("pairformer_module.layers.0.transition_s.fc1.weight")),
            "boltz-1 pairformer linears are not the published set")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "msa_module.layers.0.tri_mul_out.p_in.weight", [128, 128])),
            "boltz-1 tri_mul_out.p_in shape is not [128, 256] on pair")
    _raises(_bad(lambda r: r["shapes"].pop("confidence_module.msa_module.layers.3.outer_product_mean.norm.weight")),
            "confidence_module.msa_module.layers.3.outer_product_mean.norm.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__("s_norm.weight", [128])),
            "s_norm.weight shape is not")
    _raises(_bad(lambda r: r["shapes"].__setitem__(
        "structure_module.score_model.a_norm.weight", [384])),
            "structure_module.score_model.a_norm.weight shape is not")
    _raises(_bad(lambda r: r["config"]["hyper_parameters"].__setitem__("token_s", 128)),
            "boltz-1 token width is not the published pairformer")

    def shrink_msa(rel):
        rel["stacks"]["msa_module.layers"] = 3
        for key in [k for k in rel["shapes"] if k.startswith("msa_module.layers.3.")]:
            del rel["shapes"][key]

    _raises(_bad(shrink_msa), "boltz-1 msa blocks are not 4")

    def shrink_pair(rel):
        rel["stacks"]["pairformer_module.layers"] = 47
        for key in [k for k in rel["shapes"] if k.startswith("pairformer_module.layers.47.")]:
            del rel["shapes"][key]

    _raises(_bad(shrink_pair), "boltz-1 pairformer blocks are not 48")
    m, wl, ph = _phase(32)
    bad_layer = replace(m.layers[7], pair_linears=m.layers[7].pair_linears[:-1])
    broken = replace(m, layers=m.layers[:7] + (bad_layer,) + m.layers[8:])
    _raises(lambda: build_rank_ops(broken, 7, ph, Shard()),
            "boltz-1 pairformer linears are not the published set")
    bad_diff = replace(m.layers[148], pair_linears=m.layers[148].pair_linears[:-1])
    broken_d = replace(m, layers=m.layers[:148] + (bad_diff,) + m.layers[149:])
    _raises(lambda: build_rank_ops(broken_d, 148, ph, Shard()),
            "boltz-1 diffusion conditioner linears are not the published set")


def test_closed_form_flops_still_hold():
    from tests.test_core_061_3 import _tflop
    t, st, _ = _tflop("boltz-1", seq_len=384)
    assert abs(t - 153.115) < 0.05, t
    assert abs(st["pairformer"] / 4 - 12.207) < 0.01
    t512, _, _ = _tflop("boltz-1", seq_len=512)
    assert t512 > t
