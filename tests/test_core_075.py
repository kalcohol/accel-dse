"""0.75: per-op shapes of the structure / video models pinned against their reference code / configs, and the
parameter-holder-layer fix (no phantom block ops)."""
from accel_dse.core.catalog import get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.scenario import Workload


def _ops(mid, **kw):
    m = get_model(mid)
    wl = resolve_workload(m, Workload(**kw))
    ph = Phase("full", 1, wl.tokens, wl.ctx, frames=wl.frames, aux=wl.aux, pair=wl.pair)
    return m, wl, ph


def _find(m, ph, stack, name):
    for i, L in enumerate(m.layers):
        if L.stack == stack:
            for o in build_rank_ops(m, i, ph):
                if o.name == name:
                    return (o.m, o.k, o.n, o.count)
    raise AssertionError(f"{stack}/{name} not found")


def test_parameter_holder_layers_emit_no_ops():
    for mid in ("esmfold", "boltz-1", "protenix"):
        m, wl, ph = _ops(mid)
        n = 0
        for i, L in enumerate(m.layers):
            if not (L.attn_linears or L.ffn_linears or L.cross_linears or L.cross is not None or L.core.kind != "none"
                    or L.pair_linears or L.pair_cores):
                assert build_rank_ops(m, i, ph) == [], (mid, i)
                n += 1
        assert n >= 2, mid
        # and no zero-sized attention ops anywhere
        for i in range(m.n_layers):
            for o in build_rank_ops(m, i, ph):
                if o.kind == "attn":
                    assert o.m > 0 and o.k > 0 and o.n > 0 and o.count > 0, (mid, i, o.name)


def test_openfold_shapes_match_af2_supplement():
    m, wl, ph = _ops("openfold", seq_len=256, msa=128, recycles=1)
    N, S = 256, 132                                     # 128 MSA clusters + 4 template torsion rows
    assert _find(m, ph, "evoformer", "evoformer.row_att6.qk") == (N, 32, N, S * 8)       # row attn 8 × 32
    assert _find(m, ph, "evoformer", "evoformer.col_att5.qk") == (S, 32, S, N * 8)       # column attn 8 × 32
    assert _find(m, ph, "evoformer", "evoformer.opm0") == (N * 32, S, N * 32, 1)         # OPM c = 32
    assert _find(m, ph, "evoformer", "evoformer.trimul3") == (N, N, N, 128)              # c_z = 128
    assert _find(m, ph, "evoformer", "evoformer.tri_att1.qk") == (N, 32, N, N * 4)       # 4 × 32
    assert _find(m, ph, "extra_msa", "extra_msa.row_att6.qk") == (N, 8, N, 1024 * 8)     # c_e 64, 8 × 8
    assert _find(m, ph, "extra_msa", "extra_msa.col_att5.qk") == (1, 8, 1024, N * 8)     # global column attn


def test_esmfold_boltz_protenix_shapes():
    m, wl, ph = _ops("esmfold", seq_len=256, recycles=1)
    assert _find(m, ph, "trunk", "trunk.seq_att0.qk") == (256, 32, 256, 32)             # 1024 / 32
    assert _find(m, ph, "structure", "structure.seq_att0.qk") == (256, 28, 256, 12)      # IPA 16 + 4·3
    assert _find(m, ph, "structure", "structure.seq_att0.pv") == (256, 256, 168, 12)     # 16 + 8·3 + 128
    m, wl, ph = _ops("boltz-1")
    assert _find(m, ph, "pairformer", "pairformer.seq_att0.qk") == (512, 24, 512, 16)
    assert _find(m, ph, "diffusion", "diffusion.seq_att0.qk") == (512, 48, 512, 16)
    assert _find(m, ph, "diff_atom_enc", "diff_atom_enc.local_att0.qk") == (32, 32, 128, 4096 // 32 * 4)
    assert _find(m, ph, "msa", "msa.pwa1") == (512, 512, 4096 * 32, 8)
    m, wl, ph = _ops("protenix")
    assert _find(m, ph, "msa", "msa.pwa0") == (512, 512, 2048 * 8, 8)                   # AF3 PWA c = 8
    assert _find(m, ph, "diffusion", "diffusion.seq_att0.qk") == (512, 48, 512, 16 * 5)  # 5 samples batched


def test_video_dit_shapes_match_reference_configs():
    m, wl, ph = _ops("wan2.1-1.3b")
    assert wl.tokens == 21 * 30 * 52 and wl.ctx == 512
    assert _find(m, ph, m.layers[0].stack, "qk") == (32760, 128, 32760, 12)
    assert _find(m, ph, m.layers[0].stack, "mlp.up")[2] == 8960
    m, wl, ph = _ops("hunyuanvideo")
    assert wl.tokens == 33 * 45 * 80 + 256
    dual = [o for o in build_rank_ops(m, 0, ph) if o.name == "attn.img_mod"][0]
    single = [o for o in build_rank_ops(m, 59, ph) if o.name in ("attn.mod", "mlp.fused_out")]
    assert (dual.k, dual.n) == (3072, 6 * 3072)
    assert sorted((o.k, o.n) for o in single) == [(3072, 3 * 3072), (3072 + 12288, 3072)]
    m, wl, ph = _ops("ltx-video")
    assert wl.tokens == 21 * 16 * 22 and wl.ctx == 128
    m, wl, ph = _ops("opensora-stdit3")
    assert wl.tokens == 30 * 80 * 45 and wl.ctx == 300
    sp = [o for o in build_rank_ops(m, 0, ph) if o.name == "qk"][0]
    tm = [o for o in build_rank_ops(m, 1, ph) if o.name == "qk"][0]
    assert (sp.m, sp.count) == (3600, 16 * 30) and (tm.m, tm.count) == (30, 16 * 3600)
