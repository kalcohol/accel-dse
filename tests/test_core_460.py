"""0.460: OpenFold and AlphaFold 2 bill the published template output multiply.

TemplateEmbedder and TemplateEmbedding execute (sum(template_mask) > 0) times
the pointwise-attention output. One multiply, no transcendental, no upcast.
Width is linear_o's N (c_z) on the pair grid. Mask construction is not an op.
AlphaFold 2's Transition still returns the activation without a product.
Vector flops stay 0. The generic .ew tail stays. Both rows sit outside the
first 60 catalog entries.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.domain import resolve_workload
from accel_dse.core.ir import Phase, Shard, build_rank_ops, layer_repeat
from accel_dse.core.scenario import Workload


_BANNED = ("gelu", "silu", "sigmoid", "softplus", "clamp", "mask_mul")
_NAME = "tmpl_out_mask"
_GEMM = "embed.template_pointwise_att.mha.linear_o"


def _by(ops):
    found = {}
    for op in ops:
        assert op.name not in found, op.name
        found[op.name] = op
    return found


def _phase(mid: str, seq_len: int, batch: int = 1):
    m = get_model(mid)
    wl = resolve_workload(m, Workload(seq_len=seq_len))
    ph = Phase("full", batch, wl.tokens, 0, pair=wl.pair)
    return m, wl, ph


def _embed(m) -> int:
    hits = [i for i, L in enumerate(m.layers) if L.vec_kind == "of_embed"]
    assert hits == [hits[0]]
    return hits[0]


def _mask(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name == _NAME
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == rep


def _order(ops, gemm: str, nxt: str) -> None:
    names = [o.name for o in ops]
    i = names.index(gemm)
    assert names[i + 1] == nxt
    if nxt != gemm + ".ew":
        assert names[i + 2] == gemm + ".ew"


def _nv(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_460():
    assert __version__ == "0.460.0"


def test_template_output_multiply():
    one = 512 * 512 * 128
    for mid in ("openfold", "alphafold2"):
        m, wl, ph = _phase(mid, 512)
        assert m.id == mid and m.n_layers == 58
        assert m.params() == m.release_params == 93_237_338
        assert m.act_fmt == "bf16" and any("0.460" in n for n in m.notes)
        assert wl.pair.n == 512 and wl.pair.tmpl == 4 and wl.pair.recycles == 4
        li = _embed(m)
        layer = m.layers[li]
        assert layer.vec_kind == "of_embed" and layer.repeat == "recycle" and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        g = by[_GEMM]
        assert (g.m, g.k, g.n) == (512 * 512, 64, 128)
        assert g.w_fmt == "fp32" and g.flops == 2 * (512 * 512) * 64 * 128
        assert by[_GEMM + ".ew"].vec == (512 * 512) * (64 + 128) * 4
        _mask(by[_NAME], one, li, "bf16")
        _order(ops, _GEMM, _NAME)
        names = [o.name for o in ops]
        k = names.index("embed.template_pointwise_att.mha.linear_k")
        assert names[k - 1] == "tmpl_out_ln"
        assert by["tmpl_out_ln"].vec == float(4 * (512 * 512) * 64 * 4)
        assert all(o.name not in _BANNED for o in ops)
        assert len([o for o in ops if o.name == _NAME]) == 1
        total = 0
        request = 0
        for j, lyr in enumerate(m.layers):
            found = [o for o in build_rank_ops(m, j, ph, Shard()) if o.name == _NAME]
            if lyr.vec_kind != "of_embed":
                assert found == []
                continue
            assert len(found) == 1
            total += found[0].vec
            request += layer_repeat(lyr, ph) * found[0].vec
        assert total == one == 33_554_432
        assert request == 4 * one == 134_217_728
    af, _af_wl, af_ph = _phase("alphafold2", 512)
    evo = next(i for i, L in enumerate(af.layers) if L.vec_kind == "of_block" and L.stack == "evoformer")
    af_ops = build_rank_ops(af, evo, af_ph, Shard())
    assert all(o.name not in ("msa_trans_mask", "pair_trans_mask") for o in af_ops)
    of, _of_wl, of_ph = _phase("openfold", 512)
    of_evo = next(i for i, L in enumerate(of.layers) if L.vec_kind == "of_block" and L.stack == "evoformer")
    of_ops = build_rank_ops(of, of_evo, of_ph, Shard())
    assert {o.name for o in of_ops if o.name in ("msa_trans_mask", "pair_trans_mask")} == {
        "msa_trans_mask", "pair_trans_mask"}


def test_batch_dap_fp8_and_short():
    for mid in ("openfold", "alphafold2"):
        m, wl, ph = _phase(mid, 512)
        li = _embed(m)
        two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
        b2 = _by(build_rank_ops(m, li, two, Shard()))
        _mask(b2[_NAME], 2 * 512 * 512 * 128, li, "bf16")
        dap = _by(build_rank_ops(m, li, ph, Shard(sp=2)))
        _mask(dap[_NAME], 256 * 512 * 128, li, "bf16")
        assert dap[_GEMM].m == 256 * 512 and dap[_NAME].replicated == 1
        fp8 = replace(m, act_fmt="fp8")
        op = _by(build_rank_ops(fp8, li, ph, Shard()))
        _mask(op[_NAME], 512 * 512 * 128, li, "fp8")
        assert op["tmpl_out_ln"].a_fmt == "fp32"
        assert op[_GEMM].w_fmt == "fp32" and op[_GEMM].a_fmt == "fp8"
        short_m, _short_wl, short = _phase(mid, 128)
        sby = _by(build_rank_ops(short_m, li, short, Shard()))
        _mask(sby[_NAME], 128 * 128 * 128, li, "bf16")
        assert sby[_GEMM].m == 128 * 128


def test_non_vector_schedule_fingerprints_and_other_models():
    for mid in ("openfold", "alphafold2"):
        m, _wl, ph = _phase(mid, 128)
        cleared = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers))
        for shard in (Shard(), Shard(sp=2)):
            assert _nv(m, ph, shard) == _nv(cleared, ph, shard)
        bare = build_rank_ops(cleared, _embed(m), ph, Shard())
        assert all(o.name != _NAME for o in bare)
    for mid in ("boltz-1", "protenix", "esmfold", "esm2-3b",
                "qwen3-8b", "qwen3-32b", "qwen3-1.7b", "llama-3.1-8b"):
        other = get_model(mid)
        assert not any("0.460" in n for n in other.notes)
    for mid in ("boltz-1", "protenix", "esmfold"):
        other, _owl, oph = _phase(mid, 512)
        for li, _L in enumerate(other.layers):
            assert all(o.name != _NAME for o in build_rank_ops(other, li, oph, Shard()))
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("openfold") >= 60 and ids.index("alphafold2") >= 60
    assert "openfold" not in ids[:60] and "alphafold2" not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "1e15ae85430a04f3"
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario

    def fpx(top):
        h = hashlib.sha256()
        for e in entries()[:60]:
            if e["id"].startswith("deepseek-v4"):
                continue
            for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}},
                          {"layout": {"tp": 2}}):
                try:
                    h.update(json.dumps(evaluate(Scenario.from_dict({"model": e["id"], **top, **extra})).summary(),
                                        sort_keys=True, default=str).encode())
                except Exception as ex:  # noqa: BLE001
                    h.update(type(ex).__name__.encode())
        return h.hexdigest()[:16]

    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "8a5fa1d0881da387"
    assert "openfold" not in _FULL_0651 and "alphafold2" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
