"""0.459: OpenFold bills the published transition output multiply.

MSATransition and PairTransition execute linear_2(x) * mask. One multiply,
no transcendental, no upcast. Width is linear_2's N. Mask construction is
not an op. Published model._mask_trans is False, so Evoformer MSA and every
pair transition multiply by ones. ExtraMSABlock still passes the real MSA
mask. AlphaFold 2 returns the activation without this product. Vector flops
stay 0. The generic .ew tail stays. openfold is outside the first 60 rows.
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
_MASKS = ("msa_trans_mask", "pair_trans_mask")


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


def _idx(m, stack: str) -> int:
    return next(i for i, L in enumerate(m.layers) if L.vec_kind == "of_block" and L.stack == stack)


def _gemm(layer, leaf: str) -> str:
    hits = [ln.name for ln in layer.pair_linears if ln.name == leaf or ln.name.endswith("." + leaf)]
    assert hits == [hits[0]]
    return f"{layer.stack}.{hits[0]}"


def _mask(op, n: int, layer: int, af: str, rep: float = 1) -> None:
    assert op.kind == "vector" and op.layer == layer and op.name in _MASKS
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


def test_version_is_459():
    assert __version__ == "0.473.0"


def test_openfold_transition_output_multiply():
    m, wl, ph = _phase("openfold", 512)
    assert m.id == "openfold" and m.n_layers == 58
    assert m.params() == m.release_params == 93_237_338
    assert m.act_fmt == "bf16" and any("0.459" in n for n in m.notes)
    assert wl.pair.n == 512 and wl.pair.recycles == 4
    n, p = 512, 512 * 512
    msa_rows, xmsa_rows, tmpl = 516 * n, 1024 * n, 4 * p
    evo, extra, tmpl_i = _idx(m, "evoformer"), _idx(m, "extra_msa"), _idx(m, "template")
    assert (evo, extra, tmpl_i) == (6, 2, 0)
    cases = (
        (evo, msa_rows, 256, 1024, p, 128, 512, True),
        (extra, xmsa_rows, 64, 256, p, 128, 512, True),
        (tmpl_i, 0, 0, 0, tmpl, 64, 128, False),
    )
    for li, msa_r, msa_w, msa_h, pair_r, pair_w, pair_h, has_msa in cases:
        layer = m.layers[li]
        assert layer.vec_kind == "of_block" and layer_repeat(layer, ph) == 4
        ops = build_rank_ops(m, li, ph, Shard())
        by = _by(ops)
        if has_msa:
            g = _gemm(layer, "msa_transition.linear_2")
            assert by[g].n == msa_w and by[g].k == msa_h and by[g].m == msa_r
            assert by[g].w_fmt == "fp32" and by[g].flops == 2 * msa_r * msa_h * msa_w
            assert by[g + ".ew"].vec == msa_r * (msa_h + msa_w) * 4
            _mask(by["msa_trans_mask"], msa_r * msa_w, li, "bf16")
            _order(ops, g, "msa_trans_mask")
            _order(ops, _gemm(layer, "msa_transition.linear_1"), "msa_trans_relu")
        else:
            assert "msa_trans_mask" not in by and "msa_trans_relu" not in by
        g = _gemm(layer, "pair_transition.linear_2")
        assert by[g].n == pair_w and by[g].k == pair_h and by[g].m == pair_r
        assert by[g].w_fmt == "fp32" and by[g].flops == 2 * pair_r * pair_h * pair_w
        assert by[g + ".ew"].vec == pair_r * (pair_h + pair_w) * 4
        _mask(by["pair_trans_mask"], pair_r * pair_w, li, "bf16")
        _order(ops, g, "pair_trans_mask")
        _order(ops, _gemm(layer, "pair_transition.linear_1"), "pair_trans_relu")
        assert all(o.name not in _BANNED for o in ops)
        assert len([o for o in ops if o.name in _MASKS]) == (2 if has_msa else 1)
    total = 0
    request = 0
    for li, layer in enumerate(m.layers):
        ops = build_rank_ops(m, li, ph, Shard())
        masks = [o for o in ops if o.name in _MASKS]
        if layer.vec_kind != "of_block":
            assert masks == []
            continue
        total += sum(o.vec for o in masks)
        request += layer_repeat(layer, ph) * sum(o.vec for o in masks)
    one = (48 * (msa_rows * 256 + p * 128) + 4 * (xmsa_rows * 64 + p * 128) + 2 * (tmpl * 64))
    assert total == one == 5_259_657_216
    assert request == 4 * one == 21_038_628_864


def test_alphafold2_does_not_multiply():
    m, _wl, ph = _phase("alphafold2", 512)
    assert not any("0.459" in n for n in m.notes)
    for stack in ("evoformer", "extra_msa", "template"):
        li = _idx(m, stack)
        ops = build_rank_ops(m, li, ph, Shard())
        assert all(o.name not in _MASKS for o in ops)
        layer = m.layers[li]
        leaf = "pair_transition.linear_2"
        _order(ops, _gemm(layer, leaf), _gemm(layer, leaf) + ".ew")
    for li, layer in enumerate(m.layers):
        if layer.stack == "structure":
            ops = build_rank_ops(m, li, ph, Shard())
            assert all(o.name not in _MASKS for o in ops)


def test_batch_dap_fp8_and_short():
    m, wl, ph = _phase("openfold", 512)
    evo, extra = _idx(m, "evoformer"), _idx(m, "extra_msa")
    two = Phase("full", 2, wl.tokens, 0, pair=wl.pair)
    b2 = _by(build_rank_ops(m, evo, two, Shard()))
    _mask(b2["msa_trans_mask"], 2 * 516 * 512 * 256, evo, "bf16")
    _mask(b2["pair_trans_mask"], 2 * 512 * 512 * 128, evo, "bf16")
    dap = _by(build_rank_ops(m, evo, ph, Shard(sp=2)))
    _mask(dap["msa_trans_mask"], 258 * 512 * 256, evo, "bf16")
    _mask(dap["pair_trans_mask"], 256 * 512 * 128, evo, "bf16")
    assert dap["evoformer.core.msa_transition.linear_2"].m == 258 * 512
    assert dap["msa_trans_mask"].replicated == 1 and dap["pair_trans_mask"].replicated == 1
    dx = _by(build_rank_ops(m, extra, ph, Shard(sp=2)))
    _mask(dx["msa_trans_mask"], 512 * 512 * 64, extra, "bf16")
    _mask(dx["pair_trans_mask"], 256 * 512 * 128, extra, "bf16")
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, evo, ph, Shard()))
    _mask(op["msa_trans_mask"], 516 * 512 * 256, evo, "fp8")
    _mask(op["pair_trans_mask"], 512 * 512 * 128, evo, "fp8")
    assert op["msa_trans_ln"].a_fmt == "fp32"
    assert op["evoformer.core.msa_transition.linear_2"].w_fmt == "fp32"
    assert op["evoformer.core.msa_transition.linear_2"].a_fmt == "fp8"
    short_m, _short_wl, short = _phase("openfold", 128)
    sby = _by(build_rank_ops(short_m, evo, short, Shard()))
    _mask(sby["msa_trans_mask"], 516 * 128 * 256, evo, "bf16")
    _mask(sby["pair_trans_mask"], 128 * 128 * 128, evo, "bf16")


def test_non_vector_schedule_fingerprints_and_other_models():
    m, _wl, ph = _phase("openfold", 128)
    cleared = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers))
    for shard in (Shard(), Shard(sp=2)):
        assert _nv(m, ph, shard) == _nv(cleared, ph, shard)
    evo = _idx(m, "evoformer")
    bare = build_rank_ops(cleared, evo, ph, Shard())
    assert all(o.name not in _MASKS and "relu" not in o.name for o in bare)
    for mid in ("alphafold2", "boltz-1", "protenix", "esmfold", "esm2-3b",
                "qwen3-8b", "qwen3-32b", "qwen3-1.7b", "llama-3.1-8b"):
        other = get_model(mid)
        assert not any("0.459" in n for n in other.notes)
    af, _af_wl, af_ph = _phase("alphafold2", 64)
    for li, _L in enumerate(af.layers):
        assert all(o.name not in _MASKS for o in build_rank_ops(af, li, af_ph, Shard()))
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert ids.index("openfold") >= 60 and "openfold" not in ids[:60]
    from tests.test_core_0100 import _fp
    from tests.test_core_coverage import _FULL_0651
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"
    assert _fp({}) == "9a66da621b20a5f8"
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

    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "f8346734e98af4a2"
    assert "openfold" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"
