"""0.444: Evo 2 bills published layer-0 erf-GELU and the gated-MLP product.

vortex ParallelGatedMLP: y = l3(act(l1(z)) * l2(z)). mlp_activation gelu is
F.gelu (approximate='none'). evo2_style_activations replaces that with Identity
when layer_idx > 0. The product is one multiply on every layer. Vortex RMSNorm
keeps eps outside the norm, so the generic norm tail stays. Vector flops stay 0.
"""
from __future__ import annotations

import json
from dataclasses import replace

from accel_dse import __version__
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import _build_hyena, from_release, load_release


_SIZES = {
    "evo2-40b": ("arcinstitute/evo2_40b", 8192, 22528, 50, 41_117_574_144),
    "evo2-7b": ("arcinstitute/evo2_7b", 4096, 11264, 32, 6_581_722_880),
    "evo2-1b": ("arcinstitute/evo2_1b_base", 1920, 5120, 25, 1_107_989_760),
}
_BANNED = ("act", "silu", "relu", "sigmoid", "softplus", "clamp", "quick_gelu", "gelu_new", "mask_mul")


def _by(ops):
    found = {}
    for op in ops:
        if op.name in found and op.kind == "vector" and not op.name.endswith(".ew"):
            raise AssertionError(op.name)
        found[op.name] = op
    return found


def _gelu(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == n * 5 and op.sfu == float(n)
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _mul(op, n: int, layer: int, af: str) -> None:
    assert op.kind == "vector" and op.layer == layer
    assert op.vec == float(n) and op.sfu == 0.0
    assert op.a_fmt == af and op.w_params == 0 and op.flops == 0
    assert op.replicated == 1


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(needle)


def _bad(edit):
    rel = json.loads(json.dumps(load_release("arcinstitute/evo2_7b")))
    edit(rel["config"])
    return lambda: from_release("evo2-7b", "arcinstitute/evo2_7b", rel)


def _sched(model, ph, shard):
    out = []
    for li, _L in enumerate(model.layers):
        for o in build_rank_ops(model, li, ph, shard):
            if o.kind != "vector":
                out.append((li, o.name, o.kind, o.m, o.k, o.n, o.flops, o.w_fmt, o.a_fmt, o.replicated))
    return out


def test_version_is_444():
    assert __version__ == "0.470.0"


def test_three_sizes_bill_layer0_gelu_and_later_product_only():
    for mid, (hf, h, inner, depth, params) in _SIZES.items():
        m = from_release(mid, hf)
        assert m.params() == m.release_params == params
        assert m.n_layers == depth and m.hidden == h and m.act_fmt == "bf16"
        assert m.vec_kind == ""
        assert m.layers[0].vec_kind == "evo2_gelu"
        assert all(L.vec_kind == "evo2_id" for L in m.layers[1:])
        assert all(L.ffn.gated and L.ffn.d_ff == inner for L in m.layers)
        t = 4096
        ph = Phase("prefill", 1, t, 0)
        ops = build_rank_ops(m, 0, ph, Shard())
        by = _by(ops)
        _gelu(by["gelu"], t * inner, 0, "bf16")
        _mul(by["mlp_mul"], t * inner, 0, "bf16")
        assert by["attn_norm"].vec == t * h * 4 and by["attn_norm"].sfu == float(t)
        assert by["ffn_norm"].vec == t * h * 4 and by["ffn_norm"].sfu == float(t)
        assert by["attn_norm"].a_fmt == "bf16" and by["ffn_norm"].a_fmt == "bf16"
        up, down = by["mlp.gate_up"], by["mlp.down"]
        assert (up.m, up.k, up.n) == (t, h, 2 * inner)
        assert (down.m, down.k, down.n) == (t, inner, h)
        assert up.w_fmt == "bf16" and up.w_bits == 16.0 and up.a_fmt == "bf16"
        assert up.flops == 2 * t * h * (2 * inner)
        assert down.flops == 2 * t * inner * h
        names = [o.name for o in ops]
        assert names.index("ffn_norm") < names.index("mlp.gate_up") < names.index("gelu")
        assert names.index("gelu") < names.index("mlp_mul") < names.index("mlp.down")
        assert all(o.name not in _BANNED for o in ops)
        assert all(o.flops == 0 for o in ops if o.kind == "vector")
        last = build_rank_ops(m, depth - 1, ph, Shard())
        lby = _by(last)
        assert "gelu" not in lby and "act" not in lby
        _mul(lby["mlp_mul"], t * inner, depth - 1, "bf16")
        assert lby["mlp.gate_up"].n == 2 * inner and lby["mlp.down"].k == inner
        lnames = [o.name for o in last]
        assert lnames.index("mlp.gate_up") < lnames.index("mlp_mul") < lnames.index("mlp.down")
        head = _by(head_ops(m, ph, Shard()))
        assert head["final_norm"].vec == t * h * 4 and head["final_norm"].sfu == 0.0
        assert head["final_norm"].a_fmt == "bf16" and head["final_norm"].layer == depth


def test_decode_batch_tp_and_fp8():
    m = get_model("evo2-7b")
    h, inner = 4096, 11264
    dec = build_rank_ops(m, 0, Phase("decode", 1, 1, 4096), Shard())
    dby = _by(dec)
    _gelu(dby["gelu"], inner, 0, "bf16")
    _mul(dby["mlp_mul"], inner, 0, "bf16")
    assert dby["mlp.gate_up"].m == 1 and dby["ffn_norm"].sfu == 1.0
    later = _by(build_rank_ops(m, 31, Phase("decode", 2, 1, 8192), Shard()))
    assert "gelu" not in later
    _mul(later["mlp_mul"], 2 * inner, 31, "bf16")
    assert later["attn_norm"].vec == 2 * h * 4
    two = _by(build_rank_ops(m, 0, Phase("prefill", 2, 4096, 0), Shard()))
    _gelu(two["gelu"], 2 * 4096 * inner, 0, "bf16")
    _mul(two["mlp_mul"], 2 * 4096 * inner, 0, "bf16")
    assert two["mlp.gate_up"].m == 8192 and two["mlp.down"].flops == 2 * 8192 * inner * h
    sh = Shard(tp=2)
    loc = _by(build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh))
    half = inner // 2
    _gelu(loc["gelu"], 4096 * half, 0, "bf16")
    _mul(loc["mlp_mul"], 4096 * half, 0, "bf16")
    assert loc["mlp.gate_up"].n == inner and loc["mlp.down"].k == half
    assert loc["mlp.gate_up"].replicated == 1 and loc["ffn_norm"].vec == 4096 * h * 4
    names = [o.name for o in build_rank_ops(m, 0, Phase("prefill", 1, 4096, 0), sh)]
    assert names.index("mlp.down") < names.index("mlp_allreduce")
    fp8 = replace(m, act_fmt="fp8")
    op = _by(build_rank_ops(fp8, 0, Phase("prefill", 1, 4096, 0), Shard()))
    _gelu(op["gelu"], 4096 * inner, 0, "fp8")
    _mul(op["mlp_mul"], 4096 * inner, 0, "fp8")
    assert op["attn_norm"].a_fmt == "bf16" and op["ffn_norm"].a_fmt == "bf16"
    assert op["mlp.gate_up"].a_fmt == "fp8" and op["mlp.gate_up"].w_fmt == "bf16"
    assert op["mlp.down"].w_bits == 16.0
    tail = _by(build_rank_ops(fp8, 1, Phase("decode", 1, 1, 128), Shard()))
    assert "gelu" not in tail and tail["mlp_mul"].a_fmt == "fp8" and tail["mlp.down"].w_fmt == "bf16"


def test_non_vector_schedule_and_other_models():
    m = get_model("evo2-7b")
    ph = Phase("prefill", 1, 256, 0)
    bare = replace(m, layers=tuple(replace(L, vec_kind="") for L in m.layers))
    for shard in (Shard(), Shard(tp=2)):
        assert _sched(m, ph, shard) == _sched(bare, ph, shard)
    for mid in ("llama-3.1-8b", "qwen3-8b", "carbon-a-1.2b", "carbon-8b", "protenix", "esm2-3b"):
        other = get_model(mid)
        assert all(L.vec_kind not in ("evo2_gelu", "evo2_id") for L in other.layers)
    carbon = _by(build_rank_ops(get_model("carbon-8b"), 0, Phase("decode", 1, 1, 128), Shard()))
    assert "gelu" not in carbon and carbon["act"].sfu == 14336 * 2
    ids = [e["id"] for e in entries()]
    assert ids[:60][-2:] == ["wan2.1-14b", "wan2.1-1.3b"]
    assert "evo2-40b" not in ids[:60] and "evo2-7b" not in ids[:60] and "evo2-1b" not in ids[:60]
    from tests.test_core_coverage import _FULL_0651
    assert "evo2-40b" not in _FULL_0651 and "evo2-7b" not in _FULL_0651
    assert _FULL_0651["esmfold"] == "3bbede544308"
    assert _FULL_0651["esm2-3b"] == "34f7544287cd"
    assert _FULL_0651["wan2.1-1.3b"] == "309919c00604"


def test_published_flags_and_linear_shape_are_refused():
    _raises(_bad(lambda c: c.__setitem__("mlp_activation", "silu")), "mlp_activation is not gelu")
    _raises(_bad(lambda c: c.__setitem__("mlp_activation", "gelu_new")), "mlp_activation is not gelu")
    _raises(_bad(lambda c: c.__setitem__("evo2_style_activations", False)), "evo2_style_activations is not true")
    _raises(_bad(lambda c: c.__setitem__("use_flash_rmsnorm", True)), "use_flash_rmsnorm is not false")
    _raises(_bad(lambda c: c.__setitem__("intermediate_size", 4096)), "intermediate_size is not inner_mlp_size")
    rel = json.loads(json.dumps(load_release("arcinstitute/evo2_7b")))
    rel["config"]["model_type"] = "llama"
    _raises(lambda: _build_hyena(rel["config"]), "model_type is not stripedhyena2")
    m = get_model("evo2-7b")
    ph = Phase("prefill", 1, 8, 0)
    bad = replace(m.layers[0], ffn_linears=m.layers[0].ffn_linears[:1])
    broken = replace(m, layers=(bad,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken, 0, ph, Shard()), "evo2 mlp is not gate_up then down")
    ungated = replace(m.layers[0], ffn=replace(m.layers[0].ffn, gated=False))
    broken_g = replace(m, layers=(ungated,) + m.layers[1:])
    _raises(lambda: build_rank_ops(broken_g, 0, ph, Shard()), "evo2 mlp is not gate_up then down")


def test_ffn_gemm_flops_match_the_closed_form():
    m = get_model("evo2-40b")
    t, h, inner = 4096, 8192, 22528
    ph = Phase("prefill", 1, t, 0)
    hand = m.n_layers * (2 * t * h * (2 * inner) + 2 * t * inner * h)
    got = 0.0
    for li in range(m.n_layers):
        for op in build_rank_ops(m, li, ph, Shard()):
            if op.name in ("mlp.gate_up", "mlp.down"):
                got += op.flops
            if op.kind == "vector":
                assert op.flops == 0
    assert got == hand
