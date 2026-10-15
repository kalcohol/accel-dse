"""0.106: attention-mask mode (Scenario.attn_mode, bidir = non-causal / no KV cache); genomics models (Carbon)."""
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.model import with_attn_mode
from accel_dse.core.scenario import Scenario


def _attn(m, ph):
    ops = build_rank_ops(m, 0, ph)
    return {o.name: o for o in ops}


def test_bidir_attention_flops_are_twice_causal():
    for mid in ("qwen3-8b", "llama-3.1-8b", "carbon-3b"):
        m = get_model(mid)
        mb = with_attn_mode(m, "bidir")
        for S in (1024, 16384):
            a, b = _attn(m, Phase("prefill", 2, S, 0)), _attn(mb, Phase("prefill", 2, S, 0))
            for k in ("qk", "pv"):
                assert abs(b[k].flops / a[k].flops - 2.0) < 1e-12, (mid, S, k)
            assert a["softmax"].kv_write > 0 and b["softmax"].kv_write == 0.0      # no KV cache written
            # projections / FFN untouched
            for k in a:
                if k not in ("qk", "pv", "softmax"):
                    assert a[k].flops == b[k].flops, (mid, k)


def test_reference_mode_is_default_and_identical():
    s = Scenario.from_dict({"model": "qwen3-8b", "serving": {"phase": "prefill", "batch": 1, "prompt": 4096}})
    assert s.attn_mode == "reference" and "attn_mode" not in s.to_dict()
    r0 = evaluate(s)
    r1 = evaluate(Scenario.from_dict({**s.to_dict(), "attn_mode": "reference"}))
    assert r0.summary() == r1.summary()
    rb = evaluate(Scenario.from_dict({**s.to_dict(), "attn_mode": "bidir"}))
    assert rb.ttft > r0.ttft and any("bidir" in w for w in rb.warnings)
    assert with_attn_mode(get_model("qwen3-8b"), "reference") is get_model("qwen3-8b")


def test_bidir_rejected_for_decode_ignored_for_encoders():
    try:
        evaluate(Scenario.from_dict({"model": "qwen3-8b", "attn_mode": "bidir"}))
        raise AssertionError("decode with bidir must raise")
    except ValueError as e:
        assert "bidir" in str(e)
    r = evaluate(Scenario.from_dict({"model": "esm2-650m", "attn_mode": "bidir"}))
    assert any("attn_mode" in w for w in r.warnings)


def test_dense_causal_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "e2176af2610c144c"
    assert _fp({}) == "392692e6937f69ef"


def test_carbon_params_match_release_and_dtypes():
    for mid in ("carbon-a-1.2b", "carbon-500m", "carbon-3b", "carbon-8b"):
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0, mid
        assert all(f.fmt == "bf16" for _, f in m.formats) and m.act_fmt == "bf16", mid   # card: BF16
    a = get_model("carbon-a-1.2b")
    assert a.domain == "genomics" and not a.kv_cache and not a.vocab_head
    assert {l.core.causal for l in a.layers} == {False}
    c = a.layers[0].core
    assert (c.n_q, c.n_kv, c.qk_dim, a.n_layers, a.hidden, a.vocab) == (32, 4, 64, 26, 2048, 4128)
    assert [l.n for l in a.io_post] == [12, 12]                 # two strand heads: 6 bases x 2 classes per token
    for mid, L, h in (("carbon-500m", 28, 1024), ("carbon-3b", 30, 3072), ("carbon-8b", 32, 4096)):
        m = get_model(mid)
        assert m.domain == "llm" and m.kv_cache and m.n_layers == L and m.hidden == h
        assert {l.core.causal for l in m.layers} == {True}


def test_carbon_a_full_attention_is_twice_causal_and_no_mlm_head():
    from dataclasses import replace
    from accel_dse.core.ir import full_io_ops, Shard
    a = get_model("carbon-a-1.2b")
    ac = replace(a, layers=tuple(replace(l, core=replace(l.core, causal=True)) for l in a.layers))
    ph = Phase("full", 1, 16384, 0)
    x, y = _attn(a, ph), _attn(ac, ph)
    for k in ("qk", "pv"):
        assert abs(x[k].flops / y[k].flops - 2.0) < 1e-12
    names = [o.name for o in full_io_ops(a, ph, Shard(), "post")]
    assert "mlm_decoder" not in names and "io.score_0" in names and "io.score_1" in names


def test_carbon_a_window_workload_and_api():
    import accel_dse.api as A
    r = evaluate(Scenario.from_dict({"model": "carbon-a-1.2b"}))
    g = r.summary()["gen"]
    assert r.fits and g["unit"] == "window"
    w = g["workload"]
    assert (w["seq_len"], w["kmer"], w["bp_per_window"], w["overlap_tokens"], w["new_bp_per_window"]) == \
        (16384, 6, 98304, 1024, 92160)
    assert abs(g["bp_per_s_card"] - g["windows_per_s_card"] * 92160) < 1e-6
    assert "genomics" in A.api_health()["domains"]
    ms = {m["id"]: m for m in A.api_models()["models"]}
    assert ms["carbon-a-1.2b"]["domain"] == "genomics" and ms["carbon-3b"]["domain"] == "llm"
    assert {ms[i]["provider"] for i in ("carbon-a-1.2b", "carbon-500m", "carbon-3b", "carbon-8b")} == {"hfbio"}
