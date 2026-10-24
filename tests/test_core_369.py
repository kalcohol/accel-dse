"""0.369: Gemma-4 E2B / E4B (KV alias, PLE, double-wide MLP, gemma4_audio) and the 0.368 KV-head fix.

Counts are the release headers (2026-10-24) after per_dim_scale is kept as a parameter.
"""
from __future__ import annotations

import importlib.util

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, embed_ops
from accel_dse.core.memplan import cache_new_bytes, stage_storage
from accel_dse.core.model import _build_gemma4
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, image_grid, vision_ops

# id, layers, hidden, share, inter, wide, cached, ple
ROWS = (
    ("gemma-4-e4b-it", 42, 2560, 18, 10240, False, 55_055_616, 256),
    ("gemma-4-e4b", 42, 2560, 18, 10240, False, 55_055_616, 256),
    ("gemma-4-e2b-it", 35, 1536, 20, 6144, True, 18_880_512, 256),
    ("gemma-4-e2b", 35, 1536, 20, 6144, True, 18_880_512, 256),
)
_LLM = {"gemma-4-e2b-it": 4_647_449_891, "gemma-4-e2b": 4_647_449_891,
        "gemma-4-e4b-it": 7_518_069_034, "gemma-4-e4b": 7_518_069_034}
_VIS = {"gemma-4-e2b-it": 475_729_088, "gemma-4-e2b": 475_729_088,
        "gemma-4-e4b-it": 478_088_384, "gemma-4-e4b": 478_088_384}


def _raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _softmax(model, li, kind, ctx, q):
    ops = build_rank_ops(model, li, Phase(kind, 1, q, ctx), Shard())
    return next(o for o in ops if o.name == "softmax")


def test_e2b_e4b_param_check_and_structure():
    ids = [e["id"] for e in list_models() if e["family"] == "Gemma-4"]
    assert ids[-len(ROWS):] == [r[0] for r in ROWS]
    for mid, layers, hidden, n_share, inter, wide, cached, ple in ROWS:
        m = get_model(mid)
        pc = m.param_check()
        assert pc["rel_err"] == 0.0 and m.params() == _LLM[mid] == m.release_params, mid
        assert m.vision is not None and m.vision.params == _VIS[mid] and m.vision.extra == 0, mid
        assert m.coverage == "full" and m.coverage_reasons == (), mid
        assert m.n_layers == layers and m.hidden == hidden and m.tie_embeddings and m.vocab == 262144, mid
        assert m.act_fmt == "bf16" and m.fmt("attn").bits == 16.0 and m.fmt("other").bits == 16.0, mid
        assert m.cached_params == cached and m.lookup_bits == 16.0, mid
        assert m.lookup_params == 262144 * layers * ple and m.ple_packed == layers * ple, mid
        assert m.final_norm_params == hidden + ple, mid
        donor = [L for L in m.layers if not L.kv_alias]
        alias = [L for L in m.layers if L.kv_alias]
        assert len(alias) == n_share and len(donor) == layers - n_share, mid
        assert all(L.ple_in == ple for L in m.layers), mid
        assert all(any(lin.name == "k" for lin in L.attn_linears) and any(lin.name == "v" for lin in L.attn_linears)
                   for L in donor), mid
        assert all(not any(lin.name in ("k", "v") for lin in L.attn_linears) for L in alias), mid
        assert any(lin.name == "ple_proj" and lin.n == layers * ple and lin.role == "other"
                   for lin in m.layers[0].attn_linears), mid
        assert all(any(lin.name == "ple_gate" and lin.role == "other" for lin in L.ffn_linears) for L in m.layers), mid
        assert all(any(lin.name == "ple_up" and lin.role == "other" for lin in L.ffn_linears) for L in m.layers), mid
        if wide:
            assert m.layers[0].ffn.d_ff == inter and m.layers[n_share and (layers - n_share)].ffn.d_ff == inter * 2, mid
            assert all(L.ffn.d_ff == inter * 2 for L in alias), mid
            assert all(L.ffn.d_ff == inter for L in donor), mid
        else:
            assert all(L.ffn.d_ff == inter for L in m.layers), mid
        # Alias layers read KV and do not write it. Donors still write.
        sm = _softmax(m, layers - 1, "decode", 64, 1)
        assert sm.kv_write == 0.0 and sm.kv_read > 0.0, mid
        sm0 = _softmax(m, 0, "prefill", 0, 16)
        assert sm0.kv_write > 0.0, mid
        ctx = 1000
        kvb = 2.0
        expect = 0.0
        for L in donor:
            c = L.core
            n = min(ctx, c.window) if c.window else ctx
            expect += n * c.n_kv * (c.qk_dim + c.v_dim) * kvb
        assert cache_new_bytes(m, ctx, 0) == expect, mid
        assert stage_storage(m, 0, m.n_layers, False, False, Shard(), ctx).kv_per_seq == expect, mid
        emb = embed_ops(m, Phase("decode", 1, 1, 8))
        assert any(o.name == "ple_lookup" and o.kv_read == 1 * m.ple_packed * 2 for o in emb), mid
        assert any(o.name == "ple_norm" for o in emb), mid
        assert any(o.name == "ple_mix" for o in build_rank_ops(m, 0, Phase("decode", 1, 1, 8), Shard())), mid


def test_image_and_audio_shapes():
    m = get_model("gemma-4-e2b-it")
    g = image_grid(m.vision, 1024, 1024)
    assert (g.px, g.llm_tokens) == ((768, 768), 256)
    ops = vision_ops(m.vision, g, 1)
    q = next(o for o in ops if o.name == "vis.0.q")
    assert q.m == 280 * 9 and q.k == 768 and q.n == 768
    proj = next(o for o in ops if o.name == "vis.proj")
    assert proj.m == 256 and (proj.k, proj.n) == (768, 1536)
    au = m.vision.audio
    assert au is not None and au.kind == "g4conf"
    assert au.tokens(1.0) == 25 and au.enc_positions(1.0) == 25
    assert au.tokens(30.0) == 750
    assert au.tokens(0.0) == 0
    aops = audio_ops(au, 1.0, 1)
    conv = next(o for o in aops if o.name == "aud.conv0")
    assert (conv.m, conv.k, conv.n) == (3200, 9, 128)
    aq = next(o for o in aops if o.name == "aud.0.q")
    assert aq.m == 25 and (aq.k, aq.n) == (1024, 1024)
    qk = next(o for o in aops if o.name == "aud.0.qk")
    assert (qk.m, qk.n, qk.k, qk.count) == (36, 24, 128, 8)
    rel = next(o for o in aops if o.name == "aud.0.rel")
    assert rel.m == 13 and rel.w_params == 1024 * 1024
    e4 = get_model("gemma-4-e4b-it")
    assert e4.vision.audio.tokens(1.0) == 25
    assert (e4.vision.post[0]) == (768, 2560)
    r = evaluate(Scenario(model="gemma-4-e2b-it", serving=Serving(phase="prefill", prompt=32, audio_s=1.0)))
    assert r.vision["image_tokens"] == 25 and r.vision["audio"]["tokens"] == 25
    quiet = evaluate(Scenario(model="gemma-4-e2b-it", serving=Serving(phase="prefill", prompt=32, audio_s=0)))
    assert quiet.vision is None or "audio" not in quiet.vision


def test_per_dim_scale_is_a_parameter():
    spec = importlib.util.spec_from_file_location("summarize_release", "scripts/summarize_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    name = "model.audio_tower.layers.0.self_attn.per_dim_scale"
    assert mod.SCALE_RE.search(name) and mod.NOT_SCALE_RE.search(name)
    inv = "model.language_model.layers.0.mlp.experts.down_proj_scale_inv"
    assert mod.SCALE_RE.search(inv) and not mod.NOT_SCALE_RE.search(inv)


def test_full_kv_heads_follow_k_eq_v():
    base = dict(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=8, intermediate_size=64, num_global_key_value_heads=1,
        hidden_size_per_layer_input=0, use_double_wide_mlp=False, enable_moe_block=False,
        num_kv_shared_layers=0,
    )
    layers, _, _, cov = _build_gemma4({**base, "attention_k_eq_v": False})
    full = layers[1]
    assert cov == "full" and full.core.n_kv == 2 and full.core.qk_dim == 16
    assert any(lin.name == "v" for lin in full.attn_linears)
    layers, _, _, _ = _build_gemma4({**base, "attention_k_eq_v": True})
    full = layers[1]
    assert full.core.n_kv == 1 and not any(lin.name == "v" for lin in full.attn_linears)
    assert layers[0].core.n_kv == 2 and any(lin.name == "v" for lin in layers[0].attn_linears)
    _raises(lambda: _build_gemma4({**base, "attention_k_eq_v": False, "num_kv_shared_layers": 4}))
    bad = {**base, "attention_k_eq_v": False, "num_kv_shared_layers": 1,
           "layer_types": ["sliding_attention"] * 3 + ["full_attention"]}
    _raises(lambda: _build_gemma4(bad))
