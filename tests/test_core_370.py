"""0.370: Gemma-4 audio relative-key scores, and the 0.369 KV-alias / PLE invariants.

Gemma4AudioAttention projects a batch-1 sinusoid of context//2+1 positions, then every
padded query dots those keys (matrix_bd). Content QK stays the context matmul. The
projection does not scale with the clip batch; the query matmul does.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import _build_gemma4
from accel_dse.core.vision import audio_ops


def _sm(model, li, kind, ctx, q):
    ops = build_rank_ops(model, li, Phase(kind, 1, q, ctx), Shard())
    return next(o for o in ops if o.name == "softmax")


def test_audio_relative_keys_do_not_scale_the_projection():
    au = get_model("gemma-4-e2b-it").vision.audio
    assert au is not None and au.kind == "g4conf"
    one = {o.name: o for o in audio_ops(au, 1.0, 1)}
    relqk = one["aud.0.relqk"]
    # 1 s -> 25 tower steps, chunk 12 -> 36 padded queries, 8 heads, head 128, 13 relative keys.
    assert (relqk.m, relqk.k, relqk.n, relqk.count) == (36, 128, 13, 8)
    assert relqk.flops == 2.0 * 36 * 128 * 13 * 8
    assert one["aud.0.qk"].n == 24 and one["aud.0.rel"].m == 13
    assert one["aud.0.rel"].w_params == 1024 * 1024
    four = {o.name: o for o in audio_ops(au, 1.0, 4)}
    assert four["aud.0.rel"].m == 13 and four["aud.0.rel"].w_params == one["aud.0.rel"].w_params
    assert four["aud.0.relqk"].count == 32 and four["aud.0.qk"].count == 32
    assert four["aud.0.q"].m == one["aud.0.q"].m * 4
    assert four["aud.embed"].m == one["aud.embed"].m * 4
    # Every conformer layer has its own relative projection and query matmul.
    assert all(f"aud.{li}.relqk" in four for li in range(12))


def test_audio_tower_length_follows_the_128_sample_pad():
    au = get_model("gemma-4-e4b-it").vision.audio
    hit = None
    for samples in range(160, 8000):
        sec = samples / 16000.0
        tower, prompt = au._g4_audio_lens(sec)
        if tower > prompt > 0:
            hit = (sec, tower, prompt)
            break
    assert hit is not None
    sec, tower, prompt = hit
    assert au.tokens(sec) == prompt
    ops = {o.name: o for o in audio_ops(au, sec, 2)}
    assert ops["aud.embed"].m == tower * 2
    assert ops["aud.sub_proj"].m == tower * 2
    assert ops["aud.0.q"].m == tower * 2
    # LLM placeholders stay on the mask. 1.0 s is aligned, so both lengths are 25.
    assert au.tokens(1.0) == au.enc_positions(1.0) == 25


def test_alias_decode_reads_once_per_layer_and_does_not_write():
    m = get_model("gemma-4-e2b")
    ctx = 1000
    reads = []
    for i, L in enumerate(m.layers):
        sm = _sm(m, i, "decode", ctx, 1)
        if not L.kv_alias:
            assert sm.kv_write > 0.0
            continue
        c = L.core
        # Decode attends the donor cache after that layer has appended this token (ctx + q).
        # A sliding alias is still capped at sliding_window.
        n = min(ctx + 1, c.window) if c.window else (ctx + 1)
        expect = n * c.n_kv * (c.qk_dim + c.v_dim) * 2.0
        assert sm.kv_write == 0.0 and sm.kv_read == expect, (i, sm.kv_read, expect)
        reads.append((c.window, sm.kv_read))
        ops = build_rank_ops(m, i, Phase("decode", 1, 1, ctx), Shard())
        assert not any(o.name in ("attn.k", "attn.v") for o in ops)
    assert len(reads) == 20
    sliding = [r for w, r in reads if w]
    full = [r for w, r in reads if not w]
    assert sliding and full and max(sliding) < min(full)
    # Prefill of an alias sliding layer reads the cached prefix and still does not write.
    i = next(i for i, L in enumerate(m.layers) if L.kv_alias and L.core.window)
    sm = _sm(m, i, "prefill", 64, 16)
    c = m.layers[i].core
    assert sm.kv_write == 0.0
    assert sm.kv_read == 64 * c.n_kv * (c.qk_dim + c.v_dim) * 2.0


def test_double_wide_mlp_requires_a_shared_layer():
    base = dict(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=8, intermediate_size=64, num_global_key_value_heads=1,
        hidden_size_per_layer_input=0, use_double_wide_mlp=True, enable_moe_block=False,
        num_kv_shared_layers=0, attention_k_eq_v=False,
    )
    layers, _, notes, cov = _build_gemma4(base)
    assert cov == "full"
    assert all(L.ffn.d_ff == 64 and not L.kv_alias for L in layers)
    assert not any("双倍" in n for n in notes)
    m = get_model("gemma-4-e2b-it")
    assert sum(1 for L in m.layers if any(lin.name == "ple_proj" for lin in L.attn_linears)) == 1
    assert all(any(lin.name == "ple_gate" for lin in L.ffn_linears) for L in m.layers)


def test_granite_41_dense_gqa_matches_header():
    # ibm-granite/granite-4.1-{3b,8b,30b} and the compressed-tensors FP8 twins. Dense GQA, no sliding window.
    rows = {
        "granite-4.1-3b": (40, 2560, 40, 8, 64, 8192, 3402836480, False),
        "granite-4.1-8b": (40, 4096, 32, 8, 128, 12800, 8380551168, True),
        "granite-4.1-30b": (64, 4096, 32, 8, 128, 32768, 28865728512, False),
    }
    for mid, (n, h, heads, kv, hd, inter, params, tied_copy) in rows.items():
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == params, mid
        assert m.n_layers == n and m.hidden == h and m.tie_embeddings and m.coverage == "full"
        assert m.kv_fmt == "bf16" and m.act_fmt == "bf16"
        assert all(L.core.kind == "gqa" and L.core.n_q == heads and L.core.n_kv == kv and L.core.qk_dim == hd
                   and L.core.window is None and L.ffn.kind == "dense" and L.ffn.d_ff == inter for L in m.layers)
        assert ("tied lm_head" in " ".join(m.notes)) is tied_copy
        fp = get_model(mid + "-fp8")
        assert fp.param_check()["rel_err"] == 0.0 and fp.params() == params
        assert fp.act_fmt == "fp8" and fp.kv_fmt == "bf16"
        assert fp.fmt("attn").fmt == "fp8" and fp.fmt("mlp").fmt == "fp8"
        assert fp.fmt("embed").fmt == "bf16" and fp.fmt("lm_head").fmt == "bf16"
        assert 8.0 < fp.fmt("attn").bits < 8.02
        assert "tied lm_head" in " ".join(fp.notes)
