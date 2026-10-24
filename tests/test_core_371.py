"""0.371: Olmo-3 / EXAONE-4 q/k norms, and the 0.370 audio + Granite invariants.

Gemma4AudioAttention adds content scores to relative scores and softmaxes once over the
context axis. relative_k_proj stays on the batch-1 sinusoid. Granite-4.1 FP8 channel
scales are two bytes per output channel. Olmo-3 norms the full Q/K projection; EXAONE-4
norms head_dim and drops RoPE on hybrid full layers.
"""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import _build_std, load_release
from accel_dse.core.vision import audio_ops


def test_audio_softmax_is_once_after_the_score_add():
    au = get_model("gemma-4-e2b-it").vision.audio
    one = audio_ops(au, 1.0, 1)
    four = audio_ops(au, 1.0, 4)
    # One softmax per conformer layer. Its length is the context axis (24), after
    # content and relative scores are added. Not a second softmax over the 13 rel keys.
    for clips, ops in ((1, one), (4, four)):
        names = [o.name for o in ops]
        assert names.count("aud.0.softmax") == 1
        assert all(names.count(f"aud.{li}.softmax") == 1 for li in range(12))
        sm = next(o for o in ops if o.name == "aud.0.softmax")
        relqk = next(o for o in ops if o.name == "aud.0.relqk")
        rel = next(o for o in ops if o.name == "aud.0.rel")
        qk = next(o for o in ops if o.name == "aud.0.qk")
        assert rel.m == 13 and relqk.n == 13 and qk.n == 24
        assert sm.vec == 8 * clips * relqk.m * qk.n * 5.0
        assert relqk.count == 8 * clips and relqk.m == 36
    assert next(o for o in four if o.name == "aud.0.rel").m == 13
    assert next(o for o in four if o.name == "aud.0.rel").w_params == next(
        o for o in one if o.name == "aud.0.rel").w_params


def test_granite_fp8_scales_are_two_bytes_per_output_channel():
    # compressed-tensors per-channel weight scales. Header bytes minus logical params
    # equal 2 bytes times the Linear out-features (bf16 scales, not fp32).
    for mid in ("granite-4.1-3b-fp8", "granite-4.1-8b-fp8", "granite-4.1-30b-fp8"):
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0
        assert m.act_fmt == "fp8" and m.kv_fmt == "bf16"
        assert m.fmt("embed").fmt == "bf16" and m.fmt("lm_head").fmt == "bf16"
        # Model.fmt has no "norm" key (_ROLE_KEYS). An unknown role falls back to attn,
        # so fmt("norm") is the fp8 GEMM width. The checkpoint role is bf16, and the
        # memory plan bills LLM misc at a fixed 2 bytes, which matches that header.
        # Layer misc is the two per-layer RMSNorms; the final norm is the existing hidden term.
        rel = load_release(m.hf_id)
        norm = rel["roles"]["norm"]["bf16"]
        assert set(rel["roles"]["norm"]) == {"bf16"}
        assert norm["bytes"] == 2 * norm["params"]
        assert sum(L.misc_params for L in m.layers) + m.hidden == norm["params"]
        assert "norm" not in dict(m.formats)
        for role, lins in (
            ("attn", [lin for L in m.layers for lin in L.attn_linears]),
            ("mlp", [lin for L in m.layers for lin in L.ffn_linears]),
        ):
            bucket = rel["roles"][role]["fp8"]
            assert bucket["bytes"] - bucket["params"] == 2 * sum(lin.n for lin in lins), mid
            assert bucket["params"] == sum(lin.params for lin in lins)


def test_olmo3_qk_norm_is_the_full_projection():
    # 7B MHA: both norms are hidden (the 2*L*hidden the 0.370 index gap measured).
    # 32B GQA: k_norm is KV*head_dim, not hidden.
    rows = {
        "olmo-3-7b-think": (32, 4096, 32, 32, 128, 11008, 24, 8, 7298011136, 16384),
        "olmo-3-7b-instruct": (32, 4096, 32, 32, 128, 11008, 24, 8, 7298011136, 16384),
        "olmo-3-1025-7b": (32, 4096, 32, 32, 128, 11008, 24, 8, 7298011136, 16384),
        "olmo-3-32b-think": (64, 5120, 40, 8, 128, 27648, 48, 16, 32233522176, 16384),
        "olmo-3-1125-32b": (64, 5120, 40, 8, 128, 27648, 48, 16, 32233522176, 16384),
        "olmo-3.1-32b-instruct": (64, 5120, 40, 8, 128, 27648, 48, 16, 32233522176, 16384),
    }
    for mid, (n, h, heads, kv, hd, inter, n_sw, n_full, params, misc) in rows.items():
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == params, mid
        assert m.coverage == "full" and m.kv_fmt == "bf16" and m.act_fmt == "bf16"
        assert not m.tie_embeddings
        sw = [L for L in m.layers if L.core.window == 4096]
        full = [L for L in m.layers if L.core.window is None]
        assert len(sw) == n_sw and len(full) == n_full and m.n_layers == n
        assert all(L.core.kind == "gqa" and L.core.n_q == heads and L.core.n_kv == kv
                   and L.core.qk_dim == hd and not L.core.no_rope and L.ffn.d_ff == inter
                   and L.misc_params == misc for L in m.layers)
        assert m.hidden == h
        # RoPE still runs on a full layer (Olmo applies it to both kinds).
        i = next(i for i, L in enumerate(m.layers) if L.core.window is None)
        ops = build_rank_ops(m, i, Phase("decode", 1, 1, 64), Shard())
        assert any(o.name == "rope" for o in ops)


def test_exaone4_head_dim_norm_and_global_nope():
    rows = {
        "exaone-4.0-32b": (64, 5120, 40, 8, 128, 27392, 48, 16, 32003216384, 10496, False),
        "exaone-4.0.1-32b": (64, 5120, 40, 8, 128, 27392, 48, 16, 32003216384, 10496, False),
        "exaone-4.0-1.2b": (30, 2048, 32, 8, 64, 4096, 0, 30, 1279391488, 4224, True),
    }
    for mid, (n, h, heads, kv, hd, inter, n_sw, n_full, params, misc, tied) in rows.items():
        m = get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.params() == params and m.hidden == h, mid
        assert m.coverage == "full" and m.tie_embeddings is tied and m.n_layers == n
        sw = [L for L in m.layers if L.core.window == 4096]
        full = [L for L in m.layers if L.core.window is None]
        assert len(sw) == n_sw and len(full) == n_full
        assert all(L.core.n_q == heads and L.core.n_kv == kv and L.core.qk_dim == hd
                   and L.ffn.d_ff == inter and L.misc_params == misc for L in m.layers)
        assert all(not L.core.no_rope for L in sw)
        if n_sw:
            assert all(L.core.no_rope for L in full)
            i_sw = next(i for i, L in enumerate(m.layers) if not L.core.no_rope)
            i_full = next(i for i, L in enumerate(m.layers) if L.core.no_rope)
            sw_ops = build_rank_ops(m, i_sw, Phase("decode", 1, 1, 64), Shard())
            full_ops = build_rank_ops(m, i_full, Phase("decode", 1, 1, 64), Shard())
            assert any(o.name == "rope" for o in sw_ops)
            assert all(o.name != "rope" for o in full_ops)
            # NoPE does not shrink the KV read. Both layers store the same width.
            assert next(o for o in sw_ops if o.name == "softmax").kv_read > 0
            assert next(o for o in full_ops if o.name == "softmax").kv_read > 0
        else:
            assert all(not L.core.no_rope for L in m.layers)
            ops = build_rank_ops(m, 0, Phase("decode", 1, 1, 64), Shard())
            assert any(o.name == "rope" for o in ops)


def test_builder_does_not_invent_qk_norm_for_other_types():
    base = dict(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_hidden_layers=2, intermediate_size=64, vocab_size=10,
        layer_types=["sliding_attention", "full_attention"], sliding_window=16,
    )
    plain, _, _, _ = _build_std({**base, "model_type": "llama"})
    assert all(L.misc_params == 2 * 32 and not L.core.no_rope for L in plain)
    olmo, _, notes, cov = _build_std({**base, "model_type": "olmo3"})
    assert cov == "full"
    # Q is 4*8, K is 2*8, plus the two hidden norms.
    assert all(L.misc_params == 64 + 32 + 16 and not L.core.no_rope for L in olmo)
    assert olmo[0].core.window == 16 and olmo[1].core.window is None
    assert any("Olmo-3" in n for n in notes)
    ex, _, notes, _ = _build_std({**base, "model_type": "exaone4"})
    assert all(L.misc_params == 64 + 2 * 8 for L in ex)
    assert not ex[0].core.no_rope and ex[1].core.no_rope
    assert any("不做 RoPE" in n for n in notes)
    # sliding_window null: every layer, including full_attention, still has RoPE.
    dense, _, notes, _ = _build_std({**base, "model_type": "exaone4", "sliding_window": None,
                                     "layer_types": ["full_attention", "full_attention"]})
    assert all(not L.core.no_rope and L.core.window is None for L in dense)
    assert any("每一层都做 RoPE" in n for n in notes)
