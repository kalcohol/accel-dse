"""0.61.1 — round-1 audit fixes: prefill of windowed / top-k / compressed attention charges the exact mean keys per
query (the first positions see fewer than the cap); the sparse-attention indexer is causal in prefill; a
sliding-window-only latent layer stores min(ctx, window) KV entries; the linear-attention conv state is read and
written every step (it was stored but not moved); serving sizes are bounded and reject booleans."""

from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import AttnCore
from accel_dse.core.scenario import SERVING_MAX, Serving


def _close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def _brute(core, n):
    return sum(min(p, core.ctx_eff(p)) for p in range(1, n + 1))


def test_keys_sum_closed_form_matches_brute_force():
    cores = [AttnCore("gqa", window=128), AttnCore("mla", topk=2048), AttnCore("mla", window=128, compress=4, topk=512),
             AttnCore("mla", window=128, compress=128), AttnCore("mla", window=128, compress=2),
             AttnCore("mla", window=64, compress=4, topk=1024)]
    for c in cores:
        for n in (0, 1, 2, 63, 64, 65, 127, 128, 129, 170, 171, 172, 511, 2047, 2048, 2049, 4096, 6000):
            assert c.keys_sum(n) == _brute(c, n), (c, n)
    # hand: window 128, positions 1..256 → Σ_{p≤128} p + 128·128 = 8256 + 16384
    assert AttnCore("gqa", window=128).keys_sum(256) == 8256 + 16384
    # mean over the prompt after a cached prefix: positions 101..300 of a 128-window → (Σ_{101..128} p + 172·128)/200
    assert _close(AttnCore("gqa", window=128).keys_mean(100, 200), (sum(range(101, 129)) + 172 * 128) / 200)


def _op(ops, name):
    return next(o for o in ops if o.name == name)


def test_dsa_prefill_mean_keys_and_causal_indexer():
    m = get_model("deepseek-v3.2")
    li = next(i for i, L in enumerate(m.layers) if L.core.topk)
    c = m.layers[li].core
    assert c.topk == 2048 and c.compress == 1 and c.idx_heads
    # prompt 1024 < top-k: every key exists → causal mean (1024 + 1)/2 = 512.5 keys per query (was 1024)
    ops = build_rank_ops(m, li, Phase("prefill", 1, 1024, 0))
    qk = _op(ops, "qk")
    assert qk.n == 1024 and _close(qk.n * qk.causal, 512.5)
    # prompt 4096: Σ min(p, 2048) / 4096 = (2048·2049/2 + 2048·2048)/4096 = 1536.25 (was 2048)
    ops = build_rank_ops(m, li, Phase("prefill", 1, 4096, 0))
    qk = _op(ops, "qk")
    assert qk.n == 2048 and _close(qk.n * qk.causal, 1536.25)
    # indexer: query p scores p keys → mean 4097/2 over n_keys = 4096 (was the full square, causal 1)
    ix = _op(ops, "indexer_score")
    assert ix.n == 4096 and _close(ix.causal, 4097 / 2 / 4096)
    # decode unchanged: cap keys, causal 1
    ops = build_rank_ops(m, li, Phase("decode", 1, 1, 8192))
    assert _op(ops, "qk_latent").n == 2048 and _op(ops, "qk_latent").causal == 1.0
    assert _op(ops, "indexer_score").causal == 1.0


def test_compressed_prefill_mean_keys():
    m = get_model("deepseek-v4-flash")
    li = next(i for i, L in enumerate(m.layers) if L.core.compress == 4)
    c = m.layers[li].core
    ops = build_rank_ops(m, li, Phase("prefill", 1, 4096, 0))
    qk = _op(ops, "qk")
    assert _close(qk.n * qk.causal, _brute(c, 4096) / 4096)
    assert qk.n * qk.causal < qk.n
    ix = _op(ops, "indexer_score")
    # query p scores ⌈p/4⌉ compressed keys: Σ = 4·1024·1025/2 → mean / 1024 keys
    assert ix.n == 1024 and _close(ix.causal, 4 * 1024 * 1025 / 2 / 4096 / 1024)


def test_swa_latent_layer_stores_window_only():
    m = get_model("deepseek-v4.1-flash")
    li = next(i for i, L in enumerate(m.layers) if L.core.kind == "mla" and L.core.compress == 1 and L.core.window)
    c = m.layers[li].core
    kvb = 1 if m.kv_fmt.startswith("fp8") else 2
    st = stage_storage(m, li, li + 1, False, False, Shard(), 4096)
    assert _close(st.kv_per_seq, c.window * (c.kv_lora + c.rope_dim) * {"bf16": 2, "fp8": 1}.get(m.kv_fmt, kvb))
    # below the window: every token
    st = stage_storage(m, li, li + 1, False, False, Shard(), 100)
    assert _close(st.kv_per_seq, 100 * (c.kv_lora + c.rope_dim) * {"bf16": 2, "fp8": 1}.get(m.kv_fmt, kvb))


def test_linear_attention_conv_state_traffic():
    m = get_model("qwen3-next-80b-a3b")
    li = next(i for i, L in enumerate(m.layers) if L.core.kind == "linear")
    c = m.layers[li].core
    assert (c.n_state_heads, c.state_dk, c.state_dv, c.conv_channels, c.conv_kernel) == (32, 128, 128, 8192, 4)
    ops = build_rank_ops(m, li, Phase("decode", 3, 1, 4096))
    st = _op(ops, "state_update")
    # (32·128·128 recurrent + 8192·3 conv) fp32, read + write, 3 sequences
    assert st.state_rw == 2 * (32 * 128 * 128 + 8192 * 3) * 4 * 3
    ops = build_rank_ops(m, li, Phase("decode", 3, 1, 4096), Shard(tp=2, ep=2))
    assert _op(ops, "state_update").state_rw == 2 * (16 * 128 * 128 + 4096 * 3) * 4 * 3
    # storage carries the same bytes
    s = stage_storage(m, li, li + 1, False, False, Shard(), 4096)
    assert s.state_per_seq == (32 * 128 * 128 + 8192 * 3) * 4


def test_serving_bounds():
    for bad in ({"batch": True}, {"batch": SERVING_MAX + 1}, {"ctx": 2 ** 40}, {"prompt": 0}, {"out_len": False},
                {"microbatches": -1}, {"ctx": 1.5}):
        try:
            Serving(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    Serving(batch=SERVING_MAX, ctx=SERVING_MAX, prompt=SERVING_MAX)


def test_compressed_decode_reads_whole_entries():
    # DeepSeek-V4 CSA / HCA decode: the attended cache entries (⌈ctx/c⌉ + window, ≤ top-k + window) are whole latent
    # entries (kv_lora + rope); 0.61.0 divided by c twice (c = 4: 4× too few bytes, c = 128: 128×)
    m = get_model("deepseek-v4-flash")
    kvb = {"bf16": 2, "fp8": 1}[m.kv_fmt]
    for c, entries in ((4, 512 + 128), (128, 8193 // 128 + 1 + 128)):
        li = next(i for i, L in enumerate(m.layers) if L.core.compress == c)
        co = m.layers[li].core
        ops = build_rank_ops(m, li, Phase("decode", 2, 1, 8192))
        assert _op(ops, "qk_latent").n == entries
        assert _op(ops, "softmax").kv_read == 2 * entries * (co.kv_lora + co.rope_dim) * kvb
        # the full cache of the layer is what memplan stores — the decode read never exceeds it
        st = stage_storage(m, li, li + 1, False, False, Shard(), 8193)
        assert _op(ops, "softmax").kv_read <= 2 * st.kv_per_seq + 1e-9


def test_window_prefix_read_and_reduction_bytes():
    from accel_dse.core.ir import red_bytes
    m = get_model("gpt-oss-20b")
    li = next(i for i, L in enumerate(m.layers) if L.core.window)
    c = m.layers[li].core
    kvb = {"bf16": 2, "fp8": 1}[m.kv_fmt]
    # prefill of 512 new tokens after a 4096-token cached prefix: a 128-window layer reads the last 128 prefix tokens
    ops = build_rank_ops(m, li, Phase("prefill", 1, 512, 4096))
    assert _op(ops, "softmax").kv_read == 128 * c.n_kv * (c.qk_dim + c.v_dim) * kvb
    full = next(i for i, L in enumerate(m.layers) if not L.core.window)
    ops = build_rank_ops(m, full, Phase("prefill", 1, 512, 4096))
    assert _op(ops, "softmax").kv_read == 4096 * c.n_kv * (c.qk_dim + c.v_dim) * kvb
    # fp8-activation release: TP all-reduce / MoE combine / ETP all-reduce carry bf16, the dispatch stays fp8
    d = get_model("deepseek-v3")
    assert d.act_fmt == "fp8" and red_bytes(d) == 2.0 and red_bytes(get_model("qwen3-8b")) == 2.0
    li = next(i for i, L in enumerate(d.layers) if L.ffn.kind == "moe")
    ops = build_rank_ops(d, li, Phase("decode", 64, 1, 4096), Shard(tp=2, dp=4, ep=4, etp=2))
    t = 64 // 4
    assert _op(ops, "attn_allreduce").comm_bytes == t * d.hidden * 2
    disp, comb = _op(ops, "moe_dispatch").comm_bytes, _op(ops, "moe_combine").comm_bytes
    assert abs(comb / disp - 2.0 / (disp / (64 * 8 / 4 * d.hidden))) < 1e-9     # combine at 2 B / element
    assert _op(ops, "expert_allreduce").comm_bytes == 64 * 8 / 4 * d.hidden * 2
