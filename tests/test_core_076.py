"""0.76: LLM / VLM per-op shapes pinned against the official configs / modeling code, the softmax-attention output
gate (Qwen3-Next yes, MiniMax no), parameter totals vs the released safetensors, and decode KV-read vs memplan bytes."""
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.memplan import cache_new_bytes, stage_storage

PH = Phase("decode", 1, 1, 4096)


def _op(mid, li, name):
    m = get_model(mid)
    for o in build_rank_ops(m, li, PH):
        if o.name == name:
            return (o.m, o.k, o.n, o.count) if o.kind == "attn" else (o.k, o.n, o.count)
    raise AssertionError((mid, li, name))


def test_output_gate_and_param_totals():
    assert _op("qwen3-next-80b-a3b", 3, "attn.q") == (2048, 16 * 256 * 2, 1)      # Qwen3NextAttention q_proj ×2
    assert _op("minimax-text-01", 7, "attn.q") == (6144, 64 * 128, 1)             # MiniMaxText01Attention: no gate
    assert _op("qwen3.5-397b-a17b", 3, "attn.q") == (4096, 32 * 256 * 2, 1)       # attn_output_gate: true
    for e in entries():
        m = get_model(e["id"])
        if m.domain == "llm" and m.release_params:
            assert abs(m.params() / m.release_params - 1) < 1e-3, (e["id"], m.params(), m.release_params)


def test_llm_shapes_match_official_configs():
    # DeepSeek-V3 MLA (absorbed decode) and V3.2 DSA indexer
    assert _op("deepseek-v3", 3, "attn.q_b") == (1536, 128 * 192, 1)
    assert _op("deepseek-v3", 3, "attn.kv_a") == (7168, 512 + 64, 1)
    assert _op("deepseek-v3", 3, "attn.kv_b.uk") == (128, 512, 128)
    assert _op("deepseek-v3", 3, "qk_latent") == (128, 576, 4097, 1)
    assert _op("deepseek-v3", 3, "expert.gate_up") == (7168, 2 * 2048, 8)
    assert _op("deepseek-v3", 0, "mlp.down") == (18432, 7168, 1)
    assert _op("deepseek-v3.2", 3, "attn.idx_q") == (1536, 64 * 128, 1)
    assert _op("deepseek-v3.2", 3, "qk_latent") == (128, 576, 2048, 1)          # top-2048
    assert _op("deepseek-v3.2", 3, "indexer_score") == (64, 128, 4097, 1)
    assert _op("kimi-k2", 1, "qk_latent") == (64, 576, 4097, 1)
    assert _op("kimi-k2", 0, "mlp.down") == (18432, 7168, 1)                     # first_k_dense_replace 1
    # GQA families
    assert _op("glm-4.5", 3, "attn.q") == (5120, 96 * 128, 1)
    assert _op("glm-4.5", 3, "expert.down") == (1536, 5120, 8)
    assert _op("qwen3-235b-a22b", 0, "qk") == (16, 128, 4097, 4)
    assert _op("mixtral-8x22b", 0, "expert.gate_up") == (6144, 2 * 16384, 2)
    # gpt-oss: alternating 128-token sliding window (layer 0 sliding), 64 q × 64, 8 kv
    assert _op("gpt-oss-120b", 0, "qk") == (8, 64, 128, 8)
    assert _op("gpt-oss-120b", 1, "qk") == (8, 64, 4097, 8)
    # linear attention: Qwen3-Next / Qwen3.5 Gated DeltaNet, MiniMax lightning (every 8th layer softmax)
    assert _op("qwen3-next-80b-a3b", 0, "attn.lin_qkv") == (2048, 2 * 16 * 128 + 32 * 128, 1)
    assert _op("qwen3-next-80b-a3b", 0, "attn.lin_ba") == (2048, 2 * 32, 1)
    assert _op("qwen3.5-397b-a17b", 0, "attn.lin_qkv") == (4096, 2 * 16 * 128 + 64 * 128, 1)
    assert _op("minimax-text-01", 0, "attn.lin_qkv") == (6144, 3 * 64 * 128, 1)
    assert _op("minimax-text-01", 0, "attn.lin_gate") == (6144, 64 * 128, 1)
    m = get_model("minimax-text-01")
    assert [i for i, L in enumerate(m.layers) if L.core.kind == "gqa"] == list(range(7, 80, 8))


def test_decode_kv_read_matches_memplan():
    C = 8192
    for e in entries():
        m = get_model(e["id"])
        if m.domain != "llm":
            continue
        ss = stage_storage(m, 0, m.n_layers, False, False, Shard(), C)
        stored = ss.kv_per_seq + ss.idx_per_seq + ss.state_per_seq
        assert abs(cache_new_bytes(m, C, 0) / stored - 1) < 1e-9, e["id"]
        ph = Phase("decode", 1, 1, C)
        read = sum(o.kv_read + o.state_rw / 2 for li in range(m.n_layers) for o in build_rank_ops(m, li, ph))
        sparse = any(L.core.topk or L.core.compress > 1 or L.core.kv_owner or L.core.idx_heads for L in m.layers)
        if not sparse:           # dense: the whole cache once (+ the new token's entry)
            assert stored * (1 - 1e-9) <= read <= stored * (C + 1) / C * (1 + 1e-9), (e["id"], read, stored)


def test_vision_towers_match_configs():
    v = get_model("qwen3.5-397b-a17b").vision
    assert (v.depth, v.hidden, v.inter, v.heads, v.patch, v.tpatch, v.merge) == (27, 1152, 4304, 16, 16, 2, 2)
    assert v.post[-1] == (4608, 4096) and len(v.post) == 2                          # deepstack_visual_indexes: []
    v = get_model("kimi-k2.5").vision
    assert (v.depth, v.hidden, v.inter, v.heads, v.patch, v.merge) == (27, 1152, 4304, 16, 14, 2)
    assert v.post[-1] == (4608, 7168)
