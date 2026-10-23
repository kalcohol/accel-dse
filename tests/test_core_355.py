"""0.355: MiniMax M3.1-Flash block-level max-scored sparse attention (MSA)."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def test_minimax_m3_1_flash_loads():
    """MiniMaxAI/MiniMax-M3.1-Flash loads with block-level MSA."""
    m = get_model("minimax-m3.1-flash")
    assert m is not None
    assert m.hf_id == "MiniMaxAI/MiniMax-M3.1-Flash"
    assert m.coverage == "full"
    assert len(m.layers) == 60


def test_block_sparse_attention_mechanism():
    """Block-level MSA: sparse_block_size 128, topk_blocks 16, max scoring."""
    m = get_model("minimax-m3.1-flash")
    dense_layers = 3
    for i, layer in enumerate(m.layers):
        c = layer.core
        assert c is not None
        assert c.kind == "gqa"
        assert c.n_q == 64 and c.n_kv == 4
        assert c.qk_dim == 128 and c.v_dim == 128
        if i >= dense_layers:
            assert c.qsa == 128
            assert c.topk == 16 * 128
            assert c.idx_heads == 4
            assert c.idx_dim == 128
            assert c.idx_score_type == "max"
            assert c.rope_n == 64


def test_moe_configuration():
    """MoE: 128 local experts / 4 per tok / 1 shared; first 3 layers dense."""
    m = get_model("minimax-m3.1-flash")
    dense_layers = 3
    for i, layer in enumerate(m.layers):
        ffn = layer.ffn
        if i < dense_layers:
            assert ffn.kind == "dense"
            assert ffn.d_ff == 12288
        else:
            assert ffn.kind == "moe"
            assert ffn.n_experts == 128
            assert ffn.top_k == 4
            assert ffn.d_expert == 3072
            assert ffn.n_shared == 1
            assert ffn.d_shared == 3072


def test_params_match_release():
    """Params ~426B match release headers."""
    m = get_model("minimax-m3.1-flash")
    p = m.params()
    assert abs(p - 426.375e9) / 426.375e9 < 0.02


def test_quantization_as_released():
    """mxfp8 backbone + NVFP4 MoE as released."""
    m = get_model("minimax-m3.1-flash")
    assert 8.0 <= m.fmt("attn").bits <= 8.5
    assert 8.0 <= m.fmt("mlp").bits <= 8.5
    assert m.fmt("expert").bits < 5.0
    assert 8.0 <= m.fmt("shared_expert").bits <= 8.5


def test_changelog_355():
    """CHANGELOG contains 0.355 entry with MiniMax M3.1-Flash."""
    assert "## [0.355.0]" in CL
    sec = CL.split("## [0.355.0]")[1].split("## [")[0]
    assert "MiniMax" in sec or "minimax" in sec.lower()
    assert "M3.1" in sec or "m3.1" in sec.lower()
    assert "块" in sec or "block" in sec.lower()
    assert "max" in sec.lower()
    assert "MSA" in sec or "稀疏" in sec
